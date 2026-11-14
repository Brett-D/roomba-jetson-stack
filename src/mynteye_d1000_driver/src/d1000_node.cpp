// ROS 2 driver for the MYNT EYE D1000 stereo depth camera.
//
// Publishes (relative names; launch inside a namespace such as "camera"). Images are
// rectified 1280x720 scaled by output_scale (default 0.5 -> 640x360):
//   left/image_rect_color   sensor_msgs/Image (bgr8, rectified)
//   left/image_rect_mono    sensor_msgs/Image (mono8, rectified; for cuVSLAM)
//   left/camera_info        sensor_msgs/CameraInfo (rectified model)
//   right/image_rect_color  sensor_msgs/Image (bgr8, rectified)
//   right/image_rect_mono   sensor_msgs/Image (mono8, rectified)
//   right/camera_info       sensor_msgs/CameraInfo (P[3] = -fx * baseline)
//   depth/image_rect_raw    sensor_msgs/Image (16UC1, mm, registered to rectified left)
//   depth/camera_info       sensor_msgs/CameraInfo
//   imu/data_raw            sensor_msgs/Imu (m/s^2, rad/s, no orientation)
//   points                  sensor_msgs/PointCloud2 (optional, decimated)
//
// All stamps come from the camera's hardware clock (10 us ticks), mapped to ROS
// time with a single min-latency offset so images and IMU share one timebase.
// In IR-depth-only (interleaved) mode each depth frame is captured ~33 ms after its
// color frame; by default it is stamped with that color frame's time so consumers can
// pair them with exact-time sync (depth_stamp_from_color).
//
// cv_bridge is intentionally not used: JetPack ships OpenCV 4.8 (used by the SDK)
// while ROS Jazzy's cv_bridge links Ubuntu's 4.6; loading both in one process is fragile.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include <mynteyed/camera.h>
#include <mynteyed/utils.h>

MYNTEYE_USE_NAMESPACE

namespace mynteye_d1000_driver
{

namespace
{
constexpr double kGravity = 9.80665;
constexpr int64_t kTickNs = 10000;              // device clock tick = 10 us
constexpr uint64_t kWrapTicks = 4294900000ULL;  // wrap point used by the official wrapper

class Unwrapper
{
public:
  uint64_t operator()(uint64_t raw)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    if (init_ && raw < last_ && (last_ - raw) > kWrapTicks / 2) {
      ++epochs_;
    }
    last_ = raw;
    init_ = true;
    return epochs_ * kWrapTicks + raw;
  }

private:
  std::mutex mtx_;
  uint64_t last_{0};
  uint64_t epochs_{0};
  bool init_{false};
};

// Tracks offset = host_time - device_time as a running minimum (lowest transport
// latency seen), leaking upward slowly so it follows crystal drift.
class DeviceClock
{
public:
  explicit DeviceClock(double leak_rate) : leak_rate_(leak_rate) {}

  void Observe(int64_t dev_ns, int64_t host_ns)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    const int64_t cand = host_ns - dev_ns;
    if (!init_ || cand < offset_ || cand > offset_ + 500000000LL) {
      offset_ = cand;
      init_ = true;
    } else {
      offset_ += static_cast<int64_t>(leak_rate_ * static_cast<double>(host_ns - last_host_ns_));
    }
    last_host_ns_ = host_ns;
  }

  bool Initialized() const
  {
    std::lock_guard<std::mutex> lk(mtx_);
    return init_;
  }

  int64_t ToHostNs(int64_t dev_ns) const
  {
    std::lock_guard<std::mutex> lk(mtx_);
    return dev_ns + offset_;
  }

private:
  mutable std::mutex mtx_;
  double leak_rate_;
  int64_t offset_{0};
  int64_t last_host_ns_{0};
  bool init_{false};
};

std::array<double, 4> QuatFromRot(const double m[3][3])
{
  double x, y, z, w;
  const double tr = m[0][0] + m[1][1] + m[2][2];
  if (tr > 0.0) {
    const double s = std::sqrt(tr + 1.0) * 2.0;
    w = 0.25 * s;
    x = (m[2][1] - m[1][2]) / s;
    y = (m[0][2] - m[2][0]) / s;
    z = (m[1][0] - m[0][1]) / s;
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
    w = (m[2][1] - m[1][2]) / s;
    x = 0.25 * s;
    y = (m[0][1] + m[1][0]) / s;
    z = (m[0][2] + m[2][0]) / s;
  } else if (m[1][1] > m[2][2]) {
    const double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
    w = (m[0][2] - m[2][0]) / s;
    x = (m[0][1] + m[1][0]) / s;
    y = 0.25 * s;
    z = (m[1][2] + m[2][1]) / s;
  } else {
    const double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
    w = (m[1][0] - m[0][1]) / s;
    x = (m[0][2] + m[2][0]) / s;
    y = (m[1][2] + m[2][1]) / s;
    z = 0.25 * s;
  }
  const double n = std::sqrt(x * x + y * y + z * z + w * w);
  return {x / n, y / n, z / n, w / n};
}

void FillImage(
  sensor_msgs::msg::Image & msg, const cv::Mat & m, const std::string & encoding)
{
  msg.height = static_cast<uint32_t>(m.rows);
  msg.width = static_cast<uint32_t>(m.cols);
  msg.encoding = encoding;
  msg.is_bigendian = false;
  msg.step = static_cast<uint32_t>(m.cols * m.elemSize());
  msg.data.resize(static_cast<size_t>(msg.step) * m.rows);
  if (m.isContinuous()) {
    std::memcpy(msg.data.data(), m.data, msg.data.size());
  } else {
    for (int r = 0; r < m.rows; ++r) {
      std::memcpy(msg.data.data() + r * msg.step, m.ptr(r), msg.step);
    }
  }
}
}  // namespace

class D1000Node : public rclcpp::Node
{
public:
  D1000Node()
  : Node("mynteye_d1000")
  {
    dev_index_ = declare_parameter<int>("dev_index", 0);
    framerate_ = declare_parameter<int>("framerate", 30);
    ir_intensity_ = declare_parameter<int>("ir_intensity", 4);
    ir_depth_only_ = declare_parameter<bool>("ir_depth_only", true);
    auto_exposure_ = declare_parameter<bool>("auto_exposure", true);
    publish_right_ = declare_parameter<bool>("publish_right", true);
    publish_points_ = declare_parameter<bool>("publish_points", false);
    points_decimation_ = std::max<int>(1, declare_parameter<int>("points_decimation", 4));
    min_depth_mm_ = static_cast<int>(declare_parameter<double>("min_depth_m", 0.30) * 1000.0);
    max_depth_mm_ = static_cast<int>(declare_parameter<double>("max_depth_m", 7.0) * 1000.0);
    depth_stamp_from_color_ = declare_parameter<bool>("depth_stamp_from_color", true);
    output_scale_ = std::clamp(declare_parameter<double>("output_scale", 0.5), 0.1, 1.0);
    enable_imu_ = declare_parameter<bool>("enable_imu", true);
    imu_process_ = declare_parameter<bool>("imu_process", true);
    gyro_noise_std_ = declare_parameter<double>("gyro_noise_std", 0.005);
    accel_noise_std_ = declare_parameter<double>("accel_noise_std", 0.05);
    baseline_override_ = declare_parameter<double>("baseline_m", 0.0);
    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    base_frame_ = declare_parameter<std::string>("base_frame", "camera_link");
    left_frame_ = declare_parameter<std::string>("left_frame", "camera_left_optical_frame");
    right_frame_ = declare_parameter<std::string>("right_frame", "camera_right_optical_frame");
    imu_frame_ = declare_parameter<std::string>("imu_frame", "camera_imu_frame");
    clock_ = std::make_unique<DeviceClock>(declare_parameter<double>("clock_leak_rate", 2e-4));

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(5));
    pub_left_ = create_publisher<sensor_msgs::msg::Image>("left/image_rect_color", qos);
    pub_left_mono_ = create_publisher<sensor_msgs::msg::Image>("left/image_rect_mono", qos);
    pub_left_info_ = create_publisher<sensor_msgs::msg::CameraInfo>("left/camera_info", qos);
    if (publish_right_) {
      pub_right_ = create_publisher<sensor_msgs::msg::Image>("right/image_rect_color", qos);
      pub_right_mono_ = create_publisher<sensor_msgs::msg::Image>("right/image_rect_mono", qos);
      pub_right_info_ = create_publisher<sensor_msgs::msg::CameraInfo>("right/camera_info", qos);
    }
    pub_depth_ = create_publisher<sensor_msgs::msg::Image>("depth/image_rect_raw", qos);
    pub_depth_info_ = create_publisher<sensor_msgs::msg::CameraInfo>("depth/camera_info", qos);
    if (publish_points_) {
      pub_points_ = create_publisher<sensor_msgs::msg::PointCloud2>("points", qos);
    }
    if (enable_imu_) {
      pub_imu_ = create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", rclcpp::QoS(rclcpp::KeepLast(200)));
    }
    tf_static_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    OpenDevice();

    stats_timer_ = create_wall_timer(std::chrono::seconds(30), [this]() {
        RCLCPP_INFO(get_logger(), "rates (30s avg): left %.1f Hz, right %.1f Hz, depth %.1f Hz, imu %.1f Hz",
          n_left_.exchange(0) / 30.0, n_right_.exchange(0) / 30.0,
          n_depth_.exchange(0) / 30.0, n_imu_.exchange(0) / 30.0);
      });
  }

  ~D1000Node() override
  {
    ready_ = false;
    if (cam_.IsOpened()) {
      cam_.Close();
    }
  }

private:
  void OpenDevice()
  {
    std::vector<DeviceInfo> devs;
    cam_.GetDeviceInfos(&devs);
    if (devs.empty()) {
      throw std::runtime_error("No MYNT EYE device found (lsusb should list 1e4e:0120; check udev rules)");
    }
    if (dev_index_ < 0 || dev_index_ >= static_cast<int>(devs.size())) {
      throw std::runtime_error("dev_index out of range");
    }

    OpenParams params(devs[dev_index_].index);
    params.framerate = std::min(framerate_, 30);  // 2560x720 supports up to 30 fps
    params.stream_mode = StreamMode::STREAM_2560x720;  // factory calibration exists for 720p modes only
    params.color_mode = ColorMode::COLOR_RECTIFIED;
    params.dev_mode = DeviceMode::DEVICE_ALL;
    params.ir_intensity = static_cast<std::uint8_t>(std::clamp(ir_intensity_, 0, 10));
    params.ir_depth_only = ir_depth_only_;
    params.state_ae = auto_exposure_;

    cam_.EnableImageInfo(true);
    if (enable_imu_) {
      cam_.EnableMotionDatas(0);
      cam_.EnableProcessMode(imu_process_ ? ProcessMode::PROC_IMU_ALL : ProcessMode::PROC_NONE);
      cam_.SetMotionCallback([this](const MotionData & d) {OnMotion(d);});
    }
    cam_.SetStreamCallback(ImageType::IMAGE_LEFT_COLOR, [this](const StreamData & d) {OnColor(d, true);});
    if (publish_right_) {
      cam_.SetStreamCallback(ImageType::IMAGE_RIGHT_COLOR, [this](const StreamData & d) {OnColor(d, false);});
    }
    cam_.SetStreamCallback(ImageType::IMAGE_DEPTH, [this](const StreamData & d) {OnDepth(d);});

    if (cam_.Open(params) != ErrorCode::SUCCESS || !cam_.IsOpened()) {
      throw std::runtime_error("Failed to open MYNT EYE D1000");
    }

    bool ok = false;
    const auto in = cam_.GetStreamIntrinsics(params.stream_mode, &ok);
    if (!ok) {
      throw std::runtime_error("Factory calibration unavailable (is /dev/hidraw for 0483:5720 readable?)");
    }
    const auto ex = cam_.GetStreamExtrinsics(params.stream_mode, &ok);
    double baseline = ok ? std::abs(ex.translation[0]) / 1000.0 : 0.12;
    if (baseline < 0.05 || baseline > 0.3) {
      RCLCPP_WARN(get_logger(), "Implausible baseline %.4f m from extrinsics; using 0.120 m", baseline);
      baseline = 0.12;
    }
    if (baseline_override_ > 0.0) {
      baseline = baseline_override_;
    }
    baseline_ = baseline;

    left_info_ = MakeRectifiedInfo(in.left, left_frame_, 0.0);
    right_info_ = MakeRectifiedInfo(in.left, right_frame_, baseline_);
    RCLCPP_INFO(get_logger(), "Rectified %ux%u fx=%.2f fy=%.2f cx=%.2f cy=%.2f baseline=%.4f m",
      left_info_.width, left_info_.height, left_info_.p[0], left_info_.p[5], left_info_.p[2],
      left_info_.p[6], baseline_);

    if (publish_tf_) {
      PublishStaticTf();
    }

    ready_ = true;
    RCLCPP_INFO(get_logger(), "MYNT EYE D1000 streaming (ir_depth_only=%s, imu=%s)",
      ir_depth_only_ ? "true" : "false", enable_imu_ ? "true" : "false");
  }

  sensor_msgs::msg::CameraInfo MakeRectifiedInfo(
    const CameraIntrinsics & in, const std::string & frame, double baseline) const
  {
    // Rectified pinhole model, scaled to the published resolution (pixel-center convention).
    const double s = output_scale_;
    sensor_msgs::msg::CameraInfo info;
    info.header.frame_id = frame;
    info.width = static_cast<uint32_t>(std::lround(in.width * s));
    info.height = static_cast<uint32_t>(std::lround(in.height * s));
    const double fx = in.p[0] * s, fy = in.p[5] * s;
    const double cx = (in.p[2] + 0.5) * s - 0.5, cy = (in.p[6] + 0.5) * s - 0.5;
    info.distortion_model = "plumb_bob";
    info.d.assign(5, 0.0);
    info.k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
    info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    info.p = {fx, 0.0, cx, -fx * baseline, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0};
    return info;
  }

  void PublishStaticTf()
  {
    std::vector<geometry_msgs::msg::TransformStamped> tfs;
    const auto stamp = now();
    auto make = [&](const std::string & parent, const std::string & child,
        double x, double y, double z, const std::array<double, 4> & q) {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = parent;
        t.child_frame_id = child;
        t.transform.translation.x = x;
        t.transform.translation.y = y;
        t.transform.translation.z = z;
        t.transform.rotation.x = q[0];
        t.transform.rotation.y = q[1];
        t.transform.rotation.z = q[2];
        t.transform.rotation.w = q[3];
        return t;
      };
    // camera_link: REP-103 body frame (x forward, y left, z up) centered between the lenses.
    const std::array<double, 4> q_optical{-0.5, 0.5, -0.5, 0.5};
    tfs.push_back(make(base_frame_, left_frame_, 0.0, baseline_ / 2.0, 0.0, q_optical));
    tfs.push_back(make(base_frame_, right_frame_, 0.0, -baseline_ / 2.0, 0.0, q_optical));

    if (enable_imu_) {
      bool ok = false;
      const auto me = cam_.GetMotionExtrinsics(&ok);
      if (ok) {
        // SDK convention (verified by IMU position on the board): p_imu = R * p_left + t (mm).
        // IMU pose in the left optical frame: rotation R^T, origin -R^T * t.
        double rt[3][3];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            rt[i][j] = me.rotation[j][i];
          }
        }
        double o[3];
        for (int i = 0; i < 3; ++i) {
          o[i] = -(rt[i][0] * me.translation[0] + rt[i][1] * me.translation[1] +
            rt[i][2] * me.translation[2]) / 1000.0;
        }
        tfs.push_back(make(left_frame_, imu_frame_, o[0], o[1], o[2], QuatFromRot(rt)));
        RCLCPP_INFO(get_logger(), "IMU in %s: t=(%.4f, %.4f, %.4f) m",
          left_frame_.c_str(), o[0], o[1], o[2]);
      } else {
        RCLCPP_WARN(get_logger(), "IMU extrinsics unavailable; assuming IMU axes = flipped optical");
        const double flip[3][3] = {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
        tfs.push_back(make(left_frame_, imu_frame_, baseline_ / 2.0, 0.0, 0.0, QuatFromRot(flip)));
      }
    }
    tf_static_->sendTransform(tfs);
  }

  rclcpp::Time StampFromDevice(uint64_t dev_ticks)
  {
    const int64_t dev_ns = static_cast<int64_t>(dev_ticks) * kTickNs;
    if (!enable_imu_ || !clock_->Initialized()) {
      clock_->Observe(dev_ns, now().nanoseconds());
    }
    return rclcpp::Time(clock_->ToHostNs(dev_ns), get_clock()->get_clock_type());
  }

  void OnColor(const StreamData & d, bool left)
  {
    if (!ready_ || !d.img) {
      return;
    }
    (left ? n_left_ : n_right_)++;
    const rclcpp::Time stamp = d.img_info ?
      StampFromDevice((left ? unwrap_left_ : unwrap_right_)(d.img_info->timestamp)) : now();

    // The SDK rectifies color before delivering it, so the paired depth frame often
    // arrives first and waits in pending_ until its color frame shows up.
    PendingDepth matched;
    if (left && d.img_info) {
      std::lock_guard<std::mutex> lk(pair_mtx_);
      recent_color_[recent_idx_++ % recent_color_.size()] = {d.img_info->frame_id, stamp, true};
      if (pending_.valid && pending_.frame_id == static_cast<uint16_t>(d.img_info->frame_id + 1)) {
        matched = std::move(pending_);
        pending_.valid = false;
      }
    }
    if (matched.valid) {
      PublishDepth(matched.depth, stamp);
    }

    auto & pub = left ? pub_left_ : pub_right_;
    auto & pub_info = left ? pub_left_info_ : pub_right_info_;
    auto & pub_mono = left ? pub_left_mono_ : pub_right_mono_;
    const bool want_color = pub->get_subscription_count() > 0;
    const bool want_mono = pub_mono->get_subscription_count() > 0;
    if (!want_color && !want_mono && pub_info->get_subscription_count() == 0) {
      return;
    }

    cv::Mat bgr = d.img->To(ImageFormat::COLOR_BGR)->ToMat();
    if (output_scale_ != 1.0) {
      cv::Mat small;
      cv::resize(bgr, small, cv::Size(), output_scale_, output_scale_, cv::INTER_AREA);
      bgr = small;
    }
    const std::string & frame = left ? left_frame_ : right_frame_;
    if (want_color) {
      auto msg = std::make_unique<sensor_msgs::msg::Image>();
      FillImage(*msg, bgr, "bgr8");
      msg->header.stamp = stamp;
      msg->header.frame_id = frame;
      pub->publish(std::move(msg));
    }
    if (want_mono) {
      cv::Mat gray;
      cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
      auto msg = std::make_unique<sensor_msgs::msg::Image>();
      FillImage(*msg, gray, "mono8");
      msg->header.stamp = stamp;
      msg->header.frame_id = frame;
      pub_mono->publish(std::move(msg));
    }

    auto info = left ? left_info_ : right_info_;
    info.header.stamp = stamp;
    pub_info->publish(info);
  }

  bool WantDepth() const
  {
    return pub_depth_->get_subscription_count() > 0 || pub_depth_info_->get_subscription_count() > 0 ||
           (pub_points_ && pub_points_->get_subscription_count() > 0);
  }

  void OnDepth(const StreamData & d)
  {
    if (!ready_ || !d.img) {
      return;
    }
    n_depth_++;
    if (!WantDepth()) {
      return;
    }
    const rclcpp::Time own_stamp =
      d.img_info ? StampFromDevice(unwrap_depth_(d.img_info->timestamp)) : now();

    cv::Mat raw = d.img->To(ImageFormat::DEPTH_RAW)->ToMat();
    cv::Mat depth;
    if (output_scale_ != 1.0) {
      cv::resize(raw, depth, cv::Size(), output_scale_, output_scale_, cv::INTER_NEAREST);
    } else {
      depth = raw.clone();
    }
    for (int r = 0; r < depth.rows; ++r) {
      auto * p = depth.ptr<uint16_t>(r);
      for (int c = 0; c < depth.cols; ++c) {
        if (p[c] < min_depth_mm_ || p[c] > max_depth_mm_) {
          p[c] = 0;
        }
      }
    }

    if (!(ir_depth_only_ && depth_stamp_from_color_ && d.img_info)) {
      PublishDepth(depth, own_stamp);
      return;
    }

    // Interleaved mode: depth frame N+1 belongs to color frame N; share its stamp so
    // downstream exact-time sync pairs them 1:1.
    const uint16_t color_id = static_cast<uint16_t>(d.img_info->frame_id - 1);
    bool found = false;
    rclcpp::Time color_stamp;
    PendingDepth stale;
    {
      std::lock_guard<std::mutex> lk(pair_mtx_);
      for (const auto & c : recent_color_) {
        if (c.valid && c.frame_id == color_id) {
          found = true;
          color_stamp = c.stamp;
          break;
        }
      }
      if (!found) {
        stale = std::move(pending_);
        pending_ = PendingDepth{d.img_info->frame_id, depth, own_stamp, true};
      }
    }
    if (stale.valid) {
      PublishDepth(stale.depth, stale.own_stamp);  // its color frame never arrived
    }
    if (found) {
      PublishDepth(depth, color_stamp);
    }
  }

  void PublishDepth(const cv::Mat & depth, const rclcpp::Time & stamp)
  {
    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    FillImage(*msg, depth, "16UC1");
    msg->header.stamp = stamp;
    msg->header.frame_id = left_frame_;
    pub_depth_->publish(std::move(msg));

    auto info = left_info_;
    info.header.stamp = stamp;
    pub_depth_info_->publish(info);

    if (pub_points_ && pub_points_->get_subscription_count() > 0) {
      PublishPoints(depth, stamp);
    }
  }

  void PublishPoints(const cv::Mat & depth, const rclcpp::Time & stamp)
  {
    const double fx = left_info_.p[0], fy = left_info_.p[5], cx = left_info_.p[2], cy = left_info_.p[6];
    const int step = points_decimation_;
    std::vector<float> xyz;
    xyz.reserve(static_cast<size_t>(depth.rows / step) * (depth.cols / step) * 3);
    for (int r = 0; r < depth.rows; r += step) {
      const auto * p = depth.ptr<uint16_t>(r);
      for (int c = 0; c < depth.cols; c += step) {
        if (p[c] == 0) {
          continue;
        }
        const float z = p[c] / 1000.0f;
        xyz.push_back(static_cast<float>((c - cx) * z / fx));
        xyz.push_back(static_cast<float>((r - cy) * z / fy));
        xyz.push_back(z);
      }
    }
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = stamp;
    cloud.header.frame_id = left_frame_;
    cloud.height = 1;
    cloud.is_dense = true;
    sensor_msgs::PointCloud2Modifier mod(cloud);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(xyz.size() / 3);
    std::memcpy(cloud.data.data(), xyz.data(), xyz.size() * sizeof(float));
    pub_points_->publish(cloud);
  }

  void OnMotion(const MotionData & d)
  {
    if (!ready_ || !d.imu) {
      return;
    }
    const ImuData & imu = *d.imu;
    const int64_t dev_ns = static_cast<int64_t>(unwrap_imu_(imu.timestamp)) * kTickNs;
    clock_->Observe(dev_ns, now().nanoseconds());

    if (imu.flag == MYNTEYE_IMU_ACCEL) {
      std::lock_guard<std::mutex> lk(accel_mtx_);
      last_accel_ = imu;
      have_accel_ = true;
      return;
    }
    ImuData accel;
    if (imu.flag == MYNTEYE_IMU_GYRO) {
      std::lock_guard<std::mutex> lk(accel_mtx_);
      if (!have_accel_) {
        return;
      }
      accel = last_accel_;
    } else if (imu.flag == MYNTEYE_IMU_ACCEL_GYRO_CALIB) {
      accel = imu;
    } else {
      return;
    }

    n_imu_++;
    sensor_msgs::msg::Imu msg;
    msg.header.stamp = rclcpp::Time(clock_->ToHostNs(dev_ns), get_clock()->get_clock_type());
    msg.header.frame_id = imu_frame_;
    msg.orientation_covariance[0] = -1.0;
    msg.angular_velocity.x = imu.gyro[0] * M_PI / 180.0;
    msg.angular_velocity.y = imu.gyro[1] * M_PI / 180.0;
    msg.angular_velocity.z = imu.gyro[2] * M_PI / 180.0;
    msg.linear_acceleration.x = accel.accel[0] * kGravity;
    msg.linear_acceleration.y = accel.accel[1] * kGravity;
    msg.linear_acceleration.z = accel.accel[2] * kGravity;
    const double gv = gyro_noise_std_ * gyro_noise_std_;
    const double av = accel_noise_std_ * accel_noise_std_;
    msg.angular_velocity_covariance = {gv, 0, 0, 0, gv, 0, 0, 0, gv};
    msg.linear_acceleration_covariance = {av, 0, 0, 0, av, 0, 0, 0, av};
    pub_imu_->publish(msg);
  }

  Camera cam_;
  std::atomic<bool> ready_{false};
  std::unique_ptr<DeviceClock> clock_;
  Unwrapper unwrap_left_, unwrap_right_, unwrap_depth_, unwrap_imu_;

  std::mutex accel_mtx_;
  ImuData last_accel_;
  bool have_accel_{false};

  std::mutex pair_mtx_;
  struct ColorStamp
  {
    uint16_t frame_id{0};
    rclcpp::Time stamp;
    bool valid{false};
  };
  struct PendingDepth
  {
    uint16_t frame_id{0};
    cv::Mat depth;
    rclcpp::Time own_stamp;
    bool valid{false};
  };
  std::array<ColorStamp, 8> recent_color_;
  size_t recent_idx_{0};
  PendingDepth pending_;

  std::atomic<int> n_left_{0}, n_right_{0}, n_depth_{0}, n_imu_{0};

  int dev_index_, framerate_, ir_intensity_, points_decimation_, min_depth_mm_, max_depth_mm_;
  bool ir_depth_only_, auto_exposure_, publish_right_, publish_points_, enable_imu_, imu_process_,
    publish_tf_, depth_stamp_from_color_;
  double gyro_noise_std_, accel_noise_std_, baseline_override_, baseline_{0.12}, output_scale_{0.5};
  std::string base_frame_, left_frame_, right_frame_, imu_frame_;
  sensor_msgs::msg::CameraInfo left_info_, right_info_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_left_, pub_right_, pub_depth_,
    pub_left_mono_, pub_right_mono_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_left_info_, pub_right_info_,
    pub_depth_info_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_points_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_;
  rclcpp::TimerBase::SharedPtr stats_timer_;
};

}  // namespace mynteye_d1000_driver

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int rc = 0;
  try {
    rclcpp::spin(std::make_shared<mynteye_d1000_driver::D1000Node>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("mynteye_d1000"), "%s", e.what());
    rc = 1;
  }
  rclcpp::shutdown();
  return rc;
}
