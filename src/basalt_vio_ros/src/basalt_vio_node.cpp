// ROS 2 node running Basalt stereo-inertial odometry (Monado fork of Basalt).
//
// In:  rectified mono8 stereo images (left/image, right/image, identical stamps) + raw IMU (imu).
// Out: the camera body pose in its own odometry frame on the topics cuVSLAM used,
//      /visual_slam/tracking/odometry and /visual_slam/status, plus the /visual_slam/reset
//      service, so vslam_watchdog, RTAB-Map and the web app work unchanged.
//
// Geometry comes from a Basalt calibration JSON (pinhole intrinsics of the rectified images,
// T_imu_cam per camera, IMU noise); the body frame (base_frame) is placed via the static TF
// base_frame <- IMU frame published by the camera driver.

// Eigen first: basalt/calibration.hpp includes an Eigen-internal header directly.
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <basalt/calibration/calibration.hpp>
#include <basalt/optical_flow/optical_flow.h>
#include <basalt/serialization/headers_serialization.h>
#include <basalt/utils/vio_config.h>
#include <basalt/vi_estimator/vio_estimator.h>

#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <isaac_ros_visual_slam_interfaces/msg/visual_slam_status.hpp>
#include <isaac_ros_visual_slam_interfaces/srv/reset.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

using isaac_ros_visual_slam_interfaces::msg::VisualSlamStatus;
using isaac_ros_visual_slam_interfaces::srv::Reset;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::Imu;
using namespace std::chrono_literals;

namespace {
constexpr uint8_t VO_OK = 1;
constexpr uint8_t VO_FAILED = 2;

int64_t to_ns(const builtin_interfaces::msg::Time& t) { return int64_t(t.sec) * 1000000000LL + t.nanosec; }

builtin_interfaces::msg::Time to_msg(int64_t ns) {
  builtin_interfaces::msg::Time t;
  t.sec = static_cast<int32_t>(ns / 1000000000LL);
  t.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
  return t;
}

basalt::ImageData to_basalt(const Image& m) {
  basalt::ImageData d;
  d.img = std::make_shared<basalt::ManagedImage<uint16_t>>(m.width, m.height);
  for (uint32_t y = 0; y < m.height; ++y) {
    const uint8_t* row = &m.data[size_t(y) * m.step];
    uint16_t* out = d.img->ptr + size_t(y) * m.width;
    for (uint32_t x = 0; x < m.width; ++x) out[x] = uint16_t(row[x]) << 8;  // Basalt works on 16 bit
  }
  return d;
}
}  // namespace

class BasaltVioNode : public rclcpp::Node {
 public:
  BasaltVioNode() : Node("basalt_vio") {
    const auto calib_path = declare_parameter<std::string>("calib_path", "");
    const auto config_path = declare_parameter<std::string>("config_path", "");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "vslam_odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "camera_link");
    imu_frame_ = declare_parameter<std::string>("imu_frame", "camera_imu_frame");
    lost_timeout_ns_ = int64_t(declare_parameter<double>("lost_timeout_s", 0.5) * 1e9);

    std::ifstream is(calib_path, std::ios::binary);
    if (!is.is_open()) throw std::runtime_error("cannot open Basalt calibration '" + calib_path + "'");
    {
      cereal::JSONInputArchive archive(is);
      archive(calib_);
    }
    if (!config_path.empty()) config_.load(config_path);
    RCLCPP_INFO(get_logger(), "Basalt VIO: %zu cameras, baseline %.4f m, config %s", calib_.intrinsics.size(),
                (calib_.T_i_c[1].translation() - calib_.T_i_c[0].translation()).norm(),
                config_path.empty() ? "(defaults)" : config_path.c_str());

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/visual_slam/tracking/odometry", 20);
    status_pub_ = create_publisher<VisualSlamStatus>("/visual_slam/status", 20);

    start_estimator();

    // Reliable like the driver's publishers: losing frames or IMU samples hurts tracking.
    imu_sub_ = create_subscription<Imu>("imu", rclcpp::QoS(400).reliable(),
                                        [this](Imu::ConstSharedPtr m) { on_imu(*m); });
    left_sub_ = create_subscription<Image>("left/image", rclcpp::QoS(10).reliable(),
                                           [this](Image::ConstSharedPtr m) { on_image(m, 0); });
    right_sub_ = create_subscription<Image>("right/image", rclcpp::QoS(10).reliable(),
                                            [this](Image::ConstSharedPtr m) { on_image(m, 1); });
    reset_srv_ = create_service<Reset>(
        "/visual_slam/reset", [this](const std::shared_ptr<Reset::Request>, std::shared_ptr<Reset::Response> res) {
          if (vio_) vio_->scheduleResetState();
          RCLCPP_WARN(get_logger(), "Reset requested: Basalt state reset scheduled");
          res->success = true;
        });
    timer_ = create_wall_timer(100ms, [this] { check_lost(); });
  }

  ~BasaltVioNode() override { stop_estimator(); }

 private:
  void start_estimator() {
    opt_flow_ = basalt::OpticalFlowFactory::getOpticalFlow(config_, calib_);
    opt_flow_->start();
    vio_ = basalt::VioEstimatorFactory::getVioEstimator(config_, calib_, basalt::constants::g, true, true);
    vio_->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    opt_flow_->output_queue = &vio_->vision_data_queue;
    vio_->out_state_queue = &out_state_queue_;
    vio_->opt_flow_depth_guess_queue = &opt_flow_->input_depth_queue;
    vio_->opt_flow_state_queue = &opt_flow_->input_state_queue;
    vio_->opt_flow_lm_bundle_queue = &opt_flow_->input_lm_bundle_queue;
    output_thread_ = std::thread([this] { output_loop(); });
  }

  void stop_estimator() {
    if (!vio_) return;
    opt_flow_->input_img_queue.push(nullptr);
    vio_->imu_data_queue.push(nullptr);
    opt_flow_->input_imu_queue.push(nullptr);
    vio_->maybe_join();
    vio_->drain_input_queues();
    out_state_queue_.push(nullptr);
    if (output_thread_.joinable()) output_thread_.join();
    vio_.reset();
    opt_flow_.reset();
  }

  void on_imu(const Imu& m) {
    auto d = std::make_shared<basalt::ImuData<double>>();
    d->t_ns = to_ns(m.header.stamp);
    if (d->t_ns <= last_imu_ns_) return;
    last_imu_ns_ = d->t_ns;
    d->accel = Eigen::Vector3d(m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z);
    d->gyro = Eigen::Vector3d(m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z);
    // Both consumers need every recent sample. A full queue (e.g. before the first frame
    // reaches the backend) must not block or starve the other queue: drop its oldest sample.
    push_latest(vio_->imu_data_queue, d);
    push_latest(opt_flow_->input_imu_queue, d);
  }

  void push_latest(tbb::concurrent_bounded_queue<basalt::ImuData<double>::Ptr>& q,
                   const basalt::ImuData<double>::Ptr& d) {
    while (!q.try_push(d)) {
      basalt::ImuData<double>::Ptr old;
      if (q.try_pop(old)) ++imu_dropped_;
    }
  }

  void on_image(const Image::ConstSharedPtr& m, int cam) {
    if (m->encoding != "mono8") {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "Expected mono8 images, got %s",
                            m->encoding.c_str());
      return;
    }
    const int64_t t = to_ns(m->header.stamp);
    auto& slot = pending_[t];
    slot[cam] = m;
    if (!slot[0] || !slot[1]) {
      while (pending_.size() > 10) pending_.erase(pending_.begin());  // unmatched leftovers
      return;
    }
    auto in = std::make_shared<basalt::OpticalFlowInput>(2);
    in->t_ns = t;
    in->img_data[0] = to_basalt(*slot[0]);
    in->img_data[1] = to_basalt(*slot[1]);
    pending_.erase(pending_.begin(), std::next(pending_.find(t)));
    last_frame_ns_ = t;
    if (first_frame_ns_ == 0) first_frame_ns_ = t;
    ++frames_in_;
    if (!opt_flow_->input_img_queue.try_push(in)) {
      ++frames_dropped_;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Basalt is behind: dropped %lu frames so far",
                           static_cast<unsigned long>(frames_dropped_));
    }
  }

  bool body_from_imu(Sophus::SE3d& T_i_b) {
    if (have_T_i_b_) {
      T_i_b = T_i_b_;
      return true;
    }
    try {
      // base_frame <- IMU frame (the driver's static TF).
      const auto tf = tf_buffer_->lookupTransform(base_frame_, imu_frame_, tf2::TimePointZero);
      const auto& r = tf.transform.rotation;
      const auto& p = tf.transform.translation;
      Sophus::SE3d T_b_i(Eigen::Quaterniond(r.w, r.x, r.y, r.z).normalized(), Eigen::Vector3d(p.x, p.y, p.z));
      T_i_b_ = T_i_b = T_b_i.inverse();
      have_T_i_b_ = true;
      RCLCPP_INFO(get_logger(), "Body frame %s found; publishing odometry", base_frame_.c_str());
      return true;
    } catch (const tf2::TransformException& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for TF %s <- %s: %s", base_frame_.c_str(),
                           imu_frame_.c_str(), e.what());
      return false;
    }
  }

  void output_loop() {
    while (true) {
      basalt::PoseVelBiasState<double>::Ptr s;
      out_state_queue_.pop(s);
      if (!s) break;
      last_output_ns_ = s->t_ns;
      ++poses_out_;
      Sophus::SE3d T_i_b;
      if (!body_from_imu(T_i_b)) continue;
      const Sophus::SE3d T_w_b = s->T_w_i * T_i_b;
      const Eigen::Vector3d p = T_w_b.translation();
      const Eigen::Quaterniond q = T_w_b.unit_quaternion();
      const Eigen::Vector3d v_b = T_w_b.so3().inverse() * s->vel_w_i;

      nav_msgs::msg::Odometry o;
      o.header.stamp = to_msg(s->t_ns);
      o.header.frame_id = odom_frame_;
      o.child_frame_id = base_frame_;
      o.pose.pose.position.x = p.x();
      o.pose.pose.position.y = p.y();
      o.pose.pose.position.z = p.z();
      o.pose.pose.orientation.x = q.x();
      o.pose.pose.orientation.y = q.y();
      o.pose.pose.orientation.z = q.z();
      o.pose.pose.orientation.w = q.w();
      for (int i = 0; i < 6; ++i) o.pose.covariance[i * 7] = 1e-4;
      o.twist.twist.linear.x = v_b.x();
      o.twist.twist.linear.y = v_b.y();
      o.twist.twist.linear.z = v_b.z();
      odom_pub_->publish(o);

      VisualSlamStatus st;
      st.header = o.header;
      st.vo_state = VO_OK;
      status_pub_->publish(st);
    }
  }

  // Frames arriving but no poses coming out: report tracking failure (vslam_watchdog recovers).
  void check_lost() {
    if (++ticks_ % 300 == 0) {
      RCLCPP_INFO(get_logger(), "Basalt: %lu frames in, %lu poses out, %lu frames / %lu IMU samples dropped",
                  static_cast<unsigned long>(frames_in_), static_cast<unsigned long>(poses_out_.load()),
                  static_cast<unsigned long>(frames_dropped_), static_cast<unsigned long>(imu_dropped_));
    }
    const int64_t frame = last_frame_ns_.load();
    const int64_t out = last_output_ns_.load();
    // Before the first pose Basalt is still initializing; only call it lost if that takes > 3 s.
    if (frame == 0 || (out == 0 ? frame - first_frame_ns_ < 3000000000LL : frame - out < lost_timeout_ns_)) return;
    VisualSlamStatus st;
    st.header.stamp = to_msg(frame);
    st.header.frame_id = odom_frame_;
    st.vo_state = VO_FAILED;
    status_pub_->publish(st);
  }

  basalt::Calibration<double> calib_;
  basalt::VioConfig config_;
  basalt::OpticalFlowBase::Ptr opt_flow_;
  basalt::VioEstimatorBase::Ptr vio_;
  tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr> out_state_queue_;
  std::thread output_thread_;

  std::string odom_frame_, base_frame_, imu_frame_;
  int64_t lost_timeout_ns_;
  std::map<int64_t, std::array<Image::ConstSharedPtr, 2>> pending_;
  int64_t last_imu_ns_ = 0;
  std::atomic<int64_t> last_frame_ns_{0}, last_output_ns_{0}, first_frame_ns_{0};
  uint64_t frames_dropped_ = 0, imu_dropped_ = 0, frames_in_ = 0, ticks_ = 0;
  std::atomic<uint64_t> poses_out_{0};
  bool have_T_i_b_ = false;
  Sophus::SE3d T_i_b_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<VisualSlamStatus>::SharedPtr status_pub_;
  rclcpp::Subscription<Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<Image>::SharedPtr left_sub_, right_sub_;
  rclcpp::Service<Reset>::SharedPtr reset_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<BasaltVioNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
