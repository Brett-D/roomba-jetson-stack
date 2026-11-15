// C ABI over the C++-only libraries the person detector needs (built by build.rs, used by
// src/trt.rs and src/ocv.rs):
//  - TensorRT: deserialize an engine, one execution context, CUDA managed memory for every I/O
//    tensor (the Jetson's iGPU shares DRAM, so the CPU reads and writes them directly), enqueueV3
//    on a stream + synchronize. Same as roomba_perception/person_detector.py's TrtYolo + _Cuda, except
//    that the host waits for the GPU by blocking rather than spinning and the engine's kernels are
//    launched as one CUDA graph (both only save CPU).
//  - OpenCV: StereoBM, the drawing calls (rectangle, polylines, Hershey text), resize and JPEG
//    encoding. Everything else (letterbox, gray, blending, NMS, percentiles) is Rust.
// No exception crosses the C boundary: failures return null / non-zero with a message.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

void set_err(char* err, size_t len, const std::string& msg) {
  if (err != nullptr && len > 0) {
    std::snprintf(err, len, "%s", msg.c_str());
  }
}

// Prints warnings and errors like the Python TensorRT logger (trt.Logger(trt.Logger.WARNING)).
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity > Severity::kWARNING) {
      return;
    }
    const char* tag = severity == Severity::kINTERNAL_ERROR ? "F" : severity == Severity::kERROR ? "E" : "W";
    std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_r(&now, &tm);
    char stamp[32];
    std::strftime(stamp, sizeof stamp, "%m/%d/%Y-%H:%M:%S", &tm);
    std::fprintf(stderr, "[%s] [TRT] [%s] %s\n", stamp, tag, msg);
  }
};

Logger& trt_logger() {
  static Logger logger;
  return logger;
}

struct Tensor {
  std::string name;
  bool input = false;
  std::vector<int64_t> dims;
  int dtype = 0;
  void* ptr = nullptr;
  size_t nbytes = 0;
};

struct Engine {
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx;
  cudaStream_t stream = nullptr;
  std::vector<Tensor> tensors;
  // The engine's kernels recorded once as a CUDA graph: one launch instead of ~100 per frame.
  cudaGraphExec_t graph = nullptr;
  bool graph_tried = false;

  ~Engine() {
    if (graph != nullptr) {
      cudaGraphExecDestroy(graph);
    }
    ctx.reset();
    engine.reset();
    runtime.reset();
    for (auto& t : tensors) {
      if (t.ptr != nullptr) {
        cudaFree(t.ptr);
      }
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

size_t dtype_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kINT64:
      return 8;
    default:
      return 1;
  }
}

bool cuda_ok(cudaError_t rc, const char* what, char* err, size_t errlen) {
  if (rc == cudaSuccess) {
    return true;
  }
  set_err(err, errlen, std::string(what) + " failed (cudaError " + std::to_string(int(rc)) + ")");
  return false;
}

cv::Mat view(const uint8_t* data, int rows, int cols, size_t step, int type) {
  return cv::Mat(rows, cols, type, const_cast<uint8_t*>(data), step);
}

}  // namespace

extern "C" {

void* rr_trt_load(const char* path, char* err, size_t errlen) {
  try {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      set_err(err, errlen, std::string("[Errno 2] No such file or directory: '") + path + "'");
      return nullptr;
    }
    std::vector<char> plan((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // Wait for inference by blocking instead of spinning a core for the ~10 ms the GPU takes (CUDA's
    // default with one context). Only possible before the context exists; then it stays as it is.
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    cudaGetLastError();
    auto e = std::make_unique<Engine>();
    e->runtime.reset(nvinfer1::createInferRuntime(trt_logger()));
    if (!e->runtime) {
      set_err(err, errlen, "Failed to create the TensorRT runtime");
      return nullptr;
    }
    e->engine.reset(e->runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!e->engine) {
      set_err(err, errlen, std::string("Failed to load TensorRT engine ") + path);
      return nullptr;
    }
    e->ctx.reset(e->engine->createExecutionContext());
    if (!e->ctx) {
      set_err(err, errlen, "Failed to create a TensorRT execution context");
      return nullptr;
    }
    if (!cuda_ok(cudaStreamCreate(&e->stream), "cudaStreamCreate", err, errlen)) {
      return nullptr;
    }
    for (int i = 0; i < e->engine->getNbIOTensors(); i++) {
      Tensor t;
      t.name = e->engine->getIOTensorName(i);
      nvinfer1::Dims d = e->engine->getTensorShape(t.name.c_str());
      size_t count = 1;
      for (int k = 0; k < d.nbDims; k++) {
        if (d.d[k] < 0) {
          set_err(err, errlen, "dynamic tensor shapes are not supported (" + t.name + ")");
          return nullptr;
        }
        t.dims.push_back(d.d[k]);
        count *= size_t(d.d[k]);
      }
      nvinfer1::DataType dt = e->engine->getTensorDataType(t.name.c_str());
      t.dtype = int(dt);
      t.input = e->engine->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
      t.nbytes = count * dtype_size(dt);
      if (!cuda_ok(cudaMallocManaged(&t.ptr, t.nbytes, cudaMemAttachGlobal), "cudaMallocManaged", err, errlen)) {
        return nullptr;
      }
      e->ctx->setTensorAddress(t.name.c_str(), t.ptr);
      e->tensors.push_back(std::move(t));
    }
    return e.release();
  } catch (const std::exception& ex) {
    set_err(err, errlen, ex.what());
    return nullptr;
  }
}

void rr_trt_free(void* engine) { delete static_cast<Engine*>(engine); }

int rr_trt_num_tensors(const void* engine) { return int(static_cast<const Engine*>(engine)->tensors.size()); }

// Tensor i: name, input flag, nvinfer1::DataType, dims (up to max_dims), managed pointer, size.
int rr_trt_tensor(const void* engine, int i, const char** name, int* is_input, int* dtype, int64_t* dims,
                  int max_dims, void** ptr, size_t* nbytes) {
  const auto& t = static_cast<const Engine*>(engine)->tensors.at(size_t(i));
  *name = t.name.c_str();
  *is_input = t.input ? 1 : 0;
  *dtype = t.dtype;
  int n = int(t.dims.size());
  for (int k = 0; k < n && k < max_dims; k++) {
    dims[k] = t.dims[size_t(k)];
  }
  *ptr = t.ptr;
  *nbytes = t.nbytes;
  return n;
}

// Records the engine's work on its stream as a CUDA graph (after a first normal run, so TensorRT
// has done its lazy setup). The same kernels run on the same buffers, so results are unchanged;
// if capturing fails the engine keeps using enqueueV3. PERSON_DETECTOR_CUDA_GRAPH=0 turns it off.
static void capture_graph(Engine* e) {
  e->graph_tried = true;
  const char* env = std::getenv("PERSON_DETECTOR_CUDA_GRAPH");
  if (env != nullptr && std::strcmp(env, "0") == 0) {
    return;
  }
  cudaGraph_t g = nullptr;
  if (cudaStreamBeginCapture(e->stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    cudaGetLastError();
    return;
  }
  bool queued = e->ctx->enqueueV3(e->stream);
  if (cudaStreamEndCapture(e->stream, &g) == cudaSuccess && queued && g != nullptr) {
    if (cudaGraphInstantiate(&e->graph, g, 0) != cudaSuccess) {
      e->graph = nullptr;
    }
  }
  if (g != nullptr) {
    cudaGraphDestroy(g);
  }
  cudaGetLastError();
}

// Runs the engine on the managed buffers and waits for it (the CPU may then read the outputs).
int rr_trt_infer(void* engine, char* err, size_t errlen) {
  auto* e = static_cast<Engine*>(engine);
  if (e->graph != nullptr) {
    if (!cuda_ok(cudaGraphLaunch(e->graph, e->stream), "cudaGraphLaunch", err, errlen)) {
      return -1;
    }
  } else if (!e->ctx->enqueueV3(e->stream)) {
    set_err(err, errlen, "TensorRT execution failed");
    return -1;
  }
  if (!cuda_ok(cudaStreamSynchronize(e->stream), "cudaStreamSynchronize", err, errlen)) {
    return -1;
  }
  if (!e->graph_tried) {
    capture_graph(e);
  }
  return 0;
}


void* rr_stereo_bm_create(int num_disparities, int block_size) {
  try {
    return new cv::Ptr<cv::StereoBM>(cv::StereoBM::create(num_disparities, block_size));
  } catch (const std::exception&) {
    return nullptr;
  }
}

void rr_stereo_bm_free(void* bm) { delete static_cast<cv::Ptr<cv::StereoBM>*>(bm); }

// Disparity (CV_16S, 16 x pixels) of two mono8 images of rows x cols into disp (rows * cols).
int rr_stereo_bm_compute(void* bm, const uint8_t* left, size_t left_step, const uint8_t* right, size_t right_step,
                         int rows, int cols, int16_t* disp, char* err, size_t errlen) {
  try {
    cv::Mat out(rows, cols, CV_16SC1, disp);
    (*static_cast<cv::Ptr<cv::StereoBM>*>(bm))
        ->compute(view(left, rows, cols, left_step, CV_8UC1), view(right, rows, cols, right_step, CV_8UC1), out);
    if (out.data != reinterpret_cast<uint8_t*>(disp)) {
      set_err(err, errlen, "StereoBM reallocated its output");
      return -1;
    }
    return 0;
  } catch (const std::exception& ex) {
    set_err(err, errlen, ex.what());
    return -1;
  }
}

// cv::resize(INTER_LINEAR) of an 8-bit image with `channels` channels.
int rr_resize_linear(const uint8_t* src, int src_rows, int src_cols, size_t src_step, int channels, uint8_t* dst,
                     int dst_rows, int dst_cols, size_t dst_step) {
  try {
    cv::Mat out = view(dst, dst_rows, dst_cols, dst_step, CV_8UC(channels));
    cv::resize(view(src, src_rows, src_cols, src_step, CV_8UC(channels)), out, cv::Size(dst_cols, dst_rows), 0, 0,
               cv::INTER_LINEAR);
    return out.data == dst ? 0 : -1;
  } catch (const std::exception&) {
    return -1;
  }
}

int rr_rectangle(uint8_t* img, int rows, int cols, size_t step, int x1, int y1, int x2, int y2, int b, int g, int r,
                 int thickness, int line_type) {
  try {
    cv::Mat m = view(img, rows, cols, step, CV_8UC3);
    cv::rectangle(m, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(b, g, r), thickness, line_type);
    return 0;
  } catch (const std::exception&) {
    return -1;
  }
}

// Open polyline through n points (x0, y0, x1, y1, ...).
int rr_polyline(uint8_t* img, int rows, int cols, size_t step, const int32_t* xy, int n, int b, int g, int r,
                int thickness, int line_type) {
  try {
    cv::Mat m = view(img, rows, cols, step, CV_8UC3);
    std::vector<cv::Point> pts;
    pts.reserve(size_t(n));
    for (int i = 0; i < n; i++) {
      pts.emplace_back(xy[2 * i], xy[2 * i + 1]);
    }
    std::vector<std::vector<cv::Point>> polys{pts};
    cv::polylines(m, polys, false, cv::Scalar(b, g, r), thickness, line_type);
    return 0;
  } catch (const std::exception&) {
    return -1;
  }
}

int rr_text_size(const char* text, int font, double scale, int thickness, int* width, int* height, int* baseline) {
  try {
    cv::Size s = cv::getTextSize(text, font, scale, thickness, baseline);
    *width = s.width;
    *height = s.height;
    return 0;
  } catch (const std::exception&) {
    return -1;
  }
}

int rr_put_text(uint8_t* img, int rows, int cols, size_t step, const char* text, int x, int y, int font, double scale,
                int b, int g, int r, int thickness, int line_type) {
  try {
    cv::Mat m = view(img, rows, cols, step, CV_8UC3);
    cv::putText(m, text, cv::Point(x, y), font, scale, cv::Scalar(b, g, r), thickness, line_type);
    return 0;
  } catch (const std::exception&) {
    return -1;
  }
}

// cv::imencode(".jpg", img, {IMWRITE_JPEG_QUALITY, quality}); *out is malloc'ed (free with rr_free).
int rr_imencode_jpeg(const uint8_t* img, int rows, int cols, size_t step, int quality, uint8_t** out, size_t* len) {
  try {
    std::vector<uchar> buf;
    if (!cv::imencode(".jpg", view(img, rows, cols, step, CV_8UC3), buf, {cv::IMWRITE_JPEG_QUALITY, quality})) {
      return -1;
    }
    *out = static_cast<uint8_t*>(std::malloc(buf.empty() ? 1 : buf.size()));
    if (*out == nullptr) {
      return -1;
    }
    std::memcpy(*out, buf.data(), buf.size());
    *len = buf.size();
    return 0;
  } catch (const std::exception&) {
    return -1;
  }
}

void rr_free(void* p) { std::free(p); }

}  // extern "C"
