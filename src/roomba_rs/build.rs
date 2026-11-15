//! Builds the native code: src/web/codecs.c, the web server's image encoders (the system
//! libjpeg-turbo and libpng driven the way OpenCV's imgcodecs does), and native/person_shim.cpp, the C ABI over TensorRT and OpenCV used by the person detector
//! (src/trt.rs, src/ocv.rs). Needs the TensorRT dev headers (libnvinfer-dev), the CUDA toolkit
//! (cuda_runtime_api.h, libcudart) and OpenCV 4 dev files (libopencv-dev / nvidia-opencv-dev), all
//! part of JetPack. Override the locations with CUDA_HOME, TENSORRT_INCLUDE and OPENCV_INCLUDE.
use std::path::{Path, PathBuf};

fn first_existing(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| p.exists()).cloned()
}

fn main() {
    println!("cargo:rerun-if-changed=src/web/codecs.c");
    cc::Build::new().file("src/web/codecs.c").opt_level(2).compile("roomba_web_codecs");
    println!("cargo:rustc-link-lib=jpeg");
    println!("cargo:rustc-link-lib=png16");
    println!("cargo:rustc-link-lib=z");

    println!("cargo:rerun-if-changed=native/person_shim.cpp");
    for var in ["CUDA_HOME", "TENSORRT_INCLUDE", "OPENCV_INCLUDE"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let env_path = |var: &str| std::env::var_os(var).map(PathBuf::from);
    let cuda = env_path("CUDA_HOME").unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let opencv = env_path("OPENCV_INCLUDE")
        .or_else(|| first_existing(&["/usr/include/opencv4".into(), "/usr/local/include/opencv4".into()]))
        .expect("OpenCV 4 headers not found (install libopencv-dev or set OPENCV_INCLUDE)");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .opt_level(2)
        .file("native/person_shim.cpp")
        .include(cuda.join("include"))
        .include(&opencv)
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-unused-parameter");
    // libnvinfer-dev puts NvInfer.h in the multiarch include dir (aarch64 on the Jetson, x86_64 on a PC)
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let multiarch = format!("/usr/include/{arch}-linux-gnu/NvInfer.h");
    if let Some(trt) = env_path("TENSORRT_INCLUDE") {
        build.include(trt);
    } else if !Path::new(&multiarch).exists() && !Path::new("/usr/include/NvInfer.h").exists() {
        panic!("TensorRT headers (NvInfer.h) not found: install libnvinfer-dev or set TENSORRT_INCLUDE");
    }
    build.compile("person_shim");

    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    for lib in ["nvinfer", "cudart", "opencv_calib3d", "opencv_imgcodecs", "opencv_imgproc", "opencv_core"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
