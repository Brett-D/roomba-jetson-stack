//! TensorRT engine runner (TensorRT has no C API: native/person_shim.cpp wraps it). Like
//! roomba_perception/person_detector.py's TrtYolo: deserializes an engine, one execution context
//! and stream, and CUDA managed memory for every I/O tensor, which the CPU reads and writes
//! directly (the Jetson's iGPU shares DRAM) between runs. Unlike Python, the host blocks instead of
//! spinning while the GPU works, and the engine's kernels are replayed as one CUDA graph.

use std::ffi::{c_char, c_int, c_void, CStr, CString};

extern "C" {
    fn rr_trt_load(path: *const c_char, err: *mut c_char, errlen: usize) -> *mut c_void;
    fn rr_trt_free(engine: *mut c_void);
    fn rr_trt_num_tensors(engine: *const c_void) -> c_int;
    fn rr_trt_tensor(
        engine: *const c_void,
        i: c_int,
        name: *mut *const c_char,
        is_input: *mut c_int,
        dtype: *mut c_int,
        dims: *mut i64,
        max_dims: c_int,
        ptr: *mut *mut c_void,
        nbytes: *mut usize,
    ) -> c_int;
    fn rr_trt_infer(engine: *mut c_void, err: *mut c_char, errlen: usize) -> c_int;
}

/// nvinfer1::DataType::kFLOAT
pub const DTYPE_FLOAT: i32 = 0;

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub is_input: bool,
    pub dtype: i32,
    pub dims: Vec<i64>,
    ptr: *mut c_void,
    nbytes: usize,
}

/// A loaded engine with float32 input and output (the last input and output tensors, as in Python).
pub struct TrtEngine {
    raw: *mut c_void,
    tensors: Vec<TensorInfo>,
    input: usize,
    output: usize,
}

// SAFETY: the engine and its buffers are only used through &mut self / &self of one owner.
unsafe impl Send for TrtEngine {}

fn err_text(buf: &[u8]) -> String {
    CStr::from_bytes_until_nul(buf).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

impl TrtEngine {
    pub fn load(path: &str) -> Result<Self, String> {
        let cpath = CString::new(path).map_err(|e| e.to_string())?;
        let mut err = vec![0u8; 512];
        // SAFETY: valid C string and error buffer of the given length.
        let raw = unsafe { rr_trt_load(cpath.as_ptr(), err.as_mut_ptr().cast(), err.len()) };
        if raw.is_null() {
            return Err(err_text(&err));
        }
        let mut engine = TrtEngine { raw, tensors: Vec::new(), input: usize::MAX, output: usize::MAX };
        // SAFETY: raw is a live engine; the shim fills the out-parameters for index i < count.
        let count = unsafe { rr_trt_num_tensors(raw) };
        for i in 0..count {
            let (mut name, mut is_input, mut dtype) = (std::ptr::null(), 0, 0);
            let (mut ptr, mut nbytes) = (std::ptr::null_mut(), 0usize);
            let mut dims = [0i64; 8];
            let n = unsafe {
                rr_trt_tensor(raw, i, &mut name, &mut is_input, &mut dtype, dims.as_mut_ptr(), 8, &mut ptr, &mut nbytes)
            };
            let name = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
            engine.tensors.push(TensorInfo {
                name,
                is_input: is_input != 0,
                dtype,
                dims: dims[..(n.clamp(0, 8) as usize)].to_vec(),
                ptr,
                nbytes,
            });
            if is_input != 0 {
                engine.input = i as usize;
            } else {
                engine.output = i as usize;
            }
        }
        if engine.input == usize::MAX || engine.output == usize::MAX {
            return Err(format!("{path}: the engine needs an input and an output tensor"));
        }
        for t in [&engine.tensors[engine.input], &engine.tensors[engine.output]] {
            if t.dtype != DTYPE_FLOAT {
                return Err(format!("{path}: tensor {} is not float32 (TensorRT DataType {})", t.name, t.dtype));
            }
        }
        Ok(engine)
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn input_info(&self) -> &TensorInfo {
        &self.tensors[self.input]
    }

    pub fn output_info(&self) -> &TensorInfo {
        &self.tensors[self.output]
    }

    /// The input tensor in managed memory (write it, then [`TrtEngine::infer`]).
    pub fn input_mut(&mut self) -> &mut [f32] {
        let t = &self.tensors[self.input];
        // SAFETY: a managed allocation of nbytes owned by the engine; no inference runs concurrently.
        unsafe { std::slice::from_raw_parts_mut(t.ptr.cast(), t.nbytes / 4) }
    }

    pub fn input(&self) -> &[f32] {
        let t = &self.tensors[self.input];
        unsafe { std::slice::from_raw_parts(t.ptr.cast(), t.nbytes / 4) }
    }

    /// The output tensor of the last [`TrtEngine::infer`].
    pub fn output(&self) -> &[f32] {
        let t = &self.tensors[self.output];
        unsafe { std::slice::from_raw_parts(t.ptr.cast(), t.nbytes / 4) }
    }


    /// Runs the engine on its stream (enqueueV3, or the recorded CUDA graph), then waits for it.
    pub fn infer(&mut self) -> Result<(), String> {
        let mut err = vec![0u8; 256];
        // SAFETY: live engine; the CPU does not touch the buffers until this returns.
        match unsafe { rr_trt_infer(self.raw, err.as_mut_ptr().cast(), err.len()) } {
            0 => Ok(()),
            _ => Err(err_text(&err)),
        }
    }
}

impl Drop for TrtEngine {
    fn drop(&mut self) {
        // SAFETY: raw came from rr_trt_load and is freed once.
        unsafe { rr_trt_free(self.raw) }
    }
}
