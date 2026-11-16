//! ONNX model wrapper (tract): fixed input shapes, f32 in, the first output as f32.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tract_onnx::prelude::*;

pub struct OnnxModel {
    plan: Arc<TypedRunnableModel>,
    shapes: Vec<Vec<usize>>,
}

impl OnnxModel {
    /// Loads and optimises `path` for the given input shapes (one per model input, in order).
    pub fn load(path: impl AsRef<Path>, shapes: &[&[usize]]) -> Result<Self> {
        let path = path.as_ref();
        let mut model = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("loading {}", path.display()))?;
        for (i, s) in shapes.iter().enumerate() {
            model.set_input_fact(i, f32::fact(*s).into())?;
        }
        let plan = model.into_optimized()?.into_runnable()?;
        Ok(OnnxModel { plan, shapes: shapes.iter().map(|s| s.to_vec()).collect() })
    }

    /// Runs the model; `inputs` are row-major buffers matching the shapes given to [`OnnxModel::load`].
    pub fn run(&self, inputs: &[&[f32]]) -> Result<Vec<f32>> {
        anyhow::ensure!(inputs.len() == self.shapes.len(), "expected {} inputs, got {}", self.shapes.len(), inputs.len());
        let mut tv: TVec<TValue> = TVec::new();
        for (data, shape) in inputs.iter().zip(&self.shapes) {
            let n: usize = shape.iter().product();
            anyhow::ensure!(data.len() == n, "input of {} values for shape {:?}", data.len(), shape);
            tv.push(Tensor::from_shape(shape, data)?.into());
        }
        let mut out = self.plan.run(tv)?;
        let t = out.remove(0).into_tensor();
        Ok(t.to_plain_array_view::<f32>()?.iter().copied().collect())
    }
}
