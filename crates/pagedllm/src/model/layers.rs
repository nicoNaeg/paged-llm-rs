//! The three pieces every transformer block is built from.

use candle_core::{D, DType, Tensor};

use crate::Result;

/// A matrix multiply with no bias, which is what every projection in Qwen3 is.
#[derive(Debug)]
pub struct Linear {
    /// Transposed once at load time. safetensors stores `[out, in]` and the
    /// multiply wants `[in, out]`, and paying that per forward pass would put a
    /// copy of every weight on the critical path of every token.
    weight_t: Tensor,
    out_features: usize,
}

impl Linear {
    /// Build from a weight in the layout safetensors stores, `[out, in]`.
    pub fn new(weight: &Tensor) -> Result<Self> {
        Ok(Self {
            weight_t: weight.t()?.contiguous()?,
            out_features: weight.dim(0)?,
        })
    }

    /// Map the last dimension, leaving every leading dimension alone.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let in_features = dims[dims.len() - 1];
        let flat = x.reshape(((), in_features))?;
        let y = flat.matmul(&self.weight_t)?;
        let mut out_dims = dims.to_vec();
        out_dims[dims.len() - 1] = self.out_features;
        Ok(y.reshape(out_dims)?)
    }
}

/// Root mean square normalisation over the last dimension.
#[derive(Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Build from a gain vector as wide as the dimension being normalised.
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Normalise, then apply the gain.
    ///
    /// The statistic is accumulated in f32 whatever the input dtype, then the
    /// result is cast back before the gain is applied. That is not a detail:
    /// summing 1024 squared bf16 values in bf16 loses enough precision to move
    /// the logits, and the gain multiply has to happen in the input dtype for
    /// the output to match the reference bit for bit.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let wide = x.to_dtype(DType::F32)?;
        let variance = wide.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = wide.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(normed.to_dtype(dtype)?.broadcast_mul(&self.weight)?)
    }
}

/// The gated feed-forward network, `SwiGLU`.
// The shared suffix is the checkpoint's own naming, and renaming the fields
// would put a translation between the weight file and the code that reads it.
#[allow(clippy::struct_field_names)]
#[derive(Debug)]
pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    /// Assemble from the three projections.
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// `down(silu(gate(x)) * up(x))`.
    ///
    /// Which of the two branches carries the activation is not symmetric, and
    /// swapping them produces a network that trains and generates fluent text
    /// while disagreeing with the checkpoint it loaded.
    pub fn forward(&self, x: &Tensor, trace: &mut super::Trace, prefix: &str) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        trace.record(&format!("{prefix}.gate_proj.out"), &gate);
        trace.record(&format!("{prefix}.up_proj.out"), &up);

        let gated = (gate.silu()? * up)?;
        trace.record(&format!("{prefix}.down_proj.in"), &gated);

        let out = self.down_proj.forward(&gated)?;
        trace.record(&format!("{prefix}.down_proj.out"), &out);
        trace.record(&format!("{prefix}.out"), &out);
        Ok(out)
    }
}
