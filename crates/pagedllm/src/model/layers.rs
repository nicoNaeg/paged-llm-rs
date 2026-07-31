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

/// Several projections of the same input, multiplied as one.
///
/// Q, K and V all read the layer's input, and so do the MLP's gate and up. Run
/// separately that is one dispatch each, and on a decode step a dispatch is what
/// costs: the profile under `docs/` puts a step at about 150 command buffer
/// submissions against a bandwidth floor of 5.5 ms, so what is left to win is
/// the number of times work is handed to the GPU rather than the work itself.
///
/// Stacking the weights along their output dimension turns those into one
/// multiply whose result is the pieces laid end to end, which `split` hands back
/// as views. Whether that is faster is measured rather than assumed, by
/// `cargo run --release --features metal --example step_cost`.
#[derive(Debug)]
pub struct Fused {
    weight_t: Tensor,
    widths: Vec<usize>,
}

impl Fused {
    /// Stack projections given in the layout safetensors stores, `[out, in]`.
    ///
    /// # Errors
    ///
    /// If the weights disagree about their input width, which means they were
    /// not all reading the same thing and had no business being fused.
    pub fn new(weights: &[&Tensor]) -> Result<Self> {
        let widths = weights
            .iter()
            .map(|w| Ok(w.dim(0)?))
            .collect::<Result<Vec<_>>>()?;
        let stacked = Tensor::cat(weights, 0)?;
        Ok(Self {
            weight_t: stacked.t()?.contiguous()?,
            widths,
        })
    }

    /// Multiply once and hand back one piece per weight that was stacked.
    ///
    /// The pieces are narrows of the result, so nothing is copied here. What
    /// reads them decides whether it needs a contiguous tensor, which is where
    /// the cost of splitting shows up if it shows up at all.
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let dims = x.dims();
        let in_features = dims[dims.len() - 1];
        let y = x.reshape(((), in_features))?.matmul(&self.weight_t)?;

        let mut pieces = Vec::with_capacity(self.widths.len());
        let mut at = 0;
        for &width in &self.widths {
            let mut out_dims = dims.to_vec();
            out_dims[dims.len() - 1] = width;
            pieces.push(y.narrow(1, at, width)?.reshape(out_dims)?);
            at += width;
        }
        Ok(pieces)
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
    /// Gate and up stacked: both read `x`, so they are one multiply.
    gate_up: Fused,
    down_proj: Linear,
}

impl Mlp {
    /// Assemble from the three projections, stacking the two that share input.
    ///
    /// # Errors
    ///
    /// If gate and up disagree about their input width.
    pub fn new(gate_proj: &Tensor, up_proj: &Tensor, down_proj: Linear) -> Result<Self> {
        Ok(Self {
            gate_up: Fused::new(&[gate_proj, up_proj])?,
            down_proj,
        })
    }

    /// `down(silu(gate(x)) * up(x))`.
    ///
    /// Which of the two branches carries the activation is not symmetric, and
    /// swapping them produces a network that trains and generates fluent text
    /// while disagreeing with the checkpoint it loaded.
    pub fn forward(&self, x: &Tensor, trace: &mut super::Trace, prefix: &str) -> Result<Tensor> {
        let pieces = self.gate_up.forward(x)?;
        let (gate, up) = (&pieces[0], &pieces[1]);
        trace.record(&format!("{prefix}.gate_proj.out"), gate);
        trace.record(&format!("{prefix}.up_proj.out"), up);

        let gated = (gate.silu()? * up)?;
        trace.record(&format!("{prefix}.down_proj.in"), &gated);

        let out = self.down_proj.forward(&gated)?;
        trace.record(&format!("{prefix}.down_proj.out"), &out);
        trace.record(&format!("{prefix}.out"), &out);
        Ok(out)
    }
}
