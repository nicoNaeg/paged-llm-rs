//! Grouped-query attention with the query and key norms Qwen3 adds.

use candle_core::{D, DType, Tensor};

use super::Trace;
use super::layers::{Fused, Linear, RmsNorm};
use super::rope::Rope;
use crate::Result;
use crate::batch::{Batch, PagedCache};
use crate::kernels::{AttentionKind, PagedAttention};

/// One attention block.
#[derive(Debug)]
pub struct Attention {
    /// Q, K and V stacked: all three read the layer's input, so they are one
    /// multiply whose result is the three laid end to end.
    qkv: Fused,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    /// Assemble from the projections, the two head norms and the head counts.
    ///
    /// # Errors
    ///
    /// If the three stacked projections disagree about their input width.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj: &Tensor,
        k_proj: &Tensor,
        v_proj: &Tensor,
        o_proj: Linear,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            qkv: Fused::new(&[q_proj, k_proj, v_proj])?,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// Attend over `x`, shaped `[batch, seq, hidden]`.
    ///
    /// The order here is the part that cannot be guessed from a Llama
    /// implementation: the projections are split into heads first, the norms
    /// run over one head's width, and only then does the rotation apply.
    /// Normalising after the rotation, or over the concatenated heads, produces
    /// a model that runs and is wrong.
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &Rope,
        offset: usize,
        trace: &mut Trace,
        prefix: &str,
    ) -> Result<Tensor> {
        let (batch, seq, _) = x.dims3()?;

        let qkv = self.qkv.forward(x)?;
        trace.record(&format!("{prefix}.q_proj.out"), &qkv[0]);
        trace.record(&format!("{prefix}.k_proj.out"), &qkv[1]);
        trace.record(&format!("{prefix}.v_proj.out"), &qkv[2]);

        let q = qkv[0].reshape((batch, seq, self.num_heads, self.head_dim))?;
        let k = qkv[1].reshape((batch, seq, self.num_kv_heads, self.head_dim))?;
        let v = qkv[2].reshape((batch, seq, self.num_kv_heads, self.head_dim))?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        trace.record(&format!("{prefix}.q_norm.out"), &q);
        trace.record(&format!("{prefix}.k_norm.out"), &k);

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let q = rope.apply(&q, offset)?;
        let k = rope.apply(&k, offset)?;

        let group = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, group)?;
        let v = repeat_kv(&v, group)?;

        let keys = k.dim(D::Minus2)?;
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * self.scale)?;
        let scores = scores.broadcast_add(&causal_mask(seq, keys, scores.dtype(), x.device())?)?;
        let weights = softmax_last_dim(&scores)?;

        let out = weights.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq, self.num_heads * self.head_dim))?;
        trace.record(&format!("{prefix}.o_proj.in"), &out);

        let out = self.o_proj.forward(&out)?;
        trace.record(&format!("{prefix}.o_proj.out"), &out);
        trace.record(&format!("{prefix}.out"), &out);
        Ok(out)
    }

    /// Attend for a batch of sequences reading a shared, pre-allocated cache.
    ///
    /// The single-sequence path above owns its cache and grows it. This one is
    /// handed a slot in a pool it does not own, writes into it in place, and
    /// reads back a rectangle every row shares. That difference is the whole of
    /// stage 3.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batch(
        &self,
        x: &Tensor,
        rope: &Rope,
        layer: usize,
        batch: &Batch,
        cache: &PagedCache,
        index: &PassIndex,
        attention: AttentionKind,
    ) -> Result<Tensor> {
        let (rows, seq, _) = x.dims3()?;

        let qkv = self.qkv.forward(x)?;
        let q = qkv[0].reshape((rows, seq, self.num_heads, self.head_dim))?;
        let k = qkv[1].reshape((rows, seq, self.num_kv_heads, self.head_dim))?;
        let v = qkv[2].reshape((rows, seq, self.num_kv_heads, self.head_dim))?;

        let q = self.q_norm.forward(&q)?.transpose(1, 2)?.contiguous()?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let q = rope.apply_at(&q, &index.positions)?;
        let k = rope.apply_at(&k, &index.positions)?;

        // Written before the read, so a row attends to the token it is
        // producing as well as to its history. Writing after would leave every
        // query one position short of itself.
        cache.write(layer, &k, &v, &index.write_slots)?;

        // The kernel reads the blocks where they are. It only covers a decode,
        // where the gather below costs the most and buys the least: one token of
        // query against a rectangle as wide as the longest resident sequence.
        if attention == AttentionKind::Kernel && seq == 1 {
            let (k_pool, v_pool) = cache.layer(layer);
            let op = PagedAttention::new(
                k_pool,
                v_pool,
                batch,
                cache.config().block_size,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                // The scale is 1/sqrt(head_dim), which f32 holds exactly enough of.
                #[allow(clippy::cast_possible_truncation)]
                {
                    self.scale as f32
                },
            )?;
            let q = q
                .transpose(1, 2)?
                .reshape((rows, self.num_heads, self.head_dim))?;
            let out = op.forward(&q.contiguous()?)?.reshape((
                rows,
                seq,
                self.num_heads * self.head_dim,
            ))?;
            return self.o_proj.forward(&out);
        }

        let (keys, values) = cache.read(layer, batch, &index.read_blocks)?;

        // Grouped-query attention without materialising the group. Expanding
        // eight key heads into sixteen is a copy, per layer and per pass, on
        // data that grows with the batch. Folding the group into the query
        // dimension asks the same arithmetic of one batched multiply and copies
        // nothing: query head `kv * group + g` already sits where a reshape puts
        // it.
        let group = self.num_heads / self.num_kv_heads;
        let kv_heads = self.num_kv_heads;
        let longest = keys.dim(D::Minus2)?;
        let q = q.reshape((rows, kv_heads, group * seq, self.head_dim))?;

        let scores = (q.matmul(&keys.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;
        let scores = scores
            .reshape((rows, kv_heads, group, seq, longest))?
            .broadcast_add(&index.mask.unsqueeze(1)?)?
            .reshape((rows, kv_heads, group * seq, longest))?;
        let weights = softmax_last_dim(&scores)?;

        let out = weights
            .matmul(&values)?
            .reshape((rows, self.num_heads, seq, self.head_dim))?
            .transpose(1, 2)?
            .reshape((rows, seq, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

/// Everything a pass needs that does not change from layer to layer.
///
/// Built once and handed down, because each of these costs a transfer to the
/// device and there are twenty-eight layers to pay it in.
#[derive(Debug)]
pub struct PassIndex {
    /// Where each token's key and value is written, `[tokens, kv_heads * head_dim]`.
    pub write_slots: Tensor,
    /// The blocks the batch reads, one entry per block of the rectangle.
    pub read_blocks: Tensor,
    /// Absolute position of every token, `[rows, seq]`.
    pub positions: Tensor,
    /// What each row may read, `[rows, 1, seq, longest]`.
    pub mask: Tensor,
}

/// Expand each key or value head to cover the query heads that share it.
///
/// Query head `h` reads key head `h / group`, which is the grouping the
/// checkpoint was trained with. Interleaving instead, so that head `h` reads
/// `h % num_kv_heads`, is the same tensor shape carrying different data.
fn repeat_kv(x: &Tensor, group: usize) -> Result<Tensor> {
    if group == 1 {
        return Ok(x.clone());
    }
    let (batch, kv_heads, seq, head_dim) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .expand((batch, kv_heads, group, seq, head_dim))?
        .reshape((batch, kv_heads * group, seq, head_dim))?)
}

/// Additive mask forbidding a query at position `i` from reading a key past it.
///
/// The offset is derived from the two lengths rather than taken as an argument.
/// The number of keys is whatever the cache actually holds, and computing where
/// the queries sit from it means a cache and a mask cannot disagree.
fn causal_mask(
    seq: usize,
    keys: usize,
    dtype: DType,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let offset = keys - seq;
    let mut mask = Vec::with_capacity(seq * keys);
    for query in 0..seq {
        for key in 0..keys {
            mask.push(if key <= offset + query {
                0f32
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    // Every row keeps at least its own position, so no row is entirely masked
    // and the subtraction inside the softmax never sees infinity minus itself.
    Ok(Tensor::from_vec(mask, (seq, keys), device)?.to_dtype(dtype)?)
}

/// Softmax over the last dimension, accumulated in f32.
///
/// The reference computes it in f32 whatever the model dtype and casts back
/// afterwards. Summing a few thousand exponentials in bf16 is enough to move
/// the argmax on a close call.
fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let wide = x.to_dtype(DType::F32)?;
    let max = wide.max_keepdim(D::Minus1)?;
    let exp = wide.broadcast_sub(&max)?.exp()?;
    let sum = exp.sum_keepdim(D::Minus1)?;
    Ok(exp.broadcast_div(&sum)?.to_dtype(dtype)?)
}
