//! Attention over a history scattered across physical blocks.
//!
//! The tensor path stage 3 built gathers every resident sequence's keys and
//! values into one rectangle before it multiplies, per layer and per step, and
//! the rectangle is as wide as the longest row. This reads the blocks where they
//! are.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, DType, Layout, Shape, Tensor};

use crate::batch::Batch;
use crate::{Error, Result};

/// Which attention implementation the engine runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionKind {
    /// Gather the blocks into a rectangle and use candle's tensor ops. Stage 3's
    /// path, kept because it is what the kernel is checked against.
    #[default]
    Tensor,
    /// The hand-written kernel, which reads the blocks in place.
    Kernel,
}

impl std::fmt::Display for AttentionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Tensor => "tensor",
            Self::Kernel => "kernel",
        })
    }
}

impl std::str::FromStr for AttentionKind {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "tensor" => Ok(Self::Tensor),
            "kernel" => Ok(Self::Kernel),
            other => Err(Error::Config(format!(
                "attention {other:?}, which is neither tensor nor kernel"
            ))),
        }
    }
}

/// Threads in one threadgroup. A power of two, because the softmax reduction
/// halves it, and a multiple of the 32-lane SIMD group the scoring uses.
#[cfg(feature = "metal")]
const THREADS: usize = 128;

/// One decode step's attention, read through a block table.
///
/// Carries the pool and the tables as fields rather than as arguments because
/// candle's custom operations take at most three tensors and this needs five.
pub struct PagedAttention {
    k_pool: Tensor,
    v_pool: Tensor,
    /// `[rows, max_blocks]`, the physical block behind each logical one.
    block_tables: Tensor,
    /// `[rows]`, how many positions each row attends to.
    context_lens: Tensor,
    block_size: usize,
    num_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    max_blocks: usize,
    scale: f32,
}

// Written out rather than derived: the pool and the tables are tensors, and
// printing a few million cache entries to describe one operation is not a thing
// anyone wants from a Debug impl.
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for PagedAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedAttention")
            .field("block_size", &self.block_size)
            .field("num_heads", &self.num_heads)
            .field("kv_heads", &self.kv_heads)
            .field("head_dim", &self.head_dim)
            .finish()
    }
}

impl PagedAttention {
    /// Build from the pool and the batch that is about to read it.
    ///
    /// Only decode passes go through here. A prefill is one row of many tokens,
    /// which is a different kernel shape and a place the measurements show no
    /// problem, so it keeps the tensor path.
    // The shape of the model and the shape of the pool, which is what it takes.
    // Bundling them into a struct would move the same arguments one call up.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k_pool: &Tensor,
        v_pool: &Tensor,
        batch: &Batch,
        block_size: usize,
        num_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Self> {
        if batch.seq != 1 {
            return Err(Error::Unsupported(format!(
                "the kernel decodes one token a row, and this batch has {}",
                batch.seq
            )));
        }
        let device = k_pool.device();
        let max_blocks = batch.blocks_per_row;
        let block_tables =
            Tensor::from_slice(&batch.read_blocks, (batch.rows, max_blocks.max(1)), device)?;
        let lengths: Vec<u32> = batch
            .starts
            .iter()
            .map(|start| u32::try_from(start + 1).unwrap_or(u32::MAX))
            .collect();
        let context_lens = Tensor::from_vec(lengths, batch.rows, device)?;

        Ok(Self {
            k_pool: k_pool.clone(),
            v_pool: v_pool.clone(),
            block_tables,
            context_lens,
            block_size,
            num_heads,
            kv_heads,
            head_dim,
            max_blocks: max_blocks.max(1),
            scale,
        })
    }

    /// Attend for `q`, shaped `[rows, num_heads, head_dim]`.
    pub fn forward(&self, q: &Tensor) -> Result<Tensor> {
        Ok(q.apply_op1_no_bwd(self)?)
    }

    /// The scalar reference, which is what CI checks and what the kernel is
    /// checked against on hardware.
    ///
    /// Deliberately a loop rather than a tensor expression. A tensor expression
    /// would be a second version of the path this replaces, and would agree with
    /// it for reasons that have nothing to do with the kernel being right.
    fn reference(&self, q: &[f32], rows: usize) -> Result<Vec<f32>> {
        let read = |t: &Tensor| -> Result<Vec<f32>> {
            Ok(t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?)
        };
        let k_pool = read(&self.k_pool)?;
        let v_pool = read(&self.v_pool)?;
        let tables = self.block_tables.flatten_all()?.to_vec1::<u32>()?;
        let lengths = self.context_lens.flatten_all()?.to_vec1::<u32>()?;

        let kv_width = self.kv_heads * self.head_dim;
        let group = self.num_heads / self.kv_heads;
        let mut out = vec![0f32; rows * self.num_heads * self.head_dim];

        for row in 0..rows {
            let ctx = lengths[row] as usize;
            if ctx == 0 {
                continue;
            }
            let table = &tables[row * self.max_blocks..(row + 1) * self.max_blocks];
            // The indirection, spelled out: a logical position becomes a
            // physical slot through the table, and consecutive positions stop
            // being consecutive in memory at every block boundary.
            let slot_of = |position: usize| -> usize {
                table[position / self.block_size] as usize * self.block_size
                    + position % self.block_size
            };

            for head in 0..self.num_heads {
                let kv_head = head / group;
                let q_head = &q[(row * self.num_heads + head) * self.head_dim..][..self.head_dim];

                let mut scores = Vec::with_capacity(ctx);
                for position in 0..ctx {
                    let base = slot_of(position) * kv_width + kv_head * self.head_dim;
                    let dot: f32 = (0..self.head_dim)
                        .map(|d| q_head[d] * k_pool[base + d])
                        .sum();
                    scores.push(dot * self.scale);
                }

                let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0f32;
                for score in &mut scores {
                    *score = (*score - peak).exp();
                    total += *score;
                }

                let out_base = (row * self.num_heads + head) * self.head_dim;
                for (position, weight) in scores.iter().enumerate() {
                    let base = slot_of(position) * kv_width + kv_head * self.head_dim;
                    for d in 0..self.head_dim {
                        out[out_base + d] += weight * v_pool[base + d];
                    }
                }
                for d in 0..self.head_dim {
                    out[out_base + d] /= total;
                }
            }
        }
        Ok(out)
    }
}

impl candle_core::CustomOp1 for PagedAttention {
    fn name(&self) -> &'static str {
        "paged-attention-decode"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let shape = layout.shape().clone();
        let rows = shape.dims()[0];
        let q: Vec<f32> = match storage {
            CpuStorage::F32(v) => v.clone(),
            CpuStorage::BF16(v) => v.iter().map(|x| x.to_f32()).collect(),
            CpuStorage::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
            other => candle_core::bail!("paged attention on {:?}", other.dtype()),
        };
        let out = self
            .reference(&q, rows)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let storage = match storage.dtype() {
            DType::F32 => CpuStorage::F32(out),
            DType::BF16 => CpuStorage::BF16(out.into_iter().map(half::bf16::from_f32).collect()),
            DType::F16 => CpuStorage::F16(out.into_iter().map(half::f16::from_f32).collect()),
            dtype => candle_core::bail!("paged attention on {dtype:?}"),
        };
        Ok((storage, shape))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle_core::MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        metal::run(self, storage, layout)
    }
}

#[cfg(feature = "metal")]
mod metal {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use candle_core::backend::BackendStorage;
    use candle_core::{DType, Layout, MetalStorage, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{Buffer, ComputePipeline};
    use candle_metal_kernels::utils::set_param;
    use objc2_metal::MTLSize;

    use super::{PagedAttention, THREADS};

    /// The parameters the kernel reads, laid out as the MSL struct expects.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Params {
        block_size: u32,
        head_dim: u32,
        num_heads: u32,
        kv_heads: u32,
        max_blocks: u32,
        threads: u32,
        simdgroups: u32,
        scale: f32,
    }

    thread_local! {
        /// Compiled pipelines, keyed by the dtype they were built for.
        ///
        /// Thread local rather than shared, because the engine owns one thread
        /// and a pipeline is not worth making shareable to serve a second one
        /// that does not exist. Compiling costs milliseconds and a decode step
        /// costs milliseconds, so this cache is the difference between a kernel
        /// that is faster and one that is not.
        static PIPELINES: RefCell<HashMap<&'static str, ComputePipeline>> =
            RefCell::new(HashMap::new());
    }

    /// The scalar type the kernel is compiled against.
    fn elem(dtype: DType) -> candle_core::Result<&'static str> {
        match dtype {
            DType::BF16 => Ok("bfloat"),
            DType::F16 => Ok("half"),
            DType::F32 => Ok("float"),
            dtype => candle_core::bail!("paged attention on {dtype:?}"),
        }
    }

    /// Compile the kernel for one dtype, once per thread.
    ///
    /// From source at startup rather than ahead of time, so building this
    /// repository needs no Xcode. A syntax error is caught by the test below
    /// that compiles every dtype under `make test-metal`, not by a request. It
    /// cannot be caught by `cargo test` in CI, which has no Metal device.
    pub(super) fn pipeline(
        device: &candle_core::MetalDevice,
        dtype: DType,
    ) -> candle_core::Result<ComputePipeline> {
        let name = elem(dtype)?;
        PIPELINES.with(|cache| {
            if let Some(pipeline) = cache.borrow().get(name) {
                return Ok(pipeline.clone());
            }
            let source = format!(
                "#define ELEM {name}\n{}",
                include_str!("paged_attention.metal")
            );
            let library = device
                .metal_device()
                .new_library_with_source(&source, None)
                .map_err(|e| candle_core::Error::Msg(format!("compiling the kernel: {e}")))?;
            let function = library
                .get_function("paged_attention_decode", None)
                .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
            let pipeline = device
                .metal_device()
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
            cache.borrow_mut().insert(name, pipeline.clone());
            Ok(pipeline)
        })
    }

    /// The Metal buffer behind a tensor, which must be contiguous and unoffset.
    fn buffer(tensor: &Tensor) -> candle_core::Result<Buffer> {
        let (storage, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            candle_core::bail!("the kernel reads whole tensors, and this one is a view");
        }
        match &*storage {
            Storage::Metal(s) => Ok(s.buffer().clone()),
            _ => candle_core::bail!("the kernel needs its inputs on the same Metal device"),
        }
    }

    pub fn run(
        op: &PagedAttention,
        storage: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let shape = layout.shape().clone();
        let dims = shape.dims();
        if dims.len() != 3 {
            candle_core::bail!("queries should be [rows, heads, head_dim], got {dims:?}");
        }
        let (rows, heads, head_dim) = (dims[0], dims[1], dims[2]);
        if !layout.is_contiguous() {
            candle_core::bail!("the kernel reads whole tensors, and the queries are a view");
        }

        let device = storage.device().clone();
        let dtype = storage.dtype();
        let pipeline = pipeline(&device, dtype)?;

        let elements = rows * heads * head_dim;
        let out = device
            .new_buffer(elements, dtype, "paged-attention")
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        let params = Params {
            block_size: u32::try_from(op.block_size).unwrap_or(u32::MAX),
            head_dim: u32::try_from(head_dim).unwrap_or(u32::MAX),
            num_heads: u32::try_from(heads).unwrap_or(u32::MAX),
            kv_heads: u32::try_from(op.kv_heads).unwrap_or(u32::MAX),
            max_blocks: u32::try_from(op.max_blocks).unwrap_or(u32::MAX),
            threads: u32::try_from(THREADS).unwrap_or(u32::MAX),
            simdgroups: u32::try_from(THREADS / 32).unwrap_or(u32::MAX),
            scale: op.scale,
        };

        // The scores live in threadgroup memory, one float per position the
        // longest row reaches. Sized here rather than fixed in the source, so a
        // short batch does not reserve the memory a long one would need and cost
        // the occupancy that reservation buys nothing for.
        let longest = op.max_blocks * op.block_size;
        let scores_bytes = longest * std::mem::size_of::<f32>();
        let limit = 32 << 10;
        if scores_bytes + THREADS * std::mem::size_of::<f32>() > limit {
            candle_core::bail!(
                "a context of {longest} needs {scores_bytes} bytes of threadgroup memory, past the {limit} available; \
                 an online softmax is what removes this limit"
            );
        }

        let k = buffer(&op.k_pool)?;
        let v = buffer(&op.v_pool)?;
        let tables = buffer(&op.block_tables)?;
        let lengths = buffer(&op.context_lens)?;

        let encoder = device
            .command_encoder()
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        encoder.set_compute_pipeline_state(&pipeline);
        let raw = encoder.as_ref();
        set_param(raw, 0, storage.buffer());
        set_param(raw, 1, &k);
        set_param(raw, 2, &v);
        set_param(raw, 3, &tables);
        set_param(raw, 4, &lengths);
        set_param(raw, 5, candle_metal_kernels::Output::new(&out));
        set_param(raw, 6, &[params][..]);
        raw.set_threadgroup_memory_length(0, scores_bytes);
        raw.set_threadgroup_memory_length(1, THREADS * std::mem::size_of::<f32>());
        raw.dispatch_thread_groups(
            MTLSize {
                width: heads,
                height: rows,
                depth: 1,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((MetalStorage::new(out, device, elements, dtype), shape))
    }
}

/// The decision to compile the kernel at startup rather than ahead of time was
/// taken against a `build.rs`, whose one real advantage is that a syntax error
/// stops the build. What replaces it is this: every dtype the kernel is written
/// for is compiled here, so a broken kernel fails `make test-metal` rather than
/// a request. Without it the claim was untrue, since serving only ever compiles
/// the one dtype it was asked for and no test compiled the others at all.
///
/// It cannot run in CI, for the reason everything Metal here cannot: a hosted
/// macOS runner has no device to compile against.
#[cfg(feature = "metal")]
#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    #[test]
    fn every_dtype_the_kernel_claims_actually_compiles() {
        let Ok(device) = Device::new_metal(0) else {
            eprintln!("skipped: no Metal device");
            return;
        };
        let candle_core::Device::Metal(metal) = &device else {
            panic!("new_metal returned something else");
        };
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            super::metal::pipeline(metal, dtype)
                .unwrap_or_else(|e| panic!("the kernel does not compile for {dtype:?}: {e}"));
        }
    }

    #[test]
    fn a_dtype_the_kernel_is_not_written_for_is_refused_by_name() {
        let Ok(device) = Device::new_metal(0) else {
            eprintln!("skipped: no Metal device");
            return;
        };
        let candle_core::Device::Metal(metal) = &device else {
            panic!("new_metal returned something else");
        };
        let refused = super::metal::pipeline(metal, DType::U32);
        assert!(refused.is_err(), "u32 should have no kernel");
    }
}
