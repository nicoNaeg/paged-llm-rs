//! An LLM inference engine built from its primitives.
//!
//! The engine is the deliverable, not the transformer under it. Weight loading
//! and matrix multiplication come from candle; the scheduler, the block
//! allocator that gives the KV cache its paged layout, the attention kernels
//! that read through a block table, and the batching policy that decides what
//! runs on each step are written here.
//!
//! The engine is synchronous and owns one model. The server crate drives it
//! from an async runtime: a step is a blocking GPU dispatch either way, and a
//! synchronous engine stays testable and profilable without a runtime.

pub mod backend;
pub mod batch;
pub mod blocks;
pub mod chat;
pub mod config;
pub mod error;
pub mod kernels;
pub mod model;
pub mod sampler;
pub mod scheduler;
pub mod session;
pub mod tokenizer;
pub mod weights;

// Re-exported so a crate that drives this engine names a device without taking
// a direct dependency on the tensor library underneath it.
pub use candle_core::{DType, Device};

pub use backend::Backend;
pub use batch::{Batch, CacheConfig, PagedCache};
pub use blocks::{BlockAllocator, BlockId, BlockTable};
pub use chat::ChatTemplate;
pub use config::{Config, GenerationConfig};
pub use error::{Error, Result};
pub use kernels::{AttentionKind, PagedAttention};
pub use model::{Model, Trace};
pub use sampler::{Rng, Sampling};
pub use scheduler::{Metrics, Plan, Scheduler, Sequence};
pub use session::{Finish, Request};
pub use tokenizer::{IncrementalDecoder, Tokenizer};
pub use weights::Weights;
