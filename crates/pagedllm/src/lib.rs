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
pub mod error;

pub use backend::Backend;
pub use error::{Error, Result};
