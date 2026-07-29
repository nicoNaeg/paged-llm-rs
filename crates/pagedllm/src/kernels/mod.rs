//! Attention kernels, and the reference implementations they are checked against.
//!
//! Every kernel here ships twice. The Metal version is what serves; the scalar
//! version beside it is the oracle, and it is not a testing nicety:
//! `MTLCreateSystemDefaultDevice` returns nil inside GitHub's hosted macOS
//! runners, so the scalar path is the only thing CI can execute. It reads
//! through the block table exactly as the kernel does, which is what makes it a
//! reference for this kernel rather than for attention in general.
//!
//! The check runs three ways. CI compares the scalar path against the tensor
//! path, which came from stage 3 and knows nothing about blocks. `make
//! test-metal` compares the kernel against the scalar path on real hardware.
//! Neither comparison alone would catch a defect that moved two of them the same
//! way, which is a mistake this project has already made once.

pub mod paged_attention;

pub use paged_attention::{AttentionKind, PagedAttention};
