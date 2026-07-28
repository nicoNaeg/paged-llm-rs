//! How a request is described, and why a generation stopped.
//!
//! The loop that used to live here is gone. Stage 2 ran one sequence to
//! completion behind a `Session`; stage 3 replaced it with a scheduler that
//! advances many at once, and one sequence is that scheduler over a pool of one
//! slot. Keeping both would have left a path nothing takes.

use crate::sampler::Sampling;

/// Why a generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// The model produced a token that ends a turn.
    Stop,
    /// The caller's token budget ran out first, or the prompt was longer than a
    /// slot can hold.
    Length,
}

/// How a generation is set up.
#[derive(Debug, Clone)]
pub struct Request {
    /// The prompt, already tokenized.
    pub prompt: Vec<u32>,
    /// How the next token is chosen.
    pub sampling: Sampling,
    /// Most tokens to produce.
    pub max_tokens: usize,
    /// Token ids that end the generation, and are not part of the output.
    pub stop_tokens: Vec<u32>,
    /// Seed for the draw, so a run can be repeated exactly.
    pub seed: u64,
}
