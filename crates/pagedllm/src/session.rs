//! One sequence, generated token by token.
//!
//! This is the whole engine at stage 2: a prompt goes in, a cache grows, tokens
//! come out, and nothing else runs at the same time. Stage 3 replaces it with a
//! scheduler that advances many of these on one forward pass, and the shape of
//! this loop is what that has to reproduce for one sequence.

use candle_core::{D, DType};

use crate::cache::KvCache;
use crate::model::Model;
use crate::sampler::{Rng, Sampling};
use crate::{Error, Result};

/// Why a generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// The model produced a token that ends a turn.
    Stop,
    /// The caller's token budget ran out first.
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

/// A sequence in the middle of being generated.
#[derive(Debug)]
pub struct Session<'a> {
    model: &'a Model,
    cache: KvCache,
    sampling: Sampling,
    rng: Rng,
    stop_tokens: Vec<u32>,
    max_tokens: usize,
    /// Tokens fed on the next pass: the whole prompt first, then one token.
    pending: Vec<u32>,
    prompt_tokens: usize,
    generated: usize,
    finish: Option<Finish>,
}

impl<'a> Session<'a> {
    /// Start a generation. The prompt is not run until the first token is
    /// asked for, so a caller that gives up costs nothing.
    pub fn new(model: &'a Model, request: Request) -> Result<Self> {
        if request.prompt.is_empty() {
            return Err(Error::Config("cannot generate from an empty prompt".into()));
        }
        Ok(Self {
            cache: KvCache::new(model.config().num_hidden_layers),
            model,
            sampling: request.sampling,
            rng: Rng::new(request.seed),
            stop_tokens: request.stop_tokens,
            max_tokens: request.max_tokens,
            prompt_tokens: request.prompt.len(),
            pending: request.prompt,
            generated: 0,
            finish: None,
        })
    }

    /// The next token, or `None` once the generation is over.
    ///
    /// A token that ends a turn is consumed rather than returned: it is the
    /// signal to stop, not part of what the model said.
    pub fn next_token(&mut self) -> Result<Option<u32>> {
        if self.finish.is_some() {
            return Ok(None);
        }
        if self.generated >= self.max_tokens {
            self.finish = Some(Finish::Length);
            return Ok(None);
        }

        let logits = self.model.forward_cached(&self.pending, &mut self.cache)?;
        // Only the last position predicts the next token. The others were
        // computed because the prompt had to pass through the model to fill the
        // cache, not because anything reads them.
        let last = logits
            .narrow(1, logits.dim(1)? - 1, 1)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let token = self.sampling.sample(&last, &mut self.rng)?;

        self.pending.clear();
        self.pending.push(token);
        self.generated += 1;

        if self.stop_tokens.contains(&token) {
            self.finish = Some(Finish::Stop);
            return Ok(None);
        }
        if self.generated >= self.max_tokens {
            self.finish = Some(Finish::Length);
        }
        Ok(Some(token))
    }

    /// Why the generation stopped, once it has.
    pub fn finish_reason(&self) -> Option<Finish> {
        self.finish
    }

    /// How many tokens the prompt held.
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// How many tokens have been produced, including one that ended the turn.
    pub fn generated(&self) -> usize {
        self.generated
    }

    /// How many positions the cache holds.
    pub fn cached_tokens(&self) -> usize {
        self.cache.tokens()
    }
}

/// Logits for the last position of `tokens`, as f32 on the host.
///
/// Sampling runs on the CPU, so every step copies one vocabulary of logits back
/// from the device: 600 KB per token on Qwen3. That is a real cost and it is not
/// addressed here, because it is the same cost whatever the cache underneath
/// looks like, and stage 6 is where a profile gets to say whether it matters.
pub fn last_logits(model: &Model, tokens: &[u32]) -> Result<Vec<f32>> {
    let logits = model.forward(tokens)?;
    Ok(logits
        .narrow(1, logits.dim(1)? - 1, 1)?
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?)
}

/// The id of the highest-scoring token at the last position.
pub fn greedy_next(model: &Model, tokens: &[u32]) -> Result<u32> {
    let logits = model.forward(tokens)?;
    Ok(logits
        .narrow(1, logits.dim(1)? - 1, 1)?
        .flatten_all()?
        .argmax(D::Minus1)?
        .to_scalar::<u32>()?)
}
