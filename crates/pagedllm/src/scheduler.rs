//! What runs on the next forward pass.
//!
//! The scheduler decides and the engine executes, so nothing here touches a
//! tensor or a device. It hands out slots, builds the descriptor of the next
//! batch, and takes back the tokens that came out. That split is what makes the
//! policy testable at a hundred sequences a millisecond, without a model.
//!
//! The policy is prefill-first, the one vLLM shipped before chunked prefill: a
//! step that can admit a waiting sequence runs its whole prompt and nothing
//! else. It is not the best policy and it is not meant to be. A long prompt
//! stalls every sequence already decoding for the length of its prefill, and
//! that stall is the thing stage 8 exists to remove, which is easier to
//! demonstrate having first been built.
//!
//! When the pool is full a sequence waits. Nothing running is ever evicted,
//! because with a reservation per sequence the unit that would be evicted is the
//! whole reservation, which measures the policy rather than the structure.
//! Preemption becomes worth having at stage 5, where the unit is a block.

use std::collections::VecDeque;

use crate::batch::{Batch, SlotCache};
use crate::sampler::{Rng, Sampling};
use crate::session::Finish;
use crate::{Error, Result};

/// One request, from arrival to its last token.
#[derive(Debug)]
pub struct Sequence {
    /// Names it in the events the engine emits.
    pub id: u64,
    prompt: Vec<u32>,
    sampling: Sampling,
    max_tokens: usize,
    stop_tokens: Vec<u32>,
    rng: Rng,
    slot: Option<usize>,
    /// Fed on the next pass: the whole prompt first, then one token.
    pending: Vec<u32>,
    generated: usize,
    finish: Option<Finish>,
}

impl Sequence {
    /// A sequence that has not started.
    pub fn new(
        id: u64,
        prompt: Vec<u32>,
        sampling: Sampling,
        max_tokens: usize,
        stop_tokens: Vec<u32>,
        seed: u64,
    ) -> Result<Self> {
        if prompt.is_empty() {
            return Err(Error::Config("cannot generate from an empty prompt".into()));
        }
        Ok(Self {
            id,
            pending: prompt.clone(),
            prompt,
            sampling,
            max_tokens,
            stop_tokens,
            rng: Rng::new(seed),
            slot: None,
            generated: 0,
            finish: None,
        })
    }

    /// How the next token is chosen, and the generator it draws from.
    pub fn sampling(&mut self) -> (Sampling, &mut Rng) {
        (self.sampling, &mut self.rng)
    }

    /// How many tokens the prompt held.
    pub fn prompt_tokens(&self) -> usize {
        self.prompt.len()
    }

    /// How many tokens the model has produced.
    pub fn generated(&self) -> usize {
        self.generated
    }

    /// Why it stopped, once it has.
    pub fn finish_reason(&self) -> Option<Finish> {
        self.finish
    }
}

/// What the engine should run next.
#[derive(Debug)]
pub enum Plan {
    /// Nothing is waiting and nothing is running.
    Idle,
    /// One sequence's prompt, alone in the pass.
    Prefill {
        /// The sequence, in row order.
        ids: Vec<u64>,
        /// What to run.
        batch: Batch,
    },
    /// One token for each sequence already running.
    Decode {
        /// The sequences, in row order.
        ids: Vec<u64>,
        /// What to run.
        batch: Batch,
    },
}

impl Plan {
    /// The sequences this plan covers, in row order.
    pub fn ids(&self) -> &[u64] {
        match self {
            Self::Idle => &[],
            Self::Prefill { ids, .. } | Self::Decode { ids, .. } => ids,
        }
    }

    /// The batch to run, when there is one.
    pub fn batch(&self) -> Option<&Batch> {
        match self {
            Self::Idle => None,
            Self::Prefill { batch, .. } | Self::Decode { batch, .. } => Some(batch),
        }
    }
}

/// What a step of the engine changed.
#[derive(Debug, Default)]
pub struct Outcome {
    /// Sequences that produced a token, and which one.
    pub tokens: Vec<(u64, u32)>,
    /// Sequences that ended on this step.
    pub finished: Vec<(u64, Finish)>,
}

/// Counters the server reports and the benchmark reads.
#[derive(Debug, Default, Clone, Copy)]
pub struct Metrics {
    /// Forward passes run.
    pub steps: u64,
    /// Passes that were a prefill rather than a decode.
    pub prefills: u64,
    /// Tokens produced, summed over every sequence.
    pub tokens: u64,
    /// Rows summed over every decode pass, which divided by the decode steps is
    /// the average batch size actually achieved.
    pub decode_rows: u64,
    /// Times a waiting sequence found no free slot.
    pub admission_stalls: u64,
}

impl Metrics {
    /// Average rows per decode pass, which is what continuous batching buys.
    pub fn mean_batch(&self) -> f64 {
        let decodes = self.steps - self.prefills;
        if decodes == 0 {
            return 0.0;
        }
        self.decode_rows as f64 / decodes as f64
    }
}

/// The queue, the running set, and the pool they share.
#[derive(Debug)]
pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    cache: SlotCache,
    max_batch: usize,
    metrics: Metrics,
}

impl Scheduler {
    /// Build over a pool. `max_batch` caps the decode rows independently of the
    /// slots, so a batch can be limited without shrinking the cache.
    pub fn new(cache: SlotCache, max_batch: usize) -> Self {
        let max_batch = max_batch.max(1);
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            cache,
            max_batch,
            metrics: Metrics::default(),
        }
    }

    /// Queue a sequence.
    pub fn submit(&mut self, sequence: Sequence) {
        self.waiting.push_back(sequence);
    }

    /// How many are queued.
    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    /// How many are resident.
    pub fn running(&self) -> usize {
        self.running.len()
    }

    /// The counters so far.
    pub fn metrics(&self) -> Metrics {
        self.metrics
    }

    /// The pool, which the model writes through.
    pub fn cache_mut(&mut self) -> &mut SlotCache {
        &mut self.cache
    }

    /// The sequence with this id, while it is running.
    pub fn sequence_mut(&mut self, id: u64) -> Option<&mut Sequence> {
        self.running.iter_mut().find(|s| s.id == id)
    }

    /// Decide the next pass.
    ///
    /// Admission comes first and takes the whole pass, which is what makes a
    /// long prompt visible as a stall in every other sequence's stream.
    pub fn plan(&mut self) -> Plan {
        if let Some(plan) = self.admit() {
            return plan;
        }
        self.decode()
    }

    fn admit(&mut self) -> Option<Plan> {
        let candidate = self.waiting.front()?;
        let needed = candidate.pending.len();
        if needed > self.cache.config().max_seq {
            // Longer than any slot can hold. Refused here rather than part-way
            // through a pass, and reported as a length finish so the client
            // hears why.
            let mut sequence = self.waiting.pop_front()?;
            sequence.finish = Some(Finish::Length);
            let id = sequence.id;
            self.running.push(sequence);
            return Some(Plan::Prefill {
                ids: vec![id],
                batch: Batch::prefill(Vec::new(), 0, 0),
            });
        }
        let Some(slot) = self.cache.acquire() else {
            self.metrics.admission_stalls += 1;
            return None;
        };

        let mut sequence = self.waiting.pop_front()?;
        sequence.slot = Some(slot);
        let batch = Batch::prefill(std::mem::take(&mut sequence.pending), slot, 0);
        let id = sequence.id;
        self.running.push(sequence);
        self.metrics.steps += 1;
        self.metrics.prefills += 1;
        Some(Plan::Prefill {
            ids: vec![id],
            batch,
        })
    }

    fn decode(&mut self) -> Plan {
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut slots = Vec::new();
        let mut starts = Vec::new();
        for sequence in &self.running {
            if ids.len() == self.max_batch {
                break;
            }
            let (Some(slot), [token]) = (sequence.slot, sequence.pending.as_slice()) else {
                continue;
            };
            if sequence.finish.is_some() || !self.cache.has_room(slot, 1) {
                continue;
            }
            ids.push(sequence.id);
            tokens.push(*token);
            slots.push(slot);
            starts.push(self.cache.length(slot));
        }
        if ids.is_empty() {
            return Plan::Idle;
        }
        self.metrics.steps += 1;
        self.metrics.decode_rows += ids.len() as u64;
        Plan::Decode {
            ids,
            batch: Batch::decode(tokens, slots, starts),
        }
    }

    /// Record that a pass ran: move the cache forward, keep its tokens, retire
    /// what it finished.
    ///
    /// The cache lengths move here rather than inside the forward pass, because
    /// this is the only point that knows the pass succeeded. `sampled` is one
    /// token per row of the plan, in the same order.
    pub fn commit(&mut self, plan: &Plan, sampled: &[u32]) -> Outcome {
        if let Some(batch) = plan.batch()
            && !batch.tokens.is_empty()
        {
            self.cache.advance(&batch.slots, batch.seq);
        }
        let mut outcome = Outcome::default();
        for (&id, &token) in plan.ids().iter().zip(sampled) {
            let Some(sequence) = self.running.iter_mut().find(|s| s.id == id) else {
                continue;
            };
            sequence.pending.clear();
            sequence.pending.push(token);
            sequence.generated += 1;
            self.metrics.tokens += 1;

            if sequence.stop_tokens.contains(&token) {
                sequence.finish = Some(Finish::Stop);
            } else {
                outcome.tokens.push((id, token));
                if sequence.generated >= sequence.max_tokens {
                    sequence.finish = Some(Finish::Length);
                }
            }
        }
        self.retire(&mut outcome);
        outcome
    }

    /// Stop a sequence early, which is what a disconnected client causes.
    pub fn cancel(&mut self, id: u64) {
        self.waiting.retain(|s| s.id != id);
        if let Some(sequence) = self.running.iter_mut().find(|s| s.id == id) {
            sequence.finish = Some(Finish::Stop);
        }
        let mut outcome = Outcome::default();
        self.retire(&mut outcome);
    }

    /// Hand every finished sequence's slot back to the pool.
    fn retire(&mut self, outcome: &mut Outcome) {
        let mut index = 0;
        while index < self.running.len() {
            let Some(finish) = self.running[index].finish else {
                index += 1;
                continue;
            };
            let sequence = self.running.remove(index);
            if let Some(slot) = sequence.slot {
                self.cache.release(slot);
            }
            outcome.finished.push((sequence.id, finish));
        }
    }

    /// Sequences that finished without a pass running, which is how a prompt
    /// too long for any slot leaves the queue.
    pub fn drain_finished(&mut self) -> Outcome {
        let mut outcome = Outcome::default();
        self.retire(&mut outcome);
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::{Plan, Scheduler, Sequence};
    use crate::batch::{CacheConfig, SlotCache};
    use crate::sampler::Sampling;
    use crate::session::Finish;
    use candle_core::{DType, Device};

    fn scheduler(slots: usize, max_seq: usize, max_batch: usize) -> Scheduler {
        let cache = SlotCache::new(
            CacheConfig {
                slots,
                max_seq,
                kv_heads: 1,
                head_dim: 2,
                layers: 1,
            },
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        Scheduler::new(cache, max_batch)
    }

    fn sequence(id: u64, prompt: usize, max_tokens: usize) -> Sequence {
        Sequence::new(
            id,
            vec![1; prompt],
            Sampling::greedy(),
            max_tokens,
            vec![99],
            id,
        )
        .unwrap()
    }

    /// Run one step, answering every row with `token`.
    fn step(scheduler: &mut Scheduler, token: u32) -> (Plan, super::Outcome) {
        let plan = scheduler.plan();
        let outcome = scheduler.commit(&plan, &vec![token; plan.ids().len()]);
        (plan, outcome)
    }

    #[test]
    fn a_prompt_is_admitted_before_anything_decodes() {
        let mut scheduler = scheduler(4, 16, 4);
        scheduler.submit(sequence(1, 3, 10));
        scheduler.submit(sequence(2, 5, 10));

        assert!(matches!(step(&mut scheduler, 7).0, Plan::Prefill { .. }));
        // The second prompt is admitted next, ahead of decoding the first.
        assert!(matches!(step(&mut scheduler, 7).0, Plan::Prefill { .. }));
        assert!(matches!(step(&mut scheduler, 7).0, Plan::Decode { .. }));
        assert_eq!(scheduler.running(), 2);
    }

    #[test]
    fn sequences_that_arrive_late_join_the_batch_that_is_already_running() {
        let mut scheduler = scheduler(4, 32, 4);
        scheduler.submit(sequence(1, 3, 20));
        step(&mut scheduler, 7);
        for _ in 0..3 {
            step(&mut scheduler, 7);
        }
        // This is the whole point of continuous batching: nothing waits for the
        // first sequence to finish.
        scheduler.submit(sequence(2, 3, 20));
        assert!(matches!(step(&mut scheduler, 7).0, Plan::Prefill { .. }));
        let (plan, _) = step(&mut scheduler, 7);
        match plan {
            Plan::Decode { ids, batch } => {
                assert_eq!(ids, vec![1, 2]);
                assert_eq!(batch.rows, 2);
                // Different lengths in one pass, which is the case the mask has
                // to handle.
                assert_ne!(batch.starts[0], batch.starts[1]);
            }
            other => panic!("expected a decode, got {other:?}"),
        }
    }

    #[test]
    fn a_sequence_waits_when_every_slot_is_held_and_starts_when_one_frees() {
        let mut scheduler = scheduler(1, 16, 4);
        scheduler.submit(sequence(1, 2, 2));
        scheduler.submit(sequence(2, 2, 2));

        // The prefill already produces a token, so a budget of two is reached
        // by the first decode rather than the second.
        step(&mut scheduler, 7);
        assert_eq!(scheduler.waiting(), 1, "the second has nowhere to go");
        // Nothing has been refused yet: the first plan found a free slot and
        // took it, and the second sequence was never reached.
        assert_eq!(scheduler.metrics().admission_stalls, 0);

        let (_, outcome) = step(&mut scheduler, 7);
        assert_eq!(
            scheduler.metrics().admission_stalls,
            1,
            "that plan tried to admit the second sequence and found no slot"
        );
        assert_eq!(outcome.finished, vec![(1, Finish::Length)]);
        assert_eq!(scheduler.waiting(), 1, "not admitted until the next plan");
        // The slot came back, so the queued sequence starts on the next pass.
        assert!(matches!(step(&mut scheduler, 7).0, Plan::Prefill { .. }));
        assert_eq!(scheduler.waiting(), 0);
    }

    #[test]
    fn a_stop_token_ends_a_sequence_and_is_not_reported_as_output() {
        let mut scheduler = scheduler(2, 16, 4);
        scheduler.submit(sequence(1, 2, 10));
        step(&mut scheduler, 7);

        let (_, outcome) = step(&mut scheduler, 99);
        assert!(outcome.tokens.is_empty(), "the stop token is not output");
        assert_eq!(outcome.finished, vec![(1, Finish::Stop)]);
        assert_eq!(scheduler.running(), 0);
    }

    #[test]
    fn the_batch_is_capped_independently_of_the_pool() {
        let mut scheduler = scheduler(8, 16, 2);
        for id in 1..=4 {
            scheduler.submit(sequence(id, 2, 20));
        }
        for _ in 0..4 {
            step(&mut scheduler, 7);
        }
        assert_eq!(scheduler.running(), 4);
        match step(&mut scheduler, 7).0 {
            Plan::Decode { ids, .. } => assert_eq!(ids.len(), 2, "capped at max_batch"),
            other => panic!("expected a decode, got {other:?}"),
        }
    }

    #[test]
    fn a_prompt_longer_than_a_slot_is_refused_rather_than_truncated() {
        let mut scheduler = scheduler(2, 4, 4);
        scheduler.submit(sequence(1, 9, 10));
        let plan = scheduler.plan();
        let outcome = scheduler.commit(&plan, &[]);
        assert_eq!(outcome.finished, vec![(1, Finish::Length)]);
        assert_eq!(scheduler.waiting(), 0);
        assert_eq!(scheduler.running(), 0);
        assert_eq!(
            scheduler.cache_mut().free_slots(),
            2,
            "no slot was consumed"
        );
    }

    #[test]
    fn cancelling_gives_the_slot_back_at_once() {
        let mut scheduler = scheduler(1, 16, 4);
        scheduler.submit(sequence(1, 2, 100));
        step(&mut scheduler, 7);
        assert_eq!(scheduler.cache_mut().free_slots(), 0);
        scheduler.cancel(1);
        assert_eq!(scheduler.cache_mut().free_slots(), 1);
        assert_eq!(scheduler.running(), 0);
    }

    #[test]
    fn an_idle_scheduler_plans_nothing() {
        let mut scheduler = scheduler(2, 16, 4);
        assert!(matches!(scheduler.plan(), Plan::Idle));
    }

    #[test]
    fn the_average_batch_counts_only_decode_passes() {
        let mut scheduler = scheduler(4, 32, 4);
        scheduler.submit(sequence(1, 2, 10));
        scheduler.submit(sequence(2, 2, 10));
        step(&mut scheduler, 7);
        step(&mut scheduler, 7);
        for _ in 0..3 {
            step(&mut scheduler, 7);
        }
        let metrics = scheduler.metrics();
        assert_eq!(metrics.prefills, 2);
        assert_eq!(metrics.steps, 5);
        // Three decode passes of two rows each, and the prefills do not count.
        assert!((metrics.mean_batch() - 2.0).abs() < 1e-9, "{metrics:?}");
    }
}
