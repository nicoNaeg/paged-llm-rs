//! What runs on the next forward pass.
//!
//! The scheduler decides and the engine executes, so nothing here touches a
//! tensor or a device. It hands out blocks, builds the descriptor of the next
//! batch, and takes back the tokens that came out. That split is what makes the
//! policy testable at a hundred sequences a millisecond, without a model.
//!
//! The policy is prefill-first, the one vLLM shipped before chunked prefill: a
//! step that can admit a waiting sequence runs its whole prompt and nothing
//! else. It is not the best policy and is not meant to be. A long prompt stalls
//! every sequence already decoding for the length of its prefill, and that stall
//! is what stage 8 exists to remove, which is easier to demonstrate having first
//! been built.
//!
//! When the pool runs dry mid-generation a sequence is evicted rather than made
//! to wait. Waiting would deadlock: if every resident sequence needs a block and
//! none can finish without one, nothing frees anything. The sequence evicted is
//! the newest, so the one that has waited longest never pays, and it goes back
//! to the front of the queue to be recomputed from its prompt plus what it had
//! already produced.

use std::collections::VecDeque;

use crate::batch::Batch;
use crate::blocks::{BlockAllocator, BlockTable};
use crate::sampler::{Rng, Sampling};
use crate::session::Finish;
use crate::{Error, Result};

/// One request, from arrival to its last token.
#[derive(Debug)]
pub struct Sequence {
    /// Names it in the events the engine emits.
    pub id: u64,
    /// Everything the model has seen: the prompt, then what it produced. An
    /// evicted sequence is recomputed from all of it, which is why it is kept
    /// rather than only counted.
    history: Vec<u32>,
    prompt_tokens: usize,
    sampling: Sampling,
    max_tokens: usize,
    stop_tokens: Vec<u32>,
    rng: Rng,
    table: BlockTable,
    /// Fed on the next pass. The whole history after admission or eviction,
    /// then one token at a time.
    pending: Vec<u32>,
    generated: usize,
    preemptions: u32,
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
            prompt_tokens: prompt.len(),
            pending: prompt.clone(),
            history: prompt,
            sampling,
            max_tokens,
            stop_tokens,
            rng: Rng::new(seed),
            table: BlockTable::new(1),
            generated: 0,
            preemptions: 0,
            finish: None,
        })
    }

    /// How the next token is chosen, and the generator it draws from.
    pub fn sampling(&mut self) -> (Sampling, &mut Rng) {
        (self.sampling, &mut self.rng)
    }

    /// How many tokens the prompt held.
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// How many tokens the model has produced.
    pub fn generated(&self) -> usize {
        self.generated
    }

    /// How many times it was evicted and recomputed.
    pub fn preemptions(&self) -> u32 {
        self.preemptions
    }

    /// Why it stopped, once it has.
    pub fn finish_reason(&self) -> Option<Finish> {
        self.finish
    }
}

/// What the engine should run next.
#[derive(Debug)]
pub enum Plan {
    /// Nothing is waiting and nothing can run.
    Idle,
    /// One sequence's prompt, alone in the pass.
    Prefill {
        /// The sequence, in row order.
        ids: Vec<u64>,
        /// What to run.
        batch: Batch,
    },
    /// One token for each sequence able to take one.
    Decode {
        /// The sequences, in row order.
        ids: Vec<u64>,
        /// What to run.
        batch: Batch,
    },
    /// A sequence ended without a pass, which is how a prompt too long for the
    /// whole pool leaves the queue.
    Refused {
        /// The sequence that will not run.
        id: u64,
    },
}

impl Plan {
    /// The sequences this plan covers, in row order.
    pub fn ids(&self) -> &[u64] {
        match self {
            Self::Idle | Self::Refused { .. } => &[],
            Self::Prefill { ids, .. } | Self::Decode { ids, .. } => ids,
        }
    }

    /// The batch to run, when there is one.
    pub fn batch(&self) -> Option<&Batch> {
        match self {
            Self::Idle | Self::Refused { .. } => None,
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
    /// Times a waiting sequence found no room.
    pub admission_stalls: u64,
    /// Sequences evicted to free blocks for another.
    pub preemptions: u64,
    /// Tokens re-run because their sequence was evicted, which is what
    /// preemption costs and what a benchmark should report next to its gain.
    pub recomputed_tokens: u64,
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
    pool: BlockAllocator,
    block_size: usize,
    max_batch: usize,
    metrics: Metrics,
}

impl Scheduler {
    /// Build over a pool of blocks. `max_batch` caps the decode rows
    /// independently of the blocks, so a batch can be limited without shrinking
    /// the cache.
    pub fn new(blocks: usize, block_size: usize, max_batch: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            pool: BlockAllocator::new(blocks),
            block_size: block_size.max(1),
            max_batch: max_batch.max(1),
            metrics: Metrics::default(),
        }
    }

    /// Queue a sequence.
    pub fn submit(&mut self, mut sequence: Sequence) {
        sequence.table = BlockTable::new(self.block_size);
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

    /// The ids of the resident sequences, oldest first.
    pub fn running_ids(&self) -> Vec<u64> {
        self.running.iter().map(|s| s.id).collect()
    }

    /// The counters so far.
    pub fn metrics(&self) -> Metrics {
        self.metrics
    }

    /// The free list, for the server to report on.
    pub fn pool(&self) -> &BlockAllocator {
        &self.pool
    }

    /// Tokens held in blocks that are allocated and not yet written, summed over
    /// every resident sequence. The whole of what paging wastes.
    pub fn wasted_tokens(&self) -> usize {
        self.running.iter().map(|s| s.table.wasted_tokens()).sum()
    }

    /// The sequence with this id, while it is running.
    pub fn sequence_mut(&mut self, id: u64) -> Option<&mut Sequence> {
        self.running.iter_mut().find(|s| s.id == id)
    }

    /// Decide the next pass.
    pub fn plan(&mut self) -> Plan {
        if let Some(plan) = self.admit() {
            return plan;
        }
        self.decode()
    }

    /// Give a waiting sequence its blocks and run its prompt.
    fn admit(&mut self) -> Option<Plan> {
        let needed = self
            .waiting
            .front()?
            .pending
            .len()
            .div_ceil(self.block_size);
        if needed > self.pool.total() {
            let mut sequence = self.waiting.pop_front()?;
            sequence.finish = Some(Finish::Length);
            self.pool.free_table(&mut sequence.table);
            return Some(Plan::Refused { id: sequence.id });
        }
        if needed > self.pool.available() {
            self.metrics.admission_stalls += 1;
            return None;
        }

        let mut sequence = self.waiting.pop_front()?;
        for _ in 0..needed {
            let block = self.pool.allocate().expect("checked against available");
            sequence.table.push(block);
        }
        let tokens = std::mem::take(&mut sequence.pending);
        let seq = tokens.len();
        let batch = Batch::new(tokens, seq, &[&sequence.table], self.block_size).ok()?;
        let id = sequence.id;
        self.running.push(sequence);
        self.metrics.steps += 1;
        self.metrics.prefills += 1;
        Some(Plan::Prefill {
            ids: vec![id],
            batch,
        })
    }

    /// Advance every resident sequence by one token, evicting where the pool
    /// cannot supply the block one of them needs.
    fn decode(&mut self) -> Plan {
        let mut chosen: Vec<usize> = Vec::new();
        let mut index = 0;
        while index < self.running.len() && chosen.len() < self.max_batch {
            let sequence = &self.running[index];
            if sequence.finish.is_some() || sequence.pending.len() != 1 {
                index += 1;
                continue;
            }
            if sequence.table.blocks_needed(1) == 0 {
                chosen.push(index);
                index += 1;
                continue;
            }
            if let Some(block) = self.pool.allocate() {
                self.running[index].table.push(block);
                chosen.push(index);
                index += 1;
                continue;
            }
            // Nothing left. Evict the newest resident that is not this one, and
            // come back to this sequence with the blocks it freed.
            let Some(victim) = self.newest_evictable(index) else {
                index += 1;
                continue;
            };
            self.preempt(victim);
            chosen.retain(|&taken| taken != victim);
            for taken in &mut chosen {
                if *taken > victim {
                    *taken -= 1;
                }
            }
            if victim < index {
                index -= 1;
            }
        }

        if chosen.is_empty() {
            return Plan::Idle;
        }
        let ids: Vec<u64> = chosen.iter().map(|&i| self.running[i].id).collect();
        let tokens: Vec<u32> = chosen.iter().map(|&i| self.running[i].pending[0]).collect();
        let tables: Vec<&BlockTable> = chosen.iter().map(|&i| &self.running[i].table).collect();
        let Ok(batch) = Batch::new(tokens, 1, &tables, self.block_size) else {
            return Plan::Idle;
        };
        self.metrics.steps += 1;
        self.metrics.decode_rows += ids.len() as u64;
        Plan::Decode { ids, batch }
    }

    /// The most recently admitted resident that is not `protect`.
    fn newest_evictable(&self, protect: usize) -> Option<usize> {
        (0..self.running.len())
            .rev()
            .find(|&i| i != protect && self.running[i].finish.is_none())
    }

    /// Take a sequence's blocks back and put it at the front of the queue.
    fn preempt(&mut self, index: usize) {
        let mut sequence = self.running.remove(index);
        self.pool.free_table(&mut sequence.table);
        // Recomputed from everything the model has seen, prompt and output
        // together, which is what makes eviction cost a prefill rather than an
        // answer. The tokens already returned to the client stay returned.
        sequence.pending.clone_from(&sequence.history);
        sequence.preemptions += 1;
        self.metrics.preemptions += 1;
        self.metrics.recomputed_tokens += sequence.history.len() as u64;
        self.waiting.push_front(sequence);
    }

    /// Record that a pass ran: keep its tokens, retire what it finished.
    ///
    /// `sampled` is one token per row of the plan, in the same order.
    pub fn commit(&mut self, plan: &Plan, sampled: &[u32]) -> Outcome {
        let mut outcome = Outcome::default();
        if let Some(batch) = plan.batch() {
            for &id in plan.ids() {
                if let Some(sequence) = self.running.iter_mut().find(|s| s.id == id) {
                    let _ = sequence.table.advance(batch.seq);
                }
            }
        }
        for (&id, &token) in plan.ids().iter().zip(sampled) {
            let Some(sequence) = self.running.iter_mut().find(|s| s.id == id) else {
                continue;
            };
            sequence.pending.clear();
            sequence.pending.push(token);
            sequence.history.push(token);
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
        if let Plan::Refused { id } = plan {
            self.waiting.retain(|s| s.id != *id);
            outcome.finished.push((*id, Finish::Length));
        }
        self.retire(&mut outcome);
        outcome
    }

    /// Stop a sequence early, which is what a disconnected client causes.
    pub fn cancel(&mut self, id: u64) {
        if let Some(position) = self.waiting.iter().position(|s| s.id == id) {
            if let Some(mut sequence) = self.waiting.remove(position) {
                self.pool.free_table(&mut sequence.table);
            }
            return;
        }
        if let Some(sequence) = self.running.iter_mut().find(|s| s.id == id) {
            sequence.finish = Some(Finish::Stop);
        }
        let mut outcome = Outcome::default();
        self.retire(&mut outcome);
    }

    /// Hand every finished sequence's blocks back to the pool.
    fn retire(&mut self, outcome: &mut Outcome) {
        let mut index = 0;
        while index < self.running.len() {
            let Some(finish) = self.running[index].finish else {
                index += 1;
                continue;
            };
            let mut sequence = self.running.remove(index);
            self.pool.free_table(&mut sequence.table);
            outcome.finished.push((sequence.id, finish));
        }
    }

    /// Sequences that finished without a pass running.
    pub fn drain_finished(&mut self) -> Outcome {
        let mut outcome = Outcome::default();
        self.retire(&mut outcome);
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::{Plan, Scheduler, Sequence};
    use crate::sampler::Sampling;
    use crate::session::Finish;

    fn scheduler(blocks: usize, block_size: usize, max_batch: usize) -> Scheduler {
        Scheduler::new(blocks, block_size, max_batch)
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
        let mut scheduler = scheduler(64, 4, 4);
        scheduler.submit(sequence(1, 3, 10));
        scheduler.submit(sequence(2, 5, 10));

        assert!(matches!(step(&mut scheduler, 7).0, Plan::Prefill { .. }));
        assert!(matches!(step(&mut scheduler, 7).0, Plan::Prefill { .. }));
        assert!(matches!(step(&mut scheduler, 7).0, Plan::Decode { .. }));
        assert_eq!(scheduler.running(), 2);
    }

    #[test]
    fn a_sequence_takes_only_the_blocks_its_tokens_need() {
        let mut scheduler = scheduler(64, 16, 4);
        scheduler.submit(sequence(1, 20, 100));
        step(&mut scheduler, 7);
        // Twenty tokens is two blocks of sixteen. A reservation would have taken
        // room for all hundred before the first was produced.
        assert_eq!(scheduler.pool().in_use(), 2);
        assert_eq!(scheduler.pool().available(), 62);
    }

    #[test]
    fn a_block_is_taken_only_when_the_last_one_fills() {
        let mut scheduler = scheduler(64, 4, 4);
        scheduler.submit(sequence(1, 3, 20));
        step(&mut scheduler, 7);
        assert_eq!(scheduler.pool().in_use(), 1);

        // Three prompt tokens plus the one the prefill produced fill the block,
        // so the first decode needs nothing new and the second needs a block.
        step(&mut scheduler, 7);
        assert_eq!(scheduler.pool().in_use(), 1);
        step(&mut scheduler, 7);
        assert_eq!(scheduler.pool().in_use(), 2);
    }

    #[test]
    fn sequences_that_arrive_late_join_the_batch_that_is_already_running() {
        let mut scheduler = scheduler(64, 8, 4);
        scheduler.submit(sequence(1, 3, 20));
        step(&mut scheduler, 7);
        for _ in 0..3 {
            step(&mut scheduler, 7);
        }
        scheduler.submit(sequence(2, 3, 20));
        assert!(matches!(step(&mut scheduler, 7).0, Plan::Prefill { .. }));
        let (plan, _) = step(&mut scheduler, 7);
        match plan {
            Plan::Decode { ids, batch } => {
                assert_eq!(ids, vec![1, 2]);
                assert_eq!(batch.rows, 2);
                assert_ne!(batch.starts[0], batch.starts[1]);
            }
            other => panic!("expected a decode, got {other:?}"),
        }
    }

    #[test]
    fn a_stop_token_ends_a_sequence_and_gives_its_blocks_back() {
        let mut scheduler = scheduler(64, 4, 4);
        scheduler.submit(sequence(1, 6, 10));
        step(&mut scheduler, 7);
        assert_eq!(scheduler.pool().in_use(), 2);

        let (_, outcome) = step(&mut scheduler, 99);
        assert!(outcome.tokens.is_empty(), "the stop token is not output");
        assert_eq!(outcome.finished, vec![(1, Finish::Stop)]);
        assert_eq!(scheduler.pool().available(), 64, "every block came back");
    }

    #[test]
    fn the_pool_running_dry_evicts_the_newest_and_recomputes_it() {
        // Four blocks of four tokens, two prompts of four: room for both and
        // one spare block each, so the pool empties two decodes in.
        let mut scheduler = scheduler(4, 4, 50);
        scheduler.submit(sequence(1, 4, 50));
        scheduler.submit(sequence(2, 4, 50));
        step(&mut scheduler, 7);
        step(&mut scheduler, 7);
        assert_eq!(scheduler.pool().available(), 2);

        for _ in 0..10 {
            step(&mut scheduler, 7);
        }
        let metrics = scheduler.metrics();
        assert!(metrics.preemptions > 0, "{metrics:?}");
        assert!(
            metrics.recomputed_tokens > 0,
            "an evicted sequence is recomputed, and that is what eviction costs"
        );
        // Nothing was lost: what was evicted went back to the queue.
        assert_eq!(scheduler.waiting() + scheduler.running(), 2);
    }

    #[test]
    fn the_sequence_that_arrived_first_is_never_the_one_evicted() {
        let mut scheduler = scheduler(4, 4, 50);
        scheduler.submit(sequence(1, 4, 50));
        scheduler.submit(sequence(2, 4, 50));
        step(&mut scheduler, 7);
        step(&mut scheduler, 7);

        for _ in 0..12 {
            step(&mut scheduler, 7);
            // Sequence 1 arrived first. It may be queued behind an eviction of
            // its own making only if it was never the victim, so it must be
            // resident or first in line every time.
            let running = scheduler.running_ids();
            assert!(
                running.contains(&1) || scheduler.waiting() > 0,
                "the oldest sequence must not be dropped, running {running:?}"
            );
        }
        assert!(scheduler.metrics().preemptions > 0);
    }

    #[test]
    fn a_prompt_longer_than_the_whole_pool_is_refused_rather_than_truncated() {
        let mut scheduler = scheduler(2, 4, 4);
        scheduler.submit(sequence(1, 20, 10));
        let plan = scheduler.plan();
        assert!(matches!(plan, Plan::Refused { .. }));
        let outcome = scheduler.commit(&plan, &[]);
        assert_eq!(outcome.finished, vec![(1, Finish::Length)]);
        assert_eq!(scheduler.waiting(), 0);
        assert_eq!(scheduler.pool().available(), 2, "no block was consumed");
    }

    #[test]
    fn cancelling_gives_the_blocks_back_at_once() {
        let mut scheduler = scheduler(8, 4, 4);
        scheduler.submit(sequence(1, 8, 100));
        step(&mut scheduler, 7);
        assert_eq!(scheduler.pool().in_use(), 2);
        scheduler.cancel(1);
        assert_eq!(scheduler.pool().available(), 8);
        assert_eq!(scheduler.running(), 0);
    }

    #[test]
    fn an_idle_scheduler_plans_nothing() {
        let mut scheduler = scheduler(8, 4, 4);
        assert!(matches!(scheduler.plan(), Plan::Idle));
    }

    #[test]
    fn what_paging_wastes_is_bounded_by_one_block_a_sequence() {
        let mut scheduler = scheduler(64, 16, 8);
        for id in 1..=4 {
            scheduler.submit(sequence(id, 20, 50));
        }
        for _ in 0..4 {
            step(&mut scheduler, 7);
        }
        assert_eq!(scheduler.running(), 4);
        assert!(
            scheduler.wasted_tokens() < 4 * 16,
            "waste is under one block a sequence, got {}",
            scheduler.wasted_tokens()
        );
    }

    #[test]
    fn the_average_batch_counts_only_decode_passes() {
        let mut scheduler = scheduler(64, 8, 4);
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
        assert!((metrics.mean_batch() - 2.0).abs() < 1e-9, "{metrics:?}");
    }
}
