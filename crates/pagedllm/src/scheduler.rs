//! What runs on the next forward pass.
//!
//! The scheduler decides and the engine executes, so nothing here touches a
//! tensor or a device. It hands out blocks, builds the descriptor of the next
//! batch, and takes back the tokens that came out. That split is what makes the
//! policy testable at a hundred sequences a millisecond, without a model.
//!
//! Two policies live here and a flag picks between them. Prefill-first is the
//! one vLLM shipped before chunked prefill: a step that can admit a waiting
//! sequence runs its whole prompt and nothing else, so a long prompt stalls
//! every sequence already decoding for the length of its prefill. Chunked
//! prefill cuts that prompt into slices and puts one slice in each pass beside
//! everybody's next token, which turns one long stop into several short ones
//! without removing the work. Both stay reachable, because the comparison
//! between them is the measurement.
//!
//! When the pool runs dry mid-generation a sequence is evicted rather than made
//! to wait. Waiting would deadlock: if every resident sequence needs a block and
//! none can finish without one, nothing frees anything. The sequence evicted is
//! the newest, so the one that has waited longest never pays, and it goes back
//! to the front of the queue to be recomputed from its prompt plus what it had
//! already produced.

use std::collections::VecDeque;

use crate::batch::Batch;
use crate::blocks::{BlockAllocator, BlockTable, block_hash};
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
    /// How many prompt tokens still have to pass through the model. Non-zero
    /// only while a prompt is being run a slice at a time.
    prompt_left: usize,
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
            prompt_left: 0,
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
    /// One pass carrying a slice of one prompt next to the token every running
    /// sequence is producing.
    ///
    /// `advanced` says how many tokens each sequence wrote, because an unfolded
    /// pass gives different sequences different numbers of rows and the
    /// bookkeeping cannot be read back off the batch. `ids` are the sequences a
    /// token comes back for; a prompt still in the middle of its slices is not
    /// among them, because it produces nothing until its last token has run.
    Mixed {
        /// The sequences a token comes back for, in row order.
        ids: Vec<u64>,
        /// What each sequence wrote, as (id, tokens).
        advanced: Vec<(u64, usize)>,
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
            Self::Prefill { ids, .. } | Self::Decode { ids, .. } | Self::Mixed { ids, .. } => ids,
        }
    }

    /// How many tokens each sequence wrote, as (id, tokens).
    ///
    /// A folded pass gives every row the same width, so the batch answers for
    /// all of them. An unfolded one gives different sequences different numbers
    /// of rows, so it has to carry the answer.
    fn advanced(&self) -> Vec<(u64, usize)> {
        match self {
            Self::Mixed { advanced, .. } => advanced.clone(),
            _ => match self.batch() {
                Some(batch) => self.ids().iter().map(|&id| (id, batch.seq)).collect(),
                None => Vec::new(),
            },
        }
    }

    /// The batch to run, when there is one.
    pub fn batch(&self) -> Option<&Batch> {
        match self {
            Self::Idle | Self::Refused { .. } => None,
            Self::Prefill { batch, .. }
            | Self::Decode { batch, .. }
            | Self::Mixed { batch, .. } => Some(batch),
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
    /// Passes that carried a prompt slice next to the tokens being decoded.
    pub mixed_steps: u64,
    /// Passes that carried at least one decode row. Counted rather than
    /// subtracted, because a mixed pass is a prefill and a decode at once and no
    /// arithmetic on the other counters recovers it.
    pub decode_steps: u64,
    /// Tokens re-run because their sequence was evicted, which is what
    /// preemption costs and what a benchmark should report next to its gain.
    pub recomputed_tokens: u64,
    /// Prompt tokens a sequence did not have to compute, because a block
    /// holding them was already resident.
    pub cached_tokens: u64,
    /// Prompt tokens that had to be computed.
    pub prefilled_tokens: u64,
}

impl Metrics {
    /// The share of prompt tokens answered from the cache.
    pub fn prefix_hit_rate(&self) -> f64 {
        let asked = self.cached_tokens + self.prefilled_tokens;
        if asked == 0 {
            return 0.0;
        }
        self.cached_tokens as f64 / asked as f64
    }

    /// Average rows per decode pass, which is what continuous batching buys.
    pub fn mean_batch(&self) -> f64 {
        if self.decode_steps == 0 {
            return 0.0;
        }
        self.decode_rows as f64 / self.decode_steps as f64
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
    /// Whether a sequence may start from blocks another one left behind.
    prefix_cache: bool,
    /// Most tokens one pass may carry. `None` runs a whole prompt at once,
    /// which is what stalls everything already decoding.
    chunk: Option<usize>,
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
            prefix_cache: true,
            chunk: Some(128),
            metrics: Metrics::default(),
        }
    }

    /// Most tokens one pass may carry, prompt slices and decodes together.
    ///
    /// On by default at 128, which is where `make bench-chunk` puts the knee on
    /// this machine: it cuts the worst gap between two tokens by five against
    /// running a prompt whole, and costs the arriving prompt nothing measurable.
    /// Below 64 the fixed cost of a pass starts to dominate and the newcomer's
    /// first token gets slower for no further gain. `None` turns it off, which
    /// is what every stage before this one did.
    pub fn set_chunk(&mut self, tokens: Option<usize>) {
        self.chunk = tokens.map(|t| t.max(1));
    }

    /// The pass budget, when there is one.
    pub fn chunk(&self) -> Option<usize> {
        self.chunk
    }

    /// Whether a sequence may start from blocks another one left behind.
    ///
    /// On by default, because that is what a serving engine should do, and
    /// switchable so the two paths can be compared by a flag and checked
    /// against each other for producing the same tokens.
    pub fn set_prefix_cache(&mut self, enabled: bool) {
        self.prefix_cache = enabled;
        if !enabled {
            self.pool.clear_cache();
        }
    }

    /// Whether sharing is on.
    pub fn prefix_cache(&self) -> bool {
        self.prefix_cache
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
    ///
    /// With a budget set, a slice of the waiting prompt rides in the same pass
    /// as everybody else's next token, so a long prompt costs the sequences
    /// already running one longer step rather than a full stop. Without one, a
    /// prompt takes the whole pass, which is what every stage before this did
    /// and what the comparison measures against.
    pub fn plan(&mut self) -> Plan {
        // Refused before anything else, because no arrangement of passes makes a
        // prompt fit in a pool too small to hold it.
        if let Some(plan) = self.refuse_unservable() {
            return plan;
        }
        if self.chunk.is_some() {
            return self.mixed().unwrap_or_else(|| self.decode());
        }
        if let Some(plan) = self.admit() {
            return plan;
        }
        self.decode()
    }

    /// One pass carrying a slice of a prompt and every running sequence's next
    /// token.
    ///
    /// `None` when there is no prompt to advance, which leaves the ordinary
    /// decode to run.
    fn mixed(&mut self) -> Option<Plan> {
        let budget = self.chunk?;
        // A prompt part-way through its slices finishes before a new one starts,
        // so no prompt is left half-run while another begins.
        let mut index = self
            .running
            .iter()
            .position(|s| s.prompt_left > 0 && s.finish.is_none());

        // Decodes are taken first. They are what a client is watching, and a
        // slice that crowded them out would trade one stall for another.
        let mut rows: Vec<(usize, usize)> = Vec::new();
        let mut ids = Vec::new();
        let mut predicts = Vec::new();
        let mut advanced: Vec<(u64, usize)> = Vec::new();
        let cap = self.max_batch.min(budget);
        for row in 0..self.running.len() {
            if rows.len() >= cap {
                break;
            }
            let sequence = &self.running[row];
            if Some(row) == index || sequence.finish.is_some() || sequence.pending.len() != 1 {
                continue;
            }
            if sequence.table.blocks_needed(1) > 0 {
                let Some(block) = self.pool.allocate() else {
                    // A resident needing a block from a dry pool is a
                    // preemption, and choosing the victim is the ordinary
                    // decode's job. Give the whole pass up rather than quietly
                    // running without this sequence, which would starve it for
                    // as long as the prompt takes.
                    return None;
                };
                self.running[row].table.push(block);
            }
            let sequence = &self.running[row];
            predicts.push(rows.len());
            ids.push(sequence.id);
            advanced.push((sequence.id, 1));
            rows.push((row, sequence.table.tokens()));
        }
        let decodes = rows.len();

        // Whatever the decodes left goes to the prompt.
        let room = budget.saturating_sub(decodes);
        if room == 0 {
            return None;
        }
        // A new prompt joins only now, after the residents have taken the blocks
        // they need. Admitting it first let a resident take the last block in
        // the same pass and left the newcomer in the running set with nothing to
        // run, holding the place of the next prompt until something evicted it.
        if index.is_none() {
            self.start_next_prompt();
            index = self
                .running
                .iter()
                .position(|s| s.prompt_left > 0 && s.finish.is_none());
        }
        let index = index?;
        let done = {
            let s = &self.running[index];
            s.history.len() - s.prompt_left
        };
        let want = room.min(self.running[index].prompt_left);
        for _ in 0..self.running[index].table.blocks_needed(want) {
            let Some(block) = self.pool.allocate() else {
                break;
            };
            self.running[index].table.push(block);
        }
        let sequence = &self.running[index];
        let take = want.min(sequence.table.capacity() - sequence.table.tokens());
        if take == 0 {
            return None;
        }
        let last_slice = take == sequence.prompt_left;
        if last_slice {
            predicts.push(decodes + take - 1);
            ids.push(sequence.id);
        }
        advanced.push((sequence.id, take));
        for offset in 0..take {
            rows.push((index, done + offset));
        }

        let entries: Vec<(u32, &BlockTable, usize)> = rows
            .iter()
            .map(|&(row, position)| {
                let sequence = &self.running[row];
                let token = if row == index {
                    sequence.history[position]
                } else {
                    sequence.pending[0]
                };
                (token, &sequence.table, position)
            })
            .collect();
        let batch = Batch::unfolded(&entries, &predicts, self.block_size).ok()?;

        self.metrics.steps += 1;
        self.metrics.mixed_steps += 1;
        self.metrics.decode_rows += decodes as u64;
        if decodes > 0 {
            self.metrics.decode_steps += 1;
        }
        Some(Plan::Mixed {
            ids,
            advanced,
            batch,
        })
    }

    /// Move the first waiting sequence into the running set without running it,
    /// so its prompt can be fed a slice at a time.
    fn start_next_prompt(&mut self) {
        let Some(candidate) = self.waiting.front() else {
            return;
        };
        let prompt_len = candidate.pending.len();
        if self.pool.available() == 0 {
            self.metrics.admission_stalls += 1;
            return;
        }
        let Some(mut sequence) = self.waiting.pop_front() else {
            return;
        };
        let shared = self.claim_prefix(&mut sequence);
        // Claiming a prefix takes blocks of its own, so the room checked above
        // can be gone by now, and a sequence whose whole prefix was resident
        // still needs somewhere to write the token it is about to produce.
        // Admitting it without that leaves it in the running set unable to run,
        // holding the place of the next prompt until something evicts it.
        if sequence.table.blocks_needed(1) > 0 {
            let Some(block) = self.pool.allocate() else {
                self.pool.free_table(&mut sequence.table);
                sequence.pending.clone_from(&sequence.history);
                self.waiting.push_front(sequence);
                self.metrics.admission_stalls += 1;
                return;
            };
            sequence.table.push(block);
        }
        self.metrics.cached_tokens += shared as u64;
        self.metrics.prefilled_tokens += (prompt_len - shared) as u64;
        // Every block of it may already be resident, and it still has to run its
        // last token to produce anything, so at least one token is always left.
        sequence.prompt_left = (prompt_len - shared).max(1);
        sequence.pending.clear();
        self.metrics.prefills += 1;
        self.running.push(sequence);
    }

    /// Give a waiting sequence its blocks and run its prompt.
    ///
    /// Any leading block whose contents another sequence already computed is
    /// taken rather than filled, and the prefill that follows covers only what
    /// was not found. The sequence's table then starts part-way along, which the
    /// batch already understands: it takes each row's offset from how many
    /// tokens its table holds.
    fn admit(&mut self) -> Option<Plan> {
        let prompt_len = self.waiting.front()?.pending.len();
        let needed = prompt_len.div_ceil(self.block_size);
        if needed > self.pool.available() {
            self.metrics.admission_stalls += 1;
            return None;
        }

        let mut sequence = self.waiting.pop_front()?;
        let shared = self.claim_prefix(&mut sequence);
        // Never every block: a sequence that matched its whole prompt would
        // have nothing to run, and the token it is about to produce needs a
        // position to be written at. The last block is always its own.
        for _ in 0..sequence.table.blocks_needed(prompt_len - shared) {
            let Some(block) = self.pool.allocate() else {
                // The cache gave way and there is still nothing. Put it back
                // rather than admitting it half-served.
                self.pool.free_table(&mut sequence.table);
                sequence.pending.clone_from(&sequence.history);
                self.waiting.push_front(sequence);
                self.metrics.admission_stalls += 1;
                return None;
            };
            sequence.table.push(block);
        }

        self.metrics.cached_tokens += shared as u64;
        self.metrics.prefilled_tokens += (prompt_len - shared) as u64;

        let all = std::mem::take(&mut sequence.pending);
        let tokens = all[shared..].to_vec();
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

    /// Take the blocks that already hold this prompt's leading tokens.
    ///
    /// Returns how many tokens the sequence therefore does not have to compute.
    /// The walk stops at the first block nobody has: the chain of hashes means a
    /// later block's name depends on this one, so a gap cannot be stepped over.
    fn claim_prefix(&mut self, sequence: &mut Sequence) -> usize {
        if !self.prefix_cache {
            return 0;
        }
        let prompt = &sequence.pending;
        // The last block of the prompt is left alone even when it is full, so
        // the pass has at least one token to run and the sequence owns the block
        // it is about to write into.
        let candidates = prompt.len().div_ceil(self.block_size).saturating_sub(1);

        let mut parent = None;
        let mut shared = 0usize;
        for index in 0..candidates {
            let start = index * self.block_size;
            let tokens = &prompt[start..start + self.block_size];
            let hash = block_hash(parent, tokens);
            let Some(block) = self.pool.acquire_cached(hash) else {
                self.pool.record_miss();
                break;
            };
            sequence.table.push_cached(block, hash);
            shared += self.block_size;
            parent = Some(hash);
        }
        shared
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
        self.metrics.decode_steps += 1;
        self.metrics.decode_rows += ids.len() as u64;
        Plan::Decode { ids, batch }
    }

    /// Reject a waiting prompt the pool could never hold, whatever the pass
    /// shape. Slicing it changes how much runs at once, not how much has to be
    /// resident once it has run.
    fn refuse_unservable(&mut self) -> Option<Plan> {
        let prompt_len = self.waiting.front()?.pending.len();
        if prompt_len.div_ceil(self.block_size) <= self.pool.total() {
            return None;
        }
        let mut sequence = self.waiting.pop_front()?;
        sequence.finish = Some(Finish::Length);
        self.pool.free_table(&mut sequence.table);
        Some(Plan::Refused { id: sequence.id })
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
        sequence.prompt_left = 0;
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
        for (id, written) in plan.advanced() {
            {
                let Some(index) = self.running.iter().position(|s| s.id == id) else {
                    continue;
                };
                let sequence = &mut self.running[index];
                let _ = sequence.table.advance(written);
                sequence.prompt_left = sequence.prompt_left.saturating_sub(written);
                // Name whatever this pass filled, so the sequence after it can
                // take the blocks instead of computing them again. Done here
                // rather than in the pass, because only the caller knows the
                // pass succeeded and the tokens are really there.
                if self.prefix_cache {
                    let history = sequence.history.clone();
                    for (block, hash) in self.running[index].table.newly_full(&history) {
                        self.pool.publish(block, hash);
                    }
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
        // Prefill-first, which is what `--chunk off` still selects and what the
        // chunked comparison measures against.
        let mut scheduler = scheduler(64, 4, 4);
        scheduler.set_chunk(None);
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
        scheduler.set_chunk(None);
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
        // one spare block each, so the pool empties two decodes in. Counted on
        // the folded schedule, where each prompt is exactly one pass.
        let mut scheduler = scheduler(4, 4, 50);
        scheduler.set_chunk(None);
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

    /// A prompt seen twice is computed once.
    #[test]
    fn a_second_request_with_the_same_prompt_reuses_the_blocks_of_the_first() {
        let mut scheduler = scheduler(64, 4, 4);
        scheduler.submit(sequence(1, 12, 4));
        for _ in 0..5 {
            step(&mut scheduler, 7);
        }
        assert_eq!(scheduler.running(), 0, "the first finished and let go");
        let first = scheduler.metrics();
        assert_eq!(first.cached_tokens, 0, "nothing to share on the way in");
        assert!(first.prefilled_tokens >= 12);

        scheduler.submit(sequence(2, 12, 4));
        step(&mut scheduler, 7);
        let second = scheduler.metrics();
        // Twelve tokens is three blocks of four; the last is left alone so the
        // pass has something to run, so two are taken and four tokens computed.
        assert_eq!(second.cached_tokens, 8);
        assert_eq!(second.prefilled_tokens - first.prefilled_tokens, 4);
    }

    /// Two prompts that agree for a while share exactly that much.
    #[test]
    fn sharing_stops_where_the_prompts_stop_agreeing() {
        let mut scheduler = scheduler(64, 4, 4);
        let common: Vec<u32> = (0..12).collect();

        let mut first = common.clone();
        first.extend([100, 101, 102, 103]);
        let mut second = common.clone();
        second.extend([200, 201, 202, 203]);

        let build = |id: u64, prompt: Vec<u32>| {
            Sequence::new(id, prompt, Sampling::greedy(), 2, vec![99], id).unwrap()
        };
        scheduler.submit(build(1, first));
        for _ in 0..4 {
            step(&mut scheduler, 7);
        }
        let before = scheduler.metrics().cached_tokens;

        scheduler.submit(build(2, second));
        step(&mut scheduler, 7);
        // The twelve tokens they agree on are three blocks; the fourth differs
        // and is never a candidate anyway, being the prompt's last.
        assert_eq!(scheduler.metrics().cached_tokens - before, 12);
    }

    /// A sequence still running holds its blocks, and a second one may read them.
    #[test]
    fn a_prefix_can_be_shared_while_its_first_owner_is_still_generating() {
        let mut scheduler = scheduler(64, 4, 100);
        scheduler.submit(sequence(1, 12, 50));
        for _ in 0..3 {
            step(&mut scheduler, 7);
        }
        assert_eq!(scheduler.running(), 1, "still going");

        scheduler.submit(sequence(2, 12, 50));
        step(&mut scheduler, 7);
        assert_eq!(
            scheduler.metrics().cached_tokens,
            8,
            "shared with a live sequence"
        );
        assert_eq!(scheduler.running(), 2);
    }

    /// The flag has to actually turn it off.
    #[test]
    fn nothing_is_shared_when_the_cache_is_off() {
        let mut scheduler = scheduler(64, 4, 4);
        scheduler.set_prefix_cache(false);
        assert!(!scheduler.prefix_cache());
        for id in 1..=2 {
            scheduler.submit(sequence(id, 12, 4));
            for _ in 0..5 {
                step(&mut scheduler, 7);
            }
        }
        assert_eq!(scheduler.metrics().cached_tokens, 0);
        assert!(scheduler.metrics().prefix_hit_rate() < f64::EPSILON);
    }

    /// A pool under pressure drops names rather than refusing to serve.
    #[test]
    fn cached_blocks_give_way_before_a_sequence_is_refused() {
        // Five blocks. A twelve-token prompt takes three and its first output
        // token takes a fourth, so a sequence fits, and three of them in a row
        // do not fit together with every name kept.
        let mut scheduler = scheduler(5, 4, 4);
        for id in 1..=3u64 {
            let prompt: Vec<u32> = (0..12)
                .map(|t| t + u32::try_from(id).unwrap() * 1000)
                .collect();
            scheduler
                .submit(Sequence::new(id, prompt, Sampling::greedy(), 2, vec![99], id).unwrap());
            for _ in 0..4 {
                step(&mut scheduler, 7);
            }
        }
        // Every one of them ran, none was refused, and the cache absorbed the
        // pressure by losing names.
        assert_eq!(scheduler.waiting(), 0);
        assert_eq!(scheduler.running(), 0);
        assert!(
            scheduler.pool().evictions() > 0,
            "{:?}",
            scheduler.metrics()
        );
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
        scheduler.set_chunk(None);
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

    #[test]
    fn a_long_prompt_is_fed_a_slice_at_a_time_and_no_pass_exceeds_the_budget() {
        let mut scheduler = scheduler(64, 8, 8);
        scheduler.set_chunk(Some(16));
        scheduler.submit(sequence(1, 100, 4));

        let mut slices = 0;
        let mut fed = 0;
        loop {
            let (plan, outcome) = step(&mut scheduler, 7);
            match plan {
                Plan::Mixed { batch, .. } => {
                    assert!(batch.rows <= 16, "a pass ran {} rows over 16", batch.rows);
                    slices += 1;
                    fed += batch.rows;
                }
                // The prompt is through and the sequence is decoding normally.
                Plan::Decode { .. } => break,
                other => panic!("expected a slice, got {other:?}"),
            }
            assert!(slices < 20, "the prompt never finished");
            // Nothing comes back until the last slice has run.
            if outcome.tokens.is_empty() {
                assert!(fed < 100, "the prompt ran out without producing a token");
            }
        }
        assert!(
            slices >= 6,
            "100 tokens in slices of 16 is at least 7 passes, got {slices}"
        );
        assert_eq!(fed, 100, "every prompt token runs exactly once");
    }

    #[test]
    fn a_prompt_arriving_mid_flight_does_not_stop_the_sequences_already_decoding() {
        let mut scheduler = scheduler(64, 8, 8);
        scheduler.set_chunk(Some(16));
        scheduler.submit(sequence(1, 4, 40));
        scheduler.submit(sequence(2, 4, 40));
        // Both through their prompts and decoding.
        for _ in 0..6 {
            step(&mut scheduler, 7);
        }
        assert_eq!(scheduler.running(), 2);

        scheduler.submit(sequence(3, 90, 40));
        // Every pass the newcomer's prompt takes, the two residents still get a
        // token. Without chunking the same prompt would be one pass in which
        // they get nothing.
        for pass in 0..5 {
            let (plan, outcome) = step(&mut scheduler, 7);
            assert!(matches!(plan, Plan::Mixed { .. }), "pass {pass}: {plan:?}");
            let produced: Vec<u64> = outcome.tokens.iter().map(|&(id, _)| id).collect();
            assert!(
                produced.contains(&1) && produced.contains(&2),
                "pass {pass} starved a running sequence: {produced:?}"
            );
            assert!(
                !produced.contains(&3),
                "pass {pass} answered a half-run prompt"
            );
        }
    }

    #[test]
    fn a_prompts_last_slice_is_the_one_that_produces_its_first_token() {
        let mut scheduler = scheduler(64, 8, 8);
        scheduler.set_chunk(Some(16));
        scheduler.submit(sequence(1, 40, 4));

        let mut answered = Vec::new();
        for _ in 0..4 {
            let (_, outcome) = step(&mut scheduler, 7);
            answered.push(!outcome.tokens.is_empty());
        }
        // Forty tokens in slices of sixteen is three passes, and only the third
        // asks for logits.
        assert_eq!(answered, vec![false, false, true, true], "{answered:?}");
    }

    #[test]
    fn the_budget_covers_the_decodes_and_the_slice_together() {
        let mut scheduler = scheduler(64, 8, 8);
        scheduler.set_chunk(Some(8));
        for id in 1..=3 {
            scheduler.submit(sequence(id, 2, 40));
        }
        for _ in 0..6 {
            step(&mut scheduler, 7);
        }
        assert_eq!(scheduler.running(), 3);

        scheduler.submit(sequence(9, 60, 40));
        let (plan, _) = step(&mut scheduler, 7);
        match plan {
            Plan::Mixed {
                batch, advanced, ..
            } => {
                assert_eq!(batch.rows, 8, "the pass should fill the budget exactly");
                // Three residents took one row each, so the slice got five.
                let slice = advanced.iter().find(|&&(id, _)| id == 9).map(|&(_, n)| n);
                assert_eq!(slice, Some(5), "{advanced:?}");
            }
            other => panic!("expected a mixed pass, got {other:?}"),
        }
    }

    #[test]
    fn chunking_off_gives_a_prompt_its_own_pass_and_on_does_not() {
        for (chunk, prompt_alone) in [(None, true), (Some(16), false)] {
            let mut scheduler = scheduler(64, 8, 8);
            scheduler.set_chunk(chunk);
            scheduler.submit(sequence(1, 4, 40));
            for _ in 0..3 {
                step(&mut scheduler, 7);
            }
            scheduler.submit(sequence(2, 20, 40));
            let (_, outcome) = step(&mut scheduler, 7);
            let resident_kept_going = outcome.tokens.iter().any(|&(id, _)| id == 1);
            assert_eq!(
                resident_kept_going, !prompt_alone,
                "chunk {chunk:?} put the resident in the wrong place"
            );
        }
    }

    #[test]
    fn a_dry_pool_evicts_under_chunking_too_rather_than_stalling() {
        // Six blocks of four, three sequences: the pool runs out while a fourth
        // prompt is being fed a slice at a time.
        let mut scheduler = scheduler(6, 4, 50);
        scheduler.set_chunk(Some(8));
        for id in 1..=3 {
            scheduler.submit(sequence(id, 4, 50));
        }
        for _ in 0..20 {
            step(&mut scheduler, 7);
        }
        let metrics = scheduler.metrics();
        assert!(
            metrics.preemptions > 0,
            "nothing was ever evicted: {metrics:?}"
        );
        assert!(metrics.tokens > 20, "generation stalled: {metrics:?}");
        assert!(
            scheduler.running() + scheduler.waiting() == 3,
            "a sequence was lost between the queue and the running set"
        );
    }

    #[test]
    fn a_slices_tokens_carry_the_positions_they_hold_in_the_prompt() {
        // The differential test cannot see this one: it builds its own batches,
        // so it checks that a correct set of positions gives correct logits and
        // never that the scheduler produces a correct set. Nothing else reads
        // `starts`, so nothing else would notice a slice restarting at zero.
        let mut scheduler = scheduler(64, 8, 8);
        scheduler.set_chunk(Some(16));
        scheduler.submit(sequence(1, 4, 40));
        for _ in 0..3 {
            step(&mut scheduler, 7);
        }
        assert_eq!(
            scheduler.running(),
            1,
            "the resident should still be running"
        );

        scheduler.submit(sequence(2, 40, 4));
        let mut expected = 0;
        for pass in 0..3 {
            let plan = scheduler.plan();
            let Plan::Mixed {
                batch, advanced, ..
            } = &plan
            else {
                panic!("pass {pass}: expected a mixed pass, got {plan:?}");
            };
            let slice = advanced
                .iter()
                .find(|&&(id, _)| id == 2)
                .map(|&(_, n)| n)
                .expect("the slice is booked against its sequence");
            // The resident's rows come first, then the slice's, each carrying
            // the position it holds in the prompt.
            let first = batch.rows - slice;
            for offset in 0..slice {
                assert_eq!(
                    batch.starts[first + offset],
                    expected + offset,
                    "pass {pass}, token {offset} of the slice"
                );
            }
            expected += slice;
            let ids = plan.ids().len();
            scheduler.commit(&plan, &vec![7; ids]);
        }
        assert_eq!(expected, 40, "the whole prompt ran exactly once");
    }

    #[test]
    fn a_prompt_that_is_admitted_gets_a_slice_in_the_pass_that_admitted_it() {
        // A pool tight enough that the residents' next blocks nearly empty it.
        // Admitting the newcomer before they took theirs left it in the running
        // set with no blocks and nothing to run, holding the place of the next
        // prompt until something evicted it. The counter had already moved, so
        // the engine believed it had prefilled something.
        let mut scheduler = scheduler(9, 4, 8);
        scheduler.set_chunk(Some(64));
        for id in 1..=4 {
            scheduler.submit(sequence(id, 4, 30));
        }

        let mut before = scheduler.metrics().prefills;
        for pass in 0..60 {
            let plan = scheduler.plan();
            let after = scheduler.metrics().prefills;
            if after > before {
                let carried: Vec<u64> = match &plan {
                    Plan::Mixed { advanced, .. } => advanced.iter().map(|&(id, _)| id).collect(),
                    Plan::Prefill { ids, .. } => ids.clone(),
                    other => panic!("pass {pass}: admitted on a {other:?}"),
                };
                assert!(
                    !carried.is_empty(),
                    "pass {pass}: a sequence was admitted and the pass ran nothing for it"
                );
            }
            before = after;
            let rows = plan.ids().len();
            scheduler.commit(&plan, &vec![7; rows]);
        }
        assert!(
            scheduler.metrics().prefills >= 4,
            "nothing was ever admitted: {:?}",
            scheduler.metrics()
        );
    }
}
