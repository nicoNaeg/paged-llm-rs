//! A KV cache shared by several sequences, and the batch that reads it.
//!
//! One tensor per layer, `[slots, kv_heads, max_seq, head_dim]`, allocated once.
//! A sequence holds a slot for its whole life and writes into it in place. This
//! is the layout `PagedAttention` replaces, and it is built here first so the
//! replacement has something measured to beat.
//!
//! What it costs is visible in the shape of that tensor. Every slot reserves
//! `max_seq` tokens whatever the sequence turns out to need, so the number of
//! sequences that fit is decided before any of them arrive. On Qwen3-0.6B a
//! token of cache is 112 KiB across the 28 layers, which makes a 2048-token
//! reservation 229 MiB per slot, held whether the request stops after thirty
//! tokens or runs to the end.
//!
//! It costs a second thing, less obvious and measured rather than assumed.
//! Sequences in a batch are different lengths, so attention reads a rectangle
//! that no slot fills. Narrowing the reservation down to the longest sequence in
//! the batch makes that rectangle smaller but leaves it non-contiguous, which
//! forces a copy before the multiply. Reading it whole avoids the copy and
//! computes over the padding instead. There is no third option while a
//! sequence's cache has to be one unbroken run of memory, and removing that
//! constraint is what stage 5 is.

use candle_core::{DType, Device, Tensor};

use crate::{Error, Result};

/// How the shared cache is sized.
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// How many sequences can be resident at once.
    pub slots: usize,
    /// Tokens reserved for each of them.
    pub max_seq: usize,
    /// Key and value heads, from the model.
    pub kv_heads: usize,
    /// Width of one head, from the model.
    pub head_dim: usize,
    /// Layers, from the model.
    pub layers: usize,
}

impl CacheConfig {
    /// Bytes one token of cache costs, across every layer.
    pub fn bytes_per_token(&self, dtype: DType) -> usize {
        // Keys and values, per layer, per head.
        2 * self.layers * self.kv_heads * self.head_dim * dtype.size_in_bytes()
    }

    /// Bytes the whole pool costs.
    pub fn bytes(&self, dtype: DType) -> usize {
        self.slots * self.max_seq * self.bytes_per_token(dtype)
    }

    /// The most slots that fit in `budget` bytes, at least one.
    pub fn slots_within(&self, budget: usize, dtype: DType) -> usize {
        let per_slot = self.max_seq * self.bytes_per_token(dtype);
        (budget / per_slot.max(1)).max(1)
    }
}

/// One forward pass's worth of work.
///
/// Every row carries the same number of tokens, which is what makes a batch a
/// rectangle. Decoding is many rows of one token; a prefill is one row of many.
/// Mixing the two in a single pass is what stage 8 adds, and its absence is why
/// a long prompt arriving today stalls every sequence already decoding.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Token ids, row-major, `rows * seq` of them.
    pub tokens: Vec<u32>,
    /// The cache slot each row reads and writes.
    pub slots: Vec<usize>,
    /// Tokens already in each slot before this pass, which is also where the
    /// row's first token sits.
    pub starts: Vec<usize>,
    /// How many rows.
    pub rows: usize,
    /// How many tokens per row.
    pub seq: usize,
}

impl Batch {
    /// One sequence's prompt, filling a slot from `start`.
    pub fn prefill(tokens: Vec<u32>, slot: usize, start: usize) -> Self {
        let seq = tokens.len();
        Self {
            tokens,
            slots: vec![slot],
            starts: vec![start],
            rows: 1,
            seq,
        }
    }

    /// One token for each of several sequences.
    pub fn decode(tokens: Vec<u32>, slots: Vec<usize>, starts: Vec<usize>) -> Self {
        let rows = tokens.len();
        Self {
            tokens,
            slots,
            starts,
            rows,
            seq: 1,
        }
    }

    /// The furthest any row reaches once this pass is written.
    pub fn longest(&self) -> usize {
        self.starts.iter().map(|s| s + self.seq).max().unwrap_or(0)
    }

    /// Check the row counts agree before anything is dispatched.
    pub fn validate(&self) -> Result<()> {
        if self.rows == 0 || self.seq == 0 {
            return Err(Error::Config("an empty batch".into()));
        }
        if self.tokens.len() != self.rows * self.seq {
            return Err(Error::Config(format!(
                "{} tokens for {} rows of {}",
                self.tokens.len(),
                self.rows,
                self.seq
            )));
        }
        if self.slots.len() != self.rows || self.starts.len() != self.rows {
            return Err(Error::Config(format!(
                "{} rows against {} slots and {} offsets",
                self.rows,
                self.slots.len(),
                self.starts.len()
            )));
        }
        Ok(())
    }

    /// Absolute position of every token, `[rows, seq]`.
    pub(crate) fn positions(&self, device: &Device) -> Result<Tensor> {
        let mut positions = Vec::with_capacity(self.rows * self.seq);
        for &start in &self.starts {
            for offset in 0..self.seq {
                positions.push(u32::try_from(start + offset).unwrap_or(u32::MAX));
            }
        }
        Ok(Tensor::from_vec(positions, (self.rows, self.seq), device)?)
    }

    /// Additive mask, `[rows, 1, seq, longest]`.
    ///
    /// Two things at once, which is why it is not the single-sequence mask with
    /// a batch dimension bolted on. It forbids a query from reading a key ahead
    /// of it, as always; and it forbids every row from reading the part of the
    /// rectangle that belongs to a longer sequence than its own, which is the
    /// padding a batch of unequal lengths cannot avoid.
    pub(crate) fn mask(&self, dtype: DType, device: &Device) -> Result<Tensor> {
        let longest = self.longest();
        let mut mask = Vec::with_capacity(self.rows * self.seq * longest);
        for &start in &self.starts {
            for offset in 0..self.seq {
                let visible = start + offset;
                for key in 0..longest {
                    mask.push(if key <= visible {
                        0f32
                    } else {
                        f32::NEG_INFINITY
                    });
                }
            }
        }
        // Every row keeps at least its own position, so no row is entirely
        // masked and the subtraction inside the softmax never sees infinity
        // minus itself.
        Ok(Tensor::from_vec(mask, (self.rows, 1, self.seq, longest), device)?.to_dtype(dtype)?)
    }
}

/// The keys and values of every resident sequence.
#[derive(Debug)]
pub struct SlotCache {
    /// Per layer, `[slots, kv_heads, max_seq, head_dim]`.
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    config: CacheConfig,
    /// Tokens written into each slot.
    lengths: Vec<usize>,
    free: Vec<usize>,
}

impl SlotCache {
    /// Allocate the pool. This is the whole allocation: nothing here grows.
    pub fn new(config: CacheConfig, dtype: DType, device: &Device) -> Result<Self> {
        let shape = (
            config.slots,
            config.kv_heads,
            config.max_seq,
            config.head_dim,
        );
        let mut keys = Vec::with_capacity(config.layers);
        let mut values = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            keys.push(Tensor::zeros(shape, dtype, device)?);
            values.push(Tensor::zeros(shape, dtype, device)?);
        }
        Ok(Self {
            keys,
            values,
            config,
            lengths: vec![0; config.slots],
            // Handed out from the end so the first sequence takes slot 0, which
            // makes a trace easier to read and changes nothing else.
            free: (0..config.slots).rev().collect(),
        })
    }

    /// How the pool is shaped.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Take a slot, or `None` when every one is held.
    pub fn acquire(&mut self) -> Option<usize> {
        let slot = self.free.pop()?;
        self.lengths[slot] = 0;
        Some(slot)
    }

    /// Give a slot back. Its contents are not cleared: nothing reads past a
    /// slot's length, and the next sequence to hold it starts at zero.
    pub fn release(&mut self, slot: usize) {
        if slot < self.config.slots && !self.free.contains(&slot) {
            self.lengths[slot] = 0;
            self.free.push(slot);
        }
    }

    /// How many slots are free.
    pub fn free_slots(&self) -> usize {
        self.free.len()
    }

    /// Tokens written into `slot`.
    pub fn length(&self, slot: usize) -> usize {
        self.lengths.get(slot).copied().unwrap_or(0)
    }

    /// Whether `slot` can take `count` more tokens.
    pub fn has_room(&self, slot: usize, count: usize) -> bool {
        self.length(slot) + count <= self.config.max_seq
    }

    /// Write one batch's keys and values into their slots, in place.
    ///
    /// `k` and `v` are `[rows, kv_heads, seq, head_dim]` and `slots` names the
    /// slot each row belongs to. Nothing is reallocated: this is the property
    /// the pre-allocated pool exists for, and the one the stage 2 cache did not
    /// have.
    pub(crate) fn write(
        &self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
        slots: &[usize],
        starts: &[usize],
    ) -> Result<()> {
        let (rows, _, seq, _) = k.dims4()?;
        if rows != slots.len() || rows != starts.len() {
            return Err(Error::Config(format!(
                "{rows} rows against {} slots and {} offsets",
                slots.len(),
                starts.len()
            )));
        }
        let (keys, values) = (&self.keys[layer], &self.values[layer]);
        for (row, (&slot, &start)) in slots.iter().zip(starts).enumerate() {
            if start + seq > self.config.max_seq {
                return Err(Error::Config(format!(
                    "slot {slot} would reach {} of {} reserved tokens",
                    start + seq,
                    self.config.max_seq
                )));
            }
            keys.narrow(0, slot, 1)?
                .slice_set(&k.narrow(0, row, 1)?.contiguous()?, 2, start)?;
            values
                .narrow(0, slot, 1)?
                .slice_set(&v.narrow(0, row, 1)?.contiguous()?, 2, start)?;
        }
        Ok(())
    }

    /// Write one token into every slot, and nothing else.
    ///
    /// Exists to be timed: it is the part of a decode step whose dispatch count
    /// grows with the batch, where everything else grows with its arithmetic.
    /// Nothing in the serving path calls it.
    pub fn write_probe(&self, slots: &[usize], starts: &[usize]) -> Result<()> {
        let shape = (slots.len(), self.config.kv_heads, 1, self.config.head_dim);
        let dtype = self.keys[0].dtype();
        let k = Tensor::zeros(shape, dtype, self.keys[0].device())?;
        for layer in 0..self.config.layers {
            self.write(layer, &k, &k, slots, starts)?;
        }
        Ok(())
    }

    /// Record that every slot in `slots` grew by `count` tokens.
    ///
    /// Called by whoever ran the pass, not by the pass itself. A forward pass
    /// that failed part way through must not leave the bookkeeping claiming it
    /// succeeded, and only the caller knows which happened.
    pub fn advance(&mut self, slots: &[usize], count: usize) {
        for &slot in slots {
            self.lengths[slot] += count;
        }
    }

    /// The keys and values the attention should read for `slots`, narrowed to
    /// the longest of them.
    ///
    /// Returns `[rows, kv_heads, longest, head_dim]`. The narrowing is what
    /// keeps the multiply off the unused end of every reservation, and the copy
    /// it forces is the cost named in this module's header.
    pub(crate) fn read(
        &self,
        layer: usize,
        slots: &[usize],
        longest: usize,
    ) -> Result<(Tensor, Tensor)> {
        // A run of consecutive slots is one narrow, so the whole rectangle
        // reaches the multiply in a single copy. Anything else has to be
        // gathered row by row, and that gather is one dispatch per row per
        // layer, twice, which is what makes a decode step cost the same again
        // for every sequence added to it.
        let consecutive = slots.windows(2).all(|pair| pair[1] == pair[0] + 1);
        let gather = |pool: &Tensor| -> Result<Tensor> {
            if consecutive {
                return Ok(pool
                    .narrow(0, slots[0], slots.len())?
                    .narrow(2, 0, longest)?
                    .contiguous()?);
            }
            let rows: Result<Vec<Tensor>> = slots
                .iter()
                .map(|&slot| Ok(pool.narrow(0, slot, 1)?.narrow(2, 0, longest)?))
                .collect();
            Ok(Tensor::cat(&rows?, 0)?.contiguous()?)
        };
        Ok((gather(&self.keys[layer])?, gather(&self.values[layer])?))
    }
}

#[cfg(test)]
mod tests {
    use super::{CacheConfig, SlotCache};
    use candle_core::{DType, Device, Tensor};

    fn config() -> CacheConfig {
        CacheConfig {
            slots: 3,
            max_seq: 8,
            kv_heads: 2,
            head_dim: 4,
            layers: 2,
        }
    }

    fn step(value: f32, tokens: usize) -> Tensor {
        Tensor::full(value, (1, 2, tokens, 4), &Device::Cpu).unwrap()
    }

    #[test]
    fn a_token_of_cache_costs_what_the_arithmetic_says() {
        // Qwen3-0.6B: 28 layers, 8 key heads, 128 wide, in bf16.
        let qwen = CacheConfig {
            slots: 1,
            max_seq: 2048,
            kv_heads: 8,
            head_dim: 128,
            layers: 28,
        };
        assert_eq!(qwen.bytes_per_token(DType::BF16), 114_688);
        assert_eq!(qwen.bytes(DType::BF16), 2048 * 114_688);
        // Eight gigabytes buys this many sequences, whatever they turn out to
        // need, which is the number stage 5 has to raise.
        assert_eq!(qwen.slots_within(8 << 30, DType::BF16), 36);
    }

    #[test]
    fn slots_are_handed_out_and_taken_back() {
        let mut cache = SlotCache::new(config(), DType::F32, &Device::Cpu).unwrap();
        let held: Vec<usize> = (0..3).map(|_| cache.acquire().unwrap()).collect();
        assert_eq!(held, vec![0, 1, 2]);
        assert_eq!(cache.acquire(), None, "a fourth sequence has nowhere to go");
        cache.release(1);
        assert_eq!(cache.acquire(), Some(1));
    }

    #[test]
    fn releasing_a_slot_twice_does_not_hand_it_out_twice() {
        let mut cache = SlotCache::new(config(), DType::F32, &Device::Cpu).unwrap();
        let slot = cache.acquire().unwrap();
        cache.release(slot);
        cache.release(slot);
        assert_eq!(cache.free_slots(), 3);
    }

    #[test]
    fn what_a_slot_holds_is_what_was_written_to_it() {
        let mut cache = SlotCache::new(config(), DType::F32, &Device::Cpu).unwrap();
        let (a, b) = (cache.acquire().unwrap(), cache.acquire().unwrap());

        cache
            .write(0, &step(1.0, 3), &step(-1.0, 3), &[a], &[0])
            .unwrap();
        cache.advance(&[a], 3);
        cache
            .write(0, &step(2.0, 2), &step(-2.0, 2), &[b], &[0])
            .unwrap();
        cache.advance(&[b], 2);
        // One more token on the first sequence, written past what it already
        // holds rather than over it.
        cache
            .write(0, &step(3.0, 1), &step(-3.0, 1), &[a], &[3])
            .unwrap();
        cache.advance(&[a], 1);

        assert_eq!(cache.length(a), 4);
        assert_eq!(cache.length(b), 2);

        let (keys, values) = cache.read(0, &[a, b], 4).unwrap();
        assert_eq!(keys.dims(), &[2, 2, 4, 4]);
        let first: Vec<f32> = keys
            .narrow(0, 0, 1)
            .unwrap()
            .narrow(1, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(&first[..4], &[1.0, 1.0, 1.0, 1.0], "first token of slot a");
        assert_eq!(
            &first[12..],
            &[3.0, 3.0, 3.0, 3.0],
            "fourth token of slot a"
        );

        // The second slot is untouched past its own two tokens, and the first
        // slot's writes did not reach it.
        let second: Vec<f32> = values
            .narrow(0, 1, 1)
            .unwrap()
            .narrow(1, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(&second[..8], &[-2.0; 8]);
        assert_eq!(
            &second[8..],
            &[0.0; 8],
            "past the length, nothing was written"
        );
    }

    #[test]
    fn a_write_past_the_reservation_is_refused_rather_than_wrapping() {
        let mut cache = SlotCache::new(config(), DType::F32, &Device::Cpu).unwrap();
        let slot = cache.acquire().unwrap();
        assert!(cache.has_room(slot, 8));
        assert!(
            cache
                .write(0, &step(1.0, 8), &step(1.0, 8), &[slot], &[0])
                .is_ok()
        );
        cache.advance(&[slot], 8);
        assert!(!cache.has_room(slot, 1), "the reservation is full");
        assert!(
            cache
                .write(0, &step(1.0, 1), &step(1.0, 1), &[slot], &[8])
                .is_err()
        );
    }

    #[test]
    fn layers_do_not_see_each_others_writes() {
        let mut cache = SlotCache::new(config(), DType::F32, &Device::Cpu).unwrap();
        let slot = cache.acquire().unwrap();
        cache
            .write(0, &step(5.0, 1), &step(5.0, 1), &[slot], &[0])
            .unwrap();
        let (keys, _) = cache.read(1, &[slot], 1).unwrap();
        let values: Vec<f32> = keys.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(values, vec![0.0; values.len()]);
    }
}
