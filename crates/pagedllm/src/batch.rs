//! The KV cache as a pool of blocks, and the batch that reads it.
//!
//! One tensor per layer, `[blocks, block_size, kv_heads, head_dim]`, allocated
//! once. A sequence holds a list of blocks rather than a run of memory, so its
//! cache no longer has to be contiguous and the pool no longer has to reserve
//! the longest it might reach.
//!
//! The contiguous cache stage 3 measured is not gone, it is a setting: a block
//! as wide as the whole reservation gives every sequence exactly one block, and
//! the pool is back to handing out fixed-size slots. That is literally what a
//! reservation is, and keeping both behind one implementation is what makes the
//! comparison a flag rather than a checkout of an old commit.
//!
//! What still costs, at either setting, is the read. A batch of sequences at
//! different lengths is a rectangle no row fills, and gathering it out of the
//! pool copies it before the multiply. Stage 5 is the kernel that reads the
//! blocks in place and removes that copy. This stage removes the reservation and
//! leaves the copy alone on purpose, so the two gains can be told apart.

use candle_core::{DType, Device, Tensor};

use crate::blocks::{BlockId, BlockTable};
use crate::{Error, Result};

/// How the pool is sized.
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// Tokens per block. As wide as the context this is a reservation per
    /// sequence, which is what stage 3 measured.
    pub block_size: usize,
    /// How many blocks the pool holds.
    pub blocks: usize,
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
        self.blocks * self.block_size * self.bytes_per_token(dtype)
    }

    /// The most blocks that fit in `budget` bytes, at least one.
    pub fn blocks_within(&self, budget: usize, dtype: DType) -> usize {
        let per_block = self.block_size * self.bytes_per_token(dtype);
        (budget / per_block.max(1)).max(1)
    }
}

/// One forward pass's worth of work.
///
/// A rectangle carries the same number of tokens on every row: decoding is many
/// rows of one token, a prefill is one row of many. `unfolded` builds the other
/// shape, one row per token, which is what lets a slice of somebody's prompt
/// ride in the same pass as everybody else's next token.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Token ids, row-major, `rows * seq` of them.
    pub tokens: Vec<u32>,
    /// Where each of those tokens is written, as a flat position in the pool.
    /// This is the block table already resolved, one entry per token.
    pub write_slots: Vec<u32>,
    /// The blocks every row reads, padded to the widest row so the gather is one
    /// rectangle.
    pub read_blocks: Vec<BlockId>,
    /// How many blocks each row of `read_blocks` holds.
    pub blocks_per_row: usize,
    /// Tokens already written for each row, which is where its first token of
    /// this pass sits.
    pub starts: Vec<usize>,
    /// How many rows.
    pub rows: usize,
    /// How many tokens per row.
    pub seq: usize,
    /// Which of the `rows * seq` positions the model should project to logits,
    /// flattened row-major.
    ///
    /// Not every token needs them, and computing the ones nobody reads is the
    /// most expensive way to waste a pass: the vocabulary is 151 936 wide, so a
    /// 512-token chunk projected in full would be 311 MB of logits to produce
    /// and to move, for one row anybody looks at. A chunk that is not the last
    /// of its prompt asks for none at all.
    pub logit_rows: Vec<u32>,
}

impl Batch {
    /// Build from the tables of the rows taking part.
    ///
    /// The tables must already hold blocks for the tokens being written. Making
    /// sure of that is the scheduler's job, because it is what preempts when the
    /// blocks cannot be found.
    ///
    /// # Panics
    ///
    /// If a block index does not fit in a `u32`, which no pool this size can
    /// reach.
    pub fn new(
        tokens: Vec<u32>,
        seq: usize,
        tables: &[&BlockTable],
        block_size: usize,
    ) -> Result<Self> {
        let rows = tables.len();
        if rows == 0 || seq == 0 || tokens.len() != rows * seq {
            return Err(Error::Config(format!(
                "{} tokens for {rows} rows of {seq}",
                tokens.len()
            )));
        }
        let starts: Vec<usize> = tables.iter().map(|t| t.tokens()).collect();
        let longest = starts.iter().map(|s| s + seq).max().unwrap_or(0);
        let blocks_per_row = longest.div_ceil(block_size);

        let mut write_slots = Vec::with_capacity(rows * seq);
        let mut read_blocks = Vec::with_capacity(rows * blocks_per_row);
        for (row, table) in tables.iter().enumerate() {
            for offset in 0..seq {
                let position = starts[row] + offset;
                let slot = table.slot_of(position).ok_or_else(|| {
                    Error::Config(format!("row {row} has no block for position {position}"))
                })?;
                write_slots.push(u32::try_from(slot).expect("a pool fits in u32"));
            }
            // Padded with the row's own first block rather than an arbitrary
            // one. This changes no answer, and that was checked: the mask hides
            // the padding whatever sits behind it. It is here so that a future
            // mask defect reads this sequence's own history rather than a
            // stranger's, which is a wrong answer instead of a leak.
            let blocks = table.blocks();
            let filler = *blocks.first().unwrap_or(&0);
            for index in 0..blocks_per_row {
                read_blocks.push(blocks.get(index).copied().unwrap_or(filler));
            }
        }

        // A rectangle predicts from the last token of every row.
        let logit_rows = (0..rows)
            .map(|row| u32::try_from(row * seq + seq - 1).expect("a batch fits in u32"))
            .collect();

        Ok(Self {
            tokens,
            write_slots,
            read_blocks,
            blocks_per_row,
            starts,
            rows,
            seq,
            logit_rows,
        })
    }

    /// Build a pass out of individual tokens rather than out of equal rows.
    ///
    /// This is what lets one pass carry a slice of somebody's prompt next to
    /// everybody else's next token. A rectangle cannot: its rows all have the
    /// same length, and a 512-token chunk beside sixteen one-token decodes is
    /// not one. Unfolded, a row is a token, `seq` is one, and the mask and the
    /// kernel need no change at all, because both already take a position and a
    /// block table per row.
    ///
    /// `entries` is one `(token, table, position)` per token, and `predicts`
    /// marks the entries whose logits are wanted: the last token of a decode,
    /// and the last token of a prompt's final chunk. A chunk in the middle of a
    /// prompt asks for nothing.
    ///
    /// # Panics
    ///
    /// If a slot index does not fit in a `u32`, which needs a pool of four
    /// billion blocks.
    pub fn unfolded(
        entries: &[(u32, &BlockTable, usize)],
        predicts: &[usize],
        block_size: usize,
    ) -> Result<Self> {
        if entries.is_empty() {
            return Err(Error::Config("an empty batch".into()));
        }
        let rows = entries.len();
        let longest = entries
            .iter()
            .map(|&(_, _, position)| position + 1)
            .max()
            .unwrap_or(0);
        let blocks_per_row = longest.div_ceil(block_size);

        let mut tokens = Vec::with_capacity(rows);
        let mut write_slots = Vec::with_capacity(rows);
        let mut starts = Vec::with_capacity(rows);
        let mut read_blocks = Vec::with_capacity(rows * blocks_per_row);
        for &(token, table, position) in entries {
            let slot = table
                .slot_of(position)
                .ok_or_else(|| Error::Config(format!("no block for position {position}")))?;
            tokens.push(token);
            write_slots.push(u32::try_from(slot).expect("a pool fits in u32"));
            starts.push(position);
            // Padded with the row's own first block, for the same reason the
            // rectangle is: a mask that leaked would read this sequence's own
            // history rather than a stranger's.
            let blocks = table.blocks();
            let filler = *blocks.first().unwrap_or(&0);
            for index in 0..blocks_per_row {
                read_blocks.push(blocks.get(index).copied().unwrap_or(filler));
            }
        }

        Ok(Self {
            tokens,
            write_slots,
            read_blocks,
            blocks_per_row,
            starts,
            rows,
            seq: 1,
            logit_rows: predicts
                .iter()
                .map(|&row| u32::try_from(row).expect("a batch fits in u32"))
                .collect(),
        })
    }

    /// The furthest any row reaches once this pass is written.
    pub fn longest(&self) -> usize {
        self.starts.iter().map(|s| s + self.seq).max().unwrap_or(0)
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
    /// Two things at once. It forbids a query from reading a key ahead of it, as
    /// always; and it forbids every row from reading the part of the rectangle
    /// that belongs to a longer sequence than its own, which is the padding a
    /// batch of unequal lengths cannot avoid.
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

/// The keys and values of every resident sequence, as a pool of blocks.
#[derive(Debug)]
pub struct PagedCache {
    /// Per layer, `[blocks * block_size, kv_heads * head_dim]`, the shape a
    /// scatter writes.
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    /// The same storage as `[blocks, block_size * kv_heads * head_dim]`, the
    /// shape a gather of whole blocks reads.
    key_blocks: Vec<Tensor>,
    value_blocks: Vec<Tensor>,
    config: CacheConfig,
}

impl PagedCache {
    /// Allocate the pool. This is the whole allocation: nothing here grows.
    pub fn new(config: CacheConfig, dtype: DType, device: &Device) -> Result<Self> {
        let slots = config.blocks * config.block_size;
        let width = config.kv_heads * config.head_dim;
        let mut keys = Vec::with_capacity(config.layers);
        let mut values = Vec::with_capacity(config.layers);
        let mut key_blocks = Vec::with_capacity(config.layers);
        let mut value_blocks = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            let k = Tensor::zeros((slots, width), dtype, device)?;
            let v = Tensor::zeros((slots, width), dtype, device)?;
            key_blocks.push(k.reshape((config.blocks, config.block_size * width))?);
            value_blocks.push(v.reshape((config.blocks, config.block_size * width))?);
            keys.push(k);
            values.push(v);
        }
        Ok(Self {
            keys,
            values,
            key_blocks,
            value_blocks,
            config,
        })
    }

    /// How the pool is shaped.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// One layer's keys and values, as the flat `[slots, kv_heads * head_dim]`
    /// pools a kernel indexes directly.
    pub fn layer(&self, layer: usize) -> (&Tensor, &Tensor) {
        (&self.keys[layer], &self.values[layer])
    }

    /// Write a batch's keys and values into the blocks its rows hold.
    ///
    /// One scatter per layer whatever the batch, where the slot cache needed one
    /// call per row. The block table has already been resolved into flat
    /// positions, so the scatter does not care that a sequence's tokens are
    /// scattered.
    pub(crate) fn write(&self, layer: usize, k: &Tensor, v: &Tensor, slots: &Tensor) -> Result<()> {
        let width = self.config.kv_heads * self.config.head_dim;
        // Transposed first, because the attention hands these over head-major,
        // `[rows, kv_heads, seq, head_dim]`, and a token's whole key vector has
        // to be one run before it can be scattered to one place. Reshaping
        // without the transpose splits every token across head boundaries, and
        // reads it back the same way, so nothing disagrees until the result is
        // compared against a path that never went through the pool.
        let flat = |t: &Tensor| -> Result<Tensor> {
            Ok(t.transpose(1, 2)?.contiguous()?.reshape(((), width))?)
        };
        self.keys[layer].scatter_set(slots, &flat(k)?, 0)?;
        self.values[layer].scatter_set(slots, &flat(v)?, 0)?;
        Ok(())
    }

    /// Gather what the attention reads, `[rows, kv_heads, longest, head_dim]`.
    pub(crate) fn read(
        &self,
        layer: usize,
        batch: &Batch,
        blocks: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let longest = batch.longest();
        let tokens = batch.blocks_per_row * self.config.block_size;
        let gather = |pool: &Tensor| -> Result<Tensor> {
            Ok(pool
                .index_select(blocks, 0)?
                .reshape((
                    batch.rows,
                    tokens,
                    self.config.kv_heads,
                    self.config.head_dim,
                ))?
                // Narrowed before the transpose so the copy the transpose forces
                // carries what the batch reaches, not the whole of every block
                // it borrowed those tokens from.
                .narrow(1, 0, longest)?
                .transpose(1, 2)?
                .contiguous()?)
        };
        Ok((
            gather(&self.key_blocks[layer])?,
            gather(&self.value_blocks[layer])?,
        ))
    }

    /// The scatter index a write needs, `[tokens, kv_heads * head_dim]`.
    ///
    /// Every column of a row carries the same position, because a token's whole
    /// key vector goes to one place.
    pub(crate) fn write_index(&self, batch: &Batch, device: &Device) -> Result<Tensor> {
        let width = self.config.kv_heads * self.config.head_dim;
        let mut index = Vec::with_capacity(batch.write_slots.len() * width);
        for &slot in &batch.write_slots {
            index.extend(std::iter::repeat_n(slot, width));
        }
        Ok(Tensor::from_vec(
            index,
            (batch.write_slots.len(), width),
            device,
        )?)
    }

    /// The gather index a read needs, one entry per block of the rectangle.
    #[allow(clippy::unused_self)]
    pub(crate) fn read_index(&self, batch: &Batch, device: &Device) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            batch.read_blocks.clone(),
            batch.read_blocks.len(),
            device,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::{Batch, CacheConfig, PagedCache};
    use crate::blocks::{BlockAllocator, BlockTable};
    use candle_core::{DType, Device, Tensor};

    fn config(block_size: usize, blocks: usize) -> CacheConfig {
        CacheConfig {
            block_size,
            blocks,
            kv_heads: 2,
            head_dim: 4,
            layers: 2,
        }
    }

    /// A sequence's worth of blocks, taken from `pool`.
    fn table(pool: &mut BlockAllocator, block_size: usize, tokens: usize) -> BlockTable {
        let mut table = BlockTable::new(block_size);
        for _ in 0..table.blocks_needed(tokens) {
            table.push(pool.allocate().unwrap());
        }
        table
    }

    fn payload(value: f32, tokens: usize) -> Tensor {
        Tensor::full(value, (1, 2, tokens, 4), &Device::Cpu).unwrap()
    }

    #[test]
    fn a_token_of_cache_costs_what_the_arithmetic_says() {
        // Qwen3-0.6B: 28 layers, 8 key heads, 128 wide, in bf16.
        let qwen = CacheConfig {
            block_size: 16,
            blocks: 1,
            kv_heads: 8,
            head_dim: 128,
            layers: 28,
        };
        assert_eq!(qwen.bytes_per_token(DType::BF16), 114_688);
        assert_eq!(qwen.bytes(DType::BF16), 16 * 114_688);
        // The same 3.5 GiB stage 3 spent on 32 reservations buys this many
        // blocks, which is what raises the number of sequences that fit.
        assert_eq!(qwen.blocks_within(3_758_096_384, DType::BF16), 2048);
    }

    #[test]
    fn what_a_sequence_wrote_is_what_comes_back_out_of_its_blocks() {
        let device = Device::Cpu;
        let config = config(2, 8);
        let cache = PagedCache::new(config, DType::F32, &device).unwrap();
        let mut pool = BlockAllocator::new(config.blocks);

        // Two sequences whose blocks interleave, so neither holds a run.
        let mut first = table(&mut pool, 2, 5);
        let mut second = table(&mut pool, 2, 3);
        assert_ne!(first.blocks(), second.blocks());

        let write = |table: &BlockTable, value: f32, seq: usize| {
            let batch = Batch::new(vec![0; seq], seq, &[table], 2).unwrap();
            let index = cache.write_index(&batch, &device).unwrap();
            let k = payload(value, seq);
            cache.write(0, &k, &k.neg().unwrap(), &index).unwrap();
        };
        write(&first, 1.0, 5);
        first.advance(5).unwrap();
        write(&second, 7.0, 3);
        second.advance(3).unwrap();

        let batch = Batch::new(vec![0, 0], 1, &[&first, &second], 2).unwrap();
        let blocks = cache.read_index(&batch, &device).unwrap();
        let (keys, values) = cache.read(0, &batch, &blocks).unwrap();
        assert_eq!(keys.dims(), &[2, 2, 6, 4]);

        let head = |t: &Tensor, row: usize| -> Vec<f32> {
            t.narrow(0, row, 1)
                .unwrap()
                .narrow(1, 0, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        // Five tokens of 1.0 for the first sequence, whatever it wrote, out of
        // blocks that are not next to each other.
        assert_eq!(&head(&keys, 0)[..20], &[1.0; 20]);
        // Three tokens for the second, and its values are the negatives, so a
        // read that crossed the two would be visible in the sign.
        assert_eq!(&head(&values, 1)[..12], &[-7.0; 12]);
    }

    #[test]
    fn a_block_as_wide_as_the_context_is_one_block_a_sequence() {
        let mut pool = BlockAllocator::new(4);
        let table = table(&mut pool, 1024, 300);
        assert_eq!(table.blocks().len(), 1, "a reservation is one block");
        assert_eq!(table.capacity(), 1024);
        // Which is what stage 3 reserved, and what it wasted.
        assert_eq!(table.wasted_tokens(), 1024);
    }

    #[test]
    fn a_batch_rectangle_is_padded_with_the_rows_own_first_block() {
        let mut pool = BlockAllocator::new(8);
        // Seven tokens of room for six written, so the row has somewhere to put
        // the token this batch is about to produce.
        let mut long = table(&mut pool, 2, 7);
        long.advance(6).unwrap();
        let mut short = table(&mut pool, 2, 3);
        short.advance(2).unwrap();

        let batch = Batch::new(vec![0, 0], 1, &[&long, &short], 2).unwrap();
        assert_eq!(
            batch.blocks_per_row, 4,
            "seven tokens need four blocks of two"
        );
        assert_eq!(batch.read_blocks.len(), 8);
        let short_row = &batch.read_blocks[4..];
        assert_eq!(short_row[2], short_row[0]);
        assert_eq!(short_row[3], short_row[0]);
    }

    #[test]
    fn a_batch_whose_tables_are_too_small_is_refused() {
        let mut pool = BlockAllocator::new(4);
        let mut table = BlockTable::new(2);
        table.push(pool.allocate().unwrap());
        table.advance(2).unwrap();
        // The block is full and nothing was added, so the next token has
        // nowhere to go.
        assert!(Batch::new(vec![0], 1, &[&table], 2).is_err());
    }
}
