//! Physical blocks, and the tables that say where a sequence's tokens live.
//!
//! This is the part of `PagedAttention` that is not a kernel. Nothing here
//! touches a tensor or a device: it hands out fixed-size blocks from a free
//! list, and records for each sequence which physical block holds each stretch
//! of its history. That is a page table, and the reason the analogy is worth
//! making is that it buys the same thing: a sequence stops needing its memory to
//! be one unbroken run, so the pool stops having to reserve the longest it might
//! ever reach.
//!
//! What it costs instead is bounded and small. A sequence wastes at most one
//! partial block, so at sixteen tokens a block and 112 KiB a token the worst
//! case is 1.68 MiB per sequence, against the 96 MiB a 1024-token reservation
//! wastes on a request that stops after a hundred tokens.
//!
//! Blocks are not shared here and carry no reference count. Sharing is what
//! prefix caching is for, at stage 7, and nothing before it would increment a
//! counter past one.

use crate::{Error, Result};

/// A physical block's index in the pool.
pub type BlockId = u32;

/// The free list.
#[derive(Debug)]
pub struct BlockAllocator {
    free: Vec<BlockId>,
    total: usize,
    /// Blocks handed out over the allocator's life, which is what says whether
    /// a benchmark exercised the pool or fitted inside it.
    handed_out: u64,
}

impl BlockAllocator {
    /// A pool of `total` blocks, all free.
    ///
    /// # Panics
    ///
    /// If the pool holds more blocks than a `u32` can index, which at sixteen
    /// tokens a block is more memory than exists.
    pub fn new(total: usize) -> Self {
        Self {
            // Handed out from the end so the first block taken is block zero,
            // which makes a dump of a block table legible and changes nothing.
            free: (0..total)
                .rev()
                .map(|i| BlockId::try_from(i).expect("a pool fits in u32"))
                .collect(),
            total,
            handed_out: 0,
        }
    }

    /// Take a block, or `None` when the pool is empty.
    pub fn allocate(&mut self) -> Option<BlockId> {
        let block = self.free.pop()?;
        self.handed_out += 1;
        Some(block)
    }

    /// Give a block back.
    pub fn free(&mut self, block: BlockId) {
        debug_assert!(
            !self.free.contains(&block),
            "block {block} was freed twice, which would hand it to two sequences"
        );
        self.free.push(block);
    }

    /// Give back everything a table holds, leaving it empty.
    pub fn free_table(&mut self, table: &mut BlockTable) {
        for block in table.take_blocks() {
            self.free(block);
        }
    }

    /// How many blocks are free.
    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// How many blocks the pool holds.
    pub fn total(&self) -> usize {
        self.total
    }

    /// How many blocks are held by some sequence.
    pub fn in_use(&self) -> usize {
        self.total - self.free.len()
    }

    /// Blocks handed out since the pool was built.
    pub fn handed_out(&self) -> u64 {
        self.handed_out
    }
}

/// Where one sequence's tokens live, logical position to physical block.
#[derive(Debug, Clone)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
    block_size: usize,
    tokens: usize,
}

impl BlockTable {
    /// An empty table for a pool of this block size.
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            block_size: block_size.max(1),
            tokens: 0,
        }
    }

    /// The physical blocks, in logical order.
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// How many tokens the sequence has written.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// How many tokens the blocks already held could take.
    pub fn capacity(&self) -> usize {
        self.blocks.len() * self.block_size
    }

    /// How many blocks are still needed to hold `count` more tokens.
    pub fn blocks_needed(&self, count: usize) -> usize {
        let wanted = self.tokens + count;
        if wanted <= self.capacity() {
            return 0;
        }
        (wanted - self.capacity()).div_ceil(self.block_size)
    }

    /// Add a block to the end of the table.
    pub fn push(&mut self, block: BlockId) {
        self.blocks.push(block);
    }

    /// Record that `count` tokens were written.
    pub fn advance(&mut self, count: usize) -> Result<()> {
        if self.tokens + count > self.capacity() {
            return Err(Error::Config(format!(
                "{} tokens written into {} of capacity",
                self.tokens + count,
                self.capacity()
            )));
        }
        self.tokens += count;
        Ok(())
    }

    /// Where logical position `position` sits in a pool laid out block by block.
    ///
    /// This is the whole indirection, and it is why the attention kernel needs
    /// the table: consecutive positions of one sequence are not consecutive in
    /// memory once they cross a block boundary.
    pub fn slot_of(&self, position: usize) -> Option<usize> {
        let block = *self.blocks.get(position / self.block_size)?;
        Some(block as usize * self.block_size + position % self.block_size)
    }

    /// Every physical slot the sequence occupies, in logical order.
    ///
    /// # Panics
    ///
    /// If a written position has no block, which `advance` refuses to create.
    pub fn slots(&self) -> Vec<usize> {
        (0..self.tokens)
            .map(|position| {
                self.slot_of(position)
                    .expect("a written position has a block")
            })
            .collect()
    }

    /// Empty the table and return what it held, for the allocator to take back.
    pub fn take_blocks(&mut self) -> Vec<BlockId> {
        self.tokens = 0;
        std::mem::take(&mut self.blocks)
    }

    /// Tokens held in blocks that are allocated but not full.
    ///
    /// The whole of what paging wastes. Bounded by one block per sequence,
    /// where a reservation wastes everything it never reaches.
    pub fn wasted_tokens(&self) -> usize {
        self.capacity() - self.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockAllocator, BlockTable};

    #[test]
    fn blocks_are_handed_out_in_order_and_come_back() {
        let mut pool = BlockAllocator::new(4);
        assert_eq!(pool.available(), 4);
        let held: Vec<_> = (0..4).map(|_| pool.allocate().unwrap()).collect();
        assert_eq!(held, vec![0, 1, 2, 3]);
        assert_eq!(pool.allocate(), None, "an empty pool hands out nothing");
        assert_eq!(pool.in_use(), 4);

        pool.free(2);
        assert_eq!(pool.allocate(), Some(2));
        assert_eq!(pool.handed_out(), 5);
    }

    #[test]
    fn a_table_asks_for_a_block_only_when_the_last_one_is_full() {
        let mut table = BlockTable::new(4);
        assert_eq!(table.blocks_needed(1), 1);
        assert_eq!(table.blocks_needed(4), 1);
        assert_eq!(table.blocks_needed(5), 2, "five tokens do not fit in one");

        table.push(9);
        assert_eq!(table.capacity(), 4);
        table.advance(3).unwrap();
        // Three of four used, so the fourth token needs nothing new and the
        // fifth needs one more block.
        assert_eq!(table.blocks_needed(1), 0);
        assert_eq!(table.blocks_needed(2), 1);
    }

    #[test]
    fn a_position_past_a_block_boundary_lands_in_the_next_block() {
        let mut table = BlockTable::new(4);
        // Deliberately not consecutive: the point of a table is that logical
        // order and physical order are unrelated.
        for block in [7, 2, 5] {
            table.push(block);
        }
        table.advance(10).unwrap();

        assert_eq!(table.slot_of(0), Some(7 * 4));
        assert_eq!(table.slot_of(3), Some(7 * 4 + 3));
        assert_eq!(table.slot_of(4), Some(2 * 4), "the boundary moves it");
        assert_eq!(table.slot_of(9), Some(5 * 4 + 1));
        assert_eq!(table.slot_of(12), None, "past the blocks it holds");

        let slots = table.slots();
        assert_eq!(slots.len(), 10);
        assert_eq!(&slots[..5], &[28, 29, 30, 31, 8]);
    }

    #[test]
    fn writing_past_the_capacity_is_refused_rather_than_silently_wrapping() {
        let mut table = BlockTable::new(4);
        table.push(0);
        assert!(table.advance(4).is_ok());
        assert!(table.advance(1).is_err(), "the block is full");
        assert_eq!(table.tokens(), 4);
    }

    #[test]
    fn what_paging_wastes_is_one_partial_block_and_no_more() {
        let mut table = BlockTable::new(16);
        for block in 0..7 {
            table.push(block);
        }
        table.advance(97).unwrap();
        assert_eq!(table.capacity(), 112);
        assert_eq!(table.wasted_tokens(), 15);
        assert!(
            table.wasted_tokens() < 16,
            "waste is bounded by one block, whatever the sequence length"
        );
    }

    #[test]
    fn freeing_a_table_returns_every_block_and_empties_it() {
        let mut pool = BlockAllocator::new(8);
        let mut table = BlockTable::new(4);
        for _ in 0..3 {
            table.push(pool.allocate().unwrap());
        }
        table.advance(9).unwrap();
        assert_eq!(pool.available(), 5);

        pool.free_table(&mut table);
        assert_eq!(pool.available(), 8);
        assert_eq!(table.tokens(), 0);
        assert_eq!(table.capacity(), 0);
        assert_eq!(table.slot_of(0), None);
    }

    /// The arithmetic the whole stage exists for, at Qwen3-0.6B's shape.
    #[test]
    fn paging_holds_far_more_sequences_than_a_reservation_does() {
        let bytes_per_token = 114_688usize;
        let budget = 3_758_096_384usize; // 3.5 GiB, what stage 3 reserved.
        let block_size = 16;

        // A reservation of 1024 tokens per sequence, whatever it uses.
        let reserved = budget / (1024 * bytes_per_token);
        assert_eq!(reserved, 32);

        // The same budget in blocks, against requests that really use about 140
        // tokens: nine blocks each, the last one part full.
        let blocks = budget / (block_size * bytes_per_token);
        let per_sequence = 140usize.div_ceil(block_size);
        assert_eq!(blocks / per_sequence, 227);
        assert!(blocks / per_sequence > 7 * reserved);
    }
}
