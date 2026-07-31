//! Physical blocks, the tables that say where a sequence's tokens live, and the
//! cache that lets two sequences share the ones they agree on.
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
//! On top of that sits prefix caching. A block that is full is also final: its
//! tokens cannot change, so its keys and values are a function of the tokens
//! that came before it and of its own. Give it a name that captures exactly
//! that, and two requests that begin the same way can point at one copy.
//!
//! The name is a chain: a block's hash is its own tokens combined with the hash
//! of the block before it, so it stands for the whole prefix rather than for
//! sixteen tokens that might appear anywhere. Two sequences that agree for three
//! blocks and then differ share three blocks and no more, and that falls out of
//! the chain rather than being checked for.
//!
//! Nothing here needs copy on write, and that is worth stating because it is the
//! usual complication. Only a full block is ever shared, a full block is never
//! written to again, and the partial block a sequence is still filling is always
//! its own.

use std::collections::HashMap;

use crate::{Error, Result};

/// A physical block's index in the pool.
pub type BlockId = u32;

/// What a full block's contents are worth being called.
pub type BlockHash = u64;

/// Combine a block's tokens with the hash of everything before it.
///
/// `splitmix64` finishing an `FNV-1a` walk, hand-written for the same reason the
/// sampler's generator is: a hash whose value is fixed by this repository is
/// what lets a test assert which blocks are shared.
///
/// Collisions would hand a sequence another's keys and values, which is a wrong
/// answer rather than a slow one. Sixty-four bits over the few thousand blocks a
/// pool holds is the same argument every content-addressed cache makes, and it
/// is stated here rather than left implied.
pub fn block_hash(parent: Option<BlockHash>, tokens: &[u32]) -> BlockHash {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |value: u64| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    // The parent goes in first, so a block that follows a different prefix
    // cannot land on the same name as this one.
    mix(parent.unwrap_or(0));
    mix(tokens.len() as u64);
    for &token in tokens {
        mix(u64::from(token));
    }
    let mut z = hash.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// What the pool knows about one block.
#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    /// How many sequences hold it. Zero means it can be evicted, not that it is
    /// gone: a block at zero with a name is exactly what prefix caching keeps.
    holders: u32,
    /// Its name, once it is full and final.
    hash: Option<BlockHash>,
    /// When it last stopped being held, which orders eviction.
    released_at: u64,
}

/// The free list, the reference counts, and the cache of named blocks.
#[derive(Debug)]
pub struct BlockAllocator {
    slots: Vec<Slot>,
    free: Vec<BlockId>,
    /// Name to block, for the blocks that have one and are still resident.
    cached: HashMap<BlockHash, BlockId>,
    total: usize,
    clock: u64,
    handed_out: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
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
            slots: vec![Slot::default(); total],
            // Handed out from the end so the first block taken is block zero,
            // which makes a dump of a block table legible and changes nothing.
            free: (0..total)
                .rev()
                .map(|i| BlockId::try_from(i).expect("a pool fits in u32"))
                .collect(),
            cached: HashMap::new(),
            total,
            clock: 0,
            handed_out: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Take a block for a sequence's own use.
    ///
    /// A block carrying no name goes first, because taking it costs nothing.
    /// Only when every free block is named does one lose its name, and then it
    /// is the one unheld longest. A free list on its own would not do this: it
    /// is a stack, so it hands back what was released most recently, which is
    /// the newest cached block rather than the oldest.
    ///
    /// Cached blocks give way before anything else does. Losing one costs a
    /// recomputation that may never be asked for; the alternative, preempting a
    /// sequence that is running, costs one that is certain and immediate.
    pub fn allocate(&mut self) -> Option<BlockId> {
        // Searched from the end, which is where the free list is popped from,
        // so an untouched pool still hands out block zero first and a dump of a
        // block table stays legible.
        let anonymous = self
            .free
            .iter()
            .rposition(|&b| self.slots[b as usize].hash.is_none());
        let index = if let Some(index) = anonymous {
            index
        } else {
            let (index, _) = self
                .free
                .iter()
                .enumerate()
                .min_by_key(|&(_, &b)| self.slots[b as usize].released_at)?;
            self.evictions += 1;
            index
        };
        let block = self.free.swap_remove(index);
        self.forget(block);
        self.slots[block as usize].holders = 1;
        self.handed_out += 1;
        Some(block)
    }

    /// Take the block that already holds this prefix, if one does.
    ///
    /// A hit adds a holder to a block that may have none, which is what pulls it
    /// back out of the eviction order.
    pub fn acquire_cached(&mut self, hash: BlockHash) -> Option<BlockId> {
        let block = *self.cached.get(&hash)?;
        self.slots[block as usize].holders += 1;
        if self.slots[block as usize].holders == 1 {
            // It was resident but unheld; it is not a candidate for eviction
            // while someone is reading it.
            self.free.retain(|&b| b != block);
        }
        self.hits += 1;
        Some(block)
    }

    /// Record that a miss happened, so the hit rate counts what was asked for.
    pub fn record_miss(&mut self) {
        self.misses += 1;
    }

    /// Name a block that has just been filled, so a later sequence can find it.
    ///
    /// Only a full block gets a name. A partial one can still grow, so its keys
    /// and values are not yet a function of anything a later request could match
    /// against, and sharing it would be sharing something that is about to
    /// change.
    pub fn publish(&mut self, block: BlockId, hash: BlockHash) {
        if let Some(&existing) = self.cached.get(&hash) {
            // Another sequence filled the same prefix first. Keep the resident
            // copy named and leave this one anonymous rather than stealing the
            // name, which would leave the other sequence's block unfindable and
            // uncacheable for the rest of its life.
            if existing != block {
                return;
            }
        }
        self.slots[block as usize].hash = Some(hash);
        self.cached.insert(hash, block);
    }

    /// Give a block back. It keeps its name and its contents until something
    /// needs the space.
    pub fn release(&mut self, block: BlockId) {
        let slot = &mut self.slots[block as usize];
        if slot.holders == 0 {
            debug_assert!(false, "block {block} released more often than it was taken");
            return;
        }
        slot.holders -= 1;
        if slot.holders == 0 {
            self.clock += 1;
            self.slots[block as usize].released_at = self.clock;
            self.free.push(block);
        }
    }

    /// Give back everything a table holds, leaving it empty.
    pub fn free_table(&mut self, table: &mut BlockTable) {
        for block in table.take_blocks() {
            self.release(block);
        }
    }

    /// How many blocks no sequence holds. Some of them still carry a name, and
    /// answering a request from one of those is what the cache is for.
    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// How many blocks the pool holds.
    pub fn total(&self) -> usize {
        self.total
    }

    /// How many blocks some sequence holds.
    pub fn in_use(&self) -> usize {
        self.total - self.free.len()
    }

    /// How many blocks carry a name, held or not.
    ///
    /// A block keeps its name while the sequence that filled it is still
    /// reading, which is what lets a second request share a prefix the first has
    /// not finished with, so this is not the same as the number of blocks free
    /// to be handed out.
    pub fn cached_blocks(&self) -> usize {
        self.cached.len()
    }

    /// Blocks handed out since the pool was built.
    pub fn handed_out(&self) -> u64 {
        self.handed_out
    }

    /// Blocks answered from the cache, and blocks that had to be computed.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Blocks a sequence had to fill itself.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Named blocks dropped to make room.
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Forget every name, which is what turns the cache off without changing
    /// any other path.
    pub fn clear_cache(&mut self) {
        for slot in &mut self.slots {
            slot.hash = None;
        }
        self.cached.clear();
    }

    /// Drop a block's name, because its contents are about to be replaced.
    fn forget(&mut self, block: BlockId) {
        if let Some(hash) = self.slots[block as usize].hash.take() {
            // Only if it is still the block that name points at: another block
            // may have taken the name over in the meantime.
            if self.cached.get(&hash) == Some(&block) {
                self.cached.remove(&hash);
            }
        }
    }
}

/// Where one sequence's tokens live, logical position to physical block.
#[derive(Debug, Clone)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
    block_size: usize,
    tokens: usize,
    /// The name of each full block, in order, so the next one can chain onto it.
    hashes: Vec<BlockHash>,
}

impl BlockTable {
    /// An empty table for a pool of this block size.
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            block_size: block_size.max(1),
            tokens: 0,
            hashes: Vec::new(),
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

    /// Tokens per block.
    pub fn block_size(&self) -> usize {
        self.block_size
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

    /// Add a block that already holds the tokens up to `hash`, and count them as
    /// written.
    ///
    /// This is what a cache hit does: the block arrives full, so the sequence
    /// starts further along than its prompt would otherwise put it, and the
    /// prefill that follows covers only what was not found.
    pub fn push_cached(&mut self, block: BlockId, hash: BlockHash) {
        self.blocks.push(block);
        self.hashes.push(hash);
        self.tokens += self.block_size;
    }

    /// The name of the last full block, which the next one chains onto.
    pub fn last_hash(&self) -> Option<BlockHash> {
        self.hashes.last().copied()
    }

    /// How many full blocks have been named.
    pub fn named_blocks(&self) -> usize {
        self.hashes.len()
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

    /// The blocks this sequence has just filled, with the names they earn.
    ///
    /// Returns one entry per block that became full and was not already named,
    /// in order, so the caller can publish them. `history` is every token the
    /// sequence has written, which is what the names are computed from.
    pub fn newly_full(&mut self, history: &[u32]) -> Vec<(BlockId, BlockHash)> {
        let full = self.tokens / self.block_size;
        let mut published = Vec::new();
        while self.hashes.len() < full.min(self.blocks.len()) {
            let index = self.hashes.len();
            let start = index * self.block_size;
            let Some(tokens) = history.get(start..start + self.block_size) else {
                break;
            };
            let hash = block_hash(self.hashes.last().copied(), tokens);
            self.hashes.push(hash);
            published.push((self.blocks[index], hash));
        }
        published
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

    /// Empty the table and return what it held, for the allocator to take back.
    pub fn take_blocks(&mut self) -> Vec<BlockId> {
        self.tokens = 0;
        self.hashes.clear();
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
    use super::{BlockAllocator, BlockTable, block_hash};

    #[test]
    fn blocks_are_handed_out_in_order_and_come_back() {
        let mut pool = BlockAllocator::new(4);
        assert_eq!(pool.available(), 4);
        let held: Vec<_> = (0..4).map(|_| pool.allocate().unwrap()).collect();
        assert_eq!(held, vec![0, 1, 2, 3]);
        assert_eq!(pool.allocate(), None, "an empty pool hands out nothing");
        assert_eq!(pool.in_use(), 4);

        pool.release(2);
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
        assert_eq!(table.blocks_needed(1), 0);
        assert_eq!(table.blocks_needed(2), 1);
    }

    #[test]
    fn a_position_past_a_block_boundary_lands_in_the_next_block() {
        let mut table = BlockTable::new(4);
        for block in [7, 2, 5] {
            table.push(block);
        }
        table.advance(10).unwrap();

        assert_eq!(table.slot_of(0), Some(7 * 4));
        assert_eq!(table.slot_of(3), Some(7 * 4 + 3));
        assert_eq!(table.slot_of(4), Some(2 * 4), "the boundary moves it");
        assert_eq!(table.slot_of(9), Some(5 * 4 + 1));
        assert_eq!(table.slot_of(12), None, "past the blocks it holds");
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
        assert!(table.wasted_tokens() < 16);
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

    #[test]
    fn paging_holds_far_more_sequences_than_a_reservation_does() {
        let bytes_per_token = 114_688usize;
        let budget = 3_758_096_384usize;
        let block_size = 16;
        let reserved = budget / (1024 * bytes_per_token);
        assert_eq!(reserved, 32);
        let blocks = budget / (block_size * bytes_per_token);
        let per_sequence = 140usize.div_ceil(block_size);
        assert_eq!(blocks / per_sequence, 227);
        assert!(blocks / per_sequence > 7 * reserved);
    }

    /// The name has to stand for the whole prefix, not for the tokens in hand.
    #[test]
    fn a_block_name_depends_on_everything_before_it() {
        let tokens = [1u32, 2, 3, 4];
        let root = block_hash(None, &tokens);
        assert_eq!(
            root,
            block_hash(None, &tokens),
            "the same input names the same block"
        );
        assert_ne!(
            root,
            block_hash(Some(99), &tokens),
            "the same tokens after a different prefix are a different block"
        );
        assert_ne!(
            block_hash(None, &[1, 2, 3, 4]),
            block_hash(None, &[4, 3, 2, 1])
        );
        assert_ne!(block_hash(None, &[1, 2]), block_hash(None, &[1, 2, 0, 0]));
    }

    #[test]
    fn a_named_block_is_handed_to_a_second_sequence_rather_than_recomputed() {
        let mut pool = BlockAllocator::new(4);
        let block = pool.allocate().unwrap();
        let hash = block_hash(None, &[1, 2, 3, 4]);
        pool.publish(block, hash);
        assert_eq!(pool.cached_blocks(), 1);

        // A second sequence finds it while the first still holds it.
        assert_eq!(pool.acquire_cached(hash), Some(block));
        assert_eq!(pool.in_use(), 1, "one block, two holders");

        // Both let go before it becomes free.
        pool.release(block);
        assert_eq!(pool.in_use(), 1);
        pool.release(block);
        assert_eq!(pool.in_use(), 0);
    }

    #[test]
    fn a_block_nobody_holds_keeps_its_contents_until_the_space_is_needed() {
        let mut pool = BlockAllocator::new(2);
        let first = pool.allocate().unwrap();
        let hash = block_hash(None, &[7, 7, 7, 7]);
        pool.publish(first, hash);
        pool.release(first);

        assert_eq!(pool.available(), 2, "unheld, and still named");
        assert_eq!(
            pool.acquire_cached(hash),
            Some(first),
            "a later request finds what an earlier one left"
        );
        assert_eq!(pool.hits(), 1);
    }

    #[test]
    fn the_block_unheld_longest_is_the_one_that_gives_way() {
        let mut pool = BlockAllocator::new(2);
        let (a, b) = (pool.allocate().unwrap(), pool.allocate().unwrap());
        let (ha, hb) = (block_hash(None, &[1]), block_hash(None, &[2]));
        pool.publish(a, ha);
        pool.publish(b, hb);
        pool.release(a);
        pool.release(b);

        // Both are named and unheld, so one has to lose its name, and it is the
        // one released first. A plain free list would have given back b, which
        // was released last.
        let taken = pool.allocate().unwrap();
        assert_eq!(taken, a, "a was unheld longest");
        assert_eq!(pool.evictions(), 1);
        assert_eq!(
            pool.acquire_cached(ha),
            None,
            "its name went with its contents"
        );
        assert_eq!(pool.acquire_cached(hb), Some(b), "the newer one survived");
    }

    #[test]
    fn a_nameless_block_is_taken_before_a_cached_one_loses_its_name() {
        let mut pool = BlockAllocator::new(3);
        let cached = pool.allocate().unwrap();
        let hash = block_hash(None, &[5]);
        pool.publish(cached, hash);
        pool.release(cached);

        // Two blocks were never handed out and carry no name; both go before
        // the cached one does.
        for _ in 0..2 {
            assert_ne!(pool.allocate(), Some(cached));
        }
        assert_eq!(pool.evictions(), 0, "nothing was cached away");
        assert_eq!(pool.acquire_cached(hash), Some(cached), "still there");
    }

    #[test]
    fn a_table_names_a_block_only_once_it_is_full() {
        let mut table = BlockTable::new(4);
        let history: Vec<u32> = (0..10).collect();
        for block in [3, 1, 2] {
            table.push(block);
        }

        table.advance(3).unwrap();
        assert!(
            table.newly_full(&history).is_empty(),
            "three of four is not full"
        );

        table.advance(5).unwrap();
        let published = table.newly_full(&history);
        assert_eq!(published.len(), 2, "eight tokens fill two blocks");
        assert_eq!(published[0].0, 3);
        assert_eq!(published[1].0, 1);
        // The second chains onto the first rather than standing alone.
        assert_eq!(
            published[1].1,
            super::block_hash(Some(published[0].1), &history[4..8])
        );

        assert!(table.newly_full(&history).is_empty(), "nothing new to name");
    }

    #[test]
    fn clearing_the_cache_leaves_the_pool_working() {
        let mut pool = BlockAllocator::new(2);
        let block = pool.allocate().unwrap();
        let hash = block_hash(None, &[1, 2]);
        pool.publish(block, hash);
        pool.release(block);
        pool.clear_cache();

        assert_eq!(pool.cached_blocks(), 0);
        assert_eq!(pool.acquire_cached(hash), None);
        assert_eq!(
            pool.available(),
            2,
            "the block is still there to be handed out"
        );
        assert!(pool.allocate().is_some());
    }
}
