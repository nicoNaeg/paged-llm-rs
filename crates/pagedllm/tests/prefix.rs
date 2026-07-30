//! Prefix caching has to be invisible in the answers.
//!
//! A block taken from the cache holds keys and values some other sequence
//! computed. If the hash that named it stood for anything less than the exact
//! prefix, a request would silently continue someone else's context, and it
//! would read as a fluent answer to a question nobody asked.
//!
//! So the check is not that sharing happens. It is that a run which shares
//! produces the same tokens as a run which does not, on prompts built to make
//! sharing wrong if the naming is wrong: the same block contents after different
//! prefixes, and prefixes that agree for a while and then part.

use candle_core::{DType, Device, Tensor};
use pagedllm::{Batch, BlockAllocator, BlockTable, CacheConfig, Model, PagedCache, block_hash};

mod common;
use common::{compare, load_tiny};

const BLOCK: usize = 4;

fn pool(model: &Model, blocks: usize) -> (PagedCache, BlockAllocator) {
    let config = model.config();
    let cache = PagedCache::new(
        CacheConfig {
            block_size: BLOCK,
            blocks,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            layers: config.num_hidden_layers,
        },
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    (cache, BlockAllocator::new(blocks))
}

/// Run a prompt, optionally taking whatever leading blocks the pool already
/// holds, and return the logits of its last position.
///
/// Mirrors what the scheduler does, in the smallest form that still exercises
/// the naming: claim the leading full blocks by their chained hash, prefill only
/// what is left, then name whatever the pass filled.
fn run(
    model: &Model,
    cache: &PagedCache,
    allocator: &mut BlockAllocator,
    prompt: &[u32],
    share: bool,
) -> (Tensor, usize) {
    let mut table = BlockTable::new(BLOCK);
    let candidates = prompt.len().div_ceil(BLOCK).saturating_sub(1);

    let mut shared = 0usize;
    if share {
        let mut parent = None;
        for index in 0..candidates {
            let start = index * BLOCK;
            let hash = block_hash(parent, &prompt[start..start + BLOCK]);
            let Some(block) = allocator.acquire_cached(hash) else {
                break;
            };
            table.push_cached(block, hash);
            shared += BLOCK;
            parent = Some(hash);
        }
    }
    for _ in 0..table.blocks_needed(prompt.len() - shared) {
        table.push(allocator.allocate().expect("the pool has room"));
    }

    let rest = &prompt[shared..];
    let batch = Batch::new(rest.to_vec(), rest.len(), &[&table], BLOCK).unwrap();
    let logits = model.forward_batch(&batch, cache).unwrap();
    table.advance(rest.len()).unwrap();
    for (block, hash) in table.newly_full(prompt) {
        allocator.publish(block, hash);
    }
    // The sequence ends here, so it lets go. Its full blocks keep their names
    // and their contents until the pool needs the space, which is the whole
    // point: what a finished request leaves behind is what the next one takes.
    allocator.free_table(&mut table);
    (logits, shared)
}

/// The same prompt twice: the second run must share, and must answer the same.
#[test]
fn a_shared_prefix_answers_what_a_computed_one_answers() {
    let (model, prompt, tolerance) = load_tiny();
    let (cache, mut allocator) = pool(&model, 64);

    let (first, shared_first) = run(&model, &cache, &mut allocator, &prompt, true);
    assert_eq!(shared_first, 0, "nothing to share on the way in");

    let (second, shared_second) = run(&model, &cache, &mut allocator, &prompt, true);
    assert!(
        shared_second > 0,
        "the second run found the first one's blocks"
    );
    assert_eq!(allocator.hits(), (shared_second / BLOCK) as u64);

    let (worst, scale) = compare(&second, &first);
    assert!(
        f64::from(worst) <= tolerance,
        "sharing changed the answer, off by {worst:.3e} on values up to {scale:.3e}"
    );
}

/// Two prompts that agree and then part, run with and without sharing.
///
/// The second prompt takes the first's leading blocks and then computes its own
/// tail. If the chain were wrong it would keep reading the first prompt's
/// continuation, which is exactly the failure that stays plausible.
#[test]
fn sharing_a_common_head_does_not_leak_the_other_tail() {
    let (model, prompt, tolerance) = load_tiny();
    let head = &prompt[..4];
    let mut a: Vec<u32> = head.to_vec();
    a.extend([11u32, 42, 3, 64, 7]);
    let mut b: Vec<u32> = head.to_vec();
    b.extend([90u32, 91, 92, 93, 94]);

    // With sharing: a runs, then b takes a's first block and computes its own.
    let (cache, mut allocator) = pool(&model, 64);
    run(&model, &cache, &mut allocator, &a, true);
    let (with, shared) = run(&model, &cache, &mut allocator, &b, true);
    assert_eq!(shared, BLOCK, "they agree on exactly one block");

    // Without sharing: b alone, in a pool nothing else touched.
    let (fresh_cache, mut fresh) = pool(&model, 64);
    let (without, none) = run(&model, &fresh_cache, &mut fresh, &b, false);
    assert_eq!(none, 0);

    let (worst, scale) = compare(&with, &without);
    assert!(
        f64::from(worst) <= tolerance,
        "the shared head changed b's answer, off by {worst:.3e} on values up to {scale:.3e}"
    );
}

/// The same four tokens after a different prefix are a different block.
///
/// This is what the chained hash is for, and it is the one property a hash over
/// the block's own tokens would not have.
#[test]
fn identical_tokens_after_a_different_prefix_are_not_shared() {
    let (model, _, tolerance) = load_tiny();
    let tail = [11u32, 42, 3, 64];
    let mut a: Vec<u32> = vec![1, 2, 3, 4];
    a.extend(tail);
    a.extend([7u32]);
    let mut b: Vec<u32> = vec![90, 91, 92, 93];
    b.extend(tail);
    b.extend([7u32]);

    let (cache, mut allocator) = pool(&model, 64);
    run(&model, &cache, &mut allocator, &a, true);
    let before = allocator.hits();
    let (with, shared) = run(&model, &cache, &mut allocator, &b, true);
    assert_eq!(
        shared, 0,
        "b's first block differs, so its second cannot be taken either"
    );
    assert_eq!(allocator.hits(), before);

    let (fresh_cache, mut fresh) = pool(&model, 64);
    let (without, _) = run(&model, &fresh_cache, &mut fresh, &b, false);
    let (worst, _) = compare(&with, &without);
    assert!(f64::from(worst) <= tolerance);
}

/// The chain matters at the second block, which takes three prompts to reach.
///
/// A walk stops at the first block nobody has, so two prompts that differ from
/// their first block never reach the one where chaining decides anything: the
/// earlier test passes whether or not the parent is mixed in. What exposes it is
/// a prompt whose first block hits and whose second block holds tokens some
/// other prompt already published after a different beginning. Without the
/// chain, both name that second block the same, and this one would read keys
/// computed in a context it never had.
#[test]
fn the_second_block_of_a_prompt_is_not_confused_with_another_prompts_second() {
    let (model, _, tolerance) = load_tiny();
    let shared_middle = [11u32, 42, 3, 64];
    let head_a: [u32; 4] = [1, 2, 3, 4];
    let head_c: [u32; 4] = [90, 91, 92, 93];

    let build = |head: [u32; 4], middle: [u32; 4]| {
        let mut prompt = head.to_vec();
        prompt.extend(middle);
        prompt.extend([7u32]);
        prompt
    };
    // Published first, after a different head, so its second block carries the
    // middle tokens in the wrong context for the prompt below.
    let other = build(head_c, shared_middle);
    // Publishes the head the prompt under test needs, so its walk gets past the
    // first block and reaches the second.
    let primer = build(head_a, [50, 51, 52, 53]);
    let under_test = build(head_a, shared_middle);

    let (cache, mut allocator) = pool(&model, 64);
    run(&model, &cache, &mut allocator, &other, true);
    let (_, from_primer) = run(&model, &cache, &mut allocator, &primer, true);
    assert_eq!(
        from_primer, 0,
        "the primer shares nothing, it only publishes"
    );

    let (with, shared) = run(&model, &cache, &mut allocator, &under_test, true);
    assert_eq!(
        shared, BLOCK,
        "its head is cached and its middle is not, so exactly one block is taken"
    );

    let (fresh_cache, mut fresh) = pool(&model, 64);
    let (without, _) = run(&model, &fresh_cache, &mut fresh, &under_test, false);
    let (worst, scale) = compare(&with, &without);
    assert!(
        f64::from(worst) <= tolerance,
        "it read another prompt's second block, off by {worst:.3e} on {scale:.3e}"
    );
}

/// A block whose name was dropped is recomputed rather than misread.
#[test]
fn a_prompt_still_answers_correctly_after_its_blocks_were_evicted() {
    let (model, prompt, tolerance) = load_tiny();
    // Three blocks against two prompts of two, so the second one takes the
    // untouched block and then has to drop a name to finish.
    let (cache, mut allocator) = pool(&model, 3);
    let (first, _) = run(&model, &cache, &mut allocator, &prompt, true);

    // Kept inside the fixture's 128-token vocabulary; anything past it is not a
    // different prompt, it is an index error.
    let other: Vec<u32> = prompt.iter().map(|t| (t + 50) % 128).collect();
    run(&model, &cache, &mut allocator, &other, true);
    assert!(allocator.evictions() > 0, "the pool had to drop names");

    let (again, _) = run(&model, &cache, &mut allocator, &prompt, true);
    let (worst, scale) = compare(&again, &first);
    assert!(
        f64::from(worst) <= tolerance,
        "recomputing after eviction changed the answer, off by {worst:.3e} on {scale:.3e}"
    );
}
