//! Batching and paging have to be invisible in the answers.
//!
//! A sequence decoded next to others must produce exactly what it produces
//! alone, and a sequence whose history is scattered across blocks must produce
//! exactly what it produces in one run of memory. Everything these two stages
//! add can break that quietly: a mask that lets one row read another's padding,
//! positions taken from the wrong row, a block table resolved to the wrong slot.
//! None of them stop the model producing logits, and none are visible in a
//! single sequence held in one block.

use candle_core::{DType, Device, Tensor};
use pagedllm::{Batch, BlockAllocator, BlockTable, CacheConfig, Model, PagedCache};

mod common;
use common::{compare, load_tiny};

/// A pool at the block size under test, and its free list.
fn pool(model: &Model, block_size: usize, blocks: usize) -> (PagedCache, BlockAllocator) {
    let config = model.config();
    let cache = PagedCache::new(
        CacheConfig {
            block_size,
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

/// Give `table` enough blocks for `tokens` more.
fn grow(allocator: &mut BlockAllocator, table: &mut BlockTable, tokens: usize) {
    for _ in 0..table.blocks_needed(tokens) {
        table.push(allocator.allocate().expect("the pool has room"));
    }
}

/// Run one sequence alone in its own pool, feeding it `feed` after its prompt.
///
/// The tokens are given rather than sampled. Letting each path pick its own
/// greedy token would compare two chains instead of two implementations: this
/// fixture model has a near-uniform output over 128 tokens, so a tie decided
/// differently by a rounding bit sends the chains apart from that step on.
fn alone(model: &Model, block_size: usize, prompt: &[u32], feed: &[u32]) -> Vec<Tensor> {
    let (cache, mut allocator) = pool(model, block_size, 64);
    let mut table = BlockTable::new(block_size);
    grow(&mut allocator, &mut table, prompt.len());

    let batch = Batch::new(prompt.to_vec(), prompt.len(), &[&table], block_size).unwrap();
    let mut logits = vec![model.forward_batch(&batch, &cache).unwrap()];
    table.advance(prompt.len()).unwrap();

    for &token in feed {
        grow(&mut allocator, &mut table, 1);
        let batch = Batch::new(vec![token], 1, &[&table], block_size).unwrap();
        logits.push(model.forward_batch(&batch, &cache).unwrap());
        table.advance(1).unwrap();
    }
    logits
}

#[test]
fn a_batched_prefill_lands_where_the_uncached_path_lands() {
    let (model, prompt, tolerance) = load_tiny();
    let whole = model.forward(&prompt).unwrap();
    let want = whole
        .narrow(1, prompt.len() - 1, 1)
        .unwrap()
        .reshape((1, ()))
        .unwrap();

    let (cache, mut allocator) = pool(&model, 4, 32);
    let mut table = BlockTable::new(4);
    grow(&mut allocator, &mut table, prompt.len());
    let batch = Batch::new(prompt.clone(), prompt.len(), &[&table], 4).unwrap();
    let got = model.forward_batch(&batch, &cache).unwrap();

    let (worst, scale) = compare(&got, &want);
    assert!(
        f64::from(worst) <= tolerance,
        "off by {worst:.3e} on values up to {scale:.3e}"
    );
}

/// The whole point of paging: where a sequence's history sits must not matter.
///
/// A block of four spreads the same history over several blocks, where a block
/// as wide as the context keeps it in one. Both have to answer identically.
#[test]
fn a_history_scattered_across_blocks_answers_what_one_run_of_memory_answers() {
    let (model, prompt, tolerance) = load_tiny();
    let feed = [11u32, 42, 3, 64];

    // A block as wide as the context is one block a sequence, which is the
    // reservation stage 3 measured.
    let contiguous = alone(&model, 64, &prompt, &feed);
    let paged = alone(&model, 4, &prompt, &feed);

    for (step, (want, got)) in contiguous.iter().zip(&paged).enumerate() {
        let (worst, scale) = compare(got, want);
        assert!(
            f64::from(worst) <= tolerance,
            "step {step} off by {worst:.3e} on values up to {scale:.3e}"
        );
    }
}

#[test]
fn two_sequences_decoded_together_answer_what_they_answer_apart() {
    let (model, prompt, tolerance) = load_tiny();
    // Different lengths on purpose: equal ones make the batch rectangle exact,
    // so no row ever reads the padding a shorter neighbour leaves behind, and
    // the mask that hides it is never exercised.
    let (first, second) = (&prompt[..5], &prompt[2..]);
    assert_ne!(first.len(), second.len());
    let feed = [11u32, 42, 3, 64];
    let block_size = 4;

    let apart = [
        alone(&model, block_size, first, &feed),
        alone(&model, block_size, second, &feed),
    ];

    let (cache, mut allocator) = pool(&model, block_size, 64);
    let mut tables = [BlockTable::new(block_size), BlockTable::new(block_size)];

    // Prefills run one at a time, which is what the scheduler does: a batch is a
    // rectangle, and two prompts of different lengths are not one. Allocating in
    // turn is also what interleaves the two sequences' blocks.
    let mut together = Vec::new();
    for (tokens, table) in [first, second].iter().zip(&mut tables) {
        grow(&mut allocator, table, tokens.len());
        let batch = Batch::new(tokens.to_vec(), tokens.len(), &[table], block_size).unwrap();
        together.push(model.forward_batch(&batch, &cache).unwrap());
        table.advance(tokens.len()).unwrap();
    }
    assert_ne!(
        tables[0].blocks(),
        tables[1].blocks(),
        "the two sequences must not hold the same blocks"
    );

    let mut batched = Vec::new();
    for &token in &feed {
        for table in &mut tables {
            grow(&mut allocator, table, 1);
        }
        let refs: Vec<&BlockTable> = tables.iter().collect();
        assert_ne!(
            refs[0].tokens(),
            refs[1].tokens(),
            "the rows must sit at different lengths"
        );
        let batch = Batch::new(vec![token; 2], 1, &refs, block_size).unwrap();
        batched.push(model.forward_batch(&batch, &cache).unwrap());
        for table in &mut tables {
            table.advance(1).unwrap();
        }
    }

    let mut failures = Vec::new();
    for (row, reference) in apart.iter().enumerate() {
        let (worst, _) = compare(&together[row], &reference[0]);
        if f64::from(worst) > tolerance {
            failures.push(format!("row {row} prefill off by {worst:.3e}"));
        }
        for (step, logits) in batched.iter().enumerate() {
            let got = logits.narrow(0, row, 1).unwrap();
            let (worst, scale) = compare(&got, &reference[step + 1]);
            if f64::from(worst) > tolerance {
                failures.push(format!(
                    "row {row} step {step} off by {worst:.3e} on values up to {scale:.3e}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn feeding_tokens_one_at_a_time_lands_where_one_pass_does() {
    let (model, prompt, tolerance) = load_tiny();
    let block_size = 4;

    let (cache, mut allocator) = pool(&model, block_size, 64);
    let mut whole = BlockTable::new(block_size);
    grow(&mut allocator, &mut whole, prompt.len());
    let want = model
        .forward_batch(
            &Batch::new(prompt.clone(), prompt.len(), &[&whole], block_size).unwrap(),
            &cache,
        )
        .unwrap();

    let one_by_one = alone(&model, block_size, &prompt[..1], &prompt[1..]);
    let (worst, scale) = compare(one_by_one.last().unwrap(), &want);
    assert!(
        f64::from(worst) <= tolerance,
        "off by {worst:.3e} on values up to {scale:.3e}"
    );
}

#[test]
fn a_finished_sequence_gives_its_blocks_back_and_the_next_one_starts_clean() {
    let (model, prompt, tolerance) = load_tiny();
    let block_size = 4;
    // Exactly enough blocks for one sequence, so the second can only run if the
    // first really gave them back.
    let blocks = prompt.len().div_ceil(block_size);
    let (cache, mut allocator) = pool(&model, block_size, blocks);

    let mut table = BlockTable::new(block_size);
    grow(&mut allocator, &mut table, prompt.len());
    let batch = Batch::new(prompt.clone(), prompt.len(), &[&table], block_size).unwrap();
    let first = model.forward_batch(&batch, &cache).unwrap();
    table.advance(prompt.len()).unwrap();
    assert_eq!(allocator.available(), 0);

    allocator.free_table(&mut table);
    assert_eq!(allocator.available(), blocks);

    // The same prompt through the same blocks has to answer the same thing.
    // Anything left behind would be read as history by the second sequence.
    let mut reused = BlockTable::new(block_size);
    grow(&mut allocator, &mut reused, prompt.len());
    let batch = Batch::new(prompt.clone(), prompt.len(), &[&reused], block_size).unwrap();
    let second = model.forward_batch(&batch, &cache).unwrap();

    let (worst, _) = compare(&second, &first);
    assert!(
        f64::from(worst) <= tolerance,
        "the reused blocks answered differently, off by {worst:.3e}"
    );
}

#[test]
fn a_batch_with_no_block_for_its_next_token_is_refused() {
    let (model, prompt, _) = load_tiny();
    let block_size = 4;
    let (_cache, mut allocator) = pool(&model, block_size, 32);
    let mut table = BlockTable::new(block_size);
    grow(&mut allocator, &mut table, prompt.len());
    table.advance(prompt.len()).unwrap();

    // The table holds exactly the prompt. Placing one more token without giving
    // it a block has nowhere to go.
    if table.blocks_needed(1) > 0 {
        assert!(Batch::new(vec![1], 1, &[&table], block_size).is_err());
    }
    // A row count that does not line up is refused before anything runs.
    assert!(Batch::new(vec![1, 2], 1, &[&table], block_size).is_err());
}

/// Feed `prompt` through in slices of `slice`, returning the logits of the last
/// pass. A middle slice asks for nothing, which is what the scheduler does: a
/// prompt produces no token until its final token has run.
fn in_slices(model: &Model, block_size: usize, prompt: &[u32], slice: usize) -> Tensor {
    let (cache, mut allocator) = pool(model, block_size, 64);
    let mut table = BlockTable::new(block_size);

    let mut last = None;
    let mut done = 0;
    while done < prompt.len() {
        let take = slice.min(prompt.len() - done);
        grow(&mut allocator, &mut table, take);
        let entries: Vec<(u32, &BlockTable, usize)> = (done..done + take)
            .map(|position| (prompt[position], &table, position))
            .collect();
        let predicts: Vec<usize> = if done + take == prompt.len() {
            vec![take - 1]
        } else {
            Vec::new()
        };
        let batch = Batch::unfolded(&entries, &predicts, block_size).unwrap();
        let logits = model.forward_batch(&batch, &cache).unwrap();
        if predicts.is_empty() {
            assert_eq!(
                logits.dim(0).unwrap(),
                0,
                "a middle slice produced logits nobody asked for"
            );
        } else {
            last = Some(logits);
        }
        table.advance(take).unwrap();
        done += take;
    }
    last.expect("the last slice asks for its logits")
}

#[test]
fn a_prompt_fed_in_slices_lands_where_one_pass_over_it_lands() {
    let (model, prompt, tolerance) = load_tiny();
    let whole = model.forward(&prompt).unwrap();
    let want = whole
        .narrow(1, prompt.len() - 1, 1)
        .unwrap()
        .reshape((1, ()))
        .unwrap();

    // Slices that do and do not line up with the block size, because a boundary
    // landing mid-block is where a position offset goes wrong unnoticed.
    for slice in [1, 3, 4, 7, prompt.len()] {
        let got = in_slices(&model, 4, &prompt, slice);
        let (worst, scale) = compare(&got, &want);
        assert!(
            f64::from(worst) <= tolerance,
            "fed {slice} at a time: off by {worst:.3e} on values up to {scale:.3e}"
        );
    }
}
