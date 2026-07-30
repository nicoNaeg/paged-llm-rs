//! The paged attention kernel, against the two things that can check it.
//!
//! Three implementations of one function exist, and each pair says something the
//! other pair cannot. The tensor path gathers the blocks into a rectangle and
//! multiplies; it came from stage 3 and knows nothing about block tables. The
//! scalar path walks the table token by token; it is the oracle, and it is what
//! CI runs, because a hosted macOS runner has no Metal device. The kernel does
//! what the scalar path does, in parallel, on the GPU.
//!
//! Comparing only two of them would miss a defect that moved both the same way,
//! which is a mistake this project has already made: a reshape that split every
//! key vector across head boundaries passed two differential tests because both
//! sides were wrong identically.

use candle_core::{DType, Device, Tensor};
use pagedllm::{AttentionKind, Batch, BlockAllocator, BlockTable, CacheConfig, Model, PagedCache};

mod common;
use common::{compare, load_tiny};

/// Give `table` enough blocks for `tokens` more.
fn grow(allocator: &mut BlockAllocator, table: &mut BlockTable, tokens: usize) {
    for _ in 0..table.blocks_needed(tokens) {
        table.push(allocator.allocate().expect("the pool has room"));
    }
}

/// Decode `feed` after `prompt`, on a model set to `kind`, and return the logits
/// of each step.
fn decode(model: &Model, block_size: usize, prompt: &[u32], feed: &[u32]) -> Vec<Tensor> {
    let config = model.config();
    let cache = PagedCache::new(
        CacheConfig {
            block_size,
            blocks: 64,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            layers: config.num_hidden_layers,
        },
        model.dtype(),
        model.device(),
    )
    .unwrap();
    let mut allocator = BlockAllocator::new(64);
    let mut table = BlockTable::new(block_size);
    grow(&mut allocator, &mut table, prompt.len());

    // The prompt is a prefill, which keeps the tensor path whatever the flag
    // says: the kernel decodes one token a row.
    let batch = Batch::new(prompt.to_vec(), prompt.len(), &[&table], block_size).unwrap();
    model.forward_batch(&batch, &cache).unwrap();
    table.advance(prompt.len()).unwrap();

    let mut logits = Vec::new();
    for &token in feed {
        grow(&mut allocator, &mut table, 1);
        let batch = Batch::new(vec![token], 1, &[&table], block_size).unwrap();
        logits.push(model.forward_batch(&batch, &cache).unwrap());
        table.advance(1).unwrap();
    }
    logits
}

/// Several sequences at different lengths, decoded together.
fn decode_batch(model: &Model, block_size: usize, prompts: &[&[u32]], feed: &[u32]) -> Vec<Tensor> {
    let config = model.config();
    let cache = PagedCache::new(
        CacheConfig {
            block_size,
            blocks: 128,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            layers: config.num_hidden_layers,
        },
        model.dtype(),
        model.device(),
    )
    .unwrap();
    let mut allocator = BlockAllocator::new(128);
    let mut tables: Vec<BlockTable> = prompts
        .iter()
        .map(|_| BlockTable::new(block_size))
        .collect();

    for (prompt, table) in prompts.iter().zip(&mut tables) {
        grow(&mut allocator, table, prompt.len());
        let batch = Batch::new(prompt.to_vec(), prompt.len(), &[table], block_size).unwrap();
        model.forward_batch(&batch, &cache).unwrap();
        table.advance(prompt.len()).unwrap();
    }

    let mut logits = Vec::new();
    for &token in feed {
        for table in &mut tables {
            grow(&mut allocator, table, 1);
        }
        let refs: Vec<&BlockTable> = tables.iter().collect();
        let batch = Batch::new(vec![token; refs.len()], 1, &refs, block_size).unwrap();
        logits.push(model.forward_batch(&batch, &cache).unwrap());
        for table in &mut tables {
            table.advance(1).unwrap();
        }
    }
    logits
}

/// The scalar reference against the tensor path, which is what CI checks.
///
/// One walks a block table token by token in a loop; the other gathers the
/// blocks into a rectangle and hands it to a batched multiply. They share no
/// code, so agreeing is evidence rather than a tautology.
#[test]
fn the_scalar_reference_agrees_with_the_tensor_path() {
    let (mut model, prompt, tolerance) = load_tiny();
    let feed = [11u32, 42, 3, 64];
    let block_size = 4;

    model.set_attention(AttentionKind::Tensor);
    let tensor = decode(&model, block_size, &prompt, &feed);
    model.set_attention(AttentionKind::Kernel);
    let scalar = decode(&model, block_size, &prompt, &feed);

    assert_eq!(model.attention(), AttentionKind::Kernel);
    for (step, (want, got)) in tensor.iter().zip(&scalar).enumerate() {
        let (worst, scale) = compare(got, want);
        assert!(
            f64::from(worst) <= tolerance,
            "step {step} off by {worst:.3e} on values up to {scale:.3e}"
        );
    }
}

/// The same, with several sequences at different lengths in one pass.
///
/// A single row never exercises the part of the kernel that keeps one row's
/// context from reaching into another's, because there is no other.
#[test]
fn the_two_paths_agree_when_rows_sit_at_different_lengths() {
    let (mut model, prompt, tolerance) = load_tiny();
    let prompts: Vec<&[u32]> = vec![&prompt[..5], &prompt[2..], &prompt[..3]];
    let feed = [11u32, 42, 3];
    let block_size = 4;

    model.set_attention(AttentionKind::Tensor);
    let tensor = decode_batch(&model, block_size, &prompts, &feed);
    model.set_attention(AttentionKind::Kernel);
    let scalar = decode_batch(&model, block_size, &prompts, &feed);

    let mut failures = Vec::new();
    for (step, (want, got)) in tensor.iter().zip(&scalar).enumerate() {
        for row in 0..prompts.len() {
            let (worst, scale) = compare(
                &got.narrow(0, row, 1).unwrap(),
                &want.narrow(0, row, 1).unwrap(),
            );
            if f64::from(worst) > tolerance {
                failures.push(format!(
                    "step {step} row {row} off by {worst:.3e} on values up to {scale:.3e}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A history spread over many small blocks against one held in a single block.
///
/// The kernel resolves every position through the table, so this is the check
/// that its indirection is right rather than accidentally linear.
#[test]
fn the_kernel_does_not_care_where_a_history_sits() {
    let (mut model, prompt, tolerance) = load_tiny();
    let feed = [11u32, 42, 3, 64];
    model.set_attention(AttentionKind::Kernel);

    let contiguous = decode(&model, 64, &prompt, &feed);
    let paged = decode(&model, 2, &prompt, &feed);

    for (step, (want, got)) in contiguous.iter().zip(&paged).enumerate() {
        let (worst, scale) = compare(got, want);
        assert!(
            f64::from(worst) <= tolerance,
            "step {step} off by {worst:.3e} on values up to {scale:.3e}"
        );
    }
}

/// A prefill keeps the tensor path whatever the flag says.
#[test]
fn the_kernel_refuses_a_batch_that_is_not_a_decode() {
    let (model, prompt, _) = load_tiny();
    let config = model.config();
    let block_size = 4;
    let cache = PagedCache::new(
        CacheConfig {
            block_size,
            blocks: 32,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            layers: config.num_hidden_layers,
        },
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let mut allocator = BlockAllocator::new(32);
    let mut table = BlockTable::new(block_size);
    grow(&mut allocator, &mut table, prompt.len());
    let batch = Batch::new(prompt.clone(), prompt.len(), &[&table], block_size).unwrap();

    let (k_pool, v_pool) = cache.layer(0);
    assert!(
        pagedllm::PagedAttention::new(
            k_pool,
            v_pool,
            &batch,
            block_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            1.0,
        )
        .is_err(),
        "a prefill is a different kernel shape and is refused rather than run"
    );
}

/// The kernel on the GPU against the tensor path on the GPU.
///
/// This is the comparison the scalar reference cannot make, because it runs the
/// scalar code rather than the dispatched one. CI cannot make it either:
/// `MTLCreateSystemDefaultDevice` returns nil inside a hosted macOS runner, so
/// this only runs under `make test-metal`, on hardware.
#[cfg(feature = "metal")]
#[test]
fn the_kernel_agrees_with_the_tensor_path_on_the_gpu() {
    let Ok(device) = Device::new_metal(0) else {
        eprintln!("skipped: no Metal device");
        return;
    };
    let (_, prompt, tolerance) = load_tiny();
    let mut model = Model::load(common::fixture_dir(), &device).expect("load on metal");
    let prompts: Vec<&[u32]> = vec![&prompt[..5], &prompt[2..], &prompt[..3]];
    let feed = [11u32, 42, 3];
    let block_size = 4;

    model.set_attention(AttentionKind::Tensor);
    let tensor = decode_batch(&model, block_size, &prompts, &feed);
    model.set_attention(AttentionKind::Kernel);
    let kernel = decode_batch(&model, block_size, &prompts, &feed);

    let mut worst_seen = 0f32;
    let mut failures = Vec::new();
    for (step, (want, got)) in tensor.iter().zip(&kernel).enumerate() {
        for row in 0..prompts.len() {
            let (worst, scale) = compare(
                &got.narrow(0, row, 1).unwrap(),
                &want.narrow(0, row, 1).unwrap(),
            );
            worst_seen = worst_seen.max(worst);
            if f64::from(worst) > tolerance {
                failures.push(format!(
                    "step {step} row {row} off by {worst:.3e} on values up to {scale:.3e}"
                ));
            }
        }
    }
    println!("kernel against the tensor path on metal: worst {worst_seen:.3e}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A pass that asks for no logits at all, on the GPU.
///
/// This is what a slice in the middle of a prompt runs: it goes through the
/// model only to fill the cache, and produces nothing until its last token has
/// gone through. What this checks is that the pass runs at all on the GPU and
/// that the cache it filled is real, by predicting from it afterwards and
/// comparing against the same prompt run whole.
///
/// It does not check what actually broke, and that is worth writing down. The
/// server answered 500 on the first request whose prompt was longer than the
/// budget with nothing else running, because casting a zero-row tensor to f32
/// dispatches a Metal kernel over zero elements and candle divides by zero
/// working out its grid. This fixture is f32, so the cast is a no-op and the
/// crash cannot happen here. The rule the engine follows is that nothing may
/// touch the result of a pass that asked for nothing, and the check that would
/// have caught its breaking is over HTTP, in `scripts/smoke-server.py`.
#[cfg(feature = "metal")]
#[test]
fn a_pass_that_asks_for_no_logits_runs_on_the_gpu_and_fills_the_cache() {
    let Ok(device) = Device::new_metal(0) else {
        eprintln!("skipped: no Metal device");
        return;
    };
    let (_, prompt, tolerance) = load_tiny();
    let mut model = Model::load(common::fixture_dir(), &device).expect("load on metal");
    model.set_attention(AttentionKind::Kernel);
    let block_size = 4;

    let config = model.config();
    let cache = PagedCache::new(
        CacheConfig {
            block_size,
            blocks: 64,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            layers: config.num_hidden_layers,
        },
        model.dtype(),
        &device,
    )
    .expect("a pool on metal");
    let mut allocator = BlockAllocator::new(64);
    let mut table = BlockTable::new(block_size);

    // Every token but the last, asking for nothing.
    let head = prompt.len() - 1;
    grow(&mut allocator, &mut table, head);
    let entries: Vec<(u32, &BlockTable, usize)> =
        (0..head).map(|at| (prompt[at], &table, at)).collect();
    let batch = Batch::unfolded(&entries, &[], block_size).expect("a batch with no predictions");
    let empty = model
        .forward_batch(&batch, &cache)
        .expect("a pass asking for nothing still runs");
    assert_eq!(empty.dim(0).unwrap(), 0, "it produced logits nobody wanted");
    table.advance(head).unwrap();

    // The last token, which is the one that predicts.
    grow(&mut allocator, &mut table, 1);
    let last = [(prompt[head], &table, head)];
    let batch = Batch::unfolded(&last, &[0], block_size).expect("the predicting batch");
    let got = model
        .forward_batch(&batch, &cache)
        .expect("the last token runs");

    let whole = model.forward(&prompt).expect("the same prompt in one pass");
    let want = whole
        .narrow(1, prompt.len() - 1, 1)
        .unwrap()
        .reshape((1, ()))
        .unwrap();
    let (worst, scale) = compare(&got, &want);
    assert!(
        f64::from(worst) <= tolerance,
        "the cache the silent pass filled is wrong: off by {worst:.3e} on values up to {scale:.3e}"
    );
}
