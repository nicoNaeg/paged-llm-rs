//! Batching has to be invisible in the answers.
//!
//! A sequence decoded next to others must produce exactly what it produces
//! alone. Everything stage 3 adds can break that quietly: a mask that lets one
//! row read another's padding, positions taken from the wrong row, a slot
//! written at the offset of its neighbour. None of them stop the model
//! producing logits, and none of them are visible in a single sequence.

use candle_core::{DType, Device, Tensor};
use pagedllm::{Batch, CacheConfig, Model, SlotCache};

mod common;
use common::{compare, load_tiny};

fn cache_for(model: &Model, slots: usize, max_seq: usize) -> SlotCache {
    let config = model.config();
    SlotCache::new(
        CacheConfig {
            slots,
            max_seq,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            layers: config.num_hidden_layers,
        },
        DType::F32,
        &Device::Cpu,
    )
    .unwrap()
}

/// Run one sequence alone in its own pool, feeding it `feed` after its prompt.
///
/// The tokens are given rather than sampled. Letting each path pick its own
/// greedy token would compare two chains instead of two implementations: this
/// fixture model has a near-uniform output over 128 tokens, so a tie decided
/// differently by a rounding bit sends the chains apart from that step on, and
/// the test would report a divergence that says nothing about batching.
fn alone(model: &Model, prompt: &[u32], feed: &[u32]) -> Vec<Tensor> {
    let mut cache = cache_for(model, 1, 64);
    let slot = cache.acquire().unwrap();
    let mut logits = vec![
        model
            .forward_batch(&Batch::prefill(prompt.to_vec(), slot, 0), &cache)
            .unwrap(),
    ];
    cache.advance(&[slot], prompt.len());
    for &token in feed {
        let start = cache.length(slot);
        logits.push(
            model
                .forward_batch(&Batch::decode(vec![token], vec![slot], vec![start]), &cache)
                .unwrap(),
        );
        cache.advance(&[slot], 1);
    }
    logits
}

#[test]
fn a_batched_prefill_lands_where_the_single_sequence_path_lands() {
    let (model, prompt, tolerance) = load_tiny();
    let whole = model.forward(&prompt).unwrap();
    let want = whole
        .narrow(1, prompt.len() - 1, 1)
        .unwrap()
        .reshape((1, ()))
        .unwrap();

    let mut cache = cache_for(&model, 1, 32);
    let slot = cache.acquire().unwrap();
    let got = model
        .forward_batch(&Batch::prefill(prompt.clone(), slot, 0), &cache)
        .unwrap();
    // The pass writes the slot; recording that it happened is the caller's job,
    // which in the server is the scheduler's commit.
    cache.advance(&[slot], prompt.len());
    assert_eq!(cache.length(slot), prompt.len());
    let (worst, scale) = compare(&got, &want);
    assert!(
        f64::from(worst) <= tolerance,
        "off by {worst:.3e} on values up to {scale:.3e}"
    );
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

    let apart = [alone(&model, first, &feed), alone(&model, second, &feed)];

    let mut cache = cache_for(&model, 2, 64);
    let slots = [cache.acquire().unwrap(), cache.acquire().unwrap()];

    // Prefills run one at a time, which is what the scheduler does: a batch is a
    // rectangle, and two prompts of different lengths are not one.
    let mut together = Vec::new();
    for (tokens, slot) in [first, second].iter().zip(slots) {
        together.push(
            model
                .forward_batch(&Batch::prefill(tokens.to_vec(), slot, 0), &cache)
                .unwrap(),
        );
        cache.advance(&[slot], tokens.len());
    }

    let mut batched = Vec::new();
    for &token in &feed {
        let starts: Vec<usize> = slots.iter().map(|&s| cache.length(s)).collect();
        assert_ne!(
            starts[0], starts[1],
            "the rows must sit at different lengths"
        );
        let logits = model
            .forward_batch(
                &Batch::decode(vec![token; 2], slots.to_vec(), starts),
                &cache,
            )
            .unwrap();
        cache.advance(&slots, 1);
        batched.push(logits);
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

/// Feeding a prompt one token at a time has to land where one pass lands.
///
/// The invariant the cache exists under, and the one that breaks quietly: a
/// rotation applied at the wrong offset, a mask that forgot the history, or keys
/// written at a neighbour's position all still produce logits.
#[test]
fn feeding_tokens_one_at_a_time_lands_where_one_pass_does() {
    let (model, prompt, tolerance) = load_tiny();
    let mut whole_cache = cache_for(&model, 1, 32);
    let whole_slot = whole_cache.acquire().unwrap();
    let want = model
        .forward_batch(&Batch::prefill(prompt.clone(), whole_slot, 0), &whole_cache)
        .unwrap();
    whole_cache.advance(&[whole_slot], prompt.len());

    let mut cache = cache_for(&model, 1, 32);
    let slot = cache.acquire().unwrap();
    let mut got = None;
    for (position, token) in prompt.iter().enumerate() {
        got = Some(
            model
                .forward_batch(
                    &Batch::decode(vec![*token], vec![slot], vec![position]),
                    &cache,
                )
                .unwrap(),
        );
        cache.advance(&[slot], 1);
        assert_eq!(cache.length(slot), position + 1);
    }

    let (worst, scale) = compare(&got.unwrap(), &want);
    assert!(
        f64::from(worst) <= tolerance,
        "off by {worst:.3e} on values up to {scale:.3e}"
    );
}

/// The same, in two uneven chunks, which is a prefill followed by decoding.
#[test]
fn a_prompt_split_into_chunks_lands_where_one_pass_does() {
    let (model, prompt, tolerance) = load_tiny();
    let mut whole_cache = cache_for(&model, 1, 32);
    let whole_slot = whole_cache.acquire().unwrap();
    let want = model
        .forward_batch(&Batch::prefill(prompt.clone(), whole_slot, 0), &whole_cache)
        .unwrap();

    let mut cache = cache_for(&model, 1, 32);
    let slot = cache.acquire().unwrap();
    let split = prompt.len() - 3;
    model
        .forward_batch(&Batch::prefill(prompt[..split].to_vec(), slot, 0), &cache)
        .unwrap();
    cache.advance(&[slot], split);

    let mut got = None;
    for (offset, token) in prompt[split..].iter().enumerate() {
        got = Some(
            model
                .forward_batch(
                    &Batch::decode(vec![*token], vec![slot], vec![split + offset]),
                    &cache,
                )
                .unwrap(),
        );
        cache.advance(&[slot], 1);
    }

    let (worst, scale) = compare(&got.unwrap(), &want);
    assert!(
        f64::from(worst) <= tolerance,
        "off by {worst:.3e} on values up to {scale:.3e}"
    );
}

#[test]
fn a_finished_sequence_gives_its_slot_back_and_the_next_one_starts_clean() {
    let (model, prompt, tolerance) = load_tiny();
    let mut cache = cache_for(&model, 1, 32);

    let slot = cache.acquire().unwrap();
    let first = model
        .forward_batch(&Batch::prefill(prompt.clone(), slot, 0), &cache)
        .unwrap();
    cache.advance(&[slot], prompt.len());
    assert_eq!(cache.free_slots(), 0, "the only slot is held");
    cache.release(slot);
    assert_eq!(cache.free_slots(), 1);

    // The same prompt into the same slot has to answer the same thing. Anything
    // left behind by the first sequence would be read as history by the second.
    let reused = cache.acquire().unwrap();
    assert_eq!(reused, slot);
    let second = model
        .forward_batch(&Batch::prefill(prompt.clone(), reused, 0), &cache)
        .unwrap();

    let (worst, _) = compare(&second, &first);
    assert!(
        f64::from(worst) <= tolerance,
        "the reused slot answered differently, off by {worst:.3e}"
    );
}

#[test]
fn a_batch_that_disagrees_with_the_cache_is_refused() {
    let (model, prompt, _) = load_tiny();
    let mut cache = cache_for(&model, 2, 32);
    let slot = cache.acquire().unwrap();
    model
        .forward_batch(&Batch::prefill(prompt.clone(), slot, 0), &cache)
        .unwrap();
    cache.advance(&[slot], prompt.len());

    // Claiming the slot is empty when it holds the prompt would put the new
    // token on top of the first one and rotate it to position zero.
    assert!(
        model
            .forward_batch(&Batch::decode(vec![1], vec![slot], vec![0]), &cache)
            .is_err()
    );
    // Row counts that do not line up are refused before anything is dispatched.
    assert!(
        model
            .forward_batch(
                &Batch::decode(vec![1, 2], vec![slot], vec![prompt.len()]),
                &cache
            )
            .is_err()
    );
}
