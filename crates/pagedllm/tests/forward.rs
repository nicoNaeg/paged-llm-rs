//! The forward pass against the reference implementation, module by module.
//!
//! The fixture under `tests/fixtures/tiny` is dumped by
//! `scripts/dump_reference.py` from `HuggingFace` transformers. Comparing only
//! the logits would say "wrong" and stop there; comparing every module boundary
//! says which one, which is what makes a mistake in `RoPE` distinguishable from
//! one in the head grouping.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use pagedllm::{Config, Model, Trace, Weights};

mod common;
use common::{compare, fixture_dir, load_tiny};

fn reference(dir: &Path) -> HashMap<String, Tensor> {
    candle_core::safetensors::load(dir.join("activations.safetensors"), &Device::Cpu)
        .expect("fixture activations")
}

#[test]
fn every_module_matches_the_reference_implementation() {
    let (model, prompt, tolerance) = load_tiny();
    let want = reference(&fixture_dir());

    let mut trace = Trace::recording();
    model.forward_traced(&prompt, 0, &mut trace).unwrap();

    // The reference dumps `self_attn.out` and `o_proj.out` as the same buffer,
    // so every fixture key must be reachable; a trace that quietly stopped
    // recording would otherwise pass this test by comparing nothing.
    let mut checked = 0;
    let mut failures = Vec::new();
    for (name, expected) in &want {
        let Some(actual) = trace.get(name) else {
            panic!("the forward pass recorded nothing under {name}");
        };
        let (worst, scale) = compare(actual, expected);
        if f64::from(worst) > tolerance {
            failures.push(format!(
                "{name}: off by {worst:.3e} on values up to {scale:.3e}"
            ));
        }
        checked += 1;
    }
    assert_eq!(checked, want.len());
    assert!(checked >= 30, "only {checked} tensors compared");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_logits_match_and_the_argmax_agrees() {
    let (model, prompt, tolerance) = load_tiny();
    let want = reference(&fixture_dir());
    let logits = model.forward(&prompt).unwrap();

    let expected = &want["logits"];
    let (worst, scale) = compare(&logits, expected);
    assert!(
        f64::from(worst) <= tolerance,
        "logits off by {worst:.3e} on {scale:.3e}"
    );

    // The argmax is what generation actually consumes, and it can disagree
    // while the tensors are close.
    let argmax = |t: &Tensor| -> Vec<u32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .argmax(candle_core::D::Minus1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()
    };
    assert_eq!(argmax(&logits), argmax(expected));
}

#[test]
fn a_checkpoint_whose_shapes_disagree_with_the_config_is_refused() {
    let dir = fixture_dir();
    let weights = Weights::load(dir.join("model.safetensors"), &Device::Cpu).unwrap();
    let mut config = Config::from_file(dir.join("config.json")).unwrap();

    // One more head than the checkpoint was written with. The projections still
    // load, since q_proj is indexed by name and not by head, so nothing would
    // fail until a reshape several steps later produced a silent regrouping.
    config.num_attention_heads += 1;
    assert!(Model::from_weights(config, &weights, &Device::Cpu).is_err());
}

#[test]
fn an_empty_prompt_is_refused_rather_than_producing_an_empty_batch() {
    let (model, _, _) = load_tiny();
    assert!(model.forward(&[]).is_err());
}
