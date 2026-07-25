//! The forward pass against the reference implementation, module by module.
//!
//! The fixture under `tests/fixtures/tiny` is dumped by
//! `scripts/dump_reference.py` from `HuggingFace` transformers. Comparing only
//! the logits would say "wrong" and stop there; comparing every module boundary
//! says which one, which is what makes a mistake in `RoPE` distinguishable from
//! one in the head grouping.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use pagedllm::{Config, Model, Trace, Weights};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
}

fn reference(dir: &Path) -> HashMap<String, Tensor> {
    candle_core::safetensors::load(dir.join("activations.safetensors"), &Device::Cpu)
        .expect("fixture activations")
}

fn manifest(dir: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("fixture manifest");
    serde_json::from_str(&text).expect("fixture manifest is json")
}

/// Largest difference relative to the tensor's own scale, and that scale.
///
/// Relative rather than absolute, because activations here span several orders
/// of magnitude and an absolute threshold would mean something different at
/// each layer. Floored at one so a tensor of near-zeroes cannot turn a rounding
/// error into a large ratio.
fn compare(got: &Tensor, want: &Tensor) -> (f32, f32) {
    assert_eq!(got.dims(), want.dims(), "shape");
    let got = got
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut worst = 0f32;
    let mut scale = 0f32;
    for (g, w) in got.iter().zip(&want) {
        worst = worst.max((g - w).abs());
        scale = scale.max(w.abs());
    }
    (worst / scale.max(1.0), scale)
}

fn load_tiny() -> (Model, Vec<u32>, f64) {
    let dir = fixture_dir();
    let manifest = manifest(&dir);
    let prompt: Vec<u32> = manifest["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    let tolerance = manifest["relative_tolerance"].as_f64().unwrap();
    let model = Model::load(&dir, &Device::Cpu).expect("load the tiny model");
    (model, prompt, tolerance)
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
