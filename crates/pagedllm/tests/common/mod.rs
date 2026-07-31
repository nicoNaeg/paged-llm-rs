#![allow(dead_code)]
//! Shared by the tests that read the committed fixture.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use pagedllm::Model;

/// Where the two-layer model and its reference activations live.
pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
}

/// Largest difference relative to the tensor's own scale, and that scale.
///
/// Relative rather than absolute, because activations here span several orders
/// of magnitude and an absolute threshold would mean something different at each
/// layer. Floored at one so a tensor of near-zeroes cannot turn a rounding error
/// into a large ratio.
pub fn compare(got: &Tensor, want: &Tensor) -> (f32, f32) {
    // Shape, not just element count. A reshape that keeps every value in the
    // same flat order but calls the result a different rectangle produces
    // identical numbers here, so counting elements would pass it: exactly one
    // test in the repository caught such a mutation, and only because it
    // happened to narrow along a dimension afterwards.
    assert_eq!(got.dims(), want.dims(), "shape");
    let read = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let (got, want) = (read(got), read(want));
    let mut worst = 0f32;
    let mut scale = 0f32;
    for (g, w) in got.iter().zip(&want) {
        worst = worst.max((g - w).abs());
        scale = scale.max(w.abs());
    }
    (worst / scale.max(1.0), scale)
}

/// The fixture model, its prompt, and the tolerance it was dumped at.
pub fn load_tiny() -> (Model, Vec<u32>, f64) {
    let dir = fixture_dir();
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("fixture manifest");
    let manifest: serde_json::Value = serde_json::from_str(&text).expect("manifest is json");
    let prompt = manifest["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    let tolerance = manifest["relative_tolerance"].as_f64().unwrap();
    let model = Model::load(&dir, &Device::Cpu).expect("load the tiny model");
    (model, prompt, tolerance)
}
