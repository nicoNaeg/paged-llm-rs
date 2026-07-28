//! The forward pass at full scale, against the reference and across backends.
//!
//! Gated on `PAGEDLLM_MODEL_DIR` and `PAGEDLLM_REFERENCE_DIR` because the
//! checkpoint is 1.5 GB and is not in the repository. `make test-model` sets
//! both. Unset, these tests say so and skip; set and pointing at nothing, they
//! fail, so a run that was asked for cannot quietly do nothing.
//!
//! The tiny fixture proves the structure in f32. This proves the same code
//! still agrees at 28 layers in bf16, where the tolerance has to come from a
//! measurement rather than from a preference.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use pagedllm::{Model, Trace};

/// Returns the two directories, or `None` when the run did not ask for this.
fn dirs() -> Option<(PathBuf, PathBuf)> {
    if let (Some(model), Some(reference)) = (
        std::env::var_os("PAGEDLLM_MODEL_DIR"),
        std::env::var_os("PAGEDLLM_REFERENCE_DIR"),
    ) {
        return Some((model.into(), reference.into()));
    }
    eprintln!(
        "skipped: set PAGEDLLM_MODEL_DIR and PAGEDLLM_REFERENCE_DIR, or run `make test-model`"
    );
    None
}

fn reference(dir: &Path) -> (HashMap<String, Tensor>, Vec<u32>, f64) {
    let activations =
        candle_core::safetensors::load(dir.join("activations.safetensors"), &Device::Cpu)
            .expect("reference activations");
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("reference manifest");
    let manifest: serde_json::Value = serde_json::from_str(&text).unwrap();
    let prompt = manifest["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    let tolerance = manifest["relative_tolerance"].as_f64().unwrap();
    (activations, prompt, tolerance)
}

fn to_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// Largest difference relative to the tensor's own scale, and that scale.
///
/// Relative rather than absolute, because Qwen3's residual stream carries
/// values into the thousands: an absolute threshold either passes everything at
/// the top of the model or fails everything at the bottom. The scale used is
/// the largest magnitude in the reference tensor, floored at one so a tensor of
/// near-zeroes does not divide a rounding error into a failure.
fn compare(got: &Tensor, want: &Tensor) -> (f32, f32) {
    assert_eq!(got.dims(), want.dims(), "shape");
    let (got, want) = (to_f32(got), to_f32(want));
    let mut worst = 0f32;
    let mut scale = 0f32;
    for (g, w) in got.iter().zip(&want) {
        worst = worst.max((g - w).abs());
        scale = scale.max(w.abs());
    }
    (worst / scale.max(1.0), scale)
}

/// Largest absolute difference, without the scaling `compare` applies.
#[cfg(feature = "metal")]
fn worst_absolute(got: &Tensor, want: &Tensor) -> f32 {
    let (got, want) = (to_f32(got), to_f32(want));
    got.iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max)
}

/// Per position, how far the winning logit sits above the runner-up.
///
/// This is what says whether a position is decided or a coin flip. Two
/// implementations that agree everywhere except where this is near zero have a
/// rounding difference; one that disagrees where it is wide has a bug.
#[cfg(feature = "metal")]
fn top_two_gaps(logits: &Tensor) -> Vec<f32> {
    let dims = logits.dims();
    let vocab = dims[dims.len() - 1];
    to_f32(logits)
        .chunks(vocab)
        .map(|row| {
            let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for &v in row {
                if v > best {
                    second = best;
                    best = v;
                } else if v > second {
                    second = v;
                }
            }
            best - second
        })
        .collect()
}

fn argmax(t: &Tensor) -> Vec<u32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<u32>()
        .unwrap()
}

#[test]
fn the_real_model_matches_the_reference_at_every_layer() {
    let Some((model_dir, reference_dir)) = dirs() else {
        return;
    };
    let (want, prompt, tolerance) = reference(&reference_dir);
    let model = Model::load_as(&model_dir, &Device::Cpu, Some(DType::F32)).expect("load the model");

    let mut trace = Trace::recording();
    let logits = model.forward_traced(&prompt, 0, &mut trace).unwrap();

    let mut failures = Vec::new();
    let mut deepest = (0f32, String::new());
    for (name, expected) in &want {
        let actual = trace
            .get(name)
            .unwrap_or_else(|| panic!("nothing recorded under {name}"));
        let (worst, scale) = compare(actual, expected);
        if worst > deepest.0 {
            deepest = (worst, name.clone());
        }
        if f64::from(worst) > tolerance {
            failures.push(format!(
                "{name}: off by {worst:.3e} on values up to {scale:.3e}"
            ));
        }
    }
    // Printed on success too: the drift across 28 layers is the number this
    // test exists to keep an eye on, and a threshold nobody can see is a guess.
    println!(
        "compared {} tensors, worst {:.3e} at {}",
        want.len(),
        deepest.0,
        deepest.1
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(argmax(&logits), argmax(&want["logits"]), "argmax");
}

/// The Metal backend on its own, with the dtype held fixed at f32.
///
/// This is what separates the two variables. If this passes at the same
/// tolerance the CPU does, then any larger gap in the bf16 test below is the
/// dtype's and not the backend's, which is a claim that otherwise could not be
/// made from either test alone.
#[cfg(feature = "metal")]
#[test]
fn the_metal_backend_matches_the_reference_in_f32() {
    let Some((model_dir, reference_dir)) = dirs() else {
        return;
    };
    let (want, prompt, tolerance) = reference(&reference_dir);
    let device = Device::new_metal(0).expect("a Metal device");
    let model = Model::load_as(&model_dir, &device, Some(DType::F32)).expect("load the model");

    let mut trace = Trace::recording();
    let logits = model
        .forward_traced(&prompt, 0, &mut trace)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();

    let mut failures = Vec::new();
    let mut deepest = (0f32, String::new());
    for (name, expected) in &want {
        let actual = trace
            .get(name)
            .unwrap_or_else(|| panic!("nothing recorded under {name}"))
            .to_device(&Device::Cpu)
            .unwrap();
        let (worst, scale) = compare(&actual, expected);
        if worst > deepest.0 {
            deepest = (worst, name.clone());
        }
        if f64::from(worst) > tolerance {
            failures.push(format!(
                "{name}: off by {worst:.3e} on values up to {scale:.3e}"
            ));
        }
    }
    println!("f32 on metal: worst {:.3e} at {}", deepest.0, deepest.1);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(argmax(&logits), argmax(&want["logits"]), "argmax");
}

/// The serving path: bf16 weights on the GPU, against a reference that ran in
/// the same dtype.
///
/// Compared against the bf16 dump rather than the f32 one, because the question
/// here is whether this implementation agrees with the reference, and a
/// disagreement measured against f32 would be the dtype's doing rather than the
/// code's. What f32 to bf16 costs is reported next to it, unasserted, because
/// it is a property of the checkpoint and not of anything written here.
#[cfg(feature = "metal")]
#[test]
fn bf16_on_metal_matches_a_reference_that_ran_in_bf16() {
    let Some((model_dir, f32_dir)) = dirs() else {
        return;
    };
    let Some(bf16_dir) = std::env::var_os("PAGEDLLM_REFERENCE_BF16_DIR").map(PathBuf::from) else {
        eprintln!("skipped: set PAGEDLLM_REFERENCE_BF16_DIR, or run `make test-model`");
        return;
    };
    let (want, prompt, tolerance) = reference(&bf16_dir);
    let device = Device::new_metal(0).expect("a Metal device");
    let model = Model::load(&model_dir, &device).expect("load the model");
    assert_eq!(model.dtype(), DType::BF16, "the checkpoint should be bf16");

    let mut trace = Trace::recording();
    let logits = model
        .forward_traced(&prompt, 0, &mut trace)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();

    let mut failures = Vec::new();
    let mut deepest = (0f32, String::new());
    for (name, expected) in &want {
        let actual = trace
            .get(name)
            .unwrap_or_else(|| panic!("nothing recorded under {name}"))
            .to_device(&Device::Cpu)
            .unwrap();
        let (worst, scale) = compare(&actual, expected);
        if worst > deepest.0 {
            deepest = (worst, name.clone());
        }
        if f64::from(worst) > tolerance {
            failures.push(format!(
                "{name}: off by {worst:.3e} on values up to {scale:.3e}"
            ));
        }
    }

    let (f32_want, _, _) = reference(&f32_dir);
    let (bf16_tokens, reference_tokens) = (argmax(&logits), argmax(&want["logits"]));

    // Two implementations in bf16 will not pick the same token where the top two
    // logits sit closer together than the noise between them. That is not a
    // tolerance chosen to make this pass: the noise is measured here, from these
    // two runs, and the gap comes from the f32 reference, which has neither.
    let noise = worst_absolute(&logits, &want["logits"]);
    let gaps = top_two_gaps(&f32_want["logits"]);
    let mut decided = Vec::new();
    let mut coin_flips = 0;
    for (position, (mine, theirs)) in bf16_tokens.iter().zip(&reference_tokens).enumerate() {
        if mine == theirs {
            continue;
        }
        if gaps[position] <= noise {
            coin_flips += 1;
        } else {
            decided.push(format!(
                "position {position}: {mine} against {theirs}, decided by {:.4} against {noise:.4} of noise",
                gaps[position]
            ));
        }
    }

    let (against_f32, _) = compare(&logits, &f32_want["logits"]);
    println!(
        "bf16 on metal against bf16 reference: worst {:.3e} at {}, {} tensors past {tolerance:.0e}\n\
         logits differ by at most {noise:.4}, which flips {coin_flips} of {} positions, all of them ties\n\
         against the f32 reference: logits {against_f32:.3e} relative",
        deepest.0,
        deepest.1,
        failures.len(),
        bf16_tokens.len()
    );

    // A position the reference decided by a wide margin has to come out the
    // same. That is the assertion a real defect fails, where comparing the whole
    // token sequence would fail on arithmetic noise instead.
    assert!(decided.is_empty(), "{}", decided.join("\n"));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
