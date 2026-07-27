//! The chat template against the renderings the reference implementation
//! produces.
//!
//! This is the check that hands the job to a crate safely. The template engine
//! is a dependency, and this test is the thing that says a version of it still
//! agrees with Python's Jinja2 on a real template, byte for byte, across every
//! branch the fixture reaches.
//!
//! The fixture is committed, so this runs in CI without the checkpoint. Only the
//! template and the expected strings are needed, and together they are 16 KB.

use std::path::{Path, PathBuf};

use pagedllm::ChatTemplate;
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chat")
}

fn load() -> (ChatTemplate, Value) {
    let dir = fixture_dir();
    let source = std::fs::read_to_string(dir.join("template.jinja")).expect("template");
    let expected: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("expected.json")).expect("expected"),
    )
    .expect("expected is json");
    (
        ChatTemplate::parse(&source).expect("template parses"),
        expected,
    )
}

#[test]
fn every_rendering_matches_the_reference_implementation() {
    let (template, expected) = load();
    let renderings = expected["renderings"].as_object().expect("renderings");

    let mut failures = Vec::new();
    for (key, want) in renderings {
        let mut parts = key.split('|');
        let case = parts.next().unwrap();
        let add_generation_prompt = parts.next().unwrap() == "True";
        let thinking = match parts.next().unwrap() {
            "None" => None,
            "True" => Some(true),
            other => {
                assert_eq!(other, "False");
                Some(false)
            }
        };

        let messages = &expected["cases"][case];
        match template.render(messages, add_generation_prompt, thinking) {
            Ok(got) if got == want.as_str().unwrap() => {}
            Ok(got) => failures.push(format!(
                "{key}\n  want {:?}\n  got  {got:?}",
                want.as_str().unwrap()
            )),
            Err(e) => failures.push(format!("{key}: {e}")),
        }
    }

    // A fixture that quietly stopped covering anything would otherwise pass.
    assert!(
        renderings.len() >= 30,
        "only {} renderings",
        renderings.len()
    );
    assert!(
        failures.is_empty(),
        "{} of {} renderings differ:\n{}",
        failures.len(),
        renderings.len(),
        failures.join("\n")
    );
}

/// Leaving the variable undefined is not the same as passing `true`.
///
/// Qwen3's template asks whether `enable_thinking` is defined before asking
/// whether it is false, so a renderer that defaults it to `true` produces the
/// same output here and a different one on a template that tests it the other
/// way. The fixture carries all three, and this pins the one that is easy to
/// get wrong.
#[test]
fn turning_reasoning_off_is_the_only_case_that_prefills_a_think_block() {
    let (template, expected) = load();
    let messages = &expected["cases"]["single_turn"];

    let undefined = template.render(messages, true, None).unwrap();
    let on = template.render(messages, true, Some(true)).unwrap();
    let off = template.render(messages, true, Some(false)).unwrap();

    assert_eq!(undefined, on);
    assert!(!undefined.contains("<think>"));
    assert!(off.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

#[test]
fn a_template_that_does_not_compile_is_refused_at_load_rather_than_at_request() {
    assert!(ChatTemplate::parse("{% for x in %}").is_err());
}

#[test]
fn the_generation_prompt_is_what_decides_whether_the_model_is_asked_to_speak() {
    let (template, expected) = load();
    let messages = &expected["cases"]["single_turn"];
    assert!(
        template
            .render(messages, true, None)
            .unwrap()
            .ends_with("<|im_start|>assistant\n")
    );
    assert!(
        template
            .render(messages, false, None)
            .unwrap()
            .ends_with("<|im_end|>\n")
    );
}
