//! The Jinja template that turns a list of messages into the string a model
//! expects.
//!
//! Every model writes its own, and ships it inside `tokenizer_config.json`.
//! Qwen3's is four kilobytes that reach for `namespace()`, reverse slicing,
//! `loop.index0`, the `tojson` filter and six Python string methods, so this is
//! a template engine's job rather than a format to reimplement.
//!
//! What is not delegated is the check. `tests/fixtures/chat` holds the template
//! and thirty renderings taken from the reference implementation, and the test
//! beside it requires this to reproduce them exactly.

use minijinja::{Environment, context};
use serde_json::Value;

use crate::{Error, Result};

/// A parsed chat template.
pub struct ChatTemplate {
    environment: Environment<'static>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ChatTemplate")
    }
}

impl ChatTemplate {
    /// Parse a template, refusing one that does not compile rather than failing
    /// on the first request.
    pub fn parse(source: &str) -> Result<Self> {
        let mut environment = Environment::new();
        // Chat templates are written against Jinja2 as Python ships it and lean
        // on string methods minijinja has no reason to carry. Without this
        // callback Qwen3's template fails on its first `startswith`, and every
        // one of the thirty fixture renderings comes out an error.
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.set_lstrip_blocks(true);
        environment.set_trim_blocks(true);
        environment
            .add_template_owned("chat", source.to_string())
            .map_err(|e| Error::Config(format!("chat template does not parse: {e}")))?;
        Ok(Self { environment })
    }

    /// Render `messages`, which is the `OpenAI` message list as JSON.
    ///
    /// `enable_thinking` is left undefined when `None`, and that is not the same
    /// as passing `true`: Qwen3's template asks whether the variable is defined
    /// at all, and treats its absence as reasoning turned on. Forcing it off
    /// here would make this server quietly answer a different question than the
    /// model was asked.
    pub fn render(
        &self,
        messages: &Value,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        let template = self
            .environment
            .get_template("chat")
            .map_err(|e| Error::Config(format!("chat template: {e}")))?;
        let rendered = match enable_thinking {
            None => template.render(context! {
                messages => messages,
                add_generation_prompt => add_generation_prompt,
            }),
            Some(thinking) => template.render(context! {
                messages => messages,
                add_generation_prompt => add_generation_prompt,
                enable_thinking => thinking,
            }),
        };
        rendered.map_err(|e| Error::Config(format!("chat template: {e:#}")))
    }
}
