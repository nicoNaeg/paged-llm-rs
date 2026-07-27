//! Text in, tokens out, and the chat template that sits between the two.
//!
//! Neither half is one of the mechanics this project is about. Byte-level BPE is
//! a file format to be compatible with, and a chat template is a Jinja document
//! shipped inside `tokenizer_config.json`, which every model writes differently.
//! Both come from crates, and the check that they are right is a fixture of
//! renderings taken from the reference implementation.

use std::path::Path;

use serde_json::Value;

use crate::chat::ChatTemplate;
use crate::{Error, Result};

/// The tokenizer and chat template of one model directory.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    chat: Option<ChatTemplate>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocabulary", &self.inner.get_vocab_size(true))
            .field("chat_template", &self.chat.is_some())
            .finish()
    }
}

impl Tokenizer {
    /// Load `tokenizer.json` and the chat template beside it.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;

        let config_path = dir.join("tokenizer_config.json");
        let text =
            std::fs::read_to_string(&config_path).map_err(|e| Error::Io(config_path.clone(), e))?;
        let config: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Config(format!("{}: {e}", config_path.display())))?;
        let chat = config
            .get("chat_template")
            .and_then(Value::as_str)
            .map(ChatTemplate::parse)
            .transpose()?;

        Ok(Self { inner, chat })
    }

    /// Token ids for `text`.
    ///
    /// Special tokens are not added here. The chat template already writes the
    /// turn markers into the text, and adding them again would put two of every
    /// marker in front of the model.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|e| Error::Config(format!("encoding: {e}")))
    }

    /// Text for `tokens`, with the turn markers left out.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| Error::Config(format!("decoding: {e}")))
    }

    /// Text for `tokens`, markers included, which is what a prompt echo needs.
    pub fn decode_with_specials(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, false)
            .map_err(|e| Error::Config(format!("decoding: {e}")))
    }

    /// Render a list of chat messages into the string the model expects.
    pub fn render_chat(
        &self,
        messages: &Value,
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        self.chat_template()?
            .render(messages, add_generation_prompt, enable_thinking)
    }

    /// The parsed template, or an error naming what is missing.
    pub fn chat_template(&self) -> Result<&ChatTemplate> {
        self.chat
            .as_ref()
            .ok_or_else(|| Error::Unsupported("this model ships no chat template".into()))
    }

    /// Whether this model ships a chat template at all.
    pub fn has_chat_template(&self) -> bool {
        self.chat.is_some()
    }

    /// How many distinct tokens the vocabulary holds.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

/// Turns a stream of tokens into a stream of text.
///
/// A token is not a character. Byte-level BPE splits multi-byte characters
/// across tokens, so decoding each one on its own produces replacement
/// characters where an accent or an emoji was. This decodes the whole sequence
/// and emits what is newly complete, holding back a trailing replacement
/// character until the token that finishes it arrives.
#[derive(Debug, Default)]
pub struct IncrementalDecoder {
    tokens: Vec<u32>,
    emitted: usize,
}

impl IncrementalDecoder {
    /// A decoder with nothing emitted yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a token and return the text it completed, which is often empty.
    pub fn push(&mut self, tokenizer: &Tokenizer, token: u32) -> Result<String> {
        self.tokens.push(token);
        let text = tokenizer.decode(&self.tokens)?;
        // U+FFFD at the end means the last bytes do not yet form a character.
        let stable = match text.strip_suffix('\u{FFFD}') {
            Some(before) => before.len(),
            None => text.len(),
        };
        if stable <= self.emitted {
            return Ok(String::new());
        }
        let delta = text[self.emitted..stable].to_string();
        self.emitted = stable;
        Ok(delta)
    }

    /// Everything decoded so far, including anything still held back.
    pub fn finish(&self, tokenizer: &Tokenizer) -> Result<String> {
        tokenizer.decode(&self.tokens)
    }

    /// How many tokens have been pushed.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether nothing has been pushed.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
