//! The wire types of the `OpenAI` API, and what this server refuses.
//!
//! Parameters that are parsed and then rejected are the point of this module.
//! A server that accepts `frequency_penalty` and ignores it answers a different
//! question than it was asked, and the client has no way to find out. Every
//! field below is either honoured or named in an error.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `POST /v1/chat/completions`.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// Ignored beyond being echoed back: this server holds one model.
    #[serde(default)]
    pub model: Option<String>,
    /// The conversation, passed to the chat template as it arrived.
    pub messages: Value,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// What `max_tokens` was renamed to. Whichever is set wins, and both being
    /// set is not an error since they mean the same thing.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Not in the `OpenAI` schema, but every local server takes it and Qwen3's
    /// own generation config sets it.
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Extra variables for the chat template, which is where `enable_thinking`
    /// arrives. The same field name vLLM uses.
    #[serde(default)]
    pub chat_template_kwargs: Option<Value>,

    #[serde(flatten)]
    pub unsupported: Unsupported,
}

/// `POST /v1/completions`.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// A single string. A list of prompts is one request per prompt in the
    /// `OpenAI` schema, and that is batching, which stage 3 owns.
    pub prompt: Value,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub echo: Option<bool>,

    #[serde(flatten)]
    pub unsupported: Unsupported,
}

/// Fields this server parses in order to refuse them.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Default, Deserialize)]
pub struct Unsupported {
    /// More than one completion per request needs the batching stage 3 adds.
    #[serde(default)]
    pub n: Option<u32>,
    /// Stop strings need the decoded text watched for a suffix, and a partial
    /// match held back from the stream. Worth doing, not done.
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub functions: Option<Value>,
    #[serde(default)]
    pub logit_bias: Option<Value>,
    #[serde(default)]
    pub logprobs: Option<Value>,
    #[serde(default)]
    pub top_logprobs: Option<Value>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub response_format: Option<Value>,
}

impl Unsupported {
    /// The name of the first field that was asked for and cannot be honoured.
    ///
    /// A penalty of zero and `n` of one are what a client sends when it means
    /// "default", so those pass: refusing them would reject requests that asked
    /// for nothing this server does not do.
    pub fn refused(&self) -> Option<&'static str> {
        let zero = |v: Option<f32>| v.is_some_and(|x| x != 0.0);
        if self.n.is_some_and(|n| n != 1) {
            return Some("n above 1");
        }
        if self.stop.is_some() {
            return Some("stop");
        }
        if self.tools.is_some() {
            return Some("tools");
        }
        if self.tool_choice.is_some() {
            return Some("tool_choice");
        }
        if self.functions.is_some() {
            return Some("functions");
        }
        if self.logit_bias.is_some() {
            return Some("logit_bias");
        }
        if self
            .logprobs
            .as_ref()
            .is_some_and(|v| v != &Value::Bool(false))
        {
            return Some("logprobs");
        }
        if self.top_logprobs.is_some() {
            return Some("top_logprobs");
        }
        if zero(self.frequency_penalty) {
            return Some("frequency_penalty");
        }
        if zero(self.presence_penalty) {
            return Some("presence_penalty");
        }
        if self.response_format.is_some() {
            return Some("response_format");
        }
        None
    }
}

/// How many tokens a request cost.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// One message of a completed conversation.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Serialize)]
pub struct Message {
    pub role: &'static str,
    pub content: String,
}

/// One choice of a non-streamed chat completion.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: &'static str,
}

/// The body of a non-streamed chat completion.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// The changing part of a streamed chunk.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Default, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// One choice of a streamed chunk.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<&'static str>,
}

/// One `data:` frame of a streamed chat completion.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

/// One choice of a text completion, streamed or not.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<&'static str>,
}

/// The body of a text completion, and of one streamed frame of it.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// One entry of `GET /v1/models`.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

/// The body of `GET /v1/models`.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

/// The error shape clients expect, which is nested under `error`.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

/// What went wrong.
// These mirror a published schema, and every field carries the name that schema
// gives it. A doc comment per field would restate the name and nothing else, so
// the module header carries the reasoning and the fields carry the spelling.
#[allow(missing_docs)]
#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    pub r#type: &'static str,
    pub code: Option<&'static str>,
}

impl ApiError {
    /// A client mistake.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                message: message.into(),
                r#type: "invalid_request_error",
                code: None,
            },
        }
    }

    /// A failure on this side.
    pub fn server(message: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody {
                message: message.into(),
                r#type: "server_error",
                code: None,
            },
        }
    }
}

/// Seconds since the epoch, which every response carries.
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[cfg(test)]
mod tests {
    use super::{ChatRequest, Unsupported};

    fn refused(json: serde_json::Value) -> Option<&'static str> {
        let request: ChatRequest = serde_json::from_value(json).unwrap();
        request.unsupported.refused()
    }

    #[test]
    fn a_plain_request_is_accepted() {
        assert_eq!(
            refused(serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7,
                "max_tokens": 32
            })),
            None
        );
    }

    #[test]
    fn what_a_client_sends_meaning_default_is_not_refused() {
        assert_eq!(
            refused(serde_json::json!({
                "messages": [],
                "n": 1,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "logprobs": false
            })),
            None
        );
    }

    #[test]
    fn each_unsupported_parameter_is_named_rather_than_ignored() {
        for (field, value) in [
            ("n", serde_json::json!(2)),
            ("stop", serde_json::json!(["\n"])),
            ("tools", serde_json::json!([])),
            ("logit_bias", serde_json::json!({"1": 2.0})),
            ("logprobs", serde_json::json!(true)),
            ("frequency_penalty", serde_json::json!(0.5)),
            (
                "response_format",
                serde_json::json!({"type": "json_object"}),
            ),
        ] {
            let mut body = serde_json::json!({"messages": []});
            body[field] = value;
            assert!(
                refused(body).is_some(),
                "{field} was accepted and would be ignored"
            );
        }
    }

    #[test]
    fn an_empty_request_asks_for_nothing_unsupported() {
        assert_eq!(Unsupported::default().refused(), None);
    }
}
