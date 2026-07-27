//! The HTTP surface.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::{Stream, unfold};
use pagedllm::{Finish, GenerationConfig, Request, Sampling};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedReceiver;

use crate::engine::{Engine, Event};
use crate::openai::{
    ApiError, ChatChoice, ChatChunk, ChatChunkChoice, ChatRequest, ChatResponse, CompletionChoice,
    CompletionRequest, CompletionResponse, Delta, Message, ModelCard, ModelList, Usage, now,
};

/// Everything a handler needs.
#[derive(Clone)]
pub struct AppState {
    /// The thread holding the model.
    pub engine: Engine,
    /// The model's own generation defaults, used where a request names none.
    pub generation: Arc<GenerationConfig>,
    /// Token budget for a request that gives no limit of its own.
    pub default_max_tokens: usize,
}

/// Every route this server answers.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn models(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: state.engine.model_id().to_string(),
            object: "model",
            created: now(),
            owned_by: "pagedllm",
        }],
    })
}

/// Counts requests, so two that arrive together are still told apart.
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A request identifier, unique for the life of the process.
fn request_id(prefix: &str) -> String {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:016x}{n:x}", now())
}

/// A seed for a request that named none.
///
/// The clock alone is not enough: it moves once a second, so two sampled
/// requests arriving together would draw the same tokens and look like a
/// caching bug. The counter is what makes them differ, and mixing rather than
/// concatenating keeps neighbouring seeds from producing neighbouring streams.
fn default_seed() -> u64 {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    now()
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(n.wrapping_mul(0xBF58_476D_1CE4_E5B9))
}

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ApiError::invalid_request(message)),
    )
        .into_response()
}

fn server_error(message: impl Into<String>) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ApiError::server(message)),
    )
        .into_response()
}

fn finish_name(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop => "stop",
        Finish::Length => "length",
    }
}

/// Build the sampling settings, letting the request override the model's own.
fn sampling(
    state: &AppState,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
) -> Result<Sampling, String> {
    let temperature = temperature.or(state.generation.temperature).unwrap_or(1.0);
    // A request that asks for greedy decoding means it: top-k and top-p from
    // the model's config would otherwise be applied to a draw that never
    // happens, which is harmless, but a top-p carried into an explicit
    // temperature of zero reads like it was honoured.
    let (top_p, top_k) = if temperature <= 0.0 {
        (None, None)
    } else {
        (
            top_p.or(state.generation.top_p),
            top_k.or(state.generation.top_k),
        )
    };
    Sampling::new(temperature, top_k, top_p).map_err(|e| e.to_string())
}

/// Pull every event to the end, which is what a non-streamed response needs.
async fn collect(
    mut events: UnboundedReceiver<Event>,
) -> Result<(String, Finish, Usage), Response> {
    let mut text = String::new();
    while let Some(event) = events.recv().await {
        match event {
            Event::Text(chunk) => text.push_str(&chunk),
            Event::Failed(e) => return Err(server_error(e)),
            Event::Done {
                finish,
                prompt_tokens,
                completion_tokens,
            } => {
                return Ok((
                    text,
                    finish,
                    Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                ));
            }
        }
    }
    Err(server_error("the engine stopped without finishing"))
}

/// Turn the engine's events into server-sent events.
fn sse_stream<F>(
    events: UnboundedReceiver<Event>,
    frame: F,
) -> Sse<impl Stream<Item = Result<SseEvent, std::convert::Infallible>>>
where
    F: FnMut(Option<String>, Option<&'static str>) -> Value + Send + 'static,
{
    // The frame builder travels in the fold's state rather than being captured:
    // a closure borrowed by the async block it returns cannot outlive the call
    // that produced it.
    //
    // `[DONE]` is not JSON and is what every client watches for, so the state
    // also carries whether it has been sent, and the stream ends after it
    // rather than at the last chunk.
    let stream = unfold(
        (events, frame, false),
        |(mut events, mut frame, done)| async move {
            if done {
                return None;
            }
            let (data, done) = match events.recv().await {
                Some(Event::Text(text)) => (frame(Some(text), None).to_string(), false),
                Some(Event::Done { finish, .. }) => {
                    (frame(None, Some(finish_name(finish))).to_string(), false)
                }
                Some(Event::Failed(_)) | None => ("[DONE]".to_string(), true),
            };
            Some((Ok(SseEvent::default().data(data)), (events, frame, done)))
        },
    );
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Response {
    if let Some(field) = request.unsupported.refused() {
        return bad_request(format!("{field} is not supported by this server"));
    }
    let tokenizer = state.engine.tokenizer();
    if !tokenizer.has_chat_template() {
        return bad_request("this model ships no chat template; use /v1/completions");
    }

    let enable_thinking = request
        .chat_template_kwargs
        .as_ref()
        .and_then(|kwargs| kwargs.get("enable_thinking"))
        .and_then(Value::as_bool);

    let prompt_text = match tokenizer.render_chat(&request.messages, true, enable_thinking) {
        Ok(text) => text,
        Err(e) => return bad_request(e.to_string()),
    };
    let prompt = match tokenizer.encode(&prompt_text) {
        Ok(tokens) => tokens,
        Err(e) => return bad_request(e.to_string()),
    };
    if prompt.is_empty() {
        return bad_request("the rendered prompt is empty");
    }

    let sampling = match sampling(&state, request.temperature, request.top_p, request.top_k) {
        Ok(sampling) => sampling,
        Err(e) => return bad_request(e),
    };

    let engine_request = Request {
        prompt,
        sampling,
        max_tokens: request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(state.default_max_tokens),
        stop_tokens: state.generation.eos_token_ids.clone(),
        seed: request.seed.unwrap_or_else(default_seed),
    };

    let events = match state.engine.submit(engine_request) {
        Ok(events) => events,
        Err(e) => return server_error(e),
    };

    let id = request_id("chatcmpl");
    let model = request
        .model
        .unwrap_or_else(|| state.engine.model_id().to_string());

    if request.stream {
        let (stream_id, stream_model) = (id.clone(), model.clone());
        let mut first = true;
        return sse_stream(events, move |text, finish| {
            let delta = match (&text, finish) {
                (Some(content), _) => Delta {
                    // The first chunk announces the role, which is what clients
                    // that build a message from the stream look for.
                    role: first.then(|| {
                        first = false;
                        "assistant"
                    }),
                    content: Some(content.clone()),
                },
                (None, _) => Delta::default(),
            };
            serde_json::to_value(ChatChunk {
                id: stream_id.clone(),
                object: "chat.completion.chunk",
                created: now(),
                model: stream_model.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta,
                    finish_reason: finish,
                }],
            })
            .unwrap_or(Value::Null)
        })
        .into_response();
    }

    match collect(events).await {
        Err(response) => response,
        Ok((text, finish, usage)) => Json(ChatResponse {
            id,
            object: "chat.completion",
            created: now(),
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: Message {
                    role: "assistant",
                    content: text,
                },
                finish_reason: finish_name(finish),
            }],
            usage,
        })
        .into_response(),
    }
}

async fn completions(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    if let Some(field) = request.unsupported.refused() {
        return bad_request(format!("{field} is not supported by this server"));
    }
    if request.echo == Some(true) {
        return bad_request("echo is not supported by this server");
    }

    let Some(prompt_text) = request.prompt.as_str() else {
        return bad_request(
            "prompt has to be a single string; a list of prompts is several requests",
        );
    };

    let tokenizer = state.engine.tokenizer();
    let prompt = match tokenizer.encode(prompt_text) {
        Ok(tokens) => tokens,
        Err(e) => return bad_request(e.to_string()),
    };
    if prompt.is_empty() {
        return bad_request("the prompt encodes to no tokens");
    }

    let sampling = match sampling(&state, request.temperature, request.top_p, request.top_k) {
        Ok(sampling) => sampling,
        Err(e) => return bad_request(e),
    };

    let events = match state.engine.submit(Request {
        prompt,
        sampling,
        max_tokens: request.max_tokens.unwrap_or(state.default_max_tokens),
        stop_tokens: state.generation.eos_token_ids.clone(),
        seed: request.seed.unwrap_or_else(default_seed),
    }) {
        Ok(events) => events,
        Err(e) => return server_error(e),
    };

    let id = request_id("cmpl");
    let model = request
        .model
        .unwrap_or_else(|| state.engine.model_id().to_string());

    if request.stream {
        let (stream_id, stream_model) = (id.clone(), model.clone());
        return sse_stream(events, move |text, finish| {
            serde_json::to_value(CompletionResponse {
                id: stream_id.clone(),
                object: "text_completion",
                created: now(),
                model: stream_model.clone(),
                choices: vec![CompletionChoice {
                    index: 0,
                    text: text.unwrap_or_default(),
                    finish_reason: finish,
                }],
                usage: None,
            })
            .unwrap_or(Value::Null)
        })
        .into_response();
    }

    match collect(events).await {
        Err(response) => response,
        Ok((text, finish, usage)) => Json(CompletionResponse {
            id,
            object: "text_completion",
            created: now(),
            model,
            choices: vec![CompletionChoice {
                index: 0,
                text,
                finish_reason: Some(finish_name(finish)),
            }],
            usage: Some(usage),
        })
        .into_response(),
    }
}
