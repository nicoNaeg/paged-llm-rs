//! The thread that owns the model, and the channel that reaches it.
//!
//! One thread, one model, one request at a time. The shape is chosen for what
//! comes next rather than for what stage 2 needs: stage 3 replaces the body of
//! the loop with a scheduler that advances many sequences per forward pass, and
//! everything above this file keeps working, because it already talks to the
//! engine by sending a job and reading events off a channel.
//!
//! A dedicated thread rather than a pool, because a GPU dispatch is blocking and
//! the model is one resource. Running it on the async runtime's workers would
//! stall every connection those workers carry.

use std::path::PathBuf;
use std::sync::Arc;

use pagedllm::{DType, Device};
use pagedllm::{Finish, IncrementalDecoder, Model, Request, Session, Tokenizer};
use tokio::sync::{mpsc, oneshot};

/// What the engine reports while a generation runs.
#[derive(Debug)]
pub enum Event {
    /// Text the model produced, already decoded.
    Text(String),
    /// The generation ended.
    Done {
        /// Whether the model stopped or the budget did.
        finish: Finish,
        /// How many tokens the prompt held.
        prompt_tokens: usize,
        /// How many the model produced.
        completion_tokens: usize,
    },
    /// The generation could not run or could not continue.
    Failed(String),
}

struct Job {
    request: Request,
    events: mpsc::UnboundedSender<Event>,
}

/// A handle to the engine thread.
#[derive(Debug, Clone)]
pub struct Engine {
    jobs: mpsc::UnboundedSender<Job>,
    tokenizer: Arc<Tokenizer>,
    model_id: String,
    backend: String,
}

impl Engine {
    /// Load a model on its own thread and return once it is ready to serve.
    ///
    /// Loading is done on the engine thread rather than here so the model never
    /// has to cross one, which keeps the whole question of whether a device
    /// handle may be shared out of this design.
    pub async fn start(dir: PathBuf, device: Device, dtype: Option<DType>) -> Result<Self, String> {
        let tokenizer = Arc::new(Tokenizer::load(&dir).map_err(|e| e.to_string())?);
        let model_id = dir
            .file_name()
            .map_or_else(|| "model".to_string(), |n| n.to_string_lossy().into_owned());
        let backend = format!("{device:?}").to_lowercase();

        let (jobs, mut inbox) = mpsc::unbounded_channel::<Job>();
        let (ready, loaded) = oneshot::channel::<Result<(), String>>();
        let thread_tokenizer = Arc::clone(&tokenizer);

        std::thread::Builder::new()
            .name("pagedllm-engine".into())
            .spawn(move || {
                let model = match Model::load_as(&dir, &device, dtype) {
                    Ok(model) => {
                        let _ = ready.send(Ok(()));
                        model
                    }
                    Err(e) => {
                        let _ = ready.send(Err(e.to_string()));
                        return;
                    }
                };
                while let Some(job) = inbox.blocking_recv() {
                    run(&model, &thread_tokenizer, job);
                }
            })
            .map_err(|e| format!("starting the engine thread: {e}"))?;

        loaded
            .await
            .map_err(|_| "the engine thread stopped while loading".to_string())??;

        Ok(Self {
            jobs,
            tokenizer,
            model_id,
            backend,
        })
    }

    /// Queue a generation and read its events.
    pub fn submit(&self, request: Request) -> Result<mpsc::UnboundedReceiver<Event>, String> {
        let (events, stream) = mpsc::unbounded_channel();
        self.jobs
            .send(Job { request, events })
            .map_err(|_| "the engine thread has stopped".to_string())?;
        Ok(stream)
    }

    /// The tokenizer of the loaded model.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The name the model is served under.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Which device the model resolved to.
    pub fn backend(&self) -> &str {
        &self.backend
    }
}

fn run(model: &Model, tokenizer: &Tokenizer, job: Job) {
    let prompt_tokens = job.request.prompt.len();
    let mut session = match Session::new(model, job.request) {
        Ok(session) => session,
        Err(e) => {
            let _ = job.events.send(Event::Failed(e.to_string()));
            return;
        }
    };

    let mut decoder = IncrementalDecoder::new();
    loop {
        match session.next_token() {
            Ok(Some(token)) => match decoder.push(tokenizer, token) {
                Ok(text) if text.is_empty() => {}
                Ok(text) => {
                    // A closed channel means the client is gone. Stopping here
                    // is what keeps a cancelled request from holding the engine
                    // for the rest of its token budget.
                    if job.events.send(Event::Text(text)).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = job.events.send(Event::Failed(e.to_string()));
                    return;
                }
            },
            Ok(None) => break,
            Err(e) => {
                let _ = job.events.send(Event::Failed(e.to_string()));
                return;
            }
        }
    }

    let _ = job.events.send(Event::Done {
        finish: session.finish_reason().unwrap_or(Finish::Length),
        prompt_tokens,
        completion_tokens: session.generated(),
    });
}
