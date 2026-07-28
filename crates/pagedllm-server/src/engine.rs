//! The thread that owns the model, and the channel that reaches it.
//!
//! One thread, one model, and as many sequences as the pool holds. The loop is
//! the whole of continuous batching: drain whatever arrived, ask the scheduler
//! what to run, run it, hand the tokens back. A request that lands while others
//! are decoding joins the next pass rather than waiting for them to finish,
//! which is the property the name describes.
//!
//! A dedicated thread rather than a pool, because a GPU dispatch is blocking and
//! the model is one resource. Running it on the async runtime's workers would
//! stall every connection those workers carry.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use pagedllm::{
    Batch, CacheConfig, DType, Device, Finish, IncrementalDecoder, Model, Plan, Request, Scheduler,
    Sequence, SlotCache, Tokenizer,
};
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

/// How the pool is sized at startup.
#[derive(Debug, Clone, Copy)]
pub struct PoolConfig {
    /// Tokens reserved per sequence, prompt and completion together.
    pub max_seq: usize,
    /// Sequences that can be resident at once.
    pub slots: usize,
    /// Rows in one decode pass, capped independently of the slots.
    pub max_batch: usize,
}

/// A handle to the engine thread.
#[derive(Debug, Clone)]
pub struct Engine {
    jobs: mpsc::UnboundedSender<Job>,
    tokenizer: Arc<Tokenizer>,
    model_id: String,
    pool: PoolConfig,
    cache_bytes: usize,
}

impl Engine {
    /// Load a model on its own thread and return once it is ready to serve.
    pub async fn start(
        dir: PathBuf,
        device: Device,
        dtype: Option<DType>,
        pool: PoolConfig,
    ) -> Result<Self, String> {
        let tokenizer = Arc::new(Tokenizer::load(&dir).map_err(|e| e.to_string())?);
        let model_id = dir
            .file_name()
            .map_or_else(|| "model".to_string(), |n| n.to_string_lossy().into_owned());

        let (jobs, inbox) = mpsc::unbounded_channel::<Job>();
        let (ready, loaded) = oneshot::channel::<Result<usize, String>>();
        let thread_tokenizer = Arc::clone(&tokenizer);

        std::thread::Builder::new()
            .name("pagedllm-engine".into())
            .spawn(move || {
                let started = (|| {
                    let model = Model::load_as(&dir, &device, dtype).map_err(|e| e.to_string())?;
                    let config = model.config();
                    let cache_config = CacheConfig {
                        slots: pool.slots,
                        max_seq: pool.max_seq,
                        kv_heads: config.num_key_value_heads,
                        head_dim: config.head_dim,
                        layers: config.num_hidden_layers,
                    };
                    let bytes = cache_config.bytes(model.dtype());
                    let cache = SlotCache::new(cache_config, model.dtype(), &device)
                        .map_err(|e| e.to_string())?;
                    Ok::<_, String>((model, cache, bytes))
                })();

                let (model, cache, bytes) = match started {
                    Ok(parts) => parts,
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                let _ = ready.send(Ok(bytes));
                run(
                    &model,
                    Scheduler::new(cache, pool.max_batch),
                    &thread_tokenizer,
                    inbox,
                );
            })
            .map_err(|e| format!("starting the engine thread: {e}"))?;

        let cache_bytes = loaded
            .await
            .map_err(|_| "the engine thread stopped while loading".to_string())??;

        Ok(Self {
            jobs,
            tokenizer,
            model_id,
            pool,
            cache_bytes,
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

    /// How the pool was sized.
    pub fn pool(&self) -> PoolConfig {
        self.pool
    }

    /// What the pool cost to allocate.
    pub fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }
}

/// Everything the engine keeps for one running sequence.
struct Client {
    events: mpsc::UnboundedSender<Event>,
    decoder: IncrementalDecoder,
    prompt_tokens: usize,
}

fn run(
    model: &Model,
    mut scheduler: Scheduler,
    tokenizer: &Tokenizer,
    mut inbox: mpsc::UnboundedReceiver<Job>,
) {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    let mut clients: HashMap<u64, Client> = HashMap::new();

    loop {
        // Take everything that arrived without waiting for it, so a request that
        // lands mid-batch joins the next pass rather than the next idle moment.
        while let Ok(job) = inbox.try_recv() {
            admit(&mut scheduler, &mut clients, &NEXT_ID, job);
        }

        let plan = scheduler.plan();
        if matches!(plan, Plan::Idle) {
            // Nothing to run. Block until something arrives, and stop once every
            // handle to the engine has been dropped.
            match inbox.blocking_recv() {
                Some(job) => admit(&mut scheduler, &mut clients, &NEXT_ID, job),
                None => return,
            }
            continue;
        }

        let ids = plan.ids().to_vec();
        let sampled = match plan.batch() {
            // A prompt the scheduler refused before it ran, which it reports by
            // planning an empty batch.
            Some(batch) if batch.tokens.is_empty() => Vec::new(),
            Some(batch) => match sample(model, &mut scheduler, batch, &ids) {
                Ok(tokens) => tokens,
                Err(e) => {
                    fail(&mut scheduler, &mut clients, &ids, &e);
                    continue;
                }
            },
            None => Vec::new(),
        };

        let outcome = scheduler.commit(&plan, &sampled);
        for (id, token) in outcome.tokens {
            let Some(client) = clients.get_mut(&id) else {
                continue;
            };
            match client.decoder.push(tokenizer, token) {
                Ok(text) if text.is_empty() => {}
                Ok(text) => {
                    // A closed channel means the client is gone, and the
                    // sequence stops costing GPU time at the next plan.
                    if client.events.send(Event::Text(text)).is_err() {
                        scheduler.cancel(id);
                    }
                }
                Err(e) => {
                    let _ = client.events.send(Event::Failed(e.to_string()));
                    scheduler.cancel(id);
                }
            }
        }
        for (id, finish) in outcome.finished {
            if let Some(client) = clients.remove(&id) {
                let _ = client.events.send(Event::Done {
                    finish,
                    prompt_tokens: client.prompt_tokens,
                    completion_tokens: client.decoder.len(),
                });
            }
        }
    }
}

/// Run one batch and choose a token for each of its rows.
fn sample(
    model: &Model,
    scheduler: &mut Scheduler,
    batch: &Batch,
    ids: &[u64],
) -> Result<Vec<u32>, String> {
    let logits = model
        .forward_batch(batch, scheduler.cache_mut())
        .map_err(|e| e.to_string())?;
    // One transfer for the whole batch rather than one per row. The vocabulary
    // is 150k logits wide and all of it has to reach the host for a sampler that
    // runs there.
    let rows: Vec<f32> = logits
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1())
        .map_err(|e| e.to_string())?;
    let vocab = rows.len() / ids.len().max(1);

    let mut sampled = Vec::with_capacity(ids.len());
    for (row, &id) in ids.iter().enumerate() {
        let sequence = scheduler
            .sequence_mut(id)
            .ok_or_else(|| format!("sequence {id} left while its batch was running"))?;
        let (sampling, rng) = sequence.sampling();
        let token = sampling
            .sample(&rows[row * vocab..(row + 1) * vocab], rng)
            .map_err(|e| e.to_string())?;
        sampled.push(token);
    }
    Ok(sampled)
}

fn admit(
    scheduler: &mut Scheduler,
    clients: &mut HashMap<u64, Client>,
    next_id: &AtomicU64,
    job: Job,
) {
    let id = next_id.fetch_add(1, Ordering::Relaxed);
    let prompt_tokens = job.request.prompt.len();
    let sequence = match Sequence::new(
        id,
        job.request.prompt,
        job.request.sampling,
        job.request.max_tokens,
        job.request.stop_tokens,
        job.request.seed,
    ) {
        Ok(sequence) => sequence,
        Err(e) => {
            let _ = job.events.send(Event::Failed(e.to_string()));
            return;
        }
    };
    clients.insert(
        id,
        Client {
            events: job.events,
            decoder: IncrementalDecoder::new(),
            prompt_tokens,
        },
    );
    scheduler.submit(sequence);
}

/// A failed pass takes down every sequence that was in it, because none of them
/// can tell whether their slot was written before it stopped.
fn fail(scheduler: &mut Scheduler, clients: &mut HashMap<u64, Client>, ids: &[u64], message: &str) {
    for &id in ids {
        if let Some(client) = clients.remove(&id) {
            let _ = client.events.send(Event::Failed(message.to_string()));
        }
        scheduler.cancel(id);
    }
    scheduler.drain_finished();
}
