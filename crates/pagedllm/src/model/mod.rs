//! The Qwen3 forward pass.
//!
//! Written against candle's primitives rather than taken from
//! `candle-transformers`, because the attention here has to become able to read
//! a KV cache scattered across physical blocks, and an implementation that owns
//! one contiguous cache per sequence cannot.

mod attention;
mod layers;
mod rope;

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};

use crate::cache::KvCache;
use crate::config::Config;
use crate::weights::Weights;
use crate::{Error, Result};

use attention::Attention;
use layers::{Linear, Mlp, RmsNorm};
use rope::Rope;

/// Named intermediates captured during a forward pass.
///
/// Names match the module paths the reference implementation uses, so a fixture
/// dumped from it and a trace taken here compare key by key. Off by default:
/// recording clones a tensor handle per module, which is cheap, but keeping
/// every intermediate alive across 28 layers is not.
#[derive(Debug, Default)]
pub struct Trace {
    tensors: HashMap<String, Tensor>,
    recording: bool,
}

impl Trace {
    /// A trace that keeps what it is given.
    pub fn recording() -> Self {
        Self {
            tensors: HashMap::new(),
            recording: true,
        }
    }

    /// A trace that discards everything, which is what serving uses.
    pub fn off() -> Self {
        Self::default()
    }

    pub(crate) fn record(&mut self, name: &str, tensor: &Tensor) {
        if self.recording {
            self.tensors.insert(name.to_string(), tensor.clone());
        }
    }

    /// Look up one captured tensor.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Every name captured, sorted.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// How many tensors were captured.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether nothing was captured.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// One transformer block: attention and MLP, each behind a norm and a residual.
#[derive(Debug)]
struct Block {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        rope: &Rope,
        offset: usize,
        cache: Option<(&mut KvCache, usize)>,
        trace: &mut Trace,
        prefix: &str,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        trace.record(&format!("{prefix}.input_layernorm.out"), &normed);
        let attn = self.self_attn.forward(
            &normed,
            rope,
            offset,
            cache,
            trace,
            &format!("{prefix}.self_attn"),
        )?;
        let x = (x + attn)?;

        let normed = self.post_attention_layernorm.forward(&x)?;
        trace.record(&format!("{prefix}.post_attention_layernorm.out"), &normed);
        let mlp = self.mlp.forward(&normed, trace, &format!("{prefix}.mlp"))?;

        let out = (x + mlp)?;
        trace.record(&format!("{prefix}.out"), &out);
        Ok(out)
    }
}

/// A loaded Qwen3 model.
#[derive(Debug)]
pub struct Model {
    embed_tokens: Tensor,
    blocks: Vec<Block>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    config: Config,
    device: Device,
    dtype: DType,
}

impl Model {
    /// Load a checkpoint directory holding `config.json` and
    /// `model.safetensors`.
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Self::load_as(dir, device, None)
    }

    /// Same, converting the weights to `dtype` on the way in.
    ///
    /// `None` keeps the checkpoint's own dtype. A bf16 checkpoint needs a cast
    /// to run on the CPU at all, since candle has no bf16 matmul there.
    pub fn load_as(dir: impl AsRef<Path>, device: &Device, dtype: Option<DType>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Config::from_file(dir.join("config.json"))?;
        let mut weights = Weights::load(dir.join("model.safetensors"), device)?;
        if let Some(dtype) = dtype {
            weights = weights.cast_to(dtype);
        }
        Self::from_weights(config, &weights, device)
    }

    /// Assemble from a parsed config and a loaded checkpoint.
    pub fn from_weights(config: Config, weights: &Weights, device: &Device) -> Result<Self> {
        let dtype = weights.dtype("model.embed_tokens.weight")?;
        let hidden = config.hidden_size;
        let embed_tokens =
            weights.get("model.embed_tokens.weight", &[config.vocab_size, hidden])?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let p = format!("model.layers.{layer}");
            let linear = |name: &str, out: usize, inp: usize| -> Result<Linear> {
                Linear::new(&weights.get(&format!("{p}.{name}.weight"), &[out, inp])?)
            };
            let norm = |name: &str, width: usize| -> Result<RmsNorm> {
                Ok(RmsNorm::new(
                    weights.get(&format!("{p}.{name}.weight"), &[width])?,
                    config.rms_norm_eps,
                ))
            };
            blocks.push(Block {
                input_layernorm: norm("input_layernorm", hidden)?,
                self_attn: Attention::new(
                    linear("self_attn.q_proj", config.query_width(), hidden)?,
                    linear("self_attn.k_proj", config.kv_width(), hidden)?,
                    linear("self_attn.v_proj", config.kv_width(), hidden)?,
                    linear("self_attn.o_proj", hidden, config.query_width())?,
                    // Both norms are as wide as one head, not as wide as the
                    // concatenated heads. That is what says they run after the
                    // split.
                    norm("self_attn.q_norm", config.head_dim)?,
                    norm("self_attn.k_norm", config.head_dim)?,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                ),
                post_attention_layernorm: norm("post_attention_layernorm", hidden)?,
                mlp: Mlp::new(
                    linear("mlp.gate_proj", config.intermediate_size, hidden)?,
                    linear("mlp.up_proj", config.intermediate_size, hidden)?,
                    linear("mlp.down_proj", hidden, config.intermediate_size)?,
                ),
            });
        }

        let norm = RmsNorm::new(
            weights.get("model.norm.weight", &[hidden])?,
            config.rms_norm_eps,
        );

        // A tied checkpoint may or may not materialise the output projection.
        // Qwen3-0.6B declares the weights tied and ships `lm_head.weight`
        // anyway, so preferring the stored tensor covers both without a branch
        // that only one of them exercises.
        let head_weight =
            match weights.get_optional("lm_head.weight", &[config.vocab_size, hidden])? {
                Some(weight) => weight,
                None if config.tie_word_embeddings => embed_tokens.clone(),
                None => {
                    return Err(Error::Weight(
                        "no lm_head.weight, and the config does not tie the embeddings".into(),
                    ));
                }
            };
        let lm_head = Linear::new(&head_weight)?;

        let rope = Rope::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta()?,
            dtype,
            device,
        )?;

        Ok(Self {
            embed_tokens,
            blocks,
            norm,
            lm_head,
            rope,
            config,
            device: device.clone(),
            dtype,
        })
    }

    /// Logits for one sequence, shaped `[1, tokens.len(), vocab_size]`.
    pub fn forward(&self, tokens: &[u32]) -> Result<Tensor> {
        self.forward_traced(tokens, 0, None, &mut Trace::off())
    }

    /// Logits for `tokens`, continuing the sequence whose keys and values are
    /// already in `cache`, and adding this step's to it.
    pub fn forward_cached(&self, tokens: &[u32], cache: &mut KvCache) -> Result<Tensor> {
        let offset = cache.tokens();
        self.forward_traced(tokens, offset, Some(cache), &mut Trace::off())
    }

    /// Same, recording every intermediate into `trace`.
    ///
    /// `offset` is the position the first token sits at, which is zero while
    /// there is no cache to continue from.
    pub fn forward_traced(
        &self,
        tokens: &[u32],
        offset: usize,
        cache: Option<&mut KvCache>,
        trace: &mut Trace,
    ) -> Result<Tensor> {
        if tokens.is_empty() {
            return Err(Error::Config(
                "cannot run a forward pass on no tokens".into(),
            ));
        }
        let ids = Tensor::new(tokens, &self.device)?;
        let mut x = self.embed_tokens.index_select(&ids, 0)?.unsqueeze(0)?;
        trace.record("model.embed_tokens.out", &x);

        let mut cache = cache;
        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward(
                &x,
                &self.rope,
                offset,
                cache.as_deref_mut().map(|c| (c, layer)),
                trace,
                &format!("model.layers.{layer}"),
            )?;
        }
        if let Some(cache) = cache {
            // Once per pass rather than once per layer: every layer caches the
            // same positions, so counting per layer would multiply the sequence
            // length by the depth of the model.
            cache.advance(tokens.len());
        }

        let x = self.norm.forward(&x)?;
        trace.record("model.norm.out", &x);

        let logits = self.lm_head.forward(&x)?;
        trace.record("lm_head.out", &logits);
        trace.record("logits", &logits);
        Ok(logits)
    }

    /// The architecture this model was loaded at.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The dtype the weights were stored in, which is what the forward pass
    /// computes in outside the norms and the softmax.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Where the weights live.
    pub fn device(&self) -> &Device {
        &self.device
    }
}
