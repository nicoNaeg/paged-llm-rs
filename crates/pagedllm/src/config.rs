//! The model's shape, read from the `config.json` published alongside the
//! weights.
//!
//! Fields this engine does not honour are still parsed and then refused. A
//! config that asks for a sliding window or a bias this code silently ignores
//! would produce a model that runs, generates fluent text and is wrong, which
//! is the failure mode worth spending an error on.

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::{Error, Result};

/// Architecture of a Qwen3 model.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Width of the residual stream.
    pub hidden_size: usize,
    /// Width of the MLP's hidden layer.
    pub intermediate_size: usize,
    /// Number of transformer blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key and value heads. Lower than the query head count under
    /// grouped-query attention, which is what shrinks the KV cache.
    pub num_key_value_heads: usize,
    /// Width of one attention head.
    ///
    /// Qwen3 decouples this from `hidden_size / num_attention_heads`: at 1024
    /// hidden and 16 heads the quotient is 64, and the real head is 128 wide,
    /// so `q_proj` is 1024 by 2048 rather than square.
    pub head_dim: usize,
    /// Epsilon inside the RMS norms.
    pub rms_norm_eps: f64,
    /// Size of the vocabulary, and of the logit vector a step produces.
    pub vocab_size: usize,
    /// Longest position the rotary embedding was trained for.
    pub max_position_embeddings: usize,
    /// Whether the output projection reuses the embedding matrix.
    pub tie_word_embeddings: bool,

    /// Architecture family. Refused unless it names Qwen3.
    pub model_type: String,
    /// Activation in the MLP. Refused unless it names `SiLU`.
    pub hidden_act: String,
    /// Whether the attention projections carry a bias. Refused when true.
    #[serde(default)]
    pub attention_bias: bool,
    /// Width of the sliding attention window. Refused when set.
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// Rotary base as transformers 4 wrote it, flat. Read through
    /// [`Config::rope_theta`].
    #[serde(default, rename = "rope_theta")]
    flat_rope_theta: Option<f64>,
    /// Rotary settings as transformers 5 writes them, nested.
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    /// Rotary scaling as transformers 4 wrote it. Refused when set.
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
}

/// The nested rotary block transformers 5 writes.
#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
    #[serde(default)]
    rope_type: Option<String>,
}

impl Config {
    /// Read and validate a `config.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|e| Error::Io(path.into(), e))?;
        let config: Self = serde_json::from_str(&text).map_err(|e| Error::Config(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Reject a config whose fields this implementation would otherwise ignore.
    fn validate(&self) -> Result<()> {
        let refuse = |what: String| Err(Error::Unsupported(what));
        if self.model_type != "qwen3" {
            return refuse(format!(
                "model_type {:?}, only qwen3 is implemented",
                self.model_type
            ));
        }
        if self.hidden_act != "silu" {
            return refuse(format!(
                "hidden_act {:?}, only silu is implemented",
                self.hidden_act
            ));
        }
        if self.attention_bias {
            return refuse("attention_bias, the projections here carry no bias".into());
        }
        if let Some(window) = self.sliding_window {
            return refuse(format!("sliding_window {window}, attention here is dense"));
        }
        if self.rope_scaling.is_some() {
            return refuse("rope_scaling, the rotary table here is unscaled".into());
        }
        if let Some(kind) = self
            .rope_parameters
            .as_ref()
            .and_then(|p| p.rope_type.as_deref())
            && kind != "default"
        {
            return refuse(format!(
                "rope_type {kind:?}, the rotary table here is unscaled"
            ));
        }
        self.rope_theta()?;
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return refuse(format!(
                "{} query heads over {} key heads, which do not group evenly",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        Ok(())
    }

    /// Base of the rotary position frequencies.
    ///
    /// transformers 4 wrote this flat and transformers 5 moved it under
    /// `rope_parameters`. Both layouts are in the wild and both are in this
    /// project: Qwen3-0.6B was published under the old one, and the fixture the
    /// tests are checked against is written by the new one.
    pub fn rope_theta(&self) -> Result<f64> {
        self.flat_rope_theta
            .or_else(|| self.rope_parameters.as_ref().map(|p| p.rope_theta))
            .ok_or_else(|| {
                Error::Config("no rope_theta, in either the flat or the nested layout".into())
            })
    }

    /// How many query heads share one key and value head.
    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Width of the concatenated query heads, which is what `q_proj` outputs.
    pub fn query_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Width of the concatenated key or value heads.
    pub fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::Config;

    fn qwen3_06b() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen3",
            "hidden_act": "silu",
            "attention_bias": false,
            "sliding_window": null,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "vocab_size": 151_936,
            "max_position_embeddings": 40_960,
            "tie_word_embeddings": true
        })
    }

    fn parse(value: &serde_json::Value) -> crate::Result<Config> {
        let config: Config = serde_json::from_value(value.clone()).unwrap();
        config.validate().map(|()| config)
    }

    #[test]
    fn the_published_config_parses_and_its_widths_follow() {
        let config = parse(&qwen3_06b()).unwrap();
        assert_eq!(config.kv_group_size(), 2);
        // The whole point of head_dim being its own field: 16 * 128 is not 1024.
        assert_eq!(config.query_width(), 2048);
        assert_eq!(config.kv_width(), 1024);
        assert_ne!(config.query_width(), config.hidden_size);
    }

    #[test]
    fn a_field_this_engine_ignores_is_refused_rather_than_ignored() {
        for (field, value) in [
            ("model_type", serde_json::json!("llama")),
            ("hidden_act", serde_json::json!("gelu")),
            ("attention_bias", serde_json::json!(true)),
            ("sliding_window", serde_json::json!(4096)),
        ] {
            let mut config = qwen3_06b();
            config[field] = value;
            assert!(
                parse(&config).is_err(),
                "{field} was accepted and would be ignored"
            );
        }
    }

    #[test]
    fn the_rotary_base_is_read_from_either_layout() {
        let flat = parse(&qwen3_06b()).unwrap();
        assert!((flat.rope_theta().unwrap() - 1e6).abs() < f64::EPSILON);

        // What transformers 5 writes, which is what the test fixture carries.
        let mut nested = qwen3_06b();
        nested.as_object_mut().unwrap().remove("rope_theta");
        nested["rope_parameters"] =
            serde_json::json!({ "rope_theta": 1_000_000.0, "rope_type": "default" });
        assert!((parse(&nested).unwrap().rope_theta().unwrap() - 1e6).abs() < f64::EPSILON);

        let mut neither = qwen3_06b();
        neither.as_object_mut().unwrap().remove("rope_theta");
        assert!(parse(&neither).is_err());
    }

    #[test]
    fn a_scaled_rotary_table_is_refused_in_either_layout() {
        let mut scaled = qwen3_06b();
        scaled["rope_scaling"] = serde_json::json!({ "type": "yarn", "factor": 4.0 });
        assert!(parse(&scaled).is_err());

        let mut nested = qwen3_06b();
        nested["rope_parameters"] =
            serde_json::json!({ "rope_theta": 1_000_000.0, "rope_type": "yarn" });
        assert!(parse(&nested).is_err());
    }

    #[test]
    fn head_counts_that_do_not_group_evenly_are_refused() {
        let mut config = qwen3_06b();
        config["num_key_value_heads"] = serde_json::json!(5);
        assert!(parse(&config).is_err());
    }
}
