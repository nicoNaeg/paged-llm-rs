//! The keys and values a sequence has already produced.
//!
//! One contiguous tensor pair per layer, grown by concatenation. That is the
//! naive layout, and it is here on purpose: it is what stage 5 replaces with a
//! pool of fixed-size blocks, and replacing something that was measured first is
//! the point of building it.
//!
//! Two costs it carries, both of which the paged version exists to remove. Every
//! step reallocates and copies the whole cache to append one token, so growth is
//! quadratic in the total bytes moved. And a sequence's cache has to be one
//! unbroken run of memory, so serving several at once means reserving the
//! longest each might reach rather than what each actually uses.

use candle_core::Tensor;

use crate::{Error, Result};

/// The cached keys and values of one sequence, across every layer.
#[derive(Debug)]
pub struct KvCache {
    /// Per layer, keys and values shaped `[batch, kv_heads, tokens, head_dim]`.
    layers: Vec<Option<(Tensor, Tensor)>>,
    tokens: usize,
}

impl KvCache {
    /// An empty cache for a model of `num_layers` layers.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            tokens: 0,
        }
    }

    /// How many token positions the cache holds, which is where the next token
    /// sits.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Whether nothing has been cached yet.
    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    /// Forget everything, so the buffers can be reused by another sequence.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            *layer = None;
        }
        self.tokens = 0;
    }

    /// Append this step's keys and values to `layer`, and return everything the
    /// attention should read.
    ///
    /// The keys handed in are already rotated. Caching them before the rotation
    /// would mean re-rotating the whole history on every step, which is the one
    /// thing the cache exists to avoid.
    pub(crate) fn append(
        &mut self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let slot = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| Error::Config(format!("layer {layer} is past the end of the cache")))?;
        let merged = match slot.take() {
            None => (k.clone(), v.clone()),
            Some((past_k, past_v)) => (
                Tensor::cat(&[&past_k, k], 2)?.contiguous()?,
                Tensor::cat(&[&past_v, v], 2)?.contiguous()?,
            ),
        };
        *slot = Some(merged.clone());
        Ok(merged)
    }

    /// Record that a step added `count` positions.
    ///
    /// Called once per forward pass rather than once per layer, since every
    /// layer caches the same positions and counting them per layer would
    /// multiply the length by the depth of the model.
    pub(crate) fn advance(&mut self, count: usize) {
        self.tokens += count;
    }
}

#[cfg(test)]
mod tests {
    use super::KvCache;
    use candle_core::{Device, Tensor};

    fn step(tokens: usize) -> Tensor {
        Tensor::zeros((1, 2, tokens, 4), candle_core::DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn appending_grows_the_layer_and_leaves_the_others_alone() {
        let mut cache = KvCache::new(3);
        let (k, v) = cache.append(0, &step(5), &step(5)).unwrap();
        assert_eq!(k.dims(), &[1, 2, 5, 4]);
        assert_eq!(v.dims(), &[1, 2, 5, 4]);

        let (k, _) = cache.append(0, &step(1), &step(1)).unwrap();
        assert_eq!(k.dims(), &[1, 2, 6, 4]);

        // Layer 1 never saw those six tokens.
        let (k, _) = cache.append(1, &step(2), &step(2)).unwrap();
        assert_eq!(k.dims(), &[1, 2, 2, 4]);
    }

    #[test]
    fn the_length_counts_positions_and_not_layers() {
        let mut cache = KvCache::new(4);
        for layer in 0..4 {
            cache.append(layer, &step(7), &step(7)).unwrap();
        }
        cache.advance(7);
        assert_eq!(cache.tokens(), 7);
    }

    #[test]
    fn clearing_returns_it_to_empty() {
        let mut cache = KvCache::new(2);
        cache.append(0, &step(3), &step(3)).unwrap();
        cache.advance(3);
        cache.clear();
        assert!(cache.is_empty());
        let (k, _) = cache.append(0, &step(1), &step(1)).unwrap();
        assert_eq!(k.dims(), &[1, 2, 1, 4]);
    }

    #[test]
    fn a_layer_past_the_end_is_an_error_rather_than_a_panic() {
        let mut cache = KvCache::new(2);
        assert!(cache.append(2, &step(1), &step(1)).is_err());
    }
}
