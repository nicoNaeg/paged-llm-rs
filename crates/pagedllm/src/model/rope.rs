//! Rotary position embedding.

use candle_core::{D, DType, Device, Tensor};

use crate::Result;

/// Precomputed cosines and sines, one row per position.
///
/// The pairing convention matters and is not the only one in use. This is the
/// one `HuggingFace` ships, which splits a head in half and rotates the halves
/// against each other, so dimension `i` pairs with `i + d/2`. The interleaved
/// convention, where `2i` pairs with `2i+1`, is a different embedding on the
/// same weights: it runs, and it puts every position in the wrong place.
#[derive(Debug)]
pub struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    /// Build the table for positions `0..max_positions`.
    ///
    /// Angles are accumulated in f64 on the host and only the two tables reach
    /// the device. At a base of one million and a position in the thousands,
    /// computing the frequency in the model's dtype loses the low bits of the
    /// angle, and Metal has no f64 trigonometry to do it there instead. Both
    /// point the same way: this is a table built once, not work for the GPU.
    // A sine and a cosine live in [-1, 1], which f32 represents with room to
    // spare. The narrowing costs mantissa bits, which is the point of doing the
    // angle in f64 first, and cannot lose magnitude.
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(
        head_dim: usize,
        max_positions: usize,
        theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();
        let count = max_positions * head_dim;
        let mut cos = Vec::with_capacity(count);
        let mut sin = Vec::with_capacity(count);
        for position in 0..max_positions {
            // The table holds each angle twice, once for each half of the head,
            // which is what makes the rotation expressible as two elementwise
            // products instead of a gather.
            for _ in 0..2 {
                for f in &inv_freq {
                    let angle = position as f64 * f;
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
        let shape = (max_positions, head_dim);
        Ok(Self {
            cos: Tensor::from_vec(cos, shape, device)?.to_dtype(dtype)?,
            sin: Tensor::from_vec(sin, shape, device)?.to_dtype(dtype)?,
        })
    }

    /// Rotate `x`, shaped `[batch, heads, seq, head_dim]`, for positions
    /// starting at `offset`.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let seq = x.dim(D::Minus2)?;
        let cos = self.cos.narrow(0, offset, seq)?;
        let sin = self.sin.narrow(0, offset, seq)?;
        let rotated = Self::rotate_half(x)?;
        Ok((x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
    }

    /// `[x1, x2] -> [-x2, x1]` over the last dimension.
    fn rotate_half(x: &Tensor) -> Result<Tensor> {
        let half = x.dim(D::Minus1)? / 2;
        let first = x.narrow(D::Minus1, 0, half)?;
        let second = x.narrow(D::Minus1, half, half)?;
        Ok(Tensor::cat(&[&second.neg()?, &first], D::Minus1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::Rope;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn position_zero_leaves_a_vector_untouched() {
        let dev = Device::Cpu;
        let rope = Rope::new(8, 4, 1e6, DType::F32, &dev).unwrap();
        let x = Tensor::arange(0f32, 8f32, &dev)
            .unwrap()
            .reshape((1, 1, 1, 8))
            .unwrap();
        let out = rope.apply(&x, 0).unwrap();
        let before = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let after = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (b, a) in before.iter().zip(&after) {
            assert!((b - a).abs() < 1e-6, "{before:?} became {after:?}");
        }
    }

    #[test]
    fn rotation_preserves_the_norm_of_each_pair() {
        let dev = Device::Cpu;
        let head_dim = 8;
        let rope = Rope::new(head_dim, 16, 1e6, DType::F32, &dev).unwrap();
        let x = Tensor::arange(1f32, 9f32, &dev)
            .unwrap()
            .reshape((1, 1, 1, head_dim))
            .unwrap();
        let out = rope.apply(&x, 7).unwrap();
        let before = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let after = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // A rotation acts on the pair (i, i + half), so each pair keeps its
        // length even though neither component does.
        let half = head_dim / 2;
        for i in 0..half {
            let n0 = before[i].hypot(before[i + half]);
            let n1 = after[i].hypot(after[i + half]);
            assert!((n0 - n1).abs() < 1e-5, "pair {i}: {n0} became {n1}");
        }
    }

    #[test]
    fn a_position_offset_matches_taking_the_same_row_from_the_start() {
        let dev = Device::Cpu;
        let rope = Rope::new(8, 32, 1e6, DType::F32, &dev).unwrap();
        let x = Tensor::arange(1f32, 9f32, &dev)
            .unwrap()
            .reshape((1, 1, 1, 8))
            .unwrap();
        let long = Tensor::cat(&[&x, &x, &x, &x, &x], 2).unwrap();
        let from_zero = rope.apply(&long, 0).unwrap();
        let at_three = rope.apply(&x, 3).unwrap();
        let expected = from_zero
            .narrow(2, 3, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got = at_three.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (e, g) in expected.iter().zip(&got) {
            assert!((e - g).abs() < 1e-6, "{expected:?} against {got:?}");
        }
    }
}
