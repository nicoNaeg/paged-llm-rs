//! Picking the next token from a logit vector.
//!
//! On the serving path, so written here rather than taken from a crate. The
//! random number generator is hand-written too, and that one is not dogma: a
//! seeded generator whose sequence is fixed by this repository is what lets a
//! test assert which tokens come out, and a distribution be checked rather than
//! assumed.

use crate::{Error, Result};

/// A seeded generator, `splitmix64`.
///
/// Enough for choosing a token: the draw only has to be uniform over one
/// sequence of a few hundred picks, not to survive a statistical battery. Its
/// value here is that the same seed gives the same tokens, on any machine.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Start from `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `[0, 1)`.
    ///
    /// Built from the top 24 bits, which is what an f32 mantissa holds exactly,
    /// so every value it can produce is evenly spaced.
    #[allow(clippy::cast_precision_loss)]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / f32::from(1u16 << 8) / 65536.0
    }
}

/// How the next token is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// Flattens the distribution above one, sharpens it below. Zero means the
    /// most likely token every time, with no draw at all.
    temperature: f32,
    /// Consider only this many of the highest-scoring tokens.
    top_k: Option<usize>,
    /// Consider the shortest set of highest-scoring tokens whose probabilities
    /// reach this much.
    top_p: Option<f32>,
}

impl Sampling {
    /// Always the highest-scoring token.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        }
    }

    /// Build and validate. A parameter outside its range is refused rather than
    /// clamped, because a client that asked for it should hear that it was not
    /// honoured.
    pub fn new(temperature: f32, top_k: Option<usize>, top_p: Option<f32>) -> Result<Self> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(Error::Config(format!(
                "temperature {temperature}, which has to be zero or above"
            )));
        }
        if top_k == Some(0) {
            return Err(Error::Config(
                "top_k 0, which would leave no token to choose from".into(),
            ));
        }
        if let Some(p) = top_p
            && (!p.is_finite() || p <= 0.0 || p > 1.0)
        {
            return Err(Error::Config(format!(
                "top_p {p}, which has to sit in (0, 1]"
            )));
        }
        Ok(Self {
            temperature,
            top_k,
            top_p,
        })
    }

    /// Whether this picks the highest-scoring token without drawing.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Choose a token id from `logits`.
    ///
    /// The order is temperature, then top-k, then top-p, then the draw, which is
    /// the order the reference implementations use. Applying top-p before the
    /// temperature would measure the mass of a distribution that is not the one
    /// being drawn from.
    ///
    /// # Panics
    ///
    /// If the vocabulary holds more than `u32::MAX` tokens, which no tokenizer
    /// format can express.
    pub fn sample(&self, logits: &[f32], rng: &mut Rng) -> Result<u32> {
        if logits.is_empty() {
            return Err(Error::Config(
                "cannot sample from an empty logit vector".into(),
            ));
        }
        if self.is_greedy() {
            return Ok(argmax(logits));
        }

        let inverse = 1.0 / self.temperature;
        let mut candidates: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(id, &logit)| {
                (
                    u32::try_from(id).expect("a vocabulary fits in u32"),
                    logit * inverse,
                )
            })
            .collect();

        // Selecting the k largest costs a pass, where sorting a vocabulary of a
        // hundred and fifty thousand costs a sort, once per token generated.
        if let Some(k) = self.top_k {
            let k = k.min(candidates.len());
            if k < candidates.len() {
                candidates.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
                candidates.truncate(k);
            }
        }
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Softmax over what survived, shifted by the largest so the exponential
        // cannot overflow.
        let largest = candidates[0].1;
        let mut total = 0.0f32;
        for candidate in &mut candidates {
            candidate.1 = (candidate.1 - largest).exp();
            total += candidate.1;
        }

        if let Some(p) = self.top_p
            && p < 1.0
        {
            let target = p * total;
            let mut running = 0.0f32;
            let mut keep = 1;
            for (position, candidate) in candidates.iter().enumerate() {
                running += candidate.1;
                keep = position + 1;
                if running >= target {
                    break;
                }
            }
            // The token that crosses the threshold is kept, not dropped, so the
            // set always reaches p rather than stopping short of it.
            candidates.truncate(keep);
            total = candidates.iter().map(|c| c.1).sum();
        }

        let mut point = rng.next_f32() * total;
        for candidate in &candidates {
            point -= candidate.1;
            if point <= 0.0 {
                return Ok(candidate.0);
            }
        }
        // Reached only when rounding leaves a sliver past the last bucket.
        Ok(candidates[candidates.len() - 1].0)
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (id, value) in logits.iter().enumerate() {
        if value > &logits[best] {
            best = id;
        }
    }
    u32::try_from(best).expect("a vocabulary fits in u32")
}

#[cfg(test)]
mod tests {
    use super::{Rng, Sampling};

    /// Logits whose softmax at temperature one is close to 0.64, 0.24, 0.09,
    /// 0.03 over four tokens.
    fn spread() -> Vec<f32> {
        vec![1.0, 0.0, -1.0, -2.0]
    }

    #[test]
    fn greedy_takes_the_largest_and_never_draws() {
        let mut rng = Rng::new(1);
        let before = rng.clone();
        assert_eq!(Sampling::greedy().sample(&spread(), &mut rng).unwrap(), 0);
        // The generator has not moved, so a greedy request cannot shift the
        // sequence a later sampled request draws from.
        assert_eq!(rng.next_u64(), before.clone().next_u64());
    }

    #[test]
    fn top_k_of_one_is_greedy_whatever_the_temperature() {
        let sampling = Sampling::new(5.0, Some(1), None).unwrap();
        let mut rng = Rng::new(7);
        for _ in 0..50 {
            assert_eq!(sampling.sample(&spread(), &mut rng).unwrap(), 0);
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_tokens_and_a_different_one_does_not() {
        let sampling = Sampling::new(1.0, None, None).unwrap();
        let draw = |seed| {
            let mut rng = Rng::new(seed);
            (0..40)
                .map(|_| sampling.sample(&spread(), &mut rng).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(42), draw(42));
        assert_ne!(draw(42), draw(43));
    }

    #[test]
    fn the_draw_follows_the_distribution_it_claims() {
        let sampling = Sampling::new(1.0, None, None).unwrap();
        let mut rng = Rng::new(20_260_726);
        let logits = spread();
        let draws = 200_000;
        let mut counts = [0u32; 4];
        for _ in 0..draws {
            counts[sampling.sample(&logits, &mut rng).unwrap() as usize] += 1;
        }

        let total: f32 = logits.iter().map(|l| l.exp()).sum();
        for (id, logit) in logits.iter().enumerate() {
            let want = logit.exp() / total;
            let got = f64::from(counts[id]) / f64::from(draws);
            assert!(
                (got - f64::from(want)).abs() < 0.005,
                "token {id}: drew {got:.4}, distribution says {want:.4}"
            );
        }
    }

    #[test]
    fn top_p_keeps_the_token_that_crosses_the_threshold() {
        // The largest token alone holds about 0.64 of the mass, so a threshold
        // of 0.5 is crossed by it and nothing else survives.
        let sampling = Sampling::new(1.0, None, Some(0.5)).unwrap();
        let mut rng = Rng::new(3);
        for _ in 0..50 {
            assert_eq!(sampling.sample(&spread(), &mut rng).unwrap(), 0);
        }

        // At 0.8 the second token is needed to reach it, and nothing beyond.
        let sampling = Sampling::new(1.0, None, Some(0.8)).unwrap();
        let mut seen = [false; 4];
        for _ in 0..200 {
            seen[sampling.sample(&spread(), &mut rng).unwrap() as usize] = true;
        }
        assert_eq!(seen, [true, true, false, false]);
    }

    #[test]
    fn a_parameter_outside_its_range_is_refused_rather_than_clamped() {
        assert!(Sampling::new(-0.1, None, None).is_err());
        assert!(Sampling::new(f32::NAN, None, None).is_err());
        assert!(Sampling::new(1.0, Some(0), None).is_err());
        assert!(Sampling::new(1.0, None, Some(0.0)).is_err());
        assert!(Sampling::new(1.0, None, Some(1.5)).is_err());
        assert!(Sampling::new(1.0, Some(20), Some(0.95)).is_ok());
    }

    #[test]
    fn an_empty_logit_vector_is_an_error_rather_than_a_panic() {
        let mut rng = Rng::new(1);
        assert!(Sampling::greedy().sample(&[], &mut rng).is_err());
        assert!(
            Sampling::new(1.0, None, None)
                .unwrap()
                .sample(&[], &mut rng)
                .is_err()
        );
    }

    #[test]
    fn a_top_k_past_the_vocabulary_is_the_whole_vocabulary() {
        let sampling = Sampling::new(1.0, Some(1000), None).unwrap();
        let mut rng = Rng::new(11);
        let mut seen = [false; 4];
        for _ in 0..500 {
            seen[sampling.sample(&spread(), &mut rng).unwrap() as usize] = true;
        }
        assert_eq!(seen, [true; 4]);
    }

    // The bucket index comes from a value the assertion above pins inside
    // [0, 1), so the narrowing cannot lose a sign or overflow.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[test]
    fn the_generator_spreads_over_the_unit_interval() {
        let mut rng = Rng::new(99);
        let mut buckets = [0u32; 10];
        for _ in 0..100_000 {
            let x = rng.next_f32();
            assert!((0.0..1.0).contains(&x), "{x} is outside [0, 1)");
            buckets[(x * 10.0) as usize] += 1;
        }
        for (i, count) in buckets.iter().enumerate() {
            assert!(
                (9_000..11_000).contains(count),
                "bucket {i} holds {count} of 100000"
            );
        }
    }
}
