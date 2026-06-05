//! The one sampler (KISS): temperature + top-k, with `temperature == 0` ⇒ greedy argmax,
//! plus an optional repetition penalty. Hand-rolled xorshift PRNG (no `rand` crate).
//!
//! Defaults follow Liquid's recommendation for LFM2.5-8B-A1B: temperature 0.2, top-k 80,
//! repeat-penalty 1.05.

/// Default PRNG seed for non-greedy sampling when the caller doesn't supply one.
const DEFAULT_SEED: u64 = 0x853C_49E6_748F_EA9B;

/// Sampling configuration + PRNG state.
pub struct Sampler {
    /// `0.0` ⇒ greedy (deterministic argmax).
    pub temperature: f32,
    /// Keep only the `top_k` highest-logit candidates (`0` ⇒ no limit).
    pub top_k: usize,
    /// Divide already-seen tokens' logits by this (`1.0` ⇒ disabled).
    pub repeat_penalty: f32,
    rng: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, repeat_penalty: f32, seed: u64) -> Self {
        Self { temperature, top_k, repeat_penalty, rng: seed | 1 }
    }

    /// Deterministic greedy decoding (argmax, no penalty).
    pub fn greedy() -> Self {
        Self::new(0.0, 0, 1.0, 0)
    }

    /// Liquid's recommended decoding for LFM2.5-8B-A1B: temperature 0.2, top-k 80,
    /// repeat-penalty 1.05.
    pub fn recommended() -> Self {
        Self::new(0.2, 80, 1.05, DEFAULT_SEED)
    }

    /// Pick the next token id from `logits` (mutated in place by the penalty/temperature).
    /// `history` is the tokens generated/seen so far (for the repetition penalty).
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        if self.repeat_penalty != 1.0 {
            for &tok in history {
                let l = &mut logits[tok as usize];
                // llama.cpp convention: divide if positive, multiply if negative.
                *l = if *l > 0.0 { *l / self.repeat_penalty } else { *l * self.repeat_penalty };
            }
        }

        if self.temperature <= 0.0 {
            return argmax(logits) as u32;
        }

        // Candidate set = the top-k logits (or all of them).
        let k = if self.top_k == 0 { logits.len() } else { self.top_k.min(logits.len()) };
        let mut cand: Vec<usize> = (0..logits.len()).collect();
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
            cand.truncate(k);
        }

        // Temperature-scaled softmax over the candidates.
        let max = cand.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = cand
            .iter()
            .map(|&i| ((logits[i] - max) / self.temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();

        // Inverse-CDF sample.
        let r = self.next_f32() * sum;
        let mut acc = 0.0;
        for (j, &p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return cand[j] as u32;
            }
        }
        cand[cand.len() - 1] as u32
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545F491_4F6CDD1D)
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::greedy();
        let mut logits = [1.0f32, 3.0, 2.0, -5.0];
        assert_eq!(s.sample(&mut logits, &[]), 1);
    }

    #[test]
    fn repeat_penalty_demotes_seen_tokens() {
        let mut s = Sampler::new(0.0, 0, 2.0, 0);
        // token 0 leads, but it's in history -> 10/2 = 5 < 9, so argmax becomes 1.
        let mut logits = [10.0f32, 9.0, 1.0];
        assert_eq!(s.sample(&mut logits, &[0]), 1);
    }

    #[test]
    fn top_k_1_is_greedy_even_with_temperature() {
        let mut s = Sampler::new(1.0, 1, 1.0, 12345);
        let mut logits = [0.5f32, 2.0, 1.0];
        for _ in 0..20 {
            let mut l = logits;
            assert_eq!(s.sample(&mut l, &[]), 1);
        }
        let _ = &mut logits;
    }

    #[test]
    fn sample_returns_a_top_k_candidate() {
        let mut s = Sampler::new(1.0, 2, 1.0, 99);
        let mut logits = [5.0f32, 4.0, -10.0, -20.0];
        for _ in 0..50 {
            let mut l = logits;
            let t = s.sample(&mut l, &[]);
            assert!(t == 0 || t == 1, "sampled outside top-2: {t}");
        }
        let _ = &mut logits;
    }
}
