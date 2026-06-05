//! Decode-time state: a KV cache (attention layers) and a conv-state cache (conv layers).
//!
//! Indexed by absolute layer number (0..N_LAYERS); only the relevant slots are used per
//! layer type. The KV buffers grow by one position per token; the conv state is fixed at
//! the last `CONV_L_CACHE-1` columns of Bx.

use crate::config::{CONV_L_CACHE, HIDDEN, KV_DIM, N_LAYERS};

pub struct Cache {
    /// Per attention layer: appended key history (`KV_DIM` floats per position).
    pub k: Vec<Vec<f32>>,
    /// Per attention layer: appended value history.
    pub v: Vec<Vec<f32>>,
    /// Per conv layer: the last `CONV_L_CACHE-1` columns of Bx (oldest first), `HIDDEN` each.
    pub conv: Vec<Vec<f32>>,
    /// Number of tokens processed so far (the next token's position).
    pub pos: usize,
}

impl Cache {
    pub fn new() -> Self {
        Cache {
            k: (0..N_LAYERS).map(|_| Vec::new()).collect(),
            v: (0..N_LAYERS).map(|_| Vec::new()).collect(),
            conv: (0..N_LAYERS).map(|_| vec![0.0; HIDDEN * (CONV_L_CACHE - 1)]).collect(),
            pos: 0,
        }
    }

    /// Number of positions currently held in the KV attention window. Conv layers keep no KV
    /// history, so this is the longest layer's key buffer.
    pub fn kv_len(&self) -> usize {
        self.k.iter().map(Vec::len).max().unwrap_or(0) / KV_DIM
    }

    /// Drop the oldest `n` positions from every attention layer's KV history — a sliding window
    /// so decoding can continue past a context cap. `pos` (the absolute RoPE position) is left
    /// untouched: RoPE attention depends only on the query↔key offset, which stays in-window.
    pub fn evict_front(&mut self, n: usize) {
        let drop = n * KV_DIM;
        for (k, v) in self.k.iter_mut().zip(self.v.iter_mut()) {
            if k.len() >= drop {
                k.drain(0..drop);
                v.drain(0..drop);
            }
        }
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evict_front_slides_attention_layers() {
        let mut c = Cache::new();
        // Two attention layers accumulate 3 positions of KV; conv layers stay empty.
        c.k[2] = vec![1.0; 3 * KV_DIM];
        c.v[2] = vec![2.0; 3 * KV_DIM];
        c.k[6] = vec![1.0; 3 * KV_DIM];
        c.v[6] = vec![2.0; 3 * KV_DIM];
        assert_eq!(c.kv_len(), 3);

        c.evict_front(1);
        assert_eq!(c.kv_len(), 2);
        assert_eq!(c.k[2].len(), 2 * KV_DIM);
        assert_eq!(c.v[6].len(), 2 * KV_DIM);
        // Empty (conv-layer) buffers are untouched.
        assert!(c.k[0].is_empty());
    }
}
