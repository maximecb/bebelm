//! Decode-time state: a KV cache (attention layers) and a conv-state cache (conv layers).
//!
//! Indexed by absolute layer number (0..N_LAYERS); only the relevant slots are used per
//! layer type. The KV buffers grow by one position per token; the conv state is fixed at
//! the last `CONV_L_CACHE-1` columns of Bx.

use crate::config::{CONV_L_CACHE, HIDDEN, N_LAYERS};

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
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}
