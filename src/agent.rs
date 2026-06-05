//! A conversational session over a loaded [`Model`]: holds the running token transcript plus
//! the live KV / conv-state caches, so each turn only prefills the newly appended tokens
//! instead of replaying the whole conversation from scratch.
//!
//! Build up the prompt with the `append*` methods (which only grow the transcript), then run
//! the model with [`Agent::generate`] — or the [`Agent::assistant_turn`] convenience, which
//! wraps the ChatML assistant framing around a single `generate`.

use std::error::Error;
use std::time::{Duration, Instant};

use crate::cache::Cache;
use crate::model::Model;
use crate::sampler::Sampler;
use crate::tokenizer::{
    Tokenizer, TOKEN_BOS, TOKEN_ENDOFTEXT, TOKEN_IM_END, TOKEN_IM_START, TOKEN_PAD, TOKEN_THINK,
    TOKEN_THINK_END,
};

/// Default per-turn generation cap. A reasoning (`<think>`) turn can run long, so this is
/// generous; it only bounds a runaway turn.
const DEFAULT_MAX_GEN: usize = 2048;

/// Default cap on the transcript length (tokens) the agent will decode up to. The model
/// supports far more positions; this is a conservative session default.
const DEFAULT_MAX_CONTEXT: usize = 32_768;

/// Text appended to open an assistant turn before generating its reply.
const ASSISTANT_OPEN: &str = "<|im_start|>assistant\n";

/// Control tokens that end a turn if the model emits one as "content". Besides the normal
/// end-of-turn `<|im_end|>` (the EOS), a *sampled* turn can occasionally land on a document /
/// turn-boundary token — `<|endoftext|>`, `<|startoftext|>`, `<|pad|>`, or a stray
/// `<|im_start|>`. None is ever valid reply content, and decoding past one sends the model off
/// the rails, so we stop the turn at the first such token (as we do for EOS).
const STOP_TOKENS: [u32; 4] = [TOKEN_ENDOFTEXT, TOKEN_BOS, TOKEN_PAD, TOKEN_IM_START];

/// Why [`Agent::generate`] stopped decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted an end-of-turn (or other sequence-boundary) token.
    Eos,
    /// Hit the per-turn `max_gen` cap.
    MaxNew,
    /// The transcript reached `max_context`.
    ContextFull,
}

/// Timing + counts from a generation run.
pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill: Duration,
    pub decode: Duration,
}

impl GenStats {
    /// Prefill throughput (prompt tokens per second).
    pub fn prefill_tps(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill.as_secs_f64().max(f64::MIN_POSITIVE)
    }

    /// Decode throughput (generated tokens per second).
    pub fn decode_tps(&self) -> f64 {
        self.generated_tokens as f64 / self.decode.as_secs_f64().max(f64::MIN_POSITIVE)
    }
}

/// One generated reply: the new token ids, their decoded text, timing, and why decoding stopped.
pub struct Turn {
    pub ids: Vec<u32>,
    pub text: String,
    pub stats: GenStats,
    pub stop: StopReason,
}

/// A live conversation bound to a borrowed [`Model`]. Owns the transcript, the decode-time
/// caches, the sampler, and the per-turn limits; the heavy weights stay in the shared model,
/// so one loaded model can back several independent agents.
pub struct Agent<'m> {
    model: &'m Model,
    tok: Tokenizer,
    cache: Cache,
    sampler: Sampler,
    /// The full token transcript (every turn so far). `cache.pos` of these have already been
    /// run through the caches; the remainder is prefilled on the next [`generate`](Self::generate).
    history: Vec<u32>,
    max_gen: usize,
    max_context: usize,
    /// Cap on `<think>…</think>` reasoning tokens before `</think>` is forced; `usize::MAX`
    /// leaves reasoning unbounded.
    max_think: usize,
}

impl<'m> Agent<'m> {
    /// Create an agent over `model`, building its tokenizer from the same GGUF. Starts with
    /// Liquid's recommended sampling and a 32K context cap; override via the builder methods.
    pub fn new(model: &'m Model) -> Result<Self, Box<dyn Error>> {
        let tok = Tokenizer::from_gguf(model.gguf())?;
        Ok(Agent {
            model,
            tok,
            cache: Cache::new(),
            sampler: Sampler::recommended(),
            history: Vec::new(),
            max_gen: DEFAULT_MAX_GEN,
            max_context: DEFAULT_MAX_CONTEXT,
            max_think: usize::MAX,
        })
    }

    // --- Builder-style configuration ---

    /// Switch to deterministic greedy decoding (argmax; no temperature, top-k, or penalty).
    pub fn greedy(mut self) -> Self {
        self.sampler = Sampler::greedy();
        self
    }

    /// Set the sampling temperature (`0.0` ⇒ greedy argmax).
    pub fn temperature(mut self, t: f32) -> Self {
        self.sampler.temperature = t;
        self
    }

    /// Keep only the `k` highest-logit candidates (`0` ⇒ no limit).
    pub fn top_k(mut self, k: usize) -> Self {
        self.sampler.top_k = k;
        self
    }

    /// Divide already-seen tokens' logits by this (`1.0` ⇒ disabled).
    pub fn repeat_penalty(mut self, p: f32) -> Self {
        self.sampler.repeat_penalty = p;
        self
    }

    /// Cap the number of tokens generated per turn.
    pub fn max_gen(mut self, n: usize) -> Self {
        self.max_gen = n;
        self
    }

    /// Cap the total transcript length (tokens) the agent will decode up to.
    pub fn max_context(mut self, n: usize) -> Self {
        self.max_context = n;
        self
    }

    /// Limit the `<think>…</think>` reasoning block to `n` tokens: once `n` reasoning tokens
    /// have been produced, `</think>` is forced so the model proceeds to its answer.
    pub fn max_think(mut self, n: usize) -> Self {
        self.max_think = n;
        self
    }

    // --- Building the prompt (these only grow the transcript) ---

    /// Tokenize raw `text` and append it to the transcript. The BOS token is prepended only on
    /// the first append (while the transcript is still empty).
    pub fn append(&mut self, text: &str) {
        let add_bos = self.history.is_empty();
        let ids = self.tok.encode(text, add_bos);
        self.history.extend(ids);
    }

    /// Append a full ChatML user turn: `<|im_start|>user\n{msg}<|im_end|>\n`.
    pub fn append_user(&mut self, msg: &str) {
        self.append(&format!("<|im_start|>user\n{msg}<|im_end|>\n"));
    }

    /// Append already-tokenized ids verbatim (e.g. replaying a transcript or injecting a
    /// tool result).
    pub fn append_tokens(&mut self, ids: &[u32]) {
        self.history.extend_from_slice(ids);
    }

    // --- Generation ---

    /// Open an assistant turn, generate its reply, and close the turn so the transcript stays
    /// well-formed for the next message. `on_token` receives each visible token's id and text
    /// as it is decoded.
    pub fn assistant_turn(&mut self, on_token: impl FnMut(u32, &str)) -> Turn {
        self.append(ASSISTANT_OPEN);
        let turn = self.generate(on_token);
        // The reply excludes the closing <|im_end|> (generate stops at it), so close the turn
        // explicitly and add the trailing newline the template puts between turns.
        self.history.push(TOKEN_IM_END);
        self.append("\n");
        turn
    }

    /// Prefill any appended-but-unprocessed tokens, then decode a continuation until the model
    /// emits EOS, hits `max_gen`, or fills the context. Visible tokens are appended to the
    /// transcript and streamed to `on_token`; the terminating EOS is not.
    pub fn generate(&mut self, mut on_token: impl FnMut(u32, &str)) -> Turn {
        // Prefill: run every pending token except the last through the caches; the last token's
        // logits seed the decode loop. Invariant: cache.pos == number of absorbed history tokens.
        let t_prefill = Instant::now();
        let pending = self.history.len() - self.cache.pos;
        let (&last, rest) = self.history[self.cache.pos..]
            .split_last()
            .expect("generate: no pending tokens to generate from");
        for &tok in rest {
            self.model.run_layers(tok, &mut self.cache);
        }
        let mut logits = self.model.forward_step(last, &mut self.cache);
        let prefill = t_prefill.elapsed();

        // Decode one token at a time, feeding each back through the caches.
        let t_decode = Instant::now();
        let mut ids = Vec::new();
        // Track the <think>…</think> reasoning block so it can be capped at `max_think`.
        let mut thinking = false;
        let mut think_count = 0usize;
        let stop = loop {
            // Once the reasoning budget is spent, force `</think>` instead of sampling; the model
            // then continues from there into its answer.
            let next = if thinking && think_count >= self.max_think {
                TOKEN_THINK_END
            } else {
                self.sampler.sample(&mut logits, &self.history)
            };
            if next == self.tok.eos || STOP_TOKENS.contains(&next) {
                break StopReason::Eos;
            }
            match next {
                TOKEN_THINK => {
                    thinking = true;
                    think_count = 0;
                }
                TOKEN_THINK_END => thinking = false,
                _ if thinking => think_count += 1,
                _ => {}
            }
            let text = self.tok.decode(&[next]);
            on_token(next, &text);
            ids.push(next);
            self.history.push(next);
            if ids.len() >= self.max_gen {
                break StopReason::MaxNew;
            }
            if self.history.len() >= self.max_context {
                break StopReason::ContextFull;
            }
            logits = self.model.forward_step(next, &mut self.cache);
        };
        let decode = t_decode.elapsed();

        let text = self.tok.decode(&ids);
        let stats = GenStats {
            prompt_tokens: pending,
            generated_tokens: ids.len(),
            prefill,
            decode,
        };
        Turn { ids, text, stats, stop }
    }

    /// Clear the conversation (transcript + caches), keeping the loaded weights and config.
    pub fn clear(&mut self) {
        self.history.clear();
        self.cache = Cache::new();
    }

    /// The full token transcript so far.
    pub fn history(&self) -> &[u32] {
        &self.history
    }
}
