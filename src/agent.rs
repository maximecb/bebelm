//! A conversational session over a loaded [`Model`]: holds the running token transcript plus
//! the live KV / conv-state caches, so each turn only prefills the newly appended tokens
//! instead of replaying the whole conversation from scratch.
//!
//! Build up the prompt with the `append*` methods (which only grow the transcript), then run
//! the model with [`Agent::generate`] — or the [`Agent::assistant_turn`] convenience, which
//! wraps the ChatML assistant framing around a single `generate`.

use std::time::{Duration, Instant};

use crate::cache::Cache;
use crate::model::Model;
use crate::sampler::Sampler;
use crate::tokenizer::{
    TOKEN_BOS, TOKEN_ENDOFTEXT, TOKEN_IM_END, TOKEN_IM_START, TOKEN_PAD, TOKEN_THINK,
    TOKEN_THINK_END, TOKEN_TOOL_CALL_END, TOKEN_TOOL_CALL_START,
};
use crate::tool::{parse_tool_calls, Tool, ToolCall};

/// Default per-turn generation cap. A reasoning (`<think>`) turn can run long, so this is
/// generous; it only bounds a runaway turn.
const DEFAULT_MAX_GEN: usize = 2048;

/// Default KV attention-window cap (tokens); once exceeded, the oldest context slides out so
/// decoding can continue. A conservative session default; the model supports far more.
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
    /// The model closed a tool call (`<|tool_call_end|>`); only reachable when tools are
    /// registered. The driver in [`Agent::assistant_turn_with_tools`] acts on the call and
    /// continues, so callers of [`Agent::generate`] normally won't see this.
    ToolCall,
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
///
/// `Agent` is [`Clone`]: cloning copies the transcript and caches, so a prefilled prompt (e.g.
/// a system prompt plus a few example turns) can be built once and cheaply forked into several
/// independent continuations without re-running the prefill for the shared prefix.
#[derive(Clone)]
pub struct Agent<'m> {
    model: &'m Model,
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
    /// Tools the model may call. Empty (the default) leaves generation tool-free; registering
    /// any tool makes [`generate`](Self::generate) also stop at `<|tool_call_end|>` and enables
    /// the [`assistant_turn_with_tools`](Self::assistant_turn_with_tools) driver.
    tools: Vec<Tool>,
}

impl<'m> Agent<'m> {
    /// Create an agent over `model`. Starts with Liquid's recommended sampling and a 32K
    /// context cap; override via the builder methods.
    pub fn new(model: &'m Model) -> Self {
        Agent {
            model,
            cache: Cache::new(),
            sampler: Sampler::recommended(),
            history: Vec::new(),
            max_gen: DEFAULT_MAX_GEN,
            max_context: DEFAULT_MAX_CONTEXT,
            max_think: usize::MAX,
            tools: Vec::new(),
        }
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

    /// Cap the KV attention window (in tokens). When decoding would exceed it, the oldest
    /// context is dropped (a sliding window) instead of stopping.
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

    /// Register a tool the model may call. Add all tools before [`append_system`](Self::append_system),
    /// which advertises them in the system block. With tools registered, drive the conversation
    /// with [`assistant_turn_with_tools`](Self::assistant_turn_with_tools).
    pub fn add_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    // --- Building the prompt (these only grow the transcript) ---

    /// Tokenize raw `text` and append it to the transcript. The BOS token is prepended only on
    /// the first append (while the transcript is still empty).
    pub fn append(&mut self, text: &str) {
        let add_bos = self.history.is_empty();
        let ids = self.model.tokenizer().encode(text, add_bos);
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

    /// Append the system block: `<|im_start|>system\n{text}` plus, if any tools are registered,
    /// the `\nList of tools: [{…}, …]` preamble that advertises them — then `<|im_end|>\n`.
    /// Call this first (the system block follows BOS, before any user turn) and after all
    /// [`add_tool`](Self::add_tool) calls.
    pub fn append_system(&mut self, text: &str) {
        let mut block = format!("<|im_start|>system\n{text}");
        if !self.tools.is_empty() {
            block.push_str("\nList of tools: [");
            for (i, tool) in self.tools.iter().enumerate() {
                if i > 0 {
                    block.push_str(", ");
                }
                block.push_str(tool.schema());
            }
            block.push(']');
        }
        block.push_str("<|im_end|>\n");
        self.append(&block);
    }

    // --- Generation ---

    /// Open an assistant turn, generate its reply, and close the turn so the transcript stays
    /// well-formed for the next message. `on_token` receives each visible token's id and text
    /// as it is decoded.
    pub fn assistant_turn(&mut self, on_token: impl FnMut(u32, &str)) -> Turn {
        self.append(ASSISTANT_OPEN);
        let turn = self.generate(on_token);
        self.close_assistant_turn();
        turn
    }

    /// Close the current assistant turn in the transcript. The reply itself excludes the closing
    /// `<|im_end|>` (generation stops at it, or at a tool call), so append it explicitly plus the
    /// trailing newline the template puts between turns.
    fn close_assistant_turn(&mut self) {
        self.history.push(TOKEN_IM_END);
        self.append("\n");
    }

    /// Run the agentic tool loop. Each round opens an assistant turn and generates; when the reply
    /// is a tool call, every call is dispatched to its registered [`Tool`], the joined results are
    /// appended as a `tool`-role message, and the loop continues. It ends when the model replies
    /// with no tool call — or after `max_rounds` assistant turns (so at most `max_rounds - 1`
    /// rounds of tool results, the final turn being reserved to use them). `on_token` streams
    /// every generated token (tool calls included); `on_tool` observes each dispatched call and
    /// its result. Requires at least one registered [`add_tool`](Self::add_tool); with none, this
    /// behaves like a single [`assistant_turn`](Self::assistant_turn).
    ///
    /// The returned [`Turn`] aggregates all rounds: `ids`/`text` concatenate everything generated,
    /// `stats` sum across rounds, and `stop` is the final round's reason (`Eos` for a normal
    /// answer, or `ToolCall` if the round budget was hit while still calling tools).
    pub fn assistant_turn_with_tools(
        &mut self,
        max_rounds: usize,
        mut on_token: impl FnMut(u32, &str),
        mut on_tool: impl FnMut(&ToolCall, &str),
    ) -> Turn {
        let mut ids = Vec::new();
        let mut text = String::new();
        let mut prompt_tokens = 0;
        let mut generated_tokens = 0;
        let mut prefill = Duration::ZERO;
        let mut decode = Duration::ZERO;
        let mut stop = StopReason::Eos;

        for round in 0..max_rounds {
            self.append(ASSISTANT_OPEN);
            let turn = self.generate(&mut on_token);
            ids.extend_from_slice(&turn.ids);
            text.push_str(&turn.text);
            prompt_tokens += turn.stats.prompt_tokens;
            generated_tokens += turn.stats.generated_tokens;
            prefill += turn.stats.prefill;
            decode += turn.stats.decode;
            stop = turn.stop;
            self.close_assistant_turn();

            // No tool call ⇒ this is the model's final, plain-text answer.
            let calls = self.tool_calls_in(&turn.ids);
            if calls.is_empty() {
                break;
            }
            // The last allowed turn is reserved for the model to *use* the previous results; if it
            // still wants tools here, stop rather than dispatch a round we can't feed back.
            if round + 1 == max_rounds {
                break;
            }

            // Dispatch each call and feed the joined results back as a single `tool`-role message.
            let mut results = Vec::with_capacity(calls.len());
            for call in &calls {
                let result = self.dispatch(call);
                on_tool(call, &result);
                results.push(result);
            }
            self.append(&format!("<|im_start|>tool\n{}<|im_end|>\n", results.join("\n")));
        }

        Turn {
            ids,
            text,
            stats: GenStats { prompt_tokens, generated_tokens, prefill, decode },
            stop,
        }
    }

    /// Parse the tool calls from one turn's generated `ids`: the span between `<|tool_call_start|>`
    /// and `<|tool_call_end|>`, decoded to text and handed to [`parse_tool_calls`]. Empty if the
    /// turn carried no (complete) tool call.
    fn tool_calls_in(&self, ids: &[u32]) -> Vec<ToolCall> {
        let Some(start) = ids.iter().position(|&t| t == TOKEN_TOOL_CALL_START) else {
            return Vec::new();
        };
        let Some(len) = ids[start + 1..].iter().position(|&t| t == TOKEN_TOOL_CALL_END) else {
            return Vec::new();
        };
        let inner = self.model.tokenizer().decode(&ids[start + 1..start + 1 + len]);
        parse_tool_calls(&inner)
    }

    /// Run the registered tool matching `call.name`, or report the unknown name back to the model.
    fn dispatch(&self, call: &ToolCall) -> String {
        match self.tools.iter().find(|t| t.name() == call.name) {
            Some(tool) => tool.invoke(call),
            None => format!("Error: no tool named {:?}", call.name),
        }
    }

    /// Prefill any appended-but-unprocessed tokens, then decode a continuation until the model
    /// emits EOS or hits `max_gen`. Past `max_context` the KV window slides (oldest context
    /// dropped) rather than stopping. Visible tokens are appended to the transcript and streamed
    /// to `on_token`; the terminating EOS is not.
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
            Self::trim_context(&mut self.cache, self.max_context);
        }
        let mut logits = self.model.forward_step(last, &mut self.cache);
        Self::trim_context(&mut self.cache, self.max_context);
        let prefill = t_prefill.elapsed();

        // Decode one token at a time, feeding each back through the caches.
        let t_decode = Instant::now();
        let mut ids = Vec::new();
        // Track the <think>…</think> reasoning block so it can be capped at `max_think`.
        let mut thinking = false;
        let mut think_count = 0usize;
        // While set, the `<think>` token is barred so the model can't (re)open a reasoning block.
        // `--no-think` (max_think 0) bars it from the start, so the model answers directly with no
        // reasoning block at all; a positive budget bars it only after the block is force-closed,
        // since this model has no native no-think mode and otherwise reopens `<think>` and spirals.
        let mut think_capped = self.max_think == 0;
        // Set for the one token right after a `</think>`: it may not be a turn-ender, so the model
        // can't close the turn with an empty answer (common right after a forced `</think>`).
        let mut require_answer = false;
        // With tools registered, end the turn at `<|tool_call_end|>` so the driver can act on the
        // call without decoding the rest of the turn (the closing token is kept in the transcript).
        let tools_active = !self.tools.is_empty();
        let stop = loop {
            if think_capped {
                logits[TOKEN_THINK as usize] = f32::NEG_INFINITY;
            }
            if require_answer {
                logits[self.model.tokenizer().eos as usize] = f32::NEG_INFINITY;
                for &t in &STOP_TOKENS {
                    logits[t as usize] = f32::NEG_INFINITY;
                }
                require_answer = false;
            }
            // Once the reasoning budget is spent, force `</think>` instead of sampling; the model
            // then continues from there into its answer.
            let next = if thinking && think_count >= self.max_think {
                think_capped = true;
                TOKEN_THINK_END
            } else {
                self.sampler.sample(&mut logits, &self.history)
            };
            if next == self.model.tokenizer().eos || STOP_TOKENS.contains(&next) {
                break StopReason::Eos;
            }
            match next {
                TOKEN_THINK => {
                    thinking = true;
                    think_count = 0;
                }
                TOKEN_THINK_END => {
                    thinking = false;
                    require_answer = true;
                }
                _ if thinking => think_count += 1,
                _ => {}
            }
            let text = self.model.tokenizer().decode(&[next]);
            on_token(next, &text);
            ids.push(next);
            self.history.push(next);
            if tools_active && next == TOKEN_TOOL_CALL_END {
                break StopReason::ToolCall;
            }
            if ids.len() >= self.max_gen {
                break StopReason::MaxNew;
            }
            logits = self.model.forward_step(next, &mut self.cache);
            // Slide the KV window instead of stopping once the context cap is reached.
            Self::trim_context(&mut self.cache, self.max_context);
        };
        let decode = t_decode.elapsed();

        let text = self.model.tokenizer().decode(&ids);
        let stats = GenStats {
            prompt_tokens: pending,
            generated_tokens: ids.len(),
            prefill,
            decode,
        };
        Turn { ids, text, stats, stop }
    }

    /// Slide the KV attention window down to `max_context` positions once it grows past the cap,
    /// dropping the oldest context so decoding continues. `cache.pos` (the absolute RoPE
    /// position) is left untouched — RoPE attention depends only on the query↔key offset, which
    /// stays within the window, so the retained keys remain correctly positioned.
    fn trim_context(cache: &mut Cache, max_context: usize) {
        let len = cache.kv_len();
        if len > max_context {
            cache.evict_front(len - max_context);
        }
    }

    /// Clear the conversation (transcript + caches), keeping the loaded weights and config.
    pub fn clear(&mut self) {
        self.history.clear();
        self.cache = Cache::new();
        self.sampler.reset();
    }

    /// The full token transcript so far.
    pub fn history(&self) -> &[u32] {
        &self.history
    }
}
