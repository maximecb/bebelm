//! bebelm — CPU-only, pure-Rust inference for Liquid AI LFM2.5-8B-A1B (Q4_K_M).
//!
//! CLI over the `bebelm` library: tokenize text, run the forward pass, greedily
//! generate / complete, or hold an interactive `chat`. The weights file is taken from
//! `$BEBELM_WEIGHTS_FILE`, not the command line.

use std::error::Error;
use std::io::{self, IsTerminal, Write};
use std::process::ExitCode;

/// Default weights path when `$BEBELM_WEIGHTS_FILE` is unset (relative to the cwd).
const DEFAULT_WEIGHTS_FILE: &str = "./LFM2.5-8B-A1B-Q4_K_M.gguf";

/// Resolve the GGUF weights path from `$BEBELM_WEIGHTS_FILE`, defaulting to the file in
/// the current working directory.
fn weights_path() -> String {
    std::env::var("BEBELM_WEIGHTS_FILE").unwrap_or_else(|_| DEFAULT_WEIGHTS_FILE.to_string())
}

use bebelm::config;
use bebelm::gguf::GgufFile;
use bebelm::model::Model;
use bebelm::sampler::Sampler;
use bebelm::tokenizer::{self, Tokenizer};

type Cmd = Result<(), Box<dyn Error>>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let path = weights_path();
    let result: Cmd = match args.get(1).map(String::as_str) {
        Some("logits") => cmd_logits(&path, &args[2..]),
        Some("generate") => match args.get(2) {
            Some(max) => cmd_generate(&path, max, &args[3..]),
            None => return usage("generate <max-new-tokens> <token-id>..."),
        },
        Some("tokenize") => cmd_tokenize(&path, &args[2..]),
        Some("complete") => match args.get(2) {
            Some(max) => cmd_complete(&path, max, &args[3..]),
            None => return usage("complete <max-new-tokens> <text>..."),
        },
        Some("chat") => cmd_chat(&path, &args[2..]),
        _ => {
            eprintln!("bebelm — CPU-only LFM2.5-8B-A1B inference\n");
            eprintln!("usage:");
            eprintln!("  bebelm logits   <token-id>...        forward pass, print next-token logits");
            eprintln!("  bebelm generate <max-new> <token>... greedy-generate token ids");
            eprintln!("  bebelm tokenize <text>...            encode/decode round-trip");
            eprintln!("  bebelm complete <max-new> <text>...  greedy text completion");
            eprintln!("  bebelm chat     [max-new]            interactive chat (streams thinking + reply)");
            eprintln!("\nweights file: $BEBELM_WEIGHTS_FILE (default {DEFAULT_WEIGHTS_FILE})");
            return ExitCode::FAILURE;
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn usage(line: &str) -> ExitCode {
    eprintln!("usage: bebelm {line}");
    ExitCode::FAILURE
}

/// Parse a list of decimal token-id strings, bounds-checking against the vocab.
fn parse_tokens(args: &[String]) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut tokens = Vec::with_capacity(args.len());
    for s in args {
        let id: u32 = s
            .parse()
            .map_err(|_| format!("invalid token id {s:?} (must be a non-negative integer)"))?;
        if id as usize >= config::VOCAB {
            return Err(format!("token id {id} out of range (vocab = {})", config::VOCAB).into());
        }
        tokens.push(id);
    }
    Ok(tokens)
}

/// Greedy-generate token ids from a prompt of token ids (text I/O arrives with the tokenizer).
fn cmd_generate(path: &str, max_str: &str, token_args: &[String]) -> Cmd {
    let max_new: usize = max_str
        .parse()
        .map_err(|_| format!("invalid max-new-tokens {max_str:?}"))?;
    let prompt = parse_tokens(token_args)?;
    if prompt.is_empty() {
        return Err("need at least one prompt token id".into());
    }

    let model = Model::load(path)?;
    eprintln!("greedy-generating up to {max_new} token(s) (cached, multi-threaded)...");
    let mut sampler = Sampler::greedy();
    let generated = model.generate(&prompt, &mut sampler, max_new, tokenizer::TOKEN_EOS);

    println!("prompt    : {prompt:?}");
    println!("generated : {generated:?}");
    Ok(())
}

/// Encode text to token ids and decode back — a round-trip check on the real vocab.
fn cmd_tokenize(path: &str, text_args: &[String]) -> Cmd {
    let text = text_args.join(" ");
    let g = GgufFile::open(path)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let ids = tok.encode(&text, false);
    let decoded = tok.decode(&ids);
    println!("text       : {text:?}");
    println!("ids        : {ids:?}");
    println!("decoded    : {decoded:?}");
    println!("round-trip : {}", if decoded == text { "OK" } else { "MISMATCH" });
    Ok(())
}

/// Encode text, greedy-generate a continuation, and decode it back to text.
fn cmd_complete(path: &str, max_str: &str, text_args: &[String]) -> Cmd {
    let max_new: usize = max_str
        .parse()
        .map_err(|_| format!("invalid max-new-tokens {max_str:?}"))?;
    let text = text_args.join(" ");
    if text.is_empty() {
        return Err("need a prompt".into());
    }

    let model = Model::load(path)?;
    let tok = Tokenizer::from_gguf(model.gguf())?;
    let prompt = tok.encode(&text, true);
    eprintln!(
        "prompt = {} tokens; greedy-generating up to {max_new} (cached, multi-threaded)...",
        prompt.len()
    );
    println!("prompt       : {text:?}");
    print!("continuation : ");
    std::io::stdout().flush().ok();

    // Stream each token's text to stdout as it is generated.
    let mut sampler = Sampler::greedy();
    let (generated, stats) = model.generate_with_stats(&prompt, &mut sampler, max_new, tok.eos, |t| {
        print!("{}", tok.decode(&[t]));
        std::io::stdout().flush().ok();
    });
    println!(); // end the streamed line

    println!("prompt ids   : {prompt:?}");
    println!("gen ids      : {generated:?}");
    println!(
        "prefill      : {} tok in {:.0} ms ({:.1} tok/s)",
        stats.prompt_tokens,
        stats.prefill.as_secs_f64() * 1e3,
        stats.prefill_tps()
    );
    println!(
        "decode       : {} tok in {:.0} ms ({:.2} tok/s)",
        stats.generated_tokens,
        stats.decode.as_secs_f64() * 1e3,
        stats.decode_tps()
    );
    Ok(())
}

/// Run the forward pass on raw token ids and print the next-token logit summary.
fn cmd_logits(path: &str, token_args: &[String]) -> Cmd {
    let tokens = parse_tokens(token_args)?;
    if tokens.is_empty() {
        return Err("need at least one token id".into());
    }

    let model = Model::load(path)?;
    eprintln!(
        "running forward pass on {} token(s) (uncached, multi-threaded; may take a while)...",
        tokens.len()
    );
    let logits = model.forward(&tokens);

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut nonfinite = 0usize;
    for &v in &logits {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
            sum += v as f64;
        } else {
            nonfinite += 1;
        }
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

    println!("tokens    : {tokens:?}");
    println!("logits    : {} (vocab)", logits.len());
    println!("min/max   : {min:.4} / {max:.4}");
    println!("mean      : {:.4}", sum / logits.len() as f64);
    if nonfinite > 0 {
        println!("non-finite: {nonfinite}  (BUG — expected 0)");
    }
    println!("argmax    : token {} (logit {:.4})", idx[0], logits[idx[0]]);
    println!("top-5:");
    for &i in idx.iter().take(5) {
        println!("  token {i:>6}  logit {:.4}", logits[i]);
    }
    Ok(())
}

/// ANSI colors for the chat UI, blanked when stdout is not a terminal (e.g. piped to a file).
struct Palette {
    user: &'static str,
    model: &'static str,
    dim: &'static str,
    reset: &'static str,
}

impl Palette {
    fn detect() -> Self {
        if io::stdout().is_terminal() {
            Palette { user: "\x1b[1;32m", model: "\x1b[36m", dim: "\x1b[2m", reset: "\x1b[0m" }
        } else {
            Palette { user: "", model: "", dim: "", reset: "" }
        }
    }
}

/// Default per-turn generation cap. A reasoning turn (the `<think>` block) can run long, so
/// this is generous; it only bounds a runaway turn. Override with `bebelm chat <max-new>`.
const CHAT_MAX_NEW: usize = 2048;

/// Interactive multi-turn chat. Builds a ChatML transcript of token ids and regenerates from
/// the whole conversation each turn (simple — no cross-turn KV-cache reuse yet). The model's
/// full output, including the `<think>…</think>` reasoning, is streamed in a distinct color;
/// only the terminating `<|im_end|>` is suppressed. Decoding is greedy and there is no system
/// prompt; prior assistant turns are kept verbatim (reasoning included) in the context.
fn cmd_chat(path: &str, args: &[String]) -> Cmd {
    let max_new: usize = match args.first() {
        Some(s) => s.parse().map_err(|_| format!("invalid max-new {s:?}"))?,
        None => CHAT_MAX_NEW,
    };

    let model = Model::load(path)?;
    let tok = Tokenizer::from_gguf(model.gguf())?;
    let pal = Palette::detect();
    let eos = tok.eos;

    eprintln!("Chat ready. Type a message. Input Ctrl-D or /exit to quit.\n");

    let mut convo: Vec<u32> = Vec::new();
    let mut sampler = Sampler::greedy();
    let stdin = io::stdin();
    let mut line = String::new();

    loop {
        print!("{}User>{} ", pal.user, pal.reset);
        io::stdout().flush().ok();
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            println!(); // terminate the prompt line on Ctrl-D
            break;
        }
        let msg = line.trim();
        if msg.is_empty() {
            continue;
        }
        if msg == "/exit" || msg == "/quit" {
            break;
        }

        // Append this user turn + the assistant opener. The model emits its own `<think>`
        // after the opener, so we don't force it. Each prior turn ends at the assistant's
        // `<|im_end|>`, so turns after the first start with the separating newline. BOS is
        // prepended only on the very first segment.
        let first = convo.is_empty();
        let lead = if first { "" } else { "\n" };
        let seg = format!("{lead}<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n");
        convo.extend(tok.encode(&seg, first));

        // Stream the reply (all but the terminating <|im_end|>) in the model color.
        print!("{}", pal.model);
        io::stdout().flush().ok();
        let (generated, stats) = model.generate_with_stats(&convo, &mut sampler, max_new, eos, |t| {
            if t != eos {
                print!("{}", tok.decode(&[t]));
                io::stdout().flush().ok();
            }
        });
        println!("{}", pal.reset);
        println!(
            "{}({} tok, {:.1} tok/s){}",
            pal.dim,
            stats.generated_tokens,
            stats.decode_tps(),
            pal.reset
        );

        convo.extend(generated);
    }

    Ok(())
}
