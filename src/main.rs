//! bebelm — CPU-only, pure-Rust inference for Liquid AI LFM2.5-8B-A1B (Q4_K_M).
//!
//! CLI over the `bebelm` library: run the forward pass, greedily generate / complete, or
//! hold an interactive `chat`. The weights file is taken from `$BEBELM_WEIGHTS_FILE`, not the
//! command line.

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

use bebelm::agent::Agent;
use bebelm::config;
use bebelm::model::Model;
use bebelm::tokenizer;

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
    let max_gen: usize = max_str
        .parse()
        .map_err(|_| format!("invalid max-new-tokens {max_str:?}"))?;
    let prompt = parse_tokens(token_args)?;
    if prompt.is_empty() {
        return Err("need at least one prompt token id".into());
    }

    let model = Model::load(path)?;
    eprintln!("greedy-generating up to {max_gen} token(s) (cached, multi-threaded)...");
    let mut agent = Agent::new(&model)?.greedy().max_gen(max_gen);
    agent.append_tokens(&prompt);
    let turn = agent.generate(|_, _| {});

    println!("prompt    : {prompt:?}");
    println!("generated : {:?}", turn.ids);
    Ok(())
}

/// Encode text, greedy-generate a continuation, and decode it back to text.
fn cmd_complete(path: &str, max_str: &str, text_args: &[String]) -> Cmd {
    let max_gen: usize = max_str
        .parse()
        .map_err(|_| format!("invalid max-new-tokens {max_str:?}"))?;
    let text = text_args.join(" ");
    if text.is_empty() {
        return Err("need a prompt".into());
    }

    let model = Model::load(path)?;
    // Greedy, deterministic decoding so the continuation is reproducible.
    let mut agent = Agent::new(&model)?.greedy().max_gen(max_gen);
    agent.append(&text);
    let prompt = agent.history().to_vec();
    eprintln!(
        "prompt = {} tokens; greedy-generating up to {max_gen} (cached, multi-threaded)...",
        prompt.len()
    );
    println!("prompt       : {text:?}");
    print!("continuation : ");
    std::io::stdout().flush().ok();

    // Stream each token's text to stdout as it is generated.
    let turn = agent.generate(|_id, piece| {
        print!("{piece}");
        std::io::stdout().flush().ok();
    });
    println!(); // end the streamed line

    println!("prompt ids   : {prompt:?}");
    println!("gen ids      : {:?}", turn.ids);
    println!(
        "prefill      : {} tok in {:.0} ms ({:.1} tok/s)",
        turn.stats.prompt_tokens,
        turn.stats.prefill.as_secs_f64() * 1e3,
        turn.stats.prefill_tps()
    );
    println!(
        "decode       : {} tok in {:.0} ms ({:.2} tok/s)",
        turn.stats.generated_tokens,
        turn.stats.decode.as_secs_f64() * 1e3,
        turn.stats.decode_tps()
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
/// The model's reply is shown in two colors: the `<think>…</think>` reasoning (`think`) and
/// the actual answer that follows (`answer`).
struct Palette {
    user: &'static str,
    think: &'static str,
    answer: &'static str,
    dim: &'static str,
    reset: &'static str,
}

impl Palette {
    fn detect() -> Self {
        if io::stdout().is_terminal() {
            Palette {
                user: "\x1b[1;32m",  // bold green
                think: "\x1b[36m",   // cyan
                answer: "\x1b[31m",  // red
                dim: "\x1b[2m",
                reset: "\x1b[0m",
            }
        } else {
            Palette { user: "", think: "", answer: "", dim: "", reset: "" }
        }
    }
}

/// Interactive multi-turn chat. Keeps one [`Agent`] alive across turns, so each message only
/// prefills its own newly appended tokens — the KV / conv caches persist instead of being
/// rebuilt over the whole conversation. The reply is streamed, with the `<think>…</think>`
/// reasoning shown in a different colour from the answer; the terminating `<|im_end|>` is
/// suppressed. Sampling uses the model's recommended defaults; there is no system prompt.
fn cmd_chat(path: &str, args: &[String]) -> Cmd {
    let model = Model::load(path)?;
    let mut agent = Agent::new(&model)?;
    if let Some(s) = args.first() {
        let max_gen = s.parse().map_err(|_| format!("invalid max-new {s:?}"))?;
        agent = agent.max_gen(max_gen);
    }
    let pal = Palette::detect();

    eprintln!("Chat ready. Type a message. Input Ctrl-D or /exit to quit.\n");

    let stdin = io::stdin();
    let mut line = String::new();

    loop {
        print!("{}User>{} ", pal.user, pal.reset);
        io::stdout().flush().ok();
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            println!(); // Terminate the prompt line on Ctrl-D.
            break;
        }
        let msg = line.trim();
        if msg.is_empty() {
            continue;
        }
        if msg == "/exit" || msg == "/quit" {
            break;
        }

        agent.append_user(msg);

        // Blank line between the user's message and the model's reply.
        println!();

        // Stream the reply. The model opens with a <think>…</think> reasoning block; colour
        // that distinctly from the answer that follows. Start in the answer colour so a reply
        // with no <think> block still reads correctly.
        print!("{}", pal.answer);
        io::stdout().flush().ok();
        let turn = agent.assistant_turn(|id, text| {
            if id == tokenizer::TOKEN_THINK {
                print!("{}", pal.think);
            }
            print!("{text}");
            if id == tokenizer::TOKEN_THINK_END {
                println!(); // Blank line after the reasoning block.
                print!("{}", pal.answer);
            }
            io::stdout().flush().ok();
        });
        println!("{}", pal.reset);
        println!(
            "{}({} tok, {:.1} tok/s){}",
            pal.dim,
            turn.stats.generated_tokens,
            turn.stats.decode_tps(),
            pal.reset
        );
        println!(); // Blank line after the turn, before the next prompt.
    }

    Ok(())
}
