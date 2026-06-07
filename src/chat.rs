//! The interactive `chat` subcommand: a terminal REPL over a single long-lived [`Agent`],
//! streaming each reply with the `<think>…</think>` reasoning shown in a different colour from
//! the answer.

use std::io::{self, IsTerminal, Write};

use bebelm::agent::Agent;
use bebelm::model::Model;
use bebelm::tokenizer;

use crate::{parse_agent_options, Cmd};

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
/// suppressed. Sampling uses the model's recommended defaults unless overridden by flags
/// (`--greedy`, `--max-gen`, `--max-think`, `--no-think`); there is no system prompt.
pub(crate) fn cmd_chat(path: &str, args: &[String]) -> Cmd {
    let (opts, positional) = parse_agent_options(args)?;
    if !positional.is_empty() {
        return Err(format!("chat takes no prompt arguments; got {positional:?}").into());
    }
    let model = Model::load(path)?;
    let mut agent = opts.apply(Agent::new(&model));
    let pal = Palette::detect();

    eprintln!("Chat ready. Type a message. Input Ctrl-D or /exit to quit, /clear to reset history.\n");

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
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/exit" || input == "/quit" {
            break;
        }
        if input == "/clear" {
            agent.clear();
            eprintln!("{}Chat history cleared.{}", pal.dim, pal.reset);
            continue;
        }

        // Expand any `@path` file references into inline fenced blocks before sending.
        let msg = match expand_file_refs(input) {
            Ok(msg) => msg,
            Err(err) => {
                eprintln!("{}{err}{}", pal.dim, pal.reset);
                continue;
            }
        };

        agent.append_user(&msg);

        // Blank line between the user's message and the model's reply.
        println!();

        // Stream the reply. The model opens with a <think>...</think> reasoning block; colour
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

/// Expand `@path` file references in a typed line. The line is left untouched and the
/// contents of each referenced file are appended after it as fenced blocks, with the
/// path as the info string. Each whitespace-delimited word starting with `@` names a
/// file; duplicates are included once. Returns an error string if a referenced file
/// can't be read (which also rejects directories) or isn't valid UTF-8; a bare `@` or a
/// line with no references is returned unchanged.
fn expand_file_refs(line: &str) -> Result<String, String> {
    let mut blocks = String::new();
    let mut seen: Vec<&str> = Vec::new();

    for word in line.split_whitespace() {
        let Some(path) = word.strip_prefix('@') else { continue };
        if path.is_empty() || seen.contains(&path) {
            continue;
        }
        seen.push(path);
        let bytes = std::fs::read(path).map_err(|e| format!("@{path}: {e}"))?;
        let text = String::from_utf8(bytes).map_err(|_| format!("@{path}: not valid UTF-8"))?;
        blocks.push_str("\n```");
        blocks.push_str(path);
        blocks.push('\n');
        blocks.push_str(&text);
        if !text.ends_with('\n') {
            blocks.push('\n');
        }
        blocks.push_str("```\n");
    }

    if blocks.is_empty() {
        return Ok(line.to_string());
    }
    Ok(format!("{line}\n{blocks}"))
}
