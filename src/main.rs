//! bebelm — CPU-only, pure-Rust inference for Liquid AI LFM2.5-8B-A1B (Q4_K_M).
//!
//! CLI over the `bebelm` library: inspect a GGUF (`dump`), sanity-check the dequant
//! kernels on a tensor (`dequant`), or load + validate the whole model (`load`).

use std::error::Error;
use std::io::Write;
use std::process::ExitCode;

use bebelm::config;
use bebelm::gguf::{GgufFile, MetaValue};
use bebelm::kernels::dequant;
use bebelm::model::Model;
use bebelm::sampler::Sampler;
use bebelm::tokenizer::{self, Tokenizer};

type Cmd = Result<(), Box<dyn Error>>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let result: Cmd = match args.get(1).map(String::as_str) {
        Some("dump") => match args.get(2) {
            Some(path) => cmd_dump(path),
            None => return usage("dump <model.gguf>"),
        },
        Some("dequant") => match (args.get(2), args.get(3)) {
            (Some(path), Some(name)) => cmd_dequant(path, name),
            _ => return usage("dequant <model.gguf> <tensor-name>"),
        },
        Some("load") => match args.get(2) {
            Some(path) => cmd_load(path),
            None => return usage("load <model.gguf>"),
        },
        Some("logits") => match args.get(2) {
            Some(path) => cmd_logits(path, &args[3..]),
            None => return usage("logits <model.gguf> <token-id>..."),
        },
        Some("generate") => match (args.get(2), args.get(3)) {
            (Some(path), Some(max)) => cmd_generate(path, max, &args[4..]),
            _ => return usage("generate <model.gguf> <max-new-tokens> <token-id>..."),
        },
        Some("tokenize") => match args.get(2) {
            Some(path) => cmd_tokenize(path, &args[3..]),
            None => return usage("tokenize <model.gguf> <text>..."),
        },
        Some("complete") => match (args.get(2), args.get(3)) {
            (Some(path), Some(max)) => cmd_complete(path, max, &args[4..]),
            _ => return usage("complete <model.gguf> <max-new-tokens> <text>..."),
        },
        _ => {
            eprintln!("bebelm — CPU-only LFM2.5-8B-A1B inference\n");
            eprintln!("usage:");
            eprintln!("  bebelm dump     <model.gguf>                       list metadata and tensors");
            eprintln!("  bebelm dequant  <model.gguf> <tensor-name>         dequantize a tensor, print stats");
            eprintln!("  bebelm load     <model.gguf>                       load + validate against the config");
            eprintln!("  bebelm logits   <model.gguf> <token-id>...         forward pass, print next-token logits");
            eprintln!("  bebelm generate <model.gguf> <max-new> <token>...  greedy-generate token ids");
            eprintln!("  bebelm tokenize <model.gguf> <text>...             encode/decode round-trip");
            eprintln!("  bebelm complete <model.gguf> <max-new> <text>...   greedy text completion");
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

/// Load + validate the model, then print the resolved layer schedule.
fn cmd_load(path: &str) -> Cmd {
    Model::load(path)?;
    let attn = config::ATTENTION_LAYERS;
    let dense: Vec<usize> = (0..config::N_LAYERS).filter(|&i| config::is_dense_ffn(i)).collect();
    println!("OK: loaded and validated {path}");
    println!(
        "arch={} layers={} hidden={} heads={}/{} experts={}(top-{}) vocab={}",
        config::ARCH,
        config::N_LAYERS,
        config::HIDDEN,
        config::N_HEADS,
        config::N_KV_HEADS,
        config::N_EXPERTS,
        config::N_EXPERTS_USED,
        config::VOCAB,
    );
    println!("operators: attention at {attn:?}, gated short-conv elsewhere");
    println!("ffn      : dense at {dense:?}, sparse-MoE elsewhere");
    println!("all {} expected tensors present with correct shapes", bebelm::model::expected_tensors().len());
    Ok(())
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

/// Dequantize a named tensor and print summary statistics — a real-data sanity check on
/// the dequant kernels (weights should be small, roughly zero-mean).
fn cmd_dequant(path: &str, name: &str) -> Cmd {
    let g = GgufFile::open(path)?;
    let Some(t) = g.tensors.iter().find(|t| t.name == name) else {
        eprintln!("error: no tensor named '{name}'");
        eprintln!("hint: run `bebelm dump {path}` to list tensor names");
        std::process::exit(1);
    };

    if !dequant::supports(t.ggml_type) {
        eprintln!("error: dequant not implemented for type {}", t.ggml_type);
        std::process::exit(1);
    }

    let n = t.n_elements() as usize;
    let values = dequant::dequantize(t.ggml_type, g.tensor_data(t), n);

    let shape = t.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut nonfinite = 0usize;
    for &v in &values {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
            sum += v as f64;
        } else {
            nonfinite += 1;
        }
    }
    let mean = sum / (n.max(1) as f64);

    println!("tensor   : {name}");
    println!("type     : {}", t.ggml_type);
    println!("shape    : {shape}  ({n} elements)");
    println!("min      : {min}");
    println!("max      : {max}");
    println!("mean     : {mean:.6}");
    if nonfinite > 0 {
        println!("non-finite: {nonfinite}  (UNEXPECTED)");
    }
    let head: Vec<String> = values.iter().take(8).map(|v| format!("{v:.5}")).collect();
    println!("first 8  : [{}]", head.join(", "));
    Ok(())
}

fn cmd_dump(path: &str) -> Cmd {
    let g = GgufFile::open(path)?;

    println!("== {path} ==");
    println!("gguf version : {}", g.version);
    println!("architecture : {}", g.architecture().unwrap_or("<unknown>"));
    println!("alignment    : {}", g.alignment);
    println!("data offset  : {}", g.data_offset);
    println!("metadata keys: {}", g.metadata.len());
    println!("tensors      : {}", g.tensors.len());

    // --- metadata, sorted by key ---
    println!("\n-- metadata --");
    let mut keys: Vec<&String> = g.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = g.metadata.get(k).unwrap();
        println!("  {k} = {}", format_value(v));
    }

    // --- key config (architecture-prefixed hyperparameters), if present ---
    if let Some(arch) = g.architecture() {
        println!("\n-- key config ({arch}) --");
        let int_keys = [
            "block_count",
            "context_length",
            "embedding_length",
            "feed_forward_length",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
            "attention.value_length",
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "vocab_size",
        ];
        for suffix in int_keys {
            if let Some(v) = g.get_u32(&format!("{arch}.{suffix}")) {
                println!("  {suffix:<32} = {v}");
            }
        }
        let float_keys = ["attention.layer_norm_rms_epsilon", "rope.freq_base"];
        for suffix in float_keys {
            if let Some(v) = g.get_f32(&format!("{arch}.{suffix}")) {
                println!("  {suffix:<32} = {v}");
            }
        }
    }

    // --- tensors, in file order ---
    println!("\n-- tensors --");
    println!("  {:>4}  {:<8}  {:<22}  {:>12}  name", "idx", "type", "shape", "offset");
    for (i, t) in g.tensors.iter().enumerate() {
        let shape = t
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        println!(
            "  {:>4}  {:<8}  {:<22}  {:>12}  {}",
            i,
            t.ggml_type.to_string(),
            shape,
            t.offset,
            t.name
        );
    }

    // --- summary: count + bytes per dtype ---
    println!("\n-- dtype summary --");
    let mut by_type: std::collections::BTreeMap<String, (u64, u64)> = Default::default();
    let mut total_bytes: u64 = 0;
    for t in &g.tensors {
        let e = by_type.entry(t.ggml_type.to_string()).or_default();
        e.0 += 1;
        e.1 += t.data_len;
        total_bytes += t.data_len;
    }
    for (ty, (count, bytes)) in &by_type {
        println!("  {:<8} {:>4} tensors  {:>10}", ty, count, human_bytes(*bytes));
    }
    println!("  {:<8} {:>4} tensors  {:>10}", "TOTAL", g.tensors.len(), human_bytes(total_bytes));

    Ok(())
}

/// Render a metadata value compactly. Large arrays are summarized rather than dumped.
fn format_value(v: &MetaValue) -> String {
    match v {
        MetaValue::U8(x) => x.to_string(),
        MetaValue::I8(x) => x.to_string(),
        MetaValue::U16(x) => x.to_string(),
        MetaValue::I16(x) => x.to_string(),
        MetaValue::U32(x) => x.to_string(),
        MetaValue::I32(x) => x.to_string(),
        MetaValue::U64(x) => x.to_string(),
        MetaValue::I64(x) => x.to_string(),
        MetaValue::F32(x) => x.to_string(),
        MetaValue::F64(x) => x.to_string(),
        MetaValue::Bool(x) => x.to_string(),
        MetaValue::String(s) => format!("{:?}", truncate(s, 96)),
        MetaValue::Array { elem_type, items } => {
            // Show short arrays inline; summarize long ones (e.g. 128k-token vocabs).
            if items.len() <= 8 {
                let inner = items.iter().map(format_value).collect::<Vec<_>>().join(", ");
                format!("[{inner}]")
            } else {
                let head = items
                    .iter()
                    .take(3)
                    .map(format_value)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[<elem_type {elem_type}>; {} items] {{ {head}, ... }}", items.len())
            }
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}
