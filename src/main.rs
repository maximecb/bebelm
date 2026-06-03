//! bebelm — CPU-only, pure-Rust inference for Liquid AI LFM2.5-8B-A1B (Q4_K_M).
//!
//! Milestone 1: a GGUF loader with a `dump` command that lists the file's metadata and
//! every tensor (name / dtype / shape / size), so we can validate the real model against
//! the architecture described in design.md.

mod gguf;
mod tensor;

use std::process::ExitCode;

use gguf::{GgufFile, MetaValue};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("dump") => match args.get(2) {
            Some(path) => match cmd_dump(path) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            },
            None => {
                eprintln!("usage: bebelm dump <model.gguf>");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("bebelm — CPU-only LFM2.5-8B-A1B inference\n");
            eprintln!("usage:");
            eprintln!("  bebelm dump <model.gguf>   list metadata and tensors");
            ExitCode::FAILURE
        }
    }
}

fn cmd_dump(path: &str) -> gguf::Result<()> {
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
