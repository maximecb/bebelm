//! bebelm — CPU-only, pure-Rust inference for Liquid AI LFM2.5-8B-A1B (Q4_K_M).
//!
//! CLI over the `bebelm` library: inspect a GGUF (`dump`), sanity-check the dequant
//! kernels on a tensor (`dequant`), or load + validate the whole model (`load`).

use std::error::Error;
use std::process::ExitCode;

use bebelm::config;
use bebelm::gguf::{GgufFile, MetaValue};
use bebelm::kernels::dequant;
use bebelm::model::Model;

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
        _ => {
            eprintln!("bebelm — CPU-only LFM2.5-8B-A1B inference\n");
            eprintln!("usage:");
            eprintln!("  bebelm dump    <model.gguf>                 list metadata and tensors");
            eprintln!("  bebelm dequant <model.gguf> <tensor-name>   dequantize a tensor, print stats");
            eprintln!("  bebelm load    <model.gguf>                 load + validate against the config");
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
