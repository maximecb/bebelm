//! Byte-level BPE tokenizer (GPT-2 / `gpt2` GGUF type), built from the vocab + merges
//! embedded in the GGUF. Pure Rust, no `regex` crate: the LFM2 (Llama-3-style)
//! pre-tokenizer is hand-rolled as an ordered-alternation scanner over chars.
//!
//! Pipeline: text → pre-tokenize into pieces → map each piece's *bytes* through the GPT-2
//! byte→char table → greedily apply merges by rank → look up token ids. Decode reverses it.

use std::collections::HashMap;
use std::error::Error;

use crate::gguf::GgufFile;

pub struct Tokenizer {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// (left, right) byte-char symbol pair → merge rank (lower = merged first).
    merge_rank: HashMap<(String, String), u32>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    pub bos: u32,
    pub eos: u32,
}

impl Tokenizer {
    /// Build from a loaded GGUF's `tokenizer.ggml.*` metadata.
    pub fn from_gguf(g: &GgufFile) -> Result<Tokenizer, Box<dyn Error>> {
        let model = g.get_str("tokenizer.ggml.model").unwrap_or("");
        if model != "gpt2" {
            return Err(format!("unsupported tokenizer model {model:?} (expected gpt2)").into());
        }
        let id_to_token = g
            .get_str_array("tokenizer.ggml.tokens")
            .ok_or("missing tokenizer.ggml.tokens")?;
        let merges = g
            .get_str_array("tokenizer.ggml.merges")
            .ok_or("missing tokenizer.ggml.merges")?;

        let mut token_to_id = HashMap::with_capacity(id_to_token.len());
        for (i, t) in id_to_token.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            // Each merge is "A B"; byte-encoded symbols never contain a literal space, so
            // splitting on the first space is unambiguous.
            if let Some((a, b)) = m.split_once(' ') {
                merge_rank.insert((a.to_string(), b.to_string()), rank as u32);
            }
        }

        let byte_encoder = byte_to_unicode();
        let byte_decoder = byte_encoder.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();

        let bos = g.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(crate::config::BOS_TOKEN);
        let eos = g.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(crate::config::EOS_TOKEN);

        Ok(Tokenizer { id_to_token, token_to_id, merge_rank, byte_encoder, byte_decoder, bos, eos })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Encode text to token ids, optionally prepending BOS.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.bos);
        }
        for piece in pretokenize(text) {
            // byte-level: each UTF-8 byte of the piece maps to a visible char.
            let mapped: String = piece.bytes().map(|b| self.byte_encoder[b as usize]).collect();
            for sym in self.bpe(&mapped) {
                if let Some(&id) = self.token_to_id.get(&sym) {
                    ids.push(id);
                } else {
                    // Fallback (shouldn't happen — the vocab includes all single byte-chars).
                    for ch in sym.chars() {
                        if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
        ids
    }

    /// Decode token ids back to text (special/control tokens with non-byte chars are dropped).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok) = self.id_to_token.get(id as usize) {
                for ch in tok.chars() {
                    if let Some(&b) = self.byte_decoder.get(&ch) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Greedy BPE merge of a byte-char string into subword symbols.
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if symbols.len() < 2 {
            return symbols;
        }
        loop {
            // Find the adjacent pair with the lowest merge rank.
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len() - 1 {
                if let Some(&r) = self.merge_rank.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                    if best.is_none_or(|(_, br)| r < br) {
                        best = Some((i, r));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..i + 2, [merged]);
        }
        symbols
    }
}

/// GPT-2 `bytes_to_unicode`: a reversible map from each byte to a printable char.
fn byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut used = [false; 256];
    // "Printable" byte ranges map to the char with that codepoint.
    for &(lo, hi) in &[(b'!', b'~'), (0xA1u8, 0xACu8), (0xAEu8, 0xFFu8)] {
        for b in lo..=hi {
            table[b as usize] = char::from_u32(b as u32).unwrap();
            used[b as usize] = true;
        }
    }
    // The rest map to U+0100, U+0101, … so every byte has a distinct visible char.
    let mut n = 0u32;
    for b in 0..256 {
        if !used[b] {
            table[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    table
}

// --- pre-tokenizer (LFM2 / Llama-3 regex, hand-rolled) ---

#[inline]
fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}
#[inline]
fn is_number(c: char) -> bool {
    c.is_numeric()
}

/// Split `text` into pieces using the LFM2 pre-tokenizer pattern. At each position the
/// alternatives are tried in order; the first that matches consumes its span.
fn pretokenize(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let n = c.len();
    let mut pieces = Vec::new();
    let mut i = 0;
    while i < n {
        let len = match_contraction(&c, i)
            .or_else(|| match_letters(&c, i))
            .or_else(|| match_numbers(&c, i))
            .or_else(|| match_punct(&c, i))
            .or_else(|| match_newline_run(&c, i))
            .or_else(|| match_trailing_ws(&c, i))
            .or_else(|| match_ws(&c, i))
            .unwrap_or(1); // guarantee progress on anything unexpected
        pieces.push(c[i..i + len].iter().collect());
        i += len;
    }
    pieces
}

/// `'(?i:[sdmt]|ll|ve|re)` — contraction suffixes.
fn match_contraction(c: &[char], i: usize) -> Option<usize> {
    if c.get(i) != Some(&'\'') {
        return None;
    }
    let c1 = c.get(i + 1)?.to_ascii_lowercase();
    if matches!(c1, 's' | 'd' | 'm' | 't') {
        return Some(2);
    }
    let c2 = c.get(i + 2)?.to_ascii_lowercase();
    if matches!((c1, c2), ('l', 'l') | ('v', 'e') | ('r', 'e')) {
        return Some(3);
    }
    None
}

/// `[^\r\n\p{L}\p{N}]?\p{L}+` — letters, with an optional single non-letter/digit prefix.
fn match_letters(c: &[char], i: usize) -> Option<usize> {
    let mut j = i;
    let cur = c[i];
    let prefixable = cur != '\r' && cur != '\n' && !is_letter(cur) && !is_number(cur);
    if prefixable && c.get(i + 1).is_some_and(|&d| is_letter(d)) {
        j += 1; // consume the prefix char
    }
    let letters_start = j;
    while c.get(j).is_some_and(|&ch| is_letter(ch)) {
        j += 1;
    }
    (j > letters_start).then_some(j - i)
}

/// `\p{N}{1,3}` — one to three digits.
fn match_numbers(c: &[char], i: usize) -> Option<usize> {
    let mut j = i;
    while j < i + 3 && c.get(j).is_some_and(|&ch| is_number(ch)) {
        j += 1;
    }
    (j > i).then_some(j - i)
}

/// ` ?[^\s\p{L}\p{N}]+[\r\n]*` — punctuation/symbols with an optional leading space.
fn match_punct(c: &[char], i: usize) -> Option<usize> {
    let is_punct = |ch: char| !ch.is_whitespace() && !is_letter(ch) && !is_number(ch);
    let mut j = i;
    if c[i] == ' ' && c.get(i + 1).is_some_and(|&d| is_punct(d)) {
        j += 1; // optional single space
    }
    let punct_start = j;
    while c.get(j).is_some_and(|&ch| is_punct(ch)) {
        j += 1;
    }
    if j == punct_start {
        return None;
    }
    while c.get(j).is_some_and(|&ch| ch == '\r' || ch == '\n') {
        j += 1;
    }
    Some(j - i)
}

/// `\s*[\r\n]` — a whitespace run ending at (and including) its last newline.
fn match_newline_run(c: &[char], i: usize) -> Option<usize> {
    let mut j = i;
    while c.get(j).is_some_and(|&ch| ch.is_whitespace()) {
        j += 1;
    }
    let last_nl = (i..j).rev().find(|&p| c[p] == '\r' || c[p] == '\n')?;
    Some(last_nl + 1 - i)
}

/// `\s+(?!\S)` — a whitespace run not immediately followed by a non-space.
fn match_trailing_ws(c: &[char], i: usize) -> Option<usize> {
    let mut j = i;
    while c.get(j).is_some_and(|&ch| ch.is_whitespace()) {
        j += 1;
    }
    if j == i {
        return None;
    }
    if j >= c.len() {
        Some(j - i) // run reaches end-of-text
    } else if j - 1 > i {
        Some(j - 1 - i) // leave one space (the next char is non-ws) for the following token
    } else {
        None
    }
}

/// `\s+` — any whitespace run.
fn match_ws(c: &[char], i: usize) -> Option<usize> {
    let mut j = i;
    while c.get(j).is_some_and(|&ch| ch.is_whitespace()) {
        j += 1;
    }
    (j > i).then_some(j - i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(s: &str) -> Vec<String> {
        pretokenize(s)
    }

    #[test]
    fn byte_table_is_reversible() {
        let enc = byte_to_unicode();
        let mut seen = std::collections::HashSet::new();
        for (b, &ch) in enc.iter().enumerate() {
            assert!(seen.insert(ch), "byte {b} char not unique");
        }
    }

    #[test]
    fn pretokenize_words_and_spaces() {
        assert_eq!(pt("The capital of France is"), ["The", " capital", " of", " France", " is"]);
        assert_eq!(pt("Hello world"), ["Hello", " world"]);
        // two leading spaces: first is its own piece, second attaches to the word.
        assert_eq!(pt("  hi"), [" ", " hi"]);
    }

    #[test]
    fn pretokenize_contractions_numbers_punct() {
        assert_eq!(pt("I'm"), ["I", "'m"]);
        assert_eq!(pt("don't"), ["don", "'t"]);
        assert_eq!(pt("abc123"), ["abc", "123"]);
        assert_eq!(pt("1234"), ["123", "4"]); // \p{N}{1,3}
        assert_eq!(pt("hello!"), ["hello", "!"]);
        assert_eq!(pt("a, b"), ["a", ",", " b"]);
    }
}
