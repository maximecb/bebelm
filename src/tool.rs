//! Function calling: the tool declarations advertised to the model, and the parser for the
//! Pythonic call syntax it emits back.
//!
//! LFM2.5 is told about tools via a `List of tools: [{…}, …]` preamble in the system block
//! (each tool a JSON object with `name`/`description`/`parameters`), then emits calls as a
//! Python-style list — `[get_weather(city='Paris'), add(a=21, b=21)]` — between the
//! `<|tool_call_start|>` / `<|tool_call_end|>` control tokens. We build the JSON by hand (no
//! `serde`) and parse the call list by hand (no `regex`), consistent with the rest of the crate.
//!
//! [`Agent`](crate::agent::Agent) owns the registered [`Tool`]s and drives the call/result loop;
//! this module only models a tool and turns text into [`ToolCall`]s.

use std::sync::Arc;

/// A registered tool: its JSON declaration (advertised to the model in the system block) plus a
/// callback invoked when the model calls it. The callback is held behind an [`Arc`] so `Tool`
/// — and therefore [`Agent`](crate::agent::Agent) — stays [`Clone`].
#[derive(Clone)]
pub struct Tool {
    /// Must match the `name` inside `schema` (this is what the model emits and we dispatch on).
    name: String,
    /// Tool JSON schema: `{"name":…,"description":…,"parameters":{…}}`.
    schema: String,
    /// Invoked with the parsed call; returns the result text fed back to the model.
    call: Arc<dyn Fn(&ToolCall) -> String>,
}

impl Tool {
    /// Define a tool from structured parameters. The full tool JSON (with `name` and
    /// `description` JSON-escaped) is assembled from `params` for you.
    pub fn new(
        name: impl Into<String>,
        description: &str,
        params: Schema,
        call: impl Fn(&ToolCall) -> String + 'static,
    ) -> Self {
        let name = name.into();
        let mut schema = String::new();
        schema.push_str("{\"name\":");
        push_json_string(&mut schema, &name);
        schema.push_str(",\"description\":");
        push_json_string(&mut schema, description);
        schema.push_str(",\"parameters\":");
        params.render(&mut schema);
        schema.push('}');
        Tool { name, schema, call: Arc::new(call) }
    }

    /// Escape hatch: supply the entire tool JSON (including `parameters`) verbatim. `name` must
    /// match the `"name"` field inside `schema` so calls dispatch correctly.
    pub fn raw(
        name: impl Into<String>,
        schema: impl Into<String>,
        call: impl Fn(&ToolCall) -> String + 'static,
    ) -> Self {
        Tool { name: name.into(), schema: schema.into(), call: Arc::new(call) }
    }

    /// The tool's name (the identifier the model emits and we dispatch on).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The tool's JSON declaration, as emitted into the `List of tools:` preamble.
    pub(crate) fn schema(&self) -> &str {
        &self.schema
    }

    /// Run the tool's callback for a parsed call.
    pub(crate) fn invoke(&self, call: &ToolCall) -> String {
        (self.call)(call)
    }
}

/// A JSON-Schema parameter type. Renders the `"type":"…"` field the model was trained on.
#[derive(Clone, Copy)]
pub enum Type {
    Str,
    Int,
    Num,
    Bool,
}

impl Type {
    fn json(self) -> &'static str {
        match self {
            Type::Str => "string",
            Type::Int => "integer",
            Type::Num => "number",
            Type::Bool => "boolean",
        }
    }
}

/// One declared parameter of a [`Schema`].
struct Param {
    name: String,
    ty: Type,
    desc: String,
    required: bool,
}

/// Builder for a tool's `parameters` object. Renders the standard JSON-Schema string
/// `{"type":"object","properties":{…},"required":[…]}` via a small hand-rolled emitter (no
/// `serde`). Fields are emitted in the order they were added.
#[derive(Default)]
pub struct Schema {
    params: Vec<Param>,
}

impl Schema {
    pub fn new() -> Self {
        Schema::default()
    }

    /// Add a required parameter.
    pub fn req(mut self, name: &str, ty: Type, desc: &str) -> Self {
        self.params.push(Param { name: name.into(), ty, desc: desc.into(), required: true });
        self
    }

    /// Add an optional parameter.
    pub fn opt(mut self, name: &str, ty: Type, desc: &str) -> Self {
        self.params.push(Param { name: name.into(), ty, desc: desc.into(), required: false });
        self
    }

    /// Append the rendered `parameters` JSON object to `out`.
    fn render(&self, out: &mut String) {
        out.push_str("{\"type\":\"object\",\"properties\":{");
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            push_json_string(out, &p.name);
            out.push_str(":{\"type\":\"");
            out.push_str(p.ty.json());
            out.push_str("\",\"description\":");
            push_json_string(out, &p.desc);
            out.push('}');
        }
        out.push_str("},\"required\":[");
        let mut first = true;
        for p in self.params.iter().filter(|p| p.required) {
            if !first {
                out.push(',');
            }
            first = false;
            push_json_string(out, &p.name);
        }
        out.push_str("]}");
    }
}

/// One parsed tool call: the function name and its keyword arguments.
pub struct ToolCall {
    pub name: String,
    /// `(arg name, value text)` in call order. Quotes are stripped from simple string literals;
    /// `{…}`/`[…]` values are kept verbatim. Look values up with [`ToolCall::arg`].
    args: Vec<(String, String)>,
    /// The raw call text, e.g. `get_weather(city='Paris')`.
    pub raw: String,
}

impl ToolCall {
    /// The value passed for argument `name`, if present.
    pub fn arg(&self, name: &str) -> Option<&str> {
        self.args.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// Argument `name` parsed into type `T` (`i64`, `usize`, `f32`, `bool`, …) via
    /// [`FromStr`](std::str::FromStr), inferred from the receiver at the call site. Returns `None`
    /// if the argument is absent or its value doesn't parse as `T` — the model controls these
    /// values, so a bad one is its mistake, not a reason to panic. Pick the fallback at the call
    /// site: `.unwrap_or(default)`, or return an error string so the model can correct itself.
    pub fn parse_arg<T: std::str::FromStr>(&self, name: &str) -> Option<T> {
        self.arg(name)?.parse().ok()
    }
}

/// Parse the content between `<|tool_call_start|>` and `<|tool_call_end|>` into calls. Accepts a
/// Python-style list `[f(a=1), g(b='x')]` (the outer brackets are optional). Values may be quoted
/// strings (unescaped, quotes stripped), numbers/booleans, or brace/bracket-balanced literals
/// (kept verbatim). Hand-rolled, consistent with the no-`regex` tokenizer.
pub(crate) fn parse_tool_calls(s: &str) -> Vec<ToolCall> {
    let chars: Vec<char> = s.chars().collect();
    let mut p = Parser { c: &chars, i: 0 };
    p.skip_ws();
    if p.peek() == Some('[') {
        p.i += 1; // optional outer list bracket
    }
    let mut calls = Vec::new();
    loop {
        p.skip_ws();
        match p.peek() {
            None => break,
            Some(']') => break,         // end of the outer list
            Some(',') => p.i += 1,      // separator between calls
            _ => match p.parse_call() {
                Some(call) => calls.push(call),
                None => break,          // not a call: stop rather than spin
            },
        }
    }
    calls
}

/// Cursor over the call-list characters.
struct Parser<'a> {
    c: &'a [char],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.c.get(self.i).copied()
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.i += 1;
        }
    }

    /// `name . name …` — an identifier, allowing dotted names and underscores.
    fn parse_ident(&mut self) -> String {
        let start = self.i;
        while self.peek().is_some_and(|ch| ch.is_alphanumeric() || ch == '_' || ch == '.') {
            self.i += 1;
        }
        self.c[start..self.i].iter().collect()
    }

    /// `name(arg=value, …)` — one call. Returns `None` if the next token isn't `name(`.
    fn parse_call(&mut self) -> Option<ToolCall> {
        self.skip_ws();
        let start = self.i;
        let name = self.parse_ident();
        self.skip_ws();
        if name.is_empty() || self.peek() != Some('(') {
            return None;
        }
        self.i += 1; // consume '('
        let mut args = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                Some(')') => {
                    self.i += 1;
                    break;
                }
                None => break,
                Some(',') => {
                    self.i += 1;
                    continue;
                }
                _ => {}
            }
            let key = self.parse_ident();
            self.skip_ws();
            if self.peek() == Some('=') {
                self.i += 1;
            }
            let val = self.parse_value();
            args.push((key, val));
        }
        let raw: String = self.c[start..self.i].iter().collect();
        Some(ToolCall { name, args, raw: raw.trim().to_string() })
    }

    /// A single argument value: quoted string, balanced `{…}`/`[…]`, or a bare token.
    fn parse_value(&mut self) -> String {
        self.skip_ws();
        match self.peek() {
            Some('\'') | Some('"') => self.parse_quoted(),
            Some('{') => self.parse_balanced('{', '}'),
            Some('[') => self.parse_balanced('[', ']'),
            _ => self.parse_bare(),
        }
    }

    /// A `'…'` / `"…"` string: quotes stripped, `\n`/`\t`/`\r`/`\\`/`\'`/`\"` unescaped.
    fn parse_quoted(&mut self) -> String {
        let quote = self.peek().unwrap();
        self.i += 1; // opening quote
        let mut out = String::new();
        while let Some(ch) = self.peek() {
            self.i += 1;
            if ch == '\\' {
                if let Some(esc) = self.peek() {
                    self.i += 1;
                    out.push(match esc {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        other => other, // \' \" \\ and anything else: literal
                    });
                }
            } else if ch == quote {
                break;
            } else {
                out.push(ch);
            }
        }
        out
    }

    /// A `{…}` / `[…]` literal kept verbatim (including the brackets). String contents are
    /// skipped so a quoted bracket doesn't throw off the depth count.
    fn parse_balanced(&mut self, open: char, close: char) -> String {
        let start = self.i;
        let mut depth = 0usize;
        let mut in_str: Option<char> = None;
        while let Some(ch) = self.peek() {
            self.i += 1;
            match in_str {
                Some(q) => {
                    if ch == '\\' {
                        self.i += 1; // skip the escaped char
                    } else if ch == q {
                        in_str = None;
                    }
                }
                None => {
                    if ch == '\'' || ch == '"' {
                        in_str = Some(ch);
                    } else if ch == open {
                        depth += 1;
                    } else if ch == close {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                }
            }
        }
        self.c[start..self.i].iter().collect()
    }

    /// A bare token (number, bool, unquoted word): up to the next `,`/`)`/`]`/`}`.
    fn parse_bare(&mut self) -> String {
        let start = self.i;
        while let Some(ch) = self.peek() {
            if matches!(ch, ',' | ')' | ']' | '}') {
                break;
            }
            self.i += 1;
        }
        self.c[start..self.i].iter().collect::<String>().trim().to_string()
    }
}

/// Append `s` to `out` as a JSON string literal (surrounding quotes + escaping).
fn push_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_renders_typed_properties_and_required() {
        let s = Tool::new(
            "get_weather",
            "Current weather for a city.",
            Schema::new().req("city", Type::Str, "City to look up").opt("units", Type::Str, "C or F"),
            |_| String::new(),
        );
        assert_eq!(
            s.schema(),
            r#"{"name":"get_weather","description":"Current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City to look up"},"units":{"type":"string","description":"C or F"}},"required":["city"]}}"#
        );
    }

    #[test]
    fn schema_escapes_json_special_chars() {
        let t = Tool::new("q", "Say \"hi\"\nnow", Schema::new(), |_| String::new());
        assert_eq!(
            t.schema(),
            r#"{"name":"q","description":"Say \"hi\"\nnow","parameters":{"type":"object","properties":{},"required":[]}}"#
        );
    }

    #[test]
    fn schema_renders_all_types() {
        let t = Tool::new(
            "f",
            "d",
            Schema::new()
                .req("a", Type::Int, "i")
                .req("b", Type::Num, "n")
                .req("c", Type::Bool, "b"),
            |_| String::new(),
        );
        assert!(t.schema().contains(r#""a":{"type":"integer""#));
        assert!(t.schema().contains(r#""b":{"type":"number""#));
        assert!(t.schema().contains(r#""c":{"type":"boolean""#));
    }

    #[test]
    fn parse_single_call_with_string_arg() {
        let calls = parse_tool_calls("[get_weather(city='Paris')]");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arg("city"), Some("Paris"));
        assert_eq!(calls[0].raw, "get_weather(city='Paris')");
    }

    #[test]
    fn parse_multiple_calls_and_arg_types() {
        let calls = parse_tool_calls("[get_weather(city='Paris', units='celsius'), add(a=21, b=21)]");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arg("city"), Some("Paris"));
        assert_eq!(calls[0].arg("units"), Some("celsius"));
        assert_eq!(calls[1].name, "add");
        assert_eq!(calls[1].arg("a"), Some("21"));
        assert_eq!(calls[1].arg("b"), Some("21"));
        assert_eq!(calls[0].arg("missing"), None);
    }

    #[test]
    fn parse_arg_parses_into_receiver_type() {
        let calls = parse_tool_calls("[add(a=21, b='oops', big=4000000000, r=2.5)]");
        assert_eq!(calls[0].parse_arg::<i64>("a"), Some(21));
        assert_eq!(calls[0].parse_arg::<u8>("a"), Some(21)); // narrows to the receiver's type
        assert_eq!(calls[0].parse_arg::<f32>("r"), Some(2.5)); // not just integers
        assert_eq!(calls[0].parse_arg::<u32>("b"), None); // present but doesn't parse
        assert_eq!(calls[0].parse_arg::<i64>("missing"), None); // absent
        assert_eq!(calls[0].parse_arg::<u8>("big"), None); // out of range for u8
        assert_eq!(calls[0].parse_arg::<u64>("big"), Some(4_000_000_000));
    }

    #[test]
    fn parse_bool_and_double_quoted() {
        let calls = parse_tool_calls(r#"[set(flag=True, name="Bob")]"#);
        assert_eq!(calls[0].arg("flag"), Some("True"));
        assert_eq!(calls[0].arg("name"), Some("Bob"));
    }

    #[test]
    fn parse_keeps_dict_and_list_values_verbatim() {
        let calls = parse_tool_calls("[f(d={'k': 1, 'n': 2}, xs=[1, 2, 3])]");
        assert_eq!(calls[0].arg("d"), Some("{'k': 1, 'n': 2}"));
        assert_eq!(calls[0].arg("xs"), Some("[1, 2, 3]"));
    }

    #[test]
    fn parse_unescapes_quoted_strings() {
        let calls = parse_tool_calls(r#"[note(text='line1\nline2', q='it\'s')]"#);
        assert_eq!(calls[0].arg("text"), Some("line1\nline2"));
        assert_eq!(calls[0].arg("q"), Some("it's"));
    }

    #[test]
    fn parse_works_without_outer_brackets_and_no_args() {
        let calls = parse_tool_calls("now()");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "now");
        assert!(calls[0].arg("anything").is_none());
    }

    #[test]
    fn parse_empty_or_garbage_yields_no_calls() {
        assert!(parse_tool_calls("").is_empty());
        assert!(parse_tool_calls("   ").is_empty());
        assert!(parse_tool_calls("[]").is_empty());
    }
}
