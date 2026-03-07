/// Rule serialization / deserialization.
///
/// Supports N3/Notation3 and JSON rule formats with full round-trip fidelity,
/// metadata preservation, batch operations, and structured error reporting.
use std::fmt::Write as FmtWrite;

use thiserror::Error;

// ── Error types ───────────────────────────────────────────────────────────────

/// Line/column position within a source string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePos {
    pub line: usize,
    pub column: usize,
}

impl std::fmt::Display for SourcePos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// Errors that can occur during serialization or deserialization.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SerializerError {
    /// A parse error with position information.
    #[error("parse error at {pos}: {message}")]
    Parse { pos: SourcePos, message: String },

    /// An I/O-style formatting error.
    #[error("format error: {message}")]
    Format { message: String },

    /// A JSON structural error.
    #[error("json error at {pos}: {message}")]
    Json { pos: SourcePos, message: String },

    /// Unknown/unsupported rule format requested.
    #[error("unsupported format: {format}")]
    UnsupportedFormat { format: String },
}

impl SerializerError {
    fn parse(line: usize, column: usize, msg: impl Into<String>) -> Self {
        SerializerError::Parse {
            pos: SourcePos { line, column },
            message: msg.into(),
        }
    }

    fn json_err(line: usize, column: usize, msg: impl Into<String>) -> Self {
        SerializerError::Json {
            pos: SourcePos { line, column },
            message: msg.into(),
        }
    }
}

// ── Core data structures ──────────────────────────────────────────────────────

/// A single atom in a rule body or head (subject predicate object).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleAtom {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl RuleAtom {
    /// Create a new rule atom.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        RuleAtom {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    /// Returns `true` if all three components are identical to another atom.
    pub fn matches(&self, other: &RuleAtom) -> bool {
        self == other
    }
}

/// Serialization format selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleFormat {
    /// N3/Notation3: `{ antecedent } => { consequent } .`
    N3,
    /// JSON: `{"name": "...", "if": [...], "then": [...], ...}`
    Json,
}

/// A rule with antecedent, consequent, and optional metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerializableRule {
    /// Human-readable rule name.
    pub name: Option<String>,
    /// Rule priority (higher = applied first).
    pub priority: i32,
    /// Whether this rule is active.
    pub enabled: bool,
    /// Body (antecedent) atoms.
    pub antecedent: Vec<RuleAtom>,
    /// Head (consequent) atoms.
    pub consequent: Vec<RuleAtom>,
    /// Optional namespace prefix table used during serialization.
    pub prefixes: Vec<(String, String)>,
}

impl SerializableRule {
    /// Create a minimal rule with no metadata.
    pub fn new(antecedent: Vec<RuleAtom>, consequent: Vec<RuleAtom>) -> Self {
        SerializableRule {
            name: None,
            priority: 0,
            enabled: true,
            antecedent,
            consequent,
            prefixes: Vec::new(),
        }
    }

    /// Attach a human-readable name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set execution priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Mark as disabled.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Add a namespace prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.push((prefix.into(), iri.into()));
        self
    }
}

// ── Prefix / namespace helpers ────────────────────────────────────────────────

/// Attempt to contract an IRI using the provided prefix table.
/// Returns the CURIE form if a prefix matches, otherwise the original IRI.
fn apply_prefixes(iri: &str, prefixes: &[(String, String)]) -> String {
    for (prefix, ns) in prefixes {
        if let Some(local) = iri.strip_prefix(ns.as_str()) {
            return format!("{}:{}", prefix, local);
        }
    }
    iri.to_owned()
}

/// Expand a potentially-prefixed term using the prefix table.
/// If no match is found the term is returned unchanged.
fn expand_term(term: &str, prefixes: &[(String, String)]) -> String {
    if let Some(colon) = term.find(':') {
        let prefix = &term[..colon];
        let local = &term[colon + 1..];
        for (p, ns) in prefixes {
            if p == prefix {
                return format!("{}{}", ns, local);
            }
        }
    }
    term.to_owned()
}

// ── N3 serializer ─────────────────────────────────────────────────────────────

/// Serialize a single `SerializableRule` to N3 notation.
pub fn serialize_n3(rule: &SerializableRule) -> Result<String, SerializerError> {
    let mut out = String::new();

    // Emit @prefix declarations.
    for (prefix, iri) in &rule.prefixes {
        writeln!(out, "@prefix {}: <{}> .", prefix, iri).map_err(|e| SerializerError::Format {
            message: e.to_string(),
        })?;
    }
    if !rule.prefixes.is_empty() {
        writeln!(out).map_err(|e| SerializerError::Format {
            message: e.to_string(),
        })?;
    }

    // Optional rule name as a comment.
    if let Some(name) = &rule.name {
        writeln!(out, "# rule: {}", name).map_err(|e| SerializerError::Format {
            message: e.to_string(),
        })?;
    }
    if rule.priority != 0 {
        writeln!(out, "# priority: {}", rule.priority).map_err(|e| SerializerError::Format {
            message: e.to_string(),
        })?;
    }
    if !rule.enabled {
        writeln!(out, "# enabled: false").map_err(|e| SerializerError::Format {
            message: e.to_string(),
        })?;
    }

    // Antecedent formula.
    out.push('{');
    for (i, atom) in rule.antecedent.iter().enumerate() {
        if i > 0 {
            out.push_str(" . ");
        }
        let s = apply_prefixes(&atom.subject, &rule.prefixes);
        let p = apply_prefixes(&atom.predicate, &rule.prefixes);
        let o = apply_prefixes(&atom.object, &rule.prefixes);
        write!(out, " {} {} {}", s, p, o).map_err(|e| SerializerError::Format {
            message: e.to_string(),
        })?;
    }
    if !rule.antecedent.is_empty() {
        out.push(' ');
    }
    out.push_str("} => {");

    // Consequent formula.
    for (i, atom) in rule.consequent.iter().enumerate() {
        if i > 0 {
            out.push_str(" . ");
        }
        let s = apply_prefixes(&atom.subject, &rule.prefixes);
        let p = apply_prefixes(&atom.predicate, &rule.prefixes);
        let o = apply_prefixes(&atom.object, &rule.prefixes);
        write!(out, " {} {} {}", s, p, o).map_err(|e| SerializerError::Format {
            message: e.to_string(),
        })?;
    }
    if !rule.consequent.is_empty() {
        out.push(' ');
    }
    out.push_str("} .");
    Ok(out)
}

/// Deserialize a single rule from an N3 string.
///
/// Supports the `@prefix` declaration, comment-based metadata, and the
/// `{ antecedent } => { consequent } .` rule form.
pub fn deserialize_n3(input: &str) -> Result<SerializableRule, SerializerError> {
    let mut prefixes: Vec<(String, String)> = Vec::new();
    let mut name: Option<String> = None;
    let mut priority: i32 = 0;
    let mut enabled = true;
    let mut rule_text = String::new();

    for (line_no, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();

        if line.starts_with("@prefix") {
            // @prefix foo: <http://...> .
            let rest = line.trim_start_matches("@prefix").trim();
            let rest = rest.trim_end_matches('.').trim();
            if let Some(colon_pos) = rest.find(':') {
                let prefix = rest[..colon_pos].trim().to_owned();
                let iri_part = rest[colon_pos + 1..].trim();
                let iri = iri_part
                    .trim_start_matches('<')
                    .trim_end_matches('>')
                    .to_owned();
                prefixes.push((prefix, iri));
            } else {
                return Err(SerializerError::parse(
                    line_no + 1,
                    1,
                    "malformed @prefix declaration",
                ));
            }
        } else if line.starts_with("# rule:") {
            name = Some(line.trim_start_matches("# rule:").trim().to_owned());
        } else if line.starts_with("# priority:") {
            let p_str = line.trim_start_matches("# priority:").trim();
            priority = p_str.parse().map_err(|_| {
                SerializerError::parse(line_no + 1, 1, format!("invalid priority: {}", p_str))
            })?;
        } else if line.starts_with("# enabled: false") {
            enabled = false;
        } else if !line.is_empty() && !line.starts_with('#') {
            rule_text.push(' ');
            rule_text.push_str(line);
        }
    }

    // Parse `{ ... } => { ... } .`
    let rule_text = rule_text.trim();
    let arrow = " => ";
    let arrow_pos = rule_text
        .find(arrow)
        .ok_or_else(|| SerializerError::parse(1, 1, "missing '=>' in N3 rule"))?;

    let ante_part = rule_text[..arrow_pos].trim();
    let cons_part = rule_text[arrow_pos + arrow.len()..].trim();

    let antecedent = parse_n3_formula(ante_part, &prefixes, 1)?;
    let consequent = parse_n3_formula(cons_part, &prefixes, 1)?;

    Ok(SerializableRule {
        name,
        priority,
        enabled,
        antecedent,
        consequent,
        prefixes,
    })
}

/// Parse a `{ atom . atom . ... }` formula, returning a list of atoms.
fn parse_n3_formula(
    text: &str,
    prefixes: &[(String, String)],
    base_line: usize,
) -> Result<Vec<RuleAtom>, SerializerError> {
    let inner = text
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim_end_matches('.')
        .trim();

    if inner.is_empty() {
        return Ok(Vec::new());
    }

    let mut atoms = Vec::new();
    // Split on ' . ' to get individual triple statements.
    for (idx, part) in inner.split(" . ").enumerate() {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = part.split_whitespace().collect();
        if tokens.len() < 3 {
            return Err(SerializerError::parse(
                base_line + idx,
                1,
                format!("expected triple, got: {}", part),
            ));
        }
        // Rejoin tokens beyond index 2 as the object (handles quoted literals with spaces).
        let s = expand_term(tokens[0], prefixes);
        let p = expand_term(tokens[1], prefixes);
        let o = expand_term(&tokens[2..].join(" "), prefixes);
        atoms.push(RuleAtom::new(s, p, o));
    }
    Ok(atoms)
}

// ── JSON serializer ───────────────────────────────────────────────────────────

/// Serialize a single rule to a compact JSON string.
pub fn serialize_json(rule: &SerializableRule) -> Result<String, SerializerError> {
    let mut out = String::from("{");

    if let Some(name) = &rule.name {
        write!(out, "\"name\":{},", json_str(name)).map_err(|e| SerializerError::Format {
            message: e.to_string(),
        })?;
    }
    write!(out, "\"priority\":{},", rule.priority).map_err(|e| SerializerError::Format {
        message: e.to_string(),
    })?;
    write!(out, "\"enabled\":{},", rule.enabled).map_err(|e| SerializerError::Format {
        message: e.to_string(),
    })?;

    // Prefixes.
    out.push_str("\"prefixes\":{");
    for (i, (p, iri)) in rule.prefixes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write!(out, "{}:{}", json_str(p), json_str(iri)).map_err(|e| SerializerError::Format {
            message: e.to_string(),
        })?;
    }
    out.push_str("},");

    // Antecedent.
    out.push_str("\"if\":[");
    for (i, atom) in rule.antecedent.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write!(
            out,
            "{{\"s\":{},\"p\":{},\"o\":{}}}",
            json_str(&atom.subject),
            json_str(&atom.predicate),
            json_str(&atom.object)
        )
        .map_err(|e| SerializerError::Format {
            message: e.to_string(),
        })?;
    }
    out.push_str("],");

    // Consequent.
    out.push_str("\"then\":[");
    for (i, atom) in rule.consequent.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write!(
            out,
            "{{\"s\":{},\"p\":{},\"o\":{}}}",
            json_str(&atom.subject),
            json_str(&atom.predicate),
            json_str(&atom.object)
        )
        .map_err(|e| SerializerError::Format {
            message: e.to_string(),
        })?;
    }
    out.push_str("]}");

    Ok(out)
}

/// Minimal JSON string escaping (handles `\n`, `\t`, `\"`, `\\`).
fn json_str(s: &str) -> String {
    let mut buf = String::with_capacity(s.len() + 2);
    buf.push('"');
    for c in s.chars() {
        match c {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\t' => buf.push_str("\\t"),
            '\r' => buf.push_str("\\r"),
            other => buf.push(other),
        }
    }
    buf.push('"');
    buf
}

/// Deserialize a single rule from a JSON string.
///
/// Uses a hand-written minimal parser to remain dependency-free.
pub fn deserialize_json(input: &str) -> Result<SerializableRule, SerializerError> {
    let p = JsonParser::new(input);
    p.parse_rule()
}

/// Minimal recursive-descent JSON parser for rule objects.
struct JsonParser<'a> {
    src: &'a [u8],
    pos: usize,
    line: usize,
}

impl<'a> JsonParser<'a> {
    fn new(src: &'a str) -> Self {
        JsonParser {
            src: src.as_bytes(),
            pos: 0,
            line: 1,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied()?;
        if b == b'\n' {
            self.line += 1;
        }
        self.pos += 1;
        Some(b)
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn expect_byte(&mut self, expected: u8) -> Result<(), SerializerError> {
        self.skip_ws();
        match self.advance() {
            Some(b) if b == expected => Ok(()),
            Some(b) => Err(SerializerError::json_err(
                self.line,
                self.pos,
                format!("expected '{}', got '{}'", expected as char, b as char),
            )),
            None => Err(SerializerError::json_err(
                self.line,
                self.pos,
                "unexpected end of input",
            )),
        }
    }

    fn parse_string(&mut self) -> Result<String, SerializerError> {
        self.skip_ws();
        self.expect_byte(b'"')?;
        let mut s = String::new();
        loop {
            match self.advance() {
                None => {
                    return Err(SerializerError::json_err(
                        self.line,
                        self.pos,
                        "unterminated string",
                    ))
                }
                Some(b'"') => break,
                Some(b'\\') => match self.advance() {
                    Some(b'"') => s.push('"'),
                    Some(b'\\') => s.push('\\'),
                    Some(b'n') => s.push('\n'),
                    Some(b't') => s.push('\t'),
                    Some(b'r') => s.push('\r'),
                    Some(other) => {
                        s.push('\\');
                        s.push(other as char);
                    }
                    None => {
                        return Err(SerializerError::json_err(
                            self.line,
                            self.pos,
                            "unexpected EOF after backslash",
                        ))
                    }
                },
                Some(b) => s.push(b as char),
            }
        }
        Ok(s)
    }

    fn parse_bool(&mut self) -> Result<bool, SerializerError> {
        self.skip_ws();
        let remaining = &self.src[self.pos..];
        if remaining.starts_with(b"true") {
            self.pos += 4;
            Ok(true)
        } else if remaining.starts_with(b"false") {
            self.pos += 5;
            Ok(false)
        } else {
            Err(SerializerError::json_err(
                self.line,
                self.pos,
                "expected boolean",
            ))
        }
    }

    fn parse_i32(&mut self) -> Result<i32, SerializerError> {
        self.skip_ws();
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.advance();
        }
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.advance();
        }
        let s = std::str::from_utf8(&self.src[start..self.pos]).map_err(|_| {
            SerializerError::json_err(self.line, self.pos, "invalid UTF-8 in number")
        })?;
        s.parse::<i32>().map_err(|_| {
            SerializerError::json_err(self.line, self.pos, format!("invalid integer: {}", s))
        })
    }

    fn parse_atom(&mut self) -> Result<RuleAtom, SerializerError> {
        // { "s": "...", "p": "...", "o": "..." }
        self.expect_byte(b'{')?;
        let mut s_val = String::new();
        let mut p_val = String::new();
        let mut o_val = String::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.advance();
                break;
            }
            let key = self.parse_string()?;
            self.expect_byte(b':')?;
            let val = self.parse_string()?;
            match key.as_str() {
                "s" => s_val = val,
                "p" => p_val = val,
                "o" => o_val = val,
                _ => {}
            }
            self.skip_ws();
            if self.peek() == Some(b',') {
                self.advance();
            }
        }
        Ok(RuleAtom::new(s_val, p_val, o_val))
    }

    fn parse_atom_array(&mut self) -> Result<Vec<RuleAtom>, SerializerError> {
        self.expect_byte(b'[')?;
        let mut atoms = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(b']') {
                self.advance();
                break;
            }
            atoms.push(self.parse_atom()?);
            self.skip_ws();
            if self.peek() == Some(b',') {
                self.advance();
            }
        }
        Ok(atoms)
    }

    fn parse_prefix_object(&mut self) -> Result<Vec<(String, String)>, SerializerError> {
        self.expect_byte(b'{')?;
        let mut prefixes = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.advance();
                break;
            }
            let key = self.parse_string()?;
            self.expect_byte(b':')?;
            let val = self.parse_string()?;
            prefixes.push((key, val));
            self.skip_ws();
            if self.peek() == Some(b',') {
                self.advance();
            }
        }
        Ok(prefixes)
    }

    fn parse_rule(mut self) -> Result<SerializableRule, SerializerError> {
        self.expect_byte(b'{')?;
        let mut name: Option<String> = None;
        let mut priority: i32 = 0;
        let mut enabled = true;
        let mut antecedent: Vec<RuleAtom> = Vec::new();
        let mut consequent: Vec<RuleAtom> = Vec::new();
        let mut prefixes: Vec<(String, String)> = Vec::new();

        loop {
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.advance();
                break;
            }
            let key = self.parse_string()?;
            self.expect_byte(b':')?;
            match key.as_str() {
                "name" => name = Some(self.parse_string()?),
                "priority" => priority = self.parse_i32()?,
                "enabled" => enabled = self.parse_bool()?,
                "if" => antecedent = self.parse_atom_array()?,
                "then" => consequent = self.parse_atom_array()?,
                "prefixes" => prefixes = self.parse_prefix_object()?,
                _ => {
                    // Skip unknown fields — consume string or primitive.
                    self.skip_unknown_value()?;
                }
            }
            self.skip_ws();
            if self.peek() == Some(b',') {
                self.advance();
            }
        }

        Ok(SerializableRule {
            name,
            priority,
            enabled,
            antecedent,
            consequent,
            prefixes,
        })
    }

    fn skip_unknown_value(&mut self) -> Result<(), SerializerError> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => {
                self.parse_string()?;
            }
            Some(b'{') => {
                self.advance();
                let mut depth = 1usize;
                while depth > 0 {
                    match self.advance() {
                        Some(b'{') => depth += 1,
                        Some(b'}') => depth -= 1,
                        None => break,
                        _ => {}
                    }
                }
            }
            Some(b'[') => {
                self.advance();
                let mut depth = 1usize;
                while depth > 0 {
                    match self.advance() {
                        Some(b'[') => depth += 1,
                        Some(b']') => depth -= 1,
                        None => break,
                        _ => {}
                    }
                }
            }
            _ => {
                while self.peek().is_some_and(|b| b != b',' && b != b'}') {
                    self.advance();
                }
            }
        }
        Ok(())
    }
}

// ── Batch serialization ───────────────────────────────────────────────────────

/// Serialize a batch of rules into a single N3 string (rules separated by blank lines).
pub fn serialize_batch_n3(rules: &[SerializableRule]) -> Result<String, SerializerError> {
    let mut out = String::new();
    for (i, rule) in rules.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&serialize_n3(rule)?);
        out.push('\n');
    }
    Ok(out)
}

/// Serialize a batch of rules into a JSON array string.
pub fn serialize_batch_json(rules: &[SerializableRule]) -> Result<String, SerializerError> {
    let mut out = String::from("[");
    for (i, rule) in rules.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&serialize_json(rule)?);
    }
    out.push(']');
    Ok(out)
}

/// Deserialize a JSON array of rules.
pub fn deserialize_batch_json(input: &str) -> Result<Vec<SerializableRule>, SerializerError> {
    let input = input.trim();
    if !input.starts_with('[') || !input.ends_with(']') {
        return Err(SerializerError::json_err(1, 0, "expected JSON array"));
    }
    let inner = &input[1..input.len() - 1];
    // Split on top-level commas between objects.
    let mut rules = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, c) in inner.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                if depth == 0 {
                    return Err(SerializerError::json_err(1, i, "unexpected '}'"));
                }
                depth -= 1;
                if depth == 0 {
                    let chunk = &inner[start..=i];
                    rules.push(deserialize_json(chunk.trim())?);
                    start = i + 1;
                }
            }
            ',' if depth == 0 => {
                start = i + 1;
            }
            _ => {}
        }
    }
    Ok(rules)
}

// ── Dispatch function ─────────────────────────────────────────────────────────

/// Serialize a rule to the requested format.
pub fn serialize(rule: &SerializableRule, format: RuleFormat) -> Result<String, SerializerError> {
    match format {
        RuleFormat::N3 => serialize_n3(rule),
        RuleFormat::Json => serialize_json(rule),
    }
}

/// Deserialize a rule from the requested format.
pub fn deserialize(input: &str, format: RuleFormat) -> Result<SerializableRule, SerializerError> {
    match format {
        RuleFormat::N3 => deserialize_n3(input),
        RuleFormat::Json => deserialize_json(input),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rule() -> SerializableRule {
        SerializableRule::new(
            vec![
                RuleAtom::new("?x", "rdf:type", "owl:Animal"),
                RuleAtom::new("?x", "ex:hasLegs", "?n"),
            ],
            vec![RuleAtom::new("?x", "ex:isLandAnimal", "true")],
        )
        .with_name("animal_rule")
        .with_priority(10)
    }

    // ── N3 serialize / deserialize ────────────────────────────────────────────

    #[test]
    fn test_n3_serialize_basic() {
        let rule = sample_rule();
        let n3 = serialize_n3(&rule).expect("serialize");
        assert!(n3.contains("=>"));
        assert!(n3.contains("{"));
        assert!(n3.contains("}"));
    }

    #[test]
    fn test_n3_roundtrip() {
        let original = sample_rule();
        let serialized = serialize_n3(&original).expect("serialize");
        let recovered = deserialize_n3(&serialized).expect("deserialize");
        assert_eq!(recovered.antecedent.len(), original.antecedent.len());
        assert_eq!(recovered.consequent.len(), original.consequent.len());
        assert_eq!(recovered.name, original.name);
        assert_eq!(recovered.priority, original.priority);
    }

    #[test]
    fn test_n3_rule_name_preserved() {
        let rule = SerializableRule::new(
            vec![RuleAtom::new("?x", "a", "?c")],
            vec![RuleAtom::new("?x", "b", "?c")],
        )
        .with_name("my_rule");
        let n3 = serialize_n3(&rule).expect("serialize");
        assert!(n3.contains("# rule: my_rule"));
        let recovered = deserialize_n3(&n3).expect("deserialize");
        assert_eq!(recovered.name.as_deref(), Some("my_rule"));
    }

    #[test]
    fn test_n3_priority_preserved() {
        let rule = SerializableRule::new(
            vec![RuleAtom::new("?x", "a", "b")],
            vec![RuleAtom::new("?x", "c", "d")],
        )
        .with_priority(42);
        let n3 = serialize_n3(&rule).expect("serialize");
        let recovered = deserialize_n3(&n3).expect("deserialize");
        assert_eq!(recovered.priority, 42);
    }

    #[test]
    fn test_n3_enabled_flag_preserved() {
        let rule = SerializableRule::new(
            vec![RuleAtom::new("?x", "a", "b")],
            vec![RuleAtom::new("?x", "c", "d")],
        )
        .disabled();
        let n3 = serialize_n3(&rule).expect("serialize");
        assert!(n3.contains("# enabled: false"));
        let recovered = deserialize_n3(&n3).expect("deserialize");
        assert!(!recovered.enabled);
    }

    #[test]
    fn test_n3_prefix_handling() {
        let rule = SerializableRule::new(
            vec![RuleAtom::new(
                "http://example.org/x",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://example.org/Animal",
            )],
            vec![RuleAtom::new(
                "http://example.org/x",
                "http://example.org/kind",
                "http://example.org/Mammal",
            )],
        )
        .with_prefix("ex", "http://example.org/")
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        let n3 = serialize_n3(&rule).expect("serialize");
        assert!(n3.contains("@prefix ex:"));
        assert!(n3.contains("ex:Animal") || n3.contains("ex:x"));
    }

    #[test]
    fn test_n3_empty_antecedent() {
        let rule = SerializableRule::new(vec![], vec![RuleAtom::new("ex:s", "ex:p", "ex:o")]);
        let n3 = serialize_n3(&rule).expect("serialize");
        assert!(n3.contains("{}"));
        let recovered = deserialize_n3(&n3).expect("deserialize");
        assert_eq!(recovered.antecedent.len(), 0);
        assert_eq!(recovered.consequent.len(), 1);
    }

    #[test]
    fn test_n3_parse_error_missing_arrow() {
        let bad = "{ ?x a ?y } { ?x b ?z } .";
        let result = deserialize_n3(bad);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SerializerError::Parse { .. }));
    }

    // ── JSON serialize / deserialize ──────────────────────────────────────────

    #[test]
    fn test_json_serialize_basic() {
        let rule = sample_rule();
        let json = serialize_json(&rule).expect("serialize");
        assert!(json.contains("\"if\""));
        assert!(json.contains("\"then\""));
        assert!(json.contains("\"priority\""));
        assert!(json.contains("\"enabled\""));
    }

    #[test]
    fn test_json_roundtrip() {
        let original = sample_rule();
        let json = serialize_json(&original).expect("serialize");
        let recovered = deserialize_json(&json).expect("deserialize");
        assert_eq!(recovered.name, original.name);
        assert_eq!(recovered.priority, original.priority);
        assert_eq!(recovered.enabled, original.enabled);
        assert_eq!(recovered.antecedent.len(), original.antecedent.len());
        assert_eq!(recovered.consequent.len(), original.consequent.len());
    }

    #[test]
    fn test_json_atom_content_preserved() {
        let rule = SerializableRule::new(
            vec![RuleAtom::new(
                "http://example.org/Alice",
                "http://schema.org/age",
                "30",
            )],
            vec![RuleAtom::new(
                "http://example.org/Alice",
                "http://example.org/isAdult",
                "true",
            )],
        );
        let json = serialize_json(&rule).expect("serialize");
        let recovered = deserialize_json(&json).expect("deserialize");
        assert_eq!(recovered.antecedent[0].subject, "http://example.org/Alice");
        assert_eq!(recovered.antecedent[0].object, "30");
        assert_eq!(
            recovered.consequent[0].predicate,
            "http://example.org/isAdult"
        );
    }

    #[test]
    fn test_json_disabled_rule() {
        let rule = SerializableRule::new(
            vec![RuleAtom::new("?x", "a", "b")],
            vec![RuleAtom::new("?x", "c", "d")],
        )
        .disabled();
        let json = serialize_json(&rule).expect("serialize");
        assert!(json.contains("\"enabled\":false"));
        let recovered = deserialize_json(&json).expect("deserialize");
        assert!(!recovered.enabled);
    }

    #[test]
    fn test_json_parse_error_bad_input() {
        let bad = "not json at all!!!";
        let result = deserialize_json(bad);
        assert!(result.is_err());
    }

    // ── Format dispatch ───────────────────────────────────────────────────────

    #[test]
    fn test_dispatch_n3() {
        let rule = sample_rule();
        let out = serialize(&rule, RuleFormat::N3).expect("serialize");
        let recovered = deserialize(&out, RuleFormat::N3).expect("deserialize");
        assert_eq!(recovered.antecedent.len(), rule.antecedent.len());
    }

    #[test]
    fn test_dispatch_json() {
        let rule = sample_rule();
        let out = serialize(&rule, RuleFormat::Json).expect("serialize");
        let recovered = deserialize(&out, RuleFormat::Json).expect("deserialize");
        assert_eq!(recovered.consequent.len(), rule.consequent.len());
    }

    // ── Batch serialization ───────────────────────────────────────────────────

    #[test]
    fn test_batch_json_roundtrip() {
        let rules = vec![
            sample_rule(),
            SerializableRule::new(
                vec![RuleAtom::new("?y", "ex:type", "ex:Bird")],
                vec![RuleAtom::new("?y", "ex:canFly", "true")],
            )
            .with_name("bird_rule"),
        ];
        let json = serialize_batch_json(&rules).expect("batch serialize");
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        let recovered = deserialize_batch_json(&json).expect("batch deserialize");
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].name, rules[0].name);
        assert_eq!(recovered[1].name, rules[1].name);
    }

    #[test]
    fn test_batch_n3_multiple_rules() {
        let rules = vec![
            sample_rule(),
            SerializableRule::new(
                vec![RuleAtom::new("?z", "a", "ex:C")],
                vec![RuleAtom::new("?z", "b", "ex:D")],
            ),
        ];
        let n3 = serialize_batch_n3(&rules).expect("batch n3");
        // Both rules use `=>` so we expect at least two occurrences.
        assert_eq!(n3.matches("=>").count(), 2);
    }

    #[test]
    fn test_batch_empty_array() {
        let json = serialize_batch_json(&[]).expect("empty batch");
        assert_eq!(json, "[]");
        let recovered = deserialize_batch_json("[]").expect("empty deserialize");
        assert_eq!(recovered.len(), 0);
    }

    // ── Rule atom helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_rule_atom_matches() {
        let a = RuleAtom::new("s", "p", "o");
        let b = RuleAtom::new("s", "p", "o");
        assert!(a.matches(&b));
    }

    #[test]
    fn test_rule_atom_no_match() {
        let a = RuleAtom::new("s", "p", "o1");
        let b = RuleAtom::new("s", "p", "o2");
        assert!(!a.matches(&b));
    }

    // ── Source position display ───────────────────────────────────────────────

    #[test]
    fn test_source_pos_display() {
        let pos = SourcePos { line: 3, column: 7 };
        assert_eq!(pos.to_string(), "3:7");
    }

    // ── JSON string escaping ──────────────────────────────────────────────────

    #[test]
    fn test_json_string_escape_quotes() {
        let s = json_str("say \"hello\"");
        assert!(s.contains("\\\""));
    }

    #[test]
    fn test_json_string_escape_backslash() {
        let s = json_str("path\\to\\file");
        assert!(s.contains("\\\\"));
    }

    #[test]
    fn test_json_string_escape_newline() {
        let s = json_str("line1\nline2");
        assert!(s.contains("\\n"));
    }

    // ── Prefix expansion ──────────────────────────────────────────────────────

    #[test]
    fn test_apply_prefixes_contraction() {
        let prefixes = vec![("ex".to_owned(), "http://example.org/".to_owned())];
        let contracted = apply_prefixes("http://example.org/Alice", &prefixes);
        assert_eq!(contracted, "ex:Alice");
    }

    #[test]
    fn test_apply_prefixes_no_match() {
        let prefixes = vec![("ex".to_owned(), "http://example.org/".to_owned())];
        let term = "http://other.org/Thing";
        assert_eq!(apply_prefixes(term, &prefixes), term);
    }

    #[test]
    fn test_expand_term_expansion() {
        let prefixes = vec![("ex".to_owned(), "http://example.org/".to_owned())];
        let expanded = expand_term("ex:Alice", &prefixes);
        assert_eq!(expanded, "http://example.org/Alice");
    }

    // ── Negative priority ─────────────────────────────────────────────────────

    #[test]
    fn test_negative_priority_roundtrip() {
        let rule = SerializableRule::new(
            vec![RuleAtom::new("?x", "p", "o")],
            vec![RuleAtom::new("?x", "q", "r")],
        )
        .with_priority(-5);
        let json = serialize_json(&rule).expect("serialize");
        let recovered = deserialize_json(&json).expect("deserialize");
        assert_eq!(recovered.priority, -5);
    }

    // ── Extra coverage ────────────────────────────────────────────────────────

    #[test]
    fn test_serializable_rule_default_enabled_true() {
        let rule = SerializableRule::new(vec![], vec![]);
        assert!(rule.enabled);
    }

    #[test]
    fn test_serializable_rule_default_priority_zero() {
        let rule = SerializableRule::new(vec![], vec![]);
        assert_eq!(rule.priority, 0);
    }

    #[test]
    fn test_with_prefix_builder() {
        let rule = SerializableRule::new(vec![], vec![])
            .with_prefix("ex", "http://example.org/")
            .with_prefix("owl", "http://www.w3.org/2002/07/owl#");
        assert_eq!(rule.prefixes.len(), 2);
        assert_eq!(rule.prefixes[0].0, "ex");
        assert_eq!(rule.prefixes[1].0, "owl");
    }

    #[test]
    fn test_json_multiple_antecedent_atoms() {
        let rule = SerializableRule::new(
            vec![
                RuleAtom::new("?x", "a", "ex:Animal"),
                RuleAtom::new("?x", "ex:hasLegs", "?n"),
                RuleAtom::new("?n", "rdf:type", "xsd:integer"),
            ],
            vec![RuleAtom::new("?x", "ex:isLandAnimal", "true")],
        );
        let json = serialize_json(&rule).expect("serialize");
        let recovered = deserialize_json(&json).expect("deserialize");
        assert_eq!(recovered.antecedent.len(), 3);
    }

    #[test]
    fn test_n3_multi_consequent_atoms() {
        let rule = SerializableRule::new(
            vec![RuleAtom::new("?x", "a", "ex:A")],
            vec![
                RuleAtom::new("?x", "b", "ex:B"),
                RuleAtom::new("?x", "c", "ex:C"),
            ],
        );
        let n3 = serialize_n3(&rule).expect("serialize");
        let recovered = deserialize_n3(&n3).expect("deserialize");
        assert_eq!(recovered.consequent.len(), 2);
    }

    #[test]
    fn test_n3_priority_zero_not_emitted_as_comment() {
        // Priority 0 is the default — we do not emit the comment when it is 0.
        let rule = SerializableRule::new(
            vec![RuleAtom::new("?x", "a", "b")],
            vec![RuleAtom::new("?x", "c", "d")],
        );
        let n3 = serialize_n3(&rule).expect("serialize");
        assert!(
            !n3.contains("# priority:"),
            "zero priority should not be emitted"
        );
    }

    #[test]
    fn test_json_no_name_field_when_absent() {
        let rule = SerializableRule::new(
            vec![RuleAtom::new("?x", "a", "b")],
            vec![RuleAtom::new("?x", "c", "d")],
        );
        let json = serialize_json(&rule).expect("serialize");
        // No "name" key should appear when name is None.
        assert!(!json.contains("\"name\""));
    }

    #[test]
    fn test_json_name_present_when_set() {
        let rule = SerializableRule::new(
            vec![RuleAtom::new("?x", "a", "b")],
            vec![RuleAtom::new("?x", "c", "d")],
        )
        .with_name("my_rule");
        let json = serialize_json(&rule).expect("serialize");
        assert!(json.contains("\"name\""));
        assert!(json.contains("my_rule"));
    }

    #[test]
    fn test_batch_json_single_rule() {
        let rules = vec![sample_rule()];
        let json = serialize_batch_json(&rules).expect("serialize");
        let recovered = deserialize_batch_json(&json).expect("deserialize");
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].name, sample_rule().name);
    }

    #[test]
    fn test_batch_n3_single_rule() {
        let rules = vec![sample_rule()];
        let n3 = serialize_batch_n3(&rules).expect("serialize");
        assert!(n3.contains("=>"));
    }

    #[test]
    fn test_rule_format_n3_variant() {
        let rule = sample_rule();
        let n3_via_dispatch = serialize(&rule, RuleFormat::N3).expect("dispatch n3");
        let n3_direct = serialize_n3(&rule).expect("direct n3");
        assert_eq!(n3_via_dispatch, n3_direct);
    }

    #[test]
    fn test_rule_format_json_variant() {
        let rule = sample_rule();
        let j_via_dispatch = serialize(&rule, RuleFormat::Json).expect("dispatch json");
        let j_direct = serialize_json(&rule).expect("direct json");
        assert_eq!(j_via_dispatch, j_direct);
    }

    #[test]
    fn test_deserialize_dispatch_json() {
        let rule = sample_rule();
        let json = serialize_json(&rule).expect("serialize");
        let recovered = deserialize(&json, RuleFormat::Json).expect("deserialize");
        assert_eq!(recovered.antecedent.len(), rule.antecedent.len());
    }

    #[test]
    fn test_deserialize_dispatch_n3() {
        let rule = sample_rule();
        let n3 = serialize_n3(&rule).expect("serialize");
        let recovered = deserialize(&n3, RuleFormat::N3).expect("deserialize");
        assert_eq!(recovered.consequent.len(), rule.consequent.len());
    }

    #[test]
    fn test_json_empty_prefixes_object() {
        let rule = SerializableRule::new(
            vec![RuleAtom::new("?x", "a", "b")],
            vec![RuleAtom::new("?x", "c", "d")],
        );
        let json = serialize_json(&rule).expect("serialize");
        let recovered = deserialize_json(&json).expect("deserialize");
        assert!(recovered.prefixes.is_empty());
    }

    #[test]
    fn test_json_prefixes_roundtrip() {
        let rule = SerializableRule::new(
            vec![RuleAtom::new("?x", "a", "b")],
            vec![RuleAtom::new("?x", "c", "d")],
        )
        .with_prefix("ex", "http://example.org/")
        .with_prefix("schema", "http://schema.org/");
        let json = serialize_json(&rule).expect("serialize");
        let recovered = deserialize_json(&json).expect("deserialize");
        assert_eq!(recovered.prefixes.len(), 2);
    }
}
