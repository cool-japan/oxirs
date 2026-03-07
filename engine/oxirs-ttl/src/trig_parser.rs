//! TriG named-graph Turtle parser (v1.1.0 round 16).
//!
//! Implements a simple TriG parser that handles:
//! - `@prefix` and `PREFIX` declarations
//! - `GRAPH <iri> { ... }` named-graph blocks
//! - Default graph triples (outside GRAPH blocks)
//! - IRI references `<...>` and prefixed names `prefix:local`
//!
//! Reference: <https://www.w3.org/TR/trig/>

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A single triple situated in a named (or default) graph.
#[derive(Debug, Clone, PartialEq)]
pub struct TrigTriple {
    /// Graph IRI, or `None` for the default graph.
    pub graph: Option<String>,
    /// Subject IRI or blank-node identifier.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object IRI, blank-node identifier, or literal.
    pub object: String,
}

/// The result of parsing a complete TriG document.
#[derive(Debug, Default)]
pub struct TrigDocument {
    /// All triples found in the document.
    pub triples: Vec<TrigTriple>,
    /// Prefix declarations: prefix string (e.g. `"ex"`) → IRI base.
    pub prefixes: HashMap<String, String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Parser internals
// ──────────────────────────────────────────────────────────────────────────────

/// Character-level tokenizer state used during parsing.
struct Tokenizer<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    /// Remaining unparsed input.
    fn rest(&self) -> &str {
        &self.src[self.pos..]
    }

    /// Skip ASCII whitespace.
    fn skip_ws(&mut self) {
        while self.pos < self.src.len()
            && self
                .src
                .as_bytes()
                .get(self.pos)
                .is_some_and(|b| b.is_ascii_whitespace())
        {
            self.pos += 1;
        }
    }

    /// Skip a `# …` comment line.
    fn skip_comment(&mut self) {
        if self.rest().starts_with('#') {
            if let Some(nl) = self.rest().find('\n') {
                self.pos += nl + 1;
            } else {
                self.pos = self.src.len();
            }
        }
    }

    /// Skip whitespace and comments (interleaved).
    fn skip_ws_comments(&mut self) {
        loop {
            let before = self.pos;
            self.skip_ws();
            self.skip_comment();
            if self.pos == before {
                break;
            }
        }
    }

    /// Peek at the next non-whitespace character without advancing.
    #[allow(dead_code)]
    fn peek_char(&mut self) -> Option<char> {
        self.skip_ws_comments();
        self.rest().chars().next()
    }

    /// Consume exactly one byte if it equals `ch`.
    fn consume_char(&mut self, ch: char) -> bool {
        self.skip_ws_comments();
        if self.rest().starts_with(ch) {
            self.pos += ch.len_utf8();
            true
        } else {
            false
        }
    }

    /// Consume a literal keyword (case-insensitive for ASCII).
    #[allow(dead_code)]
    fn consume_keyword(&mut self, kw: &str) -> bool {
        self.skip_ws_comments();
        let r = self.rest();
        if r.len() >= kw.len() && r[..kw.len()].eq_ignore_ascii_case(kw) {
            self.pos += kw.len();
            true
        } else {
            false
        }
    }

    /// Read an IRI reference `<...>`.  Returns the IRI string without angle brackets.
    fn read_iri_ref(&mut self) -> Option<String> {
        self.skip_ws_comments();
        if !self.rest().starts_with('<') {
            return None;
        }
        self.pos += 1; // consume '<'
        let start = self.pos;
        // Find matching '>' — simple scan (no escape handling needed for our tests)
        loop {
            if self.pos >= self.src.len() {
                return None; // unterminated
            }
            let b = self.src.as_bytes()[self.pos];
            if b == b'>' {
                let iri = self.src[start..self.pos].to_string();
                self.pos += 1; // consume '>'
                return Some(iri);
            }
            self.pos += 1;
        }
    }

    /// Read a prefixed name `prefix:local` (or just `prefix:`).
    /// Returns the raw token string; resolution happens in the parser.
    fn read_prefixed_name(&mut self) -> Option<String> {
        self.skip_ws_comments();
        // Collect the rest into owned data to avoid borrow conflicts.
        let rest_owned: String = self.rest().to_string();
        // Must start with a letter or '_'
        let first = rest_owned.chars().next()?;
        if !first.is_ascii_alphabetic() && first != '_' {
            return None;
        }
        // Consume prefix part up to ':'
        let colon_off = rest_owned.find(':')?;
        // Make sure it's not ':' inside '<' or a URI scheme
        let prefix_part = &rest_owned[..colon_off];
        // Validate no spaces in prefix
        if prefix_part.contains(|c: char| c.is_ascii_whitespace()) {
            return None;
        }
        // Clone prefix_part before advancing pos (avoids borrow-after-mutate)
        let prefix_owned = prefix_part.to_string();
        self.pos += colon_off + 1; // consume prefix + ':'
                                   // Consume local name: letters, digits, '.', '-', '_'
        let local_start = self.pos;
        while self.pos < self.src.len() {
            let b = self.src.as_bytes()[self.pos];
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let local = self.src[local_start..self.pos].to_string();
        Some(format!("{}:{}", prefix_owned, local))
    }

    /// Read a string literal `"..."` (double-quoted, no escape handling for simplicity).
    fn read_string_literal(&mut self) -> Option<String> {
        self.skip_ws_comments();
        if !self.rest().starts_with('"') {
            return None;
        }
        self.pos += 1; // consume opening '"'
        let start = self.pos;
        loop {
            if self.pos >= self.src.len() {
                return None;
            }
            let b = self.src.as_bytes()[self.pos];
            if b == b'"' {
                let s = self.src[start..self.pos].to_string();
                self.pos += 1; // consume closing '"'
                               // Optional language tag @lang
                let mut literal = format!("\"{}\"", s);
                if self.rest().starts_with('@') {
                    self.pos += 1;
                    let tag_start = self.pos;
                    while self.pos < self.src.len() {
                        let c = self.src.as_bytes()[self.pos];
                        if c.is_ascii_alphanumeric() || c == b'-' {
                            self.pos += 1;
                        } else {
                            break;
                        }
                    }
                    let tag = &self.src[tag_start..self.pos];
                    literal = format!("\"{}\"@{}", s, tag);
                } else if self.rest().starts_with("^^") {
                    self.pos += 2;
                    let dt = if let Some(iri) = self.read_iri_ref() {
                        format!("<{}>", iri)
                    } else {
                        self.read_prefixed_name().unwrap_or_default()
                    };
                    literal = format!("\"{}\"^^{}", s, dt);
                }
                return Some(literal);
            } else if b == b'\\' {
                self.pos += 2; // skip escape sequence (simplified)
            } else {
                self.pos += 1;
            }
        }
    }

    /// Read a blank node `_:label`.
    fn read_blank_node(&mut self) -> Option<String> {
        self.skip_ws_comments();
        if !self.rest().starts_with("_:") {
            return None;
        }
        self.pos += 2;
        let start = self.pos;
        while self.pos < self.src.len() {
            let b = self.src.as_bytes()[self.pos];
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        Some(format!("_:{}", &self.src[start..self.pos]))
    }

    /// Read any RDF term: IRI ref, prefixed name, string literal, blank node.
    fn read_term(&mut self) -> Option<String> {
        self.skip_ws_comments();
        let rest = self.rest();
        if rest.starts_with('<') {
            self.read_iri_ref().map(|iri| format!("<{}>", iri))
        } else if rest.starts_with("_:") {
            self.read_blank_node()
        } else if rest.starts_with('"') {
            self.read_string_literal()
        } else {
            self.read_prefixed_name()
        }
    }

    /// Check whether we are at EOF (ignoring whitespace/comments).
    fn is_eof(&mut self) -> bool {
        self.skip_ws_comments();
        self.pos >= self.src.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TrigParser
// ──────────────────────────────────────────────────────────────────────────────

/// A simple TriG parser.
///
/// Handles `@prefix`, `PREFIX`, `GRAPH <iri> { ... }` blocks, and bare
/// Turtle-style triples in the default graph.
pub struct TrigParser;

impl TrigParser {
    /// Create a new `TrigParser`.
    pub fn new() -> Self {
        Self
    }

    /// Parse a TriG string into a [`TrigDocument`].
    ///
    /// Returns an error string if the input is syntactically invalid in a way
    /// that prevents further parsing.
    pub fn parse(&self, input: &str) -> Result<TrigDocument, String> {
        let mut doc = TrigDocument::default();
        let mut tok = Tokenizer::new(input);

        while !tok.is_eof() {
            tok.skip_ws_comments();
            if tok.is_eof() {
                break;
            }

            let rest = tok.rest();

            // ── @prefix / PREFIX declaration ─────────────────────────────────
            if rest.starts_with("@prefix")
                || rest[..rest.len().min(6)].eq_ignore_ascii_case("prefix")
            {
                let is_at = rest.starts_with("@prefix");
                if is_at {
                    tok.pos += 7; // "@prefix"
                } else {
                    tok.pos += 6; // "PREFIX"
                }
                tok.skip_ws_comments();

                // Read prefix label (ends at ':')
                let prefix_label = Self::read_prefix_label(&mut tok)?;

                tok.skip_ws_comments();
                if !tok.consume_char(':') {
                    return Err(format!(
                        "Expected ':' after prefix label '{}'",
                        prefix_label
                    ));
                }

                tok.skip_ws_comments();
                let iri = tok
                    .read_iri_ref()
                    .ok_or_else(|| "Expected IRI after prefix declaration".to_string())?;

                if is_at {
                    // Turtle-style requires terminating '.'
                    tok.skip_ws_comments();
                    if !tok.consume_char('.') {
                        return Err("Expected '.' after @prefix declaration".to_string());
                    }
                } else {
                    // SPARQL-style PREFIX — no dot required
                }

                doc.prefixes.insert(prefix_label, iri);
                continue;
            }

            // ── GRAPH block ───────────────────────────────────────────────────
            if rest.len() >= 5 && rest[..5].eq_ignore_ascii_case("graph") {
                tok.pos += 5;
                tok.skip_ws_comments();

                let graph_iri_raw = tok
                    .read_iri_ref()
                    .ok_or_else(|| "Expected IRI after GRAPH keyword".to_string())?;
                let graph_iri = Self::resolve_iri(&graph_iri_raw, &doc.prefixes);

                tok.skip_ws_comments();
                if !tok.consume_char('{') {
                    return Err("Expected '{' after GRAPH IRI".to_string());
                }

                // Parse triples inside the block
                loop {
                    tok.skip_ws_comments();
                    if tok.consume_char('}') {
                        break;
                    }
                    if tok.is_eof() {
                        return Err("Unterminated GRAPH block".to_string());
                    }
                    let triple = Self::parse_triple(&mut tok, &doc.prefixes)?;
                    doc.triples.push(TrigTriple {
                        graph: Some(graph_iri.clone()),
                        subject: triple.0,
                        predicate: triple.1,
                        object: triple.2,
                    });
                }
                continue;
            }

            // ── Default graph triple ─────────────────────────────────────────
            let triple = Self::parse_triple(&mut tok, &doc.prefixes)?;
            doc.triples.push(TrigTriple {
                graph: None,
                subject: triple.0,
                predicate: triple.1,
                object: triple.2,
            });
        }

        Ok(doc)
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Read a prefix label (everything up to the next ':' or whitespace).
    fn read_prefix_label(tok: &mut Tokenizer<'_>) -> Result<String, String> {
        tok.skip_ws_comments();
        let start = tok.pos;
        while tok.pos < tok.src.len() {
            let b = tok.src.as_bytes()[tok.pos];
            if b == b':' || b.is_ascii_whitespace() {
                break;
            }
            tok.pos += 1;
        }
        let label = tok.src[start..tok.pos].to_string();
        Ok(label)
    }

    /// Parse a single `subject predicate object .` triple statement.
    /// Returns `(subject, predicate, object)`.
    fn parse_triple(
        tok: &mut Tokenizer<'_>,
        prefixes: &HashMap<String, String>,
    ) -> Result<(String, String, String), String> {
        let subj_raw = tok.read_term().ok_or_else(|| {
            format!(
                "Expected subject, found: {:?}",
                &tok.rest()[..tok.rest().len().min(20)]
            )
        })?;
        let subj = Self::resolve_term(&subj_raw, prefixes);

        let pred_raw = tok
            .read_term()
            .ok_or_else(|| "Expected predicate".to_string())?;
        let pred = Self::resolve_term(&pred_raw, prefixes);

        let obj_raw = tok
            .read_term()
            .ok_or_else(|| "Expected object".to_string())?;
        let obj = Self::resolve_term(&obj_raw, prefixes);

        tok.skip_ws_comments();
        if !tok.consume_char('.') {
            // Also acceptable: next is '}' (end of graph block), just proceed
        }

        Ok((subj, pred, obj))
    }

    /// Resolve a raw term: expands prefixed names using the prefix map.
    fn resolve_term(raw: &str, prefixes: &HashMap<String, String>) -> String {
        if raw.starts_with('<') {
            // Already a full IRI
            return raw.to_string();
        }
        if raw.starts_with('"') || raw.starts_with("_:") {
            return raw.to_string();
        }
        // Try to resolve as prefixed name
        if let Some(colon) = raw.find(':') {
            let prefix = &raw[..colon];
            let local = &raw[colon + 1..];
            if let Some(base) = prefixes.get(prefix) {
                return format!("<{}{}>", base, local);
            }
        }
        raw.to_string()
    }

    /// Resolve an IRI string (already stripped of angle brackets).
    fn resolve_iri(iri: &str, _prefixes: &HashMap<String, String>) -> String {
        iri.to_string()
    }

    /// Count triples per graph.  Key is `None` for the default graph.
    pub fn graph_sizes(doc: &TrigDocument) -> HashMap<Option<String>, usize> {
        let mut map: HashMap<Option<String>, usize> = HashMap::new();
        for triple in &doc.triples {
            *map.entry(triple.graph.clone()).or_insert(0) += 1;
        }
        map
    }

    /// Return all distinct named-graph IRIs (excluding the default graph).
    pub fn named_graphs(doc: &TrigDocument) -> Vec<String> {
        let mut names: Vec<String> = doc
            .triples
            .iter()
            .filter_map(|t| t.graph.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        names.sort();
        names
    }
}

impl Default for TrigParser {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> TrigParser {
        TrigParser::new()
    }

    // ── @prefix / PREFIX ─────────────────────────────────────────────────────

    #[test]
    fn test_at_prefix_declaration() {
        let input = r#"@prefix ex: <http://example.org/> .
<http://example.org/s> <http://example.org/p> <http://example.org/o> ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(
            doc.prefixes.get("ex"),
            Some(&"http://example.org/".to_string())
        );
        assert_eq!(doc.triples.len(), 1);
    }

    #[test]
    fn test_sparql_prefix_declaration() {
        let input = r#"PREFIX ex: <http://example.org/>
<http://example.org/s> <http://example.org/p> <http://example.org/o> ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(
            doc.prefixes.get("ex"),
            Some(&"http://example.org/".to_string())
        );
    }

    #[test]
    fn test_multiple_prefix_declarations() {
        let input = r#"@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
<http://example.org/s> <http://example.org/p> <http://example.org/o> ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.prefixes.len(), 2);
        assert!(doc.prefixes.contains_key("ex"));
        assert!(doc.prefixes.contains_key("foaf"));
    }

    #[test]
    fn test_empty_prefix() {
        let input = r#"@prefix : <http://default.org/> .
<http://default.org/s> <http://default.org/p> <http://default.org/o> ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(
            doc.prefixes.get(""),
            Some(&"http://default.org/".to_string())
        );
    }

    #[test]
    fn test_prefix_case_insensitive_sparql_style() {
        let input = r#"PREFIX myns: <http://myns.org/>
<http://myns.org/a> <http://myns.org/b> <http://myns.org/c> ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert!(doc.prefixes.contains_key("myns"));
    }

    // ── GRAPH blocks ─────────────────────────────────────────────────────────

    #[test]
    fn test_graph_block_basic() {
        let input = r#"GRAPH <http://example.org/g1> {
    <http://example.org/s> <http://example.org/p> <http://example.org/o> .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 1);
        assert_eq!(
            doc.triples[0].graph,
            Some("http://example.org/g1".to_string())
        );
    }

    #[test]
    fn test_graph_block_multiple_triples() {
        let input = r#"GRAPH <http://example.org/g1> {
    <http://example.org/s1> <http://example.org/p> <http://example.org/o1> .
    <http://example.org/s2> <http://example.org/p> <http://example.org/o2> .
    <http://example.org/s3> <http://example.org/p> <http://example.org/o3> .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 3);
        for t in &doc.triples {
            assert_eq!(t.graph, Some("http://example.org/g1".to_string()));
        }
    }

    #[test]
    fn test_multiple_graph_blocks() {
        let input = r#"GRAPH <http://example.org/g1> {
    <http://example.org/s1> <http://example.org/p> <http://example.org/o1> .
}
GRAPH <http://example.org/g2> {
    <http://example.org/s2> <http://example.org/p> <http://example.org/o2> .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 2);
        let graphs: Vec<_> = doc.triples.iter().map(|t| t.graph.as_deref()).collect();
        assert!(graphs.contains(&Some("http://example.org/g1")));
        assert!(graphs.contains(&Some("http://example.org/g2")));
    }

    #[test]
    fn test_graph_keyword_case_insensitive() {
        let input = r#"graph <http://example.org/g1> {
    <http://example.org/s> <http://example.org/p> <http://example.org/o> .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 1);
        assert_eq!(
            doc.triples[0].graph,
            Some("http://example.org/g1".to_string())
        );
    }

    // ── Default graph triples ─────────────────────────────────────────────────

    #[test]
    fn test_default_graph_triple() {
        let input = r#"<http://example.org/s> <http://example.org/p> <http://example.org/o> ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 1);
        assert_eq!(doc.triples[0].graph, None);
    }

    #[test]
    fn test_default_graph_multiple_triples() {
        let input = r#"<http://a.org/s1> <http://a.org/p> <http://a.org/o1> .
<http://a.org/s2> <http://a.org/p> <http://a.org/o2> ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 2);
        assert!(doc.triples.iter().all(|t| t.graph.is_none()));
    }

    #[test]
    fn test_default_and_named_graph_mixed() {
        let input = r#"<http://a.org/s1> <http://a.org/p> <http://a.org/o1> .
GRAPH <http://a.org/g1> {
    <http://a.org/s2> <http://a.org/p> <http://a.org/o2> .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 2);
        let default_count = doc.triples.iter().filter(|t| t.graph.is_none()).count();
        let named_count = doc.triples.iter().filter(|t| t.graph.is_some()).count();
        assert_eq!(default_count, 1);
        assert_eq!(named_count, 1);
    }

    // ── Prefix expansion ─────────────────────────────────────────────────────

    #[test]
    fn test_prefix_expansion_in_triple() {
        let input = r#"@prefix ex: <http://example.org/> .
ex:subject ex:predicate ex:object ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 1);
        assert_eq!(doc.triples[0].subject, "<http://example.org/subject>");
        assert_eq!(doc.triples[0].predicate, "<http://example.org/predicate>");
        assert_eq!(doc.triples[0].object, "<http://example.org/object>");
    }

    #[test]
    fn test_prefix_expansion_in_graph_block() {
        let input = r#"@prefix ex: <http://example.org/> .
GRAPH <http://example.org/g1> {
    ex:subject ex:predicate ex:object .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 1);
        assert_eq!(doc.triples[0].subject, "<http://example.org/subject>");
    }

    // ── String literals ───────────────────────────────────────────────────────

    #[test]
    fn test_string_literal_object() {
        let input = r#"<http://a.org/s> <http://a.org/name> "Alice" ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples[0].object, "\"Alice\"");
    }

    #[test]
    fn test_string_literal_with_lang_tag() {
        let input = r#"<http://a.org/s> <http://a.org/name> "Hello"@en ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples[0].object, "\"Hello\"@en");
    }

    #[test]
    fn test_string_literal_with_datatype() {
        let input = r#"<http://a.org/s> <http://a.org/age> "42"^^<http://www.w3.org/2001/XMLSchema#integer> ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert!(doc.triples[0].object.contains("42"));
    }

    // ── Blank nodes ───────────────────────────────────────────────────────────

    #[test]
    fn test_blank_node_subject() {
        let input = r#"_:b0 <http://a.org/p> <http://a.org/o> ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples[0].subject, "_:b0");
    }

    #[test]
    fn test_blank_node_object() {
        let input = r#"<http://a.org/s> <http://a.org/p> _:b1 ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples[0].object, "_:b1");
    }

    // ── graph_sizes ───────────────────────────────────────────────────────────

    #[test]
    fn test_graph_sizes_empty_doc() {
        let doc = TrigDocument::default();
        let sizes = TrigParser::graph_sizes(&doc);
        assert!(sizes.is_empty());
    }

    #[test]
    fn test_graph_sizes_default_only() {
        let input = r#"<http://a.org/s> <http://a.org/p> <http://a.org/o> ."#;
        let doc = parser().parse(input).expect("parse ok");
        let sizes = TrigParser::graph_sizes(&doc);
        assert_eq!(sizes.get(&None), Some(&1));
    }

    #[test]
    fn test_graph_sizes_named_only() {
        let input = r#"GRAPH <http://example.org/g1> {
    <http://a.org/s1> <http://a.org/p> <http://a.org/o1> .
    <http://a.org/s2> <http://a.org/p> <http://a.org/o2> .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        let sizes = TrigParser::graph_sizes(&doc);
        assert_eq!(
            sizes.get(&Some("http://example.org/g1".to_string())),
            Some(&2)
        );
        assert_eq!(sizes.get(&None), None);
    }

    #[test]
    fn test_graph_sizes_mixed() {
        let input = r#"<http://a.org/s0> <http://a.org/p> <http://a.org/o0> .
GRAPH <http://example.org/g1> {
    <http://a.org/s1> <http://a.org/p> <http://a.org/o1> .
    <http://a.org/s2> <http://a.org/p> <http://a.org/o2> .
}
GRAPH <http://example.org/g2> {
    <http://a.org/s3> <http://a.org/p> <http://a.org/o3> .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        let sizes = TrigParser::graph_sizes(&doc);
        assert_eq!(sizes.get(&None), Some(&1));
        assert_eq!(
            sizes.get(&Some("http://example.org/g1".to_string())),
            Some(&2)
        );
        assert_eq!(
            sizes.get(&Some("http://example.org/g2".to_string())),
            Some(&1)
        );
    }

    #[test]
    fn test_graph_sizes_multiple_triples_same_graph() {
        let input = r#"GRAPH <http://example.org/g1> {
    <http://a.org/s1> <http://a.org/p> <http://a.org/o1> .
}
GRAPH <http://example.org/g1> {
    <http://a.org/s2> <http://a.org/p> <http://a.org/o2> .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        let sizes = TrigParser::graph_sizes(&doc);
        assert_eq!(
            sizes.get(&Some("http://example.org/g1".to_string())),
            Some(&2)
        );
    }

    // ── named_graphs ─────────────────────────────────────────────────────────

    #[test]
    fn test_named_graphs_empty() {
        let doc = TrigDocument::default();
        assert!(TrigParser::named_graphs(&doc).is_empty());
    }

    #[test]
    fn test_named_graphs_default_only() {
        let input = r#"<http://a.org/s> <http://a.org/p> <http://a.org/o> ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert!(TrigParser::named_graphs(&doc).is_empty());
    }

    #[test]
    fn test_named_graphs_returns_distinct() {
        let input = r#"GRAPH <http://example.org/g1> {
    <http://a.org/s1> <http://a.org/p> <http://a.org/o1> .
    <http://a.org/s2> <http://a.org/p> <http://a.org/o2> .
}
GRAPH <http://example.org/g2> {
    <http://a.org/s3> <http://a.org/p> <http://a.org/o3> .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        let mut graphs = TrigParser::named_graphs(&doc);
        graphs.sort();
        assert_eq!(
            graphs,
            vec!["http://example.org/g1", "http://example.org/g2"]
        );
    }

    #[test]
    fn test_named_graphs_deduplicates() {
        let input = r#"GRAPH <http://example.org/g1> {
    <http://a.org/s1> <http://a.org/p> <http://a.org/o1> .
}
GRAPH <http://example.org/g1> {
    <http://a.org/s2> <http://a.org/p> <http://a.org/o2> .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        let graphs = TrigParser::named_graphs(&doc);
        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0], "http://example.org/g1");
    }

    // ── Empty input ───────────────────────────────────────────────────────────

    #[test]
    fn test_empty_input() {
        let doc = parser().parse("").expect("parse ok");
        assert!(doc.triples.is_empty());
        assert!(doc.prefixes.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let doc = parser().parse("   \n\t  ").expect("parse ok");
        assert!(doc.triples.is_empty());
    }

    #[test]
    fn test_comments_only() {
        let doc = parser()
            .parse("# This is a comment\n# Another comment\n")
            .expect("parse ok");
        assert!(doc.triples.is_empty());
    }

    #[test]
    fn test_prefix_only_no_triples() {
        let input = "@prefix ex: <http://example.org/> .";
        let doc = parser().parse(input).expect("parse ok");
        assert!(doc.triples.is_empty());
        assert_eq!(doc.prefixes.len(), 1);
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_error_unterminated_graph_block() {
        let input = r#"GRAPH <http://example.org/g1> {
    <http://a.org/s> <http://a.org/p> <http://a.org/o> ."#;
        // Missing closing '}'
        let result = parser().parse(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_graph_keyword_without_iri() {
        let input = "GRAPH { <http://a.org/s> <http://a.org/p> <http://a.org/o> . }";
        let result = parser().parse(input);
        assert!(result.is_err());
    }

    // ── TrigDocument default ─────────────────────────────────────────────────

    #[test]
    fn test_trig_document_default() {
        let doc = TrigDocument::default();
        assert!(doc.triples.is_empty());
        assert!(doc.prefixes.is_empty());
    }

    // ── TrigTriple PartialEq ─────────────────────────────────────────────────

    #[test]
    fn test_trig_triple_equality() {
        let t1 = TrigTriple {
            graph: Some("http://g.org/g1".to_string()),
            subject: "<http://s.org/s>".to_string(),
            predicate: "<http://p.org/p>".to_string(),
            object: "<http://o.org/o>".to_string(),
        };
        let t2 = t1.clone();
        assert_eq!(t1, t2);
    }

    // ── TrigParser::new / default ─────────────────────────────────────────────

    #[test]
    fn test_parser_new_and_default() {
        let _p1 = TrigParser::new();
        let _p2 = TrigParser;
    }

    // ── Comment handling ──────────────────────────────────────────────────────

    #[test]
    fn test_comment_before_triple() {
        let input = r#"# This is a comment
<http://a.org/s> <http://a.org/p> <http://a.org/o> ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 1);
    }

    #[test]
    fn test_comment_between_declarations_and_triples() {
        let input = r#"@prefix ex: <http://example.org/> .
# This is a comment
ex:s ex:p ex:o ."#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 1);
        assert!(doc.prefixes.contains_key("ex"));
    }

    // ── Round-trip: prefixes in graph blocks ──────────────────────────────────

    #[test]
    fn test_prefix_and_graph_combined() {
        let input = r#"@prefix ex: <http://example.org/> .
GRAPH <http://example.org/g1> {
    ex:Alice ex:knows ex:Bob .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 1);
        assert_eq!(doc.triples[0].subject, "<http://example.org/Alice>");
        assert_eq!(doc.triples[0].predicate, "<http://example.org/knows>");
        assert_eq!(doc.triples[0].object, "<http://example.org/Bob>");
    }

    #[test]
    fn test_multiple_subjects_in_graph() {
        let input = r#"GRAPH <http://example.org/g1> {
    <http://a.org/a> <http://a.org/p> "hello" .
    <http://a.org/b> <http://a.org/p> "world" .
    <http://a.org/c> <http://a.org/p> "!" .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 3);
    }

    #[test]
    fn test_full_document() {
        let input = r#"@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

<http://example.org/default1> <http://example.org/type> <http://example.org/Thing> .

GRAPH <http://example.org/g1> {
    ex:Alice foaf:name "Alice" .
    ex:Alice foaf:knows ex:Bob .
}

GRAPH <http://example.org/g2> {
    ex:Bob foaf:name "Bob" .
}"#;
        let doc = parser().parse(input).expect("parse ok");
        assert_eq!(doc.triples.len(), 4);
        assert_eq!(doc.prefixes.len(), 2);
        let sizes = TrigParser::graph_sizes(&doc);
        assert_eq!(sizes.get(&None), Some(&1));
        assert_eq!(
            sizes.get(&Some("http://example.org/g1".to_string())),
            Some(&2)
        );
        assert_eq!(
            sizes.get(&Some("http://example.org/g2".to_string())),
            Some(&1)
        );
        let mut graphs = TrigParser::named_graphs(&doc);
        graphs.sort();
        assert_eq!(
            graphs,
            vec!["http://example.org/g1", "http://example.org/g2"]
        );
    }
}
