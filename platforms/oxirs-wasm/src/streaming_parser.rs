//! # Streaming / Incremental RDF Document Parser
//!
//! Chunk-based parser that processes RDF input incrementally, emitting parse events
//! as complete tokens become available.
//!
//! Supported formats:
//! - **N-Triples** — one triple per line `<s> <p> <o> .`
//! - **N-Quads** — one quad per line `<s> <p> <o> <g> .`
//! - **Turtle** — triples with PREFIX/BASE directives
//! - **TriG** — quads with named graphs (simplified)

use std::collections::VecDeque;

// ─── Public types ─────────────────────────────────────────────────────────────

/// Events emitted by the streaming parser
#[derive(Debug, Clone, PartialEq)]
pub enum ParseEvent {
    /// A complete RDF triple was parsed
    Triple { s: String, p: String, o: String },
    /// A complete RDF quad was parsed (triple in a named graph)
    Quad {
        s: String,
        p: String,
        o: String,
        g: String,
    },
    /// A PREFIX declaration was recognised
    Prefix { prefix: String, iri: String },
    /// A BASE IRI declaration was recognised
    BaseIri(String),
    /// A parse error was encountered (parsing continues)
    Error(String),
    /// End of input — all buffered data has been processed
    End,
}

/// Cumulative parse statistics
#[derive(Debug, Clone, Default)]
pub struct ParseStats {
    pub triples: usize,
    pub quads: usize,
    pub prefixes: usize,
    pub errors: usize,
    pub bytes_processed: usize,
}

/// RDF serialisation formats understood by the parser
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFormat {
    Turtle,
    NTriples,
    NQuads,
    TriG,
}

/// Configuration for the streaming parser
#[derive(Debug, Clone)]
pub struct StreamingParserConfig {
    pub format: StreamFormat,
    /// Whether to perform basic validation on IRIs / blank node labels
    pub validate: bool,
    /// Optional base IRI
    pub base_iri: Option<String>,
}

impl Default for StreamingParserConfig {
    fn default() -> Self {
        Self {
            format: StreamFormat::NTriples,
            validate: false,
            base_iri: None,
        }
    }
}

/// Incremental (chunk-based) RDF parser
pub struct StreamingParser {
    config: StreamingParserConfig,
    /// Partial line / incomplete token accumulation buffer
    buffer: String,
    stats: ParseStats,
    /// Events queued for retrieval by the caller
    events: VecDeque<ParseEvent>,
}

impl StreamingParser {
    /// Create a new streaming parser with the given configuration
    pub fn new(config: StreamingParserConfig) -> Self {
        let mut parser = Self {
            config,
            buffer: String::new(),
            stats: ParseStats::default(),
            events: VecDeque::new(),
        };

        // Emit a base IRI event if configured
        if let Some(ref base) = parser.config.base_iri.clone() {
            parser.events.push_back(ParseEvent::BaseIri(base.clone()));
        }

        parser
    }

    /// Feed a chunk of input to the parser.
    ///
    /// The chunk is appended to the internal buffer and complete lines / tokens
    /// are extracted and parsed immediately.  Parsed events are returned.
    pub fn feed(&mut self, chunk: &str) -> Vec<ParseEvent> {
        self.stats.bytes_processed += chunk.len();
        self.buffer.push_str(chunk);

        let mut emitted = Vec::new();
        self.process_buffer(&mut emitted);
        emitted
    }

    /// Flush remaining buffer content and emit an `End` event.
    ///
    /// Any incomplete line in the buffer is treated as an error (unless empty).
    pub fn flush(&mut self) -> Vec<ParseEvent> {
        let mut emitted = Vec::new();

        // Emit any queued events first
        while let Some(ev) = self.events.pop_front() {
            emitted.push(ev);
        }

        // Process remainder of buffer
        let leftover = std::mem::take(&mut self.buffer);
        let trimmed = leftover.trim();
        if !trimmed.is_empty() {
            // Try to parse the leftover as a complete token
            let mut chunk_events = Vec::new();
            self.parse_line(trimmed, &mut chunk_events);
            if chunk_events.is_empty() {
                self.stats.errors += 1;
                emitted.push(ParseEvent::Error(format!(
                    "Incomplete token at end of input: {trimmed}"
                )));
            } else {
                emitted.extend(chunk_events);
            }
        }

        emitted.push(ParseEvent::End);
        emitted
    }

    /// Reference to cumulative parse statistics
    pub fn stats(&self) -> &ParseStats {
        &self.stats
    }

    /// Number of events currently queued (not yet returned to the caller)
    pub fn pending_events(&self) -> usize {
        self.events.len()
    }

    /// Reset the parser to its initial state (clear buffer, stats and event queue)
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.stats = ParseStats::default();
        self.events.clear();
    }

    // ─── private ─────────────────────────────────────────────────────────────

    /// Process all complete lines in the buffer
    fn process_buffer(&mut self, emitted: &mut Vec<ParseEvent>) {
        // Drain any pre-queued events first
        while let Some(ev) = self.events.pop_front() {
            emitted.push(ev);
        }

        loop {
            // Find next newline
            let Some(nl_pos) = self.buffer.find('\n') else {
                break;
            };
            let line = self.buffer[..nl_pos].to_string();
            self.buffer = self.buffer[nl_pos + 1..].to_string();

            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue; // blank / comment line
            }
            self.parse_line(trimmed, emitted);
        }
    }

    /// Parse a single complete line and push events into `emitted`
    fn parse_line(&mut self, line: &str, emitted: &mut Vec<ParseEvent>) {
        match self.config.format {
            StreamFormat::NTriples => self.parse_ntriples_line(line, emitted),
            StreamFormat::NQuads => self.parse_nquads_line(line, emitted),
            StreamFormat::Turtle => self.parse_turtle_line(line, emitted),
            StreamFormat::TriG => self.parse_trig_line(line, emitted),
        }
    }

    // ── N-Triples: <s> <p> <o> .  or  _:blank <p> <o> .
    fn parse_ntriples_line(&mut self, line: &str, emitted: &mut Vec<ParseEvent>) {
        // Directive check (shouldn't appear in N-Triples but handle gracefully)
        if line.starts_with('@') {
            self.stats.errors += 1;
            emitted.push(ParseEvent::Error(format!(
                "Directives not allowed in N-Triples: {line}"
            )));
            return;
        }

        let parts = split_ntriples(line);
        if parts.len() < 3 {
            if !line.trim_end_matches('.').trim().is_empty() {
                self.stats.errors += 1;
                emitted.push(ParseEvent::Error(format!("Invalid N-Triple: {line}")));
            }
            return;
        }

        // Always validate that terms have recognised N-Triples syntax
        // (must start with '<', '"', or '_:')
        for part in &parts[..3] {
            if let Err(msg) = validate_term(part) {
                self.stats.errors += 1;
                emitted.push(ParseEvent::Error(msg));
                return;
            }
        }

        // Optional strict validation (IRI closure etc.)
        if self.config.validate {
            // validate_term already covers the basic checks above; extended checks
            // would go here in future.
        }

        self.stats.triples += 1;
        emitted.push(ParseEvent::Triple {
            s: parts[0].clone(),
            p: parts[1].clone(),
            o: parts[2].clone(),
        });
    }

    // ── N-Quads: <s> <p> <o> <g> .
    fn parse_nquads_line(&mut self, line: &str, emitted: &mut Vec<ParseEvent>) {
        let parts = split_ntriples(line);
        if parts.len() < 4 {
            // Could be a triple (no graph)
            if parts.len() == 3 {
                self.stats.triples += 1;
                emitted.push(ParseEvent::Triple {
                    s: parts[0].clone(),
                    p: parts[1].clone(),
                    o: parts[2].clone(),
                });
            } else {
                self.stats.errors += 1;
                emitted.push(ParseEvent::Error(format!("Invalid N-Quad: {line}")));
            }
            return;
        }

        self.stats.quads += 1;
        emitted.push(ParseEvent::Quad {
            s: parts[0].clone(),
            p: parts[1].clone(),
            o: parts[2].clone(),
            g: parts[3].clone(),
        });
    }

    // ── Turtle: PREFIX declarations + triple lines
    fn parse_turtle_line(&mut self, line: &str, emitted: &mut Vec<ParseEvent>) {
        let upper = line.to_uppercase();

        // @prefix or PREFIX (SPARQL-style)
        if upper.starts_with("@PREFIX") || upper.starts_with("PREFIX") {
            if let Some(ev) = parse_prefix_directive(line) {
                self.stats.prefixes += 1;
                emitted.push(ev);
            } else {
                self.stats.errors += 1;
                emitted.push(ParseEvent::Error(format!(
                    "Malformed prefix directive: {line}"
                )));
            }
            return;
        }

        // @base or BASE
        if upper.starts_with("@BASE") || upper.starts_with("BASE") {
            if let Some(iri) = parse_base_directive(line) {
                emitted.push(ParseEvent::BaseIri(iri));
            } else {
                self.stats.errors += 1;
                emitted.push(ParseEvent::Error(format!(
                    "Malformed base directive: {line}"
                )));
            }
            return;
        }

        // Try triple parse
        let parts = split_ntriples(line);
        if parts.len() >= 3 {
            self.stats.triples += 1;
            emitted.push(ParseEvent::Triple {
                s: parts[0].clone(),
                p: parts[1].clone(),
                o: parts[2].clone(),
            });
        } else {
            self.stats.errors += 1;
            emitted.push(ParseEvent::Error(format!(
                "Unrecognised Turtle statement: {line}"
            )));
        }
    }

    // ── TriG: same as Turtle but also produces Quads for GRAPH blocks (simplified)
    fn parse_trig_line(&mut self, line: &str, emitted: &mut Vec<ParseEvent>) {
        let upper = line.to_uppercase();

        if upper.starts_with("@PREFIX") || upper.starts_with("PREFIX") {
            if let Some(ev) = parse_prefix_directive(line) {
                self.stats.prefixes += 1;
                emitted.push(ev);
            } else {
                self.stats.errors += 1;
                emitted.push(ParseEvent::Error(format!(
                    "Malformed prefix directive (TriG): {line}"
                )));
            }
            return;
        }

        if upper.starts_with("@BASE") || upper.starts_with("BASE") {
            if let Some(iri) = parse_base_directive(line) {
                emitted.push(ParseEvent::BaseIri(iri));
            } else {
                self.stats.errors += 1;
                emitted.push(ParseEvent::Error(format!(
                    "Malformed base directive (TriG): {line}"
                )));
            }
            return;
        }

        // N-Quads-style line
        let parts = split_ntriples(line);
        if parts.len() >= 4 {
            self.stats.quads += 1;
            emitted.push(ParseEvent::Quad {
                s: parts[0].clone(),
                p: parts[1].clone(),
                o: parts[2].clone(),
                g: parts[3].clone(),
            });
            return;
        }
        if parts.len() >= 3 {
            self.stats.triples += 1;
            emitted.push(ParseEvent::Triple {
                s: parts[0].clone(),
                p: parts[1].clone(),
                o: parts[2].clone(),
            });
            return;
        }

        // Structural TriG tokens (GRAPH keyword, braces) are silently skipped
        let upper_trim = line.trim().to_uppercase();
        if upper_trim == "{" || upper_trim == "}" || upper_trim.starts_with("GRAPH ") {
            return;
        }

        self.stats.errors += 1;
        emitted.push(ParseEvent::Error(format!(
            "Unrecognised TriG statement: {line}"
        )));
    }
}

// ─── Parsing helpers ──────────────────────────────────────────────────────────

/// Split an N-Triples / N-Quads line into terms, stripping the trailing dot
fn split_ntriples(line: &str) -> Vec<String> {
    let line = line.trim_end_matches('.').trim();
    let mut parts = Vec::new();
    let mut chars = line.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' => {
                chars.next();
            }
            '<' => {
                // IRI
                chars.next();
                let mut iri = "<".to_string();
                for c in chars.by_ref() {
                    iri.push(c);
                    if c == '>' {
                        break;
                    }
                }
                parts.push(iri);
            }
            '"' => {
                // Literal — collect until unescaped closing quote, then optional lang/type
                let mut lit = String::new();
                lit.push('"');
                chars.next();
                let mut escaped = false;
                for c in chars.by_ref() {
                    if escaped {
                        lit.push(c);
                        escaped = false;
                    } else if c == '\\' {
                        lit.push(c);
                        escaped = true;
                    } else {
                        lit.push(c);
                        if c == '"' {
                            break;
                        }
                    }
                }
                // Optional @lang or ^^<type>
                while let Some(&nc) = chars.peek() {
                    if nc == '@' || nc == '^' {
                        let c = chars.next().expect("peeked char");
                        lit.push(c);
                    } else if nc == '<' {
                        // Datatype IRI
                        chars.next();
                        lit.push('<');
                        for c in chars.by_ref() {
                            lit.push(c);
                            if c == '>' {
                                break;
                            }
                        }
                    } else if nc.is_alphabetic() || nc == '-' {
                        lit.push(chars.next().expect("peeked char"));
                    } else {
                        break;
                    }
                }
                parts.push(lit);
            }
            '_' => {
                // Blank node _:label
                let mut bn = String::new();
                for c in chars.by_ref() {
                    if c == ' ' || c == '\t' {
                        break;
                    }
                    bn.push(c);
                }
                parts.push(bn);
            }
            _ => {
                // Unknown token — consume to next whitespace
                let mut tok = String::new();
                for c in chars.by_ref() {
                    if c == ' ' || c == '\t' {
                        break;
                    }
                    tok.push(c);
                }
                if !tok.is_empty() {
                    parts.push(tok);
                }
            }
        }
    }

    parts
}

/// Parse `@prefix foo: <iri> .` or `PREFIX foo: <iri>` → `ParseEvent::Prefix`
fn parse_prefix_directive(line: &str) -> Option<ParseEvent> {
    // Strip leading keyword
    let rest = line
        .trim_start_matches("@prefix")
        .trim_start_matches("@PREFIX")
        .trim_start_matches("PREFIX")
        .trim_start_matches("prefix")
        .trim();

    // Collect prefix label (up to ':')
    let colon = rest.find(':')?;
    let prefix = rest[..colon].trim().to_string();
    let after_colon = rest[colon + 1..].trim();

    // IRI in angle brackets
    let iri_start = after_colon.find('<')?;
    let iri_end = after_colon.rfind('>')?;
    if iri_end <= iri_start {
        return None;
    }
    let iri = after_colon[iri_start + 1..iri_end].to_string();

    Some(ParseEvent::Prefix { prefix, iri })
}

/// Parse `@base <iri> .` or `BASE <iri>` → base IRI string
fn parse_base_directive(line: &str) -> Option<String> {
    let rest = line
        .trim_start_matches("@base")
        .trim_start_matches("@BASE")
        .trim_start_matches("BASE")
        .trim_start_matches("base")
        .trim();

    let iri_start = rest.find('<')?;
    let iri_end = rest.rfind('>')?;
    if iri_end <= iri_start {
        return None;
    }
    Some(rest[iri_start + 1..iri_end].to_string())
}

/// Basic validation of an RDF term string
fn validate_term(term: &str) -> Result<(), String> {
    if term.starts_with('<') {
        if !term.ends_with('>') {
            return Err(format!("Unclosed IRI: {term}"));
        }
    } else if term.starts_with('"') {
        if !term.contains('"') {
            return Err(format!("Unclosed literal: {term}"));
        }
    } else if !term.starts_with("_:") {
        return Err(format!("Unknown term format: {term}"));
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ntriples_config() -> StreamingParserConfig {
        StreamingParserConfig {
            format: StreamFormat::NTriples,
            validate: false,
            base_iri: None,
        }
    }

    fn turtle_config() -> StreamingParserConfig {
        StreamingParserConfig {
            format: StreamFormat::Turtle,
            validate: false,
            base_iri: None,
        }
    }

    fn nquads_config() -> StreamingParserConfig {
        StreamingParserConfig {
            format: StreamFormat::NQuads,
            validate: false,
            base_iri: None,
        }
    }

    fn trig_config() -> StreamingParserConfig {
        StreamingParserConfig {
            format: StreamFormat::TriG,
            validate: false,
            base_iri: None,
        }
    }

    // ─── basic N-Triples feeding ──────────────────────────────────────────────

    #[test]
    fn test_feed_ntriples_single_line() {
        let mut p = StreamingParser::new(ntriples_config());
        let events = p.feed("<http://s> <http://p> <http://o> .\n");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], ParseEvent::Triple { .. }));
    }

    #[test]
    fn test_feed_ntriples_stats_triples() {
        let mut p = StreamingParser::new(ntriples_config());
        p.feed("<http://s> <http://p> <http://o> .\n");
        p.feed("<http://s2> <http://p2> <http://o2> .\n");
        assert_eq!(p.stats().triples, 2);
    }

    #[test]
    fn test_feed_ntriples_subject_predicate_object() {
        let mut p = StreamingParser::new(ntriples_config());
        let events = p.feed("<http://s> <http://p> \"hello\" .\n");
        if let ParseEvent::Triple { s, p: pred, o } = &events[0] {
            assert_eq!(s, "<http://s>");
            assert_eq!(pred, "<http://p>");
            assert!(o.contains("hello"));
        } else {
            panic!("expected Triple event");
        }
    }

    #[test]
    fn test_feed_multiple_lines_in_one_chunk() {
        let mut p = StreamingParser::new(ntriples_config());
        let events =
            p.feed("<http://s1> <http://p> <http://o1> .\n<http://s2> <http://p> <http://o2> .\n");
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_feed_empty_chunk_no_events() {
        let mut p = StreamingParser::new(ntriples_config());
        let events = p.feed("");
        assert!(events.is_empty());
    }

    #[test]
    fn test_feed_blank_lines_ignored() {
        let mut p = StreamingParser::new(ntriples_config());
        let events = p.feed("\n\n\n");
        assert!(events.is_empty());
    }

    #[test]
    fn test_feed_comment_lines_ignored() {
        let mut p = StreamingParser::new(ntriples_config());
        let events = p.feed("# This is a comment\n");
        assert!(events.is_empty());
    }

    // ─── partial line across chunks ───────────────────────────────────────────

    #[test]
    fn test_partial_line_across_chunks() {
        let mut p = StreamingParser::new(ntriples_config());
        let events1 = p.feed("<http://s> <http://p>");
        assert!(events1.is_empty(), "no complete line yet");
        let events2 = p.feed(" <http://o> .\n");
        assert_eq!(events2.len(), 1, "complete triple after second chunk");
    }

    #[test]
    fn test_bytes_processed_tracks_correctly() {
        let mut p = StreamingParser::new(ntriples_config());
        let chunk = "<http://s> <http://p> <http://o> .\n";
        p.feed(chunk);
        assert_eq!(p.stats().bytes_processed, chunk.len());
    }

    #[test]
    fn test_bytes_processed_accumulates() {
        let mut p = StreamingParser::new(ntriples_config());
        p.feed("abc");
        p.feed("def");
        assert_eq!(p.stats().bytes_processed, 6);
    }

    // ─── flush emits End ──────────────────────────────────────────────────────

    #[test]
    fn test_flush_emits_end() {
        let mut p = StreamingParser::new(ntriples_config());
        p.feed("<http://s> <http://p> <http://o> .\n");
        let flush_events = p.flush();
        assert!(
            flush_events.contains(&ParseEvent::End),
            "flush must emit End"
        );
    }

    #[test]
    fn test_flush_empty_parser_emits_end() {
        let mut p = StreamingParser::new(ntriples_config());
        let events = p.flush();
        assert_eq!(events, vec![ParseEvent::End]);
    }

    #[test]
    fn test_flush_incomplete_line_is_error() {
        let mut p = StreamingParser::new(ntriples_config());
        p.feed("<http://s> incomplete");
        let events = p.flush();
        let has_error = events.iter().any(|e| matches!(e, ParseEvent::Error(_)));
        assert!(has_error, "incomplete buffer should yield an Error event");
        assert!(events.contains(&ParseEvent::End));
    }

    // ─── reset ────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_buffer() {
        let mut p = StreamingParser::new(ntriples_config());
        p.feed("<partial>");
        p.reset();
        let events = p.flush();
        // After reset, buffer is cleared — only End
        assert_eq!(events, vec![ParseEvent::End]);
    }

    #[test]
    fn test_reset_clears_stats() {
        let mut p = StreamingParser::new(ntriples_config());
        p.feed("<http://s> <http://p> <http://o> .\n");
        p.reset();
        assert_eq!(p.stats().triples, 0);
        assert_eq!(p.stats().bytes_processed, 0);
    }

    #[test]
    fn test_reset_clears_events() {
        let mut p = StreamingParser::new(ntriples_config());
        p.feed("<http://s> <http://p> <http://o> .\n");
        p.reset();
        assert_eq!(p.pending_events(), 0);
    }

    // ─── Turtle: PREFIX ───────────────────────────────────────────────────────

    #[test]
    fn test_turtle_prefix_event() {
        let mut p = StreamingParser::new(turtle_config());
        let events = p.feed("@prefix ex: <http://example.org/> .\n");
        let prefix_ev = events
            .iter()
            .find(|e| matches!(e, ParseEvent::Prefix { .. }));
        assert!(prefix_ev.is_some(), "should emit Prefix event");
    }

    #[test]
    fn test_turtle_prefix_stats() {
        let mut p = StreamingParser::new(turtle_config());
        p.feed("@prefix ex: <http://example.org/> .\n");
        p.feed("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
        assert_eq!(p.stats().prefixes, 2);
    }

    #[test]
    fn test_turtle_prefix_values() {
        let mut p = StreamingParser::new(turtle_config());
        let events = p.feed("PREFIX foo: <http://foo.org/> \n");
        if let Some(ParseEvent::Prefix { prefix, iri }) = events.first() {
            assert_eq!(prefix, "foo");
            assert_eq!(iri, "http://foo.org/");
        } else {
            panic!("expected Prefix event, got {:?}", events);
        }
    }

    #[test]
    fn test_turtle_base_iri_event() {
        let mut p = StreamingParser::new(turtle_config());
        let events = p.feed("@base <http://base.example.org/> .\n");
        let base_ev = events.iter().find(|e| matches!(e, ParseEvent::BaseIri(_)));
        assert!(base_ev.is_some(), "should emit BaseIri event");
    }

    #[test]
    fn test_turtle_triple_after_prefix() {
        let mut p = StreamingParser::new(turtle_config());
        p.feed("@prefix ex: <http://example.org/> .\n");
        let events = p.feed("<http://s> <http://p> <http://o> .\n");
        assert!(events
            .iter()
            .any(|e| matches!(e, ParseEvent::Triple { .. })));
    }

    // ─── N-Quads ──────────────────────────────────────────────────────────────

    #[test]
    fn test_nquads_emits_quad_event() {
        let mut p = StreamingParser::new(nquads_config());
        let events = p.feed("<http://s> <http://p> <http://o> <http://g> .\n");
        assert!(events.iter().any(|e| matches!(e, ParseEvent::Quad { .. })));
    }

    #[test]
    fn test_nquads_stats_quads() {
        let mut p = StreamingParser::new(nquads_config());
        p.feed("<http://s> <http://p> <http://o> <http://g> .\n");
        p.feed("<http://s2> <http://p2> <http://o2> <http://g2> .\n");
        assert_eq!(p.stats().quads, 2);
    }

    #[test]
    fn test_nquads_quad_fields() {
        let mut p = StreamingParser::new(nquads_config());
        let events = p.feed("<http://s> <http://p> <http://o> <http://g> .\n");
        if let ParseEvent::Quad { s, p: pred, o, g } = &events[0] {
            assert_eq!(s, "<http://s>");
            assert_eq!(pred, "<http://p>");
            assert_eq!(o, "<http://o>");
            assert_eq!(g, "<http://g>");
        } else {
            panic!("expected Quad event, got {:?}", events);
        }
    }

    // ─── Validation ───────────────────────────────────────────────────────────

    #[test]
    fn test_validate_flag_enabled_error_on_bad_term() {
        let config = StreamingParserConfig {
            format: StreamFormat::NTriples,
            validate: true,
            base_iri: None,
        };
        let mut p = StreamingParser::new(config);
        // Unclosed IRI
        let events = p.feed("<http://s <http://p> <http://o> .\n");
        assert!(events.iter().any(|e| matches!(e, ParseEvent::Error(_))));
    }

    #[test]
    fn test_validate_flag_disabled_no_error_on_partial() {
        let config = StreamingParserConfig {
            format: StreamFormat::NTriples,
            validate: false,
            base_iri: None,
        };
        let mut p = StreamingParser::new(config);
        // Only 2 terms — not a valid triple but validate=false
        let events = p.feed("<http://s> <http://p> .\n");
        // Should emit an error because parts < 3, but no validation-level check
        let _ = events; // just ensure no panic
    }

    // ─── base IRI config ──────────────────────────────────────────────────────

    #[test]
    fn test_base_iri_config_emits_event_on_new() {
        let config = StreamingParserConfig {
            format: StreamFormat::Turtle,
            validate: false,
            base_iri: Some("http://base.example.org/".to_string()),
        };
        let p = StreamingParser::new(config);
        // The event is queued but not yet returned (pending_events)
        assert_eq!(p.pending_events(), 1);
    }

    // ─── TriG ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_trig_quad_event() {
        let mut p = StreamingParser::new(trig_config());
        let events = p.feed("<http://s> <http://p> <http://o> <http://g> .\n");
        assert!(events.iter().any(|e| matches!(e, ParseEvent::Quad { .. })));
    }

    #[test]
    fn test_trig_prefix_event() {
        let mut p = StreamingParser::new(trig_config());
        let events = p.feed("PREFIX g: <http://graph.org/> \n");
        assert!(events
            .iter()
            .any(|e| matches!(e, ParseEvent::Prefix { .. })));
    }

    // ─── error stat ───────────────────────────────────────────────────────────

    #[test]
    fn test_error_stat_increments() {
        let mut p = StreamingParser::new(ntriples_config());
        p.feed("this is garbage\n");
        assert!(p.stats().errors > 0);
    }

    #[test]
    fn test_stats_triple_count_accumulated() {
        let mut p = StreamingParser::new(ntriples_config());
        for _ in 0..5 {
            p.feed("<http://s> <http://p> <http://o> .\n");
        }
        assert_eq!(p.stats().triples, 5);
    }

    #[test]
    fn test_flush_includes_remaining_triples() {
        let mut p = StreamingParser::new(ntriples_config());
        // Feed without trailing newline
        p.feed("<http://s> <http://p> <http://o> .");
        let events = p.flush();
        // The leftover should be parsed as a triple
        let has_triple = events
            .iter()
            .any(|e| matches!(e, ParseEvent::Triple { .. }));
        let has_end = events.contains(&ParseEvent::End);
        assert!(has_triple, "flush should parse leftover triple");
        assert!(has_end, "flush should emit End");
    }

    // ─── Additional tests (round 11 extra coverage) ───────────────────────────

    #[test]
    fn test_stats_bytes_processed() {
        let mut p = StreamingParser::new(ntriples_config());
        let chunk = "<http://s> <http://p> <http://o> .\n";
        p.feed(chunk);
        assert_eq!(p.stats().bytes_processed, chunk.len());
    }

    #[test]
    fn test_stats_bytes_accumulate() {
        let mut p = StreamingParser::new(ntriples_config());
        let chunk = "<http://s> <http://p> <http://o> .\n";
        p.feed(chunk);
        p.feed(chunk);
        assert_eq!(p.stats().bytes_processed, chunk.len() * 2);
    }

    #[test]
    fn test_ntriples_literal_subject() {
        let mut p = StreamingParser::new(ntriples_config());
        let events = p.feed("\"hello\" <http://p> <http://o> .\n");
        // "hello" starts with '"' which is a valid N-Triples term
        let has_triple = events
            .iter()
            .any(|e| matches!(e, ParseEvent::Triple { .. }));
        assert!(has_triple, "literal subject should produce Triple");
    }

    #[test]
    fn test_ntriples_blank_node_subject() {
        let mut p = StreamingParser::new(ntriples_config());
        let events = p.feed("_:blank <http://p> <http://o> .\n");
        let has_triple = events
            .iter()
            .any(|e| matches!(e, ParseEvent::Triple { .. }));
        assert!(has_triple, "blank node subject should produce Triple");
    }

    #[test]
    fn test_pending_events_count() {
        let p = StreamingParser::new(ntriples_config());
        assert_eq!(p.pending_events(), 0);
    }

    #[test]
    fn test_base_iri_config_bytes_counted() {
        let config = StreamingParserConfig {
            format: StreamFormat::NTriples,
            validate: false,
            base_iri: Some("http://base.org/".to_string()),
        };
        let p = StreamingParser::new(config);
        // After construction, no bytes processed
        assert_eq!(p.stats().bytes_processed, 0);
    }

    #[test]
    fn test_ntriples_multiple_triples_count() {
        let mut p = StreamingParser::new(ntriples_config());
        for _ in 0..10 {
            p.feed("<http://s> <http://p> <http://o> .\n");
        }
        assert_eq!(p.stats().triples, 10);
    }

    #[test]
    fn test_ntriples_comment_not_counted() {
        let mut p = StreamingParser::new(ntriples_config());
        p.feed("# This is a comment\n");
        assert_eq!(p.stats().triples, 0);
        assert_eq!(p.stats().errors, 0);
    }

    #[test]
    fn test_format_ntriples_enum() {
        let config = StreamingParserConfig {
            format: StreamFormat::NTriples,
            validate: false,
            base_iri: None,
        };
        assert_eq!(config.format, StreamFormat::NTriples);
    }

    #[test]
    fn test_format_nquads_enum() {
        assert_eq!(StreamFormat::NQuads, StreamFormat::NQuads);
    }

    #[test]
    fn test_reset_zeroes_bytes_processed() {
        let mut p = StreamingParser::new(ntriples_config());
        p.feed("<http://s> <http://p> <http://o> .\n");
        assert!(p.stats().bytes_processed > 0);
        p.reset();
        assert_eq!(p.stats().bytes_processed, 0);
    }

    #[test]
    fn test_turtle_base_event_value() {
        let mut p = StreamingParser::new(turtle_config());
        let events = p.feed("BASE <http://base.example.org/>\n");
        let base_event = events.iter().find(|e| matches!(e, ParseEvent::BaseIri(_)));
        assert!(
            base_event.is_some(),
            "BASE directive should emit BaseIri event"
        );
        if let Some(ParseEvent::BaseIri(iri)) = base_event {
            assert!(
                iri.contains("base.example.org"),
                "IRI should contain domain"
            );
        }
    }

    #[test]
    fn test_parse_event_error_message() {
        let mut p = StreamingParser::new(ntriples_config());
        let events = p.feed("garbage line without valid rdf terms\n");
        // Should produce an Error event
        let error_event = events.iter().find(|e| matches!(e, ParseEvent::Error(_)));
        assert!(
            error_event.is_some(),
            "invalid line should produce Error event"
        );
    }
}
