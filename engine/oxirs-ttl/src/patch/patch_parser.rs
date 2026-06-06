//! RDF Patch parser (batch and streaming) and graph mutation helpers.
//!
//! Exports: [`PatchParser`], [`apply_patch`], [`diff_to_patch`].

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};

use super::patch_types::{
    Graph, PatchChange, PatchError, PatchHeader, PatchQuad, PatchResult, PatchStats, PatchTerm,
    PatchTriple, RdfPatch,
};

// ─── PatchParser ─────────────────────────────────────────────────────────────

/// Parser for the RDF Patch text format
pub struct PatchParser;

impl PatchParser {
    /// Parse an entire RDF Patch document from a string
    pub fn parse(input: &str) -> PatchResult<RdfPatch> {
        let mut headers = Vec::new();
        let mut changes = Vec::new();
        let mut prefixes: BTreeMap<String, String> = BTreeMap::new();

        for (idx, raw_line) in input.lines().enumerate() {
            let line_no = idx + 1;
            let line = raw_line.trim();

            // Skip blank lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(rest) = line.strip_prefix("H ") {
                let header = Self::parse_header(rest.trim(), line_no)?;
                headers.push(header);
            } else if line == "TX" {
                changes.push(PatchChange::TransactionBegin);
            } else if line == "TC" {
                changes.push(PatchChange::TransactionCommit);
            } else if line == "TA" {
                changes.push(PatchChange::TransactionAbort);
            } else if let Some(rest) = line.strip_prefix("PA ") {
                let (prefix, iri) = Self::parse_prefix_decl(rest.trim(), line_no)?;
                prefixes.insert(prefix.clone(), iri.clone());
                changes.push(PatchChange::AddPrefix { prefix, iri });
            } else if let Some(rest) = line.strip_prefix("PD ") {
                let (prefix, iri) = Self::parse_prefix_decl(rest.trim(), line_no)?;
                changes.push(PatchChange::DeletePrefix { prefix, iri });
            } else if let Some(rest) = line.strip_prefix("A ") {
                let change = Self::parse_triple_or_quad("A", rest.trim(), &prefixes, line_no)?;
                changes.push(change);
            } else if let Some(rest) = line.strip_prefix("D ") {
                let change = Self::parse_triple_or_quad("D", rest.trim(), &prefixes, line_no)?;
                changes.push(change);
            } else {
                return Err(PatchError::at(
                    line_no,
                    format!("unrecognised line: {line:?}"),
                ));
            }
        }

        Ok(RdfPatch { headers, changes })
    }

    /// Create a streaming iterator that parses one [`PatchChange`] at a time.
    /// Headers are skipped in streaming mode (only change lines are yielded).
    pub fn parse_streaming(reader: impl Read) -> impl Iterator<Item = PatchResult<PatchChange>> {
        StreamingPatchParser::new(reader)
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn parse_header(rest: &str, line_no: usize) -> PatchResult<PatchHeader> {
        // rest is `key <value>` or `key value`
        let mut parts = rest.splitn(2, ' ');
        let key = parts
            .next()
            .ok_or_else(|| PatchError::at(line_no, "missing header key"))?
            .trim();
        let value_raw = parts.next().unwrap_or("").trim();
        let value = strip_angle_brackets(value_raw);
        match key {
            "version" => Ok(PatchHeader::Version(value.to_string())),
            "prev" => Ok(PatchHeader::Previous(value.to_string())),
            "id" => Ok(PatchHeader::Id(value.to_string())),
            other => Ok(PatchHeader::Unknown {
                key: other.to_string(),
                value: value.to_string(),
            }),
        }
    }

    fn parse_prefix_decl(rest: &str, line_no: usize) -> PatchResult<(String, String)> {
        // rest is `prefix <iri>` or `prefix: <iri>`
        let mut parts = rest.splitn(2, ' ');
        let prefix_raw = parts
            .next()
            .ok_or_else(|| PatchError::at(line_no, "missing prefix name"))?
            .trim_end_matches(':');
        let iri_raw = parts
            .next()
            .ok_or_else(|| PatchError::at(line_no, "missing prefix IRI"))?
            .trim();
        let iri = strip_angle_brackets(iri_raw);
        Ok((prefix_raw.to_string(), iri.to_string()))
    }

    pub(crate) fn parse_triple_or_quad(
        op: &str,
        rest: &str,
        prefixes: &BTreeMap<String, String>,
        line_no: usize,
    ) -> PatchResult<PatchChange> {
        // Strip trailing ' .' if present
        let rest = rest.trim_end_matches('.').trim();
        let terms = tokenise_terms(rest, prefixes, line_no)?;
        match terms.len() {
            3 => {
                let triple = PatchTriple::new(terms[0].clone(), terms[1].clone(), terms[2].clone());
                if op == "A" {
                    Ok(PatchChange::AddTriple(triple))
                } else {
                    Ok(PatchChange::DeleteTriple(triple))
                }
            }
            4 => {
                let quad = PatchQuad::new(
                    terms[0].clone(),
                    terms[1].clone(),
                    terms[2].clone(),
                    terms[3].clone(),
                );
                if op == "A" {
                    Ok(PatchChange::AddQuad(quad))
                } else {
                    Ok(PatchChange::DeleteQuad(quad))
                }
            }
            n => Err(PatchError::at(
                line_no,
                format!("expected 3 or 4 terms, got {n}"),
            )),
        }
    }
}

// ─── Streaming parser ────────────────────────────────────────────────────────

struct StreamingPatchParser<R: Read> {
    reader: BufReader<R>,
    line_no: usize,
    prefixes: BTreeMap<String, String>,
    done: bool,
}

impl<R: Read> StreamingPatchParser<R> {
    fn new(reader: R) -> Self {
        Self {
            reader: BufReader::new(reader),
            line_no: 0,
            prefixes: BTreeMap::new(),
            done: false,
        }
    }
}

impl<R: Read> Iterator for StreamingPatchParser<R> {
    type Item = PatchResult<PatchChange>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            let mut raw = String::new();
            match self.reader.read_line(&mut raw) {
                Ok(0) => {
                    self.done = true;
                    return None;
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(PatchError::at(self.line_no, e.to_string())));
                }
                Ok(_) => {}
            }
            self.line_no += 1;
            let line = raw.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Headers — skip silently in streaming mode
            if line.starts_with("H ") {
                continue;
            }

            let result = if line == "TX" {
                Ok(PatchChange::TransactionBegin)
            } else if line == "TC" {
                Ok(PatchChange::TransactionCommit)
            } else if line == "TA" {
                Ok(PatchChange::TransactionAbort)
            } else if let Some(rest) = line.strip_prefix("PA ") {
                match parse_prefix_decl_inline(rest.trim(), self.line_no) {
                    Ok((prefix, iri)) => {
                        self.prefixes.insert(prefix.clone(), iri.clone());
                        Ok(PatchChange::AddPrefix { prefix, iri })
                    }
                    Err(e) => Err(e),
                }
            } else if let Some(rest) = line.strip_prefix("PD ") {
                match parse_prefix_decl_inline(rest.trim(), self.line_no) {
                    Ok((prefix, iri)) => Ok(PatchChange::DeletePrefix { prefix, iri }),
                    Err(e) => Err(e),
                }
            } else if let Some(rest) = line.strip_prefix("A ") {
                PatchParser::parse_triple_or_quad("A", rest.trim(), &self.prefixes, self.line_no)
            } else if let Some(rest) = line.strip_prefix("D ") {
                PatchParser::parse_triple_or_quad("D", rest.trim(), &self.prefixes, self.line_no)
            } else {
                Err(PatchError::at(
                    self.line_no,
                    format!("unrecognised line: {line:?}"),
                ))
            };

            return Some(result);
        }
    }
}

// ─── apply_patch ─────────────────────────────────────────────────────────────

/// Apply an [`RdfPatch`] to an in-memory [`Graph`], updating it in place.
///
/// Transactions are honoured: changes between `TX`/`TA` are rolled back on abort.
/// Returns [`PatchStats`] summarising what was modified.
pub fn apply_patch(graph: &mut Graph, patch: &RdfPatch) -> PatchResult<PatchStats> {
    let mut stats = PatchStats::default();
    let mut in_tx = false;
    // Staged changes for the current transaction block
    let mut tx_adds: Vec<PatchTriple> = Vec::new();
    let mut tx_deletes: Vec<PatchTriple> = Vec::new();
    let mut tx_prefix_adds: Vec<(String, String)> = Vec::new();

    for change in &patch.changes {
        match change {
            PatchChange::TransactionBegin => {
                in_tx = true;
                tx_adds.clear();
                tx_deletes.clear();
                tx_prefix_adds.clear();
                stats.transactions += 1;
            }
            PatchChange::TransactionCommit => {
                // Commit staged changes
                for t in tx_adds.drain(..) {
                    if graph.add_triple(t) {
                        stats.triples_added += 1;
                    }
                }
                for t in &tx_deletes {
                    if graph.remove_triple(t) {
                        stats.triples_deleted += 1;
                    }
                }
                tx_deletes.clear();
                for (p, i) in tx_prefix_adds.drain(..) {
                    graph.prefixes.insert(p, i);
                    stats.prefixes_added += 1;
                }
                in_tx = false;
            }
            PatchChange::TransactionAbort => {
                // Discard staged changes
                tx_adds.clear();
                tx_deletes.clear();
                tx_prefix_adds.clear();
                in_tx = false;
                stats.aborts += 1;
            }
            PatchChange::AddPrefix { prefix, iri } => {
                if in_tx {
                    tx_prefix_adds.push((prefix.clone(), iri.clone()));
                } else {
                    graph.prefixes.insert(prefix.clone(), iri.clone());
                    stats.prefixes_added += 1;
                }
            }
            PatchChange::DeletePrefix { prefix, .. } => {
                graph.prefixes.remove(prefix.as_str());
                stats.prefixes_deleted += 1;
            }
            PatchChange::AddTriple(t) => {
                if in_tx {
                    tx_adds.push(t.clone());
                } else if graph.add_triple(t.clone()) {
                    stats.triples_added += 1;
                }
            }
            PatchChange::DeleteTriple(t) => {
                if in_tx {
                    tx_deletes.push(t.clone());
                } else if graph.remove_triple(t) {
                    stats.triples_deleted += 1;
                }
            }
            // Quads are not supported on simple Graph; treat as triple
            PatchChange::AddQuad(q) => {
                let t = PatchTriple::new(q.subject.clone(), q.predicate.clone(), q.object.clone());
                if in_tx {
                    tx_adds.push(t);
                } else if graph.add_triple(t) {
                    stats.triples_added += 1;
                }
            }
            PatchChange::DeleteQuad(q) => {
                let t = PatchTriple::new(q.subject.clone(), q.predicate.clone(), q.object.clone());
                if in_tx {
                    tx_deletes.push(t.clone());
                } else if graph.remove_triple(&t) {
                    stats.triples_deleted += 1;
                }
            }
        }
    }

    Ok(stats)
}

// ─── diff_to_patch ───────────────────────────────────────────────────────────

/// Generate a minimal [`RdfPatch`] that transforms `old` into `new`.
///
/// All deletions come before additions in the generated patch, matching
/// the convention used by most RDF Patch tools.
pub fn diff_to_patch(old: &Graph, new: &Graph) -> RdfPatch {
    let mut changes = Vec::new();

    // Deletes: triples in old but not new
    for triple in old.iter() {
        if !new.contains(triple) {
            changes.push(PatchChange::DeleteTriple(triple.clone()));
        }
    }

    // Adds: triples in new but not old
    for triple in new.iter() {
        if !old.contains(triple) {
            changes.push(PatchChange::AddTriple(triple.clone()));
        }
    }

    // Prefix adds: in new but not old
    for (prefix, iri) in &new.prefixes {
        if old.prefixes.get(prefix) != Some(iri) {
            changes.push(PatchChange::AddPrefix {
                prefix: prefix.clone(),
                iri: iri.clone(),
            });
        }
    }

    // Prefix deletes: in old but not new
    for (prefix, iri) in &old.prefixes {
        if !new.prefixes.contains_key(prefix.as_str()) {
            changes.push(PatchChange::DeletePrefix {
                prefix: prefix.clone(),
                iri: iri.clone(),
            });
        }
    }

    RdfPatch {
        headers: Vec::new(),
        changes,
    }
}

// ─── Term tokeniser ──────────────────────────────────────────────────────────

/// Tokenise a whitespace-separated sequence of RDF terms.
/// Handles IRIs (`<...>`), blank nodes (`_:id`), literals (`"..."`), and
/// prefixed names (`prefix:local`).
pub(crate) fn tokenise_terms(
    input: &str,
    prefixes: &BTreeMap<String, String>,
    line_no: usize,
) -> PatchResult<Vec<PatchTerm>> {
    let mut terms = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0;

    while pos < chars.len() {
        // Skip whitespace
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }
        if pos >= chars.len() {
            break;
        }

        if chars[pos] == '<' {
            // IRI
            pos += 1;
            let start = pos;
            while pos < chars.len() && chars[pos] != '>' {
                pos += 1;
            }
            if pos >= chars.len() {
                return Err(PatchError::at(line_no, "unterminated IRI"));
            }
            let iri: String = chars[start..pos].iter().collect();
            pos += 1; // consume '>'
            terms.push(PatchTerm::iri(iri));
        } else if chars[pos] == '"' {
            // Literal
            pos += 1;
            let mut value = String::new();
            while pos < chars.len() {
                if chars[pos] == '\\' && pos + 1 < chars.len() {
                    pos += 1;
                    match chars[pos] {
                        '"' => value.push('"'),
                        '\\' => value.push('\\'),
                        'n' => value.push('\n'),
                        'r' => value.push('\r'),
                        't' => value.push('\t'),
                        c => {
                            value.push('\\');
                            value.push(c);
                        }
                    }
                    pos += 1;
                } else if chars[pos] == '"' {
                    break;
                } else {
                    value.push(chars[pos]);
                    pos += 1;
                }
            }
            if pos >= chars.len() {
                return Err(PatchError::at(line_no, "unterminated literal"));
            }
            pos += 1; // consume closing '"'

            // Check for language tag or datatype
            if pos < chars.len() && chars[pos] == '@' {
                pos += 1;
                let start = pos;
                while pos < chars.len() && !chars[pos].is_whitespace() {
                    pos += 1;
                }
                let lang: String = chars[start..pos].iter().collect();
                terms.push(PatchTerm::lang_literal(value, lang));
            } else if pos + 1 < chars.len() && chars[pos] == '^' && chars[pos + 1] == '^' {
                pos += 2;
                if pos >= chars.len() || chars[pos] != '<' {
                    return Err(PatchError::at(line_no, "expected '<' after '^^'"));
                }
                pos += 1;
                let start = pos;
                while pos < chars.len() && chars[pos] != '>' {
                    pos += 1;
                }
                if pos >= chars.len() {
                    return Err(PatchError::at(line_no, "unterminated datatype IRI"));
                }
                let dt: String = chars[start..pos].iter().collect();
                pos += 1;
                terms.push(PatchTerm::typed_literal(value, dt));
            } else {
                terms.push(PatchTerm::literal(value));
            }
        } else if pos + 1 < chars.len() && chars[pos] == '_' && chars[pos + 1] == ':' {
            // Blank node
            pos += 2;
            let start = pos;
            while pos < chars.len() && !chars[pos].is_whitespace() && chars[pos] != '.' {
                pos += 1;
            }
            let id: String = chars[start..pos].iter().collect();
            terms.push(PatchTerm::blank_node(id));
        } else if chars[pos] == '.' {
            // Trailing dot — stop
            pos += 1;
        } else {
            // Possibly a prefixed name `prefix:local`
            let start = pos;
            while pos < chars.len() && !chars[pos].is_whitespace() && chars[pos] != '.' {
                pos += 1;
            }
            let token: String = chars[start..pos].iter().collect();
            if let Some(colon_pos) = token.find(':') {
                let ns = &token[..colon_pos];
                let local = &token[colon_pos + 1..];
                match prefixes.get(ns) {
                    Some(base) => {
                        let full = format!("{base}{local}");
                        terms.push(PatchTerm::iri(full));
                    }
                    None => {
                        return Err(PatchError::at(
                            line_no,
                            format!("unknown prefix '{ns}' in '{token}'"),
                        ))
                    }
                }
            } else if token.is_empty() || token == "." {
                // skip
            } else {
                return Err(PatchError::at(
                    line_no,
                    format!("unexpected token '{token}'"),
                ));
            }
        }
    }

    Ok(terms)
}

/// Strip surrounding `<...>` from an IRI token, if present
pub(crate) fn strip_angle_brackets(s: &str) -> &str {
    if s.starts_with('<') && s.ends_with('>') {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Inline prefix-decl parser used in the streaming parser
pub(crate) fn parse_prefix_decl_inline(
    rest: &str,
    line_no: usize,
) -> PatchResult<(String, String)> {
    let mut parts = rest.splitn(2, ' ');
    let prefix_raw = parts
        .next()
        .ok_or_else(|| PatchError::at(line_no, "missing prefix name"))?
        .trim_end_matches(':');
    let iri_raw = parts
        .next()
        .ok_or_else(|| PatchError::at(line_no, "missing prefix IRI"))?
        .trim();
    let iri = strip_angle_brackets(iri_raw);
    Ok((prefix_raw.to_string(), iri.to_string()))
}
