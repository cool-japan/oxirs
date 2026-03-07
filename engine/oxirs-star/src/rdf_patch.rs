//! RDF Patch format (W3C) for incremental graph changes.
//!
//! RDF Patch describes additions and deletions of triples/quads using a simple
//! text format.  This module implements the data model, serialization,
//! parsing, inversion, and merging of patches.

use std::fmt;

// ─────────────────────────────────────────────────
// PatchOp
// ─────────────────────────────────────────────────

/// A single operation in an RDF patch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchOp {
    /// Add a triple to the default graph.
    AddTriple { s: String, p: String, o: String },
    /// Delete a triple from the default graph.
    DeleteTriple { s: String, p: String, o: String },
    /// Add a quad (triple + named graph).
    AddQuad {
        s: String,
        p: String,
        o: String,
        g: String,
    },
    /// Delete a quad.
    DeleteQuad {
        s: String,
        p: String,
        o: String,
        g: String,
    },
    /// Declare a prefix.
    AddPrefix { prefix: String, iri: String },
    /// Remove a prefix declaration.
    DeletePrefix { prefix: String },
    /// Patch header key-value pair.
    Header { key: String, value: String },
    /// Begin a transaction.
    TxBegin,
    /// Commit the transaction.
    TxCommit,
    /// Abort the transaction.
    TxAbort,
}

// ─────────────────────────────────────────────────
// PatchStats
// ─────────────────────────────────────────────────

/// Summary statistics for a patch.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PatchStats {
    pub adds: usize,
    pub deletes: usize,
    pub add_prefixes: usize,
    pub delete_prefixes: usize,
}

// ─────────────────────────────────────────────────
// PatchApplyResult
// ─────────────────────────────────────────────────

/// The result of applying a patch to a set of triples.
#[derive(Debug, Clone)]
pub struct PatchApplyResult {
    pub stats: PatchStats,
    pub triples_added: Vec<(String, String, String)>,
    pub triples_deleted: Vec<(String, String, String)>,
}

// ─────────────────────────────────────────────────
// PatchError
// ─────────────────────────────────────────────────

/// Errors that can occur while parsing a patch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchError {
    ParseError(String),
    InvalidFormat(String),
}

impl fmt::Display for PatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatchError::ParseError(s) => write!(f, "ParseError: {s}"),
            PatchError::InvalidFormat(s) => write!(f, "InvalidFormat: {s}"),
        }
    }
}

// ─────────────────────────────────────────────────
// RdfPatch
// ─────────────────────────────────────────────────

/// An RDF Patch document: an ordered sequence of operations.
#[derive(Debug, Clone, Default)]
pub struct RdfPatch {
    pub ops: Vec<PatchOp>,
}

impl RdfPatch {
    /// Create an empty patch.
    pub fn new() -> Self {
        RdfPatch { ops: Vec::new() }
    }

    /// Append an operation.
    pub fn add(&mut self, op: PatchOp) {
        self.ops.push(op);
    }

    /// Number of operations in this patch.
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Compute statistics for this patch.
    pub fn stats(&self) -> PatchStats {
        let mut s = PatchStats::default();
        for op in &self.ops {
            match op {
                PatchOp::AddTriple { .. } | PatchOp::AddQuad { .. } => s.adds += 1,
                PatchOp::DeleteTriple { .. } | PatchOp::DeleteQuad { .. } => s.deletes += 1,
                PatchOp::AddPrefix { .. } => s.add_prefixes += 1,
                PatchOp::DeletePrefix { .. } => s.delete_prefixes += 1,
                _ => {}
            }
        }
        s
    }

    /// Apply this patch to a mutable triple set.
    ///
    /// Quad operations are projected to triples (the graph name is ignored).
    /// Deletions remove the first matching triple; additions append new triples.
    pub fn apply_to(&self, triples: &mut Vec<(String, String, String)>) -> PatchApplyResult {
        let mut added = Vec::new();
        let mut deleted = Vec::new();
        let mut stats = PatchStats::default();

        for op in &self.ops {
            match op {
                PatchOp::AddTriple { s, p, o } | PatchOp::AddQuad { s, p, o, .. } => {
                    let t = (s.clone(), p.clone(), o.clone());
                    triples.push(t.clone());
                    added.push(t);
                    stats.adds += 1;
                }
                PatchOp::DeleteTriple { s, p, o } | PatchOp::DeleteQuad { s, p, o, .. } => {
                    let t = (s.clone(), p.clone(), o.clone());
                    if let Some(pos) = triples.iter().position(|tr| tr == &t) {
                        triples.remove(pos);
                        deleted.push(t);
                        stats.deletes += 1;
                    }
                }
                PatchOp::AddPrefix { .. } => stats.add_prefixes += 1,
                PatchOp::DeletePrefix { .. } => stats.delete_prefixes += 1,
                _ => {}
            }
        }

        PatchApplyResult {
            stats,
            triples_added: added,
            triples_deleted: deleted,
        }
    }

    /// Serialize this patch to a simple text format.
    ///
    /// Format:
    /// ```text
    /// TX .
    /// A <s> <p> <o> .
    /// D <s> <p> <o> .
    /// PA prefix: <iri> .
    /// DP prefix: .
    /// H key value .
    /// TC .
    /// ```
    pub fn serialize(&self) -> String {
        let mut out = String::new();
        for op in &self.ops {
            match op {
                PatchOp::TxBegin => out.push_str("TX .\n"),
                PatchOp::TxCommit => out.push_str("TC .\n"),
                PatchOp::TxAbort => out.push_str("TA .\n"),
                PatchOp::AddTriple { s, p, o } => out.push_str(&format!("A <{s}> <{p}> <{o}> .\n")),
                PatchOp::DeleteTriple { s, p, o } => {
                    out.push_str(&format!("D <{s}> <{p}> <{o}> .\n"))
                }
                PatchOp::AddQuad { s, p, o, g } => {
                    out.push_str(&format!("AQ <{s}> <{p}> <{o}> <{g}> .\n"))
                }
                PatchOp::DeleteQuad { s, p, o, g } => {
                    out.push_str(&format!("DQ <{s}> <{p}> <{o}> <{g}> .\n"))
                }
                PatchOp::AddPrefix { prefix, iri } => {
                    out.push_str(&format!("PA {prefix}: <{iri}> .\n"))
                }
                PatchOp::DeletePrefix { prefix } => out.push_str(&format!("DP {prefix}: .\n")),
                PatchOp::Header { key, value } => out.push_str(&format!("H {key} {value} .\n")),
            }
        }
        out
    }

    /// Parse a serialized patch back to an `RdfPatch`.
    pub fn parse(input: &str) -> Result<RdfPatch, PatchError> {
        let mut patch = RdfPatch::new();

        for (line_no, raw_line) in input.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }

            // Strip trailing " ."
            let line = line
                .strip_suffix(" .")
                .or_else(|| line.strip_suffix('.').map(|l| l.trim_end()))
                .unwrap_or(line);

            let op = if line == "TX" {
                PatchOp::TxBegin
            } else if line == "TC" {
                PatchOp::TxCommit
            } else if line == "TA" {
                PatchOp::TxAbort
            } else if let Some(rest) = line.strip_prefix("A ") {
                let (s, p, o) = parse_triple(rest)
                    .map_err(|e| PatchError::ParseError(format!("line {line_no}: {e}")))?;
                PatchOp::AddTriple { s, p, o }
            } else if let Some(rest) = line.strip_prefix("D ") {
                let (s, p, o) = parse_triple(rest)
                    .map_err(|e| PatchError::ParseError(format!("line {line_no}: {e}")))?;
                PatchOp::DeleteTriple { s, p, o }
            } else if let Some(rest) = line.strip_prefix("AQ ") {
                let (s, p, o, g) = parse_quad(rest)
                    .map_err(|e| PatchError::ParseError(format!("line {line_no}: {e}")))?;
                PatchOp::AddQuad { s, p, o, g }
            } else if let Some(rest) = line.strip_prefix("DQ ") {
                let (s, p, o, g) = parse_quad(rest)
                    .map_err(|e| PatchError::ParseError(format!("line {line_no}: {e}")))?;
                PatchOp::DeleteQuad { s, p, o, g }
            } else if let Some(rest) = line.strip_prefix("PA ") {
                let (prefix, iri) = parse_prefix_decl(rest)
                    .map_err(|e| PatchError::ParseError(format!("line {line_no}: {e}")))?;
                PatchOp::AddPrefix { prefix, iri }
            } else if let Some(rest) = line.strip_prefix("DP ") {
                let prefix = rest.trim_end_matches(':').trim().to_string();
                PatchOp::DeletePrefix { prefix }
            } else if let Some(rest) = line.strip_prefix("H ") {
                let mut parts = rest.splitn(2, ' ');
                let key = parts
                    .next()
                    .ok_or_else(|| {
                        PatchError::ParseError(format!("line {line_no}: missing header key"))
                    })?
                    .to_string();
                let value = parts.next().unwrap_or("").to_string();
                PatchOp::Header { key, value }
            } else {
                return Err(PatchError::InvalidFormat(format!(
                    "line {line_no}: unknown directive: {line}"
                )));
            };

            patch.add(op);
        }

        Ok(patch)
    }

    /// Create a new patch that is the inverse of this one (swaps adds and deletes).
    pub fn invert(&self) -> RdfPatch {
        let mut inv = RdfPatch::new();
        for op in &self.ops {
            let inverted = match op {
                PatchOp::AddTriple { s, p, o } => PatchOp::DeleteTriple {
                    s: s.clone(),
                    p: p.clone(),
                    o: o.clone(),
                },
                PatchOp::DeleteTriple { s, p, o } => PatchOp::AddTriple {
                    s: s.clone(),
                    p: p.clone(),
                    o: o.clone(),
                },
                PatchOp::AddQuad { s, p, o, g } => PatchOp::DeleteQuad {
                    s: s.clone(),
                    p: p.clone(),
                    o: o.clone(),
                    g: g.clone(),
                },
                PatchOp::DeleteQuad { s, p, o, g } => PatchOp::AddQuad {
                    s: s.clone(),
                    p: p.clone(),
                    o: o.clone(),
                    g: g.clone(),
                },
                PatchOp::AddPrefix { prefix, iri: _ } => PatchOp::DeletePrefix {
                    prefix: prefix.clone(),
                },
                PatchOp::DeletePrefix { prefix } => PatchOp::AddPrefix {
                    prefix: prefix.clone(),
                    iri: String::new(),
                },
                other => other.clone(),
            };
            inv.add(inverted);
        }
        inv
    }

    /// Merge a slice of patches into a single patch (concatenation).
    pub fn merge(patches: &[RdfPatch]) -> RdfPatch {
        let mut merged = RdfPatch::new();
        for p in patches {
            for op in &p.ops {
                merged.add(op.clone());
            }
        }
        merged
    }
}

// ─────────────────────────────────────────────────
// Parsing helpers
// ─────────────────────────────────────────────────

/// Extract an IRI from `<...>`, returning the inner string.
fn extract_iri(token: &str) -> Result<String, String> {
    let t = token.trim();
    if t.starts_with('<') && t.ends_with('>') {
        Ok(t[1..t.len() - 1].to_string())
    } else {
        Err(format!("expected IRI in angle brackets, got: {t}"))
    }
}

/// Parse "<s> <p> <o>" → (s, p, o).
fn parse_triple(rest: &str) -> Result<(String, String, String), String> {
    let tokens: Vec<&str> = tokenize_iri_sequence(rest);
    if tokens.len() < 3 {
        return Err(format!("expected 3 IRI tokens, got {}", tokens.len()));
    }
    let s = extract_iri(tokens[0])?;
    let p = extract_iri(tokens[1])?;
    let o = extract_iri(tokens[2])?;
    Ok((s, p, o))
}

/// Parse "<s> <p> <o> <g>" → (s, p, o, g).
fn parse_quad(rest: &str) -> Result<(String, String, String, String), String> {
    let tokens: Vec<&str> = tokenize_iri_sequence(rest);
    if tokens.len() < 4 {
        return Err(format!("expected 4 IRI tokens, got {}", tokens.len()));
    }
    let s = extract_iri(tokens[0])?;
    let p = extract_iri(tokens[1])?;
    let o = extract_iri(tokens[2])?;
    let g = extract_iri(tokens[3])?;
    Ok((s, p, o, g))
}

/// Parse "prefix: <iri>" → (prefix, iri).
fn parse_prefix_decl(rest: &str) -> Result<(String, String), String> {
    let mut parts = rest.splitn(2, ' ');
    let prefix_part = parts
        .next()
        .ok_or_else(|| "missing prefix".to_string())?
        .trim_end_matches(':')
        .to_string();
    let iri_part = parts
        .next()
        .ok_or_else(|| "missing IRI".to_string())?
        .trim();
    let iri = extract_iri(iri_part)?;
    Ok((prefix_part, iri))
}

/// Tokenize angle-bracketed IRIs from a string, respecting `<...>` boundaries.
fn tokenize_iri_sequence(s: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut remaining = s.trim();
    while let Some(start) = remaining.find('<') {
        let tail = &remaining[start..];
        if let Some(end) = tail.find('>') {
            tokens.push(&tail[..=end]);
            remaining = tail[end + 1..].trim();
        } else {
            break;
        }
    }
    tokens
}

// ─────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triple_op(s: &str, p: &str, o: &str) -> PatchOp {
        PatchOp::AddTriple {
            s: s.into(),
            p: p.into(),
            o: o.into(),
        }
    }

    fn del_triple_op(s: &str, p: &str, o: &str) -> PatchOp {
        PatchOp::DeleteTriple {
            s: s.into(),
            p: p.into(),
            o: o.into(),
        }
    }

    fn make_triple(s: &str, p: &str, o: &str) -> (String, String, String) {
        (s.into(), p.into(), o.into())
    }

    // ── Empty patch ────────────────────────────────────────────

    #[test]
    fn test_empty_patch() {
        let p = RdfPatch::new();
        assert_eq!(p.op_count(), 0);
        let s = p.stats();
        assert_eq!(s.adds, 0);
        assert_eq!(s.deletes, 0);
    }

    #[test]
    fn test_empty_patch_serialize() {
        let p = RdfPatch::new();
        assert_eq!(p.serialize(), "");
    }

    #[test]
    fn test_empty_patch_default() {
        let p = RdfPatch::default();
        assert!(p.ops.is_empty());
    }

    // ── Add / Delete triples ───────────────────────────────────

    #[test]
    fn test_add_triple_op_count() {
        let mut p = RdfPatch::new();
        p.add(triple_op("s1", "p1", "o1"));
        assert_eq!(p.op_count(), 1);
    }

    #[test]
    fn test_add_delete_stats() {
        let mut p = RdfPatch::new();
        p.add(triple_op("s", "p", "o"));
        p.add(del_triple_op("s", "p", "o"));
        let s = p.stats();
        assert_eq!(s.adds, 1);
        assert_eq!(s.deletes, 1);
    }

    // ── apply_to ──────────────────────────────────────────────

    #[test]
    fn test_apply_add_triple() {
        let mut p = RdfPatch::new();
        p.add(triple_op("http://s", "http://p", "http://o"));
        let mut triples: Vec<(String, String, String)> = Vec::new();
        let result = p.apply_to(&mut triples);
        assert_eq!(triples.len(), 1);
        assert_eq!(result.triples_added.len(), 1);
        assert_eq!(result.stats.adds, 1);
    }

    #[test]
    fn test_apply_delete_triple() {
        let mut p = RdfPatch::new();
        p.add(del_triple_op("http://s", "http://p", "http://o"));
        let mut triples = vec![make_triple("http://s", "http://p", "http://o")];
        let result = p.apply_to(&mut triples);
        assert!(triples.is_empty());
        assert_eq!(result.triples_deleted.len(), 1);
        assert_eq!(result.stats.deletes, 1);
    }

    #[test]
    fn test_apply_delete_nonexistent_noop() {
        let mut p = RdfPatch::new();
        p.add(del_triple_op("http://x", "http://y", "http://z"));
        let mut triples: Vec<(String, String, String)> = Vec::new();
        let result = p.apply_to(&mut triples);
        assert_eq!(result.stats.deletes, 0);
    }

    #[test]
    fn test_apply_add_then_delete() {
        let mut p = RdfPatch::new();
        p.add(triple_op("s", "p", "o"));
        p.add(del_triple_op("s", "p", "o"));
        let mut triples: Vec<(String, String, String)> = Vec::new();
        p.apply_to(&mut triples);
        assert!(triples.is_empty());
    }

    // ── Quad ops ──────────────────────────────────────────────

    #[test]
    fn test_add_quad_stats() {
        let mut p = RdfPatch::new();
        p.add(PatchOp::AddQuad {
            s: "s".into(),
            p: "p".into(),
            o: "o".into(),
            g: "g".into(),
        });
        assert_eq!(p.stats().adds, 1);
    }

    #[test]
    fn test_delete_quad_apply() {
        let mut p = RdfPatch::new();
        p.add(PatchOp::DeleteQuad {
            s: "http://s".into(),
            p: "http://p".into(),
            o: "http://o".into(),
            g: "http://g".into(),
        });
        let mut triples = vec![make_triple("http://s", "http://p", "http://o")];
        let result = p.apply_to(&mut triples);
        assert!(triples.is_empty());
        assert_eq!(result.stats.deletes, 1);
    }

    // ── Prefix ops ────────────────────────────────────────────

    #[test]
    fn test_prefix_stats() {
        let mut p = RdfPatch::new();
        p.add(PatchOp::AddPrefix {
            prefix: "ex".into(),
            iri: "http://example.org/".into(),
        });
        p.add(PatchOp::DeletePrefix {
            prefix: "old".into(),
        });
        let s = p.stats();
        assert_eq!(s.add_prefixes, 1);
        assert_eq!(s.delete_prefixes, 1);
    }

    // ── Serialize / Parse round-trip ──────────────────────────

    #[test]
    fn test_serialize_add_triple() {
        let mut p = RdfPatch::new();
        p.add(triple_op("http://s", "http://p", "http://o"));
        let s = p.serialize();
        assert!(s.contains("A <http://s> <http://p> <http://o> ."));
    }

    #[test]
    fn test_serialize_delete_triple() {
        let mut p = RdfPatch::new();
        p.add(del_triple_op("http://s", "http://p", "http://o"));
        let s = p.serialize();
        assert!(s.contains("D <http://s> <http://p> <http://o> ."));
    }

    #[test]
    fn test_serialize_tx_begin_commit() {
        let mut p = RdfPatch::new();
        p.add(PatchOp::TxBegin);
        p.add(triple_op("s", "p", "o"));
        p.add(PatchOp::TxCommit);
        let s = p.serialize();
        assert!(s.contains("TX ."));
        assert!(s.contains("TC ."));
    }

    #[test]
    fn test_serialize_tx_abort() {
        let mut p = RdfPatch::new();
        p.add(PatchOp::TxAbort);
        assert!(p.serialize().contains("TA ."));
    }

    #[test]
    fn test_parse_add_triple() {
        let input = "A <http://s> <http://p> <http://o> .\n";
        let p = RdfPatch::parse(input).unwrap();
        assert_eq!(p.op_count(), 1);
        assert!(
            matches!(&p.ops[0], PatchOp::AddTriple { s, p: pred, o } if s == "http://s" && pred == "http://p" && o == "http://o")
        );
    }

    #[test]
    fn test_parse_delete_triple() {
        let input = "D <http://s> <http://p> <http://o> .\n";
        let p = RdfPatch::parse(input).unwrap();
        assert!(matches!(&p.ops[0], PatchOp::DeleteTriple { .. }));
    }

    #[test]
    fn test_parse_tx_ops() {
        let input = "TX .\nTC .\nTA .\n";
        let p = RdfPatch::parse(input).unwrap();
        assert_eq!(p.op_count(), 3);
        assert_eq!(p.ops[0], PatchOp::TxBegin);
        assert_eq!(p.ops[1], PatchOp::TxCommit);
        assert_eq!(p.ops[2], PatchOp::TxAbort);
    }

    #[test]
    fn test_round_trip_add_delete() {
        let mut original = RdfPatch::new();
        original.add(PatchOp::TxBegin);
        original.add(triple_op("http://a", "http://b", "http://c"));
        original.add(del_triple_op("http://x", "http://y", "http://z"));
        original.add(PatchOp::TxCommit);

        let serialized = original.serialize();
        let parsed = RdfPatch::parse(&serialized).unwrap();
        assert_eq!(parsed.op_count(), original.op_count());
    }

    #[test]
    fn test_parse_invalid_directive_error() {
        let input = "UNKNOWN <a> <b> <c> .\n";
        let result = RdfPatch::parse(input);
        assert!(result.is_err());
    }

    // ── Invert ────────────────────────────────────────────────

    #[test]
    fn test_invert_swaps_add_delete() {
        let mut p = RdfPatch::new();
        p.add(triple_op("s", "p", "o"));
        let inv = p.invert();
        assert!(matches!(&inv.ops[0], PatchOp::DeleteTriple { .. }));
    }

    #[test]
    fn test_invert_swaps_delete_add() {
        let mut p = RdfPatch::new();
        p.add(del_triple_op("s", "p", "o"));
        let inv = p.invert();
        assert!(matches!(&inv.ops[0], PatchOp::AddTriple { .. }));
    }

    #[test]
    fn test_invert_swaps_quad_ops() {
        let mut p = RdfPatch::new();
        p.add(PatchOp::AddQuad {
            s: "s".into(),
            p: "p".into(),
            o: "o".into(),
            g: "g".into(),
        });
        let inv = p.invert();
        assert!(matches!(&inv.ops[0], PatchOp::DeleteQuad { .. }));
    }

    #[test]
    fn test_invert_preserves_tx_ops() {
        let mut p = RdfPatch::new();
        p.add(PatchOp::TxBegin);
        p.add(PatchOp::TxCommit);
        let inv = p.invert();
        assert_eq!(inv.ops[0], PatchOp::TxBegin);
        assert_eq!(inv.ops[1], PatchOp::TxCommit);
    }

    #[test]
    fn test_double_invert_identity() {
        let mut p = RdfPatch::new();
        p.add(triple_op("s", "p", "o"));
        p.add(del_triple_op("a", "b", "c"));
        let inv = p.invert().invert();
        assert_eq!(inv.op_count(), p.op_count());
        // The first op should be AddTriple again after double inversion
        assert!(matches!(&inv.ops[0], PatchOp::AddTriple { .. }));
        assert!(matches!(&inv.ops[1], PatchOp::DeleteTriple { .. }));
    }

    // ── Merge ─────────────────────────────────────────────────

    #[test]
    fn test_merge_empty_slices() {
        let merged = RdfPatch::merge(&[]);
        assert_eq!(merged.op_count(), 0);
    }

    #[test]
    fn test_merge_two_patches() {
        let mut p1 = RdfPatch::new();
        p1.add(triple_op("s1", "p1", "o1"));
        let mut p2 = RdfPatch::new();
        p2.add(del_triple_op("s2", "p2", "o2"));
        let merged = RdfPatch::merge(&[p1, p2]);
        assert_eq!(merged.op_count(), 2);
        assert!(matches!(&merged.ops[0], PatchOp::AddTriple { .. }));
        assert!(matches!(&merged.ops[1], PatchOp::DeleteTriple { .. }));
    }

    #[test]
    fn test_merge_preserves_order() {
        let mut p1 = RdfPatch::new();
        p1.add(PatchOp::TxBegin);
        let mut p2 = RdfPatch::new();
        p2.add(PatchOp::TxCommit);
        let merged = RdfPatch::merge(&[p1, p2]);
        assert_eq!(merged.ops[0], PatchOp::TxBegin);
        assert_eq!(merged.ops[1], PatchOp::TxCommit);
    }

    // ── Stats counting ────────────────────────────────────────

    #[test]
    fn test_stats_mixed_ops() {
        let mut p = RdfPatch::new();
        p.add(triple_op("s", "p", "o")); // add
        p.add(triple_op("a", "b", "c")); // add
        p.add(del_triple_op("x", "y", "z")); // delete
        p.add(PatchOp::AddPrefix {
            prefix: "ex".into(),
            iri: "http://example.org/".into(),
        });
        p.add(PatchOp::DeletePrefix {
            prefix: "old".into(),
        });
        p.add(PatchOp::TxBegin); // no stat
        let s = p.stats();
        assert_eq!(s.adds, 2);
        assert_eq!(s.deletes, 1);
        assert_eq!(s.add_prefixes, 1);
        assert_eq!(s.delete_prefixes, 1);
    }

    // ── PatchError Display ────────────────────────────────────

    #[test]
    fn test_patch_error_display() {
        let e1 = PatchError::ParseError("bad input".into());
        assert!(e1.to_string().contains("bad input"));
        let e2 = PatchError::InvalidFormat("unknown".into());
        assert!(e2.to_string().contains("unknown"));
    }

    // ── Apply result fields ───────────────────────────────────

    #[test]
    fn test_apply_result_added_triples() {
        let mut p = RdfPatch::new();
        p.add(triple_op("http://s", "http://p", "http://o"));
        let mut triples = Vec::new();
        let result = p.apply_to(&mut triples);
        assert_eq!(result.triples_added.len(), 1);
        assert_eq!(result.triples_added[0].0, "http://s");
    }

    #[test]
    fn test_apply_result_deleted_triples() {
        let mut p = RdfPatch::new();
        p.add(del_triple_op("http://s", "http://p", "http://o"));
        let mut triples = vec![make_triple("http://s", "http://p", "http://o")];
        let result = p.apply_to(&mut triples);
        assert_eq!(result.triples_deleted[0].0, "http://s");
    }
}
