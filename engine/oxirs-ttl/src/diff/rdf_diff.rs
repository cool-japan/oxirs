//! RDF graph diff and patch utilities
//!
//! Computes the symmetric difference between two RDF graphs and supports
//! serialising the result to a simple patch format (lines prefixed with `+`
//! or `-`) as well as parsing such patches back into [`RdfDiff`] values.
//!
//! The diff and patch operations use the lightweight [`RdfTerm`] / [`NTriple`]
//! types from this crate, so no dependency on `oxirs-core` is required here.
//!
//! # Examples
//!
//! ## Computing a diff
//!
//! ```rust
//! use oxirs_ttl::writer::RdfTerm;
//! use oxirs_ttl::diff::{compute_diff, RdfDiff};
//!
//! let before = vec![
//!     (RdfTerm::iri("http://s"), RdfTerm::iri("http://p"), RdfTerm::iri("http://o1")),
//! ];
//! let after = vec![
//!     (RdfTerm::iri("http://s"), RdfTerm::iri("http://p"), RdfTerm::iri("http://o2")),
//! ];
//!
//! let diff = compute_diff(&before, &after);
//! assert_eq!(diff.added.len(), 1);
//! assert_eq!(diff.removed.len(), 1);
//! ```
//!
//! ## Applying a diff
//!
//! ```rust
//! use oxirs_ttl::writer::RdfTerm;
//! use oxirs_ttl::diff::compute_diff;
//!
//! let before = vec![
//!     (RdfTerm::iri("http://s"), RdfTerm::iri("http://p"), RdfTerm::iri("http://o1")),
//! ];
//! let after = vec![
//!     (RdfTerm::iri("http://s"), RdfTerm::iri("http://p"), RdfTerm::iri("http://o2")),
//! ];
//!
//! let diff = compute_diff(&before, &after);
//! let mut graph = before.clone();
//! diff.apply(&mut graph);
//! assert_eq!(graph, after);
//! ```
//!
//! ## Round-trip via patch format
//!
//! ```rust
//! use oxirs_ttl::writer::RdfTerm;
//! use oxirs_ttl::diff::{compute_diff, parse_patch};
//!
//! let before = vec![
//!     (RdfTerm::iri("http://s"), RdfTerm::iri("http://p"), RdfTerm::iri("http://o1")),
//! ];
//! let after = vec![
//!     (RdfTerm::iri("http://s"), RdfTerm::iri("http://p"), RdfTerm::iri("http://o2")),
//! ];
//!
//! let diff = compute_diff(&before, &after);
//! let patch_text = diff.to_patch_format();
//! let parsed = parse_patch(&patch_text).expect("valid patch");
//! assert_eq!(parsed.added.len(), 1);
//! assert_eq!(parsed.removed.len(), 1);
//! ```

use crate::parser::{NTriplesLiteParser, ParseError as NtParseError};
use crate::writer::RdfTerm;
use std::collections::HashSet;

/// An N-Triple as used throughout this module
pub type NTriple = (RdfTerm, RdfTerm, RdfTerm);

// ─── Error type ─────────────────────────────────────────────────────────────

/// Errors produced when parsing a patch document
#[derive(Debug, Clone)]
pub struct PatchParseError {
    /// 1-based line number
    pub line: usize,
    /// Human-readable description
    pub message: String,
}

impl PatchParseError {
    fn new(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for PatchParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "patch parse error at line {}: {}",
            self.line, self.message
        )
    }
}

impl std::error::Error for PatchParseError {}

impl From<NtParseError> for PatchParseError {
    fn from(e: NtParseError) -> Self {
        Self {
            line: e.line,
            message: e.message,
        }
    }
}

// ─── RdfDiff ─────────────────────────────────────────────────────────────────

/// The set difference between two RDF graphs
///
/// `added` contains triples present in `after` but not `before`;
/// `removed` contains triples present in `before` but not `after`.
#[derive(Debug, Clone, PartialEq)]
pub struct RdfDiff {
    /// Triples that were added (present in `after` graph, absent from `before`)
    pub added: Vec<NTriple>,
    /// Triples that were removed (present in `before` graph, absent from `after`)
    pub removed: Vec<NTriple>,
}

impl RdfDiff {
    /// Construct a diff with given additions and removals
    pub fn new(added: Vec<NTriple>, removed: Vec<NTriple>) -> Self {
        Self { added, removed }
    }

    /// Return `true` when no triples were added or removed
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }

    /// Total number of changed triples (additions + removals)
    pub fn triple_count(&self) -> usize {
        self.added.len() + self.removed.len()
    }

    /// Apply this diff to a mutable graph in place.
    ///
    /// Removed triples are deleted, then added triples are appended.  The
    /// operation is idempotent: applying the same diff twice leaves the graph
    /// in the same state as applying it once.
    pub fn apply(&self, triples: &mut Vec<NTriple>) {
        // Remove triples that should be deleted
        let remove_set: HashSet<&NTriple> = self.removed.iter().collect();
        triples.retain(|t| !remove_set.contains(t));

        // Add new triples (skip duplicates) – collect additions first to avoid
        // a simultaneous mutable/immutable borrow of `triples`.
        let existing: HashSet<NTriple> = triples.iter().cloned().collect();
        let to_add: Vec<NTriple> = self
            .added
            .iter()
            .filter(|t| !existing.contains(t))
            .cloned()
            .collect();
        triples.extend(to_add);
    }

    /// Return the inverse diff (swap `added` and `removed`).
    ///
    /// Applying the inverse of a diff to the "after" graph yields the
    /// "before" graph.
    pub fn invert(&self) -> Self {
        Self {
            added: self.removed.clone(),
            removed: self.added.clone(),
        }
    }

    /// Serialise the diff to the simple patch format.
    ///
    /// Each removed triple is emitted as a line starting with `- ` followed
    /// by an N-Triples representation; each added triple starts with `+ `.
    /// A comment header records the diff statistics.
    pub fn to_patch_format(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!(
            "# RDF diff: +{} -{}\n",
            self.added.len(),
            self.removed.len()
        ));

        for triple in &self.removed {
            out.push_str("- ");
            out.push_str(&triple_to_ntriples(triple));
            out.push('\n');
        }

        for triple in &self.added {
            out.push_str("+ ");
            out.push_str(&triple_to_ntriples(triple));
            out.push('\n');
        }

        out
    }
}

// ─── Public functions ────────────────────────────────────────────────────────

/// Compute the diff between two RDF graphs
///
/// Both slices are treated as sets: duplicate triples within a single graph
/// are ignored.
pub fn compute_diff(before: &[NTriple], after: &[NTriple]) -> RdfDiff {
    let set_before: HashSet<&NTriple> = before.iter().collect();
    let set_after: HashSet<&NTriple> = after.iter().collect();

    let mut added: Vec<NTriple> = set_after
        .difference(&set_before)
        .map(|t| (*t).clone())
        .collect();
    let mut removed: Vec<NTriple> = set_before
        .difference(&set_after)
        .map(|t| (*t).clone())
        .collect();

    // Sort for deterministic output
    added.sort();
    removed.sort();

    RdfDiff { added, removed }
}

/// Parse a patch document produced by [`RdfDiff::to_patch_format`]
///
/// Lines starting with `+` or `- ` are interpreted as N-Triples additions or
/// removals respectively.  Lines starting with `#` and blank lines are skipped.
pub fn parse_patch(patch: &str) -> Result<RdfDiff, PatchParseError> {
    let mut added: Vec<NTriple> = Vec::new();
    let mut removed: Vec<NTriple> = Vec::new();
    let mut nt_parser = NTriplesLiteParser::new();

    for (line_idx, line) in patch.lines().enumerate() {
        let line_no = line_idx + 1;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("+ ") {
            let triple = parse_single_triple(rest, line_no, &mut nt_parser)?;
            added.push(triple);
        } else if let Some(rest) = trimmed.strip_prefix("- ") {
            let triple = parse_single_triple(rest, line_no, &mut nt_parser)?;
            removed.push(triple);
        } else {
            return Err(PatchParseError::new(
                line_no,
                format!("line must start with '+ ' or '- ', found: {trimmed}"),
            ));
        }
    }

    Ok(RdfDiff { added, removed })
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Serialize one triple as a single N-Triples line (without trailing newline)
fn triple_to_ntriples(triple: &NTriple) -> String {
    format!("{} {} {} .", triple.0, triple.1, triple.2)
}

/// Parse a single triple from an N-Triples line (used for patch parsing)
fn parse_single_triple(
    line: &str,
    line_no: usize,
    parser: &mut NTriplesLiteParser,
) -> Result<NTriple, PatchParseError> {
    parser.reset();
    let mut triples = parser
        .parse_str(line)
        .map_err(|e| PatchParseError::new(line_no, e.message))?;

    match triples.len() {
        0 => Err(PatchParseError::new(
            line_no,
            "expected a triple but line was empty",
        )),
        1 => Ok(triples.remove(0)),
        _ => Err(PatchParseError::new(
            line_no,
            "more than one triple on a patch line",
        )),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::RdfTerm;

    fn s() -> RdfTerm {
        RdfTerm::iri("http://example.org/s")
    }
    fn p() -> RdfTerm {
        RdfTerm::iri("http://example.org/p")
    }
    fn o1() -> RdfTerm {
        RdfTerm::iri("http://example.org/o1")
    }
    fn o2() -> RdfTerm {
        RdfTerm::iri("http://example.org/o2")
    }

    fn triple(s: RdfTerm, p: RdfTerm, o: RdfTerm) -> NTriple {
        (s, p, o)
    }

    // ── compute_diff ────────────────────────────────────────────────────────

    #[test]
    fn test_diff_identical_graphs() {
        let before = vec![triple(s(), p(), o1())];
        let after = before.clone();
        let diff = compute_diff(&before, &after);
        assert!(diff.is_empty());
        assert_eq!(diff.triple_count(), 0);
    }

    #[test]
    fn test_diff_addition() {
        let before: Vec<NTriple> = vec![];
        let after = vec![triple(s(), p(), o1())];
        let diff = compute_diff(&before, &after);
        assert_eq!(diff.added.len(), 1);
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn test_diff_removal() {
        let before = vec![triple(s(), p(), o1())];
        let after: Vec<NTriple> = vec![];
        let diff = compute_diff(&before, &after);
        assert!(diff.added.is_empty());
        assert_eq!(diff.removed.len(), 1);
    }

    #[test]
    fn test_diff_replacement() {
        let before = vec![triple(s(), p(), o1())];
        let after = vec![triple(s(), p(), o2())];
        let diff = compute_diff(&before, &after);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.added[0].2.value, "http://example.org/o2");
        assert_eq!(diff.removed[0].2.value, "http://example.org/o1");
    }

    #[test]
    fn test_diff_duplicates_treated_as_set() {
        // Providing the same triple twice in 'before' should be equivalent to
        // providing it once.
        let before = vec![triple(s(), p(), o1()), triple(s(), p(), o1())];
        let after = vec![triple(s(), p(), o2())];
        let diff = compute_diff(&before, &after);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.removed.len(), 1);
    }

    // ── apply ───────────────────────────────────────────────────────────────

    #[test]
    fn test_apply_roundtrip() {
        let before = vec![triple(s(), p(), o1())];
        let after = vec![triple(s(), p(), o2())];
        let diff = compute_diff(&before, &after);

        let mut graph = before.clone();
        diff.apply(&mut graph);

        // Sort both for comparison
        let mut graph_sorted = graph.clone();
        let mut after_sorted = after.clone();
        graph_sorted.sort();
        after_sorted.sort();

        assert_eq!(graph_sorted, after_sorted);
    }

    #[test]
    fn test_apply_idempotent() {
        let before = vec![triple(s(), p(), o1())];
        let after = vec![triple(s(), p(), o2())];
        let diff = compute_diff(&before, &after);

        let mut graph = before.clone();
        diff.apply(&mut graph);
        diff.apply(&mut graph); // second application must be a no-op

        let mut graph_sorted = graph.clone();
        let mut after_sorted = after.clone();
        graph_sorted.sort();
        after_sorted.sort();

        assert_eq!(graph_sorted, after_sorted);
    }

    #[test]
    fn test_apply_empty_diff() {
        let before = vec![triple(s(), p(), o1())];
        let diff = RdfDiff::new(vec![], vec![]);
        let mut graph = before.clone();
        diff.apply(&mut graph);
        assert_eq!(graph, before);
    }

    // ── invert ──────────────────────────────────────────────────────────────

    #[test]
    fn test_invert_roundtrip() {
        let before = vec![triple(s(), p(), o1())];
        let after = vec![triple(s(), p(), o2())];
        let diff = compute_diff(&before, &after);
        let inv = diff.invert();

        // Applying the inverse to 'after' should restore 'before'
        let mut graph = after.clone();
        inv.apply(&mut graph);

        let mut graph_sorted = graph.clone();
        let mut before_sorted = before.clone();
        graph_sorted.sort();
        before_sorted.sort();

        assert_eq!(graph_sorted, before_sorted);
    }

    // ── patch format ────────────────────────────────────────────────────────

    #[test]
    fn test_patch_format_roundtrip() {
        let before = vec![triple(s(), p(), o1())];
        let after = vec![triple(s(), p(), o2())];
        let diff = compute_diff(&before, &after);

        let patch = diff.to_patch_format();

        // Patch must contain addition and removal markers
        assert!(patch.contains("+ "), "missing '+' marker");
        assert!(patch.contains("- "), "missing '-' marker");
        assert!(patch.contains("# RDF diff:"), "missing header");

        let parsed = parse_patch(&patch).expect("patch must parse successfully");
        assert_eq!(parsed.added.len(), diff.added.len());
        assert_eq!(parsed.removed.len(), diff.removed.len());
    }

    #[test]
    fn test_patch_format_empty_diff() {
        let diff = RdfDiff::new(vec![], vec![]);
        let patch = diff.to_patch_format();
        let parsed = parse_patch(&patch).expect("empty patch parses");
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_patch_format_only_additions() {
        let before: Vec<NTriple> = vec![];
        let after = vec![triple(s(), p(), o1())];
        let diff = compute_diff(&before, &after);
        let patch = diff.to_patch_format();
        assert!(patch.contains("+ "));
        assert!(!patch.contains("- "));

        let parsed = parse_patch(&patch).expect("parse should succeed");
        assert_eq!(parsed.added.len(), 1);
        assert!(parsed.removed.is_empty());
    }

    #[test]
    fn test_patch_format_only_removals() {
        let before = vec![triple(s(), p(), o1())];
        let after: Vec<NTriple> = vec![];
        let diff = compute_diff(&before, &after);
        let patch = diff.to_patch_format();
        assert!(!patch.contains("+ "));
        assert!(patch.contains("- "));

        let parsed = parse_patch(&patch).expect("parse should succeed");
        assert!(parsed.added.is_empty());
        assert_eq!(parsed.removed.len(), 1);
    }

    #[test]
    fn test_patch_invalid_prefix() {
        let bad_patch = "? <http://s> <http://p> <http://o> .\n";
        let result = parse_patch(bad_patch);
        assert!(result.is_err(), "invalid prefix should fail");
    }

    #[test]
    fn test_patch_with_literal() {
        let before = vec![triple(s(), p(), RdfTerm::simple_literal("old"))];
        let after = vec![triple(s(), p(), RdfTerm::simple_literal("new"))];
        let diff = compute_diff(&before, &after);
        let patch = diff.to_patch_format();
        let parsed = parse_patch(&patch).expect("literal patch parses");
        assert_eq!(parsed.added.len(), 1);
        assert_eq!(parsed.removed.len(), 1);
        assert_eq!(parsed.added[0].2.value, "new");
        assert_eq!(parsed.removed[0].2.value, "old");
    }

    #[test]
    fn test_patch_apply_after_parse() {
        let before = vec![triple(s(), p(), o1())];
        let after = vec![triple(s(), p(), o2())];
        let diff = compute_diff(&before, &after);
        let patch_text = diff.to_patch_format();
        let parsed_diff = parse_patch(&patch_text).expect("parse should succeed");

        let mut graph = before.clone();
        parsed_diff.apply(&mut graph);

        let mut graph_sorted = graph.clone();
        let mut after_sorted = after.clone();
        graph_sorted.sort();
        after_sorted.sort();

        assert_eq!(graph_sorted, after_sorted);
    }
}
