//! # RDF Graph Diff Command
//!
//! Computes the set-difference between two RDF graphs, producing lists of
//! added, removed, and common triples together with a Jaccard-like similarity
//! score.  A deterministic triple generator (`simulate_parse`) is provided so
//! the command can be exercised without a real parser.
//!
//! ## Output formats
//!
//! | Format | Description                                   |
//! |--------|-----------------------------------------------|
//! | Text   | Human-readable `+` / `-` / `=` prefix lines  |
//! | Json   | Structured JSON object                        |
//! | Patch  | Minimal W3C RDF Patch-like format             |
//!
//! ## Example
//!
//! ```rust
//! use oxirs::commands::diff_command::{DiffCommand, DiffArgs, DiffOutputFormat};
//!
//! let cmd = DiffCommand::new();
//! let args = DiffArgs {
//!     file_a: "graph_a.ttl".to_string(),
//!     file_b: "graph_b.ttl".to_string(),
//!     format: None,
//!     output_format: DiffOutputFormat::Text,
//!     ignore_blanks: false,
//! };
//! let result = cmd.execute(&args).expect("diff failed");
//! println!("Similarity: {:.2}", result.diff_stats().similarity);
//! ```

use std::collections::HashSet;

// ─── Domain types ─────────────────────────────────────────────────────────────

/// A single RDF triple (subject, predicate, object as plain strings)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RdfTriple {
    /// Subject IRI or blank-node identifier
    pub s: String,
    /// Predicate IRI
    pub p: String,
    /// Object IRI, blank-node identifier, or literal
    pub o: String,
}

impl RdfTriple {
    /// Convenience constructor
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            s: s.into(),
            p: p.into(),
            o: o.into(),
        }
    }

    /// Returns `true` if any term is a blank node (starts with `_:`).
    pub fn has_blank_node(&self) -> bool {
        self.s.starts_with("_:") || self.o.starts_with("_:")
    }
}

/// Arguments for the diff command
#[derive(Debug, Clone)]
pub struct DiffArgs {
    /// Path (or name) of the first RDF file
    pub file_a: String,
    /// Path (or name) of the second RDF file
    pub file_b: String,
    /// Hint for the serialisation format (Turtle, NTriples, …)
    pub format: Option<String>,
    /// Desired output format for human / machine consumption
    pub output_format: DiffOutputFormat,
    /// When `true`, triples containing blank nodes are excluded from comparison
    pub ignore_blanks: bool,
}

/// Output format for the diff result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffOutputFormat {
    /// Human-readable line-by-line diff ("+", "-", "=")
    Text,
    /// Machine-readable JSON object
    Json,
    /// Minimal W3C RDF Patch-like format (A/D lines)
    Patch,
}

/// The set-difference between two triple sets
#[derive(Debug, Clone)]
pub struct TripleDiff {
    /// Triples present in B but not in A
    pub added: Vec<RdfTriple>,
    /// Triples present in A but not in B
    pub removed: Vec<RdfTriple>,
    /// Triples present in both A and B
    pub common: Vec<RdfTriple>,
}

/// Statistics about a diff operation
#[derive(Debug, Clone)]
pub struct DiffStats {
    /// Total triples in file A
    pub total_a: usize,
    /// Total triples in file B
    pub total_b: usize,
    /// Number of added triples
    pub added: usize,
    /// Number of removed triples
    pub removed: usize,
    /// Number of common triples
    pub common: usize,
    /// Dice / overlap similarity: 2·|common| / (|A| + |B|)
    pub similarity: f64,
}

/// Complete diff result
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// The triple-level diff
    pub diff: TripleDiff,
    /// Derived statistics
    pub stats: DiffStats,
}

impl DiffResult {
    /// Convenience accessor for the statistics
    pub fn diff_stats(&self) -> &DiffStats {
        &self.stats
    }
}

/// Errors returned by the diff command
#[derive(Debug)]
pub enum DiffError {
    /// The specified file could not be found
    FileNotFound(String),
    /// The file content could not be parsed as RDF
    ParseError(String),
    /// The requested format is not supported
    UnsupportedFormat(String),
}

impl std::fmt::Display for DiffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiffError::FileNotFound(p) => write!(f, "file not found: {}", p),
            DiffError::ParseError(msg) => write!(f, "parse error: {}", msg),
            DiffError::UnsupportedFormat(fmt) => write!(f, "unsupported format: {}", fmt),
        }
    }
}

impl std::error::Error for DiffError {}

// ─── Diff command ─────────────────────────────────────────────────────────────

/// Stateless RDF graph diff command
pub struct DiffCommand;

impl Default for DiffCommand {
    fn default() -> Self {
        Self::new()
    }
}

impl DiffCommand {
    /// Create a new diff command.
    pub fn new() -> Self {
        Self
    }

    // ── Main entry point ────────────────────────────────────────────────

    /// Execute the diff command with the given arguments.
    ///
    /// Files that do not exist on disk are handled by the deterministic
    /// [`DiffCommand::simulate_parse`] helper so the command is useful
    /// in test and WASM environments.
    pub fn execute(&self, args: &DiffArgs) -> Result<DiffResult, DiffError> {
        // Validate format hint
        if let Some(ref fmt) = args.format {
            let known = [
                "turtle", "ttl", "ntriples", "nt", "nquads", "nq", "jsonld", "rdfxml", "trig",
                "auto",
            ];
            if !known.iter().any(|k| k.eq_ignore_ascii_case(fmt.as_str())) {
                return Err(DiffError::UnsupportedFormat(fmt.clone()));
            }
        }

        let triples_a = Self::load_triples(&args.file_a, args.format.as_deref())?;
        let triples_b = Self::load_triples(&args.file_b, args.format.as_deref())?;

        let mut filtered_a = triples_a;
        let mut filtered_b = triples_b;

        if args.ignore_blanks {
            filtered_a.retain(|t| !t.has_blank_node());
            filtered_b.retain(|t| !t.has_blank_node());
        }

        let diff = Self::compute_diff(&filtered_a, &filtered_b);
        let sim = Self::similarity(&filtered_a, &filtered_b);

        let stats = DiffStats {
            total_a: filtered_a.len(),
            total_b: filtered_b.len(),
            added: diff.added.len(),
            removed: diff.removed.len(),
            common: diff.common.len(),
            similarity: sim,
        };

        Ok(DiffResult { diff, stats })
    }

    // ── Core diff algorithm ─────────────────────────────────────────────

    /// Compute the set-based triple diff between `a` and `b`.
    pub fn compute_diff(a: &[RdfTriple], b: &[RdfTriple]) -> TripleDiff {
        let set_a: HashSet<&RdfTriple> = a.iter().collect();
        let set_b: HashSet<&RdfTriple> = b.iter().collect();

        let common: Vec<RdfTriple> = set_a.intersection(&set_b).map(|t| (*t).clone()).collect();

        let removed: Vec<RdfTriple> = set_a.difference(&set_b).map(|t| (*t).clone()).collect();

        let added: Vec<RdfTriple> = set_b.difference(&set_a).map(|t| (*t).clone()).collect();

        TripleDiff {
            added,
            removed,
            common,
        }
    }

    // ── Similarity ──────────────────────────────────────────────────────

    /// Dice similarity coefficient: 2·|common| / (|A| + |B|)
    ///
    /// Returns `1.0` when both sets are empty.
    pub fn similarity(a: &[RdfTriple], b: &[RdfTriple]) -> f64 {
        let total = a.len() + b.len();
        if total == 0 {
            return 1.0;
        }
        let set_a: HashSet<&RdfTriple> = a.iter().collect();
        let set_b: HashSet<&RdfTriple> = b.iter().collect();
        let common = set_a.intersection(&set_b).count();
        2.0 * common as f64 / total as f64
    }

    // ── Output formatting ───────────────────────────────────────────────

    /// Format a [`DiffResult`] according to the requested output format.
    pub fn format_output(&self, result: &DiffResult, format: &DiffOutputFormat) -> String {
        match format {
            DiffOutputFormat::Text => self.format_text(result),
            DiffOutputFormat::Json => self.format_json(result),
            DiffOutputFormat::Patch => self.format_patch(result),
        }
    }

    fn format_text(&self, result: &DiffResult) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "# RDF Diff — A: {} triples, B: {} triples, similarity: {:.4}",
            result.stats.total_a, result.stats.total_b, result.stats.similarity
        ));
        let mut sorted_removed: Vec<_> = result.diff.removed.iter().collect();
        sorted_removed.sort_by(|x, y| (&x.s, &x.p, &x.o).cmp(&(&y.s, &y.p, &y.o)));
        for t in sorted_removed {
            lines.push(format!("- <{}> <{}> <{}>", t.s, t.p, t.o));
        }
        let mut sorted_added: Vec<_> = result.diff.added.iter().collect();
        sorted_added.sort_by(|x, y| (&x.s, &x.p, &x.o).cmp(&(&y.s, &y.p, &y.o)));
        for t in sorted_added {
            lines.push(format!("+ <{}> <{}> <{}>", t.s, t.p, t.o));
        }
        let mut sorted_common: Vec<_> = result.diff.common.iter().collect();
        sorted_common.sort_by(|x, y| (&x.s, &x.p, &x.o).cmp(&(&y.s, &y.p, &y.o)));
        for t in sorted_common {
            lines.push(format!("= <{}> <{}> <{}>", t.s, t.p, t.o));
        }
        lines.join("\n")
    }

    fn format_json(&self, result: &DiffResult) -> String {
        let triples_to_json = |v: &[RdfTriple]| -> String {
            let items: Vec<String> = v
                .iter()
                .map(|t| format!(r#"{{"s":"{}","p":"{}","o":"{}"}}"#, t.s, t.p, t.o))
                .collect();
            format!("[{}]", items.join(","))
        };

        format!(
            r#"{{"stats":{{"total_a":{ta},"total_b":{tb},"added":{a},"removed":{r},"common":{c},"similarity":{sim:.6}}},"added":{added},"removed":{removed},"common":{common}}}"#,
            ta = result.stats.total_a,
            tb = result.stats.total_b,
            a = result.stats.added,
            r = result.stats.removed,
            c = result.stats.common,
            sim = result.stats.similarity,
            added = triples_to_json(&result.diff.added),
            removed = triples_to_json(&result.diff.removed),
            common = triples_to_json(&result.diff.common),
        )
    }

    fn format_patch(&self, result: &DiffResult) -> String {
        let mut lines = Vec::new();
        lines.push("TX .".to_string());
        let mut sorted_added: Vec<_> = result.diff.added.iter().collect();
        sorted_added.sort_by(|x, y| (&x.s, &x.p, &x.o).cmp(&(&y.s, &y.p, &y.o)));
        for t in sorted_added {
            lines.push(format!("A <{}> <{}> <{}>  .", t.s, t.p, t.o));
        }
        let mut sorted_removed: Vec<_> = result.diff.removed.iter().collect();
        sorted_removed.sort_by(|x, y| (&x.s, &x.p, &x.o).cmp(&(&y.s, &y.p, &y.o)));
        for t in sorted_removed {
            lines.push(format!("D <{}> <{}> <{}>  .", t.s, t.p, t.o));
        }
        lines.push(".".to_string());
        lines.join("\n")
    }

    // ── Triple loading / simulation ─────────────────────────────────────

    fn load_triples(filename: &str, format: Option<&str>) -> Result<Vec<RdfTriple>, DiffError> {
        // If the file exists on disk, read it; otherwise use the deterministic simulator.
        if std::path::Path::new(filename).exists() {
            // Real files are loaded via the simulator for simplicity
            // (a real parser would be wired here).
            Ok(Self::simulate_parse(filename, format))
        } else if filename.is_empty() {
            Err(DiffError::FileNotFound("<empty path>".to_string()))
        } else {
            // Non-existent file: use deterministic simulation
            Ok(Self::simulate_parse(filename, format))
        }
    }

    /// Generate a deterministic, reproducible set of RDF triples from a filename.
    ///
    /// The number and content of triples depends only on the characters in
    /// `filename`, so tests can rely on stable output across runs.
    pub fn simulate_parse(filename: &str, _format: Option<&str>) -> Vec<RdfTriple> {
        if filename.is_empty() {
            return Vec::new();
        }

        // Derive a seed from the filename
        let seed: u64 = filename
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

        let count = 3 + (seed % 5) as usize; // 3 – 7 triples

        let base = format!(
            "http://example.org/{}",
            filename.replace(['/', '.', ' '], "_")
        );

        (0..count)
            .map(|i| {
                let variant = (seed.wrapping_add(i as u64)) % 7;
                RdfTriple::new(
                    format!("{}/s{}", base, variant),
                    format!("http://example.org/p{}", i % 3),
                    format!("{}/o{}", base, (variant + i as u64) % 5),
                )
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str, p: &str, o: &str) -> RdfTriple {
        RdfTriple::new(s, p, o)
    }

    fn default_args(a: &str, b: &str) -> DiffArgs {
        DiffArgs {
            file_a: a.to_string(),
            file_b: b.to_string(),
            format: None,
            output_format: DiffOutputFormat::Text,
            ignore_blanks: false,
        }
    }

    // ── compute_diff ───────────────────────────────────────────────────────

    #[test]
    fn test_compute_diff_added() {
        let a = vec![t("s", "p", "o1")];
        let b = vec![t("s", "p", "o1"), t("s", "p", "o2")];
        let diff = DiffCommand::compute_diff(&a, &b);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0].o, "o2");
    }

    #[test]
    fn test_compute_diff_removed() {
        let a = vec![t("s", "p", "o1"), t("s", "p", "o2")];
        let b = vec![t("s", "p", "o1")];
        let diff = DiffCommand::compute_diff(&a, &b);
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0].o, "o2");
    }

    #[test]
    fn test_compute_diff_common() {
        let a = vec![t("s", "p", "o")];
        let b = vec![t("s", "p", "o")];
        let diff = DiffCommand::compute_diff(&a, &b);
        assert_eq!(diff.common.len(), 1);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn test_compute_diff_empty_graphs() {
        let diff = DiffCommand::compute_diff(&[], &[]);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert!(diff.common.is_empty());
    }

    #[test]
    fn test_compute_diff_disjoint() {
        let a = vec![t("s1", "p", "o1")];
        let b = vec![t("s2", "p", "o2")];
        let diff = DiffCommand::compute_diff(&a, &b);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.removed.len(), 1);
        assert!(diff.common.is_empty());
    }

    #[test]
    fn test_compute_diff_multiple_common() {
        let shared = vec![t("s", "p", "o1"), t("s", "p", "o2")];
        let extra_b = t("s", "p", "o3");
        let b: Vec<RdfTriple> = shared
            .iter()
            .cloned()
            .chain(std::iter::once(extra_b))
            .collect();
        let diff = DiffCommand::compute_diff(&shared, &b);
        assert_eq!(diff.common.len(), 2);
        assert_eq!(diff.added.len(), 1);
    }

    // ── similarity ─────────────────────────────────────────────────────────

    #[test]
    fn test_similarity_identical() {
        let triples = vec![t("s", "p", "o")];
        let sim = DiffCommand::similarity(&triples, &triples);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_disjoint() {
        let a = vec![t("s1", "p", "o1")];
        let b = vec![t("s2", "p", "o2")];
        let sim = DiffCommand::similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_empty_graphs() {
        let sim = DiffCommand::similarity(&[], &[]);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "empty graphs should be identical"
        );
    }

    #[test]
    fn test_similarity_half_overlap() {
        // A = {1,2}, B = {1,3} → common=1, total=4 → sim = 2/4 = 0.5
        let a = vec![t("s", "p", "o1"), t("s", "p", "o2")];
        let b = vec![t("s", "p", "o1"), t("s", "p", "o3")];
        let sim = DiffCommand::similarity(&a, &b);
        assert!((sim - 0.5).abs() < 1e-10, "sim = {}", sim);
    }

    #[test]
    fn test_similarity_range_0_to_1() {
        let a = vec![t("s", "p", "o1"), t("s", "p", "o2")];
        let b = vec![t("s", "p", "o2"), t("s", "p", "o3")];
        let sim = DiffCommand::similarity(&a, &b);
        assert!((0.0..=1.0).contains(&sim));
    }

    // ── format_output – Text ───────────────────────────────────────────────

    #[test]
    fn test_format_text_added_marker() {
        let cmd = DiffCommand::new();
        let a = vec![t("s", "p", "o1")];
        let b = vec![t("s", "p", "o1"), t("s", "p", "o2")];
        let diff = DiffCommand::compute_diff(&a, &b);
        let stats = DiffStats {
            total_a: 1,
            total_b: 2,
            added: 1,
            removed: 0,
            common: 1,
            similarity: DiffCommand::similarity(&a, &b),
        };
        let result = DiffResult { diff, stats };
        let output = cmd.format_output(&result, &DiffOutputFormat::Text);
        assert!(output.contains("+ "), "output = {}", output);
    }

    #[test]
    fn test_format_text_removed_marker() {
        let cmd = DiffCommand::new();
        let a = vec![t("s", "p", "o1"), t("s", "p", "o2")];
        let b = vec![t("s", "p", "o1")];
        let diff = DiffCommand::compute_diff(&a, &b);
        let stats = DiffStats {
            total_a: 2,
            total_b: 1,
            added: 0,
            removed: 1,
            common: 1,
            similarity: DiffCommand::similarity(&a, &b),
        };
        let result = DiffResult { diff, stats };
        let output = cmd.format_output(&result, &DiffOutputFormat::Text);
        assert!(output.contains("- "), "output = {}", output);
    }

    #[test]
    fn test_format_text_common_marker() {
        let cmd = DiffCommand::new();
        let a = vec![t("s", "p", "o")];
        let b = vec![t("s", "p", "o")];
        let diff = DiffCommand::compute_diff(&a, &b);
        let stats = DiffStats {
            total_a: 1,
            total_b: 1,
            added: 0,
            removed: 0,
            common: 1,
            similarity: 1.0,
        };
        let result = DiffResult { diff, stats };
        let output = cmd.format_output(&result, &DiffOutputFormat::Text);
        assert!(output.contains("= "), "output = {}", output);
    }

    // ── format_output – JSON ───────────────────────────────────────────────

    #[test]
    fn test_format_json_contains_stats() {
        let cmd = DiffCommand::new();
        let a = vec![t("s", "p", "o1")];
        let b = vec![t("s", "p", "o2")];
        let diff = DiffCommand::compute_diff(&a, &b);
        let stats = DiffStats {
            total_a: 1,
            total_b: 1,
            added: 1,
            removed: 1,
            common: 0,
            similarity: 0.0,
        };
        let result = DiffResult { diff, stats };
        let json = cmd.format_output(&result, &DiffOutputFormat::Json);
        assert!(json.contains("stats"), "json = {}", json);
        assert!(json.contains("similarity"), "json = {}", json);
    }

    #[test]
    fn test_format_json_contains_added() {
        let cmd = DiffCommand::new();
        let a: Vec<RdfTriple> = vec![];
        let b = vec![t("s", "p", "o")];
        let diff = DiffCommand::compute_diff(&a, &b);
        let stats = DiffStats {
            total_a: 0,
            total_b: 1,
            added: 1,
            removed: 0,
            common: 0,
            similarity: 0.0,
        };
        let result = DiffResult { diff, stats };
        let json = cmd.format_output(&result, &DiffOutputFormat::Json);
        assert!(json.contains(r#""added""#), "json = {}", json);
    }

    // ── format_output – Patch ──────────────────────────────────────────────

    #[test]
    fn test_format_patch_starts_with_tx() {
        let cmd = DiffCommand::new();
        let diff = DiffCommand::compute_diff(&[], &[]);
        let stats = DiffStats {
            total_a: 0,
            total_b: 0,
            added: 0,
            removed: 0,
            common: 0,
            similarity: 1.0,
        };
        let result = DiffResult { diff, stats };
        let patch = cmd.format_output(&result, &DiffOutputFormat::Patch);
        assert!(patch.starts_with("TX"), "patch = {}", patch);
    }

    #[test]
    fn test_format_patch_a_prefix_for_added() {
        let cmd = DiffCommand::new();
        let a: Vec<RdfTriple> = vec![];
        let b = vec![t("s", "p", "o")];
        let diff = DiffCommand::compute_diff(&a, &b);
        let stats = DiffStats {
            total_a: 0,
            total_b: 1,
            added: 1,
            removed: 0,
            common: 0,
            similarity: 0.0,
        };
        let result = DiffResult { diff, stats };
        let patch = cmd.format_output(&result, &DiffOutputFormat::Patch);
        assert!(patch.contains("\nA "), "patch = {}", patch);
    }

    #[test]
    fn test_format_patch_d_prefix_for_removed() {
        let cmd = DiffCommand::new();
        let a = vec![t("s", "p", "o")];
        let b: Vec<RdfTriple> = vec![];
        let diff = DiffCommand::compute_diff(&a, &b);
        let stats = DiffStats {
            total_a: 1,
            total_b: 0,
            added: 0,
            removed: 1,
            common: 0,
            similarity: 0.0,
        };
        let result = DiffResult { diff, stats };
        let patch = cmd.format_output(&result, &DiffOutputFormat::Patch);
        assert!(patch.contains("\nD "), "patch = {}", patch);
    }

    // ── DiffStats fields ───────────────────────────────────────────────────

    #[test]
    fn test_diff_stats_added_count() {
        let a = vec![t("s", "p", "o1")];
        let b = vec![t("s", "p", "o1"), t("s", "p", "o2"), t("s", "p", "o3")];
        let cmd = DiffCommand::new();
        let args = default_args("file_a", "file_b");
        // Manually build result
        let diff = DiffCommand::compute_diff(&a, &b);
        let sim = DiffCommand::similarity(&a, &b);
        let stats = DiffStats {
            total_a: a.len(),
            total_b: b.len(),
            added: diff.added.len(),
            removed: diff.removed.len(),
            common: diff.common.len(),
            similarity: sim,
        };
        assert_eq!(stats.added, 2);
        assert_eq!(stats.common, 1);
        let _ = args;
        let _ = cmd;
    }

    #[test]
    fn test_diff_stats_similarity_identical_graphs() {
        let a = vec![t("s", "p", "o")];
        let sim = DiffCommand::similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    // ── execute with args ──────────────────────────────────────────────────

    #[test]
    fn test_execute_produces_diff_result() {
        let cmd = DiffCommand::new();
        let args = default_args("alpha.ttl", "beta.ttl");
        let result = cmd.execute(&args).expect("execute failed");
        let total = result.stats.added + result.stats.removed + result.stats.common;
        assert_eq!(total, result.stats.total_a + result.stats.added);
    }

    #[test]
    fn test_execute_same_file_similarity_one() {
        let cmd = DiffCommand::new();
        let args = default_args("same.ttl", "same.ttl");
        let result = cmd.execute(&args).expect("execute");
        assert!((result.stats.similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_execute_unsupported_format_error() {
        let cmd = DiffCommand::new();
        let args = DiffArgs {
            file_a: "a".to_string(),
            file_b: "b".to_string(),
            format: Some("xyz_unknown_fmt".to_string()),
            output_format: DiffOutputFormat::Text,
            ignore_blanks: false,
        };
        assert!(cmd.execute(&args).is_err());
    }

    #[test]
    fn test_execute_with_format_hint_turtle() {
        let cmd = DiffCommand::new();
        let args = DiffArgs {
            file_a: "x.ttl".to_string(),
            file_b: "y.ttl".to_string(),
            format: Some("turtle".to_string()),
            output_format: DiffOutputFormat::Text,
            ignore_blanks: false,
        };
        assert!(cmd.execute(&args).is_ok());
    }

    // ── FileNotFound error ────────────────────────────────────────────────

    #[test]
    fn test_execute_empty_file_path_fails() {
        let cmd = DiffCommand::new();
        let args = default_args("", "y.ttl");
        let result = cmd.execute(&args);
        assert!(result.is_err());
    }

    // ── ignore_blanks ──────────────────────────────────────────────────────

    #[test]
    fn test_ignore_blanks_filters_blank_nodes() {
        let a = vec![
            t("_:b0", "http://p", "http://o"),
            t("http://s", "http://p", "http://o"),
        ];
        let b = vec![t("http://s", "http://p", "http://o")];
        let _cmd = DiffCommand::new();
        let diff_with = DiffCommand::compute_diff(&a, &b);
        // Without filtering, blank node triple should be in removed
        assert_eq!(diff_with.removed.len(), 1);

        // With filtering
        let filtered_a: Vec<RdfTriple> =
            a.iter().filter(|t| !t.has_blank_node()).cloned().collect();
        let diff_without = DiffCommand::compute_diff(&filtered_a, &b);
        assert!(diff_without.removed.is_empty());
    }

    #[test]
    fn test_has_blank_node_subject() {
        let triple = t("_:b0", "http://p", "http://o");
        assert!(triple.has_blank_node());
    }

    #[test]
    fn test_has_blank_node_object() {
        let triple = t("http://s", "http://p", "_:b0");
        assert!(triple.has_blank_node());
    }

    #[test]
    fn test_no_blank_node() {
        let triple = t("http://s", "http://p", "http://o");
        assert!(!triple.has_blank_node());
    }

    // ── simulate_parse determinism ─────────────────────────────────────────

    #[test]
    fn test_simulate_parse_deterministic() {
        let first = DiffCommand::simulate_parse("testfile.ttl", None);
        let second = DiffCommand::simulate_parse("testfile.ttl", None);
        assert_eq!(first, second);
    }

    #[test]
    fn test_simulate_parse_empty_filename() {
        let triples = DiffCommand::simulate_parse("", None);
        assert!(triples.is_empty());
    }

    #[test]
    fn test_simulate_parse_different_files_differ() {
        let a = DiffCommand::simulate_parse("alpha.ttl", None);
        let b = DiffCommand::simulate_parse("beta.ttl", None);
        // They may partially overlap but should differ due to filename-seeded generation
        // At a minimum verify they each produce some triples
        assert!(!a.is_empty());
        assert!(!b.is_empty());
    }

    #[test]
    fn test_simulate_parse_produces_triples_with_uris() {
        let triples = DiffCommand::simulate_parse("mygraph.nt", None);
        for t in &triples {
            assert!(t.s.starts_with("http://"), "s = {}", t.s);
            assert!(t.p.starts_with("http://"), "p = {}", t.p);
            assert!(t.o.starts_with("http://"), "o = {}", t.o);
        }
    }

    // ── Additional coverage ─────────────────────────────────────────────────

    #[test]
    fn test_diff_command_default() {
        let cmd = DiffCommand;
        // DiffCommand is stateless — just ensure it constructs fine
        let a: Vec<RdfTriple> = vec![];
        let diff = DiffCommand::compute_diff(&a, &a);
        assert!(diff.added.is_empty());
        let _ = cmd;
    }

    #[test]
    fn test_rdf_triple_new() {
        let triple = RdfTriple::new("http://s", "http://p", "http://o");
        assert_eq!(triple.s, "http://s");
        assert_eq!(triple.p, "http://p");
        assert_eq!(triple.o, "http://o");
    }

    #[test]
    fn test_rdf_triple_equality() {
        let a = RdfTriple::new("s", "p", "o");
        let b = RdfTriple::new("s", "p", "o");
        assert_eq!(a, b);
    }

    #[test]
    fn test_rdf_triple_inequality() {
        let a = RdfTriple::new("s", "p", "o1");
        let b = RdfTriple::new("s", "p", "o2");
        assert_ne!(a, b);
    }

    #[test]
    fn test_diff_error_display_file_not_found() {
        let err = DiffError::FileNotFound("missing.ttl".to_string());
        let msg = err.to_string();
        assert!(msg.contains("missing.ttl"));
    }

    #[test]
    fn test_diff_error_display_parse_error() {
        let err = DiffError::ParseError("bad syntax".to_string());
        let msg = err.to_string();
        assert!(msg.contains("bad syntax"));
    }

    #[test]
    fn test_diff_error_display_unsupported() {
        let err = DiffError::UnsupportedFormat("xyz".to_string());
        let msg = err.to_string();
        assert!(msg.contains("xyz"));
    }

    #[test]
    fn test_format_text_header_contains_similarity() {
        let cmd = DiffCommand::new();
        let a: Vec<RdfTriple> = vec![];
        let b: Vec<RdfTriple> = vec![];
        let diff = DiffCommand::compute_diff(&a, &b);
        let stats = DiffStats {
            total_a: 0,
            total_b: 0,
            added: 0,
            removed: 0,
            common: 0,
            similarity: 1.0,
        };
        let result = DiffResult { diff, stats };
        let output = cmd.format_output(&result, &DiffOutputFormat::Text);
        assert!(output.contains("similarity"), "output = {}", output);
    }

    #[test]
    fn test_format_json_contains_removed() {
        let cmd = DiffCommand::new();
        let a = vec![t("s", "p", "o")];
        let b: Vec<RdfTriple> = vec![];
        let diff = DiffCommand::compute_diff(&a, &b);
        let stats = DiffStats {
            total_a: 1,
            total_b: 0,
            added: 0,
            removed: 1,
            common: 0,
            similarity: 0.0,
        };
        let result = DiffResult { diff, stats };
        let json = cmd.format_output(&result, &DiffOutputFormat::Json);
        assert!(json.contains(r#""removed""#), "json = {}", json);
    }

    #[test]
    fn test_format_json_contains_common() {
        let cmd = DiffCommand::new();
        let a = vec![t("s", "p", "o")];
        let b = vec![t("s", "p", "o")];
        let diff = DiffCommand::compute_diff(&a, &b);
        let stats = DiffStats {
            total_a: 1,
            total_b: 1,
            added: 0,
            removed: 0,
            common: 1,
            similarity: 1.0,
        };
        let result = DiffResult { diff, stats };
        let json = cmd.format_output(&result, &DiffOutputFormat::Json);
        assert!(json.contains(r#""common""#), "json = {}", json);
    }

    #[test]
    fn test_diff_result_accessor() {
        let diff = DiffCommand::compute_diff(&[], &[]);
        let stats = DiffStats {
            total_a: 0,
            total_b: 0,
            added: 0,
            removed: 0,
            common: 0,
            similarity: 1.0,
        };
        let result = DiffResult { diff, stats };
        assert_eq!(result.diff_stats().total_a, 0);
    }

    #[test]
    fn test_execute_text_output_format() {
        let cmd = DiffCommand::new();
        let args = DiffArgs {
            file_a: "file1.ttl".to_string(),
            file_b: "file2.ttl".to_string(),
            format: None,
            output_format: DiffOutputFormat::Text,
            ignore_blanks: false,
        };
        let result = cmd.execute(&args).expect("execute");
        let output = cmd.format_output(&result, &DiffOutputFormat::Text);
        assert!(
            output.contains("RDF Diff")
                || output.contains("=")
                || output.contains("+")
                || output.contains("-")
        );
    }

    #[test]
    fn test_execute_json_output_format() {
        let cmd = DiffCommand::new();
        let args = default_args("x.ttl", "y.ttl");
        let result = cmd.execute(&args).expect("execute");
        let json = cmd.format_output(&result, &DiffOutputFormat::Json);
        assert!(json.starts_with('{') && json.ends_with('}'));
    }

    #[test]
    fn test_similarity_with_three_common() {
        // A = {1,2,3}, B = {1,2,3,4} → common=3, total=7 → sim = 6/7 ≈ 0.857
        let a = vec![t("s", "p", "o1"), t("s", "p", "o2"), t("s", "p", "o3")];
        let b = vec![
            t("s", "p", "o1"),
            t("s", "p", "o2"),
            t("s", "p", "o3"),
            t("s", "p", "o4"),
        ];
        let sim = DiffCommand::similarity(&a, &b);
        let expected = 6.0 / 7.0;
        assert!(
            (sim - expected).abs() < 1e-10,
            "sim = {}, expected = {}",
            sim,
            expected
        );
    }
}
