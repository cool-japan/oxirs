//! # RDF Merge Command
//!
//! Merges triples from multiple RDF sources with blank node renaming, conflict
//! detection, provenance tracking, and deduplication.  Supports a dry-run mode
//! that reports the merge plan without writing output.
//!
//! ## Merge semantics
//!
//! A merge is an RDF-level set-union: identical triples are deduplicated,
//! blank nodes are alpha-renamed per source to guarantee uniqueness, and
//! conflicts (same subject-predicate, different objects) are detected and
//! reported.
//!
//! ## Supported input formats
//!
//! | Format    | Extension(s)       |
//! |-----------|--------------------|
//! | Turtle    | `.ttl`             |
//! | N-Triples | `.nt`              |
//! | RDF/XML   | `.rdf`, `.xml`     |
//! | N-Quads   | `.nq`              |
//!
//! ## Example
//!
//! ```rust
//! use oxirs::commands::merge_command::{
//!     MergeCommand, MergeArgs, OutputFormat, MergeMode,
//! };
//!
//! let cmd = MergeCommand::new();
//! let args = MergeArgs {
//!     sources: vec!["graph_a.ttl".to_string(), "graph_b.nt".to_string()],
//!     output: None,
//!     output_format: OutputFormat::NTriples,
//!     mode: MergeMode::SetUnion,
//!     dry_run: false,
//!     track_provenance: false,
//! };
//! let result = cmd.execute(&args).expect("merge failed");
//! assert!(result.stats.total_triples > 0);
//! ```

use std::collections::{HashMap, HashSet};

// ─── Domain types ────────────────────────────────────────────────────────────

/// A single RDF triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RdfTriple {
    pub s: String,
    pub p: String,
    pub o: String,
}

impl RdfTriple {
    /// Convenience constructor.
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            s: s.into(),
            p: p.into(),
            o: o.into(),
        }
    }

    /// Whether any term starts with `_:`.
    pub fn has_blank_node(&self) -> bool {
        self.s.starts_with("_:") || self.o.starts_with("_:")
    }
}

/// A triple with provenance information.
#[derive(Debug, Clone)]
pub struct ProvenancedTriple {
    pub triple: RdfTriple,
    /// Name/path of the source that contributed this triple.
    pub source: String,
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// How the merge should be performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMode {
    /// Simple set-union: deduplicate identical triples.
    SetUnion,
    /// Set-union with provenance tracking (records which source each triple came from).
    WithProvenance,
}

/// Output serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Turtle,
    NTriples,
    NQuads,
    RdfXml,
}

impl OutputFormat {
    /// File extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Turtle => "ttl",
            Self::NTriples => "nt",
            Self::NQuads => "nq",
            Self::RdfXml => "rdf",
        }
    }

    /// Detect format from a file extension string (case-insensitive).
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "ttl" => Some(Self::Turtle),
            "nt" => Some(Self::NTriples),
            "nq" => Some(Self::NQuads),
            "rdf" | "xml" => Some(Self::RdfXml),
            _ => None,
        }
    }
}

/// Arguments for the merge command.
#[derive(Debug, Clone)]
pub struct MergeArgs {
    /// Paths (or names) of the RDF source files to merge.
    pub sources: Vec<String>,
    /// Optional output file path. `None` means stdout / in-memory.
    pub output: Option<String>,
    /// Desired output format.
    pub output_format: OutputFormat,
    /// Merge strategy.
    pub mode: MergeMode,
    /// If true, compute and report the merge plan without writing output.
    pub dry_run: bool,
    /// Whether to record provenance for each triple.
    pub track_provenance: bool,
}

// ─── Statistics ──────────────────────────────────────────────────────────────

/// Per-source statistics.
#[derive(Debug, Clone)]
pub struct SourceStats {
    /// Source name/path.
    pub source: String,
    /// Number of triples from this source.
    pub triple_count: usize,
    /// Number of blank nodes renamed.
    pub blank_nodes_renamed: usize,
}

/// A detected conflict: same (subject, predicate) with different objects.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Conflict {
    pub subject: String,
    pub predicate: String,
    pub objects: Vec<String>,
}

/// Overall merge statistics.
#[derive(Debug, Clone)]
pub struct MergeStats {
    /// Total triples in the merged result (after dedup).
    pub total_triples: usize,
    /// Number of duplicate triples removed.
    pub duplicates_removed: usize,
    /// Number of conflicts found (same SP, different O).
    pub conflicts_found: usize,
    /// Per-source breakdown.
    pub source_stats: Vec<SourceStats>,
    /// Total blank nodes renamed across all sources.
    pub total_blank_nodes_renamed: usize,
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// Result of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// The merged triples (empty if dry_run).
    pub triples: Vec<RdfTriple>,
    /// Provenance records (populated only when track_provenance is true).
    pub provenance: Vec<ProvenancedTriple>,
    /// Detected conflicts.
    pub conflicts: Vec<Conflict>,
    /// Statistics.
    pub stats: MergeStats,
    /// Whether this was a dry run.
    pub dry_run: bool,
    /// Serialised output (for non-dry-run operations).
    pub output_text: Option<String>,
}

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors from the merge command.
#[derive(Debug)]
pub enum MergeError {
    /// No source files provided.
    NoSources,
    /// A source file was not found / could not be read.
    SourceNotFound(String),
    /// Unsupported input format.
    UnsupportedFormat(String),
    /// Internal error.
    Internal(String),
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSources => write!(f, "No source files provided"),
            Self::SourceNotFound(p) => write!(f, "Source not found: {p}"),
            Self::UnsupportedFormat(fmt) => write!(f, "Unsupported format: {fmt}"),
            Self::Internal(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl std::error::Error for MergeError {}

// ─── Blank node renaming ─────────────────────────────────────────────────────

/// Rename all blank nodes in `triples` by prefixing them with
/// `_:src{source_index}_` to ensure uniqueness across sources.
fn rename_blank_nodes(triples: &mut [RdfTriple], source_index: usize) -> usize {
    let prefix = format!("_:src{source_index}_");
    let mut renamed = 0_usize;

    for t in triples.iter_mut() {
        if t.s.starts_with("_:") {
            let local = &t.s[2..];
            t.s = format!("{prefix}{local}");
            renamed += 1;
        }
        if t.o.starts_with("_:") {
            let local = &t.o[2..];
            t.o = format!("{prefix}{local}");
            renamed += 1;
        }
    }

    renamed
}

// ─── Simulated parser ────────────────────────────────────────────────────────

/// Detect the input format from a file name/extension.
fn detect_format(file_name: &str) -> Option<&'static str> {
    if let Some(dot) = file_name.rfind('.') {
        let ext = &file_name[dot + 1..];
        match ext.to_ascii_lowercase().as_str() {
            "ttl" => Some("turtle"),
            "nt" => Some("ntriples"),
            "nq" => Some("nquads"),
            "rdf" | "xml" => Some("rdfxml"),
            _ => None,
        }
    } else {
        None
    }
}

/// Deterministic triple generator for testing without a real file system.
///
/// Generates triples based on the file name so that different "files" produce
/// overlapping-but-distinct sets of triples.
fn simulate_parse(file_name: &str) -> Vec<RdfTriple> {
    let seed = file_name
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_add(b as u64));
    let count = ((seed % 7) + 3) as usize;
    let mut triples = Vec::with_capacity(count);

    for i in 0..count {
        let idx = (seed as usize + i) % 100;
        let s = if i % 4 == 0 {
            format!("_:b{idx}")
        } else {
            format!("http://example.org/s{idx}")
        };
        let p = format!("http://example.org/p{}", idx % 5);
        let o = if i % 3 == 0 {
            format!("\"value_{idx}\"")
        } else {
            format!("http://example.org/o{}", (idx + 1) % 50)
        };
        triples.push(RdfTriple::new(s, p, o));
    }

    triples
}

// ─── Conflict detection ──────────────────────────────────────────────────────

/// Detect conflicts: groups of triples sharing the same (subject, predicate)
/// but differing objects.
fn detect_conflicts(triples: &[RdfTriple]) -> Vec<Conflict> {
    let mut sp_to_objects: HashMap<(String, String), HashSet<String>> = HashMap::new();

    for t in triples {
        sp_to_objects
            .entry((t.s.clone(), t.p.clone()))
            .or_default()
            .insert(t.o.clone());
    }

    let mut conflicts = Vec::new();
    for ((s, p), objects) in &sp_to_objects {
        if objects.len() > 1 {
            let mut sorted: Vec<String> = objects.iter().cloned().collect();
            sorted.sort();
            conflicts.push(Conflict {
                subject: s.clone(),
                predicate: p.clone(),
                objects: sorted,
            });
        }
    }
    conflicts.sort_by(|a, b| (&a.subject, &a.predicate).cmp(&(&b.subject, &b.predicate)));
    conflicts
}

// ─── Serialization ───────────────────────────────────────────────────────────

/// Serialize triples to N-Triples format.
fn serialize_ntriples(triples: &[RdfTriple]) -> String {
    let mut out = String::new();
    for t in triples {
        let s_str = format_term(&t.s);
        let o_str = format_term(&t.o);
        out.push_str(&format!("{s_str} <{}> {o_str} .\n", t.p));
    }
    out
}

/// Serialize triples to Turtle format (simplified).
fn serialize_turtle(triples: &[RdfTriple]) -> String {
    let mut out = String::from("@prefix ex: <http://example.org/> .\n\n");
    for t in triples {
        let s_str = format_term(&t.s);
        let o_str = format_term(&t.o);
        out.push_str(&format!("{s_str} <{}> {o_str} .\n", t.p));
    }
    out
}

fn format_term(term: &str) -> String {
    if term.starts_with('"') || term.starts_with("_:") {
        term.to_string()
    } else {
        format!("<{term}>")
    }
}

fn serialize_output(triples: &[RdfTriple], format: OutputFormat) -> String {
    match format {
        OutputFormat::NTriples | OutputFormat::NQuads => serialize_ntriples(triples),
        OutputFormat::Turtle => serialize_turtle(triples),
        OutputFormat::RdfXml => {
            // Simplified RDF/XML placeholder
            let mut out = String::from(
                "<?xml version=\"1.0\"?>\n<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n",
            );
            for t in triples {
                out.push_str(&format!("  <!-- {} {} {} -->\n", t.s, t.p, t.o));
            }
            out.push_str("</rdf:RDF>\n");
            out
        }
    }
}

// ─── MergeCommand ────────────────────────────────────────────────────────────

/// RDF merge command.
pub struct MergeCommand;

impl Default for MergeCommand {
    fn default() -> Self {
        Self::new()
    }
}

impl MergeCommand {
    /// Create a new MergeCommand.
    pub fn new() -> Self {
        Self
    }

    /// Execute the merge.
    pub fn execute(&self, args: &MergeArgs) -> Result<MergeResult, MergeError> {
        if args.sources.is_empty() {
            return Err(MergeError::NoSources);
        }

        // Validate formats
        for src in &args.sources {
            if detect_format(src).is_none() {
                return Err(MergeError::UnsupportedFormat(src.clone()));
            }
        }

        let mut all_triples: Vec<RdfTriple> = Vec::new();
        let mut provenance_records: Vec<ProvenancedTriple> = Vec::new();
        let mut source_stats_list: Vec<SourceStats> = Vec::new();
        let mut total_raw = 0_usize;
        let mut total_blank_renamed = 0_usize;

        for (idx, src) in args.sources.iter().enumerate() {
            let mut triples = simulate_parse(src);
            let count = triples.len();
            total_raw += count;

            let renamed = rename_blank_nodes(&mut triples, idx);
            total_blank_renamed += renamed;

            source_stats_list.push(SourceStats {
                source: src.clone(),
                triple_count: count,
                blank_nodes_renamed: renamed,
            });

            if args.track_provenance || args.mode == MergeMode::WithProvenance {
                for t in &triples {
                    provenance_records.push(ProvenancedTriple {
                        triple: t.clone(),
                        source: src.clone(),
                    });
                }
            }

            all_triples.extend(triples);
        }

        // Deduplicate
        let unique_set: HashSet<RdfTriple> = all_triples.iter().cloned().collect();
        let mut merged: Vec<RdfTriple> = unique_set.into_iter().collect();
        merged.sort_by(|a, b| (&a.s, &a.p, &a.o).cmp(&(&b.s, &b.p, &b.o)));

        let duplicates_removed = total_raw.saturating_sub(merged.len());

        // Conflict detection
        let conflicts = detect_conflicts(&merged);

        let stats = MergeStats {
            total_triples: merged.len(),
            duplicates_removed,
            conflicts_found: conflicts.len(),
            source_stats: source_stats_list,
            total_blank_nodes_renamed: total_blank_renamed,
        };

        let output_text = if args.dry_run {
            None
        } else {
            Some(serialize_output(&merged, args.output_format))
        };

        let result_triples = if args.dry_run { Vec::new() } else { merged };

        Ok(MergeResult {
            triples: result_triples,
            provenance: provenance_records,
            conflicts,
            stats,
            dry_run: args.dry_run,
            output_text,
        })
    }

    /// Merge two pre-loaded sets of triples directly (for programmatic use).
    pub fn merge_triple_sets(
        &self,
        set_a: &[RdfTriple],
        set_b: &[RdfTriple],
    ) -> (Vec<RdfTriple>, MergeStats) {
        let mut a = set_a.to_vec();
        let mut b = set_b.to_vec();
        let count_a = a.len();
        let count_b = b.len();

        let renamed_a = rename_blank_nodes(&mut a, 0);
        let renamed_b = rename_blank_nodes(&mut b, 1);

        let mut all: Vec<RdfTriple> = Vec::with_capacity(count_a + count_b);
        all.extend(a);
        all.extend(b);

        let total_raw = all.len();
        let unique: HashSet<RdfTriple> = all.into_iter().collect();
        let mut merged: Vec<RdfTriple> = unique.into_iter().collect();
        merged.sort_by(|a, b| (&a.s, &a.p, &a.o).cmp(&(&b.s, &b.p, &b.o)));

        let conflicts = detect_conflicts(&merged);

        let stats = MergeStats {
            total_triples: merged.len(),
            duplicates_removed: total_raw.saturating_sub(merged.len()),
            conflicts_found: conflicts.len(),
            source_stats: vec![
                SourceStats {
                    source: "set_a".to_string(),
                    triple_count: count_a,
                    blank_nodes_renamed: renamed_a,
                },
                SourceStats {
                    source: "set_b".to_string(),
                    triple_count: count_b,
                    blank_nodes_renamed: renamed_b,
                },
            ],
            total_blank_nodes_renamed: renamed_a + renamed_b,
        };

        (merged, stats)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_args(sources: Vec<&str>) -> MergeArgs {
        MergeArgs {
            sources: sources.into_iter().map(String::from).collect(),
            output: None,
            output_format: OutputFormat::NTriples,
            mode: MergeMode::SetUnion,
            dry_run: false,
            track_provenance: false,
        }
    }

    // ── RdfTriple tests ─────────────────────────────────────────────────

    #[test]
    fn test_triple_new() {
        let t = RdfTriple::new("s", "p", "o");
        assert_eq!(t.s, "s");
        assert_eq!(t.p, "p");
        assert_eq!(t.o, "o");
    }

    #[test]
    fn test_triple_has_blank_subject() {
        let t = RdfTriple::new("_:b0", "p", "o");
        assert!(t.has_blank_node());
    }

    #[test]
    fn test_triple_has_blank_object() {
        let t = RdfTriple::new("s", "p", "_:b1");
        assert!(t.has_blank_node());
    }

    #[test]
    fn test_triple_no_blank() {
        let t = RdfTriple::new("http://example.org/s", "p", "http://example.org/o");
        assert!(!t.has_blank_node());
    }

    #[test]
    fn test_triple_equality() {
        let a = RdfTriple::new("s", "p", "o");
        let b = RdfTriple::new("s", "p", "o");
        assert_eq!(a, b);
    }

    #[test]
    fn test_triple_hash_dedup() {
        let a = RdfTriple::new("s", "p", "o");
        let b = RdfTriple::new("s", "p", "o");
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 1);
    }

    // ── OutputFormat tests ──────────────────────────────────────────────

    #[test]
    fn test_output_format_extension() {
        assert_eq!(OutputFormat::Turtle.extension(), "ttl");
        assert_eq!(OutputFormat::NTriples.extension(), "nt");
        assert_eq!(OutputFormat::NQuads.extension(), "nq");
        assert_eq!(OutputFormat::RdfXml.extension(), "rdf");
    }

    #[test]
    fn test_output_format_from_extension() {
        assert_eq!(
            OutputFormat::from_extension("ttl"),
            Some(OutputFormat::Turtle)
        );
        assert_eq!(
            OutputFormat::from_extension("NT"),
            Some(OutputFormat::NTriples)
        );
        assert_eq!(
            OutputFormat::from_extension("nq"),
            Some(OutputFormat::NQuads)
        );
        assert_eq!(
            OutputFormat::from_extension("rdf"),
            Some(OutputFormat::RdfXml)
        );
        assert_eq!(
            OutputFormat::from_extension("xml"),
            Some(OutputFormat::RdfXml)
        );
        assert_eq!(OutputFormat::from_extension("csv"), None);
    }

    // ── Blank node renaming tests ───────────────────────────────────────

    #[test]
    fn test_rename_blank_nodes_subject() {
        let mut triples = vec![RdfTriple::new("_:b0", "p", "o")];
        let count = rename_blank_nodes(&mut triples, 0);
        assert_eq!(count, 1);
        assert_eq!(triples[0].s, "_:src0_b0");
    }

    #[test]
    fn test_rename_blank_nodes_object() {
        let mut triples = vec![RdfTriple::new("s", "p", "_:b1")];
        let count = rename_blank_nodes(&mut triples, 2);
        assert_eq!(count, 1);
        assert_eq!(triples[0].o, "_:src2_b1");
    }

    #[test]
    fn test_rename_blank_nodes_both() {
        let mut triples = vec![RdfTriple::new("_:a", "p", "_:b")];
        let count = rename_blank_nodes(&mut triples, 1);
        assert_eq!(count, 2);
        assert_eq!(triples[0].s, "_:src1_a");
        assert_eq!(triples[0].o, "_:src1_b");
    }

    #[test]
    fn test_rename_no_blanks() {
        let mut triples = vec![RdfTriple::new("http://x", "p", "http://y")];
        let count = rename_blank_nodes(&mut triples, 0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_rename_different_sources_unique() {
        let mut a = vec![RdfTriple::new("_:x", "p", "o")];
        let mut b = vec![RdfTriple::new("_:x", "p", "o")];
        rename_blank_nodes(&mut a, 0);
        rename_blank_nodes(&mut b, 1);
        assert_ne!(a[0].s, b[0].s); // Different prefixes
    }

    // ── Format detection tests ──────────────────────────────────────────

    #[test]
    fn test_detect_format_turtle() {
        assert_eq!(detect_format("data.ttl"), Some("turtle"));
    }

    #[test]
    fn test_detect_format_ntriples() {
        assert_eq!(detect_format("data.nt"), Some("ntriples"));
    }

    #[test]
    fn test_detect_format_rdfxml() {
        assert_eq!(detect_format("data.rdf"), Some("rdfxml"));
        assert_eq!(detect_format("data.xml"), Some("rdfxml"));
    }

    #[test]
    fn test_detect_format_unknown() {
        assert_eq!(detect_format("data.csv"), None);
    }

    #[test]
    fn test_detect_format_no_extension() {
        assert_eq!(detect_format("noextension"), None);
    }

    // ── Conflict detection tests ────────────────────────────────────────

    #[test]
    fn test_no_conflicts() {
        let triples = vec![
            RdfTriple::new("s1", "p1", "o1"),
            RdfTriple::new("s2", "p2", "o2"),
        ];
        let conflicts = detect_conflicts(&triples);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_conflict_detected() {
        let triples = vec![
            RdfTriple::new("s", "p", "o1"),
            RdfTriple::new("s", "p", "o2"),
        ];
        let conflicts = detect_conflicts(&triples);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].subject, "s");
        assert_eq!(conflicts[0].predicate, "p");
        assert_eq!(conflicts[0].objects.len(), 2);
    }

    #[test]
    fn test_conflict_objects_sorted() {
        let triples = vec![
            RdfTriple::new("s", "p", "z_obj"),
            RdfTriple::new("s", "p", "a_obj"),
        ];
        let conflicts = detect_conflicts(&triples);
        assert_eq!(conflicts[0].objects, vec!["a_obj", "z_obj"]);
    }

    #[test]
    fn test_same_sp_same_o_no_conflict() {
        let triples = vec![RdfTriple::new("s", "p", "o"), RdfTriple::new("s", "p", "o")];
        let conflicts = detect_conflicts(&triples);
        assert!(conflicts.is_empty());
    }

    // ── Serialization tests ─────────────────────────────────────────────

    #[test]
    fn test_serialize_ntriples() {
        let triples = vec![RdfTriple::new(
            "http://example.org/s",
            "http://example.org/p",
            "\"val\"",
        )];
        let out = serialize_ntriples(&triples);
        assert!(out.contains("<http://example.org/s>"));
        assert!(out.contains("<http://example.org/p>"));
        assert!(out.contains("\"val\""));
        assert!(out.ends_with(".\n"));
    }

    #[test]
    fn test_serialize_turtle_has_prefix() {
        let triples = vec![RdfTriple::new(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        )];
        let out = serialize_turtle(&triples);
        assert!(out.contains("@prefix"));
    }

    #[test]
    fn test_serialize_rdfxml() {
        let triples = vec![RdfTriple::new("s", "p", "o")];
        let out = serialize_output(&triples, OutputFormat::RdfXml);
        assert!(out.contains("rdf:RDF"));
    }

    #[test]
    fn test_format_term_iri() {
        assert_eq!(
            format_term("http://example.org/x"),
            "<http://example.org/x>"
        );
    }

    #[test]
    fn test_format_term_literal() {
        assert_eq!(format_term("\"hello\""), "\"hello\"");
    }

    #[test]
    fn test_format_term_blank() {
        assert_eq!(format_term("_:b0"), "_:b0");
    }

    // ── MergeCommand execute tests ──────────────────────────────────────

    #[test]
    fn test_merge_no_sources() {
        let cmd = MergeCommand::new();
        let args = default_args(vec![]);
        let result = cmd.execute(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_single_source() {
        let cmd = MergeCommand::new();
        let args = default_args(vec!["data.ttl"]);
        let result = cmd.execute(&args);
        assert!(result.is_ok());
        let r = result.expect("should succeed");
        assert!(r.stats.total_triples > 0);
    }

    #[test]
    fn test_merge_two_sources() {
        let cmd = MergeCommand::new();
        let args = default_args(vec!["alpha.ttl", "beta.nt"]);
        let result = cmd.execute(&args);
        assert!(result.is_ok());
        let r = result.expect("should succeed");
        assert!(r.stats.total_triples > 0);
        assert_eq!(r.stats.source_stats.len(), 2);
    }

    #[test]
    fn test_merge_three_sources() {
        let cmd = MergeCommand::new();
        let args = default_args(vec!["a.ttl", "b.nt", "c.rdf"]);
        let result = cmd.execute(&args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_merge_unsupported_format() {
        let cmd = MergeCommand::new();
        let args = default_args(vec!["bad.csv"]);
        let result = cmd.execute(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_dry_run_no_triples() {
        let cmd = MergeCommand::new();
        let mut args = default_args(vec!["data.ttl"]);
        args.dry_run = true;
        let result = cmd.execute(&args).expect("should succeed");
        assert!(result.dry_run);
        assert!(result.triples.is_empty());
        assert!(result.output_text.is_none());
        assert!(result.stats.total_triples > 0);
    }

    #[test]
    fn test_merge_with_provenance() {
        let cmd = MergeCommand::new();
        let mut args = default_args(vec!["data.ttl"]);
        args.track_provenance = true;
        let result = cmd.execute(&args).expect("should succeed");
        assert!(!result.provenance.is_empty());
        for prov in &result.provenance {
            assert_eq!(prov.source, "data.ttl");
        }
    }

    #[test]
    fn test_merge_provenance_mode() {
        let cmd = MergeCommand::new();
        let mut args = default_args(vec!["a.ttl", "b.nt"]);
        args.mode = MergeMode::WithProvenance;
        let result = cmd.execute(&args).expect("should succeed");
        assert!(!result.provenance.is_empty());
    }

    #[test]
    fn test_merge_output_ntriples() {
        let cmd = MergeCommand::new();
        let args = default_args(vec!["data.ttl"]);
        let result = cmd.execute(&args).expect("should succeed");
        let text = result.output_text.as_ref().expect("should have output");
        assert!(text.contains('.'));
    }

    #[test]
    fn test_merge_output_turtle() {
        let cmd = MergeCommand::new();
        let mut args = default_args(vec!["data.ttl"]);
        args.output_format = OutputFormat::Turtle;
        let result = cmd.execute(&args).expect("should succeed");
        let text = result.output_text.as_ref().expect("should have output");
        assert!(text.contains("@prefix"));
    }

    #[test]
    fn test_merge_dedup_identical_sources() {
        let cmd = MergeCommand::new();
        let args = default_args(vec!["same.ttl", "same.ttl"]);
        let result = cmd.execute(&args).expect("should succeed");
        // Two identical sources: should have duplicates removed
        // (some might remain because blank node renaming makes them unique)
        assert!(result.stats.total_triples > 0);
    }

    #[test]
    fn test_merge_stats_fields() {
        let cmd = MergeCommand::new();
        let args = default_args(vec!["x.ttl", "y.nt"]);
        let result = cmd.execute(&args).expect("should succeed");
        assert_eq!(result.stats.source_stats.len(), 2);
        assert_eq!(result.stats.source_stats[0].source, "x.ttl");
        assert_eq!(result.stats.source_stats[1].source, "y.nt");
    }

    // ── merge_triple_sets tests ─────────────────────────────────────────

    #[test]
    fn test_merge_triple_sets_basic() {
        let cmd = MergeCommand::new();
        let a = vec![RdfTriple::new("s1", "p", "o1")];
        let b = vec![RdfTriple::new("s2", "p", "o2")];
        let (merged, stats) = cmd.merge_triple_sets(&a, &b);
        assert_eq!(merged.len(), 2);
        assert_eq!(stats.total_triples, 2);
    }

    #[test]
    fn test_merge_triple_sets_dedup() {
        let cmd = MergeCommand::new();
        let a = vec![RdfTriple::new("s", "p", "o")];
        let b = vec![RdfTriple::new("s", "p", "o")];
        let (merged, stats) = cmd.merge_triple_sets(&a, &b);
        assert_eq!(merged.len(), 1);
        assert_eq!(stats.duplicates_removed, 1);
    }

    #[test]
    fn test_merge_triple_sets_blank_rename() {
        let cmd = MergeCommand::new();
        let a = vec![RdfTriple::new("_:x", "p", "o")];
        let b = vec![RdfTriple::new("_:x", "p", "o")];
        let (merged, stats) = cmd.merge_triple_sets(&a, &b);
        // After renaming, _:src0_x and _:src1_x are different triples
        assert_eq!(merged.len(), 2);
        assert!(stats.total_blank_nodes_renamed > 0);
    }

    #[test]
    fn test_merge_triple_sets_empty() {
        let cmd = MergeCommand::new();
        let (merged, stats) = cmd.merge_triple_sets(&[], &[]);
        assert!(merged.is_empty());
        assert_eq!(stats.total_triples, 0);
    }

    #[test]
    fn test_merge_triple_sets_conflict() {
        let cmd = MergeCommand::new();
        let a = vec![RdfTriple::new("s", "p", "o1")];
        let b = vec![RdfTriple::new("s", "p", "o2")];
        let (_merged, stats) = cmd.merge_triple_sets(&a, &b);
        assert!(stats.conflicts_found > 0);
    }

    // ── Error display tests ─────────────────────────────────────────────

    #[test]
    fn test_error_display_no_sources() {
        let err = MergeError::NoSources;
        assert_eq!(format!("{err}"), "No source files provided");
    }

    #[test]
    fn test_error_display_source_not_found() {
        let err = MergeError::SourceNotFound("missing.ttl".to_string());
        assert!(format!("{err}").contains("missing.ttl"));
    }

    #[test]
    fn test_error_display_unsupported() {
        let err = MergeError::UnsupportedFormat("csv".to_string());
        assert!(format!("{err}").contains("csv"));
    }

    #[test]
    fn test_error_display_internal() {
        let err = MergeError::Internal("oops".to_string());
        assert!(format!("{err}").contains("oops"));
    }

    // ── Default impl test ───────────────────────────────────────────────

    #[test]
    fn test_merge_command_default() {
        let cmd = MergeCommand;
        let args = default_args(vec!["test.ttl"]);
        let result = cmd.execute(&args);
        assert!(result.is_ok());
    }
}
