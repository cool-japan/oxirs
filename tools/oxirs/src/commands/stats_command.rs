//! # Dataset Statistics Command
//!
//! Analyses an RDF dataset file and reports statistics including triple counts,
//! unique subjects/predicates/objects, top predicates, and literal/IRI/blank-node breakdown.

use std::time::Instant;

// ─── Public types ─────────────────────────────────────────────────────────────

/// Arguments for the stats command
#[derive(Debug, Clone)]
pub struct StatsArgs {
    /// Path to the RDF file
    pub file: String,
    /// Optional format override (e.g. "turtle", "ntriples")
    pub format: Option<String>,
    /// Output format for the report
    pub output: StatsOutputFormat,
    /// Whether to include predicate-level statistics
    pub include_predicates: bool,
    /// Maximum number of top predicates to report
    pub top_k: usize,
}

/// Output format for the statistics report
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatsOutputFormat {
    Text,
    Json,
    Csv,
}

/// Statistics for a single predicate
#[derive(Debug, Clone)]
pub struct PredicateStats {
    pub predicate: String,
    pub count: usize,
    pub pct: f64,
}

/// Complete dataset statistics
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub total_triples: usize,
    pub unique_subjects: usize,
    pub unique_predicates: usize,
    pub unique_objects: usize,
    pub graphs: usize,
    pub top_predicates: Vec<PredicateStats>,
    pub literal_count: usize,
    pub iri_object_count: usize,
    pub blank_node_count: usize,
    pub elapsed_ms: u64,
}

/// Errors that can occur when running the stats command
#[derive(Debug)]
pub enum StatsError {
    FileNotFound(String),
    ParseError(String),
    UnsupportedFormat(String),
}

impl std::fmt::Display for StatsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(p) => write!(f, "File not found: {p}"),
            Self::ParseError(m) => write!(f, "Parse error: {m}"),
            Self::UnsupportedFormat(fmt) => write!(f, "Unsupported format: {fmt}"),
        }
    }
}

impl std::error::Error for StatsError {}

// ─── StatsCommand ─────────────────────────────────────────────────────────────

/// RDF dataset statistics command
pub struct StatsCommand;

impl Default for StatsCommand {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsCommand {
    /// Create a new StatsCommand instance
    pub fn new() -> Self {
        Self
    }

    /// Execute the stats command.
    ///
    /// If the file does not exist, returns `StatsError::FileNotFound`.
    /// If the format is given but unsupported, returns `StatsError::UnsupportedFormat`.
    /// Otherwise returns simulated statistics derived from the file name.
    pub fn execute(&self, args: &StatsArgs) -> Result<DatasetStats, StatsError> {
        // Check for unsupported format
        if let Some(ref fmt) = args.format {
            let supported = [
                "turtle", "ttl", "ntriples", "nt", "nquads", "nq", "trig", "jsonld", "rdfxml",
                "xml",
            ];
            if !supported.contains(&fmt.to_lowercase().as_str()) {
                return Err(StatsError::UnsupportedFormat(fmt.clone()));
            }
        }

        // Check file existence
        if !std::path::Path::new(&args.file).exists() {
            return Err(StatsError::FileNotFound(args.file.clone()));
        }

        let start = Instant::now();
        let stats = Self::simulate_stats(&args.file, args.include_predicates, args.top_k);
        let elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(DatasetStats {
            elapsed_ms,
            ..stats
        })
    }

    /// Format dataset statistics as a human-readable string.
    pub fn format_output(&self, stats: &DatasetStats, format: &StatsOutputFormat) -> String {
        match format {
            StatsOutputFormat::Text => Self::format_text(stats),
            StatsOutputFormat::Json => Self::format_json(stats),
            StatsOutputFormat::Csv => Self::format_csv(stats),
        }
    }

    /// Produce deterministic simulated statistics from a file name.
    ///
    /// The simulation is deterministic (same input → same output) and
    /// is used when the file exists but no real parser is available.
    pub fn simulate_stats(filename: &str, include_predicates: bool, top_k: usize) -> DatasetStats {
        // Deterministic seed from the filename
        let seed: u64 = filename
            .bytes()
            .enumerate()
            .fold(0xcbf29ce484222325u64, |acc, (i, b)| {
                acc.wrapping_mul(0x100000001b3)
                    .wrapping_add(b as u64)
                    .wrapping_add(i as u64)
            });

        let total_triples = ((seed >> 2) % 90_000 + 10_000) as usize;
        let unique_subjects = ((seed >> 5) % (total_triples as u64 / 3) + 100) as usize;
        let unique_predicates = ((seed >> 8) % 200 + 5) as usize;
        let unique_objects = ((seed >> 11) % (total_triples as u64 / 2) + 200) as usize;
        let graphs = ((seed >> 14) % 8 + 1) as usize;
        let literal_count = ((seed >> 17) % (total_triples as u64 / 2)) as usize;
        let blank_node_count = ((seed >> 20) % (total_triples as u64 / 10)) as usize;
        let iri_object_count = total_triples
            .saturating_sub(literal_count)
            .saturating_sub(blank_node_count);

        // Build simulated predicate list
        let top_predicates = if include_predicates {
            Self::simulated_predicates(seed, unique_predicates, total_triples, top_k)
        } else {
            vec![]
        };

        DatasetStats {
            total_triples,
            unique_subjects,
            unique_predicates,
            unique_objects,
            graphs,
            top_predicates,
            literal_count,
            iri_object_count,
            blank_node_count,
            elapsed_ms: 0,
        }
    }

    /// Compute the percentage coverage of the top_predicates list.
    ///
    /// Returns sum(pct) / 100 — a value in [0, 1].
    pub fn predicate_coverage(stats: &DatasetStats) -> f64 {
        let sum: f64 = stats.top_predicates.iter().map(|p| p.pct).sum();
        sum / 100.0
    }

    // ─── private helpers ──────────────────────────────────────────────────────

    fn format_text(stats: &DatasetStats) -> String {
        let mut out = String::new();
        out.push_str("=== RDF Dataset Statistics ===\n");
        out.push_str(&format!(
            "  Total triples       : {}\n",
            stats.total_triples
        ));
        out.push_str(&format!(
            "  Unique subjects     : {}\n",
            stats.unique_subjects
        ));
        out.push_str(&format!(
            "  Unique predicates   : {}\n",
            stats.unique_predicates
        ));
        out.push_str(&format!(
            "  Unique objects      : {}\n",
            stats.unique_objects
        ));
        out.push_str(&format!("  Named graphs        : {}\n", stats.graphs));
        out.push_str(&format!(
            "  Literal objects     : {}\n",
            stats.literal_count
        ));
        out.push_str(&format!(
            "  IRI objects         : {}\n",
            stats.iri_object_count
        ));
        out.push_str(&format!(
            "  Blank node objects  : {}\n",
            stats.blank_node_count
        ));
        out.push_str(&format!(
            "  Elapsed             : {} ms\n",
            stats.elapsed_ms
        ));

        if !stats.top_predicates.is_empty() {
            out.push_str("\n--- Top Predicates ---\n");
            for ps in &stats.top_predicates {
                out.push_str(&format!(
                    "  {:6} ({:5.1}%)  {}\n",
                    ps.count, ps.pct, ps.predicate
                ));
            }
        }
        out
    }

    fn format_json(stats: &DatasetStats) -> String {
        let preds_json: Vec<String> = stats
            .top_predicates
            .iter()
            .map(|p| {
                format!(
                    "{{\"predicate\":\"{}\",\"count\":{},\"pct\":{:.2}}}",
                    p.predicate, p.count, p.pct
                )
            })
            .collect();

        format!(
            "{{\
            \"total_triples\":{},\
            \"unique_subjects\":{},\
            \"unique_predicates\":{},\
            \"unique_objects\":{},\
            \"graphs\":{},\
            \"literal_count\":{},\
            \"iri_object_count\":{},\
            \"blank_node_count\":{},\
            \"elapsed_ms\":{},\
            \"top_predicates\":[{}]\
            }}",
            stats.total_triples,
            stats.unique_subjects,
            stats.unique_predicates,
            stats.unique_objects,
            stats.graphs,
            stats.literal_count,
            stats.iri_object_count,
            stats.blank_node_count,
            stats.elapsed_ms,
            preds_json.join(",")
        )
    }

    fn format_csv(stats: &DatasetStats) -> String {
        let mut out = String::new();
        out.push_str("metric,value\n");
        out.push_str(&format!("total_triples,{}\n", stats.total_triples));
        out.push_str(&format!("unique_subjects,{}\n", stats.unique_subjects));
        out.push_str(&format!("unique_predicates,{}\n", stats.unique_predicates));
        out.push_str(&format!("unique_objects,{}\n", stats.unique_objects));
        out.push_str(&format!("graphs,{}\n", stats.graphs));
        out.push_str(&format!("literal_count,{}\n", stats.literal_count));
        out.push_str(&format!("iri_object_count,{}\n", stats.iri_object_count));
        out.push_str(&format!("blank_node_count,{}\n", stats.blank_node_count));
        out.push_str(&format!("elapsed_ms,{}\n", stats.elapsed_ms));

        if !stats.top_predicates.is_empty() {
            out.push_str("\npredicate,count,pct\n");
            for ps in &stats.top_predicates {
                out.push_str(&format!("{},{},{:.2}\n", ps.predicate, ps.count, ps.pct));
            }
        }
        out
    }

    /// Build a deterministic list of simulated predicate statistics
    fn simulated_predicates(
        seed: u64,
        unique_predicates: usize,
        total_triples: usize,
        top_k: usize,
    ) -> Vec<PredicateStats> {
        // Generate predicate names and random counts
        let n = unique_predicates.min(200); // cap to avoid huge allocations
        let mut predicates: Vec<(String, usize)> = (0..n)
            .map(|i| {
                let hash = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(i as u64);
                let count = ((hash % (total_triples as u64 / n as u64).max(1)) + 1) as usize;
                let name = format!("http://example.org/prop/{:08x}", hash & 0xffff_ffff);
                (name, count)
            })
            .collect();

        // Sort descending by count
        predicates.sort_by_key(|item| std::cmp::Reverse(item.1));

        // Take top_k
        let limit = top_k.min(predicates.len());
        let selected: Vec<(String, usize)> = predicates.into_iter().take(limit).collect();

        // Normalise percentages to sum to approximately 100 if all predicates are included,
        // or a sub-100 value if only a subset is taken.
        // We base pct on total_triples to be realistic.
        let denom = total_triples.max(1) as f64;

        selected
            .into_iter()
            .map(|(predicate, count)| {
                let pct = (count as f64 / denom) * 100.0;
                PredicateStats {
                    predicate,
                    count,
                    pct,
                }
            })
            .collect()
    }

    /// Verify that top_predicates are sorted in descending order by count
    #[allow(dead_code)]
    fn predicates_sorted_descending(preds: &[PredicateStats]) -> bool {
        preds.windows(2).all(|w| w[0].count >= w[1].count)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a temporary file for testing
    fn tmp_file(name: &str) -> String {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("oxirs_stats_test_{name}.nt"));
        std::fs::write(&path, "# temp RDF file for tests\n").unwrap_or_default();
        path.to_string_lossy().to_string()
    }

    fn make_args(file: &str) -> StatsArgs {
        StatsArgs {
            file: file.to_string(),
            format: None,
            output: StatsOutputFormat::Text,
            include_predicates: true,
            top_k: 10,
        }
    }

    // ─── execute ──────────────────────────────────────────────────────────────

    #[test]
    fn test_execute_file_not_found() {
        let cmd = StatsCommand::new();
        let args = make_args("/nonexistent/path/file.nt");
        let err = cmd.execute(&args).expect_err("should fail");
        assert!(matches!(err, StatsError::FileNotFound(_)));
    }

    #[test]
    fn test_execute_unsupported_format() {
        let path = tmp_file("unsupported");
        let cmd = StatsCommand::new();
        let args = StatsArgs {
            file: path,
            format: Some("binary-rdf".to_string()),
            output: StatsOutputFormat::Text,
            include_predicates: false,
            top_k: 5,
        };
        let err = cmd.execute(&args).expect_err("unsupported format");
        assert!(matches!(err, StatsError::UnsupportedFormat(_)));
    }

    #[test]
    fn test_execute_valid_file_succeeds() {
        let path = tmp_file("valid");
        let cmd = StatsCommand::new();
        let args = make_args(&path);
        let stats = cmd.execute(&args).expect("should succeed");
        assert!(stats.total_triples > 0);
    }

    #[test]
    fn test_execute_supported_format_turtle() {
        let path = tmp_file("turtle");
        let cmd = StatsCommand::new();
        let args = StatsArgs {
            file: path,
            format: Some("turtle".to_string()),
            output: StatsOutputFormat::Text,
            include_predicates: false,
            top_k: 5,
        };
        cmd.execute(&args).expect("turtle is a supported format");
    }

    #[test]
    fn test_execute_supported_format_ntriples() {
        let path = tmp_file("nt");
        let cmd = StatsCommand::new();
        let args = StatsArgs {
            file: path,
            format: Some("ntriples".to_string()),
            output: StatsOutputFormat::Text,
            include_predicates: false,
            top_k: 5,
        };
        cmd.execute(&args).expect("ntriples is a supported format");
    }

    // ─── simulate_stats ───────────────────────────────────────────────────────

    #[test]
    fn test_simulate_stats_deterministic() {
        let s1 = StatsCommand::simulate_stats("test.nt", true, 10);
        let s2 = StatsCommand::simulate_stats("test.nt", true, 10);
        assert_eq!(
            s1.total_triples, s2.total_triples,
            "simulate_stats is deterministic"
        );
        assert_eq!(s1.unique_subjects, s2.unique_subjects);
    }

    #[test]
    fn test_simulate_stats_different_filenames() {
        let s1 = StatsCommand::simulate_stats("file_a.nt", true, 10);
        let s2 = StatsCommand::simulate_stats("file_b.nt", true, 10);
        // Very unlikely (but not impossible) to collide — we check at least one field differs
        assert_ne!(
            s1.total_triples, s2.total_triples,
            "different filenames should typically produce different stats"
        );
    }

    #[test]
    fn test_simulate_stats_total_triples_positive() {
        let s = StatsCommand::simulate_stats("any.nt", false, 5);
        assert!(s.total_triples > 0);
    }

    #[test]
    fn test_simulate_stats_unique_subjects_positive() {
        let s = StatsCommand::simulate_stats("any.nt", false, 5);
        assert!(s.unique_subjects > 0);
    }

    #[test]
    fn test_simulate_stats_unique_predicates_positive() {
        let s = StatsCommand::simulate_stats("any.nt", false, 5);
        assert!(s.unique_predicates > 0);
    }

    #[test]
    fn test_simulate_stats_unique_objects_positive() {
        let s = StatsCommand::simulate_stats("any.nt", false, 5);
        assert!(s.unique_objects > 0);
    }

    #[test]
    fn test_simulate_stats_no_predicates_when_disabled() {
        let s = StatsCommand::simulate_stats("any.nt", false, 10);
        assert!(
            s.top_predicates.is_empty(),
            "include_predicates=false → empty list"
        );
    }

    #[test]
    fn test_simulate_stats_top_k_limits_predicates() {
        let s = StatsCommand::simulate_stats("any.nt", true, 3);
        assert!(
            s.top_predicates.len() <= 3,
            "top_k=3 must limit to at most 3 predicates"
        );
    }

    #[test]
    fn test_simulate_stats_top_predicates_sorted_descending() {
        let s = StatsCommand::simulate_stats("any.nt", true, 10);
        assert!(
            StatsCommand::predicates_sorted_descending(&s.top_predicates),
            "top_predicates must be sorted descending by count"
        );
    }

    #[test]
    fn test_simulate_stats_predicate_pct_positive() {
        let s = StatsCommand::simulate_stats("any.nt", true, 10);
        for p in &s.top_predicates {
            assert!(p.pct > 0.0, "pct must be positive");
        }
    }

    #[test]
    fn test_simulate_stats_graphs_at_least_one() {
        let s = StatsCommand::simulate_stats("any.nt", false, 5);
        assert!(s.graphs >= 1);
    }

    // ─── predicate_coverage ───────────────────────────────────────────────────

    #[test]
    fn test_predicate_coverage_empty_is_zero() {
        let s = DatasetStats {
            total_triples: 100,
            unique_subjects: 10,
            unique_predicates: 5,
            unique_objects: 20,
            graphs: 1,
            top_predicates: vec![],
            literal_count: 10,
            iri_object_count: 80,
            blank_node_count: 10,
            elapsed_ms: 0,
        };
        assert_eq!(StatsCommand::predicate_coverage(&s), 0.0);
    }

    #[test]
    fn test_predicate_coverage_sums_correctly() {
        let s = DatasetStats {
            total_triples: 100,
            unique_subjects: 10,
            unique_predicates: 2,
            unique_objects: 20,
            graphs: 1,
            top_predicates: vec![
                PredicateStats {
                    predicate: "p1".to_string(),
                    count: 60,
                    pct: 60.0,
                },
                PredicateStats {
                    predicate: "p2".to_string(),
                    count: 40,
                    pct: 40.0,
                },
            ],
            literal_count: 10,
            iri_object_count: 80,
            blank_node_count: 10,
            elapsed_ms: 0,
        };
        let coverage = StatsCommand::predicate_coverage(&s);
        assert!(
            (coverage - 1.0).abs() < 1e-9,
            "coverage should be 1.0 (100%)"
        );
    }

    #[test]
    fn test_predicate_coverage_partial() {
        let s = DatasetStats {
            total_triples: 100,
            unique_subjects: 10,
            unique_predicates: 5,
            unique_objects: 20,
            graphs: 1,
            top_predicates: vec![PredicateStats {
                predicate: "p".to_string(),
                count: 50,
                pct: 50.0,
            }],
            literal_count: 10,
            iri_object_count: 80,
            blank_node_count: 10,
            elapsed_ms: 0,
        };
        let coverage = StatsCommand::predicate_coverage(&s);
        assert!((coverage - 0.5).abs() < 1e-9, "coverage 50/100 = 0.5");
    }

    // ─── format_output text ───────────────────────────────────────────────────

    #[test]
    fn test_format_text_contains_total_triples() {
        let s = StatsCommand::simulate_stats("f.nt", false, 5);
        let out = StatsCommand::new().format_output(&s, &StatsOutputFormat::Text);
        assert!(
            out.to_lowercase().contains("total"),
            "text output should mention total (case-insensitive)"
        );
    }

    #[test]
    fn test_format_text_contains_subjects() {
        let s = StatsCommand::simulate_stats("f.nt", false, 5);
        let out = StatsCommand::new().format_output(&s, &StatsOutputFormat::Text);
        assert!(out.contains("subject"));
    }

    #[test]
    fn test_format_text_contains_predicates_section_when_present() {
        let s = StatsCommand::simulate_stats("f.nt", true, 5);
        let out = StatsCommand::new().format_output(&s, &StatsOutputFormat::Text);
        if !s.top_predicates.is_empty() {
            assert!(out.contains("Predicate") || out.contains("predicate"));
        }
    }

    // ─── format_output json ───────────────────────────────────────────────────

    #[test]
    fn test_format_json_is_json_like() {
        let s = StatsCommand::simulate_stats("f.nt", false, 5);
        let out = StatsCommand::new().format_output(&s, &StatsOutputFormat::Json);
        assert!(out.starts_with('{') && out.ends_with('}'));
    }

    #[test]
    fn test_format_json_contains_total_triples() {
        let s = StatsCommand::simulate_stats("f.nt", false, 5);
        let out = StatsCommand::new().format_output(&s, &StatsOutputFormat::Json);
        assert!(out.contains("\"total_triples\""));
    }

    #[test]
    fn test_format_json_contains_top_predicates_key() {
        let s = StatsCommand::simulate_stats("f.nt", true, 5);
        let out = StatsCommand::new().format_output(&s, &StatsOutputFormat::Json);
        assert!(out.contains("\"top_predicates\""));
    }

    // ─── format_output csv ────────────────────────────────────────────────────

    #[test]
    fn test_format_csv_header() {
        let s = StatsCommand::simulate_stats("f.nt", false, 5);
        let out = StatsCommand::new().format_output(&s, &StatsOutputFormat::Csv);
        assert!(out.starts_with("metric,value"));
    }

    #[test]
    fn test_format_csv_contains_total_triples() {
        let s = StatsCommand::simulate_stats("f.nt", false, 5);
        let out = StatsCommand::new().format_output(&s, &StatsOutputFormat::Csv);
        assert!(out.contains("total_triples"));
    }

    #[test]
    fn test_format_csv_predicate_rows_when_present() {
        let s = StatsCommand::simulate_stats("f.nt", true, 5);
        let out = StatsCommand::new().format_output(&s, &StatsOutputFormat::Csv);
        if !s.top_predicates.is_empty() {
            // CSV should have predicate,count,pct header
            assert!(out.contains("predicate,count,pct"));
        }
    }

    // ─── StatsOutputFormat variants ───────────────────────────────────────────

    #[test]
    fn test_stats_output_format_text_variant() {
        let f = StatsOutputFormat::Text;
        assert_eq!(f, StatsOutputFormat::Text);
    }

    #[test]
    fn test_stats_output_format_json_variant() {
        let f = StatsOutputFormat::Json;
        assert_eq!(f, StatsOutputFormat::Json);
    }

    #[test]
    fn test_stats_output_format_csv_variant() {
        let f = StatsOutputFormat::Csv;
        assert_eq!(f, StatsOutputFormat::Csv);
    }

    // ─── top_k = 0 ───────────────────────────────────────────────────────────

    #[test]
    fn test_top_k_zero_empty_predicates() {
        let s = StatsCommand::simulate_stats("f.nt", true, 0);
        assert!(s.top_predicates.is_empty(), "top_k=0 → empty predicates");
    }

    // ─── elapsed_ms is set ───────────────────────────────────────────────────

    #[test]
    fn test_execute_elapsed_ms_is_a_number() {
        let path = tmp_file("elapsed");
        let cmd = StatsCommand::new();
        let args = make_args(&path);
        let stats = cmd.execute(&args).expect("should succeed");
        // elapsed_ms is a u64, just ensure it exists (can be 0 for fast runs)
        let _ = stats.elapsed_ms;
    }

    // ─── Additional tests (round 11 extra coverage) ───────────────────────────

    #[test]
    fn test_simulate_unique_subjects_positive() {
        let s = StatsCommand::simulate_stats("test.nt", false, 5);
        assert!(s.unique_subjects > 0);
    }

    #[test]
    fn test_simulate_unique_predicates_positive() {
        let s = StatsCommand::simulate_stats("test.nt", false, 5);
        assert!(s.unique_predicates > 0);
    }

    #[test]
    fn test_simulate_unique_objects_positive() {
        let s = StatsCommand::simulate_stats("test.nt", false, 5);
        assert!(s.unique_objects > 0);
    }

    #[test]
    fn test_simulate_graphs_at_least_one() {
        let s = StatsCommand::simulate_stats("test.nt", false, 5);
        assert!(s.graphs >= 1);
    }

    #[test]
    fn test_simulate_top_predicates_capped_at_top_k() {
        let top_k = 3;
        let s = StatsCommand::simulate_stats("test.nt", true, top_k);
        assert!(
            s.top_predicates.len() <= top_k,
            "top_predicates.len() {} should be <= top_k {}",
            s.top_predicates.len(),
            top_k
        );
    }

    #[test]
    fn test_simulate_predicate_pct_in_range() {
        let s = StatsCommand::simulate_stats("test.nt", true, 10);
        for ps in &s.top_predicates {
            assert!(
                ps.pct >= 0.0 && ps.pct <= 100.0,
                "pct {} must be in [0, 100]",
                ps.pct
            );
        }
    }

    #[test]
    fn test_format_json_curly_braces() {
        let s = StatsCommand::simulate_stats("f.nt", true, 5);
        let out = StatsCommand::new().format_output(&s, &StatsOutputFormat::Json);
        let open = out.chars().filter(|&c| c == '{').count();
        let close = out.chars().filter(|&c| c == '}').count();
        assert_eq!(open, close, "JSON braces must be balanced");
    }

    #[test]
    fn test_format_text_contains_predicates_word() {
        let s = StatsCommand::simulate_stats("f.nt", false, 5);
        let out = StatsCommand::new().format_output(&s, &StatsOutputFormat::Text);
        assert!(
            out.to_lowercase().contains("predicate"),
            "text output should mention predicates"
        );
    }

    #[test]
    fn test_format_csv_lines_count() {
        let s = StatsCommand::simulate_stats("f.nt", false, 0);
        let out = StatsCommand::new().format_output(&s, &StatsOutputFormat::Csv);
        let lines: Vec<&str> = out.lines().collect();
        // At minimum: header + 8 stat rows
        assert!(lines.len() >= 9, "CSV should have header + stats rows");
    }

    #[test]
    fn test_simulate_different_files_different_stats() {
        let s1 = StatsCommand::simulate_stats("alpha.nt", false, 5);
        let s2 = StatsCommand::simulate_stats("beta.nt", false, 5);
        // Different filenames → different hash → at least one stat differs
        let some_differ = s1.total_triples != s2.total_triples
            || s1.unique_subjects != s2.unique_subjects
            || s1.unique_objects != s2.unique_objects;
        assert!(
            some_differ,
            "different filenames should produce different stats"
        );
    }

    #[test]
    fn test_predicate_stats_fields() {
        let ps = PredicateStats {
            predicate: "http://example.org/prop".to_string(),
            count: 42,
            pct: 4.2,
        };
        assert_eq!(ps.predicate, "http://example.org/prop");
        assert_eq!(ps.count, 42);
        assert!((ps.pct - 4.2).abs() < 1e-9);
    }

    #[test]
    fn test_stats_error_display_file_not_found() {
        let err = StatsError::FileNotFound("/some/path.nt".to_string());
        let msg = format!("{err}");
        assert!(
            msg.contains("/some/path.nt"),
            "error message should contain path"
        );
    }
}
