//! RDF data profiling for shape inference.
//!
//! Analyses RDF triples to compute property frequencies, value type
//! distributions, cardinality statistics, pattern detection, outlier
//! identification, datatype and language-tag distributions, profile reports,
//! and incremental profiling.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Value classification
// ---------------------------------------------------------------------------

/// Classification of an RDF term value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueType {
    /// A full IRI (starts with `http://` or `https://`).
    Iri,
    /// A blank node (starts with `_:`).
    BlankNode,
    /// A plain or typed literal.
    Literal,
}

impl ValueType {
    /// Classify a string-serialised RDF term.
    pub fn classify(value: &str) -> Self {
        if value.starts_with("http://") || value.starts_with("https://") {
            ValueType::Iri
        } else if value.starts_with("_:") {
            ValueType::BlankNode
        } else {
            ValueType::Literal
        }
    }
}

// ---------------------------------------------------------------------------
// Triple observation
// ---------------------------------------------------------------------------

/// A single RDF triple observation fed into the profiler.
#[derive(Debug, Clone)]
pub struct TripleObservation {
    /// Subject IRI or blank-node label.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object value (IRI, blank-node label, or serialised literal).
    pub object: String,
    /// Optional XSD datatype IRI for literal objects.
    pub datatype: Option<String>,
    /// Optional language tag for string literals.
    pub language_tag: Option<String>,
}

impl TripleObservation {
    /// Create a minimal triple observation without datatype or language tag.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            datatype: None,
            language_tag: None,
        }
    }

    /// Set the XSD datatype.
    pub fn with_datatype(mut self, dt: impl Into<String>) -> Self {
        self.datatype = Some(dt.into());
        self
    }

    /// Set the language tag.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language_tag = Some(lang.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Per-property statistics
// ---------------------------------------------------------------------------

/// Statistics for a single predicate across all observations.
#[derive(Debug, Clone)]
pub struct PropertyStats {
    /// Predicate IRI.
    pub predicate: String,
    /// Total number of triples with this predicate.
    pub triple_count: usize,
    /// Number of distinct subjects that use this predicate.
    pub subject_count: usize,
    /// Minimum number of values per subject.
    pub min_cardinality: usize,
    /// Maximum number of values per subject.
    pub max_cardinality: usize,
    /// Average number of values per subject.
    pub avg_cardinality: f64,
    /// Distribution of value types (IRI / BlankNode / Literal).
    pub value_type_distribution: HashMap<ValueType, usize>,
    /// Distribution of XSD datatypes (if present).
    pub datatype_distribution: HashMap<String, usize>,
    /// Distribution of language tags (if present).
    pub language_distribution: HashMap<String, usize>,
    /// Number of distinct object values.
    pub distinct_values: usize,
}

// ---------------------------------------------------------------------------
// Outlier
// ---------------------------------------------------------------------------

/// An identified outlier observation.
#[derive(Debug, Clone)]
pub struct Outlier {
    /// Subject of the outlier triple.
    pub subject: String,
    /// Predicate of the outlier triple.
    pub predicate: String,
    /// Object value of the outlier triple.
    pub object: String,
    /// Human-readable reason this is an outlier.
    pub reason: String,
}

// ---------------------------------------------------------------------------
// Profile report
// ---------------------------------------------------------------------------

/// Summary profile report across all properties.
#[derive(Debug, Clone)]
pub struct ProfileReport {
    /// Total number of triples profiled.
    pub total_triples: usize,
    /// Number of distinct subjects.
    pub distinct_subjects: usize,
    /// Number of distinct predicates.
    pub distinct_predicates: usize,
    /// Per-property statistics.
    pub property_stats: Vec<PropertyStats>,
    /// Detected outliers.
    pub outliers: Vec<Outlier>,
    /// Detected common subject-property patterns.
    pub patterns: Vec<SubjectPropertyPattern>,
}

/// A commonly observed subject-property pattern.
#[derive(Debug, Clone)]
pub struct SubjectPropertyPattern {
    /// Set of predicates that co-occur for many subjects.
    pub predicates: Vec<String>,
    /// Number of subjects exhibiting this pattern.
    pub subject_count: usize,
    /// Fraction of all subjects exhibiting this pattern.
    pub coverage: f64,
}

// ---------------------------------------------------------------------------
// DataProfiler
// ---------------------------------------------------------------------------

/// RDF data profiler for analysing property usage and value distributions.
pub struct DataProfiler {
    /// Accumulated per-predicate data: predicate -> subject -> [objects].
    property_data: HashMap<String, HashMap<String, Vec<String>>>,
    /// Accumulated datatype observations: predicate -> datatype -> count.
    datatype_data: HashMap<String, HashMap<String, usize>>,
    /// Accumulated language observations: predicate -> lang -> count.
    language_data: HashMap<String, HashMap<String, usize>>,
    /// All subjects seen.
    subjects: HashSet<String>,
    /// Total triples ingested.
    total_triples: usize,
    /// IQR multiplier for numeric outlier detection.
    outlier_iqr_multiplier: f64,
}

impl DataProfiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self {
            property_data: HashMap::new(),
            datatype_data: HashMap::new(),
            language_data: HashMap::new(),
            subjects: HashSet::new(),
            total_triples: 0,
            outlier_iqr_multiplier: 1.5,
        }
    }

    /// Set the IQR multiplier used for numeric outlier detection (default 1.5).
    pub fn with_outlier_iqr(mut self, multiplier: f64) -> Self {
        self.outlier_iqr_multiplier = multiplier;
        self
    }

    // ── Ingestion ────────────────────────────────────────────────────────

    /// Ingest a single triple observation (incremental profiling).
    pub fn add(&mut self, obs: &TripleObservation) {
        self.subjects.insert(obs.subject.clone());
        self.total_triples += 1;

        self.property_data
            .entry(obs.predicate.clone())
            .or_default()
            .entry(obs.subject.clone())
            .or_default()
            .push(obs.object.clone());

        if let Some(ref dt) = obs.datatype {
            *self
                .datatype_data
                .entry(obs.predicate.clone())
                .or_default()
                .entry(dt.clone())
                .or_insert(0) += 1;
        }

        if let Some(ref lang) = obs.language_tag {
            *self
                .language_data
                .entry(obs.predicate.clone())
                .or_default()
                .entry(lang.clone())
                .or_insert(0) += 1;
        }
    }

    /// Ingest a batch of triple observations.
    pub fn add_batch(&mut self, observations: &[TripleObservation]) {
        for obs in observations {
            self.add(obs);
        }
    }

    // ── Analysis ─────────────────────────────────────────────────────────

    /// Compute per-property statistics from all ingested data.
    pub fn property_stats(&self) -> Vec<PropertyStats> {
        let mut stats = Vec::new();

        for (predicate, subject_map) in &self.property_data {
            let triple_count: usize = subject_map.values().map(|v| v.len()).sum();
            let subject_count = subject_map.len();

            let cardinalities: Vec<usize> = subject_map.values().map(|v| v.len()).collect();
            let min_card = cardinalities.iter().copied().min().unwrap_or(0);
            let max_card = cardinalities.iter().copied().max().unwrap_or(0);
            let avg_card = if cardinalities.is_empty() {
                0.0
            } else {
                cardinalities.iter().sum::<usize>() as f64 / cardinalities.len() as f64
            };

            // Value type distribution
            let mut vt_dist: HashMap<ValueType, usize> = HashMap::new();
            let mut distinct_vals: HashSet<&str> = HashSet::new();
            for vals in subject_map.values() {
                for v in vals {
                    *vt_dist.entry(ValueType::classify(v)).or_insert(0) += 1;
                    distinct_vals.insert(v.as_str());
                }
            }

            let dt_dist = self
                .datatype_data
                .get(predicate)
                .cloned()
                .unwrap_or_default();
            let lang_dist = self
                .language_data
                .get(predicate)
                .cloned()
                .unwrap_or_default();

            stats.push(PropertyStats {
                predicate: predicate.clone(),
                triple_count,
                subject_count,
                min_cardinality: min_card,
                max_cardinality: max_card,
                avg_cardinality: avg_card,
                value_type_distribution: vt_dist,
                datatype_distribution: dt_dist,
                language_distribution: lang_dist,
                distinct_values: distinct_vals.len(),
            });
        }

        // Sort by triple count descending for determinism.
        stats.sort_by(|a, b| b.triple_count.cmp(&a.triple_count));
        stats
    }

    /// Detect numeric outliers using the IQR method.
    ///
    /// Only considers objects that parse as `f64`.
    pub fn detect_outliers(&self) -> Vec<Outlier> {
        let mut outliers = Vec::new();

        for (predicate, subject_map) in &self.property_data {
            // Collect numeric values with provenance.
            let mut numeric_vals: Vec<(f64, String, String)> = Vec::new();
            for (subject, vals) in subject_map {
                for v in vals {
                    if let Ok(n) = v.parse::<f64>() {
                        numeric_vals.push((n, subject.clone(), v.clone()));
                    }
                }
            }

            if numeric_vals.len() < 4 {
                continue; // not enough data for IQR
            }

            let mut sorted: Vec<f64> = numeric_vals.iter().map(|(n, _, _)| *n).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let q1 = percentile(&sorted, 25.0);
            let q3 = percentile(&sorted, 75.0);
            let iqr = q3 - q1;
            let lower = q1 - self.outlier_iqr_multiplier * iqr;
            let upper = q3 + self.outlier_iqr_multiplier * iqr;

            for (n, subject, object) in &numeric_vals {
                if *n < lower || *n > upper {
                    outliers.push(Outlier {
                        subject: subject.clone(),
                        predicate: predicate.clone(),
                        object: object.clone(),
                        reason: format!("value {n} outside IQR range [{lower:.2}, {upper:.2}]"),
                    });
                }
            }
        }

        outliers
    }

    /// Detect common subject-property co-occurrence patterns.
    ///
    /// Returns patterns where a set of predicates co-occur for a significant
    /// fraction of subjects.
    pub fn detect_patterns(&self, min_coverage: f64) -> Vec<SubjectPropertyPattern> {
        if self.subjects.is_empty() {
            return Vec::new();
        }

        // For each subject, collect the set of predicates it uses.
        let mut subject_pred_sets: HashMap<&str, HashSet<&str>> = HashMap::new();
        for (pred, subject_map) in &self.property_data {
            for subject in subject_map.keys() {
                subject_pred_sets
                    .entry(subject.as_str())
                    .or_default()
                    .insert(pred.as_str());
            }
        }

        // Count how many subjects share the exact same predicate set.
        let mut pattern_counts: HashMap<Vec<&str>, usize> = HashMap::new();
        for preds in subject_pred_sets.values() {
            let mut sorted: Vec<&str> = preds.iter().copied().collect();
            sorted.sort();
            *pattern_counts.entry(sorted).or_insert(0) += 1;
        }

        let total_subjects = self.subjects.len();
        let mut patterns: Vec<SubjectPropertyPattern> = pattern_counts
            .into_iter()
            .filter_map(|(preds, count)| {
                let coverage = count as f64 / total_subjects as f64;
                if coverage >= min_coverage {
                    Some(SubjectPropertyPattern {
                        predicates: preds.into_iter().map(String::from).collect(),
                        subject_count: count,
                        coverage,
                    })
                } else {
                    None
                }
            })
            .collect();

        patterns.sort_by(|a, b| {
            b.coverage
                .partial_cmp(&a.coverage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        patterns
    }

    /// Generate a complete profile report.
    pub fn report(&self) -> ProfileReport {
        ProfileReport {
            total_triples: self.total_triples,
            distinct_subjects: self.subjects.len(),
            distinct_predicates: self.property_data.len(),
            property_stats: self.property_stats(),
            outliers: self.detect_outliers(),
            patterns: self.detect_patterns(0.1),
        }
    }

    /// Total number of triples ingested so far.
    pub fn total_triples(&self) -> usize {
        self.total_triples
    }

    /// Number of distinct subjects seen.
    pub fn distinct_subjects(&self) -> usize {
        self.subjects.len()
    }

    /// Number of distinct predicates seen.
    pub fn distinct_predicates(&self) -> usize {
        self.property_data.len()
    }

    /// Reset the profiler, discarding all accumulated data.
    pub fn clear(&mut self) {
        self.property_data.clear();
        self.datatype_data.clear();
        self.language_data.clear();
        self.subjects.clear();
        self.total_triples = 0;
    }
}

impl Default for DataProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the p-th percentile of a sorted slice (linear interpolation).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (sorted.len() as f64 - 1.0);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;
    if lower == upper || upper >= sorted.len() {
        sorted[lower.min(sorted.len() - 1)]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_observations() -> Vec<TripleObservation> {
        vec![
            TripleObservation::new("s1", "name", "Alice"),
            TripleObservation::new("s1", "age", "30").with_datatype("xsd:integer"),
            TripleObservation::new("s1", "knows", "http://example.org/bob"),
            TripleObservation::new("s2", "name", "Bob"),
            TripleObservation::new("s2", "age", "25").with_datatype("xsd:integer"),
            TripleObservation::new("s2", "knows", "http://example.org/charlie"),
            TripleObservation::new("s3", "name", "Charlie").with_language("en"),
            TripleObservation::new("s3", "age", "35").with_datatype("xsd:integer"),
        ]
    }

    // ── ValueType::classify ──────────────────────────────────────────────

    #[test]
    fn test_classify_iri_http() {
        assert_eq!(ValueType::classify("http://example.org/x"), ValueType::Iri);
    }

    #[test]
    fn test_classify_iri_https() {
        assert_eq!(ValueType::classify("https://example.org/x"), ValueType::Iri);
    }

    #[test]
    fn test_classify_blank_node() {
        assert_eq!(ValueType::classify("_:b0"), ValueType::BlankNode);
    }

    #[test]
    fn test_classify_literal() {
        assert_eq!(ValueType::classify("hello"), ValueType::Literal);
    }

    #[test]
    fn test_classify_number_literal() {
        assert_eq!(ValueType::classify("42"), ValueType::Literal);
    }

    // ── TripleObservation construction ────────────────────────────────────

    #[test]
    fn test_triple_observation_new() {
        let obs = TripleObservation::new("s", "p", "o");
        assert_eq!(obs.subject, "s");
        assert_eq!(obs.predicate, "p");
        assert_eq!(obs.object, "o");
        assert!(obs.datatype.is_none());
        assert!(obs.language_tag.is_none());
    }

    #[test]
    fn test_triple_observation_with_datatype() {
        let obs = TripleObservation::new("s", "p", "42").with_datatype("xsd:integer");
        assert_eq!(obs.datatype.as_deref(), Some("xsd:integer"));
    }

    #[test]
    fn test_triple_observation_with_language() {
        let obs = TripleObservation::new("s", "p", "hello").with_language("en");
        assert_eq!(obs.language_tag.as_deref(), Some("en"));
    }

    // ── DataProfiler construction ────────────────────────────────────────

    #[test]
    fn test_new_profiler_empty() {
        let profiler = DataProfiler::new();
        assert_eq!(profiler.total_triples(), 0);
        assert_eq!(profiler.distinct_subjects(), 0);
        assert_eq!(profiler.distinct_predicates(), 0);
    }

    #[test]
    fn test_default_profiler() {
        let profiler = DataProfiler::default();
        assert_eq!(profiler.total_triples(), 0);
    }

    // ── Single add ───────────────────────────────────────────────────────

    #[test]
    fn test_add_single_triple() {
        let mut profiler = DataProfiler::new();
        profiler.add(&TripleObservation::new("s1", "name", "Alice"));
        assert_eq!(profiler.total_triples(), 1);
        assert_eq!(profiler.distinct_subjects(), 1);
        assert_eq!(profiler.distinct_predicates(), 1);
    }

    #[test]
    fn test_add_increments_count() {
        let mut profiler = DataProfiler::new();
        profiler.add(&TripleObservation::new("s1", "p", "o1"));
        profiler.add(&TripleObservation::new("s1", "p", "o2"));
        assert_eq!(profiler.total_triples(), 2);
    }

    // ── Batch add ────────────────────────────────────────────────────────

    #[test]
    fn test_add_batch() {
        let mut profiler = DataProfiler::new();
        let obs = sample_observations();
        profiler.add_batch(&obs);
        assert_eq!(profiler.total_triples(), 8);
        assert_eq!(profiler.distinct_subjects(), 3);
    }

    // ── Property stats ───────────────────────────────────────────────────

    #[test]
    fn test_property_stats_count() {
        let mut profiler = DataProfiler::new();
        profiler.add_batch(&sample_observations());
        let stats = profiler.property_stats();
        // name, age, knows → 3 predicates
        assert_eq!(stats.len(), 3);
    }

    #[test]
    fn test_property_stats_triple_count() {
        let mut profiler = DataProfiler::new();
        profiler.add_batch(&sample_observations());
        let stats = profiler.property_stats();
        let name_stats = stats.iter().find(|s| s.predicate == "name");
        assert!(name_stats.is_some());
        assert_eq!(name_stats.map(|s| s.triple_count).unwrap_or(0), 3);
    }

    #[test]
    fn test_property_stats_cardinality() {
        let mut profiler = DataProfiler::new();
        profiler.add(&TripleObservation::new("s1", "tag", "a"));
        profiler.add(&TripleObservation::new("s1", "tag", "b"));
        profiler.add(&TripleObservation::new("s2", "tag", "c"));
        let stats = profiler.property_stats();
        let tag_stats = stats.iter().find(|s| s.predicate == "tag");
        assert!(tag_stats.is_some());
        let ts = tag_stats.expect("tag stats");
        assert_eq!(ts.min_cardinality, 1); // s2 has 1
        assert_eq!(ts.max_cardinality, 2); // s1 has 2
    }

    #[test]
    fn test_property_stats_avg_cardinality() {
        let mut profiler = DataProfiler::new();
        profiler.add(&TripleObservation::new("s1", "p", "o1"));
        profiler.add(&TripleObservation::new("s1", "p", "o2"));
        profiler.add(&TripleObservation::new("s1", "p", "o3"));
        profiler.add(&TripleObservation::new("s2", "p", "o4"));
        let stats = profiler.property_stats();
        let ps = &stats[0];
        // s1 has 3, s2 has 1 → avg = 2.0
        assert!((ps.avg_cardinality - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_property_stats_value_type_distribution() {
        let mut profiler = DataProfiler::new();
        profiler.add(&TripleObservation::new("s1", "rel", "http://example.org/x"));
        profiler.add(&TripleObservation::new("s1", "rel", "_:b0"));
        profiler.add(&TripleObservation::new("s1", "rel", "literal_value"));
        let stats = profiler.property_stats();
        let rs = &stats[0];
        assert_eq!(rs.value_type_distribution.get(&ValueType::Iri), Some(&1));
        assert_eq!(
            rs.value_type_distribution.get(&ValueType::BlankNode),
            Some(&1)
        );
        assert_eq!(
            rs.value_type_distribution.get(&ValueType::Literal),
            Some(&1)
        );
    }

    #[test]
    fn test_property_stats_distinct_values() {
        let mut profiler = DataProfiler::new();
        profiler.add(&TripleObservation::new("s1", "p", "v1"));
        profiler.add(&TripleObservation::new("s2", "p", "v1")); // duplicate value
        profiler.add(&TripleObservation::new("s3", "p", "v2"));
        let stats = profiler.property_stats();
        assert_eq!(stats[0].distinct_values, 2);
    }

    // ── Datatype distribution ────────────────────────────────────────────

    #[test]
    fn test_datatype_distribution() {
        let mut profiler = DataProfiler::new();
        profiler.add(&TripleObservation::new("s1", "age", "30").with_datatype("xsd:integer"));
        profiler.add(&TripleObservation::new("s2", "age", "25").with_datatype("xsd:integer"));
        profiler.add(&TripleObservation::new("s3", "age", "3.5").with_datatype("xsd:decimal"));
        let stats = profiler.property_stats();
        let age_stats = stats.iter().find(|s| s.predicate == "age");
        assert!(age_stats.is_some());
        let dt = &age_stats.expect("age").datatype_distribution;
        assert_eq!(dt.get("xsd:integer"), Some(&2));
        assert_eq!(dt.get("xsd:decimal"), Some(&1));
    }

    // ── Language distribution ────────────────────────────────────────────

    #[test]
    fn test_language_distribution() {
        let mut profiler = DataProfiler::new();
        profiler.add(&TripleObservation::new("s1", "label", "Hello").with_language("en"));
        profiler.add(&TripleObservation::new("s2", "label", "Hola").with_language("es"));
        profiler.add(&TripleObservation::new("s3", "label", "Hi").with_language("en"));
        let stats = profiler.property_stats();
        let ls = stats.iter().find(|s| s.predicate == "label");
        let lang = &ls.expect("label").language_distribution;
        assert_eq!(lang.get("en"), Some(&2));
        assert_eq!(lang.get("es"), Some(&1));
    }

    // ── Outlier detection ────────────────────────────────────────────────

    #[test]
    fn test_detect_outliers_empty() {
        let profiler = DataProfiler::new();
        assert!(profiler.detect_outliers().is_empty());
    }

    #[test]
    fn test_detect_outliers_finds_extreme() {
        let mut profiler = DataProfiler::new();
        // Normal range: 10, 12, 11, 13 → outlier: 100
        for (i, v) in [10, 12, 11, 13, 100].iter().enumerate() {
            profiler.add(&TripleObservation::new(
                format!("s{i}"),
                "score",
                v.to_string(),
            ));
        }
        let outliers = profiler.detect_outliers();
        assert!(!outliers.is_empty());
        assert!(outliers.iter().any(|o| o.object == "100"));
    }

    #[test]
    fn test_detect_outliers_no_outliers() {
        let mut profiler = DataProfiler::new();
        for (i, v) in [10, 11, 12, 13, 14].iter().enumerate() {
            profiler.add(&TripleObservation::new(
                format!("s{i}"),
                "score",
                v.to_string(),
            ));
        }
        let outliers = profiler.detect_outliers();
        assert!(outliers.is_empty());
    }

    #[test]
    fn test_detect_outliers_insufficient_data() {
        let mut profiler = DataProfiler::new();
        profiler.add(&TripleObservation::new("s1", "val", "10"));
        profiler.add(&TripleObservation::new("s2", "val", "100"));
        // Only 2 data points → not enough for IQR (needs >= 4)
        assert!(profiler.detect_outliers().is_empty());
    }

    // ── Pattern detection ────────────────────────────────────────────────

    #[test]
    fn test_detect_patterns_empty() {
        let profiler = DataProfiler::new();
        assert!(profiler.detect_patterns(0.5).is_empty());
    }

    #[test]
    fn test_detect_patterns_common() {
        let mut profiler = DataProfiler::new();
        // All subjects share {name, age}
        for i in 0..5 {
            profiler.add(&TripleObservation::new(
                format!("s{i}"),
                "name",
                format!("Name{i}"),
            ));
            profiler.add(&TripleObservation::new(
                format!("s{i}"),
                "age",
                format!("{}", 20 + i),
            ));
        }
        let patterns = profiler.detect_patterns(0.5);
        assert!(!patterns.is_empty());
        let first = &patterns[0];
        assert_eq!(first.subject_count, 5);
        assert!((first.coverage - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_detect_patterns_low_coverage_excluded() {
        let mut profiler = DataProfiler::new();
        // Only 1 out of 10 subjects has "rare"
        for i in 0..10 {
            profiler.add(&TripleObservation::new(format!("s{i}"), "common", "val"));
        }
        profiler.add(&TripleObservation::new("s0", "rare", "val"));
        let patterns = profiler.detect_patterns(0.5);
        // The "common + rare" pattern covers only 1/10 = 0.1 < 0.5
        // Only the "common" pattern (9/10 = 0.9) should appear
        assert!(patterns
            .iter()
            .all(|p| !p.predicates.contains(&"rare".to_string()) || p.coverage >= 0.5));
    }

    // ── Profile report ───────────────────────────────────────────────────

    #[test]
    fn test_report_totals() {
        let mut profiler = DataProfiler::new();
        profiler.add_batch(&sample_observations());
        let report = profiler.report();
        assert_eq!(report.total_triples, 8);
        assert_eq!(report.distinct_subjects, 3);
        assert_eq!(report.distinct_predicates, 3);
    }

    #[test]
    fn test_report_has_property_stats() {
        let mut profiler = DataProfiler::new();
        profiler.add_batch(&sample_observations());
        let report = profiler.report();
        assert!(!report.property_stats.is_empty());
    }

    // ── Clear ────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_resets_all() {
        let mut profiler = DataProfiler::new();
        profiler.add_batch(&sample_observations());
        assert!(profiler.total_triples() > 0);
        profiler.clear();
        assert_eq!(profiler.total_triples(), 0);
        assert_eq!(profiler.distinct_subjects(), 0);
        assert_eq!(profiler.distinct_predicates(), 0);
    }

    // ── with_outlier_iqr ─────────────────────────────────────────────────

    #[test]
    fn test_with_outlier_iqr() {
        let profiler = DataProfiler::new().with_outlier_iqr(3.0);
        assert!((profiler.outlier_iqr_multiplier - 3.0).abs() < 1e-10);
    }

    // ── percentile helper ────────────────────────────────────────────────

    #[test]
    fn test_percentile_empty() {
        assert!((percentile(&[], 50.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_single() {
        assert!((percentile(&[7.0], 50.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_median() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 50.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_quartiles() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let q1 = percentile(&data, 25.0);
        let q3 = percentile(&data, 75.0);
        assert!(q1 < q3);
        assert!(q1 > 1.0);
        assert!(q3 < 8.0);
    }

    // ── Outlier struct ───────────────────────────────────────────────────

    #[test]
    fn test_outlier_fields() {
        let o = Outlier {
            subject: "s1".into(),
            predicate: "p".into(),
            object: "o".into(),
            reason: "test".into(),
        };
        assert_eq!(o.subject, "s1");
        assert_eq!(o.reason, "test");
    }

    // ── SubjectPropertyPattern fields ────────────────────────────────────

    #[test]
    fn test_subject_property_pattern_fields() {
        let p = SubjectPropertyPattern {
            predicates: vec!["name".into()],
            subject_count: 5,
            coverage: 0.8,
        };
        assert_eq!(p.subject_count, 5);
        assert!((p.coverage - 0.8).abs() < 1e-10);
    }

    // ── PropertyStats subject_count ──────────────────────────────────────

    #[test]
    fn test_property_stats_subject_count() {
        let mut profiler = DataProfiler::new();
        profiler.add(&TripleObservation::new("s1", "p", "v1"));
        profiler.add(&TripleObservation::new("s2", "p", "v2"));
        profiler.add(&TripleObservation::new("s3", "p", "v3"));
        let stats = profiler.property_stats();
        assert_eq!(stats[0].subject_count, 3);
    }

    // ── Incremental profiling ────────────────────────────────────────────

    #[test]
    fn test_incremental_profiling() {
        let mut profiler = DataProfiler::new();
        profiler.add(&TripleObservation::new("s1", "name", "Alice"));
        assert_eq!(profiler.total_triples(), 1);
        profiler.add(&TripleObservation::new("s2", "name", "Bob"));
        assert_eq!(profiler.total_triples(), 2);
        let stats = profiler.property_stats();
        assert_eq!(stats[0].subject_count, 2);
    }

    // ── Empty property stats ─────────────────────────────────────────────

    #[test]
    fn test_property_stats_empty_profiler() {
        let profiler = DataProfiler::new();
        assert!(profiler.property_stats().is_empty());
    }
}
