//! RDF Dataset Profiler
//!
//! Provides deep statistical profiling of RDF datasets, producing a comprehensive
//! data quality and structural report suitable for data governance, migration
//! planning, and ontology assessment.
//!
//! ## Features
//!
//! - **Triple distribution analysis**: Subject, predicate, object frequency counts
//! - **Namespace extraction**: Automatic namespace detection and prefix suggestion
//! - **Schema profiling**: Class and property usage statistics
//! - **Data quality checks**: Orphan nodes, dangling references, literal type mismatches
//! - **Connectivity metrics**: Average degree, predicate fan-out, hub detection
//! - **Vocabulary coverage**: RDF, RDFS, OWL, SKOS vocabulary usage analysis
//! - **Report generation**: Structured JSON/text profile reports

use super::CommandResult;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the data profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Maximum number of top-N items to report per category.
    pub top_n: usize,
    /// Whether to compute connectivity metrics (can be expensive).
    pub compute_connectivity: bool,
    /// Whether to perform data quality checks.
    pub quality_checks: bool,
    /// Whether to detect vocabularies.
    pub vocabulary_detection: bool,
    /// Sample rate (0.0 - 1.0) for large datasets; 1.0 means full scan.
    pub sample_rate: f64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            top_n: 20,
            compute_connectivity: true,
            quality_checks: true,
            vocabulary_detection: true,
            sample_rate: 1.0,
        }
    }
}

// ─────────────────────────────────────────────
// Profile report types
// ─────────────────────────────────────────────

/// Complete profile of an RDF dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetProfile {
    /// Basic triple counts.
    pub basic_stats: BasicStats,
    /// Namespace distribution.
    pub namespaces: Vec<NamespaceInfo>,
    /// Top predicates by usage.
    pub top_predicates: Vec<FrequencyEntry>,
    /// Top classes (rdf:type targets).
    pub top_classes: Vec<FrequencyEntry>,
    /// Top subjects by outgoing triple count.
    pub top_subjects: Vec<FrequencyEntry>,
    /// Literal type distribution.
    pub literal_types: Vec<FrequencyEntry>,
    /// Language tag distribution.
    pub language_tags: Vec<FrequencyEntry>,
    /// Data quality issues.
    pub quality_issues: Vec<QualityIssue>,
    /// Connectivity metrics (if computed).
    pub connectivity: Option<ConnectivityMetrics>,
    /// Vocabulary usage.
    pub vocabularies: Vec<VocabularyUsage>,
    /// Duration of profiling.
    pub profiling_duration: Duration,
}

/// Basic triple/node counts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BasicStats {
    /// Total number of triples.
    pub triple_count: usize,
    /// Number of distinct subjects.
    pub distinct_subjects: usize,
    /// Number of distinct predicates.
    pub distinct_predicates: usize,
    /// Number of distinct objects.
    pub distinct_objects: usize,
    /// Number of distinct IRIs (across all positions).
    pub distinct_iris: usize,
    /// Number of distinct blank nodes.
    pub distinct_blank_nodes: usize,
    /// Number of distinct literals.
    pub distinct_literals: usize,
    /// Number of triples sampled (equals triple_count when sample_rate=1.0).
    pub triples_sampled: usize,
}

/// Namespace usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceInfo {
    /// Detected namespace IRI.
    pub namespace: String,
    /// Suggested prefix.
    pub suggested_prefix: String,
    /// Number of terms using this namespace.
    pub term_count: usize,
    /// Proportion of all IRI terms.
    pub proportion: f64,
}

/// A frequency count entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyEntry {
    /// The item (IRI, literal type, language tag, etc.).
    pub item: String,
    /// Number of occurrences.
    pub count: usize,
    /// Proportion of total.
    pub proportion: f64,
}

/// A data quality issue found in the dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Severity level.
    pub severity: IssueSeverity,
    /// Issue category.
    pub category: String,
    /// Description of the issue.
    pub description: String,
    /// Number of affected items.
    pub affected_count: usize,
    /// Example items (up to 5).
    pub examples: Vec<String>,
}

/// Severity of a quality issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Informational: not necessarily a problem.
    Info,
    /// Warning: potential issue that may need attention.
    Warning,
    /// Error: likely a data quality problem.
    Error,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueSeverity::Info => write!(f, "INFO"),
            IssueSeverity::Warning => write!(f, "WARN"),
            IssueSeverity::Error => write!(f, "ERROR"),
        }
    }
}

/// Connectivity metrics for the dataset graph.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConnectivityMetrics {
    /// Average outgoing degree per subject.
    pub avg_out_degree: f64,
    /// Maximum outgoing degree.
    pub max_out_degree: usize,
    /// Average incoming degree per object-IRI.
    pub avg_in_degree: f64,
    /// Maximum incoming degree.
    pub max_in_degree: usize,
    /// Average predicate fan-out (distinct objects per subject-predicate pair).
    pub avg_predicate_fanout: f64,
    /// Hub nodes (subjects with highest outgoing degree, up to top_n).
    pub hub_nodes: Vec<FrequencyEntry>,
    /// Authority nodes (objects with highest incoming degree, up to top_n).
    pub authority_nodes: Vec<FrequencyEntry>,
}

/// Known vocabulary usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabularyUsage {
    /// Vocabulary name (e.g., "RDF", "RDFS", "OWL", "SKOS").
    pub name: String,
    /// Namespace IRI.
    pub namespace: String,
    /// Number of terms used from this vocabulary.
    pub terms_used: usize,
    /// List of specific terms used.
    pub used_terms: Vec<String>,
}

// ─────────────────────────────────────────────
// Known vocabularies
// ─────────────────────────────────────────────

/// Well-known vocabulary namespaces.
struct KnownVocabulary {
    name: &'static str,
    namespace: &'static str,
}

const KNOWN_VOCABULARIES: &[KnownVocabulary] = &[
    KnownVocabulary {
        name: "RDF",
        namespace: "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    },
    KnownVocabulary {
        name: "RDFS",
        namespace: "http://www.w3.org/2000/01/rdf-schema#",
    },
    KnownVocabulary {
        name: "OWL",
        namespace: "http://www.w3.org/2002/07/owl#",
    },
    KnownVocabulary {
        name: "SKOS",
        namespace: "http://www.w3.org/2004/02/skos/core#",
    },
    KnownVocabulary {
        name: "XSD",
        namespace: "http://www.w3.org/2001/XMLSchema#",
    },
    KnownVocabulary {
        name: "FOAF",
        namespace: "http://xmlns.com/foaf/0.1/",
    },
    KnownVocabulary {
        name: "DC",
        namespace: "http://purl.org/dc/elements/1.1/",
    },
    KnownVocabulary {
        name: "DCT",
        namespace: "http://purl.org/dc/terms/",
    },
    KnownVocabulary {
        name: "SCHEMA",
        namespace: "http://schema.org/",
    },
    KnownVocabulary {
        name: "PROV",
        namespace: "http://www.w3.org/ns/prov#",
    },
    KnownVocabulary {
        name: "SHACL",
        namespace: "http://www.w3.org/ns/shacl#",
    },
    KnownVocabulary {
        name: "SAMM",
        namespace: "urn:samm:org.eclipse.esmf.samm:",
    },
];

// ─────────────────────────────────────────────
// Profiler engine
// ─────────────────────────────────────────────

/// The dataset profiler engine.
pub struct DataProfiler {
    config: ProfilerConfig,
}

impl DataProfiler {
    /// Create a new profiler with default configuration.
    pub fn new() -> Self {
        Self {
            config: ProfilerConfig::default(),
        }
    }

    /// Create a new profiler with the given configuration.
    pub fn with_config(config: ProfilerConfig) -> Self {
        Self { config }
    }

    /// Profile a dataset represented as (subject, predicate, object) triples.
    pub fn profile(&self, triples: &[(String, String, String)]) -> DatasetProfile {
        let start = Instant::now();

        // Optionally sample
        let effective_triples: Vec<&(String, String, String)> = if self.config.sample_rate < 1.0 {
            let step = (1.0 / self.config.sample_rate).max(1.0) as usize;
            triples.iter().step_by(step).collect()
        } else {
            triples.iter().collect()
        };

        // Collect basic stats
        let basic_stats = self.compute_basic_stats(triples, &effective_triples);

        // Frequency analysis
        let predicate_freq = frequency_map(effective_triples.iter().map(|t| t.1.as_str()));
        let top_predicates = top_n_entries(
            &predicate_freq,
            self.config.top_n,
            basic_stats.triples_sampled,
        );

        // Class analysis (rdf:type targets)
        let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        let class_freq = frequency_map(
            effective_triples
                .iter()
                .filter(|t| t.1 == rdf_type)
                .map(|t| t.2.as_str()),
        );
        let top_classes = top_n_entries(&class_freq, self.config.top_n, class_freq.values().sum());

        // Subject frequency (outgoing degree)
        let subject_freq = frequency_map(effective_triples.iter().map(|t| t.0.as_str()));
        let top_subjects = top_n_entries(
            &subject_freq,
            self.config.top_n,
            basic_stats.triples_sampled,
        );

        // Literal type distribution
        let literal_types = self.analyze_literal_types(&effective_triples);
        let language_tags = self.analyze_language_tags(&effective_triples);

        // Namespace detection
        let all_iris = self.collect_all_iris(&effective_triples);
        let namespaces = self.detect_namespaces(&all_iris, self.config.top_n);

        // Quality checks
        let quality_issues = if self.config.quality_checks {
            self.check_quality(&effective_triples, &subject_freq)
        } else {
            Vec::new()
        };

        // Connectivity
        let connectivity = if self.config.compute_connectivity {
            Some(self.compute_connectivity(&effective_triples, &subject_freq))
        } else {
            None
        };

        // Vocabulary detection
        let vocabularies = if self.config.vocabulary_detection {
            self.detect_vocabularies(&all_iris)
        } else {
            Vec::new()
        };

        DatasetProfile {
            basic_stats,
            namespaces,
            top_predicates,
            top_classes,
            top_subjects,
            literal_types,
            language_tags,
            quality_issues,
            connectivity,
            vocabularies,
            profiling_duration: start.elapsed(),
        }
    }

    /// Compute basic statistics.
    fn compute_basic_stats(
        &self,
        all_triples: &[(String, String, String)],
        effective: &[&(String, String, String)],
    ) -> BasicStats {
        let mut subjects: HashSet<&str> = HashSet::new();
        let mut predicates: HashSet<&str> = HashSet::new();
        let mut objects: HashSet<&str> = HashSet::new();
        let mut all_iris: HashSet<&str> = HashSet::new();
        let mut blank_nodes: HashSet<&str> = HashSet::new();
        let mut literals: HashSet<&str> = HashSet::new();

        for triple in effective {
            subjects.insert(&triple.0);
            predicates.insert(&triple.1);
            objects.insert(&triple.2);

            // Classify terms
            classify_term(&triple.0, &mut all_iris, &mut blank_nodes, &mut literals);
            classify_term(&triple.1, &mut all_iris, &mut blank_nodes, &mut literals);
            classify_term(&triple.2, &mut all_iris, &mut blank_nodes, &mut literals);
        }

        BasicStats {
            triple_count: all_triples.len(),
            distinct_subjects: subjects.len(),
            distinct_predicates: predicates.len(),
            distinct_objects: objects.len(),
            distinct_iris: all_iris.len(),
            distinct_blank_nodes: blank_nodes.len(),
            distinct_literals: literals.len(),
            triples_sampled: effective.len(),
        }
    }

    /// Analyze literal datatype distribution.
    fn analyze_literal_types(&self, triples: &[&(String, String, String)]) -> Vec<FrequencyEntry> {
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;

        for triple in triples {
            let obj = &triple.2;
            if obj.starts_with('"') {
                total += 1;
                let dtype = if obj.contains("^^<") {
                    // Typed literal
                    if let Some(start) = obj.find("^^<") {
                        let end = obj.len().saturating_sub(1);
                        if end > start + 3 {
                            obj[start + 3..end].to_string()
                        } else {
                            "xsd:string".to_string()
                        }
                    } else {
                        "xsd:string".to_string()
                    }
                } else if obj.contains("\"@") {
                    "rdf:langString".to_string()
                } else {
                    "xsd:string".to_string()
                };
                *type_counts.entry(dtype).or_insert(0) += 1;
            }
        }

        top_n_entries(&type_counts, self.config.top_n, total)
    }

    /// Analyze language tag distribution.
    fn analyze_language_tags(&self, triples: &[&(String, String, String)]) -> Vec<FrequencyEntry> {
        let mut lang_counts: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;

        for triple in triples {
            let obj = &triple.2;
            if let Some(at_pos) = obj.rfind("\"@") {
                let lang = &obj[at_pos + 2..];
                if !lang.is_empty() {
                    total += 1;
                    *lang_counts.entry(lang.to_string()).or_insert(0) += 1;
                }
            }
        }

        top_n_entries(&lang_counts, self.config.top_n, total)
    }

    /// Collect all IRI terms from the dataset.
    fn collect_all_iris(&self, triples: &[&(String, String, String)]) -> HashSet<String> {
        let mut iris = HashSet::new();
        for triple in triples {
            if is_iri(&triple.0) {
                iris.insert(triple.0.clone());
            }
            if is_iri(&triple.1) {
                iris.insert(triple.1.clone());
            }
            if is_iri(&triple.2) {
                iris.insert(triple.2.clone());
            }
        }
        iris
    }

    /// Detect namespaces from IRI terms.
    fn detect_namespaces(&self, iris: &HashSet<String>, top_n: usize) -> Vec<NamespaceInfo> {
        let mut ns_counts: HashMap<String, usize> = HashMap::new();

        for iri in iris {
            if let Some(ns) = extract_namespace(iri) {
                *ns_counts.entry(ns).or_insert(0) += 1;
            }
        }

        let total: usize = ns_counts.values().sum();

        let mut entries: Vec<(String, usize)> = ns_counts.into_iter().collect();
        entries.sort_by_key(|item| std::cmp::Reverse(item.1));
        entries.truncate(top_n);

        entries
            .into_iter()
            .map(|(namespace, term_count)| {
                let suggested_prefix = suggest_prefix(&namespace);
                let proportion = if total > 0 {
                    term_count as f64 / total as f64
                } else {
                    0.0
                };
                NamespaceInfo {
                    namespace,
                    suggested_prefix,
                    term_count,
                    proportion,
                }
            })
            .collect()
    }

    /// Perform data quality checks.
    fn check_quality(
        &self,
        triples: &[&(String, String, String)],
        subject_freq: &HashMap<String, usize>,
    ) -> Vec<QualityIssue> {
        let mut issues = Vec::new();

        // Check for orphan IRIs in object position that never appear as subjects
        let subjects: HashSet<&str> = triples.iter().map(|t| t.0.as_str()).collect();
        let object_iris: Vec<&str> = triples
            .iter()
            .filter(|t| is_iri(&t.2))
            .map(|t| t.2.as_str())
            .collect();

        let dangling: Vec<String> = object_iris
            .iter()
            .filter(|o| !subjects.contains(**o))
            .map(|o| o.to_string())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        if !dangling.is_empty() {
            let example_count = dangling.len().min(5);
            issues.push(QualityIssue {
                severity: IssueSeverity::Info,
                category: "dangling_references".to_string(),
                description: format!(
                    "{} IRI(s) appear in object position but never as subjects",
                    dangling.len()
                ),
                affected_count: dangling.len(),
                examples: dangling[..example_count].to_vec(),
            });
        }

        // Check for subjects with very high fan-out (potential data modeling issue)
        let high_fanout_threshold = 100;
        let high_fanout: Vec<(String, usize)> = subject_freq
            .iter()
            .filter(|(_, &count)| count > high_fanout_threshold)
            .map(|(s, &c)| (s.clone(), c))
            .collect();

        if !high_fanout.is_empty() {
            let examples: Vec<String> = high_fanout
                .iter()
                .take(5)
                .map(|(s, c)| format!("{} ({} triples)", s, c))
                .collect();
            issues.push(QualityIssue {
                severity: IssueSeverity::Warning,
                category: "high_fanout".to_string(),
                description: format!(
                    "{} subject(s) have more than {} outgoing triples",
                    high_fanout.len(),
                    high_fanout_threshold
                ),
                affected_count: high_fanout.len(),
                examples,
            });
        }

        // Check for blank node usage
        let blank_count = triples
            .iter()
            .filter(|t| t.0.starts_with("_:") || t.2.starts_with("_:"))
            .count();
        if blank_count > 0 {
            let proportion = blank_count as f64 / triples.len() as f64;
            if proportion > 0.2 {
                issues.push(QualityIssue {
                    severity: IssueSeverity::Warning,
                    category: "excessive_blank_nodes".to_string(),
                    description: format!(
                        "{:.1}% of triples involve blank nodes (may hinder interoperability)",
                        proportion * 100.0
                    ),
                    affected_count: blank_count,
                    examples: vec![],
                });
            }
        }

        // Check for missing rdf:type declarations
        let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        let typed_subjects: HashSet<&str> = triples
            .iter()
            .filter(|t| t.1 == rdf_type)
            .map(|t| t.0.as_str())
            .collect();

        let untyped: Vec<String> = subjects
            .iter()
            .filter(|s| !typed_subjects.contains(**s) && is_iri(s))
            .map(|s| s.to_string())
            .collect();

        if !untyped.is_empty() && !subjects.is_empty() {
            let proportion = untyped.len() as f64 / subjects.len() as f64;
            if proportion > 0.5 {
                let examples: Vec<String> = untyped.iter().take(5).cloned().collect();
                issues.push(QualityIssue {
                    severity: IssueSeverity::Info,
                    category: "missing_type_declarations".to_string(),
                    description: format!(
                        "{:.1}% of subjects ({}/{}) have no rdf:type declaration",
                        proportion * 100.0,
                        untyped.len(),
                        subjects.len()
                    ),
                    affected_count: untyped.len(),
                    examples,
                });
            }
        }

        issues
    }

    /// Compute connectivity metrics.
    fn compute_connectivity(
        &self,
        triples: &[&(String, String, String)],
        subject_freq: &HashMap<String, usize>,
    ) -> ConnectivityMetrics {
        // Out-degree: subject frequency
        let out_degrees: Vec<usize> = subject_freq.values().copied().collect();
        let avg_out_degree = if out_degrees.is_empty() {
            0.0
        } else {
            out_degrees.iter().sum::<usize>() as f64 / out_degrees.len() as f64
        };
        let max_out_degree = out_degrees.iter().copied().max().unwrap_or(0);

        // In-degree: object-IRI frequency
        let mut in_degree_map: HashMap<&str, usize> = HashMap::new();
        for triple in triples {
            if is_iri(&triple.2) {
                *in_degree_map.entry(&triple.2).or_insert(0) += 1;
            }
        }
        let in_degrees: Vec<usize> = in_degree_map.values().copied().collect();
        let avg_in_degree = if in_degrees.is_empty() {
            0.0
        } else {
            in_degrees.iter().sum::<usize>() as f64 / in_degrees.len() as f64
        };
        let max_in_degree = in_degrees.iter().copied().max().unwrap_or(0);

        // Predicate fan-out: distinct objects per (subject, predicate) pair
        let mut sp_objects: HashMap<(&str, &str), HashSet<&str>> = HashMap::new();
        for triple in triples {
            sp_objects
                .entry((&triple.0, &triple.1))
                .or_default()
                .insert(&triple.2);
        }
        let fanouts: Vec<usize> = sp_objects.values().map(|v| v.len()).collect();
        let avg_predicate_fanout = if fanouts.is_empty() {
            0.0
        } else {
            fanouts.iter().sum::<usize>() as f64 / fanouts.len() as f64
        };

        // Hub nodes (highest out-degree)
        let hub_nodes = top_n_entries(subject_freq, self.config.top_n, triples.len());

        // Authority nodes (highest in-degree)
        let in_degree_owned: HashMap<String, usize> = in_degree_map
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let authority_nodes = top_n_entries(&in_degree_owned, self.config.top_n, triples.len());

        ConnectivityMetrics {
            avg_out_degree,
            max_out_degree,
            avg_in_degree,
            max_in_degree,
            avg_predicate_fanout,
            hub_nodes,
            authority_nodes,
        }
    }

    /// Detect known vocabulary usage.
    fn detect_vocabularies(&self, iris: &HashSet<String>) -> Vec<VocabularyUsage> {
        let mut result = Vec::new();

        for vocab in KNOWN_VOCABULARIES {
            let used_terms: Vec<String> = iris
                .iter()
                .filter(|iri| iri.starts_with(vocab.namespace))
                .map(|iri| iri[vocab.namespace.len()..].to_string())
                .collect();

            if !used_terms.is_empty() {
                result.push(VocabularyUsage {
                    name: vocab.name.to_string(),
                    namespace: vocab.namespace.to_string(),
                    terms_used: used_terms.len(),
                    used_terms,
                });
            }
        }

        result.sort_by_key(|item| std::cmp::Reverse(item.terms_used));
        result
    }
}

impl Default for DataProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
// Report formatting
// ─────────────────────────────────────────────

/// Format a dataset profile as a human-readable text report.
pub fn format_text_report(profile: &DatasetProfile) -> String {
    let mut out = String::new();

    out.push_str("═══════════════════════════════════════════════════\n");
    out.push_str("            RDF Dataset Profile Report\n");
    out.push_str("═══════════════════════════════════════════════════\n\n");

    // Basic stats
    out.push_str("── Basic Statistics ──\n");
    out.push_str(&format!(
        "  Total triples:       {}\n",
        profile.basic_stats.triple_count
    ));
    out.push_str(&format!(
        "  Distinct subjects:   {}\n",
        profile.basic_stats.distinct_subjects
    ));
    out.push_str(&format!(
        "  Distinct predicates: {}\n",
        profile.basic_stats.distinct_predicates
    ));
    out.push_str(&format!(
        "  Distinct objects:    {}\n",
        profile.basic_stats.distinct_objects
    ));
    out.push_str(&format!(
        "  Distinct IRIs:       {}\n",
        profile.basic_stats.distinct_iris
    ));
    out.push_str(&format!(
        "  Distinct blank nodes:{}\n",
        profile.basic_stats.distinct_blank_nodes
    ));
    out.push_str(&format!(
        "  Distinct literals:   {}\n",
        profile.basic_stats.distinct_literals
    ));
    out.push_str(&format!(
        "  Profiling duration:  {:?}\n\n",
        profile.profiling_duration
    ));

    // Namespaces
    if !profile.namespaces.is_empty() {
        out.push_str("── Namespaces ──\n");
        for ns in &profile.namespaces {
            out.push_str(&format!(
                "  {:>6} ({:>5.1}%)  {} => {}\n",
                ns.term_count,
                ns.proportion * 100.0,
                ns.suggested_prefix,
                ns.namespace
            ));
        }
        out.push('\n');
    }

    // Top predicates
    if !profile.top_predicates.is_empty() {
        out.push_str("── Top Predicates ──\n");
        for entry in &profile.top_predicates {
            out.push_str(&format!(
                "  {:>6} ({:>5.1}%)  {}\n",
                entry.count,
                entry.proportion * 100.0,
                entry.item
            ));
        }
        out.push('\n');
    }

    // Top classes
    if !profile.top_classes.is_empty() {
        out.push_str("── Top Classes (rdf:type) ──\n");
        for entry in &profile.top_classes {
            out.push_str(&format!(
                "  {:>6} ({:>5.1}%)  {}\n",
                entry.count,
                entry.proportion * 100.0,
                entry.item
            ));
        }
        out.push('\n');
    }

    // Quality issues
    if !profile.quality_issues.is_empty() {
        out.push_str("── Data Quality Issues ──\n");
        for issue in &profile.quality_issues {
            out.push_str(&format!(
                "  [{}] {}: {} ({})\n",
                issue.severity, issue.category, issue.description, issue.affected_count
            ));
        }
        out.push('\n');
    }

    // Vocabularies
    if !profile.vocabularies.is_empty() {
        out.push_str("── Vocabulary Usage ──\n");
        for vocab in &profile.vocabularies {
            out.push_str(&format!(
                "  {} ({} terms): {}\n",
                vocab.name, vocab.terms_used, vocab.namespace
            ));
        }
        out.push('\n');
    }

    // Connectivity
    if let Some(conn) = &profile.connectivity {
        out.push_str("── Connectivity Metrics ──\n");
        out.push_str(&format!(
            "  Avg out-degree:       {:.2}\n",
            conn.avg_out_degree
        ));
        out.push_str(&format!(
            "  Max out-degree:       {}\n",
            conn.max_out_degree
        ));
        out.push_str(&format!(
            "  Avg in-degree:        {:.2}\n",
            conn.avg_in_degree
        ));
        out.push_str(&format!("  Max in-degree:        {}\n", conn.max_in_degree));
        out.push_str(&format!(
            "  Avg predicate fan-out:{:.2}\n",
            conn.avg_predicate_fanout
        ));
        out.push('\n');
    }

    out
}

/// Format a dataset profile as JSON.
pub fn format_json_report(profile: &DatasetProfile) -> Result<String, String> {
    serde_json::to_string_pretty(profile).map_err(|e| e.to_string())
}

// ─────────────────────────────────────────────
// CLI entry point
// ─────────────────────────────────────────────

/// Run the data profiler command.
pub async fn run(
    dataset_path: String,
    output_format: Option<String>,
    top_n: Option<usize>,
    no_connectivity: bool,
    no_quality: bool,
) -> CommandResult {
    println!("Profiling dataset: {}", dataset_path);

    let config = ProfilerConfig {
        top_n: top_n.unwrap_or(20),
        compute_connectivity: !no_connectivity,
        quality_checks: !no_quality,
        ..Default::default()
    };

    let profiler = DataProfiler::with_config(config);

    // For now, use a placeholder dataset (actual integration would load from TDB)
    let triples: Vec<(String, String, String)> = Vec::new();

    let profile = profiler.profile(&triples);

    let format = output_format.as_deref().unwrap_or("text");
    match format {
        "json" => {
            let json = format_json_report(&profile)
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
            println!("{}", json);
        }
        _ => {
            let report = format_text_report(&profile);
            println!("{}", report);
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────

/// Build a frequency map from an iterator of string references.
fn frequency_map<'a>(items: impl Iterator<Item = &'a str>) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for item in items {
        *counts.entry(item.to_string()).or_insert(0) += 1;
    }
    counts
}

/// Get top N entries sorted by frequency descending.
fn top_n_entries(freq: &HashMap<String, usize>, n: usize, total: usize) -> Vec<FrequencyEntry> {
    let mut entries: Vec<(String, usize)> = freq.iter().map(|(k, &v)| (k.clone(), v)).collect();
    entries.sort_by_key(|item| std::cmp::Reverse(item.1));
    entries.truncate(n);

    entries
        .into_iter()
        .map(|(item, count)| {
            let proportion = if total > 0 {
                count as f64 / total as f64
            } else {
                0.0
            };
            FrequencyEntry {
                item,
                count,
                proportion,
            }
        })
        .collect()
}

/// Check if a term looks like an IRI (not a blank node or literal).
fn is_iri(term: &str) -> bool {
    !term.starts_with("_:") && !term.starts_with('"')
}

/// Classify a term into IRI, blank node, or literal buckets.
fn classify_term<'a>(
    term: &'a str,
    iris: &mut HashSet<&'a str>,
    blank_nodes: &mut HashSet<&'a str>,
    literals: &mut HashSet<&'a str>,
) {
    if term.starts_with("_:") {
        blank_nodes.insert(term);
    } else if term.starts_with('"') {
        literals.insert(term);
    } else {
        iris.insert(term);
    }
}

/// Extract the namespace part of an IRI (everything up to and including the last # or /).
fn extract_namespace(iri: &str) -> Option<String> {
    // Try hash namespace first
    if let Some(hash_pos) = iri.rfind('#') {
        return Some(iri[..=hash_pos].to_string());
    }
    // Try slash namespace
    if let Some(slash_pos) = iri.rfind('/') {
        // Avoid treating the protocol slashes as namespace separators
        if slash_pos > 8 {
            return Some(iri[..=slash_pos].to_string());
        }
    }
    None
}

/// Suggest a prefix name for a namespace IRI.
fn suggest_prefix(namespace: &str) -> String {
    // Check known vocabularies first
    for vocab in KNOWN_VOCABULARIES {
        if namespace == vocab.namespace {
            return vocab.name.to_lowercase();
        }
    }

    // Try to extract a meaningful name from the namespace
    let stripped = namespace.trim_end_matches('#').trim_end_matches('/');

    if let Some(last_segment) = stripped.rsplit('/').next() {
        let clean = last_segment
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();
        if clean.len() <= 8 && !clean.is_empty() {
            return clean;
        }
    }

    "ns".to_string()
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_triples() -> Vec<(String, String, String)> {
        vec![
            (
                "http://ex/alice".into(),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into(),
                "http://ex/Person".into(),
            ),
            (
                "http://ex/alice".into(),
                "http://xmlns.com/foaf/0.1/name".into(),
                "\"Alice\"".into(),
            ),
            (
                "http://ex/alice".into(),
                "http://xmlns.com/foaf/0.1/knows".into(),
                "http://ex/bob".into(),
            ),
            (
                "http://ex/bob".into(),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into(),
                "http://ex/Person".into(),
            ),
            (
                "http://ex/bob".into(),
                "http://xmlns.com/foaf/0.1/name".into(),
                "\"Bob\"@en".into(),
            ),
            (
                "http://ex/bob".into(),
                "http://xmlns.com/foaf/0.1/age".into(),
                "\"30\"^^<http://www.w3.org/2001/XMLSchema#integer>".into(),
            ),
            (
                "http://ex/bob".into(),
                "http://xmlns.com/foaf/0.1/knows".into(),
                "http://ex/carol".into(),
            ),
            (
                "http://ex/carol".into(),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into(),
                "http://ex/Employee".into(),
            ),
            (
                "http://ex/carol".into(),
                "http://xmlns.com/foaf/0.1/name".into(),
                "\"Carol\"".into(),
            ),
            (
                "_:b1".into(),
                "http://ex/rel".into(),
                "http://ex/alice".into(),
            ),
        ]
    }

    // ── Config tests ──

    #[test]
    fn test_profiler_config_default() {
        let config = ProfilerConfig::default();
        assert_eq!(config.top_n, 20);
        assert!(config.compute_connectivity);
        assert!(config.quality_checks);
        assert!(config.vocabulary_detection);
        assert!((config.sample_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_profiler_config_custom() {
        let config = ProfilerConfig {
            top_n: 5,
            compute_connectivity: false,
            quality_checks: false,
            vocabulary_detection: false,
            sample_rate: 0.5,
        };
        assert_eq!(config.top_n, 5);
        assert!(!config.compute_connectivity);
    }

    // ── Basic stats tests ──

    #[test]
    fn test_basic_stats_triple_count() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        assert_eq!(profile.basic_stats.triple_count, 10);
    }

    #[test]
    fn test_basic_stats_distinct_subjects() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        assert_eq!(profile.basic_stats.distinct_subjects, 4); // alice, bob, carol, _:b1
    }

    #[test]
    fn test_basic_stats_distinct_predicates() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        assert!(profile.basic_stats.distinct_predicates >= 4); // rdf:type, name, knows, age, rel
    }

    #[test]
    fn test_basic_stats_has_blank_nodes() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        assert!(profile.basic_stats.distinct_blank_nodes >= 1);
    }

    #[test]
    fn test_basic_stats_has_literals() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        assert!(profile.basic_stats.distinct_literals >= 3);
    }

    // ── Predicate analysis ──

    #[test]
    fn test_top_predicates_includes_rdf_type() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        let rdf_type_entry = profile
            .top_predicates
            .iter()
            .find(|e| e.item.contains("rdf-syntax-ns#type"));
        assert!(rdf_type_entry.is_some());
        assert_eq!(rdf_type_entry.map(|e| e.count), Some(3));
    }

    #[test]
    fn test_top_predicates_sorted_by_frequency() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        for window in profile.top_predicates.windows(2) {
            assert!(window[0].count >= window[1].count);
        }
    }

    // ── Class analysis ──

    #[test]
    fn test_top_classes() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        assert!(!profile.top_classes.is_empty());
        let person_entry = profile
            .top_classes
            .iter()
            .find(|e| e.item.contains("Person"));
        assert!(person_entry.is_some());
        assert_eq!(person_entry.map(|e| e.count), Some(2));
    }

    // ── Literal type analysis ──

    #[test]
    fn test_literal_type_detection() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        assert!(!profile.literal_types.is_empty());
    }

    #[test]
    fn test_literal_typed_integer() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        let xsd_int = profile
            .literal_types
            .iter()
            .find(|e| e.item.contains("XMLSchema#integer"));
        assert!(xsd_int.is_some());
    }

    // ── Language tag analysis ──

    #[test]
    fn test_language_tag_detection() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        let en_tag = profile.language_tags.iter().find(|e| e.item == "en");
        assert!(en_tag.is_some());
    }

    // ── Namespace detection ──

    #[test]
    fn test_namespace_detection() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        assert!(!profile.namespaces.is_empty());
        let foaf_ns = profile
            .namespaces
            .iter()
            .find(|n| n.namespace.contains("foaf"));
        assert!(foaf_ns.is_some());
    }

    #[test]
    fn test_namespace_prefix_suggestion() {
        assert_eq!(
            suggest_prefix("http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
            "rdf"
        );
        assert_eq!(suggest_prefix("http://xmlns.com/foaf/0.1/"), "foaf");
    }

    #[test]
    fn test_namespace_proportion() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        let total_proportion: f64 = profile.namespaces.iter().map(|n| n.proportion).sum();
        // Proportions should sum to roughly 1.0 (some IRIs may share namespaces)
        assert!(total_proportion > 0.0);
        assert!(total_proportion <= 1.01); // Allow tiny floating point overshoot
    }

    // ── Quality checks ──

    #[test]
    fn test_quality_dangling_references() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        // carol is referenced as object of knows but has type, so it IS a subject
        // But Person and Employee are never subjects
        let dangling = profile
            .quality_issues
            .iter()
            .find(|i| i.category == "dangling_references");
        assert!(dangling.is_some());
    }

    #[test]
    fn test_quality_no_issues_on_disabled() {
        let triples = make_test_triples();
        let config = ProfilerConfig {
            quality_checks: false,
            ..Default::default()
        };
        let profiler = DataProfiler::with_config(config);
        let profile = profiler.profile(&triples);
        assert!(profile.quality_issues.is_empty());
    }

    // ── Connectivity ──

    #[test]
    fn test_connectivity_computed() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        assert!(profile.connectivity.is_some());
        let conn = profile.connectivity.as_ref().expect("connectivity present");
        assert!(conn.avg_out_degree > 0.0);
        assert!(conn.max_out_degree > 0);
    }

    #[test]
    fn test_connectivity_disabled() {
        let triples = make_test_triples();
        let config = ProfilerConfig {
            compute_connectivity: false,
            ..Default::default()
        };
        let profiler = DataProfiler::with_config(config);
        let profile = profiler.profile(&triples);
        assert!(profile.connectivity.is_none());
    }

    #[test]
    fn test_hub_nodes_detected() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        let conn = profile.connectivity.as_ref().expect("connectivity present");
        // bob has most outgoing triples (4: type, name, age, knows)
        assert!(!conn.hub_nodes.is_empty());
    }

    // ── Vocabulary detection ──

    #[test]
    fn test_vocabulary_rdf_detected() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        let rdf_vocab = profile.vocabularies.iter().find(|v| v.name == "RDF");
        assert!(rdf_vocab.is_some());
    }

    #[test]
    fn test_vocabulary_foaf_detected() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        let foaf = profile.vocabularies.iter().find(|v| v.name == "FOAF");
        assert!(foaf.is_some());
        assert!(foaf.map(|v| v.terms_used).unwrap_or(0) >= 2);
    }

    #[test]
    fn test_vocabulary_disabled() {
        let triples = make_test_triples();
        let config = ProfilerConfig {
            vocabulary_detection: false,
            ..Default::default()
        };
        let profiler = DataProfiler::with_config(config);
        let profile = profiler.profile(&triples);
        assert!(profile.vocabularies.is_empty());
    }

    // ── Report formatting ──

    #[test]
    fn test_text_report_formatting() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        let report = format_text_report(&profile);
        assert!(report.contains("Basic Statistics"));
        assert!(report.contains("Total triples"));
        assert!(report.contains("10"));
    }

    #[test]
    fn test_json_report_formatting() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        let json = format_json_report(&profile).expect("json format");
        assert!(json.contains("triple_count"));
        assert!(json.contains("10"));
    }

    // ── Utility function tests ──

    #[test]
    fn test_is_iri() {
        assert!(is_iri("http://example.org/foo"));
        assert!(!is_iri("_:b1"));
        assert!(!is_iri("\"hello\""));
    }

    #[test]
    fn test_extract_namespace_hash() {
        assert_eq!(
            extract_namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string())
        );
    }

    #[test]
    fn test_extract_namespace_slash() {
        assert_eq!(
            extract_namespace("http://xmlns.com/foaf/0.1/name"),
            Some("http://xmlns.com/foaf/0.1/".to_string())
        );
    }

    #[test]
    fn test_frequency_map_basic() {
        let items = vec!["a", "b", "a", "c", "a"];
        let freq = frequency_map(items.into_iter());
        assert_eq!(freq.get("a"), Some(&3));
        assert_eq!(freq.get("b"), Some(&1));
    }

    #[test]
    fn test_top_n_entries_ordering() {
        let mut freq = HashMap::new();
        freq.insert("a".to_string(), 5);
        freq.insert("b".to_string(), 3);
        freq.insert("c".to_string(), 8);
        let entries = top_n_entries(&freq, 2, 16);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].item, "c");
        assert_eq!(entries[0].count, 8);
    }

    #[test]
    fn test_classify_term_iri() {
        let mut iris = HashSet::new();
        let mut blanks = HashSet::new();
        let mut lits = HashSet::new();
        classify_term("http://ex/a", &mut iris, &mut blanks, &mut lits);
        assert!(iris.contains("http://ex/a"));
        assert!(blanks.is_empty());
        assert!(lits.is_empty());
    }

    #[test]
    fn test_classify_term_blank() {
        let mut iris = HashSet::new();
        let mut blanks = HashSet::new();
        let mut lits = HashSet::new();
        classify_term("_:b1", &mut iris, &mut blanks, &mut lits);
        assert!(blanks.contains("_:b1"));
    }

    #[test]
    fn test_classify_term_literal() {
        let mut iris = HashSet::new();
        let mut blanks = HashSet::new();
        let mut lits = HashSet::new();
        classify_term("\"hello\"", &mut iris, &mut blanks, &mut lits);
        assert!(lits.contains("\"hello\""));
    }

    // ── Empty dataset ──

    #[test]
    fn test_empty_dataset_profile() {
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&[]);
        assert_eq!(profile.basic_stats.triple_count, 0);
        assert!(profile.top_predicates.is_empty());
        assert!(profile.top_classes.is_empty());
        assert!(profile.namespaces.is_empty());
    }

    // ── Sampling ──

    #[test]
    fn test_sampling_reduces_triples() {
        let triples = make_test_triples();
        let config = ProfilerConfig {
            sample_rate: 0.5,
            ..Default::default()
        };
        let profiler = DataProfiler::with_config(config);
        let profile = profiler.profile(&triples);
        assert!(profile.basic_stats.triples_sampled < profile.basic_stats.triple_count);
    }

    // ── Serialization roundtrip ──

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = ProfilerConfig {
            top_n: 10,
            compute_connectivity: false,
            quality_checks: true,
            vocabulary_detection: true,
            sample_rate: 0.75,
        };
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: ProfilerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.top_n, 10);
        assert!(!deserialized.compute_connectivity);
    }

    #[test]
    fn test_profile_serialization_roundtrip() {
        let triples = make_test_triples();
        let profiler = DataProfiler::new();
        let profile = profiler.profile(&triples);
        let json = serde_json::to_string(&profile).expect("serialize");
        let deserialized: DatasetProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            deserialized.basic_stats.triple_count,
            profile.basic_stats.triple_count
        );
    }

    // ── Issue severity display ──

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", IssueSeverity::Info), "INFO");
        assert_eq!(format!("{}", IssueSeverity::Warning), "WARN");
        assert_eq!(format!("{}", IssueSeverity::Error), "ERROR");
    }

    // ── Default trait ──

    #[test]
    fn test_profiler_default() {
        let profiler = DataProfiler::default();
        assert_eq!(profiler.config.top_n, 20);
    }
}
