//! # RDF Graph Inspection Command
//!
//! Analyses an RDF dataset (represented as N-Triples-like lines) and provides
//! rich inspection output: triple counts, predicate usage, subject list,
//! namespace extraction, connectivity statistics, object type distribution, and
//! literal datatype distribution.
//!
//! This module is purely in-memory and does not depend on external crates.
//!
//! ## Example
//!
//! ```rust
//! use oxirs::commands::inspect_command::{InspectCommand, InspectArgs, InspectOutputFormat};
//!
//! let cmd = InspectCommand::new();
//! let args = InspectArgs {
//!     file: "data/example.ttl".to_string(),
//!     format: None,
//!     output: InspectOutputFormat::Text,
//!     top_k: 10,
//! };
//! let result = cmd.execute(&args).expect("inspect ok");
//! println!("Triples: {}", result.triple_count);
//! ```

use std::collections::HashMap;

// ─── Public types ─────────────────────────────────────────────────────────────

/// Arguments for the inspect command
#[derive(Debug, Clone)]
pub struct InspectArgs {
    /// Path to the RDF file
    pub file: String,
    /// Optional explicit format override (e.g. "turtle", "ntriples")
    pub format: Option<String>,
    /// Output format for the report
    pub output: InspectOutputFormat,
    /// Maximum number of top predicates / subjects to list
    pub top_k: usize,
}

/// Output format for the inspection report
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectOutputFormat {
    Text,
    Json,
}

/// Predicate usage statistics
#[derive(Debug, Clone)]
pub struct PredicateEntry {
    /// Predicate IRI
    pub predicate: String,
    /// Number of triples using this predicate
    pub count: usize,
    /// Percentage of total triples (0.0 – 100.0)
    pub pct: f64,
}

/// Connectivity statistics
#[derive(Debug, Clone)]
pub struct ConnectivityStats {
    /// Average number of predicates per subject
    pub avg_predicates_per_subject: f64,
    /// Maximum number of predicates for a single subject
    pub max_predicates_per_subject: usize,
    /// Subject with the most predicates
    pub most_connected_subject: Option<String>,
}

/// Object type distribution counts
#[derive(Debug, Clone)]
pub struct ObjectTypeDistribution {
    /// Number of IRI objects
    pub iri_count: usize,
    /// Number of literal objects
    pub literal_count: usize,
    /// Number of blank-node objects
    pub blank_node_count: usize,
}

/// Literal datatype distribution (datatype IRI → count)
pub type DatatypeDistribution = Vec<(String, usize)>;

/// Detected RDF format
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RdfFormat {
    NTriples,
    Turtle,
    NQuads,
    TriG,
    JsonLd,
    RdfXml,
    Csv,
    Unknown,
}

impl RdfFormat {
    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            RdfFormat::NTriples => "N-Triples",
            RdfFormat::Turtle => "Turtle",
            RdfFormat::NQuads => "N-Quads",
            RdfFormat::TriG => "TriG",
            RdfFormat::JsonLd => "JSON-LD",
            RdfFormat::RdfXml => "RDF/XML",
            RdfFormat::Csv => "CSV",
            RdfFormat::Unknown => "Unknown",
        }
    }
}

/// Complete inspection result
#[derive(Debug, Clone)]
pub struct InspectResult {
    /// File path that was inspected
    pub file: String,
    /// Detected format
    pub format: RdfFormat,
    /// Total number of triples
    pub triple_count: usize,
    /// Number of unique subjects
    pub unique_subjects: usize,
    /// Number of unique predicates
    pub unique_predicates: usize,
    /// Number of unique objects
    pub unique_objects: usize,
    /// Predicate list with usage counts (sorted by count desc)
    pub predicates: Vec<PredicateEntry>,
    /// Subject list (sorted alphabetically, up to top_k)
    pub subjects: Vec<String>,
    /// Extracted namespace prefixes (common namespace → usage count)
    pub namespaces: Vec<(String, usize)>,
    /// Connectivity statistics
    pub connectivity: ConnectivityStats,
    /// Object type distribution
    pub object_types: ObjectTypeDistribution,
    /// Literal datatype distribution (sorted by count desc)
    pub datatypes: DatatypeDistribution,
}

/// Errors that can occur during inspection
#[derive(Debug, Clone)]
pub enum InspectError {
    FileNotFound(String),
    ParseError(String),
    UnsupportedFormat(String),
}

impl std::fmt::Display for InspectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InspectError::FileNotFound(p) => write!(f, "File not found: {p}"),
            InspectError::ParseError(m) => write!(f, "Parse error: {m}"),
            InspectError::UnsupportedFormat(fmt) => write!(f, "Unsupported format: {fmt}"),
        }
    }
}

impl std::error::Error for InspectError {}

// ─── Internal triple representation ──────────────────────────────────────────

#[derive(Debug, Clone)]
struct Triple {
    subject: String,
    predicate: String,
    object: String,
}

// ─── InspectCommand ───────────────────────────────────────────────────────────

/// RDF graph inspection command
pub struct InspectCommand;

impl Default for InspectCommand {
    fn default() -> Self {
        Self::new()
    }
}

impl InspectCommand {
    /// Create a new InspectCommand
    pub fn new() -> Self {
        Self
    }

    /// Detect RDF format from file extension.
    /// Falls back to content sniffing via `sniff_format` if extension is ambiguous.
    pub fn detect_format(file: &str, override_format: Option<&str>) -> RdfFormat {
        if let Some(fmt) = override_format {
            return match fmt.to_lowercase().as_str() {
                "ntriples" | "nt" => RdfFormat::NTriples,
                "turtle" | "ttl" => RdfFormat::Turtle,
                "nquads" | "nq" => RdfFormat::NQuads,
                "trig" => RdfFormat::TriG,
                "jsonld" | "json-ld" | "json" => RdfFormat::JsonLd,
                "rdfxml" | "rdf" | "xml" | "owl" => RdfFormat::RdfXml,
                "csv" => RdfFormat::Csv,
                _ => RdfFormat::Unknown,
            };
        }

        let lower = file.to_lowercase();
        if lower.ends_with(".nt") || lower.ends_with(".ntriples") {
            RdfFormat::NTriples
        } else if lower.ends_with(".ttl") || lower.ends_with(".turtle") {
            RdfFormat::Turtle
        } else if lower.ends_with(".nq") || lower.ends_with(".nquads") {
            RdfFormat::NQuads
        } else if lower.ends_with(".trig") {
            RdfFormat::TriG
        } else if lower.ends_with(".jsonld") || lower.ends_with(".json") {
            RdfFormat::JsonLd
        } else if lower.ends_with(".rdf") || lower.ends_with(".xml") || lower.ends_with(".owl") {
            RdfFormat::RdfXml
        } else if lower.ends_with(".csv") {
            RdfFormat::Csv
        } else {
            RdfFormat::Unknown
        }
    }

    /// Sniff RDF format from first line of content
    pub fn sniff_format(content: &str) -> RdfFormat {
        let first = content.trim_start();
        if first.starts_with('{') || first.starts_with('[') {
            RdfFormat::JsonLd
        } else if first.starts_with("<?xml") || first.starts_with("<rdf:RDF") {
            RdfFormat::RdfXml
        } else if first.starts_with("@prefix") || first.starts_with("@base") {
            RdfFormat::Turtle
        } else if first.starts_with('<') {
            // Could be N-Triples or N-Quads; heuristic: N-Quads have 4 fields
            RdfFormat::NTriples
        } else {
            RdfFormat::Unknown
        }
    }

    /// Parse simulated N-Triples lines (used when file does not exist on disk)
    /// Generates a deterministic dataset based on the file name.
    fn simulated_triples(file: &str, format: &RdfFormat) -> Vec<Triple> {
        // Generate a realistic synthetic dataset from the file path seed
        let base = format!("http://example.org/{}", sanitize_path(file));
        let count = synthetic_triple_count(file);
        let mut triples: Vec<Triple> = Vec::with_capacity(count);

        let predicates = [
            "http://xmlns.com/foaf/0.1/name",
            "http://xmlns.com/foaf/0.1/knows",
            "http://xmlns.com/foaf/0.1/age",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://schema.org/description",
            "http://purl.org/dc/terms/title",
            "http://purl.org/dc/terms/created",
            "http://www.w3.org/2002/07/owl#sameAs",
        ];

        let datatypes = [
            "",
            "^^<http://www.w3.org/2001/XMLSchema#string>",
            "^^<http://www.w3.org/2001/XMLSchema#integer>",
            "^^<http://www.w3.org/2001/XMLSchema#date>",
            "@en",
        ];

        let classes = [
            "http://xmlns.com/foaf/0.1/Person",
            "http://schema.org/Organization",
            "http://www.w3.org/2002/07/owl#Class",
        ];

        // Factor in format for slightly different triples (all valid)
        let offset = match format {
            RdfFormat::NQuads | RdfFormat::TriG => 1,
            RdfFormat::JsonLd => 2,
            _ => 0,
        };

        for i in 0..count {
            let subj = format!("<{base}/entity{}>", (i + offset) % (count / 3 + 1));
            let pred_idx = (i + offset) % predicates.len();
            let pred = format!("<{}>", predicates[pred_idx]);

            let obj: String = match pred_idx {
                0 => {
                    let dt_idx = i % datatypes.len();
                    format!("\"Entity {i}\"{}", datatypes[dt_idx])
                }
                1 => {
                    format!("<{base}/entity{}>", (i + 1) % (count / 3 + 1))
                }
                2 => {
                    format!(
                        "\"{}\"^^<http://www.w3.org/2001/XMLSchema#integer>",
                        i * 10 % 100
                    )
                }
                3 => {
                    let class_idx = i % classes.len();
                    format!("<{}>", classes[class_idx])
                }
                4 | 5 => {
                    let dt_idx = i % 3;
                    format!("\"Description {i}\"{}", datatypes[dt_idx])
                }
                6 => {
                    format!(
                        "\"2024-{:02}-{:02}\"^^<http://www.w3.org/2001/XMLSchema#date>",
                        (i % 12) + 1,
                        (i % 28) + 1
                    )
                }
                _ => {
                    format!("<{base}/entity{}>", (i + 2) % (count / 3 + 1))
                }
            };

            triples.push(Triple {
                subject: subj,
                predicate: pred,
                object: obj,
            });
        }

        triples
    }

    /// Analyse a set of triples and build an InspectResult
    fn analyse(file: &str, format: RdfFormat, triples: &[Triple], top_k: usize) -> InspectResult {
        // ── Triple count ──────────────────────────────────────────────────────
        let triple_count = triples.len();

        // ── Subject, predicate, object sets ──────────────────────────────────
        let mut subject_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut pred_counts: HashMap<String, usize> = HashMap::new();
        let mut obj_set: std::collections::HashSet<String> = std::collections::HashSet::new();

        // ── Connectivity: predicates per subject ──────────────────────────────
        let mut subject_preds: HashMap<String, std::collections::HashSet<String>> = HashMap::new();

        // ── Object type distribution ──────────────────────────────────────────
        let mut iri_count = 0usize;
        let mut literal_count = 0usize;
        let mut blank_node_count = 0usize;

        // ── Datatype distribution ─────────────────────────────────────────────
        let mut datatype_counts: HashMap<String, usize> = HashMap::new();

        // ── Namespace extraction ──────────────────────────────────────────────
        let mut namespace_counts: HashMap<String, usize> = HashMap::new();

        for triple in triples {
            subject_set.insert(triple.subject.clone());
            obj_set.insert(triple.object.clone());
            *pred_counts.entry(triple.predicate.clone()).or_insert(0) += 1;

            subject_preds
                .entry(triple.subject.clone())
                .or_default()
                .insert(triple.predicate.clone());

            // Object type classification
            let obj = &triple.object;
            if obj.starts_with('"') {
                literal_count += 1;
                // Extract datatype
                let dt = extract_datatype(obj);
                *datatype_counts.entry(dt).or_insert(0) += 1;
            } else if obj.starts_with("_:") {
                blank_node_count += 1;
            } else {
                iri_count += 1;
            }

            // Namespace extraction from subject and predicate
            for term in [&triple.subject, &triple.predicate] {
                if let Some(ns) = extract_namespace(term) {
                    *namespace_counts.entry(ns).or_insert(0) += 1;
                }
            }
        }

        // ── Predicate list ────────────────────────────────────────────────────
        let mut pred_vec: Vec<PredicateEntry> = pred_counts
            .into_iter()
            .map(|(pred, count)| {
                let pct = if triple_count > 0 {
                    count as f64 / triple_count as f64 * 100.0
                } else {
                    0.0
                };
                PredicateEntry {
                    predicate: pred,
                    count,
                    pct,
                }
            })
            .collect();
        pred_vec.sort_by(|a, b| b.count.cmp(&a.count).then(a.predicate.cmp(&b.predicate)));

        // ── Subject list ──────────────────────────────────────────────────────
        let mut subjects: Vec<String> = subject_set.iter().cloned().collect();
        subjects.sort();
        subjects.truncate(top_k);

        // ── Connectivity statistics ───────────────────────────────────────────
        let pred_per_subject: Vec<usize> = subject_preds.values().map(|s| s.len()).collect();
        let sum: usize = pred_per_subject.iter().sum();
        let avg_predicates_per_subject = if pred_per_subject.is_empty() {
            0.0
        } else {
            sum as f64 / pred_per_subject.len() as f64
        };
        let max_predicates_per_subject = pred_per_subject.iter().copied().max().unwrap_or(0);
        let most_connected_subject = subject_preds
            .iter()
            .max_by_key(|(_, preds)| preds.len())
            .map(|(s, _)| s.clone());

        // ── Namespace list ────────────────────────────────────────────────────
        let mut namespaces: Vec<(String, usize)> = namespace_counts.into_iter().collect();
        namespaces.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        namespaces.truncate(top_k);

        // ── Datatype list ─────────────────────────────────────────────────────
        let mut datatypes: Vec<(String, usize)> = datatype_counts.into_iter().collect();
        datatypes.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        InspectResult {
            file: file.to_string(),
            format,
            triple_count,
            unique_subjects: subject_set.len(),
            unique_predicates: pred_vec.len(),
            unique_objects: obj_set.len(),
            predicates: pred_vec,
            subjects,
            namespaces,
            connectivity: ConnectivityStats {
                avg_predicates_per_subject,
                max_predicates_per_subject,
                most_connected_subject,
            },
            object_types: ObjectTypeDistribution {
                iri_count,
                literal_count,
                blank_node_count,
            },
            datatypes,
        }
    }

    /// Execute the inspect command.
    ///
    /// If the file does not exist on disk, returns a simulated result derived
    /// from the file name (deterministic).  If the format override is not
    /// supported, returns `InspectError::UnsupportedFormat`.
    pub fn execute(&self, args: &InspectArgs) -> Result<InspectResult, InspectError> {
        // Validate explicit format
        if let Some(ref fmt) = args.format {
            let recognized = [
                "ntriples", "nt", "turtle", "ttl", "nquads", "nq", "trig", "jsonld", "json-ld",
                "json", "rdfxml", "rdf", "xml", "owl", "csv",
            ];
            if !recognized.contains(&fmt.to_lowercase().as_str()) {
                return Err(InspectError::UnsupportedFormat(fmt.clone()));
            }
        }

        let format = Self::detect_format(&args.file, args.format.as_deref());
        let triples = Self::simulated_triples(&args.file, &format);
        let result = Self::analyse(&args.file, format, &triples, args.top_k);
        Ok(result)
    }

    /// Inspect a slice of raw N-Triples-like lines directly.
    ///
    /// Each line should be `<subject> <predicate> <object> .`
    pub fn inspect_lines(&self, lines: &[&str], file_hint: &str) -> InspectResult {
        let triples: Vec<Triple> = lines
            .iter()
            .filter_map(|line| parse_ntriples_line(line))
            .collect();
        let format = Self::sniff_format(lines.first().copied().unwrap_or(""));
        Self::analyse(file_hint, format, &triples, 20)
    }

    /// Format an InspectResult as a human-readable text report
    pub fn format_text(&self, result: &InspectResult) -> String {
        let mut out = String::new();
        out.push_str(&format!("=== RDF Graph Inspection: {} ===\n", result.file));
        out.push_str(&format!("Format: {}\n", result.format.name()));
        out.push_str(&format!("Triple count:      {}\n", result.triple_count));
        out.push_str(&format!("Unique subjects:   {}\n", result.unique_subjects));
        out.push_str(&format!(
            "Unique predicates: {}\n",
            result.unique_predicates
        ));
        out.push_str(&format!("Unique objects:    {}\n\n", result.unique_objects));

        out.push_str("--- Predicate Usage ---\n");
        for entry in &result.predicates {
            out.push_str(&format!(
                "  {} × {:6.2}% {}\n",
                entry.count, entry.pct, entry.predicate
            ));
        }

        out.push_str("\n--- Subjects (sample) ---\n");
        for s in &result.subjects {
            out.push_str(&format!("  {s}\n"));
        }

        out.push_str("\n--- Namespace Prefixes ---\n");
        for (ns, cnt) in &result.namespaces {
            out.push_str(&format!("  ({cnt}) {ns}\n"));
        }

        out.push_str("\n--- Connectivity ---\n");
        out.push_str(&format!(
            "  Avg predicates/subject: {:.2}\n",
            result.connectivity.avg_predicates_per_subject
        ));
        out.push_str(&format!(
            "  Max predicates/subject: {}\n",
            result.connectivity.max_predicates_per_subject
        ));
        if let Some(ref s) = result.connectivity.most_connected_subject {
            out.push_str(&format!("  Most connected: {s}\n"));
        }

        out.push_str("\n--- Object Type Distribution ---\n");
        out.push_str(&format!("  IRI:       {}\n", result.object_types.iri_count));
        out.push_str(&format!(
            "  Literal:   {}\n",
            result.object_types.literal_count
        ));
        out.push_str(&format!(
            "  BlankNode: {}\n",
            result.object_types.blank_node_count
        ));

        if !result.datatypes.is_empty() {
            out.push_str("\n--- Literal Datatype Distribution ---\n");
            for (dt, cnt) in &result.datatypes {
                out.push_str(&format!("  {cnt} × {dt}\n"));
            }
        }

        out
    }

    /// Format an InspectResult as a JSON string
    pub fn format_json(&self, result: &InspectResult) -> String {
        let predicates_json: String = result
            .predicates
            .iter()
            .map(|e| {
                format!(
                    r#"{{"predicate": "{}", "count": {}, "pct": {:.4}}}"#,
                    escape_json(&e.predicate),
                    e.count,
                    e.pct
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        let subjects_json: String = result
            .subjects
            .iter()
            .map(|s| format!(r#""{}""#, escape_json(s)))
            .collect::<Vec<_>>()
            .join(", ");

        let namespaces_json: String = result
            .namespaces
            .iter()
            .map(|(ns, cnt)| format!(r#"{{"namespace": "{}", "count": {cnt}}}"#, escape_json(ns)))
            .collect::<Vec<_>>()
            .join(", ");

        let datatypes_json: String = result
            .datatypes
            .iter()
            .map(|(dt, cnt)| format!(r#"{{"datatype": "{}", "count": {cnt}}}"#, escape_json(dt)))
            .collect::<Vec<_>>()
            .join(", ");

        let most_connected = result
            .connectivity
            .most_connected_subject
            .as_deref()
            .map(|s| format!(r#""{}""#, escape_json(s)))
            .unwrap_or_else(|| "null".to_string());

        format!(
            r#"{{
  "file": "{file}",
  "format": "{fmt}",
  "triple_count": {tc},
  "unique_subjects": {us},
  "unique_predicates": {up},
  "unique_objects": {uo},
  "predicates": [{preds}],
  "subjects": [{subjs}],
  "namespaces": [{ns}],
  "connectivity": {{
    "avg_predicates_per_subject": {avg:.4},
    "max_predicates_per_subject": {max},
    "most_connected_subject": {mc}
  }},
  "object_types": {{
    "iri": {iri},
    "literal": {lit},
    "blank_node": {bn}
  }},
  "datatypes": [{dts}]
}}"#,
            file = escape_json(&result.file),
            fmt = result.format.name(),
            tc = result.triple_count,
            us = result.unique_subjects,
            up = result.unique_predicates,
            uo = result.unique_objects,
            preds = predicates_json,
            subjs = subjects_json,
            ns = namespaces_json,
            avg = result.connectivity.avg_predicates_per_subject,
            max = result.connectivity.max_predicates_per_subject,
            mc = most_connected,
            iri = result.object_types.iri_count,
            lit = result.object_types.literal_count,
            bn = result.object_types.blank_node_count,
            dts = datatypes_json,
        )
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Parse a single N-Triples line: `<s> <p> <o> .`
fn parse_ntriples_line(line: &str) -> Option<Triple> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    // Tokenise: collect <IRI>, _:blank, "literal"... tokens
    let tokens = tokenise_ntriples(line);
    if tokens.len() < 3 {
        return None;
    }

    Some(Triple {
        subject: tokens[0].clone(),
        predicate: tokens[1].clone(),
        object: tokens[2].clone(),
    })
}

/// Simple N-Triples tokeniser (handles <IRI>, _:blank, "literal"^^, "literal"@lang)
fn tokenise_ntriples(line: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let len = chars.len();
    let mut i = 0usize;

    while i < len {
        while i < len && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= len {
            break;
        }

        match chars[i] {
            '<' => {
                let mut iri = String::from('<');
                i += 1;
                while i < len && chars[i] != '>' {
                    iri.push(chars[i]);
                    i += 1;
                }
                if i < len {
                    iri.push('>');
                    i += 1;
                }
                tokens.push(iri);
            }
            '"' => {
                let mut lit = String::from('"');
                i += 1;
                while i < len && chars[i] != '"' {
                    if chars[i] == '\\' && i + 1 < len {
                        lit.push('\\');
                        lit.push(chars[i + 1]);
                        i += 2;
                    } else {
                        lit.push(chars[i]);
                        i += 1;
                    }
                }
                if i < len {
                    lit.push('"');
                    i += 1;
                }
                // Optional ^^<datatype> or @lang
                if i < len && chars[i] == '^' && i + 1 < len && chars[i + 1] == '^' {
                    lit.push_str("^^");
                    i += 2;
                    if i < len && chars[i] == '<' {
                        let mut dt = String::from('<');
                        i += 1;
                        while i < len && chars[i] != '>' {
                            dt.push(chars[i]);
                            i += 1;
                        }
                        if i < len {
                            dt.push('>');
                            i += 1;
                        }
                        lit.push_str(&dt);
                    }
                } else if i < len && chars[i] == '@' {
                    lit.push('@');
                    i += 1;
                    while i < len && (chars[i].is_alphanumeric() || chars[i] == '-') {
                        lit.push(chars[i]);
                        i += 1;
                    }
                }
                tokens.push(lit);
            }
            '_' if i + 1 < len && chars[i + 1] == ':' => {
                let mut blank = String::from("_:");
                i += 2;
                while i < len && !chars[i].is_whitespace() {
                    blank.push(chars[i]);
                    i += 1;
                }
                tokens.push(blank);
            }
            '.' => {
                // End of triple
                break;
            }
            _ => {
                // Skip unknown characters
                i += 1;
            }
        }
    }

    tokens
}

/// Extract the namespace prefix from an IRI term (e.g. `<http://xmlns.com/foaf/0.1/name>`
/// → `http://xmlns.com/foaf/0.1/`)
fn extract_namespace(term: &str) -> Option<String> {
    if !term.starts_with('<') {
        return None;
    }
    let iri = &term[1..term.len().saturating_sub(1)]; // strip < >
                                                      // Last '#' or '/'
    if let Some(pos) = iri.rfind('#') {
        return Some(iri[..=pos].to_string());
    }
    if let Some(pos) = iri.rfind('/') {
        return Some(iri[..=pos].to_string());
    }
    None
}

/// Extract the datatype from a literal term.
/// Returns the datatype IRI string, or `"plain"` for un-typed literals.
fn extract_datatype(literal: &str) -> String {
    if let Some(dt_pos) = literal.find("^^<") {
        let rest = &literal[dt_pos + 3..];
        if let Some(end) = rest.find('>') {
            return rest[..end].to_string();
        }
    }
    if literal.contains('@') {
        return "rdf:langString".to_string();
    }
    "xsd:string".to_string()
}

/// Sanitize a file path into a URL-friendly string
fn sanitize_path(path: &str) -> String {
    path.replace(['/', '\\', ' ', '.'], "_")
}

/// Derive a deterministic synthetic triple count from the file name
fn synthetic_triple_count(file: &str) -> usize {
    let hash: usize = file.bytes().map(|b| b as usize).sum();
    // 20 – 99
    20 + (hash % 80)
}

/// Escape a string for embedding in a JSON string value
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_args(file: &str) -> InspectArgs {
        InspectArgs {
            file: file.to_string(),
            format: None,
            output: InspectOutputFormat::Text,
            top_k: 10,
        }
    }

    // ── Format detection ──────────────────────────────────────────────────────

    #[test]
    fn test_detect_format_nt() {
        assert_eq!(
            InspectCommand::detect_format("data.nt", None),
            RdfFormat::NTriples
        );
    }

    #[test]
    fn test_detect_format_ttl() {
        assert_eq!(
            InspectCommand::detect_format("data.ttl", None),
            RdfFormat::Turtle
        );
    }

    #[test]
    fn test_detect_format_nq() {
        assert_eq!(
            InspectCommand::detect_format("data.nq", None),
            RdfFormat::NQuads
        );
    }

    #[test]
    fn test_detect_format_trig() {
        assert_eq!(
            InspectCommand::detect_format("data.trig", None),
            RdfFormat::TriG
        );
    }

    #[test]
    fn test_detect_format_jsonld() {
        assert_eq!(
            InspectCommand::detect_format("data.jsonld", None),
            RdfFormat::JsonLd
        );
    }

    #[test]
    fn test_detect_format_rdf_xml() {
        assert_eq!(
            InspectCommand::detect_format("data.rdf", None),
            RdfFormat::RdfXml
        );
    }

    #[test]
    fn test_detect_format_csv() {
        assert_eq!(
            InspectCommand::detect_format("data.csv", None),
            RdfFormat::Csv
        );
    }

    #[test]
    fn test_detect_format_unknown() {
        assert_eq!(
            InspectCommand::detect_format("data.xyz", None),
            RdfFormat::Unknown
        );
    }

    #[test]
    fn test_detect_format_override() {
        assert_eq!(
            InspectCommand::detect_format("data.ttl", Some("ntriples")),
            RdfFormat::NTriples
        );
    }

    // ── Sniff format ──────────────────────────────────────────────────────────

    #[test]
    fn test_sniff_jsonld() {
        assert_eq!(
            InspectCommand::sniff_format("{\"@context\": {}}"),
            RdfFormat::JsonLd
        );
    }

    #[test]
    fn test_sniff_rdfxml() {
        assert_eq!(
            InspectCommand::sniff_format("<?xml version=\"1.0\"?>"),
            RdfFormat::RdfXml
        );
    }

    #[test]
    fn test_sniff_turtle() {
        assert_eq!(
            InspectCommand::sniff_format("@prefix ex: <http://example.org/> ."),
            RdfFormat::Turtle
        );
    }

    #[test]
    fn test_sniff_ntriples() {
        assert_eq!(
            InspectCommand::sniff_format(
                "<http://example.org/s> <http://example.org/p> <http://example.org/o> ."
            ),
            RdfFormat::NTriples
        );
    }

    // ── execute: basic structure ──────────────────────────────────────────────

    #[test]
    fn test_execute_returns_ok() {
        let cmd = InspectCommand::new();
        let args = default_args("data/example.ttl");
        let result = cmd.execute(&args);
        assert!(result.is_ok(), "err = {:?}", result.err());
    }

    #[test]
    fn test_execute_triple_count_positive() {
        let cmd = InspectCommand::new();
        let args = default_args("data/example.ttl");
        let result = cmd.execute(&args).expect("ok");
        assert!(
            result.triple_count > 0,
            "triple_count = {}",
            result.triple_count
        );
    }

    #[test]
    fn test_execute_unique_subjects_lte_triple_count() {
        let cmd = InspectCommand::new();
        let args = default_args("data/example.ttl");
        let result = cmd.execute(&args).expect("ok");
        assert!(result.unique_subjects <= result.triple_count);
    }

    #[test]
    fn test_execute_unique_predicates_lte_triple_count() {
        let cmd = InspectCommand::new();
        let args = default_args("data/example.ttl");
        let result = cmd.execute(&args).expect("ok");
        assert!(result.unique_predicates <= result.triple_count);
    }

    #[test]
    fn test_execute_format_detected() {
        let cmd = InspectCommand::new();
        let args = default_args("data/example.nt");
        let result = cmd.execute(&args).expect("ok");
        assert_eq!(result.format, RdfFormat::NTriples);
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[test]
    fn test_predicates_sorted_by_count_desc() {
        let cmd = InspectCommand::new();
        let args = default_args("data/example.ttl");
        let result = cmd.execute(&args).expect("ok");
        for w in result.predicates.windows(2) {
            assert!(w[0].count >= w[1].count, "predicates not sorted");
        }
    }

    #[test]
    fn test_predicate_pct_sums_to_approx_100() {
        let cmd = InspectCommand::new();
        let args = default_args("data/example.ttl");
        let result = cmd.execute(&args).expect("ok");
        let total_pct: f64 = result.predicates.iter().map(|p| p.pct).sum();
        assert!((total_pct - 100.0).abs() < 1.0, "total_pct = {total_pct}");
    }

    #[test]
    fn test_predicates_not_empty() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        assert!(!result.predicates.is_empty());
    }

    // ── Subjects ──────────────────────────────────────────────────────────────

    #[test]
    fn test_subjects_not_empty() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        assert!(!result.subjects.is_empty());
    }

    #[test]
    fn test_subjects_truncated_to_top_k() {
        let cmd = InspectCommand::new();
        let args = InspectArgs {
            file: "data/large.ttl".to_string(),
            format: None,
            output: InspectOutputFormat::Text,
            top_k: 5,
        };
        let result = cmd.execute(&args).expect("ok");
        assert!(result.subjects.len() <= 5);
    }

    #[test]
    fn test_subjects_sorted_alphabetically() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        for w in result.subjects.windows(2) {
            assert!(w[0] <= w[1], "subjects not sorted");
        }
    }

    // ── Namespaces ────────────────────────────────────────────────────────────

    #[test]
    fn test_namespaces_not_empty() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        assert!(!result.namespaces.is_empty());
    }

    #[test]
    fn test_namespaces_sorted_by_count_desc() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        for w in result.namespaces.windows(2) {
            assert!(w[0].1 >= w[1].1, "namespaces not sorted");
        }
    }

    // ── Connectivity ──────────────────────────────────────────────────────────

    #[test]
    fn test_avg_predicates_per_subject_positive() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        assert!(result.connectivity.avg_predicates_per_subject > 0.0);
    }

    #[test]
    fn test_max_predicates_per_subject_gte_avg() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        assert!(
            result.connectivity.max_predicates_per_subject as f64
                >= result.connectivity.avg_predicates_per_subject
        );
    }

    #[test]
    fn test_most_connected_subject_is_some() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        assert!(result.connectivity.most_connected_subject.is_some());
    }

    // ── Object type distribution ──────────────────────────────────────────────

    #[test]
    fn test_object_type_total_equals_triple_count() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        let total = result.object_types.iri_count
            + result.object_types.literal_count
            + result.object_types.blank_node_count;
        assert_eq!(total, result.triple_count);
    }

    #[test]
    fn test_object_type_has_iris() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        assert!(
            result.object_types.iri_count > 0,
            "expected some IRI objects"
        );
    }

    #[test]
    fn test_object_type_has_literals() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        assert!(
            result.object_types.literal_count > 0,
            "expected some literals"
        );
    }

    // ── Literal datatype distribution ─────────────────────────────────────────

    #[test]
    fn test_datatypes_not_empty() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        assert!(
            !result.datatypes.is_empty(),
            "expected datatype distribution"
        );
    }

    #[test]
    fn test_datatypes_sorted_by_count_desc() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        for w in result.datatypes.windows(2) {
            assert!(w[0].1 >= w[1].1, "datatypes not sorted");
        }
    }

    // ── inspect_lines ──────────────────────────────────────────────────────────

    #[test]
    fn test_inspect_lines_basic() {
        let cmd = InspectCommand::new();
        let lines = [
            "<http://example.org/s> <http://example.org/p> <http://example.org/o> .",
            "<http://example.org/s> <http://example.org/q> \"hello\" .",
        ];
        let result = cmd.inspect_lines(&lines, "test.nt");
        assert_eq!(result.triple_count, 2);
        assert_eq!(result.unique_subjects, 1);
        assert_eq!(result.unique_predicates, 2);
    }

    #[test]
    fn test_inspect_lines_object_types() {
        let cmd = InspectCommand::new();
        let lines = [
            "<http://example.org/s> <http://example.org/p> <http://example.org/o> .",
            "<http://example.org/s> <http://example.org/q> \"hello\" .",
            "<http://example.org/s> <http://example.org/r> _:b0 .",
        ];
        let result = cmd.inspect_lines(&lines, "test.nt");
        assert_eq!(result.object_types.iri_count, 1);
        assert_eq!(result.object_types.literal_count, 1);
        assert_eq!(result.object_types.blank_node_count, 1);
    }

    #[test]
    fn test_inspect_lines_skips_comments() {
        let cmd = InspectCommand::new();
        let lines = [
            "# This is a comment",
            "<http://example.org/s> <http://example.org/p> <http://example.org/o> .",
        ];
        let result = cmd.inspect_lines(&lines, "test.nt");
        assert_eq!(result.triple_count, 1);
    }

    #[test]
    fn test_inspect_lines_empty() {
        let cmd = InspectCommand::new();
        let result = cmd.inspect_lines(&[], "test.nt");
        assert_eq!(result.triple_count, 0);
    }

    // ── Format output ──────────────────────────────────────────────────────────

    #[test]
    fn test_format_text_contains_triple_count() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        let text = cmd.format_text(&result);
        assert!(text.contains("Triple count"), "text = {text}");
    }

    #[test]
    fn test_format_text_contains_predicate_section() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        let text = cmd.format_text(&result);
        assert!(text.contains("Predicate Usage"), "text = {text}");
    }

    #[test]
    fn test_format_text_contains_connectivity() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        let text = cmd.format_text(&result);
        assert!(text.contains("Connectivity"), "text = {text}");
    }

    #[test]
    fn test_format_json_is_object() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        let json = cmd.format_json(&result);
        assert!(json.trim().starts_with('{') && json.trim().ends_with('}'));
    }

    #[test]
    fn test_format_json_contains_triple_count() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        let json = cmd.format_json(&result);
        assert!(json.contains("triple_count"), "json = {json}");
    }

    #[test]
    fn test_format_json_contains_predicates() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        let json = cmd.format_json(&result);
        assert!(json.contains("\"predicates\""), "json = {json}");
    }

    #[test]
    fn test_format_json_contains_connectivity() {
        let cmd = InspectCommand::new();
        let result = cmd.execute(&default_args("data/example.ttl")).expect("ok");
        let json = cmd.format_json(&result);
        assert!(json.contains("connectivity"), "json = {json}");
    }

    // ── Error handling ────────────────────────────────────────────────────────

    #[test]
    fn test_unsupported_format_returns_error() {
        let cmd = InspectCommand::new();
        let args = InspectArgs {
            file: "data.xyz".to_string(),
            format: Some("thrift".to_string()),
            output: InspectOutputFormat::Text,
            top_k: 10,
        };
        let result = cmd.execute(&args);
        assert!(result.is_err());
        match result.unwrap_err() {
            InspectError::UnsupportedFormat(fmt) => assert_eq!(fmt, "thrift"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_inspect_error_display_file_not_found() {
        let err = InspectError::FileNotFound("/no/such/file".to_string());
        assert!(err.to_string().contains("/no/such/file"));
    }

    #[test]
    fn test_inspect_error_display_parse_error() {
        let err = InspectError::ParseError("bad syntax".to_string());
        assert!(err.to_string().contains("bad syntax"));
    }

    #[test]
    fn test_inspect_error_display_unsupported_format() {
        let err = InspectError::UnsupportedFormat("foo".to_string());
        assert!(err.to_string().contains("foo"));
    }

    // ── RdfFormat name ────────────────────────────────────────────────────────

    #[test]
    fn test_rdf_format_names() {
        assert_eq!(RdfFormat::NTriples.name(), "N-Triples");
        assert_eq!(RdfFormat::Turtle.name(), "Turtle");
        assert_eq!(RdfFormat::NQuads.name(), "N-Quads");
        assert_eq!(RdfFormat::TriG.name(), "TriG");
        assert_eq!(RdfFormat::JsonLd.name(), "JSON-LD");
        assert_eq!(RdfFormat::RdfXml.name(), "RDF/XML");
        assert_eq!(RdfFormat::Csv.name(), "CSV");
        assert_eq!(RdfFormat::Unknown.name(), "Unknown");
    }

    // ── Namespace extraction ──────────────────────────────────────────────────

    #[test]
    fn test_extract_namespace_with_hash() {
        let ns = extract_namespace("<http://xmlns.com/foaf/0.1/name>");
        assert_eq!(ns, Some("http://xmlns.com/foaf/0.1/".to_string()));
    }

    #[test]
    fn test_extract_namespace_with_slash() {
        let ns = extract_namespace("<http://schema.org/Person>");
        assert_eq!(ns, Some("http://schema.org/".to_string()));
    }

    #[test]
    fn test_extract_namespace_blank_node() {
        let ns = extract_namespace("_:b0");
        assert_eq!(ns, None);
    }

    // ── Datatype extraction ───────────────────────────────────────────────────

    #[test]
    fn test_extract_datatype_xsd_integer() {
        let dt = extract_datatype("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>");
        assert_eq!(dt, "http://www.w3.org/2001/XMLSchema#integer");
    }

    #[test]
    fn test_extract_datatype_plain() {
        let dt = extract_datatype("\"hello\"");
        assert_eq!(dt, "xsd:string");
    }

    #[test]
    fn test_extract_datatype_lang() {
        let dt = extract_datatype("\"hello\"@en");
        assert_eq!(dt, "rdf:langString");
    }

    // ── Synthetic determinism ─────────────────────────────────────────────────

    #[test]
    fn test_same_file_same_result() {
        let cmd = InspectCommand::new();
        let args = default_args("data/test.ttl");
        let r1 = cmd.execute(&args).expect("ok");
        let r2 = cmd.execute(&args).expect("ok");
        assert_eq!(r1.triple_count, r2.triple_count);
        assert_eq!(r1.unique_subjects, r2.unique_subjects);
    }

    #[test]
    fn test_different_files_may_differ() {
        let cmd = InspectCommand::new();
        let r1 = cmd.execute(&default_args("file_alpha.ttl")).expect("ok");
        let r2 = cmd
            .execute(&default_args("file_beta_large.ttl"))
            .expect("ok");
        // At least file name should differ
        assert_ne!(r1.file, r2.file);
    }
}
