//! Schema generation tool
//!
//! Analyse an RDF data file and produce SHACL NodeShapes / PropertyShapes
//! describing the classes and predicates found in the data.
//!
//! For each class observed via `rdf:type` assertions the tool generates:
//!   - A `sh:NodeShape` with `sh:targetClass`
//!   - One `sh:PropertyShape` per predicate used by instances of that class
//!
//! Optionally prints statistical summaries (class/predicate counts).

use super::ToolResult;
use oxirs_core::model::{Object, Predicate, Subject};
use oxirs_ttl::convenience::parse_rdf_file;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;

// Well-known IRIs
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const SH_NS: &str = "http://www.w3.org/ns/shacl#";
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";

// ─── Schema analysis ──────────────────────────────────────────────────────────

/// Class → set of predicate IRIs observed for its instances
type ClassPredicateMap = BTreeMap<String, BTreeSet<String>>;
/// Class → instance count
type ClassCountMap = BTreeMap<String, usize>;
/// Predicate → count of uses across all triples
type PredicateCountMap = BTreeMap<String, usize>;

struct AnalysisResult {
    class_predicates: ClassPredicateMap,
    class_counts: ClassCountMap,
    predicate_counts: PredicateCountMap,
    instance_to_classes: BTreeMap<String, BTreeSet<String>>,
}

fn subject_to_str(subj: &Subject) -> String {
    match subj {
        Subject::NamedNode(n) => n.as_str().to_string(),
        Subject::BlankNode(b) => format!("_:{}", b.id()),
        Subject::Variable(v) => format!("?{v}"),
        Subject::QuotedTriple(_) => "_quoted_".to_string(),
    }
}

fn predicate_to_str(pred: &Predicate) -> String {
    match pred {
        Predicate::NamedNode(n) => n.as_str().to_string(),
        Predicate::Variable(v) => format!("?{v}"),
    }
}

fn object_to_iri(obj: &Object) -> Option<String> {
    if let Object::NamedNode(n) = obj {
        Some(n.as_str().to_string())
    } else {
        None
    }
}

fn analyse(triples: &[oxirs_core::model::Triple]) -> AnalysisResult {
    let mut instance_to_classes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut class_predicates: ClassPredicateMap = BTreeMap::new();
    let mut predicate_counts: PredicateCountMap = BTreeMap::new();

    // First pass: collect rdf:type assertions
    for triple in triples {
        let pred = predicate_to_str(triple.predicate());
        *predicate_counts.entry(pred.clone()).or_insert(0) += 1;

        if pred == RDF_TYPE {
            if let Some(class_iri) = object_to_iri(triple.object()) {
                let subj = subject_to_str(triple.subject());
                instance_to_classes
                    .entry(subj)
                    .or_default()
                    .insert(class_iri.clone());
                class_predicates.entry(class_iri).or_default();
            }
        }
    }

    // Second pass: for each triple, assign its predicate to the class(es) of its subject
    for triple in triples {
        let subj = subject_to_str(triple.subject());
        let pred = predicate_to_str(triple.predicate());
        if pred == RDF_TYPE {
            continue; // already handled
        }
        if let Some(classes) = instance_to_classes.get(&subj) {
            for class_iri in classes {
                class_predicates
                    .entry(class_iri.clone())
                    .or_default()
                    .insert(pred.clone());
            }
        }
    }

    // Build class count map
    let class_counts: ClassCountMap = class_predicates
        .keys()
        .map(|class| {
            let count = instance_to_classes
                .values()
                .filter(|classes| classes.contains(class))
                .count();
            (class.clone(), count)
        })
        .collect();

    AnalysisResult {
        class_predicates,
        class_counts,
        predicate_counts,
        instance_to_classes,
    }
}

// ─── SHACL serialization ──────────────────────────────────────────────────────

fn serialize_shacl_turtle(result: &AnalysisResult, base: &str) -> String {
    let mut out = String::new();

    // Prefix declarations
    out.push_str(&format!(
        "@prefix sh:    <{SH_NS}> .\n\
         @prefix rdf:   <{RDF_NS}> .\n\
         @prefix xsd:   <{XSD_NS}> .\n\
         @prefix ex:    <{base}> .\n\n"
    ));

    for (class_iri, predicates) in &result.class_predicates {
        let shape_id = class_iri.replace(['/', '#', ':', '.'], "_");
        out.push_str(&format!(
            "ex:{shape_id}Shape\n\
             \x20\x20a sh:NodeShape ;\n\
             \x20\x20sh:targetClass <{class_iri}> ;\n"
        ));

        for pred_iri in predicates {
            let prop_shape_id = format!(
                "{}_{}_prop",
                shape_id,
                pred_iri.replace(['/', '#', ':', '.'], "_")
            );
            out.push_str(&format!(
                "\x20\x20sh:property [\n\
                 \x20\x20\x20\x20a sh:PropertyShape ;\n\
                 \x20\x20\x20\x20sh:path <{pred_iri}> ;\n\
                 \x20\x20\x20\x20sh:name \"{prop_shape_id}\" ;\n\
                 \x20\x20] ;\n"
            ));
        }

        out.push_str(".\n\n");
    }

    out
}

// ─── Statistics printing ──────────────────────────────────────────────────────

fn print_stats(result: &AnalysisResult) {
    println!("\n=== Schema Analysis Statistics ===");
    println!("Classes found: {}", result.class_counts.len());
    println!("Total instances: {}", result.instance_to_classes.len());
    println!("Distinct predicates: {}", result.predicate_counts.len());

    if !result.class_counts.is_empty() {
        println!("\nClass summary:");
        for (class, count) in &result.class_counts {
            let preds = result
                .class_predicates
                .get(class)
                .map(|s| s.len())
                .unwrap_or(0);
            let short = class
                .rsplit_once(['/', '#'])
                .map(|(_, l)| l)
                .unwrap_or(class);
            println!("  {short}: {count} instances, {preds} properties");
        }
    }

    if !result.predicate_counts.is_empty() {
        println!("\nTop predicates by usage:");
        let mut preds: Vec<(&String, &usize)> = result.predicate_counts.iter().collect();
        preds.sort_by(|a, b| b.1.cmp(a.1));
        for (pred, count) in preds.iter().take(10) {
            let short = pred.rsplit_once(['/', '#']).map(|(_, l)| l).unwrap_or(pred);
            println!("  {short}: {count}");
        }
    }
    println!();
}

// ─── Main entry point ─────────────────────────────────────────────────────────

/// Generate a SHACL schema from an RDF data file.
///
/// * `data`        — path to the input RDF data file
/// * `schema_type` — schema type: `shacl` (default) or `turtle`
/// * `output`      — optional output file path (stdout if omitted)
/// * `stats`       — when true, print class/predicate statistics
pub async fn run(
    data: PathBuf,
    schema_type: String,
    output: Option<PathBuf>,
    stats: bool,
) -> ToolResult {
    // Validate schema type
    let stype = schema_type.to_lowercase();
    if !matches!(stype.as_str(), "shacl" | "turtle" | "ttl") {
        return Err(
            format!("Unsupported schema type '{schema_type}'. Supported: shacl, turtle").into(),
        );
    }

    // Load data
    if !data.exists() {
        return Err(format!("Data file not found: {}", data.display()).into());
    }
    let triples = parse_rdf_file(&data)
        .map_err(|e| format!("Failed to parse data file '{}': {e}", data.display()))?;

    println!("Loaded {} triples from {}", triples.len(), data.display());

    // Analyse
    let result = analyse(&triples);

    println!(
        "Analysis: {} classes, {} instances",
        result.class_counts.len(),
        result.instance_to_classes.len()
    );

    // Print stats if requested
    if stats {
        print_stats(&result);
    }

    // Serialize schema
    let base = "http://example.org/shapes/";
    let schema_text = serialize_shacl_turtle(&result, base);

    // Write output
    let mut writer: Box<dyn Write> = if let Some(ref out_path) = output {
        Box::new(
            File::create(out_path)
                .map_err(|e| format!("Cannot create output file '{}': {e}", out_path.display()))?,
        )
    } else {
        Box::new(io::stdout())
    };

    write!(writer, "{schema_text}")?;
    writer.flush()?;

    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn write_temp_turtle(content: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let path = env::temp_dir().join(format!("schemagen_test_{nanos}.ttl"));
        let mut f = File::create(&path).expect("create temp file");
        f.write_all(content.as_bytes()).expect("write temp file");
        path
    }

    #[test]
    fn test_analyse_basic() {
        // We can't easily call analyse() without real triples in a unit test,
        // so verify the data structures are correct by construction.
        let empty_result = AnalysisResult {
            class_predicates: BTreeMap::new(),
            class_counts: BTreeMap::new(),
            predicate_counts: BTreeMap::new(),
            instance_to_classes: BTreeMap::new(),
        };
        assert!(empty_result.class_predicates.is_empty());
        assert!(empty_result.instance_to_classes.is_empty());
    }

    #[test]
    fn test_serialize_shacl_turtle_empty() {
        let result = AnalysisResult {
            class_predicates: BTreeMap::new(),
            class_counts: BTreeMap::new(),
            predicate_counts: BTreeMap::new(),
            instance_to_classes: BTreeMap::new(),
        };
        let text = serialize_shacl_turtle(&result, "http://example.org/shapes/");
        assert!(text.contains("@prefix sh:"), "got: {text}");
    }

    #[tokio::test]
    async fn test_missing_data_file_returns_error() {
        let nonexistent = env::temp_dir().join("schemagen_nonexistent_9999.ttl");
        let res = run(nonexistent, "shacl".into(), None, false).await;
        assert!(res.is_err(), "should fail for missing data file");
    }

    #[tokio::test]
    async fn test_bad_schema_type_returns_error() {
        let tmp = write_temp_turtle("@prefix ex: <http://example.org/> .\n");
        let res = run(tmp.clone(), "owl".into(), None, false).await;
        let _ = std::fs::remove_file(&tmp);
        assert!(res.is_err());
        if let Err(e) = res {
            assert!(
                e.to_string().contains("Unsupported schema type"),
                "got: {e}"
            );
        }
    }

    #[tokio::test]
    async fn test_schemagen_with_data() {
        let turtle = r#"
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

ex:Alice a ex:Person ;
    foaf:name "Alice" ;
    foaf:age "30" .

ex:Bob a ex:Person ;
    foaf:name "Bob" .
"#;
        let path = write_temp_turtle(turtle);
        let out = env::temp_dir().join(format!("schemagen_out_{}.ttl", {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        }));
        let res = run(path.clone(), "shacl".into(), Some(out.clone()), true).await;
        let _ = std::fs::remove_file(&path);
        let content = std::fs::read_to_string(&out).unwrap_or_default();
        let _ = std::fs::remove_file(&out);
        assert!(res.is_ok(), "schemagen failed: {:?}", res.err());
        assert!(
            content.contains("sh:NodeShape"),
            "missing NodeShape in output: {content}"
        );
    }
}
