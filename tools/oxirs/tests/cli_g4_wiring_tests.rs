//! Integration tests for the G4 second wiring batch (run(cli) pattern).
//!
//! Exercises the newly-wired top-level commands end-to-end through the public
//! `run(Cli)` entry point, using temp-dir files only:
//!   - `Commands::Inspect` (consolidated data profiler)
//!   - `Commands::SchemaGen { advanced: true }` (schema inferencer)
//!   - `HistoryAction::ExportCsv` / `HistoryAction::Similar`
//!   - `ProfilerAction::Run { flamegraph }`

use oxirs::{run, Cli, Commands, HistoryAction, ProfilerAction};
use std::fs;

/// Build a quiet, non-interactive CLI wrapper around a command.
fn cli(command: Commands) -> Cli {
    Cli {
        command,
        verbose: false,
        config: None,
        quiet: true,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    }
}

fn unique_path(name: &str, ext: &str) -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "oxirs_g4_{name}_{}_{nanos}.{ext}",
        std::process::id()
    ))
}

const SAMPLE_TTL: &str = "@prefix ex: <http://example.org/> .\n\
@prefix foaf: <http://xmlns.com/foaf/0.1/> .\n\
ex:alice a foaf:Person ;\n\
    foaf:name \"Alice\" ;\n\
    foaf:knows ex:bob .\n\
ex:bob a foaf:Person ;\n\
    foaf:name \"Bob\" .\n";

// ── Item 1: inspect ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_inspect_real_file_writes_report() {
    let input = unique_path("inspect_in", "ttl");
    let output = unique_path("inspect_out", "txt");
    fs::write(&input, SAMPLE_TTL).expect("write input");

    let result = run(cli(Commands::Inspect {
        file: input.clone(),
        format: "text".to_string(),
        output: Some(output.clone()),
        top: 20,
        no_connectivity: false,
        no_quality: false,
    }))
    .await;

    let report = fs::read_to_string(&output).unwrap_or_default();
    let _ = fs::remove_file(&input);
    let _ = fs::remove_file(&output);

    assert!(result.is_ok(), "inspect failed: {:?}", result.err());
    assert!(
        report.contains("Object Type Distribution"),
        "report: {report}"
    );
    assert!(report.contains("Basic Statistics"), "report: {report}");
}

#[tokio::test]
async fn test_inspect_missing_file_is_explicit_error() {
    let missing = unique_path("inspect_missing", "ttl");
    let _ = fs::remove_file(&missing);

    let result = run(cli(Commands::Inspect {
        file: missing,
        format: "text".to_string(),
        output: None,
        top: 20,
        no_connectivity: false,
        no_quality: false,
    }))
    .await;

    assert!(
        result.is_err(),
        "missing input must surface an explicit error"
    );
}

#[tokio::test]
async fn test_inspect_json_report() {
    let input = unique_path("inspect_json_in", "ttl");
    let output = unique_path("inspect_json_out", "json");
    fs::write(&input, SAMPLE_TTL).expect("write input");

    let result = run(cli(Commands::Inspect {
        file: input.clone(),
        format: "json".to_string(),
        output: Some(output.clone()),
        top: 10,
        no_connectivity: false,
        no_quality: false,
    }))
    .await;

    let report = fs::read_to_string(&output).unwrap_or_default();
    let _ = fs::remove_file(&input);
    let _ = fs::remove_file(&output);

    assert!(result.is_ok(), "inspect json failed: {:?}", result.err());
    assert!(report.contains("\"object_types\""), "report: {report}");
    assert!(report.contains("\"triple_count\""), "report: {report}");
}

// ── Item 2: schema-gen --advanced ────────────────────────────────────────────

#[tokio::test]
async fn test_schemagen_advanced_infers_owl() {
    let input = unique_path("schema_in", "ttl");
    let output = unique_path("schema_out", "ttl");
    fs::write(&input, SAMPLE_TTL).expect("write input");

    let result = run(cli(Commands::SchemaGen {
        data: input.clone(),
        schema_type: "owl".to_string(),
        output: Some(output.clone()),
        stats: true,
        advanced: true,
    }))
    .await;

    let schema = fs::read_to_string(&output).unwrap_or_default();
    let _ = fs::remove_file(&input);
    let _ = fs::remove_file(&output);

    assert!(
        result.is_ok(),
        "schemagen advanced failed: {:?}",
        result.err()
    );
    assert!(schema.contains("owl:Class"), "schema: {schema}");
    // The sample types instances as foaf:Person, so that is the inferred class.
    assert!(
        schema.contains("http://xmlns.com/foaf/0.1/Person"),
        "schema: {schema}"
    );
    // Advanced inferencer emits domain/range that plain SHACL generation omits.
    assert!(schema.contains("rdfs:domain"), "schema: {schema}");
}

#[tokio::test]
async fn test_schemagen_default_still_shacl() {
    // Default (non-advanced) behavior must be unchanged: SHACL output.
    let input = unique_path("schema_def_in", "ttl");
    let output = unique_path("schema_def_out", "ttl");
    fs::write(&input, SAMPLE_TTL).expect("write input");

    let result = run(cli(Commands::SchemaGen {
        data: input.clone(),
        schema_type: "shacl".to_string(),
        output: Some(output.clone()),
        stats: false,
        advanced: false,
    }))
    .await;

    let schema = fs::read_to_string(&output).unwrap_or_default();
    let _ = fs::remove_file(&input);
    let _ = fs::remove_file(&output);

    assert!(
        result.is_ok(),
        "schemagen default failed: {:?}",
        result.err()
    );
    assert!(schema.contains("sh:NodeShape"), "schema: {schema}");
}

// ── Item 3: history export-csv ───────────────────────────────────────────────

#[tokio::test]
async fn test_history_export_csv_writes_header() {
    // Reads the wired (default-location) history store, non-destructively, and
    // writes a CSV. Regardless of whether any history exists, the CSV must have
    // the wired-store header.
    let output = unique_path("history_csv", "csv");

    let result = run(cli(Commands::History {
        action: HistoryAction::ExportCsv {
            output: output.clone(),
        },
    }))
    .await;

    let csv = fs::read_to_string(&output).unwrap_or_default();
    let _ = fs::remove_file(&output);

    assert!(
        result.is_ok(),
        "history export-csv failed: {:?}",
        result.err()
    );
    assert!(
        csv.starts_with("id,timestamp,dataset,success,execution_time_ms,result_count,error,query"),
        "csv: {csv}"
    );
}

// ── Item 4: history similar ──────────────────────────────────────────────────

#[tokio::test]
async fn test_history_similar_runs() {
    // Non-destructive read of the wired history store; must not error even when
    // history is empty (it prints a "no history" notice instead).
    let result = run(cli(Commands::History {
        action: HistoryAction::Similar {
            query: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
            top: 3,
        },
    }))
    .await;

    assert!(result.is_ok(), "history similar failed: {:?}", result.err());
}

// ── Item 5: profile --flamegraph ─────────────────────────────────────────────

#[tokio::test]
async fn test_profile_writes_flamegraph_svg() {
    let svg_path = unique_path("flamegraph", "svg");

    let result = run(cli(Commands::Profile {
        action: ProfilerAction::Run {
            dataset: "mem".to_string(),
            query: "SELECT DISTINCT ?name WHERE { \
                    ?person foaf:name ?name . \
                    OPTIONAL { ?person foaf:age ?age } \
                    FILTER(?age > 18) } ORDER BY ?name"
                .to_string(),
            file: false,
            iterations: 3,
            suggestions: false,
            flamegraph: Some(svg_path.clone()),
        },
    }))
    .await;

    let svg = fs::read_to_string(&svg_path).unwrap_or_default();
    let _ = fs::remove_file(&svg_path);

    assert!(
        result.is_ok(),
        "profile flamegraph failed: {:?}",
        result.err()
    );
    assert!(svg.contains("<svg"), "svg: {}", &svg[..svg.len().min(200)]);
    assert!(
        svg.contains("execute") || svg.contains("parse"),
        "flamegraph should contain operator sample names"
    );
}
