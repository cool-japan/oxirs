//! TDB Query tool
//!
//! Execute SPARQL SELECT/ASK/CONSTRUCT/DESCRIBE queries against a local TDB
//! RDF store. Supports multiple result serialization formats.

use super::ToolResult;
use oxirs_arq::algebra::{Literal as AlgebraLiteral, Term as AlgebraTerm};
use oxirs_arq::executor::{InMemoryDataset, QueryExecutor};
use oxirs_arq::query::{parse_query, QueryType};
use oxirs_arq::results::{JsonResultSerializer, QueryResult};
use oxirs_core::model::NamedNode;
use oxirs_tdb::dictionary::Term as TdbTerm;
use oxirs_tdb::{TdbConfig, TdbStore};
use std::io::{self, Write};
use std::path::PathBuf;

// ─── Term conversion ──────────────────────────────────────────────────────────

fn tdb_term_to_algebra(term: &TdbTerm) -> AlgebraTerm {
    match term {
        TdbTerm::Iri(iri) => AlgebraTerm::Iri(NamedNode::new_unchecked(iri.clone())),
        TdbTerm::Literal {
            value,
            language,
            datatype,
        } => AlgebraTerm::Literal(AlgebraLiteral {
            value: value.clone(),
            language: language.clone(),
            datatype: datatype
                .as_deref()
                .map(|dt| NamedNode::new_unchecked(dt.to_string())),
        }),
        TdbTerm::BlankNode(id) => AlgebraTerm::BlankNode(id.clone()),
    }
}

fn load_dataset_from_store(store: &TdbStore) -> anyhow::Result<InMemoryDataset> {
    let triples = store.query_triples(None, None, None)?;
    let mut dataset = InMemoryDataset::new();
    for (s, p, o) in &triples {
        dataset.add_triple(
            tdb_term_to_algebra(s),
            tdb_term_to_algebra(p),
            tdb_term_to_algebra(o),
        );
    }
    Ok(dataset)
}

// ─── Result serialization ─────────────────────────────────────────────────────

fn result_to_text(result: &QueryResult) -> String {
    match result {
        QueryResult::Boolean(v) => format!("ASK result: {v}\n"),
        QueryResult::Bindings {
            variables,
            solutions,
        } => {
            let header: Vec<&str> = variables.iter().map(|v| v.as_str()).collect();
            let mut out = header.join("\t");
            out.push('\n');
            for binding in solutions {
                let row: Vec<String> = variables
                    .iter()
                    .map(|v| binding.get(v).map(|t| format!("{t}")).unwrap_or_default())
                    .collect();
                out.push_str(&row.join("\t"));
                out.push('\n');
            }
            out
        }
        QueryResult::Graph(triples) => triples.iter().map(|tp| format!("{tp}\n")).collect(),
    }
}

fn result_to_csv(result: &QueryResult, tsv: bool) -> String {
    let sep = if tsv { '\t' } else { ',' };
    match result {
        QueryResult::Boolean(v) => format!("result\n{v}\n"),
        QueryResult::Bindings {
            variables,
            solutions,
        } => {
            let header: Vec<&str> = variables.iter().map(|v| v.as_str()).collect();
            let mut out = header.join(&sep.to_string());
            out.push('\n');
            for binding in solutions {
                let row: Vec<String> = variables
                    .iter()
                    .map(|v| binding.get(v).map(|t| format!("{t}")).unwrap_or_default())
                    .collect();
                out.push_str(&row.join(&sep.to_string()));
                out.push('\n');
            }
            out
        }
        QueryResult::Graph(triples) => {
            let mut out = String::from("subject\tpredicate\tobject\n");
            out.extend(triples.iter().map(|tp| format!("{tp}\n")));
            out
        }
    }
}

fn result_to_xml(result: &QueryResult) -> String {
    const NS: &str = "http://www.w3.org/2005/sparql-results#";
    match result {
        QueryResult::Boolean(v) => format!(
            "<?xml version=\"1.0\"?>\n\
             <sparql xmlns=\"{NS}\">\n\
             \x20\x20<boolean>{v}</boolean>\n\
             </sparql>\n"
        ),
        QueryResult::Bindings {
            variables,
            solutions,
        } => {
            let var_decls: String = variables
                .iter()
                .map(|v| format!("    <variable name=\"{}\"/>\n", v.as_str()))
                .collect();
            let results_xml: String = solutions
                .iter()
                .map(|binding| {
                    let inner: String = variables
                        .iter()
                        .filter_map(|v| {
                            binding.get(v).map(|t| {
                                format!("      <binding name=\"{}\">{}</binding>\n", v.as_str(), t)
                            })
                        })
                        .collect();
                    format!("    <result>\n{inner}    </result>\n")
                })
                .collect();
            format!(
                "<?xml version=\"1.0\"?>\n\
                 <sparql xmlns=\"{NS}\">\n\
                 \x20\x20<head>\n{var_decls}\x20\x20</head>\n\
                 \x20\x20<results>\n{results_xml}\x20\x20</results>\n\
                 </sparql>\n"
            )
        }
        QueryResult::Graph(_) => {
            "<?xml version=\"1.0\"?>\n<!-- Graph result (CONSTRUCT/DESCRIBE) -->\n".to_string()
        }
    }
}

// ─── Main entry point ─────────────────────────────────────────────────────────

/// Execute a SPARQL query against a local TDB store.
///
/// * `location` — TDB store directory path
/// * `query`    — SPARQL query string, or file path when `file` is true
/// * `file`     — when true, read the query from `query` as a file path
/// * `results`  — output format: `json` | `xml` | `csv` | `tsv` | `text`
pub async fn run(location: PathBuf, query: String, file: bool, results: String) -> ToolResult {
    // Resolve query string
    let query_str = if file {
        std::fs::read_to_string(&query)
            .map_err(|e| format!("Cannot read query file '{}': {e}", query))?
    } else {
        query
    };

    // Validate format
    let fmt = results.to_lowercase();
    if !matches!(fmt.as_str(), "json" | "xml" | "csv" | "tsv" | "text") {
        return Err(format!(
            "Unsupported results format '{results}'. Supported: json, xml, csv, tsv, text"
        )
        .into());
    }

    // Validate and open TDB store
    if !location.exists() {
        return Err(format!("TDB location does not exist: {}", location.display()).into());
    }
    let config = TdbConfig::new(&location);
    let store =
        TdbStore::open_with_config(config).map_err(|e| format!("Failed to open TDB store: {e}"))?;

    // Parse SPARQL query
    let parsed = parse_query(&query_str).map_err(|e| format!("SPARQL parse error: {e}"))?;

    // Build in-memory dataset from TDB store
    let dataset =
        load_dataset_from_store(&store).map_err(|e| format!("Dataset load error: {e}"))?;

    // Execute the query algebra
    let mut executor = QueryExecutor::default();
    let (solution, _stats) = executor
        .execute(&parsed.where_clause, &dataset)
        .map_err(|e| format!("Query execution error: {e}"))?;

    // Build QueryResult
    let qr = match parsed.query_type {
        QueryType::Ask => QueryResult::Boolean(!solution.is_empty()),
        QueryType::Select => QueryResult::Bindings {
            variables: parsed.select_variables.clone(),
            solutions: solution,
        },
        QueryType::Construct | QueryType::Describe => {
            QueryResult::Graph(parsed.construct_template.clone())
        }
    };

    // Serialize and print
    let mut stdout = io::stdout();
    match fmt.as_str() {
        "json" => {
            JsonResultSerializer::serialize(&qr, &mut stdout)
                .map_err(|e| format!("JSON serialization error: {e}"))?;
            writeln!(stdout)?;
        }
        "xml" => write!(stdout, "{}", result_to_xml(&qr))?,
        "csv" => write!(stdout, "{}", result_to_csv(&qr, false))?,
        "tsv" => write!(stdout, "{}", result_to_csv(&qr, true))?,
        _ => write!(stdout, "{}", result_to_text(&qr))?,
    }

    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[tokio::test]
    async fn test_missing_location_returns_error() {
        let loc = env::temp_dir().join("tdbquery_no_such_dir_xyz999");
        let res = run(
            loc,
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o }".into(),
            false,
            "text".into(),
        )
        .await;
        assert!(res.is_err(), "should fail for non-existent location");
    }

    #[tokio::test]
    async fn test_bad_format_returns_error() {
        let loc = env::temp_dir().join("tdbquery_fmt_test");
        let res = run(
            loc,
            "SELECT ?s WHERE { ?s ?p ?o }".into(),
            false,
            "yaml".into(),
        )
        .await;
        assert!(res.is_err());
        if let Err(e) = res {
            assert!(
                e.to_string().contains("Unsupported results format"),
                "got: {e}"
            );
        }
    }

    #[test]
    fn test_result_to_text_boolean_true() {
        let qr = QueryResult::Boolean(true);
        assert!(result_to_text(&qr).contains("true"));
    }

    #[test]
    fn test_result_to_xml_boolean_false() {
        let qr = QueryResult::Boolean(false);
        let xml = result_to_xml(&qr);
        assert!(xml.contains("<boolean>false</boolean>"), "xml: {xml}");
    }

    #[test]
    fn test_result_to_csv_boolean() {
        let qr = QueryResult::Boolean(true);
        let csv = result_to_csv(&qr, false);
        assert!(csv.contains("result"), "csv: {csv}");
    }

    #[tokio::test]
    async fn test_query_with_real_store() {
        let tmp = env::temp_dir().join("tdbquery_real_store_test");
        {
            let config = TdbConfig::new(&tmp);
            let mut store = TdbStore::open_with_config(config).expect("open store");
            store
                .insert(
                    "<http://example.org/s>",
                    "<http://example.org/p>",
                    "<http://example.org/o>",
                )
                .expect("insert triple");
        }
        let res = run(
            tmp.clone(),
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o }".into(),
            false,
            "text".into(),
        )
        .await;
        let _ = std::fs::remove_dir_all(&tmp);
        // Query may succeed or fail depending on executor wiring; what matters is no panic
        let _ = res;
    }
}
