//! Result set processing tool
//!
//! Convert SPARQL SELECT result sets between JSON, CSV, TSV and XML formats.
//! Reads a SPARQL result file and writes to stdout or a target file in the
//! requested output format.

use super::ToolResult;
use oxirs_arq::{
    algebra::{Binding, Variable},
    results::{QueryResult, ResultFormat, ResultSerializer},
    Iri, Literal, Term,
};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;

/// Run rset command — transcode a SPARQL result set file.
pub async fn run(
    _input: PathBuf,
    _input_format: Option<String>,
    _output_format: String,
    _output: Option<PathBuf>,
) -> ToolResult {
    let input_fmt = _input_format
        .clone()
        .unwrap_or_else(|| detect_result_format(&_input));

    println!("Input  : {} (format: {input_fmt})", _input.display());
    println!("Output format: {_output_format}");

    let raw = std::fs::read_to_string(&_input)
        .map_err(|e| format!("Cannot read {}: {e}", _input.display()))?;

    let query_result = parse_result_set(&raw, &input_fmt)?;

    let out_format = parse_output_format(&_output_format)?;

    let mut buf: Vec<u8> = Vec::new();
    ResultSerializer::serialize(&query_result, out_format, &mut buf)
        .map_err(|e| format!("Serialization failed: {e}"))?;

    match &_output {
        None => {
            io::stdout()
                .write_all(&buf)
                .map_err(|e| format!("Write to stdout failed: {e}"))?;
            io::stdout()
                .flush()
                .map_err(|e| format!("Flush stdout failed: {e}"))?;
        }
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Cannot create directory: {e}"))?;
            }
            std::fs::write(path, &buf)
                .map_err(|e| format!("Cannot write to {}: {e}", path.display()))?;
            println!("Written: {}", path.display());
        }
    }

    Ok(())
}

/// Detect result format from file extension.
fn detect_result_format(path: &std::path::Path) -> String {
    match path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "json" | "srj" => "json",
        "xml" | "srx" => "xml",
        "csv" => "csv",
        "tsv" => "tsv",
        _ => "json",
    }
    .to_string()
}

/// Parse the output format string into a `ResultFormat`.
fn parse_output_format(format: &str) -> ToolResult<ResultFormat> {
    match format.to_lowercase().as_str() {
        "json" | "srj" => Ok(ResultFormat::Json),
        "xml" | "srx" => Ok(ResultFormat::Xml),
        "csv" => Ok(ResultFormat::Csv),
        "tsv" => Ok(ResultFormat::Tsv),
        other => Err(format!("Unsupported output format: '{other}'").into()),
    }
}

/// Parse a SPARQL result set string into a `QueryResult`.
fn parse_result_set(raw: &str, format: &str) -> ToolResult<QueryResult> {
    match format.to_lowercase().as_str() {
        "json" | "srj" => parse_sparql_results_json(raw),
        "csv" => parse_sparql_results_csv(raw),
        "tsv" => parse_sparql_results_tsv(raw),
        "xml" | "srx" => {
            Err("XML input format is not yet supported for rset. Use JSON or CSV/TSV input.".into())
        }
        other => Err(format!("Unsupported input format: '{other}'").into()),
    }
}

/// Parse SPARQL Results JSON (W3C format).
fn parse_sparql_results_json(raw: &str) -> ToolResult<QueryResult> {
    let json: JsonValue = serde_json::from_str(raw).map_err(|e| format!("Invalid JSON: {e}"))?;

    // ASK result?
    if let Some(b) = json.get("boolean").and_then(JsonValue::as_bool) {
        return Ok(QueryResult::Boolean(b));
    }

    // SELECT result
    let vars_json = json
        .pointer("/head/vars")
        .and_then(JsonValue::as_array)
        .ok_or("Missing /head/vars in SPARQL Results JSON")?;

    let variables: Vec<Variable> = vars_json
        .iter()
        .filter_map(|v| v.as_str())
        .map(|name| Variable::new(name).ok())
        .collect::<Option<Vec<_>>>()
        .ok_or("Invalid variable name in /head/vars")?;

    let bindings_json = json
        .pointer("/results/bindings")
        .and_then(JsonValue::as_array)
        .ok_or("Missing /results/bindings in SPARQL Results JSON")?;

    let mut solutions: Vec<Binding> = Vec::new();
    for row in bindings_json {
        let mut binding: Binding = HashMap::new();
        for var in &variables {
            if let Some(cell) = row.get(var.as_str()) {
                let term = json_cell_to_term(cell)?;
                binding.insert(var.clone(), term);
            }
        }
        solutions.push(binding);
    }

    Ok(QueryResult::Bindings {
        variables,
        solutions,
    })
}

/// Convert a SPARQL Results JSON cell `{"type":…,"value":…}` to a `Term`.
fn json_cell_to_term(cell: &JsonValue) -> ToolResult<Term> {
    let term_type = cell
        .get("type")
        .and_then(JsonValue::as_str)
        .ok_or("Missing 'type' in binding cell")?;

    let value = cell
        .get("value")
        .and_then(JsonValue::as_str)
        .ok_or("Missing 'value' in binding cell")?;

    match term_type {
        "uri" => {
            let iri = Iri::new(value).map_err(|e| format!("Invalid IRI '{value}': {e}"))?;
            Ok(Term::Iri(iri))
        }
        "bnode" => Ok(Term::BlankNode(value.to_string())),
        "literal" => {
            let lang = cell.get("xml:lang").and_then(JsonValue::as_str);
            let datatype = cell.get("datatype").and_then(JsonValue::as_str);
            let lit = if let Some(l) = lang {
                Literal::with_language(value.to_string(), l.to_string())
            } else if let Some(dt) = datatype {
                let dt_iri =
                    Iri::new(dt).map_err(|e| format!("Invalid datatype IRI '{dt}': {e}"))?;
                Literal {
                    value: value.to_string(),
                    language: None,
                    datatype: Some(dt_iri),
                }
            } else {
                Literal {
                    value: value.to_string(),
                    language: None,
                    datatype: None,
                }
            };
            Ok(Term::Literal(lit))
        }
        other => Err(format!("Unknown term type: '{other}'").into()),
    }
}

/// Parse SPARQL Results CSV (W3C format — header row then data rows).
fn parse_sparql_results_csv(raw: &str) -> ToolResult<QueryResult> {
    let mut lines = raw.lines();

    let header = lines.next().ok_or("CSV is empty")?;
    let var_names: Vec<&str> = header.split(',').map(str::trim).collect();

    let variables: Vec<Variable> = var_names
        .iter()
        .map(|name| Variable::new(*name).ok())
        .collect::<Option<Vec<_>>>()
        .ok_or("Invalid variable name in CSV header")?;

    let mut solutions: Vec<Binding> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = split_csv_line(line);
        let mut binding: Binding = HashMap::new();
        for (var, cell) in variables.iter().zip(cells.iter()) {
            let cell = cell.trim().trim_matches('"');
            if !cell.is_empty() {
                let term = if cell.starts_with("http://")
                    || cell.starts_with("https://")
                    || cell.starts_with("urn:")
                {
                    let iri = Iri::new(cell)
                        .map_err(|e| format!("Invalid IRI from CSV '{cell}': {e}"))?;
                    Term::Iri(iri)
                } else if let Some(bnode) = cell.strip_prefix("_:") {
                    Term::BlankNode(bnode.to_string())
                } else {
                    Term::Literal(Literal {
                        value: cell.to_string(),
                        language: None,
                        datatype: None,
                    })
                };
                binding.insert(var.clone(), term);
            }
        }
        solutions.push(binding);
    }

    Ok(QueryResult::Bindings {
        variables,
        solutions,
    })
}

/// Very simple CSV line splitter (handles double-quoted fields with commas).
fn split_csv_line(line: &str) -> Vec<&str> {
    let mut fields: Vec<&str> = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;
    let bytes = line.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' => in_quotes = !in_quotes,
            b',' if !in_quotes => {
                fields.push(&line[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    fields.push(&line[start..]);
    fields
}

/// Parse SPARQL Results TSV (header ?var1\t?var2 then data rows).
fn parse_sparql_results_tsv(raw: &str) -> ToolResult<QueryResult> {
    let mut lines = raw.lines();

    let header = lines.next().ok_or("TSV is empty")?;
    let var_names: Vec<&str> = header.split('\t').map(str::trim).collect();

    let variables: Vec<Variable> = var_names
        .iter()
        .map(|name| {
            let clean = name.trim_start_matches('?');
            Variable::new(clean).ok()
        })
        .collect::<Option<Vec<_>>>()
        .ok_or("Invalid variable name in TSV header")?;

    let mut solutions: Vec<Binding> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split('\t').collect();
        let mut binding: Binding = HashMap::new();
        for (var, cell) in variables.iter().zip(cells.iter()) {
            let cell = cell.trim();
            if !cell.is_empty() {
                let term = if cell.starts_with('<') && cell.ends_with('>') {
                    let iri_str = &cell[1..cell.len() - 1];
                    let iri = Iri::new(iri_str)
                        .map_err(|e| format!("Invalid IRI from TSV '{iri_str}': {e}"))?;
                    Term::Iri(iri)
                } else if let Some(bnode) = cell.strip_prefix("_:") {
                    Term::BlankNode(bnode.to_string())
                } else {
                    Term::Literal(Literal {
                        value: cell.to_string(),
                        language: None,
                        datatype: None,
                    })
                };
                binding.insert(var.clone(), term);
            }
        }
        solutions.push(binding);
    }

    Ok(QueryResult::Bindings {
        variables,
        solutions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_bindings() {
        let json = r#"{
            "head": {"vars": ["s", "p"]},
            "results": {"bindings": [
                {"s": {"type": "uri", "value": "http://example.org/s"},
                 "p": {"type": "uri", "value": "http://example.org/p"}}
            ]}
        }"#;
        let result = parse_sparql_results_json(json).expect("parse JSON");
        match result {
            QueryResult::Bindings {
                variables,
                solutions,
            } => {
                assert_eq!(variables.len(), 2);
                assert_eq!(solutions.len(), 1);
            }
            _ => panic!("expected Bindings"),
        }
    }

    #[test]
    fn test_parse_json_boolean() {
        let json = r#"{"head": {}, "boolean": true}"#;
        let result = parse_sparql_results_json(json).expect("parse JSON boolean");
        assert!(matches!(result, QueryResult::Boolean(true)));
    }

    #[test]
    fn test_parse_csv_bindings() {
        let csv = "s,p\nhttp://example.org/s,http://example.org/p\n";
        let result = parse_sparql_results_csv(csv).expect("parse CSV");
        match result {
            QueryResult::Bindings { solutions, .. } => {
                assert_eq!(solutions.len(), 1);
            }
            _ => panic!("expected Bindings"),
        }
    }

    #[test]
    fn test_parse_tsv_bindings() {
        let tsv = "?s\t?p\n<http://s>\t<http://p>\n";
        let result = parse_sparql_results_tsv(tsv).expect("parse TSV");
        match result {
            QueryResult::Bindings { solutions, .. } => {
                assert_eq!(solutions.len(), 1);
            }
            _ => panic!("expected Bindings"),
        }
    }

    #[test]
    fn test_unsupported_input_format() {
        let result = parse_result_set("", "xml");
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_format_json() {
        let p = std::path::Path::new("results.srj");
        assert_eq!(detect_result_format(p), "json");
    }

    #[test]
    fn test_detect_format_csv() {
        let p = std::path::Path::new("results.csv");
        assert_eq!(detect_result_format(p), "csv");
    }

    #[test]
    fn test_parse_output_format_unknown() {
        let result = parse_output_format("parquet");
        assert!(result.is_err());
    }

    #[test]
    fn test_split_csv_line_quoted() {
        let line = r#"hello,"world, earth",foo"#;
        let fields = split_csv_line(line);
        assert_eq!(fields.len(), 3);
    }
}
