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
        "xml" | "srx" => parse_sparql_results_xml(raw),
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

/// Parse SPARQL Results XML (W3C SPARQL Results XML Format, SRX).
///
/// Supports SELECT results (bindings) and ASK results (boolean).
/// Implements <https://www.w3.org/TR/rdf-sparql-XMLres/>
fn parse_sparql_results_xml(raw: &str) -> ToolResult<QueryResult> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    /// Tiny parser state machine
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum State {
        Root,
        Head,
        Results,
        Result,
        Binding,
        Uri,
        Literal,
        BNode,
    }

    let mut reader = Reader::from_str(raw);
    reader.config_mut().trim_text(true);

    let mut variables: Vec<Variable> = Vec::new();
    let mut solutions: Vec<Binding> = Vec::new();

    let mut state = State::Root;
    let mut current_binding: Option<Binding> = None;
    let mut current_var_name: Option<String> = None;
    let mut text_buf = String::new();
    // Literal attributes
    let mut lit_lang: Option<String> = None;
    let mut lit_datatype: Option<String> = None;
    // For ASK results
    let mut boolean_value: Option<bool> = None;
    let mut in_boolean = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let local = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                // Strip namespace prefix if present (e.g. "sparql:result" → "result")
                let local = local
                    .split_once(':')
                    .map(|(_, l)| l)
                    .unwrap_or(local.as_str())
                    .to_owned();
                match (state, local.as_str()) {
                    (State::Root, "sparql") => {}
                    (State::Root, "head") => state = State::Head,
                    (State::Head, "variable") => {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"name" {
                                let name = String::from_utf8_lossy(&attr.value).into_owned();
                                if let Ok(var) = Variable::new(&name) {
                                    variables.push(var);
                                }
                            }
                        }
                    }
                    (State::Root, "results") => state = State::Results,
                    (State::Results, "result") => {
                        current_binding = Some(HashMap::new());
                        state = State::Result;
                    }
                    (State::Root | State::Head, "boolean") => {
                        in_boolean = true;
                        text_buf.clear();
                    }
                    (State::Result, "binding") => {
                        current_var_name = None;
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"name" {
                                current_var_name =
                                    Some(String::from_utf8_lossy(&attr.value).into_owned());
                            }
                        }
                        state = State::Binding;
                    }
                    (State::Binding, "uri") => {
                        text_buf.clear();
                        state = State::Uri;
                    }
                    (State::Binding, "literal") => {
                        text_buf.clear();
                        lit_lang = None;
                        lit_datatype = None;
                        for attr in e.attributes().flatten() {
                            let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
                            let val = String::from_utf8_lossy(&attr.value).into_owned();
                            match key.as_str() {
                                "xml:lang" | "lang" => lit_lang = Some(val),
                                "datatype" => lit_datatype = Some(val),
                                _ => {}
                            }
                        }
                        state = State::Literal;
                    }
                    (State::Binding, "bnode") => {
                        text_buf.clear();
                        state = State::BNode;
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) if state == State::Head => {
                // Self-closing <variable name="x"/> inside <head>
                let local = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                let local = local
                    .split_once(':')
                    .map(|(_, l)| l)
                    .unwrap_or(local.as_str())
                    .to_owned();
                if local == "variable" {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"name" {
                            let name = String::from_utf8_lossy(&attr.value).into_owned();
                            if let Ok(var) = Variable::new(&name) {
                                variables.push(var);
                            }
                        }
                    }
                }
            }
            Ok(Event::Empty(_)) => {}
            Ok(Event::Text(ref e))
                if in_boolean || matches!(state, State::Uri | State::Literal | State::BNode) =>
            {
                let text = String::from_utf8_lossy(e.as_ref()).into_owned();
                text_buf.push_str(&text);
            }
            Ok(Event::Text(_)) => {}
            Ok(Event::End(ref e)) => {
                let local = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                let local = local
                    .split_once(':')
                    .map(|(_, l)| l)
                    .unwrap_or(local.as_str())
                    .to_owned();
                match (state, local.as_str()) {
                    (State::Head, "head") => state = State::Root,
                    (State::Results, "results") => state = State::Root,
                    (State::Result, "result") => {
                        if let Some(binding) = current_binding.take() {
                            solutions.push(binding);
                        }
                        state = State::Results;
                    }
                    (State::Binding, "binding") => {
                        state = State::Result;
                        current_var_name = None;
                    }
                    (State::Uri, "uri") => {
                        if let (Some(ref name), Some(ref mut binding)) =
                            (&current_var_name, &mut current_binding)
                        {
                            let iri_str = text_buf.trim();
                            match Iri::new(iri_str) {
                                Ok(iri) => {
                                    if let Ok(var) = Variable::new(name) {
                                        binding.insert(var, Term::Iri(iri));
                                    }
                                }
                                Err(e) => {
                                    return Err(
                                        format!("Invalid IRI in SRX '{iri_str}': {e}").into()
                                    );
                                }
                            }
                        }
                        text_buf.clear();
                        state = State::Binding;
                    }
                    (State::Literal, "literal") => {
                        if let (Some(ref name), Some(ref mut binding)) =
                            (&current_var_name, &mut current_binding)
                        {
                            let val = text_buf.clone();
                            let lit = if let Some(lang) = lit_lang.take() {
                                Literal::with_language(val, lang)
                            } else if let Some(dt_str) = lit_datatype.take() {
                                let dt_iri = Iri::new(&dt_str)
                                    .map_err(|e| format!("Invalid datatype IRI '{dt_str}': {e}"))?;
                                Literal {
                                    value: val,
                                    language: None,
                                    datatype: Some(dt_iri),
                                }
                            } else {
                                Literal {
                                    value: val,
                                    language: None,
                                    datatype: None,
                                }
                            };
                            if let Ok(var) = Variable::new(name) {
                                binding.insert(var, Term::Literal(lit));
                            }
                        }
                        text_buf.clear();
                        state = State::Binding;
                    }
                    (State::BNode, "bnode") => {
                        if let (Some(ref name), Some(ref mut binding)) =
                            (&current_var_name, &mut current_binding)
                        {
                            if let Ok(var) = Variable::new(name) {
                                binding.insert(var, Term::BlankNode(text_buf.trim().to_string()));
                            }
                        }
                        text_buf.clear();
                        state = State::Binding;
                    }
                    (_, "boolean") if in_boolean => {
                        let trimmed = text_buf.trim().to_lowercase();
                        boolean_value = match trimmed.as_str() {
                            "true" => Some(true),
                            "false" => Some(false),
                            other => {
                                return Err(
                                    format!("Invalid boolean value in SRX: '{other}'").into()
                                );
                            }
                        };
                        in_boolean = false;
                        text_buf.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parsing error in SRX: {e}").into()),
            _ => {}
        }
    }

    if let Some(val) = boolean_value {
        return Ok(QueryResult::Boolean(val));
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
        // "parquet" is not a supported input format
        let result = parse_result_set("", "parquet");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_xml_boolean_true() {
        let srx = r#"<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head/>
  <boolean>true</boolean>
</sparql>"#;
        let result = parse_sparql_results_xml(srx).expect("parse SRX boolean true");
        assert!(matches!(result, QueryResult::Boolean(true)));
    }

    #[test]
    fn test_parse_xml_boolean_false() {
        let srx = r#"<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head/>
  <boolean>false</boolean>
</sparql>"#;
        let result = parse_sparql_results_xml(srx).expect("parse SRX boolean false");
        assert!(matches!(result, QueryResult::Boolean(false)));
    }

    #[test]
    fn test_parse_xml_bindings_uri() {
        let srx = r#"<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head>
    <variable name="s"/>
    <variable name="p"/>
    <variable name="o"/>
  </head>
  <results>
    <result>
      <binding name="s"><uri>http://example.org/alice</uri></binding>
      <binding name="p"><uri>http://xmlns.com/foaf/0.1/name</uri></binding>
      <binding name="o"><literal>Alice</literal></binding>
    </result>
  </results>
</sparql>"#;
        let result = parse_sparql_results_xml(srx).expect("parse SRX bindings");
        match result {
            QueryResult::Bindings {
                variables,
                solutions,
            } => {
                assert_eq!(variables.len(), 3);
                assert_eq!(solutions.len(), 1);
                // Check that the IRI term was parsed
                let s_var = Variable::new("s").expect("variable s");
                let term = solutions[0].get(&s_var).expect("binding for ?s");
                assert!(matches!(term, Term::Iri(_)));
                // Check that the literal was parsed
                let o_var = Variable::new("o").expect("variable o");
                let lit = solutions[0].get(&o_var).expect("binding for ?o");
                assert!(matches!(lit, Term::Literal(_)));
            }
            _ => panic!("expected Bindings"),
        }
    }

    #[test]
    fn test_parse_xml_literal_with_lang() {
        let srx = r#"<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head><variable name="label"/></head>
  <results>
    <result>
      <binding name="label"><literal xml:lang="en">Hello</literal></binding>
    </result>
  </results>
</sparql>"#;
        let result = parse_sparql_results_xml(srx).expect("parse SRX lang literal");
        match result {
            QueryResult::Bindings { solutions, .. } => {
                assert_eq!(solutions.len(), 1);
                let label_var = Variable::new("label").expect("variable");
                if let Some(Term::Literal(lit)) = solutions[0].get(&label_var) {
                    assert_eq!(lit.value, "Hello");
                    assert_eq!(lit.language.as_deref(), Some("en"));
                } else {
                    panic!("expected literal with lang");
                }
            }
            _ => panic!("expected Bindings"),
        }
    }

    #[test]
    fn test_parse_xml_literal_with_datatype() {
        let srx = r#"<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head><variable name="age"/></head>
  <results>
    <result>
      <binding name="age">
        <literal datatype="http://www.w3.org/2001/XMLSchema#integer">42</literal>
      </binding>
    </result>
  </results>
</sparql>"#;
        let result = parse_sparql_results_xml(srx).expect("parse SRX datatype literal");
        match result {
            QueryResult::Bindings { solutions, .. } => {
                let age_var = Variable::new("age").expect("variable");
                if let Some(Term::Literal(lit)) = solutions[0].get(&age_var) {
                    assert_eq!(lit.value, "42");
                    assert!(lit.datatype.is_some());
                } else {
                    panic!("expected typed literal");
                }
            }
            _ => panic!("expected Bindings"),
        }
    }

    #[test]
    fn test_parse_xml_bnode() {
        let srx = r#"<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head><variable name="x"/></head>
  <results>
    <result>
      <binding name="x"><bnode>b0</bnode></binding>
    </result>
  </results>
</sparql>"#;
        let result = parse_sparql_results_xml(srx).expect("parse SRX bnode");
        match result {
            QueryResult::Bindings { solutions, .. } => {
                let x_var = Variable::new("x").expect("variable");
                assert!(matches!(solutions[0].get(&x_var), Some(Term::BlankNode(_))));
            }
            _ => panic!("expected Bindings"),
        }
    }

    #[test]
    fn test_parse_xml_empty_results() {
        let srx = r#"<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head><variable name="x"/></head>
  <results/>
</sparql>"#;
        let result = parse_sparql_results_xml(srx).expect("parse SRX empty results");
        match result {
            QueryResult::Bindings {
                variables,
                solutions,
            } => {
                assert_eq!(variables.len(), 1);
                assert_eq!(solutions.len(), 0);
            }
            _ => panic!("expected Bindings"),
        }
    }

    #[test]
    fn test_parse_xml_multiple_results() {
        let srx = r#"<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head><variable name="s"/></head>
  <results>
    <result>
      <binding name="s"><uri>http://example.org/a</uri></binding>
    </result>
    <result>
      <binding name="s"><uri>http://example.org/b</uri></binding>
    </result>
    <result>
      <binding name="s"><uri>http://example.org/c</uri></binding>
    </result>
  </results>
</sparql>"#;
        let result = parse_sparql_results_xml(srx).expect("parse SRX multiple results");
        match result {
            QueryResult::Bindings { solutions, .. } => {
                assert_eq!(solutions.len(), 3);
            }
            _ => panic!("expected Bindings"),
        }
    }

    #[test]
    fn test_parse_xml_via_parse_result_set() {
        // Test that the xml/srx dispatch in parse_result_set works
        let srx = r#"<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head/>
  <boolean>true</boolean>
</sparql>"#;
        let result = parse_result_set(srx, "srx").expect("parse SRX via dispatcher");
        assert!(matches!(result, QueryResult::Boolean(true)));

        let result2 = parse_result_set(srx, "xml").expect("parse XML via dispatcher");
        assert!(matches!(result2, QueryResult::Boolean(true)));
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
