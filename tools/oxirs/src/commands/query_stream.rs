//! Streaming SPARQL Query Results
//!
//! Implements incremental, memory-efficient streaming of SPARQL query results
//! supporting NDJSON, CSV-stream, and TSV-stream output formats.
//!
//! ## Features
//!
//! - NDJSON (Newline-Delimited JSON) streaming output
//! - CSV and TSV streaming without buffering full result set
//! - Configurable page size, timeout, and row limits
//! - Heartbeat emission for long-running queries
//! - Incremental header + row writing via `StreamingResultWriter`

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

/// Output format for streaming queries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFormat {
    /// Newline-Delimited JSON
    JsonStream,
    /// CSV with header row
    CsvStream,
    /// TSV with header row
    TsvStream,
}

impl StreamFormat {
    /// Parse from CLI flag string
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "json-stream" | "ndjson" => Ok(Self::JsonStream),
            "csv-stream" | "csv" => Ok(Self::CsvStream),
            "tsv-stream" | "tsv" => Ok(Self::TsvStream),
            other => Err(anyhow!(
                "Unknown streaming format '{}'. Use: json-stream, csv-stream, tsv-stream",
                other
            )),
        }
    }

    /// Content-type MIME string
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::JsonStream => "application/x-ndjson",
            Self::CsvStream => "text/csv",
            Self::TsvStream => "text/tab-separated-values",
        }
    }
}

/// Configuration for a streaming query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingQueryConfig {
    /// Number of rows to yield per page/chunk
    pub page_size: usize,
    /// Query execution timeout in milliseconds
    pub timeout_ms: u64,
    /// Maximum rows to return (None = unlimited)
    pub max_rows: Option<usize>,
    /// Emit heartbeat comment every N seconds (0 = disabled)
    pub heartbeat_interval_secs: u64,
}

impl Default for StreamingQueryConfig {
    fn default() -> Self {
        Self {
            page_size: 1000,
            timeout_ms: 30_000,
            max_rows: None,
            heartbeat_interval_secs: 5,
        }
    }
}

impl StreamingQueryConfig {
    /// Create a new config with specified page size
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = page_size;
        self
    }

    /// Set timeout in milliseconds
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set maximum rows
    pub fn with_max_rows(mut self, max_rows: usize) -> Self {
        self.max_rows = Some(max_rows);
        self
    }

    /// Set heartbeat interval in seconds (0 = disabled)
    pub fn with_heartbeat_interval(mut self, secs: u64) -> Self {
        self.heartbeat_interval_secs = secs;
        self
    }
}

/// A single SPARQL binding — maps variable names to their values
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Binding {
    /// Variable name → RDF term value (serialized as string)
    pub values: HashMap<String, String>,
}

impl Binding {
    /// Create a new binding from a map
    pub fn new(values: HashMap<String, String>) -> Self {
        Self { values }
    }

    /// Get value for a variable
    pub fn get(&self, var: &str) -> Option<&str> {
        self.values.get(var).map(|s| s.as_str())
    }

    /// Build from pairs of (variable, value)
    pub fn from_pairs(
        pairs: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self {
            values: pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }
}

/// Writes streaming SPARQL query results incrementally to any `Write` sink
///
/// Supports NDJSON, CSV-stream, and TSV-stream formats.
pub struct StreamingResultWriter<W: Write> {
    inner: BufWriter<W>,
    format: StreamFormat,
    header_written: bool,
    row_count: usize,
    last_heartbeat: Instant,
    heartbeat_interval_secs: u64,
    variables: Vec<String>,
}

impl<W: Write> StreamingResultWriter<W> {
    /// Create a new streaming writer
    pub fn new(writer: W, format: StreamFormat, config: &StreamingQueryConfig) -> Self {
        Self {
            inner: BufWriter::new(writer),
            format,
            header_written: false,
            row_count: 0,
            last_heartbeat: Instant::now(),
            heartbeat_interval_secs: config.heartbeat_interval_secs,
            variables: Vec::new(),
        }
    }

    /// Write the header row (variable names for CSV/TSV; JSON metadata for NDJSON)
    pub fn write_header(&mut self, vars: &[String]) -> Result<()> {
        self.variables = vars.to_vec();
        match self.format {
            StreamFormat::CsvStream => {
                let header = vars
                    .iter()
                    .map(|v| csv_escape(v))
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(self.inner, "{}", header)?;
            }
            StreamFormat::TsvStream => {
                let header = vars.join("\t");
                writeln!(self.inner, "{}", header)?;
            }
            StreamFormat::JsonStream => {
                // Emit a metadata header line as NDJSON
                let meta = serde_json::json!({ "type": "header", "vars": vars });
                writeln!(self.inner, "{}", meta)?;
            }
        }
        self.header_written = true;
        Ok(())
    }

    /// Write a single binding row
    pub fn write_row(&mut self, binding: &Binding) -> Result<()> {
        if !self.header_written {
            return Err(anyhow!("write_header must be called before write_row"));
        }
        match self.format {
            StreamFormat::CsvStream => {
                let row = self
                    .variables
                    .iter()
                    .map(|v| csv_escape(binding.get(v).unwrap_or("")))
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(self.inner, "{}", row)?;
            }
            StreamFormat::TsvStream => {
                let row = self
                    .variables
                    .iter()
                    .map(|v| tsv_escape(binding.get(v).unwrap_or("")))
                    .collect::<Vec<_>>()
                    .join("\t");
                writeln!(self.inner, "{}", row)?;
            }
            StreamFormat::JsonStream => {
                self.write_json_result(binding)?;
                return Ok(());
            }
        }
        self.row_count += 1;
        self.maybe_emit_heartbeat()?;
        Ok(())
    }

    /// Write a single binding as a NDJSON line
    pub fn write_json_result(&mut self, binding: &Binding) -> Result<()> {
        if !self.header_written {
            return Err(anyhow!(
                "write_header must be called before write_json_result"
            ));
        }
        let obj: serde_json::Map<String, serde_json::Value> = binding
            .values
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        let line = serde_json::to_string(&serde_json::Value::Object(obj))?;
        writeln!(self.inner, "{}", line)?;
        self.row_count += 1;
        self.maybe_emit_heartbeat()?;
        Ok(())
    }

    /// Flush the underlying writer
    pub fn flush(&mut self) -> Result<()> {
        self.inner.flush()?;
        Ok(())
    }

    /// Total rows written so far
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Emit a heartbeat comment if the interval has elapsed
    fn maybe_emit_heartbeat(&mut self) -> Result<()> {
        if self.heartbeat_interval_secs == 0 {
            return Ok(());
        }
        let elapsed = self.last_heartbeat.elapsed();
        if elapsed >= Duration::from_secs(self.heartbeat_interval_secs) {
            self.emit_heartbeat()?;
        }
        Ok(())
    }

    /// Emit a heartbeat comment immediately
    pub fn emit_heartbeat(&mut self) -> Result<()> {
        writeln!(self.inner, "#heartbeat")?;
        self.last_heartbeat = Instant::now();
        Ok(())
    }
}

/// Escape a value for CSV output
fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// Escape a value for TSV output (replace tabs and newlines)
fn tsv_escape(value: &str) -> String {
    value
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Command configuration for streaming query execution
#[derive(Debug, Clone)]
pub struct StreamingQueryCommand {
    /// SPARQL query text
    pub query: String,
    /// Output format
    pub format: StreamFormat,
    /// Streaming configuration
    pub config: StreamingQueryConfig,
}

impl StreamingQueryCommand {
    /// Create a new streaming query command
    pub fn new(
        query: impl Into<String>,
        format: StreamFormat,
        config: StreamingQueryConfig,
    ) -> Self {
        Self {
            query: query.into(),
            format,
            config,
        }
    }

    /// Execute streaming query against a pre-built result set (simulation)
    ///
    /// In production this would call the SPARQL engine. Here we accept
    /// a list of bindings to allow pure-logic testing without an engine.
    pub fn execute_with_results<W: Write>(
        &self,
        writer: W,
        variables: &[String],
        results: impl Iterator<Item = Binding>,
    ) -> Result<StreamStats> {
        let start = Instant::now();
        let mut out = StreamingResultWriter::new(writer, self.format, &self.config);
        out.write_header(variables)?;

        let deadline = start + Duration::from_millis(self.config.timeout_ms);
        let mut rows_written: usize = 0;

        for binding in results {
            // Timeout guard
            if Instant::now() >= deadline {
                break;
            }
            // Max rows guard
            if let Some(max) = self.config.max_rows {
                if rows_written >= max {
                    break;
                }
            }
            out.write_row(&binding)?;
            rows_written += 1;
        }

        out.flush()?;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(StreamStats {
            rows_written,
            elapsed_ms,
            timed_out: Instant::now() >= deadline,
            format: self.format,
        })
    }
}

/// Statistics returned after a streaming execution
#[derive(Debug, Clone, Serialize)]
pub struct StreamStats {
    pub rows_written: usize,
    pub elapsed_ms: u64,
    pub timed_out: bool,
    /// The format used for streaming (not serialised to JSON)
    #[serde(skip)]
    pub format: StreamFormat,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_binding(pairs: &[(&str, &str)]) -> Binding {
        Binding::from_pairs(pairs.iter().map(|(k, v)| (*k, *v)))
    }

    fn vars(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn config_no_heartbeat() -> StreamingQueryConfig {
        StreamingQueryConfig::default().with_heartbeat_interval(0)
    }

    // --- StreamFormat ---

    #[test]
    fn test_format_from_str_json_stream() {
        assert_eq!(
            StreamFormat::parse("json-stream").unwrap(),
            StreamFormat::JsonStream
        );
    }

    #[test]
    fn test_format_from_str_ndjson() {
        assert_eq!(
            StreamFormat::parse("ndjson").unwrap(),
            StreamFormat::JsonStream
        );
    }

    #[test]
    fn test_format_from_str_csv_stream() {
        assert_eq!(
            StreamFormat::parse("csv-stream").unwrap(),
            StreamFormat::CsvStream
        );
    }

    #[test]
    fn test_format_from_str_csv() {
        assert_eq!(StreamFormat::parse("csv").unwrap(), StreamFormat::CsvStream);
    }

    #[test]
    fn test_format_from_str_tsv_stream() {
        assert_eq!(
            StreamFormat::parse("tsv-stream").unwrap(),
            StreamFormat::TsvStream
        );
    }

    #[test]
    fn test_format_from_str_tsv() {
        assert_eq!(StreamFormat::parse("tsv").unwrap(), StreamFormat::TsvStream);
    }

    #[test]
    fn test_format_from_str_invalid() {
        assert!(StreamFormat::parse("xml").is_err());
    }

    #[test]
    fn test_format_mime_types() {
        assert_eq!(StreamFormat::JsonStream.mime_type(), "application/x-ndjson");
        assert_eq!(StreamFormat::CsvStream.mime_type(), "text/csv");
        assert_eq!(
            StreamFormat::TsvStream.mime_type(),
            "text/tab-separated-values"
        );
    }

    // --- StreamingQueryConfig ---

    #[test]
    fn test_config_defaults() {
        let cfg = StreamingQueryConfig::default();
        assert_eq!(cfg.page_size, 1000);
        assert_eq!(cfg.timeout_ms, 30_000);
        assert!(cfg.max_rows.is_none());
        assert_eq!(cfg.heartbeat_interval_secs, 5);
    }

    #[test]
    fn test_config_builder_methods() {
        let cfg = StreamingQueryConfig::default()
            .with_page_size(500)
            .with_timeout_ms(5000)
            .with_max_rows(100)
            .with_heartbeat_interval(10);
        assert_eq!(cfg.page_size, 500);
        assert_eq!(cfg.timeout_ms, 5000);
        assert_eq!(cfg.max_rows, Some(100));
        assert_eq!(cfg.heartbeat_interval_secs, 10);
    }

    // --- Binding ---

    #[test]
    fn test_binding_get() {
        let b = make_binding(&[("s", "http://example.org/a"), ("p", "rdf:type")]);
        assert_eq!(b.get("s"), Some("http://example.org/a"));
        assert_eq!(b.get("p"), Some("rdf:type"));
        assert_eq!(b.get("missing"), None);
    }

    #[test]
    fn test_binding_from_pairs() {
        let b = Binding::from_pairs([("x", "1"), ("y", "2")]);
        assert_eq!(b.get("x"), Some("1"));
        assert_eq!(b.get("y"), Some("2"));
    }

    // --- StreamingResultWriter CSV ---

    #[test]
    fn test_csv_header_row() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::CsvStream, &cfg);
        w.write_header(&vars(&["s", "p", "o"])).unwrap();
        w.flush().unwrap();
        let output = String::from_utf8(w.inner.into_inner().unwrap().into_inner()).unwrap();
        assert_eq!(output.trim(), "s,p,o");
    }

    #[test]
    fn test_csv_row_writing() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::CsvStream, &cfg);
        w.write_header(&vars(&["s", "p"])).unwrap();
        w.write_row(&make_binding(&[("s", "http://a"), ("p", "rdf:type")]))
            .unwrap();
        w.flush().unwrap();
        let out = String::from_utf8(w.inner.into_inner().unwrap().into_inner()).unwrap();
        let lines: Vec<&str> = out.trim().lines().collect();
        assert_eq!(lines[0], "s,p");
        assert_eq!(lines[1], "http://a,rdf:type");
    }

    #[test]
    fn test_csv_escaping_commas() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::CsvStream, &cfg);
        w.write_header(&vars(&["v"])).unwrap();
        w.write_row(&make_binding(&[("v", "hello, world")]))
            .unwrap();
        w.flush().unwrap();
        let out = String::from_utf8(w.inner.into_inner().unwrap().into_inner()).unwrap();
        assert!(out.contains("\"hello, world\""));
    }

    #[test]
    fn test_csv_escaping_quotes() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::CsvStream, &cfg);
        w.write_header(&vars(&["v"])).unwrap();
        w.write_row(&make_binding(&[("v", "say \"hello\"")]))
            .unwrap();
        w.flush().unwrap();
        let out = String::from_utf8(w.inner.into_inner().unwrap().into_inner()).unwrap();
        assert!(out.contains("\"say \"\"hello\"\"\""));
    }

    // --- StreamingResultWriter TSV ---

    #[test]
    fn test_tsv_header_row() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::TsvStream, &cfg);
        w.write_header(&vars(&["s", "p", "o"])).unwrap();
        w.flush().unwrap();
        let out = String::from_utf8(w.inner.into_inner().unwrap().into_inner()).unwrap();
        assert_eq!(out.trim(), "s\tp\to");
    }

    #[test]
    fn test_tsv_row_writing() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::TsvStream, &cfg);
        w.write_header(&vars(&["s", "p"])).unwrap();
        w.write_row(&make_binding(&[("s", "http://a"), ("p", "rdf:type")]))
            .unwrap();
        w.flush().unwrap();
        let out = String::from_utf8(w.inner.into_inner().unwrap().into_inner()).unwrap();
        let lines: Vec<&str> = out.trim().lines().collect();
        assert_eq!(lines[1], "http://a\trdf:type");
    }

    #[test]
    fn test_tsv_escaping_tab() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::TsvStream, &cfg);
        w.write_header(&vars(&["v"])).unwrap();
        w.write_row(&make_binding(&[("v", "a\tb")])).unwrap();
        w.flush().unwrap();
        let out = String::from_utf8(w.inner.into_inner().unwrap().into_inner()).unwrap();
        assert!(out.contains("a\\tb"));
    }

    // --- StreamingResultWriter JSON ---

    #[test]
    fn test_json_header_line() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::JsonStream, &cfg);
        w.write_header(&vars(&["s", "p"])).unwrap();
        w.flush().unwrap();
        let out = String::from_utf8(w.inner.into_inner().unwrap().into_inner()).unwrap();
        let hdr: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(hdr["type"], "header");
        assert!(hdr["vars"].is_array());
    }

    #[test]
    fn test_json_result_line() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::JsonStream, &cfg);
        w.write_header(&vars(&["s"])).unwrap();
        w.write_json_result(&make_binding(&[("s", "http://example.org")]))
            .unwrap();
        w.flush().unwrap();
        let out = String::from_utf8(w.inner.into_inner().unwrap().into_inner()).unwrap();
        let lines: Vec<&str> = out.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        let row: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(row["s"], "http://example.org");
    }

    #[test]
    fn test_row_count_increments() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::CsvStream, &cfg);
        w.write_header(&vars(&["v"])).unwrap();
        assert_eq!(w.row_count(), 0);
        w.write_row(&make_binding(&[("v", "a")])).unwrap();
        assert_eq!(w.row_count(), 1);
        w.write_row(&make_binding(&[("v", "b")])).unwrap();
        assert_eq!(w.row_count(), 2);
    }

    #[test]
    fn test_write_row_before_header_errors() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::CsvStream, &cfg);
        let result = w.write_row(&make_binding(&[("v", "a")]));
        assert!(result.is_err());
    }

    // --- Heartbeat ---

    #[test]
    fn test_heartbeat_emitted_manually() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::JsonStream, &cfg);
        w.write_header(&vars(&["v"])).unwrap();
        w.emit_heartbeat().unwrap();
        w.flush().unwrap();
        let out = String::from_utf8(w.inner.into_inner().unwrap().into_inner()).unwrap();
        assert!(out.contains("#heartbeat"));
    }

    // --- StreamingQueryCommand ---

    #[test]
    fn test_command_csv_execute() {
        let cfg = StreamingQueryConfig::default().with_heartbeat_interval(0);
        let cmd = StreamingQueryCommand::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            StreamFormat::CsvStream,
            cfg,
        );
        let bindings: Vec<Binding> = (0..5)
            .map(|i| make_binding(&[("s", &format!("http://example.org/{i}"))]))
            .collect();
        let mut out = Vec::new();
        let stats = cmd
            .execute_with_results(&mut out, &vars(&["s"]), bindings.into_iter())
            .unwrap();
        assert_eq!(stats.rows_written, 5);
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("s\n"));
        assert!(text.contains("http://example.org/0"));
    }

    #[test]
    fn test_command_max_rows_limit() {
        let cfg = StreamingQueryConfig::default()
            .with_heartbeat_interval(0)
            .with_max_rows(3);
        let cmd = StreamingQueryCommand::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            StreamFormat::CsvStream,
            cfg,
        );
        let bindings: Vec<Binding> = (0..10)
            .map(|i| make_binding(&[("s", &format!("http://example.org/{i}"))]))
            .collect();
        let mut out = Vec::new();
        let stats = cmd
            .execute_with_results(&mut out, &vars(&["s"]), bindings.into_iter())
            .unwrap();
        assert_eq!(stats.rows_written, 3);
    }

    #[test]
    fn test_command_ndjson_execute() {
        let cfg = StreamingQueryConfig::default().with_heartbeat_interval(0);
        let cmd = StreamingQueryCommand::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            StreamFormat::JsonStream,
            cfg,
        );
        let bindings: Vec<Binding> = (0..3)
            .map(|i| make_binding(&[("s", &format!("urn:{i}"))]))
            .collect();
        let mut out = Vec::new();
        let stats = cmd
            .execute_with_results(&mut out, &vars(&["s"]), bindings.into_iter())
            .unwrap();
        assert_eq!(stats.rows_written, 3);
        let text = String::from_utf8(out).unwrap();
        // First line should be header metadata
        let first: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(first["type"], "header");
    }

    #[test]
    fn test_command_tsv_execute() {
        let cfg = StreamingQueryConfig::default().with_heartbeat_interval(0);
        let cmd = StreamingQueryCommand::new(
            "SELECT ?s ?p WHERE { ?s ?p ?o }",
            StreamFormat::TsvStream,
            cfg,
        );
        let bindings = vec![make_binding(&[("s", "http://a"), ("p", "rdf:type")])];
        let mut out = Vec::new();
        let stats = cmd
            .execute_with_results(&mut out, &vars(&["s", "p"]), bindings.into_iter())
            .unwrap();
        assert_eq!(stats.rows_written, 1);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("s\tp\n"));
        assert!(text.contains("http://a\trdf:type"));
    }

    #[test]
    fn test_csv_missing_variable_emits_empty() {
        let buf: Vec<u8> = Vec::new();
        let cfg = config_no_heartbeat();
        let mut w = StreamingResultWriter::new(Cursor::new(buf), StreamFormat::CsvStream, &cfg);
        w.write_header(&vars(&["s", "label"])).unwrap();
        // "label" is not in binding
        w.write_row(&make_binding(&[("s", "http://a")])).unwrap();
        w.flush().unwrap();
        let out = String::from_utf8(w.inner.into_inner().unwrap().into_inner()).unwrap();
        let lines: Vec<&str> = out.trim().lines().collect();
        // Row should have two comma-separated values, second empty
        assert_eq!(lines[1], "http://a,");
    }

    #[test]
    fn test_stats_timed_out_flag() {
        // timeout_ms = 0 means deadline is in the past
        let cfg = StreamingQueryConfig::default()
            .with_timeout_ms(0)
            .with_heartbeat_interval(0);
        let cmd = StreamingQueryCommand::new(
            "SELECT ?s WHERE { ?s ?p ?o }",
            StreamFormat::CsvStream,
            cfg,
        );
        let bindings: Vec<Binding> = (0..5)
            .map(|i| make_binding(&[("s", &format!("urn:{i}"))]))
            .collect();
        let mut out = Vec::new();
        let stats = cmd
            .execute_with_results(&mut out, &vars(&["s"]), bindings.into_iter())
            .unwrap();
        // With timeout_ms=0 the deadline is already passed; 0 rows should be written
        assert_eq!(stats.rows_written, 0);
    }

    #[test]
    fn test_csv_escape_newline_in_value() {
        let escaped = csv_escape("line1\nline2");
        assert!(escaped.starts_with('"'));
        assert!(escaped.ends_with('"'));
    }
}
