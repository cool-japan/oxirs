//! SPARQL Result Serialization
//!
//! Implements serialization of SPARQL SELECT and ASK results into the four
//! standard output formats:
//!
//! - **JSON** — SPARQL 1.1 Query Results JSON Format (W3C REC)
//! - **XML**  — SPARQL Query Results XML Format (W3C REC)
//! - **CSV**  — comma-separated values with a header row
//! - **TSV**  — tab-separated values with `?variable` column headers
//!
//! Each format supports both SELECT result sets and ASK boolean results.

use std::fmt;

// ── Term representation ────────────────────────────────────────────────────────

/// A single RDF term that can appear in a query binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResultTerm {
    /// An IRI (named node).
    Iri(String),
    /// A plain literal (no datatype, no language tag).
    PlainLiteral(String),
    /// A language-tagged literal.
    LangLiteral {
        /// Lexical value.
        value: String,
        /// BCP-47 language tag.
        lang: String,
    },
    /// A typed literal.
    TypedLiteral {
        /// Lexical value.
        value: String,
        /// Datatype IRI.
        datatype: String,
    },
    /// A blank node.
    BlankNode(String),
}

impl ResultTerm {
    /// Construct a simple IRI term.
    pub fn iri(iri: impl Into<String>) -> Self {
        ResultTerm::Iri(iri.into())
    }

    /// Construct a plain literal term.
    pub fn plain(value: impl Into<String>) -> Self {
        ResultTerm::PlainLiteral(value.into())
    }

    /// Construct a language-tagged literal term.
    pub fn lang(value: impl Into<String>, lang: impl Into<String>) -> Self {
        ResultTerm::LangLiteral {
            value: value.into(),
            lang: lang.into(),
        }
    }

    /// Construct a typed literal term.
    pub fn typed(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        ResultTerm::TypedLiteral {
            value: value.into(),
            datatype: datatype.into(),
        }
    }

    /// Construct a blank-node term.
    pub fn blank(id: impl Into<String>) -> Self {
        ResultTerm::BlankNode(id.into())
    }

    /// Returns the lexical string value of this term.
    pub fn lexical_value(&self) -> &str {
        match self {
            ResultTerm::Iri(s) => s.as_str(),
            ResultTerm::PlainLiteral(s) => s.as_str(),
            ResultTerm::LangLiteral { value, .. } => value.as_str(),
            ResultTerm::TypedLiteral { value, .. } => value.as_str(),
            ResultTerm::BlankNode(s) => s.as_str(),
        }
    }

    /// Returns the term type label (`uri`, `literal`, `bnode`).
    pub fn term_type(&self) -> &str {
        match self {
            ResultTerm::Iri(_) => "uri",
            ResultTerm::BlankNode(_) => "bnode",
            _ => "literal",
        }
    }
}

// ── Result sets ───────────────────────────────────────────────────────────────

/// A single solution row: an ordered mapping from variable name to bound term.
///
/// Variables that are unbound in this solution map to `None`.
#[derive(Debug, Clone, Default)]
pub struct SolutionRow {
    /// (variable_name, bound_value) pairs in projection order.
    pub bindings: Vec<(String, Option<ResultTerm>)>,
}

impl SolutionRow {
    /// Create a new, empty solution row.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a variable to a term value.
    pub fn bind(&mut self, variable: impl Into<String>, term: ResultTerm) {
        self.bindings.push((variable.into(), Some(term)));
    }

    /// Mark a variable as unbound in this row.
    pub fn unbound(&mut self, variable: impl Into<String>) {
        self.bindings.push((variable.into(), None));
    }

    /// Look up the binding for `variable`.
    pub fn get(&self, variable: &str) -> Option<&ResultTerm> {
        self.bindings
            .iter()
            .find(|(v, _)| v == variable)
            .and_then(|(_, t)| t.as_ref())
    }
}

/// A SELECT result set: projection variables + rows of solution bindings.
#[derive(Debug, Clone)]
pub struct SelectResults {
    /// Projection variable names in order.
    pub variables: Vec<String>,
    /// Solution rows.
    pub rows: Vec<SolutionRow>,
}

impl SelectResults {
    /// Create an empty result set with the given projection variables.
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            variables,
            rows: Vec::new(),
        }
    }

    /// Append a solution row.
    pub fn add_row(&mut self, row: SolutionRow) {
        self.rows.push(row);
    }

    /// Number of result rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns `true` when there are no result rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

// ── Output format ─────────────────────────────────────────────────────────────

/// Supported SPARQL result serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparqlResultFormat {
    /// SPARQL 1.1 Query Results JSON Format.
    Json,
    /// SPARQL Query Results XML Format.
    Xml,
    /// Comma-separated values (RFC 4180 compatible).
    Csv,
    /// Tab-separated values with `?var` column headers.
    Tsv,
}

impl SparqlResultFormat {
    /// Returns the MIME type for this format.
    pub fn mime_type(self) -> &'static str {
        match self {
            SparqlResultFormat::Json => "application/sparql-results+json",
            SparqlResultFormat::Xml => "application/sparql-results+xml",
            SparqlResultFormat::Csv => "text/csv",
            SparqlResultFormat::Tsv => "text/tab-separated-values",
        }
    }

    /// Returns the canonical file extension for this format.
    pub fn extension(self) -> &'static str {
        match self {
            SparqlResultFormat::Json => "srj",
            SparqlResultFormat::Xml => "srx",
            SparqlResultFormat::Csv => "csv",
            SparqlResultFormat::Tsv => "tsv",
        }
    }

    /// Try to parse from a MIME type string (case-insensitive prefix match).
    pub fn from_mime(mime: &str) -> Option<Self> {
        let lower = mime.to_lowercase();
        if lower.contains("sparql-results+json") {
            Some(SparqlResultFormat::Json)
        } else if lower.contains("sparql-results+xml") {
            Some(SparqlResultFormat::Xml)
        } else if lower.contains("text/csv") {
            Some(SparqlResultFormat::Csv)
        } else if lower.contains("tab-separated") {
            Some(SparqlResultFormat::Tsv)
        } else {
            None
        }
    }
}

impl fmt::Display for SparqlResultFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.mime_type())
    }
}

// ── Formatter configuration ───────────────────────────────────────────────────

/// Configuration for the result formatter.
#[derive(Debug, Clone)]
pub struct FormatterConfig {
    /// Include the XML declaration (`<?xml …?>`) for XML output.
    pub xml_declaration: bool,
    /// Pretty-print JSON output (newlines and indentation).
    pub json_pretty: bool,
    /// Include a BOM at the start of CSV/TSV output.
    pub include_bom: bool,
    /// Line ending used in CSV/TSV (`\r\n` or `\n`).
    pub line_ending: LineEnding,
}

/// Line-ending style for CSV and TSV output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineEnding {
    /// Unix-style `\n`.
    Lf,
    /// Windows-style `\r\n` (recommended by RFC 4180 for CSV).
    CrLf,
}

impl LineEnding {
    fn as_str(self) -> &'static str {
        match self {
            LineEnding::Lf => "\n",
            LineEnding::CrLf => "\r\n",
        }
    }
}

impl Default for FormatterConfig {
    fn default() -> Self {
        Self {
            xml_declaration: true,
            json_pretty: false,
            include_bom: false,
            line_ending: LineEnding::Lf,
        }
    }
}

// ── Main formatter ────────────────────────────────────────────────────────────

/// SPARQL result formatter.
///
/// Serializes SELECT result sets and ASK boolean answers into the desired
/// format.
#[derive(Debug, Clone)]
pub struct ResultFormatter {
    config: FormatterConfig,
}

impl ResultFormatter {
    /// Create a formatter with the given configuration.
    pub fn new(config: FormatterConfig) -> Self {
        Self { config }
    }

    /// Create a formatter with default configuration.
    pub fn default_config() -> Self {
        Self::new(FormatterConfig::default())
    }

    // ── SELECT serialization ────────────────────────────────────────────────

    /// Serialize a SELECT result set into `format`.
    pub fn format_select(&self, results: &SelectResults, format: SparqlResultFormat) -> String {
        match format {
            SparqlResultFormat::Json => self.select_to_json(results),
            SparqlResultFormat::Xml => self.select_to_xml(results),
            SparqlResultFormat::Csv => self.select_to_csv(results),
            SparqlResultFormat::Tsv => self.select_to_tsv(results),
        }
    }

    /// Serialize an ASK boolean result into `format`.
    pub fn format_ask(&self, answer: bool, format: SparqlResultFormat) -> String {
        match format {
            SparqlResultFormat::Json => self.ask_to_json(answer),
            SparqlResultFormat::Xml => self.ask_to_xml(answer),
            SparqlResultFormat::Csv => self.ask_to_csv(answer),
            SparqlResultFormat::Tsv => self.ask_to_tsv(answer),
        }
    }

    // ── JSON ───────────────────────────────────────────────────────────────

    fn select_to_json(&self, results: &SelectResults) -> String {
        let nl = if self.config.json_pretty { "\n" } else { "" };
        let sp = if self.config.json_pretty { "  " } else { "" };
        let sp2 = if self.config.json_pretty { "    " } else { "" };
        let sp3 = if self.config.json_pretty {
            "      "
        } else {
            ""
        };
        let sp4 = if self.config.json_pretty {
            "        "
        } else {
            ""
        };

        // Build variable list
        let vars_str: String = results
            .variables
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let comma = if i + 1 < results.variables.len() {
                    ","
                } else {
                    ""
                };
                format!("{sp2}\"{v}\"{comma}")
            })
            .collect::<Vec<_>>()
            .join(nl);

        // Build bindings
        let bindings_str: String = results
            .rows
            .iter()
            .enumerate()
            .map(|(row_i, row)| {
                let binding_entries: String = row
                    .bindings
                    .iter()
                    .filter_map(|(var, term_opt)| term_opt.as_ref().map(|term| (var, term)))
                    .enumerate()
                    .map(|(i, (var, term))| {
                        let term_json = json_term(term, sp4);
                        let is_last =
                            i + 1 >= row.bindings.iter().filter(|(_, t)| t.is_some()).count();
                        let comma = if is_last { "" } else { "," };
                        if self.config.json_pretty {
                            format!("{sp4}\"{var}\": {term_json}{comma}")
                        } else {
                            format!("\"{var}\":{term_json}{comma}")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(nl);

                let row_comma = if row_i + 1 < results.rows.len() {
                    ","
                } else {
                    ""
                };
                if self.config.json_pretty {
                    format!("{sp3}{{{nl}{binding_entries}{nl}{sp3}}}{row_comma}")
                } else {
                    format!("{{{binding_entries}}}{row_comma}")
                }
            })
            .collect::<Vec<_>>()
            .join(nl);

        if self.config.json_pretty {
            format!(
                "{{{nl}\
                 {sp}\"head\": {{{nl}\
                 {sp2}\"vars\": [{nl}\
                 {vars_str}{nl}\
                 {sp2}]{nl}\
                 {sp}}},{nl}\
                 {sp}\"results\": {{{nl}\
                 {sp2}\"bindings\": [{nl}\
                 {bindings_str}{nl}\
                 {sp2}]{nl}\
                 {sp}}}{nl}\
                 }}"
            )
        } else {
            format!(
                "{{\"head\":{{\"vars\":[{vars_str}]}},\"results\":{{\"bindings\":[{bindings_str}]}}}}"
            )
        }
    }

    fn ask_to_json(&self, answer: bool) -> String {
        let val = if answer { "true" } else { "false" };
        if self.config.json_pretty {
            format!("{{\n  \"head\": {{}},\n  \"boolean\": {val}\n}}")
        } else {
            format!("{{\"head\":{{}},\"boolean\":{val}}}")
        }
    }

    // ── XML ────────────────────────────────────────────────────────────────

    fn select_to_xml(&self, results: &SelectResults) -> String {
        let mut out = String::new();

        if self.config.xml_declaration {
            out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        }

        out.push_str("<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">\n  <head>\n");

        for var in &results.variables {
            out.push_str(&format!("    <variable name=\"{var}\"/>\n"));
        }

        out.push_str("  </head>\n  <results>\n");

        for row in &results.rows {
            out.push_str("    <result>\n");
            for (var, term_opt) in &row.bindings {
                if let Some(term) = term_opt {
                    out.push_str(&format!("      <binding name=\"{var}\">\n"));
                    out.push_str(&xml_term(term));
                    out.push_str("      </binding>\n");
                }
            }
            out.push_str("    </result>\n");
        }

        out.push_str("  </results>\n</sparql>");
        out
    }

    fn ask_to_xml(&self, answer: bool) -> String {
        let val = if answer { "true" } else { "false" };
        let mut out = String::new();
        if self.config.xml_declaration {
            out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        }
        out.push_str(&format!(
            "<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">\n  <head/>\n  <boolean>{val}</boolean>\n</sparql>"
        ));
        out
    }

    // ── CSV ────────────────────────────────────────────────────────────────

    fn select_to_csv(&self, results: &SelectResults) -> String {
        let le = self.config.line_ending.as_str();
        let mut out = String::new();

        if self.config.include_bom {
            out.push('\u{FEFF}');
        }

        // Header row
        out.push_str(
            &results
                .variables
                .iter()
                .map(|v| csv_escape(v))
                .collect::<Vec<_>>()
                .join(","),
        );
        out.push_str(le);

        // Data rows
        for row in &results.rows {
            let cells: Vec<String> = results
                .variables
                .iter()
                .map(|var| {
                    if let Some(term) = row.get(var) {
                        csv_escape(csv_term_value(term))
                    } else {
                        String::new()
                    }
                })
                .collect();
            out.push_str(&cells.join(","));
            out.push_str(le);
        }
        out
    }

    fn ask_to_csv(&self, answer: bool) -> String {
        let le = self.config.line_ending.as_str();
        let val = if answer { "true" } else { "false" };
        format!("_askResult{le}{val}{le}")
    }

    // ── TSV ────────────────────────────────────────────────────────────────

    fn select_to_tsv(&self, results: &SelectResults) -> String {
        let le = self.config.line_ending.as_str();
        let mut out = String::new();

        if self.config.include_bom {
            out.push('\u{FEFF}');
        }

        // Header row: ?var1 \t ?var2 …
        out.push_str(
            &results
                .variables
                .iter()
                .map(|v| format!("?{v}"))
                .collect::<Vec<_>>()
                .join("\t"),
        );
        out.push_str(le);

        // Data rows
        for row in &results.rows {
            let cells: Vec<String> = results
                .variables
                .iter()
                .map(|var| {
                    if let Some(term) = row.get(var) {
                        tsv_term(term)
                    } else {
                        String::new()
                    }
                })
                .collect();
            out.push_str(&cells.join("\t"));
            out.push_str(le);
        }
        out
    }

    fn ask_to_tsv(&self, answer: bool) -> String {
        let le = self.config.line_ending.as_str();
        let val = if answer { "true" } else { "false" };
        format!("?_askResult{le}{val}{le}")
    }
}

impl Default for ResultFormatter {
    fn default() -> Self {
        Self::default_config()
    }
}

// ── Helper serialization functions ────────────────────────────────────────────

/// Serialize a term to SPARQL Results JSON object.
fn json_term(term: &ResultTerm, _indent: &str) -> String {
    match term {
        ResultTerm::Iri(iri) => {
            format!("{{\"type\":\"uri\",\"value\":{}}}", json_string(iri))
        }
        ResultTerm::BlankNode(id) => {
            format!("{{\"type\":\"bnode\",\"value\":{}}}", json_string(id))
        }
        ResultTerm::PlainLiteral(val) => {
            format!("{{\"type\":\"literal\",\"value\":{}}}", json_string(val))
        }
        ResultTerm::LangLiteral { value, lang } => {
            format!(
                "{{\"type\":\"literal\",\"xml:lang\":{},\"value\":{}}}",
                json_string(lang),
                json_string(value)
            )
        }
        ResultTerm::TypedLiteral { value, datatype } => {
            format!(
                "{{\"type\":\"literal\",\"datatype\":{},\"value\":{}}}",
                json_string(datatype),
                json_string(value)
            )
        }
    }
}

/// Escape a string as a JSON double-quoted string.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x00'..='\x1f' => {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            }
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// Serialize a term to SPARQL Results XML element content.
fn xml_term(term: &ResultTerm) -> String {
    match term {
        ResultTerm::Iri(iri) => {
            format!("        <uri>{}</uri>\n", xml_escape(iri))
        }
        ResultTerm::BlankNode(id) => {
            format!("        <bnode>{}</bnode>\n", xml_escape(id))
        }
        ResultTerm::PlainLiteral(val) => {
            format!("        <literal>{}</literal>\n", xml_escape(val))
        }
        ResultTerm::LangLiteral { value, lang } => {
            format!(
                "        <literal xml:lang=\"{}\">{}</literal>\n",
                xml_escape(lang),
                xml_escape(value)
            )
        }
        ResultTerm::TypedLiteral { value, datatype } => {
            format!(
                "        <literal datatype=\"{}\">{}</literal>\n",
                xml_escape(datatype),
                xml_escape(value)
            )
        }
    }
}

/// Escape XML special characters in element content / attribute values.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

/// Return the plain text value used in CSV/TSV rendering of a term.
fn csv_term_value(term: &ResultTerm) -> &str {
    match term {
        ResultTerm::Iri(s) => s.as_str(),
        ResultTerm::PlainLiteral(s) => s.as_str(),
        ResultTerm::LangLiteral { value, .. } => value.as_str(),
        ResultTerm::TypedLiteral { value, .. } => value.as_str(),
        ResultTerm::BlankNode(s) => s.as_str(),
    }
}

/// CSV-escape a field value: wrap in double quotes if it contains commas,
/// double-quotes, or newlines; escape embedded double-quotes by doubling.
fn csv_escape(s: &str) -> String {
    let needs_quoting = s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r');
    if needs_quoting {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for ch in s.chars() {
            if ch == '"' {
                out.push('"');
            }
            out.push(ch);
        }
        out.push('"');
        out
    } else {
        s.to_string()
    }
}

/// Serialize a term to its TSV representation (N-Triples-style).
fn tsv_term(term: &ResultTerm) -> String {
    match term {
        ResultTerm::Iri(iri) => format!("<{iri}>"),
        ResultTerm::BlankNode(id) => format!("_:{id}"),
        ResultTerm::PlainLiteral(val) => {
            format!("\"{}\"", tsv_escape_literal(val))
        }
        ResultTerm::LangLiteral { value, lang } => {
            format!("\"{}\"@{lang}", tsv_escape_literal(value))
        }
        ResultTerm::TypedLiteral { value, datatype } => {
            format!("\"{}\"^^<{datatype}>", tsv_escape_literal(value))
        }
    }
}

/// Escape special characters inside a TSV/N-Triples literal.
fn tsv_escape_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn fmt() -> ResultFormatter {
        ResultFormatter::default_config()
    }

    fn fmt_pretty() -> ResultFormatter {
        ResultFormatter::new(FormatterConfig {
            json_pretty: true,
            ..FormatterConfig::default()
        })
    }

    fn simple_results() -> SelectResults {
        let mut results = SelectResults::new(vec!["s".to_string(), "p".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("s", ResultTerm::iri("http://example.org/Alice"));
        row.bind("p", ResultTerm::plain("Alice"));
        results.add_row(row);
        results
    }

    fn two_row_results() -> SelectResults {
        let mut results = SelectResults::new(vec!["name".to_string(), "age".to_string()]);
        let mut r1 = SolutionRow::new();
        r1.bind("name", ResultTerm::plain("Alice"));
        r1.bind(
            "age",
            ResultTerm::typed("30", "http://www.w3.org/2001/XMLSchema#integer"),
        );
        results.add_row(r1);

        let mut r2 = SolutionRow::new();
        r2.bind("name", ResultTerm::plain("Bob"));
        r2.unbound("age");
        results.add_row(r2);
        results
    }

    // ── SparqlResultFormat ───────────────────────────────────────────────────

    #[test]
    fn test_format_mime_types() {
        assert_eq!(
            SparqlResultFormat::Json.mime_type(),
            "application/sparql-results+json"
        );
        assert_eq!(
            SparqlResultFormat::Xml.mime_type(),
            "application/sparql-results+xml"
        );
        assert_eq!(SparqlResultFormat::Csv.mime_type(), "text/csv");
        assert!(SparqlResultFormat::Tsv
            .mime_type()
            .contains("tab-separated"));
    }

    #[test]
    fn test_format_extensions() {
        assert_eq!(SparqlResultFormat::Json.extension(), "srj");
        assert_eq!(SparqlResultFormat::Xml.extension(), "srx");
        assert_eq!(SparqlResultFormat::Csv.extension(), "csv");
        assert_eq!(SparqlResultFormat::Tsv.extension(), "tsv");
    }

    #[test]
    fn test_format_from_mime_json() {
        let f = SparqlResultFormat::from_mime("application/sparql-results+json");
        assert_eq!(f, Some(SparqlResultFormat::Json));
    }

    #[test]
    fn test_format_from_mime_xml() {
        let f = SparqlResultFormat::from_mime("application/sparql-results+xml");
        assert_eq!(f, Some(SparqlResultFormat::Xml));
    }

    #[test]
    fn test_format_from_mime_csv() {
        assert_eq!(
            SparqlResultFormat::from_mime("text/csv"),
            Some(SparqlResultFormat::Csv)
        );
    }

    #[test]
    fn test_format_from_mime_tsv() {
        assert_eq!(
            SparqlResultFormat::from_mime("text/tab-separated-values"),
            Some(SparqlResultFormat::Tsv)
        );
    }

    #[test]
    fn test_format_from_mime_unknown() {
        assert!(SparqlResultFormat::from_mime("text/plain").is_none());
    }

    #[test]
    fn test_format_display() {
        let s = format!("{}", SparqlResultFormat::Json);
        assert!(s.contains("json"));
    }

    // ── ResultTerm ───────────────────────────────────────────────────────────

    #[test]
    fn test_term_iri_type() {
        let t = ResultTerm::iri("http://example.org");
        assert_eq!(t.term_type(), "uri");
        assert_eq!(t.lexical_value(), "http://example.org");
    }

    #[test]
    fn test_term_plain_type() {
        let t = ResultTerm::plain("hello");
        assert_eq!(t.term_type(), "literal");
        assert_eq!(t.lexical_value(), "hello");
    }

    #[test]
    fn test_term_lang_type() {
        let t = ResultTerm::lang("Hola", "es");
        assert_eq!(t.term_type(), "literal");
        assert_eq!(t.lexical_value(), "Hola");
    }

    #[test]
    fn test_term_typed_type() {
        let t = ResultTerm::typed("42", "http://www.w3.org/2001/XMLSchema#integer");
        assert_eq!(t.term_type(), "literal");
        assert_eq!(t.lexical_value(), "42");
    }

    #[test]
    fn test_term_blank_type() {
        let t = ResultTerm::blank("b1");
        assert_eq!(t.term_type(), "bnode");
        assert_eq!(t.lexical_value(), "b1");
    }

    // ── SolutionRow ──────────────────────────────────────────────────────────

    #[test]
    fn test_solution_row_bind_and_get() {
        let mut row = SolutionRow::new();
        row.bind("x", ResultTerm::plain("hello"));
        assert_eq!(row.get("x"), Some(&ResultTerm::plain("hello")));
        assert!(row.get("y").is_none());
    }

    #[test]
    fn test_solution_row_unbound() {
        let mut row = SolutionRow::new();
        row.unbound("x");
        assert!(row.get("x").is_none());
    }

    // ── SelectResults ────────────────────────────────────────────────────────

    #[test]
    fn test_select_results_empty() {
        let r = SelectResults::new(vec!["s".to_string()]);
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn test_select_results_add_row() {
        let mut r = SelectResults::new(vec!["s".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("s", ResultTerm::iri("http://example.org"));
        r.add_row(row);
        assert_eq!(r.len(), 1);
        assert!(!r.is_empty());
    }

    // ── JSON SELECT ──────────────────────────────────────────────────────────

    #[test]
    fn test_json_select_contains_head() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Json);
        assert!(out.contains("\"head\""));
        assert!(out.contains("\"vars\""));
    }

    #[test]
    fn test_json_select_contains_variables() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Json);
        assert!(out.contains("\"s\""));
        assert!(out.contains("\"p\""));
    }

    #[test]
    fn test_json_select_contains_bindings() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Json);
        assert!(out.contains("\"bindings\""));
        assert!(out.contains("http://example.org/Alice"));
        assert!(out.contains("Alice"));
    }

    #[test]
    fn test_json_select_uri_type() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Json);
        assert!(out.contains("\"type\":\"uri\""));
    }

    #[test]
    fn test_json_select_literal_type() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Json);
        assert!(out.contains("\"type\":\"literal\""));
    }

    #[test]
    fn test_json_select_lang_literal() {
        let mut results = SelectResults::new(vec!["label".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("label", ResultTerm::lang("Hello", "en"));
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Json);
        assert!(out.contains("\"xml:lang\""));
        assert!(out.contains("\"en\""));
    }

    #[test]
    fn test_json_select_typed_literal() {
        let mut results = SelectResults::new(vec!["n".to_string()]);
        let mut row = SolutionRow::new();
        row.bind(
            "n",
            ResultTerm::typed("42", "http://www.w3.org/2001/XMLSchema#integer"),
        );
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Json);
        assert!(out.contains("\"datatype\""));
        assert!(out.contains("XMLSchema#integer"));
    }

    #[test]
    fn test_json_select_bnode() {
        let mut results = SelectResults::new(vec!["b".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("b", ResultTerm::blank("b42"));
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Json);
        assert!(out.contains("\"type\":\"bnode\""));
        assert!(out.contains("b42"));
    }

    #[test]
    fn test_json_select_unbound_variable_omitted() {
        let results = two_row_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Json);
        // "age" is unbound in the second row; the binding entry should not appear
        // for Bob's row but should for Alice.
        assert!(out.contains("XMLSchema#integer")); // Alice has age
                                                    // Bob's row has no age binding
        let bob_section = out.find("Bob").expect("Bob should be present");
        let bob_part = &out[bob_section..];
        // The result section after "Bob" should not include "XMLSchema" again
        assert!(!bob_part.contains("XMLSchema"));
    }

    #[test]
    fn test_json_select_empty_results() {
        let results = SelectResults::new(vec!["x".to_string(), "y".to_string()]);
        let out = fmt().format_select(&results, SparqlResultFormat::Json);
        assert!(out.contains("\"bindings\":["));
        // empty bindings
        let bi = out.find("\"bindings\":[").expect("has bindings");
        let after = &out[bi + "\"bindings\":[".len()..];
        assert!(after.starts_with(']'));
    }

    #[test]
    fn test_json_select_multiple_rows() {
        let results = two_row_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Json);
        assert!(out.contains("Alice"));
        assert!(out.contains("Bob"));
    }

    #[test]
    fn test_json_pretty_select() {
        let results = simple_results();
        let out = fmt_pretty().format_select(&results, SparqlResultFormat::Json);
        assert!(out.contains('\n'));
        assert!(out.contains("  \"head\""));
    }

    #[test]
    fn test_json_special_chars_in_literal() {
        let mut results = SelectResults::new(vec!["v".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("v", ResultTerm::plain("He said \"hello\"\nworld"));
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Json);
        assert!(out.contains("\\\"hello\\\""));
        assert!(out.contains("\\n"));
    }

    // ── JSON ASK ─────────────────────────────────────────────────────────────

    #[test]
    fn test_json_ask_true() {
        let out = fmt().format_ask(true, SparqlResultFormat::Json);
        assert!(out.contains("\"boolean\":true"));
        assert!(out.contains("\"head\""));
    }

    #[test]
    fn test_json_ask_false() {
        let out = fmt().format_ask(false, SparqlResultFormat::Json);
        assert!(out.contains("\"boolean\":false"));
    }

    #[test]
    fn test_json_ask_pretty_true() {
        let out = fmt_pretty().format_ask(true, SparqlResultFormat::Json);
        assert!(out.contains('\n'));
        assert!(out.contains("true"));
    }

    // ── XML SELECT ───────────────────────────────────────────────────────────

    #[test]
    fn test_xml_select_declaration() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Xml);
        assert!(out.starts_with("<?xml version=\"1.0\""));
    }

    #[test]
    fn test_xml_select_no_declaration() {
        let fmt_no_decl = ResultFormatter::new(FormatterConfig {
            xml_declaration: false,
            ..FormatterConfig::default()
        });
        let results = simple_results();
        let out = fmt_no_decl.format_select(&results, SparqlResultFormat::Xml);
        assert!(!out.starts_with("<?xml"));
        assert!(out.starts_with("<sparql"));
    }

    #[test]
    fn test_xml_select_namespace() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Xml);
        assert!(out.contains("http://www.w3.org/2005/sparql-results#"));
    }

    #[test]
    fn test_xml_select_variable_elements() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Xml);
        assert!(out.contains("<variable name=\"s\"/>"));
        assert!(out.contains("<variable name=\"p\"/>"));
    }

    #[test]
    fn test_xml_select_result_element() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Xml);
        assert!(out.contains("<result>"));
        assert!(out.contains("</result>"));
    }

    #[test]
    fn test_xml_select_binding_element() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Xml);
        assert!(out.contains("<binding name=\"s\">"));
        assert!(out.contains("<uri>http://example.org/Alice</uri>"));
    }

    #[test]
    fn test_xml_select_literal_element() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Xml);
        assert!(out.contains("<literal>Alice</literal>"));
    }

    #[test]
    fn test_xml_select_lang_literal() {
        let mut results = SelectResults::new(vec!["label".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("label", ResultTerm::lang("Bonjour", "fr"));
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Xml);
        assert!(out.contains("xml:lang=\"fr\""));
        assert!(out.contains("Bonjour"));
    }

    #[test]
    fn test_xml_select_typed_literal() {
        let mut results = SelectResults::new(vec!["n".to_string()]);
        let mut row = SolutionRow::new();
        row.bind(
            "n",
            ResultTerm::typed("3.14", "http://www.w3.org/2001/XMLSchema#decimal"),
        );
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Xml);
        assert!(out.contains("datatype=\"http://www.w3.org/2001/XMLSchema#decimal\""));
        assert!(out.contains("3.14"));
    }

    #[test]
    fn test_xml_select_bnode() {
        let mut results = SelectResults::new(vec!["b".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("b", ResultTerm::blank("b0"));
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Xml);
        assert!(out.contains("<bnode>b0</bnode>"));
    }

    #[test]
    fn test_xml_escape_special_chars() {
        let mut results = SelectResults::new(vec!["v".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("v", ResultTerm::plain("a & b < c > d"));
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Xml);
        assert!(out.contains("a &amp; b &lt; c &gt; d"));
    }

    #[test]
    fn test_xml_select_unbound_skipped() {
        let results = two_row_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Xml);
        // Bob has no age binding — the result should still be valid XML with two <result> blocks
        assert_eq!(out.matches("<result>").count(), 2);
    }

    // ── XML ASK ──────────────────────────────────────────────────────────────

    #[test]
    fn test_xml_ask_true() {
        let out = fmt().format_ask(true, SparqlResultFormat::Xml);
        assert!(out.contains("<boolean>true</boolean>"));
        assert!(out.contains("<head/>"));
    }

    #[test]
    fn test_xml_ask_false() {
        let out = fmt().format_ask(false, SparqlResultFormat::Xml);
        assert!(out.contains("<boolean>false</boolean>"));
    }

    // ── CSV SELECT ───────────────────────────────────────────────────────────

    #[test]
    fn test_csv_select_header_row() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Csv);
        let first_line = out.lines().next().expect("has header");
        assert_eq!(first_line, "s,p");
    }

    #[test]
    fn test_csv_select_data_row() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Csv);
        let second_line: Vec<&str> = out.lines().collect();
        assert_eq!(second_line[1], "http://example.org/Alice,Alice");
    }

    #[test]
    fn test_csv_select_unbound_empty_cell() {
        let results = two_row_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Csv);
        let lines: Vec<&str> = out.lines().collect();
        // Bob's row: name=Bob, age=unbound → last cell is empty
        assert_eq!(lines[2], "Bob,");
    }

    #[test]
    fn test_csv_select_quoting_comma() {
        let mut results = SelectResults::new(vec!["v".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("v", ResultTerm::plain("one,two"));
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Csv);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[1], "\"one,two\"");
    }

    #[test]
    fn test_csv_select_quoting_double_quote() {
        let mut results = SelectResults::new(vec!["v".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("v", ResultTerm::plain("say \"hi\""));
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Csv);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[1], "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn test_csv_select_no_bom_by_default() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Csv);
        assert!(!out.starts_with('\u{FEFF}'));
    }

    #[test]
    fn test_csv_select_bom_when_enabled() {
        let fmt_bom = ResultFormatter::new(FormatterConfig {
            include_bom: true,
            ..FormatterConfig::default()
        });
        let results = simple_results();
        let out = fmt_bom.format_select(&results, SparqlResultFormat::Csv);
        assert!(out.starts_with('\u{FEFF}'));
    }

    #[test]
    fn test_csv_select_crlf_line_ending() {
        let fmt_crlf = ResultFormatter::new(FormatterConfig {
            line_ending: LineEnding::CrLf,
            ..FormatterConfig::default()
        });
        let results = simple_results();
        let out = fmt_crlf.format_select(&results, SparqlResultFormat::Csv);
        assert!(out.contains("\r\n"));
    }

    #[test]
    fn test_csv_select_empty() {
        let results = SelectResults::new(vec!["x".to_string()]);
        let out = fmt().format_select(&results, SparqlResultFormat::Csv);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 1); // only header
        assert_eq!(lines[0], "x");
    }

    // ── CSV ASK ──────────────────────────────────────────────────────────────

    #[test]
    fn test_csv_ask_true() {
        let out = fmt().format_ask(true, SparqlResultFormat::Csv);
        assert!(out.contains("true"));
    }

    #[test]
    fn test_csv_ask_false() {
        let out = fmt().format_ask(false, SparqlResultFormat::Csv);
        assert!(out.contains("false"));
    }

    // ── TSV SELECT ───────────────────────────────────────────────────────────

    #[test]
    fn test_tsv_select_header_with_question_marks() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Tsv);
        let first_line = out.lines().next().expect("has header");
        assert_eq!(first_line, "?s\t?p");
    }

    #[test]
    fn test_tsv_select_iri_term() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Tsv);
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines[1].starts_with("<http://example.org/Alice>"));
    }

    #[test]
    fn test_tsv_select_literal_term() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Tsv);
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines[1].ends_with("\"Alice\""));
    }

    #[test]
    fn test_tsv_select_lang_literal() {
        let mut results = SelectResults::new(vec!["label".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("label", ResultTerm::lang("Hola", "es"));
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Tsv);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[1], "\"Hola\"@es");
    }

    #[test]
    fn test_tsv_select_typed_literal() {
        let mut results = SelectResults::new(vec!["n".to_string()]);
        let mut row = SolutionRow::new();
        row.bind(
            "n",
            ResultTerm::typed("42", "http://www.w3.org/2001/XMLSchema#integer"),
        );
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Tsv);
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines[1].contains("^^<http://www.w3.org/2001/XMLSchema#integer>"));
    }

    #[test]
    fn test_tsv_select_bnode() {
        let mut results = SelectResults::new(vec!["b".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("b", ResultTerm::blank("b0"));
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Tsv);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[1], "_:b0");
    }

    #[test]
    fn test_tsv_select_unbound_empty() {
        let results = two_row_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Tsv);
        let lines: Vec<&str> = out.lines().collect();
        // Bob's row: name tab empty
        assert!(lines[2].ends_with('\t'));
    }

    #[test]
    fn test_tsv_select_tabs_between_columns() {
        let results = simple_results();
        let out = fmt().format_select(&results, SparqlResultFormat::Tsv);
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines[1].contains('\t'));
    }

    #[test]
    fn test_tsv_select_literal_escape() {
        let mut results = SelectResults::new(vec!["v".to_string()]);
        let mut row = SolutionRow::new();
        row.bind("v", ResultTerm::plain("line1\nline2"));
        results.add_row(row);
        let out = fmt().format_select(&results, SparqlResultFormat::Tsv);
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines[1].contains("\\n"));
    }

    // ── TSV ASK ──────────────────────────────────────────────────────────────

    #[test]
    fn test_tsv_ask_true() {
        let out = fmt().format_ask(true, SparqlResultFormat::Tsv);
        assert!(out.contains("true"));
        assert!(out.contains('?'));
    }

    #[test]
    fn test_tsv_ask_false() {
        let out = fmt().format_ask(false, SparqlResultFormat::Tsv);
        assert!(out.contains("false"));
    }

    // ── Helper function unit tests ───────────────────────────────────────────

    #[test]
    fn test_json_string_escapes_backslash() {
        let s = json_string("a\\b");
        assert!(s.contains("\\\\"));
    }

    #[test]
    fn test_json_string_escapes_newline() {
        let s = json_string("a\nb");
        assert!(s.contains("\\n"));
    }

    #[test]
    fn test_json_string_escapes_tab() {
        let s = json_string("a\tb");
        assert!(s.contains("\\t"));
    }

    #[test]
    fn test_xml_escape_ampersand() {
        assert_eq!(xml_escape("a&b"), "a&amp;b");
    }

    #[test]
    fn test_xml_escape_lt_gt() {
        assert_eq!(xml_escape("a<b>c"), "a&lt;b&gt;c");
    }

    #[test]
    fn test_xml_escape_quotes() {
        assert_eq!(xml_escape("\"'"), "&quot;&apos;");
    }

    #[test]
    fn test_csv_escape_plain() {
        assert_eq!(csv_escape("hello"), "hello");
    }

    #[test]
    fn test_csv_escape_with_comma() {
        let s = csv_escape("a,b");
        assert!(s.starts_with('"') && s.ends_with('"'));
    }

    #[test]
    fn test_csv_escape_with_newline() {
        let s = csv_escape("a\nb");
        assert!(s.starts_with('"'));
    }

    #[test]
    fn test_tsv_escape_literal_quotes() {
        assert!(tsv_escape_literal("say \"hi\"").contains("\\\""));
    }

    #[test]
    fn test_tsv_escape_literal_backslash() {
        assert!(tsv_escape_literal("a\\b").contains("\\\\"));
    }

    // ── Line ending ──────────────────────────────────────────────────────────

    #[test]
    fn test_line_ending_lf() {
        assert_eq!(LineEnding::Lf.as_str(), "\n");
    }

    #[test]
    fn test_line_ending_crlf() {
        assert_eq!(LineEnding::CrLf.as_str(), "\r\n");
    }

    // ── FormatterConfig default ──────────────────────────────────────────────

    #[test]
    fn test_formatter_config_default() {
        let cfg = FormatterConfig::default();
        assert!(cfg.xml_declaration);
        assert!(!cfg.json_pretty);
        assert!(!cfg.include_bom);
    }
}
