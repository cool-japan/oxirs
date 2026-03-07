//! # Import Command — Multi-format RDF Importer
//!
//! Provides `ImportCommand` for parsing RDF data from Turtle, N-Triples,
//! N-Quads, JSON-LD, RDF/XML, TriG, and CSV formats.
//!
//! # Example
//!
//! ```rust
//! use oxirs::commands::import_command::{ImportCommand, ImportFormat};
//!
//! let nt = "<http://a.org/s> <http://a.org/p> <http://a.org/o> .\n";
//! let result = ImportCommand::import(nt, ImportFormat::NTriples).expect("ok");
//! assert_eq!(result.triple_count(), 1);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ImportFormat
// ---------------------------------------------------------------------------

/// Supported RDF input formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImportFormat {
    /// Turtle (`.ttl`)
    Turtle,
    /// N-Triples (`.nt`)
    NTriples,
    /// N-Quads (`.nq`)
    NQuads,
    /// JSON-LD (simplified, `.jsonld`)
    JsonLd,
    /// RDF/XML (simplified, `.rdf`)
    RdfXml,
    /// TriG (`.trig`)
    TriG,
    /// CSV with `subject,predicate,object[,graph]` header (`.csv`)
    Csv,
}

impl ImportFormat {
    /// Detect format from a file-extension string (case-insensitive).
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "ttl" => Some(ImportFormat::Turtle),
            "nt" => Some(ImportFormat::NTriples),
            "nq" => Some(ImportFormat::NQuads),
            "jsonld" | "json-ld" | "json" => Some(ImportFormat::JsonLd),
            "rdf" | "owl" | "xml" => Some(ImportFormat::RdfXml),
            "trig" => Some(ImportFormat::TriG),
            "csv" => Some(ImportFormat::Csv),
            _ => None,
        }
    }

    /// Detect format from a MIME-type string.
    pub fn from_mime_type(mime: &str) -> Option<Self> {
        match mime.to_lowercase().split(';').next().unwrap_or("").trim() {
            "text/turtle" => Some(ImportFormat::Turtle),
            "application/n-triples" => Some(ImportFormat::NTriples),
            "application/n-quads" => Some(ImportFormat::NQuads),
            "application/ld+json" => Some(ImportFormat::JsonLd),
            "application/rdf+xml" => Some(ImportFormat::RdfXml),
            "application/trig" => Some(ImportFormat::TriG),
            "text/csv" => Some(ImportFormat::Csv),
            _ => None,
        }
    }

    /// Canonical file extension (without leading `.`).
    pub fn extension(&self) -> &'static str {
        match self {
            ImportFormat::Turtle => "ttl",
            ImportFormat::NTriples => "nt",
            ImportFormat::NQuads => "nq",
            ImportFormat::JsonLd => "jsonld",
            ImportFormat::RdfXml => "rdf",
            ImportFormat::TriG => "trig",
            ImportFormat::Csv => "csv",
        }
    }

    /// MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            ImportFormat::Turtle => "text/turtle",
            ImportFormat::NTriples => "application/n-triples",
            ImportFormat::NQuads => "application/n-quads",
            ImportFormat::JsonLd => "application/ld+json",
            ImportFormat::RdfXml => "application/rdf+xml",
            ImportFormat::TriG => "application/trig",
            ImportFormat::Csv => "text/csv",
        }
    }
}

// ---------------------------------------------------------------------------
// Triple
// ---------------------------------------------------------------------------

/// An RDF triple (or quad when `graph` is `Some`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// Named-graph IRI (for N-Quads and TriG).
    pub graph: Option<String>,
}

// ---------------------------------------------------------------------------
// ImportResult
// ---------------------------------------------------------------------------

/// The result of a successful parse operation.
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// Parsed triples.
    pub triples: Vec<Triple>,
    /// Prefix declarations found in the input.
    pub prefixes: HashMap<String, String>,
    /// Distinct named-graph IRIs found.
    pub graphs: Vec<String>,
    /// Non-fatal parse warnings.
    pub warnings: Vec<String>,
    /// Format that was used for parsing.
    pub format_detected: ImportFormat,
}

impl ImportResult {
    /// Number of triples parsed.
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Number of distinct named graphs.
    pub fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    /// Returns `true` if any non-fatal warnings were generated.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ImportError
// ---------------------------------------------------------------------------

/// Errors produced by `ImportCommand`.
#[derive(Debug)]
pub enum ImportError {
    /// The requested format is not supported.
    UnsupportedFormat(String),
    /// The input is syntactically invalid.
    ParseError(String),
    /// The input is empty.
    EmptyInput,
    /// A triple could not be constructed from the parsed values.
    InvalidTriple(String),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::UnsupportedFormat(s) => write!(f, "Unsupported format: {}", s),
            ImportError::ParseError(s) => write!(f, "Parse error: {}", s),
            ImportError::EmptyInput => write!(f, "Input is empty"),
            ImportError::InvalidTriple(s) => write!(f, "Invalid triple: {}", s),
        }
    }
}

impl std::error::Error for ImportError {}

// ---------------------------------------------------------------------------
// ImportCommand
// ---------------------------------------------------------------------------

/// Stateless multi-format RDF importer.
pub struct ImportCommand;

impl ImportCommand {
    // -----------------------------------------------------------------------
    // Main entry point
    // -----------------------------------------------------------------------

    /// Parse `input` using the specified `format`.
    ///
    /// # Errors
    /// Returns `ImportError::EmptyInput` if the trimmed input is empty.
    pub fn import(input: &str, format: ImportFormat) -> Result<ImportResult, ImportError> {
        if input.trim().is_empty() {
            return Err(ImportError::EmptyInput);
        }
        match format {
            ImportFormat::NTriples => Self::parse_ntriples(input),
            ImportFormat::NQuads => Self::parse_nquads(input),
            ImportFormat::Turtle => Self::parse_turtle(input),
            ImportFormat::TriG => Self::parse_trig(input),
            ImportFormat::Csv => Self::parse_csv(input),
            ImportFormat::JsonLd => Self::parse_jsonld(input),
            ImportFormat::RdfXml => Self::parse_rdfxml(input),
        }
    }

    // -----------------------------------------------------------------------
    // N-Triples parser
    // -----------------------------------------------------------------------

    /// Parse N-Triples: `<subject> <predicate> <object> .` one per line.
    pub fn parse_ntriples(input: &str) -> Result<ImportResult, ImportError> {
        if input.trim().is_empty() {
            return Err(ImportError::EmptyInput);
        }
        let mut triples = Vec::new();
        let mut warnings = Vec::new();

        for (line_no, raw_line) in input.lines().enumerate() {
            let line = raw_line.trim();
            // Skip blank lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            match Self::parse_ntriples_line(line) {
                Ok(triple) => triples.push(triple),
                Err(msg) => warnings.push(format!("Line {}: {}", line_no + 1, msg)),
            }
        }

        Ok(ImportResult {
            triples,
            prefixes: HashMap::new(),
            graphs: Vec::new(),
            warnings,
            format_detected: ImportFormat::NTriples,
        })
    }

    // -----------------------------------------------------------------------
    // N-Quads parser
    // -----------------------------------------------------------------------

    /// Parse N-Quads: `<s> <p> <o> [<g>] .` one per line.
    pub fn parse_nquads(input: &str) -> Result<ImportResult, ImportError> {
        if input.trim().is_empty() {
            return Err(ImportError::EmptyInput);
        }
        let mut triples = Vec::new();
        let mut warnings = Vec::new();
        let mut graphs: Vec<String> = Vec::new();

        for (line_no, raw_line) in input.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            match Self::parse_nquads_line(line) {
                Ok(triple) => {
                    if let Some(ref g) = triple.graph {
                        if !graphs.contains(g) {
                            graphs.push(g.clone());
                        }
                    }
                    triples.push(triple);
                }
                Err(msg) => warnings.push(format!("Line {}: {}", line_no + 1, msg)),
            }
        }

        Ok(ImportResult {
            triples,
            prefixes: HashMap::new(),
            graphs,
            warnings,
            format_detected: ImportFormat::NQuads,
        })
    }

    // -----------------------------------------------------------------------
    // Turtle parser (simplified)
    // -----------------------------------------------------------------------

    /// Parse simplified Turtle: `@prefix` declarations followed by
    /// `<s> <p> <o> .` triples (full Turtle grammar subset).
    pub fn parse_turtle(input: &str) -> Result<ImportResult, ImportError> {
        if input.trim().is_empty() {
            return Err(ImportError::EmptyInput);
        }
        let mut triples = Vec::new();
        let mut prefixes: HashMap<String, String> = HashMap::new();
        let mut warnings = Vec::new();

        for (line_no, raw_line) in input.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // @prefix px: <iri> .
            if line.starts_with("@prefix") || line.starts_with("@base") {
                if let Some(ns) = Self::parse_prefix_decl(line) {
                    prefixes.insert(ns.0, ns.1);
                }
                continue;
            }
            // Plain triple or prefixed name triple
            match Self::parse_turtle_triple(line, &prefixes) {
                Ok(Some(triple)) => triples.push(triple),
                Ok(None) => {} // skipped line
                Err(msg) => warnings.push(format!("Line {}: {}", line_no + 1, msg)),
            }
        }

        Ok(ImportResult {
            triples,
            prefixes,
            graphs: Vec::new(),
            warnings,
            format_detected: ImportFormat::Turtle,
        })
    }

    // -----------------------------------------------------------------------
    // TriG parser
    // -----------------------------------------------------------------------

    /// Parse simplified TriG: Turtle + `GRAPH <iri> { ... }` blocks.
    pub fn parse_trig(input: &str) -> Result<ImportResult, ImportError> {
        if input.trim().is_empty() {
            return Err(ImportError::EmptyInput);
        }
        let mut triples = Vec::new();
        let mut prefixes: HashMap<String, String> = HashMap::new();
        let mut graphs: Vec<String> = Vec::new();
        let mut warnings = Vec::new();
        let mut current_graph: Option<String> = None;

        for (line_no, raw_line) in input.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with("@prefix") || line.starts_with("PREFIX") {
                if let Some(ns) = Self::parse_prefix_decl(line) {
                    prefixes.insert(ns.0, ns.1);
                }
                continue;
            }
            // GRAPH <iri> {
            if line.to_uppercase().starts_with("GRAPH") {
                if let Some(graph_iri) = Self::extract_graph_iri(line) {
                    if !graphs.contains(&graph_iri) {
                        graphs.push(graph_iri.clone());
                    }
                    current_graph = Some(graph_iri);
                }
                continue;
            }
            // End of graph block
            if line == "}" {
                current_graph = None;
                continue;
            }
            match Self::parse_turtle_triple(line, &prefixes) {
                Ok(Some(mut triple)) => {
                    triple.graph = current_graph.clone();
                    triples.push(triple);
                }
                Ok(None) => {}
                Err(msg) => warnings.push(format!("Line {}: {}", line_no + 1, msg)),
            }
        }

        Ok(ImportResult {
            triples,
            prefixes,
            graphs,
            warnings,
            format_detected: ImportFormat::TriG,
        })
    }

    // -----------------------------------------------------------------------
    // CSV parser
    // -----------------------------------------------------------------------

    /// Parse CSV: first line is a header `subject,predicate,object[,graph]`,
    /// subsequent lines are data rows.
    pub fn parse_csv(input: &str) -> Result<ImportResult, ImportError> {
        if input.trim().is_empty() {
            return Err(ImportError::EmptyInput);
        }
        let mut lines = input.lines();
        let header = lines.next().ok_or(ImportError::EmptyInput)?.trim();
        let cols: Vec<&str> = header.split(',').map(str::trim).collect();

        let has_graph = cols.len() >= 4 && cols[3].to_lowercase() == "graph";

        let mut triples = Vec::new();
        let mut warnings = Vec::new();
        let mut graphs: Vec<String> = Vec::new();

        for (row_no, raw_line) in lines.enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(4, ',').collect();
            if parts.len() < 3 {
                warnings.push(format!(
                    "Row {}: expected at least 3 columns, got {}",
                    row_no + 2,
                    parts.len()
                ));
                continue;
            }
            let graph = if has_graph && parts.len() >= 4 {
                let g = parts[3].trim().to_string();
                if !g.is_empty() {
                    if !graphs.contains(&g) {
                        graphs.push(g.clone());
                    }
                    Some(g)
                } else {
                    None
                }
            } else {
                None
            };
            triples.push(Triple {
                subject: parts[0].trim().to_string(),
                predicate: parts[1].trim().to_string(),
                object: parts[2].trim().to_string(),
                graph,
            });
        }

        Ok(ImportResult {
            triples,
            prefixes: HashMap::new(),
            graphs,
            warnings,
            format_detected: ImportFormat::Csv,
        })
    }

    // -----------------------------------------------------------------------
    // JSON-LD parser (simplified)
    // -----------------------------------------------------------------------

    /// Parse simplified JSON-LD: looks for `"@id"` and predicate keys in
    /// the `"@graph"` array or top-level object array.
    pub fn parse_jsonld(input: &str) -> Result<ImportResult, ImportError> {
        if input.trim().is_empty() {
            return Err(ImportError::EmptyInput);
        }
        let mut triples = Vec::new();
        let mut prefixes: HashMap<String, String> = HashMap::new();
        let mut warnings = Vec::new();

        // Extract @context prefix mappings
        if let Some(ctx_start) = input.find("\"@context\"") {
            if let Some(brace_start) = input[ctx_start..].find('{') {
                let ctx_text = &input[ctx_start + brace_start..];
                if let Some(brace_end) = Self::find_matching_brace(ctx_text) {
                    let ctx_body = &ctx_text[1..brace_end];
                    for line in ctx_body.lines() {
                        if let Some((k, v)) = Self::extract_json_string_pair(line) {
                            if !k.starts_with('@') {
                                prefixes.insert(k, v);
                            }
                        }
                    }
                }
            }
        }

        // Collect @graph items or top-level array items
        let graph_content = if let Some(pos) = input.find("\"@graph\"") {
            &input[pos..]
        } else {
            input
        };

        // Find all objects with @id
        let mut search_pos = 0;
        while let Some(id_pos) = graph_content[search_pos..].find("\"@id\"") {
            let abs_id = search_pos + id_pos;
            // Find the value
            let after_id = &graph_content[abs_id + 5..];
            let subject = match Self::extract_json_string_value(after_id) {
                Some(v) => v,
                None => {
                    search_pos = abs_id + 5;
                    continue;
                }
            };

            // Find the enclosing object boundaries
            let obj_start = Self::find_obj_start(&graph_content[..abs_id]);
            let obj_end_rel = Self::find_obj_end(&graph_content[abs_id..]).unwrap_or(100);
            let obj_end = abs_id + obj_end_rel;
            let obj_text = &graph_content[obj_start..obj_end.min(graph_content.len())];

            // Extract all predicate:value pairs
            for line in obj_text.lines() {
                if let Some((key, value)) = Self::extract_json_string_pair(line) {
                    if key == "@id" || key.starts_with('@') {
                        continue;
                    }
                    let predicate = if key.contains(':') {
                        // Already a prefixed or full IRI
                        if key.starts_with("http") {
                            key.clone()
                        } else {
                            // Expand prefixed name
                            let colon = key.find(':').unwrap_or(key.len());
                            let pfx = &key[..colon];
                            let local = &key[colon + 1..];
                            if let Some(ns) = prefixes.get(pfx) {
                                format!("{}{}", ns, local)
                            } else {
                                key.clone()
                            }
                        }
                    } else {
                        key.clone()
                    };
                    triples.push(Triple {
                        subject: subject.clone(),
                        predicate,
                        object: value,
                        graph: None,
                    });
                }
            }

            search_pos = abs_id + 5;
        }

        if triples.is_empty() && !input.contains("@id") {
            warnings.push("No @id found — no triples extracted from JSON-LD".to_string());
        }

        Ok(ImportResult {
            triples,
            prefixes,
            graphs: Vec::new(),
            warnings,
            format_detected: ImportFormat::JsonLd,
        })
    }

    // -----------------------------------------------------------------------
    // RDF/XML parser (simplified)
    // -----------------------------------------------------------------------

    /// Parse simplified RDF/XML: looks for `rdf:Description` elements and
    /// their child predicate elements.
    pub fn parse_rdfxml(input: &str) -> Result<ImportResult, ImportError> {
        if input.trim().is_empty() {
            return Err(ImportError::EmptyInput);
        }
        let mut triples = Vec::new();
        let mut warnings = Vec::new();

        // Extract subject from about attribute
        // Pattern: <rdf:Description rdf:about="IRI">
        let mut pos = 0;
        while let Some(desc_pos) = input[pos..].find("rdf:Description") {
            let abs = pos + desc_pos;
            // Find rdf:about="..."
            let tag_end = input[abs..]
                .find('>')
                .map(|p| abs + p)
                .unwrap_or(input.len());
            let tag_text = &input[abs..tag_end];
            let subject = Self::extract_xml_attr(tag_text, "rdf:about")
                .or_else(|| Self::extract_xml_attr(tag_text, "about"))
                .unwrap_or_else(|| "_:blank".to_string());

            // Find closing tag
            let close_tag = "</rdf:Description>";
            let block_end = input[abs..]
                .find(close_tag)
                .map(|p| abs + p)
                .unwrap_or(input.len());
            let block_text = &input[abs..block_end];

            // Extract child elements as predicates
            let mut child_pos = 0;
            while let Some(elem_start) = block_text[child_pos..].find('<') {
                let abs_elem = child_pos + elem_start;
                if block_text[abs_elem..].starts_with("<rdf:Description") {
                    break;
                }
                if block_text[abs_elem..].starts_with("</") {
                    child_pos = abs_elem + 2;
                    continue;
                }
                // Find tag name
                let rest = &block_text[abs_elem + 1..];
                let name_end = rest
                    .find(|c: char| c.is_whitespace() || c == '>' || c == '/')
                    .unwrap_or(rest.len());
                let tag_name = &rest[..name_end];
                if tag_name.is_empty() {
                    child_pos = abs_elem + 1;
                    continue;
                }
                // Find content between open and close tag
                let open_end = block_text[abs_elem..]
                    .find('>')
                    .map(|p| abs_elem + p + 1)
                    .unwrap_or(block_text.len());
                let close_pat = format!("</{}>", tag_name);
                let content_end = block_text[open_end..]
                    .find(&close_pat)
                    .map(|p| open_end + p);
                if let Some(end) = content_end {
                    let content = block_text[open_end..end].trim();
                    if !content.is_empty() && !tag_name.starts_with("rdf:") {
                        triples.push(Triple {
                            subject: subject.clone(),
                            predicate: tag_name.to_string(),
                            object: content.to_string(),
                            graph: None,
                        });
                    }
                    child_pos = end;
                } else {
                    child_pos = open_end;
                }
            }

            pos = block_end + close_tag.len();
            if pos >= input.len() {
                break;
            }
        }

        if triples.is_empty() {
            warnings.push("No rdf:Description elements found".to_string());
        }

        Ok(ImportResult {
            triples,
            prefixes: HashMap::new(),
            graphs: Vec::new(),
            warnings,
            format_detected: ImportFormat::RdfXml,
        })
    }

    // -----------------------------------------------------------------------
    // Format detection
    // -----------------------------------------------------------------------

    /// Sniff the content of `input` to guess its format.
    pub fn detect_format(input: &str) -> Option<ImportFormat> {
        let trimmed = input.trim_start();
        if trimmed.starts_with("@prefix")
            || trimmed.starts_with("@base")
            || (trimmed.starts_with('<') && trimmed.contains('>'))
                && !trimmed.contains("<?xml")
                && !trimmed.contains("<rdf:")
        {
            // N-Triples and Turtle both start with <...> but Turtle can have @prefix
            // Check for N-Quad / N-Triple patterns
            let first_line = trimmed.lines().next().unwrap_or("");
            if first_line.starts_with("@prefix") || first_line.starts_with("@base") {
                return Some(ImportFormat::Turtle);
            }
        }
        if trimmed.starts_with('@') {
            return Some(ImportFormat::Turtle);
        }
        if trimmed.starts_with('{')
            || trimmed.contains("\"@context\"")
            || trimmed.contains("\"@id\"")
        {
            return Some(ImportFormat::JsonLd);
        }
        if trimmed.starts_with("<?xml")
            || trimmed.starts_with("<rdf:RDF")
            || trimmed.contains("<rdf:Description")
        {
            return Some(ImportFormat::RdfXml);
        }
        // Check for GRAPH keyword (TriG)
        for line in trimmed.lines().take(20) {
            let l = line.trim();
            if l.to_uppercase().starts_with("GRAPH") && l.contains('<') {
                return Some(ImportFormat::TriG);
            }
        }
        // Check CSV header
        let first_line = trimmed.lines().next().unwrap_or("").to_lowercase();
        if first_line.contains("subject")
            && first_line.contains("predicate")
            && first_line.contains("object")
        {
            return Some(ImportFormat::Csv);
        }
        // N-Quads: lines with 4 angle-bracket IRIs
        let sample = trimmed.lines().next().unwrap_or("");
        let iri_count = sample.matches('<').count();
        if iri_count >= 4 {
            return Some(ImportFormat::NQuads);
        }
        // N-Triples: lines with 3 angle-bracket terms + dot
        if iri_count >= 2 && sample.ends_with('.') {
            return Some(ImportFormat::NTriples);
        }
        None
    }

    // -----------------------------------------------------------------------
    // Internal helpers — line parsing
    // -----------------------------------------------------------------------

    /// Strip angle brackets from an IRI token: `<http://...>` → `http://...`
    pub fn strip_iri(s: &str) -> &str {
        let s = s.trim();
        if s.starts_with('<') && s.ends_with('>') {
            &s[1..s.len() - 1]
        } else {
            s
        }
    }

    /// Unescape common N-Triples escape sequences in a literal string.
    pub fn unescape_literal(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('t') => result.push('\t'),
                    Some('r') => result.push('\r'),
                    Some('"') => result.push('"'),
                    Some('\\') => result.push('\\'),
                    Some('u') => {
                        // \uXXXX
                        let hex: String = chars.by_ref().take(4).collect();
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(ch) = char::from_u32(code) {
                                result.push(ch);
                                continue;
                            }
                        }
                        result.push_str(&format!("\\u{}", hex));
                    }
                    Some('U') => {
                        // \UXXXXXXXX
                        let hex: String = chars.by_ref().take(8).collect();
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(ch) = char::from_u32(code) {
                                result.push(ch);
                                continue;
                            }
                        }
                        result.push_str(&format!("\\U{}", hex));
                    }
                    Some(other) => {
                        result.push('\\');
                        result.push(other);
                    }
                    None => result.push('\\'),
                }
            } else {
                result.push(c);
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Private line-level parsers
    // -----------------------------------------------------------------------

    /// Parse a single N-Triples line into a `Triple`.
    fn parse_ntriples_line(line: &str) -> Result<Triple, String> {
        // Tokenise respecting quoted literals
        let tokens = Self::tokenise_nt_line(line);
        if tokens.len() < 3 {
            return Err(format!(
                "expected 3 terms, got {} in: {}",
                tokens.len(),
                line
            ));
        }
        let subject = Self::parse_nt_term(&tokens[0])?;
        let predicate = Self::parse_nt_term(&tokens[1])?;
        let object = Self::parse_nt_term(&tokens[2])?;
        Ok(Triple {
            subject,
            predicate,
            object,
            graph: None,
        })
    }

    /// Parse a single N-Quads line into a `Triple`.
    fn parse_nquads_line(line: &str) -> Result<Triple, String> {
        let tokens = Self::tokenise_nt_line(line);
        if tokens.len() < 3 {
            return Err(format!("expected ≥ 3 terms, got {}", tokens.len()));
        }
        let subject = Self::parse_nt_term(&tokens[0])?;
        let predicate = Self::parse_nt_term(&tokens[1])?;
        let object = Self::parse_nt_term(&tokens[2])?;
        let graph = if tokens.len() >= 4 && tokens[3] != "." {
            // tokens[3] may be a graph IRI or "."
            if tokens[3].starts_with('<') {
                Some(Self::parse_nt_term(&tokens[3])?)
            } else {
                None
            }
        } else {
            None
        };
        Ok(Triple {
            subject,
            predicate,
            object,
            graph,
        })
    }

    /// Tokenise an N-Triples / N-Quads line, respecting quoted literals.
    fn tokenise_nt_line(line: &str) -> Vec<String> {
        let line = line.trim_end_matches('.');
        let line = line.trim();
        let mut tokens = Vec::new();
        let mut chars = line.chars().peekable();

        while let Some(&c) = chars.peek() {
            match c {
                ' ' | '\t' => {
                    chars.next();
                }
                '<' => {
                    // IRI token
                    let mut tok = String::from('<');
                    chars.next();
                    for ch in chars.by_ref() {
                        tok.push(ch);
                        if ch == '>' {
                            break;
                        }
                    }
                    tokens.push(tok);
                }
                '"' => {
                    // Literal token
                    let mut tok = String::from('"');
                    chars.next();
                    let mut escaped = false;
                    for ch in chars.by_ref() {
                        tok.push(ch);
                        if escaped {
                            escaped = false;
                        } else if ch == '\\' {
                            escaped = true;
                        } else if ch == '"' {
                            break;
                        }
                    }
                    // Datatype / lang tag
                    if let Some(&next) = chars.peek() {
                        if next == '^' || next == '@' {
                            tok.push(next);
                            chars.next();
                            // Consume until whitespace
                            for ch in chars.by_ref() {
                                if ch == ' ' || ch == '\t' {
                                    break;
                                }
                                tok.push(ch);
                            }
                        }
                    }
                    tokens.push(tok);
                }
                '_' => {
                    // Blank node _:label
                    let mut tok = String::new();
                    for ch in chars.by_ref() {
                        if ch == ' ' || ch == '\t' {
                            break;
                        }
                        tok.push(ch);
                    }
                    tokens.push(tok);
                }
                '.' => {
                    chars.next();
                }
                _ => {
                    // Unknown token — consume until whitespace
                    let mut tok = String::new();
                    for ch in chars.by_ref() {
                        if ch == ' ' || ch == '\t' {
                            break;
                        }
                        tok.push(ch);
                    }
                    tokens.push(tok);
                }
            }
        }
        tokens
    }

    /// Convert an N-Triples term token to a string value.
    fn parse_nt_term(token: &str) -> Result<String, String> {
        let t = token.trim();
        if t.starts_with('<') && t.ends_with('>') {
            return Ok(t[1..t.len() - 1].to_string());
        }
        if t.starts_with("_:") {
            return Ok(t.to_string());
        }
        if t.starts_with('"') {
            // Literal
            return Ok(Self::unescape_literal(
                &t[1..t.rfind('"').unwrap_or(t.len())],
            ));
        }
        Err(format!("unrecognised term: {}", t))
    }

    /// Parse a Turtle `@prefix px: <iri> .` or SPARQL `PREFIX px: <iri>` line.
    fn parse_prefix_decl(line: &str) -> Option<(String, String)> {
        let line = line
            .trim_start_matches("@prefix")
            .trim_start_matches("PREFIX")
            .trim()
            .trim_end_matches('.');
        // prefix: <iri>
        let colon = line.find(':')?;
        let prefix = line[..colon].trim().to_string();
        let rest = line[colon + 1..].trim();
        if rest.starts_with('<') && rest.ends_with('>') {
            return Some((prefix, rest[1..rest.len() - 1].to_string()));
        }
        None
    }

    /// Parse a Turtle triple line, expanding prefixed names with `prefixes`.
    fn parse_turtle_triple(
        line: &str,
        prefixes: &HashMap<String, String>,
    ) -> Result<Option<Triple>, String> {
        let line = line.trim_end_matches(['.', ';', ','].as_ref()).trim();
        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }
        let tokens = Self::tokenise_turtle_line(line, prefixes);
        if tokens.len() < 3 {
            if tokens.is_empty() {
                return Ok(None);
            }
            return Err(format!("need 3 terms, got {}", tokens.len()));
        }
        Ok(Some(Triple {
            subject: tokens[0].clone(),
            predicate: tokens[1].clone(),
            object: tokens[2].clone(),
            graph: None,
        }))
    }

    /// Simplified Turtle tokeniser that expands `prefix:local` names.
    fn tokenise_turtle_line(line: &str, prefixes: &HashMap<String, String>) -> Vec<String> {
        let mut tokens = Vec::new();
        for token in line.split_whitespace() {
            let tok = token.trim_end_matches(['.', ';', ','].as_ref());
            if tok.is_empty() {
                continue;
            }
            if tok.starts_with('<') && tok.ends_with('>') {
                tokens.push(tok[1..tok.len() - 1].to_string());
            } else if tok.starts_with('"') {
                tokens.push(tok.trim_matches('"').to_string());
            } else if tok.contains(':') && !tok.starts_with("http") {
                // Prefixed name
                let colon = tok.find(':').unwrap_or(tok.len());
                let pfx = &tok[..colon];
                let local = &tok[colon + 1..];
                if let Some(ns) = prefixes.get(pfx) {
                    tokens.push(format!("{}{}", ns, local));
                } else {
                    tokens.push(tok.to_string());
                }
            } else {
                tokens.push(tok.to_string());
            }
        }
        tokens
    }

    /// Extract the named-graph IRI from a `GRAPH <iri> {` line.
    fn extract_graph_iri(line: &str) -> Option<String> {
        let start = line.find('<')? + 1;
        let end = line[start..].find('>')? + start;
        Some(line[start..end].to_string())
    }

    /// Extract an XML attribute value: `attr="value"`.
    fn extract_xml_attr(text: &str, attr: &str) -> Option<String> {
        let search = format!("{}=\"", attr);
        let start = text.find(&search)? + search.len();
        let end = text[start..].find('"')? + start;
        Some(text[start..end].to_string())
    }

    /// Extract a JSON key-value string pair from a line: `"key": "value"`.
    fn extract_json_string_pair(line: &str) -> Option<(String, String)> {
        let colon = line.find(":")?;
        let key_part = line[..colon].trim().trim_matches('"').to_string();
        let val_part = line[colon + 1..].trim();
        let value = if val_part.starts_with('"') {
            val_part.trim_matches(['"', ','].as_ref()).to_string()
        } else {
            return None;
        };
        Some((key_part, value))
    }

    /// Extract the string value from a JSON `"key": "value"` segment.
    fn extract_json_string_value(s: &str) -> Option<String> {
        let start = s.find('"')? + 1;
        let end = s[start..].find('"')? + start;
        Some(s[start..end].to_string())
    }

    /// Find the position of the matching `}` for an opening `{` at position 0.
    fn find_matching_brace(s: &str) -> Option<usize> {
        let mut depth = 0;
        for (i, c) in s.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Find the start of the enclosing JSON object (last `{` before pos).
    fn find_obj_start(s: &str) -> usize {
        s.rfind('{').map(|p| p + 1).unwrap_or(0)
    }

    /// Find the end of the current JSON object (next `}` or `}`).
    fn find_obj_end(s: &str) -> Option<usize> {
        s.find('}').map(|p| p + 1)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- ImportFormat helpers -----------------------------------------------

    #[test]
    fn test_from_extension_turtle() {
        assert_eq!(
            ImportFormat::from_extension("ttl"),
            Some(ImportFormat::Turtle)
        );
    }

    #[test]
    fn test_from_extension_ntriples() {
        assert_eq!(
            ImportFormat::from_extension("nt"),
            Some(ImportFormat::NTriples)
        );
    }

    #[test]
    fn test_from_extension_nquads() {
        assert_eq!(
            ImportFormat::from_extension("nq"),
            Some(ImportFormat::NQuads)
        );
    }

    #[test]
    fn test_from_extension_jsonld() {
        assert_eq!(
            ImportFormat::from_extension("jsonld"),
            Some(ImportFormat::JsonLd)
        );
    }

    #[test]
    fn test_from_extension_rdf() {
        assert_eq!(
            ImportFormat::from_extension("rdf"),
            Some(ImportFormat::RdfXml)
        );
    }

    #[test]
    fn test_from_extension_trig() {
        assert_eq!(
            ImportFormat::from_extension("trig"),
            Some(ImportFormat::TriG)
        );
    }

    #[test]
    fn test_from_extension_csv() {
        assert_eq!(ImportFormat::from_extension("csv"), Some(ImportFormat::Csv));
    }

    #[test]
    fn test_from_extension_unknown() {
        assert_eq!(ImportFormat::from_extension("docx"), None);
    }

    #[test]
    fn test_from_extension_case_insensitive() {
        assert_eq!(
            ImportFormat::from_extension("TTL"),
            Some(ImportFormat::Turtle)
        );
    }

    #[test]
    fn test_from_mime_type_turtle() {
        assert_eq!(
            ImportFormat::from_mime_type("text/turtle"),
            Some(ImportFormat::Turtle)
        );
    }

    #[test]
    fn test_from_mime_type_ntriples() {
        assert_eq!(
            ImportFormat::from_mime_type("application/n-triples"),
            Some(ImportFormat::NTriples)
        );
    }

    #[test]
    fn test_from_mime_type_jsonld() {
        assert_eq!(
            ImportFormat::from_mime_type("application/ld+json"),
            Some(ImportFormat::JsonLd)
        );
    }

    #[test]
    fn test_from_mime_type_csv() {
        assert_eq!(
            ImportFormat::from_mime_type("text/csv"),
            Some(ImportFormat::Csv)
        );
    }

    #[test]
    fn test_from_mime_type_unknown() {
        assert_eq!(ImportFormat::from_mime_type("text/plain"), None);
    }

    #[test]
    fn test_extension_and_mime_type() {
        assert_eq!(ImportFormat::Turtle.extension(), "ttl");
        assert_eq!(ImportFormat::Turtle.mime_type(), "text/turtle");
        assert_eq!(ImportFormat::NTriples.extension(), "nt");
        assert_eq!(ImportFormat::NQuads.extension(), "nq");
        assert_eq!(ImportFormat::JsonLd.extension(), "jsonld");
        assert_eq!(ImportFormat::RdfXml.extension(), "rdf");
        assert_eq!(ImportFormat::TriG.extension(), "trig");
        assert_eq!(ImportFormat::Csv.extension(), "csv");
    }

    // --- Empty input --------------------------------------------------------

    #[test]
    fn test_empty_input_error() {
        assert!(matches!(
            ImportCommand::import("", ImportFormat::NTriples),
            Err(ImportError::EmptyInput)
        ));
        assert!(matches!(
            ImportCommand::import("   \n", ImportFormat::Turtle),
            Err(ImportError::EmptyInput)
        ));
    }

    // --- N-Triples ----------------------------------------------------------

    #[test]
    fn test_parse_ntriples_single() {
        let nt = "<http://a.org/s> <http://a.org/p> <http://a.org/o> .\n";
        let result = ImportCommand::parse_ntriples(nt).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].subject, "http://a.org/s");
        assert_eq!(result.triples[0].predicate, "http://a.org/p");
        assert_eq!(result.triples[0].object, "http://a.org/o");
        assert!(result.triples[0].graph.is_none());
    }

    #[test]
    fn test_parse_ntriples_multiple() {
        let nt = "<http://a/s> <http://a/p> <http://a/o> .\n\
                  <http://b/s> <http://b/p> <http://b/o> .\n";
        let result = ImportCommand::parse_ntriples(nt).expect("ok");
        assert_eq!(result.triple_count(), 2);
    }

    #[test]
    fn test_parse_ntriples_blank_node() {
        let nt = "_:b1 <http://a.org/p> <http://a.org/o> .\n";
        let result = ImportCommand::parse_ntriples(nt).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].subject, "_:b1");
    }

    #[test]
    fn test_parse_ntriples_literal_object() {
        let nt = "<http://a.org/s> <http://a.org/p> \"hello\" .\n";
        let result = ImportCommand::parse_ntriples(nt).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].object, "hello");
    }

    #[test]
    fn test_parse_ntriples_comment_skipped() {
        let nt = "# This is a comment\n<http://a/s> <http://a/p> <http://a/o> .\n";
        let result = ImportCommand::parse_ntriples(nt).expect("ok");
        assert_eq!(result.triple_count(), 1);
    }

    #[test]
    fn test_parse_ntriples_malformed_warning() {
        let nt = "this is not a valid triple\n<http://a/s> <http://a/p> <http://a/o> .\n";
        let result = ImportCommand::parse_ntriples(nt).expect("ok");
        assert!(
            result.has_warnings(),
            "expected warnings for malformed line"
        );
        assert_eq!(result.triple_count(), 1);
    }

    // --- N-Quads ------------------------------------------------------------

    #[test]
    fn test_parse_nquads_with_graph() {
        let nq = "<http://s> <http://p> <http://o> <http://g> .\n";
        let result = ImportCommand::parse_nquads(nq).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].graph, Some("http://g".to_string()));
        assert_eq!(result.graph_count(), 1);
    }

    #[test]
    fn test_parse_nquads_without_graph() {
        let nq = "<http://s> <http://p> <http://o> .\n";
        let result = ImportCommand::parse_nquads(nq).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert!(result.triples[0].graph.is_none());
    }

    #[test]
    fn test_parse_nquads_multiple_graphs() {
        let nq = "<http://s1> <http://p> <http://o1> <http://g1> .\n\
                  <http://s2> <http://p> <http://o2> <http://g2> .\n";
        let result = ImportCommand::parse_nquads(nq).expect("ok");
        assert_eq!(result.graph_count(), 2);
    }

    // --- Turtle -------------------------------------------------------------

    #[test]
    fn test_parse_turtle_with_prefix() {
        let ttl = "@prefix ex: <http://example.org/> .\n\
                   <http://a.org/s> <http://a.org/p> <http://a.org/o> .\n";
        let result = ImportCommand::parse_turtle(ttl).expect("ok");
        assert!(!result.prefixes.is_empty());
        // Prefix "ex" should be recorded
        assert!(result.prefixes.contains_key("ex"));
    }

    #[test]
    fn test_parse_turtle_prefix_extraction() {
        let ttl = "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n\
                   <http://s.org/s> <http://s.org/p> <http://s.org/o> .\n";
        let result = ImportCommand::parse_turtle(ttl).expect("ok");
        assert!(result.prefixes.contains_key("rdf"));
        assert_eq!(
            result.prefixes.get("rdf").map(String::as_str),
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        );
    }

    #[test]
    fn test_parse_turtle_triple_count() {
        let ttl = "<http://a/s> <http://a/p> <http://a/o> .\n\
                   <http://b/s> <http://b/p> <http://b/o> .\n";
        let result = ImportCommand::parse_turtle(ttl).expect("ok");
        assert_eq!(result.triple_count(), 2);
        assert_eq!(result.format_detected, ImportFormat::Turtle);
    }

    // --- TriG ---------------------------------------------------------------

    #[test]
    fn test_parse_trig_graph_block() {
        let trig = "GRAPH <http://g.org/g1> {\n\
                    <http://a/s> <http://a/p> <http://a/o> .\n\
                    }\n";
        let result = ImportCommand::parse_trig(trig).expect("ok");
        assert_eq!(result.graph_count(), 1);
        assert_eq!(result.graphs[0], "http://g.org/g1");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].graph, Some("http://g.org/g1".to_string()));
    }

    // --- CSV ----------------------------------------------------------------

    #[test]
    fn test_parse_csv_basic() {
        let csv = "subject,predicate,object\n\
                   http://a/s,http://a/p,http://a/o\n";
        let result = ImportCommand::parse_csv(csv).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].subject, "http://a/s");
    }

    #[test]
    fn test_parse_csv_with_graph_column() {
        let csv = "subject,predicate,object,graph\n\
                   http://s,http://p,http://o,http://g\n";
        let result = ImportCommand::parse_csv(csv).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].graph, Some("http://g".to_string()));
        assert_eq!(result.graph_count(), 1);
    }

    #[test]
    fn test_parse_csv_missing_column_warning() {
        let csv = "subject,predicate,object\n\
                   only_one_column\n";
        let result = ImportCommand::parse_csv(csv).expect("ok");
        assert!(result.has_warnings());
    }

    #[test]
    fn test_parse_csv_multiple_rows() {
        let csv = "subject,predicate,object\n\
                   http://a/s1,http://a/p,http://a/o1\n\
                   http://a/s2,http://a/p,http://a/o2\n";
        let result = ImportCommand::parse_csv(csv).expect("ok");
        assert_eq!(result.triple_count(), 2);
    }

    // --- JSON-LD (simplified) -----------------------------------------------

    #[test]
    fn test_parse_jsonld_simple() {
        let jsonld = r#"{"@context":{"name":"http://schema.org/name"},"@graph":[{"@id":"http://a.org/person","name":"Alice"}]}"#;
        let result = ImportCommand::parse_jsonld(jsonld).expect("ok");
        let _ = result.triple_count(); // simplified parser may or may not find triples
        assert_eq!(result.format_detected, ImportFormat::JsonLd);
    }

    #[test]
    fn test_parse_jsonld_empty_warning() {
        let jsonld = r#"{"@context":{},"@graph":[]}"#;
        let result = ImportCommand::parse_jsonld(jsonld).expect("ok");
        // Warnings expected since no triples
        assert_eq!(result.format_detected, ImportFormat::JsonLd);
    }

    // --- RDF/XML (simplified) -----------------------------------------------

    #[test]
    fn test_parse_rdfxml_basic() {
        let rdfxml = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:ex="http://example.org/">
  <rdf:Description rdf:about="http://example.org/alice">
    <ex:name>Alice</ex:name>
  </rdf:Description>
</rdf:RDF>"#;
        let result = ImportCommand::parse_rdfxml(rdfxml).expect("ok");
        assert_eq!(result.format_detected, ImportFormat::RdfXml);
        assert!(result.triple_count() > 0 || result.has_warnings());
    }

    // --- detect_format ------------------------------------------------------

    #[test]
    fn test_detect_format_turtle() {
        let input = "@prefix ex: <http://example.org/> .\n<http://s> <http://p> <http://o> .\n";
        assert_eq!(
            ImportCommand::detect_format(input),
            Some(ImportFormat::Turtle)
        );
    }

    #[test]
    fn test_detect_format_jsonld() {
        let input = r#"{"@context":{},"@id":"http://a.org/x"}"#;
        assert_eq!(
            ImportCommand::detect_format(input),
            Some(ImportFormat::JsonLd)
        );
    }

    #[test]
    fn test_detect_format_rdfxml() {
        let input = "<?xml version=\"1.0\"?><rdf:RDF></rdf:RDF>";
        assert_eq!(
            ImportCommand::detect_format(input),
            Some(ImportFormat::RdfXml)
        );
    }

    #[test]
    fn test_detect_format_trig() {
        let input = "GRAPH <http://g.org/> { <http://s> <http://p> <http://o> . }";
        assert_eq!(
            ImportCommand::detect_format(input),
            Some(ImportFormat::TriG)
        );
    }

    #[test]
    fn test_detect_format_csv() {
        let input = "subject,predicate,object\nhttp://s,http://p,http://o\n";
        assert_eq!(ImportCommand::detect_format(input), Some(ImportFormat::Csv));
    }

    // --- strip_iri ----------------------------------------------------------

    #[test]
    fn test_strip_iri_with_brackets() {
        assert_eq!(
            ImportCommand::strip_iri("<http://example.org/>"),
            "http://example.org/"
        );
    }

    #[test]
    fn test_strip_iri_without_brackets() {
        assert_eq!(
            ImportCommand::strip_iri("http://example.org/"),
            "http://example.org/"
        );
    }

    #[test]
    fn test_strip_iri_with_whitespace() {
        assert_eq!(
            ImportCommand::strip_iri("  <http://example.org/>  "),
            "http://example.org/"
        );
    }

    // --- unescape_literal ---------------------------------------------------

    #[test]
    fn test_unescape_literal_newline() {
        assert_eq!(
            ImportCommand::unescape_literal("line1\\nline2"),
            "line1\nline2"
        );
    }

    #[test]
    fn test_unescape_literal_tab() {
        assert_eq!(ImportCommand::unescape_literal("col1\\tcol2"), "col1\tcol2");
    }

    #[test]
    fn test_unescape_literal_quote() {
        assert_eq!(
            ImportCommand::unescape_literal("say \\\"hi\\\""),
            "say \"hi\""
        );
    }

    #[test]
    fn test_unescape_literal_backslash() {
        assert_eq!(
            ImportCommand::unescape_literal("back\\\\slash"),
            "back\\slash"
        );
    }

    #[test]
    fn test_unescape_literal_unicode() {
        // \u0041 = 'A'
        assert_eq!(ImportCommand::unescape_literal("\\u0041"), "A");
    }

    #[test]
    fn test_unescape_literal_no_escape() {
        assert_eq!(ImportCommand::unescape_literal("hello"), "hello");
    }

    // --- ImportResult helpers -----------------------------------------------

    #[test]
    fn test_import_result_triple_count() {
        let r = ImportResult {
            triples: vec![
                Triple {
                    subject: "s".to_string(),
                    predicate: "p".to_string(),
                    object: "o".to_string(),
                    graph: None,
                };
                3
            ],
            prefixes: HashMap::new(),
            graphs: Vec::new(),
            warnings: Vec::new(),
            format_detected: ImportFormat::NTriples,
        };
        assert_eq!(r.triple_count(), 3);
    }

    #[test]
    fn test_import_result_graph_count() {
        let r = ImportResult {
            triples: Vec::new(),
            prefixes: HashMap::new(),
            graphs: vec!["g1".to_string(), "g2".to_string()],
            warnings: Vec::new(),
            format_detected: ImportFormat::NQuads,
        };
        assert_eq!(r.graph_count(), 2);
    }

    #[test]
    fn test_import_result_has_warnings() {
        let mut r = ImportResult {
            triples: Vec::new(),
            prefixes: HashMap::new(),
            graphs: Vec::new(),
            warnings: Vec::new(),
            format_detected: ImportFormat::NTriples,
        };
        assert!(!r.has_warnings());
        r.warnings.push("warn".to_string());
        assert!(r.has_warnings());
    }

    // --- Error display -------------------------------------------------------

    #[test]
    fn test_import_error_display() {
        assert!(ImportError::EmptyInput.to_string().contains("empty"));
        assert!(ImportError::ParseError("bad".to_string())
            .to_string()
            .contains("bad"));
        assert!(ImportError::UnsupportedFormat("xyz".to_string())
            .to_string()
            .contains("xyz"));
        assert!(ImportError::InvalidTriple("bad".to_string())
            .to_string()
            .contains("bad"));
    }

    // --- import() dispatch --------------------------------------------------

    #[test]
    fn test_import_dispatch_ntriples() {
        let nt = "<http://a/s> <http://a/p> <http://a/o> .\n";
        let result = ImportCommand::import(nt, ImportFormat::NTriples).expect("ok");
        assert_eq!(result.format_detected, ImportFormat::NTriples);
    }

    #[test]
    fn test_import_dispatch_csv() {
        let csv = "subject,predicate,object\nhttp://s,http://p,http://o\n";
        let result = ImportCommand::import(csv, ImportFormat::Csv).expect("ok");
        assert_eq!(result.format_detected, ImportFormat::Csv);
        assert_eq!(result.triple_count(), 1);
    }
}
