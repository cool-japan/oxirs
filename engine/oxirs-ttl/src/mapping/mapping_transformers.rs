//! Transformation and execution logic: `MappingEngine`, `MappingRuleBuilder`,
//! CSV/JSON parsers, and related helpers.

use oxirs_core::model::{NamedNode, Subject, Triple};

use super::mapping_types::{
    build_triple_from_pom, DataSource, MappingError, MappingResult, MappingRule, ObjectSpec,
    PredicateObjectMap, Row, Template,
};

// ─── MappingEngine ────────────────────────────────────────────────────────────

/// Engine that executes [`MappingRule`]s and produces RDF [`Triple`]s
///
/// The engine is stateless and cheap to create.  All configuration is
/// carried by the rules themselves.
#[derive(Debug, Default, Clone)]
pub struct MappingEngine {
    /// Whether to skip rows that produce errors instead of failing fast
    pub skip_errors: bool,
}

impl MappingEngine {
    /// Create a new mapping engine with default settings (fail-fast)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an engine that silently skips rows that produce mapping errors
    pub fn new_lenient() -> Self {
        Self { skip_errors: true }
    }

    /// Execute a single mapping rule and return all produced triples
    pub fn execute(&self, rule: &MappingRule) -> MappingResult<Vec<Triple>> {
        let (headers, rows) = self.extract_rows(&rule.source)?;
        let _ = headers; // headers are embedded inside each Row already
        self.map_rows(rule, &rows)
    }

    /// Execute multiple rules and concatenate all produced triples
    pub fn execute_all(&self, rules: &[MappingRule]) -> MappingResult<Vec<Triple>> {
        let mut all_triples = Vec::new();
        for rule in rules {
            let mut triples = self.execute(rule)?;
            all_triples.append(&mut triples);
        }
        Ok(all_triples)
    }

    // ─── Internal helpers ────────────────────────────────────────────────

    fn extract_rows(&self, source: &DataSource) -> MappingResult<(Vec<String>, Vec<Row>)> {
        match source {
            DataSource::Csv { content, delimiter } => Self::parse_csv(content, *delimiter),
            DataSource::Json { content, json_path } => {
                let rows = Self::parse_json(content, json_path.as_deref())?;
                // headers are implicit in the Row keys; return empty list
                Ok((Vec::new(), rows))
            }
            DataSource::InlineValues { rows, headers } => {
                let parsed_rows: Vec<Row> = rows
                    .iter()
                    .map(|row_values| {
                        let pairs = headers
                            .iter()
                            .zip(row_values.iter())
                            .map(|(h, v)| (h.clone(), v.clone()));
                        Row::from_pairs(pairs)
                    })
                    .collect();
                Ok((headers.clone(), parsed_rows))
            }
        }
    }

    fn map_rows(&self, rule: &MappingRule, rows: &[Row]) -> MappingResult<Vec<Triple>> {
        let mut triples = Vec::with_capacity(rows.len() * rule.predicate_object_maps.len());

        for (row_idx, row) in rows.iter().enumerate() {
            // Generate subject IRI
            let subject_iri = match rule.subject_template.render(row, row_idx) {
                Ok(iri) => iri,
                Err(e) => {
                    if self.skip_errors {
                        continue;
                    }
                    return Err(e);
                }
            };

            let subject_node =
                NamedNode::new(&subject_iri).map_err(|e| MappingError::InvalidIri {
                    template: rule.subject_template.pattern.clone(),
                    iri: format!("{subject_iri} ({e})"),
                })?;
            let subject: Subject = subject_node.into();

            // Generate one triple per predicate-object map
            for pom in &rule.predicate_object_maps {
                let result = build_triple_from_pom(&subject, pom, row, row_idx);
                match result {
                    Ok(triple) => triples.push(triple),
                    Err(e) => {
                        if self.skip_errors {
                            continue;
                        }
                        return Err(e);
                    }
                }
            }
        }
        Ok(triples)
    }

    // ─── CSV parser ──────────────────────────────────────────────────────

    /// Parse CSV content into (headers, rows).
    ///
    /// Handles:
    /// - Configurable delimiter
    /// - Double-quote escaping (`""` inside a quoted field)
    /// - CRLF and LF line endings
    /// - Quoted fields that span multiple lines
    pub fn parse_csv(content: &str, delimiter: char) -> MappingResult<(Vec<String>, Vec<Row>)> {
        let lines = split_csv_lines(content);
        if lines.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        // Parse header row
        let headers = parse_csv_line(&lines[0], delimiter);
        if headers.is_empty() {
            return Err(MappingError::CsvParseError {
                line: 1,
                message: "empty header row".to_string(),
            });
        }

        let mut rows = Vec::with_capacity(lines.len().saturating_sub(1));
        for (line_idx, line) in lines.iter().enumerate().skip(1) {
            if line.trim().is_empty() {
                continue;
            }
            let values = parse_csv_line(line, delimiter);
            if values.len() != headers.len() {
                return Err(MappingError::CsvParseError {
                    line: line_idx + 1,
                    message: format!("expected {} fields but got {}", headers.len(), values.len()),
                });
            }
            let row = Row::from_pairs(headers.iter().cloned().zip(values));
            rows.push(row);
        }
        Ok((headers, rows))
    }

    // ─── JSON parser ─────────────────────────────────────────────────────

    /// Parse JSON content into rows.
    ///
    /// Behaviour:
    /// - If `json_path` is `None`, the root must be a JSON array of objects.
    /// - If `json_path` is `Some("a.b.c")`, the engine traverses object keys
    ///   `a` → `b` → `c` and expects to find an array there.
    /// - Each array element must be a JSON object; its key-value pairs become
    ///   the row fields (values are coerced to strings).
    pub fn parse_json(content: &str, json_path: Option<&str>) -> MappingResult<Vec<Row>> {
        let value: serde_json::Value =
            serde_json::from_str(content).map_err(|e| MappingError::JsonParseError {
                message: e.to_string(),
            })?;

        // Navigate to the target array using dot-separated path
        let array = if let Some(path) = json_path {
            navigate_json_path(&value, path)?
        } else {
            &value
        };

        let arr = array.as_array().ok_or_else(|| {
            let path_desc = json_path.unwrap_or("<root>");
            MappingError::JsonPathNoMatch {
                path: path_desc.to_string(),
            }
        })?;

        let mut rows = Vec::with_capacity(arr.len());
        for element in arr {
            let obj = element
                .as_object()
                .ok_or_else(|| MappingError::JsonParseError {
                    message: "JSON array element is not an object".to_string(),
                })?;
            let row = Row::from_pairs(
                obj.iter()
                    .map(|(k, v)| (k.clone(), json_value_to_string(v))),
            );
            rows.push(row);
        }
        Ok(rows)
    }
}

// ─── JSON helpers ─────────────────────────────────────────────────────────────

fn navigate_json_path<'a>(
    value: &'a serde_json::Value,
    path: &str,
) -> MappingResult<&'a serde_json::Value> {
    let mut current = value;
    for key in path.split('.') {
        current = current
            .get(key)
            .ok_or_else(|| MappingError::JsonPathNoMatch {
                path: path.to_string(),
            })?;
    }
    Ok(current)
}

fn json_value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

// ─── CSV helpers ──────────────────────────────────────────────────────────────

/// Split CSV text into logical lines, handling quoted fields that contain newlines.
fn split_csv_lines(content: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = content.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                current.push(ch);
            }
            '\r' => {
                // Handle CRLF
                if chars.peek() == Some(&'\n') {
                    let _ = chars.next();
                }
                if !in_quotes {
                    lines.push(std::mem::take(&mut current));
                } else {
                    current.push('\n');
                }
            }
            '\n' if !in_quotes => {
                lines.push(std::mem::take(&mut current));
            }
            _ => {
                current.push(ch);
            }
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// Parse a single CSV line into a vector of field values.
fn parse_csv_line(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    // Escaped double-quote inside quoted field
                    current.push('"');
                    let _ = chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == delimiter {
            fields.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    fields.push(current);
    fields
}

// ─── Builder ──────────────────────────────────────────────────────────────────

/// Fluent builder for constructing [`MappingRule`] instances
///
/// # Example
///
/// ```rust
/// use oxirs_ttl::mapping::{MappingRuleBuilder, ObjectSpec};
///
/// let rule = MappingRuleBuilder::new("employees")
///     .csv_source("id,name\n1,Alice\n2,Bob")
///     .subject_template("http://example.org/employee/{id}")
///     .map("http://xmlns.com/foaf/0.1/name", ObjectSpec::Column("name".to_string()))
///     .build();
/// ```
#[derive(Debug)]
pub struct MappingRuleBuilder {
    rule: MappingRule,
}

impl MappingRuleBuilder {
    /// Start building a new rule with the given name
    pub fn new(name: impl Into<String>) -> Self {
        let name_str = name.into();
        Self {
            rule: MappingRule {
                name: name_str,
                source: DataSource::Csv {
                    content: String::new(),
                    delimiter: ',',
                },
                subject_template: Template::new(""),
                predicate_object_maps: Vec::new(),
                graph_name: None,
            },
        }
    }

    /// Use a CSV string as the data source (comma delimiter)
    pub fn csv_source(mut self, content: impl Into<String>) -> Self {
        self.rule.source = DataSource::Csv {
            content: content.into(),
            delimiter: ',',
        };
        self
    }

    /// Use a CSV string with a custom delimiter
    pub fn csv_source_with_delimiter(
        mut self,
        content: impl Into<String>,
        delimiter: char,
    ) -> Self {
        self.rule.source = DataSource::Csv {
            content: content.into(),
            delimiter,
        };
        self
    }

    /// Use a JSON string as the data source (root must be an array)
    pub fn json_source(mut self, content: impl Into<String>) -> Self {
        self.rule.source = DataSource::Json {
            content: content.into(),
            json_path: None,
        };
        self
    }

    /// Use a JSON string with a dot-separated path to the target array
    pub fn json_source_with_path(
        mut self,
        content: impl Into<String>,
        json_path: impl Into<String>,
    ) -> Self {
        self.rule.source = DataSource::Json {
            content: content.into(),
            json_path: Some(json_path.into()),
        };
        self
    }

    /// Use pre-parsed inline values
    pub fn inline_source(mut self, headers: Vec<String>, rows: Vec<Vec<String>>) -> Self {
        self.rule.source = DataSource::InlineValues { rows, headers };
        self
    }

    /// Set the subject IRI template
    pub fn subject_template(mut self, template: impl Into<String>) -> Self {
        self.rule.subject_template = Template::new(template);
        self
    }

    /// Add a predicate-object mapping
    pub fn map(mut self, predicate: impl Into<String>, object: ObjectSpec) -> Self {
        self.rule.predicate_object_maps.push(PredicateObjectMap {
            predicate: predicate.into(),
            object_template: object,
        });
        self
    }

    /// Assign all produced triples to a named graph
    pub fn graph(mut self, graph_name: impl Into<String>) -> Self {
        self.rule.graph_name = Some(graph_name.into());
        self
    }

    /// Consume the builder and return the finished [`MappingRule`]
    pub fn build(self) -> MappingRule {
        self.rule
    }
}
