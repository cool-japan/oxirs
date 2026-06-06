//! Core data types for RML mapping: errors, rows, templates, specs, rules.

use std::collections::HashMap;
use std::fmt;

use oxirs_core::model::{NamedNode, Object, Predicate, Subject, Triple};
use thiserror::Error;

// ─── Error Types ─────────────────────────────────────────────────────────────

/// Errors that can occur during RML mapping operations
#[derive(Debug, Error)]
pub enum MappingError {
    /// A required column was not found in the data row
    #[error("Missing column '{column}' in row {row_index}")]
    MissingColumn {
        /// Column name that was not found
        column: String,
        /// Zero-based index of the row where the error occurred
        row_index: usize,
    },

    /// A template contained an unresolvable reference
    #[error("Template '{template}' references unknown column '{column}' in row {row_index}")]
    UnresolvableTemplate {
        /// The template pattern string
        template: String,
        /// Column name referenced in the template
        column: String,
        /// Zero-based index of the row
        row_index: usize,
    },

    /// An IRI generated from a template was syntactically invalid
    #[error("Invalid IRI generated from template '{template}': '{iri}'")]
    InvalidIri {
        /// The template pattern
        template: String,
        /// The invalid IRI that was generated
        iri: String,
    },

    /// A predicate IRI was syntactically invalid
    #[error("Invalid predicate IRI: '{iri}'")]
    InvalidPredicateIri {
        /// The invalid IRI
        iri: String,
    },

    /// An object IRI was syntactically invalid
    #[error("Invalid object IRI: '{iri}'")]
    InvalidObjectIri {
        /// The invalid IRI
        iri: String,
    },

    /// JSON parsing failed
    #[error("JSON parse error: {message}")]
    JsonParseError {
        /// Human-readable description of the parse failure
        message: String,
    },

    /// CSV parsing failed
    #[error("CSV parse error at line {line}: {message}")]
    CsvParseError {
        /// Line number where the error occurred (1-based)
        line: usize,
        /// Human-readable description of the parse failure
        message: String,
    },

    /// A JSON path expression was invalid or matched no data
    #[error("JSON path '{path}' did not match any array in the document")]
    JsonPathNoMatch {
        /// The JSON path that failed to match
        path: String,
    },

    /// No rows could be extracted from the data source
    #[error("Data source produced no rows")]
    EmptyDataSource,

    /// Core RDF model error
    #[error("RDF model error: {0}")]
    RdfModelError(String),
}

/// Convenience type alias for mapping results
pub type MappingResult<T> = Result<T, MappingError>;

// ─── Core Data Types ──────────────────────────────────────────────────────────

/// A data row represented as a map from column name to string value
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    /// Map from column (field) name to value
    pub values: HashMap<String, String>,
}

impl Row {
    /// Create a new empty row
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Create a row from an iterator of (key, value) pairs
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, String)>) -> Self {
        Self {
            values: pairs.into_iter().collect(),
        }
    }

    /// Get a value by column name
    pub fn get(&self, column: &str) -> Option<&str> {
        self.values.get(column).map(String::as_str)
    }

    /// Check whether a column exists (even if empty)
    pub fn contains(&self, column: &str) -> bool {
        self.values.contains_key(column)
    }

    /// Return an iterator over all (column, value) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.values.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

impl Default for Row {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Row {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut entries: Vec<_> = self.values.iter().collect();
        entries.sort_by_key(|(k, _)| k.as_str());
        write!(f, "{{")?;
        for (i, (k, v)) in entries.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{k}: {v}")?;
        }
        write!(f, "}}")
    }
}

// ─── Template ────────────────────────────────────────────────────────────────

/// A URI template that can be rendered using row values.
///
/// Template syntax: literal text with `{column_name}` placeholders.
/// Column references are URL-percent-encoded to produce valid IRIs.
///
/// # Example
///
/// ```
/// use oxirs_ttl::mapping::{Template, Row};
/// use std::collections::HashMap;
///
/// let tpl = Template::new("http://example.org/{id}/profile");
/// let mut row = Row::new();
/// row.values.insert("id".to_string(), "42".to_string());
/// let iri = tpl.render(&row, 0).expect("should succeed");
/// assert_eq!(iri, "http://example.org/42/profile");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Template {
    /// The raw pattern string, e.g. `"http://example.org/{column_name}"`
    pub pattern: String,
}

impl Template {
    /// Create a new template from any string-like value
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
        }
    }

    /// Render the template by substituting `{column}` placeholders with
    /// percent-encoded values from `row`.
    ///
    /// Returns an error if a referenced column is absent from the row.
    pub fn render(&self, row: &Row, row_index: usize) -> MappingResult<String> {
        let mut output = String::with_capacity(self.pattern.len() + 32);
        let mut chars = self.pattern.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                // Collect column name until '}'
                let mut col_name = String::new();
                let mut closed = false;
                for inner in chars.by_ref() {
                    if inner == '}' {
                        closed = true;
                        break;
                    }
                    col_name.push(inner);
                }
                if !closed {
                    // Treat un-closed brace as literal text
                    output.push('{');
                    output.push_str(&col_name);
                    continue;
                }
                let value =
                    row.get(&col_name)
                        .ok_or_else(|| MappingError::UnresolvableTemplate {
                            template: self.pattern.clone(),
                            column: col_name.clone(),
                            row_index,
                        })?;
                percent_encode_path(value, &mut output);
            } else {
                output.push(ch);
            }
        }
        Ok(output)
    }
}

impl fmt::Display for Template {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.pattern)
    }
}

/// Percent-encode characters that are illegal in IRI paths.
/// RFC 3986 unreserved chars and common path chars are kept as-is.
pub(crate) fn percent_encode_path(input: &str, out: &mut String) {
    for byte in input.bytes() {
        match byte {
            // RFC 3986 unreserved characters
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            // Additional IRI-safe characters
            b':' | b'@' | b'!' | b'$' | b'&' | b'\'' | b'(' | b')' | b'*' | b'+' | b',' | b';'
            | b'=' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push(hex_nibble(byte >> 4));
                out.push(hex_nibble(byte & 0x0F));
            }
        }
    }
}

#[inline]
pub(crate) fn hex_nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'A' + n - 10) as char,
        _ => '0',
    }
}

// ─── ObjectSpec ──────────────────────────────────────────────────────────────

/// Specifies how the object of a triple should be produced from a data row
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectSpec {
    /// Generate an IRI by rendering a template against the row
    Template(Template),

    /// Use the column value as a plain string literal (`xsd:string`)
    Column(String),

    /// Use a constant string — always the same value regardless of the row
    Constant(String),

    /// Use the column value as a typed literal
    TypedColumn {
        /// Column name containing the lexical value
        column: String,
        /// Full XSD datatype IRI (e.g. `"http://www.w3.org/2001/XMLSchema#integer"`)
        datatype: String,
    },

    /// Use the column value as a language-tagged literal; the language tag is
    /// taken from another column.
    LangColumn {
        /// Column name containing the literal text
        column: String,
        /// Column name containing the BCP 47 language tag (e.g. `"en"`)
        lang_column: String,
    },

    /// Use the column value as a language-tagged literal with a fixed language
    LangFixed {
        /// Column name containing the literal text
        column: String,
        /// BCP 47 language tag (e.g. `"en"`)
        lang: String,
    },

    /// Use a constant IRI value (no template substitution)
    ConstantIri(String),
}

// ─── PredicateObjectMap ───────────────────────────────────────────────────────

/// Pairs a predicate IRI with an [`ObjectSpec`] to produce a single
/// predicate-object component of a triple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateObjectMap {
    /// Full IRI for the predicate
    pub predicate: String,
    /// Specification for how to produce the object
    pub object_template: ObjectSpec,
}

impl PredicateObjectMap {
    /// Create a new predicate-object map
    pub fn new(predicate: impl Into<String>, object_template: ObjectSpec) -> Self {
        Self {
            predicate: predicate.into(),
            object_template,
        }
    }
}

// ─── DataSource ───────────────────────────────────────────────────────────────

/// The origin of the data to be mapped to RDF
#[derive(Debug, Clone)]
pub enum DataSource {
    /// CSV text with configurable delimiter
    Csv {
        /// The raw CSV content
        content: String,
        /// Field delimiter character (usually `','`)
        delimiter: char,
    },

    /// JSON text, optionally with a dot-separated path to the target array
    Json {
        /// The raw JSON content
        content: String,
        /// Optional dot-separated path to the array of objects (e.g. `"results.bindings"`)
        json_path: Option<String>,
    },

    /// Pre-parsed rows supplied directly as vectors
    InlineValues {
        /// Rows of string values
        rows: Vec<Vec<String>>,
        /// Column headers that correspond positionally to the values in each row
        headers: Vec<String>,
    },
}

// ─── MappingRule ──────────────────────────────────────────────────────────────

/// A complete mapping rule that produces RDF triples from a [`DataSource`]
#[derive(Debug, Clone)]
pub struct MappingRule {
    /// Human-readable name for this rule (used in error messages)
    pub name: String,
    /// The data source to read rows from
    pub source: DataSource,
    /// Template for generating the subject IRI of each triple
    pub subject_template: Template,
    /// List of predicate-object pairs to generate for each row
    pub predicate_object_maps: Vec<PredicateObjectMap>,
    /// Optional named graph to assign all generated triples to
    pub graph_name: Option<String>,
}

impl MappingRule {
    /// Create a minimal mapping rule (use [`MappingRuleBuilder`] for ergonomic construction)
    pub fn new(name: impl Into<String>, source: DataSource, subject_template: Template) -> Self {
        Self {
            name: name.into(),
            source,
            subject_template,
            predicate_object_maps: Vec::new(),
            graph_name: None,
        }
    }

    /// Add a predicate-object map to this rule
    pub fn add_predicate_object_map(&mut self, pom: PredicateObjectMap) {
        self.predicate_object_maps.push(pom);
    }
}

// ─── Object resolution helper (shared with engine) ────────────────────────────

/// Resolve an [`ObjectSpec`] against a data row to produce an RDF [`Object`].
///
/// This is a free function so that both `MappingEngine` (in `mapping_transformers`)
/// and any future code can reuse the logic without duplicating it.
pub fn resolve_object_spec(spec: &ObjectSpec, row: &Row, row_idx: usize) -> MappingResult<Object> {
    use oxirs_core::model::Literal;
    match spec {
        ObjectSpec::Template(tpl) => {
            let iri = tpl.render(row, row_idx)?;
            let node = NamedNode::new(&iri)
                .map_err(|_| MappingError::InvalidObjectIri { iri: iri.clone() })?;
            Ok(Object::NamedNode(node))
        }

        ObjectSpec::Column(col) => {
            let value = row.get(col).ok_or_else(|| MappingError::MissingColumn {
                column: col.clone(),
                row_index: row_idx,
            })?;
            Ok(Object::Literal(Literal::new(value)))
        }

        ObjectSpec::Constant(value) => Ok(Object::Literal(Literal::new(value))),

        ObjectSpec::TypedColumn { column, datatype } => {
            let value = row.get(column).ok_or_else(|| MappingError::MissingColumn {
                column: column.clone(),
                row_index: row_idx,
            })?;
            let dt_node = NamedNode::new(datatype).map_err(|_| MappingError::InvalidObjectIri {
                iri: datatype.clone(),
            })?;
            Ok(Object::Literal(Literal::new_typed_literal(value, dt_node)))
        }

        ObjectSpec::LangColumn {
            column,
            lang_column,
        } => {
            let value = row.get(column).ok_or_else(|| MappingError::MissingColumn {
                column: column.clone(),
                row_index: row_idx,
            })?;
            let lang = row
                .get(lang_column)
                .ok_or_else(|| MappingError::MissingColumn {
                    column: lang_column.clone(),
                    row_index: row_idx,
                })?;
            let lit = oxirs_core::model::Literal::new_language_tagged_literal(value, lang)
                .map_err(|e| MappingError::RdfModelError(e.to_string()))?;
            Ok(Object::Literal(lit))
        }

        ObjectSpec::LangFixed { column, lang } => {
            let value = row.get(column).ok_or_else(|| MappingError::MissingColumn {
                column: column.clone(),
                row_index: row_idx,
            })?;
            let lit = oxirs_core::model::Literal::new_language_tagged_literal(value, lang)
                .map_err(|e| MappingError::RdfModelError(e.to_string()))?;
            Ok(Object::Literal(lit))
        }

        ObjectSpec::ConstantIri(iri) => {
            let node = NamedNode::new(iri)
                .map_err(|_| MappingError::InvalidObjectIri { iri: iri.clone() })?;
            Ok(Object::NamedNode(node))
        }
    }
}

/// Build a single [`Triple`] from its components and a predicate-object map.
pub fn build_triple_from_pom(
    subject: &Subject,
    pom: &PredicateObjectMap,
    row: &Row,
    row_idx: usize,
) -> MappingResult<Triple> {
    let pred_node =
        NamedNode::new(&pom.predicate).map_err(|_| MappingError::InvalidPredicateIri {
            iri: pom.predicate.clone(),
        })?;
    let predicate: Predicate = pred_node.into();
    let object = resolve_object_spec(&pom.object_template, row, row_idx)?;
    Ok(Triple::new(subject.clone(), predicate, object))
}
