//! CSV-on-the-Web → RDF triple conversion.
//!
//! Converts parsed CSV records to [`RdfStatement`] triples using a
//! [`CsvwMetadata`] schema for subject URI generation, predicate mapping,
//! and typed literal construction.

use super::{reader::CsvRecord, schema::CsvwMetadata, CsvwError};

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// A simplified RDF statement produced by the CSVW converter.
///
/// Subject, predicate and object are represented as plain strings in the
/// familiar N-Triples surface syntax so that callers can inspect or
/// serialise them without pulling in the full oxirs-core triple model.
#[derive(Debug, Clone, PartialEq)]
pub struct RdfStatement {
    /// Subject IRI (`<…>`) or blank node (`_:…`).
    pub subject: String,
    /// Predicate IRI (`<…>`).
    pub predicate: String,
    /// Object literal (`"…"^^<…>`, `"…"@tag`) or IRI (`<…>`).
    pub object: String,
}

/// Configuration for the CSVW → RDF conversion pass.
#[derive(Debug, Clone)]
pub struct CsvwConverterConfig {
    /// Base IRI used when constructing blank-node-like row subjects and when
    /// a column name is used as a relative predicate.
    pub base_iri: String,
    /// Row index to start counting from when minting subject IRIs.
    pub start_row: usize,
}

impl Default for CsvwConverterConfig {
    fn default() -> Self {
        Self {
            base_iri: "http://example.org/row/".into(),
            start_row: 1,
        }
    }
}

/// CSVW → RDF triple converter.
pub struct CsvwConverter {
    config: CsvwConverterConfig,
}

impl CsvwConverter {
    /// Construct a converter with the given configuration.
    pub fn new(config: CsvwConverterConfig) -> Self {
        Self { config }
    }

    // ── public API ──────────────────────────────────────────────────────────

    /// Convert a set of CSV records to RDF statements using the provided
    /// CSVW metadata schema.
    ///
    /// Each non-suppressed column in each record produces one [`RdfStatement`].
    /// The column order in `headers` is matched to `metadata.table_schema.columns`
    /// by name; unmapped headers fall back to a generated predicate IRI.
    ///
    /// # Errors
    ///
    /// Returns [`CsvwError::ConversionError`] if the field count of a record
    /// does not match the header count.
    pub fn convert(
        &self,
        headers: &[String],
        records: &[CsvRecord],
        metadata: &CsvwMetadata,
    ) -> Result<Vec<RdfStatement>, CsvwError> {
        let mut statements = Vec::new();

        for (row_idx, record) in records.iter().enumerate() {
            if record.fields.len() != headers.len() {
                return Err(CsvwError::ConversionError(format!(
                    "row {} has {} fields but header has {}",
                    self.config.start_row + row_idx,
                    record.fields.len(),
                    headers.len(),
                )));
            }

            let subject =
                self.subject_for_row(self.config.start_row + row_idx, headers, record, metadata);

            for (col_idx, header) in headers.iter().enumerate() {
                // Look up column definition (may not exist for all headers).
                let col_def = metadata.column(header);

                // Respect suppressOutput.
                if col_def.map(|c| c.suppress_output).unwrap_or(false) {
                    continue;
                }

                let predicate = match col_def {
                    Some(col) => self.predicate_for_column(col),
                    None => format!("<{}{}>", self.config.base_iri, url_encode(header)),
                };

                let datatype = col_def.and_then(|c| c.datatype.as_deref());
                let raw_value = &record.fields[col_idx];
                let object = self.object_for_value(raw_value, datatype);

                statements.push(RdfStatement {
                    subject: subject.clone(),
                    predicate,
                    object,
                });
            }
        }

        Ok(statements)
    }

    // ── private helpers ─────────────────────────────────────────────────────

    /// Generate a subject IRI or blank node for the given row.
    ///
    /// If `metadata.about_url` is set the template is expanded:
    /// each `{column_name}` placeholder is replaced with the corresponding
    /// field value from the record.  Otherwise a row-numbered IRI is minted
    /// using `config.base_iri`.
    pub(crate) fn subject_for_row(
        &self,
        row_index: usize,
        headers: &[String],
        record: &CsvRecord,
        metadata: &CsvwMetadata,
    ) -> String {
        match &metadata.about_url {
            Some(template) => {
                let mut expanded = template.clone();
                for (idx, header) in headers.iter().enumerate() {
                    let placeholder = format!("{{{header}}}");
                    if let Some(value) = record.fields.get(idx) {
                        expanded = expanded.replace(&placeholder, &url_encode(value));
                    }
                }
                // Wrap in angle brackets if it looks like an IRI.
                if expanded.starts_with("http://") || expanded.starts_with("https://") {
                    format!("<{expanded}>")
                } else {
                    expanded
                }
            }
            None => {
                // Default: auto-generated IRI using base_iri + row number.
                format!("<{}{}>", self.config.base_iri, row_index)
            }
        }
    }

    /// Build the predicate IRI string for a column definition.
    ///
    /// Uses `propertyUrl` when present; otherwise constructs a predicate
    /// from `base_iri` + percent-encoded column name.
    pub(crate) fn predicate_for_column(&self, col: &super::schema::ColumnDef) -> String {
        match &col.property_url {
            Some(url) => {
                if url.starts_with('<') {
                    url.clone()
                } else {
                    format!("<{url}>")
                }
            }
            None => format!("<{}{}>", self.config.base_iri, url_encode(&col.name)),
        }
    }

    /// Build an RDF object string from a raw CSV value and an optional
    /// XSD datatype URI.
    ///
    /// - If `datatype` is `None` the value is emitted as a plain `xsd:string`
    ///   typed literal.
    /// - If the datatype is given it is attached as a `^^<type>` suffix.
    /// - Values that are already IRI-shaped (`http://…` / `https://…`) with
    ///   no datatype are wrapped in angle brackets.
    pub(crate) fn object_for_value(&self, value: &str, datatype: Option<&str>) -> String {
        // Escape the literal value for N-Triples embedding.
        let escaped = escape_literal(value);

        match datatype {
            Some(dt) => {
                // Attach the provided datatype.
                let dt_iri = if dt.starts_with('<') {
                    dt.to_owned()
                } else {
                    format!("<{dt}>")
                };
                format!("\"{escaped}\"^^{dt_iri}")
            }
            None => {
                // Default: xsd:string typed literal.
                format!("\"{escaped}\"^^<http://www.w3.org/2001/XMLSchema#string>")
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Utility functions
// ────────────────────────────────────────────────────────────────────────────

/// Minimal percent-encoding for characters that are unsafe in IRI path segments.
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'_'
            | b'.'
            | b'~'
            | b':'
            | b'/'
            | b'@'
            | b'!'
            | b'$'
            | b'&'
            | b'\''
            | b'('
            | b')'
            | b'*'
            | b'+'
            | b','
            | b';'
            | b'=' => out.push(b as char),
            other => {
                out.push('%');
                let hi = (other >> 4) & 0x0F;
                let lo = other & 0x0F;
                out.push(HEX_CHARS[hi as usize] as char);
                out.push(HEX_CHARS[lo as usize] as char);
            }
        }
    }
    out
}

const HEX_CHARS: &[u8] = b"0123456789ABCDEF";

/// Escape a literal value for inclusion between N-Triples double-quotes.
///
/// Handles the five mandatory escapes: `\n`, `\r`, `\t`, `\\`, `\"`.
fn escape_literal(s: &str) -> String {
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
