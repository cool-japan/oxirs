//! CSVW metadata schema types.
//!
//! Implements a subset of the W3C CSV on the Web (CSVW) Recommendation for
//! describing tabular data files with JSON-LD metadata.
//!
//! References:
//! - <https://www.w3.org/TR/tabular-data-primer/>
//! - <https://www.w3.org/TR/tabular-metadata/>

use serde::{Deserialize, Serialize};

use super::CsvwError;

/// CSVW table-level metadata (subset of the W3C spec).
///
/// Describes a single CSV file with its column schema and URI templates.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CsvwMetadata {
    /// The URL of the CSV file this metadata describes (optional).
    #[serde(rename = "@id", default)]
    pub id: Option<String>,

    /// Schema for the table.
    #[serde(rename = "tableSchema", default)]
    pub table_schema: TableSchema,

    /// URL template for generating subject URIs for each row.
    ///
    /// Uses `{column_name}` substitutions. Default: auto-incrementing blank node
    /// of the form `_:rowN`.
    #[serde(rename = "aboutUrl", default)]
    pub about_url: Option<String>,
}

/// Schema for a table — column definitions and primary key.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TableSchema {
    /// Column definitions in order.
    #[serde(default)]
    pub columns: Vec<ColumnDef>,

    /// Column names that form the primary key.
    #[serde(rename = "primaryKey", default)]
    pub primary_key: Vec<String>,
}

/// Definition of a single column within a CSVW table schema.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColumnDef {
    /// Column name (matches CSV header).
    pub name: String,

    /// XSD datatype URI for the column values.
    ///
    /// Default when absent: `http://www.w3.org/2001/XMLSchema#string`.
    #[serde(rename = "datatype", default)]
    pub datatype: Option<String>,

    /// Property URI to use as predicate when mapping to RDF.
    ///
    /// When absent the column name is used relative to the base IRI.
    #[serde(rename = "propertyUrl", default)]
    pub property_url: Option<String>,

    /// When `true` this column is omitted from RDF output.
    #[serde(rename = "suppressOutput", default)]
    pub suppress_output: bool,

    /// Human-readable label for the column.
    #[serde(default)]
    pub titles: Option<String>,
}

impl CsvwMetadata {
    /// Parse CSVW metadata from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`CsvwError::JsonError`] if the JSON is malformed or does not
    /// match the expected schema shape.
    pub fn from_json(json: &str) -> Result<Self, CsvwError> {
        serde_json::from_str(json).map_err(CsvwError::from)
    }

    /// Find a [`ColumnDef`] by its `name` field.
    ///
    /// Returns `None` if no column with that name exists.
    pub fn column(&self, name: &str) -> Option<&ColumnDef> {
        self.table_schema.columns.iter().find(|c| c.name == name)
    }
}
