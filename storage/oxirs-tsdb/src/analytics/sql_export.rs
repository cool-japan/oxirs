//! DuckDB-compatible SQL export for TSDB time-series data.
//!
//! Generates `CREATE TABLE` DDL and `INSERT` DML statements from TSDB metric
//! data so that recordings can be loaded into any SQL engine — including
//! DuckDB, SQLite, or PostgreSQL — without an external dependency.
//!
//! ## Usage
//!
//! ```rust
//! use oxirs_tsdb::analytics::sql_export::{SqlExporter, MetricSchema, DataValueType};
//!
//! // Describe the schema of a metric.
//! let schema = MetricSchema::builder("cpu_usage")
//!     .with_tag("host")
//!     .with_tag("region")
//!     .build();
//!
//! // Generate SQL.
//! let exporter = SqlExporter::new();
//! let ddl = exporter.create_table_sql(&schema);
//! assert!(ddl.contains("CREATE TABLE"));
//! assert!(ddl.contains("cpu_usage"));
//! ```

use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::path::Path;

// =============================================================================
// DataValueType
// =============================================================================

/// The SQL column type used to represent a metric's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DataValueType {
    /// 64-bit IEEE 754 floating-point (default).
    #[default]
    Float64,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit IEEE 754 floating-point.
    Float32,
    /// UTF-8 text (for string-valued metrics or enumerations).
    Text,
}

impl DataValueType {
    /// Return the SQL type keyword for this variant.
    pub fn sql_type(&self) -> &'static str {
        match self {
            DataValueType::Float64 => "DOUBLE",
            DataValueType::Int64 => "BIGINT",
            DataValueType::Float32 => "FLOAT",
            DataValueType::Text => "VARCHAR",
        }
    }
}

impl std::fmt::Display for DataValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.sql_type())
    }
}

// =============================================================================
// MetricSchema
// =============================================================================

/// Describes the SQL schema of a TSDB metric.
///
/// A metric has:
/// - A mandatory `timestamp_ms BIGINT NOT NULL` column.
/// - A mandatory `value <DataValueType> NOT NULL` column.
/// - Zero or more `VARCHAR` tag columns (one per tag key).
///
/// The resulting table name is `"<metric_name>"` (or a sanitised version if the
/// name contains special characters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSchema {
    /// Name of the metric (used as the SQL table name).
    pub metric_name: String,
    /// Ordered list of tag key names.
    pub tag_keys: Vec<String>,
    /// SQL type for the `value` column.
    pub value_type: DataValueType,
    /// Optional human-readable description (stored as a SQL comment).
    pub description: Option<String>,
}

impl MetricSchema {
    /// Create a `MetricSchemaBuilder` for `metric_name`.
    pub fn builder(metric_name: impl Into<String>) -> MetricSchemaBuilder {
        MetricSchemaBuilder::new(metric_name)
    }

    /// Return a sanitised SQL table name derived from `metric_name`.
    ///
    /// Non-alphanumeric / non-underscore characters are replaced with `_`.
    pub fn table_name(&self) -> String {
        sanitize_sql_identifier(&self.metric_name)
    }

    /// Return `true` if this schema has at least one tag column.
    pub fn has_tags(&self) -> bool {
        !self.tag_keys.is_empty()
    }

    /// Return the total number of columns (2 fixed + tags).
    pub fn column_count(&self) -> usize {
        2 + self.tag_keys.len()
    }
}

// =============================================================================
// MetricSchemaBuilder
// =============================================================================

/// Fluent builder for [`MetricSchema`].
pub struct MetricSchemaBuilder {
    metric_name: String,
    tag_keys: Vec<String>,
    value_type: DataValueType,
    description: Option<String>,
}

impl MetricSchemaBuilder {
    /// Create a builder with default settings.
    pub fn new(metric_name: impl Into<String>) -> Self {
        Self {
            metric_name: metric_name.into(),
            tag_keys: Vec::new(),
            value_type: DataValueType::Float64,
            description: None,
        }
    }

    /// Add a tag key column.
    pub fn with_tag(mut self, key: impl Into<String>) -> Self {
        self.tag_keys.push(key.into());
        self
    }

    /// Set the value column type.
    pub fn with_value_type(mut self, vt: DataValueType) -> Self {
        self.value_type = vt;
        self
    }

    /// Set an optional description comment.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Consume the builder and return the finished [`MetricSchema`].
    pub fn build(self) -> MetricSchema {
        MetricSchema {
            metric_name: self.metric_name,
            tag_keys: self.tag_keys,
            value_type: self.value_type,
            description: self.description,
        }
    }
}

// =============================================================================
// DataPoint (SQL export local type)
// =============================================================================

/// A single time-series point ready for SQL export.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SqlDataPoint {
    /// Unix epoch milliseconds.
    pub timestamp_ms: i64,
    /// Observed value.
    pub value: f64,
    /// Tag key-value pairs (must match the tag keys in the associated schema).
    pub tags: HashMap<String, String>,
}

impl SqlDataPoint {
    /// Create a point without tags.
    pub fn new(timestamp_ms: i64, value: f64) -> Self {
        Self {
            timestamp_ms,
            value,
            tags: HashMap::new(),
        }
    }

    /// Create a point with a tag map.
    pub fn with_tags(timestamp_ms: i64, value: f64, tags: HashMap<String, String>) -> Self {
        Self {
            timestamp_ms,
            value,
            tags,
        }
    }
}

// =============================================================================
// SqlExporter
// =============================================================================

/// Generates SQL DDL and DML statements for TSDB data.
///
/// `SqlExporter` is stateless: each method takes the relevant inputs and
/// returns a `String`.  No database connection is required.
#[derive(Debug, Clone, Default)]
pub struct SqlExporter {
    /// Number of rows per batched `INSERT` statement (0 = all rows in one statement).
    pub batch_size: usize,
    /// When `true`, a `DROP TABLE IF EXISTS` is emitted before `CREATE TABLE`.
    pub drop_existing: bool,
    /// When `true`, `IF NOT EXISTS` is added to `CREATE TABLE`.
    pub if_not_exists: bool,
}

impl SqlExporter {
    /// Create an exporter with sensible defaults.
    pub fn new() -> Self {
        Self {
            batch_size: 1000,
            drop_existing: false,
            if_not_exists: true,
        }
    }

    /// Set batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Include `DROP TABLE IF EXISTS` before `CREATE TABLE`.
    pub fn with_drop_existing(mut self, drop: bool) -> Self {
        self.drop_existing = drop;
        self
    }

    // -------------------------------------------------------------------------
    // DDL generation
    // -------------------------------------------------------------------------

    /// Generate a `CREATE TABLE` statement for the given schema.
    pub fn create_table_sql(&self, schema: &MetricSchema) -> String {
        let mut sql = String::new();

        if let Some(desc) = &schema.description {
            let _ = writeln!(sql, "-- {desc}");
        }

        if self.drop_existing {
            let _ = writeln!(
                sql,
                "DROP TABLE IF EXISTS {table};",
                table = schema.table_name()
            );
        }

        let if_not_exists = if self.if_not_exists {
            "IF NOT EXISTS "
        } else {
            ""
        };

        let _ = write!(
            sql,
            "CREATE TABLE {if_not_exists}{table} (\n    timestamp_ms BIGINT NOT NULL,\n    value {vtype} NOT NULL",
            table = schema.table_name(),
            vtype = schema.value_type.sql_type(),
        );

        for tag in &schema.tag_keys {
            let _ = write!(sql, ",\n    {} VARCHAR", sanitize_sql_identifier(tag));
        }

        let _ = write!(sql, "\n);");
        sql
    }

    /// Generate a batched `INSERT INTO` statement for a slice of points.
    ///
    /// Points are split into chunks of `batch_size` (or all at once if
    /// `batch_size == 0`).  Returns a `Vec` of SQL strings, one per batch.
    pub fn insert_sql(&self, schema: &MetricSchema, points: &[SqlDataPoint]) -> Vec<String> {
        if points.is_empty() {
            return vec![];
        }

        let chunk_size = if self.batch_size == 0 {
            points.len()
        } else {
            self.batch_size
        };

        let table = schema.table_name();
        let tag_cols: Vec<String> = schema
            .tag_keys
            .iter()
            .map(|k| sanitize_sql_identifier(k))
            .collect();

        // Column list header
        let col_list = if tag_cols.is_empty() {
            "timestamp_ms, value".to_string()
        } else {
            format!("timestamp_ms, value, {}", tag_cols.join(", "))
        };

        points
            .chunks(chunk_size)
            .map(|chunk| {
                let mut sql = format!("INSERT INTO {table} ({col_list}) VALUES\n");
                let mut first = true;
                for pt in chunk {
                    if !first {
                        sql.push_str(",\n");
                    }
                    first = false;
                    if tag_cols.is_empty() {
                        let _ =
                            write!(sql, "    ({}, {})", pt.timestamp_ms, escape_float(pt.value));
                    } else {
                        let tag_vals: Vec<String> = schema
                            .tag_keys
                            .iter()
                            .map(|k| {
                                pt.tags
                                    .get(k)
                                    .map(|v| format!("'{}'", v.replace('\'', "''")))
                                    .unwrap_or_else(|| "NULL".to_string())
                            })
                            .collect();
                        let _ = write!(
                            sql,
                            "    ({}, {}, {})",
                            pt.timestamp_ms,
                            escape_float(pt.value),
                            tag_vals.join(", ")
                        );
                    }
                }
                sql.push(';');
                sql
            })
            .collect()
    }

    /// Generate a complete SQL script (DDL + DML) and write it to `path`.
    ///
    /// The script contains:
    /// 1. `CREATE TABLE` (and optionally `DROP TABLE IF EXISTS`).
    /// 2. All `INSERT INTO` batches.
    pub fn export_to_sql_file(
        &self,
        schema: &MetricSchema,
        points: &[SqlDataPoint],
        path: &Path,
    ) -> TsdbResult<()> {
        use std::io::Write as IoWrite;
        let file = std::fs::File::create(path).map_err(|e| TsdbError::Io(e.to_string()))?;
        let mut writer = std::io::BufWriter::new(file);

        writeln!(writer, "{}", self.create_table_sql(schema))
            .map_err(|e| TsdbError::Io(e.to_string()))?;

        for batch in self.insert_sql(schema, points) {
            writeln!(writer, "{batch}").map_err(|e| TsdbError::Io(e.to_string()))?;
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Additional helpers
    // -------------------------------------------------------------------------

    /// Generate a `SELECT` query for a metric within a time range.
    pub fn select_range_sql(&self, schema: &MetricSchema, start_ms: i64, end_ms: i64) -> String {
        let tag_cols = if schema.tag_keys.is_empty() {
            String::new()
        } else {
            let cols: Vec<String> = schema
                .tag_keys
                .iter()
                .map(|k| sanitize_sql_identifier(k))
                .collect();
            format!(", {}", cols.join(", "))
        };
        format!(
            "SELECT timestamp_ms, value{tag_cols} \
             FROM {table} \
             WHERE timestamp_ms BETWEEN {start} AND {end} \
             ORDER BY timestamp_ms ASC;",
            table = schema.table_name(),
            start = start_ms,
            end = end_ms,
            tag_cols = tag_cols,
        )
    }

    /// Generate a `SELECT COUNT(*)` query.
    pub fn count_sql(&self, schema: &MetricSchema) -> String {
        format!(
            "SELECT COUNT(*) AS row_count FROM {table};",
            table = schema.table_name()
        )
    }

    /// Generate a `SELECT MIN/MAX/AVG(value)` summary query.
    pub fn summary_sql(&self, schema: &MetricSchema) -> String {
        format!(
            "SELECT \
               MIN(timestamp_ms) AS first_ts, \
               MAX(timestamp_ms) AS last_ts, \
               MIN(value) AS min_val, \
               MAX(value) AS max_val, \
               AVG(value) AS avg_val, \
               COUNT(*) AS row_count \
             FROM {table};",
            table = schema.table_name()
        )
    }

    /// Generate a `DELETE FROM` statement to purge data older than `before_ms`.
    pub fn delete_before_sql(&self, schema: &MetricSchema, before_ms: i64) -> String {
        format!(
            "DELETE FROM {table} WHERE timestamp_ms < {before_ms};",
            table = schema.table_name()
        )
    }

    /// Generate a `CREATE INDEX` for fast time-range queries.
    pub fn create_index_sql(&self, schema: &MetricSchema) -> String {
        format!(
            "CREATE INDEX IF NOT EXISTS idx_{table}_ts ON {table} (timestamp_ms);",
            table = schema.table_name()
        )
    }

    /// Generate an `ALTER TABLE ... ADD COLUMN` for a new tag.
    pub fn add_tag_column_sql(&self, schema: &MetricSchema, tag_key: &str) -> String {
        format!(
            "ALTER TABLE {table} ADD COLUMN {col} VARCHAR;",
            table = schema.table_name(),
            col = sanitize_sql_identifier(tag_key)
        )
    }

    /// Infer a [`MetricSchema`] from a slice of [`SqlDataPoint`]s and a metric name.
    pub fn infer_schema(metric_name: &str, points: &[SqlDataPoint]) -> MetricSchema {
        let mut tag_keys: Vec<String> = points
            .iter()
            .flat_map(|p| p.tags.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        tag_keys.sort();
        MetricSchema::builder(metric_name)
            .with_value_type(DataValueType::Float64)
            .build()
            .with_sorted_tags(tag_keys)
    }
}

// =============================================================================
// Private helpers
// =============================================================================

/// Sanitise an identifier by replacing invalid characters with underscores.
fn sanitize_sql_identifier(name: &str) -> String {
    if name.is_empty() {
        return "_".to_string();
    }
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    // If the first char is a digit, prepend an underscore.
    if out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

/// Format a floating-point value as a SQL literal.
///
/// - `f64::INFINITY`  → `'Infinity'`
/// - `f64::NEG_INFINITY` → `'-Infinity'`
/// - `f64::NAN` → `'NaN'`
/// - Otherwise → the standard decimal representation.
fn escape_float(v: f64) -> String {
    if v.is_nan() {
        "'NaN'".to_string()
    } else if v.is_infinite() && v > 0.0 {
        "'Infinity'".to_string()
    } else if v.is_infinite() {
        "'-Infinity'".to_string()
    } else {
        format!("{v}")
    }
}

// Extension method on MetricSchema to apply a pre-sorted tag list.
impl MetricSchema {
    fn with_sorted_tags(mut self, tags: Vec<String>) -> Self {
        self.tag_keys = tags;
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schema() -> MetricSchema {
        MetricSchema::builder("cpu_usage")
            .with_tag("host")
            .with_tag("region")
            .with_description("CPU utilisation percent")
            .build()
    }

    fn sample_points(n: usize) -> Vec<SqlDataPoint> {
        (0..n)
            .map(|i| {
                let mut tags = HashMap::new();
                tags.insert("host".to_string(), format!("srv-{i:02}"));
                tags.insert("region".to_string(), "eu-west".to_string());
                SqlDataPoint::with_tags(i as i64 * 1_000, i as f64 * 1.5, tags)
            })
            .collect()
    }

    // -- DataValueType --------------------------------------------------------

    #[test]
    fn test_value_type_sql_types() {
        assert_eq!(DataValueType::Float64.sql_type(), "DOUBLE");
        assert_eq!(DataValueType::Int64.sql_type(), "BIGINT");
        assert_eq!(DataValueType::Float32.sql_type(), "FLOAT");
        assert_eq!(DataValueType::Text.sql_type(), "VARCHAR");
    }

    #[test]
    fn test_value_type_display() {
        assert_eq!(format!("{}", DataValueType::Float64), "DOUBLE");
        assert_eq!(format!("{}", DataValueType::Text), "VARCHAR");
    }

    #[test]
    fn test_value_type_default() {
        assert_eq!(DataValueType::default(), DataValueType::Float64);
    }

    // -- MetricSchema / MetricSchemaBuilder -----------------------------------

    #[test]
    fn test_metric_schema_table_name_clean() {
        let schema = MetricSchema::builder("cpu_usage").build();
        assert_eq!(schema.table_name(), "cpu_usage");
    }

    #[test]
    fn test_metric_schema_table_name_special_chars() {
        let schema = MetricSchema::builder("my-metric.v2").build();
        let name = schema.table_name();
        // Hyphens and dots should be replaced.
        assert!(!name.contains('-'));
        assert!(!name.contains('.'));
    }

    #[test]
    fn test_metric_schema_has_tags() {
        let no_tags = MetricSchema::builder("x").build();
        assert!(!no_tags.has_tags());

        let with_tags = MetricSchema::builder("x").with_tag("host").build();
        assert!(with_tags.has_tags());
    }

    #[test]
    fn test_metric_schema_column_count() {
        let s = MetricSchema::builder("m")
            .with_tag("a")
            .with_tag("b")
            .build();
        // 2 fixed + 2 tags = 4
        assert_eq!(s.column_count(), 4);
    }

    #[test]
    fn test_metric_schema_builder_value_type() {
        let schema = MetricSchema::builder("count")
            .with_value_type(DataValueType::Int64)
            .build();
        assert_eq!(schema.value_type, DataValueType::Int64);
    }

    // -- SqlExporter DDL generation ------------------------------------------

    #[test]
    fn test_create_table_sql_contains_table_name() {
        let schema = sample_schema();
        let exporter = SqlExporter::new();
        let sql = exporter.create_table_sql(&schema);
        assert!(sql.contains("cpu_usage"), "sql = {sql}");
    }

    #[test]
    fn test_create_table_sql_has_timestamp_and_value() {
        let schema = sample_schema();
        let exporter = SqlExporter::new();
        let sql = exporter.create_table_sql(&schema);
        assert!(sql.contains("timestamp_ms BIGINT NOT NULL"));
        assert!(sql.contains("value DOUBLE NOT NULL"));
    }

    #[test]
    fn test_create_table_sql_has_tag_columns() {
        let schema = sample_schema();
        let exporter = SqlExporter::new();
        let sql = exporter.create_table_sql(&schema);
        assert!(sql.contains("host VARCHAR"));
        assert!(sql.contains("region VARCHAR"));
    }

    #[test]
    fn test_create_table_sql_no_tags() {
        let schema = MetricSchema::builder("temp").build();
        let exporter = SqlExporter::new();
        let sql = exporter.create_table_sql(&schema);
        assert!(sql.contains("timestamp_ms"));
        assert!(sql.contains("value DOUBLE"));
        assert!(!sql.contains("VARCHAR"));
    }

    #[test]
    fn test_create_table_sql_if_not_exists() {
        let exporter = SqlExporter::new(); // if_not_exists = true by default
        let schema = MetricSchema::builder("m").build();
        let sql = exporter.create_table_sql(&schema);
        assert!(sql.contains("IF NOT EXISTS"));
    }

    #[test]
    fn test_create_table_sql_drop_existing() {
        let exporter = SqlExporter::new().with_drop_existing(true);
        let schema = MetricSchema::builder("m").build();
        let sql = exporter.create_table_sql(&schema);
        assert!(sql.contains("DROP TABLE IF EXISTS"));
    }

    #[test]
    fn test_create_table_sql_description_comment() {
        let schema = MetricSchema::builder("cpu")
            .with_description("CPU usage")
            .build();
        let exporter = SqlExporter::new();
        let sql = exporter.create_table_sql(&schema);
        assert!(sql.contains("-- CPU usage"));
    }

    // -- SqlExporter INSERT generation ----------------------------------------

    #[test]
    fn test_insert_sql_empty_returns_empty_vec() {
        let schema = sample_schema();
        let exporter = SqlExporter::new();
        let batches = exporter.insert_sql(&schema, &[]);
        assert!(batches.is_empty());
    }

    #[test]
    fn test_insert_sql_contains_values() {
        let schema = MetricSchema::builder("temp").build();
        let points = vec![
            SqlDataPoint::new(1_000, 22.5),
            SqlDataPoint::new(2_000, 23.0),
        ];
        let exporter = SqlExporter::new();
        let batches = exporter.insert_sql(&schema, &points);
        assert_eq!(batches.len(), 1);
        let sql = &batches[0];
        assert!(sql.contains("INSERT INTO"));
        assert!(sql.contains("1000"));
        assert!(sql.contains("2000"));
    }

    #[test]
    fn test_insert_sql_with_tags() {
        let schema = sample_schema();
        let points = sample_points(3);
        let exporter = SqlExporter::new();
        let batches = exporter.insert_sql(&schema, &points);
        assert_eq!(batches.len(), 1);
        let sql = &batches[0];
        assert!(sql.contains("eu-west"));
        assert!(sql.contains("host"));
    }

    #[test]
    fn test_insert_sql_batching() {
        let schema = MetricSchema::builder("m").build();
        let points: Vec<SqlDataPoint> = (0..10).map(|i| SqlDataPoint::new(i, i as f64)).collect();
        let exporter = SqlExporter::new().with_batch_size(3);
        let batches = exporter.insert_sql(&schema, &points);
        // 10 rows / 3 per batch = 4 batches (3, 3, 3, 1)
        assert_eq!(batches.len(), 4);
    }

    #[test]
    fn test_insert_sql_tag_null_when_missing() {
        let schema = MetricSchema::builder("m").with_tag("host").build();
        let points = vec![SqlDataPoint::new(1_000, 5.0)]; // no tags
        let exporter = SqlExporter::new();
        let batches = exporter.insert_sql(&schema, &points);
        assert_eq!(batches.len(), 1);
        assert!(batches[0].contains("NULL"));
    }

    // -- export_to_sql_file ---------------------------------------------------

    #[test]
    fn test_export_to_sql_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxirs_tsdb_test_export.sql");

        let schema = MetricSchema::builder("sensor").with_tag("device").build();
        let points = sample_points(5);
        let exporter = SqlExporter::new();
        exporter
            .export_to_sql_file(&schema, &points, &path)
            .expect("export should succeed");

        let content = std::fs::read_to_string(&path).expect("read sql file");
        assert!(content.contains("CREATE TABLE"));
        assert!(content.contains("INSERT INTO"));
        let _ = std::fs::remove_file(&path);
    }

    // -- Helper queries -------------------------------------------------------

    #[test]
    fn test_select_range_sql() {
        let schema = sample_schema();
        let exporter = SqlExporter::new();
        let sql = exporter.select_range_sql(&schema, 1_000, 5_000);
        assert!(sql.contains("timestamp_ms BETWEEN 1000 AND 5000"));
        assert!(sql.contains("ORDER BY timestamp_ms ASC"));
    }

    #[test]
    fn test_select_range_sql_includes_tag_columns() {
        let schema = sample_schema();
        let exporter = SqlExporter::new();
        let sql = exporter.select_range_sql(&schema, 0, 9_999);
        assert!(sql.contains("host"));
        assert!(sql.contains("region"));
    }

    #[test]
    fn test_count_sql() {
        let schema = sample_schema();
        let exporter = SqlExporter::new();
        let sql = exporter.count_sql(&schema);
        assert!(sql.contains("COUNT(*)"));
        assert!(sql.contains("row_count"));
        assert!(sql.contains("cpu_usage"));
    }

    #[test]
    fn test_summary_sql() {
        let schema = sample_schema();
        let exporter = SqlExporter::new();
        let sql = exporter.summary_sql(&schema);
        assert!(sql.contains("MIN(value)"));
        assert!(sql.contains("MAX(value)"));
        assert!(sql.contains("AVG(value)"));
        assert!(sql.contains("row_count"));
    }

    #[test]
    fn test_delete_before_sql() {
        let schema = sample_schema();
        let exporter = SqlExporter::new();
        let sql = exporter.delete_before_sql(&schema, 1_000_000);
        assert!(sql.contains("DELETE FROM"));
        assert!(sql.contains("1000000"));
        assert!(sql.contains("timestamp_ms <"));
    }

    #[test]
    fn test_create_index_sql() {
        let schema = sample_schema();
        let exporter = SqlExporter::new();
        let sql = exporter.create_index_sql(&schema);
        assert!(sql.contains("CREATE INDEX IF NOT EXISTS"));
        assert!(sql.contains("timestamp_ms"));
    }

    #[test]
    fn test_add_tag_column_sql() {
        let schema = MetricSchema::builder("m").build();
        let exporter = SqlExporter::new();
        let sql = exporter.add_tag_column_sql(&schema, "datacenter");
        assert!(sql.contains("ALTER TABLE"));
        assert!(sql.contains("ADD COLUMN datacenter VARCHAR"));
    }

    // -- escape_float ---------------------------------------------------------

    #[test]
    fn test_escape_float_normal() {
        let s = escape_float(42.0);
        assert!(s.contains("42"));
    }

    #[test]
    fn test_escape_float_special_values() {
        assert_eq!(escape_float(f64::NAN), "'NaN'");
        assert_eq!(escape_float(f64::INFINITY), "'Infinity'");
        assert_eq!(escape_float(f64::NEG_INFINITY), "'-Infinity'");
    }

    // -- sanitize_sql_identifier ----------------------------------------------

    #[test]
    fn test_sanitize_identifier_clean() {
        assert_eq!(sanitize_sql_identifier("cpu_usage"), "cpu_usage");
    }

    #[test]
    fn test_sanitize_identifier_hyphen() {
        let id = sanitize_sql_identifier("my-metric");
        assert!(!id.contains('-'));
    }

    #[test]
    fn test_sanitize_identifier_leading_digit() {
        let id = sanitize_sql_identifier("1bad");
        assert!(!id.starts_with(|c: char| c.is_ascii_digit()));
    }

    #[test]
    fn test_sanitize_identifier_empty() {
        assert_eq!(sanitize_sql_identifier(""), "_");
    }

    // -- SqlDataPoint ---------------------------------------------------------

    #[test]
    fn test_sql_data_point_new() {
        let p = SqlDataPoint::new(1_000, 42.0);
        assert_eq!(p.timestamp_ms, 1_000);
        assert!((p.value - 42.0).abs() < f64::EPSILON);
        assert!(p.tags.is_empty());
    }

    #[test]
    fn test_sql_data_point_with_tags() {
        let mut tags = HashMap::new();
        tags.insert("host".to_string(), "srv-01".to_string());
        let p = SqlDataPoint::with_tags(2_000, 55.5, tags);
        assert_eq!(p.tags["host"], "srv-01");
    }

    // -- SqlExporter::infer_schema -------------------------------------------

    #[test]
    fn test_infer_schema_from_points() {
        let points = sample_points(5);
        let schema = SqlExporter::infer_schema("cpu_usage", &points);
        assert_eq!(schema.metric_name, "cpu_usage");
        assert_eq!(schema.value_type, DataValueType::Float64);
    }

    #[test]
    fn test_infer_schema_no_points() {
        let schema = SqlExporter::infer_schema("empty_metric", &[]);
        assert_eq!(schema.metric_name, "empty_metric");
        assert!(schema.tag_keys.is_empty());
    }
}
