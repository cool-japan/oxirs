//! OxiRS-native binary time-series IPC export format.
//!
//! # This is NOT Apache Arrow IPC
//!
//! The wire format implemented by this module is loosely *inspired* by the
//! layout of the Apache Arrow IPC stream format (schema message, then
//! record-batch messages, then an EOS marker) but its metadata encoding is
//! a simplified, crate-private scheme — **not** the real Arrow flatbuffers
//! schema. Bytes produced by [`OxirsIpcWriter`] can only be parsed by
//! [`OxirsIpcReader`] in this same crate; they are **not** readable by
//! `pyarrow`, `arrow-rs`, DuckDB, Polars, or any other genuine Arrow IPC
//! consumer. Do not hand this output to external analytics tooling expecting
//! Arrow compatibility.
//!
//! For genuine, spec-conformant Apache Arrow / Parquet interoperability
//! (readable by real Arrow/Parquet tooling), use [`crate::ArrowExporter`] /
//! [`crate::ParquetExporter`] instead, which wrap the real `arrow` and
//! `parquet` crates and are enabled by the `arrow-export` Cargo feature.
//!
//! This module exists as a lightweight, always-available (no extra
//! dependency), fast internal serialization format for OxiRS-to-OxiRS data
//! interchange (e.g. streaming chunk export/import between OxiRS nodes)
//! where true Arrow interoperability is not required.
//!
//! ## Wire Format (OxiRS-native, simplified)
//!
//! ```text
//! [MAGIC: 6 bytes "OXIPC1"] [padding: 2 bytes]
//! [schema message]
//! [record batch messages ...]
//! [EOS marker: 4 bytes 0xFFFFFFFF + 4 bytes 0x00000000]
//! ```
//!
//! Each message is:
//! ```text
//! [continuation: 4 bytes 0xFFFFFFFF]
//! [metadata_size: i32 LE]
//! [metadata: metadata_size bytes (OxiRS-private encoding, NOT flatbuffers)]
//! [body: aligned to 8 bytes]
//! ```

use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

/// Magic bytes for the OxiRS-native IPC stream (not Arrow-compatible).
const OXIPC_MAGIC: &[u8] = b"OXIPC1";
/// Padding after magic to align to 8 bytes.
const OXIPC_MAGIC_PADDING: &[u8] = &[0u8, 0u8];
/// Continuation marker (used in IPC stream format before each message).
const CONTINUATION_MARKER: u32 = 0xFFFF_FFFF;

// ──────────────────────────────────────────────────────────────────────────────
// OxirsIpcTimeUnit
// ──────────────────────────────────────────────────────────────────────────────

/// Timestamp resolution unit for the OxiRS-native Timestamp type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OxirsIpcTimeUnit {
    /// 1-second resolution.
    Second,
    /// 1-millisecond resolution.
    Millisecond,
    /// 1-microsecond resolution.
    Microsecond,
    /// 1-nanosecond resolution.
    Nanosecond,
}

impl OxirsIpcTimeUnit {
    fn code(&self) -> u8 {
        match self {
            OxirsIpcTimeUnit::Second => 0,
            OxirsIpcTimeUnit::Millisecond => 1,
            OxirsIpcTimeUnit::Microsecond => 2,
            OxirsIpcTimeUnit::Nanosecond => 3,
        }
    }

    fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(OxirsIpcTimeUnit::Second),
            1 => Some(OxirsIpcTimeUnit::Millisecond),
            2 => Some(OxirsIpcTimeUnit::Microsecond),
            3 => Some(OxirsIpcTimeUnit::Nanosecond),
            _ => None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OxirsIpcDataType
// ──────────────────────────────────────────────────────────────────────────────

/// Subset of scalar data types supported by the OxiRS-native IPC format (modeled loosely on Arrow's type system, but not wire-compatible with it).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OxirsIpcDataType {
    /// 64-bit signed integer.
    Int64,
    /// 64-bit IEEE 754 float.
    Float64,
    /// UTF-8 variable-length string.
    Utf8,
    /// 64-bit timestamp with a given time unit.
    Timestamp(OxirsIpcTimeUnit),
    /// Boolean (bit-packed in real Arrow; stored as one byte per value here for simplicity).
    Boolean,
}

impl OxirsIpcDataType {
    /// Numeric type tag embedded in the wire format.
    fn type_tag(&self) -> u8 {
        match self {
            OxirsIpcDataType::Int64 => 1,
            OxirsIpcDataType::Float64 => 2,
            OxirsIpcDataType::Utf8 => 3,
            OxirsIpcDataType::Timestamp(_) => 4,
            OxirsIpcDataType::Boolean => 5,
        }
    }

    fn from_tag(tag: u8, time_unit: u8) -> Option<Self> {
        match tag {
            1 => Some(OxirsIpcDataType::Int64),
            2 => Some(OxirsIpcDataType::Float64),
            3 => Some(OxirsIpcDataType::Utf8),
            4 => OxirsIpcTimeUnit::from_code(time_unit).map(OxirsIpcDataType::Timestamp),
            5 => Some(OxirsIpcDataType::Boolean),
            _ => None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OxirsIpcField
// ──────────────────────────────────────────────────────────────────────────────

/// A single column descriptor in an [`OxirsIpcSchema`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OxirsIpcField {
    /// Column name.
    pub name: String,
    /// Column data type.
    pub data_type: OxirsIpcDataType,
    /// Whether the column may contain nulls.
    pub nullable: bool,
}

impl OxirsIpcField {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, data_type: OxirsIpcDataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OxirsIpcSchema
// ──────────────────────────────────────────────────────────────────────────────

/// Schema describing the columns of an [`OxirsIpcRecordBatch`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OxirsIpcSchema {
    /// Ordered list of fields.
    pub fields: Vec<OxirsIpcField>,
}

impl OxirsIpcSchema {
    /// Create a new schema from a list of fields.
    pub fn new(fields: Vec<OxirsIpcField>) -> Self {
        Self { fields }
    }

    /// Serialise the schema to bytes (simplified flatbuffer substitute).
    ///
    /// Format:
    /// ```text
    /// [n_fields: u32 LE]
    /// for each field:
    ///   [name_len: u32 LE] [name: utf-8 bytes]
    ///   [type_tag: u8] [time_unit: u8] [nullable: u8] [_pad: u8]
    /// ```
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let n = self.fields.len() as u32;
        buf.extend_from_slice(&n.to_le_bytes());
        for field in &self.fields {
            let name_bytes = field.name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            let tag = field.data_type.type_tag();
            let tu = match &field.data_type {
                OxirsIpcDataType::Timestamp(u) => u.code(),
                _ => 0,
            };
            buf.push(tag);
            buf.push(tu);
            buf.push(field.nullable as u8);
            buf.push(0); // alignment padding
        }
        buf
    }

    /// Deserialise a schema from the format produced by `to_bytes`.
    fn from_bytes(src: &[u8]) -> TsdbResult<Self> {
        if src.len() < 4 {
            return Err(TsdbError::Arrow(
                "OxirsIpcSchema: buffer too short for field count".into(),
            ));
        }
        let n_fields = u32::from_le_bytes(
            src[0..4]
                .try_into()
                .map_err(|_| TsdbError::Arrow("OxirsIpcSchema: cannot read field count".into()))?,
        ) as usize;
        let mut fields = Vec::with_capacity(n_fields);
        let mut pos = 4usize;
        for _ in 0..n_fields {
            if pos + 4 > src.len() {
                return Err(TsdbError::Arrow(
                    "OxirsIpcSchema: buffer truncated reading name length".into(),
                ));
            }
            let name_len =
                u32::from_le_bytes(src[pos..pos + 4].try_into().map_err(|_| {
                    TsdbError::Arrow("OxirsIpcSchema: cannot read name length".into())
                })?) as usize;
            pos += 4;
            if pos + name_len > src.len() {
                return Err(TsdbError::Arrow(
                    "OxirsIpcSchema: buffer truncated reading name".into(),
                ));
            }
            let name = std::str::from_utf8(&src[pos..pos + name_len])
                .map_err(|e| {
                    TsdbError::Arrow(format!("OxirsIpcSchema: invalid UTF-8 in name: {e}"))
                })?
                .to_owned();
            pos += name_len;
            if pos + 4 > src.len() {
                return Err(TsdbError::Arrow(
                    "OxirsIpcSchema: buffer truncated reading type tag".into(),
                ));
            }
            let tag = src[pos];
            let tu = src[pos + 1];
            let nullable = src[pos + 2] != 0;
            pos += 4;
            let data_type = OxirsIpcDataType::from_tag(tag, tu).ok_or_else(|| {
                TsdbError::Arrow(format!("OxirsIpcSchema: unknown type tag {tag}"))
            })?;
            fields.push(OxirsIpcField {
                name,
                data_type,
                nullable,
            });
        }
        Ok(OxirsIpcSchema { fields })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OxirsIpcColumn
// ──────────────────────────────────────────────────────────────────────────────

/// A typed column in an [`OxirsIpcRecordBatch`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OxirsIpcColumn {
    /// Array of 64-bit signed integers.
    Int64(Vec<i64>),
    /// Array of 64-bit floats.
    Float64(Vec<f64>),
    /// Array of UTF-8 strings.
    Utf8(Vec<String>),
    /// Array of 64-bit timestamps with a time unit.
    Timestamp(Vec<i64>, OxirsIpcTimeUnit),
    /// Array of booleans.
    Boolean(Vec<bool>),
}

impl OxirsIpcColumn {
    /// Number of elements in this column.
    pub fn len(&self) -> usize {
        match self {
            OxirsIpcColumn::Int64(v) => v.len(),
            OxirsIpcColumn::Float64(v) => v.len(),
            OxirsIpcColumn::Utf8(v) => v.len(),
            OxirsIpcColumn::Timestamp(v, _) => v.len(),
            OxirsIpcColumn::Boolean(v) => v.len(),
        }
    }

    /// Returns `true` if the column has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialise the column body to bytes.
    ///
    /// Format per type:
    /// - Int64 / Timestamp: raw little-endian i64 values
    /// - Float64: raw little-endian f64 values (IEEE 754 bit pattern)
    /// - Boolean: 1 byte per value (0 or 1)
    /// - Utf8: [total_data_len: u32 LE] [offsets: (n+1)*u32 LE] [data bytes]
    fn body_bytes(&self) -> Vec<u8> {
        match self {
            OxirsIpcColumn::Int64(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            OxirsIpcColumn::Float64(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            OxirsIpcColumn::Timestamp(v, _) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            OxirsIpcColumn::Boolean(v) => v.iter().map(|&b| b as u8).collect(),
            OxirsIpcColumn::Utf8(v) => {
                // Offsets array (n+1 entries, each u32).
                let mut offsets: Vec<u32> = Vec::with_capacity(v.len() + 1);
                let mut data: Vec<u8> = Vec::new();
                offsets.push(0);
                for s in v {
                    let bytes = s.as_bytes();
                    data.extend_from_slice(bytes);
                    offsets.push(data.len() as u32);
                }
                let total_len = data.len() as u32;
                let mut buf = Vec::with_capacity(4 + offsets.len() * 4 + data.len());
                buf.extend_from_slice(&total_len.to_le_bytes());
                for off in &offsets {
                    buf.extend_from_slice(&off.to_le_bytes());
                }
                buf.extend_from_slice(&data);
                buf
            }
        }
    }

    /// Deserialise a column from raw bytes given a type descriptor.
    fn from_bytes(
        bytes: &[u8],
        data_type: &OxirsIpcDataType,
        n_rows: usize,
    ) -> TsdbResult<OxirsIpcColumn> {
        match data_type {
            OxirsIpcDataType::Int64 => {
                if bytes.len() < n_rows * 8 {
                    return Err(TsdbError::Arrow("Int64 column: buffer too short".into()));
                }
                let values: Vec<i64> = (0..n_rows)
                    .map(|i| {
                        i64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap_or_default())
                    })
                    .collect();
                Ok(OxirsIpcColumn::Int64(values))
            }
            OxirsIpcDataType::Float64 => {
                if bytes.len() < n_rows * 8 {
                    return Err(TsdbError::Arrow("Float64 column: buffer too short".into()));
                }
                let values: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        f64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap_or_default())
                    })
                    .collect();
                Ok(OxirsIpcColumn::Float64(values))
            }
            OxirsIpcDataType::Timestamp(unit) => {
                if bytes.len() < n_rows * 8 {
                    return Err(TsdbError::Arrow(
                        "Timestamp column: buffer too short".into(),
                    ));
                }
                let values: Vec<i64> = (0..n_rows)
                    .map(|i| {
                        i64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap_or_default())
                    })
                    .collect();
                Ok(OxirsIpcColumn::Timestamp(values, *unit))
            }
            OxirsIpcDataType::Boolean => {
                if bytes.len() < n_rows {
                    return Err(TsdbError::Arrow("Boolean column: buffer too short".into()));
                }
                let values: Vec<bool> = bytes[..n_rows].iter().map(|&b| b != 0).collect();
                Ok(OxirsIpcColumn::Boolean(values))
            }
            OxirsIpcDataType::Utf8 => {
                if bytes.len() < 4 {
                    return Err(TsdbError::Arrow("Utf8 column: missing total_len".into()));
                }
                let _total_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap_or_default());
                let offsets_end = 4 + (n_rows + 1) * 4;
                if bytes.len() < offsets_end {
                    return Err(TsdbError::Arrow(
                        "Utf8 column: buffer too short for offsets".into(),
                    ));
                }
                let offsets: Vec<u32> = (0..=n_rows)
                    .map(|i| {
                        let base = 4 + i * 4;
                        u32::from_le_bytes(bytes[base..base + 4].try_into().unwrap_or_default())
                    })
                    .collect();
                let data = &bytes[offsets_end..];
                let mut strings = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    let start = offsets[i] as usize;
                    let end = offsets[i + 1] as usize;
                    if end > data.len() {
                        return Err(TsdbError::Arrow(
                            "Utf8 column: data bytes out of range".into(),
                        ));
                    }
                    let s = std::str::from_utf8(&data[start..end]).map_err(|e| {
                        TsdbError::Arrow(format!("Utf8 column: invalid UTF-8: {e}"))
                    })?;
                    strings.push(s.to_owned());
                }
                Ok(OxirsIpcColumn::Utf8(strings))
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OxirsIpcRecordBatch
// ──────────────────────────────────────────────────────────────────────────────

/// A collection of columnar arrays with a shared schema and equal row counts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OxirsIpcRecordBatch {
    /// Schema describing all columns.
    pub schema: OxirsIpcSchema,
    /// Parallel array of columns; must have the same number of elements.
    pub columns: Vec<OxirsIpcColumn>,
}

impl OxirsIpcRecordBatch {
    /// Create a new record batch after validating schema/column alignment.
    pub fn new(schema: OxirsIpcSchema, columns: Vec<OxirsIpcColumn>) -> TsdbResult<Self> {
        if schema.fields.len() != columns.len() {
            return Err(TsdbError::Arrow(format!(
                "RecordBatch: schema has {} fields but {} columns provided",
                schema.fields.len(),
                columns.len()
            )));
        }
        if columns.len() > 1 {
            let n_rows = columns[0].len();
            for (i, col) in columns.iter().enumerate().skip(1) {
                if col.len() != n_rows {
                    return Err(TsdbError::Arrow(format!(
                        "RecordBatch: column 0 has {n_rows} rows but column {i} has {}",
                        col.len()
                    )));
                }
            }
        }
        Ok(Self { schema, columns })
    }

    /// Number of rows in the batch.
    pub fn num_rows(&self) -> usize {
        self.columns.first().map(|c| c.len()).unwrap_or(0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OxirsIpcWriter
// ──────────────────────────────────────────────────────────────────────────────

/// Writer for the OxiRS-native IPC stream format (see module docs — this is NOT Apache Arrow IPC).
///
/// Maintains state across `write_schema` / `write_batch` / `write_footer` calls
/// so that the caller can stream batches incrementally.
#[derive(Debug, Default)]
pub struct OxirsIpcWriter {
    /// Whether the schema has already been written.
    schema_written: bool,
    /// Number of batches written.
    batches_written: usize,
}

impl OxirsIpcWriter {
    /// Create a fresh writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Emit the IPC stream header (magic + schema message).
    ///
    /// Must be called exactly once before `write_batch`.
    pub fn write_schema(&mut self, schema: &OxirsIpcSchema) -> TsdbResult<Vec<u8>> {
        if self.schema_written {
            return Err(TsdbError::Arrow(
                "OxirsIpcWriter: schema already written".into(),
            ));
        }
        let mut buf = Vec::new();
        // Magic
        buf.extend_from_slice(OXIPC_MAGIC);
        buf.extend_from_slice(OXIPC_MAGIC_PADDING);
        // Schema message
        let schema_bytes = schema.to_bytes();
        // Message type tag: 0x01 = Schema
        let metadata = encode_message(0x01, &schema_bytes);
        buf.extend_from_slice(&metadata);
        self.schema_written = true;
        Ok(buf)
    }

    /// Emit a record batch message.
    ///
    /// `write_schema` must be called first.
    pub fn write_batch(&mut self, batch: &OxirsIpcRecordBatch) -> TsdbResult<Vec<u8>> {
        if !self.schema_written {
            return Err(TsdbError::Arrow(
                "OxirsIpcWriter: schema not yet written".into(),
            ));
        }
        let mut buf = Vec::new();

        // Batch header: n_rows (u32), n_cols (u32), then per-column body lengths.
        let n_rows = batch.num_rows() as u32;
        let n_cols = batch.columns.len() as u32;

        let mut header = Vec::new();
        header.extend_from_slice(&n_rows.to_le_bytes());
        header.extend_from_slice(&n_cols.to_le_bytes());

        // Collect column bodies.
        let bodies: Vec<Vec<u8>> = batch.columns.iter().map(|c| c.body_bytes()).collect();
        for body in &bodies {
            header.extend_from_slice(&(body.len() as u32).to_le_bytes());
        }

        // Message type tag: 0x02 = RecordBatch
        let metadata = encode_message(0x02, &header);
        buf.extend_from_slice(&metadata);

        // Emit column bodies (each 8-byte aligned).
        for body in &bodies {
            buf.extend_from_slice(body);
            let pad = (8 - body.len() % 8) % 8;
            buf.extend(std::iter::repeat(0u8).take(pad));
        }

        self.batches_written += 1;
        Ok(buf)
    }

    /// Emit the EOS (end-of-stream) marker.
    pub fn write_footer(&mut self) -> TsdbResult<Vec<u8>> {
        // EOS: continuation marker + 0 metadata length.
        let mut buf = Vec::new();
        buf.extend_from_slice(&CONTINUATION_MARKER.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        Ok(buf)
    }

    /// Convert a slice of `DataPoint`s into an [`OxirsIpcRecordBatch`].
    ///
    /// Produces two columns:
    /// - `timestamp` : Timestamp(Millisecond)
    /// - `value`     : Float64
    ///
    /// Tags are not included (use [`time_series_with_tags_to_batch`] for that).
    pub fn time_series_to_batch(series: &[crate::series::DataPoint]) -> OxirsIpcRecordBatch {
        let timestamps: Vec<i64> = series
            .iter()
            .map(|p| p.timestamp.timestamp_millis())
            .collect();
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();

        let schema = OxirsIpcSchema::new(vec![
            OxirsIpcField::new(
                "timestamp",
                OxirsIpcDataType::Timestamp(OxirsIpcTimeUnit::Millisecond),
                false,
            ),
            OxirsIpcField::new("value", OxirsIpcDataType::Float64, false),
        ]);

        OxirsIpcRecordBatch {
            schema,
            columns: vec![
                OxirsIpcColumn::Timestamp(timestamps, OxirsIpcTimeUnit::Millisecond),
                OxirsIpcColumn::Float64(values),
            ],
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Extended helper for tagged data points
// ──────────────────────────────────────────────────────────────────────────────

/// A time-series data point extended with a metric name and string tags.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaggedDataPoint {
    /// Unix epoch milliseconds.
    pub timestamp: i64,
    /// Measured value.
    pub value: f64,
    /// String tags (key → value).
    pub tags: HashMap<String, String>,
}

impl TaggedDataPoint {
    /// Create a new tagged data point.
    pub fn new(timestamp: i64, value: f64, tags: HashMap<String, String>) -> Self {
        Self {
            timestamp,
            value,
            tags,
        }
    }
}

/// Convert tagged data points into a record batch with a `tags_json` column.
pub fn time_series_with_tags_to_batch(
    series: &[TaggedDataPoint],
) -> TsdbResult<OxirsIpcRecordBatch> {
    let timestamps: Vec<i64> = series.iter().map(|p| p.timestamp).collect();
    let values: Vec<f64> = series.iter().map(|p| p.value).collect();
    let tags_json: Vec<String> = series
        .iter()
        .map(|p| serde_json::to_string(&p.tags).unwrap_or_else(|_| "{}".to_owned()))
        .collect();

    let schema = OxirsIpcSchema::new(vec![
        OxirsIpcField::new(
            "timestamp",
            OxirsIpcDataType::Timestamp(OxirsIpcTimeUnit::Millisecond),
            false,
        ),
        OxirsIpcField::new("value", OxirsIpcDataType::Float64, false),
        OxirsIpcField::new("tags_json", OxirsIpcDataType::Utf8, true),
    ]);

    OxirsIpcRecordBatch::new(
        schema,
        vec![
            OxirsIpcColumn::Timestamp(timestamps, OxirsIpcTimeUnit::Millisecond),
            OxirsIpcColumn::Float64(values),
            OxirsIpcColumn::Utf8(tags_json),
        ],
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// OxirsIpcReader
// ──────────────────────────────────────────────────────────────────────────────

/// Reader for OxiRS-native IPC stream bytes produced by [`OxirsIpcWriter`] (NOT a general-purpose Apache Arrow IPC reader).
#[derive(Debug, Default)]
pub struct OxirsIpcReader;

impl OxirsIpcReader {
    /// Parse all record batches from an IPC byte stream.
    ///
    /// The stream must start with the 8-byte magic, followed by a schema
    /// message, then zero or more record-batch messages, and finally an EOS
    /// marker.
    pub fn read_batches(data: &[u8]) -> TsdbResult<Vec<OxirsIpcRecordBatch>> {
        let mut pos = 0usize;

        // 1. Validate magic.
        if data.len() < 8 {
            return Err(TsdbError::Arrow("IPC stream too short for magic".into()));
        }
        if &data[pos..pos + 6] != OXIPC_MAGIC {
            return Err(TsdbError::Arrow("IPC stream: invalid magic".into()));
        }
        pos += 8; // magic + padding

        // 2. Parse schema message.
        let (msg_type, schema_bytes, consumed) = decode_message(&data[pos..])?;
        if msg_type != 0x01 {
            return Err(TsdbError::Arrow(format!(
                "IPC stream: expected schema message (type=1), got type={msg_type}"
            )));
        }
        let schema = OxirsIpcSchema::from_bytes(&schema_bytes)?;
        pos += consumed;

        // 3. Parse record batch messages until EOS.
        let mut batches = Vec::new();
        loop {
            if pos + 8 > data.len() {
                break; // no more data
            }
            // Peek at continuation marker.
            let cont = u32::from_le_bytes(
                data[pos..pos + 4]
                    .try_into()
                    .map_err(|_| TsdbError::Arrow("IPC: cannot read continuation".into()))?,
            );
            if cont != CONTINUATION_MARKER {
                break;
            }
            // Peek at metadata length to detect EOS.
            let meta_len = u32::from_le_bytes(
                data[pos + 4..pos + 8]
                    .try_into()
                    .map_err(|_| TsdbError::Arrow("IPC: cannot read metadata_len".into()))?,
            );
            if meta_len == 0 {
                // EOS marker.
                break;
            }

            let (msg_type, header, consumed) = decode_message(&data[pos..])?;
            pos += consumed;
            if msg_type != 0x02 {
                // Unknown message type — skip.
                continue;
            }

            // Parse batch header: n_rows, n_cols, then per-column body lengths.
            if header.len() < 8 {
                return Err(TsdbError::Arrow(
                    "RecordBatch message: header too short".into(),
                ));
            }
            let n_rows = u32::from_le_bytes(header[0..4].try_into().unwrap_or_default()) as usize;
            let n_cols = u32::from_le_bytes(header[4..8].try_into().unwrap_or_default()) as usize;
            if header.len() < 8 + n_cols * 4 {
                return Err(TsdbError::Arrow(
                    "RecordBatch message: header too short for column lengths".into(),
                ));
            }
            let col_lens: Vec<usize> = (0..n_cols)
                .map(|i| {
                    let base = 8 + i * 4;
                    u32::from_le_bytes(header[base..base + 4].try_into().unwrap_or_default())
                        as usize
                })
                .collect();

            // Read column bodies from the stream.
            let mut columns = Vec::with_capacity(n_cols);
            for (i, &col_len) in col_lens.iter().enumerate() {
                if pos + col_len > data.len() {
                    return Err(TsdbError::Arrow(format!(
                        "RecordBatch: column {i} body out of bounds"
                    )));
                }
                let field = schema.fields.get(i).ok_or_else(|| {
                    TsdbError::Arrow(format!(
                        "RecordBatch: column {i} has no matching field in schema"
                    ))
                })?;
                let col = OxirsIpcColumn::from_bytes(
                    &data[pos..pos + col_len],
                    &field.data_type,
                    n_rows,
                )?;
                columns.push(col);
                // Advance past column body + 8-byte alignment padding.
                let aligned = col_len + (8 - col_len % 8) % 8;
                pos += aligned;
            }

            let batch = OxirsIpcRecordBatch {
                schema: schema.clone(),
                columns,
            };
            batches.push(batch);
        }

        Ok(batches)
    }

    /// Convert a record batch (with `timestamp` and `value` columns) back to
    /// `DataPoint`s.
    pub fn batch_to_time_series(batch: &OxirsIpcRecordBatch) -> Vec<crate::series::DataPoint> {
        use chrono::{TimeZone, Utc};

        // Find timestamp and value columns by position (schema order).
        let timestamps = batch.columns.first().and_then(|c| {
            if let OxirsIpcColumn::Timestamp(v, _) = c {
                Some(v.as_slice())
            } else {
                None
            }
        });
        let values = batch.columns.get(1).and_then(|c| {
            if let OxirsIpcColumn::Float64(v) = c {
                Some(v.as_slice())
            } else {
                None
            }
        });

        match (timestamps, values) {
            (Some(ts), Some(vs)) => ts
                .iter()
                .zip(vs.iter())
                .filter_map(|(&t, &v)| {
                    Utc.timestamp_millis_opt(t)
                        .single()
                        .map(|dt| crate::series::DataPoint::new(dt, v))
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Private wire-format helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Encode a message: `[continuation: u32 LE] [metadata_len: i32 LE] [metadata]`.
fn encode_message(msg_type: u8, payload: &[u8]) -> Vec<u8> {
    // Our simplified metadata = [msg_type: u8] [payload].
    let metadata_len = (1 + payload.len()) as i32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&CONTINUATION_MARKER.to_le_bytes());
    buf.extend_from_slice(&metadata_len.to_le_bytes());
    buf.push(msg_type);
    buf.extend_from_slice(payload);
    buf
}

/// Decode a message, returning `(msg_type, payload_bytes, total_bytes_consumed)`.
fn decode_message(data: &[u8]) -> TsdbResult<(u8, Vec<u8>, usize)> {
    if data.len() < 8 {
        return Err(TsdbError::Arrow("message: buffer too short".into()));
    }
    let cont = u32::from_le_bytes(
        data[0..4]
            .try_into()
            .map_err(|_| TsdbError::Arrow("message: cannot read continuation".into()))?,
    );
    if cont != CONTINUATION_MARKER {
        return Err(TsdbError::Arrow(format!(
            "message: expected continuation {CONTINUATION_MARKER:#010x}, got {cont:#010x}"
        )));
    }
    let meta_len = i32::from_le_bytes(
        data[4..8]
            .try_into()
            .map_err(|_| TsdbError::Arrow("message: cannot read metadata_len".into()))?,
    ) as usize;
    if data.len() < 8 + meta_len {
        return Err(TsdbError::Arrow(
            "message: buffer too short for metadata".into(),
        ));
    }
    let msg_type = data[8];
    let payload = data[9..8 + meta_len].to_vec();
    let consumed = 8 + meta_len;
    Ok((msg_type, payload, consumed))
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::DataPoint;
    use chrono::Utc;

    // ── Schema ────────────────────────────────────────────────────────────────

    #[test]
    fn test_schema_roundtrip_single_field() {
        let schema = OxirsIpcSchema::new(vec![OxirsIpcField::new(
            "value",
            OxirsIpcDataType::Float64,
            false,
        )]);
        let bytes = schema.to_bytes();
        let decoded = OxirsIpcSchema::from_bytes(&bytes).expect("roundtrip");
        assert_eq!(schema, decoded);
    }

    #[test]
    fn test_schema_roundtrip_multiple_fields() {
        let schema = OxirsIpcSchema::new(vec![
            OxirsIpcField::new(
                "ts",
                OxirsIpcDataType::Timestamp(OxirsIpcTimeUnit::Millisecond),
                false,
            ),
            OxirsIpcField::new("value", OxirsIpcDataType::Float64, false),
            OxirsIpcField::new("label", OxirsIpcDataType::Utf8, true),
            OxirsIpcField::new("active", OxirsIpcDataType::Boolean, false),
            OxirsIpcField::new("count", OxirsIpcDataType::Int64, false),
        ]);
        let bytes = schema.to_bytes();
        let decoded = OxirsIpcSchema::from_bytes(&bytes).expect("roundtrip");
        assert_eq!(schema, decoded);
    }

    #[test]
    fn test_schema_empty_fields() {
        let schema = OxirsIpcSchema::new(vec![]);
        let bytes = schema.to_bytes();
        let decoded = OxirsIpcSchema::from_bytes(&bytes).expect("roundtrip");
        assert!(decoded.fields.is_empty());
    }

    #[test]
    fn test_schema_from_bytes_truncated_error() {
        let result = OxirsIpcSchema::from_bytes(&[0u8; 2]);
        assert!(result.is_err());
    }

    // ── Column serialization ──────────────────────────────────────────────────

    #[test]
    fn test_int64_column_roundtrip() {
        let col = OxirsIpcColumn::Int64(vec![1, -2, 1_000_000, i64::MAX]);
        let bytes = col.body_bytes();
        let decoded = OxirsIpcColumn::from_bytes(&bytes, &OxirsIpcDataType::Int64, 4)
            .expect("should succeed");
        assert_eq!(col, decoded);
    }

    #[test]
    fn test_float64_column_roundtrip() {
        let col =
            OxirsIpcColumn::Float64(vec![1.5, -std::f64::consts::PI, f64::NAN, f64::INFINITY]);
        let bytes = col.body_bytes();
        let decoded = OxirsIpcColumn::from_bytes(&bytes, &OxirsIpcDataType::Float64, 4)
            .expect("should succeed");
        // NaN != NaN so compare element-by-element.
        if let (OxirsIpcColumn::Float64(orig), OxirsIpcColumn::Float64(dec)) = (&col, &decoded) {
            assert_eq!(orig.len(), dec.len());
            assert_eq!(dec[0], 1.5);
            assert!(dec[2].is_nan());
        } else {
            panic!("unexpected column type");
        }
    }

    #[test]
    fn test_utf8_column_roundtrip() {
        let col = OxirsIpcColumn::Utf8(vec!["hello".into(), "world".into(), "".into()]);
        let bytes = col.body_bytes();
        let decoded =
            OxirsIpcColumn::from_bytes(&bytes, &OxirsIpcDataType::Utf8, 3).expect("should succeed");
        assert_eq!(col, decoded);
    }

    #[test]
    fn test_timestamp_column_roundtrip() {
        let col = OxirsIpcColumn::Timestamp(vec![0, 1_000, -500], OxirsIpcTimeUnit::Millisecond);
        let bytes = col.body_bytes();
        let decoded = OxirsIpcColumn::from_bytes(
            &bytes,
            &OxirsIpcDataType::Timestamp(OxirsIpcTimeUnit::Millisecond),
            3,
        )
        .expect("should succeed");
        assert_eq!(col, decoded);
    }

    #[test]
    fn test_boolean_column_roundtrip() {
        let col = OxirsIpcColumn::Boolean(vec![true, false, true, true, false]);
        let bytes = col.body_bytes();
        let decoded = OxirsIpcColumn::from_bytes(&bytes, &OxirsIpcDataType::Boolean, 5)
            .expect("should succeed");
        assert_eq!(col, decoded);
    }

    // ── OxirsIpcWriter / OxirsIpcReader ───────────────────────────────────────

    fn make_batch() -> OxirsIpcRecordBatch {
        let schema = OxirsIpcSchema::new(vec![
            OxirsIpcField::new(
                "ts",
                OxirsIpcDataType::Timestamp(OxirsIpcTimeUnit::Millisecond),
                false,
            ),
            OxirsIpcField::new("val", OxirsIpcDataType::Float64, false),
        ]);
        OxirsIpcRecordBatch::new(
            schema,
            vec![
                OxirsIpcColumn::Timestamp(vec![1000, 2000, 3000], OxirsIpcTimeUnit::Millisecond),
                OxirsIpcColumn::Float64(vec![10.0, 20.0, 30.0]),
            ],
        )
        .expect("should succeed")
    }

    #[test]
    fn test_write_read_single_batch() {
        let mut writer = OxirsIpcWriter::new();
        let batch = make_batch();
        let mut stream = writer.write_schema(&batch.schema).expect("should succeed");
        stream.extend(writer.write_batch(&batch).expect("should succeed"));
        stream.extend(writer.write_footer().expect("should succeed"));

        let batches = OxirsIpcReader::read_batches(&stream).expect("should succeed");
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);
        assert_eq!(batches[0].columns[0], batch.columns[0]);
        assert_eq!(batches[0].columns[1], batch.columns[1]);
    }

    #[test]
    fn test_write_read_multiple_batches() {
        let mut writer = OxirsIpcWriter::new();
        let batch = make_batch();
        let mut stream = writer.write_schema(&batch.schema).expect("should succeed");
        stream.extend(writer.write_batch(&batch).expect("should succeed"));
        stream.extend(writer.write_batch(&batch).expect("should succeed"));
        stream.extend(writer.write_footer().expect("should succeed"));

        let batches = OxirsIpcReader::read_batches(&stream).expect("should succeed");
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_write_schema_twice_error() {
        let mut writer = OxirsIpcWriter::new();
        let schema = OxirsIpcSchema::new(vec![OxirsIpcField::new(
            "x",
            OxirsIpcDataType::Int64,
            false,
        )]);
        writer.write_schema(&schema).expect("should succeed");
        let result = writer.write_schema(&schema);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_batch_before_schema_error() {
        let mut writer = OxirsIpcWriter::new();
        let batch = make_batch();
        let result = writer.write_batch(&batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_magic_error() {
        let bad = b"BADMAG\x00\x00\x00".to_vec();
        let result = OxirsIpcReader::read_batches(&bad);
        assert!(result.is_err());
    }

    // ── time_series_to_batch / batch_to_time_series ───────────────────────────

    #[test]
    fn test_time_series_to_batch_columns() {
        let points = vec![
            DataPoint::new(Utc::now(), 1.0),
            DataPoint::new(Utc::now(), 2.0),
        ];
        let batch = OxirsIpcWriter::time_series_to_batch(&points);
        assert_eq!(batch.schema.fields.len(), 2);
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn test_batch_to_time_series_roundtrip() {
        let now = Utc::now();
        let points = vec![DataPoint::new(now, 42.0), DataPoint::new(now, 99.5)];
        let batch = OxirsIpcWriter::time_series_to_batch(&points);
        let recovered = OxirsIpcReader::batch_to_time_series(&batch);
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].value, 42.0);
        assert_eq!(recovered[1].value, 99.5);
    }

    #[test]
    fn test_time_series_to_batch_empty() {
        let batch = OxirsIpcWriter::time_series_to_batch(&[]);
        assert_eq!(batch.num_rows(), 0);
    }

    // ── Tagged data points ────────────────────────────────────────────────────

    #[test]
    fn test_tagged_batch_schema() {
        let points = vec![TaggedDataPoint::new(
            1000,
            std::f64::consts::PI,
            [("host".into(), "srv1".into())].iter().cloned().collect(),
        )];
        let batch = time_series_with_tags_to_batch(&points).expect("should succeed");
        assert_eq!(batch.schema.fields.len(), 3);
        assert_eq!(batch.schema.fields[2].name, "tags_json");
    }

    #[test]
    fn test_tagged_batch_json_content() {
        let mut tags = HashMap::new();
        tags.insert("env".to_owned(), "prod".to_owned());
        let points = vec![TaggedDataPoint::new(0, 1.0, tags)];
        let batch = time_series_with_tags_to_batch(&points).expect("should succeed");
        if let OxirsIpcColumn::Utf8(json_cols) = &batch.columns[2] {
            assert!(json_cols[0].contains("env"), "expected env in tags_json");
        } else {
            panic!("expected Utf8 column");
        }
    }

    // ── OxirsIpcRecordBatch validation ───────────────────────────────────────────

    #[test]
    fn test_record_batch_mismatched_columns_error() {
        let schema = OxirsIpcSchema::new(vec![
            OxirsIpcField::new("a", OxirsIpcDataType::Int64, false),
            OxirsIpcField::new("b", OxirsIpcDataType::Float64, false),
        ]);
        let result = OxirsIpcRecordBatch::new(schema, vec![OxirsIpcColumn::Int64(vec![1, 2])]);
        assert!(result.is_err());
    }

    #[test]
    fn test_record_batch_unequal_column_lengths_error() {
        let schema = OxirsIpcSchema::new(vec![
            OxirsIpcField::new("a", OxirsIpcDataType::Int64, false),
            OxirsIpcField::new("b", OxirsIpcDataType::Float64, false),
        ]);
        let result = OxirsIpcRecordBatch::new(
            schema,
            vec![
                OxirsIpcColumn::Int64(vec![1, 2, 3]),
                OxirsIpcColumn::Float64(vec![1.0]),
            ],
        );
        assert!(result.is_err());
    }

    // ── Full IPC write+read roundtrip with Utf8 ───────────────────────────────

    #[test]
    fn test_write_read_utf8_batch() {
        let schema = OxirsIpcSchema::new(vec![
            OxirsIpcField::new("name", OxirsIpcDataType::Utf8, true),
            OxirsIpcField::new("count", OxirsIpcDataType::Int64, false),
        ]);
        let batch = OxirsIpcRecordBatch::new(
            schema.clone(),
            vec![
                OxirsIpcColumn::Utf8(vec!["alpha".into(), "beta".into(), "gamma".into()]),
                OxirsIpcColumn::Int64(vec![1, 2, 3]),
            ],
        )
        .expect("should succeed");

        let mut writer = OxirsIpcWriter::new();
        let mut stream = writer.write_schema(&schema).expect("should succeed");
        stream.extend(writer.write_batch(&batch).expect("should succeed"));
        stream.extend(writer.write_footer().expect("should succeed"));

        let batches = OxirsIpcReader::read_batches(&stream).expect("should succeed");
        assert_eq!(batches.len(), 1);
        assert_eq!(
            batches[0].columns[0],
            OxirsIpcColumn::Utf8(vec!["alpha".into(), "beta".into(), "gamma".into()])
        );
        assert_eq!(batches[0].columns[1], OxirsIpcColumn::Int64(vec![1, 2, 3]));
    }
}
