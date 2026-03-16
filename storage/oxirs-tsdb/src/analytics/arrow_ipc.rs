//! Pure-Rust Apache Arrow IPC format export (simulation).
//!
//! Implements a faithful simulation of the Apache Arrow IPC stream format
//! (see Arrow spec §IPC) in pure Rust without the `arrow` crate.
//!
//! ## Wire Format (simplified)
//!
//! ```text
//! [MAGIC: 6 bytes "ARROW1"] [padding: 2 bytes]
//! [schema message]
//! [record batch messages ...]
//! [EOS marker: 4 bytes 0xFFFFFFFF + 4 bytes 0x00000000]
//! ```
//!
//! Each message is:
//! ```text
//! [continuation: 4 bytes 0xFFFFFFFF]
//! [metadata_size: i32 LE]
//! [metadata: metadata_size bytes (flatbuffer-style but simplified)]
//! [body: aligned to 8 bytes]
//! ```

use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

/// Magic bytes for Arrow IPC stream.
const ARROW_MAGIC: &[u8] = b"ARROW1";
/// Padding after magic to align to 8 bytes.
const ARROW_MAGIC_PADDING: &[u8] = &[0u8, 0u8];
/// Continuation marker (used in IPC stream format before each message).
const CONTINUATION_MARKER: u32 = 0xFFFF_FFFF;

// ──────────────────────────────────────────────────────────────────────────────
// TimeUnit
// ──────────────────────────────────────────────────────────────────────────────

/// Timestamp resolution unit for Arrow Timestamp type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeUnit {
    /// 1-second resolution.
    Second,
    /// 1-millisecond resolution.
    Millisecond,
    /// 1-microsecond resolution.
    Microsecond,
    /// 1-nanosecond resolution.
    Nanosecond,
}

impl TimeUnit {
    fn code(&self) -> u8 {
        match self {
            TimeUnit::Second => 0,
            TimeUnit::Millisecond => 1,
            TimeUnit::Microsecond => 2,
            TimeUnit::Nanosecond => 3,
        }
    }

    fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(TimeUnit::Second),
            1 => Some(TimeUnit::Millisecond),
            2 => Some(TimeUnit::Microsecond),
            3 => Some(TimeUnit::Nanosecond),
            _ => None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ArrowDataType
// ──────────────────────────────────────────────────────────────────────────────

/// Subset of Arrow data types supported by this exporter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArrowDataType {
    /// 64-bit signed integer.
    Int64,
    /// 64-bit IEEE 754 float.
    Float64,
    /// UTF-8 variable-length string.
    Utf8,
    /// 64-bit timestamp with a given time unit.
    Timestamp(TimeUnit),
    /// Boolean (bit-packed in Arrow; stored as bytes here for simplicity).
    Boolean,
}

impl ArrowDataType {
    /// Numeric type tag embedded in the wire format.
    fn type_tag(&self) -> u8 {
        match self {
            ArrowDataType::Int64 => 1,
            ArrowDataType::Float64 => 2,
            ArrowDataType::Utf8 => 3,
            ArrowDataType::Timestamp(_) => 4,
            ArrowDataType::Boolean => 5,
        }
    }

    fn from_tag(tag: u8, time_unit: u8) -> Option<Self> {
        match tag {
            1 => Some(ArrowDataType::Int64),
            2 => Some(ArrowDataType::Float64),
            3 => Some(ArrowDataType::Utf8),
            4 => TimeUnit::from_code(time_unit).map(ArrowDataType::Timestamp),
            5 => Some(ArrowDataType::Boolean),
            _ => None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ArrowField
// ──────────────────────────────────────────────────────────────────────────────

/// A single column descriptor in an [`ArrowSchema`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrowField {
    /// Column name.
    pub name: String,
    /// Column data type.
    pub data_type: ArrowDataType,
    /// Whether the column may contain nulls.
    pub nullable: bool,
}

impl ArrowField {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, data_type: ArrowDataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ArrowSchema
// ──────────────────────────────────────────────────────────────────────────────

/// Schema describing the columns of an Arrow record batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrowSchema {
    /// Ordered list of fields.
    pub fields: Vec<ArrowField>,
}

impl ArrowSchema {
    /// Create a new schema from a list of fields.
    pub fn new(fields: Vec<ArrowField>) -> Self {
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
                ArrowDataType::Timestamp(u) => u.code(),
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
                "ArrowSchema: buffer too short for field count".into(),
            ));
        }
        let n_fields = u32::from_le_bytes(
            src[0..4]
                .try_into()
                .map_err(|_| TsdbError::Arrow("ArrowSchema: cannot read field count".into()))?,
        ) as usize;
        let mut fields = Vec::with_capacity(n_fields);
        let mut pos = 4usize;
        for _ in 0..n_fields {
            if pos + 4 > src.len() {
                return Err(TsdbError::Arrow(
                    "ArrowSchema: buffer truncated reading name length".into(),
                ));
            }
            let name_len = u32::from_le_bytes(
                src[pos..pos + 4]
                    .try_into()
                    .map_err(|_| TsdbError::Arrow("ArrowSchema: cannot read name length".into()))?,
            ) as usize;
            pos += 4;
            if pos + name_len > src.len() {
                return Err(TsdbError::Arrow(
                    "ArrowSchema: buffer truncated reading name".into(),
                ));
            }
            let name = std::str::from_utf8(&src[pos..pos + name_len])
                .map_err(|e| TsdbError::Arrow(format!("ArrowSchema: invalid UTF-8 in name: {e}")))?
                .to_owned();
            pos += name_len;
            if pos + 4 > src.len() {
                return Err(TsdbError::Arrow(
                    "ArrowSchema: buffer truncated reading type tag".into(),
                ));
            }
            let tag = src[pos];
            let tu = src[pos + 1];
            let nullable = src[pos + 2] != 0;
            pos += 4;
            let data_type = ArrowDataType::from_tag(tag, tu)
                .ok_or_else(|| TsdbError::Arrow(format!("ArrowSchema: unknown type tag {tag}")))?;
            fields.push(ArrowField {
                name,
                data_type,
                nullable,
            });
        }
        Ok(ArrowSchema { fields })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ArrowColumn
// ──────────────────────────────────────────────────────────────────────────────

/// A typed column in an Arrow record batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArrowColumn {
    /// Array of 64-bit signed integers.
    Int64(Vec<i64>),
    /// Array of 64-bit floats.
    Float64(Vec<f64>),
    /// Array of UTF-8 strings.
    Utf8(Vec<String>),
    /// Array of 64-bit timestamps with a time unit.
    Timestamp(Vec<i64>, TimeUnit),
    /// Array of booleans.
    Boolean(Vec<bool>),
}

impl ArrowColumn {
    /// Number of elements in this column.
    pub fn len(&self) -> usize {
        match self {
            ArrowColumn::Int64(v) => v.len(),
            ArrowColumn::Float64(v) => v.len(),
            ArrowColumn::Utf8(v) => v.len(),
            ArrowColumn::Timestamp(v, _) => v.len(),
            ArrowColumn::Boolean(v) => v.len(),
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
            ArrowColumn::Int64(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            ArrowColumn::Float64(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            ArrowColumn::Timestamp(v, _) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            ArrowColumn::Boolean(v) => v.iter().map(|&b| b as u8).collect(),
            ArrowColumn::Utf8(v) => {
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
        data_type: &ArrowDataType,
        n_rows: usize,
    ) -> TsdbResult<ArrowColumn> {
        match data_type {
            ArrowDataType::Int64 => {
                if bytes.len() < n_rows * 8 {
                    return Err(TsdbError::Arrow("Int64 column: buffer too short".into()));
                }
                let values: Vec<i64> = (0..n_rows)
                    .map(|i| {
                        i64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap_or_default())
                    })
                    .collect();
                Ok(ArrowColumn::Int64(values))
            }
            ArrowDataType::Float64 => {
                if bytes.len() < n_rows * 8 {
                    return Err(TsdbError::Arrow("Float64 column: buffer too short".into()));
                }
                let values: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        f64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap_or_default())
                    })
                    .collect();
                Ok(ArrowColumn::Float64(values))
            }
            ArrowDataType::Timestamp(unit) => {
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
                Ok(ArrowColumn::Timestamp(values, *unit))
            }
            ArrowDataType::Boolean => {
                if bytes.len() < n_rows {
                    return Err(TsdbError::Arrow("Boolean column: buffer too short".into()));
                }
                let values: Vec<bool> = bytes[..n_rows].iter().map(|&b| b != 0).collect();
                Ok(ArrowColumn::Boolean(values))
            }
            ArrowDataType::Utf8 => {
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
                Ok(ArrowColumn::Utf8(strings))
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ArrowRecordBatch
// ──────────────────────────────────────────────────────────────────────────────

/// A collection of columnar arrays with a shared schema and equal row counts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArrowRecordBatch {
    /// Schema describing all columns.
    pub schema: ArrowSchema,
    /// Parallel array of columns; must have the same number of elements.
    pub columns: Vec<ArrowColumn>,
}

impl ArrowRecordBatch {
    /// Create a new record batch after validating schema/column alignment.
    pub fn new(schema: ArrowSchema, columns: Vec<ArrowColumn>) -> TsdbResult<Self> {
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
// ArrowIpcWriter
// ──────────────────────────────────────────────────────────────────────────────

/// Writer for the Arrow IPC stream format.
///
/// Maintains state across `write_schema` / `write_batch` / `write_footer` calls
/// so that the caller can stream batches incrementally.
#[derive(Debug, Default)]
pub struct ArrowIpcWriter {
    /// Whether the schema has already been written.
    schema_written: bool,
    /// Number of batches written.
    batches_written: usize,
}

impl ArrowIpcWriter {
    /// Create a fresh writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Emit the IPC stream header (magic + schema message).
    ///
    /// Must be called exactly once before `write_batch`.
    pub fn write_schema(&mut self, schema: &ArrowSchema) -> TsdbResult<Vec<u8>> {
        if self.schema_written {
            return Err(TsdbError::Arrow(
                "ArrowIpcWriter: schema already written".into(),
            ));
        }
        let mut buf = Vec::new();
        // Magic
        buf.extend_from_slice(ARROW_MAGIC);
        buf.extend_from_slice(ARROW_MAGIC_PADDING);
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
    pub fn write_batch(&mut self, batch: &ArrowRecordBatch) -> TsdbResult<Vec<u8>> {
        if !self.schema_written {
            return Err(TsdbError::Arrow(
                "ArrowIpcWriter: schema not yet written".into(),
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

    /// Convert a slice of `DataPoint`s into an [`ArrowRecordBatch`].
    ///
    /// Produces two columns:
    /// - `timestamp` : Timestamp(Millisecond)
    /// - `value`     : Float64
    ///
    /// Tags are not included (use [`time_series_with_tags_to_batch`] for that).
    pub fn time_series_to_batch(series: &[crate::series::DataPoint]) -> ArrowRecordBatch {
        let timestamps: Vec<i64> = series
            .iter()
            .map(|p| p.timestamp.timestamp_millis())
            .collect();
        let values: Vec<f64> = series.iter().map(|p| p.value).collect();

        let schema = ArrowSchema::new(vec![
            ArrowField::new(
                "timestamp",
                ArrowDataType::Timestamp(TimeUnit::Millisecond),
                false,
            ),
            ArrowField::new("value", ArrowDataType::Float64, false),
        ]);

        ArrowRecordBatch {
            schema,
            columns: vec![
                ArrowColumn::Timestamp(timestamps, TimeUnit::Millisecond),
                ArrowColumn::Float64(values),
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
pub fn time_series_with_tags_to_batch(series: &[TaggedDataPoint]) -> TsdbResult<ArrowRecordBatch> {
    let timestamps: Vec<i64> = series.iter().map(|p| p.timestamp).collect();
    let values: Vec<f64> = series.iter().map(|p| p.value).collect();
    let tags_json: Vec<String> = series
        .iter()
        .map(|p| serde_json::to_string(&p.tags).unwrap_or_else(|_| "{}".to_owned()))
        .collect();

    let schema = ArrowSchema::new(vec![
        ArrowField::new(
            "timestamp",
            ArrowDataType::Timestamp(TimeUnit::Millisecond),
            false,
        ),
        ArrowField::new("value", ArrowDataType::Float64, false),
        ArrowField::new("tags_json", ArrowDataType::Utf8, true),
    ]);

    ArrowRecordBatch::new(
        schema,
        vec![
            ArrowColumn::Timestamp(timestamps, TimeUnit::Millisecond),
            ArrowColumn::Float64(values),
            ArrowColumn::Utf8(tags_json),
        ],
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// ArrowIpcReader
// ──────────────────────────────────────────────────────────────────────────────

/// Reader for Arrow IPC stream bytes produced by [`ArrowIpcWriter`].
#[derive(Debug, Default)]
pub struct ArrowIpcReader;

impl ArrowIpcReader {
    /// Parse all record batches from an IPC byte stream.
    ///
    /// The stream must start with the 8-byte magic, followed by a schema
    /// message, then zero or more record-batch messages, and finally an EOS
    /// marker.
    pub fn read_batches(data: &[u8]) -> TsdbResult<Vec<ArrowRecordBatch>> {
        let mut pos = 0usize;

        // 1. Validate magic.
        if data.len() < 8 {
            return Err(TsdbError::Arrow("IPC stream too short for magic".into()));
        }
        if &data[pos..pos + 6] != ARROW_MAGIC {
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
        let schema = ArrowSchema::from_bytes(&schema_bytes)?;
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
                let col =
                    ArrowColumn::from_bytes(&data[pos..pos + col_len], &field.data_type, n_rows)?;
                columns.push(col);
                // Advance past column body + 8-byte alignment padding.
                let aligned = col_len + (8 - col_len % 8) % 8;
                pos += aligned;
            }

            let batch = ArrowRecordBatch {
                schema: schema.clone(),
                columns,
            };
            batches.push(batch);
        }

        Ok(batches)
    }

    /// Convert a record batch (with `timestamp` and `value` columns) back to
    /// `DataPoint`s.
    pub fn batch_to_time_series(batch: &ArrowRecordBatch) -> Vec<crate::series::DataPoint> {
        use chrono::{TimeZone, Utc};

        // Find timestamp and value columns by position (schema order).
        let timestamps = batch.columns.first().and_then(|c| {
            if let ArrowColumn::Timestamp(v, _) = c {
                Some(v.as_slice())
            } else {
                None
            }
        });
        let values = batch.columns.get(1).and_then(|c| {
            if let ArrowColumn::Float64(v) = c {
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
        let schema = ArrowSchema::new(vec![ArrowField::new(
            "value",
            ArrowDataType::Float64,
            false,
        )]);
        let bytes = schema.to_bytes();
        let decoded = ArrowSchema::from_bytes(&bytes).expect("roundtrip");
        assert_eq!(schema, decoded);
    }

    #[test]
    fn test_schema_roundtrip_multiple_fields() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("ts", ArrowDataType::Timestamp(TimeUnit::Millisecond), false),
            ArrowField::new("value", ArrowDataType::Float64, false),
            ArrowField::new("label", ArrowDataType::Utf8, true),
            ArrowField::new("active", ArrowDataType::Boolean, false),
            ArrowField::new("count", ArrowDataType::Int64, false),
        ]);
        let bytes = schema.to_bytes();
        let decoded = ArrowSchema::from_bytes(&bytes).expect("roundtrip");
        assert_eq!(schema, decoded);
    }

    #[test]
    fn test_schema_empty_fields() {
        let schema = ArrowSchema::new(vec![]);
        let bytes = schema.to_bytes();
        let decoded = ArrowSchema::from_bytes(&bytes).expect("roundtrip");
        assert!(decoded.fields.is_empty());
    }

    #[test]
    fn test_schema_from_bytes_truncated_error() {
        let result = ArrowSchema::from_bytes(&[0u8; 2]);
        assert!(result.is_err());
    }

    // ── Column serialization ──────────────────────────────────────────────────

    #[test]
    fn test_int64_column_roundtrip() {
        let col = ArrowColumn::Int64(vec![1, -2, 1_000_000, i64::MAX]);
        let bytes = col.body_bytes();
        let decoded =
            ArrowColumn::from_bytes(&bytes, &ArrowDataType::Int64, 4).expect("should succeed");
        assert_eq!(col, decoded);
    }

    #[test]
    fn test_float64_column_roundtrip() {
        let col = ArrowColumn::Float64(vec![1.5, -std::f64::consts::PI, f64::NAN, f64::INFINITY]);
        let bytes = col.body_bytes();
        let decoded =
            ArrowColumn::from_bytes(&bytes, &ArrowDataType::Float64, 4).expect("should succeed");
        // NaN != NaN so compare element-by-element.
        if let (ArrowColumn::Float64(orig), ArrowColumn::Float64(dec)) = (&col, &decoded) {
            assert_eq!(orig.len(), dec.len());
            assert_eq!(dec[0], 1.5);
            assert!(dec[2].is_nan());
        } else {
            panic!("unexpected column type");
        }
    }

    #[test]
    fn test_utf8_column_roundtrip() {
        let col = ArrowColumn::Utf8(vec!["hello".into(), "world".into(), "".into()]);
        let bytes = col.body_bytes();
        let decoded =
            ArrowColumn::from_bytes(&bytes, &ArrowDataType::Utf8, 3).expect("should succeed");
        assert_eq!(col, decoded);
    }

    #[test]
    fn test_timestamp_column_roundtrip() {
        let col = ArrowColumn::Timestamp(vec![0, 1_000, -500], TimeUnit::Millisecond);
        let bytes = col.body_bytes();
        let decoded =
            ArrowColumn::from_bytes(&bytes, &ArrowDataType::Timestamp(TimeUnit::Millisecond), 3)
                .expect("should succeed");
        assert_eq!(col, decoded);
    }

    #[test]
    fn test_boolean_column_roundtrip() {
        let col = ArrowColumn::Boolean(vec![true, false, true, true, false]);
        let bytes = col.body_bytes();
        let decoded =
            ArrowColumn::from_bytes(&bytes, &ArrowDataType::Boolean, 5).expect("should succeed");
        assert_eq!(col, decoded);
    }

    // ── ArrowIpcWriter / ArrowIpcReader ───────────────────────────────────────

    fn make_batch() -> ArrowRecordBatch {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("ts", ArrowDataType::Timestamp(TimeUnit::Millisecond), false),
            ArrowField::new("val", ArrowDataType::Float64, false),
        ]);
        ArrowRecordBatch::new(
            schema,
            vec![
                ArrowColumn::Timestamp(vec![1000, 2000, 3000], TimeUnit::Millisecond),
                ArrowColumn::Float64(vec![10.0, 20.0, 30.0]),
            ],
        )
        .expect("should succeed")
    }

    #[test]
    fn test_write_read_single_batch() {
        let mut writer = ArrowIpcWriter::new();
        let batch = make_batch();
        let mut stream = writer.write_schema(&batch.schema).expect("should succeed");
        stream.extend(writer.write_batch(&batch).expect("should succeed"));
        stream.extend(writer.write_footer().expect("should succeed"));

        let batches = ArrowIpcReader::read_batches(&stream).expect("should succeed");
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);
        assert_eq!(batches[0].columns[0], batch.columns[0]);
        assert_eq!(batches[0].columns[1], batch.columns[1]);
    }

    #[test]
    fn test_write_read_multiple_batches() {
        let mut writer = ArrowIpcWriter::new();
        let batch = make_batch();
        let mut stream = writer.write_schema(&batch.schema).expect("should succeed");
        stream.extend(writer.write_batch(&batch).expect("should succeed"));
        stream.extend(writer.write_batch(&batch).expect("should succeed"));
        stream.extend(writer.write_footer().expect("should succeed"));

        let batches = ArrowIpcReader::read_batches(&stream).expect("should succeed");
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_write_schema_twice_error() {
        let mut writer = ArrowIpcWriter::new();
        let schema = ArrowSchema::new(vec![ArrowField::new("x", ArrowDataType::Int64, false)]);
        writer.write_schema(&schema).expect("should succeed");
        let result = writer.write_schema(&schema);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_batch_before_schema_error() {
        let mut writer = ArrowIpcWriter::new();
        let batch = make_batch();
        let result = writer.write_batch(&batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_magic_error() {
        let bad = b"BADMAG\x00\x00\x00".to_vec();
        let result = ArrowIpcReader::read_batches(&bad);
        assert!(result.is_err());
    }

    // ── time_series_to_batch / batch_to_time_series ───────────────────────────

    #[test]
    fn test_time_series_to_batch_columns() {
        let points = vec![
            DataPoint::new(Utc::now(), 1.0),
            DataPoint::new(Utc::now(), 2.0),
        ];
        let batch = ArrowIpcWriter::time_series_to_batch(&points);
        assert_eq!(batch.schema.fields.len(), 2);
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn test_batch_to_time_series_roundtrip() {
        let now = Utc::now();
        let points = vec![DataPoint::new(now, 42.0), DataPoint::new(now, 99.5)];
        let batch = ArrowIpcWriter::time_series_to_batch(&points);
        let recovered = ArrowIpcReader::batch_to_time_series(&batch);
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].value, 42.0);
        assert_eq!(recovered[1].value, 99.5);
    }

    #[test]
    fn test_time_series_to_batch_empty() {
        let batch = ArrowIpcWriter::time_series_to_batch(&[]);
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
        if let ArrowColumn::Utf8(json_cols) = &batch.columns[2] {
            assert!(json_cols[0].contains("env"), "expected env in tags_json");
        } else {
            panic!("expected Utf8 column");
        }
    }

    // ── ArrowRecordBatch validation ───────────────────────────────────────────

    #[test]
    fn test_record_batch_mismatched_columns_error() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int64, false),
            ArrowField::new("b", ArrowDataType::Float64, false),
        ]);
        let result = ArrowRecordBatch::new(schema, vec![ArrowColumn::Int64(vec![1, 2])]);
        assert!(result.is_err());
    }

    #[test]
    fn test_record_batch_unequal_column_lengths_error() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int64, false),
            ArrowField::new("b", ArrowDataType::Float64, false),
        ]);
        let result = ArrowRecordBatch::new(
            schema,
            vec![
                ArrowColumn::Int64(vec![1, 2, 3]),
                ArrowColumn::Float64(vec![1.0]),
            ],
        );
        assert!(result.is_err());
    }

    // ── Full IPC write+read roundtrip with Utf8 ───────────────────────────────

    #[test]
    fn test_write_read_utf8_batch() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("name", ArrowDataType::Utf8, true),
            ArrowField::new("count", ArrowDataType::Int64, false),
        ]);
        let batch = ArrowRecordBatch::new(
            schema.clone(),
            vec![
                ArrowColumn::Utf8(vec!["alpha".into(), "beta".into(), "gamma".into()]),
                ArrowColumn::Int64(vec![1, 2, 3]),
            ],
        )
        .expect("should succeed");

        let mut writer = ArrowIpcWriter::new();
        let mut stream = writer.write_schema(&schema).expect("should succeed");
        stream.extend(writer.write_batch(&batch).expect("should succeed"));
        stream.extend(writer.write_footer().expect("should succeed"));

        let batches = ArrowIpcReader::read_batches(&stream).expect("should succeed");
        assert_eq!(batches.len(), 1);
        assert_eq!(
            batches[0].columns[0],
            ArrowColumn::Utf8(vec!["alpha".into(), "beta".into(), "gamma".into()])
        );
        assert_eq!(batches[0].columns[1], ArrowColumn::Int64(vec![1, 2, 3]));
    }
}
