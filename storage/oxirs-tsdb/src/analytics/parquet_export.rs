//! Pure-Rust simplified Parquet-compatible columnar export.
//!
//! Implements a Parquet-inspired binary format for exporting time-series data
//! without requiring any external C/Fortran library.  The format is a
//! simplified (but structurally faithful) Parquet-like encoding:
//!
//! ## Wire Format
//!
//! ```text
//! [MAGIC: 4 bytes "PAR1"]
//! [column chunk 0 ... N]
//! [file footer]
//! [footer_len: u32 LE]
//! [MAGIC: 4 bytes "PAR1"]
//! ```
//!
//! Each column chunk:
//! ```text
//! [col_name_len: u16 LE] [col_name: utf-8]
//! [value_type: u8]  0=Int64, 1=Double, 2=ByteArray
//! [compression: u8] 0=None, 1=Snappy(sim), 2=Gzip(sim)
//! [n_values: u32 LE]
//! [data_len: u32 LE]
//! [data: data_len bytes]
//! ```
//!
//! File footer:
//! ```text
//! [n_columns: u32 LE]
//! ```

use crate::error::{TsdbError, TsdbResult};
use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

// ──────────────────────────────────────────────────────────────────────────────
// ParquetCompression
// ──────────────────────────────────────────────────────────────────────────────

/// Compression codec applied to each column chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ParquetCompression {
    /// No compression — raw data bytes.
    #[default]
    None,
    /// Simulated Snappy compression (identity in this pure-Rust impl).
    Snappy,
    /// Simulated Gzip compression (identity in this pure-Rust impl).
    Gzip,
}

impl ParquetCompression {
    fn code(&self) -> u8 {
        match self {
            ParquetCompression::None => 0,
            ParquetCompression::Snappy => 1,
            ParquetCompression::Gzip => 2,
        }
    }

    fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(ParquetCompression::None),
            1 => Some(ParquetCompression::Snappy),
            2 => Some(ParquetCompression::Gzip),
            _ => None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ParquetValues
// ──────────────────────────────────────────────────────────────────────────────

/// Typed value array for a Parquet column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParquetValues {
    /// 64-bit signed integer column.
    Int64(Vec<i64>),
    /// 64-bit IEEE 754 double column.
    Double(Vec<f64>),
    /// Variable-length byte array column.
    ByteArray(Vec<Vec<u8>>),
}

impl ParquetValues {
    fn type_code(&self) -> u8 {
        match self {
            ParquetValues::Int64(_) => 0,
            ParquetValues::Double(_) => 1,
            ParquetValues::ByteArray(_) => 2,
        }
    }

    fn n_values(&self) -> usize {
        match self {
            ParquetValues::Int64(v) => v.len(),
            ParquetValues::Double(v) => v.len(),
            ParquetValues::ByteArray(v) => v.len(),
        }
    }

    /// Serialise the value array to raw bytes.
    ///
    /// - Int64 / Double: packed little-endian 8-byte values.
    /// - ByteArray: `[n_values: u32 LE] + per-value [len: u32 LE] [bytes]`.
    fn encode(&self) -> Vec<u8> {
        match self {
            ParquetValues::Int64(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            ParquetValues::Double(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            ParquetValues::ByteArray(v) => {
                let mut buf = Vec::new();
                buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for item in v {
                    buf.extend_from_slice(&(item.len() as u32).to_le_bytes());
                    buf.extend_from_slice(item);
                }
                buf
            }
        }
    }

    /// Deserialise a value array from bytes given a type code and expected count.
    fn decode(bytes: &[u8], type_code: u8, n_values: usize) -> TsdbResult<Self> {
        match type_code {
            0 => {
                // Int64
                if bytes.len() < n_values * 8 {
                    return Err(TsdbError::Arrow("Parquet Int64: buffer too short".into()));
                }
                let values: Vec<i64> = (0..n_values)
                    .map(|i| {
                        i64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap_or_default())
                    })
                    .collect();
                Ok(ParquetValues::Int64(values))
            }
            1 => {
                // Double
                if bytes.len() < n_values * 8 {
                    return Err(TsdbError::Arrow("Parquet Double: buffer too short".into()));
                }
                let values: Vec<f64> = (0..n_values)
                    .map(|i| {
                        f64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap_or_default())
                    })
                    .collect();
                Ok(ParquetValues::Double(values))
            }
            2 => {
                // ByteArray
                if bytes.len() < 4 {
                    return Err(TsdbError::Arrow("Parquet ByteArray: missing count".into()));
                }
                let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap_or_default()) as usize;
                if n != n_values {
                    return Err(TsdbError::Arrow(format!(
                        "Parquet ByteArray: expected {n_values} items, encoded {n}"
                    )));
                }
                let mut pos = 4usize;
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    if pos + 4 > bytes.len() {
                        return Err(TsdbError::Arrow(
                            "Parquet ByteArray: buffer truncated reading item length".into(),
                        ));
                    }
                    let item_len =
                        u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap_or_default())
                            as usize;
                    pos += 4;
                    if pos + item_len > bytes.len() {
                        return Err(TsdbError::Arrow(
                            "Parquet ByteArray: buffer truncated reading item data".into(),
                        ));
                    }
                    items.push(bytes[pos..pos + item_len].to_vec());
                    pos += item_len;
                }
                Ok(ParquetValues::ByteArray(items))
            }
            other => Err(TsdbError::Arrow(format!(
                "Parquet: unknown value type code {other}"
            ))),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ParquetColumn
// ──────────────────────────────────────────────────────────────────────────────

/// A named, typed column in a Parquet file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParquetColumn {
    /// Column name.
    pub name: String,
    /// Typed values.
    pub values: ParquetValues,
}

impl ParquetColumn {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, values: ParquetValues) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }

    /// Number of values in this column.
    pub fn len(&self) -> usize {
        self.values.n_values()
    }

    /// Returns `true` if the column has no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ParquetWriter
// ──────────────────────────────────────────────────────────────────────────────

/// Writer for the simplified pure-Rust Parquet format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParquetWriter {
    /// Compression codec to apply to each column chunk.
    pub compression: ParquetCompression,
}

impl ParquetWriter {
    /// Create a new writer with the given compression codec.
    pub fn new(compression: ParquetCompression) -> Self {
        Self { compression }
    }

    /// Serialise a list of columns into the wire format.
    pub fn write_columns(&self, columns: &[ParquetColumn]) -> TsdbResult<Vec<u8>> {
        let mut buf = Vec::new();

        // Magic header.
        buf.extend_from_slice(PARQUET_MAGIC);

        // Column chunks.
        for col in columns {
            self.write_column(&mut buf, col)?;
        }

        // Footer: just the column count.
        let footer_start = buf.len();
        buf.extend_from_slice(&(columns.len() as u32).to_le_bytes());
        let footer_len = (buf.len() - footer_start) as u32;

        // Footer length + magic trailer.
        buf.extend_from_slice(&footer_len.to_le_bytes());
        buf.extend_from_slice(PARQUET_MAGIC);

        Ok(buf)
    }

    /// Convenience: convert `(timestamp, value, metric_name)` triples.
    ///
    /// Produces three columns: `timestamp` (Int64), `value` (Double),
    /// `metric_name` (ByteArray).
    pub fn time_series_to_parquet(series: &[(i64, f64, String)]) -> TsdbResult<Vec<u8>> {
        let writer = ParquetWriter::new(ParquetCompression::None);
        let timestamps: Vec<i64> = series.iter().map(|(t, _, _)| *t).collect();
        let values: Vec<f64> = series.iter().map(|(_, v, _)| *v).collect();
        let names: Vec<Vec<u8>> = series
            .iter()
            .map(|(_, _, n)| n.as_bytes().to_vec())
            .collect();

        let columns = vec![
            ParquetColumn::new("timestamp", ParquetValues::Int64(timestamps)),
            ParquetColumn::new("value", ParquetValues::Double(values)),
            ParquetColumn::new("metric_name", ParquetValues::ByteArray(names)),
        ];
        writer.write_columns(&columns)
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn write_column(&self, buf: &mut Vec<u8>, col: &ParquetColumn) -> TsdbResult<()> {
        let name_bytes = col.name.as_bytes();
        if name_bytes.len() > u16::MAX as usize {
            return Err(TsdbError::Arrow(format!(
                "Parquet: column name '{}' too long ({} bytes, max 65535)",
                col.name,
                name_bytes.len()
            )));
        }

        // Column name.
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(name_bytes);

        // Type & compression.
        buf.push(col.values.type_code());
        buf.push(self.compression.code());

        // Encode values (compression is simulated as identity).
        let data = col.values.encode();

        // n_values and data_len.
        buf.extend_from_slice(&(col.values.n_values() as u32).to_le_bytes());
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());

        buf.extend_from_slice(&data);
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ParquetReader
// ──────────────────────────────────────────────────────────────────────────────

/// Reader for the simplified pure-Rust Parquet format produced by
/// [`ParquetWriter`].
#[derive(Debug, Default)]
pub struct ParquetReader;

impl ParquetReader {
    /// Parse columns from bytes previously produced by [`ParquetWriter::write_columns`].
    pub fn read_columns(data: &[u8]) -> TsdbResult<Vec<ParquetColumn>> {
        // Minimum: 4 (magic) + 4 (footer: col count) + 4 (footer_len) + 4 (magic trailer) = 16
        if data.len() < 16 {
            return Err(TsdbError::Arrow(
                "Parquet: data too short to contain valid file".into(),
            ));
        }

        // Check header magic.
        if &data[0..4] != PARQUET_MAGIC {
            return Err(TsdbError::Arrow("Parquet: invalid magic header".into()));
        }
        // Check trailer magic.
        let trailer_start = data.len() - 4;
        if &data[trailer_start..] != PARQUET_MAGIC {
            return Err(TsdbError::Arrow("Parquet: invalid magic trailer".into()));
        }

        // Read footer_len (4 bytes before trailer magic).
        let footer_len_pos = data.len() - 8;
        let footer_len = u32::from_le_bytes(
            data[footer_len_pos..footer_len_pos + 4]
                .try_into()
                .map_err(|_| TsdbError::Arrow("Parquet: cannot read footer_len".into()))?,
        ) as usize;

        let footer_start = data.len() - 4 - 4 - footer_len;
        let n_columns = u32::from_le_bytes(
            data[footer_start..footer_start + 4]
                .try_into()
                .map_err(|_| TsdbError::Arrow("Parquet: cannot read n_columns".into()))?,
        ) as usize;

        // Parse column chunks from the body (after header magic).
        let mut pos = 4usize;
        let body_end = footer_start;
        let mut columns = Vec::with_capacity(n_columns);

        while pos < body_end && columns.len() < n_columns {
            let (col, consumed) = Self::read_column(&data[pos..body_end])?;
            columns.push(col);
            pos += consumed;
        }

        if columns.len() != n_columns {
            return Err(TsdbError::Arrow(format!(
                "Parquet: footer claimed {n_columns} columns, parsed {}",
                columns.len()
            )));
        }

        Ok(columns)
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn read_column(data: &[u8]) -> TsdbResult<(ParquetColumn, usize)> {
        let mut pos = 0usize;

        // Column name.
        if pos + 2 > data.len() {
            return Err(TsdbError::Arrow(
                "Parquet column: buffer too short for name length".into(),
            ));
        }
        let name_len =
            u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap_or_default()) as usize;
        pos += 2;

        if pos + name_len > data.len() {
            return Err(TsdbError::Arrow(
                "Parquet column: buffer too short for name".into(),
            ));
        }
        let name = std::str::from_utf8(&data[pos..pos + name_len])
            .map_err(|e| TsdbError::Arrow(format!("Parquet column: invalid name UTF-8: {e}")))?
            .to_owned();
        pos += name_len;

        // Type code, compression, n_values, data_len.
        if pos + 10 > data.len() {
            return Err(TsdbError::Arrow(
                "Parquet column: buffer too short for metadata".into(),
            ));
        }
        let type_code = data[pos];
        let _compression = ParquetCompression::from_code(data[pos + 1]).unwrap_or_default();
        pos += 2;

        let n_values =
            u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap_or_default()) as usize;
        pos += 4;
        let data_len =
            u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap_or_default()) as usize;
        pos += 4;

        if pos + data_len > data.len() {
            return Err(TsdbError::Arrow(
                "Parquet column: buffer too short for column data".into(),
            ));
        }
        let values = ParquetValues::decode(&data[pos..pos + data_len], type_code, n_values)?;
        pos += data_len;

        Ok((ParquetColumn { name, values }, pos))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_columns() -> Vec<ParquetColumn> {
        vec![
            ParquetColumn::new("timestamp", ParquetValues::Int64(vec![1000, 2000, 3000])),
            ParquetColumn::new("value", ParquetValues::Double(vec![1.1, 2.2, 3.3])),
            ParquetColumn::new(
                "label",
                ParquetValues::ByteArray(vec![b"cpu".to_vec(), b"mem".to_vec(), b"disk".to_vec()]),
            ),
        ]
    }

    // ── ParquetWriter / ParquetReader roundtrips ──────────────────────────────

    #[test]
    fn test_write_read_no_compression() {
        let writer = ParquetWriter::new(ParquetCompression::None);
        let cols = make_columns();
        let bytes = writer.write_columns(&cols).expect("should succeed");
        let decoded = ParquetReader::read_columns(&bytes).expect("should succeed");
        assert_eq!(cols, decoded);
    }

    #[test]
    fn test_write_read_snappy_compression() {
        let writer = ParquetWriter::new(ParquetCompression::Snappy);
        let cols = make_columns();
        let bytes = writer.write_columns(&cols).expect("should succeed");
        let decoded = ParquetReader::read_columns(&bytes).expect("should succeed");
        assert_eq!(cols, decoded);
    }

    #[test]
    fn test_write_read_gzip_compression() {
        let writer = ParquetWriter::new(ParquetCompression::Gzip);
        let cols = make_columns();
        let bytes = writer.write_columns(&cols).expect("should succeed");
        let decoded = ParquetReader::read_columns(&bytes).expect("should succeed");
        assert_eq!(cols, decoded);
    }

    #[test]
    fn test_write_read_empty_columns() {
        let writer = ParquetWriter::new(ParquetCompression::None);
        let bytes = writer.write_columns(&[]).expect("should succeed");
        let decoded = ParquetReader::read_columns(&bytes).expect("should succeed");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_write_read_int64_only() {
        let writer = ParquetWriter::new(ParquetCompression::None);
        let cols = vec![ParquetColumn::new(
            "ts",
            ParquetValues::Int64(vec![i64::MIN, 0, i64::MAX]),
        )];
        let bytes = writer.write_columns(&cols).expect("should succeed");
        let decoded = ParquetReader::read_columns(&bytes).expect("should succeed");
        assert_eq!(cols, decoded);
    }

    #[test]
    fn test_write_read_double_only() {
        let writer = ParquetWriter::new(ParquetCompression::None);
        let cols = vec![ParquetColumn::new(
            "val",
            ParquetValues::Double(vec![0.0, -1.0, f64::INFINITY]),
        )];
        let bytes = writer.write_columns(&cols).expect("should succeed");
        let decoded = ParquetReader::read_columns(&bytes).expect("should succeed");
        if let (ParquetValues::Double(orig), ParquetValues::Double(dec)) =
            (&cols[0].values, &decoded[0].values)
        {
            assert_eq!(orig[0], dec[0]);
            assert_eq!(orig[1], dec[1]);
            assert!(dec[2].is_infinite());
        } else {
            panic!("unexpected type");
        }
    }

    #[test]
    fn test_write_read_byte_array() {
        let writer = ParquetWriter::new(ParquetCompression::None);
        let cols = vec![ParquetColumn::new(
            "data",
            ParquetValues::ByteArray(vec![vec![0u8, 1, 2, 3], vec![], b"hello world".to_vec()]),
        )];
        let bytes = writer.write_columns(&cols).expect("should succeed");
        let decoded = ParquetReader::read_columns(&bytes).expect("should succeed");
        assert_eq!(cols, decoded);
    }

    #[test]
    fn test_time_series_to_parquet_roundtrip() {
        let series = vec![
            (1000i64, 1.5f64, "cpu_usage".to_owned()),
            (2000, 2.5, "cpu_usage".to_owned()),
            (3000, 0.5, "mem_free".to_owned()),
        ];
        let bytes = ParquetWriter::time_series_to_parquet(&series).expect("should succeed");
        let cols = ParquetReader::read_columns(&bytes).expect("should succeed");
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].name, "timestamp");
        assert_eq!(cols[1].name, "value");
        assert_eq!(cols[2].name, "metric_name");
        if let ParquetValues::Int64(ts) = &cols[0].values {
            assert_eq!(ts, &[1000, 2000, 3000]);
        } else {
            panic!("expected Int64 column");
        }
    }

    #[test]
    fn test_invalid_magic_error() {
        let bad_data = b"BADM\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00".to_vec();
        let result = ParquetReader::read_columns(&bad_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_too_short_error() {
        let result = ParquetReader::read_columns(&[0u8; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_len() {
        let col = ParquetColumn::new("x", ParquetValues::Int64(vec![1, 2, 3]));
        assert_eq!(col.len(), 3);
        assert!(!col.is_empty());
    }

    #[test]
    fn test_column_is_empty() {
        let col = ParquetColumn::new("x", ParquetValues::Int64(vec![]));
        assert!(col.is_empty());
    }
}
