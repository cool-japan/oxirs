//! Column-store compression implementation

use crate::compression::{
    AdvancedCompressionType, CompressedData, CompressionAlgorithm, CompressionMetadata,
};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::time::Instant;

/// Column-store compressor with per-column optimization
pub struct ColumnStoreCompressor {
    /// Column definitions
    columns: Vec<ColumnDefinition>,
}

/// Column definition
#[derive(Debug, Clone)]
pub struct ColumnDefinition {
    /// Column name
    pub name: String,
    /// Column data type
    pub data_type: ColumnDataType,
    /// Compression strategy for this column
    pub compression: ColumnCompressionType,
}

/// Column data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnDataType {
    /// 8-bit integer
    Int8,
    /// 16-bit integer
    Int16,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
    /// 32-bit float
    Float32,
    /// 64-bit float
    Float64,
    /// Variable-length string
    String,
    /// Fixed-length binary
    Binary,
}

/// Column compression types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnCompressionType {
    /// No compression
    None,
    /// Run-length encoding (good for repetitive data)
    RunLength,
    /// Delta encoding (good for sequential data)
    Delta,
    /// Dictionary encoding (good for strings)
    Dictionary,
    /// Frame of reference (good for clustered integers)
    FrameOfReference,
}

impl ColumnStoreCompressor {
    /// Create new column store compressor
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    /// Add column definition
    pub fn add_column(&mut self, column: ColumnDefinition) {
        self.columns.push(column);
    }

    /// Analyze column data to determine optimal compression strategy
    pub fn analyze_column(&mut self, column_name: &str, values: &[Vec<u8>]) {
        if values.is_empty() {
            return;
        }

        // Simple analysis to determine best compression type
        let mut unique_values = std::collections::HashSet::new();
        let mut total_size = 0;
        let mut is_sequential = true;
        let mut last_value: Option<u64> = None;

        for value in values {
            unique_values.insert(value.clone());
            total_size += value.len();

            // Check if values are sequential (for delta compression)
            if value.len() >= 8 {
                let current_value = u64::from_le_bytes([
                    value[0], value[1], value[2], value[3],
                    value[4], value[5], value[6], value[7],
                ]);
                
                if let Some(last) = last_value {
                    if current_value != last + 1 {
                        is_sequential = false;
                    }
                }
                last_value = Some(current_value);
            } else {
                is_sequential = false;
            }
        }

        let uniqueness_ratio = unique_values.len() as f64 / values.len() as f64;
        let avg_size = total_size / values.len();

        // Determine optimal compression strategy
        let compression_type = if is_sequential {
            ColumnCompressionType::Delta
        } else if uniqueness_ratio < 0.1 {
            // Many duplicate values - use run-length encoding
            ColumnCompressionType::RunLength
        } else if avg_size > 8 && uniqueness_ratio < 0.5 {
            // Large values with moderate uniqueness - use dictionary
            ColumnCompressionType::Dictionary
        } else {
            // Default to no compression
            ColumnCompressionType::None
        };

        // Find existing column or create new one
        if let Some(column) = self.columns.iter_mut().find(|c| c.name == column_name) {
            column.compression = compression_type;
        } else {
            // Create new column with appropriate data type
            let data_type = if avg_size <= 1 {
                ColumnDataType::Int8
            } else if avg_size <= 2 {
                ColumnDataType::Int16
            } else if avg_size <= 4 {
                ColumnDataType::Int32
            } else if avg_size <= 8 {
                ColumnDataType::Int64
            } else {
                ColumnDataType::Binary
            };

            self.add_column(ColumnDefinition {
                name: column_name.to_string(),
                data_type,
                compression: compression_type,
            });
        }
    }

    /// Compress row-based data to column format
    pub fn compress_rows(&self, rows: &[Vec<u8>]) -> Result<Vec<u8>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        if self.columns.is_empty() {
            return Err(anyhow!("No column definitions provided"));
        }

        let row_count = rows.len();
        let mut compressed = Vec::new();

        // Store metadata
        compressed.extend_from_slice(&(row_count as u32).to_le_bytes());
        compressed.extend_from_slice(&(self.columns.len() as u32).to_le_bytes());

        // Process each column
        for (col_idx, column) in self.columns.iter().enumerate() {
            let mut column_data = Vec::new();

            // Extract column data from all rows
            for row in rows {
                let cell_data = self.extract_cell_data(row, col_idx, &column.data_type)?;
                column_data.extend_from_slice(&cell_data);
            }

            // Compress column data
            let compressed_column = self.compress_column_data(&column_data, column.compression)?;

            // Store compressed column
            compressed.extend_from_slice(&(compressed_column.len() as u32).to_le_bytes());
            compressed.extend_from_slice(&compressed_column);
        }

        Ok(compressed)
    }

    /// Decompress column data back to rows
    pub fn decompress_to_rows(&self, compressed: &[u8]) -> Result<Vec<Vec<u8>>> {
        if compressed.is_empty() {
            return Ok(Vec::new());
        }

        if compressed.len() < 8 {
            return Err(anyhow!("Invalid compressed data"));
        }

        let row_count =
            u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]])
                as usize;

        let col_count =
            u32::from_le_bytes([compressed[4], compressed[5], compressed[6], compressed[7]])
                as usize;

        if col_count != self.columns.len() {
            return Err(anyhow!("Column count mismatch"));
        }

        let mut offset = 8;
        let mut column_data = Vec::new();

        // Decompress each column
        for column in &self.columns {
            if offset + 4 > compressed.len() {
                return Err(anyhow!("Truncated column data"));
            }

            let column_size = u32::from_le_bytes([
                compressed[offset],
                compressed[offset + 1],
                compressed[offset + 2],
                compressed[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + column_size > compressed.len() {
                return Err(anyhow!("Truncated column data"));
            }

            let compressed_column = &compressed[offset..offset + column_size];
            let decompressed_column =
                self.decompress_column_data(compressed_column, column.compression)?;
            column_data.push(decompressed_column);
            offset += column_size;
        }

        // Reconstruct rows
        let mut rows = Vec::new();
        for row_idx in 0..row_count {
            let mut row = Vec::new();
            for (col_idx, column) in self.columns.iter().enumerate() {
                let cell_data =
                    self.extract_row_cell(&column_data[col_idx], row_idx, &column.data_type)?;
                row.extend_from_slice(&cell_data);
            }
            rows.push(row);
        }

        Ok(rows)
    }

    /// Extract cell data from row (simplified implementation)
    fn extract_cell_data(
        &self,
        row: &[u8],
        col_idx: usize,
        data_type: &ColumnDataType,
    ) -> Result<Vec<u8>> {
        let cell_size = match data_type {
            ColumnDataType::Int8 => 1,
            ColumnDataType::Int16 => 2,
            ColumnDataType::Int32 | ColumnDataType::Float32 => 4,
            ColumnDataType::Int64 | ColumnDataType::Float64 => 8,
            ColumnDataType::String | ColumnDataType::Binary => {
                // For simplicity, assume fixed-size cells of 8 bytes
                8
            }
        };

        let start = col_idx * cell_size;
        if start + cell_size > row.len() {
            return Err(anyhow!("Row data too short for column {}", col_idx));
        }

        Ok(row[start..start + cell_size].to_vec())
    }

    /// Extract cell data for specific row from column data
    fn extract_row_cell(
        &self,
        column_data: &[u8],
        row_idx: usize,
        data_type: &ColumnDataType,
    ) -> Result<Vec<u8>> {
        let cell_size = match data_type {
            ColumnDataType::Int8 => 1,
            ColumnDataType::Int16 => 2,
            ColumnDataType::Int32 | ColumnDataType::Float32 => 4,
            ColumnDataType::Int64 | ColumnDataType::Float64 => 8,
            ColumnDataType::String | ColumnDataType::Binary => 8,
        };

        let start = row_idx * cell_size;
        if start + cell_size > column_data.len() {
            return Err(anyhow!("Column data too short for row {}", row_idx));
        }

        Ok(column_data[start..start + cell_size].to_vec())
    }

    /// Compress column data using specified compression
    fn compress_column_data(
        &self,
        data: &[u8],
        compression: ColumnCompressionType,
    ) -> Result<Vec<u8>> {
        match compression {
            ColumnCompressionType::None => Ok(data.to_vec()),
            ColumnCompressionType::RunLength => {
                crate::compression::run_length::RunLengthEncoder::encode(data)
            }
            ColumnCompressionType::Delta => {
                crate::compression::delta::DeltaEncoder::encode_byte_sequence(data)
            }
            _ => {
                // For simplicity, fall back to no compression for other types
                Ok(data.to_vec())
            }
        }
    }

    /// Decompress column data
    fn decompress_column_data(
        &self,
        compressed: &[u8],
        compression: ColumnCompressionType,
    ) -> Result<Vec<u8>> {
        match compression {
            ColumnCompressionType::None => Ok(compressed.to_vec()),
            ColumnCompressionType::RunLength => {
                crate::compression::run_length::RunLengthEncoder::decode(compressed)
            }
            ColumnCompressionType::Delta => {
                crate::compression::delta::DeltaEncoder::decode_byte_sequence(compressed)
            }
            _ => {
                // For simplicity, fall back to no compression for other types
                Ok(compressed.to_vec())
            }
        }
    }
}

impl Default for ColumnStoreCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionAlgorithm for ColumnStoreCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        if self.columns.is_empty() {
            return Err(anyhow!(
                "No column definitions for column store compression"
            ));
        }

        // For simplicity, treat data as single row
        let rows = vec![data.to_vec()];

        let start = Instant::now();
        let compressed_bytes = self.compress_rows(&rows)?;
        let compression_time = start.elapsed();

        let mut metadata_map = HashMap::new();
        metadata_map.insert("columns".to_string(), self.columns.len().to_string());
        metadata_map.insert("rows".to_string(), "1".to_string());

        let metadata = CompressionMetadata {
            algorithm: AdvancedCompressionType::ColumnStore,
            original_size: data.len() as u64,
            compressed_size: compressed_bytes.len() as u64,
            compression_time_us: compression_time.as_micros() as u64,
            metadata: metadata_map,
        };

        Ok(CompressedData {
            data: compressed_bytes,
            metadata,
        })
    }

    fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        if compressed.metadata.algorithm != AdvancedCompressionType::ColumnStore {
            return Err(anyhow!(
                "Invalid compression algorithm: expected ColumnStore, got {}",
                compressed.metadata.algorithm
            ));
        }

        let rows = self.decompress_to_rows(&compressed.data)?;
        if rows.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(rows[0].clone())
        }
    }

    fn algorithm_type(&self) -> AdvancedCompressionType {
        AdvancedCompressionType::ColumnStore
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_store_basic() {
        let mut compressor = ColumnStoreCompressor::new();
        compressor.add_column(ColumnDefinition {
            name: "id".to_string(),
            data_type: ColumnDataType::Int64,
            compression: ColumnCompressionType::None,
        });
        compressor.add_column(ColumnDefinition {
            name: "value".to_string(),
            data_type: ColumnDataType::Int64,
            compression: ColumnCompressionType::Delta,
        });

        // Create test data (2 columns x 8 bytes each = 16 bytes per row)
        let test_data = vec![0u8; 16];
        let compressed = compressor.compress(&test_data).unwrap();
        assert_eq!(
            compressed.metadata.algorithm,
            AdvancedCompressionType::ColumnStore
        );

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(test_data, decompressed);
    }

    #[test]
    fn test_column_definitions() {
        let mut compressor = ColumnStoreCompressor::new();

        compressor.add_column(ColumnDefinition {
            name: "test".to_string(),
            data_type: ColumnDataType::Int32,
            compression: ColumnCompressionType::RunLength,
        });

        assert_eq!(compressor.columns.len(), 1);
        assert_eq!(compressor.columns[0].name, "test");
    }
}
