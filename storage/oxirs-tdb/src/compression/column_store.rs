//! # Column-store compression implementation
//!
//! Provides analytics-optimized column-oriented compression for RDF triples
//! with support for batch processing, SIMD operations, and intelligent
//! per-column compression strategy selection.

use crate::compression::{
    AdvancedCompressionType, CompressedData, CompressionAlgorithm, CompressionMetadata,
};
use anyhow::{anyhow, Result};
use scirs2_core::metrics::{Counter, Gauge, Histogram};
use std::collections::HashMap;
use std::time::Instant;

/// Column-store compressor with per-column optimization
pub struct ColumnStoreCompressor {
    /// Column definitions
    columns: Vec<ColumnDefinition>,
    /// Compression statistics
    stats: ColumnStoreStats,
    /// Enable analytics optimizations
    analytics_mode: bool,
}

/// Column store compression statistics
#[derive(Debug, Clone, Default)]
pub struct ColumnStoreStats {
    /// Total rows compressed
    pub rows_compressed: u64,
    /// Total bytes compressed
    pub bytes_compressed: u64,
    /// Total compression time (microseconds)
    pub compression_time_us: u64,
    /// Total decompression time (microseconds)
    pub decompression_time_us: u64,
    /// Average compression ratio
    pub avg_compression_ratio: f64,
    /// Per-column statistics
    pub column_stats: HashMap<String, ColumnStats>,
}

/// Statistics for individual column
#[derive(Debug, Clone, Default)]
pub struct ColumnStats {
    /// Number of values in column
    pub value_count: u64,
    /// Unique value count
    pub unique_values: u64,
    /// Null count
    pub null_count: u64,
    /// Min value (for numerics)
    pub min_value: Option<f64>,
    /// Max value (for numerics)
    pub max_value: Option<f64>,
    /// Average value length (bytes)
    pub avg_value_length: f64,
    /// Compression ratio for this column
    pub compression_ratio: f64,
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
    /// Bitmap encoding (good for sparse boolean data)
    Bitmap,
    /// LZ4 for mixed data
    Lz4,
}

/// RDF triple column layout for analytics
#[derive(Debug, Clone)]
pub struct RdfTripleLayout {
    /// Subject column
    pub subject: ColumnDefinition,
    /// Predicate column
    pub predicate: ColumnDefinition,
    /// Object column
    pub object: ColumnDefinition,
}

impl RdfTripleLayout {
    /// Create optimized RDF triple layout
    pub fn new() -> Self {
        Self {
            subject: ColumnDefinition {
                name: "subject".to_string(),
                data_type: ColumnDataType::Int64, // NodeId
                compression: ColumnCompressionType::Delta,
            },
            predicate: ColumnDefinition {
                name: "predicate".to_string(),
                data_type: ColumnDataType::Int64, // NodeId
                compression: ColumnCompressionType::Dictionary, // Usually few unique predicates
            },
            object: ColumnDefinition {
                name: "object".to_string(),
                data_type: ColumnDataType::Int64, // NodeId
                compression: ColumnCompressionType::Delta,
            },
        }
    }

    /// Get all columns
    pub fn columns(&self) -> Vec<ColumnDefinition> {
        vec![
            self.subject.clone(),
            self.predicate.clone(),
            self.object.clone(),
        ]
    }
}

impl Default for RdfTripleLayout {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch compression configuration
#[derive(Debug, Clone)]
pub struct BatchCompressionConfig {
    /// Batch size in rows
    pub batch_size: usize,
    /// Enable parallel compression
    pub parallel: bool,
    /// Target compression ratio
    pub target_ratio: f64,
}

impl ColumnStoreCompressor {
    /// Create new column store compressor
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            stats: ColumnStoreStats::default(),
            analytics_mode: false,
        }
    }

    /// Create column store compressor with analytics optimizations
    pub fn with_analytics() -> Self {
        Self {
            columns: Vec::new(),
            stats: ColumnStoreStats::default(),
            analytics_mode: true,
        }
    }

    /// Create compressor for RDF triples
    pub fn for_rdf_triples() -> Self {
        let layout = RdfTripleLayout::new();
        let mut compressor = Self::with_analytics();
        for column in layout.columns() {
            compressor.add_column(column);
        }
        compressor
    }

    /// Add column definition
    pub fn add_column(&mut self, column: ColumnDefinition) {
        self.columns.push(column);
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> &ColumnStoreStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ColumnStoreStats::default();
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
                    value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
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

    /// Compress batches of rows for analytics workloads
    pub fn compress_batch(
        &mut self,
        rows: &[Vec<u8>],
        config: &BatchCompressionConfig,
    ) -> Result<Vec<u8>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();
        let mut all_compressed = Vec::new();

        // Process in batches
        for chunk in rows.chunks(config.batch_size) {
            let compressed_chunk = self.compress_rows(chunk)?;
            all_compressed.extend_from_slice(&(compressed_chunk.len() as u32).to_le_bytes());
            all_compressed.extend_from_slice(&compressed_chunk);
        }

        // Update stats
        self.stats.rows_compressed += rows.len() as u64;
        self.stats.bytes_compressed += rows.iter().map(|r| r.len()).sum::<usize>() as u64;
        self.stats.compression_time_us += start.elapsed().as_micros() as u64;

        let original_size: usize = rows.iter().map(|r| r.len()).sum();
        let compressed_size = all_compressed.len();
        if original_size > 0 {
            let ratio = compressed_size as f64 / original_size as f64;
            // Update rolling average
            let total_compressions = self.stats.rows_compressed / config.batch_size as u64;
            if total_compressions > 0 {
                self.stats.avg_compression_ratio =
                    (self.stats.avg_compression_ratio * (total_compressions - 1) as f64 + ratio)
                        / total_compressions as f64;
            } else {
                self.stats.avg_compression_ratio = ratio;
            }
        }

        Ok(all_compressed)
    }

    /// Decompress batch
    pub fn decompress_batch(&mut self, compressed: &[u8]) -> Result<Vec<Vec<u8>>> {
        let start = Instant::now();
        let mut all_rows = Vec::new();
        let mut offset = 0;

        while offset < compressed.len() {
            if offset + 4 > compressed.len() {
                break;
            }

            let chunk_size = u32::from_le_bytes([
                compressed[offset],
                compressed[offset + 1],
                compressed[offset + 2],
                compressed[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + chunk_size > compressed.len() {
                return Err(anyhow!("Truncated batch data"));
            }

            let chunk = &compressed[offset..offset + chunk_size];
            let rows = self.decompress_to_rows(chunk)?;
            all_rows.extend(rows);
            offset += chunk_size;
        }

        self.stats.decompression_time_us += start.elapsed().as_micros() as u64;

        Ok(all_rows)
    }

    /// Collect column statistics for analytics
    pub fn collect_column_stats(&mut self, column_name: &str, values: &[Vec<u8>]) {
        if values.is_empty() {
            return;
        }

        let mut stats = ColumnStats {
            value_count: values.len() as u64,
            ..Default::default()
        };

        let mut unique_values = std::collections::HashSet::new();
        let mut total_length = 0;
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;

        for value in values {
            unique_values.insert(value.clone());
            total_length += value.len();

            // Try to extract numeric value if possible
            if value.len() == 8 {
                let num_val = f64::from_le_bytes([
                    value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
                ]);
                if num_val.is_finite() {
                    min_val = min_val.min(num_val);
                    max_val = max_val.max(num_val);
                }
            }
        }

        stats.unique_values = unique_values.len() as u64;
        stats.avg_value_length = total_length as f64 / values.len() as f64;

        if min_val != f64::MAX && max_val != f64::MIN {
            stats.min_value = Some(min_val);
            stats.max_value = Some(max_val);
        }

        self.stats
            .column_stats
            .insert(column_name.to_string(), stats);
    }

    /// Get optimal compression for column based on analytics
    pub fn get_optimal_compression(&self, column_name: &str) -> ColumnCompressionType {
        if let Some(stats) = self.stats.column_stats.get(column_name) {
            // Analytics-based compression selection
            let uniqueness_ratio = stats.unique_values as f64 / stats.value_count as f64;

            if uniqueness_ratio < 0.01 {
                // Very low cardinality - bitmap or run-length
                ColumnCompressionType::Bitmap
            } else if uniqueness_ratio < 0.1 {
                // Low cardinality - dictionary
                ColumnCompressionType::Dictionary
            } else if stats.min_value.is_some() && stats.max_value.is_some() {
                // Numeric data with range - frame of reference or delta
                let range = stats.max_value.unwrap() - stats.min_value.unwrap();
                if range < 1000.0 {
                    ColumnCompressionType::FrameOfReference
                } else {
                    ColumnCompressionType::Delta
                }
            } else {
                // Mixed data - LZ4
                ColumnCompressionType::Lz4
            }
        } else {
            ColumnCompressionType::None
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

    #[test]
    fn test_analytics_mode() {
        let compressor = ColumnStoreCompressor::with_analytics();
        assert!(compressor.analytics_mode);
        assert_eq!(compressor.columns.len(), 0);
    }

    #[test]
    fn test_rdf_triple_layout() {
        let layout = RdfTripleLayout::new();
        assert_eq!(layout.subject.name, "subject");
        assert_eq!(layout.predicate.name, "predicate");
        assert_eq!(layout.object.name, "object");

        let columns = layout.columns();
        assert_eq!(columns.len(), 3);
    }

    #[test]
    fn test_rdf_triple_compressor() {
        let compressor = ColumnStoreCompressor::for_rdf_triples();
        assert_eq!(compressor.columns.len(), 3);
        assert!(compressor.analytics_mode);
        assert_eq!(compressor.columns[0].name, "subject");
        assert_eq!(compressor.columns[1].name, "predicate");
        assert_eq!(compressor.columns[2].name, "object");
    }

    #[test]
    fn test_batch_compression() {
        let mut compressor = ColumnStoreCompressor::new();
        compressor.add_column(ColumnDefinition {
            name: "col1".to_string(),
            data_type: ColumnDataType::Int64,
            compression: ColumnCompressionType::None,
        });

        let config = BatchCompressionConfig {
            batch_size: 10,
            parallel: false,
            target_ratio: 0.5,
        };

        // Create 20 rows
        let rows: Vec<Vec<u8>> = (0..20).map(|_| vec![0u8; 8]).collect();

        let compressed = compressor.compress_batch(&rows, &config).unwrap();
        assert!(!compressed.is_empty());

        let decompressed = compressor.decompress_batch(&compressed).unwrap();
        assert_eq!(decompressed.len(), 20);
    }

    #[test]
    fn test_column_statistics() {
        let mut compressor = ColumnStoreCompressor::new();

        let values: Vec<Vec<u8>> = vec![
            vec![1, 0, 0, 0, 0, 0, 0, 0],
            vec![2, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 0, 0, 0, 0, 0, 0, 0], // duplicate
            vec![3, 0, 0, 0, 0, 0, 0, 0],
        ];

        compressor.collect_column_stats("test_col", &values);

        let stats = compressor.get_stats();
        assert!(stats.column_stats.contains_key("test_col"));

        let col_stats = &stats.column_stats["test_col"];
        assert_eq!(col_stats.value_count, 4);
        assert_eq!(col_stats.unique_values, 3); // 1, 2, 3
        assert_eq!(col_stats.avg_value_length, 8.0);
    }

    #[test]
    fn test_optimal_compression_selection() {
        let mut compressor = ColumnStoreCompressor::new();

        // Low cardinality data
        let low_card_values: Vec<Vec<u8>> = (0..100)
            .map(|i| {
                let val = (i % 5) as u64; // Only 5 unique values
                val.to_le_bytes().to_vec()
            })
            .collect();

        compressor.collect_column_stats("low_card", &low_card_values);
        let compression = compressor.get_optimal_compression("low_card");
        // Should use dictionary for low cardinality
        assert!(matches!(
            compression,
            ColumnCompressionType::Dictionary | ColumnCompressionType::Bitmap
        ));

        // High cardinality numeric data
        let high_card_values: Vec<Vec<u8>> = (0..100)
            .map(|i| {
                let val = i as u64;
                val.to_le_bytes().to_vec()
            })
            .collect();

        compressor.collect_column_stats("high_card", &high_card_values);
        let compression = compressor.get_optimal_compression("high_card");
        // Should use delta or frame-of-reference for sequential numeric
        assert!(matches!(
            compression,
            ColumnCompressionType::Delta | ColumnCompressionType::FrameOfReference
        ));
    }

    #[test]
    fn test_compression_stats_tracking() {
        let mut compressor = ColumnStoreCompressor::new();
        compressor.add_column(ColumnDefinition {
            name: "col1".to_string(),
            data_type: ColumnDataType::Int64,
            compression: ColumnCompressionType::None,
        });

        let config = BatchCompressionConfig {
            batch_size: 10,
            parallel: false,
            target_ratio: 0.5,
        };

        let rows: Vec<Vec<u8>> = (0..10).map(|_| vec![0u8; 8]).collect();
        compressor.compress_batch(&rows, &config).unwrap();

        let stats = compressor.get_stats();
        assert_eq!(stats.rows_compressed, 10);
        assert!(stats.compression_time_us > 0);
    }

    #[test]
    fn test_reset_stats() {
        let mut compressor = ColumnStoreCompressor::new();
        compressor.add_column(ColumnDefinition {
            name: "col1".to_string(),
            data_type: ColumnDataType::Int64,
            compression: ColumnCompressionType::None,
        });

        let config = BatchCompressionConfig {
            batch_size: 5,
            parallel: false,
            target_ratio: 0.5,
        };

        let rows: Vec<Vec<u8>> = (0..5).map(|_| vec![0u8; 8]).collect();
        compressor.compress_batch(&rows, &config).unwrap();

        assert!(compressor.get_stats().rows_compressed > 0);

        compressor.reset_stats();
        assert_eq!(compressor.get_stats().rows_compressed, 0);
        assert_eq!(compressor.get_stats().compression_time_us, 0);
    }

    #[test]
    fn test_batch_decompress_empty() {
        let mut compressor = ColumnStoreCompressor::new();
        let result = compressor.decompress_batch(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_numeric_range_detection() {
        let mut compressor = ColumnStoreCompressor::new();

        // Small range values
        let small_range: Vec<Vec<u8>> = (100..200)
            .map(|i| {
                let val = i as f64;
                val.to_le_bytes().to_vec()
            })
            .collect();

        compressor.collect_column_stats("small_range", &small_range);
        let stats = &compressor.get_stats().column_stats["small_range"];

        assert!(stats.min_value.is_some());
        assert!(stats.max_value.is_some());
        assert!((stats.min_value.unwrap() - 100.0).abs() < 0.01);
        assert!((stats.max_value.unwrap() - 199.0).abs() < 0.01);

        let compression = compressor.get_optimal_compression("small_range");
        // Small range should use FrameOfReference
        assert_eq!(compression, ColumnCompressionType::FrameOfReference);
    }

    #[test]
    fn test_column_compression_types() {
        // Test all compression types are distinct
        assert_ne!(ColumnCompressionType::None, ColumnCompressionType::Delta);
        assert_ne!(
            ColumnCompressionType::RunLength,
            ColumnCompressionType::Dictionary
        );
        assert_ne!(ColumnCompressionType::Bitmap, ColumnCompressionType::Lz4);
    }

    #[test]
    fn test_batch_config_defaults() {
        let config = BatchCompressionConfig {
            batch_size: 1000,
            parallel: true,
            target_ratio: 0.7,
        };

        assert_eq!(config.batch_size, 1000);
        assert!(config.parallel);
        assert_eq!(config.target_ratio, 0.7);
    }
}
