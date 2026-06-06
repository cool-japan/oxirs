use super::arrow_export_types::{ExportMetadata, ExportStats, ExportedPoint, ParquetCompression};
#[cfg(feature = "arrow-export")]
use crate::error::{TsdbError, TsdbResult};
use std::collections::HashMap;

#[cfg(feature = "arrow-export")]
use {
    arrow::{
        array::{Float64Array, Int64Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    std::sync::Arc,
};

#[cfg(feature = "arrow-export")]
use {
    parquet::{
        arrow::ArrowWriter, basic::Compression as PqCompression, file::properties::WriterProperties,
    },
    std::{fs::File, path::Path},
};

#[derive(Debug, Default)]
pub struct ArrowExporter {
    max_rows_per_batch: usize,
}

impl ArrowExporter {
    pub fn new() -> Self {
        Self {
            max_rows_per_batch: 0,
        }
    }

    pub fn with_max_rows(max_rows: usize) -> Self {
        Self {
            max_rows_per_batch: max_rows,
        }
    }

    pub fn max_rows_per_batch(&self) -> usize {
        self.max_rows_per_batch
    }

    #[cfg(feature = "arrow-export")]
    pub fn export_batch(&self, points: &[ExportedPoint]) -> TsdbResult<RecordBatch> {
        let schema = Self::schema();

        let timestamps: Int64Array = points.iter().map(|p| p.timestamp_ms).collect();
        let metrics: StringArray = points.iter().map(|p| Some(p.metric.as_str())).collect();
        let values: Float64Array = points.iter().map(|p| p.value).collect();
        let tags: StringArray = points.iter().map(|p| Some(p.tags_json.as_str())).collect();

        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(timestamps),
                Arc::new(metrics),
                Arc::new(values),
                Arc::new(tags),
            ],
        )
        .map_err(|e| TsdbError::Arrow(e.to_string()))
    }

    #[cfg(feature = "arrow-export")]
    pub fn export_batches(&self, points: &[ExportedPoint]) -> TsdbResult<Vec<RecordBatch>> {
        if points.is_empty() {
            return Ok(vec![]);
        }
        let chunk_size = if self.max_rows_per_batch == 0 {
            points.len()
        } else {
            self.max_rows_per_batch
        };
        points
            .chunks(chunk_size)
            .map(|chunk| self.export_batch(chunk))
            .collect()
    }

    #[cfg(not(feature = "arrow-export"))]
    pub fn export_batch_count(&self, points: &[ExportedPoint]) -> usize {
        points.len()
    }

    #[cfg(feature = "arrow-export")]
    pub fn schema() -> Schema {
        Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("metric", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
            Field::new("tags_json", DataType::Utf8, false),
        ])
    }

    pub fn filter_by_metric<'a>(
        points: &'a [ExportedPoint],
        metric: &str,
    ) -> Vec<&'a ExportedPoint> {
        points.iter().filter(|p| p.metric == metric).collect()
    }

    pub fn filter_by_time_range(
        points: &[ExportedPoint],
        start_ms: i64,
        end_ms: i64,
    ) -> Vec<ExportedPoint> {
        points
            .iter()
            .filter(|p| p.timestamp_ms >= start_ms && p.timestamp_ms <= end_ms)
            .cloned()
            .collect()
    }

    pub fn compute_stats(points: &[ExportedPoint]) -> ExportStats {
        if points.is_empty() {
            return ExportStats::default();
        }
        let count = points.len();
        let sum: f64 = points.iter().map(|p| p.value).sum();
        let mean = sum / count as f64;
        let min = points.iter().map(|p| p.value).fold(f64::INFINITY, f64::min);
        let max = points
            .iter()
            .map(|p| p.value)
            .fold(f64::NEG_INFINITY, f64::max);
        let variance = if count > 1 {
            points.iter().map(|p| (p.value - mean).powi(2)).sum::<f64>() / (count - 1) as f64
        } else {
            0.0
        };
        let first_ts = points.iter().map(|p| p.timestamp_ms).min().unwrap_or(0);
        let last_ts = points.iter().map(|p| p.timestamp_ms).max().unwrap_or(0);
        let distinct_metrics: std::collections::HashSet<&str> =
            points.iter().map(|p| p.metric.as_str()).collect();
        ExportStats {
            count,
            sum,
            mean,
            min,
            max,
            variance,
            stddev: variance.sqrt(),
            first_timestamp_ms: first_ts,
            last_timestamp_ms: last_ts,
            distinct_metrics: distinct_metrics.len(),
        }
    }

    pub fn group_by_metric(points: &[ExportedPoint]) -> HashMap<String, Vec<ExportedPoint>> {
        let mut groups: HashMap<String, Vec<ExportedPoint>> = HashMap::new();
        for p in points {
            groups.entry(p.metric.clone()).or_default().push(p.clone());
        }
        groups
    }

    pub fn sort_by_timestamp(points: &mut [ExportedPoint]) {
        points.sort_by_key(|p| p.timestamp_ms);
    }
}

#[derive(Debug)]
pub struct ParquetExporter {
    #[cfg_attr(not(feature = "arrow-export"), allow(dead_code))]
    arrow: ArrowExporter,
    compression: ParquetCompression,
    row_group_size: usize,
}

impl ParquetExporter {
    pub fn new() -> Self {
        Self {
            arrow: ArrowExporter::new(),
            compression: ParquetCompression::Snappy,
            row_group_size: 134_217_728,
        }
    }

    pub fn with_compression(mut self, codec: ParquetCompression) -> Self {
        self.compression = codec;
        self
    }

    pub fn with_row_group_size(mut self, bytes: usize) -> Self {
        self.row_group_size = bytes;
        self
    }

    pub fn compression(&self) -> ParquetCompression {
        self.compression
    }

    pub fn row_group_size(&self) -> usize {
        self.row_group_size
    }

    #[cfg(feature = "arrow-export")]
    pub fn write_file(&self, points: &[ExportedPoint], path: &Path) -> TsdbResult<u64> {
        let batch = self.arrow.export_batch(points)?;

        let codec = match self.compression {
            ParquetCompression::None => PqCompression::UNCOMPRESSED,
            ParquetCompression::Snappy => PqCompression::SNAPPY,
            ParquetCompression::Zstd => PqCompression::ZSTD(Default::default()),
            ParquetCompression::Gzip => PqCompression::GZIP(Default::default()),
        };

        let props = WriterProperties::builder()
            .set_compression(codec)
            .set_max_row_group_row_count(Some(self.row_group_size / 8))
            .build();

        let file = File::create(path).map_err(|e| TsdbError::Io(e.to_string()))?;

        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
            .map_err(|e| TsdbError::Arrow(e.to_string()))?;

        writer
            .write(&batch)
            .map_err(|e| TsdbError::Arrow(e.to_string()))?;

        let metadata = writer
            .close()
            .map_err(|e| TsdbError::Arrow(e.to_string()))?;

        Ok(metadata.file_metadata().num_rows() as u64)
    }

    pub fn count_rows(&self, points: &[ExportedPoint]) -> usize {
        points.len()
    }

    pub fn export_metadata(&self, points: &[ExportedPoint]) -> ExportMetadata {
        let stats = ArrowExporter::compute_stats(points);
        ExportMetadata {
            row_count: points.len(),
            compression: self.compression,
            row_group_size: self.row_group_size,
            distinct_metrics: stats.distinct_metrics,
            time_span_ms: stats
                .last_timestamp_ms
                .saturating_sub(stats.first_timestamp_ms),
        }
    }
}

impl Default for ParquetExporter {
    fn default() -> Self {
        Self::new()
    }
}
