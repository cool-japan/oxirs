//! Columnar storage for analytical RDF workloads
//!
//! This module provides columnar storage optimized for analytical queries,
//! supporting efficient aggregations, range scans, and OLAP operations.

use crate::model::{BlankNode, Literal, NamedNode, Triple};
use crate::OxirsError;
use arrow::{
    array::{ArrayBuilder, StringArray, StringBuilder, UInt64Array, UInt64Builder},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::prelude::*;
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Columnar storage configuration
#[derive(Debug, Clone)]
pub struct ColumnarConfig {
    /// Path to columnar data
    pub path: PathBuf,
    /// Batch size for writes
    pub batch_size: usize,
    /// Enable dictionary encoding
    pub dictionary_encoding: bool,
    /// Enable compression
    pub compression: CompressionType,
    /// Partition strategy
    pub partition_strategy: PartitionStrategy,
}

impl Default for ColumnarConfig {
    fn default() -> Self {
        ColumnarConfig {
            path: PathBuf::from("/var/oxirs/columnar"),
            batch_size: 10000,
            dictionary_encoding: true,
            compression: CompressionType::Snappy,
            partition_strategy: PartitionStrategy::ByPredicate,
        }
    }
}

/// Compression type for columnar storage
#[derive(Debug, Clone)]
pub enum CompressionType {
    None,
    Snappy,
    Gzip,
    Lz4,
    Zstd,
}

/// Partition strategy
#[derive(Debug, Clone)]
pub enum PartitionStrategy {
    None,
    ByPredicate,
    ByGraph,
    ByTimeRange { bucket_hours: u32 },
    Custom(String),
}

/// Columnar storage engine
pub struct ColumnarStorage {
    config: ColumnarConfig,
    /// DataFusion execution context
    ctx: Arc<RwLock<SessionContext>>,
    /// Schema for triple storage
    schema: Arc<Schema>,
    /// Dictionary for URI compression
    uri_dictionary: Arc<RwLock<UriDictionary>>,
    /// Active writer
    writer: Arc<RwLock<Option<BatchWriter>>>,
    /// Statistics
    stats: Arc<RwLock<ColumnarStats>>,
}

/// URI dictionary for efficient storage
struct UriDictionary {
    /// URI to ID mapping
    uri_to_id: HashMap<String, u64>,
    /// ID to URI mapping
    id_to_uri: HashMap<u64, String>,
    /// Next ID
    next_id: u64,
}

/// Batch writer for columnar data
struct BatchWriter {
    /// Subject column builder
    subject_builder: UInt64Builder,
    /// Predicate column builder
    predicate_builder: UInt64Builder,
    /// Object type column builder
    object_type_builder: StringBuilder,
    /// Object value column builder
    object_value_builder: StringBuilder,
    /// Object datatype column builder
    object_datatype_builder: UInt64Builder,
    /// Object language column builder
    object_lang_builder: StringBuilder,
    /// Graph column builder
    graph_builder: UInt64Builder,
    /// Timestamp column builder
    timestamp_builder: UInt64Builder,
    /// Current batch size
    current_size: usize,
}

/// Columnar storage statistics
#[derive(Debug, Default)]
struct ColumnarStats {
    /// Total triples stored
    total_triples: u64,
    /// Total partitions
    total_partitions: u64,
    /// Total size in bytes
    total_bytes: u64,
    /// Query count
    query_count: u64,
    /// Average query time
    avg_query_time_ms: f64,
}

impl ColumnarStorage {
    /// Create new columnar storage
    pub async fn new(config: ColumnarConfig) -> Result<Self, OxirsError> {
        // Ensure directory exists
        std::fs::create_dir_all(&config.path)?;

        // Create schema for triple storage
        let schema = Arc::new(Schema::new(vec![
            Field::new("subject_id", DataType::UInt64, false),
            Field::new("predicate_id", DataType::UInt64, false),
            Field::new("object_type", DataType::Utf8, false),
            Field::new("object_value", DataType::Utf8, false),
            Field::new("object_datatype_id", DataType::UInt64, true),
            Field::new("object_lang", DataType::Utf8, true),
            Field::new("graph_id", DataType::UInt64, true),
            Field::new("timestamp", DataType::UInt64, false),
        ]));

        // Create DataFusion context
        let ctx = SessionContext::new();

        // Register existing parquet files
        let pattern = config.path.join("*.parquet");
        if let Ok(paths) = glob::glob(pattern.to_str().unwrap()) {
            for path in paths.flatten() {
                let table_name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("triples");

                ctx.register_parquet(
                    table_name,
                    path.to_str().unwrap(),
                    ParquetReadOptions::default(),
                )
                .await?;
            }
        }

        Ok(ColumnarStorage {
            config,
            ctx: Arc::new(RwLock::new(ctx)),
            schema,
            uri_dictionary: Arc::new(RwLock::new(UriDictionary::new())),
            writer: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(ColumnarStats::default())),
        })
    }

    /// Store a triple in columnar format
    pub async fn store_triple(&self, triple: &Triple) -> Result<(), OxirsError> {
        let mut writer_guard = self.writer.write().await;

        // Initialize writer if needed
        if writer_guard.is_none() {
            *writer_guard = Some(BatchWriter::new());
        }

        let writer = writer_guard.as_mut().unwrap();

        // Get IDs from dictionary
        let mut dict = self.uri_dictionary.write().await;
        
        // Handle subject
        let subject_str = match triple.subject() {
            crate::model::Subject::NamedNode(nn) => nn.as_str(),
            crate::model::Subject::BlankNode(bn) => bn.as_str(),
            crate::model::Subject::Variable(v) => v.as_str(),
            crate::model::Subject::QuotedTriple(_) => "_:quoted", // Simplified for now
        };
        let subject_id = dict.get_or_create_uri(subject_str);
        
        // Handle predicate
        let predicate_str = match triple.predicate() {
            crate::model::Predicate::NamedNode(nn) => nn.as_str(),
            crate::model::Predicate::Variable(v) => v.as_str(),
        };
        let predicate_id = dict.get_or_create_uri(predicate_str);

        // Build columns
        writer.subject_builder.append_value(subject_id);
        writer.predicate_builder.append_value(predicate_id);

        // Handle object based on type
        match triple.object() {
            crate::model::Object::NamedNode(nn) => {
                writer.object_type_builder.append_value("uri");
                writer.object_value_builder.append_value(nn.as_str());
                writer.object_datatype_builder.append_null();
                writer.object_lang_builder.append_null();
            }
            crate::model::Object::BlankNode(bn) => {
                writer.object_type_builder.append_value("bnode");
                writer.object_value_builder.append_value(bn.as_str());
                writer.object_datatype_builder.append_null();
                writer.object_lang_builder.append_null();
            }
            crate::model::Object::Literal(lit) => {
                writer.object_type_builder.append_value("literal");
                writer.object_value_builder.append_value(lit.value());

                // Literals always have a datatype
                let dt = lit.datatype();
                let dt_id = dict.get_or_create_uri(dt.as_str());
                writer.object_datatype_builder.append_value(dt_id);

                if let Some(lang) = lit.language() {
                    writer.object_lang_builder.append_value(lang);
                } else {
                    writer.object_lang_builder.append_null();
                }
            }
            _ => {
                // Handle other object types
                writer.object_type_builder.append_value("other");
                writer.object_value_builder.append_value("");
                writer.object_datatype_builder.append_null();
                writer.object_lang_builder.append_null();
            }
        }

        // Add graph and timestamp
        writer.graph_builder.append_value(0); // Default graph
        writer.timestamp_builder.append_value(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );

        writer.current_size += 1;

        // Flush if batch is full
        if writer.current_size >= self.config.batch_size {
            self.flush_batch(&mut writer_guard).await?;
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_triples += 1;

        Ok(())
    }

    /// Query triples using SQL
    pub async fn query_sql(&self, sql: &str) -> Result<Vec<Triple>, OxirsError> {
        let start = std::time::Instant::now();

        let ctx = self.ctx.read().await;
        let df = ctx.sql(sql).await?;
        let batches = df.collect().await?;

        let mut results = Vec::new();
        let dict = self.uri_dictionary.read().await;

        for batch in batches {
            let subject_ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| OxirsError::Query("Invalid subject column".to_string()))?;

            let predicate_ids = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| OxirsError::Query("Invalid predicate column".to_string()))?;

            // Handle both Utf8 and Utf8View array types
            let object_types_str: Vec<&str> = if let Some(arr) = batch.column(2).as_any().downcast_ref::<StringArray>() {
                (0..batch.num_rows()).map(|i| arr.value(i)).collect()
            } else if let Some(arr) = batch.column(2).as_any().downcast_ref::<arrow::array::StringViewArray>() {
                (0..batch.num_rows()).map(|i| arr.value(i)).collect()
            } else {
                let col_type = batch.column(2).data_type();
                return Err(OxirsError::Query(format!("Invalid object type column type: {:?}, expected Utf8 or Utf8View", col_type)));
            };

            let object_values_str: Vec<&str> = if let Some(arr) = batch.column(3).as_any().downcast_ref::<StringArray>() {
                (0..batch.num_rows()).map(|i| arr.value(i)).collect()
            } else if let Some(arr) = batch.column(3).as_any().downcast_ref::<arrow::array::StringViewArray>() {
                (0..batch.num_rows()).map(|i| arr.value(i)).collect()
            } else {
                let col_type = batch.column(3).data_type();
                return Err(OxirsError::Query(format!("Invalid object value column type: {:?}, expected Utf8 or Utf8View", col_type)));
            };

            for i in 0..batch.num_rows() {
                let subject_id = subject_ids.value(i);
                let predicate_id = predicate_ids.value(i);
                let object_type = object_types_str[i];
                let object_value = object_values_str[i];

                // Reconstruct triple
                let subject_uri = dict.get_uri(subject_id).ok_or_else(|| {
                    OxirsError::Query(format!("Unknown subject ID: {}", subject_id))
                })?;
                let predicate_uri = dict.get_uri(predicate_id).ok_or_else(|| {
                    OxirsError::Query(format!("Unknown predicate ID: {}", predicate_id))
                })?;
                
                // Construct subject
                let subject = if subject_uri.starts_with("_:") {
                    crate::model::Subject::BlankNode(BlankNode::new(subject_uri)?)
                } else {
                    crate::model::Subject::NamedNode(NamedNode::new(subject_uri)?)
                };
                
                // Construct predicate (predicates can only be named nodes in RDF)
                let predicate = crate::model::Predicate::NamedNode(NamedNode::new(predicate_uri)?);

                let object = match object_type {
                    "uri" => crate::model::Object::NamedNode(NamedNode::new(object_value)?),
                    "literal" => crate::model::Object::Literal(Literal::new(object_value)),
                    _ => continue, // Skip unknown types
                };

                results.push(Triple::new(subject, predicate, object));
            }
        }

        // Update stats
        let elapsed = start.elapsed();
        let mut stats = self.stats.write().await;
        stats.query_count += 1;
        stats.avg_query_time_ms = (stats.avg_query_time_ms * (stats.query_count - 1) as f64
            + elapsed.as_millis() as f64)
            / stats.query_count as f64;

        Ok(results)
    }

    /// Execute analytical query
    pub async fn analyze(&self, query: AnalyticalQuery) -> Result<AnalysisResult, OxirsError> {
        match query {
            AnalyticalQuery::CountByPredicate => {
                let sql = "SELECT predicate_id, COUNT(*) as count 
                          FROM triples 
                          GROUP BY predicate_id 
                          ORDER BY count DESC";

                let ctx = self.ctx.read().await;
                let df = ctx.sql(sql).await?;
                let batches = df.collect().await?;

                let mut counts = HashMap::new();
                let dict = self.uri_dictionary.read().await;

                for batch in batches {
                    let predicate_ids = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap();
                    // COUNT(*) returns Int64, not UInt64
                    let count_values = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<arrow::array::Int64Array>()
                        .ok_or_else(|| OxirsError::Query("Invalid count column type".to_string()))?;

                    for i in 0..batch.num_rows() {
                        let pred_id = predicate_ids.value(i);
                        let count = count_values.value(i) as u64;

                        if let Some(pred_uri) = dict.get_uri(pred_id) {
                            counts.insert(pred_uri.to_string(), count);
                        }
                    }
                }

                Ok(AnalysisResult::PredicateCounts(counts))
            }

            AnalyticalQuery::TopSubjects { limit } => {
                let sql = format!(
                    "SELECT subject_id, COUNT(*) as count 
                     FROM triples 
                     GROUP BY subject_id 
                     ORDER BY count DESC 
                     LIMIT {}",
                    limit
                );

                let ctx = self.ctx.read().await;
                let df = ctx.sql(&sql).await?;
                let batches = df.collect().await?;

                let mut subjects = Vec::new();
                let dict = self.uri_dictionary.read().await;

                for batch in batches {
                    let subject_ids = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap();
                    // COUNT(*) returns Int64, not UInt64
                    let count_values = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<arrow::array::Int64Array>()
                        .ok_or_else(|| OxirsError::Query("Invalid count column type".to_string()))?;

                    for i in 0..batch.num_rows() {
                        let subj_id = subject_ids.value(i);
                        let count = count_values.value(i) as u64;

                        if let Some(subj_uri) = dict.get_uri(subj_id) {
                            subjects.push((subj_uri.to_string(), count));
                        }
                    }
                }

                Ok(AnalysisResult::TopSubjects(subjects))
            }

            AnalyticalQuery::TimeSeriesAnalysis {
                predicate,
                interval,
            } => {
                // Implement time series analysis
                Ok(AnalysisResult::TimeSeries(Vec::new()))
            }
        }
    }

    /// Flush current batch to disk
    async fn flush_batch(&self, writer_guard: &mut Option<BatchWriter>) -> Result<(), OxirsError> {
        if let Some(mut writer) = writer_guard.take() {
            // Create record batch
            let batch = RecordBatch::try_new(
                self.schema.clone(),
                vec![
                    Arc::new(writer.subject_builder.finish()),
                    Arc::new(writer.predicate_builder.finish()),
                    Arc::new(writer.object_type_builder.finish()),
                    Arc::new(writer.object_value_builder.finish()),
                    Arc::new(writer.object_datatype_builder.finish()),
                    Arc::new(writer.object_lang_builder.finish()),
                    Arc::new(writer.graph_builder.finish()),
                    Arc::new(writer.timestamp_builder.finish()),
                ],
            )?;

            // Write to parquet file
            let partition = self.get_partition_name();
            let path = self.config.path.join(format!("{}.parquet", partition));

            let file = std::fs::File::create(&path)?;
            let props = WriterProperties::builder()
                .set_compression(self.get_parquet_compression())
                .build();

            let mut writer = ArrowWriter::try_new(file, self.schema.clone(), Some(props))?;
            writer.write(&batch)?;
            writer.close()?;

            // Register new file with DataFusion
            let ctx = self.ctx.write().await;
            ctx.register_parquet(
                &partition,
                path.to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
            
            // Create or update the unified triples view
            self.create_triples_view(&ctx).await?;

            // Update stats
            let mut stats = self.stats.write().await;
            stats.total_partitions += 1;
            stats.total_bytes += std::fs::metadata(&path)?.len();
        }

        Ok(())
    }

    /// Get partition name based on strategy
    fn get_partition_name(&self) -> String {
        match &self.config.partition_strategy {
            PartitionStrategy::None => "triples".to_string(),
            PartitionStrategy::ByPredicate => {
                // In real implementation, would partition by predicate
                format!("triples_{}_{}", 
                    chrono::Utc::now().format("%Y%m%d_%H%M%S"),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos())
            }
            PartitionStrategy::ByGraph => {
                format!("graph_default_{}", chrono::Utc::now().format("%Y%m%d"))
            }
            PartitionStrategy::ByTimeRange { bucket_hours } => {
                let now = chrono::Utc::now();
                let bucket = now.timestamp() / (*bucket_hours as i64 * 3600);
                format!("time_bucket_{}", bucket)
            }
            PartitionStrategy::Custom(name) => name.clone(),
        }
    }

    /// Get Parquet compression type
    fn get_parquet_compression(&self) -> parquet::basic::Compression {
        match self.config.compression {
            CompressionType::None => parquet::basic::Compression::UNCOMPRESSED,
            CompressionType::Snappy => parquet::basic::Compression::SNAPPY,
            CompressionType::Gzip => parquet::basic::Compression::GZIP(parquet::basic::GzipLevel::default()),
            CompressionType::Lz4 => parquet::basic::Compression::LZ4,
            CompressionType::Zstd => parquet::basic::Compression::ZSTD(parquet::basic::ZstdLevel::default()),
        }
    }
    
    /// Create or update the unified triples view
    async fn create_triples_view(&self, ctx: &SessionContext) -> Result<(), OxirsError> {
        // Get all registered tables
        let tables = ctx.catalog_names();
        let mut triple_tables = Vec::new();
        
        // Find all tables that are triple partitions
        for catalog in tables {
            let schemas = ctx.catalog(&catalog).unwrap().schema_names();
            for schema in schemas {
                let tables = ctx.catalog(&catalog).unwrap()
                    .schema(&schema).unwrap()
                    .table_names();
                for table in tables {
                    // Skip the view itself to avoid circular reference
                    if table == "triples" {
                        continue;
                    }
                    if table.starts_with("triples") || table.contains("time_bucket") || table.contains("graph_") {
                        triple_tables.push(format!("{}.{}.{}", catalog, schema, table));
                    }
                }
            }
        }
        
        // Create a UNION ALL view if we have tables
        if !triple_tables.is_empty() {
            let union_query = triple_tables
                .iter()
                .map(|t| format!("SELECT * FROM {}", t))
                .collect::<Vec<_>>()
                .join(" UNION ALL ");
            
            let create_view_sql = format!("CREATE OR REPLACE VIEW triples AS {}", union_query);
            ctx.sql(&create_view_sql).await?;
        }
        
        Ok(())
    }

}

impl UriDictionary {
    fn new() -> Self {
        UriDictionary {
            uri_to_id: HashMap::new(),
            id_to_uri: HashMap::new(),
            next_id: 1,
        }
    }

    fn get_or_create_uri(&mut self, uri: &str) -> u64 {
        if let Some(&id) = self.uri_to_id.get(uri) {
            return id;
        }

        let id = self.next_id;
        self.uri_to_id.insert(uri.to_string(), id);
        self.id_to_uri.insert(id, uri.to_string());
        self.next_id += 1;
        id
    }

    fn get_uri(&self, id: u64) -> Option<&str> {
        self.id_to_uri.get(&id).map(|s| s.as_str())
    }

    fn get_term(&self, id: u64) -> Option<crate::model::Subject> {
        self.get_uri(id).and_then(|uri| {
            if uri.starts_with("_:") {
                BlankNode::new(uri)
                    .ok()
                    .map(crate::model::Subject::BlankNode)
            } else {
                NamedNode::new(uri)
                    .ok()
                    .map(crate::model::Subject::NamedNode)
            }
        })
    }
}

impl BatchWriter {
    fn new() -> Self {
        BatchWriter {
            subject_builder: UInt64Builder::new(),
            predicate_builder: UInt64Builder::new(),
            object_type_builder: StringBuilder::new(),
            object_value_builder: StringBuilder::new(),
            object_datatype_builder: UInt64Builder::new(),
            object_lang_builder: StringBuilder::new(),
            graph_builder: UInt64Builder::new(),
            timestamp_builder: UInt64Builder::new(),
            current_size: 0,
        }
    }
}

/// Analytical query types
#[derive(Debug, Clone)]
pub enum AnalyticalQuery {
    /// Count triples by predicate
    CountByPredicate,
    /// Get top N subjects by triple count
    TopSubjects { limit: usize },
    /// Time series analysis
    TimeSeriesAnalysis {
        predicate: String,
        interval: TimeInterval,
    },
}

/// Time interval for analysis
#[derive(Debug, Clone)]
pub enum TimeInterval {
    Hour,
    Day,
    Week,
    Month,
}

/// Analysis result
#[derive(Debug)]
pub enum AnalysisResult {
    PredicateCounts(HashMap<String, u64>),
    TopSubjects(Vec<(String, u64)>),
    TimeSeries(Vec<(chrono::DateTime<chrono::Utc>, f64)>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_columnar_storage() {
        let test_dir = format!("/tmp/oxirs_columnar_test_{}", 
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis());
        let config = ColumnarConfig {
            path: PathBuf::from(&test_dir),
            ..Default::default()
        };

        let storage = ColumnarStorage::new(config).await.unwrap();

        // Create test triple
        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            crate::model::Object::Literal(Literal::new("test")),
        );

        // Store triple
        storage.store_triple(&triple).await.unwrap();

        // Flush to ensure it's written
        {
            let mut writer_guard = storage.writer.write().await;
            storage
                .flush_batch(&mut writer_guard)
                .await
                .unwrap();
        }

        // Query using SQL
        let results = storage
            .query_sql("SELECT * FROM triples WHERE object_value = 'test'")
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_analytical_queries() {
        let test_dir = format!("/tmp/oxirs_columnar_analytics_{}", 
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis());
        let config = ColumnarConfig {
            path: PathBuf::from(&test_dir),
            batch_size: 2,
            ..Default::default()
        };

        let storage = ColumnarStorage::new(config).await.unwrap();

        // Store multiple triples
        let predicates = ["p1", "p1", "p2", "p1", "p3"];
        for (i, pred) in predicates.iter().enumerate() {
            let triple = Triple::new(
                NamedNode::new(format!("http://example.org/s{}", i)).unwrap(),
                NamedNode::new(format!("http://example.org/{}", pred)).unwrap(),
                crate::model::Object::Literal(Literal::new(format!("value{}", i))),
            );
            storage.store_triple(&triple).await.unwrap();
        }

        // Flush remaining
        {
            let mut writer_guard = storage.writer.write().await;
            storage
                .flush_batch(&mut writer_guard)
                .await
                .unwrap();
        }

        // Count by predicate
        let result = storage
            .analyze(AnalyticalQuery::CountByPredicate)
            .await
            .unwrap();

        if let AnalysisResult::PredicateCounts(counts) = result {
            assert_eq!(counts.get("http://example.org/p1"), Some(&3));
            assert_eq!(counts.get("http://example.org/p2"), Some(&1));
            assert_eq!(counts.get("http://example.org/p3"), Some(&1));
        } else {
            panic!("Unexpected result type");
        }
    }
}
