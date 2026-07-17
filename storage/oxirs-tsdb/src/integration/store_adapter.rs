//! Hybrid RDF + Time-Series Store Adapter
//!
//! Implements the oxirs-core Store trait with automatic routing between
//! RDF storage (for semantic metadata) and time-series storage (for high-frequency data).

use crate::error::{TsdbError, TsdbResult};
use crate::series::DataPoint;
use crate::storage::{ColumnarStore, TimeChunk};
use crate::write::wal::WriteAheadLog;
use crate::write_buffer::{DataPoint as BufferPoint, FlushPolicy, WriteBuffer, WriteBufferConfig};
use chrono::{DateTime, Duration, Utc};
use oxirs_core::model::{GraphName, Object, Predicate, Quad, Subject};
use oxirs_core::rdf_store::{OxirsQueryResults, PreparedQuery, RdfStore, Store};
use oxirs_core::{OxirsError, RdfTerm, Result as OxirsResult};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

/// RDF-star annotation predicates recognized as carrying an explicit
/// observation timestamp for a quoted time-series triple, e.g.:
///
/// ```text
/// << :sensor1 qudt:numericValue "42.5"^^xsd:double >>
///     prov:generatedAtTime "2024-01-01T00:00:00Z"^^xsd:dateTime .
/// ```
///
/// Without this annotation, `HybridStore`'s `Store::insert_quad` stamps the point
/// with the current wall-clock time, which is correct for live ingestion but
/// WRONG for historical backfills. Backfill loaders must use this annotated
/// form (or call [`HybridStore::insert_ts`] directly) to preserve the true
/// observation time instead of silently getting `Utc::now()`.
const TIMESTAMP_ANNOTATION_PREDICATES: &[&str] = &[
    "http://www.w3.org/ns/prov#generatedAtTime",
    "http://www.w3.org/ns/sosa/resultTime",
    "http://example.org/ts/timestamp",
];

/// Default number of buffered points across all series before an automatic
/// flush into durable columnar storage.
const DEFAULT_FLUSH_THRESHOLD: usize = 1_000;

/// Default in-memory capacity of the staging write buffer before
/// backpressure kicks in.
const DEFAULT_BUFFER_CAPACITY: usize = 100_000;

/// Durable write path for time-series points: a Write-Ahead Log paired with
/// an in-memory staging buffer.
///
/// Both are held under a single lock so that a WAL truncation (which happens
/// once buffered points are safely compacted into [`ColumnarStore`]) can
/// never race ahead of a concurrent insert that has logged to the WAL but
/// not yet reached the buffer.
#[derive(Debug)]
struct TsWritePath {
    wal: WriteAheadLog,
    buffer: WriteBuffer,
}

/// Hybrid store that combines RDF and time-series storage
///
/// Automatically routes data based on predicates:
/// - Time-series predicates → TSDB storage
/// - Everything else → RDF storage
///
/// ## Time-Series Predicates
///
/// The following predicates are routed to time-series storage:
/// - `qudt:numericValue` - QUDT numeric values
/// - `sosa:hasSimpleResult` - SOSA sensor observations
/// - `ts:value` - Custom time-series namespace
///
/// ## Durability
///
/// Time-series points are logged to a Write-Ahead Log before being
/// acknowledged, staged in a bounded in-memory [`WriteBuffer`], and
/// periodically compacted into durable, compressed [`ColumnarStore`] chunk
/// files -- the crate's real storage pipeline. Nothing is held in an
/// unbounded in-memory map, and a crash between writes and the next flush
/// is recoverable by replaying the WAL.
#[derive(Debug)]
pub struct HybridStore {
    /// RDF store for semantic metadata
    rdf_store: Arc<RwLock<RdfStore>>,

    /// Durable, compressed columnar storage for compacted time-series chunks.
    columnar_store: Arc<ColumnarStore>,

    /// WAL + staging buffer for points not yet compacted into `columnar_store`.
    ts_write_path: Arc<Mutex<TsWritePath>>,

    /// Mapping from RDF subjects (sensor URIs) to time-series IDs
    subject_to_series: Arc<RwLock<HashMap<String, u64>>>,

    /// Reverse mapping from series IDs to RDF subjects
    series_to_subject: Arc<RwLock<HashMap<u64, String>>>,

    /// Next series ID to assign
    next_series_id: Arc<RwLock<u64>>,
}

impl HybridStore {
    /// Create a new hybrid store with time-series data durably persisted
    /// under a fresh, uniquely-named directory in the system temp directory.
    ///
    /// This is a convenient default for quick starts, demos and tests. For a
    /// production deployment, use [`HybridStore::with_storage_path`] to
    /// control the on-disk location (and reuse it across restarts so the WAL
    /// / columnar chunks can be recovered).
    pub fn new() -> TsdbResult<Self> {
        let unique = format!(
            "oxirs-tsdb-hybridstore-{}-{}",
            std::process::id(),
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        );
        Self::with_storage_path(std::env::temp_dir().join(unique))
    }

    /// Create a new hybrid store with time-series data durably persisted
    /// under `base_path`.
    ///
    /// `base_path` holds the WAL file, the series index, and one
    /// sub-directory of compressed chunk files per series. If it already
    /// contains data from a previous run, the WAL is replayed into the
    /// staging buffer so no unflushed points from before a crash are lost.
    pub fn with_storage_path<P: AsRef<Path>>(base_path: P) -> TsdbResult<Self> {
        let rdf_store = RdfStore::new()
            .map_err(|e| TsdbError::Integration(format!("Failed to create RDF store: {e}")))?;

        let base_path = base_path.as_ref();
        std::fs::create_dir_all(base_path)?;

        let columnar_store = ColumnarStore::new(base_path, Duration::hours(2), 256)?;

        let wal_path = base_path.join("hybrid_store.wal");
        let wal = WriteAheadLog::new(&wal_path, false)?;

        let mut buffer = WriteBuffer::new(WriteBufferConfig {
            max_capacity: DEFAULT_BUFFER_CAPACITY,
            flush_policy: FlushPolicy::SizeBased {
                threshold: DEFAULT_FLUSH_THRESHOLD,
            },
            // Durability is provided by `wal` directly (append-before-ack,
            // cleared only after a successful compaction), so the buffer's
            // own optional WAL hook is not used.
            enable_wal: false,
            ..Default::default()
        });

        // Recover any points that were logged before a prior crash but never
        // reached a durable columnar chunk. They stay in the WAL as-is
        // (replay reads, it does not truncate) until the next successful
        // flush + compaction clears it.
        for (series_id, point) in wal.replay()? {
            if let Err(e) = buffer.push(BufferPoint {
                series_id,
                timestamp_ms: point.timestamp.timestamp_millis(),
                value: point.value,
            }) {
                tracing::error!(
                    series_id,
                    error = %e,
                    "Failed to recover a WAL-logged point into the staging buffer during startup"
                );
            }
        }

        Ok(Self {
            rdf_store: Arc::new(RwLock::new(rdf_store)),
            columnar_store: Arc::new(columnar_store),
            ts_write_path: Arc::new(Mutex::new(TsWritePath { wal, buffer })),
            subject_to_series: Arc::new(RwLock::new(HashMap::new())),
            series_to_subject: Arc::new(RwLock::new(HashMap::new())),
            next_series_id: Arc::new(RwLock::new(1)),
        })
    }

    /// Check if a predicate indicates time-series data
    fn is_timeseries_predicate(predicate: &Predicate) -> bool {
        if let Predicate::NamedNode(node) = predicate {
            let iri = node.as_str();
            matches!(
                iri,
                "http://qudt.org/schema/qudt/numericValue"
                    | "http://www.w3.org/ns/sosa/hasSimpleResult"
                    | "http://example.org/ts/value"
            )
        } else {
            false
        }
    }

    /// Check if a predicate carries an explicit RDF-star observation
    /// timestamp for a quoted time-series triple.
    fn is_timestamp_annotation_predicate(predicate: &Predicate) -> bool {
        if let Predicate::NamedNode(node) = predicate {
            TIMESTAMP_ANNOTATION_PREDICATES.contains(&node.as_str())
        } else {
            false
        }
    }

    /// Get or create a time-series ID for an RDF subject
    fn get_or_create_series_id(&self, subject: &Subject) -> TsdbResult<u64> {
        let subject_str = match subject {
            Subject::NamedNode(node) => node.as_str().to_string(),
            Subject::BlankNode(blank) => format!("_:{}", blank.as_str()),
            Subject::Variable(var) => format!("?{}", var.as_str()),
            Subject::QuotedTriple(triple) => format!("<<{:?}>>", triple),
        };

        // Check if already mapped
        {
            let mapping = self
                .subject_to_series
                .read()
                .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;
            if let Some(&series_id) = mapping.get(&subject_str) {
                return Ok(series_id);
            }
        }

        // Create new mapping
        let mut next_id = self
            .next_series_id
            .write()
            .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;
        let series_id = *next_id;
        *next_id += 1;
        drop(next_id);

        // Store both mappings
        {
            let mut s2s = self
                .subject_to_series
                .write()
                .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;
            s2s.insert(subject_str.clone(), series_id);
        }
        {
            let mut s2s = self
                .series_to_subject
                .write()
                .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;
            s2s.insert(series_id, subject_str);
        }

        Ok(series_id)
    }

    /// Parse a literal value to f64
    fn parse_literal_to_f64(object: &Object) -> Option<f64> {
        match object {
            Object::Literal(lit) => lit.as_str().parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Parse an RFC 3339 literal timestamp (e.g. `"2024-01-01T00:00:00Z"`)
    fn parse_literal_to_timestamp(object: &Object) -> Option<DateTime<Utc>> {
        match object {
            Object::Literal(lit) => DateTime::parse_from_rfc3339(lit.as_str())
                .ok()
                .map(|dt| dt.with_timezone(&Utc)),
            _ => None,
        }
    }

    /// Record one time-series point through the durable write path: WAL
    /// append (acknowledged before returning), staging buffer, and -- once
    /// the flush policy triggers -- compaction into durable columnar chunks.
    fn record_point(&self, series_id: u64, point: DataPoint) -> TsdbResult<()> {
        let mut path = self
            .ts_write_path
            .lock()
            .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;

        // Durability first: log to the WAL before the write is acknowledged.
        path.wal.append(series_id, point)?;

        path.buffer
            .push(BufferPoint {
                series_id,
                timestamp_ms: point.timestamp.timestamp_millis(),
                value: point.value,
            })
            .map_err(|e| TsdbError::Integration(format!("Write buffer error: {e}")))?;

        if path.buffer.should_flush() {
            let flushed = path
                .buffer
                .flush()
                .map_err(|e| TsdbError::Integration(format!("Flush error: {e}")))?;
            // Compact into durable columnar chunks *before* truncating the
            // WAL: if this crashes mid-compaction, the WAL still has every
            // point and will replay them on the next startup.
            Self::persist_flushed(&self.columnar_store, &flushed)?;
            path.wal.clear()?;
        }

        Ok(())
    }

    /// Group flushed points by series and compact each series into a
    /// durable, compressed [`TimeChunk`] written through [`ColumnarStore`].
    fn persist_flushed(store: &ColumnarStore, points: &[BufferPoint]) -> TsdbResult<()> {
        if points.is_empty() {
            return Ok(());
        }

        let mut by_series: HashMap<u64, Vec<DataPoint>> = HashMap::new();
        for p in points {
            let timestamp =
                DateTime::from_timestamp_millis(p.timestamp_ms).unwrap_or_else(Utc::now);
            by_series
                .entry(p.series_id)
                .or_default()
                .push(DataPoint::new(timestamp, p.value));
        }

        for (series_id, mut series_points) in by_series {
            series_points.sort_by_key(|p| p.timestamp);
            let start = series_points
                .first()
                .map(|p| p.timestamp)
                .unwrap_or_else(Utc::now);
            let end = series_points.last().map(|p| p.timestamp).unwrap_or(start);
            // Ensure the chunk's [start, end) span covers every point in it,
            // regardless of how far apart the buffered observations were.
            let span = (end - start) + Duration::milliseconds(1);
            let chunk = TimeChunk::new(series_id, start, span, series_points)?;
            store.write_chunk(&chunk)?;
        }

        Ok(())
    }

    /// Extension method: Insert a time-series data point directly
    ///
    /// This bypasses RDF and writes directly to the time-series storage.
    ///
    /// # Arguments
    ///
    /// - `series_id` - The time-series identifier
    /// - `timestamp` - When the measurement was taken
    /// - `value` - The numerical value
    pub fn insert_ts(
        &self,
        series_id: u64,
        timestamp: DateTime<Utc>,
        value: f64,
    ) -> TsdbResult<()> {
        self.record_point(series_id, DataPoint::new(timestamp, value))
    }

    /// Extension method: Query time-series data within a time range
    ///
    /// Returns raw time-series data without RDF conversion. Merges points
    /// already compacted into durable columnar storage with points still
    /// sitting in the staging buffer.
    ///
    /// # Arguments
    ///
    /// - `series_id` - The time-series identifier
    /// - `start` - Start of time range (inclusive)
    /// - `end` - End of time range (inclusive)
    pub fn query_ts_range(
        &self,
        series_id: u64,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> TsdbResult<Vec<DataPoint>> {
        // `ColumnarStore::query_range` treats `end` as exclusive; widen by
        // 1ms and re-filter below to preserve this method's inclusive-`end`
        // contract.
        let mut points =
            self.columnar_store
                .query_range(series_id, start, end + Duration::milliseconds(1))?;
        points.retain(|p| p.timestamp <= end);

        // Points not yet compacted out of the staging buffer.
        {
            let path = self
                .ts_write_path
                .lock()
                .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;
            for p in path.buffer.iter() {
                if p.series_id != series_id {
                    continue;
                }
                let timestamp =
                    DateTime::from_timestamp_millis(p.timestamp_ms).unwrap_or_else(Utc::now);
                if timestamp >= start && timestamp <= end {
                    points.push(DataPoint::new(timestamp, p.value));
                }
            }
        }

        points.sort_by_key(|p| p.timestamp);
        Ok(points)
    }

    /// Extension method: Get the RDF subject URI for a time-series ID
    pub fn get_subject_for_series(&self, series_id: u64) -> TsdbResult<Option<String>> {
        let mapping = self
            .series_to_subject
            .read()
            .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;
        Ok(mapping.get(&series_id).cloned())
    }

    /// Extension method: Get the time-series ID for an RDF subject (if exists)
    pub fn get_series_for_subject(&self, subject: &str) -> TsdbResult<Option<u64>> {
        let mapping = self
            .subject_to_series
            .read()
            .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;
        Ok(mapping.get(subject).copied())
    }
}

impl Default for HybridStore {
    fn default() -> Self {
        Self::new().expect("Failed to create default HybridStore")
    }
}

// Implement the Store trait for HybridStore
impl Store for HybridStore {
    fn insert_quad(&self, quad: Quad) -> OxirsResult<bool> {
        // RDF-star annotation form:
        //   << subject ts-predicate value >> timestamp-predicate "iso8601" .
        // The outer quad's object supplies the true observation time for the
        // quoted (inner) time-series triple, so backfills are not corrupted
        // by a wall-clock stamp.
        if let Subject::QuotedTriple(inner) = quad.subject() {
            if Self::is_timestamp_annotation_predicate(quad.predicate())
                && Self::is_timeseries_predicate(inner.predicate())
            {
                if let (Some(value), Some(timestamp)) = (
                    Self::parse_literal_to_f64(inner.object()),
                    Self::parse_literal_to_timestamp(quad.object()),
                ) {
                    let series_id = self.get_or_create_series_id(inner.subject()).map_err(|e| {
                        OxirsError::Store(format!("Failed to map subject to series: {e}"))
                    })?;
                    self.record_point(series_id, DataPoint::new(timestamp, value))
                        .map_err(|e| {
                            OxirsError::Store(format!("Failed to record time-series point: {e}"))
                        })?;
                    return Ok(true);
                }
                // Annotation present but malformed (bad literal) -- fall
                // through so the statement isn't silently dropped.
            }
        }

        // Check if this is time-series data
        if Self::is_timeseries_predicate(quad.predicate()) {
            // Route to time-series storage
            let series_id = self
                .get_or_create_series_id(quad.subject())
                .map_err(|e| OxirsError::Store(format!("Failed to map subject to series: {e}")))?;

            if let Some(value) = Self::parse_literal_to_f64(quad.object()) {
                // No explicit timestamp annotation: this is a live-ingestion
                // write, so the current wall-clock time is correct. Callers
                // performing historical/backfill loads MUST use the
                // RDF-star annotated form above or `insert_ts` directly.
                let timestamp = Utc::now();
                self.record_point(series_id, DataPoint::new(timestamp, value))
                    .map_err(|e| {
                        OxirsError::Store(format!("Failed to record time-series point: {e}"))
                    })?;
                Ok(true)
            } else {
                // Fall back to RDF if value is not numeric
                let rdf_store = self
                    .rdf_store
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Lock poisoned: {e}")))?;
                Store::insert_quad(&*rdf_store, quad)
            }
        } else {
            // Route to RDF storage
            let rdf_store = self
                .rdf_store
                .read()
                .map_err(|e| OxirsError::Store(format!("Lock poisoned: {e}")))?;
            Store::insert_quad(&*rdf_store, quad)
        }
    }

    fn remove_quad(&self, quad: &Quad) -> OxirsResult<bool> {
        // For now, only support removal from RDF store
        // Time-series data is typically append-only
        let rdf_store = self
            .rdf_store
            .read()
            .map_err(|e| OxirsError::Store(format!("Lock poisoned: {e}")))?;
        Store::remove_quad(&*rdf_store, quad)
    }

    fn find_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> OxirsResult<Vec<Quad>> {
        // Query RDF store (time-series data is not queryable via quad patterns)
        let rdf_store = self
            .rdf_store
            .read()
            .map_err(|e| OxirsError::Store(format!("Lock poisoned: {e}")))?;
        rdf_store.query_quads(subject, predicate, object, graph_name)
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn len(&self) -> OxirsResult<usize> {
        let rdf_store = self
            .rdf_store
            .read()
            .map_err(|e| OxirsError::Store(format!("Lock poisoned: {e}")))?;
        rdf_store.len()
    }

    fn is_empty(&self) -> OxirsResult<bool> {
        let rdf_store = self
            .rdf_store
            .read()
            .map_err(|e| OxirsError::Store(format!("Lock poisoned: {e}")))?;
        rdf_store.is_empty()
    }

    fn query(&self, sparql: &str) -> OxirsResult<OxirsQueryResults> {
        // Delegate SPARQL queries to RDF store
        // Future enhancement: Temporal SPARQL extensions (ts:window, ts:resample, ts:interpolate)
        // will be added via oxirs-arq integration for automatic query routing
        let rdf_store = self
            .rdf_store
            .read()
            .map_err(|e| OxirsError::Store(format!("Lock poisoned: {e}")))?;
        rdf_store.query(sparql)
    }

    fn prepare_query(&self, sparql: &str) -> OxirsResult<PreparedQuery> {
        Ok(PreparedQuery::new(sparql.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode, Triple};

    /// Build a `HybridStore` rooted at a unique temp-dir subdirectory so
    /// parallel tests never collide on the same WAL/columnar files.
    fn temp_hybrid_store(name: &str) -> TsdbResult<(HybridStore, std::path::PathBuf)> {
        let path = std::env::temp_dir().join(format!(
            "oxirs_tsdb_store_adapter_test_{name}_{}",
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let _ = std::fs::remove_dir_all(&path);
        let store = HybridStore::with_storage_path(&path)?;
        Ok((store, path))
    }

    #[test]
    fn test_hybrid_store_creation() -> TsdbResult<()> {
        let store = HybridStore::new()?;
        assert!(store.is_ready());
        Ok(())
    }

    #[test]
    fn test_timeseries_predicate_detection() {
        let qudt = Predicate::NamedNode(
            NamedNode::new("http://qudt.org/schema/qudt/numericValue").expect("valid IRI"),
        );
        assert!(HybridStore::is_timeseries_predicate(&qudt));

        let sosa = Predicate::NamedNode(
            NamedNode::new("http://www.w3.org/ns/sosa/hasSimpleResult").expect("valid IRI"),
        );
        assert!(HybridStore::is_timeseries_predicate(&sosa));

        let rdf_type = Predicate::NamedNode(
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("valid IRI"),
        );
        assert!(!HybridStore::is_timeseries_predicate(&rdf_type));
    }

    #[test]
    fn test_series_id_mapping() -> TsdbResult<()> {
        let (store, path) = temp_hybrid_store("series_id_mapping")?;
        let subject =
            Subject::NamedNode(NamedNode::new("http://example.org/sensor1").expect("valid IRI"));

        let series_id1 = store.get_or_create_series_id(&subject)?;
        let series_id2 = store.get_or_create_series_id(&subject)?;

        assert_eq!(
            series_id1, series_id2,
            "Should return same ID for same subject"
        );

        let subject2 =
            Subject::NamedNode(NamedNode::new("http://example.org/sensor2").expect("valid IRI"));
        let series_id3 = store.get_or_create_series_id(&subject2)?;

        assert_ne!(
            series_id1, series_id3,
            "Should return different IDs for different subjects"
        );

        drop(store);
        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn test_insert_rdf_metadata() -> OxirsResult<()> {
        let (store, path) = temp_hybrid_store("insert_rdf_metadata")
            .map_err(|e| OxirsError::Store(e.to_string()))?;

        let sensor = NamedNode::new("http://example.org/sensor1")?;
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;
        let sensor_class = NamedNode::new("http://www.w3.org/ns/sosa/Sensor")?;

        let triple = Triple::new(sensor, rdf_type, sensor_class);
        let inserted = store.insert_triple(triple)?;

        assert!(inserted, "Should insert new triple");
        assert_eq!(store.len()?, 1, "Store should contain 1 quad");

        drop(store);
        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn test_insert_timeseries_data() -> OxirsResult<()> {
        let (store, path) = temp_hybrid_store("insert_timeseries_data")
            .map_err(|e| OxirsError::Store(e.to_string()))?;

        let sensor = NamedNode::new("http://example.org/sensor1")?;
        let numeric_value = NamedNode::new("http://qudt.org/schema/qudt/numericValue")?;
        let value = Literal::new("42.5");

        let triple = Triple::new(sensor, numeric_value, value);
        let inserted = store.insert_triple(triple)?;

        assert!(inserted, "Should route to time-series storage");

        // RDF store should be empty (data went to TSDB)
        assert_eq!(store.len()?, 0, "RDF store should be empty (data in TSDB)");

        drop(store);
        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn test_direct_ts_insert() -> TsdbResult<()> {
        let (store, path) = temp_hybrid_store("direct_ts_insert")?;

        let series_id = 1;
        let timestamp = Utc::now();
        let value = 42.5;

        store.insert_ts(series_id, timestamp, value)?;

        // Query back the data
        let start = timestamp - chrono::Duration::hours(1);
        let end = timestamp + chrono::Duration::hours(1);
        let points = store.query_ts_range(series_id, start, end)?;

        assert_eq!(points.len(), 1, "Should retrieve 1 data point");
        assert_eq!(points[0].value, value, "Value should match");

        drop(store);
        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn test_subject_series_lookup() -> TsdbResult<()> {
        let (store, path) = temp_hybrid_store("subject_series_lookup")?;
        let subject =
            Subject::NamedNode(NamedNode::new("http://example.org/sensor1").expect("valid IRI"));

        let series_id = store.get_or_create_series_id(&subject)?;

        // Test forward lookup
        let found_series = store.get_series_for_subject("http://example.org/sensor1")?;
        assert_eq!(found_series, Some(series_id));

        // Test reverse lookup
        let found_subject = store.get_subject_for_series(series_id)?;
        assert_eq!(
            found_subject,
            Some("http://example.org/sensor1".to_string())
        );

        drop(store);
        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    // -- Regression tests for the P1 findings fixed in this pass ------------

    /// Regression test: `insert_quad` on a plain (non-backfill) time-series
    /// triple must not silently drop the observation -- it should still be
    /// queryable, and the RDF store must stay empty (routed correctly).
    #[test]
    fn test_insert_quad_live_uses_current_time_window() -> OxirsResult<()> {
        let (store, path) =
            temp_hybrid_store("live_time_window").map_err(|e| OxirsError::Store(e.to_string()))?;

        let sensor = NamedNode::new("http://example.org/sensor-live")?;
        let numeric_value = NamedNode::new("http://qudt.org/schema/qudt/numericValue")?;
        let value = Literal::new("7.0");

        // Floor to millisecond precision: the durable write path stores
        // timestamps at millisecond granularity (see `BufferPoint`), so
        // comparing against a floored lower bound avoids a spurious failure
        // when `before` and the internal `Utc::now()` land in the same
        // millisecond but `before` has a larger sub-millisecond remainder.
        let before = DateTime::from_timestamp_millis(Utc::now().timestamp_millis())
            .expect("valid timestamp");
        store.insert_triple(Triple::new(sensor, numeric_value, value))?;
        let after = Utc::now();

        let series_id = store
            .get_series_for_subject("http://example.org/sensor-live")
            .map_err(|e| OxirsError::Store(e.to_string()))?
            .expect("series should have been created");

        let points = store
            .query_ts_range(
                series_id,
                before - chrono::Duration::seconds(1),
                after + chrono::Duration::seconds(1),
            )
            .map_err(|e| OxirsError::Store(e.to_string()))?;

        assert_eq!(points.len(), 1);
        assert_eq!(points[0].value, 7.0);
        assert!(points[0].timestamp >= before && points[0].timestamp <= after);

        drop(store);
        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    /// Regression test: the RDF-star annotated form must extract the
    /// *historical* timestamp from the annotation instead of stamping
    /// `Utc::now()`, fixing corrupted-backfill data loss.
    #[test]
    fn test_insert_quad_annotated_backfill_preserves_historical_timestamp() -> OxirsResult<()> {
        use oxirs_core::model::QuotedTriple;

        let (store, path) = temp_hybrid_store("annotated_backfill")
            .map_err(|e| OxirsError::Store(e.to_string()))?;

        let sensor = NamedNode::new("http://example.org/sensor-backfill")?;
        let numeric_value = NamedNode::new("http://qudt.org/schema/qudt/numericValue")?;
        let value = Literal::new("99.25");
        let inner_triple = Triple::new(sensor, numeric_value, value);

        let generated_at = NamedNode::new("http://www.w3.org/ns/prov#generatedAtTime")?;
        let historical_timestamp = "2019-06-01T12:00:00Z";
        let ts_literal = Literal::new(historical_timestamp);

        let quad = Quad::new(
            Subject::QuotedTriple(Box::new(QuotedTriple::new(inner_triple))),
            generated_at,
            ts_literal,
            GraphName::DefaultGraph,
        );

        let inserted = store
            .insert_quad(quad)
            .map_err(|e| OxirsError::Store(e.to_string()))?;
        assert!(inserted, "Annotated backfill quad should be accepted");

        let series_id = store
            .get_series_for_subject("http://example.org/sensor-backfill")
            .map_err(|e| OxirsError::Store(e.to_string()))?
            .expect("series should have been created");

        let expected_timestamp: DateTime<Utc> = historical_timestamp
            .parse()
            .expect("valid RFC3339 timestamp");

        let points = store
            .query_ts_range(
                series_id,
                expected_timestamp - chrono::Duration::seconds(1),
                expected_timestamp + chrono::Duration::seconds(1),
            )
            .map_err(|e| OxirsError::Store(e.to_string()))?;

        assert_eq!(points.len(), 1, "Should retrieve the backfilled point");
        assert_eq!(points[0].value, 99.25);
        assert_eq!(
            points[0].timestamp, expected_timestamp,
            "Timestamp must come from the annotation, not Utc::now()"
        );

        drop(store);
        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    /// Regression test: time-series writes must survive a flush cycle
    /// (compaction into `ColumnarStore`) and remain queryable -- proving
    /// data is not held only in an unbounded, non-durable in-memory map.
    #[test]
    fn test_ts_writes_survive_flush_into_columnar_store() -> TsdbResult<()> {
        let (store, path) = temp_hybrid_store("survives_flush")?;

        let series_id = 42;
        let base = Utc::now();

        // Push more points than the default flush threshold so at least one
        // automatic compaction into `ColumnarStore` happens.
        for i in 0..(DEFAULT_FLUSH_THRESHOLD + 10) {
            store.insert_ts(series_id, base + Duration::milliseconds(i as i64), i as f64)?;
        }

        let points = store.query_ts_range(
            series_id,
            base - Duration::seconds(1),
            base + Duration::seconds((DEFAULT_FLUSH_THRESHOLD as i64) + 10),
        )?;

        assert_eq!(points.len(), DEFAULT_FLUSH_THRESHOLD + 10);

        // At least one chunk must have actually been written to disk --
        // confirming this did not stay in an unbounded in-memory map.
        let chunk_count = store.columnar_store.index().chunk_count()?;
        assert!(chunk_count >= 1, "Expected at least one compacted chunk");

        drop(store);
        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }
}
