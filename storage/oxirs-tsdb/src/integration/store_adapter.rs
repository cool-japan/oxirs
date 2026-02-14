//! Hybrid RDF + Time-Series Store Adapter
//!
//! Implements the oxirs-core Store trait with automatic routing between
//! RDF storage (for semantic metadata) and time-series storage (for high-frequency data).

use crate::error::{TsdbError, TsdbResult};
use crate::series::DataPoint;
use chrono::{DateTime, Utc};
use oxirs_core::model::{GraphName, Object, Predicate, Quad, Subject};
use oxirs_core::rdf_store::{OxirsQueryResults, PreparedQuery, RdfStore, Store};
use oxirs_core::{OxirsError, RdfTerm, Result as OxirsResult};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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
#[derive(Debug)]
pub struct HybridStore {
    /// RDF store for semantic metadata
    rdf_store: Arc<RwLock<RdfStore>>,

    /// Time-series data storage (series_id -> `Vec<DataPoint>`)
    tsdb_data: Arc<RwLock<HashMap<u64, Vec<DataPoint>>>>,

    /// Mapping from RDF subjects (sensor URIs) to time-series IDs
    subject_to_series: Arc<RwLock<HashMap<String, u64>>>,

    /// Reverse mapping from series IDs to RDF subjects
    series_to_subject: Arc<RwLock<HashMap<u64, String>>>,

    /// Next series ID to assign
    next_series_id: Arc<RwLock<u64>>,
}

impl HybridStore {
    /// Create a new hybrid store with in-memory backends
    pub fn new() -> TsdbResult<Self> {
        let rdf_store = RdfStore::new()
            .map_err(|e| TsdbError::Integration(format!("Failed to create RDF store: {e}")))?;

        Ok(Self {
            rdf_store: Arc::new(RwLock::new(rdf_store)),
            tsdb_data: Arc::new(RwLock::new(HashMap::new())),
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
        let point = DataPoint::new(timestamp, value);
        let mut data = self
            .tsdb_data
            .write()
            .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;
        data.entry(series_id).or_insert_with(Vec::new).push(point);
        Ok(())
    }

    /// Extension method: Query time-series data within a time range
    ///
    /// Returns raw time-series data without RDF conversion.
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
        let data = self
            .tsdb_data
            .read()
            .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;

        let points = data.get(&series_id).map_or(Vec::new(), |series_data| {
            series_data
                .iter()
                .filter(|p| p.timestamp >= start && p.timestamp <= end)
                .copied()
                .collect()
        });

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
        // Check if this is time-series data
        if Self::is_timeseries_predicate(quad.predicate()) {
            // Route to time-series storage
            let series_id = self
                .get_or_create_series_id(quad.subject())
                .map_err(|e| OxirsError::Store(format!("Failed to map subject to series: {e}")))?;

            if let Some(value) = Self::parse_literal_to_f64(quad.object()) {
                let timestamp = Utc::now(); // In production, extract from quad metadata
                let point = DataPoint::new(timestamp, value);

                let mut data = self
                    .tsdb_data
                    .write()
                    .map_err(|e| OxirsError::Store(format!("Lock poisoned: {e}")))?;
                data.entry(series_id).or_insert_with(Vec::new).push(point);

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
        let store = HybridStore::new()?;
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

        Ok(())
    }

    #[test]
    fn test_insert_rdf_metadata() -> OxirsResult<()> {
        let store = HybridStore::new().map_err(|e| OxirsError::Store(e.to_string()))?;

        let sensor = NamedNode::new("http://example.org/sensor1")?;
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;
        let sensor_class = NamedNode::new("http://www.w3.org/ns/sosa/Sensor")?;

        let triple = Triple::new(sensor, rdf_type, sensor_class);
        let inserted = store.insert_triple(triple)?;

        assert!(inserted, "Should insert new triple");
        assert_eq!(store.len()?, 1, "Store should contain 1 quad");

        Ok(())
    }

    #[test]
    fn test_insert_timeseries_data() -> OxirsResult<()> {
        let store = HybridStore::new().map_err(|e| OxirsError::Store(e.to_string()))?;

        let sensor = NamedNode::new("http://example.org/sensor1")?;
        let numeric_value = NamedNode::new("http://qudt.org/schema/qudt/numericValue")?;
        let value = Literal::new("42.5");

        let triple = Triple::new(sensor, numeric_value, value);
        let inserted = store.insert_triple(triple)?;

        assert!(inserted, "Should route to time-series storage");

        // RDF store should be empty (data went to TSDB)
        assert_eq!(store.len()?, 0, "RDF store should be empty (data in TSDB)");

        Ok(())
    }

    #[test]
    fn test_direct_ts_insert() -> TsdbResult<()> {
        let store = HybridStore::new()?;

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

        Ok(())
    }

    #[test]
    fn test_subject_series_lookup() -> TsdbResult<()> {
        let store = HybridStore::new()?;
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

        Ok(())
    }
}
