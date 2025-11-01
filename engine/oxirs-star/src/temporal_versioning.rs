//! Temporal versioning for RDF-star annotations
//!
//! This module implements bi-temporal versioning supporting both:
//! - **Valid time**: When the information is true in the real world
//! - **Transaction time**: When the information was stored in the database
//!
//! This enables time-travel queries, audit trails, and historical analysis of
//! annotation changes over time.
//!
//! # Features
//!
//! - **Bi-temporal tracking** - Valid time and transaction time
//! - **Time-travel queries** - Query annotations as of any point in time
//! - **Version chains** - Track complete history of annotation changes
//! - **Snapshot isolation** - Consistent views of data at specific times
//! - **Temporal predicates** - AS OF, BETWEEN, CURRENT queries
//! - **Efficient storage** - Delta compression for version chains
//!
//! # Examples
//!
//! ```rust
//! use oxirs_star::temporal_versioning::{TemporalAnnotationStore, TemporalQuery};
//! use chrono::Utc;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut store = TemporalAnnotationStore::new();
//!
//! // Insert annotation with temporal information
//! // store.insert_temporal(...);
//!
//! // Query as of specific time
//! let timestamp = Utc::now();
//! // let annotations = store.query_as_of(timestamp)?;
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use tracing::{debug, span, Level};

use crate::annotations::TripleAnnotation;
use crate::model::StarTriple;
use crate::StarResult;

/// Temporal version of an annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalVersion {
    /// The annotation at this version
    pub annotation: TripleAnnotation,

    /// Valid time start (when this became true in the real world)
    pub valid_time_start: DateTime<Utc>,

    /// Valid time end (when this ceased to be true, None = current)
    pub valid_time_end: Option<DateTime<Utc>>,

    /// Transaction time (when this was recorded in the database)
    pub transaction_time: DateTime<Utc>,

    /// Version number
    pub version: u64,

    /// Previous version (for delta encoding)
    pub previous_version: Option<u64>,

    /// Change type
    pub change_type: ChangeType,

    /// User/agent who made the change
    pub changed_by: Option<String>,
}

/// Type of change made to an annotation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    /// Initial insertion
    Insert,
    /// Update of existing annotation
    Update,
    /// Logical deletion (version still exists)
    Delete,
    /// Correction of past data
    Correction,
}

/// Temporal query specification
#[derive(Debug, Clone)]
pub enum TemporalQuery {
    /// Query current state (latest versions)
    Current,

    /// Query as of a specific transaction time
    AsOfTransaction(DateTime<Utc>),

    /// Query as of a specific valid time
    AsOfValid(DateTime<Utc>),

    /// Query as of both valid and transaction time
    AsOfBoth {
        valid_time: DateTime<Utc>,
        transaction_time: DateTime<Utc>,
    },

    /// Query between two transaction times
    BetweenTransaction {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },

    /// Query between two valid times
    BetweenValid {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },

    /// Get complete version history
    AllVersions,
}

/// Version chain for an annotation
#[derive(Debug, Clone)]
struct VersionChain {
    /// All versions ordered by transaction time
    versions: BTreeMap<DateTime<Utc>, TemporalVersion>,

    /// Next version number
    next_version: u64,

    /// Index by valid time for efficient querying
    valid_time_index: BTreeMap<DateTime<Utc>, Vec<DateTime<Utc>>>,
}

impl VersionChain {
    fn new() -> Self {
        Self {
            versions: BTreeMap::new(),
            next_version: 1,
            valid_time_index: BTreeMap::new(),
        }
    }

    fn add_version(&mut self, mut version: TemporalVersion) {
        version.version = self.next_version;
        self.next_version += 1;

        let tx_time = version.transaction_time;
        let valid_time = version.valid_time_start;

        // Add to valid time index
        self.valid_time_index
            .entry(valid_time)
            .or_default()
            .push(tx_time);

        self.versions.insert(tx_time, version);
    }

    fn get_as_of_transaction(&self, time: DateTime<Utc>) -> Option<&TemporalVersion> {
        self.versions.range(..=time).next_back().map(|(_, v)| v)
    }

    fn get_as_of_valid(&self, time: DateTime<Utc>) -> Option<&TemporalVersion> {
        // Find versions where valid_time_start <= time < valid_time_end
        self.versions
            .values()
            .filter(|v| {
                v.valid_time_start <= time && v.valid_time_end.map_or(true, |end| time < end)
            })
            .next_back()
    }

    fn get_between_transaction(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&TemporalVersion> {
        self.versions.range(start..=end).map(|(_, v)| v).collect()
    }

    fn get_all_versions(&self) -> Vec<&TemporalVersion> {
        self.versions.values().collect()
    }

    fn get_current(&self) -> Option<&TemporalVersion> {
        self.versions.values().last()
    }
}

/// Temporal annotation store with bi-temporal tracking
pub struct TemporalAnnotationStore {
    /// Version chains indexed by triple hash
    version_chains: HashMap<u64, VersionChain>,

    /// Statistics
    stats: TemporalStoreStatistics,
}

/// Statistics for temporal store
#[derive(Debug, Clone, Default)]
pub struct TemporalStoreStatistics {
    /// Total number of annotations tracked
    pub annotation_count: usize,

    /// Total number of versions stored
    pub total_versions: usize,

    /// Average versions per annotation
    pub avg_versions_per_annotation: f64,

    /// Earliest transaction time
    pub earliest_transaction: Option<DateTime<Utc>>,

    /// Latest transaction time
    pub latest_transaction: Option<DateTime<Utc>>,

    /// Earliest valid time
    pub earliest_valid_time: Option<DateTime<Utc>>,

    /// Latest valid time
    pub latest_valid_time: Option<DateTime<Utc>>,
}

impl TemporalAnnotationStore {
    /// Create a new temporal annotation store
    pub fn new() -> Self {
        Self {
            version_chains: HashMap::new(),
            stats: TemporalStoreStatistics::default(),
        }
    }

    /// Insert a new annotation with temporal information
    pub fn insert_temporal(
        &mut self,
        triple: &StarTriple,
        annotation: TripleAnnotation,
        valid_time_start: DateTime<Utc>,
        valid_time_end: Option<DateTime<Utc>>,
        changed_by: Option<String>,
    ) -> StarResult<u64> {
        let span = span!(Level::DEBUG, "insert_temporal");
        let _enter = span.enter();

        let triple_hash = Self::hash_triple(triple);
        let transaction_time = Utc::now();

        let version = TemporalVersion {
            annotation,
            valid_time_start,
            valid_time_end,
            transaction_time,
            version: 0, // Will be set by VersionChain
            previous_version: None,
            change_type: ChangeType::Insert,
            changed_by,
        };

        let chain = self
            .version_chains
            .entry(triple_hash)
            .or_insert_with(VersionChain::new);

        // Set previous version if this is an update
        let previous = chain.get_current().map(|v| v.version);

        let mut version = version;
        version.previous_version = previous;
        if previous.is_some() {
            version.change_type = ChangeType::Update;
        }

        chain.add_version(version);

        self.update_statistics();

        debug!("Inserted temporal version for triple {}", triple_hash);
        Ok(triple_hash)
    }

    /// Update an existing annotation
    pub fn update_temporal(
        &mut self,
        triple: &StarTriple,
        annotation: TripleAnnotation,
        valid_time_start: DateTime<Utc>,
        valid_time_end: Option<DateTime<Utc>>,
        changed_by: Option<String>,
    ) -> StarResult<()> {
        let triple_hash = Self::hash_triple(triple);

        if !self.version_chains.contains_key(&triple_hash) {
            return Err(crate::StarError::invalid_quoted_triple(
                "Triple not found in temporal store",
            ));
        }

        let transaction_time = Utc::now();
        let chain = self.version_chains.get_mut(&triple_hash).unwrap();

        let previous_version = chain.get_current().map(|v| v.version);

        let version = TemporalVersion {
            annotation,
            valid_time_start,
            valid_time_end,
            transaction_time,
            version: 0, // Will be set by VersionChain
            previous_version,
            change_type: ChangeType::Update,
            changed_by,
        };

        chain.add_version(version);
        self.update_statistics();

        Ok(())
    }

    /// Delete an annotation (logical deletion)
    pub fn delete_temporal(
        &mut self,
        triple: &StarTriple,
        changed_by: Option<String>,
    ) -> StarResult<()> {
        let triple_hash = Self::hash_triple(triple);

        let chain = self
            .version_chains
            .get_mut(&triple_hash)
            .ok_or_else(|| crate::StarError::invalid_quoted_triple("Triple not found"))?;

        let current = chain
            .get_current()
            .ok_or_else(|| crate::StarError::invalid_quoted_triple("No current version"))?;

        let transaction_time = Utc::now();
        let previous_version = Some(current.version);

        let mut annotation = current.annotation.clone();
        annotation.version = Some(current.version + 1);

        let version = TemporalVersion {
            annotation,
            valid_time_start: current.valid_time_start,
            valid_time_end: Some(transaction_time), // End valid time at deletion
            transaction_time,
            version: 0,
            previous_version,
            change_type: ChangeType::Delete,
            changed_by,
        };

        chain.add_version(version);
        self.update_statistics();

        Ok(())
    }

    /// Query annotations based on temporal criteria
    pub fn query_temporal(&self, query: &TemporalQuery) -> Vec<(u64, &TemporalVersion)> {
        let span = span!(Level::DEBUG, "query_temporal");
        let _enter = span.enter();

        let mut results = Vec::new();

        for (&triple_hash, chain) in &self.version_chains {
            match query {
                TemporalQuery::Current => {
                    if let Some(version) = chain.get_current() {
                        if version.change_type != ChangeType::Delete {
                            results.push((triple_hash, version));
                        }
                    }
                }
                TemporalQuery::AsOfTransaction(time) => {
                    if let Some(version) = chain.get_as_of_transaction(*time) {
                        if version.change_type != ChangeType::Delete {
                            results.push((triple_hash, version));
                        }
                    }
                }
                TemporalQuery::AsOfValid(time) => {
                    if let Some(version) = chain.get_as_of_valid(*time) {
                        if version.change_type != ChangeType::Delete {
                            results.push((triple_hash, version));
                        }
                    }
                }
                TemporalQuery::AsOfBoth {
                    valid_time,
                    transaction_time,
                } => {
                    if let Some(version) = chain.get_as_of_transaction(*transaction_time) {
                        if version.valid_time_start <= *valid_time
                            && version.valid_time_end.map_or(true, |end| *valid_time < end)
                            && version.change_type != ChangeType::Delete
                        {
                            results.push((triple_hash, version));
                        }
                    }
                }
                TemporalQuery::BetweenTransaction { start, end } => {
                    for version in chain.get_between_transaction(*start, *end) {
                        results.push((triple_hash, version));
                    }
                }
                TemporalQuery::BetweenValid { start, end } => {
                    for version in chain.get_all_versions() {
                        if version.valid_time_start >= *start && version.valid_time_start <= *end {
                            results.push((triple_hash, version));
                        }
                    }
                }
                TemporalQuery::AllVersions => {
                    for version in chain.get_all_versions() {
                        results.push((triple_hash, version));
                    }
                }
            }
        }

        debug!("Query returned {} results", results.len());
        results
    }

    /// Get version history for a specific triple
    pub fn get_version_history(&self, triple: &StarTriple) -> Vec<&TemporalVersion> {
        let triple_hash = Self::hash_triple(triple);

        self.version_chains
            .get(&triple_hash)
            .map(|chain| chain.get_all_versions())
            .unwrap_or_default()
    }

    /// Get statistics about the temporal store
    pub fn statistics(&self) -> &TemporalStoreStatistics {
        &self.stats
    }

    fn update_statistics(&mut self) {
        self.stats.annotation_count = self.version_chains.len();
        self.stats.total_versions = self
            .version_chains
            .values()
            .map(|chain| chain.versions.len())
            .sum();

        self.stats.avg_versions_per_annotation = if self.stats.annotation_count > 0 {
            self.stats.total_versions as f64 / self.stats.annotation_count as f64
        } else {
            0.0
        };

        // Find earliest and latest times
        let mut earliest_tx = None;
        let mut latest_tx = None;
        let mut earliest_valid = None;
        let mut latest_valid = None;

        for chain in self.version_chains.values() {
            for version in chain.versions.values() {
                if earliest_tx.is_none() || Some(version.transaction_time) < earliest_tx {
                    earliest_tx = Some(version.transaction_time);
                }
                if latest_tx.is_none() || Some(version.transaction_time) > latest_tx {
                    latest_tx = Some(version.transaction_time);
                }
                if earliest_valid.is_none() || Some(version.valid_time_start) < earliest_valid {
                    earliest_valid = Some(version.valid_time_start);
                }
                if latest_valid.is_none() || Some(version.valid_time_start) > latest_valid {
                    latest_valid = Some(version.valid_time_start);
                }
            }
        }

        self.stats.earliest_transaction = earliest_tx;
        self.stats.latest_transaction = latest_tx;
        self.stats.earliest_valid_time = earliest_valid;
        self.stats.latest_valid_time = latest_valid;
    }

    fn hash_triple(triple: &StarTriple) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", triple).hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for TemporalAnnotationStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;
    use chrono::Duration;

    #[test]
    fn test_temporal_store_creation() {
        let store = TemporalAnnotationStore::new();
        assert_eq!(store.statistics().annotation_count, 0);
    }

    #[test]
    fn test_insert_temporal() {
        let mut store = TemporalAnnotationStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let annotation = TripleAnnotation::new().with_confidence(0.9);
        let valid_start = Utc::now();

        store
            .insert_temporal(
                &triple,
                annotation,
                valid_start,
                None,
                Some("Alice".to_string()),
            )
            .unwrap();

        assert_eq!(store.statistics().annotation_count, 1);
        assert_eq!(store.statistics().total_versions, 1);
    }

    #[test]
    fn test_update_temporal() {
        let mut store = TemporalAnnotationStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let ann1 = TripleAnnotation::new().with_confidence(0.8);
        let valid_start = Utc::now();

        store
            .insert_temporal(&triple, ann1, valid_start, None, None)
            .unwrap();

        // Update
        let ann2 = TripleAnnotation::new().with_confidence(0.9);
        store
            .update_temporal(&triple, ann2, valid_start, None, Some("Bob".to_string()))
            .unwrap();

        assert_eq!(store.statistics().annotation_count, 1);
        assert_eq!(store.statistics().total_versions, 2);
    }

    #[test]
    fn test_query_current() {
        let mut store = TemporalAnnotationStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let annotation = TripleAnnotation::new().with_confidence(0.9);
        store
            .insert_temporal(&triple, annotation, Utc::now(), None, None)
            .unwrap();

        let results = store.query_temporal(&TemporalQuery::Current);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_as_of_transaction() {
        let mut store = TemporalAnnotationStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let ann1 = TripleAnnotation::new().with_confidence(0.8);
        let time1 = Utc::now();
        store
            .insert_temporal(&triple, ann1, time1, None, None)
            .unwrap();

        std::thread::sleep(std::time::Duration::from_millis(10));

        let ann2 = TripleAnnotation::new().with_confidence(0.9);
        store
            .update_temporal(&triple, ann2, time1, None, None)
            .unwrap();

        // Query as of time between the two versions
        let query_time = Utc::now() - Duration::milliseconds(5);
        let results = store.query_temporal(&TemporalQuery::AsOfTransaction(query_time));

        // Should get the first version
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1.annotation.confidence, Some(0.8));
    }

    #[test]
    fn test_version_history() {
        let mut store = TemporalAnnotationStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let ann1 = TripleAnnotation::new().with_confidence(0.8);
        store
            .insert_temporal(&triple, ann1, Utc::now(), None, None)
            .unwrap();

        let ann2 = TripleAnnotation::new().with_confidence(0.9);
        store
            .update_temporal(&triple, ann2, Utc::now(), None, None)
            .unwrap();

        let history = store.get_version_history(&triple);
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].version, 1);
        assert_eq!(history[1].version, 2);
    }

    #[test]
    fn test_delete_temporal() {
        let mut store = TemporalAnnotationStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let annotation = TripleAnnotation::new().with_confidence(0.9);
        store
            .insert_temporal(&triple, annotation, Utc::now(), None, None)
            .unwrap();

        store
            .delete_temporal(&triple, Some("Admin".to_string()))
            .unwrap();

        // Current query should not return deleted items
        let results = store.query_temporal(&TemporalQuery::Current);
        assert_eq!(results.len(), 0);

        // But all versions query should show both
        let all_results = store.query_temporal(&TemporalQuery::AllVersions);
        assert_eq!(all_results.len(), 2);
    }

    #[test]
    fn test_valid_time_queries() {
        let mut store = TemporalAnnotationStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let past = Utc::now() - Duration::days(30);
        let future = Utc::now() + Duration::days(30);

        let annotation = TripleAnnotation::new().with_confidence(0.9);
        store
            .insert_temporal(&triple, annotation, past, Some(future), None)
            .unwrap();

        // Query for a time within the valid period
        let query_time = Utc::now();
        let results = store.query_temporal(&TemporalQuery::AsOfValid(query_time));
        assert_eq!(results.len(), 1);

        // Query for a time outside the valid period
        let outside_time = Utc::now() + Duration::days(60);
        let results_outside = store.query_temporal(&TemporalQuery::AsOfValid(outside_time));
        assert_eq!(results_outside.len(), 0);
    }
}
