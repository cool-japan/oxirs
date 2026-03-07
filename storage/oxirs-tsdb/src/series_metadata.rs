//! Time-series metadata management for TSDB.
//!
//! Stores and queries metadata about time series including tags, units, retention, and value types.

use std::collections::HashMap;

/// A key-value tag attached to a time series.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeriesTag {
    pub key: String,
    pub value: String,
}

impl SeriesTag {
    /// Create a new tag.
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// Value type for time series data points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    Float64,
    Int64,
    Boolean,
    String,
}

/// Metadata describing a time series.
#[derive(Debug, Clone)]
pub struct TimeSeriesMetadata {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub unit: Option<String>,
    pub tags: Vec<SeriesTag>,
    pub created_at: u64,
    pub retention_days: Option<u32>,
    pub sample_rate_hz: Option<f64>,
    pub value_type: ValueType,
}

impl TimeSeriesMetadata {
    /// Create a new metadata entry with required fields.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        value_type: ValueType,
        created_at: u64,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            unit: None,
            tags: Vec::new(),
            created_at,
            retention_days: None,
            sample_rate_hz: None,
            value_type,
        }
    }

    /// Add a tag to this metadata.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.push(SeriesTag::new(key, value));
        self
    }

    /// Set a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set a unit.
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Set retention days.
    pub fn with_retention(mut self, days: u32) -> Self {
        self.retention_days = Some(days);
        self
    }

    /// Set sample rate in Hz.
    pub fn with_sample_rate(mut self, hz: f64) -> Self {
        self.sample_rate_hz = Some(hz);
        self
    }
}

/// Filter for searching metadata.
#[derive(Debug, Clone, Default)]
pub struct MetadataFilter {
    pub name_prefix: Option<String>,
    pub tags: Vec<SeriesTag>,
    pub value_type: Option<ValueType>,
}

impl MetadataFilter {
    /// Create an empty filter (matches all).
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by name prefix.
    pub fn with_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.name_prefix = Some(prefix.into());
        self
    }

    /// Filter by a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.push(SeriesTag::new(key, value));
        self
    }

    /// Filter by value type.
    pub fn with_value_type(mut self, vt: ValueType) -> Self {
        self.value_type = Some(vt);
        self
    }

    /// Check if a metadata entry matches this filter.
    pub fn matches(&self, meta: &TimeSeriesMetadata) -> bool {
        // Check name prefix
        if let Some(prefix) = &self.name_prefix {
            if !meta.name.starts_with(prefix.as_str()) {
                return false;
            }
        }

        // Check value type
        if let Some(vt) = &self.value_type {
            if &meta.value_type != vt {
                return false;
            }
        }

        // Check all required tags
        for required_tag in &self.tags {
            let found = meta.tags.iter().any(|t| {
                t.key == required_tag.key && t.value == required_tag.value
            });
            if !found {
                return false;
            }
        }

        true
    }
}

/// An update to apply to existing metadata.
#[derive(Debug, Clone, Default)]
pub struct MetadataUpdate {
    /// If `Some(Some(desc))`, set description; `Some(None)` clears it.
    pub description: Option<Option<String>>,
    /// If `Some(Some(unit))`, set unit; `Some(None)` clears it.
    pub unit: Option<Option<String>>,
    /// If `Some(Some(days))`, set retention; `Some(None)` clears it.
    pub retention_days: Option<Option<u32>>,
}

/// Errors for metadata operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataError {
    DuplicateId(String),
    NotFound(String),
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetadataError::DuplicateId(id) => write!(f, "Duplicate series ID: {id}"),
            MetadataError::NotFound(id) => write!(f, "Series not found: {id}"),
        }
    }
}

impl std::error::Error for MetadataError {}

/// Store for time-series metadata.
pub struct TimeSeriesMetadataStore {
    store: HashMap<String, TimeSeriesMetadata>,
}

impl Default for TimeSeriesMetadataStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSeriesMetadataStore {
    /// Create a new empty metadata store.
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    /// Register a new time series metadata entry.
    pub fn register(&mut self, metadata: TimeSeriesMetadata) -> Result<(), MetadataError> {
        if self.store.contains_key(&metadata.id) {
            return Err(MetadataError::DuplicateId(metadata.id.clone()));
        }
        self.store.insert(metadata.id.clone(), metadata);
        Ok(())
    }

    /// Get a metadata entry by series ID.
    pub fn get(&self, id: &str) -> Option<&TimeSeriesMetadata> {
        self.store.get(id)
    }

    /// Update an existing metadata entry.
    pub fn update(&mut self, id: &str, update: MetadataUpdate) -> Result<(), MetadataError> {
        let meta = self
            .store
            .get_mut(id)
            .ok_or_else(|| MetadataError::NotFound(id.to_string()))?;

        if let Some(desc) = update.description {
            meta.description = desc;
        }
        if let Some(unit) = update.unit {
            meta.unit = unit;
        }
        if let Some(ret) = update.retention_days {
            meta.retention_days = ret;
        }

        Ok(())
    }

    /// Delete a metadata entry by ID.
    ///
    /// Returns `true` if the entry existed and was removed.
    pub fn delete(&mut self, id: &str) -> bool {
        self.store.remove(id).is_some()
    }

    /// Search for series matching the given filter.
    pub fn search(&self, filter: &MetadataFilter) -> Vec<&TimeSeriesMetadata> {
        self.store
            .values()
            .filter(|m| filter.matches(m))
            .collect()
    }

    /// Get total series count.
    pub fn count(&self) -> usize {
        self.store.len()
    }

    /// Get all (key, value) tag pairs across all series (may include duplicates from multiple series).
    pub fn all_tags(&self) -> Vec<(&str, &str)> {
        let mut tags = Vec::new();
        for meta in self.store.values() {
            for tag in &meta.tags {
                tags.push((tag.key.as_str(), tag.value.as_str()));
            }
        }
        tags
    }

    /// Find all series with a specific tag key and value.
    pub fn series_by_tag(&self, key: &str, value: &str) -> Vec<&TimeSeriesMetadata> {
        self.store
            .values()
            .filter(|m| m.tags.iter().any(|t| t.key == key && t.value == value))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> TimeSeriesMetadataStore {
        TimeSeriesMetadataStore::new()
    }

    fn make_meta(id: &str, name: &str) -> TimeSeriesMetadata {
        TimeSeriesMetadata::new(id, name, ValueType::Float64, 1000)
    }

    #[test]
    fn test_register_and_get() {
        let mut store = make_store();
        let meta = make_meta("ts-1", "temperature");
        store.register(meta).unwrap();
        let found = store.get("ts-1");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "temperature");
    }

    #[test]
    fn test_register_duplicate_id_error() {
        let mut store = make_store();
        store.register(make_meta("ts-1", "temp")).unwrap();
        let result = store.register(make_meta("ts-1", "other"));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MetadataError::DuplicateId(_)));
    }

    #[test]
    fn test_get_nonexistent_returns_none() {
        let store = make_store();
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn test_update_description() {
        let mut store = make_store();
        store.register(make_meta("ts-1", "temp")).unwrap();
        let upd = MetadataUpdate {
            description: Some(Some("Room temperature".to_string())),
            ..Default::default()
        };
        store.update("ts-1", upd).unwrap();
        let m = store.get("ts-1").unwrap();
        assert_eq!(m.description, Some("Room temperature".to_string()));
    }

    #[test]
    fn test_update_unit() {
        let mut store = make_store();
        store.register(make_meta("ts-1", "temp")).unwrap();
        let upd = MetadataUpdate {
            unit: Some(Some("Celsius".to_string())),
            ..Default::default()
        };
        store.update("ts-1", upd).unwrap();
        let m = store.get("ts-1").unwrap();
        assert_eq!(m.unit, Some("Celsius".to_string()));
    }

    #[test]
    fn test_update_retention_days() {
        let mut store = make_store();
        store.register(make_meta("ts-1", "temp")).unwrap();
        let upd = MetadataUpdate {
            retention_days: Some(Some(30)),
            ..Default::default()
        };
        store.update("ts-1", upd).unwrap();
        let m = store.get("ts-1").unwrap();
        assert_eq!(m.retention_days, Some(30));
    }

    #[test]
    fn test_update_clear_description() {
        let mut store = make_store();
        let meta = make_meta("ts-1", "temp").with_description("old desc");
        store.register(meta).unwrap();
        let upd = MetadataUpdate {
            description: Some(None),
            ..Default::default()
        };
        store.update("ts-1", upd).unwrap();
        assert!(store.get("ts-1").unwrap().description.is_none());
    }

    #[test]
    fn test_update_not_found_error() {
        let mut store = make_store();
        let upd = MetadataUpdate {
            description: Some(Some("desc".to_string())),
            ..Default::default()
        };
        let result = store.update("nonexistent", upd);
        assert!(matches!(result.unwrap_err(), MetadataError::NotFound(_)));
    }

    #[test]
    fn test_delete_returns_true_when_found() {
        let mut store = make_store();
        store.register(make_meta("ts-1", "temp")).unwrap();
        assert!(store.delete("ts-1"));
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_delete_returns_false_when_not_found() {
        let mut store = make_store();
        assert!(!store.delete("nonexistent"));
    }

    #[test]
    fn test_search_by_name_prefix() {
        let mut store = make_store();
        store.register(make_meta("ts-1", "temp_room")).unwrap();
        store.register(make_meta("ts-2", "temp_outdoor")).unwrap();
        store.register(make_meta("ts-3", "humidity")).unwrap();

        let filter = MetadataFilter::new().with_name_prefix("temp");
        let results = store.search(&filter);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_by_tag() {
        let mut store = make_store();
        let m1 = make_meta("ts-1", "temp").with_tag("location", "room1");
        let m2 = make_meta("ts-2", "humidity").with_tag("location", "room1");
        let m3 = make_meta("ts-3", "pressure").with_tag("location", "room2");
        store.register(m1).unwrap();
        store.register(m2).unwrap();
        store.register(m3).unwrap();

        let filter = MetadataFilter::new().with_tag("location", "room1");
        let results = store.search(&filter);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_by_value_type() {
        let mut store = make_store();
        store
            .register(TimeSeriesMetadata::new("ts-1", "temp", ValueType::Float64, 0))
            .unwrap();
        store
            .register(TimeSeriesMetadata::new("ts-2", "counter", ValueType::Int64, 0))
            .unwrap();
        store
            .register(TimeSeriesMetadata::new("ts-3", "active", ValueType::Boolean, 0))
            .unwrap();

        let filter = MetadataFilter::new().with_value_type(ValueType::Float64);
        let results = store.search(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "ts-1");
    }

    #[test]
    fn test_search_empty_filter_returns_all() {
        let mut store = make_store();
        store.register(make_meta("ts-1", "a")).unwrap();
        store.register(make_meta("ts-2", "b")).unwrap();
        store.register(make_meta("ts-3", "c")).unwrap();

        let filter = MetadataFilter::new();
        let results = store.search(&filter);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_count() {
        let mut store = make_store();
        assert_eq!(store.count(), 0);
        store.register(make_meta("ts-1", "a")).unwrap();
        assert_eq!(store.count(), 1);
        store.register(make_meta("ts-2", "b")).unwrap();
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn test_all_tags() {
        let mut store = make_store();
        let m1 = make_meta("ts-1", "a").with_tag("k1", "v1").with_tag("k2", "v2");
        let m2 = make_meta("ts-2", "b").with_tag("k1", "v1");
        store.register(m1).unwrap();
        store.register(m2).unwrap();

        let tags = store.all_tags();
        // ts-1 has 2 tags, ts-2 has 1 tag = 3 total
        assert_eq!(tags.len(), 3);
    }

    #[test]
    fn test_all_tags_empty_store() {
        let store = make_store();
        assert!(store.all_tags().is_empty());
    }

    #[test]
    fn test_series_by_tag() {
        let mut store = make_store();
        let m1 = make_meta("ts-1", "a").with_tag("env", "prod");
        let m2 = make_meta("ts-2", "b").with_tag("env", "prod");
        let m3 = make_meta("ts-3", "c").with_tag("env", "dev");
        store.register(m1).unwrap();
        store.register(m2).unwrap();
        store.register(m3).unwrap();

        let results = store.series_by_tag("env", "prod");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_series_by_tag_no_match() {
        let mut store = make_store();
        store.register(make_meta("ts-1", "a")).unwrap();
        let results = store.series_by_tag("env", "prod");
        assert!(results.is_empty());
    }

    #[test]
    fn test_metadata_tags_stored() {
        let mut store = make_store();
        let m = make_meta("ts-1", "temp").with_tag("region", "eu-west");
        store.register(m).unwrap();
        let found = store.get("ts-1").unwrap();
        assert_eq!(found.tags.len(), 1);
        assert_eq!(found.tags[0].key, "region");
        assert_eq!(found.tags[0].value, "eu-west");
    }

    #[test]
    fn test_metadata_with_sample_rate() {
        let mut store = make_store();
        let m = make_meta("ts-1", "vibration").with_sample_rate(1000.0);
        store.register(m).unwrap();
        let found = store.get("ts-1").unwrap();
        assert_eq!(found.sample_rate_hz, Some(1000.0));
    }

    #[test]
    fn test_search_combined_tag_and_prefix() {
        let mut store = make_store();
        let m1 = make_meta("ts-1", "temp_room")
            .with_tag("env", "prod");
        let m2 = make_meta("ts-2", "temp_outdoor")
            .with_tag("env", "dev");
        let m3 = make_meta("ts-3", "humidity")
            .with_tag("env", "prod");
        store.register(m1).unwrap();
        store.register(m2).unwrap();
        store.register(m3).unwrap();

        let filter = MetadataFilter::new()
            .with_name_prefix("temp")
            .with_tag("env", "prod");
        let results = store.search(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "ts-1");
    }

    #[test]
    fn test_register_with_all_fields() {
        let mut store = make_store();
        let m = TimeSeriesMetadata::new("ts-1", "cpu_load", ValueType::Float64, 0)
            .with_description("CPU load percentage")
            .with_unit("%")
            .with_retention(90)
            .with_sample_rate(0.1)
            .with_tag("host", "server-01");
        store.register(m).unwrap();
        let found = store.get("ts-1").unwrap();
        assert_eq!(found.description, Some("CPU load percentage".to_string()));
        assert_eq!(found.unit, Some("%".to_string()));
        assert_eq!(found.retention_days, Some(90));
        assert!((found.sample_rate_hz.unwrap() - 0.1).abs() < 1e-10);
    }
}
