//! Storage tier and query-rewriter implementations for materialized views.
//!
//! This sibling module provides:
//! - [`ViewStorage`] — two-tier (memory + disk) view-data persistence with
//!   automatic spillover when the in-memory budget is exhausted.
//! - JSON-safe on-disk serialisation for [`ViewData`].
//! - [`QueryRewriter`] and [`ViewIndex`] — the simplified rewrite engine and
//!   its supporting index, used when matching incoming queries against
//!   registered views.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use anyhow::{anyhow, Result};
use tracing::debug;

use crate::algebra::{Algebra, Solution, Term, Variable};
use crate::cost_model::CostModel;
use crate::materialized_views_types::{
    MaterializedView, QueryCharacteristic, QueryRewriter, StorageStatistics, ViewData, ViewIndex,
    ViewStorage,
};

// ---------------------------------------------------------------------------
// On-disk view-data persistence
// ---------------------------------------------------------------------------

/// A binding (variable-name → Term) pair, flattened from `HashMap<Variable, Term>` for
/// JSON-safe serialisation.  JSON object keys must be strings, so we store the
/// variable name rather than the full `Variable` struct.
type BindingOnDisk = Vec<(String, Term)>;

/// Serialisable representation of `ViewData` for disk persistence.
///
/// Two impedance-mismatches are resolved here:
/// 1. `SystemTime` is not directly serialisable by serde — we store seconds
///    since the UNIX epoch instead.
/// 2. `HashMap<Variable, Term>` cannot be used as a JSON map because JSON
///    map keys must be strings, not arbitrary structs.  We flatten each
///    binding to a list of (variable-name, term) pairs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ViewDataOnDisk {
    /// One entry per `Binding` in the original `Solution`.
    results: Vec<BindingOnDisk>,
    size_bytes: usize,
    row_count: usize,
    /// Seconds since the UNIX epoch.
    materialized_at_secs: u64,
    checksum: u64,
}

impl From<&ViewData> for ViewDataOnDisk {
    fn from(v: &ViewData) -> Self {
        let materialized_at_secs = v
            .materialized_at
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let results = v
            .results
            .iter()
            .map(|binding| {
                binding
                    .iter()
                    .map(|(var, term)| (var.name().to_string(), term.clone()))
                    .collect()
            })
            .collect();
        Self {
            results,
            size_bytes: v.size_bytes,
            row_count: v.row_count,
            materialized_at_secs,
            checksum: v.checksum,
        }
    }
}

impl From<ViewDataOnDisk> for ViewData {
    fn from(d: ViewDataOnDisk) -> Self {
        let materialized_at = std::time::UNIX_EPOCH + Duration::from_secs(d.materialized_at_secs);
        let results: Solution = d
            .results
            .into_iter()
            .map(|pairs| {
                pairs
                    .into_iter()
                    .filter_map(|(name, term)| Variable::new(name).ok().map(|var| (var, term)))
                    .collect()
            })
            .collect();
        Self {
            results,
            size_bytes: d.size_bytes,
            row_count: d.row_count,
            materialized_at,
            checksum: d.checksum,
        }
    }
}

// ---------------------------------------------------------------------------
// ViewStorage impl
// ---------------------------------------------------------------------------

impl ViewStorage {
    pub(crate) fn new(max_memory: usize) -> Self {
        Self {
            memory_storage: HashMap::new(),
            disk_storage_path: None,
            max_memory,
            memory_usage: 0,
            storage_stats: StorageStatistics::default(),
        }
    }

    /// Ensure a disk-storage directory exists, initialising one in the system
    /// temp directory if none has been configured yet.
    fn ensure_disk_path(&mut self) -> Result<&std::path::Path> {
        if self.disk_storage_path.is_none() {
            let dir = std::env::temp_dir().join(format!("oxirs-arq-views-{}", std::process::id()));
            std::fs::create_dir_all(&dir)?;
            self.disk_storage_path = Some(dir);
        }
        Ok(self
            .disk_storage_path
            .as_deref()
            .expect("just initialised above"))
    }

    /// Persist `data` for `view_id` to disk.
    fn write_to_disk(&mut self, view_id: &str, data: &ViewData) -> Result<()> {
        let dir = self.ensure_disk_path()?.to_owned();
        // Sanitise the view_id so it is safe to use as a filename component.
        let safe_name: String = view_id
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let path = dir.join(format!("{}.json", safe_name));
        let on_disk = ViewDataOnDisk::from(data);
        let json = serde_json::to_vec(&on_disk)?;
        std::fs::write(&path, &json)?;
        self.storage_stats.total_disk_usage += json.len();
        self.storage_stats.disk_view_count += 1;
        debug!("Wrote view '{}' to disk at {:?}", view_id, path);
        Ok(())
    }

    /// Load a previously persisted view from disk.
    fn read_from_disk(&self, view_id: &str) -> Result<ViewData> {
        let dir = self
            .disk_storage_path
            .as_deref()
            .ok_or_else(|| anyhow!("No disk storage path configured for view '{}'", view_id))?;
        let safe_name: String = view_id
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let path = dir.join(format!("{}.json", safe_name));
        let json = std::fs::read(&path)
            .map_err(|e| anyhow!("Failed to read view '{}' from disk: {}", view_id, e))?;
        let on_disk: ViewDataOnDisk = serde_json::from_slice(&json)
            .map_err(|e| anyhow!("Failed to deserialise view '{}': {}", view_id, e))?;
        Ok(ViewData::from(on_disk))
    }

    pub(crate) fn store_view_data(&mut self, view_id: String, data: ViewData) -> Result<()> {
        let data_size = data.size_bytes;
        if self.memory_usage + data_size <= self.max_memory {
            // Fits in memory — keep it hot.
            self.memory_storage.insert(view_id, data);
            self.memory_usage += data_size;
            self.storage_stats.memory_view_count += 1;
        } else {
            // Memory budget exceeded — spill to disk.
            self.write_to_disk(&view_id, &data)?;
        }
        Ok(())
    }

    /// Retrieve a view's data, checking memory first then disk.
    ///
    /// Returns `None` when the view is not found in either tier.
    pub fn get_view_data(&self, view_id: &str) -> Option<ViewData> {
        if let Some(data) = self.memory_storage.get(view_id) {
            return Some(data.clone());
        }
        self.read_from_disk(view_id).ok()
    }
}

// ---------------------------------------------------------------------------
// QueryRewriter impl
// ---------------------------------------------------------------------------

impl QueryRewriter {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            view_index: ViewIndex::new(),
            rewrite_rules: Vec::new(),
            cost_threshold: 0.8, // Only rewrite if 80% cost reduction
        })
    }

    pub(crate) fn rewrite_query(
        &self,
        query: &Algebra,
        views: &Arc<RwLock<HashMap<String, MaterializedView>>>,
        _cost_model: &Arc<Mutex<CostModel>>,
    ) -> Result<(Algebra, Vec<String>)> {
        // Simplified rewrite logic
        let _views_guard = views.read().expect("lock poisoned");
        let used_views = Vec::new();

        // For now, return original query
        // In full implementation, would analyze query and find matching views
        Ok((query.clone(), used_views))
    }

    pub(crate) fn update_view_index(&mut self, view_id: &str, definition: &Algebra) -> Result<()> {
        self.view_index.add_view(view_id.to_string(), definition)
    }
}

// ---------------------------------------------------------------------------
// ViewIndex impl
// ---------------------------------------------------------------------------

impl ViewIndex {
    fn new() -> Self {
        Self {
            pattern_index: HashMap::new(),
            variable_index: HashMap::new(),
            predicate_index: HashMap::new(),
            characteristic_index: HashMap::new(),
        }
    }

    fn add_view(&mut self, view_id: String, definition: &Algebra) -> Result<()> {
        // Extract patterns and characteristics for indexing
        let characteristics = self.extract_characteristics(definition);

        for characteristic in characteristics {
            self.characteristic_index
                .entry(characteristic)
                .or_default()
                .push(view_id.clone());
        }

        Ok(())
    }

    fn extract_characteristics(&self, algebra: &Algebra) -> Vec<QueryCharacteristic> {
        let mut characteristics = Vec::new();

        match algebra {
            Algebra::Join { .. } => characteristics.push(QueryCharacteristic::HasJoin),
            Algebra::Union { .. } => characteristics.push(QueryCharacteristic::HasUnion),
            Algebra::Filter { .. } => characteristics.push(QueryCharacteristic::HasFilter),
            Algebra::Group { .. } => characteristics.push(QueryCharacteristic::HasAggregation),
            Algebra::Bgp(patterns) => {
                characteristics.push(QueryCharacteristic::PatternCount(patterns.len()));
            }
            _ => {}
        }

        characteristics
    }
}
