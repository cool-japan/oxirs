//! Store extension trait providing convenience methods for the new modules

use crate::error::FusekiResult;
use crate::store::Store;
use std::sync::Arc;

/// Extension trait for Store to provide convenience methods
pub trait StoreExt {
    /// Check if a dataset exists
    fn dataset_exists(&self, name: &str) -> bool;

    /// Get the number of triples in a dataset
    fn count_triples(&self, dataset_name: &str) -> usize;
}

impl StoreExt for Arc<Store> {
    fn dataset_exists(&self, name: &str) -> bool {
        self.list_datasets()
            .map(|datasets| datasets.contains(&name.to_string()))
            .unwrap_or(false)
    }

    fn count_triples(&self, dataset_name: &str) -> usize {
        self.get_stats(Some(dataset_name))
            .ok()
            .map(|stats| stats.triple_count)
            .unwrap_or(0)
    }
}

impl StoreExt for Store {
    fn dataset_exists(&self, name: &str) -> bool {
        self.list_datasets()
            .map(|datasets| datasets.contains(&name.to_string()))
            .unwrap_or(false)
    }

    fn count_triples(&self, dataset_name: &str) -> usize {
        self.get_stats(Some(dataset_name))
            .ok()
            .map(|stats| stats.triple_count)
            .unwrap_or(0)
    }
}
