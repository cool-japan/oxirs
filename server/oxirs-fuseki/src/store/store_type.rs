//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

/// Multi-dataset store manager
#[derive(Clone)]
pub struct Store {
    /// Default dataset store
    pub(crate) default_store: Arc<RwLock<dyn CoreStore>>,
    /// Named datasets
    pub(super) datasets: DatasetMap,
    /// Query engine for SPARQL execution
    pub(super) query_engine: Arc<QueryEngine>,
    /// Store metadata
    pub(super) metadata: Arc<RwLock<StoreMetadata>>,
}
