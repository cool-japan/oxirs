//! Production-ready RDF store implementation integrating with oxirs-core
//!
//! This module has been refactored into smaller, focused modules using SplitRS
//! to comply with the 2000-line file size guideline.

// Common imports used across store modules - re-exported so child modules can access via `use super::*`
pub use crate::config::DatasetConfig;
pub use crate::error::{FusekiError, FusekiResult};
pub use oxirs_core::model::*;
pub use oxirs_core::parser::{Parser, RdfFormat as CoreRdfFormat};
pub use oxirs_core::query::{QueryEngine, QueryResult as CoreQueryResult};
pub use oxirs_core::serializer::Serializer;
pub use oxirs_core::{RdfStore, Store as CoreStore};
pub use serde::Serialize;
pub use std::collections::HashMap;
pub use std::path::Path;
pub use std::sync::{Arc, RwLock};
pub use std::time::{Duration, Instant};
pub use tracing::{debug, info, warn};

/// Type alias for dataset storage mapping
pub type DatasetMap = Arc<RwLock<HashMap<String, Arc<RwLock<dyn CoreStore>>>>>;

pub mod functions;
pub mod store_accessors;
pub mod store_accessors_1;
pub mod store_accessors_2;
pub mod store_accessors_3;
pub mod store_accessors_4;
pub mod store_cleanup_old_changes_group;
pub mod store_clear_default_graph_group;
pub mod store_count_triples_group;
pub mod store_detect_rdf_format_group;
pub mod store_execute_sparql_update_group;
pub mod store_extract_data_block_group;
pub mod store_extract_graph_iri_for_management_group;
pub mod store_extract_graph_iri_group;
pub mod store_fetch_rdf_from_url_group;
pub mod store_is_ready_group;
pub mod store_list_datasets_group;
pub mod store_load_data_group;
pub mod store_new_group;
pub mod store_parse_load_statement_group;
pub mod store_query_group;
pub mod store_remove_dataset_group;
pub mod store_traits;
pub mod store_type;
pub mod store_update_group;
pub mod store_watch_changes_group;
pub mod types;

// Re-export all types from child modules
pub use functions::*;
pub use store_accessors::*;
pub use store_accessors_1::*;
pub use store_accessors_2::*;
pub use store_accessors_3::*;
pub use store_accessors_4::*;
pub use store_cleanup_old_changes_group::*;
pub use store_clear_default_graph_group::*;
pub use store_count_triples_group::*;
pub use store_detect_rdf_format_group::*;
pub use store_execute_sparql_update_group::*;
pub use store_extract_data_block_group::*;
pub use store_extract_graph_iri_for_management_group::*;
pub use store_extract_graph_iri_group::*;
pub use store_fetch_rdf_from_url_group::*;
pub use store_is_ready_group::*;
pub use store_list_datasets_group::*;
pub use store_load_data_group::*;
pub use store_new_group::*;
pub use store_parse_load_statement_group::*;
pub use store_query_group::*;
pub use store_remove_dataset_group::*;
pub use store_traits::*;
pub use store_type::*;
pub use store_update_group::*;
pub use store_watch_changes_group::*;
pub use types::*;
