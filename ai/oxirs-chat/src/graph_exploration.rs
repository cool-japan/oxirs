//! Graph exploration module — thin facade re-exporting from sub-modules.
//!
//! Split into:
//! - `graph_exploration_types` — data structures (config, path, entity, schema types)
//! - `graph_exploration_traversal` — BFS/DFS traversal, path finding, `GraphExplorer` struct
//! - `graph_exploration_analytics` — centrality, ranking, query guidance, SHACL analytics
//! - `graph_exploration_tests` — unit tests

pub use crate::graph_exploration_traversal::GraphExplorer;
pub use crate::graph_exploration_types::*;
