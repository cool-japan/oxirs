//! SPARQL 1.1 UPDATE Graph Management Operations — facade module.
//!
//! This is a thin re-export facade. The implementation is split across:
//! - `update_graph_management_types` — structs, enums, data model types
//! - `update_graph_management_ops` — SPARQL Update graph management operations
//! - `update_graph_management_protocol` — HTTP/SPARQL protocol handling
pub use crate::update_graph_management_ops::GraphManagementExecutor;
pub use crate::update_graph_management_protocol::{
    GraphManagementHttpResponse, GraphManagementParser, GraphManagementProtocolError,
    GraphManagementRequestHandler,
};
pub use crate::update_graph_management_types::{
    GraphManagementDataset, GraphManagementOp, GraphManagementResult, GraphManagementTarget, Triple,
};
