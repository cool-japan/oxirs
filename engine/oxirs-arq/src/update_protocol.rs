//! SPARQL 1.1 Update Protocol — standalone parser and in-memory executor (facade).
//!
//! This module provides a self-contained representation of all SPARQL 1.1
//! graph-update operations together with a text parser and an in-memory
//! store-backed executor.  It is intentionally independent of the
//! store-coupled `UpdateExecutor` in `update.rs` so that it can be used in
//! contexts where only an in-memory triple set is required (e.g. unit tests
//! and HTTP protocol adapters).
//!
//! The implementation is split across sibling modules (declared in `lib.rs`)
//! and re-exported here so the original public API is preserved:
//!
//! - [`update_protocol_types`](crate::update_protocol_types) — domain types,
//!   error types, and the [`SparqlUpdate`] enum.
//! - [`update_protocol_parser`](crate::update_protocol_parser) — tokeniser
//!   and recursive-descent parser ([`SparqlUpdateParser`]).
//! - [`update_protocol_executor`](crate::update_protocol_executor) —
//!   in-memory [`UpdateExecutor`] and pattern-matching helpers.
//!
//! ## Supported operations
//!
//! | SPARQL keyword | Variant |
//! |---|---|
//! | `INSERT DATA` | `SparqlUpdate::InsertData` |
//! | `DELETE DATA` | `SparqlUpdate::DeleteData` |
//! | `INSERT { … } WHERE { … }` | `SparqlUpdate::InsertWhere` |
//! | `DELETE { … } WHERE { … }` | `SparqlUpdate::DeleteWhere` |
//! | `DELETE { … } INSERT { … } WHERE { … }` | `SparqlUpdate::Modify` |
//! | `CREATE [SILENT] GRAPH <…>` | `SparqlUpdate::CreateGraph` |
//! | `DROP [SILENT] [GRAPH <…> \| DEFAULT \| NAMED \| ALL]` | `SparqlUpdate::DropGraph` |
//! | `CLEAR [SILENT] [GRAPH <…> \| DEFAULT \| NAMED \| ALL]` | `SparqlUpdate::ClearGraph` |
//! | `COPY [SILENT] <…> TO <…>` | `SparqlUpdate::CopyGraph` |
//! | `MOVE [SILENT] <…> TO <…>` | `SparqlUpdate::MoveGraph` |
//! | `ADD [SILENT] <…> TO <…>` | `SparqlUpdate::AddGraph` |
//! | `LOAD [SILENT] <…> [INTO GRAPH <…>]` | `SparqlUpdate::Load` |

pub use crate::update_protocol_executor::*;
pub use crate::update_protocol_parser::*;
pub use crate::update_protocol_types::*;
