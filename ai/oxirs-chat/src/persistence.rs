//! Session Persistence and Recovery System for OxiRS Chat.
//!
//! Provides robust session persistence with backup/recovery capabilities,
//! automatic expiration handling, and concurrent session management.
//!
//! This module is a thin facade re-exporting the public API of the three
//! sibling modules that make up the implementation:
//!
//! - [`crate::persistence_types`] — data model, configuration, and stats.
//! - [`crate::persistence_storage`] — main `SessionPersistenceManager`
//!   with save, load, delete, list, checkpoint, and background tasks.
//! - [`crate::persistence_recovery`] — corruption recovery, checkpoint
//!   loading, and partial-recovery code paths.

// Note: `persistence_recovery` adds inherent methods to
// `SessionPersistenceManager` (declared in `persistence_storage`) but
// exports no items of its own. The impl block is reachable through the
// type re-export below; no `pub use` from `persistence_recovery` needed.
pub use crate::persistence_storage::*;
pub use crate::persistence_types::*;
