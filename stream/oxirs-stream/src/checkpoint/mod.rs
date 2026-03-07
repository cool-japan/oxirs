//! # Checkpoint Coordination
//!
//! Chandy-Lamport inspired distributed checkpoint coordinator for consistent
//! global snapshots and failure recovery.
//!
//! ## Modules
//!
//! - [`coordinator`]: `CheckpointCoordinator`, `GlobalCheckpoint`,
//!   `OperatorSnapshot`, and `CheckpointPhase`.

pub mod coordinator;

pub use coordinator::{CheckpointCoordinator, CheckpointPhase, GlobalCheckpoint, OperatorSnapshot};
