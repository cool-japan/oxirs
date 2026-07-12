//! Advanced backup policy DSL for oxirs-cluster.
//!
//! This module provides a complete backup management system:
//!
//! - [`backup::policy::BackupPolicy`] — schedule + retention + GFS + encryption + destination
//! - [`backup::retention::RetentionTier`] — hot/warm/cold retention windows
//! - [`backup::gfs::GfsRotation`] — Grandfather-Father-Son rotation
//! - [`backup::executor::BackupExecutor`] — write artefacts + audit log
//! - [`backup::destination::DestinationConfig`] — local FS (default) or S3 (feature-gated)
//!
//! # Quick start
//!
//! ```rust
//! use std::env;
//! use oxirs_cluster::backup::{
//!     BackupPolicy, RetentionTier, GfsRotation, BackupExecutor,
//!     destination::DestinationConfig,
//!     policy::{CronSchedule, EncryptionConfig},
//! };
//!
//! let dir = env::temp_dir().join("my_backups");
//! let policy = BackupPolicy {
//!     name: "daily".into(),
//!     schedule: CronSchedule::daily(),
//!     retention: RetentionTier::standard(),
//!     gfs: Some(GfsRotation::default()),
//!     encryption: EncryptionConfig::none(),
//!     destination: DestinationConfig::Filesystem { path: dir.clone() },
//! };
//!
//! let executor = BackupExecutor::new();
//! let size = executor.execute_backup(&policy, b"my data", &dir).unwrap();
//! assert_eq!(size, 7);
//! # let _ = std::fs::remove_dir_all(&dir);
//! ```

pub mod destination;
pub mod executor;
pub mod gfs;
pub mod policy;
pub mod retention;

pub use executor::{AuditAction, BackupAuditEntry, BackupError, BackupExecutor};
pub use gfs::{BackupRecord, GfsRotation};
pub use policy::BackupPolicy;
pub use retention::RetentionTier;
