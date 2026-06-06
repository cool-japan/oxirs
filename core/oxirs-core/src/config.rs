//! Advanced Configuration System for OxiRS Core — thin facade module.
//!
//! This module provides a comprehensive configuration system with performance
//! profiles, environment-based overrides, and dynamic reconfiguration capabilities.
//!
//! The implementation is split across sibling modules to keep each file within the
//! workspace size policy:
//! - [`crate::config_types`] — composes `OxirsConfig`, `ConfigurationManager`,
//!   `ConfigWatcher` and re-exports the per-section types from
//!   `config_types_core` / `config_types_network` / `config_types_storage`.
//! - [`crate::config_parser`] — the `ConfigurationManager` implementation
//!   (file/environment loading, profile management, watchers).
//! - [`crate::config_validation`] — `validate_config` and its section validators.
//!
//! All public items are re-exported here so callers can keep using `crate::config::*`.

pub use crate::config_types::*;

// --- Tests (extracted to config_tests.rs for file size compliance) ---

include!("config_tests.rs");
