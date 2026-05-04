//! Model zoo for pretrained SHACL shape-learning models.
//!
//! This module provides:
//!
//! - [`ShapeModelManifest`] — TOML-serialisable descriptor for a checkpoint.
//! - [`ShapeModelZoo`] — in-memory catalogue; `registry()` returns the global
//!   static instance pre-populated with four built-in manifests.
//! - [`ShapeModelZooLoader`] — SHA-256-verified checkpoint loader.
//! - [`ShapeModelZooError`] — error type for registry and loader operations.
//! - [`LoadedShapeModel`] — output of a successful `load()` call.
//!
//! ## Quick start
//!
//! ```rust
//! use oxirs_shacl_ai::model_zoo::{ShapeModelZoo, ShapeModelZooLoader};
//!
//! // List all bundled models
//! for m in ShapeModelZoo::registry().list() {
//!     println!("{}: {} ({})", m.name, m.model_type, m.license);
//! }
//!
//! // Search by keyword
//! let gat_models = ShapeModelZoo::registry().by_model_type("GAT");
//! assert!(!gat_models.is_empty());
//!
//! // Create a loader (would load from disk in production)
//! let _loader = ShapeModelZooLoader::new(std::env::temp_dir());
//! ```

pub mod loader;
pub mod manifest;
pub mod registry;

pub use loader::{LoadedShapeModel, ShapeModelZooError, ShapeModelZooLoader};
pub use manifest::ShapeModelManifest;
pub use registry::ShapeModelZoo;
