//! Model zoo: registry of pretrained (and synthetic-seed) embedding models.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use oxirs_embed::model_zoo::{ModelZoo, ModelZooLoader};
//!
//! // List all catalog entries
//! for m in ModelZoo::registry().list() {
//!     println!("{}: {} on {} ({}d)", m.name, m.model_type, m.dataset, m.dimensions);
//! }
//!
//! // Load a synthetic-seed model (example; requires the checkpoint on disk)
//! // let loader = ModelZooLoader::new(std::env::temp_dir()).accept_license();
//! // let model = loader.load("transe-fb15k237")?;
//! ```

pub mod loader;
pub mod manifest;
pub mod registry;

pub use loader::{sha256_hex, ModelZooError, ModelZooLoader};
pub use manifest::ModelManifest;
pub use registry::ModelZoo;
