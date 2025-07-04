//! Multi-modal embeddings module with organized sub-modules

pub mod config;
pub mod encoders;
pub mod learning;
pub mod adaptation;
pub mod model;

// Re-export all public types for convenience
pub use self::config::*;
pub use self::encoders::*;
pub use self::learning::*;
pub use self::adaptation::*;
pub use self::model::*;