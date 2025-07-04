//! Federated learning module with organized sub-modules

pub mod aggregation;
pub mod config;
pub mod participant;
pub mod privacy;
pub mod federated_learning_impl;

// Re-export the main implementation
pub use self::aggregation::*;
pub use self::config::*;
pub use self::participant::*;
pub use self::privacy::*;
pub use self::federated_learning_impl::*;