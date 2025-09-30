//! Federated learning module with organized sub-modules

pub mod aggregation;
pub mod config;
pub mod federated_learning_impl;
pub mod participant;
pub mod privacy;

// Re-export the main implementation
pub use self::aggregation::*;
pub use self::config::*;
pub use self::federated_learning_impl::*;
pub use self::participant::*;
pub use self::privacy::*;
