//! Collaborative Shape Development
//!
//! This module implements comprehensive collaborative features for shape development,
//! including multi-user management, real-time collaboration, conflict resolution,
//! and shape reusability.
//!
//! The module is organized into several sub-modules:
//! - `config`: Configuration and basic types
//! - `types`: Core data types and structures  
//! - `workspace`: Workspace management and sessions
//! - `real_time`: Real-time collaboration engine
//! - `conflict_resolution`: Conflict detection and resolution
//! - `library`: Shape library and contribution system
//! - `review`: Review system and workflows
//! - `statistics`: Statistics and analytics

// Re-export all collaborative shape functionality from modular structure
pub use self::collaborative_shapes::*;

// Import the modular collaborative_shapes implementation
mod collaborative_shapes;