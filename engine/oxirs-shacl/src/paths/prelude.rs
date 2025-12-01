//! Common imports for paths module
//!
//! This prelude provides all common types and functions used across the paths modules.

pub use crate::{Result, ShaclError};
pub use oxirs_core::{
    model::{NamedNode, RdfTerm, Term},
    OxirsError, Store,
};
pub use serde::{Deserialize, Serialize};
pub use std::collections::{HashMap, HashSet, VecDeque};

// Re-export all types from this module
pub use super::functions::*;
pub use super::propertypathevaluator_type::*;
pub use super::types::*;
