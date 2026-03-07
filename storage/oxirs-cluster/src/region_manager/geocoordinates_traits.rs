//! # GeoCoordinates - Trait Implementations
//!
//! This module contains trait implementations for `GeoCoordinates`.
//!
//! ## Implemented Traits
//!
//! - `Hash`
//! - `Eq`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::GeoCoordinates;

impl std::hash::Hash for GeoCoordinates {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.latitude.to_bits().hash(state);
        self.longitude.to_bits().hash(state);
    }
}

impl Eq for GeoCoordinates {}
