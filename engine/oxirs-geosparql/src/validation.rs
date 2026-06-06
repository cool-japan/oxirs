//! Geometry validation and quality checks
//!
//! This module provides validation and quality checking for geometries,
//! including validity checks, repair operations, and simplification algorithms.
//!
//! # Features
//!
//! - Validity checking (self-intersections, proper topology)
//! - Automatic geometry repair
//! - Simplification (Douglas-Peucker, Visvalingam-Whyatt)
//! - Precision model handling
//! - Topology validation
//!
//! # Examples
//!
//! ```
//! use oxirs_geosparql::validation::*;
//! use oxirs_geosparql::geometry::Geometry;
//! use geo_types::{Point, Geometry as GeoGeometry};
//!
//! let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
//!
//! // Check if geometry is valid
//! let validation = validate_geometry(&geom);
//! assert!(validation.is_valid);
//!
//! // Simplify geometry
//! let simplified = simplify_geometry(&geom, 0.01).expect("simplification should succeed");
//! ```

pub use crate::validation_algorithms::*;
pub use crate::validation_core::*;
pub use crate::validation_topology::*;
