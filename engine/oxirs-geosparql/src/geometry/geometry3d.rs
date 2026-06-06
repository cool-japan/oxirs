//! 3D Geometry Types and Operations
//!
//! Extends 2D geometry with Z (elevation) coordinate following:
//! - ISO 19107 Geographic Information — Spatial Schema
//! - OGC Simple Features for SQL specification
//! - WKT serialization as per OGC 06-103r4
//!
//! These types provide rich 3D operations including WKT roundtrip,
//! 3D bounding boxes, z-range queries, and flattening to 2D.
//!
//! This module is a thin facade. The implementation is split across:
//! - [`super::geometry3d_types`] — the value types (`Point3D`, `LinearRing3D`,
//!   `LineString3D`, `Polygon3D`, `BoundingBox3D`, `Geometry3DEnum`) and their
//!   geometric operations.
//! - [`super::geometry3d_wkt`]   — WKT parsing/serialization and Z-range helpers.

pub use super::geometry3d_types::{
    BoundingBox3D, Geometry3DEnum, LineString3D, LinearRing3D, Point3D, Polygon3D,
};
