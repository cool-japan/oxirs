//! 3D Coordinate Support
//!
//! This module provides support for Z and M coordinates in geometries.
//! Since geo_types only supports 2D coordinates (x, y), we store Z and M
//! values separately and handle them during parsing/serialization.

use serde::{Deserialize, Serialize};

/// Coordinate dimension type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoordDim {
    /// 2D coordinates (X, Y)
    XY,
    /// 3D coordinates with Z (X, Y, Z)
    XYZ,
    /// 2D coordinates with M (measure) (X, Y, M)
    XYM,
    /// 3D coordinates with Z and M (X, Y, Z, M)
    XYZM,
}

impl CoordDim {
    /// Check if this dimension includes Z coordinate
    pub fn has_z(&self) -> bool {
        matches!(self, CoordDim::XYZ | CoordDim::XYZM)
    }

    /// Check if this dimension includes M coordinate
    pub fn has_m(&self) -> bool {
        matches!(self, CoordDim::XYM | CoordDim::XYZM)
    }

    /// Get the number of coordinate values per point
    pub fn coord_count(&self) -> usize {
        match self {
            CoordDim::XY => 2,
            CoordDim::XYZ | CoordDim::XYM => 3,
            CoordDim::XYZM => 4,
        }
    }

    /// Parse from WKT modifier string (e.g., "Z", "M", "ZM")
    pub fn from_wkt_modifier(modifier: Option<&str>) -> Self {
        match modifier {
            Some("Z") => CoordDim::XYZ,
            Some("M") => CoordDim::XYM,
            Some("ZM") => CoordDim::XYZM,
            _ => CoordDim::XY,
        }
    }

    /// Convert to WKT modifier string
    pub fn to_wkt_modifier(&self) -> Option<&'static str> {
        match self {
            CoordDim::XY => None,
            CoordDim::XYZ => Some("Z"),
            CoordDim::XYM => Some("M"),
            CoordDim::XYZM => Some("ZM"),
        }
    }
}

/// Storage for 3D coordinates (Z values)
/// Each value corresponds to a coordinate in the geometry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZCoords {
    /// Z coordinate values
    pub values: Vec<f64>,
}

impl ZCoords {
    /// Create new Z coordinates
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get number of Z values
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Get Z value at index
    pub fn get(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied()
    }
}

/// Storage for measured coordinates (M values)
/// Each value corresponds to a coordinate in the geometry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MCoords {
    /// M coordinate values
    pub values: Vec<f64>,
}

impl MCoords {
    /// Create new M coordinates
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get number of M values
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Get M value at index
    pub fn get(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied()
    }
}

/// 3D coordinate metadata for a geometry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Coord3D {
    /// Coordinate dimension type
    pub dim: CoordDim,
    /// Z coordinates (if present)
    pub z_coords: Option<ZCoords>,
    /// M coordinates (if present)
    pub m_coords: Option<MCoords>,
}

impl Coord3D {
    /// Create new 2D coordinates (no Z or M)
    pub fn xy() -> Self {
        Self {
            dim: CoordDim::XY,
            z_coords: None,
            m_coords: None,
        }
    }

    /// Create new 3D coordinates with Z
    pub fn xyz(z_values: Vec<f64>) -> Self {
        Self {
            dim: CoordDim::XYZ,
            z_coords: Some(ZCoords::new(z_values)),
            m_coords: None,
        }
    }

    /// Create new 2D coordinates with M
    pub fn xym(m_values: Vec<f64>) -> Self {
        Self {
            dim: CoordDim::XYM,
            z_coords: None,
            m_coords: Some(MCoords::new(m_values)),
        }
    }

    /// Create new 3D coordinates with Z and M
    pub fn xyzm(z_values: Vec<f64>, m_values: Vec<f64>) -> Self {
        Self {
            dim: CoordDim::XYZM,
            z_coords: Some(ZCoords::new(z_values)),
            m_coords: Some(MCoords::new(m_values)),
        }
    }

    /// Check if this has Z coordinates
    pub fn has_z(&self) -> bool {
        self.dim.has_z()
    }

    /// Check if this has M coordinates
    pub fn has_m(&self) -> bool {
        self.dim.has_m()
    }

    /// Get Z value at index
    pub fn z_at(&self, index: usize) -> Option<f64> {
        self.z_coords.as_ref().and_then(|z| z.get(index))
    }

    /// Get M value at index
    pub fn m_at(&self, index: usize) -> Option<f64> {
        self.m_coords.as_ref().and_then(|m| m.get(index))
    }

    /// Validate that Z/M coordinate counts match the expected number of points
    pub fn validate(&self, expected_point_count: usize) -> Result<(), String> {
        if let Some(ref z) = self.z_coords {
            if z.len() != expected_point_count {
                return Err(format!(
                    "Z coordinate count ({}) doesn't match point count ({})",
                    z.len(),
                    expected_point_count
                ));
            }
        }

        if let Some(ref m) = self.m_coords {
            if m.len() != expected_point_count {
                return Err(format!(
                    "M coordinate count ({}) doesn't match point count ({})",
                    m.len(),
                    expected_point_count
                ));
            }
        }

        Ok(())
    }
}

impl Default for Coord3D {
    fn default() -> Self {
        Self::xy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord_dim_has_z() {
        assert!(!CoordDim::XY.has_z());
        assert!(CoordDim::XYZ.has_z());
        assert!(!CoordDim::XYM.has_z());
        assert!(CoordDim::XYZM.has_z());
    }

    #[test]
    fn test_coord_dim_has_m() {
        assert!(!CoordDim::XY.has_m());
        assert!(!CoordDim::XYZ.has_m());
        assert!(CoordDim::XYM.has_m());
        assert!(CoordDim::XYZM.has_m());
    }

    #[test]
    fn test_coord_dim_coord_count() {
        assert_eq!(CoordDim::XY.coord_count(), 2);
        assert_eq!(CoordDim::XYZ.coord_count(), 3);
        assert_eq!(CoordDim::XYM.coord_count(), 3);
        assert_eq!(CoordDim::XYZM.coord_count(), 4);
    }

    #[test]
    fn test_coord_dim_from_wkt_modifier() {
        assert_eq!(CoordDim::from_wkt_modifier(None), CoordDim::XY);
        assert_eq!(CoordDim::from_wkt_modifier(Some("Z")), CoordDim::XYZ);
        assert_eq!(CoordDim::from_wkt_modifier(Some("M")), CoordDim::XYM);
        assert_eq!(CoordDim::from_wkt_modifier(Some("ZM")), CoordDim::XYZM);
    }

    #[test]
    fn test_coord_dim_to_wkt_modifier() {
        assert_eq!(CoordDim::XY.to_wkt_modifier(), None);
        assert_eq!(CoordDim::XYZ.to_wkt_modifier(), Some("Z"));
        assert_eq!(CoordDim::XYM.to_wkt_modifier(), Some("M"));
        assert_eq!(CoordDim::XYZM.to_wkt_modifier(), Some("ZM"));
    }

    #[test]
    fn test_coord3d_xy() {
        let coord = Coord3D::xy();
        assert!(!coord.has_z());
        assert!(!coord.has_m());
        assert_eq!(coord.dim, CoordDim::XY);
    }

    #[test]
    fn test_coord3d_xyz() {
        let coord = Coord3D::xyz(vec![1.0, 2.0, 3.0]);
        assert!(coord.has_z());
        assert!(!coord.has_m());
        assert_eq!(coord.z_at(0), Some(1.0));
        assert_eq!(coord.z_at(1), Some(2.0));
        assert_eq!(coord.z_at(2), Some(3.0));
    }

    #[test]
    fn test_coord3d_xym() {
        let coord = Coord3D::xym(vec![10.0, 20.0, 30.0]);
        assert!(!coord.has_z());
        assert!(coord.has_m());
        assert_eq!(coord.m_at(0), Some(10.0));
        assert_eq!(coord.m_at(1), Some(20.0));
        assert_eq!(coord.m_at(2), Some(30.0));
    }

    #[test]
    fn test_coord3d_xyzm() {
        let coord = Coord3D::xyzm(vec![1.0, 2.0], vec![10.0, 20.0]);
        assert!(coord.has_z());
        assert!(coord.has_m());
        assert_eq!(coord.z_at(0), Some(1.0));
        assert_eq!(coord.m_at(1), Some(20.0));
    }

    #[test]
    fn test_coord3d_validate() {
        let coord = Coord3D::xyz(vec![1.0, 2.0, 3.0]);
        assert!(coord.validate(3).is_ok());
        assert!(coord.validate(2).is_err());
    }
}
