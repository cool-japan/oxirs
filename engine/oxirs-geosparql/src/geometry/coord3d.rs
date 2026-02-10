//! 3D Coordinate Support
//!
//! This module provides support for Z and M coordinates in geometries.
//! Since geo_types only supports 2D coordinates (x, y), we store Z and M
//! values separately and handle them during parsing/serialization.
//!
//! Additionally provides true 3D coordinate operations for spatial analysis.

use serde::{Deserialize, Serialize};
use scirs2_core::ndarray_ext::Array2;

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

// ============================================================================
// TRUE 3D COORDINATE OPERATIONS
// ============================================================================

/// A true 3D point with x, y, z coordinates
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Coord3DPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Coord3DPoint {
    /// Create a new 3D coordinate
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Calculate 3D Euclidean distance to another point
    pub fn distance_3d(&self, other: &Coord3DPoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculate dot product with another 3D vector
    pub fn dot_product(&self, other: &Coord3DPoint) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Calculate cross product with another 3D vector
    pub fn cross_product(&self, other: &Coord3DPoint) -> Coord3DPoint {
        Coord3DPoint {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Calculate magnitude (length) of this vector
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize this vector to unit length
    pub fn normalize(&self) -> Coord3DPoint {
        let mag = self.magnitude();
        if mag < 1e-10 {
            return *self;
        }
        Coord3DPoint {
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }

    /// Convert to array
    pub fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Convert from array
    pub fn from_array(arr: [f64; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }
}

/// 3D geometry type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Geometry3DType {
    /// Single 3D point
    Point3D,
    /// Line string in 3D
    LineString3D,
    /// Polygon in 3D
    Polygon3D,
    /// Multiple 3D points
    MultiPoint3D,
    /// Multiple 3D line strings
    MultiLineString3D,
    /// Multiple 3D polygons
    MultiPolygon3D,
    /// Polyhedral surface (collection of polygons)
    PolyhedralSurface,
    /// 3D solid (volumetric geometry)
    Solid,
}

/// Complete 3D geometry structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Geometry3D {
    /// 3D coordinates
    pub coords: Vec<Coord3DPoint>,
    /// Geometry type
    pub geometry_type: Geometry3DType,
}

impl Geometry3D {
    /// Create a new 3D geometry
    pub fn new(coords: Vec<Coord3DPoint>, geometry_type: Geometry3DType) -> Self {
        Self {
            coords,
            geometry_type,
        }
    }

    /// Create a 3D point
    pub fn point(x: f64, y: f64, z: f64) -> Self {
        Self {
            coords: vec![Coord3DPoint::new(x, y, z)],
            geometry_type: Geometry3DType::Point3D,
        }
    }

    /// Create a 3D line string
    pub fn linestring(coords: Vec<Coord3DPoint>) -> Self {
        Self {
            coords,
            geometry_type: Geometry3DType::LineString3D,
        }
    }

    /// Create a 3D polygon
    pub fn polygon(coords: Vec<Coord3DPoint>) -> Self {
        Self {
            coords,
            geometry_type: Geometry3DType::Polygon3D,
        }
    }

    /// Get the centroid of this geometry
    pub fn centroid_3d(&self) -> Option<Coord3DPoint> {
        if self.coords.is_empty() {
            return None;
        }

        let n = self.coords.len() as f64;
        let sum_x: f64 = self.coords.iter().map(|c| c.x).sum();
        let sum_y: f64 = self.coords.iter().map(|c| c.y).sum();
        let sum_z: f64 = self.coords.iter().map(|c| c.z).sum();

        Some(Coord3DPoint::new(sum_x / n, sum_y / n, sum_z / n))
    }

    /// Get 3D bounding box (min, max)
    pub fn bbox_3d(&self) -> Option<(Coord3DPoint, Coord3DPoint)> {
        if self.coords.is_empty() {
            return None;
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut max_z = f64::NEG_INFINITY;

        for coord in &self.coords {
            min_x = min_x.min(coord.x);
            min_y = min_y.min(coord.y);
            min_z = min_z.min(coord.z);
            max_x = max_x.max(coord.x);
            max_y = max_y.max(coord.y);
            max_z = max_z.max(coord.z);
        }

        Some((
            Coord3DPoint::new(min_x, min_y, min_z),
            Coord3DPoint::new(max_x, max_y, max_z),
        ))
    }

    /// Calculate volume (for 3D solid geometries)
    pub fn volume(&self) -> f64 {
        match self.geometry_type {
            Geometry3DType::Solid => {
                // Simplified volume calculation using bounding box
                if let Some((min, max)) = self.bbox_3d() {
                    (max.x - min.x) * (max.y - min.y) * (max.z - min.z)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// Calculate surface area (for 3D geometries)
    pub fn surface_area(&self) -> f64 {
        match self.geometry_type {
            Geometry3DType::Polygon3D | Geometry3DType::PolyhedralSurface => {
                // Simplified surface area calculation
                if self.coords.len() < 3 {
                    return 0.0;
                }

                let mut area = 0.0;
                for i in 1..self.coords.len() - 1 {
                    let v1 = Coord3DPoint::new(
                        self.coords[i].x - self.coords[0].x,
                        self.coords[i].y - self.coords[0].y,
                        self.coords[i].z - self.coords[0].z,
                    );
                    let v2 = Coord3DPoint::new(
                        self.coords[i + 1].x - self.coords[0].x,
                        self.coords[i + 1].y - self.coords[0].y,
                        self.coords[i + 1].z - self.coords[0].z,
                    );
                    let cross = v1.cross_product(&v2);
                    area += cross.magnitude() / 2.0;
                }
                area
            }
            Geometry3DType::Solid => {
                // Simplified: surface area of bounding box
                if let Some((min, max)) = self.bbox_3d() {
                    let dx = max.x - min.x;
                    let dy = max.y - min.y;
                    let dz = max.z - min.z;
                    2.0 * (dx * dy + dy * dz + dz * dx)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// Convert coordinates to ndarray for batch processing
    pub fn to_array(&self) -> Result<Array2<f64>, String> {
        let n = self.coords.len();
        let mut data = vec![0.0; n * 3];
        for (i, coord) in self.coords.iter().enumerate() {
            data[i * 3] = coord.x;
            data[i * 3 + 1] = coord.y;
            data[i * 3 + 2] = coord.z;
        }
        Array2::from_shape_vec((n, 3), data)
            .map_err(|e| format!("Array conversion failed: {}", e))
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

    // Tests for true 3D operations

    #[test]
    fn test_coord3d_point_creation() {
        let p = Coord3DPoint::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
    }

    #[test]
    fn test_distance_3d() {
        let p1 = Coord3DPoint::new(0.0, 0.0, 0.0);
        let p2 = Coord3DPoint::new(3.0, 4.0, 0.0);
        assert!((p1.distance_3d(&p2) - 5.0).abs() < 1e-10);

        let p3 = Coord3DPoint::new(3.0, 4.0, 12.0);
        assert!((p1.distance_3d(&p3) - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let v1 = Coord3DPoint::new(1.0, 2.0, 3.0);
        let v2 = Coord3DPoint::new(4.0, 5.0, 6.0);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((v1.dot_product(&v2) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_product() {
        let v1 = Coord3DPoint::new(1.0, 0.0, 0.0);
        let v2 = Coord3DPoint::new(0.0, 1.0, 0.0);
        let cross = v1.cross_product(&v2);
        assert!((cross.x - 0.0).abs() < 1e-10);
        assert!((cross.y - 0.0).abs() < 1e-10);
        assert!((cross.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_magnitude() {
        let v = Coord3DPoint::new(3.0, 4.0, 0.0);
        assert!((v.magnitude() - 5.0).abs() < 1e-10);

        let v2 = Coord3DPoint::new(1.0, 2.0, 2.0);
        assert!((v2.magnitude() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let v = Coord3DPoint::new(3.0, 4.0, 0.0);
        let normalized = v.normalize();
        assert!((normalized.magnitude() - 1.0).abs() < 1e-10);
        assert!((normalized.x - 0.6).abs() < 1e-10);
        assert!((normalized.y - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_geometry3d_point() {
        let geom = Geometry3D::point(1.0, 2.0, 3.0);
        assert_eq!(geom.geometry_type, Geometry3DType::Point3D);
        assert_eq!(geom.coords.len(), 1);
        assert_eq!(geom.coords[0].x, 1.0);
    }

    #[test]
    fn test_geometry3d_centroid() {
        let coords = vec![
            Coord3DPoint::new(0.0, 0.0, 0.0),
            Coord3DPoint::new(2.0, 0.0, 0.0),
            Coord3DPoint::new(2.0, 2.0, 0.0),
            Coord3DPoint::new(0.0, 2.0, 0.0),
        ];
        let geom = Geometry3D::polygon(coords);
        let centroid = geom.centroid_3d().expect("should have centroid");
        assert!((centroid.x - 1.0).abs() < 1e-10);
        assert!((centroid.y - 1.0).abs() < 1e-10);
        assert!((centroid.z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometry3d_bbox() {
        let coords = vec![
            Coord3DPoint::new(1.0, 2.0, 3.0),
            Coord3DPoint::new(5.0, 6.0, 7.0),
            Coord3DPoint::new(2.0, 3.0, 4.0),
        ];
        let geom = Geometry3D::linestring(coords);
        let (min, max) = geom.bbox_3d().expect("should have bbox");
        assert_eq!(min.x, 1.0);
        assert_eq!(min.y, 2.0);
        assert_eq!(min.z, 3.0);
        assert_eq!(max.x, 5.0);
        assert_eq!(max.y, 6.0);
        assert_eq!(max.z, 7.0);
    }

    #[test]
    fn test_geometry3d_volume() {
        let coords = vec![
            Coord3DPoint::new(0.0, 0.0, 0.0),
            Coord3DPoint::new(2.0, 3.0, 4.0),
        ];
        let mut geom = Geometry3D::new(coords, Geometry3DType::Solid);
        let volume = geom.volume();
        assert!((volume - 24.0).abs() < 1e-10); // 2 * 3 * 4 = 24
    }

    #[test]
    fn test_geometry3d_surface_area() {
        let coords = vec![
            Coord3DPoint::new(0.0, 0.0, 0.0),
            Coord3DPoint::new(1.0, 0.0, 0.0),
            Coord3DPoint::new(1.0, 1.0, 0.0),
            Coord3DPoint::new(0.0, 1.0, 0.0),
        ];
        let geom = Geometry3D::polygon(coords);
        let area = geom.surface_area();
        assert!(area > 0.0);
    }
}
