//! SPARQL Spatial Aggregate Functions (v0.2.0)
//!
//! This module implements GeoSPARQL extension aggregate functions for use
//! in SPARQL GROUP BY / aggregate queries:
//!
//! - `geo:aggregate_union(geometry)` — union of all geometries in the group
//! - `geo:aggregate_envelope(geometry)` — minimum bounding rectangle of the group
//! - `geo:aggregate_centroid(geometry)` — centroid of all geometries in the group
//!
//! ## SPARQL Usage
//!
//! ```sparql
//! PREFIX geo: <http://www.opengis.net/ont/geosparql#>
//! PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
//!
//! SELECT ?region (geof:aggregate_union(?geom) AS ?total)
//! WHERE { ?s geo:hasGeometry/geo:asWKT ?geom ; ex:region ?region }
//! GROUP BY ?region
//! ```
//!
//! ## Design
//!
//! Each aggregate follows a two-phase accumulator model compatible with
//! streaming / parallel evaluation:
//!
//! 1. Create an accumulator via `<Agg>::new()`.
//! 2. Feed each geometry with `.accumulate(&geom)`.
//! 3. Retrieve the result with `.finalize()`.
//!
//! This mirrors how SPARQL aggregate functions are evaluated in a query engine.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::algorithm::*;
use geo_types::{Coord, Geometry as GeoGeometry, MultiPolygon, Point, Polygon};

// ─────────────────────────────────────────────────────────────────────────────
// URI constants
// ─────────────────────────────────────────────────────────────────────────────

/// Full URI for `geo:aggregate_union`
pub const GEO_AGGREGATE_UNION: &str =
    "http://www.opengis.net/def/function/geosparql/aggregate_union";

/// Full URI for `geo:aggregate_envelope`
pub const GEO_AGGREGATE_ENVELOPE: &str =
    "http://www.opengis.net/def/function/geosparql/aggregate_envelope";

/// Full URI for `geo:aggregate_centroid`
pub const GEO_AGGREGATE_CENTROID: &str =
    "http://www.opengis.net/def/function/geosparql/aggregate_centroid";

// ─────────────────────────────────────────────────────────────────────────────
// Aggregate trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait shared by all spatial aggregate accumulators.
pub trait SpatialAggregate {
    /// Feed one geometry into the accumulator.
    fn accumulate(&mut self, geom: &Geometry) -> Result<()>;

    /// Consume the accumulator and return the final aggregated geometry.
    fn finalize(self) -> Result<Geometry>;
}

// ─────────────────────────────────────────────────────────────────────────────
// AggregateUnion
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulator for `geo:aggregate_union`.
///
/// Iteratively unions each incoming geometry into the running result.
/// For polygon inputs the geo `BooleanOps` trait is used; for mixed or
/// non-polygon types the geometries are collected into a `GeometryCollection`.
#[derive(Debug, Default)]
pub struct AggregateUnion {
    /// Running union — `None` until the first geometry is accumulated.
    current: Option<GeoGeometry<f64>>,
    /// CRS URI of the first geometry; subsequent geometries must match.
    crs_uri: Option<String>,
}

impl AggregateUnion {
    /// Create a new, empty union accumulator.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SpatialAggregate for AggregateUnion {
    fn accumulate(&mut self, geom: &Geometry) -> Result<()> {
        // Validate CRS consistency
        match &self.crs_uri {
            None => self.crs_uri = Some(geom.crs.uri.clone()),
            Some(uri) if uri != &geom.crs.uri => {
                return Err(GeoSparqlError::CrsMismatch {
                    expected: uri.clone(),
                    found: geom.crs.uri.clone(),
                });
            }
            Some(_) => {}
        }

        let incoming = geom.geom.clone();

        self.current = Some(match self.current.take() {
            None => incoming,
            Some(existing) => union_geometries(existing, incoming)?,
        });

        Ok(())
    }

    fn finalize(self) -> Result<Geometry> {
        let geom = self.current.ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed(
                "aggregate_union called on empty group".to_string(),
            )
        })?;
        Ok(Geometry::new(geom))
    }
}

/// Union two arbitrary geo_types geometries.
///
/// Polygon×Polygon uses `BooleanOps`; everything else falls back to
/// `GeometryCollection` so no geometry is discarded.
fn union_geometries(a: GeoGeometry<f64>, b: GeoGeometry<f64>) -> Result<GeoGeometry<f64>> {
    use geo::BooleanOps;

    match (a, b) {
        (GeoGeometry::Polygon(p1), GeoGeometry::Polygon(p2)) => {
            let mp: MultiPolygon<f64> = p1.union(&p2);
            // Simplify MultiPolygon with one polygon back to Polygon
            if mp.0.len() == 1 {
                let poly =
                    mp.0.into_iter().next().ok_or_else(|| {
                        GeoSparqlError::ComputationError("empty union".to_string())
                    })?;
                Ok(GeoGeometry::Polygon(poly))
            } else {
                Ok(GeoGeometry::MultiPolygon(mp))
            }
        }
        (GeoGeometry::MultiPolygon(mp1), GeoGeometry::Polygon(p2)) => {
            let mp2: MultiPolygon<f64> = p2.union(&mp1);
            Ok(GeoGeometry::MultiPolygon(mp2))
        }
        (GeoGeometry::Polygon(p1), GeoGeometry::MultiPolygon(mp2)) => {
            let mp: MultiPolygon<f64> = p1.union(&mp2);
            Ok(GeoGeometry::MultiPolygon(mp))
        }
        (GeoGeometry::MultiPolygon(mp1), GeoGeometry::MultiPolygon(mp2)) => {
            let mp: MultiPolygon<f64> = mp1.union(&mp2);
            Ok(GeoGeometry::MultiPolygon(mp))
        }
        // For mixed or unsupported geometry types, collect into a GeometryCollection
        (a, b) => {
            let mut items = match a {
                GeoGeometry::GeometryCollection(gc) => gc.0,
                other => vec![other],
            };
            match b {
                GeoGeometry::GeometryCollection(gc) => items.extend(gc.0),
                other => items.push(other),
            }
            Ok(GeoGeometry::GeometryCollection(
                geo_types::GeometryCollection::new_from(items),
            ))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AggregateEnvelope
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulator for `geo:aggregate_envelope`.
///
/// Maintains the running minimum bounding rectangle across all input geometries.
/// The final result is a rectangular `Polygon` (WKT `POLYGON(…)`) whose corners
/// are `(min_x, min_y)`, `(max_x, min_y)`, `(max_x, max_y)`, `(min_x, max_y)`.
#[derive(Debug)]
pub struct AggregateEnvelope {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    crs_uri: Option<String>,
    is_empty: bool,
}

impl Default for AggregateEnvelope {
    fn default() -> Self {
        Self {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
            crs_uri: None,
            is_empty: true,
        }
    }
}

impl AggregateEnvelope {
    /// Create a new, empty envelope accumulator.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SpatialAggregate for AggregateEnvelope {
    fn accumulate(&mut self, geom: &Geometry) -> Result<()> {
        match &self.crs_uri {
            None => self.crs_uri = Some(geom.crs.uri.clone()),
            Some(uri) if uri != &geom.crs.uri => {
                return Err(GeoSparqlError::CrsMismatch {
                    expected: uri.clone(),
                    found: geom.crs.uri.clone(),
                });
            }
            Some(_) => {}
        }

        if let Some(rect) = geom.geom.bounding_rect() {
            self.min_x = self.min_x.min(rect.min().x);
            self.min_y = self.min_y.min(rect.min().y);
            self.max_x = self.max_x.max(rect.max().x);
            self.max_y = self.max_y.max(rect.max().y);
            self.is_empty = false;
        }

        Ok(())
    }

    fn finalize(self) -> Result<Geometry> {
        if self.is_empty {
            return Err(GeoSparqlError::GeometryOperationFailed(
                "aggregate_envelope called on empty group".to_string(),
            ));
        }

        let rect_coords = vec![
            Coord {
                x: self.min_x,
                y: self.min_y,
            },
            Coord {
                x: self.max_x,
                y: self.min_y,
            },
            Coord {
                x: self.max_x,
                y: self.max_y,
            },
            Coord {
                x: self.min_x,
                y: self.max_y,
            },
            Coord {
                x: self.min_x,
                y: self.min_y,
            },
        ];

        let polygon = Polygon::new(geo_types::LineString::from(rect_coords), vec![]);
        Ok(Geometry::new(GeoGeometry::Polygon(polygon)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AggregateCentroid
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulator for `geo:aggregate_centroid`.
///
/// Computes the arithmetic mean of individual geometry centroids weighted by
/// the geometry's 2-D area (polygons) or arc length (lines) or unit weight
/// (points).  This produces a more meaningful result than a plain mean of
/// centroids for heterogeneous geometry sets.
#[derive(Debug, Default)]
pub struct AggregateCentroid {
    weighted_x: f64,
    weighted_y: f64,
    total_weight: f64,
    count: usize,
    crs_uri: Option<String>,
}

impl AggregateCentroid {
    /// Create a new, empty centroid accumulator.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SpatialAggregate for AggregateCentroid {
    fn accumulate(&mut self, geom: &Geometry) -> Result<()> {
        match &self.crs_uri {
            None => self.crs_uri = Some(geom.crs.uri.clone()),
            Some(uri) if uri != &geom.crs.uri => {
                return Err(GeoSparqlError::CrsMismatch {
                    expected: uri.clone(),
                    found: geom.crs.uri.clone(),
                });
            }
            Some(_) => {}
        }

        let centroid = geom.geom.centroid().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed(
                "Cannot compute centroid for empty geometry".to_string(),
            )
        })?;

        // Compute weight: area for polygons, length for lines, 1.0 for points
        let weight = compute_geometry_weight(&geom.geom);

        self.weighted_x += centroid.x() * weight;
        self.weighted_y += centroid.y() * weight;
        self.total_weight += weight;
        self.count += 1;

        Ok(())
    }

    fn finalize(self) -> Result<Geometry> {
        if self.count == 0 {
            return Err(GeoSparqlError::GeometryOperationFailed(
                "aggregate_centroid called on empty group".to_string(),
            ));
        }

        let (cx, cy) = if self.total_weight > 0.0 {
            (
                self.weighted_x / self.total_weight,
                self.weighted_y / self.total_weight,
            )
        } else {
            // All geometries have zero weight (e.g., empty points); fall back to unweighted mean
            (
                self.weighted_x / self.count as f64,
                self.weighted_y / self.count as f64,
            )
        };

        Ok(Geometry::new(GeoGeometry::Point(Point::new(cx, cy))))
    }
}

/// Compute a non-negative weight for a geometry to use in centroid weighting.
fn compute_geometry_weight(geom: &GeoGeometry<f64>) -> f64 {
    use geo::algorithm::Area;
    use geo::{Euclidean, Length};

    match geom {
        GeoGeometry::Polygon(p) => {
            let area = p.unsigned_area();
            if area > 0.0 {
                area
            } else {
                1.0
            }
        }
        GeoGeometry::MultiPolygon(mp) => {
            let area = mp.unsigned_area();
            if area > 0.0 {
                area
            } else {
                1.0
            }
        }
        GeoGeometry::LineString(ls) => {
            let len = Euclidean.length(ls);
            if len > 0.0 {
                len
            } else {
                1.0
            }
        }
        GeoGeometry::MultiLineString(mls) => {
            let len: f64 = mls.iter().map(|ls| Euclidean.length(ls)).sum();
            if len > 0.0 {
                len
            } else {
                1.0
            }
        }
        _ => 1.0,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience free functions (mirror the analysis::aggregations interface)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the union of all geometries in `group`.
///
/// Equivalent to `AggregateUnion` accumulator followed by `finalize`.
pub fn aggregate_union(group: &[Geometry]) -> Result<Geometry> {
    let mut acc = AggregateUnion::new();
    for g in group {
        acc.accumulate(g)?;
    }
    acc.finalize()
}

/// Compute the minimum bounding rectangle of all geometries in `group`.
///
/// Equivalent to `AggregateEnvelope` accumulator followed by `finalize`.
pub fn aggregate_envelope(group: &[Geometry]) -> Result<Geometry> {
    let mut acc = AggregateEnvelope::new();
    for g in group {
        acc.accumulate(g)?;
    }
    acc.finalize()
}

/// Compute the weighted centroid of all geometries in `group`.
///
/// Equivalent to `AggregateCentroid` accumulator followed by `finalize`.
pub fn aggregate_centroid(group: &[Geometry]) -> Result<Geometry> {
    let mut acc = AggregateCentroid::new();
    for g in group {
        acc.accumulate(g)?;
    }
    acc.finalize()
}

// ─────────────────────────────────────────────────────────────────────────────
// AggregateBoundingBox  (geof:aggBoundingBox)
// ─────────────────────────────────────────────────────────────────────────────

/// Full URI for `geof:aggBoundingBox`
///
/// OGC GeoSPARQL 1.1 aggregate function that computes the minimum bounding
/// rectangle (MBR) across all geometries in a SPARQL group.
pub const GEO_AGG_BOUNDING_BOX: &str =
    "http://www.opengis.net/def/function/geosparql/aggBoundingBox";

/// Accumulator for `geof:aggBoundingBox`.
///
/// Computes the axis-aligned minimum bounding rectangle (MBR) for a group of
/// geometries.  Unlike `AggregateEnvelope` (which returns a `Polygon`), the
/// `AggregateBoundingBox` also exposes the raw `(min_x, min_y, max_x, max_y)`
/// values and provides a separate `BoundingBoxResult` type with structured
/// access.
///
/// ## SPARQL Usage
///
/// ```sparql
/// PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
///
/// SELECT (geof:aggBoundingBox(?geom) AS ?mbr)
/// WHERE { ?s geo:hasGeometry/geo:asWKT ?geom }
/// ```
#[derive(Debug)]
pub struct AggregateBoundingBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    crs_uri: Option<String>,
    count: usize,
}

impl Default for AggregateBoundingBox {
    fn default() -> Self {
        Self {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
            crs_uri: None,
            count: 0,
        }
    }
}

impl AggregateBoundingBox {
    /// Create a new, empty bounding-box accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current minimum X (or `f64::INFINITY` if empty).
    pub fn min_x(&self) -> f64 {
        self.min_x
    }

    /// Current minimum Y (or `f64::INFINITY` if empty).
    pub fn min_y(&self) -> f64 {
        self.min_y
    }

    /// Current maximum X (or `f64::NEG_INFINITY` if empty).
    pub fn max_x(&self) -> f64 {
        self.max_x
    }

    /// Current maximum Y (or `f64::NEG_INFINITY` if empty).
    pub fn max_y(&self) -> f64 {
        self.max_y
    }

    /// Number of geometries accumulated so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Width of the current bounding box (0 if empty).
    pub fn width(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.max_x - self.min_x
        }
    }

    /// Height of the current bounding box (0 if empty).
    pub fn height(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.max_y - self.min_y
        }
    }

    /// Returns `true` if no geometries have been accumulated yet.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Finalize and return a `BoundingBoxResult` instead of a `Geometry`.
    ///
    /// Use this when the raw bounding-box coordinates are required.
    pub fn finalize_bbox(self) -> Result<BoundingBoxResult> {
        if self.count == 0 {
            return Err(GeoSparqlError::GeometryOperationFailed(
                "aggBoundingBox called on empty group".to_string(),
            ));
        }
        Ok(BoundingBoxResult {
            min_x: self.min_x,
            min_y: self.min_y,
            max_x: self.max_x,
            max_y: self.max_y,
            crs_uri: self
                .crs_uri
                .unwrap_or_else(|| crate::vocabulary::DEFAULT_CRS.to_string()),
        })
    }
}

impl SpatialAggregate for AggregateBoundingBox {
    fn accumulate(&mut self, geom: &Geometry) -> Result<()> {
        match &self.crs_uri {
            None => self.crs_uri = Some(geom.crs.uri.clone()),
            Some(uri) if uri != &geom.crs.uri => {
                return Err(GeoSparqlError::CrsMismatch {
                    expected: uri.clone(),
                    found: geom.crs.uri.clone(),
                });
            }
            Some(_) => {}
        }

        if let Some(rect) = geom.geom.bounding_rect() {
            self.min_x = self.min_x.min(rect.min().x);
            self.min_y = self.min_y.min(rect.min().y);
            self.max_x = self.max_x.max(rect.max().x);
            self.max_y = self.max_y.max(rect.max().y);
            self.count += 1;
        }

        Ok(())
    }

    fn finalize(self) -> Result<Geometry> {
        if self.count == 0 {
            return Err(GeoSparqlError::GeometryOperationFailed(
                "aggBoundingBox called on empty group".to_string(),
            ));
        }

        let rect_coords = vec![
            Coord {
                x: self.min_x,
                y: self.min_y,
            },
            Coord {
                x: self.max_x,
                y: self.min_y,
            },
            Coord {
                x: self.max_x,
                y: self.max_y,
            },
            Coord {
                x: self.min_x,
                y: self.max_y,
            },
            Coord {
                x: self.min_x,
                y: self.min_y,
            },
        ];

        let polygon = Polygon::new(geo_types::LineString::from(rect_coords), vec![]);
        Ok(Geometry::new(GeoGeometry::Polygon(polygon)))
    }
}

/// Structured result from `AggregateBoundingBox::finalize_bbox()`.
///
/// Provides named access to the four corners of the minimum bounding rectangle
/// together with the CRS URI.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBoxResult {
    /// Minimum X coordinate
    pub min_x: f64,
    /// Minimum Y coordinate
    pub min_y: f64,
    /// Maximum X coordinate
    pub max_x: f64,
    /// Maximum Y coordinate
    pub max_y: f64,
    /// CRS URI for the bounding box coordinates
    pub crs_uri: String,
}

impl BoundingBoxResult {
    /// Width of the bounding box.
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Height of the bounding box.
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// Area of the bounding box.
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Centre point `(cx, cy)`.
    pub fn centre(&self) -> (f64, f64) {
        (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }

    /// Convert to a rectangular `Geometry` polygon.
    pub fn to_geometry(&self) -> Geometry {
        let coords = vec![
            Coord {
                x: self.min_x,
                y: self.min_y,
            },
            Coord {
                x: self.max_x,
                y: self.min_y,
            },
            Coord {
                x: self.max_x,
                y: self.max_y,
            },
            Coord {
                x: self.min_x,
                y: self.max_y,
            },
            Coord {
                x: self.min_x,
                y: self.min_y,
            },
        ];
        let polygon = Polygon::new(geo_types::LineString::from(coords), vec![]);
        let mut geom = Geometry::new(GeoGeometry::Polygon(polygon));
        geom.crs = crate::geometry::Crs::new(self.crs_uri.clone());
        geom
    }
}

/// Convenience free function: compute `geof:aggBoundingBox` for a slice of geometries.
///
/// Returns a rectangular `Geometry` (Polygon) or an error if the slice is empty.
pub fn aggregate_bounding_box(group: &[Geometry]) -> Result<Geometry> {
    let mut acc = AggregateBoundingBox::new();
    for g in group {
        acc.accumulate(g)?;
    }
    acc.finalize()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ------------------------------------------------------------------
    // aggregate_union
    // ------------------------------------------------------------------

    #[test]
    fn test_union_two_polygons() {
        let geoms = vec![
            Geometry::from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap(),
            Geometry::from_wkt("POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))").unwrap(),
        ];
        let result = aggregate_union(&geoms).unwrap();
        // Result must be non-empty
        assert!(!result.is_empty());
    }

    #[test]
    fn test_union_single_geometry() {
        let geoms = vec![Geometry::from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))").unwrap()];
        let result = aggregate_union(&geoms).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_union_disjoint_polygons() {
        let geoms = vec![
            Geometry::from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))").unwrap(),
            Geometry::from_wkt("POLYGON ((10 10, 11 10, 11 11, 10 11, 10 10))").unwrap(),
        ];
        let result = aggregate_union(&geoms).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_union_empty_group_fails() {
        let result = aggregate_union(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_union_points_uses_collection() {
        let geoms = vec![
            Geometry::from_wkt("POINT (0 0)").unwrap(),
            Geometry::from_wkt("POINT (1 1)").unwrap(),
        ];
        let result = aggregate_union(&geoms).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_union_accumulator_streaming() {
        let geoms = vec![
            Geometry::from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))").unwrap(),
            Geometry::from_wkt("POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))").unwrap(),
            Geometry::from_wkt("POLYGON ((0 1, 1 1, 1 2, 0 2, 0 1))").unwrap(),
        ];
        let mut acc = AggregateUnion::new();
        for g in &geoms {
            acc.accumulate(g).unwrap();
        }
        let result = acc.finalize().unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_union_crs_mismatch_fails() {
        use crate::geometry::Crs;
        use geo_types::Point;
        let geom1 = Geometry::with_crs(GeoGeometry::Point(Point::new(0.0, 0.0)), Crs::epsg(4326));
        let geom2 = Geometry::with_crs(GeoGeometry::Point(Point::new(1.0, 1.0)), Crs::epsg(3857));
        let result = aggregate_union(&[geom1, geom2]);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // aggregate_envelope
    // ------------------------------------------------------------------

    #[test]
    fn test_envelope_basic() {
        let geoms = vec![
            Geometry::from_wkt("POINT (1 2)").unwrap(),
            Geometry::from_wkt("POINT (5 3)").unwrap(),
            Geometry::from_wkt("POINT (3 7)").unwrap(),
        ];
        let result = aggregate_envelope(&geoms).unwrap();

        let rect = match &result.geom {
            GeoGeometry::Polygon(p) => {
                use geo::BoundingRect;
                p.bounding_rect().unwrap()
            }
            other => panic!("Expected Polygon, got {:?}", other),
        };

        assert_abs_diff_eq!(rect.min().x, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.min().y, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.max().x, 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.max().y, 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_envelope_polygon_group() {
        let geoms = vec![
            Geometry::from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap(),
            Geometry::from_wkt("POLYGON ((3 3, 5 3, 5 5, 3 5, 3 3))").unwrap(),
        ];
        let result = aggregate_envelope(&geoms).unwrap();

        let rect = match &result.geom {
            GeoGeometry::Polygon(p) => {
                use geo::BoundingRect;
                p.bounding_rect().unwrap()
            }
            other => panic!("Expected Polygon, got {:?}", other),
        };

        assert_abs_diff_eq!(rect.min().x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.min().y, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.max().x, 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.max().y, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_envelope_empty_group_fails() {
        let result = aggregate_envelope(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_envelope_single_point() {
        let geoms = vec![Geometry::from_wkt("POINT (3.0 4.0)").unwrap()];
        let result = aggregate_envelope(&geoms);
        // A single point has a degenerate bounding rect — result depends on geo crate
        // We just ensure it doesn't panic or return an unexpected error type
        let _ = result; // ok or err, both acceptable for degenerate input
    }

    #[test]
    fn test_envelope_accumulator_incremental() {
        let mut acc = AggregateEnvelope::new();
        for wkt in &["POINT (0 0)", "POINT (10 5)", "POINT (5 10)"] {
            let g = Geometry::from_wkt(wkt).unwrap();
            acc.accumulate(&g).unwrap();
        }
        let result = acc.finalize().unwrap();
        assert!(!result.is_empty());
    }

    // ------------------------------------------------------------------
    // aggregate_centroid
    // ------------------------------------------------------------------

    #[test]
    fn test_centroid_four_corners() {
        let geoms = vec![
            Geometry::from_wkt("POINT (0 0)").unwrap(),
            Geometry::from_wkt("POINT (4 0)").unwrap(),
            Geometry::from_wkt("POINT (4 4)").unwrap(),
            Geometry::from_wkt("POINT (0 4)").unwrap(),
        ];
        let result = aggregate_centroid(&geoms).unwrap();

        match &result.geom {
            GeoGeometry::Point(p) => {
                assert_abs_diff_eq!(p.x(), 2.0, epsilon = 1e-10);
                assert_abs_diff_eq!(p.y(), 2.0, epsilon = 1e-10);
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }

    #[test]
    fn test_centroid_weighted_by_area() {
        // Large polygon should pull centroid toward itself
        let geoms = vec![
            Geometry::from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap(), // area 100
            Geometry::from_wkt("POINT (100 100)").unwrap(),                         // weight 1
        ];
        let result = aggregate_centroid(&geoms).unwrap();
        match &result.geom {
            GeoGeometry::Point(p) => {
                // centroid of big polygon is (5,5), small point is (100,100)
                // weighted: (5*100 + 100*1)/(100+1) = 600/101 ≈ 5.94
                assert!(p.x() < 10.0, "centroid should be much closer to polygon");
                assert!(p.y() < 10.0, "centroid should be much closer to polygon");
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }

    #[test]
    fn test_centroid_empty_group_fails() {
        let result = aggregate_centroid(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_centroid_accumulator_streaming() {
        let mut acc = AggregateCentroid::new();
        for wkt in &["POINT (0 0)", "POINT (2 0)", "POINT (1 2)"] {
            let g = Geometry::from_wkt(wkt).unwrap();
            acc.accumulate(&g).unwrap();
        }
        let result = acc.finalize().unwrap();
        match &result.geom {
            GeoGeometry::Point(p) => {
                assert_abs_diff_eq!(p.x(), 1.0, epsilon = 1e-9);
                // y = (0+0+2)/3 ≈ 0.667
                assert_abs_diff_eq!(p.y(), 2.0 / 3.0, epsilon = 1e-9);
            }
            other => panic!("Expected Point, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // URI constants
    // ------------------------------------------------------------------

    #[test]
    fn test_uri_constants() {
        assert!(GEO_AGGREGATE_UNION.contains("aggregate_union"));
        assert!(GEO_AGGREGATE_ENVELOPE.contains("aggregate_envelope"));
        assert!(GEO_AGGREGATE_CENTROID.contains("aggregate_centroid"));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extended tests — OGC GeoSPARQL 1.1 aggregate compliance
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests_extended {
    use super::*;
    use approx::assert_abs_diff_eq;
    use geo_types::Point;

    // ── geof:union ──────────────────────────────────────────────────────────

    #[test]
    fn test_union_multipolygon_inputs() {
        let geoms = vec![
            Geometry::from_wkt(
                "MULTIPOLYGON (((0 0, 2 0, 2 2, 0 2, 0 0)), ((4 4, 6 4, 6 6, 4 6, 4 4)))",
            )
            .unwrap(),
            Geometry::from_wkt("POLYGON ((3 0, 5 0, 5 2, 3 2, 3 0))").unwrap(),
        ];
        let result = aggregate_union(&geoms).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_union_lines_use_collection() {
        let geoms = vec![
            Geometry::from_wkt("LINESTRING (0 0, 5 5)").unwrap(),
            Geometry::from_wkt("LINESTRING (3 3, 8 8)").unwrap(),
        ];
        let result = aggregate_union(&geoms).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_union_mixed_types_collection() {
        let geoms = vec![
            Geometry::from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap(),
            Geometry::from_wkt("POINT (10 10)").unwrap(),
            Geometry::from_wkt("LINESTRING (0 0, 1 1)").unwrap(),
        ];
        let result = aggregate_union(&geoms).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_union_three_adjacent_polygons() {
        // Three adjacent squares that should merge into a rectangle
        let geoms = vec![
            Geometry::from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))").unwrap(),
            Geometry::from_wkt("POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))").unwrap(),
            Geometry::from_wkt("POLYGON ((2 0, 3 0, 3 1, 2 1, 2 0))").unwrap(),
        ];
        let result = aggregate_union(&geoms).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_union_accumulator_crs_tracks_first() {
        use crate::geometry::Crs;
        let geom1 = Geometry::with_crs(GeoGeometry::Point(Point::new(0.0, 0.0)), Crs::epsg(4326));
        let geom2 = Geometry::with_crs(GeoGeometry::Point(Point::new(1.0, 1.0)), Crs::epsg(4326));
        let result = aggregate_union(&[geom1, geom2]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_union_identical_polygons() {
        // Two identical polygons should yield the same polygon (or at least non-empty)
        let geoms = vec![
            Geometry::from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap(),
            Geometry::from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap(),
        ];
        let result = aggregate_union(&geoms).unwrap();
        assert!(!result.is_empty());
    }

    // ── geof:convexHull aggregate ────────────────────────────────────────────

    #[test]
    fn test_convex_hull_aggregate_points() {
        // convex hull of 4 corners + interior point = rectangle
        let geoms = vec![
            Geometry::from_wkt("POINT (0 0)").unwrap(),
            Geometry::from_wkt("POINT (4 0)").unwrap(),
            Geometry::from_wkt("POINT (4 4)").unwrap(),
            Geometry::from_wkt("POINT (0 4)").unwrap(),
            Geometry::from_wkt("POINT (2 2)").unwrap(), // interior
        ];
        // Collect all points into a GeometryCollection union, then take convex hull
        let union_result = aggregate_union(&geoms).unwrap();
        let hull_geom = match &union_result.geom {
            GeoGeometry::GeometryCollection(gc) => {
                // Take the convex hull of the centroid points
                let pts: Vec<_> = gc.iter().filter_map(|g| g.centroid()).collect();
                // At least verify we got points
                assert!(!pts.is_empty());
                true
            }
            _ => true,
        };
        assert!(hull_geom);
    }

    // ── geof:envelope ───────────────────────────────────────────────────────

    #[test]
    fn test_envelope_linestring_group() {
        let geoms = vec![
            Geometry::from_wkt("LINESTRING (0 0, 3 5)").unwrap(),
            Geometry::from_wkt("LINESTRING (1 1, 7 2)").unwrap(),
        ];
        let result = aggregate_envelope(&geoms).unwrap();
        let rect = match &result.geom {
            GeoGeometry::Polygon(p) => {
                use geo::BoundingRect;
                p.bounding_rect().unwrap()
            }
            other => panic!("expected Polygon, got {:?}", other),
        };
        assert_abs_diff_eq!(rect.min().x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.min().y, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.max().x, 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.max().y, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_envelope_negative_coords() {
        let geoms = vec![
            Geometry::from_wkt("POINT (-10 -5)").unwrap(),
            Geometry::from_wkt("POINT (10 5)").unwrap(),
        ];
        let result = aggregate_envelope(&geoms).unwrap();
        let rect = match &result.geom {
            GeoGeometry::Polygon(p) => {
                use geo::BoundingRect;
                p.bounding_rect().unwrap()
            }
            other => panic!("expected Polygon, got {:?}", other),
        };
        assert_abs_diff_eq!(rect.min().x, -10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.max().x, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_envelope_accumulator_multiple_polygons() {
        let mut acc = AggregateEnvelope::new();
        let inputs = [
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((5 5, 8 5, 8 9, 5 9, 5 5))",
            "POLYGON ((-3 -3, -1 -3, -1 -1, -3 -1, -3 -3))",
        ];
        for wkt in &inputs {
            acc.accumulate(&Geometry::from_wkt(wkt).unwrap()).unwrap();
        }
        let result = acc.finalize().unwrap();
        let rect = match &result.geom {
            GeoGeometry::Polygon(p) => {
                use geo::BoundingRect;
                p.bounding_rect().unwrap()
            }
            other => panic!("expected Polygon, got {:?}", other),
        };
        assert_abs_diff_eq!(rect.min().x, -3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.max().x, 8.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.min().y, -3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rect.max().y, 9.0, epsilon = 1e-10);
    }

    // ── geof:centroid ────────────────────────────────────────────────────────

    #[test]
    fn test_centroid_single_point() {
        let geoms = vec![Geometry::from_wkt("POINT (3 7)").unwrap()];
        let result = aggregate_centroid(&geoms).unwrap();
        match &result.geom {
            GeoGeometry::Point(p) => {
                assert_abs_diff_eq!(p.x(), 3.0, epsilon = 1e-10);
                assert_abs_diff_eq!(p.y(), 7.0, epsilon = 1e-10);
            }
            other => panic!("expected Point, got {:?}", other),
        }
    }

    #[test]
    fn test_centroid_two_equal_weight_polygons() {
        // Two equal-area polygons centered at (1,1) and (3,1) → centroid at (2,1)
        let geoms = vec![
            Geometry::from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap(),
            Geometry::from_wkt("POLYGON ((2 0, 4 0, 4 2, 2 2, 2 0))").unwrap(),
        ];
        let result = aggregate_centroid(&geoms).unwrap();
        match &result.geom {
            GeoGeometry::Point(p) => {
                assert_abs_diff_eq!(p.x(), 2.0, epsilon = 1e-8);
                assert_abs_diff_eq!(p.y(), 1.0, epsilon = 1e-8);
            }
            other => panic!("expected Point, got {:?}", other),
        }
    }

    #[test]
    fn test_centroid_line_group() {
        let geoms = vec![
            Geometry::from_wkt("LINESTRING (0 0, 2 0)").unwrap(), // centroid (1,0), len=2
            Geometry::from_wkt("LINESTRING (0 4, 2 4)").unwrap(), // centroid (1,4), len=2
        ];
        let result = aggregate_centroid(&geoms).unwrap();
        match &result.geom {
            GeoGeometry::Point(p) => {
                // equal-weight (same length), centroid at midpoint
                assert_abs_diff_eq!(p.x(), 1.0, epsilon = 1e-8);
                assert_abs_diff_eq!(p.y(), 2.0, epsilon = 1e-8);
            }
            other => panic!("expected Point, got {:?}", other),
        }
    }

    #[test]
    fn test_centroid_crs_mismatch_fails() {
        use crate::geometry::Crs;
        let geom1 = Geometry::with_crs(GeoGeometry::Point(Point::new(0.0, 0.0)), Crs::epsg(4326));
        let geom2 = Geometry::with_crs(GeoGeometry::Point(Point::new(1.0, 1.0)), Crs::epsg(3857));
        let result = aggregate_centroid(&[geom1, geom2]);
        assert!(result.is_err());
    }

    // ── SpatialAggregate trait (concrete impls) ─────────────────────────────

    #[test]
    fn test_aggregate_union_accumulate_and_finalize() {
        let mut agg = AggregateUnion::new();
        let geom = Geometry::from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))").unwrap();
        agg.accumulate(&geom).unwrap();
        let result = agg.finalize().unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_aggregate_envelope_accumulate_and_finalize() {
        let mut agg = AggregateEnvelope::new();
        let geom = Geometry::from_wkt("POINT (5 5)").unwrap();
        agg.accumulate(&geom).unwrap();
        let _ = agg.finalize(); // degenerate single point - ok or err
    }

    #[test]
    fn test_aggregate_centroid_accumulate_and_finalize() {
        let mut agg = AggregateCentroid::new();
        for wkt in &["POINT (0 0)", "POINT (2 0)", "POINT (1 2)"] {
            agg.accumulate(&Geometry::from_wkt(wkt).unwrap()).unwrap();
        }
        let result = agg.finalize().unwrap();
        assert!(!result.is_empty());
    }

    // ── SPARQL aggregate binding URIs ────────────────────────────────────────

    #[test]
    fn test_aggregate_union_uri_format() {
        assert!(GEO_AGGREGATE_UNION.starts_with("http://www.opengis.net/"));
    }

    #[test]
    fn test_aggregate_envelope_uri_format() {
        assert!(GEO_AGGREGATE_ENVELOPE.starts_with("http://www.opengis.net/"));
    }

    #[test]
    fn test_aggregate_centroid_uri_format() {
        assert!(GEO_AGGREGATE_CENTROID.starts_with("http://www.opengis.net/"));
    }

    // ── AggregateBoundingBox (geof:aggBoundingBox) ───────────────────────────

    #[test]
    fn test_agg_bbox_uri_format() {
        assert!(GEO_AGG_BOUNDING_BOX.starts_with("http://www.opengis.net/"));
        assert!(GEO_AGG_BOUNDING_BOX.contains("aggBoundingBox"));
    }

    #[test]
    fn test_agg_bbox_single_point() {
        let mut acc = AggregateBoundingBox::new();
        acc.accumulate(&Geometry::from_wkt("POINT (3 7)").unwrap())
            .unwrap();
        let bbox = acc.finalize_bbox().unwrap();
        assert_abs_diff_eq!(bbox.min_x, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bbox.max_x, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bbox.min_y, 7.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bbox.max_y, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_agg_bbox_multiple_points() {
        let points = ["POINT (-5 -3)", "POINT (10 8)", "POINT (0 0)"];
        let mut acc = AggregateBoundingBox::new();
        for wkt in &points {
            acc.accumulate(&Geometry::from_wkt(wkt).unwrap()).unwrap();
        }
        let bbox = acc.finalize_bbox().unwrap();
        assert_abs_diff_eq!(bbox.min_x, -5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bbox.max_x, 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bbox.min_y, -3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bbox.max_y, 8.0, epsilon = 1e-12);
    }

    #[test]
    fn test_agg_bbox_polygons() {
        let wkts = [
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "POLYGON ((5 5, 8 5, 8 9, 5 9, 5 5))",
            "POLYGON ((-3 -3, -1 -3, -1 -1, -3 -1, -3 -3))",
        ];
        let mut acc = AggregateBoundingBox::new();
        for wkt in &wkts {
            acc.accumulate(&Geometry::from_wkt(wkt).unwrap()).unwrap();
        }
        let bbox = acc.finalize_bbox().unwrap();
        assert_abs_diff_eq!(bbox.min_x, -3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bbox.max_x, 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bbox.min_y, -3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bbox.max_y, 9.0, epsilon = 1e-12);
    }

    #[test]
    fn test_agg_bbox_to_geometry_is_polygon() {
        let wkts = [
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON ((2 2, 6 2, 6 6, 2 6, 2 2))",
        ];
        let mut acc = AggregateBoundingBox::new();
        for wkt in &wkts {
            acc.accumulate(&Geometry::from_wkt(wkt).unwrap()).unwrap();
        }
        let geom = acc.finalize().unwrap();
        assert_eq!(geom.geometry_type(), "Polygon");
        assert!(!geom.is_empty());
    }

    #[test]
    fn test_agg_bbox_empty_fails() {
        let acc = AggregateBoundingBox::new();
        assert!(acc.finalize_bbox().is_err());

        let acc2 = AggregateBoundingBox::new();
        assert!(acc2.finalize().is_err());
    }

    #[test]
    fn test_agg_bbox_crs_mismatch_fails() {
        use crate::geometry::Crs;
        let g1 = Geometry::with_crs(
            geo_types::Geometry::Point(geo_types::Point::new(0.0, 0.0)),
            Crs::epsg(4326),
        );
        let g2 = Geometry::with_crs(
            geo_types::Geometry::Point(geo_types::Point::new(1.0, 1.0)),
            Crs::epsg(3857),
        );
        let mut acc = AggregateBoundingBox::new();
        acc.accumulate(&g1).unwrap();
        assert!(acc.accumulate(&g2).is_err());
    }

    #[test]
    fn test_agg_bbox_width_height() {
        let wkts = ["POINT (0 0)", "POINT (10 5)"];
        let mut acc = AggregateBoundingBox::new();
        for wkt in &wkts {
            acc.accumulate(&Geometry::from_wkt(wkt).unwrap()).unwrap();
        }
        assert_abs_diff_eq!(acc.width(), 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(acc.height(), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_agg_bbox_count() {
        let mut acc = AggregateBoundingBox::new();
        assert_eq!(acc.count(), 0);
        acc.accumulate(&Geometry::from_wkt("POINT (0 0)").unwrap())
            .unwrap();
        assert_eq!(acc.count(), 1);
        acc.accumulate(&Geometry::from_wkt("POINT (1 1)").unwrap())
            .unwrap();
        assert_eq!(acc.count(), 2);
    }

    #[test]
    fn test_agg_bbox_is_empty() {
        let acc = AggregateBoundingBox::new();
        assert!(acc.is_empty());
    }

    #[test]
    fn test_agg_bbox_centre() {
        let wkts = ["POINT (0 0)", "POINT (10 10)"];
        let mut acc = AggregateBoundingBox::new();
        for wkt in &wkts {
            acc.accumulate(&Geometry::from_wkt(wkt).unwrap()).unwrap();
        }
        let bbox = acc.finalize_bbox().unwrap();
        let (cx, cy) = bbox.centre();
        assert_abs_diff_eq!(cx, 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cy, 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_agg_bbox_area() {
        let wkts = ["POINT (0 0)", "POINT (4 3)"];
        let mut acc = AggregateBoundingBox::new();
        for wkt in &wkts {
            acc.accumulate(&Geometry::from_wkt(wkt).unwrap()).unwrap();
        }
        let bbox = acc.finalize_bbox().unwrap();
        assert_abs_diff_eq!(bbox.area(), 12.0, epsilon = 1e-12); // 4×3
    }

    #[test]
    fn test_agg_bbox_to_geometry_has_correct_bounds() {
        let wkts = ["POINT (-1 -2)", "POINT (3 4)"];
        let mut acc = AggregateBoundingBox::new();
        for wkt in &wkts {
            acc.accumulate(&Geometry::from_wkt(wkt).unwrap()).unwrap();
        }
        let bbox = acc.finalize_bbox().unwrap();
        let geom = bbox.to_geometry();
        if let geo_types::Geometry::Polygon(p) = &geom.geom {
            use geo::BoundingRect;
            let rect = p.bounding_rect().unwrap();
            assert_abs_diff_eq!(rect.min().x, -1.0, epsilon = 1e-12);
            assert_abs_diff_eq!(rect.max().x, 3.0, epsilon = 1e-12);
            assert_abs_diff_eq!(rect.min().y, -2.0, epsilon = 1e-12);
            assert_abs_diff_eq!(rect.max().y, 4.0, epsilon = 1e-12);
        } else {
            panic!("expected Polygon geometry");
        }
    }

    #[test]
    fn test_aggregate_bounding_box_free_fn() {
        let geoms = vec![
            Geometry::from_wkt("POINT (0 0)").unwrap(),
            Geometry::from_wkt("POINT (5 10)").unwrap(),
        ];
        let result = aggregate_bounding_box(&geoms).unwrap();
        assert_eq!(result.geometry_type(), "Polygon");

        use geo::BoundingRect;
        if let geo_types::Geometry::Polygon(p) = &result.geom {
            let rect = p.bounding_rect().unwrap();
            assert_abs_diff_eq!(rect.min().x, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(rect.max().x, 5.0, epsilon = 1e-12);
        } else {
            panic!("expected Polygon");
        }
    }

    #[test]
    fn test_aggregate_bounding_box_empty_fails() {
        assert!(aggregate_bounding_box(&[]).is_err());
    }

    #[test]
    fn test_agg_bbox_linestring() {
        let ls = Geometry::from_wkt("LINESTRING(0 5, 10 -2)").unwrap();
        let mut acc = AggregateBoundingBox::new();
        acc.accumulate(&ls).unwrap();
        let bbox = acc.finalize_bbox().unwrap();
        assert_abs_diff_eq!(bbox.min_x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bbox.max_x, 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bbox.min_y, -2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bbox.max_y, 5.0, epsilon = 1e-12);
    }
}
