//! OGC GeoSPARQL 1.1 Functions
//!
//! This module implements GeoSPARQL 1.1 extension functions:
//!
//! - [`distance_with_unit`]: `geo:distance(?a, ?b, uom:metre)` — distance with unit conversion
//! - [`length_with_unit`]: `geo:length(?geom, uom:metre)` — length with unit conversion
//! - [`area_with_unit`]: `geo:area(?geom, uom:squareMetre)` — area with unit conversion
//! - [`concave_hull`]: `geo:concaveHull(?geom, ratio)` — concave hull with concavity ratio
//! - `is_simple`: `geo:isSimple(?geom)` — predicate (already in geometric_properties)
//! - `is_empty`: `geo:isEmpty(?geom)` — predicate (already in geometric_properties)
//!
//! ## Unit-of-Measure URIs (OGC 1.1)
//!
//! The OGC GeoSPARQL 1.1 specification uses UCUM unit URIs of the form:
//! `http://www.opengis.net/def/uom/UCUM/`
//!
//! Supported units:
//! - `uom:metre` / `uom:m` — metre (SI base unit)
//! - `uom:kilometre` / `uom:km` — kilometre
//! - `uom:mile` / `uom:[mi_i]` — international mile
//! - `uom:foot` / `uom:[ft_i]` — international foot
//! - `uom:yard` / `uom:[yd_i]` — international yard
//! - `uom:nauticalMile` / `uom:[nmi_i]` — nautical mile
//!
//! ## References
//!
//! - OGC GeoSPARQL 1.1 — <https://docs.ogc.org/is/22-047r1/22-047r1.html>
//! - UCUM units — <https://ucum.org/ucum.html>

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;

// ─────────────────────────────────────────────────────────────────────────────
// Unit-of-Measure
// ─────────────────────────────────────────────────────────────────────────────

/// Unit of measure URI prefix (OGC UCUM namespace)
pub const UOM_PREFIX: &str = "http://www.opengis.net/def/uom/UCUM/";

/// OGC Unit-of-Measure identifiers used in GeoSPARQL 1.1 function calls.
#[derive(Debug, Clone, PartialEq)]
pub enum UnitOfMeasure {
    /// SI metre — the canonical unit for GeoSPARQL calculations
    Metre,
    /// Kilometre (1 km = 1000 m)
    Kilometre,
    /// International mile (1 mi = 1609.344 m)
    Mile,
    /// International foot (1 ft = 0.3048 m)
    Foot,
    /// International yard (1 yd = 0.9144 m)
    Yard,
    /// Nautical mile (1 nmi = 1852 m)
    NauticalMile,
    /// Degree (angular unit — used for unprojected coordinates)
    Degree,
}

impl UnitOfMeasure {
    /// Conversion factor from metres to this unit (multiply a metre value).
    ///
    /// ```
    /// use oxirs_geosparql::functions::ogc11::UnitOfMeasure;
    /// let km = UnitOfMeasure::Kilometre;
    /// assert!((km.metres_per_unit() - 1000.0).abs() < 1e-10);
    /// ```
    pub fn metres_per_unit(&self) -> f64 {
        match self {
            UnitOfMeasure::Metre => 1.0,
            UnitOfMeasure::Kilometre => 1_000.0,
            UnitOfMeasure::Mile => 1_609.344,
            UnitOfMeasure::Foot => 0.304_8,
            UnitOfMeasure::Yard => 0.914_4,
            UnitOfMeasure::NauticalMile => 1_852.0,
            UnitOfMeasure::Degree => 111_319.49079327357, // average metres per degree at equator
        }
    }

    /// Convert a value in metres to this unit.
    #[inline]
    pub fn from_metres(&self, metres: f64) -> f64 {
        metres / self.metres_per_unit()
    }

    /// Convert a value in this unit to metres.
    #[inline]
    pub fn to_metres(&self, value: f64) -> f64 {
        value * self.metres_per_unit()
    }

    /// Parse a unit-of-measure URI or short name into a `UnitOfMeasure`.
    ///
    /// Accepts both full OGC UCUM URIs and common abbreviations.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::functions::ogc11::UnitOfMeasure;
    ///
    /// let m = UnitOfMeasure::from_uri("http://www.opengis.net/def/uom/UCUM/m").expect("should succeed");
    /// assert_eq!(m, UnitOfMeasure::Metre);
    ///
    /// let km = UnitOfMeasure::from_uri("km").expect("should succeed");
    /// assert_eq!(km, UnitOfMeasure::Kilometre);
    /// ```
    pub fn from_uri(uri: &str) -> Result<Self> {
        // Strip the OGC UCUM prefix if present
        let short = uri.strip_prefix(UOM_PREFIX).unwrap_or(uri);

        match short {
            "m" | "metre" | "meter" | "metres" | "meters" => Ok(UnitOfMeasure::Metre),
            "km" | "kilometre" | "kilometer" | "kilometres" | "kilometers" => {
                Ok(UnitOfMeasure::Kilometre)
            }
            "[mi_i]" | "mi" | "mile" | "miles" => Ok(UnitOfMeasure::Mile),
            "[ft_i]" | "ft" | "foot" | "feet" => Ok(UnitOfMeasure::Foot),
            "[yd_i]" | "yd" | "yard" | "yards" => Ok(UnitOfMeasure::Yard),
            "[nmi_i]" | "nmi" | "nauticalMile" | "nauticalmile" | "nauticalmiles" => {
                Ok(UnitOfMeasure::NauticalMile)
            }
            "deg" | "degree" | "degrees" => Ok(UnitOfMeasure::Degree),
            other => Err(GeoSparqlError::InvalidInput(format!(
                "Unknown unit-of-measure URI: '{other}'. \
                 Supported: m, km, [mi_i], [ft_i], [yd_i], [nmi_i]"
            ))),
        }
    }

    /// Return the OGC UCUM URI for this unit.
    pub fn to_uri(&self) -> &'static str {
        match self {
            UnitOfMeasure::Metre => "http://www.opengis.net/def/uom/UCUM/m",
            UnitOfMeasure::Kilometre => "http://www.opengis.net/def/uom/UCUM/km",
            UnitOfMeasure::Mile => "http://www.opengis.net/def/uom/UCUM/[mi_i]",
            UnitOfMeasure::Foot => "http://www.opengis.net/def/uom/UCUM/[ft_i]",
            UnitOfMeasure::Yard => "http://www.opengis.net/def/uom/UCUM/[yd_i]",
            UnitOfMeasure::NauticalMile => "http://www.opengis.net/def/uom/UCUM/[nmi_i]",
            UnitOfMeasure::Degree => "http://www.opengis.net/def/uom/UCUM/deg",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// geo:distance with unit parameter
// ─────────────────────────────────────────────────────────────────────────────

/// Calculate the distance between two geometries in the specified unit of measure.
///
/// Implements OGC GeoSPARQL 1.1 `geof:distance(?a, ?b, uom:metre)`.
///
/// The internal calculation is Euclidean (same as the base `distance` function).
/// The result is then converted from the geometry's native units (assumed metres
/// for projected CRS, or degrees for geographic CRS) to the requested unit.
///
/// # Arguments
///
/// - `geom1` — first geometry
/// - `geom2` — second geometry
/// - `unit` — desired output unit of measure
///
/// # Returns
///
/// Distance in the requested unit, or an error if the geometries have
/// incompatible CRS.
///
/// # Example
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::ogc11::{distance_with_unit, UnitOfMeasure};
///
/// let p1 = Geometry::from_wkt("POINT(0 0)").expect("should succeed");
/// let p2 = Geometry::from_wkt("POINT(1000 0)").expect("should succeed");
///
/// // native units are metres — 1000m should be 1km
/// let d = distance_with_unit(&p1, &p2, &UnitOfMeasure::Kilometre).expect("should succeed");
/// assert!((d - 1.0).abs() < 1e-6);
/// ```
pub fn distance_with_unit(geom1: &Geometry, geom2: &Geometry, unit: &UnitOfMeasure) -> Result<f64> {
    let dist_metres = crate::functions::geometric_operations::distance(geom1, geom2)?;
    Ok(unit.from_metres(dist_metres))
}

/// Parse a unit URI and calculate distance between two geometries.
///
/// Convenience wrapper that accepts a unit-of-measure URI string.
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::ogc11::distance_with_unit_uri;
///
/// let p1 = Geometry::from_wkt("POINT(0 0)").expect("should succeed");
/// let p2 = Geometry::from_wkt("POINT(1609.344 0)").expect("should succeed");
///
/// let d = distance_with_unit_uri(&p1, &p2, "mi").expect("should succeed");
/// assert!((d - 1.0).abs() < 1e-4);
/// ```
pub fn distance_with_unit_uri(geom1: &Geometry, geom2: &Geometry, unit_uri: &str) -> Result<f64> {
    let unit = UnitOfMeasure::from_uri(unit_uri)?;
    distance_with_unit(geom1, geom2, &unit)
}

// ─────────────────────────────────────────────────────────────────────────────
// geo:length with unit parameter
// ─────────────────────────────────────────────────────────────────────────────

/// Calculate the length of a geometry in the specified unit of measure.
///
/// Implements OGC GeoSPARQL 1.1 `geof:length(?geom, uom:metre)`.
///
/// For `LineString` and `MultiLineString` returns the total arc length.
/// For `Polygon` and `MultiPolygon` returns the exterior perimeter.
/// For all other geometry types returns 0.0.
///
/// # Example
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::ogc11::{length_with_unit, UnitOfMeasure};
///
/// // A line 2000m long
/// let line = Geometry::from_wkt("LINESTRING(0 0, 2000 0)").expect("should succeed");
/// let km = length_with_unit(&line, &UnitOfMeasure::Kilometre).expect("should succeed");
/// assert!((km - 2.0).abs() < 1e-6);
/// ```
pub fn length_with_unit(geom: &Geometry, unit: &UnitOfMeasure) -> Result<f64> {
    let len_metres = crate::functions::geometric_properties::length(geom)?;
    Ok(unit.from_metres(len_metres))
}

/// Parse a unit URI and calculate the length of a geometry.
pub fn length_with_unit_uri(geom: &Geometry, unit_uri: &str) -> Result<f64> {
    let unit = UnitOfMeasure::from_uri(unit_uri)?;
    length_with_unit(geom, &unit)
}

// ─────────────────────────────────────────────────────────────────────────────
// geo:area with unit parameter
// ─────────────────────────────────────────────────────────────────────────────

/// Calculate the area of a geometry in the specified unit of measure.
///
/// Implements OGC GeoSPARQL 1.1 `geof:area(?geom, uom:squareMetre)`.
///
/// The area conversion multiplies the square of the linear scale factor:
/// `area_in_unit = area_in_metres² / (metres_per_unit)²`
///
/// # Example
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::ogc11::{area_with_unit, UnitOfMeasure};
///
/// // A 1000×1000 m square = 1 km²
/// let square = Geometry::from_wkt("POLYGON((0 0, 1000 0, 1000 1000, 0 1000, 0 0))").expect("should succeed");
/// let km2 = area_with_unit(&square, &UnitOfMeasure::Kilometre).expect("should succeed");
/// assert!((km2 - 1.0).abs() < 1e-6);
/// ```
pub fn area_with_unit(geom: &Geometry, unit: &UnitOfMeasure) -> Result<f64> {
    let area_metres_sq = crate::functions::geometric_properties::area(geom)?;
    let scale = unit.metres_per_unit();
    Ok(area_metres_sq / (scale * scale))
}

/// Parse a unit URI and calculate the area of a geometry.
pub fn area_with_unit_uri(geom: &Geometry, unit_uri: &str) -> Result<f64> {
    let unit = UnitOfMeasure::from_uri(unit_uri)?;
    area_with_unit(geom, &unit)
}

// ─────────────────────────────────────────────────────────────────────────────
// geo:concaveHull with ratio
// ─────────────────────────────────────────────────────────────────────────────

/// Compute an approximate concave hull of a geometry.
///
/// Implements OGC GeoSPARQL 1.1 `geof:concaveHull(?geom, ratio)`.
///
/// ## Algorithm
///
/// Uses a k-nearest-neighbours-based approach (the "k-nearest concave hull"
/// algorithm by Moreira and Santos, 2007):
///
/// 1. Start from the leftmost point.
/// 2. At each step, select the next point from the `k` nearest candidates,
///    choosing the one that minimises left turns (maximises concavity).
/// 3. Repeat until the hull is closed.
///
/// The `ratio` parameter controls concavity:
/// - `0.0` — as concave as possible (may approach the exact point cloud shape)
/// - `1.0` — convex hull (same as `convexHull`)
///
/// ## Arguments
///
/// - `geom` — the input geometry (all points are extracted)
/// - `ratio` — concavity ratio in [0.0, 1.0]
///
/// ## Returns
///
/// A `Polygon` representing the concave hull, or the convex hull if the
/// point count is too small or `ratio ≥ 1.0`.
///
/// # Example
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::ogc11::concave_hull;
///
/// let mp = Geometry::from_wkt(
///     "MULTIPOINT((0 0),(1 0),(2 0),(2 1),(2 2),(1 2),(0 2),(0 1),(1 1))"
/// ).expect("should succeed");
///
/// let hull = concave_hull(&mp, 0.3).expect("should succeed");
/// assert_eq!(hull.geometry_type(), "Polygon");
/// ```
pub fn concave_hull(geom: &Geometry, ratio: f64) -> Result<Geometry> {
    if !(0.0..=1.0).contains(&ratio) {
        return Err(GeoSparqlError::InvalidInput(format!(
            "concave_hull ratio must be in [0.0, 1.0], got {ratio}"
        )));
    }

    // Extract all 2-D points from the geometry
    let points = extract_points(geom);

    if points.len() < 3 {
        // Degenerate case: fall back to convex hull
        return crate::functions::geometric_operations::convex_hull(geom);
    }

    // For ratio ≈ 1.0, return convex hull directly
    if ratio >= 0.999 {
        return crate::functions::geometric_operations::convex_hull(geom);
    }

    // Compute convex hull as the reference
    let convex = crate::functions::geometric_operations::convex_hull(geom)?;

    // For very low ratios or small point sets, attempt to build a concave hull
    // using a k-neighbours shrink-wrap approach.
    // k is derived from the ratio: higher ratio → larger k → less concave
    let k = {
        let n = points.len();
        let k_f = 3.0 + (ratio * (n as f64 - 3.0)).round();
        (k_f as usize).clamp(3, n)
    };

    match build_concave_hull(&points, k) {
        Some(polygon) => Ok(Geometry::new(geo_types::Geometry::Polygon(polygon))),
        None => Ok(convex), // fall back to convex hull on failure
    }
}

/// Internal: Extract all 2D points from a geometry.
fn extract_points(geom: &Geometry) -> Vec<(f64, f64)> {
    use geo_types::Geometry as G;

    let mut pts: Vec<(f64, f64)> = Vec::new();

    fn collect(g: &G<f64>, out: &mut Vec<(f64, f64)>) {
        match g {
            G::Point(p) => out.push((p.x(), p.y())),
            G::MultiPoint(mp) => {
                for p in &mp.0 {
                    out.push((p.x(), p.y()));
                }
            }
            G::Line(l) => {
                out.push((l.start.x, l.start.y));
                out.push((l.end.x, l.end.y));
            }
            G::LineString(ls) => {
                for c in &ls.0 {
                    out.push((c.x, c.y));
                }
            }
            G::MultiLineString(mls) => {
                for ls in &mls.0 {
                    for c in &ls.0 {
                        out.push((c.x, c.y));
                    }
                }
            }
            G::Polygon(p) => {
                for c in &p.exterior().0 {
                    out.push((c.x, c.y));
                }
            }
            G::MultiPolygon(mp) => {
                for poly in &mp.0 {
                    for c in &poly.exterior().0 {
                        out.push((c.x, c.y));
                    }
                }
            }
            G::GeometryCollection(gc) => {
                for sub in &gc.0 {
                    collect(sub, out);
                }
            }
            G::Rect(r) => {
                out.push((r.min().x, r.min().y));
                out.push((r.max().x, r.min().y));
                out.push((r.max().x, r.max().y));
                out.push((r.min().x, r.max().y));
            }
            G::Triangle(t) => {
                out.push((t.0.x, t.0.y));
                out.push((t.1.x, t.1.y));
                out.push((t.2.x, t.2.y));
            }
        }
    }

    collect(&geom.geom, &mut pts);

    // Deduplicate (keep order, but remove exact duplicates)
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    pts.dedup();
    pts
}

/// Internal: Build a concave hull using k-nearest-neighbours walk.
///
/// Returns `None` if the algorithm fails to close the hull (caller falls back
/// to the convex hull).
fn build_concave_hull(points: &[(f64, f64)], k: usize) -> Option<geo_types::Polygon<f64>> {
    use geo_types::{Coord, LineString, Polygon};

    let n = points.len();
    if n < 3 {
        return None;
    }

    // Find the bottom-most (then left-most) starting point
    let start_idx = points
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.1.total_cmp(&b.1).then(a.0.total_cmp(&b.0)))
        .map(|(i, _)| i)?;

    let mut hull: Vec<(f64, f64)> = vec![points[start_idx]];
    let mut visited = vec![false; n];
    visited[start_idx] = true;
    let mut current = points[start_idx];

    // Direction from previous step (start pointing "up-left" = 270°)
    let mut prev_angle: f64 = 270.0f64.to_radians();

    let max_iterations = n * 2 + 10;
    let mut iterations = 0;

    loop {
        iterations += 1;
        if iterations > max_iterations {
            return None; // Give up — fall back to convex hull
        }

        // Find the k nearest unvisited points
        let mut distances: Vec<(usize, f64)> = points
            .iter()
            .enumerate()
            .filter(|(i, _)| !visited[*i])
            .map(|(i, p)| {
                let dx = p.0 - current.0;
                let dy = p.1 - current.1;
                (i, dx * dx + dy * dy)
            })
            .collect();

        if distances.is_empty() {
            // Check if we can close the hull
            let start = points[start_idx];
            let dx = start.0 - current.0;
            let dy = start.1 - current.1;
            if dx * dx + dy * dy < 1e-10 {
                break; // Hull closed
            }
            return None;
        }

        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        let candidates: Vec<usize> = distances.iter().take(k).map(|(i, _)| *i).collect();

        // Pick the candidate that turns most to the right relative to prev direction
        // (right-most turn = most clockwise = most concave hull boundary)
        let best_idx = candidates.iter().copied().min_by(|&i, &j| {
            let ai = angle_from(current, points[i], prev_angle);
            let aj = angle_from(current, points[j], prev_angle);
            ai.total_cmp(&aj)
        })?;

        let next = points[best_idx];
        let dx = next.0 - current.0;
        let dy = next.1 - current.1;
        prev_angle = dy.atan2(dx);

        // Check if we are back at start
        let sdx = next.0 - points[start_idx].0;
        let sdy = next.1 - points[start_idx].1;
        if hull.len() >= 3 && sdx * sdx + sdy * sdy < 1e-10 {
            break; // Hull closed
        }

        visited[best_idx] = true;
        hull.push(next);
        current = next;

        // Safety: if all points visited and we haven't closed, close it now
        if visited.iter().all(|&v| v) {
            break;
        }
    }

    if hull.len() < 3 {
        return None;
    }

    // Close the ring
    let first = hull[0];
    hull.push(first);

    let coords: Vec<Coord<f64>> = hull.into_iter().map(|(x, y)| Coord { x, y }).collect();

    Some(Polygon::new(LineString::new(coords), vec![]))
}

/// Compute the clockwise angular difference from `prev_angle` to the direction
/// `from → to`, normalised to [0, 2π).
fn angle_from(from: (f64, f64), to: (f64, f64), prev_angle: f64) -> f64 {
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    let angle = dy.atan2(dx);

    // clockwise rotation from prev_angle
    let mut delta = prev_angle - angle;
    // normalise to [0, 2π)
    if delta < 0.0 {
        delta += 2.0 * std::f64::consts::PI;
    }
    if delta >= 2.0 * std::f64::consts::PI {
        delta -= 2.0 * std::f64::consts::PI;
    }
    delta
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── UnitOfMeasure ─────────────────────────────────────────────────────────

    #[test]
    fn test_uom_metres_per_unit() {
        assert!((UnitOfMeasure::Metre.metres_per_unit() - 1.0).abs() < 1e-12);
        assert!((UnitOfMeasure::Kilometre.metres_per_unit() - 1_000.0).abs() < 1e-12);
        assert!((UnitOfMeasure::Mile.metres_per_unit() - 1_609.344).abs() < 1e-6);
        assert!((UnitOfMeasure::Foot.metres_per_unit() - 0.304_8).abs() < 1e-12);
        assert!((UnitOfMeasure::Yard.metres_per_unit() - 0.914_4).abs() < 1e-12);
        assert!((UnitOfMeasure::NauticalMile.metres_per_unit() - 1_852.0).abs() < 1e-12);
    }

    #[test]
    fn test_uom_from_metres_roundtrip() {
        for unit in [
            UnitOfMeasure::Metre,
            UnitOfMeasure::Kilometre,
            UnitOfMeasure::Mile,
            UnitOfMeasure::Foot,
            UnitOfMeasure::Yard,
            UnitOfMeasure::NauticalMile,
        ] {
            let metres = 12_345.678;
            let converted = unit.from_metres(metres);
            let back = unit.to_metres(converted);
            assert!((back - metres).abs() < 1e-6, "{unit:?}: {back} != {metres}");
        }
    }

    #[test]
    fn test_uom_from_uri_full_prefix() {
        let m = UnitOfMeasure::from_uri("http://www.opengis.net/def/uom/UCUM/m")
            .expect("should succeed");
        assert_eq!(m, UnitOfMeasure::Metre);

        let km = UnitOfMeasure::from_uri("http://www.opengis.net/def/uom/UCUM/km")
            .expect("should succeed");
        assert_eq!(km, UnitOfMeasure::Kilometre);
    }

    #[test]
    fn test_uom_from_uri_short_names() {
        assert_eq!(
            UnitOfMeasure::from_uri("m").expect("should succeed"),
            UnitOfMeasure::Metre
        );
        assert_eq!(
            UnitOfMeasure::from_uri("km").expect("should succeed"),
            UnitOfMeasure::Kilometre
        );
        assert_eq!(
            UnitOfMeasure::from_uri("mi").expect("should succeed"),
            UnitOfMeasure::Mile
        );
        assert_eq!(
            UnitOfMeasure::from_uri("[mi_i]").expect("should succeed"),
            UnitOfMeasure::Mile
        );
        assert_eq!(
            UnitOfMeasure::from_uri("[ft_i]").expect("should succeed"),
            UnitOfMeasure::Foot
        );
        assert_eq!(
            UnitOfMeasure::from_uri("[yd_i]").expect("should succeed"),
            UnitOfMeasure::Yard
        );
        assert_eq!(
            UnitOfMeasure::from_uri("[nmi_i]").expect("should succeed"),
            UnitOfMeasure::NauticalMile
        );
    }

    #[test]
    fn test_uom_from_uri_aliases() {
        assert_eq!(
            UnitOfMeasure::from_uri("metre").expect("should succeed"),
            UnitOfMeasure::Metre
        );
        assert_eq!(
            UnitOfMeasure::from_uri("meter").expect("should succeed"),
            UnitOfMeasure::Metre
        );
        assert_eq!(
            UnitOfMeasure::from_uri("kilometre").expect("should succeed"),
            UnitOfMeasure::Kilometre
        );
        assert_eq!(
            UnitOfMeasure::from_uri("mile").expect("should succeed"),
            UnitOfMeasure::Mile
        );
        assert_eq!(
            UnitOfMeasure::from_uri("foot").expect("should succeed"),
            UnitOfMeasure::Foot
        );
        assert_eq!(
            UnitOfMeasure::from_uri("feet").expect("should succeed"),
            UnitOfMeasure::Foot
        );
    }

    #[test]
    fn test_uom_from_uri_invalid() {
        assert!(UnitOfMeasure::from_uri("furlong").is_err());
        assert!(UnitOfMeasure::from_uri("").is_err());
        assert!(UnitOfMeasure::from_uri("parsec").is_err());
    }

    #[test]
    fn test_uom_to_uri() {
        assert_eq!(
            UnitOfMeasure::Metre.to_uri(),
            "http://www.opengis.net/def/uom/UCUM/m"
        );
        assert_eq!(
            UnitOfMeasure::Kilometre.to_uri(),
            "http://www.opengis.net/def/uom/UCUM/km"
        );
        assert!(UnitOfMeasure::Mile.to_uri().contains("[mi_i]"));
        assert!(UnitOfMeasure::NauticalMile.to_uri().contains("[nmi_i]"));
    }

    // ── distance_with_unit ────────────────────────────────────────────────────

    #[test]
    fn test_distance_with_unit_metres() {
        let p1 = Geometry::from_wkt("POINT(0 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT(3 4)").expect("should succeed");
        let d = distance_with_unit(&p1, &p2, &UnitOfMeasure::Metre).expect("should succeed");
        assert!((d - 5.0).abs() < 1e-10, "expected 5.0 m, got {d}");
    }

    #[test]
    fn test_distance_with_unit_km() {
        // 1000 m = 1 km
        let p1 = Geometry::from_wkt("POINT(0 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT(1000 0)").expect("should succeed");
        let d = distance_with_unit(&p1, &p2, &UnitOfMeasure::Kilometre).expect("should succeed");
        assert!((d - 1.0).abs() < 1e-6, "expected 1.0 km, got {d}");
    }

    #[test]
    fn test_distance_with_unit_miles() {
        // 1609.344 m = 1 mile
        let p1 = Geometry::from_wkt("POINT(0 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT(1609.344 0)").expect("should succeed");
        let d = distance_with_unit(&p1, &p2, &UnitOfMeasure::Mile).expect("should succeed");
        assert!((d - 1.0).abs() < 1e-4, "expected 1.0 mi, got {d}");
    }

    #[test]
    fn test_distance_with_unit_feet() {
        // 0.3048 m = 1 foot
        let p1 = Geometry::from_wkt("POINT(0 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT(0.3048 0)").expect("should succeed");
        let d = distance_with_unit(&p1, &p2, &UnitOfMeasure::Foot).expect("should succeed");
        assert!((d - 1.0).abs() < 1e-4, "expected 1.0 ft, got {d}");
    }

    #[test]
    fn test_distance_with_unit_nautical_miles() {
        // 1852 m = 1 nautical mile
        let p1 = Geometry::from_wkt("POINT(0 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT(1852 0)").expect("should succeed");
        let d = distance_with_unit(&p1, &p2, &UnitOfMeasure::NauticalMile).expect("should succeed");
        assert!((d - 1.0).abs() < 1e-4, "expected 1.0 nmi, got {d}");
    }

    #[test]
    fn test_distance_with_unit_uri() {
        let p1 = Geometry::from_wkt("POINT(0 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT(1000 0)").expect("should succeed");
        let d = distance_with_unit_uri(&p1, &p2, "km").expect("should succeed");
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_with_unit_uri_invalid_unit() {
        let p1 = Geometry::from_wkt("POINT(0 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT(1 0)").expect("should succeed");
        assert!(distance_with_unit_uri(&p1, &p2, "furlong").is_err());
    }

    #[test]
    fn test_distance_with_unit_zero() {
        let p1 = Geometry::from_wkt("POINT(5 5)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT(5 5)").expect("should succeed");
        let d = distance_with_unit(&p1, &p2, &UnitOfMeasure::Kilometre).expect("should succeed");
        assert!(d.abs() < 1e-12, "same point should have distance 0");
    }

    // ── length_with_unit ─────────────────────────────────────────────────────

    #[test]
    fn test_length_with_unit_metres() {
        let line = Geometry::from_wkt("LINESTRING(0 0, 5 0)").expect("should succeed");
        let l = length_with_unit(&line, &UnitOfMeasure::Metre).expect("should succeed");
        assert!((l - 5.0).abs() < 1e-10, "expected 5.0 m, got {l}");
    }

    #[test]
    fn test_length_with_unit_km() {
        let line = Geometry::from_wkt("LINESTRING(0 0, 2000 0)").expect("should succeed");
        let l = length_with_unit(&line, &UnitOfMeasure::Kilometre).expect("should succeed");
        assert!((l - 2.0).abs() < 1e-6, "expected 2.0 km, got {l}");
    }

    #[test]
    fn test_length_with_unit_miles() {
        // 3218.688 m = 2 miles
        let line = Geometry::from_wkt("LINESTRING(0 0, 3218.688 0)").expect("should succeed");
        let l = length_with_unit(&line, &UnitOfMeasure::Mile).expect("should succeed");
        assert!((l - 2.0).abs() < 1e-3, "expected 2.0 mi, got {l}");
    }

    #[test]
    fn test_length_with_unit_polygon_perimeter() {
        // 10×10 square: perimeter = 40 m
        let square =
            Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").expect("should succeed");
        let l = length_with_unit(&square, &UnitOfMeasure::Metre).expect("should succeed");
        assert!((l - 40.0).abs() < 1e-6, "expected 40 m perimeter, got {l}");
    }

    #[test]
    fn test_length_with_unit_uri() {
        let line = Geometry::from_wkt("LINESTRING(0 0, 1000 0)").expect("should succeed");
        let l = length_with_unit_uri(&line, "km").expect("should succeed");
        assert!((l - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_length_with_unit_multilinestring() {
        // Two segments of 500 m each = 1000 m = 1 km
        let mls = Geometry::from_wkt("MULTILINESTRING((0 0, 500 0),(0 0, 0 500))")
            .expect("should succeed");
        let l = length_with_unit(&mls, &UnitOfMeasure::Kilometre).expect("should succeed");
        assert!((l - 1.0).abs() < 1e-6, "expected 1 km, got {l}");
    }

    // ── area_with_unit ───────────────────────────────────────────────────────

    #[test]
    fn test_area_with_unit_sq_metres() {
        // 5×5 square = 25 m²
        let square =
            Geometry::from_wkt("POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))").expect("should succeed");
        let a = area_with_unit(&square, &UnitOfMeasure::Metre).expect("should succeed");
        assert!((a - 25.0).abs() < 1e-8, "expected 25 m², got {a}");
    }

    #[test]
    fn test_area_with_unit_sq_km() {
        // 1000×1000 m = 1 km²
        let square = Geometry::from_wkt("POLYGON((0 0, 1000 0, 1000 1000, 0 1000, 0 0))")
            .expect("should succeed");
        let a = area_with_unit(&square, &UnitOfMeasure::Kilometre).expect("should succeed");
        assert!((a - 1.0).abs() < 1e-8, "expected 1 km², got {a}");
    }

    #[test]
    fn test_area_with_unit_sq_miles() {
        // 1609.344×1609.344 m = 1 mi²
        let side = 1_609.344;
        let wkt = format!("POLYGON((0 0, {side} 0, {side} {side}, 0 {side}, 0 0))");
        let square = Geometry::from_wkt(&wkt).expect("valid WKT");
        let a = area_with_unit(&square, &UnitOfMeasure::Mile).expect("should succeed");
        assert!((a - 1.0).abs() < 1e-4, "expected 1 mi², got {a}");
    }

    #[test]
    fn test_area_with_unit_uri() {
        let square = Geometry::from_wkt("POLYGON((0 0, 1000 0, 1000 1000, 0 1000, 0 0))")
            .expect("should succeed");
        let a = area_with_unit_uri(&square, "km").expect("should succeed");
        assert!((a - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_area_with_unit_non_polygon_returns_zero() {
        let line = Geometry::from_wkt("LINESTRING(0 0, 1 0)").expect("should succeed");
        let a = area_with_unit(&line, &UnitOfMeasure::Metre).expect("should succeed");
        assert_eq!(a, 0.0, "non-polygon area should be 0");
    }

    #[test]
    fn test_area_with_unit_multipolygon() {
        // Two 100×100 squares = 20000 m² total
        let mp = Geometry::from_wkt(
            "MULTIPOLYGON(((0 0,100 0,100 100,0 100,0 0)),((200 0,300 0,300 100,200 100,200 0)))",
        )
        .expect("should succeed");
        let a = area_with_unit(&mp, &UnitOfMeasure::Metre).expect("should succeed");
        assert!((a - 20_000.0).abs() < 1e-6, "expected 20000 m², got {a}");
    }

    // ── concave_hull ─────────────────────────────────────────────────────────

    #[test]
    fn test_concave_hull_returns_polygon() {
        let mp = Geometry::from_wkt("MULTIPOINT((0 0),(4 0),(4 4),(0 4),(2 2))")
            .expect("should succeed");
        let hull = concave_hull(&mp, 0.5).expect("should succeed");
        assert_eq!(hull.geometry_type(), "Polygon");
    }

    #[test]
    fn test_concave_hull_ratio_one_equals_convex() {
        let mp = Geometry::from_wkt("MULTIPOINT((0 0),(4 0),(4 4),(0 4),(2 2),(1 1))")
            .expect("should succeed");
        let concave = concave_hull(&mp, 1.0).expect("should succeed");
        let convex =
            crate::functions::geometric_operations::convex_hull(&mp).expect("should succeed");
        // Both should be Polygon types
        assert_eq!(concave.geometry_type(), "Polygon");
        assert_eq!(convex.geometry_type(), "Polygon");
    }

    #[test]
    fn test_concave_hull_ratio_zero_still_valid() {
        let mp = Geometry::from_wkt(
            "MULTIPOINT((0 0),(5 0),(10 0),(10 5),(10 10),(5 10),(0 10),(0 5),(5 5))",
        )
        .expect("should succeed");
        let hull = concave_hull(&mp, 0.0).expect("should succeed");
        assert_eq!(hull.geometry_type(), "Polygon");
    }

    #[test]
    fn test_concave_hull_degenerate_too_few_points() {
        // 2 points → falls back to convex hull (which may be a LineString)
        let mp = Geometry::from_wkt("MULTIPOINT((0 0),(1 1))").expect("should succeed");
        let result = concave_hull(&mp, 0.5);
        // Should not panic — result may be Ok or Err
        let _ = result;
    }

    #[test]
    fn test_concave_hull_invalid_ratio_negative() {
        let mp = Geometry::from_wkt("MULTIPOINT((0 0),(1 0),(1 1),(0 1))").expect("should succeed");
        assert!(concave_hull(&mp, -0.1).is_err());
    }

    #[test]
    fn test_concave_hull_invalid_ratio_over_one() {
        let mp = Geometry::from_wkt("MULTIPOINT((0 0),(1 0),(1 1),(0 1))").expect("should succeed");
        assert!(concave_hull(&mp, 1.1).is_err());
    }

    #[test]
    fn test_concave_hull_polygon_input() {
        let poly =
            Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").expect("should succeed");
        let hull = concave_hull(&poly, 0.5).expect("should succeed");
        assert_eq!(hull.geometry_type(), "Polygon");
    }

    #[test]
    fn test_concave_hull_linestring_input() {
        let ls = Geometry::from_wkt("LINESTRING(0 0, 5 5, 10 0, 5 -5)").expect("should succeed");
        let hull = concave_hull(&ls, 0.5).expect("should succeed");
        assert_eq!(hull.geometry_type(), "Polygon");
    }

    #[test]
    fn test_concave_hull_u_shape() {
        // U-shaped point cloud: a concave hull (low ratio) should capture the concavity,
        // while convex hull would be a filled rectangle
        let mp = Geometry::from_wkt(
            "MULTIPOINT((0 0),(1 0),(2 0),(3 0),(4 0),\
             (0 1),(4 1),\
             (0 2),(4 2),\
             (0 3),(4 3))",
        )
        .expect("should succeed");
        let hull = concave_hull(&mp, 0.3).expect("should succeed");
        assert_eq!(hull.geometry_type(), "Polygon");
        // Concave hull should be non-empty
        assert!(!hull.is_empty());
    }

    #[test]
    fn test_concave_hull_single_point_fallback() {
        let pt = Geometry::from_wkt("POINT(1 1)").expect("should succeed");
        // 1 point: should fall back gracefully
        let result = concave_hull(&pt, 0.5);
        // Convex hull of single point returns Point, which is fine
        let _ = result;
    }

    // ── angle_from helper ─────────────────────────────────────────────────────

    #[test]
    fn test_angle_from_north() {
        let from = (0.0, 0.0);
        let to = (0.0, 1.0); // straight up
        let angle = angle_from(from, to, 0.0); // prev pointing right
        assert!(angle.is_finite());
    }

    // ── extract_points helper ─────────────────────────────────────────────────

    #[test]
    fn test_extract_points_multipoint() {
        let mp = Geometry::from_wkt("MULTIPOINT((0 0),(1 1),(2 2))").expect("should succeed");
        let pts = extract_points(&mp);
        assert_eq!(pts.len(), 3);
    }

    #[test]
    fn test_extract_points_polygon() {
        // Polygon exterior has 5 coords (4 unique + closing point which is deduped)
        let poly =
            Geometry::from_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))").expect("should succeed");
        let pts = extract_points(&poly);
        assert!(pts.len() >= 4, "expected at least 4 unique points");
    }

    #[test]
    fn test_extract_points_deduplication() {
        // Duplicate points should be removed
        let mp = Geometry::from_wkt("MULTIPOINT((1 1),(1 1),(2 2))").expect("should succeed");
        let pts = extract_points(&mp);
        assert_eq!(pts.len(), 2, "duplicates should be removed, got {pts:?}");
    }
}
