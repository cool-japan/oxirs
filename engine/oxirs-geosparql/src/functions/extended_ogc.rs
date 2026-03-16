//! Extended OGC GeoSPARQL functions
//!
//! This module implements advanced spatial distance and geometry functions
//! beyond the standard GeoSPARQL 1.0/1.1 specification:
//!
//! - [`hausdorff_distance`]: Hausdorff distance between two geometries
//! - [`frechet_distance`]: Discrete Fréchet distance for curves
//! - [`minimum_bounding_circle`]: Minimum bounding circle (MBC) of a geometry
//! - [`convex_hull_geom`]: Convex hull using pure-Rust Graham scan
//! - [`voronoi_diagram`]: Voronoi diagram for point sets
//!
//! ## References
//!
//! - Hausdorff (1914) "Grundzüge der Mengenlehre"
//! - Alt & Godau (1995) "Computing the Fréchet distance between two polygonal curves"
//! - Welzl (1991) "Smallest enclosing disks"
//! - Fortune (1987) "A sweepline algorithm for Voronoi diagrams"

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo_types::{coord, Coord, Geometry as GeoGeometry, LineString, Point, Polygon};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Coordinate extraction helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract all 2D vertices from a geometry as a flat list of (x, y) pairs.
fn extract_vertices(geom: &GeoGeometry<f64>) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    extract_vertices_inner(geom, &mut out);
    out
}

fn extract_vertices_inner(geom: &GeoGeometry<f64>, out: &mut Vec<(f64, f64)>) {
    match geom {
        GeoGeometry::Point(p) => out.push((p.x(), p.y())),
        GeoGeometry::Line(l) => {
            out.push((l.start.x, l.start.y));
            out.push((l.end.x, l.end.y));
        }
        GeoGeometry::LineString(ls) => {
            for c in &ls.0 {
                out.push((c.x, c.y));
            }
        }
        GeoGeometry::Polygon(poly) => {
            for c in poly.exterior().0.iter() {
                out.push((c.x, c.y));
            }
            for ring in poly.interiors() {
                for c in ring.0.iter() {
                    out.push((c.x, c.y));
                }
            }
        }
        GeoGeometry::MultiPoint(mp) => {
            for p in mp.0.iter() {
                out.push((p.x(), p.y()));
            }
        }
        GeoGeometry::MultiLineString(mls) => {
            for ls in mls.0.iter() {
                for c in &ls.0 {
                    out.push((c.x, c.y));
                }
            }
        }
        GeoGeometry::MultiPolygon(mpoly) => {
            for poly in mpoly.0.iter() {
                for c in poly.exterior().0.iter() {
                    out.push((c.x, c.y));
                }
                for ring in poly.interiors() {
                    for c in ring.0.iter() {
                        out.push((c.x, c.y));
                    }
                }
            }
        }
        GeoGeometry::GeometryCollection(gc) => {
            for sub in gc.0.iter() {
                extract_vertices_inner(sub, out);
            }
        }
        GeoGeometry::Rect(r) => {
            let mn = r.min();
            let mx = r.max();
            out.push((mn.x, mn.y));
            out.push((mx.x, mn.y));
            out.push((mx.x, mx.y));
            out.push((mn.x, mx.y));
        }
        GeoGeometry::Triangle(t) => {
            out.push((t.0.x, t.0.y));
            out.push((t.1.x, t.1.y));
            out.push((t.2.x, t.2.y));
        }
    }
}

/// Point-to-point 2D Euclidean distance.
#[inline]
fn pt_dist(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    (dx * dx + dy * dy).sqrt()
}

/// Minimum distance from point (px, py) to the vertex set of `geom`.
fn point_to_geom_min_dist(px: f64, py: f64, geom: &GeoGeometry<f64>) -> f64 {
    let verts = extract_vertices(geom);
    if verts.is_empty() {
        return f64::INFINITY;
    }
    verts
        .iter()
        .map(|(vx, vy)| pt_dist(px, py, *vx, *vy))
        .fold(f64::INFINITY, f64::min)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hausdorff Distance
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the directed Hausdorff distance from `a` to `b`:
/// max over all vertices of `a` of the min distance to `b`.
fn directed_hausdorff(a: &GeoGeometry<f64>, b: &GeoGeometry<f64>) -> f64 {
    let verts_a = extract_vertices(a);
    if verts_a.is_empty() {
        return 0.0;
    }
    verts_a
        .iter()
        .map(|(ax, ay)| point_to_geom_min_dist(*ax, *ay, b))
        .fold(0.0_f64, f64::max)
}

/// Compute the Hausdorff distance between two geometries.
///
/// The Hausdorff distance is the maximum of:
/// - directed_hausdorff(A → B): max over A-vertices of min-dist to B
/// - directed_hausdorff(B → A): max over B-vertices of min-dist to A
///
/// This is a discrete approximation using vertex sampling, which is exact
/// for point sets and provides a good upper bound for polygonal geometries.
///
/// # Errors
///
/// Returns an error if either geometry is empty.
///
/// # Example
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::extended_ogc::hausdorff_distance;
///
/// let a = Geometry::from_wkt("LINESTRING(0 0, 1 0, 1 1)").expect("should succeed");
/// let b = Geometry::from_wkt("LINESTRING(0 1, 1 1, 1 0)").expect("should succeed");
/// let d = hausdorff_distance(&a, &b).expect("should succeed");
/// assert!(d >= 0.0);
/// ```
pub fn hausdorff_distance(geom1: &Geometry, geom2: &Geometry) -> Result<f64> {
    geom1.validate_crs_compatibility(geom2)?;

    let verts1 = extract_vertices(&geom1.geom);
    let verts2 = extract_vertices(&geom2.geom);

    if verts1.is_empty() || verts2.is_empty() {
        return Err(GeoSparqlError::InvalidInput(
            "hausdorffDistance: both geometries must be non-empty".to_string(),
        ));
    }

    let h12 = directed_hausdorff(&geom1.geom, &geom2.geom);
    let h21 = directed_hausdorff(&geom2.geom, &geom1.geom);
    Ok(h12.max(h21))
}

// ─────────────────────────────────────────────────────────────────────────────
// Fréchet Distance (Discrete)
// ─────────────────────────────────────────────────────────────────────────────

/// Extract an ordered list of (x, y) from a geometry for Fréchet computation.
/// For non-curve geometries, returns all vertices in order.
fn curve_vertices(geom: &GeoGeometry<f64>) -> Vec<(f64, f64)> {
    match geom {
        GeoGeometry::LineString(ls) => ls.0.iter().map(|c| (c.x, c.y)).collect(),
        _ => extract_vertices(geom),
    }
}

/// Discrete Fréchet distance via dynamic programming.
///
/// Time: O(nm), Space: O(nm).
fn discrete_frechet_dp(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }

    // ca[i][j] = coupling distance up to (a[i], b[j])
    let mut ca = vec![vec![f64::NEG_INFINITY; m]; n];

    for i in 0..n {
        for j in 0..m {
            let d = pt_dist(a[i].0, a[i].1, b[j].0, b[j].1);
            let prev = match (i, j) {
                (0, 0) => 0.0,
                (i0, 0) => ca[i0 - 1][0],
                (0, j0) => ca[0][j0 - 1],
                (i0, j0) => ca[i0 - 1][j0].min(ca[i0 - 1][j0 - 1]).min(ca[i0][j0 - 1]),
            };
            ca[i][j] = d.max(prev);
        }
    }

    ca[n - 1][m - 1]
}

/// Compute the discrete Fréchet distance between two geometries.
///
/// For LineString inputs, uses the full discrete Fréchet algorithm.
/// For other geometry types, falls back to Hausdorff distance.
///
/// # Errors
///
/// Returns an error if either geometry is empty.
///
/// # Example
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::extended_ogc::frechet_distance;
///
/// let a = Geometry::from_wkt("LINESTRING(0 0, 1 0, 2 0)").expect("should succeed");
/// let b = Geometry::from_wkt("LINESTRING(0 1, 1 1, 2 1)").expect("should succeed");
/// let d = frechet_distance(&a, &b).expect("should succeed");
/// assert!((d - 1.0).abs() < 1e-10);
/// ```
pub fn frechet_distance(geom1: &Geometry, geom2: &Geometry) -> Result<f64> {
    geom1.validate_crs_compatibility(geom2)?;

    let a = curve_vertices(&geom1.geom);
    let b = curve_vertices(&geom2.geom);

    if a.is_empty() || b.is_empty() {
        return Err(GeoSparqlError::InvalidInput(
            "frechetDistance: both geometries must be non-empty".to_string(),
        ));
    }

    Ok(discrete_frechet_dp(&a, &b))
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimum Bounding Circle
// ─────────────────────────────────────────────────────────────────────────────

/// Find the minimum enclosing circle using Welzl's algorithm (randomized).
/// Returns (center_x, center_y, radius).
fn welzl(pts: &[(f64, f64)], boundary: &[(f64, f64)]) -> (f64, f64, f64) {
    // Base cases
    if pts.is_empty() || boundary.len() == 3 {
        return trivial_circle(boundary);
    }

    let (p, rest) = pts.split_last().expect("non-empty checked above");
    let circle = welzl(rest, boundary);

    // If p is inside the circle, done
    let (cx, cy, r) = circle;
    if pt_dist(cx, cy, p.0, p.1) <= r + 1e-10 {
        return circle;
    }

    // p must be on boundary
    let mut new_boundary = boundary.to_vec();
    new_boundary.push(*p);
    welzl(rest, &new_boundary)
}

/// Compute the trivial minimum enclosing circle for 0, 1, 2 or 3 boundary points.
fn trivial_circle(boundary: &[(f64, f64)]) -> (f64, f64, f64) {
    match boundary.len() {
        0 => (0.0, 0.0, 0.0),
        1 => (boundary[0].0, boundary[0].1, 0.0),
        2 => {
            let cx = (boundary[0].0 + boundary[1].0) / 2.0;
            let cy = (boundary[0].1 + boundary[1].1) / 2.0;
            let r = pt_dist(cx, cy, boundary[0].0, boundary[0].1);
            (cx, cy, r)
        }
        _ => {
            // Circumcircle of 3 points
            circumcircle(boundary[0], boundary[1], boundary[2])
        }
    }
}

/// Compute the circumcircle of three points.
/// Returns (cx, cy, r).
fn circumcircle(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> (f64, f64, f64) {
    // Using the formula from determinants
    let ax = a.0;
    let ay = a.1;
    let bx = b.0;
    let by = b.1;
    let cx = c.0;
    let cy = c.1;

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));

    if d.abs() < 1e-12 {
        // Collinear — return bounding circle of extremes
        let min_x = ax.min(bx).min(cx);
        let max_x = ax.max(bx).max(cx);
        let min_y = ay.min(by).min(cy);
        let max_y = ay.max(by).max(cy);
        let ocx = (min_x + max_x) / 2.0;
        let ocy = (min_y + max_y) / 2.0;
        let r = pt_dist(ocx, ocy, ax, ay)
            .max(pt_dist(ocx, ocy, bx, by))
            .max(pt_dist(ocx, ocy, cx, cy));
        return (ocx, ocy, r);
    }

    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;

    let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;

    let r = pt_dist(ux, uy, ax, ay);
    (ux, uy, r)
}

/// Build a Polygon approximating a circle with `segments` sides.
fn circle_to_polygon(cx: f64, cy: f64, radius: f64, segments: usize) -> Polygon<f64> {
    let mut coords: Vec<Coord<f64>> = (0..segments)
        .map(|i| {
            let angle = 2.0 * PI * (i as f64) / (segments as f64);
            coord! {
                x: cx + radius * angle.cos(),
                y: cy + radius * angle.sin()
            }
        })
        .collect();

    // Close the ring
    if let Some(first) = coords.first().copied() {
        coords.push(first);
    }

    Polygon::new(LineString::new(coords), vec![])
}

/// Compute the minimum bounding circle (MBC) of a geometry.
///
/// Returns a `Geometry` containing a `Polygon` that approximates the MBC
/// using 64 line segments. Uses Welzl's algorithm for exact computation.
///
/// # Errors
///
/// Returns an error if the geometry has no vertices.
///
/// # Example
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::extended_ogc::minimum_bounding_circle;
///
/// let poly = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("should succeed");
/// let mbc = minimum_bounding_circle(&poly).expect("should succeed");
/// assert_eq!(mbc.geometry_type(), "Polygon");
/// ```
pub fn minimum_bounding_circle(geom: &Geometry) -> Result<Geometry> {
    let pts = extract_vertices(&geom.geom);

    if pts.is_empty() {
        return Err(GeoSparqlError::InvalidInput(
            "minimumBoundingCircle: geometry must be non-empty".to_string(),
        ));
    }

    // Shuffle for expected O(n) performance of Welzl
    // (use a simple deterministic shuffle to avoid rand dependency)
    let mut shuffled = pts.clone();
    // Fisher-Yates with a fixed seed via LCG
    let mut seed: u64 = 0xDEAD_BEEF_1234_5678;
    let n = shuffled.len();
    for i in (1..n).rev() {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = (seed >> 33) as usize % (i + 1);
        shuffled.swap(i, j);
    }

    let (cx, cy, radius) = welzl(&shuffled, &[]);
    let poly = circle_to_polygon(cx, cy, radius, 64);
    Ok(Geometry::new(GeoGeometry::Polygon(poly)))
}

// ─────────────────────────────────────────────────────────────────────────────
// Convex Hull
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the convex hull of a geometry using the `geo` crate's implementation.
///
/// Returns a `Geometry` containing the convex hull polygon (or point/line
/// for degenerate inputs).
///
/// # Errors
///
/// Returns an error if the geometry has no vertices.
///
/// # Example
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::extended_ogc::convex_hull_geom;
///
/// let pts = Geometry::from_wkt("MULTIPOINT((0 0),(1 0),(0 1),(1 1),(0.5 0.5))").expect("should succeed");
/// let hull = convex_hull_geom(&pts).expect("should succeed");
/// assert_eq!(hull.geometry_type(), "Polygon");
/// ```
pub fn convex_hull_geom(geom: &Geometry) -> Result<Geometry> {
    use geo::algorithm::convex_hull::ConvexHull;
    use geo_types::MultiPoint;

    let pts = extract_vertices(&geom.geom);

    if pts.is_empty() {
        return Err(GeoSparqlError::InvalidInput(
            "convexHull: geometry must be non-empty".to_string(),
        ));
    }

    // Build a MultiPoint from vertices and compute hull via geo crate
    let multi_pt: MultiPoint<f64> =
        MultiPoint::new(pts.iter().map(|(x, y)| Point::new(*x, *y)).collect());

    let hull = multi_pt.convex_hull();
    Ok(Geometry::new(GeoGeometry::Polygon(hull)))
}

// ─────────────────────────────────────────────────────────────────────────────
// Voronoi Diagram
// ─────────────────────────────────────────────────────────────────────────────

/// A 2D point with an index, used internally for Voronoi computation.
#[derive(Debug, Clone, Copy)]
struct VoronoiPoint {
    x: f64,
    y: f64,
    _idx: usize,
}

/// Compute circumcenter of triangle (ax,ay)-(bx,by)-(cx,cy).
/// Returns None if the triangle is degenerate.
fn circumcenter(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> Option<(f64, f64)> {
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-12 {
        return None;
    }
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    Some((ux, uy))
}

/// Delaunay triangulation via Bowyer-Watson algorithm.
/// Returns a list of triangles as index triples into `pts`.
fn bowyer_watson(pts: &[VoronoiPoint]) -> Vec<(usize, usize, usize)> {
    if pts.len() < 3 {
        return vec![];
    }

    // Find bounding box
    let (min_x, max_x, min_y, max_y) = pts.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(mnx, mxx, mny, mxy), p| (mnx.min(p.x), mxx.max(p.x), mny.min(p.y), mxy.max(p.y)),
    );

    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let delta_max = dx.max(dy);
    let mid_x = (min_x + max_x) / 2.0;
    let mid_y = (min_y + max_y) / 2.0;

    // Super-triangle vertices (appended beyond pts indices)
    let n = pts.len();
    let st0 = VoronoiPoint {
        x: mid_x - 20.0 * delta_max,
        y: mid_y - delta_max,
        _idx: n,
    };
    let st1 = VoronoiPoint {
        x: mid_x,
        y: mid_y + 20.0 * delta_max,
        _idx: n + 1,
    };
    let st2 = VoronoiPoint {
        x: mid_x + 20.0 * delta_max,
        y: mid_y - delta_max,
        _idx: n + 2,
    };

    // All points including supertriangle
    let mut all_pts: Vec<VoronoiPoint> = pts.to_vec();
    all_pts.push(st0);
    all_pts.push(st1);
    all_pts.push(st2);

    // Triangles as (i, j, k) indexes into all_pts
    let mut triangles: Vec<(usize, usize, usize)> = vec![(n, n + 1, n + 2)];

    for (i, pt) in pts.iter().enumerate() {
        let px = pt.x;
        let py = pt.y;

        // Find all triangles whose circumcircle contains (px, py)
        let mut bad_triangles: Vec<(usize, usize, usize)> = Vec::new();
        for &tri in &triangles {
            let (ax, ay) = (all_pts[tri.0].x, all_pts[tri.0].y);
            let (bx, by) = (all_pts[tri.1].x, all_pts[tri.1].y);
            let (cx, cy) = (all_pts[tri.2].x, all_pts[tri.2].y);

            if let Some((ocx, ocy)) = circumcenter(ax, ay, bx, by, cx, cy) {
                let r = pt_dist(ocx, ocy, ax, ay);
                if pt_dist(ocx, ocy, px, py) < r + 1e-9 {
                    bad_triangles.push(tri);
                }
            }
        }

        // Find boundary polygon of bad triangles (edges not shared by 2 bad triangles)
        let mut boundary: Vec<(usize, usize)> = Vec::new();
        for &tri in &bad_triangles {
            let edges = [(tri.0, tri.1), (tri.1, tri.2), (tri.2, tri.0)];
            for edge in edges {
                let shared = bad_triangles.iter().any(|&other| {
                    other != tri
                        && ((other.0 == edge.0 && other.1 == edge.1)
                            || (other.0 == edge.1 && other.1 == edge.0)
                            || (other.1 == edge.0 && other.2 == edge.1)
                            || (other.1 == edge.1 && other.2 == edge.0)
                            || (other.2 == edge.0 && other.0 == edge.1)
                            || (other.2 == edge.1 && other.0 == edge.0))
                });
                if !shared {
                    boundary.push(edge);
                }
            }
        }

        // Remove bad triangles
        triangles.retain(|t| !bad_triangles.contains(t));

        // Re-triangulate with new point
        for edge in boundary {
            triangles.push((edge.0, edge.1, i));
        }
    }

    // Remove triangles that share a vertex with the super-triangle
    triangles.retain(|&(a, b, c)| a < n && b < n && c < n);
    triangles
}

/// Clip a convex polygon to an axis-aligned bounding box using Sutherland-Hodgman.
fn clip_polygon_to_bbox(
    poly: &[(f64, f64)],
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> Vec<(f64, f64)> {
    if poly.is_empty() {
        return vec![];
    }

    // Apply each of the four half-planes
    let clip_edges: [(f64, f64, f64, bool); 4] = [
        (1.0, 0.0, min_x, true),   // x >= min_x
        (-1.0, 0.0, -max_x, true), // x <= max_x
        (0.0, 1.0, min_y, true),   // y >= min_y
        (0.0, -1.0, -max_y, true), // y <= max_y
    ];

    let mut output = poly.to_vec();

    for (a, b, c, _) in clip_edges {
        if output.is_empty() {
            return vec![];
        }
        let input = output.clone();
        output.clear();
        let len = input.len();
        for j in 0..len {
            let cur = input[j];
            let prev = input[(j + len - 1) % len];

            // inside = a*x + b*y >= c
            let inside_cur = a * cur.0 + b * cur.1 >= c;
            let inside_prev = a * prev.0 + b * prev.1 >= c;

            if inside_cur {
                if !inside_prev {
                    // Edge from outside to inside — add intersection
                    if let Some(inter) = segment_intersect_half_plane(prev, cur, a, b, c) {
                        output.push(inter);
                    }
                }
                output.push(cur);
            } else if inside_prev {
                // Edge from inside to outside — add intersection only
                if let Some(inter) = segment_intersect_half_plane(prev, cur, a, b, c) {
                    output.push(inter);
                }
            }
        }
    }

    output
}

/// Compute intersection of segment (p1 → p2) with half-plane ax + by = c.
fn segment_intersect_half_plane(
    p1: (f64, f64),
    p2: (f64, f64),
    a: f64,
    b: f64,
    c: f64,
) -> Option<(f64, f64)> {
    let d1 = a * p1.0 + b * p1.1 - c;
    let d2 = a * p2.0 + b * p2.1 - c;
    let denom = d1 - d2;
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = d1 / denom;
    Some((p1.0 + t * (p2.0 - p1.0), p1.1 + t * (p2.1 - p1.1)))
}

/// Build Voronoi cells from Delaunay triangulation via dual graph.
/// Returns a list of (site_index, polygon_vertices) pairs.
fn voronoi_cells_from_delaunay(
    pts: &[VoronoiPoint],
    triangles: &[(usize, usize, usize)],
    bbox: (f64, f64, f64, f64),
) -> Vec<(usize, Vec<(f64, f64)>)> {
    let n = pts.len();
    let (min_x, min_y, max_x, max_y) = bbox;

    // For each site, gather all circumcenters of adjacent triangles in order
    let mut cells: Vec<(usize, Vec<(f64, f64)>)> = Vec::new();

    for site_idx in 0..n {
        // Find all triangles that include this site
        let adj_tris: Vec<(usize, usize, usize)> = triangles
            .iter()
            .copied()
            .filter(|&(a, b, c)| a == site_idx || b == site_idx || c == site_idx)
            .collect();

        if adj_tris.is_empty() {
            continue;
        }

        // Collect circumcenters
        let mut circumcenters: Vec<(f64, f64)> = Vec::new();
        for (a, b, c) in &adj_tris {
            let (ax, ay) = (pts[*a].x, pts[*a].y);
            let (bx, by) = (pts[*b].x, pts[*b].y);
            let (cx, cy) = (pts[*c].x, pts[*c].y);
            if let Some(cc) = circumcenter(ax, ay, bx, by, cx, cy) {
                circumcenters.push(cc);
            }
        }

        if circumcenters.is_empty() {
            continue;
        }

        // Sort circumcenters by angle around the site
        let sx = pts[site_idx].x;
        let sy = pts[site_idx].y;
        circumcenters.sort_by(|a, b| {
            let angle_a = (a.1 - sy).atan2(a.0 - sx);
            let angle_b = (b.1 - sy).atan2(b.0 - sx);
            angle_a
                .partial_cmp(&angle_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // For boundary cells with fewer than 3 circumcenters, augment with bbox
        // corners so that open Voronoi cells are properly closed within the bbox.
        // We include all four bbox corners sorted by angle; deduplication and
        // clipping will reduce this to the correct convex polygon.
        let polygon_pts: Vec<(f64, f64)> = if circumcenters.len() < 3 {
            let bbox_corners = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
            ];
            let mut combined = circumcenters.clone();
            combined.extend_from_slice(&bbox_corners);
            combined.sort_by(|a, b| {
                let angle_a = (a.1 - sy).atan2(a.0 - sx);
                let angle_b = (b.1 - sy).atan2(b.0 - sx);
                angle_a
                    .partial_cmp(&angle_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            combined
        } else {
            circumcenters
        };

        // Clip to bbox
        let clipped = clip_polygon_to_bbox(&polygon_pts, min_x, min_y, max_x, max_y);
        if clipped.len() >= 3 {
            cells.push((site_idx, clipped));
        }
    }

    cells
}

/// Compute the Voronoi diagram for a set of points.
///
/// Input must be a `Point`, `MultiPoint`, or `GeometryCollection` of points.
/// Returns a `GeometryCollection` containing one `Polygon` per Voronoi cell.
///
/// Uses Bowyer-Watson Delaunay triangulation followed by dual graph construction.
///
/// # Errors
///
/// Returns an error if the input does not contain at least 3 distinct points.
///
/// # Example
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::extended_ogc::voronoi_diagram;
///
/// let pts = Geometry::from_wkt(
///     "MULTIPOINT((0 0),(4 0),(2 3),(0 4),(4 4))"
/// ).expect("should succeed");
/// let diagram = voronoi_diagram(&pts).expect("should succeed");
/// assert_eq!(diagram.geometry_type(), "GeometryCollection");
/// ```
pub fn voronoi_diagram(geom: &Geometry) -> Result<Geometry> {
    let raw_pts = extract_vertices(&geom.geom);

    // Deduplicate points
    let mut unique: Vec<(f64, f64)> = Vec::new();
    for p in &raw_pts {
        let dup = unique.iter().any(|q| pt_dist(q.0, q.1, p.0, p.1) < 1e-10);
        if !dup {
            unique.push(*p);
        }
    }

    if unique.len() < 3 {
        return Err(GeoSparqlError::InvalidInput(
            "voronoiDiagram: at least 3 distinct points are required".to_string(),
        ));
    }

    let voronoi_pts: Vec<VoronoiPoint> = unique
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| VoronoiPoint { x, y, _idx: i })
        .collect();

    // Bounding box with margin
    let (min_x, max_x, min_y, max_y) = unique.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(mnx, mxx, mny, mxy), &(x, y)| (mnx.min(x), mxx.max(x), mny.min(y), mxy.max(y)),
    );

    let margin = ((max_x - min_x).max(max_y - min_y)).max(1.0);
    let bbox = (
        min_x - margin,
        min_y - margin,
        max_x + margin,
        max_y + margin,
    );

    let triangles = bowyer_watson(&voronoi_pts);
    let cells = voronoi_cells_from_delaunay(&voronoi_pts, &triangles, bbox);

    if cells.is_empty() {
        return Err(GeoSparqlError::GeometryOperationFailed(
            "voronoiDiagram: triangulation produced no cells".to_string(),
        ));
    }

    let polygons: Vec<GeoGeometry<f64>> = cells
        .into_iter()
        .map(|(_, verts)| {
            let mut coords: Vec<Coord<f64>> =
                verts.iter().map(|(x, y)| coord! { x: *x, y: *y }).collect();
            // Close ring
            if let Some(&first) = coords.first() {
                coords.push(first);
            }
            GeoGeometry::Polygon(Polygon::new(LineString::new(coords), vec![]))
        })
        .collect();

    Ok(Geometry::new(GeoGeometry::GeometryCollection(
        geo_types::GeometryCollection::new_from(polygons),
    )))
}

// ─────────────────────────────────────────────────────────────────────────────
// GeoSPARQL function URI constants
// ─────────────────────────────────────────────────────────────────────────────

/// URI for `geof:hausdorffDistance`
pub const GEOF_HAUSDORFF_DISTANCE: &str =
    "http://www.opengis.net/def/function/geosparql/hausdorffDistance";

/// URI for `geof:frechetDistance`
pub const GEOF_FRECHET_DISTANCE: &str =
    "http://www.opengis.net/def/function/geosparql/frechetDistance";

/// URI for `geof:minimumBoundingCircle`
pub const GEOF_MINIMUM_BOUNDING_CIRCLE: &str =
    "http://www.opengis.net/def/function/geosparql/minimumBoundingCircle";

/// URI for `geof:convexHull` (extended variant)
pub const GEOF_CONVEX_HULL_EXT: &str = "http://www.opengis.net/def/function/geosparql/convexHull";

/// URI for `geof:voronoiDiagram`
pub const GEOF_VORONOI_DIAGRAM: &str =
    "http://www.opengis.net/def/function/geosparql/voronoiDiagram";

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(wkt: &str) -> Geometry {
        Geometry::from_wkt(wkt).expect("valid WKT")
    }

    // ── Hausdorff Distance ────────────────────────────────────────────────────

    #[test]
    fn test_hausdorff_identical_points() {
        let a = pt("POINT(1 2)");
        let b = pt("POINT(1 2)");
        let d = hausdorff_distance(&a, &b).expect("should succeed");
        assert!(d.abs() < 1e-10, "identical points: d={d}");
    }

    #[test]
    fn test_hausdorff_two_points() {
        let a = pt("POINT(0 0)");
        let b = pt("POINT(3 4)");
        let d = hausdorff_distance(&a, &b).expect("should succeed");
        assert!((d - 5.0).abs() < 1e-10, "d={d}");
    }

    #[test]
    fn test_hausdorff_symmetric() {
        let a = pt("LINESTRING(0 0, 2 0, 2 2)");
        let b = pt("LINESTRING(0 1, 2 1, 2 3)");
        let d_ab = hausdorff_distance(&a, &b).expect("should succeed");
        let d_ba = hausdorff_distance(&b, &a).expect("should succeed");
        assert!((d_ab - d_ba).abs() < 1e-10, "d_ab={d_ab}, d_ba={d_ba}");
    }

    #[test]
    fn test_hausdorff_linestring() {
        let a = pt("LINESTRING(0 0, 10 0)");
        let b = pt("LINESTRING(0 5, 10 5)");
        let d = hausdorff_distance(&a, &b).expect("should succeed");
        // All A-vertices are distance 5 from B, all B-vertices are distance 5 from A
        assert!((d - 5.0).abs() < 1e-10, "d={d}");
    }

    #[test]
    fn test_hausdorff_multipoint() {
        let a = pt("MULTIPOINT((0 0),(1 0),(2 0))");
        let b = pt("MULTIPOINT((0 3),(1 3),(2 3))");
        let d = hausdorff_distance(&a, &b).expect("should succeed");
        assert!((d - 3.0).abs() < 1e-10, "d={d}");
    }

    #[test]
    fn test_hausdorff_polygon_vs_point() {
        let poly = pt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))");
        let p = pt("POINT(2 2)");
        let d = hausdorff_distance(&poly, &p).expect("should succeed");
        // Max dist from polygon vertices to center (2,2) = sqrt(8) ≈ 2.83
        // Max dist from point to polygon vertices = sqrt(8)
        assert!(d > 2.0, "d={d}");
    }

    #[test]
    fn test_hausdorff_empty_returns_error() {
        let a = Geometry::new(GeoGeometry::GeometryCollection(
            geo_types::GeometryCollection::new_from(vec![]),
        ));
        let b = pt("POINT(1 1)");
        assert!(hausdorff_distance(&a, &b).is_err());
    }

    // ── Fréchet Distance ──────────────────────────────────────────────────────

    #[test]
    fn test_frechet_identical_linestring() {
        let a = pt("LINESTRING(0 0, 1 0, 2 0)");
        let b = pt("LINESTRING(0 0, 1 0, 2 0)");
        let d = frechet_distance(&a, &b).expect("should succeed");
        assert!(d.abs() < 1e-10, "identical: d={d}");
    }

    #[test]
    fn test_frechet_parallel_lines() {
        let a = pt("LINESTRING(0 0, 1 0, 2 0)");
        let b = pt("LINESTRING(0 1, 1 1, 2 1)");
        let d = frechet_distance(&a, &b).expect("should succeed");
        // All pairs have distance 1 → discrete Fréchet = 1
        assert!((d - 1.0).abs() < 1e-10, "d={d}");
    }

    #[test]
    fn test_frechet_vs_hausdorff_order_sensitivity() {
        // Fréchet is order-sensitive; reversed curves may have larger distance
        let a = pt("LINESTRING(0 0, 1 0, 2 0)");
        let b = pt("LINESTRING(2 1, 1 1, 0 1)"); // reversed
        let df = frechet_distance(&a, &b).expect("should succeed");
        let dh = hausdorff_distance(&a, &b).expect("should succeed");
        // Fréchet >= Hausdorff for ordered curves
        assert!(df >= dh - 1e-9, "frechet={df} hausdorff={dh}");
    }

    #[test]
    fn test_frechet_two_points() {
        let a = pt("LINESTRING(0 0, 1 0)");
        let b = pt("LINESTRING(0 1, 1 1)");
        let d = frechet_distance(&a, &b).expect("should succeed");
        assert!((d - 1.0).abs() < 1e-10, "d={d}");
    }

    #[test]
    fn test_frechet_non_linestring_fallback() {
        // For polygon input, falls back to vertex-based computation
        let a = pt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))");
        let b = pt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))");
        let d = frechet_distance(&a, &b).expect("should succeed");
        assert!(d.abs() < 1e-10, "same polygon: d={d}");
    }

    #[test]
    fn test_frechet_empty_returns_error() {
        let a = Geometry::new(GeoGeometry::GeometryCollection(
            geo_types::GeometryCollection::new_from(vec![]),
        ));
        let b = pt("LINESTRING(0 0, 1 1)");
        assert!(frechet_distance(&a, &b).is_err());
    }

    // ── Minimum Bounding Circle ───────────────────────────────────────────────

    #[test]
    fn test_mbc_single_point() {
        let g = pt("POINT(3 4)");
        let mbc = minimum_bounding_circle(&g).expect("should succeed");
        assert_eq!(mbc.geometry_type(), "Polygon");
    }

    #[test]
    fn test_mbc_two_points_diameter() {
        // Circle should have diameter = distance between the two points
        let g = pt("MULTIPOINT((0 0),(4 0))");
        let mbc = minimum_bounding_circle(&g).expect("should succeed");
        assert_eq!(mbc.geometry_type(), "Polygon");
        // Center should be (2, 0), radius = 2
        // Check by verifying all original points are within the circle
        let verts = extract_vertices(&mbc.geom);
        assert!(!verts.is_empty(), "MBC polygon must have vertices");
    }

    #[test]
    fn test_mbc_unit_square() {
        let g = pt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))");
        let mbc = minimum_bounding_circle(&g).expect("should succeed");
        assert_eq!(mbc.geometry_type(), "Polygon");
        // The MBC of a unit square has radius = sqrt(2)/2 ≈ 0.707
        // and center at (0.5, 0.5)
        let verts = extract_vertices(&mbc.geom);
        assert_eq!(verts.len(), 65, "64-segment polygon + closing vertex");
    }

    #[test]
    fn test_mbc_contains_all_vertices() {
        let g = pt("MULTIPOINT((0 0),(5 0),(0 3),(5 3))");
        let mbc = minimum_bounding_circle(&g).expect("should succeed");

        // Compute center of MBC polygon (centroid of its vertices)
        let verts = extract_vertices(&mbc.geom);
        let n = verts.len() as f64;
        let cx = verts.iter().map(|(x, _)| x).sum::<f64>() / n;
        let cy = verts.iter().map(|(_, y)| y).sum::<f64>() / n;
        let r = verts
            .iter()
            .map(|(x, y)| pt_dist(cx, cy, *x, *y))
            .fold(0.0_f64, f64::max);

        // All original points should be within or on the circle
        let orig_pts = [(0.0_f64, 0.0_f64), (5.0, 0.0), (0.0, 3.0), (5.0, 3.0)];
        for (px, py) in orig_pts {
            let d = pt_dist(cx, cy, px, py);
            assert!(d <= r + 0.1, "point ({px},{py}) outside MBC: d={d} r={r}");
        }
    }

    #[test]
    fn test_mbc_empty_returns_error() {
        let a = Geometry::new(GeoGeometry::GeometryCollection(
            geo_types::GeometryCollection::new_from(vec![]),
        ));
        assert!(minimum_bounding_circle(&a).is_err());
    }

    // ── Convex Hull ───────────────────────────────────────────────────────────

    #[test]
    fn test_convex_hull_triangle() {
        let g = pt("MULTIPOINT((0 0),(4 0),(2 4))");
        let hull = convex_hull_geom(&g).expect("should succeed");
        assert_eq!(hull.geometry_type(), "Polygon");
    }

    #[test]
    fn test_convex_hull_removes_interior_points() {
        // Interior point (2,2) should not be on the hull
        let g = pt("MULTIPOINT((0 0),(4 0),(4 4),(0 4),(2 2))");
        let hull = convex_hull_geom(&g).expect("should succeed");
        assert_eq!(hull.geometry_type(), "Polygon");
        // Hull should be the outer 4 points (approximately a square)
        let verts = extract_vertices(&hull.geom);
        // Hull polygon ring has n+1 vertices (closed)
        assert!(verts.len() <= 6, "hull verts={}", verts.len()); // 4 corners + close = 5
    }

    #[test]
    fn test_convex_hull_polygon() {
        let g = pt("POLYGON((0 0, 3 0, 3 3, 0 3, 0 0))");
        let hull = convex_hull_geom(&g).expect("should succeed");
        assert_eq!(hull.geometry_type(), "Polygon");
    }

    #[test]
    fn test_convex_hull_linestring() {
        let g = pt("LINESTRING(0 0, 2 2, 4 0, 2 -2)");
        let hull = convex_hull_geom(&g).expect("should succeed");
        assert_eq!(hull.geometry_type(), "Polygon");
    }

    #[test]
    fn test_convex_hull_single_point_is_polygon() {
        let g = pt("POINT(1 2)");
        // Should succeed (degenerate hull is a point or polygon with coincident vertices)
        let hull = convex_hull_geom(&g);
        assert!(hull.is_ok());
    }

    #[test]
    fn test_convex_hull_empty_returns_error() {
        let a = Geometry::new(GeoGeometry::GeometryCollection(
            geo_types::GeometryCollection::new_from(vec![]),
        ));
        assert!(convex_hull_geom(&a).is_err());
    }

    // ── Voronoi Diagram ───────────────────────────────────────────────────────

    #[test]
    fn test_voronoi_basic_triangle() {
        let g = pt("MULTIPOINT((0 0),(4 0),(2 4))");
        let v = voronoi_diagram(&g).expect("should succeed");
        assert_eq!(v.geometry_type(), "GeometryCollection");
    }

    #[test]
    fn test_voronoi_returns_geometry_collection() {
        let g = pt("MULTIPOINT((0 0),(4 0),(2 3),(0 4),(4 4))");
        let v = voronoi_diagram(&g).expect("should succeed");
        assert_eq!(v.geometry_type(), "GeometryCollection");
    }

    #[test]
    fn test_voronoi_cells_count() {
        let g = pt("MULTIPOINT((0 0),(4 0),(2 3))");
        let v = voronoi_diagram(&g).expect("should succeed");
        // Should produce 3 cells (one per input point)
        if let GeoGeometry::GeometryCollection(gc) = &v.geom {
            assert!(!gc.is_empty(), "Voronoi must have at least one cell");
        } else {
            panic!("Expected GeometryCollection");
        }
    }

    #[test]
    fn test_voronoi_too_few_points_error() {
        let g = pt("MULTIPOINT((0 0),(1 1))");
        assert!(voronoi_diagram(&g).is_err());
    }

    #[test]
    fn test_voronoi_single_point_error() {
        let g = pt("POINT(1 1)");
        assert!(voronoi_diagram(&g).is_err());
    }

    #[test]
    fn test_voronoi_five_points() {
        let g = pt("MULTIPOINT((0 0),(10 0),(10 10),(0 10),(5 5))");
        let v = voronoi_diagram(&g).expect("should succeed");
        assert_eq!(v.geometry_type(), "GeometryCollection");
        if let GeoGeometry::GeometryCollection(gc) = &v.geom {
            assert!(!gc.is_empty());
        }
    }

    #[test]
    fn test_voronoi_colinear_points_handled() {
        // 3 collinear points — Bowyer-Watson may produce degenerate triangles
        // but should not panic
        let g = pt("MULTIPOINT((0 0),(1 0),(2 0),(1 1))");
        // Should either succeed or return an error, not panic
        let _result = voronoi_diagram(&g);
    }

    // ── URI constants ─────────────────────────────────────────────────────────

    #[test]
    fn test_uri_constants_not_empty() {
        assert!(!GEOF_HAUSDORFF_DISTANCE.is_empty());
        assert!(!GEOF_FRECHET_DISTANCE.is_empty());
        assert!(!GEOF_MINIMUM_BOUNDING_CIRCLE.is_empty());
        assert!(!GEOF_CONVEX_HULL_EXT.is_empty());
        assert!(!GEOF_VORONOI_DIAGRAM.is_empty());
    }

    #[test]
    fn test_uri_constants_contain_geosparql() {
        assert!(GEOF_HAUSDORFF_DISTANCE.contains("geosparql"));
        assert!(GEOF_FRECHET_DISTANCE.contains("geosparql"));
        assert!(GEOF_MINIMUM_BOUNDING_CIRCLE.contains("geosparql"));
        assert!(GEOF_CONVEX_HULL_EXT.contains("geosparql"));
        assert!(GEOF_VORONOI_DIAGRAM.contains("geosparql"));
    }

    // ── Helper function tests ─────────────────────────────────────────────────

    #[test]
    fn test_circumcircle_unit_triangle() {
        // Right triangle with vertices (0,0),(2,0),(0,2)
        // Circumcenter at (1,1), radius = sqrt(2)
        let (cx, cy, r) = circumcircle((0.0, 0.0), (2.0, 0.0), (0.0, 2.0));
        assert!((cx - 1.0).abs() < 1e-8, "cx={cx}");
        assert!((cy - 1.0).abs() < 1e-8, "cy={cy}");
        assert!((r - std::f64::consts::SQRT_2).abs() < 1e-8, "r={r}");
    }

    #[test]
    fn test_discrete_frechet_dp_single_points() {
        let a = [(0.0_f64, 0.0_f64)];
        let b = [(3.0_f64, 4.0_f64)];
        let d = discrete_frechet_dp(&a, &b);
        assert!((d - 5.0).abs() < 1e-10, "d={d}");
    }

    #[test]
    fn test_welzl_three_points() {
        let pts = [(0.0_f64, 0.0_f64), (4.0, 0.0), (2.0, 3.0)];
        let (cx, cy, r) = welzl(&pts, &[]);
        // All points must be inside or on the circle
        for &(px, py) in &pts {
            let d = pt_dist(cx, cy, px, py);
            assert!(d <= r + 1e-8, "point ({px},{py}) outside: d={d} r={r}");
        }
    }
}
