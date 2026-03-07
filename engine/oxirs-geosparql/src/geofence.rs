//! Geographic fence/boundary detection.
//!
//! Implements circle, bounding-box, and polygon geofences with enter/exit event
//! tracking for moving assets.

use std::collections::{HashMap, HashSet};

/// A latitude/longitude coordinate.
#[derive(Debug, Clone, PartialEq)]
pub struct LatLon {
    /// Latitude in decimal degrees (−90 … +90).
    pub lat: f64,
    /// Longitude in decimal degrees (−180 … +180).
    pub lon: f64,
}

impl LatLon {
    /// Construct a coordinate.
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }
}

/// Geometry describing a fence boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum FenceShape {
    /// Circular fence defined by a centre and a radius in metres.
    Circle { center: LatLon, radius_m: f64 },
    /// Axis-aligned bounding box.
    BoundingBox {
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    },
    /// Arbitrary polygon (vertices ordered, implicitly closed).
    Polygon { vertices: Vec<LatLon> },
}

/// A named geofence.
#[derive(Debug, Clone)]
pub struct Geofence {
    /// Unique identifier for the fence.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Geometry of this fence.
    pub shape: FenceShape,
}

impl Geofence {
    /// Construct a new fence.
    pub fn new(id: impl Into<String>, name: impl Into<String>, shape: FenceShape) -> Self {
        Self { id: id.into(), name: name.into(), shape }
    }
}

/// Events generated when an asset's position changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenceEvent {
    /// Asset entered the fence (fence id).
    Entered(String),
    /// Asset exited the fence (fence id).
    Exited(String),
    /// Asset is currently inside the fence (fence id).
    Inside(String),
}

/// Manages fences and tracks per-asset inside/outside state.
#[derive(Debug, Default)]
pub struct GeofenceManager {
    fences: HashMap<String, Geofence>,
    /// `inside[asset_id]` = set of fence ids the asset is currently inside.
    inside: HashMap<String, HashSet<String>>,
}

impl GeofenceManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new fence.
    pub fn add_fence(&mut self, fence: Geofence) {
        self.fences.insert(fence.id.clone(), fence);
    }

    /// Remove a fence by id. Returns `true` if the fence existed.
    pub fn remove_fence(&mut self, id: &str) -> bool {
        let removed = self.fences.remove(id).is_some();
        if removed {
            // Remove fence from all inside sets.
            for inside_set in self.inside.values_mut() {
                inside_set.remove(id);
            }
        }
        removed
    }

    /// Update asset position and produce enter/exit/inside events.
    pub fn check(&mut self, asset_id: &str, pos: &LatLon) -> Vec<FenceEvent> {
        let mut events = Vec::new();
        let prev_inside = self
            .inside
            .entry(asset_id.to_string())
            .or_default()
            .clone();

        let mut current_inside: HashSet<String> = HashSet::new();
        for (fence_id, fence) in &self.fences {
            if is_inside(pos, &fence.shape) {
                current_inside.insert(fence_id.clone());
            }
        }

        // Entered fences.
        for fid in current_inside.difference(&prev_inside) {
            events.push(FenceEvent::Entered(fid.clone()));
        }
        // Exited fences.
        for fid in prev_inside.difference(&current_inside) {
            events.push(FenceEvent::Exited(fid.clone()));
        }
        // Still inside.
        for fid in current_inside.intersection(&prev_inside) {
            events.push(FenceEvent::Inside(fid.clone()));
        }

        *self.inside.entry(asset_id.to_string()).or_default() = current_inside;
        events.sort_by(|a, b| {
            fence_event_order(a).cmp(&fence_event_order(b))
        });
        events
    }

    /// Return the ids of fences that `asset_id` is currently inside.
    pub fn inside_fences(&self, asset_id: &str) -> Vec<&str> {
        match self.inside.get(asset_id) {
            None => Vec::new(),
            Some(set) => set.iter().map(String::as_str).collect(),
        }
    }

    /// Number of registered fences.
    pub fn fence_count(&self) -> usize {
        self.fences.len()
    }
}

fn fence_event_order(e: &FenceEvent) -> u8 {
    match e {
        FenceEvent::Entered(_) => 0,
        FenceEvent::Inside(_) => 1,
        FenceEvent::Exited(_) => 2,
    }
}

/// Check whether `pos` lies inside the given `shape`.
fn is_inside(pos: &LatLon, shape: &FenceShape) -> bool {
    match shape {
        FenceShape::Circle { center, radius_m } => point_in_circle(pos, center, *radius_m),
        FenceShape::BoundingBox { min_lat, min_lon, max_lat, max_lon } => {
            point_in_bbox(pos, *min_lat, *min_lon, *max_lat, *max_lon)
        }
        FenceShape::Polygon { vertices } => point_in_polygon(pos, vertices),
    }
}

/// Compute the great-circle distance between two coordinates in metres
/// (Haversine formula).
pub fn haversine_distance_m(a: &LatLon, b: &LatLon) -> f64 {
    const EARTH_RADIUS_M: f64 = 6_371_000.0;
    let dlat = (b.lat - a.lat).to_radians();
    let dlon = (b.lon - a.lon).to_radians();
    let lat1 = a.lat.to_radians();
    let lat2 = b.lat.to_radians();
    let sin_dlat = (dlat / 2.0).sin();
    let sin_dlon = (dlon / 2.0).sin();
    let h = sin_dlat * sin_dlat + lat1.cos() * lat2.cos() * sin_dlon * sin_dlon;
    let c = 2.0 * h.sqrt().asin();
    EARTH_RADIUS_M * c
}

/// Check whether `point` lies within the polygon defined by `vertices`
/// using the ray-casting algorithm.
pub fn point_in_polygon(point: &LatLon, vertices: &[LatLon]) -> bool {
    if vertices.len() < 3 {
        return false;
    }
    let n = vertices.len();
    let mut inside = false;
    let (px, py) = (point.lon, point.lat);

    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (vertices[i].lon, vertices[i].lat);
        let (xj, yj) = (vertices[j].lon, vertices[j].lat);

        let intersect =
            ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Check whether `point` is within `radius_m` metres of `center`.
pub fn point_in_circle(point: &LatLon, center: &LatLon, radius_m: f64) -> bool {
    haversine_distance_m(point, center) <= radius_m
}

/// Check whether `point` lies within or on the axis-aligned bounding box.
pub fn point_in_bbox(
    point: &LatLon,
    min_lat: f64,
    min_lon: f64,
    max_lat: f64,
    max_lon: f64,
) -> bool {
    point.lat >= min_lat
        && point.lat <= max_lat
        && point.lon >= min_lon
        && point.lon <= max_lon
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── haversine_distance_m ─────────────────────────────────────────────────

    #[test]
    fn test_haversine_same_point() {
        let p = LatLon::new(51.5, -0.1);
        let d = haversine_distance_m(&p, &p);
        assert!(d < 1e-6);
    }

    #[test]
    fn test_haversine_london_paris_approx() {
        let london = LatLon::new(51.5074, -0.1278);
        let paris = LatLon::new(48.8566, 2.3522);
        let d = haversine_distance_m(&london, &paris);
        // ~340 km
        assert!(d > 300_000.0 && d < 380_000.0, "got {d}");
    }

    #[test]
    fn test_haversine_symmetric() {
        let a = LatLon::new(40.0, 10.0);
        let b = LatLon::new(50.0, 20.0);
        let d1 = haversine_distance_m(&a, &b);
        let d2 = haversine_distance_m(&b, &a);
        assert!((d1 - d2).abs() < 1e-6);
    }

    #[test]
    fn test_haversine_equator_step() {
        // 1 degree longitude on the equator ≈ 111 km
        let a = LatLon::new(0.0, 0.0);
        let b = LatLon::new(0.0, 1.0);
        let d = haversine_distance_m(&a, &b);
        assert!(d > 110_000.0 && d < 112_000.0);
    }

    // ── point_in_circle ──────────────────────────────────────────────────────

    #[test]
    fn test_point_in_circle_inside() {
        let center = LatLon::new(51.5, -0.1);
        let point = LatLon::new(51.5001, -0.1001); // very close
        assert!(point_in_circle(&point, &center, 100.0));
    }

    #[test]
    fn test_point_in_circle_outside() {
        let center = LatLon::new(51.5, -0.1);
        let far = LatLon::new(52.0, 0.0); // ~56 km away
        assert!(!point_in_circle(&far, &center, 1000.0));
    }

    #[test]
    fn test_point_in_circle_on_boundary() {
        let center = LatLon::new(0.0, 0.0);
        let point = LatLon::new(0.0, 0.0);
        assert!(point_in_circle(&point, &center, 0.0));
    }

    // ── point_in_bbox ────────────────────────────────────────────────────────

    #[test]
    fn test_point_in_bbox_inside() {
        assert!(point_in_bbox(&LatLon::new(5.0, 5.0), 0.0, 0.0, 10.0, 10.0));
    }

    #[test]
    fn test_point_in_bbox_on_edge() {
        assert!(point_in_bbox(&LatLon::new(0.0, 5.0), 0.0, 0.0, 10.0, 10.0));
    }

    #[test]
    fn test_point_in_bbox_outside() {
        assert!(!point_in_bbox(&LatLon::new(15.0, 5.0), 0.0, 0.0, 10.0, 10.0));
    }

    #[test]
    fn test_point_in_bbox_lon_outside() {
        assert!(!point_in_bbox(&LatLon::new(5.0, 15.0), 0.0, 0.0, 10.0, 10.0));
    }

    // ── point_in_polygon ─────────────────────────────────────────────────────

    fn unit_square() -> Vec<LatLon> {
        vec![
            LatLon::new(0.0, 0.0),
            LatLon::new(0.0, 1.0),
            LatLon::new(1.0, 1.0),
            LatLon::new(1.0, 0.0),
        ]
    }

    #[test]
    fn test_polygon_inside() {
        assert!(point_in_polygon(&LatLon::new(0.5, 0.5), &unit_square()));
    }

    #[test]
    fn test_polygon_outside() {
        assert!(!point_in_polygon(&LatLon::new(2.0, 2.0), &unit_square()));
    }

    #[test]
    fn test_polygon_too_few_vertices() {
        let verts = vec![LatLon::new(0.0, 0.0), LatLon::new(1.0, 1.0)];
        assert!(!point_in_polygon(&LatLon::new(0.5, 0.5), &verts));
    }

    #[test]
    fn test_polygon_empty() {
        assert!(!point_in_polygon(&LatLon::new(0.0, 0.0), &[]));
    }

    // ── Geofence / FenceShape ────────────────────────────────────────────────

    #[test]
    fn test_geofence_construction() {
        let f = Geofence::new("f1", "My Fence", FenceShape::BoundingBox {
            min_lat: 0.0, min_lon: 0.0, max_lat: 10.0, max_lon: 10.0,
        });
        assert_eq!(f.id, "f1");
        assert_eq!(f.name, "My Fence");
    }

    // ── GeofenceManager ──────────────────────────────────────────────────────

    #[test]
    fn test_manager_add_fence() {
        let mut mgr = GeofenceManager::new();
        mgr.add_fence(Geofence::new("f1", "F1", FenceShape::BoundingBox {
            min_lat: 0.0, min_lon: 0.0, max_lat: 10.0, max_lon: 10.0,
        }));
        assert_eq!(mgr.fence_count(), 1);
    }

    #[test]
    fn test_manager_remove_fence() {
        let mut mgr = GeofenceManager::new();
        mgr.add_fence(Geofence::new("f1", "F1", FenceShape::BoundingBox {
            min_lat: 0.0, min_lon: 0.0, max_lat: 10.0, max_lon: 10.0,
        }));
        assert!(mgr.remove_fence("f1"));
        assert_eq!(mgr.fence_count(), 0);
    }

    #[test]
    fn test_manager_remove_nonexistent() {
        let mut mgr = GeofenceManager::new();
        assert!(!mgr.remove_fence("missing"));
    }

    #[test]
    fn test_manager_check_enter() {
        let mut mgr = GeofenceManager::new();
        mgr.add_fence(Geofence::new("f1", "Box", FenceShape::BoundingBox {
            min_lat: 0.0, min_lon: 0.0, max_lat: 10.0, max_lon: 10.0,
        }));
        let events = mgr.check("asset1", &LatLon::new(5.0, 5.0));
        assert!(events.contains(&FenceEvent::Entered("f1".to_string())));
    }

    #[test]
    fn test_manager_check_inside_on_second_call() {
        let mut mgr = GeofenceManager::new();
        mgr.add_fence(Geofence::new("f1", "Box", FenceShape::BoundingBox {
            min_lat: 0.0, min_lon: 0.0, max_lat: 10.0, max_lon: 10.0,
        }));
        mgr.check("a", &LatLon::new(5.0, 5.0)); // enter
        let events = mgr.check("a", &LatLon::new(6.0, 6.0)); // still inside
        assert!(events.contains(&FenceEvent::Inside("f1".to_string())));
    }

    #[test]
    fn test_manager_check_exit() {
        let mut mgr = GeofenceManager::new();
        mgr.add_fence(Geofence::new("f1", "Box", FenceShape::BoundingBox {
            min_lat: 0.0, min_lon: 0.0, max_lat: 10.0, max_lon: 10.0,
        }));
        mgr.check("a", &LatLon::new(5.0, 5.0)); // enter
        let events = mgr.check("a", &LatLon::new(20.0, 20.0)); // exit
        assert!(events.contains(&FenceEvent::Exited("f1".to_string())));
    }

    #[test]
    fn test_manager_inside_fences_empty() {
        let mgr = GeofenceManager::new();
        assert!(mgr.inside_fences("asset1").is_empty());
    }

    #[test]
    fn test_manager_inside_fences_after_enter() {
        let mut mgr = GeofenceManager::new();
        mgr.add_fence(Geofence::new("f1", "Box", FenceShape::BoundingBox {
            min_lat: 0.0, min_lon: 0.0, max_lat: 10.0, max_lon: 10.0,
        }));
        mgr.check("a", &LatLon::new(5.0, 5.0));
        let inside = mgr.inside_fences("a");
        assert!(inside.contains(&"f1"));
    }

    #[test]
    fn test_manager_circle_fence() {
        let mut mgr = GeofenceManager::new();
        mgr.add_fence(Geofence::new("c1", "Circle", FenceShape::Circle {
            center: LatLon::new(51.5, -0.1),
            radius_m: 5000.0,
        }));
        // Point ~1 km away
        let events = mgr.check("a", &LatLon::new(51.508, -0.1));
        assert!(events.contains(&FenceEvent::Entered("c1".to_string())));
    }

    #[test]
    fn test_manager_polygon_fence() {
        let mut mgr = GeofenceManager::new();
        mgr.add_fence(Geofence::new("p1", "Polygon", FenceShape::Polygon {
            vertices: vec![
                LatLon::new(0.0, 0.0),
                LatLon::new(0.0, 10.0),
                LatLon::new(10.0, 10.0),
                LatLon::new(10.0, 0.0),
            ],
        }));
        let events = mgr.check("a", &LatLon::new(5.0, 5.0));
        assert!(events.contains(&FenceEvent::Entered("p1".to_string())));
    }

    #[test]
    fn test_manager_multiple_fences() {
        let mut mgr = GeofenceManager::new();
        mgr.add_fence(Geofence::new("f1", "F1", FenceShape::BoundingBox {
            min_lat: 0.0, min_lon: 0.0, max_lat: 10.0, max_lon: 10.0,
        }));
        mgr.add_fence(Geofence::new("f2", "F2", FenceShape::BoundingBox {
            min_lat: 0.0, min_lon: 0.0, max_lat: 20.0, max_lon: 20.0,
        }));
        let events = mgr.check("a", &LatLon::new(5.0, 5.0));
        let entered: Vec<_> = events.iter().filter(|e| matches!(e, FenceEvent::Entered(_))).collect();
        assert_eq!(entered.len(), 2);
    }

    #[test]
    fn test_manager_multiple_assets_independent() {
        let mut mgr = GeofenceManager::new();
        mgr.add_fence(Geofence::new("f1", "F1", FenceShape::BoundingBox {
            min_lat: 0.0, min_lon: 0.0, max_lat: 10.0, max_lon: 10.0,
        }));
        mgr.check("a1", &LatLon::new(5.0, 5.0)); // a1 enters
        let events = mgr.check("a2", &LatLon::new(5.0, 5.0)); // a2 also enters independently
        assert!(events.contains(&FenceEvent::Entered("f1".to_string())));
    }

    #[test]
    fn test_manager_no_event_outside() {
        let mut mgr = GeofenceManager::new();
        mgr.add_fence(Geofence::new("f1", "F1", FenceShape::BoundingBox {
            min_lat: 0.0, min_lon: 0.0, max_lat: 10.0, max_lon: 10.0,
        }));
        let events = mgr.check("a", &LatLon::new(50.0, 50.0));
        assert!(events.is_empty());
    }

    #[test]
    fn test_fence_count_after_multiple_adds() {
        let mut mgr = GeofenceManager::new();
        for i in 0..5 {
            mgr.add_fence(Geofence::new(format!("f{i}"), "F", FenceShape::BoundingBox {
                min_lat: 0.0, min_lon: 0.0, max_lat: 1.0, max_lon: 1.0,
            }));
        }
        assert_eq!(mgr.fence_count(), 5);
    }
}
