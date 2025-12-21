//! Transformation cache for CRS operations

#[cfg(feature = "proj-support")]
use crate::error::{GeoSparqlError, Result};
#[cfg(feature = "proj-support")]
use crate::geometry::{Crs, Geometry};
#[cfg(feature = "proj-support")]
use parking_lot::RwLock;
#[cfg(feature = "proj-support")]
use std::collections::HashMap;

/// Cache for CRS transformation parameters
///
/// This cache stores transformation parameters (EPSG strings) to avoid repeated
/// lookups when transforming multiple geometries with the same CRS pairs.
///
/// # Thread Safety
///
/// This cache is thread-safe and can be shared across threads using `Arc`.
///
/// # Examples
///
/// ```rust,ignore
/// use oxirs_geosparql::functions::transformation_cache::TransformationCache;
///
/// let cache = TransformationCache::new();
/// // Transform geometries - parameters will be cached
/// ```
#[cfg(feature = "proj-support")]
pub struct TransformationCache {
    cache: RwLock<HashMap<(String, String), (String, String)>>,
}

#[cfg(feature = "proj-support")]
impl TransformationCache {
    /// Create a new transformation cache
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use oxirs_geosparql::functions::transformation_cache::TransformationCache;
    ///
    /// let cache = TransformationCache::new();
    /// ```
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    fn get_transform_params(&self, source_crs: &Crs, target_crs: &Crs) -> Result<(String, String)> {
        let source_epsg = source_crs.epsg_code().ok_or_else(|| {
            GeoSparqlError::CrsTransformationFailed(format!(
                "Source CRS must have EPSG code: {}",
                source_crs.uri
            ))
        })?;

        let target_epsg = target_crs.epsg_code().ok_or_else(|| {
            GeoSparqlError::CrsTransformationFailed(format!(
                "Target CRS must have EPSG code: {}",
                target_crs.uri
            ))
        })?;

        let source_string = format!("EPSG:{}", source_epsg);
        let target_string = format!("EPSG:{}", target_epsg);
        let key = (source_crs.uri.clone(), target_crs.uri.clone());

        {
            let cache = self.cache.read();
            if let Some(params) = cache.get(&key) {
                return Ok(params.clone());
            }
        }

        {
            let mut cache = self.cache.write();
            cache.insert(key, (source_string.clone(), target_string.clone()));
        }

        Ok((source_string, target_string))
    }

    /// Transform a geometry to a target CRS using cached parameters
    ///
    /// # Arguments
    ///
    /// * `geom` - The geometry to transform
    /// * `target_crs` - The target CRS
    ///
    /// # Returns
    ///
    /// Transformed geometry in the target CRS
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let cache = TransformationCache::new();
    /// let transformed = cache.transform(&geom, &Crs::epsg(3857))?;
    /// ```
    pub fn transform(&self, geom: &Geometry, target_crs: &Crs) -> Result<Geometry> {
        use geo::algorithm::map_coords::MapCoords;
        use proj::Proj;

        if &geom.crs == target_crs {
            return Ok(geom.clone());
        }

        let (source_string, target_string) = self.get_transform_params(&geom.crs, target_crs)?;

        let proj = Proj::new_known_crs(&source_string, &target_string, None).map_err(|e| {
            GeoSparqlError::CrsTransformationFailed(format!(
                "Failed to create PROJ transformation from {} to {}: {}",
                source_string, target_string, e
            ))
        })?;

        let transformed_geom = geom.geom.map_coords(|coord| {
            let point: (f64, f64) = proj.convert((coord.x, coord.y)).unwrap_or_else(|e| {
                tracing::warn!(
                    "PROJ conversion failed for ({}, {}): {}. Using original coordinates.",
                    coord.x,
                    coord.y,
                    e
                );
                (coord.x, coord.y)
            });
            geo_types::Coord {
                x: point.0,
                y: point.1,
            }
        });

        Ok(Geometry::with_crs(transformed_geom, target_crs.clone()))
    }

    /// Get the number of cached transformation parameter pairs
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let cache = TransformationCache::new();
    /// assert_eq!(cache.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Check if the cache is empty
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let cache = TransformationCache::new();
    /// assert!(cache.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }
}

#[cfg(feature = "proj-support")]
impl Default for TransformationCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "proj-support"))]
mod tests {
    use super::*;
    use geo_types::{coord, Point};

    #[test]
    fn test_cache_creation() {
        let cache = TransformationCache::new();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_default() {
        let cache = TransformationCache::default();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_transform_same_crs() {
        let cache = TransformationCache::new();
        let crs = Crs::epsg(4326);
        let point = Geometry::with_crs(Point::new(10.0, 20.0).into(), crs.clone());

        let result = cache.transform(&point, &crs).unwrap();

        // Should return clone without transformation
        assert_eq!(result.crs, crs);
        match result.geom {
            geo_types::Geometry::Point(p) => {
                assert_eq!(p.x(), 10.0);
                assert_eq!(p.y(), 20.0);
            }
            _ => panic!("Expected Point geometry"),
        }

        // Cache should remain empty when CRS is the same
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_transform_wgs84_to_web_mercator() {
        let cache = TransformationCache::new();
        let wgs84 = Crs::epsg(4326);
        let web_mercator = Crs::epsg(3857);

        // Create a point in WGS84 (longitude, latitude)
        let point = Geometry::with_crs(Point::new(10.0, 20.0).into(), wgs84.clone());

        // Transform to Web Mercator
        let result = cache.transform(&point, &web_mercator).unwrap();

        assert_eq!(result.crs, web_mercator);

        // Cache should now have one entry
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        // Transform another point - should use cached parameters
        let point2 = Geometry::with_crs(Point::new(15.0, 25.0).into(), wgs84);
        let result2 = cache.transform(&point2, &web_mercator).unwrap();

        assert_eq!(result2.crs, web_mercator);

        // Cache should still have one entry (same CRS pair)
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_transform_multiple_crs_pairs() {
        let cache = TransformationCache::new();
        let wgs84 = Crs::epsg(4326);
        let web_mercator = Crs::epsg(3857);
        let utm_zone_32n = Crs::epsg(32632);

        let point = Geometry::with_crs(Point::new(10.0, 50.0).into(), wgs84.clone());

        // First transformation: WGS84 -> Web Mercator
        let _ = cache.transform(&point, &web_mercator).unwrap();
        assert_eq!(cache.len(), 1);

        // Second transformation: WGS84 -> UTM Zone 32N
        let _ = cache.transform(&point, &utm_zone_32n).unwrap();
        assert_eq!(cache.len(), 2);

        // Third transformation: Web Mercator -> WGS84 (reverse)
        let web_merc_point = Geometry::with_crs(
            Point::new(1113194.91, 6446275.84).into(),
            web_mercator.clone(),
        );
        let _ = cache.transform(&web_merc_point, &wgs84).unwrap();
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_transform_invalid_crs() {
        let cache = TransformationCache::new();

        // Create a CRS with non-EPSG URI
        let invalid_crs = Crs::new("http://example.com/custom-crs");
        let wgs84 = Crs::epsg(4326);

        let point = Geometry::with_crs(Point::new(10.0, 20.0).into(), invalid_crs);

        // Should fail because source CRS doesn't have EPSG code
        let result = cache.transform(&point, &wgs84);
        assert!(result.is_err());

        // Cache should remain empty
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_transform_roundtrip() {
        let cache = TransformationCache::new();
        let wgs84 = Crs::epsg(4326);
        let web_mercator = Crs::epsg(3857);

        // Original point in WGS84
        let original = Geometry::with_crs(Point::new(10.0, 50.0).into(), wgs84.clone());

        // Transform to Web Mercator
        let transformed = cache.transform(&original, &web_mercator).unwrap();

        // Transform back to WGS84
        let roundtrip = cache.transform(&transformed, &wgs84).unwrap();

        // Should be approximately equal (within floating point tolerance)
        match (&original.geom, &roundtrip.geom) {
            (geo_types::Geometry::Point(p1), geo_types::Geometry::Point(p2)) => {
                assert!((p1.x() - p2.x()).abs() < 1e-6);
                assert!((p1.y() - p2.y()).abs() < 1e-6);
            }
            _ => panic!("Expected Point geometries"),
        }

        // Cache should have 2 entries (forward and reverse)
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(TransformationCache::new());
        let wgs84 = Crs::epsg(4326);
        let web_mercator = Crs::epsg(3857);

        let mut handles = vec![];

        // Spawn multiple threads that all transform geometries
        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let wgs84_clone = wgs84.clone();
            let web_merc_clone = web_mercator.clone();

            let handle = thread::spawn(move || {
                let point =
                    Geometry::with_crs(Point::new(10.0 + i as f64, 50.0).into(), wgs84_clone);
                cache_clone.transform(&point, &web_merc_clone).unwrap()
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // All threads used the same CRS pair, so cache should have 1 entry
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_transform_linestring() {
        let cache = TransformationCache::new();
        let wgs84 = Crs::epsg(4326);
        let web_mercator = Crs::epsg(3857);

        use geo_types::LineString;
        let line = LineString::from(vec![
            coord! { x: 10.0, y: 50.0 },
            coord! { x: 11.0, y: 51.0 },
            coord! { x: 12.0, y: 52.0 },
        ]);

        let geom = Geometry::with_crs(line.into(), wgs84);

        let transformed = cache.transform(&geom, &web_mercator).unwrap();

        assert_eq!(transformed.crs, web_mercator);
        assert_eq!(cache.len(), 1);

        // Verify it's still a LineString
        match transformed.geom {
            geo_types::Geometry::LineString(ls) => {
                assert_eq!(ls.coords().count(), 3);
            }
            _ => panic!("Expected LineString geometry"),
        }
    }

    #[test]
    fn test_transform_polygon() {
        let cache = TransformationCache::new();
        let wgs84 = Crs::epsg(4326);
        let web_mercator = Crs::epsg(3857);

        use geo_types::Polygon;
        let poly = Polygon::new(
            geo_types::LineString::from(vec![
                coord! { x: 10.0, y: 50.0 },
                coord! { x: 11.0, y: 50.0 },
                coord! { x: 11.0, y: 51.0 },
                coord! { x: 10.0, y: 51.0 },
                coord! { x: 10.0, y: 50.0 },
            ]),
            vec![],
        );

        let geom = Geometry::with_crs(poly.into(), wgs84);

        let transformed = cache.transform(&geom, &web_mercator).unwrap();

        assert_eq!(transformed.crs, web_mercator);
        assert_eq!(cache.len(), 1);

        // Verify it's still a Polygon
        match transformed.geom {
            geo_types::Geometry::Polygon(p) => {
                assert_eq!(p.exterior().coords().count(), 5);
            }
            _ => panic!("Expected Polygon geometry"),
        }
    }
}
