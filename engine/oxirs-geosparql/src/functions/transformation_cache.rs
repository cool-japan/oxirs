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
