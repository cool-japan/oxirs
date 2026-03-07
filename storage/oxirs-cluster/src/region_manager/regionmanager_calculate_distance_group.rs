//! # RegionManager - calculate_distance_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::GeoCoordinates;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Calculate distance between two coordinates (Haversine formula)
    pub(super) fn calculate_distance(
        &self,
        coord_a: &GeoCoordinates,
        coord_b: &GeoCoordinates,
    ) -> f64 {
        const EARTH_RADIUS_KM: f64 = 6371.0;
        let lat1_rad = coord_a.latitude.to_radians();
        let lat2_rad = coord_b.latitude.to_radians();
        let delta_lat = (coord_b.latitude - coord_a.latitude).to_radians();
        let delta_lon = (coord_b.longitude - coord_a.longitude).to_radians();
        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        EARTH_RADIUS_KM * c
    }
}
