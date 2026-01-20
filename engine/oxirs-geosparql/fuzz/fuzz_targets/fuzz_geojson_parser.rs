#![no_main]

use libfuzzer_sys::fuzz_target;
use oxirs_geosparql::geometry::Geometry;

fuzz_target!(|data: &[u8]| {
    // Try to parse as GeoJSON - should never panic, only return Result
    #[cfg(feature = "geojson-support")]
    {
        let _ = Geometry::from_geojson(data);

        // Also test parsing feature collections
        let _ = oxirs_geosparql::geometry::geojson_parser::parse_geojson_feature_collection(data);
    }

    #[cfg(not(feature = "geojson-support"))]
    {
        // Still try to parse to ensure error handling works without feature
        let _ = Geometry::from_geojson(data);
    }
});
