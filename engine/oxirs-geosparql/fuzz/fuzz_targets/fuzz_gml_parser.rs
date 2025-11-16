#![no_main]

use libfuzzer_sys::fuzz_target;
use oxirs_geosparql::geometry::Geometry;

fuzz_target!(|data: &[u8]| {
    // Try to parse as GML - should never panic, only return Result
    #[cfg(feature = "gml-support")]
    {
        let _ = Geometry::from_gml(data);

        // Also test direct GML parsing function
        let _ = oxirs_geosparql::geometry::gml_parser::parse_gml(data);
    }

    #[cfg(not(feature = "gml-support"))]
    {
        // Still try to parse to ensure error handling works without feature
        let _ = Geometry::from_gml(data);
    }
});
