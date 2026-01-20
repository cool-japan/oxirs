#![no_main]

use libfuzzer_sys::fuzz_target;
use oxirs_geosparql::geometry::Geometry;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string, ignoring invalid UTF-8
    if let Ok(wkt_string) = std::str::from_utf8(data) {
        // Try to parse as WKT - should never panic, only return Result
        let _ = Geometry::from_wkt(wkt_string);

        // Also test the direct WKT parsing function
        let _ = oxirs_geosparql::geometry::wkt_parser::parse_wkt(wkt_string);
    }
});
