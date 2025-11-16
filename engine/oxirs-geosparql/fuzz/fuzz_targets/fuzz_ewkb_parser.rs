#![no_main]

use libfuzzer_sys::fuzz_target;
use oxirs_geosparql::geometry::Geometry;

fuzz_target!(|data: &[u8]| {
    // Try to parse as EWKB - should never panic, only return Result
    let _ = Geometry::from_ewkb(data);

    // Also test direct EWKB parsing function
    let _ = oxirs_geosparql::geometry::ewkb_parser::parse_ewkb(data);

    // Test EWKT parsing if data is valid UTF-8
    if let Ok(ewkt_string) = std::str::from_utf8(data) {
        let _ = Geometry::from_ewkt(ewkt_string);
        let _ = oxirs_geosparql::geometry::ewkt_parser::parse_ewkt(ewkt_string);
    }
});
