#![no_main]

use libfuzzer_sys::fuzz_target;
use oxirs_geosparql::geometry::zero_copy_wkt::{WktArena, ZeroCopyWktParser};

fuzz_target!(|data: &[u8]| {
    // Try to parse with zero-copy WKT parser - should never panic
    if let Ok(wkt_string) = std::str::from_utf8(data) {
        let arena = WktArena::new();
        let mut parser = ZeroCopyWktParser::new(&arena);

        // Try parsing - should handle all malformed input gracefully
        let _ = parser.parse_wkt(wkt_string);

        // Test arena allocation patterns
        let _ = arena.store_str(wkt_string);
    }
});
