#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // Try to parse as FlatGeobuf - should never panic, only return Result
    #[cfg(feature = "flatgeobuf-support")]
    {
        let cursor = Cursor::new(data);
        let _ = oxirs_geosparql::geometry::flatgeobuf_parser::parse_flatgeobuf(cursor);

        // Also test byte slice parsing
        let _ = oxirs_geosparql::geometry::flatgeobuf_parser::parse_flatgeobuf_bytes(data);
    }

    #[cfg(not(feature = "flatgeobuf-support"))]
    {
        // Still try to parse to ensure error handling works without feature
        let _ = oxirs_geosparql::geometry::flatgeobuf_parser::parse_flatgeobuf_bytes(data);
    }
});
