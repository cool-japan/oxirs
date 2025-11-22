#![no_main]

use libfuzzer_sys::fuzz_target;
use oxirs_ttl::nquads::NQuadsParser;
use oxirs_ttl::Parser;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string (lossy is fine for fuzzing)
    if let Ok(input) = std::str::from_utf8(data) {
        // Try to parse the input
        let parser = NQuadsParser::new();
        let cursor = Cursor::new(input);
        let _ = parser.for_reader(cursor).collect::<Result<Vec<_>, _>>();
    }
});
