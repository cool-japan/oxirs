#![no_main]

use libfuzzer_sys::fuzz_target;
use oxirs_ttl::formats::turtle::TurtleParser;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string (lossy is fine for fuzzing)
    if let Ok(input) = std::str::from_utf8(data) {
        // Try to parse the input
        let parser = TurtleParser::new();
        let _ = parser.parse_document(input);

        // Also try lenient mode
        let lenient_parser = TurtleParser::new_lenient();
        let _ = lenient_parser.parse_document(input);
    }
});
