#![no_main]

use libfuzzer_sys::fuzz_target;
use oxirs_ttl::formats::turtle::{TurtleParser, TurtleSerializer};

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string (lossy is fine for fuzzing)
    if let Ok(input) = std::str::from_utf8(data) {
        // Try to parse and then serialize
        let parser = TurtleParser::new_lenient();
        if let Ok(triples) = parser.parse_document(input) {
            let serializer = TurtleSerializer::new();
            let mut output = Vec::new();
            let _ = serializer.serialize(&triples, &mut output);
        }
    }
});
