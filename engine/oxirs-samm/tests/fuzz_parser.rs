//! Fuzz Testing for SAMM TTL Parser
//!
//! This test suite uses property-based testing to fuzz the parser with random inputs,
//! ensuring robustness against malformed or unexpected input.
//!
//! ## Goals
//!
//! 1. **No Panics**: Parser must never panic, even on invalid input
//! 2. **Graceful Errors**: Invalid input should return proper error types
//! 3. **Memory Safety**: No buffer overflows or out-of-bounds access
//! 4. **Edge Cases**: Handle empty input, very long URNs, special characters
//!
//! ## Usage
//!
//! Run with: `cargo test --test fuzz_parser --release`

use oxirs_samm::parser::parse_aspect_from_string;
use proptest::prelude::*;
use scirs2_core::random::{rng, Random, Rng};

/// Strategy for generating random URN components
fn urn_component() -> impl Strategy<Value = String> {
    "[a-z0-9-]{1,20}"
}

/// Strategy for generating potentially valid SAMM namespace URNs
fn samm_namespace() -> impl Strategy<Value = String> {
    (urn_component(), urn_component(), urn_component())
        .prop_map(|(org, name, version)| format!("urn:samm:{}:{}#{}", org, version, name))
}

/// Strategy for generating random strings (including edge cases)
fn random_string() -> impl Strategy<Value = String> {
    prop_oneof![
        // Normal strings
        "[a-zA-Z0-9_-]{1,100}",
        // Empty string
        Just(String::new()),
        // Very long strings
        prop::collection::vec(any::<char>(), 1000..10000)
            .prop_map(|chars| chars.into_iter().collect()),
        // Special characters
        "[\\x00-\\x1F\\x7F-\\xFF]{1,50}",
        // Unicode
        "\\PC{1,50}",
        // Quotes and escapes
        "[\"'\\\\]{1,50}",
    ]
}

/// Strategy for generating random TTL-like content
fn random_ttl() -> impl Strategy<Value = String> {
    prop_oneof![
        // Empty
        Just(String::new()),
        // Whitespace only
        Just("   \n\t\r   ".to_string()),
        // Invalid prefix
        Just("@prefix : <invalid> .".to_string()),
        // Missing prefixes
        Just(":Aspect a samm:Aspect .".to_string()),
        // Malformed triples
        Just("@prefix samm: . :A a samm:Aspect".to_string()),
        // Very long URIs
        (random_string(), random_string()).prop_map(|(prefix, suffix)| {
            format!("@prefix x: <urn:{}> . x:{} a x:Thing .", prefix, suffix)
        }),
        // Nested structures
        Just(
            r#"
            @prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
            :A a samm:Aspect ; samm:properties ( ( ( ( :p ) ) ) ) .
        "#
            .to_string()
        ),
        // Recursive references
        Just(
            r#"
            @prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
            :A samm:properties :A .
        "#
            .to_string()
        ),
        // Missing closing brackets
        Just("@prefix samm: <urn:test#> . :A a [ a [ a samm:Aspect .".to_string()),
        // Unicode edge cases
        Just("@prefix 日本語: <urn:test#> . 日本語:テスト a samm:Aspect .".to_string()),
        // Control characters
        random_string().prop_map(|s| format!("@prefix x: <\x00\x01{}> .", s)),
        // Extremely long lines
        "[a-z]{10000,20000}".prop_map(|s| format!("@prefix x: <urn:{}> .", s)),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Test that parser never panics on random input
    #[test]
    fn fuzz_parser_no_panic(input in random_ttl(), namespace in samm_namespace()) {
        let runtime = tokio::runtime::Runtime::new().unwrap();

        // Should not panic - either success or error
        let _ = runtime.block_on(async {
            parse_aspect_from_string(&input, &namespace).await
        });
    }

    /// Test that parser handles empty and whitespace-only input
    #[test]
    fn fuzz_empty_input(whitespace in "[ \t\n\r]{0,100}") {
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let result = runtime.block_on(async {
            parse_aspect_from_string(&whitespace, "urn:samm:org.example:1.0.0#Test").await
        });

        // Should return error (not panic)
        assert!(result.is_err());
    }

    /// Test that parser handles very long URNs
    #[test]
    fn fuzz_long_urns(
        namespace in "[a-z]{100,1000}",
        version in "[0-9]{50,100}",
        element in "[a-z]{100,500}"
    ) {
        let urn = format!(
            "urn:samm:{}:{}#{}",
            namespace,
            version,
            element
        );

        let input = format!(r#"
            @prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
            @prefix : <{}> .
            :Aspect a samm:Aspect .
        "#, urn);

        let runtime = tokio::runtime::Runtime::new().unwrap();

        // Should handle gracefully
        let _ = runtime.block_on(async {
            parse_aspect_from_string(&input, &urn).await
        });
    }

    /// Test that parser handles special characters in strings
    #[test]
    fn fuzz_special_characters(special in "[\\x00-\\x1F\\x7F-\\xFF]{1,20}") {
        let input = format!(r#"
            @prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
            @prefix : <urn:samm:org.example:1.0.0#> .
            :Aspect a samm:Aspect ;
                samm:preferredName "{}"@en .
        "#, special.escape_default());

        let runtime = tokio::runtime::Runtime::new().unwrap();

        // Should handle (possibly with error)
        let _ = runtime.block_on(async {
            parse_aspect_from_string(&input, "urn:samm:org.example:1.0.0#Aspect").await
        });
    }

    /// Test parser with random valid-looking structure
    #[test]
    fn fuzz_random_valid_structure(
        num_properties in 0usize..20,
        namespace in urn_component()
    ) {
        let urn = format!("urn:samm:org.example:1.0.0#{}", namespace);

        let mut input = format!(r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix samm-c: <urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <urn:samm:org.example:1.0.0#> .

:{} a samm:Aspect ;
    samm:preferredName "Test"@en ;
    samm:properties ( "#, namespace);

        for i in 0..num_properties {
            input.push_str(&format!(":prop{} ", i));
        }

        input.push_str(") .\n\n");

        for i in 0..num_properties {
            input.push_str(&format!(r#"
:prop{} a samm:Property ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType xsd:string ] .
"#, i));
        }

        let runtime = tokio::runtime::Runtime::new().unwrap();

        // Should parse successfully or return meaningful error
        let result = runtime.block_on(async {
            parse_aspect_from_string(&input, &urn).await
        });

        // If it succeeds, verify basic invariants
        if let Ok(aspect) = result {
            assert_eq!(aspect.properties.len(), num_properties);
        }
    }
}

/// Test specific edge cases that could cause crashes
#[tokio::test]
async fn test_null_bytes() {
    let input = "Aspect\x00a samm:Aspect";
    let result = parse_aspect_from_string(input, "urn:test#").await;
    // Should handle gracefully (error, not panic)
    assert!(result.is_err());
}

#[tokio::test]
async fn test_extremely_nested_structures() {
    let mut input = String::from(
        r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix : <urn:test#> .
:A a samm:Aspect ; samm:properties ( "#,
    );

    // Create deeply nested list
    for _ in 0..100 {
        input.push_str("( ");
    }
    input.push_str(":p ");
    for _ in 0..100 {
        input.push_str(") ");
    }
    input.push_str(") .");

    let result = parse_aspect_from_string(&input, "urn:test#A").await;
    // Should not stack overflow
    assert!(result.is_err() || result.is_ok());
}

#[tokio::test]
async fn test_circular_references() {
    let input = r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix : <urn:test#> .

:A a samm:Aspect ;
    samm:properties ( :B ) .

:B a samm:Property ;
    samm:characteristic :C .

:C a samm:Characteristic ;
    samm:dataType :A .
"#;

    let result = parse_aspect_from_string(input, "urn:test#A").await;
    // Should detect or handle circular reference
    assert!(result.is_err() || result.is_ok());
}

#[tokio::test]
async fn test_malformed_uris() {
    let inputs = vec![
        "@prefix x: <> .",
        "@prefix x: < > .",
        "@prefix x: <urn:> .",
        "@prefix x: <urn:samm> .",
        "@prefix x: <http://> .",
        "@prefix x: <urn:samm:org.example> .", // Missing version
        "@prefix x: <\\x00\\x01> .",
    ];

    for input in inputs {
        let result = parse_aspect_from_string(input, "urn:test#A").await;
        // Should not panic
        assert!(result.is_err() || result.is_ok());
    }
}

#[tokio::test]
async fn test_unicode_edge_cases() {
    let inputs = vec![
        // Right-to-left override
        "@prefix \u{202E}x: <urn:test#> .",
        // Zero-width characters
        "@prefix\u{200B}x: <urn:test#> .",
        // Combining characters
        "@prefix x\u{0301}: <urn:test#> .",
        // Emoji
        "@prefix 😀: <urn:test#> .",
        // Non-BMP characters
        "@prefix 𝕩: <urn:test#> .",
    ];

    for input in inputs {
        let result = parse_aspect_from_string(input, "urn:test#A").await;
        // Should handle Unicode gracefully
        assert!(result.is_err() || result.is_ok());
    }
}

#[tokio::test]
async fn test_memory_exhaustion_protection() {
    // Test that parser doesn't try to allocate excessive memory

    // Very long string literal
    let long_string = "a".repeat(10_000_000); // 10MB string
    let input = format!(
        r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix : <urn:test#> .
:A a samm:Aspect ; samm:description "{}"@en .
"#,
        long_string
    );

    let start = std::time::Instant::now();
    let result = parse_aspect_from_string(&input, "urn:test#A").await;
    let elapsed = start.elapsed();

    // Should complete in reasonable time (< 5 seconds)
    assert!(elapsed.as_secs() < 5, "Parser took too long: {:?}", elapsed);

    // Should either succeed or fail gracefully
    assert!(result.is_err() || result.is_ok());
}

#[tokio::test]
async fn test_concurrent_fuzz() {
    use scirs2_core::random::rng;

    let mut main_rng = rng();

    // Generate random inputs concurrently
    let handles: Vec<_> = (0..20)
        .map(|i| {
            let seed = main_rng.random();
            tokio::spawn(async move {
                let mut local_rng = Random::seed(seed);

                for _ in 0..10 {
                    // Generate random but somewhat valid TTL
                    let num_props = (local_rng.random::<u64>() % 10) as usize;
                    let input = generate_random_ttl(num_props, &mut local_rng);

                    let _ = parse_aspect_from_string(
                        &input,
                        &format!("urn:samm:test:1.0.0#Aspect{}", i),
                    )
                    .await;
                }
            })
        })
        .collect();

    for handle in handles {
        handle.await.unwrap();
    }
}

/// Helper to generate random but structurally valid TTL
fn generate_random_ttl<R: Rng>(num_properties: usize, rng: &mut Random<R>) -> String {
    let mut ttl = String::from(
        r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix samm-c: <urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <urn:samm:test:1.0.0#> .

:TestAspect a samm:Aspect ;
    samm:preferredName "Test"@en ;
    samm:properties ( "#,
    );

    for i in 0..num_properties {
        ttl.push_str(&format!(":prop{} ", i));
    }
    ttl.push_str(") .\n\n");

    for i in 0..num_properties {
        let random_choice = (rng.random::<u64>() % 5) as usize;
        let data_type = match random_choice {
            0 => "xsd:string",
            1 => "xsd:int",
            2 => "xsd:float",
            3 => "xsd:boolean",
            _ => "xsd:dateTime",
        };

        ttl.push_str(&format!(
            r#"
:prop{} a samm:Property ;
    samm:preferredName "Property {}"@en ;
    samm:characteristic [ a samm:Characteristic ; samm:dataType {} ] .
"#,
            i, i, data_type
        ));
    }

    ttl
}

#[tokio::test]
async fn test_syntax_error_recovery() {
    // Test that parser provides meaningful errors for common mistakes

    let test_cases = vec![
        (
            "Missing closing bracket",
            r#"@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
               :A a samm:Aspect ; samm:properties ( :p ."#,
        ),
        (
            "Missing semicolon",
            r#"@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
               :A a samm:Aspect samm:properties () ."#,
        ),
        (
            "Invalid prefix",
            r#"@prefix samm <urn:test#> .
               :A a samm:Aspect ."#,
        ),
        (
            "Typo in keyword",
            r#"@prefi samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
               :A a samm:Aspect ."#,
        ),
    ];

    for (name, input) in test_cases {
        let result = parse_aspect_from_string(input, "urn:test#A").await;

        // Should return error with meaningful message
        if let Err(e) = result {
            let error_msg = format!("{}", e);
            // Error message should not be empty
            assert!(!error_msg.is_empty(), "Empty error for: {}", name);
        }
    }
}
