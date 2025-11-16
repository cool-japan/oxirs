//! SIMD-Accelerated URN Processing Demo
//!
//! This example demonstrates the performance benefits of SIMD-accelerated string operations
//! for processing SAMM URNs. It compares standard iteration with SIMD operations and shows
//! batch processing capabilities.
//!
//! Run this example with:
//! ```bash
//! cargo run --example simd_performance_demo --release
//! ```
//!
//! Note: Run in release mode to see real performance benefits!

use oxirs_samm::simd_ops::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SIMD-Accelerated URN Processing Demo\n");
    println!("This example demonstrates the performance benefits of SIMD operations");
    println!("for processing SAMM URNs in large-scale scenarios.");
    println!("{}", "=".repeat(80));
    println!();

    // Example 1: URN Validation
    example_1_urn_validation()?;

    // Example 2: Character Counting Performance
    example_2_character_counting()?;

    // Example 3: Batch URN Extraction
    example_3_batch_extraction()?;

    // Example 4: Finding URNs in Documentation
    example_4_find_urns_in_text()?;

    // Example 5: Performance Comparison
    example_5_performance_comparison()?;

    println!();
    println!("{}", "=".repeat(80));
    println!("✅ All examples completed successfully!");
    println!("💡 Tip: Run with --release for realistic performance measurements");

    Ok(())
}

/// Example 1: URN Validation
fn example_1_urn_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Example 1: URN Validation");
    println!("{}", "-".repeat(80));

    let urns = vec![
        "urn:samm:org.example:1.0.0#MyAspect",
        "urn:samm:com.company.product:2.1.3#Temperature",
        "invalid-urn",
        "urn:samm:io.sample:1.5.0#Characteristic",
        "urn:bamm:org.old:1.0.0#Legacy", // Wrong prefix
    ];

    println!("Validating {} URNs...", urns.len());
    let results = validate_urns_batch(&urns)?;

    for (i, (urn, valid)) in urns.iter().zip(results.iter()).enumerate() {
        let status = if *valid { "✓ VALID" } else { "✗ INVALID" };
        println!("  {}. {} - {}", i + 1, status, urn);
    }

    let valid_count = results.iter().filter(|&&v| v).count();
    println!("\n  Result: {}/{} URNs are valid", valid_count, urns.len());
    println!();

    Ok(())
}

/// Example 2: Character Counting Performance
fn example_2_character_counting() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔢 Example 2: SIMD Character Counting");
    println!("{}", "-".repeat(80));

    let sample_urn = "urn:samm:org.example.domain.subdomain:1.0.0#MyAspect";

    println!("Analyzing URN: {}", sample_urn);
    println!();

    // Count different characters
    let colon_count = count_char_simd(sample_urn, ':');
    let dot_count = count_char_simd(sample_urn, '.');
    let hash_count = count_char_simd(sample_urn, '#');

    println!("  Character counts:");
    println!("    Colons (:)  : {}", colon_count);
    println!("    Dots (.)    : {}", dot_count);
    println!("    Hashes (#)  : {}", hash_count);
    println!();

    // Validate based on character counts
    let is_valid = colon_count == 3 && hash_count == 1;
    println!(
        "  Quick validation: {}",
        if is_valid {
            "✓ URN structure looks valid"
        } else {
            "✗ URN structure invalid"
        }
    );
    println!();

    Ok(())
}

/// Example 3: Batch URN Extraction
fn example_3_batch_extraction() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Example 3: Batch URN Part Extraction");
    println!("{}", "-".repeat(80));

    let urns = vec![
        "urn:samm:org.example:1.0.0#VehicleStatus",
        "urn:samm:com.company:2.0.0#Temperature",
        "urn:samm:io.sample.domain:1.5.0#Pressure",
        "urn:samm:net.test:3.0.0#Characteristic",
    ];

    println!(
        "Extracting components from {} URNs using parallel processing...",
        urns.len()
    );
    println!();

    let start = Instant::now();
    let results = extract_urn_parts_batch(&urns)?;
    let duration = start.elapsed();

    for (urn, parts) in urns.iter().zip(results.iter()) {
        if let Some((namespace, version, element)) = parts {
            println!("  URN: {}", urn);
            println!("    ├─ Namespace: {}", namespace);
            println!("    ├─ Version:   {}", version);
            println!("    └─ Element:   {}", element);
            println!();
        }
    }

    println!("  ⚡ Extraction time: {:?}", duration);
    println!();

    Ok(())
}

/// Example 4: Finding URNs in Documentation
fn example_4_find_urns_in_text() -> Result<(), Box<dyn std::error::Error>> {
    println!("📄 Example 4: Finding URNs in Documentation");
    println!("{}", "-".repeat(80));

    let documentation = r#"
        # Vehicle Status Aspect Model

        The VehicleStatus aspect (urn:samm:org.example.vehicle:1.0.0#VehicleStatus)
        provides real-time telemetry data from connected vehicles.

        ## Properties

        - engineTemperature (urn:samm:org.example.vehicle:1.0.0#engineTemperature)
          Measurement of engine temperature in Celsius.

        - fuelLevel (urn:samm:org.example.vehicle:1.0.0#fuelLevel)
          Current fuel level as percentage (0-100).

        - speed (urn:samm:org.example.vehicle:1.0.0#speed)
          Current vehicle speed in km/h.

        ## Related Models

        See also the diagnostic aspect: urn:samm:org.example.vehicle:1.0.0#Diagnostics
    "#;

    println!("Scanning documentation for SAMM URNs...");
    println!();

    let start = Instant::now();
    let found_urns = find_urns_in_text(documentation)?;
    let duration = start.elapsed();

    println!("  Found {} URNs in {:?}:", found_urns.len(), duration);
    println!();

    for (i, urn) in found_urns.iter().enumerate() {
        println!("  {}. {}", i + 1, urn);

        // Extract and display parts
        if let Some(namespace) = extract_namespace_fast(urn) {
            if let Some(element) = extract_element_fast(urn) {
                println!("     └─ {}.{}", namespace, element);
            }
        }
    }
    println!();

    Ok(())
}

/// Example 5: Performance Comparison
fn example_5_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Example 5: Performance Comparison (SIMD vs Standard)");
    println!("{}", "-".repeat(80));

    // Generate test data
    let test_sizes = vec![10, 100, 1000, 10000];

    println!("Comparing batch URN validation performance...");
    println!();

    for size in test_sizes {
        // Generate URNs
        let urns: Vec<String> = (0..size)
            .map(|i| format!("urn:samm:org.example.test{}:1.0.0#Property{}", i % 10, i))
            .collect();

        let urn_refs: Vec<&str> = urns.iter().map(|s| s.as_str()).collect();

        // SIMD validation (with parallel processing for large batches)
        let start = Instant::now();
        let _results = validate_urns_batch(&urn_refs)?;
        let simd_duration = start.elapsed();

        // Standard validation (sequential)
        let start = Instant::now();
        let _standard_results: Vec<bool> = urn_refs
            .iter()
            .map(|urn| {
                // Simple validation
                urn.starts_with("urn:samm:") && urn.contains('#') && urn.contains(':')
            })
            .collect();
        let standard_duration = start.elapsed();

        let speedup = standard_duration.as_secs_f64() / simd_duration.as_secs_f64();

        println!("  Batch size: {:>5} URNs", size);
        println!("    Standard:  {:>8?}", standard_duration);
        println!("    SIMD:      {:>8?}", simd_duration);
        println!("    Speedup:   {:.2}x faster", speedup);
        println!();
    }

    println!("  💡 Note: Speedup increases with larger batches due to parallel processing");
    println!();

    // Character counting comparison
    println!("Comparing character counting performance...");
    println!();

    let large_text: String = "urn:samm:org.example:1.0.0#Test".repeat(1000);

    // SIMD counting
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = count_char_simd(&large_text, ':');
    }
    let simd_duration = start.elapsed();

    // Standard counting
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = large_text.chars().filter(|&c| c == ':').count();
    }
    let standard_duration = start.elapsed();

    let speedup = standard_duration.as_secs_f64() / simd_duration.as_secs_f64();

    println!("  Character counting (1000 iterations on 32KB text):");
    println!("    Standard:  {:>8?}", standard_duration);
    println!("    SIMD:      {:>8?}", simd_duration);
    println!("    Speedup:   {:.2}x faster", speedup);
    println!();

    println!("  ✨ SIMD operations provide significant performance improvements!");
    println!();

    Ok(())
}
