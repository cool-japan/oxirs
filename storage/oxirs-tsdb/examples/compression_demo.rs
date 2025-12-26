//! Time-series compression demonstration
//!
//! This example demonstrates the Gorilla and Delta-of-delta compression
//! algorithms and their compression ratios for different data patterns.

use oxirs_tsdb::{DeltaOfDeltaCompressor, GorillaCompressor};

fn main() {
    println!("=== OxiRS Time-Series Compression Demo ===\n");

    // Demo 1: Stable temperature sensor (constant value)
    demo_stable_sensor();

    // Demo 2: Regular 1Hz sampling (predictable timestamps)
    demo_regular_sampling();

    // Demo 3: Temperature sensor with small variations
    demo_temperature_variations();

    println!("\n✅ Compression demo completed!");
}

fn demo_stable_sensor() {
    println!("Demo 1: Stable Temperature Sensor (constant 22.5°C)");
    println!("------------------------------------------------");

    let mut compressor = GorillaCompressor::new(22.5);
    for _ in 0..1000 {
        compressor.compress(22.5); // Constant temperature
    }

    let ratio = compressor.compression_ratio(1001);
    let compressed = compressor.finish();

    println!("  Data points: 1,001");
    println!("  Uncompressed: {} bytes (8 bytes per f64)", 1001 * 8);
    println!("  Compressed: {} bytes", compressed.len());
    println!("  Compression ratio: {:.1}:1", ratio);
    println!("  ✓ Excellent compression for stable values\n");
}

fn demo_regular_sampling() {
    println!("Demo 2: Regular 1Hz Sampling (timestamps)");
    println!("------------------------------------------");

    let base_ts = 1640000000000i64; // Jan 2022
    let mut compressor = DeltaOfDeltaCompressor::new(base_ts);

    for i in 1..1000 {
        compressor.compress(base_ts + i * 1000); // Exactly 1 second intervals
    }

    let ratio = compressor.compression_ratio(1000);
    let compressed = compressor.finish();

    println!("  Timestamps: 1,000");
    println!("  Interval: 1 second (regular)");
    println!("  Uncompressed: {} bytes (8 bytes per i64)", 1000 * 8);
    println!("  Compressed: {} bytes", compressed.len());
    println!("  Compression ratio: {:.1}:1", ratio);
    println!("  ✓ Excellent compression for regular sampling\n");
}

fn demo_temperature_variations() {
    println!("Demo 3: Temperature with Small Variations");
    println!("------------------------------------------");

    let mut compressor = GorillaCompressor::new(20.0);

    // Simulate temperature fluctuations (±2°C around 20°C)
    for i in 0..1000 {
        let temp = 20.0 + (i % 10) as f64 * 0.2 - 1.0; // 19-21°C range
        compressor.compress(temp);
    }

    let ratio = compressor.compression_ratio(1001);
    let compressed = compressor.finish();

    println!("  Data points: 1,001");
    println!("  Temperature range: 19.0°C - 21.0°C");
    println!("  Uncompressed: {} bytes", 1001 * 8);
    println!("  Compressed: {} bytes", compressed.len());
    println!("  Compression ratio: {:.1}:1", ratio);
    println!("  ✓ Good compression even with variations\n");
}
