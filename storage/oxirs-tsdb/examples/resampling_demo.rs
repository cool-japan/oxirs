//! Time Resampling Demonstration
//!
//! Demonstrates time bucketing and downsampling operations:
//! - Second/Minute/Hour/Day bucketing
//! - Various aggregation functions per bucket
//! - Gap filling strategies
//! - Bucket alignment
//!
//! # Usage
//!
//! ```bash
//! cargo run --example resampling_demo
//! ```

use chrono::{Duration, Timelike, Utc};
use oxirs_tsdb::{
    query::resample::{FillMethod, Resampler},
    Aggregation, DataPoint, ResampleBucket,
};

fn main() {
    println!("=== OxiRS Time Resampling Demo ===\n");

    // Demo 1: Basic resampling
    demo_basic_resampling();

    // Demo 2: Different aggregation functions
    demo_aggregation_types();

    // Demo 3: Gap filling strategies
    demo_gap_filling();

    // Demo 4: Multi-level downsampling
    demo_multi_level_downsampling();

    // Demo 5: Real-world IoT data processing
    demo_iot_data_pipeline();

    println!("\n=== Resampling Demo Complete ===");
}

fn demo_basic_resampling() {
    println!("Demo 1: Basic Resampling");
    println!("------------------------");

    // Create 1Hz data for 5 minutes
    let base_time = Utc::now()
        .with_second(0)
        .unwrap()
        .with_nanosecond(0)
        .unwrap();

    let data: Vec<DataPoint> = (0..300)
        .map(|i| DataPoint {
            timestamp: base_time + Duration::seconds(i),
            value: 20.0 + (i as f64 / 60.0) + (i % 7) as f64 * 0.3,
        })
        .collect();

    println!("Input: {} points at 1Hz (5 minutes of data)", data.len());
    println!();

    // Resample to 1-minute buckets
    let result = Resampler::new(ResampleBucket::Minute(1), Aggregation::Avg)
        .resample(&data)
        .unwrap();

    println!("Resampled to 1-minute averages:");
    println!("  Output: {} buckets", result.len());
    println!();

    for (i, point) in result.iter().enumerate() {
        println!("  Minute {}: {:.2} (avg of 60 points)", i, point.value);
    }

    // Different bucket sizes
    println!();
    println!("Comparison of bucket sizes:");

    let buckets = [
        ("10 seconds", ResampleBucket::Second(10)),
        ("30 seconds", ResampleBucket::Second(30)),
        ("1 minute", ResampleBucket::Minute(1)),
        ("5 minutes", ResampleBucket::Minute(5)),
    ];

    for (name, bucket) in buckets {
        let result = Resampler::new(bucket, Aggregation::Avg)
            .resample(&data)
            .unwrap();
        let reduction = data.len() as f64 / result.len().max(1) as f64;
        println!(
            "  {}: {} buckets ({:.1}:1 reduction)",
            name,
            result.len(),
            reduction
        );
    }
    println!();
}

fn demo_aggregation_types() {
    println!("Demo 2: Different Aggregation Functions");
    println!("---------------------------------------");

    let base_time = Utc::now()
        .with_second(0)
        .unwrap()
        .with_nanosecond(0)
        .unwrap();

    // Create data with clear patterns for each minute
    let data: Vec<DataPoint> = (0..180)
        .map(|i| {
            let minute = i / 60;
            let base = match minute {
                0 => 10.0,  // Low values
                1 => 50.0,  // Medium values
                _ => 100.0, // High values
            };
            let variation = (i % 60) as f64 - 30.0; // -30 to +29
            DataPoint {
                timestamp: base_time + Duration::seconds(i),
                value: base + variation,
            }
        })
        .collect();

    println!("Input: 3 minutes of data with distinct patterns");
    println!("  Minute 0: Values around 10 (range: -20 to 39)");
    println!("  Minute 1: Values around 50 (range: 20 to 79)");
    println!("  Minute 2: Values around 100 (range: 70 to 129)");
    println!();

    let aggregations = [
        ("AVG", Aggregation::Avg),
        ("MIN", Aggregation::Min),
        ("MAX", Aggregation::Max),
        ("SUM", Aggregation::Sum),
        ("COUNT", Aggregation::Count),
        ("FIRST", Aggregation::First),
        ("LAST", Aggregation::Last),
    ];

    println!(
        "  {:>8} | {:>10} | {:>10} | {:>10}",
        "Function", "Minute 0", "Minute 1", "Minute 2"
    );
    println!("  {:-<8}-+-{:-<10}-+-{:-<10}-+-{:-<10}", "", "", "", "");

    for (name, agg) in aggregations {
        let result = Resampler::new(ResampleBucket::Minute(1), agg)
            .resample(&data)
            .unwrap();

        println!(
            "  {:>8} | {:>10.1} | {:>10.1} | {:>10.1}",
            name,
            result.first().map(|p| p.value).unwrap_or(0.0),
            result.get(1).map(|p| p.value).unwrap_or(0.0),
            result.get(2).map(|p| p.value).unwrap_or(0.0),
        );
    }
    println!();
}

fn demo_gap_filling() {
    println!("Demo 3: Gap Filling Strategies");
    println!("------------------------------");

    let base_time = Utc::now()
        .with_second(0)
        .unwrap()
        .with_nanosecond(0)
        .unwrap();

    // Create data with gaps
    let data: Vec<DataPoint> = vec![
        DataPoint {
            timestamp: base_time,
            value: 10.0,
        },
        DataPoint {
            timestamp: base_time + Duration::seconds(10),
            value: 15.0,
        },
        // Gap: 20s, 30s, 40s, 50s missing
        DataPoint {
            timestamp: base_time + Duration::minutes(1),
            value: 50.0,
        },
        DataPoint {
            timestamp: base_time + Duration::minutes(1) + Duration::seconds(10),
            value: 55.0,
        },
        // Gap: 1:20, 1:30, 1:40, 1:50 missing
        DataPoint {
            timestamp: base_time + Duration::minutes(2),
            value: 20.0,
        },
    ];

    println!("Input data (with gaps):");
    for point in &data {
        let offset = point
            .timestamp
            .signed_duration_since(base_time)
            .num_seconds();
        println!("  t={:>3}s: {:.1}", offset, point.value);
    }
    println!();

    // Different fill methods
    let fill_methods = [
        ("None (gaps)", FillMethod::None),
        ("Previous value", FillMethod::Previous),
        ("Next value", FillMethod::Next),
        ("Linear interpolation", FillMethod::Linear),
        ("Constant (0.0)", FillMethod::Constant(0.0)),
    ];

    for (name, method) in fill_methods {
        let result = Resampler::new(ResampleBucket::Second(10), Aggregation::Avg)
            .with_fill(method)
            .resample(&data)
            .unwrap();

        print!("{}: ", name);
        let values: Vec<String> = result.iter().map(|p| format!("{:.0}", p.value)).collect();
        println!("[{}]", values.join(", "));
    }
    println!();
}

fn demo_multi_level_downsampling() {
    println!("Demo 4: Multi-Level Downsampling");
    println!("--------------------------------");
    println!("Typical IoT data retention strategy:");
    println!("  - Raw: 1 second resolution, 1 day retention");
    println!("  - Level 1: 1 minute averages, 1 week retention");
    println!("  - Level 2: 1 hour averages, 1 month retention");
    println!("  - Level 3: 1 day averages, 1 year retention");
    println!();

    // Simulate 1 hour of raw data at 1Hz
    let base_time = Utc::now()
        .with_minute(0)
        .unwrap()
        .with_second(0)
        .unwrap()
        .with_nanosecond(0)
        .unwrap();

    let raw_data: Vec<DataPoint> = (0..3600)
        .map(|i| {
            let hour_cycle = (i as f64 / 3600.0 * std::f64::consts::PI * 2.0).sin();
            let value = 50.0 + 10.0 * hour_cycle + (i % 13) as f64 * 0.2;
            DataPoint {
                timestamp: base_time + Duration::seconds(i),
                value,
            }
        })
        .collect();

    println!("Raw data: {} points (1 hour at 1Hz)", raw_data.len());
    println!(
        "Storage: {} bytes (16 bytes per point)",
        raw_data.len() * 16
    );
    println!();

    // Level 1: 1-minute averages
    let level1 = Resampler::new(ResampleBucket::Minute(1), Aggregation::Avg)
        .resample(&raw_data)
        .unwrap();

    println!(
        "Level 1 (1-minute): {} points ({:.1}:1 reduction)",
        level1.len(),
        raw_data.len() as f64 / level1.len() as f64
    );

    // Level 2: 5-minute averages
    let level2 = Resampler::new(ResampleBucket::Minute(5), Aggregation::Avg)
        .resample(&raw_data)
        .unwrap();

    println!(
        "Level 2 (5-minute): {} points ({:.1}:1 reduction)",
        level2.len(),
        raw_data.len() as f64 / level2.len() as f64
    );

    // Level 3: 15-minute averages
    let level3 = Resampler::new(ResampleBucket::Minute(15), Aggregation::Avg)
        .resample(&raw_data)
        .unwrap();

    println!(
        "Level 3 (15-minute): {} points ({:.1}:1 reduction)",
        level3.len(),
        raw_data.len() as f64 / level3.len() as f64
    );

    // Level 4: Hourly average
    let level4 = Resampler::new(ResampleBucket::Hour(1), Aggregation::Avg)
        .resample(&raw_data)
        .unwrap();

    println!(
        "Level 4 (1-hour): {} point ({:.1}:1 reduction)",
        level4.len(),
        raw_data.len() as f64 / level4.len().max(1) as f64
    );

    println!();
    println!("Sample comparison (first few values):");
    println!(
        "  {:>12} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Time", "Raw", "1-min", "5-min", "15-min"
    );
    println!(
        "  {:-<12}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}",
        "", "", "", "", ""
    );

    // Show values at 0, 5, 10, 15 minutes
    let checkpoints = [0, 5, 10, 15];
    for &min in &checkpoints {
        let raw_idx = min * 60;
        let l1_idx = min;
        let l2_idx = min / 5;
        let l3_idx = min / 15;

        let raw = raw_data.get(raw_idx).map(|p| p.value).unwrap_or(0.0);
        let l1 = level1.get(l1_idx).map(|p| p.value).unwrap_or(0.0);
        let l2 = level2.get(l2_idx).map(|p| p.value).unwrap_or(0.0);
        let l3 = level3.get(l3_idx).map(|p| p.value).unwrap_or(0.0);

        println!(
            "  {:>10}m | {:>10.2} | {:>10.2} | {:>10.2} | {:>10.2}",
            min, raw, l1, l2, l3
        );
    }
    println!();
}

fn demo_iot_data_pipeline() {
    println!("Demo 5: Real-World IoT Data Pipeline");
    println!("-------------------------------------");
    println!("Scenario: Factory sensor data aggregation");
    println!();

    let base_time = Utc::now()
        .with_hour(8)
        .unwrap()
        .with_minute(0)
        .unwrap()
        .with_second(0)
        .unwrap()
        .with_nanosecond(0)
        .unwrap();

    // Simulate 8-hour shift of machine temperature data
    let shift_duration = 8 * 3600; // 8 hours in seconds
    let raw_data: Vec<DataPoint> = (0..shift_duration)
        .map(|i| {
            let hour = i / 3600;
            // Temperature pattern: warm-up, production, cool-down
            let base_temp = match hour {
                0 => 20.0 + (i % 3600) as f64 / 3600.0 * 40.0, // Warm-up
                1..=6 => 60.0 + (hour as f64 * 2.0),           // Production (rising)
                7 => 72.0 - (i % 3600) as f64 / 3600.0 * 20.0, // Cool-down
                _ => 50.0,
            };
            let noise = ((i * 17) % 10) as f64 * 0.5 - 2.5;
            DataPoint {
                timestamp: base_time + Duration::seconds(i as i64),
                value: base_temp + noise,
            }
        })
        .collect();

    println!("Raw data: {} points (8-hour shift at 1Hz)", raw_data.len());

    // Generate hourly report
    let hourly = Resampler::new(ResampleBucket::Hour(1), Aggregation::Avg)
        .resample(&raw_data)
        .unwrap();

    let hourly_min = Resampler::new(ResampleBucket::Hour(1), Aggregation::Min)
        .resample(&raw_data)
        .unwrap();

    let hourly_max = Resampler::new(ResampleBucket::Hour(1), Aggregation::Max)
        .resample(&raw_data)
        .unwrap();

    println!();
    println!("Hourly Temperature Report:");
    println!(
        "  {:>4} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Hour", "Avg", "Min", "Max", "Range"
    );
    println!(
        "  {:-<4}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}",
        "", "", "", "", ""
    );

    for i in 0..hourly.len() {
        let avg = hourly[i].value;
        let min = hourly_min[i].value;
        let max = hourly_max[i].value;
        let range = max - min;
        let hour = 8 + i;

        println!(
            "  {:>02}:00 | {:>8.1}degC | {:>8.1}degC | {:>8.1}degC | {:>8.1}degC",
            hour, avg, min, max, range
        );
    }

    // Calculate shift statistics
    let shift_min = hourly_min
        .iter()
        .map(|p| p.value)
        .fold(f64::INFINITY, f64::min);
    let shift_max = hourly_max
        .iter()
        .map(|p| p.value)
        .fold(f64::NEG_INFINITY, f64::max);
    let shift_avg: f64 = hourly.iter().map(|p| p.value).sum::<f64>() / hourly.len() as f64;

    println!();
    println!("Shift Summary:");
    println!("  Average temperature: {:.1}degC", shift_avg);
    println!(
        "  Temperature range: {:.1}degC - {:.1}degC",
        shift_min, shift_max
    );
    println!("  Data points processed: {}", raw_data.len());
    println!("  Summary points generated: {}", hourly.len());
    println!();
}
