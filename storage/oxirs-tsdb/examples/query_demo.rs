//! Query Engine Demonstration
//!
//! Demonstrates the query engine capabilities including:
//! - Range queries
//! - Aggregations (AVG, MIN, MAX, SUM, COUNT)
//! - Time-based filters
//! - Limiting and ordering results
//!
//! # Usage
//!
//! ```bash
//! cargo run --example query_demo
//! ```

use chrono::{Duration, Utc};
use oxirs_tsdb::{
    Aggregation, DataPoint, QueryEngine, ResampleBucket, TimeChunk, WindowFunction, WindowSpec,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS Time-Series Query Engine Demo ===\n");

    // Create query engine and load test data
    let engine = create_test_engine()?;

    // Demo 1: Basic range query
    demo_range_query(&engine)?;

    // Demo 2: Aggregation queries
    demo_aggregations(&engine)?;

    // Demo 3: Window functions
    demo_window_functions(&engine)?;

    // Demo 4: Resampling (downsampling)
    demo_resampling(&engine)?;

    // Demo 5: Combined query features
    demo_combined_query(&engine)?;

    println!("\n=== Query Demo Complete ===");
    Ok(())
}

/// Create a query engine with simulated sensor data
fn create_test_engine() -> Result<QueryEngine, Box<dyn std::error::Error>> {
    let mut engine = QueryEngine::new();
    let base_time = Utc::now() - Duration::hours(2);

    // Simulate temperature sensor (Series ID: 1)
    // 7200 points at 1Hz = 2 hours of data
    let temp_points: Vec<DataPoint> = (0..7200)
        .map(|i| {
            // Temperature with daily cycle pattern (15-25 degC)
            let hour_fraction = (i as f64 / 3600.0) * std::f64::consts::PI;
            let temp = 20.0 + 5.0 * hour_fraction.sin() + (i % 10) as f64 * 0.05;
            DataPoint {
                timestamp: base_time + Duration::seconds(i),
                value: temp,
            }
        })
        .collect();

    let temp_chunk = TimeChunk::new(1, base_time, Duration::hours(2), temp_points)?;
    engine.add_chunk(temp_chunk);

    // Simulate humidity sensor (Series ID: 2)
    let humidity_points: Vec<DataPoint> = (0..7200)
        .map(|i| {
            let humidity = 50.0 + (i % 20) as f64 * 0.5 - 5.0;
            DataPoint {
                timestamp: base_time + Duration::seconds(i),
                value: humidity,
            }
        })
        .collect();

    let humidity_chunk = TimeChunk::new(2, base_time, Duration::hours(2), humidity_points)?;
    engine.add_chunk(humidity_chunk);

    // Simulate power consumption (Series ID: 3) - more variable
    let power_points: Vec<DataPoint> = (0..7200)
        .map(|i| {
            let base_power = 100.0;
            let spike = if i % 300 < 30 { 50.0 } else { 0.0 }; // Power spikes every 5 minutes
            let noise = (i % 7) as f64 * 2.0;
            DataPoint {
                timestamp: base_time + Duration::seconds(i),
                value: base_power + spike + noise,
            }
        })
        .collect();

    let power_chunk = TimeChunk::new(3, base_time, Duration::hours(2), power_points)?;
    engine.add_chunk(power_chunk);

    println!("Loaded 3 sensor series with 7200 points each (2 hours at 1Hz)\n");
    Ok(engine)
}

fn demo_range_query(engine: &QueryEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 1: Range Queries");
    println!("---------------------");

    // Query all data for temperature sensor
    let result = engine.query().series(1).execute()?;

    println!("Full Query (Series 1 - Temperature):");
    println!("  Total points: {}", result.points.len());
    println!("  Chunks scanned: {}", result.chunks_scanned);
    println!("  Execution time: {}ms", result.execution_time_ms);

    // Query last 30 minutes
    let result = engine
        .query()
        .series(1)
        .last(Duration::minutes(30))
        .execute()?;

    println!("\nLast 30 minutes:");
    println!("  Points: {}", result.points.len());
    if let Some(first) = result.points.first() {
        println!("  First value: {:.2} degC", first.value);
    }
    if let Some(last) = result.points.last() {
        println!("  Last value: {:.2} degC", last.value);
    }

    // Query with limit (get latest 10 readings)
    let result = engine.query().series(1).descending().limit(10).execute()?;

    println!("\nLatest 10 readings (descending):");
    for (i, point) in result.points.iter().take(5).enumerate() {
        println!(
            "  {}: {:.2} degC at {}",
            i + 1,
            point.value,
            point.timestamp
        );
    }
    println!("  ... and {} more", result.points.len().saturating_sub(5));

    println!();
    Ok(())
}

fn demo_aggregations(engine: &QueryEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 2: Aggregation Queries");
    println!("---------------------------");

    // Temperature statistics
    println!("Temperature Sensor Statistics (last 2 hours):");

    let avg = engine
        .query()
        .series(1)
        .aggregate(Aggregation::Avg)
        .execute()?;
    println!("  Average: {:.2} degC", avg.aggregated_value.unwrap_or(0.0));

    let min = engine
        .query()
        .series(1)
        .aggregate(Aggregation::Min)
        .execute()?;
    println!("  Minimum: {:.2} degC", min.aggregated_value.unwrap_or(0.0));

    let max = engine
        .query()
        .series(1)
        .aggregate(Aggregation::Max)
        .execute()?;
    println!("  Maximum: {:.2} degC", max.aggregated_value.unwrap_or(0.0));

    // Power consumption
    println!("\nPower Consumption Statistics:");

    let sum = engine
        .query()
        .series(3)
        .aggregate(Aggregation::Sum)
        .execute()?;
    println!(
        "  Total energy: {:.0} Wh (sum of instantaneous power readings)",
        sum.aggregated_value.unwrap_or(0.0)
    );

    let count = engine
        .query()
        .series(3)
        .aggregate(Aggregation::Count)
        .execute()?;
    println!(
        "  Sample count: {:.0}",
        count.aggregated_value.unwrap_or(0.0)
    );

    // Statistical aggregations
    println!("\nAdvanced Statistics (Temperature):");

    let stddev = engine
        .query()
        .series(1)
        .aggregate(Aggregation::StdDev)
        .execute()?;
    println!(
        "  Std Dev: {:.3} degC",
        stddev.aggregated_value.unwrap_or(0.0)
    );

    let variance = engine
        .query()
        .series(1)
        .aggregate(Aggregation::Variance)
        .execute()?;
    println!(
        "  Variance: {:.3} degC^2",
        variance.aggregated_value.unwrap_or(0.0)
    );

    let median = engine
        .query()
        .series(1)
        .aggregate(Aggregation::Median)
        .execute()?;
    println!(
        "  Median: {:.2} degC",
        median.aggregated_value.unwrap_or(0.0)
    );

    let p95 = engine
        .query()
        .series(1)
        .aggregate(Aggregation::Percentile(95))
        .execute()?;
    println!("  P95: {:.2} degC", p95.aggregated_value.unwrap_or(0.0));

    println!();
    Ok(())
}

fn demo_window_functions(engine: &QueryEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 3: Window Functions");
    println!("------------------------");

    // 5-minute moving average on temperature
    let result = engine
        .query()
        .series(1)
        .last(Duration::minutes(30))
        .window(WindowSpec::count_based(300, WindowFunction::MovingAverage)) // 5-min at 1Hz
        .execute()?;

    println!("5-Minute Moving Average (Temperature):");
    println!("  Input points: {}", result.points_processed);
    println!("  Output points: {}", result.points.len());
    if let Some(last) = result.points.last() {
        println!("  Latest smoothed value: {:.2} degC", last.value);
    }

    // Exponential moving average
    let result = engine
        .query()
        .series(3) // Power consumption
        .last(Duration::minutes(10))
        .window(WindowSpec::count_based(
            1,
            WindowFunction::ExponentialMovingAverage(0.1),
        ))
        .execute()?;

    println!("\nExponential Moving Average (Power, alpha=0.1):");
    println!("  Points: {}", result.points.len());
    if let Some(first) = result.points.first() {
        if let Some(last) = result.points.last() {
            println!("  First EMA: {:.2} W", first.value);
            println!("  Last EMA: {:.2} W", last.value);
        }
    }

    // Rate of change (derivative) for power
    let result = engine
        .query()
        .series(3)
        .last(Duration::minutes(5))
        .window(WindowSpec::count_based(1, WindowFunction::RateOfChange))
        .limit(10)
        .execute()?;

    println!("\nRate of Change (Power - last 5 min, first 10 changes):");
    for (i, point) in result.points.iter().take(5).enumerate() {
        let change_type = if point.value > 0.0 {
            "increase"
        } else if point.value < 0.0 {
            "decrease"
        } else {
            "stable"
        };
        println!("  {}: {:.1} W/s ({})", i + 1, point.value, change_type);
    }

    println!();
    Ok(())
}

fn demo_resampling(engine: &QueryEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 4: Resampling (Downsampling)");
    println!("---------------------------------");

    // Resample temperature to 5-minute averages
    let result = engine
        .query()
        .series(1)
        .last(Duration::hours(1))
        .resample(ResampleBucket::Minute(5))
        .execute()?;

    println!("Temperature - 5-minute averages:");
    println!("  Input: 3600 points (1 hour at 1Hz)");
    println!("  Output: {} points (5-min buckets)", result.points.len());
    println!("  Reduction: {}:1", 3600 / result.points.len().max(1));

    for (i, point) in result.points.iter().take(6).enumerate() {
        println!("    Bucket {}: {:.2} degC", i, point.value);
    }
    println!(
        "    ... and {} more buckets",
        result.points.len().saturating_sub(6)
    );

    // Resample to hourly statistics
    let result = engine
        .query()
        .series(3) // Power
        .resample(ResampleBucket::Hour(1))
        .execute()?;

    println!("\nPower Consumption - Hourly averages:");
    println!("  Output buckets: {}", result.points.len());
    for (i, point) in result.points.iter().enumerate() {
        println!("    Hour {}: {:.1} W average", i, point.value);
    }

    println!();
    Ok(())
}

fn demo_combined_query(engine: &QueryEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 5: Combined Query Features");
    println!("-------------------------------");

    // Complex query: Last hour, 1-minute moving average, resampled to 5-minute buckets
    let result = engine
        .query()
        .series(1)
        .last(Duration::hours(1))
        .window(WindowSpec::count_based(60, WindowFunction::MovingAverage)) // 1-min MA
        .resample(ResampleBucket::Minute(5))
        .execute()?;

    println!("Temperature Analysis Pipeline:");
    println!("  1. Query last 1 hour");
    println!("  2. Apply 1-minute moving average");
    println!("  3. Resample to 5-minute buckets");
    println!("\nResult:");
    println!("  Chunks scanned: {}", result.chunks_scanned);
    println!("  Points processed: {}", result.points_processed);
    println!("  Final output points: {}", result.points.len());
    println!("  Execution time: {}ms", result.execution_time_ms);

    println!("\nSmoothed 5-minute averages:");
    for (i, point) in result.points.iter().take(6).enumerate() {
        println!("    {}: {:.2} degC", i, point.value);
    }

    // Another example: Peak detection query
    println!("\nPeak Detection (Max power in 10-minute windows):");

    let result = engine
        .query()
        .series(3)
        .window(WindowSpec::count_based(600, WindowFunction::MovingMax))
        .limit(12)
        .execute()?;

    for (i, point) in result.points.iter().take(6).enumerate() {
        println!("    Window {}: {:.1} W max", i, point.value);
    }

    println!();
    Ok(())
}
