//! Window Functions Demonstration
//!
//! Demonstrates time-series window functions including:
//! - Simple Moving Average (SMA)
//! - Exponential Moving Average (EMA)
//! - Moving Min/Max
//! - Rolling Standard Deviation
//! - Cumulative Sum
//! - Rate of Change (Derivative)
//!
//! # Usage
//!
//! ```bash
//! cargo run --example window_functions
//! ```

// Allow indexed loops for demo clarity
#![allow(clippy::needless_range_loop)]

use chrono::{Duration, Utc};
use oxirs_tsdb::{DataPoint, WindowFunction, WindowSpec};

fn main() {
    println!("=== OxiRS Window Functions Demo ===\n");

    // Demo 1: Simple Moving Average
    demo_moving_average();

    // Demo 2: Exponential Moving Average
    demo_exponential_moving_average();

    // Demo 3: Moving Min/Max
    demo_moving_min_max();

    // Demo 4: Rolling Standard Deviation
    demo_rolling_stddev();

    // Demo 5: Cumulative Sum
    demo_cumulative_sum();

    // Demo 6: Rate of Change
    demo_rate_of_change();

    // Demo 7: Real-world use case - Anomaly Detection
    demo_anomaly_detection();

    println!("\n=== Window Functions Demo Complete ===");
}

fn create_test_data() -> Vec<DataPoint> {
    let base_time = Utc::now();

    // Simulated sensor data with noise
    (0..100)
        .map(|i| {
            let trend = i as f64 * 0.1; // Gradual upward trend
            let cycle = (i as f64 * std::f64::consts::PI / 20.0).sin() * 3.0; // Cyclical component
            let noise = ((i * 7) % 10) as f64 * 0.3 - 1.5; // Deterministic "noise"
            DataPoint {
                timestamp: base_time + Duration::seconds(i),
                value: 50.0 + trend + cycle + noise,
            }
        })
        .collect()
}

fn demo_moving_average() {
    println!("Demo 1: Simple Moving Average (SMA)");
    println!("------------------------------------");

    let data = create_test_data();
    let spec = WindowSpec::count_based(5, WindowFunction::MovingAverage);
    let mut calculator = oxirs_tsdb::query::window::WindowCalculator::new(spec);

    let smoothed: Vec<DataPoint> = data.iter().filter_map(|p| calculator.add(*p)).collect();

    println!("Window size: 5 points");
    println!("Input points: {}", data.len());
    println!(
        "Output points: {} (first {} skipped - window warming)",
        smoothed.len(),
        data.len() - smoothed.len()
    );
    println!();
    println!("Comparison (first 10 with results):");
    println!(
        "  {:>6} | {:>10} | {:>10} | {:>10}",
        "Index", "Original", "SMA(5)", "Smoothing"
    );

    for (i, (orig, smooth)) in data
        .iter()
        .skip(4)
        .zip(smoothed.iter())
        .take(10)
        .enumerate()
    {
        let diff = smooth.value - orig.value;
        println!(
            "  {:>6} | {:>10.2} | {:>10.2} | {:>+10.2}",
            i + 4,
            orig.value,
            smooth.value,
            diff
        );
    }
    println!();
}

fn demo_exponential_moving_average() {
    println!("Demo 2: Exponential Moving Average (EMA)");
    println!("-----------------------------------------");

    let data = create_test_data();

    // Compare different alpha values
    let alphas = [0.1, 0.3, 0.5, 0.9];

    println!("EMA comparison with different alpha values:");
    println!("  - alpha=0.1: Very smooth, slow to react");
    println!("  - alpha=0.3: Moderate smoothing");
    println!("  - alpha=0.5: Balanced responsiveness");
    println!("  - alpha=0.9: Minimal smoothing, fast response");
    println!();
    println!(
        "  {:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Index", "Original", "a=0.1", "a=0.3", "a=0.5", "a=0.9"
    );

    let mut emas: Vec<Vec<f64>> = Vec::new();

    for &alpha in &alphas {
        let spec = WindowSpec::count_based(1, WindowFunction::ExponentialMovingAverage(alpha));
        let mut calc = oxirs_tsdb::query::window::WindowCalculator::new(spec);
        let values: Vec<f64> = data
            .iter()
            .filter_map(|p| calc.add(*p))
            .map(|p| p.value)
            .collect();
        emas.push(values);
    }

    for i in 0..10 {
        println!(
            "  {:>6} | {:>10.2} | {:>10.2} | {:>10.2} | {:>10.2} | {:>10.2}",
            i,
            data[i].value,
            emas[0].get(i).unwrap_or(&0.0),
            emas[1].get(i).unwrap_or(&0.0),
            emas[2].get(i).unwrap_or(&0.0),
            emas[3].get(i).unwrap_or(&0.0),
        );
    }
    println!();
}

fn demo_moving_min_max() {
    println!("Demo 3: Moving Min/Max");
    println!("----------------------");

    // Create data with clear peaks and valleys
    let base_time = Utc::now();
    let data: Vec<DataPoint> = (0..30)
        .map(|i| {
            let value = match i % 10 {
                0 => 100.0, // Peak
                5 => 20.0,  // Valley
                _ => 50.0 + (i % 5) as f64 * 5.0,
            };
            DataPoint {
                timestamp: base_time + Duration::seconds(i),
                value,
            }
        })
        .collect();

    let spec_min = WindowSpec::count_based(5, WindowFunction::MovingMin);
    let spec_max = WindowSpec::count_based(5, WindowFunction::MovingMax);

    let mut calc_min = oxirs_tsdb::query::window::WindowCalculator::new(spec_min);
    let mut calc_max = oxirs_tsdb::query::window::WindowCalculator::new(spec_max);

    let mins: Vec<f64> = data
        .iter()
        .filter_map(|p| calc_min.add(*p))
        .map(|p| p.value)
        .collect();
    let maxs: Vec<f64> = data
        .iter()
        .filter_map(|p| calc_max.add(*p))
        .map(|p| p.value)
        .collect();

    println!("Window size: 5 points");
    println!("Use case: Finding local extrema for envelope detection");
    println!();
    println!(
        "  {:>5} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Index", "Value", "Min(5)", "Max(5)", "Range"
    );

    for i in 0..20 {
        let min_val = mins.get(i).unwrap_or(&0.0);
        let max_val = maxs.get(i).unwrap_or(&0.0);
        println!(
            "  {:>5} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.1}",
            i,
            data[i].value,
            min_val,
            max_val,
            max_val - min_val
        );
    }
    println!();
}

fn demo_rolling_stddev() {
    println!("Demo 4: Rolling Standard Deviation");
    println!("-----------------------------------");

    let base_time = Utc::now();

    // Create data with varying volatility
    let data: Vec<DataPoint> = (0..60)
        .map(|i| {
            let value = if i < 20 {
                // Low volatility period
                50.0 + (i % 2) as f64 * 0.5
            } else if i < 40 {
                // High volatility period
                50.0 + ((i * 7) % 20) as f64 - 10.0
            } else {
                // Medium volatility
                50.0 + ((i * 3) % 10) as f64 - 5.0
            };
            DataPoint {
                timestamp: base_time + Duration::seconds(i),
                value,
            }
        })
        .collect();

    let spec = WindowSpec::count_based(10, WindowFunction::RollingStdDev);
    let mut calc = oxirs_tsdb::query::window::WindowCalculator::new(spec);

    let stddevs: Vec<f64> = data
        .iter()
        .filter_map(|p| calc.add(*p))
        .map(|p| p.value)
        .collect();

    println!("Window size: 10 points");
    println!("Use case: Volatility monitoring and risk assessment");
    println!();
    println!("Period Analysis:");
    println!("  Points 0-19:  Low volatility (stable data)");
    println!("  Points 20-39: High volatility (noisy data)");
    println!("  Points 40-59: Medium volatility");
    println!();

    // Show stddev at key points
    let checkpoints: [usize; 5] = [15, 25, 35, 45, 55];
    for &i in &checkpoints {
        if let Some(std) = stddevs.get(i.saturating_sub(9)) {
            let volatility = if *std < 1.0 {
                "LOW"
            } else if *std < 5.0 {
                "MEDIUM"
            } else {
                "HIGH"
            };
            println!("  Point {}: StdDev = {:.3} ({})", i, std, volatility);
        }
    }
    println!();
}

fn demo_cumulative_sum() {
    println!("Demo 5: Cumulative Sum");
    println!("----------------------");

    let base_time = Utc::now();

    // Energy consumption increments
    let data: Vec<DataPoint> = (0..24)
        .map(|i| {
            // Variable power consumption (kWh per hour)
            let consumption = match i {
                0..=5 => 0.5,   // Night (low)
                6..=8 => 2.0,   // Morning peak
                9..=16 => 1.0,  // Daytime (medium)
                17..=20 => 2.5, // Evening peak
                _ => 0.5,       // Late night
            };
            DataPoint {
                timestamp: base_time + Duration::hours(i),
                value: consumption,
            }
        })
        .collect();

    let spec = WindowSpec::count_based(1, WindowFunction::CumulativeSum);
    let mut calc = oxirs_tsdb::query::window::WindowCalculator::new(spec);

    let cumsum: Vec<f64> = data
        .iter()
        .filter_map(|p| calc.add(*p))
        .map(|p| p.value)
        .collect();

    println!("Use case: Daily energy consumption tracking");
    println!();
    println!(
        "  {:>5} | {:>15} | {:>15} | {:>15}",
        "Hour", "Consumption", "Cumulative", "Period"
    );

    for i in 0..24 {
        let period = match i {
            0..=5 => "Night",
            6..=8 => "Morning",
            9..=16 => "Daytime",
            17..=20 => "Evening",
            _ => "Late Night",
        };
        println!(
            "  {:>5} | {:>12.1} kWh | {:>12.1} kWh | {:>15}",
            i,
            data[i].value,
            cumsum.get(i).unwrap_or(&0.0),
            period
        );
    }

    println!();
    println!(
        "Total daily consumption: {:.1} kWh",
        cumsum.last().unwrap_or(&0.0)
    );
    println!();
}

fn demo_rate_of_change() {
    println!("Demo 6: Rate of Change (Derivative)");
    println!("------------------------------------");

    let base_time = Utc::now();

    // Temperature with ramp-up and ramp-down
    let data: Vec<DataPoint> = (0..30)
        .map(|i| {
            let temp = if i < 10 {
                20.0 + i as f64 * 2.0 // Heating: +2 degC/sec
            } else if i < 20 {
                40.0 // Stable
            } else {
                40.0 - (i - 20) as f64 * 1.0 // Cooling: -1 degC/sec
            };
            DataPoint {
                timestamp: base_time + Duration::seconds(i),
                value: temp,
            }
        })
        .collect();

    let spec = WindowSpec::count_based(1, WindowFunction::RateOfChange);
    let mut calc = oxirs_tsdb::query::window::WindowCalculator::new(spec);

    let roc: Vec<f64> = data
        .iter()
        .filter_map(|p| calc.add(*p))
        .map(|p| p.value)
        .collect();

    println!("Use case: Temperature rate monitoring (heating/cooling detection)");
    println!();
    println!(
        "  {:>5} | {:>10} | {:>12} | {:>15}",
        "Time", "Temp (degC)", "Rate (degC/s)", "State"
    );

    for i in 0_usize..30 {
        let rate = roc.get(i.saturating_sub(1)).unwrap_or(&0.0);
        let state = if *rate > 0.5 {
            "HEATING"
        } else if *rate < -0.5 {
            "COOLING"
        } else {
            "STABLE"
        };

        if i % 3 == 0 || i < 3 {
            // Show every 3rd point
            println!(
                "  {:>5} | {:>10.1} | {:>+12.1} | {:>15}",
                i, data[i].value, rate, state
            );
        }
    }
    println!();
}

fn demo_anomaly_detection() {
    println!("Demo 7: Real-World Use Case - Anomaly Detection");
    println!("-------------------------------------------------");

    let base_time = Utc::now();

    // Simulated machine vibration data with anomalies
    let data: Vec<DataPoint> = (0..100)
        .map(|i| {
            let normal = 10.0 + (i % 5) as f64 * 0.2;
            // Inject anomalies at specific points
            let value = match i {
                45..=50 => normal * 3.0, // Anomaly: high vibration
                75..=78 => normal * 2.5, // Another anomaly
                _ => normal,
            };
            DataPoint {
                timestamp: base_time + Duration::seconds(i),
                value,
            }
        })
        .collect();

    // Calculate moving average and rolling stddev
    let spec_avg = WindowSpec::count_based(10, WindowFunction::MovingAverage);
    let spec_std = WindowSpec::count_based(10, WindowFunction::RollingStdDev);

    let mut calc_avg = oxirs_tsdb::query::window::WindowCalculator::new(spec_avg);
    let mut calc_std = oxirs_tsdb::query::window::WindowCalculator::new(spec_std);

    let avgs: Vec<f64> = data
        .iter()
        .filter_map(|p| calc_avg.add(*p))
        .map(|p| p.value)
        .collect();
    let stds: Vec<f64> = data
        .iter()
        .filter_map(|p| calc_std.add(*p))
        .map(|p| p.value)
        .collect();

    println!("Machine Vibration Monitoring with Z-Score Anomaly Detection");
    println!("Threshold: |z-score| > 2.0 indicates anomaly");
    println!();
    println!("Detected Anomalies:");

    let mut anomaly_count = 0;
    for i in 9..data.len() {
        let idx = i - 9;
        if let (Some(avg), Some(std)) = (avgs.get(idx), stds.get(idx)) {
            if *std > 0.1 {
                let z_score = (data[i].value - avg) / std;
                if z_score.abs() > 2.0 {
                    anomaly_count += 1;
                    println!(
                        "  Point {}: Value={:.1}, Avg={:.1}, StdDev={:.2}, Z-Score={:.2} {}",
                        i,
                        data[i].value,
                        avg,
                        std,
                        z_score,
                        if z_score > 0.0 { "HIGH" } else { "LOW" }
                    );
                }
            }
        }
    }

    if anomaly_count == 0 {
        println!("  No anomalies detected");
    }

    println!();
    println!(
        "Summary: {} anomalies detected out of {} data points",
        anomaly_count,
        data.len()
    );
    println!();
}
