//! Temporal SPARQL Functions Demonstration
//!
//! Shows how to use temporal extensions for time-series queries in SPARQL.

use chrono::Utc;
use oxirs_tsdb::{
    interpolate_function, register_temporal_functions, resample_function, window_function,
    TemporalFunctionRegistry, TemporalValue,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS Temporal SPARQL Functions Demo ===\n");

    // Create function registry
    let registry = TemporalFunctionRegistry::new();
    register_temporal_functions(&registry)?;
    println!("✓ Registered temporal functions\n");

    // List registered functions
    println!("Available functions:");
    for func in registry.list_functions()? {
        println!("  - {func}");
    }
    println!();

    // 1. Window aggregation
    println!("1. Window Aggregation: ts:window(?value, ?window_size, \"AVG\")");
    let window_fn = window_function();
    let result = window_fn(&[
        TemporalValue::Float(42.5),
        TemporalValue::Integer(600), // 10 minutes
        TemporalValue::String("AVG".to_string()),
    ])?;
    println!("   Input: value=42.5, window=600s, agg=AVG");
    println!("   Output: {result:?}\n");

    // 2. Time resampling
    println!("2. Time Resampling: ts:resample(?timestamp, \"1h\")");
    let resample_fn = resample_function();
    let timestamp = Utc::now();
    let result = resample_fn(&[
        TemporalValue::Timestamp(timestamp),
        TemporalValue::String("1h".to_string()),
    ])?;
    println!("   Input: timestamp={timestamp}, interval=1h");
    println!("   Output: {result:?}");
    println!("   (Rounded to hour boundary)\n");

    // 3. Value interpolation
    println!("3. Value Interpolation: ts:interpolate(?timestamp, ?value, \"linear\")");
    let interpolate_fn = interpolate_function();
    let result = interpolate_fn(&[
        TemporalValue::Timestamp(timestamp),
        TemporalValue::Float(25.5),
        TemporalValue::String("linear".to_string()),
    ])?;
    println!("   Input: timestamp={timestamp}, value=25.5, method=linear");
    println!("   Output: {result:?}\n");

    // 4. Handling null values
    println!("4. Interpolating null values:");
    let result = interpolate_fn(&[
        TemporalValue::Timestamp(timestamp),
        TemporalValue::Null,
        TemporalValue::String("forward".to_string()),
    ])?;
    println!("   Input: timestamp={timestamp}, value=NULL, method=forward");
    println!("   Output: {result:?} (filled)\n");

    // 5. Example SPARQL queries
    println!("=== Example SPARQL Queries ===\n");

    println!("Moving average over 10-minute window:");
    println!(
        r#"
PREFIX ts: <http://oxirs.org/ts#>
PREFIX qudt: <http://qudt.org/schema/qudt/>

SELECT ?sensor (ts:window(?temp, 600, "AVG") AS ?avg_temp)
WHERE {{
  ?sensor qudt:numericValue ?temp ;
          :timestamp ?time .
}}
    "#
    );

    println!("\nResample to hourly buckets:");
    println!(
        r#"
PREFIX ts: <http://oxirs.org/ts#>

SELECT (ts:resample(?time, "1h") AS ?hour) (AVG(?value) AS ?avg)
WHERE {{
  ?sensor :value ?value ;
          :timestamp ?time .
}}
GROUP BY (ts:resample(?time, "1h"))
    "#
    );

    println!("\nLinear interpolation for missing values:");
    println!(
        r#"
PREFIX ts: <http://oxirs.org/ts#>

SELECT ?sensor (ts:interpolate(?time, ?value, "linear") AS ?filled)
WHERE {{
  ?sensor :value ?value ;
          :timestamp ?time .
}}
    "#
    );

    println!("\n✓ Temporal functions ready for oxirs-arq integration!");

    Ok(())
}
