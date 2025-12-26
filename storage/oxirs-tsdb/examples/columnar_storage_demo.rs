//! Columnar Storage Demonstration
//!
//! Shows how to use columnar storage for efficient disk-backed time-series data.

use chrono::{Duration, Utc};
use oxirs_tsdb::{ColumnarStore, DataPoint, TimeChunk};
use std::env;

#[allow(dead_code)]
fn create_quad(
    subject: &str,
    predicate: &str,
    object: &str,
) -> Result<oxirs_core::model::Quad, Box<dyn std::error::Error>> {
    use oxirs_core::model::{GraphName, Literal, NamedNode, Quad};
    let s = NamedNode::new(subject)?;
    let p = NamedNode::new(predicate)?;
    let o = Literal::new(object);
    Ok(Quad::new(s, p, o, GraphName::DefaultGraph))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS Columnar Storage Demo ===\n");

    let temp_dir = env::temp_dir().join("oxirs_columnar_demo");
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Create columnar store
    let mut store = ColumnarStore::new(&temp_dir, Duration::hours(2), 1000)?;
    store.set_fsync(true); // Enable for production
    println!("✓ Created columnar store at: {}", temp_dir.display());
    println!("  - Chunk duration: 2 hours");
    println!("  - Cache size: 1000 chunks");
    println!("  - Fsync: enabled\n");

    // Generate sample data for three sensors
    println!("1. Generating sample data...");
    let series = vec![
        (100, "Temperature Sensor", 20.0),
        (200, "Humidity Sensor", 45.0),
        (300, "Pressure Sensor", 1013.0),
    ];

    let base_time = Utc::now();
    let mut total_points = 0;

    for (series_id, name, base_value) in &series {
        let mut points = Vec::new();
        for i in 0..100 {
            let timestamp = base_time + Duration::seconds(i);
            let value = base_value + (i as f64 * 0.1).sin();
            points.push(DataPoint::new(timestamp, value));
        }

        println!("   - Series {series_id} ({name}): {} points", points.len());
        total_points += points.len();

        // Create and write chunk
        let chunk = TimeChunk::new(*series_id, base_time, Duration::hours(2), points)?;
        let entry = store.write_chunk(&chunk)?;

        println!(
            "     ✓ Chunk written: ID={}, compressed={} bytes, ratio={:.1}x",
            entry.chunk_id,
            entry.compressed_size,
            entry.compression_ratio()
        );
    }

    println!("\n2. Storage statistics:");
    println!("   - Total data points: {total_points}");
    println!("   - Series count: {}", store.index().series_count()?);
    println!("   - Chunk count: {}", store.index().chunk_count()?);

    // Query data back
    println!("\n3. Querying data...");
    let query_start = base_time;
    let query_end = base_time + Duration::seconds(50);

    for (series_id, name, _) in &series {
        let points = store.query_range(*series_id, query_start, query_end)?;
        println!(
            "   - Series {series_id} ({name}): {} points retrieved",
            points.len()
        );
        if !points.is_empty() {
            println!(
                "     First: timestamp={}, value={:.2}",
                points[0].timestamp, points[0].value
            );
            println!(
                "     Last:  timestamp={}, value={:.2}",
                points[points.len() - 1].timestamp,
                points[points.len() - 1].value
            );
        }
    }

    println!("\n4. Chunk file structure:");
    println!("   {}/", temp_dir.display());
    for series_id in [100, 200, 300] {
        let series_dir = temp_dir.join(format!("series_{series_id}"));
        if series_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&series_dir) {
                for entry in entries.flatten() {
                    let metadata = entry.metadata()?;
                    println!(
                        "   ├─ series_{series_id}/{}  ({} bytes)",
                        entry.file_name().to_string_lossy(),
                        metadata.len()
                    );
                }
            }
        }
    }
    println!("   └─ index.json");

    println!("\n✓ Columnar storage demo complete!");
    println!("\nCleanup: rm -rf {}", temp_dir.display());

    // Cleanup
    std::fs::remove_dir_all(&temp_dir)?;

    Ok(())
}
