//! Streaming geometry processing example
//!
//! Demonstrates memory-efficient processing of large geometry datasets using
//! the streaming API.

use geo_types::{Coord, Geometry as GeoGeometry, LineString, Point};
use oxirs_geosparql::geometry::streaming::{
    GeometryStream, MonitoredStreamProcessor, StreamProcessor,
};
use oxirs_geosparql::geometry::Geometry;

fn main() {
    println!("=== Streaming Geometry Processing Example ===\n");

    // Example 1: Basic streaming with filtering
    println!("1. Basic Streaming with Filtering");
    basic_streaming_example();

    // Example 2: Chunked processing
    println!("\n2. Chunked Processing");
    chunked_processing_example();

    // Example 3: Stream transformations
    println!("\n3. Stream Transformations");
    transformation_example();

    // Example 4: Monitored streaming with statistics
    println!("\n4. Monitored Streaming with Statistics");
    monitored_streaming_example();

    // Example 5: Parallel processing (large dataset)
    #[cfg(feature = "parallel")]
    {
        println!("\n5. Parallel Processing");
        parallel_processing_example();
    }

    // Example 6: Chained operations
    println!("\n6. Chained Operations");
    chained_operations_example();
}

fn basic_streaming_example() {
    // Create a large collection of geometries
    let geometries: Vec<_> = (0..1000)
        .map(|i| Geometry::new(GeoGeometry::Point(Point::new(i as f64, i as f64))))
        .collect();

    println!("  Created {} geometries", geometries.len());

    // Create a stream
    let stream = GeometryStream::from_vec(geometries);

    // Filter to only points with x > 500
    let processor = StreamProcessor::new().filter(|g| {
        if let geo_types::Geometry::Point(pt) = &g.geom {
            pt.x() > 500.0
        } else {
            false
        }
    });

    // Process and collect results
    let results: Vec<_> = processor.process(stream).filter_map(|r| r.ok()).collect();

    println!("  Filtered to {} geometries (x > 500)", results.len());
    assert_eq!(results.len(), 499); // 501-999 = 499 geometries
}

fn chunked_processing_example() {
    // Create geometries
    let geometries: Vec<_> = (0..1000)
        .map(|i| Geometry::new(GeoGeometry::Point(Point::new(i as f64, i as f64))))
        .collect();

    // Create stream with chunking
    let stream = GeometryStream::from_vec(geometries);
    let chunks: Vec<_> = stream.chunks(100).collect();

    println!("  Processed {} chunks", chunks.len());
    println!("  First chunk size: {}", chunks[0].len());
    println!("  Last chunk size: {}", chunks.last().unwrap().len());

    assert_eq!(chunks.len(), 10); // 1000 / 100 = 10 chunks
}

fn transformation_example() {
    // Create a linestring
    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 1.0, y: 1.0 },
        Coord { x: 2.0, y: 2.0 },
    ];
    let linestring = LineString::new(coords);
    let geometries = vec![Geometry::new(GeoGeometry::LineString(linestring))];

    let stream = GeometryStream::from_vec(geometries);

    // Apply a transformation (identity in this case)
    let processor = StreamProcessor::new().transform(|geom| {
        // In a real scenario, you might apply CRS transformation,
        // simplification, or other geometry operations
        Ok(geom)
    });

    let results: Vec<_> = processor.process(stream).filter_map(|r| r.ok()).collect();

    println!("  Transformed {} geometries", results.len());

    if let geo_types::Geometry::LineString(ls) = &results[0].geom {
        println!("  LineString has {} coordinates", ls.0.len());
    }
}

fn monitored_streaming_example() {
    // Create geometries with some that will be filtered out
    let geometries: Vec<_> = (0..1000)
        .map(|i| Geometry::new(GeoGeometry::Point(Point::new(i as f64, i as f64))))
        .collect();

    println!("  Input: {} geometries", geometries.len());

    let stream = GeometryStream::from_vec(geometries);

    // Create processor with filter
    let processor = StreamProcessor::new()
        .filter(|g| {
            if let geo_types::Geometry::Point(pt) = &g.geom {
                pt.x() >= 200.0 && pt.x() < 800.0
            } else {
                false
            }
        })
        .buffer(100);

    // Wrap with monitoring
    let monitored = MonitoredStreamProcessor::new(processor);

    // Process and get statistics handle
    let (processed_stream, stats_ref) = monitored.process(stream);

    let results: Vec<_> = processed_stream.filter_map(|r| r.ok()).collect();

    // Get final statistics
    let stats = stats_ref.lock().clone();

    println!("  Results: {} geometries", results.len());
    println!("  Statistics:");
    println!("    - Total processed: {}", stats.total);
    println!("    - Passed filters: {}", stats.passed);
    println!("    - Filtered out: {}", stats.filtered);
    println!("    - Errors: {}", stats.errors);
    println!("    - Pass rate: {:.1}%", stats.pass_rate() * 100.0);
}

#[cfg(feature = "parallel")]
fn parallel_processing_example() {
    use std::time::Instant;

    // Create a large dataset
    let geometries: Vec<_> = (0..10000)
        .map(|i| Geometry::new(GeoGeometry::Point(Point::new(i as f64, i as f64))))
        .collect();

    println!("  Processing {} geometries in parallel", geometries.len());

    let stream = GeometryStream::from_vec(geometries);

    // Filter to even x values
    let processor = StreamProcessor::new()
        .filter(|g| {
            if let geo_types::Geometry::Point(pt) = &g.geom {
                (pt.x() as i32) % 2 == 0
            } else {
                false
            }
        })
        .buffer(500); // Process in chunks of 500

    let start = Instant::now();
    let results: Vec<_> = processor
        .process_parallel(stream)
        .filter_map(|r| r.ok())
        .collect();
    let duration = start.elapsed();

    println!("  Results: {} geometries", results.len());
    println!("  Processing time: {:?}", duration);
    println!(
        "  Throughput: {:.0} geom/sec",
        results.len() as f64 / duration.as_secs_f64()
    );

    assert_eq!(results.len(), 5000); // Half should pass the even filter
}

fn chained_operations_example() {
    // Create geometries
    let geometries: Vec<_> = (0..1000)
        .map(|i| Geometry::new(GeoGeometry::Point(Point::new(i as f64, i as f64))))
        .collect();

    println!("  Starting with {} geometries", geometries.len());

    let stream = GeometryStream::from_vec(geometries);

    // Chain multiple operations
    let processor = StreamProcessor::new()
        // First filter: x >= 100
        .filter(|g| {
            if let geo_types::Geometry::Point(pt) = &g.geom {
                pt.x() >= 100.0
            } else {
                false
            }
        })
        // Second filter: x < 900
        .filter(|g| {
            if let geo_types::Geometry::Point(pt) = &g.geom {
                pt.x() < 900.0
            } else {
                false
            }
        })
        // Third filter: divisible by 10
        .filter(|g| {
            if let geo_types::Geometry::Point(pt) = &g.geom {
                (pt.x() as i32) % 10 == 0
            } else {
                false
            }
        })
        // Transform (identity in this example)
        .transform(Ok)
        .buffer(50);

    let results: Vec<_> = processor.process(stream).filter_map(|r| r.ok()).collect();

    println!("  After all filters: {} geometries", results.len());

    // Should have: 100, 110, 120, ..., 890 = 80 geometries
    assert_eq!(results.len(), 80);

    // Verify first and last
    if let geo_types::Geometry::Point(pt) = &results[0].geom {
        println!("  First result: x = {}", pt.x());
        assert_eq!(pt.x(), 100.0);
    }

    if let geo_types::Geometry::Point(pt) = &results.last().unwrap().geom {
        println!("  Last result: x = {}", pt.x());
        assert_eq!(pt.x(), 890.0);
    }
}
