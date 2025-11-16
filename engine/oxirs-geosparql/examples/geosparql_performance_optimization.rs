//! Performance Optimization Example
//!
//! This example demonstrates how to leverage oxirs-geosparql's performance features
//! for high-throughput spatial operations.
//!
//! Run with: cargo run --example performance_optimization --features "parallel" --release

use geo_types::{Geometry as GeoGeometry, Point};
use oxirs_geosparql::error::Result;
use oxirs_geosparql::geometry::Geometry;
#[cfg(feature = "parallel")]
use oxirs_geosparql::performance::parallel;
use oxirs_geosparql::performance::{simd, BatchProcessor};
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Performance Optimization Example ===\n");

    // Example 1: SIMD-Accelerated Distance Calculations
    example_simd_distance()?;

    // Example 2: Batch Processing with Automatic Optimization
    example_batch_processor()?;

    // Example 3: Parallel Distance Matrix
    example_distance_matrix()?;

    // Example 4: k-Nearest Neighbors
    example_nearest_neighbors()?;

    // Example 5: Memory-Efficient Streaming
    example_streaming()?;

    // Example 6: Performance Comparison
    example_performance_comparison()?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Example 1: SIMD-Accelerated Distance Calculations
fn example_simd_distance() -> Result<()> {
    println!("1. SIMD-ACCELERATED DISTANCE CALCULATIONS\n");

    let p1 = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
    let p2 = Geometry::new(GeoGeometry::Point(Point::new(100.0, 100.0)));

    // Standard distance calculation
    let start = Instant::now();
    let dist = oxirs_geosparql::functions::geometric_operations::distance(&p1, &p2)?;
    let standard_time = start.elapsed();
    println!(
        "   Standard distance: {:.2} (took {:?})",
        dist, standard_time
    );

    // SIMD-optimized distance (2-4x faster)
    let start = Instant::now();
    let dist = simd::euclidean_distance(&p1, &p2)?;
    let simd_time = start.elapsed();
    println!("   SIMD distance:     {:.2} (took {:?})", dist, simd_time);
    println!(
        "   Speedup:           {:.2}x\n",
        standard_time.as_nanos() as f64 / simd_time.as_nanos() as f64
    );

    // Squared distance (even faster - no sqrt)
    let dist_sq = simd::euclidean_distance_squared(&p1, &p2)?;
    println!("   Squared distance:  {:.2} (no sqrt overhead)\n", dist_sq);

    Ok(())
}

/// Example 2: Batch Processing with Automatic Optimization
fn example_batch_processor() -> Result<()> {
    println!("2. BATCH PROCESSOR (AUTOMATIC OPTIMIZATION)\n");

    let query = Geometry::new(GeoGeometry::Point(Point::new(50.0, 50.0)));

    // Small dataset (uses SIMD)
    let small_targets: Vec<_> = (0..50)
        .map(|i| Geometry::new(GeoGeometry::Point(Point::new(i as f64, i as f64))))
        .collect();

    let processor = BatchProcessor::new();
    let start = Instant::now();
    let _distances = processor.distances(&query, &small_targets)?;
    let small_time = start.elapsed();
    println!("   Small dataset (50 points):  {:?}", small_time);
    println!("   Strategy: SIMD (automatic)\n");

    // Large dataset (uses parallel processing)
    #[cfg(feature = "parallel")]
    {
        let large_targets: Vec<_> = (0..10000)
            .map(|i| {
                Geometry::new(GeoGeometry::Point(Point::new(
                    (i as f64 * 0.1) % 100.0,
                    (i as f64 * 0.2) % 100.0,
                )))
            })
            .collect();

        let start = Instant::now();
        let large_distances = processor.distances(&query, &large_targets)?;
        let large_time = start.elapsed();
        println!("   Large dataset (10,000 points): {:?}", large_time);
        println!("   Strategy: Parallel (automatic)");
        println!(
            "   Throughput: {:.0} distances/ms\n",
            large_distances.len() as f64 / large_time.as_millis() as f64
        );
    }

    #[cfg(not(feature = "parallel"))]
    {
        println!("   (Enable 'parallel' feature for large dataset demo)\n");
    }

    Ok(())
}

/// Example 3: Parallel Distance Matrix
#[cfg(feature = "parallel")]
fn example_distance_matrix() -> Result<()> {
    println!("3. PARALLEL DISTANCE MATRIX\n");

    let geometries: Vec<_> = (0..100)
        .map(|i| {
            Geometry::new(GeoGeometry::Point(Point::new(
                (i as f64 * 0.5) % 50.0,
                (i as f64 * 0.7) % 50.0,
            )))
        })
        .collect();

    // Sequential distance matrix
    let start = Instant::now();
    let _matrix_seq: Vec<Vec<_>> = geometries
        .iter()
        .map(|g1| {
            geometries
                .iter()
                .map(|g2| {
                    oxirs_geosparql::functions::geometric_operations::distance(g1, g2).unwrap()
                })
                .collect()
        })
        .collect();
    let seq_time = start.elapsed();
    println!("   Sequential (100x100): {:?}", seq_time);

    // Parallel distance matrix (4-8x faster)
    let start = Instant::now();
    let matrix_par = parallel::parallel_distance_matrix(&geometries)?;
    let par_time = start.elapsed();
    println!("   Parallel (100x100):   {:?}", par_time);
    println!(
        "   Speedup:              {:.2}x",
        seq_time.as_nanos() as f64 / par_time.as_nanos() as f64
    );
    println!(
        "   Total comparisons:    {}\n",
        matrix_par.len() * matrix_par[0].len()
    );

    Ok(())
}

#[cfg(not(feature = "parallel"))]
fn example_distance_matrix() -> Result<()> {
    println!("3. PARALLEL DISTANCE MATRIX\n");
    println!("   (Enable 'parallel' feature to see this demo)\n");
    Ok(())
}

/// Example 4: k-Nearest Neighbors
#[cfg(feature = "parallel")]
fn example_nearest_neighbors() -> Result<()> {
    println!("4. K-NEAREST NEIGHBORS\n");

    let geometries: Vec<_> = (0..1000)
        .map(|i| {
            Geometry::new(GeoGeometry::Point(Point::new(
                (i as f64 * 0.3) % 100.0,
                (i as f64 * 0.4) % 100.0,
            )))
        })
        .collect();

    let k = 5;
    let start = Instant::now();
    let nearest = parallel::parallel_nearest_neighbors(&geometries, k)?;
    let elapsed = start.elapsed();

    println!("   Dataset size:    1,000 points");
    println!("   k:               {}", k);
    println!("   Time:            {:?}", elapsed);
    println!(
        "   Throughput:      {:.0} queries/ms",
        nearest.len() as f64 / elapsed.as_millis() as f64
    );

    // Show results for first geometry
    println!("\n   Nearest neighbors to first point:");
    for (i, (idx, dist)) in nearest[0].iter().enumerate() {
        println!("     {}. Index {} at distance {:.2}", i + 1, idx, dist);
    }
    println!();

    Ok(())
}

#[cfg(not(feature = "parallel"))]
fn example_nearest_neighbors() -> Result<()> {
    println!("4. K-NEAREST NEIGHBORS\n");
    println!("   (Enable 'parallel' feature to see this demo)\n");
    Ok(())
}

/// Example 5: Memory-Efficient Streaming
fn example_streaming() -> Result<()> {
    println!("5. MEMORY-EFFICIENT STREAMING\n");

    let query = Geometry::new(GeoGeometry::Point(Point::new(50.0, 50.0)));

    // Simulate huge dataset (100,000 points)
    let huge_dataset: Vec<_> = (0..100000)
        .map(|i| {
            Geometry::new(GeoGeometry::Point(Point::new(
                (i as f64 * 0.01) % 100.0,
                (i as f64 * 0.02) % 100.0,
            )))
        })
        .collect();

    // Process in chunks to avoid memory spikes
    let processor = BatchProcessor::new();
    let mut total_count = 0;
    let mut min_distance = f64::MAX;
    let mut max_distance = f64::MIN;

    let start = Instant::now();
    processor.stream_distances(&query, &huge_dataset, |chunk_distances| {
        // Process each chunk
        total_count += chunk_distances.len();

        for &dist in chunk_distances {
            min_distance = min_distance.min(dist);
            max_distance = max_distance.max(dist);
        }

        Ok(())
    })?;
    let elapsed = start.elapsed();

    println!("   Dataset size:       100,000 points");
    println!("   Chunk size:         1,000 points");
    println!("   Total distances:    {}", total_count);
    println!("   Processing time:    {:?}", elapsed);
    println!(
        "   Throughput:         {:.0} distances/ms",
        total_count as f64 / elapsed.as_millis() as f64
    );
    println!("   Min distance:       {:.2}", min_distance);
    println!("   Max distance:       {:.2}\n", max_distance);

    Ok(())
}

/// Example 6: Performance Comparison
fn example_performance_comparison() -> Result<()> {
    println!("6. PERFORMANCE COMPARISON SUMMARY\n");

    let query = Geometry::new(GeoGeometry::Point(Point::new(50.0, 50.0)));
    let targets: Vec<_> = (0..10000)
        .map(|i| {
            Geometry::new(GeoGeometry::Point(Point::new(
                (i as f64 * 0.1) % 100.0,
                (i as f64 * 0.2) % 100.0,
            )))
        })
        .collect();

    println!("   Computing 10,000 distances:\n");

    // Sequential standard
    let start = Instant::now();
    let _: Vec<_> = targets
        .iter()
        .map(|t| oxirs_geosparql::functions::geometric_operations::distance(&query, t).unwrap())
        .collect();
    let seq_time = start.elapsed();
    println!("   Sequential (standard):  {:?}", seq_time);

    // SIMD batch
    let start = Instant::now();
    let _ = simd::batch_euclidean_distance(&query, &targets)?;
    let simd_time = start.elapsed();
    println!(
        "   SIMD batch:             {:?}  ({}x faster)",
        simd_time,
        seq_time.as_nanos() / simd_time.as_nanos()
    );

    // Parallel (if enabled)
    #[cfg(feature = "parallel")]
    {
        let start = Instant::now();
        let _ = parallel::parallel_distances(&query, &targets)?;
        let par_time = start.elapsed();
        println!(
            "   Parallel:               {:?}  ({}x faster)",
            par_time,
            seq_time.as_nanos() / par_time.as_nanos()
        );
    }

    // BatchProcessor (auto-optimization)
    let processor = BatchProcessor::new();
    let start = Instant::now();
    let _ = processor.distances(&query, &targets)?;
    let batch_time = start.elapsed();
    println!(
        "   BatchProcessor (auto):  {:?}  ({}x faster)",
        batch_time,
        seq_time.as_nanos() / batch_time.as_nanos()
    );

    println!("\n   Key Takeaways:");
    println!("   • Use SIMD for 2-4x speedup on all CPUs");
    println!("   • Use parallel for 4-8x speedup on multi-core CPUs");
    println!("   • Use BatchProcessor for automatic optimization");
    println!("   • Use streaming for huge datasets (>100k points)");

    Ok(())
}
