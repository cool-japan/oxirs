//! GPU-accelerated spatial operations example for oxirs-geosparql
//!
//! This example demonstrates:
//! - Creating GPU contexts with automatic CPU fallback
//! - Computing pairwise distance matrices on GPU
//! - Batch Euclidean distance calculations
//! - Spatial joins within distance thresholds
//! - K-nearest neighbor queries
//! - Performance optimization for large datasets
//!
//! Run with: cargo run --example gpu_operations --features gpu

use oxirs_geosparql::error::Result;
use oxirs_geosparql::geometry::Geometry;

#[cfg(feature = "gpu")]
use oxirs_geosparql::performance::gpu::GpuGeometryContext;

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL GPU Operations Example ===\n");

    #[cfg(feature = "gpu")]
    {
        // 1. GPU Context Creation
        println!("1. GPU CONTEXT CREATION:");

        let ctx = GpuGeometryContext::new()?;
        println!("   GPU Context created successfully");
        println!("   Backend: {:?}", ctx.backend());
        println!("   Max batch size: {}", ctx.max_batch_size());
        println!(
            "   Note: Currently uses optimized CPU fallback while awaiting GPU backend support\n"
        );

        // 2. Create test geometries
        println!("2. CREATING TEST GEOMETRIES:");

        let cities = vec![
            Geometry::from_wkt("POINT(-122.4194 37.7749)")?, // San Francisco
            Geometry::from_wkt("POINT(-118.2437 34.0522)")?, // Los Angeles
            Geometry::from_wkt("POINT(-87.6298 41.8781)")?,  // Chicago
            Geometry::from_wkt("POINT(-73.9352 40.7306)")?,  // New York
            Geometry::from_wkt("POINT(-95.3698 29.7604)")?,  // Houston
        ];

        let city_names = [
            "San Francisco",
            "Los Angeles",
            "Chicago",
            "New York",
            "Houston",
        ];

        println!("   Created {} city locations:", cities.len());
        for (name, city) in city_names.iter().zip(cities.iter()) {
            println!("     {}: {}", name, city.to_wkt());
        }

        // 3. Pairwise Distance Matrix
        println!("\n3. PAIRWISE DISTANCE MATRIX:");
        println!("   Computing all-to-all distances...");

        let distances = ctx.pairwise_distance_matrix(&cities)?;
        println!("   Distance matrix shape: {:?}", distances.shape());
        println!("\n   Distance Matrix (approximate distances in degrees):");
        println!("                    SF      LA     CHI      NY     HOU");

        for (i, from_city) in city_names.iter().enumerate() {
            print!("   {:12}", from_city);
            for j in 0..city_names.len() {
                print!("{:8.2}", distances[[i, j]]);
            }
            println!();
        }

        // 4. Batch Euclidean Distance
        println!("\n4. BATCH EUCLIDEAN DISTANCE:");
        println!("   Computing distances from queries to targets...");

        let query_points = vec![
            Geometry::from_wkt("POINT(-122.0 38.0)")?, // Near San Francisco
        ];

        let batch_distances = ctx.batch_euclidean_distance(&query_points, &cities)?;
        println!("   Query point: {}", query_points[0].to_wkt());
        println!("\n   Distances to cities:");

        for (i, city_name) in city_names.iter().enumerate() {
            println!("     {}: {:.4} degrees", city_name, batch_distances[[0, i]]);
        }

        // 5. Spatial Join Within Distance
        println!("\n5. SPATIAL JOIN WITHIN DISTANCE:");
        println!("   Finding city pairs within specified distance...");

        let max_distance = 5.0; // degrees (approximately)
        let pairs = ctx.spatial_join_within_distance(&cities, max_distance)?;

        println!(
            "   Found {} pairs within {} degrees:",
            pairs.len(),
            max_distance
        );
        for (i, j, dist) in &pairs {
            println!(
                "     {} ↔ {}: {:.2} degrees",
                city_names[*i], city_names[*j], dist
            );
        }

        // 6. K-Nearest Neighbors
        println!("\n6. K-NEAREST NEIGHBORS:");
        println!("   Finding 3 nearest cities to each query point...");

        let query_location = vec![Geometry::from_wkt("POINT(-100.0 35.0)")?]; // Central US

        let k = 3;
        let knn_results = ctx.k_nearest_neighbors(&query_location, &cities, k)?;

        println!("   Query location: {}", query_location[0].to_wkt());
        println!("   {} nearest cities:", k);

        for (idx, dist) in &knn_results[0] {
            println!("     {}: {:.2} degrees", city_names[*idx], dist);
        }

        // 7. Large-scale batch processing
        println!("\n\n=== LARGE-SCALE BATCH PROCESSING ===\n");

        println!("Creating 1000 random points...");
        let mut large_dataset = Vec::new();
        for i in 0..1000 {
            let lon = -180.0 + (i as f64 * 0.36); // Spread across longitudes
            let lat = -90.0 + ((i * 13) % 180) as f64; // Spread across latitudes
            large_dataset.push(Geometry::from_wkt(&format!("POINT({} {})", lon, lat))?);
        }

        println!("Computing pairwise distances for 1000 points...");
        let large_distances = ctx.pairwise_distance_matrix(&large_dataset)?;

        println!("   Result matrix shape: {:?}", large_distances.shape());
        println!("   Matrix contains {} distance computations", 1000 * 1000);
        println!("   ✓ Computation completed successfully");

        // Calculate some statistics
        let mut min_dist = f32::MAX;
        let mut max_dist = f32::MIN;
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..1000 {
            for j in (i + 1)..1000 {
                let dist = large_distances[[i, j]];
                min_dist = min_dist.min(dist);
                max_dist = max_dist.max(dist);
                sum += dist as f64;
                count += 1;
            }
        }

        let avg_dist = sum / count as f64;
        println!("\n   Distance Statistics:");
        println!("     Min distance: {:.2}", min_dist);
        println!("     Max distance: {:.2}", max_dist);
        println!("     Average distance: {:.2}", avg_dist);

        // 8. K-NN on large dataset
        println!("\n8. K-NN ON LARGE DATASET:");

        let large_queries = vec![
            Geometry::from_wkt("POINT(0.0 0.0)")?,
            Geometry::from_wkt("POINT(-100.0 40.0)")?,
            Geometry::from_wkt("POINT(100.0 -40.0)")?,
        ];

        println!("   Finding 5 nearest neighbors for 3 query points...");
        let large_knn = ctx.k_nearest_neighbors(&large_queries, &large_dataset, 5)?;

        for (query_idx, neighbors) in large_knn.iter().enumerate() {
            println!(
                "\n   Query {}: {}",
                query_idx + 1,
                large_queries[query_idx].to_wkt()
            );
            println!("     Nearest neighbors:");
            for (i, (point_idx, dist)) in neighbors.iter().enumerate() {
                println!("       {}. Point {}: {:.4} degrees", i + 1, point_idx, dist);
            }
        }

        // 9. Spatial join on subset
        println!("\n9. SPATIAL JOIN ON LARGE DATASET:");

        let subset = &large_dataset[0..100];
        println!("   Using subset of 100 points");

        let threshold = 50.0;
        let subset_pairs = ctx.spatial_join_within_distance(subset, threshold)?;

        println!(
            "   Found {} pairs within {} degrees",
            subset_pairs.len(),
            threshold
        );
        if !subset_pairs.is_empty() {
            println!("   Sample pairs:");
            for (i, j, dist) in subset_pairs.iter().take(5) {
                println!("     Point {} ↔ Point {}: {:.2} degrees", i, j, dist);
            }
            if subset_pairs.len() > 5 {
                println!("     ... and {} more pairs", subset_pairs.len() - 5);
            }
        }

        // 10. Performance tips
        println!("\n\n=== PERFORMANCE TIPS ===\n");

        println!("1. Batch Operations:");
        println!("   - Use batch_euclidean_distance() instead of individual distance() calls");
        println!("   - 10-100x faster for large datasets");

        println!("\n2. GPU Context Reuse:");
        println!("   - Create GpuGeometryContext once and reuse it");
        println!("   - Avoids initialization overhead");

        println!("\n3. Dataset Size:");
        println!("   - GPU acceleration most effective for >1000 geometries");
        println!("   - Use SIMD operations for smaller datasets");

        println!("\n4. CRS Compatibility:");
        println!("   - Ensure all geometries use the same CRS");
        println!("   - CRS validation is performed automatically");

        println!("\n5. Future GPU Support:");
        println!("   - Current version uses optimized CPU implementation");
        println!("   - True GPU acceleration coming in future release");
        println!("   - Expect 10-100x speedup for massive datasets (10,000+ geometries)");

        // 11. Real-world use case
        println!("\n\n=== REAL-WORLD USE CASE: POI CLUSTERING ===\n");

        println!("Scenario: Finding Points of Interest (POI) clusters");

        // Create sample POI data
        let pois = vec![
            Geometry::from_wkt("POINT(-122.408 37.783)")?, // Cluster 1
            Geometry::from_wkt("POINT(-122.410 37.785)")?,
            Geometry::from_wkt("POINT(-122.409 37.784)")?,
            Geometry::from_wkt("POINT(-118.243 34.052)")?, // Cluster 2
            Geometry::from_wkt("POINT(-118.245 34.054)")?,
            Geometry::from_wkt("POINT(-118.244 34.053)")?,
        ];

        let poi_names = [
            "Restaurant A",
            "Cafe B",
            "Shop C",
            "Restaurant D",
            "Cafe E",
            "Shop F",
        ];

        println!("Step 1 - Find POIs within 0.01 degrees (~1km):");
        let poi_clusters = ctx.spatial_join_within_distance(&pois, 0.01)?;

        println!("   Found {} nearby pairs:", poi_clusters.len());
        for (i, j, dist) in &poi_clusters {
            println!(
                "     {} ↔ {}: {:.4} degrees",
                poi_names[*i], poi_names[*j], dist
            );
        }

        println!("\nStep 2 - Find nearest competitors for each POI:");
        let nearest_competitors = ctx.k_nearest_neighbors(&pois, &pois, 2)?; // k=2 (skip self)

        for (i, neighbors) in nearest_competitors.iter().enumerate() {
            println!("\n   {}:", poi_names[i]);
            for (j, (neighbor_idx, dist)) in neighbors.iter().enumerate() {
                if *neighbor_idx != i {
                    // Skip self
                    println!(
                        "     Nearest competitor: {} ({:.4} degrees)",
                        poi_names[*neighbor_idx], dist
                    );
                    break;
                }
                if j == 1 && neighbors.len() > 1 {
                    println!(
                        "     Nearest competitor: {} ({:.4} degrees)",
                        poi_names[neighbors[1].0], neighbors[1].1
                    );
                }
            }
        }

        println!("\n=== Example completed successfully! ===");
        println!("\nNote: GPU features require 'gpu' feature flag.");
        println!("Run with: cargo run --example gpu_operations --features gpu");
        println!("\nFor true GPU acceleration (CUDA/Metal), use:");
        println!("  cargo run --example gpu_operations --features cuda");
        println!("  cargo run --example gpu_operations --features metal");
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("❌ This example requires the 'gpu' feature to be enabled.\n");
        println!("Please run with:");
        println!("   cargo run --example gpu_operations --features gpu\n");
        println!("Or for GPU-specific backends:");
        println!("   cargo run --example gpu_operations --features cuda");
        println!("   cargo run --example gpu_operations --features metal\n");
    }

    Ok(())
}
