//! Parallel Validation Example
//!
//! Demonstrates high-performance shape processing using:
//! - Rayon-powered parallel iteration
//! - Performance metrics
//! - Batch processing
//!
//! Run with:
//! ```bash
//! cargo run --example parallel_validation --features parallel
//! ```

#[cfg(feature = "parallel")]
use std::time::Instant;

#[cfg(feature = "parallel")]
use oxirs_shacl::{Shape, ShapeId};

#[cfg(feature = "parallel")]
fn main() {
    println!("⚡ OxiRS SHACL - Parallel Validation Example\n");

    // Create test shapes
    let shapes: Vec<Shape> = (0..1000)
        .map(|i| Shape::node_shape(ShapeId(format!("http://example.org/Shape{}", i))))
        .collect();

    println!("📊 Performance Comparison\n");

    // Sequential processing
    println!("1️⃣  Sequential processing of {} shapes...", shapes.len());
    let start = Instant::now();
    let mut count = 0;
    for shape in &shapes {
        // Simulate work
        count += shape.constraints.len();
    }
    let sequential_time = start.elapsed();
    println!("   ⏱️  Time: {:?}", sequential_time);
    println!("   📈 Processed: {} constraint evaluations", count);
    println!();

    // Parallel processing
    println!("2️⃣  Parallel processing of {} shapes...", shapes.len());
    let start = Instant::now();

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let parallel_count = AtomicUsize::new(0);
        shapes.par_iter().for_each(|shape| {
            // Simulate work
            parallel_count.fetch_add(shape.constraints.len(), Ordering::Relaxed);
        });

        let parallel_time = start.elapsed();
        println!("   ⏱️  Time: {:?}", parallel_time);
        println!(
            "   📈 Processed: {} constraint evaluations",
            parallel_count.load(Ordering::Relaxed)
        );
        println!();

        // Calculate speedup
        let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
        println!("📈 Results:");
        println!("   Speedup: {:.2}x faster", speedup);
        println!("   CPU cores: {}", num_cpus::get());
        println!(
            "   Efficiency: {:.1}%",
            (speedup / num_cpus::get() as f64) * 100.0
        );
    }

    println!();
    println!("✨ Parallel processing example complete!");
    println!("\n💡 Performance Tips:");
    println!("   - Use parallel processing for large datasets (>1000 shapes)");
    println!("   - Enable caching for repeated validations");
    println!("   - Use batch processing for memory efficiency");
}

#[cfg(not(feature = "parallel"))]
fn main() {
    println!("⚠️  This example requires the 'parallel' feature.");
    println!("Run with: cargo run --example parallel_validation --features parallel");
}
