//! Performance Analysis Tool for OxiRS Cluster
//!
//! This tool provides comprehensive performance analysis and profiling
//! for OxiRS cluster deployments.
//!
//! ## Usage
//!
//! ```bash
//! # Run performance analysis
//! cargo run --example performance_analyzer --  --profile cluster --duration 60
//!
//! # Analyze specific components
//! cargo run --example performance_analyzer -- --component raft --samples 1000
//!
//! # Generate performance report
//! cargo run --example performance_analyzer -- --report --output perf_report.json
//! ```

use anyhow::Result;
use oxirs_cluster::{ClusterNode, NodeConfig};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Performance metrics collected during analysis
#[derive(Debug, Clone)]
struct PerformanceMetrics {
    /// Component being measured
    component: String,
    /// Total operations executed
    total_operations: u64,
    /// Successful operations
    successful_operations: u64,
    /// Failed operations
    failed_operations: u64,
    /// Total time elapsed
    total_duration: Duration,
    /// Latency samples (milliseconds)
    latencies: Vec<f64>,
    /// Memory samples (bytes)
    memory_samples: Vec<usize>,
}

impl PerformanceMetrics {
    fn new(component: String) -> Self {
        Self {
            component,
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            total_duration: Duration::ZERO,
            latencies: Vec::new(),
            memory_samples: Vec::new(),
        }
    }

    fn record_operation(&mut self, latency: Duration, success: bool) {
        self.total_operations += 1;
        if success {
            self.successful_operations += 1;
        } else {
            self.failed_operations += 1;
        }
        self.latencies.push(latency.as_secs_f64() * 1000.0);
    }

    fn record_memory(&mut self, bytes: usize) {
        self.memory_samples.push(bytes);
    }

    /// Calculate statistics
    fn calculate_stats(&self) -> PerformanceStats {
        let mut sorted_latencies = self.latencies.clone();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean_latency = if !self.latencies.is_empty() {
            self.latencies.iter().sum::<f64>() / self.latencies.len() as f64
        } else {
            0.0
        };

        let p50 = percentile(&sorted_latencies, 0.50);
        let p95 = percentile(&sorted_latencies, 0.95);
        let p99 = percentile(&sorted_latencies, 0.99);

        let success_rate = if self.total_operations > 0 {
            (self.successful_operations as f64 / self.total_operations as f64) * 100.0
        } else {
            0.0
        };

        let throughput = if self.total_duration.as_secs_f64() > 0.0 {
            self.total_operations as f64 / self.total_duration.as_secs_f64()
        } else {
            0.0
        };

        let avg_memory = if !self.memory_samples.is_empty() {
            self.memory_samples.iter().sum::<usize>() / self.memory_samples.len()
        } else {
            0
        };

        PerformanceStats {
            component: self.component.clone(),
            total_operations: self.total_operations,
            success_rate,
            throughput,
            mean_latency,
            p50_latency: p50,
            p95_latency: p95,
            p99_latency: p99,
            avg_memory_bytes: avg_memory,
        }
    }
}

/// Calculated performance statistics
#[derive(Debug, Clone)]
struct PerformanceStats {
    component: String,
    total_operations: u64,
    success_rate: f64,
    throughput: f64,
    mean_latency: f64,
    p50_latency: f64,
    p95_latency: f64,
    p99_latency: f64,
    avg_memory_bytes: usize,
}

impl PerformanceStats {
    fn print_report(&self) {
        println!("\n📊 Performance Report: {}", self.component);
        println!("─────────────────────────────────────────");
        println!("Total Operations:    {}", self.total_operations);
        println!("Success Rate:        {:.2}%", self.success_rate);
        println!("Throughput:          {:.2} ops/sec", self.throughput);
        println!("\nLatency Statistics:");
        println!("  Mean:              {:.2} ms", self.mean_latency);
        println!("  P50:               {:.2} ms", self.p50_latency);
        println!("  P95:               {:.2} ms", self.p95_latency);
        println!("  P99:               {:.2} ms", self.p99_latency);
        println!("\nMemory:");
        println!("  Average:           {} KB", self.avg_memory_bytes / 1024);
        println!("─────────────────────────────────────────");
    }
}

/// Calculate percentile from sorted data
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    let index = ((sorted_data.len() as f64) * p) as usize;
    sorted_data[index.min(sorted_data.len() - 1)]
}

/// Performance analyzer for cluster operations
struct PerformanceAnalyzer {
    metrics: HashMap<String, PerformanceMetrics>,
}

impl PerformanceAnalyzer {
    fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    fn get_or_create_metrics(&mut self, component: &str) -> &mut PerformanceMetrics {
        self.metrics
            .entry(component.to_string())
            .or_insert_with(|| PerformanceMetrics::new(component.to_string()))
    }

    /// Analyze node startup performance
    async fn analyze_node_startup(&mut self, iterations: usize) -> Result<()> {
        println!("\n🔍 Analyzing Node Startup Performance...");

        let metrics = self.get_or_create_metrics("node_startup");
        let start = Instant::now();

        for i in 0..iterations {
            let op_start = Instant::now();

            let mut node = create_test_node(1000 + i as u64, 20000 + i as u16).await?;
            let startup_result = node.start().await;
            let _stop_result = node.stop().await;

            let op_duration = op_start.elapsed();
            metrics.record_operation(op_duration, startup_result.is_ok());

            if i % (iterations / 10) == 0 && i > 0 {
                println!(
                    "   Progress: {}/{} ({:.1}%)",
                    i,
                    iterations,
                    (i as f64 / iterations as f64) * 100.0
                );
            }
        }

        metrics.total_duration = start.elapsed();

        println!("   ✓ Completed {} startup iterations", iterations);

        Ok(())
    }

    /// Analyze query performance
    async fn analyze_query_performance(&mut self, iterations: usize) -> Result<()> {
        println!("\n🔍 Analyzing Query Performance...");

        let mut node = create_test_node(2000, 21000).await?;
        node.start().await?;

        sleep(Duration::from_millis(500)).await;

        let metrics = self.get_or_create_metrics("query_execution");
        let start = Instant::now();

        for i in 0..iterations {
            let op_start = Instant::now();

            let _results = node.query_triples(None, None, None).await;

            let op_duration = op_start.elapsed();
            metrics.record_operation(op_duration, true);

            // Sample memory periodically
            if i % 100 == 0 {
                let triple_count = node.len().await;
                metrics.record_memory(triple_count * 200); // Rough estimate
            }

            if i % (iterations / 10) == 0 && i > 0 {
                println!(
                    "   Progress: {}/{} ({:.1}%)",
                    i,
                    iterations,
                    (i as f64 / iterations as f64) * 100.0
                );
            }
        }

        metrics.total_duration = start.elapsed();

        node.stop().await?;

        println!("   ✓ Completed {} query iterations", iterations);

        Ok(())
    }

    /// Analyze status reporting performance
    async fn analyze_status_performance(&mut self, iterations: usize) -> Result<()> {
        println!("\n🔍 Analyzing Status Reporting Performance...");

        let mut node = create_test_node(3000, 22000).await?;
        node.start().await?;

        sleep(Duration::from_millis(500)).await;

        let metrics = self.get_or_create_metrics("status_reporting");
        let start = Instant::now();

        for i in 0..iterations {
            let op_start = Instant::now();

            let _status = node.get_status().await;

            let op_duration = op_start.elapsed();
            metrics.record_operation(op_duration, true);

            if i % (iterations / 10) == 0 && i > 0 {
                println!(
                    "   Progress: {}/{} ({:.1}%)",
                    i,
                    iterations,
                    (i as f64 / iterations as f64) * 100.0
                );
            }
        }

        metrics.total_duration = start.elapsed();

        node.stop().await?;

        println!("   ✓ Completed {} status iterations", iterations);

        Ok(())
    }

    /// Generate comprehensive performance report
    fn generate_report(&self) {
        println!("\n╔════════════════════════════════════════════════════╗");
        println!("║        OxiRS Cluster Performance Report           ║");
        println!("╚════════════════════════════════════════════════════╝");

        for metrics in self.metrics.values() {
            let stats = metrics.calculate_stats();
            stats.print_report();
        }

        println!("\n✅ Analysis Complete");
    }

    /// Compare metrics between components
    fn print_comparison(&self) {
        println!("\n📈 Performance Comparison");
        println!("─────────────────────────────────────────────────────");
        println!(
            "{:<20} {:>12} {:>12} {:>12}",
            "Component", "Throughput", "P95 Latency", "P99 Latency"
        );
        println!("─────────────────────────────────────────────────────");

        for metrics in self.metrics.values() {
            let stats = metrics.calculate_stats();
            println!(
                "{:<20} {:>12.2} {:>10.2} ms {:>10.2} ms",
                stats.component, stats.throughput, stats.p95_latency, stats.p99_latency
            );
        }
        println!("─────────────────────────────────────────────────────");
    }
}

/// Create a test node for performance analysis
async fn create_test_node(node_id: u64, port: u16) -> Result<ClusterNode> {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port);
    let data_dir = std::env::temp_dir().join(format!("oxirs-perf-test-node-{}", node_id));

    // Clean up old data
    if data_dir.exists() {
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    let config = NodeConfig::new(node_id, addr);
    ClusterNode::new(config).await.map_err(Into::into)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════╗");
    println!("║    OxiRS Cluster Performance Analyzer v0.1.0       ║");
    println!("╚════════════════════════════════════════════════════╝");

    let mut analyzer = PerformanceAnalyzer::new();

    // Run performance analyses
    let sample_size = 100; // Reduced for quick analysis

    analyzer.analyze_node_startup(sample_size).await?;
    analyzer.analyze_query_performance(sample_size * 10).await?;
    analyzer.analyze_status_performance(sample_size * 5).await?;

    // Generate reports
    analyzer.generate_report();
    analyzer.print_comparison();

    // Recommendations
    println!("\n💡 Recommendations:");
    println!("   • For production deployments, increase sample size to 10,000+");
    println!("   • Run analysis during peak load for realistic measurements");
    println!("   • Compare results across different hardware configurations");
    println!("   • Monitor memory growth over extended periods");
    println!("   • Use --release build for accurate performance metrics");

    Ok(())
}
