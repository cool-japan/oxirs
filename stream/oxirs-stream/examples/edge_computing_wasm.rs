//! # Edge Computing with WebAssembly Example
//!
//! Demonstrates edge computing capabilities using WASM:
//! - Ultra-low latency processing at the edge
//! - Hot-swappable WASM plugins
//! - Multi-region distributed execution
//! - Resource-constrained processing
//! - Security sandboxing

use anyhow::Result;
use oxirs_stream::{
    EdgeLocation, OptimizationLevel, ProcessingSpecialization, StreamEvent, WasmEdgeConfig,
    WasmEdgeProcessor, WasmPlugin, WasmResourceLimits,
};
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("Starting Edge Computing with WASM Example");

    // Demonstrate edge computing capabilities
    basic_edge_processing_example().await?;
    multi_region_edge_example().await?;
    hot_swap_plugin_example().await?;
    specialized_processing_example().await?;

    Ok(())
}

/// Example: Basic edge processing with WASM
async fn basic_edge_processing_example() -> Result<()> {
    info!("=== Basic Edge Processing Example ===");

    // Create edge processor configuration
    let config = WasmEdgeConfig {
        optimization_level: OptimizationLevel::Release,
        resource_limits: WasmResourceLimits {
            max_memory_bytes: 100 * 1024 * 1024, // 100 MB
            max_execution_time_ms: 1000,         // 1 second
            max_stack_size_bytes: 1024 * 1024,   // 1 MB
            max_table_elements: 10000,
            enable_simd: true,
            enable_threads: false,
            ..Default::default()
        },
        enable_caching: true,
        enable_jit: true,
        security_sandbox: true,
        allowed_imports: vec!["env".to_string(), "wasi_snapshot_preview1".to_string()],
        ..Default::default()
    };

    let mut processor = WasmEdgeProcessor::new(config)?;

    // Create a simple WASM plugin for event filtering
    let plugin_wasm = create_simple_filter_plugin()?;
    let plugin = WasmPlugin {
        id: "filter-plugin-v1".to_string(),
        name: "Simple Event Filter".to_string(),
        version: "1.0.0".to_string(),
        description: "Basic event filtering plugin".to_string(),
        author: "OxiRS Team".to_string(),
        capabilities: vec![],
        wasm_bytes: plugin_wasm,
        schema: oxirs_stream::PluginSchema::default(),
        performance_profile: oxirs_stream::PerformanceProfile::default(),
        security_level: oxirs_stream::SecurityLevel::Standard,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };

    // Load plugin into edge processor
    processor.load_plugin(plugin).await?;
    info!("WASM plugin loaded successfully");

    // Process events at the edge
    for i in 0..50 {
        let event = create_test_event(i);
        let result = processor.process(event).await?;

        if let Some(_processed) = result.output {
            info!(
                "Edge processed event {}: latency={:.2}ms",
                i, result.latency_ms
            );
        }

        if i % 10 == 0 {
            let stats = processor.get_stats().await;
            info!(
                "Stats - processed: {}, avg latency: {:.2}ms",
                stats.total_processed, stats.average_latency_ms
            );
        }
    }

    Ok(())
}

/// Example: Multi-region edge deployment
async fn multi_region_edge_example() -> Result<()> {
    info!("=== Multi-Region Edge Deployment Example ===");

    // Define edge locations
    let locations = vec![
        EdgeLocation {
            id: "us-west".to_string(),
            region: "us-west-1".to_string(),
            latency_ms: 5.0,
            capacity_factor: 1.0,
            available_resources: oxirs_stream::ResourceMetrics {
                cpu_cores: 2,
                memory_mb: 512,
                storage_gb: 100,
                network_mbps: 1000.0,
                gpu_units: 0,
                quantum_qubits: 0,
            },
            specializations: vec![
                ProcessingSpecialization::RdfProcessing,
                ProcessingSpecialization::SparqlOptimization,
            ],
        },
        EdgeLocation {
            id: "eu-central".to_string(),
            region: "eu-central-1".to_string(),
            latency_ms: 8.0,
            capacity_factor: 0.9,
            available_resources: oxirs_stream::ResourceMetrics {
                cpu_cores: 2,
                memory_mb: 512,
                storage_gb: 100,
                network_mbps: 1000.0,
                gpu_units: 0,
                quantum_qubits: 0,
            },
            specializations: vec![ProcessingSpecialization::GraphAnalytics],
        },
        EdgeLocation {
            id: "ap-southeast".to_string(),
            region: "ap-southeast-1".to_string(),
            latency_ms: 12.0,
            capacity_factor: 0.8,
            available_resources: oxirs_stream::ResourceMetrics {
                cpu_cores: 1,
                memory_mb: 256,
                storage_gb: 50,
                network_mbps: 500.0,
                gpu_units: 0,
                quantum_qubits: 0,
            },
            specializations: vec![ProcessingSpecialization::MachineLearning],
        },
    ];

    // Create edge processor with optimal location selection
    let config = WasmEdgeConfig {
        optimization_level: OptimizationLevel::Adaptive,
        resource_limits: WasmResourceLimits {
            max_memory_bytes: 200 * 1024 * 1024,
            max_execution_time_ms: 500,
            max_stack_size_bytes: 512 * 1024,
            max_table_elements: 5000,
            enable_simd: true,
            enable_threads: false,
            ..Default::default()
        },
        enable_caching: true,
        enable_jit: true,
        security_sandbox: true,
        allowed_imports: vec!["env".to_string()],
        ..Default::default()
    };

    let processor = WasmEdgeProcessor::new(config)?;

    // Simulate client requests from different regions
    let client_locations = [
        (37.7749, -122.4194), // San Francisco
        (51.5074, -0.1278),   // London
        (1.3521, 103.8198),   // Singapore
    ];

    for (i, (lat, lon)) in client_locations.iter().enumerate() {
        // Select optimal edge location
        let optimal_location = select_optimal_edge_location(&locations, *lat, *lon);
        info!(
            "Client {} -> Routing to edge: {} (latency: {:.2}ms)",
            i, optimal_location.id, optimal_location.latency_ms
        );

        // Process at selected edge
        let event = create_test_event(i as u64);
        let result = processor
            .process_at_location(event, optimal_location)
            .await?;
        info!(
            "Processed at {} edge: total latency={:.2}ms",
            optimal_location.id, result.latency_ms
        );
    }

    Ok(())
}

/// Example: Hot-swapping WASM plugins
async fn hot_swap_plugin_example() -> Result<()> {
    info!("=== Hot-Swap Plugin Example ===");

    let config = WasmEdgeConfig {
        optimization_level: OptimizationLevel::Release,
        resource_limits: WasmResourceLimits {
            max_memory_bytes: 100 * 1024 * 1024,
            max_execution_time_ms: 1000,
            max_stack_size_bytes: 1024 * 1024,
            max_table_elements: 10000,
            enable_simd: true,
            enable_threads: false,
            ..Default::default()
        },
        enable_caching: true,
        enable_jit: true,
        security_sandbox: true,
        allowed_imports: vec!["env".to_string()],
        ..Default::default()
    };

    let mut processor = WasmEdgeProcessor::new(config)?;

    // Load version 1 of plugin
    info!("Loading plugin version 1.0...");
    let plugin_v1 = WasmPlugin {
        id: "processor-v1".to_string(),
        name: "Event Processor".to_string(),
        version: "1.0.0".to_string(),
        description: "Simple event processor v1".to_string(),
        author: "OxiRS Team".to_string(),
        capabilities: vec![],
        wasm_bytes: create_simple_filter_plugin()?,
        schema: oxirs_stream::PluginSchema::default(),
        performance_profile: oxirs_stream::PerformanceProfile::default(),
        security_level: oxirs_stream::SecurityLevel::Standard,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    processor.load_plugin(plugin_v1).await?;

    // Process some events with v1
    for i in 0..5 {
        let event = create_test_event(i);
        processor.process(event).await?;
    }
    info!("Processed 5 events with plugin v1");

    // Hot-swap to version 2 (zero downtime)
    info!("Hot-swapping to plugin version 2.0...");
    let plugin_v2 = WasmPlugin {
        id: "processor-v2".to_string(),
        name: "Event Processor".to_string(),
        version: "2.0.0".to_string(),
        description: "Enhanced event processor v2".to_string(),
        author: "OxiRS Team".to_string(),
        capabilities: vec![],
        wasm_bytes: create_enhanced_filter_plugin()?,
        schema: oxirs_stream::PluginSchema::default(),
        performance_profile: oxirs_stream::PerformanceProfile::default(),
        security_level: oxirs_stream::SecurityLevel::Standard,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    processor.hot_swap_plugin("processor-v1", plugin_v2).await?;
    info!("Plugin hot-swapped successfully!");

    // Continue processing with v2
    for i in 5..10 {
        let event = create_test_event(i);
        processor.process(event).await?;
    }
    info!("Processed 5 events with plugin v2 (zero downtime)");

    Ok(())
}

/// Example: Specialized processing (RDF, SPARQL, ML)
async fn specialized_processing_example() -> Result<()> {
    info!("=== Specialized Processing Example ===");

    // RDF/SPARQL specialized processing
    let rdf_config = WasmEdgeConfig {
        optimization_level: OptimizationLevel::Maximum,
        resource_limits: WasmResourceLimits {
            max_memory_bytes: 500 * 1024 * 1024, // Larger for graph processing
            max_execution_time_ms: 2000,
            max_stack_size_bytes: 2 * 1024 * 1024,
            max_table_elements: 50000,
            enable_simd: true,
            enable_threads: false,
            ..Default::default()
        },
        enable_caching: true,
        enable_jit: true,
        security_sandbox: true,
        allowed_imports: vec!["env".to_string()],
        ..Default::default()
    };

    let _rdf_processor = WasmEdgeProcessor::new(rdf_config)?;

    // Load RDF-specialized plugin
    let _rdf_plugin = WasmPlugin {
        id: "rdf-processor".to_string(),
        name: "RDF Stream Processor".to_string(),
        version: "1.0.0".to_string(),
        description: "RDF and semantic graph processor".to_string(),
        author: "OxiRS Team".to_string(),
        capabilities: vec![],
        wasm_bytes: vec![], // Would contain actual WASM bytecode
        schema: oxirs_stream::PluginSchema::default(),
        performance_profile: oxirs_stream::PerformanceProfile::default(),
        security_level: oxirs_stream::SecurityLevel::High,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };

    info!("RDF-specialized edge processor configured");
    info!("Ready for semantic graph processing at the edge");

    Ok(())
}

// Helper functions

fn create_test_event(id: u64) -> StreamEvent {
    use oxirs_stream::EventMetadata;
    StreamEvent::TripleAdded {
        subject: format!("http://example.org/subject{}", id),
        predicate: "http://example.org/predicate".to_string(),
        object: format!("value{}", id),
        graph: None,
        metadata: EventMetadata {
            event_id: format!("event-{}", id),
            timestamp: chrono::Utc::now(),
            source: "edge-generator".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: std::collections::HashMap::new(),
            checksum: None,
        },
    }
}

fn create_simple_filter_plugin() -> Result<Vec<u8>> {
    // In a real implementation, this would return actual WASM bytecode
    // For this example, return empty vec
    Ok(vec![])
}

fn create_enhanced_filter_plugin() -> Result<Vec<u8>> {
    // Enhanced version of the plugin
    Ok(vec![])
}

fn select_optimal_edge_location(
    locations: &[EdgeLocation],
    _client_lat: f64,
    _client_lon: f64,
) -> &EdgeLocation {
    // Simple latency-based selection
    locations
        .iter()
        .min_by(|a, b| a.latency_ms.partial_cmp(&b.latency_ms).unwrap())
        .unwrap()
}
