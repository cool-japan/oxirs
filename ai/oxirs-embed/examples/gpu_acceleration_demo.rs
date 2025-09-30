//! GPU Acceleration and Optimization Demo
//!
//! This example demonstrates the advanced GPU acceleration features in OxiRS Embed
//! including memory pooling, tensor caching, mixed precision training, and
//! multi-stream processing for high-performance embedding generation.

use anyhow::Result;
use oxirs_embed::{
    EmbeddingModel, GpuAccelerationConfig, GpuAccelerationManager, GpuMemoryPool,
    MixedPrecisionProcessor, ModelConfig, MultiStreamProcessor, NamedNode, TensorCache, TransE,
    Triple,
};
use scirs2_core::ndarray_ext::{Array1, Array2};
use std::time::Instant;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 OxiRS GPU Acceleration Demo");
    println!("==============================\n");

    // 1. GPU Memory Management
    demo_gpu_memory_management().await?;

    // 2. Tensor Caching System
    demo_tensor_caching().await?;

    // 3. Mixed Precision Training
    demo_mixed_precision().await?;

    // 4. Multi-Stream Processing
    demo_multi_stream_processing().await?;

    // 5. End-to-End Accelerated Training
    demo_accelerated_training().await?;

    // 6. Performance Benchmarks
    demo_performance_benchmarks().await?;

    Ok(())
}

/// Demonstrate GPU memory management and pooling
async fn demo_gpu_memory_management() -> Result<()> {
    println!("💾 1. GPU Memory Management");
    println!("──────────────────────────\n");

    // Create GPU memory pool with custom configuration
    let config = GpuAccelerationConfig {
        enabled: true,
        device_ids: vec![0, 1],    // Multi-GPU setup
        memory_pool_size_mb: 4096, // 4GB pool
        mixed_precision: true,
        tensor_caching: true,
        cache_size_mb: 1024, // 1GB cache
        kernel_fusion: true,
        memory_mapping: true,
        unified_memory: false,
        multi_stream: true,
        num_streams: 8,
        pipeline_parallelism: true,
        pipeline_stages: 4,
    };

    let memory_pool = GpuMemoryPool::new(config.clone());

    println!("🔧 Configuring GPU memory pool:");
    println!("   Pool size: {} MB", config.memory_pool_size_mb);
    println!("   Devices: {:?}", config.device_ids);
    println!("   Streams: {}", config.num_streams);
    println!("   Pipeline stages: {}", config.pipeline_stages);

    // Demonstrate memory allocation and deallocation
    println!("\n📦 Memory allocation demonstration:");

    let mut allocated_blocks = Vec::new();

    // Allocate several memory blocks
    for i in 0..5 {
        let size_mb = (i + 1) * 128; // 128MB, 256MB, 384MB, 512MB, 640MB
        let size_bytes = size_mb * 1024 * 1024;
        let device_id = i % config.device_ids.len();

        println!("   Allocating {size_mb} MB on device {device_id}...");
        let block_id = memory_pool.allocate(size_bytes, device_id)?;
        allocated_blocks.push(block_id);

        // Show allocation stats
        let stats = memory_pool.get_stats();
        println!(
            "     Block ID: {}, Current usage: {} MB",
            block_id,
            stats.current_memory_usage / (1024 * 1024)
        );
    }

    // Deallocate some blocks
    println!("\n🗑️  Deallocating blocks for reuse:");
    for &block_id in &allocated_blocks[..3] {
        memory_pool.deallocate(block_id)?;
        println!("   Deallocated block {block_id}");
    }

    // Allocate new blocks (should reuse deallocated ones)
    println!("\n♻️  Reallocating (should reuse freed blocks):");
    for _i in 0..2 {
        let size_bytes = 256 * 1024 * 1024; // 256MB
        let block_id = memory_pool.allocate(size_bytes, 0)?;
        println!("   Reused block ID: {block_id}");
    }

    // Show final statistics
    let final_stats = memory_pool.get_stats();
    println!("\n📊 Final memory statistics:");
    println!("   Total allocations: {}", final_stats.total_allocations);
    println!(
        "   Total deallocations: {}",
        final_stats.total_deallocations
    );
    println!(
        "   Peak usage: {} MB",
        final_stats.peak_memory_usage / (1024 * 1024)
    );
    println!(
        "   Current usage: {} MB",
        final_stats.current_memory_usage / (1024 * 1024)
    );
    println!("   Cache hits: {}", final_stats.cache_hits);
    println!("   Cache misses: {}", final_stats.cache_misses);

    // Demonstrate memory defragmentation
    println!("\n🔧 Performing memory defragmentation...");
    memory_pool.defragment()?;
    println!("   ✅ Defragmentation completed");

    println!();
    Ok(())
}

/// Demonstrate intelligent tensor caching
async fn demo_tensor_caching() -> Result<()> {
    println!("🗂️  2. Tensor Caching System");
    println!("──────────────────────────\n");

    let config = GpuAccelerationConfig::default();
    let tensor_cache = TensorCache::new(config);

    println!("💾 Demonstrating tensor caching capabilities:");

    // Cache some entity tensors
    let entity_tensors = vec![
        (
            "entity_1",
            Array2::from_shape_vec(
                (64, 128),
                (0..8192).map(|i| i as f32 / 1000.0).collect(),
            )?,
        ),
        (
            "entity_2",
            Array2::from_shape_vec(
                (64, 128),
                (1000..9192).map(|i| i as f32 / 1000.0).collect(),
            )?,
        ),
        (
            "entity_3",
            Array2::from_shape_vec(
                (64, 128),
                (2000..10192).map(|i| i as f32 / 1000.0).collect(),
            )?,
        ),
    ];

    // Cache entity tensors
    println!("\n📥 Caching entity tensors:");
    for (entity, tensor) in &entity_tensors {
        tensor_cache.cache_entity_tensor(entity, tensor.clone(), 0);
        println!(
            "   Cached tensor for {} (shape: {:?})",
            entity,
            tensor.shape()
        );
    }

    // Cache attention weights
    println!("\n🎯 Caching attention weights:");
    let attention_weights = Array2::from_shape_vec(
        (32, 32),
        (0..1024).map(|i| (i as f32).sin() / 10.0).collect(),
    )?;
    tensor_cache.cache_attention_weights("attention_layer_1", attention_weights.clone(), 0);
    println!(
        "   Cached attention weights (shape: {:?})",
        attention_weights.shape()
    );

    // Demonstrate cache hits and misses
    println!("\n🎯 Testing cache performance:");

    let start = Instant::now();
    for i in 0..1000 {
        let entity = format!("entity_{}", (i % 3) + 1);
        if let Some(cached_tensor) = tensor_cache.get_entity_tensor(&entity) {
            // Simulate using the tensor
            let _norm = cached_tensor.iter().map(|x| x * x).sum::<f32>().sqrt();
        }
    }
    let cache_time = start.elapsed();

    // Test cache misses
    let start = Instant::now();
    for i in 0..100 {
        let entity = format!("missing_entity_{i}");
        let _result = tensor_cache.get_entity_tensor(&entity);
    }
    let miss_time = start.elapsed();

    println!("   Cache hits (1000 ops): {cache_time:?}");
    println!("   Cache misses (100 ops): {miss_time:?}");

    // Show cache statistics
    let cache_stats = tensor_cache.get_stats();
    println!("\n📊 Cache statistics:");
    println!("   Hits: {}", cache_stats.hits);
    println!("   Misses: {}", cache_stats.misses);
    println!(
        "   Hit rate: {:.2}%",
        (cache_stats.hits as f64 / (cache_stats.hits + cache_stats.misses) as f64) * 100.0
    );
    println!(
        "   Memory usage: {} MB",
        cache_stats.total_memory_usage / (1024 * 1024)
    );
    println!("   Evictions: {}", cache_stats.evictions);

    println!();
    Ok(())
}

/// Demonstrate mixed precision training and inference
async fn demo_mixed_precision() -> Result<()> {
    println!("⚡ 3. Mixed Precision Training");
    println!("────────────────────────────\n");

    let config = GpuAccelerationConfig {
        mixed_precision: true,
        ..Default::default()
    };

    let mut mixed_precision = MixedPrecisionProcessor::new(config);

    println!("🔢 Demonstrating mixed precision capabilities:");

    // Create sample tensors for demonstration
    let fp32_tensor = Array2::from_shape_vec(
        (128, 256),
        (0..32768).map(|i| (i as f32) / 1000.0 + 0.5).collect(),
    )?;

    println!("\n📊 Original FP32 tensor:");
    println!("   Shape: {:?}", fp32_tensor.shape());
    println!(
        "   Min value: {:.6}",
        fp32_tensor.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "   Max value: {:.6}",
        fp32_tensor.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!("   Mean: {:.6}", fp32_tensor.mean().unwrap());

    // Convert to FP16 for computation
    let fp16_tensor = mixed_precision.to_fp16(&fp32_tensor);
    println!("\n🎯 Converted to FP16:");
    println!("   Shape: {:?}", fp16_tensor.shape());
    println!(
        "   Min value: {:.6}",
        fp16_tensor.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "   Max value: {:.6}",
        fp16_tensor.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!("   Mean: {:.6}", fp16_tensor.mean().unwrap());

    // Demonstrate precision difference
    let precision_loss = (&fp32_tensor - &fp16_tensor)
        .mapv(|x| x.abs())
        .mean()
        .unwrap();
    println!("   Precision loss: {precision_loss:.8}");

    // Demonstrate loss scaling
    println!("\n⚖️  Loss scaling demonstration:");
    let base_loss = 0.001234f32;
    let scaled_loss = mixed_precision.scale_loss(base_loss);
    println!("   Original loss: {base_loss:.6}");
    println!("   Scaled loss: {scaled_loss:.2}");
    println!("   Scaling factor: {:.0}", scaled_loss / base_loss);

    // Simulate gradient computation and unscaling
    println!("\n🔄 Gradient processing:");
    let mut gradients = Array2::from_shape_vec(
        (64, 128),
        (0..8192).map(|i| (i as f32) / 10000.0).collect(),
    )?;

    println!(
        "   Original gradient norm: {:.6}",
        gradients.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    let success = mixed_precision.unscale_gradients(&mut gradients);
    println!("   Unscaling successful: {success}");
    println!(
        "   Unscaled gradient norm: {:.6}",
        gradients.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // Demonstrate overflow detection and loss scaling adjustment
    println!("\n🚨 Overflow detection:");
    let mut overflow_gradients = Array2::from_elem((32, 32), f32::INFINITY);
    let overflow_detected = !mixed_precision.unscale_gradients(&mut overflow_gradients);
    println!("   Overflow detected: {overflow_detected}");

    if overflow_detected {
        mixed_precision.adjust_loss_scaling(true);
        println!("   Loss scaling reduced for next iteration");
    }

    println!();
    Ok(())
}

/// Demonstrate multi-stream parallel processing
async fn demo_multi_stream_processing() -> Result<()> {
    println!("🌊 4. Multi-Stream Processing");
    println!("────────────────────────────\n");

    let config = GpuAccelerationConfig {
        multi_stream: true,
        num_streams: 4,
        ..Default::default()
    };

    let mut multi_stream = MultiStreamProcessor::new(config);

    println!("🚀 Demonstrating parallel GPU streams:");
    println!("   Number of streams: {}", multi_stream.stream_ids.len());

    // Create a batch of entities to process
    let entities = (0..16).map(|i| format!("entity_{i}")).collect::<Vec<_>>();
    println!("   Processing {} entities in parallel", entities.len());

    // Define a mock embedding computation function
    let compute_embedding = |entity: String, stream_id: usize| -> Array1<f32> {
        // Simulate complex embedding computation
        let seed = entity.len() + stream_id;
        let embedding: Vec<f32> = (0..128).map(|i| ((seed + i) as f32).sin()).collect();
        Array1::from_vec(embedding)
    };

    // Measure serial processing time
    println!("\n⏱️  Performance comparison:");
    let start = Instant::now();
    let mut serial_results = Vec::new();
    for entity in &entities {
        let embedding = compute_embedding(entity.clone(), 0);
        serial_results.push(embedding);
    }
    let serial_time = start.elapsed();
    println!("   Serial processing: {serial_time:?}");

    // Measure parallel processing time
    let start = Instant::now();
    let parallel_results = multi_stream
        .process_batch_parallel(entities.clone(), compute_embedding)
        .await?;
    let parallel_time = start.elapsed();
    println!("   Parallel processing: {parallel_time:?}");

    // Calculate speedup
    let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();
    println!("   Speedup: {speedup:.2}x");

    // Verify results are equivalent
    let results_match = serial_results.len() == parallel_results.len()
        && serial_results
            .iter()
            .zip(&parallel_results)
            .all(|(a, b)| a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-6));
    println!(
        "   Results match: {}",
        if results_match { "✅" } else { "❌" }
    );

    // Demonstrate stream assignment
    println!("\n🔄 Stream assignment demonstration:");
    for i in 0..8 {
        let stream_id = multi_stream.get_next_stream();
        println!("   Task {i} → Stream {stream_id}");
    }

    multi_stream.synchronize_all();
    println!("   ✅ All streams synchronized");

    println!();
    Ok(())
}

/// Demonstrate end-to-end accelerated training
async fn demo_accelerated_training() -> Result<()> {
    println!("🎓 5. End-to-End Accelerated Training");
    println!("────────────────────────────────────\n");

    // Create GPU acceleration manager
    let config = GpuAccelerationConfig {
        enabled: true,
        memory_pool_size_mb: 2048,
        mixed_precision: true,
        tensor_caching: true,
        cache_size_mb: 512,
        multi_stream: true,
        num_streams: 4,
        ..Default::default()
    };

    let mut gpu_manager = GpuAccelerationManager::new(config);

    println!("🔧 GPU acceleration configuration:");
    println!("   Memory pool: 2048 MB");
    println!("   Tensor cache: 512 MB");
    println!("   Mixed precision: enabled");
    println!("   Multi-stream: 4 streams");

    // Create a TransE model for demonstration
    let model_config = ModelConfig {
        dimensions: 128,
        learning_rate: 0.01,
        batch_size: 1000,
        max_epochs: 50,
        use_gpu: true,
        ..Default::default()
    };

    let mut model = TransE::new(model_config);

    // Add sample knowledge graph data
    println!("\n📚 Creating sample knowledge graph:");
    let sample_triples = vec![
        (
            "http://example.org/person/alice",
            "http://example.org/knows",
            "http://example.org/person/bob",
        ),
        (
            "http://example.org/person/bob",
            "http://example.org/works_at",
            "http://example.org/company/tech_corp",
        ),
        (
            "http://example.org/person/alice",
            "http://example.org/lives_in",
            "http://example.org/city/new_york",
        ),
        (
            "http://example.org/company/tech_corp",
            "http://example.org/located_in",
            "http://example.org/city/san_francisco",
        ),
        (
            "http://example.org/person/charlie",
            "http://example.org/knows",
            "http://example.org/person/alice",
        ),
        (
            "http://example.org/person/bob",
            "http://example.org/friend_of",
            "http://example.org/person/charlie",
        ),
    ];

    for (s, p, o) in sample_triples {
        let triple = Triple::new(NamedNode::new(s)?, NamedNode::new(p)?, NamedNode::new(o)?);
        model.add_triple(triple)?;
    }

    println!(
        "   Added {} triples to the model",
        model.get_stats().num_triples
    );

    // Define embedding computation function
    let embedding_fn = |entity: &str| -> Array1<f32> {
        // Simulate embedding computation
        let hash = entity.bytes().map(|b| b as f32).sum::<f32>();
        let embedding: Vec<f32> = (0..128).map(|i| (hash + i as f32).sin() / 10.0).collect();
        Array1::from_vec(embedding)
    };

    // Demonstrate accelerated embedding generation
    println!("\n🚀 Accelerated embedding generation:");
    let entities = model.get_entities();
    println!("   Processing {} entities", entities.len());

    let start = Instant::now();
    let accelerated_embeddings = gpu_manager
        .accelerated_embedding_generation(entities.clone(), embedding_fn)
        .await?;
    let accelerated_time = start.elapsed();

    println!("   Accelerated generation: {accelerated_time:?}");
    println!("   Generated {} embeddings", accelerated_embeddings.len());
    println!(
        "   Average embedding norm: {:.4}",
        accelerated_embeddings
            .iter()
            .map(|emb| emb.iter().map(|x| x * x).sum::<f32>().sqrt())
            .sum::<f32>()
            / accelerated_embeddings.len() as f32
    );

    // Show GPU performance statistics
    println!("\n📊 GPU Performance Statistics:");
    let perf_stats = gpu_manager.get_performance_stats();
    println!("   Memory allocations: {}", perf_stats.memory_allocations);
    println!(
        "   Peak memory usage: {} MB",
        perf_stats.peak_memory_usage_mb
    );
    println!("   Memory pool hits: {}", perf_stats.memory_pool_hits);
    println!("   Memory pool misses: {}", perf_stats.memory_pool_misses);
    println!("   Tensor cache hits: {}", perf_stats.tensor_cache_hits);
    println!("   Tensor cache misses: {}", perf_stats.tensor_cache_misses);
    println!(
        "   Cache hit rate: {:.2}%",
        (perf_stats.tensor_cache_hits as f64
            / (perf_stats.tensor_cache_hits + perf_stats.tensor_cache_misses) as f64)
            * 100.0
    );
    println!("   Active streams: {}", perf_stats.num_active_streams);
    println!(
        "   Loss scaling factor: {:.1}",
        perf_stats.loss_scaling_factor
    );

    println!();
    Ok(())
}

/// Demonstrate comprehensive performance benchmarks
async fn demo_performance_benchmarks() -> Result<()> {
    println!("🏁 6. Performance Benchmarks");
    println!("───────────────────────────\n");

    // Test different configurations
    let configs = vec![
        (
            "Baseline",
            GpuAccelerationConfig {
                enabled: false,
                ..Default::default()
            },
        ),
        (
            "GPU Basic",
            GpuAccelerationConfig {
                enabled: true,
                mixed_precision: false,
                tensor_caching: false,
                multi_stream: false,
                ..Default::default()
            },
        ),
        (
            "GPU + Mixed Precision",
            GpuAccelerationConfig {
                enabled: true,
                mixed_precision: true,
                tensor_caching: false,
                multi_stream: false,
                ..Default::default()
            },
        ),
        (
            "GPU + Caching",
            GpuAccelerationConfig {
                enabled: true,
                mixed_precision: false,
                tensor_caching: true,
                multi_stream: false,
                ..Default::default()
            },
        ),
        (
            "GPU Full Optimization",
            GpuAccelerationConfig {
                enabled: true,
                mixed_precision: true,
                tensor_caching: true,
                multi_stream: true,
                num_streams: 8,
                ..Default::default()
            },
        ),
    ];

    println!("🔥 Running performance benchmarks:");

    for (name, config) in configs {
        println!("\n📊 Configuration: {name}");

        let mut gpu_manager = GpuAccelerationManager::new(config);

        // Create larger dataset for benchmarking
        let entities: Vec<String> = (0..1000).map(|i| format!("entity_{i}")).collect();

        // Benchmark embedding function
        let benchmark_fn = |entity: &str| -> Array1<f32> {
            let hash = entity.bytes().map(|b| b as f32).sum::<f32>();
            let embedding: Vec<f32> = (0..256)
                .map(|i| {
                    let val = (hash + i as f32).sin() * (hash + i as f32).cos();
                    val / 100.0
                })
                .collect();
            Array1::from_vec(embedding)
        };

        // Run benchmark
        let start = Instant::now();
        let results = gpu_manager
            .accelerated_embedding_generation(entities.clone(), benchmark_fn)
            .await?;
        let duration = start.elapsed();

        // Calculate metrics
        let throughput = entities.len() as f64 / duration.as_secs_f64();
        let avg_latency = duration.as_micros() as f64 / entities.len() as f64;

        println!("   Entities processed: {}", results.len());
        println!("   Total time: {duration:?}");
        println!("   Throughput: {throughput:.1} embeddings/sec");
        println!("   Average latency: {avg_latency:.1} μs/embedding");

        // Memory efficiency
        let perf_stats = gpu_manager.get_performance_stats();
        println!("   Peak memory: {} MB", perf_stats.peak_memory_usage_mb);
        println!(
            "   Cache hit rate: {:.1}%",
            if perf_stats.tensor_cache_hits + perf_stats.tensor_cache_misses > 0 {
                (perf_stats.tensor_cache_hits as f64
                    / (perf_stats.tensor_cache_hits + perf_stats.tensor_cache_misses) as f64)
                    * 100.0
            } else {
                0.0
            }
        );

        // Brief pause between benchmarks
        sleep(Duration::from_millis(100)).await;
    }

    println!("\n🎉 Performance benchmarking completed!");
    println!("\n💡 Key Takeaways:");
    println!("   • GPU acceleration provides significant speedup for large batches");
    println!("   • Mixed precision reduces memory usage with minimal accuracy loss");
    println!("   • Tensor caching improves performance for repeated computations");
    println!("   • Multi-stream processing maximizes GPU utilization");
    println!("   • Combined optimizations deliver exponential performance gains");

    Ok(())
}
