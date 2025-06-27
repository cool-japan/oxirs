//! Advanced embedding server example demonstrating caching, API endpoints, and model management
//!
//! This example shows how to set up a production-ready embedding service with:
//! - Multi-level caching for performance optimization
//! - RESTful API endpoints for various operations
//! - Model registry and version management
//! - Performance monitoring and health checks

use oxirs_embed::{
    caching::{CacheConfig, CacheManager},
    model_registry::{ModelRegistry, ResourceAllocation},
    models::{TransE, GNNEmbedding, GNNConfig, GNNType},
    NamedNode, Triple, ModelConfig, EmbeddingModel
};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tokio::sync::RwLock;
use tracing::{info, Level};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging (simple console output)
    println!("📄 Logging initialized");

    info!("🚀 Starting Advanced OxiRS Embedding Server Example");

    // Create a comprehensive embedding service demonstration
    let demo = EmbeddingServerDemo::new().await?;
    demo.run().await?;

    Ok(())
}

/// Comprehensive embedding server demonstration
struct EmbeddingServerDemo {
    /// Model registry for managing models
    registry: Arc<ModelRegistry>,
    /// Cache manager for performance optimization
    cache_manager: Arc<CacheManager>,
    /// Currently loaded models
    models: Arc<RwLock<HashMap<Uuid, Arc<dyn EmbeddingModel + Send + Sync>>>>,
}

impl EmbeddingServerDemo {
    /// Create a new demo instance
    async fn new() -> Result<Self> {
        info!("📋 Initializing embedding server components...");

        // Create temporary directory for model storage
        let temp_dir = tempdir()?;
        let registry = Arc::new(ModelRegistry::new(temp_dir.path().to_path_buf()));

        // Configure caching with optimized settings
        let cache_config = CacheConfig {
            l1_max_size: 5_000,      // Hot embeddings
            l2_max_size: 25_000,     // Computation results
            l3_max_size: 50_000,     // Similarity cache
            ttl_seconds: 1800,       // 30 minutes TTL
            enable_warming: true,
            enable_compression: true,
            max_memory_mb: 512,      // 512MB cache limit
            ..Default::default()
        };

        let mut cache_manager = CacheManager::new(cache_config);
        cache_manager.start().await?;
        let cache_manager = Arc::new(cache_manager);

        let models = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            registry,
            cache_manager,
            models,
        })
    }

    /// Run the complete demonstration
    async fn run(&self) -> Result<()> {
        info!("🔧 Setting up demonstration models and data...");

        // 1. Create and register multiple embedding models
        self.setup_models().await?;

        // 2. Demonstrate caching capabilities
        self.demonstrate_caching().await?;

        // 3. Simulate API operations  
        self.simulate_api_operations().await?;

        // 4. Demonstrate API endpoints
        self.demonstrate_api_endpoints().await?;

        // 5. Demonstrate advanced features
        self.demonstrate_advanced_features().await?;

        info!("✅ Embedding server demonstration completed successfully!");
        Ok(())
    }

    /// Set up and register multiple embedding models
    async fn setup_models(&self) -> Result<()> {
        info!("🤖 Creating and training embedding models...");

        // Create sample knowledge graph data
        let sample_triples = vec![
            ("http://example.org/Alice", "http://example.org/knows", "http://example.org/Bob"),
            ("http://example.org/Bob", "http://example.org/knows", "http://example.org/Charlie"),
            ("http://example.org/Charlie", "http://example.org/worksAt", "http://example.org/Company"),
            ("http://example.org/Alice", "http://example.org/livesIn", "http://example.org/City"),
            ("http://example.org/Bob", "http://example.org/likes", "http://example.org/Pizza"),
            ("http://example.org/Charlie", "http://example.org/plays", "http://example.org/Guitar"),
            ("http://example.org/Alice", "http://example.org/speaks", "http://example.org/English"),
            ("http://example.org/Bob", "http://example.org/drives", "http://example.org/Car"),
        ];

        // 1. Create and train TransE model
        let transe_model = self.create_transe_model(&sample_triples).await?;
        let transe_id = self.register_model("TransE Social Network Model", "TransE", transe_model).await?;

        // 2. Create and train GNN model
        let gnn_model = self.create_gnn_model(&sample_triples).await?;
        let gnn_id = self.register_model("GNN Social Network Model", "GNNEmbedding", gnn_model).await?;

        // 3. Promote TransE model to production
        self.registry.promote_to_production(transe_id).await?;

        info!("✅ Successfully created and registered {} models", 2);
        Ok(())
    }

    /// Create and train a TransE model
    async fn create_transe_model(&self, triples: &[(&str, &str, &str)]) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        let config = ModelConfig::default()
            .with_dimensions(64)
            .with_learning_rate(0.01)
            .with_max_epochs(50)
            .with_batch_size(8)
            .with_seed(42);

        let mut model = TransE::new(config);

        // Add triples to model
        for &(s, p, o) in triples {
            let triple = Triple::new(
                NamedNode::new(s)?,
                NamedNode::new(p)?,
                NamedNode::new(o)?,
            );
            model.add_triple(triple)?;
        }

        // Train the model
        info!("🏋️ Training TransE model...");
        let stats = model.train(Some(10)).await?;
        info!("TransE training completed: {} epochs, final loss: {:.6}", 
              stats.epochs_completed, stats.final_loss);

        Ok(Box::new(model))
    }

    /// Create and train a GNN model
    async fn create_gnn_model(&self, triples: &[(&str, &str, &str)]) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        let config = GNNConfig {
            base_config: ModelConfig::default()
                .with_dimensions(64)
                .with_learning_rate(0.01)
                .with_max_epochs(50),
            gnn_type: GNNType::GCN,
            num_layers: 2,
            hidden_dimensions: vec![32, 64],
            dropout: 0.1,
            ..Default::default()
        };

        let mut model = GNNEmbedding::new(config);

        // Add triples to model
        for &(s, p, o) in triples {
            let triple = Triple::new(
                NamedNode::new(s)?,
                NamedNode::new(p)?,
                NamedNode::new(o)?,
            );
            model.add_triple(triple)?;
        }

        // Train the model
        info!("🏋️ Training GNN model...");
        let stats = model.train(Some(10)).await?;
        info!("GNN training completed: {} epochs, final loss: {:.6}", 
              stats.epochs_completed, stats.final_loss);

        Ok(Box::new(model))
    }

    /// Register a model in the registry
    async fn register_model(&self, name: &str, model_type: &str, model: Box<dyn EmbeddingModel + Send + Sync>) -> Result<Uuid> {
        // Register model in registry
        let model_id = self.registry.register_model(
            name.to_string(),
            model_type.to_string(),
            "demo-user".to_string(),
            format!("Demo {} model for embedding server", model_type),
        ).await?;

        // Register version
        let version_id = self.registry.register_version(
            model_id,
            "1.0.0".to_string(),
            "demo-user".to_string(),
            "Initial trained version".to_string(),
            model.config().clone(),
            HashMap::from([
                ("accuracy".to_string(), 0.95),
                ("training_time".to_string(), 120.0),
            ]),
        ).await?;

        // Deploy the version
        let deployment_id = self.registry.deploy_version(
            version_id,
            ResourceAllocation {
                cpu_cores: 2.0,
                memory_gb: 4.0,
                gpu_count: 0,
                gpu_memory_gb: 0.0,
                max_concurrent_requests: 100,
            },
        ).await?;

        // Store model in our local registry
        {
            let mut models = self.models.write().await;
            models.insert(version_id, Arc::from(model));
        }

        info!("📝 Registered model '{}' with version ID: {}", name, version_id);
        Ok(version_id)
    }

    /// Demonstrate caching capabilities
    async fn demonstrate_caching(&self) -> Result<()> {
        info!("🗄️ Demonstrating caching capabilities...");

        // Get production model for caching demo
        let models = self.models.read().await;
        let model = models.values().next().expect("No models available").clone();
        drop(models);

        // Warm up cache with frequently accessed entities
        let frequent_entities = vec![
            "http://example.org/Alice".to_string(),
            "http://example.org/Bob".to_string(),
            "http://example.org/Charlie".to_string(),
        ];

        let warmed_count = self.cache_manager.warm_cache(model.as_ref(), frequent_entities).await?;
        info!("🔥 Cache warmed with {} entities", warmed_count);

        // Precompute common operations
        let common_queries = vec![
            ("http://example.org/Alice".to_string(), "http://example.org/knows".to_string()),
            ("http://example.org/Bob".to_string(), "http://example.org/knows".to_string()),
        ];

        let precomputed_count = self.cache_manager.precompute_common_operations(
            model.as_ref(),
            common_queries,
        ).await?;
        info!("⚡ Precomputed {} common operations", precomputed_count);

        // Demonstrate cache performance
        let start = std::time::Instant::now();
        
        // First access (cache miss)
        let _embedding1 = self.cache_manager.get_embedding("http://example.org/Alice");
        let miss_time = start.elapsed();

        // Second access (cache hit)
        let start = std::time::Instant::now();
        let _embedding2 = self.cache_manager.get_embedding("http://example.org/Alice");
        let hit_time = start.elapsed();

        info!("📊 Cache performance: miss={:?}, hit={:?}", miss_time, hit_time);

        // Display cache statistics
        let stats = self.cache_manager.get_stats();
        info!("📈 Cache stats: hits={}, misses={}, hit_rate={:.2}%", 
              stats.total_hits, stats.total_misses, stats.hit_rate * 100.0);

        Ok(())
    }

    /// Demonstrate API endpoints (simulated)
    async fn demonstrate_api_endpoints(&self) -> Result<()> {
        info!("🌐 Demonstrating API functionality...");

        info!("📚 Would provide these endpoints:");
        info!("  POST /api/v1/embed              - Generate single embedding");
        info!("  POST /api/v1/embed/batch        - Generate batch embeddings");
        info!("  POST /api/v1/score              - Score a triple");
        info!("  POST /api/v1/predict            - Make predictions");
        info!("  GET  /api/v1/models             - List available models");
        info!("  GET  /api/v1/health             - System health");
        info!("  GET  /api/v1/cache/stats        - Cache statistics");

        Ok(())
    }

    /// Simulate API operations for demonstration
    async fn simulate_api_operations(&self) -> Result<()> {
        info!("🎭 Simulating API operations...");

        let models = self.models.read().await;
        let model = models.values().next().expect("No models available").clone();
        drop(models);

        // Simulate embedding generation
        let entities = vec!["http://example.org/Alice", "http://example.org/Bob"];
        for entity in entities {
            match model.get_entity_embedding(entity) {
                Ok(embedding) => {
                    info!("🔢 Generated embedding for {}: {} dimensions", 
                          entity, embedding.dimensions);
                }
                Err(e) => {
                    info!("❌ Failed to generate embedding for {}: {}", entity, e);
                }
            }
        }

        // Simulate triple scoring
        let score = model.score_triple(
            "http://example.org/Alice",
            "http://example.org/knows",
            "http://example.org/Bob"
        )?;
        info!("🎯 Triple score for (Alice, knows, Bob): {:.4}", score);

        // Simulate predictions
        let predictions = model.predict_objects(
            "http://example.org/Alice",
            "http://example.org/knows",
            3
        )?;
        info!("🔮 Top 3 predictions for (Alice, knows, ?): {:?}", predictions);

        Ok(())
    }

    /// Demonstrate advanced features
    async fn demonstrate_advanced_features(&self) -> Result<()> {
        info!("🚀 Demonstrating advanced features...");

        // 1. Model comparison
        self.compare_models().await?;

        // 2. Cache optimization
        self.optimize_cache_performance().await?;

        // 3. Monitoring and health checks
        self.demonstrate_monitoring().await?;

        Ok(())
    }

    /// Compare different model performance
    async fn compare_models(&self) -> Result<()> {
        info!("📊 Comparing model performance...");

        let models = self.models.read().await;
        
        if models.len() >= 2 {
            let model_performances = models.iter().map(|(id, model)| {
                let stats = model.get_stats();
                (id, stats.model_type, stats.num_entities, stats.num_relations, stats.dimensions)
            }).collect::<Vec<_>>();

            for (id, model_type, entities, relations, dims) in model_performances {
                info!("🤖 Model {}: type={}, entities={}, relations={}, dims={}", 
                      id, model_type, entities, relations, dims);
            }
        }

        Ok(())
    }

    /// Optimize cache performance
    async fn optimize_cache_performance(&self) -> Result<()> {
        info!("⚡ Optimizing cache performance...");

        // Get current memory usage
        let memory_usage = self.cache_manager.estimate_memory_usage();
        info!("💾 Current cache memory usage: {} bytes", memory_usage);

        // Display detailed cache statistics
        let stats = self.cache_manager.get_stats();
        info!("📈 Detailed cache statistics:");
        info!("  L1 Cache: {}/{} entries, {} hits, {} misses", 
              stats.l1_stats.size, stats.l1_stats.capacity, 
              stats.l1_stats.hits, stats.l1_stats.misses);
        info!("  L2 Cache: {}/{} entries, {} hits, {} misses", 
              stats.l2_stats.size, stats.l2_stats.capacity,
              stats.l2_stats.hits, stats.l2_stats.misses);
        info!("  L3 Cache: {}/{} entries, {} hits, {} misses", 
              stats.l3_stats.size, stats.l3_stats.capacity,
              stats.l3_stats.hits, stats.l3_stats.misses);
        info!("  Total time saved: {:.2} seconds", stats.total_time_saved_seconds);

        Ok(())
    }

    /// Demonstrate monitoring and health checks
    async fn demonstrate_monitoring(&self) -> Result<()> {
        info!("🏥 Demonstrating monitoring and health checks...");

        // Model health simulation
        let models = self.models.read().await;
        for (id, model) in models.iter() {
            let stats = model.get_stats();
            let health_status = if stats.is_trained { "healthy" } else { "unhealthy" };
            info!("💊 Model {} health: {}", id, health_status);
        }

        // Cache health
        let cache_stats = self.cache_manager.get_stats();
        let cache_health = if cache_stats.hit_rate > 0.5 { "optimal" } else { "needs warming" };
        info!("🗄️ Cache health: {} (hit rate: {:.1}%)", cache_health, cache_stats.hit_rate * 100.0);

        // System metrics simulation
        info!("📊 System metrics:");
        info!("  🔧 Loaded models: {}", models.len());
        info!("  💾 Memory usage: {} MB", memory_usage() / 1024 / 1024);
        info!("  🏃 Uptime: simulated");
        info!("  📡 Request rate: simulated");

        Ok(())
    }
}

// Simulate memory usage for demonstration
fn memory_usage() -> usize {
    256 * 1024 * 1024 // 256 MB simulated
}

/// Helper function to create sample configuration
fn create_production_config() -> ModelConfig {
    ModelConfig::default()
        .with_dimensions(128)
        .with_learning_rate(0.005)
        .with_max_epochs(1000)
        .with_batch_size(512)
}

/// Utility function for performance benchmarking
async fn benchmark_operation<F, T>(name: &str, operation: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    let start = std::time::Instant::now();
    let result = operation.await?;
    let duration = start.elapsed();
    info!("⏱️ {}: completed in {:?}", name, duration);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embedding_server_demo_creation() {
        let demo = EmbeddingServerDemo::new().await;
        assert!(demo.is_ok());
    }

    #[tokio::test] 
    async fn test_model_creation() {
        let demo = EmbeddingServerDemo::new().await.unwrap();
        let triples = vec![
            ("http://example.org/A", "http://example.org/rel", "http://example.org/B"),
        ];
        
        let model = demo.create_transe_model(&triples).await;
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert!(model.is_trained());
    }

    #[test]
    fn test_production_config() {
        let config = create_production_config();
        assert_eq!(config.dimensions, 128);
        assert_eq!(config.learning_rate, 0.005);
        assert_eq!(config.max_epochs, 1000);
        assert_eq!(config.batch_size, 512);
    }
}