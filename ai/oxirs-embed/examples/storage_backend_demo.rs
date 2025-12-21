//! Storage Backend Demo
//!
//! This example demonstrates how to persist knowledge graph embeddings
//! using different storage backends.
//!
//! Run with:
//! ```bash
//! cargo run --example storage_backend_demo --features basic-models
//! ```

use anyhow::Result;
use chrono::Utc;
use oxirs_embed::{
    EmbeddingMetadata,
    EmbeddingModel, // Trait import
    ModelConfig,
    NamedNode,
    StorageBackendConfig,
    StorageBackendManager,
    StorageBackendType,
    TransE,
    Triple,
};
use std::collections::HashMap;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== OxiRS Storage Backend Demo ===\n");

    // Step 1: Create and train a model
    println!("1. Creating and training a TransE model...");
    let model_config = ModelConfig::default()
        .with_dimensions(64)
        .with_learning_rate(0.01)
        .with_max_epochs(50);

    let mut model = TransE::new(model_config.clone());

    // Add sample triples
    let triples = vec![
        ("Alice", "knows", "Bob"),
        ("Bob", "worksFor", "Company"),
        ("Alice", "livesIn", "NewYork"),
        ("Company", "locatedIn", "NewYork"),
        ("Bob", "knows", "Charlie"),
    ];

    for (s, p, o) in &triples {
        let triple = Triple::new(
            NamedNode::new(&format!("http://example.org/{}", s))?,
            NamedNode::new(&format!("http://example.org/{}", p))?,
            NamedNode::new(&format!("http://example.org/{}", o))?,
        );
        model.add_triple(triple)?;
    }

    println!("   Added {} triples", triples.len());
    println!("   Training model...");

    let training_stats = model.train(Some(20)).await?;
    println!(
        "   Training complete: loss = {:.6}",
        training_stats.final_loss
    );

    // Get entity and relation embeddings
    let entities = model.get_entities();
    let relations = model.get_relations();

    let mut entity_embeddings = HashMap::new();
    for entity in &entities {
        if let Ok(embedding) = model.get_entity_embedding(entity) {
            entity_embeddings.insert(entity.clone(), embedding);
        }
    }

    let mut relation_embeddings = HashMap::new();
    for relation in &relations {
        if let Ok(embedding) = model.get_relation_embedding(relation) {
            relation_embeddings.insert(relation.clone(), embedding);
        }
    }

    println!("   Extracted {} entity embeddings", entity_embeddings.len());
    println!(
        "   Extracted {} relation embeddings",
        relation_embeddings.len()
    );

    // Step 2: Demonstrate Memory Backend
    println!("\n2. Testing Memory Backend...");
    let memory_config = StorageBackendConfig {
        backend_type: StorageBackendType::Memory,
        ..Default::default()
    };

    let mut memory_backend = StorageBackendManager::new(memory_config).await?;

    // Save embeddings to memory
    memory_backend
        .save_embeddings(&entity_embeddings, &relation_embeddings)
        .await?;
    println!("   ✓ Saved embeddings to memory");

    // Save metadata
    let metadata = EmbeddingMetadata {
        model_id: *model.model_id(),
        model_type: model.model_type().to_string(),
        model_config: model_config.clone(),
        model_stats: model.get_stats(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        version: "1.0.0".to_string(),
    };
    memory_backend.save_metadata(&metadata).await?;
    println!("   ✓ Saved metadata");

    // Load embeddings from memory
    let (loaded_entities, loaded_relations) = memory_backend.load_embeddings().await?;
    println!("   ✓ Loaded {} entity embeddings", loaded_entities.len());
    println!("   ✓ Loaded {} relation embeddings", loaded_relations.len());

    // Get memory backend statistics
    let memory_stats = memory_backend.get_stats().await?;
    println!("\n   Memory Backend Statistics:");
    println!("     Total Embeddings: {}", memory_stats.total_embeddings);
    println!("     Total Size: {} bytes", memory_stats.total_size_bytes);
    println!(
        "     Compression Ratio: {:.2}",
        memory_stats.compression_ratio
    );

    // Step 3: Demonstrate Disk Backend
    println!("\n3. Testing Disk Backend...");

    // Use temporary directory
    let temp_dir = std::env::temp_dir().join(format!("oxirs_embed_demo_{}", Uuid::new_v4()));
    std::fs::create_dir_all(&temp_dir)?;

    let disk_config = StorageBackendConfig {
        backend_type: StorageBackendType::Disk {
            path: temp_dir.clone(),
            use_mmap: true,
        },
        compression: true,
        versioning: true,
        max_versions: 5,
        enable_cache: true,
        cache_size_mb: 100,
        ..Default::default()
    };

    let mut disk_backend = StorageBackendManager::new(disk_config).await?;
    println!("   Storage path: {}", temp_dir.display());

    // Save embeddings to disk
    disk_backend
        .save_embeddings(&entity_embeddings, &relation_embeddings)
        .await?;
    println!("   ✓ Saved embeddings to disk");

    disk_backend.save_metadata(&metadata).await?;
    println!("   ✓ Saved metadata to disk");

    // Get disk backend statistics
    let disk_stats = disk_backend.get_stats().await?;
    println!("\n   Disk Backend Statistics:");
    println!("     Total Embeddings: {}", disk_stats.total_embeddings);
    println!(
        "     Uncompressed Size: {} bytes",
        disk_stats.total_size_bytes
    );
    println!(
        "     Compressed Size: {} bytes",
        disk_stats.compressed_size_bytes
    );
    println!(
        "     Compression Ratio: {:.2} ({:.1}% reduction)",
        disk_stats.compression_ratio,
        (1.0 - disk_stats.compression_ratio) * 100.0
    );
    println!("     Versions: {}", disk_stats.num_versions);

    // Step 4: Demonstrate Checkpointing
    println!("\n4. Testing Checkpointing...");

    // Create first checkpoint
    disk_backend.create_checkpoint("checkpoint_v1").await?;
    println!("   ✓ Created checkpoint: checkpoint_v1");

    // Modify embeddings
    let mut modified_entities = entity_embeddings.clone();
    if let Some(first_key) = modified_entities.keys().next().cloned() {
        if let Some(embedding) = modified_entities.get_mut(&first_key) {
            // Modify the first value
            if !embedding.values.is_empty() {
                embedding.values[0] += 1.0;
            }
        }
    }

    disk_backend
        .save_embeddings(&modified_entities, &relation_embeddings)
        .await?;
    println!("   ✓ Modified and saved embeddings");

    // Create second checkpoint
    disk_backend.create_checkpoint("checkpoint_v2").await?;
    println!("   ✓ Created checkpoint: checkpoint_v2");

    // Restore from first checkpoint
    disk_backend.restore_checkpoint("checkpoint_v1").await?;
    println!("   ✓ Restored checkpoint: checkpoint_v1");

    let (restored_entities, _) = disk_backend.load_embeddings().await?;
    println!("   ✓ Loaded restored embeddings");

    // Verify restoration
    if let (Some(original_key), Some(restored_key)) = (
        entity_embeddings.keys().next(),
        restored_entities.keys().next(),
    ) {
        if original_key == restored_key {
            let original = &entity_embeddings[original_key];
            let restored = &restored_entities[restored_key];
            let matches = original.values == restored.values;
            println!(
                "   ✓ Checkpoint restoration verified: {}",
                if matches { "Success" } else { "Mismatch" }
            );
        }
    }

    // Step 5: Compare backends
    println!("\n5. Backend Comparison:");
    println!("\n   Memory Backend:");
    println!("     ✓ Fastest access (in-memory)");
    println!("     ✗ Volatile (data lost on restart)");
    println!("     ✓ No disk I/O overhead");
    println!("     Use case: Temporary computations, caching");

    println!("\n   Disk Backend:");
    println!("     ✓ Persistent storage");
    println!("     ✓ Compression support");
    println!("     ✓ Checkpointing and versioning");
    println!("     ✓ Memory-mapped file support");
    println!("     Use case: Long-term storage, production deployments");

    // Step 6: Demonstrate use case scenarios
    println!("\n6. Use Case Scenarios:");

    println!("\n   Scenario 1: Development & Testing");
    println!("     → Use Memory Backend for fast iteration");
    println!("     → No disk cleanup needed");

    println!("\n   Scenario 2: Production Deployment");
    println!("     → Use Disk Backend with compression");
    println!("     → Enable checkpointing for recovery");
    println!("     → Use versioning for rollback capability");

    println!("\n   Scenario 3: Distributed Systems");
    println!("     → Use S3 Backend (planned) for shared storage");
    println!("     → Enable replication for high availability");

    println!("\n   Scenario 4: High Performance");
    println!("     → Use RocksDB Backend (planned) for fast key-value access");
    println!("     → Enable caching layer");

    // Step 7: Cleanup
    println!("\n7. Cleanup...");
    std::fs::remove_dir_all(&temp_dir)?;
    println!("   ✓ Removed temporary directory");

    println!("\n=== Demo Complete ===");
    println!("\nKey Takeaways:");
    println!("✓ Multiple storage backends for different use cases");
    println!("✓ Compression reduces storage requirements significantly");
    println!("✓ Checkpointing enables version control and rollback");
    println!("✓ Memory backend for fast temporary storage");
    println!("✓ Disk backend for persistent, production storage");
    println!("✓ Future: S3, Redis, PostgreSQL, RocksDB support");

    Ok(())
}
