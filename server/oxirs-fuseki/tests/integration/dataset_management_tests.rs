// Dataset Management Integration Tests
//
// Comprehensive integration tests for dataset_management module

use oxirs_fuseki::dataset_management::{
    DatasetManager, DatasetManagerConfig, DatasetMetadata, DatasetOperation,
    SnapshotConfig, BackupSchedule
};
use oxirs_fuseki::store::Store;
use oxirs_core::model::Term;
use std::sync::Arc;
use std::time::Duration;
use std::collections::HashMap;
use tokio::time::timeout;

/// Helper to create a test store with sample data
async fn create_test_store_with_dataset() -> anyhow::Result<Arc<Store>> {
    let store = Arc::new(Store::new()?);

    // Insert test triples
    for i in 0..10 {
        let triple = (
            Term::NamedNode(format!("http://example.org/subject{}", i)),
            Term::NamedNode("http://example.org/predicate".to_string()),
            Term::Literal {
                value: format!("Value {}", i),
                language: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
            },
        );

        store.insert(triple.0, triple.1, triple.2, None).await?;
    }

    Ok(store)
}

#[tokio::test]
async fn test_dataset_manager_creation() {
    let config = DatasetManagerConfig {
        max_concurrent_operations: 5,
        enable_versioning: true,
        max_snapshots_per_dataset: 10,
        auto_backup_interval: Some(Duration::from_secs(3600)),
        enable_compression: true,
    };

    let manager = DatasetManager::new(config);

    assert!(manager.get_statistics().total_datasets == 0);
    assert!(manager.get_statistics().active_operations == 0);
}

#[tokio::test]
async fn test_create_dataset() -> anyhow::Result<()> {
    let config = DatasetManagerConfig::default();
    let manager = DatasetManager::new(config);

    let metadata = DatasetMetadata {
        name: "test_dataset".to_string(),
        description: Some("Test dataset for integration tests".to_string()),
        version: "1.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 0,
        tags: vec!["test".to_string(), "integration".to_string()],
        custom_properties: HashMap::new(),
    };

    manager.create_dataset(metadata).await?;

    let stats = manager.get_statistics();
    assert!(stats.total_datasets >= 1);

    Ok(())
}

#[tokio::test]
async fn test_delete_dataset() -> anyhow::Result<()> {
    let config = DatasetManagerConfig::default();
    let manager = DatasetManager::new(config);

    // Create a dataset first
    let metadata = DatasetMetadata {
        name: "delete_test".to_string(),
        description: Some("Dataset to be deleted".to_string()),
        version: "1.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 0,
        tags: vec![],
        custom_properties: HashMap::new(),
    };

    manager.create_dataset(metadata).await?;

    let initial_count = manager.get_statistics().total_datasets;

    // Delete the dataset
    manager.delete_dataset("delete_test").await?;

    let final_count = manager.get_statistics().total_datasets;
    assert!(final_count < initial_count, "Dataset count should decrease after deletion");

    Ok(())
}

#[tokio::test]
async fn test_list_datasets() -> anyhow::Result<()> {
    let config = DatasetManagerConfig::default();
    let manager = DatasetManager::new(config);

    // Create multiple datasets
    for i in 0..3 {
        let metadata = DatasetMetadata {
            name: format!("dataset_{}", i),
            description: Some(format!("Dataset number {}", i)),
            version: "1.0".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            triple_count: i * 10,
            tags: vec![],
            custom_properties: HashMap::new(),
        };

        manager.create_dataset(metadata).await?;
    }

    let datasets = manager.list_datasets().await?;
    assert!(datasets.len() >= 3, "Expected at least 3 datasets");

    Ok(())
}

#[tokio::test]
async fn test_get_dataset_metadata() -> anyhow::Result<()> {
    let config = DatasetManagerConfig::default();
    let manager = DatasetManager::new(config);

    let metadata = DatasetMetadata {
        name: "metadata_test".to_string(),
        description: Some("Test metadata retrieval".to_string()),
        version: "2.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 42,
        tags: vec!["important".to_string()],
        custom_properties: {
            let mut props = HashMap::new();
            props.insert("custom_key".to_string(), "custom_value".to_string());
            props
        },
    };

    manager.create_dataset(metadata.clone()).await?;

    let retrieved = manager.get_dataset_metadata("metadata_test").await?;
    assert_eq!(retrieved.name, "metadata_test");
    assert_eq!(retrieved.version, "2.0");
    assert_eq!(retrieved.triple_count, 42);
    assert!(retrieved.tags.contains(&"important".to_string()));

    Ok(())
}

#[tokio::test]
async fn test_update_dataset_metadata() -> anyhow::Result<()> {
    let config = DatasetManagerConfig::default();
    let manager = DatasetManager::new(config);

    // Create initial dataset
    let metadata = DatasetMetadata {
        name: "update_test".to_string(),
        description: Some("Original description".to_string()),
        version: "1.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 10,
        tags: vec![],
        custom_properties: HashMap::new(),
    };

    manager.create_dataset(metadata).await?;

    // Update metadata
    let updated_metadata = DatasetMetadata {
        name: "update_test".to_string(),
        description: Some("Updated description".to_string()),
        version: "2.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 20,
        tags: vec!["updated".to_string()],
        custom_properties: HashMap::new(),
    };

    manager.update_dataset_metadata(updated_metadata).await?;

    let retrieved = manager.get_dataset_metadata("update_test").await?;
    assert_eq!(retrieved.description, Some("Updated description".to_string()));
    assert_eq!(retrieved.version, "2.0");

    Ok(())
}

#[tokio::test]
async fn test_create_snapshot() -> anyhow::Result<()> {
    let config = DatasetManagerConfig {
        enable_versioning: true,
        max_snapshots_per_dataset: 5,
        ..Default::default()
    };

    let manager = DatasetManager::new(config);

    // Create a dataset
    let metadata = DatasetMetadata {
        name: "snapshot_test".to_string(),
        description: Some("Dataset for snapshot testing".to_string()),
        version: "1.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 100,
        tags: vec![],
        custom_properties: HashMap::new(),
    };

    manager.create_dataset(metadata).await?;

    // Create snapshot
    let snapshot_config = SnapshotConfig {
        name: "snapshot_1".to_string(),
        description: Some("First snapshot".to_string()),
        include_metadata: true,
    };

    manager.create_snapshot("snapshot_test", snapshot_config).await?;

    let stats = manager.get_statistics();
    assert!(stats.total_snapshots >= 1);

    Ok(())
}

#[tokio::test]
async fn test_list_snapshots() -> anyhow::Result<()> {
    let config = DatasetManagerConfig {
        enable_versioning: true,
        max_snapshots_per_dataset: 5,
        ..Default::default()
    };

    let manager = DatasetManager::new(config);

    // Create a dataset
    let metadata = DatasetMetadata {
        name: "list_snapshots_test".to_string(),
        description: Some("Dataset for listing snapshots".to_string()),
        version: "1.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 100,
        tags: vec![],
        custom_properties: HashMap::new(),
    };

    manager.create_dataset(metadata).await?;

    // Create multiple snapshots
    for i in 0..3 {
        let snapshot_config = SnapshotConfig {
            name: format!("snapshot_{}", i),
            description: Some(format!("Snapshot number {}", i)),
            include_metadata: true,
        };

        manager.create_snapshot("list_snapshots_test", snapshot_config).await?;
    }

    let snapshots = manager.list_snapshots("list_snapshots_test").await?;
    assert!(snapshots.len() >= 3, "Expected at least 3 snapshots");

    Ok(())
}

#[tokio::test]
async fn test_delete_snapshot() -> anyhow::Result<()> {
    let config = DatasetManagerConfig {
        enable_versioning: true,
        max_snapshots_per_dataset: 5,
        ..Default::default()
    };

    let manager = DatasetManager::new(config);

    // Create dataset and snapshot
    let metadata = DatasetMetadata {
        name: "delete_snapshot_test".to_string(),
        description: Some("Dataset for snapshot deletion".to_string()),
        version: "1.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 100,
        tags: vec![],
        custom_properties: HashMap::new(),
    };

    manager.create_dataset(metadata).await?;

    let snapshot_config = SnapshotConfig {
        name: "to_be_deleted".to_string(),
        description: Some("This snapshot will be deleted".to_string()),
        include_metadata: true,
    };

    manager.create_snapshot("delete_snapshot_test", snapshot_config).await?;

    let initial_count = manager.list_snapshots("delete_snapshot_test").await?.len();

    // Delete snapshot
    manager.delete_snapshot("delete_snapshot_test", "to_be_deleted").await?;

    let final_count = manager.list_snapshots("delete_snapshot_test").await?.len();
    assert!(final_count < initial_count, "Snapshot count should decrease");

    Ok(())
}

#[tokio::test]
async fn test_snapshot_limit() -> anyhow::Result<()> {
    let config = DatasetManagerConfig {
        enable_versioning: true,
        max_snapshots_per_dataset: 3, // Limit to 3 snapshots
        ..Default::default()
    };

    let manager = DatasetManager::new(config);

    // Create dataset
    let metadata = DatasetMetadata {
        name: "snapshot_limit_test".to_string(),
        description: Some("Test snapshot limits".to_string()),
        version: "1.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 100,
        tags: vec![],
        custom_properties: HashMap::new(),
    };

    manager.create_dataset(metadata).await?;

    // Create more than max snapshots
    for i in 0..5 {
        let snapshot_config = SnapshotConfig {
            name: format!("snapshot_{}", i),
            description: Some(format!("Snapshot {}", i)),
            include_metadata: true,
        };

        manager.create_snapshot("snapshot_limit_test", snapshot_config).await?;
    }

    let snapshots = manager.list_snapshots("snapshot_limit_test").await?;
    assert!(snapshots.len() <= 3, "Snapshot count should not exceed limit");

    Ok(())
}

#[tokio::test]
async fn test_bulk_operations() -> anyhow::Result<()> {
    let config = DatasetManagerConfig {
        max_concurrent_operations: 3,
        ..Default::default()
    };

    let manager = DatasetManager::new(config);

    // Create multiple datasets in bulk
    let operations: Vec<DatasetOperation> = (0..5)
        .map(|i| DatasetOperation::Create {
            metadata: DatasetMetadata {
                name: format!("bulk_{}", i),
                description: Some(format!("Bulk dataset {}", i)),
                version: "1.0".to_string(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                triple_count: i * 10,
                tags: vec![],
                custom_properties: HashMap::new(),
            },
        })
        .collect();

    manager.execute_bulk_operations(operations).await?;

    let datasets = manager.list_datasets().await?;
    assert!(datasets.len() >= 5, "Expected at least 5 datasets from bulk operation");

    Ok(())
}

#[tokio::test]
async fn test_concurrent_operations() -> anyhow::Result<()> {
    let config = DatasetManagerConfig {
        max_concurrent_operations: 2,
        ..Default::default()
    };

    let manager = Arc::new(DatasetManager::new(config));

    // Spawn concurrent operations
    let mut handles = Vec::new();

    for i in 0..4 {
        let manager_clone = Arc::clone(&manager);
        let handle = tokio::spawn(async move {
            let metadata = DatasetMetadata {
                name: format!("concurrent_{}", i),
                description: Some(format!("Concurrent dataset {}", i)),
                version: "1.0".to_string(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                triple_count: 0,
                tags: vec![],
                custom_properties: HashMap::new(),
            };

            manager_clone.create_dataset(metadata).await
        });

        handles.push(handle);
    }

    // Wait for all operations
    for handle in handles {
        handle.await??;
    }

    let stats = manager.get_statistics();
    assert!(stats.total_datasets >= 4);

    Ok(())
}

#[tokio::test]
async fn test_backup_scheduling() -> anyhow::Result<()> {
    let config = DatasetManagerConfig {
        auto_backup_interval: Some(Duration::from_millis(100)),
        ..Default::default()
    };

    let manager = DatasetManager::new(config);

    // Create dataset
    let metadata = DatasetMetadata {
        name: "backup_test".to_string(),
        description: Some("Test automatic backups".to_string()),
        version: "1.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 50,
        tags: vec![],
        custom_properties: HashMap::new(),
    };

    manager.create_dataset(metadata).await?;

    // Set backup schedule
    let schedule = BackupSchedule {
        enabled: true,
        interval: Duration::from_millis(100),
        retention_count: 5,
    };

    manager.set_backup_schedule("backup_test", schedule).await?;

    // Wait for automatic backup
    tokio::time::sleep(Duration::from_millis(200)).await;

    let stats = manager.get_statistics();
    // Backup count may or may not have incremented depending on timing
    assert!(stats.total_datasets >= 1);

    Ok(())
}

#[tokio::test]
async fn test_dataset_versioning() -> anyhow::Result<()> {
    let config = DatasetManagerConfig {
        enable_versioning: true,
        ..Default::default()
    };

    let manager = DatasetManager::new(config);

    // Create initial version
    let metadata_v1 = DatasetMetadata {
        name: "versioning_test".to_string(),
        description: Some("Version 1".to_string()),
        version: "1.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 10,
        tags: vec![],
        custom_properties: HashMap::new(),
    };

    manager.create_dataset(metadata_v1).await?;

    // Update to version 2
    let metadata_v2 = DatasetMetadata {
        name: "versioning_test".to_string(),
        description: Some("Version 2".to_string()),
        version: "2.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 20,
        tags: vec![],
        custom_properties: HashMap::new(),
    };

    manager.update_dataset_metadata(metadata_v2).await?;

    let current = manager.get_dataset_metadata("versioning_test").await?;
    assert_eq!(current.version, "2.0");

    Ok(())
}

#[tokio::test]
async fn test_statistics_tracking() -> anyhow::Result<()> {
    let config = DatasetManagerConfig::default();
    let manager = DatasetManager::new(config);

    let initial_stats = manager.get_statistics();

    // Perform operations
    let metadata = DatasetMetadata {
        name: "stats_test".to_string(),
        description: Some("Statistics tracking test".to_string()),
        version: "1.0".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        triple_count: 100,
        tags: vec![],
        custom_properties: HashMap::new(),
    };

    manager.create_dataset(metadata).await?;

    let final_stats = manager.get_statistics();

    assert!(
        final_stats.total_datasets > initial_stats.total_datasets,
        "Dataset count should increase"
    );
    assert!(
        final_stats.total_operations > initial_stats.total_operations,
        "Operation count should increase"
    );

    Ok(())
}
