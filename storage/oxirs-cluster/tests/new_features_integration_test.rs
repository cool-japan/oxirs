//! Integration tests for new v0.1.0 features
//!
//! This file tests the integration of:
//! - Distributed Tracing
//! - Alerting System
//! - Visualization Dashboard
//! - Zero-Downtime Migrations
//! - Disaster Recovery

use oxirs_cluster::{
    alerting::{AlertCategory, AlertSeverity, AlertingConfig, AlertingManager},
    disaster_recovery::{BackupType, DisasterScenario, RecoveryConfig, RecoveryManager},
    distributed_tracing::{SamplingStrategy, TracingConfig, TracingManager},
    visualization_dashboard::{DashboardConfig, DashboardServer},
    zero_downtime_migration::{
        MigrationConfig, MigrationManager, MigrationMetadata, MigrationStrategy,
    },
};

#[tokio::test]
async fn test_distributed_tracing_lifecycle() {
    // Create tracing config
    let config = TracingConfig::default()
        .with_service_name("test-cluster")
        .with_enabled(false) // Disable actual export for test
        .with_sampling_strategy(SamplingStrategy::AlwaysOn);

    // Create manager
    let manager = TracingManager::new(config).await;
    assert!(manager.is_ok());
    let manager = manager.unwrap();

    // Check initial state
    assert!(!manager.is_initialized().await);

    // Start manager
    let result = manager.start().await;
    assert!(result.is_ok());

    // Note: With tracing disabled, it won't actually initialize
    // This is expected behavior

    // Stop manager
    let result = manager.stop().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_alerting_system_basic_flow() {
    // Create alerting config without real channels (for testing)
    let config = AlertingConfig::default().with_min_severity(AlertSeverity::Info);

    // Create manager
    let manager = AlertingManager::new(config).await;
    assert!(manager.is_ok());
    let mut manager = manager.unwrap();

    // Start manager
    let result = manager.start().await;
    assert!(result.is_ok());

    // Send test alert (will be throttled/skipped since no channels configured)
    let result = manager
        .send_alert(
            AlertSeverity::Warning,
            "Test Alert",
            "This is a test message",
        )
        .await;
    // Should succeed even without channels
    assert!(result.is_ok());

    // Get statistics
    let _stats = manager.get_statistics().await;
    // Note: Alerts are tracked even if no channels are configured
    // This allows monitoring of alert generation rate

    // Stop manager
    let result = manager.stop().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_alerting_categorized_alerts() {
    let config = AlertingConfig::default();
    let manager = AlertingManager::new(config).await.unwrap();

    // Test different alert categories
    let categories = vec![
        AlertCategory::NodeHealth,
        AlertCategory::Consensus,
        AlertCategory::Replication,
        AlertCategory::Performance,
        AlertCategory::Security,
    ];

    for category in categories {
        let result = manager
            .send_categorized_alert(
                AlertSeverity::Info,
                category.clone(),
                "Category Test",
                "Testing alert category",
            )
            .await;
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_visualization_dashboard_creation() {
    // Create dashboard config
    let config = DashboardConfig::default()
        .with_bind_address("127.0.0.1:0") // Use port 0 for testing
        .with_refresh_interval(500);

    // Create server
    let server = DashboardServer::new(config).await;
    assert!(server.is_ok());
}

#[tokio::test]
async fn test_zero_downtime_migration_workflow() {
    // Create migration config
    let config = MigrationConfig::default()
        .with_strategy(MigrationStrategy::Rolling { batch_percent: 25 })
        .with_batch_size(100)
        .with_validation(true);

    // Create manager
    let manager = MigrationManager::new(config).await;
    assert!(manager.is_ok());
    let mut manager = manager.unwrap();

    // Start a migration
    let result = manager.start_migration("v1.0", "v2.0").await;
    assert!(result.is_ok());

    let migration_id = result.unwrap();
    assert!(!migration_id.is_empty());

    // Give it a moment to process
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Check migration status
    let status = manager.get_migration_status(&migration_id).await;
    assert!(status.is_some());

    let metadata = status.unwrap();
    assert_eq!(metadata.from_version, "v1.0");
    assert_eq!(metadata.to_version, "v2.0");
}

#[tokio::test]
async fn test_migration_metadata() {
    let metadata = MigrationMetadata::new("v1", "v2", "Test migration");

    assert_eq!(metadata.from_version, "v1");
    assert_eq!(metadata.to_version, "v2");
    assert_eq!(metadata.progress_percent, 0.0);
    assert_eq!(metadata.items_migrated, 0);
}

#[tokio::test]
async fn test_disaster_recovery_manager_creation() {
    // Create recovery config
    let config = RecoveryConfig::default()
        .with_backup_retention_days(7)
        .with_auto_recovery(true);

    // Create manager
    let manager = RecoveryManager::new(config).await;
    assert!(manager.is_ok());
    let mut manager = manager.unwrap();

    // Start manager
    let result = manager.start().await;
    assert!(result.is_ok());

    // Stop manager
    let result = manager.stop().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_disaster_recovery_backup_creation() {
    let config = RecoveryConfig::default();
    let manager = RecoveryManager::new(config).await.unwrap();

    // Create a full backup
    let result = manager.create_backup(BackupType::Full).await;
    assert!(result.is_ok());

    let backup = result.unwrap();
    assert!(!backup.backup_id.is_empty());
    assert_eq!(backup.backup_type, BackupType::Full);

    // Get backup history
    let history = manager.get_backup_history().await;
    assert_eq!(history.len(), 1);
}

#[tokio::test]
async fn test_disaster_recovery_scenarios() {
    let config = RecoveryConfig::default();
    let manager = RecoveryManager::new(config).await.unwrap();

    // Test different disaster scenarios
    let scenarios = vec![
        DisasterScenario::NodeFailure,
        DisasterScenario::DataCorruption,
    ];

    for scenario in scenarios {
        let result = manager.initiate_recovery(scenario).await;
        assert!(result.is_ok());

        let operation_id = result.unwrap();
        assert!(!operation_id.is_empty());

        // Give recovery time to start
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Check recovery status
        let status = manager.get_recovery_status(&operation_id).await;
        assert!(status.is_some());
    }
}

#[tokio::test]
async fn test_alert_severity_ordering() {
    assert!(AlertSeverity::Info < AlertSeverity::Warning);
    assert!(AlertSeverity::Warning < AlertSeverity::Error);
    assert!(AlertSeverity::Error < AlertSeverity::Critical);
}

#[tokio::test]
async fn test_sampling_strategies() {
    let strategies = vec![
        SamplingStrategy::AlwaysOn,
        SamplingStrategy::AlwaysOff,
        SamplingStrategy::TraceIdRatioBased(0.5),
    ];

    for strategy in strategies {
        let config = TracingConfig::default().with_sampling_strategy(strategy.clone());

        assert_eq!(config.sampling_strategy, strategy);
    }
}

#[tokio::test]
async fn test_migration_strategies() {
    let strategies = vec![
        MigrationStrategy::AllAtOnce,
        MigrationStrategy::BlueGreen,
        MigrationStrategy::Rolling { batch_percent: 20 },
        MigrationStrategy::Canary { canary_percent: 10 },
    ];

    for strategy in strategies {
        let config = MigrationConfig::default().with_strategy(strategy.clone());

        assert_eq!(config.strategy, strategy);
    }
}

#[tokio::test]
async fn test_full_feature_integration() {
    // This test demonstrates how all features work together

    // 1. Set up distributed tracing
    let tracing_config = TracingConfig::default()
        .with_service_name("integration-test")
        .with_enabled(false); // Disabled for test

    let tracing_manager = TracingManager::new(tracing_config).await.unwrap();
    tracing_manager.start().await.unwrap();

    // 2. Set up alerting
    let alerting_config = AlertingConfig::default();
    let mut alerting_manager = AlertingManager::new(alerting_config).await.unwrap();
    alerting_manager.start().await.unwrap();

    // 3. Set up disaster recovery
    let recovery_config = RecoveryConfig::default();
    let mut recovery_manager = RecoveryManager::new(recovery_config).await.unwrap();
    recovery_manager.start().await.unwrap();

    // 4. Create a backup
    let backup = recovery_manager
        .create_backup(BackupType::Full)
        .await
        .unwrap();
    assert!(!backup.backup_id.is_empty());

    // 5. Send an alert
    alerting_manager
        .send_alert(
            AlertSeverity::Info,
            "Integration Test",
            "All systems operational",
        )
        .await
        .unwrap();

    // 6. Clean up
    alerting_manager.stop().await.unwrap();
    recovery_manager.stop().await.unwrap();
    tracing_manager.stop().await.unwrap();
}
