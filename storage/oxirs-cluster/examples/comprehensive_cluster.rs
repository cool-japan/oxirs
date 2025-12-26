//! Comprehensive OxiRS Cluster Example
//!
//! This example demonstrates how to use all the major features of oxirs-cluster v0.1.0-rc.1:
//! - Distributed Tracing
//! - Alerting System
//! - Visualization Dashboard
//! - Zero-Downtime Migrations
//! - Disaster Recovery
//! - Core Clustering Features
//!
//! Run with: cargo run --example comprehensive_cluster --all-features

use oxirs_cluster::{
    alerting::{AlertCategory, AlertSeverity, AlertingConfig, AlertingManager},
    disaster_recovery::{BackupType, RecoveryConfig, RecoveryManager, RecoveryObjectives},
    distributed_tracing::{SamplingStrategy, TracingConfig, TracingManager},
    visualization_dashboard::{DashboardConfig, DashboardServer},
    zero_downtime_migration::{MigrationConfig, MigrationManager, MigrationStrategy},
};
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 OxiRS Cluster v0.1.0-rc.1 - Comprehensive Example");
    println!("=====================================================\n");

    // ==========================================
    // 1. Distributed Tracing Setup
    // ==========================================
    println!("📊 Setting up Distributed Tracing...");

    let tracing_config = TracingConfig::default()
        .with_service_name("oxirs-cluster-demo")
        .with_sampling_strategy(SamplingStrategy::TraceIdRatioBased(1.0))
        .with_enabled(true); // Enable for comprehensive tracing

    let tracing_manager = TracingManager::new(tracing_config).await?;
    tracing_manager.start().await?;

    println!("   ✅ Distributed tracing initialized");
    println!("   - Service: oxirs-cluster-demo");
    println!("   - Sampling: 100% (for demo)");
    println!("   - Export: Console output\n");

    // ==========================================
    // 2. Alerting System Setup
    // ==========================================
    println!("🔔 Setting up Alerting System...");

    let alerting_config = AlertingConfig::default().with_min_severity(AlertSeverity::Info);

    let mut alerting_manager = AlertingManager::new(alerting_config).await?;
    alerting_manager.start().await?;

    println!("   ✅ Alerting system initialized");
    println!("   - Minimum severity: Info");
    println!("   - Channels: Console (Email/Slack available)");
    println!("   - Throttling: Enabled\n");

    // Send a welcome alert
    alerting_manager
        .send_categorized_alert(
            AlertSeverity::Info,
            AlertCategory::Custom("System".to_string()),
            "System Initialized",
            "OxiRS cluster is starting up with all features enabled",
        )
        .await?;

    // ==========================================
    // 3. Disaster Recovery Setup
    // ==========================================
    println!("🛡️  Setting up Disaster Recovery...");

    let recovery_config = RecoveryConfig::default()
        .with_backup_retention_days(30)
        .with_auto_recovery(true)
        .with_objectives(RecoveryObjectives {
            rto_seconds: 300,    // 5 minutes RTO
            rpo_seconds: 60,     // 1 minute RPO
            mtdl_bytes: 1048576, // 1MB max data loss
            slo_percent: 99.99,  // Four nines
        });

    let mut recovery_manager = RecoveryManager::new(recovery_config).await?;
    recovery_manager.start().await?;

    println!("   ✅ Disaster recovery initialized");
    println!("   - RTO: 5 minutes");
    println!("   - RPO: 1 minute");
    println!("   - Retention: 30 days");
    println!("   - Auto-recovery: Enabled\n");

    // Create an initial backup
    println!("💾 Creating initial backup...");
    let backup = recovery_manager.create_backup(BackupType::Full).await?;
    println!("   ✅ Backup created: {}", backup.backup_id);
    println!("   - Type: Full");
    println!("   - Compressed: {}", backup.compressed);
    println!("   - Location: {}\n", backup.location);

    // Send backup alert
    alerting_manager
        .send_categorized_alert(
            AlertSeverity::Info,
            AlertCategory::Custom("Backup".to_string()),
            "Backup Created",
            &format!("Full backup completed: {}", backup.backup_id),
        )
        .await?;

    // ==========================================
    // 4. Migration Manager Setup
    // ==========================================
    println!("🔄 Setting up Zero-Downtime Migration...");

    let migration_config = MigrationConfig::default()
        .with_strategy(MigrationStrategy::Rolling { batch_percent: 25 })
        .with_batch_size(1000)
        .with_validation(true);

    let mut migration_manager = MigrationManager::new(migration_config).await?;

    println!("   ✅ Migration manager initialized");
    println!("   - Strategy: Rolling (25% batches)");
    println!("   - Batch size: 1000");
    println!("   - Validation: Enabled\n");

    // Start a demo migration
    println!("📦 Starting demo migration (v1.0 → v2.0)...");
    let migration_id = migration_manager.start_migration("v1.0", "v2.0").await?;
    println!("   ✅ Migration initiated: {}", migration_id);

    // Wait a bit for migration to progress
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Check migration status
    if let Some(status) = migration_manager.get_migration_status(&migration_id).await {
        println!("   - Status: {:?}", status.status);
        println!("   - Progress: {:.1}%", status.progress_percent);
        println!("   - Phase: {:?}\n", status.current_phase);
    }

    // Send migration alert
    alerting_manager
        .send_categorized_alert(
            AlertSeverity::Info,
            AlertCategory::Custom("Migration".to_string()),
            "Migration In Progress",
            &format!("Schema migration {} is running", migration_id),
        )
        .await?;

    // ==========================================
    // 5. Visualization Dashboard Setup
    // ==========================================
    println!("📈 Setting up Visualization Dashboard...");

    let dashboard_config = DashboardConfig::default()
        .with_bind_address("127.0.0.1:8080")
        .with_refresh_interval(1000);

    let dashboard_server = DashboardServer::new(dashboard_config).await?;
    dashboard_server.start().await?;

    println!("   ✅ Dashboard server started");
    println!("   - Address: http://127.0.0.1:8080");
    println!("   - Refresh: 1 second");
    println!("   - Features: Real-time metrics, node management, alerts\n");

    // ==========================================
    // 6. Demonstrate Feature Integration
    // ==========================================
    println!("🔗 Demonstrating Feature Integration...\n");

    // Scenario 1: Node health alert
    println!("📡 Scenario 1: Node Health Monitoring");
    alerting_manager
        .send_categorized_alert(
            AlertSeverity::Warning,
            AlertCategory::NodeHealth,
            "High CPU Usage",
            "Node 2 CPU usage: 85%",
        )
        .await?;
    println!("   ✅ Health alert sent\n");

    // Scenario 2: Performance degradation
    println!("⚡ Scenario 2: Performance Monitoring");
    alerting_manager
        .send_categorized_alert(
            AlertSeverity::Warning,
            AlertCategory::Performance,
            "Increased Latency",
            "Average query latency: 250ms (threshold: 100ms)",
        )
        .await?;
    println!("   ✅ Performance alert sent\n");

    // Scenario 3: Create incremental backup
    println!("💾 Scenario 3: Incremental Backup");
    let incremental_backup = recovery_manager
        .create_backup(BackupType::Incremental)
        .await?;
    println!("   ✅ Incremental backup: {}", incremental_backup.backup_id);

    alerting_manager
        .send_categorized_alert(
            AlertSeverity::Info,
            AlertCategory::Custom("Backup".to_string()),
            "Incremental Backup",
            &format!("Backup {} completed", incremental_backup.backup_id),
        )
        .await?;
    println!("   ✅ Backup alert sent\n");

    // ==========================================
    // 7. System Status Summary
    // ==========================================
    println!("📊 System Status Summary");
    println!("========================");

    // Get alerting statistics
    let alert_stats = alerting_manager.get_statistics().await;
    println!("\n🔔 Alerting:");
    println!("   - Total alerts: {}", alert_stats.total_alerts);
    println!("   - Throttled: {}", alert_stats.is_throttled);
    println!("   - Current window: {}", alert_stats.current_window_count);

    // Get backup history
    let backup_history = recovery_manager.get_backup_history().await;
    println!("\n💾 Disaster Recovery:");
    println!("   - Total backups: {}", backup_history.len());
    println!(
        "   - Latest: {}",
        backup_history
            .last()
            .map(|b| b.backup_id.as_str())
            .unwrap_or("None")
    );

    // Get migration history
    let migration_history = migration_manager.get_migration_history().await;
    println!("\n🔄 Migrations:");
    println!("   - Completed: {}", migration_history.len());

    println!("\n📈 Dashboard:");
    println!("   - URL: http://127.0.0.1:8080");
    println!("   - Status: Running");

    // ==========================================
    // 8. Keep running for demonstration
    // ==========================================
    println!("\n✅ All systems operational!");
    println!("\n🌐 Dashboard is running at http://127.0.0.1:8080");
    println!("   Press Ctrl+C to stop...\n");

    // Send final alert
    alerting_manager
        .send_categorized_alert(
            AlertSeverity::Info,
            AlertCategory::Custom("System".to_string()),
            "All Systems Operational",
            "Cluster is fully initialized and ready",
        )
        .await?;

    // Keep the example running
    tokio::time::sleep(Duration::from_secs(3600)).await;

    // ==========================================
    // 9. Cleanup
    // ==========================================
    println!("\n🛑 Shutting down gracefully...");

    dashboard_server.stop().await?;
    alerting_manager.stop().await?;
    recovery_manager.stop().await?;
    tracing_manager.stop().await?;

    println!("   ✅ All systems shut down cleanly");
    println!("\n👋 Goodbye!\n");

    Ok(())
}
