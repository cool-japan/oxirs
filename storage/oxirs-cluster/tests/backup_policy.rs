//! Integration tests for the backup policy DSL.
//!
//! These tests exercise the public API exposed by `oxirs_cluster::backup`.

use oxirs_cluster::backup::{
    destination::DestinationConfig,
    gfs::{BackupRecord, GfsRotation},
    policy::{BackupPolicy, CronSchedule, EncryptionConfig},
    retention::RetentionTier,
    BackupExecutor,
};
use std::env;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn unique_dir(tag: &str) -> std::path::PathBuf {
    env::temp_dir().join(format!(
        "oxirs_bp_test_{}_{}",
        tag,
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos()
    ))
}

fn daily_policy(dir: &std::path::Path) -> BackupPolicy {
    BackupPolicy {
        name: "daily-test".into(),
        schedule: CronSchedule::daily(),
        retention: RetentionTier::standard(),
        gfs: Some(GfsRotation::default()),
        encryption: EncryptionConfig::none(),
        destination: DestinationConfig::Filesystem {
            path: dir.to_owned(),
        },
    }
}

fn days_ago(n: u64, now: SystemTime) -> SystemTime {
    now - Duration::from_secs(n * 86_400)
}

// ---------------------------------------------------------------------------
// BackupPolicy
// ---------------------------------------------------------------------------

#[test]
fn backup_policy_builds_correctly() {
    let dir = unique_dir("build");
    let policy = daily_policy(&dir);
    assert_eq!(policy.name, "daily-test");
    assert_eq!(policy.schedule.expression, "0 2 * * *");
    assert!(!policy.encryption.enabled);
    assert!(policy.gfs.is_some());
}

#[test]
fn backup_policy_serialises_and_deserialises() {
    let dir = unique_dir("serde");
    let policy = daily_policy(&dir);
    let json = serde_json::to_string(&policy).unwrap();
    let back: BackupPolicy = serde_json::from_str(&json).unwrap();
    assert_eq!(back.name, policy.name);
    assert_eq!(back.schedule.expression, policy.schedule.expression);
    assert_eq!(back.retention.hot_days, policy.retention.hot_days);
}

// ---------------------------------------------------------------------------
// CronSchedule
// ---------------------------------------------------------------------------

#[test]
fn cron_schedules_have_correct_expressions() {
    assert_eq!(CronSchedule::daily().expression, "0 2 * * *");
    assert_eq!(CronSchedule::weekly().expression, "0 2 * * 0");
    assert_eq!(CronSchedule::monthly().expression, "0 2 1 * *");
}

// ---------------------------------------------------------------------------
// EncryptionConfig
// ---------------------------------------------------------------------------

#[test]
fn encryption_config_none() {
    let cfg = EncryptionConfig::none();
    assert!(!cfg.enabled);
    assert!(cfg.key_id.is_none());
}

#[test]
fn encryption_config_with_key_serialises() {
    let cfg = EncryptionConfig::with_key("key-123");
    let json = serde_json::to_string(&cfg).unwrap();
    let back: EncryptionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.key_id.as_deref(), Some("key-123"));
    assert!(back.enabled);
}

// ---------------------------------------------------------------------------
// RetentionTier
// ---------------------------------------------------------------------------

#[test]
fn retention_tier_hot_keeps_recent() {
    let tier = RetentionTier::standard(); // hot_days = 7
    let now = SystemTime::now();
    assert!(tier.should_retain(days_ago(3, now), now, false, false));
    assert!(tier.should_retain(days_ago(7, now), now, false, false));
}

#[test]
fn retention_tier_prunes_old_daily() {
    let tier = RetentionTier::standard();
    let now = SystemTime::now();
    assert!(!tier.should_retain(days_ago(14, now), now, false, false));
}

#[test]
fn retention_tier_warm_keeps_weekly() {
    let tier = RetentionTier::standard(); // warm_weeks = 4 → 28 days
    let now = SystemTime::now();
    assert!(tier.should_retain(days_ago(21, now), now, true, false));
}

#[test]
fn retention_tier_cold_keeps_monthly() {
    let tier = RetentionTier::standard(); // cold_months = 12 → 360 days
    let now = SystemTime::now();
    assert!(tier.should_retain(days_ago(300, now), now, false, true));
}

#[test]
fn retention_tier_standard_defaults() {
    let t = RetentionTier::standard();
    assert_eq!(t.hot_days, 7);
    assert_eq!(t.warm_weeks, 4);
    assert_eq!(t.cold_months, 12);
}

#[test]
fn retention_tier_minimal_defaults() {
    let t = RetentionTier::minimal();
    assert_eq!(t.hot_days, 1);
    assert_eq!(t.warm_weeks, 1);
    assert_eq!(t.cold_months, 1);
}

// ---------------------------------------------------------------------------
// GfsRotation
// ---------------------------------------------------------------------------

#[test]
fn gfs_rotation_prunes_old_plain_backup() {
    let gfs = GfsRotation {
        daily_count: 7,
        weekly_count: 2,
        monthly_count: 2,
    };
    let now = SystemTime::now();
    let records = vec![
        BackupRecord {
            id: 1,
            created_at: now - Duration::from_secs(100 * 86_400),
            size_bytes: 100,
            is_weekly: false,
            is_monthly: false,
        },
        BackupRecord {
            id: 2,
            created_at: now - Duration::from_secs(86_400),
            size_bytes: 100,
            is_weekly: false,
            is_monthly: false,
        },
    ];
    let pruned = gfs.prune_candidates(&records, now);
    assert!(pruned.contains(&1), "100-day-old backup should be pruned");
    assert!(!pruned.contains(&2), "1-day-old backup should be kept");
}

#[test]
fn gfs_rotation_100_day_simulation() {
    let gfs = GfsRotation {
        daily_count: 7,
        weekly_count: 4,
        monthly_count: 3,
    };
    let now = SystemTime::now();

    let records: Vec<BackupRecord> = (0u64..100)
        .map(|day| BackupRecord {
            id: day,
            created_at: now - Duration::from_secs((100 - day) * 86_400),
            size_bytes: 1024,
            is_weekly: day % 7 == 0,
            is_monthly: day % 30 == 0,
        })
        .collect();

    let pruned = gfs.prune_candidates(&records, now);
    let kept_count = 100 - pruned.len();

    // At minimum the 7 most recent daily backups must be retained
    assert!(
        kept_count >= 7,
        "Expected ≥7 retained backups; got {kept_count}"
    );
}

#[test]
fn gfs_rotation_weekly_in_window_kept() {
    let gfs = GfsRotation {
        daily_count: 7,
        weekly_count: 4,
        monthly_count: 3,
    };
    let now = SystemTime::now();
    let records = vec![BackupRecord {
        id: 1,
        created_at: now - Duration::from_secs(20 * 86_400), // 20 days ≈ 3 weeks < 4-week window
        size_bytes: 512,
        is_weekly: true,
        is_monthly: false,
    }];
    let pruned = gfs.prune_candidates(&records, now);
    assert!(pruned.is_empty(), "20-day-old weekly backup should be kept");
}

#[test]
fn gfs_rotation_monthly_in_window_kept() {
    let gfs = GfsRotation {
        daily_count: 7,
        weekly_count: 4,
        monthly_count: 3,
    };
    let now = SystemTime::now();
    let records = vec![BackupRecord {
        id: 1,
        created_at: now - Duration::from_secs(80 * 86_400), // ~2.7 months < 3-month window
        size_bytes: 512,
        is_weekly: false,
        is_monthly: true,
    }];
    let pruned = gfs.prune_candidates(&records, now);
    assert!(
        pruned.is_empty(),
        "80-day-old monthly backup should be kept"
    );
}

// ---------------------------------------------------------------------------
// BackupExecutor
// ---------------------------------------------------------------------------

#[test]
fn backup_executor_writes_to_filesystem() {
    let dir = unique_dir("exec");
    let executor = BackupExecutor::new();
    let policy = daily_policy(&dir);
    let data = b"test backup data 12345";

    let size = executor
        .execute_backup(&policy, data, &dir)
        .expect("backup should succeed");

    assert_eq!(size, data.len() as u64);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn backup_executor_audit_log_has_started_and_completed() {
    let dir = unique_dir("audit");
    let executor = BackupExecutor::new();
    let policy = daily_policy(&dir);

    executor.execute_backup(&policy, b"payload", &dir).ok();

    let entries = executor.audit_entries();
    assert!(
        entries.len() >= 2,
        "expected ≥2 audit entries; got {}",
        entries.len()
    );
    assert!(
        entries.iter().any(|e| e.contains("Started")),
        "missing Started entry"
    );
    assert!(
        entries.iter().any(|e| e.contains("Completed")),
        "missing Completed entry"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn backup_executor_record_prune_appears_in_audit() {
    let executor = BackupExecutor::new();
    executor.record_prune("daily-test", 99);
    let entries = executor.audit_entries();
    assert!(
        entries.iter().any(|e| e.contains("Pruned")),
        "missing Pruned entry"
    );
}

#[test]
fn backup_executor_multiple_backups_accumulate_audit() {
    let dir = unique_dir("multi");
    let executor = BackupExecutor::new();
    let policy = daily_policy(&dir);

    executor.execute_backup(&policy, b"data1", &dir).ok();
    executor.execute_backup(&policy, b"data2", &dir).ok();

    // Each backup produces at least 2 entries (Started + Completed)
    assert!(
        executor.audit_len() >= 4,
        "expected ≥4 audit entries for 2 backups; got {}",
        executor.audit_len()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// DestinationConfig
// ---------------------------------------------------------------------------

#[test]
fn destination_config_default_is_filesystem() {
    use oxirs_cluster::backup::destination::DestinationConfig;
    let d = DestinationConfig::default();
    assert!(d.local_path().is_some());
}

#[test]
fn destination_config_serialises() {
    let dir = unique_dir("dest_ser");
    let d = DestinationConfig::Filesystem { path: dir.clone() };
    let json = serde_json::to_string(&d).unwrap();
    let back: DestinationConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.local_path(), Some(&dir));
}
