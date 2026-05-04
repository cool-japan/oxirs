//! Grandfather-Father-Son (GFS) backup rotation.
//!
//! GFS is a classic tape-rotation scheme that keeps:
//! - **Daily** (Son): the last N daily backups.
//! - **Weekly** (Father): the last N weekly backups.
//! - **Monthly** (Grandfather): the last N monthly backups.
//!
//! `GfsRotation::prune_candidates` computes which backup IDs should be
//! deleted based on their age and tier membership.

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// GFS rotation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GfsRotation {
    /// Number of daily (Son) backups to keep.
    pub daily_count: u32,
    /// Number of weekly (Father) backups to keep.
    pub weekly_count: u32,
    /// Number of monthly (Grandfather) backups to keep.
    pub monthly_count: u32,
}

impl Default for GfsRotation {
    fn default() -> Self {
        GfsRotation {
            daily_count: 7,
            weekly_count: 4,
            monthly_count: 12,
        }
    }
}

/// Metadata describing a single backup artefact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRecord {
    /// Stable unique identifier (monotonically increasing is fine).
    pub id: u64,
    /// When the backup was created.
    pub created_at: SystemTime,
    /// Compressed size in bytes.
    pub size_bytes: u64,
    /// True if this backup represents the weekly rotation point.
    pub is_weekly: bool,
    /// True if this backup represents the monthly rotation point.
    pub is_monthly: bool,
}

impl GfsRotation {
    /// Returns the IDs of backup records that should be pruned.
    ///
    /// A record is pruned when it falls outside every applicable retention
    /// window:
    /// - outside the daily window (`now - daily_count * 1 day`)
    /// - outside the weekly window (only if `is_weekly`)
    /// - outside the monthly window (only if `is_monthly`)
    ///
    /// `records` does not need to be sorted.
    pub fn prune_candidates(&self, records: &[BackupRecord], now: SystemTime) -> Vec<u64> {
        let daily_cutoff = now
            .checked_sub(Duration::from_secs(u64::from(self.daily_count) * 86_400))
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let weekly_cutoff = now
            .checked_sub(Duration::from_secs(
                u64::from(self.weekly_count) * 7 * 86_400,
            ))
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let monthly_cutoff = now
            .checked_sub(Duration::from_secs(
                u64::from(self.monthly_count) * 30 * 86_400,
            ))
            .unwrap_or(SystemTime::UNIX_EPOCH);

        records
            .iter()
            .filter_map(|r| {
                let in_daily = r.created_at >= daily_cutoff;
                let in_weekly = r.is_weekly && r.created_at >= weekly_cutoff;
                let in_monthly = r.is_monthly && r.created_at >= monthly_cutoff;

                if !in_daily && !in_weekly && !in_monthly {
                    Some(r.id)
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn days_ago(n: u64) -> SystemTime {
        SystemTime::now() - Duration::from_secs(n * 86_400)
    }

    fn record(id: u64, days: u64, is_weekly: bool, is_monthly: bool) -> BackupRecord {
        BackupRecord {
            id,
            created_at: days_ago(days),
            size_bytes: 1024,
            is_weekly,
            is_monthly,
        }
    }

    #[test]
    fn recent_daily_not_pruned() {
        let gfs = GfsRotation {
            daily_count: 7,
            weekly_count: 4,
            monthly_count: 3,
        };
        let records = vec![record(1, 1, false, false)];
        let pruned = gfs.prune_candidates(&records, SystemTime::now());
        assert!(pruned.is_empty(), "1-day-old backup should be kept");
    }

    #[test]
    fn old_non_tier_backup_pruned() {
        let gfs = GfsRotation {
            daily_count: 7,
            weekly_count: 4,
            monthly_count: 3,
        };
        let records = vec![record(1, 100, false, false)];
        let pruned = gfs.prune_candidates(&records, SystemTime::now());
        assert_eq!(pruned, vec![1], "100-day-old plain backup should be pruned");
    }

    #[test]
    fn weekly_inside_window_kept() {
        let gfs = GfsRotation {
            daily_count: 7,
            weekly_count: 4,
            monthly_count: 3,
        };
        let records = vec![record(1, 20, true, false)]; // 20 days = ~3 weeks, inside 4-week window
        let pruned = gfs.prune_candidates(&records, SystemTime::now());
        assert!(
            pruned.is_empty(),
            "20-day-old weekly backup should be kept (4-week window)"
        );
    }

    #[test]
    fn weekly_outside_window_pruned() {
        let gfs = GfsRotation {
            daily_count: 7,
            weekly_count: 2,
            monthly_count: 1,
        };
        // weekly_count=2 => window = 14 days
        let records = vec![record(1, 20, true, false)];
        let pruned = gfs.prune_candidates(&records, SystemTime::now());
        assert_eq!(
            pruned,
            vec![1],
            "20-day-old weekly should be pruned (2-week window)"
        );
    }

    #[test]
    fn monthly_inside_window_kept() {
        let gfs = GfsRotation {
            daily_count: 7,
            weekly_count: 4,
            monthly_count: 3,
        };
        let records = vec![record(1, 80, false, true)]; // ~2.7 months
        let pruned = gfs.prune_candidates(&records, SystemTime::now());
        assert!(
            pruned.is_empty(),
            "80-day-old monthly backup should be kept (3-month window)"
        );
    }

    #[test]
    fn gfs_100_day_simulation() {
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
                size_bytes: 512,
                is_weekly: day % 7 == 0,
                is_monthly: day % 30 == 0,
            })
            .collect();

        let pruned = gfs.prune_candidates(&records, now);
        let kept_count = 100 - pruned.len();

        // At minimum the last 7 daily backups must be kept
        assert!(
            kept_count >= 7,
            "Expected ≥7 kept backups; got {kept_count}"
        );
    }

    #[test]
    fn empty_records_returns_empty() {
        let gfs = GfsRotation::default();
        let pruned = gfs.prune_candidates(&[], SystemTime::now());
        assert!(pruned.is_empty());
    }

    #[test]
    fn serialises_round_trip() {
        let gfs = GfsRotation::default();
        let json = serde_json::to_string(&gfs).unwrap();
        let back: GfsRotation = serde_json::from_str(&json).unwrap();
        assert_eq!(back.daily_count, gfs.daily_count);
        assert_eq!(back.weekly_count, gfs.weekly_count);
        assert_eq!(back.monthly_count, gfs.monthly_count);
    }
}
