//! Retention tier model: hot / warm / cold.
//!
//! Each tier specifies how many days/weeks/months a backup of that tier should
//! be kept.  `RetentionTier::should_retain` decides whether a given backup is
//! still within its retention window.

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// Three-tier retention model.
///
/// | Tier | Granularity | Field          |
/// |------|-------------|----------------|
/// | Hot  | days        | `hot_days`     |
/// | Warm | weeks       | `warm_weeks`   |
/// | Cold | months      | `cold_months`  |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionTier {
    /// Number of days to keep every daily backup (hot tier).
    pub hot_days: u32,
    /// Number of weeks to keep weekly backups (warm tier).
    pub warm_weeks: u32,
    /// Number of months to keep monthly backups (cold tier).
    pub cold_months: u32,
}

impl RetentionTier {
    /// Sensible production defaults: 7-day hot, 4-week warm, 12-month cold.
    pub fn standard() -> Self {
        RetentionTier {
            hot_days: 7,
            warm_weeks: 4,
            cold_months: 12,
        }
    }

    /// Minimal retention useful for tests: 1-day hot, 1-week warm, 1-month cold.
    pub fn minimal() -> Self {
        RetentionTier {
            hot_days: 1,
            warm_weeks: 1,
            cold_months: 1,
        }
    }

    /// Returns `true` if `backup_time` should still be retained given `now`.
    ///
    /// Rules (applied in order — first match wins):
    /// 1. If age ≤ `hot_days` → retain (hot tier, every backup).
    /// 2. If `is_weekly` and age ≤ `warm_weeks * 7 days` → retain (warm tier).
    /// 3. If `is_monthly` and age ≤ `cold_months * 30 days` → retain (cold tier).
    /// 4. Otherwise → prune.
    ///
    /// If `now` is earlier than `backup_time` (clock drift) the backup is
    /// always retained.
    pub fn should_retain(
        &self,
        backup_time: SystemTime,
        now: SystemTime,
        is_weekly: bool,
        is_monthly: bool,
    ) -> bool {
        let age = match now.duration_since(backup_time) {
            Ok(d) => d,
            Err(_) => return true, // backup_time is in the future — always keep
        };

        let hot_cutoff = Duration::from_secs(u64::from(self.hot_days) * 86_400);
        if age <= hot_cutoff {
            return true;
        }

        if is_weekly {
            let warm_cutoff = Duration::from_secs(u64::from(self.warm_weeks) * 7 * 86_400);
            if age <= warm_cutoff {
                return true;
            }
        }

        if is_monthly {
            let cold_cutoff = Duration::from_secs(u64::from(self.cold_months) * 30 * 86_400);
            if age <= cold_cutoff {
                return true;
            }
        }

        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn days_ago(n: u64, now: SystemTime) -> SystemTime {
        now - Duration::from_secs(n * 86_400)
    }

    #[test]
    fn hot_tier_retains_recent() {
        let tier = RetentionTier::standard(); // hot = 7d
        let now = SystemTime::now();
        assert!(tier.should_retain(days_ago(3, now), now, false, false));
        assert!(tier.should_retain(days_ago(7, now), now, false, false));
    }

    #[test]
    fn hot_tier_prunes_old_non_weekly() {
        let tier = RetentionTier::standard(); // hot = 7d
        let now = SystemTime::now();
        assert!(!tier.should_retain(days_ago(8, now), now, false, false));
        assert!(!tier.should_retain(days_ago(14, now), now, false, false));
    }

    #[test]
    fn warm_tier_retains_weekly_within_window() {
        let tier = RetentionTier::standard(); // warm = 4w = 28d
        let now = SystemTime::now();
        assert!(tier.should_retain(days_ago(21, now), now, true, false));
        assert!(tier.should_retain(days_ago(28, now), now, true, false));
    }

    #[test]
    fn warm_tier_prunes_weekly_outside_window() {
        let tier = RetentionTier::standard(); // warm = 4w = 28d
                                              // 35 days old, is_weekly=true but outside warm window, not monthly
        let now = SystemTime::now();
        assert!(!tier.should_retain(days_ago(35, now), now, true, false));
    }

    #[test]
    fn cold_tier_retains_monthly_within_window() {
        let tier = RetentionTier::standard(); // cold = 12m = 360d
        let now = SystemTime::now();
        assert!(tier.should_retain(days_ago(90, now), now, false, true));
        assert!(tier.should_retain(days_ago(360, now), now, false, true));
    }

    #[test]
    fn cold_tier_prunes_monthly_outside_window() {
        let tier = RetentionTier::standard(); // cold = 12m = 360d
        let now = SystemTime::now();
        assert!(!tier.should_retain(days_ago(365, now), now, false, true));
    }

    #[test]
    fn future_backup_always_retained() {
        let tier = RetentionTier::standard();
        let future = SystemTime::now() + Duration::from_secs(3_600);
        assert!(tier.should_retain(future, SystemTime::now(), false, false));
    }

    #[test]
    fn minimal_tier_defaults() {
        let t = RetentionTier::minimal();
        assert_eq!(t.hot_days, 1);
        assert_eq!(t.warm_weeks, 1);
        assert_eq!(t.cold_months, 1);
    }

    #[test]
    fn serialises_round_trip() {
        let tier = RetentionTier::standard();
        let json = serde_json::to_string(&tier).unwrap();
        let back: RetentionTier = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hot_days, tier.hot_days);
        assert_eq!(back.warm_weeks, tier.warm_weeks);
        assert_eq!(back.cold_months, tier.cold_months);
    }
}
