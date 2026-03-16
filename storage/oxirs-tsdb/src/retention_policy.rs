//! Time-series retention policy management.
//!
//! Provides configurable retention policies that control how long data
//! is kept and when it should be downsampled to lower resolution.

use std::collections::HashMap;

use crate::DataPoint;

/// How long data should be retained.
#[derive(Debug, Clone, PartialEq)]
pub enum RetentionDuration {
    Hours(u64),
    Days(u64),
    Weeks(u64),
    Forever,
}

impl RetentionDuration {
    const MS_PER_HOUR: u64 = 3_600_000;
    const MS_PER_DAY: u64 = 86_400_000;
    const MS_PER_WEEK: u64 = 604_800_000;

    /// Convert to milliseconds. Returns `None` for `Forever`.
    pub fn to_ms(&self) -> Option<u64> {
        match self {
            RetentionDuration::Hours(h) => Some(h * Self::MS_PER_HOUR),
            RetentionDuration::Days(d) => Some(d * Self::MS_PER_DAY),
            RetentionDuration::Weeks(w) => Some(w * Self::MS_PER_WEEK),
            RetentionDuration::Forever => None,
        }
    }
}

/// A retention policy definition for a time-series.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    pub name: String,
    pub duration: RetentionDuration,
    /// After this duration, downsample data instead of deleting it.
    pub downsample_after: Option<RetentionDuration>,
    /// Resolution for downsampled data in milliseconds.
    pub downsample_resolution_ms: u64,
}

impl RetentionPolicy {
    pub fn new(name: impl Into<String>, duration: RetentionDuration) -> Self {
        RetentionPolicy {
            name: name.into(),
            duration,
            downsample_after: None,
            downsample_resolution_ms: 60_000, // default 1 minute
        }
    }

    pub fn with_downsample(mut self, after: RetentionDuration, resolution_ms: u64) -> Self {
        self.downsample_after = Some(after);
        self.downsample_resolution_ms = resolution_ms;
        self
    }
}

/// Errors from retention management operations.
#[derive(Debug, PartialEq)]
pub enum RetentionError {
    PolicyAlreadyExists(String),
    PolicyNotFound(String),
    DefaultNotSet,
}

impl std::fmt::Display for RetentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetentionError::PolicyAlreadyExists(n) => {
                write!(f, "Retention policy already exists: {n}")
            }
            RetentionError::PolicyNotFound(n) => {
                write!(f, "Retention policy not found: {n}")
            }
            RetentionError::DefaultNotSet => {
                write!(f, "No default retention policy has been set")
            }
        }
    }
}

impl std::error::Error for RetentionError {}

/// Manager for multiple named retention policies.
pub struct RetentionManager {
    policies: HashMap<String, RetentionPolicy>,
    default_policy: Option<String>,
}

impl RetentionManager {
    /// Create an empty manager with no policies.
    pub fn new() -> Self {
        RetentionManager {
            policies: HashMap::new(),
            default_policy: None,
        }
    }

    /// Register a new retention policy.
    pub fn add_policy(&mut self, policy: RetentionPolicy) -> Result<(), RetentionError> {
        if self.policies.contains_key(&policy.name) {
            return Err(RetentionError::PolicyAlreadyExists(policy.name.clone()));
        }
        self.policies.insert(policy.name.clone(), policy);
        Ok(())
    }

    /// Remove a named retention policy.
    pub fn remove_policy(&mut self, name: &str) -> Result<(), RetentionError> {
        if self.policies.remove(name).is_none() {
            return Err(RetentionError::PolicyNotFound(name.to_string()));
        }
        if self.default_policy.as_deref() == Some(name) {
            self.default_policy = None;
        }
        Ok(())
    }

    /// Set the default policy (must already be registered).
    pub fn set_default(&mut self, name: &str) -> Result<(), RetentionError> {
        if !self.policies.contains_key(name) {
            return Err(RetentionError::PolicyNotFound(name.to_string()));
        }
        self.default_policy = Some(name.to_string());
        Ok(())
    }

    /// Look up the effective policy for a series name.
    ///
    /// Tries an exact match by series name prefix, then falls back to the default.
    pub fn effective_policy(&self, series_name: &str) -> Option<&RetentionPolicy> {
        // Check for an exact name match first
        if let Some(p) = self.policies.get(series_name) {
            return Some(p);
        }
        // Check for prefix match (longest wins)
        let best = self
            .policies
            .keys()
            .filter(|k| series_name.starts_with(k.as_str()))
            .max_by_key(|k| k.len());
        if let Some(name) = best {
            return self.policies.get(name);
        }
        // Fall back to default
        self.default_policy
            .as_deref()
            .and_then(|n| self.policies.get(n))
    }

    /// Determine whether a data point with the given timestamp should be retained.
    pub fn should_retain(&self, policy_name: &str, timestamp_ms: u64, now_ms: u64) -> bool {
        let policy = match self.policies.get(policy_name) {
            Some(p) => p,
            None => return true, // unknown policy — retain by default
        };
        match policy.duration.to_ms() {
            None => true, // Forever
            Some(duration_ms) => {
                let cutoff = now_ms.saturating_sub(duration_ms);
                timestamp_ms >= cutoff
            }
        }
    }

    /// Remove expired data points from a mutable vector.
    ///
    /// Returns the number of removed points.
    pub fn apply_retention(
        &self,
        data: &mut Vec<DataPoint>,
        policy_name: &str,
        now_ms: u64,
    ) -> usize {
        let before = data.len();
        data.retain(|dp| {
            let ts_ms = dp.timestamp.timestamp_millis() as u64;
            self.should_retain(policy_name, ts_ms, now_ms)
        });
        before - data.len()
    }

    /// Return all policies sorted by name.
    pub fn list_policies(&self) -> Vec<&RetentionPolicy> {
        let mut v: Vec<&RetentionPolicy> = self.policies.values().collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    /// Return the name of the default policy, if any.
    pub fn default_policy_name(&self) -> Option<&str> {
        self.default_policy.as_deref()
    }

    /// Return the count of registered policies.
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }
}

impl Default for RetentionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn make_policy(name: &str) -> RetentionPolicy {
        RetentionPolicy::new(name, RetentionDuration::Days(30))
    }

    fn ts_ms_to_point(ts_ms: u64) -> DataPoint {
        let secs = (ts_ms / 1000) as i64;
        let nanos = ((ts_ms % 1000) * 1_000_000) as u32;
        DataPoint {
            timestamp: Utc.timestamp_opt(secs, nanos).unwrap(),
            value: 1.0,
        }
    }

    // RetentionDuration tests
    #[test]
    fn test_hours_to_ms() {
        assert_eq!(RetentionDuration::Hours(1).to_ms(), Some(3_600_000));
        assert_eq!(RetentionDuration::Hours(24).to_ms(), Some(86_400_000));
    }

    #[test]
    fn test_days_to_ms() {
        assert_eq!(RetentionDuration::Days(1).to_ms(), Some(86_400_000));
        assert_eq!(RetentionDuration::Days(7).to_ms(), Some(604_800_000));
    }

    #[test]
    fn test_weeks_to_ms() {
        assert_eq!(RetentionDuration::Weeks(1).to_ms(), Some(604_800_000));
        assert_eq!(RetentionDuration::Weeks(2).to_ms(), Some(1_209_600_000));
    }

    #[test]
    fn test_forever_to_ms_none() {
        assert_eq!(RetentionDuration::Forever.to_ms(), None);
    }

    // RetentionManager creation
    #[test]
    fn test_new_empty() {
        let m = RetentionManager::new();
        assert_eq!(m.policy_count(), 0);
        assert!(m.default_policy_name().is_none());
    }

    #[test]
    fn test_default_trait() {
        let m = RetentionManager::default();
        assert_eq!(m.policy_count(), 0);
    }

    // add_policy
    #[test]
    fn test_add_policy_ok() {
        let mut m = RetentionManager::new();
        m.add_policy(make_policy("raw")).expect("should succeed");
        assert_eq!(m.policy_count(), 1);
    }

    #[test]
    fn test_add_policy_duplicate_error() {
        let mut m = RetentionManager::new();
        m.add_policy(make_policy("raw")).expect("should succeed");
        let err = m.add_policy(make_policy("raw")).unwrap_err();
        assert_eq!(err, RetentionError::PolicyAlreadyExists("raw".to_string()));
    }

    // remove_policy
    #[test]
    fn test_remove_policy_ok() {
        let mut m = RetentionManager::new();
        m.add_policy(make_policy("raw")).expect("should succeed");
        m.remove_policy("raw").expect("should succeed");
        assert_eq!(m.policy_count(), 0);
    }

    #[test]
    fn test_remove_policy_not_found() {
        let mut m = RetentionManager::new();
        let err = m.remove_policy("missing").unwrap_err();
        assert_eq!(err, RetentionError::PolicyNotFound("missing".to_string()));
    }

    #[test]
    fn test_remove_clears_default() {
        let mut m = RetentionManager::new();
        m.add_policy(make_policy("raw")).expect("should succeed");
        m.set_default("raw").expect("should succeed");
        m.remove_policy("raw").expect("should succeed");
        assert!(m.default_policy_name().is_none());
    }

    // set_default
    #[test]
    fn test_set_default_ok() {
        let mut m = RetentionManager::new();
        m.add_policy(make_policy("raw")).expect("should succeed");
        m.set_default("raw").expect("should succeed");
        assert_eq!(m.default_policy_name(), Some("raw"));
    }

    #[test]
    fn test_set_default_not_found() {
        let mut m = RetentionManager::new();
        let err = m.set_default("missing").unwrap_err();
        assert_eq!(err, RetentionError::PolicyNotFound("missing".to_string()));
    }

    // effective_policy
    #[test]
    fn test_effective_policy_exact_match() {
        let mut m = RetentionManager::new();
        m.add_policy(make_policy("sensor.temp"))
            .expect("should succeed");
        let p = m.effective_policy("sensor.temp").expect("should succeed");
        assert_eq!(p.name, "sensor.temp");
    }

    #[test]
    fn test_effective_policy_prefix_match() {
        let mut m = RetentionManager::new();
        m.add_policy(make_policy("sensor")).expect("should succeed");
        let p = m
            .effective_policy("sensor.temperature")
            .expect("should succeed");
        assert_eq!(p.name, "sensor");
    }

    #[test]
    fn test_effective_policy_falls_back_to_default() {
        let mut m = RetentionManager::new();
        m.add_policy(make_policy("default"))
            .expect("should succeed");
        m.set_default("default").expect("should succeed");
        let p = m
            .effective_policy("unknown.series")
            .expect("should succeed");
        assert_eq!(p.name, "default");
    }

    #[test]
    fn test_effective_policy_none_when_no_match() {
        let m = RetentionManager::new();
        assert!(m.effective_policy("unknown").is_none());
    }

    // should_retain
    #[test]
    fn test_should_retain_within_window() {
        let mut m = RetentionManager::new();
        m.add_policy(RetentionPolicy::new("hourly", RetentionDuration::Hours(1)))
            .expect("should succeed");
        let now = 3_600_000u64 * 2; // 2 hours in ms
        let ts = now - 1_800_000; // 30 min ago
        assert!(m.should_retain("hourly", ts, now));
    }

    #[test]
    fn test_should_retain_outside_window() {
        let mut m = RetentionManager::new();
        m.add_policy(RetentionPolicy::new("hourly", RetentionDuration::Hours(1)))
            .expect("should succeed");
        let now = 3_600_000u64 * 2;
        let ts = now - 3_600_001; // just over 1 hour ago
        assert!(!m.should_retain("hourly", ts, now));
    }

    #[test]
    fn test_should_retain_forever() {
        let mut m = RetentionManager::new();
        m.add_policy(RetentionPolicy::new("forever", RetentionDuration::Forever))
            .expect("should succeed");
        assert!(m.should_retain("forever", 0, u64::MAX));
    }

    #[test]
    fn test_should_retain_unknown_policy_returns_true() {
        let m = RetentionManager::new();
        assert!(m.should_retain("nonexistent", 0, 9999999));
    }

    // apply_retention
    #[test]
    fn test_apply_retention_removes_old() {
        let mut m = RetentionManager::new();
        m.add_policy(RetentionPolicy::new("hourly", RetentionDuration::Hours(1)))
            .expect("should succeed");
        let now_ms = 7_200_000u64; // 2 hours
        let mut data = vec![
            ts_ms_to_point(now_ms - 500_000),   // 8.3 min ago — keep
            ts_ms_to_point(now_ms - 3_700_000), // just over 1 hour — drop
        ];
        let removed = m.apply_retention(&mut data, "hourly", now_ms);
        assert_eq!(removed, 1);
        assert_eq!(data.len(), 1);
    }

    #[test]
    fn test_apply_retention_keeps_all_in_window() {
        let mut m = RetentionManager::new();
        m.add_policy(RetentionPolicy::new("daily", RetentionDuration::Days(1)))
            .expect("should succeed");
        let now_ms = 86_400_000u64 * 2;
        let mut data: Vec<DataPoint> = (0..5)
            .map(|i| ts_ms_to_point(now_ms - i * 10_000))
            .collect();
        let removed = m.apply_retention(&mut data, "daily", now_ms);
        assert_eq!(removed, 0);
        assert_eq!(data.len(), 5);
    }

    #[test]
    fn test_apply_retention_returns_count() {
        let mut m = RetentionManager::new();
        m.add_policy(RetentionPolicy::new("short", RetentionDuration::Hours(1)))
            .expect("should succeed");
        let now_ms = 7_200_000u64;
        let mut data = vec![ts_ms_to_point(0), ts_ms_to_point(1), ts_ms_to_point(2)];
        let removed = m.apply_retention(&mut data, "short", now_ms);
        assert_eq!(removed, 3);
        assert!(data.is_empty());
    }

    // list_policies
    #[test]
    fn test_list_policies_sorted() {
        let mut m = RetentionManager::new();
        m.add_policy(make_policy("zeta")).expect("should succeed");
        m.add_policy(make_policy("alpha")).expect("should succeed");
        m.add_policy(make_policy("mu")).expect("should succeed");
        let list = m.list_policies();
        assert_eq!(list[0].name, "alpha");
        assert_eq!(list[1].name, "mu");
        assert_eq!(list[2].name, "zeta");
    }

    #[test]
    fn test_list_policies_empty() {
        let m = RetentionManager::new();
        assert!(m.list_policies().is_empty());
    }

    // Error display
    #[test]
    fn test_error_display_already_exists() {
        let e = RetentionError::PolicyAlreadyExists("raw".to_string());
        assert!(e.to_string().contains("already exists"));
    }

    #[test]
    fn test_error_display_not_found() {
        let e = RetentionError::PolicyNotFound("raw".to_string());
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn test_error_display_default_not_set() {
        let e = RetentionError::DefaultNotSet;
        assert!(e.to_string().contains("default"));
    }

    // RetentionPolicy builder
    #[test]
    fn test_policy_with_downsample() {
        let p = RetentionPolicy::new("raw", RetentionDuration::Days(7))
            .with_downsample(RetentionDuration::Days(1), 300_000);
        assert!(p.downsample_after.is_some());
        assert_eq!(p.downsample_resolution_ms, 300_000);
    }

    #[test]
    fn test_policy_count_after_operations() {
        let mut m = RetentionManager::new();
        m.add_policy(make_policy("a")).expect("should succeed");
        m.add_policy(make_policy("b")).expect("should succeed");
        m.remove_policy("a").expect("should succeed");
        assert_eq!(m.policy_count(), 1);
    }

    #[test]
    fn test_should_retain_boundary_exact() {
        let mut m = RetentionManager::new();
        // 1 day = 86_400_000 ms
        m.add_policy(RetentionPolicy::new("daily", RetentionDuration::Days(1)))
            .expect("should succeed");
        let now_ms = 86_400_000u64 * 2;
        let cutoff = now_ms - 86_400_000;
        // Exactly at cutoff should be retained (>=)
        assert!(m.should_retain("daily", cutoff, now_ms));
        // One ms before cutoff should not be retained
        assert!(!m.should_retain("daily", cutoff - 1, now_ms));
    }
}
