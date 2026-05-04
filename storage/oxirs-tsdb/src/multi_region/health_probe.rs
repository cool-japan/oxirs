//! Region health probe with timeout-based failure detection.
//!
//! Tracks how many ticks have elapsed since each region's last heartbeat. A
//! region is considered:
//!
//! - `Healthy`  — heartbeat was received within `suspect_after` ticks.
//! - `Suspect`  — `suspect_after ≤ silence < failure_threshold` ticks elapsed.
//! - `Failed`   — `silence ≥ failure_threshold` ticks.
//!
//! The probe is purely synchronous and has no I/O dependencies; production
//! callers can wire `record_heartbeat` to a background async task that pings
//! every region every few seconds.
//!
//! Snapshot semantics: [`HealthProbe::snapshot`] returns a stable, owned
//! [`RegionHealthSnapshot`] that does not change underneath callers — useful
//! when the routing layer evaluates many writes against a consistent view.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::routing::RegionId;

// ─────────────────────────────────────────────────────────────────────────────
// Status
// ─────────────────────────────────────────────────────────────────────────────

/// Health status of a region as seen by the probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionStatus {
    /// Heartbeats are flowing; the region is fully reachable.
    Healthy,
    /// Heartbeats have been silent for longer than `suspect_after` but less
    /// than `failure_threshold` ticks.
    Suspect,
    /// Heartbeats have been silent for at least `failure_threshold` ticks.
    Failed,
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Tunables for [`HealthProbe`].
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Number of consecutive silent ticks after which a region transitions
    /// from `Healthy` to `Suspect`.
    pub suspect_after: u32,
    /// Total silent ticks after which a region transitions to `Failed`.
    pub failure_threshold: u32,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            suspect_after: 5,
            failure_threshold: 15,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Probe
// ─────────────────────────────────────────────────────────────────────────────

/// Per-region health probe state.
#[derive(Debug, Clone)]
pub struct HealthProbe {
    config: HealthConfig,
    silence: BTreeMap<RegionId, u32>,
    forced: BTreeMap<RegionId, RegionStatus>,
}

impl HealthProbe {
    /// Build a probe that tracks `regions`. Each region starts `Healthy`.
    pub fn new(regions: Vec<RegionId>, config: HealthConfig) -> Self {
        let silence = regions.into_iter().map(|r| (r, 0)).collect();
        Self {
            config,
            silence,
            forced: BTreeMap::new(),
        }
    }

    /// Returns the configured tunables.
    pub fn config(&self) -> &HealthConfig {
        &self.config
    }

    /// Number of regions tracked by the probe.
    pub fn region_count(&self) -> usize {
        self.silence.len()
    }

    /// Reset the silent-tick counter for `region`. Call this whenever a
    /// heartbeat (or an actual cross-region message) is received from that
    /// region.
    pub fn record_heartbeat(&mut self, region: &RegionId) {
        if let Some(slot) = self.silence.get_mut(region) {
            *slot = 0;
        }
        // A heartbeat clears any sticky forced status.
        self.forced.remove(region);
    }

    /// Advance the probe by one logical tick: every region's silence counter
    /// is incremented.
    pub fn tick(&mut self) {
        for slot in self.silence.values_mut() {
            *slot = slot.saturating_add(1);
        }
    }

    /// Override the status of `region` until the next heartbeat. Useful in
    /// tests or for operator-driven manual failover.
    pub fn force_status(&mut self, region: impl Into<RegionId>, status: RegionStatus) {
        let r = region.into();
        if self.silence.contains_key(&r) {
            self.forced.insert(r, status);
        }
    }

    /// Compute the current status of `region` (or `Failed` if the region is
    /// unknown to the probe).
    pub fn status_of(&self, region: &RegionId) -> RegionStatus {
        if let Some(forced) = self.forced.get(region) {
            return *forced;
        }
        match self.silence.get(region) {
            None => RegionStatus::Failed,
            Some(&silence) => {
                if silence >= self.config.failure_threshold {
                    RegionStatus::Failed
                } else if silence >= self.config.suspect_after {
                    RegionStatus::Suspect
                } else {
                    RegionStatus::Healthy
                }
            }
        }
    }

    /// Take a stable snapshot of every region's current status.
    pub fn snapshot(&self) -> RegionHealthSnapshot {
        let entries = self
            .silence
            .keys()
            .map(|r| (r.clone(), self.status_of(r)))
            .collect();
        RegionHealthSnapshot { entries }
    }

    /// List regions currently considered `Failed`.
    pub fn failed_regions(&self) -> Vec<RegionId> {
        self.silence
            .keys()
            .filter(|r| self.status_of(r) == RegionStatus::Failed)
            .cloned()
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// Stable snapshot of region health.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegionHealthSnapshot {
    entries: BTreeMap<RegionId, RegionStatus>,
}

impl RegionHealthSnapshot {
    /// Build a snapshot from raw entries (mostly used by tests).
    pub fn from_entries<I>(entries: I) -> Self
    where
        I: IntoIterator<Item = (RegionId, RegionStatus)>,
    {
        Self {
            entries: entries.into_iter().collect(),
        }
    }

    /// Number of regions in the snapshot.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the snapshot contains no regions.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over `(region, status)` pairs in lexicographic region order.
    pub fn iter(&self) -> impl Iterator<Item = (&RegionId, &RegionStatus)> {
        self.entries.iter()
    }

    /// Look up a region's status. Unknown regions return `Failed`.
    pub fn status_of(&self, region: &RegionId) -> RegionStatus {
        self.entries
            .get(region)
            .copied()
            .unwrap_or(RegionStatus::Failed)
    }

    /// Returns the regions that are not `Failed`.
    pub fn healthy_regions(&self) -> Vec<RegionId> {
        self.entries
            .iter()
            .filter(|(_, s)| **s != RegionStatus::Failed)
            .map(|(r, _)| r.clone())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(regions: &[&str], cfg: HealthConfig) -> HealthProbe {
        HealthProbe::new(regions.iter().map(|s| s.to_string()).collect(), cfg)
    }

    #[test]
    fn freshly_built_probe_is_healthy() {
        let p = probe(&["us", "eu"], HealthConfig::default());
        assert_eq!(p.status_of(&"us".to_string()), RegionStatus::Healthy);
        assert_eq!(p.status_of(&"eu".to_string()), RegionStatus::Healthy);
    }

    #[test]
    fn unknown_region_is_failed() {
        let p = probe(&["us"], HealthConfig::default());
        assert_eq!(p.status_of(&"void".to_string()), RegionStatus::Failed);
    }

    #[test]
    fn ticking_eventually_marks_suspect_then_failed() {
        let cfg = HealthConfig {
            suspect_after: 3,
            failure_threshold: 6,
        };
        let mut p = probe(&["us"], cfg);
        let r = "us".to_string();
        for _ in 0..2 {
            p.tick();
        }
        assert_eq!(p.status_of(&r), RegionStatus::Healthy);
        for _ in 0..2 {
            p.tick();
        }
        assert_eq!(p.status_of(&r), RegionStatus::Suspect);
        for _ in 0..3 {
            p.tick();
        }
        assert_eq!(p.status_of(&r), RegionStatus::Failed);
    }

    #[test]
    fn heartbeat_resets_silence_counter() {
        let cfg = HealthConfig {
            suspect_after: 2,
            failure_threshold: 4,
        };
        let mut p = probe(&["us"], cfg);
        let r = "us".to_string();
        for _ in 0..3 {
            p.tick();
        }
        assert_eq!(p.status_of(&r), RegionStatus::Suspect);
        p.record_heartbeat(&r);
        assert_eq!(p.status_of(&r), RegionStatus::Healthy);
    }

    #[test]
    fn force_status_overrides_until_heartbeat() {
        let mut p = probe(&["us"], HealthConfig::default());
        let r = "us".to_string();
        p.force_status("us", RegionStatus::Failed);
        assert_eq!(p.status_of(&r), RegionStatus::Failed);
        p.record_heartbeat(&r);
        assert_eq!(p.status_of(&r), RegionStatus::Healthy);
    }

    #[test]
    fn snapshot_contains_all_regions() {
        let p = probe(&["us", "eu", "ap"], HealthConfig::default());
        let snap = p.snapshot();
        assert_eq!(snap.len(), 3);
        for r in ["us", "eu", "ap"] {
            assert_eq!(snap.status_of(&r.to_string()), RegionStatus::Healthy);
        }
    }

    #[test]
    fn snapshot_iter_is_sorted() {
        let p = probe(&["c", "a", "b"], HealthConfig::default());
        let snap = p.snapshot();
        let names: Vec<&RegionId> = snap.iter().map(|(r, _)| r).collect();
        assert_eq!(
            names,
            vec![&"a".to_string(), &"b".to_string(), &"c".to_string()]
        );
    }

    #[test]
    fn failed_regions_listed() {
        let cfg = HealthConfig {
            suspect_after: 2,
            failure_threshold: 4,
        };
        let mut p = probe(&["us", "eu"], cfg);
        for _ in 0..5 {
            p.tick();
        }
        let failed = p.failed_regions();
        assert!(failed.contains(&"us".to_string()));
        assert!(failed.contains(&"eu".to_string()));
    }

    #[test]
    fn snapshot_unknown_region_failed() {
        let snap = RegionHealthSnapshot::default();
        assert_eq!(snap.status_of(&"x".to_string()), RegionStatus::Failed);
    }

    #[test]
    fn record_heartbeat_for_unknown_region_is_noop() {
        let mut p = probe(&["us"], HealthConfig::default());
        p.record_heartbeat(&"void".to_string());
        assert_eq!(p.status_of(&"us".to_string()), RegionStatus::Healthy);
        assert_eq!(p.status_of(&"void".to_string()), RegionStatus::Failed);
    }

    #[test]
    fn force_status_only_for_known_regions() {
        let mut p = probe(&["us"], HealthConfig::default());
        p.force_status("ghost", RegionStatus::Suspect);
        // Ghost remains unknown ⇒ Failed; no forced status retained.
        assert_eq!(p.status_of(&"ghost".to_string()), RegionStatus::Failed);
    }

    #[test]
    fn healthy_regions_filters_failed() {
        let mut p = probe(&["a", "b", "c"], HealthConfig::default());
        p.force_status("b", RegionStatus::Failed);
        let snap = p.snapshot();
        let healthy = snap.healthy_regions();
        assert!(healthy.contains(&"a".to_string()));
        assert!(!healthy.contains(&"b".to_string()));
        assert!(healthy.contains(&"c".to_string()));
    }
}
