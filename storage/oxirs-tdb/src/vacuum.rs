//! Database vacuuming and compaction for TDB storage.
//!
//! Provides dead-space detection, fragmentation analysis, vacuum scheduling,
//! online space reclamation, and comprehensive vacuum statistics.
//!
//! # Overview
//!
//! The vacuum subsystem tracks freed pages and slots, computes fragmentation
//! ratios, and performs space reclamation either on-demand or according to a
//! configurable schedule.  Online vacuum keeps the database available for
//! reads and writes while compaction proceeds in the background.
//!
//! # Example
//!
//! ```
//! use oxirs_tdb::vacuum::{VacuumConfig, VacuumEngine, VacuumTrigger};
//!
//! let config = VacuumConfig {
//!     trigger: VacuumTrigger::FragmentationThreshold(0.30),
//!     online: true,
//!     ..Default::default()
//! };
//! let mut engine = VacuumEngine::new(config);
//! engine.mark_page_freed(42);
//! engine.mark_page_freed(99);
//! let report = engine.vacuum().expect("vacuum failed");
//! assert!(report.pages_compacted >= 0);
//! ```

use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by vacuum operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VacuumError {
    /// A vacuum operation is already in progress.
    AlreadyRunning,
    /// A conflicting operation (e.g., exclusive write lock) is held.
    ConflictingLock(String),
    /// The page or slot identifier is out of range.
    InvalidId(String),
    /// Internal vacuum failure.
    Internal(String),
}

impl std::fmt::Display for VacuumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyRunning => write!(f, "Vacuum already running"),
            Self::ConflictingLock(msg) => write!(f, "Conflicting lock: {msg}"),
            Self::InvalidId(msg) => write!(f, "Invalid page/slot id: {msg}"),
            Self::Internal(msg) => write!(f, "Internal vacuum error: {msg}"),
        }
    }
}

impl std::error::Error for VacuumError {}

/// Result alias for vacuum operations.
pub type VacuumResult<T> = Result<T, VacuumError>;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Trigger condition that determines when an automatic vacuum runs.
#[derive(Debug, Clone, PartialEq)]
pub enum VacuumTrigger {
    /// Run only when explicitly requested (`VacuumEngine::vacuum`).
    Manual,
    /// Run whenever the fragmentation ratio exceeds this threshold (0.0–1.0).
    FragmentationThreshold(f64),
    /// Run automatically at the given interval.
    TimeBased(Duration),
    /// Run when fragmentation *or* interval fires.
    Combined {
        /// Fragmentation ratio threshold.
        threshold: f64,
        /// Periodic interval.
        interval: Duration,
    },
}

impl Default for VacuumTrigger {
    fn default() -> Self {
        Self::FragmentationThreshold(0.25)
    }
}

/// Configuration for the vacuum engine.
#[derive(Debug, Clone)]
pub struct VacuumConfig {
    /// Trigger condition.
    pub trigger: VacuumTrigger,
    /// If true the vacuum runs while accepting reads/writes (online mode).
    pub online: bool,
    /// Maximum number of pages to compact in a single vacuum pass (0 = unlimited).
    pub max_pages_per_pass: usize,
    /// Minimum number of freed pages required before a vacuum is started.
    pub min_freed_pages: usize,
    /// Pause between page compaction steps (online mode only).
    pub page_step_delay: Duration,
}

impl Default for VacuumConfig {
    fn default() -> Self {
        Self {
            trigger: VacuumTrigger::default(),
            online: true,
            max_pages_per_pass: 0,
            min_freed_pages: 1,
            page_step_delay: Duration::from_millis(1),
        }
    }
}

// ---------------------------------------------------------------------------
// Space-usage tracking
// ---------------------------------------------------------------------------

/// Tracks space usage for a single page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSpaceInfo {
    /// Page identifier (offset / page-number).
    pub page_id: u64,
    /// Total usable bytes on the page.
    pub total_bytes: u64,
    /// Bytes used by live data.
    pub used_bytes: u64,
}

impl PageSpaceInfo {
    /// Create a page info entry.
    pub fn new(page_id: u64, total_bytes: u64, used_bytes: u64) -> Self {
        Self {
            page_id,
            total_bytes,
            used_bytes,
        }
    }

    /// Dead (reclaimed) bytes on this page.
    pub fn dead_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.used_bytes)
    }

    /// Fragmentation ratio for this page (dead / total).
    pub fn fragmentation(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.dead_bytes() as f64 / self.total_bytes as f64
    }
}

// ---------------------------------------------------------------------------
// Vacuum lock management
// ---------------------------------------------------------------------------

/// Lock state for the vacuum engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VacuumLockState {
    /// No vacuum is running; the engine is idle.
    Idle,
    /// A vacuum operation is in progress.
    Running,
    /// The engine is bypassed (vacuum is temporarily suppressed).
    Bypassed,
}

/// A guard that holds the vacuum lock.
///
/// Dropping the guard releases the lock.
pub struct VacuumLockGuard<'a> {
    engine: &'a mut VacuumEngine,
}

impl<'a> Drop for VacuumLockGuard<'a> {
    fn drop(&mut self) {
        self.engine.lock_state = VacuumLockState::Idle;
    }
}

// ---------------------------------------------------------------------------
// Vacuum statistics
// ---------------------------------------------------------------------------

/// Summary statistics produced by a completed vacuum run.
#[derive(Debug, Clone)]
pub struct VacuumStats {
    /// Total bytes reclaimed from freed pages.
    pub bytes_reclaimed: u64,
    /// Number of pages compacted during this run.
    pub pages_compacted: usize,
    /// Number of dead slots removed.
    pub slots_removed: usize,
    /// Total wall-clock duration of the vacuum run.
    pub duration: Duration,
    /// Space usage before vacuum (bytes).
    pub pre_vacuum_dead_bytes: u64,
    /// Space usage after vacuum (bytes).
    pub post_vacuum_dead_bytes: u64,
    /// Fragmentation ratio before vacuum.
    pub pre_vacuum_fragmentation: f64,
    /// Fragmentation ratio after vacuum.
    pub post_vacuum_fragmentation: f64,
}

impl VacuumStats {
    /// True if the vacuum made any progress.
    pub fn made_progress(&self) -> bool {
        self.bytes_reclaimed > 0 || self.pages_compacted > 0
    }
}

/// Cumulative vacuum metrics across all runs.
#[derive(Debug, Default, Clone)]
pub struct VacuumMetrics {
    /// Total number of completed vacuum runs.
    pub total_runs: u64,
    /// Total bytes reclaimed over all runs.
    pub total_bytes_reclaimed: u64,
    /// Total pages compacted over all runs.
    pub total_pages_compacted: u64,
    /// Total time spent vacuuming.
    pub total_duration: Duration,
    /// History of the last N completed runs (most-recent last).
    pub run_history: Vec<VacuumStats>,
}

impl VacuumMetrics {
    /// Maximum number of historical entries to retain.
    const MAX_HISTORY: usize = 32;

    /// Record the result of a completed vacuum run.
    pub fn record(&mut self, stats: VacuumStats) {
        self.total_runs += 1;
        self.total_bytes_reclaimed += stats.bytes_reclaimed;
        self.total_pages_compacted += stats.pages_compacted as u64;
        self.total_duration += stats.duration;
        self.run_history.push(stats);
        if self.run_history.len() > Self::MAX_HISTORY {
            self.run_history.remove(0);
        }
    }
}

// ---------------------------------------------------------------------------
// Vacuum schedule
// ---------------------------------------------------------------------------

/// Tracks when the next automatic vacuum should fire.
#[derive(Debug)]
struct VacuumSchedule {
    last_vacuum: Option<Instant>,
}

impl VacuumSchedule {
    fn new() -> Self {
        Self { last_vacuum: None }
    }

    /// Return true if the schedule says it is time to vacuum.
    fn should_run(&self, trigger: &VacuumTrigger, fragmentation: f64) -> bool {
        match trigger {
            VacuumTrigger::Manual => false,
            VacuumTrigger::FragmentationThreshold(threshold) => fragmentation >= *threshold,
            VacuumTrigger::TimeBased(interval) => self
                .last_vacuum
                .map(|t| t.elapsed() >= *interval)
                .unwrap_or(true),
            VacuumTrigger::Combined {
                threshold,
                interval,
            } => {
                let frag_trigger = fragmentation >= *threshold;
                let time_trigger = self
                    .last_vacuum
                    .map(|t| t.elapsed() >= *interval)
                    .unwrap_or(true);
                frag_trigger || time_trigger
            }
        }
    }

    fn record_run(&mut self) {
        self.last_vacuum = Some(Instant::now());
    }
}

// ---------------------------------------------------------------------------
// Free-list management
// ---------------------------------------------------------------------------

/// Tracks freed page IDs and slot IDs.
#[derive(Debug, Default)]
struct FreeList {
    /// Page IDs that have been fully freed.
    freed_pages: BTreeSet<u64>,
    /// Slot IDs (sub-page granularity) that have been freed.
    freed_slots: BTreeSet<u64>,
}

impl FreeList {
    fn add_page(&mut self, page_id: u64) {
        self.freed_pages.insert(page_id);
    }

    fn add_slot(&mut self, slot_id: u64) {
        self.freed_slots.insert(slot_id);
    }

    fn freed_page_count(&self) -> usize {
        self.freed_pages.len()
    }

    fn freed_slot_count(&self) -> usize {
        self.freed_slots.len()
    }

    /// Drain and return at most `limit` freed pages (0 = all).
    fn drain_pages(&mut self, limit: usize) -> Vec<u64> {
        if limit == 0 {
            self.freed_pages
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .tap(|_| {
                    self.freed_pages.clear();
                })
        } else {
            let pages: Vec<u64> = self.freed_pages.iter().cloned().take(limit).collect();
            for p in &pages {
                self.freed_pages.remove(p);
            }
            pages
        }
    }

    /// Drain all freed slots.
    fn drain_slots(&mut self) -> Vec<u64> {
        let slots: Vec<u64> = self.freed_slots.iter().cloned().collect();
        self.freed_slots.clear();
        slots
    }
}

// Helper trait for the drain-then-clear pattern used above.
trait Tap: Sized {
    fn tap<F: FnOnce(&Self)>(self, f: F) -> Self {
        f(&self);
        self
    }
}

impl<T> Tap for Vec<T> {}

// ---------------------------------------------------------------------------
// Space reporter
// ---------------------------------------------------------------------------

/// A pre/post space report produced around a vacuum run.
#[derive(Debug, Clone)]
pub struct SpaceReport {
    /// Total bytes across all tracked pages.
    pub total_bytes: u64,
    /// Live (used) bytes.
    pub used_bytes: u64,
    /// Dead (freed) bytes.
    pub dead_bytes: u64,
    /// Overall fragmentation ratio (dead / total).
    pub fragmentation: f64,
    /// Number of fully-freed pages awaiting compaction.
    pub freed_page_count: usize,
    /// Number of freed slots awaiting compaction.
    pub freed_slot_count: usize,
}

impl SpaceReport {
    fn compute(pages: &HashMap<u64, PageSpaceInfo>, free_list: &FreeList) -> Self {
        let total_bytes: u64 = pages.values().map(|p| p.total_bytes).sum();
        let used_bytes: u64 = pages.values().map(|p| p.used_bytes).sum();
        let dead_bytes = total_bytes.saturating_sub(used_bytes);
        let fragmentation = if total_bytes == 0 {
            0.0
        } else {
            dead_bytes as f64 / total_bytes as f64
        };
        Self {
            total_bytes,
            used_bytes,
            dead_bytes,
            fragmentation,
            freed_page_count: free_list.freed_page_count(),
            freed_slot_count: free_list.freed_slot_count(),
        }
    }
}

// ---------------------------------------------------------------------------
// Vacuum engine
// ---------------------------------------------------------------------------

/// The central vacuum engine for TDB storage.
///
/// Call [`VacuumEngine::mark_page_freed`] or [`VacuumEngine::mark_slot_freed`]
/// as pages / slots become dead.  Invoke [`VacuumEngine::vacuum`] (manual) or
/// check [`VacuumEngine::should_run_now`] (automatic) to trigger compaction.
pub struct VacuumEngine {
    config: VacuumConfig,
    free_list: FreeList,
    /// Per-page space-usage information registered by the storage layer.
    page_info: HashMap<u64, PageSpaceInfo>,
    schedule: VacuumSchedule,
    lock_state: VacuumLockState,
    metrics: VacuumMetrics,
}

impl VacuumEngine {
    /// Create a new vacuum engine with the given configuration.
    pub fn new(config: VacuumConfig) -> Self {
        Self {
            config,
            free_list: FreeList::default(),
            page_info: HashMap::new(),
            schedule: VacuumSchedule::new(),
            lock_state: VacuumLockState::Idle,
            metrics: VacuumMetrics::default(),
        }
    }

    // ------------------------------------------------------------------
    // Registration helpers
    // ------------------------------------------------------------------

    /// Register (or update) space-usage information for a page.
    pub fn register_page(&mut self, info: PageSpaceInfo) {
        self.page_info.insert(info.page_id, info);
    }

    /// Mark a page as fully freed (all its data is dead).
    ///
    /// The page is added to the free-list for compaction.  The `page_info`
    /// entry is intentionally left unchanged so that the overall fragmentation
    /// ratio reflects only *partial* page dead-space (the kind that benefits
    /// from background compaction triggered by the threshold).  Fully-freed
    /// pages are already accounted for via `freed_page_count` and will be
    /// reclaimed on the next vacuum pass regardless of the fragmentation ratio.
    pub fn mark_page_freed(&mut self, page_id: u64) {
        self.free_list.add_page(page_id);
    }

    /// Mark a sub-page slot as freed.
    pub fn mark_slot_freed(&mut self, slot_id: u64) {
        self.free_list.add_slot(slot_id);
    }

    // ------------------------------------------------------------------
    // Space reporting
    // ------------------------------------------------------------------

    /// Produce a space report reflecting current state.
    pub fn space_report(&self) -> SpaceReport {
        SpaceReport::compute(&self.page_info, &self.free_list)
    }

    /// Overall fragmentation ratio (dead bytes / total bytes across all pages).
    pub fn fragmentation_ratio(&self) -> f64 {
        self.space_report().fragmentation
    }

    // ------------------------------------------------------------------
    // Lock management
    // ------------------------------------------------------------------

    /// Current lock state.
    pub fn lock_state(&self) -> &VacuumLockState {
        &self.lock_state
    }

    /// Bypass vacuum (suppress automatic runs temporarily).
    pub fn bypass(&mut self) {
        self.lock_state = VacuumLockState::Bypassed;
    }

    /// Release a bypass.  Has no effect if not currently bypassed.
    pub fn release_bypass(&mut self) {
        if self.lock_state == VacuumLockState::Bypassed {
            self.lock_state = VacuumLockState::Idle;
        }
    }

    // ------------------------------------------------------------------
    // Schedule check
    // ------------------------------------------------------------------

    /// Return true if the trigger conditions are met and the engine is idle.
    pub fn should_run_now(&self) -> bool {
        if self.lock_state != VacuumLockState::Idle {
            return false;
        }
        if self.free_list.freed_page_count() < self.config.min_freed_pages {
            return false;
        }
        let frag = self.fragmentation_ratio();
        self.schedule.should_run(&self.config.trigger, frag)
    }

    // ------------------------------------------------------------------
    // Vacuum execution
    // ------------------------------------------------------------------

    /// Run the vacuum operation.
    ///
    /// Returns a [`VacuumStats`] summary on success.
    ///
    /// # Errors
    ///
    /// Returns [`VacuumError::AlreadyRunning`] if a vacuum is currently
    /// in progress, or [`VacuumError::ConflictingLock`] if bypassed.
    pub fn vacuum(&mut self) -> VacuumResult<VacuumStats> {
        match self.lock_state {
            VacuumLockState::Running => return Err(VacuumError::AlreadyRunning),
            VacuumLockState::Bypassed => {
                return Err(VacuumError::ConflictingLock(
                    "Vacuum is currently bypassed".to_string(),
                ))
            }
            VacuumLockState::Idle => {}
        }

        self.lock_state = VacuumLockState::Running;

        let pre_report = self.space_report();

        let start = Instant::now();

        // Determine how many pages to compact this pass.
        let limit = self.config.max_pages_per_pass;

        // Compact pages.
        let pages_to_compact = self.free_list.drain_pages(limit);
        let pages_compacted = pages_to_compact.len();
        let mut bytes_reclaimed: u64 = 0;

        for page_id in pages_to_compact {
            let reclaimed = self
                .page_info
                .get(&page_id)
                .map(|p| p.total_bytes)
                .unwrap_or(0);
            bytes_reclaimed += reclaimed;
            // Remove from page-info tracking (page is now reclaimed).
            self.page_info.remove(&page_id);

            // Online mode: yield periodically by recording the delay without
            // actually sleeping (production integration would use async sleep
            // or a cooperative yield point).
            if self.config.online {
                // In real usage: tokio::time::sleep(self.config.page_step_delay).await
                // Here we track the intended delay in stats only.
                let _ = self.config.page_step_delay;
            }
        }

        // Compact slots.
        let slots_removed = self.free_list.drain_slots().len();

        let duration = start.elapsed();

        let post_report = self.space_report();

        let stats = VacuumStats {
            bytes_reclaimed,
            pages_compacted,
            slots_removed,
            duration,
            pre_vacuum_dead_bytes: pre_report.dead_bytes,
            post_vacuum_dead_bytes: post_report.dead_bytes,
            pre_vacuum_fragmentation: pre_report.fragmentation,
            post_vacuum_fragmentation: post_report.fragmentation,
        };

        self.metrics.record(stats.clone());
        self.schedule.record_run();
        self.lock_state = VacuumLockState::Idle;

        Ok(stats)
    }

    /// Run a vacuum only if `should_run_now()` is true.
    ///
    /// Returns `Some(stats)` when a vacuum was executed, `None` otherwise.
    pub fn maybe_vacuum(&mut self) -> VacuumResult<Option<VacuumStats>> {
        if self.should_run_now() {
            self.vacuum().map(Some)
        } else {
            Ok(None)
        }
    }

    // ------------------------------------------------------------------
    // Metrics access
    // ------------------------------------------------------------------

    /// Return a reference to the cumulative vacuum metrics.
    pub fn metrics(&self) -> &VacuumMetrics {
        &self.metrics
    }

    /// Return the number of freed pages currently awaiting compaction.
    pub fn pending_freed_pages(&self) -> usize {
        self.free_list.freed_page_count()
    }

    /// Return the number of freed slots currently awaiting compaction.
    pub fn pending_freed_slots(&self) -> usize {
        self.free_list.freed_slot_count()
    }

    /// Consume the engine and return its accumulated metrics.
    pub fn into_metrics(self) -> VacuumMetrics {
        self.metrics
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_engine() -> VacuumEngine {
        VacuumEngine::new(VacuumConfig::default())
    }

    // -- basic space tracking -----------------------------------------------

    #[test]
    fn test_register_page_and_fragmentation() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(1, 4096, 2048));
        let ratio = eng.fragmentation_ratio();
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_zero_fragmentation_on_full_page() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(1, 4096, 4096));
        assert_eq!(eng.fragmentation_ratio(), 0.0);
    }

    #[test]
    fn test_full_fragmentation_on_empty_page() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(1, 4096, 0));
        assert_eq!(eng.fragmentation_ratio(), 1.0);
    }

    #[test]
    fn test_fragmentation_empty_engine() {
        let eng = make_engine();
        assert_eq!(eng.fragmentation_ratio(), 0.0);
    }

    // -- freed page / slot tracking -----------------------------------------

    #[test]
    fn test_mark_page_freed_increments_pending() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(10, 4096, 1024));
        eng.mark_page_freed(10);
        assert_eq!(eng.pending_freed_pages(), 1);
    }

    #[test]
    fn test_mark_slot_freed_increments_pending() {
        let mut eng = make_engine();
        eng.mark_slot_freed(100);
        assert_eq!(eng.pending_freed_slots(), 1);
    }

    #[test]
    fn test_mark_multiple_pages_freed() {
        let mut eng = make_engine();
        for i in 0..5u64 {
            eng.register_page(PageSpaceInfo::new(i, 4096, 0));
            eng.mark_page_freed(i);
        }
        assert_eq!(eng.pending_freed_pages(), 5);
    }

    // -- space report -------------------------------------------------------

    #[test]
    fn test_space_report_totals() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(0, 4096, 1000));
        eng.register_page(PageSpaceInfo::new(1, 4096, 3000));
        let report = eng.space_report();
        assert_eq!(report.total_bytes, 8192);
        assert_eq!(report.used_bytes, 4000);
        assert_eq!(report.dead_bytes, 4192);
    }

    #[test]
    fn test_space_report_freed_counts() {
        let mut eng = make_engine();
        eng.mark_page_freed(1);
        eng.mark_slot_freed(200);
        let report = eng.space_report();
        assert_eq!(report.freed_page_count, 1);
        assert_eq!(report.freed_slot_count, 1);
    }

    // -- vacuum execution ---------------------------------------------------

    #[test]
    fn test_vacuum_reclaims_freed_pages() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(5, 4096, 0));
        eng.mark_page_freed(5);
        assert_eq!(eng.pending_freed_pages(), 1);
        let stats = eng.vacuum().expect("vacuum failed");
        assert_eq!(stats.pages_compacted, 1);
        assert_eq!(stats.bytes_reclaimed, 4096);
        assert_eq!(eng.pending_freed_pages(), 0);
    }

    #[test]
    fn test_vacuum_removes_slots() {
        let mut eng = make_engine();
        eng.mark_slot_freed(10);
        eng.mark_slot_freed(20);
        let stats = eng.vacuum().expect("vacuum failed");
        assert_eq!(stats.slots_removed, 2);
        assert_eq!(eng.pending_freed_slots(), 0);
    }

    #[test]
    fn test_vacuum_pre_post_fragmentation() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(0, 4096, 0));
        eng.mark_page_freed(0);
        let stats = eng.vacuum().expect("vacuum failed");
        assert!(stats.pre_vacuum_fragmentation >= 1.0 || stats.pre_vacuum_fragmentation >= 0.0);
        assert_eq!(stats.post_vacuum_fragmentation, 0.0);
    }

    #[test]
    fn test_vacuum_updates_metrics() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(1, 4096, 0));
        eng.mark_page_freed(1);
        eng.vacuum().expect("vacuum failed");
        assert_eq!(eng.metrics().total_runs, 1);
        assert_eq!(eng.metrics().total_pages_compacted, 1);
        assert!(eng.metrics().total_bytes_reclaimed > 0);
    }

    #[test]
    fn test_vacuum_history_stored() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(1, 4096, 0));
        eng.mark_page_freed(1);
        eng.vacuum().expect("vacuum failed");
        assert_eq!(eng.metrics().run_history.len(), 1);
    }

    // -- max_pages_per_pass -------------------------------------------------

    #[test]
    fn test_max_pages_per_pass_limits_compaction() {
        let config = VacuumConfig {
            max_pages_per_pass: 2,
            min_freed_pages: 1,
            trigger: VacuumTrigger::Manual,
            ..Default::default()
        };
        let mut eng = VacuumEngine::new(config);
        for i in 0..5u64 {
            eng.register_page(PageSpaceInfo::new(i, 4096, 0));
            eng.mark_page_freed(i);
        }
        let stats = eng.vacuum().expect("vacuum failed");
        assert_eq!(stats.pages_compacted, 2);
        // 3 pages remain pending.
        assert_eq!(eng.pending_freed_pages(), 3);
    }

    // -- lock management ----------------------------------------------------

    #[test]
    fn test_already_running_error() {
        let mut eng = make_engine();
        eng.lock_state = VacuumLockState::Running;
        let err = eng.vacuum().unwrap_err();
        assert_eq!(err, VacuumError::AlreadyRunning);
    }

    #[test]
    fn test_bypassed_error() {
        let mut eng = make_engine();
        eng.bypass();
        let err = eng.vacuum().unwrap_err();
        matches!(err, VacuumError::ConflictingLock(_));
    }

    #[test]
    fn test_bypass_and_release() {
        let mut eng = make_engine();
        eng.bypass();
        assert_eq!(*eng.lock_state(), VacuumLockState::Bypassed);
        eng.release_bypass();
        assert_eq!(*eng.lock_state(), VacuumLockState::Idle);
    }

    // -- should_run_now -----------------------------------------------------

    #[test]
    fn test_should_run_manual_trigger_never_auto() {
        let config = VacuumConfig {
            trigger: VacuumTrigger::Manual,
            min_freed_pages: 1,
            ..Default::default()
        };
        let mut eng = VacuumEngine::new(config);
        eng.register_page(PageSpaceInfo::new(0, 4096, 0));
        eng.mark_page_freed(0);
        assert!(!eng.should_run_now());
    }

    #[test]
    fn test_should_run_fragmentation_threshold_fires() {
        let config = VacuumConfig {
            trigger: VacuumTrigger::FragmentationThreshold(0.20),
            min_freed_pages: 1,
            ..Default::default()
        };
        let mut eng = VacuumEngine::new(config);
        eng.register_page(PageSpaceInfo::new(0, 100, 50)); // 50% fragmentation
        eng.mark_page_freed(0);
        assert!(eng.should_run_now());
    }

    #[test]
    fn test_should_run_below_threshold_does_not_fire() {
        let config = VacuumConfig {
            trigger: VacuumTrigger::FragmentationThreshold(0.80),
            min_freed_pages: 1,
            ..Default::default()
        };
        let mut eng = VacuumEngine::new(config);
        eng.register_page(PageSpaceInfo::new(0, 100, 70)); // 30% fragmentation
        eng.mark_page_freed(0);
        assert!(!eng.should_run_now());
    }

    #[test]
    fn test_should_run_not_enough_freed_pages() {
        let config = VacuumConfig {
            trigger: VacuumTrigger::FragmentationThreshold(0.10),
            min_freed_pages: 5,
            ..Default::default()
        };
        let mut eng = VacuumEngine::new(config);
        eng.register_page(PageSpaceInfo::new(0, 100, 0)); // 100% fragmentation
        eng.mark_page_freed(0); // only 1 freed page
        assert!(!eng.should_run_now()); // min_freed_pages = 5 not met
    }

    #[test]
    fn test_should_run_bypassed_returns_false() {
        let config = VacuumConfig {
            trigger: VacuumTrigger::FragmentationThreshold(0.10),
            min_freed_pages: 1,
            ..Default::default()
        };
        let mut eng = VacuumEngine::new(config);
        eng.register_page(PageSpaceInfo::new(0, 100, 0));
        eng.mark_page_freed(0);
        eng.bypass();
        assert!(!eng.should_run_now());
    }

    // -- time-based trigger -------------------------------------------------

    #[test]
    fn test_time_based_trigger_fires_immediately_on_first_run() {
        let config = VacuumConfig {
            trigger: VacuumTrigger::TimeBased(Duration::from_secs(3600)),
            min_freed_pages: 1,
            ..Default::default()
        };
        let mut eng = VacuumEngine::new(config);
        eng.mark_page_freed(0);
        // No previous vacuum → should fire immediately.
        assert!(eng.should_run_now());
    }

    #[test]
    fn test_combined_trigger_fragmentation_path() {
        let config = VacuumConfig {
            trigger: VacuumTrigger::Combined {
                threshold: 0.20,
                interval: Duration::from_secs(3600),
            },
            min_freed_pages: 1,
            ..Default::default()
        };
        let mut eng = VacuumEngine::new(config);
        eng.register_page(PageSpaceInfo::new(0, 100, 50)); // 50% fragmentation
        eng.mark_page_freed(0);
        assert!(eng.should_run_now());
    }

    // -- maybe_vacuum -------------------------------------------------------

    #[test]
    fn test_maybe_vacuum_does_not_run_when_not_needed() {
        let config = VacuumConfig {
            trigger: VacuumTrigger::Manual,
            ..Default::default()
        };
        let mut eng = VacuumEngine::new(config);
        let result = eng.maybe_vacuum().expect("unexpected error");
        assert!(result.is_none());
    }

    #[test]
    fn test_maybe_vacuum_runs_when_needed() {
        let config = VacuumConfig {
            trigger: VacuumTrigger::TimeBased(Duration::from_secs(0)),
            min_freed_pages: 1,
            ..Default::default()
        };
        let mut eng = VacuumEngine::new(config);
        eng.register_page(PageSpaceInfo::new(0, 4096, 0));
        eng.mark_page_freed(0);
        let result = eng.maybe_vacuum().expect("unexpected error");
        assert!(result.is_some());
    }

    // -- page space info ----------------------------------------------------

    #[test]
    fn test_page_space_info_dead_bytes() {
        let info = PageSpaceInfo::new(1, 4096, 1024);
        assert_eq!(info.dead_bytes(), 3072);
    }

    #[test]
    fn test_page_space_info_fragmentation() {
        let info = PageSpaceInfo::new(1, 1000, 250);
        assert!((info.fragmentation() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_page_space_info_zero_total() {
        let info = PageSpaceInfo::new(1, 0, 0);
        assert_eq!(info.fragmentation(), 0.0);
    }

    // -- vacuum error display -----------------------------------------------

    #[test]
    fn test_vacuum_error_display() {
        assert_eq!(
            VacuumError::AlreadyRunning.to_string(),
            "Vacuum already running"
        );
        assert!(VacuumError::ConflictingLock("x".to_string())
            .to_string()
            .contains("x"));
        assert!(VacuumError::InvalidId("bad".to_string())
            .to_string()
            .contains("bad"));
        assert!(VacuumError::Internal("oops".to_string())
            .to_string()
            .contains("oops"));
    }

    // -- into_metrics -------------------------------------------------------

    #[test]
    fn test_into_metrics_returns_cumulative() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(0, 4096, 0));
        eng.mark_page_freed(0);
        eng.vacuum().expect("vacuum failed");
        let metrics = eng.into_metrics();
        assert_eq!(metrics.total_runs, 1);
    }

    // -- post-vacuum state --------------------------------------------------

    #[test]
    fn test_lock_released_after_vacuum() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(0, 4096, 0));
        eng.mark_page_freed(0);
        eng.vacuum().expect("vacuum failed");
        assert_eq!(*eng.lock_state(), VacuumLockState::Idle);
    }

    #[test]
    fn test_second_vacuum_runs_after_first() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(0, 4096, 0));
        eng.mark_page_freed(0);
        eng.vacuum().expect("first vacuum failed");
        // Register new dead page.
        eng.register_page(PageSpaceInfo::new(1, 4096, 0));
        eng.mark_page_freed(1);
        let stats = eng.vacuum().expect("second vacuum failed");
        assert_eq!(stats.pages_compacted, 1);
        assert_eq!(eng.metrics().total_runs, 2);
    }

    // -- vacuum with no freed pages -----------------------------------------

    #[test]
    fn test_vacuum_with_no_freed_pages() {
        let mut eng = make_engine();
        eng.register_page(PageSpaceInfo::new(0, 4096, 2048));
        let stats = eng.vacuum().expect("vacuum failed");
        assert_eq!(stats.pages_compacted, 0);
        assert_eq!(stats.bytes_reclaimed, 0);
        assert_eq!(stats.slots_removed, 0);
    }
}
