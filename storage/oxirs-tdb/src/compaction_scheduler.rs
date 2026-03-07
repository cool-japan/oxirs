//! Background compaction task scheduler for OxiRS TDB.
//!
//! Manages a queue of `CompactionTask`s triggered by file-size thresholds,
//! file-count thresholds, periodic timers, or manual requests.  Completed and
//! failed tasks are kept in a bounded history for observability.

use std::collections::VecDeque;

// ── Trigger ───────────────────────────────────────────────────────────────────

/// Reason a compaction task was created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionTrigger {
    /// A single file exceeded the given byte size.
    FileSizeThreshold(usize),
    /// The total number of SST files exceeded the given count.
    FileCountThreshold(usize),
    /// Periodic compaction after the given interval (milliseconds).
    Periodic(u64),
    /// Triggered explicitly by the operator.
    Manual,
}

// ── Task ──────────────────────────────────────────────────────────────────────

/// A single compaction task submitted to the scheduler.
#[derive(Debug, Clone)]
pub struct CompactionTask {
    /// Unique monotonically-increasing task identifier.
    pub id: u64,
    /// What caused this task to be scheduled.
    pub trigger: CompactionTrigger,
    /// File paths (SST segments) to compact.
    pub files: Vec<String>,
    /// Unix timestamp (milliseconds) when the task was created.
    pub created_at: u64,
}

// ── Status ────────────────────────────────────────────────────────────────────

/// Lifecycle state of a compaction task after it leaves the queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    /// Waiting in the queue.
    Pending,
    /// Currently executing.
    Running,
    /// Finished successfully; value is the completion timestamp (ms).
    Completed(u64),
    /// Failed; value is the error message.
    Failed(String),
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

/// Manages a queue of pending [`CompactionTask`]s and a bounded history of
/// completed / failed tasks.
pub struct CompactionScheduler {
    tasks: VecDeque<CompactionTask>,
    history: Vec<(CompactionTask, TaskStatus)>,
    next_id: u64,
    max_history: usize,
}

impl CompactionScheduler {
    /// Create a new scheduler with the given history capacity.
    pub fn new(max_history: usize) -> Self {
        Self {
            tasks: VecDeque::new(),
            history: Vec::new(),
            next_id: 1,
            max_history,
        }
    }

    /// Schedule a new compaction task.
    ///
    /// Returns the assigned task ID.
    pub fn schedule(&mut self, trigger: CompactionTrigger, files: Vec<String>, now: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.tasks.push_back(CompactionTask {
            id,
            trigger,
            files,
            created_at: now,
        });
        id
    }

    /// Decide whether compaction should be triggered right now.
    ///
    /// Returns `true` when *any* of the following hold:
    /// - At least one file in `file_sizes` exceeds `FILE_SIZE_THRESHOLD` (16 MiB default).
    /// - `file_count` exceeds `FILE_COUNT_THRESHOLD` (10 default).
    /// - More than `period` milliseconds have elapsed since the last compaction.
    ///
    /// The threshold constants are tunable by the caller through the
    /// `file_sizes` / `file_count` parameters; period-based checking uses
    /// [`Self::last_compaction_time`].
    pub fn should_compact(
        &self,
        file_sizes: &[usize],
        file_count: usize,
        now: u64,
        period: u64,
    ) -> bool {
        // File-size trigger: any file larger than 16 MiB.
        const FILE_SIZE_THRESHOLD: usize = 16 * 1024 * 1024;
        if file_sizes.iter().any(|&s| s >= FILE_SIZE_THRESHOLD) {
            return true;
        }
        // File-count trigger.
        const FILE_COUNT_THRESHOLD: usize = 10;
        if file_count >= FILE_COUNT_THRESHOLD {
            return true;
        }
        // Periodic trigger.
        if period > 0 {
            if let Some(last) = self.last_compaction_time() {
                if now.saturating_sub(last) >= period {
                    return true;
                }
            }
        }
        false
    }

    /// Dequeue and return the next pending task (FIFO).
    ///
    /// Returns `None` when the queue is empty.
    pub fn next_pending(&mut self) -> Option<CompactionTask> {
        self.tasks.pop_front()
    }

    /// Mark a pending task as successfully completed at time `now`.
    pub fn complete(&mut self, id: u64, now: u64) {
        // Find in queue first (task might still be pending).
        if let Some(pos) = self.tasks.iter().position(|t| t.id == id) {
            let task = self.tasks.remove(pos).expect("just found it");
            self.push_history(task, TaskStatus::Completed(now));
            return;
        }
        // Not in queue — task was already dequeued; record a synthetic entry.
        let task = CompactionTask {
            id,
            trigger: CompactionTrigger::Manual,
            files: Vec::new(),
            created_at: 0,
        };
        self.push_history(task, TaskStatus::Completed(now));
    }

    /// Mark a pending task as failed with the given reason.
    pub fn fail(&mut self, id: u64, reason: String) {
        if let Some(pos) = self.tasks.iter().position(|t| t.id == id) {
            let task = self.tasks.remove(pos).expect("just found it");
            self.push_history(task, TaskStatus::Failed(reason));
        } else {
            let task = CompactionTask {
                id,
                trigger: CompactionTrigger::Manual,
                files: Vec::new(),
                created_at: 0,
            };
            self.push_history(task, TaskStatus::Failed(reason));
        }
    }

    /// Number of tasks currently waiting in the queue.
    pub fn pending_count(&self) -> usize {
        self.tasks.len()
    }

    /// Number of entries in the history (completed + failed).
    pub fn history_count(&self) -> usize {
        self.history.len()
    }

    /// Timestamp of the most recent successfully completed compaction, if any.
    pub fn last_compaction_time(&self) -> Option<u64> {
        self.history.iter().rev().find_map(|(_, status)| {
            if let TaskStatus::Completed(ts) = status {
                Some(*ts)
            } else {
                None
            }
        })
    }

    // Push to history and trim if over capacity.
    fn push_history(&mut self, task: CompactionTask, status: TaskStatus) {
        if self.max_history > 0 {
            if self.history.len() >= self.max_history {
                self.history.remove(0);
            }
            self.history.push((task, status));
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sched(max: usize) -> CompactionScheduler {
        CompactionScheduler::new(max)
    }

    // ── schedule / pending_count ───────────────────────────────────────────────

    #[test]
    fn test_new_empty_queue() {
        let s = sched(10);
        assert_eq!(s.pending_count(), 0);
    }

    #[test]
    fn test_schedule_returns_id() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 1000);
        assert_eq!(id, 1);
    }

    #[test]
    fn test_schedule_increments_id() {
        let mut s = sched(10);
        let id1 = s.schedule(CompactionTrigger::Manual, vec![], 1000);
        let id2 = s.schedule(CompactionTrigger::Manual, vec![], 1001);
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn test_schedule_increments_pending() {
        let mut s = sched(10);
        s.schedule(CompactionTrigger::Manual, vec![], 1000);
        assert_eq!(s.pending_count(), 1);
    }

    #[test]
    fn test_schedule_multiple_pending() {
        let mut s = sched(10);
        for _ in 0..5 {
            s.schedule(CompactionTrigger::Manual, vec![], 1000);
        }
        assert_eq!(s.pending_count(), 5);
    }

    // ── next_pending ──────────────────────────────────────────────────────────

    #[test]
    fn test_next_pending_empty_returns_none() {
        let mut s = sched(10);
        assert!(s.next_pending().is_none());
    }

    #[test]
    fn test_next_pending_returns_task() {
        let mut s = sched(10);
        s.schedule(CompactionTrigger::Manual, vec!["f1".into()], 1000);
        let t = s.next_pending().expect("should have a task");
        assert_eq!(t.id, 1);
        assert_eq!(t.files, vec!["f1".to_string()]);
    }

    #[test]
    fn test_next_pending_fifo_order() {
        let mut s = sched(10);
        s.schedule(CompactionTrigger::Manual, vec!["a".into()], 1000);
        s.schedule(CompactionTrigger::Manual, vec!["b".into()], 1001);
        let t1 = s.next_pending().expect("first");
        let t2 = s.next_pending().expect("second");
        assert_eq!(t1.id, 1);
        assert_eq!(t2.id, 2);
    }

    #[test]
    fn test_next_pending_decrements_count() {
        let mut s = sched(10);
        s.schedule(CompactionTrigger::Manual, vec![], 1000);
        s.next_pending();
        assert_eq!(s.pending_count(), 0);
    }

    // ── complete / history ────────────────────────────────────────────────────

    #[test]
    fn test_complete_removes_from_queue() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 1000);
        s.complete(id, 2000);
        assert_eq!(s.pending_count(), 0);
    }

    #[test]
    fn test_complete_adds_to_history() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 1000);
        s.complete(id, 2000);
        assert_eq!(s.history_count(), 1);
    }

    #[test]
    fn test_last_compaction_time_after_complete() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 1000);
        s.complete(id, 5000);
        assert_eq!(s.last_compaction_time(), Some(5000));
    }

    #[test]
    fn test_last_compaction_time_none_if_no_history() {
        let s = sched(10);
        assert_eq!(s.last_compaction_time(), None);
    }

    #[test]
    fn test_last_compaction_time_most_recent() {
        let mut s = sched(10);
        let id1 = s.schedule(CompactionTrigger::Manual, vec![], 1000);
        let id2 = s.schedule(CompactionTrigger::Manual, vec![], 1001);
        s.complete(id1, 2000);
        s.complete(id2, 3000);
        assert_eq!(s.last_compaction_time(), Some(3000));
    }

    // ── fail ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_fail_removes_from_queue() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 1000);
        s.fail(id, "disk full".into());
        assert_eq!(s.pending_count(), 0);
    }

    #[test]
    fn test_fail_adds_to_history() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 1000);
        s.fail(id, "io error".into());
        assert_eq!(s.history_count(), 1);
    }

    #[test]
    fn test_fail_does_not_update_last_compaction_time() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 1000);
        s.fail(id, "error".into());
        assert_eq!(s.last_compaction_time(), None);
    }

    // ── history cap ───────────────────────────────────────────────────────────

    #[test]
    fn test_history_capped_at_max() {
        let mut s = sched(3);
        for _ in 0..5 {
            let id = s.schedule(CompactionTrigger::Manual, vec![], 1000);
            s.complete(id, 2000);
        }
        assert_eq!(s.history_count(), 3);
    }

    #[test]
    fn test_history_zero_max_stores_nothing() {
        let mut s = sched(0);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 1000);
        s.complete(id, 2000);
        assert_eq!(s.history_count(), 0);
    }

    // ── should_compact ────────────────────────────────────────────────────────

    #[test]
    fn test_should_compact_no_trigger_false() {
        let s = sched(10);
        assert!(!s.should_compact(&[100, 200], 2, 10_000, 0));
    }

    #[test]
    fn test_should_compact_file_size_trigger() {
        let s = sched(10);
        let big = 20 * 1024 * 1024; // 20 MiB > 16 MiB threshold
        assert!(s.should_compact(&[big], 1, 10_000, 0));
    }

    #[test]
    fn test_should_compact_file_count_trigger() {
        let s = sched(10);
        assert!(s.should_compact(&[], 10, 10_000, 0)); // exactly at threshold
    }

    #[test]
    fn test_should_compact_periodic_trigger() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 0);
        s.complete(id, 1000); // last compaction at 1000 ms
                              // now=2001, period=1000 → 2001-1000=1001 >= 1000 → true
        assert!(s.should_compact(&[], 0, 2001, 1000));
    }

    #[test]
    fn test_should_compact_periodic_not_yet() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 0);
        s.complete(id, 1000); // last at 1000
                              // now=1500, period=1000 → 500 < 1000 → false
        assert!(!s.should_compact(&[], 0, 1500, 1000));
    }

    #[test]
    fn test_should_compact_file_below_threshold_false() {
        let s = sched(10);
        let small = 1024; // 1 KiB
        assert!(!s.should_compact(&[small, small], 2, 10_000, 0));
    }

    // ── trigger variants ──────────────────────────────────────────────────────

    #[test]
    fn test_trigger_file_size_threshold() {
        let mut s = sched(10);
        let id = s.schedule(
            CompactionTrigger::FileSizeThreshold(1024 * 1024),
            vec!["big.sst".into()],
            1000,
        );
        let t = s.next_pending().expect("task");
        assert!(matches!(t.trigger, CompactionTrigger::FileSizeThreshold(_)));
        let _ = id;
    }

    #[test]
    fn test_trigger_file_count_threshold() {
        let mut s = sched(10);
        s.schedule(CompactionTrigger::FileCountThreshold(8), vec![], 1000);
        let t = s.next_pending().expect("task");
        assert!(matches!(
            t.trigger,
            CompactionTrigger::FileCountThreshold(8)
        ));
    }

    #[test]
    fn test_trigger_periodic() {
        let mut s = sched(10);
        s.schedule(CompactionTrigger::Periodic(60_000), vec![], 1000);
        let t = s.next_pending().expect("task");
        assert!(matches!(t.trigger, CompactionTrigger::Periodic(60_000)));
    }

    #[test]
    fn test_created_at_preserved() {
        let mut s = sched(10);
        s.schedule(CompactionTrigger::Manual, vec![], 99999);
        let t = s.next_pending().expect("task");
        assert_eq!(t.created_at, 99999);
    }

    #[test]
    fn test_files_preserved_in_task() {
        let mut s = sched(10);
        s.schedule(
            CompactionTrigger::Manual,
            vec!["a.sst".into(), "b.sst".into()],
            1000,
        );
        let t = s.next_pending().expect("task");
        assert_eq!(t.files.len(), 2);
    }

    #[test]
    fn test_history_count_mixed_outcomes() {
        let mut s = sched(10);
        let id1 = s.schedule(CompactionTrigger::Manual, vec![], 1000);
        let id2 = s.schedule(CompactionTrigger::Manual, vec![], 1001);
        s.complete(id1, 2000);
        s.fail(id2, "error".into());
        assert_eq!(s.history_count(), 2);
    }

    // ── Additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_schedule_many_returns_sequential_ids() {
        let mut s = sched(100);
        let ids: Vec<u64> = (0..10)
            .map(|_| s.schedule(CompactionTrigger::Manual, vec![], 0))
            .collect();
        for i in 1..ids.len() {
            assert_eq!(ids[i], ids[i - 1] + 1);
        }
    }

    #[test]
    fn test_complete_unknown_id_adds_to_history() {
        let mut s = sched(10);
        s.complete(999, 1000);
        assert_eq!(s.history_count(), 1);
    }

    #[test]
    fn test_fail_unknown_id_adds_to_history() {
        let mut s = sched(10);
        s.fail(888, "unknown".into());
        assert_eq!(s.history_count(), 1);
    }

    #[test]
    fn test_last_compaction_none_only_failures() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 0);
        s.fail(id, "err".into());
        assert_eq!(s.last_compaction_time(), None);
    }

    #[test]
    fn test_should_compact_zero_files() {
        let s = sched(10);
        assert!(!s.should_compact(&[], 0, 100, 0));
    }

    #[test]
    fn test_should_compact_exactly_at_file_count() {
        let s = sched(10);
        assert!(s.should_compact(&[], 10, 0, 0));
    }

    #[test]
    fn test_trigger_manual_variant() {
        let mut s = sched(10);
        s.schedule(CompactionTrigger::Manual, vec![], 0);
        let t = s.next_pending().expect("task");
        assert_eq!(t.trigger, CompactionTrigger::Manual);
    }

    #[test]
    fn test_pending_count_after_mix() {
        let mut s = sched(10);
        s.schedule(CompactionTrigger::Manual, vec![], 0);
        s.schedule(CompactionTrigger::Manual, vec![], 1);
        s.next_pending();
        assert_eq!(s.pending_count(), 1);
    }

    #[test]
    fn test_history_max_one() {
        let mut s = sched(1);
        let id1 = s.schedule(CompactionTrigger::Manual, vec![], 0);
        let id2 = s.schedule(CompactionTrigger::Manual, vec![], 1);
        s.complete(id1, 100);
        s.complete(id2, 200);
        assert_eq!(s.history_count(), 1);
        // Last entry should be the most recent.
        assert_eq!(s.last_compaction_time(), Some(200));
    }

    #[test]
    fn test_should_compact_no_history_no_period() {
        let s = sched(10);
        assert!(!s.should_compact(&[1024], 2, 1000, 0));
    }

    #[test]
    fn test_task_id_field() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 42);
        let t = s.next_pending().expect("task");
        assert_eq!(t.id, id);
    }

    #[test]
    fn test_complete_decrements_pending() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 0);
        s.complete(id, 100);
        assert_eq!(s.pending_count(), 0);
    }

    #[test]
    fn test_fail_decrements_pending() {
        let mut s = sched(10);
        let id = s.schedule(CompactionTrigger::Manual, vec![], 0);
        s.fail(id, "fail".into());
        assert_eq!(s.pending_count(), 0);
    }

    #[test]
    fn test_multiple_history_entries_order() {
        let mut s = sched(10);
        let id1 = s.schedule(CompactionTrigger::Manual, vec![], 0);
        let id2 = s.schedule(CompactionTrigger::Manual, vec![], 1);
        s.complete(id1, 100);
        s.complete(id2, 200);
        // Last compaction time should reflect the most recently completed one.
        assert_eq!(s.last_compaction_time(), Some(200));
    }
}
