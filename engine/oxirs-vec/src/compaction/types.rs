//! Core types for compaction system

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// Compaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactionState {
    /// Idle, not compacting
    Idle,
    /// Currently compacting
    Running,
    /// Paused (can be resumed)
    Paused,
    /// Failed (error occurred)
    Failed,
    /// Completed successfully
    Completed,
}

/// Result of a compaction operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResult {
    /// Start time of compaction
    pub start_time: SystemTime,
    /// End time of compaction
    pub end_time: SystemTime,
    /// Duration of compaction
    pub duration: Duration,
    /// Number of vectors processed
    pub vectors_processed: usize,
    /// Number of vectors removed (deleted/duplicates)
    pub vectors_removed: usize,
    /// Bytes reclaimed
    pub bytes_reclaimed: u64,
    /// Fragmentation before compaction
    pub fragmentation_before: f64,
    /// Fragmentation after compaction
    pub fragmentation_after: f64,
    /// Success flag
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Compaction statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionStatistics {
    /// Total number of compactions performed
    pub total_compactions: u64,
    /// Successful compactions
    pub successful_compactions: u64,
    /// Failed compactions
    pub failed_compactions: u64,
    /// Total vectors processed
    pub total_vectors_processed: usize,
    /// Total vectors removed
    pub total_vectors_removed: usize,
    /// Total bytes reclaimed
    pub total_bytes_reclaimed: u64,
    /// Current fragmentation ratio (0.0 - 1.0)
    pub current_fragmentation: f64,
    /// Average compaction duration
    pub avg_compaction_duration: Duration,
    /// Last compaction time
    pub last_compaction_time: Option<SystemTime>,
    /// Last compaction result
    pub last_compaction_result: Option<CompactionResult>,
}

/// Fragment information
#[derive(Debug, Clone)]
pub struct FragmentInfo {
    /// Offset in the index
    pub offset: usize,
    /// Size of the fragment
    pub size: usize,
    /// Is this fragment free (deleted)?
    pub is_free: bool,
    /// Age of the fragment (time since creation)
    pub age: Duration,
}

/// Compaction phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionPhase {
    /// Analyzing fragmentation
    Analyzing,
    /// Identifying candidates for compaction
    IdentifyingCandidates,
    /// Moving vectors
    MovingVectors,
    /// Updating indices
    UpdatingIndices,
    /// Reclaiming space
    ReclaimingSpace,
    /// Verifying integrity
    Verifying,
    /// Completed
    Completed,
}

/// Compaction progress
#[derive(Debug, Clone)]
pub struct CompactionProgress {
    /// Current phase
    pub phase: CompactionPhase,
    /// Progress within current phase (0.0 - 1.0)
    pub phase_progress: f64,
    /// Overall progress (0.0 - 1.0)
    pub overall_progress: f64,
    /// Estimated time remaining
    pub estimated_time_remaining: Option<Duration>,
    /// Current throughput (vectors/sec)
    pub throughput: f64,
}

/// Compaction candidate (vector to be compacted)
#[derive(Debug, Clone)]
pub struct CompactionCandidate {
    /// Vector ID
    pub vector_id: String,
    /// Current offset
    pub current_offset: usize,
    /// Size in bytes
    pub size_bytes: usize,
    /// Priority (higher = more important to compact)
    pub priority: f64,
    /// Reason for compaction
    pub reason: CompactionReason,
}

/// Reason for compaction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactionReason {
    /// Fragmentation
    Fragmentation,
    /// Deleted vector cleanup
    DeletedCleanup,
    /// Duplicate removal
    DuplicateRemoval,
    /// Size optimization
    SizeOptimization,
    /// Manual trigger
    Manual,
}

/// Compaction batch
#[derive(Debug, Clone)]
pub struct CompactionBatch {
    /// Batch ID
    pub batch_id: u64,
    /// Candidates in this batch
    pub candidates: Vec<CompactionCandidate>,
    /// Total size of batch in bytes
    pub total_size_bytes: usize,
    /// Estimated processing time
    pub estimated_duration: Duration,
}
