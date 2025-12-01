//! Main compaction manager

use super::config::CompactionConfig;
use super::metrics::CompactionMetrics;
use super::strategies::StrategyEvaluator;
use super::types::{
    CompactionBatch, CompactionCandidate, CompactionPhase, CompactionProgress, CompactionReason,
    CompactionResult, CompactionState, CompactionStatistics, FragmentInfo,
};
use anyhow::Result;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Main compaction manager
pub struct CompactionManager {
    /// Configuration
    config: CompactionConfig,
    /// Metrics collector
    metrics: Arc<CompactionMetrics>,
    /// Strategy evaluator
    strategy: Arc<RwLock<StrategyEvaluator>>,
    /// Fragment map (vector_id -> fragment info)
    fragments: Arc<RwLock<HashMap<String, FragmentInfo>>>,
    /// Current progress
    progress: Arc<RwLock<Option<CompactionProgress>>>,
    /// Compaction enabled flag
    enabled: Arc<RwLock<bool>>,
}

impl CompactionManager {
    /// Create a new compaction manager
    pub fn new(config: CompactionConfig) -> Result<Self> {
        config.validate()?;

        let strategy = StrategyEvaluator::new(config.strategy);

        Ok(Self {
            config,
            metrics: Arc::new(CompactionMetrics::default()),
            strategy: Arc::new(RwLock::new(strategy)),
            fragments: Arc::new(RwLock::new(HashMap::new())),
            progress: Arc::new(RwLock::new(None)),
            enabled: Arc::new(RwLock::new(true)),
        })
    }

    /// Register a vector fragment
    pub fn register_fragment(&self, vector_id: String, offset: usize, size: usize) {
        let mut fragments = self.fragments.write();
        fragments.insert(
            vector_id,
            FragmentInfo {
                offset,
                size,
                is_free: false,
                age: Duration::ZERO,
            },
        );
    }

    /// Mark a vector as deleted (creates free fragment)
    pub fn mark_deleted(&self, vector_id: &str) -> Result<()> {
        let mut fragments = self.fragments.write();
        if let Some(fragment) = fragments.get_mut(vector_id) {
            fragment.is_free = true;
            Ok(())
        } else {
            anyhow::bail!("Vector {} not found", vector_id)
        }
    }

    /// Calculate current fragmentation ratio
    pub fn calculate_fragmentation(&self) -> f64 {
        let fragments = self.fragments.read();

        if fragments.is_empty() {
            return 0.0;
        }

        let total_size: usize = fragments.values().map(|f| f.size).sum();
        let free_size: usize = fragments
            .values()
            .filter(|f| f.is_free)
            .map(|f| f.size)
            .sum();

        if total_size == 0 {
            0.0
        } else {
            free_size as f64 / total_size as f64
        }
    }

    /// Check if compaction should be triggered
    pub fn should_compact(&self) -> bool {
        if !*self.enabled.read() {
            return false;
        }

        let fragmentation = self.calculate_fragmentation();
        let wasted_bytes = self.calculate_wasted_bytes();
        let time_since_last = self.strategy.read().time_since_last_compaction();

        let strategy = self.strategy.read();
        strategy.should_compact(
            fragmentation,
            wasted_bytes,
            time_since_last,
            self.config.compaction_interval,
            self.config.fragmentation_threshold,
            self.config.min_free_space_bytes,
        )
    }

    /// Calculate wasted bytes (from deleted vectors)
    fn calculate_wasted_bytes(&self) -> u64 {
        let fragments = self.fragments.read();
        fragments
            .values()
            .filter(|f| f.is_free)
            .map(|f| f.size as u64)
            .sum()
    }

    /// Trigger manual compaction
    pub fn compact_now(&self) -> Result<CompactionResult> {
        if !*self.enabled.read() {
            anyhow::bail!("Compaction is disabled");
        }

        self.perform_compaction()
    }

    /// Perform compaction
    fn perform_compaction(&self) -> Result<CompactionResult> {
        let start_time = SystemTime::now();

        // Update state
        self.metrics.update_state(CompactionState::Running);

        let fragmentation_before = self.calculate_fragmentation();

        // Phase 1: Analyze and identify candidates
        self.update_progress(CompactionPhase::IdentifyingCandidates, 0.0);
        let candidates = self.identify_candidates()?;

        if candidates.is_empty() {
            // Nothing to compact
            return Ok(CompactionResult {
                start_time,
                end_time: SystemTime::now(),
                duration: start_time.elapsed().unwrap_or(Duration::ZERO),
                vectors_processed: 0,
                vectors_removed: 0,
                bytes_reclaimed: 0,
                fragmentation_before,
                fragmentation_after: fragmentation_before,
                success: true,
                error: None,
            });
        }

        // Phase 2: Create batches
        let batches = self.create_batches(candidates);

        // Phase 3: Process batches
        let mut vectors_processed = 0;
        let mut vectors_removed = 0;
        let mut bytes_reclaimed = 0u64;

        for (i, batch) in batches.iter().enumerate() {
            self.update_progress(
                CompactionPhase::MovingVectors,
                i as f64 / batches.len() as f64,
            );

            let result = self.process_batch(batch)?;
            vectors_processed += result.0;
            vectors_removed += result.1;
            bytes_reclaimed += result.2;

            // Pause between batches
            std::thread::sleep(self.config.pause_between_batches);
        }

        // Phase 4: Reclaim space
        self.update_progress(CompactionPhase::ReclaimingSpace, 0.9);
        self.reclaim_space();

        // Phase 5: Verify (if enabled)
        if self.config.enable_verification {
            self.update_progress(CompactionPhase::Verifying, 0.95);
            self.verify_integrity()?;
        }

        let end_time = SystemTime::now();
        let duration = end_time
            .duration_since(start_time)
            .unwrap_or(Duration::ZERO);
        let fragmentation_after = self.calculate_fragmentation();

        // Update state
        self.metrics.update_state(CompactionState::Completed);
        self.update_progress(CompactionPhase::Completed, 1.0);

        // Record compaction
        self.strategy.write().record_compaction();

        let result = CompactionResult {
            start_time,
            end_time,
            duration,
            vectors_processed,
            vectors_removed,
            bytes_reclaimed,
            fragmentation_before,
            fragmentation_after,
            success: true,
            error: None,
        };

        self.metrics.record_compaction(result.clone());

        Ok(result)
    }

    /// Identify compaction candidates
    fn identify_candidates(&self) -> Result<Vec<CompactionCandidate>> {
        let fragments = self.fragments.read();
        let mut candidates = Vec::new();

        for (vector_id, fragment) in fragments.iter() {
            if fragment.is_free {
                candidates.push(CompactionCandidate {
                    vector_id: vector_id.clone(),
                    current_offset: fragment.offset,
                    size_bytes: fragment.size,
                    priority: 1.0, // Highest priority for deleted vectors
                    reason: CompactionReason::DeletedCleanup,
                });
            }
        }

        Ok(candidates)
    }

    /// Create batches from candidates
    fn create_batches(&self, mut candidates: Vec<CompactionCandidate>) -> Vec<CompactionBatch> {
        // Sort by priority (highest first)
        candidates.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_size = 0;

        for candidate in candidates {
            current_batch.push(candidate.clone());
            current_size += candidate.size_bytes;

            if current_batch.len() >= self.config.batch_size {
                batches.push(CompactionBatch {
                    batch_id: batches.len() as u64,
                    candidates: current_batch.clone(),
                    total_size_bytes: current_size,
                    estimated_duration: Duration::from_millis(100), // Estimate
                });
                current_batch.clear();
                current_size = 0;
            }
        }

        // Add remaining candidates
        if !current_batch.is_empty() {
            batches.push(CompactionBatch {
                batch_id: batches.len() as u64,
                candidates: current_batch,
                total_size_bytes: current_size,
                estimated_duration: Duration::from_millis(100),
            });
        }

        batches
    }

    /// Process a single batch
    fn process_batch(&self, batch: &CompactionBatch) -> Result<(usize, usize, u64)> {
        let mut vectors_processed = 0;
        let mut vectors_removed = 0;
        let mut bytes_reclaimed = 0u64;

        let mut fragments = self.fragments.write();

        for candidate in &batch.candidates {
            if let Some(_fragment) = fragments.get(&candidate.vector_id) {
                // Remove the fragment
                fragments.remove(&candidate.vector_id);
                vectors_processed += 1;
                vectors_removed += 1;
                bytes_reclaimed += candidate.size_bytes as u64;
            }
        }

        Ok((vectors_processed, vectors_removed, bytes_reclaimed))
    }

    /// Reclaim space (cleanup internal structures)
    fn reclaim_space(&self) {
        // In a real implementation, this would:
        // 1. Compact the underlying storage
        // 2. Update offsets
        // 3. Rebuild indices
        // For now, we just update fragmentation
        let fragmentation = self.calculate_fragmentation();
        self.metrics.update_fragmentation(fragmentation);
    }

    /// Verify integrity after compaction
    fn verify_integrity(&self) -> Result<()> {
        // In a real implementation, this would:
        // 1. Verify all vector IDs are accessible
        // 2. Check index consistency
        // 3. Validate checksums
        Ok(())
    }

    /// Update progress
    fn update_progress(&self, phase: CompactionPhase, progress: f64) {
        let mut prog = self.progress.write();
        *prog = Some(CompactionProgress {
            phase,
            phase_progress: progress,
            overall_progress: self.calculate_overall_progress(phase, progress),
            estimated_time_remaining: None,
            throughput: 0.0,
        });
    }

    /// Calculate overall progress
    fn calculate_overall_progress(&self, phase: CompactionPhase, phase_progress: f64) -> f64 {
        if matches!(phase, CompactionPhase::Completed) {
            return 1.0;
        }

        let phase_weight = match phase {
            CompactionPhase::Analyzing => 0.05,
            CompactionPhase::IdentifyingCandidates => 0.1,
            CompactionPhase::MovingVectors => 0.6,
            CompactionPhase::UpdatingIndices => 0.1,
            CompactionPhase::ReclaimingSpace => 0.1,
            CompactionPhase::Verifying => 0.05,
            CompactionPhase::Completed => 0.0,
        };

        let base_progress = match phase {
            CompactionPhase::Analyzing => 0.0,
            CompactionPhase::IdentifyingCandidates => 0.05,
            CompactionPhase::MovingVectors => 0.15,
            CompactionPhase::UpdatingIndices => 0.75,
            CompactionPhase::ReclaimingSpace => 0.85,
            CompactionPhase::Verifying => 0.95,
            CompactionPhase::Completed => 1.0,
        };

        let progress = base_progress + (phase_progress * phase_weight);
        progress.min(1.0) // Clamp to max 1.0
    }

    /// Get current progress
    pub fn get_progress(&self) -> Option<CompactionProgress> {
        self.progress.read().clone()
    }

    /// Get statistics
    pub fn get_statistics(&self) -> CompactionStatistics {
        self.metrics.get_statistics()
    }

    /// Enable/disable compaction
    pub fn set_enabled(&self, enabled: bool) {
        *self.enabled.write() = enabled;
    }

    /// Check if compaction is enabled
    pub fn is_enabled(&self) -> bool {
        *self.enabled.read()
    }

    /// Get metrics
    pub fn get_metrics(&self) -> Arc<CompactionMetrics> {
        self.metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compaction_manager_creation() {
        let config = CompactionConfig::default();
        let manager = CompactionManager::new(config).unwrap();
        assert!(manager.is_enabled());
    }

    #[test]
    fn test_fragment_registration() {
        let config = CompactionConfig::default();
        let manager = CompactionManager::new(config).unwrap();

        manager.register_fragment("vec1".to_string(), 0, 1024);
        manager.register_fragment("vec2".to_string(), 1024, 1024);

        assert_eq!(manager.calculate_fragmentation(), 0.0);
    }

    #[test]
    fn test_fragmentation_calculation() {
        let config = CompactionConfig::default();
        let manager = CompactionManager::new(config).unwrap();

        manager.register_fragment("vec1".to_string(), 0, 1024);
        manager.register_fragment("vec2".to_string(), 1024, 1024);
        manager.register_fragment("vec3".to_string(), 2048, 1024);

        // Mark one as deleted
        manager.mark_deleted("vec2").unwrap();

        // Fragmentation should be ~33% (1024 / 3072)
        let frag = manager.calculate_fragmentation();
        assert!((frag - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_should_compact_threshold() {
        let config = CompactionConfig {
            strategy: super::super::strategies::CompactionStrategy::ThresholdBased,
            fragmentation_threshold: 0.3,
            ..Default::default()
        };
        let manager = CompactionManager::new(config).unwrap();

        manager.register_fragment("vec1".to_string(), 0, 1024);
        manager.register_fragment("vec2".to_string(), 1024, 1024);

        assert!(!manager.should_compact());

        manager.mark_deleted("vec1").unwrap();
        manager.mark_deleted("vec2").unwrap();

        // Should compact with high fragmentation
        assert!(manager.should_compact());
    }

    #[test]
    fn test_compact_empty() {
        let config = CompactionConfig::default();
        let manager = CompactionManager::new(config).unwrap();

        let result = manager.compact_now().unwrap();
        assert!(result.success);
        assert_eq!(result.vectors_removed, 0);
    }

    #[test]
    fn test_compact_with_deletions() {
        let config = CompactionConfig::default();
        let manager = CompactionManager::new(config).unwrap();

        manager.register_fragment("vec1".to_string(), 0, 1024);
        manager.register_fragment("vec2".to_string(), 1024, 1024);
        manager.register_fragment("vec3".to_string(), 2048, 1024);

        manager.mark_deleted("vec1").unwrap();
        manager.mark_deleted("vec3").unwrap();

        let result = manager.compact_now().unwrap();
        assert!(result.success);
        assert_eq!(result.vectors_removed, 2);
        assert_eq!(result.bytes_reclaimed, 2048);
    }
}
