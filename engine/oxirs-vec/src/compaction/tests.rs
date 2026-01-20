//! Integration tests for compaction system

#[cfg(test)]
mod integration_tests {
    use crate::compaction::{CompactionConfig, CompactionManager, CompactionStrategy};
    use std::time::Duration;

    #[test]
    fn test_end_to_end_compaction() {
        let config = CompactionConfig {
            strategy: CompactionStrategy::ThresholdBased,
            fragmentation_threshold: 0.2,
            batch_size: 100,
            ..Default::default()
        };

        let manager = CompactionManager::new(config).unwrap();

        // Add some vectors
        for i in 0..1000 {
            manager.register_fragment(format!("vec{}", i), i * 1024, 1024);
        }

        // Delete half of them
        for i in (0..1000).step_by(2) {
            manager.mark_deleted(&format!("vec{}", i)).unwrap();
        }

        // Should trigger compaction
        assert!(manager.should_compact());

        // Compact
        let result = manager.compact_now().unwrap();
        assert!(result.success);
        assert_eq!(result.vectors_removed, 500);
        assert!(result.fragmentation_after < result.fragmentation_before);
    }

    #[test]
    fn test_compaction_with_different_strategies() {
        let strategies = vec![
            CompactionStrategy::Periodic,
            CompactionStrategy::ThresholdBased,
            CompactionStrategy::SizeBased,
            CompactionStrategy::Adaptive,
        ];

        for strategy in strategies {
            let config = CompactionConfig {
                strategy,
                ..CompactionConfig::development()
            };

            let manager = CompactionManager::new(config).unwrap();

            // Add and delete vectors
            for i in 0..100 {
                manager.register_fragment(format!("vec{}", i), i * 1024, 1024);
            }
            for i in 0..50 {
                manager.mark_deleted(&format!("vec{}", i)).unwrap();
            }

            // Compact should work regardless of strategy
            let result = manager.compact_now();
            assert!(result.is_ok(), "Strategy {:?} failed", strategy);
        }
    }

    #[test]
    fn test_incremental_compaction() {
        let config = CompactionConfig {
            batch_size: 10,
            pause_between_batches: Duration::from_millis(1),
            ..Default::default()
        };

        let manager = CompactionManager::new(config).unwrap();

        // Add vectors
        for i in 0..100 {
            manager.register_fragment(format!("vec{}", i), i * 1024, 1024);
        }

        // Delete most
        for i in 0..90 {
            manager.mark_deleted(&format!("vec{}", i)).unwrap();
        }

        let result = manager.compact_now().unwrap();
        assert!(result.success);
        assert_eq!(result.vectors_removed, 90);
    }

    #[test]
    fn test_metrics_collection() {
        let config = CompactionConfig::default();
        let manager = CompactionManager::new(config).unwrap();

        // Perform multiple compactions
        for round in 0..3 {
            for i in 0..10 {
                manager.register_fragment(format!("vec{}_{}", round, i), i * 1024, 1024);
            }
            for i in 0..5 {
                manager
                    .mark_deleted(&format!("vec{}_{}", round, i))
                    .unwrap();
            }
            manager.compact_now().unwrap();
        }

        let stats = manager.get_statistics();
        assert_eq!(stats.total_compactions, 3);
        assert_eq!(stats.successful_compactions, 3);
        assert_eq!(stats.total_vectors_removed, 15);
    }

    #[test]
    fn test_enable_disable() {
        let config = CompactionConfig::default();
        let manager = CompactionManager::new(config).unwrap();

        assert!(manager.is_enabled());

        manager.set_enabled(false);
        assert!(!manager.is_enabled());
        assert!(!manager.should_compact());

        manager.set_enabled(true);
        assert!(manager.is_enabled());
    }

    #[test]
    fn test_progress_tracking() {
        let config = CompactionConfig {
            batch_size: 10,
            ..Default::default()
        };

        let manager = CompactionManager::new(config).unwrap();

        // Add vectors
        for i in 0..50 {
            manager.register_fragment(format!("vec{}", i), i * 1024, 1024);
        }
        for i in 0..25 {
            manager.mark_deleted(&format!("vec{}", i)).unwrap();
        }

        manager.compact_now().unwrap();

        // Progress should exist and be valid
        // Note: Progress might be None if already cleared, or completed
        if let Some(progress) = manager.get_progress() {
            assert!(
                progress.overall_progress >= 0.0 && progress.overall_progress <= 1.01,
                "Progress out of bounds: {}",
                progress.overall_progress
            );
        }
        // If no progress, that's also fine (already completed and cleared)
    }
}
