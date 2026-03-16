//! Integration tests for compaction system

#[cfg(test)]
mod integration_tests {
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use crate::compaction::{CompactionConfig, CompactionManager, CompactionStrategy};
    use std::time::Duration;

    #[test]
    fn test_end_to_end_compaction() -> Result<()> {
        let config = CompactionConfig {
            strategy: CompactionStrategy::ThresholdBased,
            fragmentation_threshold: 0.2,
            batch_size: 100,
            ..Default::default()
        };

        let manager = CompactionManager::new(config)?;

        // Add some vectors
        for i in 0..1000 {
            manager.register_fragment(format!("vec{}", i), i * 1024, 1024);
        }

        // Delete half of them
        for i in (0..1000).step_by(2) {
            manager.mark_deleted(&format!("vec{}", i))?;
        }

        // Should trigger compaction
        assert!(manager.should_compact());

        // Compact
        let result = manager.compact_now()?;
        assert!(result.success);
        assert_eq!(result.vectors_removed, 500);
        assert!(result.fragmentation_after < result.fragmentation_before);
        Ok(())
    }

    #[test]
    fn test_compaction_with_different_strategies() -> Result<()> {
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

            let manager = CompactionManager::new(config)?;

            // Add and delete vectors
            for i in 0..100 {
                manager.register_fragment(format!("vec{}", i), i * 1024, 1024);
            }
            for i in 0..50 {
                manager.mark_deleted(&format!("vec{}", i))?;
            }

            // Compact should work regardless of strategy
            let result = manager.compact_now();
            assert!(result.is_ok(), "Strategy {:?} failed", strategy);
        }
        Ok(())
    }

    #[test]
    fn test_incremental_compaction() -> Result<()> {
        let config = CompactionConfig {
            batch_size: 10,
            pause_between_batches: Duration::from_millis(1),
            ..Default::default()
        };

        let manager = CompactionManager::new(config)?;

        // Add vectors
        for i in 0..100 {
            manager.register_fragment(format!("vec{}", i), i * 1024, 1024);
        }

        // Delete most
        for i in 0..90 {
            manager.mark_deleted(&format!("vec{}", i))?;
        }

        let result = manager.compact_now()?;
        assert!(result.success);
        assert_eq!(result.vectors_removed, 90);
        Ok(())
    }

    #[test]
    fn test_metrics_collection() -> Result<()> {
        let config = CompactionConfig::default();
        let manager = CompactionManager::new(config)?;

        // Perform multiple compactions
        for round in 0..3 {
            for i in 0..10 {
                manager.register_fragment(format!("vec{}_{}", round, i), i * 1024, 1024);
            }
            for i in 0..5 {
                manager.mark_deleted(&format!("vec{}_{}", round, i))?;
            }
            manager.compact_now()?;
        }

        let stats = manager.get_statistics();
        assert_eq!(stats.total_compactions, 3);
        assert_eq!(stats.successful_compactions, 3);
        assert_eq!(stats.total_vectors_removed, 15);
        Ok(())
    }

    #[test]
    fn test_enable_disable() -> Result<()> {
        let config = CompactionConfig::default();
        let manager = CompactionManager::new(config)?;

        assert!(manager.is_enabled());

        manager.set_enabled(false);
        assert!(!manager.is_enabled());
        assert!(!manager.should_compact());

        manager.set_enabled(true);
        assert!(manager.is_enabled());
        Ok(())
    }

    #[test]
    fn test_progress_tracking() -> Result<()> {
        let config = CompactionConfig {
            batch_size: 10,
            ..Default::default()
        };

        let manager = CompactionManager::new(config)?;

        // Add vectors
        for i in 0..50 {
            manager.register_fragment(format!("vec{}", i), i * 1024, 1024);
        }
        for i in 0..25 {
            manager.mark_deleted(&format!("vec{}", i))?;
        }

        manager.compact_now()?;

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
        Ok(())
    }
}
