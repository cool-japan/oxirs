//! Unit tests for the diagnostics subsystem.

#[cfg(test)]
mod tests {
    use crate::diagnostics_collectors::{
        BufferPoolCheck, CorruptionDetectionCheck, DiagnosticCheck, DiagnosticContext,
        DiagnosticEngine, DictionaryConsistencyCheck, IndexConsistencyCheck, MemoryUsageCheck,
        PerformanceCheck, StorageEfficiencyCheck, WalIntegrityCheck,
    };
    use crate::diagnostics_types::{
        DiagnosticLevel, DiagnosticReport, DiagnosticResult, DiagnosticSummary, HealthStatus,
        RepairAction, Severity,
    };
    use crate::dictionary::NodeId;
    use crate::index::Triple;
    use crate::storage::BufferPoolStats;
    use std::collections::HashSet;
    use std::time::Duration;

    fn create_test_context() -> DiagnosticContext {
        DiagnosticContext::quick(
            1000,                       // triple_count
            BufferPoolStats::default(), // buffer_pool_stats
            5000,                       // dictionary_size
            200_000,                    // storage_size_bytes
            50_000_000,                 // memory_usage_bytes (50MB)
        )
    }

    /// Unique, non-existent temp directory path string used as a `data_dir`
    /// for diagnostics checks (the WAL integrity check probes `<data_dir>/wal`,
    /// which is expected to be absent).
    fn test_data_dir(tag: &str) -> String {
        std::env::temp_dir()
            .join(format!("oxirs_tdb_diag_{}_{}", tag, std::process::id()))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_diagnostic_engine_creation() {
        let engine = DiagnosticEngine::new();
        assert!(!engine.checks.is_empty());
    }

    #[test]
    fn test_run_quick_diagnostics() {
        let engine = DiagnosticEngine::new();
        let context = create_test_context();

        let report = engine.run(DiagnosticLevel::Quick, &context);

        assert!(!report.results.is_empty());
        assert_eq!(report.level, DiagnosticLevel::Quick);
    }

    #[test]
    fn test_run_standard_diagnostics() {
        let engine = DiagnosticEngine::new();
        let context = create_test_context();

        let report = engine.run(DiagnosticLevel::Standard, &context);

        assert!(!report.results.is_empty());
        // Standard should run more checks than quick
        let standard_count = report.results.len();
        assert!(standard_count > 0);
    }

    #[test]
    fn test_health_status_determination() {
        let engine = DiagnosticEngine::new();
        let context = create_test_context();

        // Set low hit rate to trigger warnings
        context
            .buffer_pool_stats
            .total_fetches
            .store(1000, std::sync::atomic::Ordering::Relaxed);
        context
            .buffer_pool_stats
            .cache_hits
            .store(400, std::sync::atomic::Ordering::Relaxed);

        let report = engine.run(DiagnosticLevel::Quick, &context);

        // Should have some warnings
        assert!(!report.warnings().is_empty());
    }

    #[test]
    fn test_diagnostic_result_builder() {
        let result = DiagnosticResult::new("Test", "Check", Severity::Warning)
            .with_description("Test description")
            .with_recommendation("Fix it")
            .with_detail("key", "value");

        assert_eq!(result.category, "Test");
        assert_eq!(result.name, "Check");
        assert_eq!(result.severity, Severity::Warning);
        assert_eq!(result.description, "Test description");
        assert_eq!(result.recommendation, Some("Fix it".to_string()));
        assert_eq!(result.details.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_results_by_severity() {
        let engine = DiagnosticEngine::new();
        let context = create_test_context();

        let report = engine.run(DiagnosticLevel::Standard, &context);

        let warnings = report.results_by_severity(Severity::Warning);
        let info = report.results_by_severity(Severity::Info);

        // Should have some results
        assert!(warnings.len() + info.len() > 0);
    }

    #[test]
    fn test_diagnostic_summary() {
        let results = vec![
            DiagnosticResult::new("Cat1", "Check1", Severity::Info),
            DiagnosticResult::new("Cat2", "Check2", Severity::Warning),
            DiagnosticResult::new("Cat3", "Check3", Severity::Error),
            DiagnosticResult::new("Cat4", "Check4", Severity::Critical),
        ];

        let summary = DiagnosticSummary::from_results(&results);

        assert_eq!(summary.info_count, 1);
        assert_eq!(summary.warning_count, 1);
        assert_eq!(summary.error_count, 1);
        assert_eq!(summary.critical_count, 1);
    }

    #[test]
    fn test_buffer_pool_check() {
        let check = BufferPoolCheck;
        let context = create_test_context();

        // Set up low hit rate
        context
            .buffer_pool_stats
            .total_fetches
            .store(1000, std::sync::atomic::Ordering::Relaxed);
        context
            .buffer_pool_stats
            .cache_hits
            .store(400, std::sync::atomic::Ordering::Relaxed);

        let results = check.run(&context).unwrap();

        // Should detect low hit rate
        assert!(!results.is_empty());
        assert_eq!(results[0].severity, Severity::Warning);
    }

    #[test]
    fn test_memory_usage_check() {
        let check = MemoryUsageCheck;
        let context = create_test_context();

        let results = check.run(&context).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_storage_efficiency_check() {
        let check = StorageEfficiencyCheck;
        let context = create_test_context();

        let results = check.run(&context).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_performance_check() {
        let check = PerformanceCheck;
        let context = create_test_context();

        context
            .buffer_pool_stats
            .total_fetches
            .store(1000, std::sync::atomic::Ordering::Relaxed);
        context
            .buffer_pool_stats
            .evictions
            .store(50, std::sync::atomic::Ordering::Relaxed);

        let results = check.run(&context).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn test_health_status_from_summary() {
        let engine = DiagnosticEngine::new();
        let context = create_test_context();

        let report = engine.run(DiagnosticLevel::Quick, &context);

        // With good settings, should be healthy or degraded
        assert!(
            report.health_status == HealthStatus::Healthy
                || report.health_status == HealthStatus::Degraded
        );
    }

    // Advanced diagnostic tests

    #[test]
    fn test_index_consistency_check_no_data() {
        let check = IndexConsistencyCheck;
        let context = create_test_context();

        let results = check.run(&context).unwrap();

        // Should warn about insufficient data
        assert!(!results.is_empty());
        assert_eq!(results[0].severity, Severity::Warning);
        assert!(results[0].name.contains("Insufficient"));
    }

    #[test]
    fn test_index_consistency_check_consistent() {
        let check = IndexConsistencyCheck;

        let triple1 = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triple2 = Triple::new(NodeId::new(4), NodeId::new(5), NodeId::new(6));
        let triples = vec![triple1, triple2];

        let context = DiagnosticContext::deep(
            2,                               // triple_count
            BufferPoolStats::default(),      // buffer_pool_stats
            10,                              // dictionary_size
            1000,                            // storage_size_bytes
            50_000_000,                      // memory_usage_bytes
            triples.clone(),                 // spo_triples
            triples.clone(),                 // pos_triples
            triples,                         // osp_triples
            HashSet::new(),                  // dictionary_node_ids
            test_data_dir("idx_consistent"), // data_dir
        );

        let results = check.run(&context).unwrap();

        // Should report consistent indexes
        assert!(!results.is_empty());
        let has_consistent = results.iter().any(|r| r.name.contains("Consistent"));
        assert!(has_consistent);
    }

    #[test]
    fn test_index_consistency_check_inconsistent() {
        let check = IndexConsistencyCheck;

        let triple1 = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triple2 = Triple::new(NodeId::new(4), NodeId::new(5), NodeId::new(6));
        let triple3 = Triple::new(NodeId::new(7), NodeId::new(8), NodeId::new(9));

        let spo_triples = vec![triple1, triple2];
        let pos_triples = vec![triple1, triple3]; // Different!
        let osp_triples = vec![triple1, triple2];

        let context = DiagnosticContext::deep(
            2,
            BufferPoolStats::default(),
            10,
            1000,
            50_000_000,
            spo_triples,
            pos_triples,
            osp_triples,
            HashSet::new(),
            test_data_dir("diag"),
        );

        let results = check.run(&context).unwrap();

        // Should detect inconsistency
        assert!(!results.is_empty());
        let has_critical = results.iter().any(|r| r.severity == Severity::Critical);
        assert!(has_critical);
    }

    #[test]
    fn test_dictionary_consistency_check_no_data() {
        let check = DictionaryConsistencyCheck;
        let context = create_test_context();

        let results = check.run(&context).unwrap();

        // Should warn about insufficient data
        assert!(!results.is_empty());
        assert_eq!(results[0].severity, Severity::Warning);
    }

    #[test]
    fn test_dictionary_consistency_check_consistent() {
        let check = DictionaryConsistencyCheck;

        let triple1 = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triples = vec![triple1];

        let mut dict_ids = HashSet::new();
        dict_ids.insert(NodeId::new(1));
        dict_ids.insert(NodeId::new(2));
        dict_ids.insert(NodeId::new(3));

        let context = DiagnosticContext::deep(
            1,
            BufferPoolStats::default(),
            3,
            500,
            50_000_000,
            triples,
            vec![],
            vec![],
            dict_ids,
            test_data_dir("diag"),
        );

        let results = check.run(&context).unwrap();

        // Should report consistent dictionary
        assert!(!results.is_empty());
        let has_consistent = results.iter().any(|r| r.name.contains("Consistent"));
        assert!(has_consistent);
    }

    #[test]
    fn test_dictionary_consistency_check_missing_entries() {
        let check = DictionaryConsistencyCheck;

        let triple1 = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triples = vec![triple1];

        let mut dict_ids = HashSet::new();
        dict_ids.insert(NodeId::new(1));
        // Missing NodeId 2 and 3!

        let context = DiagnosticContext::deep(
            1,
            BufferPoolStats::default(),
            1,
            500,
            50_000_000,
            triples,
            vec![],
            vec![],
            dict_ids,
            test_data_dir("diag"),
        );

        let results = check.run(&context).unwrap();

        // Should detect missing entries
        assert!(!results.is_empty());
        let has_critical = results.iter().any(|r| r.severity == Severity::Critical);
        assert!(has_critical);
        let has_missing = results.iter().any(|r| r.name.contains("Missing"));
        assert!(has_missing);
    }

    #[test]
    fn test_dictionary_consistency_check_orphaned_entries() {
        let check = DictionaryConsistencyCheck;

        let triple1 = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triples = vec![triple1];

        let mut dict_ids = HashSet::new();
        dict_ids.insert(NodeId::new(1));
        dict_ids.insert(NodeId::new(2));
        dict_ids.insert(NodeId::new(3));
        // Add many orphaned entries
        for i in 100..250 {
            dict_ids.insert(NodeId::new(i));
        }

        let context = DiagnosticContext::deep(
            1,
            BufferPoolStats::default(),
            dict_ids.len() as u64,
            500,
            50_000_000,
            triples,
            vec![],
            vec![],
            dict_ids,
            test_data_dir("diag"),
        );

        let results = check.run(&context).unwrap();

        // Should warn about orphaned entries
        let has_orphaned = results.iter().any(|r| r.name.contains("Orphaned"));
        assert!(has_orphaned);
    }

    #[test]
    fn test_wal_integrity_check_no_data() {
        let check = WalIntegrityCheck;
        let context = create_test_context();

        let results = check.run(&context).unwrap();

        // Should be info level (no data directory provided)
        assert!(!results.is_empty());
        assert_eq!(results[0].severity, Severity::Info);
    }

    #[test]
    fn test_wal_integrity_check_no_wal() {
        let check = WalIntegrityCheck;

        use std::env;
        let temp_dir = env::temp_dir().join("oxirs_test_wal_missing");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut context = create_test_context();
        context.data_dir = Some(temp_dir.to_str().unwrap_or_default().to_string());

        let results = check.run(&context).unwrap();

        // Should warn about missing WAL
        assert!(!results.is_empty());
        let has_warning = results.iter().any(|r| r.name.contains("Not Found"));
        assert!(has_warning);

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_corruption_detection_check_healthy() {
        let check = CorruptionDetectionCheck;
        let context = create_test_context();

        let results = check.run(&context).unwrap();

        // Should report no corruption
        assert!(!results.is_empty());
        let has_ok = results.iter().any(|r| r.name.contains("No Corruption"));
        assert!(has_ok);
    }

    #[test]
    fn test_corruption_detection_check_suspicious_state() {
        let check = CorruptionDetectionCheck;

        let mut context = create_test_context();
        context.triple_count = 1000;
        context.dictionary_size = 0; // Suspicious!

        let results = check.run(&context).unwrap();

        // Should detect suspicious state
        assert!(!results.is_empty());
        let has_critical = results.iter().any(|r| r.severity == Severity::Critical);
        assert!(has_critical);
    }

    #[test]
    fn test_corruption_detection_check_storage_overhead() {
        let check = CorruptionDetectionCheck;

        let mut context = create_test_context();
        context.triple_count = 1000;
        context.storage_size_bytes = 20_000_000; // 20KB per triple - excessive!

        let results = check.run(&context).unwrap();

        // Should detect excessive overhead
        assert!(!results.is_empty());
        let has_warning = results.iter().any(|r| r.name.contains("Excessive Storage"));
        assert!(has_warning);
    }

    #[test]
    fn test_corruption_detection_check_dictionary_bloat() {
        let check = CorruptionDetectionCheck;

        let mut context = create_test_context();
        context.triple_count = 1000;
        context.dictionary_size = 15_000; // 15x ratio - bloated!

        let results = check.run(&context).unwrap();

        // Should detect dictionary bloat
        assert!(!results.is_empty());
        let has_bloat = results.iter().any(|r| r.name.contains("Bloat"));
        assert!(has_bloat);
    }

    #[test]
    fn test_repair_recommendations_index_issues() {
        let results =
            vec![
                DiagnosticResult::new("Integrity", "Index Inconsistency", Severity::Error)
                    .with_description("Indexes are inconsistent"),
            ];

        let report = DiagnosticReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            level: DiagnosticLevel::Deep,
            duration: Duration::from_secs(1),
            health_status: HealthStatus::Unhealthy,
            results,
            summary: DiagnosticSummary {
                info_count: 0,
                warning_count: 0,
                error_count: 1,
                critical_count: 0,
            },
        };

        let recommendations = report.repair_recommendations();

        assert!(!recommendations.is_empty());
        let has_rebuild = recommendations
            .iter()
            .any(|r| matches!(r.action, RepairAction::RebuildIndexes));
        assert!(has_rebuild);
    }

    #[test]
    fn test_repair_recommendations_dictionary_issues() {
        let results =
            vec![
                DiagnosticResult::new("Integrity", "Dictionary Corruption", Severity::Critical)
                    .with_description("Dictionary is corrupted"),
            ];

        let report = DiagnosticReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            level: DiagnosticLevel::Deep,
            duration: Duration::from_secs(1),
            health_status: HealthStatus::Critical,
            results,
            summary: DiagnosticSummary {
                info_count: 0,
                warning_count: 0,
                error_count: 0,
                critical_count: 1,
            },
        };

        let recommendations = report.repair_recommendations();

        assert!(!recommendations.is_empty());
        let has_restore = recommendations
            .iter()
            .any(|r| matches!(r.action, RepairAction::RestoreFromBackup));
        assert!(has_restore);
    }

    #[test]
    fn test_repair_recommendations_wal_issues() {
        let results = vec![
            DiagnosticResult::new("Integrity", "Large WAL File", Severity::Warning)
                .with_description("WAL file is large"),
        ];

        let report = DiagnosticReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            level: DiagnosticLevel::Standard,
            duration: Duration::from_secs(1),
            health_status: HealthStatus::Degraded,
            results,
            summary: DiagnosticSummary {
                info_count: 0,
                warning_count: 1,
                error_count: 0,
                critical_count: 0,
            },
        };

        let recommendations = report.repair_recommendations();

        assert!(!recommendations.is_empty());
        let has_checkpoint = recommendations
            .iter()
            .any(|r| matches!(r.action, RepairAction::CheckpointWal));
        assert!(has_checkpoint);
    }

    #[test]
    fn test_diagnostic_context_quick_constructor() {
        let context =
            DiagnosticContext::quick(100, BufferPoolStats::default(), 500, 10_000, 1_000_000);

        assert_eq!(context.triple_count, 100);
        assert_eq!(context.dictionary_size, 500);
        assert!(context.spo_triples.is_none());
        assert!(context.pos_triples.is_none());
        assert!(context.osp_triples.is_none());
        assert!(context.dictionary_node_ids.is_none());
        assert!(context.data_dir.is_none());
    }

    #[test]
    fn test_diagnostic_context_deep_constructor() {
        let triple = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triples = vec![triple];
        let dict_ids = HashSet::new();

        let context = DiagnosticContext::deep(
            1,
            BufferPoolStats::default(),
            3,
            500,
            1_000_000,
            triples.clone(),
            triples.clone(),
            triples,
            dict_ids,
            test_data_dir("diag"),
        );

        assert_eq!(context.triple_count, 1);
        assert!(context.spo_triples.is_some());
        assert!(context.pos_triples.is_some());
        assert!(context.osp_triples.is_some());
        assert!(context.dictionary_node_ids.is_some());
        assert!(context.data_dir.is_some());
    }

    #[test]
    fn test_deep_diagnostics_runs_all_checks() {
        let engine = DiagnosticEngine::new();

        let triple = Triple::new(NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let triples = vec![triple];
        let mut dict_ids = HashSet::new();
        dict_ids.insert(NodeId::new(1));
        dict_ids.insert(NodeId::new(2));
        dict_ids.insert(NodeId::new(3));

        let context = DiagnosticContext::deep(
            1,
            BufferPoolStats::default(),
            3,
            500,
            50_000_000,
            triples.clone(),
            triples.clone(),
            triples,
            dict_ids,
            test_data_dir("diag"),
        );

        let report = engine.run(DiagnosticLevel::Deep, &context);

        // Deep diagnostics should run more checks than quick
        assert!(report.results.len() > 4);

        // Should include integrity checks
        let has_integrity = report.results.iter().any(|r| r.category == "Integrity");
        assert!(has_integrity);
    }
}
