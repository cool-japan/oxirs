#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OxirsConfig::default();
        assert_eq!(config.performance.profile, PerformanceProfile::Balanced);
        assert!(config.monitoring.enabled);
    }

    #[test]
    fn test_performance_profiles_get_config() {
        for profile in [
            PerformanceProfile::Development,
            PerformanceProfile::Balanced,
            PerformanceProfile::HighPerformance,
            PerformanceProfile::MaxThroughput,
            PerformanceProfile::MemoryEfficient,
            PerformanceProfile::LowLatency,
            PerformanceProfile::BatchProcessing,
            PerformanceProfile::RealTime,
            PerformanceProfile::EdgeComputing,
        ] {
            let config = profile.get_config();
            assert!(!config.is_empty(), "Profile {:?} should return non-empty settings map", profile);
        }
    }

    #[test]
    fn test_performance_profile_descriptions() {
        let profiles = [
            PerformanceProfile::Development,
            PerformanceProfile::Balanced,
            PerformanceProfile::HighPerformance,
            PerformanceProfile::MaxThroughput,
            PerformanceProfile::MemoryEfficient,
            PerformanceProfile::LowLatency,
            PerformanceProfile::BatchProcessing,
            PerformanceProfile::RealTime,
            PerformanceProfile::EdgeComputing,
            PerformanceProfile::Custom,
        ];
        for profile in &profiles {
            let desc = profile.description();
            assert!(!desc.is_empty(), "Profile {:?} description should not be empty", profile);
        }
    }

    #[test]
    fn test_configuration_manager_new_has_default_config() {
        let manager = ConfigurationManager::new();
        let config = manager.get_config();
        assert_eq!(config.performance.profile, PerformanceProfile::Balanced);
    }

    #[test]
    fn test_configuration_manager_set_and_get_profile() {
        let mut manager = ConfigurationManager::new();
        manager
            .set_performance_profile(PerformanceProfile::HighPerformance)
            .expect("set_performance_profile should succeed");
        assert_eq!(
            manager.get_performance_profile(),
            PerformanceProfile::HighPerformance
        );
    }

    #[test]
    fn test_configuration_manager_profile_round_trip() {
        let mut manager = ConfigurationManager::new();
        let profiles = [
            PerformanceProfile::LowLatency,
            PerformanceProfile::BatchProcessing,
            PerformanceProfile::MaxThroughput,
        ];
        for profile in profiles {
            manager
                .set_performance_profile(profile)
                .expect("set_performance_profile should succeed");
            assert_eq!(
                manager.get_performance_profile(),
                profile,
                "Round-trip failed for {:?}",
                profile
            );
        }
    }

    #[test]
    fn test_update_config_rejects_zero_worker_threads() {
        let mut manager = ConfigurationManager::new();
        let mut invalid_config = OxirsConfig::default();
        invalid_config.concurrency.thread_pool.worker_threads = 0;
        assert!(
            manager.update_config(invalid_config).is_err(),
            "Config with zero worker threads should fail validation"
        );
    }

    #[test]
    fn test_update_config_accepts_valid_config() {
        let mut manager = ConfigurationManager::new();
        let mut valid_config = OxirsConfig::default();
        valid_config.concurrency.thread_pool.worker_threads = 8;
        assert!(
            manager.update_config(valid_config).is_ok(),
            "Valid config with 8 worker threads should succeed"
        );
    }

    #[test]
    fn test_config_serialization_json_round_trip() {
        let config = OxirsConfig::default();
        let json = serde_json::to_string_pretty(&config)
            .expect("serde_json serialization should succeed");
        let deserialized: OxirsConfig =
            serde_json::from_str(&json).expect("serde_json deserialization should succeed");
        assert_eq!(config.performance.profile, deserialized.performance.profile);
        assert_eq!(
            config.memory.arena.initial_size,
            deserialized.memory.arena.initial_size
        );
    }

    #[test]
    fn test_default_memory_config_arena_sizes() {
        let config = OxirsConfig::default();
        assert!(
            config.memory.arena.initial_size > 0,
            "Initial arena size should be positive"
        );
        assert!(
            config.memory.arena.max_size >= config.memory.arena.initial_size,
            "Max arena size should be >= initial size"
        );
        assert!(
            config.memory.arena.growth_factor > 1.0,
            "Growth factor should be > 1.0"
        );
    }

    #[test]
    fn test_default_optimization_config() {
        let config = OxirsConfig::default();
        assert!(
            config.optimization.simd.enabled,
            "SIMD should be enabled by default"
        );
        assert!(
            config.optimization.simd.fallback_to_scalar,
            "SIMD scalar fallback should be enabled by default"
        );
        assert!(
            config.optimization.zero_copy.enabled,
            "Zero-copy should be enabled by default"
        );
        assert!(
            config.optimization.prefetching.enabled,
            "Prefetching should be enabled by default"
        );
    }

    #[test]
    fn test_default_security_config() {
        let config = OxirsConfig::default();
        assert!(
            config.security.encryption.in_transit.enabled,
            "In-transit encryption should be enabled by default"
        );
        assert_eq!(
            config.security.encryption.in_transit.tls_version,
            TlsVersion::TLSv1_3,
            "Default TLS version should be 1.3"
        );
    }

    #[test]
    fn test_default_concurrency_config() {
        let config = OxirsConfig::default();
        assert!(
            config.concurrency.thread_pool.worker_threads > 0,
            "Default worker thread count should be positive"
        );
    }

    #[test]
    fn test_load_from_nonexistent_file_returns_error() {
        let mut manager = ConfigurationManager::new();
        let result = manager.load_from_file("/nonexistent/path/config.toml");
        assert!(
            result.is_err(),
            "Loading from nonexistent file should return an error"
        );
        match result.expect_err("must be an error") {
            ConfigError::FileNotFound(_) => {}
            other => panic!("Expected FileNotFound error, got: {:?}", other),
        }
    }

    #[test]
    fn test_load_config_from_json_file() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("oxirs_test_config.json");

        let config = OxirsConfig::default();
        let json = serde_json::to_string(&config)
            .expect("Serialization should succeed");
        let mut file = std::fs::File::create(&config_path)
            .expect("Creating temp config file should succeed");
        file.write_all(json.as_bytes())
            .expect("Writing config file should succeed");

        let mut manager = ConfigurationManager::new();
        manager
            .load_from_file(&config_path)
            .expect("Loading from valid JSON config file should succeed");

        assert_eq!(
            manager.get_performance_profile(),
            PerformanceProfile::Balanced
        );

        // Clean up
        let _ = std::fs::remove_file(&config_path);
    }

    #[test]
    fn test_global_cache_config_defaults() {
        let config = OxirsConfig::default();
        let global = &config.optimization.caching.global;
        assert!(global.memory_limit > 0, "Cache memory limit should be positive");
        assert!(global.enable_statistics, "Cache statistics should be enabled by default");
        assert!(global.enable_warming, "Cache warming should be enabled by default");
    }
}
