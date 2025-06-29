//! Basic smoke test for oxirs-shacl-ai core functionality
//!
//! This test validates that the core modules can be instantiated
//! and basic functionality works without detailed API testing.

use oxirs_shacl_ai::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shacl_ai_assistant_creation() {
        let assistant = ShaclAiAssistant::new();
        let stats = assistant.get_ai_statistics().unwrap();

        assert_eq!(stats.shapes_learned, 0);
        assert_eq!(stats.quality_assessments, 0);
        assert_eq!(stats.predictions_made, 0);
    }

    #[test]
    fn test_shacl_ai_config_default() {
        let config = ShaclAiConfig::default();

        assert!(config.global.enable_parallel_processing);
        assert_eq!(config.global.max_memory_mb, 1024);
        assert!(config.global.enable_caching);
        assert_eq!(config.global.cache_size_limit, 10000);
    }

    #[test]
    fn test_shacl_ai_builder() {
        let assistant = ShaclAiAssistantBuilder::new()
            .enable_parallel_processing(false)
            .max_memory_mb(512)
            .enable_caching(false)
            .build();

        assert!(!assistant.config().global.enable_parallel_processing);
        assert_eq!(assistant.config().global.max_memory_mb, 512);
        assert!(!assistant.config().global.enable_caching);
    }

    #[test]
    fn test_deployment_manager_creation() {
        let deployment_manager = DeploymentManager::new();
        let stats = deployment_manager.get_statistics();

        assert_eq!(stats.successful_deployments, 0);
        assert_eq!(stats.failed_deployments, 0);
    }

    #[test]
    fn test_predictive_analytics_creation() {
        let analytics_engine = PredictiveAnalyticsEngine::new();
        let stats = analytics_engine.get_statistics();

        assert_eq!(stats.forecasts_generated, 0);
        assert_eq!(stats.recommendations_generated, 0);
    }

    // Note: Additional modules can be tested once their APIs are stabilized

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert!(init().is_ok());
    }
}
