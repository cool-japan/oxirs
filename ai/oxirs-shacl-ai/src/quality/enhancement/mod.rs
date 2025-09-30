//! Quality Enhancement Recommendations for SHACL-AI
//!
//! This module implements comprehensive quality enhancement recommendations including
//! data enhancement, process optimization, and automated improvement strategies.

pub mod core;

// Re-export main types for easy access
pub use core::{
    ActionStatus, AutomationModel, CostBenefitModel, DataEnhancementModel, EnhancementAction,
    EnhancementActionType, EnhancementCategory, EnhancementConfig, EnhancementRecommendation,
    EnhancementStatistics, EnhancementStrategy, ImpactPredictionModel, ImplementationEffort,
    Priority, ProcessOptimizationModel, QualityEnhancementEngine, RecommendationModels,
};

/// Create a default quality enhancement engine
pub fn create_default_engine() -> QualityEnhancementEngine {
    QualityEnhancementEngine::default()
}

/// Create a quality enhancement engine with custom configuration
pub fn create_engine_with_config(config: EnhancementConfig) -> QualityEnhancementEngine {
    QualityEnhancementEngine::new(config)
}

/// Utility function to create a conservative enhancement configuration
pub fn create_conservative_config() -> EnhancementConfig {
    EnhancementConfig {
        strategy_preference: EnhancementStrategy::Conservative,
        priority_threshold: 0.8,
        min_recommendation_confidence: 0.85,
        max_recommendations_per_category: 5,
        ..Default::default()
    }
}

/// Utility function to create an aggressive enhancement configuration
pub fn create_aggressive_config() -> EnhancementConfig {
    EnhancementConfig {
        strategy_preference: EnhancementStrategy::Aggressive,
        priority_threshold: 0.5,
        min_recommendation_confidence: 0.65,
        max_recommendations_per_category: 15,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = create_default_engine();
        assert_eq!(engine.config().priority_threshold, 0.7);
        assert!(engine.config().enable_data_enhancement);
    }

    #[test]
    fn test_conservative_config() {
        let config = create_conservative_config();
        assert_eq!(config.priority_threshold, 0.8);
        assert_eq!(config.min_recommendation_confidence, 0.85);
        assert!(matches!(
            config.strategy_preference,
            EnhancementStrategy::Conservative
        ));
    }

    #[test]
    fn test_aggressive_config() {
        let config = create_aggressive_config();
        assert_eq!(config.priority_threshold, 0.5);
        assert_eq!(config.min_recommendation_confidence, 0.65);
        assert!(matches!(
            config.strategy_preference,
            EnhancementStrategy::Aggressive
        ));
    }

    #[test]
    fn test_custom_engine() {
        let config = EnhancementConfig {
            enable_automated_improvements: false,
            max_recommendations_per_category: 20,
            ..Default::default()
        };

        let engine = create_engine_with_config(config);
        assert!(!engine.config().enable_automated_improvements);
        assert_eq!(engine.config().max_recommendations_per_category, 20);
    }

    #[test]
    fn test_enhancement_action() {
        let action = EnhancementAction {
            action_id: "test_action".to_string(),
            action_type: EnhancementActionType::DataQualityImprovement,
            timestamp: chrono::Utc::now(),
            status: ActionStatus::Pending,
            metadata: std::collections::HashMap::new(),
        };

        assert_eq!(action.action_id, "test_action");
        assert!(matches!(
            action.action_type,
            EnhancementActionType::DataQualityImprovement
        ));
        assert!(matches!(action.status, ActionStatus::Pending));
    }

    #[test]
    fn test_enhancement_recommendation() {
        let recommendation = EnhancementRecommendation {
            id: "test_rec".to_string(),
            title: "Test Recommendation".to_string(),
            description: "A test recommendation".to_string(),
            category: EnhancementCategory::DataQuality,
            priority: Priority::High,
            confidence: 0.85,
            estimated_impact: 0.7,
            implementation_effort: ImplementationEffort::Medium,
            automated: false,
        };

        assert_eq!(recommendation.id, "test_rec");
        assert_eq!(recommendation.confidence, 0.85);
        assert!(matches!(recommendation.priority, Priority::High));
        assert!(!recommendation.automated);
    }
}
