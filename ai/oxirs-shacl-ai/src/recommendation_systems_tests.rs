//! Tests for the recommendation systems module.

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use uuid::Uuid;

    use chrono::Utc;

    use crate::recommendation_systems_engine::RecommendationEngine;
    use crate::recommendation_systems_types::{
        EffortComplexity, EstimatedImpact, ImplementationEffort, Recommendation,
        RecommendationPriority, RecommendationType, RiskLevel,
    };

    #[test]
    fn test_recommendation_engine_creation() {
        let engine = RecommendationEngine::new();
        assert!(engine.recommendation_history.is_empty());
    }

    #[test]
    fn test_recommendation_priority_ordering() {
        let mut recommendations = [
            Recommendation {
                id: Uuid::new_v4(),
                priority: RecommendationPriority::Low,
                confidence: 0.8,
                recommendation_type: RecommendationType::ShapeImprovement,
                title: "Low Priority".to_string(),
                description: "".to_string(),
                rationale: "".to_string(),
                estimated_impact: EstimatedImpact {
                    categories: vec![],
                    quantitative_benefits: HashMap::new(),
                    qualitative_benefits: vec![],
                    potential_risks: vec![],
                    roi_estimate: None,
                    payback_period_months: None,
                },
                implementation_effort: ImplementationEffort {
                    complexity: EffortComplexity::Simple,
                    estimated_hours: 1.0,
                    required_skills: vec![],
                    required_resources: vec![],
                    dependencies: vec![],
                    risk_level: RiskLevel::Low,
                },
                prerequisites: vec![],
                expected_outcomes: vec![],
                implementation_steps: vec![],
                success_metrics: vec![],
                related_recommendations: vec![],
                tags: HashSet::new(),
                created_at: Utc::now(),
                valid_until: None,
                applied: false,
                effectiveness_score: None,
            },
            Recommendation {
                id: Uuid::new_v4(),
                priority: RecommendationPriority::Critical,
                confidence: 0.9,
                recommendation_type: RecommendationType::ShapeImprovement,
                title: "Critical Priority".to_string(),
                description: "".to_string(),
                rationale: "".to_string(),
                estimated_impact: EstimatedImpact {
                    categories: vec![],
                    quantitative_benefits: HashMap::new(),
                    qualitative_benefits: vec![],
                    potential_risks: vec![],
                    roi_estimate: None,
                    payback_period_months: None,
                },
                implementation_effort: ImplementationEffort {
                    complexity: EffortComplexity::Simple,
                    estimated_hours: 1.0,
                    required_skills: vec![],
                    required_resources: vec![],
                    dependencies: vec![],
                    risk_level: RiskLevel::Low,
                },
                prerequisites: vec![],
                expected_outcomes: vec![],
                implementation_steps: vec![],
                success_metrics: vec![],
                related_recommendations: vec![],
                tags: HashSet::new(),
                created_at: Utc::now(),
                valid_until: None,
                applied: false,
                effectiveness_score: None,
            },
        ];

        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));
        assert_eq!(
            recommendations[0].priority,
            RecommendationPriority::Critical
        );
        assert_eq!(recommendations[1].priority, RecommendationPriority::Low);
    }

    #[test]
    fn test_ml_score_calculation() {
        let engine = RecommendationEngine::new();
        let recommendation = Recommendation {
            id: Uuid::new_v4(),
            recommendation_type: RecommendationType::PerformanceOptimization,
            priority: RecommendationPriority::High,
            title: "Test".to_string(),
            description: "".to_string(),
            rationale: "".to_string(),
            confidence: 0.8,
            estimated_impact: EstimatedImpact {
                categories: vec![],
                quantitative_benefits: HashMap::new(),
                qualitative_benefits: vec![],
                potential_risks: vec![],
                roi_estimate: Some(3.0),
                payback_period_months: None,
            },
            implementation_effort: ImplementationEffort {
                complexity: EffortComplexity::Simple,
                estimated_hours: 8.0,
                required_skills: vec![],
                required_resources: vec![],
                dependencies: vec![],
                risk_level: RiskLevel::Low,
            },
            prerequisites: vec![],
            expected_outcomes: vec![],
            implementation_steps: vec![],
            success_metrics: vec![],
            related_recommendations: vec![],
            tags: HashSet::new(),
            created_at: Utc::now(),
            valid_until: None,
            applied: false,
            effectiveness_score: None,
        };

        let score = engine.calculate_ml_score(&recommendation);
        assert!(score > 0.0);
        assert!(score >= recommendation.confidence);
    }

    #[test]
    fn test_recommendation_statistics() {
        let engine = RecommendationEngine::new();
        let stats = engine.get_recommendation_statistics();

        assert_eq!(stats.get("total_recommendations"), Some(&0.0));
        assert_eq!(stats.get("applied_recommendations"), Some(&0.0));
        assert_eq!(stats.get("successful_recommendations"), Some(&0.0));
    }
}
