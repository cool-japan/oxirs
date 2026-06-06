//! Unit tests for the consciousness-inspired computing module.

use super::*;
use crate::query::pattern_optimizer::IndexStats;
use std::sync::Arc;

#[test]
fn test_consciousness_module_creation() {
    let stats = Arc::new(IndexStats::new());
    let consciousness = ConsciousnessModule::new(stats);

    assert_eq!(consciousness.consciousness_level, 0.5);
    assert_eq!(consciousness.emotional_state, EmotionalState::Calm);
    assert_eq!(consciousness.integration_level, 0.3);
    assert!(matches!(
        consciousness.dream_processor.dream_state,
        DreamState::Awake
    ));
    assert!(!consciousness
        .quantum_consciousness
        .consciousness_superposition
        .state_amplitudes
        .is_empty());
    assert!(
        consciousness
            .emotional_learning
            .empathy_engine
            .empathy_level
            > 0.0
    );
}

#[test]
fn test_consciousness_adjustment() {
    let stats = Arc::new(IndexStats::new());
    let mut consciousness = ConsciousnessModule::new(stats);

    // Test positive feedback
    consciousness.adjust_consciousness(0.9);
    assert!(consciousness.consciousness_level > 0.5);
    assert_eq!(consciousness.emotional_state, EmotionalState::Confident);

    // Test negative feedback
    consciousness.adjust_consciousness(0.2);
    assert_eq!(consciousness.emotional_state, EmotionalState::Cautious);
}

#[test]
fn test_emotional_influence() {
    let stats = Arc::new(IndexStats::new());
    let mut consciousness = ConsciousnessModule::new(stats);

    // Base emotional influence should be affected by consciousness and integration levels
    let base_influence = consciousness.emotional_influence();
    assert!(base_influence > 0.8 && base_influence < 1.2); // Calm with modifiers

    consciousness.enter_creative_mode();
    let creative_influence = consciousness.emotional_influence();
    assert!(creative_influence > base_influence); // Creative boost
}

#[test]
fn test_quantum_consciousness_measurement() {
    let stats = Arc::new(IndexStats::new());
    let mut consciousness = ConsciousnessModule::new(stats);

    let measurement = consciousness.quantum_consciousness_measurement();
    assert!(measurement.is_ok());

    let measurement = measurement.expect("measurement should succeed");
    assert!(measurement.probability >= 0.0 && measurement.probability <= 1.0);
    assert!(measurement.fidelity >= 0.0 && measurement.fidelity <= 1.0);
    assert_eq!(consciousness.emotional_state, measurement.measured_state);
}

#[test]
fn test_dream_state_processing() {
    let stats = Arc::new(IndexStats::new());
    let mut consciousness = ConsciousnessModule::new(stats);

    // Enter dream state
    let result = consciousness.enter_dream_state(DreamState::CreativeDreaming);
    assert!(result.is_ok());
    assert!(matches!(
        consciousness.dream_processor.dream_state,
        DreamState::CreativeDreaming
    ));

    // Process dream step
    let step_result = consciousness.process_dream_step();
    assert!(step_result.is_ok());

    // Wake up
    let wake_report = consciousness.wake_up_from_dream();
    assert!(wake_report.is_ok());
    assert!(matches!(
        consciousness.dream_processor.dream_state,
        DreamState::Awake
    ));
}

#[test]
fn test_consciousness_insights() {
    let stats = Arc::new(IndexStats::new());
    let consciousness = ConsciousnessModule::new(stats);

    let patterns = vec![]; // Empty patterns for simplicity
    let insights = consciousness.get_consciousness_insights(&patterns);
    assert!(insights.is_ok());

    let insights = insights.expect("insights should be available");
    assert!(insights.quantum_advantage >= 1.0);
    assert!(insights.consciousness_level >= 0.0 && insights.consciousness_level <= 1.0);
    assert!(insights.integration_level >= 0.0 && insights.integration_level <= 1.0);
    assert!(insights.recommended_approach.confidence_level >= 0.0);
}

#[test]
fn test_consciousness_evolution() {
    let stats = Arc::new(IndexStats::new());
    let mut consciousness = ConsciousnessModule::new(stats);

    let initial_consciousness = consciousness.consciousness_level;

    let feedback = ExperienceFeedback {
        context: "test_experience".to_string(),
        performance_score: 0.9,
        satisfaction_level: 0.8,
        emotional_outcome: EmotionalState::Confident,
        complexity_level: 0.5,
        related_pattern: Some("related_test".to_string()),
        pattern_similarity: 0.7,
    };

    let result = consciousness.evolve_consciousness(&feedback);
    assert!(result.is_ok());

    // High performance should increase consciousness
    assert!(consciousness.consciousness_level >= initial_consciousness);
    assert_eq!(consciousness.emotional_state, EmotionalState::Confident);
}

#[test]
fn test_meta_consciousness_creation() {
    let meta_consciousness = MetaConsciousness::new();

    assert_eq!(meta_consciousness.self_awareness, 0.3);
    assert_eq!(
        meta_consciousness.sync_state,
        IntegrationSyncState::NeedsSync
    );
    assert!(meta_consciousness.component_effectiveness.is_empty());
    assert!(meta_consciousness.performance_history.is_empty());
}

#[test]
fn test_meta_consciousness_effectiveness_tracking() {
    let mut meta_consciousness = MetaConsciousness::new();

    meta_consciousness.update_component_effectiveness("quantum", 0.8);
    meta_consciousness.update_component_effectiveness("emotional", 0.7);

    assert_eq!(
        meta_consciousness.component_effectiveness.get("quantum"),
        Some(&0.8)
    );
    assert_eq!(
        meta_consciousness.component_effectiveness.get("emotional"),
        Some(&0.7)
    );
    assert_eq!(meta_consciousness.performance_history.len(), 2);
    assert!(meta_consciousness.self_awareness > 0.3); // Should have increased
}

#[test]
fn test_consciousness_message_system() {
    let meta_consciousness = MetaConsciousness::new();

    let message = ConsciousnessMessage {
        source: "quantum".to_string(),
        target: "emotional".to_string(),
        message_type: MessageType::QuantumMeasurement,
        content: "measurement_complete".to_string(),
        priority: 0.8,
        timestamp: std::time::Instant::now(),
    };

    let result = meta_consciousness.send_message(message);
    assert!(result.is_ok());

    let messages = meta_consciousness.receive_messages("emotional");
    assert!(messages.is_ok());
    let messages = messages.expect("messages should be available");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].source, "quantum");
    assert_eq!(messages[0].message_type, MessageType::QuantumMeasurement);
}

#[test]
fn test_adaptive_recommendations() {
    let mut meta_consciousness = MetaConsciousness::new();

    // Add some performance history
    meta_consciousness.update_component_effectiveness("quantum", 0.9);
    meta_consciousness.update_component_effectiveness("emotional", 0.8);
    meta_consciousness.update_component_effectiveness("dream", 0.7);

    let recommendations = meta_consciousness.calculate_adaptive_recommendations();

    assert!(recommendations.recommended_consciousness_level >= 0.0);
    assert!(recommendations.recommended_consciousness_level <= 1.0);
    assert!(recommendations.confidence > 0.0);
}

#[test]
fn test_consciousness_integration_with_meta() {
    let stats = Arc::new(IndexStats::new());
    let mut consciousness = ConsciousnessModule::new(stats);
    let mut meta_consciousness = MetaConsciousness::new();

    let result = consciousness.integrate_with_meta_consciousness(&mut meta_consciousness);
    assert!(result.is_ok());

    // Should have updated component effectiveness
    assert!(!meta_consciousness.component_effectiveness.is_empty());

    // Should have performance history
    assert!(!meta_consciousness.performance_history.is_empty());
}

#[test]
fn test_pattern_complexity_calculation() {
    let stats = Arc::new(IndexStats::new());
    let consciousness = ConsciousnessModule::new(stats);

    // Empty patterns should have 0 complexity
    let complexity = consciousness.calculate_pattern_complexity(&[]);
    assert_eq!(complexity, 0.0);

    // Would need actual AlgebraTriplePattern instances for more detailed testing
    // This is a basic structural test
}

#[test]
fn test_adaptive_consciousness_adjustment() {
    let stats = Arc::new(IndexStats::new());
    let mut consciousness = ConsciousnessModule::new(stats);

    let metrics = QueryExecutionMetrics {
        success_rate: 0.9,
        execution_time_improvement: 0.2,
        resource_efficiency: 0.8,
        user_satisfaction: 0.85,
        pattern_similarity: 0.7,
    };

    let initial_consciousness = consciousness.consciousness_level;
    let result = consciousness.adapt_to_query_patterns(&[], &metrics);
    assert!(result.is_ok());

    // High success rate should not decrease consciousness
    assert!(consciousness.consciousness_level >= initial_consciousness);
}

#[test]
fn test_consciousness_query_optimization() {
    let stats = Arc::new(IndexStats::new());
    let consciousness = ConsciousnessModule::new(stats);

    // Create a simple execution plan for testing
    let plan = crate::query::plan::ExecutionPlan::TripleScan {
        pattern: crate::model::pattern::TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        },
    };

    let result = consciousness.optimize_query_with_consciousness(&plan);
    assert!(result.is_ok());

    let optimized = result.expect("should have value");
    assert!(optimized.expected_improvement >= 1.0);
    assert!(optimized.consciousness_metadata.consciousness_level >= 0.0);
    assert!(optimized.consciousness_metadata.integration_level >= 0.0);
}
