//! Explainable AI module for SHACL-AI interpretability
//!
//! This module provides comprehensive explainability and interpretability capabilities
//! for the SHACL-AI system, enabling users to understand how AI decisions are made,
//! why certain patterns are recognized, and how validation outcomes are determined.

pub mod analyzers;
pub mod explainers;
pub mod processors;
pub mod traits;
pub mod types;

// Re-export key types and traits for easy access
pub use analyzers::*;
pub use explainers::*;
pub use processors::*;
pub use traits::{DecisionTracker, ExplanationGenerator, InterpretabilityAnalyzer};
pub use types::*;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

use oxirs_core::{
    model::{NamedNode, Term, Triple},
    Store,
};
use oxirs_shacl::{
    constraints::*, Shape, ShapeId, ValidationConfig, ValidationReport, ValidationViolation,
};

use crate::neural_patterns::{NeuralPattern, NeuralPatternRecognizer};
use crate::quantum_neural_patterns::{QuantumNeuralPatternRecognizer, QuantumPattern};
use crate::self_adaptive_ai::{AdaptationResult, PerformanceMetrics};
use crate::{Result, ShaclAiError};

/// Explainable AI engine for providing interpretability and transparency
#[derive(Debug)]
pub struct ExplainableAI {
    /// Explanation generators for different AI components
    explanation_generators: Arc<RwLock<HashMap<String, Box<dyn ExplanationGenerator>>>>,
    /// Interpretability analyzers
    interpretability_analyzers: Arc<RwLock<HashMap<String, Box<dyn InterpretabilityAnalyzer>>>>,
    /// Decision trackers for audit trails
    decision_trackers: Arc<RwLock<HashMap<String, Box<dyn DecisionTracker>>>>,
    /// Explanation cache for performance
    explanation_cache: Arc<RwLock<HashMap<String, CachedExplanation>>>,
    /// Configuration
    config: ExplainableAIConfig,
    /// Natural language processor
    nlp_processor: Arc<Mutex<NaturalLanguageProcessor>>,
}

impl ExplainableAI {
    /// Create a new explainable AI system
    pub fn new(config: ExplainableAIConfig) -> Self {
        let mut generators: HashMap<String, Box<dyn ExplanationGenerator>> = HashMap::new();
        let mut analyzers: HashMap<String, Box<dyn InterpretabilityAnalyzer>> = HashMap::new();

        // Register explanation generators
        generators.insert(
            "neural_decisions".to_string(),
            Box::new(NeuralDecisionExplainer::new()),
        );
        generators.insert(
            "pattern_recognition".to_string(),
            Box::new(PatternRecognitionExplainer::new()),
        );
        generators.insert(
            "validation_reasoning".to_string(),
            Box::new(ValidationReasoningExplainer::new()),
        );
        generators.insert(
            "quantum_patterns".to_string(),
            Box::new(QuantumPatternExplainer::new()),
        );
        generators.insert(
            "adaptation_logic".to_string(),
            Box::new(AdaptationLogicExplainer::new()),
        );

        // Register interpretability analyzers
        analyzers.insert(
            "feature_importance".to_string(),
            Box::new(FeatureImportanceAnalyzer::new()),
        );
        analyzers.insert(
            "attention_analysis".to_string(),
            Box::new(AttentionAnalyzer::new()),
        );
        analyzers.insert(
            "decision_paths".to_string(),
            Box::new(DecisionPathAnalyzer::new()),
        );
        analyzers.insert(
            "model_behavior".to_string(),
            Box::new(ModelBehaviorAnalyzer::new()),
        );
        analyzers.insert(
            "counterfactual".to_string(),
            Box::new(CounterfactualAnalyzer::new()),
        );

        Self {
            explanation_generators: Arc::new(RwLock::new(generators)),
            interpretability_analyzers: Arc::new(RwLock::new(analyzers)),
            decision_trackers: Arc::new(RwLock::new(HashMap::new())),
            explanation_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            nlp_processor: Arc::new(Mutex::new(NaturalLanguageProcessor::new())),
        }
    }

    /// Get an explanation generator by name
    pub async fn get_explanation_generator(
        &self,
        name: &str,
    ) -> Result<Box<dyn ExplanationGenerator>> {
        let generators = self.explanation_generators.read().await;
        if let Some(generator) = generators.get(name) {
            Ok(generator.clone_box())
        } else {
            Err(
                ShaclAiError::Configuration(format!("Explanation generator '{}' not found", name))
                    .into(),
            )
        }
    }

    /// Get an interpretability analyzer by name
    pub async fn get_interpretability_analyzer(
        &self,
        name: &str,
    ) -> Result<Box<dyn InterpretabilityAnalyzer>> {
        let analyzers = self.interpretability_analyzers.read().await;
        if let Some(analyzer) = analyzers.get(name) {
            Ok(analyzer.clone_box())
        } else {
            Err(ShaclAiError::Configuration(format!(
                "Interpretability analyzer '{}' not found",
                name
            ))
            .into())
        }
    }

    /// Generate explanation for any input data
    pub async fn explain(
        &self,
        generator_type: &str,
        data: &ExplanationData,
    ) -> Result<ProcessedExplanation> {
        // Check cache first if enabled
        if self.config.cache_explanations {
            let cache_key = format!("{}_{}", generator_type, data.input_type);
            let cache = self.explanation_cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.explanation.clone());
            }
        }

        // Generate raw explanation
        let generator = self.get_explanation_generator(generator_type).await?;
        let raw_explanation = generator.generate_explanation(data).await?;

        // Process into final format
        let processed = self.process_explanation(&raw_explanation).await?;

        // Cache if enabled
        if self.config.cache_explanations {
            let cache_key = format!("{}_{}", generator_type, data.input_type);
            let mut cache = self.explanation_cache.write().await;
            cache.insert(
                cache_key,
                CachedExplanation {
                    explanation: processed.clone(),
                    cache_timestamp: SystemTime::now(),
                    access_count: 1,
                    last_accessed: SystemTime::now(),
                },
            );
        }

        Ok(processed)
    }

    /// Analyze feature importance for given data
    pub async fn analyze_feature_importance(
        &self,
        data: &ExplanationData,
    ) -> Result<FeatureImportanceAnalysis> {
        let analyzer = self
            .get_interpretability_analyzer("feature_importance")
            .await?;
        analyzer.analyze(data).await
    }

    /// Process a raw explanation into a user-friendly format
    async fn process_explanation(&self, raw: &RawExplanation) -> Result<ProcessedExplanation> {
        let mut processor = self.nlp_processor.lock().await;

        // Generate natural language if enabled
        let natural_language = if self.config.enable_natural_language {
            Some(processor.convert_to_natural_language(raw).await?)
        } else {
            None
        };

        // Generate technical summary
        let technical_summary = processor.generate_technical_summary(raw);

        // Create visualization elements if enabled
        let visual_elements = if self.config.generate_visualizations {
            processor.create_visualization_elements(raw)
        } else {
            Vec::new()
        };

        // Extract supporting evidence
        let supporting_evidence = processor.extract_evidence(raw);

        Ok(ProcessedExplanation {
            explanation_id: raw.explanation_id,
            natural_language,
            technical_summary,
            visual_elements,
            confidence_score: raw.confidence_score,
            supporting_evidence,
            alternative_explanations: Vec::new(), // Could be populated based on config
            compliance_info: if self.config.compliance_mode {
                Some(ComplianceInfo {
                    regulation_type: "AI_TRANSPARENCY".to_string(),
                    compliance_level: "FULL".to_string(),
                    audit_trail: vec![format!("Explanation generated: {}", raw.explanation_id)],
                    verification_timestamp: SystemTime::now(),
                })
            } else {
                None
            },
        })
    }

    /// Clear explanation cache
    pub async fn clear_cache(&self) {
        let mut cache = self.explanation_cache.write().await;
        cache.clear();
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> HashMap<String, u64> {
        let cache = self.explanation_cache.read().await;
        let mut stats = HashMap::new();
        stats.insert("total_entries".to_string(), cache.len() as u64);

        let total_access_count: u64 = cache.values().map(|c| c.access_count).sum();
        stats.insert("total_access_count".to_string(), total_access_count);

        stats
    }
}

/// Simple decision tracker implementation
#[derive(Debug, Clone)]
pub struct SimpleDecisionTracker {
    history: Vec<DecisionContext>,
}

impl SimpleDecisionTracker {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }
}

impl DecisionTracker for SimpleDecisionTracker {
    fn track_decision(&mut self, decision: DecisionContext) {
        self.history.push(decision);
    }

    fn get_history(&self) -> Vec<DecisionContext> {
        self.history.clone()
    }

    fn clear_history(&mut self) {
        self.history.clear();
    }
}
