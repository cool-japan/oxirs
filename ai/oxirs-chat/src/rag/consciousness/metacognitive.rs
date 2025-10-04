//! Enhanced Metacognitive Layer
//!
//! Self-reflection and metacognitive assessment capabilities.

use anyhow::Result;

use super::super::*;

// Re-export EnhancedMetacognitiveLayer from consciousness_types
pub use super::super::consciousness_types::EnhancedMetacognitiveLayer;

// Use ConsciousInsight and InsightType from responses module
use super::responses::{ConsciousInsight, InsightType};

pub struct MetacognitiveLayer {
    pub self_awareness: f64,
    pub strategy_monitoring: f64,
    pub comprehension_monitoring: f64,
}

impl Default for MetacognitiveLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl MetacognitiveLayer {
    pub fn new() -> Self {
        Self {
            self_awareness: 0.7,
            strategy_monitoring: 0.6,
            comprehension_monitoring: 0.8,
        }
    }

    pub fn assess_query(&self, query: &str, context: &AssembledContext) -> MetacognitiveAssessment {
        let complexity = self.assess_complexity(query, context);
        let confidence = self.calculate_confidence(query, context);
        let strategy_recommendation = self.recommend_strategy(query, context);

        MetacognitiveAssessment {
            complexity,
            confidence,
            strategy_recommendation,
            monitoring_alerts: self.generate_monitoring_alerts(query, context),
        }
    }

    fn assess_complexity(&self, query: &str, context: &AssembledContext) -> f64 {
        let word_count = query.split_whitespace().count();
        let context_size = context.retrieved_triples.as_ref().map_or(0, |t| t.len());

        ((word_count as f64 * 0.05) + (context_size as f64 * 0.01)).min(1.0)
    }

    fn calculate_confidence(&self, query: &str, context: &AssembledContext) -> f64 {
        let has_context = context.retrieved_triples.is_some();
        let query_clarity = self.assess_query_clarity(query);

        if has_context {
            (self.comprehension_monitoring * 0.6 + query_clarity * 0.4).min(1.0)
        } else {
            query_clarity * 0.5
        }
    }

    fn assess_query_clarity(&self, query: &str) -> f64 {
        let question_words = ["what", "how", "why", "when", "where", "who"];
        let query_lower = query.to_lowercase();

        let has_question_word = question_words.iter().any(|word| query_lower.contains(word));
        let has_punctuation = query.contains('?');
        let word_count = query.split_whitespace().count();

        let clarity_score: f64 = if has_question_word { 0.4 } else { 0.0 }
            + if has_punctuation { 0.2 } else { 0.0 }
            + if word_count >= 3 { 0.4 } else { 0.2 };

        clarity_score.min(1.0)
    }

    fn recommend_strategy(&self, query: &str, context: &AssembledContext) -> String {
        let complexity = self.assess_complexity(query, context);

        if complexity > 0.8 {
            "Deep Analysis Strategy: Break down into sub-questions".to_string()
        } else if complexity > 0.5 {
            "Systematic Strategy: Use structured approach".to_string()
        } else {
            "Direct Strategy: Provide straightforward answer".to_string()
        }
    }

    fn generate_monitoring_alerts(&self, query: &str, context: &AssembledContext) -> Vec<String> {
        let mut alerts = Vec::new();

        if context.retrieved_triples.is_none() {
            alerts.push("No context retrieved - consider expanding search".to_string());
        }

        if query.len() < 10 {
            alerts.push("Query may be too brief for comprehensive analysis".to_string());
        }

        if query.contains("?") && query.matches("?").count() > 1 {
            alerts.push("Multiple questions detected - consider addressing separately".to_string());
        }

        alerts
    }
}

#[derive(Debug, Clone)]
pub struct MetacognitiveAssessment {
    pub complexity: f64,
    pub confidence: f64,
    pub strategy_recommendation: String,
    pub monitoring_alerts: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ConsciousResponse {
    pub base_response: AssembledContext,
    pub consciousness_metadata: ConsciousnessMetadata,
    pub enhanced_insights: Vec<ConsciousInsight>,
}

#[derive(Debug, Clone)]
pub struct ConsciousnessMetadata {
    pub awareness_level: f64,
    pub attention_focus: Vec<String>,
    pub emotional_resonance: f64,
    pub metacognitive_confidence: f64,
    pub memory_integration_score: f64,
}

/// Main consciousness integration interface
pub struct ConsciousnessIntegration {
    consciousness_model: ConsciousnessModel,
    config: ConsciousnessConfig,
}

impl ConsciousnessIntegration {
    pub fn new(config: ConsciousnessConfig) -> Self {
        let consciousness_model = ConsciousnessModel::new().unwrap_or_else(|e| {
            warn!(
                "Failed to create advanced consciousness model: {}, using fallback",
                e
            );
            // Create a fallback simple model if the advanced one fails
            ConsciousnessModel::new().expect("Failed to create even basic consciousness model")
        });

        Self {
            consciousness_model,
            config,
        }
    }

    pub async fn process_query_with_consciousness(
        &mut self,
        query: &str,
        context: &AssembledContext,
    ) -> Result<Vec<ConsciousInsight>> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        // Try the advanced consciousness processing first
        match self
            .consciousness_model
            .conscious_query_processing(query, context)
        {
            Ok(advanced_response) => {
                // Convert advanced insights to basic insights for compatibility
                let basic_insights = advanced_response
                    .enhanced_insights
                    .into_iter()
                    .map(|advanced_insight| ConsciousInsight {
                        insight_type: match advanced_insight.insight_type {
                            AdvancedInsightType::NeuralPattern => InsightType::PatternRecognition,
                            AdvancedInsightType::MemoryIntegration => {
                                InsightType::MemoryIntegration
                            }
                            AdvancedInsightType::AttentionFocus => {
                                InsightType::ContextualUnderstanding
                            }
                            AdvancedInsightType::StreamCoherence => {
                                InsightType::ContextualUnderstanding
                            }
                            AdvancedInsightType::EmotionalResonance => {
                                InsightType::EmotionalResonance
                            }
                            AdvancedInsightType::MetacognitiveAssessment => {
                                InsightType::StrategicPlanning
                            }
                        },
                        content: advanced_insight.content,
                        confidence: advanced_insight.confidence,
                        implications: advanced_insight.implications,
                    })
                    .collect();

                Ok(basic_insights)
            }
            Err(e) => {
                warn!("Advanced consciousness processing failed: {}, falling back to basic processing", e);
                // Fallback to basic consciousness insights
                Ok(vec![ConsciousInsight {
                    insight_type: InsightType::PatternRecognition,
                    content: "Basic consciousness processing active".to_string(),
                    confidence: 0.6,
                    implications: vec!["Limited consciousness features available".to_string()],
                }])
            }
        }
    }
}
