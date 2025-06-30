//! Processing utilities for explanations
//!
//! This module contains processors for converting raw explanations into
//! user-friendly formats, including natural language processing.

use std::collections::HashMap;
use crate::Result;
use super::types::*;

/// Natural language processor for converting technical explanations to human-readable text
#[derive(Debug, Clone)]
pub struct NaturalLanguageProcessor {
    templates: HashMap<String, String>,
}

impl NaturalLanguageProcessor {
    pub fn new() -> Self {
        let mut templates = HashMap::new();
        
        // Validation templates
        templates.insert(
            "validation_success".to_string(),
            "The validation succeeded because {reason}. Key factors include: {factors}.".to_string(),
        );
        templates.insert(
            "validation_failure".to_string(),
            "The validation failed because {reason}. Issues identified: {factors}.".to_string(),
        );
        
        // Pattern recognition templates
        templates.insert(
            "pattern_recognition".to_string(), 
            "Pattern recognition identified {count} patterns with {confidence}% confidence. The most significant pattern is {primary_pattern}.".to_string()
        );
        
        // Neural decision templates
        templates.insert(
            "neural_decision".to_string(),
            "The neural network made this decision with {confidence}% confidence based on {reasoning}.".to_string(),
        );
        
        // Quantum pattern templates
        templates.insert(
            "quantum_pattern".to_string(),
            "Quantum pattern analysis identified {count} quantum states with {confidence}% coherence.".to_string(),
        );
        
        // Adaptation logic templates
        templates.insert(
            "adaptation_logic".to_string(),
            "The adaptation system applied {count} optimizations with {confidence}% improvement confidence.".to_string(),
        );

        Self { templates }
    }

    /// Convert a raw explanation to natural language
    pub async fn convert_to_natural_language(
        &mut self,
        explanation: &RawExplanation,
    ) -> Result<String> {
        let template = self.templates.get(&explanation.explanation_type)
            .unwrap_or(&"The AI system made this decision with {confidence}% confidence.".to_string());

        let mut result = template.clone();
        
        // Replace confidence placeholder
        result = result.replace(
            "{confidence}", 
            &format!("{:.0}", explanation.confidence_score * 100.0)
        );
        
        // Replace reasoning placeholder if it exists in metadata
        if let Some(reasoning) = explanation.metadata.get("reasoning") {
            if let Some(reasoning_str) = reasoning.as_str() {
                result = result.replace("{reasoning}", reasoning_str);
            }
        }
        
        // Replace count placeholder if it exists in metadata
        if let Some(count) = explanation.metadata.get("count") {
            if let Some(count_num) = count.as_u64() {
                result = result.replace("{count}", &count_num.to_string());
            }
        }
        
        // Replace factors placeholder if it exists in metadata
        if let Some(factors) = explanation.metadata.get("factors") {
            if let Some(factors_str) = factors.as_str() {
                result = result.replace("{factors}", factors_str);
            }
        }
        
        // Replace reason placeholder if it exists in metadata
        if let Some(reason) = explanation.metadata.get("reason") {
            if let Some(reason_str) = reason.as_str() {
                result = result.replace("{reason}", reason_str);
            }
        }
        
        // Replace primary_pattern placeholder if it exists in metadata
        if let Some(primary_pattern) = explanation.metadata.get("primary_pattern") {
            if let Some(pattern_str) = primary_pattern.as_str() {
                result = result.replace("{primary_pattern}", pattern_str);
            }
        }

        Ok(result)
    }

    /// Generate technical summary from raw explanation
    pub fn generate_technical_summary(&self, explanation: &RawExplanation) -> String {
        format!(
            "Explanation ID: {}\nType: {}\nConfidence: {:.2}\nGeneration Time: {:?}ms\nTechnical Details: {}",
            explanation.explanation_id,
            explanation.explanation_type,
            explanation.confidence_score,
            explanation.generation_time.as_millis(),
            explanation.technical_details
        )
    }

    /// Create visualization elements from explanation data
    pub fn create_visualization_elements(&self, explanation: &RawExplanation) -> Vec<VisualizationElement> {
        let mut elements = Vec::new();
        
        // Create confidence visualization
        elements.push(VisualizationElement {
            element_type: "confidence_meter".to_string(),
            data: serde_json::json!({
                "value": explanation.confidence_score,
                "max": 1.0,
                "color": if explanation.confidence_score > 0.8 { "green" } else if explanation.confidence_score > 0.6 { "yellow" } else { "red" }
            }),
            rendering_hints: {
                let mut hints = HashMap::new();
                hints.insert("width".to_string(), "300".to_string());
                hints.insert("height".to_string(), "50".to_string());
                hints
            },
        });
        
        // Create explanation type visualization
        elements.push(VisualizationElement {
            element_type: "explanation_type".to_string(),
            data: serde_json::json!({
                "type": explanation.explanation_type,
                "icon": match explanation.explanation_type.as_str() {
                    "neural_decision" => "brain",
                    "pattern_recognition" => "search",
                    "validation_reasoning" => "check",
                    "quantum_pattern" => "atom",
                    "adaptation_logic" => "settings",
                    _ => "info"
                }
            }),
            rendering_hints: HashMap::new(),
        });
        
        elements
    }

    /// Extract supporting evidence from metadata
    pub fn extract_evidence(&self, explanation: &RawExplanation) -> Vec<EvidenceItem> {
        let mut evidence = Vec::new();
        
        // Extract evidence from metadata
        for (key, value) in &explanation.metadata {
            if key.contains("evidence") || key.contains("support") {
                evidence.push(EvidenceItem {
                    evidence_type: "metadata".to_string(),
                    description: key.clone(),
                    strength: 0.7, // Default strength
                    source: "system_analysis".to_string(),
                    data: value.clone(),
                });
            }
        }
        
        // Add default evidence if none found
        if evidence.is_empty() {
            evidence.push(EvidenceItem {
                evidence_type: "confidence_score".to_string(),
                description: "High confidence score indicates reliable decision".to_string(),
                strength: explanation.confidence_score,
                source: "confidence_analysis".to_string(),
                data: serde_json::json!({"score": explanation.confidence_score}),
            });
        }
        
        evidence
    }
}