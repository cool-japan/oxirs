//! Intent Recognition System
//!
//! This module provides intent classification for chat messages, helping the system
//! understand what the user wants to accomplish.

use anyhow::{Context, Result};
use scirs2_stats::distributions::Categorical;
use scirs2_text::classification::{ClassificationConfig, TextClassifier};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// User intent types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentType {
    /// User wants to query data (SELECT, ASK)
    Query,
    /// User wants to explore/browse data
    Exploration,
    /// User wants an explanation
    Explanation,
    /// User wants to insert/modify data
    Modification,
    /// User wants help or guidance
    Help,
    /// User is having a conversation
    Conversation,
    /// User wants to perform aggregation/analytics
    Analytics,
    /// User wants to navigate relationships
    Navigation,
    /// User wants recommendations
    Recommendation,
    /// Unknown/ambiguous intent
    Unknown,
}

impl IntentType {
    /// Get all possible intent types
    pub fn all() -> Vec<IntentType> {
        vec![
            IntentType::Query,
            IntentType::Exploration,
            IntentType::Explanation,
            IntentType::Modification,
            IntentType::Help,
            IntentType::Conversation,
            IntentType::Analytics,
            IntentType::Navigation,
            IntentType::Recommendation,
            IntentType::Unknown,
        ]
    }

    /// Get intent description
    pub fn description(&self) -> &str {
        match self {
            IntentType::Query => "Searching or querying for specific information",
            IntentType::Exploration => "Browsing or exploring available data",
            IntentType::Explanation => "Requesting explanations or understanding",
            IntentType::Modification => "Inserting, updating, or deleting data",
            IntentType::Help => "Asking for help or guidance",
            IntentType::Conversation => "General conversation or chitchat",
            IntentType::Analytics => "Performing aggregation or statistical analysis",
            IntentType::Navigation => "Navigating through entity relationships",
            IntentType::Recommendation => "Requesting recommendations or suggestions",
            IntentType::Unknown => "Intent cannot be determined",
        }
    }
}

/// Intent recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentResult {
    /// Primary intent
    pub primary_intent: IntentType,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Alternative intents with scores
    pub alternatives: Vec<(IntentType, f32)>,
    /// Extracted entities related to intent
    pub entities: Vec<String>,
    /// Intent-specific parameters
    pub parameters: HashMap<String, String>,
}

/// Intent recognition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentRecognitionConfig {
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Number of alternative intents to return
    pub num_alternatives: usize,
    /// Enable entity extraction
    pub enable_entity_extraction: bool,
    /// Enable parameter extraction
    pub enable_parameter_extraction: bool,
    /// Use LLM for ambiguous cases
    pub use_llm_fallback: bool,
}

impl Default for IntentRecognitionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            num_alternatives: 3,
            enable_entity_extraction: true,
            enable_parameter_extraction: true,
            use_llm_fallback: true,
        }
    }
}

/// Intent recognizer using pattern matching and ML
pub struct IntentRecognizer {
    config: IntentRecognitionConfig,
    patterns: HashMap<IntentType, Vec<String>>,
    classifier: Option<TextClassifier>,
}

impl IntentRecognizer {
    /// Create a new intent recognizer
    pub fn new(config: IntentRecognitionConfig) -> Result<Self> {
        let patterns = Self::build_intent_patterns();

        info!(
            "Initialized intent recognizer with {} intent types",
            patterns.len()
        );

        Ok(Self {
            config,
            patterns,
            classifier: None,
        })
    }

    /// Build pattern database for intent matching
    fn build_intent_patterns() -> HashMap<IntentType, Vec<String>> {
        let mut patterns = HashMap::new();

        // Query patterns
        patterns.insert(
            IntentType::Query,
            vec![
                "find", "search", "show", "get", "what", "who", "which", "list", "select", "query",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        // Exploration patterns
        patterns.insert(
            IntentType::Exploration,
            vec![
                "explore", "browse", "discover", "see all", "show me", "look at",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        // Explanation patterns
        patterns.insert(
            IntentType::Explanation,
            vec![
                "explain",
                "why",
                "how",
                "describe",
                "tell me about",
                "what does",
                "what is",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        // Modification patterns
        patterns.insert(
            IntentType::Modification,
            vec![
                "insert", "add", "create", "update", "modify", "delete", "remove",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        // Help patterns
        patterns.insert(
            IntentType::Help,
            vec![
                "help",
                "how do i",
                "how can i",
                "tutorial",
                "guide",
                "documentation",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        // Conversation patterns
        patterns.insert(
            IntentType::Conversation,
            vec!["hello", "hi", "thanks", "thank you", "bye", "goodbye"]
                .into_iter()
                .map(String::from)
                .collect(),
        );

        // Analytics patterns
        patterns.insert(
            IntentType::Analytics,
            vec![
                "count",
                "average",
                "sum",
                "total",
                "statistics",
                "analyze",
                "aggregate",
                "group by",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        // Navigation patterns
        patterns.insert(
            IntentType::Navigation,
            vec![
                "related",
                "connected",
                "linked",
                "associated with",
                "navigate to",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        );

        // Recommendation patterns
        patterns.insert(
            IntentType::Recommendation,
            vec!["recommend", "suggest", "similar", "like this", "related to"]
                .into_iter()
                .map(String::from)
                .collect(),
        );

        patterns
    }

    /// Recognize intent from user message
    pub fn recognize(&self, message: &str) -> Result<IntentResult> {
        debug!(
            "Recognizing intent for message: {}",
            message.chars().take(100).collect::<String>()
        );

        let lowercase = message.to_lowercase();

        // Pattern-based matching
        let mut scores: HashMap<IntentType, f32> = HashMap::new();

        for (intent, patterns) in &self.patterns {
            let mut score = 0.0;
            let pattern_count = patterns.len() as f32;

            for pattern in patterns {
                if lowercase.contains(pattern) {
                    score += 1.0 / pattern_count;
                }
            }

            if score > 0.0 {
                scores.insert(*intent, score);
            }
        }

        // Determine primary intent
        let (primary_intent, confidence) = if let Some((intent, score)) = scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            (*intent, *score)
        } else {
            (IntentType::Unknown, 0.0)
        };

        // Get alternatives
        let mut alternatives: Vec<(IntentType, f32)> = scores
            .iter()
            .filter(|(intent, _)| **intent != primary_intent)
            .map(|(intent, score)| (*intent, *score))
            .collect();

        alternatives.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        alternatives.truncate(self.config.num_alternatives);

        // Extract entities (simplified - basic word extraction)
        let entities = if self.config.enable_entity_extraction {
            self.extract_entities(message)
        } else {
            Vec::new()
        };

        // Extract parameters
        let parameters = if self.config.enable_parameter_extraction {
            self.extract_parameters(message, primary_intent)
        } else {
            HashMap::new()
        };

        debug!(
            "Recognized intent: {:?} (confidence: {:.2})",
            primary_intent, confidence
        );

        Ok(IntentResult {
            primary_intent,
            confidence,
            alternatives,
            entities,
            parameters,
        })
    }

    /// Extract entities from message (simplified implementation)
    fn extract_entities(&self, message: &str) -> Vec<String> {
        // Very basic entity extraction - just extract capitalized words
        // In production, use scirs2-text NER capabilities
        message
            .split_whitespace()
            .filter(|word| {
                word.chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
                    && word.len() > 2
            })
            .map(String::from)
            .collect()
    }

    /// Extract intent-specific parameters
    fn extract_parameters(&self, message: &str, intent: IntentType) -> HashMap<String, String> {
        let mut params = HashMap::new();

        match intent {
            IntentType::Query | IntentType::Analytics => {
                // Extract limit if present
                if let Some(limit) = self.extract_limit(message) {
                    params.insert("limit".to_string(), limit.to_string());
                }

                // Extract time range
                if message.contains("today") {
                    params.insert("time_range".to_string(), "today".to_string());
                } else if message.contains("week") {
                    params.insert("time_range".to_string(), "week".to_string());
                } else if message.contains("month") {
                    params.insert("time_range".to_string(), "month".to_string());
                }
            }
            IntentType::Navigation => {
                // Extract relationship type
                if message.contains("parent") {
                    params.insert("direction".to_string(), "parent".to_string());
                } else if message.contains("child") {
                    params.insert("direction".to_string(), "child".to_string());
                }
            }
            _ => {}
        }

        params
    }

    /// Extract limit from message
    fn extract_limit(&self, message: &str) -> Option<usize> {
        // Look for patterns like "top 10", "first 5", "limit 20"
        let words: Vec<&str> = message.split_whitespace().collect();

        for i in 0..words.len() {
            if words[i] == "top" || words[i] == "first" || words[i] == "limit" {
                if i + 1 < words.len() {
                    if let Ok(num) = words[i + 1].parse::<usize>() {
                        return Some(num);
                    }
                }
            }
        }

        None
    }

    /// Update intent recognizer with training data
    pub fn train(&mut self, examples: Vec<(String, IntentType)>) -> Result<()> {
        info!(
            "Training intent recognizer with {} examples",
            examples.len()
        );

        // Use scirs2-text for training if available
        let config = ClassificationConfig::default();
        let mut classifier = TextClassifier::new(config)?;

        // Train classifier
        let training_data: Vec<(String, String)> = examples
            .into_iter()
            .map(|(text, intent)| (text, format!("{:?}", intent)))
            .collect();

        classifier.train(&training_data)?;
        self.classifier = Some(classifier);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_recognition_query() {
        let recognizer = IntentRecognizer::new(IntentRecognitionConfig::default()).unwrap();
        let result = recognizer
            .recognize("Show me all movies from 2023")
            .unwrap();

        assert_eq!(result.primary_intent, IntentType::Query);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_intent_recognition_explanation() {
        let recognizer = IntentRecognizer::new(IntentRecognitionConfig::default()).unwrap();
        let result = recognizer.recognize("Explain how SPARQL works").unwrap();

        assert_eq!(result.primary_intent, IntentType::Explanation);
    }

    #[test]
    fn test_intent_recognition_analytics() {
        let recognizer = IntentRecognizer::new(IntentRecognitionConfig::default()).unwrap();
        let result = recognizer
            .recognize("Count the total number of users")
            .unwrap();

        assert_eq!(result.primary_intent, IntentType::Analytics);
    }

    #[test]
    fn test_parameter_extraction() {
        let recognizer = IntentRecognizer::new(IntentRecognitionConfig::default()).unwrap();
        let result = recognizer.recognize("Show me the top 10 movies").unwrap();

        assert_eq!(result.parameters.get("limit"), Some(&"10".to_string()));
    }
}
