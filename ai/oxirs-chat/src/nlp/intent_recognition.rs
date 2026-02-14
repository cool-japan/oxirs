//! Intent Recognition System
//!
//! This module provides intent classification for chat messages, helping the system
//! understand what the user wants to accomplish.

use crate::utils::nlp::{
    LexiconSentimentAnalyzer, POSTagger, RuleBasedNER, Tokenizer, WordTokenizer,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

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
    ner: Option<RuleBasedNER>,
    tokenizer: WordTokenizer,
    pos_tagger: Option<POSTagger>,
    sentiment_analyzer: Option<LexiconSentimentAnalyzer>,
    conversation_history: Vec<(String, IntentType)>, // Recent (message, intent) pairs
    intent_embeddings: HashMap<IntentType, Vec<f32>>, // Intent prototype vectors
}

impl IntentRecognizer {
    /// Create a new intent recognizer
    pub fn new(config: IntentRecognitionConfig) -> Result<Self> {
        let patterns = Self::build_intent_patterns();

        // Initialize NER for entity extraction
        let ner = Some(RuleBasedNER::new());

        // Initialize tokenizer
        let tokenizer = WordTokenizer;

        // Initialize POS tagger
        let pos_tagger = Some(POSTagger::new());

        // Semantic similarity not yet implemented
        // let semantic_sim: Option<()> = None;

        // Initialize sentiment analyzer for emotion-aware intent detection
        let sentiment_analyzer = Some(LexiconSentimentAnalyzer::with_basiclexicon());

        // Build intent embeddings (prototype vectors for each intent type)
        let intent_embeddings = Self::build_intent_embeddings(&patterns);

        info!(
            "Initialized advanced intent recognizer with {} intent types, NER: {}, POS: {}",
            patterns.len(),
            ner.is_some(),
            pos_tagger.is_some()
        );

        Ok(Self {
            config,
            patterns,
            ner,
            tokenizer,
            pos_tagger,
            sentiment_analyzer,
            conversation_history: Vec::new(),
            intent_embeddings,
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

    /// Build intent embeddings (prototype vectors for each intent)
    fn build_intent_embeddings(
        patterns: &HashMap<IntentType, Vec<String>>,
    ) -> HashMap<IntentType, Vec<f32>> {
        let mut embeddings = HashMap::new();

        for (intent, keywords) in patterns {
            // Create a simple embedding based on keywords (in production, use word embeddings)
            // For now, use a hash-based approach to create consistent vectors
            let embedding_dim = 128;
            let mut embedding = vec![0.0; embedding_dim];

            for keyword in keywords {
                // Hash keyword to create embedding components
                let hash = keyword.chars().fold(0usize, |acc, c| {
                    acc.wrapping_mul(31).wrapping_add(c as usize)
                });

                for i in 0..embedding_dim {
                    let idx = (hash + i) % embedding_dim;
                    embedding[idx] += 1.0 / keywords.len() as f32;
                }
            }

            // Normalize embedding
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut embedding {
                    *val /= norm;
                }
            }

            embeddings.insert(*intent, embedding);
        }

        embeddings
    }

    /// Update conversation history with recent message and intent
    pub fn update_history(&mut self, message: String, intent: IntentType) {
        self.conversation_history.push((message, intent));

        // Keep only last 10 messages for context
        if self.conversation_history.len() > 10 {
            self.conversation_history.remove(0);
        }
    }

    /// Get contextual hints from conversation history
    fn get_contextual_hints(&self) -> HashMap<IntentType, f32> {
        let mut hints = HashMap::new();

        // Recent intents get higher weight
        for (i, (_, intent)) in self.conversation_history.iter().rev().enumerate() {
            let weight = 1.0 / (i as f32 + 1.0); // Exponential decay
            *hints.entry(*intent).or_insert(0.0) += weight * 0.2; // 20% context influence
        }

        hints
    }

    /// Recognize intent from user message with advanced NLP
    pub fn recognize(&self, message: &str) -> Result<IntentResult> {
        debug!(
            "Recognizing intent for message: {}",
            message.chars().take(100).collect::<String>()
        );

        let lowercase = message.to_lowercase();
        let mut scores: HashMap<IntentType, f32> = HashMap::new();

        // STEP 1: Pattern-based matching (base score)
        for (intent, patterns) in &self.patterns {
            let mut score = 0.0;
            let pattern_count = patterns.len() as f32;

            for pattern in patterns {
                if lowercase.contains(pattern) {
                    score += 1.0 / pattern_count;
                }
            }

            if score > 0.0 {
                scores.insert(*intent, score * 0.4); // 40% weight for pattern matching
            }
        }

        // STEP 2: POS-based feature extraction
        if let Some(ref pos_tagger) = self.pos_tagger {
            if let Ok(tokens) = self.tokenizer.tokenize(message) {
                let tagged = pos_tagger.tag(&tokens);
                {
                    // Boost scores based on POS tags
                    let verb_count = tagged
                        .iter()
                        .filter(|(_, tag)| tag.starts_with("VB"))
                        .count();
                    let noun_count = tagged
                        .iter()
                        .filter(|(_, tag)| tag.starts_with("NN"))
                        .count();
                    let adj_count = tagged
                        .iter()
                        .filter(|(_, tag)| tag.starts_with("JJ"))
                        .count();

                    // Queries typically have more verbs (find, show, get)
                    if verb_count > 0 {
                        *scores.entry(IntentType::Query).or_insert(0.0) += 0.1;
                        *scores.entry(IntentType::Exploration).or_insert(0.0) += 0.05;
                    }

                    // Explanations often have adjectives (what is X)
                    if adj_count > noun_count / 2 {
                        *scores.entry(IntentType::Explanation).or_insert(0.0) += 0.1;
                    }

                    // Analytics has specific numeric patterns
                    if tagged.iter().any(|(_, tag)| tag == "CD") {
                        // CD = Cardinal number
                        *scores.entry(IntentType::Analytics).or_insert(0.0) += 0.1;
                    }
                }
            }
        }

        // STEP 3: Sentiment-based intent adjustment
        if let Some(ref sentiment_analyzer) = self.sentiment_analyzer {
            if let Ok(sentiment) = sentiment_analyzer.analyze(message) {
                // Positive sentiment might indicate help seeking or conversation
                if sentiment.score > 0.5 {
                    *scores.entry(IntentType::Help).or_insert(0.0) += 0.05;
                    *scores.entry(IntentType::Conversation).or_insert(0.0) += 0.05;
                }

                // Negative sentiment might indicate issues or modification intent
                if sentiment.score < -0.3 {
                    *scores.entry(IntentType::Modification).or_insert(0.0) += 0.05;
                }
            }
        }

        // STEP 4: Add contextual hints from conversation history
        let context_hints = self.get_contextual_hints();
        for (intent, context_score) in context_hints {
            *scores.entry(intent).or_insert(0.0) += context_score;
        }

        // STEP 5: ML classifier integration (placeholder for future implementation)
        // Note: ML classifier will be added when scirs2-text classification API is stabilized

        // STEP 6: Determine primary intent and confidence
        let (primary_intent, mut confidence) = if let Some((intent, score)) = scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            (*intent, *score)
        } else {
            (IntentType::Unknown, 0.0)
        };

        // Normalize confidence to 0.0-1.0 range
        confidence = confidence.min(1.0);

        // STEP 7: Get alternatives
        let mut alternatives: Vec<(IntentType, f32)> = scores
            .iter()
            .filter(|(intent, _)| **intent != primary_intent)
            .map(|(intent, score)| (*intent, *score))
            .collect();

        alternatives
            .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        alternatives.truncate(self.config.num_alternatives);

        // STEP 8: Extract entities using advanced NER
        let entities = if self.config.enable_entity_extraction {
            self.extract_entities_advanced(message)
        } else {
            Vec::new()
        };

        // STEP 9: Extract parameters
        let parameters = if self.config.enable_parameter_extraction {
            self.extract_parameters(message, primary_intent)
        } else {
            HashMap::new()
        };

        debug!(
            "Recognized intent: {:?} (confidence: {:.2}, alternatives: {})",
            primary_intent,
            confidence,
            alternatives.len()
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

    /// Advanced entity extraction using scirs2-text NER
    fn extract_entities_advanced(&self, message: &str) -> Vec<String> {
        let mut entities = Vec::new();

        // Use NER if available
        if let Some(ref ner) = self.ner {
            if let Ok(extracted) = ner.extract_entities(message) {
                for entity in extracted {
                    entities.push(entity.text);
                }
                return entities;
            } else {
                warn!("NER extraction failed, falling back to basic extraction");
            }
        }

        // Fallback to basic extraction
        self.extract_entities(message)
    }

    /// Extract intent-specific parameters
    fn extract_parameters(&self, message: &str, intent: IntentType) -> HashMap<String, String> {
        let mut params = HashMap::new();

        match intent {
            IntentType::Query | IntentType::Analytics | IntentType::Exploration => {
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
            if (words[i] == "top" || words[i] == "first" || words[i] == "limit")
                && i + 1 < words.len()
            {
                if let Ok(num) = words[i + 1].parse::<usize>() {
                    return Some(num);
                }
            }
        }

        None
    }

    /// Update intent recognizer with training data
    pub fn train(&mut self, examples: Vec<(String, IntentType)>) -> Result<()> {
        info!(
            "Training intent recognizer with {} examples (training currently not implemented)",
            examples.len()
        );

        // TODO: Implement training when scirs2-text classification API is stabilized
        // For now, pattern-based matching is sufficient

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_recognition_query() {
        let recognizer = IntentRecognizer::new(IntentRecognitionConfig::default()).unwrap();
        // "Show me" is an exploration pattern, more specific than just "show"
        let result = recognizer
            .recognize("Show me all movies from 2023")
            .unwrap();

        assert_eq!(result.primary_intent, IntentType::Exploration);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_intent_recognition_specific_query() {
        let recognizer = IntentRecognizer::new(IntentRecognitionConfig::default()).unwrap();
        // "Find" is a query pattern
        let result = recognizer
            .recognize("Find movies directed by Christopher Nolan")
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
