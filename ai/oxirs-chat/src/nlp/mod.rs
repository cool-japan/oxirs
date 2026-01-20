//! Natural Language Processing Module
//!
//! This module provides comprehensive NLP capabilities for the OxiRS Chat system,
//! including intent recognition, entity extraction, coreference resolution, and sentiment analysis.
//!
//! ## Features
//!
//! - **Intent Recognition**: Classify user queries into intents (Query, Exploration, Analytics, etc.)
//! - **Entity Extraction**: Extract named entities, dates, numbers, RDF resources from text
//! - **Coreference Resolution**: Resolve pronouns and references across multi-turn conversations
//! - **Sentiment Analysis**: Analyze sentiment and emotions in user messages
//!
//! ## Architecture
//!
//! The NLP pipeline leverages:
//! - scirs2-text for advanced text processing
//! - scirs2-neural for ML-based analysis
//! - Rule-based approaches for deterministic extraction
//! - Hybrid ML+rule-based methods for robustness
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use oxirs_chat::nlp::{IntentRecognizer, IntentRecognitionConfig, EntityExtractor, EntityExtractionConfig};
//!
//! # fn example() -> anyhow::Result<()> {
//! // Intent recognition
//! let intent_recognizer = IntentRecognizer::new(IntentRecognitionConfig::default())?;
//! let intent_result = intent_recognizer.recognize("Show me all movies from 2023")?;
//! println!("Intent: {:?}", intent_result.primary_intent);
//!
//! // Entity extraction
//! let entity_extractor = EntityExtractor::new(EntityExtractionConfig::default())?;
//! let entities = entity_extractor.extract("Contact support@example.com")?;
//! println!("Found {} entities", entities.len());
//! # Ok(())
//! # }
//! ```

pub mod coreference;
pub mod entity_extraction;
pub mod intent_recognition;
pub mod sentiment_analysis;

// Re-export commonly used types
pub use coreference::{CoreferenceChain, CoreferenceConfig, CoreferenceResolver, Mention};
pub use entity_extraction::{EntityExtractionConfig, EntityExtractor, EntityType, ExtractedEntity};
pub use intent_recognition::{IntentRecognitionConfig, IntentRecognizer, IntentResult, IntentType};
pub use sentiment_analysis::{
    Emotion, SentimentAnalyzer, SentimentConfig, SentimentPolarity, SentimentResult,
};

/// Complete NLP pipeline for processing user messages
pub struct NLPPipeline {
    intent_recognizer: IntentRecognizer,
    entity_extractor: EntityExtractor,
    coreference_resolver: CoreferenceResolver,
    sentiment_analyzer: SentimentAnalyzer,
}

impl NLPPipeline {
    /// Create a new NLP pipeline with default configuration
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            intent_recognizer: IntentRecognizer::new(IntentRecognitionConfig::default())?,
            entity_extractor: EntityExtractor::new(EntityExtractionConfig::default())?,
            coreference_resolver: CoreferenceResolver::new(CoreferenceConfig::default())?,
            sentiment_analyzer: SentimentAnalyzer::new(SentimentConfig::default())?,
        })
    }

    /// Create a new NLP pipeline with custom configuration
    pub fn with_config(
        intent_config: IntentRecognitionConfig,
        entity_config: EntityExtractionConfig,
        coref_config: CoreferenceConfig,
        sentiment_config: SentimentConfig,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            intent_recognizer: IntentRecognizer::new(intent_config)?,
            entity_extractor: EntityExtractor::new(entity_config)?,
            coreference_resolver: CoreferenceResolver::new(coref_config)?,
            sentiment_analyzer: SentimentAnalyzer::new(sentiment_config)?,
        })
    }

    /// Process a message through the complete NLP pipeline
    pub fn process(&mut self, message_id: String, text: &str) -> anyhow::Result<NLPResult> {
        // Add message to coreference history
        self.coreference_resolver
            .add_message(message_id.clone(), text.to_string());

        // Run all NLP components
        let intent = self.intent_recognizer.recognize(text)?;
        let entities = self.entity_extractor.extract(text)?;
        let coreferences = self.coreference_resolver.resolve(&message_id)?;
        let sentiment = self.sentiment_analyzer.analyze(text)?;

        // Update intent history for context
        self.intent_recognizer
            .update_history(text.to_string(), intent.primary_intent);

        Ok(NLPResult {
            intent,
            entities,
            coreferences,
            sentiment,
        })
    }

    /// Get mutable reference to intent recognizer for training
    pub fn intent_recognizer_mut(&mut self) -> &mut IntentRecognizer {
        &mut self.intent_recognizer
    }

    /// Get mutable reference to coreference resolver
    pub fn coreference_resolver_mut(&mut self) -> &mut CoreferenceResolver {
        &mut self.coreference_resolver
    }
}

/// Complete NLP analysis result
#[derive(Debug, Clone)]
pub struct NLPResult {
    pub intent: IntentResult,
    pub entities: Vec<ExtractedEntity>,
    pub coreferences: Vec<CoreferenceChain>,
    pub sentiment: SentimentResult,
}

impl Default for NLPPipeline {
    fn default() -> Self {
        Self::new().expect("Failed to create default NLP pipeline")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nlp_pipeline_creation() {
        let pipeline = NLPPipeline::new();
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_nlp_pipeline_processing() {
        let mut pipeline = NLPPipeline::new().unwrap();
        let result = pipeline.process("msg1".to_string(), "Show me all movies from 2023");
        assert!(result.is_ok());

        let nlp_result = result.unwrap();
        assert_eq!(nlp_result.intent.primary_intent, IntentType::Exploration);
    }
}
