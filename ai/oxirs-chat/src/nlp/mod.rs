//! Natural Language Processing Module
//!
//! This module provides comprehensive NLP capabilities for chat including:
//! - Intent recognition
//! - Sentiment analysis
//! - Entity extraction
//! - Coreference resolution

pub mod coreference;
pub mod entity_extraction;
pub mod intent_recognition;
pub mod sentiment_analysis;

// Re-export main types
pub use coreference::{CoreferenceChain, CoreferenceConfig, CoreferenceResolver};
pub use entity_extraction::{EntityExtractionConfig, EntityExtractor, EntityType, ExtractedEntity};
pub use intent_recognition::{IntentRecognitionConfig, IntentRecognizer, IntentResult, IntentType};
pub use sentiment_analysis::{
    SentimentAnalyzer, SentimentConfig, SentimentPolarity, SentimentResult,
};
