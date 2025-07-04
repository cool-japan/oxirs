//! Text content validators

use std::collections::HashMap;
use async_trait::async_trait;

use crate::{Result, ShaclAiError};
use super::types::*;
use super::traits::*;

/// Natural language text validator
#[derive(Debug)]
pub struct NaturalLanguageValidator;

impl NaturalLanguageValidator {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl TextValidator for NaturalLanguageValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let text = String::from_utf8_lossy(&content.data);
        
        let is_valid = !text.is_empty() && text.len() > 10;
        let confidence = if is_valid { 0.9 } else { 0.1 };
        
        let mut details = HashMap::new();
        details.insert("text_length".to_string(), text.len().to_string());
        details.insert("word_count".to_string(), text.split_whitespace().count().to_string());
        
        Ok(Some(ValidationResult {
            is_valid,
            confidence,
            error_message: if is_valid { None } else { Some("Text is too short or empty".to_string()) },
            details,
        }))
    }

    fn name(&self) -> &str {
        "natural_language"
    }

    fn description(&self) -> &str {
        "Validates natural language text content"
    }
}

/// Sentiment analysis validator
#[derive(Debug)]
pub struct SentimentValidator;

impl SentimentValidator {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl TextValidator for SentimentValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let text = String::from_utf8_lossy(&content.data);
        
        // Simple sentiment analysis (in real implementation, use proper sentiment analysis)
        let sentiment_score = self.analyze_sentiment(&text);
        let is_valid = sentiment_score.abs() <= 1.0;
        
        let mut details = HashMap::new();
        details.insert("sentiment_score".to_string(), sentiment_score.to_string());
        details.insert("sentiment_label".to_string(), self.get_sentiment_label(sentiment_score));
        
        Ok(Some(ValidationResult {
            is_valid,
            confidence: 0.8,
            error_message: if is_valid { None } else { Some("Invalid sentiment score".to_string()) },
            details,
        }))
    }

    fn name(&self) -> &str {
        "sentiment"
    }

    fn description(&self) -> &str {
        "Analyzes sentiment of text content"
    }
}

impl SentimentValidator {
    fn analyze_sentiment(&self, text: &str) -> f64 {
        // Simplified sentiment analysis
        let positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"];
        let negative_words = ["bad", "terrible", "awful", "horrible", "worst", "hate"];
        
        let lowercase_text = text.to_lowercase();
        let words: Vec<&str> = lowercase_text.split_whitespace().collect();
        let positive_count = words.iter().filter(|word| positive_words.contains(word)).count() as f64;
        let negative_count = words.iter().filter(|word| negative_words.contains(word)).count() as f64;
        
        if positive_count + negative_count == 0.0 {
            0.0 // Neutral
        } else {
            (positive_count - negative_count) / (positive_count + negative_count)
        }
    }
    
    fn get_sentiment_label(&self, score: f64) -> String {
        if score > 0.3 {
            "positive".to_string()
        } else if score < -0.3 {
            "negative".to_string()
        } else {
            "neutral".to_string()
        }
    }
}

/// Language detection validator
#[derive(Debug)]
pub struct LanguageDetectionValidator;

impl LanguageDetectionValidator {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl TextValidator for LanguageDetectionValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let text = String::from_utf8_lossy(&content.data);
        
        let detected_language = self.detect_language(&text);
        let confidence = self.calculate_language_confidence(&text, &detected_language);
        let is_valid = confidence > 0.5;
        
        let mut details = HashMap::new();
        details.insert("detected_language".to_string(), detected_language);
        details.insert("confidence".to_string(), confidence.to_string());
        
        Ok(Some(ValidationResult {
            is_valid,
            confidence,
            error_message: if is_valid { None } else { Some("Language detection confidence too low".to_string()) },
            details,
        }))
    }

    fn name(&self) -> &str {
        "language_detection"
    }

    fn description(&self) -> &str {
        "Detects the language of text content"
    }
}

impl LanguageDetectionValidator {
    fn detect_language(&self, text: &str) -> String {
        // Simplified language detection
        let english_indicators = ["the", "and", "is", "are", "was", "were", "have", "has"];
        let spanish_indicators = ["el", "la", "los", "las", "y", "es", "son", "fue", "fueron"];
        let french_indicators = ["le", "la", "les", "et", "est", "sont", "était", "étaient"];
        
        let lowercase_text = text.to_lowercase();
        let words: Vec<&str> = lowercase_text.split_whitespace().collect();
        
        let english_score = words.iter().filter(|word| english_indicators.contains(word)).count();
        let spanish_score = words.iter().filter(|word| spanish_indicators.contains(word)).count();
        let french_score = words.iter().filter(|word| french_indicators.contains(word)).count();
        
        if english_score >= spanish_score && english_score >= french_score {
            "en".to_string()
        } else if spanish_score >= french_score {
            "es".to_string()
        } else {
            "fr".to_string()
        }
    }
    
    fn calculate_language_confidence(&self, text: &str, _language: &str) -> f64 {
        // Simplified confidence calculation
        let word_count = text.split_whitespace().count() as f64;
        (word_count / (word_count + 10.0)).max(0.1).min(0.95)
    }
}

/// Entity extraction validator
#[derive(Debug)]
pub struct EntityExtractionValidator;

impl EntityExtractionValidator {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl TextValidator for EntityExtractionValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let text = String::from_utf8_lossy(&content.data);
        
        let entities = self.extract_entities(&text);
        let is_valid = true; // Entity extraction is always valid, just informational
        
        let mut details = HashMap::new();
        details.insert("entity_count".to_string(), entities.len().to_string());
        details.insert("entities".to_string(), entities.join(", "));
        
        Ok(Some(ValidationResult {
            is_valid,
            confidence: 0.7,
            error_message: None,
            details,
        }))
    }

    fn name(&self) -> &str {
        "entity_extraction"
    }

    fn description(&self) -> &str {
        "Extracts named entities from text content"
    }
}

impl EntityExtractionValidator {
    fn extract_entities(&self, text: &str) -> Vec<String> {
        // Simplified entity extraction (just find capitalized words)
        let mut entities = Vec::new();
        
        for word in text.split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphabetic());
            if clean_word.len() > 2 && clean_word.chars().next().unwrap().is_uppercase() {
                entities.push(clean_word.to_string());
            }
        }
        
        entities.sort();
        entities.dedup();
        entities
    }
}

/// Text quality validator
#[derive(Debug)]
pub struct TextQualityValidator {
    min_length: usize,
    max_length: usize,
    require_punctuation: bool,
}

impl TextQualityValidator {
    pub fn new() -> Self {
        Self {
            min_length: 10,
            max_length: 100000,
            require_punctuation: false,
        }
    }
    
    pub fn with_length_limits(min_length: usize, max_length: usize) -> Self {
        Self {
            min_length,
            max_length,
            require_punctuation: false,
        }
    }
    
    pub fn with_punctuation_requirement(mut self, require: bool) -> Self {
        self.require_punctuation = require;
        self
    }
}

#[async_trait]
impl TextValidator for TextQualityValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let text = String::from_utf8_lossy(&content.data);
        
        let mut is_valid = true;
        let mut issues = Vec::new();
        let mut details = HashMap::new();
        
        // Check length
        if text.len() < self.min_length {
            is_valid = false;
            issues.push(format!("Text too short: {} < {}", text.len(), self.min_length));
        }
        
        if text.len() > self.max_length {
            is_valid = false;
            issues.push(format!("Text too long: {} > {}", text.len(), self.max_length));
        }
        
        // Check punctuation if required
        if self.require_punctuation && !text.chars().any(|c| c.is_ascii_punctuation()) {
            is_valid = false;
            issues.push("Text lacks punctuation".to_string());
        }
        
        details.insert("text_length".to_string(), text.len().to_string());
        details.insert("word_count".to_string(), text.split_whitespace().count().to_string());
        details.insert("issues".to_string(), issues.join("; "));
        
        let confidence = if is_valid { 0.9 } else { 0.3 };
        
        Ok(Some(ValidationResult {
            is_valid,
            confidence,
            error_message: if issues.is_empty() { None } else { Some(issues.join("; ")) },
            details,
        }))
    }

    fn name(&self) -> &str {
        "text_quality"
    }

    fn description(&self) -> &str {
        "Validates text quality based on length and content requirements"
    }
}

/// Profanity detection validator
#[derive(Debug)]
pub struct ProfanityValidator {
    profanity_list: Vec<String>,
    strict_mode: bool,
}

impl ProfanityValidator {
    pub fn new() -> Self {
        Self {
            profanity_list: vec![
                "spam".to_string(),
                "abuse".to_string(),
                // Add more words as needed
            ],
            strict_mode: false,
        }
    }
    
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }
    
    pub fn with_custom_list(mut self, words: Vec<String>) -> Self {
        self.profanity_list = words;
        self
    }
}

#[async_trait]
impl TextValidator for ProfanityValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let text = String::from_utf8_lossy(&content.data).to_lowercase();
        
        let mut found_profanity = Vec::new();
        
        for word in &self.profanity_list {
            if text.contains(word) {
                found_profanity.push(word.clone());
            }
        }
        
        let is_valid = found_profanity.is_empty() || !self.strict_mode;
        let confidence = if found_profanity.is_empty() { 0.95 } else { 0.8 };
        
        let mut details = HashMap::new();
        details.insert("profanity_count".to_string(), found_profanity.len().to_string());
        if !found_profanity.is_empty() {
            details.insert("found_words".to_string(), found_profanity.len().to_string()); // Don't expose actual words
        }
        
        let error_message = if !is_valid && !found_profanity.is_empty() {
            Some("Inappropriate content detected".to_string())
        } else {
            None
        };
        
        Ok(Some(ValidationResult {
            is_valid,
            confidence,
            error_message,
            details,
        }))
    }

    fn name(&self) -> &str {
        "profanity_detection"
    }

    fn description(&self) -> &str {
        "Detects inappropriate or profane content in text"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn create_test_content(text: &str) -> MultiModalContent {
        MultiModalContent {
            id: "test".to_string(),
            content_type: ContentType::Text,
            data: text.as_bytes().to_vec(),
            metadata: ContentMetadata::default(),
            source_url: None,
            timestamp: SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn test_natural_language_validator() {
        let validator = NaturalLanguageValidator::new();
        let content = create_test_content("This is a test of natural language validation.");
        
        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert!(result.confidence > 0.5);
    }

    #[tokio::test]
    async fn test_sentiment_validator() {
        let validator = SentimentValidator::new();
        let content = create_test_content("This is a great and wonderful day!");
        
        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert!(result.details.contains_key("sentiment_score"));
    }

    #[tokio::test]
    async fn test_language_detection_validator() {
        let validator = LanguageDetectionValidator::new();
        let content = create_test_content("The quick brown fox jumps over the lazy dog.");
        
        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.details.get("detected_language").unwrap() == "en");
    }

    #[tokio::test]
    async fn test_entity_extraction_validator() {
        let validator = EntityExtractionValidator::new();
        let content = create_test_content("John Smith visited New York last Tuesday.");
        
        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert!(result.details.contains_key("entity_count"));
    }

    #[tokio::test]
    async fn test_text_quality_validator() {
        let validator = TextQualityValidator::new();
        let content = create_test_content("This is a quality text with sufficient length.");
        
        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
    }

    #[tokio::test]
    async fn test_profanity_validator() {
        let validator = ProfanityValidator::new();
        let content = create_test_content("This is a clean text without any inappropriate content.");
        
        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
    }
}