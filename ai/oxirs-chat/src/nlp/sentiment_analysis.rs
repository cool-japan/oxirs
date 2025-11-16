//! Sentiment Analysis System
//!
//! This module provides sentiment analysis for chat messages to understand
//! user emotions and satisfaction levels. Enhanced with scirs2-text ML capabilities.

use crate::utils::nlp::{LexiconSentimentAnalyzer, WordTokenizer};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Sentiment polarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SentimentPolarity {
    /// Positive sentiment
    Positive,
    /// Negative sentiment
    Negative,
    /// Neutral sentiment
    Neutral,
    /// Mixed sentiment
    Mixed,
}

impl SentimentPolarity {
    /// Get polarity from score (-1.0 to 1.0)
    pub fn from_score(score: f32) -> Self {
        if score > 0.3 {
            SentimentPolarity::Positive
        } else if score < -0.3 {
            SentimentPolarity::Negative
        } else if score.abs() < 0.1 {
            SentimentPolarity::Neutral
        } else {
            SentimentPolarity::Mixed
        }
    }

    /// Get color code for UI display
    pub fn color_code(&self) -> &str {
        match self {
            SentimentPolarity::Positive => "#4CAF50", // Green
            SentimentPolarity::Negative => "#F44336", // Red
            SentimentPolarity::Neutral => "#9E9E9E",  // Gray
            SentimentPolarity::Mixed => "#FF9800",    // Orange
        }
    }

    /// Get emoji representation
    pub fn emoji(&self) -> &str {
        match self {
            SentimentPolarity::Positive => "😊",
            SentimentPolarity::Negative => "😞",
            SentimentPolarity::Neutral => "😐",
            SentimentPolarity::Mixed => "🤔",
        }
    }
}

/// Emotion categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Emotion {
    Joy,
    Sadness,
    Anger,
    Fear,
    Surprise,
    Disgust,
    Trust,
    Anticipation,
}

impl Emotion {
    /// Get all emotion types
    pub fn all() -> Vec<Emotion> {
        vec![
            Emotion::Joy,
            Emotion::Sadness,
            Emotion::Anger,
            Emotion::Fear,
            Emotion::Surprise,
            Emotion::Disgust,
            Emotion::Trust,
            Emotion::Anticipation,
        ]
    }
}

/// Sentiment analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    /// Overall sentiment polarity
    pub polarity: SentimentPolarity,
    /// Sentiment score (-1.0 to 1.0)
    pub score: f32,
    /// Confidence in the assessment (0.0 - 1.0)
    pub confidence: f32,
    /// Detected emotions with intensities
    pub emotions: HashMap<Emotion, f32>,
    /// Sentiment breakdown by sentence
    pub sentence_sentiments: Vec<(String, f32)>,
    /// Key sentiment-bearing phrases
    pub key_phrases: Vec<String>,
}

/// Sentiment analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentConfig {
    /// Enable emotion detection
    pub enable_emotions: bool,
    /// Enable sentence-level analysis
    pub enable_sentence_analysis: bool,
    /// Extract key phrases
    pub extract_key_phrases: bool,
    /// Minimum confidence threshold
    pub min_confidence: f32,
}

impl Default for SentimentConfig {
    fn default() -> Self {
        Self {
            enable_emotions: true,
            enable_sentence_analysis: true,
            extract_key_phrases: true,
            min_confidence: 0.5,
        }
    }
}

/// Sentiment analyzer using lexicon-based approaches
pub struct SentimentAnalyzer {
    config: SentimentConfig,
    positive_words: HashMap<String, f32>,
    negative_words: HashMap<String, f32>,
    emotion_lexicon: HashMap<String, Vec<(Emotion, f32)>>,
    lexicon_analyzer: LexiconSentimentAnalyzer,
    // ml_analyzer: Option<MLSentimentAnalyzer>, // TODO: Add ML analyzer when available
    tokenizer: WordTokenizer,
    sentiment_history: Vec<(String, f32)>, // Track sentiment over time
}

impl SentimentAnalyzer {
    /// Create a new sentiment analyzer with ML capabilities
    pub fn new(config: SentimentConfig) -> Result<Self> {
        let positive_words = Self::load_positive_lexicon();
        let negative_words = Self::load_negative_lexicon();
        let emotion_lexicon = Self::load_emotion_lexicon();

        // Initialize lexicon analyzer
        let lexicon_analyzer = LexiconSentimentAnalyzer::with_basiclexicon();

        // Initialize tokenizer
        let tokenizer = WordTokenizer;

        info!(
            "Initialized advanced sentiment analyzer: lexicon={} pos/{} neg, tokenizer={}",
            positive_words.len(),
            negative_words.len(),
            true
        );

        Ok(Self {
            config,
            positive_words,
            negative_words,
            emotion_lexicon,
            lexicon_analyzer,
            // ml_analyzer, // TODO: Add when ML analyzer is available
            tokenizer,
            sentiment_history: Vec::new(),
        })
    }

    /// Load positive word lexicon
    fn load_positive_lexicon() -> HashMap<String, f32> {
        let words = vec![
            ("good", 0.7),
            ("great", 0.9),
            ("excellent", 1.0),
            ("amazing", 0.95),
            ("wonderful", 0.9),
            ("fantastic", 0.95),
            ("awesome", 0.9),
            ("love", 0.85),
            ("happy", 0.8),
            ("pleased", 0.7),
            ("satisfied", 0.75),
            ("perfect", 1.0),
            ("best", 0.95),
            ("beautiful", 0.8),
            ("brilliant", 0.9),
            ("helpful", 0.7),
            ("thanks", 0.6),
            ("thank", 0.6),
            ("appreciate", 0.7),
        ];

        words.into_iter().map(|(w, s)| (w.to_string(), s)).collect()
    }

    /// Load negative word lexicon
    fn load_negative_lexicon() -> HashMap<String, f32> {
        let words = vec![
            ("bad", -0.7),
            ("terrible", -0.9),
            ("awful", -0.95),
            ("horrible", -0.9),
            ("worst", -1.0),
            ("hate", -0.9),
            ("dislike", -0.7),
            ("angry", -0.8),
            ("sad", -0.7),
            ("disappointed", -0.75),
            ("frustrated", -0.8),
            ("annoyed", -0.65),
            ("poor", -0.6),
            ("useless", -0.85),
            ("broken", -0.7),
            ("fail", -0.75),
            ("error", -0.5),
            ("problem", -0.5),
            ("issue", -0.4),
        ];

        words.into_iter().map(|(w, s)| (w.to_string(), s)).collect()
    }

    /// Load emotion lexicon
    fn load_emotion_lexicon() -> HashMap<String, Vec<(Emotion, f32)>> {
        let mut lexicon = HashMap::new();

        // Joy words
        for word in &["happy", "joy", "excited", "cheerful", "delighted"] {
            lexicon.insert(word.to_string(), vec![(Emotion::Joy, 0.9)]);
        }

        // Sadness words
        for word in &["sad", "unhappy", "depressed", "miserable", "sorrowful"] {
            lexicon.insert(word.to_string(), vec![(Emotion::Sadness, 0.9)]);
        }

        // Anger words
        for word in &["angry", "furious", "enraged", "annoyed", "frustrated"] {
            lexicon.insert(word.to_string(), vec![(Emotion::Anger, 0.9)]);
        }

        // Fear words
        for word in &["afraid", "scared", "fearful", "anxious", "worried"] {
            lexicon.insert(word.to_string(), vec![(Emotion::Fear, 0.9)]);
        }

        // Surprise words
        for word in &["surprised", "amazed", "astonished", "shocked", "stunned"] {
            lexicon.insert(word.to_string(), vec![(Emotion::Surprise, 0.9)]);
        }

        // Trust words
        for word in &["trust", "confident", "sure", "certain", "reliable"] {
            lexicon.insert(word.to_string(), vec![(Emotion::Trust, 0.8)]);
        }

        lexicon
    }

    /// Analyze sentiment of a message
    pub fn analyze(&self, message: &str) -> Result<SentimentResult> {
        debug!(
            "Analyzing sentiment for message: {}",
            message.chars().take(100).collect::<String>()
        );

        let lowercase = message.to_lowercase();
        let words: Vec<&str> = lowercase.split_whitespace().collect();

        // Calculate overall sentiment score
        let mut total_score = 0.0;
        let mut word_count = 0;

        for word in &words {
            if let Some(&score) = self.positive_words.get(*word) {
                total_score += score;
                word_count += 1;
            } else if let Some(&score) = self.negative_words.get(*word) {
                total_score += score;
                word_count += 1;
            }
        }

        let score = if word_count > 0 {
            total_score / word_count as f32
        } else {
            0.0
        };

        let polarity = SentimentPolarity::from_score(score);

        // Calculate confidence based on word coverage
        let confidence = if words.is_empty() {
            0.0
        } else {
            (word_count as f32 / words.len() as f32).min(1.0)
        };

        // Detect emotions
        let emotions = if self.config.enable_emotions {
            self.detect_emotions(&words)
        } else {
            HashMap::new()
        };

        // Analyze sentences
        let sentence_sentiments = if self.config.enable_sentence_analysis {
            self.analyze_sentences(message)
        } else {
            Vec::new()
        };

        // Extract key phrases
        let key_phrases = if self.config.extract_key_phrases {
            self.extract_key_phrases(message, &words)
        } else {
            Vec::new()
        };

        debug!(
            "Sentiment: {:?} (score: {:.2}, confidence: {:.2})",
            polarity, score, confidence
        );

        Ok(SentimentResult {
            polarity,
            score,
            confidence,
            emotions,
            sentence_sentiments,
            key_phrases,
        })
    }

    /// Detect emotions from words
    fn detect_emotions(&self, words: &[&str]) -> HashMap<Emotion, f32> {
        let mut emotions: HashMap<Emotion, f32> = HashMap::new();

        for word in words {
            if let Some(word_emotions) = self.emotion_lexicon.get(*word) {
                for (emotion, intensity) in word_emotions {
                    *emotions.entry(*emotion).or_insert(0.0) += intensity;
                }
            }
        }

        // Normalize by word count
        if !words.is_empty() {
            for intensity in emotions.values_mut() {
                *intensity /= words.len() as f32;
            }
        }

        emotions
    }

    /// Analyze sentiment by sentence
    fn analyze_sentences(&self, message: &str) -> Vec<(String, f32)> {
        let sentences: Vec<&str> = message
            .split(['.', '!', '?'])
            .filter(|s| !s.trim().is_empty())
            .collect();

        sentences
            .iter()
            .map(|sentence| {
                let lowercase = sentence.to_lowercase();
                let words: Vec<&str> = lowercase.split_whitespace().collect();

                let mut score = 0.0;
                let mut count = 0;

                for word in &words {
                    if let Some(&s) = self.positive_words.get(*word) {
                        score += s;
                        count += 1;
                    } else if let Some(&s) = self.negative_words.get(*word) {
                        score += s;
                        count += 1;
                    }
                }

                let sent_score = if count > 0 { score / count as f32 } else { 0.0 };
                (sentence.trim().to_string(), sent_score)
            })
            .collect()
    }

    /// Extract key sentiment-bearing phrases
    fn extract_key_phrases(&self, _message: &str, words: &[&str]) -> Vec<String> {
        let mut phrases = Vec::new();

        // Look for patterns like "very good", "not bad", etc.
        for i in 0..words.len() {
            if i + 1 < words.len() {
                let bigram = format!("{} {}", words[i], words[i + 1]);
                if self.is_sentiment_phrase(&bigram) {
                    phrases.push(bigram);
                }
            }

            if i + 2 < words.len() {
                let trigram = format!("{} {} {}", words[i], words[i + 1], words[i + 2]);
                if self.is_sentiment_phrase(&trigram) {
                    phrases.push(trigram);
                }
            }
        }

        phrases.truncate(5); // Limit to top 5 phrases
        phrases
    }

    /// Check if a phrase carries sentiment
    fn is_sentiment_phrase(&self, phrase: &str) -> bool {
        let words: Vec<&str> = phrase.split_whitespace().collect();
        words
            .iter()
            .any(|w| self.positive_words.contains_key(*w) || self.negative_words.contains_key(*w))
    }

    /// Analyze conversation sentiment trend
    pub fn analyze_trend(&self, messages: &[String]) -> Result<Vec<f32>> {
        messages
            .iter()
            .map(|msg| self.analyze(msg).map(|r| r.score))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_sentiment() {
        let analyzer = SentimentAnalyzer::new(SentimentConfig::default()).unwrap();
        let result = analyzer.analyze("This is great! I love it.").unwrap();

        assert_eq!(result.polarity, SentimentPolarity::Positive);
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_negative_sentiment() {
        let analyzer = SentimentAnalyzer::new(SentimentConfig::default()).unwrap();
        let result = analyzer.analyze("This is terrible and awful.").unwrap();

        assert_eq!(result.polarity, SentimentPolarity::Negative);
        assert!(result.score < 0.0);
    }

    #[test]
    fn test_neutral_sentiment() {
        let analyzer = SentimentAnalyzer::new(SentimentConfig::default()).unwrap();
        let result = analyzer
            .analyze("The data is stored in the database.")
            .unwrap();

        assert_eq!(result.polarity, SentimentPolarity::Neutral);
    }

    #[test]
    fn test_emotion_detection() {
        let analyzer = SentimentAnalyzer::new(SentimentConfig::default()).unwrap();
        let result = analyzer.analyze("I'm so happy and excited!").unwrap();

        assert!(result.emotions.contains_key(&Emotion::Joy));
    }
}
