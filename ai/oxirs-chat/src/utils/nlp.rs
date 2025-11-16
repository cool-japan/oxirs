//! Natural Language Processing utilities
//!
//! This module provides NLP functionality including Part-of-Speech tagging
//! and Named Entity Recognition for text analysis in the chat system.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Part-of-Speech tagger
#[derive(Debug, Clone)]
pub struct POSTagger {
    /// Dictionary of word->tag mappings
    dictionary: HashMap<String, String>,
    /// Default tag for unknown words
    default_tag: String,
}

impl POSTagger {
    /// Create a new POS tagger with default English rules
    pub fn new() -> Self {
        let mut dictionary = HashMap::new();

        // Common words with their POS tags
        Self::add_common_words(&mut dictionary);

        Self {
            dictionary,
            default_tag: "NN".to_string(), // Default to noun
        }
    }

    /// Add common English words to the dictionary
    fn add_common_words(dict: &mut HashMap<String, String>) {
        // Pronouns
        for word in &["i", "you", "he", "she", "it", "we", "they"] {
            dict.insert(word.to_string(), "PRP".to_string());
        }

        // Common verbs
        for word in &["is", "are", "was", "were", "be", "been", "being"] {
            dict.insert(word.to_string(), "VB".to_string());
        }
        for word in &["have", "has", "had"] {
            dict.insert(word.to_string(), "VB".to_string());
        }
        for word in &["do", "does", "did"] {
            dict.insert(word.to_string(), "VB".to_string());
        }

        // Determiners
        for word in &["the", "a", "an"] {
            dict.insert(word.to_string(), "DT".to_string());
        }

        // Prepositions
        for word in &["in", "on", "at", "by", "for", "with", "about", "from", "to"] {
            dict.insert(word.to_string(), "IN".to_string());
        }

        // Conjunctions
        for word in &["and", "or", "but"] {
            dict.insert(word.to_string(), "CC".to_string());
        }

        // Wh-words
        for word in &["what", "who", "where", "when", "why", "how", "which"] {
            dict.insert(word.to_string(), "WH".to_string());
        }
    }

    /// Tag a sequence of tokens with their part-of-speech
    pub fn tag(&self, tokens: &[String]) -> Vec<(String, String)> {
        tokens
            .iter()
            .map(|token| {
                let lower = token.to_lowercase();
                let tag = self
                    .dictionary
                    .get(&lower)
                    .cloned()
                    .unwrap_or_else(|| self.infer_tag(&lower));
                (token.clone(), tag)
            })
            .collect()
    }

    /// Infer POS tag based on word characteristics
    fn infer_tag(&self, word: &str) -> String {
        // Check for common suffixes
        if word.ends_with("ing") {
            return "VBG".to_string(); // Verb, gerund
        }
        if word.ends_with("ed") {
            return "VBD".to_string(); // Verb, past tense
        }
        if word.ends_with("ly") {
            return "RB".to_string(); // Adverb
        }
        if word.ends_with("tion") || word.ends_with("ness") || word.ends_with("ment") {
            return "NN".to_string(); // Noun
        }
        if word.ends_with("s") && word.len() > 2 {
            return "NNS".to_string(); // Plural noun
        }

        // Check if it starts with capital (proper noun)
        if word.chars().next().is_some_and(|c| c.is_uppercase()) {
            return "NNP".to_string(); // Proper noun
        }

        self.default_tag.clone()
    }

    /// Add a word-tag pair to the dictionary
    pub fn add_word(&mut self, word: String, tag: String) {
        self.dictionary.insert(word.to_lowercase(), tag);
    }
}

impl Default for POSTagger {
    fn default() -> Self {
        Self::new()
    }
}

/// Named entity tag types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityTag {
    /// Person name
    Person,
    /// Organization
    Organization,
    /// Location
    Location,
    /// Date or time
    Date,
    /// Number or quantity
    Number,
    /// Miscellaneous entity
    Miscellaneous,
}

impl EntityTag {
    /// Get the string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityTag::Person => "PERSON",
            EntityTag::Organization => "ORG",
            EntityTag::Location => "LOC",
            EntityTag::Date => "DATE",
            EntityTag::Number => "NUM",
            EntityTag::Miscellaneous => "MISC",
        }
    }
}

/// Named entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedEntity {
    /// The entity text
    pub text: String,
    /// Entity type
    pub tag: EntityTag,
    /// Start position in the original text
    pub start: usize,
    /// End position in the original text
    pub end: usize,
    /// Confidence score
    pub confidence: f64,
}

/// Named Entity Recognizer
#[derive(Debug, Clone)]
pub struct NamedEntityRecognizer {
    /// Known entities database
    entities: HashMap<String, EntityTag>,
    /// POS tagger for context
    pos_tagger: POSTagger,
}

impl NamedEntityRecognizer {
    /// Create a new NER with default entity database
    pub fn new() -> Self {
        let mut entities = HashMap::new();

        // Common location names
        for loc in &[
            "america",
            "europe",
            "asia",
            "africa",
            "usa",
            "uk",
            "china",
            "japan",
            "germany",
            "france",
            "canada",
            "australia",
            "london",
            "paris",
            "tokyo",
            "newyork",
            "berlin",
        ] {
            entities.insert(loc.to_string(), EntityTag::Location);
        }

        // Common organization keywords
        for org in &[
            "university",
            "company",
            "corporation",
            "institute",
            "foundation",
            "association",
        ] {
            entities.insert(org.to_string(), EntityTag::Organization);
        }

        Self {
            entities,
            pos_tagger: POSTagger::new(),
        }
    }

    /// Recognize named entities in text
    pub fn recognize(&self, text: &str) -> Vec<NamedEntity> {
        let mut result = Vec::new();
        let tokens: Vec<&str> = text.split_whitespace().collect();

        let mut pos = 0;
        for token in tokens {
            let start = text[pos..].find(token).map(|p| pos + p).unwrap_or(pos);
            let end = start + token.len();

            if let Some(entity) = self.classify_entity(token) {
                result.push(NamedEntity {
                    text: token.to_string(),
                    tag: entity,
                    start,
                    end,
                    confidence: 0.8,
                });
            }

            pos = end;
        }

        result
    }

    /// Classify a token as an entity
    fn classify_entity(&self, token: &str) -> Option<EntityTag> {
        let lower = token.to_lowercase();

        // Check dictionary
        if let Some(tag) = self.entities.get(&lower) {
            return Some(*tag);
        }

        // Pattern-based recognition
        // Check if it's a number
        if token
            .chars()
            .all(|c| c.is_numeric() || c == ',' || c == '.')
        {
            return Some(EntityTag::Number);
        }

        // Check if it starts with capital (likely proper noun)
        if token.chars().next().is_some_and(|c| c.is_uppercase()) {
            // Could be person, organization, or location
            // Use simple heuristics
            if token.ends_with("Inc") || token.ends_with("Corp") || token.ends_with("Ltd") {
                return Some(EntityTag::Organization);
            }
            // Default to person for capitalized words
            return Some(EntityTag::Person);
        }

        // Check for date patterns
        if self.is_date_like(token) {
            return Some(EntityTag::Date);
        }

        None
    }

    /// Check if a token looks like a date
    fn is_date_like(&self, token: &str) -> bool {
        let months = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ];

        let lower = token.to_lowercase();
        months.contains(&lower.as_str())
            || token.contains('/')
            || token.contains('-')
            || (token.len() == 4 && token.chars().all(|c| c.is_numeric()))
    }

    /// Add a known entity to the database
    pub fn add_entity(&mut self, text: String, tag: EntityTag) {
        self.entities.insert(text.to_lowercase(), tag);
    }
}

impl Default for NamedEntityRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Entity for NER (simpler version for internal use)
#[derive(Debug, Clone)]
pub struct Entity {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub entity_type: String,
}

/// Rule-based NER (wrapper around NamedEntityRecognizer)
#[derive(Debug, Clone)]
pub struct RuleBasedNER {
    inner: NamedEntityRecognizer,
}

impl RuleBasedNER {
    pub fn new() -> Self {
        Self {
            inner: NamedEntityRecognizer::new(),
        }
    }

    pub fn extract_entities(&self, text: &str) -> Result<Vec<Entity>, String> {
        let named_entities = self.inner.recognize(text);
        let entities = named_entities
            .into_iter()
            .map(|ne| Entity {
                text: ne.text,
                start: ne.start,
                end: ne.end,
                entity_type: ne.tag.as_str().to_string(),
            })
            .collect();
        Ok(entities)
    }
}

impl Default for RuleBasedNER {
    fn default() -> Self {
        Self::new()
    }
}

/// Tokenizer trait
pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, String>;
}

/// Word tokenizer (simple whitespace-based)
#[derive(Debug, Clone, Default)]
pub struct WordTokenizer;

impl WordTokenizer {
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for WordTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, String> {
        Ok(text.split_whitespace().map(|s| s.to_string()).collect())
    }
}

/// Sentiment score
#[derive(Debug, Clone)]
pub struct SentimentScore {
    pub score: f32,     // -1.0 (negative) to 1.0 (positive)
    pub magnitude: f32, // 0.0 to 1.0
}

/// Lexicon-based sentiment analyzer
#[derive(Debug, Clone)]
pub struct LexiconSentimentAnalyzer {
    positive_words: HashMap<String, f32>,
    negative_words: HashMap<String, f32>,
}

impl LexiconSentimentAnalyzer {
    pub fn with_basiclexicon() -> Self {
        let mut positive_words = HashMap::new();
        let mut negative_words = HashMap::new();

        // Basic positive words
        for word in &[
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "love",
            "best",
            "perfect",
            "happy",
            "nice",
        ] {
            positive_words.insert(word.to_string(), 1.0);
        }

        // Basic negative words
        for word in &[
            "bad", "terrible", "awful", "hate", "worst", "horrible", "poor", "sad", "wrong", "fail",
        ] {
            negative_words.insert(word.to_string(), -1.0);
        }

        Self {
            positive_words,
            negative_words,
        }
    }

    pub fn analyze(&self, text: &str) -> Result<SentimentScore, String> {
        let lowercase = text.to_lowercase();
        let words: Vec<&str> = lowercase.split_whitespace().collect();
        let mut total_score = 0.0;
        let mut count = 0;

        for word in &words {
            if let Some(&score) = self.positive_words.get(*word) {
                total_score += score;
                count += 1;
            } else if let Some(&score) = self.negative_words.get(*word) {
                total_score += score;
                count += 1;
            }
        }

        let score = if count > 0 {
            total_score / words.len() as f32
        } else {
            0.0
        };

        let magnitude = score.abs();

        Ok(SentimentScore { score, magnitude })
    }
}

impl Default for LexiconSentimentAnalyzer {
    fn default() -> Self {
        Self::with_basiclexicon()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pos_tagger() {
        let tagger = POSTagger::new();
        let tokens = vec!["The".to_string(), "cat".to_string(), "runs".to_string()];
        let tagged = tagger.tag(&tokens);

        assert_eq!(tagged.len(), 3);
        assert_eq!(tagged[0].1, "DT"); // "the" is a determiner
    }

    #[test]
    fn test_ner() {
        let ner = NamedEntityRecognizer::new();
        let text = "John visited London in 2023";
        let entities = ner.recognize(text);

        assert!(!entities.is_empty());
        // Should recognize at least "John" (person), "London" (location), and "2023" (number/date)
        assert!(entities.iter().any(|e| e.tag == EntityTag::Person));
        assert!(entities.iter().any(|e| e.tag == EntityTag::Location));
    }

    #[test]
    fn test_entity_tag_string() {
        assert_eq!(EntityTag::Person.as_str(), "PERSON");
        assert_eq!(EntityTag::Location.as_str(), "LOC");
        assert_eq!(EntityTag::Organization.as_str(), "ORG");
    }

    #[test]
    fn test_number_recognition() {
        let ner = NamedEntityRecognizer::new();
        let text = "The price is 123.45 dollars";
        let entities = ner.recognize(text);

        let numbers: Vec<_> = entities
            .iter()
            .filter(|e| e.tag == EntityTag::Number)
            .collect();
        assert!(!numbers.is_empty());
    }
}
