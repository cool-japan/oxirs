//! Coreference Resolution System
//!
//! Resolves pronouns and references to their antecedents in multi-turn conversations.
//! Enhanced with scirs2-text for advanced coreference resolution capabilities.

use crate::utils::nlp::{Entity, POSTagger, RuleBasedNER, WordTokenizer};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Coreference chain - links mentions of the same entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreferenceChain {
    /// Chain ID
    pub id: String,
    /// All mentions in the chain
    pub mentions: Vec<Mention>,
    /// Representative mention (usually the most informative one)
    pub representative: Mention,
    /// Entity type if known
    pub entity_type: Option<String>,
}

/// A mention of an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mention {
    /// Mention text
    pub text: String,
    /// Message ID where this mention appears
    pub message_id: String,
    /// Start position in message
    pub start: usize,
    /// End position in message
    pub end: usize,
    /// Is this a pronoun?
    pub is_pronoun: bool,
}

/// Coreference resolution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreferenceConfig {
    /// Maximum distance (in messages) to look back for antecedents
    pub max_lookback: usize,
    /// Enable pronoun resolution
    pub resolve_pronouns: bool,
    /// Enable definite description resolution (e.g., "the movie", "that dataset")
    pub resolve_definite_descriptions: bool,
    /// Minimum confidence for resolution
    pub min_confidence: f32,
}

impl Default for CoreferenceConfig {
    fn default() -> Self {
        Self {
            max_lookback: 5,
            resolve_pronouns: true,
            resolve_definite_descriptions: true,
            min_confidence: 0.7,
        }
    }
}

/// Coreference resolver with advanced NLP capabilities
pub struct CoreferenceResolver {
    config: CoreferenceConfig,
    chains: Vec<CoreferenceChain>,
    message_history: Vec<(String, String)>, // (message_id, message_text)
    // scirs2_resolver: Option<CoreferenceResolver>, // TODO: Add when scirs2-text coreference is available
    ner: Option<RuleBasedNER>,
    pos_tagger: Option<POSTagger>,
    tokenizer: WordTokenizer,
    entity_cache: HashMap<String, Vec<Entity>>, // Cache of extracted entities per message
}

impl CoreferenceResolver {
    /// Create a new coreference resolver with advanced NLP
    pub fn new(config: CoreferenceConfig) -> Result<Self> {
        // Initialize NER for entity extraction
        let ner = Some(RuleBasedNER::new());

        // Initialize POS tagger for grammatical analysis
        let pos_tagger = Some(POSTagger::new());

        // Initialize tokenizer
        let tokenizer = WordTokenizer;

        info!(
            "Initialized advanced coreference resolver (NER: {}, POS: {})",
            ner.is_some(),
            pos_tagger.is_some()
        );

        Ok(Self {
            config,
            chains: Vec::new(),
            message_history: Vec::new(),
            // scirs2_resolver, // TODO: Add when available
            ner,
            pos_tagger,
            tokenizer,
            entity_cache: HashMap::new(),
        })
    }

    /// Add a message to the conversation history with entity extraction
    pub fn add_message(&mut self, message_id: String, text: String) {
        // Extract and cache entities from the message
        if let Some(ref ner) = self.ner {
            if let Ok(entities) = ner.extract_entities(&text) {
                self.entity_cache.insert(message_id.clone(), entities);
                debug!(
                    "Cached {} entities for message {}",
                    self.entity_cache
                        .get(&message_id)
                        .map(|e| e.len())
                        .unwrap_or(0),
                    message_id
                );
            }
        }

        self.message_history.push((message_id.clone(), text));

        // Keep only recent history
        if self.message_history.len() > self.config.max_lookback {
            if let Some((old_msg_id, _)) = self.message_history.first() {
                self.entity_cache.remove(old_msg_id);
            }
            self.message_history.remove(0);
        }
    }

    /// Resolve coreferences in the latest message with advanced NLP
    pub fn resolve(&mut self, message_id: &str) -> Result<Vec<CoreferenceChain>> {
        debug!("Resolving coreferences for message: {}", message_id);

        let message_text = self
            .message_history
            .iter()
            .find(|(id, _)| id == message_id)
            .map(|(_, text)| text.clone())
            .context("Message not found in history")?;

        let mut new_chains = Vec::new();

        // STEP 1: Rule-based pronoun resolution
        // TODO: Add scirs2-text coreference resolution when available
        // if let Some(ref mut scirs2_resolver) = self.scirs2_resolver {
        //     ... scirs2 coreference logic ...
        // }

        // STEP 2: Rule-based resolution
        // Find pronouns in the current message
        if self.config.resolve_pronouns {
            let pronouns = self.extract_pronouns(&message_text, message_id);

            for pronoun in pronouns {
                // Try to resolve pronoun to an entity using NER cache
                if let Some(antecedent) = self.find_antecedent_advanced(&pronoun) {
                    let chain = self.create_or_update_chain(&pronoun, &antecedent);
                    new_chains.push(chain);
                }
            }
        }

        // STEP 3: Resolve definite descriptions
        if self.config.resolve_definite_descriptions {
            let descriptions = self.extract_definite_descriptions(&message_text, message_id);

            for description in descriptions {
                if let Some(antecedent) = self.find_matching_entity_advanced(&description) {
                    let chain = self.create_or_update_chain(&description, &antecedent);
                    new_chains.push(chain);
                }
            }
        }

        // Store chains for future reference
        self.chains.extend(new_chains.clone());

        Ok(new_chains)
    }

    /// Check if a word is a pronoun
    fn is_pronoun(&self, word: &str) -> bool {
        let pronouns = vec![
            "it", "its", "they", "them", "their", "this", "that", "these", "those", "he", "him",
            "his", "she", "her", "hers",
        ];
        pronouns.contains(&word.to_lowercase().as_str())
    }

    /// Extract pronouns from text
    fn extract_pronouns(&self, text: &str, message_id: &str) -> Vec<Mention> {
        let pronouns = vec![
            "it", "its", "they", "them", "their", "this", "that", "these", "those", "he", "him",
            "his", "she", "her", "hers",
        ];

        let mut mentions = Vec::new();
        let lowercase = text.to_lowercase();
        let words: Vec<&str> = lowercase.split_whitespace().collect();
        let mut pos = 0;

        for word in words {
            if pronouns.contains(&word) {
                let start = text[pos..].find(word).map(|p| p + pos).unwrap_or(pos);
                let end = start + word.len();
                pos = end;

                mentions.push(Mention {
                    text: word.to_string(),
                    message_id: message_id.to_string(),
                    start,
                    end,
                    is_pronoun: true,
                });
            }
        }

        mentions
    }

    /// Extract definite descriptions (e.g., "the movie", "that dataset")
    fn extract_definite_descriptions(&self, text: &str, message_id: &str) -> Vec<Mention> {
        let mut mentions = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        for i in 0..words.len() {
            // Look for patterns like "the X", "that X"
            if i + 1 < words.len() {
                let first = words[i].to_lowercase();
                if first == "the" || first == "that" || first == "this" {
                    let phrase = format!("{} {}", words[i], words[i + 1]);
                    let start = text.find(&phrase).unwrap_or(0);
                    let end = start + phrase.len();

                    mentions.push(Mention {
                        text: phrase,
                        message_id: message_id.to_string(),
                        start,
                        end,
                        is_pronoun: false,
                    });
                }
            }
        }

        mentions
    }

    /// Find antecedent for a pronoun using advanced entity cache
    fn find_antecedent_advanced(&self, pronoun: &Mention) -> Option<Mention> {
        // STEP 1: Try to find entities from cached NER results
        for (msg_id, text) in self.message_history.iter().rev() {
            if msg_id == &pronoun.message_id {
                continue; // Skip the same message
            }

            // Check entity cache first
            if let Some(entities) = self.entity_cache.get(msg_id) {
                if let Some(entity) = entities.first() {
                    return Some(Mention {
                        text: entity.text.clone(),
                        message_id: msg_id.clone(),
                        start: entity.start,
                        end: entity.end,
                        is_pronoun: false,
                    });
                }
            }

            // STEP 2: Fallback to pattern-based extraction
            let words: Vec<&str> = text.split_whitespace().collect();
            for word in words {
                if word
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
                    && word.len() > 2
                {
                    return Some(Mention {
                        text: word.to_string(),
                        message_id: msg_id.clone(),
                        start: 0,
                        end: word.len(),
                        is_pronoun: false,
                    });
                }
            }
        }

        None
    }

    /// Find antecedent for a pronoun (legacy method for backward compatibility)
    fn find_antecedent(&self, pronoun: &Mention) -> Option<Mention> {
        self.find_antecedent_advanced(pronoun)
    }

    /// Find matching entity for a definite description with advanced matching
    fn find_matching_entity_advanced(&self, description: &Mention) -> Option<Mention> {
        // Extract the head noun from the description
        let words: Vec<&str> = description.text.split_whitespace().collect();
        let head_noun = words.last()?;

        // STEP 1: Try to match against cached entities
        for (msg_id, text) in self.message_history.iter().rev() {
            if msg_id == &description.message_id {
                continue;
            }

            // Check entity cache for matches
            if let Some(entities) = self.entity_cache.get(msg_id) {
                for entity in entities {
                    // Match by semantic similarity (simple word containment for now)
                    if entity
                        .text
                        .to_lowercase()
                        .contains(head_noun.to_lowercase().as_str())
                        || head_noun
                            .to_lowercase()
                            .contains(&entity.text.to_lowercase())
                    {
                        return Some(Mention {
                            text: entity.text.clone(),
                            message_id: msg_id.clone(),
                            start: entity.start,
                            end: entity.end,
                            is_pronoun: false,
                        });
                    }
                }
            }

            // STEP 2: Fallback to text matching
            if text.to_lowercase().contains(head_noun) {
                return Some(Mention {
                    text: head_noun.to_string(),
                    message_id: msg_id.clone(),
                    start: 0,
                    end: head_noun.len(),
                    is_pronoun: false,
                });
            }
        }

        None
    }

    /// Find matching entity for a definite description (legacy method)
    fn find_matching_entity(&self, description: &Mention) -> Option<Mention> {
        self.find_matching_entity_advanced(description)
    }

    /// Create or update a coreference chain
    fn create_or_update_chain(
        &mut self,
        mention1: &Mention,
        mention2: &Mention,
    ) -> CoreferenceChain {
        let chain_id = uuid::Uuid::new_v4().to_string();

        // Determine representative (prefer non-pronoun mentions)
        let representative = if !mention2.is_pronoun {
            mention2.clone()
        } else {
            mention1.clone()
        };

        CoreferenceChain {
            id: chain_id,
            mentions: vec![mention1.clone(), mention2.clone()],
            representative,
            entity_type: None,
        }
    }

    /// Get all active coreference chains
    pub fn get_chains(&self) -> &[CoreferenceChain] {
        &self.chains
    }

    /// Clear conversation history and cached entities
    pub fn clear_history(&mut self) {
        self.message_history.clear();
        self.chains.clear();
        self.entity_cache.clear();
        debug!("Cleared conversation history and entity cache");
    }

    /// Resolve text with coreferences replaced
    pub fn resolve_text(&self, text: &str) -> String {
        let mut resolved = text.to_string();

        // Replace pronouns with their antecedents
        for chain in &self.chains {
            for mention in &chain.mentions {
                if mention.is_pronoun {
                    resolved = resolved.replace(&mention.text, &chain.representative.text);
                }
            }
        }

        resolved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pronoun_extraction() {
        let resolver = CoreferenceResolver::new(CoreferenceConfig::default()).unwrap();
        let pronouns = resolver.extract_pronouns("Show me it and them", "msg1");

        assert_eq!(pronouns.len(), 2);
        assert!(pronouns.iter().any(|p| p.text == "it"));
        assert!(pronouns.iter().any(|p| p.text == "them"));
    }

    #[test]
    fn test_definite_description_extraction() {
        let resolver = CoreferenceResolver::new(CoreferenceConfig::default()).unwrap();
        let descriptions =
            resolver.extract_definite_descriptions("Tell me about the movie", "msg1");

        assert!(!descriptions.is_empty());
        assert!(descriptions.iter().any(|d| d.text.contains("the movie")));
    }

    #[test]
    fn test_coreference_resolution() {
        let mut resolver = CoreferenceResolver::new(CoreferenceConfig::default()).unwrap();

        // Add conversation history
        resolver.add_message(
            "msg1".to_string(),
            "I'm looking for information about Inception".to_string(),
        );
        resolver.add_message("msg2".to_string(), "Tell me more about it".to_string());

        let chains = resolver.resolve("msg2").unwrap();

        // Should find coreference between "it" and "Inception"
        assert!(!chains.is_empty());
    }
}
