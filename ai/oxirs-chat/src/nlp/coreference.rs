//! Coreference Resolution System
//!
//! Resolves pronouns and references to their antecedents in multi-turn conversations.

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

/// Coreference resolver
pub struct CoreferenceResolver {
    config: CoreferenceConfig,
    chains: Vec<CoreferenceChain>,
    message_history: Vec<(String, String)>, // (message_id, message_text)
}

impl CoreferenceResolver {
    /// Create a new coreference resolver
    pub fn new(config: CoreferenceConfig) -> Result<Self> {
        info!("Initialized coreference resolver");

        Ok(Self {
            config,
            chains: Vec::new(),
            message_history: Vec::new(),
        })
    }

    /// Add a message to the conversation history
    pub fn add_message(&mut self, message_id: String, text: String) {
        self.message_history.push((message_id, text));

        // Keep only recent history
        if self.message_history.len() > self.config.max_lookback {
            self.message_history.remove(0);
        }
    }

    /// Resolve coreferences in the latest message
    pub fn resolve(&mut self, message_id: &str) -> Result<Vec<CoreferenceChain>> {
        debug!("Resolving coreferences for message: {}", message_id);

        let message_text = self
            .message_history
            .iter()
            .find(|(id, _)| id == message_id)
            .map(|(_, text)| text.clone())
            .context("Message not found in history")?;

        let mut new_chains = Vec::new();

        // Find pronouns in the current message
        if self.config.resolve_pronouns {
            let pronouns = self.extract_pronouns(&message_text, message_id);

            for pronoun in pronouns {
                // Try to resolve pronoun to an entity in recent messages
                if let Some(antecedent) = self.find_antecedent(&pronoun) {
                    // Create or update coreference chain
                    let chain = self.create_or_update_chain(&pronoun, &antecedent);
                    new_chains.push(chain);
                }
            }
        }

        // Resolve definite descriptions
        if self.config.resolve_definite_descriptions {
            let descriptions = self.extract_definite_descriptions(&message_text, message_id);

            for description in descriptions {
                if let Some(antecedent) = self.find_matching_entity(&description) {
                    let chain = self.create_or_update_chain(&description, &antecedent);
                    new_chains.push(chain);
                }
            }
        }

        Ok(new_chains)
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

    /// Find antecedent for a pronoun
    fn find_antecedent(&self, pronoun: &Mention) -> Option<Mention> {
        // Simple heuristic: find the most recent noun phrase
        // In production, use more sophisticated algorithms

        for (msg_id, text) in self.message_history.iter().rev() {
            if msg_id == &pronoun.message_id {
                continue; // Skip the same message
            }

            // Look for capitalized words (potential entity mentions)
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

    /// Find matching entity for a definite description
    fn find_matching_entity(&self, description: &Mention) -> Option<Mention> {
        // Extract the head noun from the description
        let words: Vec<&str> = description.text.split_whitespace().collect();
        let head_noun = words.last()?;

        // Look for matching entities in history
        for (msg_id, text) in self.message_history.iter().rev() {
            if msg_id == &description.message_id {
                continue;
            }

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

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.message_history.clear();
        self.chains.clear();
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
