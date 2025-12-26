//! Natural language query parsing

use crate::GraphRAGResult;
use serde::{Deserialize, Serialize};

/// Parsed query with extracted intent and entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedQuery {
    /// Original query text
    pub original: String,
    /// Extracted keywords
    pub keywords: Vec<String>,
    /// Detected intent
    pub intent: QueryIntent,
    /// Named entities extracted
    pub entities: Vec<ExtractedEntity>,
    /// Temporal constraints if any
    pub temporal: Option<TemporalConstraint>,
}

/// Query intent classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QueryIntent {
    /// Factual question (who, what, when, where)
    Factual,
    /// Explanation (why, how)
    Explanation,
    /// Comparison
    Comparison,
    /// List/enumeration
    List,
    /// Definition
    Definition,
    /// Relationship query
    Relationship,
    /// Unknown intent
    Unknown,
}

/// Extracted entity from query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Entity text as it appears in query
    pub text: String,
    /// Entity type (Person, Organization, Location, etc.)
    pub entity_type: String,
    /// Start position in original query
    pub start: usize,
    /// End position in original query
    pub end: usize,
    /// Confidence score
    pub confidence: f32,
}

/// Temporal constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraint {
    /// Constraint type
    pub constraint_type: TemporalType,
    /// Start date/time (ISO 8601)
    pub start: Option<String>,
    /// End date/time (ISO 8601)
    pub end: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TemporalType {
    Before,
    After,
    During,
    Between,
}

/// Query parser
pub struct QueryParser {
    /// Stop words to filter out
    stop_words: std::collections::HashSet<String>,
}

impl Default for QueryParser {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryParser {
    pub fn new() -> Self {
        let stop_words: std::collections::HashSet<String> = [
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "of", "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "and", "but", "if", "or", "because", "as", "until", "while", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "it",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self { stop_words }
    }

    /// Parse a natural language query
    pub fn parse(&self, query: &str) -> GraphRAGResult<ParsedQuery> {
        let keywords = self.extract_keywords(query);
        let intent = self.detect_intent(query);
        let entities = self.extract_entities(query);
        let temporal = self.extract_temporal(query);

        Ok(ParsedQuery {
            original: query.to_string(),
            keywords,
            intent,
            entities,
            temporal,
        })
    }

    /// Extract keywords from query
    fn extract_keywords(&self, query: &str) -> Vec<String> {
        query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty() && word.len() > 2 && !self.stop_words.contains(*word))
            .map(String::from)
            .collect()
    }

    /// Detect query intent
    fn detect_intent(&self, query: &str) -> QueryIntent {
        let lower = query.to_lowercase();

        if lower.starts_with("what is") || lower.starts_with("define") {
            QueryIntent::Definition
        } else if lower.starts_with("why") || lower.starts_with("how does") {
            QueryIntent::Explanation
        } else if lower.contains("compare") || lower.contains("difference between") {
            QueryIntent::Comparison
        } else if lower.starts_with("list") || lower.contains("what are") {
            QueryIntent::List
        } else if lower.contains("related to")
            || lower.contains("connected to")
            || lower.contains("relationship")
        {
            QueryIntent::Relationship
        } else if lower.starts_with("what")
            || lower.starts_with("who")
            || lower.starts_with("when")
            || lower.starts_with("where")
        {
            QueryIntent::Factual
        } else {
            QueryIntent::Unknown
        }
    }

    /// Extract named entities (simplified)
    fn extract_entities(&self, query: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();

        // Simple capitalization-based entity extraction
        for word in query.split_whitespace() {
            if word.len() > 1
                && word
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
                && ![
                    "What", "Who", "When", "Where", "Why", "How", "Is", "Are", "The", "A",
                ]
                .contains(&word)
            {
                if let Some(start) = query.find(word) {
                    entities.push(ExtractedEntity {
                        text: word.to_string(),
                        entity_type: "Unknown".to_string(),
                        start,
                        end: start + word.len(),
                        confidence: 0.5,
                    });
                }
            }
        }

        entities
    }

    /// Extract temporal constraints (simplified)
    fn extract_temporal(&self, query: &str) -> Option<TemporalConstraint> {
        let lower = query.to_lowercase();

        if lower.contains("before") {
            Some(TemporalConstraint {
                constraint_type: TemporalType::Before,
                start: None,
                end: None,
            })
        } else if lower.contains("after") {
            Some(TemporalConstraint {
                constraint_type: TemporalType::After,
                start: None,
                end: None,
            })
        } else if lower.contains("during") || lower.contains("in ") {
            Some(TemporalConstraint {
                constraint_type: TemporalType::During,
                start: None,
                end: None,
            })
        } else if lower.contains("between") {
            Some(TemporalConstraint {
                constraint_type: TemporalType::Between,
                start: None,
                end: None,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_extraction() {
        let parser = QueryParser::new();
        let keywords = parser.extract_keywords("What are the safety issues with battery cells?");
        assert!(keywords.contains(&"safety".to_string()));
        assert!(keywords.contains(&"issues".to_string()));
        assert!(keywords.contains(&"battery".to_string()));
        assert!(keywords.contains(&"cells".to_string()));
    }

    #[test]
    fn test_intent_detection() {
        let parser = QueryParser::new();

        assert_eq!(
            parser.detect_intent("What is a battery?"),
            QueryIntent::Definition
        );
        assert_eq!(
            parser.detect_intent("Why does the battery overheat?"),
            QueryIntent::Explanation
        );
        assert_eq!(
            parser.detect_intent("Compare lithium and nickel batteries"),
            QueryIntent::Comparison
        );
        assert_eq!(
            parser.detect_intent("List all safety hazards"),
            QueryIntent::List
        );
    }
}
