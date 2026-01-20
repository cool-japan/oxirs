//! Context building for LLM generation

use crate::{CommunitySummary, GraphRAGResult, Triple};
use serde::{Deserialize, Serialize};

/// Context builder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Maximum context length in characters
    pub max_length: usize,
    /// Include community summaries
    pub include_communities: bool,
    /// Include raw triples
    pub include_triples: bool,
    /// Triple format
    pub triple_format: TripleFormat,
    /// Prioritize triples by score
    pub score_weighted: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_length: 8000,
            include_communities: true,
            include_triples: true,
            triple_format: TripleFormat::NaturalLanguage,
            score_weighted: true,
        }
    }
}

/// Triple formatting options
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TripleFormat {
    /// Natural language: "Entity A is related to Entity B"
    NaturalLanguage,
    /// Structured: "subject → predicate → object"
    Structured,
    /// Turtle-like: `<subject> <predicate> <object> .`
    Turtle,
    /// JSON-LD style
    JsonLd,
}

/// Context builder for LLM input
pub struct ContextBuilder {
    config: ContextConfig,
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new(ContextConfig::default())
    }
}

impl ContextBuilder {
    pub fn new(config: ContextConfig) -> Self {
        Self { config }
    }

    /// Build context string from subgraph and communities
    pub fn build(
        &self,
        query: &str,
        triples: &[Triple],
        communities: &[CommunitySummary],
    ) -> GraphRAGResult<String> {
        let mut context = String::new();
        let mut remaining_length = self.config.max_length;

        // Add query context
        let query_section = format!("## Query\n{}\n\n", query);
        if query_section.len() < remaining_length {
            context.push_str(&query_section);
            remaining_length -= query_section.len();
        }

        // Add community summaries
        if self.config.include_communities && !communities.is_empty() {
            let community_section = self.format_communities(communities, remaining_length / 3);
            if community_section.len() < remaining_length {
                context.push_str(&community_section);
                remaining_length -= community_section.len();
            }
        }

        // Add triples
        if self.config.include_triples && !triples.is_empty() {
            let triples_section = self.format_triples(triples, remaining_length);
            context.push_str(&triples_section);
        }

        Ok(context)
    }

    /// Format community summaries
    fn format_communities(&self, communities: &[CommunitySummary], max_length: usize) -> String {
        let mut result = String::from("## Knowledge Graph Communities\n\n");

        for community in communities {
            let entry = format!(
                "### {}\n{}\n**Entities:** {}\n\n",
                community.id,
                community.summary,
                community
                    .entities
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            if result.len() + entry.len() > max_length {
                break;
            }
            result.push_str(&entry);
        }

        result
    }

    /// Format triples according to configured format
    fn format_triples(&self, triples: &[Triple], max_length: usize) -> String {
        let mut result = String::from("## Knowledge Graph Facts\n\n");

        for triple in triples {
            let entry = match self.config.triple_format {
                TripleFormat::NaturalLanguage => self.triple_to_natural_language(triple),
                TripleFormat::Structured => self.triple_to_structured(triple),
                TripleFormat::Turtle => self.triple_to_turtle(triple),
                TripleFormat::JsonLd => self.triple_to_jsonld(triple),
            };

            if result.len() + entry.len() > max_length {
                break;
            }
            result.push_str(&entry);
            result.push('\n');
        }

        result
    }

    /// Convert triple to natural language
    fn triple_to_natural_language(&self, triple: &Triple) -> String {
        let subject = self.extract_local_name(&triple.subject);
        let predicate = self.predicate_to_phrase(&triple.predicate);
        let object = self.extract_local_name(&triple.object);

        format!("- {} {} {}", subject, predicate, object)
    }

    /// Convert triple to structured format
    fn triple_to_structured(&self, triple: &Triple) -> String {
        let subject = self.extract_local_name(&triple.subject);
        let predicate = self.extract_local_name(&triple.predicate);
        let object = self.extract_local_name(&triple.object);

        format!("- {} → {} → {}", subject, predicate, object)
    }

    /// Convert triple to Turtle format
    fn triple_to_turtle(&self, triple: &Triple) -> String {
        format!(
            "<{}> <{}> <{}> .",
            triple.subject, triple.predicate, triple.object
        )
    }

    /// Convert triple to JSON-LD style
    fn triple_to_jsonld(&self, triple: &Triple) -> String {
        let subject = self.extract_local_name(&triple.subject);
        let predicate = self.extract_local_name(&triple.predicate);
        let object = self.extract_local_name(&triple.object);

        format!(
            "{{ \"@id\": \"{}\", \"{}\": \"{}\" }}",
            subject, predicate, object
        )
    }

    /// Extract local name from URI
    fn extract_local_name(&self, uri: &str) -> String {
        // Try '#' first (for RDF namespace URIs), then '/'
        uri.rsplit('#')
            .next()
            .filter(|s| s != &uri) // Only use if '#' was found
            .or_else(|| uri.rsplit('/').next())
            .unwrap_or(uri)
            .to_string()
    }

    /// Convert predicate URI to natural language phrase
    fn predicate_to_phrase(&self, predicate: &str) -> String {
        let local = self.extract_local_name(predicate);

        // Common predicate mappings
        match local.as_str() {
            "type" | "rdf:type" => "is a".to_string(),
            "label" | "rdfs:label" => "is labeled".to_string(),
            "subClassOf" => "is a subclass of".to_string(),
            "partOf" => "is part of".to_string(),
            "hasPart" => "has part".to_string(),
            "relatedTo" => "is related to".to_string(),
            "sameAs" => "is the same as".to_string(),
            "knows" => "knows".to_string(),
            "worksFor" => "works for".to_string(),
            "locatedIn" => "is located in".to_string(),
            _ => {
                // Convert camelCase to spaces
                let mut result = String::new();
                for (i, c) in local.chars().enumerate() {
                    if i > 0 && c.is_uppercase() {
                        result.push(' ');
                    }
                    result.push(c.to_lowercase().next().unwrap_or(c));
                }
                result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_building() {
        let builder = ContextBuilder::default();

        let triples = vec![
            Triple::new(
                "http://example.org/Battery1",
                "http://example.org/hasStatus",
                "http://example.org/Critical",
            ),
            Triple::new(
                "http://example.org/Battery1",
                "http://example.org/temperature",
                "85",
            ),
        ];

        let communities = vec![CommunitySummary {
            id: "community_0".to_string(),
            summary: "Battery monitoring entities".to_string(),
            entities: vec!["Battery1".to_string(), "Sensor1".to_string()],
            representative_triples: vec![],
            level: 0,
            modularity: 0.5,
        }];

        let context = builder
            .build("What is the battery status?", &triples, &communities)
            .unwrap();

        assert!(context.contains("Query"));
        assert!(context.contains("Battery1"));
    }

    #[test]
    fn test_predicate_to_phrase() {
        let builder = ContextBuilder::default();

        assert_eq!(
            builder.predicate_to_phrase("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            "is a"
        );
        assert_eq!(
            builder.predicate_to_phrase("http://example.org/partOf"),
            "is part of"
        );
        assert_eq!(
            builder.predicate_to_phrase("http://example.org/hasTemperature"),
            "has temperature"
        );
    }
}
