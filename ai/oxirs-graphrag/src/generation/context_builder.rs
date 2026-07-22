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

/// A knowledge-graph [`Triple`] paired with a relevance score.
///
/// [`ContextBuilder::build`] honors [`ContextConfig::score_weighted`] by
/// sorting these descending by `score` before truncating to the character
/// budget. `Triple` itself carries no score (it is a plain RDF fact), so
/// callers that want prioritized ordering must supply one explicitly — use
/// [`ScoredTriple::unscored`] for callers with no real relevance signal,
/// which makes `score_weighted` a stable (input-order-preserving) no-op
/// rather than silently claiming to weight triples it has no data to weight.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredTriple {
    pub triple: Triple,
    pub score: f64,
}

impl ScoredTriple {
    /// Wrap a triple with an explicit relevance score.
    pub fn new(triple: Triple, score: f64) -> Self {
        Self { triple, score }
    }

    /// Wrap a triple with a neutral score, for callers with no relevance
    /// signal available. `Vec::sort_by` is stable, so a slice of
    /// all-neutral-score triples is left in its original order even when
    /// `score_weighted` is enabled.
    pub fn unscored(triple: Triple) -> Self {
        Self { triple, score: 0.0 }
    }
}

impl From<Triple> for ScoredTriple {
    fn from(triple: Triple) -> Self {
        Self::unscored(triple)
    }
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

    /// Build context string from subgraph and communities.
    ///
    /// When [`ContextConfig::score_weighted`] is set, `triples` are sorted
    /// by descending [`ScoredTriple::score`] before truncation, so the most
    /// relevant facts survive the character budget first. Pass
    /// [`ScoredTriple::unscored`] triples (or use [`Self::build_unscored`])
    /// if no real relevance score is available — `score_weighted` then has
    /// no effect (stable sort preserves input order), rather than silently
    /// pretending to prioritize triples it has no signal to prioritize by.
    pub fn build(
        &self,
        query: &str,
        triples: &[ScoredTriple],
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

        // Add triples, honoring `score_weighted` if requested.
        if self.config.include_triples && !triples.is_empty() {
            let ordered: Vec<&ScoredTriple> = if self.config.score_weighted {
                let mut refs: Vec<&ScoredTriple> = triples.iter().collect();
                refs.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                refs
            } else {
                triples.iter().collect()
            };
            let triples_section = self.format_triples(&ordered, remaining_length);
            context.push_str(&triples_section);
        }

        Ok(context)
    }

    /// Convenience wrapper for callers with no relevance score available:
    /// wraps every triple as [`ScoredTriple::unscored`] and delegates to
    /// [`Self::build`]. `score_weighted` becomes a no-op in this case, since
    /// there is no real score to weight by.
    pub fn build_unscored(
        &self,
        query: &str,
        triples: &[Triple],
        communities: &[CommunitySummary],
    ) -> GraphRAGResult<String> {
        let scored: Vec<ScoredTriple> = triples
            .iter()
            .cloned()
            .map(ScoredTriple::unscored)
            .collect();
        self.build(query, &scored, communities)
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

    /// Format triples according to configured format. `triples` is assumed
    /// to already be in the order they should be considered for inclusion
    /// (score-weighted or input order, decided by the caller in
    /// [`Self::build`]).
    fn format_triples(&self, triples: &[&ScoredTriple], max_length: usize) -> String {
        let mut result = String::from("## Knowledge Graph Facts\n\n");

        for scored in triples {
            let triple = &scored.triple;
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
            .build_unscored("What is the battery status?", &triples, &communities)
            .expect("should succeed");

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

    // ── Regression: score_weighted actually reorders triples (P2) ──────────

    fn triple_n(n: u32) -> Triple {
        Triple::new(
            format!("http://example.org/s{n}"),
            "http://example.org/rel",
            format!("http://example.org/o{n}"),
        )
    }

    #[test]
    fn regression_score_weighted_true_sorts_descending_by_score() {
        let builder = ContextBuilder::new(ContextConfig {
            triple_format: TripleFormat::Turtle,
            score_weighted: true,
            ..ContextConfig::default()
        });

        // Deliberately supplied in ascending score order — a correct
        // implementation must reorder to descending (low score last).
        let triples = vec![
            ScoredTriple::new(triple_n(1), 0.1),
            ScoredTriple::new(triple_n(2), 0.9),
            ScoredTriple::new(triple_n(3), 0.5),
        ];

        let context = builder.build("q", &triples, &[]).expect("should succeed");

        let pos2 = context.find("s2").expect("s2 present");
        let pos3 = context.find("s3").expect("s3 present");
        let pos1 = context.find("s1").expect("s1 present");
        assert!(
            pos2 < pos3 && pos3 < pos1,
            "expected order by descending score (s2=0.9, s3=0.5, s1=0.1), got: {context}"
        );
    }

    #[test]
    fn regression_score_weighted_false_preserves_input_order() {
        let builder = ContextBuilder::new(ContextConfig {
            triple_format: TripleFormat::Turtle,
            score_weighted: false,
            ..ContextConfig::default()
        });

        // Same triples as the sort test, but score_weighted is off: input
        // order (ascending score here) must be preserved verbatim.
        let triples = vec![
            ScoredTriple::new(triple_n(1), 0.1),
            ScoredTriple::new(triple_n(2), 0.9),
            ScoredTriple::new(triple_n(3), 0.5),
        ];

        let context = builder.build("q", &triples, &[]).expect("should succeed");

        let pos1 = context.find("s1").expect("s1 present");
        let pos2 = context.find("s2").expect("s2 present");
        let pos3 = context.find("s3").expect("s3 present");
        assert!(
            pos1 < pos2 && pos2 < pos3,
            "expected original input order preserved, got: {context}"
        );
    }

    #[test]
    fn regression_unscored_triples_are_stable_regardless_of_score_weighted() {
        // Callers with no real relevance signal (build_unscored) must get
        // input-order-preserving output even with score_weighted enabled —
        // `score_weighted` should never fabricate an ordering it has no
        // data to justify.
        let builder = ContextBuilder::new(ContextConfig {
            triple_format: TripleFormat::Turtle,
            score_weighted: true,
            ..ContextConfig::default()
        });

        let triples = vec![triple_n(1), triple_n(2), triple_n(3)];
        let context = builder
            .build_unscored("q", &triples, &[])
            .expect("should succeed");

        let pos1 = context.find("s1").expect("s1 present");
        let pos2 = context.find("s2").expect("s2 present");
        let pos3 = context.find("s3").expect("s3 present");
        assert!(pos1 < pos2 && pos2 < pos3);
    }
}
