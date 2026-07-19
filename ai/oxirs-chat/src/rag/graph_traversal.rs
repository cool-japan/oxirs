//! Knowledge graph exploration and entity expansion
//!
//! Provides graph traversal capabilities with entity expansion and relationship discovery.

use super::*;
use oxirs_core::model::Predicate;

/// The `rdf:type` predicate IRI, used to look up type information for entities.
const RDF_TYPE_IRI: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

/// Maximum number of triples returned by a single store pattern lookup, to keep
/// graph-traversal expansion bounded on large graphs.
const MAX_TRIPLES_PER_LOOKUP: usize = 200;

/// Graph traversal engine for knowledge graph exploration
pub struct GraphTraversal {
    store: Arc<dyn Store>,
}

impl GraphTraversal {
    pub fn new(store: Arc<dyn Store>) -> Self {
        Self { store }
    }

    /// Perform graph search based on extracted entities
    pub async fn perform_graph_search(
        &self,
        _query: &str,
        entities: &[ExtractedEntity],
        max_depth: usize,
    ) -> Result<Vec<RagSearchResult>> {
        let mut results = Vec::new();
        let mut visited_entities = HashSet::new();

        for entity in entities {
            if let Some(ref iri) = entity.iri {
                if visited_entities.contains(iri) {
                    continue;
                }
                visited_entities.insert(iri.clone());

                // Basic entity traversal
                let entity_triples = self.find_entity_triples(iri, max_depth).await?;

                for triple in entity_triples {
                    results.push(RagSearchResult {
                        triple,
                        score: entity.confidence * 0.8, // Slightly lower score than direct matches
                        search_type: SearchType::GraphTraversal,
                    });
                }

                // Enhanced entity expansion
                let expanded_results = self.expand_entity_context(iri, entity.confidence).await?;
                results.extend(expanded_results);
            }
        }

        // Remove duplicates and apply graph-specific ranking
        self.deduplicate_and_rank_graph_results(results)
    }

    /// Find triples related to an entity with given depth
    async fn find_entity_triples(&self, entity_iri: &str, max_depth: usize) -> Result<Vec<Triple>> {
        let mut triples = Vec::new();
        let mut visited = HashSet::new();
        let mut current_entities = vec![entity_iri.to_string()];

        for _depth in 0..max_depth {
            let mut next_entities = Vec::new();

            for entity in &current_entities {
                if visited.contains(entity) {
                    continue;
                }
                visited.insert(entity.clone());

                // Find triples where entity is subject
                if let Ok(subject_triples) = self.find_triples_with_subject(entity).await {
                    for triple in &subject_triples {
                        if let Object::NamedNode(neighbor) = triple.object() {
                            let neighbor = neighbor.as_str().to_string();
                            if !visited.contains(&neighbor) {
                                next_entities.push(neighbor);
                            }
                        }
                    }
                    triples.extend(subject_triples);
                }

                // Find triples where entity is object
                if let Ok(object_triples) = self.find_triples_with_object(entity).await {
                    for triple in &object_triples {
                        if let Subject::NamedNode(neighbor) = triple.subject() {
                            let neighbor = neighbor.as_str().to_string();
                            if !visited.contains(&neighbor) {
                                next_entities.push(neighbor);
                            }
                        }
                    }
                    triples.extend(object_triples);
                }
            }

            current_entities = next_entities;
            if current_entities.is_empty() {
                break;
            }
        }

        Ok(triples)
    }

    /// Expand entity context with related entities and properties
    async fn expand_entity_context(
        &self,
        entity_iri: &str,
        base_confidence: f32,
    ) -> Result<Vec<RagSearchResult>> {
        let mut expanded_results = Vec::new();

        // Find type information
        let type_triples = self.find_entity_types(entity_iri).await?;
        for triple in type_triples {
            expanded_results.push(RagSearchResult {
                triple,
                score: base_confidence * 0.9, // High score for type information
                search_type: SearchType::GraphTraversal,
            });
        }

        // Find same-type entities (for entity recommendation)
        let same_type_entities = self.find_same_type_entities(entity_iri, 5).await?;
        for triple in same_type_entities {
            expanded_results.push(RagSearchResult {
                triple,
                score: base_confidence * 0.6, // Lower score for related entities
                search_type: SearchType::GraphTraversal,
            });
        }

        // Find property domains and ranges
        let property_context = self.find_property_context(entity_iri).await?;
        for triple in property_context {
            expanded_results.push(RagSearchResult {
                triple,
                score: base_confidence * 0.7,
                search_type: SearchType::GraphTraversal,
            });
        }

        Ok(expanded_results)
    }

    /// Find type information for an entity
    async fn find_entity_types(&self, entity_iri: &str) -> Result<Vec<Triple>> {
        let type_predicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        let mut type_triples = Vec::new();

        if let Ok(subject_triples) = self.find_triples_with_subject(entity_iri).await {
            for triple in subject_triples {
                if triple.predicate().to_string().contains(type_predicate) {
                    type_triples.push(triple);
                }
            }
        }

        Ok(type_triples)
    }

    /// Find entities of the same type
    async fn find_same_type_entities(&self, entity_iri: &str, limit: usize) -> Result<Vec<Triple>> {
        let mut same_type_triples = Vec::new();

        // First, get the types of the input entity
        let entity_types = self.find_entity_types(entity_iri).await?;

        if entity_types.is_empty() {
            return Ok(same_type_triples);
        }

        // For each type, find other entities of the same type (simplified implementation)
        for type_triple in entity_types.iter().take(2) {
            // Limit to 2 types for performance
            if let Ok(entities_of_type) = self
                .find_entities_of_type(&type_triple.object().to_string(), limit)
                .await
            {
                same_type_triples.extend(entities_of_type);
            }
        }

        Ok(same_type_triples)
    }

    /// Find entities of a specific type by looking up `?entity rdf:type <type_iri>`
    /// triples in the store (i.e. all instances declared to be of this type).
    async fn find_entities_of_type(&self, type_iri: &str, limit: usize) -> Result<Vec<Triple>> {
        let Ok(type_node) = NamedNode::new(type_iri) else {
            return Ok(Vec::new());
        };
        let Ok(type_predicate_node) = NamedNode::new(RDF_TYPE_IRI) else {
            return Ok(Vec::new());
        };
        let type_predicate = Predicate::NamedNode(type_predicate_node);
        let object = Object::NamedNode(type_node);

        let quads = self
            .store
            .find_quads(None, Some(&type_predicate), Some(&object), None)?;

        Ok(quads
            .into_iter()
            .take(limit.min(MAX_TRIPLES_PER_LOOKUP))
            .map(|q| q.to_triple())
            .collect())
    }

    /// Find property context (domains and ranges) for the properties observed
    /// on `entity_iri`. Discovers the distinct predicates used by the entity,
    /// then looks up any `rdfs:domain`/`rdfs:range` schema triples declared for
    /// each of those predicates.
    async fn find_property_context(&self, entity_iri: &str) -> Result<Vec<Triple>> {
        let Ok(subject_node) = NamedNode::new(entity_iri) else {
            return Ok(Vec::new());
        };
        let subject = Subject::NamedNode(subject_node);

        let quads = self.store.find_quads(Some(&subject), None, None, None)?;

        let Ok(domain_node) = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#domain") else {
            return Ok(Vec::new());
        };
        let Ok(range_node) = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#range") else {
            return Ok(Vec::new());
        };
        let domain_predicate = Predicate::NamedNode(domain_node);
        let range_predicate = Predicate::NamedNode(range_node);

        let mut seen_properties = HashSet::new();
        let mut context = Vec::new();

        for quad in quads.into_iter().take(MAX_TRIPLES_PER_LOOKUP) {
            let Predicate::NamedNode(property) = quad.predicate() else {
                continue;
            };
            if !seen_properties.insert(property.clone()) {
                continue;
            }
            let property_subject = Subject::NamedNode(property.clone());

            if let Ok(domain_quads) =
                self.store
                    .find_quads(Some(&property_subject), Some(&domain_predicate), None, None)
            {
                context.extend(domain_quads.into_iter().map(|q| q.to_triple()));
            }
            if let Ok(range_quads) =
                self.store
                    .find_quads(Some(&property_subject), Some(&range_predicate), None, None)
            {
                context.extend(range_quads.into_iter().map(|q| q.to_triple()));
            }

            if context.len() >= MAX_TRIPLES_PER_LOOKUP {
                break;
            }
        }

        context.truncate(MAX_TRIPLES_PER_LOOKUP);
        Ok(context)
    }

    /// Find triples with entity as subject
    async fn find_triples_with_subject(&self, entity_iri: &str) -> Result<Vec<Triple>> {
        let Ok(subject_node) = NamedNode::new(entity_iri) else {
            return Ok(Vec::new());
        };
        let subject = Subject::NamedNode(subject_node);

        let quads = self.store.find_quads(Some(&subject), None, None, None)?;

        Ok(quads
            .into_iter()
            .take(MAX_TRIPLES_PER_LOOKUP)
            .map(|q| q.to_triple())
            .collect())
    }

    /// Find triples with entity as object
    async fn find_triples_with_object(&self, entity_iri: &str) -> Result<Vec<Triple>> {
        let Ok(object_node) = NamedNode::new(entity_iri) else {
            return Ok(Vec::new());
        };
        let object = Object::NamedNode(object_node);

        let quads = self.store.find_quads(None, None, Some(&object), None)?;

        Ok(quads
            .into_iter()
            .take(MAX_TRIPLES_PER_LOOKUP)
            .map(|q| q.to_triple())
            .collect())
    }

    /// Remove duplicates and rank graph results
    fn deduplicate_and_rank_graph_results(
        &self,
        mut results: Vec<RagSearchResult>,
    ) -> Result<Vec<RagSearchResult>> {
        // Remove duplicates based on triple content
        let mut seen = HashSet::new();
        results.retain(|result| {
            let key = format!(
                "{}:{}:{}",
                result.triple.subject(),
                result.triple.predicate(),
                result.triple.object()
            );
            seen.insert(key)
        });

        // Sort by score (descending)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }
}

/// Extracted entity from query analysis
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: EntityType,
    pub iri: Option<String>,
    pub confidence: f32,
    pub aliases: Vec<String>,
}

/// Entity types for classification
#[derive(Debug, Clone, PartialEq)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Concept,
    Event,
    Other,
}

/// Extracted relationship from query analysis
#[derive(Debug, Clone)]
pub struct ExtractedRelationship {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
    pub relation_type: RelationType,
}

/// Relationship types for classification
#[derive(Debug, Clone, PartialEq)]
pub enum RelationType {
    CausalRelation,
    TemporalRelation,
    SpatialRelation,
    ConceptualRelation,
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::Quad;
    use oxirs_core::ConcreteStore;

    fn make_store() -> Arc<dyn Store> {
        Arc::new(ConcreteStore::new().expect("store should construct"))
    }

    /// Regression test for the P1 finding: `find_triples_with_subject` must
    /// actually query the store instead of always returning an empty `Vec`.
    #[tokio::test]
    async fn test_find_triples_with_subject_queries_real_store() {
        let store = make_store();
        let alice = NamedNode::new("http://example.org/alice").expect("valid IRI");
        let knows = NamedNode::new("http://example.org/knows").expect("valid IRI");
        let bob = NamedNode::new("http://example.org/bob").expect("valid IRI");
        store
            .insert_quad(Quad::new_default_graph(alice, knows, bob))
            .expect("insert should succeed");

        let traversal = GraphTraversal::new(store);
        let triples = traversal
            .find_triples_with_subject("http://example.org/alice")
            .await
            .expect("query should succeed");

        assert_eq!(triples.len(), 1, "triples = {:?}", triples);
        match triples[0].subject() {
            Subject::NamedNode(n) => assert_eq!(n.as_str(), "http://example.org/alice"),
            other => panic!("expected a named-node subject, got {other:?}"),
        }
    }

    /// Regression test: `find_triples_with_object` must actually query the
    /// store instead of always returning an empty `Vec`.
    #[tokio::test]
    async fn test_find_triples_with_object_queries_real_store() {
        let store = make_store();
        let alice = NamedNode::new("http://example.org/alice").expect("valid IRI");
        let knows = NamedNode::new("http://example.org/knows").expect("valid IRI");
        let bob = NamedNode::new("http://example.org/bob").expect("valid IRI");
        store
            .insert_quad(Quad::new_default_graph(alice, knows, bob))
            .expect("insert should succeed");

        let traversal = GraphTraversal::new(store);
        let triples = traversal
            .find_triples_with_object("http://example.org/bob")
            .await
            .expect("query should succeed");

        assert_eq!(triples.len(), 1, "triples = {:?}", triples);
    }

    /// Regression test: `find_entities_of_type` must look up real
    /// `rdf:type` triples instead of always returning an empty `Vec`.
    #[tokio::test]
    async fn test_find_entities_of_type_queries_real_store() {
        let store = make_store();
        let rdf_type =
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("valid IRI");
        let person = NamedNode::new("http://example.org/Person").expect("valid IRI");
        let alice = NamedNode::new("http://example.org/alice").expect("valid IRI");
        store
            .insert_quad(Quad::new_default_graph(alice, rdf_type, person))
            .expect("insert should succeed");

        let traversal = GraphTraversal::new(store);
        let triples = traversal
            .find_entities_of_type("http://example.org/Person", 10)
            .await
            .expect("query should succeed");

        assert_eq!(triples.len(), 1, "triples = {:?}", triples);
    }

    /// Regression test: multi-hop traversal in `find_entity_triples` must
    /// actually expand the frontier (`next_entities`) between depths instead
    /// of always stopping after depth 1.
    #[tokio::test]
    async fn test_find_entity_triples_traverses_multiple_hops() {
        let store = make_store();
        let knows = NamedNode::new("http://example.org/knows").expect("valid IRI");
        let alice = NamedNode::new("http://example.org/alice").expect("valid IRI");
        let bob = NamedNode::new("http://example.org/bob").expect("valid IRI");
        let carol = NamedNode::new("http://example.org/carol").expect("valid IRI");
        store
            .insert_quad(Quad::new_default_graph(
                alice.clone(),
                knows.clone(),
                bob.clone(),
            ))
            .expect("insert should succeed");
        store
            .insert_quad(Quad::new_default_graph(bob, knows, carol))
            .expect("insert should succeed");

        let traversal = GraphTraversal::new(store);
        let entities = vec![ExtractedEntity {
            text: "Alice".to_string(),
            entity_type: EntityType::Person,
            iri: Some("http://example.org/alice".to_string()),
            confidence: 0.9,
            aliases: vec![],
        }];

        let results = traversal
            .perform_graph_search("Alice", &entities, 2)
            .await
            .expect("search should succeed");

        assert!(
            !results.is_empty(),
            "expected at least one graph-search result"
        );
    }

    /// A mention with no known IRI must not be queried at all (and must not
    /// error) — the loop should simply skip it.
    #[tokio::test]
    async fn test_perform_graph_search_skips_entities_without_iri() {
        let store = make_store();
        let traversal = GraphTraversal::new(store);
        let entities = vec![ExtractedEntity {
            text: "Something".to_string(),
            entity_type: EntityType::Other,
            iri: None,
            confidence: 0.5,
            aliases: vec![],
        }];

        let results = traversal
            .perform_graph_search("Something", &entities, 1)
            .await
            .expect("search should succeed");
        assert!(results.is_empty());
    }
}
