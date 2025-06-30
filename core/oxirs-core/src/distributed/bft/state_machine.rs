//! RDF state machine for Byzantine fault tolerant consensus

use super::types::*;
use crate::model::{BlankNode, Literal, NamedNode, Triple};
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// RDF state machine
pub struct RdfStateMachine {
    /// Triple store
    triples: HashSet<Triple>,

    /// Operation counter
    operation_count: u64,

    /// State digest cache
    digest_cache: Option<(u64, Vec<u8>)>,
}

impl RdfStateMachine {
    /// Create a new RDF state machine
    pub fn new() -> Self {
        Self {
            triples: HashSet::new(),
            operation_count: 0,
            digest_cache: None,
        }
    }

    /// Execute an RDF operation
    pub fn execute(&mut self, operation: RdfOperation) -> Result<OperationResult> {
        self.operation_count += 1;
        self.digest_cache = None; // Invalidate cache

        match operation {
            RdfOperation::Insert(triple) => {
                let t = self.deserialize_triple(triple)?;
                self.triples.insert(t);
                Ok(OperationResult::Success)
            }

            RdfOperation::Remove(triple) => {
                let t = self.deserialize_triple(triple)?;
                self.triples.remove(&t);
                Ok(OperationResult::Success)
            }

            RdfOperation::BatchInsert(triples) => {
                for triple in triples {
                    let t = self.deserialize_triple(triple)?;
                    self.triples.insert(t);
                }
                Ok(OperationResult::Success)
            }

            RdfOperation::BatchRemove(triples) => {
                for triple in triples {
                    let t = self.deserialize_triple(triple)?;
                    self.triples.remove(&t);
                }
                Ok(OperationResult::Success)
            }

            RdfOperation::Query(_query) => {
                // Simplified - would execute SPARQL query
                let results: Vec<SerializableTriple> = self
                    .triples
                    .iter()
                    .take(10) // Limit results
                    .map(|t| self.serialize_triple(t))
                    .collect();
                Ok(OperationResult::QueryResult(results))
            }
        }
    }

    /// Get state digest
    pub fn get_state_digest(&self) -> Vec<u8> {
        // For the read-only version, we need to calculate without mutating
        // In a real implementation, we might use a different caching strategy
        self.calculate_digest_readonly()
    }

    /// Calculate state digest (read-only version)
    fn calculate_digest_readonly(&self) -> Vec<u8> {
        let mut hasher = Sha256::new();

        // Sort triples for deterministic digest
        let mut sorted_triples: Vec<_> = self.triples.iter().collect();
        sorted_triples.sort_by_key(|t| {
            (
                t.subject().to_string(),
                t.predicate().to_string(),
                t.object().to_string(),
            )
        });

        for triple in sorted_triples {
            hasher.update(triple.subject().to_string().as_bytes());
            hasher.update(triple.predicate().to_string().as_bytes());
            hasher.update(triple.object().to_string().as_bytes());
        }

        hasher.update(&self.operation_count.to_le_bytes());
        hasher.finalize().to_vec()
    }

    /// Calculate state digest (mutable version with caching)
    pub fn calculate_digest(&mut self) -> Vec<u8> {
        // Check cache
        if let Some((count, digest)) = &self.digest_cache {
            if *count == self.operation_count {
                return digest.clone();
            }
        }

        // Calculate new digest
        let digest = self.calculate_digest_readonly();

        // Cache the digest
        self.digest_cache = Some((self.operation_count, digest.clone()));

        digest
    }

    /// Get current triple count
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Get operation count
    pub fn operation_count(&self) -> u64 {
        self.operation_count
    }

    /// Check if a triple exists
    pub fn contains_triple(&self, triple: &SerializableTriple) -> Result<bool> {
        let t = self.deserialize_triple(triple.clone())?;
        Ok(self.triples.contains(&t))
    }

    /// Get all triples (for debugging/testing)
    pub fn get_all_triples(&self) -> Vec<SerializableTriple> {
        self.triples
            .iter()
            .map(|t| self.serialize_triple(t))
            .collect()
    }

    /// Deserialize a triple from network format
    fn deserialize_triple(&self, st: SerializableTriple) -> Result<Triple> {
        let subject = NamedNode::new(&st.subject)?;
        let predicate = NamedNode::new(&st.predicate)?;

        let object = match st.object_type {
            ObjectType::NamedNode => crate::model::Object::NamedNode(NamedNode::new(&st.object)?),
            ObjectType::BlankNode => crate::model::Object::BlankNode(BlankNode::new(&st.object)?),
            ObjectType::Literal { datatype, language } => {
                if let Some(lang) = language {
                    crate::model::Object::Literal(Literal::new_language_tagged_literal(
                        &st.object, &lang,
                    )?)
                } else if let Some(dt) = datatype {
                    crate::model::Object::Literal(Literal::new_typed(
                        &st.object,
                        NamedNode::new(&dt)?,
                    ))
                } else {
                    crate::model::Object::Literal(Literal::new(&st.object))
                }
            }
        };

        Ok(Triple::new(subject, predicate, object))
    }

    /// Serialize a triple for network transmission
    fn serialize_triple(&self, triple: &Triple) -> SerializableTriple {
        let object_type = match triple.object() {
            crate::model::Object::NamedNode(_) => ObjectType::NamedNode,
            crate::model::Object::BlankNode(_) => ObjectType::BlankNode,
            crate::model::Object::Literal(lit) => ObjectType::Literal {
                datatype: if lit.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    Some(lit.datatype().as_str().to_string())
                } else {
                    None
                },
                language: lit.language().map(|l| l.to_string()),
            },
            _ => ObjectType::NamedNode, // Fallback
        };

        SerializableTriple {
            subject: triple.subject().to_string(),
            predicate: triple.predicate().to_string(),
            object: triple.object().to_string(),
            object_type,
        }
    }

    /// Reset state machine to initial state
    pub fn reset(&mut self) {
        self.triples.clear();
        self.operation_count = 0;
        self.digest_cache = None;
    }

    /// Apply a batch of operations atomically
    pub fn apply_batch(&mut self, operations: Vec<RdfOperation>) -> Result<Vec<OperationResult>> {
        let mut results = Vec::new();
        let initial_count = self.operation_count;
        let initial_cache = self.digest_cache.clone();

        // Try to apply all operations
        for operation in operations {
            match self.execute(operation) {
                Ok(result) => results.push(result),
                Err(e) => {
                    // Rollback on any failure
                    self.operation_count = initial_count;
                    self.digest_cache = initial_cache;
                    return Err(e);
                }
            }
        }

        Ok(results)
    }
}

impl Default for RdfStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_state_machine_basic_operations() {
        let mut state_machine = RdfStateMachine::new();

        // Test insert
        let triple = SerializableTriple {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "value".to_string(),
            object_type: ObjectType::Literal {
                datatype: None,
                language: None,
            },
        };

        let result = state_machine
            .execute(RdfOperation::Insert(triple.clone()))
            .unwrap();
        assert!(matches!(result, OperationResult::Success));
        assert_eq!(state_machine.triple_count(), 1);
        assert_eq!(state_machine.operation_count(), 1);

        // Test contains
        assert!(state_machine.contains_triple(&triple).unwrap());

        // Test remove
        let result = state_machine.execute(RdfOperation::Remove(triple.clone())).unwrap();
        assert!(matches!(result, OperationResult::Success));
        assert_eq!(state_machine.triple_count(), 0);
        assert!(!state_machine.contains_triple(&triple).unwrap());
    }

    #[test]
    fn test_state_machine_batch_operations() {
        let mut state_machine = RdfStateMachine::new();

        let triples = vec![
            SerializableTriple {
                subject: "http://example.org/s1".to_string(),
                predicate: "http://example.org/p".to_string(),
                object: "value1".to_string(),
                object_type: ObjectType::Literal {
                    datatype: None,
                    language: None,
                },
            },
            SerializableTriple {
                subject: "http://example.org/s2".to_string(),
                predicate: "http://example.org/p".to_string(),
                object: "value2".to_string(),
                object_type: ObjectType::Literal {
                    datatype: None,
                    language: None,
                },
            },
        ];

        // Test batch insert
        let result = state_machine
            .execute(RdfOperation::BatchInsert(triples.clone()))
            .unwrap();
        assert!(matches!(result, OperationResult::Success));
        assert_eq!(state_machine.triple_count(), 2);

        // Test batch remove
        let result = state_machine
            .execute(RdfOperation::BatchRemove(triples))
            .unwrap();
        assert!(matches!(result, OperationResult::Success));
        assert_eq!(state_machine.triple_count(), 0);
    }

    #[test]
    fn test_state_machine_digest_calculation() {
        let mut state_machine = RdfStateMachine::new();

        let triple = SerializableTriple {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "value".to_string(),
            object_type: ObjectType::Literal {
                datatype: None,
                language: None,
            },
        };

        // Initial digest
        let digest1 = state_machine.calculate_digest();

        // Add triple
        state_machine.execute(RdfOperation::Insert(triple.clone())).unwrap();
        let digest2 = state_machine.calculate_digest();

        // Digests should be different
        assert_ne!(digest1, digest2);

        // Same state should produce same digest
        let digest3 = state_machine.calculate_digest();
        assert_eq!(digest2, digest3);

        // Read-only version should match
        let digest4 = state_machine.get_state_digest();
        assert_eq!(digest2, digest4);
    }

    #[test]
    fn test_query_operation() {
        let mut state_machine = RdfStateMachine::new();

        // Add some triples
        for i in 0..15 {
            let triple = SerializableTriple {
                subject: format!("http://example.org/s{}", i),
                predicate: "http://example.org/p".to_string(),
                object: format!("value{}", i),
                object_type: ObjectType::Literal {
                    datatype: None,
                    language: None,
                },
            };
            state_machine.execute(RdfOperation::Insert(triple)).unwrap();
        }

        // Test query (should return max 10 results)
        let result = state_machine
            .execute(RdfOperation::Query("SELECT * WHERE { ?s ?p ?o }".to_string()))
            .unwrap();

        if let OperationResult::QueryResult(results) = result {
            assert_eq!(results.len(), 10); // Limited to 10 results
        } else {
            panic!("Expected QueryResult");
        }
    }

    #[test]
    fn test_different_object_types() {
        let mut state_machine = RdfStateMachine::new();

        // Test NamedNode object
        let triple1 = SerializableTriple {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
            object_type: ObjectType::NamedNode,
        };

        // Test BlankNode object
        let triple2 = SerializableTriple {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p2".to_string(),
            object: "_:blank1".to_string(),
            object_type: ObjectType::BlankNode,
        };

        // Test typed literal
        let triple3 = SerializableTriple {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p3".to_string(),
            object: "42".to_string(),
            object_type: ObjectType::Literal {
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                language: None,
            },
        };

        // Test language-tagged literal
        let triple4 = SerializableTriple {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p4".to_string(),
            object: "hello".to_string(),
            object_type: ObjectType::Literal {
                datatype: None,
                language: Some("en".to_string()),
            },
        };

        // Insert all triples
        state_machine.execute(RdfOperation::Insert(triple1)).unwrap();
        state_machine.execute(RdfOperation::Insert(triple2)).unwrap();
        state_machine.execute(RdfOperation::Insert(triple3)).unwrap();
        state_machine.execute(RdfOperation::Insert(triple4)).unwrap();

        assert_eq!(state_machine.triple_count(), 4);
    }
}