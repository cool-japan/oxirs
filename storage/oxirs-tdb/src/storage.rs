//! # TDB Storage Backend
//!
//! Core storage layer for TDB implementation.

use anyhow::Result;

/// Storage backend interface
pub trait StorageBackend {
    /// Insert a quad into storage
    fn insert_quad(&mut self, subject: &str, predicate: &str, object: &str, graph: Option<&str>) -> Result<()>;
    
    /// Query quads from storage
    fn query_quads(&self, pattern: QuadPattern) -> Result<Vec<Quad>>;
    
    /// Remove a quad from storage
    fn remove_quad(&mut self, subject: &str, predicate: &str, object: &str, graph: Option<&str>) -> Result<()>;
}

/// Quad pattern for querying
#[derive(Debug, Clone)]
pub struct QuadPattern {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub graph: Option<String>,
}

/// RDF Quad representation
#[derive(Debug, Clone)]
pub struct Quad {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub graph: Option<String>,
}

/// Memory storage backend implementation
pub struct MemoryStorage {
    quads: Vec<Quad>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            quads: Vec::new(),
        }
    }
}

impl StorageBackend for MemoryStorage {
    fn insert_quad(&mut self, subject: &str, predicate: &str, object: &str, graph: Option<&str>) -> Result<()> {
        let quad = Quad {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            graph: graph.map(|g| g.to_string()),
        };
        self.quads.push(quad);
        Ok(())
    }
    
    fn query_quads(&self, pattern: QuadPattern) -> Result<Vec<Quad>> {
        let results = self.quads.iter()
            .filter(|quad| {
                if let Some(ref subject) = pattern.subject {
                    if &quad.subject != subject {
                        return false;
                    }
                }
                if let Some(ref predicate) = pattern.predicate {
                    if &quad.predicate != predicate {
                        return false;
                    }
                }
                if let Some(ref object) = pattern.object {
                    if &quad.object != object {
                        return false;
                    }
                }
                if let Some(ref graph) = pattern.graph {
                    if quad.graph.as_ref() != Some(graph) {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();
        Ok(results)
    }
    
    fn remove_quad(&mut self, subject: &str, predicate: &str, object: &str, graph: Option<&str>) -> Result<()> {
        self.quads.retain(|quad| {
            !(quad.subject == subject && 
              quad.predicate == predicate && 
              quad.object == object && 
              quad.graph.as_deref() == graph)
        });
        Ok(())
    }
}