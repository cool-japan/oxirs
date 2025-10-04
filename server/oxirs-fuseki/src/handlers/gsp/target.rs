//! Graph Target Determination and Access

use super::types::{GraphTarget, GspError};
use oxirs_core::model::{GraphName, NamedNode, Quad, Triple};
use oxirs_core::Store;
use std::sync::Arc;

/// Represents access to a specific graph in the store
pub struct GraphAccess {
    target: GraphTarget,
    exists: bool,
}

impl GraphAccess {
    /// Create a new graph access
    pub fn new(target: GraphTarget, store: &dyn Store) -> Self {
        let exists = Self::check_exists(&target, store);
        Self { target, exists }
    }

    /// Check if the graph exists in the store
    fn check_exists(target: &GraphTarget, store: &dyn Store) -> bool {
        match target {
            GraphTarget::Default => {
                // Default graph always exists
                true
            }
            GraphTarget::Union => {
                // Union graph is a virtual view, always exists
                true
            }
            GraphTarget::Named(uri) => {
                // Check if any quads exist in this named graph
                if let Ok(node) = NamedNode::new(uri) {
                    let graph_name = GraphName::NamedNode(node);
                    // Try to get at least one quad from this graph
                    store
                        .find_quads(None, None, None, Some(&graph_name))
                        .map(|quads| !quads.is_empty())
                        .unwrap_or(false)
                } else {
                    false
                }
            }
        }
    }

    /// Check if this graph exists
    pub fn exists(&self) -> bool {
        self.exists
    }

    /// Get the graph target
    pub fn target(&self) -> &GraphTarget {
        &self.target
    }

    /// Get label for logging/errors
    pub fn label(&self) -> String {
        self.target.label()
    }

    /// Check if this graph is writable
    pub fn is_writable(&self) -> bool {
        self.target.is_writable()
    }

    /// Get all triples from this graph
    pub fn get_triples(&self, store: &dyn Store) -> Result<Vec<Triple>, GspError> {
        match &self.target {
            GraphTarget::Default => {
                // Get all triples from default graph
                let quads = store
                    .find_quads(None, None, None, Some(&GraphName::DefaultGraph))
                    .map_err(|e| GspError::StoreError(e.to_string()))?;

                let triples: Vec<Triple> = quads
                    .into_iter()
                    .map(|quad| {
                        Triple::new(
                            quad.subject().clone(),
                            quad.predicate().clone(),
                            quad.object().clone(),
                        )
                    })
                    .collect();
                Ok(triples)
            }
            GraphTarget::Union => {
                // Get all triples from all graphs (union)
                let quads = store
                    .find_quads(None, None, None, None)
                    .map_err(|e| GspError::StoreError(e.to_string()))?;

                let triples: Vec<Triple> = quads
                    .into_iter()
                    .map(|quad| {
                        Triple::new(
                            quad.subject().clone(),
                            quad.predicate().clone(),
                            quad.object().clone(),
                        )
                    })
                    .collect();
                Ok(triples)
            }
            GraphTarget::Named(uri) => {
                // Get all triples from named graph
                let node = NamedNode::new(uri)
                    .map_err(|e| GspError::BadRequest(format!("Invalid graph URI: {}", e)))?;
                let graph_name = GraphName::NamedNode(node);

                let quads = store
                    .find_quads(None, None, None, Some(&graph_name))
                    .map_err(|e| GspError::StoreError(e.to_string()))?;

                let triples: Vec<Triple> = quads
                    .into_iter()
                    .map(|quad| {
                        Triple::new(
                            quad.subject().clone(),
                            quad.predicate().clone(),
                            quad.object().clone(),
                        )
                    })
                    .collect();
                Ok(triples)
            }
        }
    }

    /// Replace all triples in this graph
    pub fn replace_triples(
        &self,
        store: &dyn Store,
        triples: Vec<Triple>,
    ) -> Result<usize, GspError> {
        if !self.is_writable() {
            return Err(GspError::MethodNotAllowed(format!(
                "Cannot write to {}",
                self.label()
            )));
        }

        match &self.target {
            GraphTarget::Default => {
                // Clear default graph
                let existing = store
                    .find_quads(None, None, None, Some(&GraphName::DefaultGraph))
                    .map_err(|e| GspError::StoreError(e.to_string()))?;

                for quad in existing {
                    store
                        .remove_quad(&quad)
                        .map_err(|e| GspError::StoreError(e.to_string()))?;
                }

                // Insert new triples
                for triple in &triples {
                    let quad = Quad::new(
                        triple.subject().clone(),
                        triple.predicate().clone(),
                        triple.object().clone(),
                        GraphName::DefaultGraph,
                    );
                    store
                        .insert_quad(quad)
                        .map_err(|e| GspError::StoreError(e.to_string()))?;
                }

                Ok(triples.len())
            }
            GraphTarget::Named(uri) => {
                let node = NamedNode::new(uri)
                    .map_err(|e| GspError::BadRequest(format!("Invalid graph URI: {}", e)))?;
                let graph_name = GraphName::NamedNode(node);

                // Clear named graph
                let existing = store
                    .find_quads(None, None, None, Some(&graph_name))
                    .map_err(|e| GspError::StoreError(e.to_string()))?;

                for quad in existing {
                    store
                        .remove_quad(&quad)
                        .map_err(|e| GspError::StoreError(e.to_string()))?;
                }

                // Insert new triples
                for triple in &triples {
                    let quad = Quad::new(
                        triple.subject().clone(),
                        triple.predicate().clone(),
                        triple.object().clone(),
                        graph_name.clone(),
                    );
                    store
                        .insert_quad(quad)
                        .map_err(|e| GspError::StoreError(e.to_string()))?;
                }

                Ok(triples.len())
            }
            GraphTarget::Union => Err(GspError::MethodNotAllowed(
                "Cannot write to union graph".to_string(),
            )),
        }
    }

    /// Add triples to this graph
    pub fn add_triples(&self, store: &dyn Store, triples: Vec<Triple>) -> Result<usize, GspError> {
        if !self.is_writable() {
            return Err(GspError::MethodNotAllowed(format!(
                "Cannot write to {}",
                self.label()
            )));
        }

        let graph_name = match &self.target {
            GraphTarget::Default => GraphName::DefaultGraph,
            GraphTarget::Named(uri) => {
                let node = NamedNode::new(uri)
                    .map_err(|e| GspError::BadRequest(format!("Invalid graph URI: {}", e)))?;
                GraphName::NamedNode(node)
            }
            GraphTarget::Union => {
                return Err(GspError::MethodNotAllowed(
                    "Cannot write to union graph".to_string(),
                ))
            }
        };

        // Insert new triples
        for triple in &triples {
            let quad = Quad::new(
                triple.subject().clone(),
                triple.predicate().clone(),
                triple.object().clone(),
                graph_name.clone(),
            );
            store
                .insert_quad(quad)
                .map_err(|e| GspError::StoreError(e.to_string()))?;
        }

        Ok(triples.len())
    }

    /// Delete all triples from this graph
    pub fn delete_graph(&self, store: &dyn Store) -> Result<usize, GspError> {
        if !self.is_writable() {
            return Err(GspError::MethodNotAllowed(format!(
                "Cannot delete {}",
                self.label()
            )));
        }

        let graph_name = match &self.target {
            GraphTarget::Default => GraphName::DefaultGraph,
            GraphTarget::Named(uri) => {
                let node = NamedNode::new(uri)
                    .map_err(|e| GspError::BadRequest(format!("Invalid graph URI: {}", e)))?;
                GraphName::NamedNode(node)
            }
            GraphTarget::Union => {
                return Err(GspError::MethodNotAllowed(
                    "Cannot delete union graph".to_string(),
                ))
            }
        };

        // Get all quads in this graph
        let existing = store
            .find_quads(None, None, None, Some(&graph_name))
            .map_err(|e| GspError::StoreError(e.to_string()))?;

        let count = existing.len();

        // Delete all quads
        for quad in existing {
            store
                .remove_quad(&quad)
                .map_err(|e| GspError::StoreError(e.to_string()))?;
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::Literal;
    use oxirs_core::rdf_store::ConcreteStore;

    fn setup_test_store() -> ConcreteStore {
        let store = ConcreteStore::new().unwrap();

        // Add some test data to default graph
        let s1 = NamedNode::new("http://example.org/s1").unwrap();
        let p1 = NamedNode::new("http://example.org/p1").unwrap();
        let o1 = Literal::new_simple_literal("value1");
        let triple1 = Triple::new(s1, p1, o1);
        store.insert_triple(triple1).unwrap();

        store
    }

    #[test]
    fn test_graph_access_default() {
        let store = setup_test_store();
        let target = GraphTarget::Default;
        let access = GraphAccess::new(target, &store);

        assert!(access.exists());
        assert!(access.is_writable());
    }

    #[test]
    fn test_graph_access_union() {
        let store = setup_test_store();
        let target = GraphTarget::Union;
        let access = GraphAccess::new(target, &store);

        assert!(access.exists());
        assert!(!access.is_writable());
    }

    #[test]
    fn test_get_triples_default_graph() {
        let store = setup_test_store();
        let target = GraphTarget::Default;
        let access = GraphAccess::new(target, &store);

        let triples = access.get_triples(&store).unwrap();
        assert!(!triples.is_empty());
    }

    #[test]
    fn test_replace_triples() {
        let store = setup_test_store();
        let target = GraphTarget::Default;
        let access = GraphAccess::new(target, &store);

        let s = NamedNode::new("http://example.org/new").unwrap();
        let p = NamedNode::new("http://example.org/pred").unwrap();
        let o = Literal::new_simple_literal("new value");
        let triple = Triple::new(s, p, o);

        let count = access.replace_triples(&store, vec![triple]).unwrap();
        assert_eq!(count, 1);

        let triples = access.get_triples(&store).unwrap();
        assert_eq!(triples.len(), 1);
    }
}
