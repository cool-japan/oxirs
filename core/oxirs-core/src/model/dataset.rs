//! RDF Dataset implementation

use std::collections::HashMap;
use std::iter::FromIterator;
use crate::model::{
    Graph, Quad, QuadRef, GraphName, GraphNameRef,
    Subject, Predicate, Object
};

/// An in-memory RDF Dataset
/// 
/// A dataset is a collection of named graphs plus a default graph.
/// Each named graph is identified by an IRI or blank node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dataset {
    default_graph: Graph,
    named_graphs: HashMap<GraphName, Graph>,
}

impl Dataset {
    /// Creates a new empty dataset
    pub fn new() -> Self {
        Dataset {
            default_graph: Graph::new(),
            named_graphs: HashMap::new(),
        }
    }
    
    /// Creates a new dataset with the specified capacity for named graphs
    pub fn with_capacity(capacity: usize) -> Self {
        Dataset {
            default_graph: Graph::new(),
            named_graphs: HashMap::with_capacity(capacity),
        }
    }
    
    /// Returns a reference to the default graph
    pub fn default_graph(&self) -> &Graph {
        &self.default_graph
    }
    
    /// Returns a mutable reference to the default graph
    pub fn default_graph_mut(&mut self) -> &mut Graph {
        &mut self.default_graph
    }
    
    /// Returns a reference to the named graph with the given name
    pub fn named_graph(&self, name: &GraphName) -> Option<&Graph> {
        if name.is_default_graph() {
            Some(&self.default_graph)
        } else {
            self.named_graphs.get(name)
        }
    }
    
    /// Returns a mutable reference to the named graph with the given name
    /// 
    /// Creates the graph if it doesn't exist.
    pub fn named_graph_mut(&mut self, name: &GraphName) -> &mut Graph {
        if name.is_default_graph() {
            &mut self.default_graph
        } else {
            self.named_graphs.entry(name.clone()).or_insert_with(Graph::new)
        }
    }
    
    /// Inserts a quad into the dataset
    /// 
    /// Returns `true` if the quad was not already present, `false` otherwise.
    pub fn insert(&mut self, quad: Quad) -> bool {
        let triple = quad.to_triple();
        let graph = self.named_graph_mut(quad.graph_name());
        graph.insert(triple)
    }
    
    /// Removes a quad from the dataset
    /// 
    /// Returns `true` if the quad was present, `false` otherwise.
    pub fn remove(&mut self, quad: &Quad) -> bool {
        let triple = quad.to_triple();
        if let Some(graph) = self.named_graphs.get_mut(quad.graph_name()) {
            graph.remove(&triple)
        } else if quad.graph_name().is_default_graph() {
            self.default_graph.remove(&triple)
        } else {
            false
        }
    }
    
    /// Returns `true` if the dataset contains the specified quad
    pub fn contains(&self, quad: &Quad) -> bool {
        let triple = quad.to_triple();
        if let Some(graph) = self.named_graph(quad.graph_name()) {
            graph.contains(&triple)
        } else {
            false
        }
    }
    
    /// Returns the total number of quads in the dataset
    pub fn len(&self) -> usize {
        self.default_graph.len() + 
        self.named_graphs.values().map(|g| g.len()).sum::<usize>()
    }
    
    /// Returns `true` if the dataset contains no quads
    pub fn is_empty(&self) -> bool {
        self.default_graph.is_empty() && 
        self.named_graphs.values().all(|g| g.is_empty())
    }
    
    /// Returns the number of named graphs (excluding default graph)
    pub fn named_graph_count(&self) -> usize {
        self.named_graphs.len()
    }
    
    /// Returns an iterator over all graph names (excluding default graph)
    pub fn graph_names(&self) -> impl Iterator<Item = &GraphName> {
        self.named_graphs.keys()
    }
    
    /// Returns an iterator over all named graphs
    pub fn named_graphs(&self) -> impl Iterator<Item = (&GraphName, &Graph)> {
        self.named_graphs.iter()
    }
    
    /// Clears the dataset, removing all quads
    pub fn clear(&mut self) {
        self.default_graph.clear();
        self.named_graphs.clear();
    }
    
    /// Removes a named graph from the dataset
    /// 
    /// Returns the removed graph if it existed.
    pub fn remove_graph(&mut self, name: &GraphName) -> Option<Graph> {
        if name.is_default_graph() {
            let mut graph = Graph::new();
            std::mem::swap(&mut graph, &mut self.default_graph);
            Some(graph)
        } else {
            self.named_graphs.remove(name)
        }
    }
    
    /// Returns an iterator over all quads in the dataset
    pub fn iter(&self) -> impl Iterator<Item = Quad> + '_ {
        let default_quads = self.default_graph.iter()
            .map(|triple| Quad::from_triple(triple.clone()));
            
        let named_quads = self.named_graphs.iter()
            .flat_map(|(name, graph)| {
                graph.iter().map(move |triple| {
                    Quad::from_triple_in_graph(triple.clone(), name.clone())
                })
            });
            
        default_quads.chain(named_quads)
    }
    
    /// Returns an iterator over all quads in the dataset as references
    pub fn iter_ref(&self) -> impl Iterator<Item = QuadRef<'_>> + '_ {
        let default_quads = self.default_graph.iter()
            .map(|triple| QuadRef::new(
                triple.subject().into(),
                triple.predicate().into(),
                triple.object().into(),
                GraphNameRef::DefaultGraph,
            ));
            
        let named_quads = self.named_graphs.iter()
            .flat_map(|(name, graph)| {
                graph.iter().map(move |triple| {
                    QuadRef::new(
                        triple.subject().into(),
                        triple.predicate().into(),
                        triple.object().into(),
                        name.into(),
                    )
                })
            });
            
        default_quads.chain(named_quads)
    }
    
    /// Finds all quads matching the given pattern
    /// 
    /// `None` values in the pattern act as wildcards.
    pub fn quads_for_pattern<'a>(
        &'a self,
        subject: Option<&'a Subject>,
        predicate: Option<&'a Predicate>,
        object: Option<&'a Object>,
        graph_name: Option<&'a GraphName>,
    ) -> Box<dyn Iterator<Item = Quad> + 'a> {
        if let Some(graph_name) = graph_name {
            if let Some(graph) = self.named_graph(graph_name) {
                let graph_name = graph_name.clone();
                Box::new(
                    graph.triples_for_pattern(subject, predicate, object)
                        .map(move |triple| Quad::from_triple_in_graph(triple.clone(), graph_name.clone()))
                ) as Box<dyn Iterator<Item = Quad> + '_>
            } else {
                Box::new(std::iter::empty())
            }
        } else {
            // Search all graphs
            Box::new(self.iter().filter(move |quad| {
                let triple = quad.to_triple();
                if let Some(s) = subject {
                    if triple.subject() != s {
                        return false;
                    }
                }
                if let Some(p) = predicate {
                    if triple.predicate() != p {
                        return false;
                    }
                }
                if let Some(o) = object {
                    if triple.object() != o {
                        return false;
                    }
                }
                true
            }))
        }
    }
    
    /// Extends the dataset with quads from an iterator
    pub fn extend<I>(&mut self, quads: I)
    where
        I: IntoIterator<Item = Quad>,
    {
        for quad in quads {
            self.insert(quad);
        }
    }
    
    /// Creates the union of this dataset with another dataset
    pub fn union(&self, other: &Dataset) -> Dataset {
        let mut result = self.clone();
        result.extend(other.iter());
        result
    }
}

impl Default for Dataset {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<Quad> for Dataset {
    fn from_iter<T: IntoIterator<Item = Quad>>(iter: T) -> Self {
        let mut dataset = Dataset::new();
        dataset.extend(iter);
        dataset
    }
}

impl Extend<Quad> for Dataset {
    fn extend<T: IntoIterator<Item = Quad>>(&mut self, iter: T) {
        for quad in iter {
            self.insert(quad);
        }
    }
}

impl IntoIterator for Dataset {
    type Item = Quad;
    type IntoIter = std::vec::IntoIter<Quad>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.iter().collect::<Vec<_>>().into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NamedNode, Literal};
    
    fn create_test_quad(graph_name: Option<NamedNode>) -> Quad {
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("object");
        
        if let Some(graph_name) = graph_name {
            Quad::new(subject, predicate, object, graph_name)
        } else {
            Quad::new_default_graph(subject, predicate, object)
        }
    }
    
    #[test]
    fn test_dataset_basic_operations() {
        let mut dataset = Dataset::new();
        let quad = create_test_quad(None);
        
        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);
        assert_eq!(dataset.named_graph_count(), 0);
        
        assert!(dataset.insert(quad.clone()));
        assert!(!dataset.is_empty());
        assert_eq!(dataset.len(), 1);
        assert!(dataset.contains(&quad));
        
        assert!(!dataset.insert(quad.clone())); // Already exists
        assert_eq!(dataset.len(), 1);
        
        assert!(dataset.remove(&quad));
        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);
        assert!(!dataset.contains(&quad));
    }
    
    #[test]
    fn test_dataset_named_graphs() {
        let mut dataset = Dataset::new();
        
        let quad1 = create_test_quad(None); // Default graph
        let graph_name = NamedNode::new("http://example.org/graph1").unwrap();
        let quad2 = create_test_quad(Some(graph_name.clone()));
        
        dataset.insert(quad1.clone());
        dataset.insert(quad2.clone());
        
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.named_graph_count(), 1);
        assert_eq!(dataset.default_graph().len(), 1);
        
        let named_graph = dataset.named_graph(&GraphName::NamedNode(graph_name.clone())).unwrap();
        assert_eq!(named_graph.len(), 1);
        
        // Test graph names iterator
        let graph_names: Vec<_> = dataset.graph_names().collect();
        assert_eq!(graph_names.len(), 1);
        assert!(graph_names.contains(&&GraphName::NamedNode(graph_name)));
    }
    
    #[test]
    fn test_dataset_pattern_matching() {
        let mut dataset = Dataset::new();
        
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate1 = NamedNode::new("http://example.org/predicate1").unwrap();
        let predicate2 = NamedNode::new("http://example.org/predicate2").unwrap();
        let object = Literal::new("object");
        let graph_name = NamedNode::new("http://example.org/graph").unwrap();
        
        let quad1 = Quad::new_default_graph(subject.clone(), predicate1.clone(), object.clone());
        let quad2 = Quad::new(subject.clone(), predicate2, object, graph_name.clone());
        
        dataset.insert(quad1.clone());
        dataset.insert(quad2.clone());
        
        // Find by subject
        let by_subject: Vec<_> = dataset.quads_for_pattern(
            Some(&Subject::NamedNode(subject.clone())),
            None,
            None,
            None
        ).collect();
        assert_eq!(by_subject.len(), 2);
        
        // Find by graph
        let by_graph: Vec<_> = dataset.quads_for_pattern(
            None,
            None,
            None,
            Some(&GraphName::NamedNode(graph_name))
        ).collect();
        assert_eq!(by_graph.len(), 1);
        assert_eq!(by_graph[0], quad2);
        
        // Find in default graph
        let by_default_graph: Vec<_> = dataset.quads_for_pattern(
            None,
            None,
            None,
            Some(&GraphName::DefaultGraph)
        ).collect();
        assert_eq!(by_default_graph.len(), 1);
        assert_eq!(by_default_graph[0], quad1);
    }
    
    #[test]
    fn test_dataset_iteration() {
        let mut dataset = Dataset::new();
        
        let quad1 = create_test_quad(None);
        let graph_name = NamedNode::new("http://example.org/graph").unwrap();
        let quad2 = create_test_quad(Some(graph_name));
        
        dataset.insert(quad1.clone());
        dataset.insert(quad2.clone());
        
        let quads: Vec<_> = dataset.iter().collect();
        assert_eq!(quads.len(), 2);
        assert!(quads.contains(&quad1));
        assert!(quads.contains(&quad2));
    }
    
    #[test]
    fn test_dataset_remove_graph() {
        let mut dataset = Dataset::new();
        
        let graph_name = NamedNode::new("http://example.org/graph").unwrap();
        let quad = create_test_quad(Some(graph_name.clone()));
        
        dataset.insert(quad);
        assert_eq!(dataset.named_graph_count(), 1);
        
        let removed_graph = dataset.remove_graph(&GraphName::NamedNode(graph_name));
        assert!(removed_graph.is_some());
        assert_eq!(dataset.named_graph_count(), 0);
        assert_eq!(dataset.len(), 0);
    }
}