//! Concrete store implementations

use super::*;
use crate::Result;
use std::sync::RwLock;

pub struct ConcreteStore {
    inner: RwLock<RdfStore>,
}

impl ConcreteStore {
    pub fn new() -> Result<Self> {
        Ok(ConcreteStore {
            inner: RwLock::new(RdfStore::new()?),
        })
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(ConcreteStore {
            inner: RwLock::new(RdfStore::open(path)?),
        })
    }

    /// Insert a quad into the store
    pub fn insert_quad(&self, quad: Quad) -> Result<bool> {
        let mut inner = self.inner.write().map_err(|e| {
            crate::OxirsError::Store(format!("Failed to acquire write lock: {}", e))
        })?;
        RdfStore::insert_quad(&mut inner, quad)
    }

    /// Remove a quad from the store
    pub fn remove_quad(&self, quad: &Quad) -> Result<bool> {
        let mut inner = self.inner.write().map_err(|e| {
            crate::OxirsError::Store(format!("Failed to acquire write lock: {}", e))
        })?;
        RdfStore::remove_quad(&mut inner, quad)
    }

    /// Insert a triple into the default graph
    pub fn insert_triple(&self, triple: crate::model::Triple) -> Result<bool> {
        let mut inner = self.inner.write().map_err(|e| {
            crate::OxirsError::Store(format!("Failed to acquire write lock: {}", e))
        })?;
        RdfStore::insert_triple(&mut inner, triple)
    }
}

impl Default for ConcreteStore {
    fn default() -> Self {
        ConcreteStore::new().unwrap()
    }
}

// Implement Store trait for ConcreteStore
#[async_trait]
impl Store for ConcreteStore {
    fn insert_quad(&self, quad: Quad) -> Result<bool> {
        ConcreteStore::insert_quad(self, quad)
    }

    fn remove_quad(&self, quad: &Quad) -> Result<bool> {
        ConcreteStore::remove_quad(self, quad)
    }

    fn find_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Result<Vec<Quad>> {
        let inner = self
            .inner
            .read()
            .map_err(|e| crate::OxirsError::Store(format!("Failed to acquire read lock: {}", e)))?;
        inner.find_quads(subject, predicate, object, graph_name)
    }

    fn is_ready(&self) -> bool {
        self.inner
            .read()
            .map(|inner| inner.is_ready())
            .unwrap_or(false)
    }

    fn len(&self) -> Result<usize> {
        let inner = self
            .inner
            .read()
            .map_err(|e| crate::OxirsError::Store(format!("Failed to acquire read lock: {}", e)))?;
        inner.len()
    }

    fn is_empty(&self) -> Result<bool> {
        let inner = self
            .inner
            .read()
            .map_err(|e| crate::OxirsError::Store(format!("Failed to acquire read lock: {}", e)))?;
        inner.is_empty()
    }

    fn query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        let inner = self
            .inner
            .read()
            .map_err(|e| crate::OxirsError::Store(format!("Failed to acquire read lock: {}", e)))?;
        inner.query(sparql)
    }

    fn prepare_query(&self, sparql: &str) -> Result<PreparedQuery> {
        let inner = self
            .inner
            .read()
            .map_err(|e| crate::OxirsError::Store(format!("Failed to acquire read lock: {}", e)))?;
        inner.prepare_query(sparql)
    }
}

/// Query results container for SPARQL queries
#[derive(Debug, Clone)]
pub struct OxirsQueryResults {
    results: QueryResults,
    variables: Vec<String>,
}

impl OxirsQueryResults {
    pub fn new() -> Self {
        OxirsQueryResults {
            results: QueryResults::empty_bindings(),
            variables: Vec::new(),
        }
    }

    pub fn from_bindings(bindings: Vec<VariableBinding>, variables: Vec<String>) -> Self {
        OxirsQueryResults {
            results: QueryResults::Bindings(bindings),
            variables,
        }
    }

    pub fn from_boolean(value: bool) -> Self {
        OxirsQueryResults {
            results: QueryResults::Boolean(value),
            variables: Vec::new(),
        }
    }

    pub fn from_graph(quads: Vec<Quad>) -> Self {
        OxirsQueryResults {
            results: QueryResults::Graph(quads),
            variables: Vec::new(),
        }
    }

    pub fn results(&self) -> &QueryResults {
        &self.results
    }

    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    pub fn len(&self) -> usize {
        self.results.len()
    }
}

impl Default for OxirsQueryResults {
    fn default() -> Self {
        Self::new()
    }
}
