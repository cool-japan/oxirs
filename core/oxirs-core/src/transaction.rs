//! Transaction support for OxiRS

use crate::{model::Quad, rdf_store::RdfStore, OxirsError, Result};

/// A transaction for atomic operations on the RDF store
pub struct Transaction {
    /// Changes to be applied when the transaction commits
    pending_inserts: Vec<Quad>,
    /// Quads to be removed when the transaction commits
    pending_removes: Vec<Quad>,
    /// Reference to the underlying store
    store: *mut RdfStore,
    /// Whether the transaction has been committed or aborted
    finished: bool,
}

// SAFETY: Transaction is only accessed from a single thread
unsafe impl Send for Transaction {}

impl Transaction {
    /// Create a new transaction
    pub(crate) fn new(store: &mut RdfStore) -> Self {
        Self {
            pending_inserts: Vec::new(),
            pending_removes: Vec::new(),
            store: store as *mut RdfStore,
            finished: false,
        }
    }

    /// Insert a quad into the transaction
    pub fn insert<'a>(&mut self, quad: impl Into<crate::model::QuadRef<'a>>) -> Result<bool> {
        if self.finished {
            return Err(OxirsError::Store(
                "Transaction already finished".to_string(),
            ));
        }

        let quad_ref = quad.into();
        let quad = Quad::new(
            quad_ref.subject().to_owned(),
            quad_ref.predicate().to_owned(),
            quad_ref.object().to_owned(),
            quad_ref.graph_name().to_owned(),
        );

        // Check if already in pending inserts
        if self.pending_inserts.contains(&quad) {
            return Ok(false);
        }

        // Add to pending inserts
        self.pending_inserts.push(quad);
        Ok(true)
    }

    /// Remove a quad from the transaction
    pub fn remove<'a>(&mut self, quad: impl Into<crate::model::QuadRef<'a>>) -> Result<bool> {
        if self.finished {
            return Err(OxirsError::Store(
                "Transaction already finished".to_string(),
            ));
        }

        let quad_ref = quad.into();
        let quad = Quad::new(
            quad_ref.subject().to_owned(),
            quad_ref.predicate().to_owned(),
            quad_ref.object().to_owned(),
            quad_ref.graph_name().to_owned(),
        );

        // Check if already in pending removes
        if self.pending_removes.contains(&quad) {
            return Ok(false);
        }

        // Add to pending removes
        self.pending_removes.push(quad);
        Ok(true)
    }

    /// Check if the transaction contains a quad (considering pending changes)
    pub fn contains<'a>(&self, quad: impl Into<crate::model::QuadRef<'a>>) -> Result<bool> {
        if self.finished {
            return Err(OxirsError::Store(
                "Transaction already finished".to_string(),
            ));
        }

        let quad_ref = quad.into();
        let quad = Quad::new(
            quad_ref.subject().to_owned(),
            quad_ref.predicate().to_owned(),
            quad_ref.object().to_owned(),
            quad_ref.graph_name().to_owned(),
        );

        // Check pending removes first
        if self.pending_removes.contains(&quad) {
            return Ok(false);
        }

        // Check pending inserts
        if self.pending_inserts.contains(&quad) {
            return Ok(true);
        }

        // Check the underlying store
        // SAFETY: We ensure the store pointer remains valid for the transaction lifetime
        unsafe {
            let store = &*self.store;
            store.contains_quad(&quad)
        }
    }

    /// Commit the transaction
    pub fn commit(mut self) -> Result<()> {
        if self.finished {
            return Err(OxirsError::Store(
                "Transaction already finished".to_string(),
            ));
        }

        // SAFETY: We ensure the store pointer remains valid for the transaction lifetime
        unsafe {
            let store = &mut *self.store;

            // Apply all pending removes first
            for quad in &self.pending_removes {
                store.remove_quad(quad)?;
            }

            // Then apply all pending inserts
            for quad in &self.pending_inserts {
                store.insert_quad(quad.clone())?;
            }
        }

        self.finished = true;
        Ok(())
    }

    /// Abort the transaction (drop without committing)
    pub fn abort(mut self) {
        self.finished = true;
        // Pending changes are just dropped
    }

    /// Get the number of pending inserts
    pub fn pending_insert_count(&self) -> usize {
        self.pending_inserts.len()
    }

    /// Get the number of pending removes
    pub fn pending_remove_count(&self) -> usize {
        self.pending_removes.len()
    }

    /// Check if the transaction has any pending changes
    pub fn has_pending_changes(&self) -> bool {
        !self.pending_inserts.is_empty() || !self.pending_removes.is_empty()
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        if !self.finished && self.has_pending_changes() {
            // Log a warning if transaction is dropped without explicit commit/abort
            eprintln!("Warning: Transaction dropped without explicit commit or abort");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::{GraphNameRef, Literal, NamedNode, ObjectRef, PredicateRef, QuadRef, SubjectRef},
        rdf_store::RdfStore,
    };

    #[test]
    fn test_transaction_basic_operations() {
        let mut store = RdfStore::new().unwrap();
        let mut transaction = Transaction::new(&mut store);

        // Create test data
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("object");

        // Insert into transaction
        let inserted = transaction
            .insert(QuadRef::new(
                SubjectRef::NamedNode(&subject),
                PredicateRef::NamedNode(&predicate),
                ObjectRef::Literal(&object),
                GraphNameRef::DefaultGraph,
            ))
            .unwrap();
        assert!(inserted);

        // Check it's in the transaction
        let contains = transaction
            .contains(QuadRef::new(
                SubjectRef::NamedNode(&subject),
                PredicateRef::NamedNode(&predicate),
                ObjectRef::Literal(&object),
                GraphNameRef::DefaultGraph,
            ))
            .unwrap();
        assert!(contains);

        // Commit transaction
        transaction.commit().unwrap();

        // Check it's now in the store
        let quad = Quad::new(
            crate::model::Subject::NamedNode(subject),
            crate::model::Predicate::NamedNode(predicate),
            crate::model::Object::Literal(object),
            crate::model::GraphName::DefaultGraph,
        );
        assert!(store.contains_quad(&quad).unwrap());
    }

    #[test]
    fn test_transaction_abort() {
        let mut store = RdfStore::new().unwrap();
        let mut transaction = Transaction::new(&mut store);

        // Create test data
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("object");

        // Insert into transaction
        transaction
            .insert(QuadRef::new(
                SubjectRef::NamedNode(&subject),
                PredicateRef::NamedNode(&predicate),
                ObjectRef::Literal(&object),
                GraphNameRef::DefaultGraph,
            ))
            .unwrap();

        // Abort transaction
        transaction.abort();

        // Check it's not in the store
        let quad = Quad::new(
            crate::model::Subject::NamedNode(subject),
            crate::model::Predicate::NamedNode(predicate),
            crate::model::Object::Literal(object),
            crate::model::GraphName::DefaultGraph,
        );
        assert!(!store.contains_quad(&quad).unwrap());
    }
}
