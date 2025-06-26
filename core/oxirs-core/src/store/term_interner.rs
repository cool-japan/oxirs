//! High-performance term interning for RDF graphs
//!
//! This module provides efficient term interning specifically designed for RDF data,
//! with separate interners for subjects, predicates, and objects to maximize cache locality.

use bimap::BiMap;
use crate::model::{Subject, Predicate, Object, NamedNode, BlankNode, Literal, Term};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::path::Path;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};

/// Thread-safe term interner with bidirectional mapping between terms and IDs
#[derive(Debug)]
pub struct TermInterner {
    /// Interner for subject terms
    subjects: Arc<RwLock<BiMap<u32, SubjectTerm>>>,
    /// Interner for predicate terms (always IRIs)
    predicates: Arc<RwLock<BiMap<u32, String>>>,
    /// Interner for object terms
    objects: Arc<RwLock<BiMap<u32, ObjectTerm>>>,
    /// Next available IDs
    next_subject_id: Arc<RwLock<u32>>,
    next_predicate_id: Arc<RwLock<u32>>,
    next_object_id: Arc<RwLock<u32>>,
    /// Statistics
    stats: Arc<RwLock<InternerStats>>,
}

/// Interned subject representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubjectTerm {
    NamedNode(String),
    BlankNode(String),
}

/// Interned object representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectTerm {
    NamedNode(String),
    BlankNode(String),
    Literal {
        value: String,
        datatype: Option<String>,
        language: Option<String>,
    },
}

/// Statistics for monitoring interner performance
#[derive(Debug, Clone, Default)]
pub struct InternerStats {
    pub subject_count: usize,
    pub predicate_count: usize,
    pub object_count: usize,
    pub total_lookups: usize,
    pub cache_hits: usize,
    pub memory_bytes: usize,
}

impl InternerStats {
    /// Calculate cache hit ratio
    pub fn hit_ratio(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_lookups as f64
        }
    }

    /// Get total number of interned terms
    pub fn total_terms(&self) -> usize {
        self.subject_count + self.predicate_count + self.object_count
    }
}

impl TermInterner {
    /// Create a new term interner
    pub fn new() -> Self {
        TermInterner {
            subjects: Arc::new(RwLock::new(BiMap::new())),
            predicates: Arc::new(RwLock::new(BiMap::new())),
            objects: Arc::new(RwLock::new(BiMap::new())),
            next_subject_id: Arc::new(RwLock::new(0)),
            next_predicate_id: Arc::new(RwLock::new(0)),
            next_object_id: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(InternerStats::default())),
        }
    }

    /// Intern a subject and return its ID
    pub fn intern_subject(&self, subject: &Subject) -> u32 {
        let term = match subject {
            Subject::NamedNode(n) => SubjectTerm::NamedNode(n.as_str().to_string()),
            Subject::BlankNode(b) => SubjectTerm::BlankNode(b.id().to_string()),
            Subject::Variable(_) | Subject::QuotedTriple(_) => {
                panic!("Variables and quoted triples cannot be interned in storage")
            }
        };

        // Try to find existing ID
        {
            let subjects = self.subjects.read().unwrap();
            if let Some(&id) = subjects.get_by_right(&term) {
                let mut stats = self.stats.write().unwrap();
                stats.total_lookups += 1;
                stats.cache_hits += 1;
                return id;
            }
        }

        // Create new ID
        let mut subjects = self.subjects.write().unwrap();
        // Double-check in case another thread added it
        if let Some(&id) = subjects.get_by_right(&term) {
            let mut stats = self.stats.write().unwrap();
            stats.total_lookups += 1;
            stats.cache_hits += 1;
            return id;
        }

        let mut next_id = self.next_subject_id.write().unwrap();
        let id = *next_id;
        *next_id += 1;
        subjects.insert(id, term.clone());

        // Update stats
        let mut stats = self.stats.write().unwrap();
        stats.total_lookups += 1;
        stats.subject_count += 1;
        stats.memory_bytes += estimate_subject_memory(&term);

        id
    }

    /// Intern a predicate and return its ID
    pub fn intern_predicate(&self, predicate: &Predicate) -> u32 {
        let iri = match predicate {
            Predicate::NamedNode(n) => n.as_str().to_string(),
            Predicate::Variable(_) => {
                panic!("Variables cannot be interned as predicates in storage")
            }
        };

        // Try to find existing ID
        {
            let predicates = self.predicates.read().unwrap();
            if let Some(&id) = predicates.get_by_right(&iri) {
                let mut stats = self.stats.write().unwrap();
                stats.total_lookups += 1;
                stats.cache_hits += 1;
                return id;
            }
        }

        // Create new ID
        let mut predicates = self.predicates.write().unwrap();
        // Double-check
        if let Some(&id) = predicates.get_by_right(&iri) {
            let mut stats = self.stats.write().unwrap();
            stats.total_lookups += 1;
            stats.cache_hits += 1;
            return id;
        }

        let mut next_id = self.next_predicate_id.write().unwrap();
        let id = *next_id;
        *next_id += 1;
        predicates.insert(id, iri.clone());

        // Update stats
        let mut stats = self.stats.write().unwrap();
        stats.total_lookups += 1;
        stats.predicate_count += 1;
        stats.memory_bytes += iri.len() + 4; // String + ID

        id
    }

    /// Intern an object and return its ID
    pub fn intern_object(&self, object: &Object) -> u32 {
        let term = match object {
            Object::NamedNode(n) => ObjectTerm::NamedNode(n.as_str().to_string()),
            Object::BlankNode(b) => ObjectTerm::BlankNode(b.id().to_string()),
            Object::Literal(l) => ObjectTerm::Literal {
                value: l.value().to_string(),
                datatype: Some(l.datatype().as_str().to_string()),
                language: l.language().map(|lang| lang.to_string()),
            },
            Object::Variable(_) | Object::QuotedTriple(_) => {
                panic!("Variables and quoted triples cannot be interned in storage")
            }
        };

        // Try to find existing ID
        {
            let objects = self.objects.read().unwrap();
            if let Some(&id) = objects.get_by_right(&term) {
                let mut stats = self.stats.write().unwrap();
                stats.total_lookups += 1;
                stats.cache_hits += 1;
                return id;
            }
        }

        // Create new ID
        let mut objects = self.objects.write().unwrap();
        // Double-check
        if let Some(&id) = objects.get_by_right(&term) {
            let mut stats = self.stats.write().unwrap();
            stats.total_lookups += 1;
            stats.cache_hits += 1;
            return id;
        }

        let mut next_id = self.next_object_id.write().unwrap();
        let id = *next_id;
        *next_id += 1;
        objects.insert(id, term.clone());

        // Update stats
        let mut stats = self.stats.write().unwrap();
        stats.total_lookups += 1;
        stats.object_count += 1;
        stats.memory_bytes += estimate_object_memory(&term);

        id
    }

    /// Get subject by ID
    pub fn get_subject(&self, id: u32) -> Option<Subject> {
        let subjects = self.subjects.read().unwrap();
        subjects.get_by_left(&id).map(|term| match term {
            SubjectTerm::NamedNode(iri) => Subject::NamedNode(NamedNode::new(iri).unwrap()),
            SubjectTerm::BlankNode(id) => Subject::BlankNode(BlankNode::new(id).unwrap()),
        })
    }

    /// Get subject ID without interning (returns None if not found)
    pub fn get_subject_id(&self, subject: &Subject) -> Option<u32> {
        let term = match subject {
            Subject::NamedNode(n) => SubjectTerm::NamedNode(n.as_str().to_string()),
            Subject::BlankNode(b) => SubjectTerm::BlankNode(b.id().to_string()),
            Subject::Variable(_) | Subject::QuotedTriple(_) => {
                panic!("Variables and quoted triples cannot be interned in storage")
            }
        };
        let subjects = self.subjects.read().unwrap();
        subjects.get_by_right(&term).copied()
    }

    /// Get predicate by ID
    pub fn get_predicate(&self, id: u32) -> Option<Predicate> {
        let predicates = self.predicates.read().unwrap();
        predicates
            .get_by_left(&id)
            .map(|iri| Predicate::NamedNode(NamedNode::new(iri).unwrap()))
    }

    /// Get predicate ID without interning (returns None if not found)
    pub fn get_predicate_id(&self, predicate: &Predicate) -> Option<u32> {
        let iri = match predicate {
            Predicate::NamedNode(n) => n.as_str().to_string(),
            Predicate::Variable(_) => {
                panic!("Variables cannot be interned as predicates in storage")
            }
        };
        let predicates = self.predicates.read().unwrap();
        predicates.get_by_right(&iri).copied()
    }

    /// Get object by ID
    pub fn get_object(&self, id: u32) -> Option<Object> {
        let objects = self.objects.read().unwrap();
        objects.get_by_left(&id).map(|term| match term {
            ObjectTerm::NamedNode(iri) => Object::NamedNode(NamedNode::new(iri).unwrap()),
            ObjectTerm::BlankNode(id) => Object::BlankNode(BlankNode::new(id).unwrap()),
            ObjectTerm::Literal { value, datatype, language } => {
                let literal = if let Some(lang) = language {
                    Literal::new_language_tagged_literal(value, lang).unwrap()
                } else if let Some(dt) = datatype {
                    Literal::new_typed(value, NamedNode::new(dt).unwrap())
                } else {
                    Literal::new(value)
                };
                Object::Literal(literal)
            }
        })
    }

    /// Get object ID without interning (returns None if not found)
    pub fn get_object_id(&self, object: &Object) -> Option<u32> {
        let term = match object {
            Object::NamedNode(n) => ObjectTerm::NamedNode(n.as_str().to_string()),
            Object::BlankNode(b) => ObjectTerm::BlankNode(b.id().to_string()),
            Object::Literal(l) => ObjectTerm::Literal {
                value: l.value().to_string(),
                datatype: Some(l.datatype().as_str().to_string()),
                language: l.language().map(|lang| lang.to_string()),
            },
            Object::Variable(_) | Object::QuotedTriple(_) => {
                panic!("Variables and quoted triples cannot be interned in storage")
            }
        };
        let objects = self.objects.read().unwrap();
        objects.get_by_right(&term).copied()
    }

    /// Get current statistics
    pub fn stats(&self) -> InternerStats {
        self.stats.read().unwrap().clone()
    }

    /// Clear all interned terms (useful for memory management)
    pub fn clear(&self) {
        let mut subjects = self.subjects.write().unwrap();
        let mut predicates = self.predicates.write().unwrap();
        let mut objects = self.objects.write().unwrap();
        
        subjects.clear();
        predicates.clear();
        objects.clear();

        *self.next_subject_id.write().unwrap() = 0;
        *self.next_predicate_id.write().unwrap() = 0;
        *self.next_object_id.write().unwrap() = 0;

        let mut stats = self.stats.write().unwrap();
        *stats = InternerStats::default();
    }

    /// Get memory usage estimate
    pub fn memory_usage(&self) -> usize {
        self.stats.read().unwrap().memory_bytes
    }
    
    /// Intern a named node and return its ID (for compatibility with mmap_store)
    pub fn intern_named_node(&self, node: &NamedNode) -> u64 {
        self.intern_subject(&Subject::NamedNode(node.clone())) as u64
    }
    
    /// Intern a blank node and return its ID (for compatibility with mmap_store)
    pub fn intern_blank_node(&self, node: &BlankNode) -> u64 {
        self.intern_subject(&Subject::BlankNode(node.clone())) as u64
    }
    
    /// Intern a literal and return its ID (for compatibility with mmap_store)
    pub fn intern_literal(&self, literal: &Literal) -> u64 {
        self.intern_object(&Object::Literal(literal.clone())) as u64
    }
    
    /// Get named node ID without interning (for compatibility with mmap_store)
    pub fn get_named_node_id(&self, node: &NamedNode) -> Option<u64> {
        self.get_subject_id(&Subject::NamedNode(node.clone())).map(|id| id as u64)
    }
    
    /// Get blank node ID without interning (for compatibility with mmap_store)
    pub fn get_blank_node_id(&self, node: &BlankNode) -> Option<u64> {
        self.get_subject_id(&Subject::BlankNode(node.clone())).map(|id| id as u64)
    }
    
    /// Get literal ID without interning (for compatibility with mmap_store)
    pub fn get_literal_id(&self, literal: &Literal) -> Option<u64> {
        self.get_object_id(&Object::Literal(literal.clone())).map(|id| id as u64)
    }
    
    /// Save the interner to disk
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .context("Failed to create terms file")?;
        
        let mut writer = BufWriter::new(file);
        
        // Save subjects
        {
            let subjects = self.subjects.read().unwrap();
            let subject_data: Vec<(u32, SubjectTerm)> = subjects
                .iter()
                .map(|(id, term)| (*id, term.clone()))
                .collect();
            bincode::serialize_into(&mut writer, &subject_data)
                .context("Failed to serialize subjects")?;
        }
        
        // Save predicates
        {
            let predicates = self.predicates.read().unwrap();
            let predicate_data: Vec<(u32, String)> = predicates
                .iter()
                .map(|(id, iri)| (*id, iri.clone()))
                .collect();
            bincode::serialize_into(&mut writer, &predicate_data)
                .context("Failed to serialize predicates")?;
        }
        
        // Save objects
        {
            let objects = self.objects.read().unwrap();
            let object_data: Vec<(u32, ObjectTerm)> = objects
                .iter()
                .map(|(id, term)| (*id, term.clone()))
                .collect();
            bincode::serialize_into(&mut writer, &object_data)
                .context("Failed to serialize objects")?;
        }
        
        // Save next IDs
        let next_subject_id = *self.next_subject_id.read().unwrap();
        let next_predicate_id = *self.next_predicate_id.read().unwrap();
        let next_object_id = *self.next_object_id.read().unwrap();
        
        bincode::serialize_into(&mut writer, &next_subject_id)?;
        bincode::serialize_into(&mut writer, &next_predicate_id)?;
        bincode::serialize_into(&mut writer, &next_object_id)?;
        
        writer.flush()?;
        Ok(())
    }
    
    /// Load the interner from disk
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).context("Failed to open terms file")?;
        let mut reader = BufReader::new(file);
        
        // Load subjects
        let subject_data: Vec<(u32, SubjectTerm)> = bincode::deserialize_from(&mut reader)
            .context("Failed to deserialize subjects")?;
        let mut subjects = BiMap::new();
        for (id, term) in subject_data {
            subjects.insert(id, term);
        }
        
        // Load predicates
        let predicate_data: Vec<(u32, String)> = bincode::deserialize_from(&mut reader)
            .context("Failed to deserialize predicates")?;
        let mut predicates = BiMap::new();
        for (id, iri) in predicate_data {
            predicates.insert(id, iri);
        }
        
        // Load objects
        let object_data: Vec<(u32, ObjectTerm)> = bincode::deserialize_from(&mut reader)
            .context("Failed to deserialize objects")?;
        let mut objects = BiMap::new();
        for (id, term) in object_data {
            objects.insert(id, term);
        }
        
        // Load next IDs
        let next_subject_id: u32 = bincode::deserialize_from(&mut reader)?;
        let next_predicate_id: u32 = bincode::deserialize_from(&mut reader)?;
        let next_object_id: u32 = bincode::deserialize_from(&mut reader)?;
        
        // Calculate stats
        let mut stats = InternerStats::default();
        stats.subject_count = subjects.len();
        stats.predicate_count = predicates.len();
        stats.object_count = objects.len();
        
        Ok(TermInterner {
            subjects: Arc::new(RwLock::new(subjects)),
            predicates: Arc::new(RwLock::new(predicates)),
            objects: Arc::new(RwLock::new(objects)),
            next_subject_id: Arc::new(RwLock::new(next_subject_id)),
            next_predicate_id: Arc::new(RwLock::new(next_predicate_id)),
            next_object_id: Arc::new(RwLock::new(next_object_id)),
            stats: Arc::new(RwLock::new(stats)),
        })
    }
}

impl Default for TermInterner {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate memory usage for a subject term
fn estimate_subject_memory(term: &SubjectTerm) -> usize {
    match term {
        SubjectTerm::NamedNode(iri) => iri.len() + 4 + 8, // String + ID + enum overhead
        SubjectTerm::BlankNode(id) => id.len() + 4 + 8,
    }
}

/// Estimate memory usage for an object term
fn estimate_object_memory(term: &ObjectTerm) -> usize {
    match term {
        ObjectTerm::NamedNode(iri) => iri.len() + 4 + 8,
        ObjectTerm::BlankNode(id) => id.len() + 4 + 8,
        ObjectTerm::Literal { value, datatype, language } => {
            value.len() 
                + datatype.as_ref().map_or(0, |s| s.len())
                + language.as_ref().map_or(0, |s| s.len())
                + 4 + 24 // ID + Option overhead
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subject_interning() {
        let interner = TermInterner::new();
        
        let subject1 = Subject::NamedNode(NamedNode::new("http://example.org/s1").unwrap());
        let subject2 = Subject::BlankNode(BlankNode::new("b1").unwrap());
        let subject3 = Subject::NamedNode(NamedNode::new("http://example.org/s1").unwrap());

        let id1 = interner.intern_subject(&subject1);
        let id2 = interner.intern_subject(&subject2);
        let id3 = interner.intern_subject(&subject3);

        // Same subject should get same ID
        assert_eq!(id1, id3);
        assert_ne!(id1, id2);

        // Check retrieval
        assert_eq!(interner.get_subject(id1), Some(subject1.clone()));
        assert_eq!(interner.get_subject(id2), Some(subject2));

        // Check stats
        let stats = interner.stats();
        assert_eq!(stats.subject_count, 2);
        assert_eq!(stats.cache_hits, 1); // subject3 was a hit
    }

    #[test]
    fn test_predicate_interning() {
        let interner = TermInterner::new();
        
        let pred1 = Predicate::NamedNode(NamedNode::new("http://example.org/p1").unwrap());
        let pred2 = Predicate::NamedNode(NamedNode::new("http://example.org/p2").unwrap());
        let pred3 = Predicate::NamedNode(NamedNode::new("http://example.org/p1").unwrap());

        let id1 = interner.intern_predicate(&pred1);
        let id2 = interner.intern_predicate(&pred2);
        let id3 = interner.intern_predicate(&pred3);

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);

        assert_eq!(interner.get_predicate(id1), Some(pred1));
        assert_eq!(interner.get_predicate(id2), Some(pred2));
    }

    #[test]
    fn test_object_interning() {
        let interner = TermInterner::new();
        
        let obj1 = Object::Literal(Literal::new("test"));
        let obj2 = Object::NamedNode(NamedNode::new("http://example.org/o1").unwrap());
        let obj3 = Object::Literal(Literal::new("test"));

        let id1 = interner.intern_object(&obj1);
        let id2 = interner.intern_object(&obj2);
        let id3 = interner.intern_object(&obj3);

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);

        assert_eq!(interner.get_object(id1), Some(obj1));
        assert_eq!(interner.get_object(id2), Some(obj2));
    }

    #[test]
    fn test_clear() {
        let interner = TermInterner::new();
        
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/s1").unwrap());
        let id = interner.intern_subject(&subject);
        
        assert!(interner.get_subject(id).is_some());
        
        interner.clear();
        
        assert!(interner.get_subject(id).is_none());
        assert_eq!(interner.stats().total_terms(), 0);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let interner = Arc::new(TermInterner::new());
        let mut handles = vec![];

        for i in 0..10 {
            let interner_clone = Arc::clone(&interner);
            let handle = thread::spawn(move || {
                let subject = Subject::NamedNode(
                    NamedNode::new(&format!("http://example.org/s{}", i % 3)).unwrap()
                );
                interner_clone.intern_subject(&subject)
            });
            handles.push(handle);
        }

        let ids: Vec<u32> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        
        // Should have only 3 unique IDs (s0, s1, s2)
        let unique_ids: std::collections::HashSet<_> = ids.iter().collect();
        assert!(unique_ids.len() <= 3);
    }
}