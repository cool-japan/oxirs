//! In-memory registry of vocabularies served by the VocPrez handler.
//!
//! Each registered vocabulary is keyed by a short identifier (slug) and
//! describes a namespace, human-readable title, description and contributor
//! list. The registry is intentionally simple: persistence and store-backed
//! enumeration are added later by the integration layer.

use std::collections::HashMap;

/// A vocabulary description registered with the publishing handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabularyEntry {
    /// Short identifier used in URLs (e.g. `foaf`).
    pub id: String,
    /// Namespace IRI of the vocabulary.
    pub namespace: String,
    /// Human-readable title.
    pub title: String,
    /// Long-form description (may be empty).
    pub description: String,
    /// Optional list of contributor names.
    pub contributors: Vec<String>,
}

impl VocabularyEntry {
    /// Construct a minimal entry with empty description and no contributors.
    pub fn new(
        id: impl Into<String>,
        namespace: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            namespace: namespace.into(),
            title: title.into(),
            description: String::new(),
            contributors: Vec::new(),
        }
    }

    /// Builder-style setter for the long-form description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Builder-style accumulator that appends a single contributor.
    pub fn with_contributor(mut self, c: impl Into<String>) -> Self {
        self.contributors.push(c.into());
        self
    }
}

/// Registry of vocabularies keyed by [`VocabularyEntry::id`].
#[derive(Debug, Default, Clone)]
pub struct VocabularyRegistry {
    entries: HashMap<String, VocabularyEntry>,
}

impl VocabularyRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a registry entry, returning any previous value.
    pub fn register(&mut self, entry: VocabularyEntry) -> Option<VocabularyEntry> {
        self.entries.insert(entry.id.clone(), entry)
    }

    /// Look up an entry by identifier.
    pub fn get(&self, id: &str) -> Option<&VocabularyEntry> {
        self.entries.get(id)
    }

    /// Borrow every registered entry. Iteration order is unspecified.
    pub fn all(&self) -> Vec<&VocabularyEntry> {
        self.entries.values().collect()
    }

    /// Number of registered vocabularies.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no vocabularies are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove an entry by identifier, returning the removed value.
    pub fn remove(&mut self, id: &str) -> Option<VocabularyEntry> {
        self.entries.remove(id)
    }
}
