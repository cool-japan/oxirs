//! Per-vocabulary metadata used by the publishing serializers.

use super::registry::VocabularyEntry;

/// Aggregated vocabulary metadata suitable for HTML/JSON-LD/Turtle emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabularyMetadata {
    pub id: String,
    pub namespace: String,
    pub title: String,
    pub description: String,
    pub contributors: Vec<String>,
    /// Number of concepts (classes, properties, individuals, …) defined
    /// within the vocabulary.  Computed by the integration layer in real
    /// deployments.
    pub concept_count: usize,
}

/// Combine a registered [`VocabularyEntry`] with a computed concept count to
/// produce a [`VocabularyMetadata`] record ready for serialization.
pub fn build_metadata(entry: &VocabularyEntry, concept_count: usize) -> VocabularyMetadata {
    VocabularyMetadata {
        id: entry.id.clone(),
        namespace: entry.namespace.clone(),
        title: entry.title.clone(),
        description: entry.description.clone(),
        contributors: entry.contributors.clone(),
        concept_count,
    }
}
