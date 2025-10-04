//! Module for biomedical embeddings

use crate::{ModelConfig, ModelStats, TrainingStats, Triple};
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiomedicalEntityType {
    Gene,
    Protein,
    Disease,
    Drug,
    Compound,
    Pathway,
    Cell,
    Tissue,
    Organ,
    Phenotype,
    GoTerm,
    MeshTerm,
    SnomedCt,
    IcdCode,
}

impl BiomedicalEntityType {
    /// Get the namespace prefix for this entity type
    pub fn namespace(&self) -> &'static str {
        match self {
            BiomedicalEntityType::Gene => "gene",
            BiomedicalEntityType::Protein => "protein",
            BiomedicalEntityType::Disease => "disease",
            BiomedicalEntityType::Drug => "drug",
            BiomedicalEntityType::Compound => "compound",
            BiomedicalEntityType::Pathway => "pathway",
            BiomedicalEntityType::Cell => "cell",
            BiomedicalEntityType::Tissue => "tissue",
            BiomedicalEntityType::Organ => "organ",
            BiomedicalEntityType::Phenotype => "phenotype",
            BiomedicalEntityType::GoTerm => "go",
            BiomedicalEntityType::MeshTerm => "mesh",
            BiomedicalEntityType::SnomedCt => "snomed",
            BiomedicalEntityType::IcdCode => "icd",
        }
    }

    /// Parse entity type from IRI
    pub fn from_iri(iri: &str) -> Option<Self> {
        if iri.contains("gene") || iri.contains("HGNC") {
            Some(BiomedicalEntityType::Gene)
        } else if iri.contains("protein") || iri.contains("UniProt") {
            Some(BiomedicalEntityType::Protein)
        } else if iri.contains("disease") || iri.contains("OMIM") || iri.contains("DOID") {
            Some(BiomedicalEntityType::Disease)
        } else if iri.contains("drug") || iri.contains("DrugBank") {
            Some(BiomedicalEntityType::Drug)
        } else if iri.contains("compound") || iri.contains("CHEBI") {
            Some(BiomedicalEntityType::Compound)
        } else if iri.contains("pathway") || iri.contains("KEGG") || iri.contains("Reactome") {
            Some(BiomedicalEntityType::Pathway)
        } else if iri.contains("GO:") {
            Some(BiomedicalEntityType::GoTerm)
        } else if iri.contains("MESH") {
            Some(BiomedicalEntityType::MeshTerm)
        } else if iri.contains("SNOMED") {
            Some(BiomedicalEntityType::SnomedCt)
        } else if iri.contains("ICD") {
            Some(BiomedicalEntityType::IcdCode)
        } else {
            None
        }
    }
}

/// Biomedical relation types for specialized handling
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiomedicalRelationType {
    /// Gene-disease associations
    CausesDisease,
    AssociatedWithDisease,
    PredisposesToDisease,
    /// Drug-target interactions
    TargetsProtein,
    InhibitsProtein,
    ActivatesProtein,
    BindsToProtein,
    /// Pathway relationships
    ParticipatesInPathway,
    RegulatesPathway,
    UpstreamOfPathway,
    DownstreamOfPathway,
    /// Protein interactions
    InteractsWith,
    PhysicallyInteractsWith,
    FunctionallyInteractsWith,
    /// Chemical relationships
    MetabolizedBy,
    TransportedBy,
    Catalyzes,
    /// Hierarchical relationships
    IsASubtypeOf,
    PartOf,
    HasPhenotype,
    /// Expression relationships
    ExpressedIn,
    Overexpressed,
    Underexpressed,
}

impl BiomedicalRelationType {
    /// Parse relation type from predicate IRI
    pub fn from_iri(iri: &str) -> Option<Self> {
        match iri.to_lowercase().as_str() {
            s if s.contains("causes") => Some(BiomedicalRelationType::CausesDisease),
            s if s.contains("associated_with") => {
                Some(BiomedicalRelationType::AssociatedWithDisease)
            }
            s if s.contains("targets") => Some(BiomedicalRelationType::TargetsProtein),
            s if s.contains("inhibits") => Some(BiomedicalRelationType::InhibitsProtein),
            s if s.contains("activates") => Some(BiomedicalRelationType::ActivatesProtein),
            s if s.contains("binds") => Some(BiomedicalRelationType::BindsToProtein),
            s if s.contains("participates") => Some(BiomedicalRelationType::ParticipatesInPathway),
            s if s.contains("interacts") => Some(BiomedicalRelationType::InteractsWith),
            s if s.contains("metabolized") => Some(BiomedicalRelationType::MetabolizedBy),
            s if s.contains("expressed") => Some(BiomedicalRelationType::ExpressedIn),
            s if s.contains("subtype") => Some(BiomedicalRelationType::IsASubtypeOf),
            s if s.contains("part_of") => Some(BiomedicalRelationType::PartOf),
            _ => None,
        }
    }
}

/// Configuration for biomedical embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomedicalEmbeddingConfig {
    pub base_config: ModelConfig,
    /// Weight for gene-disease associations
    pub gene_disease_weight: f32,
    /// Weight for drug-target interactions
    pub drug_target_weight: f32,
    /// Weight for pathway relationships
    pub pathway_weight: f32,
    /// Weight for protein interactions
    pub protein_interaction_weight: f32,
    /// Enable sequence similarity features
    pub use_sequence_similarity: bool,
    /// Enable chemical structure features
    pub use_chemical_structure: bool,
    /// Enable taxonomic hierarchy
    pub use_taxonomy: bool,
    /// Enable temporal relationships
    pub use_temporal_features: bool,
    /// Species filter (e.g., "Homo sapiens", "Mus musculus")
    pub species_filter: Option<String>,
}

impl Default for BiomedicalEmbeddingConfig {
    fn default() -> Self {
        Self {
            base_config: ModelConfig::default(),
            gene_disease_weight: 2.0,
            drug_target_weight: 1.5,
            pathway_weight: 1.2,
            protein_interaction_weight: 1.0,
            use_sequence_similarity: true,
            use_chemical_structure: true,
            use_taxonomy: true,
            use_temporal_features: false,
            species_filter: Some("Homo sapiens".to_string()),
        }
    }
}

/// Biomedical knowledge graph embedding model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomedicalEmbedding {
    pub config: BiomedicalEmbeddingConfig,
    pub model_id: Uuid,
    /// Entity embeddings by type
    pub gene_embeddings: HashMap<String, Array1<f32>>,
    pub protein_embeddings: HashMap<String, Array1<f32>>,
    pub disease_embeddings: HashMap<String, Array1<f32>>,
    pub drug_embeddings: HashMap<String, Array1<f32>>,
    pub compound_embeddings: HashMap<String, Array1<f32>>,
    pub pathway_embeddings: HashMap<String, Array1<f32>>,
    /// Relation embeddings by type
    pub relation_embeddings: HashMap<String, Array1<f32>>,
    /// Entity type mappings
    pub entity_types: HashMap<String, BiomedicalEntityType>,
    /// Relation type mappings
    pub relation_types: HashMap<String, BiomedicalRelationType>,
    /// Training data
    pub triples: Vec<Triple>,
    /// Biomedical-specific features
    pub features: BiomedicalFeatures,
    /// Training and model stats
    pub training_stats: TrainingStats,
    pub model_stats: ModelStats,
    pub is_trained: bool,
}

/// Biomedical-specific features for enhanced embeddings
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BiomedicalFeatures {
    /// Gene-disease association scores
    pub gene_disease_associations: HashMap<(String, String), f32>,
    /// Drug-target binding affinities
    pub drug_target_affinities: HashMap<(String, String), f32>,
    /// Pathway membership scores
    pub pathway_memberships: HashMap<(String, String), f32>,
    /// Protein-protein interaction scores
    pub protein_interactions: HashMap<(String, String), f32>,
    /// Sequence similarity scores
    pub sequence_similarities: HashMap<(String, String), f32>,
    /// Chemical structure similarities
    pub structure_similarities: HashMap<(String, String), f32>,
    /// Expression correlations
    pub expression_correlations: HashMap<(String, String), f32>,
    /// Tissue-specific expression
    pub tissue_expression: HashMap<(String, String), f32>,
}
