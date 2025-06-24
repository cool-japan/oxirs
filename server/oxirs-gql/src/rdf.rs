//! RDF integration functionality
//!
//! This module provides RDF-specific scalars, SPARQL mapping, and RDF store integration.

// Re-export RDF functionality
pub use crate::rdf_scalars::*;
pub use crate::mapping::*;

/// RDF components
pub mod components {
    //! Individual RDF components
    
    /// RDF scalar types
    pub mod scalars {
        pub use crate::rdf_scalars::*;
    }
    
    /// GraphQL to SPARQL mapping
    pub mod mapping {
        pub use crate::mapping::*;
    }
}

/// Common RDF utilities and helpers
pub mod utils {
    use anyhow::Result;
    
    /// Validate IRI format
    pub fn is_valid_iri(iri: &str) -> bool {
        crate::rdf_scalars::IRI::new(iri.to_string()).is_ok()
    }
    
    /// Create a prefixed name from namespace and local name
    pub fn create_prefixed_name(namespace: &str, local: &str) -> String {
        format!("{}{}", namespace, local)
    }
    
    /// Extract namespace from IRI
    pub fn extract_namespace(iri: &str) -> Option<String> {
        if let Some(pos) = iri.rfind('/') {
            Some(iri[..pos + 1].to_string())
        } else if let Some(pos) = iri.rfind('#') {
            Some(iri[..pos + 1].to_string())
        } else {
            None
        }
    }
    
    /// Extract local name from IRI
    pub fn extract_local_name(iri: &str) -> Option<String> {
        if let Some(pos) = iri.rfind('/') {
            Some(iri[pos + 1..].to_string())
        } else if let Some(pos) = iri.rfind('#') {
            Some(iri[pos + 1..].to_string())
        } else {
            None
        }
    }
}