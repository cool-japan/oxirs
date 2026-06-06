//! Reification utilities for converting between RDF-star and standard RDF.

pub mod vocab {
    pub const RDF_STATEMENT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement";
    pub const RDF_SUBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject";
    pub const RDF_PREDICATE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate";
    pub const RDF_OBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object";
    pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
}

pub mod converter;
pub mod mapper;
#[cfg(test)]
mod tests;
pub mod types;

pub use converter::{AdvancedReificator, Reificator};
pub use mapper::{count_reifications, has_reifications, validate_reifications, ReificationBridge};
pub use types::{
    AdvancedReificationStrategy, AnnotationStyle, EmbeddedTriple, ReificationCondition,
    ReificationContext, ReificationRule, ReificationStatistics, ReificationStrategy, TermType,
};

pub mod utils {
    pub use super::mapper::{count_reifications, has_reifications, validate_reifications};
}
