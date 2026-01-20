//! Error Types for Physics Module

use thiserror::Error;

/// Physics module errors
#[derive(Debug, Error)]
pub enum PhysicsError {
    #[error("Simulation error: {0}")]
    Simulation(String),

    #[error("Parameter extraction error: {0}")]
    ParameterExtraction(String),

    #[error("Result injection error: {0}")]
    ResultInjection(String),

    #[error("Physics constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Conservation law violation: {law}, expected: {expected}, actual: {actual}")]
    ConservationViolation {
        law: String,
        expected: f64,
        actual: f64,
    },

    #[error("Unit conversion error: {0}")]
    UnitConversion(String),

    #[error("RDF query error: {0}")]
    RdfQuery(String),

    #[error("SAMM parsing error: {0}")]
    SammParsing(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for physics operations
pub type PhysicsResult<T> = Result<T, PhysicsError>;
