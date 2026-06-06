pub mod ast;
pub mod evaluator;
#[cfg(test)]
mod tests;

pub use ast::{Iri, NpsItem, PathDirection, PropertyPath};
pub use evaluator::PathAlgebraEvaluator;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PathAlgebraError {
    #[error("evaluation depth limit exceeded: {0}")]
    DepthLimitExceeded(usize),
    #[error("graph oracle error: {0}")]
    OracleError(String),
}
