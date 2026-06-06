//! Neural-Symbolic Integration
//!
//! This module implements neural-symbolic integration for combining
//! neural learning with symbolic reasoning, logic-based constraints,
//! and knowledge-guided embeddings.
//!
//! This is a thin facade that re-exports the public API from the focused
//! companion modules:
//! - [`neural_symbolic_integration_types`](crate::neural_symbolic_integration_types):
//!   configuration structures, reasoning/logic enumerations, and the symbolic
//!   representation types (`LogicalFormula`, `FormulaStructure`, `KnowledgeRule`).
//! - [`neural_symbolic_integration_engine`](crate::neural_symbolic_integration_engine):
//!   the [`NeuralSymbolicModel`] and its `EmbeddingModel` implementation.
//! - [`neural_symbolic_integration_loss`](crate::neural_symbolic_integration_loss):
//!   standalone semantic/constraint/rule loss functions.

pub use crate::neural_symbolic_integration_engine::NeuralSymbolicModel;
pub use crate::neural_symbolic_integration_types::*;
