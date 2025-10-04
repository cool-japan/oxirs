//! SWRL (Semantic Web Rule Language) - Core Types
//!
//! This module implements SWRL rule components.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SWRL atom types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SwrlAtom {
    /// Class atom: C(x)
    Class {
        class_predicate: String,
        argument: SwrlArgument,
    },
    /// Individual property atom: P(x, y)
    IndividualProperty {
        property_predicate: String,
        argument1: SwrlArgument,
        argument2: SwrlArgument,
    },
    /// Datavalue property atom: P(x, v)
    DatavalueProperty {
        property_predicate: String,
        argument1: SwrlArgument,
        argument2: SwrlArgument,
    },
    /// Built-in atom: builtin(args...)
    Builtin {
        builtin_predicate: String,
        arguments: Vec<SwrlArgument>,
    },
    /// Same individual atom: sameAs(x, y)
    SameIndividual {
        argument1: SwrlArgument,
        argument2: SwrlArgument,
    },
    /// Different individuals atom: differentFrom(x, y)
    DifferentIndividuals {
        argument1: SwrlArgument,
        argument2: SwrlArgument,
    },
}

/// SWRL argument types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SwrlArgument {
    /// Variable
    Variable(String),
    /// Individual
    Individual(String),
    /// Literal value
    Literal(String),
}

/// SWRL rule structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwrlRule {
    /// Rule identifier
    pub id: String,
    /// Rule body (antecedent)
    pub body: Vec<SwrlAtom>,
    /// Rule head (consequent)
    pub head: Vec<SwrlAtom>,
    /// Rule metadata
    pub metadata: HashMap<String, String>,
}

/// Built-in function definition
#[derive(Debug, Clone)]
pub struct BuiltinFunction {
    /// Function name
    pub name: String,
    /// Function namespace
    pub namespace: String,
    /// Minimum number of arguments
    pub min_args: usize,
    /// Maximum number of arguments (None for unlimited)
    pub max_args: Option<usize>,
    /// Function implementation
    pub implementation: fn(&[SwrlArgument]) -> Result<bool>,
}

/// SWRL execution context
#[derive(Debug, Clone, Default)]
pub struct SwrlContext {
    /// Variable bindings
    pub bindings: HashMap<String, SwrlArgument>,
    /// Execution trace
    pub trace: Vec<String>,
}
