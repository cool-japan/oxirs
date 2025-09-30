//! Generic toolkit for building RDF parsers and serializers
//!
//! This module provides the core framework for implementing streaming RDF parsers
//! and serializers. It's designed around a token-rule architecture where:
//!
//! 1. **TokenRecognizer** - Converts byte streams into tokens
//! 2. **RuleRecognizer** - Converts token streams into RDF elements
//! 3. **Parser** - Orchestrates the parsing process
//! 4. **Serializer** - Provides serialization functionality

pub mod error;
pub mod lexer;
pub mod parser;
pub mod serializer;

// Re-export the main traits and types
pub use error::*;
pub use lexer::*;
pub use parser::*;
pub use serializer::*;
