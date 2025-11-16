//! Generic toolkit for building RDF parsers and serializers
//!
//! This module provides the core framework for implementing streaming RDF parsers
//! and serializers. It's designed around a token-rule architecture where:
//!
//! 1. **TokenRecognizer** - Converts byte streams into tokens
//! 2. **RuleRecognizer** - Converts token streams into RDF elements
//! 3. **Parser** - Orchestrates the parsing process
//! 4. **Serializer** - Provides serialization functionality

pub mod buffer_manager;
pub mod error;
pub mod fast_scanner;
pub mod format_detector;
pub mod lazy_iri;
pub mod lexer;
pub mod parser;
pub mod serializer;
pub mod simd_lexer;
pub mod string_interner;
pub mod zero_copy;

// Re-export the main traits and types
pub use buffer_manager::{BufferManager, GlobalBufferManager};
pub use error::*;
pub use fast_scanner::FastScanner;
pub use format_detector::{DetectionMethod, DetectionResult, FormatDetector, RdfFormat};
pub use lazy_iri::{CachedIriResolver, IriResolutionError, LazyIri, ResolverStats};
pub use lexer::*;
pub use parser::*;
pub use serializer::*;
pub use simd_lexer::{SimdLexer, SimdStats};
pub use string_interner::StringInterner;
pub use zero_copy::{ParseError as ZeroCopyParseError, ZeroCopyIriParser, ZeroCopyLiteralParser};
