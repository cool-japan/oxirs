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
pub mod error_reporter;
pub mod fast_scanner;
pub mod format_converter;
pub mod format_detector;
pub mod graph_utils;
pub mod iri_normalizer;
pub mod iri_validator;
pub mod lazy_iri;
pub mod lexer;
pub mod parser;
pub mod pattern_matcher;
pub mod rdf_validator;
pub mod serializer;
pub mod simd_lexer;
pub mod string_interner;
pub mod zero_copy;

// Re-export the main traits and types
pub use buffer_manager::{BufferManager, GlobalBufferManager};
pub use error::*;
pub use error_reporter::{
    create_error_report, format_simple_error, ErrorReporter, ErrorSuggestion,
};
pub use fast_scanner::FastScanner;
pub use format_converter::{
    ConversionConfig, ConversionError, ConversionResult, ConversionStats, FormatConverter,
};
pub use format_detector::{DetectionMethod, DetectionResult, FormatDetector, RdfFormat};
pub use graph_utils::{AdvancedGraphStats, DiffSummary, GraphDiff, GraphMerger, GraphTransformer};
pub use iri_normalizer::{
    iris_equivalent, normalize_iri, normalize_iri_cow, NormalizationError, NormalizationResult,
    NormalizedIri,
};
pub use iri_validator::{
    validate_iri, validate_iri_reference, IriValidationError, IriValidationResult,
};
pub use lazy_iri::{CachedIriResolver, IriResolutionError, LazyIri, ResolverStats};
pub use lexer::*;
pub use parser::*;
pub use pattern_matcher::{PatternMatcher, QueryBuilder, TriplePattern};
pub use rdf_validator::{
    check_duplicates, check_orphaned_blank_nodes, compute_stats, validate_quad, validate_triple,
    DatasetStats, Severity, ValidationIssue, ValidationResult,
};
pub use serializer::*;
pub use simd_lexer::{SimdLexer, SimdStats};
pub use string_interner::StringInterner;
pub use zero_copy::{ParseError as ZeroCopyParseError, ZeroCopyIriParser, ZeroCopyLiteralParser};
