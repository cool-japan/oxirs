//! Caching subsystem for SHACL validation
//!
//! Provides:
//! - [`validation_cache`]: TTL-based, thread-safe constraint satisfaction cache
//! - [`parallel_validator`]: Rayon-powered parallel constraint evaluation

pub mod parallel_validator;
pub mod validation_cache;

pub use parallel_validator::{
    ParallelConstraintConfig, ParallelConstraintOutcome, ParallelConstraintValidator,
    ParallelValidationStats, ParallelValidationSummary,
};
pub use validation_cache::{
    CacheStats, CachedValidationResult, TripleKey, ValidationCache, ValidationCacheKey,
};
