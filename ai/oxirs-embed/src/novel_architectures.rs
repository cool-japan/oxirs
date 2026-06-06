//! Novel architectures module facade.
//!
//! This file re-exports the public surface of the split sibling modules so
//! that existing call sites continue to resolve symbols such as
//! [`NovelArchitectureModel`], every architecture parameter and state struct,
//! and the related `EmbeddingModel` trait implementation through
//! `crate::novel_architectures::*`.
//!
//! The implementation lives in:
//! - [`crate::novel_arch_types`]: configuration enums, parameter and state
//!   types.
//! - [`crate::novel_arch_impl`]: inherent impl blocks and the
//!   `EmbeddingModel` impl.
//! - `crate::novel_arch_tests` (test-only): exhaustive unit and async tests.

pub use crate::novel_arch_types::*;
