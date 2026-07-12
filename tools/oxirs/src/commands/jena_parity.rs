//! # Jena Parity Verifier — thin facade module.
//!
//! Verification tool that checks OxiRS feature parity with Apache Jena.
//! Tracks 100+ features across all major Jena ecosystem components and
//! identifies where OxiRS matches, partially implements, extends, or
//! has gaps relative to the Jena/Fuseki baseline.
//!
//! The implementation is split across sibling modules (declared in
//! `commands/mod.rs`):
//!
//! - [`jena_parity_types`](crate::commands::jena_parity_types): `ParityCategory`,
//!   `FeatureStatus`, `ParityFeature`, `ParitySummary`
//! - [`jena_parity_checker`](crate::commands::jena_parity_checker): `JenaParityChecker`
//!
//! Unit tests live in `jena_parity_tests` (also declared in `commands/mod.rs`).
//! All public items are re-exported below so existing imports continue to work.

pub use super::jena_parity_checker::*;
pub use super::jena_parity_types::*;
