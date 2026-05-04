//! # ML Model Certification Suite for OxiRS SHACL-AI
//!
//! Provides tools to measure how accurately the ML models in `oxirs-shacl-ai`
//! match the ground-truth results produced by the deterministic SHACL engine.
//!
//! ## Quick start
//!
//! ```rust
//! use oxirs_shacl_ai::certification::{
//!     CertificationCase, CertificationRunner, CertificationSuite,
//! };
//!
//! let cases: Vec<CertificationCase> = (0..20)
//!     .map(|i| CertificationCase {
//!         id: format!("case-{i}"),
//!         constraint_type: "sh:minCount".to_string(),
//!         ground_truth_violation: i % 3 == 0,
//!         model_predicted_violation: i % 3 == 0,
//!         confidence: Some(0.95),
//!     })
//!     .collect();
//!
//! let suite = CertificationSuite::from_cases("demo", cases);
//! let runner = CertificationRunner::new();
//! let report = runner.run(&suite);
//!
//! println!("{}", report.to_markdown());
//! assert!(report.passed());
//! ```

pub mod metrics;
pub mod report;
pub mod runner;

pub use metrics::{ClassificationMetrics, ConfusionMatrix, ConstraintTypeMetrics};
pub use report::{CertificationReport, CertificationStatus};
pub use runner::{CertificationCase, CertificationRunner, CertificationSuite};
