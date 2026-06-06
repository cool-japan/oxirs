//! # Notation3 (N3) Logic Support
//!
//! N3 extends Turtle with logical rules using `@forAll`, `@forSome`, and `=>` (implies).
//! Used in reasoning systems like Notation3Py, EYE reasoner, etc.

pub mod n3logic_evaluator;
pub mod n3logic_parser;
pub mod n3logic_tests;
pub mod n3logic_types;

// Re-export all public types so callers can use `n3logic::N3Rule` etc.
pub use n3logic_evaluator::N3Engine;
pub use n3logic_parser::N3Parser;
pub use n3logic_types::{Bindings, N3BuiltIn, N3Formula, N3Rule, N3Term, Triple};
