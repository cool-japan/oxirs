//! SHACL property path evaluation.
//!
//! Property paths are SPARQL 1.1 path expressions reused by SHACL Core §2.3.1
//! to declare what values a property shape applies to. This module implements
//! every operator from the SHACL spec:
//!
//! | Operator                   | Spec section | Construct (in [`PropertyPath`])     |
//! |----------------------------|--------------|--------------------------------------|
//! | Predicate path             | §2.3.1.1     | [`PropertyPath::Predicate`]          |
//! | Sequence path              | §2.3.1.2     | [`PropertyPath::Sequence`]           |
//! | Alternative path           | §2.3.1.3     | [`PropertyPath::Alternative`]        |
//! | Inverse path               | §2.3.1.4     | [`PropertyPath::Inverse`]            |
//! | Zero-or-more path          | §2.3.1.5     | [`PropertyPath::ZeroOrMore`]         |
//! | One-or-more path           | §2.3.1.6     | [`PropertyPath::OneOrMore`]          |
//! | Zero-or-one path           | §2.3.1.7     | [`PropertyPath::ZeroOrOne`]          |
//!
//! Path evaluation is exposed through `PropertyPathEvaluator`, with caching,
//! cycle detection, and an optional SPARQL fallback for paths that cannot be
//! materialised in-memory.

pub mod functions;
pub mod pathcacheconfig_traits;
pub mod pathoptimizationhints_traits;
pub mod prelude;
pub mod propertypath_traits;
pub mod propertypathevaluator_accessors;
pub mod propertypathevaluator_accessors_1;
pub mod propertypathevaluator_accessors_2;
pub mod propertypathevaluator_accessors_3;
pub mod propertypathevaluator_clear_cache_group;
pub mod propertypathevaluator_evaluate_complex_inverse_group;
pub mod propertypathevaluator_evaluate_inverse_predicate_group;
pub mod propertypathevaluator_evaluate_path_impl_group;
pub mod propertypathevaluator_evaluate_path_with_sparql_group;
pub mod propertypathevaluator_evaluate_predicate_direct_group;
pub mod propertypathevaluator_evaluate_predicate_group;
pub mod propertypathevaluator_generate_common_prefixes_group;
pub mod propertypathevaluator_generate_fallback_sparql_query_group;
pub mod propertypathevaluator_generate_hybrid_sparql_query_group;
pub mod propertypathevaluator_has_potential_cycles_group;
pub mod propertypathevaluator_manage_cache_size_group;
pub mod propertypathevaluator_new_group;
pub mod propertypathevaluator_optimize_path_group;
pub mod propertypathevaluator_should_cache_result_group;
pub mod propertypathevaluator_traits;
pub mod propertypathevaluator_type;
pub mod types;

// Re-export all types
pub use functions::*;
pub use propertypathevaluator_type::*;
pub use types::*;
