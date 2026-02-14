//! Auto-generated module structure

pub mod functionregistry_traits;
pub mod queryexecutor_accessors;
pub mod queryexecutor_apply_order_by_group;
pub mod queryexecutor_apply_projection_group;
pub mod queryexecutor_apply_slice_group;
pub mod queryexecutor_estimate_cardinality_group;
pub mod queryexecutor_estimate_complexity_group;
pub mod queryexecutor_execute_algebra_for_update_group;
pub mod queryexecutor_execute_algebra_group;
pub mod queryexecutor_execute_group;
pub mod queryexecutor_execute_index_optimized_join_group;
pub mod queryexecutor_execute_single_pattern_group;
pub mod queryexecutor_extract_join_variables_group;
pub mod queryexecutor_lang_function_group;
pub mod queryexecutor_new_group;
pub mod queryexecutor_numeric_comparison_group;
pub mod queryexecutor_pattern_might_match_group;
pub mod queryexecutor_predicates;
pub mod queryexecutor_predicates_1;
pub mod queryexecutor_predicates_2;
pub mod queryexecutor_queries;
pub mod queryexecutor_traits;
pub mod queryexecutor_type;
pub mod queryexecutor_union_solutions_group;
pub mod types;

// Executor function modules
pub mod adaptive_executor;
pub mod config;
pub mod dataset;
pub mod parallel;
pub mod parallel_optimized;
pub mod spill_manager;
pub mod stats;
pub mod streaming;

// Re-export all types
pub use queryexecutor_new_group::*;
pub use queryexecutor_type::*;
pub use types::*;

// Re-export executor function modules
pub use adaptive_executor::*;
pub use config::*;
pub use dataset::*;
pub use parallel::*;
pub use parallel_optimized::*;
pub use spill_manager::*;
pub use stats::*;
pub use streaming::*;
