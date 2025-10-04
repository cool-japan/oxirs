//! SPARQL query processing modules

pub mod aggregates;
pub mod executor;
pub mod expressions;
pub mod filter;
pub mod modifiers;
pub mod parser;
pub mod patterns;

// Re-export commonly used types
pub use aggregates::{
    apply_aggregates, extract_aggregates, find_matching_paren, AggregateExpression,
    AggregateFunction,
};
pub use executor::QueryExecutor;
pub use expressions::{
    apply_bind_expressions, evaluate_expression, extract_bind_expressions, parse_expression,
    split_function_args, term_to_number, term_to_string, ArithmeticOperator, BindExpression,
    Expression,
};
pub use filter::{
    evaluate_filters, extract_filter_expressions, ComparisonOp, FilterExpression, FilterValue,
};
pub use modifiers::{
    compare_terms, extract_limit, extract_offset, extract_order_by, remove_duplicate_bindings,
    sort_results, OrderBy,
};
pub use parser::{extract_and_expand_prefixes, extract_select_variables};
pub use patterns::{
    apply_optional_patterns, extract_pattern_groups, extract_union_groups, find_matching_brace,
    has_union, parse_simple_pattern, parse_union_branch, PatternGroup, SimpleTriplePattern,
    UnionGroup,
};
