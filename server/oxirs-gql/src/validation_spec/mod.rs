//! GraphQL June 2018 specification validation rules.
//!
//! Implements all 25 validation rules from the GraphQL June 2018 specification,
//! covering executable document validation, fragment rules, variable rules,
//! directive rules, argument rules, type coercion, and field-merge conflict detection.
//!
//! The implementation is split across sibling modules:
//! - `operations`: operation/type/variable rules + shared helpers
//! - `fragments`: fragment rules
//! - `directives`: directive/argument/value/field-merge rules
//! - `tests`: integration tests (cfg(test) only)

mod directives;
mod fragments;
mod operations;

#[cfg(test)]
mod tests;

use crate::ast::Document;
use crate::types::Schema;
use crate::validation::ValidationError;

use directives::{
    check_known_argument_names, check_known_directives, check_overlapping_fields_can_be_merged,
    check_provided_required_arguments, check_unique_argument_names,
    check_unique_directives_per_location, check_values_of_correct_type,
};
use fragments::{
    check_known_fragment_names, check_no_fragment_cycles, check_no_unused_fragments,
    check_unique_fragment_names,
};
use operations::{
    check_executable_definitions, check_fragments_on_composite_types, check_known_type_names,
    check_leaf_field_selections, check_lone_anonymous_operation, check_no_undefined_variables,
    check_no_unused_variables, check_single_field_subscriptions, check_unique_operation_names,
    check_unique_variable_names, check_variables_are_input_types,
};

// ─────────────────────────────────────────────────────────────────────────────
// SpecValidator
// ─────────────────────────────────────────────────────────────────────────────

/// Validates a GraphQL document against the GraphQL June 2018 specification
/// validation rules. Returns a list of validation errors.
pub struct SpecValidator<'a> {
    schema: &'a Schema,
}

impl<'a> SpecValidator<'a> {
    pub fn new(schema: &'a Schema) -> Self {
        Self { schema }
    }

    /// Run all spec validation rules against the document.
    pub fn validate(&self, doc: &Document) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        errors.extend(check_executable_definitions(doc));
        errors.extend(check_unique_operation_names(doc));
        errors.extend(check_lone_anonymous_operation(doc));
        errors.extend(check_single_field_subscriptions(doc));
        errors.extend(check_known_type_names(self.schema, doc));
        errors.extend(check_fragments_on_composite_types(self.schema, doc));
        errors.extend(check_variables_are_input_types(self.schema, doc));
        errors.extend(check_leaf_field_selections(self.schema, doc));
        errors.extend(check_unique_fragment_names(doc));
        errors.extend(check_known_fragment_names(doc));
        errors.extend(check_no_fragment_cycles(doc));
        errors.extend(check_no_unused_fragments(doc));
        errors.extend(check_unique_variable_names(doc));
        errors.extend(check_no_undefined_variables(doc));
        errors.extend(check_no_unused_variables(doc));
        errors.extend(check_known_directives(self.schema, doc));
        errors.extend(check_unique_directives_per_location(doc));
        errors.extend(check_known_argument_names(self.schema, doc));
        errors.extend(check_unique_argument_names(doc));
        errors.extend(check_values_of_correct_type(self.schema, doc));
        errors.extend(check_provided_required_arguments(self.schema, doc));
        errors.extend(check_overlapping_fields_can_be_merged(doc));

        errors
    }
}
