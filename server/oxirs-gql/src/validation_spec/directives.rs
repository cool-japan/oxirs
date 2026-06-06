//! Directive, argument, value, and field-merge validation rules.
//!
//! Covers: KnownDirectives, UniqueDirectivesPerLocation, KnownArgumentNames,
//! UniqueArgumentNames, ValuesOfCorrectType, ProvidedRequiredArguments,
//! OverlappingFieldsCanBeMerged.

use std::collections::{HashMap, HashSet};

use crate::ast::{Definition, Document, FragmentDefinition, Selection, SelectionSet, Value};
use crate::types::{GraphQLType, Schema};
use crate::validation::{SpecRule, ValidationError, ValidationRule};

use super::operations::{collect_fragments, get_field_def, unwrap_type};

// ─────────────────────────────────────────────────────────────────────────────
// KnownDirectives
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_known_directives(schema: &Schema, doc: &Document) -> Vec<ValidationError> {
    let known = ["skip", "include", "deprecated", "specifiedBy"];
    let mut errors = Vec::new();

    for def in &doc.definitions {
        match def {
            Definition::Operation(op) => {
                for dir in &op.directives {
                    if !known.contains(&dir.name.as_str())
                        && !schema.directives.contains_key(&dir.name)
                    {
                        errors.push(ValidationError::new(
                            format!("Unknown directive \"@{}\".", dir.name),
                            ValidationRule::Spec(SpecRule::KnownDirectives),
                        ));
                    }
                }
                collect_directive_errors_in_ss(schema, &op.selection_set, &known, &mut errors);
            }
            Definition::Fragment(frag) => {
                for dir in &frag.directives {
                    if !known.contains(&dir.name.as_str())
                        && !schema.directives.contains_key(&dir.name)
                    {
                        errors.push(ValidationError::new(
                            format!("Unknown directive \"@{}\".", dir.name),
                            ValidationRule::Spec(SpecRule::KnownDirectives),
                        ));
                    }
                }
                collect_directive_errors_in_ss(schema, &frag.selection_set, &known, &mut errors);
            }
            _ => {}
        }
    }

    errors
}

fn collect_directive_errors_in_ss(
    schema: &Schema,
    ss: &SelectionSet,
    known: &[&str],
    errors: &mut Vec<ValidationError>,
) {
    for sel in &ss.selections {
        match sel {
            Selection::Field(f) => {
                for dir in &f.directives {
                    if !known.contains(&dir.name.as_str())
                        && !schema.directives.contains_key(&dir.name)
                    {
                        errors.push(ValidationError::new(
                            format!("Unknown directive \"@{}\".", dir.name),
                            ValidationRule::Spec(SpecRule::KnownDirectives),
                        ));
                    }
                }
                if let Some(ss2) = &f.selection_set {
                    collect_directive_errors_in_ss(schema, ss2, known, errors);
                }
            }
            Selection::InlineFragment(inl) => {
                for dir in &inl.directives {
                    if !known.contains(&dir.name.as_str())
                        && !schema.directives.contains_key(&dir.name)
                    {
                        errors.push(ValidationError::new(
                            format!("Unknown directive \"@{}\".", dir.name),
                            ValidationRule::Spec(SpecRule::KnownDirectives),
                        ));
                    }
                }
                collect_directive_errors_in_ss(schema, &inl.selection_set, known, errors);
            }
            Selection::FragmentSpread(spread) => {
                for dir in &spread.directives {
                    if !known.contains(&dir.name.as_str())
                        && !schema.directives.contains_key(&dir.name)
                    {
                        errors.push(ValidationError::new(
                            format!("Unknown directive \"@{}\".", dir.name),
                            ValidationRule::Spec(SpecRule::KnownDirectives),
                        ));
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// UniqueDirectivesPerLocation
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_unique_directives_per_location(doc: &Document) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for def in &doc.definitions {
        match def {
            Definition::Operation(op) => {
                check_directive_uniqueness(&op.directives, &mut errors);
                collect_unique_directive_errors_in_ss(&op.selection_set, &mut errors);
            }
            Definition::Fragment(frag) => {
                check_directive_uniqueness(&frag.directives, &mut errors);
                collect_unique_directive_errors_in_ss(&frag.selection_set, &mut errors);
            }
            _ => {}
        }
    }

    errors
}

fn check_directive_uniqueness(
    directives: &[crate::ast::Directive],
    errors: &mut Vec<ValidationError>,
) {
    let mut seen: HashMap<String, usize> = HashMap::new();
    for dir in directives {
        let count = seen.entry(dir.name.clone()).or_insert(0);
        *count += 1;
        if *count == 2 {
            errors.push(ValidationError::new(
                format!(
                    "The directive \"@{}\" can only be used once at this location.",
                    dir.name
                ),
                ValidationRule::Spec(SpecRule::UniqueDirectivesPerLocation),
            ));
        }
    }
}

fn collect_unique_directive_errors_in_ss(ss: &SelectionSet, errors: &mut Vec<ValidationError>) {
    for sel in &ss.selections {
        match sel {
            Selection::Field(f) => {
                check_directive_uniqueness(&f.directives, errors);
                if let Some(ss2) = &f.selection_set {
                    collect_unique_directive_errors_in_ss(ss2, errors);
                }
            }
            Selection::InlineFragment(inl) => {
                check_directive_uniqueness(&inl.directives, errors);
                collect_unique_directive_errors_in_ss(&inl.selection_set, errors);
            }
            Selection::FragmentSpread(spread) => {
                check_directive_uniqueness(&spread.directives, errors);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KnownArgumentNames
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_known_argument_names(schema: &Schema, doc: &Document) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let fragments = collect_fragments(doc);

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            let root_type_name = match op.operation_type {
                crate::ast::OperationType::Query => schema.query_type.as_deref(),
                crate::ast::OperationType::Mutation => schema.mutation_type.as_deref(),
                crate::ast::OperationType::Subscription => schema.subscription_type.as_deref(),
            };
            if let Some(root) = root_type_name {
                collect_known_arg_errors(
                    schema,
                    &op.selection_set,
                    root,
                    &fragments,
                    &mut errors,
                    &mut HashSet::new(),
                );
            }
        }
    }

    errors
}

fn collect_known_arg_errors(
    schema: &Schema,
    ss: &SelectionSet,
    parent_type_name: &str,
    fragments: &HashMap<String, &FragmentDefinition>,
    errors: &mut Vec<ValidationError>,
    visited: &mut HashSet<String>,
) {
    let parent_type = schema.get_type(parent_type_name);

    for sel in &ss.selections {
        match sel {
            Selection::Field(f) => {
                if f.name.starts_with("__") {
                    if let Some(ss2) = &f.selection_set {
                        collect_known_arg_errors(schema, ss2, "", fragments, errors, visited);
                    }
                    continue;
                }

                let field_def = parent_type.and_then(|pt| get_field_def(pt, &f.name));

                if let Some(ft) = field_def {
                    for arg in &f.arguments {
                        if !ft.arguments.contains_key(&arg.name) {
                            errors.push(ValidationError::new(
                                format!(
                                    "Unknown argument \"{}\" on field \"{}.{}\".",
                                    arg.name, parent_type_name, f.name
                                ),
                                ValidationRule::Spec(SpecRule::KnownArgumentNames),
                            ));
                        }
                    }
                    if let Some(ss2) = &f.selection_set {
                        let return_type_name = unwrap_type(&ft.field_type).name().to_string();
                        collect_known_arg_errors(
                            schema,
                            ss2,
                            &return_type_name,
                            fragments,
                            errors,
                            visited,
                        );
                    }
                } else if let Some(ss2) = &f.selection_set {
                    collect_known_arg_errors(schema, ss2, "", fragments, errors, visited);
                }
            }
            Selection::InlineFragment(inl) => {
                let type_name = inl.type_condition.as_deref().unwrap_or(parent_type_name);
                collect_known_arg_errors(
                    schema,
                    &inl.selection_set,
                    type_name,
                    fragments,
                    errors,
                    visited,
                );
            }
            Selection::FragmentSpread(spread) => {
                if !visited.contains(&spread.fragment_name) {
                    visited.insert(spread.fragment_name.clone());
                    if let Some(frag) = fragments.get(spread.fragment_name.as_str()) {
                        collect_known_arg_errors(
                            schema,
                            &frag.selection_set,
                            &frag.type_condition,
                            fragments,
                            errors,
                            visited,
                        );
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// UniqueArgumentNames
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_unique_argument_names(doc: &Document) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            collect_unique_arg_errors_in_ss(&op.selection_set, &mut errors);
        }
    }

    errors
}

fn collect_unique_arg_errors_in_ss(ss: &SelectionSet, errors: &mut Vec<ValidationError>) {
    for sel in &ss.selections {
        match sel {
            Selection::Field(f) => {
                let mut seen: HashMap<String, usize> = HashMap::new();
                for arg in &f.arguments {
                    let count = seen.entry(arg.name.clone()).or_insert(0);
                    *count += 1;
                    if *count == 2 {
                        errors.push(ValidationError::new(
                            format!("There can be only one argument named \"{}\".", arg.name),
                            ValidationRule::Spec(SpecRule::UniqueArgumentNames),
                        ));
                    }
                }
                if let Some(ss2) = &f.selection_set {
                    collect_unique_arg_errors_in_ss(ss2, errors);
                }
            }
            Selection::InlineFragment(inl) => {
                collect_unique_arg_errors_in_ss(&inl.selection_set, errors);
            }
            Selection::FragmentSpread(_) => {}
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ValuesOfCorrectType
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_values_of_correct_type(
    schema: &Schema,
    doc: &Document,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let fragments = collect_fragments(doc);

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            let root_type_name = match op.operation_type {
                crate::ast::OperationType::Query => schema.query_type.as_deref(),
                crate::ast::OperationType::Mutation => schema.mutation_type.as_deref(),
                crate::ast::OperationType::Subscription => schema.subscription_type.as_deref(),
            };
            if let Some(root) = root_type_name {
                collect_type_correctness_errors(
                    schema,
                    &op.selection_set,
                    root,
                    &fragments,
                    &mut errors,
                    &mut HashSet::new(),
                );
            }
        }
    }

    errors
}

fn collect_type_correctness_errors(
    schema: &Schema,
    ss: &SelectionSet,
    parent_type_name: &str,
    fragments: &HashMap<String, &FragmentDefinition>,
    errors: &mut Vec<ValidationError>,
    visited: &mut HashSet<String>,
) {
    let parent_type = schema.get_type(parent_type_name);

    for sel in &ss.selections {
        match sel {
            Selection::Field(f) => {
                if f.name.starts_with("__") {
                    if let Some(ss2) = &f.selection_set {
                        collect_type_correctness_errors(
                            schema, ss2, "", fragments, errors, visited,
                        );
                    }
                    continue;
                }

                let field_def = parent_type.and_then(|pt| get_field_def(pt, &f.name));

                if let Some(ft) = field_def {
                    for arg in &f.arguments {
                        if let Some(arg_def) = ft.arguments.get(&arg.name) {
                            if let Some(err) = check_value_type_compat(
                                &arg.value,
                                &arg_def.argument_type,
                                &arg.name,
                            ) {
                                errors.push(err);
                            }
                        }
                    }
                    if let Some(ss2) = &f.selection_set {
                        let return_type_name = unwrap_type(&ft.field_type).name().to_string();
                        collect_type_correctness_errors(
                            schema,
                            ss2,
                            &return_type_name,
                            fragments,
                            errors,
                            visited,
                        );
                    }
                } else if let Some(ss2) = &f.selection_set {
                    collect_type_correctness_errors(schema, ss2, "", fragments, errors, visited);
                }
            }
            Selection::InlineFragment(inl) => {
                let type_name = inl.type_condition.as_deref().unwrap_or(parent_type_name);
                collect_type_correctness_errors(
                    schema,
                    &inl.selection_set,
                    type_name,
                    fragments,
                    errors,
                    visited,
                );
            }
            Selection::FragmentSpread(spread) => {
                if !visited.contains(&spread.fragment_name) {
                    visited.insert(spread.fragment_name.clone());
                    if let Some(frag) = fragments.get(spread.fragment_name.as_str()) {
                        collect_type_correctness_errors(
                            schema,
                            &frag.selection_set,
                            &frag.type_condition,
                            fragments,
                            errors,
                            visited,
                        );
                    }
                }
            }
        }
    }
}

fn check_value_type_compat(
    value: &Value,
    expected: &GraphQLType,
    arg_name: &str,
) -> Option<ValidationError> {
    let inner = unwrap_type(expected);
    let mismatch = match value {
        Value::Variable(_) => return None,
        Value::NullValue => return None,
        Value::IntValue(i) => match inner.name() {
            "Int" => {
                if *i > i64::from(i32::MAX) || *i < i64::from(i32::MIN) {
                    return Some(ValidationError::new(
                        format!(
                            "Int cannot represent value: {} (out of 32-bit signed range)",
                            i
                        ),
                        ValidationRule::Spec(SpecRule::ValuesOfCorrectType),
                    ));
                }
                false
            }
            "Float" => false,
            "ID" => false,
            _ => true,
        },
        Value::FloatValue(_) => !matches!(inner.name(), "Float"),
        Value::StringValue(_) => !matches!(inner.name(), "String" | "ID"),
        Value::BooleanValue(_) => inner.name() != "Boolean",
        Value::EnumValue(ev) => match inner {
            GraphQLType::Enum(enum_type) => {
                if !enum_type.values.contains_key(ev.as_str()) {
                    return Some(ValidationError::new(
                        format!(
                            "Value \"{}\" does not exist in \"{}\" enum.",
                            ev, enum_type.name
                        ),
                        ValidationRule::Spec(SpecRule::ValuesOfCorrectType),
                    ));
                }
                false
            }
            _ => true,
        },
        Value::ListValue(_) => !matches!(expected, GraphQLType::List(_)),
        Value::ObjectValue(_) => !matches!(inner, GraphQLType::InputObject(_)),
    };

    if mismatch {
        Some(ValidationError::new(
            format!(
                "Expected value of type \"{}\", found incompatible value for argument \"{}\".",
                inner.name(),
                arg_name
            ),
            ValidationRule::Spec(SpecRule::ValuesOfCorrectType),
        ))
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProvidedRequiredArguments
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_provided_required_arguments(
    schema: &Schema,
    doc: &Document,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let fragments = collect_fragments(doc);

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            let root_type_name = match op.operation_type {
                crate::ast::OperationType::Query => schema.query_type.as_deref(),
                crate::ast::OperationType::Mutation => schema.mutation_type.as_deref(),
                crate::ast::OperationType::Subscription => schema.subscription_type.as_deref(),
            };
            if let Some(root) = root_type_name {
                collect_required_arg_errors(
                    schema,
                    &op.selection_set,
                    root,
                    &fragments,
                    &mut errors,
                    &mut HashSet::new(),
                );
            }
        }
    }

    errors
}

fn collect_required_arg_errors(
    schema: &Schema,
    ss: &SelectionSet,
    parent_type_name: &str,
    fragments: &HashMap<String, &FragmentDefinition>,
    errors: &mut Vec<ValidationError>,
    visited: &mut HashSet<String>,
) {
    let parent_type = schema.get_type(parent_type_name);

    for sel in &ss.selections {
        match sel {
            Selection::Field(f) => {
                if f.name.starts_with("__") {
                    if let Some(ss2) = &f.selection_set {
                        collect_required_arg_errors(schema, ss2, "", fragments, errors, visited);
                    }
                    continue;
                }

                let field_def = parent_type.and_then(|pt| get_field_def(pt, &f.name));

                if let Some(ft) = field_def {
                    let provided: HashSet<&str> =
                        f.arguments.iter().map(|a| a.name.as_str()).collect();
                    for (arg_name, arg_def) in &ft.arguments {
                        let is_required = matches!(&arg_def.argument_type, GraphQLType::NonNull(_))
                            && arg_def.default_value.is_none();
                        if is_required && !provided.contains(arg_name.as_str()) {
                            errors.push(ValidationError::new(
                                format!(
                                    "Field \"{}.{}\" argument \"{}\" of type \"{}\" is required, but it was not provided.",
                                    parent_type_name, f.name, arg_name,
                                    arg_def.argument_type
                                ),
                                ValidationRule::Spec(SpecRule::ProvidedRequiredArguments),
                            ));
                        }
                    }

                    if let Some(ss2) = &f.selection_set {
                        let return_type_name = unwrap_type(&ft.field_type).name().to_string();
                        collect_required_arg_errors(
                            schema,
                            ss2,
                            &return_type_name,
                            fragments,
                            errors,
                            visited,
                        );
                    }
                } else if let Some(ss2) = &f.selection_set {
                    collect_required_arg_errors(schema, ss2, "", fragments, errors, visited);
                }
            }
            Selection::InlineFragment(inl) => {
                let type_name = inl.type_condition.as_deref().unwrap_or(parent_type_name);
                collect_required_arg_errors(
                    schema,
                    &inl.selection_set,
                    type_name,
                    fragments,
                    errors,
                    visited,
                );
            }
            Selection::FragmentSpread(spread) => {
                if !visited.contains(&spread.fragment_name) {
                    visited.insert(spread.fragment_name.clone());
                    if let Some(frag) = fragments.get(spread.fragment_name.as_str()) {
                        collect_required_arg_errors(
                            schema,
                            &frag.selection_set,
                            &frag.type_condition,
                            fragments,
                            errors,
                            visited,
                        );
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OverlappingFieldsCanBeMerged
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_overlapping_fields_can_be_merged(doc: &Document) -> Vec<ValidationError> {
    let fragments = collect_fragments(doc);
    let mut errors = Vec::new();

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            let mut response_names: HashMap<String, (&str, &[crate::ast::Argument])> =
                HashMap::new();
            collect_response_name_conflicts(
                &op.selection_set,
                &fragments,
                &mut response_names,
                &mut errors,
                &mut HashSet::new(),
            );
        }
    }

    errors
}

fn collect_response_name_conflicts<'b>(
    ss: &'b SelectionSet,
    fragments: &'b HashMap<String, &'b FragmentDefinition>,
    response_names: &mut HashMap<String, (&'b str, &'b [crate::ast::Argument])>,
    errors: &mut Vec<ValidationError>,
    visited: &mut HashSet<String>,
) {
    for sel in &ss.selections {
        match sel {
            Selection::Field(f) => {
                let response_name = f.alias.as_deref().unwrap_or(f.name.as_str());

                if let Some((prev_name, prev_args)) = response_names.get(response_name) {
                    if *prev_name != f.name.as_str() {
                        errors.push(ValidationError::new(
                            format!(
                                "Fields \"{}\" conflict because \"{}\" and \"{}\" are different fields. Use different aliases on the fields to fetch both if this was intentional.",
                                response_name, prev_name, f.name
                            ),
                            ValidationRule::Spec(SpecRule::OverlappingFieldsCanBeMerged),
                        ));
                    } else if !args_compatible(prev_args, &f.arguments) {
                        errors.push(ValidationError::new(
                            format!(
                                "Fields \"{}\" conflict because they have differing arguments. Use different aliases on the fields to fetch both if this was intentional.",
                                response_name
                            ),
                            ValidationRule::Spec(SpecRule::OverlappingFieldsCanBeMerged),
                        ));
                    }
                } else {
                    response_names
                        .insert(response_name.to_string(), (f.name.as_str(), &f.arguments));
                }

                if let Some(ss2) = &f.selection_set {
                    let mut sub_response_names = HashMap::new();
                    collect_response_name_conflicts(
                        ss2,
                        fragments,
                        &mut sub_response_names,
                        errors,
                        visited,
                    );
                }
            }
            Selection::InlineFragment(inl) => {
                collect_response_name_conflicts(
                    &inl.selection_set,
                    fragments,
                    response_names,
                    errors,
                    visited,
                );
            }
            Selection::FragmentSpread(spread) => {
                if !visited.contains(&spread.fragment_name) {
                    visited.insert(spread.fragment_name.clone());
                    if let Some(frag) = fragments.get(spread.fragment_name.as_str()) {
                        collect_response_name_conflicts(
                            &frag.selection_set,
                            fragments,
                            response_names,
                            errors,
                            visited,
                        );
                    }
                }
            }
        }
    }
}

fn args_compatible(a: &[crate::ast::Argument], b: &[crate::ast::Argument]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let a_map: HashMap<&str, &Value> = a
        .iter()
        .map(|arg| (arg.name.as_str(), &arg.value))
        .collect();
    for arg in b {
        match a_map.get(arg.name.as_str()) {
            Some(av) => {
                if *av != &arg.value {
                    return false;
                }
            }
            None => return false,
        }
    }
    true
}
