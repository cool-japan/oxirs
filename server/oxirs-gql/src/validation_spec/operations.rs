//! Operation-level and type-level validation rules.
//!
//! Covers: ExecutableDefinitions, UniqueOperationNames, LoneAnonymousOperation,
//! SingleFieldSubscriptions, KnownTypeNames, FragmentsOnCompositeTypes,
//! VariablesAreInputTypes, LeafFieldSelections, UniqueVariableNames,
//! NoUndefinedVariables, NoUnusedVariables, and shared helper utilities.

use std::collections::{HashMap, HashSet};

use crate::ast::{Definition, Document, FragmentDefinition, Selection, SelectionSet, Value};
use crate::types::{GraphQLType, Schema};
use crate::validation::{SpecRule, ValidationError, ValidationRule};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers (pub(super) so validation_spec.rs can call them)
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn base_type_name(t: &crate::ast::Type) -> &str {
    match t {
        crate::ast::Type::NamedType(name) => name.as_str(),
        crate::ast::Type::ListType(inner) => base_type_name(inner),
        crate::ast::Type::NonNullType(inner) => base_type_name(inner),
    }
}

pub(super) fn unwrap_type(t: &GraphQLType) -> &GraphQLType {
    match t {
        GraphQLType::NonNull(inner) | GraphQLType::List(inner) => unwrap_type(inner),
        other => other,
    }
}

pub(super) fn collect_fragments(doc: &Document) -> HashMap<String, &FragmentDefinition> {
    doc.definitions
        .iter()
        .filter_map(|d| {
            if let Definition::Fragment(f) = d {
                Some((f.name.clone(), f))
            } else {
                None
            }
        })
        .collect()
}

pub(super) fn collect_spread_names(ss: &SelectionSet, out: &mut Vec<String>) {
    for sel in &ss.selections {
        match sel {
            Selection::FragmentSpread(spread) => out.push(spread.fragment_name.clone()),
            Selection::InlineFragment(inl) => collect_spread_names(&inl.selection_set, out),
            Selection::Field(f) => {
                if let Some(ss2) = &f.selection_set {
                    collect_spread_names(ss2, out);
                }
            }
        }
    }
}

pub(super) fn collect_vars_from_value(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Variable(var) => out.push(var.name.clone()),
        Value::ListValue(list) => {
            for item in list {
                collect_vars_from_value(item, out);
            }
        }
        Value::ObjectValue(obj) => {
            for v in obj.values() {
                collect_vars_from_value(v, out);
            }
        }
        _ => {}
    }
}

pub(super) fn get_field_return_type<'b>(
    parent: &'b GraphQLType,
    field_name: &str,
) -> Option<&'b GraphQLType> {
    match parent {
        GraphQLType::Object(obj) => obj.fields.get(field_name).map(|f| &f.field_type),
        GraphQLType::Interface(iface) => iface.fields.get(field_name).map(|f| &f.field_type),
        _ => None,
    }
}

pub(super) fn get_field_def<'b>(
    parent: &'b GraphQLType,
    field_name: &str,
) -> Option<&'b crate::types::FieldType> {
    match parent {
        GraphQLType::Object(obj) => obj.fields.get(field_name),
        GraphQLType::Interface(iface) => iface.fields.get(field_name),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExecutableDefinitions
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_executable_definitions(doc: &Document) -> Vec<ValidationError> {
    doc.definitions
        .iter()
        .filter_map(|def| match def {
            Definition::Operation(_) | Definition::Fragment(_) => None,
            _ => Some(ValidationError::new(
                "Non-executable definitions are not allowed in executable documents".to_string(),
                ValidationRule::Spec(SpecRule::ExecutableDefinitions),
            )),
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// UniqueOperationNames
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_unique_operation_names(doc: &Document) -> Vec<ValidationError> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut errors = Vec::new();

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            if let Some(name) = &op.name {
                let count = seen.entry(name.clone()).or_insert(0);
                *count += 1;
                if *count == 2 {
                    errors.push(ValidationError::new(
                        format!("There can be only one operation named \"{name}\""),
                        ValidationRule::Spec(SpecRule::UniqueOperationNames),
                    ));
                }
            }
        }
    }

    errors
}

// ─────────────────────────────────────────────────────────────────────────────
// LoneAnonymousOperation
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_lone_anonymous_operation(doc: &Document) -> Vec<ValidationError> {
    let operations: Vec<_> = doc
        .definitions
        .iter()
        .filter_map(|d| {
            if let Definition::Operation(op) = d {
                Some(op)
            } else {
                None
            }
        })
        .collect();

    let anonymous_count = operations.iter().filter(|op| op.name.is_none()).count();
    if anonymous_count > 0 && operations.len() > 1 {
        return vec![ValidationError::new(
            "This anonymous operation must be the only defined operation.".to_string(),
            ValidationRule::Spec(SpecRule::LoneAnonymousOperation),
        )];
    }

    vec![]
}

// ─────────────────────────────────────────────────────────────────────────────
// SingleFieldSubscriptions
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_single_field_subscriptions(doc: &Document) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            if op.operation_type == crate::ast::OperationType::Subscription {
                let non_introspection_fields: usize = op
                    .selection_set
                    .selections
                    .iter()
                    .filter(|sel| {
                        if let Selection::Field(f) = sel {
                            !f.name.starts_with("__")
                        } else {
                            true
                        }
                    })
                    .count();
                if non_introspection_fields != 1 {
                    let op_name = op.name.as_deref().unwrap_or("<anonymous>");
                    errors.push(ValidationError::new(
                        format!(
                            "Subscription \"{}\" must select only one top level field.",
                            op_name
                        ),
                        ValidationRule::Spec(SpecRule::SingleFieldSubscriptions),
                    ));
                }
            }
        }
    }

    errors
}

// ─────────────────────────────────────────────────────────────────────────────
// KnownTypeNames
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_known_type_names(schema: &Schema, doc: &Document) -> Vec<ValidationError> {
    let builtins = [
        "String",
        "Int",
        "Float",
        "Boolean",
        "ID",
        "__Schema",
        "__Type",
        "__Field",
        "__InputValue",
        "__EnumValue",
        "__Directive",
    ];
    let mut errors = Vec::new();

    let check_type_name = |name: &str| -> Option<ValidationError> {
        if builtins.contains(&name) || schema.types.contains_key(name) {
            None
        } else {
            Some(ValidationError::new(
                format!("Unknown type \"{name}\"."),
                ValidationRule::Spec(SpecRule::KnownTypeNames),
            ))
        }
    };

    for def in &doc.definitions {
        match def {
            Definition::Operation(op) => {
                for var_def in &op.variable_definitions {
                    let named = base_type_name(&var_def.type_);
                    if let Some(err) = check_type_name(named) {
                        errors.push(err);
                    }
                }
                collect_known_type_errors_from_selection(
                    schema,
                    &op.selection_set,
                    &mut errors,
                    &check_type_name,
                );
            }
            Definition::Fragment(frag) => {
                if let Some(err) = check_type_name(frag.type_condition.as_str()) {
                    errors.push(err);
                }
            }
            _ => {}
        }
    }

    errors
}

fn collect_known_type_errors_from_selection(
    _schema: &Schema,
    ss: &SelectionSet,
    errors: &mut Vec<ValidationError>,
    check: &dyn Fn(&str) -> Option<ValidationError>,
) {
    for sel in &ss.selections {
        match sel {
            Selection::InlineFragment(inl) => {
                if let Some(tc) = &inl.type_condition {
                    if let Some(err) = check(tc.as_str()) {
                        errors.push(err);
                    }
                }
                collect_known_type_errors_from_selection(
                    _schema,
                    &inl.selection_set,
                    errors,
                    check,
                );
            }
            Selection::Field(f) => {
                if let Some(ss2) = &f.selection_set {
                    collect_known_type_errors_from_selection(_schema, ss2, errors, check);
                }
            }
            Selection::FragmentSpread(_) => {}
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FragmentsOnCompositeTypes
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_fragments_on_composite_types(
    schema: &Schema,
    doc: &Document,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for def in &doc.definitions {
        if let Definition::Fragment(frag) = def {
            let is_composite = schema
                .get_type(&frag.type_condition)
                .map(|t| {
                    matches!(
                        t,
                        GraphQLType::Object(_) | GraphQLType::Interface(_) | GraphQLType::Union(_)
                    )
                })
                .unwrap_or(false);

            if !is_composite && schema.types.contains_key(&frag.type_condition) {
                errors.push(ValidationError::new(
                    format!(
                        "Fragment cannot condition on non composite type \"{}\".",
                        frag.type_condition
                    ),
                    ValidationRule::Spec(SpecRule::FragmentsOnCompositeTypes),
                ));
            }
        }
    }

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            collect_fragment_composite_errors(schema, &op.selection_set, &mut errors);
        }
    }

    errors
}

fn collect_fragment_composite_errors(
    schema: &Schema,
    ss: &SelectionSet,
    errors: &mut Vec<ValidationError>,
) {
    for sel in &ss.selections {
        match sel {
            Selection::InlineFragment(inl) => {
                if let Some(tc) = &inl.type_condition {
                    let is_composite = schema
                        .get_type(tc)
                        .map(|t| {
                            matches!(
                                t,
                                GraphQLType::Object(_)
                                    | GraphQLType::Interface(_)
                                    | GraphQLType::Union(_)
                            )
                        })
                        .unwrap_or(true);
                    if !is_composite {
                        errors.push(ValidationError::new(
                            format!("Fragment cannot condition on non composite type \"{tc}\"."),
                            ValidationRule::Spec(SpecRule::FragmentsOnCompositeTypes),
                        ));
                    }
                }
                collect_fragment_composite_errors(schema, &inl.selection_set, errors);
            }
            Selection::Field(f) => {
                if let Some(ss2) = &f.selection_set {
                    collect_fragment_composite_errors(schema, ss2, errors);
                }
            }
            Selection::FragmentSpread(_) => {}
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VariablesAreInputTypes
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_variables_are_input_types(
    schema: &Schema,
    doc: &Document,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            for var_def in &op.variable_definitions {
                let type_name = base_type_name(&var_def.type_);
                let is_input = schema.get_type(type_name).map(|t| {
                    matches!(
                        t,
                        GraphQLType::Scalar(_) | GraphQLType::Enum(_) | GraphQLType::InputObject(_)
                    )
                });

                if let Some(false) = is_input {
                    errors.push(ValidationError::new(
                        format!(
                            "Variable \"${}\" cannot be non-input type \"{}\".",
                            var_def.variable.name, type_name
                        ),
                        ValidationRule::Spec(SpecRule::VariablesAreInputTypes),
                    ));
                }
            }
        }
    }

    errors
}

// ─────────────────────────────────────────────────────────────────────────────
// LeafFieldSelections
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_leaf_field_selections(schema: &Schema, doc: &Document) -> Vec<ValidationError> {
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
                collect_leaf_errors(
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

fn collect_leaf_errors(
    schema: &Schema,
    ss: &SelectionSet,
    parent_type_name: &str,
    fragments: &HashMap<String, &FragmentDefinition>,
    errors: &mut Vec<ValidationError>,
    visited_fragments: &mut HashSet<String>,
) {
    let parent_type = schema.get_type(parent_type_name);

    for sel in &ss.selections {
        match sel {
            Selection::Field(field) => {
                if field.name.starts_with("__") {
                    continue;
                }

                let field_return_type =
                    parent_type.and_then(|pt| get_field_return_type(pt, &field.name));

                if let Some(return_type) = field_return_type {
                    let inner = unwrap_type(return_type);
                    let is_leaf = matches!(inner, GraphQLType::Scalar(_) | GraphQLType::Enum(_));
                    let has_subsel = field.selection_set.is_some();

                    if is_leaf && has_subsel {
                        errors.push(ValidationError::new(
                            format!(
                                "Field \"{}\" must not have a selection since type \"{}\" has no subfields.",
                                field.name,
                                inner.name()
                            ),
                            ValidationRule::Spec(SpecRule::LeafFieldSelections),
                        ));
                    } else if !is_leaf && !has_subsel {
                        errors.push(ValidationError::new(
                            format!(
                                "Field \"{}\" of type \"{}\" must have a selection of subfields. Did you mean \"{}\" {{ ... }}?",
                                field.name,
                                inner.name(),
                                field.name
                            ),
                            ValidationRule::Spec(SpecRule::LeafFieldSelections),
                        ));
                    }

                    if let Some(sub_ss) = &field.selection_set {
                        let inner_name = inner.name().to_string();
                        collect_leaf_errors(
                            schema,
                            sub_ss,
                            &inner_name,
                            fragments,
                            errors,
                            visited_fragments,
                        );
                    }
                } else if let Some(sub_ss) = &field.selection_set {
                    collect_leaf_errors(schema, sub_ss, "", fragments, errors, visited_fragments);
                }
            }
            Selection::InlineFragment(inl) => {
                let type_name = inl.type_condition.as_deref().unwrap_or(parent_type_name);
                collect_leaf_errors(
                    schema,
                    &inl.selection_set,
                    type_name,
                    fragments,
                    errors,
                    visited_fragments,
                );
            }
            Selection::FragmentSpread(spread) => {
                if visited_fragments.contains(&spread.fragment_name) {
                    continue;
                }
                visited_fragments.insert(spread.fragment_name.clone());
                if let Some(frag) = fragments.get(spread.fragment_name.as_str()) {
                    collect_leaf_errors(
                        schema,
                        &frag.selection_set,
                        &frag.type_condition,
                        fragments,
                        errors,
                        visited_fragments,
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// UniqueVariableNames
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_unique_variable_names(doc: &Document) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            let mut seen: HashMap<String, usize> = HashMap::new();
            for var_def in &op.variable_definitions {
                let count = seen.entry(var_def.variable.name.clone()).or_insert(0);
                *count += 1;
                if *count == 2 {
                    errors.push(ValidationError::new(
                        format!(
                            "There can be only one variable named \"${}\".",
                            var_def.variable.name
                        ),
                        ValidationRule::Spec(SpecRule::UniqueVariableNames),
                    ));
                }
            }
        }
    }

    errors
}

// ─────────────────────────────────────────────────────────────────────────────
// NoUndefinedVariables
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_no_undefined_variables(doc: &Document) -> Vec<ValidationError> {
    let fragments = collect_fragments(doc);
    let mut errors = Vec::new();

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            let defined: HashSet<String> = op
                .variable_definitions
                .iter()
                .map(|v| v.variable.name.clone())
                .collect();

            let mut used_vars = Vec::new();
            collect_variable_usages(
                &op.selection_set,
                &mut used_vars,
                &fragments,
                &mut HashSet::new(),
            );

            for var_name in &used_vars {
                if !defined.contains(var_name) {
                    let op_name = op.name.as_deref().unwrap_or("<anonymous>");
                    errors.push(ValidationError::new(
                        format!(
                            "Variable \"${}\" is not defined by operation \"{}\".",
                            var_name, op_name
                        ),
                        ValidationRule::Spec(SpecRule::NoUndefinedVariables),
                    ));
                }
            }
        }
    }

    errors
}

// ─────────────────────────────────────────────────────────────────────────────
// NoUnusedVariables
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_no_unused_variables(doc: &Document) -> Vec<ValidationError> {
    let fragments = collect_fragments(doc);
    let mut errors = Vec::new();

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            let mut used_vars: Vec<String> = Vec::new();
            collect_variable_usages(
                &op.selection_set,
                &mut used_vars,
                &fragments,
                &mut HashSet::new(),
            );

            let used_set: HashSet<String> = used_vars.into_iter().collect();

            for var_def in &op.variable_definitions {
                if !used_set.contains(&var_def.variable.name) {
                    let op_name = op.name.as_deref().unwrap_or("<anonymous>");
                    errors.push(ValidationError::new(
                        format!(
                            "Variable \"${}\" is never used in operation \"{}\".",
                            var_def.variable.name, op_name
                        ),
                        ValidationRule::Spec(SpecRule::NoUnusedVariables),
                    ));
                }
            }
        }
    }

    errors
}

pub(super) fn collect_variable_usages(
    ss: &SelectionSet,
    out: &mut Vec<String>,
    fragments: &HashMap<String, &FragmentDefinition>,
    visited: &mut HashSet<String>,
) {
    for sel in &ss.selections {
        match sel {
            Selection::Field(f) => {
                for arg in &f.arguments {
                    collect_vars_from_value(&arg.value, out);
                }
                if let Some(ss2) = &f.selection_set {
                    collect_variable_usages(ss2, out, fragments, visited);
                }
            }
            Selection::InlineFragment(inl) => {
                collect_variable_usages(&inl.selection_set, out, fragments, visited);
            }
            Selection::FragmentSpread(spread) => {
                if !visited.contains(&spread.fragment_name) {
                    visited.insert(spread.fragment_name.clone());
                    if let Some(frag) = fragments.get(spread.fragment_name.as_str()) {
                        collect_variable_usages(&frag.selection_set, out, fragments, visited);
                    }
                }
            }
        }
    }
}
