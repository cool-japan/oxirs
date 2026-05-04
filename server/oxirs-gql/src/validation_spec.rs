//! GraphQL June 2018 specification validation rules.
//!
//! Implements all 25 validation rules from the GraphQL June 2018 specification,
//! covering executable document validation, fragment rules, variable rules,
//! directive rules, argument rules, type coercion, and field-merge conflict detection.

use std::collections::{HashMap, HashSet};

use crate::ast::{
    Definition, Document, FragmentDefinition, OperationDefinition, Selection, SelectionSet, Value,
};
use crate::types::{GraphQLType, Schema};
use crate::validation::{SpecRule, ValidationError, ValidationRule};

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

        errors.extend(self.check_executable_definitions(doc));
        errors.extend(self.check_unique_operation_names(doc));
        errors.extend(self.check_lone_anonymous_operation(doc));
        errors.extend(self.check_single_field_subscriptions(doc));
        errors.extend(self.check_known_type_names(doc));
        errors.extend(self.check_fragments_on_composite_types(doc));
        errors.extend(self.check_variables_are_input_types(doc));
        errors.extend(self.check_leaf_field_selections(doc));
        errors.extend(self.check_unique_fragment_names(doc));
        errors.extend(self.check_known_fragment_names(doc));
        errors.extend(self.check_no_fragment_cycles(doc));
        errors.extend(self.check_no_unused_fragments(doc));
        errors.extend(self.check_unique_variable_names(doc));
        errors.extend(self.check_no_undefined_variables(doc));
        errors.extend(self.check_no_unused_variables(doc));
        errors.extend(self.check_known_directives(doc));
        errors.extend(self.check_unique_directives_per_location(doc));
        errors.extend(self.check_known_argument_names(doc));
        errors.extend(self.check_unique_argument_names(doc));
        errors.extend(self.check_values_of_correct_type(doc));
        errors.extend(self.check_provided_required_arguments(doc));
        errors.extend(self.check_overlapping_fields_can_be_merged(doc));

        errors
    }

    // ── ExecutableDefinitions ────────────────────────────────────────────────
    fn check_executable_definitions(&self, doc: &Document) -> Vec<ValidationError> {
        doc.definitions
            .iter()
            .filter_map(|def| match def {
                Definition::Operation(_) | Definition::Fragment(_) => None,
                _ => Some(ValidationError::new(
                    "Non-executable definitions are not allowed in executable documents"
                        .to_string(),
                    ValidationRule::Spec(SpecRule::ExecutableDefinitions),
                )),
            })
            .collect()
    }

    // ── UniqueOperationNames ─────────────────────────────────────────────────
    fn check_unique_operation_names(&self, doc: &Document) -> Vec<ValidationError> {
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

    // ── LoneAnonymousOperation ────────────────────────────────────────────────
    fn check_lone_anonymous_operation(&self, doc: &Document) -> Vec<ValidationError> {
        let operations: Vec<&OperationDefinition> = doc
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

    // ── SingleFieldSubscriptions ──────────────────────────────────────────────
    fn check_single_field_subscriptions(&self, doc: &Document) -> Vec<ValidationError> {
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

    // ── KnownTypeNames ────────────────────────────────────────────────────────
    fn check_known_type_names(&self, doc: &Document) -> Vec<ValidationError> {
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
            if builtins.contains(&name) || self.schema.types.contains_key(name) {
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
                        let named = Self::base_type_name(&var_def.type_);
                        if let Some(err) = check_type_name(named) {
                            errors.push(err);
                        }
                    }
                    self.collect_known_type_errors_from_selection(
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
        &self,
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
                    self.collect_known_type_errors_from_selection(
                        &inl.selection_set,
                        errors,
                        check,
                    );
                }
                Selection::Field(f) => {
                    if let Some(ss2) = &f.selection_set {
                        self.collect_known_type_errors_from_selection(ss2, errors, check);
                    }
                }
                Selection::FragmentSpread(_) => {}
            }
        }
    }

    fn base_type_name(t: &crate::ast::Type) -> &str {
        match t {
            crate::ast::Type::NamedType(name) => name.as_str(),
            crate::ast::Type::ListType(inner) => Self::base_type_name(inner),
            crate::ast::Type::NonNullType(inner) => Self::base_type_name(inner),
        }
    }

    // ── FragmentsOnCompositeTypes ─────────────────────────────────────────────
    fn check_fragments_on_composite_types(&self, doc: &Document) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for def in &doc.definitions {
            if let Definition::Fragment(frag) = def {
                let is_composite = self
                    .schema
                    .get_type(&frag.type_condition)
                    .map(|t| {
                        matches!(
                            t,
                            GraphQLType::Object(_)
                                | GraphQLType::Interface(_)
                                | GraphQLType::Union(_)
                        )
                    })
                    .unwrap_or(false);

                if !is_composite && self.schema.types.contains_key(&frag.type_condition) {
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
                self.collect_fragment_composite_errors(&op.selection_set, &mut errors);
            }
        }

        errors
    }

    fn collect_fragment_composite_errors(
        &self,
        ss: &SelectionSet,
        errors: &mut Vec<ValidationError>,
    ) {
        for sel in &ss.selections {
            match sel {
                Selection::InlineFragment(inl) => {
                    if let Some(tc) = &inl.type_condition {
                        let is_composite = self
                            .schema
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
                                format!(
                                    "Fragment cannot condition on non composite type \"{tc}\"."
                                ),
                                ValidationRule::Spec(SpecRule::FragmentsOnCompositeTypes),
                            ));
                        }
                    }
                    self.collect_fragment_composite_errors(&inl.selection_set, errors);
                }
                Selection::Field(f) => {
                    if let Some(ss2) = &f.selection_set {
                        self.collect_fragment_composite_errors(ss2, errors);
                    }
                }
                Selection::FragmentSpread(_) => {}
            }
        }
    }

    // ── VariablesAreInputTypes ────────────────────────────────────────────────
    fn check_variables_are_input_types(&self, doc: &Document) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for def in &doc.definitions {
            if let Definition::Operation(op) = def {
                for var_def in &op.variable_definitions {
                    let type_name = Self::base_type_name(&var_def.type_);
                    let is_input = self.schema.get_type(type_name).map(|t| {
                        matches!(
                            t,
                            GraphQLType::Scalar(_)
                                | GraphQLType::Enum(_)
                                | GraphQLType::InputObject(_)
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

    // ── LeafFieldSelections ───────────────────────────────────────────────────
    fn check_leaf_field_selections(&self, doc: &Document) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        let fragments = Self::collect_fragments(doc);

        for def in &doc.definitions {
            if let Definition::Operation(op) = def {
                let root_type_name = match op.operation_type {
                    crate::ast::OperationType::Query => self.schema.query_type.as_deref(),
                    crate::ast::OperationType::Mutation => self.schema.mutation_type.as_deref(),
                    crate::ast::OperationType::Subscription => {
                        self.schema.subscription_type.as_deref()
                    }
                };
                if let Some(root) = root_type_name {
                    self.collect_leaf_errors(
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
        &self,
        ss: &SelectionSet,
        parent_type_name: &str,
        fragments: &HashMap<String, &FragmentDefinition>,
        errors: &mut Vec<ValidationError>,
        visited_fragments: &mut HashSet<String>,
    ) {
        let parent_type = self.schema.get_type(parent_type_name);

        for sel in &ss.selections {
            match sel {
                Selection::Field(field) => {
                    if field.name.starts_with("__") {
                        continue;
                    }

                    let field_return_type =
                        parent_type.and_then(|pt| self.get_field_return_type(pt, &field.name));

                    if let Some(return_type) = field_return_type {
                        let inner = Self::unwrap_type(return_type);
                        let is_leaf =
                            matches!(inner, GraphQLType::Scalar(_) | GraphQLType::Enum(_));
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
                            self.collect_leaf_errors(
                                sub_ss,
                                &inner_name,
                                fragments,
                                errors,
                                visited_fragments,
                            );
                        }
                    } else if let Some(sub_ss) = &field.selection_set {
                        self.collect_leaf_errors(sub_ss, "", fragments, errors, visited_fragments);
                    }
                }
                Selection::InlineFragment(inl) => {
                    let type_name = inl.type_condition.as_deref().unwrap_or(parent_type_name);
                    self.collect_leaf_errors(
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
                        self.collect_leaf_errors(
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

    fn get_field_return_type<'b>(
        &self,
        parent: &'b GraphQLType,
        field_name: &str,
    ) -> Option<&'b GraphQLType> {
        match parent {
            GraphQLType::Object(obj) => obj.fields.get(field_name).map(|f| &f.field_type),
            GraphQLType::Interface(iface) => iface.fields.get(field_name).map(|f| &f.field_type),
            _ => None,
        }
    }

    fn unwrap_type(t: &GraphQLType) -> &GraphQLType {
        match t {
            GraphQLType::NonNull(inner) | GraphQLType::List(inner) => Self::unwrap_type(inner),
            other => other,
        }
    }

    // ── UniqueFragmentNames ───────────────────────────────────────────────────
    fn check_unique_fragment_names(&self, doc: &Document) -> Vec<ValidationError> {
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut errors = Vec::new();

        for def in &doc.definitions {
            if let Definition::Fragment(frag) = def {
                let count = seen.entry(frag.name.clone()).or_insert(0);
                *count += 1;
                if *count == 2 {
                    errors.push(ValidationError::new(
                        format!("There can be only one fragment named \"{}\".", frag.name),
                        ValidationRule::Spec(SpecRule::UniqueFragmentNames),
                    ));
                }
            }
        }

        errors
    }

    // ── KnownFragmentNames ────────────────────────────────────────────────────
    fn check_known_fragment_names(&self, doc: &Document) -> Vec<ValidationError> {
        let defined: HashSet<String> = doc
            .definitions
            .iter()
            .filter_map(|d| {
                if let Definition::Fragment(f) = d {
                    Some(f.name.clone())
                } else {
                    None
                }
            })
            .collect();

        let mut errors = Vec::new();
        for def in &doc.definitions {
            if let Definition::Operation(op) = def {
                self.collect_unknown_fragment_errors(&op.selection_set, &defined, &mut errors);
            }
        }

        errors
    }

    fn collect_unknown_fragment_errors(
        &self,
        ss: &SelectionSet,
        defined: &HashSet<String>,
        errors: &mut Vec<ValidationError>,
    ) {
        for sel in &ss.selections {
            match sel {
                Selection::FragmentSpread(spread) => {
                    if !defined.contains(&spread.fragment_name) {
                        errors.push(ValidationError::new(
                            format!(
                                "Unknown fragment \"{}\". Did you mean to define it?",
                                spread.fragment_name
                            ),
                            ValidationRule::Spec(SpecRule::KnownFragmentNames),
                        ));
                    }
                }
                Selection::InlineFragment(inl) => {
                    self.collect_unknown_fragment_errors(&inl.selection_set, defined, errors);
                }
                Selection::Field(f) => {
                    if let Some(ss2) = &f.selection_set {
                        self.collect_unknown_fragment_errors(ss2, defined, errors);
                    }
                }
            }
        }
    }

    // ── NoFragmentCycles ──────────────────────────────────────────────────────
    fn check_no_fragment_cycles(&self, doc: &Document) -> Vec<ValidationError> {
        let fragments = Self::collect_fragments(doc);

        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        for (name, frag) in &fragments {
            let mut deps = Vec::new();
            Self::collect_spread_names(&frag.selection_set, &mut deps);
            adjacency.insert(name.to_string(), deps);
        }

        let mut errors = Vec::new();
        let mut color: HashMap<String, u8> = HashMap::new();

        for name in adjacency.keys().cloned().collect::<Vec<_>>() {
            if !color.contains_key(&name) {
                self.dfs_cycle_detect(&name, &adjacency, &mut color, &mut errors, &mut Vec::new());
            }
        }

        errors
    }

    fn dfs_cycle_detect(
        &self,
        node: &str,
        adj: &HashMap<String, Vec<String>>,
        color: &mut HashMap<String, u8>,
        errors: &mut Vec<ValidationError>,
        stack: &mut Vec<String>,
    ) {
        color.insert(node.to_string(), 1);
        stack.push(node.to_string());

        if let Some(neighbors) = adj.get(node) {
            for neighbor in neighbors {
                match color.get(neighbor.as_str()).copied().unwrap_or(0) {
                    1 => {
                        let cycle_start = stack.iter().position(|s| s == neighbor).unwrap_or(0);
                        let cycle_path: Vec<&str> =
                            stack[cycle_start..].iter().map(|s| s.as_str()).collect();
                        errors.push(ValidationError::new(
                            format!(
                                "Cannot spread fragment \"{}\" within itself via {}.",
                                neighbor,
                                cycle_path.join(" \u{2192} ")
                            ),
                            ValidationRule::Spec(SpecRule::NoFragmentCycles),
                        ));
                    }
                    0 => {
                        self.dfs_cycle_detect(neighbor, adj, color, errors, stack);
                    }
                    _ => {}
                }
            }
        }

        stack.pop();
        color.insert(node.to_string(), 2);
    }

    fn collect_spread_names(ss: &SelectionSet, out: &mut Vec<String>) {
        for sel in &ss.selections {
            match sel {
                Selection::FragmentSpread(spread) => out.push(spread.fragment_name.clone()),
                Selection::InlineFragment(inl) => {
                    Self::collect_spread_names(&inl.selection_set, out)
                }
                Selection::Field(f) => {
                    if let Some(ss2) = &f.selection_set {
                        Self::collect_spread_names(ss2, out);
                    }
                }
            }
        }
    }

    fn collect_fragments(doc: &Document) -> HashMap<String, &FragmentDefinition> {
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

    // ── NoUnusedFragments ─────────────────────────────────────────────────────
    fn check_no_unused_fragments(&self, doc: &Document) -> Vec<ValidationError> {
        let defined: HashMap<String, &FragmentDefinition> = Self::collect_fragments(doc);
        let mut used: HashSet<String> = HashSet::new();

        for def in &doc.definitions {
            if let Definition::Operation(op) = def {
                self.collect_used_fragments(&op.selection_set, &mut used, &defined);
            }
        }

        defined
            .keys()
            .filter(|name| !used.contains(*name))
            .map(|name| {
                ValidationError::new(
                    format!("Fragment \"{}\" is never used.", name),
                    ValidationRule::Spec(SpecRule::NoUnusedFragments),
                )
            })
            .collect()
    }

    fn collect_used_fragments(
        &self,
        ss: &SelectionSet,
        used: &mut HashSet<String>,
        defined: &HashMap<String, &FragmentDefinition>,
    ) {
        for sel in &ss.selections {
            match sel {
                Selection::FragmentSpread(spread) => {
                    if !used.contains(&spread.fragment_name) {
                        used.insert(spread.fragment_name.clone());
                        if let Some(frag) = defined.get(&spread.fragment_name) {
                            self.collect_used_fragments(&frag.selection_set, used, defined);
                        }
                    }
                }
                Selection::InlineFragment(inl) => {
                    self.collect_used_fragments(&inl.selection_set, used, defined);
                }
                Selection::Field(f) => {
                    if let Some(ss2) = &f.selection_set {
                        self.collect_used_fragments(ss2, used, defined);
                    }
                }
            }
        }
    }

    // ── UniqueVariableNames ───────────────────────────────────────────────────
    fn check_unique_variable_names(&self, doc: &Document) -> Vec<ValidationError> {
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

    // ── NoUndefinedVariables ──────────────────────────────────────────────────
    fn check_no_undefined_variables(&self, doc: &Document) -> Vec<ValidationError> {
        let fragments = Self::collect_fragments(doc);
        let mut errors = Vec::new();

        for def in &doc.definitions {
            if let Definition::Operation(op) = def {
                let defined: HashSet<String> = op
                    .variable_definitions
                    .iter()
                    .map(|v| v.variable.name.clone())
                    .collect();

                let mut used_vars = Vec::new();
                self.collect_variable_usages(
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

    // ── NoUnusedVariables ─────────────────────────────────────────────────────
    fn check_no_unused_variables(&self, doc: &Document) -> Vec<ValidationError> {
        let fragments = Self::collect_fragments(doc);
        let mut errors = Vec::new();

        for def in &doc.definitions {
            if let Definition::Operation(op) = def {
                let mut used_vars: Vec<String> = Vec::new();
                self.collect_variable_usages(
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

    fn collect_variable_usages(
        &self,
        ss: &SelectionSet,
        out: &mut Vec<String>,
        fragments: &HashMap<String, &FragmentDefinition>,
        visited: &mut HashSet<String>,
    ) {
        for sel in &ss.selections {
            match sel {
                Selection::Field(f) => {
                    for arg in &f.arguments {
                        Self::collect_vars_from_value(&arg.value, out);
                    }
                    if let Some(ss2) = &f.selection_set {
                        self.collect_variable_usages(ss2, out, fragments, visited);
                    }
                }
                Selection::InlineFragment(inl) => {
                    self.collect_variable_usages(&inl.selection_set, out, fragments, visited);
                }
                Selection::FragmentSpread(spread) => {
                    if !visited.contains(&spread.fragment_name) {
                        visited.insert(spread.fragment_name.clone());
                        if let Some(frag) = fragments.get(spread.fragment_name.as_str()) {
                            self.collect_variable_usages(
                                &frag.selection_set,
                                out,
                                fragments,
                                visited,
                            );
                        }
                    }
                }
            }
        }
    }

    fn collect_vars_from_value(value: &Value, out: &mut Vec<String>) {
        match value {
            Value::Variable(var) => out.push(var.name.clone()),
            Value::ListValue(list) => {
                for item in list {
                    Self::collect_vars_from_value(item, out);
                }
            }
            Value::ObjectValue(obj) => {
                for v in obj.values() {
                    Self::collect_vars_from_value(v, out);
                }
            }
            _ => {}
        }
    }

    // ── KnownDirectives ───────────────────────────────────────────────────────
    fn check_known_directives(&self, doc: &Document) -> Vec<ValidationError> {
        let known = ["skip", "include", "deprecated", "specifiedBy"];
        let mut errors = Vec::new();

        for def in &doc.definitions {
            match def {
                Definition::Operation(op) => {
                    for dir in &op.directives {
                        if !known.contains(&dir.name.as_str())
                            && !self.schema.directives.contains_key(&dir.name)
                        {
                            errors.push(ValidationError::new(
                                format!("Unknown directive \"@{}\".", dir.name),
                                ValidationRule::Spec(SpecRule::KnownDirectives),
                            ));
                        }
                    }
                    self.collect_directive_errors_in_ss(&op.selection_set, &known, &mut errors);
                }
                Definition::Fragment(frag) => {
                    for dir in &frag.directives {
                        if !known.contains(&dir.name.as_str())
                            && !self.schema.directives.contains_key(&dir.name)
                        {
                            errors.push(ValidationError::new(
                                format!("Unknown directive \"@{}\".", dir.name),
                                ValidationRule::Spec(SpecRule::KnownDirectives),
                            ));
                        }
                    }
                    self.collect_directive_errors_in_ss(&frag.selection_set, &known, &mut errors);
                }
                _ => {}
            }
        }

        errors
    }

    fn collect_directive_errors_in_ss(
        &self,
        ss: &SelectionSet,
        known: &[&str],
        errors: &mut Vec<ValidationError>,
    ) {
        for sel in &ss.selections {
            match sel {
                Selection::Field(f) => {
                    for dir in &f.directives {
                        if !known.contains(&dir.name.as_str())
                            && !self.schema.directives.contains_key(&dir.name)
                        {
                            errors.push(ValidationError::new(
                                format!("Unknown directive \"@{}\".", dir.name),
                                ValidationRule::Spec(SpecRule::KnownDirectives),
                            ));
                        }
                    }
                    if let Some(ss2) = &f.selection_set {
                        self.collect_directive_errors_in_ss(ss2, known, errors);
                    }
                }
                Selection::InlineFragment(inl) => {
                    for dir in &inl.directives {
                        if !known.contains(&dir.name.as_str())
                            && !self.schema.directives.contains_key(&dir.name)
                        {
                            errors.push(ValidationError::new(
                                format!("Unknown directive \"@{}\".", dir.name),
                                ValidationRule::Spec(SpecRule::KnownDirectives),
                            ));
                        }
                    }
                    self.collect_directive_errors_in_ss(&inl.selection_set, known, errors);
                }
                Selection::FragmentSpread(spread) => {
                    for dir in &spread.directives {
                        if !known.contains(&dir.name.as_str())
                            && !self.schema.directives.contains_key(&dir.name)
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

    // ── UniqueDirectivesPerLocation ───────────────────────────────────────────
    fn check_unique_directives_per_location(&self, doc: &Document) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for def in &doc.definitions {
            match def {
                Definition::Operation(op) => {
                    Self::check_directive_uniqueness(&op.directives, &mut errors);
                    self.collect_unique_directive_errors_in_ss(&op.selection_set, &mut errors);
                }
                Definition::Fragment(frag) => {
                    Self::check_directive_uniqueness(&frag.directives, &mut errors);
                    self.collect_unique_directive_errors_in_ss(&frag.selection_set, &mut errors);
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

    fn collect_unique_directive_errors_in_ss(
        &self,
        ss: &SelectionSet,
        errors: &mut Vec<ValidationError>,
    ) {
        for sel in &ss.selections {
            match sel {
                Selection::Field(f) => {
                    Self::check_directive_uniqueness(&f.directives, errors);
                    if let Some(ss2) = &f.selection_set {
                        self.collect_unique_directive_errors_in_ss(ss2, errors);
                    }
                }
                Selection::InlineFragment(inl) => {
                    Self::check_directive_uniqueness(&inl.directives, errors);
                    self.collect_unique_directive_errors_in_ss(&inl.selection_set, errors);
                }
                Selection::FragmentSpread(spread) => {
                    Self::check_directive_uniqueness(&spread.directives, errors);
                }
            }
        }
    }

    // ── KnownArgumentNames ────────────────────────────────────────────────────
    fn check_known_argument_names(&self, doc: &Document) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        let fragments = Self::collect_fragments(doc);

        for def in &doc.definitions {
            if let Definition::Operation(op) = def {
                let root_type_name = match op.operation_type {
                    crate::ast::OperationType::Query => self.schema.query_type.as_deref(),
                    crate::ast::OperationType::Mutation => self.schema.mutation_type.as_deref(),
                    crate::ast::OperationType::Subscription => {
                        self.schema.subscription_type.as_deref()
                    }
                };
                if let Some(root) = root_type_name {
                    self.collect_known_arg_errors(
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
        &self,
        ss: &SelectionSet,
        parent_type_name: &str,
        fragments: &HashMap<String, &FragmentDefinition>,
        errors: &mut Vec<ValidationError>,
        visited: &mut HashSet<String>,
    ) {
        let parent_type = self.schema.get_type(parent_type_name);

        for sel in &ss.selections {
            match sel {
                Selection::Field(f) => {
                    if f.name.starts_with("__") {
                        if let Some(ss2) = &f.selection_set {
                            self.collect_known_arg_errors(ss2, "", fragments, errors, visited);
                        }
                        continue;
                    }

                    let field_def = parent_type.and_then(|pt| self.get_field_def(pt, &f.name));

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
                            let return_type_name =
                                Self::unwrap_type(&ft.field_type).name().to_string();
                            self.collect_known_arg_errors(
                                ss2,
                                &return_type_name,
                                fragments,
                                errors,
                                visited,
                            );
                        }
                    } else if let Some(ss2) = &f.selection_set {
                        self.collect_known_arg_errors(ss2, "", fragments, errors, visited);
                    }
                }
                Selection::InlineFragment(inl) => {
                    let type_name = inl.type_condition.as_deref().unwrap_or(parent_type_name);
                    self.collect_known_arg_errors(
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
                            self.collect_known_arg_errors(
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

    fn get_field_def<'b>(
        &self,
        parent: &'b GraphQLType,
        field_name: &str,
    ) -> Option<&'b crate::types::FieldType> {
        match parent {
            GraphQLType::Object(obj) => obj.fields.get(field_name),
            GraphQLType::Interface(iface) => iface.fields.get(field_name),
            _ => None,
        }
    }

    // ── UniqueArgumentNames ───────────────────────────────────────────────────
    fn check_unique_argument_names(&self, doc: &Document) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for def in &doc.definitions {
            if let Definition::Operation(op) = def {
                self.collect_unique_arg_errors_in_ss(&op.selection_set, &mut errors);
            }
        }

        errors
    }

    fn collect_unique_arg_errors_in_ss(
        &self,
        ss: &SelectionSet,
        errors: &mut Vec<ValidationError>,
    ) {
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
                        self.collect_unique_arg_errors_in_ss(ss2, errors);
                    }
                }
                Selection::InlineFragment(inl) => {
                    self.collect_unique_arg_errors_in_ss(&inl.selection_set, errors);
                }
                Selection::FragmentSpread(_) => {}
            }
        }
    }

    // ── ValuesOfCorrectType ───────────────────────────────────────────────────
    fn check_values_of_correct_type(&self, doc: &Document) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        let fragments = Self::collect_fragments(doc);

        for def in &doc.definitions {
            if let Definition::Operation(op) = def {
                let root_type_name = match op.operation_type {
                    crate::ast::OperationType::Query => self.schema.query_type.as_deref(),
                    crate::ast::OperationType::Mutation => self.schema.mutation_type.as_deref(),
                    crate::ast::OperationType::Subscription => {
                        self.schema.subscription_type.as_deref()
                    }
                };
                if let Some(root) = root_type_name {
                    self.collect_type_correctness_errors(
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
        &self,
        ss: &SelectionSet,
        parent_type_name: &str,
        fragments: &HashMap<String, &FragmentDefinition>,
        errors: &mut Vec<ValidationError>,
        visited: &mut HashSet<String>,
    ) {
        let parent_type = self.schema.get_type(parent_type_name);

        for sel in &ss.selections {
            match sel {
                Selection::Field(f) => {
                    if f.name.starts_with("__") {
                        if let Some(ss2) = &f.selection_set {
                            self.collect_type_correctness_errors(
                                ss2, "", fragments, errors, visited,
                            );
                        }
                        continue;
                    }

                    let field_def = parent_type.and_then(|pt| self.get_field_def(pt, &f.name));

                    if let Some(ft) = field_def {
                        for arg in &f.arguments {
                            if let Some(arg_def) = ft.arguments.get(&arg.name) {
                                if let Some(err) = self.check_value_type_compat(
                                    &arg.value,
                                    &arg_def.argument_type,
                                    &arg.name,
                                ) {
                                    errors.push(err);
                                }
                            }
                        }
                        if let Some(ss2) = &f.selection_set {
                            let return_type_name =
                                Self::unwrap_type(&ft.field_type).name().to_string();
                            self.collect_type_correctness_errors(
                                ss2,
                                &return_type_name,
                                fragments,
                                errors,
                                visited,
                            );
                        }
                    } else if let Some(ss2) = &f.selection_set {
                        self.collect_type_correctness_errors(ss2, "", fragments, errors, visited);
                    }
                }
                Selection::InlineFragment(inl) => {
                    let type_name = inl.type_condition.as_deref().unwrap_or(parent_type_name);
                    self.collect_type_correctness_errors(
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
                            self.collect_type_correctness_errors(
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
        &self,
        value: &Value,
        expected: &GraphQLType,
        arg_name: &str,
    ) -> Option<ValidationError> {
        let inner = Self::unwrap_type(expected);
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

    // ── ProvidedRequiredArguments ─────────────────────────────────────────────
    fn check_provided_required_arguments(&self, doc: &Document) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        let fragments = Self::collect_fragments(doc);

        for def in &doc.definitions {
            if let Definition::Operation(op) = def {
                let root_type_name = match op.operation_type {
                    crate::ast::OperationType::Query => self.schema.query_type.as_deref(),
                    crate::ast::OperationType::Mutation => self.schema.mutation_type.as_deref(),
                    crate::ast::OperationType::Subscription => {
                        self.schema.subscription_type.as_deref()
                    }
                };
                if let Some(root) = root_type_name {
                    self.collect_required_arg_errors(
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
        &self,
        ss: &SelectionSet,
        parent_type_name: &str,
        fragments: &HashMap<String, &FragmentDefinition>,
        errors: &mut Vec<ValidationError>,
        visited: &mut HashSet<String>,
    ) {
        let parent_type = self.schema.get_type(parent_type_name);

        for sel in &ss.selections {
            match sel {
                Selection::Field(f) => {
                    if f.name.starts_with("__") {
                        if let Some(ss2) = &f.selection_set {
                            self.collect_required_arg_errors(ss2, "", fragments, errors, visited);
                        }
                        continue;
                    }

                    let field_def = parent_type.and_then(|pt| self.get_field_def(pt, &f.name));

                    if let Some(ft) = field_def {
                        let provided: HashSet<&str> =
                            f.arguments.iter().map(|a| a.name.as_str()).collect();
                        for (arg_name, arg_def) in &ft.arguments {
                            let is_required =
                                matches!(&arg_def.argument_type, GraphQLType::NonNull(_))
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
                            let return_type_name =
                                Self::unwrap_type(&ft.field_type).name().to_string();
                            self.collect_required_arg_errors(
                                ss2,
                                &return_type_name,
                                fragments,
                                errors,
                                visited,
                            );
                        }
                    } else if let Some(ss2) = &f.selection_set {
                        self.collect_required_arg_errors(ss2, "", fragments, errors, visited);
                    }
                }
                Selection::InlineFragment(inl) => {
                    let type_name = inl.type_condition.as_deref().unwrap_or(parent_type_name);
                    self.collect_required_arg_errors(
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
                            self.collect_required_arg_errors(
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

    // ── OverlappingFieldsCanBeMerged ──────────────────────────────────────────
    fn check_overlapping_fields_can_be_merged(&self, doc: &Document) -> Vec<ValidationError> {
        let fragments = Self::collect_fragments(doc);
        let mut errors = Vec::new();

        for def in &doc.definitions {
            if let Definition::Operation(op) = def {
                let mut response_names: HashMap<String, (&str, &[crate::ast::Argument])> =
                    HashMap::new();
                self.collect_response_name_conflicts(
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
        &'b self,
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
                        } else if !Self::args_compatible(prev_args, &f.arguments) {
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
                        self.collect_response_name_conflicts(
                            ss2,
                            fragments,
                            &mut sub_response_names,
                            errors,
                            visited,
                        );
                    }
                }
                Selection::InlineFragment(inl) => {
                    self.collect_response_name_conflicts(
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
                            self.collect_response_name_conflicts(
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BuiltinScalars, FieldType, ObjectType};

    fn make_schema() -> Schema {
        let mut schema = Schema::new();

        let query_type = ObjectType::new("Query".to_string())
            .with_field(
                "hello".to_string(),
                FieldType::new(
                    "hello".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::string()),
                ),
            )
            .with_field(
                "world".to_string(),
                FieldType::new(
                    "world".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::string()),
                ),
            );

        schema.add_type(GraphQLType::Object(query_type));
        schema.set_query_type("Query".to_string());

        schema
    }

    fn validate(schema: &Schema, query: &str) -> Vec<ValidationError> {
        let doc = crate::parser::parse_document(query).expect("parse failed");
        SpecValidator::new(schema).validate(&doc)
    }

    // ── ExecutableDefinitions ────────────────────────────────────────────────

    #[test]
    fn test_executable_definitions_valid() {
        let schema = make_schema();
        let errors = validate(&schema, "{ hello }");
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    // ── UniqueOperationNames ─────────────────────────────────────────────────

    #[test]
    fn test_unique_operation_names_valid() {
        let schema = make_schema();
        let errors = validate(&schema, "query Foo { hello } query Bar { hello }");
        let unique_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.rule == ValidationRule::Spec(SpecRule::UniqueOperationNames))
            .collect();
        assert!(
            unique_errors.is_empty(),
            "unexpected errors: {:?}",
            unique_errors
        );
    }

    #[test]
    fn test_unique_operation_names_invalid() {
        let schema = make_schema();
        let errors = validate(&schema, "query Foo { hello } query Foo { world }");
        assert!(
            errors
                .iter()
                .any(|e| e.rule == ValidationRule::Spec(SpecRule::UniqueOperationNames)),
            "expected UniqueOperationNames error, got: {:?}",
            errors
        );
    }

    // ── LoneAnonymousOperation ────────────────────────────────────────────────

    #[test]
    fn test_lone_anonymous_operation_valid() {
        let schema = make_schema();
        let errors = validate(&schema, "{ hello }");
        let lone_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.rule == ValidationRule::Spec(SpecRule::LoneAnonymousOperation))
            .collect();
        assert!(
            lone_errors.is_empty(),
            "unexpected errors: {:?}",
            lone_errors
        );
    }

    #[test]
    fn test_lone_anonymous_operation_invalid() {
        let schema = make_schema();
        let errors = validate(&schema, "{ hello } query Foo { world }");
        assert!(
            errors
                .iter()
                .any(|e| e.rule == ValidationRule::Spec(SpecRule::LoneAnonymousOperation)),
            "expected LoneAnonymousOperation error, got: {:?}",
            errors
        );
    }

    // ── UniqueFragmentNames ───────────────────────────────────────────────────

    #[test]
    fn test_unique_fragment_names_valid() {
        let schema = make_schema();
        let errors = validate(&schema, "fragment A on Query { hello } query Q { ...A }");
        let frag_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.rule == ValidationRule::Spec(SpecRule::UniqueFragmentNames))
            .collect();
        assert!(
            frag_errors.is_empty(),
            "unexpected errors: {:?}",
            frag_errors
        );
    }

    // ── NoFragmentCycles ──────────────────────────────────────────────────────

    #[test]
    fn test_no_fragment_cycles_valid() {
        let schema = make_schema();
        let errors = validate(&schema, "fragment A on Query { hello } query Q { ...A }");
        let cycle_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.rule == ValidationRule::Spec(SpecRule::NoFragmentCycles))
            .collect();
        assert!(
            cycle_errors.is_empty(),
            "unexpected cycle errors: {:?}",
            cycle_errors
        );
    }

    #[test]
    fn test_no_fragment_cycles_invalid() {
        let schema = make_schema();
        // Fragment A spreads B which spreads A — cycle
        let errors = validate(
            &schema,
            "fragment A on Query { ...B } fragment B on Query { ...A } query Q { ...A }",
        );
        assert!(
            errors
                .iter()
                .any(|e| e.rule == ValidationRule::Spec(SpecRule::NoFragmentCycles)),
            "expected NoFragmentCycles error, got: {:?}",
            errors
        );
    }

    // ── NoUnusedFragments ─────────────────────────────────────────────────────

    #[test]
    fn test_no_unused_fragments_valid() {
        let schema = make_schema();
        let errors = validate(&schema, "fragment A on Query { hello } query Q { ...A }");
        let unused_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.rule == ValidationRule::Spec(SpecRule::NoUnusedFragments))
            .collect();
        assert!(
            unused_errors.is_empty(),
            "unexpected errors: {:?}",
            unused_errors
        );
    }

    #[test]
    fn test_no_unused_fragments_invalid() {
        let schema = make_schema();
        // Fragment A is defined but never used
        let errors = validate(&schema, "fragment A on Query { hello } query Q { world }");
        assert!(
            errors
                .iter()
                .any(|e| e.rule == ValidationRule::Spec(SpecRule::NoUnusedFragments)),
            "expected NoUnusedFragments error, got: {:?}",
            errors
        );
    }

    // ── UniqueVariableNames ───────────────────────────────────────────────────

    #[test]
    fn test_unique_variable_names_invalid() {
        let schema = make_schema();
        // Parser won't surface this for us — build doc manually
        let doc = Document {
            definitions: vec![Definition::Operation(crate::ast::OperationDefinition {
                operation_type: crate::ast::OperationType::Query,
                name: Some("Q".to_string()),
                variable_definitions: vec![
                    crate::ast::VariableDefinition {
                        variable: crate::ast::Variable {
                            name: "x".to_string(),
                        },
                        type_: crate::ast::Type::NamedType("String".to_string()),
                        default_value: None,
                        directives: vec![],
                    },
                    crate::ast::VariableDefinition {
                        variable: crate::ast::Variable {
                            name: "x".to_string(),
                        },
                        type_: crate::ast::Type::NamedType("String".to_string()),
                        default_value: None,
                        directives: vec![],
                    },
                ],
                directives: vec![],
                selection_set: SelectionSet {
                    selections: vec![Selection::Field(crate::ast::Field {
                        alias: None,
                        name: "hello".to_string(),
                        arguments: vec![],
                        directives: vec![],
                        selection_set: None,
                    })],
                },
            })],
        };
        let errors = SpecValidator::new(&schema).validate(&doc);
        assert!(
            errors
                .iter()
                .any(|e| e.rule == ValidationRule::Spec(SpecRule::UniqueVariableNames)),
            "expected UniqueVariableNames error, got: {:?}",
            errors
        );
    }

    // ── KnownFragmentNames ────────────────────────────────────────────────────

    #[test]
    fn test_known_fragment_names_invalid() {
        let schema = make_schema();
        // Spread refers to fragment "Undefined" which doesn't exist
        let doc = Document {
            definitions: vec![Definition::Operation(crate::ast::OperationDefinition {
                operation_type: crate::ast::OperationType::Query,
                name: None,
                variable_definitions: vec![],
                directives: vec![],
                selection_set: SelectionSet {
                    selections: vec![Selection::FragmentSpread(crate::ast::FragmentSpread {
                        fragment_name: "Undefined".to_string(),
                        directives: vec![],
                    })],
                },
            })],
        };
        let errors = SpecValidator::new(&schema).validate(&doc);
        assert!(
            errors
                .iter()
                .any(|e| e.rule == ValidationRule::Spec(SpecRule::KnownFragmentNames)),
            "expected KnownFragmentNames error, got: {:?}",
            errors
        );
    }

    // ── OverlappingFieldsCanBeMerged ──────────────────────────────────────────

    #[test]
    fn test_overlapping_fields_conflict() {
        let schema = make_schema();
        // Same response name "x" maps to different fields
        let doc = Document {
            definitions: vec![Definition::Operation(crate::ast::OperationDefinition {
                operation_type: crate::ast::OperationType::Query,
                name: None,
                variable_definitions: vec![],
                directives: vec![],
                selection_set: SelectionSet {
                    selections: vec![
                        Selection::Field(crate::ast::Field {
                            alias: Some("x".to_string()),
                            name: "hello".to_string(),
                            arguments: vec![],
                            directives: vec![],
                            selection_set: None,
                        }),
                        Selection::Field(crate::ast::Field {
                            alias: Some("x".to_string()),
                            name: "world".to_string(),
                            arguments: vec![],
                            directives: vec![],
                            selection_set: None,
                        }),
                    ],
                },
            })],
        };
        let errors = SpecValidator::new(&schema).validate(&doc);
        assert!(
            errors
                .iter()
                .any(|e| e.rule == ValidationRule::Spec(SpecRule::OverlappingFieldsCanBeMerged)),
            "expected OverlappingFieldsCanBeMerged error, got: {:?}",
            errors
        );
    }

    // ── ValuesOfCorrectType: Int 32-bit boundary ──────────────────────────────

    #[test]
    fn test_spec_validator_new() {
        let schema = make_schema();
        let _validator = SpecValidator::new(&schema);
    }
}
