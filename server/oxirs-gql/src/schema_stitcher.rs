//! GraphQL schema stitching — merging multiple [`SchemaDef`] instances into one.
//!
//! Supports three conflict-resolution policies:
//! * [`ConflictPolicy::SkipConflicting`] — skip the second definition when a
//!   conflict is detected.
//! * [`ConflictPolicy::OverwriteWithLast`] — the last schema wins.
//! * [`ConflictPolicy::Error`] — return [`StitchError::ConflictOnError`].

use std::collections::HashMap;

// ── Type kinds ──────────────────────────────────────────────────────────────

/// The kind of a GraphQL type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeKind {
    Object,
    Interface,
    Union,
    Enum,
    Input,
    Scalar,
}

// ── Type definitions ────────────────────────────────────────────────────────

/// A single field definition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldDef {
    pub name: String,
    pub field_type: String,
    pub args: Vec<(String, String)>,
    pub description: Option<String>,
}

impl FieldDef {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, field_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            field_type: field_type.into(),
            args: Vec::new(),
            description: None,
        }
    }

    /// Attach a human-readable description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Append an argument `(name, type)`.
    pub fn with_arg(mut self, name: impl Into<String>, arg_type: impl Into<String>) -> Self {
        self.args.push((name.into(), arg_type.into()));
        self
    }
}

/// A single type definition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeDef {
    pub name: String,
    pub kind: TypeKind,
    pub fields: Vec<FieldDef>,
    pub description: Option<String>,
}

impl TypeDef {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, kind: TypeKind) -> Self {
        Self {
            name: name.into(),
            kind,
            fields: Vec::new(),
            description: None,
        }
    }

    /// Attach a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Append a field.
    pub fn with_field(mut self, field: FieldDef) -> Self {
        self.fields.push(field);
        self
    }
}

// ── Schema definition ───────────────────────────────────────────────────────

/// A complete (possibly partial) GraphQL schema.
#[derive(Clone, Debug, Default)]
pub struct SchemaDef {
    pub types: HashMap<String, TypeDef>,
    pub query_type: String,
    pub mutation_type: Option<String>,
    pub subscription_type: Option<String>,
}

impl SchemaDef {
    /// Create a schema with the given root query type name.
    pub fn new(query_type: impl Into<String>) -> Self {
        Self {
            types: HashMap::new(),
            query_type: query_type.into(),
            mutation_type: None,
            subscription_type: None,
        }
    }

    /// Add (or overwrite) a type in this schema.
    pub fn add_type(&mut self, type_def: TypeDef) {
        self.types.insert(type_def.name.clone(), type_def);
    }

    /// Retrieve a type by name.
    pub fn get_type(&self, name: &str) -> Option<&TypeDef> {
        self.types.get(name)
    }

    /// Set the mutation root type name.
    pub fn with_mutation(mut self, name: impl Into<String>) -> Self {
        self.mutation_type = Some(name.into());
        self
    }

    /// Set the subscription root type name.
    pub fn with_subscription(mut self, name: impl Into<String>) -> Self {
        self.subscription_type = Some(name.into());
        self
    }
}

// ── Conflicts ───────────────────────────────────────────────────────────────

/// Describes a conflict that arose during stitching.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MergeConflict {
    /// Two schemas define the same type name with different [`TypeKind`]s.
    TypeConflict {
        name: String,
        kinds: (TypeKind, TypeKind),
    },
    /// Two schemas define the same field on the same type.
    FieldConflict {
        type_name: String,
        field_name: String,
    },
}

// ── Conflict policy ─────────────────────────────────────────────────────────

/// How the stitcher resolves conflicts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConflictPolicy {
    /// Skip the conflicting definition (keep the first seen).
    SkipConflicting,
    /// Overwrite with the later definition (last schema wins).
    OverwriteWithLast,
    /// Return an error immediately on the first conflict.
    Error,
}

// ── Result ───────────────────────────────────────────────────────────────────

/// The output of a successful stitch operation.
#[derive(Debug)]
pub struct StitchResult {
    /// The merged schema.
    pub schema: SchemaDef,
    /// Conflicts that were resolved according to the active policy.
    pub conflicts: Vec<MergeConflict>,
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by [`SchemaStitcher::stitch`].
#[derive(Debug)]
pub enum StitchError {
    /// No schemas were added before calling `stitch`.
    NoSchemas,
    /// A conflict was encountered under the [`ConflictPolicy::Error`] policy.
    ConflictOnError(MergeConflict),
}

impl std::fmt::Display for StitchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StitchError::NoSchemas => write!(f, "No schemas to stitch"),
            StitchError::ConflictOnError(c) => write!(f, "Conflict during stitching: {:?}", c),
        }
    }
}

impl std::error::Error for StitchError {}

// ── SchemaStitcher ───────────────────────────────────────────────────────────

/// Merges multiple [`SchemaDef`]s into a single unified schema.
pub struct SchemaStitcher {
    schemas: Vec<SchemaDef>,
    conflict_policy: ConflictPolicy,
}

impl SchemaStitcher {
    /// Create a new stitcher with the given conflict policy.
    pub fn new(conflict_policy: ConflictPolicy) -> Self {
        Self {
            schemas: Vec::new(),
            conflict_policy,
        }
    }

    /// Register a schema to be merged.
    pub fn add_schema(&mut self, schema: SchemaDef) {
        self.schemas.push(schema);
    }

    /// Merge all registered schemas into a single [`StitchResult`].
    pub fn stitch(&self) -> Result<StitchResult, StitchError> {
        if self.schemas.is_empty() {
            return Err(StitchError::NoSchemas);
        }

        // Start with the first schema as the base.
        let first = &self.schemas[0];
        let mut merged_types: HashMap<String, TypeDef> = first.types.clone();
        let base_query = first.query_type.clone();
        let mut merged_mutation = first.mutation_type.clone();
        let mut merged_subscription = first.subscription_type.clone();
        let mut conflicts: Vec<MergeConflict> = Vec::new();

        for schema in self.schemas.iter().skip(1) {
            // Propagate mutation / subscription roots (first non-None wins).
            if merged_mutation.is_none() {
                merged_mutation = schema.mutation_type.clone();
            }
            if merged_subscription.is_none() {
                merged_subscription = schema.subscription_type.clone();
            }

            for (type_name, incoming_type) in &schema.types {
                match merged_types.get(type_name) {
                    None => {
                        // No conflict — just insert.
                        merged_types.insert(type_name.clone(), incoming_type.clone());
                    }
                    Some(existing_type) => {
                        // Check for kind conflict.
                        if existing_type.kind != incoming_type.kind {
                            let conflict = MergeConflict::TypeConflict {
                                name: type_name.clone(),
                                kinds: (existing_type.kind.clone(), incoming_type.kind.clone()),
                            };
                            match self.conflict_policy {
                                ConflictPolicy::Error => {
                                    return Err(StitchError::ConflictOnError(conflict));
                                }
                                ConflictPolicy::SkipConflicting => {
                                    conflicts.push(conflict);
                                    continue; // keep existing
                                }
                                ConflictPolicy::OverwriteWithLast => {
                                    conflicts.push(conflict);
                                    merged_types.insert(type_name.clone(), incoming_type.clone());
                                    continue;
                                }
                            }
                        }

                        // Same kind — merge fields.
                        // Collect existing field name→index mapping with owned Strings to
                        // avoid holding borrows into `merged_entry.fields` while mutating it.
                        let existing_fields: HashMap<String, usize> = {
                            let entry = merged_types.get(type_name).unwrap();
                            entry
                                .fields
                                .iter()
                                .enumerate()
                                .map(|(i, f)| (f.name.clone(), i))
                                .collect()
                        };

                        // Determine what to do per-field before any mutation.
                        enum FieldAction {
                            Skip(MergeConflict),
                            Overwrite(usize, MergeConflict, FieldDef),
                            Append(FieldDef),
                            ReturnError(MergeConflict),
                        }

                        let mut actions: Vec<FieldAction> = Vec::new();
                        for incoming_field in &incoming_type.fields {
                            if let Some(&idx) = existing_fields.get(&incoming_field.name) {
                                let field_conflict = MergeConflict::FieldConflict {
                                    type_name: type_name.clone(),
                                    field_name: incoming_field.name.clone(),
                                };
                                match self.conflict_policy {
                                    ConflictPolicy::Error => {
                                        actions.push(FieldAction::ReturnError(field_conflict));
                                    }
                                    ConflictPolicy::SkipConflicting => {
                                        actions.push(FieldAction::Skip(field_conflict));
                                    }
                                    ConflictPolicy::OverwriteWithLast => {
                                        actions.push(FieldAction::Overwrite(
                                            idx,
                                            field_conflict,
                                            incoming_field.clone(),
                                        ));
                                    }
                                }
                            } else {
                                actions.push(FieldAction::Append(incoming_field.clone()));
                            }
                        }

                        // Apply actions (now we can mutably borrow merged_entry).
                        for action in actions {
                            match action {
                                FieldAction::ReturnError(c) => {
                                    return Err(StitchError::ConflictOnError(c));
                                }
                                FieldAction::Skip(c) => {
                                    conflicts.push(c);
                                }
                                FieldAction::Overwrite(idx, c, new_field) => {
                                    conflicts.push(c);
                                    let entry = merged_types.get_mut(type_name).unwrap();
                                    entry.fields[idx] = new_field;
                                }
                                FieldAction::Append(new_field) => {
                                    let entry = merged_types.get_mut(type_name).unwrap();
                                    entry.fields.push(new_field);
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut schema = SchemaDef::new(base_query);
        schema.mutation_type = merged_mutation;
        schema.subscription_type = merged_subscription;
        for (_, td) in merged_types {
            schema.add_type(td);
        }

        Ok(StitchResult { schema, conflicts })
    }

    /// Total number of types across all registered schemas.
    pub fn type_count(&self) -> usize {
        self.schemas.iter().map(|s| s.types.len()).sum()
    }

    /// Number of registered schemas.
    pub fn schema_count(&self) -> usize {
        self.schemas.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_schema(query_name: &str) -> SchemaDef {
        SchemaDef::new(query_name)
    }

    fn query_type(fields: &[(&str, &str)]) -> TypeDef {
        let mut td = TypeDef::new("Query", TypeKind::Object);
        for (n, t) in fields {
            td = td.with_field(FieldDef::new(*n, *t));
        }
        td
    }

    // ── schema_count / type_count ───────────────────────────────────────────

    #[test]
    fn test_schema_count_empty() {
        let s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        assert_eq!(s.schema_count(), 0);
    }

    #[test]
    fn test_schema_count_after_add() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        s.add_schema(make_schema("Query"));
        s.add_schema(make_schema("Query2"));
        assert_eq!(s.schema_count(), 2);
    }

    #[test]
    fn test_type_count_total() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        let mut schema1 = make_schema("Query");
        schema1.add_type(TypeDef::new("TypeA", TypeKind::Object));
        schema1.add_type(TypeDef::new("TypeB", TypeKind::Scalar));
        s.add_schema(schema1);

        let mut schema2 = make_schema("Query");
        schema2.add_type(TypeDef::new("TypeC", TypeKind::Enum));
        s.add_schema(schema2);

        assert_eq!(s.type_count(), 3);
    }

    // ── no schemas ──────────────────────────────────────────────────────────

    #[test]
    fn test_stitch_no_schemas_error() {
        let s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        assert!(matches!(s.stitch(), Err(StitchError::NoSchemas)));
    }

    // ── single schema passthrough ───────────────────────────────────────────

    #[test]
    fn test_single_schema_passthrough() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        let mut schema = make_schema("Query");
        schema.add_type(query_type(&[("hello", "String")]));
        s.add_schema(schema);

        let result = s.stitch().unwrap();
        assert!(result.conflicts.is_empty());
        let q = result.schema.get_type("Query").unwrap();
        assert_eq!(q.fields.len(), 1);
        assert_eq!(q.fields[0].name, "hello");
    }

    // ── merge non-conflicting types ─────────────────────────────────────────

    #[test]
    fn test_merge_disjoint_types() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        let mut schema1 = make_schema("Query");
        schema1.add_type(TypeDef::new("User", TypeKind::Object));
        s.add_schema(schema1);

        let mut schema2 = make_schema("Query");
        schema2.add_type(TypeDef::new("Product", TypeKind::Object));
        s.add_schema(schema2);

        let result = s.stitch().unwrap();
        assert!(result.conflicts.is_empty());
        assert!(result.schema.get_type("User").is_some());
        assert!(result.schema.get_type("Product").is_some());
    }

    #[test]
    fn test_merge_adds_non_conflicting_fields() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        let mut schema1 = make_schema("Query");
        schema1.add_type(query_type(&[("users", "[User]")]));
        s.add_schema(schema1);

        let mut schema2 = make_schema("Query");
        schema2.add_type(query_type(&[("products", "[Product]")]));
        s.add_schema(schema2);

        let result = s.stitch().unwrap();
        assert!(result.conflicts.is_empty());
        let q = result.schema.get_type("Query").unwrap();
        assert_eq!(q.fields.len(), 2);
    }

    // ── type conflict detection ─────────────────────────────────────────────

    #[test]
    fn test_type_conflict_skip() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        let mut schema1 = make_schema("Query");
        schema1.add_type(TypeDef::new("Overlap", TypeKind::Object));
        s.add_schema(schema1);

        let mut schema2 = make_schema("Query");
        schema2.add_type(TypeDef::new("Overlap", TypeKind::Enum));
        s.add_schema(schema2);

        let result = s.stitch().unwrap();
        assert_eq!(result.conflicts.len(), 1);
        // First definition (Object) wins.
        assert_eq!(
            result.schema.get_type("Overlap").unwrap().kind,
            TypeKind::Object
        );
    }

    #[test]
    fn test_type_conflict_overwrite() {
        let mut s = SchemaStitcher::new(ConflictPolicy::OverwriteWithLast);
        let mut schema1 = make_schema("Query");
        schema1.add_type(TypeDef::new("Overlap", TypeKind::Object));
        s.add_schema(schema1);

        let mut schema2 = make_schema("Query");
        schema2.add_type(TypeDef::new("Overlap", TypeKind::Enum));
        s.add_schema(schema2);

        let result = s.stitch().unwrap();
        assert_eq!(result.conflicts.len(), 1);
        // Last definition (Enum) wins.
        assert_eq!(
            result.schema.get_type("Overlap").unwrap().kind,
            TypeKind::Enum
        );
    }

    #[test]
    fn test_type_conflict_error_policy() {
        let mut s = SchemaStitcher::new(ConflictPolicy::Error);
        let mut schema1 = make_schema("Query");
        schema1.add_type(TypeDef::new("Overlap", TypeKind::Object));
        s.add_schema(schema1);

        let mut schema2 = make_schema("Query");
        schema2.add_type(TypeDef::new("Overlap", TypeKind::Scalar));
        s.add_schema(schema2);

        let result = s.stitch();
        assert!(matches!(result, Err(StitchError::ConflictOnError(_))));
    }

    // ── field-level conflict ────────────────────────────────────────────────

    #[test]
    fn test_field_conflict_skip() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        let mut schema1 = make_schema("Query");
        schema1.add_type(query_type(&[("hello", "String")]));
        s.add_schema(schema1);

        let mut schema2 = make_schema("Query");
        schema2.add_type(query_type(&[("hello", "Int")]));
        s.add_schema(schema2);

        let result = s.stitch().unwrap();
        assert_eq!(result.conflicts.len(), 1);
        // First field type wins.
        let q = result.schema.get_type("Query").unwrap();
        assert_eq!(q.fields[0].field_type, "String");
    }

    #[test]
    fn test_field_conflict_overwrite() {
        let mut s = SchemaStitcher::new(ConflictPolicy::OverwriteWithLast);
        let mut schema1 = make_schema("Query");
        schema1.add_type(query_type(&[("hello", "String")]));
        s.add_schema(schema1);

        let mut schema2 = make_schema("Query");
        schema2.add_type(query_type(&[("hello", "Int")]));
        s.add_schema(schema2);

        let result = s.stitch().unwrap();
        assert_eq!(result.conflicts.len(), 1);
        let q = result.schema.get_type("Query").unwrap();
        assert_eq!(q.fields[0].field_type, "Int");
    }

    #[test]
    fn test_field_conflict_error_policy() {
        let mut s = SchemaStitcher::new(ConflictPolicy::Error);
        let mut schema1 = make_schema("Query");
        schema1.add_type(query_type(&[("hello", "String")]));
        s.add_schema(schema1);

        let mut schema2 = make_schema("Query");
        schema2.add_type(query_type(&[("hello", "Int")]));
        s.add_schema(schema2);

        assert!(matches!(s.stitch(), Err(StitchError::ConflictOnError(_))));
    }

    // ── query_type / mutation / subscription roots ──────────────────────────

    #[test]
    fn test_query_root_from_first_schema() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        s.add_schema(make_schema("MyQuery"));
        let result = s.stitch().unwrap();
        assert_eq!(result.schema.query_type, "MyQuery");
    }

    #[test]
    fn test_mutation_type_propagated() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        s.add_schema(make_schema("Query").with_mutation("Mutation"));
        let result = s.stitch().unwrap();
        assert_eq!(result.schema.mutation_type, Some("Mutation".to_owned()));
    }

    #[test]
    fn test_mutation_from_second_schema_if_first_lacks_it() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        s.add_schema(make_schema("Query"));
        s.add_schema(make_schema("Query").with_mutation("Mutation2"));
        let result = s.stitch().unwrap();
        assert_eq!(result.schema.mutation_type, Some("Mutation2".to_owned()));
    }

    #[test]
    fn test_subscription_type_propagated() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        s.add_schema(make_schema("Query").with_subscription("Subscription"));
        let result = s.stitch().unwrap();
        assert_eq!(
            result.schema.subscription_type,
            Some("Subscription".to_owned())
        );
    }

    // ── SchemaDef helpers ───────────────────────────────────────────────────

    #[test]
    fn test_schema_def_add_and_get_type() {
        let mut schema = SchemaDef::new("Query");
        schema.add_type(TypeDef::new("User", TypeKind::Object));
        assert!(schema.get_type("User").is_some());
        assert!(schema.get_type("Missing").is_none());
    }

    #[test]
    fn test_type_def_with_fields() {
        let td = TypeDef::new("Query", TypeKind::Object)
            .with_field(FieldDef::new("id", "ID"))
            .with_field(FieldDef::new("name", "String"));
        assert_eq!(td.fields.len(), 2);
    }

    #[test]
    fn test_field_def_with_args() {
        let f = FieldDef::new("search", "String").with_arg("query", "String");
        assert_eq!(f.args.len(), 1);
        assert_eq!(f.args[0], ("query".into(), "String".into()));
    }

    // ── large merge ─────────────────────────────────────────────────────────

    #[test]
    fn test_merge_many_schemas() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        for i in 0..20usize {
            let mut schema = make_schema("Query");
            schema.add_type(TypeDef::new(format!("Type{}", i), TypeKind::Object));
            s.add_schema(schema);
        }
        let result = s.stitch().unwrap();
        assert!(result.conflicts.is_empty());
        assert_eq!(result.schema.types.len(), 20);
    }

    #[test]
    fn test_no_conflicts_reported_when_no_conflict() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        let mut schema1 = make_schema("Query");
        schema1.add_type(TypeDef::new("A", TypeKind::Object));
        s.add_schema(schema1);

        let mut schema2 = make_schema("Query");
        schema2.add_type(TypeDef::new("B", TypeKind::Object));
        s.add_schema(schema2);

        let result = s.stitch().unwrap();
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn test_empty_schemas_still_merged() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        s.add_schema(make_schema("Query"));
        s.add_schema(make_schema("Query"));
        let result = s.stitch().unwrap();
        assert!(result.conflicts.is_empty());
        assert_eq!(result.schema.query_type, "Query");
    }

    // ── additional coverage ──────────────────────────────────────────────────

    #[test]
    fn test_stitch_result_schema_has_query_type() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        s.add_schema(make_schema("RootQuery"));
        let r = s.stitch().unwrap();
        assert_eq!(r.schema.query_type, "RootQuery");
    }

    #[test]
    fn test_type_kind_object() {
        let td = TypeDef::new("Foo", TypeKind::Object);
        assert_eq!(td.kind, TypeKind::Object);
    }

    #[test]
    fn test_type_kind_interface() {
        let td = TypeDef::new("INode", TypeKind::Interface);
        assert_eq!(td.kind, TypeKind::Interface);
    }

    #[test]
    fn test_type_kind_union() {
        let td = TypeDef::new("SearchResult", TypeKind::Union);
        assert_eq!(td.kind, TypeKind::Union);
    }

    #[test]
    fn test_type_kind_enum() {
        let td = TypeDef::new("Status", TypeKind::Enum);
        assert_eq!(td.kind, TypeKind::Enum);
    }

    #[test]
    fn test_type_kind_input() {
        let td = TypeDef::new("CreateUserInput", TypeKind::Input);
        assert_eq!(td.kind, TypeKind::Input);
    }

    #[test]
    fn test_type_kind_scalar() {
        let td = TypeDef::new("DateTime", TypeKind::Scalar);
        assert_eq!(td.kind, TypeKind::Scalar);
    }

    #[test]
    fn test_schema_def_overwrite_type() {
        let mut schema = SchemaDef::new("Query");
        schema.add_type(TypeDef::new("User", TypeKind::Object));
        schema.add_type(TypeDef::new("User", TypeKind::Interface)); // overwrite
        assert_eq!(schema.get_type("User").unwrap().kind, TypeKind::Interface);
    }

    #[test]
    fn test_field_def_description() {
        let f = FieldDef::new("id", "ID").with_description("Primary key");
        assert_eq!(f.description, Some("Primary key".into()));
    }

    #[test]
    fn test_type_def_description() {
        let td = TypeDef::new("User", TypeKind::Object).with_description("A user");
        assert_eq!(td.description, Some("A user".into()));
    }

    #[test]
    fn test_merge_conflict_type_conflict_display() {
        let c = MergeConflict::TypeConflict {
            name: "Foo".into(),
            kinds: (TypeKind::Object, TypeKind::Enum),
        };
        // Just ensure it formats without panic.
        let _s = format!("{:?}", c);
    }

    #[test]
    fn test_merge_conflict_field_conflict_display() {
        let c = MergeConflict::FieldConflict {
            type_name: "Query".into(),
            field_name: "users".into(),
        };
        let _s = format!("{:?}", c);
    }

    #[test]
    fn test_stitch_error_display() {
        let e1 = StitchError::NoSchemas;
        assert!(!e1.to_string().is_empty());
        let e2 = StitchError::ConflictOnError(MergeConflict::TypeConflict {
            name: "X".into(),
            kinds: (TypeKind::Object, TypeKind::Scalar),
        });
        assert!(!e2.to_string().is_empty());
    }

    #[test]
    fn test_overwrite_policy_no_conflicts_when_all_unique() {
        let mut s = SchemaStitcher::new(ConflictPolicy::OverwriteWithLast);
        let mut schema1 = make_schema("Query");
        schema1.add_type(TypeDef::new("A", TypeKind::Object));
        s.add_schema(schema1);
        let mut schema2 = make_schema("Query");
        schema2.add_type(TypeDef::new("B", TypeKind::Scalar));
        s.add_schema(schema2);
        let r = s.stitch().unwrap();
        assert!(r.conflicts.is_empty());
        assert_eq!(r.schema.types.len(), 2);
    }

    #[test]
    fn test_type_count_sums_across_schemas() {
        let mut s = SchemaStitcher::new(ConflictPolicy::SkipConflicting);
        let mut schema1 = make_schema("Query");
        schema1.add_type(TypeDef::new("A", TypeKind::Object));
        s.add_schema(schema1);
        let mut schema2 = make_schema("Query");
        schema2.add_type(TypeDef::new("B", TypeKind::Scalar));
        schema2.add_type(TypeDef::new("C", TypeKind::Enum));
        s.add_schema(schema2);
        assert_eq!(s.type_count(), 3);
    }

    #[test]
    fn test_merge_interface_and_object_type_conflict() {
        let mut s = SchemaStitcher::new(ConflictPolicy::Error);
        let mut schema1 = make_schema("Query");
        schema1.add_type(TypeDef::new("Node", TypeKind::Interface));
        s.add_schema(schema1);
        let mut schema2 = make_schema("Query");
        schema2.add_type(TypeDef::new("Node", TypeKind::Object));
        s.add_schema(schema2);
        let r = s.stitch();
        assert!(matches!(r, Err(StitchError::ConflictOnError(MergeConflict::TypeConflict { .. }))));
    }

    #[test]
    fn test_field_def_no_args_initially() {
        let f = FieldDef::new("foo", "String");
        assert!(f.args.is_empty());
        assert!(f.description.is_none());
    }

    #[test]
    fn test_add_multiple_args_to_field() {
        let f = FieldDef::new("search", "Result")
            .with_arg("query", "String")
            .with_arg("limit", "Int")
            .with_arg("offset", "Int");
        assert_eq!(f.args.len(), 3);
    }
}
