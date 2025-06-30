//! Schema management for GraphQL Federation
//!
//! This module handles schema registration, merging, validation, and unified schema creation
//! for GraphQL federation.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tracing::{debug, warn};

use super::types::*;

impl GraphQLFederation {
    /// Create a unified schema from all registered schemas
    pub async fn create_unified_schema(&self) -> Result<UnifiedSchema> {
        let schemas = self.schemas.read().await;

        if schemas.is_empty() {
            return Err(anyhow!("No schemas registered for federation"));
        }

        let mut unified = UnifiedSchema {
            types: HashMap::new(),
            queries: HashMap::new(),
            mutations: HashMap::new(),
            subscriptions: HashMap::new(),
            directives: HashMap::new(),
            schema_mapping: HashMap::new(),
        };

        // Merge all schemas
        for (service_id, schema) in schemas.iter() {
            self.merge_schema_into_unified(&mut unified, service_id, schema)?;
        }

        // Validate the unified schema
        self.validate_unified_schema(&unified)?;

        Ok(unified)
    }

    /// Merge a single schema into the unified schema
    fn merge_schema_into_unified(
        &self,
        unified: &mut UnifiedSchema,
        service_id: &str,
        schema: &FederatedSchema,
    ) -> Result<()> {
        // Merge types
        for (type_name, type_def) in &schema.types {
            if let Some(existing) = unified.types.get(type_name) {
                // Handle type conflicts
                match self.config.type_conflict_resolution {
                    TypeConflictResolution::Error => {
                        return Err(anyhow!(
                            "Type conflict: {} exists in multiple schemas",
                            type_name
                        ));
                    }
                    TypeConflictResolution::Merge => {
                        let merged_type = self.merge_type_definitions(existing, type_def)?;
                        unified.types.insert(type_name.clone(), merged_type);
                    }
                    TypeConflictResolution::ServicePriority => {
                        // Keep existing (first wins)
                    }
                }
            } else {
                unified.types.insert(type_name.clone(), type_def.clone());
            }

            // Track which service owns this type
            unified
                .schema_mapping
                .entry(type_name.clone())
                .or_insert_with(Vec::new)
                .push(service_id.to_string());
        }

        // Merge queries
        for (field_name, field_def) in &schema.queries {
            if unified.queries.contains_key(field_name) {
                match self.config.field_conflict_resolution {
                    FieldConflictResolution::Error => {
                        return Err(anyhow!(
                            "Query field conflict: {} exists in multiple schemas",
                            field_name
                        ));
                    }
                    FieldConflictResolution::Namespace => {
                        let namespaced_name = format!("{}_{}", service_id, field_name);
                        unified.queries.insert(namespaced_name, field_def.clone());
                    }
                    FieldConflictResolution::FirstWins => {
                        // Keep existing
                    }
                }
            } else {
                unified
                    .queries
                    .insert(field_name.clone(), field_def.clone());
            }
        }

        // Merge mutations
        for (field_name, field_def) in &schema.mutations {
            if unified.mutations.contains_key(field_name) {
                match self.config.field_conflict_resolution {
                    FieldConflictResolution::Error => {
                        return Err(anyhow!(
                            "Mutation field conflict: {} exists in multiple schemas",
                            field_name
                        ));
                    }
                    FieldConflictResolution::Namespace => {
                        let namespaced_name = format!("{}_{}", service_id, field_name);
                        unified.mutations.insert(namespaced_name, field_def.clone());
                    }
                    FieldConflictResolution::FirstWins => {
                        // Keep existing
                    }
                }
            } else {
                unified
                    .mutations
                    .insert(field_name.clone(), field_def.clone());
            }
        }

        // Merge subscriptions
        for (field_name, field_def) in &schema.subscriptions {
            if unified.subscriptions.contains_key(field_name) {
                warn!("Subscription field conflict: {}", field_name);
            } else {
                unified
                    .subscriptions
                    .insert(field_name.clone(), field_def.clone());
            }
        }

        // Merge directives
        for (directive_name, directive_def) in &schema.directives {
            if !unified.directives.contains_key(directive_name) {
                unified
                    .directives
                    .insert(directive_name.clone(), directive_def.clone());
            }
        }

        Ok(())
    }

    /// Merge two type definitions
    fn merge_type_definitions(
        &self,
        existing: &TypeDefinition,
        new: &TypeDefinition,
    ) -> Result<TypeDefinition> {
        match (&existing.kind, &new.kind) {
            (
                TypeKind::Object {
                    fields: existing_fields,
                },
                TypeKind::Object { fields: new_fields },
            ) => {
                let mut merged_fields = existing_fields.clone();

                for (field_name, field_def) in new_fields {
                    if merged_fields.contains_key(field_name) {
                        // Handle field conflicts within types
                        match self.config.field_merge_strategy {
                            FieldMergeStrategy::Union => {
                                // Keep both fields (error if incompatible)
                                if merged_fields[field_name] != *field_def {
                                    return Err(anyhow!(
                                        "Incompatible field definitions for {}.{}",
                                        existing.name,
                                        field_name
                                    ));
                                }
                            }
                            FieldMergeStrategy::Override => {
                                merged_fields.insert(field_name.clone(), field_def.clone());
                            }
                        }
                    } else {
                        merged_fields.insert(field_name.clone(), field_def.clone());
                    }
                }

                Ok(TypeDefinition {
                    name: existing.name.clone(),
                    description: existing.description.clone(),
                    kind: TypeKind::Object {
                        fields: merged_fields,
                    },
                    directives: self.merge_directives(&existing.directives, &new.directives),
                })
            }
            (
                TypeKind::Interface {
                    fields: existing_fields,
                },
                TypeKind::Interface { fields: new_fields },
            ) => {
                // Similar logic for interfaces
                let mut merged_fields = existing_fields.clone();
                merged_fields.extend(new_fields.clone());

                Ok(TypeDefinition {
                    name: existing.name.clone(),
                    description: existing.description.clone(),
                    kind: TypeKind::Interface {
                        fields: merged_fields,
                    },
                    directives: existing.directives.clone(),
                })
            }
            (
                TypeKind::Union {
                    possible_types: existing_types,
                },
                TypeKind::Union {
                    possible_types: new_types,
                },
            ) => {
                let mut merged_types = existing_types.clone();
                for new_type in new_types {
                    if !merged_types.contains(new_type) {
                        merged_types.push(new_type.clone());
                    }
                }

                Ok(TypeDefinition {
                    name: existing.name.clone(),
                    description: existing.description.clone(),
                    kind: TypeKind::Union {
                        possible_types: merged_types,
                    },
                    directives: self.merge_directives(&existing.directives, &new.directives),
                })
            }
            (
                TypeKind::Enum {
                    values: existing_values,
                },
                TypeKind::Enum { values: new_values },
            ) => {
                let mut merged_values = existing_values.clone();
                for new_value in new_values {
                    if !merged_values.iter().any(|v| v.name == new_value.name) {
                        merged_values.push(new_value.clone());
                    }
                }

                Ok(TypeDefinition {
                    name: existing.name.clone(),
                    description: existing.description.clone(),
                    kind: TypeKind::Enum {
                        values: merged_values,
                    },
                    directives: self.merge_directives(&existing.directives, &new.directives),
                })
            }
            _ => {
                // Cannot merge different kinds of types
                Err(anyhow!(
                    "Cannot merge different type kinds for type {}",
                    existing.name
                ))
            }
        }
    }

    /// Validate the unified schema for consistency
    fn validate_unified_schema(&self, schema: &UnifiedSchema) -> Result<()> {
        debug!(
            "Validating unified schema with {} types",
            schema.types.len()
        );

        let mut validation_errors = Vec::new();

        // Check that all field types exist
        for type_def in schema.types.values() {
            match &type_def.kind {
                TypeKind::Object { fields } | TypeKind::Interface { fields } => {
                    for field_def in fields.values() {
                        if !self.type_exists_in_schema(&field_def.field_type, schema) {
                            validation_errors.push(SchemaValidationError {
                                error_type: SchemaErrorType::TypeNotFound,
                                message: format!(
                                    "Unknown type '{}' used in field '{}.{}'",
                                    field_def.field_type, type_def.name, field_def.name
                                ),
                                location: Some(format!("{}.{}", type_def.name, field_def.name)),
                            });
                        }

                        // Validate field arguments
                        for arg_def in field_def.arguments.values() {
                            if !self.type_exists_in_schema(&arg_def.argument_type, schema) {
                                validation_errors.push(SchemaValidationError {
                                    error_type: SchemaErrorType::TypeNotFound,
                                    message: format!(
                                        "Unknown type '{}' used in argument '{}.{}.{}'",
                                        arg_def.argument_type,
                                        type_def.name,
                                        field_def.name,
                                        arg_def.name
                                    ),
                                    location: Some(format!(
                                        "{}.{}.{}",
                                        type_def.name, field_def.name, arg_def.name
                                    )),
                                });
                            }
                        }
                    }
                }
                TypeKind::Union { possible_types } => {
                    for union_type in possible_types {
                        if !schema.types.contains_key(union_type) {
                            validation_errors.push(SchemaValidationError {
                                error_type: SchemaErrorType::TypeNotFound,
                                message: format!(
                                    "Unknown type '{}' in union '{}'",
                                    union_type, type_def.name
                                ),
                                location: Some(type_def.name.clone()),
                            });
                        }
                    }
                }
                TypeKind::InputObject { fields } => {
                    for field_def in fields.values() {
                        if !self.type_exists_in_schema(&field_def.field_type, schema) {
                            validation_errors.push(SchemaValidationError {
                                error_type: SchemaErrorType::TypeNotFound,
                                message: format!(
                                    "Unknown type '{}' used in input field '{}.{}'",
                                    field_def.field_type, type_def.name, field_def.name
                                ),
                                location: Some(format!("{}.{}", type_def.name, field_def.name)),
                            });
                        }
                    }
                }
                _ => {} // Scalar and Enum types don't need field validation
            }
        }

        // Check for circular dependencies
        if let Err(cycle_error) = self.check_circular_dependencies(schema) {
            validation_errors.push(SchemaValidationError {
                error_type: SchemaErrorType::CircularDependency,
                message: cycle_error.to_string(),
                location: None,
            });
        }

        // Validate directive usage
        self.validate_directive_usage(schema, &mut validation_errors)?;

        if !validation_errors.is_empty() {
            let error_messages: Vec<String> = validation_errors
                .iter()
                .map(|e| format!("{}: {}", format!("{:?}", e.error_type), e.message))
                .collect();
            return Err(anyhow!(
                "Schema validation failed with {} errors:\n{}",
                validation_errors.len(),
                error_messages.join("\n")
            ));
        }

        Ok(())
    }

    /// Check if a type exists in the schema (including built-in types)
    fn type_exists_in_schema(&self, type_name: &str, schema: &UnifiedSchema) -> bool {
        // Remove list and non-null modifiers
        let base_type = type_name
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim_end_matches('!');

        // Check built-in scalar types
        if self.is_builtin_type(base_type) {
            return true;
        }

        // Check types defined in the schema
        schema.types.contains_key(base_type)
    }

    /// Check for circular dependencies in type definitions
    fn check_circular_dependencies(&self, schema: &UnifiedSchema) -> Result<()> {
        let mut visited = std::collections::HashSet::new();
        let mut visiting = std::collections::HashSet::new();

        for type_name in schema.types.keys() {
            if !visited.contains(type_name) {
                self.visit_type_for_cycles(
                    type_name,
                    schema,
                    &mut visited,
                    &mut visiting,
                    &mut Vec::new(),
                )?;
            }
        }

        Ok(())
    }

    /// Recursive helper for circular dependency detection
    fn visit_type_for_cycles(
        &self,
        type_name: &str,
        schema: &UnifiedSchema,
        visited: &mut std::collections::HashSet<String>,
        visiting: &mut std::collections::HashSet<String>,
        path: &mut Vec<String>,
    ) -> Result<()> {
        if visiting.contains(type_name) {
            return Err(anyhow!(
                "Circular dependency detected: {} -> {}",
                path.join(" -> "),
                type_name
            ));
        }

        if visited.contains(type_name) {
            return Ok(());
        }

        visiting.insert(type_name.to_string());
        path.push(type_name.to_string());

        if let Some(type_def) = schema.types.get(type_name) {
            match &type_def.kind {
                TypeKind::Object { fields } | TypeKind::Interface { fields } => {
                    for field_def in fields.values() {
                        let field_type = self.extract_base_type(&field_def.field_type);
                        if schema.types.contains_key(&field_type) {
                            self.visit_type_for_cycles(
                                &field_type,
                                schema,
                                visited,
                                visiting,
                                path,
                            )?;
                        }
                    }
                }
                TypeKind::Union { possible_types } => {
                    for union_type in possible_types {
                        if schema.types.contains_key(union_type) {
                            self.visit_type_for_cycles(
                                union_type, schema, visited, visiting, path,
                            )?;
                        }
                    }
                }
                TypeKind::InputObject { fields } => {
                    for field_def in fields.values() {
                        let field_type = self.extract_base_type(&field_def.field_type);
                        if schema.types.contains_key(&field_type) {
                            self.visit_type_for_cycles(
                                &field_type,
                                schema,
                                visited,
                                visiting,
                                path,
                            )?;
                        }
                    }
                }
                _ => {} // Scalar and Enum types don't create dependencies
            }
        }

        visiting.remove(type_name);
        path.pop();
        visited.insert(type_name.to_string());

        Ok(())
    }

    /// Extract base type name from GraphQL type notation
    fn extract_base_type(&self, type_notation: &str) -> String {
        type_notation
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim_end_matches('!')
            .to_string()
    }

    /// Validate directive usage in the schema
    fn validate_directive_usage(
        &self,
        schema: &UnifiedSchema,
        validation_errors: &mut Vec<SchemaValidationError>,
    ) -> Result<()> {
        // Check that all used directives are defined
        for type_def in schema.types.values() {
            for directive in &type_def.directives {
                if !schema.directives.contains_key(&directive.name) {
                    validation_errors.push(SchemaValidationError {
                        error_type: SchemaErrorType::InvalidDirective,
                        message: format!(
                            "Unknown directive '@{}' used on type '{}'",
                            directive.name, type_def.name
                        ),
                        location: Some(type_def.name.clone()),
                    });
                }
            }

            // Check directives on fields
            match &type_def.kind {
                TypeKind::Object { fields } | TypeKind::Interface { fields } => {
                    for field_def in fields.values() {
                        for directive in &field_def.directives {
                            if !schema.directives.contains_key(&directive.name) {
                                validation_errors.push(SchemaValidationError {
                                    error_type: SchemaErrorType::InvalidDirective,
                                    message: format!(
                                        "Unknown directive '@{}' used on field '{}.{}'",
                                        directive.name, type_def.name, field_def.name
                                    ),
                                    location: Some(format!("{}.{}", type_def.name, field_def.name)),
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Analyze schema capabilities for a specific service
    pub async fn analyze_schema_capabilities(
        &self,
        service_id: &str,
    ) -> Result<SchemaCapabilities> {
        let schemas = self.schemas.read().await;

        let schema = schemas
            .get(service_id)
            .ok_or_else(|| anyhow!("Schema not found for service: {}", service_id))?;

        let mut capabilities = SchemaCapabilities {
            supports_federation: false,
            supports_subscriptions: !schema.subscriptions.is_empty(),
            supports_defer_stream: false,
            entity_types: Vec::new(),
            custom_directives: Vec::new(),
            scalar_types: Vec::new(),
            estimated_complexity: 0.0,
        };

        // Check for federation directives
        for directive_def in schema.directives.values() {
            match directive_def.name.as_str() {
                "key" | "external" | "requires" | "provides" | "extends" => {
                    capabilities.supports_federation = true;
                }
                "defer" | "stream" => {
                    capabilities.supports_defer_stream = true;
                }
                _ => {
                    capabilities
                        .custom_directives
                        .push(directive_def.name.clone());
                }
            }
        }

        // Analyze types
        for type_def in schema.types.values() {
            match &type_def.kind {
                TypeKind::Scalar => {
                    if !self.is_builtin_type(&type_def.name) {
                        capabilities.scalar_types.push(type_def.name.clone());
                    }
                }
                TypeKind::Object { .. } => {
                    // Check if this is an entity type (has @key directive)
                    if type_def.directives.iter().any(|d| d.name == "key") {
                        capabilities.entity_types.push(type_def.name.clone());
                    }
                }
                _ => {}
            }
        }

        // Estimate complexity based on schema size
        capabilities.estimated_complexity = (schema.types.len() as f64)
            + (schema.queries.len() as f64 * 2.0)
            + (schema.mutations.len() as f64 * 3.0)
            + (schema.subscriptions.len() as f64 * 4.0);

        Ok(capabilities)
    }

    /// Update a schema dynamically and check for breaking changes
    pub async fn update_schema(
        &self,
        service_id: String,
        new_schema: FederatedSchema,
    ) -> Result<SchemaUpdateResult> {
        let schemas = self.schemas.read().await;
        let old_schema = schemas.get(&service_id).cloned();
        drop(schemas); // Release read lock

        let mut breaking_changes = Vec::new();
        let mut warnings = Vec::new();

        // Compare schemas if old schema exists
        if let Some(ref old_schema) = old_schema {
            self.detect_breaking_changes(old_schema, &new_schema, &mut breaking_changes)?;
            self.detect_warnings(old_schema, &new_schema, &mut warnings)?;
        }

        // Update the schema
        let mut schemas = self.schemas.write().await;
        schemas.insert(service_id.clone(), new_schema);

        Ok(SchemaUpdateResult {
            service_id,
            update_successful: true,
            breaking_changes,
            warnings,
            rollback_available: old_schema.is_some(),
        })
    }

    /// Detect breaking changes between schema versions
    fn detect_breaking_changes(
        &self,
        old_schema: &FederatedSchema,
        new_schema: &FederatedSchema,
        breaking_changes: &mut Vec<BreakingChange>,
    ) -> Result<()> {
        // Check for removed types
        for (type_name, _) in &old_schema.types {
            if !new_schema.types.contains_key(type_name) {
                breaking_changes.push(BreakingChange {
                    change_type: BreakingChangeType::TypeRemoved,
                    description: format!("Type '{}' was removed", type_name),
                    severity: BreakingChangeSeverity::High,
                });
            }
        }

        // Check for removed fields
        for (type_name, old_type) in &old_schema.types {
            if let Some(new_type) = new_schema.types.get(type_name) {
                self.detect_field_changes(type_name, old_type, new_type, breaking_changes)?;
            }
        }

        Ok(())
    }

    /// Detect field-level breaking changes
    fn detect_field_changes(
        &self,
        type_name: &str,
        old_type: &TypeDefinition,
        new_type: &TypeDefinition,
        breaking_changes: &mut Vec<BreakingChange>,
    ) -> Result<()> {
        match (&old_type.kind, &new_type.kind) {
            (TypeKind::Object { fields: old_fields }, TypeKind::Object { fields: new_fields })
            | (
                TypeKind::Interface { fields: old_fields },
                TypeKind::Interface { fields: new_fields },
            ) => {
                // Check for removed fields
                for (field_name, _) in old_fields {
                    if !new_fields.contains_key(field_name) {
                        breaking_changes.push(BreakingChange {
                            change_type: BreakingChangeType::FieldRemoved,
                            description: format!(
                                "Field '{}.{}' was removed",
                                type_name, field_name
                            ),
                            severity: BreakingChangeSeverity::High,
                        });
                    }
                }

                // Check for field type changes
                for (field_name, old_field) in old_fields {
                    if let Some(new_field) = new_fields.get(field_name) {
                        if old_field.field_type != new_field.field_type {
                            breaking_changes.push(BreakingChange {
                                change_type: BreakingChangeType::TypeChanged,
                                description: format!(
                                    "Field '{}.{}' changed type from '{}' to '{}'",
                                    type_name,
                                    field_name,
                                    old_field.field_type,
                                    new_field.field_type
                                ),
                                severity: BreakingChangeSeverity::Medium,
                            });
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Detect non-breaking warnings
    fn detect_warnings(
        &self,
        old_schema: &FederatedSchema,
        new_schema: &FederatedSchema,
        warnings: &mut Vec<String>,
    ) -> Result<()> {
        // Check for deprecated fields being removed
        for (type_name, old_type) in &old_schema.types {
            if let Some(new_type) = new_schema.types.get(type_name) {
                if let (
                    TypeKind::Object { fields: old_fields },
                    TypeKind::Object { fields: new_fields },
                ) = (&old_type.kind, &new_type.kind)
                {
                    for (field_name, old_field) in old_fields {
                        if !new_fields.contains_key(field_name) {
                            if old_field.directives.iter().any(|d| d.name == "deprecated") {
                                warnings.push(format!(
                                    "Deprecated field '{}.{}' was removed",
                                    type_name, field_name
                                ));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
