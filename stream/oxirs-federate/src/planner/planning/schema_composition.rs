//! Schema Composition and Validation for Apollo Federation
//!
//! This module handles the composition of multiple GraphQL schemas into a unified
//! federated schema, including conflict resolution, directive processing, and validation.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::types::*;

/// Schema composition utilities
pub struct SchemaComposer;

impl SchemaComposer {
    /// Merge a single schema into the unified schema
    pub fn merge_schema_into_unified(
        unified: &mut UnifiedSchema,
        service_id: &str,
        schema: &FederatedSchema,
        config: &GraphQLFederationConfig,
    ) -> Result<()> {
        // Merge types
        for (type_name, type_def) in &schema.types {
            if let Some(existing) = unified.types.get(type_name) {
                // Handle type conflicts
                match config.type_conflict_resolution {
                    TypeConflictResolution::Error => {
                        return Err(anyhow!(
                            "Type conflict: {} exists in multiple schemas",
                            type_name
                        ));
                    }
                    TypeConflictResolution::Merge => {
                        let merged_type = Self::merge_type_definitions(existing, type_def, config)?;
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
                match config.field_conflict_resolution {
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
                match config.field_conflict_resolution {
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

        Ok(())
    }

    /// Merge two type definitions
    fn merge_type_definitions(
        existing: &TypeDefinition,
        new: &TypeDefinition,
        config: &GraphQLFederationConfig,
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
                        match config.field_merge_strategy {
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
                    directives: existing.directives.clone(), // TODO: Merge directives
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
    pub fn validate_unified_schema(schema: &UnifiedSchema) -> Result<()> {
        // Check for circular dependencies
        // Validate field types exist
        // Check directive usage
        // Validate federation constraints

        debug!(
            "Validating unified schema with {} types",
            schema.types.len()
        );

        // Basic validation - check that all field types exist
        for type_def in schema.types.values() {
            if let TypeKind::Object { fields } = &type_def.kind {
                for field_def in fields.values() {
                    if !schema.types.contains_key(&field_def.field_type) {
                        // Check if it's a built-in scalar type
                        if !Self::is_builtin_type(&field_def.field_type) {
                            return Err(anyhow!(
                                "Unknown type '{}' used in field",
                                field_def.field_type
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if a type is a built-in GraphQL scalar type
    fn is_builtin_type(type_name: &str) -> bool {
        matches!(type_name, "String" | "Int" | "Float" | "Boolean" | "ID")
    }

    /// Process schema for federation directives (@key, @external, @requires, @provides)
    pub fn process_schema_for_federation(
        composed: &mut ComposedSchema,
        service_id: &str,
        schema: &FederatedSchema,
    ) -> Result<()> {
        for (type_name, type_def) in &schema.types {
            // Check for @key directive (entity definition)
            for directive in &type_def.directives {
                match directive.name.as_str() {
                    "key" => {
                        let key_fields = Self::extract_key_fields_from_directive(directive)?;
                        composed.entity_types.insert(
                            type_name.clone(),
                            EntityTypeInfo {
                                key_fields,
                                owning_service: service_id.to_string(),
                                extending_services: Vec::new(),
                            },
                        );
                    }
                    _ => {}
                }
            }

            // Process field-level directives
            if let TypeKind::Object { fields } = &type_def.kind {
                for (field_name, field_def) in fields {
                    let field_key = format!("{}.{}", type_name, field_name);

                    for directive in &field_def.directives {
                        match directive.name.as_str() {
                            "external" => {
                                // Field is defined in another service
                                composed
                                    .field_ownership
                                    .insert(field_key.clone(), FieldOwnershipType::External);
                            }
                            "requires" => {
                                // Field requires other fields to be resolved first
                                let required_fields =
                                    Self::extract_requires_fields_from_directive(directive)?;
                                composed.field_ownership.insert(
                                    field_key.clone(),
                                    FieldOwnershipType::Requires(required_fields),
                                );
                            }
                            "provides" => {
                                // Field provides data for other services
                                let provided_fields =
                                    Self::extract_provides_fields_from_directive(directive)?;
                                composed.field_ownership.insert(
                                    field_key.clone(),
                                    FieldOwnershipType::Provides(provided_fields),
                                );
                            }
                            _ => {}
                        }
                    }

                    // Track field ownership by service
                    if !composed.field_ownership.contains_key(&field_key) {
                        composed
                            .field_ownership
                            .insert(field_key, FieldOwnershipType::Owned(service_id.to_string()));
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract key fields from @key directive
    fn extract_key_fields_from_directive(directive: &DirectiveUsage) -> Result<Vec<String>> {
        // Parse fields argument from @key(fields: "id name")
        if let Some(fields_value) = directive.arguments.get("fields") {
            if let Some(fields_str) = fields_value.as_str() {
                return Ok(fields_str
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect());
            }
        }
        Err(anyhow!("@key directive missing fields argument"))
    }

    /// Extract required fields from @requires directive
    fn extract_requires_fields_from_directive(directive: &DirectiveUsage) -> Result<Vec<String>> {
        if let Some(fields_value) = directive.arguments.get("fields") {
            if let Some(fields_str) = fields_value.as_str() {
                return Ok(fields_str
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect());
            }
        }
        Err(anyhow!("@requires directive missing fields argument"))
    }

    /// Extract provided fields from @provides directive
    fn extract_provides_fields_from_directive(directive: &DirectiveUsage) -> Result<Vec<String>> {
        if let Some(fields_value) = directive.arguments.get("fields") {
            if let Some(fields_str) = fields_value.as_str() {
                return Ok(fields_str
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect());
            }
        }
        Err(anyhow!("@provides directive missing fields argument"))
    }

    /// Extract federation directives from composed schema
    pub fn extract_federation_directives(_composed: &ComposedSchema) -> Result<Vec<String>> {
        // Federation directives that should be available in the composed schema
        Ok(vec![
            "key".to_string(),
            "external".to_string(),
            "requires".to_string(),
            "provides".to_string(),
            "extends".to_string(),
            "shareable".to_string(),
            "inaccessible".to_string(),
            "override".to_string(),
            "composeDirective".to_string(),
            "interfaceObject".to_string(),
        ])
    }

    /// Validate composed schema for federation compliance
    pub fn validate_composed_schema(composed: &ComposedSchema) -> Result<()> {
        // Validate entity types have proper key directives
        for (type_name, entity_info) in &composed.entity_types {
            if entity_info.key_fields.is_empty() {
                return Err(anyhow!(
                    "Entity type '{}' must have at least one key field",
                    type_name
                ));
            }
        }

        // Validate field ownership
        for (field_path, ownership) in &composed.field_ownership {
            match ownership {
                FieldOwnershipType::Requires(fields) if fields.is_empty() => {
                    return Err(anyhow!(
                        "Field '{}' with @requires directive must specify required fields",
                        field_path
                    ));
                }
                FieldOwnershipType::Provides(fields) if fields.is_empty() => {
                    return Err(anyhow!(
                        "Field '{}' with @provides directive must specify provided fields",
                        field_path
                    ));
                }
                _ => {}
            }
        }

        info!("Composed schema validation passed");
        Ok(())
    }

    /// Validate federated composition for consistency
    pub fn validate_federated_composition(composed: &ComposedSchema) -> Result<()> {
        debug!("Validating federated schema composition");

        // Check that all required fields are satisfied
        for (field_key, ownership) in &composed.field_ownership {
            if let FieldOwnershipType::Requires(required_fields) = ownership {
                for required_field in required_fields {
                    let required_key = format!(
                        "{}.{}",
                        field_key.split('.').next().unwrap_or(""),
                        required_field
                    );
                    if !composed.field_ownership.contains_key(&required_key) {
                        return Err(anyhow!(
                            "Required field {} not found for {}",
                            required_field,
                            field_key
                        ));
                    }
                }
            }
        }

        // Validate entity key fields exist
        for (type_name, entity_info) in &composed.entity_types {
            for key_field in &entity_info.key_fields {
                let field_key = format!("{}.{}", type_name, key_field);
                if !composed.field_ownership.contains_key(&field_key) {
                    return Err(anyhow!(
                        "Key field {} not found for entity {}",
                        key_field,
                        type_name
                    ));
                }
            }
        }

        info!("Federated schema composition validation successful");
        Ok(())
    }

    /// Detect breaking changes between schema versions
    pub fn detect_breaking_changes(
        old_schema: &FederatedSchema,
        new_schema: &FederatedSchema,
    ) -> Result<Vec<BreakingChange>> {
        let mut breaking_changes = Vec::new();

        // Check for removed types
        for type_name in old_schema.types.keys() {
            if !new_schema.types.contains_key(type_name) {
                breaking_changes.push(BreakingChange {
                    change_type: BreakingChangeType::TypeRemoved,
                    description: format!("Type '{}' was removed", type_name),
                    severity: BreakingChangeSeverity::High,
                });
            }
        }

        // Check for field changes in existing types
        for (type_name, new_type) in &new_schema.types {
            if let Some(old_type) = old_schema.types.get(type_name) {
                let type_changes =
                    Self::detect_type_breaking_changes(type_name, old_type, new_type)?;
                breaking_changes.extend(type_changes);
            }
        }

        Ok(breaking_changes)
    }

    /// Detect breaking changes in a specific type
    fn detect_type_breaking_changes(
        type_name: &str,
        old_type: &TypeDefinition,
        new_type: &TypeDefinition,
    ) -> Result<Vec<BreakingChange>> {
        let mut changes = Vec::new();

        // Check for removed fields
        if let (TypeKind::Object { fields: old_fields }, TypeKind::Object { fields: new_fields }) =
            (&old_type.kind, &new_type.kind)
        {
            for field_name in old_fields.keys() {
                if !new_fields.contains_key(field_name) {
                    changes.push(BreakingChange {
                        change_type: BreakingChangeType::FieldRemoved,
                        description: format!("Field '{}.{}' was removed", type_name, field_name),
                        severity: BreakingChangeSeverity::High,
                    });
                }
            }

            // Check for argument changes in existing fields
            for (field_name, new_field) in new_fields {
                if let Some(old_field) = old_fields.get(field_name) {
                    // Check for required arguments added
                    for (arg_name, new_arg) in &new_field.arguments {
                        if let Some(old_arg) = old_field.arguments.get(arg_name) {
                            // Check if argument became required
                            if old_arg.default_value.is_some() && new_arg.default_value.is_none() {
                                changes.push(BreakingChange {
                                    change_type: BreakingChangeType::ArgumentMadeRequired,
                                    description: format!(
                                        "Argument '{}.{}.{}' is now required",
                                        type_name, field_name, arg_name
                                    ),
                                    severity: BreakingChangeSeverity::Medium,
                                });
                            }
                        } else if new_arg.default_value.is_none() {
                            // New required argument
                            changes.push(BreakingChange {
                                change_type: BreakingChangeType::RequiredArgumentAdded,
                                description: format!(
                                    "Required argument '{}.{}.{}' was added",
                                    type_name, field_name, arg_name
                                ),
                                severity: BreakingChangeSeverity::High,
                            });
                        }
                    }
                }
            }
        }

        Ok(changes)
    }

    /// Generate composed SDL from federated schemas
    pub fn generate_composed_sdl(composed: &ComposedSchema) -> Result<String> {
        let mut sdl = String::new();

        // Add federation directives
        sdl.push_str("directive @key(fields: String!) on OBJECT | INTERFACE\n");
        sdl.push_str("directive @external on FIELD_DEFINITION\n");
        sdl.push_str("directive @requires(fields: String!) on FIELD_DEFINITION\n");
        sdl.push_str("directive @provides(fields: String!) on FIELD_DEFINITION\n\n");

        // Add entity types
        for (type_name, entity_info) in &composed.entity_types {
            sdl.push_str(&format!(
                "type {} @key(fields: \"{}\") {{\n",
                type_name,
                entity_info.key_fields.join(" ")
            ));
            sdl.push_str("  # Entity fields would be listed here\n");
            sdl.push_str("}\n\n");
        }

        Ok(sdl)
    }
}
