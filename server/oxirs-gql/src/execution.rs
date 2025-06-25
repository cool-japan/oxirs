//! GraphQL query execution engine
//!
//! This module provides the core execution engine for GraphQL queries,
//! including field resolution, error handling, and async execution.

use crate::ast::{
    Document, Field, OperationDefinition, OperationType, Selection, SelectionSet, Value,
};
use crate::types::{GraphQLType, Schema};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Execution context containing request-specific data
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub variables: HashMap<String, Value>,
    pub operation_name: Option<String>,
    pub request_id: String,
    pub fragments: HashMap<String, crate::ast::FragmentDefinition>,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            operation_name: None,
            request_id: uuid::Uuid::new_v4().to_string(),
            fragments: HashMap::new(),
        }
    }

    pub fn with_variables(mut self, variables: HashMap<String, Value>) -> Self {
        self.variables = variables;
        self
    }

    pub fn with_operation_name(mut self, operation_name: String) -> Self {
        self.operation_name = Some(operation_name);
        self
    }

    pub fn with_fragments(
        mut self,
        fragments: HashMap<String, crate::ast::FragmentDefinition>,
    ) -> Self {
        self.fragments = fragments;
        self
    }

    pub fn add_fragment(&mut self, name: String, fragment: crate::ast::FragmentDefinition) {
        self.fragments.insert(name, fragment);
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Fragment execution context for type condition checking
#[derive(Debug, Clone)]
pub struct FragmentContext {
    pub parent_type: String,
    pub object_type: Option<String>,
}

impl FragmentContext {
    pub fn new(parent_type: String) -> Self {
        Self {
            parent_type,
            object_type: None,
        }
    }

    pub fn with_object_type(mut self, object_type: String) -> Self {
        self.object_type = Some(object_type);
        self
    }

    pub fn can_apply_fragment(&self, type_condition: &str) -> bool {
        // Check if the fragment can be applied to the current type
        if let Some(ref obj_type) = self.object_type {
            obj_type == type_condition || self.parent_type == type_condition
        } else {
            self.parent_type == type_condition
        }
    }
}

/// Field resolver trait for resolving GraphQL fields
#[async_trait]
pub trait FieldResolver: Send + Sync {
    async fn resolve_field(
        &self,
        field_name: &str,
        args: &HashMap<String, Value>,
        context: &ExecutionContext,
    ) -> Result<Value>;
}

/// Execution result containing data and errors
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub data: Option<JsonValue>,
    pub errors: Vec<GraphQLError>,
}

impl ExecutionResult {
    pub fn new() -> Self {
        Self {
            data: None,
            errors: Vec::new(),
        }
    }

    pub fn with_data(mut self, data: JsonValue) -> Self {
        self.data = Some(data);
        self
    }

    pub fn with_error(mut self, error: GraphQLError) -> Self {
        self.errors.push(error);
        self
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

impl Default for ExecutionResult {
    fn default() -> Self {
        Self::new()
    }
}

/// GraphQL execution error
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphQLError {
    pub message: String,
    pub path: Vec<String>,
    pub locations: Vec<SourceLocation>,
    pub extensions: HashMap<String, JsonValue>,
}

impl GraphQLError {
    pub fn new(message: String) -> Self {
        Self {
            message,
            path: Vec::new(),
            locations: Vec::new(),
            extensions: HashMap::new(),
        }
    }

    pub fn with_path(mut self, path: Vec<String>) -> Self {
        self.path = path;
        self
    }

    pub fn with_location(mut self, location: SourceLocation) -> Self {
        self.locations.push(location);
        self
    }

    pub fn with_extension(mut self, key: String, value: JsonValue) -> Self {
        self.extensions.insert(key, value);
        self
    }
}

/// Source location for error reporting
#[derive(Debug, Clone, serde::Serialize)]
pub struct SourceLocation {
    pub line: usize,
    pub column: usize,
}

impl SourceLocation {
    pub fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }
}

/// GraphQL query executor
pub struct QueryExecutor {
    schema: Arc<RwLock<Schema>>,
    resolvers: HashMap<String, Arc<dyn FieldResolver>>,
}

impl QueryExecutor {
    pub fn new(schema: Schema) -> Self {
        Self {
            schema: Arc::new(RwLock::new(schema)),
            resolvers: HashMap::new(),
        }
    }

    pub fn add_resolver(&mut self, type_name: String, resolver: Arc<dyn FieldResolver>) {
        self.resolvers.insert(type_name, resolver);
    }

    pub async fn execute(
        &self,
        document: &Document,
        context: &ExecutionContext,
    ) -> Result<ExecutionResult> {
        let schema = self.schema.read().await;

        // Collect fragments from the document
        let mut execution_context = context.clone();
        self.collect_fragments(document, &mut execution_context)?;

        // Find the operation to execute
        let operation = self.get_operation(document, &execution_context.operation_name)?;

        // Execute based on operation type
        match operation.operation_type {
            OperationType::Query => {
                let query_type = schema
                    .query_type
                    .as_ref()
                    .ok_or_else(|| anyhow!("Schema does not define a query type"))?;

                self.execute_selection_set(
                    &operation.selection_set,
                    query_type,
                    &execution_context,
                    &schema,
                    Vec::new(),
                )
                .await
            }
            OperationType::Mutation => {
                let mutation_type = schema
                    .mutation_type
                    .as_ref()
                    .ok_or_else(|| anyhow!("Schema does not define a mutation type"))?;

                self.execute_selection_set(
                    &operation.selection_set,
                    mutation_type,
                    &execution_context,
                    &schema,
                    Vec::new(),
                )
                .await
            }
            OperationType::Subscription => {
                Err(anyhow!("Subscription execution not yet implemented"))
            }
        }
    }

    fn collect_fragments(&self, document: &Document, context: &mut ExecutionContext) -> Result<()> {
        for definition in &document.definitions {
            if let crate::ast::Definition::Fragment(fragment) = definition {
                context.add_fragment(fragment.name.clone(), fragment.clone());
            }
        }
        Ok(())
    }

    fn get_operation<'a>(
        &self,
        document: &'a Document,
        operation_name: &Option<String>,
    ) -> Result<&'a OperationDefinition> {
        let operations: Vec<_> = document
            .definitions
            .iter()
            .filter_map(|def| match def {
                crate::ast::Definition::Operation(op) => Some(op),
                _ => None,
            })
            .collect();

        match (operations.len(), operation_name) {
            (0, _) => Err(anyhow!("No operations found in document")),
            (1, _) => Ok(operations[0]),
            (_, Some(name)) => operations
                .iter()
                .find(|op| op.name.as_ref() == Some(name))
                .copied()
                .ok_or_else(|| anyhow!("Operation '{}' not found", name)),
            (_, None) => Err(anyhow!(
                "Multiple operations found but no operation name specified"
            )),
        }
    }

    fn execute_selection_set<'a>(
        &'a self,
        selection_set: &'a SelectionSet,
        type_name: &'a str,
        context: &'a ExecutionContext,
        schema: &'a Schema,
        path: Vec<String>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExecutionResult>> + Send + 'a>>
    {
        Box::pin(async move {
            let mut result_data = HashMap::new();
            let mut errors = Vec::new();

            for selection in &selection_set.selections {
                match selection {
                    Selection::Field(field) => {
                        let field_path = {
                            let mut p = path.clone();
                            p.push(field.alias.as_ref().unwrap_or(&field.name).clone());
                            p
                        };

                        match self
                            .execute_field(field, type_name, context, schema, field_path.clone())
                            .await
                        {
                            Ok(value) => {
                                let field_name = field.alias.as_ref().unwrap_or(&field.name);
                                result_data.insert(field_name.clone(), value);
                            }
                            Err(err) => {
                                errors
                                    .push(GraphQLError::new(err.to_string()).with_path(field_path));
                            }
                        }
                    }
                    Selection::InlineFragment(inline_fragment) => {
                        match self
                            .execute_inline_fragment(
                                inline_fragment,
                                type_name,
                                context,
                                schema,
                                path.clone(),
                            )
                            .await
                        {
                            Ok(fragment_data) => {
                                // Merge fragment data into result
                                if let Some(fragment_object) = fragment_data.as_object() {
                                    for (key, value) in fragment_object {
                                        result_data.insert(key.clone(), value.clone());
                                    }
                                }
                            }
                            Err(err) => {
                                errors.push(
                                    GraphQLError::new(err.to_string()).with_path(path.clone()),
                                );
                            }
                        }
                    }
                    Selection::FragmentSpread(fragment_spread) => {
                        match self
                            .execute_fragment_spread(
                                fragment_spread,
                                type_name,
                                context,
                                schema,
                                path.clone(),
                            )
                            .await
                        {
                            Ok(fragment_data) => {
                                // Merge fragment data into result
                                if let Some(fragment_object) = fragment_data.as_object() {
                                    for (key, value) in fragment_object {
                                        result_data.insert(key.clone(), value.clone());
                                    }
                                }
                            }
                            Err(err) => {
                                errors.push(
                                    GraphQLError::new(err.to_string()).with_path(path.clone()),
                                );
                            }
                        }
                    }
                }
            }

            let data = serde_json::to_value(result_data)
                .map_err(|e| anyhow!("Failed to serialize result data: {}", e))?;

            Ok(ExecutionResult {
                data: Some(data),
                errors,
            })
        })
    }

    async fn execute_field(
        &self,
        field: &Field,
        parent_type: &str,
        context: &ExecutionContext,
        schema: &Schema,
        path: Vec<String>,
    ) -> Result<JsonValue> {
        // Get field arguments
        let args = self.collect_field_arguments(&field.arguments, context)?;

        // Get the field type from schema
        let field_type = self.get_field_type(parent_type, &field.name, schema)?;

        // Resolve the field value
        let field_value = if let Some(resolver) = self.resolvers.get(parent_type) {
            resolver.resolve_field(&field.name, &args, context).await?
        } else {
            // Default resolver - return null for missing resolvers
            Value::NullValue
        };

        // Convert to JSON value and handle nested selections
        self.complete_value(
            &field_type,
            &field_value,
            &field.selection_set,
            context,
            schema,
            path,
        )
        .await
    }

    fn collect_field_arguments(
        &self,
        arguments: &[crate::ast::Argument],
        context: &ExecutionContext,
    ) -> Result<HashMap<String, Value>> {
        let mut args = HashMap::new();

        for arg in arguments {
            let value = self.resolve_value(&arg.value, context)?;
            args.insert(arg.name.clone(), value);
        }

        Ok(args)
    }

    fn resolve_value(&self, value: &Value, context: &ExecutionContext) -> Result<Value> {
        match value {
            Value::Variable(var) => context
                .variables
                .get(&var.name)
                .cloned()
                .ok_or_else(|| anyhow!("Variable '{}' not defined", var.name)),
            _ => Ok(value.clone()),
        }
    }

    fn get_field_type<'a>(
        &self,
        parent_type: &str,
        field_name: &str,
        schema: &'a Schema,
    ) -> Result<&'a GraphQLType> {
        let parent_type_def = schema
            .get_type(parent_type)
            .ok_or_else(|| anyhow!("Type '{}' not found in schema", parent_type))?;

        match parent_type_def {
            GraphQLType::Object(obj) => obj
                .fields
                .get(field_name)
                .map(|field| &field.field_type)
                .ok_or_else(|| {
                    anyhow!("Field '{}' not found on type '{}'", field_name, parent_type)
                }),
            GraphQLType::Interface(iface) => iface
                .fields
                .get(field_name)
                .map(|field| &field.field_type)
                .ok_or_else(|| {
                    anyhow!(
                        "Field '{}' not found on interface '{}'",
                        field_name,
                        parent_type
                    )
                }),
            _ => Err(anyhow!(
                "Cannot select field '{}' on non-composite type '{}'",
                field_name,
                parent_type
            )),
        }
    }

    fn complete_value<'a>(
        &'a self,
        field_type: &'a GraphQLType,
        value: &'a Value,
        selection_set: &'a Option<SelectionSet>,
        context: &'a ExecutionContext,
        schema: &'a Schema,
        path: Vec<String>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<JsonValue>> + Send + 'a>> {
        Box::pin(async move {
            match field_type {
                GraphQLType::NonNull(inner_type) => {
                    if matches!(value, Value::NullValue) {
                        return Err(anyhow!("Cannot return null for non-null field"));
                    }
                    self.complete_value(inner_type, value, selection_set, context, schema, path)
                        .await
                }
                GraphQLType::List(inner_type) => match value {
                    Value::ListValue(list) => {
                        let mut result = Vec::new();
                        for (i, item) in list.iter().enumerate() {
                            let item_path = {
                                let mut p = path.clone();
                                p.push(i.to_string());
                                p
                            };
                            let completed = self
                                .complete_value(
                                    inner_type,
                                    item,
                                    selection_set,
                                    context,
                                    schema,
                                    item_path,
                                )
                                .await?;
                            result.push(completed);
                        }
                        Ok(JsonValue::Array(result))
                    }
                    Value::NullValue => Ok(JsonValue::Null),
                    _ => Err(anyhow!("Expected list value but got {:?}", value)),
                },
                GraphQLType::Scalar(_) => {
                    // For scalars, convert the value directly
                    self.serialize_scalar_value(value)
                }
                GraphQLType::Object(_) | GraphQLType::Interface(_) => {
                    if let Some(selection_set) = selection_set {
                        if matches!(value, Value::NullValue) {
                            return Ok(JsonValue::Null);
                        }

                        // Execute sub-selection
                        let result = self
                            .execute_selection_set(
                                selection_set,
                                field_type.name(),
                                context,
                                schema,
                                path,
                            )
                            .await?;

                        result
                            .data
                            .ok_or_else(|| anyhow!("No data returned from sub-selection"))
                    } else {
                        Err(anyhow!("Selection set required for object/interface type"))
                    }
                }
                GraphQLType::Union(union_type) => {
                    // Union type completion
                    self.complete_union_value(
                        union_type,
                        value,
                        selection_set,
                        context,
                        schema,
                        path,
                    )
                    .await
                }
                GraphQLType::Enum(enum_type) => {
                    // Enum type completion
                    self.complete_enum_value(enum_type, value)
                }
                GraphQLType::InputObject(_) => {
                    Err(anyhow!("Input object types cannot be used as output types"))
                }
            }
        })
    }

    fn serialize_scalar_value(&self, value: &Value) -> Result<JsonValue> {
        match value {
            Value::NullValue => Ok(JsonValue::Null),
            Value::IntValue(i) => Ok(JsonValue::Number((*i).into())),
            Value::FloatValue(f) => serde_json::Number::from_f64(*f)
                .map(JsonValue::Number)
                .ok_or_else(|| anyhow!("Invalid float value: {}", f)),
            Value::StringValue(s) => Ok(JsonValue::String(s.clone())),
            Value::BooleanValue(b) => Ok(JsonValue::Bool(*b)),
            Value::EnumValue(s) => Ok(JsonValue::String(s.clone())),
            Value::ListValue(list) => {
                let json_list: Result<Vec<JsonValue>> = list
                    .iter()
                    .map(|v| self.serialize_scalar_value(v))
                    .collect();
                Ok(JsonValue::Array(json_list?))
            }
            Value::ObjectValue(obj) => {
                let json_obj: Result<serde_json::Map<String, JsonValue>> = obj
                    .iter()
                    .map(|(k, v)| {
                        self.serialize_scalar_value(v)
                            .map(|json_v| (k.clone(), json_v))
                    })
                    .collect();
                Ok(JsonValue::Object(json_obj?))
            }
            Value::Variable(_) => Err(anyhow!("Variables should be resolved before serialization")),
        }
    }

    async fn execute_inline_fragment(
        &self,
        inline_fragment: &crate::ast::InlineFragment,
        parent_type: &str,
        context: &ExecutionContext,
        schema: &Schema,
        path: Vec<String>,
    ) -> Result<JsonValue> {
        // Check type condition if present
        if let Some(ref type_condition) = inline_fragment.type_condition {
            let fragment_context = FragmentContext::new(parent_type.to_string());
            if !fragment_context.can_apply_fragment(type_condition) {
                // Fragment doesn't apply to this type, return empty object
                return Ok(JsonValue::Object(serde_json::Map::new()));
            }
        }

        // Execute the fragment's selection set
        let result = self
            .execute_selection_set(
                &inline_fragment.selection_set,
                parent_type,
                context,
                schema,
                path,
            )
            .await?;

        result
            .data
            .ok_or_else(|| anyhow!("No data returned from inline fragment"))
    }

    async fn execute_fragment_spread(
        &self,
        fragment_spread: &crate::ast::FragmentSpread,
        parent_type: &str,
        context: &ExecutionContext,
        schema: &Schema,
        path: Vec<String>,
    ) -> Result<JsonValue> {
        // Look up the fragment definition
        let fragment_def = context
            .fragments
            .get(&fragment_spread.fragment_name)
            .ok_or_else(|| anyhow!("Fragment '{}' not found", fragment_spread.fragment_name))?;

        // Check type condition
        let fragment_context = FragmentContext::new(parent_type.to_string());
        if !fragment_context.can_apply_fragment(&fragment_def.type_condition) {
            // Fragment doesn't apply to this type, return empty object
            return Ok(JsonValue::Object(serde_json::Map::new()));
        }

        // Execute the fragment's selection set
        let result = self
            .execute_selection_set(
                &fragment_def.selection_set,
                parent_type,
                context,
                schema,
                path,
            )
            .await?;

        result
            .data
            .ok_or_else(|| anyhow!("No data returned from fragment spread"))
    }

    async fn complete_union_value(
        &self,
        union_type: &crate::types::UnionType,
        value: &Value,
        selection_set: &Option<SelectionSet>,
        context: &ExecutionContext,
        schema: &Schema,
        path: Vec<String>,
    ) -> Result<JsonValue> {
        if matches!(value, Value::NullValue) {
            return Ok(JsonValue::Null);
        }

        // For union types, we need to determine the concrete type
        // This is a simplified implementation - in practice, you'd need
        // type resolution logic based on the actual data

        let concrete_type = self.resolve_union_type(union_type, value, schema)?;

        if let Some(selection_set) = selection_set {
            let result = self
                .execute_selection_set(selection_set, &concrete_type, context, schema, path)
                .await?;

            let mut object_result = result
                .data
                .unwrap_or(JsonValue::Object(serde_json::Map::new()));

            // Add __typename field for union types
            if let JsonValue::Object(ref mut obj) = object_result {
                obj.insert("__typename".to_string(), JsonValue::String(concrete_type));
            }

            Ok(object_result)
        } else {
            Err(anyhow!("Selection set required for union type"))
        }
    }

    fn complete_enum_value(
        &self,
        enum_type: &crate::types::EnumType,
        value: &Value,
    ) -> Result<JsonValue> {
        match value {
            Value::NullValue => Ok(JsonValue::Null),
            Value::EnumValue(enum_val) => {
                // Validate that the enum value is valid for this enum type
                if enum_type.values.contains_key(enum_val) {
                    Ok(JsonValue::String(enum_val.clone()))
                } else {
                    Err(anyhow!(
                        "Invalid enum value '{}' for enum type '{}'",
                        enum_val,
                        enum_type.name
                    ))
                }
            }
            Value::StringValue(string_val) => {
                // Allow string values to be coerced to enum values
                if enum_type.values.contains_key(string_val) {
                    Ok(JsonValue::String(string_val.clone()))
                } else {
                    Err(anyhow!(
                        "Invalid enum value '{}' for enum type '{}'",
                        string_val,
                        enum_type.name
                    ))
                }
            }
            _ => Err(anyhow!(
                "Cannot coerce {:?} to enum type '{}'",
                value,
                enum_type.name
            )),
        }
    }

    fn resolve_union_type(
        &self,
        union_type: &crate::types::UnionType,
        value: &Value,
        schema: &Schema,
    ) -> Result<String> {
        // This is a simplified type resolution - in practice, you'd need
        // more sophisticated logic to determine the concrete type

        // For now, we'll use the first available type in the union
        // or try to infer from the value structure

        if union_type.types.is_empty() {
            return Err(anyhow!(
                "Union type '{}' has no possible types",
                union_type.name
            ));
        }

        match value {
            Value::ObjectValue(obj) => {
                // Try to determine type based on available fields
                for type_name in &union_type.types {
                    if let Some(GraphQLType::Object(object_type)) = schema.get_type(type_name) {
                        // Check if the object has fields that match this type
                        let has_matching_fields =
                            obj.keys().any(|key| object_type.fields.contains_key(key));

                        if has_matching_fields {
                            return Ok(type_name.clone());
                        }
                    }
                }

                // Fallback to first type
                Ok(union_type.types[0].clone())
            }
            _ => {
                // For non-object values, return the first type
                Ok(union_type.types[0].clone())
            }
        }
    }
}

/// Default resolver for simple field access
pub struct DefaultResolver;

#[async_trait]
impl FieldResolver for DefaultResolver {
    async fn resolve_field(
        &self,
        field_name: &str,
        _args: &HashMap<String, Value>,
        _context: &ExecutionContext,
    ) -> Result<Value> {
        // Default implementation returns null
        // In a real implementation, this would resolve from a data source
        match field_name {
            "hello" => Ok(Value::StringValue("Hello, World!".to_string())),
            "id" => Ok(Value::StringValue(uuid::Uuid::new_v4().to_string())),
            "__typename" => Ok(Value::StringValue("Query".to_string())), // Default typename
            _ => Ok(Value::NullValue),
        }
    }
}

/// Type information resolver for GraphQL type system
pub struct TypeInfoResolver {
    type_name: String,
}

impl TypeInfoResolver {
    pub fn new(type_name: String) -> Self {
        Self { type_name }
    }
}

#[async_trait]
impl FieldResolver for TypeInfoResolver {
    async fn resolve_field(
        &self,
        field_name: &str,
        _args: &HashMap<String, Value>,
        _context: &ExecutionContext,
    ) -> Result<Value> {
        match field_name {
            "__typename" => Ok(Value::StringValue(self.type_name.clone())),
            _ => Ok(Value::NullValue),
        }
    }
}
