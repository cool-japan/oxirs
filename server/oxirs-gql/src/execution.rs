//! GraphQL query execution engine
//!
//! This module provides the core execution engine for GraphQL queries,
//! including field resolution, error handling, and async execution.

use crate::ast::{Document, OperationDefinition, OperationType, Selection, SelectionSet, Field, Value};
use crate::types::{Schema, GraphQLType};
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
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            operation_name: None,
            request_id: uuid::Uuid::new_v4().to_string(),
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
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
        
        // Find the operation to execute
        let operation = self.get_operation(document, &context.operation_name)?;
        
        // Execute based on operation type
        match operation.operation_type {
            OperationType::Query => {
                let query_type = schema.query_type.as_ref()
                    .ok_or_else(|| anyhow!("Schema does not define a query type"))?;
                
                self.execute_selection_set(
                    &operation.selection_set,
                    query_type,
                    context,
                    &schema,
                    Vec::new(),
                ).await
            }
            OperationType::Mutation => {
                let mutation_type = schema.mutation_type.as_ref()
                    .ok_or_else(|| anyhow!("Schema does not define a mutation type"))?;
                
                self.execute_selection_set(
                    &operation.selection_set,
                    mutation_type,
                    context,
                    &schema,
                    Vec::new(),
                ).await
            }
            OperationType::Subscription => {
                Err(anyhow!("Subscription execution not yet implemented"))
            }
        }
    }

    fn get_operation<'a>(
        &self,
        document: &'a Document,
        operation_name: &Option<String>,
    ) -> Result<&'a OperationDefinition> {
        let operations: Vec<_> = document.definitions.iter()
            .filter_map(|def| match def {
                crate::ast::Definition::Operation(op) => Some(op),
                _ => None,
            })
            .collect();

        match (operations.len(), operation_name) {
            (0, _) => Err(anyhow!("No operations found in document")),
            (1, _) => Ok(operations[0]),
            (_, Some(name)) => {
                operations.iter()
                    .find(|op| op.name.as_ref() == Some(name))
                    .copied()
                    .ok_or_else(|| anyhow!("Operation '{}' not found", name))
            }
            (_, None) => Err(anyhow!("Multiple operations found but no operation name specified")),
        }
    }

    async fn execute_selection_set(
        &self,
        selection_set: &SelectionSet,
        type_name: &str,
        context: &ExecutionContext,
        schema: &Schema,
        path: Vec<String>,
    ) -> Result<ExecutionResult> {
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

                    match self.execute_field(field, type_name, context, schema, field_path.clone()).await {
                        Ok(value) => {
                            let field_name = field.alias.as_ref().unwrap_or(&field.name);
                            result_data.insert(field_name.clone(), value);
                        }
                        Err(err) => {
                            errors.push(GraphQLError::new(err.to_string()).with_path(field_path));
                        }
                    }
                }
                Selection::InlineFragment(_) => {
                    // TODO: Implement inline fragment execution
                    errors.push(GraphQLError::new("Inline fragments not yet supported".to_string()));
                }
                Selection::FragmentSpread(_) => {
                    // TODO: Implement fragment spread execution
                    errors.push(GraphQLError::new("Fragment spreads not yet supported".to_string()));
                }
            }
        }

        let data = serde_json::to_value(result_data)
            .map_err(|e| anyhow!("Failed to serialize result data: {}", e))?;

        Ok(ExecutionResult {
            data: Some(data),
            errors,
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
        self.complete_value(&field_type, &field_value, &field.selection_set, context, schema, path).await
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
            Value::Variable(var) => {
                context.variables.get(&var.name)
                    .cloned()
                    .ok_or_else(|| anyhow!("Variable '{}' not defined", var.name))
            }
            _ => Ok(value.clone()),
        }
    }

    fn get_field_type<'a>(&self, parent_type: &str, field_name: &str, schema: &'a Schema) -> Result<&'a GraphQLType> {
        let parent_type_def = schema.get_type(parent_type)
            .ok_or_else(|| anyhow!("Type '{}' not found in schema", parent_type))?;

        match parent_type_def {
            GraphQLType::Object(obj) => {
                obj.fields.get(field_name)
                    .map(|field| &field.field_type)
                    .ok_or_else(|| anyhow!("Field '{}' not found on type '{}'", field_name, parent_type))
            }
            GraphQLType::Interface(iface) => {
                iface.fields.get(field_name)
                    .map(|field| &field.field_type)
                    .ok_or_else(|| anyhow!("Field '{}' not found on interface '{}'", field_name, parent_type))
            }
            _ => Err(anyhow!("Cannot select field '{}' on non-composite type '{}'", field_name, parent_type)),
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
                self.complete_value(inner_type, value, selection_set, context, schema, path).await
            }
            GraphQLType::List(inner_type) => {
                match value {
                    Value::ListValue(list) => {
                        let mut result = Vec::new();
                        for (i, item) in list.iter().enumerate() {
                            let item_path = {
                                let mut p = path.clone();
                                p.push(i.to_string());
                                p
                            };
                            let completed = self.complete_value(inner_type, item, selection_set, context, schema, item_path).await?;
                            result.push(completed);
                        }
                        Ok(JsonValue::Array(result))
                    }
                    Value::NullValue => Ok(JsonValue::Null),
                    _ => Err(anyhow!("Expected list value but got {:?}", value)),
                }
            }
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
                    let result = self.execute_selection_set(
                        selection_set,
                        field_type.name(),
                        context,
                        schema,
                        path,
                    ).await?;
                    
                    result.data.ok_or_else(|| anyhow!("No data returned from sub-selection"))
                } else {
                    Err(anyhow!("Selection set required for object/interface type"))
                }
            }
            GraphQLType::Union(_) => {
                // TODO: Implement union type completion
                Err(anyhow!("Union types not yet supported"))
            }
            GraphQLType::Enum(_) => {
                // TODO: Implement enum type completion
                self.serialize_scalar_value(value)
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
            Value::FloatValue(f) => {
                serde_json::Number::from_f64(*f)
                    .map(JsonValue::Number)
                    .ok_or_else(|| anyhow!("Invalid float value: {}", f))
            }
            Value::StringValue(s) => Ok(JsonValue::String(s.clone())),
            Value::BooleanValue(b) => Ok(JsonValue::Bool(*b)),
            Value::EnumValue(s) => Ok(JsonValue::String(s.clone())),
            Value::ListValue(list) => {
                let json_list: Result<Vec<JsonValue>> = list.iter()
                    .map(|v| self.serialize_scalar_value(v))
                    .collect();
                Ok(JsonValue::Array(json_list?))
            }
            Value::ObjectValue(obj) => {
                let json_obj: Result<serde_json::Map<String, JsonValue>> = obj.iter()
                    .map(|(k, v)| self.serialize_scalar_value(v).map(|json_v| (k.clone(), json_v)))
                    .collect();
                Ok(JsonValue::Object(json_obj?))
            }
            Value::Variable(_) => Err(anyhow!("Variables should be resolved before serialization")),
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
            _ => Ok(Value::NullValue),
        }
    }
}