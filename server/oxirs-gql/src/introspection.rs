//! GraphQL Schema Introspection System
//!
//! This module provides the complete GraphQL introspection system, allowing clients
//! to query the schema structure, types, fields, and directives at runtime.

use crate::ast::Value;
use crate::execution::{ExecutionContext, FieldResolver};
use crate::types::{
    ArgumentType, EnumType, EnumValue, FieldType, GraphQLType, InputObjectType, InterfaceType,
    ObjectType, ScalarType, Schema, UnionType,
};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;

/// The __Schema type for GraphQL introspection
#[derive(Debug, Clone)]
pub struct SchemaIntrospection {
    schema: Arc<Schema>,
}

impl SchemaIntrospection {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema }
    }

    /// Get all types in the schema
    pub fn get_types(&self) -> Vec<TypeIntrospection> {
        let mut types = Vec::new();

        for (_, gql_type) in &self.schema.types {
            types.push(TypeIntrospection::new(
                gql_type.clone(),
                Arc::clone(&self.schema),
            ));
        }

        // Add built-in scalar types
        types.extend(self.get_builtin_scalar_types());

        types
    }

    /// Get the query type
    pub fn get_query_type(&self) -> Option<TypeIntrospection> {
        self.schema
            .query_type
            .as_ref()
            .and_then(|type_name| self.schema.get_type(type_name))
            .map(|gql_type| TypeIntrospection::new(gql_type.clone(), Arc::clone(&self.schema)))
    }

    /// Get the mutation type
    pub fn get_mutation_type(&self) -> Option<TypeIntrospection> {
        self.schema
            .mutation_type
            .as_ref()
            .and_then(|type_name| self.schema.get_type(type_name))
            .map(|gql_type| TypeIntrospection::new(gql_type.clone(), Arc::clone(&self.schema)))
    }

    /// Get the subscription type
    pub fn get_subscription_type(&self) -> Option<TypeIntrospection> {
        self.schema
            .subscription_type
            .as_ref()
            .and_then(|type_name| self.schema.get_type(type_name))
            .map(|gql_type| TypeIntrospection::new(gql_type.clone(), Arc::clone(&self.schema)))
    }

    /// Get schema directives
    pub fn get_directives(&self) -> Vec<DirectiveIntrospection> {
        vec![
            DirectiveIntrospection::deprecated(),
            DirectiveIntrospection::skip(),
            DirectiveIntrospection::include(),
            DirectiveIntrospection::specified_by(),
        ]
    }

    fn get_builtin_scalar_types(&self) -> Vec<TypeIntrospection> {
        vec![
            TypeIntrospection::builtin_scalar(
                "String",
                "The `String` scalar type represents textual data",
            ),
            TypeIntrospection::builtin_scalar(
                "Int",
                "The `Int` scalar type represents non-fractional signed whole numeric values",
            ),
            TypeIntrospection::builtin_scalar(
                "Float",
                "The `Float` scalar type represents signed double-precision fractional values",
            ),
            TypeIntrospection::builtin_scalar(
                "Boolean",
                "The `Boolean` scalar type represents `true` or `false`",
            ),
            TypeIntrospection::builtin_scalar(
                "ID",
                "The `ID` scalar type represents a unique identifier",
            ),
            // RDF-specific scalars
            TypeIntrospection::builtin_scalar("IRI", "Internationalized Resource Identifier"),
            TypeIntrospection::builtin_scalar("Literal", "RDF Literal value"),
            TypeIntrospection::builtin_scalar("DateTime", "ISO 8601 date and time"),
            TypeIntrospection::builtin_scalar("Duration", "ISO 8601 duration"),
            TypeIntrospection::builtin_scalar("GeoLocation", "Geographic coordinates"),
            TypeIntrospection::builtin_scalar("LangString", "Language-tagged string"),
        ]
    }
}

/// The __Type type for GraphQL introspection
#[derive(Debug, Clone)]
pub struct TypeIntrospection {
    gql_type: GraphQLType,
    schema: Arc<Schema>,
}

impl TypeIntrospection {
    pub fn new(gql_type: GraphQLType, schema: Arc<Schema>) -> Self {
        Self { gql_type, schema }
    }

    pub fn builtin_scalar(name: &str, description: &str) -> Self {
        let scalar = ScalarType::new(name.to_string()).with_description(description.to_string());

        Self {
            gql_type: GraphQLType::Scalar(scalar),
            schema: Arc::new(Schema::new()), // Empty schema for built-ins
        }
    }

    /// Get the type kind
    pub fn kind(&self) -> TypeKind {
        match &self.gql_type {
            GraphQLType::Scalar(_) => TypeKind::Scalar,
            GraphQLType::Object(_) => TypeKind::Object,
            GraphQLType::Interface(_) => TypeKind::Interface,
            GraphQLType::Union(_) => TypeKind::Union,
            GraphQLType::Enum(_) => TypeKind::Enum,
            GraphQLType::InputObject(_) => TypeKind::InputObject,
            GraphQLType::List(_) => TypeKind::List,
            GraphQLType::NonNull(_) => TypeKind::NonNull,
        }
    }

    /// Get the type name
    pub fn name(&self) -> Option<String> {
        match &self.gql_type {
            GraphQLType::Scalar(s) => Some(s.name.clone()),
            GraphQLType::Object(o) => Some(o.name.clone()),
            GraphQLType::Interface(i) => Some(i.name.clone()),
            GraphQLType::Union(u) => Some(u.name.clone()),
            GraphQLType::Enum(e) => Some(e.name.clone()),
            GraphQLType::InputObject(io) => Some(io.name.clone()),
            GraphQLType::List(_) => None,
            GraphQLType::NonNull(_) => None,
        }
    }

    /// Get the type description
    pub fn description(&self) -> Option<String> {
        match &self.gql_type {
            GraphQLType::Scalar(s) => s.description.clone(),
            GraphQLType::Object(o) => o.description.clone(),
            GraphQLType::Interface(i) => i.description.clone(),
            GraphQLType::Union(u) => u.description.clone(),
            GraphQLType::Enum(e) => e.description.clone(),
            GraphQLType::InputObject(io) => io.description.clone(),
            GraphQLType::List(_) => None,
            GraphQLType::NonNull(_) => None,
        }
    }

    /// Get fields (for Object and Interface types)
    pub fn fields(&self, include_deprecated: bool) -> Option<Vec<FieldIntrospection>> {
        match &self.gql_type {
            GraphQLType::Object(obj) => {
                let mut fields: Vec<FieldIntrospection> = obj
                    .fields
                    .values()
                    .filter(|field| include_deprecated || true) // No deprecation support yet
                    .map(|field| FieldIntrospection::new(field.clone(), Arc::clone(&self.schema)))
                    .collect();

                // Add introspection fields for root types
                if self.is_query_type() {
                    fields.extend(self.get_introspection_fields());
                }

                Some(fields)
            }
            GraphQLType::Interface(iface) => {
                Some(
                    iface
                        .fields
                        .values()
                        .filter(|field| include_deprecated || true) // No deprecation support yet
                        .map(|field| {
                            FieldIntrospection::new(field.clone(), Arc::clone(&self.schema))
                        })
                        .collect(),
                )
            }
            _ => None,
        }
    }

    /// Get interfaces implemented by this type (for Object types)
    pub fn interfaces(&self) -> Option<Vec<TypeIntrospection>> {
        match &self.gql_type {
            GraphQLType::Object(obj) => Some(
                obj.interfaces
                    .iter()
                    .filter_map(|interface_name| {
                        self.schema.get_type(interface_name).map(|gql_type| {
                            TypeIntrospection::new(gql_type.clone(), Arc::clone(&self.schema))
                        })
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Get possible types (for Interface and Union types)
    pub fn possible_types(&self) -> Option<Vec<TypeIntrospection>> {
        match &self.gql_type {
            GraphQLType::Interface(iface) => Some(
                self.schema
                    .types
                    .values()
                    .filter_map(|gql_type| match gql_type {
                        GraphQLType::Object(obj) if obj.interfaces.contains(&iface.name) => Some(
                            TypeIntrospection::new(gql_type.clone(), Arc::clone(&self.schema)),
                        ),
                        _ => None,
                    })
                    .collect(),
            ),
            GraphQLType::Union(union) => Some(
                union
                    .types
                    .iter()
                    .filter_map(|type_name| {
                        self.schema.get_type(type_name).map(|gql_type| {
                            TypeIntrospection::new(gql_type.clone(), Arc::clone(&self.schema))
                        })
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Get enum values (for Enum types)
    pub fn enum_values(&self, include_deprecated: bool) -> Option<Vec<EnumValueIntrospection>> {
        match &self.gql_type {
            GraphQLType::Enum(enum_type) => Some(
                enum_type
                    .values
                    .values()
                    .filter(|value| include_deprecated || value.deprecated.is_none())
                    .map(|value| EnumValueIntrospection::new(value.clone()))
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Get input fields (for InputObject types)
    pub fn input_fields(&self) -> Option<Vec<InputValueIntrospection>> {
        match &self.gql_type {
            GraphQLType::InputObject(input_obj) => {
                Some(
                    input_obj
                        .fields
                        .values()
                        .map(|field| {
                            // Convert InputFieldType to ArgumentType for compatibility
                            let arg =
                                ArgumentType::new(field.name.clone(), field.field_type.clone())
                                    .with_description(field.description.clone().unwrap_or_default())
                                    .with_default_value(
                                        field
                                            .default_value
                                            .clone()
                                            .unwrap_or(crate::ast::Value::NullValue),
                                    );
                            InputValueIntrospection::new(arg, Arc::clone(&self.schema))
                        })
                        .collect(),
                )
            }
            _ => None,
        }
    }

    /// Get the inner type (for List and NonNull types)
    pub fn of_type(&self) -> Option<TypeIntrospection> {
        match &self.gql_type {
            GraphQLType::List(inner_type) => Some(TypeIntrospection::new(
                *inner_type.clone(),
                Arc::clone(&self.schema),
            )),
            GraphQLType::NonNull(inner_type) => Some(TypeIntrospection::new(
                *inner_type.clone(),
                Arc::clone(&self.schema),
            )),
            _ => None,
        }
    }

    /// Get specified by URL (for Scalar types)
    pub fn specified_by_url(&self) -> Option<String> {
        match &self.gql_type {
            GraphQLType::Scalar(_scalar) => None, // Field not available in current ScalarType
            _ => None,
        }
    }

    fn is_query_type(&self) -> bool {
        if let Some(query_type_name) = &self.schema.query_type {
            self.name().as_ref() == Some(query_type_name)
        } else {
            false
        }
    }

    fn get_introspection_fields(&self) -> Vec<FieldIntrospection> {
        vec![
            FieldIntrospection::introspection_field(
                "__schema",
                "__Schema!",
                "Access the current type schema of this server.",
            ),
            FieldIntrospection::introspection_field(
                "__type",
                "__Type",
                "Request the type information of a single type.",
            ),
        ]
    }
}

/// Type kinds for GraphQL introspection
#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    Scalar,
    Object,
    Interface,
    Union,
    Enum,
    InputObject,
    List,
    NonNull,
}

impl TypeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            TypeKind::Scalar => "SCALAR",
            TypeKind::Object => "OBJECT",
            TypeKind::Interface => "INTERFACE",
            TypeKind::Union => "UNION",
            TypeKind::Enum => "ENUM",
            TypeKind::InputObject => "INPUT_OBJECT",
            TypeKind::List => "LIST",
            TypeKind::NonNull => "NON_NULL",
        }
    }
}

/// The __Field type for GraphQL introspection
#[derive(Debug, Clone)]
pub struct FieldIntrospection {
    field: FieldType,
    schema: Arc<Schema>,
}

impl FieldIntrospection {
    pub fn new(field: FieldType, schema: Arc<Schema>) -> Self {
        Self { field, schema }
    }

    pub fn introspection_field(name: &str, type_name: &str, description: &str) -> Self {
        let field = FieldType::new(
            name.to_string(),
            GraphQLType::Scalar(
                ScalarType::new(type_name.to_string()).with_description(description.to_string()),
            ),
        )
        .with_description(description.to_string());

        Self {
            field,
            schema: Arc::new(Schema::new()),
        }
    }

    pub fn name(&self) -> String {
        self.field.name.clone()
    }

    pub fn description(&self) -> Option<String> {
        self.field.description.clone()
    }

    pub fn args(&self) -> Vec<InputValueIntrospection> {
        self.field
            .arguments
            .values()
            .map(|arg| InputValueIntrospection::new(arg.clone(), Arc::clone(&self.schema)))
            .collect()
    }

    pub fn field_type(&self) -> TypeIntrospection {
        TypeIntrospection::new(self.field.field_type.clone(), Arc::clone(&self.schema))
    }

    pub fn is_deprecated(&self) -> bool {
        false // Deprecation not supported in current FieldType
    }

    pub fn deprecation_reason(&self) -> Option<String> {
        None // Deprecation not supported in current FieldType
    }
}

/// The __InputValue type for GraphQL introspection
#[derive(Debug, Clone)]
pub struct InputValueIntrospection {
    arg: ArgumentType,
    schema: Arc<Schema>,
}

impl InputValueIntrospection {
    pub fn new(arg: ArgumentType, schema: Arc<Schema>) -> Self {
        Self { arg, schema }
    }

    pub fn name(&self) -> String {
        self.arg.name.clone()
    }

    pub fn description(&self) -> Option<String> {
        self.arg.description.clone()
    }

    pub fn input_type(&self) -> TypeIntrospection {
        TypeIntrospection::new(self.arg.argument_type.clone(), Arc::clone(&self.schema))
    }

    pub fn default_value(&self) -> Option<String> {
        self.arg.default_value.as_ref().map(|value| {
            // Serialize the value to a GraphQL string representation
            match value {
                Value::StringValue(s) => format!("\"{}\"", s),
                Value::IntValue(i) => i.to_string(),
                Value::FloatValue(f) => f.to_string(),
                Value::BooleanValue(b) => b.to_string(),
                Value::NullValue => "null".to_string(),
                Value::EnumValue(e) => e.clone(),
                Value::ListValue(list) => {
                    let items: Vec<String> = list
                        .iter()
                        .map(|v| self.serialize_value_string(v))
                        .collect();
                    format!("[{}]", items.join(", "))
                }
                Value::ObjectValue(obj) => {
                    let fields: Vec<String> = obj
                        .iter()
                        .map(|(k, v)| format!("{}: {}", k, self.serialize_value_string(v)))
                        .collect();
                    format!("{{{}}}", fields.join(", "))
                }
                Value::Variable(var) => format!("${}", var.name),
            }
        })
    }

    fn serialize_value_string(&self, value: &Value) -> String {
        match value {
            Value::StringValue(s) => format!("\"{}\"", s),
            Value::IntValue(i) => i.to_string(),
            Value::FloatValue(f) => f.to_string(),
            Value::BooleanValue(b) => b.to_string(),
            Value::NullValue => "null".to_string(),
            Value::EnumValue(e) => e.clone(),
            _ => "null".to_string(), // Simplified for nested cases
        }
    }
}

/// The __EnumValue type for GraphQL introspection
#[derive(Debug, Clone)]
pub struct EnumValueIntrospection {
    value: EnumValue,
}

impl EnumValueIntrospection {
    pub fn new(value: EnumValue) -> Self {
        Self { value }
    }

    pub fn name(&self) -> String {
        self.value.name.clone()
    }

    pub fn description(&self) -> Option<String> {
        self.value.description.clone()
    }

    pub fn is_deprecated(&self) -> bool {
        self.value.deprecated.is_some()
    }

    pub fn deprecation_reason(&self) -> Option<String> {
        self.value.deprecated.clone()
    }
}

/// The __Directive type for GraphQL introspection
#[derive(Debug, Clone)]
pub struct DirectiveIntrospection {
    name: String,
    description: Option<String>,
    locations: Vec<DirectiveLocation>,
    args: Vec<InputValueIntrospection>,
    is_repeatable: bool,
}

impl DirectiveIntrospection {
    pub fn new(
        name: String,
        description: Option<String>,
        locations: Vec<DirectiveLocation>,
        args: Vec<InputValueIntrospection>,
        is_repeatable: bool,
    ) -> Self {
        Self {
            name,
            description,
            locations,
            args,
            is_repeatable,
        }
    }

    pub fn deprecated() -> Self {
        Self::new(
            "deprecated".to_string(),
            Some("Marks an element of a GraphQL schema as no longer supported.".to_string()),
            vec![
                DirectiveLocation::FieldDefinition,
                DirectiveLocation::ArgumentDefinition,
                DirectiveLocation::InputFieldDefinition,
                DirectiveLocation::EnumValue,
            ],
            vec![InputValueIntrospection::new(
                ArgumentType::new(
                    "reason".to_string(),
                    GraphQLType::Scalar(
                        ScalarType::new("String".to_string())
                            .with_description("Reason for deprecation".to_string()),
                    ),
                )
                .with_default_value(Value::StringValue("No longer supported".to_string())),
                Arc::new(Schema::new()),
            )],
            false,
        )
    }

    pub fn skip() -> Self {
        Self::new(
            "skip".to_string(),
            Some("Directs the executor to skip this field or fragment when the `if` argument is true.".to_string()),
            vec![
                DirectiveLocation::Field,
                DirectiveLocation::FragmentSpread,
                DirectiveLocation::InlineFragment,
            ],
            vec![InputValueIntrospection::new(
                ArgumentType::new(
                    "if".to_string(),
                    GraphQLType::NonNull(Box::new(GraphQLType::Scalar(ScalarType::new("Boolean".to_string())
                        .with_description("Skip condition".to_string())))),
                ),
                Arc::new(Schema::new()),
            )],
            false,
        )
    }

    pub fn include() -> Self {
        Self::new(
            "include".to_string(),
            Some("Directs the executor to include this field or fragment only when the `if` argument is true.".to_string()),
            vec![
                DirectiveLocation::Field,
                DirectiveLocation::FragmentSpread,
                DirectiveLocation::InlineFragment,
            ],
            vec![InputValueIntrospection::new(
                ArgumentType::new(
                    "if".to_string(),
                    GraphQLType::NonNull(Box::new(GraphQLType::Scalar(ScalarType::new("Boolean".to_string())
                        .with_description("Include condition".to_string())))),
                ),
                Arc::new(Schema::new()),
            )],
            false,
        )
    }

    pub fn specified_by() -> Self {
        Self::new(
            "specifiedBy".to_string(),
            Some("Exposes a URL that specifies the behaviour of this scalar.".to_string()),
            vec![DirectiveLocation::Scalar],
            vec![InputValueIntrospection::new(
                ArgumentType::new(
                    "url".to_string(),
                    GraphQLType::NonNull(Box::new(GraphQLType::Scalar(
                        ScalarType::new("String".to_string())
                            .with_description("Specification URL".to_string()),
                    ))),
                ),
                Arc::new(Schema::new()),
            )],
            false,
        )
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn description(&self) -> Option<String> {
        self.description.clone()
    }

    pub fn locations(&self) -> Vec<DirectiveLocation> {
        self.locations.clone()
    }

    pub fn args(&self) -> Vec<InputValueIntrospection> {
        self.args.clone()
    }

    pub fn is_repeatable(&self) -> bool {
        self.is_repeatable
    }
}

/// Directive locations for GraphQL introspection
#[derive(Debug, Clone, PartialEq)]
pub enum DirectiveLocation {
    // Executable directive locations
    Query,
    Mutation,
    Subscription,
    Field,
    FragmentDefinition,
    FragmentSpread,
    InlineFragment,
    VariableDefinition,
    // Type system directive locations
    Schema,
    Scalar,
    Object,
    FieldDefinition,
    ArgumentDefinition,
    Interface,
    Union,
    Enum,
    EnumValue,
    InputObject,
    InputFieldDefinition,
}

impl DirectiveLocation {
    pub fn as_str(&self) -> &'static str {
        match self {
            DirectiveLocation::Query => "QUERY",
            DirectiveLocation::Mutation => "MUTATION",
            DirectiveLocation::Subscription => "SUBSCRIPTION",
            DirectiveLocation::Field => "FIELD",
            DirectiveLocation::FragmentDefinition => "FRAGMENT_DEFINITION",
            DirectiveLocation::FragmentSpread => "FRAGMENT_SPREAD",
            DirectiveLocation::InlineFragment => "INLINE_FRAGMENT",
            DirectiveLocation::VariableDefinition => "VARIABLE_DEFINITION",
            DirectiveLocation::Schema => "SCHEMA",
            DirectiveLocation::Scalar => "SCALAR",
            DirectiveLocation::Object => "OBJECT",
            DirectiveLocation::FieldDefinition => "FIELD_DEFINITION",
            DirectiveLocation::ArgumentDefinition => "ARGUMENT_DEFINITION",
            DirectiveLocation::Interface => "INTERFACE",
            DirectiveLocation::Union => "UNION",
            DirectiveLocation::Enum => "ENUM",
            DirectiveLocation::EnumValue => "ENUM_VALUE",
            DirectiveLocation::InputObject => "INPUT_OBJECT",
            DirectiveLocation::InputFieldDefinition => "INPUT_FIELD_DEFINITION",
        }
    }
}

/// Introspection resolver that handles __schema and __type queries
pub struct IntrospectionResolver {
    schema: Arc<Schema>,
}

impl IntrospectionResolver {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema }
    }
}

#[async_trait]
impl FieldResolver for IntrospectionResolver {
    async fn resolve_field(
        &self,
        field_name: &str,
        args: &HashMap<String, Value>,
        context: &ExecutionContext,
    ) -> Result<Value> {
        match field_name {
            "__schema" => Ok(self.resolve_schema().await?),
            "__type" => {
                let type_name = args
                    .get("name")
                    .ok_or_else(|| anyhow!("__type field requires 'name' argument"))?;

                if let Value::StringValue(name) = type_name {
                    Ok(self.resolve_type(name).await?)
                } else {
                    Err(anyhow!("__type 'name' argument must be a string"))
                }
            }
            "queryType" => Ok(self.resolve_query_type().await?),
            "mutationType" => Ok(self.resolve_mutation_type().await?),
            "subscriptionType" => Ok(self.resolve_subscription_type().await?),
            "types" => Ok(self.resolve_types().await?),
            "directives" => Ok(self.resolve_directives().await?),
            _ => Err(anyhow!("Unknown introspection field: {}", field_name)),
        }
    }
}

impl IntrospectionResolver {
    async fn resolve_schema(&self) -> Result<Value> {
        let schema_introspection = SchemaIntrospection::new(Arc::clone(&self.schema));

        // This would be a complex object with all schema information
        // For now, return a simplified representation
        Ok(Value::ObjectValue(HashMap::from([
            ("queryType".to_string(), self.resolve_query_type().await?),
            (
                "mutationType".to_string(),
                self.resolve_mutation_type().await?,
            ),
            (
                "subscriptionType".to_string(),
                self.resolve_subscription_type().await?,
            ),
            ("types".to_string(), self.resolve_types().await?),
            ("directives".to_string(), self.resolve_directives().await?),
        ])))
    }

    async fn resolve_type(&self, type_name: &str) -> Result<Value> {
        if let Some(gql_type) = self.schema.get_type(type_name) {
            let type_introspection =
                TypeIntrospection::new(gql_type.clone(), Arc::clone(&self.schema));
            Ok(self.type_to_value(&type_introspection))
        } else {
            // Check built-in types
            let builtin_types =
                SchemaIntrospection::new(Arc::clone(&self.schema)).get_builtin_scalar_types();
            for builtin_type in builtin_types {
                if builtin_type.name().as_ref() == Some(&type_name.to_string()) {
                    return Ok(self.type_to_value(&builtin_type));
                }
            }
            Ok(Value::NullValue)
        }
    }

    async fn resolve_query_type(&self) -> Result<Value> {
        if let Some(query_type) =
            SchemaIntrospection::new(Arc::clone(&self.schema)).get_query_type()
        {
            Ok(self.type_to_value(&query_type))
        } else {
            Ok(Value::NullValue)
        }
    }

    async fn resolve_mutation_type(&self) -> Result<Value> {
        if let Some(mutation_type) =
            SchemaIntrospection::new(Arc::clone(&self.schema)).get_mutation_type()
        {
            Ok(self.type_to_value(&mutation_type))
        } else {
            Ok(Value::NullValue)
        }
    }

    async fn resolve_subscription_type(&self) -> Result<Value> {
        if let Some(subscription_type) =
            SchemaIntrospection::new(Arc::clone(&self.schema)).get_subscription_type()
        {
            Ok(self.type_to_value(&subscription_type))
        } else {
            Ok(Value::NullValue)
        }
    }

    async fn resolve_types(&self) -> Result<Value> {
        let schema_introspection = SchemaIntrospection::new(Arc::clone(&self.schema));
        let types = schema_introspection.get_types();

        let type_values: Vec<Value> = types
            .iter()
            .map(|type_info| self.type_to_value(type_info))
            .collect();

        Ok(Value::ListValue(type_values))
    }

    async fn resolve_directives(&self) -> Result<Value> {
        let schema_introspection = SchemaIntrospection::new(Arc::clone(&self.schema));
        let directives = schema_introspection.get_directives();

        let directive_values: Vec<Value> = directives
            .iter()
            .map(|directive| self.directive_to_value(directive))
            .collect();

        Ok(Value::ListValue(directive_values))
    }

    fn type_to_value(&self, type_info: &TypeIntrospection) -> Value {
        let mut obj = HashMap::new();

        obj.insert(
            "kind".to_string(),
            Value::EnumValue(type_info.kind().as_str().to_string()),
        );

        if let Some(name) = type_info.name() {
            obj.insert("name".to_string(), Value::StringValue(name));
        } else {
            obj.insert("name".to_string(), Value::NullValue);
        }

        if let Some(description) = type_info.description() {
            obj.insert("description".to_string(), Value::StringValue(description));
        } else {
            obj.insert("description".to_string(), Value::NullValue);
        }

        // Add fields for object and interface types
        if let Some(fields) = type_info.fields(false) {
            let field_values: Vec<Value> = fields
                .iter()
                .map(|field| self.field_to_value(field))
                .collect();
            obj.insert("fields".to_string(), Value::ListValue(field_values));
        } else {
            obj.insert("fields".to_string(), Value::NullValue);
        }

        // Add interfaces for object types
        if let Some(interfaces) = type_info.interfaces() {
            let interface_values: Vec<Value> = interfaces
                .iter()
                .map(|interface| self.type_to_value(interface))
                .collect();
            obj.insert("interfaces".to_string(), Value::ListValue(interface_values));
        } else {
            obj.insert("interfaces".to_string(), Value::NullValue);
        }

        // Add possible types for union and interface types
        if let Some(possible_types) = type_info.possible_types() {
            let possible_type_values: Vec<Value> = possible_types
                .iter()
                .map(|possible_type| self.type_to_value(possible_type))
                .collect();
            obj.insert(
                "possibleTypes".to_string(),
                Value::ListValue(possible_type_values),
            );
        } else {
            obj.insert("possibleTypes".to_string(), Value::NullValue);
        }

        // Add enum values for enum types
        if let Some(enum_values) = type_info.enum_values(false) {
            let enum_value_values: Vec<Value> = enum_values
                .iter()
                .map(|enum_value| self.enum_value_to_value(enum_value))
                .collect();
            obj.insert(
                "enumValues".to_string(),
                Value::ListValue(enum_value_values),
            );
        } else {
            obj.insert("enumValues".to_string(), Value::NullValue);
        }

        // Add input fields for input object types
        if let Some(input_fields) = type_info.input_fields() {
            let input_field_values: Vec<Value> = input_fields
                .iter()
                .map(|input_field| self.input_value_to_value(input_field))
                .collect();
            obj.insert(
                "inputFields".to_string(),
                Value::ListValue(input_field_values),
            );
        } else {
            obj.insert("inputFields".to_string(), Value::NullValue);
        }

        // Add ofType for list and non-null types
        if let Some(of_type) = type_info.of_type() {
            obj.insert("ofType".to_string(), self.type_to_value(&of_type));
        } else {
            obj.insert("ofType".to_string(), Value::NullValue);
        }

        // Add specifiedByURL for scalar types
        if let Some(specified_by_url) = type_info.specified_by_url() {
            obj.insert(
                "specifiedByURL".to_string(),
                Value::StringValue(specified_by_url),
            );
        } else {
            obj.insert("specifiedByURL".to_string(), Value::NullValue);
        }

        Value::ObjectValue(obj)
    }

    fn field_to_value(&self, field: &FieldIntrospection) -> Value {
        let mut obj = HashMap::new();

        obj.insert("name".to_string(), Value::StringValue(field.name()));

        if let Some(description) = field.description() {
            obj.insert("description".to_string(), Value::StringValue(description));
        } else {
            obj.insert("description".to_string(), Value::NullValue);
        }

        let args: Vec<Value> = field
            .args()
            .iter()
            .map(|arg| self.input_value_to_value(arg))
            .collect();
        obj.insert("args".to_string(), Value::ListValue(args));

        obj.insert("type".to_string(), self.type_to_value(&field.field_type()));
        obj.insert(
            "isDeprecated".to_string(),
            Value::BooleanValue(field.is_deprecated()),
        );

        if let Some(deprecation_reason) = field.deprecation_reason() {
            obj.insert(
                "deprecationReason".to_string(),
                Value::StringValue(deprecation_reason),
            );
        } else {
            obj.insert("deprecationReason".to_string(), Value::NullValue);
        }

        Value::ObjectValue(obj)
    }

    fn input_value_to_value(&self, input_value: &InputValueIntrospection) -> Value {
        let mut obj = HashMap::new();

        obj.insert("name".to_string(), Value::StringValue(input_value.name()));

        if let Some(description) = input_value.description() {
            obj.insert("description".to_string(), Value::StringValue(description));
        } else {
            obj.insert("description".to_string(), Value::NullValue);
        }

        obj.insert(
            "type".to_string(),
            self.type_to_value(&input_value.input_type()),
        );

        if let Some(default_value) = input_value.default_value() {
            obj.insert(
                "defaultValue".to_string(),
                Value::StringValue(default_value),
            );
        } else {
            obj.insert("defaultValue".to_string(), Value::NullValue);
        }

        Value::ObjectValue(obj)
    }

    fn enum_value_to_value(&self, enum_value: &EnumValueIntrospection) -> Value {
        let mut obj = HashMap::new();

        obj.insert("name".to_string(), Value::StringValue(enum_value.name()));

        if let Some(description) = enum_value.description() {
            obj.insert("description".to_string(), Value::StringValue(description));
        } else {
            obj.insert("description".to_string(), Value::NullValue);
        }

        obj.insert(
            "isDeprecated".to_string(),
            Value::BooleanValue(enum_value.is_deprecated()),
        );

        if let Some(deprecation_reason) = enum_value.deprecation_reason() {
            obj.insert(
                "deprecationReason".to_string(),
                Value::StringValue(deprecation_reason),
            );
        } else {
            obj.insert("deprecationReason".to_string(), Value::NullValue);
        }

        Value::ObjectValue(obj)
    }

    fn directive_to_value(&self, directive: &DirectiveIntrospection) -> Value {
        let mut obj = HashMap::new();

        obj.insert("name".to_string(), Value::StringValue(directive.name()));

        if let Some(description) = directive.description() {
            obj.insert("description".to_string(), Value::StringValue(description));
        } else {
            obj.insert("description".to_string(), Value::NullValue);
        }

        let locations: Vec<Value> = directive
            .locations()
            .iter()
            .map(|location| Value::EnumValue(location.as_str().to_string()))
            .collect();
        obj.insert("locations".to_string(), Value::ListValue(locations));

        let args: Vec<Value> = directive
            .args()
            .iter()
            .map(|arg| self.input_value_to_value(arg))
            .collect();
        obj.insert("args".to_string(), Value::ListValue(args));

        obj.insert(
            "isRepeatable".to_string(),
            Value::BooleanValue(directive.is_repeatable()),
        );

        Value::ObjectValue(obj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BuiltinScalars;

    fn create_test_schema() -> Schema {
        let mut schema = Schema::new();

        let query_type = ObjectType::new("Query".to_string())
            .with_description("The root query type".to_string())
            .with_field(
                "hello".to_string(),
                FieldType::new(
                    "hello".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::string()),
                )
                .with_description("A simple greeting".to_string()),
            );

        schema.add_type(GraphQLType::Object(query_type));
        schema.set_query_type("Query".to_string());

        schema
    }

    #[test]
    fn test_schema_introspection() {
        let schema = create_test_schema();
        let introspection = SchemaIntrospection::new(Arc::new(schema));

        assert!(introspection.get_query_type().is_some());
        assert!(introspection.get_mutation_type().is_none());
        assert!(introspection.get_subscription_type().is_none());

        let types = introspection.get_types();
        assert!(!types.is_empty());

        let directives = introspection.get_directives();
        assert_eq!(directives.len(), 4); // deprecated, skip, include, specifiedBy
    }

    #[test]
    fn test_type_introspection() {
        let schema = create_test_schema();
        let query_type = schema.get_type("Query").unwrap();
        let type_introspection = TypeIntrospection::new(query_type.clone(), Arc::new(schema));

        assert_eq!(type_introspection.kind(), TypeKind::Object);
        assert_eq!(type_introspection.name(), Some("Query".to_string()));
        assert!(type_introspection.description().is_some());

        let fields = type_introspection.fields(false);
        assert!(fields.is_some());
        assert!(!fields.unwrap().is_empty());
    }

    #[test]
    fn test_type_kind_string_conversion() {
        assert_eq!(TypeKind::Scalar.as_str(), "SCALAR");
        assert_eq!(TypeKind::Object.as_str(), "OBJECT");
        assert_eq!(TypeKind::Interface.as_str(), "INTERFACE");
        assert_eq!(TypeKind::Union.as_str(), "UNION");
        assert_eq!(TypeKind::Enum.as_str(), "ENUM");
        assert_eq!(TypeKind::InputObject.as_str(), "INPUT_OBJECT");
        assert_eq!(TypeKind::List.as_str(), "LIST");
        assert_eq!(TypeKind::NonNull.as_str(), "NON_NULL");
    }

    #[test]
    fn test_directive_location_string_conversion() {
        assert_eq!(DirectiveLocation::Query.as_str(), "QUERY");
        assert_eq!(DirectiveLocation::Field.as_str(), "FIELD");
        assert_eq!(
            DirectiveLocation::FragmentSpread.as_str(),
            "FRAGMENT_SPREAD"
        );
        assert_eq!(
            DirectiveLocation::InlineFragment.as_str(),
            "INLINE_FRAGMENT"
        );
    }

    #[test]
    fn test_builtin_directive_creation() {
        let deprecated = DirectiveIntrospection::deprecated();
        assert_eq!(deprecated.name(), "deprecated");
        assert!(deprecated.description().is_some());
        assert!(!deprecated.args().is_empty());

        let skip = DirectiveIntrospection::skip();
        assert_eq!(skip.name(), "skip");
        assert!(!skip.args().is_empty());

        let include = DirectiveIntrospection::include();
        assert_eq!(include.name(), "include");
        assert!(!include.args().is_empty());
    }

    #[tokio::test]
    async fn test_introspection_resolver() {
        let schema = create_test_schema();
        let resolver = IntrospectionResolver::new(Arc::new(schema));
        let context = ExecutionContext::new();

        // Test __schema field
        let schema_result = resolver
            .resolve_field("__schema", &HashMap::new(), &context)
            .await;
        assert!(schema_result.is_ok());

        // Test __type field
        let mut args = HashMap::new();
        args.insert("name".to_string(), Value::StringValue("Query".to_string()));
        let type_result = resolver.resolve_field("__type", &args, &context).await;
        assert!(type_result.is_ok());

        // Test with non-existent type
        let mut args = HashMap::new();
        args.insert(
            "name".to_string(),
            Value::StringValue("NonExistent".to_string()),
        );
        let type_result = resolver.resolve_field("__type", &args, &context).await;
        assert!(type_result.is_ok());
        assert!(matches!(type_result.unwrap(), Value::NullValue));
    }

    #[test]
    fn test_field_introspection() {
        let field = FieldType::new(
            "test".to_string(),
            GraphQLType::Scalar(BuiltinScalars::string()),
        )
        .with_description("Test field".to_string());

        let field_introspection = FieldIntrospection::new(field, Arc::new(create_test_schema()));

        assert_eq!(field_introspection.name(), "test");
        assert_eq!(
            field_introspection.description(),
            Some("Test field".to_string())
        );
        assert!(!field_introspection.is_deprecated());
        assert!(field_introspection.deprecation_reason().is_none());
    }

    #[test]
    fn test_input_value_introspection() {
        let arg = ArgumentType::new(
            "test".to_string(),
            GraphQLType::Scalar(BuiltinScalars::string()),
        )
        .with_description("Test argument".to_string())
        .with_default_value(Value::StringValue("default".to_string()));

        let input_value_introspection =
            InputValueIntrospection::new(arg, Arc::new(create_test_schema()));

        assert_eq!(input_value_introspection.name(), "test");
        assert_eq!(
            input_value_introspection.description(),
            Some("Test argument".to_string())
        );
        assert_eq!(
            input_value_introspection.default_value(),
            Some("\"default\"".to_string())
        );
    }
}
