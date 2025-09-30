//! GraphQL type system implementation
//!
//! This module provides the GraphQL type system including scalar types,
//! object types, interfaces, unions, and enums.

use crate::ast::Value;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fmt;

/// GraphQL type information
#[derive(Debug, Clone, PartialEq)]
pub enum GraphQLType {
    Scalar(ScalarType),
    Object(ObjectType),
    Interface(InterfaceType),
    Union(UnionType),
    Enum(EnumType),
    InputObject(InputObjectType),
    List(Box<GraphQLType>),
    NonNull(Box<GraphQLType>),
}

impl GraphQLType {
    pub fn name(&self) -> &str {
        match self {
            GraphQLType::Scalar(s) => &s.name,
            GraphQLType::Object(o) => &o.name,
            GraphQLType::Interface(i) => &i.name,
            GraphQLType::Union(u) => &u.name,
            GraphQLType::Enum(e) => &e.name,
            GraphQLType::InputObject(io) => &io.name,
            GraphQLType::List(inner) => inner.name(),
            GraphQLType::NonNull(inner) => inner.name(),
        }
    }

    pub fn is_nullable(&self) -> bool {
        !matches!(self, GraphQLType::NonNull(_))
    }

    pub fn is_list(&self) -> bool {
        matches!(self, GraphQLType::List(_))
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self, GraphQLType::Scalar(_))
    }

    pub fn is_object(&self) -> bool {
        matches!(self, GraphQLType::Object(_))
    }
}

impl fmt::Display for GraphQLType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphQLType::Scalar(s) => write!(f, "{}", s.name),
            GraphQLType::Object(o) => write!(f, "{}", o.name),
            GraphQLType::Interface(i) => write!(f, "{}", i.name),
            GraphQLType::Union(u) => write!(f, "{}", u.name),
            GraphQLType::Enum(e) => write!(f, "{}", e.name),
            GraphQLType::InputObject(io) => write!(f, "{}", io.name),
            GraphQLType::List(inner) => write!(f, "[{inner}]"),
            GraphQLType::NonNull(inner) => write!(f, "{inner}!"),
        }
    }
}

/// Scalar type definition
#[derive(Debug, Clone)]
pub struct ScalarType {
    pub name: String,
    pub description: Option<String>,
    pub serialize: fn(&Value) -> Result<Value>,
    pub parse_value: fn(&Value) -> Result<Value>,
    pub parse_literal: fn(&Value) -> Result<Value>,
}

impl PartialEq for ScalarType {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.description == other.description
        // Note: Function pointers are not compared as they cannot be reliably compared
    }
}

impl ScalarType {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            serialize: |v| Ok(v.clone()),
            parse_value: |v| Ok(v.clone()),
            parse_literal: |v| Ok(v.clone()),
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_serializer(mut self, serialize: fn(&Value) -> Result<Value>) -> Self {
        self.serialize = serialize;
        self
    }

    pub fn with_value_parser(mut self, parse_value: fn(&Value) -> Result<Value>) -> Self {
        self.parse_value = parse_value;
        self
    }

    pub fn with_literal_parser(mut self, parse_literal: fn(&Value) -> Result<Value>) -> Self {
        self.parse_literal = parse_literal;
        self
    }
}

/// Object type definition
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectType {
    pub name: String,
    pub description: Option<String>,
    pub fields: HashMap<String, FieldType>,
    pub interfaces: Vec<String>,
}

impl ObjectType {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            fields: HashMap::new(),
            interfaces: Vec::new(),
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_field(mut self, name: String, field: FieldType) -> Self {
        self.fields.insert(name, field);
        self
    }

    pub fn with_interface(mut self, interface: String) -> Self {
        self.interfaces.push(interface);
        self
    }
}

/// Field type definition
#[derive(Debug, Clone, PartialEq)]
pub struct FieldType {
    pub name: String,
    pub description: Option<String>,
    pub field_type: GraphQLType,
    pub arguments: HashMap<String, ArgumentType>,
}

impl FieldType {
    pub fn new(name: String, field_type: GraphQLType) -> Self {
        Self {
            name,
            description: None,
            field_type,
            arguments: HashMap::new(),
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_argument(mut self, name: String, argument: ArgumentType) -> Self {
        self.arguments.insert(name, argument);
        self
    }
}

/// Argument type definition
#[derive(Debug, Clone, PartialEq)]
pub struct ArgumentType {
    pub name: String,
    pub description: Option<String>,
    pub argument_type: GraphQLType,
    pub default_value: Option<Value>,
}

impl ArgumentType {
    pub fn new(name: String, argument_type: GraphQLType) -> Self {
        Self {
            name,
            description: None,
            argument_type,
            default_value: None,
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_default_value(mut self, default_value: Value) -> Self {
        self.default_value = Some(default_value);
        self
    }
}

/// Interface type definition
#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceType {
    pub name: String,
    pub description: Option<String>,
    pub fields: HashMap<String, FieldType>,
    pub interfaces: Vec<String>,
}

impl InterfaceType {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            fields: HashMap::new(),
            interfaces: Vec::new(),
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_field(mut self, name: String, field: FieldType) -> Self {
        self.fields.insert(name, field);
        self
    }
}

/// Union type definition
#[derive(Debug, Clone, PartialEq)]
pub struct UnionType {
    pub name: String,
    pub description: Option<String>,
    pub types: Vec<String>,
}

impl UnionType {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            types: Vec::new(),
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_type(mut self, type_name: String) -> Self {
        self.types.push(type_name);
        self
    }
}

/// Enum type definition
#[derive(Debug, Clone, PartialEq)]
pub struct EnumType {
    pub name: String,
    pub description: Option<String>,
    pub values: HashMap<String, EnumValue>,
}

impl EnumType {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            values: HashMap::new(),
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_value(mut self, name: String, value: EnumValue) -> Self {
        self.values.insert(name, value);
        self
    }
}

/// Enum value definition
#[derive(Debug, Clone, PartialEq)]
pub struct EnumValue {
    pub name: String,
    pub description: Option<String>,
    pub value: Value,
    pub deprecated: Option<String>,
}

impl EnumValue {
    pub fn new(name: String, value: Value) -> Self {
        Self {
            name,
            description: None,
            value,
            deprecated: None,
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_deprecation(mut self, reason: String) -> Self {
        self.deprecated = Some(reason);
        self
    }
}

/// Input object type definition
#[derive(Debug, Clone, PartialEq)]
pub struct InputObjectType {
    pub name: String,
    pub description: Option<String>,
    pub fields: HashMap<String, InputFieldType>,
}

impl InputObjectType {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            fields: HashMap::new(),
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_field(mut self, name: String, field: InputFieldType) -> Self {
        self.fields.insert(name, field);
        self
    }
}

/// Input field type definition
#[derive(Debug, Clone, PartialEq)]
pub struct InputFieldType {
    pub name: String,
    pub description: Option<String>,
    pub field_type: GraphQLType,
    pub default_value: Option<Value>,
}

impl InputFieldType {
    pub fn new(name: String, field_type: GraphQLType) -> Self {
        Self {
            name,
            description: None,
            field_type,
            default_value: None,
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_default_value(mut self, default_value: Value) -> Self {
        self.default_value = Some(default_value);
        self
    }
}

/// Built-in scalar types
pub struct BuiltinScalars;

impl BuiltinScalars {
    pub fn string() -> ScalarType {
        ScalarType::new("String".to_string())
            .with_description("The `String` scalar type represents textual data, represented as UTF-8 character sequences.".to_string())
            .with_serializer(|v| match v {
                Value::StringValue(s) => Ok(Value::StringValue(s.clone())),
                _ => Err(anyhow!("Cannot serialize {:?} as String", v)),
            })
            .with_value_parser(|v| match v {
                Value::StringValue(_) => Ok(v.clone()),
                _ => Err(anyhow!("Cannot parse {:?} as String", v)),
            })
            .with_literal_parser(|v| match v {
                Value::StringValue(_) => Ok(v.clone()),
                _ => Err(anyhow!("Cannot parse literal {:?} as String", v)),
            })
    }

    pub fn int() -> ScalarType {
        ScalarType::new("Int".to_string())
            .with_description(
                "The `Int` scalar type represents non-fractional signed whole numeric values."
                    .to_string(),
            )
            .with_serializer(|v| match v {
                Value::IntValue(i) => Ok(Value::IntValue(*i)),
                _ => Err(anyhow!("Cannot serialize {:?} as Int", v)),
            })
            .with_value_parser(|v| match v {
                Value::IntValue(_) => Ok(v.clone()),
                _ => Err(anyhow!("Cannot parse {:?} as Int", v)),
            })
            .with_literal_parser(|v| match v {
                Value::IntValue(_) => Ok(v.clone()),
                _ => Err(anyhow!("Cannot parse literal {:?} as Int", v)),
            })
    }

    pub fn float() -> ScalarType {
        ScalarType::new("Float".to_string())
            .with_description(
                "The `Float` scalar type represents signed double-precision fractional values."
                    .to_string(),
            )
            .with_serializer(|v| match v {
                Value::FloatValue(f) => Ok(Value::FloatValue(*f)),
                Value::IntValue(i) => Ok(Value::FloatValue(*i as f64)),
                _ => Err(anyhow!("Cannot serialize {:?} as Float", v)),
            })
            .with_value_parser(|v| match v {
                Value::FloatValue(_) => Ok(v.clone()),
                Value::IntValue(i) => Ok(Value::FloatValue(*i as f64)),
                _ => Err(anyhow!("Cannot parse {:?} as Float", v)),
            })
            .with_literal_parser(|v| match v {
                Value::FloatValue(_) => Ok(v.clone()),
                Value::IntValue(i) => Ok(Value::FloatValue(*i as f64)),
                _ => Err(anyhow!("Cannot parse literal {:?} as Float", v)),
            })
    }

    pub fn boolean() -> ScalarType {
        ScalarType::new("Boolean".to_string())
            .with_description("The `Boolean` scalar type represents `true` or `false`.".to_string())
            .with_serializer(|v| match v {
                Value::BooleanValue(b) => Ok(Value::BooleanValue(*b)),
                _ => Err(anyhow!("Cannot serialize {:?} as Boolean", v)),
            })
            .with_value_parser(|v| match v {
                Value::BooleanValue(_) => Ok(v.clone()),
                _ => Err(anyhow!("Cannot parse {:?} as Boolean", v)),
            })
            .with_literal_parser(|v| match v {
                Value::BooleanValue(_) => Ok(v.clone()),
                _ => Err(anyhow!("Cannot parse literal {:?} as Boolean", v)),
            })
    }

    pub fn id() -> ScalarType {
        ScalarType::new("ID".to_string())
            .with_description("The `ID` scalar type represents a unique identifier.".to_string())
            .with_serializer(|v| match v {
                Value::StringValue(s) => Ok(Value::StringValue(s.clone())),
                Value::IntValue(i) => Ok(Value::StringValue(i.to_string())),
                _ => Err(anyhow!("Cannot serialize {:?} as ID", v)),
            })
            .with_value_parser(|v| match v {
                Value::StringValue(_) => Ok(v.clone()),
                Value::IntValue(i) => Ok(Value::StringValue(i.to_string())),
                _ => Err(anyhow!("Cannot parse {:?} as ID", v)),
            })
            .with_literal_parser(|v| match v {
                Value::StringValue(_) => Ok(v.clone()),
                Value::IntValue(i) => Ok(Value::StringValue(i.to_string())),
                _ => Err(anyhow!("Cannot parse literal {:?} as ID", v)),
            })
    }
}

/// Schema containing all types
#[derive(Debug, Clone)]
pub struct Schema {
    pub query_type: Option<String>,
    pub mutation_type: Option<String>,
    pub subscription_type: Option<String>,
    pub types: HashMap<String, GraphQLType>,
    pub directives: HashMap<String, DirectiveType>,
}

impl Schema {
    pub fn new() -> Self {
        let mut schema = Self {
            query_type: None,
            mutation_type: None,
            subscription_type: None,
            types: HashMap::new(),
            directives: HashMap::new(),
        };

        // Add built-in scalar types
        schema.add_type(GraphQLType::Scalar(BuiltinScalars::string()));
        schema.add_type(GraphQLType::Scalar(BuiltinScalars::int()));
        schema.add_type(GraphQLType::Scalar(BuiltinScalars::float()));
        schema.add_type(GraphQLType::Scalar(BuiltinScalars::boolean()));
        schema.add_type(GraphQLType::Scalar(BuiltinScalars::id()));

        schema
    }

    pub fn add_type(&mut self, graphql_type: GraphQLType) {
        let name = graphql_type.name().to_string();
        self.types.insert(name, graphql_type);
    }

    pub fn get_type(&self, name: &str) -> Option<&GraphQLType> {
        self.types.get(name)
    }

    pub fn set_query_type(&mut self, type_name: String) {
        self.query_type = Some(type_name);
    }

    pub fn set_mutation_type(&mut self, type_name: String) {
        self.mutation_type = Some(type_name);
    }

    pub fn set_subscription_type(&mut self, type_name: String) {
        self.subscription_type = Some(type_name);
    }
}

impl Default for Schema {
    fn default() -> Self {
        Self::new()
    }
}

/// Directive type definition
#[derive(Debug, Clone, PartialEq)]
pub struct DirectiveType {
    pub name: String,
    pub description: Option<String>,
    pub locations: Vec<DirectiveLocation>,
    pub arguments: HashMap<String, ArgumentType>,
    pub repeatable: bool,
}

/// Directive locations
#[derive(Debug, Clone, PartialEq)]
pub enum DirectiveLocation {
    Query,
    Mutation,
    Subscription,
    Field,
    FragmentDefinition,
    FragmentSpread,
    InlineFragment,
    VariableDefinition,
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
