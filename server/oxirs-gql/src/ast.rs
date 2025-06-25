//! GraphQL Abstract Syntax Tree (AST) definitions
//!
//! This module contains the AST node types for representing GraphQL documents.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A GraphQL document containing executable definitions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub definitions: Vec<Definition>,
}

/// Top-level GraphQL definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Definition {
    Operation(OperationDefinition),
    Fragment(FragmentDefinition),
    Schema(SchemaDefinition),
    Type(TypeDefinition),
    Directive(DirectiveDefinition),
    SchemaExtension(SchemaExtension),
    TypeExtension(TypeExtension),
}

/// GraphQL operation definition (query, mutation, subscription)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationDefinition {
    pub operation_type: OperationType,
    pub name: Option<String>,
    pub variable_definitions: Vec<VariableDefinition>,
    pub directives: Vec<Directive>,
    pub selection_set: SelectionSet,
}

/// GraphQL operation type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperationType {
    Query,
    Mutation,
    Subscription,
}

/// Variable definition in operation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariableDefinition {
    pub variable: Variable,
    pub type_: Type,
    pub default_value: Option<Value>,
    pub directives: Vec<Directive>,
}

/// Variable reference
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
}

/// Selection set containing fields
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectionSet {
    pub selections: Vec<Selection>,
}

/// Field or fragment selection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Selection {
    Field(Field),
    InlineFragment(InlineFragment),
    FragmentSpread(FragmentSpread),
}

/// Field selection with arguments
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field {
    pub alias: Option<String>,
    pub name: String,
    pub arguments: Vec<Argument>,
    pub directives: Vec<Directive>,
    pub selection_set: Option<SelectionSet>,
}

/// Field argument
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Argument {
    pub name: String,
    pub value: Value,
}

/// Fragment definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FragmentDefinition {
    pub name: String,
    pub type_condition: String,
    pub directives: Vec<Directive>,
    pub selection_set: SelectionSet,
}

/// Fragment spread
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FragmentSpread {
    pub fragment_name: String,
    pub directives: Vec<Directive>,
}

/// Inline fragment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InlineFragment {
    pub type_condition: Option<String>,
    pub directives: Vec<Directive>,
    pub selection_set: SelectionSet,
}

/// GraphQL value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Variable(Variable),
    IntValue(i64),
    FloatValue(f64),
    StringValue(String),
    BooleanValue(bool),
    NullValue,
    EnumValue(String),
    ListValue(Vec<Value>),
    ObjectValue(HashMap<String, Value>),
}

/// GraphQL type reference
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Type {
    NamedType(String),
    ListType(Box<Type>),
    NonNullType(Box<Type>),
}

impl Type {
    pub fn name(&self) -> String {
        match self {
            Type::NamedType(name) => name.clone(),
            Type::ListType(inner) => format!("[{}]", inner.name()),
            Type::NonNullType(inner) => format!("{}!", inner.name()),
        }
    }
}

/// Directive application
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Directive {
    pub name: String,
    pub arguments: Vec<Argument>,
}

/// Schema definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaDefinition {
    pub description: Option<String>,
    pub directives: Vec<Directive>,
    pub root_operation_types: Vec<RootOperationTypeDefinition>,
}

/// Root operation type definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RootOperationTypeDefinition {
    pub operation_type: OperationType,
    pub named_type: String,
}

/// Type definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeDefinition {
    Scalar(ScalarTypeDefinition),
    Object(ObjectTypeDefinition),
    Interface(InterfaceTypeDefinition),
    Union(UnionTypeDefinition),
    Enum(EnumTypeDefinition),
    InputObject(InputObjectTypeDefinition),
}

/// Scalar type definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarTypeDefinition {
    pub description: Option<String>,
    pub name: String,
    pub directives: Vec<Directive>,
}

/// Object type definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectTypeDefinition {
    pub description: Option<String>,
    pub name: String,
    pub implements_interfaces: Vec<String>,
    pub directives: Vec<Directive>,
    pub fields: Vec<FieldDefinition>,
}

/// Field definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub description: Option<String>,
    pub name: String,
    pub arguments: Vec<InputValueDefinition>,
    pub type_: Type,
    pub directives: Vec<Directive>,
}

/// Input value definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputValueDefinition {
    pub description: Option<String>,
    pub name: String,
    pub type_: Type,
    pub default_value: Option<Value>,
    pub directives: Vec<Directive>,
}

/// Interface type definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterfaceTypeDefinition {
    pub description: Option<String>,
    pub name: String,
    pub implements_interfaces: Vec<String>,
    pub directives: Vec<Directive>,
    pub fields: Vec<FieldDefinition>,
}

/// Union type definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnionTypeDefinition {
    pub description: Option<String>,
    pub name: String,
    pub directives: Vec<Directive>,
    pub types: Vec<String>,
}

/// Enum type definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumTypeDefinition {
    pub description: Option<String>,
    pub name: String,
    pub directives: Vec<Directive>,
    pub values: Vec<EnumValueDefinition>,
}

/// Enum value definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumValueDefinition {
    pub description: Option<String>,
    pub value: String,
    pub directives: Vec<Directive>,
}

/// Input object type definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputObjectTypeDefinition {
    pub description: Option<String>,
    pub name: String,
    pub directives: Vec<Directive>,
    pub fields: Vec<InputValueDefinition>,
}

/// Directive definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectiveDefinition {
    pub description: Option<String>,
    pub name: String,
    pub arguments: Vec<InputValueDefinition>,
    pub repeatable: bool,
    pub locations: Vec<DirectiveLocation>,
}

/// Directive location
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

/// Schema extension
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaExtension {
    pub directives: Vec<Directive>,
    pub root_operation_types: Vec<RootOperationTypeDefinition>,
}

/// Type extension
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeExtension {
    Scalar(ScalarTypeExtension),
    Object(ObjectTypeExtension),
    Interface(InterfaceTypeExtension),
    Union(UnionTypeExtension),
    Enum(EnumTypeExtension),
    InputObject(InputObjectTypeExtension),
}

/// Scalar type extension
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarTypeExtension {
    pub name: String,
    pub directives: Vec<Directive>,
}

/// Object type extension
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectTypeExtension {
    pub name: String,
    pub implements_interfaces: Vec<String>,
    pub directives: Vec<Directive>,
    pub fields: Vec<FieldDefinition>,
}

/// Interface type extension
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterfaceTypeExtension {
    pub name: String,
    pub implements_interfaces: Vec<String>,
    pub directives: Vec<Directive>,
    pub fields: Vec<FieldDefinition>,
}

/// Union type extension
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnionTypeExtension {
    pub name: String,
    pub directives: Vec<Directive>,
    pub types: Vec<String>,
}

/// Enum type extension
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumTypeExtension {
    pub name: String,
    pub directives: Vec<Directive>,
    pub values: Vec<EnumValueDefinition>,
}

/// Input object type extension
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputObjectTypeExtension {
    pub name: String,
    pub directives: Vec<Directive>,
    pub fields: Vec<InputValueDefinition>,
}
