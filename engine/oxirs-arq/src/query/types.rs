//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{
    Algebra, Expression, GroupCondition, Iri, OrderCondition, TriplePattern, Variable,
};
use crate::update::UpdateOperation;
use std::collections::HashMap;

/// SPARQL query types
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    Select,
    Construct,
    Ask,
    Describe,
}
/// Dataset clause for FROM and FROM NAMED
#[derive(Debug, Clone, Default)]
pub struct DatasetClause {
    pub default_graphs: Vec<Iri>,
    pub named_graphs: Vec<Iri>,
}
/// SPARQL UPDATE request representation
#[derive(Debug, Clone)]
pub struct UpdateRequest {
    pub operations: Vec<UpdateOperation>,
    pub prefixes: HashMap<String, String>,
    pub base_iri: Option<String>,
}
/// Token types for SPARQL parsing
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Select,
    Construct,
    Ask,
    Describe,
    Where,
    Optional,
    Union,
    Minus,
    Filter,
    Bind,
    Service,
    Graph,
    From,
    Named,
    Prefix,
    Base,
    Distinct,
    Reduced,
    OrderBy,
    GroupBy,
    Having,
    Limit,
    Offset,
    Asc,
    Desc,
    As,
    Values,
    Exists,
    NotExists,
    Insert,
    Delete,
    Update,
    Create,
    Drop,
    Clear,
    Load,
    Copy,
    Move,
    Add,
    Data,
    With,
    Using,
    Silent,
    All,
    Default,
    To,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    Not,
    Plus,
    Minus_,
    Multiply,
    Divide,
    Pipe,
    Caret,
    Slash,
    Question,
    Star,
    Bang,
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Dot,
    Semicolon,
    Comma,
    Colon,
    Iri(String),
    PrefixedName(String, String),
    Variable(String),
    StringLiteral(String),
    NumericLiteral(String),
    BooleanLiteral(bool),
    BlankNode(String),
    Eof,
    Newline,
}
/// SPARQL query representation
#[derive(Debug, Clone)]
pub struct Query {
    pub query_type: QueryType,
    pub select_variables: Vec<Variable>,
    pub where_clause: Algebra,
    pub order_by: Vec<OrderCondition>,
    pub group_by: Vec<GroupCondition>,
    pub having: Option<Expression>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub distinct: bool,
    pub reduced: bool,
    pub construct_template: Vec<TriplePattern>,
    pub prefixes: HashMap<String, String>,
    pub base_iri: Option<String>,
    pub dataset: DatasetClause,
}
