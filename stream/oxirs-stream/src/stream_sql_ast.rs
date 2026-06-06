//! # Stream SQL — AST Types and Lexer
//!
//! SQL-like query language for stream processing:
//! - Configuration types (`StreamSqlConfig`)
//! - Lexer token definitions and `Lexer`
//! - AST node types: `Expression`, `SelectStatement`, `WindowSpec`, etc.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Configuration for Stream SQL engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamSqlConfig {
    /// Maximum query execution time
    pub max_execution_time: Duration,
    /// Enable query optimization
    pub enable_optimization: bool,
    /// Maximum memory for query execution
    pub max_memory_bytes: usize,
    /// Enable parallel execution
    pub parallel_execution: bool,
    /// Number of worker threads
    pub worker_threads: usize,
    /// Enable query caching
    pub enable_query_cache: bool,
    /// Query cache size
    pub cache_size: usize,
    /// Enable query logging
    pub enable_query_logging: bool,
    /// Default window size
    pub default_window_size: Duration,
}

impl Default for StreamSqlConfig {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_secs(60),
            enable_optimization: true,
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            parallel_execution: true,
            worker_threads: 4,
            enable_query_cache: true,
            cache_size: 1000,
            enable_query_logging: true,
            default_window_size: Duration::from_secs(60),
        }
    }
}

/// SQL query types
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    /// SELECT query
    Select,
    /// CREATE STREAM
    CreateStream,
    /// DROP STREAM
    DropStream,
    /// INSERT INTO
    Insert,
    /// CREATE VIEW
    CreateView,
    /// DESCRIBE
    Describe,
    /// EXPLAIN
    Explain,
}

/// SQL token types for lexer
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Select,
    From,
    Where,
    Group,
    By,
    Having,
    Order,
    Limit,
    Window,
    Tumbling,
    Sliding,
    Session,
    Size,
    Slide,
    Gap,
    Create,
    Stream,
    View,
    Drop,
    Insert,
    Into,
    Values,
    As,
    And,
    Or,
    Not,
    In,
    Like,
    Between,
    Is,
    Null,
    Join,
    Inner,
    Left,
    Right,
    Full,
    Outer,
    On,
    Describe,
    Explain,
    Distinct,
    Case,
    When,
    Then,
    Else,
    End,

    // Data types
    Int,
    Float,
    String,
    Boolean,
    Timestamp,

    // Operators
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,

    // Punctuation
    Comma,
    Dot,
    Semicolon,
    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,

    // Literals
    Identifier(String),
    StringLiteral(String),
    NumberLiteral(f64),
    BooleanLiteral(bool),

    // Aggregate functions
    Count,
    Sum,
    Avg,
    Min,
    Max,
    StdDev,
    Variance,

    // Special
    Star,
    Eof,
}

/// Lexer for SQL tokenization
pub struct Lexer {
    input: Vec<char>,
    position: usize,
    current_char: Option<char>,
}

impl Lexer {
    /// Create a new lexer
    pub fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let current_char = chars.first().copied();
        Self {
            input: chars,
            position: 0,
            current_char,
        }
    }

    /// Advance to next character
    fn advance(&mut self) {
        self.position += 1;
        self.current_char = self.input.get(self.position).copied();
    }

    /// Skip whitespace
    fn skip_whitespace(&mut self) {
        while let Some(c) = self.current_char {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Read identifier or keyword
    fn read_identifier(&mut self) -> String {
        let mut result = String::new();
        while let Some(c) = self.current_char {
            if c.is_alphanumeric() || c == '_' {
                result.push(c);
                self.advance();
            } else {
                break;
            }
        }
        result
    }

    /// Read number literal
    fn read_number(&mut self) -> f64 {
        let mut result = String::new();
        while let Some(c) = self.current_char {
            if c.is_ascii_digit() || c == '.' {
                result.push(c);
                self.advance();
            } else {
                break;
            }
        }
        result.parse().unwrap_or(0.0)
    }

    /// Read string literal
    fn read_string(&mut self) -> String {
        let quote = self
            .current_char
            .expect("current_char should be Some when read_string is called");
        self.advance(); // Skip opening quote
        let mut result = String::new();
        while let Some(c) = self.current_char {
            if c == quote {
                self.advance(); // Skip closing quote
                break;
            } else if c == '\\' {
                self.advance();
                if let Some(escaped) = self.current_char {
                    result.push(escaped);
                    self.advance();
                }
            } else {
                result.push(c);
                self.advance();
            }
        }
        result
    }

    /// Get next token
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        match self.current_char {
            None => Token::Eof,
            Some(c) => {
                if c.is_alphabetic() || c == '_' {
                    let ident = self.read_identifier();
                    match ident.to_uppercase().as_str() {
                        "SELECT" => Token::Select,
                        "FROM" => Token::From,
                        "WHERE" => Token::Where,
                        "GROUP" => Token::Group,
                        "BY" => Token::By,
                        "HAVING" => Token::Having,
                        "ORDER" => Token::Order,
                        "LIMIT" => Token::Limit,
                        "WINDOW" => Token::Window,
                        "TUMBLING" => Token::Tumbling,
                        "SLIDING" => Token::Sliding,
                        "SESSION" => Token::Session,
                        "SIZE" => Token::Size,
                        "SLIDE" => Token::Slide,
                        "GAP" => Token::Gap,
                        "CREATE" => Token::Create,
                        "STREAM" => Token::Stream,
                        "VIEW" => Token::View,
                        "DROP" => Token::Drop,
                        "INSERT" => Token::Insert,
                        "INTO" => Token::Into,
                        "VALUES" => Token::Values,
                        "AS" => Token::As,
                        "AND" => Token::And,
                        "OR" => Token::Or,
                        "NOT" => Token::Not,
                        "IN" => Token::In,
                        "LIKE" => Token::Like,
                        "BETWEEN" => Token::Between,
                        "IS" => Token::Is,
                        "NULL" => Token::Null,
                        "JOIN" => Token::Join,
                        "INNER" => Token::Inner,
                        "LEFT" => Token::Left,
                        "RIGHT" => Token::Right,
                        "FULL" => Token::Full,
                        "OUTER" => Token::Outer,
                        "ON" => Token::On,
                        "DESCRIBE" => Token::Describe,
                        "EXPLAIN" => Token::Explain,
                        "DISTINCT" => Token::Distinct,
                        "CASE" => Token::Case,
                        "WHEN" => Token::When,
                        "THEN" => Token::Then,
                        "ELSE" => Token::Else,
                        "END" => Token::End,
                        "INT" | "INTEGER" => Token::Int,
                        "FLOAT" | "DOUBLE" => Token::Float,
                        "STRING" | "VARCHAR" | "TEXT" => Token::String,
                        "BOOLEAN" | "BOOL" => Token::Boolean,
                        "TIMESTAMP" | "DATETIME" => Token::Timestamp,
                        "COUNT" => Token::Count,
                        "SUM" => Token::Sum,
                        "AVG" => Token::Avg,
                        "MIN" => Token::Min,
                        "MAX" => Token::Max,
                        "STDDEV" => Token::StdDev,
                        "VARIANCE" | "VAR" => Token::Variance,
                        "TRUE" => Token::BooleanLiteral(true),
                        "FALSE" => Token::BooleanLiteral(false),
                        _ => Token::Identifier(ident),
                    }
                } else if c.is_ascii_digit() {
                    Token::NumberLiteral(self.read_number())
                } else if c == '\'' || c == '"' {
                    Token::StringLiteral(self.read_string())
                } else {
                    match c {
                        '+' => {
                            self.advance();
                            Token::Plus
                        }
                        '-' => {
                            self.advance();
                            Token::Minus
                        }
                        '*' => {
                            self.advance();
                            Token::Star
                        }
                        '/' => {
                            self.advance();
                            Token::Divide
                        }
                        '%' => {
                            self.advance();
                            Token::Modulo
                        }
                        '=' => {
                            self.advance();
                            Token::Equal
                        }
                        '<' => {
                            self.advance();
                            if self.current_char == Some('=') {
                                self.advance();
                                Token::LessThanOrEqual
                            } else if self.current_char == Some('>') {
                                self.advance();
                                Token::NotEqual
                            } else {
                                Token::LessThan
                            }
                        }
                        '>' => {
                            self.advance();
                            if self.current_char == Some('=') {
                                self.advance();
                                Token::GreaterThanOrEqual
                            } else {
                                Token::GreaterThan
                            }
                        }
                        '!' => {
                            self.advance();
                            if self.current_char == Some('=') {
                                self.advance();
                                Token::NotEqual
                            } else {
                                Token::Not
                            }
                        }
                        ',' => {
                            self.advance();
                            Token::Comma
                        }
                        '.' => {
                            self.advance();
                            Token::Dot
                        }
                        ';' => {
                            self.advance();
                            Token::Semicolon
                        }
                        '(' => {
                            self.advance();
                            Token::OpenParen
                        }
                        ')' => {
                            self.advance();
                            Token::CloseParen
                        }
                        '[' => {
                            self.advance();
                            Token::OpenBracket
                        }
                        ']' => {
                            self.advance();
                            Token::CloseBracket
                        }
                        _ => {
                            self.advance();
                            Token::Eof
                        }
                    }
                }
            }
        }
    }

    /// Tokenize entire input
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            if token == Token::Eof {
                tokens.push(token);
                break;
            }
            tokens.push(token);
        }
        tokens
    }
}

/// Expression in SQL AST
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    /// Column reference
    Column(String),
    /// Qualified column (table.column)
    QualifiedColumn(String, String),
    /// Literal value
    Literal(SqlValue),
    /// Binary operation
    BinaryOp {
        left: Box<Expression>,
        op: BinaryOperator,
        right: Box<Expression>,
    },
    /// Unary operation
    UnaryOp {
        op: UnaryOperator,
        expr: Box<Expression>,
    },
    /// Function call
    Function {
        name: String,
        args: Vec<Expression>,
        distinct: bool,
    },
    /// Aggregate function
    Aggregate {
        func: AggregateFunction,
        expr: Box<Expression>,
        distinct: bool,
    },
    /// CASE expression
    Case {
        operand: Option<Box<Expression>>,
        when_clauses: Vec<(Expression, Expression)>,
        else_clause: Option<Box<Expression>>,
    },
    /// Subquery
    Subquery(Box<SelectStatement>),
    /// IN expression
    InList {
        expr: Box<Expression>,
        list: Vec<Expression>,
        negated: bool,
    },
    /// BETWEEN expression
    Between {
        expr: Box<Expression>,
        low: Box<Expression>,
        high: Box<Expression>,
        negated: bool,
    },
    /// IS NULL
    IsNull {
        expr: Box<Expression>,
        negated: bool,
    },
    /// Star (*)
    Star,
}

/// SQL value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SqlValue {
    Null,
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Timestamp(DateTime<Utc>),
}

/// Binary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinaryOperator {
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    And,
    Or,
    Like,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not,
    Minus,
}

/// Aggregate functions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    StdDev,
    Variance,
}

/// Window specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowSpec {
    /// Window type
    pub window_type: WindowType,
    /// Window size
    pub size: Duration,
    /// Slide interval (for sliding windows)
    pub slide: Option<Duration>,
    /// Session gap (for session windows)
    pub gap: Option<Duration>,
    /// Time attribute
    pub time_attribute: Option<String>,
}

/// Window types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WindowType {
    Tumbling,
    Sliding,
    Session,
}

/// SELECT column specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectItem {
    /// Expression
    pub expr: Expression,
    /// Alias
    pub alias: Option<String>,
}

/// FROM clause item
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FromClause {
    /// Simple table/stream reference
    Table { name: String, alias: Option<String> },
    /// Join
    Join {
        left: Box<FromClause>,
        right: Box<FromClause>,
        join_type: JoinType,
        condition: Option<Expression>,
    },
    /// Subquery
    Subquery {
        query: Box<SelectStatement>,
        alias: String,
    },
}

/// Join types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

/// ORDER BY specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderByItem {
    /// Expression to order by
    pub expr: Expression,
    /// Ascending or descending
    pub ascending: bool,
    /// NULLS FIRST or NULLS LAST
    pub nulls_first: Option<bool>,
}

/// SELECT statement AST
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectStatement {
    /// DISTINCT flag
    pub distinct: bool,
    /// SELECT list
    pub columns: Vec<SelectItem>,
    /// FROM clause
    pub from: Option<FromClause>,
    /// WHERE clause
    pub where_clause: Option<Expression>,
    /// GROUP BY clause
    pub group_by: Vec<Expression>,
    /// HAVING clause
    pub having: Option<Expression>,
    /// ORDER BY clause
    pub order_by: Vec<OrderByItem>,
    /// LIMIT
    pub limit: Option<usize>,
    /// OFFSET
    pub offset: Option<usize>,
    /// Window specification
    pub window: Option<WindowSpec>,
}

/// CREATE STREAM statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateStreamStatement {
    /// Stream name
    pub name: String,
    /// Column definitions
    pub columns: Vec<ColumnDefinition>,
    /// Stream properties
    pub properties: HashMap<String, String>,
}

/// Column definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnDefinition {
    /// Column name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// NOT NULL constraint
    pub not_null: bool,
    /// DEFAULT value
    pub default: Option<Expression>,
}

/// SQL data types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    Integer,
    Float,
    String,
    Boolean,
    Timestamp,
    Array(Box<DataType>),
    Map(Box<DataType>, Box<DataType>),
}

/// Query result row
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultRow {
    /// Column values
    pub values: Vec<SqlValue>,
}

/// Query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Column names
    pub columns: Vec<String>,
    /// Result rows
    pub rows: Vec<ResultRow>,
    /// Execution time
    pub execution_time: Duration,
    /// Rows affected
    pub rows_affected: usize,
}

/// Stream metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetadata {
    /// Stream name
    pub name: String,
    /// Column definitions
    pub columns: Vec<ColumnDefinition>,
    /// Stream properties
    pub properties: HashMap<String, String>,
    /// Created at
    pub created_at: DateTime<Utc>,
}

/// Stream SQL statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamSqlStats {
    /// Total queries executed
    pub queries_executed: u64,
    /// Queries succeeded
    pub queries_succeeded: u64,
    /// Queries failed
    pub queries_failed: u64,
    /// Average execution time
    pub avg_execution_time_ms: f64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
}
