//! # Stream SQL - SQL-like Query Language for Streams
//!
//! This module provides a SQL-like query language specifically designed
//! for stream processing, enabling developers to write familiar SQL queries
//! for real-time data processing.
//!
//! ## Features
//! - SQL parsing and execution for streams
//! - Window functions (TUMBLING, SLIDING, SESSION)
//! - Aggregate functions (COUNT, SUM, AVG, MIN, MAX, STDDEV)
//! - Stream joins with temporal semantics
//! - Pattern matching in streams
//! - Query optimization
//!
//! ## Example
//! ```sql
//! SELECT sensor_id, AVG(temperature) as avg_temp
//! FROM sensor_stream
//! WINDOW TUMBLING (SIZE 5 MINUTES)
//! GROUP BY sensor_id
//! HAVING AVG(temperature) > 30
//! ```

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info};

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

    /// Peek at next character
    fn peek(&self) -> Option<char> {
        self.input.get(self.position + 1).copied()
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
        let quote = self.current_char.unwrap();
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

/// Parser for SQL queries
pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    /// Create a new parser
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    /// Get current token
    fn current_token(&self) -> &Token {
        self.tokens.get(self.position).unwrap_or(&Token::Eof)
    }

    /// Peek at next token
    fn peek_token(&self) -> &Token {
        self.tokens.get(self.position + 1).unwrap_or(&Token::Eof)
    }

    /// Advance to next token
    fn advance(&mut self) {
        self.position += 1;
    }

    /// Expect a specific token
    fn expect(&mut self, expected: Token) -> Result<()> {
        if self.current_token() == &expected {
            self.advance();
            Ok(())
        } else {
            Err(anyhow!(
                "Expected {:?}, got {:?}",
                expected,
                self.current_token()
            ))
        }
    }

    /// Parse a SELECT statement
    pub fn parse_select(&mut self) -> Result<SelectStatement> {
        self.expect(Token::Select)?;

        // DISTINCT
        let distinct = if self.current_token() == &Token::Distinct {
            self.advance();
            true
        } else {
            false
        };

        // SELECT list
        let columns = self.parse_select_list()?;

        // FROM clause
        let from = if self.current_token() == &Token::From {
            self.advance();
            Some(self.parse_from_clause()?)
        } else {
            None
        };

        // WINDOW clause
        let window = if self.current_token() == &Token::Window {
            self.advance();
            Some(self.parse_window_spec()?)
        } else {
            None
        };

        // WHERE clause
        let where_clause = if self.current_token() == &Token::Where {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        // GROUP BY clause
        let group_by = if self.current_token() == &Token::Group {
            self.advance();
            // Expect 'BY'
            if self.current_token() == &Token::By {
                self.advance();
            }
            self.parse_expression_list()?
        } else {
            Vec::new()
        };

        // HAVING clause
        let having = if self.current_token() == &Token::Having {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        // ORDER BY clause
        let order_by = if self.current_token() == &Token::Order {
            self.advance();
            // Expect 'BY'
            if self.current_token() == &Token::By {
                self.advance();
            }
            self.parse_order_by_list()?
        } else {
            Vec::new()
        };

        // LIMIT
        let limit = if self.current_token() == &Token::Limit {
            self.advance();
            if let Token::NumberLiteral(n) = self.current_token() {
                let limit = *n as usize;
                self.advance();
                Some(limit)
            } else {
                None
            }
        } else {
            None
        };

        Ok(SelectStatement {
            distinct,
            columns,
            from,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            offset: None,
            window,
        })
    }

    /// Parse SELECT list
    fn parse_select_list(&mut self) -> Result<Vec<SelectItem>> {
        let mut items = Vec::new();

        loop {
            let expr = self.parse_expression()?;

            // Check for alias
            let alias = if self.current_token() == &Token::As {
                self.advance();
                if let Token::Identifier(name) = self.current_token().clone() {
                    self.advance();
                    Some(name)
                } else {
                    None
                }
            } else if let Token::Identifier(name) = self.current_token().clone() {
                // Alias without AS
                if name.to_uppercase() != "FROM"
                    && name.to_uppercase() != "WHERE"
                    && name.to_uppercase() != "GROUP"
                    && name.to_uppercase() != "ORDER"
                    && name.to_uppercase() != "WINDOW"
                {
                    self.advance();
                    Some(name)
                } else {
                    None
                }
            } else {
                None
            };

            items.push(SelectItem { expr, alias });

            if self.current_token() != &Token::Comma {
                break;
            }
            self.advance(); // Skip comma
        }

        Ok(items)
    }

    /// Parse FROM clause
    fn parse_from_clause(&mut self) -> Result<FromClause> {
        let mut from = self.parse_table_reference()?;

        // Check for joins
        while matches!(
            self.current_token(),
            Token::Join | Token::Inner | Token::Left | Token::Right | Token::Full
        ) {
            let join_type = self.parse_join_type()?;
            let right = self.parse_table_reference()?;

            let condition = if self.current_token() == &Token::On {
                self.advance();
                Some(self.parse_expression()?)
            } else {
                None
            };

            from = FromClause::Join {
                left: Box::new(from),
                right: Box::new(right),
                join_type,
                condition,
            };
        }

        Ok(from)
    }

    /// Parse table reference
    fn parse_table_reference(&mut self) -> Result<FromClause> {
        if let Token::Identifier(name) = self.current_token().clone() {
            self.advance();

            let alias = if self.current_token() == &Token::As {
                self.advance();
                if let Token::Identifier(alias) = self.current_token().clone() {
                    self.advance();
                    Some(alias)
                } else {
                    None
                }
            } else if let Token::Identifier(alias) = self.current_token().clone() {
                // Check if this is an alias or a keyword
                if !matches!(
                    alias.to_uppercase().as_str(),
                    "WHERE"
                        | "GROUP"
                        | "ORDER"
                        | "HAVING"
                        | "LIMIT"
                        | "JOIN"
                        | "INNER"
                        | "LEFT"
                        | "RIGHT"
                        | "FULL"
                        | "ON"
                        | "WINDOW"
                ) {
                    self.advance();
                    Some(alias)
                } else {
                    None
                }
            } else {
                None
            };

            Ok(FromClause::Table { name, alias })
        } else {
            Err(anyhow!("Expected table name"))
        }
    }

    /// Parse join type
    fn parse_join_type(&mut self) -> Result<JoinType> {
        let join_type = match self.current_token() {
            Token::Inner => {
                self.advance();
                JoinType::Inner
            }
            Token::Left => {
                self.advance();
                if self.current_token() == &Token::Outer {
                    self.advance();
                }
                JoinType::Left
            }
            Token::Right => {
                self.advance();
                if self.current_token() == &Token::Outer {
                    self.advance();
                }
                JoinType::Right
            }
            Token::Full => {
                self.advance();
                if self.current_token() == &Token::Outer {
                    self.advance();
                }
                JoinType::Full
            }
            _ => JoinType::Inner,
        };

        // Expect JOIN keyword
        if self.current_token() == &Token::Join {
            self.advance();
        }

        Ok(join_type)
    }

    /// Parse window specification
    fn parse_window_spec(&mut self) -> Result<WindowSpec> {
        let window_type = match self.current_token() {
            Token::Tumbling => {
                self.advance();
                WindowType::Tumbling
            }
            Token::Sliding => {
                self.advance();
                WindowType::Sliding
            }
            Token::Session => {
                self.advance();
                WindowType::Session
            }
            _ => WindowType::Tumbling,
        };

        self.expect(Token::OpenParen)?;

        let mut size = Duration::from_secs(60);
        let mut slide = None;
        let mut gap = None;

        // Parse window parameters
        while self.current_token() != &Token::CloseParen {
            match self.current_token() {
                Token::Size => {
                    self.advance();
                    size = self.parse_duration()?;
                }
                Token::Slide => {
                    self.advance();
                    slide = Some(self.parse_duration()?);
                }
                Token::Gap => {
                    self.advance();
                    gap = Some(self.parse_duration()?);
                }
                Token::Comma => {
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }

        self.expect(Token::CloseParen)?;

        Ok(WindowSpec {
            window_type,
            size,
            slide,
            gap,
            time_attribute: None,
        })
    }

    /// Parse duration (e.g., "5 MINUTES")
    fn parse_duration(&mut self) -> Result<Duration> {
        let value = if let Token::NumberLiteral(n) = self.current_token() {
            let v = *n as u64;
            self.advance();
            v
        } else {
            return Err(anyhow!("Expected number for duration"));
        };

        let unit = if let Token::Identifier(unit) = self.current_token().clone() {
            self.advance();
            unit.to_uppercase()
        } else {
            "SECONDS".to_string()
        };

        let duration = match unit.as_str() {
            "MILLISECONDS" | "MILLIS" | "MS" => Duration::from_millis(value),
            "SECONDS" | "SECOND" | "S" => Duration::from_secs(value),
            "MINUTES" | "MINUTE" | "M" => Duration::from_secs(value * 60),
            "HOURS" | "HOUR" | "H" => Duration::from_secs(value * 3600),
            "DAYS" | "DAY" | "D" => Duration::from_secs(value * 86400),
            _ => Duration::from_secs(value),
        };

        Ok(duration)
    }

    /// Parse expression
    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_or_expression()
    }

    /// Parse OR expression
    fn parse_or_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_and_expression()?;

        while self.current_token() == &Token::Or {
            self.advance();
            let right = self.parse_and_expression()?;
            left = Expression::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::Or,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse AND expression
    fn parse_and_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_comparison_expression()?;

        while self.current_token() == &Token::And {
            self.advance();
            let right = self.parse_comparison_expression()?;
            left = Expression::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse comparison expression
    fn parse_comparison_expression(&mut self) -> Result<Expression> {
        let left = self.parse_additive_expression()?;

        let op = match self.current_token() {
            Token::Equal => Some(BinaryOperator::Equal),
            Token::NotEqual => Some(BinaryOperator::NotEqual),
            Token::LessThan => Some(BinaryOperator::LessThan),
            Token::LessThanOrEqual => Some(BinaryOperator::LessThanOrEqual),
            Token::GreaterThan => Some(BinaryOperator::GreaterThan),
            Token::GreaterThanOrEqual => Some(BinaryOperator::GreaterThanOrEqual),
            Token::Like => Some(BinaryOperator::Like),
            _ => None,
        };

        if let Some(op) = op {
            self.advance();
            let right = self.parse_additive_expression()?;
            Ok(Expression::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
        } else {
            Ok(left)
        }
    }

    /// Parse additive expression
    fn parse_additive_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_multiplicative_expression()?;

        loop {
            let op = match self.current_token() {
                Token::Plus => Some(BinaryOperator::Plus),
                Token::Minus => Some(BinaryOperator::Minus),
                _ => None,
            };

            if let Some(op) = op {
                self.advance();
                let right = self.parse_multiplicative_expression()?;
                left = Expression::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Parse multiplicative expression
    fn parse_multiplicative_expression(&mut self) -> Result<Expression> {
        let mut left = self.parse_unary_expression()?;

        loop {
            let op = match self.current_token() {
                Token::Multiply | Token::Star => Some(BinaryOperator::Multiply),
                Token::Divide => Some(BinaryOperator::Divide),
                Token::Modulo => Some(BinaryOperator::Modulo),
                _ => None,
            };

            if let Some(op) = op {
                self.advance();
                let right = self.parse_unary_expression()?;
                left = Expression::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Parse unary expression
    fn parse_unary_expression(&mut self) -> Result<Expression> {
        match self.current_token() {
            Token::Not => {
                self.advance();
                let expr = self.parse_unary_expression()?;
                Ok(Expression::UnaryOp {
                    op: UnaryOperator::Not,
                    expr: Box::new(expr),
                })
            }
            Token::Minus => {
                self.advance();
                let expr = self.parse_unary_expression()?;
                Ok(Expression::UnaryOp {
                    op: UnaryOperator::Minus,
                    expr: Box::new(expr),
                })
            }
            _ => self.parse_primary_expression(),
        }
    }

    /// Parse primary expression
    fn parse_primary_expression(&mut self) -> Result<Expression> {
        match self.current_token().clone() {
            Token::Star => {
                self.advance();
                Ok(Expression::Star)
            }
            Token::NumberLiteral(n) => {
                self.advance();
                if n.fract() == 0.0 {
                    Ok(Expression::Literal(SqlValue::Integer(n as i64)))
                } else {
                    Ok(Expression::Literal(SqlValue::Float(n)))
                }
            }
            Token::StringLiteral(s) => {
                self.advance();
                Ok(Expression::Literal(SqlValue::String(s)))
            }
            Token::BooleanLiteral(b) => {
                self.advance();
                Ok(Expression::Literal(SqlValue::Boolean(b)))
            }
            Token::Null => {
                self.advance();
                Ok(Expression::Literal(SqlValue::Null))
            }
            Token::Count
            | Token::Sum
            | Token::Avg
            | Token::Min
            | Token::Max
            | Token::StdDev
            | Token::Variance => {
                let func = match self.current_token() {
                    Token::Count => AggregateFunction::Count,
                    Token::Sum => AggregateFunction::Sum,
                    Token::Avg => AggregateFunction::Avg,
                    Token::Min => AggregateFunction::Min,
                    Token::Max => AggregateFunction::Max,
                    Token::StdDev => AggregateFunction::StdDev,
                    Token::Variance => AggregateFunction::Variance,
                    _ => unreachable!(),
                };
                self.advance();
                self.expect(Token::OpenParen)?;

                let distinct = if self.current_token() == &Token::Distinct {
                    self.advance();
                    true
                } else {
                    false
                };

                let expr = self.parse_expression()?;
                self.expect(Token::CloseParen)?;

                Ok(Expression::Aggregate {
                    func,
                    expr: Box::new(expr),
                    distinct,
                })
            }
            Token::Identifier(name) => {
                self.advance();

                // Check for function call
                if self.current_token() == &Token::OpenParen {
                    self.advance();
                    let mut args = Vec::new();

                    if self.current_token() != &Token::CloseParen {
                        loop {
                            args.push(self.parse_expression()?);
                            if self.current_token() != &Token::Comma {
                                break;
                            }
                            self.advance();
                        }
                    }

                    self.expect(Token::CloseParen)?;

                    Ok(Expression::Function {
                        name,
                        args,
                        distinct: false,
                    })
                } else if self.current_token() == &Token::Dot {
                    // Qualified column name
                    self.advance();
                    if let Token::Identifier(column) = self.current_token().clone() {
                        self.advance();
                        Ok(Expression::QualifiedColumn(name, column))
                    } else {
                        Ok(Expression::Column(name))
                    }
                } else {
                    Ok(Expression::Column(name))
                }
            }
            Token::OpenParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(Token::CloseParen)?;
                Ok(expr)
            }
            _ => Err(anyhow!("Unexpected token: {:?}", self.current_token())),
        }
    }

    /// Parse expression list
    fn parse_expression_list(&mut self) -> Result<Vec<Expression>> {
        let mut exprs = Vec::new();

        loop {
            exprs.push(self.parse_expression()?);
            if self.current_token() != &Token::Comma {
                break;
            }
            self.advance();
        }

        Ok(exprs)
    }

    /// Parse ORDER BY list
    fn parse_order_by_list(&mut self) -> Result<Vec<OrderByItem>> {
        let mut items = Vec::new();

        loop {
            let expr = self.parse_expression()?;

            let ascending = if let Token::Identifier(dir) = self.current_token().clone() {
                match dir.to_uppercase().as_str() {
                    "ASC" => {
                        self.advance();
                        true
                    }
                    "DESC" => {
                        self.advance();
                        false
                    }
                    _ => true,
                }
            } else {
                true
            };

            items.push(OrderByItem {
                expr,
                ascending,
                nulls_first: None,
            });

            if self.current_token() != &Token::Comma {
                break;
            }
            self.advance();
        }

        Ok(items)
    }
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

/// Stream SQL engine
pub struct StreamSqlEngine {
    /// Configuration
    config: StreamSqlConfig,
    /// Registered streams
    streams: Arc<RwLock<HashMap<String, StreamMetadata>>>,
    /// Query cache
    query_cache: Arc<RwLock<HashMap<String, SelectStatement>>>,
    /// Statistics
    stats: Arc<RwLock<StreamSqlStats>>,
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

impl StreamSqlEngine {
    /// Create a new Stream SQL engine
    pub fn new(config: StreamSqlConfig) -> Self {
        Self {
            config,
            streams: Arc::new(RwLock::new(HashMap::new())),
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StreamSqlStats::default())),
        }
    }

    /// Parse a SQL query
    pub fn parse(&self, sql: &str) -> Result<SelectStatement> {
        let mut lexer = Lexer::new(sql);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        parser.parse_select()
    }

    /// Execute a SQL query
    pub async fn execute(&self, sql: &str) -> Result<QueryResult> {
        let start_time = std::time::Instant::now();

        // Check cache
        if self.config.enable_query_cache {
            let cache = self.query_cache.read().await;
            if cache.contains_key(sql) {
                let mut stats = self.stats.write().await;
                stats.cache_hits += 1;
                debug!("Query cache hit");
            } else {
                let mut stats = self.stats.write().await;
                stats.cache_misses += 1;
            }
        }

        // Parse query
        let statement = self.parse(sql)?;

        // Update cache
        if self.config.enable_query_cache {
            let mut cache = self.query_cache.write().await;
            if cache.len() < self.config.cache_size {
                cache.insert(sql.to_string(), statement.clone());
            }
        }

        // Execute query (placeholder - actual execution would process the AST)
        let result = QueryResult {
            columns: statement
                .columns
                .iter()
                .map(|c| c.alias.clone().unwrap_or_else(|| format!("column_{}", 0)))
                .collect(),
            rows: Vec::new(),
            execution_time: start_time.elapsed(),
            rows_affected: 0,
        };

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.queries_executed += 1;
        stats.queries_succeeded += 1;
        stats.avg_execution_time_ms = (stats.avg_execution_time_ms
            * (stats.queries_executed - 1) as f64
            + result.execution_time.as_millis() as f64)
            / stats.queries_executed as f64;

        if self.config.enable_query_logging {
            info!(
                "Executed query in {:?}: {}",
                result.execution_time,
                &sql[..sql.len().min(100)]
            );
        }

        Ok(result)
    }

    /// Register a stream
    pub async fn register_stream(&self, metadata: StreamMetadata) -> Result<()> {
        let mut streams = self.streams.write().await;
        info!("Registering stream: {}", metadata.name);
        streams.insert(metadata.name.clone(), metadata);
        Ok(())
    }

    /// Unregister a stream
    pub async fn unregister_stream(&self, name: &str) -> Result<()> {
        let mut streams = self.streams.write().await;
        if streams.remove(name).is_some() {
            info!("Unregistered stream: {}", name);
            Ok(())
        } else {
            Err(anyhow!("Stream not found: {}", name))
        }
    }

    /// Get stream metadata
    pub async fn get_stream(&self, name: &str) -> Option<StreamMetadata> {
        let streams = self.streams.read().await;
        streams.get(name).cloned()
    }

    /// List all streams
    pub async fn list_streams(&self) -> Vec<String> {
        let streams = self.streams.read().await;
        streams.keys().cloned().collect()
    }

    /// Get statistics
    pub async fn get_stats(&self) -> StreamSqlStats {
        self.stats.read().await.clone()
    }

    /// Clear query cache
    pub async fn clear_cache(&self) {
        let mut cache = self.query_cache.write().await;
        cache.clear();
        info!("Query cache cleared");
    }

    /// Validate a query without executing
    pub fn validate(&self, sql: &str) -> Result<()> {
        self.parse(sql)?;
        Ok(())
    }

    /// Explain a query
    pub fn explain(&self, sql: &str) -> Result<String> {
        let statement = self.parse(sql)?;
        Ok(format!("{:#?}", statement))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_basic() {
        let mut lexer = Lexer::new("SELECT * FROM events");
        let tokens = lexer.tokenize();

        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], Token::Select);
        assert_eq!(tokens[1], Token::Star);
        assert_eq!(tokens[2], Token::From);
        assert_eq!(tokens[3], Token::Identifier("events".to_string()));
        assert_eq!(tokens[4], Token::Eof);
    }

    #[test]
    fn test_lexer_with_literals() {
        let mut lexer = Lexer::new("SELECT name, 42, 'hello' FROM events");
        let tokens = lexer.tokenize();

        assert!(matches!(tokens[1], Token::Identifier(_)));
        assert!(matches!(tokens[3], Token::NumberLiteral(_)));
        assert!(matches!(tokens[5], Token::StringLiteral(_)));
    }

    #[test]
    fn test_parser_simple_select() {
        let mut lexer = Lexer::new("SELECT id, name FROM users WHERE id = 1");
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let result = parser.parse_select();

        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert_eq!(stmt.columns.len(), 2);
        assert!(stmt.where_clause.is_some());
    }

    #[test]
    fn test_parser_aggregate() {
        let mut lexer = Lexer::new("SELECT COUNT(*), AVG(value) FROM events");
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let result = parser.parse_select();

        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert_eq!(stmt.columns.len(), 2);
    }

    #[test]
    fn test_parser_window() {
        let mut lexer = Lexer::new("SELECT * FROM events WINDOW TUMBLING (SIZE 5 MINUTES)");
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let result = parser.parse_select();

        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert!(stmt.window.is_some());
        let window = stmt.window.unwrap();
        assert_eq!(window.window_type, WindowType::Tumbling);
        assert_eq!(window.size, Duration::from_secs(300));
    }

    #[test]
    fn test_parser_group_by() {
        let mut lexer = Lexer::new("SELECT sensor_id, AVG(temp) FROM sensors GROUP BY sensor_id");
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let result = parser.parse_select();

        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert!(!stmt.group_by.is_empty());
    }

    #[test]
    fn test_parser_join() {
        let mut lexer = Lexer::new("SELECT * FROM a JOIN b ON a.id = b.aid");
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let result = parser.parse_select();

        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert!(matches!(stmt.from, Some(FromClause::Join { .. })));
    }

    #[tokio::test]
    async fn test_engine_basic() {
        let config = StreamSqlConfig::default();
        let engine = StreamSqlEngine::new(config);

        let result = engine.execute("SELECT * FROM events").await;
        assert!(result.is_ok());

        let stats = engine.get_stats().await;
        assert_eq!(stats.queries_executed, 1);
        assert_eq!(stats.queries_succeeded, 1);
    }

    #[tokio::test]
    async fn test_engine_stream_registration() {
        let config = StreamSqlConfig::default();
        let engine = StreamSqlEngine::new(config);

        let metadata = StreamMetadata {
            name: "events".to_string(),
            columns: vec![
                ColumnDefinition {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                    not_null: true,
                    default: None,
                },
                ColumnDefinition {
                    name: "value".to_string(),
                    data_type: DataType::Float,
                    not_null: false,
                    default: None,
                },
            ],
            properties: HashMap::new(),
            created_at: Utc::now(),
        };

        engine.register_stream(metadata).await.unwrap();

        let streams = engine.list_streams().await;
        assert_eq!(streams.len(), 1);
        assert!(streams.contains(&"events".to_string()));

        let stream = engine.get_stream("events").await;
        assert!(stream.is_some());
        assert_eq!(stream.unwrap().columns.len(), 2);

        engine.unregister_stream("events").await.unwrap();
        let streams = engine.list_streams().await;
        assert!(streams.is_empty());
    }

    #[test]
    fn test_engine_validate() {
        let config = StreamSqlConfig::default();
        let engine = StreamSqlEngine::new(config);

        assert!(engine.validate("SELECT * FROM events").is_ok());
        assert!(engine.validate("INVALID SQL").is_err());
    }

    #[test]
    fn test_engine_explain() {
        let config = StreamSqlConfig::default();
        let engine = StreamSqlEngine::new(config);

        let result = engine.explain("SELECT COUNT(*) FROM events WHERE value > 10");
        assert!(result.is_ok());
        let explanation = result.unwrap();
        assert!(!explanation.is_empty());
    }

    #[tokio::test]
    async fn test_engine_caching() {
        let config = StreamSqlConfig {
            enable_query_cache: true,
            cache_size: 100,
            ..Default::default()
        };
        let engine = StreamSqlEngine::new(config);

        // First execution - cache miss
        engine.execute("SELECT * FROM events").await.unwrap();

        // Second execution - cache hit
        engine.execute("SELECT * FROM events").await.unwrap();

        let stats = engine.get_stats().await;
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn test_parser_complex_expression() {
        let mut lexer = Lexer::new(
            "SELECT * FROM events WHERE (value > 10 AND status = 'active') OR priority = 1",
        );
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let result = parser.parse_select();

        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert!(stmt.where_clause.is_some());
    }

    #[test]
    fn test_parser_order_by() {
        let mut lexer = Lexer::new("SELECT * FROM events ORDER BY created_at DESC, id ASC");
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let result = parser.parse_select();

        assert!(result.is_ok(), "Parse failed: {:?}", result);
        let stmt = result.unwrap();
        assert_eq!(stmt.order_by.len(), 2);
        assert!(!stmt.order_by[0].ascending);
        assert!(stmt.order_by[1].ascending);
    }

    #[test]
    fn test_parser_distinct() {
        let mut lexer = Lexer::new("SELECT DISTINCT sensor_id FROM readings");
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let result = parser.parse_select();

        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert!(stmt.distinct);
    }

    #[test]
    fn test_parser_sliding_window() {
        let mut lexer =
            Lexer::new("SELECT * FROM events WINDOW SLIDING (SIZE 10 SECONDS, SLIDE 5 SECONDS)");
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let result = parser.parse_select();

        assert!(result.is_ok());
        let stmt = result.unwrap();
        assert!(stmt.window.is_some());
        let window = stmt.window.unwrap();
        assert_eq!(window.window_type, WindowType::Sliding);
        assert_eq!(window.size, Duration::from_secs(10));
        assert_eq!(window.slide, Some(Duration::from_secs(5)));
    }
}
