//! # Stream SQL — Parser and Execution Engine
//!
//! Contains the recursive-descent `Parser` that turns token streams into AST nodes,
//! and `StreamSqlEngine` which drives lexing, parsing, caching, and execution.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::stream_sql_ast::{
    AggregateFunction, BinaryOperator, Expression, FromClause, JoinType, Lexer, OrderByItem,
    QueryResult, SelectItem, SelectStatement, SqlValue, StreamMetadata, StreamSqlConfig,
    StreamSqlStats, Token, UnaryOperator, WindowSpec, WindowType,
};

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

// ---------------------------------------------------------------------------
// Stream SQL Engine
// ---------------------------------------------------------------------------

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
                .map(|c| c.alias.clone().unwrap_or_else(|| "column_0".to_string()))
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
