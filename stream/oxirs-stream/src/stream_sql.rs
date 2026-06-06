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
//!
//! ## Module layout
//! - `stream_sql_ast`      — config, token types, lexer, AST nodes, result types
//! - `stream_sql_executor` — `Parser` + `StreamSqlEngine`
//! - `stream_sql_tests`    — integration tests

// Re-export everything from sibling modules so downstream code keeps working.
pub use crate::stream_sql_ast::*;
pub use crate::stream_sql_executor::*;
