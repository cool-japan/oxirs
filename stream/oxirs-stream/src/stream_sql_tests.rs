//! # Stream SQL — Tests

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use std::collections::HashMap;
    use std::time::Duration;

    use crate::stream_sql_ast::{
        ColumnDefinition, DataType, FromClause, Lexer, StreamMetadata, StreamSqlConfig, Token,
        WindowType,
    };
    use crate::stream_sql_executor::{Parser, StreamSqlEngine};

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
