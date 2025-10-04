//! Request Logging Integration Tests
//!
//! Tests for HTTP request logging and statistics

use oxirs_fuseki::handlers::request_log::{LogEntry, LogFormat, LoggerConfig, RequestLogger};
use std::sync::Arc;

/// Test creating request logger
#[tokio::test]
async fn test_logger_creation() {
    let logger = RequestLogger::new();
    let config = logger.get_config().unwrap();

    assert!(config.enabled);
    assert_eq!(config.max_entries, 10000);
    assert_eq!(config.format, LogFormat::Json);
}

/// Test logging a single request
#[tokio::test]
async fn test_log_single_request() {
    let logger = Arc::new(RequestLogger::new());

    let mut entry = LogEntry::new("req-1".to_string(), "GET".to_string(), "/query".to_string());
    entry.status_code = 200;
    entry.duration_ms = 150;

    logger.log_request(entry).unwrap();

    let logs = logger.get_logs(None, None).unwrap();
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0].request_id, "req-1");
    assert_eq!(logs[0].method, "GET");
}

/// Test logging multiple requests
#[tokio::test]
async fn test_log_multiple_requests() {
    let logger = Arc::new(RequestLogger::new());

    for i in 0..10 {
        let mut entry = LogEntry::new(
            format!("req-{}", i),
            "GET".to_string(),
            format!("/path{}", i),
        );
        entry.status_code = 200;
        entry.duration_ms = 100 + i * 10;
        logger.log_request(entry).unwrap();
    }

    let logs = logger.get_logs(None, None).unwrap();
    assert_eq!(logs.len(), 10);
}

/// Test log entry limit enforcement
#[tokio::test]
async fn test_log_limit() {
    let config = LoggerConfig {
        max_entries: 5,
        ..Default::default()
    };
    let logger = Arc::new(RequestLogger::with_config(config));

    // Log 10 entries
    for i in 0..10 {
        let entry = LogEntry::new(format!("req-{}", i), "GET".to_string(), "/test".to_string());
        logger.log_request(entry).unwrap();
    }

    let logs = logger.get_logs(None, None).unwrap();
    assert_eq!(logs.len(), 5); // Should only keep last 5

    // Verify we have the most recent ones
    assert_eq!(logs[0].request_id, "req-9"); // Most recent first
}

/// Test statistics tracking
#[tokio::test]
async fn test_statistics() {
    let logger = Arc::new(RequestLogger::new());

    // Log successful request
    let mut entry1 = LogEntry::new("req-1".to_string(), "GET".to_string(), "/query".to_string());
    entry1.status_code = 200;
    entry1.duration_ms = 100;
    entry1.request_size = Some(500);
    entry1.response_size = Some(1000);
    logger.log_request(entry1).unwrap();

    // Log failed request
    let mut entry2 = LogEntry::new(
        "req-2".to_string(),
        "POST".to_string(),
        "/update".to_string(),
    );
    entry2.status_code = 500;
    entry2.duration_ms = 50;
    entry2.request_size = Some(300);
    entry2.response_size = Some(200);
    logger.log_request(entry2).unwrap();

    let stats = logger.get_statistics().unwrap();
    assert_eq!(stats.total_requests, 2);
    assert_eq!(stats.successful_requests, 1);
    assert_eq!(stats.failed_requests, 1);
    assert_eq!(stats.total_duration_ms, 150);
    assert_eq!(stats.min_duration_ms, 50);
    assert_eq!(stats.max_duration_ms, 100);
    assert_eq!(stats.total_request_bytes, 800);
    assert_eq!(stats.total_response_bytes, 1200);
    assert_eq!(stats.avg_duration_ms(), 75.0);
    assert_eq!(stats.success_rate(), 50.0);
}

/// Test filtering by method
#[tokio::test]
async fn test_filter_by_method() {
    use oxirs_fuseki::handlers::request_log::LogFilter;

    let logger = Arc::new(RequestLogger::new());

    // Log GET requests
    for i in 0..3 {
        let entry = LogEntry::new(
            format!("get-{}", i),
            "GET".to_string(),
            "/query".to_string(),
        );
        logger.log_request(entry).unwrap();
    }

    // Log POST requests
    for i in 0..2 {
        let entry = LogEntry::new(
            format!("post-{}", i),
            "POST".to_string(),
            "/update".to_string(),
        );
        logger.log_request(entry).unwrap();
    }

    // Filter GET requests
    let filter = LogFilter {
        method: Some("GET".to_string()),
        min_duration_ms: None,
        status_code: None,
        errors_only: None,
    };
    let logs = logger.get_logs(None, Some(filter)).unwrap();
    assert_eq!(logs.len(), 3);
    assert!(logs.iter().all(|l| l.method == "GET"));

    // Filter POST requests
    let filter = LogFilter {
        method: Some("POST".to_string()),
        min_duration_ms: None,
        status_code: None,
        errors_only: None,
    };
    let logs = logger.get_logs(None, Some(filter)).unwrap();
    assert_eq!(logs.len(), 2);
    assert!(logs.iter().all(|l| l.method == "POST"));
}

/// Test filtering by duration
#[tokio::test]
async fn test_filter_by_duration() {
    use oxirs_fuseki::handlers::request_log::LogFilter;

    let logger = Arc::new(RequestLogger::new());

    // Log requests with varying durations
    for i in 0..5 {
        let mut entry = LogEntry::new(format!("req-{}", i), "GET".to_string(), "/test".to_string());
        entry.duration_ms = i * 100; // 0, 100, 200, 300, 400
        logger.log_request(entry).unwrap();
    }

    // Filter requests >= 200ms
    let filter = LogFilter {
        method: None,
        min_duration_ms: Some(200),
        status_code: None,
        errors_only: None,
    };
    let logs = logger.get_logs(None, Some(filter)).unwrap();
    assert_eq!(logs.len(), 3); // 200, 300, 400
    assert!(logs.iter().all(|l| l.duration_ms >= 200));
}

/// Test filtering by status code
#[tokio::test]
async fn test_filter_by_status() {
    use oxirs_fuseki::handlers::request_log::LogFilter;

    let logger = Arc::new(RequestLogger::new());

    // Log requests with different status codes
    let statuses = vec![200, 404, 500, 200, 503];
    for (i, status) in statuses.iter().enumerate() {
        let mut entry = LogEntry::new(format!("req-{}", i), "GET".to_string(), "/test".to_string());
        entry.status_code = *status;
        logger.log_request(entry).unwrap();
    }

    // Filter 200 OK requests
    let filter = LogFilter {
        method: None,
        min_duration_ms: None,
        status_code: Some(200),
        errors_only: None,
    };
    let logs = logger.get_logs(None, Some(filter)).unwrap();
    assert_eq!(logs.len(), 2);
    assert!(logs.iter().all(|l| l.status_code == 200));
}

/// Test filtering errors only
#[tokio::test]
async fn test_filter_errors_only() {
    use oxirs_fuseki::handlers::request_log::LogFilter;

    let logger = Arc::new(RequestLogger::new());

    // Log mix of successful and error requests
    let statuses = vec![200, 404, 200, 500, 503];
    for (i, status) in statuses.iter().enumerate() {
        let mut entry = LogEntry::new(format!("req-{}", i), "GET".to_string(), "/test".to_string());
        entry.status_code = *status;
        logger.log_request(entry).unwrap();
    }

    // Filter errors only
    let filter = LogFilter {
        method: None,
        min_duration_ms: None,
        status_code: None,
        errors_only: Some(true),
    };
    let logs = logger.get_logs(None, Some(filter)).unwrap();
    assert_eq!(logs.len(), 3); // 404, 500, 503
    assert!(logs.iter().all(|l| l.is_error()));
}

/// Test limit parameter
#[tokio::test]
async fn test_limit_parameter() {
    let logger = Arc::new(RequestLogger::new());

    // Log 10 requests
    for i in 0..10 {
        let entry = LogEntry::new(format!("req-{}", i), "GET".to_string(), "/test".to_string());
        logger.log_request(entry).unwrap();
    }

    // Get only 5 most recent
    let logs = logger.get_logs(Some(5), None).unwrap();
    assert_eq!(logs.len(), 5);
    // Should be most recent (9, 8, 7, 6, 5)
    assert_eq!(logs[0].request_id, "req-9");
    assert_eq!(logs[4].request_id, "req-5");
}

/// Test clearing logs
#[tokio::test]
async fn test_clear_logs() {
    let logger = Arc::new(RequestLogger::new());

    // Log some requests
    for i in 0..5 {
        let entry = LogEntry::new(format!("req-{}", i), "GET".to_string(), "/test".to_string());
        logger.log_request(entry).unwrap();
    }

    assert_eq!(logger.get_logs(None, None).unwrap().len(), 5);

    // Clear logs
    logger.clear_logs().unwrap();

    assert_eq!(logger.get_logs(None, None).unwrap().len(), 0);

    // Statistics should still be preserved
    let stats = logger.get_statistics().unwrap();
    assert_eq!(stats.total_requests, 5);
}

/// Test updating configuration
#[tokio::test]
async fn test_update_config() {
    let logger = Arc::new(RequestLogger::new());

    let new_config = LoggerConfig {
        enabled: false,
        max_entries: 1000,
        format: LogFormat::Text,
        slow_query_threshold_ms: Some(500),
        log_request_body: true,
        log_response_body: true,
        log_sparql: false,
    };

    logger.update_config(new_config.clone()).unwrap();

    let config = logger.get_config().unwrap();
    assert!(!config.enabled);
    assert_eq!(config.max_entries, 1000);
    assert_eq!(config.format, LogFormat::Text);
    assert_eq!(config.slow_query_threshold_ms, Some(500));
}

/// Test disabled logging
#[tokio::test]
async fn test_disabled_logging() {
    let config = LoggerConfig {
        enabled: false,
        ..Default::default()
    };
    let logger = Arc::new(RequestLogger::with_config(config));

    // Try to log request
    let entry = LogEntry::new("req-1".to_string(), "GET".to_string(), "/test".to_string());
    logger.log_request(entry).unwrap();

    // Should not be stored
    let logs = logger.get_logs(None, None).unwrap();
    assert_eq!(logs.len(), 0);
}

/// Test slow query detection
#[tokio::test]
async fn test_slow_query_detection() {
    let config = LoggerConfig {
        slow_query_threshold_ms: Some(1000),
        ..Default::default()
    };
    let logger = Arc::new(RequestLogger::with_config(config));

    // Fast query
    let mut entry1 = LogEntry::new("req-1".to_string(), "GET".to_string(), "/query".to_string());
    entry1.duration_ms = 500;
    logger.log_request(entry1).unwrap();

    // Slow query (should trigger warning log)
    let mut entry2 = LogEntry::new("req-2".to_string(), "GET".to_string(), "/query".to_string());
    entry2.duration_ms = 1500;
    logger.log_request(entry2).unwrap();

    let logs = logger.get_logs(None, None).unwrap();
    assert_eq!(logs.len(), 2);
}

/// Test concurrent logging
#[tokio::test]
async fn test_concurrent_logging() {
    use tokio::task;

    let logger = Arc::new(RequestLogger::new());
    let mut handles = vec![];

    // Log 10 requests concurrently
    for i in 0..10 {
        let logger_clone = logger.clone();
        let handle = task::spawn(async move {
            let entry = LogEntry::new(format!("req-{}", i), "GET".to_string(), "/test".to_string());
            logger_clone.log_request(entry).unwrap();
        });
        handles.push(handle);
    }

    // Wait for all
    for handle in handles {
        handle.await.unwrap();
    }

    let logs = logger.get_logs(None, None).unwrap();
    assert_eq!(logs.len(), 10);

    let stats = logger.get_statistics().unwrap();
    assert_eq!(stats.total_requests, 10);
}

/// Test log entry formatting
#[tokio::test]
async fn test_log_entry_formatting() {
    let mut entry = LogEntry::new(
        "req-123".to_string(),
        "GET".to_string(),
        "/query".to_string(),
    );
    entry.status_code = 200;
    entry.duration_ms = 150;

    // Test JSON formatting
    let json = entry.to_json();
    assert!(json.contains("req-123"));
    assert!(json.contains("GET"));
    assert!(json.contains("/query"));

    // Test text formatting
    let text = entry.to_text();
    assert!(text.contains("req-123"));
    assert!(text.contains("GET"));
    assert!(text.contains("/query"));
    assert!(text.contains("200"));
    assert!(text.contains("150ms"));
}

/// Test log entry status checks
#[tokio::test]
async fn test_log_entry_status_checks() {
    let mut entry = LogEntry::new("req-1".to_string(), "GET".to_string(), "/test".to_string());

    // 2xx - success
    entry.status_code = 200;
    assert!(entry.is_success());
    assert!(!entry.is_error());

    // 4xx - client error
    entry.status_code = 404;
    assert!(!entry.is_success());
    assert!(entry.is_error());

    // 5xx - server error
    entry.status_code = 500;
    assert!(!entry.is_success());
    assert!(entry.is_error());
}

/// Test SPARQL query logging
#[tokio::test]
async fn test_sparql_query_logging() {
    let logger = Arc::new(RequestLogger::new());

    let mut entry = LogEntry::new(
        "req-1".to_string(),
        "POST".to_string(),
        "/query".to_string(),
    );
    entry.sparql_query = Some("SELECT * WHERE { ?s ?p ?o }".to_string());
    entry.operation_type = Some("query".to_string());
    entry.status_code = 200;
    entry.duration_ms = 250;

    logger.log_request(entry).unwrap();

    let logs = logger.get_logs(None, None).unwrap();
    assert_eq!(logs.len(), 1);
    assert!(logs[0].sparql_query.is_some());
    assert_eq!(
        logs[0].sparql_query.as_ref().unwrap(),
        "SELECT * WHERE { ?s ?p ?o }"
    );
}

/// Test metadata storage
#[tokio::test]
async fn test_log_metadata() {
    let logger = Arc::new(RequestLogger::new());

    let mut entry = LogEntry::new("req-1".to_string(), "GET".to_string(), "/query".to_string());
    entry
        .metadata
        .insert("user".to_string(), "admin".to_string());
    entry
        .metadata
        .insert("dataset".to_string(), "test-db".to_string());

    logger.log_request(entry).unwrap();

    let logs = logger.get_logs(None, None).unwrap();
    assert_eq!(logs[0].metadata.get("user"), Some(&"admin".to_string()));
    assert_eq!(
        logs[0].metadata.get("dataset"),
        Some(&"test-db".to_string())
    );
}
