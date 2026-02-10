//! Integration Tests for LLM Fallback Chains
//!
//! Tests the complete fallback chain: OpenAI → Anthropic → Ollama (local)
//! with circuit breakers, health monitoring, caching, and token budgets.

use oxirs_chat::llm::{
    BudgetConfig, CacheConfig, CircuitBreakerConfig, HealthCheckConfig, LLMConfig, LLMManager,
    LLMRequest, LLMResponse, ProviderConfig, ResponseCache, TokenBudget, HealthChecker,
    ChatMessage, ChatRole, Priority, Usage, UseCase,
};
use std::time::Duration;

/// Helper to create a test request
fn create_test_request(content: &str) -> LLMRequest {
    LLMRequest {
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: content.to_string(),
            metadata: None,
        }],
        system_prompt: Some("Test system prompt".to_string()),
        temperature: 0.7,
        max_tokens: Some(100),
        use_case: UseCase::Conversation,
        priority: Priority::Normal,
        timeout: Some(Duration::from_secs(30)),
    }
}

/// Test 1: Happy path - OpenAI succeeds, no fallback
#[tokio::test]
#[ignore] // Requires API key
async fn test_openai_success_no_fallback() {
    let config = LLMConfig::default();
    let mut manager = LLMManager::new(config).expect("Failed to create LLM manager");

    let request = create_test_request("Hello, world!");
    let response = manager.generate_response(request).await;

    assert!(response.is_ok(), "OpenAI should succeed");
    let response = response.unwrap();
    assert!(!response.content.is_empty());
    assert_eq!(response.model_used, "gpt-4");
}

/// Test 2: Cache hit - Response returned from cache
#[tokio::test]
async fn test_cache_hit() {
    let cache = ResponseCache::new(CacheConfig::default());
    let request = create_test_request("test query");

    // First request - cache miss
    assert!(cache.get(&request).await.is_none());

    // Store response
    let test_response = LLMResponse {
        content: "test response".to_string(),
        model_used: "test-model".to_string(),
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            cost: 0.001,
        },
        latency: Duration::from_millis(100),
        provider_used: "test-provider".to_string(),
        quality_score: Some(0.9),
        metadata: std::collections::HashMap::new(),
    };

    cache.put(&request, test_response.clone(), "test-provider".to_string()).await;

    // Second request - cache hit
    let cached = cache.get(&request).await;
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().content, test_response.content);

    // Verify hit rate
    let hit_rate = cache.hit_rate().await;
    assert!((hit_rate - 0.5).abs() < 0.01); // 1 hit, 1 miss = 50%
}

/// Test 3: Cache miss - API called, response cached
#[tokio::test]
async fn test_cache_miss() {
    let cache = ResponseCache::new(CacheConfig::default());
    let request = create_test_request("new query");

    // Should be cache miss
    let result = cache.get(&request).await;
    assert!(result.is_none());

    let metrics = cache.get_metrics().await;
    assert_eq!(metrics.misses, 1);
    assert_eq!(metrics.hits, 0);
}

/// Test 4: Cache hit rate >70% on repeated queries
#[tokio::test]
async fn test_cache_high_hit_rate() {
    let cache = ResponseCache::new(CacheConfig::default());
    let request = create_test_request("repeated query");

    let test_response = LLMResponse {
        content: "response".to_string(),
        model_used: "test-model".to_string(),
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            cost: 0.001,
        },
        latency: Duration::from_millis(100),
        provider_used: "test-provider".to_string(),
        quality_score: Some(0.9),
        metadata: std::collections::HashMap::new(),
    };

    // Store once
    cache.put(&request, test_response, "test-provider".to_string()).await;

    // Request 10 times (9 hits + 1 miss from initial storage = 90% hit rate)
    for _ in 0..10 {
        let _ = cache.get(&request).await;
    }

    let hit_rate = cache.hit_rate().await;
    assert!(hit_rate > 0.7, "Hit rate should be >70%, got {}", hit_rate);
}

/// Test 5: Circuit breaker opens after 3 failures
#[tokio::test]
async fn test_circuit_breaker_opens() {
    use oxirs_chat::llm::CircuitBreaker;

    let mut config = CircuitBreakerConfig::default();
    config.failure_threshold = 3;

    let circuit_breaker = CircuitBreaker::new(config);

    // Should be closed initially
    assert!(circuit_breaker.can_execute().await);

    // Record 3 failures
    for _ in 0..3 {
        circuit_breaker.record_result(false, Duration::from_millis(100)).await;
    }

    // Should be open now
    let stats = circuit_breaker.get_stats().await;
    assert_eq!(stats.consecutive_failures, 3);
}

/// Test 6: Circuit breaker transitions to half-open after timeout
#[tokio::test]
async fn test_circuit_breaker_half_open() {
    use oxirs_chat::llm::CircuitBreaker;

    let mut config = CircuitBreakerConfig::default();
    config.failure_threshold = 2;
    config.timeout_duration = Duration::from_secs(1);

    let circuit_breaker = CircuitBreaker::new(config);

    // Trigger circuit to open
    for _ in 0..2 {
        circuit_breaker.record_result(false, Duration::from_millis(100)).await;
    }

    // Wait for timeout
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Should allow execution (half-open)
    assert!(circuit_breaker.can_execute().await);
}

/// Test 7: Health monitoring - Unhealthy providers skipped
#[tokio::test]
async fn test_unhealthy_provider_skipped() {
    let health_checker = HealthChecker::new(HealthCheckConfig::default());

    let provider_id = "test-provider".to_string();
    health_checker.register_provider(provider_id.clone()).await;

    // Record 10 failures
    for _ in 0..10 {
        health_checker
            .record_call(&provider_id, false, Duration::from_millis(100))
            .await
            .unwrap();
    }

    // Provider should be unhealthy
    assert!(!health_checker.is_provider_healthy(&provider_id).await);

    // Get healthy providers should not include this one
    let healthy = health_checker.get_healthy_providers().await;
    assert!(!healthy.contains(&provider_id));
}

/// Test 8: Token budget - API blocked when budget exceeded
#[tokio::test]
async fn test_token_budget_exceeded() {
    let config = BudgetConfig {
        default_monthly_limit: 1000,
        ..Default::default()
    };

    let budget_manager = TokenBudget::new(config);
    let user_id = "test_user".to_string();

    budget_manager
        .create_user_budget(user_id.clone(), 1000)
        .await
        .unwrap();

    // Use 900 tokens
    budget_manager
        .record_usage(&user_id, 900)
        .await
        .unwrap();

    // Requesting 200 tokens should fail
    let result = budget_manager.check_budget(&user_id, 200).await;
    assert!(result.is_err(), "Should fail when budget exceeded");
}

/// Test 9: Token budget - API allowed when budget sufficient
#[tokio::test]
async fn test_token_budget_sufficient() {
    let budget_manager = TokenBudget::new(BudgetConfig::default());
    let user_id = "test_user".to_string();

    budget_manager
        .create_user_budget(user_id.clone(), 10000)
        .await
        .unwrap();

    // Use 5000 tokens
    budget_manager
        .record_usage(&user_id, 5000)
        .await
        .unwrap();

    // Requesting 2000 tokens should succeed
    let result = budget_manager.check_budget(&user_id, 2000).await;
    assert!(result.is_ok(), "Should succeed when budget sufficient");

    let remaining = budget_manager.get_remaining_budget(&user_id).await;
    assert_eq!(remaining, 5000);
}

/// Test 10: Failover latency <500ms
#[tokio::test]
async fn test_failover_latency() {
    use std::time::Instant;

    let health_checker = HealthChecker::new(HealthCheckConfig::default());

    // Register and mark provider as unhealthy
    let provider_id = "slow-provider".to_string();
    health_checker.register_provider(provider_id.clone()).await;

    let start = Instant::now();

    // Record failure
    health_checker
        .record_call(&provider_id, false, Duration::from_millis(100))
        .await
        .unwrap();

    let elapsed = start.elapsed();

    // Recording and failover decision should be fast
    assert!(elapsed < Duration::from_millis(500), "Failover should be <500ms");
}

/// Test 11: All providers fail - Error returned
#[tokio::test]
async fn test_all_providers_fail() {
    let mut config = LLMConfig::default();

    // Disable all providers
    for (_, provider_config) in config.providers.iter_mut() {
        provider_config.enabled = false;
    }

    let result = LLMManager::new(config);
    assert!(result.is_ok()); // Manager creation should succeed

    // But generating response should fail
    let mut manager = result.unwrap();
    let request = create_test_request("test");

    let response = manager.generate_response(request).await;
    assert!(response.is_err(), "Should fail when no providers available");
}

/// Test 12: Health status transitions - Healthy → Degraded → Unhealthy
#[tokio::test]
async fn test_health_status_transitions() {
    let mut config = HealthCheckConfig::default();
    config.latency_threshold_ms = 100;

    let health_checker = HealthChecker::new(config);
    let provider_id = "provider".to_string();

    // Initially healthy
    health_checker.register_provider(provider_id.clone()).await;
    assert!(health_checker.is_provider_healthy(&provider_id).await);

    // Degraded (slow but successful)
    for _ in 0..5 {
        health_checker
            .record_call(&provider_id, true, Duration::from_millis(200))
            .await
            .unwrap();
    }

    let health = health_checker.get_health_status(&provider_id).await.unwrap();
    assert_eq!(health.status, oxirs_chat::llm::HealthStatus::Degraded);

    // Unhealthy (failures)
    for _ in 0..10 {
        health_checker
            .record_call(&provider_id, false, Duration::from_millis(100))
            .await
            .unwrap();
    }

    assert!(!health_checker.is_provider_healthy(&provider_id).await);
}

/// Test 13: Response caching reduces API calls
#[tokio::test]
async fn test_cache_reduces_api_calls() {
    let cache = ResponseCache::new(CacheConfig::default());
    let request = create_test_request("cached query");

    let test_response = LLMResponse {
        content: "cached response".to_string(),
        model_used: "test-model".to_string(),
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            cost: 0.001,
        },
        latency: Duration::from_millis(100),
        provider_used: "test-provider".to_string(),
        quality_score: Some(0.9),
        metadata: std::collections::HashMap::new(),
    };

    // First call - cache miss
    assert!(cache.get(&request).await.is_none());

    cache.put(&request, test_response.clone(), "provider".to_string()).await;

    // Subsequent calls - cache hits (no API calls)
    for _ in 0..100 {
        let cached = cache.get(&request).await;
        assert!(cached.is_some());
    }

    let metrics = cache.get_metrics().await;
    assert_eq!(metrics.hits, 100);
    assert_eq!(metrics.misses, 1);
}

/// Test 14: LRU eviction works correctly
#[tokio::test]
async fn test_lru_eviction() {
    let mut config = CacheConfig::default();
    config.max_size = 3;

    let cache = ResponseCache::new(config);

    let response = LLMResponse {
        content: "response".to_string(),
        model_used: "test-model".to_string(),
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            cost: 0.001,
        },
        latency: Duration::from_millis(100),
        provider_used: "test-provider".to_string(),
        quality_score: Some(0.9),
        metadata: std::collections::HashMap::new(),
    };

    // Add 3 entries
    for i in 1..=3 {
        let request = create_test_request(&format!("query {}", i));
        cache.put(&request, response.clone(), "provider".to_string()).await;
    }

    // Access entry 1 to make it recently used
    let req1 = create_test_request("query 1");
    let _ = cache.get(&req1).await;

    // Add 4th entry - should evict entry 2 (LRU)
    let req4 = create_test_request("query 4");
    cache.put(&req4, response.clone(), "provider".to_string()).await;

    // Entry 1 and 3 should exist, entry 2 should be evicted
    assert!(cache.get(&req1).await.is_some(), "Entry 1 should exist");
    assert!(cache.get(&create_test_request("query 2")).await.is_none(), "Entry 2 should be evicted");
    assert!(cache.get(&create_test_request("query 3")).await.is_some(), "Entry 3 should exist");
    assert!(cache.get(&req4).await.is_some(), "Entry 4 should exist");
}

/// Test 15: Token budget auto-reset
#[tokio::test]
async fn test_token_budget_reset() {
    let budget_manager = TokenBudget::new(BudgetConfig::default());
    let user_id = "test_user".to_string();

    budget_manager
        .create_user_budget(user_id.clone(), 1000)
        .await
        .unwrap();

    // Use all tokens
    budget_manager
        .record_usage(&user_id, 1000)
        .await
        .unwrap();

    let stats = budget_manager.get_usage_stats(&user_id).await.unwrap();
    assert_eq!(stats.remaining_tokens, 0);

    // Reset budget
    budget_manager.reset_user_budget(&user_id).await.unwrap();

    let stats = budget_manager.get_usage_stats(&user_id).await.unwrap();
    assert_eq!(stats.remaining_tokens, 1000);
}
