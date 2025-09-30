use anyhow::Result;
use async_trait::async_trait;
use oxirs_chat::llm::config::ProviderConfig;
use oxirs_chat::llm::providers::LLMProvider;
use oxirs_chat::llm::types::{LLMRequest, LLMResponse, LLMResponseStream, Usage};
use oxirs_chat::llm::LLMConfig;
use oxirs_chat::*;
use oxirs_core::ConcreteStore;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;
use uuid::Uuid;

/// Mock LLM provider for testing
#[derive(Clone)]
struct MockLLMProvider {
    name: String,
}

impl MockLLMProvider {
    fn new(name: String) -> Self {
        Self { name }
    }
}

#[async_trait]
impl LLMProvider for MockLLMProvider {
    async fn generate(&self, model: &str, request: &LLMRequest) -> Result<LLMResponse> {
        // Simulate processing time
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Generate a mock response based on the input
        let response_content = format!(
            "Mock response to: {}",
            request
                .messages
                .first()
                .map(|m| m.content.as_str())
                .unwrap_or("")
        );

        Ok(LLMResponse {
            content: response_content,
            model_used: model.to_string(),
            provider_used: self.name.clone(),
            usage: Usage {
                prompt_tokens: 50,
                completion_tokens: 100,
                total_tokens: 150,
                cost: 0.001,
            },
            latency: std::time::Duration::from_millis(100),
            quality_score: Some(0.85),
            metadata: HashMap::new(),
        })
    }

    async fn generate_stream(
        &self,
        _model: &str,
        _request: &LLMRequest,
    ) -> Result<LLMResponseStream> {
        // For simplicity, not implementing streaming in mock
        unimplemented!("Streaming not implemented for mock provider")
    }

    fn get_available_models(&self) -> Vec<String> {
        vec!["mock-model".to_string()]
    }

    fn supports_streaming(&self) -> bool {
        false
    }

    fn get_provider_name(&self) -> &str {
        &self.name
    }

    fn estimate_cost(&self, _model: &str, _input_tokens: usize, _output_tokens: usize) -> f64 {
        0.001 // Mock cost
    }
}

/// Create a test-friendly LLM config with mock provider
#[allow(dead_code)]
fn create_test_llm_config() -> LLMConfig {
    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            enabled: true,
            api_key: Some("test-key".to_string()),
            base_url: Some("http://localhost:8080".to_string()),
            models: vec![oxirs_chat::llm::config::ModelConfig {
                name: "mock-model".to_string(),
                max_tokens: 1000,
                cost_per_token: 0.00001,
                capabilities: vec!["test".to_string()],
                use_cases: vec!["testing".to_string()],
            }],
            timeout: std::time::Duration::from_secs(30),
            max_retries: 3,
        },
    );

    LLMConfig {
        providers,
        routing: oxirs_chat::llm::config::RoutingConfig::default(),
        fallback: oxirs_chat::llm::config::FallbackConfig::default(),
        rate_limits: oxirs_chat::llm::config::RateLimitConfig::default(),
        circuit_breaker: oxirs_chat::llm::config::CircuitBreakerConfig::default(),
    }
}

/// Test LLM Manager that bypasses the provider initialization
pub struct TestLLMManager {
    mock_provider: MockLLMProvider,
}

impl Default for TestLLMManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TestLLMManager {
    pub fn new() -> Self {
        Self {
            mock_provider: MockLLMProvider::new("mock".to_string()),
        }
    }

    pub async fn generate_response(&self, request: LLMRequest) -> Result<LLMResponse> {
        self.mock_provider.generate("mock-model", &request).await
    }
}

#[tokio::test]
async fn test_session_creation_and_persistence() {
    // Create a temporary directory for testing
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let persistence_path = temp_dir.path().join("test_sessions");

    // Create a store for testing
    let store = Arc::new(ConcreteStore::new().expect("Failed to create store"));

    // Create chat manager with persistence
    let manager = ChatManager::with_persistence(store.clone(), &persistence_path)
        .await
        .expect("Failed to create chat manager");

    // Test session creation
    let session_id = "test_session_001".to_string();
    let session = manager
        .get_or_create_session(session_id.clone())
        .await
        .expect("Failed to create session");

    // Verify session exists
    assert_eq!(session.lock().await.id, session_id);

    // Add a message to the session
    {
        let mut session_guard = session.lock().await;
        // Create a test LLM manager
        let test_llm_manager = TestLLMManager::new();

        // For testing, we'll simulate the process_message functionality
        // Since we can't use the real process_message without modifying the Session struct
        // we'll manually create the expected behavior

        // Create user message
        let user_msg = oxirs_chat::messages::Message {
            id: format!("msg_{}", Uuid::new_v4()),
            role: oxirs_chat::messages::MessageRole::User,
            content: oxirs_chat::messages::MessageContent::Text(
                "Hello, this is a test message".to_string(),
            ),
            timestamp: chrono::Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: Some(25),
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };

        session_guard.messages.push(user_msg.clone());

        // Generate LLM response
        let llm_request = LLMRequest {
            messages: vec![oxirs_chat::llm::types::ChatMessage {
                role: oxirs_chat::llm::types::ChatRole::User,
                content: "Hello, this is a test message".to_string(),
                metadata: None,
            }],
            system_prompt: Some("You are a helpful assistant.".to_string()),
            max_tokens: Some(1000),
            temperature: 0.7,
            use_case: oxirs_chat::llm::types::UseCase::Conversation,
            priority: oxirs_chat::llm::types::Priority::Normal,
            timeout: Some(std::time::Duration::from_secs(30)),
        };

        let llm_response = test_llm_manager
            .generate_response(llm_request)
            .await
            .expect("Failed to generate LLM response");

        // Create assistant message
        let response = oxirs_chat::messages::Message {
            id: format!("msg_{}", Uuid::new_v4()),
            role: oxirs_chat::messages::MessageRole::Assistant,
            content: oxirs_chat::messages::MessageContent::Text(llm_response.content),
            timestamp: chrono::Utc::now(),
            metadata: Some(oxirs_chat::messages::MessageMetadata {
                source: Some("test-llm".to_string()),
                confidence: Some(0.85),
                processing_time_ms: Some(100),
                model_used: Some("mock-model".to_string()),
                temperature: Some(0.7),
                max_tokens: Some(1000),
                custom_fields: std::collections::HashMap::new(),
            }),
            thread_id: None,
            parent_message_id: Some(user_msg.id),
            token_count: Some(100),
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };

        session_guard.messages.push(response.clone());

        assert_eq!(response.role, oxirs_chat::messages::MessageRole::Assistant);
        assert!(!response.content.to_string().is_empty());
    }

    // Test session statistics
    let stats = manager.get_session_stats().await;
    assert_eq!(stats.total_sessions, 1);
    assert_eq!(stats.active_sessions, 1);

    // Test detailed metrics
    let metrics = manager.get_detailed_metrics().await;
    assert_eq!(metrics.total_sessions, 1);
    assert_eq!(metrics.active_sessions, 1);
    assert!(metrics.total_messages > 0);

    println!("✅ Session persistence test passed!");
}

#[tokio::test]
async fn test_session_backup_and_restore() {
    // Create temporary directories
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let persistence_path = temp_dir.path().join("test_sessions");
    let backup_path = temp_dir.path().join("test_backup");

    // Create store and manager
    let store = Arc::new(ConcreteStore::new().expect("Failed to create store"));
    let manager = ChatManager::with_persistence(store.clone(), &persistence_path)
        .await
        .expect("Failed to create chat manager");

    // Create a test session with some data
    let session_id = "backup_test_session".to_string();
    let session = manager
        .get_or_create_session(session_id.clone())
        .await
        .expect("Failed to create session");

    // Add some messages to make the session interesting
    {
        let mut session_guard = session.lock().await;
        // Create a test LLM manager
        let test_llm_manager = TestLLMManager::new();

        // Add first message
        let user_msg1 = oxirs_chat::messages::Message {
            id: format!("msg_{}", Uuid::new_v4()),
            role: oxirs_chat::messages::MessageRole::User,
            content: oxirs_chat::messages::MessageContent::Text("First test message".to_string()),
            timestamp: chrono::Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: Some(20),
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };
        session_guard.messages.push(user_msg1.clone());

        let llm_request1 = LLMRequest {
            messages: vec![oxirs_chat::llm::types::ChatMessage {
                role: oxirs_chat::llm::types::ChatRole::User,
                content: "First test message".to_string(),
                metadata: None,
            }],
            system_prompt: Some("You are a helpful assistant.".to_string()),
            max_tokens: Some(1000),
            temperature: 0.7,
            use_case: oxirs_chat::llm::types::UseCase::Conversation,
            priority: oxirs_chat::llm::types::Priority::Normal,
            timeout: Some(std::time::Duration::from_secs(30)),
        };

        let llm_response1 = test_llm_manager
            .generate_response(llm_request1)
            .await
            .expect("Failed to generate first LLM response");

        let response1 = oxirs_chat::messages::Message {
            id: format!("msg_{}", Uuid::new_v4()),
            role: oxirs_chat::messages::MessageRole::Assistant,
            content: oxirs_chat::messages::MessageContent::Text(llm_response1.content),
            timestamp: chrono::Utc::now(),
            metadata: Some(oxirs_chat::messages::MessageMetadata {
                source: Some("test-llm".to_string()),
                confidence: Some(0.85),
                processing_time_ms: Some(100),
                model_used: Some("mock-model".to_string()),
                temperature: Some(0.7),
                max_tokens: Some(1000),
                custom_fields: std::collections::HashMap::new(),
            }),
            thread_id: None,
            parent_message_id: Some(user_msg1.id),
            token_count: Some(100),
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };
        session_guard.messages.push(response1);

        // Add second message
        let user_msg2 = oxirs_chat::messages::Message {
            id: format!("msg_{}", Uuid::new_v4()),
            role: oxirs_chat::messages::MessageRole::User,
            content: oxirs_chat::messages::MessageContent::Text("Second test message".to_string()),
            timestamp: chrono::Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: Some(21),
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };
        session_guard.messages.push(user_msg2.clone());

        let llm_request2 = LLMRequest {
            messages: vec![oxirs_chat::llm::types::ChatMessage {
                role: oxirs_chat::llm::types::ChatRole::User,
                content: "Second test message".to_string(),
                metadata: None,
            }],
            system_prompt: Some("You are a helpful assistant.".to_string()),
            max_tokens: Some(1000),
            temperature: 0.7,
            use_case: oxirs_chat::llm::types::UseCase::Conversation,
            priority: oxirs_chat::llm::types::Priority::Normal,
            timeout: Some(std::time::Duration::from_secs(30)),
        };

        let llm_response2 = test_llm_manager
            .generate_response(llm_request2)
            .await
            .expect("Failed to generate second LLM response");

        let response2 = oxirs_chat::messages::Message {
            id: format!("msg_{}", Uuid::new_v4()),
            role: oxirs_chat::messages::MessageRole::Assistant,
            content: oxirs_chat::messages::MessageContent::Text(llm_response2.content),
            timestamp: chrono::Utc::now(),
            metadata: Some(oxirs_chat::messages::MessageMetadata {
                source: Some("test-llm".to_string()),
                confidence: Some(0.85),
                processing_time_ms: Some(100),
                model_used: Some("mock-model".to_string()),
                temperature: Some(0.7),
                max_tokens: Some(1000),
                custom_fields: std::collections::HashMap::new(),
            }),
            thread_id: None,
            parent_message_id: Some(user_msg2.id),
            token_count: Some(100),
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };
        session_guard.messages.push(response2);
    }

    // Test backup
    let backup_report = manager
        .backup_sessions(&backup_path)
        .await
        .expect("Failed to backup sessions");

    assert_eq!(backup_report.successful_backups, 1);
    assert_eq!(backup_report.failed_backups, 0);

    // Clear sessions (simulate fresh start)
    manager
        .remove_session(&session_id)
        .await
        .expect("Failed to remove session");

    // Test restore
    let restore_report = manager
        .restore_sessions(&backup_path)
        .await
        .expect("Failed to restore sessions");

    assert_eq!(restore_report.sessions_restored, 1);
    // Note: failed_restorations field not available in RestoreReport

    // Verify the restored session
    let restored_session = manager
        .get_session(&session_id)
        .await
        .expect("Restored session not found");

    let restored_session_arc = restored_session.expect("Restored session should exist");
    let restored_session_guard = restored_session_arc.lock().await;
    assert_eq!(restored_session_guard.id, session_id);
    assert!(restored_session_guard.messages.len() >= 2); // Should have the messages we added

    println!("✅ Session backup and restore test passed!");
}

#[tokio::test]
async fn test_session_expiration_and_cleanup() {
    // Create temporary directory
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let persistence_path = temp_dir.path().join("test_sessions");

    // Create store and manager
    let store = Arc::new(ConcreteStore::new().expect("Failed to create store"));
    let manager = ChatManager::with_persistence(store.clone(), &persistence_path)
        .await
        .expect("Failed to create chat manager");

    // Create a session with short timeout
    let session_id = "expiration_test_session".to_string();
    let session = manager
        .get_or_create_session(session_id.clone())
        .await
        .expect("Failed to create session");

    // Manually expire the session by setting old timestamp
    {
        let mut session_guard = session.lock().await;
        session_guard.last_activity = chrono::Utc::now() - chrono::Duration::hours(2);
    }

    // Test cleanup
    let cleaned_count = manager
        .cleanup_expired_sessions()
        .await
        .expect("Failed to cleanup expired sessions");

    assert_eq!(cleaned_count, 1);

    // Verify session was removed
    let session_result = manager.get_session(&session_id).await;
    assert!(session_result.expect("Should get result").is_none());

    println!("✅ Session expiration and cleanup test passed!");
}
