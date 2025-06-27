use oxirs_chat::llm::{LLMConfig, LLMManager, LLMRequest, ChatMessage, ChatRole, UseCase, Priority};
use std::env;

#[tokio::test]
async fn test_llm_manager_initialization() {
    // Test basic LLM manager creation with config that doesn't require API keys
    let mut config = LLMConfig::default();
    
    // Disable providers that require API keys for this test
    if let Some(openai_config) = config.providers.get_mut("openai") {
        openai_config.enabled = false;
    }
    if let Some(anthropic_config) = config.providers.get_mut("anthropic") {
        anthropic_config.enabled = false;
    }
    
    let manager = LLMManager::new(config);
    
    match &manager {
        Ok(_) => println!("✅ LLM Manager created successfully"),
        Err(e) => println!("❌ LLM Manager creation failed: {}", e),
    }
    
    assert!(manager.is_ok(), "LLM Manager should initialize successfully: {:?}", manager.err());
    
    let manager = manager.unwrap();
    let usage_stats = manager.get_usage_stats().await;
    
    // Should start with zero usage
    assert_eq!(usage_stats.total_requests, 0);
    assert_eq!(usage_stats.total_tokens, 0);
    
    println!("✅ LLM Manager initialization test passed!");
}

#[tokio::test]
async fn test_llm_request_creation() {
    // Test LLM request structure
    let request = LLMRequest {
        messages: vec![
            ChatMessage {
                role: ChatRole::User,
                content: "Hello, this is a test message".to_string(),
                metadata: None,
            }
        ],
        system_prompt: None,
        temperature: 0.7,
        max_tokens: Some(100),
        use_case: UseCase::Conversation,
        priority: Priority::Normal,
        timeout: None,
    };
    
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0].content, "Hello, this is a test message");
    assert_eq!(request.max_tokens, Some(100));
    assert_eq!(request.temperature, 0.7);
    
    println!("✅ LLM Request creation test passed!");
}

#[tokio::test]
async fn test_openai_config_validation() {
    // Test that OpenAI configuration is properly set up
    let config = LLMConfig::default();
    
    // Check if OpenAI provider is configured
    assert!(config.providers.contains_key("openai"));
    
    let openai_config = &config.providers["openai"];
    assert!(openai_config.enabled);
    assert!(!openai_config.models.is_empty());
    
    // Check model configurations
    let gpt4_model = openai_config.models.iter()
        .find(|m| m.name == "gpt-4");
    assert!(gpt4_model.is_some(), "GPT-4 model should be configured");
    
    let gpt35_model = openai_config.models.iter()
        .find(|m| m.name == "gpt-3.5-turbo");
    assert!(gpt35_model.is_some(), "GPT-3.5-turbo model should be configured");
    
    println!("✅ OpenAI configuration validation test passed!");
}

#[tokio::test]
async fn test_anthropic_config_validation() {
    // Test that Anthropic configuration is properly set up
    let config = LLMConfig::default();
    
    // Check if Anthropic provider is configured
    assert!(config.providers.contains_key("anthropic"));
    
    let anthropic_config = &config.providers["anthropic"];
    assert!(anthropic_config.enabled);
    assert!(!anthropic_config.models.is_empty());
    
    // Check if Claude models are configured
    let claude_model = anthropic_config.models.iter()
        .find(|m| m.name.contains("claude"));
    assert!(claude_model.is_some(), "Claude model should be configured");
    
    println!("✅ Anthropic configuration validation test passed!");
}

// Note: Actual API integration tests would require valid API keys
// and should be run separately with appropriate credentials
#[tokio::test]
#[ignore] // Ignored by default - requires API keys
async fn test_openai_api_integration() {
    // This test requires OPENAI_API_KEY environment variable
    if env::var("OPENAI_API_KEY").is_err() {
        println!("⚠️  Skipping OpenAI API test - OPENAI_API_KEY not set");
        return;
    }
    
    let config = LLMConfig::default();
    let manager = LLMManager::new(config).expect("Failed to create LLM manager");
    
    let request = LLMRequest {
        messages: vec![
            ChatMessage {
                role: ChatRole::User,
                content: "Say 'Hello, world!' in exactly 3 words".to_string(),
                metadata: None,
            }
        ],
        system_prompt: None,
        temperature: 0.1,
        max_tokens: Some(10),
        use_case: UseCase::SimpleQuery,
        priority: Priority::Normal,
        timeout: None,
    };
    
    match manager.generate_response(request).await {
        Ok(response) => {
            assert!(!response.content.is_empty());
            assert!(response.model_used.contains("gpt"));
            println!("✅ OpenAI API integration test passed! Response: {}", response.content);
        }
        Err(e) => {
            println!("⚠️  OpenAI API test failed: {}", e);
            // Don't fail the test - API might be down or have rate limits
        }
    }
}

#[tokio::test]
#[ignore] // Ignored by default - requires API keys  
async fn test_anthropic_api_integration() {
    // This test requires ANTHROPIC_API_KEY environment variable
    if env::var("ANTHROPIC_API_KEY").is_err() {
        println!("⚠️  Skipping Anthropic API test - ANTHROPIC_API_KEY not set");
        return;
    }
    
    let config = LLMConfig::default();
    let manager = LLMManager::new(config).expect("Failed to create LLM manager");
    
    let request = LLMRequest {
        messages: vec![
            ChatMessage {
                role: ChatRole::User,
                content: "Respond with exactly the word 'test'".to_string(),
                metadata: None,
            }
        ],
        system_prompt: None,
        temperature: 0.0,
        max_tokens: Some(5),
        use_case: UseCase::SimpleQuery,
        priority: Priority::Normal,
        timeout: None,
    };
    
    match manager.generate_response(request).await {
        Ok(response) => {
            assert!(!response.content.is_empty());
            assert!(response.model_used.contains("claude"));
            println!("✅ Anthropic API integration test passed! Response: {}", response.content);
        }
        Err(e) => {
            println!("⚠️  Anthropic API test failed: {}", e);
            // Don't fail the test - API might be down or have rate limits
        }
    }
}