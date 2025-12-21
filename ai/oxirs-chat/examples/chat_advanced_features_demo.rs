//! Comprehensive Demo of Advanced OxiRS Chat Features
//!
//! This example demonstrates the advanced capabilities of oxirs-chat including:
//! - RAG with quantum and consciousness enhancements
//! - Multi-provider LLM integration
//! - Real-time collaboration
//! - Schema-aware query generation
//! - Analytics and monitoring
//!
//! Run with: cargo run --example advanced_features_demo

use anyhow::Result;
use oxirs_chat::{
    collaboration::{CollaborationConfig, CollaborationManager},
    ChatConfig, OxiRSChat,
};
use oxirs_core::ConcreteStore;
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== OxiRS Chat Advanced Features Demo ===\n");

    // 1. Initialize RDF Store
    info!("1. Initializing RDF store...");
    let store = Arc::new(ConcreteStore::new()?);
    info!("   ✓ Store initialized\n");

    // 2. Create Chat System with default configuration
    info!("2. Creating OxiRS Chat system...");
    let chat_config = ChatConfig::default();
    let chat = OxiRSChat::new(chat_config, store.clone()).await?;
    info!("   ✓ Chat system initialized with advanced RAG, quantum, and consciousness features\n");

    // 3. Schema Discovery
    info!("3. Performing schema discovery...");
    chat.discover_schema().await?;
    if let Some(schema) = chat.get_discovered_schema().await {
        info!(
            "   ✓ Discovered {} classes and {} properties",
            schema.classes.len(),
            schema.properties.len()
        );
    } else {
        info!("   ℹ No schema discovered (empty store)");
    }
    info!("");

    // 4. Create Chat Sessions
    info!("4. Creating chat sessions...");
    let _session1 = chat.create_session("demo_user_1".to_string()).await?;
    let _session2 = chat.create_session("demo_user_2".to_string()).await?;
    info!("   ✓ Created {} sessions\n", chat.session_count().await);

    // 5. Real-time Collaboration Setup
    info!("5. Setting up real-time collaboration...");
    let collab_config = CollaborationConfig::default();
    let collab_manager = CollaborationManager::new(collab_config);
    let _shared_session = collab_manager
        .create_shared_session("shared_session_1".to_string(), None)
        .await?;
    info!("   ✓ Collaboration session created\n");

    // 6. Process Messages with Advanced AI
    info!("6. Processing messages with advanced AI features...");

    let test_queries = [
        "What is the structure of this knowledge graph?",
        "Find all entities related to semantic web",
        "Explain the relationship between RDF and SPARQL",
    ];

    for (i, query) in test_queries.iter().enumerate() {
        info!("   Query {}: {}", i + 1, query);

        match chat.process_message("demo_user_1", query.to_string()).await {
            Ok(response) => {
                info!(
                    "   ✓ Response generated ({} chars)",
                    response.content.to_string().len()
                );

                if let Some(metadata) = &response.metadata {
                    if let Some(confidence) = metadata.confidence {
                        info!("   ✓ Confidence: {:.2}%", confidence * 100.0);
                    }
                    if let Some(processing_time) = metadata.processing_time_ms {
                        info!("   ✓ Processing time: {}ms", processing_time);
                    }
                }

                // Show rich content elements
                if !response.rich_elements.is_empty() {
                    info!("   ✓ Rich elements: {}", response.rich_elements.len());
                    for element in &response.rich_elements {
                        match element {
                            oxirs_chat::RichContentElement::SPARQLResults { results, .. } => {
                                info!("     - SPARQL results: {} rows", results.len());
                            }
                            oxirs_chat::RichContentElement::ReasoningChain {
                                reasoning_steps,
                                confidence_score,
                            } => {
                                info!(
                                    "     - Reasoning chain: {} steps (confidence: {:.2})",
                                    reasoning_steps.len(),
                                    confidence_score
                                );
                            }
                            oxirs_chat::RichContentElement::QuantumVisualization {
                                results,
                                ..
                            } => {
                                info!("     - Quantum results: {}", results.len());
                            }
                            oxirs_chat::RichContentElement::ConsciousnessInsights {
                                insights,
                                awareness_level,
                            } => {
                                info!(
                                    "     - Consciousness insights: {} (awareness: {:.2})",
                                    insights.len(),
                                    awareness_level
                                );
                            }
                            _ => {}
                        }
                    }
                }
            }
            Err(e) => {
                info!("   ✗ Error: {}", e);
            }
        }
        info!("");
    }

    // 7. Session Count
    info!("7. Session management...");
    info!("   ✓ Active sessions: {}", chat.session_count().await);
    info!("");

    // 8. Session Statistics
    info!("8. Session statistics...");
    if let Ok(stats) = chat.get_session_statistics("demo_user_1").await {
        info!("   ✓ Session stats for demo_user_1:");
        info!("     - Total messages: {}", stats.total_messages);
        info!("     - User messages: {}", stats.user_messages);
        info!("     - Assistant messages: {}", stats.assistant_messages);
        info!("     - Total tokens: {}", stats.total_tokens);
        info!(
            "     - Avg response time: {:.2}ms",
            stats.avg_response_time_ms
        );
    }
    info!("");

    // 9. Circuit Breaker Status
    info!("9. Circuit breaker status...");
    let cb_stats = chat.get_circuit_breaker_stats().await?;
    for (provider, stats) in cb_stats {
        info!("   Provider: {}", provider);
        info!("     - State: {:?}", stats.state);
        info!("     - Total calls: {}", stats.total_calls);
        info!("     - Failed calls: {}", stats.failed_calls);
        info!(
            "     - Success rate: {:.1}%",
            if stats.total_calls > 0 {
                (stats.successful_calls as f64 / stats.total_calls as f64) * 100.0
            } else {
                0.0
            }
        );
    }
    info!("");

    // 10. Session Persistence
    info!("10. Testing session persistence...");
    let saved = chat
        .save_sessions(std::env::temp_dir().join("oxirs_chat_demo"))
        .await?;
    info!("   ✓ Saved {} sessions", saved);

    // Clean up
    let removed = chat.remove_session("demo_user_1").await;
    let removed2 = chat.remove_session("demo_user_2").await;
    info!("   ✓ Removed sessions: {}", removed && removed2);

    let loaded = chat
        .load_sessions(std::env::temp_dir().join("oxirs_chat_demo"))
        .await?;
    info!("   ✓ Loaded {} sessions", loaded);
    info!("");

    // 11. Cleanup Expired Sessions
    info!("11. Cleanup expired sessions...");
    let cleaned = chat.cleanup_expired_sessions().await;
    info!("   ✓ Cleaned up {} expired sessions\n", cleaned);

    info!("=== Demo Complete ===");
    info!("\nKey Features Demonstrated:");
    info!("  ✓ Advanced RAG with quantum and consciousness enhancements");
    info!("  ✓ Multi-provider LLM integration with circuit breaker");
    info!("  ✓ Real-time collaboration support");
    info!("  ✓ Schema-aware query generation");
    info!("  ✓ Rich content visualization");
    info!("  ✓ Session persistence and management");
    info!("  ✓ Analytics and monitoring");
    info!("\nFor production use, see DEPLOYMENT.md");

    Ok(())
}
