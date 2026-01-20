//! Streaming Response Demo
//!
//! This example demonstrates the streaming API for real-time response generation.
//!
//! Run with: cargo run --example streaming_demo

use anyhow::Result;
use oxirs_chat::{ChatConfig, OxiRSChat, StreamResponseChunk};
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

    info!("=== OxiRS Chat Streaming Demo ===\n");

    // Initialize
    let store = Arc::new(ConcreteStore::new()?);
    let config = ChatConfig::default();
    let chat = OxiRSChat::new(config, store).await?;

    // Create session
    let _session = chat
        .create_session("streaming_demo_user".to_string())
        .await?;
    info!("Session created\n");

    // Test queries
    let queries = [
        "What is RDF?",
        "Explain SPARQL queries",
        "How does semantic web work?",
    ];

    for (i, query) in queries.iter().enumerate() {
        info!("Query {}: {}\n", i + 1, query);

        // Get streaming response
        let mut stream = chat
            .process_message_stream("streaming_demo_user", query.to_string())
            .await?;

        let mut response_text = String::new();
        let mut _start_time = std::time::Instant::now();

        while let Some(chunk) = stream.recv().await {
            match chunk {
                StreamResponseChunk::Status {
                    stage,
                    progress,
                    message,
                } => {
                    info!(
                        "[{:.1}%] {} - {}",
                        progress * 100.0,
                        stage.display_name(),
                        message.unwrap_or_default()
                    );
                }

                StreamResponseChunk::Context {
                    facts,
                    sparql_results,
                    entities,
                } => {
                    if !facts.is_empty() {
                        info!("📚 Found {} relevant facts", facts.len());
                    }
                    if !entities.is_empty() {
                        info!("🏷️  Extracted {} entities", entities.len());
                    }
                    if let Some(_results) = sparql_results {
                        info!("📊 SPARQL results available");
                    }
                }

                StreamResponseChunk::Content { text, is_complete } => {
                    print!("{}", text);
                    response_text.push_str(&text);
                    std::io::Write::flush(&mut std::io::stdout())?;

                    if is_complete {
                        println!();
                    }
                }

                StreamResponseChunk::Complete {
                    total_time,
                    token_count,
                    final_message,
                } => {
                    println!();
                    info!(
                        "✓ Complete in {:.2}s ({} tokens)",
                        total_time.as_secs_f64(),
                        token_count
                    );
                    if let Some(msg) = final_message {
                        info!("  {}", msg);
                    }
                }

                StreamResponseChunk::Error { error, recoverable } => {
                    eprintln!("✗ Error: {} (recoverable: {})", error.message, recoverable);
                }
            }
        }

        println!("\n");
    }

    info!("=== Streaming Demo Complete ===");

    Ok(())
}
