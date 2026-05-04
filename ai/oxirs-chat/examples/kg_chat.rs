//! Knowledge Graph Chat Example
//!
//! Demonstrates starting a session, sending natural-language queries grounded
//! in an RDF knowledge graph, and printing the conversation to stdout.
//!
//! The example populates a small in-memory store with biomedical triples and
//! queries them via natural language. When an LLM API key is present in the
//! environment the real LLM is called; otherwise an error is printed and
//! the example continues to exercise the session and session-persistence APIs.
//!
//! Run with:
//!   cargo run --example kg_chat
//!
//! Optional environment variables:
//!   OPENAI_API_KEY     - Use OpenAI for response generation
//!   ANTHROPIC_API_KEY  - Use Anthropic for response generation

use anyhow::Result;
use oxirs_chat::{ChatConfig, OxiRSChat};
use oxirs_core::{
    model::{Literal, NamedNode, Triple},
    ConcreteStore, Store,
};
use std::sync::Arc;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

/// Populate the store with a small gene-disease knowledge graph.
fn populate_store(store: &mut ConcreteStore) -> Result<()> {
    let assoc = NamedNode::new("http://example.org/associatedWith")?;
    let label = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#label")?;

    // BRCA1 — breast cancer and ovarian cancer
    let brca1 = NamedNode::new("http://example.org/gene/BRCA1")?;
    let breast_cancer = NamedNode::new("http://example.org/disease/BreastCancer")?;
    let ovarian_cancer = NamedNode::new("http://example.org/disease/OvarianCancer")?;
    store.insert_triple(Triple::new(
        brca1.clone(),
        assoc.clone(),
        breast_cancer.clone(),
    ))?;
    store.insert_triple(Triple::new(
        brca1.clone(),
        assoc.clone(),
        ovarian_cancer.clone(),
    ))?;
    store.insert_triple(Triple::new(
        brca1,
        label.clone(),
        Literal::new("BRCA1 tumor suppressor gene"),
    ))?;

    // BRCA2 — breast cancer
    let brca2 = NamedNode::new("http://example.org/gene/BRCA2")?;
    store.insert_triple(Triple::new(
        brca2.clone(),
        assoc.clone(),
        breast_cancer.clone(),
    ))?;
    store.insert_triple(Triple::new(
        brca2,
        label.clone(),
        Literal::new("BRCA2 tumor suppressor gene"),
    ))?;

    // TP53 — multiple cancers
    let tp53 = NamedNode::new("http://example.org/gene/TP53")?;
    let lung_cancer = NamedNode::new("http://example.org/disease/LungCancer")?;
    let colorectal = NamedNode::new("http://example.org/disease/ColorectalCancer")?;
    store.insert_triple(Triple::new(
        tp53.clone(),
        assoc.clone(),
        lung_cancer.clone(),
    ))?;
    store.insert_triple(Triple::new(tp53.clone(), assoc.clone(), colorectal.clone()))?;
    store.insert_triple(Triple::new(
        tp53,
        label.clone(),
        Literal::new("TP53 tumour protein p53"),
    ))?;

    // EGFR — lung cancer
    let egfr = NamedNode::new("http://example.org/gene/EGFR")?;
    store.insert_triple(Triple::new(
        egfr.clone(),
        assoc.clone(),
        lung_cancer.clone(),
    ))?;
    store.insert_triple(Triple::new(
        egfr,
        label,
        Literal::new("Epidermal growth factor receptor"),
    ))?;

    info!("Populated store with {} triples", store.len()?);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialise logging.
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== OxiRS Chat — Knowledge Graph Chat Example ===");

    // 1. Create and populate the in-memory RDF store.
    let mut raw_store = ConcreteStore::new()?;
    populate_store(&mut raw_store)?;
    let store: Arc<dyn Store> = Arc::new(raw_store);

    // 2. Build the chat system (RAG + NL2SPARQL enabled by default).
    let config = ChatConfig {
        max_context_tokens: 4096,
        sliding_window_size: 10,
        enable_context_compression: true,
        temperature: 0.7,
        max_tokens: 1024,
        timeout_seconds: 30,
        enable_topic_tracking: true,
        enable_sentiment_analysis: false,
        enable_intent_detection: true,
    };

    let chat = OxiRSChat::new(config, store).await?;

    // 3. Optionally trigger explicit schema discovery (also runs in background
    //    automatically, but calling it explicitly ensures completion before the
    //    first query).
    if let Err(e) = chat.discover_schema().await {
        warn!("Schema discovery failed (non-fatal): {}", e);
    }
    if let Some(schema) = chat.get_discovered_schema().await {
        info!(
            "Schema: {} class(es), {} propert(ies)",
            schema.classes.len(),
            schema.properties.len()
        );
    }

    // 4. Create a named user session.
    let session_id = "kg_chat_demo".to_string();
    let _session = chat.create_session(session_id.clone()).await?;
    info!("Session '{}' created.\n", session_id);

    // 5. Send natural-language queries.
    let queries = [
        "What genes are associated with breast cancer?",
        "List all diseases related to TP53.",
        "Is EGFR linked to ovarian cancer?",
    ];

    for query in &queries {
        println!("You: {query}");

        match chat.process_message(&session_id, query.to_string()).await {
            Ok(response) => {
                println!("Assistant: {}", response.content.to_text());
                if let Some(meta) = &response.metadata {
                    if let Some(ms) = meta.processing_time_ms {
                        println!("  [processed in {}ms]", ms);
                    }
                    if let Some(score) = meta.confidence {
                        println!("  [context confidence: {:.2}]", score);
                    }
                }
            }
            Err(e) => {
                // LLM calls fail without API keys; show the error and continue.
                eprintln!(
                    "Note: LLM call failed ({}). Set OPENAI_API_KEY or ANTHROPIC_API_KEY \
                     to enable full response generation.",
                    e
                );
            }
        }
        println!();
    }

    // 6. Show session statistics.
    match chat.get_session_statistics(&session_id).await {
        Ok(stats) => {
            info!(
                "Session stats — messages: {}, avg response time: {:.0}ms",
                stats.total_messages, stats.avg_response_time_ms
            );
        }
        Err(e) => warn!("Could not retrieve session stats: {}", e),
    }

    // 7. Persist the session to disk (uses temp dir in this demo).
    let session_dir = std::env::temp_dir().join("oxirs-chat-kg-demo");
    match chat.save_sessions(&session_dir).await {
        Ok(n) => info!("Saved {} session(s) to {:?}", n, session_dir),
        Err(e) => warn!("Session save failed (non-fatal): {}", e),
    }

    info!("Done.");
    Ok(())
}
