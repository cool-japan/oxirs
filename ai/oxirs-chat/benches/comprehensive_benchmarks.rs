//! Comprehensive Benchmarks for OxiRS Chat
//!
//! This benchmark suite measures performance across all major components.
//!
//! Run with: cargo bench --bench comprehensive_benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_chat::{
    rag::{RagConfig, RagEngine},
    schema_introspection::SchemaIntrospector,
    ChatConfig, ChatSession, Message, MessageContent, MessageRole, OxiRSChat,
};
use oxirs_core::ConcreteStore;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn bench_session_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("session_creation", |b| {
        b.to_async(&rt).iter(|| async {
            let store = Arc::new(ConcreteStore::new().unwrap());
            let session_id = format!("bench_session_{}", uuid::Uuid::new_v4());
            let session = ChatSession::new(session_id, store);
            black_box(session);
        });
    });
}

fn bench_message_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let store = Arc::new(rt.block_on(async { ConcreteStore::new().unwrap() }));

    let mut session = ChatSession::new("bench_session".to_string(), store.clone());

    let mut group = c.benchmark_group("message_processing");

    for size in [10, 50, 100, 500].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let message = Message {
                    id: uuid::Uuid::new_v4().to_string(),
                    role: MessageRole::User,
                    content: MessageContent::from_text("x".repeat(size)),
                    timestamp: chrono::Utc::now(),
                    metadata: None,
                    thread_id: None,
                    parent_message_id: None,
                    token_count: Some(size / 4),
                    reactions: Vec::new(),
                    attachments: Vec::new(),
                    rich_elements: Vec::new(),
                };

                session.add_message(message).unwrap();
                black_box(&session);
            });
        });
    }

    group.finish();
}

fn bench_rag_retrieval(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("rag_retrieval", |b| {
        b.to_async(&rt).iter(|| async {
            let store = Arc::new(ConcreteStore::new().unwrap());
            let config = RagConfig::default();
            let mut rag_engine = RagEngine::new(config, store as Arc<dyn oxirs_core::Store>);
            let _ = rag_engine.initialize().await;

            let query = "Find semantic web resources";
            let result = rag_engine.retrieve(query).await;
            let _ = black_box(result);
        });
    });
}

fn bench_schema_introspection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("schema_introspection", |b| {
        b.to_async(&rt).iter(|| async {
            let store = Arc::new(ConcreteStore::new().unwrap());
            let introspector = SchemaIntrospector::new(store as Arc<dyn oxirs_core::Store>);

            let schema = introspector.discover_schema().await;
            let _ = black_box(schema);
        });
    });
}

fn bench_chat_system_initialization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("chat_system_init", |b| {
        b.to_async(&rt).iter(|| async {
            let store = Arc::new(ConcreteStore::new().unwrap());
            let config = ChatConfig::default();
            let chat = OxiRSChat::new(config, store).await.unwrap();
            black_box(chat);
        });
    });
}

fn bench_concurrent_sessions(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_sessions");

    for session_count in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(session_count),
            session_count,
            |b, &session_count| {
                b.to_async(&rt).iter(|| async move {
                    let store = Arc::new(ConcreteStore::new().unwrap());
                    let config = ChatConfig::default();
                    let chat = Arc::new(OxiRSChat::new(config, store).await.unwrap());

                    let mut handles = vec![];
                    for i in 0..session_count {
                        let chat_clone = Arc::clone(&chat);
                        let handle = tokio::spawn(async move {
                            let session_id = format!("concurrent_session_{}", i);
                            chat_clone.create_session(session_id).await
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        let _ = handle.await;
                    }

                    black_box(chat);
                });
            },
        );
    }

    group.finish();
}

fn bench_session_persistence(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("session_save_load", |b| {
        b.to_async(&rt).iter(|| async {
            let store = Arc::new(ConcreteStore::new().unwrap());
            let config = ChatConfig::default();
            let chat = OxiRSChat::new(config, store).await.unwrap();

            // Create sessions
            for i in 0..10 {
                let _ = chat.create_session(format!("persist_session_{}", i)).await;
            }

            let temp_dir = std::env::temp_dir().join("oxirs_bench");

            // Save
            let saved = chat.save_sessions(&temp_dir).await.unwrap();

            // Clear
            for i in 0..10 {
                chat.remove_session(&format!("persist_session_{}", i)).await;
            }

            // Load
            let loaded = chat.load_sessions(&temp_dir).await.unwrap();

            black_box((saved, loaded));
        });
    });
}

fn bench_message_statistics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let store = Arc::new(rt.block_on(async { ConcreteStore::new().unwrap() }));

    let mut session = ChatSession::new("stats_session".to_string(), store.clone());

    // Add messages
    for i in 0..100 {
        let message = Message {
            id: uuid::Uuid::new_v4().to_string(),
            role: if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            },
            content: MessageContent::from_text(format!("Message {}", i)),
            timestamp: chrono::Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: Some(10),
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };
        session.add_message(message).unwrap();
    }

    c.bench_function("session_statistics", |b| {
        b.iter(|| {
            let stats = session.get_statistics();
            black_box(stats);
        });
    });
}

criterion_group!(
    benches,
    bench_session_creation,
    bench_message_processing,
    bench_rag_retrieval,
    bench_schema_introspection,
    bench_chat_system_initialization,
    bench_concurrent_sessions,
    bench_session_persistence,
    bench_message_statistics,
);

criterion_main!(benches);
