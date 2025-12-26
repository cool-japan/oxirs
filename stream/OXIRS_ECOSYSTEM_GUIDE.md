# OxiRS Ecosystem: Comprehensive Usage Guide & Tutorials

<div align="center">

![OxiRS Logo](https://raw.githubusercontent.com/cool-japan/oxirs/main/assets/oxirs-logo.png)

**Ultra-High Performance RDF Streaming & Federation Platform**

[![Rust](https://img.shields.io/badge/rust-1.70+-brightgreen.svg)](https://www.rust-lang.org)
[![Performance](https://img.shields.io/badge/throughput-100K%2B%20events%2Fsec-blue.svg)](#performance)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-brightgreen.svg)](#documentation)

**Real-time RDF streaming with federated SPARQL queries, GraphQL federation, and enterprise-grade features**

</div>

---

## 📋 Table of Contents

- [🚀 Quick Start](#quick-start)
- [🏗️ Architecture Overview](#architecture-overview)
- [🌊 Stream Processing Tutorials](#stream-processing-tutorials)
- [🌐 Federation Tutorials](#federation-tutorials)
- [🔗 Integration Patterns](#integration-patterns)
- [📊 Performance Optimization](#performance-optimization)
- [🔐 Security Best Practices](#security-best-practices)
- [📈 Monitoring & Observability](#monitoring--observability)
- [🏭 Production Deployment](#production-deployment)
- [🧪 Testing Strategies](#testing-strategies)
- [❓ FAQ & Troubleshooting](#faq--troubleshooting)
- [🤝 Contributing](#contributing)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the OxiRS repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs

# Build the entire ecosystem
cargo build --workspace --release

# Run tests to ensure everything works
cargo nextest run --workspace --no-fail-fast
```

### Your First Stream

```rust
use oxirs_stream::{StreamManager, StreamConfig, Backend, Event, RdfPatchEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a high-performance streaming configuration
    let config = StreamConfig {
        backend: Backend::Memory,
        max_events_per_sec: 10000,
        enable_performance_optimization: true,
        enable_compression: true,
        ..Default::default()
    };

    // Initialize the stream manager
    let stream_manager = StreamManager::new(config).await?;
    let producer = stream_manager.create_producer(Backend::Memory, "my-topic").await?;
    let consumer = stream_manager.create_consumer(Backend::Memory, "my-topic").await?;

    // Stream some RDF data
    let event = Event::RdfPatch(RdfPatchEvent {
        patch_id: uuid::Uuid::new_v4(),
        patch_data: "A <http://example.org/Alice> <http://xmlns.com/foaf/0.1/name> \"Alice Smith\" .".to_string(),
        metadata: Default::default(),
    });

    producer.send(event).await?;
    
    // Consume the event
    if let Some(received_event) = consumer.consume().await? {
        println!("Received: {:?}", received_event);
    }

    Ok(())
}
```

### Your First Federation Query

```rust
use oxirs_federate::{FederationEngine, FederatedService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create federation engine
    let federation_engine = FederationEngine::new();

    // Register SPARQL services
    let dbpedia_service = FederatedService::new_sparql(
        "dbpedia".to_string(),
        "DBpedia SPARQL Endpoint".to_string(),
        "https://dbpedia.org/sparql".to_string(),
    );

    federation_engine.register_service(dbpedia_service).await?;

    // Execute federated query
    let query = r#"
        SELECT ?name ?abstract WHERE {
            ?person foaf:name ?name .
            ?person dbo:abstract ?abstract .
            FILTER(lang(?abstract) = "en")
            FILTER(?name = "Albert Einstein")
        }
        LIMIT 5
    "#;

    let result = federation_engine.execute_sparql(query).await?;
    println!("Query executed in {:?}", result.metadata.execution_time);
    println!("Found {} results", result.result_count());

    Ok(())
}
```

---

## 🏗️ Architecture Overview

OxiRS is built with a modular, high-performance architecture designed for scalability and enterprise use:

```
┌─────────────────────────────────────────────────────────────┐
│                    OxiRS Ecosystem                          │
├─────────────────────┬───────────────────┬───────────────────┤
│   Stream Layer      │  Federation Layer │  Integration      │
│                     │                   │                   │
│ ┌─────────────────┐ │ ┌───────────────┐ │ ┌───────────────┐ │
│ │ Event Sourcing  │ │ │ Query Planner │ │ │ Webhooks      │ │
│ │ CQRS           │ │ │ Service Reg.  │ │ │ Bridges       │ │
│ │ Time Travel    │ │ │ Cache Manager │ │ │ Monitoring    │ │
│ └─────────────────┘ │ └───────────────┘ │ └───────────────┘ │
│                     │                   │                   │
│ ┌─────────────────┐ │ ┌───────────────┐ │ ┌───────────────┐ │
│ │ Multi-Backend   │ │ │ GraphQL Fed.  │ │ │ Security      │ │
│ │ • Kafka         │ │ │ SPARQL Fed.   │ │ │ Auth/Authz    │ │
│ │ • NATS          │ │ │ Auto Discovery│ │ │ Encryption    │ │
│ │ • Redis         │ │ │ Load Balancer │ │ │ Audit         │ │
│ │ • Memory        │ │ │ Circuit Break │ │ │ Rate Limiting │ │
│ └─────────────────┘ │ └───────────────┘ │ └───────────────┘ │
└─────────────────────┴───────────────────┴───────────────────┘
```

### Core Components

- **Stream Layer**: Ultra-high performance RDF streaming with multiple backend support
- **Federation Layer**: Distributed query processing across SPARQL and GraphQL services  
- **Integration Layer**: Bridges, webhooks, security, and monitoring

### Performance Characteristics

| Component | Throughput | Latency | Scalability |
|-----------|------------|---------|-------------|
| Stream Processing | 100K+ events/sec | <10ms P99 | Linear to 1000+ partitions |
| Federation Queries | 1K+ queries/sec | <100ms avg | 100+ services |
| Cache Hit Rate | N/A | <5ms | 80%+ hit rate |
| Memory Usage | Efficient | <10GB | 100 services |

---

## 🌊 Stream Processing Tutorials

### Tutorial 1: Basic RDF Streaming

Learn how to stream RDF data in real-time with high performance:

```rust
use oxirs_stream::*;
use chrono::Utc;

async fn basic_rdf_streaming() -> Result<()> {
    // Setup high-performance configuration
    let config = StreamConfig {
        backend: Backend::Kafka,
        topic: "rdf-events".to_string(),
        max_events_per_sec: 50000,
        batch_size: 1000,
        enable_compression: true,
        compression_type: CompressionType::Zstd,
        enable_circuit_breaker: true,
        circuit_breaker_config: CircuitBreakerConfig {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
            ..Default::default()
        },
        ..Default::default()
    };

    let stream_manager = StreamManager::new(config).await?;
    
    // Create producer with connection pooling
    let producer = stream_manager.create_producer_with_pool(
        Backend::Kafka,
        "rdf-events",
        PoolConfig {
            min_connections: 5,
            max_connections: 20,
            connection_timeout: Duration::from_secs(10),
            health_check_interval: Duration::from_secs(30),
            ..Default::default()
        }
    ).await?;

    // Stream RDF patch operations
    for i in 0..10000 {
        let patch_event = Event::RdfPatch(RdfPatchEvent {
            patch_id: uuid::Uuid::new_v4(),
            patch_data: format!(
                "A <http://example.org/entity/{}> <http://xmlns.com/foaf/0.1/name> \"Entity {}\" .",
                i, i
            ),
            metadata: EventMetadata {
                timestamp: Utc::now(),
                source: "tutorial-producer".to_string(),
                correlation_id: Some(uuid::Uuid::new_v4()),
                schema_version: "1.0".to_string(),
                compression: Some(CompressionType::Zstd),
                checksum: None, // Auto-generated
                ..Default::default()
            },
        });

        producer.send(patch_event).await?;

        if i % 1000 == 0 {
            println!("Sent {} events", i);
            producer.flush().await?; // Ensure delivery
        }
    }

    Ok(())
}
```

### Tutorial 2: Event Sourcing with CQRS

Implement event sourcing patterns for audit and replay capabilities:

```rust
use oxirs_stream::{EventSourcingManager, CQRSSystem, Command, Query};

async fn event_sourcing_tutorial() -> Result<()> {
    // Initialize event sourcing
    let event_store = EventSourcingManager::new(EventSourcingConfig {
        storage_backend: StorageBackend::Persistent,
        snapshot_frequency: 1000,
        compression_enabled: true,
        encryption_enabled: true,
        ..Default::default()
    }).await?;

    // Initialize CQRS system
    let cqrs_system = CQRSSystem::new(CQRSConfig {
        command_timeout: Duration::from_secs(30),
        query_timeout: Duration::from_secs(5),
        enable_caching: true,
        cache_ttl: Duration::from_secs(300),
        ..Default::default()
    }).await?;

    // Define a command to create an entity
    let create_command = Command::new(
        "CreateEntity",
        serde_json::json!({
            "entity_id": "user-12345",
            "name": "John Doe",
            "email": "john@example.com",
            "created_at": Utc::now().to_rfc3339()
        })
    );

    // Execute command (writes to event store)
    let command_result = cqrs_system.handle_command(create_command).await?;
    println!("Command executed: {:?}", command_result);

    // Query the read model
    let entity_query = Query::new(
        "GetEntity",
        serde_json::json!({
            "entity_id": "user-12345"
        })
    );

    let query_result = cqrs_system.handle_query(entity_query).await?;
    println!("Query result: {:?}", query_result);

    // Demonstrate event replay
    let events = event_store.get_events_for_aggregate("user-12345").await?;
    println!("Found {} events for entity", events.len());

    // Replay events to rebuild state
    let replayed_state = event_store.replay_events("user-12345", None).await?;
    println!("Replayed state: {:?}", replayed_state);

    Ok(())
}
```

### Tutorial 3: Real-Time Pattern Detection

Implement complex event processing with pattern detection:

```rust
use oxirs_stream::{PatternDetector, WindowAggregator, StreamProcessor};

async fn pattern_detection_tutorial() -> Result<()> {
    // Create stream processor
    let processor = StreamProcessor::new("pattern-detection").await?;

    // Define patterns to detect
    let pattern_detector = PatternDetector::new(vec![
        PatternRule {
            name: "high-frequency-updates".to_string(),
            pattern: "SEQUENCE(A, A, A) WITHIN 1 SECOND".to_string(),
            threshold: 3,
            window: Duration::from_secs(1),
        },
        PatternRule {
            name: "data-quality-issue".to_string(),
            pattern: "ERROR RATE > 5% WITHIN 10 SECONDS".to_string(),
            threshold: 0.05,
            window: Duration::from_secs(10),
        },
        PatternRule {
            name: "anomaly-detection".to_string(),
            pattern: "THROUGHPUT < 50% OF BASELINE".to_string(),
            threshold: 0.5,
            window: Duration::from_secs(30),
        },
    ]);

    // Set up windowed aggregation
    let aggregator = WindowAggregator::new(
        Duration::from_secs(60), // 1-minute windows
        vec![
            AggregationFunction::Count,
            AggregationFunction::Average,
            AggregationFunction::Max,
            AggregationFunction::Min,
            AggregationFunction::StandardDeviation,
        ]
    );

    // Add components to processor
    processor.add_pattern_detector(pattern_detector).await?;
    processor.add_aggregator(aggregator).await?;

    // Set up alert handlers
    processor.on_pattern_detected(|pattern_name, events, metadata| {
        async move {
            println!("🚨 Pattern detected: {}", pattern_name);
            println!("Events involved: {}", events.len());
            println!("Metadata: {:?}", metadata);

            // Send alert to monitoring system
            send_alert(AlertLevel::Warning, &format!("Pattern {} detected", pattern_name)).await;
        }
    }).await;

    // Start processing
    processor.start().await?;

    Ok(())
}
```

### Tutorial 4: Multi-Region Replication

Set up global data replication across multiple regions:

```rust
use oxirs_stream::{MultiRegionReplication, ReplicationConfig, ConflictResolution};

async fn multi_region_tutorial() -> Result<()> {
    // Configure multi-region replication
    let replication_config = ReplicationConfig {
        regions: vec![
            RegionConfig {
                name: "us-east-1".to_string(),
                endpoint: "kafka-us-east-1.example.com:9092".to_string(),
                priority: 1,
                write_preference: WritePreference::Primary,
            },
            RegionConfig {
                name: "eu-west-1".to_string(),
                endpoint: "kafka-eu-west-1.example.com:9092".to_string(),
                priority: 2,
                write_preference: WritePreference::Secondary,
            },
            RegionConfig {
                name: "ap-south-1".to_string(),
                endpoint: "kafka-ap-south-1.example.com:9092".to_string(),
                priority: 3,
                write_preference: WritePreference::ReadOnly,
            },
        ],
        replication_strategy: ReplicationStrategy::AsyncWithVectorClocks,
        conflict_resolution: ConflictResolution::LastWriterWins,
        consistency_level: ConsistencyLevel::EventualConsistency,
        max_replication_lag: Duration::from_secs(5),
        enable_compression: true,
        enable_encryption: true,
        ..Default::default()
    };

    // Initialize multi-region replication
    let replication_manager = MultiRegionReplication::new(replication_config).await?;

    // Start replication
    replication_manager.start().await?;

    // Monitor replication health
    let health_monitor = replication_manager.health_monitor();
    tokio::spawn(async move {
        loop {
            let health = health_monitor.get_health().await;
            println!("Replication health: {:?}", health);
            
            for (region, status) in health.region_status {
                if status.lag > Duration::from_secs(10) {
                    println!("⚠️ High replication lag in region {}: {:?}", region, status.lag);
                }
            }
            
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
    });

    Ok(())
}
```

---

## 🌐 Federation Tutorials

### Tutorial 1: SPARQL Federation Basics

Learn how to execute federated SPARQL queries across multiple endpoints:

```rust
use oxirs_federate::*;

async fn sparql_federation_tutorial() -> Result<()> {
    // Create federation engine with advanced configuration
    let federation_config = FederationConfig {
        cache_config: CacheConfig {
            enable_query_cache: true,
            enable_metadata_cache: true,
            query_cache_ttl: Duration::from_secs(300),
            metadata_cache_ttl: Duration::from_secs(600),
            max_cache_size: 10000,
            cache_compression: true,
            ..Default::default()
        },
        planner_config: PlannerConfig {
            enable_cost_based_optimization: true,
            enable_parallel_execution: true,
            max_concurrent_requests: 50,
            query_timeout: Duration::from_secs(30),
            ..Default::default()
        },
        monitor_config: FederationMonitorConfig {
            enable_metrics: true,
            enable_tracing: true,
            metrics_interval: Duration::from_secs(10),
            ..Default::default()
        },
        ..Default::default()
    };

    let federation_engine = FederationEngine::with_config(federation_config);

    // Register multiple SPARQL services
    let services = vec![
        FederatedService::new_sparql(
            "wikidata".to_string(),
            "Wikidata SPARQL Endpoint".to_string(),
            "https://query.wikidata.org/sparql".to_string(),
        ),
        FederatedService::new_sparql(
            "dbpedia".to_string(),
            "DBpedia SPARQL Endpoint".to_string(),
            "https://dbpedia.org/sparql".to_string(),
        ),
        create_local_sparql_service()?,
    ];

    for service in services {
        federation_engine.register_service(service).await?;
    }

    // Execute complex federated query
    let federated_query = r#"
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?person ?name ?birthPlace ?description WHERE {
            # Get person data from Wikidata
            SERVICE <https://query.wikidata.org/sparql> {
                ?person wdt:P31 wd:Q5 .          # Human
                ?person wdt:P106 wd:Q901 .       # Scientist
                ?person rdfs:label ?name .
                ?person wdt:P19 ?birthPlace .
                FILTER(lang(?name) = "en")
            }
            
            # Get additional info from DBpedia
            SERVICE <https://dbpedia.org/sparql> {
                ?person dbo:abstract ?description .
                FILTER(lang(?description) = "en")
                FILTER(strlen(?description) > 100)
            }
            
            # Filter by local data
            SERVICE <http://localhost:3030/ds/sparql> {
                ?person foaf:knows ?colleague .
                ?colleague foaf:name "Albert Einstein" .
            }
        }
        LIMIT 10
    "#;

    println!("Executing federated query...");
    let start_time = std::time::Instant::now();

    let result = federation_engine.execute_sparql(federated_query).await?;

    println!("✅ Query completed in {:?}", start_time.elapsed());
    println!("📊 Query Statistics:");
    println!("   - Services used: {}", result.metadata.services_used);
    println!("   - Execution time: {:?}", result.metadata.execution_time);
    println!("   - Results count: {}", result.result_count());
    println!("   - Cache hit: {}", result.metadata.cache_hit);
    println!("   - Plan summary: {}", result.metadata.plan_summary);

    // Display results
    if let QueryResult::Sparql(bindings) = result.data {
        for (i, binding) in bindings.iter().enumerate() {
            println!("Result {}:", i + 1);
            for (var, value) in binding {
                println!("  {}: {}", var, value);
            }
            println!();
        }
    }

    Ok(())
}

fn create_local_sparql_service() -> Result<FederatedService> {
    let mut service = FederatedService::new_sparql(
        "local-data".to_string(),
        "Local RDF Dataset".to_string(),
        "http://localhost:3030/ds/sparql".to_string(),
    );

    // Add authentication
    service.auth = Some(AuthConfig {
        auth_type: AuthType::Basic,
        credentials: AuthCredentials {
            username: Some("admin".to_string()),
            password: Some("password".to_string()),
            ..Default::default()
        },
    });

    // Configure rate limiting
    service.rate_limits = Some(RateLimits {
        requests_per_second: 10.0,
        burst_size: 20,
        ..Default::default()
    });

    Ok(service)
}
```

### Tutorial 2: GraphQL Federation

Implement GraphQL federation with schema stitching:

```rust
use oxirs_federate::graphql::*;

async fn graphql_federation_tutorial() -> Result<()> {
    let federation_engine = FederationEngine::new();

    // Register GraphQL services with federation directives
    let user_service = create_user_graphql_service().await?;
    let post_service = create_post_graphql_service().await?;
    let comment_service = create_comment_graphql_service().await?;

    federation_engine.register_service(user_service).await?;
    federation_engine.register_service(post_service).await?;
    federation_engine.register_service(comment_service).await?;

    // Execute federated GraphQL query
    let federated_query = r#"
        query GetUserWithPosts($userId: ID!) {
            user(id: $userId) {
                id
                name
                email
                profile {
                    bio
                    avatar
                }
                posts {
                    id
                    title
                    content
                    publishedAt
                    comments {
                        id
                        content
                        author {
                            name
                        }
                        createdAt
                    }
                    tags {
                        name
                        color
                    }
                }
                followers {
                    id
                    name
                }
                followingCount
            }
        }
    "#;

    let variables = serde_json::json!({
        "userId": "user-12345"
    });

    println!("Executing federated GraphQL query...");
    let result = federation_engine.execute_graphql(federated_query, Some(variables)).await?;

    println!("✅ GraphQL query completed");
    println!("📊 Execution time: {:?}", result.metadata.execution_time);
    println!("Services involved: {}", result.metadata.services_used);

    if let QueryResult::GraphQL(data) = result.data {
        println!("Result: {}", serde_json::to_string_pretty(&data)?);
    }

    Ok(())
}

async fn create_user_graphql_service() -> Result<FederatedService> {
    let mut service = FederatedService::new_graphql(
        "user-service".to_string(),
        "User Management Service".to_string(),
        "http://localhost:4001/graphql".to_string(),
    );

    // Define federation schema
    service.federation_schema = Some(r#"
        type User @key(fields: "id") {
            id: ID!
            name: String!
            email: String!
            profile: UserProfile
            followerCount: Int
            followingCount: Int
        }

        type UserProfile {
            bio: String
            avatar: String
            location: String
        }

        extend type Query {
            user(id: ID!): User
            users(limit: Int): [User!]!
        }
    "#.to_string());

    Ok(service)
}

async fn create_post_graphql_service() -> Result<FederatedService> {
    let mut service = FederatedService::new_graphql(
        "post-service".to_string(),
        "Post Management Service".to_string(),
        "http://localhost:4002/graphql".to_string(),
    );

    service.federation_schema = Some(r#"
        extend type User @key(fields: "id") {
            id: ID! @external
            posts: [Post!]!
        }

        type Post @key(fields: "id") {
            id: ID!
            title: String!
            content: String!
            author: User!
            publishedAt: DateTime!
            tags: [Tag!]!
        }

        type Tag {
            name: String!
            color: String!
        }

        extend type Query {
            post(id: ID!): Post
            posts(authorId: ID, limit: Int): [Post!]!
        }
    "#.to_string());

    Ok(service)
}

async fn create_comment_graphql_service() -> Result<FederatedService> {
    let mut service = FederatedService::new_graphql(
        "comment-service".to_string(),
        "Comment Management Service".to_string(),
        "http://localhost:4003/graphql".to_string(),
    );

    service.federation_schema = Some(r#"
        extend type Post @key(fields: "id") {
            id: ID! @external
            comments: [Comment!]!
        }

        type Comment @key(fields: "id") {
            id: ID!
            content: String!
            author: User!
            post: Post!
            createdAt: DateTime!
            likes: Int!
        }

        extend type User @key(fields: "id") {
            id: ID! @external
            comments: [Comment!]!
        }
    "#.to_string());

    Ok(service)
}
```

### Tutorial 3: Auto-Discovery & Health Monitoring

Implement automatic service discovery and health monitoring:

```rust
use oxirs_federate::{AutoDiscovery, AutoDiscoveryConfig, HealthMonitor};

async fn auto_discovery_tutorial() -> Result<()> {
    let federation_engine = FederationEngine::new();

    // Configure auto-discovery
    let discovery_config = AutoDiscoveryConfig {
        enable_mdns: true,
        enable_dns_discovery: true,
        enable_kubernetes_discovery: true,
        dns_domains: vec![
            "sparql.company.com".to_string(),
            "graphql.company.com".to_string(),
        ],
        service_patterns: vec![
            ServicePattern {
                protocol: "sparql".to_string(),
                port_range: (3030, 3040),
                path: Some("/sparql".to_string()),
            },
            ServicePattern {
                protocol: "graphql".to_string(),
                port_range: (4000, 4010),
                path: Some("/graphql".to_string()),
            },
        ],
        kubernetes_config: Some(KubernetesConfig {
            namespace: "default".to_string(),
            label_selector: "app=sparql-service,app=graphql-service".to_string(),
            annotation_selector: "federation.oxirs.io/enabled=true".to_string(),
        }),
        discovery_interval: Duration::from_secs(30),
        health_check_interval: Duration::from_secs(60),
        max_concurrent_discoveries: 10,
    };

    // Start auto-discovery
    federation_engine.start_auto_discovery(discovery_config).await?;

    // Set up health monitoring
    let health_monitor = HealthMonitor::new(HealthMonitorConfig {
        check_interval: Duration::from_secs(30),
        timeout: Duration::from_secs(10),
        retry_attempts: 3,
        enable_alerts: true,
        alert_thresholds: HealthThresholds {
            max_response_time: Duration::from_millis(5000),
            min_success_rate: 0.95,
            max_error_rate: 0.05,
        },
    });

    // Monitor discovered services
    health_monitor.on_service_unhealthy(|service_id, health_status| {
        async move {
            println!("🚨 Service {} is unhealthy: {:?}", service_id, health_status);
            
            // Automatically attempt recovery
            attempt_service_recovery(service_id).await;
        }
    });

    health_monitor.on_service_recovered(|service_id| {
        async move {
            println!("✅ Service {} has recovered", service_id);
        }
    });

    // Periodically report discovery status
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            match federation_engine.get_auto_discovered_services().await {
                Ok(services) => {
                    println!("📡 Auto-discovered services: {}", services.len());
                    for service in services {
                        println!("  - {} ({}): {}", 
                               service.name, 
                               service.service_type, 
                               service.endpoint);
                    }
                }
                Err(e) => println!("❌ Failed to get discovered services: {}", e),
            }
        }
    });

    Ok(())
}

async fn attempt_service_recovery(service_id: &str) {
    println!("🔧 Attempting recovery for service: {}", service_id);
    
    // Implementation would include:
    // - Restart service containers
    // - Clear caches
    // - Check network connectivity
    // - Validate configuration
    // - Send alerts to operations team
}
```

---

## 🔗 Integration Patterns

### Pattern 1: Stream-Fed Federation

Real-time federation updates driven by streaming events:

```rust
use oxirs_stream::*;
use oxirs_federate::*;

async fn stream_fed_federation_pattern() -> Result<()> {
    // Initialize both stream and federation
    let stream_manager = StreamManager::new(StreamConfig::default()).await?;
    let federation_engine = Arc::new(FederationEngine::new());
    
    // Create stream consumer for RDF updates
    let consumer = stream_manager.create_consumer(Backend::Kafka, "rdf-updates").await?;
    
    // Set up real-time integration
    let federation_clone = federation_engine.clone();
    tokio::spawn(async move {
        while let Some(event) = consumer.consume().await? {
            match event {
                Event::RdfPatch(patch_event) => {
                    // Process RDF patch and invalidate relevant caches
                    process_rdf_patch_for_federation(&federation_clone, &patch_event).await?;
                }
                Event::SparqlUpdate(update_event) => {
                    // Process SPARQL update and refresh materialized views
                    process_sparql_update_for_federation(&federation_clone, &update_event).await?;
                }
                _ => {}
            }
        }
        Ok::<(), anyhow::Error>(())
    });

    // Example: Execute query that benefits from real-time updates
    let query = r#"
        SELECT ?entity ?property ?value ?lastModified WHERE {
            ?entity ?property ?value .
            ?entity <http://purl.org/dc/terms/modified> ?lastModified .
            FILTER(?lastModified > "2025-12-20T00:00:00Z"^^xsd:dateTime)
        }
        ORDER BY DESC(?lastModified)
        LIMIT 100
    "#;

    let result = federation_engine.execute_sparql(query).await?;
    println!("Real-time query result: {} items", result.result_count());

    Ok(())
}

async fn process_rdf_patch_for_federation(
    federation: &FederationEngine, 
    patch: &RdfPatchEvent
) -> Result<()> {
    // Parse the patch to identify affected resources
    let affected_resources = parse_patch_resources(&patch.patch_data)?;
    
    // Invalidate caches for affected resources
    for resource in affected_resources {
        federation.invalidate_resource_cache(&resource).await;
    }
    
    // Update materialized views if needed
    if patch.metadata.schema_version == "2.0" {
        federation.refresh_materialized_views().await?;
    }
    
    // Notify subscribers of changes
    federation.notify_change_subscribers(&patch.patch_id.to_string()).await?;
    
    Ok(())
}
```

### Pattern 2: Webhook Integration

Integrate with external systems via webhooks:

```rust
use oxirs_stream::webhook::*;

async fn webhook_integration_pattern() -> Result<()> {
    let webhook_manager = WebhookManager::new(WebhookConfig {
        max_concurrent_requests: 100,
        timeout: Duration::from_secs(30),
        retry_attempts: 3,
        retry_backoff: BackoffStrategy::Exponential {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
        },
        enable_security: true,
        signature_algorithm: SignatureAlgorithm::HmacSha256,
        ..Default::default()
    }).await?;

    // Register webhooks for different event types
    webhook_manager.register_webhook(
        "data-lake-sync",
        "https://datalake.company.com/webhook/rdf-updates",
        HttpMethod::Post,
        vec!["triple_added".to_string(), "triple_removed".to_string()],
        WebhookAuth::Hmac {
            secret: "shared-secret-key".to_string(),
            header: "X-Hub-Signature-256".to_string(),
        },
    ).await?;

    webhook_manager.register_webhook(
        "analytics-pipeline",
        "https://analytics.company.com/api/events",
        HttpMethod::Post,
        vec!["sparql_update".to_string(), "schema_change".to_string()],
        WebhookAuth::Bearer {
            token: "analytics-api-token".to_string(),
        },
    ).await?;

    webhook_manager.register_webhook(
        "slack-notifications",
        "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
        HttpMethod::Post,
        vec!["error".to_string(), "alert".to_string()],
        WebhookAuth::None,
    ).await?;

    // Set up event handlers
    webhook_manager.on_delivery_success(|webhook_id, response| {
        async move {
            println!("✅ Webhook {} delivered successfully: {}", webhook_id, response.status());
        }
    });

    webhook_manager.on_delivery_failure(|webhook_id, error, retry_count| {
        async move {
            println!("❌ Webhook {} failed (attempt {}): {}", webhook_id, retry_count, error);
        }
    });

    // Connect to stream for automatic webhook triggering
    let stream_manager = StreamManager::new(StreamConfig::default()).await?;
    let consumer = stream_manager.create_consumer(Backend::Kafka, "all-events").await?;

    while let Some(event) = consumer.consume().await? {
        // Transform stream event to webhook payload
        let webhook_payload = transform_event_to_webhook_payload(&event)?;
        
        // Send to appropriate webhooks based on event type
        webhook_manager.trigger_webhooks(&event.event_type(), webhook_payload).await?;
    }

    Ok(())
}

fn transform_event_to_webhook_payload(event: &Event) -> Result<serde_json::Value> {
    match event {
        Event::RdfPatch(patch) => Ok(serde_json::json!({
            "event_type": "rdf_patch",
            "patch_id": patch.patch_id,
            "patch_data": patch.patch_data,
            "timestamp": patch.metadata.timestamp,
            "source": patch.metadata.source,
        })),
        Event::SparqlUpdate(update) => Ok(serde_json::json!({
            "event_type": "sparql_update",
            "query": update.query,
            "timestamp": update.metadata.timestamp,
            "source": update.metadata.source,
        })),
        _ => Ok(serde_json::json!({
            "event_type": "unknown",
            "timestamp": chrono::Utc::now(),
        })),
    }
}
```

### Pattern 3: Microservices Bridge

Bridge between different messaging systems:

```rust
use oxirs_stream::bridge::*;

async fn microservices_bridge_pattern() -> Result<()> {
    let bridge_manager = MessageBridgeManager::new().await?;

    // Bridge between Kafka and HTTP REST APIs
    let kafka_to_rest_bridge = MessageBridge::new(
        BridgeConfig {
            name: "kafka-to-rest".to_string(),
            source: BridgeEndpoint::Kafka {
                topic: "rdf-events".to_string(),
                consumer_group: "rest-bridge".to_string(),
                config: KafkaConfig::default(),
            },
            destination: BridgeEndpoint::Http {
                url: "https://api.company.com/rdf/events".to_string(),
                method: HttpMethod::Post,
                headers: vec![
                    ("Content-Type".to_string(), "application/json".to_string()),
                    ("Authorization".to_string(), "Bearer api-token".to_string()),
                ],
                timeout: Duration::from_secs(30),
            },
            transformer: Some(TransformConfig {
                script: r#"
                    function transform(event) {
                        return {
                            id: event.patch_id,
                            type: 'rdf_patch',
                            data: {
                                patch: event.patch_data,
                                metadata: event.metadata
                            },
                            timestamp: new Date().toISOString()
                        };
                    }
                "#.to_string(),
                language: ScriptLanguage::JavaScript,
            }),
            batch_config: Some(BatchConfig {
                max_batch_size: 100,
                max_batch_time: Duration::from_secs(5),
            }),
            retry_config: RetryConfig {
                max_attempts: 3,
                backoff: BackoffStrategy::Exponential {
                    initial_delay: Duration::from_millis(500),
                    max_delay: Duration::from_secs(30),
                    multiplier: 2.0,
                },
            },
            ..Default::default()
        }
    ).await?;

    // Bridge between NATS and GraphQL subscriptions
    let nats_to_graphql_bridge = MessageBridge::new(
        BridgeConfig {
            name: "nats-to-graphql".to_string(),
            source: BridgeEndpoint::Nats {
                subject: "rdf.updates.>".to_string(),
                queue_group: Some("graphql-bridge".to_string()),
                config: NatsConfig::default(),
            },
            destination: BridgeEndpoint::GraphQL {
                url: "wss://api.company.com/graphql".to_string(),
                subscription: r#"
                    subscription OnRdfUpdate {
                        rdfUpdate {
                            id
                            patch
                            timestamp
                        }
                    }
                "#.to_string(),
                headers: vec![
                    ("Authorization".to_string(), "Bearer graphql-token".to_string()),
                ],
            },
            transformer: Some(TransformConfig {
                script: r#"
                    function transform(event) {
                        return {
                            rdfUpdate: {
                                id: event.patch_id,
                                patch: event.patch_data,
                                timestamp: event.metadata.timestamp
                            }
                        };
                    }
                "#.to_string(),
                language: ScriptLanguage::JavaScript,
            }),
            ..Default::default()
        }
    ).await?;

    // Register bridges with manager
    bridge_manager.add_bridge(kafka_to_rest_bridge).await?;
    bridge_manager.add_bridge(nats_to_graphql_bridge).await?;

    // Start all bridges
    bridge_manager.start_all().await?;

    // Monitor bridge health
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            let health = bridge_manager.get_health_status().await;
            println!("Bridge Health Summary:");
            
            for (bridge_name, status) in health.bridge_status {
                println!("  {} : {:?}", bridge_name, status);
                
                if let BridgeStatus::Error(ref error) = status {
                    println!("    Error: {}", error);
                }
            }
        }
    });

    Ok(())
}
```

---

## 📊 Performance Optimization

### Optimization Guide

#### Stream Performance Tuning

```rust
use oxirs_stream::*;

async fn optimize_stream_performance() -> Result<()> {
    // High-performance configuration
    let optimized_config = StreamConfig {
        backend: Backend::Kafka,
        
        // Throughput optimization
        batch_size: 10000,              // Large batches for high throughput
        max_events_per_sec: 100000,     // Allow high event rates
        
        // Latency optimization
        flush_interval: Duration::from_millis(1), // Aggressive flushing
        
        // Memory optimization
        enable_memory_pooling: true,
        memory_pool_size: 256 * 1024 * 1024, // 256MB pool
        
        // Compression optimization
        enable_compression: true,
        compression_type: CompressionType::Zstd,
        compression_level: 3, // Balance between speed and ratio
        
        // Network optimization
        enable_tcp_nodelay: true,
        socket_buffer_size: 1024 * 1024, // 1MB socket buffers
        
        // Performance monitoring
        enable_performance_monitoring: true,
        performance_sample_rate: 0.1, // Sample 10% for monitoring
        
        // Circuit breaker for reliability
        enable_circuit_breaker: true,
        circuit_breaker_config: CircuitBreakerConfig {
            failure_threshold: 10,
            recovery_timeout: Duration::from_secs(30),
            sample_size: 100,
        },
        
        ..Default::default()
    };

    let stream_manager = StreamManager::new(optimized_config).await?;
    
    // Use performance optimizer
    let optimizer = PerformanceOptimizer::new(OptimizerConfig {
        target_latency: Duration::from_millis(5),
        target_throughput: 100000,
        enable_adaptive_batching: true,
        enable_auto_scaling: true,
        enable_predictive_optimization: true,
    });

    let producer = stream_manager.create_optimized_producer(
        Backend::Kafka, 
        "high-perf-topic", 
        optimizer
    ).await?;

    // Benchmark throughput
    let start_time = std::time::Instant::now();
    let event_count = 100000;

    for i in 0..event_count {
        let event = Event::RdfPatch(RdfPatchEvent {
            patch_id: uuid::Uuid::new_v4(),
            patch_data: format!("A <http://example.org/entity/{}> <http://xmlns.com/foaf/0.1/name> \"Entity {}\" .", i, i),
            metadata: EventMetadata::default(),
        });

        producer.send(event).await?;
    }

    producer.flush().await?;
    let duration = start_time.elapsed();
    
    println!("📊 Performance Results:");
    println!("  Events: {}", event_count);
    println!("  Duration: {:?}", duration);
    println!("  Throughput: {:.2} events/sec", event_count as f64 / duration.as_secs_f64());
    
    Ok(())
}
```

#### Federation Performance Tuning

```rust
use oxirs_federate::*;

async fn optimize_federation_performance() -> Result<()> {
    // High-performance federation configuration
    let federation_config = FederationConfig {
        // Cache optimization
        cache_config: CacheConfig {
            enable_query_cache: true,
            enable_metadata_cache: true,
            enable_result_cache: true,
            query_cache_size: 10000,
            metadata_cache_size: 5000,
            result_cache_size: 20000,
            query_cache_ttl: Duration::from_secs(600),
            use_adaptive_ttl: true,
            cache_compression: true,
            enable_distributed_cache: true,
        },
        
        // Query optimization
        planner_config: PlannerConfig {
            enable_cost_based_optimization: true,
            enable_join_optimization: true,
            enable_filter_pushdown: true,
            enable_projection_pushdown: true,
            enable_parallel_execution: true,
            max_concurrent_requests: 100,
            enable_query_rewriting: true,
            enable_materialized_views: true,
        },
        
        // Connection optimization
        executor_config: FederatedExecutorConfig {
            connection_pool_size: 50,
            connection_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            enable_connection_pooling: true,
            enable_keep_alive: true,
            max_retries: 3,
        },
        
        // Network optimization
        network_config: NetworkConfig {
            enable_compression: true,
            compression_threshold: 1024, // Compress responses > 1KB
            enable_pipelining: true,
            max_pipeline_requests: 10,
            tcp_no_delay: true,
            buffer_size: 64 * 1024, // 64KB buffers
        },
        
        ..Default::default()
    };

    let federation_engine = FederationEngine::with_config(federation_config);
    
    // Register optimized services
    let services = create_optimized_services().await?;
    for service in services {
        federation_engine.register_service(service).await?;
    }

    // Benchmark query performance
    let queries = vec![
        "SELECT * WHERE { ?s ?p ?o } LIMIT 100",
        "SELECT ?name WHERE { ?person foaf:name ?name } LIMIT 50",
        "SELECT ?subject ?predicate WHERE { ?subject ?predicate ?object . FILTER(?predicate != rdf:type) } LIMIT 200",
    ];

    for (i, query) in queries.iter().enumerate() {
        let start_time = std::time::Instant::now();
        
        let result = federation_engine.execute_sparql(query).await?;
        
        let duration = start_time.elapsed();
        
        println!("Query {} Performance:", i + 1);
        println!("  Execution time: {:?}", duration);
        println!("  Results: {}", result.result_count());
        println!("  Cache hit: {}", result.metadata.cache_hit);
        println!("  Services used: {}", result.metadata.services_used);
        println!();
    }

    Ok(())
}

async fn create_optimized_services() -> Result<Vec<FederatedService>> {
    Ok(vec![
        create_optimized_service("fast-sparql", "http://fast.sparql.endpoint").await?,
        create_optimized_service("cached-sparql", "http://cached.sparql.endpoint").await?,
        create_optimized_service("local-sparql", "http://localhost:3030/sparql").await?,
    ])
}

async fn create_optimized_service(id: &str, endpoint: &str) -> Result<FederatedService> {
    let mut service = FederatedService::new_sparql(
        id.to_string(),
        format!("Optimized {}", id),
        endpoint.to_string(),
    );

    // Optimize service configuration
    service.connection_config = Some(ConnectionConfig {
        pool_size: 20,
        max_idle_connections: 5,
        connection_timeout: Duration::from_secs(5),
        idle_timeout: Duration::from_secs(60),
        enable_keep_alive: true,
        tcp_no_delay: true,
    });

    service.performance_config = Some(PerformanceConfig {
        enable_compression: true,
        enable_caching: true,
        cache_ttl: Duration::from_secs(300),
        enable_batching: true,
        batch_size: 100,
        batch_timeout: Duration::from_millis(10),
    });

    Ok(service)
}
```

---

## 🔐 Security Best Practices

### Authentication & Authorization

```rust
use oxirs_stream::security::*;

async fn implement_security_best_practices() -> Result<()> {
    // Initialize security manager with comprehensive configuration
    let security_config = SecurityConfig {
        // Authentication configuration
        auth_config: AuthConfig {
            enable_multi_factor: true,
            require_strong_passwords: true,
            password_min_length: 12,
            session_timeout: Duration::from_secs(3600), // 1 hour
            max_concurrent_sessions: 5,
            enable_session_monitoring: true,
        },
        
        // Authorization configuration
        authz_config: AuthzConfig {
            enable_rbac: true,
            enable_abac: true,
            default_deny: true,
            enable_audit_logging: true,
            policy_cache_ttl: Duration::from_secs(300),
        },
        
        // Encryption configuration
        encryption_config: EncryptionConfig {
            algorithm: EncryptionAlgorithm::AES256GCM,
            key_rotation_interval: Duration::from_secs(86400 * 30), // 30 days
            enable_envelope_encryption: true,
            enable_field_level_encryption: true,
        },
        
        // TLS configuration
        tls_config: TlsConfig {
            min_version: TlsVersion::V1_3,
            cipher_suites: vec![
                CipherSuite::TLS13_AES_256_GCM_SHA384,
                CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
            ],
            enable_client_cert_auth: true,
            verify_hostname: true,
        },
        
        ..Default::default()
    };

    let security_manager = SecurityManager::new(security_config).await?;

    // Set up role-based access control
    security_manager.create_role("stream_admin", vec![
        Permission::StreamCreate,
        Permission::StreamRead,
        Permission::StreamWrite,
        Permission::StreamDelete,
        Permission::ConfigRead,
        Permission::ConfigWrite,
    ]).await?;

    security_manager.create_role("stream_user", vec![
        Permission::StreamRead,
        Permission::StreamWrite,
    ]).await?;

    security_manager.create_role("federation_admin", vec![
        Permission::FederationQuery,
        Permission::FederationManage,
        Permission::ServiceRegister,
        Permission::ServiceUnregister,
    ]).await?;

    // Set up attribute-based access control policies
    security_manager.create_abac_policy("data_sensitivity", r#"
        permit(principal, action, resource) :-
            principal.clearance_level >= resource.classification_level,
            principal.department = resource.owning_department OR
            resource.shared_departments contains principal.department
    "#).await?;

    security_manager.create_abac_policy("time_based_access", r#"
        permit(principal, action, resource) :-
            time.hour >= 6,
            time.hour <= 22,
            time.day_of_week != "saturday",
            time.day_of_week != "sunday"
    "#).await?;

    // Demonstrate secure operations
    demonstrate_secure_streaming(&security_manager).await?;
    demonstrate_secure_federation(&security_manager).await?;

    Ok(())
}

async fn demonstrate_secure_streaming(security_manager: &SecurityManager) -> Result<()> {
    // Create authenticated user session
    let user_session = security_manager.authenticate(
        "alice@company.com",
        "SecureP@ssw0rd123!",
        Some(MultiFactorAuth::TOTP("123456".to_string()))
    ).await?;

    // Check authorization for streaming operation
    let stream_permission = security_manager.authorize(
        &user_session,
        Permission::StreamWrite,
        &ResourceContext {
            resource_type: "stream".to_string(),
            resource_id: "sensitive-data-stream".to_string(),
            classification_level: 3,
            owning_department: "research".to_string(),
            shared_departments: vec!["engineering".to_string()],
        }
    ).await?;

    if stream_permission.is_allowed() {
        // Create encrypted stream
        let stream_config = StreamConfig {
            backend: Backend::Kafka,
            enable_encryption: true,
            encryption_key_id: Some("stream-encryption-key-v1".to_string()),
            enable_field_encryption: true,
            encrypted_fields: vec!["email".to_string(), "ssn".to_string()],
            ..Default::default()
        };

        let stream_manager = StreamManager::new_with_security(stream_config, security_manager.clone()).await?;
        let producer = stream_manager.create_secure_producer(
            Backend::Kafka,
            "sensitive-data-stream",
            &user_session
        ).await?;

        // Send encrypted event
        let sensitive_event = Event::RdfPatch(RdfPatchEvent {
            patch_id: uuid::Uuid::new_v4(),
            patch_data: "A <http://example.org/person/123> <http://xmlns.com/foaf/0.1/mbox> \"alice@company.com\" .".to_string(),
            metadata: EventMetadata {
                security_context: Some(SecurityContext {
                    classification_level: 3,
                    access_controls: vec!["RESEARCH_ONLY".to_string()],
                    encryption_required: true,
                }),
                ..Default::default()
            },
        });

        producer.send_secure(sensitive_event, &user_session).await?;
        println!("✅ Sensitive data streamed securely");
    }

    Ok(())
}

async fn demonstrate_secure_federation(security_manager: &SecurityManager) -> Result<()> {
    // Create service account for federation
    let service_session = security_manager.authenticate_service_account(
        "federation-service",
        "service-account-token",
        &["federation_admin".to_string()]
    ).await?;

    // Create secure federation engine
    let federation_config = FederationConfig {
        security_config: FederationSecurityConfig {
            enable_query_authorization: true,
            enable_result_filtering: true,
            enable_audit_logging: true,
            require_tls: true,
            verify_service_certificates: true,
        },
        ..Default::default()
    };

    let federation_engine = FederationEngine::new_with_security(
        federation_config, 
        security_manager.clone()
    ).await?;

    // Execute query with security context
    let secure_query = r#"
        SELECT ?person ?email WHERE {
            ?person foaf:name ?name .
            ?person foaf:mbox ?email .
            FILTER(?name = "Alice Smith")
        }
    "#;

    let query_context = QuerySecurityContext {
        user_session: service_session,
        required_permissions: vec![Permission::DataRead],
        classification_level: 2,
        purpose: "Analytics".to_string(),
    };

    let result = federation_engine.execute_sparql_secure(
        secure_query, 
        &query_context
    ).await?;

    println!("✅ Secure federated query executed");
    println!("  Results (filtered): {}", result.result_count());

    Ok(())
}
```

### Data Privacy & Compliance

```rust
use oxirs_stream::privacy::*;

async fn implement_privacy_compliance() -> Result<()> {
    // Initialize privacy manager for GDPR/CCPA compliance
    let privacy_manager = PrivacyManager::new(PrivacyConfig {
        enable_data_minimization: true,
        enable_purpose_limitation: true,
        enable_consent_management: true,
        enable_right_to_erasure: true,
        enable_data_portability: true,
        retention_policies: vec![
            RetentionPolicy {
                data_category: "personal_data".to_string(),
                retention_period: Duration::from_secs(86400 * 365 * 7), // 7 years
                auto_delete: true,
            },
            RetentionPolicy {
                data_category: "analytics_data".to_string(),
                retention_period: Duration::from_secs(86400 * 365 * 2), // 2 years
                auto_delete: true,
            },
        ],
        anonymization_config: AnonymizationConfig {
            enable_k_anonymity: true,
            k_value: 5,
            enable_differential_privacy: true,
            privacy_budget: 1.0,
            noise_scale: 0.1,
        },
        ..Default::default()
    }).await?;

    // Demonstrate privacy-preserving operations
    demonstrate_consent_management(&privacy_manager).await?;
    demonstrate_data_anonymization(&privacy_manager).await?;
    demonstrate_right_to_erasure(&privacy_manager).await?;

    Ok(())
}

async fn demonstrate_consent_management(privacy_manager: &PrivacyManager) -> Result<()> {
    // Record user consent
    let consent = ConsentRecord {
        subject_id: "user-12345".to_string(),
        purposes: vec![
            DataPurpose::Analytics,
            DataPurpose::PersonalizationRecommendations,
        ],
        legal_basis: LegalBasis::Consent,
        consent_timestamp: chrono::Utc::now(),
        expiry_date: Some(chrono::Utc::now() + chrono::Duration::days(365)),
        withdrawal_instructions: "Email privacy@company.com".to_string(),
    };

    privacy_manager.record_consent(&consent).await?;

    // Check consent before processing
    let processing_request = ProcessingRequest {
        subject_id: "user-12345".to_string(),
        purpose: DataPurpose::Analytics,
        data_categories: vec!["browsing_history".to_string(), "preferences".to_string()],
    };

    if privacy_manager.check_consent(&processing_request).await? {
        println!("✅ Consent verified for analytics processing");
    } else {
        println!("❌ No valid consent for requested processing");
    }

    Ok(())
}

async fn demonstrate_data_anonymization(privacy_manager: &PrivacyManager) -> Result<()> {
    // Original RDF data with personal information
    let personal_data = r#"
        <http://example.org/person/123> foaf:name "John Smith" .
        <http://example.org/person/123> foaf:age "35" .
        <http://example.org/person/123> foaf:mbox "john.smith@email.com" .
        <http://example.org/person/123> ex:zipCode "12345" .
        <http://example.org/person/123> ex:salary "75000" .
    "#;

    // Anonymize the data
    let anonymized_data = privacy_manager.anonymize_rdf(
        personal_data,
        &AnonymizationRequest {
            techniques: vec![
                AnonymizationTechnique::KAnonymity { k: 5 },
                AnonymizationTechnique::LDiversity { l: 3 },
                AnonymizationTechnique::Generalization {
                    fields: vec!["zipCode".to_string(), "age".to_string()],
                },
                AnonymizationTechnique::Suppression {
                    fields: vec!["mbox".to_string()],
                },
            ],
            preserve_utility: true,
            target_privacy_level: PrivacyLevel::High,
        }
    ).await?;

    println!("Anonymized RDF data:");
    println!("{}", anonymized_data);

    Ok(())
}

async fn demonstrate_right_to_erasure(privacy_manager: &PrivacyManager) -> Result<()> {
    // Process erasure request (GDPR Article 17)
    let erasure_request = ErasureRequest {
        subject_id: "user-12345".to_string(),
        reason: ErasureReason::ConsentWithdrawn,
        scope: ErasureScope::Complete,
        verify_identity: true,
        request_timestamp: chrono::Utc::now(),
    };

    let erasure_result = privacy_manager.process_erasure_request(&erasure_request).await?;
    
    println!("✅ Erasure request processed:");
    println!("  Records deleted: {}", erasure_result.records_deleted);
    println!("  Systems updated: {}", erasure_result.systems_updated.len());
    println!("  Completion time: {:?}", erasure_result.completion_time);

    Ok(())
}
```

---

## 📈 Monitoring & Observability

### Comprehensive Monitoring Setup

```rust
use oxirs_stream::monitoring_dashboard::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize comprehensive monitoring dashboard
    let server_config = ServerConfig {
        host: "0.0.0.0".to_string(),
        port: 8080,
        enable_cors: true,
        static_files_dir: Some("./dashboard/static".to_string()),
        api_prefix: "/api/v1".to_string(),
    };

    let metrics_config = MetricsConfig {
        collection_interval: Duration::from_secs(5),
        retention_period: Duration::from_secs(86400 * 7), // 7 days
        enable_detailed_metrics: true,
        enable_profiling: true,
        alert_thresholds: AlertThresholds {
            stream_max_latency_ms: 100.0,
            stream_min_throughput: 1000.0,
            stream_max_error_rate: 0.05,
            federation_max_response_time_ms: 200.0,
            federation_min_cache_hit_rate: 0.8,
            system_max_cpu_percent: 80.0,
            system_max_memory_percent: 85.0,
            integration_max_latency_ms: 150.0,
        },
    };

    // Create and start monitoring dashboard
    let dashboard = OxiRSMonitoringDashboard::new(server_config, metrics_config).await?;
    
    println!("🚀 Starting OxiRS Monitoring Dashboard");
    println!("📊 Dashboard URL: http://localhost:8080/dashboard");
    println!("🔧 API Endpoints: http://localhost:8080/api/v1");
    
    dashboard.start().await?;

    Ok(())
}
```

The monitoring dashboard provides:

- **Real-time Metrics**: Live streaming of performance data
- **Historical Analytics**: Time-series data analysis and trends  
- **Alert Management**: Configurable alerts and notifications
- **Health Monitoring**: System health scores and diagnostics
- **Interactive Dashboards**: Web-based interface with charts

### Custom Metrics Integration

```rust
use oxirs_stream::monitoring::*;

async fn setup_custom_monitoring() -> Result<()> {
    // Create custom metrics collector
    let metrics_collector = CustomMetricsCollector::new(MetricsConfig {
        enable_business_metrics: true,
        enable_technical_metrics: true,
        enable_security_metrics: true,
        custom_metric_definitions: vec![
            MetricDefinition {
                name: "rdf_data_quality_score".to_string(),
                metric_type: MetricType::Gauge,
                description: "Data quality score based on RDF validation".to_string(),
                labels: vec!["dataset".to_string(), "schema".to_string()],
            },
            MetricDefinition {
                name: "federated_query_complexity".to_string(),
                metric_type: MetricType::Histogram,
                description: "Complexity score of federated queries".to_string(),
                labels: vec!["query_type".to_string(), "service_count".to_string()],
            },
            MetricDefinition {
                name: "security_threat_level".to_string(),
                metric_type: MetricType::Counter,
                description: "Security threat detection events".to_string(),
                labels: vec!["threat_type".to_string(), "severity".to_string()],
            },
        ],
    }).await?;

    // Set up metric collection hooks
    metrics_collector.on_rdf_event(|event| {
        async move {
            // Calculate data quality metrics
            let quality_score = calculate_data_quality(&event).await;
            metrics_collector.record_gauge(
                "rdf_data_quality_score",
                quality_score,
                &[("dataset", &event.dataset), ("schema", &event.schema_version)]
            ).await;
        }
    });

    metrics_collector.on_federation_query(|query, result| {
        async move {
            // Calculate query complexity
            let complexity = calculate_query_complexity(&query).await;
            metrics_collector.record_histogram(
                "federated_query_complexity",
                complexity,
                &[("query_type", &query.query_type), ("service_count", &result.services_used.to_string())]
            ).await;
        }
    });

    metrics_collector.on_security_event(|event| {
        async move {
            // Record security metrics
            metrics_collector.record_counter(
                "security_threat_level",
                1.0,
                &[("threat_type", &event.threat_type), ("severity", &event.severity)]
            ).await;
        }
    });

    Ok(())
}

async fn calculate_data_quality(event: &RdfEvent) -> f64 {
    // Implement data quality scoring logic
    let mut quality_score = 1.0;
    
    // Check for required properties
    if !event.has_required_properties() {
        quality_score -= 0.2;
    }
    
    // Check for data consistency
    if !event.is_consistent() {
        quality_score -= 0.3;
    }
    
    // Check for proper formatting
    if !event.is_well_formed() {
        quality_score -= 0.1;
    }
    
    quality_score.max(0.0)
}

async fn calculate_query_complexity(query: &FederatedQuery) -> f64 {
    let mut complexity = 0.0;
    
    // Count triple patterns
    complexity += query.triple_patterns.len() as f64 * 1.0;
    
    // Count joins
    complexity += query.joins.len() as f64 * 2.0;
    
    // Count services
    complexity += query.services.len() as f64 * 3.0;
    
    // Count filters
    complexity += query.filters.len() as f64 * 1.5;
    
    // Add complexity for optional patterns
    complexity += query.optional_patterns.len() as f64 * 2.5;
    
    complexity
}
```

---

## 🏭 Production Deployment

### Kubernetes Deployment

```yaml
# oxirs-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-stream
  labels:
    app: oxirs-stream
    version: v1.2.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oxirs-stream
  template:
    metadata:
      labels:
        app: oxirs-stream
        version: v1.2.0
    spec:
      containers:
      - name: oxirs-stream
        image: oxirs/stream:v1.2.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: OXIRS_BACKEND
          value: "kafka"
        - name: KAFKA_BROKERS
          value: "kafka-cluster:9092"
        - name: OXIRS_LOG_LEVEL
          value: "info"
        - name: OXIRS_METRICS_ENABLED
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /etc/oxirs/config.toml
          subPath: config.toml
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: oxirs-config

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-federate
  labels:
    app: oxirs-federate
    version: v1.2.0
spec:
  replicas: 2
  selector:
    matchLabels:
      app: oxirs-federate
  template:
    metadata:
      labels:
        app: oxirs-federate
        version: v1.2.0
    spec:
      containers:
      - name: oxirs-federate
        image: oxirs/federate:v1.2.0
        ports:
        - containerPort: 8081
          name: http
        - containerPort: 9091
          name: metrics
        env:
        - name: OXIRS_CACHE_ENABLED
          value: "true"
        - name: REDIS_URL
          value: "redis://redis-cluster:6379"
        - name: OXIRS_AUTO_DISCOVERY
          value: "true"
        resources:
          requests:
            memory: "1Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: oxirs-stream-service
spec:
  selector:
    app: oxirs-stream
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer

---
apiVersion: v1
kind: Service
metadata:
  name: oxirs-federate-service
spec:
  selector:
    app: oxirs-federate
  ports:
  - name: http
    port: 80
    targetPort: 8081
  - name: metrics
    port: 9091
    targetPort: 9091
  type: LoadBalancer

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: oxirs-config
data:
  config.toml: |
    [stream]
    backend = "kafka"
    max_events_per_sec = 100000
    enable_compression = true
    enable_monitoring = true
    
    [stream.kafka]
    brokers = ["kafka-cluster:9092"]
    topic = "rdf-events"
    consumer_group = "oxirs-consumers"
    
    [federation]
    enable_cache = true
    cache_ttl = 300
    max_concurrent_queries = 100
    enable_auto_discovery = true
    
    [security]
    enable_authentication = true
    enable_authorization = true
    enable_encryption = true
    
    [monitoring]
    enable_metrics = true
    metrics_port = 9090
    enable_tracing = true
    
    [logging]
    level = "info"
    format = "json"
```

### Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  oxirs-stream:
    build:
      context: .
      dockerfile: Dockerfile.stream
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      - OXIRS_BACKEND=kafka
      - KAFKA_BROKERS=kafka:9092
      - RUST_LOG=info
    depends_on:
      - kafka
      - redis
    volumes:
      - ./config:/etc/oxirs
    networks:
      - oxirs-network

  oxirs-federate:
    build:
      context: .
      dockerfile: Dockerfile.federate
    ports:
      - "8081:8081"
      - "9091:9091"
    environment:
      - REDIS_URL=redis://redis:6379
      - RUST_LOG=info
    depends_on:
      - redis
    volumes:
      - ./config:/etc/oxirs
    networks:
      - oxirs-network

  oxirs-dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "3000:3000"
    environment:
      - OXIRS_STREAM_URL=http://oxirs-stream:8080
      - OXIRS_FEDERATE_URL=http://oxirs-federate:8081
    depends_on:
      - oxirs-stream
      - oxirs-federate
    networks:
      - oxirs-network

  kafka:
    image: confluentinc/cp-kafka:latest
    ports:
      - "9092:9092"
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    depends_on:
      - zookeeper
    networks:
      - oxirs-network

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    ports:
      - "2181:2181"
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    networks:
      - oxirs-network

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - oxirs-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - oxirs-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    networks:
      - oxirs-network

volumes:
  redis_data:
  grafana_data:

networks:
  oxirs-network:
    driver: bridge
```

### Production Configuration

```toml
# production.toml
[server]
host = "0.0.0.0"
port = 8080
workers = 16
enable_cors = false
request_timeout = 30
max_connections = 10000

[stream]
backend = "kafka"
max_events_per_sec = 100000
batch_size = 10000
flush_interval = 1
enable_compression = true
compression_type = "zstd"
compression_level = 3
enable_circuit_breaker = true
enable_metrics = true

[stream.kafka]
brokers = [
    "kafka-1.internal:9092",
    "kafka-2.internal:9092", 
    "kafka-3.internal:9092"
]
topic = "rdf-events-prod"
consumer_group = "oxirs-prod-consumers"
batch_size = 10000
linger_ms = 1
compression_type = "zstd"
acks = "all"
retries = 2147483647
enable_idempotence = true

[federation]
enable_cache = true
cache_ttl = 600
max_cache_size = 100000
enable_distributed_cache = true
redis_cluster = [
    "redis-1.internal:6379",
    "redis-2.internal:6379",
    "redis-3.internal:6379"
]
max_concurrent_queries = 200
query_timeout = 30
enable_auto_discovery = true
enable_load_balancing = true

[security]
enable_authentication = true
enable_authorization = true
enable_encryption = true
jwt_secret = "${JWT_SECRET}"
encryption_key = "${ENCRYPTION_KEY}"
tls_cert_path = "/etc/ssl/certs/oxirs.crt"
tls_key_path = "/etc/ssl/private/oxirs.key"

[monitoring]
enable_metrics = true
metrics_port = 9090
enable_tracing = true
tracing_endpoint = "http://jaeger:14268/api/traces"
enable_profiling = false
log_level = "info"
log_format = "json"

[database]
url = "${DATABASE_URL}"
max_connections = 50
min_connections = 5
connection_timeout = 30
idle_timeout = 600

[performance]
enable_memory_pool = true
memory_pool_size = 1073741824  # 1GB
enable_zero_copy = true
enable_numa_awareness = true
gc_target_percentage = 10
```

---

## 🧪 Testing Strategies

### Integration Testing

```rust
use oxirs_stream::*;
use oxirs_federate::*;

#[cfg(test)]
mod integration_tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_end_to_end_stream_federation() {
        // Set up test environment
        let test_env = TestEnvironment::new().await.unwrap();
        
        // Initialize stream and federation components
        let stream_manager = test_env.create_stream_manager().await.unwrap();
        let federation_engine = test_env.create_federation_engine().await.unwrap();
        
        // Register test services
        let test_services = test_env.create_test_services().await.unwrap();
        for service in test_services {
            federation_engine.register_service(service).await.unwrap();
        }
        
        // Test streaming RDF data
        let producer = stream_manager.create_producer(Backend::Memory, "test-topic").await.unwrap();
        let consumer = stream_manager.create_consumer(Backend::Memory, "test-topic").await.unwrap();
        
        // Send test RDF patch
        let test_patch = Event::RdfPatch(RdfPatchEvent {
            patch_id: uuid::Uuid::new_v4(),
            patch_data: "A <http://test.org/entity> <http://xmlns.com/foaf/0.1/name> \"Test Entity\" .".to_string(),
            metadata: EventMetadata::default(),
        });
        
        producer.send(test_patch.clone()).await.unwrap();
        
        // Consume and verify
        let received = consumer.consume().await.unwrap().unwrap();
        assert_eq!(format!("{:?}", test_patch), format!("{:?}", received));
        
        // Test federated query
        let test_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10";
        let result = federation_engine.execute_sparql(test_query).await.unwrap();
        
        assert!(result.is_success());
        assert!(result.metadata.execution_time.as_millis() < 1000);
        
        test_env.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_high_throughput_streaming() {
        let stream_manager = StreamManager::new(StreamConfig {
            backend: Backend::Memory,
            max_events_per_sec: 50000,
            enable_performance_optimization: true,
            ..Default::default()
        }).await.unwrap();
        
        let producer = stream_manager.create_producer(Backend::Memory, "perf-test").await.unwrap();
        let consumer = stream_manager.create_consumer(Backend::Memory, "perf-test").await.unwrap();
        
        // Performance test
        let event_count = 10000;
        let start_time = std::time::Instant::now();
        
        // Send events
        for i in 0..event_count {
            let event = Event::RdfPatch(RdfPatchEvent {
                patch_id: uuid::Uuid::new_v4(),
                patch_data: format!("A <http://test.org/entity/{}> <http://xmlns.com/foaf/0.1/name> \"Entity {}\" .", i, i),
                metadata: EventMetadata::default(),
            });
            
            producer.send(event).await.unwrap();
        }
        
        producer.flush().await.unwrap();
        
        // Consume events
        let mut received_count = 0;
        for _ in 0..event_count {
            if consumer.consume().await.unwrap().is_some() {
                received_count += 1;
            }
        }
        
        let duration = start_time.elapsed();
        let throughput = event_count as f64 / duration.as_secs_f64();
        
        assert_eq!(received_count, event_count);
        assert!(throughput > 5000.0, "Throughput {} events/sec is below target", throughput);
        println!("Achieved throughput: {:.2} events/sec", throughput);
    }

    #[tokio::test]
    async fn test_federation_fault_tolerance() {
        let federation_engine = FederationEngine::new();
        
        // Register services with different reliability
        let reliable_service = create_test_service("reliable", true).await;
        let unreliable_service = create_test_service("unreliable", false).await;
        
        federation_engine.register_service(reliable_service).await.unwrap();
        federation_engine.register_service(unreliable_service).await.unwrap();
        
        // Execute query that should gracefully handle service failures
        let federated_query = r#"
            SELECT ?data WHERE {
                {
                    SERVICE <http://reliable.test/sparql> {
                        ?s ?p ?data .
                    }
                }
                UNION
                {
                    SERVICE <http://unreliable.test/sparql> {
                        ?s ?p ?data .
                    }
                }
            }
        "#;
        
        let result = federation_engine.execute_sparql(federated_query).await.unwrap();
        
        // Should succeed with partial results
        assert!(result.is_success());
        assert!(!result.errors.is_empty()); // Should have errors from unreliable service
        assert!(result.result_count() > 0); // Should have results from reliable service
    }

    async fn create_test_service(name: &str, reliable: bool) -> FederatedService {
        let mut service = FederatedService::new_sparql(
            name.to_string(),
            format!("Test {} Service", name),
            format!("http://{}.test/sparql", name),
        );
        
        if !reliable {
            // Configure as unreliable for testing
            service.health_check_config = Some(HealthCheckConfig {
                enabled: true,
                interval: Duration::from_secs(5),
                timeout: Duration::from_secs(1),
                failure_threshold: 1,
            });
        }
        
        service
    }
}

struct TestEnvironment {
    temp_dir: tempfile::TempDir,
    test_services: Vec<TestService>,
}

impl TestEnvironment {
    async fn new() -> Result<Self> {
        let temp_dir = tempfile::tempdir()?;
        Ok(Self {
            temp_dir,
            test_services: Vec::new(),
        })
    }
    
    async fn create_stream_manager(&self) -> Result<StreamManager> {
        let config = StreamConfig {
            backend: Backend::Memory,
            enable_performance_optimization: true,
            ..Default::default()
        };
        StreamManager::new(config).await
    }
    
    async fn create_federation_engine(&self) -> Result<FederationEngine> {
        Ok(FederationEngine::new())
    }
    
    async fn create_test_services(&mut self) -> Result<Vec<FederatedService>> {
        // Start mock SPARQL services
        let services = vec![
            self.start_mock_sparql_service("test-sparql-1", 3001).await?,
            self.start_mock_sparql_service("test-sparql-2", 3002).await?,
        ];
        
        Ok(services)
    }
    
    async fn start_mock_sparql_service(&mut self, name: &str, port: u16) -> Result<FederatedService> {
        let service = MockSparqlService::new(port).await?;
        self.test_services.push(service);
        
        Ok(FederatedService::new_sparql(
            name.to_string(),
            format!("Mock {} Service", name),
            format!("http://localhost:{}/sparql", port),
        ))
    }
    
    async fn cleanup(&self) -> Result<()> {
        // Cleanup test services and temporary files
        for service in &self.test_services {
            service.stop().await?;
        }
        Ok(())
    }
}

struct MockSparqlService {
    port: u16,
    server_handle: Option<tokio::task::JoinHandle<()>>,
}

impl MockSparqlService {
    async fn new(port: u16) -> Result<TestService> {
        // Start a mock HTTP server that responds to SPARQL queries
        let app = axum::Router::new()
            .route("/sparql", axum::routing::post(mock_sparql_handler))
            .route("/health", axum::routing::get(|| async { "OK" }));
        
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
        let server = axum::Server::bind(&addr).serve(app.into_make_service());
        
        let handle = tokio::spawn(async move {
            if let Err(e) = server.await {
                eprintln!("Mock server error: {}", e);
            }
        });
        
        // Give server time to start
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        Ok(TestService::MockSparql(MockSparqlService {
            port,
            server_handle: Some(handle),
        }))
    }
    
    async fn stop(&self) -> Result<()> {
        if let Some(handle) = &self.server_handle {
            handle.abort();
        }
        Ok(())
    }
}

async fn mock_sparql_handler(body: String) -> String {
    // Return mock SPARQL results
    serde_json::json!({
        "head": {
            "vars": ["s", "p", "o"]
        },
        "results": {
            "bindings": [
                {
                    "s": {"type": "uri", "value": "http://test.org/subject"},
                    "p": {"type": "uri", "value": "http://test.org/predicate"},
                    "o": {"type": "literal", "value": "test object"}
                }
            ]
        }
    }).to_string()
}

enum TestService {
    MockSparql(MockSparqlService),
}

impl TestService {
    async fn stop(&self) -> Result<()> {
        match self {
            TestService::MockSparql(service) => service.stop().await,
        }
    }
}
```

### Load Testing

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

fn benchmark_streaming_performance(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("streaming_performance");
    
    for event_size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Bytes(*event_size as u64));
        group.bench_with_input(
            BenchmarkId::new("event_size", event_size),
            event_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let stream_manager = StreamManager::new(StreamConfig::default()).await.unwrap();
                    let producer = stream_manager.create_producer(Backend::Memory, "bench-topic").await.unwrap();
                    
                    let large_data = "x".repeat(size);
                    let event = Event::RdfPatch(RdfPatchEvent {
                        patch_id: uuid::Uuid::new_v4(),
                        patch_data: format!("A <http://test.org/subject> <http://test.org/predicate> \"{}\" .", large_data),
                        metadata: EventMetadata::default(),
                    });
                    
                    producer.send(event).await.unwrap();
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_federation_performance(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("federation_performance");
    
    let queries = vec![
        ("simple", "SELECT ?s WHERE { ?s ?p ?o } LIMIT 10"),
        ("complex", "SELECT ?s ?name WHERE { ?s foaf:name ?name . ?s foaf:age ?age . FILTER(?age > 18) } LIMIT 50"),
        ("federated", "SELECT ?s WHERE { SERVICE <http://test.org/sparql> { ?s ?p ?o } } LIMIT 25"),
    ];
    
    for (query_name, query) in queries {
        group.bench_function(query_name, |b| {
            b.to_async(&rt).iter(|| async {
                let federation_engine = FederationEngine::new();
                // Would register test services here
                federation_engine.execute_sparql(query).await.unwrap();
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_streaming_performance, benchmark_federation_performance);
criterion_main!(benches);
```

---

## ❓ FAQ & Troubleshooting

### Common Issues

**Q: My stream throughput is lower than expected. How can I optimize it?**

A: Check these optimization areas:
1. **Batch Size**: Increase `batch_size` in StreamConfig
2. **Compression**: Enable compression with `CompressionType::Zstd`
3. **Backend Configuration**: Tune Kafka/NATS producer settings
4. **Memory Pooling**: Enable `enable_memory_pooling`
5. **Parallel Processing**: Use multiple producers with connection pooling

**Q: Federation queries are timing out. What should I check?**

A: Common causes and solutions:
1. **Network Issues**: Check connectivity to federated services
2. **Service Overload**: Monitor service response times and load
3. **Query Complexity**: Simplify queries or add more specific filters
4. **Cache Configuration**: Enable and tune query result caching
5. **Timeout Settings**: Increase `query_timeout` in FederationConfig

**Q: How do I handle service failures in federation?**

A: OxiRS provides several fault tolerance mechanisms:
1. **Circuit Breakers**: Automatically stop querying failed services
2. **Graceful Degradation**: Return partial results when some services fail
3. **Retry Logic**: Configurable retry attempts with exponential backoff
4. **Health Monitoring**: Continuous health checks with automatic recovery
5. **Fallback Services**: Configure backup services for critical queries

**Q: Can I use OxiRS with existing RDF stores?**

A: Yes, OxiRS integrates with:
- **Apache Jena/Fuseki**: Full SPARQL 1.1 compatibility
- **Blazegraph**: Native triple store integration
- **GraphDB**: Enterprise RDF database support
- **Virtuoso**: Commercial RDF platform integration
- **Custom Stores**: Implement the `RdfStore` trait

### Performance Tuning

**Stream Performance:**
- Use appropriate backends for your use case (Kafka for durability, NATS for speed)
- Enable compression for large payloads
- Tune batch sizes based on your latency/throughput requirements
- Use connection pooling for high-concurrency scenarios

**Federation Performance:**
- Enable query result caching with appropriate TTLs
- Use materialized views for frequently accessed data
- Implement service-specific optimizations (e.g., filter pushdown)
- Monitor and tune connection pool sizes

**System Performance:**
- Allocate sufficient memory for caching and buffering
- Use SSD storage for persistent components
- Configure appropriate JVM settings if using Java-based components
- Monitor system resources and scale horizontally when needed

### Monitoring & Debugging

Use the built-in monitoring dashboard to track:
- Stream throughput and latency metrics
- Federation query performance and cache hit rates
- System resource utilization
- Error rates and failure patterns

Enable debug logging for detailed troubleshooting:
```rust
export RUST_LOG=oxirs_stream=debug,oxirs_federate=debug
```

---

## 🤝 Contributing

We welcome contributions to the OxiRS ecosystem! Here's how you can help:

### Development Setup

```bash
# Clone the repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the project
cargo build --workspace

# Run tests
cargo nextest run --workspace --no-fail-fast

# Run specific component tests
cargo nextest run -p oxirs-stream --no-fail-fast
cargo nextest run -p oxirs-federate --no-fail-fast
```

### Contribution Guidelines

1. **Code Quality**: Follow Rust best practices and ensure all tests pass
2. **Documentation**: Add comprehensive documentation for new features
3. **Performance**: Benchmark new features and ensure they meet performance targets
4. **Security**: Follow security best practices and add security tests
5. **Compatibility**: Maintain backward compatibility when possible

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the full test suite (`cargo nextest run --workspace --no-fail-fast`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Areas Where We Need Help

- **Additional Backend Support**: Redis Clusters, Apache Pulsar, AWS Kinesis
- **Federation Protocols**: Additional GraphQL federation features
- **Performance Optimization**: Query optimization algorithms
- **Security Features**: Advanced authentication and authorization
- **Documentation**: Tutorials, examples, and use case guides
- **Testing**: More comprehensive integration and performance tests

---

## 📚 Additional Resources

### Documentation
- [Architecture Guide](./docs/ARCHITECTURE.md)
- [API Reference](./docs/API.md)
- [Performance Benchmarks](./docs/BENCHMARKS.md)
- [Security Guide](./docs/SECURITY.md)

### Examples
- [Basic Streaming](./examples/basic_streaming.rs)
- [Federation Queries](./examples/federation_queries.rs)
- [Integration Patterns](./examples/integration_patterns.rs)
- [Production Deployment](./examples/production_deployment/)

### Community
- [GitHub Discussions](https://github.com/cool-japan/oxirs/discussions)
- [Discord Server](https://discord.gg/oxirs)
- [Stack Overflow Tag: oxirs](https://stackoverflow.com/questions/tagged/oxirs)

---

**🎉 Congratulations! You now have a comprehensive understanding of the OxiRS ecosystem and are ready to build high-performance, enterprise-grade RDF streaming and federation applications.**

The OxiRS platform provides everything you need for modern semantic web applications, from real-time data streaming to complex federated queries across distributed services. Start with the quick start examples and gradually explore the advanced features as your requirements grow.

For questions or support, please reach out through our community channels or create an issue on GitHub. Happy coding with OxiRS! 🚀