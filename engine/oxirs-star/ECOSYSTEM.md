# OxiRS-Star Ecosystem Integration Guide

This document provides comprehensive guidance for integrating OxiRS-Star with the broader OxiRS ecosystem and external systems.

## Integration Overview

OxiRS-Star is designed to work seamlessly with:

- **oxirs-core**: Foundation RDF/SPARQL support
- **oxirs-arq**: Query optimization and execution
- **oxirs-vec**: Vector search and semantic similarity
- **oxirs-shacl**: Schema validation with RDF-star support
- **oxirs-fuseki**: HTTP server with RDF-star endpoints
- **oxirs-gql**: GraphQL to SPARQL-star translation

## Core Integration Examples

### Integration with oxirs-arq for Query Optimization

```rust
use oxirs_star::{StarStore, StarTriple, StarTerm};
use oxirs_arq::{QueryExecutor, SparqlStarTranslator};

// Create a store with RDF-star data
let mut store = StarStore::new();
let quoted_triple = StarTriple::new(
    StarTerm::iri("http://example.org/alice")?,
    StarTerm::iri("http://foaf.org/knows")?,
    StarTerm::iri("http://example.org/bob")?,
);

let confidence_triple = StarTriple::new(
    StarTerm::quoted_triple(quoted_triple),
    StarTerm::iri("http://example.org/confidence")?,
    StarTerm::literal("0.95")?,
);

store.insert(&confidence_triple)?;

// Execute SPARQL-star queries with ARQ optimization
let query = "SELECT ?stmt ?conf WHERE { ?stmt <http://example.org/confidence> ?conf }";
let mut executor = QueryExecutor::new();
let translator = SparqlStarTranslator::new();

let results = executor.execute_star_query(&store, query, &translator)?;
for result in results {
    println!("Statement: {}, Confidence: {}", result.get("stmt")?, result.get("conf")?);
}
```

### Integration with oxirs-vec for Semantic Search

```rust
use oxirs_star::{StarStore, StarTriple, StarTerm};
use oxirs_vec::{VectorStore, EmbeddingManager, EmbeddingStrategy};

// Combine RDF-star metadata with vector search
let mut star_store = StarStore::new();
let mut vector_store = VectorStore::new();
let embedding_manager = EmbeddingManager::new(EmbeddingStrategy::SentenceTransformers);

// Create RDF-star data with provenance
let base_triple = StarTriple::new(
    StarTerm::iri("http://example.org/paper1")?,
    StarTerm::iri("http://purl.org/dc/terms/title")?,
    StarTerm::literal("Machine Learning in RDF Processing")?,
);

let provenance_triple = StarTriple::new(
    StarTerm::quoted_triple(base_triple.clone()),
    StarTerm::iri("http://example.org/extractedFrom")?,
    StarTerm::iri("http://arxiv.org/abs/2023.12345")?,
);

// Store metadata
star_store.insert(&provenance_triple)?;

// Generate embeddings for content
let title_embedding = embedding_manager.embed_text("Machine Learning in RDF Processing")?;
vector_store.insert("paper1_title", title_embedding)?;

// Perform hybrid search: vector similarity + metadata filtering
let query_embedding = embedding_manager.embed_text("AI RDF graph processing")?;
let similar_items = vector_store.search(&query_embedding, 10, 0.7)?;

// Filter by provenance metadata
for item in similar_items {
    if let Some(provenance) = star_store.get_metadata(&item.id)? {
        if provenance.contains("arxiv.org") {
            println!("High-quality match: {} (similarity: {})", item.id, item.score);
        }
    }
}
```

### Integration with oxirs-shacl for Validation

```rust
use oxirs_star::{StarStore, StarTriple, StarTerm};
use oxirs_shacl::{ShaclValidator, ValidationReport};

// Define SHACL shapes for RDF-star validation
let shapes_graph = r#"
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .

ex:ConfidenceShape a sh:NodeShape ;
    sh:targetNode [ sh:path [ sh:inversePath ex:confidence ] ] ;
    sh:property [
        sh:path ex:confidence ;
        sh:datatype xsd:decimal ;
        sh:minInclusive 0.0 ;
        sh:maxInclusive 1.0 ;
    ] .
"#;

let mut star_store = StarStore::new();
let validator = ShaclValidator::new();

// Add RDF-star data
let confidence_triple = StarTriple::new(
    StarTerm::quoted_triple(StarTriple::new(
        StarTerm::iri("http://example.org/alice")?,
        StarTerm::iri("http://example.org/knows")?,
        StarTerm::iri("http://example.org/bob")?,
    )),
    StarTerm::iri("http://example.org/confidence")?,
    StarTerm::typed_literal("0.85", "http://www.w3.org/2001/XMLSchema#decimal")?,
);

star_store.insert(&confidence_triple)?;

// Validate RDF-star data against SHACL shapes
let report = validator.validate_star_data(&star_store, shapes_graph)?;
if report.conforms() {
    println!("RDF-star data is valid!");
} else {
    for violation in report.violations() {
        println!("Validation error: {}", violation.message());
    }
}
```

## Advanced Integration Patterns

### Federation with Multiple RDF-star Endpoints

```rust
use oxirs_star::integration::{FederationManager, EndpointConfig};
use std::collections::HashMap;

// Configure multiple RDF-star endpoints
let endpoints = vec![
    EndpointConfig {
        name: "academic-papers".to_string(),
        url: "https://papers.example.org/sparql-star".to_string(),
        supported_formats: vec![StarFormat::TurtleStar, StarFormat::JsonLdStar],
        timeout: 30,
        ..Default::default()
    },
    EndpointConfig {
        name: "knowledge-base".to_string(),
        url: "https://kb.example.org/sparql-star".to_string(),
        supported_formats: vec![StarFormat::TrigStar, StarFormat::NQuadsStar],
        timeout: 15,
        ..Default::default()
    },
];

let federation_manager = FederationManager::new(endpoints);

// Execute federated SPARQL-star query
let federated_query = r#"
    PREFIX ex: <http://example.org/>
    
    SELECT ?paper ?confidence ?source WHERE {
        SERVICE <academic-papers> {
            << ?paper ex:hasAuthor ?author >> ex:confidence ?confidence .
        }
        SERVICE <knowledge-base> {
            << ?paper ex:hasAuthor ?author >> ex:source ?source .
        }
        FILTER(?confidence > 0.8)
    }
"#;

let results = federation_manager.execute_federated_query(federated_query).await?;
for result in results {
    println!("Paper: {}, Confidence: {}, Source: {}", 
             result.get("paper")?, result.get("confidence")?, result.get("source")?);
}
```

### Real-time Streaming Integration

```rust
use oxirs_star::{StarStore, StarTriple};
use oxirs_stream::{StreamProcessor, StarEventProcessor};
use tokio_stream::StreamExt;

// Create streaming processor for RDF-star events
let mut processor = StarEventProcessor::new();
let mut store = StarStore::new();

// Configure real-time processing of RDF-star streams
processor.on_triple_insert(|triple: StarTriple| async move {
    // Process new RDF-star triples in real-time
    if triple.has_quoted_subject() || triple.has_quoted_object() {
        // Handle metadata updates
        update_metadata_index(&triple).await?;
        
        // Trigger validation if needed
        if requires_validation(&triple) {
            validate_triple_async(&triple).await?;
        }
        
        // Update search indexes
        update_search_indexes(&triple).await?;
    }
    Ok(())
});

// Process streaming RDF-star data
let stream = create_rdf_star_stream().await?;
tokio::pin!(stream);

while let Some(event) = stream.next().await {
    processor.process(event).await?;
}
```

## Monitoring and Observability

### Performance Monitoring Integration

```rust
use oxirs_star::profiling::{StarProfiler, PerformanceCollector};
use oxirs_monitoring::{MetricsCollector, AlertManager};

// Set up comprehensive monitoring
let profiler = StarProfiler::new();
let metrics = MetricsCollector::new();
let alerts = AlertManager::new();

// Monitor RDF-star operations
profiler.start_monitoring();

// Set up alerts for performance issues
alerts.add_threshold("parsing_latency_ms", 1000.0);
alerts.add_threshold("query_execution_time_ms", 5000.0);
alerts.add_threshold("memory_usage_mb", 1024.0);

// Collect and export metrics
let report = profiler.generate_detailed_report();
metrics.export_prometheus_metrics(&report)?;

// Real-time dashboard integration
let dashboard_data = serde_json::to_string(&report)?;
send_to_dashboard("rdf-star-metrics", &dashboard_data).await?;
```

## Production Deployment Patterns

### Docker Integration

```dockerfile
# Dockerfile for RDF-star service
FROM rust:1.70 as builder

WORKDIR /app
COPY . .
RUN cargo build --release --package oxirs-star

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/oxirs-star /usr/local/bin/
COPY config/production.toml /etc/oxirs/config.toml

EXPOSE 8080
CMD ["oxirs-star", "--config", "/etc/oxirs/config.toml"]
```

### Kubernetes Deployment

```yaml
# kubernetes/rdf-star-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-star
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oxirs-star
  template:
    metadata:
      labels:
        app: oxirs-star
    spec:
      containers:
      - name: oxirs-star
        image: oxirs/star:latest
        ports:
        - containerPort: 8080
        env:
        - name: RUST_LOG
          value: "info"
        - name: OXIRS_CONFIG
          value: "/etc/oxirs/config.toml"
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
---
apiVersion: v1
kind: Service
metadata:
  name: oxirs-star-service
spec:
  selector:
    app: oxirs-star
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

## CLI Integration Examples

### Batch Processing

```bash
# Convert between RDF-star formats
oxirs-star convert \
    --input data.ttls \
    --output data.nts \
    --from turtle-star \
    --to ntriples-star \
    --validate

# Validate RDF-star data
oxirs-star validate \
    --data data.ttls \
    --shapes shapes.ttl \
    --format turtle-star \
    --report validation-report.json

# Performance profiling
oxirs-star profile \
    --operation parsing \
    --data large-dataset.ttls \
    --iterations 100 \
    --output profile-report.json
```

### Integration with CI/CD Pipelines

```yaml
# .github/workflows/rdf-star-validation.yml
name: RDF-star Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Install OxiRS-Star
      run: |
        cargo install oxirs-star
        
    - name: Validate RDF-star Data
      run: |
        oxirs-star validate \
          --data data/**/*.ttls \
          --recursive \
          --fail-on-error \
          --report validation-report.json
          
    - name: Upload Validation Report
      uses: actions/upload-artifact@v3
      with:
        name: validation-report
        path: validation-report.json
```

## Best Practices for Production

### Configuration Management

```toml
# production.toml
[star]
# Performance optimization
enable_indexing = true
index_quoted_triples = true
cache_size = 100000
parallel_parsing = true
memory_limit_mb = 2048

# Security settings
[security]
enable_authentication = true
rate_limiting = true
max_query_complexity = 1000
sandbox_sparql = true

# Monitoring
[monitoring]
enable_metrics = true
metrics_port = 9090
health_check_port = 8081
log_level = "info"

# Integration endpoints
[[endpoints]]
name = "primary-store"
url = "https://store.example.org/sparql-star"
timeout = 30
pool_size = 10
```

### Error Handling and Recovery

```rust
use oxirs_star::{StarStore, StarError, StarErrorKind};
use anyhow::{Context, Result};

// Robust error handling for production
async fn robust_rdf_star_processing(data: &str) -> Result<()> {
    let mut store = StarStore::new();
    
    match store.parse_and_insert(data).await {
        Ok(_) => {
            info!("Successfully processed RDF-star data");
            Ok(())
        },
        Err(StarError { kind: StarErrorKind::ParseError, .. }) => {
            warn!("Parse error encountered, attempting recovery");
            // Attempt graceful recovery
            let cleaned_data = sanitize_rdf_star_input(data)?;
            store.parse_and_insert(&cleaned_data).await
                .context("Recovery attempt failed")
        },
        Err(StarError { kind: StarErrorKind::ValidationError, .. }) => {
            error!("Validation failed, data rejected");
            Err(anyhow::anyhow!("Data validation failed"))
        },
        Err(e) => {
            error!("Unexpected error: {}", e);
            Err(e.into())
        }
    }
}
```

## Migration and Upgrade Paths

### From Standard RDF to RDF-star

```rust
use oxirs_star::migration::{RdfToStarMigrator, MigrationStrategy};

// Migrate existing RDF data to RDF-star format
let migrator = RdfToStarMigrator::new();
let strategy = MigrationStrategy::ConservativeReification;

let rdf_data = load_existing_rdf_data()?;
let star_data = migrator.migrate(rdf_data, strategy)?;

// Validate migration results
let validation_report = migrator.validate_migration(&star_data)?;
if validation_report.success_rate() > 0.95 {
    commit_migration(&star_data)?;
} else {
    rollback_migration()?;
}
```

This comprehensive ecosystem integration guide provides production-ready patterns and examples for deploying OxiRS-Star in enterprise environments with full observability, monitoring, and integration capabilities.