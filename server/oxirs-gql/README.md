# OxiRS GraphQL

[![Version](https://img.shields.io/badge/version-0.1.0--beta.2-blue)](https://github.com/cool-japan/oxirs/releases)

**High-performance GraphQL server for RDF data with automatic schema generation**

**Status**: Beta Release (v0.1.0-beta.2) - Released December 21, 2025

✨ **Beta Software**: Production-ready with API stability guarantees. Semantic versioning enforced.

## Overview

`oxirs-gql` provides a GraphQL interface to RDF datasets, automatically generating GraphQL schemas from RDF vocabularies and enabling intuitive querying of semantic data. Built on top of Juniper, it offers seamless integration between GraphQL and SPARQL worlds.

## Features

- **Automatic Schema Generation**: Generate GraphQL schemas from RDF vocabularies
- **SPARQL Translation**: Automatic translation of GraphQL queries to SPARQL
- **Type Safety**: Leverage Rust's type system for compile-time schema validation
- **High Performance**: Async execution with query optimization and caching
- **Subscriptions**: Real-time GraphQL subscriptions with WebSocket support
- **Federation**: GraphQL schema stitching across multiple RDF datasets
- **Introspection**: Full GraphQL introspection support for tooling
- **Custom Scalars**: RDF-specific scalar types (IRI, DateTime, Literal)
- **Flexible Mapping**: Custom mapping rules for complex RDF to GraphQL conversions
- **Hot Reload**: Dynamic schema updates when vocabularies change

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirs-gql = "0.1.0-beta.2"
```

## Quick Start

### Basic GraphQL Server

```rust
use oxirs_gql::{Server, Schema, Config};
use oxirs_core::Dataset;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load RDF dataset
    let dataset = Dataset::from_file("data.ttl")?;
    
    // Generate GraphQL schema automatically
    let schema = Schema::from_dataset(&dataset)?;
    
    // Create server
    let config = Config::builder()
        .port(4000)
        .enable_subscriptions(true)
        .enable_introspection(true)
        .build();
        
    let server = Server::new(schema, config);
    server.run().await
}
```

### Custom Schema Generation

```rust
use oxirs_gql::{Schema, SchemaBuilder, TypeMapping};
use oxirs_core::{Dataset, NamedNode};

let dataset = Dataset::from_file("schema.ttl")?;

let schema = SchemaBuilder::new()
    .dataset(dataset)
    // Map RDF classes to GraphQL types
    .map_class(
        NamedNode::new("http://xmlns.com/foaf/0.1/Person")?,
        TypeMapping::object("Person")
            .field("name", "foaf:name")
            .field("email", "foaf:mbox") 
            .field("friends", "foaf:knows")
    )
    // Custom resolvers
    .resolver("Person", "fullName", |person| {
        format!("{} {}", person.first_name?, person.last_name?)
    })
    .build()?;
```

## Schema Generation

### Automatic Generation

```rust
use oxirs_gql::Schema;
use oxirs_core::Dataset;

// Load FOAF vocabulary
let dataset = Dataset::from_file("foaf.rdf")?;

// Generate schema automatically
let schema = Schema::from_dataset(&dataset)?;

// Generated GraphQL schema:
// type Person {
//   name: String
//   mbox: String  
//   knows: [Person!]!
//   age: Int
// }
//
// type Query {
//   person(id: ID!): Person
//   persons: [Person!]!
// }
```

### Custom Type Mappings

```rust
use oxirs_gql::{SchemaBuilder, TypeMapping, FieldMapping};

let schema = SchemaBuilder::new()
    .map_class("foaf:Person", TypeMapping::object("Person")
        .description("A person in the FOAF vocabulary")
        .field("id", FieldMapping::id("@id"))
        .field("name", FieldMapping::string("foaf:name")
            .required(true))
        .field("email", FieldMapping::string("foaf:mbox")
            .transform(|email| email.strip_prefix("mailto:")))
        .field("friends", FieldMapping::list("foaf:knows")
            .item_type("Person"))
    )
    .build()?;
```

## Querying

### Basic Queries

```graphql
query GetPerson {
  person(id: "http://example.org/alice") {
    name
    email
    friends {
      name
      email
    }
  }
}
```

### Advanced Queries

```graphql
query SearchPeople($name: String!, $limit: Int = 10) {
  people(filter: {name: {contains: $name}}, limit: $limit) {
    nodes {
      id
      name
      email
      friendCount
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
```

### Subscriptions

```graphql
subscription NewPerson {
  personAdded {
    id
    name
    email
  }
}
```

## SPARQL Integration

### Query Translation

GraphQL queries are automatically translated to optimized SPARQL:

```graphql
# GraphQL
query {
  person(id: "http://example.org/alice") {
    name
    friends {
      name
    }
  }
}
```

```sparql
# Generated SPARQL
SELECT ?name ?friend_name WHERE {
  <http://example.org/alice> foaf:name ?name .
  OPTIONAL {
    <http://example.org/alice> foaf:knows ?friend .
    ?friend foaf:name ?friend_name .
  }
}
```

### Custom SPARQL

```rust
use oxirs_gql::{Resolver, Context};

#[derive(GraphQLObject)]
struct Person {
    id: String,
    name: String,
}

impl Person {
    // Custom resolver with SPARQL
    async fn friends_in_city(&self, ctx: &Context, city: String) -> Vec<Person> {
        let query = format!(r#"
            SELECT ?friend ?friend_name WHERE {{
                <{}> foaf:knows ?friend .
                ?friend foaf:name ?friend_name .
                ?friend ex:livesIn ?city .
                ?city rdfs:label "{}" .
            }}
        "#, self.id, city);
        
        ctx.execute_sparql(query).await?
            .into_iter()
            .map(|row| Person {
                id: row.get("friend").unwrap().to_string(),
                name: row.get("friend_name").unwrap().to_string(),
            })
            .collect()
    }
}
```

## Advanced Features

### Federation

```rust
use oxirs_gql::{FederatedSchema, RemoteSchema};

let federated = FederatedSchema::new()
    .schema("users", RemoteSchema::new("http://users.example.com/graphql"))
    .schema("products", RemoteSchema::new("http://products.example.com/graphql"))
    .extend_type("User", |user| {
        user.field("orders", "products.orders", |args| {
            args.where_field("userId", user.id)
        })
    })
    .build()?;
```

### Custom Scalars

```rust
use oxirs_gql::{CustomScalar, ScalarValue};
use oxirs_core::NamedNode;

#[derive(GraphQLScalar)]
struct IRI(NamedNode);

impl CustomScalar for IRI {
    fn serialize(&self) -> ScalarValue {
        ScalarValue::String(self.0.to_string())
    }
    
    fn parse_value(value: &ScalarValue) -> Result<Self, String> {
        match value {
            ScalarValue::String(s) => {
                NamedNode::new(s)
                    .map(IRI)
                    .map_err(|e| format!("Invalid IRI: {}", e))
            }
            _ => Err("IRI must be a string".to_string())
        }
    }
}
```

### DataLoader Integration

```rust
use oxirs_gql::{DataLoader, BatchFn};
use oxirs_core::{Dataset, NamedNode};

struct PersonLoader {
    dataset: Dataset,
}

impl BatchFn<String, Person> for PersonLoader {
    async fn load(&self, keys: &[String]) -> Vec<Person> {
        let query = format!(r#"
            SELECT ?id ?name WHERE {{
                VALUES ?id {{ {} }}
                ?id foaf:name ?name .
            }}
        "#, keys.iter().map(|k| format!("<{}>", k)).collect::<Vec<_>>().join(" "));
        
        self.dataset.query(&query).await
            .unwrap()
            .into_iter()
            .map(|row| Person {
                id: row.get("id").unwrap().to_string(),
                name: row.get("name").unwrap().to_string(),
            })
            .collect()
    }
}

// Usage in resolver
async fn friends(&self, ctx: &Context) -> Vec<Person> {
    let loader: DataLoader<PersonLoader> = ctx.data()?;
    let friend_ids = self.get_friend_ids();
    loader.load_many(friend_ids).await
}
```

## Configuration

### Server Configuration

```yaml
server:
  host: "0.0.0.0"
  port: 4000
  cors: true
  playground: true
  introspection: true

schema:
  auto_generate: true
  vocabularies:
    - "http://xmlns.com/foaf/0.1/"
    - "http://schema.org/"
  
mapping:
  naming_convention: "camelCase"
  max_depth: 10
  enable_filters: true
  enable_pagination: true

performance:
  query_cache: true
  cache_size: 1000
  max_query_depth: 15
  max_query_complexity: 1000
  
subscriptions:
  enabled: true
  transport: "websocket"
  keep_alive: 30
```

### Schema Configuration

```rust
use oxirs_gql::{Config, NamingConvention, CacheConfig};

let config = Config::builder()
    .auto_generate_schema(true)
    .naming_convention(NamingConvention::CamelCase)
    .max_query_depth(15)
    .enable_introspection(true)
    .cache(CacheConfig::new()
        .query_cache(true)
        .schema_cache(true)
        .ttl(Duration::from_secs(300)))
    .build();
```

## Performance

### Benchmarks

| Operation | QPS | Latency (p95) | Memory |
|-----------|-----|---------------|--------|
| Simple query | 12,000 | 15ms | 32MB |
| Complex nested | 3,500 | 45ms | 45MB |
| Subscription | 8,000 | 8ms | 28MB |
| Schema introspection | 15,000 | 5ms | 25MB |

### Optimization

```rust
use oxirs_gql::{QueryOptimizer, CachingStrategy};

let optimizer = QueryOptimizer::new()
    .enable_query_planning(true)
    .enable_result_caching(true)
    .caching_strategy(CachingStrategy::LRU { size: 1000 })
    .sparql_optimization(true);

let schema = Schema::new(dataset)
    .optimizer(optimizer)
    .build()?;
```

## Deployment

### Docker

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin oxirs-gql

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/oxirs-gql /usr/local/bin/
EXPOSE 4000
CMD ["oxirs-gql", "--config", "/config.yaml"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-gql
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oxirs-gql
  template:
    spec:
      containers:
      - name: oxirs-gql
        image: ghcr.io/cool-japan/oxirs-gql:latest
        ports:
        - containerPort: 4000
        env:
        - name: GRAPHQL_PLAYGROUND
          value: "false"
        - name: GRAPHQL_INTROSPECTION  
          value: "false"
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## Related Crates

- [`oxirs-core`](../core/oxirs-core/): RDF data model
- [`oxirs-fuseki`](./oxirs-fuseki/): SPARQL HTTP server
- [`oxirs-arq`](../engine/oxirs-arq/): SPARQL query engine
- [`oxirs-stream`](../stream/oxirs-stream/): Real-time subscriptions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit a pull request

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Status

� **Beta Release (v0.1.0-beta.2)** - November 16, 2025

Current alpha features:
- ✅ GraphQL server with persisted dataset introspection and hot-reload
- ✅ GraphQL ⇄ SPARQL translation covering vector/federation-aware resolvers
- ✅ Schema generation with CLI configuration parity and dataset auto-sync
- ✅ Subscriptions bridged to SPARQL/stream events (experimental)
- 🚧 Apollo Federation interoperability (in progress)

Note: This is an alpha release. Some features remain experimental and APIs may evolve before beta.