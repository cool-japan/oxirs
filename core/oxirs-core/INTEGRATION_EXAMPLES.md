# OxiRS Core Integration Examples

## 🔗 Advanced Integration Patterns

This document provides comprehensive examples for integrating OxiRS Core with various systems and frameworks.

## 🚀 High-Performance Web Services

### Actix Web Integration

```rust
use actix_web::{web, App, HttpServer, HttpResponse, Result};
use oxirs_core::{Graph, Dataset, Query};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    dataset: Arc<RwLock<Dataset>>,
    query_cache: Arc<RwLock<HashMap<String, String>>>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize high-performance dataset
    let dataset = Dataset::with_config(
        DatasetConfig::builder()
            .enable_concurrent_access(true)
            .max_concurrent_queries(10_000)
            .enable_query_caching(true)
            .cache_size(1_000_000)
            .build()
    );

    let app_state = AppState {
        dataset: Arc::new(RwLock::new(dataset)),
        query_cache: Arc::new(RwLock::new(HashMap::new())),
    };

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(app_state.clone()))
            .route("/sparql", web::post().to(sparql_endpoint))
            .route("/health", web::get().to(health_check))
            .route("/metrics", web::get().to(metrics_endpoint))
    })
    .workers(num_cpus::get())
    .bind("0.0.0.0:3030")?
    .run()
    .await
}

async fn sparql_endpoint(
    query: web::Bytes,
    state: web::Data<AppState>,
) -> Result<HttpResponse> {
    let query_str = String::from_utf8_lossy(&query);
    
    // Check cache first
    {
        let cache = state.query_cache.read().await;
        if let Some(cached_result) = cache.get(&query_str) {
            return Ok(HttpResponse::Ok()
                .content_type("application/sparql-results+json")
                .body(cached_result.clone()));
        }
    }

    // Execute query with high performance
    let dataset = state.dataset.read().await;
    let results = dataset.execute_query_with_config(
        &query_str,
        QueryConfig::builder()
            .enable_parallel_execution(true)
            .timeout(Duration::from_secs(30))
            .max_results(1_000_000)
            .build()
    ).await?;

    let json_result = results.to_json()?;

    // Cache successful results
    {
        let mut cache = state.query_cache.write().await;
        cache.insert(query_str.to_string(), json_result.clone());
    }

    Ok(HttpResponse::Ok()
        .content_type("application/sparql-results+json")
        .body(json_result))
}
```

### Warp Integration with Streaming

```rust
use warp::{Filter, Reply};
use oxirs_core::{Dataset, parser::AsyncStreamingParser};
use tokio_stream::StreamExt;
use futures::stream::TryStreamExt;

pub fn sparql_routes(
    dataset: Arc<RwLock<Dataset>>
) -> impl Filter<Extract = impl Reply, Error = warp::Rejection> + Clone {
    
    let query_route = warp::path("sparql")
        .and(warp::post())
        .and(warp::body::stream())
        .and(with_dataset(dataset.clone()))
        .and_then(handle_sparql_stream);

    let upload_route = warp::path("upload")
        .and(warp::post())
        .and(warp::body::stream())
        .and(with_dataset(dataset))
        .and_then(handle_upload_stream);

    query_route.or(upload_route)
}

async fn handle_sparql_stream(
    body: impl futures::Stream<Item = Result<impl warp::Buf, warp::Error>> + Send,
    dataset: Arc<RwLock<Dataset>>,
) -> Result<impl Reply, warp::Rejection> {
    
    // Convert body stream to bytes
    let body_bytes: Vec<u8> = body
        .try_fold(Vec::new(), |mut acc, chunk| async move {
            acc.extend_from_slice(chunk.chunk());
            Ok(acc)
        })
        .await
        .map_err(|_| warp::reject::custom(QueryError::InvalidBody))?;

    let query = String::from_utf8_lossy(&body_bytes);
    
    // Execute streaming query
    let dataset = dataset.read().await;
    let result_stream = dataset.execute_query_stream(&query).await
        .map_err(|_| warp::reject::custom(QueryError::ExecutionFailed))?;

    // Stream results back as JSON lines
    let json_stream = result_stream.map(|result| {
        Ok::<_, warp::Error>(
            warp::hyper::Body::from(format!("{}\n", result.to_json_line()))
        )
    });

    Ok(warp::reply::Response::new(
        warp::hyper::Body::wrap_stream(json_stream)
    ))
}
```

## 🗄️ Database Integration

### PostgreSQL Integration

```rust
use sqlx::{PgPool, Row};
use oxirs_core::{Graph, Triple, NamedNode, Literal};
use anyhow::Result;

pub struct PostgresRdfBridge {
    pool: PgPool,
    graph: Graph,
}

impl PostgresRdfBridge {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPool::connect(database_url).await?;
        
        // Create optimized RDF storage tables
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS rdf_triples (
                id BIGSERIAL PRIMARY KEY,
                subject_type VARCHAR(10) NOT NULL,
                subject_value TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_type VARCHAR(10) NOT NULL,
                object_value TEXT NOT NULL,
                object_datatype TEXT,
                object_language VARCHAR(10),
                graph_name TEXT DEFAULT 'default',
                created_at TIMESTAMP DEFAULT NOW()
            );
            
            -- Optimized indexes for RDF access patterns
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rdf_spo 
                ON rdf_triples(subject_value, predicate, object_value);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rdf_pos 
                ON rdf_triples(predicate, object_value, subject_value);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rdf_osp 
                ON rdf_triples(object_value, subject_value, predicate);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rdf_graph 
                ON rdf_triples(graph_name);
        "#).execute(&pool).await?;

        let graph = Graph::with_config(
            GraphConfig::builder()
                .enable_external_persistence(true)
                .persistence_backend(PersistenceBackend::Custom)
                .build()
        );

        Ok(Self { pool, graph })
    }

    pub async fn sync_from_postgres(&mut self) -> Result<usize> {
        let mut count = 0;
        let mut rows = sqlx::query(
            "SELECT subject_type, subject_value, predicate, 
                    object_type, object_value, object_datatype, object_language 
             FROM rdf_triples ORDER BY id"
        ).fetch(&self.pool);

        while let Some(row) = rows.try_next().await? {
            let subject = match row.get::<&str, _>("subject_type") {
                "iri" => Term::NamedNode(NamedNode::new_unchecked(
                    row.get::<&str, _>("subject_value")
                )),
                "blank" => Term::BlankNode(BlankNode::new_unchecked(
                    row.get::<&str, _>("subject_value")
                )),
                _ => continue,
            };

            let predicate = NamedNode::new_unchecked(
                row.get::<&str, _>("predicate")
            );

            let object = match row.get::<&str, _>("object_type") {
                "iri" => Term::NamedNode(NamedNode::new_unchecked(
                    row.get::<&str, _>("object_value")
                )),
                "literal" => {
                    let value = row.get::<&str, _>("object_value");
                    let datatype = row.get::<Option<&str>, _>("object_datatype");
                    let language = row.get::<Option<&str>, _>("object_language");
                    
                    Term::Literal(match (datatype, language) {
                        (Some(dt), None) => Literal::new_typed_literal(value, dt)?,
                        (None, Some(lang)) => Literal::new_language_tagged_literal(value, lang)?,
                        _ => Literal::new_simple_literal(value),
                    })
                },
                _ => continue,
            };

            let triple = Triple::new(subject, predicate, object);
            self.graph.insert(triple);
            count += 1;
        }

        Ok(count)
    }

    pub async fn sync_to_postgres(&self) -> Result<usize> {
        let mut tx = self.pool.begin().await?;
        let mut count = 0;

        // Clear existing data
        sqlx::query("TRUNCATE rdf_triples").execute(&mut *tx).await?;

        // Batch insert triples
        for triple in self.graph.iter() {
            let (subject_type, subject_value) = match triple.subject() {
                Term::NamedNode(node) => ("iri", node.as_str()),
                Term::BlankNode(node) => ("blank", node.as_str()),
                _ => continue,
            };

            let predicate = triple.predicate().as_str();

            let (object_type, object_value, object_datatype, object_language) = 
                match triple.object() {
                    Term::NamedNode(node) => ("iri", node.as_str(), None, None),
                    Term::Literal(literal) => {
                        let value = literal.value();
                        let datatype = literal.datatype().map(|dt| dt.as_str());
                        let language = literal.language();
                        ("literal", value, datatype, language)
                    },
                    _ => continue,
                };

            sqlx::query(r#"
                INSERT INTO rdf_triples 
                (subject_type, subject_value, predicate, 
                 object_type, object_value, object_datatype, object_language)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#)
            .bind(subject_type)
            .bind(subject_value)
            .bind(predicate)
            .bind(object_type)
            .bind(object_value)
            .bind(object_datatype)
            .bind(object_language)
            .execute(&mut *tx).await?;

            count += 1;
        }

        tx.commit().await?;
        Ok(count)
    }
}
```

### Redis Integration for Caching

```rust
use redis::{Client, Commands, AsyncCommands, Connection};
use oxirs_core::{Graph, Query, QueryResult};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct CachedQueryResult {
    query_hash: String,
    result: String,
    timestamp: u64,
    ttl: u64,
}

pub struct RedisQueryCache {
    client: Client,
    default_ttl: u64,
}

impl RedisQueryCache {
    pub fn new(redis_url: &str, default_ttl: u64) -> Result<Self> {
        let client = Client::open(redis_url)?;
        Ok(Self { client, default_ttl })
    }

    pub async fn get_cached_result(&self, query: &str) -> Result<Option<QueryResult>> {
        let mut conn = self.client.get_async_connection().await?;
        let query_hash = self.hash_query(query);
        
        let cached: Option<String> = conn.get(&query_hash).await?;
        
        if let Some(cached_json) = cached {
            let cached_result: CachedQueryResult = serde_json::from_str(&cached_json)?;
            
            // Check if cache is still valid
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
                
            if now < cached_result.timestamp + cached_result.ttl {
                return Ok(Some(QueryResult::from_json(&cached_result.result)?));
            } else {
                // Remove expired cache entry
                let _: () = conn.del(&query_hash).await?;
            }
        }
        
        Ok(None)
    }

    pub async fn cache_result(
        &self, 
        query: &str, 
        result: &QueryResult, 
        ttl: Option<u64>
    ) -> Result<()> {
        let mut conn = self.client.get_async_connection().await?;
        let query_hash = self.hash_query(query);
        let ttl = ttl.unwrap_or(self.default_ttl);
        
        let cached_result = CachedQueryResult {
            query_hash: query_hash.clone(),
            result: result.to_json()?,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            ttl,
        };
        
        let cached_json = serde_json::to_string(&cached_result)?;
        let _: () = conn.set_ex(&query_hash, cached_json, ttl).await?;
        
        Ok(())
    }

    fn hash_query(&self, query: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        format!("oxirs:query:{:x}", hasher.finish())
    }
}

// Usage in query service
pub async fn execute_cached_query(
    query: &str,
    graph: &Graph,
    cache: &RedisQueryCache,
) -> Result<QueryResult> {
    // Try cache first
    if let Some(cached_result) = cache.get_cached_result(query).await? {
        return Ok(cached_result);
    }
    
    // Execute query
    let result = graph.execute_query(query).await?;
    
    // Cache result asynchronously
    let cache_clone = cache.clone();
    let query_clone = query.to_string();
    let result_clone = result.clone();
    
    tokio::spawn(async move {
        if let Err(e) = cache_clone.cache_result(&query_clone, &result_clone, None).await {
            eprintln!("Failed to cache query result: {}", e);
        }
    });
    
    Ok(result)
}
```

## ☁️ Cloud Integration

### AWS S3 Integration

```rust
use aws_sdk_s3::{Client, Config, Region};
use oxirs_core::{Graph, parser::AsyncStreamingParser};
use tokio_util::io::StreamReader;
use tokio::io::AsyncRead;

pub struct S3RdfLoader {
    s3_client: Client,
    bucket: String,
}

impl S3RdfLoader {
    pub async fn new(region: &str, bucket: &str) -> Self {
        let config = Config::builder()
            .region(Region::new(region.to_string()))
            .build();
        let s3_client = Client::from_conf(config);

        Self {
            s3_client,
            bucket: bucket.to_string(),
        }
    }

    pub async fn load_rdf_from_s3(
        &self,
        key: &str,
        graph: &mut Graph,
    ) -> Result<usize> {
        // Stream download from S3
        let response = self.s3_client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await?;

        let stream = response.body;
        let reader = StreamReader::new(stream.map(|result| {
            result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        }));

        // Parse streaming RDF data
        let mut parser = AsyncStreamingParser::new();
        let mut count = 0;

        parser.parse_async(reader, |triple| {
            graph.insert(triple);
            count += 1;
            Ok(())
        }).await?;

        Ok(count)
    }

    pub async fn save_rdf_to_s3(
        &self,
        key: &str,
        graph: &Graph,
        format: RdfFormat,
    ) -> Result<()> {
        // Serialize graph to bytes
        let serialized = graph.serialize_to_bytes(format)?;
        
        // Upload to S3
        self.s3_client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(serialized.into())
            .content_type(format.mime_type())
            .send()
            .await?;

        Ok(())
    }

    pub async fn sync_directory(
        &self,
        prefix: &str,
        graph: &mut Graph,
    ) -> Result<usize> {
        let mut total_triples = 0;
        let mut continuation_token = None;

        loop {
            let mut request = self.s3_client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(prefix);

            if let Some(token) = continuation_token {
                request = request.continuation_token(token);
            }

            let response = request.send().await?;

            if let Some(objects) = response.contents {
                for object in objects {
                    if let Some(key) = object.key {
                        if key.ends_with(".nt") || key.ends_with(".ttl") || key.ends_with(".rdf") {
                            let triples = self.load_rdf_from_s3(&key, graph).await?;
                            total_triples += triples;
                        }
                    }
                }
            }

            continuation_token = response.next_continuation_token;
            if continuation_token.is_none() {
                break;
            }
        }

        Ok(total_triples)
    }
}
```

### Apache Kafka Integration

```rust
use rdkafka::{
    consumer::{StreamConsumer, Consumer},
    producer::{FutureProducer, FutureRecord},
    config::ClientConfig,
    Message, TopicPartitionList,
};
use oxirs_core::{Graph, Triple, parser::Parser};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct RdfMessage {
    pub operation: String, // "insert", "delete", "replace"
    pub graph_name: Option<String>,
    pub triples: Vec<String>, // Serialized triples
    pub timestamp: u64,
    pub source: String,
}

pub struct KafkaRdfStreamer {
    consumer: StreamConsumer,
    producer: FutureProducer,
    graph: Graph,
}

impl KafkaRdfStreamer {
    pub async fn new(brokers: &str, group_id: &str) -> Result<Self> {
        let consumer: StreamConsumer = ClientConfig::new()
            .set("group.id", group_id)
            .set("bootstrap.servers", brokers)
            .set("enable.partition.eof", "false")
            .set("session.timeout.ms", "6000")
            .set("enable.auto.commit", "true")
            .set("auto.offset.reset", "earliest")
            .create()?;

        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", brokers)
            .set("message.timeout.ms", "5000")
            .create()?;

        let graph = Graph::with_config(
            GraphConfig::builder()
                .enable_streaming_updates(true)
                .enable_change_tracking(true)
                .build()
        );

        Ok(Self { consumer, producer, graph })
    }

    pub async fn subscribe_to_rdf_updates(&mut self, topics: &[&str]) -> Result<()> {
        let mut topic_partition_list = TopicPartitionList::new();
        for topic in topics {
            topic_partition_list.add_partition(topic, 0);
        }
        
        self.consumer.subscribe(topics)?;

        // Start consuming messages
        loop {
            match self.consumer.recv().await {
                Ok(message) => {
                    if let Some(payload) = message.payload() {
                        let rdf_message: RdfMessage = serde_json::from_slice(payload)?;
                        self.process_rdf_message(rdf_message).await?;
                    }
                }
                Err(e) => {
                    eprintln!("Kafka error: {}", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    async fn process_rdf_message(&mut self, message: RdfMessage) -> Result<()> {
        match message.operation.as_str() {
            "insert" => {
                for triple_str in message.triples {
                    let triple = Parser::parse_triple(&triple_str)?;
                    self.graph.insert(triple);
                }
            }
            "delete" => {
                for triple_str in message.triples {
                    let triple = Parser::parse_triple(&triple_str)?;
                    self.graph.remove(&triple);
                }
            }
            "replace" => {
                // Clear graph and insert new triples
                self.graph.clear();
                for triple_str in message.triples {
                    let triple = Parser::parse_triple(&triple_str)?;
                    self.graph.insert(triple);
                }
            }
            _ => {
                eprintln!("Unknown operation: {}", message.operation);
            }
        }

        Ok(())
    }

    pub async fn publish_graph_changes(&self, topic: &str) -> Result<()> {
        // Get recent changes from graph
        let changes = self.graph.get_recent_changes()?;
        
        for change in changes {
            let rdf_message = RdfMessage {
                operation: change.operation,
                graph_name: change.graph_name,
                triples: change.triples.iter().map(|t| t.to_string()).collect(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs(),
                source: "oxirs-core".to_string(),
            };

            let payload = serde_json::to_string(&rdf_message)?;
            
            self.producer
                .send(
                    FutureRecord::to(topic)
                        .key(&change.id)
                        .payload(&payload),
                    Duration::from_secs(0),
                )
                .await
                .map_err(|(e, _)| e)?;
        }

        Ok(())
    }
}
```

## 🤖 AI/ML Integration

### TensorFlow Integration

```rust
use tensorflow::{Graph as TfGraph, Session, SessionOptions, Tensor};
use oxirs_core::{Graph, Triple, NamedNode};
use ndarray::{Array2, Array1};
use std::collections::HashMap;

pub struct GraphEmbeddings {
    tf_session: Session,
    node_to_id: HashMap<String, usize>,
    id_to_node: HashMap<usize, String>,
    embedding_size: usize,
}

impl GraphEmbeddings {
    pub fn new(model_path: &str, embedding_size: usize) -> Result<Self> {
        // Load pre-trained graph embedding model
        let mut graph_def = std::fs::read(format!("{}/model.pb", model_path))?;
        let graph = TfGraph::new();
        graph.import_graph_def(&graph_def, &tensorflow::ImportGraphDefOptions::new())?;
        
        let session = Session::new(&SessionOptions::new(), &graph)?;

        Ok(Self {
            tf_session: session,
            node_to_id: HashMap::new(),
            id_to_node: HashMap::new(),
            embedding_size,
        })
    }

    pub fn extract_features_from_graph(&mut self, graph: &Graph) -> Result<Array2<f32>> {
        // Build node vocabulary
        let mut node_id = 0;
        for triple in graph.iter() {
            for term in [triple.subject(), triple.object()] {
                if let Term::NamedNode(node) = term {
                    let node_str = node.as_str();
                    if !self.node_to_id.contains_key(node_str) {
                        self.node_to_id.insert(node_str.to_string(), node_id);
                        self.id_to_node.insert(node_id, node_str.to_string());
                        node_id += 1;
                    }
                }
            }
        }

        let num_nodes = self.node_to_id.len();
        let mut adjacency_matrix = Array2::<f32>::zeros((num_nodes, num_nodes));

        // Build adjacency matrix
        for triple in graph.iter() {
            if let (Term::NamedNode(subj), Term::NamedNode(obj)) = 
                (triple.subject(), triple.object()) {
                
                let subj_id = self.node_to_id[subj.as_str()];
                let obj_id = self.node_to_id[obj.as_str()];
                adjacency_matrix[[subj_id, obj_id]] = 1.0;
            }
        }

        // Run through TensorFlow model to get embeddings
        let input_tensor = Tensor::new(&[num_nodes as u64, num_nodes as u64])
            .with_values(&adjacency_matrix.as_slice().unwrap())?;

        let mut output_tensor = Tensor::<f32>::new(&[num_nodes as u64, self.embedding_size as u64]);
        
        self.tf_session.run(&mut [
            ("input:0", &input_tensor),
        ], &mut [
            ("embeddings:0", &mut output_tensor),
        ], &mut [])?;

        // Convert to ndarray
        let output_shape = output_tensor.shape();
        let output_data = output_tensor.to_vec();
        let embeddings = Array2::from_shape_vec(
            (output_shape[0] as usize, output_shape[1] as usize),
            output_data
        )?;

        Ok(embeddings)
    }

    pub fn find_similar_nodes(&self, node: &str, embeddings: &Array2<f32>, k: usize) -> Vec<(String, f32)> {
        if let Some(&target_id) = self.node_to_id.get(node) {
            let target_embedding = embeddings.row(target_id);
            
            let mut similarities: Vec<(usize, f32)> = (0..embeddings.nrows())
                .map(|i| {
                    let embedding = embeddings.row(i);
                    let similarity = cosine_similarity(&target_embedding, &embedding);
                    (i, similarity)
                })
                .collect();

            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            similarities.into_iter()
                .take(k)
                .filter_map(|(id, sim)| {
                    self.id_to_node.get(&id).map(|node| (node.clone(), sim))
                })
                .collect()
        } else {
            Vec::new()
        }
    }
}

fn cosine_similarity(a: &ndarray::ArrayView1<f32>, b: &ndarray::ArrayView1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    dot_product / (norm_a * norm_b)
}
```

### OpenAI Integration for Natural Language Queries

```rust
use openai_api_rs::v1::api::OpenAIClient;
use openai_api_rs::v1::chat_completion::{ChatCompletionRequest, ChatCompletionMessage};
use oxirs_core::{Graph, Query};
use serde_json::Value;

pub struct NaturalLanguageQueryProcessor {
    openai_client: OpenAIClient,
    graph_schema: String,
    examples: Vec<(String, String)>, // (natural language, SPARQL)
}

impl NaturalLanguageQueryProcessor {
    pub fn new(api_key: &str, graph: &Graph) -> Result<Self> {
        let client = OpenAIClient::new(api_key.to_string());
        let schema = Self::extract_graph_schema(graph)?;
        
        let examples = vec![
            ("Find all people".to_string(), 
             "SELECT ?person WHERE { ?person a foaf:Person }".to_string()),
            ("What are the names of all organizations?".to_string(),
             "SELECT ?name WHERE { ?org a org:Organization ; rdfs:label ?name }".to_string()),
        ];

        Ok(Self {
            openai_client: client,
            graph_schema: schema,
            examples,
        })
    }

    pub async fn natural_language_to_sparql(&self, question: &str) -> Result<String> {
        let prompt = self.build_prompt(question);
        
        let request = ChatCompletionRequest::new(
            "gpt-4".to_string(),
            vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: prompt,
                name: None,
            }],
        );

        let response = self.openai_client.chat_completion(request).await?;
        
        if let Some(choice) = response.choices.first() {
            let sparql_query = choice.message.content.trim();
            
            // Basic validation
            if sparql_query.to_uppercase().contains("SELECT") ||
               sparql_query.to_uppercase().contains("CONSTRUCT") ||
               sparql_query.to_uppercase().contains("ASK") {
                Ok(sparql_query.to_string())
            } else {
                Err(anyhow::anyhow!("Generated response is not a valid SPARQL query"))
            }
        } else {
            Err(anyhow::anyhow!("No response from OpenAI"))
        }
    }

    pub async fn query_with_natural_language(
        &self,
        question: &str,
        graph: &Graph,
    ) -> Result<QueryResult> {
        let sparql_query = self.natural_language_to_sparql(question).await?;
        let result = graph.execute_query(&sparql_query).await?;
        Ok(result)
    }

    fn build_prompt(&self, question: &str) -> String {
        let mut prompt = format!(r#"
You are an expert at converting natural language questions into SPARQL queries.

Graph Schema:
{}

Example conversions:
"#, self.graph_schema);

        for (nl, sparql) in &self.examples {
            prompt.push_str(&format!("Q: {}\nA: {}\n\n", nl, sparql));
        }

        prompt.push_str(&format!(r#"
Now convert this question to SPARQL:
Q: {}
A: "#, question));

        prompt
    }

    fn extract_graph_schema(graph: &Graph) -> Result<String> {
        let mut classes = std::collections::HashSet::new();
        let mut properties = std::collections::HashSet::new();

        for triple in graph.iter() {
            if let Term::NamedNode(predicate) = triple.predicate() {
                let pred_str = predicate.as_str();
                
                if pred_str.ends_with("#type") || pred_str.ends_with("/type") {
                    if let Term::NamedNode(class) = triple.object() {
                        classes.insert(class.as_str());
                    }
                } else {
                    properties.insert(pred_str);
                }
            }
        }

        let mut schema = String::new();
        schema.push_str("Classes:\n");
        for class in &classes {
            schema.push_str(&format!("  {}\n", class));
        }
        
        schema.push_str("\nProperties:\n");
        for property in &properties {
            schema.push_str(&format!("  {}\n", property));
        }

        Ok(schema)
    }
}
```

## 🎯 Conclusion

These integration examples demonstrate how OxiRS Core can be seamlessly integrated with:

- **Web Frameworks**: High-performance HTTP services with Actix Web and Warp
- **Databases**: PostgreSQL for persistent storage, Redis for caching
- **Cloud Services**: AWS S3 for object storage, Apache Kafka for streaming
- **AI/ML Platforms**: TensorFlow for graph embeddings, OpenAI for natural language processing

Each integration pattern is optimized for production use with proper error handling, performance optimization, and scalability considerations.

For specific integration needs, these examples can be adapted and extended based on your particular requirements and constraints.