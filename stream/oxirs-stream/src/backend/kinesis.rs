//! # AWS Kinesis Backend
//!
//! AWS Kinesis Data Streams backend implementation for the streaming module.

use async_trait::async_trait;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_kinesis::{
    Client as KinesisClient,
    types::{
        DescribeStreamOutput, ListStreamsOutput, PutRecordsRequestEntry, PutRecordsResultEntry,
        Record, Shard, ShardIteratorType, StreamDescription, StreamStatus,
    },
    config::Credentials,
};
use base64::{Engine as _, engine::general_purpose};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::backend::StreamBackend;
use crate::consumer::ConsumerGroup;
use crate::error::{StreamError, StreamResult};
use crate::event::{StreamEvent, StreamEventType};
use crate::types::{EventMetadata, Offset, PartitionId, StreamPosition, TopicName};

const DEFAULT_SHARD_COUNT: u32 = 4;
const DEFAULT_RETENTION_HOURS: i32 = 24;
const MAX_BATCH_SIZE: usize = 500; // Kinesis limit
const MAX_RECORD_SIZE: usize = 1_048_576; // 1MB
const DEFAULT_ITERATOR_AGE_MS: u64 = 300_000; // 5 minutes
const DEFAULT_POLL_INTERVAL_MS: u64 = 100;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KinesisBackendConfig {
    pub region: String,
    pub stream_prefix: String,
    pub shard_count: u32,
    pub retention_hours: i32,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub session_token: Option<String>,
    pub endpoint_url: Option<String>,
    pub enhanced_fan_out: bool,
    pub encryption_type: Option<String>,
    pub kms_key_id: Option<String>,
    pub tags: HashMap<String, String>,
}

impl Default for KinesisBackendConfig {
    fn default() -> Self {
        Self {
            region: "us-east-1".to_string(),
            stream_prefix: "oxirs-".to_string(),
            shard_count: DEFAULT_SHARD_COUNT,
            retention_hours: DEFAULT_RETENTION_HOURS,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            endpoint_url: None,
            enhanced_fan_out: false,
            encryption_type: None,
            kms_key_id: None,
            tags: HashMap::from([
                ("Application".to_string(), "OxiRS".to_string()),
                ("Backend".to_string(), "Kinesis".to_string()),
            ]),
        }
    }
}

pub struct KinesisBackend {
    config: KinesisBackendConfig,
    client: Arc<KinesisClient>,
    stream_cache: Arc<RwLock<HashMap<String, StreamInfo>>>,
    consumer_state: Arc<Mutex<HashMap<String, ConsumerState>>>,
    stats: Arc<RwLock<KinesisStats>>,
}

#[derive(Clone, Debug)]
struct StreamInfo {
    stream_name: String,
    stream_arn: String,
    status: StreamStatus,
    shard_count: usize,
    creation_timestamp: SystemTime,
}

#[derive(Clone, Debug)]
struct ConsumerState {
    consumer_group: String,
    stream_name: String,
    shard_iterators: HashMap<String, String>,
    last_sequence_numbers: HashMap<String, String>,
    checkpoint_map: HashMap<String, u64>,
}

#[derive(Clone, Debug, Default)]
struct KinesisStats {
    records_sent: u64,
    records_received: u64,
    records_failed: u64,
    bytes_sent: u64,
    bytes_received: u64,
    put_records_calls: u64,
    get_records_calls: u64,
    throttling_errors: u64,
    last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KinesisEventRecord {
    event_id: String,
    event_type: String,
    timestamp: u64,
    data: serde_json::Value,
    metadata: Option<EventMetadata>,
}

impl From<StreamEvent> for KinesisEventRecord {
    fn from(event: StreamEvent) -> Self {
        let (event_type, data) = match &event.event_type {
            StreamEventType::TripleAdded { subject, predicate, object, graph } => (
                "triple_added".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
            ),
            StreamEventType::TripleRemoved { subject, predicate, object, graph } => (
                "triple_removed".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
            ),
            StreamEventType::QuadAdded { subject, predicate, object, graph } => (
                "quad_added".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
            ),
            StreamEventType::QuadRemoved { subject, predicate, object, graph } => (
                "quad_removed".to_string(),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "graph": graph
                }),
            ),
            StreamEventType::GraphCreated { graph } => (
                "graph_created".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
            ),
            StreamEventType::GraphCleared { graph } => (
                "graph_cleared".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
            ),
            StreamEventType::GraphDeleted { graph } => (
                "graph_deleted".to_string(),
                serde_json::json!({
                    "graph": graph
                }),
            ),
            StreamEventType::SparqlUpdate { query } => (
                "sparql_update".to_string(),
                serde_json::json!({
                    "query": query
                }),
            ),
            StreamEventType::TransactionBegin { transaction_id } => (
                "transaction_begin".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
            ),
            StreamEventType::TransactionCommit { transaction_id } => (
                "transaction_commit".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
            ),
            StreamEventType::TransactionAbort { transaction_id } => (
                "transaction_abort".to_string(),
                serde_json::json!({
                    "transaction_id": transaction_id
                }),
            ),
        };

        Self {
            event_id: Uuid::new_v4().to_string(),
            event_type,
            timestamp: event.timestamp,
            data,
            metadata: event.metadata,
        }
    }
}

impl KinesisEventRecord {
    fn to_stream_event(&self) -> StreamResult<StreamEvent> {
        let event_type = match self.event_type.as_str() {
            "triple_added" => {
                let subject = self.data["subject"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing subject".to_string()))?
                    .to_string();
                let predicate = self.data["predicate"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing predicate".to_string()))?
                    .to_string();
                let object = self.data["object"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing object".to_string()))?
                    .to_string();
                let graph = self.data["graph"].as_str().map(|s| s.to_string());
                
                StreamEventType::TripleAdded { subject, predicate, object, graph }
            }
            "triple_removed" => {
                let subject = self.data["subject"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing subject".to_string()))?
                    .to_string();
                let predicate = self.data["predicate"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing predicate".to_string()))?
                    .to_string();
                let object = self.data["object"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing object".to_string()))?
                    .to_string();
                let graph = self.data["graph"].as_str().map(|s| s.to_string());
                
                StreamEventType::TripleRemoved { subject, predicate, object, graph }
            }
            "graph_created" => {
                let graph = self.data["graph"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing graph".to_string()))?
                    .to_string();
                StreamEventType::GraphCreated { graph }
            }
            "graph_cleared" => {
                let graph = self.data["graph"].as_str().map(|s| s.to_string());
                StreamEventType::GraphCleared { graph }
            }
            "graph_deleted" => {
                let graph = self.data["graph"].as_str()
                    .ok_or_else(|| StreamError::Deserialization("Missing graph".to_string()))?
                    .to_string();
                StreamEventType::GraphDeleted { graph }
            }
            _ => return Err(StreamError::Deserialization(format!("Unknown event type: {}", self.event_type))),
        };

        Ok(StreamEvent {
            event_type,
            timestamp: self.timestamp,
            metadata: self.metadata.clone(),
        })
    }
}

impl KinesisBackend {
    pub async fn new(config: KinesisBackendConfig) -> StreamResult<Self> {
        let aws_config = Self::build_aws_config(&config).await?;
        let client = KinesisClient::new(&aws_config);

        Ok(Self {
            config,
            client: Arc::new(client),
            stream_cache: Arc::new(RwLock::new(HashMap::new())),
            consumer_state: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(RwLock::new(KinesisStats::default())),
        })
    }

    async fn build_aws_config(config: &KinesisBackendConfig) -> StreamResult<aws_config::SdkConfig> {
        let region = Region::new(config.region.clone());
        
        let mut config_builder = aws_config::defaults(BehaviorVersion::latest())
            .region(region);

        if let (Some(access_key), Some(secret_key)) = (&config.access_key_id, &config.secret_access_key) {
            let credentials = Credentials::new(
                access_key,
                secret_key,
                config.session_token.clone(),
                None,
                "oxirs-kinesis",
            );
            config_builder = config_builder.credentials_provider(credentials);
        }

        if let Some(endpoint) = &config.endpoint_url {
            config_builder = config_builder.endpoint_url(endpoint);
        }

        let aws_config = config_builder.load().await;
        Ok(aws_config)
    }

    fn get_stream_name(&self, topic: &TopicName) -> String {
        format!("{}{}", self.config.stream_prefix, topic.as_str())
    }

    async fn wait_for_stream_active(&self, stream_name: &str) -> StreamResult<()> {
        let mut attempts = 0;
        const MAX_ATTEMPTS: u32 = 60;
        
        loop {
            let describe_result = self.client
                .describe_stream()
                .stream_name(stream_name)
                .send()
                .await
                .map_err(|e| StreamError::Backend(format!("Failed to describe stream: {}", e)))?;

            if let Some(description) = describe_result.stream_description {
                if let Some(status) = description.stream_status {
                    match status {
                        StreamStatus::Active => return Ok(()),
                        StreamStatus::Creating | StreamStatus::Updating => {
                            attempts += 1;
                            if attempts >= MAX_ATTEMPTS {
                                return Err(StreamError::Timeout(
                                    format!("Stream {} did not become active after {} attempts", stream_name, MAX_ATTEMPTS)
                                ));
                            }
                            sleep(Duration::from_secs(1)).await;
                        }
                        _ => {
                            return Err(StreamError::Backend(
                                format!("Stream {} is in unexpected state: {:?}", stream_name, status)
                            ));
                        }
                    }
                }
            }
        }
    }

    async fn get_shard_iterator(
        &self,
        stream_name: &str,
        shard_id: &str,
        iterator_type: ShardIteratorType,
        sequence_number: Option<String>,
    ) -> StreamResult<String> {
        let mut request = self.client
            .get_shard_iterator()
            .stream_name(stream_name)
            .shard_id(shard_id)
            .shard_iterator_type(iterator_type);

        if let Some(seq) = sequence_number {
            request = request.starting_sequence_number(seq);
        }

        let response = request
            .send()
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to get shard iterator: {}", e)))?;

        response.shard_iterator
            .ok_or_else(|| StreamError::Backend("No shard iterator returned".to_string()))
    }

    async fn update_stream_cache(&self, stream_name: &str) -> StreamResult<()> {
        let describe_result = self.client
            .describe_stream()
            .stream_name(stream_name)
            .send()
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to describe stream: {}", e)))?;

        if let Some(description) = describe_result.stream_description {
            let info = StreamInfo {
                stream_name: stream_name.to_string(),
                stream_arn: description.stream_arn.unwrap_or_default(),
                status: description.stream_status.unwrap_or(StreamStatus::Creating),
                shard_count: description.shards.len(),
                creation_timestamp: SystemTime::now(),
            };

            self.stream_cache.write().await.insert(stream_name.to_string(), info);
        }

        Ok(())
    }
}

#[async_trait]
impl StreamBackend for KinesisBackend {
    fn name(&self) -> &'static str {
        "kinesis"
    }

    async fn connect(&mut self) -> StreamResult<()> {
        // Test connection by listing streams
        let _result = self.client
            .list_streams()
            .limit(1)
            .send()
            .await
            .map_err(|e| StreamError::Connection(format!("Failed to connect to Kinesis: {}", e)))?;

        info!("Successfully connected to AWS Kinesis in region {}", self.config.region);
        Ok(())
    }

    async fn disconnect(&mut self) -> StreamResult<()> {
        self.stream_cache.write().await.clear();
        self.consumer_state.lock().await.clear();
        info!("Disconnected from AWS Kinesis");
        Ok(())
    }

    async fn create_topic(&self, topic: &TopicName, partitions: u32) -> StreamResult<()> {
        let stream_name = self.get_stream_name(topic);
        
        // Check if stream already exists
        match self.client.describe_stream().stream_name(&stream_name).send().await {
            Ok(_) => {
                info!("Kinesis stream {} already exists", stream_name);
                return Ok(());
            }
            Err(e) => {
                let error_str = e.to_string();
                if !error_str.contains("ResourceNotFoundException") {
                    return Err(StreamError::Backend(format!("Failed to check stream existence: {}", e)));
                }
            }
        }

        // Create stream
        let shard_count = if partitions > 0 { partitions } else { self.config.shard_count };
        
        let mut request = self.client
            .create_stream()
            .stream_name(&stream_name)
            .shard_count(shard_count as i32);

        if let Some(enc_type) = &self.config.encryption_type {
            request = request.stream_mode_details(
                aws_sdk_kinesis::types::StreamModeDetails::builder()
                    .stream_mode(aws_sdk_kinesis::types::StreamMode::Provisioned)
                    .build()
            );
        }

        request
            .send()
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to create stream: {}", e)))?;

        // Wait for stream to become active
        self.wait_for_stream_active(&stream_name).await?;
        
        // Update cache
        self.update_stream_cache(&stream_name).await?;

        info!("Created Kinesis stream {} with {} shards", stream_name, shard_count);
        Ok(())
    }

    async fn delete_topic(&self, topic: &TopicName) -> StreamResult<()> {
        let stream_name = self.get_stream_name(topic);
        
        self.client
            .delete_stream()
            .stream_name(&stream_name)
            .send()
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to delete stream: {}", e)))?;

        self.stream_cache.write().await.remove(&stream_name);
        
        info!("Deleted Kinesis stream {}", stream_name);
        Ok(())
    }

    async fn list_topics(&self) -> StreamResult<Vec<TopicName>> {
        let mut topics = Vec::new();
        let mut next_token = None;

        loop {
            let mut request = self.client.list_streams();
            if let Some(token) = next_token {
                request = request.next_token(token);
            }

            let response = request
                .send()
                .await
                .map_err(|e| StreamError::Backend(format!("Failed to list streams: {}", e)))?;

            for stream_name in response.stream_names() {
                if let Some(topic_name) = stream_name.strip_prefix(&self.config.stream_prefix) {
                    topics.push(TopicName::new(topic_name.to_string()));
                }
            }

            if response.has_more_streams() {
                next_token = response.next_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        Ok(topics)
    }

    async fn send_event(&self, topic: &TopicName, event: StreamEvent) -> StreamResult<Offset> {
        let stream_name = self.get_stream_name(topic);
        let kinesis_record = KinesisEventRecord::from(event);
        
        let data = serde_json::to_vec(&kinesis_record)
            .map_err(|e| StreamError::Serialization(e.to_string()))?;

        if data.len() > MAX_RECORD_SIZE {
            return Err(StreamError::Backend(format!(
                "Record size {} exceeds maximum allowed size of {}",
                data.len(),
                MAX_RECORD_SIZE
            )));
        }

        let partition_key = kinesis_record.event_id.clone();
        
        let record = PutRecordsRequestEntry::builder()
            .data(aws_sdk_kinesis::primitives::Blob::new(data.clone()))
            .partition_key(partition_key)
            .build()
            .map_err(|e| StreamError::Backend(format!("Failed to build record: {}", e)))?;

        let response = self.client
            .put_records()
            .stream_name(&stream_name)
            .records(record)
            .send()
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to put record: {}", e)))?;

        if response.failed_record_count() > 0 {
            self.stats.write().await.records_failed += 1;
            return Err(StreamError::Backend("Failed to put record to Kinesis".to_string()));
        }

        self.stats.write().await.records_sent += 1;
        self.stats.write().await.bytes_sent += data.len() as u64;
        self.stats.write().await.put_records_calls += 1;

        // Use timestamp as offset (Kinesis doesn't provide immediate sequence numbers)
        Ok(Offset::new(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64))
    }

    async fn send_batch(&self, topic: &TopicName, events: Vec<StreamEvent>) -> StreamResult<Vec<Offset>> {
        let stream_name = self.get_stream_name(topic);
        let mut offsets = Vec::new();
        
        // Process in chunks due to Kinesis batch limits
        for chunk in events.chunks(MAX_BATCH_SIZE) {
            let mut records = Vec::new();
            
            for event in chunk {
                let kinesis_record = KinesisEventRecord::from(event.clone());
                let data = serde_json::to_vec(&kinesis_record)
                    .map_err(|e| StreamError::Serialization(e.to_string()))?;

                if data.len() > MAX_RECORD_SIZE {
                    self.stats.write().await.records_failed += 1;
                    continue;
                }

                let record = PutRecordsRequestEntry::builder()
                    .data(aws_sdk_kinesis::primitives::Blob::new(data.clone()))
                    .partition_key(kinesis_record.event_id)
                    .build()
                    .map_err(|e| StreamError::Backend(format!("Failed to build record: {}", e)))?;

                records.push(record);
                self.stats.write().await.bytes_sent += data.len() as u64;
            }

            if !records.is_empty() {
                let response = self.client
                    .put_records()
                    .stream_name(&stream_name)
                    .set_records(Some(records))
                    .send()
                    .await
                    .map_err(|e| StreamError::Backend(format!("Failed to put records: {}", e)))?;

                self.stats.write().await.put_records_calls += 1;
                self.stats.write().await.records_sent += (chunk.len() - response.failed_record_count() as usize) as u64;
                self.stats.write().await.records_failed += response.failed_record_count() as u64;

                // Generate offsets for successful records
                for _ in 0..chunk.len() {
                    offsets.push(Offset::new(
                        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
                    ));
                }
            }
        }

        Ok(offsets)
    }

    async fn receive_events(
        &self,
        topic: &TopicName,
        consumer_group: Option<&ConsumerGroup>,
        position: StreamPosition,
        max_events: usize,
    ) -> StreamResult<Vec<(StreamEvent, Offset)>> {
        let stream_name = self.get_stream_name(topic);
        let mut events = Vec::new();

        // Get stream description to find shards
        let describe_result = self.client
            .describe_stream()
            .stream_name(&stream_name)
            .send()
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to describe stream: {}", e)))?;

        let shards = describe_result
            .stream_description
            .and_then(|d| d.shards)
            .unwrap_or_default();

        if shards.is_empty() {
            return Ok(events);
        }

        // Initialize consumer state if needed
        let consumer_key = consumer_group
            .map(|g| format!("{}:{}", g.name(), stream_name))
            .unwrap_or_else(|| format!("default:{}", stream_name));

        let mut consumer_state = self.consumer_state.lock().await;
        let state = consumer_state.entry(consumer_key.clone()).or_insert_with(|| {
            ConsumerState {
                consumer_group: consumer_group.map(|g| g.name().to_string()).unwrap_or_default(),
                stream_name: stream_name.clone(),
                shard_iterators: HashMap::new(),
                last_sequence_numbers: HashMap::new(),
                checkpoint_map: HashMap::new(),
            }
        });

        // Process each shard
        for shard in shards.iter() {
            let shard_id = match &shard.shard_id {
                Some(id) => id,
                None => continue,
            };

            // Get or create shard iterator
            let iterator = if let Some(iter) = state.shard_iterators.get(shard_id) {
                iter.clone()
            } else {
                let iterator_type = match position {
                    StreamPosition::Beginning => ShardIteratorType::TrimHorizon,
                    StreamPosition::End => ShardIteratorType::Latest,
                    StreamPosition::Offset(_) => ShardIteratorType::AtSequenceNumber,
                };

                let sequence_number = if matches!(position, StreamPosition::Offset(_)) {
                    state.last_sequence_numbers.get(shard_id).cloned()
                } else {
                    None
                };

                let iterator = self.get_shard_iterator(
                    &stream_name,
                    shard_id,
                    iterator_type,
                    sequence_number,
                ).await?;

                state.shard_iterators.insert(shard_id.clone(), iterator.clone());
                iterator
            };

            // Get records
            let get_records_result = self.client
                .get_records()
                .shard_iterator(&iterator)
                .limit((max_events - events.len()) as i32)
                .send()
                .await;

            match get_records_result {
                Ok(response) => {
                    self.stats.write().await.get_records_calls += 1;

                    // Update iterator for next call
                    if let Some(next_iterator) = response.next_shard_iterator {
                        state.shard_iterators.insert(shard_id.clone(), next_iterator);
                    }

                    // Process records
                    for record in response.records() {
                        if events.len() >= max_events {
                            break;
                        }

                        let data = record.data.as_ref();
                        match serde_json::from_slice::<KinesisEventRecord>(data) {
                            Ok(kinesis_record) => {
                                match kinesis_record.to_stream_event() {
                                    Ok(event) => {
                                        let offset = Offset::new(
                                            record.sequence_number()
                                                .and_then(|s| s.parse::<u64>().ok())
                                                .unwrap_or(0)
                                        );
                                        events.push((event, offset));
                                        self.stats.write().await.records_received += 1;
                                        self.stats.write().await.bytes_received += data.len() as u64;

                                        // Update sequence number
                                        if let Some(seq) = record.sequence_number() {
                                            state.last_sequence_numbers.insert(shard_id.clone(), seq.to_string());
                                        }
                                    }
                                    Err(e) => {
                                        error!("Failed to convert Kinesis record to stream event: {}", e);
                                        self.stats.write().await.records_failed += 1;
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Failed to deserialize Kinesis record: {}", e);
                                self.stats.write().await.records_failed += 1;
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to get records from shard {}: {}", shard_id, e);
                    if e.to_string().contains("ExpiredIteratorException") {
                        // Reset iterator on expiration
                        state.shard_iterators.remove(shard_id);
                    }
                }
            }
        }

        Ok(events)
    }

    async fn commit_offset(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        partition: PartitionId,
        offset: Offset,
    ) -> StreamResult<()> {
        let stream_name = self.get_stream_name(topic);
        let consumer_key = format!("{}:{}", consumer_group.name(), stream_name);

        let mut consumer_state = self.consumer_state.lock().await;
        if let Some(state) = consumer_state.get_mut(&consumer_key) {
            state.checkpoint_map.insert(partition.to_string(), offset.value());
        }

        debug!("Committed offset {} for partition {} in stream {}", offset, partition, stream_name);
        Ok(())
    }

    async fn seek(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        _partition: PartitionId,
        position: StreamPosition,
    ) -> StreamResult<()> {
        let stream_name = self.get_stream_name(topic);
        let consumer_key = format!("{}:{}", consumer_group.name(), stream_name);

        let mut consumer_state = self.consumer_state.lock().await;
        if let Some(state) = consumer_state.get_mut(&consumer_key) {
            // Clear existing iterators to force recreation with new position
            state.shard_iterators.clear();
            
            if let StreamPosition::Offset(offset) = position {
                // Store offset as sequence number hint
                for shard_id in state.last_sequence_numbers.keys() {
                    state.last_sequence_numbers.insert(shard_id.clone(), offset.to_string());
                }
            }
        }

        info!("Seek consumer group {} to position {:?} for stream {}", consumer_group.name(), position, stream_name);
        Ok(())
    }

    async fn get_consumer_lag(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
    ) -> StreamResult<HashMap<PartitionId, u64>> {
        let stream_name = self.get_stream_name(topic);
        let consumer_key = format!("{}:{}", consumer_group.name(), stream_name);
        
        let mut lag_map = HashMap::new();

        // Get stream metrics
        let describe_result = self.client
            .describe_stream_summary()
            .stream_name(&stream_name)
            .send()
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to describe stream: {}", e)))?;

        if let Some(summary) = describe_result.stream_description_summary {
            let consumer_state = self.consumer_state.lock().await;
            if let Some(state) = consumer_state.get(&consumer_key) {
                // Estimate lag based on checkpoint map
                for (partition_str, checkpoint) in &state.checkpoint_map {
                    if let Ok(partition_id) = partition_str.parse::<u32>() {
                        // This is a simplified lag calculation
                        // In production, you'd need to track the latest sequence numbers
                        lag_map.insert(PartitionId::new(partition_id), 0);
                    }
                }
            } else {
                // No consumer state, assume full lag
                for i in 0..summary.open_shard_count.unwrap_or(0) as u32 {
                    lag_map.insert(PartitionId::new(i), u64::MAX);
                }
            }
        }

        Ok(lag_map)
    }

    async fn get_topic_metadata(&self, topic: &TopicName) -> StreamResult<HashMap<String, String>> {
        let stream_name = self.get_stream_name(topic);
        let mut metadata = HashMap::new();

        let describe_result = self.client
            .describe_stream_summary()
            .stream_name(&stream_name)
            .send()
            .await
            .map_err(|e| StreamError::Backend(format!("Failed to describe stream: {}", e)))?;

        if let Some(summary) = describe_result.stream_description_summary {
            metadata.insert("backend".to_string(), "kinesis".to_string());
            metadata.insert("stream_name".to_string(), stream_name);
            metadata.insert("stream_arn".to_string(), summary.stream_arn.unwrap_or_default());
            metadata.insert("status".to_string(), format!("{:?}", summary.stream_status.unwrap_or(StreamStatus::Creating)));
            metadata.insert("creation_timestamp".to_string(), 
                summary.stream_creation_timestamp
                    .map(|t| t.to_string())
                    .unwrap_or_default()
            );
            metadata.insert("retention_hours".to_string(), 
                summary.retention_period_hours.unwrap_or(0).to_string()
            );
            metadata.insert("open_shard_count".to_string(), 
                summary.open_shard_count.unwrap_or(0).to_string()
            );
            metadata.insert("encryption_type".to_string(),
                summary.encryption_type
                    .map(|e| format!("{:?}", e))
                    .unwrap_or_else(|| "NONE".to_string())
            );
        }

        // Add stats
        let stats = self.stats.read().await;
        metadata.insert("records_sent".to_string(), stats.records_sent.to_string());
        metadata.insert("records_received".to_string(), stats.records_received.to_string());
        metadata.insert("bytes_sent".to_string(), stats.bytes_sent.to_string());
        metadata.insert("bytes_received".to_string(), stats.bytes_received.to_string());

        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::StreamEventType;

    #[test]
    fn test_kinesis_event_conversion() {
        let event = StreamEvent {
            event_type: StreamEventType::TripleAdded {
                subject: "http://example.org/s".to_string(),
                predicate: "http://example.org/p".to_string(),
                object: "http://example.org/o".to_string(),
                graph: None,
            },
            timestamp: 12345,
            metadata: Some(EventMetadata {
                source: "test".to_string(),
                user: Some("user1".to_string()),
                session_id: None,
                trace_id: None,
                causality_token: None,
                version: None,
            }),
        };

        let kinesis_record = KinesisEventRecord::from(event.clone());
        assert_eq!(kinesis_record.event_type, "triple_added");
        assert_eq!(kinesis_record.timestamp, 12345);

        let converted_event = kinesis_record.to_stream_event().unwrap();
        assert_eq!(converted_event.timestamp, event.timestamp);
    }

    #[tokio::test]
    async fn test_kinesis_backend_creation() {
        let config = KinesisBackendConfig::default();
        let backend = KinesisBackend::new(config).await;
        assert!(backend.is_ok());
    }
}