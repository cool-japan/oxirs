//! [`StreamBackend`](oxirs_stream::backend::StreamBackend) trait implementation for [`KafkaBackend`].
//!
//! Wires the generic streaming backend interface (connect/disconnect, topic
//! administration, event send/receive, offset commit/seek, lag and metadata
//! queries) onto the Kafka-specific [`KafkaBackend`].

use super::backend_producer::KafkaBackend;
use super::KafkaEvent;
use async_trait::async_trait;
use oxirs_stream::backend::StreamBackend as StreamBackendTrait;
use oxirs_stream::consumer::ConsumerGroup;
use oxirs_stream::error::{StreamError, StreamResult};
use oxirs_stream::types::{Offset, PartitionId, StreamPosition, TopicName};
use oxirs_stream::StreamEvent;
use std::collections::HashMap;
use tracing::info;

#[cfg(feature = "kafka")]
use rdkafka::{
    admin::{AdminClient, AdminOptions, NewTopic, TopicReplication},
    config::ClientConfig,
    producer::{FutureProducer, Producer},
};
#[cfg(feature = "kafka")]
use std::time::Duration;
#[cfg(feature = "kafka")]
use tracing::{debug, warn};

#[async_trait]
impl StreamBackendTrait for KafkaBackend {
    fn name(&self) -> &'static str {
        "kafka"
    }

    async fn connect(&mut self) -> StreamResult<()> {
        #[cfg(feature = "kafka")]
        {
            // Initialize Kafka producer
            let mut kafka_config = ClientConfig::new();
            kafka_config
                .set("bootstrap.servers", self.kafka_config.brokers.join(","))
                .set(
                    "message.timeout.ms",
                    self.kafka_config.request_timeout_ms.to_string(),
                )
                .set(
                    "compression.type",
                    self.kafka_config.compression_type.to_string(),
                )
                .set("acks", self.kafka_config.acks.to_string())
                .set("client.id", &self.kafka_config.client_id);

            let producer: FutureProducer = kafka_config.create().map_err(|e| {
                StreamError::Connection(format!("Failed to create Kafka producer: {}", e))
            })?;

            self.producer = Some(producer);

            // Initialize admin client for topic management
            let admin_client: AdminClient<_> = kafka_config.create().map_err(|e| {
                StreamError::Connection(format!("Failed to create Kafka admin client: {}", e))
            })?;

            self.admin_client = Some(admin_client);

            info!(
                "Connected to Kafka cluster: {}",
                self.kafka_config.brokers.join(",")
            );
        }

        #[cfg(not(feature = "kafka"))]
        {
            info!("Mock Kafka connection (kafka feature not enabled)");
        }

        Ok(())
    }

    async fn disconnect(&mut self) -> StreamResult<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(producer) = self.producer.take() {
                // Flush any pending messages
                producer.flush(Duration::from_secs(5)).map_err(|e| {
                    StreamError::Connection(format!("Failed to flush Kafka producer: {}", e))
                })?;
            }
            self.admin_client = None;
            self.consumer = None;
        }

        info!("Disconnected from Kafka");
        Ok(())
    }

    async fn create_topic(&self, topic: &TopicName, partitions: u32) -> StreamResult<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref admin_client) = self.admin_client {
                let new_topic = NewTopic::new(
                    topic.as_str(),
                    partitions as i32,
                    TopicReplication::Fixed(1),
                );
                let opts = AdminOptions::new().operation_timeout(Some(Duration::from_secs(30)));

                admin_client
                    .create_topics(&[new_topic], &opts)
                    .await
                    .map_err(|e| {
                        StreamError::TopicCreation(format!(
                            "Failed to create topic {}: {}",
                            topic, e
                        ))
                    })?;

                info!(
                    "Created Kafka topic: {} with {} partitions",
                    topic, partitions
                );
            } else {
                return Err(StreamError::Connection(
                    "Admin client not initialized".to_string(),
                ));
            }
        }

        #[cfg(not(feature = "kafka"))]
        {
            info!(
                "Mock create topic: {} with {} partitions",
                topic, partitions
            );
        }

        Ok(())
    }

    async fn delete_topic(&self, topic: &TopicName) -> StreamResult<()> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref admin_client) = self.admin_client {
                let opts = AdminOptions::new().operation_timeout(Some(Duration::from_secs(30)));

                admin_client
                    .delete_topics(&[topic.as_str()], &opts)
                    .await
                    .map_err(|e| {
                        StreamError::TopicDeletion(format!(
                            "Failed to delete topic {}: {}",
                            topic, e
                        ))
                    })?;

                info!("Deleted Kafka topic: {}", topic);
            } else {
                return Err(StreamError::Connection(
                    "Admin client not initialized".to_string(),
                ));
            }
        }

        #[cfg(not(feature = "kafka"))]
        {
            info!("Mock delete topic: {}", topic);
        }

        Ok(())
    }

    async fn list_topics(&self) -> StreamResult<Vec<TopicName>> {
        #[cfg(feature = "kafka")]
        {
            if let Some(ref admin_client) = self.admin_client {
                let metadata = admin_client
                    .inner()
                    .fetch_metadata(None, Duration::from_secs(10))
                    .map_err(|e| {
                        StreamError::TopicList(format!("Failed to fetch metadata: {}", e))
                    })?;

                Ok(metadata
                    .topics()
                    .iter()
                    .map(|topic| topic.name().to_string().into())
                    .collect())
            } else {
                Err(StreamError::Connection(
                    "Admin client not initialized".to_string(),
                ))
            }
        }

        #[cfg(not(feature = "kafka"))]
        {
            Ok(vec!["mock-topic-1".to_string(), "mock-topic-2".to_string()])
        }
    }

    async fn send_event(&self, topic: &TopicName, event: StreamEvent) -> StreamResult<Offset> {
        #[cfg(feature = "kafka")]
        {
            use rdkafka::producer::FutureRecord;

            if let Some(ref producer) = self.producer {
                let kafka_event = KafkaEvent::from_stream_event(event);
                let serialized = serde_json::to_string(&kafka_event)
                    .map_err(|e| StreamError::Serialization(e.to_string()))?;

                let record = FutureRecord::to(topic.as_str())
                    .key(&kafka_event.event_id)
                    .payload(&serialized);

                let result = producer
                    .send(record, Duration::from_secs(5))
                    .await
                    .map_err(|(e, _)| {
                        StreamError::Send(format!("Failed to send to Kafka: {}", e))
                    })?;

                Ok(Offset::new(result.offset as u64))
            } else {
                Err(StreamError::Connection(
                    "Producer not initialized".to_string(),
                ))
            }
        }

        #[cfg(not(feature = "kafka"))]
        {
            debug!("Mock send event to topic: {}", topic);
            Ok(Offset::new(0))
        }
    }

    async fn send_batch(
        &self,
        topic: &TopicName,
        events: Vec<StreamEvent>,
    ) -> StreamResult<Vec<Offset>> {
        let mut offsets = Vec::new();
        for event in events {
            let offset = self.send_event(topic, event).await?;
            offsets.push(offset);
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
        #[cfg(feature = "kafka")]
        {
            use rdkafka::config::ClientConfig;
            use rdkafka::consumer::{Consumer, StreamConsumer};
            use rdkafka::message::Message;
            use rdkafka::TopicPartitionList;
            use tokio::time::{timeout, Duration};

            let group_id = consumer_group
                .map(|cg| cg.name())
                .unwrap_or("oxirs-default-group");

            // Create consumer configuration
            let mut consumer_config = ClientConfig::new();
            consumer_config
                .set("bootstrap.servers", self.kafka_config.brokers.join(","))
                .set("group.id", group_id)
                .set("enable.auto.commit", "false") // Manual commit for better control
                .set(
                    "auto.offset.reset",
                    match position {
                        StreamPosition::Beginning => "earliest",
                        StreamPosition::End => "latest",
                        StreamPosition::Offset(_) => "none", // Will seek manually
                    },
                )
                .set("session.timeout.ms", "30000")
                .set("heartbeat.interval.ms", "3000");

            // Apply security configuration if available
            if let Some(ref security_config) = self.kafka_config.security_config {
                consumer_config.set(
                    "security.protocol",
                    security_config.security_protocol.to_string(),
                );

                if let Some(ref sasl_config) = security_config.sasl_config {
                    consumer_config
                        .set("sasl.mechanism", sasl_config.mechanism.to_string())
                        .set("sasl.username", &sasl_config.username)
                        .set("sasl.password", &sasl_config.password);
                }
            }

            let consumer: StreamConsumer = consumer_config.create().map_err(|e| {
                StreamError::Connection(format!("Failed to create Kafka consumer: {}", e))
            })?;

            // Subscribe to topic
            consumer.subscribe(&[topic.as_str()]).map_err(|e| {
                StreamError::Connection(format!("Failed to subscribe to topic: {}", e))
            })?;

            // Handle specific offset positioning
            if let StreamPosition::Offset(offset_value) = position {
                let mut tpl = TopicPartitionList::new();
                tpl.add_partition_offset(
                    topic.as_str(),
                    0,
                    rdkafka::Offset::Offset(offset_value as i64),
                )
                .map_err(|e| {
                    StreamError::SeekError(format!("Failed to add partition offset: {}", e))
                })?;

                consumer
                    .seek_partitions(tpl, Duration::from_secs(10))
                    .map_err(|e| {
                        StreamError::SeekError(format!("Failed to seek to offset: {}", e))
                    })?;
            }

            let mut events = Vec::with_capacity(max_events);
            let timeout_duration = Duration::from_secs(5);

            // Consume messages up to max_events
            while events.len() < max_events {
                match timeout(timeout_duration, consumer.recv()).await {
                    Ok(Ok(message)) => {
                        if let Some(payload) = message.payload() {
                            // Deserialize the KafkaEvent
                            match serde_json::from_slice::<KafkaEvent>(payload) {
                                Ok(kafka_event) => {
                                    let event_id = kafka_event.event_id.clone();
                                    let stream_event = kafka_event.to_stream_event();
                                    let offset = Offset::new(message.offset() as u64);
                                    events.push((stream_event, offset));

                                    debug!(
                                        "Received event from Kafka: {} at offset {}",
                                        event_id,
                                        message.offset()
                                    );
                                }
                                Err(e) => {
                                    warn!("Failed to deserialize Kafka message: {}", e);
                                    continue;
                                }
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        warn!("Kafka receive error: {}", e);
                        break;
                    }
                    Err(_) => {
                        // Timeout - return what we have
                        debug!("Kafka receive timeout, returning {} events", events.len());
                        break;
                    }
                }
            }

            info!(
                "Received {} events from Kafka topic: {}",
                events.len(),
                topic
            );
            Ok(events)
        }

        #[cfg(not(feature = "kafka"))]
        {
            debug!(
                "Mock receive events from topic: {}, max: {}",
                topic, max_events
            );
            Ok(vec![])
        }
    }

    async fn commit_offset(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        partition: PartitionId,
        offset: Offset,
    ) -> StreamResult<()> {
        #[cfg(feature = "kafka")]
        {
            use rdkafka::config::ClientConfig;
            use rdkafka::consumer::{BaseConsumer, CommitMode, Consumer};
            use rdkafka::TopicPartitionList;

            // Create consumer configuration for the specific group
            let mut consumer_config = ClientConfig::new();
            consumer_config
                .set("bootstrap.servers", self.kafka_config.brokers.join(","))
                .set("group.id", consumer_group.name())
                .set("enable.auto.commit", "false")
                .set("session.timeout.ms", "30000")
                .set("heartbeat.interval.ms", "3000");

            // Apply security configuration if available
            if let Some(ref security_config) = self.kafka_config.security_config {
                consumer_config.set(
                    "security.protocol",
                    security_config.security_protocol.to_string(),
                );

                if let Some(ref sasl_config) = security_config.sasl_config {
                    consumer_config
                        .set("sasl.mechanism", sasl_config.mechanism.to_string())
                        .set("sasl.username", &sasl_config.username)
                        .set("sasl.password", &sasl_config.password);
                }
            }

            let consumer: BaseConsumer = consumer_config.create().map_err(|e| {
                StreamError::Connection(format!(
                    "Failed to create Kafka consumer for commit: {}",
                    e
                ))
            })?;

            // Create TopicPartitionList with the specific offset to commit
            let mut tpl = TopicPartitionList::new();
            tpl.add_partition_offset(
                topic.as_str(),
                partition.value() as i32,
                rdkafka::Offset::Offset(offset.value() as i64 + 1), // Kafka commits the next offset
            )
            .map_err(|e| {
                StreamError::CommitError(format!(
                    "Failed to add partition offset for commit: {}",
                    e
                ))
            })?;

            // Commit the offset
            consumer
                .commit(&tpl, CommitMode::Sync)
                .map_err(|e| StreamError::CommitError(format!("Failed to commit offset: {}", e)))?;

            debug!(
                "Committed offset {} for topic: {}, partition: {}, group: {}",
                offset.value(),
                topic,
                partition.value(),
                consumer_group.name()
            );
            Ok(())
        }

        #[cfg(not(feature = "kafka"))]
        {
            debug!(
                "Mock commit offset for topic: {}, partition: {}, offset: {}",
                topic,
                partition.value(),
                offset.value()
            );
            Ok(())
        }
    }

    async fn seek(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        partition: PartitionId,
        position: StreamPosition,
    ) -> StreamResult<()> {
        #[cfg(feature = "kafka")]
        {
            use rdkafka::config::ClientConfig;
            use rdkafka::consumer::{BaseConsumer, Consumer};
            use rdkafka::TopicPartitionList;
            use tokio::time::Duration;

            // Create consumer configuration for the specific group
            let mut consumer_config = ClientConfig::new();
            consumer_config
                .set("bootstrap.servers", self.kafka_config.brokers.join(","))
                .set("group.id", consumer_group.name())
                .set("enable.auto.commit", "false")
                .set("session.timeout.ms", "30000")
                .set("heartbeat.interval.ms", "3000");

            // Apply security configuration if available
            if let Some(ref security_config) = self.kafka_config.security_config {
                consumer_config.set(
                    "security.protocol",
                    security_config.security_protocol.to_string(),
                );

                if let Some(ref sasl_config) = security_config.sasl_config {
                    consumer_config
                        .set("sasl.mechanism", sasl_config.mechanism.to_string())
                        .set("sasl.username", &sasl_config.username)
                        .set("sasl.password", &sasl_config.password);
                }
            }

            let consumer: BaseConsumer = consumer_config.create().map_err(|e| {
                StreamError::Connection(format!("Failed to create Kafka consumer for seek: {}", e))
            })?;

            // Create TopicPartitionList with the position to seek to
            let mut tpl = TopicPartitionList::new();

            let kafka_offset = match position {
                StreamPosition::Beginning => rdkafka::Offset::Beginning,
                StreamPosition::End => rdkafka::Offset::End,
                StreamPosition::Offset(offset_value) => {
                    rdkafka::Offset::Offset(offset_value as i64)
                }
            };

            tpl.add_partition_offset(topic.as_str(), partition.value() as i32, kafka_offset)
                .map_err(|e| {
                    StreamError::SeekError(format!("Failed to add partition for seek: {}", e))
                })?;

            // Perform the seek operation
            consumer
                .seek_partitions(tpl, Duration::from_secs(10))
                .map_err(|e| StreamError::SeekError(format!("Failed to seek: {}", e)))?;

            info!(
                "Seeked to position {:?} for topic: {}, partition: {}, group: {}",
                position,
                topic,
                partition.value(),
                consumer_group.name()
            );
            Ok(())
        }

        #[cfg(not(feature = "kafka"))]
        {
            debug!(
                "Mock seek for topic: {}, partition: {} to position: {:?}",
                topic,
                partition.value(),
                position
            );
            Ok(())
        }
    }

    async fn get_consumer_lag(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
    ) -> StreamResult<HashMap<PartitionId, u64>> {
        #[cfg(feature = "kafka")]
        {
            use rdkafka::config::ClientConfig;
            use rdkafka::consumer::{BaseConsumer, Consumer};
            use rdkafka::TopicPartitionList;
            use std::collections::HashMap;

            // Create consumer configuration for the specific group
            let mut consumer_config = ClientConfig::new();
            consumer_config
                .set("bootstrap.servers", self.kafka_config.brokers.join(","))
                .set("group.id", consumer_group.name())
                .set("enable.auto.commit", "false")
                .set("session.timeout.ms", "30000")
                .set("heartbeat.interval.ms", "3000");

            // Apply security configuration if available
            if let Some(ref security_config) = self.kafka_config.security_config {
                consumer_config.set(
                    "security.protocol",
                    security_config.security_protocol.to_string(),
                );

                if let Some(ref sasl_config) = security_config.sasl_config {
                    consumer_config
                        .set("sasl.mechanism", sasl_config.mechanism.to_string())
                        .set("sasl.username", &sasl_config.username)
                        .set("sasl.password", &sasl_config.password);
                }
            }

            let consumer: BaseConsumer = consumer_config.create().map_err(|e| {
                StreamError::Connection(format!(
                    "Failed to create Kafka consumer for lag check: {}",
                    e
                ))
            })?;

            // Get topic metadata to find all partitions
            let metadata = consumer
                .fetch_metadata(Some(topic.as_str()), Duration::from_secs(10))
                .map_err(|e| {
                    StreamError::TopicMetadata(format!("Failed to fetch topic metadata: {}", e))
                })?;

            let mut lag_map = HashMap::new();

            if let Some(topic_metadata) = metadata.topics().first() {
                for partition_metadata in topic_metadata.partitions() {
                    let partition_id = PartitionId::new(partition_metadata.id() as u32);

                    // Get high water mark (latest offset)
                    let mut tpl = TopicPartitionList::new();
                    tpl.add_partition_offset(
                        topic.as_str(),
                        partition_metadata.id(),
                        rdkafka::Offset::End,
                    )
                    .map_err(|e| {
                        StreamError::TopicMetadata(format!(
                            "Failed to add partition for high water mark: {}",
                            e
                        ))
                    })?;

                    let high_water_marks = consumer
                        .committed_offsets(tpl, Duration::from_secs(10))
                        .map_err(|e| {
                            StreamError::TopicMetadata(format!(
                                "Failed to get high water marks: {}",
                                e
                            ))
                        })?;

                    // Get consumer group's committed offset
                    let mut committed_tpl = TopicPartitionList::new();
                    committed_tpl.add_partition(topic.as_str(), partition_metadata.id());

                    let committed_offsets = consumer
                        .committed_offsets(committed_tpl, Duration::from_secs(10))
                        .map_err(|e| {
                            StreamError::TopicMetadata(format!(
                                "Failed to get committed offsets: {}",
                                e
                            ))
                        })?;

                    // Calculate lag
                    if let Some(high_water_element) = high_water_marks.elements().first() {
                        if let Some(committed_element) = committed_offsets.elements().first() {
                            let high_water_offset = match high_water_element.offset() {
                                rdkafka::Offset::Offset(offset) => offset as u64,
                                rdkafka::Offset::End => 0, // Topic is empty
                                _ => 0,
                            };

                            let committed_offset = match committed_element.offset() {
                                rdkafka::Offset::Offset(offset) => offset as u64,
                                rdkafka::Offset::Invalid => 0, // No committed offset yet
                                _ => 0,
                            };

                            let lag = high_water_offset.saturating_sub(committed_offset);

                            lag_map.insert(partition_id, lag);
                            debug!(
                                "Partition {}: lag = {} (high water: {}, committed: {})",
                                partition_metadata.id(),
                                lag,
                                high_water_offset,
                                committed_offset
                            );
                        }
                    }
                }
            }

            info!(
                "Retrieved consumer lag for topic: {} with {} partitions",
                topic,
                lag_map.len()
            );
            Ok(lag_map)
        }

        #[cfg(not(feature = "kafka"))]
        {
            debug!("Mock get consumer lag for topic: {}", topic);
            let mut lag_map = HashMap::new();
            lag_map.insert(PartitionId::new(0), 0);
            Ok(lag_map)
        }
    }

    async fn get_topic_metadata(&self, topic: &TopicName) -> StreamResult<HashMap<String, String>> {
        #[cfg(feature = "kafka")]
        {
            let mut metadata = HashMap::new();
            metadata.insert("backend".to_string(), "kafka".to_string());
            metadata.insert("topic".to_string(), topic.to_string());
            metadata.insert("brokers".to_string(), self.kafka_config.brokers.join(","));
            Ok(metadata)
        }

        #[cfg(not(feature = "kafka"))]
        {
            let mut metadata = HashMap::new();
            metadata.insert("backend".to_string(), "kafka".to_string());
            metadata.insert("topic".to_string(), topic.to_string());
            metadata.insert("brokers".to_string(), "mock-broker:9092".to_string());
            Ok(metadata)
        }
    }
}
