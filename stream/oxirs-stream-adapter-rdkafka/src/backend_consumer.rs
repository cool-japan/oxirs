//! Consumer-side operations for the Kafka backend.
//!
//! Implements persistent consumer creation, streaming consumption with
//! callbacks, lifecycle management (start/stop/pause/resume), offset seeking,
//! and per-consumer metrics for [`KafkaBackend`].

use super::backend_producer::KafkaBackend;
use super::backend_types::{ConsumerId, ConsumerMetrics, MessageCallback, PartitionAssignment};
use anyhow::Result;
use std::sync::atomic::Ordering;
use tracing::info;
use uuid::Uuid;

#[cfg(feature = "kafka")]
use super::backend_types::ConsumerInstance;
#[cfg(feature = "kafka")]
use super::KafkaEvent;
#[cfg(feature = "kafka")]
use rdkafka::{
    config::ClientConfig,
    consumer::{Consumer, StreamConsumer},
    message::{BorrowedMessage, Message},
};
#[cfg(feature = "kafka")]
use std::sync::{
    atomic::{AtomicBool, AtomicU64},
    Arc,
};
#[cfg(feature = "kafka")]
use std::time::Duration;
#[cfg(feature = "kafka")]
use tokio::sync::{oneshot, RwLock};
#[cfg(feature = "kafka")]
use tracing::error;

impl KafkaBackend {
    /// Create a persistent consumer for a consumer group
    #[cfg(feature = "kafka")]
    pub async fn create_persistent_consumer(
        &mut self,
        group_id: &str,
        topics: Vec<String>,
    ) -> Result<ConsumerId> {
        let consumer_id = Uuid::new_v4();
        let mut client_config = ClientConfig::new();

        // Apply configuration from KafkaProducerConfig
        self.apply_client_config(&mut client_config);

        // Consumer-specific configuration
        client_config
            .set("group.id", group_id)
            .set("enable.partition.eof", "false")
            .set("session.timeout.ms", "30000")
            .set("enable.auto.commit", "false")
            .set("auto.offset.reset", "earliest");

        // Create consumer without custom context (use default)
        let consumer: StreamConsumer = client_config
            .create()
            .map_err(|e| anyhow::anyhow!("Failed to create consumer: {}", e))?;

        // Subscribe to topics
        let topic_refs: Vec<&str> = topics.iter().map(|s| s.as_str()).collect();
        consumer
            .subscribe(&topic_refs)
            .map_err(|e| anyhow::anyhow!("Failed to subscribe to topics: {}", e))?;

        let consumer_instance = ConsumerInstance {
            id: consumer_id,
            group_id: group_id.to_string(),
            consumer: Arc::new(consumer),
            is_active: Arc::new(AtomicBool::new(true)),
            message_count: Arc::new(AtomicU64::new(0)),
            error_count: Arc::new(AtomicU64::new(0)),
            last_message_time: Arc::new(RwLock::new(None)),
            stop_signal: None,
        };

        // Store the consumer instance
        let mut consumers = self.active_consumers.write().await;
        consumers.insert(consumer_id, consumer_instance);

        info!(
            "Created persistent consumer {} for group {}",
            consumer_id, group_id
        );
        Ok(consumer_id)
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn create_persistent_consumer(
        &mut self,
        group_id: &str,
        topics: Vec<String>,
    ) -> Result<ConsumerId> {
        let consumer_id = Uuid::new_v4();
        info!(
            "Mock consumer {} created for group {} with topics {:?}",
            consumer_id, group_id, topics
        );
        Ok(consumer_id)
    }

    /// Start streaming consumer with callback-based processing
    #[cfg(feature = "kafka")]
    pub async fn start_streaming_consumer(
        &self,
        consumer_id: ConsumerId,
        callback: MessageCallback,
    ) -> Result<()> {
        let consumers = self.active_consumers.read().await;
        let consumer_instance = consumers
            .get(&consumer_id)
            .ok_or_else(|| anyhow::anyhow!("Consumer {} not found", consumer_id))?;

        if !consumer_instance.is_active.load(Ordering::SeqCst) {
            return Err(anyhow::anyhow!("Consumer {} is not active", consumer_id));
        }

        let consumer = consumer_instance.consumer.clone();
        let is_active = consumer_instance.is_active.clone();
        let message_count = consumer_instance.message_count.clone();
        let error_count = consumer_instance.error_count.clone();
        let last_message_time = consumer_instance.last_message_time.clone();
        let (stop_tx, mut stop_rx) = oneshot::channel();

        // Update the consumer instance with the stop signal
        drop(consumers);
        let mut consumers = self.active_consumers.write().await;
        if let Some(instance) = consumers.get_mut(&consumer_id) {
            instance.stop_signal = Some(stop_tx);
        }
        drop(consumers);

        // Spawn the consumer loop
        let consumer_loop = async move {
            loop {
                tokio::select! {
                    _ = &mut stop_rx => {
                        info!("Consumer {} received stop signal", consumer_id);
                        break;
                    }
                    message_result = consumer.recv() => {
                        match message_result {
                            Ok(borrowed_message) => {
                                // Update last message time
                                {
                                    let mut time_guard = last_message_time.write().await;
                                    *time_guard = Some(chrono::Utc::now());
                                }

                                match Self::process_kafka_message(&borrowed_message, &callback).await {
                                    Ok(_) => {
                                        message_count.fetch_add(1, Ordering::SeqCst);
                                        // Commit the message offset
                                        if let Err(e) = consumer.commit_message(&borrowed_message, rdkafka::consumer::CommitMode::Async) {
                                            error!("Failed to commit message: {}", e);
                                            error_count.fetch_add(1, Ordering::SeqCst);
                                        }
                                    }
                                    Err(e) => {
                                        error!("Error processing message: {}", e);
                                        error_count.fetch_add(1, Ordering::SeqCst);
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Error receiving message: {}", e);
                                error_count.fetch_add(1, Ordering::SeqCst);
                                tokio::time::sleep(Duration::from_millis(100)).await;
                            }
                        }
                    }
                }
            }
            is_active.store(false, Ordering::SeqCst);
        };

        tokio::spawn(consumer_loop);
        info!("Started streaming consumer {}", consumer_id);
        Ok(())
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn start_streaming_consumer(
        &self,
        consumer_id: ConsumerId,
        _callback: MessageCallback,
    ) -> Result<()> {
        info!("Mock streaming consumer {} started", consumer_id);
        Ok(())
    }

    /// Process a Kafka message and convert it to StreamEvent
    #[cfg(feature = "kafka")]
    async fn process_kafka_message(
        message: &BorrowedMessage<'_>,
        callback: &MessageCallback,
    ) -> Result<()> {
        let payload = message
            .payload()
            .ok_or_else(|| anyhow::anyhow!("Message has no payload"))?;

        let kafka_event = KafkaEvent::from_bytes(payload)?;
        let stream_event = kafka_event.to_stream_event();

        callback(stream_event)?;
        Ok(())
    }

    /// Stop a streaming consumer
    pub async fn stop_consumer(&mut self, consumer_id: ConsumerId) -> Result<()> {
        let mut consumers = self.active_consumers.write().await;
        if let Some(mut instance) = consumers.remove(&consumer_id) {
            instance.is_active.store(false, Ordering::SeqCst);
            if let Some(stop_signal) = instance.stop_signal.take() {
                let _ = stop_signal.send(());
            }
            info!("Stopped consumer {}", consumer_id);
        }
        Ok(())
    }

    /// Pause a consumer
    #[cfg(feature = "kafka")]
    pub async fn pause_consumer(&self, consumer_id: ConsumerId) -> Result<()> {
        let consumers = self.active_consumers.read().await;
        if let Some(instance) = consumers.get(&consumer_id) {
            instance
                .consumer
                .pause(&instance.consumer.assignment()?)
                .map_err(|e| anyhow::anyhow!("Failed to pause consumer: {}", e))?;
            info!("Paused consumer {}", consumer_id);
        } else {
            return Err(anyhow::anyhow!("Consumer {} not found", consumer_id));
        }
        Ok(())
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn pause_consumer(&self, consumer_id: ConsumerId) -> Result<()> {
        info!("Mock consumer {} paused", consumer_id);
        Ok(())
    }

    /// Resume a consumer
    #[cfg(feature = "kafka")]
    pub async fn resume_consumer(&self, consumer_id: ConsumerId) -> Result<()> {
        let consumers = self.active_consumers.read().await;
        if let Some(instance) = consumers.get(&consumer_id) {
            instance
                .consumer
                .resume(&instance.consumer.assignment()?)
                .map_err(|e| anyhow::anyhow!("Failed to resume consumer: {}", e))?;
            info!("Resumed consumer {}", consumer_id);
        } else {
            return Err(anyhow::anyhow!("Consumer {} not found", consumer_id));
        }
        Ok(())
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn resume_consumer(&self, consumer_id: ConsumerId) -> Result<()> {
        info!("Mock consumer {} resumed", consumer_id);
        Ok(())
    }

    /// Get consumer metrics
    pub async fn get_consumer_metrics(&self, consumer_id: ConsumerId) -> Result<ConsumerMetrics> {
        let consumers = self.active_consumers.read().await;
        if let Some(instance) = consumers.get(&consumer_id) {
            let partition_assignments = self.get_partition_assignments(&instance.consumer).await?;
            let last_message_time = *instance.last_message_time.read().await;
            Ok(ConsumerMetrics {
                consumer_id,
                group_id: instance.group_id.clone(),
                messages_processed: instance.message_count.load(Ordering::SeqCst),
                errors_encountered: instance.error_count.load(Ordering::SeqCst),
                is_active: instance.is_active.load(Ordering::SeqCst),
                last_message_time,
                partition_assignments,
            })
        } else {
            Err(anyhow::anyhow!("Consumer {} not found", consumer_id))
        }
    }

    /// Get partition assignments for a consumer
    #[cfg(feature = "kafka")]
    async fn get_partition_assignments(
        &self,
        consumer: &Arc<StreamConsumer>,
    ) -> Result<Vec<PartitionAssignment>> {
        let assignment = consumer.assignment()?;
        let mut assignments = Vec::new();

        for partition in assignment.elements() {
            let topic = partition.topic();
            let partition_id = partition.partition();
            let current_offset = partition.offset().to_raw().unwrap_or(-1);

            // Get high water mark (latest offset)
            let high_water_mark = consumer
                .fetch_watermarks(topic, partition_id, Duration::from_secs(1))
                .map(|(_, high)| high)
                .unwrap_or(-1);

            let lag = if current_offset >= 0 && high_water_mark >= 0 {
                high_water_mark - current_offset
            } else {
                0
            };

            assignments.push(PartitionAssignment {
                topic: topic.to_string(),
                partition: partition_id,
                current_offset,
                high_water_mark,
                lag,
            });
        }

        Ok(assignments)
    }

    #[cfg(not(feature = "kafka"))]
    async fn get_partition_assignments(&self, _consumer: &()) -> Result<Vec<PartitionAssignment>> {
        Ok(Vec::new())
    }

    /// Seek consumer to specific offset
    #[cfg(feature = "kafka")]
    pub async fn seek_consumer_to_offset(
        &self,
        consumer_id: ConsumerId,
        topic: &str,
        partition: i32,
        offset: i64,
    ) -> Result<()> {
        let consumers = self.active_consumers.read().await;
        if let Some(instance) = consumers.get(&consumer_id) {
            use rdkafka::{Offset, TopicPartitionList};
            let mut topic_partition_list = TopicPartitionList::new();
            topic_partition_list.add_partition_offset(topic, partition, Offset::Offset(offset))?;
            instance
                .consumer
                .seek_partitions(topic_partition_list, Duration::from_secs(10))
                .map_err(|e| anyhow::anyhow!("Failed to seek consumer: {}", e))?;
            info!(
                "Seeked consumer {} to offset {} on {}:{}",
                consumer_id, offset, topic, partition
            );
        } else {
            return Err(anyhow::anyhow!("Consumer {} not found", consumer_id));
        }
        Ok(())
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn seek_consumer_to_offset(
        &self,
        consumer_id: ConsumerId,
        topic: &str,
        partition: i32,
        offset: i64,
    ) -> Result<()> {
        info!(
            "Mock consumer {} seeked to offset {} on {}:{}",
            consumer_id, offset, topic, partition
        );
        Ok(())
    }

    /// Get list of active consumers
    pub async fn get_active_consumers(&self) -> Vec<ConsumerId> {
        let consumers = self.active_consumers.read().await;
        consumers.keys().cloned().collect()
    }

    /// Get consumer count
    pub async fn get_consumer_count(&self) -> usize {
        let consumers = self.active_consumers.read().await;
        consumers.len()
    }

    /// Consume a single event (not supported - Kafka uses streaming consumers)
    ///
    /// Note: Kafka backend uses persistent streaming consumers with callbacks.
    /// Use create_persistent_consumer() and start_streaming_consumer() instead.
    pub async fn consume(&mut self) -> Result<Option<oxirs_stream::StreamEvent>> {
        Err(anyhow::anyhow!(
            "Direct consume() not supported for Kafka backend. \
             Use create_persistent_consumer() and start_streaming_consumer() instead."
        ))
    }
}
