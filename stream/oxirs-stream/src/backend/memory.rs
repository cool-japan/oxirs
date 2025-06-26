//! # Memory Backend
//!
//! In-memory backend implementation for testing and development.

use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::backend::StreamBackend;
use crate::consumer::ConsumerGroup;
use crate::error::{StreamError, StreamResult};
use crate::event::StreamEvent;
use crate::types::{Offset, PartitionId, StreamPosition, TopicName};

/// In-memory stream storage
#[derive(Clone)]
struct MemoryStorage {
    topics: Arc<RwLock<HashMap<TopicName, TopicData>>>,
}

#[derive(Clone)]
struct TopicData {
    events: VecDeque<(StreamEvent, Offset)>,
    next_offset: u64,
    consumer_offsets: HashMap<String, u64>,
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self {
            topics: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

/// Memory backend for testing
pub struct MemoryBackend {
    storage: MemoryStorage,
    connected: bool,
}

impl MemoryBackend {
    pub fn new() -> Self {
        Self {
            storage: MemoryStorage::default(),
            connected: false,
        }
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StreamBackend for MemoryBackend {
    fn name(&self) -> &'static str {
        "memory"
    }

    async fn connect(&mut self) -> StreamResult<()> {
        self.connected = true;
        Ok(())
    }

    async fn disconnect(&mut self) -> StreamResult<()> {
        self.connected = false;
        Ok(())
    }

    async fn create_topic(&self, topic: &TopicName, _partitions: u32) -> StreamResult<()> {
        let mut topics = self.storage.topics.write().await;
        topics.entry(topic.clone()).or_insert_with(|| TopicData {
            events: VecDeque::new(),
            next_offset: 0,
            consumer_offsets: HashMap::new(),
        });
        Ok(())
    }

    async fn delete_topic(&self, topic: &TopicName) -> StreamResult<()> {
        let mut topics = self.storage.topics.write().await;
        topics.remove(topic);
        Ok(())
    }

    async fn list_topics(&self) -> StreamResult<Vec<TopicName>> {
        let topics = self.storage.topics.read().await;
        Ok(topics.keys().cloned().collect())
    }

    async fn send_event(&self, topic: &TopicName, event: StreamEvent) -> StreamResult<Offset> {
        let mut topics = self.storage.topics.write().await;
        let topic_data = topics
            .get_mut(topic)
            .ok_or_else(|| StreamError::TopicNotFound(topic.to_string()))?;

        let offset = Offset::new(topic_data.next_offset);
        topic_data.next_offset += 1;
        topic_data.events.push_back((event, offset));

        // Limit memory usage
        if topic_data.events.len() > 10000 {
            topic_data.events.pop_front();
        }

        Ok(offset)
    }

    async fn send_batch(&self, topic: &TopicName, events: Vec<StreamEvent>) -> StreamResult<Vec<Offset>> {
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
        let mut topics = self.storage.topics.write().await;
        let topic_data = topics
            .get_mut(topic)
            .ok_or_else(|| StreamError::TopicNotFound(topic.to_string()))?;

        let start_offset = if let Some(group) = consumer_group {
            let group_name = group.name();
            let current_offset = topic_data.consumer_offsets.get(group_name).copied().unwrap_or(0);
            
            match position {
                StreamPosition::Beginning => 0,
                StreamPosition::End => topic_data.next_offset,
                StreamPosition::Offset(offset) => offset,
            }
        } else {
            match position {
                StreamPosition::Beginning => 0,
                StreamPosition::End => topic_data.next_offset,
                StreamPosition::Offset(offset) => offset,
            }
        };

        let mut events = Vec::new();
        for (event, offset) in &topic_data.events {
            if offset.value() >= start_offset && events.len() < max_events {
                events.push((event.clone(), *offset));
            }
        }

        // Update consumer offset if using consumer group
        if let Some(group) = consumer_group {
            if let Some((_, last_offset)) = events.last() {
                topic_data.consumer_offsets.insert(
                    group.name().to_string(),
                    last_offset.value() + 1,
                );
            }
        }

        Ok(events)
    }

    async fn commit_offset(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        _partition: PartitionId,
        offset: Offset,
    ) -> StreamResult<()> {
        let mut topics = self.storage.topics.write().await;
        let topic_data = topics
            .get_mut(topic)
            .ok_or_else(|| StreamError::TopicNotFound(topic.to_string()))?;

        topic_data.consumer_offsets.insert(
            consumer_group.name().to_string(),
            offset.value() + 1,
        );
        Ok(())
    }

    async fn seek(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
        _partition: PartitionId,
        position: StreamPosition,
    ) -> StreamResult<()> {
        let mut topics = self.storage.topics.write().await;
        let topic_data = topics
            .get_mut(topic)
            .ok_or_else(|| StreamError::TopicNotFound(topic.to_string()))?;

        let offset = match position {
            StreamPosition::Beginning => 0,
            StreamPosition::End => topic_data.next_offset,
            StreamPosition::Offset(offset) => offset,
        };

        topic_data.consumer_offsets.insert(
            consumer_group.name().to_string(),
            offset,
        );
        Ok(())
    }

    async fn get_consumer_lag(
        &self,
        topic: &TopicName,
        consumer_group: &ConsumerGroup,
    ) -> StreamResult<HashMap<PartitionId, u64>> {
        let topics = self.storage.topics.read().await;
        let topic_data = topics
            .get(topic)
            .ok_or_else(|| StreamError::TopicNotFound(topic.to_string()))?;

        let current_offset = topic_data.consumer_offsets
            .get(consumer_group.name())
            .copied()
            .unwrap_or(0);

        let lag = topic_data.next_offset.saturating_sub(current_offset);
        let mut result = HashMap::new();
        result.insert(PartitionId::new(0), lag);
        Ok(result)
    }

    async fn get_topic_metadata(&self, topic: &TopicName) -> StreamResult<HashMap<String, String>> {
        let topics = self.storage.topics.read().await;
        let topic_data = topics
            .get(topic)
            .ok_or_else(|| StreamError::TopicNotFound(topic.to_string()))?;

        let mut metadata = HashMap::new();
        metadata.insert("backend".to_string(), "memory".to_string());
        metadata.insert("event_count".to_string(), topic_data.events.len().to_string());
        metadata.insert("next_offset".to_string(), topic_data.next_offset.to_string());
        metadata.insert("consumer_groups".to_string(), topic_data.consumer_offsets.len().to_string());
        
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{StreamEvent, StreamEventType};

    #[tokio::test]
    async fn test_memory_backend_basic_operations() {
        let mut backend = MemoryBackend::new();
        assert_eq!(backend.name(), "memory");

        // Connect
        backend.connect().await.unwrap();

        // Create topic
        let topic = TopicName::new("test-topic".to_string());
        backend.create_topic(&topic, 1).await.unwrap();

        // List topics
        let topics = backend.list_topics().await.unwrap();
        assert_eq!(topics.len(), 1);
        assert_eq!(topics[0].as_str(), "test-topic");

        // Send event
        let event = StreamEvent::new(StreamEventType::TripleAdded {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
            graph: None,
        });

        let offset = backend.send_event(&topic, event.clone()).await.unwrap();
        assert_eq!(offset.value(), 0);

        // Receive events
        let events = backend
            .receive_events(&topic, None, StreamPosition::Beginning, 10)
            .await
            .unwrap();
        assert_eq!(events.len(), 1);

        // Delete topic
        backend.delete_topic(&topic).await.unwrap();
        let topics = backend.list_topics().await.unwrap();
        assert_eq!(topics.len(), 0);
    }

    #[tokio::test]
    async fn test_consumer_groups() {
        let mut backend = MemoryBackend::new();
        backend.connect().await.unwrap();

        let topic = TopicName::new("test-topic".to_string());
        backend.create_topic(&topic, 1).await.unwrap();

        // Send some events
        for i in 0..5 {
            let event = StreamEvent::new(StreamEventType::GraphCreated {
                graph: format!("http://example.org/graph{}", i),
            });
            backend.send_event(&topic, event).await.unwrap();
        }

        // Create consumer group
        let group = ConsumerGroup::new("test-group".to_string());

        // First read - should get all events
        let events = backend
            .receive_events(&topic, Some(&group), StreamPosition::Beginning, 3)
            .await
            .unwrap();
        assert_eq!(events.len(), 3);

        // Second read - should get remaining events
        let events = backend
            .receive_events(&topic, Some(&group), StreamPosition::Beginning, 10)
            .await
            .unwrap();
        assert_eq!(events.len(), 2);

        // Check consumer lag
        let lag = backend.get_consumer_lag(&topic, &group).await.unwrap();
        assert_eq!(lag.get(&PartitionId::new(0)), Some(&0));
    }
}