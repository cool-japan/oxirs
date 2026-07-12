//! Supporting types for the Kafka backend.
//!
//! Defines consumer identifiers, the per-consumer instance bookkeeping
//! structure, metrics, partition-assignment information, and the rdkafka
//! rebalance callback context.

use anyhow::Result;
use oxirs_stream::StreamEvent;
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, AtomicU64},
    Arc,
};
use tokio::sync::{oneshot, RwLock};
use uuid::Uuid;

#[cfg(feature = "kafka")]
use rdkafka::{
    consumer::{BaseConsumer, ConsumerContext, Rebalance, StreamConsumer},
    ClientContext,
};
#[cfg(feature = "kafka")]
use tracing::{debug, error, info};

/// Consumer identifier type
pub type ConsumerId = Uuid;

/// Message callback type for streaming consumers
pub type MessageCallback = Arc<dyn Fn(StreamEvent) -> Result<()> + Send + Sync>;

/// Consumer instance management
pub(crate) struct ConsumerInstance {
    pub(crate) id: ConsumerId,
    pub(crate) group_id: String,
    #[cfg(feature = "kafka")]
    pub(crate) consumer: Arc<StreamConsumer>,
    pub(crate) is_active: Arc<AtomicBool>,
    pub(crate) message_count: Arc<AtomicU64>,
    pub(crate) error_count: Arc<AtomicU64>,
    pub(crate) last_message_time: Arc<RwLock<Option<chrono::DateTime<chrono::Utc>>>>,
    pub(crate) stop_signal: Option<oneshot::Sender<()>>,
}

/// Consumer metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerMetrics {
    pub consumer_id: ConsumerId,
    pub group_id: String,
    pub messages_processed: u64,
    pub errors_encountered: u64,
    pub is_active: bool,
    pub last_message_time: Option<chrono::DateTime<chrono::Utc>>,
    pub partition_assignments: Vec<PartitionAssignment>,
}

/// Partition assignment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionAssignment {
    pub topic: String,
    pub partition: i32,
    pub current_offset: i64,
    pub high_water_mark: i64,
    pub lag: i64,
}

/// Consumer rebalance callback context
#[cfg(feature = "kafka")]
pub(crate) struct ConsumerRebalanceContext {
    pub(crate) consumer_id: ConsumerId,
}

#[cfg(feature = "kafka")]
impl ClientContext for ConsumerRebalanceContext {}

#[cfg(feature = "kafka")]
impl ConsumerContext for ConsumerRebalanceContext {
    fn pre_rebalance(&self, _consumer: &BaseConsumer<Self>, rebalance: &Rebalance) {
        info!(
            "Consumer {} pre-rebalance: {:?}",
            self.consumer_id, rebalance
        );
    }

    fn post_rebalance(&self, _consumer: &BaseConsumer<Self>, rebalance: &Rebalance) {
        info!(
            "Consumer {} post-rebalance: {:?}",
            self.consumer_id, rebalance
        );
    }

    fn commit_callback(
        &self,
        result: rdkafka::error::KafkaResult<()>,
        offsets: &rdkafka::TopicPartitionList,
    ) {
        match result {
            Ok(_) => debug!(
                "Consumer {} committed offsets: {:?}",
                self.consumer_id, offsets
            ),
            Err(e) => error!("Consumer {} commit failed: {}", self.consumer_id, e),
        }
    }
}
