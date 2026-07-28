//! Apache Kafka integration for event streaming.
//!
//! # Fail-loud contract
//!
//! A production-grade Kafka client requires the full Kafka binary wire protocol
//! (ApiVersions/Metadata/Produce request framing, record-batch encoding, CRC32C,
//! compression). The only mature Rust Kafka clients (`rdkafka`) are C-FFI (librdkafka)
//! bindings, which the COOLJAPAN Pure-Rust policy forbids in default features, and
//! a from-scratch pure-Rust implementation is out of scope for this crate.
//!
//! Rather than *silently dropping* every event an operator publishes (the previous
//! behaviour, which logged "Would send to Kafka" and returned `Ok(())`), this module
//! now **fails loudly**: constructing a Kafka producer or consumer returns an explicit
//! error, so a server configured with `streaming.kafka` refuses to start instead of
//! pretending to deliver events to a message bus that never receives them.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    error::{FusekiError, FusekiResult},
    streaming::{RDFEvent, StreamConsumer, StreamProducer},
};

/// Error message shared by every Kafka construction path.
const KAFKA_UNSUPPORTED: &str = "Kafka streaming is configured but not supported in this build: \
no Pure-Rust Kafka backend is available (rdkafka is C-FFI and excluded by policy). \
Remove `streaming.kafka` from the configuration or use the NATS backend instead.";

/// Additional detail appended to [`KAFKA_UNSUPPORTED`] when
/// `enable_transactions` was requested: transactional Kafka semantics
/// (`init_transactions`/`begin_transaction`/`commit_transaction`) require
/// the same missing wire-protocol client, so this is called out explicitly
/// rather than silently downgrading to non-transactional (or, as before
/// this fix, simply not existing at all in the error message).
const KAFKA_TRANSACTIONS_UNSUPPORTED: &str = " `enable_transactions` was also requested: \
exactly-once/transactional producer semantics additionally require \
init_transactions/begin_transaction/commit_transaction support, which this build cannot \
provide either.";

/// Kafka-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KafkaConfig {
    /// Kafka broker addresses
    pub brokers: Vec<String>,
    /// Security protocol
    pub security_protocol: Option<String>,
    /// SASL mechanism
    pub sasl_mechanism: Option<String>,
    /// SASL username
    pub sasl_username: Option<String>,
    /// SASL password
    pub sasl_password: Option<String>,
    /// Additional Kafka properties
    pub properties: HashMap<String, String>,
    /// Whether the caller requested transactional producer semantics
    /// (`streaming.kafka.enable_transactions`). Carried through purely so
    /// the fail-loud error below can call it out explicitly; this build
    /// cannot honour it either way (see module docs).
    pub enable_transactions: bool,
}

impl Default for KafkaConfig {
    fn default() -> Self {
        Self {
            brokers: vec!["localhost:9092".to_string()],
            security_protocol: None,
            sasl_mechanism: None,
            sasl_username: None,
            sasl_password: None,
            properties: HashMap::new(),
            enable_transactions: false,
        }
    }
}

impl From<crate::streaming::KafkaConfig> for KafkaConfig {
    fn from(config: crate::streaming::KafkaConfig) -> Self {
        Self {
            brokers: config.brokers,
            security_protocol: None,
            sasl_mechanism: None,
            sasl_username: None,
            sasl_password: None,
            properties: HashMap::new(),
            enable_transactions: config.enable_transactions,
        }
    }
}

/// Kafka producer implementation.
#[derive(Debug)]
pub struct KafkaProducer {
    #[allow(dead_code)]
    config: KafkaConfig,
}

impl KafkaProducer {
    /// Create a new Kafka producer.
    ///
    /// Fails loudly: there is no Pure-Rust Kafka backend available, so rather than
    /// silently discard published events this refuses construction with an explicit
    /// error (which propagates out of `StreamingManager::initialize` and prevents
    /// the server from starting with a dead streaming sink). When
    /// `config.enable_transactions` is set the error additionally calls out that
    /// transactional semantics are unsupported, rather than silently ignoring the
    /// flag.
    pub async fn new(config: KafkaConfig) -> FusekiResult<Self> {
        let mut message = KAFKA_UNSUPPORTED.to_string();
        if config.enable_transactions {
            message.push_str(KAFKA_TRANSACTIONS_UNSUPPORTED);
        }
        Err(FusekiError::service_unavailable(message))
    }
}

#[async_trait]
impl StreamProducer for KafkaProducer {
    async fn send(&self, _event: RDFEvent) -> FusekiResult<()> {
        Err(FusekiError::service_unavailable(KAFKA_UNSUPPORTED))
    }

    async fn send_batch(&self, _events: Vec<RDFEvent>) -> FusekiResult<()> {
        Err(FusekiError::service_unavailable(KAFKA_UNSUPPORTED))
    }

    async fn flush(&self) -> FusekiResult<()> {
        Err(FusekiError::service_unavailable(KAFKA_UNSUPPORTED))
    }
}

/// Kafka consumer implementation.
pub struct KafkaConsumer {
    #[allow(dead_code)]
    config: KafkaConfig,
}

impl KafkaConsumer {
    /// Create a new Kafka consumer. Fails loudly (see [`KafkaProducer::new`]).
    pub async fn new(_config: KafkaConfig) -> FusekiResult<Self> {
        Err(FusekiError::service_unavailable(KAFKA_UNSUPPORTED))
    }
}

#[async_trait]
impl StreamConsumer for KafkaConsumer {
    async fn subscribe(
        &self,
        _handler: Box<dyn crate::streaming::EventHandler>,
    ) -> FusekiResult<()> {
        Err(FusekiError::service_unavailable(KAFKA_UNSUPPORTED))
    }

    async fn unsubscribe(&self) -> FusekiResult<()> {
        Err(FusekiError::service_unavailable(KAFKA_UNSUPPORTED))
    }

    async fn commit(&self) -> FusekiResult<()> {
        Err(FusekiError::service_unavailable(KAFKA_UNSUPPORTED))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn regression_kafka_producer_fails_loud() {
        // Configuring Kafka must produce an explicit error, never a silent no-op.
        assert!(KafkaProducer::new(KafkaConfig::default()).await.is_err());
        assert!(KafkaConsumer::new(KafkaConfig::default()).await.is_err());
    }

    /// Regression test for the `KafkaConfig.enable_transactions` finding:
    /// the flag must be threaded from `crate::streaming::KafkaConfig`
    /// through to the Kafka-crate `KafkaConfig`, and the resulting
    /// fail-loud error message must call out that transactional semantics
    /// were requested, instead of the flag being silently dropped.
    #[tokio::test]
    async fn regression_enable_transactions_reflected_in_fail_loud_error() {
        let config = KafkaConfig {
            enable_transactions: true,
            ..KafkaConfig::default()
        };
        let err = KafkaProducer::new(config)
            .await
            .expect_err("Kafka producer construction must fail loud");
        let message = err.to_string();
        assert!(
            message.contains("enable_transactions"),
            "error message should call out enable_transactions, got: {message}"
        );

        let config_no_tx = KafkaConfig::default();
        let err_no_tx = KafkaProducer::new(config_no_tx)
            .await
            .expect_err("Kafka producer construction must fail loud");
        assert!(
            !err_no_tx.to_string().contains("enable_transactions"),
            "error message should not mention enable_transactions when it was not requested"
        );
    }

    /// The `enable_transactions` flag set on the streaming-level
    /// `crate::streaming::KafkaConfig` must survive the `From` conversion
    /// into the Kafka-crate `KafkaConfig`, rather than being dropped like
    /// the other fields this conversion never carried (topic_prefix,
    /// producer/consumer tuning).
    #[test]
    fn regression_enable_transactions_survives_from_conversion() {
        let streaming_config = crate::streaming::KafkaConfig {
            brokers: vec!["localhost:9092".to_string()],
            topic_prefix: "rdf".to_string(),
            producer: crate::streaming::ProducerConfig::default(),
            consumer: crate::streaming::ConsumerConfig::default(),
            enable_transactions: true,
        };
        let kafka_crate_config: KafkaConfig = streaming_config.into();
        assert!(kafka_crate_config.enable_transactions);
    }
}
