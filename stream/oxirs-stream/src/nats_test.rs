//! Tests for NATS integration

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::StreamConfig;

    #[test]
    fn test_nats_config_creation() {
        let config = NatsConfig::default();
        assert_eq!(config.url, "nats://localhost:4222");
        assert_eq!(config.stream_name, "OXIRS_RDF");
        assert_eq!(config.subject_prefix, "oxirs.rdf");
    }

    #[test]
    fn test_consumer_creation() {
        let stream_config = StreamConfig::default();
        let consumer = NatsConsumer::new(stream_config);
        assert!(consumer.is_ok());
    }

    #[test]
    fn test_producer_creation() {
        let stream_config = StreamConfig::default();
        let producer = NatsProducer::new(stream_config);
        assert!(producer.is_ok());
    }
}