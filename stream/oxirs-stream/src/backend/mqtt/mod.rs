//! MQTT Backend for OxiRS Stream
//!
//! Provides MQTT 3.1.1 and 5.0 protocol support for IoT and Industry 4.0 integration.
//! Compatible with:
//! - Standard MQTT brokers (Mosquitto, EMQX, HiveMQ)
//! - Industrial MQTT (Sparkplug B)
//! - Azure IoT Hub, AWS IoT Core, Google Cloud IoT

pub mod client;
pub mod payload_parser;
#[cfg(feature = "sparkplug")]
pub mod sparkplug_b;
pub mod topic_mapping;
pub mod types;

pub use client::{MqttBackend, MqttClient};
pub use payload_parser::{ParsedPayload, PayloadParser};
pub use topic_mapping::{TopicMapper, TopicMatch};
pub use types::{
    MqttConfig, MqttMessage, MqttProtocolVersion, MqttStats, PayloadFormat, QoS, TopicRdfMapping,
    TopicSubscription,
};
