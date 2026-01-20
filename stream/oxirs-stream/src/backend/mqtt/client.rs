//! MQTT Client Implementation
//!
//! Async MQTT client using rumqttc for high-performance message streaming

use super::payload_parser::PayloadParser;
use super::topic_mapping::TopicMapper;
use super::types::{MqttConfig, MqttMessage, MqttStats, QoS, TopicSubscription};
use crate::error::{StreamError, StreamResult};
use crate::event::StreamEvent;
use chrono::Utc;
use rumqttc::{AsyncClient, Event, EventLoop, MqttOptions, Packet, QoS as RumqttQoS};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, warn};

/// MQTT Client for streaming
pub struct MqttClient {
    config: MqttConfig,
    client: Arc<RwLock<Option<AsyncClient>>>,
    topic_mapper: Arc<TopicMapper>,
    payload_parser: Arc<PayloadParser>,
    stats: Arc<RwLock<MqttStats>>,
    event_sender: broadcast::Sender<StreamEvent>,
    subscriptions: Arc<RwLock<Vec<TopicSubscription>>>,
    connected: Arc<RwLock<bool>>,
}

impl MqttClient {
    /// Create a new MQTT client
    pub fn new(config: MqttConfig) -> Self {
        let (tx, _) = broadcast::channel(10000);

        Self {
            config,
            client: Arc::new(RwLock::new(None)),
            topic_mapper: Arc::new(TopicMapper::new()),
            payload_parser: Arc::new(PayloadParser::new()),
            stats: Arc::new(RwLock::new(MqttStats::default())),
            event_sender: tx,
            subscriptions: Arc::new(RwLock::new(Vec::new())),
            connected: Arc::new(RwLock::new(false)),
        }
    }

    /// Connect to MQTT broker and return event loop
    pub async fn connect(&mut self) -> StreamResult<EventLoop> {
        let mqtt_options = self.build_mqtt_options()?;

        let (client, event_loop) = AsyncClient::new(mqtt_options, 100);

        *self.client.write().await = Some(client);

        // Mark as connected
        *self.connected.write().await = true;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.connection_count += 1;
            stats.last_connected_at = Some(Utc::now());
        }

        info!("MQTT client connected to {}", self.config.broker_url);

        Ok(event_loop)
    }

    /// Disconnect from MQTT broker
    pub async fn disconnect(&mut self) -> StreamResult<()> {
        if let Some(client) = self.client.read().await.as_ref() {
            client
                .disconnect()
                .await
                .map_err(|e| StreamError::Connection(format!("MQTT disconnect failed: {}", e)))?;
        }

        *self.connected.write().await = false;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.last_disconnected_at = Some(Utc::now());
        }

        info!("MQTT client disconnected");

        Ok(())
    }

    /// Subscribe to topics with RDF mappings
    pub async fn subscribe(&self, subscriptions: Vec<TopicSubscription>) -> StreamResult<()> {
        let client_guard = self.client.read().await;
        let client = client_guard
            .as_ref()
            .ok_or_else(|| StreamError::NotConnected("MQTT client not connected".to_string()))?;

        for sub in &subscriptions {
            let qos = Self::convert_qos(sub.qos);

            client
                .subscribe(&sub.topic_pattern, qos)
                .await
                .map_err(|e| {
                    StreamError::Backend(format!(
                        "Failed to subscribe to {}: {}",
                        sub.topic_pattern, e
                    ))
                })?;

            debug!(
                "Subscribed to topic: {} (QoS: {:?})",
                sub.topic_pattern, sub.qos
            );
        }

        // Store subscriptions
        *self.subscriptions.write().await = subscriptions;

        Ok(())
    }

    /// Start the message processing loop
    pub async fn start_message_loop(
        self: Arc<Self>,
        mut event_loop: EventLoop,
    ) -> StreamResult<()> {
        tokio::spawn(async move {
            loop {
                match event_loop.poll().await {
                    Ok(notification) => {
                        if let Err(e) = self.handle_notification(notification).await {
                            error!("Error handling MQTT notification: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("MQTT event loop error: {}", e);

                        // Update connected status
                        *self.connected.write().await = false;

                        // Try to reconnect
                        {
                            let mut stats = self.stats.write().await;
                            stats.reconnection_count += 1;
                        }

                        // Wait before reconnecting
                        tokio::time::sleep(std::time::Duration::from_millis(
                            self.config.reconnect.initial_delay_ms,
                        ))
                        .await;
                    }
                }
            }
        });

        Ok(())
    }

    /// Handle incoming MQTT notification
    async fn handle_notification(&self, event: Event) -> StreamResult<()> {
        match event {
            Event::Incoming(Packet::Publish(publish)) => {
                let message = MqttMessage {
                    topic: publish.topic.clone(),
                    qos: Self::convert_qos_from_rumqtt(publish.qos),
                    retain: publish.retain,
                    payload: publish.payload.to_vec(),
                    properties: None, // TODO: Extract MQTT 5.0 properties
                    received_at: Utc::now(),
                };

                self.process_message(message).await?;
            }
            Event::Incoming(Packet::ConnAck(_)) => {
                info!("MQTT connection acknowledged");
                *self.connected.write().await = true;
            }
            Event::Incoming(Packet::SubAck(_)) => {
                debug!("MQTT subscription acknowledged");
            }
            Event::Incoming(Packet::PingResp) => {
                debug!("MQTT ping response received");
            }
            Event::Incoming(Packet::Disconnect) => {
                warn!("MQTT disconnect received");
                *self.connected.write().await = false;
            }
            Event::Outgoing(_) => {
                // Outgoing packets (publish, subscribe, etc.)
            }
            _ => {
                debug!("MQTT event: {:?}", event);
            }
        }

        Ok(())
    }

    /// Process an MQTT message and convert to StreamEvents
    async fn process_message(&self, message: MqttMessage) -> StreamResult<()> {
        // Find matching subscription
        let subscriptions = self.subscriptions.read().await;
        let matching_sub = subscriptions
            .iter()
            .find(|sub| {
                self.topic_mapper
                    .matches(&message.topic, &sub.topic_pattern)
            })
            .ok_or_else(|| {
                StreamError::Backend(format!(
                    "No subscription found for topic: {}",
                    message.topic
                ))
            })?;

        // Parse payload
        let parsed = self
            .payload_parser
            .parse(&message.payload, &matching_sub.payload_format)?;

        // Convert to StreamEvents using RDF mapping
        let events = self.topic_mapper.to_stream_events(
            &message.topic,
            &parsed,
            &matching_sub.rdf_mapping,
        )?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.messages_received += 1;
            stats.bytes_received += message.payload.len() as u64;

            match message.qos {
                QoS::AtMostOnce => stats.qos0_count += 1,
                QoS::AtLeastOnce => stats.qos1_count += 1,
                QoS::ExactlyOnce => stats.qos2_count += 1,
            }
        }

        // Send events to subscribers
        for event in events {
            if let Err(e) = self.event_sender.send(event) {
                warn!("Failed to send event: {}", e);
            }
        }

        Ok(())
    }

    /// Publish message to MQTT broker
    pub async fn publish(
        &self,
        topic: &str,
        payload: Vec<u8>,
        qos: QoS,
        retain: bool,
    ) -> StreamResult<()> {
        let client_guard = self.client.read().await;
        let client = client_guard
            .as_ref()
            .ok_or_else(|| StreamError::NotConnected("MQTT client not connected".to_string()))?;

        let rumqtt_qos = Self::convert_qos(qos);

        client
            .publish(topic, rumqtt_qos, retain, payload.clone())
            .await
            .map_err(|e| StreamError::Send(format!("MQTT publish failed: {}", e)))?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.messages_published += 1;
            stats.bytes_sent += payload.len() as u64;

            match qos {
                QoS::AtMostOnce => stats.qos0_count += 1,
                QoS::AtLeastOnce => stats.qos1_count += 1,
                QoS::ExactlyOnce => stats.qos2_count += 1,
            }
        }

        Ok(())
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> MqttStats {
        self.stats.read().await.clone()
    }

    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }

    /// Subscribe to event stream
    pub fn subscribe_events(&self) -> broadcast::Receiver<StreamEvent> {
        self.event_sender.subscribe()
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Build MQTT options from config
    fn build_mqtt_options(&self) -> StreamResult<MqttOptions> {
        let broker_url = &self.config.broker_url;

        // Parse broker URL (tcp://host:port or ssl://host:port)
        let (scheme, host_port) = broker_url.split_once("://").ok_or_else(|| {
            StreamError::Configuration(format!("Invalid broker URL: {}", broker_url))
        })?;

        let (host, port) = if let Some((h, p)) = host_port.split_once(':') {
            let port = p
                .parse::<u16>()
                .map_err(|e| StreamError::Configuration(format!("Invalid port: {}", e)))?;
            (h.to_string(), port)
        } else {
            // Default ports
            let port = match scheme {
                "tcp" | "mqtt" => 1883,
                "ssl" | "mqtts" => 8883,
                _ => {
                    return Err(StreamError::Configuration(format!(
                        "Unknown scheme: {}",
                        scheme
                    )))
                }
            };
            (host_port.to_string(), port)
        };

        let mut mqtt_options = MqttOptions::new(&self.config.client_id, host, port);

        // Set keep alive
        mqtt_options.set_keep_alive(std::time::Duration::from_secs(
            self.config.keep_alive_secs as u64,
        ));

        // Set clean session
        mqtt_options.set_clean_session(self.config.clean_session);

        // Set credentials
        if let (Some(ref username), Some(ref password)) =
            (&self.config.username, &self.config.password)
        {
            mqtt_options.set_credentials(username, password);
        }

        // Set last will
        if let Some(ref lw) = self.config.last_will {
            mqtt_options.set_last_will(rumqttc::LastWill {
                topic: lw.topic.clone(),
                message: lw.payload.clone().into(),
                qos: Self::convert_qos(lw.qos),
                retain: lw.retain,
            });
        }

        Ok(mqtt_options)
    }

    /// Convert QoS to rumqttc QoS
    fn convert_qos(qos: QoS) -> RumqttQoS {
        match qos {
            QoS::AtMostOnce => RumqttQoS::AtMostOnce,
            QoS::AtLeastOnce => RumqttQoS::AtLeastOnce,
            QoS::ExactlyOnce => RumqttQoS::ExactlyOnce,
        }
    }

    /// Convert from rumqttc QoS
    fn convert_qos_from_rumqtt(qos: RumqttQoS) -> QoS {
        match qos {
            RumqttQoS::AtMostOnce => QoS::AtMostOnce,
            RumqttQoS::AtLeastOnce => QoS::AtLeastOnce,
            RumqttQoS::ExactlyOnce => QoS::ExactlyOnce,
        }
    }
}

/// MQTT StreamBackend implementation
pub struct MqttBackend {
    client: Arc<MqttClient>,
}

impl MqttBackend {
    /// Create a new MQTT backend
    pub fn new(config: MqttConfig) -> Self {
        Self {
            client: Arc::new(MqttClient::new(config)),
        }
    }

    /// Get the underlying client
    pub fn client(&self) -> Arc<MqttClient> {
        Arc::clone(&self.client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mqtt_client_creation() {
        let config = MqttConfig::default();
        let _client = MqttClient::new(config);
        // Client created successfully
    }

    #[test]
    fn test_qos_conversion() {
        assert_eq!(
            MqttClient::convert_qos(QoS::AtMostOnce),
            RumqttQoS::AtMostOnce
        );
        assert_eq!(
            MqttClient::convert_qos(QoS::AtLeastOnce),
            RumqttQoS::AtLeastOnce
        );
        assert_eq!(
            MqttClient::convert_qos(QoS::ExactlyOnce),
            RumqttQoS::ExactlyOnce
        );

        assert_eq!(
            MqttClient::convert_qos_from_rumqtt(RumqttQoS::AtMostOnce),
            QoS::AtMostOnce
        );
    }

    #[test]
    fn test_build_mqtt_options() {
        let config = MqttConfig {
            broker_url: "tcp://localhost:1883".to_string(),
            client_id: "test-client".to_string(),
            ..Default::default()
        };

        let client = MqttClient::new(config);
        let _options = client.build_mqtt_options().unwrap();

        // Can't directly test private fields, but we verified it doesn't error
    }
}
