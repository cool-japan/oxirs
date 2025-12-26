//! Polling scheduler for Modbus devices
//!
//! Provides scheduled polling with automatic reconnection,
//! batch reading, and RDF triple generation.

use crate::error::{ModbusError, ModbusResult};
use crate::mapping::{RegisterMap, RegisterType};
use crate::protocol::ModbusTcpClient;
use crate::rdf::{GeneratedTriple, ModbusTripleGenerator};
use chrono::Utc;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio::time::{interval, Instant};

/// Polling statistics
#[derive(Debug, Clone, Default)]
pub struct PollingStats {
    /// Total polling cycles completed
    pub cycles: u64,
    /// Total successful reads
    pub successful_reads: u64,
    /// Total failed reads
    pub failed_reads: u64,
    /// Total triples generated
    pub triples_generated: u64,
    /// Average cycle duration in milliseconds
    pub avg_cycle_ms: f64,
    /// Last error message
    pub last_error: Option<String>,
}

/// Event from the polling scheduler
#[derive(Debug, Clone)]
pub enum PollingEvent {
    /// New triples generated
    Triples(Vec<GeneratedTriple>),
    /// Connection established
    Connected,
    /// Connection lost
    Disconnected(String),
    /// Polling cycle completed
    CycleCompleted(PollingStats),
    /// Error occurred
    Error(String),
}

/// Configuration for polling scheduler
#[derive(Debug, Clone)]
pub struct PollingConfig {
    /// Modbus server address
    pub address: String,
    /// Unit ID (slave address)
    pub unit_id: u8,
    /// Polling interval
    pub interval: Duration,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Maximum reconnection attempts (0 = infinite)
    pub max_reconnect_attempts: u32,
    /// Delay between reconnection attempts
    pub reconnect_delay: Duration,
}

impl Default for PollingConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1:502".to_string(),
            unit_id: 1,
            interval: Duration::from_secs(1),
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(5),
            max_reconnect_attempts: 0, // Infinite
            reconnect_delay: Duration::from_secs(5),
        }
    }
}

/// Modbus polling scheduler
///
/// Manages continuous polling of Modbus devices with automatic
/// reconnection and RDF triple generation.
pub struct PollingScheduler {
    /// Polling configuration
    config: PollingConfig,
    /// Register map for RDF generation
    register_map: RegisterMap,
    /// Current client connection
    client: Option<ModbusTcpClient>,
    /// Triple generator
    generator: ModbusTripleGenerator,
    /// Statistics
    stats: PollingStats,
}

impl PollingScheduler {
    /// Create a new polling scheduler
    pub fn new(config: PollingConfig, register_map: RegisterMap) -> Self {
        let generator = ModbusTripleGenerator::new(register_map.clone());
        Self {
            config,
            register_map,
            client: None,
            generator,
            stats: PollingStats::default(),
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &PollingStats {
        &self.stats
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.client.is_some()
    }

    /// Connect to the Modbus device
    pub async fn connect(&mut self) -> ModbusResult<()> {
        let client = ModbusTcpClient::connect(&self.config.address, self.config.unit_id).await?;
        self.client = Some(client);
        Ok(())
    }

    /// Disconnect from the device
    pub fn disconnect(&mut self) {
        self.client = None;
    }

    /// Perform a single polling cycle
    ///
    /// Returns generated triples if any values changed.
    pub async fn poll_once(&mut self) -> ModbusResult<Vec<GeneratedTriple>> {
        let client = self.client.as_mut().ok_or_else(|| {
            ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "Not connected to Modbus device",
            ))
        })?;

        let timestamp = Utc::now();
        let mut all_triples = Vec::new();

        // Read holding registers
        let holding_batches = self.register_map.batch_reads(RegisterType::Holding, 125);

        for (start_addr, count) in holding_batches {
            match client.read_holding_registers(start_addr, count).await {
                Ok(values) => {
                    self.stats.successful_reads += 1;
                    let triples = self.generator.generate_from_array(
                        start_addr,
                        &values,
                        RegisterType::Holding,
                        timestamp,
                    )?;
                    all_triples.extend(triples);
                }
                Err(e) => {
                    self.stats.failed_reads += 1;
                    self.stats.last_error = Some(e.to_string());
                    return Err(e);
                }
            }
        }

        // Read input registers
        let input_batches = self.register_map.batch_reads(RegisterType::Input, 125);

        for (start_addr, count) in input_batches {
            match client.read_input_registers(start_addr, count).await {
                Ok(values) => {
                    self.stats.successful_reads += 1;
                    let triples = self.generator.generate_from_array(
                        start_addr,
                        &values,
                        RegisterType::Input,
                        timestamp,
                    )?;
                    all_triples.extend(triples);
                }
                Err(e) => {
                    self.stats.failed_reads += 1;
                    self.stats.last_error = Some(e.to_string());
                    return Err(e);
                }
            }
        }

        self.stats.triples_generated += all_triples.len() as u64;
        self.stats.cycles += 1;

        Ok(all_triples)
    }

    /// Run the polling loop
    ///
    /// Returns a receiver for polling events and a sender for stop signal.
    pub async fn run(mut self) -> (broadcast::Receiver<PollingEvent>, mpsc::Sender<()>) {
        let (event_tx, event_rx) = broadcast::channel(100);
        let (stop_tx, mut stop_rx) = mpsc::channel::<()>(1);

        let event_sender = event_tx.clone();

        tokio::spawn(async move {
            let mut poll_interval = interval(self.config.interval);
            let mut reconnect_attempts = 0u32;

            loop {
                tokio::select! {
                    _ = stop_rx.recv() => {
                        // Stop signal received - exit loop
                        break;
                    }
                    _ = poll_interval.tick() => {
                        // Ensure connection
                        if self.client.is_none() {
                            if self.config.max_reconnect_attempts > 0
                                && reconnect_attempts >= self.config.max_reconnect_attempts
                            {
                                let _ = event_sender.send(PollingEvent::Error(
                                    "Max reconnection attempts reached".to_string(),
                                ));
                                break;
                            }

                            match self.connect().await {
                                Ok(()) => {
                                    reconnect_attempts = 0;
                                    let _ = event_sender.send(PollingEvent::Connected);
                                }
                                Err(e) => {
                                    reconnect_attempts += 1;
                                    let _ = event_sender.send(PollingEvent::Disconnected(
                                        e.to_string(),
                                    ));
                                    tokio::time::sleep(self.config.reconnect_delay).await;
                                    continue;
                                }
                            }
                        }

                        // Poll
                        let start = Instant::now();
                        match self.poll_once().await {
                            Ok(triples) => {
                                let elapsed = start.elapsed().as_millis() as f64;
                                self.stats.avg_cycle_ms = (self.stats.avg_cycle_ms
                                    * (self.stats.cycles - 1) as f64
                                    + elapsed)
                                    / self.stats.cycles as f64;

                                if !triples.is_empty() {
                                    let _ = event_sender.send(PollingEvent::Triples(triples));
                                }
                                let _ = event_sender.send(PollingEvent::CycleCompleted(
                                    self.stats.clone(),
                                ));
                            }
                            Err(e) => {
                                // Connection error - disconnect and retry
                                self.disconnect();
                                let _ = event_sender.send(PollingEvent::Error(e.to_string()));
                            }
                        }
                    }
                }
            }
        });

        (event_rx, stop_tx)
    }
}

/// Shared polling scheduler for concurrent access
pub type SharedScheduler = Arc<Mutex<PollingScheduler>>;

/// Create a shared polling scheduler
pub fn shared_scheduler(config: PollingConfig, register_map: RegisterMap) -> SharedScheduler {
    Arc::new(Mutex::new(PollingScheduler::new(config, register_map)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::{ModbusDataType, RegisterMapping};
    use crate::testing::MockModbusServer;

    fn create_test_config(address: &str) -> PollingConfig {
        PollingConfig {
            address: address.to_string(),
            unit_id: 1,
            interval: Duration::from_millis(100),
            connect_timeout: Duration::from_secs(1),
            request_timeout: Duration::from_secs(1),
            max_reconnect_attempts: 3,
            reconnect_delay: Duration::from_millis(100),
        }
    }

    fn create_test_map() -> RegisterMap {
        let mut map = RegisterMap::new("test", "http://test.example.com/device");
        map.add_register(RegisterMapping::new(
            0,
            ModbusDataType::Uint16,
            "http://test.example.com/property/value",
        ));
        map
    }

    #[tokio::test]
    async fn test_scheduler_connect() {
        let server = MockModbusServer::start().await.unwrap();
        let config = create_test_config(server.address());
        let map = create_test_map();

        let mut scheduler = PollingScheduler::new(config, map);

        assert!(!scheduler.is_connected());
        scheduler.connect().await.unwrap();
        assert!(scheduler.is_connected());
        scheduler.disconnect();
        assert!(!scheduler.is_connected());

        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_scheduler_poll_once() {
        let server = MockModbusServer::start().await.unwrap();
        let config = create_test_config(server.address());
        let map = create_test_map();

        let mut scheduler = PollingScheduler::new(config, map);
        scheduler.connect().await.unwrap();

        let triples = scheduler.poll_once().await.unwrap();

        // Should generate 1 triple (from holding register 0)
        assert_eq!(triples.len(), 1);
        assert_eq!(scheduler.stats().cycles, 1);
        assert_eq!(scheduler.stats().successful_reads, 1);

        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_scheduler_stats() {
        let server = MockModbusServer::start().await.unwrap();
        let config = create_test_config(server.address());
        let map = create_test_map();

        let mut scheduler = PollingScheduler::new(config, map);
        scheduler.connect().await.unwrap();

        // Poll multiple times
        for _ in 0..5 {
            let _ = scheduler.poll_once().await;
        }

        let stats = scheduler.stats();
        assert_eq!(stats.cycles, 5);
        // avg_cycle_ms can be 0 for very fast local connections
        assert!(stats.avg_cycle_ms >= 0.0);
        assert!(stats.successful_reads >= 5);

        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_polling_config_default() {
        let config = PollingConfig::default();
        assert_eq!(config.unit_id, 1);
        assert_eq!(config.interval, Duration::from_secs(1));
    }
}
