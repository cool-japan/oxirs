//! Polling scheduler and change detection
//!
//! This module provides efficient polling mechanisms with
//! deadband filtering, batch operations, and automatic reconnection.
//!
//! # Overview
//!
//! The polling module provides:
//! - Scheduled polling with configurable intervals
//! - Automatic connection management and reconnection
//! - Batch reading for efficiency
//! - Change detection with deadband filtering
//! - Event-driven triple generation
//!
//! # Example
//!
//! ```ignore
//! use oxirs_modbus::polling::{PollingScheduler, PollingConfig, PollingEvent};
//! use oxirs_modbus::mapping::{RegisterMap, RegisterMapping, ModbusDataType};
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Configure polling
//!     let config = PollingConfig {
//!         address: "192.168.1.100:502".to_string(),
//!         unit_id: 1,
//!         interval: Duration::from_secs(1),
//!         ..Default::default()
//!     };
//!
//!     // Create register map
//!     let mut map = RegisterMap::new("plc001", "http://factory.example.com/device");
//!     map.add_register(RegisterMapping::new(
//!         0, ModbusDataType::Float32,
//!         "http://factory.example.com/property/temperature"
//!     ));
//!
//!     // Start polling
//!     let scheduler = PollingScheduler::new(config, map);
//!     let (mut events, stop) = scheduler.run().await;
//!
//!     // Process events
//!     while let Ok(event) = events.recv().await {
//!         match event {
//!             PollingEvent::Triples(triples) => {
//!                 println!("Got {} triples", triples.len());
//!             }
//!             PollingEvent::Error(e) => {
//!                 eprintln!("Error: {}", e);
//!             }
//!             _ => {}
//!         }
//!     }
//! }
//! ```

pub mod scheduler;

// Re-exports
pub use scheduler::{
    shared_scheduler, PollingConfig, PollingEvent, PollingScheduler, PollingStats, SharedScheduler,
};
