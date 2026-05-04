//! Modbus ↔ OPC UA bidirectional bridge.
//!
//! This module provides a pure-Rust, facade-based bridge between Modbus TCP/RTU
//! registers and an OPC UA address space.  Real transport implementations
//! (e.g. `async-opcua`) can be plugged in behind the [`OpcuaServerFacade`] and
//! [`ModbusClientFacade`] traits; tests use the bundled mock implementations
//! without opening any real sockets.
//!
//! # Module layout
//!
//! | Sub-module | Purpose |
//! |------------|---------|
//! | [`config`] | TOML-deserializable bridge configuration |
//! | [`type_coercion`] | Pure coercion functions between Modbus words and typed values |
//! | [`mapper`] | Lookup table for register ↔ OPC UA node mappings |
//! | [`server`] | [`OpcuaServerFacade`] trait + [`MockOpcuaServer`] |
//! | [`client`] | [`ModbusClientFacade`] trait + [`MockModbusClient`] |
//! | [`bridge`] | [`OpcuaModbusBridge`] — the main tokio task |
//!
//! # Quick start (with mocks)
//!
//! ```rust,no_run
//! use oxirs_modbus::opcua::{BridgeConfig, OpcuaModbusBridge, RegisterMapping, DataTypeSpec, Direction};
//! use oxirs_modbus::opcua::server::MockOpcuaServer;
//! use oxirs_modbus::opcua::client::MockModbusClient;
//! use tokio_util::sync::CancellationToken;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = BridgeConfig {
//!         poll_interval_ms: 1000,
//!         modbus_host: "192.168.1.100".to_owned(),
//!         modbus_port: 502,
//!         opcua_endpoint: Some("opc.tcp://localhost:4840".to_owned()),
//!         mappings: vec![RegisterMapping {
//!             modbus_register: 100,
//!             opcua_node_id: "ns=2;s=Temperature".to_owned(),
//!             data_type: DataTypeSpec::F32,
//!             direction: Direction::Read,
//!         }],
//!     };
//!
//!     let cancel = CancellationToken::new();
//!     let bridge = OpcuaModbusBridge::new(
//!         config,
//!         MockOpcuaServer::new(),
//!         MockModbusClient::new(),
//!         cancel,
//!     );
//!
//!     // bridge.run().await?;
//!     Ok(())
//! }
//! ```

pub mod bridge;
pub mod client;
pub mod config;
pub mod mapper;
pub mod server;
pub mod type_coercion;

// Flat re-exports for ergonomic use.
pub use bridge::{BridgeError, OpcuaModbusBridge};
pub use client::{MockModbusClient, ModbusClientError, ModbusClientFacade};
pub use config::{BridgeConfig, DataTypeSpec, Direction, RegisterMapping};
pub use mapper::RegisterMapper;
pub use server::{MockOpcuaServer, OpcuaError, OpcuaServerFacade};
pub use type_coercion::{registers_to_value, value_to_registers, CoercionError, DataValue};
