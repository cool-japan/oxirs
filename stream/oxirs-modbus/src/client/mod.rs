//! Modbus client infrastructure

pub mod connection_pool;
pub mod tls;

pub use connection_pool::{
    ConnectionStats, ModbusConnectionPool, PoolConfig, PoolStats, PooledModbusClient,
};
pub use tls::{TlsConfig, TlsConfigBuilder, TlsMinVersion, TlsModbusClient, MODBUS_TLS_PORT};
