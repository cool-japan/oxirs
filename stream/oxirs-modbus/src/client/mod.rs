//! Modbus client infrastructure

pub mod connection_pool;

pub use connection_pool::{
    ConnectionStats, ModbusConnectionPool, PoolConfig, PoolStats, PooledModbusClient,
};
