use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Modbus client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModbusConfig {
    /// Protocol type (TCP or RTU)
    pub protocol: ModbusProtocol,

    /// TCP host address (TCP only)
    pub host: Option<String>,

    /// TCP port number (TCP only, default: 502)
    pub port: Option<u16>,

    /// Serial port path (RTU only)
    pub serial_port: Option<String>,

    /// Baud rate (RTU only, default: 9600)
    pub baud_rate: Option<u32>,

    /// Modbus unit ID (slave address)
    pub unit_id: u8,

    /// Polling interval (in seconds)
    #[serde(default = "default_polling_interval")]
    pub polling_interval_secs: u64,

    /// Connection timeout (in seconds)
    #[serde(default = "default_connection_timeout")]
    pub connection_timeout_secs: u64,

    /// Retry attempts on failure
    pub retry_attempts: usize,
}

/// Modbus protocol type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModbusProtocol {
    /// Modbus TCP (port 502, Ethernet)
    TCP,
    /// Modbus RTU (serial RS-232/RS-485)
    RTU,
}

fn default_polling_interval() -> u64 {
    1
}

fn default_connection_timeout() -> u64 {
    5
}

impl Default for ModbusConfig {
    fn default() -> Self {
        Self {
            protocol: ModbusProtocol::TCP,
            host: Some("127.0.0.1".to_string()),
            port: Some(502),
            serial_port: None,
            baud_rate: None,
            unit_id: 1,
            polling_interval_secs: 1,
            connection_timeout_secs: 5,
            retry_attempts: 3,
        }
    }
}

impl ModbusConfig {
    /// Get polling interval as Duration
    pub fn polling_interval(&self) -> Duration {
        Duration::from_secs(self.polling_interval_secs)
    }

    /// Get connection timeout as Duration
    pub fn connection_timeout(&self) -> Duration {
        Duration::from_secs(self.connection_timeout_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ModbusConfig::default();
        assert_eq!(config.protocol, ModbusProtocol::TCP);
        assert_eq!(config.port, Some(502));
        assert_eq!(config.unit_id, 1);
    }
}
