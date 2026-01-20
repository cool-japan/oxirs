use thiserror::Error;

/// Errors that can occur when working with Modbus protocols
#[derive(Error, Debug)]
pub enum ModbusError {
    /// I/O error during communication
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// CRC checksum mismatch (RTU only)
    #[error("CRC error: expected {expected:04x}, got {actual:04x}")]
    CrcError {
        /// Expected CRC value
        expected: u16,
        /// Actual CRC value received
        actual: u16,
    },

    /// Invalid Modbus function code
    #[error("Invalid function code: {0}")]
    InvalidFunctionCode(u8),

    /// Connection timeout
    #[error("Connection timeout after {0:?}")]
    Timeout(std::time::Duration),

    /// Invalid register address
    #[error("Invalid register address: {0} (max: 65535)")]
    InvalidAddress(u16),

    /// Invalid register count
    #[error("Invalid register count: {0} (max: 125)")]
    InvalidCount(u16),

    /// Modbus exception response
    #[error("Modbus exception: code {code}, function {function}")]
    ModbusException {
        /// Exception code (0x01-0x0B)
        code: u8,
        /// Function code that caused the exception
        function: u8,
    },

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// RDF mapping error
    #[error("RDF mapping error: {0}")]
    RdfMapping(String),
}

/// Result type for Modbus operations
pub type ModbusResult<T> = Result<T, ModbusError>;
