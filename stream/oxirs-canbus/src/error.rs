use thiserror::Error;

/// Errors that can occur when working with CANbus
#[derive(Error, Debug)]
pub enum CanbusError {
    /// I/O error during CAN communication
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid CAN identifier
    #[error("Invalid CAN ID: {0:08X} (max: 0x1FFFFFFF for extended)")]
    InvalidCanId(u32),

    /// DBC file parsing error
    #[error("DBC parse error at line {line}: {message}")]
    DbcParseError {
        /// Line number where error occurred
        line: usize,
        /// Error message
        message: String,
    },

    /// CAN interface not found
    #[error("CAN interface not found: {0}")]
    InterfaceNotFound(String),

    /// CAN frame too large
    #[error("CAN frame too large: {0} bytes (max: 8 for CAN 2.0, 64 for CAN FD)")]
    FrameTooLarge(usize),

    /// Signal not found in DBC
    #[error("Signal not found in DBC: {0}")]
    SignalNotFound(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// RDF mapping error
    #[error("RDF mapping error: {0}")]
    RdfMapping(String),
}

/// Result type for CANbus operations
pub type CanbusResult<T> = Result<T, CanbusError>;
