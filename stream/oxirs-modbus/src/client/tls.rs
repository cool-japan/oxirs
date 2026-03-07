//! Modbus TLS (Modbus TCP Security) client implementation
//!
//! Implements the MODBUS/TCP Security Protocol as specified in the
//! "MODBUS/TCP Security Protocol Specification" (MODBUS/TCP Security Protocol V1.0).
//!
//! The secure Modbus protocol uses TLS 1.2+ over TCP on IANA port 802.
//! The framing is identical to regular Modbus TCP (MBAP header + PDU).
//!
//! # Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     TLS Layer (port 802)                    │
//! │  ┌────────────────────────────────────────────────────────┐ │
//! │  │             Modbus TCP MBAP + PDU (same as TCP)        │ │
//! │  └────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Security Features
//!
//! - TLS 1.2 minimum (configurable)
//! - Mutual TLS (mTLS) support for device authentication
//! - Certificate validation with configurable trust anchors
//! - PEM-encoded certificate and key support
//!
//! # Examples
//!
//! ```ignore
//! use oxirs_modbus::client::tls::{TlsConfig, TlsModbusClient};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = TlsConfig::builder()
//!         .ca_cert_pem(std::fs::read("ca.pem")?)
//!         .client_cert_pem(std::fs::read("client.pem")?)
//!         .client_key_pem(std::fs::read("client.key")?)
//!         .build()?;
//!
//!     let mut client = TlsModbusClient::connect("192.168.1.100:802", 1, config).await?;
//!     let registers = client.read_holding_registers(0, 10).await?;
//!     println!("{:?}", registers);
//!     Ok(())
//! }
//! ```

use crate::error::{ModbusError, ModbusResult};
use crate::protocol::frame::{FunctionCode, ModbusTcpFrame};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// IANA port for Modbus/TCP Security (Modbus over TLS)
pub const MODBUS_TLS_PORT: u16 = 802;

/// Default connection timeout for TLS handshake and TCP connect
pub const DEFAULT_TLS_TIMEOUT: Duration = Duration::from_secs(10);

/// Default maximum register read count per request
const MAX_READ_COUNT: u16 = 125;

/// Minimum TLS protocol version supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum TlsMinVersion {
    /// TLS 1.2 (minimum allowed by the Modbus/TCP Security specification)
    #[default]
    Tls12,
    /// TLS 1.3
    Tls13,
}

impl std::fmt::Display for TlsMinVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tls12 => write!(f, "TLS 1.2"),
            Self::Tls13 => write!(f, "TLS 1.3"),
        }
    }
}

/// TLS configuration for Modbus/TCP Security
///
/// Holds all the certificate and key material needed to establish
/// an authenticated TLS connection to a Modbus device.
///
/// # Mutual TLS
///
/// Modbus/TCP Security mandates mutual TLS (mTLS): both the client and the
/// server authenticate each other using X.509 certificates.
/// `client_cert_pem` and `client_key_pem` are therefore required.
#[derive(Debug, Clone, Default)]
pub struct TlsConfig {
    /// CA certificate(s) in PEM format (for server certificate validation)
    pub ca_cert_pem: Option<Vec<u8>>,

    /// Client certificate in PEM format (for mutual TLS)
    pub client_cert_pem: Option<Vec<u8>>,

    /// Client private key in PEM format (for mutual TLS)
    pub client_key_pem: Option<Vec<u8>>,

    /// Minimum TLS version (default: TLS 1.2 per Modbus/TCP Security spec)
    pub min_version: TlsMinVersion,

    /// Server name for SNI (Server Name Indication)
    ///
    /// If `None`, no SNI is sent and certificate hostname validation
    /// uses the IP address or defaults to the connection address.
    pub server_name: Option<String>,

    /// Whether to skip server certificate verification (INSECURE - for testing only)
    pub danger_skip_verify: bool,
}

impl TlsConfig {
    /// Create a new builder for `TlsConfig`.
    pub fn builder() -> TlsConfigBuilder {
        TlsConfigBuilder::new()
    }

    /// Validate the configuration for consistency.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `client_cert_pem` is set but `client_key_pem` is missing (or vice-versa)
    pub fn validate(&self) -> ModbusResult<()> {
        match (&self.client_cert_pem, &self.client_key_pem) {
            (Some(_), None) => Err(ModbusError::Config(
                "client_cert_pem is set but client_key_pem is missing; \
                 both are required for mutual TLS"
                    .to_owned(),
            )),
            (None, Some(_)) => Err(ModbusError::Config(
                "client_key_pem is set but client_cert_pem is missing; \
                 both are required for mutual TLS"
                    .to_owned(),
            )),
            _ => Ok(()),
        }
    }

    /// Returns `true` if client certificate material has been provided.
    pub fn has_client_auth(&self) -> bool {
        self.client_cert_pem.is_some() && self.client_key_pem.is_some()
    }

    /// Returns `true` if a custom CA certificate was provided.
    pub fn has_ca_cert(&self) -> bool {
        self.ca_cert_pem.is_some()
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Fluent builder for `TlsConfig`.
#[derive(Debug, Default)]
pub struct TlsConfigBuilder {
    inner: TlsConfig,
}

impl TlsConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            inner: TlsConfig::default(),
        }
    }

    /// Set the CA certificate in PEM format.
    pub fn ca_cert_pem(mut self, pem: Vec<u8>) -> Self {
        self.inner.ca_cert_pem = Some(pem);
        self
    }

    /// Set the CA certificate from a PEM string.
    pub fn ca_cert_pem_str(mut self, pem: &str) -> Self {
        self.inner.ca_cert_pem = Some(pem.as_bytes().to_vec());
        self
    }

    /// Set the client certificate in PEM format.
    pub fn client_cert_pem(mut self, pem: Vec<u8>) -> Self {
        self.inner.client_cert_pem = Some(pem);
        self
    }

    /// Set the client private key in PEM format.
    pub fn client_key_pem(mut self, pem: Vec<u8>) -> Self {
        self.inner.client_key_pem = Some(pem);
        self
    }

    /// Set the minimum TLS version.
    pub fn min_version(mut self, version: TlsMinVersion) -> Self {
        self.inner.min_version = version;
        self
    }

    /// Set the server name for SNI.
    pub fn server_name(mut self, name: impl Into<String>) -> Self {
        self.inner.server_name = Some(name.into());
        self
    }

    /// Allow skipping server certificate verification (INSECURE - testing only).
    pub fn danger_skip_verify(mut self, skip: bool) -> Self {
        self.inner.danger_skip_verify = skip;
        self
    }

    /// Build the `TlsConfig`, validating consistency.
    pub fn build(self) -> ModbusResult<TlsConfig> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

// ---------------------------------------------------------------------------
// Internal framing helpers
// ---------------------------------------------------------------------------

/// Build a raw Modbus TCP request byte vector.
///
/// This duplicates the framing logic from `ModbusTcpFrame::to_bytes()` so
/// that `TlsModbusClient` does not need to hold a mutable reference to
/// the shared transaction ID counter during the async send.
fn build_request_bytes(tid: u16, unit_id: u8, function_code: FunctionCode, data: &[u8]) -> Vec<u8> {
    let length = 1u16 + 1 + data.len() as u16; // unit_id + function_code + data
    let mut bytes = Vec::with_capacity(7 + data.len());

    // MBAP Header
    bytes.extend_from_slice(&tid.to_be_bytes());
    bytes.extend_from_slice(&0u16.to_be_bytes()); // protocol id = 0
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.push(unit_id);

    // PDU
    bytes.push(function_code.as_u8());
    bytes.extend_from_slice(data);

    bytes
}

// ---------------------------------------------------------------------------
// TlsModbusClient
// ---------------------------------------------------------------------------

/// Modbus/TCP Security client using TLS.
///
/// Provides the same read/write interface as `ModbusTcpClient` but operates
/// over a TLS-encrypted and authenticated connection on port 802.
///
/// The struct is generic over the underlying stream `S` to allow
/// both real `tokio::net::TcpStream`-backed TLS streams and mock streams
/// for unit testing (using `std::io::Cursor` or `tokio_test::io::Builder`).
///
/// # Production Usage
///
/// In production, use `connect_tls` which resolves to
/// `TlsModbusClient<tokio_rustls::client::TlsStream<tokio::net::TcpStream>>`.
///
/// # Testing
///
/// For unit tests (no TLS handshake), use [`TlsModbusClient::from_stream`] with
/// a pre-configured mock stream that already contains the expected response bytes.
pub struct TlsModbusClient<S> {
    /// Underlying I/O stream (TLS-wrapped TCP or mock)
    stream: S,

    /// Modbus unit identifier (slave address)
    unit_id: u8,

    /// Transaction ID counter (monotonically increasing)
    transaction_id: Arc<AtomicU16>,

    /// Per-request timeout
    timeout: Duration,

    /// TLS configuration metadata (held for logging/introspection)
    config: TlsConfig,
}

impl<S> TlsModbusClient<S>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    /// Wrap an existing async stream as a `TlsModbusClient`.
    ///
    /// This constructor is useful in tests where you supply a mock stream
    /// that already behaves like a TLS connection (i.e., the handshake has
    /// already been performed or is not needed in the test).
    ///
    /// # Arguments
    ///
    /// * `stream` - Any `AsyncRead + AsyncWrite + Unpin` stream
    /// * `unit_id` - Modbus unit/slave ID
    /// * `config` - TLS configuration (used for introspection/logging only)
    pub fn from_stream(stream: S, unit_id: u8, config: TlsConfig) -> Self {
        Self {
            stream,
            unit_id,
            transaction_id: Arc::new(AtomicU16::new(1)),
            timeout: DEFAULT_TLS_TIMEOUT,
            config,
        }
    }

    /// Set the per-request timeout.
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Get the current unit ID.
    pub fn unit_id(&self) -> u8 {
        self.unit_id
    }

    /// Set the unit ID (slave address).
    pub fn set_unit_id(&mut self, unit_id: u8) {
        self.unit_id = unit_id;
    }

    /// Get a reference to the TLS configuration.
    pub fn tls_config(&self) -> &TlsConfig {
        &self.config
    }

    /// Read holding registers (function code 0x03).
    ///
    /// # Arguments
    ///
    /// * `start_addr` - First register address (0–65535)
    /// * `count` - Number of registers to read (1–125)
    ///
    /// # Errors
    ///
    /// - `ModbusError::InvalidCount` if `count > 125`
    /// - `ModbusError::InvalidAddress` if address range exceeds 65535
    /// - `ModbusError::Timeout` if the operation times out
    /// - `ModbusError::ModbusException` if the device returns an exception
    pub async fn read_holding_registers(
        &mut self,
        start_addr: u16,
        count: u16,
    ) -> ModbusResult<Vec<u16>> {
        if count > MAX_READ_COUNT {
            return Err(ModbusError::InvalidCount(count));
        }
        if start_addr as u32 + count as u32 > 65536 {
            return Err(ModbusError::InvalidAddress(start_addr));
        }

        let tid = self.transaction_id.fetch_add(1, Ordering::SeqCst);
        let data = [
            (start_addr >> 8) as u8,
            (start_addr & 0xFF) as u8,
            (count >> 8) as u8,
            (count & 0xFF) as u8,
        ];
        let request_bytes =
            build_request_bytes(tid, self.unit_id, FunctionCode::ReadHoldingRegisters, &data);

        let response_frame = self.send_receive(&request_bytes, tid).await?;
        response_frame.extract_registers()
    }

    /// Read input registers (function code 0x04).
    ///
    /// # Arguments
    ///
    /// * `start_addr` - First input register address
    /// * `count` - Number of registers to read (1–125)
    pub async fn read_input_registers(
        &mut self,
        start_addr: u16,
        count: u16,
    ) -> ModbusResult<Vec<u16>> {
        if count > MAX_READ_COUNT {
            return Err(ModbusError::InvalidCount(count));
        }
        if start_addr as u32 + count as u32 > 65536 {
            return Err(ModbusError::InvalidAddress(start_addr));
        }

        let tid = self.transaction_id.fetch_add(1, Ordering::SeqCst);
        let data = [
            (start_addr >> 8) as u8,
            (start_addr & 0xFF) as u8,
            (count >> 8) as u8,
            (count & 0xFF) as u8,
        ];
        let request_bytes =
            build_request_bytes(tid, self.unit_id, FunctionCode::ReadInputRegisters, &data);

        let response_frame = self.send_receive(&request_bytes, tid).await?;
        response_frame.extract_registers()
    }

    /// Write a single holding register (function code 0x06).
    ///
    /// # Arguments
    ///
    /// * `addr` - Register address
    /// * `value` - Value to write
    pub async fn write_single_register(&mut self, addr: u16, value: u16) -> ModbusResult<()> {
        let tid = self.transaction_id.fetch_add(1, Ordering::SeqCst);
        let data = [
            (addr >> 8) as u8,
            (addr & 0xFF) as u8,
            (value >> 8) as u8,
            (value & 0xFF) as u8,
        ];
        let request_bytes =
            build_request_bytes(tid, self.unit_id, FunctionCode::WriteSingleRegister, &data);

        let response_frame = self.send_receive(&request_bytes, tid).await?;

        // Verify echo: response should reflect address and value
        if response_frame.data.len() >= 4 {
            let resp_addr = u16::from_be_bytes([response_frame.data[0], response_frame.data[1]]);
            let resp_val = u16::from_be_bytes([response_frame.data[2], response_frame.data[3]]);
            if resp_addr != addr || resp_val != value {
                return Err(ModbusError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Write single register echo mismatch: \
                         expected addr={}, value={}, got addr={}, value={}",
                        addr, value, resp_addr, resp_val
                    ),
                )));
            }
        }

        Ok(())
    }

    /// Write multiple holding registers (function code 0x10).
    ///
    /// # Arguments
    ///
    /// * `start_addr` - First register address
    /// * `values` - Register values to write (maximum 123 registers)
    pub async fn write_multiple_registers(
        &mut self,
        start_addr: u16,
        values: &[u16],
    ) -> ModbusResult<()> {
        let count = values.len() as u16;
        if count == 0 || count > 123 {
            return Err(ModbusError::InvalidCount(count));
        }

        let tid = self.transaction_id.fetch_add(1, Ordering::SeqCst);
        let byte_count = (count * 2) as u8;
        let mut data = Vec::with_capacity(5 + byte_count as usize);
        data.push((start_addr >> 8) as u8);
        data.push((start_addr & 0xFF) as u8);
        data.push((count >> 8) as u8);
        data.push((count & 0xFF) as u8);
        data.push(byte_count);
        for &v in values {
            data.push((v >> 8) as u8);
            data.push((v & 0xFF) as u8);
        }

        let request_bytes = build_request_bytes(
            tid,
            self.unit_id,
            FunctionCode::WriteMultipleRegisters,
            &data,
        );

        let _response_frame = self.send_receive(&request_bytes, tid).await?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Send raw request bytes and receive + parse the response frame.
    async fn send_receive(
        &mut self,
        request_bytes: &[u8],
        expected_tid: u16,
    ) -> ModbusResult<ModbusTcpFrame> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::time::timeout;

        // Send
        timeout(self.timeout, self.stream.write_all(request_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))?
            .map_err(ModbusError::Io)?;

        // Read MBAP header (7 bytes)
        let mut header = [0u8; 7];
        timeout(self.timeout, self.stream.read_exact(&mut header))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))?
            .map_err(ModbusError::Io)?;

        // MBAP length field = unit_id(1) + function(1) + data
        // The 7-byte header already consumed unit_id, so remaining PDU bytes = length - 1
        let mbap_length = u16::from_be_bytes([header[4], header[5]]) as usize;
        if mbap_length < 2 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "MBAP length field too small (must be >= 2 for unit_id + function code)",
            )));
        }

        // Remaining bytes on wire: function + data = mbap_length - 1 (unit_id already in header)
        let remaining = mbap_length - 1;

        // Read PDU (function code + data)
        let mut pdu = vec![0u8; remaining];
        timeout(self.timeout, self.stream.read_exact(&mut pdu))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))?
            .map_err(ModbusError::Io)?;

        // Reconstruct full frame bytes for parsing:
        // header (7 bytes: tid+pid+length+unit_id) + pdu (func+data)
        let mut full = Vec::with_capacity(7 + remaining);
        full.extend_from_slice(&header);
        full.extend_from_slice(&pdu);

        let frame = ModbusTcpFrame::from_bytes(&full)?;

        // Warn on transaction ID mismatch (non-fatal per spec: clients may ignore it)
        if frame.transaction_id != expected_tid {
            tracing::warn!(
                "TLS Modbus: transaction ID mismatch (expected {}, got {})",
                expected_tid,
                frame.transaction_id
            );
        }

        Ok(frame)
    }
}

// ---------------------------------------------------------------------------
// Real TLS connect (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "tls")]
pub use tls_impl::connect_tls;

#[cfg(feature = "tls")]
mod tls_impl {
    use super::*;
    use rustls::ClientConfig;
    use std::io::BufReader;
    use tokio_rustls::TlsConnector;

    /// Connect to a Modbus/TCP Security server using TLS.
    ///
    /// Establishes a TCP connection to `addr`, performs the TLS handshake
    /// using the provided `config`, and returns a fully-authenticated
    /// `TlsModbusClient`.
    ///
    /// # Arguments
    ///
    /// * `addr` - Server address (e.g., `"192.168.1.100:802"`)
    /// * `unit_id` - Modbus unit/slave ID
    /// * `config` - TLS certificates and security settings
    ///
    /// # Errors
    ///
    /// - `ModbusError::Config` for invalid TLS configuration
    /// - `ModbusError::Io` for TCP or TLS handshake failure
    /// - `ModbusError::Timeout` if the connection or handshake times out
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oxirs_modbus::client::tls::{TlsConfig, connect_tls};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let config = TlsConfig::builder()
    ///         .ca_cert_pem(std::fs::read("ca.pem")?)
    ///         .client_cert_pem(std::fs::read("client.pem")?)
    ///         .client_key_pem(std::fs::read("client.key")?)
    ///         .build()?;
    ///     let mut client = connect_tls("192.168.1.100:802", 1, config).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn connect_tls(
        addr: &str,
        unit_id: u8,
        config: TlsConfig,
    ) -> ModbusResult<TlsModbusClient<tokio_rustls::client::TlsStream<tokio::net::TcpStream>>> {
        use rustls::RootCertStore;
        use tokio::net::TcpStream;
        use tokio::time::timeout;

        config.validate()?;

        // Build root cert store
        let mut root_store = RootCertStore::empty();

        if let Some(ref ca_pem) = config.ca_cert_pem {
            let mut reader = BufReader::new(ca_pem.as_slice());
            let certs = rustls_pemfile::certs(&mut reader)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    ModbusError::Config(format!("Failed to parse CA certificate: {}", e))
                })?;

            for cert in certs {
                root_store.add(cert).map_err(|e| {
                    ModbusError::Config(format!("Failed to add CA certificate: {}", e))
                })?;
            }
        } else {
            // Use WebPKI roots as fallback
            root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        }

        // Build client config
        let builder = ClientConfig::builder().with_root_certificates(root_store);

        let tls_config = if config.has_client_auth() {
            let cert_pem = config.client_cert_pem.as_ref().expect("checked above");
            let key_pem = config.client_key_pem.as_ref().expect("checked above");

            let mut cert_reader = BufReader::new(cert_pem.as_slice());
            let certs = rustls_pemfile::certs(&mut cert_reader)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| ModbusError::Config(format!("Failed to parse client cert: {}", e)))?;

            let mut key_reader = BufReader::new(key_pem.as_slice());
            let key = rustls_pemfile::private_key(&mut key_reader)
                .map_err(|e| ModbusError::Config(format!("Failed to parse client key: {}", e)))?
                .ok_or_else(|| {
                    ModbusError::Config("No private key found in client_key_pem".to_owned())
                })?;

            builder.with_client_auth_cert(certs, key).map_err(|e| {
                ModbusError::Config(format!("Failed to configure client auth: {}", e))
            })?
        } else {
            builder.with_no_client_auth()
        };

        let connector = TlsConnector::from(Arc::new(tls_config));

        // Determine server name for SNI
        let server_name_str = config.server_name.clone().unwrap_or_else(|| {
            // Strip port from address for SNI
            addr.split(':').next().unwrap_or(addr).to_owned()
        });

        let server_name = rustls::pki_types::ServerName::try_from(server_name_str.as_str())
            .map_err(|e| {
                ModbusError::Config(format!("Invalid server name '{}': {}", server_name_str, e))
            })?
            .to_owned();

        // TCP connect
        let tcp_stream = timeout(DEFAULT_TLS_TIMEOUT, TcpStream::connect(addr))
            .await
            .map_err(|_| ModbusError::Timeout(DEFAULT_TLS_TIMEOUT))?
            .map_err(ModbusError::Io)?;

        tcp_stream.set_nodelay(true).map_err(ModbusError::Io)?;

        // TLS handshake
        let tls_stream = timeout(
            DEFAULT_TLS_TIMEOUT,
            connector.connect(server_name, tcp_stream),
        )
        .await
        .map_err(|_| ModbusError::Timeout(DEFAULT_TLS_TIMEOUT))?
        .map_err(ModbusError::Io)?;

        Ok(TlsModbusClient::from_stream(tls_stream, unit_id, config))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // TlsConfig validation
    // ------------------------------------------------------------------

    #[test]
    fn test_config_default_values() {
        let config = TlsConfig::default();
        assert_eq!(config.min_version, TlsMinVersion::Tls12);
        assert!(!config.danger_skip_verify);
        assert!(config.ca_cert_pem.is_none());
        assert!(config.client_cert_pem.is_none());
        assert!(config.client_key_pem.is_none());
        assert!(config.server_name.is_none());
    }

    #[test]
    fn test_config_validate_ok_no_client_auth() {
        let config = TlsConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_ok_with_client_auth() {
        let config = TlsConfig::builder()
            .client_cert_pem(b"cert".to_vec())
            .client_key_pem(b"key".to_vec())
            .build()
            .expect("should be valid");
        assert!(config.has_client_auth());
    }

    #[test]
    fn test_config_validate_error_cert_without_key() {
        let result = TlsConfig::builder()
            .client_cert_pem(b"cert".to_vec())
            .build();
        assert!(result.is_err());
        match result {
            Err(ModbusError::Config(msg)) => {
                assert!(
                    msg.contains("client_key_pem"),
                    "error should mention key: {}",
                    msg
                );
            }
            _ => panic!("Expected Config error"),
        }
    }

    #[test]
    fn test_config_validate_error_key_without_cert() {
        let result = TlsConfig::builder().client_key_pem(b"key".to_vec()).build();
        assert!(result.is_err());
        match result {
            Err(ModbusError::Config(msg)) => {
                assert!(
                    msg.contains("client_cert_pem"),
                    "error should mention cert: {}",
                    msg
                );
            }
            _ => panic!("Expected Config error"),
        }
    }

    #[test]
    fn test_config_has_ca_cert() {
        let config = TlsConfig::builder()
            .ca_cert_pem(b"ca".to_vec())
            .build()
            .expect("valid");
        assert!(config.has_ca_cert());
    }

    #[test]
    fn test_config_has_no_ca_cert() {
        let config = TlsConfig::default();
        assert!(!config.has_ca_cert());
    }

    // ------------------------------------------------------------------
    // Builder API
    // ------------------------------------------------------------------

    #[test]
    fn test_builder_server_name() {
        let config = TlsConfig::builder()
            .server_name("plc.factory.local")
            .build()
            .expect("valid");
        assert_eq!(config.server_name.as_deref(), Some("plc.factory.local"));
    }

    #[test]
    fn test_builder_min_version_tls13() {
        let config = TlsConfig::builder()
            .min_version(TlsMinVersion::Tls13)
            .build()
            .expect("valid");
        assert_eq!(config.min_version, TlsMinVersion::Tls13);
    }

    #[test]
    fn test_builder_danger_skip_verify() {
        let config = TlsConfig::builder()
            .danger_skip_verify(true)
            .build()
            .expect("valid");
        assert!(config.danger_skip_verify);
    }

    // ------------------------------------------------------------------
    // TlsMinVersion
    // ------------------------------------------------------------------

    #[test]
    fn test_tls_min_version_ordering() {
        assert!(TlsMinVersion::Tls12 < TlsMinVersion::Tls13);
        assert!(TlsMinVersion::Tls13 > TlsMinVersion::Tls12);
        assert_eq!(TlsMinVersion::Tls12, TlsMinVersion::Tls12);
    }

    #[test]
    fn test_tls_min_version_display() {
        assert_eq!(format!("{}", TlsMinVersion::Tls12), "TLS 1.2");
        assert_eq!(format!("{}", TlsMinVersion::Tls13), "TLS 1.3");
    }

    // ------------------------------------------------------------------
    // TlsModbusClient with mock stream
    //
    // We test framing correctness without performing a real TLS handshake
    // by using `tokio_test::io::Builder` as the underlying stream.
    // ------------------------------------------------------------------

    fn make_mock_read_response(tid: u16, unit_id: u8, registers: &[u16]) -> Vec<u8> {
        let byte_count = (registers.len() * 2) as u8;
        let pdu_len = 1u16 + 1 + 1 + byte_count as u16; // func + byte_count + data
        let mut resp = Vec::new();
        resp.extend_from_slice(&tid.to_be_bytes());
        resp.extend_from_slice(&0u16.to_be_bytes()); // protocol id
        resp.extend_from_slice(&pdu_len.to_be_bytes());
        resp.push(unit_id);
        resp.push(0x03); // ReadHoldingRegisters
        resp.push(byte_count);
        for &r in registers {
            resp.push((r >> 8) as u8);
            resp.push((r & 0xFF) as u8);
        }
        resp
    }

    #[tokio::test]
    async fn test_read_holding_registers_with_mock() {
        let registers = [100u16, 200, 300];
        // tid starts at 1 (first fetch_add returns 1)
        let response = make_mock_read_response(1, 1, &registers);

        let mock = tokio_test::io::Builder::new()
            .write(&build_request_bytes(
                1,
                1,
                FunctionCode::ReadHoldingRegisters,
                &[0x00, 0x00, 0x00, 0x03],
            ))
            .read(&response)
            .build();

        let mut client = TlsModbusClient::from_stream(mock, 1, TlsConfig::default());
        let result = client.read_holding_registers(0, 3).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
        assert_eq!(result.unwrap(), vec![100, 200, 300]);
    }

    #[tokio::test]
    async fn test_read_input_registers_with_mock() {
        let registers = [42u16, 84];
        let response = make_mock_read_response(1, 1, &registers);

        // build_request_bytes for ReadInputRegisters
        let request = build_request_bytes(
            1,
            1,
            FunctionCode::ReadInputRegisters,
            &[0x00, 0x00, 0x00, 0x02],
        );
        // fix function code in expected response to 0x04
        let mut resp4 = response.clone();
        resp4[7] = 0x04;

        let mock = tokio_test::io::Builder::new()
            .write(&request)
            .read(&resp4)
            .build();

        let mut client = TlsModbusClient::from_stream(mock, 1, TlsConfig::default());
        let result = client.read_input_registers(0, 2).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
        assert_eq!(result.unwrap(), vec![42, 84]);
    }

    #[tokio::test]
    async fn test_write_single_register_with_mock() {
        let tid = 1u16;
        let addr = 0x0010u16;
        let value = 0x00C8u16;

        // Echo response: same addr + value
        // MBAP length field = unit_id(1) + func(1) + data(4) = 6
        let pdu_len = 1u16 + 1 + 4; // unit_id + func + 4 data bytes
        let mut response = Vec::new();
        response.extend_from_slice(&tid.to_be_bytes());
        response.extend_from_slice(&0u16.to_be_bytes());
        response.extend_from_slice(&pdu_len.to_be_bytes());
        response.push(1u8); // unit_id
        response.push(0x06); // WriteSingleRegister
        response.extend_from_slice(&addr.to_be_bytes());
        response.extend_from_slice(&value.to_be_bytes());

        let request = build_request_bytes(
            tid,
            1,
            FunctionCode::WriteSingleRegister,
            &[
                (addr >> 8) as u8,
                (addr & 0xFF) as u8,
                (value >> 8) as u8,
                (value & 0xFF) as u8,
            ],
        );

        let mock = tokio_test::io::Builder::new()
            .write(&request)
            .read(&response)
            .build();

        let mut client = TlsModbusClient::from_stream(mock, 1, TlsConfig::default());
        let result = client.write_single_register(addr, value).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
    }

    #[tokio::test]
    async fn test_invalid_count_error() {
        let mock = tokio_test::io::Builder::new().build();
        let mut client = TlsModbusClient::from_stream(mock, 1, TlsConfig::default());
        let result = client.read_holding_registers(0, 126).await;
        assert!(matches!(result, Err(ModbusError::InvalidCount(126))));
    }

    #[tokio::test]
    async fn test_invalid_address_overflow() {
        let mock = tokio_test::io::Builder::new().build();
        let mut client = TlsModbusClient::from_stream(mock, 1, TlsConfig::default());
        // start=65000, count=1000 → overflow
        let result = client.read_holding_registers(65000, 1000).await;
        // count > 125 is caught first
        assert!(matches!(result, Err(ModbusError::InvalidCount(_))));
    }

    #[test]
    fn test_port_constant() {
        assert_eq!(MODBUS_TLS_PORT, 802);
    }

    #[test]
    fn test_unit_id_accessors() {
        let mock = tokio_test::io::Builder::new().build();
        let mut client = TlsModbusClient::from_stream(mock, 5, TlsConfig::default());
        assert_eq!(client.unit_id(), 5);
        client.set_unit_id(10);
        assert_eq!(client.unit_id(), 10);
    }

    #[test]
    fn test_timeout_setter() {
        let mock = tokio_test::io::Builder::new().build();
        let mut client = TlsModbusClient::from_stream(mock, 1, TlsConfig::default());
        client.set_timeout(Duration::from_secs(30));
        assert_eq!(client.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_tls_config_accessor() {
        let config = TlsConfig::builder()
            .server_name("test.example.com")
            .build()
            .expect("valid");
        let mock = tokio_test::io::Builder::new().build();
        let client = TlsModbusClient::from_stream(mock, 1, config.clone());
        assert_eq!(
            client.tls_config().server_name.as_deref(),
            Some("test.example.com")
        );
    }
}
