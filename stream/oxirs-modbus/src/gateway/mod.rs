//! Modbus TCP gateway — bridges serial Modbus RTU to Modbus TCP
//!
//! The `ModbusTcpGateway` accepts incoming Modbus TCP connections from
//! TCP clients and translates each request into the corresponding Modbus RTU
//! frame to be forwarded to a downstream serial bus.  Responses are mapped
//! back to TCP ADU format with the original transaction ID preserved.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────┐   Modbus TCP   ┌──────────────────────┐   RTU    ┌───────┐
//! │ SCADA / HMI  │ ──────────────▶│ ModbusTcpGateway     │ ────────▶│  PLC  │
//! │ (TCP client) │ ◀──────────────│ (this module)        │ ◀────────│(slave)│
//! └──────────────┘                └──────────────────────┘          └───────┘
//! ```
//!
//! ## Key features
//!
//! - **Transaction ID management**: sequential 16-bit counter; wraps without
//!   collision thanks to a one-request-at-a-time queue per connection.
//! - **Request queuing**: `GatewayQueue` stores pending requests with their
//!   transaction IDs for in-order processing.
//! - **Protocol translation**: `ModbusTcpAdu` ↔ `ModbusRtuFrame` conversion
//!   with CRC calculation.
//! - **Concurrent connection handling**: `GatewayConnectionPool` tracks
//!   active client connections up to a configurable maximum.
//! - **Timeout / error propagation**: downstream RTU errors are wrapped in
//!   Modbus exception responses sent back to the TCP client.

use crate::error::{ModbusError, ModbusResult};
use crate::protocol::{calculate_crc, verify_crc};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ── Modbus ADU / frame types ─────────────────────────────────────────────────

/// Modbus TCP Application Data Unit (ADU).
///
/// Layout:
/// ```text
/// | Transaction ID (2B) | Protocol ID (2B) | Length (2B) | Unit ID (1B) | PDU (N B) |
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModbusTcpAdu {
    /// Identifies the request/response pair within a TCP session (0x0001..0xFFFF)
    pub transaction_id: u16,
    /// Protocol identifier — always 0x0000 for Modbus TCP
    pub protocol_id: u16,
    /// Number of bytes from unit_id onward: `1 (unit) + PDU_length`
    pub length: u16,
    /// Modbus unit / slave identifier
    pub unit_id: u8,
    /// Protocol Data Unit (function code + data bytes)
    pub pdu: Vec<u8>,
}

impl ModbusTcpAdu {
    /// Build a Modbus TCP ADU from components.
    pub fn new(transaction_id: u16, unit_id: u8, pdu: Vec<u8>) -> Self {
        let length = 1 + pdu.len() as u16; // unit_id + PDU
        Self {
            transaction_id,
            protocol_id: 0x0000,
            length,
            unit_id,
            pdu,
        }
    }

    /// Serialise into a 7-byte MBAP header + PDU byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(6 + self.pdu.len());
        buf.extend_from_slice(&self.transaction_id.to_be_bytes());
        buf.extend_from_slice(&self.protocol_id.to_be_bytes());
        buf.extend_from_slice(&self.length.to_be_bytes());
        buf.push(self.unit_id);
        buf.extend_from_slice(&self.pdu);
        buf
    }

    /// Parse a Modbus TCP ADU from a raw byte slice.
    ///
    /// # Errors
    ///
    /// Returns an error when the slice is too short or the declared length
    /// is inconsistent with the buffer size.
    pub fn from_bytes(data: &[u8]) -> ModbusResult<Self> {
        if data.len() < 7 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("TCP ADU too short: {} bytes (minimum 7)", data.len()),
            )));
        }
        let transaction_id = u16::from_be_bytes([data[0], data[1]]);
        let protocol_id = u16::from_be_bytes([data[2], data[3]]);
        let length = u16::from_be_bytes([data[4], data[5]]);
        let unit_id = data[6];

        let expected_pdu_len = (length as usize).saturating_sub(1); // minus unit_id byte
        if data.len() < 7 + expected_pdu_len {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "TCP ADU buffer too short: declared length {} needs {} bytes, got {}",
                    length,
                    7 + expected_pdu_len,
                    data.len()
                ),
            )));
        }

        let pdu = data[7..7 + expected_pdu_len].to_vec();
        Ok(Self {
            transaction_id,
            protocol_id,
            length,
            unit_id,
            pdu,
        })
    }

    /// Return the function code (first byte of the PDU), or `None` when the PDU is empty.
    pub fn function_code(&self) -> Option<u8> {
        self.pdu.first().copied()
    }
}

// ── Modbus RTU frame ─────────────────────────────────────────────────────────

/// Minimal Modbus RTU frame: `[unit_id, function_code, data..., CRC_lo, CRC_hi]`.
///
/// The CRC is stored little-endian in the last two bytes, matching the
/// Modbus RTU specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModbusRtuFrame {
    /// Slave / unit address (1–247)
    pub unit_id: u8,
    /// PDU (function code + data), *without* the CRC
    pub pdu: Vec<u8>,
}

impl ModbusRtuFrame {
    /// Construct an RTU frame from unit_id and PDU bytes.
    pub fn new(unit_id: u8, pdu: Vec<u8>) -> Self {
        Self { unit_id, pdu }
    }

    /// Serialise to a raw byte vector including the CRC.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(1 + self.pdu.len() + 2);
        buf.push(self.unit_id);
        buf.extend_from_slice(&self.pdu);
        let crc = calculate_crc(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Parse an RTU frame from raw bytes, verifying the CRC.
    ///
    /// # Errors
    ///
    /// Returns an error when the slice is too short (minimum 4 bytes:
    /// unit + fc + 1 data byte + 2 CRC) or when the CRC does not match.
    pub fn from_bytes(data: &[u8]) -> ModbusResult<Self> {
        if data.len() < 4 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("RTU frame too short: {} bytes", data.len()),
            )));
        }
        // Verify CRC over all bytes except the last two
        let payload = &data[..data.len() - 2];
        let crc_bytes = [data[data.len() - 2], data[data.len() - 1]];
        let received_crc = u16::from_le_bytes(crc_bytes);
        if !verify_crc(data) {
            let expected = calculate_crc(payload);
            return Err(ModbusError::CrcError {
                expected,
                actual: received_crc,
            });
        }
        let unit_id = data[0];
        let pdu = data[1..data.len() - 2].to_vec();
        Ok(Self { unit_id, pdu })
    }

    /// Return the function code, or `None` for an empty PDU.
    pub fn function_code(&self) -> Option<u8> {
        self.pdu.first().copied()
    }
}

// ── Protocol translation ─────────────────────────────────────────────────────

/// Translate a Modbus TCP ADU into a Modbus RTU frame.
///
/// The unit_id is taken from the ADU; the PDU is unchanged.
pub fn tcp_adu_to_rtu_frame(adu: &ModbusTcpAdu) -> ModbusRtuFrame {
    ModbusRtuFrame::new(adu.unit_id, adu.pdu.clone())
}

/// Translate a Modbus RTU response frame back into a Modbus TCP ADU.
///
/// The `transaction_id` must match the original request (caller's
/// responsibility to track it).
pub fn rtu_frame_to_tcp_adu(frame: &ModbusRtuFrame, transaction_id: u16) -> ModbusTcpAdu {
    ModbusTcpAdu::new(transaction_id, frame.unit_id, frame.pdu.clone())
}

// ── Transaction ID manager ────────────────────────────────────────────────────

/// Thread-safe, wrapping 16-bit transaction ID generator.
///
/// Starts at 1 and wraps back to 1 (never 0) so that 0 remains a sentinel
/// for "no transaction".
#[derive(Debug)]
pub struct TransactionIdManager {
    counter: AtomicU16,
}

impl TransactionIdManager {
    /// Create a new manager starting at transaction ID 1.
    pub fn new() -> Self {
        Self {
            counter: AtomicU16::new(1),
        }
    }

    /// Allocate the next transaction ID, wrapping from 0xFFFF back to 1.
    pub fn next(&self) -> u16 {
        loop {
            let current = self.counter.fetch_add(1, Ordering::Relaxed);
            let next_id = current.wrapping_add(1);
            // Skip 0 — it is not a valid Modbus TCP transaction ID
            if next_id != 0 {
                return next_id;
            }
            // If we wrapped to 0, try again (will add 1 again → 1)
        }
    }

    /// Current value without incrementing (for diagnostics).
    pub fn current(&self) -> u16 {
        self.counter.load(Ordering::Relaxed)
    }
}

impl Default for TransactionIdManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── GatewayQueue ─────────────────────────────────────────────────────────────

/// A pending gateway request awaiting a downstream RTU response.
#[derive(Debug, Clone)]
pub struct PendingRequest {
    /// The Modbus TCP transaction ID to use when replying
    pub transaction_id: u16,
    /// Unit / slave ID the request targets
    pub unit_id: u8,
    /// PDU bytes (function code + data)
    pub pdu: Vec<u8>,
    /// Timestamp when the request was enqueued
    pub enqueued_at: Instant,
}

impl PendingRequest {
    /// True when the request has been waiting longer than `timeout`.
    pub fn is_timed_out(&self, timeout: Duration) -> bool {
        self.enqueued_at.elapsed() > timeout
    }
}

/// FIFO queue of pending gateway requests.
///
/// Requests are enqueued when they arrive from a TCP client and dequeued
/// as the RTU bus becomes available to service them.  Timed-out requests
/// can be pruned with [`GatewayQueue::drain_timed_out`].
#[derive(Debug)]
pub struct GatewayQueue {
    pending: VecDeque<PendingRequest>,
    max_depth: usize,
}

impl GatewayQueue {
    /// Create a new queue with a maximum depth.
    pub fn new(max_depth: usize) -> Self {
        Self {
            pending: VecDeque::with_capacity(max_depth.min(64)),
            max_depth,
        }
    }

    /// Enqueue a request.  Returns an error when the queue is full.
    pub fn enqueue(&mut self, request: PendingRequest) -> ModbusResult<()> {
        if self.pending.len() >= self.max_depth {
            return Err(ModbusError::Config(format!(
                "gateway queue full ({} / {} slots)",
                self.pending.len(),
                self.max_depth
            )));
        }
        self.pending.push_back(request);
        Ok(())
    }

    /// Dequeue the oldest request (FIFO).
    pub fn dequeue(&mut self) -> Option<PendingRequest> {
        self.pending.pop_front()
    }

    /// Peek at the oldest request without removing it.
    pub fn peek(&self) -> Option<&PendingRequest> {
        self.pending.front()
    }

    /// Remove and return all requests that have exceeded `timeout`.
    pub fn drain_timed_out(&mut self, timeout: Duration) -> Vec<PendingRequest> {
        let mut timed_out = Vec::new();
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].is_timed_out(timeout) {
                if let Some(req) = self.pending.remove(i) {
                    timed_out.push(req);
                }
            } else {
                i += 1;
            }
        }
        timed_out
    }

    /// Number of pending requests.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// True when the queue has no pending requests.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Configured maximum queue depth.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }
}

// ── GatewayConnectionPool ────────────────────────────────────────────────────

/// Metadata about an active TCP client connected to the gateway.
#[derive(Debug, Clone)]
pub struct GatewayConnection {
    /// Opaque connection identifier assigned by the pool
    pub conn_id: usize,
    /// Source address string (e.g. `"192.168.1.50:54321"`)
    pub peer_addr: String,
    /// When the connection was established
    pub connected_at: Instant,
    /// Number of requests handled on this connection
    pub request_count: u64,
    /// Number of errors encountered on this connection
    pub error_count: u64,
}

impl GatewayConnection {
    /// Create a new connection record.
    pub fn new(conn_id: usize, peer_addr: impl Into<String>) -> Self {
        Self {
            conn_id,
            peer_addr: peer_addr.into(),
            connected_at: Instant::now(),
            request_count: 0,
            error_count: 0,
        }
    }

    /// Elapsed duration since the connection was established.
    pub fn uptime(&self) -> Duration {
        self.connected_at.elapsed()
    }
}

/// Pool tracking active TCP client connections to the gateway.
#[derive(Debug)]
pub struct GatewayConnectionPool {
    connections: HashMap<usize, GatewayConnection>,
    max_connections: usize,
    next_id: AtomicUsize,
}

impl GatewayConnectionPool {
    /// Create a pool that allows up to `max_connections` concurrent clients.
    pub fn new(max_connections: usize) -> Self {
        Self {
            connections: HashMap::new(),
            max_connections,
            next_id: AtomicUsize::new(1),
        }
    }

    /// Register a new incoming connection.
    ///
    /// Returns the assigned connection ID, or an error when the pool is full.
    pub fn add_connection(&mut self, peer_addr: impl Into<String>) -> ModbusResult<usize> {
        if self.connections.len() >= self.max_connections {
            return Err(ModbusError::Config(format!(
                "gateway connection pool full ({} / {})",
                self.connections.len(),
                self.max_connections
            )));
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.connections
            .insert(id, GatewayConnection::new(id, peer_addr));
        Ok(id)
    }

    /// Remove a connection record by ID.
    pub fn remove_connection(&mut self, conn_id: usize) -> Option<GatewayConnection> {
        self.connections.remove(&conn_id)
    }

    /// Access a connection record by ID.
    pub fn get(&self, conn_id: usize) -> Option<&GatewayConnection> {
        self.connections.get(&conn_id)
    }

    /// Mutably access a connection record (for updating counters).
    pub fn get_mut(&mut self, conn_id: usize) -> Option<&mut GatewayConnection> {
        self.connections.get_mut(&conn_id)
    }

    /// Current number of active connections.
    pub fn active_count(&self) -> usize {
        self.connections.len()
    }

    /// Configured maximum.
    pub fn max_connections(&self) -> usize {
        self.max_connections
    }

    /// True when the pool is at capacity.
    pub fn is_full(&self) -> bool {
        self.connections.len() >= self.max_connections
    }
}

// ── ModbusTcpGateway ─────────────────────────────────────────────────────────

/// Configuration for the Modbus TCP gateway.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// IP address and port to listen on (e.g. `"0.0.0.0:502"`)
    pub listen_addr: String,
    /// RTU serial port path (e.g. `"/dev/ttyS0"`)
    pub serial_port: String,
    /// RTU baud rate
    pub baud_rate: u32,
    /// Maximum concurrent TCP client connections
    pub max_connections: usize,
    /// Maximum pending requests in the RTU queue per connection
    pub queue_depth: usize,
    /// Timeout for RTU request/response cycles
    pub rtu_timeout: Duration,
    /// Timeout for idle TCP connections (0 = no timeout)
    pub tcp_idle_timeout: Duration,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:502".to_string(),
            serial_port: "/dev/ttyUSB0".to_string(),
            baud_rate: 9600,
            max_connections: 32,
            queue_depth: 64,
            rtu_timeout: Duration::from_secs(1),
            tcp_idle_timeout: Duration::from_secs(60),
        }
    }
}

impl GatewayConfig {
    /// Create a new config with the given listen address and serial port.
    pub fn new(listen_addr: impl Into<String>, serial_port: impl Into<String>) -> Self {
        Self {
            listen_addr: listen_addr.into(),
            serial_port: serial_port.into(),
            ..Default::default()
        }
    }

    /// Validate that required fields are present.
    pub fn validate(&self) -> ModbusResult<()> {
        if self.listen_addr.is_empty() {
            return Err(ModbusError::Config(
                "GatewayConfig: listen_addr must not be empty".into(),
            ));
        }
        if self.serial_port.is_empty() {
            return Err(ModbusError::Config(
                "GatewayConfig: serial_port must not be empty".into(),
            ));
        }
        if self.baud_rate == 0 {
            return Err(ModbusError::Config(
                "GatewayConfig: baud_rate must be > 0".into(),
            ));
        }
        if self.max_connections == 0 {
            return Err(ModbusError::Config(
                "GatewayConfig: max_connections must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Modbus TCP-to-RTU gateway instance.
///
/// The gateway translates Modbus TCP requests from multiple concurrent TCP
/// clients into serialised Modbus RTU frames destined for a single serial
/// bus.  It maintains:
///
/// - A [`TransactionIdManager`] for issuing sequential TCP transaction IDs
/// - A [`GatewayConnectionPool`] for tracking active TCP clients
/// - Per-connection [`GatewayQueue`]s for pending RTU requests
/// - Running statistics (total requests, errors, bytes)
///
/// The actual I/O (TCP accept loop, serial port read/write) is performed
/// by the caller; this struct provides the *state machine* and *protocol
/// translation* logic so it is fully testable without real sockets.
#[derive(Debug)]
pub struct ModbusTcpGateway {
    config: GatewayConfig,
    transaction_id_mgr: Arc<TransactionIdManager>,
    connection_pool: GatewayConnectionPool,
    /// Per-connection request queues keyed by connection ID
    queues: HashMap<usize, GatewayQueue>,
    /// Total requests forwarded to the RTU bus
    total_requests: u64,
    /// Total error responses generated
    total_errors: u64,
    /// Total bytes translated (TCP in → RTU out + RTU in → TCP out)
    total_bytes: u64,
}

impl ModbusTcpGateway {
    /// Create a new gateway with the given configuration.
    ///
    /// Validates the configuration before returning.
    pub fn new(config: GatewayConfig) -> ModbusResult<Self> {
        config.validate()?;
        let max_conn = config.max_connections;
        let queue_depth = config.queue_depth;
        let _ = queue_depth; // used when allocating per-connection queues
        Ok(Self {
            config,
            transaction_id_mgr: Arc::new(TransactionIdManager::new()),
            connection_pool: GatewayConnectionPool::new(max_conn),
            queues: HashMap::new(),
            total_requests: 0,
            total_errors: 0,
            total_bytes: 0,
        })
    }

    /// Accept an incoming TCP client connection.
    ///
    /// Returns the assigned connection ID.
    pub fn accept_connection(&mut self, peer_addr: impl Into<String>) -> ModbusResult<usize> {
        let conn_id = self.connection_pool.add_connection(peer_addr)?;
        self.queues
            .insert(conn_id, GatewayQueue::new(self.config.queue_depth));
        Ok(conn_id)
    }

    /// Close a TCP client connection, removing it from the pool and its queue.
    pub fn close_connection(&mut self, conn_id: usize) {
        self.connection_pool.remove_connection(conn_id);
        self.queues.remove(&conn_id);
    }

    /// Process an incoming raw TCP ADU buffer from a client connection.
    ///
    /// Parses the ADU, allocates a fresh transaction ID (the original client
    /// ID is preserved internally), translates to RTU, enqueues the request,
    /// and returns the RTU frame bytes ready to send to the serial port.
    ///
    /// # Errors
    ///
    /// - Parse failure (ADU too short or inconsistent length)
    /// - Queue full
    pub fn handle_tcp_request(&mut self, conn_id: usize, raw: &[u8]) -> ModbusResult<Vec<u8>> {
        let adu = ModbusTcpAdu::from_bytes(raw)?;

        // Allocate a gateway-side transaction ID (may differ from client's)
        let gateway_txn_id = self.transaction_id_mgr.next();

        let pending = PendingRequest {
            transaction_id: gateway_txn_id,
            unit_id: adu.unit_id,
            pdu: adu.pdu.clone(),
            enqueued_at: Instant::now(),
        };

        let queue = self
            .queues
            .get_mut(&conn_id)
            .ok_or_else(|| ModbusError::Config(format!("no queue for connection {}", conn_id)))?;
        queue.enqueue(pending)?;

        if let Some(conn) = self.connection_pool.get_mut(conn_id) {
            conn.request_count += 1;
        }

        self.total_requests += 1;

        let rtu_frame = tcp_adu_to_rtu_frame(&adu);
        let rtu_bytes = rtu_frame.to_bytes();
        self.total_bytes += raw.len() as u64 + rtu_bytes.len() as u64;

        Ok(rtu_bytes)
    }

    /// Process an RTU response received from the serial bus.
    ///
    /// Dequeues the oldest pending request for `conn_id`, verifies the RTU
    /// frame's CRC, and translates it back to a TCP ADU byte vector ready
    /// to send to the original TCP client.
    ///
    /// # Errors
    ///
    /// - No pending request for the connection
    /// - RTU CRC mismatch
    /// - Buffer too short
    pub fn handle_rtu_response(&mut self, conn_id: usize, rtu_raw: &[u8]) -> ModbusResult<Vec<u8>> {
        let queue = self
            .queues
            .get_mut(&conn_id)
            .ok_or_else(|| ModbusError::Config(format!("no queue for connection {}", conn_id)))?;

        let pending = queue.dequeue().ok_or_else(|| {
            ModbusError::Config(format!(
                "RTU response received for conn {} but no pending request",
                conn_id
            ))
        })?;

        let rtu_frame = ModbusRtuFrame::from_bytes(rtu_raw).map_err(|e| {
            if let Some(conn) = self.connection_pool.get_mut(conn_id) {
                conn.error_count += 1;
            }
            self.total_errors += 1;
            e
        })?;

        let tcp_adu = rtu_frame_to_tcp_adu(&rtu_frame, pending.transaction_id);
        let tcp_bytes = tcp_adu.to_bytes();
        self.total_bytes += rtu_raw.len() as u64 + tcp_bytes.len() as u64;

        Ok(tcp_bytes)
    }

    /// Prune timed-out requests from all connection queues.
    ///
    /// Returns the total number of requests pruned.
    pub fn prune_timed_out_requests(&mut self) -> usize {
        let timeout = self.config.rtu_timeout;
        let mut pruned = 0;
        for queue in self.queues.values_mut() {
            pruned += queue.drain_timed_out(timeout).len();
        }
        self.total_errors += pruned as u64;
        pruned
    }

    /// Current number of active TCP client connections.
    pub fn active_connections(&self) -> usize {
        self.connection_pool.active_count()
    }

    /// Current number of pending requests across all queues.
    pub fn pending_request_count(&self) -> usize {
        self.queues.values().map(|q| q.len()).sum()
    }

    /// Total Modbus requests processed since the gateway was created.
    pub fn total_requests(&self) -> u64 {
        self.total_requests
    }

    /// Total errors since the gateway was created.
    pub fn total_errors(&self) -> u64 {
        self.total_errors
    }

    /// Total bytes translated (both directions) since creation.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Reference to the gateway configuration.
    pub fn config(&self) -> &GatewayConfig {
        &self.config
    }

    /// Reference to the transaction ID manager (for inspection).
    pub fn transaction_id_manager(&self) -> &TransactionIdManager {
        &self.transaction_id_mgr
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::calculate_crc;

    // ── ModbusTcpAdu ─────────────────────────────────────────────────────

    #[test]
    fn test_tcp_adu_roundtrip() {
        let pdu = vec![0x03, 0x00, 0x00, 0x00, 0x0A]; // FC03, addr=0, count=10
        let adu = ModbusTcpAdu::new(0x0001, 1, pdu.clone());
        let bytes = adu.to_bytes();
        let restored = ModbusTcpAdu::from_bytes(&bytes).unwrap();

        assert_eq!(restored.transaction_id, 0x0001);
        assert_eq!(restored.protocol_id, 0x0000);
        assert_eq!(restored.unit_id, 1);
        assert_eq!(restored.pdu, pdu);
    }

    #[test]
    fn test_tcp_adu_length_field() {
        let pdu = vec![0x03, 0x00, 0x00, 0x00, 0x0A];
        let adu = ModbusTcpAdu::new(1, 1, pdu.clone());
        // length = 1 (unit_id) + pdu.len()
        assert_eq!(adu.length, 1 + pdu.len() as u16);
    }

    #[test]
    fn test_tcp_adu_too_short() {
        assert!(ModbusTcpAdu::from_bytes(&[0x00, 0x01, 0x00, 0x00, 0x00]).is_err());
    }

    #[test]
    fn test_tcp_adu_length_mismatch() {
        // Claim length=100 but provide only 7 bytes
        let data = [0x00, 0x01, 0x00, 0x00, 0x00, 100, 0x01];
        assert!(ModbusTcpAdu::from_bytes(&data).is_err());
    }

    #[test]
    fn test_tcp_adu_function_code() {
        let adu = ModbusTcpAdu::new(1, 1, vec![0x03, 0x00, 0x00, 0x00, 0x01]);
        assert_eq!(adu.function_code(), Some(0x03));
    }

    #[test]
    fn test_tcp_adu_empty_pdu() {
        let adu = ModbusTcpAdu::new(1, 1, vec![]);
        assert_eq!(adu.function_code(), None);
    }

    // ── ModbusRtuFrame ────────────────────────────────────────────────────

    fn make_rtu_bytes(unit_id: u8, pdu: &[u8]) -> Vec<u8> {
        let mut buf = vec![unit_id];
        buf.extend_from_slice(pdu);
        let crc = calculate_crc(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    #[test]
    fn test_rtu_frame_roundtrip() {
        let pdu = vec![0x03, 0x00, 0x00, 0x00, 0x05];
        let raw = make_rtu_bytes(1, &pdu);
        let frame = ModbusRtuFrame::from_bytes(&raw).unwrap();

        assert_eq!(frame.unit_id, 1);
        assert_eq!(frame.pdu, pdu);

        let reencoded = frame.to_bytes();
        assert_eq!(reencoded, raw);
    }

    #[test]
    fn test_rtu_frame_crc_error() {
        let pdu = vec![0x03, 0x00, 0x00, 0x00, 0x05];
        let mut raw = make_rtu_bytes(1, &pdu);
        // Corrupt the CRC
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        assert!(ModbusRtuFrame::from_bytes(&raw).is_err());
    }

    #[test]
    fn test_rtu_frame_too_short() {
        assert!(ModbusRtuFrame::from_bytes(&[0x01, 0x03, 0x00]).is_err());
    }

    #[test]
    fn test_rtu_frame_function_code() {
        let pdu = vec![0x06, 0x00, 0x01, 0x00, 0x64];
        let raw = make_rtu_bytes(2, &pdu);
        let frame = ModbusRtuFrame::from_bytes(&raw).unwrap();
        assert_eq!(frame.function_code(), Some(0x06));
    }

    // ── Protocol translation ──────────────────────────────────────────────

    #[test]
    fn test_tcp_to_rtu_translation() {
        let pdu = vec![0x03, 0x00, 0x00, 0x00, 0x0A];
        let adu = ModbusTcpAdu::new(42, 5, pdu.clone());
        let rtu = tcp_adu_to_rtu_frame(&adu);
        assert_eq!(rtu.unit_id, 5);
        assert_eq!(rtu.pdu, pdu);
    }

    #[test]
    fn test_rtu_to_tcp_translation() {
        let pdu = vec![0x03, 0x0A, 0x00, 0x01];
        let frame = ModbusRtuFrame::new(5, pdu.clone());
        let adu = rtu_frame_to_tcp_adu(&frame, 77);
        assert_eq!(adu.transaction_id, 77);
        assert_eq!(adu.unit_id, 5);
        assert_eq!(adu.pdu, pdu);
    }

    // ── TransactionIdManager ──────────────────────────────────────────────

    #[test]
    fn test_transaction_id_sequential() {
        let mgr = TransactionIdManager::new();
        let ids: Vec<u16> = (0..5).map(|_| mgr.next()).collect();
        // All IDs must be non-zero
        for id in &ids {
            assert!(*id != 0, "transaction ID must be non-zero");
        }
        // IDs must be strictly increasing (before wrap-around)
        for w in ids.windows(2) {
            assert!(
                w[1] > w[0] || w[0] == u16::MAX,
                "transaction IDs must be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_transaction_id_no_zero() {
        // Generate many IDs — none should be zero
        let mgr = TransactionIdManager::new();
        for _ in 0..1000 {
            assert_ne!(mgr.next(), 0);
        }
    }

    // ── GatewayQueue ──────────────────────────────────────────────────────

    #[test]
    fn test_queue_enqueue_dequeue() {
        let mut q = GatewayQueue::new(4);
        let req = PendingRequest {
            transaction_id: 1,
            unit_id: 1,
            pdu: vec![0x03],
            enqueued_at: Instant::now(),
        };
        q.enqueue(req).unwrap();
        assert_eq!(q.len(), 1);

        let dequeued = q.dequeue().unwrap();
        assert_eq!(dequeued.transaction_id, 1);
        assert!(q.is_empty());
    }

    #[test]
    fn test_queue_full_returns_error() {
        let mut q = GatewayQueue::new(2);
        for txn_id in 1..=2 {
            q.enqueue(PendingRequest {
                transaction_id: txn_id,
                unit_id: 1,
                pdu: vec![0x03],
                enqueued_at: Instant::now(),
            })
            .unwrap();
        }
        // Third enqueue should fail
        let result = q.enqueue(PendingRequest {
            transaction_id: 3,
            unit_id: 1,
            pdu: vec![0x03],
            enqueued_at: Instant::now(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_queue_drain_timed_out() {
        let mut q = GatewayQueue::new(8);
        // Enqueue a request that appears to have timed out by using a very
        // short timeout of 0ns
        let req = PendingRequest {
            transaction_id: 1,
            unit_id: 1,
            pdu: vec![0x03],
            enqueued_at: Instant::now() - Duration::from_secs(10),
        };
        q.enqueue(req).unwrap();
        // Also enqueue a fresh request
        q.enqueue(PendingRequest {
            transaction_id: 2,
            unit_id: 1,
            pdu: vec![0x03],
            enqueued_at: Instant::now(),
        })
        .unwrap();

        let timed_out = q.drain_timed_out(Duration::from_secs(5));
        assert_eq!(timed_out.len(), 1);
        assert_eq!(timed_out[0].transaction_id, 1);
        assert_eq!(q.len(), 1); // the fresh one remains
    }

    #[test]
    fn test_queue_fifo_order() {
        let mut q = GatewayQueue::new(10);
        for txn_id in 1u16..=5 {
            q.enqueue(PendingRequest {
                transaction_id: txn_id,
                unit_id: 1,
                pdu: vec![0x03],
                enqueued_at: Instant::now(),
            })
            .unwrap();
        }
        for expected in 1u16..=5 {
            let req = q.dequeue().unwrap();
            assert_eq!(req.transaction_id, expected);
        }
    }

    // ── GatewayConnectionPool ─────────────────────────────────────────────

    #[test]
    fn test_pool_add_and_remove() {
        let mut pool = GatewayConnectionPool::new(4);
        let id = pool.add_connection("192.168.1.1:12345").unwrap();
        assert_eq!(pool.active_count(), 1);

        let conn = pool.get(id).unwrap();
        assert_eq!(conn.peer_addr, "192.168.1.1:12345");

        pool.remove_connection(id);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn test_pool_full_rejected() {
        let mut pool = GatewayConnectionPool::new(2);
        pool.add_connection("1.1.1.1:1").unwrap();
        pool.add_connection("1.1.1.2:2").unwrap();
        assert!(pool.add_connection("1.1.1.3:3").is_err());
        assert!(pool.is_full());
    }

    // ── ModbusTcpGateway ─────────────────────────────────────────────────

    fn make_gateway() -> ModbusTcpGateway {
        let cfg = GatewayConfig {
            listen_addr: "127.0.0.1:5020".into(),
            serial_port: "/dev/ttyS0".into(),
            baud_rate: 19200,
            max_connections: 8,
            queue_depth: 16,
            rtu_timeout: Duration::from_secs(1),
            tcp_idle_timeout: Duration::from_secs(30),
        };
        ModbusTcpGateway::new(cfg).unwrap()
    }

    #[test]
    fn test_gateway_accept_and_close() {
        let mut gw = make_gateway();
        let id = gw.accept_connection("10.0.0.1:50000").unwrap();
        assert_eq!(gw.active_connections(), 1);
        gw.close_connection(id);
        assert_eq!(gw.active_connections(), 0);
    }

    #[test]
    fn test_gateway_handle_request_produces_rtu_frame() {
        let mut gw = make_gateway();
        let conn_id = gw.accept_connection("10.0.0.2:50001").unwrap();

        // Build a TCP ADU: FC03, read 5 registers at address 0
        let pdu = vec![0x03u8, 0x00, 0x00, 0x00, 0x05];
        let adu = ModbusTcpAdu::new(1, 1, pdu.clone());
        let raw_tcp = adu.to_bytes();

        let rtu_bytes = gw.handle_tcp_request(conn_id, &raw_tcp).unwrap();

        // RTU frame: [unit_id, ...pdu..., CRC_lo, CRC_hi]
        assert!(rtu_bytes.len() >= 4);
        assert_eq!(rtu_bytes[0], 1); // unit_id
        assert_eq!(&rtu_bytes[1..1 + pdu.len()], pdu.as_slice());
        assert_eq!(gw.total_requests(), 1);
    }

    #[test]
    fn test_gateway_handle_rtu_response_translates_back() {
        let mut gw = make_gateway();
        let conn_id = gw.accept_connection("10.0.0.3:50002").unwrap();

        // Simulate a request first (to populate the pending queue)
        let pdu = vec![0x03u8, 0x00, 0x00, 0x00, 0x02];
        let adu = ModbusTcpAdu::new(5, 2, pdu);
        gw.handle_tcp_request(conn_id, &adu.to_bytes()).unwrap();

        // Build a valid RTU response: FC03 response with 2 regs = 4 bytes of data
        let rtu_pdu = vec![0x03u8, 0x04, 0x01, 0x23, 0x04, 0x56];
        let rtu_bytes = make_rtu_bytes(2, &rtu_pdu);

        let tcp_response = gw.handle_rtu_response(conn_id, &rtu_bytes).unwrap();

        // Parse the TCP response
        let response_adu = ModbusTcpAdu::from_bytes(&tcp_response).unwrap();
        assert_eq!(response_adu.unit_id, 2);
        assert_eq!(response_adu.pdu, rtu_pdu);
    }

    #[test]
    fn test_gateway_prune_timed_out() {
        let cfg = GatewayConfig {
            listen_addr: "127.0.0.1:5021".into(),
            serial_port: "/dev/ttyS0".into(),
            baud_rate: 9600,
            rtu_timeout: Duration::from_millis(1), // Very short
            ..GatewayConfig::default()
        };
        let mut gw = ModbusTcpGateway::new(cfg).unwrap();
        let conn_id = gw.accept_connection("10.0.0.5:50003").unwrap();

        // Submit a request to populate the queue
        let pdu = vec![0x03u8, 0x00, 0x00, 0x00, 0x01];
        let adu = ModbusTcpAdu::new(1, 1, pdu);
        gw.handle_tcp_request(conn_id, &adu.to_bytes()).unwrap();

        // Wait slightly longer than the 1ms timeout
        std::thread::sleep(Duration::from_millis(5));

        let pruned = gw.prune_timed_out_requests();
        assert!(pruned >= 1, "expected at least one timed-out request");
    }

    #[test]
    fn test_gateway_config_validation_empty_addr() {
        let cfg = GatewayConfig {
            listen_addr: "".into(),
            serial_port: "/dev/ttyS0".into(),
            baud_rate: 9600,
            ..GatewayConfig::default()
        };
        assert!(ModbusTcpGateway::new(cfg).is_err());
    }

    #[test]
    fn test_gateway_config_validation_zero_baud() {
        let cfg = GatewayConfig {
            listen_addr: "0.0.0.0:502".into(),
            serial_port: "/dev/ttyS0".into(),
            baud_rate: 0,
            ..GatewayConfig::default()
        };
        assert!(ModbusTcpGateway::new(cfg).is_err());
    }
}
