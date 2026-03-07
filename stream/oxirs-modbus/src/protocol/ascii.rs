//! Modbus ASCII protocol implementation
//!
//! Modbus ASCII uses a human-readable framing over serial lines.
//! Each frame is ASCII-encoded with the format:
//!
//! ```text
//! :LLAAFFFCR\n
//! ```
//!
//! Where:
//! - `:` - Start character (colon, 0x3A)
//! - `LL` - Device address (hex-encoded, 2 chars)
//! - `AA` - Function code (hex-encoded, 2 chars)
//! - `FF...` - Data bytes (hex-encoded, variable length)
//! - `C` - LRC checksum (hex-encoded, 2 chars)
//! - `CR\n` - End sequence (0x0D 0x0A)
//!
//! The LRC is the two's complement of the sum of all binary values in the message,
//! excluding the colon, CR, and LF characters, and the LRC itself.

use crate::error::{ModbusError, ModbusResult};
use crate::protocol::frame::FunctionCode;
use std::fmt;
use std::io::{Read, Write};
use std::time::Duration;

/// ASCII frame start character
const ASCII_START: u8 = b':';

/// ASCII frame end: CR (0x0D) followed by LF (0x0A)
const ASCII_CR: u8 = 0x0D;
const ASCII_LF: u8 = 0x0A;

/// Maximum number of data bytes in an ASCII frame (255 bytes binary = 510 hex chars)
const MAX_DATA_BYTES: usize = 252;

/// Modbus ASCII frame structure
///
/// Represents a complete Modbus ASCII Application Data Unit (ADU).
///
/// # Format
///
/// The on-wire format is:
/// ```text
/// : AA FF [DD...] CC CR LF
/// ```
/// where each byte is represented as two uppercase hex digits.
///
/// # Examples
///
/// ```rust
/// use oxirs_modbus::protocol::ascii::{AsciiFrame, encode_ascii, decode_ascii};
///
/// let frame = AsciiFrame::new(0x01, 0x03, vec![0x00, 0x00, 0x00, 0x0A]);
/// let encoded = encode_ascii(&frame);
/// let decoded = decode_ascii(&encoded).unwrap();
/// assert_eq!(decoded.device_addr, 0x01);
/// assert_eq!(decoded.function_code, 0x03);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsciiFrame {
    /// Start delimiter (always ':')
    pub start: u8,

    /// Device address (slave address, 1–247)
    pub device_addr: u8,

    /// Function code (matches Modbus function codes)
    pub function_code: u8,

    /// Data payload (raw binary, before ASCII encoding)
    pub data: Vec<u8>,

    /// LRC checksum (two's complement of sum of addr + func + data)
    pub lrc: u8,

    /// End delimiter (CR LF = [0x0D, 0x0A])
    pub end: [u8; 2],
}

impl AsciiFrame {
    /// Create a new ASCII frame, computing the LRC automatically.
    ///
    /// # Arguments
    ///
    /// * `device_addr` - Slave address (1–247)
    /// * `function_code` - Modbus function code byte
    /// * `data` - PDU data payload
    ///
    /// # Returns
    ///
    /// A new `AsciiFrame` with the LRC field computed.
    pub fn new(device_addr: u8, function_code: u8, data: Vec<u8>) -> Self {
        let mut lrc_bytes = vec![device_addr, function_code];
        lrc_bytes.extend_from_slice(&data);
        let lrc = compute_lrc(&lrc_bytes);

        Self {
            start: ASCII_START,
            device_addr,
            function_code,
            data,
            lrc,
            end: [ASCII_CR, ASCII_LF],
        }
    }

    /// Create from a known FunctionCode enum variant.
    pub fn from_function(device_addr: u8, fc: FunctionCode, data: Vec<u8>) -> Self {
        Self::new(device_addr, fc.as_u8(), data)
    }

    /// Verify that the stored LRC matches the computed LRC.
    pub fn verify_lrc(&self) -> bool {
        let mut lrc_bytes = vec![self.device_addr, self.function_code];
        lrc_bytes.extend_from_slice(&self.data);
        compute_lrc(&lrc_bytes) == self.lrc
    }

    /// Total number of binary bytes that will be encoded (addr + func + data + lrc).
    pub fn payload_len(&self) -> usize {
        2 + self.data.len() + 1
    }

    /// Try to parse the function code as a typed `FunctionCode`.
    pub fn typed_function_code(&self) -> ModbusResult<FunctionCode> {
        FunctionCode::from_u8(self.function_code)
    }
}

impl fmt::Display for AsciiFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            ":{}{}",
            encode_hex_byte(self.device_addr),
            encode_hex_byte(self.function_code)
        )?;
        for &b in &self.data {
            write!(f, "{}", encode_hex_byte(b))?;
        }
        write!(f, "{}CRLF", encode_hex_byte(self.lrc))
    }
}

// ---------------------------------------------------------------------------
// LRC computation
// ---------------------------------------------------------------------------

/// Compute the Modbus ASCII LRC (Longitudinal Redundancy Check).
///
/// The LRC is computed as the two's complement of the arithmetic sum
/// (modulo 256) of all bytes in the message (address + function + data).
/// It does NOT include the start colon, end CR, or end LF.
///
/// # Arguments
///
/// * `bytes` - The raw binary bytes to checksum (addr, func, data concatenated)
///
/// # Returns
///
/// The LRC byte.
///
/// # Examples
///
/// ```rust
/// use oxirs_modbus::protocol::ascii::compute_lrc;
///
/// // For a Read Holding Registers request: addr=0x01, func=0x03, data=[0x00,0x00,0x00,0x0A]
/// let lrc = compute_lrc(&[0x01, 0x03, 0x00, 0x00, 0x00, 0x0A]);
/// // sum = 0x01 + 0x03 + 0x00 + 0x00 + 0x00 + 0x0A = 0x0E
/// // two's complement = 0x100 - 0x0E = 0xF2
/// assert_eq!(lrc, 0xF2);
/// ```
pub fn compute_lrc(bytes: &[u8]) -> u8 {
    let sum: u8 = bytes.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
    (!sum).wrapping_add(1)
}

// ---------------------------------------------------------------------------
// Hex encoding helpers
// ---------------------------------------------------------------------------

/// Encode a single byte as two uppercase hex ASCII characters.
fn encode_hex_byte(byte: u8) -> String {
    format!("{:02X}", byte)
}

/// Decode two ASCII hex characters into a byte.
///
/// # Errors
///
/// Returns `ModbusError::Io` if either character is not a valid hex digit.
fn decode_hex_byte(hi: u8, lo: u8) -> ModbusResult<u8> {
    let high = hex_nibble(hi).map_err(|_| {
        ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid hex digit: 0x{:02X}", hi),
        ))
    })?;
    let low = hex_nibble(lo).map_err(|_| {
        ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid hex digit: 0x{:02X}", lo),
        ))
    })?;
    Ok((high << 4) | low)
}

/// Convert a single ASCII hex character to its nibble value.
fn hex_nibble(c: u8) -> Result<u8, ()> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        _ => Err(()),
    }
}

// ---------------------------------------------------------------------------
// Encode / Decode
// ---------------------------------------------------------------------------

/// Encode an `AsciiFrame` into its on-wire ASCII byte representation.
///
/// The output has the form:
/// ```text
/// :AAFFDD...CCCRLF
/// ```
/// where each byte is two uppercase hex digits.
///
/// # Arguments
///
/// * `frame` - The frame to encode
///
/// # Returns
///
/// A `Vec<u8>` containing the complete on-wire representation including `:` and CRLF.
///
/// # Examples
///
/// ```rust
/// use oxirs_modbus::protocol::ascii::{AsciiFrame, encode_ascii};
///
/// let frame = AsciiFrame::new(0x01, 0x03, vec![0x00, 0x00, 0x00, 0x0A]);
/// let bytes = encode_ascii(&frame);
/// assert_eq!(bytes[0], b':');
/// assert_eq!(*bytes.last().unwrap(), b'\n');
/// ```
pub fn encode_ascii(frame: &AsciiFrame) -> Vec<u8> {
    // Pre-allocate: 1 (colon) + 2 (addr) + 2 (func) + 2*data.len() + 2 (lrc) + 2 (CRLF)
    let capacity = 1 + 2 + 2 + 2 * frame.data.len() + 2 + 2;
    let mut out = Vec::with_capacity(capacity);

    // Start delimiter
    out.push(ASCII_START);

    // Device address (2 hex digits)
    push_hex_byte(&mut out, frame.device_addr);

    // Function code (2 hex digits)
    push_hex_byte(&mut out, frame.function_code);

    // Data bytes (2 hex digits each)
    for &b in &frame.data {
        push_hex_byte(&mut out, b);
    }

    // LRC (2 hex digits)
    push_hex_byte(&mut out, frame.lrc);

    // CRLF end delimiter
    out.push(ASCII_CR);
    out.push(ASCII_LF);

    out
}

/// Push two uppercase hex ASCII bytes for `byte` into `buf`.
fn push_hex_byte(buf: &mut Vec<u8>, byte: u8) {
    let hi = b"0123456789ABCDEF"[(byte >> 4) as usize];
    let lo = b"0123456789ABCDEF"[(byte & 0x0F) as usize];
    buf.push(hi);
    buf.push(lo);
}

/// Decode a Modbus ASCII frame from its on-wire byte representation.
///
/// # Expected format
///
/// ```text
/// :AAFFDD...CCRLF
/// ```
///
/// - Must start with `:` (0x3A)
/// - Must end with CR LF (0x0D 0x0A)
/// - All intermediate bytes must be valid uppercase or lowercase hex digits
/// - The LRC in the frame must match the computed LRC of (addr + func + data)
///
/// # Errors
///
/// | Condition | Error |
/// |-----------|-------|
/// | Does not start with `:` | `ModbusError::Io(InvalidData)` |
/// | Does not end with CRLF | `ModbusError::Io(UnexpectedEof)` |
/// | Too short (< 9 bytes: `:` + 2+2+2 + CRLF) | `ModbusError::Io(UnexpectedEof)` |
/// | Odd number of data nibbles | `ModbusError::Io(InvalidData)` |
/// | Invalid hex characters | `ModbusError::Io(InvalidData)` |
/// | LRC mismatch | `ModbusError::CrcError` |
///
/// # Examples
///
/// ```rust
/// use oxirs_modbus::protocol::ascii::{AsciiFrame, encode_ascii, decode_ascii};
///
/// let frame = AsciiFrame::new(0x01, 0x03, vec![0x00, 0x00, 0x00, 0x0A]);
/// let encoded = encode_ascii(&frame);
/// let decoded = decode_ascii(&encoded).unwrap();
/// assert_eq!(decoded, frame);
/// ```
pub fn decode_ascii(bytes: &[u8]) -> ModbusResult<AsciiFrame> {
    // Minimum: ':' + AA + FF + CC + CR + LF = 1 + 2 + 2 + 2 + 1 + 1 = 9 bytes
    if bytes.len() < 9 {
        return Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("Frame too short: {} bytes (minimum 9)", bytes.len()),
        )));
    }

    // Verify start character
    if bytes[0] != ASCII_START {
        return Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Invalid start byte: expected ':' (0x3A), got 0x{:02X}",
                bytes[0]
            ),
        )));
    }

    // Verify end sequence
    let len = bytes.len();
    if bytes[len - 2] != ASCII_CR || bytes[len - 1] != ASCII_LF {
        return Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "Invalid end sequence: expected CRLF (0x0D 0x0A), got 0x{:02X} 0x{:02X}",
                bytes[len - 2],
                bytes[len - 1]
            ),
        )));
    }

    // The hex payload is between ':' and CRLF
    let hex_payload = &bytes[1..len - 2];

    // Must have an even number of hex digits
    if hex_payload.len() % 2 != 0 {
        return Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Odd number of hex digits in payload: {}", hex_payload.len()),
        )));
    }

    // Must have at least addr(1) + func(1) + lrc(1) = 3 decoded bytes => 6 hex chars
    if hex_payload.len() < 6 {
        return Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Payload too short: need at least address, function code, and LRC",
        )));
    }

    // Decode all hex pairs into binary
    let decoded_len = hex_payload.len() / 2;
    let mut decoded = Vec::with_capacity(decoded_len);
    for i in 0..decoded_len {
        let hi = hex_payload[2 * i];
        let lo = hex_payload[2 * i + 1];
        decoded.push(decode_hex_byte(hi, lo)?);
    }

    // Check max data length
    let data_len = decoded_len - 3; // subtract addr, func, lrc
    if data_len > MAX_DATA_BYTES {
        return Err(ModbusError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Data payload too large: {} bytes (maximum {})",
                data_len, MAX_DATA_BYTES
            ),
        )));
    }

    let device_addr = decoded[0];
    let function_code = decoded[1];
    let data = decoded[2..decoded_len - 1].to_vec();
    let received_lrc = decoded[decoded_len - 1];

    // Verify LRC
    let mut lrc_input = vec![device_addr, function_code];
    lrc_input.extend_from_slice(&data);
    let computed_lrc = compute_lrc(&lrc_input);

    if computed_lrc != received_lrc {
        return Err(ModbusError::CrcError {
            expected: u16::from(computed_lrc),
            actual: u16::from(received_lrc),
        });
    }

    Ok(AsciiFrame {
        start: ASCII_START,
        device_addr,
        function_code,
        data,
        lrc: received_lrc,
        end: [ASCII_CR, ASCII_LF],
    })
}

// ---------------------------------------------------------------------------
// ASCII Codec
// ---------------------------------------------------------------------------

/// Stateful encoder/decoder for Modbus ASCII frames.
///
/// The codec accumulates bytes and extracts complete frames, handling
/// the boundary detection (`:` start, CRLF end) for streaming use.
pub struct AsciiCodec {
    /// Internal buffer for partial frame accumulation
    buffer: Vec<u8>,

    /// Maximum allowed buffer size before clearing (prevents memory exhaustion)
    max_buffer: usize,
}

impl AsciiCodec {
    /// Create a new `AsciiCodec` with default buffer limit (1 KB).
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(64),
            max_buffer: 1024,
        }
    }

    /// Create a new `AsciiCodec` with a custom buffer limit.
    pub fn with_max_buffer(max_buffer: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(64),
            max_buffer,
        }
    }

    /// Feed raw bytes into the codec buffer.
    ///
    /// Returns a complete frame if one is found, or `None` if more data
    /// is needed. Malformed partial frames are discarded when a new start
    /// delimiter is encountered.
    pub fn feed(&mut self, data: &[u8]) -> ModbusResult<Option<AsciiFrame>> {
        for &byte in data {
            // New start delimiter: discard any partial frame
            if byte == ASCII_START {
                self.buffer.clear();
            }

            self.buffer.push(byte);

            // Overflow protection
            if self.buffer.len() > self.max_buffer {
                self.buffer.clear();
                return Err(ModbusError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "ASCII frame buffer overflow: frame exceeds maximum size",
                )));
            }

            // Check for CRLF terminator
            let buf_len = self.buffer.len();
            if buf_len >= 2
                && self.buffer[buf_len - 2] == ASCII_CR
                && self.buffer[buf_len - 1] == ASCII_LF
            {
                let frame_bytes = self.buffer.clone();
                self.buffer.clear();
                return decode_ascii(&frame_bytes).map(Some);
            }
        }

        Ok(None)
    }

    /// Encode an `AsciiFrame` to bytes.
    pub fn encode(&self, frame: &AsciiFrame) -> Vec<u8> {
        encode_ascii(frame)
    }

    /// Clear the internal buffer, discarding any partial frame.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

impl Default for AsciiCodec {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ASCII Transport
// ---------------------------------------------------------------------------

/// Serial line transport configuration for Modbus ASCII mode.
///
/// Wraps any `Read + Write` stream (typically a serial port) and provides
/// send/receive operations at the Modbus ASCII frame level.
///
/// # Design Notes
///
/// Unlike the async RTU client, `AsciiTransport` is synchronous and
/// generic over the stream type, making it usable with `tokio_serial`,
/// `serialport`, or any mock stream in tests.
pub struct AsciiTransport<S: Read + Write> {
    /// Underlying I/O stream
    stream: S,

    /// Receive-side codec
    codec: AsciiCodec,

    /// Inter-character timeout (used when configuring underlying serial port)
    timeout: Duration,

    /// Unit ID for this transport (used as default device address)
    unit_id: u8,
}

impl<S: Read + Write> AsciiTransport<S> {
    /// Create a new `AsciiTransport` over the given stream.
    ///
    /// # Arguments
    ///
    /// * `stream` - The underlying I/O stream
    /// * `unit_id` - Default Modbus unit/slave ID
    pub fn new(stream: S, unit_id: u8) -> Self {
        Self {
            stream,
            codec: AsciiCodec::new(),
            timeout: Duration::from_secs(1),
            unit_id,
        }
    }

    /// Set the I/O timeout.
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Get the current unit ID.
    pub fn unit_id(&self) -> u8 {
        self.unit_id
    }

    /// Set the unit ID.
    pub fn set_unit_id(&mut self, unit_id: u8) {
        self.unit_id = unit_id;
    }

    /// Send a frame over the serial line.
    ///
    /// # Errors
    ///
    /// Returns `ModbusError::Io` if the underlying write fails.
    pub fn send(&mut self, frame: &AsciiFrame) -> ModbusResult<()> {
        let bytes = encode_ascii(frame);
        self.stream.write_all(&bytes).map_err(ModbusError::Io)?;
        self.stream.flush().map_err(ModbusError::Io)?;
        Ok(())
    }

    /// Receive a frame from the serial line.
    ///
    /// Reads bytes one at a time and feeds them to the codec until a
    /// complete frame is assembled or an error occurs.
    ///
    /// # Errors
    ///
    /// Returns `ModbusError::Io` on read failure, or frame-level errors
    /// from the codec (bad LRC, bad hex, etc.).
    pub fn receive(&mut self) -> ModbusResult<AsciiFrame> {
        let mut byte = [0u8; 1];
        loop {
            self.stream.read_exact(&mut byte).map_err(ModbusError::Io)?;
            if let Some(frame) = self.codec.feed(&byte)? {
                return Ok(frame);
            }
        }
    }

    /// Send a request and receive the corresponding response.
    ///
    /// Builds a frame from the given address, function code, and data,
    /// sends it, then waits for and returns the response frame.
    pub fn request(
        &mut self,
        function_code: FunctionCode,
        data: Vec<u8>,
    ) -> ModbusResult<AsciiFrame> {
        let frame = AsciiFrame::from_function(self.unit_id, function_code, data);
        self.send(&frame)?;
        self.receive()
    }

    /// Send a Read Holding Registers request and return raw response data.
    pub fn read_holding_registers(
        &mut self,
        start_addr: u16,
        count: u16,
    ) -> ModbusResult<AsciiFrame> {
        let data = vec![
            (start_addr >> 8) as u8,
            (start_addr & 0xFF) as u8,
            (count >> 8) as u8,
            (count & 0xFF) as u8,
        ];
        self.request(FunctionCode::ReadHoldingRegisters, data)
    }

    /// Send a Write Single Register request.
    pub fn write_single_register(&mut self, addr: u16, value: u16) -> ModbusResult<AsciiFrame> {
        let data = vec![
            (addr >> 8) as u8,
            (addr & 0xFF) as u8,
            (value >> 8) as u8,
            (value & 0xFF) as u8,
        ];
        self.request(FunctionCode::WriteSingleRegister, data)
    }

    /// Consume this transport, returning the underlying stream.
    pub fn into_inner(self) -> S {
        self.stream
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ------------------------------------------------------------------
    // LRC computation
    // ------------------------------------------------------------------

    /// Verify the well-known example from the Modbus specification:
    /// addr=0x01, func=0x03, data=[0x00, 0x00, 0x00, 0x0A]
    /// sum = 0x0E → LRC = 0xF2
    #[test]
    fn test_compute_lrc_known_value() {
        let bytes = [0x01u8, 0x03, 0x00, 0x00, 0x00, 0x0A];
        let lrc = compute_lrc(&bytes);
        assert_eq!(lrc, 0xF2, "Expected 0xF2, got 0x{:02X}", lrc);
    }

    #[test]
    fn test_compute_lrc_zero_sum() {
        // sum = 0 → LRC = two's complement of 0 = 0
        let bytes = [0x00u8, 0x00];
        assert_eq!(compute_lrc(&bytes), 0x00);
    }

    #[test]
    fn test_compute_lrc_single_byte() {
        // sum = 0xFF → two's complement = (!0xFF).wrapping_add(1) = 0x00.wrapping_add(1) = 0x01
        let bytes = [0xFFu8];
        assert_eq!(compute_lrc(&bytes), 0x01);
    }

    #[test]
    fn test_compute_lrc_wrap() {
        // sum wraps around: 0xFF + 0x01 = 0x00 → LRC = 0x00
        let bytes = [0xFFu8, 0x01];
        assert_eq!(compute_lrc(&bytes), 0x00);
    }

    #[test]
    fn test_compute_lrc_empty() {
        // Empty input: sum = 0 → LRC = 0
        assert_eq!(compute_lrc(&[]), 0x00);
    }

    // ------------------------------------------------------------------
    // Frame construction
    // ------------------------------------------------------------------

    #[test]
    fn test_frame_new_computes_lrc() {
        let frame = AsciiFrame::new(0x01, 0x03, vec![0x00, 0x00, 0x00, 0x0A]);
        assert_eq!(frame.device_addr, 0x01);
        assert_eq!(frame.function_code, 0x03);
        assert_eq!(frame.lrc, 0xF2);
        assert!(frame.verify_lrc());
    }

    #[test]
    fn test_frame_from_function_code() {
        let frame = AsciiFrame::from_function(
            0x01,
            FunctionCode::ReadHoldingRegisters,
            vec![0x00, 0x6B, 0x00, 0x03],
        );
        assert_eq!(frame.function_code, 0x03);
        assert!(frame.verify_lrc());
    }

    #[test]
    fn test_frame_start_end_delimiters() {
        let frame = AsciiFrame::new(0x01, 0x03, vec![]);
        assert_eq!(frame.start, b':');
        assert_eq!(frame.end, [0x0D, 0x0A]);
    }

    // ------------------------------------------------------------------
    // Encode
    // ------------------------------------------------------------------

    #[test]
    fn test_encode_ascii_format() {
        let frame = AsciiFrame::new(0x01, 0x03, vec![0x00, 0x00, 0x00, 0x0A]);
        let encoded = encode_ascii(&frame);

        // Must start with ':'
        assert_eq!(encoded[0], b':');

        // Must end with CRLF
        let last = encoded.len();
        assert_eq!(encoded[last - 2], 0x0D);
        assert_eq!(encoded[last - 1], 0x0A);

        // Known good frame from the Modbus ASCII spec
        // :01030000000AF2CRLF
        let expected = b":01030000000AF2\r\n";
        assert_eq!(&encoded, expected);
    }

    #[test]
    fn test_encode_ascii_uppercase_hex() {
        // All hex digits must be uppercase
        let frame = AsciiFrame::new(0xAB, 0xCD, vec![0xEF]);
        let encoded = encode_ascii(&frame);

        // The payload portion (excluding ':' and CRLF)
        let payload = &encoded[1..encoded.len() - 2];
        for &byte in payload {
            if byte.is_ascii_alphabetic() {
                assert!(
                    byte.is_ascii_uppercase(),
                    "Byte 0x{:02X} is not uppercase",
                    byte
                );
            }
        }
    }

    #[test]
    fn test_encode_empty_data() {
        let frame = AsciiFrame::new(0x01, 0x03, vec![]);
        let encoded = encode_ascii(&frame);
        // ':' + 'AA' + 'FF' + 'CC' + CRLF = 9 bytes
        assert_eq!(encoded.len(), 9);
    }

    // ------------------------------------------------------------------
    // Decode
    // ------------------------------------------------------------------

    #[test]
    fn test_decode_ascii_round_trip() {
        let original = AsciiFrame::new(0x01, 0x03, vec![0x00, 0x00, 0x00, 0x0A]);
        let encoded = encode_ascii(&original);
        let decoded = decode_ascii(&encoded).expect("decode should succeed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_decode_ascii_known_frame() {
        // addr=0x01, func=0x03, data=[0x00,0x6B,0x00,0x03]
        // sum = 0x01+0x03+0x00+0x6B+0x00+0x03 = 0x72
        // LRC = two's complement of 0x72 = (!0x72).wrapping_add(1) = 0x8D+1 = 0x8E
        let frame_bytes = b":0103006B00038E\r\n";
        let frame = decode_ascii(frame_bytes).expect("decode known frame");
        assert_eq!(frame.device_addr, 0x01);
        assert_eq!(frame.function_code, 0x03);
        assert_eq!(frame.data, vec![0x00, 0x6B, 0x00, 0x03]);
        assert_eq!(frame.lrc, 0x8E);
    }

    #[test]
    fn test_decode_ascii_lowercase_hex() {
        // Lowercase hex should also be accepted
        let frame_bytes = b":01030000000af2\r\n";
        let frame = decode_ascii(frame_bytes).expect("decode lowercase hex");
        assert_eq!(frame.device_addr, 0x01);
        assert_eq!(frame.function_code, 0x03);
        assert_eq!(frame.lrc, 0xF2);
    }

    // ------------------------------------------------------------------
    // Error cases
    // ------------------------------------------------------------------

    #[test]
    fn test_decode_error_bad_start_byte() {
        // Frame starting with 'X' instead of ':'
        let bad = b"X01030000000AF2\r\n";
        let result = decode_ascii(bad);
        assert!(result.is_err());
        match result {
            Err(ModbusError::Io(e)) => {
                assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
            }
            _ => panic!("Expected Io(InvalidData) error"),
        }
    }

    #[test]
    fn test_decode_error_missing_crlf() {
        // No CRLF at end
        let bad = b":01030000000AF2";
        let result = decode_ascii(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_error_truncated() {
        // Too short
        let bad = b":0103\r\n";
        let result = decode_ascii(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_error_bad_lrc() {
        // Corrupt the LRC byte: change 0xF2 → 0xFF
        let bad = b":01030000000AFF\r\n";
        let result = decode_ascii(bad);
        assert!(result.is_err());
        match result {
            Err(ModbusError::CrcError { expected, actual }) => {
                assert_eq!(expected, 0xF2);
                assert_ne!(actual, expected);
            }
            _ => panic!("Expected CrcError"),
        }
    }

    #[test]
    fn test_decode_error_invalid_hex() {
        // 'GG' is not valid hex
        let bad = b":GG030000000AXX\r\n";
        let result = decode_ascii(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_error_odd_nibble_count() {
        // Odd number of hex digits between ':' and CRLF
        let _bad = b":010300000A\r\n";
        // Payload = "010300000A" = 10 hex chars = 5 bytes = odd? No, 10 is even.
        // We need an odd count, e.g., 11 chars: "01030000000"
        let bad2 = b":01030000000\r\n";
        let result = decode_ascii(bad2);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_error_too_short() {
        // Just ":CRLF" = 3 bytes
        let bad = b":\r\n";
        let result = decode_ascii(bad);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // Codec streaming
    // ------------------------------------------------------------------

    #[test]
    fn test_codec_complete_frame_single_feed() {
        let mut codec = AsciiCodec::new();
        let frame = AsciiFrame::new(0x01, 0x03, vec![0x00, 0x00, 0x00, 0x0A]);
        let encoded = encode_ascii(&frame);

        let result = codec.feed(&encoded).expect("codec feed should succeed");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), frame);
    }

    #[test]
    fn test_codec_byte_by_byte_feed() {
        let mut codec = AsciiCodec::new();
        let frame = AsciiFrame::new(0x01, 0x03, vec![0x00, 0x6B, 0x00, 0x03]);
        let encoded = encode_ascii(&frame);

        let mut result = None;
        for byte in &encoded {
            result = codec.feed(&[*byte]).expect("codec feed should succeed");
            if result.is_some() {
                break;
            }
        }
        assert!(result.is_some());
        assert_eq!(result.unwrap(), frame);
    }

    #[test]
    fn test_codec_discards_garbage_before_start() {
        let mut codec = AsciiCodec::new();
        // Feed garbage then a valid frame
        let _ = codec.feed(b"GARBAGE_DATA");
        let frame = AsciiFrame::new(0x02, 0x06, vec![0x00, 0x01, 0x00, 0x64]);
        let encoded = encode_ascii(&frame);
        let result = codec
            .feed(&encoded)
            .expect("codec should decode valid frame");
        assert!(result.is_some());
    }

    #[test]
    fn test_codec_reset_clears_buffer() {
        let mut codec = AsciiCodec::new();
        let _ = codec.feed(b":0103"); // partial
        codec.reset();
        // After reset, a complete new frame should decode correctly
        let frame = AsciiFrame::new(0x01, 0x03, vec![0x00, 0x00, 0x00, 0x0A]);
        let encoded = encode_ascii(&frame);
        let result = codec.feed(&encoded).expect("should decode after reset");
        assert!(result.is_some());
    }

    // ------------------------------------------------------------------
    // Transport (using in-memory Cursor)
    // ------------------------------------------------------------------

    #[test]
    fn test_transport_send() {
        let expected_frame = AsciiFrame::new(0x01, 0x03, vec![0x00, 0x00, 0x00, 0x0A]);
        let expected_bytes = encode_ascii(&expected_frame);

        let buffer = Vec::new();
        let cursor = Cursor::new(buffer);
        let mut transport = AsciiTransport::new(cursor, 0x01);
        transport
            .send(&expected_frame)
            .expect("send should succeed");

        let written = transport.into_inner().into_inner();
        assert_eq!(written, expected_bytes);
    }

    #[test]
    fn test_transport_receive() {
        let frame = AsciiFrame::new(0x01, 0x03, vec![0x00, 0x00, 0x00, 0x0A]);
        let encoded = encode_ascii(&frame);

        let cursor = Cursor::new(encoded);
        let mut transport = AsciiTransport::new(cursor, 0x01);
        let received = transport.receive().expect("receive should succeed");
        assert_eq!(received, frame);
    }

    #[test]
    fn test_transport_unit_id() {
        let cursor = Cursor::new(Vec::<u8>::new());
        let mut transport = AsciiTransport::new(cursor, 0x05);
        assert_eq!(transport.unit_id(), 0x05);
        transport.set_unit_id(0x0A);
        assert_eq!(transport.unit_id(), 0x0A);
    }
}
