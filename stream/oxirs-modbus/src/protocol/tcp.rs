//! Modbus TCP client implementation
//!
//! This module provides a Modbus TCP client that communicates over Ethernet
//! on port 502 (standard Modbus TCP port).

use crate::error::{ModbusError, ModbusResult};
use crate::protocol::frame::{FunctionCode, ModbusTcpFrame};
use bytes::BufMut;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

/// Modbus TCP client
///
/// # Examples
///
/// ```no_run
/// use oxirs_modbus::protocol::ModbusTcpClient;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut client = ModbusTcpClient::connect("192.168.1.100:502", 1).await?;
///     let registers = client.read_holding_registers(0, 10).await?;
///     println!("Registers: {:?}", registers);
///     Ok(())
/// }
/// ```
pub struct ModbusTcpClient {
    /// TCP stream connection
    stream: TcpStream,

    /// Unit identifier (slave address)
    unit_id: u8,

    /// Transaction ID counter (atomic for thread safety)
    transaction_id: Arc<AtomicU16>,

    /// Request timeout
    timeout: Duration,
}

impl ModbusTcpClient {
    /// Connect to a Modbus TCP server
    ///
    /// # Arguments
    ///
    /// * `addr` - Server address (e.g., "192.168.1.100:502")
    /// * `unit_id` - Modbus unit identifier (slave address, typically 1)
    ///
    /// # Returns
    ///
    /// Connected Modbus TCP client
    ///
    /// # Errors
    ///
    /// Returns error if connection fails or times out.
    pub async fn connect(addr: &str, unit_id: u8) -> ModbusResult<Self> {
        let stream = TcpStream::connect(addr).await?;

        // Disable Nagle's algorithm for low latency
        stream.set_nodelay(true)?;

        Ok(Self {
            stream,
            unit_id,
            transaction_id: Arc::new(AtomicU16::new(1)),
            timeout: Duration::from_secs(5),
        })
    }

    /// Set request timeout
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Read a complete Modbus TCP response ADU from the stream.
    ///
    /// Reads the fixed 7-byte MBAP header first, then parses the `length`
    /// field (bytes 4..6) to determine exactly how many more bytes belong
    /// to this response's PDU (function code + data). This covers both
    /// normal responses and exception responses uniformly, since both
    /// share the same MBAP framing and the `length` field always reflects
    /// the true remaining byte count on the wire.
    ///
    /// Blindly reading a fixed 9-byte header and then interpreting the
    /// 9th byte as a "byte count" (as older code in this module did) is
    /// wrong for exception responses, where that byte is actually the
    /// exception code: it either causes the client to block waiting for
    /// bytes that will never arrive (masking a fast `ModbusException` as
    /// a slow `Timeout`), or, on a connection with more inbound data
    /// pending, silently consumes bytes belonging to the *next* response
    /// and permanently desyncs the TCP byte stream.
    async fn read_response(&mut self) -> ModbusResult<ModbusTcpFrame> {
        // Read MBAP header (7 bytes: transaction_id(2) + protocol_id(2) + length(2) + unit_id(1))
        let mut header = [0u8; 7];
        timeout(self.timeout, self.stream.read_exact(&mut header))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        // MBAP length field = unit_id(1) + function_code(1) + data.
        // The 7-byte header already consumed unit_id, so the remaining
        // bytes on the wire (function code + data) = length - 1.
        let mbap_length = u16::from_be_bytes([header[4], header[5]]) as usize;
        if mbap_length < 2 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "MBAP length field too small (must be >= 2 for unit_id + function code)",
            )));
        }
        let remaining = mbap_length - 1;

        let mut pdu = vec![0u8; remaining];
        timeout(self.timeout, self.stream.read_exact(&mut pdu))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        let mut response_bytes = Vec::with_capacity(7 + remaining);
        response_bytes.extend_from_slice(&header);
        response_bytes.extend_from_slice(&pdu);

        ModbusTcpFrame::from_bytes(&response_bytes)
    }

    /// Read holding registers (function code 0x03)
    ///
    /// # Arguments
    ///
    /// * `start_addr` - Starting register address (0-65535)
    /// * `count` - Number of registers to read (1-125)
    ///
    /// # Returns
    ///
    /// Vector of register values (u16)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `start_addr` + `count` > 65535
    /// - `count` > 125 (Modbus limit)
    /// - Device returns exception
    /// - Timeout occurs
    pub async fn read_holding_registers(
        &mut self,
        start_addr: u16,
        count: u16,
    ) -> ModbusResult<Vec<u16>> {
        // Validate parameters
        if count > 125 {
            return Err(ModbusError::InvalidCount(count));
        }

        if start_addr as u32 + count as u32 > 65536 {
            return Err(ModbusError::InvalidAddress(start_addr));
        }

        // Get next transaction ID
        let tid = self.transaction_id.fetch_add(1, Ordering::SeqCst);

        // Build request frame
        let request = ModbusTcpFrame::read_holding_registers(tid, self.unit_id, start_addr, count);
        let request_bytes = request.to_bytes();

        // Send request with timeout
        timeout(self.timeout, self.stream.write_all(&request_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        // Read response ADU (handles both normal and exception responses
        // correctly by parsing the MBAP length field instead of assuming
        // a fixed header size).
        let response = self.read_response().await?;

        // Verify transaction ID matches
        if response.transaction_id != tid {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Transaction ID mismatch: expected {}, got {}",
                    tid, response.transaction_id
                ),
            )));
        }

        // Extract register values
        response.extract_registers()
    }

    /// Read input registers (function code 0x04)
    ///
    /// Similar to read_holding_registers but reads input registers (read-only).
    pub async fn read_input_registers(
        &mut self,
        start_addr: u16,
        count: u16,
    ) -> ModbusResult<Vec<u16>> {
        // Implementation similar to read_holding_registers
        // but uses FunctionCode::ReadInputRegisters
        if count > 125 {
            return Err(ModbusError::InvalidCount(count));
        }

        let tid = self.transaction_id.fetch_add(1, Ordering::SeqCst);

        let mut data = bytes::BytesMut::with_capacity(4);
        data.put_u16(start_addr);
        data.put_u16(count);

        let request = ModbusTcpFrame {
            transaction_id: tid,
            protocol_id: 0,
            unit_id: self.unit_id,
            function_code: FunctionCode::ReadInputRegisters,
            data: data.to_vec(),
        };

        let request_bytes = request.to_bytes();

        timeout(self.timeout, self.stream.write_all(&request_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        let response = self.read_response().await?;
        response.extract_registers()
    }

    /// Write single register (function code 0x06)
    ///
    /// # Arguments
    ///
    /// * `addr` - Register address (0-65535)
    /// * `value` - Value to write (0-65535)
    ///
    /// # Returns
    ///
    /// Ok(()) on success
    pub async fn write_single_register(&mut self, addr: u16, value: u16) -> ModbusResult<()> {
        let tid = self.transaction_id.fetch_add(1, Ordering::SeqCst);

        let request = ModbusTcpFrame::write_single_register(tid, self.unit_id, addr, value);
        let request_bytes = request.to_bytes();

        timeout(self.timeout, self.stream.write_all(&request_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        // Read response (echoes back address and value on success; a
        // shorter exception ADU on failure).
        let response = self.read_response().await?;

        // Verify write was successful (response echoes request)
        if response.data.len() >= 4 {
            let resp_addr = u16::from_be_bytes([response.data[0], response.data[1]]);
            let resp_value = u16::from_be_bytes([response.data[2], response.data[3]]);

            if resp_addr != addr || resp_value != value {
                return Err(ModbusError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Write verification failed",
                )));
            }
        }

        Ok(())
    }

    /// Read coils (function code 0x01)
    ///
    /// # Arguments
    ///
    /// * `start_addr` - Starting coil address (0-65535)
    /// * `count` - Number of coils to read (1-2000)
    ///
    /// # Returns
    ///
    /// Vector of coil states (true = ON, false = OFF)
    pub async fn read_coils(&mut self, start_addr: u16, count: u16) -> ModbusResult<Vec<bool>> {
        if count > 2000 {
            return Err(ModbusError::InvalidCount(count));
        }

        let tid = self.transaction_id.fetch_add(1, Ordering::SeqCst);

        let mut data = bytes::BytesMut::with_capacity(4);
        data.put_u16(start_addr);
        data.put_u16(count);

        let request = ModbusTcpFrame {
            transaction_id: tid,
            protocol_id: 0,
            unit_id: self.unit_id,
            function_code: FunctionCode::ReadCoils,
            data: data.to_vec(),
        };

        let request_bytes = request.to_bytes();

        timeout(self.timeout, self.stream.write_all(&request_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        let response = self.read_response().await?;

        // Response PDU data: byte_count(1) + packed coil bytes
        let data_bytes = response.data.get(1..).ok_or_else(|| {
            ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Incomplete coil data",
            ))
        })?;

        // Extract coil values from packed bytes
        let mut coils = Vec::with_capacity(count as usize);
        for (byte_idx, &byte) in data_bytes.iter().enumerate() {
            for bit_idx in 0..8 {
                let coil_idx = byte_idx * 8 + bit_idx;
                if coil_idx >= count as usize {
                    break;
                }
                coils.push((byte >> bit_idx) & 1 == 1);
            }
        }

        Ok(coils)
    }

    /// Read discrete inputs (function code 0x02)
    ///
    /// # Arguments
    ///
    /// * `start_addr` - Starting address (0-65535)
    /// * `count` - Number of inputs to read (1-2000)
    ///
    /// # Returns
    ///
    /// Vector of input states (true = ON, false = OFF)
    pub async fn read_discrete_inputs(
        &mut self,
        start_addr: u16,
        count: u16,
    ) -> ModbusResult<Vec<bool>> {
        if count > 2000 {
            return Err(ModbusError::InvalidCount(count));
        }

        let tid = self.transaction_id.fetch_add(1, Ordering::SeqCst);

        let mut data = bytes::BytesMut::with_capacity(4);
        data.put_u16(start_addr);
        data.put_u16(count);

        let request = ModbusTcpFrame {
            transaction_id: tid,
            protocol_id: 0,
            unit_id: self.unit_id,
            function_code: FunctionCode::ReadDiscreteInputs,
            data: data.to_vec(),
        };

        let request_bytes = request.to_bytes();

        timeout(self.timeout, self.stream.write_all(&request_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        let response = self.read_response().await?;

        // Response PDU data: byte_count(1) + packed input bytes
        let data_bytes = response.data.get(1..).ok_or_else(|| {
            ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Incomplete discrete input data",
            ))
        })?;

        // Extract input values from packed bytes
        let mut inputs = Vec::with_capacity(count as usize);
        for (byte_idx, &byte) in data_bytes.iter().enumerate() {
            for bit_idx in 0..8 {
                let input_idx = byte_idx * 8 + bit_idx;
                if input_idx >= count as usize {
                    break;
                }
                inputs.push((byte >> bit_idx) & 1 == 1);
            }
        }

        Ok(inputs)
    }

    /// Write multiple coils (function code 0x0F)
    ///
    /// # Arguments
    ///
    /// * `start_addr` - Starting coil address (0-65535)
    /// * `coils` - Coil values to write (true = ON, false = OFF)
    ///
    /// # Returns
    ///
    /// Ok(()) on success
    ///
    /// # Errors
    ///
    /// Returns error if coils.len() > 1968 (Modbus limit)
    pub async fn write_multiple_coils(
        &mut self,
        start_addr: u16,
        coils: &[bool],
    ) -> ModbusResult<()> {
        if coils.is_empty() {
            return Err(ModbusError::InvalidCount(0));
        }

        if coils.len() > 1968 {
            return Err(ModbusError::InvalidCount(coils.len() as u16));
        }

        let tid = self.transaction_id.fetch_add(1, Ordering::SeqCst);
        let count = coils.len() as u16;

        // Pack coils into bytes (LSB first)
        let byte_count = (coils.len() + 7) / 8;
        let mut coil_bytes = vec![0u8; byte_count];

        for (i, &coil) in coils.iter().enumerate() {
            if coil {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                coil_bytes[byte_idx] |= 1 << bit_idx;
            }
        }

        // Build request: start_addr (2) + count (2) + byte_count (1) + coil_bytes
        let mut data = bytes::BytesMut::with_capacity(5 + byte_count);
        data.put_u16(start_addr);
        data.put_u16(count);
        data.put_u8(byte_count as u8);
        data.extend_from_slice(&coil_bytes);

        let request = ModbusTcpFrame {
            transaction_id: tid,
            protocol_id: 0,
            unit_id: self.unit_id,
            function_code: FunctionCode::WriteMultipleCoils,
            data: data.to_vec(),
        };

        let request_bytes = request.to_bytes();

        timeout(self.timeout, self.stream.write_all(&request_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        // Read response: normal reply echoes start_addr(2) + quantity(2);
        // an exception reply is shorter. `read_response` sizes the read
        // from the MBAP length field so both cases are handled correctly.
        let response = self.read_response().await?;

        // Verify response
        if response.data.len() >= 4 {
            let resp_addr = u16::from_be_bytes([response.data[0], response.data[1]]);
            let resp_count = u16::from_be_bytes([response.data[2], response.data[3]]);

            if resp_addr != start_addr || resp_count != count {
                return Err(ModbusError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Write verification failed: expected addr={}, count={}, got addr={}, count={}",
                        start_addr, count, resp_addr, resp_count
                    ),
                )));
            }
        }

        Ok(())
    }

    /// Write multiple registers (function code 0x10)
    ///
    /// # Arguments
    ///
    /// * `start_addr` - Starting register address (0-65535)
    /// * `values` - Register values to write
    ///
    /// # Returns
    ///
    /// Ok(()) on success
    ///
    /// # Errors
    ///
    /// Returns error if values.len() > 123 (Modbus limit)
    pub async fn write_multiple_registers(
        &mut self,
        start_addr: u16,
        values: &[u16],
    ) -> ModbusResult<()> {
        if values.is_empty() {
            return Err(ModbusError::InvalidCount(0));
        }

        if values.len() > 123 {
            return Err(ModbusError::InvalidCount(values.len() as u16));
        }

        let tid = self.transaction_id.fetch_add(1, Ordering::SeqCst);
        let count = values.len() as u16;
        let byte_count = (values.len() * 2) as u8;

        // Build request: start_addr (2) + count (2) + byte_count (1) + values
        let mut data = bytes::BytesMut::with_capacity(5 + values.len() * 2);
        data.put_u16(start_addr);
        data.put_u16(count);
        data.put_u8(byte_count);

        for &value in values {
            data.put_u16(value);
        }

        let request = ModbusTcpFrame {
            transaction_id: tid,
            protocol_id: 0,
            unit_id: self.unit_id,
            function_code: FunctionCode::WriteMultipleRegisters,
            data: data.to_vec(),
        };

        let request_bytes = request.to_bytes();

        timeout(self.timeout, self.stream.write_all(&request_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        // Read response: normal reply echoes start_addr(2) + quantity(2);
        // an exception reply is shorter. `read_response` sizes the read
        // from the MBAP length field so both cases are handled correctly.
        let response = self.read_response().await?;

        // Verify response
        if response.data.len() >= 4 {
            let resp_addr = u16::from_be_bytes([response.data[0], response.data[1]]);
            let resp_count = u16::from_be_bytes([response.data[2], response.data[3]]);

            if resp_addr != start_addr || resp_count != count {
                return Err(ModbusError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Write verification failed: expected addr={}, count={}, got addr={}, count={}",
                        start_addr, count, resp_addr, resp_count
                    ),
                )));
            }
        }

        Ok(())
    }

    /// Get current unit ID
    pub fn unit_id(&self) -> u8 {
        self.unit_id
    }

    /// Set unit ID (slave address)
    pub fn set_unit_id(&mut self, unit_id: u8) {
        self.unit_id = unit_id;
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::ModbusTcpClient;
    use crate::error::ModbusError;
    use std::time::Duration;

    #[tokio::test]
    async fn test_connection_parameters() {
        // This test doesn't actually connect, just verifies construction
        // Real connection tests require a Modbus server running

        // We can't test real connection without a server, but we can test
        // that the function signature is correct and compiles
        let addr = "127.0.0.1:502";
        let unit_id: u8 = 1;

        // This would require a running server:
        // let _client = ModbusTcpClient::connect(addr, unit_id).await;

        // For now, just verify types compile
        let _: &str = addr;
        let _: u8 = unit_id;
    }

    #[test]
    fn test_set_timeout() {
        // Can't create client without async, so we'll test this when we have mock server
        use std::time::Duration;
        let _ = Duration::from_secs(5); // Placeholder
    }

    // ------------------------------------------------------------------
    // Regression tests for exception-response desync bug (P0)
    //
    // Prior to the fix, the client read a fixed 9-byte header and then
    // treated header[8] as a "byte count" even for exception responses
    // (where that byte is actually the exception code). This caused a
    // second, wrongly-sized `read_exact` that either hung until timeout
    // or consumed bytes belonging to the *next* response, permanently
    // desyncing the TCP stream.
    // ------------------------------------------------------------------
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::net::TcpListener;

    /// Build a raw Modbus TCP exception response ADU.
    fn build_exception_response(
        tid: u16,
        unit_id: u8,
        function_code: u8,
        exception_code: u8,
    ) -> Vec<u8> {
        let mut resp = Vec::with_capacity(9);
        resp.extend_from_slice(&tid.to_be_bytes());
        resp.extend_from_slice(&0u16.to_be_bytes()); // protocol id
        resp.extend_from_slice(&3u16.to_be_bytes()); // length = unit_id + func + exception_code
        resp.push(unit_id);
        resp.push(function_code | 0x80);
        resp.push(exception_code);
        resp
    }

    /// Build a raw Modbus TCP normal "read holding registers" response ADU.
    fn build_read_holding_registers_response(tid: u16, unit_id: u8, registers: &[u16]) -> Vec<u8> {
        let byte_count = (registers.len() * 2) as u8;
        let length = 1u16 + 1 + 1 + byte_count as u16; // unit_id + func + byte_count + data
        let mut resp = Vec::new();
        resp.extend_from_slice(&tid.to_be_bytes());
        resp.extend_from_slice(&0u16.to_be_bytes());
        resp.extend_from_slice(&length.to_be_bytes());
        resp.push(unit_id);
        resp.push(0x03); // ReadHoldingRegisters
        resp.push(byte_count);
        for &r in registers {
            resp.extend_from_slice(&r.to_be_bytes());
        }
        resp
    }

    #[tokio::test]
    async fn regression_exception_response_does_not_desync_stream() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock server");
        let addr = listener.local_addr().expect("local addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");

            // First request: read_holding_registers request is 12 bytes
            // (MBAP 7 + FC 1 + start_addr 2 + count 2).
            let mut req1 = vec![0u8; 12];
            socket.read_exact(&mut req1).await.expect("read req1");
            let tid1 = u16::from_be_bytes([req1[0], req1[1]]);

            // Respond with an IllegalDataAddress exception (9 bytes total).
            let exc = build_exception_response(tid1, 1, 0x03, 0x02);
            socket.write_all(&exc).await.expect("write exception");

            // Second request on the SAME connection: if the client desynced
            // after the exception, this read (or the response it triggers)
            // would be misaligned.
            let mut req2 = vec![0u8; 12];
            socket.read_exact(&mut req2).await.expect("read req2");
            let tid2 = u16::from_be_bytes([req2[0], req2[1]]);

            let ok = build_read_holding_registers_response(tid2, 1, &[100, 200]);
            socket.write_all(&ok).await.expect("write ok response");
        });

        let mut client = ModbusTcpClient::connect(&addr.to_string(), 1)
            .await
            .expect("connect");
        client.set_timeout(Duration::from_secs(2));

        // First call must fail fast with a typed ModbusException, not a
        // Timeout (which would indicate the client is blocked waiting for
        // bytes that never arrive).
        let first = client.read_holding_registers(0, 2).await;
        match first {
            Err(ModbusError::ModbusException { code, function }) => {
                assert_eq!(code, 0x02);
                assert_eq!(function, 0x03);
            }
            other => panic!("expected ModbusException, got: {:?}", other),
        }

        // Second call on the same connection must succeed and return the
        // correct registers -- proving the byte stream was not desynced by
        // the first (exception) response.
        let second = client
            .read_holding_registers(0, 2)
            .await
            .expect("second call should succeed");
        assert_eq!(second, vec![100, 200]);

        server.await.expect("server task");
    }

    #[tokio::test]
    async fn regression_write_single_register_exception_is_fast_not_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock server");
        let addr = listener.local_addr().expect("local addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");

            // write_single_register request is 12 bytes (MBAP 7 + FC 1 + addr 2 + value 2).
            let mut req = vec![0u8; 12];
            socket.read_exact(&mut req).await.expect("read req");
            let tid = u16::from_be_bytes([req[0], req[1]]);

            // Exception response is only 9 bytes -- shorter than the 12
            // bytes the old fixed-size read assumed.
            let exc = build_exception_response(tid, 1, 0x06, 0x04);
            socket.write_all(&exc).await.expect("write exception");
        });

        let mut client = ModbusTcpClient::connect(&addr.to_string(), 1)
            .await
            .expect("connect");
        client.set_timeout(Duration::from_millis(500));

        let start = std::time::Instant::now();
        let result = client.write_single_register(10, 42).await;
        let elapsed = start.elapsed();

        match result {
            Err(ModbusError::ModbusException { code, function }) => {
                assert_eq!(code, 0x04);
                assert_eq!(function, 0x06);
            }
            other => panic!("expected ModbusException, got: {:?}", other),
        }
        // Must resolve well within the configured timeout, proving the
        // response was parsed rather than blocking on missing bytes.
        assert!(
            elapsed < Duration::from_millis(400),
            "exception response took too long ({:?}), suggests desync/blocking read",
            elapsed
        );

        server.await.expect("server task");
    }
}
