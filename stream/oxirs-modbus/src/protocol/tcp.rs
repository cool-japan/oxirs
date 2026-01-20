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

        // Read response header (MBAP header = 7 bytes + function code = 8 bytes + byte count = 9 bytes minimum)
        let mut header = vec![0u8; 9];
        timeout(self.timeout, self.stream.read_exact(&mut header))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        // Parse byte count from response
        let byte_count = header[8] as usize;

        // Read remaining data
        let mut data = vec![0u8; byte_count];
        timeout(self.timeout, self.stream.read_exact(&mut data))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        // Combine header + data and parse
        let mut response_bytes = header;
        response_bytes.extend_from_slice(&data);

        let response = ModbusTcpFrame::from_bytes(&response_bytes)?;

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

        let mut header = vec![0u8; 9];
        timeout(self.timeout, self.stream.read_exact(&mut header))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        let byte_count = header[8] as usize;
        let mut data_bytes = vec![0u8; byte_count];
        timeout(self.timeout, self.stream.read_exact(&mut data_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        let mut response_bytes = header;
        response_bytes.extend_from_slice(&data_bytes);

        let response = ModbusTcpFrame::from_bytes(&response_bytes)?;
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

        // Read response (echoes back address and value)
        let mut response_bytes = vec![0u8; 12];
        timeout(self.timeout, self.stream.read_exact(&mut response_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        let response = ModbusTcpFrame::from_bytes(&response_bytes)?;

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

        let mut header = vec![0u8; 9];
        timeout(self.timeout, self.stream.read_exact(&mut header))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        let byte_count = header[8] as usize;
        let mut data_bytes = vec![0u8; byte_count];
        timeout(self.timeout, self.stream.read_exact(&mut data_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

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

        let mut header = vec![0u8; 9];
        timeout(self.timeout, self.stream.read_exact(&mut header))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        let byte_count = header[8] as usize;
        let mut data_bytes = vec![0u8; byte_count];
        timeout(self.timeout, self.stream.read_exact(&mut data_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

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

        // Read response (12 bytes: MBAP header 7 + unit 1 + FC 1 + addr 2 + count 2 = 13? Check frame)
        // Response: MBAP (7) + unit (1) + FC (1) + start_addr (2) + quantity (2) = 13 bytes
        // But MBAP includes length (6 for unit+FC+addr+quantity)
        // So: transaction (2) + protocol (2) + length (2) + unit (1) + FC (1) + addr (2) + qty (2) = 12
        let mut response_bytes = vec![0u8; 12];
        timeout(self.timeout, self.stream.read_exact(&mut response_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        let response = ModbusTcpFrame::from_bytes(&response_bytes)?;

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

        // Read response (12 bytes)
        let mut response_bytes = vec![0u8; 12];
        timeout(self.timeout, self.stream.read_exact(&mut response_bytes))
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        let response = ModbusTcpFrame::from_bytes(&response_bytes)?;

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
}
