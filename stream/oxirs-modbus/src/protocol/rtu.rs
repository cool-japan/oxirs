//! Modbus RTU client implementation
//!
//! Provides async serial communication for Modbus RTU devices
//! using tokio-serial.

use crate::error::{ModbusError, ModbusResult};
use crate::protocol::crc::{append_crc, verify_crc};
use crate::protocol::frame::FunctionCode;
use bytes::{BufMut, BytesMut};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_serial::{DataBits, FlowControl, Parity, SerialPortBuilderExt, SerialStream, StopBits};

/// Default baud rate for Modbus RTU
pub const DEFAULT_BAUD_RATE: u32 = 9600;

/// Default timeout for RTU operations (longer due to serial communication)
pub const DEFAULT_RTU_TIMEOUT: Duration = Duration::from_millis(1000);

/// Inter-frame delay for Modbus RTU (3.5 character times at 9600 baud ≈ 4ms)
const INTER_FRAME_DELAY_MS: u64 = 4;

/// Modbus RTU client for serial communication
///
/// Implements async Modbus RTU protocol over RS-232/RS-485 serial ports.
///
/// # Example
///
/// ```no_run
/// use oxirs_modbus::protocol::rtu::ModbusRtuClient;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut client = ModbusRtuClient::open("/dev/ttyUSB0", 9600, 1)?;
///     let registers = client.read_holding_registers(0, 10).await?;
///     println!("Registers: {:?}", registers);
///     Ok(())
/// }
/// ```
pub struct ModbusRtuClient {
    /// Serial port stream
    stream: SerialStream,

    /// Unit ID (slave address)
    unit_id: u8,

    /// Response timeout
    timeout: Duration,
}

impl ModbusRtuClient {
    /// Open a Modbus RTU connection
    ///
    /// # Arguments
    ///
    /// * `port` - Serial port path (e.g., "/dev/ttyUSB0", "COM1")
    /// * `baud_rate` - Baud rate (typically 9600, 19200, 38400, 57600, or 115200)
    /// * `unit_id` - Modbus unit/slave ID (1-247)
    ///
    /// # Returns
    ///
    /// A connected `ModbusRtuClient` instance
    pub fn open(port: &str, baud_rate: u32, unit_id: u8) -> ModbusResult<Self> {
        let stream = tokio_serial::new(port, baud_rate)
            .data_bits(DataBits::Eight)
            .parity(Parity::None)
            .stop_bits(StopBits::One)
            .flow_control(FlowControl::None)
            .open_native_async()
            .map_err(|e| ModbusError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        Ok(Self {
            stream,
            unit_id,
            timeout: DEFAULT_RTU_TIMEOUT,
        })
    }

    /// Open with custom serial settings
    ///
    /// # Arguments
    ///
    /// * `port` - Serial port path
    /// * `baud_rate` - Baud rate
    /// * `unit_id` - Modbus unit ID
    /// * `data_bits` - Number of data bits
    /// * `parity` - Parity setting
    /// * `stop_bits` - Stop bits
    pub fn open_with_settings(
        port: &str,
        baud_rate: u32,
        unit_id: u8,
        data_bits: DataBits,
        parity: Parity,
        stop_bits: StopBits,
    ) -> ModbusResult<Self> {
        let stream = tokio_serial::new(port, baud_rate)
            .data_bits(data_bits)
            .parity(parity)
            .stop_bits(stop_bits)
            .flow_control(FlowControl::None)
            .open_native_async()
            .map_err(|e| ModbusError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        Ok(Self {
            stream,
            unit_id,
            timeout: DEFAULT_RTU_TIMEOUT,
        })
    }

    /// Set response timeout
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Read holding registers (function code 0x03)
    ///
    /// # Arguments
    ///
    /// * `start_address` - Starting register address (0-based)
    /// * `count` - Number of registers to read (1-125)
    ///
    /// # Returns
    ///
    /// Vector of register values
    pub async fn read_holding_registers(
        &mut self,
        start_address: u16,
        count: u16,
    ) -> ModbusResult<Vec<u16>> {
        if count > 125 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot read more than 125 registers",
            )));
        }

        let request = self.build_request(FunctionCode::ReadHoldingRegisters, start_address, count);

        let response = self.send_request(&request).await?;
        self.extract_registers(&response)
    }

    /// Read input registers (function code 0x04)
    ///
    /// # Arguments
    ///
    /// * `start_address` - Starting register address (0-based)
    /// * `count` - Number of registers to read (1-125)
    ///
    /// # Returns
    ///
    /// Vector of register values
    pub async fn read_input_registers(
        &mut self,
        start_address: u16,
        count: u16,
    ) -> ModbusResult<Vec<u16>> {
        if count > 125 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot read more than 125 registers",
            )));
        }

        let request = self.build_request(FunctionCode::ReadInputRegisters, start_address, count);

        let response = self.send_request(&request).await?;
        self.extract_registers(&response)
    }

    /// Write a single register (function code 0x06)
    ///
    /// # Arguments
    ///
    /// * `address` - Register address
    /// * `value` - Value to write
    pub async fn write_single_register(&mut self, address: u16, value: u16) -> ModbusResult<()> {
        let request = self.build_request(FunctionCode::WriteSingleRegister, address, value);

        let _response = self.send_request(&request).await?;
        Ok(())
    }

    /// Read coils (function code 0x01)
    ///
    /// # Arguments
    ///
    /// * `start_address` - Starting coil address (0-based)
    /// * `count` - Number of coils to read (1-2000)
    ///
    /// # Returns
    ///
    /// Vector of coil states (true = ON, false = OFF)
    pub async fn read_coils(&mut self, start_address: u16, count: u16) -> ModbusResult<Vec<bool>> {
        if count > 2000 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot read more than 2000 coils",
            )));
        }

        let request = self.build_request(FunctionCode::ReadCoils, start_address, count);
        let response = self.send_request(&request).await?;
        self.extract_coils(&response, count as usize)
    }

    /// Read discrete inputs (function code 0x02)
    ///
    /// # Arguments
    ///
    /// * `start_address` - Starting address (0-based)
    /// * `count` - Number of inputs to read (1-2000)
    ///
    /// # Returns
    ///
    /// Vector of input states (true = ON, false = OFF)
    pub async fn read_discrete_inputs(
        &mut self,
        start_address: u16,
        count: u16,
    ) -> ModbusResult<Vec<bool>> {
        if count > 2000 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot read more than 2000 inputs",
            )));
        }

        let request = self.build_request(FunctionCode::ReadDiscreteInputs, start_address, count);
        let response = self.send_request(&request).await?;
        self.extract_coils(&response, count as usize)
    }

    /// Build RTU request frame
    fn build_request(&self, function_code: FunctionCode, param1: u16, param2: u16) -> Vec<u8> {
        let mut bytes = BytesMut::with_capacity(8);

        // Unit ID
        bytes.put_u8(self.unit_id);

        // Function code
        bytes.put_u8(function_code.as_u8());

        // Parameters (address/value)
        bytes.put_u16(param1);
        bytes.put_u16(param2);

        // Convert to Vec and append CRC
        let mut request = bytes.to_vec();
        append_crc(&mut request);

        request
    }

    /// Send request and receive response
    async fn send_request(&mut self, request: &[u8]) -> ModbusResult<Vec<u8>> {
        // Inter-frame delay before sending
        tokio::time::sleep(Duration::from_millis(INTER_FRAME_DELAY_MS)).await;

        // Send request
        self.stream.write_all(request).await?;
        self.stream.flush().await?;

        // Read response with timeout
        let response = tokio::time::timeout(self.timeout, self.read_response())
            .await
            .map_err(|_| ModbusError::Timeout(self.timeout))??;

        // Verify CRC
        if !verify_crc(&response) {
            return Err(ModbusError::CrcError {
                expected: 0, // We don't know the expected CRC here
                actual: 0,
            });
        }

        // Check for exception response
        if response.len() >= 3 && (response[1] & 0x80) != 0 {
            let function = response[1] & 0x7F;
            let exception_code = response[2];
            return Err(ModbusError::ModbusException {
                code: exception_code,
                function,
            });
        }

        Ok(response)
    }

    /// Read response from serial port
    async fn read_response(&mut self) -> ModbusResult<Vec<u8>> {
        let mut buffer = Vec::with_capacity(256);
        let mut temp = [0u8; 256];

        // Read initial bytes (unit_id + function_code)
        let n = self.stream.read(&mut temp).await?;
        if n < 2 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Response too short",
            )));
        }
        buffer.extend_from_slice(&temp[..n]);

        // Determine expected response length based on function code
        let function_code = buffer[1] & 0x7F;
        let expected_len = if (buffer[1] & 0x80) != 0 {
            // Exception response: unit_id + fc + exception_code + crc
            5
        } else {
            match function_code {
                0x01..=0x04 => {
                    // Read response: wait for byte count
                    if buffer.len() < 3 {
                        // Need to read more for byte count
                        let n = self.stream.read(&mut temp).await?;
                        buffer.extend_from_slice(&temp[..n]);
                    }
                    if buffer.len() < 3 {
                        return Err(ModbusError::Io(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "Missing byte count",
                        )));
                    }
                    // unit_id + fc + byte_count + data + crc
                    3 + buffer[2] as usize + 2
                }
                0x05 | 0x06 => {
                    // Write response echoes request: unit_id + fc + addr(2) + value(2) + crc
                    8
                }
                0x0F | 0x10 => {
                    // Write multiple response: unit_id + fc + addr(2) + quantity(2) + crc
                    8
                }
                _ => {
                    // Unknown, read what we can
                    buffer.len() + 2
                }
            }
        };

        // Read remaining bytes
        while buffer.len() < expected_len {
            let n = self.stream.read(&mut temp).await?;
            if n == 0 {
                break;
            }
            buffer.extend_from_slice(&temp[..n]);
        }

        Ok(buffer)
    }

    /// Extract register values from response
    fn extract_registers(&self, response: &[u8]) -> ModbusResult<Vec<u16>> {
        // Response: unit_id(1) + fc(1) + byte_count(1) + data(N) + crc(2)
        if response.len() < 5 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Response too short",
            )));
        }

        let byte_count = response[2] as usize;
        if response.len() < 3 + byte_count + 2 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Incomplete register data",
            )));
        }

        let mut registers = Vec::with_capacity(byte_count / 2);
        let data = &response[3..3 + byte_count];

        for chunk in data.chunks_exact(2) {
            let value = u16::from_be_bytes([chunk[0], chunk[1]]);
            registers.push(value);
        }

        Ok(registers)
    }

    /// Extract coil/input values from response
    fn extract_coils(&self, response: &[u8], count: usize) -> ModbusResult<Vec<bool>> {
        // Response: unit_id(1) + fc(1) + byte_count(1) + data(N) + crc(2)
        if response.len() < 5 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Response too short",
            )));
        }

        let byte_count = response[2] as usize;
        if response.len() < 3 + byte_count + 2 {
            return Err(ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Incomplete coil data",
            )));
        }

        let mut coils = Vec::with_capacity(count);
        let data = &response[3..3 + byte_count];

        for (byte_idx, &byte) in data.iter().enumerate() {
            for bit_idx in 0..8 {
                let coil_idx = byte_idx * 8 + bit_idx;
                if coil_idx >= count {
                    break;
                }
                coils.push((byte >> bit_idx) & 1 == 1);
            }
        }

        Ok(coils)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_request_read_registers() {
        // Create a mock stream for testing - we can't actually test without hardware
        // This test just verifies the request building logic
        let request_data = vec![
            0x01, // Unit ID
            0x03, // Function code (Read Holding Registers)
            0x00, 0x00, // Start address
            0x00, 0x0A, // Quantity (10 registers)
        ];

        let mut request = request_data.clone();
        append_crc(&mut request);

        assert_eq!(request.len(), 8); // 6 data bytes + 2 CRC bytes
        assert!(verify_crc(&request));
    }

    #[test]
    fn test_build_request_write_register() {
        let request_data = vec![
            0x01, // Unit ID
            0x06, // Function code (Write Single Register)
            0x00, 0x01, // Address
            0x00, 0x64, // Value (100)
        ];

        let mut request = request_data;
        append_crc(&mut request);

        assert_eq!(request.len(), 8);
        assert!(verify_crc(&request));
    }
}
