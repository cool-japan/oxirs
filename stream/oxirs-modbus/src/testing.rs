//! Testing utilities for Modbus
//!
//! Provides a mock Modbus TCP server for integration testing.

use crate::error::ModbusResult;
use crate::protocol::frame::{FunctionCode, ModbusTcpFrame};
use bytes::BufMut;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{oneshot, Mutex};

/// Test data for mock server
#[derive(Debug, Clone)]
pub struct MockServerData {
    /// Holding registers (FC 0x03, 0x06, 0x10)
    pub holding_registers: HashMap<u16, u16>,
    /// Input registers (FC 0x04)
    pub input_registers: HashMap<u16, u16>,
    /// Coils (FC 0x01, 0x05, 0x0F)
    pub coils: HashMap<u16, bool>,
    /// Discrete inputs (FC 0x02)
    pub discrete_inputs: HashMap<u16, bool>,
}

impl Default for MockServerData {
    fn default() -> Self {
        Self::with_test_data()
    }
}

impl MockServerData {
    /// Create empty data store
    pub fn new() -> Self {
        Self {
            holding_registers: HashMap::new(),
            input_registers: HashMap::new(),
            coils: HashMap::new(),
            discrete_inputs: HashMap::new(),
        }
    }

    /// Create with standard test data
    pub fn with_test_data() -> Self {
        let mut data = Self::new();

        // Holding registers: incrementing pattern
        for i in 0..100 {
            data.holding_registers.insert(i, (i + 1) * 100);
        }

        // Input registers: sensor simulation data
        data.input_registers.insert(0, 225); // Temperature: 22.5°C
        data.input_registers.insert(1, 501); // Humidity: 50.1%
        data.input_registers.insert(2, 1013); // Pressure: 1013 hPa
        data.input_registers.insert(3, 3300); // Voltage: 330.0V
        data.input_registers.insert(4, 150); // Current: 15.0A

        // Coils: alternating pattern
        for i in 0..32 {
            data.coils.insert(i, i % 2 == 0);
        }

        // Discrete inputs: all off except first few
        for i in 0..32 {
            data.discrete_inputs.insert(i, i < 4);
        }

        data
    }
}

/// Mock Modbus TCP server for testing
pub struct MockModbusServer {
    /// Server address
    address: String,
    /// Server data
    data: Arc<Mutex<MockServerData>>,
    /// Shutdown signal sender
    shutdown_tx: Option<oneshot::Sender<()>>,
    /// Server task handle
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl MockModbusServer {
    /// Create and start a new mock server
    ///
    /// Returns the server and the address it's listening on.
    pub async fn start() -> ModbusResult<Self> {
        Self::start_with_data(MockServerData::with_test_data()).await
    }

    /// Create and start with custom data
    pub async fn start_with_data(data: MockServerData) -> ModbusResult<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?.to_string();
        let data = Arc::new(Mutex::new(data));

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let server_data = data.clone();

        let handle = tokio::spawn(async move {
            Self::run_server(listener, server_data, shutdown_rx).await;
        });

        Ok(Self {
            address,
            data,
            shutdown_tx: Some(shutdown_tx),
            handle: Some(handle),
        })
    }

    /// Get server address (host:port)
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Get mutable access to server data
    pub async fn data(&self) -> tokio::sync::MutexGuard<'_, MockServerData> {
        self.data.lock().await
    }

    /// Run the server loop
    async fn run_server(
        listener: TcpListener,
        data: Arc<Mutex<MockServerData>>,
        mut shutdown_rx: oneshot::Receiver<()>,
    ) {
        loop {
            tokio::select! {
                result = listener.accept() => {
                    if let Ok((stream, _addr)) = result {
                        let client_data = data.clone();
                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_client(stream, client_data).await {
                                eprintln!("Mock server client error: {:?}", e);
                            }
                        });
                    }
                }
                _ = &mut shutdown_rx => {
                    break;
                }
            }
        }
    }

    /// Handle a single client connection
    async fn handle_client(
        mut stream: TcpStream,
        data: Arc<Mutex<MockServerData>>,
    ) -> ModbusResult<()> {
        let mut buffer = [0u8; 260]; // Max Modbus frame size

        loop {
            // Read MBAP header (7 bytes) + at least 1 byte PDU
            let n = stream.read(&mut buffer).await?;
            if n < 8 {
                break; // Connection closed or incomplete
            }

            // Parse request
            let request = ModbusTcpFrame::from_bytes(&buffer[..n])?;

            // Generate response
            let response = Self::process_request(&request, &data).await?;

            // Send response
            let response_bytes = response.to_bytes();
            stream.write_all(&response_bytes).await?;
        }

        Ok(())
    }

    /// Process a Modbus request and generate response
    async fn process_request(
        request: &ModbusTcpFrame,
        data: &Arc<Mutex<MockServerData>>,
    ) -> ModbusResult<ModbusTcpFrame> {
        let mut data = data.lock().await;

        let response_data = match request.function_code {
            FunctionCode::ReadHoldingRegisters => {
                Self::handle_read_registers(&request.data, &data.holding_registers)?
            }
            FunctionCode::ReadInputRegisters => {
                Self::handle_read_registers(&request.data, &data.input_registers)?
            }
            FunctionCode::WriteSingleRegister => {
                // Echo request data for write confirmation
                if request.data.len() >= 4 {
                    let addr = u16::from_be_bytes([request.data[0], request.data[1]]);
                    let value = u16::from_be_bytes([request.data[2], request.data[3]]);
                    data.holding_registers.insert(addr, value);
                }
                request.data.clone()
            }
            FunctionCode::WriteSingleCoil => {
                // Echo request data for write confirmation
                if request.data.len() >= 4 {
                    let addr = u16::from_be_bytes([request.data[0], request.data[1]]);
                    let value = u16::from_be_bytes([request.data[2], request.data[3]]);
                    // 0xFF00 = ON, 0x0000 = OFF
                    data.coils.insert(addr, value == 0xFF00);
                }
                request.data.clone()
            }
            FunctionCode::ReadCoils => Self::handle_read_coils(&request.data, &data.coils)?,
            FunctionCode::ReadDiscreteInputs => {
                Self::handle_read_coils(&request.data, &data.discrete_inputs)?
            }
            FunctionCode::WriteMultipleCoils => {
                Self::handle_write_multiple_coils(&request.data, &mut data.coils)?
            }
            FunctionCode::WriteMultipleRegisters => {
                Self::handle_write_multiple_registers(&request.data, &mut data.holding_registers)?
            } // All function codes are now covered
        };

        Ok(ModbusTcpFrame {
            transaction_id: request.transaction_id,
            protocol_id: 0,
            unit_id: request.unit_id,
            function_code: request.function_code,
            data: response_data,
        })
    }

    /// Handle read registers request
    fn handle_read_registers(
        request_data: &[u8],
        registers: &HashMap<u16, u16>,
    ) -> ModbusResult<Vec<u8>> {
        if request_data.len() < 4 {
            return Ok(vec![0x03]); // Illegal Data Value
        }

        let start_addr = u16::from_be_bytes([request_data[0], request_data[1]]);
        let count = u16::from_be_bytes([request_data[2], request_data[3]]);

        let byte_count = (count * 2) as u8;
        let mut response = Vec::with_capacity(1 + byte_count as usize);
        response.push(byte_count);

        for i in 0..count {
            let addr = start_addr + i;
            let value = registers.get(&addr).copied().unwrap_or(0);
            response.put_u16(value);
        }

        Ok(response)
    }

    /// Handle read coils/discrete inputs request
    fn handle_read_coils(request_data: &[u8], coils: &HashMap<u16, bool>) -> ModbusResult<Vec<u8>> {
        if request_data.len() < 4 {
            return Ok(vec![0x03]); // Illegal Data Value
        }

        let start_addr = u16::from_be_bytes([request_data[0], request_data[1]]);
        let count = u16::from_be_bytes([request_data[2], request_data[3]]) as usize;

        let byte_count = (count + 7) / 8;
        let mut response = Vec::with_capacity(1 + byte_count);
        response.push(byte_count as u8);

        for byte_idx in 0..byte_count {
            let mut byte = 0u8;
            for bit_idx in 0..8 {
                let coil_idx = byte_idx * 8 + bit_idx;
                if coil_idx >= count {
                    break;
                }
                let addr = start_addr + coil_idx as u16;
                if coils.get(&addr).copied().unwrap_or(false) {
                    byte |= 1 << bit_idx;
                }
            }
            response.push(byte);
        }

        Ok(response)
    }

    /// Handle write multiple coils request (FC 0x0F)
    fn handle_write_multiple_coils(
        request_data: &[u8],
        coils: &mut HashMap<u16, bool>,
    ) -> ModbusResult<Vec<u8>> {
        if request_data.len() < 5 {
            return Ok(vec![0x03]); // Illegal Data Value
        }

        let start_addr = u16::from_be_bytes([request_data[0], request_data[1]]);
        let count = u16::from_be_bytes([request_data[2], request_data[3]]);
        let byte_count = request_data[4] as usize;

        if request_data.len() < 5 + byte_count {
            return Ok(vec![0x03]); // Illegal Data Value
        }

        // Unpack coils from bytes
        for i in 0..count {
            let byte_idx = (i / 8) as usize;
            let bit_idx = (i % 8) as usize;
            let byte = request_data[5 + byte_idx];
            let coil_value = (byte >> bit_idx) & 1 == 1;
            coils.insert(start_addr + i, coil_value);
        }

        // Response: start_addr + count
        let mut response = Vec::with_capacity(4);
        response.put_u16(start_addr);
        response.put_u16(count);
        Ok(response)
    }

    /// Handle write multiple registers request (FC 0x10)
    fn handle_write_multiple_registers(
        request_data: &[u8],
        registers: &mut HashMap<u16, u16>,
    ) -> ModbusResult<Vec<u8>> {
        if request_data.len() < 5 {
            return Ok(vec![0x03]); // Illegal Data Value
        }

        let start_addr = u16::from_be_bytes([request_data[0], request_data[1]]);
        let count = u16::from_be_bytes([request_data[2], request_data[3]]);
        let byte_count = request_data[4] as usize;

        if request_data.len() < 5 + byte_count {
            return Ok(vec![0x03]); // Illegal Data Value
        }

        // Write registers
        for i in 0..count {
            let offset = 5 + (i as usize * 2);
            let value = u16::from_be_bytes([request_data[offset], request_data[offset + 1]]);
            registers.insert(start_addr + i, value);
        }

        // Response: start_addr + count
        let mut response = Vec::with_capacity(4);
        response.put_u16(start_addr);
        response.put_u16(count);
        Ok(response)
    }

    /// Shutdown the server
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

impl Drop for MockModbusServer {
    fn drop(&mut self) {
        // Send shutdown signal if not already done
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_server_starts() {
        let server = MockModbusServer::start().await.unwrap();
        assert!(!server.address().is_empty());
        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_mock_server_data() {
        let server = MockModbusServer::start().await.unwrap();

        // Verify test data - use scope to drop MutexGuard before shutdown
        {
            let data = server.data().await;
            assert_eq!(data.holding_registers.get(&0), Some(&100));
            assert_eq!(data.holding_registers.get(&1), Some(&200));
            assert_eq!(data.input_registers.get(&0), Some(&225)); // Temperature
        }

        server.shutdown().await;
    }
}
