//! Modbus function code dispatch table.
//!
//! Provides a registry of handlers for each Modbus function code, dispatching
//! incoming PDU requests to the correct handler and returning PDU responses.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Modbus Function Code enum
// ---------------------------------------------------------------------------

/// The nine standard Modbus function codes supported by this implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FunctionCode {
    ReadCoils = 0x01,
    ReadDiscreteInputs = 0x02,
    ReadHoldingRegisters = 0x03,
    ReadInputRegisters = 0x04,
    WriteSingleCoil = 0x05,
    WriteSingleRegister = 0x06,
    WriteMultipleCoils = 0x0F,
    WriteMultipleRegisters = 0x10,
    ReadDeviceIdentification = 0x2B,
}

impl FunctionCode {
    /// Return the raw byte value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Try to convert a raw byte to a `FunctionCode`.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::ReadCoils),
            0x02 => Some(Self::ReadDiscreteInputs),
            0x03 => Some(Self::ReadHoldingRegisters),
            0x04 => Some(Self::ReadInputRegisters),
            0x05 => Some(Self::WriteSingleCoil),
            0x06 => Some(Self::WriteSingleRegister),
            0x0F => Some(Self::WriteMultipleCoils),
            0x10 => Some(Self::WriteMultipleRegisters),
            0x2B => Some(Self::ReadDeviceIdentification),
            _ => None,
        }
    }

    /// Human-readable name for logging.
    pub fn name(self) -> &'static str {
        match self {
            Self::ReadCoils => "ReadCoils",
            Self::ReadDiscreteInputs => "ReadDiscreteInputs",
            Self::ReadHoldingRegisters => "ReadHoldingRegisters",
            Self::ReadInputRegisters => "ReadInputRegisters",
            Self::WriteSingleCoil => "WriteSingleCoil",
            Self::WriteSingleRegister => "WriteSingleRegister",
            Self::WriteMultipleCoils => "WriteMultipleCoils",
            Self::WriteMultipleRegisters => "WriteMultipleRegisters",
            Self::ReadDeviceIdentification => "ReadDeviceIdentification",
        }
    }
}

// ---------------------------------------------------------------------------
// Request / Response PDU types
// ---------------------------------------------------------------------------

/// A raw Modbus function code request PDU (function code byte + data bytes).
#[derive(Debug, Clone)]
pub struct FunctionCodeRequest {
    /// Function code byte.
    pub code: u8,
    /// Request data bytes (excluding the function code byte).
    pub data: Vec<u8>,
}

/// A raw Modbus function code response PDU.
#[derive(Debug, Clone)]
pub struct FunctionCodeResponse {
    /// Function code byte (may have bit 7 set on error).
    pub code: u8,
    /// Response data bytes.
    pub data: Vec<u8>,
    /// `true` when the response carries an exception code (code has bit 7 set).
    pub is_error: bool,
}

// ---------------------------------------------------------------------------
// Handler trait
// ---------------------------------------------------------------------------

/// Trait for objects that can handle a specific Modbus function code.
pub trait FunctionCodeHandler: Send + Sync {
    /// Process the request and return a response.
    fn handle(&self, req: &FunctionCodeRequest) -> FunctionCodeResponse;

    /// The Modbus function code byte this handler is responsible for.
    fn function_code(&self) -> u8;

    /// Short textual description of what this handler does.
    fn description(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Dispatch table
// ---------------------------------------------------------------------------

/// A mapping from function code bytes to boxed `FunctionCodeHandler`s.
pub struct DispatchTable {
    handlers: HashMap<u8, Box<dyn FunctionCodeHandler>>,
}

impl Default for DispatchTable {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatchTable {
    /// Create an empty dispatch table.
    pub fn new() -> Self {
        DispatchTable {
            handlers: HashMap::new(),
        }
    }

    /// Create a dispatch table pre-loaded with echo handlers for all nine
    /// standard function codes.
    pub fn with_defaults() -> Self {
        let mut table = Self::new();
        let codes = [
            FunctionCode::ReadCoils,
            FunctionCode::ReadDiscreteInputs,
            FunctionCode::ReadHoldingRegisters,
            FunctionCode::ReadInputRegisters,
            FunctionCode::WriteSingleCoil,
            FunctionCode::WriteSingleRegister,
            FunctionCode::WriteMultipleCoils,
            FunctionCode::WriteMultipleRegisters,
            FunctionCode::ReadDeviceIdentification,
        ];
        for fc in codes {
            table.register(Box::new(EchoHandler { fc: fc.as_u8() }));
        }
        table
    }

    /// Register (or replace) a handler.
    pub fn register(&mut self, handler: Box<dyn FunctionCodeHandler>) {
        self.handlers.insert(handler.function_code(), handler);
    }

    /// Dispatch a request to its registered handler.
    ///
    /// If no handler is registered for `req.code`, an error response with
    /// Modbus exception code `0x01` (Illegal Function) is returned.
    pub fn dispatch(&self, req: &FunctionCodeRequest) -> FunctionCodeResponse {
        if let Some(handler) = self.handlers.get(&req.code) {
            handler.handle(req)
        } else {
            Self::error_response(req, 0x01) // Illegal Function
        }
    }

    /// Return a sorted list of all supported function code bytes.
    pub fn supported_codes(&self) -> Vec<u8> {
        let mut codes: Vec<u8> = self.handlers.keys().copied().collect();
        codes.sort_unstable();
        codes
    }

    /// `true` if a handler is registered for `code`.
    pub fn is_supported(&self, code: u8) -> bool {
        self.handlers.contains_key(&code)
    }

    /// Number of registered handlers.
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Build an error response: function code byte has bit 7 set, and the
    /// data contains the single `error_code` byte.
    pub fn error_response(req: &FunctionCodeRequest, error_code: u8) -> FunctionCodeResponse {
        FunctionCodeResponse {
            code: req.code | 0x80,
            data: vec![error_code],
            is_error: true,
        }
    }
}

// ---------------------------------------------------------------------------
// EchoHandler: returns request data unchanged
// ---------------------------------------------------------------------------

/// A simple handler that echoes the request data back as the response data.
/// Useful for testing and as a default placeholder.
pub struct EchoHandler {
    /// The function code this handler responds to.
    pub fc: u8,
}

impl FunctionCodeHandler for EchoHandler {
    fn handle(&self, req: &FunctionCodeRequest) -> FunctionCodeResponse {
        FunctionCodeResponse {
            code: req.code,
            data: req.data.clone(),
            is_error: false,
        }
    }

    fn function_code(&self) -> u8 {
        self.fc
    }

    fn description(&self) -> &str {
        "Echo handler (returns request data)"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn req(code: u8, data: &[u8]) -> FunctionCodeRequest {
        FunctionCodeRequest {
            code,
            data: data.to_vec(),
        }
    }

    // --- FunctionCode enum ---

    #[test]
    fn test_fc_read_coils_value() {
        assert_eq!(FunctionCode::ReadCoils.as_u8(), 0x01);
    }

    #[test]
    fn test_fc_read_discrete_inputs_value() {
        assert_eq!(FunctionCode::ReadDiscreteInputs.as_u8(), 0x02);
    }

    #[test]
    fn test_fc_read_holding_registers_value() {
        assert_eq!(FunctionCode::ReadHoldingRegisters.as_u8(), 0x03);
    }

    #[test]
    fn test_fc_read_input_registers_value() {
        assert_eq!(FunctionCode::ReadInputRegisters.as_u8(), 0x04);
    }

    #[test]
    fn test_fc_write_single_coil_value() {
        assert_eq!(FunctionCode::WriteSingleCoil.as_u8(), 0x05);
    }

    #[test]
    fn test_fc_write_single_register_value() {
        assert_eq!(FunctionCode::WriteSingleRegister.as_u8(), 0x06);
    }

    #[test]
    fn test_fc_write_multiple_coils_value() {
        assert_eq!(FunctionCode::WriteMultipleCoils.as_u8(), 0x0F);
    }

    #[test]
    fn test_fc_write_multiple_registers_value() {
        assert_eq!(FunctionCode::WriteMultipleRegisters.as_u8(), 0x10);
    }

    #[test]
    fn test_fc_read_device_identification_value() {
        assert_eq!(FunctionCode::ReadDeviceIdentification.as_u8(), 0x2B);
    }

    #[test]
    fn test_fc_from_u8_valid() {
        assert_eq!(FunctionCode::from_u8(0x01), Some(FunctionCode::ReadCoils));
        assert_eq!(
            FunctionCode::from_u8(0x10),
            Some(FunctionCode::WriteMultipleRegisters)
        );
    }

    #[test]
    fn test_fc_from_u8_invalid() {
        assert_eq!(FunctionCode::from_u8(0x00), None);
        assert_eq!(FunctionCode::from_u8(0xFF), None);
    }

    // --- DispatchTable ---

    #[test]
    fn test_dispatch_table_new_empty() {
        let table = DispatchTable::new();
        assert_eq!(table.handler_count(), 0);
    }

    #[test]
    fn test_with_defaults_has_nine_handlers() {
        let table = DispatchTable::with_defaults();
        assert_eq!(table.handler_count(), 9);
    }

    #[test]
    fn test_is_supported_true_for_defaults() {
        let table = DispatchTable::with_defaults();
        assert!(table.is_supported(0x01));
        assert!(table.is_supported(0x2B));
    }

    #[test]
    fn test_is_supported_false_for_unknown() {
        let table = DispatchTable::with_defaults();
        assert!(!table.is_supported(0x99));
    }

    #[test]
    fn test_supported_codes_sorted() {
        let table = DispatchTable::with_defaults();
        let codes = table.supported_codes();
        let mut sorted = codes.clone();
        sorted.sort_unstable();
        assert_eq!(codes, sorted);
    }

    #[test]
    fn test_dispatch_known_fc_returns_non_error() {
        let table = DispatchTable::with_defaults();
        let r = req(0x01, &[0x00, 0x00, 0x00, 0x10]);
        let resp = table.dispatch(&r);
        assert!(!resp.is_error);
        assert_eq!(resp.code, 0x01);
    }

    #[test]
    fn test_dispatch_unknown_fc_returns_error() {
        let table = DispatchTable::with_defaults();
        let r = req(0x99, &[]);
        let resp = table.dispatch(&r);
        assert!(resp.is_error);
        // Code should have bit 7 set: 0x99 | 0x80 = 0x99 (already set)
        assert_eq!(resp.code, 0x99 | 0x80);
    }

    #[test]
    fn test_error_response_sets_high_bit() {
        let r = req(0x03, &[0x00, 0x10]);
        let resp = DispatchTable::error_response(&r, 0x02);
        assert_eq!(resp.code, 0x83); // 0x03 | 0x80
        assert!(resp.is_error);
        assert_eq!(resp.data, vec![0x02]);
    }

    #[test]
    fn test_error_response_error_code_in_data() {
        let r = req(0x01, &[]);
        let resp = DispatchTable::error_response(&r, 0x03);
        assert_eq!(resp.data, vec![0x03]);
    }

    #[test]
    fn test_register_replaces_existing_handler() {
        let mut table = DispatchTable::new();
        table.register(Box::new(EchoHandler { fc: 0x01 }));
        assert_eq!(table.handler_count(), 1);

        // Register a different handler for the same fc
        struct AlwaysErrorHandler;
        impl FunctionCodeHandler for AlwaysErrorHandler {
            fn handle(&self, req: &FunctionCodeRequest) -> FunctionCodeResponse {
                DispatchTable::error_response(req, 0x04)
            }
            fn function_code(&self) -> u8 {
                0x01
            }
            fn description(&self) -> &str {
                "always error"
            }
        }
        table.register(Box::new(AlwaysErrorHandler));
        // Count should not increase
        assert_eq!(table.handler_count(), 1);

        let r = req(0x01, &[]);
        let resp = table.dispatch(&r);
        // The new handler produces an error
        assert!(resp.is_error);
    }

    // --- EchoHandler ---

    #[test]
    fn test_echo_handler_echoes_data() {
        let handler = EchoHandler { fc: 0x03 };
        let r = req(0x03, &[0xAA, 0xBB, 0xCC]);
        let resp = handler.handle(&r);
        assert_eq!(resp.data, vec![0xAA, 0xBB, 0xCC]);
        assert!(!resp.is_error);
    }

    #[test]
    fn test_echo_handler_function_code() {
        let handler = EchoHandler { fc: 0x06 };
        assert_eq!(handler.function_code(), 0x06);
    }

    #[test]
    fn test_echo_handler_description_non_empty() {
        let handler = EchoHandler { fc: 0x01 };
        assert!(!handler.description().is_empty());
    }

    #[test]
    fn test_echo_handler_empty_data() {
        let handler = EchoHandler { fc: 0x01 };
        let r = req(0x01, &[]);
        let resp = handler.handle(&r);
        assert!(resp.data.is_empty());
    }

    #[test]
    fn test_handler_count_after_multiple_registers() {
        let mut table = DispatchTable::new();
        table.register(Box::new(EchoHandler { fc: 0x01 }));
        table.register(Box::new(EchoHandler { fc: 0x02 }));
        table.register(Box::new(EchoHandler { fc: 0x03 }));
        assert_eq!(table.handler_count(), 3);
    }

    #[test]
    fn test_dispatch_fc0f_multiple_coils() {
        let table = DispatchTable::with_defaults();
        let r = req(0x0F, &[0x00, 0x10, 0x00, 0x03, 0x01, 0x05]);
        let resp = table.dispatch(&r);
        assert!(!resp.is_error);
        assert_eq!(resp.code, 0x0F);
    }

    #[test]
    fn test_dispatch_fc10_multiple_registers() {
        let table = DispatchTable::with_defaults();
        let r = req(
            0x10,
            &[0x00, 0x01, 0x00, 0x02, 0x04, 0x00, 0x0A, 0x01, 0x02],
        );
        let resp = table.dispatch(&r);
        assert!(!resp.is_error);
    }

    #[test]
    fn test_dispatch_fc2b_device_identification() {
        let table = DispatchTable::with_defaults();
        let r = req(0x2B, &[0x0E, 0x01, 0x00]);
        let resp = table.dispatch(&r);
        assert!(!resp.is_error);
        assert_eq!(resp.code, 0x2B);
    }

    #[test]
    fn test_default_dispatch_table() {
        let table = DispatchTable::default();
        assert_eq!(table.handler_count(), 0);
    }

    #[test]
    fn test_fc_name() {
        assert_eq!(FunctionCode::ReadCoils.name(), "ReadCoils");
        assert_eq!(
            FunctionCode::WriteMultipleRegisters.name(),
            "WriteMultipleRegisters"
        );
    }

    #[test]
    fn test_supported_codes_contains_all_default_fcs() {
        let table = DispatchTable::with_defaults();
        let codes = table.supported_codes();
        for fc in &[0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x0F, 0x10, 0x2B] {
            assert!(codes.contains(fc), "Missing FC 0x{:02X}", fc);
        }
    }
}
