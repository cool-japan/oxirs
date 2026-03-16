//! # WASM Error Handling Utilities
//!
//! Structured error types and an error-history handler for WebAssembly
//! applications built on OxiRS.

// ---------------------------------------------------------------------------
// ErrorCode
// ---------------------------------------------------------------------------

/// Numeric error classification for WASM clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ErrorCode {
    /// Malformed input (e.g. bad Turtle syntax).
    ParseError = 1,
    /// Network / HTTP failure.
    NetworkError = 2,
    /// Authentication or authorisation failure.
    AuthError = 3,
    /// Requested resource was not found.
    NotFound = 4,
    /// Internal server / engine error.
    ServerError = 5,
    /// Operation timed out.
    Timeout = 6,
    /// Caller supplied an invalid argument.
    InvalidInput = 7,
    /// Feature / operation not supported.
    Unsupported = 8,
}

impl ErrorCode {
    /// Convert from a raw `u32` discriminant.  Returns `None` for unknown codes.
    pub fn from_u32(n: u32) -> Option<Self> {
        match n {
            1 => Some(Self::ParseError),
            2 => Some(Self::NetworkError),
            3 => Some(Self::AuthError),
            4 => Some(Self::NotFound),
            5 => Some(Self::ServerError),
            6 => Some(Self::Timeout),
            7 => Some(Self::InvalidInput),
            8 => Some(Self::Unsupported),
            _ => None,
        }
    }

    /// Return the raw discriminant.
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::ParseError => "ParseError",
            Self::NetworkError => "NetworkError",
            Self::AuthError => "AuthError",
            Self::NotFound => "NotFound",
            Self::ServerError => "ServerError",
            Self::Timeout => "Timeout",
            Self::InvalidInput => "InvalidInput",
            Self::Unsupported => "Unsupported",
        };
        write!(f, "{name}")
    }
}

// ---------------------------------------------------------------------------
// WasmError
// ---------------------------------------------------------------------------

/// A structured WASM error.
#[derive(Debug, Clone)]
pub struct WasmError {
    /// Error category.
    pub code: ErrorCode,
    /// Human-readable message.
    pub message: String,
    /// Optional additional detail (stack trace, inner error, etc.).
    pub details: Option<String>,
    /// `true` if the caller may retry or recover; `false` for fatal errors.
    pub recoverable: bool,
}

impl WasmError {
    /// Create a new, recoverable error.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: None,
            recoverable: true,
        }
    }

    /// Attach additional detail text (builder pattern).
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Mark the error as unrecoverable (builder pattern).
    pub fn unrecoverable(mut self) -> Self {
        self.recoverable = false;
        self
    }

    /// Serialise to a compact JSON string.
    pub fn to_json(&self) -> String {
        let details_part = match &self.details {
            Some(d) => format!(r#","details":"{}""#, escape_json(d)),
            None => String::new(),
        };
        format!(
            r#"{{"code":{},"name":"{}","message":"{}","recoverable":{}{}}}"#,
            self.code.as_u32(),
            self.code,
            escape_json(&self.message),
            self.recoverable,
            details_part,
        )
    }

    /// Convenience constructor: attempt to build from a numeric code.
    ///
    /// Returns `None` when `code` does not map to a known `ErrorCode`.
    pub fn from_code(code: u32, message: &str) -> Option<Self> {
        ErrorCode::from_u32(code).map(|ec| Self::new(ec, message))
    }

    /// Returns `true` for "client-side" errors: codes 1, 4, 7, 8.
    pub fn is_client_error(&self) -> bool {
        matches!(
            self.code,
            ErrorCode::ParseError
                | ErrorCode::NotFound
                | ErrorCode::InvalidInput
                | ErrorCode::Unsupported
        )
    }

    /// Returns `true` for "server/transport" errors: codes 2, 5, 6.
    pub fn is_server_error(&self) -> bool {
        matches!(
            self.code,
            ErrorCode::NetworkError | ErrorCode::ServerError | ErrorCode::Timeout
        )
    }
}

impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for WasmError {}

// ---------------------------------------------------------------------------
// ErrorHandler
// ---------------------------------------------------------------------------

/// Accumulates `WasmError` values and provides diagnostic helpers.
pub struct ErrorHandler {
    history: Vec<WasmError>,
    max_history: usize,
}

impl ErrorHandler {
    /// Create a new handler that keeps at most `max_history` errors.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history: max_history.max(1),
        }
    }

    /// Record `error`, store it in history (evicting oldest when full), and
    /// return its JSON representation.
    pub fn handle(&mut self, error: WasmError) -> String {
        let json = error.to_json();
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(error);
        json
    }

    /// The most recently recorded error.
    pub fn last_error(&self) -> Option<&WasmError> {
        self.history.last()
    }

    /// Total number of errors currently in history.
    pub fn error_count(&self) -> usize {
        self.history.len()
    }

    /// `true` if any error in history is unrecoverable.
    pub fn has_unrecoverable(&self) -> bool {
        self.history.iter().any(|e| !e.recoverable)
    }

    /// Remove all errors from history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Errors matching the given numeric code.
    pub fn errors_by_code(&self, code: u32) -> Vec<&WasmError> {
        self.history
            .iter()
            .filter(|e| e.code.as_u32() == code)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

fn escape_json(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            '\r' => vec!['\\', 'r'],
            '\t' => vec!['\\', 't'],
            other => vec![other],
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ErrorCode
    // -------------------------------------------------------------------------
    #[test]
    fn test_error_code_as_u32() {
        assert_eq!(ErrorCode::ParseError.as_u32(), 1);
        assert_eq!(ErrorCode::NetworkError.as_u32(), 2);
        assert_eq!(ErrorCode::AuthError.as_u32(), 3);
        assert_eq!(ErrorCode::NotFound.as_u32(), 4);
        assert_eq!(ErrorCode::ServerError.as_u32(), 5);
        assert_eq!(ErrorCode::Timeout.as_u32(), 6);
        assert_eq!(ErrorCode::InvalidInput.as_u32(), 7);
        assert_eq!(ErrorCode::Unsupported.as_u32(), 8);
    }

    #[test]
    fn test_error_code_from_u32_valid() {
        assert_eq!(ErrorCode::from_u32(1), Some(ErrorCode::ParseError));
        assert_eq!(ErrorCode::from_u32(8), Some(ErrorCode::Unsupported));
    }

    #[test]
    fn test_error_code_from_u32_invalid() {
        assert_eq!(ErrorCode::from_u32(0), None);
        assert_eq!(ErrorCode::from_u32(9), None);
        assert_eq!(ErrorCode::from_u32(u32::MAX), None);
    }

    #[test]
    fn test_error_code_display() {
        assert_eq!(ErrorCode::ParseError.to_string(), "ParseError");
        assert_eq!(ErrorCode::Unsupported.to_string(), "Unsupported");
    }

    #[test]
    fn test_error_code_equality() {
        assert_eq!(ErrorCode::AuthError, ErrorCode::AuthError);
        assert_ne!(ErrorCode::AuthError, ErrorCode::NotFound);
    }

    #[test]
    fn test_error_code_all_round_trip() {
        for n in 1u32..=8 {
            let code = ErrorCode::from_u32(n).expect("valid code");
            assert_eq!(code.as_u32(), n);
        }
    }

    // -------------------------------------------------------------------------
    // WasmError::new
    // -------------------------------------------------------------------------
    #[test]
    fn test_wasm_error_new_code() {
        let e = WasmError::new(ErrorCode::ParseError, "bad syntax");
        assert_eq!(e.code, ErrorCode::ParseError);
    }

    #[test]
    fn test_wasm_error_new_message() {
        let e = WasmError::new(ErrorCode::NotFound, "not found");
        assert_eq!(e.message, "not found");
    }

    #[test]
    fn test_wasm_error_new_recoverable_default() {
        let e = WasmError::new(ErrorCode::ParseError, "x");
        assert!(e.recoverable);
    }

    #[test]
    fn test_wasm_error_new_no_details() {
        let e = WasmError::new(ErrorCode::ParseError, "x");
        assert!(e.details.is_none());
    }

    // -------------------------------------------------------------------------
    // WasmError builders
    // -------------------------------------------------------------------------
    #[test]
    fn test_with_details() {
        let e = WasmError::new(ErrorCode::ServerError, "msg").with_details("inner error");
        assert_eq!(e.details.as_deref(), Some("inner error"));
    }

    #[test]
    fn test_unrecoverable() {
        let e = WasmError::new(ErrorCode::ServerError, "fatal").unrecoverable();
        assert!(!e.recoverable);
    }

    #[test]
    fn test_builder_chaining() {
        let e = WasmError::new(ErrorCode::Timeout, "timed out")
            .with_details("after 30s")
            .unrecoverable();
        assert!(!e.recoverable);
        assert_eq!(e.details.as_deref(), Some("after 30s"));
    }

    // -------------------------------------------------------------------------
    // WasmError::from_code
    // -------------------------------------------------------------------------
    #[test]
    fn test_from_code_valid() {
        let e = WasmError::from_code(2, "network failure");
        assert!(e.is_some());
        let e = e.expect("should succeed");
        assert_eq!(e.code, ErrorCode::NetworkError);
    }

    #[test]
    fn test_from_code_invalid() {
        assert!(WasmError::from_code(99, "bad").is_none());
    }

    #[test]
    fn test_from_code_zero() {
        assert!(WasmError::from_code(0, "zero").is_none());
    }

    // -------------------------------------------------------------------------
    // is_client_error / is_server_error
    // -------------------------------------------------------------------------
    #[test]
    fn test_is_client_error_parse() {
        let e = WasmError::new(ErrorCode::ParseError, "x");
        assert!(e.is_client_error());
        assert!(!e.is_server_error());
    }

    #[test]
    fn test_is_client_error_not_found() {
        assert!(WasmError::new(ErrorCode::NotFound, "x").is_client_error());
    }

    #[test]
    fn test_is_client_error_invalid_input() {
        assert!(WasmError::new(ErrorCode::InvalidInput, "x").is_client_error());
    }

    #[test]
    fn test_is_client_error_unsupported() {
        assert!(WasmError::new(ErrorCode::Unsupported, "x").is_client_error());
    }

    #[test]
    fn test_is_server_error_network() {
        let e = WasmError::new(ErrorCode::NetworkError, "x");
        assert!(e.is_server_error());
        assert!(!e.is_client_error());
    }

    #[test]
    fn test_is_server_error_internal() {
        assert!(WasmError::new(ErrorCode::ServerError, "x").is_server_error());
    }

    #[test]
    fn test_is_server_error_timeout() {
        assert!(WasmError::new(ErrorCode::Timeout, "x").is_server_error());
    }

    #[test]
    fn test_auth_error_neither_client_nor_server() {
        let e = WasmError::new(ErrorCode::AuthError, "x");
        // AuthError (3) is neither client nor server per the spec
        assert!(!e.is_client_error());
        assert!(!e.is_server_error());
    }

    // -------------------------------------------------------------------------
    // WasmError::to_json
    // -------------------------------------------------------------------------
    #[test]
    fn test_to_json_contains_code() {
        let e = WasmError::new(ErrorCode::ParseError, "bad syntax");
        let json = e.to_json();
        assert!(json.contains("\"code\":1"), "json={json}");
    }

    #[test]
    fn test_to_json_contains_message() {
        let e = WasmError::new(ErrorCode::NotFound, "resource missing");
        let json = e.to_json();
        assert!(json.contains("resource missing"), "json={json}");
    }

    #[test]
    fn test_to_json_no_details_field_absent() {
        let e = WasmError::new(ErrorCode::ParseError, "x");
        let json = e.to_json();
        assert!(!json.contains("\"details\""), "json={json}");
    }

    #[test]
    fn test_to_json_with_details() {
        let e = WasmError::new(ErrorCode::ServerError, "crash").with_details("line 42");
        let json = e.to_json();
        assert!(json.contains("\"details\""), "json={json}");
        assert!(json.contains("line 42"), "json={json}");
    }

    #[test]
    fn test_to_json_escapes_quotes() {
        let e = WasmError::new(ErrorCode::ParseError, r#"He said "hello""#);
        let json = e.to_json();
        assert!(json.contains(r#"\""#), "json={json}");
    }

    #[test]
    fn test_to_json_recoverable_true() {
        let json = WasmError::new(ErrorCode::InvalidInput, "x").to_json();
        assert!(json.contains("\"recoverable\":true"), "json={json}");
    }

    #[test]
    fn test_to_json_recoverable_false() {
        let json = WasmError::new(ErrorCode::ServerError, "x")
            .unrecoverable()
            .to_json();
        assert!(json.contains("\"recoverable\":false"), "json={json}");
    }

    // -------------------------------------------------------------------------
    // ErrorHandler
    // -------------------------------------------------------------------------
    #[test]
    fn test_handler_new_empty() {
        let h = ErrorHandler::new(10);
        assert_eq!(h.error_count(), 0);
    }

    #[test]
    fn test_handler_handle_increments_count() {
        let mut h = ErrorHandler::new(10);
        h.handle(WasmError::new(ErrorCode::ParseError, "e1"));
        assert_eq!(h.error_count(), 1);
    }

    #[test]
    fn test_handler_handle_returns_json() {
        let mut h = ErrorHandler::new(10);
        let json = h.handle(WasmError::new(ErrorCode::NotFound, "missing"));
        assert!(json.contains("\"code\":4"), "json={json}");
    }

    #[test]
    fn test_handler_last_error() {
        let mut h = ErrorHandler::new(10);
        h.handle(WasmError::new(ErrorCode::ParseError, "first"));
        h.handle(WasmError::new(ErrorCode::NotFound, "second"));
        let last = h.last_error().expect("some last error");
        assert_eq!(last.message, "second");
    }

    #[test]
    fn test_handler_last_error_empty() {
        let h = ErrorHandler::new(10);
        assert!(h.last_error().is_none());
    }

    #[test]
    fn test_handler_eviction_when_full() {
        let mut h = ErrorHandler::new(3);
        for i in 0..5u32 {
            h.handle(WasmError::new(ErrorCode::ParseError, format!("e{i}")));
        }
        assert_eq!(h.error_count(), 3);
    }

    #[test]
    fn test_handler_has_unrecoverable_false() {
        let mut h = ErrorHandler::new(10);
        h.handle(WasmError::new(ErrorCode::ParseError, "x"));
        assert!(!h.has_unrecoverable());
    }

    #[test]
    fn test_handler_has_unrecoverable_true() {
        let mut h = ErrorHandler::new(10);
        h.handle(WasmError::new(ErrorCode::ServerError, "fatal").unrecoverable());
        assert!(h.has_unrecoverable());
    }

    #[test]
    fn test_handler_clear_history() {
        let mut h = ErrorHandler::new(10);
        h.handle(WasmError::new(ErrorCode::ParseError, "x"));
        h.clear_history();
        assert_eq!(h.error_count(), 0);
    }

    #[test]
    fn test_handler_errors_by_code() {
        let mut h = ErrorHandler::new(20);
        h.handle(WasmError::new(ErrorCode::ParseError, "p1"));
        h.handle(WasmError::new(ErrorCode::NotFound, "n1"));
        h.handle(WasmError::new(ErrorCode::ParseError, "p2"));
        let parse_errs = h.errors_by_code(1);
        assert_eq!(parse_errs.len(), 2);
    }

    #[test]
    fn test_handler_errors_by_code_none() {
        let h = ErrorHandler::new(10);
        assert!(h.errors_by_code(1).is_empty());
    }

    #[test]
    fn test_display_impl() {
        let e = WasmError::new(ErrorCode::Timeout, "took too long");
        let s = e.to_string();
        assert!(s.contains("Timeout"), "display={s}");
        assert!(s.contains("took too long"), "display={s}");
    }

    #[test]
    fn test_wasm_error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(WasmError::new(ErrorCode::ParseError, "x"));
        assert!(e.to_string().contains("ParseError"));
    }
}
