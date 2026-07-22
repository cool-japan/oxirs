//! WASM error types

use thiserror::Error;
use wasm_bindgen::prelude::*;

/// WASM error type
#[derive(Error, Debug)]
pub enum WasmError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Query error: {0}")]
    QueryError(String),

    #[error("Store error: {0}")]
    StoreError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

impl WasmError {
    /// A stable, machine-readable name for this error's variant.
    ///
    /// This is set as both the `name` and `code` properties of the
    /// [`js_sys::Error`] built by `From<WasmError> for JsValue`, so a JS
    /// `catch` block can branch on error kind (`err.code === "ParseError"`)
    /// instead of pattern-matching the human-readable message text.
    pub fn code(&self) -> &'static str {
        match self {
            WasmError::ParseError(_) => "ParseError",
            WasmError::QueryError(_) => "QueryError",
            WasmError::StoreError(_) => "StoreError",
            WasmError::ValidationError(_) => "ValidationError",
            WasmError::SerializationError(_) => "SerializationError",
            WasmError::NotImplemented(_) => "NotImplemented",
        }
    }
}

/// Convert a [`WasmError`] into a real `js_sys::Error` (so JS sees
/// `instanceof Error` with a stack trace) carrying a stable `name`/`code` so
/// callers can distinguish a `ParseError` from a `StoreError` from a
/// `ValidationError` etc. without parsing the message string.
///
/// `js_sys::Error::new` calls into an imported JS binding that only exists
/// when actually running inside a `wasm32` + JS host — calling it from a
/// native (non-`wasm32`) test binary panics with "cannot call wasm-bindgen
/// imported functions on non-wasm targets". Since this crate's own design
/// keeps its Rust-level logic natively testable (see `api::wasm_api`'s module
/// doc), non-`wasm32` builds fall back to the plain string this conversion
/// used before, so `cargo test`/`cargo nextest` keeps working; only real
/// `wasm32` builds — the only place a JS `catch` block ever actually sees
/// this value — get the richer `js_sys::Error`.
impl From<WasmError> for JsValue {
    #[cfg(target_arch = "wasm32")]
    fn from(error: WasmError) -> Self {
        let code = error.code();
        let js_error = js_sys::Error::new(&error.to_string());
        js_error.set_name(code);
        // Best-effort: `Reflect::set` on a freshly constructed `Error` object
        // cannot fail in a spec-compliant JS engine, but we degrade gracefully
        // (still a proper Error, just without the extra `code` property)
        // rather than panicking if it somehow does.
        let _ = js_sys::Reflect::set(
            &js_error,
            &JsValue::from_str("code"),
            &JsValue::from_str(code),
        );
        js_error.into()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn from(error: WasmError) -> Self {
        JsValue::from_str(&error.to_string())
    }
}

pub type WasmResult<T> = Result<T, WasmError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_error_code_matches_variant() {
        assert_eq!(WasmError::ParseError("x".into()).code(), "ParseError");
        assert_eq!(WasmError::QueryError("x".into()).code(), "QueryError");
        assert_eq!(WasmError::StoreError("x".into()).code(), "StoreError");
        assert_eq!(
            WasmError::ValidationError("x".into()).code(),
            "ValidationError"
        );
        assert_eq!(
            WasmError::SerializationError("x".into()).code(),
            "SerializationError"
        );
        assert_eq!(
            WasmError::NotImplemented("x".into()).code(),
            "NotImplemented"
        );
    }
}
