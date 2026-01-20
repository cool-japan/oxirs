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

impl From<WasmError> for JsValue {
    fn from(error: WasmError) -> Self {
        JsValue::from_str(&error.to_string())
    }
}

pub type WasmResult<T> = Result<T, WasmError>;
