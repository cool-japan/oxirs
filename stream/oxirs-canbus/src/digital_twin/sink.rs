//! DTDL sink facade: trait definition and mock implementation.
//!
//! The [`DtdlSinkFacade`] trait abstracts over the DTDL property-write destination,
//! allowing a real Azure Digital Twins endpoint, an in-process twin store, or
//! (for tests) an in-memory mock to be used interchangeably in the bridge.

use oxirs_physics::digital_twin::twin_value::TwinValue;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that may be returned by a [`DtdlSinkFacade`] implementation.
#[derive(Debug, thiserror::Error)]
pub enum DtdlSinkError {
    /// The named property is not defined in the twin's DTDL schema.
    #[error("DTDL property not found: {0}")]
    NotFound(String),
    /// The value type does not match the property's DTDL schema type.
    #[error("DTDL type mismatch for property '{0}'")]
    TypeMismatch(String),
    /// An unexpected I/O or transport error occurred.
    #[error("DTDL sink I/O error: {0}")]
    Io(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// DtdlSinkFacade trait
// ─────────────────────────────────────────────────────────────────────────────

/// Asynchronous sink for DTDL digital-twin property updates.
///
/// Implementors may wrap an Azure Digital Twins REST client, a local
/// [`oxirs_physics::digital_twin::DigitalTwin`] instance, or (for tests) an
/// in-memory [`MockDtdlSink`].
///
/// # Contract
///
/// - `set_property` must be idempotent with respect to the same name+value pair.
/// - `get_property` returns `Ok(None)` when the property exists in the schema but
///   has not yet been written (rather than `Err(NotFound)`).
/// - Implementations are required to be `Send + Sync` so they can be shared
///   across async tasks.
#[async_trait::async_trait]
pub trait DtdlSinkFacade: Send + Sync {
    /// Write a named property.
    async fn set_property(&self, name: &str, value: TwinValue) -> Result<(), DtdlSinkError>;

    /// Read a named property.
    ///
    /// Returns `Ok(None)` when the property has not been written yet.
    async fn get_property(&self, name: &str) -> Result<Option<TwinValue>, DtdlSinkError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// MockDtdlSink
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory DTDL sink that records all property writes.
///
/// The inner map is guarded by an `Arc<Mutex<…>>` so the sink can be cloned and
/// shared between the bridge and test assertions without additional synchronisation
/// infrastructure.
///
/// # Example
///
/// ```
/// use oxirs_canbus::digital_twin::sink::{DtdlSinkFacade, MockDtdlSink};
/// use oxirs_physics::digital_twin::twin_value::TwinValue;
///
/// # #[tokio::main]
/// # async fn main() {
/// let sink = MockDtdlSink::new();
/// sink.set_property("engine.rpm", TwinValue::Integer(1500))
///     .await
///     .expect("set should succeed");
///
/// let val = sink.get("engine.rpm").expect("property should exist");
/// assert_eq!(val, TwinValue::Integer(1500));
/// # }
/// ```
#[derive(Debug, Default, Clone)]
pub struct MockDtdlSink {
    /// Shared property store (name → value).
    pub properties: Arc<Mutex<HashMap<String, TwinValue>>>,
}

impl MockDtdlSink {
    /// Create a new empty mock sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience accessor for tests: reads a property without async overhead.
    ///
    /// Returns `None` when the property has not been written.
    pub fn get(&self, name: &str) -> Option<TwinValue> {
        self.properties
            .lock()
            .ok()
            .and_then(|guard| guard.get(name).cloned())
    }

    /// Returns the number of distinct properties that have been written.
    pub fn len(&self) -> usize {
        self.properties.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Returns `true` when no properties have been written.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait::async_trait]
impl DtdlSinkFacade for MockDtdlSink {
    async fn set_property(&self, name: &str, value: TwinValue) -> Result<(), DtdlSinkError> {
        let mut guard = self
            .properties
            .lock()
            .map_err(|_| DtdlSinkError::Io("mutex poisoned".to_string()))?;
        guard.insert(name.to_string(), value);
        Ok(())
    }

    async fn get_property(&self, name: &str) -> Result<Option<TwinValue>, DtdlSinkError> {
        let guard = self
            .properties
            .lock()
            .map_err(|_| DtdlSinkError::Io("mutex poisoned".to_string()))?;
        Ok(guard.get(name).cloned())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_sink_set_and_get() {
        let sink = MockDtdlSink::new();
        sink.set_property("engine.coolant_temp_c", TwinValue::Integer(75))
            .await
            .expect("set should succeed");

        let val = sink
            .get_property("engine.coolant_temp_c")
            .await
            .expect("get should succeed");
        assert_eq!(val, Some(TwinValue::Integer(75)));
    }

    #[tokio::test]
    async fn mock_sink_get_missing_returns_none() {
        let sink = MockDtdlSink::new();
        let val = sink
            .get_property("nonexistent")
            .await
            .expect("get should succeed");
        assert!(val.is_none());
    }

    #[tokio::test]
    async fn mock_sink_overwrite_property() {
        let sink = MockDtdlSink::new();
        sink.set_property("x", TwinValue::Integer(1))
            .await
            .expect("first set");
        sink.set_property("x", TwinValue::Integer(2))
            .await
            .expect("second set");
        assert_eq!(sink.get("x"), Some(TwinValue::Integer(2)));
    }

    #[tokio::test]
    async fn mock_sink_clone_shares_state() {
        let sink = MockDtdlSink::new();
        let clone = sink.clone();
        sink.set_property("shared", TwinValue::Boolean(true))
            .await
            .expect("set should succeed");
        assert_eq!(clone.get("shared"), Some(TwinValue::Boolean(true)));
    }

    #[test]
    fn mock_sink_len_is_empty() {
        let sink = MockDtdlSink::new();
        assert!(sink.is_empty());
        assert_eq!(sink.len(), 0);
    }

    #[tokio::test]
    async fn mock_sink_len_increases_after_set() {
        let sink = MockDtdlSink::new();
        sink.set_property("a", TwinValue::Float(1.0))
            .await
            .expect("set");
        sink.set_property("b", TwinValue::Float(2.0))
            .await
            .expect("set");
        assert_eq!(sink.len(), 2);
    }
}
