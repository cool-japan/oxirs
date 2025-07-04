//! Main quantum stream processor

use super::algorithms::QuantumAlgorithmSuite;
use crate::error::{StreamError, StreamResult};
use crate::event::StreamEvent;

// Re-export the main types (they're implemented in types.rs)
pub use super::types::{QuantumEvent, QuantumProcessingStats, QuantumStreamProcessor};

/// Additional processor utilities could go here
impl QuantumStreamProcessor {
    /// Process multiple events in batch
    pub async fn process_batch(
        &mut self,
        events: Vec<StreamEvent>,
    ) -> StreamResult<Vec<QuantumEvent>> {
        let mut quantum_events = Vec::new();

        for event in events {
            // Convert StreamEvent to QuantumEvent
            let quantum_event = QuantumEvent {
                id: format!("q-{}", uuid::Uuid::new_v4()),
                timestamp: chrono::Utc::now().timestamp_millis() as u64,
                quantum_state: super::types::QuantumState::default(),
                operation: super::types::QuantumOperation::Hadamard,
                metadata: std::collections::HashMap::new(),
            };

            // Process the quantum event
            self.process_event(quantum_event.clone()).await?;
            quantum_events.push(quantum_event);
        }

        Ok(quantum_events)
    }
}
