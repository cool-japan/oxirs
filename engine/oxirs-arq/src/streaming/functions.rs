//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;
use anyhow::Result;
use tracing::{debug, warn};

use super::types::StreamStats;

/// Generic data stream interface
pub trait DataStream: Send + Sync {
    /// Get the next batch of data
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>>;
    /// Check if there is more data available
    fn has_more(&self) -> bool;
    /// Get estimated size of remaining data
    fn estimated_size(&self) -> Option<usize>;
    /// Reset the stream to the beginning
    fn reset(&mut self) -> Result<()>;
    /// Get stream statistics
    fn get_stats(&self) -> StreamStats;
}
/// Evaluate a literal as a boolean according to SPARQL Effective Boolean Value (EBV) semantics
///
/// According to SPARQL 1.1 specification:
/// - xsd:boolean "true" → true
/// - xsd:boolean "false" → false
/// - Other datatypes → error (but we return false for safety)
pub(super) fn evaluate_literal_as_boolean(literal: &crate::algebra::Literal) -> Result<bool> {
    if literal.is_boolean() {
        match literal.value.as_str() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => {
                warn!(
                    "Invalid boolean literal value: '{}', treating as false",
                    literal.value
                );
                Ok(false)
            }
        }
    } else {
        debug!(
            "Non-boolean literal in boolean context (datatype: {:?}), treating as false",
            literal.datatype
        );
        Ok(false)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{
        EmptyStream, MemoryMonitor, StreamingConfig, StreamingExecutor, StreamingUnion,
    };
    #[test]
    fn test_streaming_executor_creation() {
        let config = StreamingConfig::default();
        let executor = StreamingExecutor::new(config);
        assert!(executor.is_ok());
    }
    #[test]
    fn test_memory_monitor() {
        let monitor = MemoryMonitor::new(1000);
        assert!(monitor.allocate(500, "test"));
        assert_eq!(monitor.get_current_usage(), 500);
        assert!(monitor.allocate(400, "test2"));
        assert_eq!(monitor.get_current_usage(), 900);
        assert!(!monitor.allocate(200, "test3"));
        monitor.deallocate(400);
        assert_eq!(monitor.get_current_usage(), 500);
    }
    #[test]
    fn test_streaming_union() {
        let left = Box::new(EmptyStream::new());
        let right = Box::new(EmptyStream::new());
        let mut union = StreamingUnion::new(left, right);
        let result = union.next_batch().unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }
}
