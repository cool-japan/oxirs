//! Advanced Event Processing Module
//!
//! This module provides sophisticated event processing capabilities including:
//! - Time-based and count-based windowing
//! - Streaming aggregations (count, sum, average, min, max)
//! - Complex event pattern detection
//! - Event correlation and causality tracking
//! - Real-time analytics and metrics computation
//!
//! The module is organized into several sub-modules:
//! - `window`: Window management and windowing logic
//! - `aggregation`: Aggregation functions and state management
//! - `processor`: Main event processing engine
//! - `complex`: Complex event processing patterns
//! - `analytics`: Real-time analytics and metrics
//! - `anomaly`: Anomaly detection capabilities
//! - `temporal`: Advanced temporal processing
//! - `causality`: Causality analysis and correlation

pub mod aggregation;
pub mod processor;
pub mod window;

// Re-export commonly used types
pub use aggregation::{AggregateFunction, AggregationManager, AggregationState};
pub use processor::{EventProcessor, ProcessorConfig, ProcessorStats};
pub use window::{EventWindow, Watermark, WindowConfig, WindowResult, WindowTrigger, WindowType};

// TODO: The following modules would be created in subsequent refactoring steps:
// pub mod complex;      // Complex event processing
// pub mod analytics;    // Real-time analytics
// pub mod anomaly;      // Anomaly detection
// pub mod temporal;     // Temporal processing
// pub mod causality;    // Causality analysis

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamEvent;
    use chrono::{Duration as ChronoDuration, Utc};

    #[test]
    fn test_window_creation() {
        let config = WindowConfig {
            window_type: WindowType::Tumbling {
                duration: ChronoDuration::seconds(10),
            },
            aggregates: vec![AggregateFunction::Count],
            group_by: vec![],
            filter: None,
            allow_lateness: None,
            trigger: WindowTrigger::OnTime,
        };

        let window = EventWindow::new(config);
        assert!(!window.id().is_empty());
        assert_eq!(window.event_count(), 0);
    }

    #[test]
    fn test_processor_creation() {
        let processor = EventProcessor::default();
        assert_eq!(processor.active_windows().len(), 0);
    }

    #[test]
    fn test_aggregation_manager() {
        let mut manager = AggregationManager::new();
        manager.add_aggregation("count".to_string(), AggregateFunction::Count);

        let event = StreamEvent::TripleAdded {
            subject: "test".to_string(),
            predicate: "hasValue".to_string(),
            object: "42".to_string(),
            graph: None,
            metadata: crate::event::EventMetadata::default(),
        };

        assert!(manager.update(&event).is_ok());
        let results = manager.results().unwrap();
        assert!(results.contains_key("count"));
    }

    #[test]
    fn test_watermark_update() {
        let mut watermark = Watermark::new();
        let initial_time = watermark.current();

        let future_time = initial_time + ChronoDuration::seconds(10);
        watermark.update(future_time);

        assert_eq!(watermark.current(), future_time);
    }

    #[test]
    fn test_window_trigger_conditions() {
        let config = WindowConfig {
            window_type: WindowType::CountBased { size: 5 },
            aggregates: vec![AggregateFunction::Count],
            group_by: vec![],
            filter: None,
            allow_lateness: None,
            trigger: WindowTrigger::OnCount(3),
        };

        let mut window = EventWindow::new(config);

        // Add events
        for i in 0..3 {
            let event = StreamEvent::TripleAdded {
                subject: format!("test_{i}"),
                predicate: "hasValue".to_string(),
                object: i.to_string(),
                graph: None,
                metadata: crate::event::EventMetadata::default(),
            };
            window.add_event(event).unwrap();
        }

        assert!(window.should_trigger(Utc::now()));
    }
}
