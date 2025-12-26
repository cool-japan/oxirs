//! RDF Bridge for intelligent time-series detection
//!
//! This module provides heuristics and rules for automatically detecting
//! which RDF triples should be stored as time-series data.

use crate::error::{TsdbError, TsdbResult};
use chrono::DateTime;
use oxirs_core::model::{Object, Predicate, Quad, Subject};
use oxirs_core::RdfTerm;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Confidence level for time-series detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    /// Very low confidence (<20%)
    VeryLow,
    /// Low confidence (20-40%)
    Low,
    /// Medium confidence (40-60%)
    Medium,
    /// High confidence (60-80%)
    High,
    /// Very high confidence (>80%)
    VeryHigh,
}

impl Confidence {
    /// Get numeric confidence percentage
    pub fn percentage(&self) -> u8 {
        match self {
            Confidence::VeryLow => 10,
            Confidence::Low => 30,
            Confidence::Medium => 50,
            Confidence::High => 70,
            Confidence::VeryHigh => 90,
        }
    }

    /// Check if confidence is high enough for automatic routing
    pub fn is_sufficient(&self) -> bool {
        matches!(self, Confidence::High | Confidence::VeryHigh)
    }
}

/// Detection result
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Is this likely time-series data?
    pub is_timeseries: bool,
    /// Confidence level
    pub confidence: Confidence,
    /// Reason for the decision
    pub reason: String,
}

impl DetectionResult {
    /// Create a positive detection
    pub fn timeseries(confidence: Confidence, reason: String) -> Self {
        Self {
            is_timeseries: true,
            confidence,
            reason,
        }
    }

    /// Create a negative detection
    pub fn not_timeseries(reason: String) -> Self {
        Self {
            is_timeseries: false,
            confidence: Confidence::VeryHigh,
            reason,
        }
    }
}

/// RDF Bridge for auto-detection
#[derive(Debug)]
pub struct RdfBridge {
    /// Frequency tracker: (subject, predicate) → insert count
    frequency_tracker: Arc<RwLock<HashMap<(String, String), usize>>>,

    /// Frequency threshold for auto-routing (default: 10)
    frequency_threshold: usize,

    /// Known time-series predicates (QUDT, SOSA, etc.)
    known_ts_predicates: Vec<String>,

    /// Known metadata predicates (RDF Schema, Dublin Core, etc.)
    known_metadata_predicates: Vec<String>,
}

impl RdfBridge {
    /// Create a new RDF bridge
    pub fn new() -> Self {
        Self {
            frequency_tracker: Arc::new(RwLock::new(HashMap::new())),
            frequency_threshold: 10,
            known_ts_predicates: vec![
                "http://qudt.org/schema/qudt/numericValue".to_string(),
                "http://www.w3.org/ns/sosa/hasSimpleResult".to_string(),
                "http://www.w3.org/ns/sosa/hasResult".to_string(),
                "http://example.org/ts/value".to_string(),
            ],
            known_metadata_predicates: vec![
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                "http://www.w3.org/2000/01/rdf-schema#label".to_string(),
                "http://www.w3.org/2000/01/rdf-schema#comment".to_string(),
                "http://purl.org/dc/terms/title".to_string(),
                "http://purl.org/dc/terms/description".to_string(),
                "http://xmlns.com/foaf/0.1/name".to_string(),
            ],
        }
    }

    /// Set frequency threshold
    pub fn set_frequency_threshold(&mut self, threshold: usize) {
        self.frequency_threshold = threshold;
    }

    /// Detect if a quad should be stored as time-series
    ///
    /// Uses multiple heuristics:
    /// 1. Predicate-based (known TS predicates)
    /// 2. Value type analysis (numeric + timestamp)
    /// 3. Frequency-based (>N inserts → likely time-series)
    pub fn detect(&self, quad: &Quad) -> TsdbResult<DetectionResult> {
        // 1. Check known metadata predicates (early exit)
        if self.is_known_metadata_predicate(quad.predicate()) {
            return Ok(DetectionResult::not_timeseries(
                "Known metadata predicate".to_string(),
            ));
        }

        // 2. Check known time-series predicates
        if self.is_known_timeseries_predicate(quad.predicate()) {
            return Ok(DetectionResult::timeseries(
                Confidence::VeryHigh,
                "Known time-series predicate".to_string(),
            ));
        }

        // 3. Analyze value type
        let value_detection = self.analyze_value_type(quad.object())?;

        // If value type definitively says NOT time-series, trust that
        if !value_detection.is_timeseries && value_detection.confidence.is_sufficient() {
            return Ok(value_detection);
        }

        // 4. Check frequency (only if value type allows it)
        let frequency_detection = self.check_frequency(quad)?;

        // Combine detection results
        if value_detection.is_timeseries && frequency_detection.is_timeseries {
            // Both indicate time-series, use higher confidence
            if frequency_detection.confidence > value_detection.confidence {
                return Ok(frequency_detection);
            } else {
                return Ok(value_detection);
            }
        } else if value_detection.is_timeseries {
            // Value type says yes, even if frequency is low
            return Ok(value_detection);
        } else if frequency_detection.is_timeseries
            && frequency_detection.confidence.is_sufficient()
        {
            // High frequency overrides neutral value type
            return Ok(frequency_detection);
        }

        // Default: not time-series
        Ok(DetectionResult::not_timeseries(
            "No time-series indicators found".to_string(),
        ))
    }

    /// Check if predicate is a known time-series predicate
    fn is_known_timeseries_predicate(&self, predicate: &Predicate) -> bool {
        if let Predicate::NamedNode(node) = predicate {
            let iri = node.as_str();
            self.known_ts_predicates.contains(&iri.to_string())
        } else {
            false
        }
    }

    /// Check if predicate is a known metadata predicate
    fn is_known_metadata_predicate(&self, predicate: &Predicate) -> bool {
        if let Predicate::NamedNode(node) = predicate {
            let iri = node.as_str();
            self.known_metadata_predicates.contains(&iri.to_string())
        } else {
            false
        }
    }

    /// Analyze object value type
    fn analyze_value_type(&self, object: &Object) -> TsdbResult<DetectionResult> {
        match object {
            Object::Literal(lit) => {
                let value_str = lit.as_str();

                // Try to parse as number
                if value_str.parse::<f64>().is_ok() {
                    return Ok(DetectionResult::timeseries(
                        Confidence::Medium,
                        "Numeric literal value".to_string(),
                    ));
                }

                // Try to parse as timestamp
                if DateTime::parse_from_rfc3339(value_str).is_ok() {
                    return Ok(DetectionResult::timeseries(
                        Confidence::Low,
                        "Timestamp literal value".to_string(),
                    ));
                }

                Ok(DetectionResult::not_timeseries(
                    "Non-numeric literal".to_string(),
                ))
            }
            Object::NamedNode(_) | Object::BlankNode(_) => Ok(DetectionResult::not_timeseries(
                "Object is URI or blank node (not numeric)".to_string(),
            )),
            Object::Variable(_) | Object::QuotedTriple(_) => Ok(DetectionResult::not_timeseries(
                "Object is variable or quoted triple".to_string(),
            )),
        }
    }

    /// Check insertion frequency for subject-predicate pair
    fn check_frequency(&self, quad: &Quad) -> TsdbResult<DetectionResult> {
        let subject_str = match quad.subject() {
            Subject::NamedNode(node) => node.as_str().to_string(),
            Subject::BlankNode(blank) => format!("_:{}", blank.as_str()),
            Subject::Variable(var) => format!("?{}", var.as_str()),
            Subject::QuotedTriple(triple) => format!("<<{:?}>>", triple),
        };

        let predicate_str = match quad.predicate() {
            Predicate::NamedNode(node) => node.as_str().to_string(),
            Predicate::Variable(var) => format!("?{}", var.as_str()),
        };

        let key = (subject_str, predicate_str);

        // Increment counter
        let count = {
            let mut tracker = self
                .frequency_tracker
                .write()
                .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;
            let entry = tracker.entry(key).or_insert(0);
            *entry += 1;
            *entry
        };

        // Check against threshold
        if count >= self.frequency_threshold {
            Ok(DetectionResult::timeseries(
                Confidence::High,
                format!("High insertion frequency: {count} inserts"),
            ))
        } else {
            Ok(DetectionResult::timeseries(
                Confidence::VeryLow,
                format!("Low insertion frequency: {count} inserts"),
            ))
        }
    }

    /// Get frequency statistics
    pub fn get_frequency_stats(&self) -> TsdbResult<Vec<((String, String), usize)>> {
        let tracker = self
            .frequency_tracker
            .read()
            .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;
        Ok(tracker.iter().map(|(k, v)| (k.clone(), *v)).collect())
    }

    /// Reset frequency tracker
    pub fn reset_frequency_tracker(&self) -> TsdbResult<()> {
        let mut tracker = self
            .frequency_tracker
            .write()
            .map_err(|e| TsdbError::Integration(format!("Lock poisoned: {e}")))?;
        tracker.clear();
        Ok(())
    }

    /// Add custom time-series predicate
    pub fn add_timeseries_predicate(&mut self, predicate_iri: String) {
        if !self.known_ts_predicates.contains(&predicate_iri) {
            self.known_ts_predicates.push(predicate_iri);
        }
    }

    /// Add custom metadata predicate
    pub fn add_metadata_predicate(&mut self, predicate_iri: String) {
        if !self.known_metadata_predicates.contains(&predicate_iri) {
            self.known_metadata_predicates.push(predicate_iri);
        }
    }
}

impl Default for RdfBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode};

    fn create_quad(subject: &str, predicate: &str, object: &str) -> Quad {
        let s = NamedNode::new(subject).unwrap();
        let p = NamedNode::new(predicate).unwrap();
        let o = Literal::new(object);
        Quad::new(s, p, o, oxirs_core::model::GraphName::DefaultGraph)
    }

    #[test]
    fn test_known_timeseries_predicate() -> TsdbResult<()> {
        let bridge = RdfBridge::new();
        let quad = create_quad(
            "http://example.org/sensor1",
            "http://qudt.org/schema/qudt/numericValue",
            "42.5",
        );

        let result = bridge.detect(&quad)?;
        assert!(result.is_timeseries);
        assert_eq!(result.confidence, Confidence::VeryHigh);
        assert!(result.reason.contains("Known time-series predicate"));

        Ok(())
    }

    #[test]
    fn test_known_metadata_predicate() -> TsdbResult<()> {
        let bridge = RdfBridge::new();
        let quad = create_quad(
            "http://example.org/sensor1",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://www.w3.org/ns/sosa/Sensor",
        );

        let result = bridge.detect(&quad)?;
        assert!(!result.is_timeseries);
        assert!(result.reason.contains("Known metadata predicate"));

        Ok(())
    }

    #[test]
    fn test_numeric_value_detection() -> TsdbResult<()> {
        let bridge = RdfBridge::new();
        let quad = create_quad(
            "http://example.org/sensor1",
            "http://example.org/customValue",
            "123.45",
        );

        let result = bridge.detect(&quad)?;
        assert!(result.is_timeseries);
        assert_eq!(result.confidence, Confidence::Medium);
        assert!(result.reason.contains("Numeric literal"));

        Ok(())
    }

    #[test]
    fn test_non_numeric_value() -> TsdbResult<()> {
        let bridge = RdfBridge::new();
        let quad = create_quad(
            "http://example.org/sensor1",
            "http://example.org/name",
            "Temperature Sensor",
        );

        let result = bridge.detect(&quad)?;
        assert!(!result.is_timeseries);

        Ok(())
    }

    #[test]
    fn test_frequency_based_detection() -> TsdbResult<()> {
        let mut bridge = RdfBridge::new();
        bridge.set_frequency_threshold(5);

        let quad = create_quad(
            "http://example.org/sensor1",
            "http://example.org/reading",
            "10.0",
        );

        // First few inserts: low confidence
        for _ in 0..4 {
            let result = bridge.detect(&quad)?;
            assert!(!result.confidence.is_sufficient());
        }

        // After threshold: high confidence
        let result = bridge.detect(&quad)?;
        assert!(result.is_timeseries);
        assert_eq!(result.confidence, Confidence::High);
        assert!(result.reason.contains("High insertion frequency"));

        Ok(())
    }

    #[test]
    fn test_custom_predicate_registration() -> TsdbResult<()> {
        let mut bridge = RdfBridge::new();
        bridge.add_timeseries_predicate("http://example.org/custom/measurement".to_string());

        let quad = create_quad(
            "http://example.org/sensor1",
            "http://example.org/custom/measurement",
            "42.0",
        );

        let result = bridge.detect(&quad)?;
        assert!(result.is_timeseries);
        assert_eq!(result.confidence, Confidence::VeryHigh);

        Ok(())
    }

    #[test]
    fn test_frequency_stats() -> TsdbResult<()> {
        let bridge = RdfBridge::new();

        let quad1 = create_quad(
            "http://example.org/sensor1",
            "http://example.org/reading",
            "10.0",
        );
        let quad2 = create_quad(
            "http://example.org/sensor2",
            "http://example.org/reading",
            "20.0",
        );

        // Insert multiple times
        for _ in 0..3 {
            bridge.detect(&quad1)?;
        }
        for _ in 0..5 {
            bridge.detect(&quad2)?;
        }

        let stats = bridge.get_frequency_stats()?;
        assert_eq!(stats.len(), 2);

        Ok(())
    }

    #[test]
    fn test_reset_frequency_tracker() -> TsdbResult<()> {
        let bridge = RdfBridge::new();

        let quad = create_quad(
            "http://example.org/sensor1",
            "http://example.org/reading",
            "10.0",
        );

        // Insert multiple times
        for _ in 0..5 {
            bridge.detect(&quad)?;
        }

        let stats_before = bridge.get_frequency_stats()?;
        assert!(!stats_before.is_empty());

        bridge.reset_frequency_tracker()?;

        let stats_after = bridge.get_frequency_stats()?;
        assert!(stats_after.is_empty());

        Ok(())
    }

    #[test]
    fn test_confidence_levels() {
        assert_eq!(Confidence::VeryLow.percentage(), 10);
        assert_eq!(Confidence::Low.percentage(), 30);
        assert_eq!(Confidence::Medium.percentage(), 50);
        assert_eq!(Confidence::High.percentage(), 70);
        assert_eq!(Confidence::VeryHigh.percentage(), 90);

        assert!(!Confidence::VeryLow.is_sufficient());
        assert!(!Confidence::Low.is_sufficient());
        assert!(!Confidence::Medium.is_sufficient());
        assert!(Confidence::High.is_sufficient());
        assert!(Confidence::VeryHigh.is_sufficient());
    }

    #[test]
    fn test_timestamp_detection() -> TsdbResult<()> {
        let bridge = RdfBridge::new();
        let quad = create_quad(
            "http://example.org/sensor1",
            "http://example.org/timestamp",
            "2024-01-01T00:00:00Z",
        );

        let result = bridge.detect(&quad)?;
        assert!(result.is_timeseries);
        assert_eq!(result.confidence, Confidence::Low);
        assert!(result.reason.contains("Timestamp literal"));

        Ok(())
    }
}
