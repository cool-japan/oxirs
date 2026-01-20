//! SWRL (Semantic Web Rule Language) - Temporal Extensions
//!
//! This module implements SWRL rule components.

use anyhow::Result;

/// Temporal interval representation
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalInterval {
    pub start: f64,
    pub end: f64,
}

impl TemporalInterval {
    pub fn new(start: f64, end: f64) -> Result<Self> {
        if start > end {
            return Err(anyhow::anyhow!(
                "Invalid interval: start ({}) > end ({})",
                start,
                end
            ));
        }
        Ok(Self { start, end })
    }

    /// Check if this interval is before another interval
    pub fn before(&self, other: &TemporalInterval) -> bool {
        self.end < other.start
    }

    /// Check if this interval is after another interval
    pub fn after(&self, other: &TemporalInterval) -> bool {
        self.start > other.end
    }

    /// Check if this interval overlaps with another interval
    pub fn overlaps(&self, other: &TemporalInterval) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Check if this interval meets another interval
    pub fn meets(&self, other: &TemporalInterval) -> bool {
        (self.end - other.start).abs() < f64::EPSILON
    }

    /// Check if this interval contains a time point
    pub fn contains(&self, time: f64) -> bool {
        time >= self.start && time <= self.end
    }

    /// Get the duration of this interval
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}
