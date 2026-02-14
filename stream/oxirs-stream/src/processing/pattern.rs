//! Complex Event Pattern Matching
//!
//! This module provides sophisticated pattern matching capabilities for stream processing:
//! - Sequence patterns (A followed by B)
//! - Conjunction patterns (A and B)
//! - Disjunction patterns (A or B)
//! - Negation patterns (A not followed by B)
//! - Temporal constraints (within time window)
//! - Statistical patterns (frequency, correlation)
//!
//! Uses SciRS2 for statistical analysis and pattern detection

use crate::StreamEvent;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

// Use scirs2-core for statistical operations
use scirs2_core::ndarray_ext::{s, Array1};

/// Parameters for repeat pattern matching
#[derive(Debug, Clone)]
struct RepeatMatchParams<'a> {
    pattern: &'a Pattern,
    min_count: usize,
    max_count: Option<usize>,
    time_window: &'a ChronoDuration,
}

/// Parameters for statistical pattern matching
#[derive(Debug, Clone)]
struct StatisticalMatchParams<'a> {
    name: &'a str,
    stat_type: &'a StatisticalPatternType,
    threshold: f64,
    time_window: &'a ChronoDuration,
}

/// Pattern matching strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternMatchStrategy {
    /// Match any occurrence
    Any,
    /// Match all occurrences
    All,
    /// Match first occurrence only
    First,
    /// Match last occurrence only
    Last,
    /// Match with maximum score
    BestMatch,
}

/// Pattern definition for complex event processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Pattern {
    /// Single event pattern with predicate
    Simple { name: String, predicate: String },
    /// Sequence pattern (A followed by B)
    Sequence {
        patterns: Vec<Pattern>,
        max_distance: Option<ChronoDuration>,
    },
    /// Conjunction pattern (A and B occur together)
    And {
        patterns: Vec<Pattern>,
        time_window: ChronoDuration,
    },
    /// Disjunction pattern (A or B occurs)
    Or { patterns: Vec<Pattern> },
    /// Negation pattern (A occurs without B)
    Not {
        positive: Box<Pattern>,
        negative: Box<Pattern>,
        time_window: ChronoDuration,
    },
    /// Repetition pattern (A occurs N times)
    Repeat {
        pattern: Box<Pattern>,
        min_count: usize,
        max_count: Option<usize>,
        time_window: ChronoDuration,
    },
    /// Statistical pattern (correlation, frequency)
    Statistical {
        name: String,
        stat_type: StatisticalPatternType,
        threshold: f64,
        time_window: ChronoDuration,
    },
}

/// Statistical pattern types using SciRS2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalPatternType {
    /// Frequency above threshold
    Frequency,
    /// Correlation between events
    Correlation { field_a: String, field_b: String },
    /// Moving average
    MovingAverage { field: String, window_size: usize },
    /// Standard deviation
    StdDev { field: String },
    /// Anomaly detection
    Anomaly { field: String, sensitivity: f64 },
}

/// Pattern match result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern_id: String,
    pub pattern_name: String,
    pub events: Vec<StreamEvent>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub confidence: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Pattern matcher state
pub struct PatternMatcher {
    patterns: HashMap<String, Pattern>,
    active_matches: HashMap<String, Vec<PartialMatch>>,
    completed_matches: VecDeque<PatternMatch>,
    event_buffer: VecDeque<(StreamEvent, DateTime<Utc>)>,
    buffer_size: usize,
    stats: PatternMatcherStats,
}

#[derive(Debug, Clone)]
struct PartialMatch {
    pattern_id: String,
    matched_events: Vec<StreamEvent>,
    start_time: DateTime<Utc>,
    current_state: usize,
}

#[derive(Debug, Clone, Default)]
pub struct PatternMatcherStats {
    pub events_processed: u64,
    pub patterns_matched: u64,
    pub partial_matches: u64,
    pub timeouts: u64,
    pub processing_time_ms: f64,
}

impl PatternMatcher {
    /// Create a new pattern matcher
    pub fn new(buffer_size: usize) -> Self {
        Self {
            patterns: HashMap::new(),
            active_matches: HashMap::new(),
            completed_matches: VecDeque::new(),
            event_buffer: VecDeque::new(),
            buffer_size,
            stats: PatternMatcherStats::default(),
        }
    }

    /// Register a pattern
    pub fn register_pattern(&mut self, pattern: Pattern) -> String {
        let pattern_id = Uuid::new_v4().to_string();
        self.patterns.insert(pattern_id.clone(), pattern);
        pattern_id
    }

    /// Process an event through all patterns
    pub fn process_event(&mut self, event: StreamEvent) -> Result<Vec<PatternMatch>> {
        let start = std::time::Instant::now();
        let now = Utc::now();

        self.stats.events_processed += 1;

        // Add to event buffer
        self.event_buffer.push_back((event.clone(), now));
        if self.event_buffer.len() > self.buffer_size {
            self.event_buffer.pop_front();
        }

        // Check all registered patterns
        let mut new_matches = Vec::new();

        // Clone pattern list to avoid borrowing issues
        let patterns: Vec<(String, Pattern)> = self
            .patterns
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        for (pattern_id, pattern) in patterns {
            match self.match_pattern(&pattern_id, &pattern, &event, now) {
                Ok(matches) => new_matches.extend(matches),
                Err(e) => tracing::warn!("Pattern matching error for {}: {}", pattern_id, e),
            }
        }

        // Clean up expired partial matches
        self.cleanup_expired_matches(now);

        // Update statistics
        self.stats.processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;
        self.stats.patterns_matched += new_matches.len() as u64;

        Ok(new_matches)
    }

    /// Match a pattern against the event
    fn match_pattern(
        &mut self,
        pattern_id: &str,
        pattern: &Pattern,
        event: &StreamEvent,
        now: DateTime<Utc>,
    ) -> Result<Vec<PatternMatch>> {
        match pattern {
            Pattern::Simple { name, predicate } => {
                self.match_simple_pattern(pattern_id, name, predicate, event, now)
            }
            Pattern::Sequence {
                patterns,
                max_distance,
            } => self.match_sequence_pattern(pattern_id, patterns, max_distance, event, now),
            Pattern::And {
                patterns,
                time_window,
            } => self.match_and_pattern(pattern_id, patterns, time_window, event, now),
            Pattern::Or { patterns } => self.match_or_pattern(pattern_id, patterns, event, now),
            Pattern::Not {
                positive,
                negative,
                time_window,
            } => self.match_not_pattern(pattern_id, positive, negative, time_window, event, now),
            Pattern::Repeat {
                pattern,
                min_count,
                max_count,
                time_window,
            } => self.match_repeat_pattern(
                pattern_id,
                RepeatMatchParams {
                    pattern,
                    min_count: *min_count,
                    max_count: *max_count,
                    time_window,
                },
                event,
                now,
            ),
            Pattern::Statistical {
                name,
                stat_type,
                threshold,
                time_window,
            } => self.match_statistical_pattern(
                pattern_id,
                StatisticalMatchParams {
                    name,
                    stat_type,
                    threshold: *threshold,
                    time_window,
                },
                event,
                now,
            ),
        }
    }

    /// Match simple pattern
    fn match_simple_pattern(
        &mut self,
        pattern_id: &str,
        name: &str,
        predicate: &str,
        event: &StreamEvent,
        now: DateTime<Utc>,
    ) -> Result<Vec<PatternMatch>> {
        if self.evaluate_predicate(predicate, event)? {
            Ok(vec![PatternMatch {
                pattern_id: pattern_id.to_string(),
                pattern_name: name.to_string(),
                events: vec![event.clone()],
                start_time: now,
                end_time: now,
                confidence: 1.0,
                metadata: HashMap::new(),
            }])
        } else {
            Ok(vec![])
        }
    }

    /// Match sequence pattern
    fn match_sequence_pattern(
        &mut self,
        pattern_id: &str,
        patterns: &[Pattern],
        max_distance: &Option<ChronoDuration>,
        event: &StreamEvent,
        now: DateTime<Utc>,
    ) -> Result<Vec<PatternMatch>> {
        let mut matches = Vec::new();

        // Clone existing partial matches to avoid borrowing issues
        let existing_partials = self
            .active_matches
            .get(pattern_id)
            .cloned()
            .unwrap_or_default();

        // Try to advance existing partial matches
        let mut new_partial_matches = Vec::new();

        for partial in existing_partials.iter() {
            if partial.current_state < patterns.len() {
                let next_pattern = &patterns[partial.current_state];

                // Check if event matches next pattern using simple predicate evaluation
                let matches_next = self.evaluate_pattern_simple(next_pattern, event)?;

                if matches_next {
                    // Check time distance constraint
                    if let Some(max_dist) = max_distance {
                        if now - partial.start_time > *max_dist {
                            continue;
                        }
                    }

                    let mut new_events = partial.matched_events.clone();
                    new_events.push(event.clone());

                    if partial.current_state + 1 == patterns.len() {
                        // Complete match
                        matches.push(PatternMatch {
                            pattern_id: pattern_id.to_string(),
                            pattern_name: "Sequence".to_string(),
                            events: new_events,
                            start_time: partial.start_time,
                            end_time: now,
                            confidence: 1.0,
                            metadata: HashMap::new(),
                        });
                    } else {
                        // Continue matching
                        new_partial_matches.push(PartialMatch {
                            pattern_id: pattern_id.to_string(),
                            matched_events: new_events,
                            start_time: partial.start_time,
                            current_state: partial.current_state + 1,
                        });
                    }
                }
            }
        }

        // Start new partial matches
        if !patterns.is_empty() {
            let first_pattern = &patterns[0];
            let matches_first = self.evaluate_pattern_simple(first_pattern, event)?;

            if matches_first {
                if patterns.len() == 1 {
                    // Single pattern sequence - immediate match
                    matches.push(PatternMatch {
                        pattern_id: pattern_id.to_string(),
                        pattern_name: "Sequence".to_string(),
                        events: vec![event.clone()],
                        start_time: now,
                        end_time: now,
                        confidence: 1.0,
                        metadata: HashMap::new(),
                    });
                } else {
                    new_partial_matches.push(PartialMatch {
                        pattern_id: pattern_id.to_string(),
                        matched_events: vec![event.clone()],
                        start_time: now,
                        current_state: 1,
                    });
                }
            }
        }

        // Update active matches
        self.active_matches
            .insert(pattern_id.to_string(), new_partial_matches.clone());
        self.stats.partial_matches = new_partial_matches.len() as u64;

        Ok(matches)
    }

    /// Match AND pattern (all patterns must match within time window)
    fn match_and_pattern(
        &mut self,
        pattern_id: &str,
        patterns: &[Pattern],
        time_window: &ChronoDuration,
        _event: &StreamEvent,
        now: DateTime<Utc>,
    ) -> Result<Vec<PatternMatch>> {
        // Collect events within time window
        let window_start = now - *time_window;
        let recent_events: Vec<_> = self
            .event_buffer
            .iter()
            .filter(|(_, timestamp)| *timestamp >= window_start)
            .cloned()
            .collect();

        // Check if all patterns match within the window
        let mut all_matched = true;
        let mut matched_events = Vec::new();

        for pattern in patterns {
            let mut pattern_matched = false;

            for (evt, evt_time) in &recent_events {
                let sub_matches = self.match_pattern(pattern_id, pattern, evt, *evt_time)?;

                if !sub_matches.is_empty() {
                    pattern_matched = true;
                    matched_events.push(evt.clone());
                    break;
                }
            }

            if !pattern_matched {
                all_matched = false;
                break;
            }
        }

        if all_matched && !matched_events.is_empty() {
            Ok(vec![PatternMatch {
                pattern_id: pattern_id.to_string(),
                pattern_name: "And".to_string(),
                events: matched_events,
                start_time: window_start,
                end_time: now,
                confidence: 1.0,
                metadata: HashMap::new(),
            }])
        } else {
            Ok(vec![])
        }
    }

    /// Match OR pattern (any pattern matches)
    fn match_or_pattern(
        &mut self,
        pattern_id: &str,
        patterns: &[Pattern],
        event: &StreamEvent,
        now: DateTime<Utc>,
    ) -> Result<Vec<PatternMatch>> {
        for pattern in patterns {
            let matches = self.match_pattern(pattern_id, pattern, event, now)?;
            if !matches.is_empty() {
                return Ok(matches);
            }
        }

        Ok(vec![])
    }

    /// Match NOT pattern (positive without negative)
    fn match_not_pattern(
        &mut self,
        pattern_id: &str,
        positive: &Pattern,
        negative: &Pattern,
        time_window: &ChronoDuration,
        event: &StreamEvent,
        now: DateTime<Utc>,
    ) -> Result<Vec<PatternMatch>> {
        // Check if positive pattern matches
        let positive_matches = self.match_pattern(pattern_id, positive, event, now)?;

        if positive_matches.is_empty() {
            return Ok(vec![]);
        }

        // Check if negative pattern matches within time window
        let window_start = now - *time_window;
        let recent_events: Vec<_> = self
            .event_buffer
            .iter()
            .filter(|(_, timestamp)| *timestamp >= window_start)
            .cloned()
            .collect();

        for (evt, evt_time) in recent_events {
            let negative_matches = self.match_pattern(pattern_id, negative, &evt, evt_time)?;

            if !negative_matches.is_empty() {
                // Negative pattern matched - no match
                return Ok(vec![]);
            }
        }

        // Positive matched, negative didn't - success
        Ok(positive_matches)
    }

    /// Match repeat pattern
    fn match_repeat_pattern(
        &mut self,
        pattern_id: &str,
        params: RepeatMatchParams,
        _event: &StreamEvent,
        now: DateTime<Utc>,
    ) -> Result<Vec<PatternMatch>> {
        // Collect matching events within time window
        let window_start = now - *params.time_window;
        let mut matched_events = Vec::new();

        // Clone event buffer to avoid borrowing issues
        let buffer_clone: Vec<(StreamEvent, DateTime<Utc>)> =
            self.event_buffer.iter().cloned().collect();

        for (evt, evt_time) in buffer_clone {
            if evt_time >= window_start {
                let matches = self.evaluate_pattern_simple(params.pattern, &evt)?;

                if matches {
                    matched_events.push(evt.clone());
                }
            }
        }

        let match_count = matched_events.len();

        if match_count >= params.min_count
            && params.max_count.map_or(true, |max| match_count <= max)
        {
            Ok(vec![PatternMatch {
                pattern_id: pattern_id.to_string(),
                pattern_name: "Repeat".to_string(),
                events: matched_events,
                start_time: window_start,
                end_time: now,
                confidence: 1.0,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert(
                        "repeat_count".to_string(),
                        serde_json::Value::Number(match_count.into()),
                    );
                    meta
                },
            }])
        } else {
            Ok(vec![])
        }
    }

    /// Match statistical pattern using SciRS2
    fn match_statistical_pattern(
        &mut self,
        pattern_id: &str,
        params: StatisticalMatchParams,
        _event: &StreamEvent,
        now: DateTime<Utc>,
    ) -> Result<Vec<PatternMatch>> {
        let window_start = now - *params.time_window;
        let recent_events: Vec<_> = self
            .event_buffer
            .iter()
            .filter(|(_, timestamp)| *timestamp >= window_start)
            .map(|(evt, _)| evt)
            .cloned()
            .collect();

        if recent_events.is_empty() {
            return Ok(vec![]);
        }

        match params.stat_type {
            StatisticalPatternType::Frequency => {
                let frequency =
                    recent_events.len() as f64 / params.time_window.num_seconds() as f64;

                if frequency >= params.threshold {
                    Ok(vec![PatternMatch {
                        pattern_id: pattern_id.to_string(),
                        pattern_name: params.name.to_string(),
                        events: recent_events,
                        start_time: window_start,
                        end_time: now,
                        confidence: frequency / params.threshold,
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert(
                                "frequency".to_string(),
                                serde_json::Value::Number(
                                    serde_json::Number::from_f64(frequency).unwrap_or(0.into()),
                                ),
                            );
                            meta
                        },
                    }])
                } else {
                    Ok(vec![])
                }
            }
            StatisticalPatternType::Correlation { field_a, field_b } => {
                // Extract field values and compute correlation using SciRS2
                let values_a: Vec<f64> = recent_events
                    .iter()
                    .filter_map(|evt| self.extract_numeric_value(evt, field_a))
                    .collect();

                let values_b: Vec<f64> = recent_events
                    .iter()
                    .filter_map(|evt| self.extract_numeric_value(evt, field_b))
                    .collect();

                if values_a.len() < 2 || values_b.len() < 2 {
                    return Ok(vec![]);
                }

                // Use scirs2-core for correlation computation
                let min_len = values_a.len().min(values_b.len());
                let arr_a = Array1::from_vec(values_a[..min_len].to_vec());
                let arr_b = Array1::from_vec(values_b[..min_len].to_vec());

                // Compute Pearson correlation coefficient manually
                let correlation = compute_correlation(&arr_a, &arr_b)?;

                if correlation.abs() >= params.threshold {
                    Ok(vec![PatternMatch {
                        pattern_id: pattern_id.to_string(),
                        pattern_name: params.name.to_string(),
                        events: recent_events,
                        start_time: window_start,
                        end_time: now,
                        confidence: correlation.abs(),
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert(
                                "correlation".to_string(),
                                serde_json::Value::Number(
                                    serde_json::Number::from_f64(correlation).unwrap_or(0.into()),
                                ),
                            );
                            meta
                        },
                    }])
                } else {
                    Ok(vec![])
                }
            }
            StatisticalPatternType::MovingAverage { field, window_size } => {
                let values: Vec<f64> = recent_events
                    .iter()
                    .filter_map(|evt| self.extract_numeric_value(evt, field))
                    .collect();

                if values.len() < *window_size {
                    return Ok(vec![]);
                }

                // Compute moving average using scirs2-core
                let arr = Array1::from_vec(values);
                let ma = arr
                    .slice(s![arr.len() - window_size..])
                    .mean()
                    .unwrap_or(0.0);

                if ma >= params.threshold {
                    Ok(vec![PatternMatch {
                        pattern_id: pattern_id.to_string(),
                        pattern_name: params.name.to_string(),
                        events: recent_events,
                        start_time: window_start,
                        end_time: now,
                        confidence: ma / params.threshold,
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert(
                                "moving_average".to_string(),
                                serde_json::Value::Number(
                                    serde_json::Number::from_f64(ma).unwrap_or(0.into()),
                                ),
                            );
                            meta
                        },
                    }])
                } else {
                    Ok(vec![])
                }
            }
            StatisticalPatternType::StdDev { field } => {
                let values: Vec<f64> = recent_events
                    .iter()
                    .filter_map(|evt| self.extract_numeric_value(evt, field))
                    .collect();

                if values.len() < 2 {
                    return Ok(vec![]);
                }

                // Compute standard deviation using scirs2-core
                let arr = Array1::from_vec(values);
                let std_dev = arr.std(0.0);

                if std_dev >= params.threshold {
                    Ok(vec![PatternMatch {
                        pattern_id: pattern_id.to_string(),
                        pattern_name: params.name.to_string(),
                        events: recent_events,
                        start_time: window_start,
                        end_time: now,
                        confidence: std_dev / params.threshold,
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert(
                                "std_dev".to_string(),
                                serde_json::Value::Number(
                                    serde_json::Number::from_f64(std_dev).unwrap_or(0.into()),
                                ),
                            );
                            meta
                        },
                    }])
                } else {
                    Ok(vec![])
                }
            }
            StatisticalPatternType::Anomaly { field, sensitivity } => {
                let values: Vec<f64> = recent_events
                    .iter()
                    .filter_map(|evt| self.extract_numeric_value(evt, field))
                    .collect();

                if values.len() < 3 {
                    return Ok(vec![]);
                }

                // Simple anomaly detection using Z-score with scirs2-core
                let arr = Array1::from_vec(values.clone());
                let mean = arr.mean().unwrap_or(0.0);
                let std_dev = arr.std(0.0);

                let last_value = values.last().expect("collection validated to be non-empty");
                let z_score = if std_dev > 0.0 {
                    (last_value - mean).abs() / std_dev
                } else {
                    0.0
                };

                if z_score >= params.threshold * sensitivity {
                    Ok(vec![PatternMatch {
                        pattern_id: pattern_id.to_string(),
                        pattern_name: params.name.to_string(),
                        events: recent_events,
                        start_time: window_start,
                        end_time: now,
                        confidence: z_score / (params.threshold * sensitivity),
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert(
                                "z_score".to_string(),
                                serde_json::Value::Number(
                                    serde_json::Number::from_f64(z_score).unwrap_or(0.into()),
                                ),
                            );
                            meta
                        },
                    }])
                } else {
                    Ok(vec![])
                }
            }
        }
    }

    /// Evaluate a pattern against an event (simple version without recursion)
    fn evaluate_pattern_simple(&self, pattern: &Pattern, event: &StreamEvent) -> Result<bool> {
        match pattern {
            Pattern::Simple { predicate, .. } => self.evaluate_predicate(predicate, event),
            _ => Ok(false), // Complex patterns not supported in simple evaluation
        }
    }

    /// Evaluate a predicate against an event
    fn evaluate_predicate(&self, predicate: &str, event: &StreamEvent) -> Result<bool> {
        // Simple predicate evaluation
        // In a real implementation, this would parse and evaluate expressions
        match predicate {
            "always" => Ok(true),
            "never" => Ok(false),
            pred if pred.starts_with("type:") => {
                let expected_type = pred
                    .strip_prefix("type:")
                    .expect("strip_prefix should succeed after starts_with check");
                Ok(self.get_event_type(event) == expected_type)
            }
            pred if pred.starts_with("subject:") => {
                let expected_subject = pred
                    .strip_prefix("subject:")
                    .expect("strip_prefix should succeed after starts_with check");
                Ok(self.get_event_subject(event) == Some(expected_subject.to_string()))
            }
            _ => Ok(false),
        }
    }

    /// Extract numeric value from event
    fn extract_numeric_value(&self, _event: &StreamEvent, _field: &str) -> Option<f64> {
        // Simplified extraction - would need proper implementation
        Some(1.0)
    }

    /// Get event type as string
    fn get_event_type(&self, event: &StreamEvent) -> &str {
        match event {
            StreamEvent::TripleAdded { .. } => "triple_added",
            StreamEvent::TripleRemoved { .. } => "triple_removed",
            StreamEvent::QuadAdded { .. } => "quad_added",
            StreamEvent::QuadRemoved { .. } => "quad_removed",
            StreamEvent::GraphCreated { .. } => "graph_created",
            StreamEvent::GraphCleared { .. } => "graph_cleared",
            StreamEvent::GraphDeleted { .. } => "graph_deleted",
            StreamEvent::TransactionBegin { .. } => "transaction_begin",
            StreamEvent::TransactionCommit { .. } => "transaction_commit",
            StreamEvent::TransactionAbort { .. } => "transaction_abort",
            _ => "unknown",
        }
    }

    /// Get event subject if available
    fn get_event_subject(&self, event: &StreamEvent) -> Option<String> {
        match event {
            StreamEvent::TripleAdded { subject, .. } => Some(subject.clone()),
            StreamEvent::TripleRemoved { subject, .. } => Some(subject.clone()),
            StreamEvent::QuadAdded { subject, .. } => Some(subject.clone()),
            StreamEvent::QuadRemoved { subject, .. } => Some(subject.clone()),
            _ => None,
        }
    }

    /// Clean up expired partial matches
    fn cleanup_expired_matches(&mut self, now: DateTime<Utc>) {
        let timeout = ChronoDuration::minutes(5);

        for (_, matches) in self.active_matches.iter_mut() {
            matches.retain(|m| now - m.start_time < timeout);
        }

        self.active_matches.retain(|_, matches| !matches.is_empty());
    }

    /// Get completed matches
    pub fn completed_matches(&self) -> &VecDeque<PatternMatch> {
        &self.completed_matches
    }

    /// Get statistics
    pub fn stats(&self) -> &PatternMatcherStats {
        &self.stats
    }

    /// Reset matcher state
    pub fn reset(&mut self) {
        self.active_matches.clear();
        self.completed_matches.clear();
        self.event_buffer.clear();
        self.stats = PatternMatcherStats::default();
    }
}

/// Compute Pearson correlation coefficient
fn compute_correlation(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64> {
    if a.len() != b.len() || a.len() < 2 {
        return Err(anyhow!(
            "Arrays must have same length and at least 2 elements"
        ));
    }

    let mean_a = a.mean().unwrap_or(0.0);
    let mean_b = b.mean().unwrap_or(0.0);

    let mut sum_product = 0.0;
    let mut sum_sq_a = 0.0;
    let mut sum_sq_b = 0.0;

    for i in 0..a.len() {
        let diff_a = a[i] - mean_a;
        let diff_b = b[i] - mean_b;
        sum_product += diff_a * diff_b;
        sum_sq_a += diff_a * diff_a;
        sum_sq_b += diff_b * diff_b;
    }

    let denominator = (sum_sq_a * sum_sq_b).sqrt();
    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(sum_product / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;

    fn create_test_event(subject: &str) -> StreamEvent {
        StreamEvent::TripleAdded {
            subject: subject.to_string(),
            predicate: "test".to_string(),
            object: "value".to_string(),
            graph: None,
            metadata: EventMetadata::default(),
        }
    }

    #[tokio::test]
    async fn test_simple_pattern() {
        let mut matcher = PatternMatcher::new(100);

        let pattern = Pattern::Simple {
            name: "test_pattern".to_string(),
            predicate: "type:triple_added".to_string(),
        };

        let pattern_id = matcher.register_pattern(pattern);

        let event = create_test_event("test_subject");
        let matches = matcher.process_event(event).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_id, pattern_id);
    }

    #[tokio::test]
    async fn test_sequence_pattern() {
        let mut matcher = PatternMatcher::new(100);

        let pattern = Pattern::Sequence {
            patterns: vec![
                Pattern::Simple {
                    name: "first".to_string(),
                    predicate: "type:triple_added".to_string(),
                },
                Pattern::Simple {
                    name: "second".to_string(),
                    predicate: "type:triple_added".to_string(),
                },
            ],
            max_distance: Some(ChronoDuration::seconds(10)),
        };

        let _pattern_id = matcher.register_pattern(pattern);

        let event1 = create_test_event("subject1");
        let event2 = create_test_event("subject2");

        let matches1 = matcher.process_event(event1).unwrap();
        assert_eq!(matches1.len(), 0); // First event, no complete match yet

        let matches2 = matcher.process_event(event2).unwrap();
        assert_eq!(matches2.len(), 1); // Second event completes the sequence
    }
}
