//! # Temporal Reasoning Module
//!
//! This module provides temporal reasoning capabilities using Allen's Interval Algebra
//! for reasoning about time intervals and their relationships.
//!
//! ## Features
//!
//! - **Allen's Interval Algebra**: 13 basic relations between time intervals
//! - **Temporal Constraint Networks**: Reasoning about temporal constraints
//! - **Path Consistency**: Enforcing consistency in temporal networks
//! - **Temporal Rules**: Rules with temporal conditions
//! - **Event Ordering**: Reasoning about event sequences
//!
//! ## Allen's 13 Relations
//!
//! 1. Before (p < q)
//! 2. After (p > q)
//! 3. Meets (p m q)
//! 4. Met-by (p mi q)
//! 5. Overlaps (p o q)
//! 6. Overlapped-by (p oi q)
//! 7. During (p d q)
//! 8. Contains (p di q)
//! 9. Starts (p s q)
//! 10. Started-by (p si q)
//! 11. Finishes (p f q)
//! 12. Finished-by (p fi q)
//! 13. Equals (p = q)
//!
//! ## Example
//!
//! ```text
//! use oxirs_rule::temporal::*;
//!
//! // Create time intervals
//! let morning = TimeInterval::new(8.0, 12.0);
//! let lunch = TimeInterval::new(12.0, 13.0);
//! let afternoon = TimeInterval::new(13.0, 17.0);
//!
//! // Check Allen relations
//! assert_eq!(morning.allen_relation(&lunch), AllenRelation::Meets);
//! assert_eq!(lunch.allen_relation(&afternoon), AllenRelation::Meets);
//! assert_eq!(morning.allen_relation(&afternoon), AllenRelation::Before);
//!
//! // Create a temporal constraint network
//! let mut tcn = TemporalConstraintNetwork::new();
//! tcn.add_interval("morning".to_string(), morning);
//! tcn.add_interval("lunch".to_string(), lunch);
//! tcn.add_constraint("morning".to_string(), "lunch".to_string(), AllenRelation::Meets);
//!
//! // Check consistency
//! assert!(tcn.is_consistent().unwrap());
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::Rule;
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};

/// Time interval represented by start and end points
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeInterval {
    /// Start time
    pub start: f64,
    /// End time
    pub end: f64,
}

impl TimeInterval {
    /// Create a new time interval
    pub fn new(start: f64, end: f64) -> Result<Self> {
        if start > end {
            return Err(anyhow!("Start time must be before end time"));
        }
        Ok(Self { start, end })
    }

    /// Get the duration of the interval
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }

    /// Check if a point in time is within the interval
    pub fn contains_point(&self, point: f64) -> bool {
        point >= self.start && point <= self.end
    }

    /// Compute the Allen relation between two intervals
    pub fn allen_relation(&self, other: &TimeInterval) -> AllenRelation {
        let eps = 1e-10; // Epsilon for floating point comparison

        // Equal
        if (self.start - other.start).abs() < eps && (self.end - other.end).abs() < eps {
            return AllenRelation::Equals;
        }

        // Before
        if self.end < other.start - eps {
            return AllenRelation::Before;
        }

        // After
        if self.start > other.end + eps {
            return AllenRelation::After;
        }

        // Meets
        if (self.end - other.start).abs() < eps {
            return AllenRelation::Meets;
        }

        // Met-by
        if (self.start - other.end).abs() < eps {
            return AllenRelation::MetBy;
        }

        // During
        if self.start > other.start + eps && self.end < other.end - eps {
            return AllenRelation::During;
        }

        // Contains
        if self.start < other.start - eps && self.end > other.end + eps {
            return AllenRelation::Contains;
        }

        // Starts
        if (self.start - other.start).abs() < eps && self.end < other.end - eps {
            return AllenRelation::Starts;
        }

        // Started-by
        if (self.start - other.start).abs() < eps && self.end > other.end + eps {
            return AllenRelation::StartedBy;
        }

        // Finishes
        if self.start > other.start + eps && (self.end - other.end).abs() < eps {
            return AllenRelation::Finishes;
        }

        // Finished-by
        if self.start < other.start - eps && (self.end - other.end).abs() < eps {
            return AllenRelation::FinishedBy;
        }

        // Overlaps
        if self.start < other.start - eps
            && self.end > other.start + eps
            && self.end < other.end - eps
        {
            return AllenRelation::Overlaps;
        }

        // Overlapped-by
        if self.start > other.start + eps
            && self.start < other.end - eps
            && self.end > other.end + eps
        {
            return AllenRelation::OverlappedBy;
        }

        // Fallback (shouldn't happen with correct logic)
        AllenRelation::Before
    }

    /// Get the intersection of two intervals (if they overlap)
    pub fn intersection(&self, other: &TimeInterval) -> Option<TimeInterval> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);

        if start <= end {
            TimeInterval::new(start, end).ok()
        } else {
            None
        }
    }

    /// Get the union of two intervals (if they overlap or meet)
    pub fn union(&self, other: &TimeInterval) -> Option<TimeInterval> {
        // Check if they overlap or meet
        let relation = self.allen_relation(other);
        match relation {
            AllenRelation::Before | AllenRelation::After => None,
            _ => {
                let start = self.start.min(other.start);
                let end = self.end.max(other.end);
                TimeInterval::new(start, end).ok()
            }
        }
    }
}

/// Allen's 13 basic relations between time intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllenRelation {
    /// p before q: p.end < q.start
    Before,
    /// p after q: p.start > q.end
    After,
    /// p meets q: p.end = q.start
    Meets,
    /// p met-by q: p.start = q.end
    MetBy,
    /// p overlaps q: p.start < q.start < p.end < q.end
    Overlaps,
    /// p overlapped-by q: q.start < p.start < q.end < p.end
    OverlappedBy,
    /// p during q: q.start < p.start < p.end < q.end
    During,
    /// p contains q: p.start < q.start < q.end < p.end
    Contains,
    /// p starts q: p.start = q.start < p.end < q.end
    Starts,
    /// p started-by q: p.start = q.start < q.end < p.end
    StartedBy,
    /// p finishes q: q.start < p.start < p.end = q.end
    Finishes,
    /// p finished-by q: p.start < q.start < p.end = q.end
    FinishedBy,
    /// p equals q: p.start = q.start, p.end = q.end
    Equals,
}

impl AllenRelation {
    /// Get the inverse relation
    pub fn inverse(&self) -> AllenRelation {
        match self {
            AllenRelation::Before => AllenRelation::After,
            AllenRelation::After => AllenRelation::Before,
            AllenRelation::Meets => AllenRelation::MetBy,
            AllenRelation::MetBy => AllenRelation::Meets,
            AllenRelation::Overlaps => AllenRelation::OverlappedBy,
            AllenRelation::OverlappedBy => AllenRelation::Overlaps,
            AllenRelation::During => AllenRelation::Contains,
            AllenRelation::Contains => AllenRelation::During,
            AllenRelation::Starts => AllenRelation::StartedBy,
            AllenRelation::StartedBy => AllenRelation::Starts,
            AllenRelation::Finishes => AllenRelation::FinishedBy,
            AllenRelation::FinishedBy => AllenRelation::Finishes,
            AllenRelation::Equals => AllenRelation::Equals,
        }
    }

    /// Get all possible relations
    pub fn all() -> Vec<AllenRelation> {
        vec![
            AllenRelation::Before,
            AllenRelation::After,
            AllenRelation::Meets,
            AllenRelation::MetBy,
            AllenRelation::Overlaps,
            AllenRelation::OverlappedBy,
            AllenRelation::During,
            AllenRelation::Contains,
            AllenRelation::Starts,
            AllenRelation::StartedBy,
            AllenRelation::Finishes,
            AllenRelation::FinishedBy,
            AllenRelation::Equals,
        ]
    }
}

/// Temporal constraint: a set of possible Allen relations
#[derive(Debug, Clone)]
pub struct TemporalConstraint {
    /// Possible relations between two intervals
    pub relations: HashSet<AllenRelation>,
}

impl TemporalConstraint {
    /// Create a constraint with all possible relations
    pub fn universal() -> Self {
        Self {
            relations: AllenRelation::all().into_iter().collect(),
        }
    }

    /// Create a constraint with a single relation
    pub fn single(relation: AllenRelation) -> Self {
        let mut relations = HashSet::new();
        relations.insert(relation);
        Self { relations }
    }

    /// Intersect two constraints
    pub fn intersect(&self, other: &TemporalConstraint) -> TemporalConstraint {
        TemporalConstraint {
            relations: self
                .relations
                .intersection(&other.relations)
                .copied()
                .collect(),
        }
    }

    /// Check if the constraint is empty (no possible relations)
    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }

    /// Compose two constraints using Allen's composition table
    pub fn compose(&self, other: &TemporalConstraint) -> TemporalConstraint {
        let mut result = HashSet::new();

        for r1 in &self.relations {
            for r2 in &other.relations {
                // Add all possible compositions
                result.extend(Self::compose_relations(*r1, *r2));
            }
        }

        TemporalConstraint { relations: result }
    }

    /// Compose two Allen relations (simplified version)
    fn compose_relations(r1: AllenRelation, r2: AllenRelation) -> Vec<AllenRelation> {
        // This is a simplified composition table
        // Full implementation would use Allen's complete composition table
        use AllenRelation::*;

        match (r1, r2) {
            (Before, Before) => vec![Before],
            (Before, Meets) => vec![Before],
            (Meets, Before) => vec![Before],
            (Meets, Meets) => vec![Meets],
            (Equals, r) => vec![r],
            (r, Equals) => vec![r],
            _ => AllenRelation::all(), // Conservative: all possible relations
        }
    }
}

/// Temporal Constraint Network
#[derive(Debug, Clone)]
pub struct TemporalConstraintNetwork {
    /// Named intervals
    intervals: HashMap<String, TimeInterval>,
    /// Constraints between intervals
    constraints: HashMap<(String, String), TemporalConstraint>,
}

impl TemporalConstraintNetwork {
    /// Create a new temporal constraint network
    pub fn new() -> Self {
        Self {
            intervals: HashMap::new(),
            constraints: HashMap::new(),
        }
    }

    /// Add an interval to the network
    pub fn add_interval(&mut self, name: String, interval: TimeInterval) {
        self.intervals.insert(name, interval);
    }

    /// Add a constraint between two intervals
    pub fn add_constraint(&mut self, from: String, to: String, relation: AllenRelation) {
        let constraint = TemporalConstraint::single(relation);
        self.constraints
            .insert((from.clone(), to.clone()), constraint.clone());

        // Add inverse constraint
        let inverse_constraint = TemporalConstraint::single(relation.inverse());
        self.constraints.insert((to, from), inverse_constraint);
    }

    /// Get constraint between two intervals
    pub fn get_constraint(&self, from: &str, to: &str) -> TemporalConstraint {
        self.constraints
            .get(&(from.to_string(), to.to_string()))
            .cloned()
            .unwrap_or_else(TemporalConstraint::universal)
    }

    /// Check if the network is consistent
    pub fn is_consistent(&self) -> Result<bool> {
        // Use path consistency algorithm (PC-2)
        let mut network = self.clone();

        // Iterate until no changes
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            let interval_names: Vec<String> = network.intervals.keys().cloned().collect();

            for i in &interval_names {
                for j in &interval_names {
                    if i == j {
                        continue;
                    }

                    for k in &interval_names {
                        if k == i || k == j {
                            continue;
                        }

                        // Path consistency: C[i,j] = C[i,j] ∩ (C[i,k] ∘ C[k,j])
                        let c_ij = network.get_constraint(i, j);
                        let c_ik = network.get_constraint(i, k);
                        let c_kj = network.get_constraint(k, j);

                        let composed = c_ik.compose(&c_kj);
                        let new_c_ij = c_ij.intersect(&composed);

                        if new_c_ij.is_empty() {
                            return Ok(false); // Inconsistent
                        }

                        if new_c_ij.relations != c_ij.relations {
                            changed = true;
                            network
                                .constraints
                                .insert((i.clone(), j.clone()), new_c_ij.clone());

                            // Update inverse
                            let inverse_relations: HashSet<AllenRelation> =
                                new_c_ij.relations.iter().map(|r| r.inverse()).collect();
                            network.constraints.insert(
                                (j.clone(), i.clone()),
                                TemporalConstraint {
                                    relations: inverse_relations,
                                },
                            );
                        }
                    }
                }
            }
        }

        Ok(true)
    }

    /// Get all intervals
    pub fn get_intervals(&self) -> &HashMap<String, TimeInterval> {
        &self.intervals
    }

    /// Query temporal relationships
    pub fn query_relation(&self, from: &str, to: &str) -> Option<HashSet<AllenRelation>> {
        self.constraints
            .get(&(from.to_string(), to.to_string()))
            .map(|c| c.relations.clone())
    }
}

impl Default for TemporalConstraintNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Temporal rule with time-based conditions
#[derive(Debug, Clone)]
pub struct TemporalRule {
    /// Base rule
    pub rule: Rule,
    /// Temporal constraints on rule variables
    pub temporal_constraints: HashMap<String, TimeInterval>,
}

impl TemporalRule {
    /// Create a new temporal rule
    pub fn new(rule: Rule) -> Self {
        Self {
            rule,
            temporal_constraints: HashMap::new(),
        }
    }

    /// Add a temporal constraint to a variable
    pub fn add_temporal_constraint(&mut self, variable: String, interval: TimeInterval) {
        self.temporal_constraints.insert(variable, interval);
    }

    /// Check if the rule is applicable at a given time
    pub fn is_applicable_at(&self, time: f64) -> bool {
        self.temporal_constraints
            .values()
            .all(|interval| interval.contains_point(time))
    }
}

/// Event with a time point or interval
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    /// Event name
    pub name: String,
    /// Time interval when the event occurs
    pub interval: TimeInterval,
    /// Event properties
    pub properties: HashMap<String, String>,
}

impl TemporalEvent {
    /// Create a new temporal event
    pub fn new(name: String, interval: TimeInterval) -> Self {
        Self {
            name,
            interval,
            properties: HashMap::new(),
        }
    }

    /// Add a property to the event
    pub fn add_property(&mut self, key: String, value: String) {
        self.properties.insert(key, value);
    }

    /// Check the Allen relation with another event
    pub fn relation_with(&self, other: &TemporalEvent) -> AllenRelation {
        self.interval.allen_relation(&other.interval)
    }
}

/// Temporal reasoning engine
#[derive(Debug)]
pub struct TemporalReasoningEngine {
    /// Temporal constraint network
    tcn: TemporalConstraintNetwork,
    /// Temporal rules
    rules: Vec<TemporalRule>,
    /// Events
    events: Vec<TemporalEvent>,
}

impl TemporalReasoningEngine {
    /// Create a new temporal reasoning engine
    pub fn new() -> Self {
        Self {
            tcn: TemporalConstraintNetwork::new(),
            rules: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Add an interval to the network
    pub fn add_interval(&mut self, name: String, interval: TimeInterval) {
        self.tcn.add_interval(name, interval);
    }

    /// Add a temporal constraint
    pub fn add_constraint(&mut self, from: String, to: String, relation: AllenRelation) {
        self.tcn.add_constraint(from, to, relation);
    }

    /// Add a temporal rule
    pub fn add_temporal_rule(&mut self, rule: TemporalRule) {
        self.rules.push(rule);
    }

    /// Add an event
    pub fn add_event(&mut self, event: TemporalEvent) {
        self.events.push(event);
    }

    /// Get applicable rules at a given time
    pub fn get_applicable_rules(&self, time: f64) -> Vec<&TemporalRule> {
        self.rules
            .iter()
            .filter(|rule| rule.is_applicable_at(time))
            .collect()
    }

    /// Get events occurring at a given time
    pub fn get_events_at(&self, time: f64) -> Vec<&TemporalEvent> {
        self.events
            .iter()
            .filter(|event| event.interval.contains_point(time))
            .collect()
    }

    /// Find events with a specific relation to a given event
    pub fn find_events_with_relation(
        &self,
        event_name: &str,
        relation: AllenRelation,
    ) -> Vec<&TemporalEvent> {
        if let Some(target_event) = self.events.iter().find(|e| e.name == event_name) {
            self.events
                .iter()
                .filter(|e| e.name != event_name && e.relation_with(target_event) == relation)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Check network consistency
    pub fn is_consistent(&self) -> Result<bool> {
        self.tcn.is_consistent()
    }

    /// Get the temporal constraint network
    pub fn get_network(&self) -> &TemporalConstraintNetwork {
        &self.tcn
    }
}

impl Default for TemporalReasoningEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_interval_creation() {
        let interval = TimeInterval::new(1.0, 5.0).unwrap();
        assert_eq!(interval.start, 1.0);
        assert_eq!(interval.end, 5.0);
        assert_eq!(interval.duration(), 4.0);

        // Invalid interval
        assert!(TimeInterval::new(5.0, 1.0).is_err());
    }

    #[test]
    fn test_allen_before() {
        let i1 = TimeInterval::new(1.0, 3.0).unwrap();
        let i2 = TimeInterval::new(5.0, 7.0).unwrap();

        assert_eq!(i1.allen_relation(&i2), AllenRelation::Before);
        assert_eq!(i2.allen_relation(&i1), AllenRelation::After);
    }

    #[test]
    fn test_allen_meets() {
        let i1 = TimeInterval::new(1.0, 3.0).unwrap();
        let i2 = TimeInterval::new(3.0, 5.0).unwrap();

        assert_eq!(i1.allen_relation(&i2), AllenRelation::Meets);
        assert_eq!(i2.allen_relation(&i1), AllenRelation::MetBy);
    }

    #[test]
    fn test_allen_overlaps() {
        let i1 = TimeInterval::new(1.0, 4.0).unwrap();
        let i2 = TimeInterval::new(3.0, 6.0).unwrap();

        assert_eq!(i1.allen_relation(&i2), AllenRelation::Overlaps);
        assert_eq!(i2.allen_relation(&i1), AllenRelation::OverlappedBy);
    }

    #[test]
    fn test_allen_during() {
        let i1 = TimeInterval::new(2.0, 4.0).unwrap();
        let i2 = TimeInterval::new(1.0, 5.0).unwrap();

        assert_eq!(i1.allen_relation(&i2), AllenRelation::During);
        assert_eq!(i2.allen_relation(&i1), AllenRelation::Contains);
    }

    #[test]
    fn test_allen_equals() {
        let i1 = TimeInterval::new(1.0, 5.0).unwrap();
        let i2 = TimeInterval::new(1.0, 5.0).unwrap();

        assert_eq!(i1.allen_relation(&i2), AllenRelation::Equals);
    }

    #[test]
    fn test_interval_intersection() {
        let i1 = TimeInterval::new(1.0, 5.0).unwrap();
        let i2 = TimeInterval::new(3.0, 7.0).unwrap();

        let intersection = i1.intersection(&i2).unwrap();
        assert_eq!(intersection.start, 3.0);
        assert_eq!(intersection.end, 5.0);
    }

    #[test]
    fn test_interval_union() {
        let i1 = TimeInterval::new(1.0, 5.0).unwrap();
        let i2 = TimeInterval::new(3.0, 7.0).unwrap();

        let union = i1.union(&i2).unwrap();
        assert_eq!(union.start, 1.0);
        assert_eq!(union.end, 7.0);
    }

    #[test]
    fn test_temporal_constraint_network() {
        let mut tcn = TemporalConstraintNetwork::new();

        let morning = TimeInterval::new(8.0, 12.0).unwrap();
        let lunch = TimeInterval::new(12.0, 13.0).unwrap();
        let afternoon = TimeInterval::new(13.0, 17.0).unwrap();

        tcn.add_interval("morning".to_string(), morning);
        tcn.add_interval("lunch".to_string(), lunch);
        tcn.add_interval("afternoon".to_string(), afternoon);

        tcn.add_constraint(
            "morning".to_string(),
            "lunch".to_string(),
            AllenRelation::Meets,
        );
        tcn.add_constraint(
            "lunch".to_string(),
            "afternoon".to_string(),
            AllenRelation::Meets,
        );

        assert!(tcn.is_consistent().unwrap());
    }

    #[test]
    fn test_temporal_rule() {
        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![],
            head: vec![],
        };

        let mut temporal_rule = TemporalRule::new(rule);
        let interval = TimeInterval::new(9.0, 17.0).unwrap();
        temporal_rule.add_temporal_constraint("working_hours".to_string(), interval);

        assert!(temporal_rule.is_applicable_at(10.0));
        assert!(!temporal_rule.is_applicable_at(20.0));
    }

    #[test]
    fn test_temporal_event() {
        let meeting = TemporalEvent::new(
            "team_meeting".to_string(),
            TimeInterval::new(14.0, 15.0).unwrap(),
        );

        let lunch = TemporalEvent::new(
            "lunch_break".to_string(),
            TimeInterval::new(12.0, 13.0).unwrap(),
        );

        assert_eq!(lunch.relation_with(&meeting), AllenRelation::Before);
    }

    #[test]
    fn test_temporal_reasoning_engine() {
        let mut engine = TemporalReasoningEngine::new();

        let event1 = TemporalEvent::new("event1".to_string(), TimeInterval::new(1.0, 3.0).unwrap());

        let event2 = TemporalEvent::new("event2".to_string(), TimeInterval::new(2.0, 4.0).unwrap());

        engine.add_event(event1);
        engine.add_event(event2);

        let events_at_2_5 = engine.get_events_at(2.5);
        assert_eq!(events_at_2_5.len(), 2);
    }

    #[test]
    fn test_allen_relation_inverse() {
        assert_eq!(AllenRelation::Before.inverse(), AllenRelation::After);
        assert_eq!(AllenRelation::Meets.inverse(), AllenRelation::MetBy);
        assert_eq!(
            AllenRelation::Overlaps.inverse(),
            AllenRelation::OverlappedBy
        );
        assert_eq!(AllenRelation::Equals.inverse(), AllenRelation::Equals);
    }
}
