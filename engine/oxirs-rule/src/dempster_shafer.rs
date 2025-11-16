//! # Dempster-Shafer Theory Module
//!
//! This module provides Dempster-Shafer theory of evidence for evidential reasoning
//! with uncertainty. It supports belief functions, mass functions, and Dempster's
//! rule of combination.
//!
//! ## Features
//!
//! - **Mass Functions**: Basic probability assignments to sets of hypotheses
//! - **Belief Functions**: Lower bounds on probabilities
//! - **Plausibility Functions**: Upper bounds on probabilities
//! - **Dempster's Rule**: Combining independent evidence
//! - **Uncertainty Intervals**: [Bel, Pl] intervals for hypotheses
//! - **Focal Elements**: Sets with non-zero mass assignments
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::dempster_shafer::*;
//!
//! // Create a frame of discernment (set of all possible hypotheses)
//! let frame = vec!["A".to_string(), "B".to_string(), "C".to_string()];
//! let mut ds = DempsterShaferSystem::new(frame);
//!
//! // Add evidence: mass of 0.6 to hypothesis A, 0.3 to B, 0.1 to uncertainty
//! let mut evidence1 = MassFunction::new();
//! evidence1.assign_mass(vec!["A".to_string()], 0.6).unwrap();
//! evidence1.assign_mass(vec!["B".to_string()], 0.3).unwrap();
//! evidence1.assign_mass(vec!["A".to_string(), "B".to_string(), "C".to_string()], 0.1).unwrap();
//!
//! ds.add_evidence(evidence1).unwrap();
//!
//! // Compute belief and plausibility
//! let belief_a = ds.belief(&vec!["A".to_string()]).unwrap();
//! let plausibility_a = ds.plausibility(&vec!["A".to_string()]).unwrap();
//!
//! println!("Belief(A) = {}, Plausibility(A) = {}", belief_a, plausibility_a);
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## Theory Background
//!
//! Dempster-Shafer theory generalizes Bayesian probability by allowing probability
//! mass to be assigned to sets of hypotheses rather than just single hypotheses.
//!
//! - **Mass Function m(A)**: Probability mass assigned to exactly the set A
//! - **Belief Bel(A)**: Sum of masses of all subsets of A (lower bound)
//! - **Plausibility Pl(A)**: Sum of masses of all sets that intersect A (upper bound)
//! - **Dempster's Rule**: Combines independent evidence sources

use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};

/// Mass function (basic probability assignment)
/// Maps each focal element (subset of frame) to its mass
#[derive(Debug, Clone)]
pub struct MassFunction {
    /// Map from focal elements to their mass values
    /// Key is a sorted vector of hypotheses (to ensure uniqueness)
    masses: HashMap<Vec<String>, f64>,
}

impl MassFunction {
    /// Create a new empty mass function
    pub fn new() -> Self {
        Self {
            masses: HashMap::new(),
        }
    }

    /// Assign mass to a set of hypotheses
    pub fn assign_mass(&mut self, mut hypotheses: Vec<String>, mass: f64) -> Result<()> {
        if !(0.0..=1.0).contains(&mass) {
            return Err(anyhow!("Mass must be between 0 and 1, got {}", mass));
        }

        // Sort hypotheses to ensure consistent keys
        hypotheses.sort();

        if hypotheses.is_empty() {
            return Err(anyhow!("Cannot assign mass to empty set"));
        }

        // Add or update mass
        *self.masses.entry(hypotheses).or_insert(0.0) += mass;

        Ok(())
    }

    /// Get mass assigned to a specific set
    pub fn get_mass(&self, hypotheses: &[String]) -> f64 {
        let mut sorted = hypotheses.to_vec();
        sorted.sort();
        *self.masses.get(&sorted).unwrap_or(&0.0)
    }

    /// Get all focal elements (sets with non-zero mass)
    pub fn focal_elements(&self) -> Vec<&Vec<String>> {
        self.masses
            .iter()
            .filter(|(_, &mass)| mass > 1e-10)
            .map(|(elem, _)| elem)
            .collect()
    }

    /// Get total mass (should sum to 1.0)
    pub fn total_mass(&self) -> f64 {
        self.masses.values().sum()
    }

    /// Normalize masses to sum to 1.0
    pub fn normalize(&mut self) -> Result<()> {
        let total = self.total_mass();
        if total < 1e-10 {
            return Err(anyhow!(
                "Cannot normalize mass function with total mass near zero"
            ));
        }

        for mass in self.masses.values_mut() {
            *mass /= total;
        }

        Ok(())
    }
}

impl Default for MassFunction {
    fn default() -> Self {
        Self::new()
    }
}

/// Dempster-Shafer system for evidential reasoning
#[derive(Debug, Clone)]
pub struct DempsterShaferSystem {
    /// Frame of discernment (all possible hypotheses)
    frame: Vec<String>,
    /// Combined mass function from all evidence
    combined_mass: MassFunction,
}

impl DempsterShaferSystem {
    /// Create a new Dempster-Shafer system with a frame of discernment
    pub fn new(frame: Vec<String>) -> Self {
        let mut combined_mass = MassFunction::new();
        // Initially, all mass is assigned to the entire frame (maximum uncertainty)
        let _ = combined_mass.assign_mass(frame.clone(), 1.0);

        Self {
            frame,
            combined_mass,
        }
    }

    /// Add new evidence using Dempster's rule of combination
    pub fn add_evidence(&mut self, evidence: MassFunction) -> Result<()> {
        // Validate evidence is normalized
        let total_mass = evidence.total_mass();
        if (total_mass - 1.0).abs() > 1e-6 {
            return Err(anyhow!("Evidence mass must sum to 1.0, got {}", total_mass));
        }

        // Combine using Dempster's rule
        self.combined_mass = self.dempster_combine(&self.combined_mass, &evidence)?;

        Ok(())
    }

    /// Combine two mass functions using Dempster's rule
    ///
    /// m1 ⊕ m2(A) = (1/(1-K)) * Σ_{B∩C=A} m1(B) * m2(C)
    /// where K = Σ_{B∩C=∅} m1(B) * m2(C) is the conflict
    fn dempster_combine(&self, m1: &MassFunction, m2: &MassFunction) -> Result<MassFunction> {
        let mut combined = MassFunction::new();
        let mut conflict = 0.0;

        // Iterate over all pairs of focal elements
        for focal1 in m1.focal_elements() {
            for focal2 in m2.focal_elements() {
                let mass1 = m1.get_mass(focal1);
                let mass2 = m2.get_mass(focal2);

                // Compute intersection
                let intersection = self.intersect(focal1, focal2);

                if intersection.is_empty() {
                    // Conflict: focal elements don't intersect
                    conflict += mass1 * mass2;
                } else {
                    // Add to combined mass
                    combined
                        .assign_mass(intersection, mass1 * mass2)
                        .map_err(|e| anyhow!("Failed to combine masses: {}", e))?;
                }
            }
        }

        // Check for total conflict
        if (conflict - 1.0).abs() < 1e-10 {
            return Err(anyhow!(
                "Total conflict: evidence is completely contradictory"
            ));
        }

        // Normalize by (1 - conflict)
        for mass in combined.masses.values_mut() {
            *mass /= 1.0 - conflict;
        }

        Ok(combined)
    }

    /// Compute intersection of two hypothesis sets
    fn intersect(&self, set1: &[String], set2: &[String]) -> Vec<String> {
        let s1: HashSet<_> = set1.iter().collect();
        let s2: HashSet<_> = set2.iter().collect();

        let mut intersection: Vec<_> = s1.intersection(&s2).map(|&s| s.clone()).collect();
        intersection.sort();
        intersection
    }

    /// Compute belief function Bel(A) - lower bound on probability
    ///
    /// Bel(A) = Σ_{B⊆A} m(B)
    pub fn belief(&self, hypotheses: &[String]) -> Result<f64> {
        self.validate_hypotheses(hypotheses)?;

        let target_set: HashSet<_> = hypotheses.iter().collect();
        let mut belief = 0.0;

        // Sum masses of all subsets of the target set
        for focal in self.combined_mass.focal_elements() {
            let focal_set: HashSet<_> = focal.iter().collect();

            // Check if focal is a subset of target
            if focal_set.is_subset(&target_set) {
                belief += self.combined_mass.get_mass(focal);
            }
        }

        Ok(belief)
    }

    /// Compute plausibility function Pl(A) - upper bound on probability
    ///
    /// Pl(A) = Σ_{B∩A≠∅} m(B)
    pub fn plausibility(&self, hypotheses: &[String]) -> Result<f64> {
        self.validate_hypotheses(hypotheses)?;

        let target_set: HashSet<_> = hypotheses.iter().collect();
        let mut plausibility = 0.0;

        // Sum masses of all sets that intersect with target
        for focal in self.combined_mass.focal_elements() {
            let focal_set: HashSet<_> = focal.iter().collect();

            // Check if focal intersects with target
            if !focal_set.is_disjoint(&target_set) {
                plausibility += self.combined_mass.get_mass(focal);
            }
        }

        Ok(plausibility)
    }

    /// Compute uncertainty interval [Bel(A), Pl(A)]
    pub fn uncertainty_interval(&self, hypotheses: &[String]) -> Result<(f64, f64)> {
        let belief = self.belief(hypotheses)?;
        let plausibility = self.plausibility(hypotheses)?;
        Ok((belief, plausibility))
    }

    /// Compute pignistic probability (for decision making)
    ///
    /// BetP(h) = Σ_{A: h∈A} m(A) / |A|
    pub fn pignistic_probability(&self, hypothesis: &str) -> Result<f64> {
        if !self.frame.contains(&hypothesis.to_string()) {
            return Err(anyhow!("Hypothesis '{}' not in frame", hypothesis));
        }

        let mut prob = 0.0;

        // Sum over all focal elements containing this hypothesis
        for focal in self.combined_mass.focal_elements() {
            if focal.contains(&hypothesis.to_string()) {
                let mass = self.combined_mass.get_mass(focal);
                let cardinality = focal.len() as f64;
                prob += mass / cardinality;
            }
        }

        Ok(prob)
    }

    /// Get all pignistic probabilities (probability distribution over single hypotheses)
    pub fn pignistic_distribution(&self) -> Result<HashMap<String, f64>> {
        let mut distribution = HashMap::new();

        for hypothesis in &self.frame {
            let prob = self.pignistic_probability(hypothesis)?;
            distribution.insert(hypothesis.clone(), prob);
        }

        Ok(distribution)
    }

    /// Get the combined mass function
    pub fn get_combined_mass(&self) -> &MassFunction {
        &self.combined_mass
    }

    /// Get the frame of discernment
    pub fn get_frame(&self) -> &[String] {
        &self.frame
    }

    /// Validate that hypotheses are in the frame
    fn validate_hypotheses(&self, hypotheses: &[String]) -> Result<()> {
        for h in hypotheses {
            if !self.frame.contains(h) {
                return Err(anyhow!("Hypothesis '{}' not in frame of discernment", h));
            }
        }
        Ok(())
    }

    /// Compute conflict between two evidence sources
    pub fn compute_conflict(&self, evidence1: &MassFunction, evidence2: &MassFunction) -> f64 {
        let mut conflict = 0.0;

        for focal1 in evidence1.focal_elements() {
            for focal2 in evidence2.focal_elements() {
                let intersection = self.intersect(focal1, focal2);
                if intersection.is_empty() {
                    conflict += evidence1.get_mass(focal1) * evidence2.get_mass(focal2);
                }
            }
        }

        conflict
    }
}

/// Rule-based interface for Dempster-Shafer reasoning
#[derive(Debug, Clone)]
pub struct DempsterShaferReasoner {
    /// Underlying DS system
    system: DempsterShaferSystem,
    /// Evidence sources with labels
    evidence_sources: HashMap<String, MassFunction>,
}

impl DempsterShaferReasoner {
    /// Create a new DS reasoner
    pub fn new(hypotheses: Vec<String>) -> Self {
        Self {
            system: DempsterShaferSystem::new(hypotheses),
            evidence_sources: HashMap::new(),
        }
    }

    /// Add named evidence source
    pub fn add_named_evidence(&mut self, name: String, evidence: MassFunction) -> Result<()> {
        // Validate and add to system
        self.system.add_evidence(evidence.clone())?;

        // Store for later reference
        self.evidence_sources.insert(name, evidence);

        Ok(())
    }

    /// Query belief in a hypothesis or set of hypotheses
    pub fn query_belief(&self, hypotheses: Vec<String>) -> Result<f64> {
        self.system.belief(&hypotheses)
    }

    /// Query plausibility
    pub fn query_plausibility(&self, hypotheses: Vec<String>) -> Result<f64> {
        self.system.plausibility(&hypotheses)
    }

    /// Get decision probabilities (pignistic transformation)
    pub fn get_decision_probabilities(&self) -> Result<HashMap<String, f64>> {
        self.system.pignistic_distribution()
    }

    /// Get uncertainty intervals for all hypotheses
    pub fn get_all_uncertainty_intervals(&self) -> Result<HashMap<String, (f64, f64)>> {
        let mut intervals = HashMap::new();

        for hypothesis in self.system.get_frame() {
            let interval = self
                .system
                .uncertainty_interval(std::slice::from_ref(hypothesis))?;
            intervals.insert(hypothesis.clone(), interval);
        }

        Ok(intervals)
    }

    /// Get most plausible hypothesis
    pub fn get_most_plausible(&self) -> Result<(String, f64)> {
        let dist = self.system.pignistic_distribution()?;

        dist.into_iter()
            .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
            .ok_or_else(|| anyhow!("No hypotheses in system"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_function_basic() {
        let mut mf = MassFunction::new();
        mf.assign_mass(vec!["A".to_string()], 0.6).unwrap();
        mf.assign_mass(vec!["B".to_string()], 0.4).unwrap();

        assert!((mf.get_mass(&["A".to_string()]) - 0.6).abs() < 1e-10);
        assert!((mf.total_mass() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mass_function_normalization() {
        let mut mf = MassFunction::new();
        mf.assign_mass(vec!["A".to_string()], 0.3).unwrap();
        mf.assign_mass(vec!["B".to_string()], 0.2).unwrap();

        mf.normalize().unwrap();
        assert!((mf.total_mass() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ds_system_belief() {
        let frame = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut ds = DempsterShaferSystem::new(frame);

        let mut evidence = MassFunction::new();
        evidence.assign_mass(vec!["A".to_string()], 0.6).unwrap();
        evidence
            .assign_mass(vec!["A".to_string(), "B".to_string()], 0.3)
            .unwrap();
        evidence
            .assign_mass(vec!["A".to_string(), "B".to_string(), "C".to_string()], 0.1)
            .unwrap();

        ds.add_evidence(evidence).unwrap();

        // Bel(A) = m({A}) = 0.6
        let belief_a = ds.belief(&["A".to_string()]).unwrap();
        assert!((belief_a - 0.6).abs() < 1e-10);

        // Bel({A,B}) = m({A}) + m({A,B}) = 0.6 + 0.3 = 0.9
        let belief_ab = ds.belief(&["A".to_string(), "B".to_string()]).unwrap();
        assert!((belief_ab - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_ds_system_plausibility() {
        let frame = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut ds = DempsterShaferSystem::new(frame);

        let mut evidence = MassFunction::new();
        evidence.assign_mass(vec!["A".to_string()], 0.6).unwrap();
        evidence.assign_mass(vec!["B".to_string()], 0.3).unwrap();
        evidence.assign_mass(vec!["C".to_string()], 0.1).unwrap();

        ds.add_evidence(evidence).unwrap();

        // Pl(A) = m({A}) = 0.6 (only {A} intersects with {A})
        let pl_a = ds.plausibility(&["A".to_string()]).unwrap();
        assert!((pl_a - 0.6).abs() < 1e-10);

        // Pl({A,B}) = m({A}) + m({B}) = 0.9 ({A}, {B} intersect with {A,B})
        let pl_ab = ds
            .plausibility(&["A".to_string(), "B".to_string()])
            .unwrap();
        assert!((pl_ab - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_dempster_combination() {
        let frame = vec!["A".to_string(), "B".to_string()];
        let mut ds = DempsterShaferSystem::new(frame);

        // First evidence: 70% A, 20% B, 10% {A,B}
        let mut ev1 = MassFunction::new();
        ev1.assign_mass(vec!["A".to_string()], 0.7).unwrap();
        ev1.assign_mass(vec!["B".to_string()], 0.2).unwrap();
        ev1.assign_mass(vec!["A".to_string(), "B".to_string()], 0.1)
            .unwrap();

        // Second evidence: 60% A, 30% B, 10% {A,B}
        let mut ev2 = MassFunction::new();
        ev2.assign_mass(vec!["A".to_string()], 0.6).unwrap();
        ev2.assign_mass(vec!["B".to_string()], 0.3).unwrap();
        ev2.assign_mass(vec!["A".to_string(), "B".to_string()], 0.1)
            .unwrap();

        ds.add_evidence(ev1).unwrap();
        ds.add_evidence(ev2).unwrap();

        // After combination, belief in A should increase
        let belief_a = ds.belief(&["A".to_string()]).unwrap();
        assert!(belief_a > 0.7); // Should be stronger than individual evidence
    }

    #[test]
    fn test_pignistic_probability() {
        let frame = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut ds = DempsterShaferSystem::new(frame);

        let mut evidence = MassFunction::new();
        evidence.assign_mass(vec!["A".to_string()], 0.6).unwrap();
        evidence
            .assign_mass(vec!["A".to_string(), "B".to_string()], 0.4)
            .unwrap();

        ds.add_evidence(evidence).unwrap();

        // BetP(A) = 0.6 + 0.4/2 = 0.8
        let prob_a = ds.pignistic_probability("A").unwrap();
        assert!((prob_a - 0.8).abs() < 1e-10);

        // BetP(B) = 0.4/2 = 0.2
        let prob_b = ds.pignistic_probability("B").unwrap();
        assert!((prob_b - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_ds_reasoner() {
        let hypotheses = vec!["Rain".to_string(), "NoRain".to_string()];
        let mut reasoner = DempsterShaferReasoner::new(hypotheses);

        // Weather forecast: 70% rain
        let mut forecast = MassFunction::new();
        forecast.assign_mass(vec!["Rain".to_string()], 0.7).unwrap();
        forecast
            .assign_mass(vec!["NoRain".to_string()], 0.3)
            .unwrap();

        reasoner
            .add_named_evidence("forecast".to_string(), forecast)
            .unwrap();

        // Ground sensor: 80% rain
        let mut sensor = MassFunction::new();
        sensor.assign_mass(vec!["Rain".to_string()], 0.8).unwrap();
        sensor.assign_mass(vec!["NoRain".to_string()], 0.2).unwrap();

        reasoner
            .add_named_evidence("sensor".to_string(), sensor)
            .unwrap();

        // Combined belief in rain should be high
        let (most_plausible, prob) = reasoner.get_most_plausible().unwrap();
        assert_eq!(most_plausible, "Rain");
        assert!(prob > 0.8);
    }

    #[test]
    fn test_uncertainty_intervals() {
        let frame = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut ds = DempsterShaferSystem::new(frame);

        let mut evidence = MassFunction::new();
        evidence.assign_mass(vec!["A".to_string()], 0.4).unwrap();
        evidence
            .assign_mass(vec!["A".to_string(), "B".to_string()], 0.3)
            .unwrap();
        evidence
            .assign_mass(vec!["A".to_string(), "B".to_string(), "C".to_string()], 0.3)
            .unwrap();

        ds.add_evidence(evidence).unwrap();

        let (bel, pl) = ds.uncertainty_interval(&["A".to_string()]).unwrap();

        // Bel(A) = m({A}) = 0.4
        assert!((bel - 0.4).abs() < 1e-10);

        // Pl(A) = m({A}) + m({A,B}) + m({A,B,C}) = 1.0
        assert!((pl - 1.0).abs() < 1e-10);

        // Uncertainty = Pl - Bel
        assert!((pl - bel - 0.6).abs() < 1e-10);
    }
}
