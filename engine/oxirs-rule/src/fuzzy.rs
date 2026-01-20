//! # Fuzzy Logic Reasoning Module
//!
//! This module provides fuzzy logic reasoning capabilities for handling
//! vague and uncertain knowledge in rule-based systems.
//!
//! ## Features
//!
//! - **Fuzzy Sets**: Membership functions (triangular, trapezoidal, gaussian)
//! - **Fuzzy Rules**: IF-THEN rules with fuzzy predicates
//! - **Fuzzy Inference**: Mamdani and Sugeno inference systems
//! - **Defuzzification**: Centroid, bisector, mean of maximum methods
//! - **Fuzzy Ontologies**: OWL ontologies with fuzzy concepts
//!
//! ## Example
//!
//! ```text
//! use oxirs_rule::fuzzy::*;
//!
//! // Create a fuzzy set for "tall"
//! let tall = FuzzySet::new(
//!     "tall".to_string(),
//!     MembershipFunction::Triangular { a: 160.0, b: 180.0, c: 200.0 }
//! );
//!
//! // Check membership degree
//! let degree = tall.membership(175.0);
//! assert!(degree > 0.5 && degree < 1.0);
//!
//! // Create fuzzy rules
//! let mut system = MamdaniFuzzySystem::new();
//! system.add_input_variable("height".to_string(), 0.0, 220.0);
//! system.add_output_variable("category".to_string(), 0.0, 10.0);
//!
//! // Add a fuzzy rule: IF height is tall THEN category is high
//! system.add_rule(FuzzyRule {
//!     antecedents: vec![("height".to_string(), "tall".to_string())],
//!     consequents: vec![("category".to_string(), "high".to_string())],
//!     weight: 1.0,
//! });
//!
//! // Perform inference
//! let result = system.infer(&[("height".to_string(), 175.0)]).unwrap();
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array1;
use std::collections::HashMap;

/// Membership function types for fuzzy sets
#[derive(Debug, Clone)]
pub enum MembershipFunction {
    /// Triangular membership function: μ(x) = max(min((x-a)/(b-a), (c-x)/(c-b)), 0)
    Triangular { a: f64, b: f64, c: f64 },
    /// Trapezoidal membership function
    Trapezoidal { a: f64, b: f64, c: f64, d: f64 },
    /// Gaussian membership function: μ(x) = exp(-((x-c)^2)/(2*σ^2))
    Gaussian { center: f64, sigma: f64 },
    /// Sigmoid membership function: μ(x) = 1/(1 + exp(-a(x-c)))
    Sigmoid { a: f64, c: f64 },
    /// Singleton (crisp value)
    Singleton { value: f64 },
}

impl MembershipFunction {
    /// Compute membership degree for a given value
    pub fn membership(&self, x: f64) -> f64 {
        match self {
            MembershipFunction::Triangular { a, b, c } => {
                if x <= *a || x >= *c {
                    0.0
                } else if x <= *b {
                    (x - a) / (b - a)
                } else {
                    (c - x) / (c - b)
                }
            }
            MembershipFunction::Trapezoidal { a, b, c, d } => {
                if x <= *a || x >= *d {
                    0.0
                } else if x >= *b && x <= *c {
                    1.0
                } else if x < *b {
                    (x - a) / (b - a)
                } else {
                    (d - x) / (d - c)
                }
            }
            MembershipFunction::Gaussian { center, sigma } => {
                let exp_arg = -((x - center).powi(2)) / (2.0 * sigma.powi(2));
                exp_arg.exp()
            }
            MembershipFunction::Sigmoid { a, c } => 1.0 / (1.0 + (-a * (x - c)).exp()),
            MembershipFunction::Singleton { value } => {
                if (x - value).abs() < 1e-10 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// Fuzzy set with a name and membership function
#[derive(Debug, Clone)]
pub struct FuzzySet {
    /// Name of the fuzzy set
    pub name: String,
    /// Membership function
    pub function: MembershipFunction,
}

impl FuzzySet {
    /// Create a new fuzzy set
    pub fn new(name: String, function: MembershipFunction) -> Self {
        Self { name, function }
    }

    /// Get membership degree for a value
    pub fn membership(&self, x: f64) -> f64 {
        self.function.membership(x)
    }

    /// Get the support (values where membership > 0)
    pub fn support(&self, min: f64, max: f64, num_points: usize) -> Vec<f64> {
        let step = (max - min) / (num_points as f64);
        (0..num_points)
            .map(|i| min + step * (i as f64))
            .filter(|&x| self.membership(x) > 1e-10)
            .collect()
    }

    /// Get the core (values where membership = 1)
    pub fn core(&self, min: f64, max: f64, num_points: usize) -> Vec<f64> {
        let step = (max - min) / (num_points as f64);
        (0..num_points)
            .map(|i| min + step * (i as f64))
            .filter(|&x| (self.membership(x) - 1.0).abs() < 1e-10)
            .collect()
    }
}

/// Fuzzy rule with antecedents and consequents
#[derive(Debug, Clone)]
pub struct FuzzyRule {
    /// Antecedents: (variable, fuzzy_set_name)
    pub antecedents: Vec<(String, String)>,
    /// Consequents: (variable, fuzzy_set_name)
    pub consequents: Vec<(String, String)>,
    /// Rule weight (default: 1.0)
    pub weight: f64,
}

/// Fuzzy variable with domain and associated fuzzy sets
#[derive(Debug, Clone)]
pub struct FuzzyVariable {
    /// Variable name
    pub name: String,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Associated fuzzy sets
    pub fuzzy_sets: HashMap<String, FuzzySet>,
}

impl FuzzyVariable {
    /// Create a new fuzzy variable
    pub fn new(name: String, min: f64, max: f64) -> Self {
        Self {
            name,
            min,
            max,
            fuzzy_sets: HashMap::new(),
        }
    }

    /// Add a fuzzy set to this variable
    pub fn add_fuzzy_set(&mut self, fuzzy_set: FuzzySet) {
        self.fuzzy_sets.insert(fuzzy_set.name.clone(), fuzzy_set);
    }

    /// Get membership degree for a value in a specific fuzzy set
    pub fn membership(&self, fuzzy_set_name: &str, value: f64) -> Result<f64> {
        self.fuzzy_sets
            .get(fuzzy_set_name)
            .map(|fs| fs.membership(value))
            .ok_or_else(|| anyhow!("Fuzzy set {} not found", fuzzy_set_name))
    }
}

/// Mamdani-style fuzzy inference system
#[derive(Debug, Clone)]
pub struct MamdaniFuzzySystem {
    /// Input variables
    input_variables: HashMap<String, FuzzyVariable>,
    /// Output variables
    output_variables: HashMap<String, FuzzyVariable>,
    /// Fuzzy rules
    rules: Vec<FuzzyRule>,
    /// T-norm (AND operation): min, product, lukasiewicz
    t_norm: TNorm,
    /// T-conorm (OR operation): max, probabilistic_sum
    t_conorm: TConorm,
    /// Defuzzification method
    defuzz_method: DefuzzificationMethod,
}

/// T-norm (fuzzy AND)
#[derive(Debug, Clone, Copy)]
pub enum TNorm {
    /// Minimum: min(a, b)
    Min,
    /// Product: a * b
    Product,
    /// Lukasiewicz: max(0, a + b - 1)
    Lukasiewicz,
}

impl TNorm {
    /// Apply the T-norm
    pub fn apply(&self, a: f64, b: f64) -> f64 {
        match self {
            TNorm::Min => a.min(b),
            TNorm::Product => a * b,
            TNorm::Lukasiewicz => (a + b - 1.0).max(0.0),
        }
    }
}

/// T-conorm (fuzzy OR)
#[derive(Debug, Clone, Copy)]
pub enum TConorm {
    /// Maximum: max(a, b)
    Max,
    /// Probabilistic sum: a + b - a*b
    ProbabilisticSum,
}

impl TConorm {
    /// Apply the T-conorm
    pub fn apply(&self, a: f64, b: f64) -> f64 {
        match self {
            TConorm::Max => a.max(b),
            TConorm::ProbabilisticSum => a + b - a * b,
        }
    }
}

/// Defuzzification method
#[derive(Debug, Clone, Copy)]
pub enum DefuzzificationMethod {
    /// Centroid (center of gravity)
    Centroid,
    /// Bisector (divides area in half)
    Bisector,
    /// Mean of maximum
    MeanOfMaximum,
    /// Smallest of maximum
    SmallestOfMaximum,
    /// Largest of maximum
    LargestOfMaximum,
}

impl MamdaniFuzzySystem {
    /// Create a new Mamdani fuzzy system
    pub fn new() -> Self {
        Self {
            input_variables: HashMap::new(),
            output_variables: HashMap::new(),
            rules: Vec::new(),
            t_norm: TNorm::Min,
            t_conorm: TConorm::Max,
            defuzz_method: DefuzzificationMethod::Centroid,
        }
    }

    /// Add an input variable
    pub fn add_input_variable(&mut self, name: String, min: f64, max: f64) {
        let var = FuzzyVariable::new(name.clone(), min, max);
        self.input_variables.insert(name, var);
    }

    /// Add an output variable
    pub fn add_output_variable(&mut self, name: String, min: f64, max: f64) {
        let var = FuzzyVariable::new(name.clone(), min, max);
        self.output_variables.insert(name, var);
    }

    /// Add a fuzzy set to an input variable
    pub fn add_input_fuzzy_set(&mut self, var_name: &str, fuzzy_set: FuzzySet) -> Result<()> {
        self.input_variables
            .get_mut(var_name)
            .ok_or_else(|| anyhow!("Input variable {} not found", var_name))?
            .add_fuzzy_set(fuzzy_set);
        Ok(())
    }

    /// Add a fuzzy set to an output variable
    pub fn add_output_fuzzy_set(&mut self, var_name: &str, fuzzy_set: FuzzySet) -> Result<()> {
        self.output_variables
            .get_mut(var_name)
            .ok_or_else(|| anyhow!("Output variable {} not found", var_name))?
            .add_fuzzy_set(fuzzy_set);
        Ok(())
    }

    /// Add a fuzzy rule
    pub fn add_rule(&mut self, rule: FuzzyRule) {
        self.rules.push(rule);
    }

    /// Set T-norm
    pub fn set_t_norm(&mut self, t_norm: TNorm) {
        self.t_norm = t_norm;
    }

    /// Set T-conorm
    pub fn set_t_conorm(&mut self, t_conorm: TConorm) {
        self.t_conorm = t_conorm;
    }

    /// Set defuzzification method
    pub fn set_defuzzification_method(&mut self, method: DefuzzificationMethod) {
        self.defuzz_method = method;
    }

    /// Perform fuzzy inference
    pub fn infer(&self, inputs: &[(String, f64)]) -> Result<HashMap<String, f64>> {
        // Step 1: Fuzzification - compute membership degrees for inputs
        let mut input_memberships: HashMap<String, HashMap<String, f64>> = HashMap::new();

        for (var_name, value) in inputs {
            let var = self
                .input_variables
                .get(var_name)
                .ok_or_else(|| anyhow!("Input variable {} not found", var_name))?;

            let mut memberships = HashMap::new();
            for (fs_name, fuzzy_set) in &var.fuzzy_sets {
                memberships.insert(fs_name.clone(), fuzzy_set.membership(*value));
            }
            input_memberships.insert(var_name.clone(), memberships);
        }

        // Step 2: Rule evaluation - compute firing strength for each rule
        let mut output_fuzzy_values: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        for rule in &self.rules {
            // Compute antecedent firing strength using T-norm (AND)
            let mut firing_strength = 1.0;
            for (var_name, fs_name) in &rule.antecedents {
                let membership = input_memberships
                    .get(var_name)
                    .and_then(|m| m.get(fs_name))
                    .copied()
                    .unwrap_or(0.0);
                firing_strength = self.t_norm.apply(firing_strength, membership);
            }

            // Apply rule weight
            firing_strength *= rule.weight;

            // Apply to consequents
            for (var_name, fs_name) in &rule.consequents {
                output_fuzzy_values
                    .entry(var_name.clone())
                    .or_default()
                    .push((fs_name.clone(), firing_strength));
            }
        }

        // Step 3: Aggregation and Defuzzification
        let mut results = HashMap::new();

        for (var_name, fuzzy_values) in output_fuzzy_values {
            let output_var = self
                .output_variables
                .get(&var_name)
                .ok_or_else(|| anyhow!("Output variable {} not found", var_name))?;

            // Build aggregated fuzzy output
            let num_points = 1000;
            let step = (output_var.max - output_var.min) / (num_points as f64);

            let mut aggregated_membership = vec![0.0; num_points];

            for (i, agg_mem) in aggregated_membership.iter_mut().enumerate() {
                let x = output_var.min + step * (i as f64);

                for (fs_name, firing_strength) in &fuzzy_values {
                    let fs = output_var
                        .fuzzy_sets
                        .get(fs_name)
                        .ok_or_else(|| anyhow!("Fuzzy set {} not found", fs_name))?;

                    let membership = fs.membership(x).min(*firing_strength);
                    *agg_mem = self.t_conorm.apply(*agg_mem, membership);
                }
            }

            // Defuzzify
            let crisp_value = self.defuzzify(
                &Array1::from_vec(aggregated_membership),
                output_var.min,
                output_var.max,
            )?;

            results.insert(var_name, crisp_value);
        }

        Ok(results)
    }

    /// Defuzzify an aggregated fuzzy output
    fn defuzzify(&self, membership: &Array1<f64>, min: f64, max: f64) -> Result<f64> {
        let num_points = membership.len();
        let step = (max - min) / (num_points as f64);

        match self.defuzz_method {
            DefuzzificationMethod::Centroid => {
                // Centroid: ∫ x * μ(x) dx / ∫ μ(x) dx
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for i in 0..num_points {
                    let x = min + step * (i as f64);
                    numerator += x * membership[i];
                    denominator += membership[i];
                }

                if denominator < 1e-10 {
                    return Err(anyhow!("Cannot defuzzify: membership sum is zero"));
                }

                Ok(numerator / denominator)
            }
            DefuzzificationMethod::MeanOfMaximum => {
                // Find maximum membership degree
                let max_membership = membership
                    .iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);

                if max_membership < 1e-10 {
                    return Err(anyhow!("Cannot defuzzify: max membership is zero"));
                }

                // Find all indices with maximum membership
                let max_indices: Vec<usize> = membership
                    .iter()
                    .enumerate()
                    .filter(|(_, &m)| (m - max_membership).abs() < 1e-10)
                    .map(|(i, _)| i)
                    .collect();

                // Return mean of maximum values
                let sum: f64 = max_indices.iter().map(|&i| min + step * (i as f64)).sum();
                Ok(sum / (max_indices.len() as f64))
            }
            DefuzzificationMethod::SmallestOfMaximum => {
                let max_membership = membership
                    .iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);

                if max_membership < 1e-10 {
                    return Err(anyhow!("Cannot defuzzify: max membership is zero"));
                }

                membership
                    .iter()
                    .enumerate()
                    .find(|(_, &m)| (m - max_membership).abs() < 1e-10)
                    .map(|(i, _)| min + step * (i as f64))
                    .ok_or_else(|| anyhow!("Cannot find maximum"))
            }
            DefuzzificationMethod::LargestOfMaximum => {
                let max_membership = membership
                    .iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);

                if max_membership < 1e-10 {
                    return Err(anyhow!("Cannot defuzzify: max membership is zero"));
                }

                membership
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, &m)| (m - max_membership).abs() < 1e-10)
                    .map(|(i, _)| min + step * (i as f64))
                    .ok_or_else(|| anyhow!("Cannot find maximum"))
            }
            DefuzzificationMethod::Bisector => {
                // Find area under curve
                let total_area: f64 = membership.iter().sum::<f64>() * step;
                let half_area = total_area / 2.0;

                // Find x where area to the left equals half the total area
                let mut cumulative_area = 0.0;
                for i in 0..num_points {
                    cumulative_area += membership[i] * step;
                    if cumulative_area >= half_area {
                        return Ok(min + step * (i as f64));
                    }
                }

                Err(anyhow!("Cannot compute bisector"))
            }
        }
    }
}

impl Default for MamdaniFuzzySystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Fuzzy rule engine integrating fuzzy logic with standard rules
#[derive(Debug)]
pub struct FuzzyRuleEngine {
    /// Mamdani fuzzy system
    fuzzy_system: MamdaniFuzzySystem,
    /// Mapping from standard rules to fuzzy rules
    rule_mapping: HashMap<String, FuzzyRule>,
}

impl FuzzyRuleEngine {
    /// Create a new fuzzy rule engine
    pub fn new() -> Self {
        Self {
            fuzzy_system: MamdaniFuzzySystem::new(),
            rule_mapping: HashMap::new(),
        }
    }

    /// Add a fuzzy rule from a standard rule
    pub fn add_fuzzy_rule(&mut self, rule: Rule, fuzzy_rule: FuzzyRule) {
        self.rule_mapping
            .insert(rule.name.clone(), fuzzy_rule.clone());
        self.fuzzy_system.add_rule(fuzzy_rule);
    }

    /// Infer with fuzzy logic
    pub fn fuzzy_infer(&self, inputs: &[(String, f64)]) -> Result<HashMap<String, f64>> {
        self.fuzzy_system.infer(inputs)
    }

    /// Convert RuleAtoms to fuzzy inputs
    pub fn atoms_to_fuzzy_inputs(&self, atoms: &[RuleAtom]) -> Vec<(String, f64)> {
        let mut inputs = Vec::new();

        for atom in atoms {
            if let RuleAtom::Triple {
                subject,
                predicate,
                object,
            } = atom
            {
                if let (Term::Constant(s), Term::Constant(p), Term::Literal(o)) =
                    (subject, predicate, object)
                {
                    // Try to parse the object as a number
                    if let Ok(value) = o.parse::<f64>() {
                        inputs.push((format!("{}_{}", s, p), value));
                    }
                }
            }
        }

        inputs
    }

    /// Get the fuzzy system reference
    pub fn fuzzy_system(&self) -> &MamdaniFuzzySystem {
        &self.fuzzy_system
    }

    /// Get mutable fuzzy system reference
    pub fn fuzzy_system_mut(&mut self) -> &mut MamdaniFuzzySystem {
        &mut self.fuzzy_system
    }
}

impl Default for FuzzyRuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangular_membership() {
        let mf = MembershipFunction::Triangular {
            a: 0.0,
            b: 5.0,
            c: 10.0,
        };

        assert_eq!(mf.membership(-1.0), 0.0);
        assert_eq!(mf.membership(0.0), 0.0);
        assert_eq!(mf.membership(5.0), 1.0);
        assert_eq!(mf.membership(10.0), 0.0);
        assert_eq!(mf.membership(11.0), 0.0);
    }

    #[test]
    fn test_trapezoidal_membership() {
        let mf = MembershipFunction::Trapezoidal {
            a: 0.0,
            b: 2.0,
            c: 8.0,
            d: 10.0,
        };

        assert_eq!(mf.membership(-1.0), 0.0);
        assert_eq!(mf.membership(0.0), 0.0);
        assert_eq!(mf.membership(5.0), 1.0);
        assert_eq!(mf.membership(10.0), 0.0);
        assert_eq!(mf.membership(11.0), 0.0);
    }

    #[test]
    fn test_gaussian_membership() {
        let mf = MembershipFunction::Gaussian {
            center: 5.0,
            sigma: 1.0,
        };

        let mem_center = mf.membership(5.0);
        let mem_away = mf.membership(8.0);

        assert!((mem_center - 1.0).abs() < 1e-10);
        assert!(mem_away < 0.1);
    }

    #[test]
    fn test_fuzzy_set() {
        let fs = FuzzySet::new(
            "medium".to_string(),
            MembershipFunction::Triangular {
                a: 0.0,
                b: 5.0,
                c: 10.0,
            },
        );

        assert_eq!(fs.membership(5.0), 1.0);
        assert!(fs.membership(2.5) > 0.0 && fs.membership(2.5) < 1.0);
    }

    #[test]
    fn test_t_norm() {
        let min_norm = TNorm::Min;
        assert_eq!(min_norm.apply(0.7, 0.3), 0.3);

        let product_norm = TNorm::Product;
        assert_eq!(product_norm.apply(0.5, 0.5), 0.25);
    }

    #[test]
    fn test_t_conorm() {
        let max_conorm = TConorm::Max;
        assert_eq!(max_conorm.apply(0.7, 0.3), 0.7);

        let prob_conorm = TConorm::ProbabilisticSum;
        let result = prob_conorm.apply(0.5, 0.5);
        assert!((result - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_mamdani_fuzzy_system() {
        let mut system = MamdaniFuzzySystem::new();

        // Add input variable "temperature" with range 0-40
        system.add_input_variable("temperature".to_string(), 0.0, 40.0);

        // Add fuzzy sets for temperature
        system
            .add_input_fuzzy_set(
                "temperature",
                FuzzySet::new(
                    "cold".to_string(),
                    MembershipFunction::Triangular {
                        a: 0.0,
                        b: 0.0,
                        c: 20.0,
                    },
                ),
            )
            .unwrap();

        system
            .add_input_fuzzy_set(
                "temperature",
                FuzzySet::new(
                    "hot".to_string(),
                    MembershipFunction::Triangular {
                        a: 20.0,
                        b: 40.0,
                        c: 40.0,
                    },
                ),
            )
            .unwrap();

        // Add output variable "fan_speed" with range 0-100
        system.add_output_variable("fan_speed".to_string(), 0.0, 100.0);

        system
            .add_output_fuzzy_set(
                "fan_speed",
                FuzzySet::new(
                    "low".to_string(),
                    MembershipFunction::Triangular {
                        a: 0.0,
                        b: 0.0,
                        c: 50.0,
                    },
                ),
            )
            .unwrap();

        system
            .add_output_fuzzy_set(
                "fan_speed",
                FuzzySet::new(
                    "high".to_string(),
                    MembershipFunction::Triangular {
                        a: 50.0,
                        b: 100.0,
                        c: 100.0,
                    },
                ),
            )
            .unwrap();

        // Add rules
        system.add_rule(FuzzyRule {
            antecedents: vec![("temperature".to_string(), "cold".to_string())],
            consequents: vec![("fan_speed".to_string(), "low".to_string())],
            weight: 1.0,
        });

        system.add_rule(FuzzyRule {
            antecedents: vec![("temperature".to_string(), "hot".to_string())],
            consequents: vec![("fan_speed".to_string(), "high".to_string())],
            weight: 1.0,
        });

        // Test inference
        let result = system.infer(&[("temperature".to_string(), 30.0)]).unwrap();

        assert!(result.contains_key("fan_speed"));
        let fan_speed = result["fan_speed"];
        assert!((0.0..=100.0).contains(&fan_speed));
    }

    #[test]
    fn test_fuzzy_rule_engine() {
        let mut engine = FuzzyRuleEngine::new();

        // Setup input variable
        engine
            .fuzzy_system_mut()
            .add_input_variable("age".to_string(), 0.0, 100.0);

        engine
            .fuzzy_system_mut()
            .add_input_fuzzy_set(
                "age",
                FuzzySet::new(
                    "young".to_string(),
                    MembershipFunction::Triangular {
                        a: 0.0,
                        b: 0.0,
                        c: 30.0,
                    },
                ),
            )
            .unwrap();

        // Setup output variable
        engine
            .fuzzy_system_mut()
            .add_output_variable("category".to_string(), 0.0, 10.0);

        engine
            .fuzzy_system_mut()
            .add_output_fuzzy_set(
                "category",
                FuzzySet::new(
                    "student".to_string(),
                    MembershipFunction::Triangular {
                        a: 0.0,
                        b: 5.0,
                        c: 10.0,
                    },
                ),
            )
            .unwrap();

        // Add fuzzy rule
        let rule = Rule {
            name: "young_student".to_string(),
            body: vec![],
            head: vec![],
        };

        let fuzzy_rule = FuzzyRule {
            antecedents: vec![("age".to_string(), "young".to_string())],
            consequents: vec![("category".to_string(), "student".to_string())],
            weight: 1.0,
        };

        engine.add_fuzzy_rule(rule, fuzzy_rule);

        // Test inference
        let result = engine.fuzzy_infer(&[("age".to_string(), 20.0)]).unwrap();
        assert!(result.contains_key("category"));
    }
}
