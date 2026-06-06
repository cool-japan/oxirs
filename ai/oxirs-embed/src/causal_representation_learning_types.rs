//! Causal Representation Learning — Types
//!
//! Causal model types, intervention types, counterfactual types, and SCM (Structural Causal
//! Model) types extracted from the parent module.

use crate::ModelConfig;
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for causal representation learning
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CausalRepresentationConfig {
    pub base_config: ModelConfig,
    /// Causal discovery configuration
    pub causal_discovery: CausalDiscoveryConfig,
    /// Structural causal model configuration
    pub scm_config: StructuralCausalModelConfig,
    /// Interventional learning configuration
    pub intervention_config: InterventionConfig,
    /// Counterfactual reasoning configuration
    pub counterfactual_config: CounterfactualConfig,
    /// Disentanglement configuration
    pub disentanglement_config: DisentanglementConfig,
}

// ── Discovery ─────────────────────────────────────────────────────────────────

/// Causal discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalDiscoveryConfig {
    pub algorithm: CausalDiscoveryAlgorithm,
    pub significance_threshold: f32,
    pub max_parents: usize,
    pub use_interventions: bool,
    pub constraint_settings: ConstraintSettings,
    pub score_settings: ScoreSettings,
}

impl Default for CausalDiscoveryConfig {
    fn default() -> Self {
        Self {
            algorithm: CausalDiscoveryAlgorithm::PC,
            significance_threshold: 0.05,
            max_parents: 5,
            use_interventions: true,
            constraint_settings: ConstraintSettings::default(),
            score_settings: ScoreSettings::default(),
        }
    }
}

/// Causal discovery algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CausalDiscoveryAlgorithm {
    PC,
    FCI,
    GES,
    LiNGAM,
    NOTEARS,
    DirectLiNGAM,
    CAM,
}

/// Constraint-based algorithm settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSettings {
    pub independence_test: IndependenceTest,
    pub alpha: f32,
    pub stable: bool,
    pub max_cond_set_size: usize,
}

impl Default for ConstraintSettings {
    fn default() -> Self {
        Self {
            independence_test: IndependenceTest::PartialCorrelation,
            alpha: 0.05,
            stable: true,
            max_cond_set_size: 3,
        }
    }
}

/// Independence tests for constraint-based algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndependenceTest {
    PartialCorrelation,
    MutualInformation,
    KernelTest,
    DistanceCorrelation,
    HilbertSchmidt,
}

/// Score-based algorithm settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreSettings {
    pub score_function: ScoreFunction,
    pub penalty: f32,
    pub search_strategy: SearchStrategy,
    pub max_iterations: usize,
}

impl Default for ScoreSettings {
    fn default() -> Self {
        Self {
            score_function: ScoreFunction::BIC,
            penalty: 1.0,
            search_strategy: SearchStrategy::GreedyHillClimbing,
            max_iterations: 1000,
        }
    }
}

/// Scoring functions for causal models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoreFunction {
    BIC,
    AIC,
    LogLikelihood,
    MDL,
    BDeu,
    BGe,
}

/// Search strategies for structure learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    GreedyHillClimbing,
    TabuSearch,
    SimulatedAnnealing,
    GeneticAlgorithm,
    BeamSearch,
}

// ── SCM ──────────────────────────────────────────────────────────────────────

/// Structural causal model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralCausalModelConfig {
    pub variable_types: HashMap<String, VariableType>,
    pub functional_forms: HashMap<String, FunctionalForm>,
    pub noise_model: NoiseModel,
    pub identification: IdentificationStrategy,
}

impl Default for StructuralCausalModelConfig {
    fn default() -> Self {
        Self {
            variable_types: HashMap::new(),
            functional_forms: HashMap::new(),
            noise_model: NoiseModel::Gaussian,
            identification: IdentificationStrategy::BackDoorCriterion,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableType {
    Continuous,
    Discrete,
    Binary,
    Categorical,
    Ordinal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FunctionalForm {
    Linear,
    Nonlinear,
    Additive,
    Multiplicative,
    Polynomial,
    NeuralNetwork,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseModel {
    Gaussian,
    Uniform,
    Exponential,
    Laplace,
    StudentT,
    Mixture,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IdentificationStrategy {
    BackDoorCriterion,
    FrontDoorCriterion,
    InstrumentalVariable,
    DoCalculus,
    NaturalExperiment,
}

// ── Intervention ──────────────────────────────────────────────────────────────

/// Intervention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionConfig {
    pub intervention_types: Vec<InterventionType>,
    pub intervention_strength: f32,
    pub max_intervention_targets: usize,
    pub soft_interventions: bool,
    pub intervention_distribution: InterventionDistribution,
}

impl Default for InterventionConfig {
    fn default() -> Self {
        Self {
            intervention_types: vec![
                InterventionType::Do,
                InterventionType::Soft,
                InterventionType::Shift,
            ],
            intervention_strength: 1.0,
            max_intervention_targets: 3,
            soft_interventions: true,
            intervention_distribution: InterventionDistribution::Gaussian,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionType {
    Do,
    Soft,
    Shift,
    Noise,
    Mechanism,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionDistribution {
    Gaussian,
    Uniform,
    Delta,
    Mixture,
}

/// Concrete intervention specification
#[derive(Debug, Clone)]
pub struct Intervention {
    pub targets: Vec<String>,
    pub values: Array1<f32>,
    pub intervention_type: InterventionType,
    pub strength: f32,
}

impl Intervention {
    pub fn new(
        targets: Vec<String>,
        values: Array1<f32>,
        intervention_type: InterventionType,
    ) -> Self {
        Self {
            targets,
            values,
            intervention_type,
            strength: 1.0,
        }
    }
}

// ── Counterfactual ────────────────────────────────────────────────────────────

/// Counterfactual configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualConfig {
    pub reasoning_method: CounterfactualMethod,
    pub twin_network: TwinNetworkConfig,
    pub fairness_constraints: FairnessConstraints,
    pub explanation_config: ExplanationConfig,
}

impl Default for CounterfactualConfig {
    fn default() -> Self {
        Self {
            reasoning_method: CounterfactualMethod::TwinNetwork,
            twin_network: TwinNetworkConfig::default(),
            fairness_constraints: FairnessConstraints::default(),
            explanation_config: ExplanationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CounterfactualMethod {
    TwinNetwork,
    StructuralEquations,
    GAN,
    VAE,
    NormalizingFlows,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinNetworkConfig {
    pub shared_layers: usize,
    pub factual_layers: usize,
    pub counterfactual_layers: usize,
    pub consistency_weight: f32,
}

impl Default for TwinNetworkConfig {
    fn default() -> Self {
        Self {
            shared_layers: 3,
            factual_layers: 2,
            counterfactual_layers: 2,
            consistency_weight: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessConstraints {
    pub protected_attributes: Vec<String>,
    pub fairness_criteria: Vec<FairnessCriterion>,
    pub constraint_strength: f32,
}

impl Default for FairnessConstraints {
    fn default() -> Self {
        Self {
            protected_attributes: Vec::new(),
            fairness_criteria: vec![FairnessCriterion::CounterfactualFairness],
            constraint_strength: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FairnessCriterion {
    CounterfactualFairness,
    IndividualFairness,
    GroupFairness,
    EqualOpportunity,
    DemographicParity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationConfig {
    pub explanation_types: Vec<ExplanationType>,
    pub max_explanation_length: usize,
    pub include_confidence: bool,
}

impl Default for ExplanationConfig {
    fn default() -> Self {
        Self {
            explanation_types: vec![
                ExplanationType::Causal,
                ExplanationType::Counterfactual,
                ExplanationType::Contrastive,
            ],
            max_explanation_length: 10,
            include_confidence: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplanationType {
    Causal,
    Counterfactual,
    Contrastive,
    Abductive,
    Necessary,
    Sufficient,
}

/// Counterfactual query
#[derive(Debug, Clone)]
pub struct CounterfactualQuery {
    pub factual_evidence: HashMap<String, f32>,
    pub intervention: Intervention,
    pub query_variables: Vec<String>,
}

// ── Disentanglement ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisentanglementConfig {
    pub method: DisentanglementMethod,
    pub beta: f32,
    pub num_factors: usize,
    pub supervision: FactorSupervision,
}

impl Default for DisentanglementConfig {
    fn default() -> Self {
        Self {
            method: DisentanglementMethod::BetaVAE,
            beta: 4.0,
            num_factors: 10,
            supervision: FactorSupervision::Unsupervised,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisentanglementMethod {
    BetaVAE,
    FactorVAE,
    BetaTCVAE,
    ICA,
    SlowFeatureAnalysis,
    CausalVAE,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorSupervision {
    Unsupervised,
    WeaklySupervised,
    FullySupervised,
}

// ── Graph & Equations ─────────────────────────────────────────────────────────

/// Causal graph representation
#[derive(Debug, Clone)]
pub struct CausalGraph {
    pub variables: Vec<String>,
    pub adjacency: Array2<f32>,
    pub edge_weights: Array2<f32>,
    pub confounders: HashSet<(usize, usize)>,
}

impl CausalGraph {
    pub fn new(variables: Vec<String>) -> Self {
        let n = variables.len();
        Self {
            variables,
            adjacency: Array2::zeros((n, n)),
            edge_weights: Array2::zeros((n, n)),
            confounders: HashSet::new(),
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f32) {
        if from < self.adjacency.nrows() && to < self.adjacency.ncols() {
            self.adjacency[[from, to]] = 1.0;
            self.edge_weights[[from, to]] = weight;
        }
    }

    pub fn remove_edge(&mut self, from: usize, to: usize) {
        if from < self.adjacency.nrows() && to < self.adjacency.ncols() {
            self.adjacency[[from, to]] = 0.0;
            self.edge_weights[[from, to]] = 0.0;
        }
    }

    pub fn get_parents(&self, node: usize) -> Vec<usize> {
        let mut parents = Vec::new();
        for i in 0..self.adjacency.nrows() {
            if self.adjacency[[i, node]] > 0.0 {
                parents.push(i);
            }
        }
        parents
    }

    pub fn get_children(&self, node: usize) -> Vec<usize> {
        let mut children = Vec::new();
        for j in 0..self.adjacency.ncols() {
            if self.adjacency[[node, j]] > 0.0 {
                children.push(j);
            }
        }
        children
    }

    pub fn is_acyclic(&self) -> bool {
        let n = self.variables.len();
        let mut visited = vec![false; n];
        let mut rec_stack = vec![false; n];
        for i in 0..n {
            if !visited[i] && self.has_cycle_dfs(i, &mut visited, &mut rec_stack) {
                return false;
            }
        }
        true
    }

    fn has_cycle_dfs(
        &self,
        node: usize,
        visited: &mut Vec<bool>,
        rec_stack: &mut Vec<bool>,
    ) -> bool {
        visited[node] = true;
        rec_stack[node] = true;
        for child in self.get_children(node) {
            if (!visited[child] && self.has_cycle_dfs(child, visited, rec_stack))
                || rec_stack[child]
            {
                return true;
            }
        }
        rec_stack[node] = false;
        false
    }
}

/// Structural equation for a variable
#[derive(Debug, Clone)]
pub struct StructuralEquation {
    pub target: String,
    pub parents: Vec<String>,
    pub linear_coefficients: Array1<f32>,
    pub nonlinear_function: Option<Array2<f32>>,
    pub noise_variance: f32,
}

impl StructuralEquation {
    pub fn new(target: String, parents: Vec<String>) -> Self {
        let num_parents = parents.len();
        Self {
            target,
            parents,
            linear_coefficients: Array1::zeros(num_parents),
            nonlinear_function: None,
            noise_variance: 1.0,
        }
    }

    pub fn evaluate(&self, parent_values: &Array1<f32>) -> f32 {
        let mut result = 0.0;

        if parent_values.len() == self.linear_coefficients.len() {
            result += self.linear_coefficients.dot(parent_values);
        }

        if let Some(ref weights) = self.nonlinear_function {
            if weights.ncols() == parent_values.len() {
                let hidden = weights.dot(parent_values);
                result += hidden.mapv(|x| x.tanh()).sum();
            }
        }

        {
            use scirs2_core::random::{Random, RngExt};
            let mut random = Random::default();
            result += random.random::<f32>() * self.noise_variance.sqrt();
        }

        result
    }
}
