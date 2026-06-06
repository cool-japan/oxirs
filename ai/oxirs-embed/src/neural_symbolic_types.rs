//! Neural-Symbolic Types
//!
//! Type definitions: NeuralSymbolicConfig, SymbolicRule, NeuralEvidence, IntegrationResult,
//! rule weight types, unification types, and all configuration structs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for neural-symbolic integration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NeuralSymbolicConfig {
    pub base_config: crate::ModelConfig,
    /// Symbolic reasoning configuration
    pub symbolic_config: SymbolicReasoningConfig,
    /// Logic integration configuration
    pub logic_config: LogicIntegrationConfig,
    /// Knowledge integration configuration
    pub knowledge_config: KnowledgeIntegrationConfig,
    /// Neuro-symbolic architecture configuration
    pub architecture_config: NeuroSymbolicArchitectureConfig,
    /// Constraint satisfaction configuration
    pub constraint_config: ConstraintSatisfactionConfig,
}

/// Symbolic reasoning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicReasoningConfig {
    /// Reasoning engines to use
    pub reasoning_engines: Vec<ReasoningEngine>,
    /// Logic programming settings
    pub logic_programming: LogicProgrammingConfig,
    /// Rule-based reasoning settings
    pub rule_based: RuleBasedConfig,
    /// Ontological reasoning settings
    pub ontological: OntologicalConfig,
}

impl Default for SymbolicReasoningConfig {
    fn default() -> Self {
        Self {
            reasoning_engines: vec![
                ReasoningEngine::Description,
                ReasoningEngine::RuleBased,
                ReasoningEngine::FirstOrder,
            ],
            logic_programming: LogicProgrammingConfig::default(),
            rule_based: RuleBasedConfig::default(),
            ontological: OntologicalConfig::default(),
        }
    }
}

/// Reasoning engine types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningEngine {
    /// Description Logic
    Description,
    /// Rule-based reasoning
    RuleBased,
    /// First-order logic
    FirstOrder,
    /// Probabilistic logic
    Probabilistic,
    /// Temporal logic
    Temporal,
    /// Modal logic
    Modal,
    /// Non-monotonic reasoning
    NonMonotonic,
}

/// Logic programming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicProgrammingConfig {
    pub use_datalog: bool,
    pub use_prolog: bool,
    pub use_asp: bool,
    pub use_clp: bool,
}

impl Default for LogicProgrammingConfig {
    fn default() -> Self {
        Self {
            use_datalog: true,
            use_prolog: false,
            use_asp: true,
            use_clp: false,
        }
    }
}

/// Rule-based reasoning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleBasedConfig {
    pub forward_chaining: bool,
    pub backward_chaining: bool,
    pub confidence_threshold: f32,
    pub max_depth: usize,
}

impl Default for RuleBasedConfig {
    fn default() -> Self {
        Self {
            forward_chaining: true,
            backward_chaining: true,
            confidence_threshold: 0.7,
            max_depth: 10,
        }
    }
}

/// Ontological reasoning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologicalConfig {
    pub owl_profile: OWLProfile,
    pub class_hierarchy: bool,
    pub property_reasoning: bool,
    pub consistency_checking: bool,
}

impl Default for OntologicalConfig {
    fn default() -> Self {
        Self {
            owl_profile: OWLProfile::OWL2EL,
            class_hierarchy: true,
            property_reasoning: true,
            consistency_checking: true,
        }
    }
}

/// OWL profiles for ontological reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OWLProfile {
    OWL2EL,
    OWL2QL,
    OWL2RL,
    OWL2Full,
}

/// Logic integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicIntegrationConfig {
    pub integration_methods: Vec<IntegrationMethod>,
    pub fuzzy_logic: FuzzyLogicConfig,
    pub probabilistic_logic: ProbabilisticLogicConfig,
    pub temporal_logic: TemporalLogicConfig,
}

impl Default for LogicIntegrationConfig {
    fn default() -> Self {
        Self {
            integration_methods: vec![
                IntegrationMethod::LogicTensors,
                IntegrationMethod::NeuralModuleNetworks,
                IntegrationMethod::DifferentiableReasoning,
            ],
            fuzzy_logic: FuzzyLogicConfig::default(),
            probabilistic_logic: ProbabilisticLogicConfig::default(),
            temporal_logic: TemporalLogicConfig::default(),
        }
    }
}

/// Integration methods for neural-symbolic systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationMethod {
    LogicTensors,
    NeuralModuleNetworks,
    DifferentiableReasoning,
    SemanticLoss,
    LogicAttention,
    SymbolicGrounding,
}

/// Fuzzy logic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyLogicConfig {
    pub t_norm: TNorm,
    pub t_conorm: TConorm,
    pub implication: ImplicationOperator,
    pub negation: NegationOperator,
}

impl Default for FuzzyLogicConfig {
    fn default() -> Self {
        Self {
            t_norm: TNorm::Product,
            t_conorm: TConorm::ProbabilisticSum,
            implication: ImplicationOperator::Lukasiewicz,
            negation: NegationOperator::Standard,
        }
    }
}

/// T-norms for fuzzy conjunction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TNorm {
    Minimum,
    Product,
    Lukasiewicz,
    Drastic,
    Nilpotent,
}

/// T-conorms for fuzzy disjunction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TConorm {
    Maximum,
    ProbabilisticSum,
    BoundedSum,
    Drastic,
    Nilpotent,
}

/// Implication operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplicationOperator {
    Lukasiewicz,
    Godel,
    Product,
    Kleene,
}

/// Negation operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NegationOperator {
    Standard,
    Sugeno,
    Yager,
}

/// Probabilistic logic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticLogicConfig {
    pub use_mln: bool,
    pub use_psl: bool,
    pub use_problog: bool,
    pub inference_method: ProbabilisticInference,
}

impl Default for ProbabilisticLogicConfig {
    fn default() -> Self {
        Self {
            use_mln: true,
            use_psl: false,
            use_problog: false,
            inference_method: ProbabilisticInference::VariationalInference,
        }
    }
}

/// Probabilistic inference methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProbabilisticInference {
    ExactInference,
    VariationalInference,
    MCMC,
    BeliefPropagation,
    ExpectationMaximization,
}

/// Temporal logic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalLogicConfig {
    pub use_ltl: bool,
    pub use_ctl: bool,
    pub use_mtl: bool,
    pub time_window: usize,
}

impl Default for TemporalLogicConfig {
    fn default() -> Self {
        Self {
            use_ltl: true,
            use_ctl: false,
            use_mtl: false,
            time_window: 10,
        }
    }
}

/// Knowledge integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeIntegrationConfig {
    pub knowledge_sources: Vec<KnowledgeSource>,
    pub grounding_methods: Vec<GroundingMethod>,
    pub external_kbs: Vec<String>,
    pub validation_config: ValidationConfig,
}

impl Default for KnowledgeIntegrationConfig {
    fn default() -> Self {
        Self {
            knowledge_sources: vec![
                KnowledgeSource::Ontologies,
                KnowledgeSource::Rules,
                KnowledgeSource::CommonSense,
            ],
            grounding_methods: vec![
                GroundingMethod::EntityLinking,
                GroundingMethod::ConceptAlignment,
                GroundingMethod::SemanticParsing,
            ],
            external_kbs: vec![
                "DBpedia".to_string(),
                "Wikidata".to_string(),
                "ConceptNet".to_string(),
            ],
            validation_config: ValidationConfig::default(),
        }
    }
}

/// Knowledge sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KnowledgeSource {
    Ontologies,
    Rules,
    CommonSense,
    Domain,
    Factual,
    Procedural,
}

/// Grounding methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroundingMethod {
    EntityLinking,
    ConceptAlignment,
    SemanticParsing,
    SymbolGrounding,
    Contextualization,
}

/// Knowledge validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub consistency_check: bool,
    pub completeness_check: bool,
    pub confidence_threshold: f32,
    pub validation_frequency: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            consistency_check: true,
            completeness_check: false,
            confidence_threshold: 0.8,
            validation_frequency: 100,
        }
    }
}

/// Neuro-symbolic architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroSymbolicArchitectureConfig {
    pub architecture_type: NeuroSymbolicArchitecture,
    pub neural_config: NeuralComponentConfig,
    pub symbolic_config: SymbolicComponentConfig,
    pub integration_config: IntegrationLayerConfig,
}

impl Default for NeuroSymbolicArchitectureConfig {
    fn default() -> Self {
        Self {
            architecture_type: NeuroSymbolicArchitecture::HybridPipeline,
            neural_config: NeuralComponentConfig::default(),
            symbolic_config: SymbolicComponentConfig::default(),
            integration_config: IntegrationLayerConfig::default(),
        }
    }
}

/// Neuro-symbolic architecture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuroSymbolicArchitecture {
    HybridPipeline,
    DeepIntegration,
    LooseCoupling,
    CoProcessing,
    EndToEndDifferentiable,
}

/// Neural component configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralComponentConfig {
    pub layers: Vec<LayerConfig>,
    pub activations: Vec<ActivationFunction>,
    pub dropout_rates: Vec<f32>,
}

impl Default for NeuralComponentConfig {
    fn default() -> Self {
        Self {
            layers: vec![
                LayerConfig {
                    size: 512,
                    layer_type: LayerType::Dense,
                },
                LayerConfig {
                    size: 256,
                    layer_type: LayerType::Dense,
                },
                LayerConfig {
                    size: 128,
                    layer_type: LayerType::Dense,
                },
            ],
            activations: vec![
                ActivationFunction::ReLU,
                ActivationFunction::ReLU,
                ActivationFunction::Sigmoid,
            ],
            dropout_rates: vec![0.1, 0.2, 0.1],
        }
    }
}

/// Layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub size: usize,
    pub layer_type: LayerType,
}

/// Layer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Dense,
    Convolutional,
    Attention,
    Logic,
    Symbolic,
}

/// Activation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    GELU,
    Swish,
    LogicActivation,
}

/// Symbolic component configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicComponentConfig {
    pub vocab_size: usize,
    pub max_formula_length: usize,
    pub operators: Vec<LogicOperator>,
    pub reasoning_depth: usize,
}

impl Default for SymbolicComponentConfig {
    fn default() -> Self {
        Self {
            vocab_size: 10000,
            max_formula_length: 50,
            operators: vec![
                LogicOperator::And,
                LogicOperator::Or,
                LogicOperator::Not,
                LogicOperator::Implies,
                LogicOperator::Exists,
                LogicOperator::ForAll,
            ],
            reasoning_depth: 5,
        }
    }
}

/// Logic operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicOperator {
    And,
    Or,
    Not,
    Implies,
    Equivalent,
    Exists,
    ForAll,
    Equals,
    GreaterThan,
    LessThan,
}

/// Integration layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationLayerConfig {
    pub method: LayerIntegrationMethod,
    pub attention_config: AttentionConfig,
    pub fusion_strategy: FusionStrategy,
}

impl Default for IntegrationLayerConfig {
    fn default() -> Self {
        Self {
            method: LayerIntegrationMethod::CrossAttention,
            attention_config: AttentionConfig::default(),
            fusion_strategy: FusionStrategy::Concatenation,
        }
    }
}

/// Layer integration methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerIntegrationMethod {
    Concatenation,
    CrossAttention,
    FeatureFusion,
    LogicAttention,
    SymbolicGrounding,
}

/// Attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout_rate: f32,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            dropout_rate: 0.1,
        }
    }
}

/// Fusion strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    Concatenation,
    Addition,
    Multiplication,
    Attention,
    Gating,
}

/// Constraint satisfaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSatisfactionConfig {
    pub constraint_types: Vec<ConstraintType>,
    pub solver_config: SolverConfig,
    pub soft_constraints: bool,
    pub constraint_weights: HashMap<String, f32>,
}

impl Default for ConstraintSatisfactionConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("logical_consistency".to_string(), 1.0);
        weights.insert("domain_constraints".to_string(), 0.8);
        weights.insert("type_constraints".to_string(), 0.9);

        Self {
            constraint_types: vec![
                ConstraintType::Logical,
                ConstraintType::Semantic,
                ConstraintType::Domain,
                ConstraintType::Type,
            ],
            solver_config: SolverConfig::default(),
            soft_constraints: true,
            constraint_weights: weights,
        }
    }
}

/// Constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    Logical,
    Semantic,
    Domain,
    Type,
    Temporal,
    Causal,
}

/// Solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub solver_type: SolverType,
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub timeout: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            solver_type: SolverType::GradientDescent,
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            timeout: 10.0,
        }
    }
}

/// Solver types for constraint satisfaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolverType {
    GradientDescent,
    SimulatedAnnealing,
    GeneticAlgorithm,
    TabuSearch,
    ConstraintPropagation,
    BacktrackingSearch,
}

/// Logical formula representation
#[derive(Debug, Clone)]
pub struct LogicalFormula {
    pub structure: FormulaStructure,
    pub truth_value: f32,
    pub confidence: f32,
    pub variables: std::collections::HashSet<String>,
}

/// Formula structure
#[derive(Debug, Clone)]
pub enum FormulaStructure {
    Atom(String),
    Negation(Box<FormulaStructure>),
    Conjunction(Vec<FormulaStructure>),
    Disjunction(Vec<FormulaStructure>),
    Implication(Box<FormulaStructure>, Box<FormulaStructure>),
    Equivalence(Box<FormulaStructure>, Box<FormulaStructure>),
    Exists(String, Box<FormulaStructure>),
    ForAll(String, Box<FormulaStructure>),
}

impl LogicalFormula {
    pub fn new_atom(predicate: String) -> Self {
        let mut variables = std::collections::HashSet::new();
        variables.insert(predicate.clone());

        Self {
            structure: FormulaStructure::Atom(predicate),
            truth_value: 1.0,
            confidence: 1.0,
            variables,
        }
    }

    pub fn evaluate(&self, assignment: &HashMap<String, f32>) -> f32 {
        self.evaluate_structure(&self.structure, assignment)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn evaluate_structure(
        &self,
        structure: &FormulaStructure,
        assignment: &HashMap<String, f32>,
    ) -> f32 {
        match structure {
            FormulaStructure::Atom(predicate) => assignment.get(predicate).copied().unwrap_or(0.0),
            FormulaStructure::Negation(sub) => 1.0 - self.evaluate_structure(sub, assignment),
            FormulaStructure::Conjunction(formulas) => formulas
                .iter()
                .map(|f| self.evaluate_structure(f, assignment))
                .fold(1.0, |acc, val| acc * val),
            FormulaStructure::Disjunction(formulas) => formulas
                .iter()
                .map(|f| self.evaluate_structure(f, assignment))
                .fold(0.0, |acc, val| acc + val - acc * val),
            FormulaStructure::Implication(antecedent, consequent) => {
                let ante = self.evaluate_structure(antecedent, assignment);
                let cons = self.evaluate_structure(consequent, assignment);
                1.0 - ante + ante * cons
            }
            FormulaStructure::Equivalence(left, right) => {
                let left_val = self.evaluate_structure(left, assignment);
                let right_val = self.evaluate_structure(right, assignment);
                1.0 - (left_val - right_val).abs()
            }
            FormulaStructure::Exists(_, sub) => self.evaluate_structure(sub, assignment),
            FormulaStructure::ForAll(_, sub) => self.evaluate_structure(sub, assignment),
        }
    }
}

/// Knowledge rule representation
#[derive(Debug, Clone)]
pub struct KnowledgeRule {
    pub id: String,
    pub antecedent: LogicalFormula,
    pub consequent: LogicalFormula,
    pub confidence: f32,
    pub weight: f32,
}

impl KnowledgeRule {
    pub fn new(id: String, antecedent: LogicalFormula, consequent: LogicalFormula) -> Self {
        Self {
            id,
            antecedent,
            consequent,
            confidence: 1.0,
            weight: 1.0,
        }
    }

    pub fn apply(&self, facts: &HashMap<String, f32>) -> Option<(String, f32)> {
        let antecedent_value = self.antecedent.evaluate(facts);

        if antecedent_value > 0.5 {
            if let FormulaStructure::Atom(predicate) = &self.consequent.structure {
                let consequent_value = antecedent_value * self.confidence;
                return Some((predicate.clone(), consequent_value));
            }
        }

        None
    }
}
