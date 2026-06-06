//! Neural-Symbolic Integration — type definitions
//!
//! Configuration structures, reasoning/logic enumerations, and the symbolic
//! representation types (`LogicalFormula`, `FormulaStructure`, `KnowledgeRule`)
//! used by the neural-symbolic integration model.

use crate::ModelConfig;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for neural-symbolic integration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NeuralSymbolicConfig {
    pub base_config: ModelConfig,
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
    /// Use Datalog
    pub use_datalog: bool,
    /// Use Prolog-style resolution
    pub use_prolog: bool,
    /// Answer set programming
    pub use_asp: bool,
    /// Constraint logic programming
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
    /// Forward chaining
    pub forward_chaining: bool,
    /// Backward chaining
    pub backward_chaining: bool,
    /// Rule confidence thresholds
    pub confidence_threshold: f32,
    /// Maximum inference depth
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
    /// OWL reasoning levels
    pub owl_profile: OWLProfile,
    /// Use class hierarchy reasoning
    pub class_hierarchy: bool,
    /// Use property reasoning
    pub property_reasoning: bool,
    /// Use consistency checking
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
    /// OWL 2 EL (Existential Language)
    OWL2EL,
    /// OWL 2 QL (Query Language)
    OWL2QL,
    /// OWL 2 RL (Rule Language)
    OWL2RL,
    /// Full OWL 2
    OWL2Full,
}

/// Logic integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicIntegrationConfig {
    /// Integration methods
    pub integration_methods: Vec<IntegrationMethod>,
    /// Fuzzy logic settings
    pub fuzzy_logic: FuzzyLogicConfig,
    /// Probabilistic logic settings
    pub probabilistic_logic: ProbabilisticLogicConfig,
    /// Temporal logic settings
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
    /// Logic Tensor Networks
    LogicTensors,
    /// Neural Module Networks
    NeuralModuleNetworks,
    /// Differentiable reasoning
    DifferentiableReasoning,
    /// Semantic loss functions
    SemanticLoss,
    /// Logic-guided attention
    LogicAttention,
    /// Symbolic grounding
    SymbolicGrounding,
}

/// Fuzzy logic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyLogicConfig {
    /// T-norm for conjunction
    pub t_norm: TNorm,
    /// T-conorm for disjunction
    pub t_conorm: TConorm,
    /// Implication operator
    pub implication: ImplicationOperator,
    /// Negation operator
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
    /// Use Markov Logic Networks
    pub use_mln: bool,
    /// Use Probabilistic Soft Logic
    pub use_psl: bool,
    /// Use ProbLog
    pub use_problog: bool,
    /// Inference method
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
    /// Linear Temporal Logic
    pub use_ltl: bool,
    /// Computation Tree Logic
    pub use_ctl: bool,
    /// Metric Temporal Logic
    pub use_mtl: bool,
    /// Time window size
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
    /// Knowledge sources
    pub knowledge_sources: Vec<KnowledgeSource>,
    /// Knowledge grounding methods
    pub grounding_methods: Vec<GroundingMethod>,
    /// External knowledge bases
    pub external_kbs: Vec<String>,
    /// Knowledge validation
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
    /// Consistency checking
    pub consistency_check: bool,
    /// Completeness checking
    pub completeness_check: bool,
    /// Confidence thresholds
    pub confidence_threshold: f32,
    /// Validation frequency
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
    /// Architecture type
    pub architecture_type: NeuroSymbolicArchitecture,
    /// Neural component configuration
    pub neural_config: NeuralComponentConfig,
    /// Symbolic component configuration
    pub symbolic_config: SymbolicComponentConfig,
    /// Integration layer configuration
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
    /// Neural and symbolic components in pipeline
    HybridPipeline,
    /// Tightly integrated components
    DeepIntegration,
    /// Loosely coupled components
    LooseCoupling,
    /// Neural-symbolic co-processing
    CoProcessing,
    /// End-to-end differentiable
    EndToEndDifferentiable,
}

/// Neural component configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralComponentConfig {
    /// Neural network layers
    pub layers: Vec<LayerConfig>,
    /// Activation functions
    pub activations: Vec<ActivationFunction>,
    /// Dropout rates
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
    /// Symbol vocabulary size
    pub vocab_size: usize,
    /// Maximum formula length
    pub max_formula_length: usize,
    /// Logic operators
    pub operators: Vec<LogicOperator>,
    /// Reasoning depth
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
    /// Integration method
    pub method: LayerIntegrationMethod,
    /// Attention mechanisms
    pub attention_config: AttentionConfig,
    /// Fusion strategies
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
    /// Constraint types
    pub constraint_types: Vec<ConstraintType>,
    /// Solver configuration
    pub solver_config: SolverConfig,
    /// Soft constraint handling
    pub soft_constraints: bool,
    /// Constraint weights
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
    /// Solver type
    pub solver_type: SolverType,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f32,
    /// Timeout in seconds
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
    /// Formula structure
    pub structure: FormulaStructure,
    /// Truth value (for fuzzy logic)
    pub truth_value: f32,
    /// Confidence score
    pub confidence: f32,
    /// Variables involved
    pub variables: HashSet<String>,
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
        let mut variables = HashSet::new();
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
            FormulaStructure::Conjunction(formulas) => {
                formulas
                    .iter()
                    .map(|f| self.evaluate_structure(f, assignment))
                    .fold(1.0, |acc, val| acc * val) // Product T-norm
            }
            FormulaStructure::Disjunction(formulas) => {
                formulas
                    .iter()
                    .map(|f| self.evaluate_structure(f, assignment))
                    .fold(0.0, |acc, val| acc + val - acc * val) // Probabilistic sum
            }
            FormulaStructure::Implication(antecedent, consequent) => {
                let ante = self.evaluate_structure(antecedent, assignment);
                let cons = self.evaluate_structure(consequent, assignment);
                1.0 - ante + ante * cons // Lukasiewicz implication
            }
            FormulaStructure::Equivalence(left, right) => {
                let left_val = self.evaluate_structure(left, assignment);
                let right_val = self.evaluate_structure(right, assignment);
                1.0 - (left_val - right_val).abs()
            }
            FormulaStructure::Exists(_, sub) => {
                // Simplified existential quantification
                self.evaluate_structure(sub, assignment)
            }
            FormulaStructure::ForAll(_, sub) => {
                // Simplified universal quantification
                self.evaluate_structure(sub, assignment)
            }
        }
    }
}

/// Knowledge rule representation
#[derive(Debug, Clone)]
pub struct KnowledgeRule {
    /// Rule identifier
    pub id: String,
    /// Antecedent (if part)
    pub antecedent: LogicalFormula,
    /// Consequent (then part)
    pub consequent: LogicalFormula,
    /// Rule confidence
    pub confidence: f32,
    /// Rule weight
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
            // Threshold for rule activation
            // Find the main predicate in consequent
            if let FormulaStructure::Atom(predicate) = &self.consequent.structure {
                let consequent_value = antecedent_value * self.confidence;
                return Some((predicate.clone(), consequent_value));
            }
        }

        None
    }
}
