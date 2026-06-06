//! Causal Representation Learning — facade
//!
//! Re-exports all public items from the sibling modules.

pub use crate::causal_representation_learning_eval::{
    compute_disentanglement_score, evaluate_counterfactual_consistency,
    evaluate_intervention_accuracy, full_evaluation, CausalEvalResult,
};
pub use crate::causal_representation_learning_model::CausalRepresentationModel;
pub use crate::causal_representation_learning_types::{
    CausalDiscoveryAlgorithm, CausalDiscoveryConfig, CausalGraph, CausalRepresentationConfig,
    ConstraintSettings, CounterfactualConfig, CounterfactualMethod, CounterfactualQuery,
    DisentanglementConfig, DisentanglementMethod, ExplanationConfig, ExplanationType,
    FactorSupervision, FairnessConstraints, FairnessCriterion, FunctionalForm,
    IdentificationStrategy, IndependenceTest, Intervention, InterventionConfig,
    InterventionDistribution, InterventionType, NoiseModel, ScoreFunction, ScoreSettings,
    SearchStrategy, StructuralCausalModelConfig, StructuralEquation, TwinNetworkConfig,
    VariableType,
};
