//! Consciousness-Inspired Intuitive Query Planner
//!
//! This module implements artificial intuition for query optimization using
//! pattern memory, gut feeling calculations, and creative optimization techniques.

use crate::query::algebra::{AlgebraTriplePattern, TermPattern as AlgebraTermPattern};
use crate::query::pattern_optimizer::IndexStats;
use crate::OxirsError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Pattern memory for storing learned query patterns and their effectiveness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMemory {
    /// Historical pattern performance data
    pub pattern_history: HashMap<String, PatternPerformance>,
    /// Intuitive weights for different pattern characteristics
    pub intuitive_weights: HashMap<PatternCharacteristic, f64>,
    /// Successful pattern combinations
    pub successful_combinations: Vec<PatternCombination>,
}

/// Performance metrics for a specific pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternPerformance {
    /// Average execution time in microseconds
    pub avg_execution_time: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Frequency of use
    pub usage_count: usize,
    /// Last used timestamp
    pub last_used: std::time::SystemTime,
    /// Intuitive effectiveness score
    pub intuitive_score: f64,
}

/// Characteristics that influence intuitive decision making
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternCharacteristic {
    /// High selectivity patterns (few results)
    HighSelectivity,
    /// Join-heavy patterns
    JoinIntensive,
    /// Variable-heavy patterns
    VariableRich,
    /// Literal-bound patterns
    LiteralBound,
    /// Temporal patterns (time-based queries)
    Temporal,
    /// Hierarchical patterns (class/subclass)
    Hierarchical,
    /// Spatial patterns (geographic)
    Spatial,
    /// Numerical patterns (mathematical)
    Numerical,
}

/// Successful pattern combination for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternCombination {
    /// Patterns in the combination
    pub patterns: Vec<String>,
    /// Total effectiveness score
    pub effectiveness: f64,
    /// Context where this combination was successful
    pub context: QueryContext,
}

/// Context information for query optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    /// Dataset size category
    pub dataset_size: DatasetSize,
    /// Query complexity level
    pub complexity: ComplexityLevel,
    /// Performance requirements
    pub performance_req: PerformanceRequirement,
    /// Domain type (e.g., scientific, business, social)
    pub domain: String,
}

/// Dataset size categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetSize {
    Small,     // < 1M triples
    Medium,    // 1M - 100M triples
    Large,     // 100M - 1B triples
    VeryLarge, // > 1B triples
}

/// Query complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Simple,      // Single pattern queries
    Moderate,    // 2-5 patterns with basic joins
    Complex,     // 6-20 patterns with multiple joins
    VeryComplex, // 20+ patterns with nested operations
}

/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceRequirement {
    Interactive, // < 100ms response time
    Fast,        // < 1s response time
    Balanced,    // Balanced performance/quality tradeoff
    Batch,       // < 1min response time
    Background,  // No strict time limit
}

/// Neural network simulation for intuitive decision making
pub struct IntuitionNetwork {
    /// Input weights for pattern characteristics
    input_weights: HashMap<PatternCharacteristic, f64>,
    /// Hidden layer connections
    hidden_connections: Vec<Vec<f64>>,
    /// Output layer for decision scoring
    output_weights: Vec<f64>,
    /// Learning rate for weight updates
    learning_rate: f64,
}

impl IntuitionNetwork {
    /// Create a new intuition network with random weights
    pub fn new() -> Self {
        let characteristics = vec![
            PatternCharacteristic::HighSelectivity,
            PatternCharacteristic::JoinIntensive,
            PatternCharacteristic::VariableRich,
            PatternCharacteristic::LiteralBound,
            PatternCharacteristic::Temporal,
            PatternCharacteristic::Hierarchical,
            PatternCharacteristic::Spatial,
            PatternCharacteristic::Numerical,
        ];

        let mut input_weights = HashMap::new();
        for characteristic in characteristics {
            input_weights.insert(characteristic, rand::random::<f64>() * 2.0 - 1.0);
        }

        // Create hidden layer (8 inputs -> 5 hidden -> 1 output)
        let hidden_connections = vec![
            vec![rand::random::<f64>() * 2.0 - 1.0; 8], // Hidden neuron 1
            vec![rand::random::<f64>() * 2.0 - 1.0; 8], // Hidden neuron 2
            vec![rand::random::<f64>() * 2.0 - 1.0; 8], // Hidden neuron 3
            vec![rand::random::<f64>() * 2.0 - 1.0; 8], // Hidden neuron 4
            vec![rand::random::<f64>() * 2.0 - 1.0; 8], // Hidden neuron 5
        ];

        let output_weights = vec![rand::random::<f64>() * 2.0 - 1.0; 5];

        Self {
            input_weights,
            hidden_connections,
            output_weights,
            learning_rate: 0.01,
        }
    }

    /// Calculate intuitive score for a pattern
    pub fn calculate_intuitive_score(&self, characteristics: &[PatternCharacteristic]) -> f64 {
        // Convert characteristics to input vector
        let mut input_vector = [0.0; 8];
        let characteristic_types = [
            PatternCharacteristic::HighSelectivity,
            PatternCharacteristic::JoinIntensive,
            PatternCharacteristic::VariableRich,
            PatternCharacteristic::LiteralBound,
            PatternCharacteristic::Temporal,
            PatternCharacteristic::Hierarchical,
            PatternCharacteristic::Spatial,
            PatternCharacteristic::Numerical,
        ];

        for (i, char_type) in characteristic_types.iter().enumerate() {
            if characteristics.contains(char_type) {
                input_vector[i] = 1.0;
            }
        }

        // Forward pass through network
        let mut hidden_activations = Vec::new();
        for connections in &self.hidden_connections {
            let mut activation = 0.0;
            for (i, &weight) in connections.iter().enumerate() {
                activation += input_vector[i] * weight;
            }
            hidden_activations.push(self.sigmoid(activation));
        }

        // Output layer
        let mut output = 0.0;
        for (i, &weight) in self.output_weights.iter().enumerate() {
            output += hidden_activations[i] * weight;
        }

        self.sigmoid(output)
    }

    /// Sigmoid activation function
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Update weights based on feedback
    pub fn learn_from_feedback(
        &mut self,
        characteristics: &[PatternCharacteristic],
        expected_score: f64,
        actual_performance: f64,
    ) {
        let predicted_score = self.calculate_intuitive_score(characteristics);
        let error = actual_performance - predicted_score;

        // Simple weight update (would be more sophisticated in real implementation)
        for char_type in characteristics {
            if let Some(weight) = self.input_weights.get_mut(char_type) {
                *weight += self.learning_rate * error;
            }
        }
    }
}

/// Gut feeling engine for making intuitive decisions
pub struct GutFeelingEngine {
    /// Historical success patterns
    success_patterns: HashMap<String, f64>,
    /// Confidence thresholds for different decisions
    confidence_thresholds: HashMap<String, f64>,
    /// Emotional weighting factors
    emotional_weights: HashMap<String, f64>,
}

impl GutFeelingEngine {
    /// Create a new gut feeling engine
    pub fn new() -> Self {
        let mut confidence_thresholds = HashMap::new();
        confidence_thresholds.insert("high_confidence".to_string(), 0.8);
        confidence_thresholds.insert("medium_confidence".to_string(), 0.6);
        confidence_thresholds.insert("low_confidence".to_string(), 0.4);

        let mut emotional_weights = HashMap::new();
        emotional_weights.insert("excitement".to_string(), 1.2); // Boost for novel patterns
        emotional_weights.insert("caution".to_string(), 0.8); // Reduce for risky patterns
        emotional_weights.insert("confidence".to_string(), 1.1); // Boost for familiar patterns

        Self {
            success_patterns: HashMap::new(),
            confidence_thresholds,
            emotional_weights,
        }
    }

    /// Calculate gut feeling score for a decision
    pub fn calculate_gut_feeling(&self, pattern_signature: &str, context: &QueryContext) -> f64 {
        let base_confidence = self
            .success_patterns
            .get(pattern_signature)
            .copied()
            .unwrap_or(0.5);

        // Apply contextual adjustments
        let context_multiplier = match (&context.complexity, &context.performance_req) {
            (ComplexityLevel::Simple, PerformanceRequirement::Interactive) => 1.2,
            (ComplexityLevel::VeryComplex, PerformanceRequirement::Interactive) => 0.7,
            (ComplexityLevel::Complex, PerformanceRequirement::Background) => 1.1,
            _ => 1.0,
        };

        // Apply emotional weighting
        let emotional_factor = if base_confidence > 0.8 {
            self.emotional_weights
                .get("confidence")
                .copied()
                .unwrap_or(1.0)
        } else if pattern_signature.contains("novel") {
            self.emotional_weights
                .get("excitement")
                .copied()
                .unwrap_or(1.0)
        } else {
            self.emotional_weights
                .get("caution")
                .copied()
                .unwrap_or(1.0)
        };

        (base_confidence * context_multiplier * emotional_factor).min(1.0)
    }

    /// Update gut feeling based on actual results
    pub fn update_gut_feeling(&mut self, pattern_signature: String, success_rate: f64) {
        let current = self
            .success_patterns
            .get(&pattern_signature)
            .copied()
            .unwrap_or(0.5);
        // Exponential moving average
        let updated = current * 0.8 + success_rate * 0.2;
        self.success_patterns.insert(pattern_signature, updated);
    }
}

/// Creativity engine for generating novel optimization strategies
pub struct CreativityEngine {
    /// Repository of creative optimization techniques
    techniques: Vec<CreativeTechnique>,
    /// Randomness factor for exploration
    exploration_factor: f64,
    /// Combination history for avoiding repetition
    combination_history: HashSet<String>,
}

/// Creative optimization techniques
#[derive(Debug, Clone)]
pub enum CreativeTechnique {
    /// Reverse the usual optimization order
    ReverseOptimization,
    /// Try parallel execution paths
    ParallelPaths,
    /// Use predictive prefetching
    PredictivePrefetch,
    /// Apply genetic algorithm mutations
    GeneticMutation,
    /// Use chaos theory for exploration
    ChaoticExploration,
    /// Apply artistic principles (golden ratio, symmetry)
    ArtisticPrinciples,
    /// Use biomimetic approaches
    BiomimeticOptimization,
}

impl CreativityEngine {
    /// Create a new creativity engine
    pub fn new() -> Self {
        let techniques = vec![
            CreativeTechnique::ReverseOptimization,
            CreativeTechnique::ParallelPaths,
            CreativeTechnique::PredictivePrefetch,
            CreativeTechnique::GeneticMutation,
            CreativeTechnique::ChaoticExploration,
            CreativeTechnique::ArtisticPrinciples,
            CreativeTechnique::BiomimeticOptimization,
        ];

        Self {
            techniques,
            exploration_factor: 0.1,
            combination_history: HashSet::new(),
        }
    }

    /// Generate creative optimization suggestions
    pub fn generate_creative_optimizations(
        &mut self,
        patterns: &[AlgebraTriplePattern],
    ) -> Vec<CreativeOptimization> {
        let mut optimizations = Vec::new();

        // Apply each technique with some probability
        for technique in &self.techniques {
            if rand::random::<f64>() < self.exploration_factor {
                match technique {
                    CreativeTechnique::ReverseOptimization => {
                        optimizations.push(CreativeOptimization {
                            technique: technique.clone(),
                            description: "Try executing patterns in reverse selectivity order"
                                .to_string(),
                            confidence: 0.7,
                            novelty: 0.8,
                        });
                    }
                    CreativeTechnique::ParallelPaths => {
                        optimizations.push(CreativeOptimization {
                            technique: technique.clone(),
                            description: "Execute independent patterns in parallel".to_string(),
                            confidence: 0.9,
                            novelty: 0.6,
                        });
                    }
                    CreativeTechnique::ArtisticPrinciples => {
                        optimizations.push(CreativeOptimization {
                            technique: technique.clone(),
                            description: "Apply golden ratio to join ordering".to_string(),
                            confidence: 0.5,
                            novelty: 0.95,
                        });
                    }
                    _ => {
                        // Other techniques would be implemented similarly
                    }
                }
            }
        }

        optimizations
    }
}

/// A creative optimization suggestion
#[derive(Debug, Clone)]
pub struct CreativeOptimization {
    /// The technique used
    pub technique: CreativeTechnique,
    /// Human-readable description
    pub description: String,
    /// Confidence in this optimization (0.0 to 1.0)
    pub confidence: f64,
    /// Novelty factor (0.0 to 1.0)
    pub novelty: f64,
}

/// Intuitive query planner that combines all consciousness-inspired components
pub struct IntuitiveQueryPlanner {
    /// Pattern memory for learning
    pattern_memory: Arc<RwLock<PatternMemory>>,
    /// Neural network for intuitive scoring
    intuition_network: Arc<RwLock<IntuitionNetwork>>,
    /// Gut feeling engine
    gut_feeling: Arc<RwLock<GutFeelingEngine>>,
    /// Creativity engine
    creativity: Arc<RwLock<CreativityEngine>>,
    /// Traditional optimizer for baseline comparison
    traditional_stats: Arc<IndexStats>,
}

impl IntuitiveQueryPlanner {
    /// Create a new intuitive query planner
    pub fn new(traditional_stats: Arc<IndexStats>) -> Self {
        Self {
            pattern_memory: Arc::new(RwLock::new(PatternMemory {
                pattern_history: HashMap::new(),
                intuitive_weights: HashMap::new(),
                successful_combinations: Vec::new(),
            })),
            intuition_network: Arc::new(RwLock::new(IntuitionNetwork::new())),
            gut_feeling: Arc::new(RwLock::new(GutFeelingEngine::new())),
            creativity: Arc::new(RwLock::new(CreativityEngine::new())),
            traditional_stats,
        }
    }

    /// Plan query execution using intuitive optimization
    pub fn plan_intuitive_execution(
        &self,
        patterns: &[AlgebraTriplePattern],
        context: &QueryContext,
    ) -> Result<IntuitiveExecutionPlan, OxirsError> {
        // Extract pattern characteristics
        let characteristics = self.extract_pattern_characteristics(patterns);

        // Calculate intuitive scores
        let intuitive_scores = if let Ok(network) = self.intuition_network.read() {
            patterns
                .iter()
                .enumerate()
                .map(|(i, _)| network.calculate_intuitive_score(&characteristics[i]))
                .collect::<Vec<_>>()
        } else {
            vec![0.5; patterns.len()]
        };

        // Get gut feeling assessment
        let gut_feelings = if let Ok(gut) = self.gut_feeling.read() {
            patterns
                .iter()
                .enumerate()
                .map(|(i, pattern)| {
                    let signature = format!("{pattern:?}");
                    gut.calculate_gut_feeling(&signature, context)
                })
                .collect::<Vec<_>>()
        } else {
            vec![0.5; patterns.len()]
        };

        // Generate creative optimizations
        let creative_opts = if let Ok(mut creativity) = self.creativity.write() {
            creativity.generate_creative_optimizations(patterns)
        } else {
            Vec::new()
        };

        // Combine all factors to create plan
        let mut pattern_rankings = Vec::new();
        for (i, pattern) in patterns.iter().enumerate() {
            let combined_score = (intuitive_scores[i] + gut_feelings[i]) * 0.5;
            pattern_rankings.push((i, pattern.clone(), combined_score));
        }

        // Sort by combined intuitive score
        pattern_rankings.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        Ok(IntuitiveExecutionPlan {
            ordered_patterns: pattern_rankings,
            creative_optimizations: creative_opts,
            confidence_level: intuitive_scores.iter().sum::<f64>() / intuitive_scores.len() as f64,
            gut_feeling_average: gut_feelings.iter().sum::<f64>() / gut_feelings.len() as f64,
            context: context.clone(),
        })
    }

    /// Extract pattern characteristics for intuitive analysis
    fn extract_pattern_characteristics(
        &self,
        patterns: &[AlgebraTriplePattern],
    ) -> Vec<Vec<PatternCharacteristic>> {
        patterns
            .iter()
            .map(|pattern| {
                let mut characteristics = Vec::new();

                // Check for high selectivity (bound terms)
                let bound_terms = [&pattern.subject, &pattern.predicate, &pattern.object]
                    .iter()
                    .filter(|term| !matches!(term, AlgebraTermPattern::Variable(_)))
                    .count();

                if bound_terms >= 2 {
                    characteristics.push(PatternCharacteristic::HighSelectivity);
                }

                // Check for literal bounds
                if matches!(pattern.object, AlgebraTermPattern::Literal(_)) {
                    characteristics.push(PatternCharacteristic::LiteralBound);
                }

                // Count variables
                let variable_count = [&pattern.subject, &pattern.predicate, &pattern.object]
                    .iter()
                    .filter(|term| matches!(term, AlgebraTermPattern::Variable(_)))
                    .count();

                if variable_count >= 2 {
                    characteristics.push(PatternCharacteristic::VariableRich);
                }

                // Check for temporal patterns (basic heuristic)
                if let AlgebraTermPattern::NamedNode(pred) = &pattern.predicate {
                    if pred.as_str().contains("time") || pred.as_str().contains("date") {
                        characteristics.push(PatternCharacteristic::Temporal);
                    }
                }

                // Check for hierarchical patterns
                if let AlgebraTermPattern::NamedNode(pred) = &pattern.predicate {
                    if pred.as_str().contains("type") || pred.as_str().contains("subClass") {
                        characteristics.push(PatternCharacteristic::Hierarchical);
                    }
                }

                characteristics
            })
            .collect()
    }

    /// Learn from execution results to improve future planning
    pub fn learn_from_execution(
        &self,
        plan: &IntuitiveExecutionPlan,
        actual_performance: &ExecutionResults,
    ) {
        // Update pattern memory
        if let Ok(mut memory) = self.pattern_memory.write() {
            for (i, (_, pattern, predicted_score)) in plan.ordered_patterns.iter().enumerate() {
                let signature = format!("{pattern:?}");
                let actual_score = actual_performance
                    .pattern_scores
                    .get(&i)
                    .copied()
                    .unwrap_or(0.5);

                let performance = PatternPerformance {
                    avg_execution_time: actual_performance
                        .execution_times
                        .get(&i)
                        .copied()
                        .unwrap_or(1000.0),
                    success_rate: actual_score,
                    usage_count: memory
                        .pattern_history
                        .get(&signature)
                        .map(|p| p.usage_count + 1)
                        .unwrap_or(1),
                    last_used: std::time::SystemTime::now(),
                    intuitive_score: *predicted_score,
                };

                memory.pattern_history.insert(signature, performance);
            }
        }

        // Update intuition network
        if let Ok(mut network) = self.intuition_network.write() {
            for (i, (_, pattern, predicted_score)) in plan.ordered_patterns.iter().enumerate() {
                let characteristics =
                    self.extract_pattern_characteristics(&[pattern.clone()])[0].clone();
                let actual_score = actual_performance
                    .pattern_scores
                    .get(&i)
                    .copied()
                    .unwrap_or(0.5);

                network.learn_from_feedback(&characteristics, *predicted_score, actual_score);
            }
        }

        // Update gut feeling
        if let Ok(mut gut) = self.gut_feeling.write() {
            for (i, (_, pattern, _)) in plan.ordered_patterns.iter().enumerate() {
                let signature = format!("{pattern:?}");
                let success_rate = actual_performance
                    .pattern_scores
                    .get(&i)
                    .copied()
                    .unwrap_or(0.5);

                gut.update_gut_feeling(signature, success_rate);
            }
        }
    }
}

/// Execution plan generated by intuitive planner
#[derive(Debug, Clone)]
pub struct IntuitiveExecutionPlan {
    /// Patterns ordered by intuitive ranking
    pub ordered_patterns: Vec<(usize, AlgebraTriplePattern, f64)>,
    /// Creative optimization suggestions
    pub creative_optimizations: Vec<CreativeOptimization>,
    /// Overall confidence in the plan
    pub confidence_level: f64,
    /// Average gut feeling score
    pub gut_feeling_average: f64,
    /// Context used for planning
    pub context: QueryContext,
}

/// Results from executing an intuitive plan
#[derive(Debug)]
pub struct ExecutionResults {
    /// Performance scores for each pattern (0.0 to 1.0)
    pub pattern_scores: HashMap<usize, f64>,
    /// Execution times in microseconds
    pub execution_times: HashMap<usize, f64>,
    /// Overall success rate
    pub overall_success: f64,
    /// Creative optimizations that were applied
    pub applied_optimizations: Vec<CreativeTechnique>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Variable, NamedNode, Literal};

    #[test]
    fn test_intuition_network_creation() {
        let network = IntuitionNetwork::new();
        assert_eq!(network.input_weights.len(), 8);
        assert_eq!(network.hidden_connections.len(), 5);
        assert_eq!(network.output_weights.len(), 5);
    }

    #[test]
    fn test_gut_feeling_calculation() {
        let engine = GutFeelingEngine::new();
        let context = QueryContext {
            dataset_size: DatasetSize::Medium,
            complexity: ComplexityLevel::Simple,
            performance_req: PerformanceRequirement::Interactive,
            domain: "test".to_string(),
        };

        let score = engine.calculate_gut_feeling("test_pattern", &context);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_creativity_engine() {
        let mut engine = CreativityEngine::new();
        let patterns = vec![AlgebraTriplePattern::new(
            AlgebraTermPattern::Variable(Variable::new("s").unwrap()),
            AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/pred").unwrap()),
            AlgebraTermPattern::Variable(Variable::new("o").unwrap()),
        )];

        let optimizations = engine.generate_creative_optimizations(&patterns);
        // Should generate some optimizations based on exploration factor
        assert!(optimizations.len() <= engine.techniques.len());
    }

    #[test]
    fn test_pattern_characteristics_extraction() {
        let stats = Arc::new(IndexStats::new());
        let planner = IntuitiveQueryPlanner::new(stats);

        let patterns = vec![AlgebraTriplePattern::new(
            AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/subj").unwrap()),
            AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/type").unwrap()),
            AlgebraTermPattern::Literal(Literal::new("test")),
        )];

        let characteristics = planner.extract_pattern_characteristics(&patterns);
        assert_eq!(characteristics.len(), 1);
        assert!(characteristics[0].contains(&PatternCharacteristic::HighSelectivity));
        assert!(characteristics[0].contains(&PatternCharacteristic::LiteralBound));
    }
}
