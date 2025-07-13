//! Vector-Aware Query Optimizer
//!
//! This module provides integration between SPARQL query optimization and vector search
//! capabilities from oxirs-vec. It enables intelligent query planning that can leverage
//! vector indexes for semantic similarity queries and hybrid text/vector search.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use tracing::{info, span, Level};

use crate::algebra::{Algebra, Expression, Term, TriplePattern, Variable};
use crate::integrated_query_planner::{
    IntegratedExecutionPlan, IntegratedPlannerConfig, IntegratedQueryPlanner,
};

/// Vector search integration configuration
#[derive(Debug, Clone)]
pub struct VectorOptimizerConfig {
    /// Enable vector similarity search optimization
    pub enable_vector_optimization: bool,
    /// Threshold for semantic similarity search (0.0 to 1.0)
    pub similarity_threshold: f32,
    /// Maximum number of vector candidates to consider
    pub max_vector_candidates: usize,
    /// Vector index cache size
    pub vector_cache_size: usize,
    /// Enable hybrid text-vector search
    pub enable_hybrid_search: bool,
    /// Vector embedding dimension
    pub embedding_dimension: usize,
    /// Distance metric for vector search
    pub distance_metric: VectorDistanceMetric,
    /// Vector index types to consider
    pub preferred_index_types: Vec<VectorIndexType>,
    /// Minimum query complexity to enable vector optimization
    pub complexity_threshold: f64,
}

impl Default for VectorOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_vector_optimization: true,
            similarity_threshold: 0.8,
            max_vector_candidates: 1000,
            vector_cache_size: 10_000,
            enable_hybrid_search: true,
            embedding_dimension: 768, // Common for transformer models
            distance_metric: VectorDistanceMetric::Cosine,
            preferred_index_types: vec![
                VectorIndexType::Hnsw,
                VectorIndexType::IvfFlat,
                VectorIndexType::IvfPq,
            ],
            complexity_threshold: 10.0,
        }
    }
}

/// Supported vector distance metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorDistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

/// Types of vector indexes available
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorIndexType {
    Hnsw,
    IvfFlat,
    IvfPq,
    FlatIndex,
    Lsh,
}

/// Vector search strategy for SPARQL queries
#[derive(Debug, Clone)]
pub enum VectorSearchStrategy {
    /// Pure vector similarity search
    PureVector {
        query_vector: Vec<f32>,
        similarity_threshold: f32,
        k: usize,
    },
    /// Hybrid search combining text and vector
    Hybrid {
        text_query: String,
        query_vector: Option<Vec<f32>>,
        text_weight: f32,
        vector_weight: f32,
    },
    /// Vector-constrained SPARQL query
    VectorConstrained {
        sparql_patterns: Vec<TriplePattern>,
        vector_filter: VectorFilter,
    },
    /// Semantic expansion using vectors
    SemanticExpansion {
        original_terms: Vec<Term>,
        expansion_candidates: Vec<(Term, f32)>,
        max_expansions: usize,
    },
}

/// Vector-based filters for SPARQL queries
#[derive(Debug, Clone)]
pub struct VectorFilter {
    pub subject_vector: Option<Vec<f32>>,
    pub object_vector: Option<Vec<f32>>,
    pub predicate_vector: Option<Vec<f32>>,
    pub similarity_threshold: f32,
    pub max_matches: usize,
}

/// Vector-aware query optimizer
pub struct VectorQueryOptimizer {
    config: VectorOptimizerConfig,
    integrated_planner: IntegratedQueryPlanner,
    vector_indexes: Arc<Mutex<HashMap<String, VectorIndexInfo>>>,
    #[allow(dead_code)]
    embedding_cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    #[allow(dead_code)]
    #[allow(clippy::type_complexity)]
    semantic_cache: Arc<Mutex<HashMap<String, Vec<(String, f32)>>>>,
    #[allow(dead_code)]
    query_patterns: Arc<Mutex<HashMap<u64, VectorSearchStrategy>>>,
    performance_metrics: Arc<Mutex<VectorPerformanceMetrics>>,
}

/// Information about available vector indexes
#[derive(Debug, Clone)]
pub struct VectorIndexInfo {
    pub index_type: VectorIndexType,
    pub dimension: usize,
    pub size: usize,
    pub distance_metric: VectorDistanceMetric,
    pub build_time: Duration,
    pub last_updated: Instant,
    pub accuracy_stats: IndexAccuracyStats,
    pub performance_stats: IndexPerformanceStats,
}

/// Accuracy statistics for vector indexes
#[derive(Debug, Clone, Default)]
pub struct IndexAccuracyStats {
    pub recall_at_k: HashMap<usize, f32>,
    pub precision_at_k: HashMap<usize, f32>,
    pub average_distance_error: f32,
    pub query_count: usize,
}

/// Performance statistics for vector indexes
#[derive(Debug, Clone, Default)]
pub struct IndexPerformanceStats {
    pub average_query_time: Duration,
    pub queries_per_second: f32,
    pub memory_usage: usize,
    pub cache_hit_rate: f32,
    pub index_efficiency: f32,
}

/// Vector optimization performance metrics
#[derive(Debug, Clone, Default)]
pub struct VectorPerformanceMetrics {
    pub vector_queries_optimized: usize,
    pub hybrid_queries_optimized: usize,
    pub semantic_expansions_performed: usize,
    pub average_optimization_speedup: f32,
    pub vector_cache_hit_rate: f32,
    pub embedding_generation_time: Duration,
    pub total_optimization_time: Duration,
}

/// Vector-enhanced execution plan
#[derive(Debug, Clone)]
pub struct VectorEnhancedPlan {
    /// Base integrated execution plan
    pub base_plan: IntegratedExecutionPlan,
    /// Vector search strategy to apply
    pub vector_strategy: Option<VectorSearchStrategy>,
    /// Recommended vector index to use
    pub recommended_vector_index: Option<String>,
    /// Expected vector search performance
    pub vector_performance_estimate: VectorPerformanceEstimate,
    /// Hybrid search configuration
    pub hybrid_config: Option<HybridSearchConfig>,
}

/// Performance estimate for vector operations
#[derive(Debug, Clone, Default)]
pub struct VectorPerformanceEstimate {
    pub estimated_query_time: Duration,
    pub estimated_recall: f32,
    pub estimated_memory_usage: usize,
    pub confidence: f32,
}

/// Configuration for hybrid search
#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    pub text_weight: f32,
    pub vector_weight: f32,
    pub reranking_k: usize,
    pub fusion_method: ResultFusionMethod,
}

/// Methods for fusing text and vector search results
#[derive(Debug, Clone, Copy)]
pub enum ResultFusionMethod {
    LinearCombination,
    RankFusion,
    BayesianFusion,
    LearningToRank,
}

impl VectorQueryOptimizer {
    /// Create a new vector-aware query optimizer
    pub fn new(
        vector_config: VectorOptimizerConfig,
        planner_config: IntegratedPlannerConfig,
    ) -> Result<Self> {
        let integrated_planner = IntegratedQueryPlanner::new(planner_config)?;

        Ok(Self {
            config: vector_config,
            integrated_planner,
            vector_indexes: Arc::new(Mutex::new(HashMap::new())),
            embedding_cache: Arc::new(Mutex::new(HashMap::new())),
            semantic_cache: Arc::new(Mutex::new(HashMap::new())),
            query_patterns: Arc::new(Mutex::new(HashMap::new())),
            performance_metrics: Arc::new(Mutex::new(VectorPerformanceMetrics::default())),
        })
    }

    /// Register a vector index for use in query optimization
    pub fn register_vector_index(&self, name: String, index_info: VectorIndexInfo) -> Result<()> {
        let mut indexes = self.vector_indexes.lock().unwrap();
        let size = index_info.size;
        indexes.insert(name.clone(), index_info);

        info!("Registered vector index: {} with {} vectors", name, size);
        Ok(())
    }

    /// Create an optimized execution plan with vector awareness
    pub fn create_vector_enhanced_plan(&mut self, algebra: &Algebra) -> Result<VectorEnhancedPlan> {
        let span = span!(Level::DEBUG, "vector_enhanced_planning");
        let _enter = span.enter();

        // First get the base plan from integrated planner
        let base_plan = self.integrated_planner.create_plan(algebra)?;

        // Analyze the query for vector optimization opportunities
        let vector_opportunities = self.analyze_vector_opportunities(algebra)?;

        if vector_opportunities.is_empty() {
            // No vector optimization opportunities
            return Ok(VectorEnhancedPlan {
                base_plan,
                vector_strategy: None,
                recommended_vector_index: None,
                vector_performance_estimate: VectorPerformanceEstimate::default(),
                hybrid_config: None,
            });
        }

        // Select the best vector optimization strategy
        let vector_strategy = self.select_vector_strategy(&vector_opportunities, algebra)?;

        // Choose the optimal vector index
        let recommended_vector_index = self.select_vector_index(&vector_strategy)?;

        // Estimate vector performance
        let vector_performance_estimate =
            self.estimate_vector_performance(&vector_strategy, &recommended_vector_index)?;

        // Configure hybrid search if applicable
        let hybrid_config = self.configure_hybrid_search(&vector_strategy)?;

        // Update performance metrics
        self.update_optimization_metrics(&vector_strategy);

        Ok(VectorEnhancedPlan {
            base_plan,
            vector_strategy: Some(vector_strategy),
            recommended_vector_index,
            vector_performance_estimate,
            hybrid_config,
        })
    }

    /// Analyze SPARQL algebra for vector optimization opportunities
    fn analyze_vector_opportunities(&self, algebra: &Algebra) -> Result<Vec<VectorOpportunity>> {
        let mut opportunities = Vec::new();

        match algebra {
            Algebra::Bgp(patterns) => {
                opportunities.extend(self.analyze_bgp_patterns(patterns)?);
            }
            Algebra::Filter { pattern, condition } => {
                // Check if filter condition involves semantic similarity
                if self.is_semantic_filter(condition) {
                    opportunities.push(VectorOpportunity::SemanticFilter {
                        condition: condition.clone(),
                        estimated_selectivity: 0.1, // Conservative estimate
                    });
                }
                opportunities.extend(self.analyze_vector_opportunities(pattern)?);
            }
            Algebra::Join { left, right } => {
                opportunities.extend(self.analyze_vector_opportunities(left)?);
                opportunities.extend(self.analyze_vector_opportunities(right)?);

                // Check for join optimization opportunities
                if let Some(join_opportunity) = self.analyze_join_opportunity(left, right)? {
                    opportunities.push(join_opportunity);
                }
            }
            Algebra::Union { left, right } => {
                opportunities.extend(self.analyze_vector_opportunities(left)?);
                opportunities.extend(self.analyze_vector_opportunities(right)?);
            }
            Algebra::LeftJoin {
                left,
                right,
                filter: _,
            } => {
                opportunities.extend(self.analyze_vector_opportunities(left)?);
                opportunities.extend(self.analyze_vector_opportunities(right)?);
            }
            _ => {
                // Recursively analyze sub-patterns
                if let Some(subpattern) = self.extract_subpattern(algebra) {
                    opportunities.extend(self.analyze_vector_opportunities(&subpattern)?);
                }
            }
        }

        Ok(opportunities)
    }

    /// Analyze BGP patterns for vector opportunities
    fn analyze_bgp_patterns(&self, patterns: &[TriplePattern]) -> Result<Vec<VectorOpportunity>> {
        let mut opportunities = Vec::new();

        for pattern in patterns {
            // Check for text matching patterns that could benefit from semantic search
            if self.is_text_matching_pattern(pattern) {
                opportunities.push(VectorOpportunity::TextSimilarity {
                    pattern: pattern.clone(),
                    estimated_matches: 100, // Conservative estimate
                });
            }

            // Check for entity similarity patterns
            if self.is_entity_similarity_pattern(pattern) {
                opportunities.push(VectorOpportunity::EntitySimilarity {
                    pattern: pattern.clone(),
                    similarity_type: EntitySimilarityType::Conceptual,
                });
            }

            // Check for property path patterns that could benefit from vector expansion
            if self.is_expandable_property_pattern(pattern) {
                opportunities.push(VectorOpportunity::PropertyExpansion {
                    pattern: pattern.clone(),
                    expansion_depth: 2,
                });
            }
        }

        Ok(opportunities)
    }

    /// Check if a pattern involves text matching
    fn is_text_matching_pattern(&self, pattern: &TriplePattern) -> bool {
        // Look for patterns with literal objects that might be text
        match &pattern.object {
            Term::Literal(literal) => {
                // Check if literal contains text that could benefit from semantic search
                literal.value.len() > 5 && literal.value.chars().any(|c| c.is_alphabetic())
            }
            _ => false,
        }
    }

    /// Check if a pattern involves entity similarity
    fn is_entity_similarity_pattern(&self, pattern: &TriplePattern) -> bool {
        // Look for patterns that query for related entities
        match &pattern.predicate {
            Term::Iri(iri) => {
                // Common predicates that indicate entity relationships
                iri.as_str().contains("similar")
                    || iri.as_str().contains("related")
                    || iri.as_str().contains("type")
                    || iri.as_str().contains("category")
            }
            _ => false,
        }
    }

    /// Check if a pattern could benefit from property expansion
    fn is_expandable_property_pattern(&self, pattern: &TriplePattern) -> bool {
        // Look for patterns with specific predicates that could be semantically expanded
        match &pattern.predicate {
            Term::Variable(_) => true, // Variable predicates can often be expanded
            Term::Iri(iri) => {
                // Common expandable predicates
                let expandable_predicates = [
                    "type",
                    "category",
                    "topic",
                    "subject",
                    "theme",
                    "describes",
                    "about",
                    "concerns",
                    "deals_with",
                ];

                expandable_predicates
                    .iter()
                    .any(|pred| iri.as_str().contains(pred))
            }
            _ => false,
        }
    }

    /// Check if an expression is a semantic filter
    fn is_semantic_filter(&self, expression: &Expression) -> bool {
        // Look for filter expressions that involve text similarity functions
        match expression {
            Expression::Function { name, .. } => {
                name.as_str().contains("similarity")
                    || name.as_str().contains("match")
                    || name.as_str().contains("distance")
                    || name.as_str().contains("semantic")
            }
            _ => false,
        }
    }

    /// Analyze join opportunities for vector optimization
    fn analyze_join_opportunity(
        &self,
        left: &Algebra,
        right: &Algebra,
    ) -> Result<Option<VectorOpportunity>> {
        // Check if the join involves patterns that could benefit from vector-based join optimization
        let left_vars = self.extract_variables(left);
        let right_vars = self.extract_variables(right);
        let shared_vars: Vec<_> = left_vars.intersection(&right_vars).collect();

        if !shared_vars.is_empty() {
            // Check if any shared variables represent entities that could benefit from vector similarity
            for var in shared_vars {
                if self.is_vector_suitable_variable(var, left)
                    || self.is_vector_suitable_variable(var, right)
                {
                    return Ok(Some(VectorOpportunity::VectorJoin {
                        left_pattern: Box::new(left.clone()),
                        right_pattern: Box::new(right.clone()),
                        join_variable: var.clone(),
                        estimated_selectivity: 0.2,
                    }));
                }
            }
        }

        Ok(None)
    }

    /// Extract variables from algebra expression
    fn extract_variables(&self, algebra: &Algebra) -> HashSet<Variable> {
        let mut vars = HashSet::new();

        match algebra {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    if let Term::Variable(var) = &pattern.subject {
                        vars.insert(var.clone());
                    }
                    if let Term::Variable(var) = &pattern.predicate {
                        vars.insert(var.clone());
                    }
                    if let Term::Variable(var) = &pattern.object {
                        vars.insert(var.clone());
                    }
                }
            }
            _ => {
                // Recursively extract from subpatterns
                // Implementation would continue for other algebra types
            }
        }

        vars
    }

    /// Check if a variable is suitable for vector operations
    fn is_vector_suitable_variable(&self, _var: &Variable, _context: &Algebra) -> bool {
        // Heuristics to determine if a variable represents entities suitable for vector similarity
        // This could be based on type information, predicates used, etc.
        true // Simplified for now
    }

    /// Extract subpattern from algebra for recursive analysis
    fn extract_subpattern(&self, algebra: &Algebra) -> Option<Algebra> {
        match algebra {
            Algebra::Project { pattern, .. } => Some((**pattern).clone()),
            Algebra::Distinct { pattern } => Some((**pattern).clone()),
            Algebra::Reduced { pattern } => Some((**pattern).clone()),
            Algebra::OrderBy { pattern, .. } => Some((**pattern).clone()),
            Algebra::Slice { pattern, .. } => Some((**pattern).clone()),
            Algebra::Group { pattern, .. } => Some((**pattern).clone()),
            Algebra::Having { pattern, .. } => Some((**pattern).clone()),
            _ => None,
        }
    }

    /// Select the best vector search strategy
    fn select_vector_strategy(
        &self,
        opportunities: &[VectorOpportunity],
        _algebra: &Algebra,
    ) -> Result<VectorSearchStrategy> {
        if opportunities.is_empty() {
            return Err(anyhow!("No vector opportunities available"));
        }

        // Simple strategy selection - could be enhanced with ML
        let primary_opportunity = &opportunities[0];

        match primary_opportunity {
            VectorOpportunity::TextSimilarity { pattern, .. } => {
                Ok(VectorSearchStrategy::Hybrid {
                    text_query: self.extract_text_from_pattern(pattern)?,
                    query_vector: None, // Will be generated during execution
                    text_weight: 0.6,
                    vector_weight: 0.4,
                })
            }
            VectorOpportunity::EntitySimilarity { pattern, .. } => {
                Ok(VectorSearchStrategy::SemanticExpansion {
                    original_terms: vec![pattern.subject.clone()],
                    expansion_candidates: Vec::new(), // Will be populated during execution
                    max_expansions: 10,
                })
            }
            VectorOpportunity::VectorJoin { .. } => {
                Ok(VectorSearchStrategy::VectorConstrained {
                    sparql_patterns: vec![], // Will be populated based on join patterns
                    vector_filter: VectorFilter {
                        subject_vector: None,
                        object_vector: None,
                        predicate_vector: None,
                        similarity_threshold: self.config.similarity_threshold,
                        max_matches: self.config.max_vector_candidates,
                    },
                })
            }
            _ => {
                Ok(VectorSearchStrategy::PureVector {
                    query_vector: Vec::new(), // Will be generated during execution
                    similarity_threshold: self.config.similarity_threshold,
                    k: 100,
                })
            }
        }
    }

    /// Extract text content from a triple pattern
    fn extract_text_from_pattern(&self, pattern: &TriplePattern) -> Result<String> {
        match &pattern.object {
            Term::Literal(literal) => Ok(literal.value.clone()),
            Term::Iri(iri) => {
                // Extract local name from IRI
                let iri_str = iri.as_str();
                if let Some(fragment) = iri_str.split('#').next_back() {
                    Ok(fragment.to_string())
                } else if let Some(local) = iri_str.split('/').next_back() {
                    Ok(local.to_string())
                } else {
                    Ok(iri_str.to_string())
                }
            }
            _ => Err(anyhow!("Cannot extract text from pattern")),
        }
    }

    /// Select the optimal vector index for the strategy
    fn select_vector_index(&self, strategy: &VectorSearchStrategy) -> Result<Option<String>> {
        let indexes = self.vector_indexes.lock().unwrap();

        if indexes.is_empty() {
            return Ok(None);
        }

        // Select index based on strategy requirements and performance characteristics
        let mut best_index = None;
        let mut best_score = 0.0f32;

        for (name, info) in indexes.iter() {
            let score = self.calculate_index_score(info, strategy);
            if score > best_score {
                best_score = score;
                best_index = Some(name.clone());
            }
        }

        Ok(best_index)
    }

    /// Calculate suitability score for an index given a strategy
    fn calculate_index_score(
        &self,
        info: &VectorIndexInfo,
        strategy: &VectorSearchStrategy,
    ) -> f32 {
        let mut score = 0.0f32;

        // Base score from index type preferences
        let type_bonus = match info.index_type {
            VectorIndexType::Hnsw => 1.0,
            VectorIndexType::IvfPq => 0.8,
            VectorIndexType::IvfFlat => 0.7,
            VectorIndexType::FlatIndex => 0.5,
            VectorIndexType::Lsh => 0.6,
        };
        score += type_bonus;

        // Performance-based scoring
        score += info.performance_stats.queries_per_second / 1000.0; // Normalize QPS
        score += info.performance_stats.cache_hit_rate;
        score += info.performance_stats.index_efficiency;

        // Accuracy-based scoring
        if let Some(recall_10) = info.accuracy_stats.recall_at_k.get(&10) {
            score += recall_10;
        }

        // Strategy-specific adjustments
        match strategy {
            VectorSearchStrategy::PureVector { k, .. } => {
                // Prefer indexes optimized for k-NN search
                if *k <= 100 && matches!(info.index_type, VectorIndexType::Hnsw) {
                    score += 0.2;
                }
            }
            VectorSearchStrategy::Hybrid { .. } => {
                // Prefer indexes with good recall for hybrid search
                score += 0.1;
            }
            _ => {}
        }

        score
    }

    /// Estimate vector search performance
    fn estimate_vector_performance(
        &self,
        strategy: &VectorSearchStrategy,
        index_name: &Option<String>,
    ) -> Result<VectorPerformanceEstimate> {
        let mut estimate = VectorPerformanceEstimate::default();

        if let Some(name) = index_name {
            let indexes = self.vector_indexes.lock().unwrap();
            if let Some(info) = indexes.get(name) {
                estimate.estimated_query_time = info.performance_stats.average_query_time;
                estimate.estimated_memory_usage = info.performance_stats.memory_usage;

                // Estimate recall based on strategy
                estimate.estimated_recall = match strategy {
                    VectorSearchStrategy::PureVector { .. } => {
                        *info.accuracy_stats.recall_at_k.get(&10).unwrap_or(&0.9)
                    }
                    VectorSearchStrategy::Hybrid { .. } => {
                        // Hybrid search typically has higher effective recall
                        info.accuracy_stats.recall_at_k.get(&10).unwrap_or(&0.9) * 1.1
                    }
                    _ => 0.8, // Conservative estimate
                };

                estimate.confidence = 0.8; // Base confidence
            }
        } else {
            // No vector index available - provide conservative estimates
            estimate.estimated_query_time = Duration::from_millis(100);
            estimate.estimated_recall = 0.7;
            estimate.estimated_memory_usage = 1024 * 1024; // 1MB
            estimate.confidence = 0.5;
        }

        Ok(estimate)
    }

    /// Configure hybrid search parameters
    fn configure_hybrid_search(
        &self,
        strategy: &VectorSearchStrategy,
    ) -> Result<Option<HybridSearchConfig>> {
        match strategy {
            VectorSearchStrategy::Hybrid {
                text_weight,
                vector_weight,
                ..
            } => Ok(Some(HybridSearchConfig {
                text_weight: *text_weight,
                vector_weight: *vector_weight,
                reranking_k: 100,
                fusion_method: ResultFusionMethod::LinearCombination,
            })),
            _ => Ok(None),
        }
    }

    /// Update optimization performance metrics
    fn update_optimization_metrics(&self, strategy: &VectorSearchStrategy) {
        let mut metrics = self.performance_metrics.lock().unwrap();

        match strategy {
            VectorSearchStrategy::PureVector { .. } => {
                metrics.vector_queries_optimized += 1;
            }
            VectorSearchStrategy::Hybrid { .. } => {
                metrics.hybrid_queries_optimized += 1;
            }
            VectorSearchStrategy::SemanticExpansion { .. } => {
                metrics.semantic_expansions_performed += 1;
            }
            _ => {}
        }
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> VectorPerformanceMetrics {
        self.performance_metrics.lock().unwrap().clone()
    }

    /// Update execution feedback for vector operations
    pub fn update_vector_execution_feedback(
        &mut self,
        _strategy_hash: u64,
        actual_duration: Duration,
        _actual_recall: f32,
        _actual_memory: usize,
        success: bool,
    ) -> Result<()> {
        // Update performance metrics and adaptive thresholds based on execution feedback
        let mut metrics = self.performance_metrics.lock().unwrap();

        if success {
            // Update average speedup calculation
            let base_time = Duration::from_millis(500); // Estimated base query time
            let speedup = base_time.as_millis() as f32 / actual_duration.as_millis() as f32;

            let total_optimizations =
                metrics.vector_queries_optimized + metrics.hybrid_queries_optimized;

            if total_optimizations > 0 {
                metrics.average_optimization_speedup = (metrics.average_optimization_speedup
                    * (total_optimizations - 1) as f32
                    + speedup)
                    / total_optimizations as f32;
            }
        }

        Ok(())
    }
}

/// Vector optimization opportunities discovered in SPARQL queries
#[derive(Debug, Clone)]
pub enum VectorOpportunity {
    /// Text similarity search opportunity
    TextSimilarity {
        pattern: TriplePattern,
        estimated_matches: usize,
    },
    /// Entity similarity search opportunity
    EntitySimilarity {
        pattern: TriplePattern,
        similarity_type: EntitySimilarityType,
    },
    /// Property expansion opportunity
    PropertyExpansion {
        pattern: TriplePattern,
        expansion_depth: usize,
    },
    /// Semantic filter opportunity
    SemanticFilter {
        condition: Expression,
        estimated_selectivity: f32,
    },
    /// Vector-based join opportunity
    VectorJoin {
        left_pattern: Box<Algebra>,
        right_pattern: Box<Algebra>,
        join_variable: Variable,
        estimated_selectivity: f32,
    },
}

/// Types of entity similarity
#[derive(Debug, Clone, Copy)]
pub enum EntitySimilarityType {
    Conceptual,
    Taxonomic,
    Relational,
    Contextual,
}

/// Index recommendation for vector search
#[derive(Debug, Clone)]
pub struct VectorIndexRecommendation {
    pub recommended_type: VectorIndexType,
    pub estimated_benefit: f32,
    pub creation_cost_estimate: Duration,
    pub memory_requirement: usize,
    pub maintenance_overhead: f32,
}
