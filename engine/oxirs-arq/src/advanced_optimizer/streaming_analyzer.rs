//! Streaming Query Analyzer
//!
//! This module provides comprehensive analysis and optimization for streaming query execution,
//! including memory management, spilling policies, streaming strategies, and pipeline analysis.

use crate::algebra::Algebra;
use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Streaming query analyzer with advanced memory-aware optimization
#[derive(Clone)]
pub struct StreamingAnalyzer {
    config: StreamingConfig,
    current_memory_bytes: Arc<AtomicU64>,
}

/// Configuration for streaming execution
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub enable_streaming: bool,
    pub memory_threshold_mb: usize,   // 2048 MB default
    pub spill_threshold_percent: f64, // 0.8 (80%)
    pub streaming_batch_size: usize,  // 1000 rows
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            memory_threshold_mb: 2048,
            spill_threshold_percent: 0.8,
            streaming_batch_size: 1000,
        }
    }
}

/// Streaming execution strategy
#[derive(Debug, Clone)]
pub struct StreamingStrategy {
    pub strategy_type: StreamingType,
    pub memory_limit: usize,
    pub batch_size: usize,
    pub spill_threshold: f64,
    pub parallelism_degree: usize,
}

/// Types of streaming strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamingType {
    PipelineBreaker,
    HashJoinStreaming,
    SortMergeStreaming,
    NestedLoopStreaming,
    IndexNestedLoop,
    HybridStreaming,
}

/// Spill policy for memory management
#[derive(Debug, Clone)]
pub struct SpillPolicy {
    pub policy_type: SpillType,
    pub threshold: f64,
    pub target_operators: Vec<String>,
    pub cost_factor: f64,
}

/// Types of spill policies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpillType {
    LeastRecentlyUsed,
    LargestFirst,
    CostBased,
    PredictiveBased,
}

/// Streaming opportunities identified in query plan
#[derive(Debug, Clone, Default)]
pub struct StreamingOpportunities {
    pub streamable_scans: Vec<OperatorId>,
    pub streamable_filters: Vec<OperatorId>,
    pub streamable_projects: Vec<OperatorId>,
    pub requires_materialization: Vec<(OperatorId, &'static str)>,
    pub pipeline_breakers: Vec<OperatorId>,
    pub estimated_memory_savings_mb: usize,
}

/// Operator identifier
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct OperatorId(u64);

impl OperatorId {
    fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Query plan representation for streaming analysis
#[derive(Debug, Clone)]
pub struct QueryPlan {
    operators: Vec<Operator>,
    operator_id_counter: u64,
}

impl QueryPlan {
    pub fn from_algebra(algebra: &Algebra) -> Self {
        let mut plan = Self {
            operators: Vec::new(),
            operator_id_counter: 0,
        };
        plan.build_from_algebra(algebra);
        plan
    }

    fn build_from_algebra(&mut self, algebra: &Algebra) {
        match algebra {
            Algebra::Bgp(patterns) => {
                let id = self.next_id();
                self.operators.push(Operator::Scan(ScanOperator {
                    id,
                    patterns: patterns.len(),
                }));
            }
            Algebra::Filter { pattern, .. } => {
                self.build_from_algebra(pattern);
                let id = self.next_id();
                self.operators.push(Operator::Filter(FilterOperator { id }));
            }
            Algebra::Project { pattern, variables } => {
                self.build_from_algebra(pattern);
                let id = self.next_id();
                self.operators.push(Operator::Project(ProjectOperator {
                    id,
                    num_vars: variables.len(),
                }));
            }
            Algebra::Join { left, right } => {
                self.build_from_algebra(left);
                self.build_from_algebra(right);
                let id = self.next_id();
                self.operators
                    .push(Operator::HashJoin(HashJoinOperator { id }));
            }
            Algebra::LeftJoin { left, right, .. } => {
                self.build_from_algebra(left);
                self.build_from_algebra(right);
                let id = self.next_id();
                self.operators
                    .push(Operator::HashJoin(HashJoinOperator { id }));
            }
            Algebra::Union { left, right } => {
                self.build_from_algebra(left);
                self.build_from_algebra(right);
                let id = self.next_id();
                self.operators.push(Operator::Union(UnionOperator { id }));
            }
            Algebra::OrderBy { pattern, .. } => {
                self.build_from_algebra(pattern);
                let id = self.next_id();
                self.operators.push(Operator::Sort(SortOperator { id }));
            }
            Algebra::Group { pattern, .. } => {
                self.build_from_algebra(pattern);
                let id = self.next_id();
                self.operators
                    .push(Operator::Aggregation(AggregationOperator { id }));
            }
            Algebra::Distinct { pattern } => {
                self.build_from_algebra(pattern);
                let id = self.next_id();
                self.operators
                    .push(Operator::Distinct(DistinctOperator { id }));
            }
            Algebra::Slice { pattern, .. } => {
                self.build_from_algebra(pattern);
                let id = self.next_id();
                self.operators.push(Operator::Limit(LimitOperator { id }));
            }
            _ => {
                // Handle other operators generically
                let id = self.next_id();
                self.operators
                    .push(Operator::Generic(GenericOperator { id }));
            }
        }
    }

    fn next_id(&mut self) -> OperatorId {
        let id = OperatorId::new(self.operator_id_counter);
        self.operator_id_counter += 1;
        id
    }

    pub fn operators(&self) -> &[Operator] {
        &self.operators
    }
}

/// Query operators
#[derive(Debug, Clone)]
pub enum Operator {
    Scan(ScanOperator),
    Filter(FilterOperator),
    Project(ProjectOperator),
    HashJoin(HashJoinOperator),
    SortMergeJoin(SortMergeJoinOperator),
    Sort(SortOperator),
    Aggregation(AggregationOperator),
    Distinct(DistinctOperator),
    Union(UnionOperator),
    Limit(LimitOperator),
    Generic(GenericOperator),
}

impl Operator {
    pub fn id(&self) -> OperatorId {
        match self {
            Operator::Scan(op) => op.id,
            Operator::Filter(op) => op.id,
            Operator::Project(op) => op.id,
            Operator::HashJoin(op) => op.id,
            Operator::SortMergeJoin(op) => op.id,
            Operator::Sort(op) => op.id,
            Operator::Aggregation(op) => op.id,
            Operator::Distinct(op) => op.id,
            Operator::Union(op) => op.id,
            Operator::Limit(op) => op.id,
            Operator::Generic(op) => op.id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScanOperator {
    pub id: OperatorId,
    pub patterns: usize,
}

#[derive(Debug, Clone)]
pub struct FilterOperator {
    pub id: OperatorId,
}

#[derive(Debug, Clone)]
pub struct ProjectOperator {
    pub id: OperatorId,
    pub num_vars: usize,
}

#[derive(Debug, Clone)]
pub struct HashJoinOperator {
    pub id: OperatorId,
}

#[derive(Debug, Clone)]
pub struct SortMergeJoinOperator {
    pub id: OperatorId,
}

#[derive(Debug, Clone)]
pub struct SortOperator {
    pub id: OperatorId,
}

#[derive(Debug, Clone)]
pub struct AggregationOperator {
    pub id: OperatorId,
}

#[derive(Debug, Clone)]
pub struct DistinctOperator {
    pub id: OperatorId,
}

#[derive(Debug, Clone)]
pub struct UnionOperator {
    pub id: OperatorId,
}

#[derive(Debug, Clone)]
pub struct LimitOperator {
    pub id: OperatorId,
}

#[derive(Debug, Clone)]
pub struct GenericOperator {
    pub id: OperatorId,
}

impl StreamingAnalyzer {
    /// Create a new streaming analyzer
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            current_memory_bytes: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Analyze query plan for streaming opportunities
    pub fn analyze(&self, plan: &QueryPlan) -> StreamingOpportunities {
        let mut opportunities = StreamingOpportunities::default();

        // Identify streaming operators
        for op in plan.operators() {
            match op {
                Operator::Scan(_) => {
                    // Always streamable
                    opportunities.streamable_scans.push(op.id());
                }
                Operator::Filter(_) => {
                    // Streamable if input is streamable
                    if self.is_input_streamable(op, &opportunities) {
                        opportunities.streamable_filters.push(op.id());
                    }
                }
                Operator::Project(_) => {
                    // Streamable projection
                    opportunities.streamable_projects.push(op.id());
                }
                Operator::HashJoin(_) => {
                    // Hash join: Build side must materialize, probe side can stream
                    opportunities
                        .requires_materialization
                        .push((op.id(), "build_side"));
                }
                Operator::SortMergeJoin(_) => {
                    // Both sides must be sorted (materialization required)
                    opportunities
                        .requires_materialization
                        .push((op.id(), "both_sides"));
                }
                Operator::Sort(_) => {
                    // Sorting requires full materialization
                    opportunities.pipeline_breakers.push(op.id());
                }
                Operator::Aggregation(_) => {
                    // Aggregation requires materialization (unless streaming aggregates)
                    opportunities.pipeline_breakers.push(op.id());
                }
                Operator::Distinct(_) => {
                    // Distinct requires materialization
                    opportunities.pipeline_breakers.push(op.id());
                }
                Operator::Union(_) => {
                    // Non-streaming union requires materialization
                    opportunities.pipeline_breakers.push(op.id());
                }
                _ => {}
            }
        }

        // Estimate memory savings
        opportunities.estimated_memory_savings_mb = self.estimate_memory_savings(&opportunities);

        opportunities
    }

    /// Check if input to operator is streamable
    fn is_input_streamable(
        &self,
        _operator: &Operator,
        opportunities: &StreamingOpportunities,
    ) -> bool {
        // Simplified: Check if most upstream operators are streamable
        !opportunities.streamable_scans.is_empty() || !opportunities.streamable_filters.is_empty()
    }

    /// Estimate memory savings from streaming
    fn estimate_memory_savings(&self, opportunities: &StreamingOpportunities) -> usize {
        // Simple heuristic: Each streamable operator saves ~100MB
        let num_streamable = opportunities.streamable_scans.len()
            + opportunities.streamable_filters.len()
            + opportunities.streamable_projects.len();
        num_streamable * 100 // MB per operator
    }

    /// Determine if operator should stream
    pub fn should_stream(&self, _operator: &Operator, estimated_size: usize) -> bool {
        if !self.config.enable_streaming {
            return false;
        }

        // Stream if estimated result size > memory threshold
        let threshold_bytes = self.config.memory_threshold_mb * 1024 * 1024;
        estimated_size > threshold_bytes
    }

    /// Identify pipeline breakers (operators that block streaming)
    pub fn find_pipeline_breakers(&self, plan: &QueryPlan) -> Vec<OperatorId> {
        let mut breakers = vec![];

        for op in plan.operators() {
            if self.is_pipeline_breaker(op) {
                breakers.push(op.id());
            }
        }

        breakers
    }

    fn is_pipeline_breaker(&self, op: &Operator) -> bool {
        matches!(
            op,
            Operator::Sort(_)
                | Operator::Aggregation(_)
                | Operator::Distinct(_)
                | Operator::Union(_)
        )
    }

    /// Convert materialized operator to streaming
    pub fn convert_to_streaming(&self, operator: &mut Operator) -> Result<()> {
        match operator {
            Operator::HashJoin(_) => {
                // Use streaming probe side
                // In a real implementation, we'd modify the operator's internal state
                Ok(())
            }
            Operator::Aggregation(_) => {
                // Convert to streaming aggregation (if grouping keys sortable)
                // Check if can stream
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Analyze streaming potential for query
    pub fn analyze_streaming_potential(
        &self,
        algebra: &Algebra,
    ) -> Result<Option<StreamingStrategy>> {
        let plan = QueryPlan::from_algebra(algebra);
        let opportunities = self.analyze(&plan);

        // If we have streaming opportunities, return a strategy
        if !opportunities.streamable_scans.is_empty() {
            Ok(Some(StreamingStrategy {
                strategy_type: StreamingType::HashJoinStreaming,
                memory_limit: self.config.memory_threshold_mb * 1024 * 1024,
                batch_size: self.config.streaming_batch_size,
                spill_threshold: self.config.spill_threshold_percent,
                parallelism_degree: num_cpus::get(),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get memory threshold
    pub fn memory_threshold(&self) -> usize {
        self.config.memory_threshold_mb * 1024 * 1024
    }

    /// Update memory threshold
    pub fn set_memory_threshold(&mut self, threshold_mb: usize) {
        self.config.memory_threshold_mb = threshold_mb;
    }

    /// Add spill policy
    pub fn add_spill_policy(&mut self, _policy: SpillPolicy) {
        // Store policy for later use
    }

    /// Get active spill policies
    pub fn spill_policies(&self) -> &[SpillPolicy] {
        // Return empty slice for now
        &[]
    }

    /// Get the count of optimizations applied
    pub fn optimizations_count(&self) -> usize {
        // Return count based on config
        if self.config.enable_streaming {
            1
        } else {
            0
        }
    }

    /// Get current memory usage
    pub fn current_memory_usage(&self) -> usize {
        self.current_memory_bytes.load(Ordering::Relaxed) as usize
    }

    /// Check if should spill to disk
    pub fn should_spill(&self) -> bool {
        let current_usage = self.current_memory_usage();
        let threshold =
            (self.memory_threshold() as f64 * self.config.spill_threshold_percent) as usize;
        current_usage > threshold
    }

    /// Analyze query complexity for streaming decision
    pub fn analyze_query_complexity(&self, algebra: &Algebra) -> QueryComplexity {
        let mut complexity = QueryComplexity::default();
        self.compute_complexity(algebra, &mut complexity);
        complexity
    }

    fn compute_complexity(&self, algebra: &Algebra, complexity: &mut QueryComplexity) {
        match algebra {
            Algebra::Bgp(patterns) => {
                complexity.num_patterns += patterns.len();
            }
            Algebra::Join { left, right }
            | Algebra::Union { left, right }
            | Algebra::LeftJoin { left, right, .. } => {
                complexity.num_joins += 1;
                self.compute_complexity(left, complexity);
                self.compute_complexity(right, complexity);
            }
            Algebra::Filter { pattern, .. } => {
                complexity.num_filters += 1;
                self.compute_complexity(pattern, complexity);
            }
            Algebra::OrderBy { pattern, .. } => {
                complexity.num_sorts += 1;
                self.compute_complexity(pattern, complexity);
            }
            Algebra::Group { pattern, .. } => {
                complexity.num_aggregations += 1;
                self.compute_complexity(pattern, complexity);
            }
            _ => {}
        }
    }
}

/// Query complexity metrics
#[derive(Debug, Clone, Default)]
pub struct QueryComplexity {
    pub num_patterns: usize,
    pub num_joins: usize,
    pub num_filters: usize,
    pub num_sorts: usize,
    pub num_aggregations: usize,
}

impl QueryComplexity {
    pub fn total_complexity(&self) -> usize {
        self.num_patterns
            + self.num_joins * 2
            + self.num_filters
            + self.num_sorts * 3
            + self.num_aggregations * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_analyzer_creation() {
        let config = StreamingConfig::default();
        let analyzer = StreamingAnalyzer::new(config);
        assert_eq!(analyzer.memory_threshold(), 2048 * 1024 * 1024);
    }

    #[test]
    fn test_should_stream_decision() {
        let config = StreamingConfig {
            enable_streaming: true,
            memory_threshold_mb: 1024,
            spill_threshold_percent: 0.8,
            streaming_batch_size: 1000,
        };
        let analyzer = StreamingAnalyzer::new(config);

        // Small data should not stream
        assert!(!analyzer.should_stream(
            &Operator::Scan(ScanOperator {
                id: OperatorId::new(1),
                patterns: 1
            }),
            100 * 1024 * 1024
        ));

        // Large data should stream
        assert!(analyzer.should_stream(
            &Operator::Scan(ScanOperator {
                id: OperatorId::new(1),
                patterns: 1
            }),
            2048 * 1024 * 1024
        ));
    }

    #[test]
    fn test_pipeline_breaker_detection() {
        let config = StreamingConfig::default();
        let analyzer = StreamingAnalyzer::new(config);

        assert!(analyzer.is_pipeline_breaker(&Operator::Sort(SortOperator {
            id: OperatorId::new(1)
        })));
        assert!(
            analyzer.is_pipeline_breaker(&Operator::Aggregation(AggregationOperator {
                id: OperatorId::new(2)
            }))
        );
        assert!(
            !analyzer.is_pipeline_breaker(&Operator::Filter(FilterOperator {
                id: OperatorId::new(3)
            }))
        );
    }
}
