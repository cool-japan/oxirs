//! Cost Model Module
//!
//! Provides comprehensive cost modeling for query optimization including
//! I/O cost modeling, CPU cost estimation, and memory usage prediction.

use crate::algebra::{Algebra, Expression, TriplePattern, Variable};
use crate::statistics_collector::StatisticsCollector;
use anyhow::Result;
use std::collections::HashMap;

/// Cost model configuration
#[derive(Debug, Clone)]
pub struct CostModelConfig {
    /// CPU cost per operation (relative units)
    pub cpu_cost_per_op: f64,
    /// I/O cost per page read
    pub io_cost_per_page: f64,
    /// Memory cost per byte allocated
    pub memory_cost_per_byte: f64,
    /// Network cost per byte transferred
    pub network_cost_per_byte: f64,
    /// Page size in bytes
    pub page_size: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Cost model calibration factors
    pub calibration: CostCalibration,
}

impl Default for CostModelConfig {
    fn default() -> Self {
        Self {
            cpu_cost_per_op: 1.0,
            io_cost_per_page: 10.0,
            memory_cost_per_byte: 0.001,
            network_cost_per_byte: 0.1,
            page_size: 4096,
            available_memory: 1024 * 1024 * 1024, // 1GB
            calibration: CostCalibration::default(),
        }
    }
}

/// Cost calibration factors based on system characteristics
#[derive(Debug, Clone)]
pub struct CostCalibration {
    /// CPU scaling factor
    pub cpu_scale: f64,
    /// I/O scaling factor
    pub io_scale: f64,
    /// Memory scaling factor
    pub memory_scale: f64,
    /// Network scaling factor
    pub network_scale: f64,
    /// Join algorithm cost factors
    pub join_factors: JoinCostFactors,
}

impl Default for CostCalibration {
    fn default() -> Self {
        Self {
            cpu_scale: 1.0,
            io_scale: 1.0,
            memory_scale: 1.0,
            network_scale: 1.0,
            join_factors: JoinCostFactors::default(),
        }
    }
}

/// Cost factors for different join algorithms
#[derive(Debug, Clone)]
pub struct JoinCostFactors {
    /// Hash join cost factor
    pub hash_join_factor: f64,
    /// Sort-merge join cost factor
    pub sort_merge_join_factor: f64,
    /// Nested loop join cost factor
    pub nested_loop_join_factor: f64,
    /// Index nested loop join cost factor
    pub index_join_factor: f64,
}

impl Default for JoinCostFactors {
    fn default() -> Self {
        Self {
            hash_join_factor: 1.0,
            sort_merge_join_factor: 1.2,
            nested_loop_join_factor: 2.0,
            index_join_factor: 0.8,
        }
    }
}

/// Comprehensive cost estimate
#[derive(Debug, Clone)]
pub struct CostEstimate {
    /// CPU cost in relative units
    pub cpu_cost: f64,
    /// I/O cost in relative units
    pub io_cost: f64,
    /// Memory cost in relative units
    pub memory_cost: f64,
    /// Network cost in relative units
    pub network_cost: f64,
    /// Total cost (weighted combination)
    pub total_cost: f64,
    /// Estimated cardinality
    pub cardinality: usize,
    /// Estimated selectivity
    pub selectivity: f64,
    /// Cost breakdown by operation
    pub operation_costs: HashMap<String, f64>,
}

impl CostEstimate {
    pub fn new(cpu: f64, io: f64, memory: f64, network: f64, cardinality: usize) -> Self {
        let total = cpu + io + memory + network;
        Self {
            cpu_cost: cpu,
            io_cost: io,
            memory_cost: memory,
            network_cost: network,
            total_cost: total,
            cardinality,
            selectivity: 1.0,
            operation_costs: HashMap::new(),
        }
    }

    pub fn with_selectivity(mut self, selectivity: f64) -> Self {
        self.selectivity = selectivity;
        self
    }

    pub fn add_operation_cost(&mut self, operation: &str, cost: f64) {
        self.operation_costs.insert(operation.to_string(), cost);
        self.total_cost += cost;
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0)
    }

    pub fn infinite() -> Self {
        Self::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            usize::MAX,
        )
    }
}

/// I/O access pattern for cost estimation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IOPattern {
    /// Sequential read pattern
    Sequential,
    /// Random read pattern
    Random,
    /// Index scan pattern
    IndexScan,
    /// Full table scan pattern
    FullScan,
}

/// Memory usage pattern
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Average memory usage in bytes
    pub average_usage: usize,
    /// Memory access pattern
    pub access_pattern: MemoryAccessPattern,
    /// Duration of memory usage
    pub duration_estimate: f64,
}

/// Memory access patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryAccessPattern {
    /// Sequential access
    Sequential,
    /// Random access
    Random,
    /// Locality of reference
    Locality,
    /// Cache-friendly access
    CacheFriendly,
}

/// Cost model for query optimization
#[derive(Debug, Clone)]
pub struct CostModel {
    config: CostModelConfig,
    statistics: Option<StatisticsCollector>,
    cached_estimates: HashMap<String, CostEstimate>,
}

impl CostModel {
    /// Create a new cost model
    pub fn new(config: CostModelConfig) -> Self {
        Self {
            config,
            statistics: None,
            cached_estimates: HashMap::new(),
        }
    }

    /// Create cost model with statistics
    pub fn with_statistics(config: CostModelConfig, statistics: StatisticsCollector) -> Self {
        Self {
            config,
            statistics: Some(statistics),
            cached_estimates: HashMap::new(),
        }
    }

    /// Estimate cost of an algebra expression
    pub fn estimate_cost(&mut self, algebra: &Algebra) -> Result<CostEstimate> {
        // Check cache first
        let algebra_key = self.algebra_to_key(algebra);
        if let Some(cached) = self.cached_estimates.get(&algebra_key) {
            return Ok(cached.clone());
        }

        let estimate = self.estimate_cost_recursive(algebra)?;

        // Cache the result
        self.cached_estimates.insert(algebra_key, estimate.clone());

        Ok(estimate)
    }

    fn estimate_cost_recursive(&self, algebra: &Algebra) -> Result<CostEstimate> {
        match algebra {
            Algebra::Bgp(patterns) if patterns.len() == 1 => {
                self.estimate_triple_pattern_cost(&patterns[0])
            }
            Algebra::Bgp(patterns) => self.estimate_bgp_cost(patterns),
            Algebra::Join { left, right } => self.estimate_join_cost(left, right),
            Algebra::LeftJoin { left, right, .. } => self.estimate_left_join_cost(left, right),
            Algebra::Union { left, right } => self.estimate_union_cost(left, right),
            Algebra::Filter { condition, pattern } => self.estimate_filter_cost(condition, pattern),
            Algebra::Project { pattern, .. } => self.estimate_project_cost(pattern),
            Algebra::Extend { expr, pattern, .. } => self.estimate_extend_cost(&expr, pattern),
            Algebra::Distinct { pattern } => self.estimate_distinct_cost(pattern),
            Algebra::Reduced { pattern } => self.estimate_reduced_cost(pattern),
            Algebra::OrderBy { pattern, .. } => self.estimate_order_by_cost(pattern),
            Algebra::Slice {
                pattern,
                offset,
                limit,
            } => self.estimate_slice_cost(pattern, *offset, *limit),
            Algebra::Group { pattern, .. } => self.estimate_group_cost(pattern),
            Algebra::PropertyPath { .. } => {
                // Property path has higher cost due to path traversal
                Ok(CostEstimate::new(5000.0, 1000.0, 500.0, 0.0, 2000))
            }
            Algebra::Minus { left, right } => {
                let left_cost = self.estimate_cost_recursive(left)?;
                let right_cost = self.estimate_cost_recursive(right)?;
                Ok(CostEstimate::new(
                    left_cost.total_cost + right_cost.total_cost * 2.0,
                    0.0,
                    0.0,
                    0.0,
                    left_cost.cardinality,
                ))
            }
            Algebra::Service { pattern, .. } => {
                // Service calls have high latency
                let pattern_cost = self.estimate_cost_recursive(pattern)?;
                Ok(CostEstimate::new(
                    pattern_cost.total_cost,
                    0.0,
                    0.0,
                    pattern_cost.total_cost * 10.0 + 1000.0, // Network overhead
                    pattern_cost.cardinality,
                ))
            }
            Algebra::Graph { pattern, .. } => {
                // Graph patterns similar to regular patterns
                self.estimate_cost_recursive(pattern)
            }
            Algebra::Having { pattern, condition } => {
                let pattern_cost = self.estimate_cost_recursive(pattern)?;
                let filter_selectivity = self.estimate_expression_selectivity(condition);
                Ok(CostEstimate::new(
                    pattern_cost.total_cost * 1.1, // Small overhead for having
                    0.0,
                    0.0,
                    0.0,
                    (pattern_cost.cardinality as f64 * filter_selectivity) as usize,
                ))
            }
            Algebra::Values { bindings, .. } => {
                // Values clause cost depends on number of bindings
                Ok(CostEstimate::new(
                    bindings.len() as f64 * 0.1,
                    0.0,
                    0.0,
                    0.0,
                    bindings.len(),
                ))
            }
            Algebra::Table => {
                // Empty table - minimal cost
                Ok(CostEstimate::new(0.1, 0.0, 0.0, 0.0, 1))
            }
            Algebra::Zero => {
                // Zero results - minimal cost
                Ok(CostEstimate::new(0.1, 0.0, 0.0, 0.0, 0))
            }
        }
    }

    /// Estimate cost of a triple pattern
    fn estimate_triple_pattern_cost(&self, pattern: &TriplePattern) -> Result<CostEstimate> {
        // Calculate selectivity based on pattern specificity
        let selectivity = self.estimate_pattern_selectivity(pattern);

        // Estimate cardinality (would use actual statistics in production)
        let base_cardinality = 100000; // Assume 100k triples
        let cardinality = (base_cardinality as f64 * selectivity) as usize;

        // I/O cost depends on access pattern
        let io_pattern = self.determine_io_pattern(pattern);
        let pages_accessed = self.estimate_pages_accessed(cardinality, io_pattern);
        let io_cost =
            pages_accessed as f64 * self.config.io_cost_per_page * self.config.calibration.io_scale;

        // CPU cost for pattern matching
        let cpu_cost =
            cardinality as f64 * self.config.cpu_cost_per_op * self.config.calibration.cpu_scale;

        // Memory cost for buffering results
        let memory_usage = cardinality * 100; // Assume 100 bytes per triple
        let memory_cost = memory_usage as f64
            * self.config.memory_cost_per_byte
            * self.config.calibration.memory_scale;

        let mut estimate = CostEstimate::new(cpu_cost, io_cost, memory_cost, 0.0, cardinality)
            .with_selectivity(selectivity);

        estimate.add_operation_cost("pattern_scan", cpu_cost + io_cost);

        Ok(estimate)
    }

    /// Estimate cost of a basic graph pattern
    fn estimate_bgp_cost(&self, patterns: &[TriplePattern]) -> Result<CostEstimate> {
        if patterns.is_empty() {
            return Ok(CostEstimate::zero());
        }

        if patterns.len() == 1 {
            return self.estimate_triple_pattern_cost(&patterns[0]);
        }

        // For multiple patterns, estimate as a series of joins
        let mut total_cost = CostEstimate::zero();
        let mut current_cardinality = 1;

        for pattern in patterns {
            let pattern_cost = self.estimate_triple_pattern_cost(pattern)?;

            // Join cost with previous results
            let join_cost = self.estimate_join_cost_detailed(
                current_cardinality,
                pattern_cost.cardinality,
                0.1, // Assume 10% selectivity
                JoinAlgorithm::HashJoin,
            );

            total_cost.cpu_cost += pattern_cost.cpu_cost + join_cost.cpu_cost;
            total_cost.io_cost += pattern_cost.io_cost + join_cost.io_cost;
            total_cost.memory_cost += pattern_cost.memory_cost + join_cost.memory_cost;

            current_cardinality = join_cost.cardinality;
        }

        total_cost.cardinality = current_cardinality;
        total_cost.total_cost = total_cost.cpu_cost + total_cost.io_cost + total_cost.memory_cost;

        Ok(total_cost)
    }

    /// Estimate cost of a join operation
    fn estimate_join_cost(&self, left: &Algebra, right: &Algebra) -> Result<CostEstimate> {
        let left_cost = self.estimate_cost_recursive(left)?;
        let right_cost = self.estimate_cost_recursive(right)?;

        // Choose join algorithm based on sizes
        let algorithm = self.choose_join_algorithm(left_cost.cardinality, right_cost.cardinality);

        // Estimate join selectivity (would be based on actual statistics)
        let join_selectivity = 0.1; // Assume 10% selectivity

        let join_cost = self.estimate_join_cost_detailed(
            left_cost.cardinality,
            right_cost.cardinality,
            join_selectivity,
            algorithm,
        );

        let total_cpu = left_cost.cpu_cost + right_cost.cpu_cost + join_cost.cpu_cost;
        let total_io = left_cost.io_cost + right_cost.io_cost + join_cost.io_cost;
        let total_memory =
            left_cost.memory_cost.max(right_cost.memory_cost) + join_cost.memory_cost;

        let mut estimate = CostEstimate::new(
            total_cpu,
            total_io,
            total_memory,
            0.0,
            join_cost.cardinality,
        );
        estimate.add_operation_cost("join", join_cost.total_cost);

        Ok(estimate)
    }

    /// Estimate cost of a left join (optional) operation
    fn estimate_left_join_cost(&self, left: &Algebra, right: &Algebra) -> Result<CostEstimate> {
        let left_cost = self.estimate_cost_recursive(left)?;
        let right_cost = self.estimate_cost_recursive(right)?;

        // Left join preserves all left-side tuples
        let result_cardinality =
            left_cost.cardinality + (right_cost.cardinality as f64 * 0.5) as usize;

        // Cost similar to inner join but with additional overhead
        let base_join_cost = self.estimate_join_cost_detailed(
            left_cost.cardinality,
            right_cost.cardinality,
            0.5, // Higher selectivity for left join
            self.choose_join_algorithm(left_cost.cardinality, right_cost.cardinality),
        );

        let total_cpu = left_cost.cpu_cost + right_cost.cpu_cost + base_join_cost.cpu_cost * 1.2;
        let total_io = left_cost.io_cost + right_cost.io_cost + base_join_cost.io_cost;
        let total_memory =
            left_cost.memory_cost.max(right_cost.memory_cost) + base_join_cost.memory_cost;

        Ok(CostEstimate::new(
            total_cpu,
            total_io,
            total_memory,
            0.0,
            result_cardinality,
        ))
    }

    /// Estimate cost of a union operation
    fn estimate_union_cost(&self, left: &Algebra, right: &Algebra) -> Result<CostEstimate> {
        let left_cost = self.estimate_cost_recursive(left)?;
        let right_cost = self.estimate_cost_recursive(right)?;

        // Union combines both sides
        let result_cardinality = left_cost.cardinality + right_cost.cardinality;

        // Cost is sum of both sides plus union overhead
        let union_overhead = (result_cardinality as f64 * 0.1) * self.config.cpu_cost_per_op;

        let total_cpu = left_cost.cpu_cost + right_cost.cpu_cost + union_overhead;
        let total_io = left_cost.io_cost + right_cost.io_cost;
        let total_memory = left_cost.memory_cost + right_cost.memory_cost;

        Ok(CostEstimate::new(
            total_cpu,
            total_io,
            total_memory,
            0.0,
            result_cardinality,
        ))
    }

    /// Estimate cost of a filter operation
    fn estimate_filter_cost(
        &self,
        expression: &Expression,
        input: &Algebra,
    ) -> Result<CostEstimate> {
        let input_cost = self.estimate_cost_recursive(input)?;

        // Estimate filter selectivity
        let filter_selectivity = self.estimate_expression_selectivity(expression);
        let result_cardinality = (input_cost.cardinality as f64 * filter_selectivity) as usize;

        // CPU cost for evaluating filter on each tuple
        let filter_cpu_cost = input_cost.cardinality as f64
            * self.config.cpu_cost_per_op
            * self.estimate_expression_complexity(expression);

        let total_cpu = input_cost.cpu_cost + filter_cpu_cost;
        let total_io = input_cost.io_cost;
        let total_memory = input_cost.memory_cost;

        Ok(
            CostEstimate::new(total_cpu, total_io, total_memory, 0.0, result_cardinality)
                .with_selectivity(filter_selectivity),
        )
    }

    /// Estimate cost of a projection operation
    fn estimate_project_cost(&self, input: &Algebra) -> Result<CostEstimate> {
        let input_cost = self.estimate_cost_recursive(input)?;

        // Projection doesn't change cardinality but may reduce memory usage
        let memory_reduction = 0.8; // Assume 20% memory reduction
        let projection_cpu = input_cost.cardinality as f64 * self.config.cpu_cost_per_op * 0.1;

        let total_cpu = input_cost.cpu_cost + projection_cpu;
        let total_memory = input_cost.memory_cost * memory_reduction;

        Ok(CostEstimate::new(
            total_cpu,
            input_cost.io_cost,
            total_memory,
            0.0,
            input_cost.cardinality,
        ))
    }

    /// Estimate cost of an extend (BIND) operation
    fn estimate_extend_cost(
        &self,
        expression: &Expression,
        input: &Algebra,
    ) -> Result<CostEstimate> {
        let input_cost = self.estimate_cost_recursive(input)?;

        // Cost of evaluating expression for each tuple
        let expression_cost = input_cost.cardinality as f64
            * self.config.cpu_cost_per_op
            * self.estimate_expression_complexity(expression);

        let total_cpu = input_cost.cpu_cost + expression_cost;

        Ok(CostEstimate::new(
            total_cpu,
            input_cost.io_cost,
            input_cost.memory_cost,
            0.0,
            input_cost.cardinality,
        ))
    }

    /// Estimate cost of a distinct operation
    fn estimate_distinct_cost(&self, input: &Algebra) -> Result<CostEstimate> {
        let input_cost = self.estimate_cost_recursive(input)?;

        // Distinct requires sorting or hashing
        let distinct_cardinality = (input_cost.cardinality as f64 * 0.8) as usize; // Assume 20% duplicates
        let sort_cost = input_cost.cardinality as f64
            * (input_cost.cardinality as f64).log2()
            * self.config.cpu_cost_per_op;

        let total_cpu = input_cost.cpu_cost + sort_cost;
        let total_memory = input_cost.memory_cost * 1.5; // Additional memory for sorting

        Ok(CostEstimate::new(
            total_cpu,
            input_cost.io_cost,
            total_memory,
            0.0,
            distinct_cardinality,
        ))
    }

    /// Estimate cost of a reduced operation
    fn estimate_reduced_cost(&self, input: &Algebra) -> Result<CostEstimate> {
        // Similar to distinct but with less strict requirements
        let input_cost = self.estimate_cost_recursive(input)?;
        let reduced_cpu = input_cost.cardinality as f64 * self.config.cpu_cost_per_op * 0.1;

        let total_cpu = input_cost.cpu_cost + reduced_cpu;

        Ok(CostEstimate::new(
            total_cpu,
            input_cost.io_cost,
            input_cost.memory_cost,
            0.0,
            input_cost.cardinality,
        ))
    }

    /// Estimate cost of an order by operation
    fn estimate_order_by_cost(&self, input: &Algebra) -> Result<CostEstimate> {
        let input_cost = self.estimate_cost_recursive(input)?;

        // Sorting cost
        let sort_cost = input_cost.cardinality as f64
            * (input_cost.cardinality as f64).log2()
            * self.config.cpu_cost_per_op;

        let total_cpu = input_cost.cpu_cost + sort_cost;
        let total_memory = input_cost.memory_cost * 2.0; // Additional memory for sorting

        Ok(CostEstimate::new(
            total_cpu,
            input_cost.io_cost,
            total_memory,
            0.0,
            input_cost.cardinality,
        ))
    }

    /// Estimate cost of a slice (LIMIT/OFFSET) operation
    fn estimate_slice_cost(
        &self,
        input: &Algebra,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<CostEstimate> {
        let input_cost = self.estimate_cost_recursive(input)?;

        let offset_val = offset.unwrap_or(0);
        let limit_val = limit.unwrap_or(input_cost.cardinality);
        let result_cardinality = limit_val.min(input_cost.cardinality.saturating_sub(offset_val));

        // Slice doesn't add significant CPU cost
        let slice_cpu = result_cardinality as f64 * self.config.cpu_cost_per_op * 0.01;

        Ok(CostEstimate::new(
            input_cost.cpu_cost + slice_cpu,
            input_cost.io_cost,
            input_cost.memory_cost,
            0.0,
            result_cardinality,
        ))
    }

    /// Estimate cost of a group by operation
    fn estimate_group_cost(&self, input: &Algebra) -> Result<CostEstimate> {
        let input_cost = self.estimate_cost_recursive(input)?;

        // Grouping requires sorting or hashing
        let group_cardinality = (input_cost.cardinality as f64 * 0.1) as usize; // Assume 10 groups per 100 tuples
        let group_cost = input_cost.cardinality as f64
            * (input_cost.cardinality as f64).log2()
            * self.config.cpu_cost_per_op;

        let total_cpu = input_cost.cpu_cost + group_cost;
        let total_memory = input_cost.memory_cost * 1.5;

        Ok(CostEstimate::new(
            total_cpu,
            input_cost.io_cost,
            total_memory,
            0.0,
            group_cardinality,
        ))
    }

    /// Detailed join cost estimation with algorithm selection
    fn estimate_join_cost_detailed(
        &self,
        left_cardinality: usize,
        right_cardinality: usize,
        selectivity: f64,
        algorithm: JoinAlgorithm,
    ) -> CostEstimate {
        let result_cardinality =
            (left_cardinality as f64 * right_cardinality as f64 * selectivity) as usize;

        match algorithm {
            JoinAlgorithm::HashJoin => {
                let build_cost =
                    left_cardinality.min(right_cardinality) as f64 * self.config.cpu_cost_per_op;
                let probe_cost =
                    left_cardinality.max(right_cardinality) as f64 * self.config.cpu_cost_per_op;
                let cpu_cost = (build_cost + probe_cost)
                    * self.config.calibration.join_factors.hash_join_factor;

                let memory_cost = (left_cardinality.min(right_cardinality) * 50) as f64
                    * self.config.memory_cost_per_byte;
                let io_cost = 0.0; // Hash join is memory-based

                CostEstimate::new(cpu_cost, io_cost, memory_cost, 0.0, result_cardinality)
            }
            JoinAlgorithm::SortMergeJoin => {
                let sort_cost_left = left_cardinality as f64
                    * (left_cardinality as f64).log2()
                    * self.config.cpu_cost_per_op;
                let sort_cost_right = right_cardinality as f64
                    * (right_cardinality as f64).log2()
                    * self.config.cpu_cost_per_op;
                let merge_cost =
                    (left_cardinality + right_cardinality) as f64 * self.config.cpu_cost_per_op;
                let cpu_cost = (sort_cost_left + sort_cost_right + merge_cost)
                    * self.config.calibration.join_factors.sort_merge_join_factor;

                let memory_cost = (left_cardinality + right_cardinality) as f64
                    * 50.0
                    * self.config.memory_cost_per_byte;
                let io_cost = 0.0;

                CostEstimate::new(cpu_cost, io_cost, memory_cost, 0.0, result_cardinality)
            }
            JoinAlgorithm::NestedLoopJoin => {
                let cpu_cost = (left_cardinality as f64 * right_cardinality as f64)
                    * self.config.cpu_cost_per_op
                    * self.config.calibration.join_factors.nested_loop_join_factor;
                let memory_cost = 1000.0 * self.config.memory_cost_per_byte; // Minimal memory usage
                let io_cost = 0.0;

                CostEstimate::new(cpu_cost, io_cost, memory_cost, 0.0, result_cardinality)
            }
            JoinAlgorithm::IndexJoin => {
                let cpu_cost = (left_cardinality as f64 * (right_cardinality as f64).log2())
                    * self.config.cpu_cost_per_op
                    * self.config.calibration.join_factors.index_join_factor;
                let io_cost = left_cardinality as f64 * self.config.io_cost_per_page * 0.1; // Index lookups
                let memory_cost = 5000.0 * self.config.memory_cost_per_byte;

                CostEstimate::new(cpu_cost, io_cost, memory_cost, 0.0, result_cardinality)
            }
        }
    }

    /// Choose optimal join algorithm based on input sizes
    fn choose_join_algorithm(&self, left_size: usize, right_size: usize) -> JoinAlgorithm {
        let smaller = left_size.min(right_size);
        let larger = left_size.max(right_size);

        if smaller < 1000 {
            JoinAlgorithm::NestedLoopJoin
        } else if smaller < 10000 && larger > smaller * 10 {
            JoinAlgorithm::IndexJoin
        } else if smaller * 8 < self.config.available_memory / 100 {
            JoinAlgorithm::HashJoin
        } else {
            JoinAlgorithm::SortMergeJoin
        }
    }

    /// Estimate pattern selectivity based on specificity
    fn estimate_pattern_selectivity(&self, pattern: &TriplePattern) -> f64 {
        let mut specificity = 0;

        if !matches!(pattern.subject, crate::algebra::Term::Variable(_)) {
            specificity += 1;
        }
        if !matches!(pattern.predicate, crate::algebra::Term::Variable(_)) {
            specificity += 1;
        }
        if !matches!(pattern.object, crate::algebra::Term::Variable(_)) {
            specificity += 1;
        }

        match specificity {
            0 => 1.0,   // All variables
            1 => 0.1,   // One constant
            2 => 0.01,  // Two constants
            3 => 0.001, // All constants
            _ => 0.0001,
        }
    }

    /// Determine I/O access pattern for a triple pattern
    fn determine_io_pattern(&self, pattern: &TriplePattern) -> IOPattern {
        // Simple heuristic based on pattern structure
        if matches!(pattern.subject, crate::algebra::Term::Variable(_))
            && matches!(pattern.predicate, crate::algebra::Term::Variable(_))
            && matches!(pattern.object, crate::algebra::Term::Variable(_))
        {
            IOPattern::FullScan
        } else if !matches!(pattern.predicate, crate::algebra::Term::Variable(_)) {
            IOPattern::IndexScan
        } else {
            IOPattern::Random
        }
    }

    /// Estimate number of pages accessed
    fn estimate_pages_accessed(&self, cardinality: usize, pattern: IOPattern) -> usize {
        let bytes_per_triple = 100; // Estimated average
        let total_bytes = cardinality * bytes_per_triple;
        let pages = (total_bytes + self.config.page_size - 1) / self.config.page_size;

        match pattern {
            IOPattern::Sequential => pages,
            IOPattern::Random => pages * 2, // Random access penalty
            IOPattern::IndexScan => pages / 2, // Index efficiency
            IOPattern::FullScan => pages,
        }
    }

    /// Estimate expression selectivity
    fn estimate_expression_selectivity(&self, expression: &Expression) -> f64 {
        match expression {
            Expression::Binary { op: operator, .. } => match operator {
                crate::algebra::BinaryOperator::Equal => 0.1,
                crate::algebra::BinaryOperator::NotEqual => 0.9,
                crate::algebra::BinaryOperator::Less => 0.3,
                crate::algebra::BinaryOperator::LessEqual => 0.4,
                crate::algebra::BinaryOperator::Greater => 0.3,
                crate::algebra::BinaryOperator::GreaterEqual => 0.4,
                _ => 0.5,
            },
            Expression::Function { name, .. } => match name.as_str() {
                "contains" => 0.2,
                "startsWith" => 0.1,
                "endsWith" => 0.1,
                "regex" => 0.05,
                _ => 0.5,
            },
            _ => 0.5, // Default selectivity
        }
    }

    /// Estimate expression complexity for CPU cost
    fn estimate_expression_complexity(&self, expression: &Expression) -> f64 {
        match expression {
            Expression::Variable(_) | Expression::Literal(_) => 1.0,
            Expression::Binary { .. } => 2.0,
            Expression::Unary { .. } => 1.5,
            Expression::Function { name, args } => {
                let base_cost = match name.as_str() {
                    "regex" => 10.0,
                    "contains" => 3.0,
                    "startsWith" | "endsWith" => 2.0,
                    _ => 5.0,
                };
                base_cost + args.len() as f64
            }
            Expression::Conditional { .. } => 3.0,
            _ => 2.0,
        }
    }

    /// Generate a cache key for an algebra expression
    fn algebra_to_key(&self, algebra: &Algebra) -> String {
        // Simple implementation - in production would use better hashing
        format!("{:?}", algebra)
    }

    /// Clear the cost estimation cache
    pub fn clear_cache(&mut self) {
        self.cached_estimates.clear();
    }

    /// Update cost model with execution feedback
    pub fn update_with_feedback(
        &mut self,
        algebra: &Algebra,
        actual_cost: f64,
        actual_cardinality: usize,
    ) {
        // Update calibration factors based on actual vs predicted costs
        let predicted = self.estimate_cost(algebra).unwrap_or(CostEstimate::zero());

        if predicted.total_cost > 0.0 {
            let cost_ratio = actual_cost / predicted.total_cost;
            let cardinality_ratio = actual_cardinality as f64 / predicted.cardinality as f64;

            // Adjust calibration factors (simple approach)
            self.config.calibration.cpu_scale =
                (self.config.calibration.cpu_scale + cost_ratio) / 2.0;

            // Clear cache to force recalculation with new factors
            self.clear_cache();
        }
    }
}

/// Available join algorithms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinAlgorithm {
    HashJoin,
    SortMergeJoin,
    NestedLoopJoin,
    IndexJoin,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Term, Variable};
    use oxirs_core::model::NamedNode;

    #[test]
    fn test_triple_pattern_cost_estimation() {
        let mut cost_model = CostModel::new(CostModelConfig::default());

        let pattern = TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/predicate")),
            object: Term::Variable(Variable::new("o").unwrap()),
        };

        let cost = cost_model.estimate_triple_pattern_cost(&pattern).unwrap();

        assert!(cost.total_cost > 0.0);
        assert!(cost.cardinality > 0);
        assert!(cost.selectivity > 0.0 && cost.selectivity <= 1.0);
    }

    #[test]
    fn test_join_algorithm_selection() {
        let cost_model = CostModel::new(CostModelConfig::default());

        assert_eq!(
            cost_model.choose_join_algorithm(100, 1000),
            JoinAlgorithm::NestedLoopJoin
        );
        assert_eq!(
            cost_model.choose_join_algorithm(10000, 100000),
            JoinAlgorithm::HashJoin
        );
    }

    #[test]
    fn test_cost_estimate_operations() {
        let mut estimate = CostEstimate::new(10.0, 5.0, 2.0, 1.0, 1000);
        estimate.add_operation_cost("test_op", 3.0);

        assert_eq!(estimate.total_cost, 21.0); // 10+5+2+1+3
        assert!(estimate.operation_costs.contains_key("test_op"));
    }

    #[test]
    fn test_pattern_selectivity_estimation() {
        let cost_model = CostModel::new(CostModelConfig::default());

        // All variables
        let pattern1 = TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Variable(Variable::new("p").unwrap()),
            object: Term::Variable(Variable::new("o").unwrap()),
        };

        let selectivity1 = cost_model.estimate_pattern_selectivity(&pattern1);
        assert_eq!(selectivity1, 1.0);

        // One constant
        let pattern2 = TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/predicate")),
            object: Term::Variable(Variable::new("o").unwrap()),
        };

        let selectivity2 = cost_model.estimate_pattern_selectivity(&pattern2);
        assert_eq!(selectivity2, 0.1);
    }
}
