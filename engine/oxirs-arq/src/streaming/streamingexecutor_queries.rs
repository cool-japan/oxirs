//! # StreamingExecutor - queries Methods
//!
//! This module contains method implementations for `StreamingExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{Algebra, Term, TriplePattern, Variable};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;
use tracing::debug;

use super::functions::DataStream;
use super::types::{
    BufferedPatternScan, EmptyStream, MemoryMonitor, SpillManager, StreamingConfig,
    StreamingHashJoin, StreamingMinus, StreamingPatternScan, StreamingProjection,
    StreamingSelection, StreamingSort, StreamingSortMergeJoin, StreamingStats, StreamingUnion,
};

use super::streamingexecutor_type::StreamingExecutor;

impl StreamingExecutor {
    /// Create a new streaming executor
    pub fn new(config: StreamingConfig) -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let memory_monitor = MemoryMonitor::new(config.max_memory_usage);
        let spill_manager = Arc::new(Mutex::new(SpillManager::new(
            temp_dir.path().to_path_buf(),
            config.compression_level,
        )?));
        Ok(Self {
            config,
            memory_monitor,
            spill_manager: spill_manager.clone(),
            temp_dir,
            active_streams: HashMap::new(),
            execution_stats: StreamingStats::default(),
        })
    }
    /// Execute algebra node with streaming
    pub(super) fn execute_algebra_streaming(
        &mut self,
        algebra: &Algebra,
    ) -> Result<Box<dyn DataStream>> {
        match algebra {
            Algebra::Join { left, right } => {
                let left_stream = self.execute_algebra_streaming(left)?;
                let right_stream = self.execute_algebra_streaming(right)?;
                let join_variables = self.extract_join_variables(left, right);
                self.create_streaming_join(left_stream, right_stream, join_variables)
            }
            Algebra::Union { left, right } => {
                let left_stream = self.execute_algebra_streaming(left)?;
                let right_stream = self.execute_algebra_streaming(right)?;
                self.create_streaming_union(left_stream, right_stream)
            }
            Algebra::Bgp(patterns) => self.create_bgp_stream(patterns),
            Algebra::Filter { pattern, condition } => {
                let input_stream = self.execute_algebra_streaming(pattern)?;
                let filtered_stream = StreamingSelection {
                    input: input_stream,
                    condition: condition.clone(),
                };
                Ok(Box::new(filtered_stream))
            }
            _ => Err(anyhow!("Unsupported algebra node for streaming")),
        }
    }
    /// Create streaming hash join
    pub(super) fn create_streaming_join(
        &mut self,
        left: Box<dyn DataStream>,
        right: Box<dyn DataStream>,
        join_variables: Vec<Variable>,
    ) -> Result<Box<dyn DataStream>> {
        let left_size = left.estimated_size().unwrap_or(0);
        let right_size = right.estimated_size().unwrap_or(0);
        if left_size + right_size > self.config.max_memory_usage {
            Ok(Box::new(StreamingSortMergeJoin::new(
                left,
                right,
                join_variables,
                Arc::new(self.memory_monitor.clone()),
                self.spill_manager.clone(),
                self.config.clone(),
            )?))
        } else {
            Ok(Box::new(StreamingHashJoin::new(
                left,
                right,
                join_variables,
                Arc::new(self.memory_monitor.clone()),
                self.spill_manager.clone(),
                self.config.clone(),
            )?))
        }
    }
    /// Create streaming union
    pub(super) fn create_streaming_union(
        &mut self,
        left: Box<dyn DataStream>,
        right: Box<dyn DataStream>,
    ) -> Result<Box<dyn DataStream>> {
        Ok(Box::new(StreamingUnion::new(left, right)))
    }
    /// Create streaming minus operation
    #[allow(dead_code)]
    pub(super) fn create_streaming_minus(
        &mut self,
        left: Box<dyn DataStream>,
        right: Box<dyn DataStream>,
    ) -> Result<Box<dyn DataStream>> {
        Ok(Box::new(StreamingMinus::new(
            left,
            right,
            Arc::new(self.memory_monitor.clone()),
            self.spill_manager.clone(),
        )))
    }
    /// Create streaming projection
    #[allow(dead_code)]
    pub(super) fn create_streaming_projection(
        &mut self,
        input: Box<dyn DataStream>,
        variables: Vec<Variable>,
    ) -> Result<Box<dyn DataStream>> {
        Ok(Box::new(StreamingProjection::new(input, variables)))
    }
    /// Create streaming selection
    #[allow(dead_code)]
    pub(super) fn create_streaming_selection(
        &mut self,
        input: Box<dyn DataStream>,
        condition: crate::algebra::Expression,
    ) -> Result<Box<dyn DataStream>> {
        Ok(Box::new(StreamingSelection::new(input, condition)))
    }
    /// Create streaming sort
    #[allow(dead_code)]
    pub(super) fn create_streaming_sort(
        &mut self,
        input: Box<dyn DataStream>,
        sort_variables: Vec<Variable>,
    ) -> Result<Box<dyn DataStream>> {
        Ok(Box::new(StreamingSort::new(
            input,
            sort_variables,
            Arc::new(self.memory_monitor.clone()),
            self.spill_manager.clone(),
            self.config.clone(),
        )?))
    }
    /// Create BGP stream with optimized pattern execution
    pub(super) fn create_bgp_stream(
        &mut self,
        patterns: &[TriplePattern],
    ) -> Result<Box<dyn DataStream>> {
        if patterns.is_empty() {
            return Ok(Box::new(EmptyStream::new()));
        }
        if patterns.len() == 1 {
            self.create_pattern_stream(&patterns[0])
        } else {
            self.create_optimized_bgp_join_stream(patterns)
        }
    }
    /// Create pattern stream from triple pattern
    pub(super) fn create_pattern_stream(
        &mut self,
        pattern: &TriplePattern,
    ) -> Result<Box<dyn DataStream>> {
        let estimated_cardinality = self.estimate_pattern_cardinality(pattern);
        if estimated_cardinality > self.config.max_memory_usage / 1000 {
            Ok(Box::new(StreamingPatternScan::new(
                pattern.clone(),
                Arc::new(self.memory_monitor.clone()),
                self.spill_manager.clone(),
                self.config.clone(),
            )?))
        } else {
            Ok(Box::new(BufferedPatternScan::new(
                pattern.clone(),
                self.config.batch_size,
            )?))
        }
    }
    /// Find join variables between two streams
    pub(super) fn find_join_variables_between_streams(
        &self,
        _left_stream: &dyn DataStream,
        _right_stream: &dyn DataStream,
    ) -> Result<Vec<Variable>> {
        Ok(Vec::new())
    }
    /// Extract all variables from an algebra expression
    pub(super) fn extract_variables_from_algebra(&self, algebra: &Algebra) -> Vec<Variable> {
        let mut variables = Vec::new();
        match algebra {
            Algebra::Join { left, right } => {
                variables.extend(self.extract_variables_from_algebra(left));
                variables.extend(self.extract_variables_from_algebra(right));
            }
            Algebra::Union { left, right } => {
                variables.extend(self.extract_variables_from_algebra(left));
                variables.extend(self.extract_variables_from_algebra(right));
            }
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    if let Term::Variable(var) = &pattern.subject {
                        variables.push(var.clone());
                    }
                    if let Term::Variable(var) = &pattern.predicate {
                        variables.push(var.clone());
                    }
                    if let Term::Variable(var) = &pattern.object {
                        variables.push(var.clone());
                    }
                }
            }
            Algebra::Filter { pattern, condition } => {
                variables.extend(self.extract_variables_from_algebra(pattern));
                variables.extend(self.extract_variables_from_expression(condition));
            }
            Algebra::Project {
                pattern,
                variables: proj_vars,
            } => {
                variables.extend(self.extract_variables_from_algebra(pattern));
                variables.extend(proj_vars.clone());
            }
            _ => {
                debug!("Variable extraction not implemented for algebra type");
            }
        }
        variables.sort();
        variables.dedup();
        variables
    }
    /// Extract variables from a SPARQL expression
    pub(super) fn extract_variables_from_expression(
        &self,
        expr: &crate::algebra::Expression,
    ) -> Vec<Variable> {
        use crate::algebra::Expression;
        let mut variables = Vec::new();
        match expr {
            Expression::Variable(var) => {
                variables.push(var.clone());
            }
            Expression::Binary { left, right, .. } => {
                variables.extend(self.extract_variables_from_expression(left));
                variables.extend(self.extract_variables_from_expression(right));
            }
            Expression::Unary { operand, .. } => {
                variables.extend(self.extract_variables_from_expression(operand));
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                variables.extend(self.extract_variables_from_expression(condition));
                variables.extend(self.extract_variables_from_expression(then_expr));
                variables.extend(self.extract_variables_from_expression(else_expr));
            }
            Expression::Bound(var) => {
                variables.push(var.clone());
            }
            Expression::Function { args, .. } => {
                for arg in args {
                    variables.extend(self.extract_variables_from_expression(arg));
                }
            }
            Expression::Exists(algebra) | Expression::NotExists(algebra) => {
                variables.extend(self.extract_variables_from_algebra(algebra));
            }
            _ => {}
        }
        variables
    }
}
