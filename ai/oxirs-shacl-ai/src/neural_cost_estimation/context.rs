//! Context-aware cost adjustment for neural cost estimation

use super::{types::*, core::QueryExecutionContext};
use crate::{Result, ShaclAiError};

/// Context-aware cost adjuster
#[derive(Debug)]
pub struct ContextAwareCostAdjuster {
    /// Context adjustment factors
    adjustment_factors: ContextAdjustmentFactors,
    
    /// Historical context patterns
    context_patterns: Vec<ContextPattern>,
    
    /// Adjustment statistics
    adjustment_stats: AdjustmentStatistics,
}

/// Context adjustment factors
#[derive(Debug, Clone)]
pub struct ContextAdjustmentFactors {
    pub system_load_factor: f64,
    pub memory_pressure_factor: f64,
    pub concurrent_query_factor: f64,
    pub cache_efficiency_factor: f64,
    pub time_of_day_factor: f64,
}

/// Context pattern
#[derive(Debug, Clone)]
pub struct ContextPattern {
    pub context_signature: ContextSignature,
    pub adjustment_multiplier: f64,
    pub confidence: f64,
    pub usage_count: usize,
}

/// Context signature for pattern matching
#[derive(Debug, Clone)]
pub struct ContextSignature {
    pub system_load_range: (f64, f64),
    pub memory_usage_range: (f64, f64),
    pub concurrent_queries_range: (usize, usize),
    pub time_bucket: usize, // Hour of day bucket (0-23)
}

/// Adjustment statistics
#[derive(Debug, Clone)]
pub struct AdjustmentStatistics {
    pub total_adjustments: usize,
    pub average_adjustment_factor: f64,
    pub improvement_score: f64,
    pub pattern_match_rate: f64,
}

impl ContextAwareCostAdjuster {
    pub fn new() -> Self {
        Self {
            adjustment_factors: ContextAdjustmentFactors::default(),
            context_patterns: Vec::new(),
            adjustment_stats: AdjustmentStatistics::default(),
        }
    }

    /// Adjust cost prediction based on context
    pub fn adjust_cost(
        &mut self,
        base_prediction: &CostPrediction,
        context: &QueryExecutionContext,
    ) -> Result<CostPrediction> {
        let adjustment_factor = self.calculate_adjustment_factor(context)?;
        
        let adjusted_cost = base_prediction.estimated_cost * adjustment_factor;
        
        let mut adjusted_prediction = base_prediction.clone();
        adjusted_prediction.estimated_cost = adjusted_cost;
        adjusted_prediction.execution_time = 
            std::time::Duration::from_millis((adjusted_cost * 1000.0) as u64);
        
        // Adjust resource usage proportionally
        adjusted_prediction.resource_usage.cpu_usage *= adjustment_factor;
        adjusted_prediction.resource_usage.memory_usage *= adjustment_factor;
        adjusted_prediction.resource_usage.disk_io *= adjustment_factor;
        adjusted_prediction.resource_usage.network_io *= adjustment_factor;
        
        // Update uncertainty based on context confidence
        let context_confidence = self.calculate_context_confidence(context);
        adjusted_prediction.uncertainty = base_prediction.uncertainty * (2.0 - context_confidence);
        adjusted_prediction.confidence = base_prediction.confidence * context_confidence;
        
        // Update statistics
        self.update_adjustment_stats(adjustment_factor);
        
        Ok(adjusted_prediction)
    }

    fn calculate_adjustment_factor(&mut self, context: &QueryExecutionContext) -> Result<f64> {
        let mut total_factor = 1.0;
        
        // System load adjustment
        total_factor *= 1.0 + (context.system_load - 0.5) * self.adjustment_factors.system_load_factor;
        
        // Memory pressure adjustment
        let memory_usage = context.available_memory as f64 / (8.0 * 1024.0 * 1024.0 * 1024.0); // Normalize to 8GB
        total_factor *= 1.0 + (1.0 - memory_usage) * self.adjustment_factors.memory_pressure_factor;
        
        // Concurrent query adjustment
        let concurrent_factor = (context.concurrent_queries as f64 / 10.0).min(2.0);
        total_factor *= 1.0 + concurrent_factor * self.adjustment_factors.concurrent_query_factor;
        
        // Time of day adjustment
        let hour = self.extract_hour_from_timestamp(context.timestamp);
        let time_factor = self.get_time_of_day_factor(hour);
        total_factor *= time_factor;
        
        // Apply pattern-based adjustments
        if let Some(pattern_factor) = self.find_matching_pattern(context) {
            total_factor *= pattern_factor;
        }
        
        Ok(total_factor.max(0.1).min(10.0)) // Clamp to reasonable range
    }

    fn calculate_context_confidence(&self, context: &QueryExecutionContext) -> f64 {
        let mut confidence = 1.0;
        
        // Reduce confidence for extreme conditions
        if context.system_load > 0.9 {
            confidence *= 0.7;
        }
        
        if context.concurrent_queries > 20 {
            confidence *= 0.8;
        }
        
        let memory_pressure = 1.0 - (context.available_memory as f64 / (8.0 * 1024.0 * 1024.0 * 1024.0));
        if memory_pressure > 0.9 {
            confidence *= 0.6;
        }
        
        confidence.max(0.1f64)
    }

    fn extract_hour_from_timestamp(&self, timestamp: std::time::SystemTime) -> usize {
        let duration = timestamp.duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::from_secs(0));
        ((duration.as_secs() / 3600) % 24) as usize
    }

    fn get_time_of_day_factor(&self, hour: usize) -> f64 {
        // Simple time-based adjustment
        match hour {
            0..=6 => 0.8,   // Night time - less system load
            7..=9 => 1.2,   // Morning peak
            10..=16 => 1.0, // Normal business hours
            17..=19 => 1.3, // Evening peak
            20..=23 => 0.9, // Evening decline
            _ => 1.0,
        }
    }

    fn find_matching_pattern(&mut self, context: &QueryExecutionContext) -> Option<f64> {
        // First find the matching pattern index without borrowing self
        let mut matching_index = None;
        for (i, pattern) in self.context_patterns.iter().enumerate() {
            if self.matches_pattern(&pattern.context_signature, context) {
                matching_index = Some(i);
                break;
            }
        }
        
        // Then update the pattern if found
        if let Some(index) = matching_index {
            let pattern = &mut self.context_patterns[index];
            pattern.usage_count += 1;
            Some(pattern.adjustment_multiplier)
        } else {
            None
        }
    }

    fn matches_pattern(&self, signature: &ContextSignature, context: &QueryExecutionContext) -> bool {
        let hour = self.extract_hour_from_timestamp(context.timestamp);
        let memory_usage = context.available_memory as f64 / (8.0 * 1024.0 * 1024.0 * 1024.0);
        
        context.system_load >= signature.system_load_range.0 &&
        context.system_load <= signature.system_load_range.1 &&
        memory_usage >= signature.memory_usage_range.0 &&
        memory_usage <= signature.memory_usage_range.1 &&
        context.concurrent_queries >= signature.concurrent_queries_range.0 &&
        context.concurrent_queries <= signature.concurrent_queries_range.1 &&
        hour == signature.time_bucket
    }

    fn update_adjustment_stats(&mut self, adjustment_factor: f64) {
        self.adjustment_stats.total_adjustments += 1;
        
        // Update running average
        let n = self.adjustment_stats.total_adjustments as f64;
        self.adjustment_stats.average_adjustment_factor = 
            (self.adjustment_stats.average_adjustment_factor * (n - 1.0) + adjustment_factor) / n;
    }

    /// Learn new context pattern
    pub fn learn_pattern(
        &mut self,
        context: &QueryExecutionContext,
        optimal_adjustment: f64,
    ) -> Result<()> {
        let signature = ContextSignature {
            system_load_range: (context.system_load - 0.1, context.system_load + 0.1),
            memory_usage_range: (
                (context.available_memory as f64 / (8.0 * 1024.0 * 1024.0 * 1024.0)) - 0.1,
                (context.available_memory as f64 / (8.0 * 1024.0 * 1024.0 * 1024.0)) + 0.1,
            ),
            concurrent_queries_range: (
                context.concurrent_queries.saturating_sub(2),
                context.concurrent_queries + 2,
            ),
            time_bucket: self.extract_hour_from_timestamp(context.timestamp),
        };

        let pattern = ContextPattern {
            context_signature: signature,
            adjustment_multiplier: optimal_adjustment,
            confidence: 0.5, // Initial confidence
            usage_count: 1,
        };

        self.context_patterns.push(pattern);

        // Keep only the most useful patterns
        if self.context_patterns.len() > 100 {
            self.context_patterns.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));
            self.context_patterns.truncate(100);
        }

        Ok(())
    }

    /// Get adjustment statistics
    pub fn get_stats(&self) -> &AdjustmentStatistics {
        &self.adjustment_stats
    }
}

impl Default for ContextAdjustmentFactors {
    fn default() -> Self {
        Self {
            system_load_factor: 0.5,
            memory_pressure_factor: 0.3,
            concurrent_query_factor: 0.2,
            cache_efficiency_factor: 0.4,
            time_of_day_factor: 0.1,
        }
    }
}

impl Default for AdjustmentStatistics {
    fn default() -> Self {
        Self {
            total_adjustments: 0,
            average_adjustment_factor: 1.0,
            improvement_score: 0.0,
            pattern_match_rate: 0.0,
        }
    }
}