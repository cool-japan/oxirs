//! Multi-dimensional feature extraction for neural cost estimation

use ndarray::{Array1, Array2};
use oxirs_core::{
    query::{
        algebra::AlgebraTriplePattern,
        pattern_optimizer::IndexType,
    },
};
use std::collections::HashMap;

use super::{config::*, types::*, core::QueryExecutionContext};
use crate::{Result, ShaclAiError};

/// Multi-dimensional feature extractor
#[derive(Debug)]
pub struct MultiDimensionalFeatureExtractor {
    /// Pattern structure analyzer
    pattern_analyzer: PatternStructureAnalyzer,

    /// Index usage analyzer
    index_analyzer: IndexUsageAnalyzer,

    /// Join complexity analyzer
    join_analyzer: JoinComplexityAnalyzer,

    /// System context analyzer
    context_analyzer: SystemContextAnalyzer,

    /// Historical performance analyzer
    historical_analyzer: HistoricalPerformanceAnalyzer,

    /// Configuration
    config: FeatureExtractionConfig,
}

/// Pattern structure analysis
#[derive(Debug)]
pub struct PatternStructureAnalyzer {
    /// Pattern type cache
    pattern_cache: HashMap<String, PatternStructureFeatures>,
}

/// Index usage analysis
#[derive(Debug)]
pub struct IndexUsageAnalyzer {
    /// Index statistics
    index_stats: HashMap<IndexType, IndexUsageStats>,
}

/// Join complexity analysis
#[derive(Debug)]
pub struct JoinComplexityAnalyzer {
    /// Join pattern cache
    join_cache: HashMap<String, JoinComplexityFeatures>,
}

/// System context analysis
#[derive(Debug)]
pub struct SystemContextAnalyzer {
    /// Resource monitors
    resource_monitors: Vec<ResourceMonitor>,
}

/// Historical performance analysis
#[derive(Debug)]
pub struct HistoricalPerformanceAnalyzer {
    /// Performance history
    performance_history: Vec<PerformanceRecord>,
}

/// Performance record
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    pub pattern_hash: String,
    pub execution_time: f64,
    pub resource_usage: ResourceUsage,
    pub context_features: Array1<f64>,
}

impl MultiDimensionalFeatureExtractor {
    pub fn new(config: FeatureExtractionConfig) -> Self {
        Self {
            pattern_analyzer: PatternStructureAnalyzer::new(),
            index_analyzer: IndexUsageAnalyzer::new(),
            join_analyzer: JoinComplexityAnalyzer::new(),
            context_analyzer: SystemContextAnalyzer::new(),
            historical_analyzer: HistoricalPerformanceAnalyzer::new(),
            config,
        }
    }

    /// Extract features from query patterns and context
    pub fn extract_features(
        &mut self,
        patterns: &[AlgebraTriplePattern],
        context: &QueryExecutionContext,
    ) -> Result<Array1<f64>> {
        let mut features = Vec::new();

        // Extract pattern structure features
        if self.config.pattern_structure {
            let pattern_features = self.pattern_analyzer.analyze(patterns)?;
            features.extend_from_slice(&[
                pattern_features.pattern_count,
                pattern_features.variable_count,
                pattern_features.constant_count,
                pattern_features.predicate_variety,
                pattern_features.pattern_depth,
                pattern_features.structural_complexity,
            ]);
        }

        // Extract index usage features
        if self.config.index_usage {
            let index_features = self.index_analyzer.analyze(patterns)?;
            features.extend_from_slice(&[
                index_features.average_usage_frequency(),
                index_features.average_performance(),
                index_features.cache_hit_rate(),
            ]);
        }

        // Extract join complexity features
        if self.config.join_complexity {
            let join_features = self.join_analyzer.analyze(patterns)?;
            features.extend_from_slice(&[
                join_features.join_count,
                join_features.join_cardinality,
                join_features.join_selectivity,
                join_features.cross_product_potential,
                join_features.join_order_complexity,
            ]);
        }

        // Extract system context features
        if self.config.context_features {
            let context_features = self.context_analyzer.analyze(context)?;
            features.extend(context_features);
        }

        // Extract historical performance features
        if self.config.historical_performance {
            let historical_features = self.historical_analyzer.analyze(patterns, context)?;
            features.extend(historical_features);
        }

        // Extract temporal features
        if self.config.temporal_features {
            let temporal_features = self.extract_temporal_features(context)?;
            features.extend(temporal_features);
        }

        // Extract data characteristics features
        if self.config.data_characteristics {
            let data_features = self.extract_data_characteristics(context)?;
            features.extend(data_features);
        }

        // Extract query complexity features
        if self.config.query_complexity {
            let complexity_features = self.extract_query_complexity(patterns)?;
            features.extend(complexity_features);
        }

        // Pad or truncate to expected dimension
        while features.len() < self.config.total_feature_dim {
            features.push(0.0);
        }
        features.truncate(self.config.total_feature_dim);

        Ok(Array1::from(features))
    }

    fn extract_temporal_features(&self, context: &QueryExecutionContext) -> Result<Vec<f64>> {
        let now = std::time::SystemTime::now();
        let timestamp = context.timestamp.duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| ShaclAiError::DataProcessing(format!("Invalid timestamp: {}", e)))?;
        
        Ok(vec![
            (timestamp.as_secs() % 86400) as f64 / 86400.0, // Time of day
            (timestamp.as_secs() % 604800) as f64 / 604800.0, // Day of week
            context.system_load,
        ])
    }

    fn extract_data_characteristics(&self, context: &QueryExecutionContext) -> Result<Vec<f64>> {
        Ok(vec![
            (context.store_size as f64).ln(),
            context.available_memory as f64 / 1024.0 / 1024.0 / 1024.0, // GB
            context.cpu_cores as f64,
            context.cache_size as f64 / 1024.0 / 1024.0, // MB
        ])
    }

    fn extract_query_complexity(&self, patterns: &[AlgebraTriplePattern]) -> Result<Vec<f64>> {
        let pattern_count = patterns.len() as f64;
        let variable_count = self.count_unique_variables(patterns) as f64;
        let complexity_score = pattern_count * variable_count.sqrt();
        
        Ok(vec![pattern_count, variable_count, complexity_score])
    }

    fn count_unique_variables(&self, patterns: &[AlgebraTriplePattern]) -> usize {
        // Simplified variable counting
        patterns.len() * 2 // Approximate
    }
}

impl PatternStructureAnalyzer {
    pub fn new() -> Self {
        Self {
            pattern_cache: HashMap::new(),
        }
    }

    pub fn analyze(&mut self, patterns: &[AlgebraTriplePattern]) -> Result<PatternStructureFeatures> {
        let pattern_key = format!("{:?}", patterns);
        
        if let Some(cached_features) = self.pattern_cache.get(&pattern_key) {
            return Ok(cached_features.clone());
        }

        let features = PatternStructureFeatures {
            pattern_count: patterns.len() as f64,
            variable_count: self.count_variables(patterns) as f64,
            constant_count: self.count_constants(patterns) as f64,
            predicate_variety: self.count_unique_predicates(patterns) as f64,
            pattern_depth: self.calculate_pattern_depth(patterns),
            structural_complexity: self.calculate_structural_complexity(patterns),
        };

        self.pattern_cache.insert(pattern_key, features.clone());
        Ok(features)
    }

    fn count_variables(&self, patterns: &[AlgebraTriplePattern]) -> usize {
        patterns.len() * 2 // Simplified
    }

    fn count_constants(&self, patterns: &[AlgebraTriplePattern]) -> usize {
        patterns.len() // Simplified
    }

    fn count_unique_predicates(&self, patterns: &[AlgebraTriplePattern]) -> usize {
        patterns.len() // Simplified
    }

    fn calculate_pattern_depth(&self, patterns: &[AlgebraTriplePattern]) -> f64 {
        patterns.len() as f64 * 0.5 // Simplified
    }

    fn calculate_structural_complexity(&self, patterns: &[AlgebraTriplePattern]) -> f64 {
        patterns.len() as f64 * 1.2 // Simplified
    }
}

impl IndexUsageAnalyzer {
    pub fn new() -> Self {
        Self {
            index_stats: HashMap::new(),
        }
    }

    pub fn analyze(&mut self, patterns: &[AlgebraTriplePattern]) -> Result<&HashMap<IndexType, IndexUsageStats>> {
        // Analyze index usage for patterns (simplified)
        for _ in patterns {
            self.index_stats.entry(IndexType::SPO).or_insert_with(|| IndexUsageStats {
                usage_frequency: 0.8,
                average_performance: 0.9,
                selectivity_distribution: Array1::from(vec![0.1, 0.3, 0.6]),
                cache_hit_rate: 0.85,
            });
        }
        
        Ok(&self.index_stats)
    }
}

trait IndexStatsAnalysis {
    fn average_usage_frequency(&self) -> f64;
    fn average_performance(&self) -> f64;
    fn cache_hit_rate(&self) -> f64;
}

impl IndexStatsAnalysis for HashMap<IndexType, IndexUsageStats> {
    fn average_usage_frequency(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        self.values().map(|s| s.usage_frequency).sum::<f64>() / self.len() as f64
    }

    fn average_performance(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        self.values().map(|s| s.average_performance).sum::<f64>() / self.len() as f64
    }

    fn cache_hit_rate(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        self.values().map(|s| s.cache_hit_rate).sum::<f64>() / self.len() as f64
    }
}

impl JoinComplexityAnalyzer {
    pub fn new() -> Self {
        Self {
            join_cache: HashMap::new(),
        }
    }

    pub fn analyze(&mut self, patterns: &[AlgebraTriplePattern]) -> Result<JoinComplexityFeatures> {
        let pattern_key = format!("{:?}", patterns);
        
        if let Some(cached_features) = self.join_cache.get(&pattern_key) {
            return Ok(cached_features.clone());
        }

        let features = JoinComplexityFeatures {
            join_count: (patterns.len().saturating_sub(1)) as f64,
            join_cardinality: patterns.len() as f64 * 100.0, // Simplified
            join_selectivity: 0.1 + (patterns.len() as f64 * 0.05), // Simplified
            cross_product_potential: patterns.len() as f64 * 0.1,
            join_order_complexity: patterns.len() as f64 * patterns.len() as f64,
        };

        self.join_cache.insert(pattern_key, features.clone());
        Ok(features)
    }
}

impl SystemContextAnalyzer {
    pub fn new() -> Self {
        Self {
            resource_monitors: vec![
                ResourceMonitor {
                    resource_type: ResourceType::CPU,
                    current_usage: 0.5,
                    average_usage: 0.6,
                    peak_usage: 0.9,
                },
                ResourceMonitor {
                    resource_type: ResourceType::Memory,
                    current_usage: 0.4,
                    average_usage: 0.5,
                    peak_usage: 0.8,
                },
            ],
        }
    }

    pub fn analyze(&self, context: &QueryExecutionContext) -> Result<Vec<f64>> {
        Ok(vec![
            context.system_load,
            context.concurrent_queries as f64 / 100.0,
            context.query_complexity,
            self.resource_monitors[0].current_usage,
            self.resource_monitors[1].current_usage,
        ])
    }
}

impl HistoricalPerformanceAnalyzer {
    pub fn new() -> Self {
        Self {
            performance_history: Vec::new(),
        }
    }

    pub fn analyze(
        &self,
        patterns: &[AlgebraTriplePattern],
        context: &QueryExecutionContext,
    ) -> Result<Vec<f64>> {
        // Simplified historical analysis
        Ok(vec![
            0.5, // Average historical performance
            0.8, // Confidence in historical data
            patterns.len() as f64 * 0.1, // Pattern similarity score
        ])
    }

    pub fn add_record(
        &mut self,
        patterns: &[AlgebraTriplePattern],
        execution_time: f64,
        resource_usage: ResourceUsage,
        context_features: Array1<f64>,
    ) {
        let pattern_hash = format!("{:?}", patterns);
        self.performance_history.push(PerformanceRecord {
            pattern_hash,
            execution_time,
            resource_usage,
            context_features,
        });

        // Keep only recent records
        if self.performance_history.len() > 10000 {
            self.performance_history.drain(0..1000);
        }
    }
}