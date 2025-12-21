//! Profiled Query Plan Builder
//!
//! This module bridges the query profiler and query plan visualizer,
//! automatically generating visual query plans from profiling data.
//!
//! # Features
//! - Automatic query plan generation from profiling sessions
//! - Real execution statistics overlay
//! - Performance bottleneck highlighting
//! - Optimization recommendations based on actual execution
//!
//! # Example
//! ```rust,no_run
//! use oxirs_core::query::profiled_plan_builder::ProfiledPlanBuilder;
//! use oxirs_core::query::query_profiler::{QueryProfiler, ProfilerConfig};
//!
//! let profiler = QueryProfiler::new(ProfilerConfig::default());
//! let session = profiler.start_session("SELECT ?s ?p ?o WHERE { ?s ?p ?o }");
//! // ... execute query ...
//! let stats = session.finish();
//!
//! let builder = ProfiledPlanBuilder::new();
//! let plan = builder.build_from_stats(&stats, "SELECT query");
//! let visualizer = QueryPlanVisualizer::new();
//! println!("{}", visualizer.visualize_as_tree(&plan));
//! ```

use crate::query::query_plan_visualizer::{
    HintSeverity, OptimizationHint, QueryPlanNode, QueryPlanVisualizer,
};
use crate::query::query_profiler::QueryStatistics;

/// Builder for creating query plans from profiling data
pub struct ProfiledPlanBuilder {
    /// Visualizer for rendering plans
    visualizer: QueryPlanVisualizer,
    /// Whether to include optimization hints
    include_hints: bool,
}

impl Default for ProfiledPlanBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfiledPlanBuilder {
    /// Create a new profiled plan builder
    pub fn new() -> Self {
        Self {
            visualizer: QueryPlanVisualizer::new(),
            include_hints: true,
        }
    }

    /// Enable or disable optimization hints
    pub fn with_hints(mut self, enable: bool) -> Self {
        self.include_hints = enable;
        self
    }

    /// Build a query plan from profiling statistics
    pub fn build_from_stats(&self, stats: &QueryStatistics, query_text: &str) -> QueryPlanNode {
        let mut root = self.create_root_node(query_text, stats);

        // Add parsing phase
        if stats.parse_time_ms > 0 {
            let parse_node = QueryPlanNode::new("Parse", "Query parsing")
                .with_execution_time(stats.parse_time_ms * 1000) // Convert to μs
                .with_metadata("phase", "parsing");
            root.add_child(parse_node);
        }

        // Add planning phase
        if stats.planning_time_ms > 0 {
            let plan_node = QueryPlanNode::new("Planning", "Query optimization")
                .with_execution_time(stats.planning_time_ms * 1000)
                .with_metadata("phase", "planning");
            root.add_child(plan_node);
        }

        // Add execution phase with pattern details
        let mut exec_node = QueryPlanNode::new("Execution", "Query execution")
            .with_execution_time(stats.execution_time_ms * 1000)
            .with_actual_cardinality(stats.results_count as usize)
            .with_metadata("phase", "execution");

        // Add pattern matching details
        for (pattern, count) in &stats.pattern_matches {
            let pattern_node = QueryPlanNode::new("TriplePattern", pattern)
                .with_actual_cardinality(*count as usize)
                .with_metadata("matches", count.to_string());
            exec_node.add_child(pattern_node);
        }

        // Add join operations
        if stats.join_operations > 0 {
            let join_node =
                QueryPlanNode::new("Join", format!("{} join operations", stats.join_operations))
                    .with_metadata("count", stats.join_operations.to_string());
            exec_node.add_child(join_node);
        }

        // Add index usage
        for (index, count) in &stats.index_accesses {
            let index_node = QueryPlanNode::new("IndexScan", format!("Index: {}", index))
                .with_actual_cardinality(*count as usize)
                .with_index(index.clone())
                .with_metadata("accesses", count.to_string());
            exec_node.add_child(index_node);
        }

        root.add_child(exec_node);
        root
    }

    /// Create root node with overall statistics
    fn create_root_node(&self, query_text: &str, stats: &QueryStatistics) -> QueryPlanNode {
        let description = if query_text.len() > 60 {
            format!("{}...", &query_text[..57])
        } else {
            query_text.to_string()
        };

        QueryPlanNode::new("Query", description)
            .with_execution_time(stats.total_time_ms * 1000) // Convert to μs
            .with_actual_cardinality(stats.results_count as usize)
            .with_metadata("total_triples", stats.triples_matched.to_string())
            .with_metadata(
                "cache_hit_rate",
                format!("{:.1}%", stats.cache_hit_rate * 100.0),
            )
            .with_metadata(
                "memory_peak",
                format!("{}KB", stats.peak_memory_bytes / 1024),
            )
    }

    /// Generate a complete profiling report with visualization
    pub fn generate_report(&self, stats: &QueryStatistics, query_text: &str) -> ProfilingReport {
        let plan = self.build_from_stats(stats, query_text);
        let tree_visualization = self.visualizer.visualize_as_tree(&plan);
        let summary = self.visualizer.generate_summary(&plan);

        let hints = if self.include_hints {
            self.visualizer.suggest_optimizations(&plan)
        } else {
            Vec::new()
        };

        ProfilingReport {
            query: query_text.to_string(),
            statistics: stats.clone(),
            plan,
            tree_visualization,
            summary,
            optimization_hints: hints,
        }
    }

    /// Analyze query performance and generate recommendations
    pub fn analyze_performance(&self, stats: &QueryStatistics) -> PerformanceAnalysis {
        let mut analysis = PerformanceAnalysis {
            is_slow: stats.total_time_ms > 1000,
            slow_phases: Vec::new(),
            inefficient_patterns: Vec::new(),
            index_recommendations: Vec::new(),
            cache_effectiveness: CacheEffectiveness::Unknown,
            overall_grade: PerformanceGrade::Unknown,
        };

        // Analyze phases
        if stats.parse_time_ms > stats.total_time_ms / 4 {
            analysis
                .slow_phases
                .push(format!("Parsing is slow ({}ms)", stats.parse_time_ms));
        }
        if stats.planning_time_ms > stats.total_time_ms / 4 {
            analysis
                .slow_phases
                .push(format!("Planning is slow ({}ms)", stats.planning_time_ms));
        }
        if stats.execution_time_ms > stats.total_time_ms / 2 {
            analysis
                .slow_phases
                .push(format!("Execution is slow ({}ms)", stats.execution_time_ms));
        }

        // Analyze patterns
        let total_matches: u64 = stats.pattern_matches.values().sum();
        for (pattern, count) in &stats.pattern_matches {
            if *count > 10000 {
                analysis.inefficient_patterns.push(format!(
                    "Pattern '{}' matched {} triples (consider adding selectivity)",
                    pattern, count
                ));
            }
        }

        // Analyze index usage
        if stats.index_accesses.is_empty() && total_matches > 1000 {
            analysis.index_recommendations.push(
                "No indexes used with large result set - consider adding indexes".to_string(),
            );
        }

        // Analyze cache
        analysis.cache_effectiveness = if stats.cache_hit_rate > 0.8 {
            CacheEffectiveness::Excellent
        } else if stats.cache_hit_rate > 0.5 {
            CacheEffectiveness::Good
        } else if stats.cache_hit_rate > 0.2 {
            CacheEffectiveness::Fair
        } else {
            CacheEffectiveness::Poor
        };

        // Overall grade
        analysis.overall_grade = self.calculate_grade(stats, &analysis);

        analysis
    }

    /// Calculate overall performance grade
    fn calculate_grade(
        &self,
        stats: &QueryStatistics,
        analysis: &PerformanceAnalysis,
    ) -> PerformanceGrade {
        let mut score = 100.0;

        // Penalize slow execution
        if stats.total_time_ms > 5000 {
            score -= 40.0;
        } else if stats.total_time_ms > 1000 {
            score -= 20.0;
        } else if stats.total_time_ms > 100 {
            score -= 5.0;
        }

        // Penalize inefficient patterns
        score -= (analysis.inefficient_patterns.len() as f64 * 10.0).min(30.0);

        // Penalize missing indexes
        score -= (analysis.index_recommendations.len() as f64 * 15.0).min(20.0);

        // Reward good cache usage
        score += stats.cache_hit_rate as f64 * 10.0;

        match score {
            s if s >= 90.0 => PerformanceGrade::Excellent,
            s if s >= 75.0 => PerformanceGrade::Good,
            s if s >= 60.0 => PerformanceGrade::Fair,
            s if s >= 40.0 => PerformanceGrade::Poor,
            _ => PerformanceGrade::Critical,
        }
    }

    /// Compare two profiling sessions
    pub fn compare_executions(
        &self,
        baseline: &QueryStatistics,
        current: &QueryStatistics,
    ) -> ExecutionComparison {
        let time_diff_pct = if baseline.total_time_ms > 0 {
            ((current.total_time_ms as f64 - baseline.total_time_ms as f64)
                / baseline.total_time_ms as f64)
                * 100.0
        } else {
            0.0
        };

        let memory_diff_pct = if baseline.peak_memory_bytes > 0 {
            ((current.peak_memory_bytes as f64 - baseline.peak_memory_bytes as f64)
                / baseline.peak_memory_bytes as f64)
                * 100.0
        } else {
            0.0
        };

        let improvement = if time_diff_pct < -5.0 {
            ImprovementLevel::Significant
        } else if time_diff_pct < 0.0 {
            ImprovementLevel::Minor
        } else if time_diff_pct < 5.0 {
            ImprovementLevel::None
        } else if time_diff_pct < 20.0 {
            ImprovementLevel::Regression
        } else {
            ImprovementLevel::Critical
        };

        ExecutionComparison {
            time_diff_ms: (current.total_time_ms as i64) - (baseline.total_time_ms as i64),
            time_diff_pct,
            memory_diff_bytes: (current.peak_memory_bytes as i64)
                - (baseline.peak_memory_bytes as i64),
            memory_diff_pct,
            results_diff: (current.results_count as i64) - (baseline.results_count as i64),
            cache_hit_diff: current.cache_hit_rate - baseline.cache_hit_rate,
            improvement,
        }
    }
}

/// Complete profiling report with visualization
#[derive(Debug)]
pub struct ProfilingReport {
    /// Original query text
    pub query: String,
    /// Profiling statistics
    pub statistics: QueryStatistics,
    /// Query plan
    pub plan: QueryPlanNode,
    /// ASCII tree visualization
    pub tree_visualization: String,
    /// Plan summary
    pub summary: crate::query::query_plan_visualizer::QueryPlanSummary,
    /// Optimization hints
    pub optimization_hints: Vec<OptimizationHint>,
}

impl ProfilingReport {
    /// Print a formatted report to stdout
    pub fn print(&self) {
        println!("=== Query Profiling Report ===\n");
        println!("Query: {}\n", self.query);
        println!("Execution Plan:\n{}", self.tree_visualization);
        println!("\n{}", self.summary);

        if !self.optimization_hints.is_empty() {
            println!("\nOptimization Hints:");
            println!("-------------------");
            for hint in &self.optimization_hints {
                let icon = match hint.severity {
                    HintSeverity::Info => "ℹ️",
                    HintSeverity::Warning => "⚠️",
                    HintSeverity::Critical => "🔴",
                };
                println!("{} {}", icon, hint);
            }
        }
    }
}

/// Performance analysis results
#[derive(Debug)]
pub struct PerformanceAnalysis {
    /// Whether query is considered slow
    pub is_slow: bool,
    /// List of slow execution phases
    pub slow_phases: Vec<String>,
    /// Inefficient pattern descriptions
    pub inefficient_patterns: Vec<String>,
    /// Index recommendations
    pub index_recommendations: Vec<String>,
    /// Cache effectiveness rating
    pub cache_effectiveness: CacheEffectiveness,
    /// Overall performance grade
    pub overall_grade: PerformanceGrade,
}

/// Cache effectiveness rating
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEffectiveness {
    Excellent, // > 80%
    Good,      // > 50%
    Fair,      // > 20%
    Poor,      // <= 20%
    Unknown,
}

/// Overall performance grade
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PerformanceGrade {
    Critical,  // < 40 (lowest)
    Poor,      // >= 40
    Fair,      // >= 60
    Good,      // >= 75
    Excellent, // >= 90 (highest)
    Unknown,
}

/// Comparison between two query executions
#[derive(Debug)]
pub struct ExecutionComparison {
    /// Time difference in milliseconds
    pub time_diff_ms: i64,
    /// Time difference as percentage
    pub time_diff_pct: f64,
    /// Memory difference in bytes
    pub memory_diff_bytes: i64,
    /// Memory difference as percentage
    pub memory_diff_pct: f64,
    /// Results count difference
    pub results_diff: i64,
    /// Cache hit rate difference
    pub cache_hit_diff: f32,
    /// Overall improvement level
    pub improvement: ImprovementLevel,
}

/// Improvement level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImprovementLevel {
    /// Significant improvement (>5% faster)
    Significant,
    /// Minor improvement (0-5% faster)
    Minor,
    /// No change (within 5%)
    None,
    /// Performance regression (5-20% slower)
    Regression,
    /// Critical regression (>20% slower)
    Critical,
}

impl std::fmt::Display for ExecutionComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Execution Comparison:")?;
        writeln!(
            f,
            "  Time:   {:+}ms ({:+.1}%)",
            self.time_diff_ms, self.time_diff_pct
        )?;
        writeln!(
            f,
            "  Memory: {:+}KB ({:+.1}%)",
            self.memory_diff_bytes / 1024,
            self.memory_diff_pct
        )?;
        writeln!(f, "  Results: {:+}", self.results_diff)?;
        writeln!(f, "  Cache:   {:+.1}%", self.cache_hit_diff * 100.0)?;
        writeln!(f, "  Overall: {:?}", self.improvement)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_sample_stats() -> QueryStatistics {
        let mut pattern_matches = HashMap::new();
        pattern_matches.insert("?s rdf:type foaf:Person".to_string(), 500);
        pattern_matches.insert("?s foaf:name ?name".to_string(), 500);

        let mut index_accesses = HashMap::new();
        index_accesses.insert("SPO".to_string(), 2);

        QueryStatistics {
            total_time_ms: 150,
            parse_time_ms: 10,
            planning_time_ms: 20,
            execution_time_ms: 120,
            triples_matched: 1000,
            results_count: 50,
            peak_memory_bytes: 1024 * 1024, // 1MB
            join_operations: 2,
            cache_hit_rate: 0.75,
            pattern_matches,
            index_accesses,
            ..Default::default()
        }
    }

    #[test]
    fn test_plan_builder_basic() {
        let builder = ProfiledPlanBuilder::new();
        let stats = create_sample_stats();
        let plan = builder.build_from_stats(
            &stats,
            "SELECT ?s ?name WHERE { ?s a foaf:Person . ?s foaf:name ?name }",
        );

        // Should have root query node
        assert_eq!(plan.node_type, "Query");
        assert!(plan.execution_time_us.is_some());
        assert_eq!(plan.actual_cardinality, Some(50));

        // Should have child nodes for phases
        assert!(!plan.children.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let builder = ProfiledPlanBuilder::new();
        let stats = create_sample_stats();
        let report = builder.generate_report(&stats, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }");

        assert_eq!(report.query, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }");
        assert!(!report.tree_visualization.is_empty());
        assert!(report.summary.total_nodes > 0);
    }

    #[test]
    fn test_performance_analysis() {
        let builder = ProfiledPlanBuilder::new();
        let stats = create_sample_stats();
        let analysis = builder.analyze_performance(&stats);

        assert!(!analysis.is_slow); // 150ms is not slow
        assert_eq!(analysis.cache_effectiveness, CacheEffectiveness::Good);
    }

    #[test]
    fn test_slow_query_detection() {
        let builder = ProfiledPlanBuilder::new();
        let mut stats = create_sample_stats();
        stats.total_time_ms = 5000; // 5 seconds - slow!
        stats.cache_hit_rate = 0.1; // Poor cache usage
        stats.index_accesses.clear(); // No index usage

        // Add inefficient pattern
        stats.pattern_matches.insert("?s ?p ?o".to_string(), 50000);

        let analysis = builder.analyze_performance(&stats);
        assert!(analysis.is_slow);
        // With poor stats, should be Poor or Critical
        assert!(matches!(
            analysis.overall_grade,
            PerformanceGrade::Poor | PerformanceGrade::Critical
        ));
    }

    #[test]
    fn test_execution_comparison() {
        let builder = ProfiledPlanBuilder::new();
        let baseline = create_sample_stats();

        let mut improved = baseline.clone();
        improved.total_time_ms = 100; // 33% faster

        let comparison = builder.compare_executions(&baseline, &improved);
        assert_eq!(comparison.time_diff_ms, -50);
        assert!(comparison.time_diff_pct < 0.0);
        assert_eq!(comparison.improvement, ImprovementLevel::Significant);
    }

    #[test]
    fn test_regression_detection() {
        let builder = ProfiledPlanBuilder::new();
        let baseline = create_sample_stats();

        let mut regressed = baseline.clone();
        regressed.total_time_ms = 200; // 33% slower

        let comparison = builder.compare_executions(&baseline, &regressed);
        assert!(comparison.time_diff_ms > 0);
        assert!(comparison.time_diff_pct > 20.0);
        assert_eq!(comparison.improvement, ImprovementLevel::Critical);
    }

    #[test]
    fn test_cache_effectiveness() {
        let builder = ProfiledPlanBuilder::new();

        let mut stats_excellent = create_sample_stats();
        stats_excellent.cache_hit_rate = 0.9;
        let analysis = builder.analyze_performance(&stats_excellent);
        assert_eq!(analysis.cache_effectiveness, CacheEffectiveness::Excellent);

        let mut stats_poor = create_sample_stats();
        stats_poor.cache_hit_rate = 0.1;
        let analysis = builder.analyze_performance(&stats_poor);
        assert_eq!(analysis.cache_effectiveness, CacheEffectiveness::Poor);
    }

    #[test]
    fn test_inefficient_pattern_detection() {
        let builder = ProfiledPlanBuilder::new();
        let mut stats = create_sample_stats();
        stats.pattern_matches.insert("?s ?p ?o".to_string(), 50000); // Very broad pattern

        let analysis = builder.analyze_performance(&stats);
        assert!(!analysis.inefficient_patterns.is_empty());
    }
}
