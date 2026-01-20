//! Production Pipeline Example
//!
//! This example demonstrates production features for SPARQL endpoints:
//! - Query Priority System for intelligent scheduling
//! - Query Cost Estimator for proactive optimization
//! - Performance Baseline Tracker for regression detection
//!
//! # Scenario
//!
//! A production SPARQL endpoint that:
//! 1. Estimates query costs before execution
//! 2. Schedules queries based on priority and estimated cost
//! 3. Tracks performance baselines and detects regressions
//! 4. Provides comprehensive monitoring and alerting
//!
//! # Usage
//!
//! ```bash
//! cargo run --example production_pipeline
//! ```

use oxirs_arq::production::{
    BaselineTrackerConfig, CostEstimatorConfig, CostRecommendation, PerformanceBaselineTracker,
    PrioritySchedulerConfig, QueryCostEstimator, QueryFeatures, QueryPriority,
    QueryPriorityScheduler, RegressionSeverity,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Represents a query submission to the production pipeline
#[derive(Debug, Clone)]
struct QuerySubmission {
    query_text: String,
    user_id: Option<String>,
    priority: QueryPriority,
}

/// Production query pipeline orchestrating all Beta.2 features
struct ProductionQueryPipeline {
    cost_estimator: Arc<QueryCostEstimator>,
    priority_scheduler: Arc<QueryPriorityScheduler>,
    baseline_tracker: Arc<PerformanceBaselineTracker>,
}

impl ProductionQueryPipeline {
    fn new() -> Self {
        // Configure cost estimator with custom weights
        let cost_config = CostEstimatorConfig {
            pattern_weight: 10.0,
            join_weight: 50.0,
            filter_weight: 20.0,
            aggregate_weight: 30.0,
            path_weight: 100.0,
            enable_ml_prediction: false,
        };

        // Configure priority scheduler with reasonable limits
        let scheduler_config = PrioritySchedulerConfig {
            max_per_priority: 100,
            max_total_queued: 500,
            max_concurrent_per_priority: {
                let mut map = std::collections::HashMap::new();
                map.insert(QueryPriority::Critical, 10);
                map.insert(QueryPriority::High, 8);
                map.insert(QueryPriority::Normal, 5);
                map.insert(QueryPriority::Low, 3);
                map.insert(QueryPriority::Batch, 2);
                map
            },
            enable_aging: true,
            aging_threshold: Duration::from_secs(30),
        };

        // Configure baseline tracker for regression detection
        let baseline_config = BaselineTrackerConfig {
            window_size: 100,
            regression_threshold: 0.2, // 20% degradation triggers alert
            min_samples: 10,
            auto_update_baseline: true,
        };

        Self {
            cost_estimator: Arc::new(QueryCostEstimator::new(cost_config)),
            priority_scheduler: Arc::new(QueryPriorityScheduler::new(scheduler_config)),
            baseline_tracker: Arc::new(PerformanceBaselineTracker::new(baseline_config)),
        }
    }

    /// Submit a query to the production pipeline
    fn submit_query(&self, submission: QuerySubmission) -> Result<u64, String> {
        // Step 1: Extract query features (in production, this would parse the query)
        let features = self.extract_query_features(&submission.query_text);

        // Step 2: Estimate query cost
        let cost_estimate = self.cost_estimator.estimate_cost(&features);

        println!(
            "📊 Cost Analysis for query from {:?}:",
            submission.user_id.as_deref().unwrap_or("anonymous")
        );
        println!("   Estimated cost: {:.2}", cost_estimate.estimated_cost);
        println!(
            "   Estimated duration: {:.2}ms",
            cost_estimate.estimated_duration_ms
        );
        println!(
            "   Estimated memory: {:.2}MB",
            cost_estimate.estimated_memory_mb
        );
        println!("   Complexity score: {:.2}", cost_estimate.complexity_score);
        println!("   Recommendation: {:?}", cost_estimate.recommendation);

        // Step 3: Adjust priority based on cost recommendation
        let adjusted_priority =
            self.adjust_priority_by_cost(submission.priority, &cost_estimate.recommendation);

        if adjusted_priority != submission.priority {
            println!(
                "⚠️  Priority adjusted from {:?} to {:?} based on cost estimate",
                submission.priority, adjusted_priority
            );
        }

        // Step 4: Submit to priority scheduler
        let query_id = self
            .priority_scheduler
            .submit_query(
                submission.query_text.clone(),
                adjusted_priority,
                submission.user_id,
                Some(cost_estimate.estimated_cost),
            )
            .map_err(|e| format!("Failed to queue query: {}", e))?;

        println!(
            "✅ Query {} queued with priority {:?}\n",
            query_id, adjusted_priority
        );

        Ok(query_id)
    }

    /// Execute the next query from the queue
    fn execute_next_query(&self) -> Option<QueryExecutionResult> {
        // Step 1: Get next query from scheduler
        let query = self.priority_scheduler.next_query()?;

        println!(
            "🚀 Executing query {} (priority: {:?})",
            query.query_id, query.priority
        );

        // Step 2: Simulate query execution
        let start = Instant::now();
        let (duration_ms, result_count) = self.simulate_query_execution(&query.query_text);
        let elapsed = start.elapsed();

        println!(
            "✅ Query {} completed in {:.2}ms ({} results)",
            query.query_id, duration_ms, result_count
        );

        // Step 3: Record actual cost for learning
        let features = self.extract_query_features(&query.query_text);
        self.cost_estimator
            .record_actual_cost(features.clone(), duration_ms);

        // Step 4: Track performance baseline
        let pattern = self.normalize_query_pattern(&query.query_text);
        self.baseline_tracker.record_execution(
            pattern.clone(),
            duration_ms,
            10.0, // Simulated memory usage
            result_count,
        );

        // Step 5: Check for performance regression
        if let Some(regression) = self
            .baseline_tracker
            .check_regression(&pattern, duration_ms)
        {
            println!("⚠️  PERFORMANCE REGRESSION DETECTED!");
            println!("   Pattern: {}", regression.query_pattern);
            println!(
                "   Baseline: {:.2}ms, Current: {:.2}ms",
                regression.baseline_duration_ms, regression.current_duration_ms
            );
            println!("   Degradation: {:.1}%", regression.degradation_percentage);
            println!("   Severity: {:?}", regression.severity);

            match regression.severity {
                RegressionSeverity::Critical => {
                    println!("   🔴 CRITICAL: Immediate investigation required!");
                }
                RegressionSeverity::High => {
                    println!("   🟠 HIGH: Investigation recommended");
                }
                RegressionSeverity::Moderate => {
                    println!("   🟡 MODERATE: Monitor closely");
                }
            }
            println!();
        }

        // Step 6: Mark query as complete in scheduler
        self.priority_scheduler.complete_query(query.query_id);

        Some(QueryExecutionResult {
            query_id: query.query_id,
            duration: elapsed,
            result_count,
        })
    }

    /// Extract query features (simplified for demonstration)
    fn extract_query_features(&self, query: &str) -> QueryFeatures {
        // In production, this would parse the SPARQL query
        // For demo, we estimate based on query text characteristics
        let pattern_count = query.matches("?").count();
        let join_count = pattern_count.saturating_sub(1);
        let filter_count = query.matches("FILTER").count();
        let aggregate_count = query.matches("COUNT").count()
            + query.matches("SUM").count()
            + query.matches("AVG").count();
        let path_count = query.matches("/").count();
        let optional_count = query.matches("OPTIONAL").count();
        let union_count = query.matches("UNION").count();
        let distinct = query.contains("DISTINCT");
        let order_by = query.contains("ORDER BY");
        let group_by = query.contains("GROUP BY");

        let limit = if query.contains("LIMIT") {
            Some(10) // Simplified
        } else {
            None
        };

        QueryFeatures {
            pattern_count,
            join_count,
            filter_count,
            aggregate_count,
            path_count,
            optional_count,
            union_count,
            distinct,
            order_by,
            group_by,
            limit,
        }
    }

    /// Adjust priority based on cost recommendation
    fn adjust_priority_by_cost(
        &self,
        original_priority: QueryPriority,
        recommendation: &CostRecommendation,
    ) -> QueryPriority {
        match (original_priority, recommendation) {
            // Critical stays critical
            (QueryPriority::Critical, _) => QueryPriority::Critical,

            // Very expensive queries get downgraded unless critical
            (QueryPriority::High, CostRecommendation::VeryExpensive) => QueryPriority::Normal,
            (QueryPriority::Normal, CostRecommendation::VeryExpensive) => QueryPriority::Low,
            (QueryPriority::Low, CostRecommendation::VeryExpensive) => QueryPriority::Batch,

            // Lightweight queries can stay at current priority
            _ => original_priority,
        }
    }

    /// Simulate query execution (in production, this would actually execute the query)
    fn simulate_query_execution(&self, query: &str) -> (f64, usize) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        let hash = hasher.finish();

        // Simulate varying execution times based on query complexity
        let base_time = 10.0 + (hash % 100) as f64;
        let complexity_factor = if query.contains("OPTIONAL") { 2.0 } else { 1.0 };
        let duration_ms = base_time * complexity_factor;

        let result_count = 10 + (hash % 90) as usize;

        (duration_ms, result_count)
    }

    /// Normalize query pattern for baseline tracking
    fn normalize_query_pattern(&self, query: &str) -> String {
        // In production, this would extract the query structure
        // For demo, we use a simplified pattern
        if query.contains("SELECT") && query.contains("COUNT") {
            "SELECT_COUNT_PATTERN".to_string()
        } else if query.contains("SELECT") {
            "SELECT_PATTERN".to_string()
        } else if query.contains("ASK") {
            "ASK_PATTERN".to_string()
        } else {
            "UNKNOWN_PATTERN".to_string()
        }
    }

    /// Get comprehensive statistics from all components
    fn get_statistics(&self) -> PipelineStatistics {
        let scheduler_stats = self.priority_scheduler.get_stats();
        let cost_stats = self.cost_estimator.get_statistics();
        let tracked_patterns = self.baseline_tracker.get_tracked_patterns();

        PipelineStatistics {
            queued_queries: scheduler_stats.total_queued,
            active_queries: scheduler_stats.total_active,
            cost_samples: cost_stats.sample_count,
            avg_cost: cost_stats.avg_cost,
            tracked_patterns: tracked_patterns.len(),
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)] // Used for debugging and tracking
struct QueryExecutionResult {
    query_id: u64,
    duration: Duration,
    result_count: usize,
}

#[derive(Debug)]
struct PipelineStatistics {
    queued_queries: usize,
    active_queries: usize,
    cost_samples: usize,
    avg_cost: f64,
    tracked_patterns: usize,
}

fn main() {
    println!("🚀 OxiRS ARQ - Production Pipeline Example (Beta.2)\n");
    println!("This example demonstrates the integration of:");
    println!("  • Query Priority System");
    println!("  • Query Cost Estimator");
    println!("  • Performance Baseline Tracker\n");
    println!("{}", "═".repeat(80));
    println!();

    // Initialize production pipeline
    let pipeline = ProductionQueryPipeline::new();

    // Example queries with different characteristics
    let queries = [
        QuerySubmission {
            query_text: "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10".to_string(),
            user_id: Some("user1".to_string()),
            priority: QueryPriority::Normal,
        },
        QuerySubmission {
            query_text: "SELECT (COUNT(?s) as ?count) WHERE { ?s ?p ?o }".to_string(),
            user_id: Some("admin".to_string()),
            priority: QueryPriority::Critical,
        },
        QuerySubmission {
            query_text: "SELECT ?s ?p ?o WHERE { ?s ?p ?o . OPTIONAL { ?s ?x ?y } }".to_string(),
            user_id: Some("user2".to_string()),
            priority: QueryPriority::High,
        },
        QuerySubmission {
            query_text: "SELECT DISTINCT ?s WHERE { ?s ?p ?o FILTER (?o > 100) } ORDER BY ?s"
                .to_string(),
            user_id: None,
            priority: QueryPriority::Low,
        },
        QuerySubmission {
            query_text: "ASK WHERE { ?s ?p ?o }".to_string(),
            user_id: Some("batch_job".to_string()),
            priority: QueryPriority::Batch,
        },
    ];

    // Submit all queries
    println!(
        "📥 Submitting {} queries to the pipeline...\n",
        queries.len()
    );
    let mut submitted_ids = Vec::new();

    for (i, query) in queries.iter().enumerate() {
        println!("Query {}: {}", i + 1, query.query_text);
        match pipeline.submit_query(query.clone()) {
            Ok(query_id) => {
                submitted_ids.push(query_id);
            }
            Err(e) => {
                println!("❌ Failed to submit query: {}\n", e);
            }
        }
    }

    println!("{}", "═".repeat(80));
    println!("\n📊 Pipeline Statistics After Submission:");
    let stats = pipeline.get_statistics();
    println!("   Queued queries: {}", stats.queued_queries);
    println!("   Active queries: {}", stats.active_queries);
    println!();

    // Execute all queries
    println!("{}", "═".repeat(80));
    println!("\n⚙️  Executing queries in priority order...\n");

    let mut execution_results = Vec::new();
    while let Some(result) = pipeline.execute_next_query() {
        execution_results.push(result);
    }

    // Final statistics
    println!("{}", "═".repeat(80));
    println!("\n📈 Final Pipeline Statistics:");
    let final_stats = pipeline.get_statistics();
    println!("   Total queries processed: {}", execution_results.len());
    println!("   Queued queries: {}", final_stats.queued_queries);
    println!("   Active queries: {}", final_stats.active_queries);
    println!("   Cost estimation samples: {}", final_stats.cost_samples);
    println!("   Average cost: {:.2}", final_stats.avg_cost);
    println!(
        "   Tracked performance patterns: {}",
        final_stats.tracked_patterns
    );
    println!();

    // Summary
    println!("{}", "═".repeat(80));
    println!("\n✅ Production Pipeline Example Completed!\n");
    println!("Key Takeaways:");
    println!("  1. Queries were estimated and prioritized based on cost");
    println!("  2. Priority-based scheduling ensured critical queries ran first");
    println!("  3. Performance baselines were established and monitored");
    println!("  4. Regression detection would alert on performance degradation");
    println!("\nThis production-ready pipeline provides:");
    println!("  • Intelligent resource allocation");
    println!("  • Proactive cost management");
    println!("  • Automated performance monitoring");
    println!("  • Quality assurance through regression detection");
    println!();
}
