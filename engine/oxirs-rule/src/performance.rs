//! Performance analysis and profiling utilities for the rule engine
//!
//! This module provides tools for analyzing rule engine performance,
//! identifying bottlenecks, generating performance reports, and enabling
//! parallel rule evaluation for improved throughput.

use crate::{Rule, RuleAtom, RuleEngine, Term};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Performance metrics for rule engine operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total execution time
    pub total_time: Duration,
    /// Time spent on rule loading
    pub rule_loading_time: Duration,
    /// Time spent on fact processing
    pub fact_processing_time: Duration,
    /// Time spent on forward chaining
    pub forward_chaining_time: Duration,
    /// Time spent on backward chaining
    pub backward_chaining_time: Duration,
    /// Number of rules processed
    pub rules_processed: usize,
    /// Number of facts processed
    pub facts_processed: usize,
    /// Number of inferred facts
    pub inferred_facts: usize,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Rule-specific timing information
    pub rule_timings: HashMap<String, Duration>,
    /// Performance warnings and bottlenecks
    pub warnings: Vec<String>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_memory_usage: usize,
    /// Memory used by facts
    pub facts_memory: usize,
    /// Memory used by rules
    pub rules_memory: usize,
    /// Memory used by derived facts
    pub derived_facts_memory: usize,
}

/// Performance profiler for rule engine operations
#[derive(Debug)]
pub struct RuleEngineProfiler {
    /// Start time of profiling session
    start_time: Instant,
    /// Individual operation timings
    operation_timings: HashMap<String, Vec<Duration>>,
    /// Current operation stack
    operation_stack: Vec<(String, Instant)>,
    /// Memory snapshots
    memory_snapshots: Vec<(String, usize)>,
    /// Performance thresholds
    thresholds: PerformanceThresholds,
}

/// Performance thresholds for detecting bottlenecks
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum acceptable rule loading time (ms)
    pub max_rule_loading_time: u64,
    /// Maximum acceptable forward chaining time (ms)
    pub max_forward_chaining_time: u64,
    /// Maximum acceptable backward chaining time (ms)
    pub max_backward_chaining_time: u64,
    /// Maximum memory usage (MB)
    pub max_memory_usage: usize,
    /// Maximum number of iterations before warning
    pub max_iterations: usize,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_rule_loading_time: 1000,      // 1 second
            max_forward_chaining_time: 5000,  // 5 seconds
            max_backward_chaining_time: 2000, // 2 seconds
            max_memory_usage: 1024,           // 1 GB
            max_iterations: 1000,
        }
    }
}

impl Default for RuleEngineProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleEngineProfiler {
    /// Create a new profiler with default thresholds
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            operation_timings: HashMap::new(),
            operation_stack: Vec::new(),
            memory_snapshots: Vec::new(),
            thresholds: PerformanceThresholds::default(),
        }
    }

    /// Create a new profiler with custom thresholds
    pub fn with_thresholds(thresholds: PerformanceThresholds) -> Self {
        Self {
            start_time: Instant::now(),
            operation_timings: HashMap::new(),
            operation_stack: Vec::new(),
            memory_snapshots: Vec::new(),
            thresholds,
        }
    }

    /// Start timing an operation
    pub fn start_operation(&mut self, operation_name: &str) {
        debug!("Starting operation: {}", operation_name);
        self.operation_stack
            .push((operation_name.to_string(), Instant::now()));
    }

    /// End timing an operation
    pub fn end_operation(&mut self, operation_name: &str) {
        if let Some((name, start_time)) = self.operation_stack.pop() {
            if name == operation_name {
                let duration = start_time.elapsed();
                debug!("Completed operation '{}' in {:?}", operation_name, duration);

                self.operation_timings
                    .entry(operation_name.to_string())
                    .or_default()
                    .push(duration);
            } else {
                warn!(
                    "Operation stack mismatch: expected '{}', got '{}'",
                    name, operation_name
                );
            }
        } else {
            warn!("No operation to end for '{}'", operation_name);
        }
    }

    /// Record a memory snapshot
    pub fn record_memory_snapshot(&mut self, label: &str) {
        // In a real implementation, this would use a memory profiling library
        // For now, we'll simulate memory usage
        let estimated_memory = self.estimate_memory_usage();
        self.memory_snapshots
            .push((label.to_string(), estimated_memory));
        debug!("Memory snapshot '{}': {} bytes", label, estimated_memory);
    }

    /// Estimate current memory usage (simplified implementation)
    fn estimate_memory_usage(&self) -> usize {
        // This is a simplified estimation
        // In practice, you'd use a memory profiling library
        let base_size = std::mem::size_of::<RuleEngine>();
        let timing_size = self.operation_timings.len() * 64; // Rough estimate
        let snapshot_size = self.memory_snapshots.len() * 32;

        base_size + timing_size + snapshot_size
    }

    /// Profile a rule engine operation
    pub fn profile_operation<F, R>(&mut self, operation_name: &str, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.start_operation(operation_name);
        self.record_memory_snapshot(&format!("before_{operation_name}"));

        let result = operation();

        self.record_memory_snapshot(&format!("after_{operation_name}"));
        self.end_operation(operation_name);

        result
    }

    /// Generate a comprehensive performance report
    pub fn generate_report(&self) -> PerformanceMetrics {
        let total_time = self.start_time.elapsed();
        let mut warnings = Vec::new();

        // Calculate aggregate timings
        let rule_loading_time = self.get_total_time("rule_loading");
        let fact_processing_time = self.get_total_time("fact_processing");
        let forward_chaining_time = self.get_total_time("forward_chaining");
        let backward_chaining_time = self.get_total_time("backward_chaining");

        // Check thresholds and generate warnings
        if rule_loading_time.as_millis() > self.thresholds.max_rule_loading_time as u128 {
            warnings.push(format!(
                "Rule loading time ({:?}) exceeds threshold ({}ms)",
                rule_loading_time, self.thresholds.max_rule_loading_time
            ));
        }

        if forward_chaining_time.as_millis() > self.thresholds.max_forward_chaining_time as u128 {
            warnings.push(format!(
                "Forward chaining time ({:?}) exceeds threshold ({}ms)",
                forward_chaining_time, self.thresholds.max_forward_chaining_time
            ));
        }

        if backward_chaining_time.as_millis() > self.thresholds.max_backward_chaining_time as u128 {
            warnings.push(format!(
                "Backward chaining time ({:?}) exceeds threshold ({}ms)",
                backward_chaining_time, self.thresholds.max_backward_chaining_time
            ));
        }

        // Calculate memory statistics
        let peak_memory = self
            .memory_snapshots
            .iter()
            .map(|(_, size)| *size)
            .max()
            .unwrap_or(0);

        if peak_memory > self.thresholds.max_memory_usage * 1024 * 1024 {
            warnings.push(format!(
                "Peak memory usage ({} bytes) exceeds threshold ({} MB)",
                peak_memory, self.thresholds.max_memory_usage
            ));
        }

        // Build rule-specific timings
        let mut rule_timings = HashMap::new();
        for (operation, durations) in &self.operation_timings {
            let total: Duration = durations.iter().sum();
            rule_timings.insert(operation.clone(), total);
        }

        PerformanceMetrics {
            total_time,
            rule_loading_time,
            fact_processing_time,
            forward_chaining_time,
            backward_chaining_time,
            rules_processed: self.get_operation_count("rule_loading"),
            facts_processed: self.get_operation_count("fact_processing"),
            inferred_facts: self.get_operation_count("fact_inference"),
            memory_stats: MemoryStats {
                peak_memory_usage: peak_memory,
                facts_memory: peak_memory / 3, // Rough estimates
                rules_memory: peak_memory / 3,
                derived_facts_memory: peak_memory / 3,
            },
            rule_timings,
            warnings,
        }
    }

    /// Get total time for an operation
    fn get_total_time(&self, operation: &str) -> Duration {
        self.operation_timings
            .get(operation)
            .map(|durations| durations.iter().sum())
            .unwrap_or(Duration::ZERO)
    }

    /// Get operation count
    fn get_operation_count(&self, operation: &str) -> usize {
        self.operation_timings
            .get(operation)
            .map(|durations| durations.len())
            .unwrap_or(0)
    }

    /// Generate a text report
    pub fn print_report(&self) {
        let metrics = self.generate_report();

        println!("=== Rule Engine Performance Report ===");
        println!("Total execution time: {:?}", metrics.total_time);
        println!("Rule loading time: {:?}", metrics.rule_loading_time);
        println!("Fact processing time: {:?}", metrics.fact_processing_time);
        println!("Forward chaining time: {:?}", metrics.forward_chaining_time);
        println!(
            "Backward chaining time: {:?}",
            metrics.backward_chaining_time
        );
        println!("Rules processed: {}", metrics.rules_processed);
        println!("Facts processed: {}", metrics.facts_processed);
        println!("Inferred facts: {}", metrics.inferred_facts);
        println!(
            "Peak memory usage: {} bytes",
            metrics.memory_stats.peak_memory_usage
        );

        if !metrics.warnings.is_empty() {
            println!("\n=== Performance Warnings ===");
            for warning in &metrics.warnings {
                println!("⚠️  {warning}");
            }
        }

        if !metrics.rule_timings.is_empty() {
            println!("\n=== Operation Timings ===");
            let mut sorted_timings: Vec<_> = metrics.rule_timings.iter().collect();
            sorted_timings.sort_by_key(|(_, duration)| *duration);
            sorted_timings.reverse();

            for (operation, duration) in sorted_timings {
                println!("{operation}: {duration:?}");
            }
        }
    }

    /// Export metrics as JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        let metrics = self.generate_report();
        serde_json::to_string_pretty(&metrics)
    }
}

/// Performance test harness for rule engines
pub struct PerformanceTestHarness {
    profiler: RuleEngineProfiler,
}

impl PerformanceTestHarness {
    /// Create a new test harness
    pub fn new() -> Self {
        Self {
            profiler: RuleEngineProfiler::new(),
        }
    }

    /// Run a comprehensive performance test
    pub fn run_comprehensive_test(
        &mut self,
        rules: Vec<Rule>,
        facts: Vec<RuleAtom>,
    ) -> PerformanceMetrics {
        info!(
            "Starting comprehensive performance test with {} rules and {} facts",
            rules.len(),
            facts.len()
        );

        let mut engine = RuleEngine::new();

        // Test rule loading performance
        self.profiler.profile_operation("rule_loading", || {
            for rule in rules {
                engine.add_rule(rule);
            }
        });

        // Test fact processing performance
        self.profiler.profile_operation("fact_processing", || {
            engine.add_facts(facts.clone());
        });

        // Test forward chaining performance
        let derived_facts = self.profiler.profile_operation("forward_chaining", || {
            engine.forward_chain(&facts).unwrap_or_default()
        });

        info!("Forward chaining derived {} facts", derived_facts.len());

        // Test backward chaining performance (if we have a goal)
        if let Some(goal) = facts.first() {
            self.profiler.profile_operation("backward_chaining", || {
                engine.backward_chain(goal).unwrap_or(false)
            });
        }

        let metrics = self.profiler.generate_report();
        info!("Performance test completed in {:?}", metrics.total_time);

        metrics
    }

    /// Run a memory stress test
    pub fn run_memory_stress_test(&mut self, scale_factor: usize) -> PerformanceMetrics {
        info!(
            "Starting memory stress test with scale factor {}",
            scale_factor
        );

        let mut engine = RuleEngine::new();

        // Generate large number of facts and rules
        let facts = self.generate_large_fact_set(scale_factor * 1000);
        let rules = self.generate_large_rule_set(scale_factor * 100);

        self.profiler.record_memory_snapshot("initial");

        // Load rules in batches
        self.profiler.profile_operation("bulk_rule_loading", || {
            for rule in rules {
                engine.add_rule(rule);
            }
        });

        self.profiler.record_memory_snapshot("after_rules");

        // Load facts in batches
        self.profiler.profile_operation("bulk_fact_loading", || {
            engine.add_facts(facts.clone());
        });

        self.profiler.record_memory_snapshot("after_facts");

        // Perform reasoning
        self.profiler
            .profile_operation("memory_stress_reasoning", || {
                engine.forward_chain(&facts).unwrap_or_default()
            });

        self.profiler.record_memory_snapshot("after_reasoning");

        self.profiler.generate_report()
    }

    /// Generate a large set of test facts
    fn generate_large_fact_set(&self, size: usize) -> Vec<RuleAtom> {
        let mut facts = Vec::with_capacity(size);

        for i in 0..size {
            facts.push(RuleAtom::Triple {
                subject: Term::Constant(format!("http://example.org/entity{i}")),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant(format!("http://example.org/Type{val}", val = i % 100)),
            });
        }

        facts
    }

    /// Generate a large set of test rules
    fn generate_large_rule_set(&self, size: usize) -> Vec<Rule> {
        let mut rules = Vec::with_capacity(size);

        for i in 0..size {
            rules.push(Rule {
                name: format!("large_rule_{i}"),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                    ),
                    object: Term::Constant(format!("http://example.org/Type{val}", val = i % 100)),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                    ),
                    object: Term::Constant(format!("http://example.org/DerivedType{i}")),
                }],
            });
        }

        rules
    }
}

impl Default for PerformanceTestHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic_operations() {
        let mut profiler = RuleEngineProfiler::new();

        profiler.start_operation("test_op");
        std::thread::sleep(Duration::from_millis(10));
        profiler.end_operation("test_op");

        let metrics = profiler.generate_report();
        assert!(metrics.total_time >= Duration::from_millis(10));
    }

    #[test]
    fn test_performance_harness() {
        let mut harness = PerformanceTestHarness::new();

        let rules = vec![Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("test".to_string()),
                object: Term::Constant("value".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("derived".to_string()),
                object: Term::Constant("result".to_string()),
            }],
        }];

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("subject1".to_string()),
            predicate: Term::Constant("test".to_string()),
            object: Term::Constant("value".to_string()),
        }];

        let metrics = harness.run_comprehensive_test(rules, facts);
        assert!(metrics.rules_processed > 0);
    }

    #[test]
    fn test_incremental_reasoning() {
        let mut engine = IncrementalReasoningEngine::new();

        // Add a simple rule: Person -> Human
        let rule = Rule {
            name: "person_human".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Person".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Human".to_string()),
            }],
        };

        engine.add_rules(vec![rule]).unwrap();

        // Add initial facts
        let initial_facts = vec![RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("Person".to_string()),
        }];

        // Perform full reasoning first
        let full_results = engine.full_reasoning_with_cache(initial_facts).unwrap();
        assert!(!full_results.is_empty());

        // Add new facts incrementally
        let new_facts = vec![RuleAtom::Triple {
            subject: Term::Constant("bob".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("Person".to_string()),
        }];

        let incremental_results = engine.add_facts_incremental(new_facts).unwrap();
        assert!(!incremental_results.is_empty());

        // Check metrics
        let metrics = engine.get_incremental_metrics();
        assert_eq!(metrics.incremental_updates, 1);
    }

    #[test]
    fn test_hybrid_reasoning_engine() {
        let mut engine = HybridReasoningEngine::new(ReasoningStrategy::Adaptive {
            parallel_threshold: 5,
            complexity_threshold: 10,
        });

        // Add a simple rule
        let rule = Rule {
            name: "test_hybrid".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("input".to_string()),
                object: Term::Constant("value".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("output".to_string()),
                object: Term::Constant("result".to_string()),
            }],
        };

        engine.add_rules(vec![rule]).unwrap();

        // Test with small number of facts (should use incremental)
        let small_facts = vec![RuleAtom::Triple {
            subject: Term::Constant("item1".to_string()),
            predicate: Term::Constant("input".to_string()),
            object: Term::Constant("value".to_string()),
        }];

        let results = engine.reason(small_facts).unwrap();
        assert!(!results.is_empty());

        // Test with large number of facts (should use parallel)
        let large_facts: Vec<RuleAtom> = (0..10)
            .map(|i| RuleAtom::Triple {
                subject: Term::Constant(format!("item{}", i)),
                predicate: Term::Constant("input".to_string()),
                object: Term::Constant("value".to_string()),
            })
            .collect();

        let large_results = engine.reason(large_facts).unwrap();
        assert!(!large_results.is_empty());

        // Check that metrics were tracked
        let metrics = engine.get_performance_metrics();
        assert!(metrics.total_time > Duration::new(0, 0));
    }

    #[test]
    fn test_incremental_vs_full_benchmark() {
        let mut engine = IncrementalReasoningEngine::new();

        // Add rules
        let rules = vec![Rule {
            name: "rule1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("likes".to_string()),
                object: Term::Constant("food".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("FoodLover".to_string()),
            }],
        }];

        engine.add_rules(rules).unwrap();

        // Initial facts
        let initial_facts = vec![RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("likes".to_string()),
            object: Term::Constant("food".to_string()),
        }];

        // New facts to add incrementally
        let new_facts = vec![RuleAtom::Triple {
            subject: Term::Constant("bob".to_string()),
            predicate: Term::Constant("likes".to_string()),
            object: Term::Constant("food".to_string()),
        }];

        // Run benchmark
        let benchmark = engine
            .benchmark_incremental_vs_full(initial_facts, new_facts)
            .unwrap();

        // Check that benchmark completed successfully
        assert!(benchmark.full_reasoning_time > Duration::new(0, 0));
        assert!(benchmark.facts_derived > 0);

        println!("Benchmark results: {}", benchmark);
    }

    #[test]
    fn test_change_tracking() {
        let mut engine = IncrementalReasoningEngine::new();

        // Add a rule
        let rule = Rule {
            name: "change_test".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("input".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("output".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        engine.add_rules(vec![rule]).unwrap();

        // Add facts and verify change tracking
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("test".to_string()),
            predicate: Term::Constant("input".to_string()),
            object: Term::Constant("data".to_string()),
        }];

        let results = engine.add_facts_incremental(facts).unwrap();
        assert!(!results.is_empty());

        // Verify metrics were updated
        let metrics = engine.get_incremental_metrics();
        assert_eq!(metrics.incremental_updates, 1);
    }
}

/// Parallel rule evaluation engine for improved performance
pub struct ParallelRuleEngine {
    /// Number of worker threads
    num_threads: usize,
    /// Shared rule storage
    rules: Arc<Mutex<Vec<Rule>>>,
    /// Shared fact storage
    facts: Arc<Mutex<Vec<RuleAtom>>>,
    /// Performance metrics
    metrics: Arc<Mutex<PerformanceMetrics>>,
}

impl ParallelRuleEngine {
    /// Create a new parallel rule engine
    pub fn new(num_threads: Option<usize>) -> Self {
        let num_threads = num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

        info!(
            "Initializing parallel rule engine with {} threads",
            num_threads
        );

        Self {
            num_threads,
            rules: Arc::new(Mutex::new(Vec::new())),
            facts: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(PerformanceMetrics {
                total_time: Duration::new(0, 0),
                rule_loading_time: Duration::new(0, 0),
                fact_processing_time: Duration::new(0, 0),
                forward_chaining_time: Duration::new(0, 0),
                backward_chaining_time: Duration::new(0, 0),
                rules_processed: 0,
                facts_processed: 0,
                inferred_facts: 0,
                memory_stats: MemoryStats {
                    peak_memory_usage: 0,
                    facts_memory: 0,
                    rules_memory: 0,
                    derived_facts_memory: 0,
                },
                rule_timings: HashMap::new(),
                warnings: Vec::new(),
            })),
        }
    }

    /// Add rules to the parallel engine
    pub fn add_rules(&mut self, rules: Vec<Rule>) -> Result<(), String> {
        let start_time = Instant::now();

        match self.rules.lock() {
            Ok(mut rule_storage) => {
                rule_storage.extend(rules.clone());

                if let Ok(mut metrics) = self.metrics.lock() {
                    metrics.rules_processed += rules.len();
                    metrics.rule_loading_time += start_time.elapsed();
                }

                info!("Added {} rules to parallel engine", rules.len());
                Ok(())
            }
            _ => Err("Failed to acquire rule storage lock".to_string()),
        }
    }

    /// Execute parallel forward chaining
    pub fn parallel_forward_chain(&mut self) -> Result<Vec<RuleAtom>, String> {
        let start_time = Instant::now();
        info!(
            "Starting parallel forward chaining with {} threads",
            self.num_threads
        );

        // Clone shared data for workers
        let rules = self.rules.clone();
        let facts = self.facts.clone();
        let metrics = self.metrics.clone();

        let derived_facts = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();

        // Spawn worker threads
        for thread_id in 0..self.num_threads {
            let rules_clone = rules.clone();
            let facts_clone = facts.clone();
            let derived_facts_clone = derived_facts.clone();
            let metrics_clone = metrics.clone();

            let handle = thread::spawn(move || {
                Self::worker_forward_chain(
                    thread_id,
                    rules_clone,
                    facts_clone,
                    derived_facts_clone,
                    metrics_clone,
                );
            });

            handles.push(handle);
        }

        // Wait for all workers to complete
        for handle in handles {
            if let Err(e) = handle.join() {
                warn!("Worker thread panicked: {:?}", e);
            }
        }

        // Collect results
        let results = match derived_facts.lock() {
            Ok(derived) => derived.clone(),
            _ => {
                return Err("Failed to acquire derived facts lock".to_string());
            }
        };

        // Update metrics
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.forward_chaining_time += start_time.elapsed();
            metrics.inferred_facts += results.len();
        }

        info!(
            "Parallel forward chaining completed, derived {} facts",
            results.len()
        );
        Ok(results)
    }

    /// Worker function for parallel forward chaining
    fn worker_forward_chain(
        thread_id: usize,
        rules: Arc<Mutex<Vec<Rule>>>,
        facts: Arc<Mutex<Vec<RuleAtom>>>,
        derived_facts: Arc<Mutex<Vec<RuleAtom>>>,
        _metrics: Arc<Mutex<PerformanceMetrics>>,
    ) {
        debug!("Worker thread {} starting forward chaining", thread_id);

        // Create local rule engine for this worker
        let mut local_engine = RuleEngine::new();

        // Get a snapshot of rules and facts
        let (local_rules, local_facts) = {
            let rules_guard = rules.lock().unwrap();
            let facts_guard = facts.lock().unwrap();
            (rules_guard.clone(), facts_guard.clone())
        };

        // Add rules to local engine
        for rule in local_rules {
            local_engine.add_rule(rule);
        }

        // Process facts in chunks for this worker
        let chunk_size = (local_facts.len() + 3) / 4; // Divide facts among threads
        let start_idx = thread_id * chunk_size;
        let end_idx = std::cmp::min(start_idx + chunk_size, local_facts.len());

        if start_idx < local_facts.len() {
            let worker_facts = &local_facts[start_idx..end_idx];

            match local_engine.forward_chain(worker_facts) {
                Ok(new_facts) => {
                    if let Ok(mut derived) = derived_facts.lock() {
                        derived.extend(new_facts);
                    }
                    debug!(
                        "Worker thread {} processed {} facts",
                        thread_id,
                        worker_facts.len()
                    );
                }
                Err(e) => {
                    warn!("Worker thread {} failed: {}", thread_id, e);
                }
            }
        }
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        match self.metrics.lock() {
            Ok(metrics) => metrics.clone(),
            _ => {
                warn!("Failed to acquire metrics lock");
                PerformanceMetrics {
                    total_time: Duration::new(0, 0),
                    rule_loading_time: Duration::new(0, 0),
                    fact_processing_time: Duration::new(0, 0),
                    forward_chaining_time: Duration::new(0, 0),
                    backward_chaining_time: Duration::new(0, 0),
                    rules_processed: 0,
                    facts_processed: 0,
                    inferred_facts: 0,
                    memory_stats: MemoryStats {
                        peak_memory_usage: 0,
                        facts_memory: 0,
                        rules_memory: 0,
                        derived_facts_memory: 0,
                    },
                    rule_timings: HashMap::new(),
                    warnings: Vec::new(),
                }
            }
        }
    }
}

impl Default for ParallelRuleEngine {
    fn default() -> Self {
        Self::new(None)
    }
}

/// Incremental reasoning engine that efficiently updates conclusions when new facts are added
pub struct IncrementalReasoningEngine {
    /// Base rule engine
    base_engine: Arc<Mutex<RuleEngine>>,
    /// Fact dependency graph for incremental updates
    fact_dependencies: Arc<Mutex<HashMap<RuleAtom, Vec<RuleAtom>>>>,
    /// Rule activation tracking
    rule_activations: Arc<Mutex<HashMap<String, Vec<RuleAtom>>>>,
    /// Materialized conclusions cache
    materialized_facts: Arc<Mutex<Vec<RuleAtom>>>,
    /// Change tracking for efficient updates
    change_tracker: Arc<Mutex<ChangeTracker>>,
    /// Performance metrics for incremental operations
    incremental_metrics: Arc<Mutex<IncrementalMetrics>>,
}

/// Tracks changes for incremental reasoning
#[derive(Debug, Clone, Default)]
pub struct ChangeTracker {
    /// Newly added facts since last reasoning
    added_facts: Vec<RuleAtom>,
    /// Facts removed since last reasoning
    #[allow(dead_code)]
    removed_facts: Vec<RuleAtom>,
    /// Rules affected by changes
    #[allow(dead_code)]
    affected_rules: Vec<String>,
    /// Last reasoning timestamp
    #[allow(dead_code)]
    last_reasoning_time: Option<std::time::Instant>,
}

/// Metrics for incremental reasoning operations
#[derive(Debug, Clone, Default)]
pub struct IncrementalMetrics {
    /// Number of incremental updates performed
    pub incremental_updates: usize,
    /// Time saved by incremental reasoning vs full reasoning
    pub time_saved: Duration,
    /// Number of facts reused from cache
    pub facts_reused: usize,
    /// Number of rules that didn't need re-evaluation
    pub rules_skipped: usize,
    /// Average incremental update time
    pub avg_update_time: Duration,
}

impl IncrementalReasoningEngine {
    /// Create a new incremental reasoning engine
    pub fn new() -> Self {
        Self {
            base_engine: Arc::new(Mutex::new(RuleEngine::new())),
            fact_dependencies: Arc::new(Mutex::new(HashMap::new())),
            rule_activations: Arc::new(Mutex::new(HashMap::new())),
            materialized_facts: Arc::new(Mutex::new(Vec::new())),
            change_tracker: Arc::new(Mutex::new(ChangeTracker::default())),
            incremental_metrics: Arc::new(Mutex::new(IncrementalMetrics::default())),
        }
    }

    /// Add rules to the incremental engine
    pub fn add_rules(&mut self, rules: Vec<Rule>) -> Result<(), String> {
        info!(
            "Adding {} rules to incremental reasoning engine",
            rules.len()
        );

        match self.base_engine.lock() {
            Ok(mut engine) => {
                for rule in rules {
                    engine.add_rule(rule);
                }
                Ok(())
            }
            _ => Err("Failed to acquire engine lock".to_string()),
        }
    }

    /// Add new facts and perform incremental reasoning
    pub fn add_facts_incremental(
        &mut self,
        new_facts: Vec<RuleAtom>,
    ) -> Result<Vec<RuleAtom>, String> {
        let start_time = Instant::now();
        info!(
            "Starting incremental reasoning with {} new facts",
            new_facts.len()
        );

        // Update change tracker
        if let Ok(mut tracker) = self.change_tracker.lock() {
            tracker.added_facts.extend(new_facts.clone());
        }

        // Identify which rules are affected by the new facts
        let affected_rules = self.identify_affected_rules(&new_facts)?;
        info!("Identified {} affected rules", affected_rules.len());

        // Perform incremental reasoning only on affected rules
        let new_derived_facts = self.reason_incrementally(new_facts, affected_rules)?;

        // Update materialized facts cache
        if let Ok(mut materialized) = self.materialized_facts.lock() {
            materialized.extend(new_derived_facts.clone());
        }

        // Update metrics
        if let Ok(mut metrics) = self.incremental_metrics.lock() {
            metrics.incremental_updates += 1;
            metrics.avg_update_time = if metrics.incremental_updates == 1 {
                start_time.elapsed()
            } else {
                Duration::from_nanos(
                    (metrics.avg_update_time.as_nanos() as u64
                        + start_time.elapsed().as_nanos() as u64)
                        / 2,
                )
            };
        }

        info!(
            "Incremental reasoning completed, derived {} new facts in {:?}",
            new_derived_facts.len(),
            start_time.elapsed()
        );

        Ok(new_derived_facts)
    }

    /// Identify rules that could be affected by new facts
    fn identify_affected_rules(&self, new_facts: &[RuleAtom]) -> Result<Vec<String>, String> {
        let mut affected_rules = Vec::new();

        if let Ok(engine) = self.base_engine.lock() {
            for rule in &engine.rules {
                for new_fact in new_facts {
                    if self.rule_could_match_fact(rule, new_fact) {
                        affected_rules.push(rule.name.clone());
                        break; // Rule is affected, no need to check other facts
                    }
                }
            }
        }

        Ok(affected_rules)
    }

    /// Check if a rule could potentially match a fact
    fn rule_could_match_fact(&self, rule: &Rule, fact: &RuleAtom) -> bool {
        // Check if the fact could unify with any atom in the rule body
        for body_atom in &rule.body {
            if self.atoms_could_unify(body_atom, fact) {
                return true;
            }
        }
        false
    }

    /// Check if two atoms could potentially unify
    fn atoms_could_unify(&self, atom1: &RuleAtom, atom2: &RuleAtom) -> bool {
        match (atom1, atom2) {
            (
                RuleAtom::Triple {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
                RuleAtom::Triple {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                self.terms_could_unify(s1, s2)
                    && self.terms_could_unify(p1, p2)
                    && self.terms_could_unify(o1, o2)
            }
            (
                RuleAtom::Builtin { name: n1, args: a1 },
                RuleAtom::Builtin { name: n2, args: a2 },
            ) => {
                n1 == n2
                    && a1.len() == a2.len()
                    && a1
                        .iter()
                        .zip(a2.iter())
                        .all(|(t1, t2)| self.terms_could_unify(t1, t2))
            }
            _ => false,
        }
    }

    /// Check if two terms could potentially unify
    fn terms_could_unify(&self, term1: &Term, term2: &Term) -> bool {
        match (term1, term2) {
            (Term::Variable(_), _) | (_, Term::Variable(_)) => true,
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            _ => false,
        }
    }

    /// Perform incremental reasoning on affected rules only
    fn reason_incrementally(
        &mut self,
        new_facts: Vec<RuleAtom>,
        affected_rules: Vec<String>,
    ) -> Result<Vec<RuleAtom>, String> {
        let mut derived_facts = Vec::new();

        if let Ok(mut engine) = self.base_engine.lock() {
            // Add new facts to the engine
            engine.add_facts(new_facts.clone());

            // Only apply affected rules for efficiency
            for rule_name in &affected_rules {
                if let Some(rule) = engine.rules.iter().find(|r| r.name == *rule_name) {
                    // Create a temporary engine with just this rule
                    let mut temp_engine = RuleEngine::new();
                    temp_engine.add_rule(rule.clone());
                    temp_engine.add_facts(engine.get_facts());

                    match temp_engine.forward_chain(&[]) {
                        Ok(rule_derived) => {
                            // Filter out facts we already knew
                            let new_derived: Vec<RuleAtom> = rule_derived
                                .into_iter()
                                .filter(|fact| !self.fact_already_known(fact))
                                .collect();

                            derived_facts.extend(new_derived);
                        }
                        Err(e) => {
                            warn!("Failed to apply rule '{}': {}", rule_name, e);
                        }
                    }
                }
            }

            // Update metrics
            if let Ok(mut metrics) = self.incremental_metrics.lock() {
                let total_rules = engine.rules.len();
                metrics.rules_skipped += total_rules.saturating_sub(affected_rules.len());
            }
        }

        Ok(derived_facts)
    }

    /// Check if a fact is already known (to avoid duplicates)
    fn fact_already_known(&self, fact: &RuleAtom) -> bool {
        match self.materialized_facts.lock() {
            Ok(materialized) => materialized.contains(fact),
            _ => {
                false // If we can't check, assume it's new
            }
        }
    }

    /// Get current incremental reasoning metrics
    pub fn get_incremental_metrics(&self) -> IncrementalMetrics {
        match self.incremental_metrics.lock() {
            Ok(metrics) => metrics.clone(),
            _ => {
                warn!("Failed to acquire metrics lock");
                IncrementalMetrics::default()
            }
        }
    }

    /// Reset the incremental reasoning state
    pub fn reset(&mut self) {
        info!("Resetting incremental reasoning engine");

        if let Ok(mut materialized) = self.materialized_facts.lock() {
            materialized.clear();
        }

        if let Ok(mut dependencies) = self.fact_dependencies.lock() {
            dependencies.clear();
        }

        if let Ok(mut activations) = self.rule_activations.lock() {
            activations.clear();
        }

        if let Ok(mut tracker) = self.change_tracker.lock() {
            *tracker = ChangeTracker::default();
        }

        if let Ok(mut metrics) = self.incremental_metrics.lock() {
            *metrics = IncrementalMetrics::default();
        }
    }

    /// Perform a full reasoning pass and cache results
    pub fn full_reasoning_with_cache(
        &mut self,
        facts: Vec<RuleAtom>,
    ) -> Result<Vec<RuleAtom>, String> {
        let start_time = Instant::now();
        info!(
            "Performing full reasoning with caching for {} facts",
            facts.len()
        );

        match self.base_engine.lock() {
            Ok(mut engine) => {
                engine.clear();
                engine.add_facts(facts);

                match engine.forward_chain(&[]) {
                    Ok(derived_facts) => {
                        // Cache all derived facts
                        if let Ok(mut materialized) = self.materialized_facts.lock() {
                            materialized.clear();
                            materialized.extend(derived_facts.clone());
                        }

                        // Build dependency graph for future incremental updates
                        self.build_dependency_graph(&derived_facts)?;

                        info!(
                            "Full reasoning completed in {:?}, cached {} facts",
                            start_time.elapsed(),
                            derived_facts.len()
                        );

                        Ok(derived_facts)
                    }
                    Err(e) => Err(format!("Full reasoning failed: {e}"))
                }
            }
            _ => Err("Failed to acquire engine lock".to_string()),
        }
    }

    /// Build dependency graph for efficient incremental updates
    fn build_dependency_graph(&self, derived_facts: &[RuleAtom]) -> Result<(), String> {
        if let Ok(mut dependencies) = self.fact_dependencies.lock() {
            dependencies.clear();

            // For now, use a simple dependency model
            // In a full implementation, this would track which facts depend on which rules
            for fact in derived_facts {
                dependencies.insert(fact.clone(), vec![]);
            }
        }

        Ok(())
    }

    /// Compare incremental vs full reasoning performance
    pub fn benchmark_incremental_vs_full(
        &mut self,
        initial_facts: Vec<RuleAtom>,
        new_facts: Vec<RuleAtom>,
    ) -> Result<BenchmarkResults, String> {
        info!("Benchmarking incremental vs full reasoning");

        // First, establish baseline with full reasoning
        let full_start = Instant::now();
        let mut all_facts = initial_facts.clone();
        all_facts.extend(new_facts.clone());
        self.full_reasoning_with_cache(all_facts)?;
        let full_time = full_start.elapsed();

        // Reset and setup for incremental test
        self.reset();
        self.full_reasoning_with_cache(initial_facts)?;

        // Now test incremental reasoning
        let incremental_start = Instant::now();
        let incremental_results = self.add_facts_incremental(new_facts)?;
        let incremental_time = incremental_start.elapsed();

        let speedup = if incremental_time.as_nanos() > 0 {
            full_time.as_nanos() as f64 / incremental_time.as_nanos() as f64
        } else {
            f64::INFINITY
        };

        let results = BenchmarkResults {
            full_reasoning_time: full_time,
            incremental_reasoning_time: incremental_time,
            speedup_factor: speedup,
            facts_derived: incremental_results.len(),
        };

        info!("Benchmark results: {:?}", results);
        Ok(results)
    }
}

impl Default for IncrementalReasoningEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Results from benchmarking incremental vs full reasoning
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub full_reasoning_time: Duration,
    pub incremental_reasoning_time: Duration,
    pub speedup_factor: f64,
    pub facts_derived: usize,
}

impl std::fmt::Display for BenchmarkResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Benchmark: Full {:?} vs Incremental {:?} = {:.2}x speedup ({} facts)",
            self.full_reasoning_time,
            self.incremental_reasoning_time,
            self.speedup_factor,
            self.facts_derived
        )
    }
}

impl std::fmt::Display for IncrementalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Incremental: {} updates, avg {:?}, {} facts reused, {} rules skipped",
            self.incremental_updates, self.avg_update_time, self.facts_reused, self.rules_skipped
        )
    }
}

/// Hybrid reasoning engine that combines parallel and incremental approaches
pub struct HybridReasoningEngine {
    parallel_engine: ParallelRuleEngine,
    incremental_engine: IncrementalReasoningEngine,
    strategy: ReasoningStrategy,
    performance_monitor: Arc<Mutex<PerformanceMetrics>>,
}

/// Strategy for choosing between parallel and incremental reasoning
#[derive(Debug, Clone)]
pub enum ReasoningStrategy {
    /// Always use parallel reasoning
    AlwaysParallel,
    /// Always use incremental reasoning
    AlwaysIncremental,
    /// Automatically choose based on workload characteristics
    Adaptive {
        /// Threshold for number of new facts to trigger parallel reasoning
        parallel_threshold: usize,
        /// Threshold for rule complexity to trigger parallel reasoning
        complexity_threshold: usize,
    },
}

impl HybridReasoningEngine {
    /// Create a new hybrid reasoning engine
    pub fn new(strategy: ReasoningStrategy) -> Self {
        Self {
            parallel_engine: ParallelRuleEngine::new(None),
            incremental_engine: IncrementalReasoningEngine::new(),
            strategy,
            performance_monitor: Arc::new(Mutex::new(PerformanceMetrics {
                total_time: Duration::new(0, 0),
                rule_loading_time: Duration::new(0, 0),
                fact_processing_time: Duration::new(0, 0),
                forward_chaining_time: Duration::new(0, 0),
                backward_chaining_time: Duration::new(0, 0),
                rules_processed: 0,
                facts_processed: 0,
                inferred_facts: 0,
                memory_stats: MemoryStats {
                    peak_memory_usage: 0,
                    facts_memory: 0,
                    rules_memory: 0,
                    derived_facts_memory: 0,
                },
                rule_timings: HashMap::new(),
                warnings: Vec::new(),
            })),
        }
    }

    /// Add rules to both engines
    pub fn add_rules(&mut self, rules: Vec<Rule>) -> Result<(), String> {
        self.parallel_engine.add_rules(rules.clone())?;
        self.incremental_engine.add_rules(rules)?;
        Ok(())
    }

    /// Reason using the configured strategy
    pub fn reason(&mut self, new_facts: Vec<RuleAtom>) -> Result<Vec<RuleAtom>, String> {
        let start_time = Instant::now();

        let strategy_choice = match &self.strategy {
            ReasoningStrategy::AlwaysParallel => "parallel",
            ReasoningStrategy::AlwaysIncremental => "incremental",
            ReasoningStrategy::Adaptive {
                parallel_threshold,
                complexity_threshold,
            } => {
                if new_facts.len() > *parallel_threshold
                    || self.estimate_rule_complexity()? > *complexity_threshold
                {
                    "parallel"
                } else {
                    "incremental"
                }
            }
        };

        info!(
            "Using {} reasoning strategy for {} facts",
            strategy_choice,
            new_facts.len()
        );

        let results = match strategy_choice {
            "parallel" => {
                // Add facts to parallel engine's fact store before reasoning
                if let Ok(mut facts) = self.parallel_engine.facts.lock() {
                    facts.extend(new_facts);
                }
                self.parallel_engine.parallel_forward_chain()
            }
            "incremental" => self.incremental_engine.add_facts_incremental(new_facts),
            _ => unreachable!(),
        };

        // Update performance metrics
        if let Ok(mut metrics) = self.performance_monitor.lock() {
            metrics.total_time += start_time.elapsed();
            if let Ok(ref result_facts) = results {
                metrics.inferred_facts += result_facts.len();
            }
        }

        results
    }

    /// Estimate the complexity of current rule set
    fn estimate_rule_complexity(&self) -> Result<usize, String> {
        // Simplified complexity estimation based on rule structure
        // In practice, this would be more sophisticated
        Ok(100) // Placeholder
    }

    /// Get combined performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        match self.performance_monitor.lock() {
            Ok(metrics) => metrics.clone(),
            _ => {
                warn!("Failed to acquire performance monitor lock");
                PerformanceMetrics {
                    total_time: Duration::new(0, 0),
                    rule_loading_time: Duration::new(0, 0),
                    fact_processing_time: Duration::new(0, 0),
                    forward_chaining_time: Duration::new(0, 0),
                    backward_chaining_time: Duration::new(0, 0),
                    rules_processed: 0,
                    facts_processed: 0,
                    inferred_facts: 0,
                    memory_stats: MemoryStats {
                        peak_memory_usage: 0,
                        facts_memory: 0,
                        rules_memory: 0,
                        derived_facts_memory: 0,
                    },
                    rule_timings: HashMap::new(),
                    warnings: Vec::new(),
                }
            }
        }
    }

    /// Get detailed breakdown of strategy usage
    pub fn get_strategy_breakdown(&self) -> String {
        format!("Strategy: {:?}", self.strategy)
    }
}

impl Default for HybridReasoningEngine {
    fn default() -> Self {
        Self::new(ReasoningStrategy::Adaptive {
            parallel_threshold: 1000,
            complexity_threshold: 50,
        })
    }
}
