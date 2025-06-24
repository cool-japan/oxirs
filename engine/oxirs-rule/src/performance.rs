//! Performance analysis and profiling utilities for the rule engine
//!
//! This module provides tools for analyzing rule engine performance,
//! identifying bottlenecks, and generating performance reports.

use crate::{RuleEngine, Rule, RuleAtom, Term};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, debug, warn};

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
        self.operation_stack.push((operation_name.to_string(), Instant::now()));
    }
    
    /// End timing an operation
    pub fn end_operation(&mut self, operation_name: &str) {
        if let Some((name, start_time)) = self.operation_stack.pop() {
            if name == operation_name {
                let duration = start_time.elapsed();
                debug!("Completed operation '{}' in {:?}", operation_name, duration);
                
                self.operation_timings
                    .entry(operation_name.to_string())
                    .or_insert_with(Vec::new)
                    .push(duration);
            } else {
                warn!("Operation stack mismatch: expected '{}', got '{}'", name, operation_name);
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
        self.memory_snapshots.push((label.to_string(), estimated_memory));
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
        self.record_memory_snapshot(&format!("before_{}", operation_name));
        
        let result = operation();
        
        self.record_memory_snapshot(&format!("after_{}", operation_name));
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
        let peak_memory = self.memory_snapshots.iter()
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
                facts_memory: peak_memory / 3,      // Rough estimates
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
        println!("Backward chaining time: {:?}", metrics.backward_chaining_time);
        println!("Rules processed: {}", metrics.rules_processed);
        println!("Facts processed: {}", metrics.facts_processed);
        println!("Inferred facts: {}", metrics.inferred_facts);
        println!("Peak memory usage: {} bytes", metrics.memory_stats.peak_memory_usage);
        
        if !metrics.warnings.is_empty() {
            println!("\n=== Performance Warnings ===");
            for warning in &metrics.warnings {
                println!("⚠️  {}", warning);
            }
        }
        
        if !metrics.rule_timings.is_empty() {
            println!("\n=== Operation Timings ===");
            let mut sorted_timings: Vec<_> = metrics.rule_timings.iter().collect();
            sorted_timings.sort_by_key(|(_, duration)| *duration);
            sorted_timings.reverse();
            
            for (operation, duration) in sorted_timings {
                println!("{}: {:?}", operation, duration);
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
    pub fn run_comprehensive_test(&mut self, 
                                 rules: Vec<Rule>, 
                                 facts: Vec<RuleAtom>) -> PerformanceMetrics {
        info!("Starting comprehensive performance test with {} rules and {} facts", 
              rules.len(), facts.len());
        
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
    pub fn run_memory_stress_test(&mut self, 
                                 scale_factor: usize) -> PerformanceMetrics {
        info!("Starting memory stress test with scale factor {}", scale_factor);
        
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
        self.profiler.profile_operation("memory_stress_reasoning", || {
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
                subject: Term::Constant(format!("http://example.org/entity{}", i)),
                predicate: Term::Constant("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
                object: Term::Constant(format!("http://example.org/Type{}", i % 100)),
            });
        }
        
        facts
    }
    
    /// Generate a large set of test rules
    fn generate_large_rule_set(&self, size: usize) -> Vec<Rule> {
        let mut rules = Vec::with_capacity(size);
        
        for i in 0..size {
            rules.push(Rule {
                name: format!("large_rule_{}", i),
                body: vec![
                    RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
                        object: Term::Constant(format!("http://example.org/Type{}", i % 100)),
                    },
                ],
                head: vec![
                    RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
                        object: Term::Constant(format!("http://example.org/DerivedType{}", i)),
                    },
                ],
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
        
        let rules = vec![
            Rule {
                name: "test_rule".to_string(),
                body: vec![
                    RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("test".to_string()),
                        object: Term::Constant("value".to_string()),
                    },
                ],
                head: vec![
                    RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("derived".to_string()),
                        object: Term::Constant("result".to_string()),
                    },
                ],
            },
        ];
        
        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("subject1".to_string()),
                predicate: Term::Constant("test".to_string()),
                object: Term::Constant("value".to_string()),
            },
        ];
        
        let metrics = harness.run_comprehensive_test(rules, facts);
        assert!(metrics.rules_processed > 0);
    }
}