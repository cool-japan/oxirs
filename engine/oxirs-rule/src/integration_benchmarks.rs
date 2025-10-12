//! Integration Performance Benchmarks
//!
//! Comprehensive benchmarks for measuring and tracking performance of:
//! - SPARQL Integration (query execution modes)
//! - SHACL Integration (validation workflows)
//! - Distributed Reasoning (scaling characteristics)
//! - Rule Composition (module management overhead)
//!
//! # Running Benchmarks
//!
//! ```rust
//! use oxirs_rule::integration_benchmarks::IntegrationBenchmarkSuite;
//!
//! let mut suite = IntegrationBenchmarkSuite::new();
//! let results = suite.run_all_benchmarks().unwrap();
//! println!("{:?}", results);
//! ```

use crate::{
    composition::{CompositionManager, RuleModule},
    distributed::{DistributedReasoner, Node, PartitionStrategy},
    shacl_integration::{ShaclRuleIntegration, ShapeConstraint, ValidationMode},
    sparql_integration::{QueryMode, QueryPattern, SparqlRuleIntegration},
    Rule, RuleAtom, RuleEngine, Term,
};
use anyhow::Result;
use std::time::{Duration, Instant};

/// Benchmark result for a single test
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Number of operations performed
    pub operations: usize,
    /// Total execution time
    pub duration: Duration,
    /// Operations per second
    pub ops_per_sec: f64,
    /// Average time per operation (microseconds)
    pub avg_time_us: f64,
}

impl BenchmarkResult {
    pub fn new(name: String, operations: usize, duration: Duration) -> Self {
        let secs = duration.as_secs_f64();
        let ops_per_sec = if secs > 0.0 {
            operations as f64 / secs
        } else {
            0.0
        };
        let avg_time_us = if operations > 0 {
            duration.as_micros() as f64 / operations as f64
        } else {
            0.0
        };

        Self {
            name,
            operations,
            duration,
            ops_per_sec,
            avg_time_us,
        }
    }
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Benchmark: {}", self.name)?;
        writeln!(f, "  Operations: {}", self.operations)?;
        writeln!(f, "  Duration: {:?}", self.duration)?;
        writeln!(f, "  Throughput: {:.2} ops/sec", self.ops_per_sec)?;
        writeln!(f, "  Avg Time: {:.2} μs/op", self.avg_time_us)?;
        Ok(())
    }
}

/// Complete benchmark suite for integration features
pub struct IntegrationBenchmarkSuite {
    /// Results from all benchmarks
    results: Vec<BenchmarkResult>,
}

impl IntegrationBenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Run all benchmarks
    pub fn run_all_benchmarks(&mut self) -> Result<&[BenchmarkResult]> {
        println!("Running Integration Performance Benchmarks...\n");

        // SPARQL Integration Benchmarks
        self.benchmark_sparql_direct_query(1000)?;
        self.benchmark_sparql_forward_reasoning(100)?;
        self.benchmark_sparql_backward_reasoning(100)?;
        self.benchmark_sparql_hybrid_reasoning(100)?;

        // SHACL Integration Benchmarks
        self.benchmark_shacl_direct_validation(1000)?;
        self.benchmark_shacl_pre_reasoning(100)?;
        self.benchmark_shacl_post_reasoning(100)?;
        self.benchmark_shacl_full_validation(100)?;

        // Distributed Reasoning Benchmarks
        self.benchmark_distributed_round_robin(100)?;
        self.benchmark_distributed_load_balanced(100)?;
        self.benchmark_distributed_scaling()?;

        // Rule Composition Benchmarks
        self.benchmark_module_registration(1000)?;
        self.benchmark_template_instantiation(1000)?;

        Ok(&self.results)
    }

    /// Benchmark SPARQL direct query (no reasoning)
    fn benchmark_sparql_direct_query(&mut self, iterations: usize) -> Result<()> {
        let mut engine = RuleEngine::new();

        // Add test facts
        for i in 0..100 {
            engine.add_fact(RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant(format!("value_{}", i)),
            });
        }

        let mut sparql = SparqlRuleIntegration::new(engine);
        sparql.set_mode(QueryMode::Direct);

        let pattern = QueryPattern::new(None, Some("hasProperty".to_string()), None);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = sparql.query_with_reasoning(&[pattern.clone()])?;
        }
        let duration = start.elapsed();

        self.results.push(BenchmarkResult::new(
            "SPARQL Direct Query".to_string(),
            iterations,
            duration,
        ));

        Ok(())
    }

    /// Benchmark SPARQL forward reasoning
    fn benchmark_sparql_forward_reasoning(&mut self, iterations: usize) -> Result<()> {
        let mut engine = RuleEngine::new();

        // Add rule
        engine.add_rule(Rule {
            name: "inference_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("derived".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        });

        // Add facts
        for i in 0..50 {
            engine.add_fact(RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant(format!("value_{}", i)),
            });
        }

        let mut sparql = SparqlRuleIntegration::new(engine);
        sparql.set_mode(QueryMode::ForwardReasoning);

        let pattern = QueryPattern::new(None, Some("derived".to_string()), None);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = sparql.query_with_reasoning(&[pattern.clone()])?;
        }
        let duration = start.elapsed();

        self.results.push(BenchmarkResult::new(
            "SPARQL Forward Reasoning".to_string(),
            iterations,
            duration,
        ));

        Ok(())
    }

    /// Benchmark SPARQL backward reasoning
    fn benchmark_sparql_backward_reasoning(&mut self, iterations: usize) -> Result<()> {
        let mut engine = RuleEngine::new();

        engine.add_rule(Rule {
            name: "inference_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("derived".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        });

        for i in 0..50 {
            engine.add_fact(RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant(format!("value_{}", i)),
            });
        }

        let mut sparql = SparqlRuleIntegration::new(engine);
        sparql.set_mode(QueryMode::BackwardReasoning);

        let pattern = QueryPattern::new(
            Some("entity_0".to_string()),
            Some("derived".to_string()),
            Some("value_0".to_string()),
        );

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = sparql.query_with_reasoning(&[pattern.clone()])?;
        }
        let duration = start.elapsed();

        self.results.push(BenchmarkResult::new(
            "SPARQL Backward Reasoning".to_string(),
            iterations,
            duration,
        ));

        Ok(())
    }

    /// Benchmark SPARQL hybrid reasoning
    fn benchmark_sparql_hybrid_reasoning(&mut self, iterations: usize) -> Result<()> {
        let mut engine = RuleEngine::new();

        engine.add_rule(Rule {
            name: "inference_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("derived".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        });

        for i in 0..50 {
            engine.add_fact(RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant(format!("value_{}", i)),
            });
        }

        let mut sparql = SparqlRuleIntegration::new(engine);
        sparql.set_mode(QueryMode::Hybrid);

        let pattern = QueryPattern::new(None, Some("derived".to_string()), None);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = sparql.query_with_reasoning(&[pattern.clone()])?;
        }
        let duration = start.elapsed();

        self.results.push(BenchmarkResult::new(
            "SPARQL Hybrid Reasoning".to_string(),
            iterations,
            duration,
        ));

        Ok(())
    }

    /// Benchmark SHACL direct validation
    fn benchmark_shacl_direct_validation(&mut self, iterations: usize) -> Result<()> {
        let engine = RuleEngine::new();
        let mut shacl = ShaclRuleIntegration::new(engine);
        shacl.set_mode(ValidationMode::Direct);

        let constraint =
            ShapeConstraint::new("test_constraint".to_string(), "sh:minCount".to_string());

        let facts: Vec<RuleAtom> = (0..100)
            .map(|i| RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant(format!("value_{}", i)),
            })
            .collect();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = shacl.validate_with_reasoning(&[constraint.clone()], &facts)?;
        }
        let duration = start.elapsed();

        self.results.push(BenchmarkResult::new(
            "SHACL Direct Validation".to_string(),
            iterations,
            duration,
        ));

        Ok(())
    }

    /// Benchmark SHACL pre-reasoning validation
    fn benchmark_shacl_pre_reasoning(&mut self, iterations: usize) -> Result<()> {
        let engine = RuleEngine::new();
        let mut shacl = ShaclRuleIntegration::new(engine);
        shacl.set_mode(ValidationMode::PreReasoning);

        let constraint =
            ShapeConstraint::new("test_constraint".to_string(), "sh:minCount".to_string());

        let facts: Vec<RuleAtom> = (0..50)
            .map(|i| RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant(format!("value_{}", i)),
            })
            .collect();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = shacl.validate_with_reasoning(&[constraint.clone()], &facts)?;
        }
        let duration = start.elapsed();

        self.results.push(BenchmarkResult::new(
            "SHACL Pre-Reasoning Validation".to_string(),
            iterations,
            duration,
        ));

        Ok(())
    }

    /// Benchmark SHACL post-reasoning validation
    fn benchmark_shacl_post_reasoning(&mut self, iterations: usize) -> Result<()> {
        let engine = RuleEngine::new();
        let mut shacl = ShaclRuleIntegration::new(engine);
        shacl.set_mode(ValidationMode::PostReasoning);

        let constraint =
            ShapeConstraint::new("test_constraint".to_string(), "sh:minCount".to_string());

        let facts: Vec<RuleAtom> = (0..50)
            .map(|i| RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant(format!("value_{}", i)),
            })
            .collect();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = shacl.validate_with_reasoning(&[constraint.clone()], &facts)?;
        }
        let duration = start.elapsed();

        self.results.push(BenchmarkResult::new(
            "SHACL Post-Reasoning Validation".to_string(),
            iterations,
            duration,
        ));

        Ok(())
    }

    /// Benchmark SHACL full validation (pre + post reasoning)
    fn benchmark_shacl_full_validation(&mut self, iterations: usize) -> Result<()> {
        let engine = RuleEngine::new();
        let mut shacl = ShaclRuleIntegration::new(engine);
        shacl.set_mode(ValidationMode::Full);

        let constraint =
            ShapeConstraint::new("test_constraint".to_string(), "sh:minCount".to_string());

        let facts: Vec<RuleAtom> = (0..50)
            .map(|i| RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant(format!("value_{}", i)),
            })
            .collect();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = shacl.validate_with_reasoning(&[constraint.clone()], &facts)?;
        }
        let duration = start.elapsed();

        self.results.push(BenchmarkResult::new(
            "SHACL Full Validation".to_string(),
            iterations,
            duration,
        ));

        Ok(())
    }

    /// Benchmark distributed reasoning with round-robin
    fn benchmark_distributed_round_robin(&mut self, iterations: usize) -> Result<()> {
        let mut reasoner = DistributedReasoner::new(PartitionStrategy::RoundRobin);

        // Register nodes
        for i in 1..=4 {
            reasoner.register_node(Node::new(
                format!("node_{}", i),
                format!("localhost:800{}", i),
            ))?;
        }

        let rules = vec![Rule {
            name: "test_rule".to_string(),
            body: vec![],
            head: vec![],
        }];

        let facts: Vec<RuleAtom> = (0..100)
            .map(|i| RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant(format!("value_{}", i)),
            })
            .collect();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = reasoner.execute_distributed(&rules, &facts)?;
        }
        let duration = start.elapsed();

        self.results.push(BenchmarkResult::new(
            "Distributed Round-Robin (4 nodes)".to_string(),
            iterations,
            duration,
        ));

        Ok(())
    }

    /// Benchmark distributed reasoning with load balancing
    fn benchmark_distributed_load_balanced(&mut self, iterations: usize) -> Result<()> {
        let mut reasoner = DistributedReasoner::new(PartitionStrategy::LoadBalanced);

        // Register nodes with different capacities
        reasoner.register_node(
            Node::new("node_1".to_string(), "localhost:8001".to_string()).with_capacity(1000),
        )?;
        reasoner.register_node(
            Node::new("node_2".to_string(), "localhost:8002".to_string()).with_capacity(2000),
        )?;
        reasoner.register_node(
            Node::new("node_3".to_string(), "localhost:8003".to_string()).with_capacity(1500),
        )?;

        let rules = vec![Rule {
            name: "test_rule".to_string(),
            body: vec![],
            head: vec![],
        }];

        let facts: Vec<RuleAtom> = (0..100)
            .map(|i| RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant(format!("value_{}", i)),
            })
            .collect();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = reasoner.execute_distributed(&rules, &facts)?;
        }
        let duration = start.elapsed();

        self.results.push(BenchmarkResult::new(
            "Distributed Load-Balanced (3 nodes)".to_string(),
            iterations,
            duration,
        ));

        Ok(())
    }

    /// Benchmark distributed reasoning scaling (1, 2, 4, 8 nodes)
    fn benchmark_distributed_scaling(&mut self) -> Result<()> {
        let node_counts = [1, 2, 4, 8];
        let iterations = 50;

        for &node_count in &node_counts {
            let mut reasoner = DistributedReasoner::new(PartitionStrategy::LoadBalanced);

            for i in 1..=node_count {
                reasoner.register_node(Node::new(
                    format!("node_{}", i),
                    format!("localhost:800{}", i),
                ))?;
            }

            let rules = vec![Rule {
                name: "test_rule".to_string(),
                body: vec![],
                head: vec![],
            }];

            let facts: Vec<RuleAtom> = (0..200)
                .map(|i| RuleAtom::Triple {
                    subject: Term::Constant(format!("entity_{}", i)),
                    predicate: Term::Constant("hasProperty".to_string()),
                    object: Term::Constant(format!("value_{}", i)),
                })
                .collect();

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = reasoner.execute_distributed(&rules, &facts)?;
            }
            let duration = start.elapsed();

            self.results.push(BenchmarkResult::new(
                format!("Distributed Scaling ({} nodes)", node_count),
                iterations,
                duration,
            ));
        }

        Ok(())
    }

    /// Benchmark module registration overhead
    fn benchmark_module_registration(&mut self, iterations: usize) -> Result<()> {
        let start = Instant::now();
        for i in 0..iterations {
            let mut manager = CompositionManager::new();
            let module = RuleModule::new(format!("module_{}", i));
            let _ = manager.register_module(module);
        }
        let duration = start.elapsed();

        self.results.push(BenchmarkResult::new(
            "Module Registration".to_string(),
            iterations,
            duration,
        ));

        Ok(())
    }

    /// Benchmark template instantiation
    fn benchmark_template_instantiation(&mut self, iterations: usize) -> Result<()> {
        use crate::composition::RuleTemplate;
        use std::collections::HashMap;

        let mut template = RuleTemplate::new(
            "test_template".to_string(),
            vec!["param1".to_string(), "param2".to_string()],
        );

        template.set_body(vec![RuleAtom::Triple {
            subject: Term::Constant("${param1}".to_string()),
            predicate: Term::Constant("hasProperty".to_string()),
            object: Term::Constant("${param2}".to_string()),
        }]);

        let mut args = HashMap::new();
        args.insert("param1".to_string(), Term::Constant("entity".to_string()));
        args.insert("param2".to_string(), Term::Constant("value".to_string()));

        let start = Instant::now();
        for i in 0..iterations {
            let _ = template.instantiate(format!("rule_{}", i), &args)?;
        }
        let duration = start.elapsed();

        self.results.push(BenchmarkResult::new(
            "Template Instantiation".to_string(),
            iterations,
            duration,
        ));

        Ok(())
    }

    /// Get all benchmark results
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Print summary report
    pub fn print_summary(&self) {
        println!("\n=== Integration Performance Benchmark Summary ===\n");

        for result in &self.results {
            println!("{}", result);
        }

        println!("=== End of Benchmark Report ===\n");
    }
}

impl Default for IntegrationBenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result_creation() {
        let result =
            BenchmarkResult::new("Test Benchmark".to_string(), 1000, Duration::from_secs(1));

        assert_eq!(result.operations, 1000);
        assert!(result.ops_per_sec > 900.0); // Allow for some timing variance
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = IntegrationBenchmarkSuite::new();
        assert_eq!(suite.results.len(), 0);
    }

    #[test]
    fn test_sparql_direct_query_benchmark() {
        let mut suite = IntegrationBenchmarkSuite::new();
        suite.benchmark_sparql_direct_query(10).unwrap();

        assert_eq!(suite.results.len(), 1);
        assert_eq!(suite.results[0].operations, 10);
    }

    #[test]
    fn test_shacl_validation_benchmark() {
        let mut suite = IntegrationBenchmarkSuite::new();
        suite.benchmark_shacl_direct_validation(10).unwrap();

        assert_eq!(suite.results.len(), 1);
        assert_eq!(suite.results[0].operations, 10);
    }

    #[test]
    fn test_distributed_benchmark() {
        let mut suite = IntegrationBenchmarkSuite::new();
        suite.benchmark_distributed_round_robin(5).unwrap();

        assert_eq!(suite.results.len(), 1);
        assert_eq!(suite.results[0].operations, 5);
    }

    #[test]
    fn test_module_registration_benchmark() {
        let mut suite = IntegrationBenchmarkSuite::new();
        suite.benchmark_module_registration(10).unwrap();

        assert_eq!(suite.results.len(), 1);
        assert_eq!(suite.results[0].operations, 10);
    }
}
