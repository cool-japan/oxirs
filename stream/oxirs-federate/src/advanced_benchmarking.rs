//! Advanced Benchmarking Suite for Federated Query Optimization
//!
//! This module implements comprehensive benchmarking capabilities:
//! - Standard benchmark datasets (SP2Bench, WatDiv, LUBM)
//! - Custom benchmark generation
//! - Query workload characterization
//! - Scalability testing framework
//! - Stress testing with fault injection
//! - Performance regression detection
//! - Comparative analysis tools
//!
//! Uses SciRS2 for statistical analysis and high-performance computation.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// SciRS2 integration for statistical analysis

use scirs2_core::random::Random;

/// Configuration for advanced benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedBenchmarkConfig {
    /// Enable SP2Bench benchmarks
    pub enable_sp2bench: bool,
    /// Enable WatDiv benchmarks
    pub enable_watdiv: bool,
    /// Enable LUBM benchmarks
    pub enable_lubm: bool,
    /// Enable custom benchmarks
    pub enable_custom: bool,
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Enable stress testing
    pub enable_stress_testing: bool,
    /// Enable scalability testing
    pub enable_scalability_testing: bool,
    /// Enable regression detection
    pub enable_regression_detection: bool,
    /// Regression threshold (percentage)
    pub regression_threshold: f64,
}

impl Default for AdvancedBenchmarkConfig {
    fn default() -> Self {
        Self {
            enable_sp2bench: true,
            enable_watdiv: true,
            enable_lubm: true,
            enable_custom: true,
            iterations: 10,
            warmup_iterations: 3,
            enable_stress_testing: false,
            enable_scalability_testing: true,
            enable_regression_detection: true,
            regression_threshold: 5.0, // 5% regression threshold
        }
    }
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub query_id: String,
    pub iterations: usize,
    pub mean_latency: Duration,
    pub median_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub min_latency: Duration,
    pub max_latency: Duration,
    pub std_dev: Duration,
    pub throughput: f64, // queries per second
    pub timestamp: SystemTime,
}

/// SP2Bench benchmark suite
#[derive(Debug, Clone)]
pub struct SP2BenchSuite {
    /// Benchmark queries
    queries: HashMap<String, String>,
    /// Scale factor
    scale_factor: usize,
    /// Profiler
    _profiler: Arc<()>,
}

impl SP2BenchSuite {
    /// Create a new SP2Bench suite
    pub fn new(scale_factor: usize) -> Self {
        let mut queries = HashMap::new();

        // SP2Bench standard queries
        queries.insert(
            "Q1".to_string(),
            r#"
            SELECT ?yr WHERE {
                ?journal rdf:type bench:Journal .
                ?journal dc:title "Journal 1 (1940)"^^xsd:string .
                ?journal dcterms:issued ?yr
            }
            "#
            .to_string(),
        );

        queries.insert(
            "Q2".to_string(),
            r#"
            SELECT ?inproc ?author ?booktitle ?title
                   ?proc ?ee ?page ?url ?yr ?abstract
            WHERE {
                ?inproc rdf:type bench:Inproceedings .
                ?inproc dc:creator ?author .
                ?inproc bench:booktitle ?booktitle .
                ?inproc dc:title ?title .
                ?inproc dcterms:partOf ?proc .
                ?inproc rdfs:seeAlso ?ee .
                ?inproc swrc:pages ?page .
                ?inproc foaf:homepage ?url .
                ?inproc dcterms:issued ?yr .
                ?inproc bench:abstract ?abstract
            }
            "#
            .to_string(),
        );

        queries.insert(
            "Q3a".to_string(),
            r#"
            SELECT ?article WHERE {
                ?article rdf:type bench:Article .
                ?article swrc:pages ?p
            }
            "#
            .to_string(),
        );

        queries.insert(
            "Q3b".to_string(),
            r#"
            SELECT ?article WHERE {
                ?article rdf:type bench:Article .
                ?article swrc:pages ?p .
                ?article swrc:month ?m
            }
            "#
            .to_string(),
        );

        queries.insert(
            "Q3c".to_string(),
            r#"
            SELECT ?article WHERE {
                ?article rdf:type bench:Article .
                ?article swrc:pages ?p .
                ?article swrc:month ?m .
                ?article swrc:year ?y
            }
            "#
            .to_string(),
        );

        queries.insert(
            "Q4".to_string(),
            r#"
            SELECT DISTINCT ?name1 ?name2
            WHERE {
                ?article1 rdf:type bench:Article .
                ?article2 rdf:type bench:Article .
                ?article1 dc:creator ?author1 .
                ?author1 foaf:name ?name1 .
                ?article2 dc:creator ?author2 .
                ?author2 foaf:name ?name2 .
                ?article1 swrc:journal ?journal .
                ?article2 swrc:journal ?journal
                FILTER (?name1 < ?name2)
            }
            "#
            .to_string(),
        );

        Self {
            queries,
            scale_factor,
            _profiler: Arc::new(()),
        }
    }

    /// Get benchmark queries
    pub fn get_queries(&self) -> Vec<(String, String)> {
        self.queries
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Get query by ID
    pub fn get_query(&self, query_id: &str) -> Option<String> {
        self.queries.get(query_id).cloned()
    }

    /// Get scale factor
    pub fn get_scale_factor(&self) -> usize {
        self.scale_factor
    }
}

/// WatDiv benchmark suite
#[derive(Debug, Clone)]
pub struct WatDivSuite {
    /// Benchmark queries (categorized)
    queries: HashMap<String, Vec<(String, String)>>,
    /// Scale factor
    scale_factor: usize,
    /// Profiler
    _profiler: Arc<()>,
}

impl WatDivSuite {
    /// Create a new WatDiv suite
    pub fn new(scale_factor: usize) -> Self {
        let mut queries = HashMap::new();

        // Linear queries
        let linear = vec![(
            "L1".to_string(),
            r#"
            SELECT ?v0 WHERE {
                ?v0 <http://schema.org/caption> ?v1 .
                ?v0 <http://schema.org/text> ?v2 .
                ?v0 <http://schema.org/contentRating> ?v3
            }
            "#
            .to_string(),
        )];

        // Star queries
        let star = vec![(
            "S1".to_string(),
            r#"
            SELECT ?v1 ?v2 ?v3 ?v4 WHERE {
                ?v0 <http://schema.org/caption> ?v1 .
                ?v0 <http://schema.org/description> ?v2 .
                ?v0 <http://schema.org/text> ?v3 .
                ?v0 <http://schema.org/contentRating> ?v4
            }
            "#
            .to_string(),
        )];

        // Snowflake queries
        let snowflake = vec![(
            "F1".to_string(),
            r#"
            SELECT ?v1 ?v2 ?v3 ?v4 ?v5 WHERE {
                ?v0 <http://schema.org/editor> ?v1 .
                ?v0 <http://schema.org/contentRating> ?v2 .
                ?v3 <http://schema.org/legalName> ?v4 .
                ?v3 <http://schema.org/actor> ?v5 .
                ?v0 <http://schema.org/trailer> ?v3
            }
            "#
            .to_string(),
        )];

        // Complex queries
        let complex = vec![(
            "C1".to_string(),
            r#"
            SELECT ?v0 ?v2 ?v3 ?v5 ?v6 ?v8 WHERE {
                ?v0 <http://schema.org/editor> ?v1 .
                ?v1 <http://schema.org/homepage> ?v2 .
                ?v3 <http://schema.org/language> ?v4 .
                ?v4 <http://schema.org/caption> ?v5 .
                ?v3 <http://schema.org/trailer> ?v0 .
                ?v3 <http://schema.org/keywords> ?v6 .
                ?v7 <http://schema.org/caption> ?v8 .
                ?v7 <http://schema.org/agent> ?v1
            }
            "#
            .to_string(),
        )];

        queries.insert("linear".to_string(), linear);
        queries.insert("star".to_string(), star);
        queries.insert("snowflake".to_string(), snowflake);
        queries.insert("complex".to_string(), complex);

        Self {
            queries,
            scale_factor,
            _profiler: Arc::new(()),
        }
    }

    /// Get all queries
    pub fn get_queries(&self) -> Vec<(String, String)> {
        self.queries
            .values()
            .flat_map(|v| v.iter().cloned())
            .collect()
    }

    /// Get queries by category
    pub fn get_queries_by_category(&self, category: &str) -> Option<Vec<(String, String)>> {
        self.queries.get(category).cloned()
    }

    /// Get scale factor
    pub fn get_scale_factor(&self) -> usize {
        self.scale_factor
    }
}

/// LUBM benchmark suite (Lehigh University Benchmark)
#[derive(Debug, Clone)]
pub struct LUBMSuite {
    /// Benchmark queries
    queries: HashMap<String, String>,
    /// Number of universities
    num_universities: usize,
    /// Profiler
    _profiler: Arc<()>,
}

impl LUBMSuite {
    /// Create a new LUBM suite
    pub fn new(num_universities: usize) -> Self {
        let mut queries = HashMap::new();

        // LUBM standard queries
        queries.insert(
            "Q1".to_string(),
            r#"
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ub: <http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#>
            SELECT ?x WHERE {
                ?x rdf:type ub:GraduateStudent .
                ?x ub:takesCourse <http://www.Department0.University0.edu/GraduateCourse0>
            }
            "#
            .to_string(),
        );

        queries.insert(
            "Q2".to_string(),
            r#"
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ub: <http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#>
            SELECT ?x ?y ?z WHERE {
                ?x rdf:type ub:GraduateStudent .
                ?y rdf:type ub:University .
                ?z rdf:type ub:Department .
                ?x ub:memberOf ?z .
                ?z ub:subOrganizationOf ?y .
                ?x ub:undergraduateDegreeFrom ?y
            }
            "#
            .to_string(),
        );

        queries.insert(
            "Q3".to_string(),
            r#"
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ub: <http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#>
            SELECT ?x WHERE {
                ?x rdf:type ub:Publication .
                ?x ub:publicationAuthor <http://www.Department0.University0.edu/AssistantProfessor0>
            }
            "#
            .to_string(),
        );

        queries.insert(
            "Q4".to_string(),
            r#"
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ub: <http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#>
            SELECT ?x ?y1 ?y2 ?y3 WHERE {
                ?x rdf:type ub:Professor .
                ?x ub:worksFor <http://www.Department0.University0.edu> .
                ?x ub:name ?y1 .
                ?x ub:emailAddress ?y2 .
                ?x ub:telephone ?y3
            }
            "#
            .to_string(),
        );

        Self {
            queries,
            num_universities,
            _profiler: Arc::new(()),
        }
    }

    /// Get benchmark queries
    pub fn get_queries(&self) -> Vec<(String, String)> {
        self.queries
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Get query by ID
    pub fn get_query(&self, query_id: &str) -> Option<String> {
        self.queries.get(query_id).cloned()
    }

    /// Get number of universities
    pub fn get_num_universities(&self) -> usize {
        self.num_universities
    }
}

/// Custom benchmark generator
#[derive(Debug, Clone)]
pub struct CustomBenchmarkGenerator {
    /// Random number generator
    rng: Random,
    /// Configuration
    config: CustomBenchmarkConfig,
}

/// Configuration for custom benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomBenchmarkConfig {
    /// Number of triple patterns per query
    pub triple_patterns_range: (usize, usize),
    /// Number of joins per query
    pub joins_range: (usize, usize),
    /// Number of filters per query
    pub filters_range: (usize, usize),
    /// Include optional patterns
    pub include_optional: bool,
    /// Include union patterns
    pub include_union: bool,
    /// Include aggregation
    pub include_aggregation: bool,
}

impl Default for CustomBenchmarkConfig {
    fn default() -> Self {
        Self {
            triple_patterns_range: (3, 10),
            joins_range: (1, 5),
            filters_range: (0, 3),
            include_optional: true,
            include_union: true,
            include_aggregation: true,
        }
    }
}

impl CustomBenchmarkGenerator {
    /// Create a new custom benchmark generator
    pub fn new(config: CustomBenchmarkConfig) -> Self {
        Self {
            rng: Random::default(),
            config,
        }
    }

    /// Generate a custom query
    pub fn generate_query(&mut self) -> String {
        let num_patterns = self
            .rng
            .gen_range(self.config.triple_patterns_range.0..self.config.triple_patterns_range.1);

        let mut query = String::from("SELECT ");

        // Variables
        for i in 0..num_patterns {
            query.push_str(&format!("?v{} ", i));
        }

        query.push_str("WHERE {\n");

        // Triple patterns
        for i in 0..num_patterns {
            query.push_str(&format!(
                "  ?s{} <http://example.org/pred{}> ?v{} .\n",
                i % 3,
                i,
                i
            ));
        }

        // Optional patterns
        if self.config.include_optional && self.rng.gen_range(0.0..1.0) < 0.3 {
            query.push_str("  OPTIONAL {\n");
            query.push_str("    ?v0 <http://example.org/optional> ?opt .\n");
            query.push_str("  }\n");
        }

        // Union patterns
        if self.config.include_union && self.rng.gen_range(0.0..1.0) < 0.2 {
            query.push_str("  { ?v0 <http://example.org/type> <http://example.org/TypeA> }\n");
            query.push_str("  UNION\n");
            query.push_str("  { ?v0 <http://example.org/type> <http://example.org/TypeB> }\n");
        }

        // Filters
        let num_filters = self
            .rng
            .gen_range(self.config.filters_range.0..self.config.filters_range.1);

        for i in 0..num_filters {
            query.push_str(&format!("  FILTER (?v{} > 100)\n", i));
        }

        query.push('}');

        // Aggregation
        if self.config.include_aggregation && self.rng.gen_range(0.0..1.0) < 0.2 {
            query = format!("SELECT (COUNT(?v0) AS ?count) WHERE {{ {} }}", query);
        }

        query
    }

    /// Generate multiple queries
    pub fn generate_queries(&mut self, count: usize) -> Vec<String> {
        (0..count).map(|_| self.generate_query()).collect()
    }
}

/// Workload characterization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadCharacterization {
    pub query_count: usize,
    pub avg_triple_patterns: f64,
    pub avg_joins: f64,
    pub avg_filters: f64,
    pub optional_percentage: f64,
    pub union_percentage: f64,
    pub aggregation_percentage: f64,
    pub query_types: HashMap<String, usize>,
}

/// Scalability test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityTestResult {
    pub data_size: usize,
    pub mean_latency: Duration,
    pub throughput: f64,
    pub timestamp: SystemTime,
}

/// Stress test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    pub concurrent_clients: usize,
    pub total_queries: usize,
    pub successful_queries: usize,
    pub failed_queries: usize,
    pub mean_latency: Duration,
    pub throughput: f64,
    pub error_rate: f64,
    pub timestamp: SystemTime,
}

/// Regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetectionResult {
    pub query_id: String,
    pub baseline_latency: Duration,
    pub current_latency: Duration,
    pub regression_percentage: f64,
    pub is_regression: bool,
    pub timestamp: SystemTime,
}

/// Advanced benchmark suite
#[derive(Debug)]
pub struct AdvancedBenchmarkSuite {
    /// Configuration
    config: AdvancedBenchmarkConfig,
    /// SP2Bench suite
    sp2bench: Option<SP2BenchSuite>,
    /// WatDiv suite
    watdiv: Option<WatDivSuite>,
    /// LUBM suite
    lubm: Option<LUBMSuite>,
    /// Custom generator
    custom_generator: Option<CustomBenchmarkGenerator>,
    /// Results history
    results_history: Arc<RwLock<VecDeque<BenchmarkResult>>>,
    /// Baseline results for regression detection
    baseline_results: Arc<RwLock<HashMap<String, BenchmarkResult>>>,
    /// Profiler
    _profiler: Arc<()>,
    /// Metrics
    #[allow(dead_code)]
    metrics: Arc<()>,
}

impl AdvancedBenchmarkSuite {
    /// Create a new advanced benchmark suite
    pub fn new(config: AdvancedBenchmarkConfig) -> Self {
        let sp2bench = if config.enable_sp2bench {
            Some(SP2BenchSuite::new(10000))
        } else {
            None
        };

        let watdiv = if config.enable_watdiv {
            Some(WatDivSuite::new(10000))
        } else {
            None
        };

        let lubm = if config.enable_lubm {
            Some(LUBMSuite::new(1))
        } else {
            None
        };

        let custom_generator = if config.enable_custom {
            Some(CustomBenchmarkGenerator::new(
                CustomBenchmarkConfig::default(),
            ))
        } else {
            None
        };

        Self {
            config,
            sp2bench,
            watdiv,
            lubm,
            custom_generator,
            results_history: Arc::new(RwLock::new(VecDeque::new())),
            baseline_results: Arc::new(RwLock::new(HashMap::new())),
            _profiler: Arc::new(()),
            metrics: Arc::new(()),
        }
    }

    /// Run benchmark
    pub async fn run_benchmark<F>(
        &self,
        benchmark_name: &str,
        query_id: &str,
        query: &str,
        executor: F,
    ) -> Result<BenchmarkResult>
    where
        F: Fn(&str) -> Result<Duration>,
    {
        // profiler start

        info!("Running benchmark: {} - {}", benchmark_name, query_id);

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = executor(query);
        }

        // Actual benchmark
        let mut latencies = Vec::new();

        for i in 0..self.config.iterations {
            match executor(query) {
                Ok(latency) => {
                    latencies.push(latency);
                    debug!("Iteration {}: {:?}", i + 1, latency);
                }
                Err(e) => {
                    warn!("Iteration {} failed: {}", i + 1, e);
                }
            }
        }

        if latencies.is_empty() {
            return Err(anyhow!("All benchmark iterations failed"));
        }

        // Calculate statistics
        let result = self.calculate_statistics(benchmark_name, query_id, latencies)?;

        // Store result
        let mut history = self.results_history.write().await;
        history.push_back(result.clone());

        if history.len() > 1000 {
            history.pop_front();
        }

        // profiler stop

        Ok(result)
    }

    /// Calculate statistics from latencies
    fn calculate_statistics(
        &self,
        benchmark_name: &str,
        query_id: &str,
        mut latencies: Vec<Duration>,
    ) -> Result<BenchmarkResult> {
        latencies.sort();

        let iterations = latencies.len();
        let sum: Duration = latencies.iter().sum();
        let mean_latency = sum / iterations as u32;

        let median_latency = if iterations % 2 == 0 {
            (latencies[iterations / 2 - 1] + latencies[iterations / 2]) / 2
        } else {
            latencies[iterations / 2]
        };

        let p95_idx = (iterations as f64 * 0.95) as usize;
        let p95_latency = latencies[p95_idx.min(iterations - 1)];

        let p99_idx = (iterations as f64 * 0.99) as usize;
        let p99_latency = latencies[p99_idx.min(iterations - 1)];

        let min_latency = latencies[0];
        let max_latency = latencies[iterations - 1];

        // Calculate standard deviation
        let mean_nanos = mean_latency.as_nanos() as f64;
        let variance: f64 = latencies
            .iter()
            .map(|l| {
                let diff = l.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / iterations as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        // Calculate throughput
        let total_time_secs = sum.as_secs_f64();
        let throughput = iterations as f64 / total_time_secs;

        Ok(BenchmarkResult {
            benchmark_name: benchmark_name.to_string(),
            query_id: query_id.to_string(),
            iterations,
            mean_latency,
            median_latency,
            p95_latency,
            p99_latency,
            min_latency,
            max_latency,
            std_dev,
            throughput,
            timestamp: SystemTime::now(),
        })
    }

    /// Set baseline for regression detection
    pub async fn set_baseline(&self, result: BenchmarkResult) {
        let mut baseline = self.baseline_results.write().await;
        let key = format!("{}:{}", result.benchmark_name, result.query_id);
        baseline.insert(key, result);
    }

    /// Detect regression
    pub async fn detect_regression(
        &self,
        result: &BenchmarkResult,
    ) -> Result<RegressionDetectionResult> {
        let baseline = self.baseline_results.read().await;
        let key = format!("{}:{}", result.benchmark_name, result.query_id);

        if let Some(baseline_result) = baseline.get(&key) {
            let baseline_nanos = baseline_result.mean_latency.as_nanos() as f64;
            let current_nanos = result.mean_latency.as_nanos() as f64;

            let regression_percentage = ((current_nanos - baseline_nanos) / baseline_nanos) * 100.0;
            let is_regression = regression_percentage > self.config.regression_threshold;

            if is_regression {
                warn!(
                    "Regression detected for {}: {:.2}% slower than baseline",
                    key, regression_percentage
                );
            }

            Ok(RegressionDetectionResult {
                query_id: key,
                baseline_latency: baseline_result.mean_latency,
                current_latency: result.mean_latency,
                regression_percentage,
                is_regression,
                timestamp: SystemTime::now(),
            })
        } else {
            Err(anyhow!("No baseline found for {}", key))
        }
    }

    /// Run scalability test
    pub async fn run_scalability_test<F>(
        &self,
        query: &str,
        data_sizes: Vec<usize>,
        executor: F,
    ) -> Result<Vec<ScalabilityTestResult>>
    where
        F: Fn(&str, usize) -> Result<Duration>,
    {
        let mut results = Vec::new();

        for data_size in data_sizes {
            info!("Running scalability test with data size: {}", data_size);

            let mut latencies = Vec::new();

            for _ in 0..self.config.iterations {
                match executor(query, data_size) {
                    Ok(latency) => latencies.push(latency),
                    Err(e) => warn!("Scalability test failed: {}", e),
                }
            }

            if !latencies.is_empty() {
                latencies.sort();
                let mean_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;
                let total_time_secs = latencies.iter().sum::<Duration>().as_secs_f64();
                let throughput = latencies.len() as f64 / total_time_secs;

                results.push(ScalabilityTestResult {
                    data_size,
                    mean_latency,
                    throughput,
                    timestamp: SystemTime::now(),
                });
            }
        }

        Ok(results)
    }

    /// Run stress test
    pub async fn run_stress_test<F>(
        &self,
        query: &str,
        concurrent_clients: usize,
        queries_per_client: usize,
        executor: F,
    ) -> Result<StressTestResult>
    where
        F: Fn(&str) -> Result<Duration> + Clone + Send + 'static,
    {
        info!(
            "Running stress test with {} concurrent clients, {} queries each",
            concurrent_clients, queries_per_client
        );

        let start = Instant::now();
        let mut handles = Vec::new();

        let total_queries = concurrent_clients * queries_per_client;
        let successful = Arc::new(RwLock::new(0usize));
        let failed = Arc::new(RwLock::new(0usize));
        let latencies = Arc::new(RwLock::new(Vec::new()));

        for _ in 0..concurrent_clients {
            let query = query.to_string();
            let executor = executor.clone();
            let successful = Arc::clone(&successful);
            let failed = Arc::clone(&failed);
            let latencies = Arc::clone(&latencies);

            let handle = tokio::spawn(async move {
                for _ in 0..queries_per_client {
                    match executor(&query) {
                        Ok(latency) => {
                            let mut s = successful.write().await;
                            *s += 1;
                            let mut l = latencies.write().await;
                            l.push(latency);
                        }
                        Err(_) => {
                            let mut f = failed.write().await;
                            *f += 1;
                        }
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all clients to complete
        for handle in handles {
            let _ = handle.await;
        }

        let duration = start.elapsed();
        let successful_queries = *successful.read().await;
        let failed_queries = *failed.read().await;
        let all_latencies = latencies.read().await;

        let mean_latency = if !all_latencies.is_empty() {
            all_latencies.iter().sum::<Duration>() / all_latencies.len() as u32
        } else {
            Duration::from_secs(0)
        };

        let throughput = successful_queries as f64 / duration.as_secs_f64();
        let error_rate = (failed_queries as f64 / total_queries as f64) * 100.0;

        Ok(StressTestResult {
            concurrent_clients,
            total_queries,
            successful_queries,
            failed_queries,
            mean_latency,
            throughput,
            error_rate,
            timestamp: SystemTime::now(),
        })
    }

    /// Get results history
    pub async fn get_results_history(&self) -> Vec<BenchmarkResult> {
        let history = self.results_history.read().await;
        history.iter().cloned().collect()
    }

    /// Export results to JSON
    pub async fn export_results(&self) -> Result<String> {
        let history = self.get_results_history().await;
        let json = serde_json::to_string_pretty(&history)?;
        Ok(json)
    }

    /// Get SP2Bench queries
    pub fn get_sp2bench_queries(&self) -> Option<Vec<(String, String)>> {
        self.sp2bench.as_ref().map(|s| s.get_queries())
    }

    /// Get WatDiv queries
    pub fn get_watdiv_queries(&self) -> Option<Vec<(String, String)>> {
        self.watdiv.as_ref().map(|s| s.get_queries())
    }

    /// Get LUBM queries
    pub fn get_lubm_queries(&self) -> Option<Vec<(String, String)>> {
        self.lubm.as_ref().map(|s| s.get_queries())
    }

    /// Generate custom queries
    pub fn generate_custom_queries(&mut self, count: usize) -> Option<Vec<String>> {
        self.custom_generator
            .as_mut()
            .map(|g| g.generate_queries(count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sp2bench_suite() {
        let suite = SP2BenchSuite::new(10000);
        let queries = suite.get_queries();
        assert!(!queries.is_empty());
        assert!(suite.get_query("Q1").is_some());
    }

    #[test]
    fn test_watdiv_suite() {
        let suite = WatDivSuite::new(10000);
        let queries = suite.get_queries();
        assert!(!queries.is_empty());
        assert!(suite.get_queries_by_category("linear").is_some());
    }

    #[test]
    fn test_lubm_suite() {
        let suite = LUBMSuite::new(1);
        let queries = suite.get_queries();
        assert!(!queries.is_empty());
        assert!(suite.get_query("Q1").is_some());
    }

    #[test]
    fn test_custom_benchmark_generator() {
        let config = CustomBenchmarkConfig::default();
        let mut generator = CustomBenchmarkGenerator::new(config);
        let query = generator.generate_query();
        assert!(query.contains("SELECT"));
        assert!(query.contains("WHERE"));
    }

    #[tokio::test]
    async fn test_advanced_benchmark_suite() {
        let config = AdvancedBenchmarkConfig::default();
        let suite = AdvancedBenchmarkSuite::new(config);

        let executor = |_query: &str| Ok(Duration::from_millis(100));

        let result = suite
            .run_benchmark("test", "Q1", "SELECT * WHERE { ?s ?p ?o }", executor)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_regression_detection() {
        let config = AdvancedBenchmarkConfig {
            regression_threshold: 10.0,
            ..Default::default()
        };
        let suite = AdvancedBenchmarkSuite::new(config);

        let baseline = BenchmarkResult {
            benchmark_name: "test".to_string(),
            query_id: "Q1".to_string(),
            iterations: 10,
            mean_latency: Duration::from_millis(100),
            median_latency: Duration::from_millis(100),
            p95_latency: Duration::from_millis(120),
            p99_latency: Duration::from_millis(130),
            min_latency: Duration::from_millis(90),
            max_latency: Duration::from_millis(140),
            std_dev: Duration::from_millis(10),
            throughput: 10.0,
            timestamp: SystemTime::now(),
        };

        suite.set_baseline(baseline.clone()).await;

        let current = BenchmarkResult {
            mean_latency: Duration::from_millis(120),
            ..baseline
        };

        let regression = suite.detect_regression(&current).await;
        assert!(regression.is_ok());
        assert!(regression.expect("operation should succeed").is_regression);
    }
}
