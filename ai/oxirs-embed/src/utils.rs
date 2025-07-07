//! Utility functions and helpers for embedding operations

// Removed unused imports
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::{thread_rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

/// Data loading utilities
pub mod data_loader {
    use super::*;
    use std::io::{BufRead, BufReader};

    /// Load triples from TSV file format
    pub fn load_triples_from_tsv<P: AsRef<Path>>(
        file_path: P,
    ) -> Result<Vec<(String, String, String)>> {
        let file = fs::File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut triples = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() || line.starts_with('#') {
                continue; // Skip empty lines and comments
            }

            // Skip header line (first line that contains common header terms)
            if line_num == 0
                && (line.contains("subject")
                    || line.contains("predicate")
                    || line.contains("object"))
            {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let subject = parts[0].trim().to_string();
                let predicate = parts[1].trim().to_string();
                let object = parts[2].trim().to_string();
                triples.push((subject, predicate, object));
            } else {
                eprintln!(
                    "Warning: Invalid triple format at line {}: {}",
                    line_num + 1,
                    line
                );
            }
        }

        Ok(triples)
    }

    /// Load triples from CSV file format
    pub fn load_triples_from_csv<P: AsRef<Path>>(
        file_path: P,
    ) -> Result<Vec<(String, String, String)>> {
        let file = fs::File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut triples = Vec::new();
        let mut is_first_line = true;

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            if is_first_line {
                is_first_line = false;
                // Skip header if it looks like one
                if line.to_lowercase().contains("subject")
                    && line.to_lowercase().contains("predicate")
                {
                    continue;
                }
            }

            if line.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 3 {
                let subject = parts[0].trim().trim_matches('"').to_string();
                let predicate = parts[1].trim().trim_matches('"').to_string();
                let object = parts[2].trim().trim_matches('"').to_string();
                triples.push((subject, predicate, object));
            } else {
                eprintln!(
                    "Warning: Invalid triple format at line {}: {}",
                    line_num + 1,
                    line
                );
            }
        }

        Ok(triples)
    }

    /// Load triples from N-Triples format
    pub fn load_triples_from_ntriples<P: AsRef<Path>>(
        file_path: P,
    ) -> Result<Vec<(String, String, String)>> {
        let file = fs::File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut triples = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Simple N-Triples parser (very basic)
            if let Some(triple) = parse_ntriple_line(line) {
                triples.push(triple);
            } else {
                eprintln!(
                    "Warning: Failed to parse N-Triple at line {}: {}",
                    line_num + 1,
                    line
                );
            }
        }

        Ok(triples)
    }

    /// Parse a single N-Triple line
    fn parse_ntriple_line(line: &str) -> Option<(String, String, String)> {
        let line = line.trim_end_matches(" .");
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() >= 3 {
            let subject = clean_uri_or_literal(parts[0]);
            let predicate = clean_uri_or_literal(parts[1]);
            let object = clean_uri_or_literal(&parts[2..].join(" "));

            Some((subject, predicate, object))
        } else {
            None
        }
    }

    /// Clean URI or literal from N-Triple format
    fn clean_uri_or_literal(term: &str) -> String {
        if term.starts_with('<') && term.ends_with('>') {
            term[1..term.len() - 1].to_string()
        } else if term.starts_with('"') && term.contains('"') {
            // Handle literals - just take the string part for now
            let end_quote = term.rfind('"').unwrap_or(term.len());
            term[1..end_quote].to_string()
        } else {
            term.to_string()
        }
    }

    /// Load triples from JSON Lines format (one triple per line as JSON)
    pub fn load_triples_from_jsonl<P: AsRef<Path>>(
        file_path: P,
    ) -> Result<Vec<(String, String, String)>> {
        let file = fs::File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut triples = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<serde_json::Value>(&line) {
                Ok(json) => {
                    if let (Some(subject), Some(predicate), Some(object)) = (
                        json["subject"].as_str(),
                        json["predicate"].as_str(),
                        json["object"].as_str(),
                    ) {
                        triples.push((
                            subject.to_string(),
                            predicate.to_string(),
                            object.to_string(),
                        ));
                    } else {
                        eprintln!(
                            "Warning: Invalid JSON structure at line {}: {}",
                            line_num + 1,
                            line
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to parse JSON at line {}: {} - Error: {}",
                        line_num + 1,
                        line,
                        e
                    );
                }
            }
        }

        Ok(triples)
    }

    /// Save triples to TSV format
    pub fn save_triples_to_tsv<P: AsRef<Path>>(
        triples: &[(String, String, String)],
        file_path: P,
    ) -> Result<()> {
        let mut content = String::new();
        content.push_str("subject\tpredicate\tobject\n");

        for (subject, predicate, object) in triples {
            content.push_str(&format!("{subject}\t{predicate}\t{object}\n"));
        }

        fs::write(file_path, content)?;
        Ok(())
    }

    /// Save triples to JSON Lines format
    pub fn save_triples_to_jsonl<P: AsRef<Path>>(
        triples: &[(String, String, String)],
        file_path: P,
    ) -> Result<()> {
        use std::io::Write;
        let mut file = fs::File::create(file_path)?;

        for (subject, predicate, object) in triples {
            let json = serde_json::json!({
                "subject": subject,
                "predicate": predicate,
                "object": object
            });
            writeln!(file, "{json}")?;
        }

        Ok(())
    }

    /// Auto-detect file format and load triples accordingly
    pub fn load_triples_auto_detect<P: AsRef<Path>>(
        file_path: P,
    ) -> Result<Vec<(String, String, String)>> {
        let path = file_path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "tsv" => load_triples_from_tsv(path),
            "csv" => load_triples_from_csv(path),
            "nt" | "ntriples" => load_triples_from_ntriples(path),
            "jsonl" | "ndjson" => load_triples_from_jsonl(path),
            _ => {
                // Try to auto-detect based on content
                eprintln!(
                    "Warning: Unknown file extension '{}', attempting auto-detection",
                    extension
                );

                // Try TSV first (most common)
                if let Ok(triples) = load_triples_from_tsv(path) {
                    if !triples.is_empty() {
                        return Ok(triples);
                    }
                }

                // Try N-Triples
                if let Ok(triples) = load_triples_from_ntriples(path) {
                    if !triples.is_empty() {
                        return Ok(triples);
                    }
                }

                // Try JSON Lines
                if let Ok(triples) = load_triples_from_jsonl(path) {
                    if !triples.is_empty() {
                        return Ok(triples);
                    }
                }

                // Finally try CSV
                load_triples_from_csv(path)
            }
        }
    }
}

/// Dataset splitting utilities
pub mod dataset_splitter {
    use super::*;

    /// Split dataset into train/validation/test sets
    pub fn split_dataset(
        triples: Vec<(String, String, String)>,
        train_ratio: f64,
        val_ratio: f64,
        seed: Option<u64>,
    ) -> Result<DatasetSplit> {
        if train_ratio + val_ratio >= 1.0 {
            return Err(anyhow!(
                "Train and validation ratios must sum to less than 1.0"
            ));
        }

        let mut rng = if let Some(s) = seed {
            StdRng::seed_from_u64(s)
        } else {
            StdRng::from_rng(&mut thread_rng()).expect("Failed to create RNG")
        };

        let mut shuffled_triples = triples;
        shuffled_triples.shuffle(&mut rng);

        let total = shuffled_triples.len();
        let train_end = (total as f64 * train_ratio) as usize;
        let val_end = train_end + (total as f64 * val_ratio) as usize;

        let train_triples = shuffled_triples[..train_end].to_vec();
        let val_triples = shuffled_triples[train_end..val_end].to_vec();
        let test_triples = shuffled_triples[val_end..].to_vec();

        Ok(DatasetSplit {
            train: train_triples,
            validation: val_triples,
            test: test_triples,
        })
    }

    /// Split dataset ensuring no entity leakage between splits
    pub fn split_dataset_no_leakage(
        triples: Vec<(String, String, String)>,
        train_ratio: f64,
        val_ratio: f64,
        seed: Option<u64>,
    ) -> Result<DatasetSplit> {
        // Group triples by entities with pre-allocated capacity for better performance
        let mut entity_triples: HashMap<String, Vec<(String, String, String)>> =
            HashMap::with_capacity(triples.len() / 2); // Estimate capacity

        for triple in &triples {
            let entities = [&triple.0, &triple.2];
            for entity in entities {
                entity_triples
                    .entry(entity.clone())
                    .or_insert_with(Vec::new)
                    .push(triple.clone());
            }
        }

        // Split entities first, then assign triples - optimized allocation
        let entities: Vec<String> = entity_triples.keys().cloned().collect();
        let dummy_string = "dummy".to_string();
        let entity_split = split_dataset(
            entities
                .into_iter()
                .map(|e| (e, dummy_string.clone(), dummy_string.clone()))
                .collect(),
            train_ratio,
            val_ratio,
            seed,
        )?;

        let train_entities: HashSet<String> =
            entity_split.train.into_iter().map(|(e, _, _)| e).collect();
        let val_entities: HashSet<String> = entity_split
            .validation
            .into_iter()
            .map(|(e, _, _)| e)
            .collect();
        let test_entities: HashSet<String> =
            entity_split.test.into_iter().map(|(e, _, _)| e).collect();

        // Assign triples based on entity membership with pre-allocated capacity
        let estimated_capacity = triples.len() / 3;
        let mut train_triples = Vec::with_capacity(estimated_capacity);
        let mut val_triples = Vec::with_capacity(estimated_capacity);
        let mut test_triples = Vec::with_capacity(estimated_capacity);

        for (entity, entity_triple_list) in entity_triples {
            if train_entities.contains(&entity) {
                train_triples.extend(entity_triple_list);
            } else if val_entities.contains(&entity) {
                val_triples.extend(entity_triple_list);
            } else if test_entities.contains(&entity) {
                test_triples.extend(entity_triple_list);
            }
        }

        // Remove duplicates
        train_triples.sort();
        train_triples.dedup();
        val_triples.sort();
        val_triples.dedup();
        test_triples.sort();
        test_triples.dedup();

        Ok(DatasetSplit {
            train: train_triples,
            validation: val_triples,
            test: test_triples,
        })
    }
}

/// Dataset split result
#[derive(Debug, Clone)]
pub struct DatasetSplit {
    pub train: Vec<(String, String, String)>,
    pub validation: Vec<(String, String, String)>,
    pub test: Vec<(String, String, String)>,
}

/// Statistics about a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    pub num_triples: usize,
    pub num_entities: usize,
    pub num_relations: usize,
    pub entity_frequency: HashMap<String, usize>,
    pub relation_frequency: HashMap<String, usize>,
    pub avg_degree: f64,
    pub density: f64,
}

/// Compute dataset statistics
pub fn compute_dataset_statistics(triples: &[(String, String, String)]) -> DatasetStatistics {
    let mut entities = HashSet::new();
    let mut relations = HashSet::new();
    let mut entity_frequency = HashMap::new();
    let mut relation_frequency = HashMap::new();

    for (subject, predicate, object) in triples {
        entities.insert(subject.clone());
        entities.insert(object.clone());
        relations.insert(predicate.clone());

        *entity_frequency.entry(subject.clone()).or_insert(0) += 1;
        *entity_frequency.entry(object.clone()).or_insert(0) += 1;
        *relation_frequency.entry(predicate.clone()).or_insert(0) += 1;
    }

    let num_entities = entities.len();
    let num_relations = relations.len();
    let num_triples = triples.len();

    let avg_degree = if num_entities > 0 {
        (num_triples * 2) as f64 / num_entities as f64
    } else {
        0.0
    };

    let max_possible_edges = num_entities * num_entities;
    let density = if max_possible_edges > 0 {
        num_triples as f64 / max_possible_edges as f64
    } else {
        0.0
    };

    DatasetStatistics {
        num_triples,
        num_entities,
        num_relations,
        entity_frequency,
        relation_frequency,
        avg_degree,
        density,
    }
}

/// Embedding dimension analysis utilities
pub mod embedding_analysis {
    use super::*;

    /// Analyze embedding distribution
    pub fn analyze_embedding_distribution(embeddings: &Array2<f64>) -> EmbeddingDistributionStats {
        let flat_values: Vec<f64> = embeddings.iter().cloned().collect();

        let mean = flat_values.iter().sum::<f64>() / flat_values.len() as f64;
        let variance =
            flat_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / flat_values.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted_values = flat_values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_val = sorted_values[0];
        let max_val = sorted_values[sorted_values.len() - 1];
        let median = sorted_values[sorted_values.len() / 2];

        EmbeddingDistributionStats {
            mean,
            std_dev,
            variance,
            min: min_val,
            max: max_val,
            median,
            num_parameters: embeddings.len(),
        }
    }

    /// Compute embedding norms
    pub fn compute_embedding_norms(embeddings: &Array2<f64>) -> Vec<f64> {
        embeddings
            .rows()
            .into_iter()
            .map(|row| row.dot(&row).sqrt())
            .collect()
    }

    /// Analyze embedding similarities
    pub fn analyze_embedding_similarities(
        embeddings: &Array2<f64>,
        sample_size: usize,
    ) -> SimilarityStats {
        let num_embeddings = embeddings.nrows();
        let mut similarities = Vec::new();

        let sample_size = sample_size.min(num_embeddings * (num_embeddings - 1) / 2);
        let mut rng = thread_rng();

        for _ in 0..sample_size {
            let i = rng.gen_range(0..num_embeddings);
            let j = rng.gen_range(0..num_embeddings);

            if i != j {
                let emb_i = embeddings.row(i);
                let emb_j = embeddings.row(j);
                let similarity = cosine_similarity(&emb_i.to_owned(), &emb_j.to_owned());
                similarities.push(similarity);
            }
        }

        similarities.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean_similarity = similarities.iter().sum::<f64>() / similarities.len() as f64;
        let min_similarity = similarities[0];
        let max_similarity = similarities[similarities.len() - 1];
        let median_similarity = similarities[similarities.len() / 2];

        SimilarityStats {
            mean_similarity,
            min_similarity,
            max_similarity,
            median_similarity,
            num_comparisons: similarities.len(),
        }
    }

    /// Cosine similarity between two vectors
    fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a > 1e-10 && norm_b > 1e-10 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Embedding distribution statistics
#[derive(Debug, Clone)]
pub struct EmbeddingDistributionStats {
    pub mean: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub num_parameters: usize,
}

/// Similarity statistics
#[derive(Debug, Clone)]
pub struct SimilarityStats {
    pub mean_similarity: f64,
    pub min_similarity: f64,
    pub max_similarity: f64,
    pub median_similarity: f64,
    pub num_comparisons: usize,
}

/// Graph analysis utilities
pub mod graph_analysis {
    use super::*;

    /// Compute graph metrics for knowledge graph - optimized for performance
    pub fn compute_graph_metrics(triples: &[(String, String, String)]) -> GraphMetrics {
        // Pre-allocate with estimated capacity for better performance
        let estimated_entities = triples.len(); // Conservative estimate
        let estimated_relations = triples.len() / 10; // Rough estimate

        let mut entity_degrees: HashMap<String, usize> = HashMap::with_capacity(estimated_entities);
        let mut relation_counts: HashMap<String, usize> =
            HashMap::with_capacity(estimated_relations);
        let mut entities = HashSet::with_capacity(estimated_entities);

        for (subject, predicate, object) in triples {
            entities.insert(subject.clone());
            entities.insert(object.clone());

            *entity_degrees.entry(subject.clone()).or_insert(0) += 1;
            *entity_degrees.entry(object.clone()).or_insert(0) += 1;
            *relation_counts.entry(predicate.clone()).or_insert(0) += 1;
        }

        let num_entities = entities.len();
        let num_relations = relation_counts.len();
        let num_triples = triples.len();

        let degrees: Vec<usize> = entity_degrees.values().cloned().collect();
        let avg_degree = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;
        let max_degree = degrees.iter().max().cloned().unwrap_or(0);
        let min_degree = degrees.iter().min().cloned().unwrap_or(0);

        GraphMetrics {
            num_entities,
            num_relations,
            num_triples,
            avg_degree,
            max_degree,
            min_degree,
            density: num_triples as f64 / (num_entities * num_entities) as f64,
        }
    }
}

/// Graph metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    pub num_entities: usize,
    pub num_relations: usize,
    pub num_triples: usize,
    pub avg_degree: f64,
    pub max_degree: usize,
    pub min_degree: usize,
    pub density: f64,
}

/// Progress tracking utilities
#[derive(Debug)]
pub struct ProgressTracker {
    total: usize,
    current: usize,
    start_time: std::time::Instant,
    last_update: std::time::Instant,
    update_interval: std::time::Duration,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new(total: usize) -> Self {
        let now = std::time::Instant::now();
        Self {
            total,
            current: 0,
            start_time: now,
            last_update: now,
            update_interval: std::time::Duration::from_secs(1),
        }
    }

    /// Update progress
    pub fn update(&mut self, current: usize) {
        self.current = current;
        let now = std::time::Instant::now();

        if now.duration_since(self.last_update) >= self.update_interval {
            self.print_progress();
            self.last_update = now;
        }
    }

    /// Print current progress
    fn print_progress(&self) {
        let percentage = (self.current as f64 / self.total as f64) * 100.0;
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = self.current as f64 / elapsed;

        println!(
            "Progress: {}/{} ({:.1}%) - {:.1} items/sec",
            self.current, self.total, percentage, rate
        );
    }

    /// Finish and print final statistics
    pub fn finish(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = self.total as f64 / elapsed;

        println!(
            "Completed: {} items in {:.2}s ({:.1} items/sec)",
            self.total, elapsed, rate
        );
    }
}

/// Performance benchmarking and profiling utilities
pub mod performance_benchmark {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::{Duration, Instant};

    /// Comprehensive performance benchmarking for embedding operations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BenchmarkSuite {
        /// Results organized by operation type
        pub results: BTreeMap<String, BenchmarkResult>,
        /// Overall benchmark statistics
        pub summary: BenchmarkSummary,
        /// Benchmark configuration
        pub config: BenchmarkConfig,
    }

    /// Individual benchmark result for a specific operation
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BenchmarkResult {
        /// Operation name
        pub operation: String,
        /// Total number of iterations
        pub iterations: usize,
        /// Total elapsed time
        pub total_duration: Duration,
        /// Average time per operation
        pub avg_duration: Duration,
        /// Minimum time observed
        pub min_duration: Duration,
        /// Maximum time observed
        pub max_duration: Duration,
        /// Standard deviation of durations
        pub std_deviation: Duration,
        /// Operations per second
        pub ops_per_second: f64,
        /// Memory usage statistics
        pub memory_stats: MemoryStats,
        /// Additional metrics
        pub custom_metrics: HashMap<String, f64>,
    }

    /// Memory usage statistics
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MemoryStats {
        /// Peak memory usage (bytes)
        pub peak_memory_bytes: usize,
        /// Average memory usage (bytes)
        pub avg_memory_bytes: usize,
        /// Memory allocations count
        pub allocations: usize,
        /// Memory deallocations count
        pub deallocations: usize,
    }

    /// Overall benchmark summary
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BenchmarkSummary {
        /// Total benchmark duration
        pub total_duration: Duration,
        /// Number of operations benchmarked
        pub total_operations: usize,
        /// Overall throughput (ops/sec)
        pub overall_throughput: f64,
        /// Performance efficiency score (0.0-1.0)
        pub efficiency_score: f64,
        /// Bottleneck analysis
        pub bottlenecks: Vec<String>,
    }

    /// Benchmarking configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BenchmarkConfig {
        /// Number of warmup iterations
        pub warmup_iterations: usize,
        /// Number of measurement iterations
        pub measurement_iterations: usize,
        /// Target confidence level (0.0-1.0)
        pub confidence_level: f64,
        /// Enable memory profiling
        pub enable_memory_profiling: bool,
        /// Enable detailed timing analysis
        pub enable_detailed_timing: bool,
    }

    impl Default for BenchmarkConfig {
        fn default() -> Self {
            Self {
                warmup_iterations: 100,
                measurement_iterations: 1000,
                confidence_level: 0.95,
                enable_memory_profiling: true,
                enable_detailed_timing: true,
            }
        }
    }

    /// High-precision timer for micro-benchmarking
    pub struct PrecisionTimer {
        start_time: Instant,
        lap_times: Vec<Duration>,
    }

    impl PrecisionTimer {
        pub fn new() -> Self {
            Self {
                start_time: Instant::now(),
                lap_times: Vec::new(),
            }
        }

        /// Start timing
        pub fn start(&mut self) {
            self.start_time = Instant::now();
            self.lap_times.clear();
        }

        /// Record a lap time
        pub fn lap(&mut self) -> Duration {
            let lap_duration = self.start_time.elapsed();
            self.lap_times.push(lap_duration);
            lap_duration
        }

        /// Stop timing and return final duration
        pub fn stop(&self) -> Duration {
            self.start_time.elapsed()
        }

        /// Get all recorded lap times
        pub fn lap_times(&self) -> &[Duration] {
            &self.lap_times
        }
    }

    /// Benchmarking framework for embedding operations
    pub struct EmbeddingBenchmark {
        config: BenchmarkConfig,
        results: BTreeMap<String, BenchmarkResult>,
    }

    impl EmbeddingBenchmark {
        pub fn new(config: BenchmarkConfig) -> Self {
            Self {
                config,
                results: BTreeMap::new(),
            }
        }

        /// Benchmark a function with comprehensive timing and memory analysis
        pub fn benchmark<F, T>(&mut self, name: &str, mut operation: F) -> Result<T>
        where
            F: FnMut() -> Result<T>,
        {
            // Warmup phase
            for _ in 0..self.config.warmup_iterations {
                let _ = operation()?;
            }

            let mut durations = Vec::with_capacity(self.config.measurement_iterations);
            let mut memory_snapshots = Vec::new();
            let mut result = None;

            // Measurement phase
            for i in 0..self.config.measurement_iterations {
                let memory_before = self.get_memory_usage();
                let start = Instant::now();

                let op_result = operation()?;

                let duration = start.elapsed();
                let memory_after = self.get_memory_usage();

                durations.push(duration);

                if self.config.enable_memory_profiling {
                    memory_snapshots.push((memory_before, memory_after));
                }

                // Store result from the first iteration
                if i == 0 {
                    result = Some(op_result);
                }
            }

            // Calculate statistics
            let total_duration: Duration = durations.iter().sum();
            let avg_duration = total_duration / durations.len() as u32;
            let min_duration = *durations.iter().min().unwrap();
            let max_duration = *durations.iter().max().unwrap();

            // Calculate standard deviation
            let variance: f64 = durations
                .iter()
                .map(|d| {
                    let diff = d.as_nanos() as f64 - avg_duration.as_nanos() as f64;
                    diff * diff
                })
                .sum::<f64>()
                / durations.len() as f64;
            let std_deviation = Duration::from_nanos(variance.sqrt() as u64);

            let ops_per_second = 1_000_000_000.0 / avg_duration.as_nanos() as f64;

            // Memory statistics
            let memory_stats = if self.config.enable_memory_profiling
                && !memory_snapshots.is_empty()
            {
                let peak_memory = memory_snapshots
                    .iter()
                    .map(|(_, after)| after.peak_memory_bytes)
                    .max()
                    .unwrap_or(0);

                let avg_memory = memory_snapshots
                    .iter()
                    .map(|(before, after)| (before.avg_memory_bytes + after.avg_memory_bytes) / 2)
                    .sum::<usize>()
                    / memory_snapshots.len();

                MemoryStats {
                    peak_memory_bytes: peak_memory,
                    avg_memory_bytes: avg_memory,
                    allocations: memory_snapshots.len(),
                    deallocations: 0, // Simplified for now
                }
            } else {
                MemoryStats {
                    peak_memory_bytes: 0,
                    avg_memory_bytes: 0,
                    allocations: 0,
                    deallocations: 0,
                }
            };

            let benchmark_result = BenchmarkResult {
                operation: name.to_string(),
                iterations: self.config.measurement_iterations,
                total_duration,
                avg_duration,
                min_duration,
                max_duration,
                std_deviation,
                ops_per_second,
                memory_stats,
                custom_metrics: HashMap::new(),
            };

            self.results.insert(name.to_string(), benchmark_result);

            result.ok_or_else(|| anyhow!("Failed to capture benchmark result"))
        }

        /// Generate comprehensive benchmark report
        pub fn generate_report(&self) -> BenchmarkSuite {
            let total_duration = self.results.values().map(|r| r.total_duration).sum();

            let total_operations = self.results.len();

            let overall_throughput = self.results.values().map(|r| r.ops_per_second).sum::<f64>()
                / total_operations as f64;

            // Calculate efficiency score based on consistency and performance
            let efficiency_score = self.calculate_efficiency_score();

            // Identify bottlenecks
            let bottlenecks = self.identify_bottlenecks();

            let summary = BenchmarkSummary {
                total_duration,
                total_operations,
                overall_throughput,
                efficiency_score,
                bottlenecks,
            };

            BenchmarkSuite {
                results: self.results.clone(),
                summary,
                config: self.config.clone(),
            }
        }

        /// Calculate efficiency score based on performance consistency
        fn calculate_efficiency_score(&self) -> f64 {
            if self.results.is_empty() {
                return 0.0;
            }

            let consistency_scores: Vec<f64> = self
                .results
                .values()
                .map(|result| {
                    // Calculate coefficient of variation (std_dev / mean)
                    let cv = result.std_deviation.as_nanos() as f64
                        / result.avg_duration.as_nanos() as f64;
                    // Convert to consistency score (lower CV = higher consistency)
                    1.0 / (1.0 + cv)
                })
                .collect();

            consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64
        }

        /// Identify performance bottlenecks
        fn identify_bottlenecks(&self) -> Vec<String> {
            let mut bottlenecks = Vec::new();

            // Find operations with high standard deviation (inconsistent performance)
            for (name, result) in &self.results {
                let cv =
                    result.std_deviation.as_nanos() as f64 / result.avg_duration.as_nanos() as f64;
                if cv > 0.2 {
                    // 20% coefficient of variation threshold
                    bottlenecks.push(format!("High variance in {}: {:.2}% CV", name, cv * 100.0));
                }
            }

            // Find slow operations (below average throughput)
            let avg_throughput = self.results.values().map(|r| r.ops_per_second).sum::<f64>()
                / self.results.len() as f64;

            for (name, result) in &self.results {
                if result.ops_per_second < avg_throughput * 0.5 {
                    // 50% below average
                    bottlenecks.push(format!(
                        "Slow operation {}: {:.0} ops/sec",
                        name, result.ops_per_second
                    ));
                }
            }

            bottlenecks
        }

        /// Get current memory usage (simplified implementation)
        fn get_memory_usage(&self) -> MemoryStats {
            // This is a simplified implementation
            // In a real-world scenario, you'd use proper memory profiling tools
            MemoryStats {
                peak_memory_bytes: 0,
                avg_memory_bytes: 0,
                allocations: 0,
                deallocations: 0,
            }
        }
    }

    /// Utility functions for performance analysis
    pub mod analysis {
        use super::*;

        /// Compare two benchmark results
        pub fn compare_benchmarks(
            baseline: &BenchmarkResult,
            comparison: &BenchmarkResult,
        ) -> BenchmarkComparison {
            let throughput_improvement =
                (comparison.ops_per_second - baseline.ops_per_second) / baseline.ops_per_second;

            let latency_improvement = (baseline.avg_duration.as_nanos() as f64
                - comparison.avg_duration.as_nanos() as f64)
                / baseline.avg_duration.as_nanos() as f64;

            let consistency_improvement = {
                let baseline_cv = baseline.std_deviation.as_nanos() as f64
                    / baseline.avg_duration.as_nanos() as f64;
                let comparison_cv = comparison.std_deviation.as_nanos() as f64
                    / comparison.avg_duration.as_nanos() as f64;
                (baseline_cv - comparison_cv) / baseline_cv
            };

            BenchmarkComparison {
                baseline_name: baseline.operation.clone(),
                comparison_name: comparison.operation.clone(),
                throughput_improvement,
                latency_improvement,
                consistency_improvement,
                is_improvement: throughput_improvement > 0.0 && latency_improvement > 0.0,
            }
        }

        /// Generate performance regression analysis
        pub fn analyze_regression(
            historical_results: &[BenchmarkResult],
            current_result: &BenchmarkResult,
        ) -> RegressionAnalysis {
            if historical_results.is_empty() {
                return RegressionAnalysis::default();
            }

            let historical_avg_throughput = historical_results
                .iter()
                .map(|r| r.ops_per_second)
                .sum::<f64>()
                / historical_results.len() as f64;

            let throughput_change = (current_result.ops_per_second - historical_avg_throughput)
                / historical_avg_throughput;

            let is_regression = throughput_change < -0.05; // 5% threshold

            RegressionAnalysis {
                throughput_change,
                is_regression,
                confidence_level: 0.95, // Simplified
                analysis_notes: if is_regression {
                    vec!["Performance regression detected".to_string()]
                } else {
                    vec!["Performance within expected range".to_string()]
                },
            }
        }
    }

    /// Benchmark comparison result
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BenchmarkComparison {
        pub baseline_name: String,
        pub comparison_name: String,
        pub throughput_improvement: f64,
        pub latency_improvement: f64,
        pub consistency_improvement: f64,
        pub is_improvement: bool,
    }

    /// Performance regression analysis
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RegressionAnalysis {
        pub throughput_change: f64,
        pub is_regression: bool,
        pub confidence_level: f64,
        pub analysis_notes: Vec<String>,
    }

    impl Default for RegressionAnalysis {
        fn default() -> Self {
            Self {
                throughput_change: 0.0,
                is_regression: false,
                confidence_level: 0.0,
                analysis_notes: vec!["No historical data available".to_string()],
            }
        }
    }
}

/// Convenience functions for quick operations
pub mod convenience {
    use super::*;
    use crate::{EmbeddingModel, ModelConfig, NamedNode, TransE, Triple};

    /// Create a simple TransE model with sensible defaults for quick prototyping
    pub fn create_simple_transe_model() -> TransE {
        let config = ModelConfig::default()
            .with_dimensions(128)
            .with_learning_rate(0.01)
            .with_max_epochs(100);
        TransE::new(config)
    }

    /// Parse a triple from a simple string format "subject predicate object"
    pub fn parse_triple_from_string(triple_str: &str) -> Result<Triple> {
        let parts: Vec<&str> = triple_str.trim().split_whitespace().collect();
        if parts.len() != 3 {
            return Err(anyhow!(
                "Invalid triple format. Expected 'subject predicate object', got: '{}'",
                triple_str
            ));
        }

        let subject = if parts[0].starts_with("http") {
            NamedNode::new(parts[0])?
        } else {
            NamedNode::new(&format!("http://example.org/{}", parts[0]))?
        };

        let predicate = if parts[1].starts_with("http") {
            NamedNode::new(parts[1])?
        } else {
            NamedNode::new(&format!("http://example.org/{}", parts[1]))?
        };

        let object = if parts[2].starts_with("http") {
            NamedNode::new(parts[2])?
        } else {
            NamedNode::new(&format!("http://example.org/{}", parts[2]))?
        };

        Ok(Triple::new(subject, predicate, object))
    }

    /// Add multiple triples from string array to a model
    pub fn add_triples_from_strings(
        model: &mut dyn EmbeddingModel,
        triple_strings: &[&str],
    ) -> Result<usize> {
        let mut added_count = 0;
        for triple_str in triple_strings {
            match parse_triple_from_string(triple_str) {
                Ok(triple) => {
                    model.add_triple(triple)?;
                    added_count += 1;
                }
                Err(e) => {
                    eprintln!("Warning: Failed to parse triple '{triple_str}': {e}");
                }
            }
        }
        Ok(added_count)
    }

    /// Create a quick biomedical model with default settings
    #[cfg(feature = "biomedical")]
    pub fn create_biomedical_model() -> crate::BiomedicalEmbedding {
        let config = crate::BiomedicalEmbeddingConfig::default();
        crate::BiomedicalEmbedding::new(config)
    }

    /// Quick function to compute similarity between two embedding vectors
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(anyhow!(
                "Vector dimensions don't match: {} vs {}",
                a.len(),
                b.len()
            ));
        }

        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Generate sample knowledge graph data for testing
    pub fn generate_sample_kg_data(
        num_entities: usize,
        num_relations: usize,
    ) -> Vec<(String, String, String)> {
        let mut rng = thread_rng();
        let mut triples = Vec::new();

        let entities: Vec<String> = (0..num_entities).map(|i| format!("entity_{i}")).collect();

        let relations: Vec<String> = (0..num_relations)
            .map(|i| format!("relation_{i}"))
            .collect();

        // Generate random triples
        for _ in 0..(num_entities * 2) {
            let subject = entities.choose(&mut rng).unwrap().clone();
            let relation = relations.choose(&mut rng).unwrap().clone();
            let object = entities.choose(&mut rng).unwrap().clone();

            if subject != object {
                triples.push((subject, relation, object));
            }
        }

        triples
    }

    /// Quick performance test function
    pub fn quick_performance_test<F>(
        name: &str,
        iterations: usize,
        operation: F,
    ) -> std::time::Duration
    where
        F: Fn() -> (),
    {
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            operation();
        }
        let duration = start.elapsed();

        println!(
            "Performance test '{}': {} iterations in {:?} ({:.2} ops/sec)",
            name,
            iterations,
            duration,
            iterations as f64 / duration.as_secs_f64()
        );

        duration
    }
}

/// Advanced performance utilities for embedding operations
pub mod performance_utils {
    use super::*;
    use std::time::Instant;

    /// Memory-efficient batch processor for large datasets
    pub struct BatchProcessor<T> {
        batch_size: usize,
        current_batch: Vec<T>,
        processor_fn: Box<dyn Fn(&[T]) -> Result<()> + Send + Sync>,
    }

    impl<T> BatchProcessor<T> {
        pub fn new<F>(batch_size: usize, processor_fn: F) -> Self
        where
            F: Fn(&[T]) -> Result<()> + Send + Sync + 'static,
        {
            Self {
                batch_size,
                current_batch: Vec::with_capacity(batch_size),
                processor_fn: Box::new(processor_fn),
            }
        }

        pub fn add(&mut self, item: T) -> Result<()> {
            self.current_batch.push(item);
            if self.current_batch.len() >= self.batch_size {
                return self.flush();
            }
            Ok(())
        }

        pub fn flush(&mut self) -> Result<()> {
            if !self.current_batch.is_empty() {
                (self.processor_fn)(&self.current_batch)?;
                self.current_batch.clear();
            }
            Ok(())
        }
    }

    /// Enhanced memory monitoring for embedding operations
    #[derive(Debug, Clone)]
    pub struct MemoryMonitor {
        peak_usage: usize,
        current_usage: usize,
        allocations: usize,
        deallocations: usize,
    }

    impl MemoryMonitor {
        pub fn new() -> Self {
            Self {
                peak_usage: 0,
                current_usage: 0,
                allocations: 0,
                deallocations: 0,
            }
        }

        pub fn record_allocation(&mut self, size: usize) {
            self.current_usage += size;
            self.allocations += 1;
            if self.current_usage > self.peak_usage {
                self.peak_usage = self.current_usage;
            }
        }

        pub fn record_deallocation(&mut self, size: usize) {
            self.current_usage = self.current_usage.saturating_sub(size);
            self.deallocations += 1;
        }

        pub fn peak_usage(&self) -> usize {
            self.peak_usage
        }

        pub fn current_usage(&self) -> usize {
            self.current_usage
        }

        pub fn allocation_count(&self) -> usize {
            self.allocations
        }
    }

    impl Default for MemoryMonitor {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Parallel processing utilities for embedding operations
pub mod parallel_utils {
    use super::*;
    use rayon::prelude::*;

    /// Parallel computation of embedding similarities
    pub fn parallel_cosine_similarities(
        query_embedding: &[f32],
        candidate_embeddings: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        let similarities: Vec<f32> = candidate_embeddings
            .par_iter()
            .map(|embedding| oxirs_vec::similarity::cosine_similarity(query_embedding, embedding))
            .collect();
        Ok(similarities)
    }

    /// Parallel batch processing with configurable thread pool
    pub fn parallel_batch_process<T, R, F>(
        items: &[T],
        batch_size: usize,
        processor: F,
    ) -> Result<Vec<R>>
    where
        T: Sync,
        R: Send,
        F: Fn(&[T]) -> Result<Vec<R>> + Sync,
    {
        let results: Result<Vec<Vec<R>>> = items
            .par_chunks(batch_size)
            .map(|chunk| processor(chunk))
            .collect();

        Ok(results?.into_iter().flatten().collect())
    }

    /// Parallel graph analysis with optimized memory usage
    pub fn parallel_entity_frequencies(
        triples: &[(String, String, String)],
    ) -> HashMap<String, usize> {
        let entity_counts: HashMap<String, usize> = triples
            .par_iter()
            .fold(HashMap::new, |mut acc, (subject, _predicate, object)| {
                *acc.entry(subject.clone()).or_insert(0) += 1;
                *acc.entry(object.clone()).or_insert(0) += 1;
                acc
            })
            .reduce(HashMap::new, |mut acc1, acc2| {
                for (entity, count) in acc2 {
                    *acc1.entry(entity).or_insert(0) += count;
                }
                acc1
            });
        entity_counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quick_start::{
        add_triples_from_strings, cosine_similarity, create_simple_transe_model,
        generate_sample_kg_data, parse_triple_from_string, quick_performance_test,
    };
    use crate::EmbeddingModel;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_triples_from_tsv() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "subject\tpredicate\tobject")?;
        writeln!(temp_file, "alice\tknows\tbob")?;
        writeln!(temp_file, "bob\tlikes\tcharlie")?;

        let triples = data_loader::load_triples_from_tsv(temp_file.path())?;
        assert_eq!(triples.len(), 2);
        assert_eq!(
            triples[0],
            ("alice".to_string(), "knows".to_string(), "bob".to_string())
        );

        Ok(())
    }

    #[test]
    fn test_dataset_split() -> Result<()> {
        let triples = vec![
            ("a".to_string(), "r1".to_string(), "b".to_string()),
            ("b".to_string(), "r2".to_string(), "c".to_string()),
            ("c".to_string(), "r3".to_string(), "d".to_string()),
            ("d".to_string(), "r4".to_string(), "e".to_string()),
        ];

        let split = dataset_splitter::split_dataset(triples, 0.6, 0.2, Some(42))?;

        assert_eq!(split.train.len(), 2);
        assert_eq!(split.validation.len(), 0); // 0.2 * 4 = 0.8, rounded down
        assert_eq!(split.test.len(), 2);

        Ok(())
    }

    #[test]
    fn test_load_triples_from_jsonl() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(
            temp_file,
            r#"{{"subject": "alice", "predicate": "knows", "object": "bob"}}"#
        )?;
        writeln!(
            temp_file,
            r#"{{"subject": "bob", "predicate": "likes", "object": "charlie"}}"#
        )?;

        let triples = data_loader::load_triples_from_jsonl(temp_file.path())?;
        assert_eq!(triples.len(), 2);
        assert_eq!(
            triples[0],
            ("alice".to_string(), "knows".to_string(), "bob".to_string())
        );

        Ok(())
    }

    #[test]
    fn test_save_triples_to_jsonl() -> Result<()> {
        let triples = vec![
            ("alice".to_string(), "knows".to_string(), "bob".to_string()),
            (
                "bob".to_string(),
                "likes".to_string(),
                "charlie".to_string(),
            ),
        ];

        let temp_file = NamedTempFile::new()?;
        data_loader::save_triples_to_jsonl(&triples, temp_file.path())?;

        let loaded_triples = data_loader::load_triples_from_jsonl(temp_file.path())?;
        assert_eq!(loaded_triples, triples);

        Ok(())
    }

    #[test]
    fn test_load_triples_auto_detect() -> Result<()> {
        // Test TSV auto-detection
        let mut tsv_file = NamedTempFile::with_suffix(".tsv")?;
        writeln!(tsv_file, "subject\tpredicate\tobject")?;
        writeln!(tsv_file, "alice\tknows\tbob")?;

        let triples = data_loader::load_triples_auto_detect(tsv_file.path())?;
        assert_eq!(triples.len(), 1);

        // Test JSON Lines auto-detection
        let mut jsonl_file = NamedTempFile::with_suffix(".jsonl")?;
        writeln!(
            jsonl_file,
            r#"{{"subject": "alice", "predicate": "knows", "object": "bob"}}"#
        )?;

        let triples = data_loader::load_triples_auto_detect(jsonl_file.path())?;
        assert_eq!(triples.len(), 1);
        assert_eq!(
            triples[0],
            ("alice".to_string(), "knows".to_string(), "bob".to_string())
        );

        Ok(())
    }

    #[test]
    fn test_dataset_statistics() {
        let triples = vec![
            ("alice".to_string(), "knows".to_string(), "bob".to_string()),
            (
                "bob".to_string(),
                "knows".to_string(),
                "charlie".to_string(),
            ),
            (
                "alice".to_string(),
                "likes".to_string(),
                "charlie".to_string(),
            ),
        ];

        let stats = compute_dataset_statistics(&triples);

        assert_eq!(stats.num_triples, 3);
        assert_eq!(stats.num_entities, 3); // alice, bob, charlie
        assert_eq!(stats.num_relations, 2); // knows, likes
        assert!(stats.avg_degree > 0.0);
    }

    // Tests for convenience functions
    #[test]
    fn test_create_simple_transe_model() {
        let model = create_simple_transe_model();
        assert_eq!(model.config().dimensions, 128);
        assert_eq!(model.config().learning_rate, 0.01);
        assert_eq!(model.config().max_epochs, 100);
    }

    #[test]
    fn test_parse_triple_from_string() -> Result<()> {
        let triple = parse_triple_from_string("alice knows bob")?;
        assert_eq!(triple.subject.iri.as_str(), "http://example.org/alice");
        assert_eq!(triple.predicate.iri.as_str(), "http://example.org/knows");
        assert_eq!(triple.object.iri.as_str(), "http://example.org/bob");

        // Test with full URIs
        let triple2 = parse_triple_from_string(
            "http://example.org/alice http://example.org/knows http://example.org/bob",
        )?;
        assert_eq!(triple2.subject.iri.as_str(), "http://example.org/alice");

        // Test invalid format
        assert!(parse_triple_from_string("alice knows").is_err());

        Ok(())
    }

    #[test]
    fn test_add_triples_from_strings() -> Result<()> {
        let mut model = create_simple_transe_model();
        let triple_strings = &[
            "alice knows bob",
            "bob likes charlie",
            "charlie follows alice",
        ];

        let added_count = add_triples_from_strings(&mut model, triple_strings)?;
        assert_eq!(added_count, 3);

        Ok(())
    }

    #[test]
    fn test_cosine_similarity() -> Result<()> {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&a, &b)?;
        assert!((similarity - 1.0).abs() < 1e-10);

        let c = vec![0.0, 1.0, 0.0];
        let similarity2 = cosine_similarity(&a, &c)?;
        assert!((similarity2 - 0.0).abs() < 1e-10);

        // Test different dimensions
        let d = vec![1.0, 0.0];
        assert!(cosine_similarity(&a, &d).is_err());

        Ok(())
    }

    #[test]
    fn test_generate_sample_kg_data() {
        let triples = generate_sample_kg_data(5, 3);
        assert!(!triples.is_empty());

        // Check that all subjects and objects are in the expected format
        for (subject, relation, object) in &triples {
            assert!(subject.starts_with("http://example.org/entity_"));
            assert!(relation.starts_with("http://example.org/relation_"));
            assert!(object.starts_with("http://example.org/entity_"));
            assert_ne!(subject, object); // No self-loops
        }
    }

    #[test]
    fn test_quick_performance_test() {
        let duration = quick_performance_test("test_operation", 100, || {
            // Simple operation for testing
            let _sum: i32 = (1..10).sum();
        });

        assert!(duration.as_nanos() > 0);
    }
}
