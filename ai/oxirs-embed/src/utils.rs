//! Utility functions and helpers for embedding operations

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use oxirs_core::Triple;
use oxirs_vec::Vector;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

/// Data loading utilities
pub mod data_loader {
    use super::*;
    use std::io::{BufRead, BufReader};
    
    /// Load triples from TSV file format
    pub fn load_triples_from_tsv<P: AsRef<Path>>(file_path: P) -> Result<Vec<(String, String, String)>> {
        let file = fs::File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut triples = Vec::new();
        
        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() || line.starts_with('#') {
                continue; // Skip empty lines and comments
            }
            
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let subject = parts[0].trim().to_string();
                let predicate = parts[1].trim().to_string();
                let object = parts[2].trim().to_string();
                triples.push((subject, predicate, object));
            } else {
                eprintln!("Warning: Invalid triple format at line {}: {}", line_num + 1, line);
            }
        }
        
        Ok(triples)
    }
    
    /// Load triples from CSV file format
    pub fn load_triples_from_csv<P: AsRef<Path>>(file_path: P) -> Result<Vec<(String, String, String)>> {
        let file = fs::File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut triples = Vec::new();
        let mut is_first_line = true;
        
        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            if is_first_line {
                is_first_line = false;
                // Skip header if it looks like one
                if line.to_lowercase().contains("subject") && line.to_lowercase().contains("predicate") {
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
                eprintln!("Warning: Invalid triple format at line {}: {}", line_num + 1, line);
            }
        }
        
        Ok(triples)
    }
    
    /// Load triples from N-Triples format
    pub fn load_triples_from_ntriples<P: AsRef<Path>>(file_path: P) -> Result<Vec<(String, String, String)>> {
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
                eprintln!("Warning: Failed to parse N-Triple at line {}: {}", line_num + 1, line);
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
            term[1..term.len()-1].to_string()
        } else if term.starts_with('"') && term.contains('"') {
            // Handle literals - just take the string part for now
            let end_quote = term.rfind('"').unwrap_or(term.len());
            term[1..end_quote].to_string()
        } else {
            term.to_string()
        }
    }
    
    /// Save triples to TSV format
    pub fn save_triples_to_tsv<P: AsRef<Path>>(
        triples: &[(String, String, String)],
        file_path: P,
    ) -> Result<()> {
        let mut content = String::new();
        content.push_str("subject\tpredicate\tobject\n");
        
        for (subject, predicate, object) in triples {
            content.push_str(&format!("{}\t{}\t{}\n", subject, predicate, object));
        }
        
        fs::write(file_path, content)?;
        Ok(())
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
            return Err(anyhow!("Train and validation ratios must sum to less than 1.0"));
        }
        
        let mut rng = if let Some(s) = seed {
            StdRng::seed_from_u64(s)
        } else {
            StdRng::from_entropy()
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
        // Group triples by entities
        let mut entity_triples: HashMap<String, Vec<(String, String, String)>> = HashMap::new();
        
        for triple in triples {
            let entities = vec![&triple.0, &triple.2];
            for entity in entities {
                entity_triples.entry(entity.clone()).or_default().push(triple.clone());
            }
        }
        
        // Split entities first, then assign triples
        let entities: Vec<String> = entity_triples.keys().cloned().collect();
        let entity_split = split_dataset(
            entities.into_iter().map(|e| (e.clone(), "dummy".to_string(), "dummy".to_string())).collect(),
            train_ratio,
            val_ratio,
            seed,
        )?;
        
        let train_entities: HashSet<String> = entity_split.train.into_iter().map(|(e, _, _)| e).collect();
        let val_entities: HashSet<String> = entity_split.validation.into_iter().map(|(e, _, _)| e).collect();
        let test_entities: HashSet<String> = entity_split.test.into_iter().map(|(e, _, _)| e).collect();
        
        // Assign triples based on entity membership
        let mut train_triples = Vec::new();
        let mut val_triples = Vec::new();
        let mut test_triples = Vec::new();
        
        for (entity, triples) in entity_triples {
            if train_entities.contains(&entity) {
                train_triples.extend(triples);
            } else if val_entities.contains(&entity) {
                val_triples.extend(triples);
            } else if test_entities.contains(&entity) {
                test_triples.extend(triples);
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
        let variance = flat_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / flat_values.len() as f64;
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
    pub fn analyze_embedding_similarities(embeddings: &Array2<f64>, sample_size: usize) -> SimilarityStats {
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
    
    /// Compute graph metrics for knowledge graph
    pub fn compute_graph_metrics(triples: &[(String, String, String)]) -> GraphMetrics {
        let mut entity_degrees: HashMap<String, usize> = HashMap::new();
        let mut relation_counts: HashMap<String, usize> = HashMap::new();
        let mut entities = HashSet::new();
        
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
        
        println!("Progress: {}/{} ({:.1}%) - {:.1} items/sec", 
                 self.current, self.total, percentage, rate);
    }
    
    /// Finish and print final statistics
    pub fn finish(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = self.total as f64 / elapsed;
        
        println!("Completed: {} items in {:.2}s ({:.1} items/sec)", 
                 self.total, elapsed, rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    #[test]
    fn test_load_triples_from_tsv() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "subject\tpredicate\tobject")?;
        writeln!(temp_file, "alice\tknows\tbob")?;
        writeln!(temp_file, "bob\tlikes\tcharlie")?;
        
        let triples = data_loader::load_triples_from_tsv(temp_file.path())?;
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0], ("alice".to_string(), "knows".to_string(), "bob".to_string()));
        
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
    fn test_dataset_statistics() {
        let triples = vec![
            ("alice".to_string(), "knows".to_string(), "bob".to_string()),
            ("bob".to_string(), "knows".to_string(), "charlie".to_string()),
            ("alice".to_string(), "likes".to_string(), "charlie".to_string()),
        ];
        
        let stats = compute_dataset_statistics(&triples);
        
        assert_eq!(stats.num_triples, 3);
        assert_eq!(stats.num_entities, 3); // alice, bob, charlie
        assert_eq!(stats.num_relations, 2); // knows, likes
        assert!(stats.avg_degree > 0.0);
    }
}