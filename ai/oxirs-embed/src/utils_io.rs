//! Serialization/deserialization helpers and file I/O utilities for embeddings.

use crate::utils_types::DatasetSplit;
use anyhow::{anyhow, Result};
use scirs2_core::random::Random;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Data loading utilities
pub mod data_loader {
    use super::*;

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
                continue;
            }

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

    fn clean_uri_or_literal(term: &str) -> String {
        if term.starts_with('<') && term.ends_with('>') {
            term[1..term.len() - 1].to_string()
        } else if term.starts_with('"') && term.contains('"') {
            let end_quote = term.rfind('"').unwrap_or(term.len());
            term[1..end_quote].to_string()
        } else {
            term.to_string()
        }
    }

    /// Load triples from JSON Lines format
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
                eprintln!(
                    "Warning: Unknown file extension '{extension}', attempting auto-detection"
                );

                if let Ok(triples) = load_triples_from_tsv(path) {
                    if !triples.is_empty() {
                        return Ok(triples);
                    }
                }

                if let Ok(triples) = load_triples_from_ntriples(path) {
                    if !triples.is_empty() {
                        return Ok(triples);
                    }
                }

                if let Ok(triples) = load_triples_from_jsonl(path) {
                    if !triples.is_empty() {
                        return Ok(triples);
                    }
                }

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
            Random::seed(s)
        } else {
            Random::seed(42)
        };

        let mut shuffled_triples = triples;
        for i in (1..shuffled_triples.len()).rev() {
            let j = rng.random_range(0..i + 1);
            shuffled_triples.swap(i, j);
        }

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
        let mut entity_triples: HashMap<String, Vec<(String, String, String)>> =
            HashMap::with_capacity(triples.len() / 2);

        for triple in &triples {
            let entities = [&triple.0, &triple.2];
            for entity in entities {
                entity_triples
                    .entry(entity.clone())
                    .or_default()
                    .push(triple.clone());
            }
        }

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

/// Parallel processing utilities for embedding operations
pub mod parallel_utils {
    use anyhow::Result;
    use rayon::prelude::*;
    use std::collections::HashMap;

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
        F: Fn(&[T]) -> Result<Vec<R>> + Sync + Send,
    {
        let results: Result<Vec<Vec<R>>> = items.par_chunks(batch_size).map(processor).collect();
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
