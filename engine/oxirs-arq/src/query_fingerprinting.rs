//! Advanced Query Fingerprinting for SPARQL Query Analysis
//!
//! This module provides sophisticated query fingerprinting for:
//! - Query deduplication and caching
//! - Query pattern recognition
//! - Workload analysis
//! - Query similarity detection
//!
//! # Features
//!
//! - **Structural Fingerprinting**: Captures query structure independent of literal values
//! - **Semantic Fingerprinting**: Incorporates predicate and type information
//! - **Normalized Fingerprints**: Canonicalized representation for comparison
//! - **Parameterized Templates**: Extracts query templates with parameter slots
//! - **Similarity Metrics**: Computes query similarity scores
//!
//! # Example
//!
//! ```rust
//! use oxirs_arq::query_fingerprinting::{QueryFingerprinter, FingerprintConfig};
//!
//! let config = FingerprintConfig::default();
//! let fingerprinter = QueryFingerprinter::new(config);
//!
//! let query1 = "SELECT ?s WHERE { ?s <http://example.org/name> \"Alice\" }";
//! let query2 = "SELECT ?s WHERE { ?s <http://example.org/name> \"Bob\" }";
//!
//! let fp1 = fingerprinter.fingerprint(query1).unwrap();
//! let fp2 = fingerprinter.fingerprint(query2).unwrap();
//!
//! // Same structure, different literals -> similar fingerprints
//! let similarity = fingerprinter.similarity(&fp1, &fp2);
//! println!("Similarity: {:.2}", similarity);
//! ```

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

/// Configuration for query fingerprinting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintConfig {
    /// Include variable names in fingerprint
    pub include_variables: bool,
    /// Normalize literal values (replace with placeholders)
    pub normalize_literals: bool,
    /// Normalize numeric values
    pub normalize_numbers: bool,
    /// Normalize IRI local names
    pub normalize_iri_locals: bool,
    /// Preserve predicate IRIs
    pub preserve_predicates: bool,
    /// Preserve type assertions (rdf:type)
    pub preserve_types: bool,
    /// Hash algorithm to use
    pub hash_algorithm: HashAlgorithm,
    /// Maximum fingerprint cache size
    pub cache_size: usize,
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self {
            include_variables: false,
            normalize_literals: true,
            normalize_numbers: true,
            normalize_iri_locals: true,
            preserve_predicates: true,
            preserve_types: true,
            hash_algorithm: HashAlgorithm::Sha256,
            cache_size: 10000,
        }
    }
}

/// Hash algorithm options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashAlgorithm {
    /// MD5 (fast, 128-bit)
    Md5,
    /// SHA-256 (secure, 256-bit)
    Sha256,
    /// FNV-1a (very fast, 64-bit)
    Fnv1a,
}

/// Query fingerprint
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryFingerprint {
    /// Primary hash of normalized query structure
    pub hash: String,
    /// Short hash for quick comparison
    pub short_hash: String,
    /// Query template with parameter placeholders
    pub template: String,
    /// Extracted parameters
    pub parameters: Vec<ParameterSlot>,
    /// Query form (SELECT, ASK, CONSTRUCT, DESCRIBE)
    pub query_form: QueryForm,
    /// Structural features
    pub features: QueryFeatures,
    /// Original query length
    pub original_length: usize,
}

impl QueryFingerprint {
    /// Get the template ID (short hash)
    pub fn template_id(&self) -> &str {
        &self.short_hash
    }

    /// Check if this fingerprint matches another (same template)
    pub fn matches_template(&self, other: &Self) -> bool {
        self.short_hash == other.short_hash
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    /// Check if query has parameters
    pub fn is_parameterized(&self) -> bool {
        !self.parameters.is_empty()
    }
}

/// Parameter slot in a query template
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParameterSlot {
    /// Slot name ($1, $2, etc.)
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Position in original query
    pub position: usize,
    /// Original value (if extracted)
    pub original_value: Option<String>,
}

/// Types of parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParameterType {
    /// String literal
    StringLiteral,
    /// Numeric literal
    NumericLiteral,
    /// Date/time literal
    DateTimeLiteral,
    /// IRI reference
    Iri,
    /// Boolean literal
    Boolean,
    /// Language-tagged literal
    LangLiteral,
    /// Unknown type
    Unknown,
}

/// Query form types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryForm {
    /// SELECT query
    Select,
    /// ASK query
    Ask,
    /// CONSTRUCT query
    Construct,
    /// DESCRIBE query
    Describe,
    /// INSERT/DELETE (update)
    Update,
    /// Unknown
    Unknown,
}

impl QueryForm {
    fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "SELECT" => QueryForm::Select,
            "ASK" => QueryForm::Ask,
            "CONSTRUCT" => QueryForm::Construct,
            "DESCRIBE" => QueryForm::Describe,
            "INSERT" | "DELETE" => QueryForm::Update,
            _ => QueryForm::Unknown,
        }
    }
}

/// Structural features of a query
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryFeatures {
    /// Number of triple patterns
    pub triple_pattern_count: usize,
    /// Number of variables
    pub variable_count: usize,
    /// Number of filters
    pub filter_count: usize,
    /// Number of OPTIONAL blocks
    pub optional_count: usize,
    /// Number of UNION blocks
    pub union_count: usize,
    /// Number of subqueries
    pub subquery_count: usize,
    /// Has GROUP BY
    pub has_group_by: bool,
    /// Has ORDER BY
    pub has_order_by: bool,
    /// Has LIMIT
    pub has_limit: bool,
    /// Has OFFSET
    pub has_offset: bool,
    /// Has DISTINCT
    pub has_distinct: bool,
    /// Has SERVICE (federated)
    pub has_service: bool,
    /// Has VALUES clause
    pub has_values: bool,
    /// Has property paths
    pub has_property_paths: bool,
    /// Has aggregates
    pub has_aggregates: bool,
    /// Has BIND
    pub has_bind: bool,
    /// Has MINUS
    pub has_minus: bool,
    /// Estimated complexity score
    pub complexity_score: u32,
}

/// Query fingerprinter
pub struct QueryFingerprinter {
    /// Configuration
    config: FingerprintConfig,
    /// Fingerprint cache
    cache: Arc<RwLock<HashMap<String, QueryFingerprint>>>,
    /// Statistics
    stats: FingerprintStats,
}

/// Fingerprinting statistics
#[derive(Debug, Default)]
struct FingerprintStats {
    /// Total fingerprints computed
    computed: AtomicU64,
    /// Cache hits
    cache_hits: AtomicU64,
    /// Cache misses
    cache_misses: AtomicU64,
}

impl QueryFingerprinter {
    /// Create new fingerprinter
    pub fn new(config: FingerprintConfig) -> Self {
        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: FingerprintStats::default(),
        }
    }

    /// Compute fingerprint for a query
    pub fn fingerprint(&self, query: &str) -> Result<QueryFingerprint> {
        // Check cache first
        let query_hash = self.quick_hash(query);
        {
            let cache = self.cache.read().unwrap();
            if let Some(fp) = cache.get(&query_hash) {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(fp.clone());
            }
        }

        self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);

        // Compute fingerprint
        let fingerprint = self.compute_fingerprint(query)?;

        // Update cache
        {
            let mut cache = self.cache.write().unwrap();
            if cache.len() >= self.config.cache_size {
                // Simple eviction: clear half the cache
                let keys_to_remove: Vec<_> = cache.keys().take(cache.len() / 2).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
            cache.insert(query_hash, fingerprint.clone());
        }

        self.stats.computed.fetch_add(1, Ordering::Relaxed);
        Ok(fingerprint)
    }

    /// Compute fingerprint without caching
    fn compute_fingerprint(&self, query: &str) -> Result<QueryFingerprint> {
        // Extract query form
        let query_form = self.extract_query_form(query);

        // Normalize the query
        let (normalized, parameters) = self.normalize_query(query)?;

        // Extract structural features
        let features = self.extract_features(query);

        // Compute hash
        let hash = self.compute_hash(&normalized);
        let short_hash = hash[..16].to_string();

        Ok(QueryFingerprint {
            hash,
            short_hash,
            template: normalized,
            parameters,
            query_form,
            features,
            original_length: query.len(),
        })
    }

    /// Extract query form
    fn extract_query_form(&self, query: &str) -> QueryForm {
        let trimmed = query.trim_start();
        let first_word = trimmed.split_whitespace().next().unwrap_or("");
        QueryForm::from_str(first_word)
    }

    /// Normalize query and extract parameters
    fn normalize_query(&self, query: &str) -> Result<(String, Vec<ParameterSlot>)> {
        let mut normalized = query.to_string();
        let mut parameters = Vec::new();
        let mut param_index = 0;

        // Remove comments
        normalized = remove_comments(&normalized);

        // Normalize whitespace
        normalized = normalize_whitespace(&normalized);

        // Normalize literals
        if self.config.normalize_literals {
            let (new_query, new_params) =
                self.normalize_string_literals(&normalized, param_index)?;
            normalized = new_query;
            param_index += new_params.len();
            parameters.extend(new_params);
        }

        // Normalize numbers
        if self.config.normalize_numbers {
            let (new_query, new_params) =
                self.normalize_numeric_literals(&normalized, param_index)?;
            normalized = new_query;
            param_index += new_params.len();
            parameters.extend(new_params);
        }

        // Normalize IRI locals (unless predicate or type)
        if self.config.normalize_iri_locals {
            let (new_query, new_params) = self.normalize_iri_locals(&normalized, param_index)?;
            normalized = new_query;
            parameters.extend(new_params);
        }

        // Normalize variable names if configured
        if !self.config.include_variables {
            normalized = self.normalize_variables(&normalized);
        }

        // Final cleanup
        normalized = normalize_whitespace(&normalized);

        Ok((normalized, parameters))
    }

    /// Normalize string literals
    fn normalize_string_literals(
        &self,
        query: &str,
        start_index: usize,
    ) -> Result<(String, Vec<ParameterSlot>)> {
        let string_pattern = string_literal_regex();
        let mut result = query.to_string();
        let mut params = Vec::new();
        let mut index = start_index;

        // Find all string literals
        let matches: Vec<_> = string_pattern.find_iter(query).collect();

        // Replace from end to preserve positions
        for m in matches.into_iter().rev() {
            let original = m.as_str().to_string();
            let param_name = format!("${}", index);

            // Determine type
            let param_type = if original.contains("@") {
                ParameterType::LangLiteral
            } else if original.contains("^^") {
                if original.contains("dateTime") || original.contains("date") {
                    ParameterType::DateTimeLiteral
                } else {
                    ParameterType::StringLiteral
                }
            } else {
                ParameterType::StringLiteral
            };

            params.push(ParameterSlot {
                name: param_name.clone(),
                param_type,
                position: m.start(),
                original_value: Some(original),
            });

            result = format!(
                "{}{}{}",
                &result[..m.start()],
                param_name,
                &result[m.end()..]
            );
            index += 1;
        }

        params.reverse(); // Correct order
        Ok((result, params))
    }

    /// Normalize numeric literals
    fn normalize_numeric_literals(
        &self,
        query: &str,
        start_index: usize,
    ) -> Result<(String, Vec<ParameterSlot>)> {
        let number_pattern = numeric_literal_regex();
        let mut result = query.to_string();
        let mut params = Vec::new();
        let mut index = start_index;

        let matches: Vec<_> = number_pattern.find_iter(query).collect();

        for m in matches.into_iter().rev() {
            let original = m.as_str().to_string();
            let param_name = format!("${}", index);

            params.push(ParameterSlot {
                name: param_name.clone(),
                param_type: ParameterType::NumericLiteral,
                position: m.start(),
                original_value: Some(original),
            });

            result = format!(
                "{}{}{}",
                &result[..m.start()],
                param_name,
                &result[m.end()..]
            );
            index += 1;
        }

        params.reverse();
        Ok((result, params))
    }

    /// Normalize IRI local names
    fn normalize_iri_locals(
        &self,
        query: &str,
        start_index: usize,
    ) -> Result<(String, Vec<ParameterSlot>)> {
        // Skip predicates and type IRIs if configured
        let iri_pattern = iri_local_regex();
        let mut result = query.to_string();
        let mut params = Vec::new();
        let mut index = start_index;

        let matches: Vec<_> = iri_pattern.find_iter(query).collect();

        for m in matches.into_iter().rev() {
            let iri = m.as_str();

            // Check if this should be preserved
            if self.config.preserve_predicates && is_likely_predicate(query, m.start()) {
                continue;
            }
            if self.config.preserve_types && iri.contains("rdf:type") || iri.contains("#type") {
                continue;
            }

            let param_name = format!("${}", index);

            params.push(ParameterSlot {
                name: param_name.clone(),
                param_type: ParameterType::Iri,
                position: m.start(),
                original_value: Some(iri.to_string()),
            });

            // Replace only the local name part
            if let Some(local_start) = iri.rfind('#').or_else(|| iri.rfind('/')) {
                let prefix = &iri[..=local_start];
                result = format!(
                    "{}{}{}{}",
                    &result[..m.start()],
                    prefix,
                    param_name,
                    &result[m.end()..]
                );
            }
            index += 1;
        }

        params.reverse();
        Ok((result, params))
    }

    /// Normalize variable names
    fn normalize_variables(&self, query: &str) -> String {
        let var_pattern = variable_regex();
        let mut result = query.to_string();
        let mut var_mapping: HashMap<String, String> = HashMap::new();
        let mut var_index = 0;

        // Find all variables and create mapping
        for m in var_pattern.find_iter(query) {
            let var_name = m.as_str().to_string();
            if !var_mapping.contains_key(&var_name) {
                var_mapping.insert(var_name.clone(), format!("?v{}", var_index));
                var_index += 1;
            }
        }

        // Replace variables (sorted by length descending to avoid partial replacements)
        let mut sorted_vars: Vec<_> = var_mapping.iter().collect();
        sorted_vars.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        for (original, normalized) in sorted_vars {
            result = result.replace(original, normalized);
        }

        result
    }

    /// Extract structural features
    fn extract_features(&self, query: &str) -> QueryFeatures {
        let upper_query = query.to_uppercase();

        let triple_pattern_count = count_triple_patterns(query);
        let variable_count = count_variables(query);
        let filter_count = upper_query.matches("FILTER").count();
        let optional_count = upper_query.matches("OPTIONAL").count();
        let union_count = upper_query.matches("UNION").count();
        let subquery_count = upper_query.matches("SELECT").count().saturating_sub(1);

        let has_group_by = upper_query.contains("GROUP BY");
        let has_order_by = upper_query.contains("ORDER BY");
        let has_limit = upper_query.contains("LIMIT");
        let has_offset = upper_query.contains("OFFSET");
        let has_distinct = upper_query.contains("DISTINCT");
        let has_service = upper_query.contains("SERVICE");
        let has_values = upper_query.contains("VALUES");
        let has_property_paths = query.contains("/")
            || query.contains("*")
            || query.contains("+")
            || query.contains("|");
        let has_aggregates = upper_query.contains("COUNT")
            || upper_query.contains("SUM")
            || upper_query.contains("AVG")
            || upper_query.contains("MIN")
            || upper_query.contains("MAX");
        let has_bind = upper_query.contains("BIND");
        let has_minus = upper_query.contains("MINUS");

        // Calculate complexity score
        let complexity_score = (triple_pattern_count * 10
            + filter_count * 5
            + optional_count * 8
            + union_count * 6
            + subquery_count * 15
            + if has_group_by { 5 } else { 0 }
            + if has_order_by { 3 } else { 0 }
            + if has_service { 20 } else { 0 }
            + if has_property_paths { 10 } else { 0 }
            + if has_aggregates { 7 } else { 0 }
            + if has_minus { 5 } else { 0 }) as u32;

        QueryFeatures {
            triple_pattern_count,
            variable_count,
            filter_count,
            optional_count,
            union_count,
            subquery_count,
            has_group_by,
            has_order_by,
            has_limit,
            has_offset,
            has_distinct,
            has_service,
            has_values,
            has_property_paths,
            has_aggregates,
            has_bind,
            has_minus,
            complexity_score,
        }
    }

    /// Compute hash of normalized query
    fn compute_hash(&self, normalized: &str) -> String {
        match self.config.hash_algorithm {
            HashAlgorithm::Md5 => {
                let digest = md5::compute(normalized.as_bytes());
                format!("{:x}", digest)
            }
            HashAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                hasher.update(normalized.as_bytes());
                format!("{:x}", hasher.finalize())
            }
            HashAlgorithm::Fnv1a => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                normalized.hash(&mut hasher);
                format!("{:016x}", hasher.finish())
            }
        }
    }

    /// Quick hash for cache lookup
    fn quick_hash(&self, query: &str) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        query.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Calculate similarity between two fingerprints
    pub fn similarity(&self, fp1: &QueryFingerprint, fp2: &QueryFingerprint) -> f64 {
        // If same hash, they're identical
        if fp1.hash == fp2.hash {
            return 1.0;
        }

        // Different query forms = very different
        if fp1.query_form != fp2.query_form {
            return 0.0;
        }

        // Calculate structural similarity
        let structure_sim = self.structural_similarity(&fp1.features, &fp2.features);

        // Calculate template similarity (string similarity)
        let template_sim = string_similarity(&fp1.template, &fp2.template);

        // Weighted combination
        0.6 * structure_sim + 0.4 * template_sim
    }

    /// Calculate structural similarity between features
    fn structural_similarity(&self, f1: &QueryFeatures, f2: &QueryFeatures) -> f64 {
        let mut matches = 0.0;
        let mut total = 0.0;

        // Compare numeric features
        let num_features = [
            (f1.triple_pattern_count, f2.triple_pattern_count, 2.0),
            (f1.variable_count, f2.variable_count, 1.0),
            (f1.filter_count, f2.filter_count, 1.5),
            (f1.optional_count, f2.optional_count, 1.5),
            (f1.union_count, f2.union_count, 1.5),
            (f1.subquery_count, f2.subquery_count, 2.0),
        ];

        for (v1, v2, weight) in num_features {
            total += weight;
            if v1 == v2 {
                matches += weight;
            } else {
                let max_val = v1.max(v2) as f64;
                let min_val = v1.min(v2) as f64;
                if max_val > 0.0 {
                    matches += weight * (min_val / max_val);
                }
            }
        }

        // Compare boolean features
        let bool_features = [
            (f1.has_group_by, f2.has_group_by),
            (f1.has_order_by, f2.has_order_by),
            (f1.has_limit, f2.has_limit),
            (f1.has_distinct, f2.has_distinct),
            (f1.has_service, f2.has_service),
            (f1.has_values, f2.has_values),
            (f1.has_property_paths, f2.has_property_paths),
            (f1.has_aggregates, f2.has_aggregates),
            (f1.has_bind, f2.has_bind),
            (f1.has_minus, f2.has_minus),
        ];

        for (b1, b2) in bool_features {
            total += 1.0;
            if b1 == b2 {
                matches += 1.0;
            }
        }

        matches / total
    }

    /// Find similar fingerprints in a collection
    pub fn find_similar(
        &self,
        target: &QueryFingerprint,
        fingerprints: &[QueryFingerprint],
        threshold: f64,
    ) -> Vec<(usize, f64)> {
        let mut results: Vec<(usize, f64)> = fingerprints
            .iter()
            .enumerate()
            .filter_map(|(idx, fp)| {
                let sim = self.similarity(target, fp);
                if sim >= threshold {
                    Some((idx, sim))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Group fingerprints by similarity
    pub fn cluster_fingerprints(
        &self,
        fingerprints: &[QueryFingerprint],
        threshold: f64,
    ) -> Vec<Vec<usize>> {
        let n = fingerprints.len();
        let mut visited = vec![false; n];
        let mut clusters = Vec::new();

        for i in 0..n {
            if visited[i] {
                continue;
            }

            let mut cluster = vec![i];
            visited[i] = true;

            for j in (i + 1)..n {
                if visited[j] {
                    continue;
                }

                let sim = self.similarity(&fingerprints[i], &fingerprints[j]);
                if sim >= threshold {
                    cluster.push(j);
                    visited[j] = true;
                }
            }

            clusters.push(cluster);
        }

        clusters
    }

    /// Get fingerprinting statistics
    pub fn statistics(&self) -> FingerprintingStatistics {
        FingerprintingStatistics {
            fingerprints_computed: self.stats.computed.load(Ordering::Relaxed),
            cache_hits: self.stats.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.stats.cache_misses.load(Ordering::Relaxed),
            cache_size: self.cache.read().unwrap().len(),
        }
    }

    /// Clear the fingerprint cache
    pub fn clear_cache(&self) {
        self.cache.write().unwrap().clear();
    }
}

/// Fingerprinting statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintingStatistics {
    /// Total fingerprints computed
    pub fingerprints_computed: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Current cache size
    pub cache_size: usize,
}

impl FingerprintingStatistics {
    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

// Helper functions

fn remove_comments(query: &str) -> String {
    let mut result = String::new();
    let mut in_string = false;
    let mut escape = false;
    let mut chars = query.chars().peekable();

    while let Some(c) = chars.next() {
        if escape {
            result.push(c);
            escape = false;
            continue;
        }

        if c == '\\' && in_string {
            result.push(c);
            escape = true;
            continue;
        }

        if c == '"' && !in_string {
            in_string = true;
            result.push(c);
            continue;
        }

        if c == '"' && in_string {
            in_string = false;
            result.push(c);
            continue;
        }

        if !in_string && c == '#' {
            // Skip to end of line
            while let Some(&nc) = chars.peek() {
                if nc == '\n' {
                    break;
                }
                chars.next();
            }
            result.push(' ');
            continue;
        }

        result.push(c);
    }

    result
}

fn normalize_whitespace(query: &str) -> String {
    let whitespace_pattern = whitespace_regex();
    whitespace_pattern
        .replace_all(query, " ")
        .trim()
        .to_string()
}

fn count_triple_patterns(query: &str) -> usize {
    // Count occurrences of patterns like "?var predicate ?var" or ". ?var predicate"
    let triple_pattern = triple_pattern_regex();
    triple_pattern.find_iter(query).count()
}

fn count_variables(query: &str) -> usize {
    let var_pattern = variable_regex();
    let vars: HashSet<_> = var_pattern.find_iter(query).map(|m| m.as_str()).collect();
    vars.len()
}

fn is_likely_predicate(query: &str, position: usize) -> bool {
    // Check if this IRI is in predicate position (middle of triple pattern)
    // This is a heuristic - look for variable before and after
    let before = &query[..position];
    let after = &query[position..];

    // Check if there's a variable or IRI before (subject)
    let has_subject = before
        .trim_end()
        .ends_with(|c: char| c == '>' || c.is_alphabetic() || c == '_');
    // Check if there's a variable or literal after (object)
    let has_object = after.trim_start().starts_with(['?', '<', '"']);

    has_subject && has_object
}

fn string_similarity(s1: &str, s2: &str) -> f64 {
    // Simple Jaccard similarity on tokens
    let tokens1: HashSet<_> = s1.split_whitespace().collect();
    let tokens2: HashSet<_> = s2.split_whitespace().collect();

    let intersection = tokens1.intersection(&tokens2).count();
    let union = tokens1.union(&tokens2).count();

    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

// Lazy static regex patterns

fn string_literal_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r#""(?:[^"\\]|\\.)*"(?:@[a-zA-Z-]+)?(?:\^\^<[^>]+>)?"#).expect("Invalid regex")
    })
}

fn numeric_literal_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        // Simple numeric pattern without look-around (which regex crate doesn't support)
        // Match numbers that appear as standalone tokens
        Regex::new(r"\b[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\b").expect("Invalid regex")
    })
}

fn iri_local_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"<[^>]+>").expect("Invalid regex"))
}

fn variable_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\?[a-zA-Z_][a-zA-Z0-9_]*").expect("Invalid regex"))
}

fn whitespace_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\s+").expect("Invalid regex"))
}

fn triple_pattern_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r#"(?:\?[a-zA-Z_]\w*|<[^>]+>)\s+(?:\?[a-zA-Z_]\w*|<[^>]+>|[a-zA-Z]+:[a-zA-Z_]\w*)\s+(?:\?[a-zA-Z_]\w*|<[^>]+>|"[^"]*"|[0-9]+)"#).expect("Invalid regex")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_fingerprint() {
        let config = FingerprintConfig::default();
        let fingerprinter = QueryFingerprinter::new(config);

        let query = "SELECT ?s WHERE { ?s <http://example.org/name> \"Alice\" }";
        let fp = fingerprinter.fingerprint(query).unwrap();

        assert_eq!(fp.query_form, QueryForm::Select);
        assert!(!fp.hash.is_empty());
        assert!(!fp.template.is_empty());
    }

    #[test]
    fn test_similar_queries_same_template() {
        let config = FingerprintConfig::default();
        let fingerprinter = QueryFingerprinter::new(config);

        let query1 = "SELECT ?s WHERE { ?s <http://example.org/name> \"Alice\" }";
        let query2 = "SELECT ?s WHERE { ?s <http://example.org/name> \"Bob\" }";

        let fp1 = fingerprinter.fingerprint(query1).unwrap();
        let fp2 = fingerprinter.fingerprint(query2).unwrap();

        // Same structure, different literals
        let similarity = fingerprinter.similarity(&fp1, &fp2);
        assert!(
            similarity > 0.7,
            "Similarity should be high for same structure"
        );
    }

    #[test]
    fn test_different_queries_different_template() {
        let config = FingerprintConfig::default();
        let fingerprinter = QueryFingerprinter::new(config);

        let query1 = "SELECT ?s WHERE { ?s <http://example.org/name> \"Alice\" }";
        let query2 = "ASK { ?x <http://example.org/age> ?y . ?y <http://example.org/value> ?z }";

        let fp1 = fingerprinter.fingerprint(query1).unwrap();
        let fp2 = fingerprinter.fingerprint(query2).unwrap();

        let similarity = fingerprinter.similarity(&fp1, &fp2);
        assert!(
            similarity < 0.5,
            "Similarity should be low for different structures"
        );
    }

    #[test]
    fn test_feature_extraction() {
        let config = FingerprintConfig::default();
        let fingerprinter = QueryFingerprinter::new(config);

        let query = r#"
            SELECT ?s ?name (COUNT(?o) AS ?count)
            WHERE {
                ?s <http://example.org/type> <http://example.org/Person> .
                ?s <http://example.org/name> ?name .
                OPTIONAL { ?s <http://example.org/friend> ?o }
                FILTER(?name != "")
            }
            GROUP BY ?s ?name
            ORDER BY DESC(?count)
            LIMIT 10
        "#;

        let fp = fingerprinter.fingerprint(query).unwrap();

        assert!(fp.features.has_group_by);
        assert!(fp.features.has_order_by);
        assert!(fp.features.has_limit);
        assert!(fp.features.has_aggregates);
        assert!(fp.features.optional_count > 0);
        assert!(fp.features.filter_count > 0);
    }

    #[test]
    fn test_cache_behavior() {
        let config = FingerprintConfig::default();
        let fingerprinter = QueryFingerprinter::new(config);

        let query = "SELECT ?s WHERE { ?s ?p ?o }";

        // First call - cache miss
        let _fp1 = fingerprinter.fingerprint(query).unwrap();
        let stats1 = fingerprinter.statistics();
        assert_eq!(stats1.cache_misses, 1);

        // Second call - cache hit
        let _fp2 = fingerprinter.fingerprint(query).unwrap();
        let stats2 = fingerprinter.statistics();
        assert_eq!(stats2.cache_hits, 1);
    }

    #[test]
    fn test_query_forms() {
        let config = FingerprintConfig::default();
        let fingerprinter = QueryFingerprinter::new(config);

        let select = "SELECT ?s WHERE { ?s ?p ?o }";
        let ask = "ASK { ?s ?p ?o }";
        let construct = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }";
        let describe = "DESCRIBE <http://example.org/thing>";

        assert_eq!(
            fingerprinter.fingerprint(select).unwrap().query_form,
            QueryForm::Select
        );
        assert_eq!(
            fingerprinter.fingerprint(ask).unwrap().query_form,
            QueryForm::Ask
        );
        assert_eq!(
            fingerprinter.fingerprint(construct).unwrap().query_form,
            QueryForm::Construct
        );
        assert_eq!(
            fingerprinter.fingerprint(describe).unwrap().query_form,
            QueryForm::Describe
        );
    }

    #[test]
    fn test_parameter_extraction() {
        let config = FingerprintConfig::default();
        let fingerprinter = QueryFingerprinter::new(config);

        let query = r#"SELECT ?s WHERE { ?s <http://example.org/age> 25 . ?s <http://example.org/name> "Alice" }"#;
        let fp = fingerprinter.fingerprint(query).unwrap();

        assert!(!fp.parameters.is_empty());
    }

    #[test]
    fn test_find_similar() {
        let config = FingerprintConfig::default();
        let fingerprinter = QueryFingerprinter::new(config);

        let queries = [
            "SELECT ?s WHERE { ?s <http://example.org/name> \"Alice\" }",
            "SELECT ?s WHERE { ?s <http://example.org/name> \"Bob\" }",
            "SELECT ?s WHERE { ?s <http://example.org/name> \"Charlie\" }",
            "ASK { ?x <http://example.org/age> ?y }",
        ];

        let fingerprints: Vec<_> = queries
            .iter()
            .map(|q| fingerprinter.fingerprint(q).unwrap())
            .collect();

        let target = &fingerprints[0];
        let similar = fingerprinter.find_similar(target, &fingerprints, 0.7);

        // Should find at least the first 3 as similar
        assert!(similar.len() >= 2);
    }

    #[test]
    fn test_cluster_fingerprints() {
        let config = FingerprintConfig::default();
        let fingerprinter = QueryFingerprinter::new(config);

        let queries = [
            "SELECT ?s WHERE { ?s <http://example.org/name> \"Alice\" }",
            "SELECT ?s WHERE { ?s <http://example.org/name> \"Bob\" }",
            "ASK { ?x <http://example.org/age> ?y }",
            "ASK { ?x <http://example.org/age> ?z }",
        ];

        let fingerprints: Vec<_> = queries
            .iter()
            .map(|q| fingerprinter.fingerprint(q).unwrap())
            .collect();

        let clusters = fingerprinter.cluster_fingerprints(&fingerprints, 0.7);

        // Should have 2 clusters: one for SELECT queries, one for ASK queries
        assert!(clusters.len() >= 2);
    }

    #[test]
    fn test_complexity_score() {
        let config = FingerprintConfig::default();
        let fingerprinter = QueryFingerprinter::new(config);

        let simple = "SELECT ?s WHERE { ?s ?p ?o }";
        let complex = r#"
            SELECT ?s ?name (COUNT(?friend) AS ?friendCount)
            WHERE {
                ?s <http://example.org/type> <http://example.org/Person> .
                ?s <http://example.org/name> ?name .
                OPTIONAL { ?s <http://example.org/friend> ?friend }
                FILTER(?name != "")
                UNION {
                    ?s <http://example.org/nickname> ?name
                }
            }
            GROUP BY ?s ?name
            ORDER BY DESC(?friendCount)
            LIMIT 100
        "#;

        let fp_simple = fingerprinter.fingerprint(simple).unwrap();
        let fp_complex = fingerprinter.fingerprint(complex).unwrap();

        assert!(fp_complex.features.complexity_score > fp_simple.features.complexity_score);
    }

    #[test]
    fn test_hash_algorithms() {
        let queries = "SELECT ?s WHERE { ?s ?p ?o }";

        // MD5
        let config_md5 = FingerprintConfig {
            hash_algorithm: HashAlgorithm::Md5,
            ..Default::default()
        };
        let fp_md5 = QueryFingerprinter::new(config_md5)
            .fingerprint(queries)
            .unwrap();
        assert_eq!(fp_md5.hash.len(), 32); // MD5 = 128 bits = 32 hex chars

        // SHA-256
        let config_sha = FingerprintConfig {
            hash_algorithm: HashAlgorithm::Sha256,
            ..Default::default()
        };
        let fp_sha = QueryFingerprinter::new(config_sha)
            .fingerprint(queries)
            .unwrap();
        assert_eq!(fp_sha.hash.len(), 64); // SHA256 = 256 bits = 64 hex chars

        // FNV-1a
        let config_fnv = FingerprintConfig {
            hash_algorithm: HashAlgorithm::Fnv1a,
            ..Default::default()
        };
        let fp_fnv = QueryFingerprinter::new(config_fnv)
            .fingerprint(queries)
            .unwrap();
        assert_eq!(fp_fnv.hash.len(), 16); // FNV-1a 64-bit = 16 hex chars
    }

    #[test]
    fn test_statistics() {
        let config = FingerprintConfig::default();
        let fingerprinter = QueryFingerprinter::new(config);

        let _ = fingerprinter.fingerprint("SELECT ?s WHERE { ?s ?p ?o }");
        let _ = fingerprinter.fingerprint("SELECT ?s WHERE { ?s ?p ?o }"); // Cache hit

        let stats = fingerprinter.statistics();
        assert_eq!(stats.fingerprints_computed, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert!(stats.cache_hit_rate() > 0.0);
    }

    #[test]
    fn test_clear_cache() {
        let config = FingerprintConfig::default();
        let fingerprinter = QueryFingerprinter::new(config);

        let _ = fingerprinter.fingerprint("SELECT ?s WHERE { ?s ?p ?o }");
        assert_eq!(fingerprinter.statistics().cache_size, 1);

        fingerprinter.clear_cache();
        assert_eq!(fingerprinter.statistics().cache_size, 0);
    }
}
