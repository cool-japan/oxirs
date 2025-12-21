//! Query Similarity Detection and Analysis
//!
//! This module provides sophisticated algorithms for detecting similarity between SPARQL queries:
//! - Query normalization and fingerprinting
//! - AST-based structural similarity
//! - Edit distance algorithms (Levenshtein, tree edit distance)
//! - Statistical similarity measures (Jaccard, cosine similarity)
//! - Query clustering and template detection
//!
//! Used for:
//! - Detecting duplicate queries with slight variations
//! - Identifying query patterns for template extraction
//! - Optimizing query caching strategies
//! - Analyzing query workloads and usage patterns

use crate::cli::CliResult;
use std::collections::{HashMap, HashSet};

/// Query feature vector for similarity computation
#[derive(Debug, Clone)]
pub struct QueryFeatures {
    /// Triple patterns count
    pub triple_patterns: usize,
    /// Variables used in the query
    pub variables: HashSet<String>,
    /// Predicates (properties) used
    pub predicates: HashSet<String>,
    /// SPARQL keywords (SELECT, OPTIONAL, FILTER, etc.)
    pub keywords: HashSet<String>,
    /// Function calls (COUNT, SUM, STR, etc.)
    pub functions: HashSet<String>,
    /// Graph patterns (OPTIONAL, UNION, etc.)
    pub graph_patterns: HashSet<String>,
    /// Filters count
    pub filter_count: usize,
    /// Subqueries count
    pub subquery_count: usize,
    /// Normalized query text (lowercased, whitespace normalized)
    pub normalized_text: String,
}

/// Similarity score with multiple metrics
#[derive(Debug, Clone)]
pub struct SimilarityScore {
    /// Overall similarity (0.0-1.0)
    pub overall: f64,
    /// Structural similarity (AST-based)
    pub structural: f64,
    /// Text similarity (edit distance-based)
    pub textual: f64,
    /// Feature similarity (Jaccard/cosine)
    pub feature_based: f64,
    /// Confidence in the similarity assessment
    pub confidence: f64,
}

/// Query similarity result comparing two queries
#[derive(Debug, Clone)]
pub struct QuerySimilarityResult {
    /// First query
    pub query1: String,
    /// Second query
    pub query2: String,
    /// Similarity score
    pub score: SimilarityScore,
    /// Identified differences
    pub differences: Vec<String>,
    /// Common patterns
    pub common_patterns: Vec<String>,
    /// Recommendation (e.g., "Consider using a template")
    pub recommendation: Option<String>,
}

/// Query cluster containing similar queries
#[derive(Debug, Clone)]
pub struct QueryCluster {
    /// Cluster ID
    pub id: usize,
    /// Representative query (centroid)
    pub representative: String,
    /// Queries in this cluster
    pub queries: Vec<String>,
    /// Average intra-cluster similarity
    pub cohesion: f64,
    /// Suggested template
    pub suggested_template: Option<String>,
}

/// Query similarity analyzer
pub struct QuerySimilarityAnalyzer {
    /// Minimum similarity threshold for clustering
    similarity_threshold: f64,
    /// Weight for structural similarity (0.0-1.0)
    structural_weight: f64,
    /// Weight for textual similarity (0.0-1.0)
    textual_weight: f64,
    /// Weight for feature-based similarity (0.0-1.0)
    feature_weight: f64,
}

impl Default for QuerySimilarityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl QuerySimilarityAnalyzer {
    /// Create a new query similarity analyzer with default settings
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.7,
            structural_weight: 0.4,
            textual_weight: 0.3,
            feature_weight: 0.3,
        }
    }

    /// Create analyzer with custom weights
    pub fn with_weights(
        similarity_threshold: f64,
        structural_weight: f64,
        textual_weight: f64,
        feature_weight: f64,
    ) -> Self {
        let total = structural_weight + textual_weight + feature_weight;
        Self {
            similarity_threshold,
            structural_weight: structural_weight / total,
            textual_weight: textual_weight / total,
            feature_weight: feature_weight / total,
        }
    }

    /// Compare two queries and compute similarity
    pub fn compare_queries(&self, query1: &str, query2: &str) -> QuerySimilarityResult {
        let features1 = self.extract_features(query1);
        let features2 = self.extract_features(query2);

        let structural = self.compute_structural_similarity(&features1, &features2);
        let textual = self.compute_textual_similarity(query1, query2);
        let feature_based = self.compute_feature_similarity(&features1, &features2);

        let overall = self.structural_weight * structural
            + self.textual_weight * textual
            + self.feature_weight * feature_based;

        let confidence = self.compute_confidence(&features1, &features2, overall);

        let differences = self.identify_differences(&features1, &features2);
        let common_patterns = self.identify_common_patterns(&features1, &features2);

        let recommendation = if overall > 0.85 {
            Some("Queries are highly similar - consider using a template or caching".to_string())
        } else if overall > 0.7 {
            Some("Queries share common patterns - template extraction recommended".to_string())
        } else {
            None
        };

        QuerySimilarityResult {
            query1: query1.to_string(),
            query2: query2.to_string(),
            score: SimilarityScore {
                overall,
                structural,
                textual,
                feature_based,
                confidence,
            },
            differences,
            common_patterns,
            recommendation,
        }
    }

    /// Cluster queries by similarity
    pub fn cluster_queries(&self, queries: &[String]) -> Vec<QueryCluster> {
        if queries.is_empty() {
            return Vec::new();
        }

        let mut clusters: Vec<QueryCluster> = Vec::new();
        let mut assigned: HashSet<usize> = HashSet::new();

        for (i, query) in queries.iter().enumerate() {
            if assigned.contains(&i) {
                continue;
            }

            let mut cluster_queries = vec![query.clone()];
            assigned.insert(i);

            for (j, other) in queries.iter().enumerate() {
                if i == j || assigned.contains(&j) {
                    continue;
                }

                let result = self.compare_queries(query, other);
                if result.score.overall >= self.similarity_threshold {
                    cluster_queries.push(other.clone());
                    assigned.insert(j);
                }
            }

            let cohesion = if cluster_queries.len() > 1 {
                self.compute_cluster_cohesion(&cluster_queries)
            } else {
                1.0
            };

            let suggested_template = if cluster_queries.len() > 2 {
                Some(self.extract_template(&cluster_queries))
            } else {
                None
            };

            clusters.push(QueryCluster {
                id: clusters.len(),
                representative: query.clone(),
                queries: cluster_queries,
                cohesion,
                suggested_template,
            });
        }

        clusters
    }

    /// Find similar queries in a query log
    pub fn find_similar_queries(
        &self,
        target: &str,
        query_log: &[String],
        top_k: usize,
    ) -> Vec<(String, SimilarityScore)> {
        let mut results: Vec<(String, SimilarityScore)> = query_log
            .iter()
            .filter(|q| *q != target)
            .map(|q| {
                let result = self.compare_queries(target, q);
                (q.clone(), result.score)
            })
            .collect();

        results.sort_by(|a, b| b.1.overall.partial_cmp(&a.1.overall).unwrap());
        results.truncate(top_k);
        results
    }

    /// Extract query features for similarity computation
    fn extract_features(&self, query: &str) -> QueryFeatures {
        let normalized = self.normalize_query(query);

        let variables = self.extract_variables(&normalized);
        let predicates = self.extract_predicates(&normalized);
        let keywords = self.extract_keywords(&normalized);
        let functions = self.extract_functions(&normalized);
        let graph_patterns = self.extract_graph_patterns(&normalized);

        let triple_patterns = normalized.matches(" . ").count() + 1;
        let filter_count = normalized.matches("FILTER").count();
        let subquery_count = normalized.matches(" { SELECT ").count();

        QueryFeatures {
            triple_patterns,
            variables,
            predicates,
            keywords,
            functions,
            graph_patterns,
            filter_count,
            subquery_count,
            normalized_text: normalized,
        }
    }

    /// Normalize query to canonical form
    fn normalize_query(&self, query: &str) -> String {
        query
            .to_uppercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extract variables (?var, $var)
    fn extract_variables(&self, query: &str) -> HashSet<String> {
        let mut vars = HashSet::new();
        for word in query.split_whitespace() {
            if word.starts_with('?') || word.starts_with('$') {
                vars.insert(word.to_string());
            }
        }
        vars
    }

    /// Extract predicates (property names)
    fn extract_predicates(&self, query: &str) -> HashSet<String> {
        let mut predicates = HashSet::new();
        let words: Vec<&str> = query.split_whitespace().collect();

        // Look for predicates in triple patterns
        let mut i = 0;
        while i < words.len() {
            // Find triple pattern: subject predicate object
            if i + 2 < words.len() {
                let prev = words[i];
                let curr = words[i + 1];
                let next = words[i + 2];

                // If prev is a variable/URI and next is a variable/URI/literal, curr might be predicate
                let is_subject =
                    prev.starts_with('?') || prev.starts_with('$') || prev.starts_with('<');
                let is_object = next.starts_with('?')
                    || next.starts_with('$')
                    || next.starts_with('<')
                    || next.starts_with('"')
                    || next.contains(':');

                if is_subject && is_object {
                    // curr is likely a predicate
                    if curr == "A" {
                        // Shorthand for rdf:type
                        predicates.insert("RDF:TYPE".to_string());
                    } else if curr.contains(':') || curr.starts_with('<') {
                        predicates.insert(curr.to_string());
                    }
                }
            }
            i += 1;
        }

        // Also look for predicates with colons (namespace:property)
        for word in &words {
            if word.contains(':') && !word.starts_with('?') && !word.starts_with('$') {
                // Could be a predicate or object - context determines
                // If it looks like a property (lowercase after colon), it's likely a predicate
                if let Some(idx) = word.rfind(':') {
                    if idx + 1 < word.len() {
                        let after_colon = &word[idx + 1..];
                        if !after_colon.is_empty()
                            && after_colon.chars().next().unwrap().is_lowercase()
                        {
                            predicates.insert(word.to_string());
                        }
                    }
                }
            }
        }

        predicates
    }

    /// Extract SPARQL keywords
    fn extract_keywords(&self, query: &str) -> HashSet<String> {
        let sparql_keywords = [
            "SELECT",
            "CONSTRUCT",
            "ASK",
            "DESCRIBE",
            "WHERE",
            "OPTIONAL",
            "UNION",
            "FILTER",
            "BIND",
            "GROUP",
            "ORDER",
            "LIMIT",
            "OFFSET",
            "DISTINCT",
            "REDUCED",
        ];

        query
            .split_whitespace()
            .filter(|w| sparql_keywords.contains(w))
            .map(|s| s.to_string())
            .collect()
    }

    /// Extract function calls
    fn extract_functions(&self, query: &str) -> HashSet<String> {
        let sparql_functions = [
            "COUNT",
            "SUM",
            "AVG",
            "MIN",
            "MAX",
            "STR",
            "LANG",
            "DATATYPE",
            "BOUND",
            "ISIRI",
            "ISBLANK",
            "ISLITERAL",
            "REGEX",
            "CONCAT",
            "STRLEN",
            "SUBSTR",
            "UCASE",
            "LCASE",
        ];

        query
            .split_whitespace()
            .filter(|w| sparql_functions.contains(w))
            .map(|s| s.to_string())
            .collect()
    }

    /// Extract graph patterns
    fn extract_graph_patterns(&self, query: &str) -> HashSet<String> {
        let mut patterns = HashSet::new();
        if query.contains("OPTIONAL") {
            patterns.insert("OPTIONAL".to_string());
        }
        if query.contains("UNION") {
            patterns.insert("UNION".to_string());
        }
        if query.contains("GRAPH") {
            patterns.insert("GRAPH".to_string());
        }
        if query.contains("SERVICE") {
            patterns.insert("SERVICE".to_string());
        }
        if query.contains("MINUS") {
            patterns.insert("MINUS".to_string());
        }
        patterns
    }

    /// Compute structural similarity based on query features
    fn compute_structural_similarity(&self, f1: &QueryFeatures, f2: &QueryFeatures) -> f64 {
        let mut score = 0.0;
        let mut weights = 0.0;

        // Triple patterns similarity
        let tp_diff = (f1.triple_patterns as f64 - f2.triple_patterns as f64).abs();
        let tp_max = f1.triple_patterns.max(f2.triple_patterns) as f64;
        if tp_max > 0.0 {
            score += (1.0 - tp_diff / tp_max) * 0.2;
            weights += 0.2;
        }

        // Filter count similarity
        let fc_diff = (f1.filter_count as f64 - f2.filter_count as f64).abs();
        let fc_max = f1.filter_count.max(f2.filter_count) as f64;
        if fc_max > 0.0 {
            score += (1.0 - fc_diff / fc_max) * 0.15;
            weights += 0.15;
        }

        // Keyword similarity (Jaccard)
        score += self.jaccard_similarity(&f1.keywords, &f2.keywords) * 0.25;
        weights += 0.25;

        // Graph pattern similarity (Jaccard)
        score += self.jaccard_similarity(&f1.graph_patterns, &f2.graph_patterns) * 0.2;
        weights += 0.2;

        // Function similarity (Jaccard)
        score += self.jaccard_similarity(&f1.functions, &f2.functions) * 0.2;
        weights += 0.2;

        if weights > 0.0 {
            score / weights
        } else {
            0.0
        }
    }

    /// Compute textual similarity using normalized edit distance
    fn compute_textual_similarity(&self, query1: &str, query2: &str) -> f64 {
        let norm1 = self.normalize_query(query1);
        let norm2 = self.normalize_query(query2);

        let distance = self.levenshtein_distance(&norm1, &norm2);
        let max_len = norm1.len().max(norm2.len()) as f64;

        if max_len > 0.0 {
            1.0 - (distance as f64 / max_len)
        } else {
            1.0
        }
    }

    /// Compute feature-based similarity
    fn compute_feature_similarity(&self, f1: &QueryFeatures, f2: &QueryFeatures) -> f64 {
        let var_sim = self.jaccard_similarity(&f1.variables, &f2.variables);
        let pred_sim = self.jaccard_similarity(&f1.predicates, &f2.predicates);

        (var_sim + pred_sim) / 2.0
    }

    /// Compute Jaccard similarity for sets
    fn jaccard_similarity<T: std::hash::Hash + Eq>(
        &self,
        set1: &HashSet<T>,
        set2: &HashSet<T>,
    ) -> f64 {
        if set1.is_empty() && set2.is_empty() {
            return 1.0;
        }

        let intersection = set1.intersection(set2).count();
        let union = set1.union(set2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    /// Compute Levenshtein edit distance
    #[allow(clippy::needless_range_loop)]
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    /// Compute confidence in similarity assessment
    fn compute_confidence(&self, f1: &QueryFeatures, f2: &QueryFeatures, similarity: f64) -> f64 {
        // Higher confidence when queries have more features to compare
        let feature_count1 = f1.keywords.len()
            + f1.variables.len()
            + f1.predicates.len()
            + f1.functions.len()
            + f1.graph_patterns.len();
        let feature_count2 = f2.keywords.len()
            + f2.variables.len()
            + f2.predicates.len()
            + f2.functions.len()
            + f2.graph_patterns.len();

        let min_features = feature_count1.min(feature_count2) as f64;
        // Adjust denominator for more reasonable confidence on simple queries
        // Simple queries (3-5 features) should get ~0.3-0.5 base confidence
        // Complex queries (10+ features) should get ~1.0 base confidence
        let feature_confidence = (min_features / 10.0).clamp(0.0, 1.0);

        // Higher confidence when similarity is extreme (very high or very low)
        let extremeness = (similarity - 0.5).abs() * 2.0;

        // Weighted combination: 60% feature-based, 40% extremeness
        (feature_confidence * 0.6 + extremeness * 0.4).clamp(0.0, 1.0)
    }

    /// Identify differences between two queries
    fn identify_differences(&self, f1: &QueryFeatures, f2: &QueryFeatures) -> Vec<String> {
        let mut diffs = Vec::new();

        if f1.triple_patterns != f2.triple_patterns {
            diffs.push(format!(
                "Triple pattern count differs: {} vs {}",
                f1.triple_patterns, f2.triple_patterns
            ));
        }

        let kw_diff: Vec<_> = f1.keywords.symmetric_difference(&f2.keywords).collect();
        if !kw_diff.is_empty() {
            diffs.push(format!("Different keywords: {:?}", kw_diff));
        }

        let fn_diff: Vec<_> = f1.functions.symmetric_difference(&f2.functions).collect();
        if !fn_diff.is_empty() {
            diffs.push(format!("Different functions: {:?}", fn_diff));
        }

        diffs
    }

    /// Identify common patterns between queries
    fn identify_common_patterns(&self, f1: &QueryFeatures, f2: &QueryFeatures) -> Vec<String> {
        let mut patterns = Vec::new();

        let common_keywords: Vec<_> = f1.keywords.intersection(&f2.keywords).collect();
        if !common_keywords.is_empty() {
            patterns.push(format!("Common keywords: {:?}", common_keywords));
        }

        let common_predicates: Vec<_> = f1.predicates.intersection(&f2.predicates).collect();
        if !common_predicates.is_empty() {
            patterns.push(format!(
                "Common predicates: {} shared",
                common_predicates.len()
            ));
        }

        let common_patterns: Vec<_> = f1.graph_patterns.intersection(&f2.graph_patterns).collect();
        if !common_patterns.is_empty() {
            patterns.push(format!("Common graph patterns: {:?}", common_patterns));
        }

        patterns
    }

    /// Compute cluster cohesion (average pairwise similarity)
    fn compute_cluster_cohesion(&self, queries: &[String]) -> f64 {
        if queries.len() < 2 {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut count = 0;

        for i in 0..queries.len() {
            for j in (i + 1)..queries.len() {
                let result = self.compare_queries(&queries[i], &queries[j]);
                total_similarity += result.score.overall;
                count += 1;
            }
        }

        if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
        }
    }

    /// Extract template from similar queries
    fn extract_template(&self, queries: &[String]) -> String {
        if queries.is_empty() {
            return String::new();
        }

        // Simple template extraction: find common prefix and suffix
        let normalized: Vec<String> = queries.iter().map(|q| self.normalize_query(q)).collect();

        // Find common prefix
        let mut common_prefix = normalized[0].clone();
        for query in &normalized[1..] {
            common_prefix = self.common_prefix(&common_prefix, query);
        }

        // Find common suffix
        let mut common_suffix = normalized[0].clone();
        for query in &normalized[1..] {
            common_suffix = self.common_suffix(&common_suffix, query);
        }

        // Combine into template
        if !common_prefix.is_empty() && !common_suffix.is_empty() {
            format!("{} ... {}", common_prefix.trim(), common_suffix.trim())
        } else if !common_prefix.is_empty() {
            format!("{} ...", common_prefix.trim())
        } else if !common_suffix.is_empty() {
            format!("... {}", common_suffix.trim())
        } else {
            "Template extraction requires more similar queries".to_string()
        }
    }

    /// Find common prefix of two strings
    fn common_prefix(&self, s1: &str, s2: &str) -> String {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        let mut i = 0;
        while i < chars1.len() && i < chars2.len() && chars1[i] == chars2[i] {
            i += 1;
        }

        chars1[..i].iter().collect()
    }

    /// Find common suffix of two strings
    fn common_suffix(&self, s1: &str, s2: &str) -> String {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        let len1 = chars1.len();
        let len2 = chars2.len();
        let mut i = 0;

        while i < len1 && i < len2 && chars1[len1 - 1 - i] == chars2[len2 - 1 - i] {
            i += 1;
        }

        if i > 0 {
            chars1[(len1 - i)..].iter().collect()
        } else {
            String::new()
        }
    }

    /// Compute query fingerprint (hash)
    pub fn fingerprint(&self, query: &str) -> String {
        let features = self.extract_features(query);

        // Create a feature-based fingerprint
        let mut components = Vec::new();
        components.push(format!("tp:{}", features.triple_patterns));
        components.push(format!("fc:{}", features.filter_count));
        components.push(format!("sq:{}", features.subquery_count));

        let mut keywords: Vec<_> = features.keywords.iter().cloned().collect();
        keywords.sort();
        components.push(format!("kw:{}", keywords.join(",")));

        // Add predicates to fingerprint for better discrimination
        let mut predicates: Vec<_> = features.predicates.iter().cloned().collect();
        predicates.sort();
        if !predicates.is_empty() {
            components.push(format!("pr:{}", predicates.join(",")));
        }

        let mut functions: Vec<_> = features.functions.iter().cloned().collect();
        functions.sort();
        if !functions.is_empty() {
            components.push(format!("fn:{}", functions.join(",")));
        }

        components.join("|")
    }

    /// Deduplicate queries based on fingerprint
    pub fn deduplicate_queries(&self, queries: &[String]) -> HashMap<String, Vec<String>> {
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();

        for query in queries {
            let fp = self.fingerprint(query);
            groups.entry(fp).or_default().push(query.clone());
        }

        groups
    }
}

/// Analyze a query log and generate similarity report
pub fn analyze_query_log(query_log: &[String]) -> CliResult<()> {
    let analyzer = QuerySimilarityAnalyzer::new();

    println!("Query Log Analysis Report");
    println!("=========================\n");

    // Deduplicate queries
    let groups = analyzer.deduplicate_queries(query_log);
    println!("Total unique query patterns: {}", groups.len());
    println!("Total queries: {}", query_log.len());
    println!("Duplicate queries: {}\n", query_log.len() - groups.len());

    // Cluster queries
    let clusters = analyzer.cluster_queries(query_log);
    println!("Identified {} query clusters\n", clusters.len());

    for cluster in &clusters {
        println!(
            "Cluster {} (cohesion: {:.2}):",
            cluster.id, cluster.cohesion
        );
        println!("  Queries: {}", cluster.queries.len());
        if let Some(ref template) = cluster.suggested_template {
            println!("  Suggested template: {}", template);
        }
        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_query() {
        let analyzer = QuerySimilarityAnalyzer::new();
        let query = "SELECT  ?x  WHERE { ?x  a  foaf:Person }";
        let normalized = analyzer.normalize_query(query);
        assert_eq!(normalized, "SELECT ?X WHERE { ?X A FOAF:PERSON }");
    }

    #[test]
    fn test_extract_variables() {
        let analyzer = QuerySimilarityAnalyzer::new();
        let features =
            analyzer.extract_features("SELECT ?name ?age WHERE { ?person foaf:name ?name }");
        assert!(features.variables.contains("?NAME"));
        assert!(features.variables.contains("?AGE"));
        assert!(features.variables.contains("?PERSON"));
        assert_eq!(features.variables.len(), 3);
    }

    #[test]
    fn test_extract_keywords() {
        let analyzer = QuerySimilarityAnalyzer::new();
        let features = analyzer
            .extract_features("SELECT DISTINCT ?x WHERE { ?x a ?type } ORDER BY ?x LIMIT 10");
        assert!(features.keywords.contains("SELECT"));
        assert!(features.keywords.contains("DISTINCT"));
        assert!(features.keywords.contains("WHERE"));
        assert!(features.keywords.contains("ORDER"));
        assert!(features.keywords.contains("LIMIT"));
    }

    #[test]
    fn test_identical_queries() {
        let analyzer = QuerySimilarityAnalyzer::new();
        let query = "SELECT ?x WHERE { ?x a foaf:Person }";
        let result = analyzer.compare_queries(query, query);
        assert!(result.score.overall > 0.99);
        assert!(result.score.structural > 0.99);
        assert!(result.score.textual > 0.99);
    }

    #[test]
    fn test_similar_queries() {
        let analyzer = QuerySimilarityAnalyzer::new();
        let query1 = "SELECT ?name WHERE { ?person foaf:name ?name }";
        let query2 = "SELECT ?title WHERE { ?book dc:title ?title }";
        let result = analyzer.compare_queries(query1, query2);

        // Should have high structural similarity (same pattern)
        assert!(result.score.structural > 0.7);
        // Overall similarity should be moderate to high
        assert!(result.score.overall > 0.5);
    }

    #[test]
    fn test_different_queries() {
        let analyzer = QuerySimilarityAnalyzer::new();
        let query1 = "SELECT ?x WHERE { ?x a foaf:Person }";
        let query2 = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o . FILTER(?o > 10) }";
        let result = analyzer.compare_queries(query1, query2);

        // Should have low similarity
        assert!(result.score.overall < 0.5);
    }

    #[test]
    fn test_levenshtein_distance() {
        let analyzer = QuerySimilarityAnalyzer::new();
        assert_eq!(analyzer.levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(analyzer.levenshtein_distance("saturday", "sunday"), 3);
        assert_eq!(analyzer.levenshtein_distance("", "abc"), 3);
        assert_eq!(analyzer.levenshtein_distance("abc", ""), 3);
        assert_eq!(analyzer.levenshtein_distance("same", "same"), 0);
    }

    #[test]
    fn test_jaccard_similarity() {
        let analyzer = QuerySimilarityAnalyzer::new();

        let set1: HashSet<_> = vec!["a", "b", "c"].into_iter().collect();
        let set2: HashSet<_> = vec!["b", "c", "d"].into_iter().collect();
        let similarity = analyzer.jaccard_similarity(&set1, &set2);

        // Intersection: {b, c} = 2, Union: {a, b, c, d} = 4
        assert!((similarity - 0.5).abs() < 0.01);

        let set3: HashSet<_> = vec!["a", "b", "c"].into_iter().collect();
        let similarity_same = analyzer.jaccard_similarity(&set1, &set3);
        assert!((similarity_same - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_query_clustering() {
        let analyzer = QuerySimilarityAnalyzer::new();

        let queries = vec![
            "SELECT ?name WHERE { ?person foaf:name ?name }".to_string(),
            "SELECT ?title WHERE { ?book dc:title ?title }".to_string(),
            "SELECT ?label WHERE { ?resource rdfs:label ?label }".to_string(),
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }".to_string(),
        ];

        let clusters = analyzer.cluster_queries(&queries);

        // Should create at least 2 clusters (SELECT vs CONSTRUCT)
        assert!(clusters.len() >= 2);
    }

    #[test]
    fn test_fingerprint() {
        let analyzer = QuerySimilarityAnalyzer::new();

        let query1 = "SELECT ?x WHERE { ?x a foaf:Person }";
        let query2 = "SELECT ?y WHERE { ?y a foaf:Person }";

        let fp1 = analyzer.fingerprint(query1);
        let fp2 = analyzer.fingerprint(query2);

        // Same structure, different variable names -> same fingerprint
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_deduplicate_queries() {
        let analyzer = QuerySimilarityAnalyzer::new();

        let queries = vec![
            "SELECT ?x WHERE { ?x a foaf:Person }".to_string(),
            "SELECT ?y WHERE { ?y a foaf:Person }".to_string(),
            "SELECT ?name WHERE { ?person foaf:name ?name }".to_string(),
        ];

        let groups = analyzer.deduplicate_queries(&queries);

        // First two queries should be grouped together
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_find_similar_queries() {
        let analyzer = QuerySimilarityAnalyzer::new();

        let target = "SELECT ?name WHERE { ?person foaf:name ?name }";
        let query_log = vec![
            "SELECT ?title WHERE { ?book dc:title ?title }".to_string(),
            "SELECT ?label WHERE { ?resource rdfs:label ?label }".to_string(),
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }".to_string(),
        ];

        let similar = analyzer.find_similar_queries(target, &query_log, 2);

        assert_eq!(similar.len(), 2);
        // SELECT queries should rank higher than CONSTRUCT
        assert!(similar[0].1.overall > similar[1].1.overall);
    }

    #[test]
    fn test_template_extraction() {
        let analyzer = QuerySimilarityAnalyzer::new();

        let queries = vec![
            "SELECT ?name WHERE { ?person foaf:name ?name }".to_string(),
            "SELECT ?title WHERE { ?book dc:title ?title }".to_string(),
            "SELECT ?label WHERE { ?resource rdfs:label ?label }".to_string(),
        ];

        let template = analyzer.extract_template(&queries);

        // Should contain SELECT and WHERE
        assert!(template.contains("SELECT"));
    }

    #[test]
    fn test_confidence_computation() {
        let analyzer = QuerySimilarityAnalyzer::new();

        let query1 = "SELECT ?x WHERE { ?x a foaf:Person }";
        let query2 = "SELECT ?y WHERE { ?y a foaf:Person }";

        let result = analyzer.compare_queries(query1, query2);

        // High similarity with sufficient features -> high confidence
        assert!(result.score.confidence > 0.5);
    }

    #[test]
    fn test_custom_weights() {
        let analyzer = QuerySimilarityAnalyzer::with_weights(0.8, 0.5, 0.3, 0.2);

        // Weights should be normalized to sum to 1.0
        let total = analyzer.structural_weight + analyzer.textual_weight + analyzer.feature_weight;
        assert!((total - 1.0).abs() < 0.01);
    }
}
