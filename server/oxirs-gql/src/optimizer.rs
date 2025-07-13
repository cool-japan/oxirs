//! GraphQL Query Optimization and Caching
//!
//! This module provides query optimization, caching strategies, and performance
//! enhancements for GraphQL execution over RDF data.

use crate::ast::{Document, Field, Selection, SelectionSet, Value};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::RwLock as AsyncRwLock;

/// Query optimization and caching configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub enable_query_caching: bool,
    pub enable_result_caching: bool,
    pub enable_query_planning: bool,
    pub enable_field_batching: bool,
    pub cache_ttl: Duration,
    pub max_cache_size: usize,
    pub max_query_depth: usize,
    pub max_query_complexity: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_query_caching: true,
            enable_result_caching: true,
            enable_query_planning: true,
            enable_field_batching: true,
            cache_ttl: Duration::from_secs(300), // 5 minutes
            max_cache_size: 1000,
            max_query_depth: 15,
            max_query_complexity: 1000,
        }
    }
}

/// Query complexity analysis result
#[derive(Debug, Clone)]
pub struct QueryComplexity {
    pub depth: usize,
    pub field_count: usize,
    pub complexity_score: usize,
    pub estimated_cost: f64,
}

impl QueryComplexity {
    pub fn is_valid(&self, config: &OptimizationConfig) -> bool {
        self.depth <= config.max_query_depth && self.complexity_score <= config.max_query_complexity
    }
}

/// Cached query plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    pub query_hash: u64,
    pub sparql_query: String,
    pub field_order: Vec<String>,
    pub required_joins: Vec<String>,
    pub estimated_cost: f64,
    pub created_at: std::time::SystemTime,
}

/// Cached result entry
#[derive(Debug, Clone)]
pub struct CachedResult {
    pub result: serde_json::Value,
    pub created_at: Instant,
    pub access_count: u32,
    pub ttl: Duration,
}

impl CachedResult {
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Query optimization and caching engine
pub struct QueryOptimizer {
    config: OptimizationConfig,
    query_plan_cache: Arc<RwLock<HashMap<u64, QueryPlan>>>,
    result_cache: Arc<AsyncRwLock<HashMap<String, CachedResult>>>,
    complexity_analyzer: ComplexityAnalyzer,
    query_planner: QueryPlanner,
}

impl QueryOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config: config.clone(),
            query_plan_cache: Arc::new(RwLock::new(HashMap::new())),
            result_cache: Arc::new(AsyncRwLock::new(HashMap::new())),
            complexity_analyzer: ComplexityAnalyzer::new(config.clone()),
            query_planner: QueryPlanner::new(config),
        }
    }

    /// Analyze query complexity and validate against limits
    pub fn analyze_complexity(&self, document: &Document) -> Result<QueryComplexity> {
        self.complexity_analyzer.analyze(document)
    }

    /// Get or create optimized query plan
    pub async fn get_query_plan(&self, document: &Document) -> Result<QueryPlan> {
        if !self.config.enable_query_planning {
            return self.query_planner.create_basic_plan(document).await;
        }

        let query_hash = self.calculate_query_hash(document);

        // Check cache first
        if let Some(cached_plan) = self.get_cached_plan(query_hash) {
            return Ok(cached_plan);
        }

        // Create new plan
        let plan = self.query_planner.create_optimized_plan(document).await?;

        // Cache the plan
        self.cache_query_plan(query_hash, plan.clone()).await;

        Ok(plan)
    }

    /// Get cached result if available and not expired
    pub async fn get_cached_result(&self, cache_key: &str) -> Option<serde_json::Value> {
        if !self.config.enable_result_caching {
            return None;
        }

        let cache = self.result_cache.read().await;
        if let Some(entry) = cache.get(cache_key) {
            if !entry.is_expired() {
                return Some(entry.result.clone());
            }
        }
        None
    }

    /// Cache query result
    pub async fn cache_result(&self, cache_key: String, result: serde_json::Value) {
        if !self.config.enable_result_caching {
            return;
        }

        let mut cache = self.result_cache.write().await;

        // Evict expired entries and enforce size limit
        self.evict_expired_results(&mut cache);

        if cache.len() >= self.config.max_cache_size {
            self.evict_lru_result(&mut cache);
        }

        cache.insert(
            cache_key,
            CachedResult {
                result,
                created_at: Instant::now(),
                access_count: 1,
                ttl: self.config.cache_ttl,
            },
        );
    }

    /// Generate cache key for a query with variables
    pub fn generate_cache_key(
        &self,
        document: &Document,
        variables: &HashMap<String, Value>,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{document:?}").hash(&mut hasher);
        format!("{variables:?}").hash(&mut hasher);
        let finish = hasher.finish();
        format!("query_{finish}")
    }

    /// Optimize field selections for batching
    pub fn optimize_field_batching(&self, selections: &[Selection]) -> Vec<Vec<Selection>> {
        if !self.config.enable_field_batching {
            return vec![selections.to_vec()];
        }

        // Group fields by their expected cost/complexity
        let mut scalar_fields = Vec::new();
        let mut object_fields = Vec::new();
        let mut complex_fields = Vec::new();

        for selection in selections {
            match selection {
                Selection::Field(field) => {
                    if field.selection_set.is_some() {
                        if self.is_complex_field(field) {
                            complex_fields.push(selection.clone());
                        } else {
                            object_fields.push(selection.clone());
                        }
                    } else {
                        scalar_fields.push(selection.clone());
                    }
                }
                _ => complex_fields.push(selection.clone()),
            }
        }

        // Return batches: scalars first, then objects, then complex fields
        let mut batches = Vec::new();
        if !scalar_fields.is_empty() {
            batches.push(scalar_fields);
        }
        if !object_fields.is_empty() {
            batches.push(object_fields);
        }
        if !complex_fields.is_empty() {
            batches.push(complex_fields);
        }

        if batches.is_empty() {
            vec![selections.to_vec()]
        } else {
            batches
        }
    }

    fn calculate_query_hash(&self, document: &Document) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{document:?}").hash(&mut hasher);
        hasher.finish()
    }

    fn get_cached_plan(&self, query_hash: u64) -> Option<QueryPlan> {
        let cache = self.query_plan_cache.read().ok()?;
        cache.get(&query_hash).cloned()
    }

    async fn cache_query_plan(&self, query_hash: u64, plan: QueryPlan) {
        if let Ok(mut cache) = self.query_plan_cache.write() {
            if cache.len() >= self.config.max_cache_size {
                // Simple LRU: remove oldest entry
                if let Some(oldest_key) = cache.keys().next().copied() {
                    cache.remove(&oldest_key);
                }
            }
            cache.insert(query_hash, plan);
        }
    }

    fn evict_expired_results(&self, cache: &mut HashMap<String, CachedResult>) {
        let expired_keys: Vec<String> = cache
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            cache.remove(&key);
        }
    }

    fn evict_lru_result(&self, cache: &mut HashMap<String, CachedResult>) {
        // Find entry with lowest access count
        let lru_key = cache
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(key, _)| key.clone());

        if let Some(key) = lru_key {
            cache.remove(&key);
        }
    }

    fn is_complex_field(&self, field: &Field) -> bool {
        // Determine if a field is complex based on arguments and nesting
        field.arguments.len() > 2
            || field
                .selection_set
                .as_ref()
                .map(|ss| ss.selections.len() > 5)
                .unwrap_or(false)
    }
}

/// Query complexity analyzer
pub struct ComplexityAnalyzer {
    #[allow(dead_code)]
    config: OptimizationConfig,
}

impl ComplexityAnalyzer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }

    pub fn analyze(&self, document: &Document) -> Result<QueryComplexity> {
        let mut depth = 0;
        let mut field_count = 0;
        let mut complexity_score = 0;

        for definition in &document.definitions {
            if let crate::ast::Definition::Operation(operation) = definition {
                let (op_depth, op_fields, op_complexity) =
                    self.analyze_selection_set(&operation.selection_set, 1)?;

                depth = depth.max(op_depth);
                field_count += op_fields;
                complexity_score += op_complexity;
            }
        }

        let estimated_cost = self.calculate_estimated_cost(depth, field_count, complexity_score);

        Ok(QueryComplexity {
            depth,
            field_count,
            complexity_score,
            estimated_cost,
        })
    }

    fn analyze_selection_set(
        &self,
        selection_set: &SelectionSet,
        current_depth: usize,
    ) -> Result<(usize, usize, usize)> {
        let mut max_depth = current_depth;
        let mut total_fields = 0;
        let mut total_complexity = 0;

        for selection in &selection_set.selections {
            match selection {
                Selection::Field(field) => {
                    total_fields += 1;
                    total_complexity += self.calculate_field_complexity(field);

                    if let Some(nested_selection) = &field.selection_set {
                        let (nested_depth, nested_fields, nested_complexity) =
                            self.analyze_selection_set(nested_selection, current_depth + 1)?;

                        max_depth = max_depth.max(nested_depth);
                        total_fields += nested_fields;
                        total_complexity += nested_complexity;
                    }
                }
                Selection::InlineFragment(fragment) => {
                    let (frag_depth, frag_fields, frag_complexity) =
                        self.analyze_selection_set(&fragment.selection_set, current_depth)?;

                    max_depth = max_depth.max(frag_depth);
                    total_fields += frag_fields;
                    total_complexity += frag_complexity;
                }
                Selection::FragmentSpread(_) => {
                    // Fragment spreads add complexity
                    total_complexity += 5;
                }
            }
        }

        Ok((max_depth, total_fields, total_complexity))
    }

    fn calculate_field_complexity(&self, field: &Field) -> usize {
        let mut complexity = 1; // Base complexity

        // Arguments increase complexity
        complexity += field.arguments.len() * 2;

        // Directives increase complexity
        complexity += field.directives.len() * 3;

        // Certain field names are known to be expensive
        match field.name.as_str() {
            "sparql" => complexity += 10,
            "search" | "filter" => complexity += 5,
            _ if field.name.contains("aggregate") => complexity += 8,
            _ => {}
        }

        complexity
    }

    fn calculate_estimated_cost(
        &self,
        depth: usize,
        field_count: usize,
        complexity_score: usize,
    ) -> f64 {
        // Simple cost estimation formula
        let depth_cost = depth as f64 * 1.5;
        let field_cost = field_count as f64 * 0.5;
        let complexity_cost = complexity_score as f64 * 0.1;

        depth_cost + field_cost + complexity_cost
    }
}

/// Query planner for optimizing SPARQL generation
pub struct QueryPlanner {
    #[allow(dead_code)]
    config: OptimizationConfig,
}

impl QueryPlanner {
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }

    pub async fn create_basic_plan(&self, document: &Document) -> Result<QueryPlan> {
        let query_hash = self.calculate_document_hash(document);

        Ok(QueryPlan {
            query_hash,
            sparql_query: "SELECT * WHERE { ?s ?p ?o } LIMIT 10".to_string(),
            field_order: vec!["s".to_string(), "p".to_string(), "o".to_string()],
            required_joins: vec![],
            estimated_cost: 1.0,
            created_at: std::time::SystemTime::now(),
        })
    }

    pub async fn create_optimized_plan(&self, document: &Document) -> Result<QueryPlan> {
        let query_hash = self.calculate_document_hash(document);

        // Analyze the query structure
        let field_analysis = self.analyze_query_fields(document)?;
        let join_analysis = self.analyze_required_joins(&field_analysis);
        let optimization_hints = self.generate_optimization_hints(&field_analysis);

        // Build optimized SPARQL query
        let sparql_query =
            self.build_optimized_sparql(&field_analysis, &join_analysis, &optimization_hints);

        // Calculate estimated cost
        let estimated_cost = self.estimate_query_cost(&field_analysis, &join_analysis);

        Ok(QueryPlan {
            query_hash,
            sparql_query,
            field_order: field_analysis.field_names,
            required_joins: join_analysis,
            estimated_cost,
            created_at: std::time::SystemTime::now(),
        })
    }

    fn calculate_document_hash(&self, document: &Document) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{document:?}").hash(&mut hasher);
        hasher.finish()
    }

    fn analyze_query_fields(&self, document: &Document) -> Result<FieldAnalysis> {
        let mut field_names = Vec::new();
        let mut scalar_fields = HashSet::new();
        let mut object_fields = HashSet::new();
        let mut filter_conditions = Vec::new();

        for definition in &document.definitions {
            if let crate::ast::Definition::Operation(operation) = definition {
                self.collect_field_info(
                    &operation.selection_set,
                    &mut field_names,
                    &mut scalar_fields,
                    &mut object_fields,
                    &mut filter_conditions,
                );
            }
        }

        Ok(FieldAnalysis {
            field_names,
            scalar_fields,
            object_fields,
            filter_conditions,
        })
    }

    #[allow(clippy::only_used_in_recursion)]
    fn collect_field_info(
        &self,
        selection_set: &SelectionSet,
        field_names: &mut Vec<String>,
        scalar_fields: &mut HashSet<String>,
        object_fields: &mut HashSet<String>,
        filter_conditions: &mut Vec<String>,
    ) {
        for selection in &selection_set.selections {
            if let Selection::Field(field) = selection {
                field_names.push(field.name.clone());

                if field.selection_set.is_some() {
                    object_fields.insert(field.name.clone());
                    if let Some(nested) = &field.selection_set {
                        self.collect_field_info(
                            nested,
                            field_names,
                            scalar_fields,
                            object_fields,
                            filter_conditions,
                        );
                    }
                } else {
                    scalar_fields.insert(field.name.clone());
                }

                // Extract filter conditions from arguments
                for arg in &field.arguments {
                    if arg.name == "where" || arg.name == "filter" {
                        if let Value::StringValue(condition) = &arg.value {
                            filter_conditions.push(condition.clone());
                        }
                    }
                }
            }
        }
    }

    fn analyze_required_joins(&self, field_analysis: &FieldAnalysis) -> Vec<String> {
        let mut joins = Vec::new();

        // Analyze object fields that require joins
        for object_field in &field_analysis.object_fields {
            match object_field.as_str() {
                "knows" => joins.push("PERSON_KNOWS_JOIN".to_string()),
                "worksFor" => joins.push("PERSON_ORG_JOIN".to_string()),
                "memberOf" => joins.push("PERSON_GROUP_JOIN".to_string()),
                _ => {}
            }
        }

        joins
    }

    fn generate_optimization_hints(&self, field_analysis: &FieldAnalysis) -> Vec<String> {
        let mut hints = Vec::new();

        // Generate hints based on field patterns
        if field_analysis.scalar_fields.contains("id") {
            hints.push("USE_ID_INDEX".to_string());
        }

        if field_analysis.field_names.len() > 10 {
            hints.push("LIMIT_RESULT_SIZE".to_string());
        }

        if !field_analysis.filter_conditions.is_empty() {
            hints.push("FILTER_EARLY".to_string());
        }

        hints
    }

    fn build_optimized_sparql(
        &self,
        field_analysis: &FieldAnalysis,
        joins: &[String],
        hints: &[String],
    ) -> String {
        let mut sparql_parts = Vec::new();

        // Add prefixes
        sparql_parts.push("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>".to_string());
        sparql_parts.push("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>".to_string());
        sparql_parts.push("PREFIX foaf: <http://xmlns.com/foaf/0.1/>".to_string());

        // Build SELECT clause
        let select_vars: Vec<String> = field_analysis
            .field_names
            .iter()
            .map(|name| format!("?{name}"))
            .collect();

        if select_vars.is_empty() {
            sparql_parts.push("SELECT ?s ?p ?o".to_string());
        } else {
            sparql_parts.push(format!("SELECT {}", select_vars.join(" ")));
        }

        // Build WHERE clause
        let mut where_patterns = Vec::new();

        // Basic triple pattern
        where_patterns.push("?s ?p ?o".to_string());

        // Add specific patterns based on fields
        for field in &field_analysis.scalar_fields {
            match field.as_str() {
                "name" => where_patterns.push("?s foaf:name ?name".to_string()),
                "email" => where_patterns.push("?s foaf:mbox ?email".to_string()),
                "age" => where_patterns.push("?s foaf:age ?age".to_string()),
                _ => {}
            }
        }

        // Add join patterns
        for join in joins {
            match join.as_str() {
                "PERSON_KNOWS_JOIN" => where_patterns.push("?s foaf:knows ?knows".to_string()),
                "PERSON_ORG_JOIN" => {
                    where_patterns.push("?s foaf:workplaceHomepage ?org".to_string())
                }
                _ => {}
            }
        }

        // Add filter conditions
        for condition in &field_analysis.filter_conditions {
            where_patterns.push(format!("FILTER({condition})"));
        }

        sparql_parts.push(format!("WHERE {{ {} }}", where_patterns.join(" . ")));

        // Add optimization hints as comments
        for hint in hints {
            sparql_parts.push(format!("# HINT: {hint}"));
        }

        // Add LIMIT if suggested
        if hints.contains(&"LIMIT_RESULT_SIZE".to_string()) {
            sparql_parts.push("LIMIT 100".to_string());
        } else {
            sparql_parts.push("LIMIT 1000".to_string());
        }

        sparql_parts.join("\n")
    }

    fn estimate_query_cost(&self, field_analysis: &FieldAnalysis, joins: &[String]) -> f64 {
        let base_cost = 1.0;
        let field_cost = field_analysis.field_names.len() as f64 * 0.1;
        let join_cost = joins.len() as f64 * 2.0;
        let filter_cost = field_analysis.filter_conditions.len() as f64 * 0.5;

        base_cost + field_cost + join_cost + filter_cost
    }
}

/// Analysis result for query fields
#[derive(Debug)]
struct FieldAnalysis {
    field_names: Vec<String>,
    scalar_fields: HashSet<String>,
    object_fields: HashSet<String>,
    filter_conditions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_document;

    #[test]
    fn test_complexity_analyzer() {
        let query = "{ hello world }";
        let document = parse_document(query).unwrap();

        let analyzer = ComplexityAnalyzer::new(OptimizationConfig::default());
        let complexity = analyzer.analyze(&document).unwrap();

        assert!(complexity.depth >= 1);
        assert!(complexity.field_count >= 2);
    }

    #[tokio::test]
    async fn test_query_optimizer() {
        let config = OptimizationConfig::default();
        let optimizer = QueryOptimizer::new(config);

        let query = "{ hello version }";
        let document = parse_document(query).unwrap();

        let complexity = optimizer.analyze_complexity(&document).unwrap();
        assert!(complexity.is_valid(&optimizer.config));

        let plan = optimizer.get_query_plan(&document).await.unwrap();
        assert!(!plan.sparql_query.is_empty());
    }

    #[tokio::test]
    async fn test_result_caching() {
        let config = OptimizationConfig::default();
        let optimizer = QueryOptimizer::new(config);

        let cache_key = "test_key".to_string();
        let result = serde_json::json!({"hello": "world"});

        optimizer
            .cache_result(cache_key.clone(), result.clone())
            .await;

        let cached = optimizer.get_cached_result(&cache_key).await;
        assert_eq!(cached, Some(result));
    }
}
