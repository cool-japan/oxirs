//! [`EnhancedBindProcessor`] — BIND clause processing with optimization and caching.
//!
//! Split out of the original `bind_values_enhanced` module (Round 32 refactor).
//! Contains the BIND-specific entry points: expression extraction from query
//! text, evaluation, caching, and the `Default` impl.

use crate::error::FusekiResult;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::bind_values_enhanced_types::{
    AdvancedBindOptimizer, BindStatistics, CachedExpression, EnhancedBindProcessor,
    ExpressionCache, ExpressionEvaluator,
};

impl Default for EnhancedBindProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedBindProcessor {
    pub fn new() -> Self {
        Self {
            expression_evaluator: ExpressionEvaluator::new(),
            bind_optimizer: AdvancedBindOptimizer::new(),
            expression_cache: Arc::new(RwLock::new(ExpressionCache::new())),
            statistics: Arc::new(RwLock::new(BindStatistics::default())),
        }
    }

    /// Process BIND clauses with optimization
    pub async fn process_bind_clauses(
        &self,
        query: &str,
        bindings: &mut [HashMap<String, serde_json::Value>],
    ) -> FusekiResult<()> {
        let bind_expressions = self.extract_bind_expressions(query)?;

        for (var, expr) in bind_expressions {
            // Optimize the expression
            let optimized_expr = self.bind_optimizer.optimize_expression(&expr)?;

            // Check cache
            let cache_key = self.compute_cache_key(&optimized_expr);
            if let Some(cached_result) = self.get_cached_result(&cache_key).await? {
                self.apply_bind_result(bindings, &var, cached_result);
                continue;
            }

            // Evaluate the expression
            let start_time = std::time::Instant::now();
            let result = self
                .expression_evaluator
                .evaluate(&optimized_expr, bindings)?;
            let eval_time = start_time.elapsed().as_millis() as f64;

            // Cache the result
            self.cache_result(cache_key, result.clone(), eval_time)
                .await?;

            // Apply to bindings
            self.apply_bind_result(bindings, &var, result);

            // Update statistics
            self.update_statistics(eval_time).await;
        }

        Ok(())
    }

    pub(crate) fn extract_bind_expressions(
        &self,
        query: &str,
    ) -> FusekiResult<Vec<(String, String)>> {
        let mut expressions = Vec::new();

        // Find BIND clauses
        let query_lower = query.to_lowercase();
        let mut pos = 0;

        while let Some(bind_pos) = query_lower[pos..].find("bind(") {
            let bind_start = pos + bind_pos;

            // Extract the BIND expression
            if let Some(expr_end) = self.find_expression_end(&query[bind_start + 5..]) {
                let expr = &query[bind_start + 5..bind_start + 5 + expr_end];

                // Extract the variable (AS ?var) - case insensitive
                let expr_lower = expr.to_lowercase();
                if let Some(as_pos) = expr_lower.rfind(" as ") {
                    let var_part = expr[as_pos + 4..].trim();
                    if var_part.starts_with('?') {
                        let var = var_part.trim_end_matches(')').to_string();
                        let expression = expr[..as_pos].trim().to_string();
                        expressions.push((var, expression));
                    }
                }
            }

            pos = bind_start + 5;
        }

        Ok(expressions)
    }

    fn find_expression_end(&self, expr: &str) -> Option<usize> {
        let mut paren_count = 1;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, ch) in expr.chars().enumerate() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match ch {
                '\\' => escape_next = true,
                '"' => in_string = !in_string,
                '(' if !in_string => paren_count += 1,
                ')' if !in_string => {
                    paren_count -= 1;
                    if paren_count == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }

        None
    }

    fn compute_cache_key(&self, expr: &str) -> String {
        format!("{:x}", md5::compute(expr))
    }

    async fn get_cached_result(&self, key: &str) -> FusekiResult<Option<serde_json::Value>> {
        let cache = self.expression_cache.read().await;

        if let Some(cached) = cache.cache.get(key) {
            Ok(Some(cached.result.clone()))
        } else {
            Ok(None)
        }
    }

    async fn cache_result(
        &self,
        key: String,
        result: serde_json::Value,
        compute_time: f64,
    ) -> FusekiResult<()> {
        let mut cache = self.expression_cache.write().await;

        let cached_expr = CachedExpression {
            expression_hash: key.clone(),
            result,
            compute_time_ms: compute_time,
            access_count: 1,
            last_accessed: chrono::Utc::now(),
        };

        cache.cache.insert(key, cached_expr);

        // Evict if necessary
        if cache.cache.len() > cache.max_size {
            cache.evict_lru();
        }

        Ok(())
    }

    fn apply_bind_result(
        &self,
        bindings: &mut [HashMap<String, serde_json::Value>],
        var: &str,
        result: serde_json::Value,
    ) {
        for binding in bindings.iter_mut() {
            binding.insert(var.to_string(), result.clone());
        }
    }

    async fn update_statistics(&self, eval_time: f64) {
        {
            let mut stats = self.statistics.write().await;
            stats.total_bind_expressions += 1;

            let total_time = stats.average_evaluation_time_ms * stats.total_bind_expressions as f64;
            stats.average_evaluation_time_ms =
                (total_time + eval_time) / (stats.total_bind_expressions as f64);
        }
    }
}
