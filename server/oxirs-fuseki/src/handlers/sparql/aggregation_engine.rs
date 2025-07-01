//! Enhanced aggregation engine for SPARQL 1.2
//!
//! This module provides advanced aggregation function support
//! and optimization for SPARQL queries.

use crate::error::FusekiResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Enhanced aggregation processor
#[derive(Debug, Clone)]
pub struct EnhancedAggregationProcessor {
    functions: HashMap<String, AggregationFunction>,
    optimization_cache: HashMap<String, String>,
}

impl EnhancedAggregationProcessor {
    pub fn new() -> Self {
        let mut processor = Self {
            functions: HashMap::new(),
            optimization_cache: HashMap::new(),
        };

        processor.register_builtin_functions();
        processor
    }

    /// Register built-in aggregation functions
    fn register_builtin_functions(&mut self) {
        // Enhanced GROUP_CONCAT with custom separators
        self.functions.insert(
            "GROUP_CONCAT".to_string(),
            AggregationFunction {
                name: "GROUP_CONCAT".to_string(),
                return_type: "literal".to_string(),
                supports_distinct: true,
                supports_separator: true,
                parallel_safe: true,
            },
        );

        // SAMPLE with deterministic mode
        self.functions.insert(
            "SAMPLE".to_string(),
            AggregationFunction {
                name: "SAMPLE".to_string(),
                return_type: "any".to_string(),
                supports_distinct: false,
                supports_separator: false,
                parallel_safe: true,
            },
        );

        // Extended statistical functions
        self.functions.insert(
            "MEDIAN".to_string(),
            AggregationFunction {
                name: "MEDIAN".to_string(),
                return_type: "numeric".to_string(),
                supports_distinct: true,
                supports_separator: false,
                parallel_safe: false,
            },
        );

        self.functions.insert(
            "MODE".to_string(),
            AggregationFunction {
                name: "MODE".to_string(),
                return_type: "any".to_string(),
                supports_distinct: false,
                supports_separator: false,
                parallel_safe: false,
            },
        );
    }

    /// Process aggregation functions in a query
    pub fn process_aggregations(&mut self, query: &str) -> FusekiResult<String> {
        let mut processed = query.to_string();

        // Find and process each aggregation function
        let functions: Vec<(String, AggregationFunction)> = self
            .functions
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (name, function) in functions {
            processed = self.process_function(&processed, &name, &function)?;
        }

        Ok(processed)
    }

    /// Process a specific aggregation function
    fn process_function(
        &mut self,
        query: &str,
        function_name: &str,
        function: &AggregationFunction,
    ) -> FusekiResult<String> {
        let pattern = format!("{}(", function_name);
        let mut result = query.to_string();

        while let Some(pos) = result.find(&pattern) {
            if let Some(func_call) = self.extract_function_call(&result[pos..]) {
                let optimized = self.optimize_function_call(&func_call, function)?;
                result = result.replace(&func_call, &optimized);
            } else {
                break;
            }
        }

        Ok(result)
    }

    /// Extract complete function call
    fn extract_function_call(&self, text: &str) -> Option<String> {
        let mut paren_count = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for (i, ch) in text.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }

            match ch {
                '\\' => escape_next = true,
                '"' | '\'' => in_string = !in_string,
                '(' if !in_string => paren_count += 1,
                ')' if !in_string => {
                    paren_count -= 1;
                    if paren_count == 0 {
                        return Some(text[..=i].to_string());
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Optimize a function call
    fn optimize_function_call(
        &mut self,
        func_call: &str,
        function: &AggregationFunction,
    ) -> FusekiResult<String> {
        // Check cache
        if let Some(cached) = self.optimization_cache.get(func_call) {
            return Ok(cached.clone());
        }

        let optimized = match function.name.as_str() {
            "GROUP_CONCAT" => self.optimize_group_concat(func_call)?,
            "SAMPLE" => self.optimize_sample(func_call)?,
            "MEDIAN" => self.optimize_median(func_call)?,
            "MODE" => self.optimize_mode(func_call)?,
            _ => func_call.to_string(),
        };

        // Cache the optimization
        self.optimization_cache
            .insert(func_call.to_string(), optimized.clone());

        Ok(optimized)
    }

    /// Optimize GROUP_CONCAT function
    fn optimize_group_concat(&self, func_call: &str) -> FusekiResult<String> {
        // Parse arguments
        let args = self.parse_function_args(func_call)?;

        if args.is_empty() {
            return Ok(func_call.to_string());
        }

        // Add default separator if not specified
        if args.len() == 1 && !func_call.contains("SEPARATOR") {
            let expr = &args[0];
            Ok(format!("GROUP_CONCAT({} ; SEPARATOR=',')", expr))
        } else if func_call.contains("DISTINCT") {
            // Optimize DISTINCT GROUP_CONCAT
            Ok(format!("OPTIMIZED_{}", func_call))
        } else {
            Ok(func_call.to_string())
        }
    }

    /// Optimize SAMPLE function
    fn optimize_sample(&self, func_call: &str) -> FusekiResult<String> {
        // Make SAMPLE deterministic for better caching
        if func_call.contains("DISTINCT") {
            Ok(format!(
                "DETERMINISTIC_SAMPLE({})",
                &func_call[7..func_call.len() - 1]
            )) // Remove SAMPLE( and )
        } else {
            Ok(func_call.to_string())
        }
    }

    /// Optimize MEDIAN function
    fn optimize_median(&self, _func_call: &str) -> FusekiResult<String> {
        // MEDIAN requires sorted data - add optimization hint
        Ok(format!("SORTED_{}", _func_call))
    }

    /// Optimize MODE function
    fn optimize_mode(&self, _func_call: &str) -> FusekiResult<String> {
        // MODE requires grouping - add optimization hint
        Ok(format!("GROUPED_{}", _func_call))
    }

    /// Parse function arguments
    fn parse_function_args(&self, func_call: &str) -> FusekiResult<Vec<String>> {
        // Find the opening parenthesis
        let open_paren = func_call
            .find('(')
            .ok_or_else(|| crate::error::FusekiError::query_parsing("Invalid function call"))?;

        // Extract arguments between parentheses
        let args_str = &func_call[open_paren + 1..func_call.len() - 1];

        if args_str.trim().is_empty() {
            return Ok(Vec::new());
        }

        // Simple argument parsing (doesn't handle nested functions well)
        let args: Vec<String> = args_str
            .split(',')
            .map(|arg| arg.trim().to_string())
            .collect();

        Ok(args)
    }

    /// Check if a function is supported
    pub fn is_supported(&self, function_name: &str) -> bool {
        self.functions.contains_key(function_name)
    }

    /// Get function metadata
    pub fn get_function(&self, function_name: &str) -> Option<&AggregationFunction> {
        self.functions.get(function_name)
    }
}

/// Aggregation function metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationFunction {
    pub name: String,
    pub return_type: String,
    pub supports_distinct: bool,
    pub supports_separator: bool,
    pub parallel_safe: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_concat_optimization() {
        let mut processor = EnhancedAggregationProcessor::new();

        // Test simple GROUP_CONCAT
        let query = "SELECT (GROUP_CONCAT(?name) as ?names) WHERE { ?s foaf:name ?name }";
        let result = processor.process_aggregations(query).unwrap();
        assert!(result.contains("SEPARATOR"));

        // Test GROUP_CONCAT with DISTINCT
        let query_distinct =
            "SELECT (GROUP_CONCAT(DISTINCT ?name) as ?names) WHERE { ?s foaf:name ?name }";
        let result_distinct = processor.process_aggregations(query_distinct).unwrap();
        assert!(result_distinct.contains("OPTIMIZED"));
    }

    #[test]
    fn test_sample_optimization() {
        let mut processor = EnhancedAggregationProcessor::new();

        let query = "SELECT (SAMPLE(?value) as ?sample) WHERE { ?s ?p ?value }";
        let result = processor.process_aggregations(query).unwrap();
        // Should remain unchanged for simple SAMPLE
        assert!(result.contains("SAMPLE"));
    }

    #[test]
    fn test_function_detection() {
        let processor = EnhancedAggregationProcessor::new();

        assert!(processor.is_supported("GROUP_CONCAT"));
        assert!(processor.is_supported("SAMPLE"));
        assert!(processor.is_supported("MEDIAN"));
        assert!(processor.is_supported("MODE"));
        assert!(!processor.is_supported("UNKNOWN_FUNCTION"));
    }

    #[test]
    fn test_argument_parsing() {
        let processor = EnhancedAggregationProcessor::new();

        let args = processor.parse_function_args("COUNT(?x)").unwrap();
        assert_eq!(args, vec!["?x"]);

        let args = processor
            .parse_function_args("GROUP_CONCAT(?name, ',')")
            .unwrap();
        assert_eq!(args, vec!["?name", "','"]);

        let args = processor.parse_function_args("SUM()").unwrap();
        assert!(args.is_empty());
    }
}
