//! SPARQL vector function implementations

use super::config::{
    VectorQuery, VectorServiceArg, VectorServiceFunction, VectorServiceParameter,
    VectorParameterType, VectorServiceResult,
};
use super::query_executor::QueryExecutor;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Custom vector function trait for user-defined functions
pub trait CustomVectorFunction: Send + Sync {
    fn execute(&self, args: &[VectorServiceArg]) -> Result<VectorServiceResult>;
    fn arity(&self) -> usize;
    fn description(&self) -> String;
}

/// SPARQL vector functions implementation
pub struct SparqlVectorFunctions {
    function_registry: HashMap<String, VectorServiceFunction>,
    custom_functions: HashMap<String, Box<dyn CustomVectorFunction>>,
}

impl SparqlVectorFunctions {
    pub fn new() -> Self {
        let mut functions = Self {
            function_registry: HashMap::new(),
            custom_functions: HashMap::new(),
        };
        
        functions.register_default_functions();
        functions
    }

    /// Register all default SPARQL vector functions
    fn register_default_functions(&mut self) {
        // vec:similarity function
        self.function_registry.insert("similarity".to_string(), VectorServiceFunction {
            name: "similarity".to_string(),
            arity: 2,
            description: "Calculate similarity between two resources".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "resource1".to_string(),
                    param_type: VectorParameterType::IRI,
                    required: true,
                    description: "First resource for similarity comparison".to_string(),
                },
                VectorServiceParameter {
                    name: "resource2".to_string(),
                    param_type: VectorParameterType::IRI,
                    required: true,
                    description: "Second resource for similarity comparison".to_string(),
                },
            ],
        });

        // vec:similar function
        self.function_registry.insert("similar".to_string(), VectorServiceFunction {
            name: "similar".to_string(),
            arity: 3,
            description: "Find similar resources to a given resource".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "resource".to_string(),
                    param_type: VectorParameterType::IRI,
                    required: true,
                    description: "Resource to find similar items for".to_string(),
                },
                VectorServiceParameter {
                    name: "limit".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Maximum number of results to return".to_string(),
                },
                VectorServiceParameter {
                    name: "threshold".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Minimum similarity threshold".to_string(),
                },
            ],
        });

        // vec:search function
        self.function_registry.insert("search".to_string(), VectorServiceFunction {
            name: "search".to_string(),
            arity: 6,
            description: "Search for resources using text query with cross-language support".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "query_text".to_string(),
                    param_type: VectorParameterType::String,
                    required: true,
                    description: "Text query for search".to_string(),
                },
                VectorServiceParameter {
                    name: "limit".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Maximum number of results to return".to_string(),
                },
                VectorServiceParameter {
                    name: "threshold".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Minimum similarity threshold".to_string(),
                },
                VectorServiceParameter {
                    name: "metric".to_string(),
                    param_type: VectorParameterType::String,
                    required: false,
                    description: "Similarity metric to use".to_string(),
                },
                VectorServiceParameter {
                    name: "cross_language".to_string(),
                    param_type: VectorParameterType::String,
                    required: false,
                    description: "Enable cross-language search (true/false)".to_string(),
                },
                VectorServiceParameter {
                    name: "languages".to_string(),
                    param_type: VectorParameterType::String,
                    required: false,
                    description: "Comma-separated list of target languages".to_string(),
                },
            ],
        });

        // vec:searchIn function  
        self.function_registry.insert("searchIn".to_string(), VectorServiceFunction {
            name: "searchIn".to_string(),
            arity: 5,
            description: "Search within a specific graph with scoping options".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "query".to_string(),
                    param_type: VectorParameterType::String,
                    required: true,
                    description: "Text query for search".to_string(),
                },
                VectorServiceParameter {
                    name: "graph".to_string(),
                    param_type: VectorParameterType::IRI,
                    required: true,
                    description: "Target graph IRI for scoped search".to_string(),
                },
                VectorServiceParameter {
                    name: "limit".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Maximum number of results to return".to_string(),
                },
                VectorServiceParameter {
                    name: "scope".to_string(),
                    param_type: VectorParameterType::String,
                    required: false,
                    description: "Search scope: 'exact', 'children', 'parents', 'hierarchy', 'related'".to_string(),
                },
                VectorServiceParameter {
                    name: "threshold".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Minimum similarity threshold for results".to_string(),
                },
            ],
        });

        // vec:embed function
        self.function_registry.insert("embed".to_string(), VectorServiceFunction {
            name: "embed".to_string(),
            arity: 1,
            description: "Generate embedding for text content".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "text".to_string(),
                    param_type: VectorParameterType::String,
                    required: true,
                    description: "Text content to generate embedding for".to_string(),
                },
            ],
        });

        // vec:cluster function
        self.function_registry.insert("cluster".to_string(), VectorServiceFunction {
            name: "cluster".to_string(),
            arity: 2,
            description: "Cluster similar resources".to_string(),
            parameters: vec![
                VectorServiceParameter {
                    name: "resources".to_string(),
                    param_type: VectorParameterType::String,
                    required: true,
                    description: "List of resources to cluster".to_string(),
                },
                VectorServiceParameter {
                    name: "num_clusters".to_string(),
                    param_type: VectorParameterType::Number,
                    required: false,
                    description: "Number of clusters to create".to_string(),
                },
            ],
        });
    }

    /// Register a custom vector service function
    pub fn register_function(&mut self, function: VectorServiceFunction) {
        self.function_registry.insert(function.name.clone(), function);
    }

    /// Register a custom vector function implementation
    pub fn register_custom_function(
        &mut self,
        name: String,
        function: Box<dyn CustomVectorFunction>,
    ) {
        self.custom_functions.insert(name, function);
    }

    /// Execute a SPARQL vector function
    pub fn execute_function(
        &self,
        function_name: &str,
        args: &[VectorServiceArg],
        executor: &mut QueryExecutor,
    ) -> Result<VectorServiceResult> {
        // Check if it's a custom function first
        if let Some(custom_func) = self.custom_functions.get(function_name) {
            return custom_func.execute(args);
        }

        // Check if it's a built-in function
        if let Some(func_def) = self.function_registry.get(function_name) {
            // Validate arity if specified
            if func_def.arity > 0 && args.len() > func_def.arity {
                return Err(anyhow!(
                    "Function {} expects at most {} arguments, got {}",
                    function_name,
                    func_def.arity,
                    args.len()
                ));
            }

            // Create a query for the function
            let query = VectorQuery::new(function_name.to_string(), args.to_vec());
            let result = executor.execute_optimized_query(&query)?;

            // Convert VectorQueryResult to VectorServiceResult
            Ok(VectorServiceResult::SimilarityList(result.results))
        } else {
            Err(anyhow!("Unknown function: {}", function_name))
        }
    }

    /// Get function definition
    pub fn get_function(&self, name: &str) -> Option<&VectorServiceFunction> {
        self.function_registry.get(name)
    }

    /// Get all registered functions
    pub fn get_all_functions(&self) -> &HashMap<String, VectorServiceFunction> {
        &self.function_registry
    }

    /// Check if a function is registered
    pub fn is_function_registered(&self, name: &str) -> bool {
        self.function_registry.contains_key(name) || self.custom_functions.contains_key(name)
    }

    /// Get function documentation
    pub fn get_function_documentation(&self, name: &str) -> Option<String> {
        if let Some(func) = self.function_registry.get(name) {
            let mut doc = format!("Function: {}\n", func.name);
            doc.push_str(&format!("Description: {}\n", func.description));
            doc.push_str(&format!("Arity: {}\n", func.arity));
            doc.push_str("Parameters:\n");
            
            for param in &func.parameters {
                doc.push_str(&format!(
                    "  - {} ({:?}{}): {}\n",
                    param.name,
                    param.param_type,
                    if param.required { ", required" } else { ", optional" },
                    param.description
                ));
            }
            
            Some(doc)
        } else if let Some(custom_func) = self.custom_functions.get(name) {
            Some(format!(
                "Custom Function: {}\nDescription: {}\nArity: {}",
                name,
                custom_func.description(),
                custom_func.arity()
            ))
        } else {
            None
        }
    }

    /// Generate SPARQL function definitions for documentation
    pub fn generate_sparql_definitions(&self) -> String {
        let mut definitions = String::new();
        definitions.push_str("# OxiRS Vector SPARQL Functions\n\n");
        
        for (name, func) in &self.function_registry {
            definitions.push_str(&format!("## vec:{}\n\n", name));
            definitions.push_str(&format!("**Description:** {}\n\n", func.description));
            
            if func.arity > 0 {
                definitions.push_str(&format!("**Arity:** {}\n\n", func.arity));
            }
            
            definitions.push_str("**Parameters:**\n\n");
            for param in &func.parameters {
                definitions.push_str(&format!(
                    "- `{}` ({:?}{}) - {}\n",
                    param.name,
                    param.param_type,
                    if param.required { ", required" } else { ", optional" },
                    param.description
                ));
            }
            
            // Add usage example
            definitions.push_str("\n**Example:**\n\n");
            definitions.push_str("```sparql\n");
            match name.as_str() {
                "similarity" => {
                    definitions.push_str("SELECT ?score WHERE {\n");
                    definitions.push_str("  BIND(vec:similarity(<http://example.org/doc1>, <http://example.org/doc2>) AS ?score)\n");
                    definitions.push_str("}\n");
                }
                "similar" => {
                    definitions.push_str("SELECT ?similar ?score WHERE {\n");
                    definitions.push_str("  (?similar ?score) vec:similar (<http://example.org/doc1>, 10, 0.7)\n");
                    definitions.push_str("}\n");
                }
                "search" => {
                    definitions.push_str("SELECT ?resource ?score WHERE {\n");
                    definitions.push_str("  (?resource ?score) vec:search (\"machine learning\", 10, 0.7)\n");
                    definitions.push_str("}\n");
                }
                "searchIn" => {
                    definitions.push_str("SELECT ?resource ?score WHERE {\n");
                    definitions.push_str("  (?resource ?score) vec:searchIn (\"AI research\", <http://example.org/graph1>, 10, \"exact\", 0.7)\n");
                    definitions.push_str("}\n");
                }
                "embed" => {
                    definitions.push_str("SELECT ?embedding WHERE {\n");
                    definitions.push_str("  BIND(vec:embed(\"example text\") AS ?embedding)\n");
                    definitions.push_str("}\n");
                }
                _ => {
                    definitions.push_str(&format!("# Example usage for vec:{}\n", name));
                }
            }
            definitions.push_str("```\n\n");
        }
        
        definitions
    }
}

impl Default for SparqlVectorFunctions {
    fn default() -> Self {
        Self::new()
    }
}

/// Example custom function implementation
pub struct CosineSimilarityFunction;

impl CustomVectorFunction for CosineSimilarityFunction {
    fn execute(&self, args: &[VectorServiceArg]) -> Result<VectorServiceResult> {
        if args.len() != 2 {
            return Err(anyhow!("Cosine similarity function requires exactly 2 arguments"));
        }

        let vector1 = match &args[0] {
            VectorServiceArg::Vector(v) => v,
            _ => return Err(anyhow!("First argument must be a vector")),
        };

        let vector2 = match &args[1] {
            VectorServiceArg::Vector(v) => v,
            _ => return Err(anyhow!("Second argument must be a vector")),
        };

        let similarity = crate::similarity::cosine_similarity(vector1, vector2)
            .map_err(|e| anyhow!("Failed to calculate cosine similarity: {}", e))?;

        Ok(VectorServiceResult::Number(similarity))
    }

    fn arity(&self) -> usize {
        2
    }

    fn description(&self) -> String {
        "Calculate cosine similarity between two vectors".to_string()
    }
}

/// Example aggregate function implementation
pub struct AverageSimilarityFunction;

impl CustomVectorFunction for AverageSimilarityFunction {
    fn execute(&self, args: &[VectorServiceArg]) -> Result<VectorServiceResult> {
        if args.is_empty() {
            return Err(anyhow!("Average similarity function requires at least 1 argument"));
        }

        let mut similarities = Vec::new();
        
        for arg in args {
            match arg {
                VectorServiceArg::Number(sim) => similarities.push(*sim),
                _ => return Err(anyhow!("All arguments must be numbers (similarity scores)")),
            }
        }

        let average = similarities.iter().sum::<f32>() / similarities.len() as f32;
        Ok(VectorServiceResult::Number(average))
    }

    fn arity(&self) -> usize {
        0 // Variable arity
    }

    fn description(&self) -> String {
        "Calculate average of multiple similarity scores".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;

    #[test]
    fn test_function_registration() {
        let functions = SparqlVectorFunctions::new();
        
        assert!(functions.is_function_registered("similarity"));
        assert!(functions.is_function_registered("similar"));
        assert!(functions.is_function_registered("search"));
        assert!(functions.is_function_registered("searchIn"));
        assert!(!functions.is_function_registered("nonexistent"));
    }

    #[test]
    fn test_custom_function_registration() {
        let mut functions = SparqlVectorFunctions::new();
        
        let custom_func = Box::new(CosineSimilarityFunction);
        functions.register_custom_function("custom_cosine".to_string(), custom_func);
        
        assert!(functions.is_function_registered("custom_cosine"));
    }

    #[test]
    fn test_custom_function_execution() {
        let func = CosineSimilarityFunction;
        
        let vector1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let vector2 = Vector::new(vec![0.0, 1.0, 0.0]);
        
        let args = vec![
            VectorServiceArg::Vector(vector1),
            VectorServiceArg::Vector(vector2),
        ];
        
        let result = func.execute(&args).unwrap();
        
        match result {
            VectorServiceResult::Number(similarity) => {
                assert!((similarity - 0.0).abs() < 1e-6); // Orthogonal vectors
            }
            _ => panic!("Expected number result"),
        }
    }

    #[test]
    fn test_function_documentation() {
        let functions = SparqlVectorFunctions::new();
        
        let doc = functions.get_function_documentation("similarity").unwrap();
        assert!(doc.contains("similarity"));
        assert!(doc.contains("Calculate similarity"));
        assert!(doc.contains("resource1"));
        assert!(doc.contains("resource2"));
    }

    #[test]
    fn test_sparql_definitions_generation() {
        let functions = SparqlVectorFunctions::new();
        
        let definitions = functions.generate_sparql_definitions();
        assert!(definitions.contains("vec:similarity"));
        assert!(definitions.contains("vec:search"));
        assert!(definitions.contains("SELECT"));
        assert!(definitions.contains("```sparql"));
    }

    #[test]
    fn test_average_similarity_function() {
        let func = AverageSimilarityFunction;
        
        let args = vec![
            VectorServiceArg::Number(0.8),
            VectorServiceArg::Number(0.9),
            VectorServiceArg::Number(0.7),
        ];
        
        let result = func.execute(&args).unwrap();
        
        match result {
            VectorServiceResult::Number(average) => {
                assert!((average - 0.8).abs() < 1e-6);
            }
            _ => panic!("Expected number result"),
        }
    }
}