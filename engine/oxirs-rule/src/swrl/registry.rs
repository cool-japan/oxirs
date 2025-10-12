//! SWRL (Semantic Web Rule Language) - Custom Builtin Registry
//!
//! This module implements SWRL rule components.

use anyhow::Result;
use std::collections::HashMap;
use tracing::info;

use super::types::SwrlArgument;

/// Type alias for built-in function implementations
type BuiltinFunctionImpl = Box<dyn Fn(&[SwrlArgument]) -> Result<bool> + Send + Sync>;

/// Custom built-in registry for user-defined functions
pub struct CustomBuiltinRegistry {
    /// Registry of custom built-in functions
    functions: HashMap<String, BuiltinFunctionImpl>,
    /// Metadata about registered functions
    metadata: HashMap<String, BuiltinMetadata>,
}

/// Metadata for custom built-in functions
#[derive(Debug, Clone)]
pub struct BuiltinMetadata {
    pub name: String,
    pub namespace: String,
    pub description: String,
    pub min_args: usize,
    pub max_args: Option<usize>,
    pub example_usage: String,
}

impl CustomBuiltinRegistry {
    /// Create a new custom built-in registry
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Register a custom built-in function
    pub fn register<F>(&mut self, metadata: BuiltinMetadata, function: F) -> Result<()>
    where
        F: Fn(&[SwrlArgument]) -> Result<bool> + Send + Sync + 'static,
    {
        let full_name = format!("{}{}", metadata.namespace, metadata.name);

        if self.functions.contains_key(&full_name) {
            return Err(anyhow::anyhow!(
                "Built-in function '{}' already registered",
                full_name
            ));
        }

        self.functions.insert(full_name.clone(), Box::new(function));
        self.metadata.insert(full_name.clone(), metadata);

        info!("Registered custom built-in function: {}", full_name);
        Ok(())
    }

    /// Execute a custom built-in function
    pub fn execute(&self, name: &str, args: &[SwrlArgument]) -> Result<bool> {
        if let Some(function) = self.functions.get(name) {
            // Validate argument count
            if let Some(meta) = self.metadata.get(name) {
                if args.len() < meta.min_args {
                    return Err(anyhow::anyhow!(
                        "Too few arguments for '{}': expected at least {}, got {}",
                        name,
                        meta.min_args,
                        args.len()
                    ));
                }
                if let Some(max_args) = meta.max_args {
                    if args.len() > max_args {
                        return Err(anyhow::anyhow!(
                            "Too many arguments for '{}': expected at most {}, got {}",
                            name,
                            max_args,
                            args.len()
                        ));
                    }
                }
            }

            function(args)
        } else {
            Err(anyhow::anyhow!("Unknown built-in function: {}", name))
        }
    }

    /// List all registered custom built-in functions
    pub fn list_functions(&self) -> Vec<&BuiltinMetadata> {
        self.metadata.values().collect()
    }

    /// Get metadata for a specific function
    pub fn get_metadata(&self, name: &str) -> Option<&BuiltinMetadata> {
        self.metadata.get(name)
    }
}

impl Default for CustomBuiltinRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swrl::{builtin_add, builtin_equal, builtin_string_concat, SwrlEngine};

    #[test]
    fn test_swrl_engine_creation() {
        let engine = SwrlEngine::new();
        let stats = engine.get_stats();
        assert!(stats.total_builtins > 0);
    }

    #[test]
    fn test_builtin_equal() {
        let args = vec![
            SwrlArgument::Literal("5".to_string()),
            SwrlArgument::Literal("5".to_string()),
        ];
        assert!(builtin_equal(&args).unwrap());

        let args = vec![
            SwrlArgument::Literal("5".to_string()),
            SwrlArgument::Literal("6".to_string()),
        ];
        assert!(!builtin_equal(&args).unwrap());
    }

    #[test]
    fn test_builtin_add() {
        let args = vec![
            SwrlArgument::Literal("5".to_string()),
            SwrlArgument::Literal("3".to_string()),
            SwrlArgument::Literal("8".to_string()),
        ];
        assert!(builtin_add(&args).unwrap());

        let args = vec![
            SwrlArgument::Literal("5".to_string()),
            SwrlArgument::Literal("3".to_string()),
            SwrlArgument::Literal("7".to_string()),
        ];
        assert!(!builtin_add(&args).unwrap());
    }

    #[test]
    fn test_string_concat() {
        let args = vec![
            SwrlArgument::Literal("Hello".to_string()),
            SwrlArgument::Literal(" ".to_string()),
            SwrlArgument::Literal("World".to_string()),
            SwrlArgument::Literal("Hello World".to_string()),
        ];
        assert!(builtin_string_concat(&args).unwrap());
    }

    // TODO: Re-enable when convert_swrl_to_rule is made public or test is refactored
    // #[test]
    // fn test_swrl_rule_conversion() {
    //     let engine = SwrlEngine::new();
    //
    //     let swrl_rule = SwrlRule {
    //         id: "test_rule".to_string(),
    //         body: vec![SwrlAtom::Class {
    //             class_predicate: "Person".to_string(),
    //             argument: SwrlArgument::Variable("x".to_string()),
    //         }],
    //         head: vec![SwrlAtom::Class {
    //             class_predicate: "Human".to_string(),
    //             argument: SwrlArgument::Variable("x".to_string()),
    //         }],
    //         metadata: HashMap::new(),
    //     };
    //
    //     let rule = engine.convert_swrl_to_rule(&swrl_rule).unwrap();
    //     assert_eq!(rule.name, "test_rule");
    //     assert_eq!(rule.body.len(), 1);
    //     assert_eq!(rule.head.len(), 1);
    // }
}
