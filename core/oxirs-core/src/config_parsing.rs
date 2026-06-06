//! Parser configuration for OxiRS Core.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsingConfig {
    pub buffer_sizes: HashMap<String, usize>,
    pub parsers: HashMap<String, ParserConfig>,
    pub error_handling: ParserErrorConfig,
    pub validation: ValidationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParserConfig {
    pub name: String,
    pub enable_streaming: bool,
    pub chunk_size: usize,
    pub enable_parallel: bool,
    pub worker_threads: usize,
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParserErrorConfig {
    pub tolerance: f64,
    pub continue_on_error: bool,
    pub collect_errors: bool,
    pub max_errors: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub enable_iri_validation: bool,
    pub enable_literal_validation: bool,
    pub enable_language_validation: bool,
    pub custom_rules: Vec<ValidationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub name: String,
    pub pattern: String,
    pub error_message: String,
    pub enabled: bool,
}

impl Default for ParsingConfig {
    fn default() -> Self {
        Self {
            buffer_sizes: HashMap::from([
                ("ntriples".to_string(), 64 * 1024),
                ("turtle".to_string(), 32 * 1024),
                ("rdfxml".to_string(), 128 * 1024),
                ("jsonld".to_string(), 64 * 1024),
            ]),
            parsers: HashMap::new(),
            error_handling: ParserErrorConfig::default(),
            validation: ValidationConfig::default(),
        }
    }
}

impl Default for ParserErrorConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.01,
            continue_on_error: true,
            collect_errors: true,
            max_errors: 1000,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_iri_validation: true,
            enable_literal_validation: true,
            enable_language_validation: true,
            custom_rules: Vec::new(),
        }
    }
}
