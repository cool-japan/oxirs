//! Advanced argument validation
//!
//! Provides comprehensive validation for CLI arguments with helpful error messages.

use super::error::{CliError, CliResult};
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use url::Url;

/// Argument validator with chainable validation methods
pub struct ArgumentValidator<'a> {
    name: &'a str,
    value: Option<&'a str>,
    errors: Vec<String>,
}

impl<'a> ArgumentValidator<'a> {
    /// Create a new validator for an argument
    pub fn new(name: &'a str, value: Option<&'a str>) -> Self {
        Self {
            name,
            value,
            errors: Vec::new(),
        }
    }

    /// Validate that the argument is present
    pub fn required(mut self) -> Self {
        if self.value.is_none() || self.value.map(|v| v.trim().is_empty()).unwrap_or(true) {
            self.errors.push(format!("{} is required", self.name));
        }
        self
    }

    /// Validate that the value matches a regex pattern
    pub fn matches_pattern(mut self, pattern: &str, description: &str) -> Self {
        if let Some(value) = self.value {
            if let Ok(re) = Regex::new(pattern) {
                if !re.is_match(value) {
                    self.errors.push(format!(
                        "{} must be {}, got: {}",
                        self.name, description, value
                    ));
                }
            }
        }
        self
    }

    /// Validate that the value is one of allowed values
    pub fn one_of(mut self, allowed: &[&str]) -> Self {
        if let Some(value) = self.value {
            if !allowed.contains(&value) {
                self.errors.push(format!(
                    "{} must be one of: {}, got: {}",
                    self.name,
                    allowed.join(", "),
                    value
                ));
            }
        }
        self
    }

    /// Validate that the value is a valid file path
    pub fn is_file(mut self) -> Self {
        if let Some(value) = self.value {
            let path = Path::new(value);
            if !path.exists() {
                self.errors
                    .push(format!("{} file does not exist: {}", self.name, value));
            } else if !path.is_file() {
                self.errors
                    .push(format!("{} is not a file: {}", self.name, value));
            }
        }
        self
    }

    /// Validate that the value is a valid directory path
    pub fn is_directory(mut self) -> Self {
        if let Some(value) = self.value {
            let path = Path::new(value);
            if !path.exists() {
                self.errors
                    .push(format!("{} directory does not exist: {}", self.name, value));
            } else if !path.is_dir() {
                self.errors
                    .push(format!("{} is not a directory: {}", self.name, value));
            }
        }
        self
    }

    /// Validate that the value is a valid URL
    pub fn is_url(mut self) -> Self {
        if let Some(value) = self.value {
            if Url::parse(value).is_err() {
                self.errors
                    .push(format!("{} must be a valid URL, got: {}", self.name, value));
            }
        }
        self
    }

    /// Validate that the value is a valid IRI
    pub fn is_iri(mut self) -> Self {
        if let Some(value) = self.value {
            if let Err(e) = validate_iri(value) {
                self.errors
                    .push(format!("{} is not a valid IRI: {}", self.name, e));
            }
        }
        self
    }

    /// Validate that the value is a valid port number
    pub fn is_port(mut self) -> Self {
        if let Some(value) = self.value {
            match value.parse::<u16>() {
                Ok(port) if port > 0 => {}
                _ => {
                    self.errors.push(format!(
                        "{} must be a valid port number (1-65535), got: {}",
                        self.name, value
                    ));
                }
            }
        }
        self
    }

    /// Validate integer within range
    pub fn integer_range(mut self, min: Option<i64>, max: Option<i64>) -> Self {
        if let Some(value) = self.value {
            match value.parse::<i64>() {
                Ok(num) => {
                    if let Some(min_val) = min {
                        if num < min_val {
                            self.errors.push(format!(
                                "{} must be at least {}, got: {}",
                                self.name, min_val, num
                            ));
                        }
                    }
                    if let Some(max_val) = max {
                        if num > max_val {
                            self.errors.push(format!(
                                "{} must be at most {}, got: {}",
                                self.name, max_val, num
                            ));
                        }
                    }
                }
                Err(_) => {
                    self.errors.push(format!(
                        "{} must be a valid integer, got: {}",
                        self.name, value
                    ));
                }
            }
        }
        self
    }

    /// Custom validation function
    pub fn custom<F>(mut self, validator: F, error_msg: &str) -> Self
    where
        F: Fn(&str) -> bool,
    {
        if let Some(value) = self.value {
            if !validator(value) {
                self.errors.push(format!("{}: {}", self.name, error_msg));
            }
        }
        self
    }

    /// Complete validation and return result
    pub fn validate(self) -> CliResult<()> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(CliError::invalid_arguments(self.errors.join("; "))
                .with_context(format!("Validating argument: {}", self.name)))
        }
    }

    /// Get validation errors without failing
    pub fn errors(self) -> Vec<String> {
        self.errors
    }
}

/// Validate IRI according to RFC 3987
pub fn validate_iri(iri: &str) -> Result<(), String> {
    if iri.is_empty() {
        return Err("IRI cannot be empty".to_string());
    }

    // Basic validation - a more complete implementation would follow RFC 3987
    if !iri.contains(':') {
        return Err("IRI must contain a scheme".to_string());
    }

    // Check for invalid characters
    for (i, ch) in iri.chars().enumerate() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                return Err(format!("IRI contains whitespace at position {i}"));
            }
            '<' | '>' | '"' | '{' | '}' | '|' | '^' | '`' => {
                return Err(format!(
                    "IRI contains invalid character '{ch}' at position {i}"
                ));
            }
            _ => {}
        }
    }

    Ok(())
}

/// Validate SPARQL endpoint URL
pub fn validate_sparql_endpoint(url: &str) -> CliResult<Url> {
    let parsed = Url::parse(url).map_err(|e| {
        CliError::invalid_arguments(format!("Invalid SPARQL endpoint URL: {e}"))
            .with_suggestion("URL should be in format: http://host:port/path")
    })?;

    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(
            CliError::invalid_arguments("SPARQL endpoint must use HTTP or HTTPS")
                .with_suggestion("Use http:// or https:// scheme"),
        );
    }

    Ok(parsed)
}

/// Validate RDF format
pub fn validate_rdf_format(format: &str) -> CliResult<&str> {
    const VALID_FORMATS: &[&str] = &[
        "turtle", "ttl", "ntriples", "nt", "rdfxml", "rdf", "xml", "jsonld", "json-ld", "trig",
        "nquads", "nq",
    ];

    let normalized = format.to_lowercase();
    if VALID_FORMATS.contains(&normalized.as_str()) {
        Ok(format)
    } else {
        Err(CliError::invalid_format(format)
            .with_context("Invalid RDF format")
            .with_suggestions(vec![
                format!("Valid formats: {}", VALID_FORMATS.join(", ")),
                "Use file extension for auto-detection".to_string(),
            ]))
    }
}

/// Builder for validating multiple arguments
pub struct MultiValidator {
    errors: Vec<String>,
}

impl MultiValidator {
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Add a validator
    pub fn add(&mut self, validator: ArgumentValidator<'_>) -> &mut Self {
        let errors = validator.errors();
        self.errors.extend(errors);
        self
    }

    /// Validate argument
    pub fn validate<'a>(&mut self, name: &'a str, value: Option<&'a str>) -> ArgumentValidator<'a> {
        ArgumentValidator::new(name, value)
    }

    /// Complete validation
    pub fn finish(self) -> CliResult<()> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(CliError::invalid_arguments(self.errors.join("\n"))
                .with_context("Multiple validation errors"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_validation() {
        let result = ArgumentValidator::new("input", None).required().validate();
        assert!(result.is_err());

        let result = ArgumentValidator::new("input", Some("value"))
            .required()
            .validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_pattern_validation() {
        let result = ArgumentValidator::new("lang", Some("en-US"))
            .matches_pattern(r"^[a-z]{2}-[A-Z]{2}$", "language-COUNTRY format")
            .validate();
        assert!(result.is_ok());

        let result = ArgumentValidator::new("lang", Some("invalid"))
            .matches_pattern(r"^[a-z]{2}-[A-Z]{2}$", "language-COUNTRY format")
            .validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_one_of_validation() {
        let result = ArgumentValidator::new("format", Some("turtle"))
            .one_of(&["turtle", "ntriples", "rdfxml"])
            .validate();
        assert!(result.is_ok());

        let result = ArgumentValidator::new("format", Some("invalid"))
            .one_of(&["turtle", "ntriples", "rdfxml"])
            .validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_url_validation() {
        let result = ArgumentValidator::new("endpoint", Some("http://localhost:3030"))
            .is_url()
            .validate();
        assert!(result.is_ok());

        let result = ArgumentValidator::new("endpoint", Some("not a url"))
            .is_url()
            .validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_iri_validation() {
        assert!(validate_iri("http://example.org/resource").is_ok());
        assert!(validate_iri("urn:uuid:12345").is_ok());
        assert!(validate_iri("").is_err());
        assert!(validate_iri("no scheme").is_err());
        assert!(validate_iri("http://example.org/has space").is_err());
    }

    #[test]
    fn test_multi_validator() {
        let mut validator = MultiValidator::new();

        let port_validator = ArgumentValidator::new("port", Some("abc")).is_port();
        validator.add(port_validator);

        let format_validator =
            ArgumentValidator::new("format", Some("invalid")).one_of(&["turtle", "ntriples"]);
        validator.add(format_validator);

        let result = validator.finish();
        assert!(result.is_err());
    }
}

/// Advanced validation context for complex validation scenarios
pub struct ValidationContext {
    pub environment: HashMap<String, String>,
    pub dependencies: HashMap<String, Vec<String>>,
    pub mutually_exclusive: Vec<Vec<String>>,
    pub required_together: Vec<Vec<String>>,
}

impl ValidationContext {
    pub fn new() -> Self {
        Self {
            environment: std::env::vars().collect(),
            dependencies: HashMap::new(),
            mutually_exclusive: Vec::new(),
            required_together: Vec::new(),
        }
    }

    /// Add dependency: if arg1 is present, arg2 must also be present
    pub fn add_dependency(&mut self, arg: &str, depends_on: &str) {
        self.dependencies
            .entry(arg.to_string())
            .or_default()
            .push(depends_on.to_string());
    }

    /// Add mutually exclusive group
    pub fn add_mutually_exclusive(&mut self, args: Vec<&str>) {
        self.mutually_exclusive
            .push(args.into_iter().map(|s| s.to_string()).collect());
    }

    /// Add required together group
    pub fn add_required_together(&mut self, args: Vec<&str>) {
        self.required_together
            .push(args.into_iter().map(|s| s.to_string()).collect());
    }

    /// Validate argument dependencies
    pub fn validate_dependencies(&self, present_args: &[&str]) -> CliResult<()> {
        let mut errors = Vec::new();

        // Check dependencies
        for arg in present_args {
            if let Some(deps) = self.dependencies.get(*arg) {
                for dep in deps {
                    if !present_args.contains(&dep.as_str()) {
                        errors.push(format!("--{arg} requires --{dep} to be specified"));
                    }
                }
            }
        }

        // Check mutually exclusive
        for group in &self.mutually_exclusive {
            let present_in_group: Vec<_> = group
                .iter()
                .filter(|arg| present_args.contains(&arg.as_str()))
                .collect();

            if present_in_group.len() > 1 {
                errors.push(format!(
                    "The following arguments cannot be used together: {}",
                    present_in_group
                        .iter()
                        .map(|s| format!("--{s}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        // Check required together
        for group in &self.required_together {
            let present_in_group: Vec<_> = group
                .iter()
                .filter(|arg| present_args.contains(&arg.as_str()))
                .collect();

            if !present_in_group.is_empty() && present_in_group.len() != group.len() {
                let missing: Vec<_> = group
                    .iter()
                    .filter(|arg| !present_args.contains(&arg.as_str()))
                    .collect();

                errors.push(format!(
                    "When using --{}, you must also specify: {}",
                    present_in_group[0],
                    missing
                        .iter()
                        .map(|s| format!("--{s}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CliError::invalid_arguments(errors.join("\n")))
        }
    }
}

/// File system validation utilities
pub mod fs_validation {
    use super::*;
    use std::fs;

    /// Validate that a path is writable
    pub fn validate_writable_path(path: &Path) -> CliResult<()> {
        if path.exists() {
            // Check if we can write to existing file/directory
            let metadata = fs::metadata(path).map_err(|e| {
                CliError::io_error(e)
                    .with_context(format!("Cannot access path: {}", path.display()))
            })?;

            if metadata.permissions().readonly() {
                return Err(CliError::invalid_arguments(format!(
                    "Path is read-only: {}",
                    path.display()
                )));
            }
        } else {
            // Check if we can create in parent directory
            if let Some(parent) = path.parent() {
                if parent.exists() && !parent.is_dir() {
                    return Err(CliError::invalid_arguments(format!(
                        "Parent path is not a directory: {}",
                        parent.display()
                    )));
                }

                if parent.exists() {
                    let parent_metadata = fs::metadata(parent).map_err(|e| {
                        CliError::io_error(e).with_context(format!(
                            "Cannot access parent directory: {}",
                            parent.display()
                        ))
                    })?;

                    if parent_metadata.permissions().readonly() {
                        return Err(CliError::invalid_arguments(format!(
                            "Parent directory is read-only: {}",
                            parent.display()
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Validate that a file has expected extension
    pub fn validate_file_extension(path: &Path, valid_extensions: &[&str]) -> CliResult<()> {
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        if !valid_extensions.contains(&extension) {
            return Err(CliError::invalid_arguments(format!(
                "Invalid file extension '{}'. Expected one of: {}",
                extension,
                valid_extensions.join(", ")
            ))
            .with_suggestion(format!(
                "Use a file with extension: {}",
                valid_extensions.join(" or ")
            )));
        }
        Ok(())
    }

    /// Validate file size constraints
    pub fn validate_file_size(path: &Path, max_size: Option<u64>) -> CliResult<()> {
        if let Some(max) = max_size {
            let metadata = fs::metadata(path).map_err(|e| {
                CliError::io_error(e)
                    .with_context(format!("Cannot access file: {}", path.display()))
            })?;

            if metadata.len() > max {
                return Err(CliError::invalid_arguments(format!(
                    "File too large: {} bytes (max: {} bytes)",
                    metadata.len(),
                    max
                ))
                .with_suggestion("Use a smaller file or increase the limit"));
            }
        }
        Ok(())
    }
}

/// Dataset validation utilities
pub mod dataset_validation {
    use super::*;

    /// Validate dataset name format
    pub fn validate_dataset_name(name: &str) -> CliResult<()> {
        if name.is_empty() {
            return Err(CliError::invalid_arguments("Dataset name cannot be empty"));
        }

        // Check for valid characters
        let valid_pattern = Regex::new(r"^[a-zA-Z0-9_-]+$").unwrap();
        if !valid_pattern.is_match(name) {
            return Err(CliError::invalid_arguments(
                format!("Invalid dataset name: '{name}'. Must contain only letters, numbers, underscores, and hyphens")
            ));
        }

        // Check length
        if name.len() > 255 {
            return Err(CliError::invalid_arguments(
                "Dataset name too long (max 255 characters)",
            ));
        }

        Ok(())
    }

    /// Validate graph URI
    pub fn validate_graph_uri(uri: &str) -> CliResult<()> {
        if uri == "default" || uri.is_empty() {
            return Ok(()); // Default graph is valid
        }

        validate_iri(uri).map_err(|e| {
            CliError::invalid_arguments(format!("Invalid graph URI: {e}"))
                .with_suggestion("Use 'default' for the default graph or a valid IRI")
        })
    }
}

/// Query validation utilities
pub mod query_validation {
    use super::*;

    /// Enhanced SPARQL query syntax validation with optimization hints
    pub fn validate_sparql_syntax(query: &str) -> CliResult<()> {
        if query.trim().is_empty() {
            return Err(CliError::invalid_arguments("Query cannot be empty"));
        }

        let query_upper = query.to_uppercase();

        // Check for basic SPARQL keywords
        let has_valid_keyword = ["SELECT", "CONSTRUCT", "DESCRIBE", "ASK"]
            .iter()
            .any(|&kw| query_upper.contains(kw));

        if !has_valid_keyword {
            return Err(CliError::invalid_arguments(
                "Query must contain SELECT, CONSTRUCT, DESCRIBE, or ASK",
            )
            .with_suggestion("Check your SPARQL query syntax"));
        }

        // Enhanced validation checks
        let mut warnings: Vec<String> = Vec::new();

        // Check for WHERE clause in SELECT/CONSTRUCT/DESCRIBE queries
        if (query_upper.contains("SELECT")
            || query_upper.contains("CONSTRUCT")
            || query_upper.contains("DESCRIBE"))
            && !query_upper.contains("WHERE")
        {
            warnings.push("Query should typically include a WHERE clause".to_string());
        }

        // Check for balanced braces
        let open_braces = query.matches('{').count();
        let close_braces = query.matches('}').count();
        if open_braces != close_braces {
            return Err(CliError::invalid_arguments(format!(
                "Unbalanced braces: {} opening, {} closing",
                open_braces, close_braces
            ))
            .with_suggestion("Ensure all {{ have matching }}"));
        }

        // Check for missing LIMIT on SELECT queries (performance hint)
        if query_upper.contains("SELECT")
            && !query_upper.contains("LIMIT")
            && !query_upper.contains("COUNT")
        {
            warnings
                .push("Consider adding LIMIT for better performance on large datasets".to_string());
        }

        // Check for SELECT * (best practice hint)
        if query_upper.contains("SELECT *") && !query_upper.contains("COUNT") {
            warnings.push(
                "SELECT * may be inefficient; consider selecting specific variables".to_string(),
            );
        }

        // Check for common prefix usage without PREFIX declaration
        let common_prefixes = [
            (
                "rdf:",
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            ),
            (
                "rdfs:",
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            ),
            ("owl:", "PREFIX owl: <http://www.w3.org/2002/07/owl#>"),
            ("xsd:", "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"),
        ];

        for (prefix, declaration) in &common_prefixes {
            if query.contains(prefix)
                && !query_upper.contains(&format!(
                    "PREFIX {}",
                    prefix.to_uppercase().trim_end_matches(':')
                ))
            {
                warnings.push(format!(
                    "Missing prefix declaration? Consider: {}",
                    declaration
                ));
            }
        }

        // Log warnings if any (for verbose mode)
        if !warnings.is_empty() {
            tracing::debug!("SPARQL validation hints: {}", warnings.join("; "));
        }

        Ok(())
    }

    /// Basic SPARQL update syntax validation
    pub fn validate_sparql_update_syntax(update: &str) -> CliResult<()> {
        if update.trim().is_empty() {
            return Err(CliError::invalid_arguments("Update cannot be empty"));
        }

        // Check for basic SPARQL Update keywords
        let update_upper = update.to_uppercase();
        let has_valid_keyword = [
            "INSERT", "DELETE", "LOAD", "CLEAR", "CREATE", "DROP", "COPY", "MOVE", "ADD",
        ]
        .iter()
        .any(|&kw| update_upper.contains(kw));

        if !has_valid_keyword {
            return Err(CliError::invalid_arguments(
                "Update must contain valid SPARQL Update operation",
            )
            .with_suggestion(
                "Valid operations: INSERT, DELETE, LOAD, CLEAR, CREATE, DROP, COPY, MOVE, ADD",
            ));
        }

        Ok(())
    }

    /// Estimate query complexity for performance hints
    ///
    /// Returns a complexity score (1-10) where:
    /// - 1-3: Simple query (fast)
    /// - 4-6: Moderate query (may take time on large datasets)
    /// - 7-10: Complex query (may be slow, consider optimization)
    pub fn estimate_query_complexity(query: &str) -> u8 {
        let query_upper = query.to_uppercase();
        let mut complexity = 1;

        // Base complexity on query type
        if query_upper.contains("SELECT") {
            complexity += 1;
        }
        if query_upper.contains("CONSTRUCT") || query_upper.contains("DESCRIBE") {
            complexity += 2;
        }

        // Check for operations that increase complexity
        if query_upper.contains("OPTIONAL") {
            complexity += 1;
        }
        if query_upper.contains("UNION") {
            complexity += 1;
        }
        if query_upper.contains("FILTER") {
            complexity += 1;
        }
        if query_upper.contains("REGEX") {
            complexity += 2; // Regex can be expensive
        }
        if query_upper.contains("GROUP BY") {
            complexity += 1;
        }
        if query_upper.contains("ORDER BY") {
            complexity += 1;
        }
        if query_upper.contains("SERVICE") {
            complexity += 2; // Federated queries can be slow
        }

        // Check for patterns that may indicate Cartesian products
        let triple_pattern_count = query.matches("?").count();
        if triple_pattern_count > 10 {
            complexity += 1;
        }

        // Check if query lacks LIMIT (can be very slow)
        if query_upper.contains("SELECT") && !query_upper.contains("LIMIT") {
            complexity += 1;
        }

        // Check for nested subqueries
        let brace_count = query.matches('{').count();
        let subquery_count = brace_count.saturating_sub(1); // -1 for main WHERE, prevent underflow
        complexity += (subquery_count / 2) as u8;

        complexity.min(10)
    }

    /// Get human-readable complexity description
    pub fn complexity_description(complexity: u8) -> &'static str {
        match complexity {
            1..=3 => "Simple query - should execute quickly",
            4..=6 => "Moderate complexity - may take time on large datasets",
            7..=10 => "Complex query - consider optimization or adding LIMIT",
            _ => "Unknown complexity",
        }
    }
}

/// Environment-aware validation
pub mod env_validation {
    use super::*;

    /// Validate based on environment variables
    pub fn validate_with_env(name: &str, value: Option<&str>) -> Option<String> {
        if value.is_some() {
            return value.map(|s| s.to_string());
        }

        // Try to get from environment
        std::env::var(format!("OXIRS_{}", name.to_uppercase())).ok()
    }

    /// Check if running in production mode
    pub fn is_production() -> bool {
        std::env::var("OXIRS_ENV")
            .unwrap_or_else(|_| "development".to_string())
            .to_lowercase()
            == "production"
    }

    /// Validate production requirements
    pub fn validate_production_config() -> CliResult<()> {
        if !is_production() {
            return Ok(());
        }

        let mut errors = Vec::new();

        // Check required production environment variables
        let required_vars = ["OXIRS_SECRET_KEY", "OXIRS_DATABASE_URL"];
        for var in &required_vars {
            if std::env::var(var).is_err() {
                errors.push(format!("{var} must be set in production"));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CliError::invalid_arguments(errors.join("\n"))
                .with_context("Production configuration validation failed"))
        }
    }
}
impl Default for ValidationContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MultiValidator {
    fn default() -> Self {
        Self::new()
    }
}
