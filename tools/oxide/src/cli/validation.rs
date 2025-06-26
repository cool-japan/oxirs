//! Advanced argument validation
//!
//! Provides comprehensive validation for CLI arguments with helpful error messages.

use super::error::{CliError, CliResult};
use regex::Regex;
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
                self.errors.push(format!("{} is not a valid IRI: {}", self.name, e));
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
                    self.errors
                        .push(format!("{} must be a valid integer, got: {}", self.name, value));
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
                return Err(format!("IRI contains whitespace at position {}", i));
            }
            '<' | '>' | '"' | '{' | '}' | '|' | '^' | '`' => {
                return Err(format!("IRI contains invalid character '{}' at position {}", ch, i));
            }
            _ => {}
        }
    }

    Ok(())
}

/// Validate SPARQL endpoint URL
pub fn validate_sparql_endpoint(url: &str) -> CliResult<Url> {
    let parsed = Url::parse(url).map_err(|e| {
        CliError::invalid_arguments(format!("Invalid SPARQL endpoint URL: {}", e))
            .with_suggestion("URL should be in format: http://host:port/path")
    })?;

    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(CliError::invalid_arguments("SPARQL endpoint must use HTTP or HTTPS")
            .with_suggestion("Use http:// or https:// scheme"));
    }

    Ok(parsed)
}

/// Validate RDF format
pub fn validate_rdf_format(format: &str) -> CliResult<&str> {
    const VALID_FORMATS: &[&str] = &[
        "turtle", "ttl", "ntriples", "nt", "rdfxml", "rdf", "xml", 
        "jsonld", "json-ld", "trig", "nquads", "nq"
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
    pub fn add<'a>(&mut self, validator: ArgumentValidator<'a>) -> &mut Self {
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

        let result = ArgumentValidator::new("input", Some("value")).required().validate();
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
        
        let format_validator = ArgumentValidator::new("format", Some("invalid"))
            .one_of(&["turtle", "ntriples"]);
        validator.add(format_validator);
        
        let result = validator.finish();
        assert!(result.is_err());
    }
}