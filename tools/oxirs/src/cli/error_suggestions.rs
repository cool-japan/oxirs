//! Smart error message suggestions
//!
//! Provides intelligent suggestions for common errors based on context analysis.

use super::error::{CliError, CliErrorKind};

/// Enhance error messages with context-aware suggestions
pub fn enhance_error(mut error: CliError) -> CliError {
    // Clone the message before matching to avoid borrow checker issues
    let kind_message = match &error.kind {
        CliErrorKind::NotFound(msg) => Some(("not_found", msg.clone())),
        CliErrorKind::InvalidFormat(msg) => Some(("invalid_format", msg.clone())),
        CliErrorKind::ValidationError(msg) => Some(("validation", msg.clone())),
        CliErrorKind::ConfigError(msg) => Some(("config", msg.clone())),
        _ => None,
    };

    if let Some((kind, msg)) = kind_message {
        error = match kind {
            "not_found" => enhance_not_found_error(error, &msg),
            "invalid_format" => enhance_invalid_format_error(error, &msg),
            "validation" => enhance_validation_error(error, &msg),
            "config" => enhance_config_error(error, &msg),
            _ => error,
        };
    }

    error
}

/// Enhance "not found" errors with suggestions
fn enhance_not_found_error(mut error: CliError, resource: &str) -> CliError {
    // Dataset not found suggestions
    if resource.contains("dataset") || resource.contains("Dataset") {
        error = error
            .with_suggestion("Initialize a new dataset with: oxirs init <dataset-name>")
            .with_suggestion("List available datasets with: ls ~/.oxirs/datasets/")
            .with_suggestion("Check the dataset path in your configuration file")
            .with_code("E001");
    }
    // File not found suggestions
    else if resource.contains(".ttl")
        || resource.contains(".nt")
        || resource.contains(".rdf")
        || resource.contains(".jsonld")
    {
        error = error
            .with_suggestion("Verify the file path is correct and the file exists")
            .with_suggestion("Check file permissions with: ls -l <file-path>")
            .with_suggestion("Use absolute paths to avoid ambiguity")
            .with_code("E002");
    }
    // Query file not found
    else if resource.contains(".sparql") || resource.contains(".rq") {
        error = error
            .with_suggestion("Verify the query file exists: ls -l <query-file>")
            .with_suggestion("Try providing the query inline instead of using --file")
            .with_code("E003");
    }
    // Config file not found
    else if resource.contains("oxirs.toml") || resource.contains(".toml") {
        error = error
            .with_suggestion("Initialize configuration with: oxirs config init")
            .with_suggestion("Specify a different config path with --config <path>")
            .with_suggestion("Check example configurations in docs/examples/")
            .with_code("E004");
    }

    error
}

/// Enhance "invalid format" errors with suggestions
fn enhance_invalid_format_error(mut error: CliError, msg: &str) -> CliError {
    let msg_lower = msg.to_lowercase();

    // RDF format errors
    if msg_lower.contains("format")
        && (msg_lower.contains("turtle")
            || msg_lower.contains("ntriples")
            || msg_lower.contains("rdf"))
    {
        error = error
            .with_suggestion(
                "Supported RDF formats: turtle, ntriples, nquads, trig, rdfxml, jsonld, n3",
            )
            .with_suggestion("Use --format <format> to specify the format explicitly")
            .with_suggestion("Let oxirs auto-detect the format by omitting --format")
            .with_code("E010");
    }
    // Output format errors
    else if msg_lower.contains("output") || msg_lower.contains("results") {
        error = error
            .with_suggestion(
                "Supported output formats: table, json, csv, tsv, xml, html, markdown, pdf",
            )
            .with_suggestion("Use --output <format> to specify the output format")
            .with_code("E011");
    }
    // SPARQL syntax errors
    else if msg_lower.contains("sparql") || msg_lower.contains("query") {
        error = error
            .with_suggestion("Validate your SPARQL syntax with: oxirs qparse <query-file>")
            .with_suggestion(
                "Check for common mistakes: missing PREFIX declarations, unbalanced braces",
            )
            .with_suggestion(
                "Consult SPARQL 1.1 specification: https://www.w3.org/TR/sparql11-query/",
            )
            .with_code("E012");
    }

    error
}

/// Enhance validation errors with suggestions
fn enhance_validation_error(mut error: CliError, msg: &str) -> CliError {
    let msg_lower = msg.to_lowercase();

    // Dataset name validation
    if msg_lower.contains("dataset name") || msg_lower.contains("dataset") {
        error = error
            .with_suggestion(
                "Dataset names must be alphanumeric (allow _, - but no dots or slashes)",
            )
            .with_suggestion("Valid examples: my-dataset, test_db, production2024")
            .with_suggestion("Invalid examples: my.dataset, test/db, prod.v1")
            .with_code("E020");
    }
    // IRI/URI validation
    else if msg_lower.contains("iri") || msg_lower.contains("uri") {
        error = error
            .with_suggestion(
                "IRIs must be valid URIs enclosed in angle brackets: <http://example.org/resource>",
            )
            .with_suggestion("Use prefixed names for brevity: ex:resource")
            .with_suggestion("Validate IRIs with: oxirs iri <uri>")
            .with_code("E021");
    }
    // SPARQL query complexity
    else if msg_lower.contains("complex") || msg_lower.contains("complexity") {
        error = error
            .with_suggestion("Simplify your query by breaking it into smaller subqueries")
            .with_suggestion("Use LIMIT to reduce result set size")
            .with_suggestion("Add indexes for frequently queried properties")
            .with_suggestion("Profile query execution with: oxirs explain <dataset> <query>")
            .with_code("E022");
    }

    error
}

/// Enhance configuration errors with suggestions
fn enhance_config_error(mut error: CliError, msg: &str) -> CliError {
    let msg_lower = msg.to_lowercase();

    // Missing configuration
    if msg_lower.contains("missing") || msg_lower.contains("not found") {
        error = error
            .with_suggestion("Create a default configuration with: oxirs config init")
            .with_suggestion("Copy an example config: cp docs/examples/oxirs.toml .")
            .with_code("E030");
    }
    // Invalid TOML syntax
    else if msg_lower.contains("toml") || msg_lower.contains("parse") {
        error = error
            .with_suggestion("Validate TOML syntax with: oxirs config validate")
            .with_suggestion("Common TOML errors: unquoted strings, missing closing brackets")
            .with_suggestion("Check TOML syntax: https://toml.io/en/")
            .with_code("E031");
    }
    // Invalid dataset configuration
    else if msg_lower.contains("dataset") {
        error = error
            .with_suggestion("Ensure [datasets.<name>] section exists in oxirs.toml")
            .with_suggestion("Required fields: name, location, dataset_type")
            .with_suggestion("Example: [datasets.mydb]\\nname = \"mydb\"\\nlocation = \"./data\"")
            .with_code("E032");
    }

    error
}

/// Create an enhanced error from a generic error message
pub fn enhanced_error_from_message(message: impl Into<String>) -> CliError {
    let msg = message.into();
    let msg_lower = msg.to_lowercase();

    // Detect error type from message content
    let error = if msg_lower.contains("not found") || msg_lower.contains("no such file") {
        CliError::not_found(&msg)
    } else if msg_lower.contains("invalid")
        || msg_lower.contains("unsupported")
        || msg_lower.contains("unknown")
    {
        CliError::invalid_format(&msg)
    } else if msg_lower.contains("permission denied") {
        CliError::permission_denied(&msg)
    } else if msg_lower.contains("syntax") || msg_lower.contains("parse") {
        CliError::validation_error(&msg)
    } else if msg_lower.contains("config") {
        CliError::config_error(&msg)
    } else {
        CliError::new(CliErrorKind::Other(msg))
    };

    enhance_error(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhance_dataset_not_found() {
        let error = CliError::not_found("Dataset 'mydb' not found");
        let enhanced = enhance_error(error);

        assert_eq!(enhanced.suggestions.len(), 3);
        assert!(enhanced
            .suggestions
            .iter()
            .any(|s| s.contains("oxirs init")));
        assert_eq!(enhanced.code.as_deref(), Some("E001"));
    }

    #[test]
    fn test_enhance_invalid_format() {
        let error = CliError::invalid_format("Unsupported RDF format 'xyz'");
        let enhanced = enhance_error(error);

        assert!(!enhanced.suggestions.is_empty());
        assert!(enhanced
            .suggestions
            .iter()
            .any(|s| s.contains("turtle") || s.contains("format")));
    }

    #[test]
    fn test_enhance_validation_error() {
        let error = CliError::validation_error("Invalid dataset name 'my.dataset'");
        let enhanced = enhance_error(error);

        assert!(enhanced.suggestions.len() >= 2);
        assert!(enhanced
            .suggestions
            .iter()
            .any(|s| s.contains("alphanumeric")));
        assert_eq!(enhanced.code.as_deref(), Some("E020"));
    }

    #[test]
    fn test_enhanced_error_from_message() {
        let error = enhanced_error_from_message("File not found: data.ttl");
        assert!(matches!(error.kind, CliErrorKind::NotFound(_)));
        assert!(!error.suggestions.is_empty());
    }
}
