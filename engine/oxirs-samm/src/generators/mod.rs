//! Code generators for SAMM models
//!
//! Supports multiple output formats:
//! - Rust structs with serde
//! - Markdown documentation
//! - JSON Schema
//! - OpenAPI 3.1
//! - AsyncAPI 2.6
//! - HTML documentation
//! - **AAS (Asset Administration Shell)** - Industry 4.0 digital twin standard
//! - **DTDL (Digital Twins Definition Language)** - Azure Digital Twins modeling language
//! - **Diagram (SVG/PNG)** - Visual diagrams via Graphviz
//! - **SQL Schema** - Database DDL for PostgreSQL, MySQL, SQLite
//! - **JSON-LD** - Linked Data with semantic context
//! - **JSON Payload** - Sample test data generation
//! - **GraphQL Schema** - GraphQL type definitions and queries
//! - **TypeScript** - TypeScript interfaces and types
//! - **Python** - Python dataclasses with type hints
//! - **Java** - Java POJOs and Records
//! - **Scala** - Scala case classes
//! - **Multi-file Generation** - Organize code across multiple files
//! - **Plugin System** - Register custom code generators

pub mod aas;
pub mod diagram;
pub mod dtdl;
pub mod graphql;
pub mod java;
pub mod jsonld;
pub mod multifile;
pub mod parallel;
pub mod payload;
pub mod plugin;
pub mod python;
pub mod scala;
pub mod sql;
pub mod typescript;

// Re-export for convenience
pub use aas::{generate_aas, AasFormat};
pub use diagram::{generate_diagram, DiagramFormat, DiagramStyle};
pub use dtdl::{generate_dtdl, generate_dtdl_with_options, DtdlOptions};
pub use graphql::generate_graphql;
pub use java::{generate_java, JavaOptions};
pub use jsonld::generate_jsonld;
pub use multifile::{GeneratedFile, MultiFileGenerator, MultiFileOptions, OutputLayout};
pub use parallel::{ParallelGenerator, ParallelGeneratorResult};
pub use payload::generate_payload;
pub use plugin::{CodeGenerator, GeneratorMetadata, GeneratorRef, GeneratorRegistry};
pub use python::{generate_python, PythonOptions};
pub use scala::{generate_scala, ScalaOptions};
pub use sql::{generate_sql, SqlDialect};
pub use typescript::{generate_typescript, TsOptions};

// ── Shared identifier sanitization ──────────────────────────────────────────

/// Ensure `candidate` is a syntactically valid identifier for a generated
/// target language (TypeScript enum member, Java enum constant, Python
/// class attribute, …).
///
/// Any character that is not ASCII alphanumeric or `_` is replaced with
/// `_` (SAMM enumeration/state values are free-form strings and may
/// contain characters — spaces aside from the caller's own separator
/// handling, symbols, etc. — that are not valid in a bare identifier).
/// Additionally, since none of the supported target languages accept a
/// bare identifier that starts with a digit (a common real-world case:
/// numeric status codes such as `"1"`, `"2"`, `"3"`), `prefix` is prepended
/// whenever the sanitized name would start with a digit or be empty.
///
/// `prefix` should already carry the separator/casing appropriate for the
/// target language, e.g. `"Value"` for PascalCase TypeScript enum members
/// or `"VALUE_"` for `SCREAMING_SNAKE_CASE` Java/Python constants.
pub(crate) fn ensure_valid_identifier(candidate: &str, prefix: &str) -> String {
    let sanitized: String = candidate
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if sanitized.is_empty() {
        return format!("{prefix}Empty");
    }
    if sanitized.starts_with(|c: char| c.is_ascii_digit()) {
        format!("{prefix}{sanitized}")
    } else {
        sanitized
    }
}

#[cfg(test)]
mod identifier_sanitization_tests {
    use super::ensure_valid_identifier;

    #[test]
    fn regression_ensure_valid_identifier_prefixes_leading_digit() {
        assert_eq!(ensure_valid_identifier("1", "Value"), "Value1");
        assert_eq!(ensure_valid_identifier("2", "VALUE_"), "VALUE_2");
    }

    #[test]
    fn regression_ensure_valid_identifier_replaces_invalid_characters() {
        assert_eq!(ensure_valid_identifier("A+B", "Value"), "A_B");
        assert_eq!(ensure_valid_identifier("50%", "VALUE_"), "VALUE_50_");
    }

    #[test]
    fn regression_ensure_valid_identifier_leaves_valid_identifier_untouched() {
        assert_eq!(ensure_valid_identifier("Active", "Value"), "Active");
        assert_eq!(ensure_valid_identifier("ACTIVE", "VALUE_"), "ACTIVE");
    }

    #[test]
    fn regression_ensure_valid_identifier_handles_empty_input() {
        assert_eq!(ensure_valid_identifier("", "Value"), "ValueEmpty");
    }
}
