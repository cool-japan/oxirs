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
//! - **Diagram (SVG/PNG)** - Visual diagrams via Graphviz
//! - **SQL Schema** - Database DDL for PostgreSQL, MySQL, SQLite
//! - **JSON-LD** - Linked Data with semantic context
//! - **JSON Payload** - Sample test data generation
//! - **GraphQL Schema** - GraphQL type definitions and queries
//! - **TypeScript** - TypeScript interfaces and types
//! - **Python** - Python dataclasses with type hints
//! - **Java** - Java POJOs and Records
//! - **Scala** - Scala case classes

pub mod aas;
pub mod diagram;
pub mod graphql;
pub mod java;
pub mod jsonld;
pub mod payload;
pub mod python;
pub mod scala;
pub mod sql;
pub mod typescript;

// Re-export for convenience
pub use aas::{generate_aas, AasFormat};
pub use diagram::{generate_diagram, DiagramFormat};
pub use graphql::generate_graphql;
pub use java::{generate_java, JavaOptions};
pub use jsonld::generate_jsonld;
pub use payload::generate_payload;
pub use python::{generate_python, PythonOptions};
pub use scala::{generate_scala, ScalaOptions};
pub use sql::{generate_sql, SqlDialect};
pub use typescript::{generate_typescript, TsOptions};
