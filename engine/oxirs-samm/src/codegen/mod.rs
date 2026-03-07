//! SAMM Code Generation Modules
//!
//! This module contains additional code generators that complement the existing
//! generators in the `generators` module:
//!
//! - [`json_schema`] – generates JSON Schema (draft-07 / 2020-12) documents
//! - [`openapi`] – generates OpenAPI 3.0.3 specification documents
//!
//! # Quick start
//!
//! ```rust,no_run
//! use oxirs_samm::metamodel::Aspect;
//! use oxirs_samm::codegen::{JsonSchemaGenerator, OpenApiGenerator};
//!
//! let aspect = Aspect::new("urn:samm:org.example:1.0.0#Movement".to_string());
//!
//! // JSON Schema
//! let json_gen = JsonSchemaGenerator::new().with_descriptions();
//! let schema = json_gen.generate(&aspect).unwrap();
//!
//! // OpenAPI 3.0
//! let oa_gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
//! let spec = oa_gen.generate(&aspect).unwrap();
//! ```

pub mod json_schema;
pub mod openapi;

pub use json_schema::{
    JsonSchemaGenerator, JsonSchemaOptions, JsonSchemaValidator, ValidationError,
};
pub use openapi::{HttpMethod, OpenApiGenerator, OpenApiOptions};
