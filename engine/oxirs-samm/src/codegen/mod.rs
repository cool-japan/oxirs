//! SAMM Code Generation Modules
//!
//! This module contains additional code generators that complement the existing
//! generators in the `generators` module:
//!
//! - [`json_schema`] – generates JSON Schema (draft-07 / 2020-12) documents
//! - [`openapi`] – generates OpenAPI 3.0.3 and 3.1.0 specification documents
//!
//! # Quick start
//!
//! ```rust,no_run
//! use oxirs_samm::metamodel::Aspect;
//! use oxirs_samm::codegen::{JsonSchemaGenerator, OpenApiGenerator, OpenApiOptions, OpenApiVersion};
//!
//! let aspect = Aspect::new("urn:samm:org.example:1.0.0#Movement".to_string());
//!
//! // JSON Schema
//! let json_gen = JsonSchemaGenerator::new().with_descriptions();
//! let schema = json_gen.generate(&aspect).expect("should succeed");
//!
//! // OpenAPI 3.0
//! let oa_gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
//! let spec = oa_gen.generate(&aspect).expect("should succeed");
//!
//! // OpenAPI 3.1 (JSON Schema 2020-12)
//! let options = OpenApiOptions { version: OpenApiVersion::V31, ..OpenApiOptions::default() };
//! let oa31_gen = OpenApiGenerator::with_options(options);
//! let spec31 = oa31_gen.generate(&aspect).expect("should succeed");
//! ```

pub mod json_schema;
pub mod openapi;

pub use json_schema::{
    JsonSchemaGenerator, JsonSchemaOptions, JsonSchemaValidator, ValidationError,
};
pub use openapi::{HttpMethod, OpenApiGenerator, OpenApiOptions, OpenApiVersion};
