//! Documentation and examples
//!
//! This module provides documentation, examples, and guides for using OxiRS GraphQL.

/// Usage examples for different components
pub mod examples {
    //! Code examples for common use cases

    /// Basic server setup example
    pub mod basic_server {
        //! Example: Setting up a basic GraphQL server
        //!
        //! ```rust,no_run
        //! use oxirs_gql::{GraphQLServer, RdfStore, GraphQLConfig};
        //! use std::sync::Arc;
        //!
        //! #[tokio::main]
        //! async fn main() -> Result<(), Box<dyn std::error::Error>> {
        //!     // Create RDF store
        //!     let store = Arc::new(RdfStore::new()?);
        //!     
        //!     // Configure server
        //!     let config = GraphQLConfig::default();
        //!     
        //!     // Start server
        //!     let server = GraphQLServer::new(store)
        //!         .with_config(config);
        //!     
        //!     server.start("127.0.0.1:8080").await?;
        //!     
        //!     Ok(())
        //! }
        //! ```
    }

    /// Custom scalar example
    pub mod custom_scalars {
        //! Example: Using RDF-specific scalars
        //!
        //! ```rust
        //! use oxirs_gql::rdf_scalars::*;
        //!
        //! // IRI scalar
        //! let iri = iri::IRI::new("https://example.org/resource").unwrap();
        //!
        //! // DateTime scalar  
        //! let dt = datetime::DateTime::now();
        //!
        //! // Literal scalar
        //! let literal = literal::Literal::new("Hello World", Some("en"));
        //! ```
    }

    /// Query optimization example
    pub mod optimization {
        //! Example: Using query optimization
        //!
        //! ```rust,no_run
        //! use oxirs_gql::optimizer::*;
        //! use oxirs_gql::types::Schema;
        //!
        //! async fn setup_optimizer() -> Result<(), Box<dyn std::error::Error>> {
        //!     let schema = Schema::new();
        //!     let config = OptimizerConfig::default();
        //!     
        //!     let optimizer = QueryOptimizer::new(config, schema).await?;
        //!     
        //!     // Use optimizer with queries...
        //!     
        //!     Ok(())
        //! }
        //! ```
    }

    /// Validation example
    pub mod validation {
        //! Example: Setting up query validation
        //!
        //! ```rust
        //! use oxirs_gql::validation::*;
        //! use oxirs_gql::features::security::{SecurityPatterns, Environment};
        //!
        //! // Create validation config for production
        //! let config = SecurityPatterns::recommended_validation_config(Environment::Production);
        //!
        //! // Check for dangerous patterns
        //! let is_safe = !SecurityPatterns::has_dangerous_patterns("query { users { password } }");
        //! ```
    }

    /// Subscription example
    pub mod subscriptions {
        //! Example: Setting up GraphQL subscriptions
        //!
        //! ```rust,no_run
        //! use oxirs_gql::subscriptions::*;
        //! use oxirs_gql::types::Schema;
        //! use oxirs_gql::execution::QueryExecutor;
        //!
        //! async fn setup_subscriptions() -> Result<(), Box<dyn std::error::Error>> {
        //!     let schema = Schema::new();
        //!     let executor = QueryExecutor::new(schema.clone());
        //!     let config = SubscriptionConfig::default();
        //!     
        //!     let manager = SubscriptionManager::new(config, schema, executor);
        //!     
        //!     // Start subscription server
        //!     manager.start_server("127.0.0.1:8081").await?;
        //!     
        //!     Ok(())
        //! }
        //! ```
    }
}

/// Best practices and guides
pub mod guides {
    //! Best practices for using OxiRS GraphQL

    /// Performance optimization guide
    pub mod performance {
        //! # Performance Optimization Guide
        //!
        //! ## Query Optimization
        //!
        //! - Enable query caching for frequently used queries
        //! - Use query complexity analysis to prevent expensive operations
        //! - Implement proper pagination for large result sets
        //!
        //! ## Schema Design
        //!
        //! - Keep query depth reasonable (< 10 levels)
        //! - Use non-null types where appropriate to reduce validation overhead
        //! - Consider using fragments for repeated field selections
        //!
        //! ## Server Configuration
        //!
        //! - Set appropriate timeout values
        //! - Configure connection limits based on your infrastructure
        //! - Enable compression for large responses
    }

    /// Security best practices
    pub mod security {
        //! # Security Best Practices
        //!
        //! ## Query Validation
        //!
        //! - Always enable query validation in production
        //! - Set conservative depth and complexity limits
        //! - Disable introspection in production environments
        //!
        //! ## Rate Limiting
        //!
        //! - Implement rate limiting per client/IP
        //! - Use query complexity for more accurate limiting
        //! - Consider implementing query whitelisting for high-security environments
        //!
        //! ## Data Access
        //!
        //! - Implement proper authentication and authorization
        //! - Use field-level permissions where needed
        //! - Audit query logs for suspicious patterns
    }

    /// RDF integration guide
    pub mod rdf_integration {
        //! # RDF Integration Guide
        //!
        //! ## Working with RDF Data
        //!
        //! - Use appropriate RDF scalars (IRI, Literal, etc.)
        //! - Implement proper namespace handling
        //! - Consider SPARQL query optimization
        //!
        //! ## Schema Generation
        //!
        //! - Generate GraphQL schemas from RDF ontologies
        //! - Map RDF properties to GraphQL fields appropriately
        //! - Handle RDF collections and containers properly
        //!
        //! ## Performance Considerations
        //!
        //! - Index frequently queried RDF properties
        //! - Use SPARQL query optimization techniques
        //! - Consider result set size limits
    }
}

/// Configuration templates for different use cases
pub mod templates {
    use crate::{features::security::Environment, validation::ValidationConfig, GraphQLConfig};

    /// Development environment configuration
    pub fn development_config() -> GraphQLConfig {
        let mut config = GraphQLConfig::default();
        config.enable_playground = true;
        config.enable_introspection = true;
        config.validation_config =
            crate::features::security::SecurityPatterns::recommended_validation_config(
                Environment::Development,
            );
        config
    }

    /// Production environment configuration  
    pub fn production_config() -> GraphQLConfig {
        let mut config = GraphQLConfig::default();
        config.enable_playground = false;
        config.enable_introspection = false;
        config.validation_config =
            crate::features::security::SecurityPatterns::recommended_validation_config(
                Environment::Production,
            );
        config
    }

    /// Testing environment configuration
    pub fn testing_config() -> GraphQLConfig {
        let mut config = GraphQLConfig::default();
        config.enable_playground = true;
        config.enable_introspection = true;
        config.validation_config =
            crate::features::security::SecurityPatterns::recommended_validation_config(
                Environment::Testing,
            );
        config
    }
}
