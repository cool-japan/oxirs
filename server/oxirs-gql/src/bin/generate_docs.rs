//! Documentation Generation Tool
//!
//! This binary generates comprehensive documentation for the OxiRS GraphQL module.

use anyhow::Result;
use oxirs_gql::docs::{generate_documentation_with_config, DocConfig, DocFormat, DocTheme};
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("Starting OxiRS GraphQL Documentation Generation");

    // Configure documentation generation
    let config = DocConfig {
        output_dir: "docs/generated".to_string(),
        include_examples: true,
        include_benchmarks: true,
        include_performance_guides: true,
        generate_openapi: true,
        generate_graphql_schema: true,
        generate_federation_docs: true,
        theme: DocTheme::Modern,
        formats: vec![DocFormat::Html, DocFormat::Markdown, DocFormat::Json],
    };

    info!("Documentation configuration:");
    info!("  Output directory: {}", config.output_dir);
    info!("  Formats: {:?}", config.formats);
    info!("  Theme: {:?}", config.theme);

    // Generate documentation
    generate_documentation_with_config(config).await?;

    info!("Documentation generation completed successfully!");
    info!("Check the docs/generated/ directory for the generated documentation.");

    Ok(())
}
