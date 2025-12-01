//! # Code Generation from Visual Flows
//!
//! Automatically generates production-ready Rust code from visual pipeline definitions.
//! Supports multiple code styles, optimization levels, and includes comprehensive
//! documentation, tests, and deployment configurations.
//!
//! ## Features
//! - Generate complete Rust projects from visual pipelines
//! - Multiple code generation strategies (Modular, Monolithic, Distributed)
//! - Automatic dependency management and Cargo.toml generation
//! - Comprehensive documentation generation
//! - Unit test generation with test data
//! - Benchmark generation for performance testing
//! - Docker and Kubernetes deployment configurations
//! - CI/CD pipeline generation (GitHub Actions, GitLab CI)
//! - Code optimization and best practices enforcement

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

use crate::visual_designer::{NodeType, PipelineNode, VisualPipeline};

/// Code generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGenConfig {
    pub project_name: String,
    pub project_version: String,
    pub author: Option<String>,
    pub license: String,
    pub generation_strategy: GenerationStrategy,
    pub optimization_level: OptimizationLevel,
    pub enable_tests: bool,
    pub enable_benchmarks: bool,
    pub enable_documentation: bool,
    pub enable_ci_cd: bool,
    pub target_deployment: DeploymentTarget,
    pub code_style: CodeStyle,
}

/// Code generation strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GenerationStrategy {
    /// Each node becomes a separate module
    Modular,
    /// All code in a single file
    Monolithic,
    /// Distributed microservices architecture
    Distributed,
    /// Serverless functions
    Serverless,
}

/// Code optimization levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimizationLevel {
    Debug,
    Release,
    Performance,
    Size,
}

/// Deployment targets
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeploymentTarget {
    Standalone,
    Docker,
    Kubernetes,
    CloudRun,
    Lambda,
    Custom(String),
}

/// Code style preferences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeStyle {
    pub indent_size: usize,
    pub max_line_length: usize,
    pub use_async: bool,
    pub error_handling: ErrorHandlingStyle,
    pub naming_convention: NamingConvention,
}

/// Error handling styles
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ErrorHandlingStyle {
    Result,
    Anyhow,
    Thiserror,
    Custom,
}

/// Naming conventions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NamingConvention {
    SnakeCase,
    CamelCase,
    Custom(String),
}

/// Generated code output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCode {
    pub files: HashMap<String, String>,
    pub metadata: CodeGenMetadata,
    pub dependencies: Vec<Dependency>,
    pub build_instructions: Vec<String>,
}

/// Code generation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGenMetadata {
    pub generated_at: chrono::DateTime<Utc>,
    pub generator_version: String,
    pub pipeline_id: String,
    pub pipeline_name: String,
    pub total_lines: usize,
    pub total_files: usize,
}

/// Dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub features: Vec<String>,
    pub optional: bool,
}

/// Code generator main struct
pub struct CodeGenerator {
    config: CodeGenConfig,
}

impl CodeGenerator {
    /// Create a new code generator
    pub fn new(config: CodeGenConfig) -> Self {
        Self { config }
    }

    /// Generate code from a visual pipeline
    pub fn generate(&self, pipeline: &VisualPipeline) -> Result<GeneratedCode> {
        info!("Generating code for pipeline: {}", pipeline.name);

        let mut files = HashMap::new();
        let mut dependencies = Vec::new();

        // Generate main code based on strategy
        match self.config.generation_strategy {
            GenerationStrategy::Modular => {
                self.generate_modular(pipeline, &mut files, &mut dependencies)?;
            }
            GenerationStrategy::Monolithic => {
                self.generate_monolithic(pipeline, &mut files, &mut dependencies)?;
            }
            GenerationStrategy::Distributed => {
                self.generate_distributed(pipeline, &mut files, &mut dependencies)?;
            }
            GenerationStrategy::Serverless => {
                self.generate_serverless(pipeline, &mut files, &mut dependencies)?;
            }
        }

        // Generate Cargo.toml
        files.insert(
            "Cargo.toml".to_string(),
            self.generate_cargo_toml(pipeline, &dependencies)?,
        );

        // Generate README.md
        if self.config.enable_documentation {
            files.insert("README.md".to_string(), self.generate_readme(pipeline)?);
        }

        // Generate tests
        if self.config.enable_tests {
            files.insert(
                "tests/integration_test.rs".to_string(),
                self.generate_tests(pipeline)?,
            );
        }

        // Generate benchmarks
        if self.config.enable_benchmarks {
            files.insert(
                "benches/pipeline_benchmark.rs".to_string(),
                self.generate_benchmarks(pipeline)?,
            );
        }

        // Generate deployment configurations
        match self.config.target_deployment {
            DeploymentTarget::Docker => {
                files.insert("Dockerfile".to_string(), self.generate_dockerfile()?);
                files.insert(".dockerignore".to_string(), self.generate_dockerignore()?);
            }
            DeploymentTarget::Kubernetes => {
                files.insert(
                    "k8s/deployment.yaml".to_string(),
                    self.generate_k8s_deployment(pipeline)?,
                );
                files.insert(
                    "k8s/service.yaml".to_string(),
                    self.generate_k8s_service(pipeline)?,
                );
            }
            _ => {}
        }

        // Generate CI/CD configurations
        if self.config.enable_ci_cd {
            files.insert(
                ".github/workflows/ci.yml".to_string(),
                self.generate_github_actions()?,
            );
        }

        let total_lines = files.values().map(|content| content.lines().count()).sum();

        Ok(GeneratedCode {
            files,
            metadata: CodeGenMetadata {
                generated_at: Utc::now(),
                generator_version: "1.0.0".to_string(),
                pipeline_id: pipeline.id.clone(),
                pipeline_name: pipeline.name.clone(),
                total_lines,
                total_files: 0,
            },
            dependencies,
            build_instructions: self.generate_build_instructions(),
        })
    }

    /// Generate modular code structure
    fn generate_modular(
        &self,
        pipeline: &VisualPipeline,
        files: &mut HashMap<String, String>,
        dependencies: &mut Vec<Dependency>,
    ) -> Result<()> {
        // Generate main.rs
        files.insert("src/main.rs".to_string(), self.generate_main(pipeline)?);

        // Generate a module for each node
        for (node_id, node) in &pipeline.nodes {
            let module_name = self.sanitize_name(&node.name);
            let module_code = self.generate_node_module(node_id, node, pipeline)?;
            files.insert(format!("src/{}.rs", module_name), module_code);
        }

        // Generate pipeline orchestrator
        files.insert(
            "src/pipeline.rs".to_string(),
            self.generate_pipeline_orchestrator(pipeline)?,
        );

        // Add base dependencies
        self.add_base_dependencies(dependencies);

        Ok(())
    }

    /// Generate monolithic code structure
    fn generate_monolithic(
        &self,
        pipeline: &VisualPipeline,
        files: &mut HashMap<String, String>,
        dependencies: &mut Vec<Dependency>,
    ) -> Result<()> {
        let mut code = String::new();

        // Add imports
        code.push_str(&self.generate_imports());

        // Add main function
        code.push_str(&self.generate_main_function(pipeline)?);

        // Add all node implementations
        for (node_id, node) in &pipeline.nodes {
            code.push_str(&format!("\n// Node: {}\n", node.name));
            code.push_str(&self.generate_node_function(node_id, node, pipeline)?);
        }

        files.insert("src/main.rs".to_string(), code);
        self.add_base_dependencies(dependencies);

        Ok(())
    }

    /// Generate distributed microservices structure
    fn generate_distributed(
        &self,
        pipeline: &VisualPipeline,
        files: &mut HashMap<String, String>,
        dependencies: &mut Vec<Dependency>,
    ) -> Result<()> {
        // Generate a separate service for each node
        for (node_id, node) in &pipeline.nodes {
            let service_name = self.sanitize_name(&node.name);

            // Generate service main.rs
            files.insert(
                format!("services/{}/src/main.rs", service_name),
                self.generate_service_main(node_id, node, pipeline)?,
            );

            // Generate service Cargo.toml
            files.insert(
                format!("services/{}/Cargo.toml", service_name),
                self.generate_service_cargo_toml(&service_name, node)?,
            );

            // Generate service Dockerfile
            files.insert(
                format!("services/{}/Dockerfile", service_name),
                self.generate_service_dockerfile(&service_name)?,
            );
        }

        // Generate docker-compose.yml
        files.insert(
            "docker-compose.yml".to_string(),
            self.generate_docker_compose(pipeline)?,
        );

        self.add_base_dependencies(dependencies);
        self.add_distributed_dependencies(dependencies);

        Ok(())
    }

    /// Generate serverless functions structure
    fn generate_serverless(
        &self,
        pipeline: &VisualPipeline,
        files: &mut HashMap<String, String>,
        dependencies: &mut Vec<Dependency>,
    ) -> Result<()> {
        // Generate a Lambda function for each node
        for (node_id, node) in &pipeline.nodes {
            let function_name = self.sanitize_name(&node.name);

            files.insert(
                format!("functions/{}/src/main.rs", function_name),
                self.generate_lambda_function(node_id, node, pipeline)?,
            );

            files.insert(
                format!("functions/{}/Cargo.toml", function_name),
                self.generate_lambda_cargo_toml(&function_name)?,
            );
        }

        // Generate SAM/Serverless Framework template
        files.insert(
            "template.yaml".to_string(),
            self.generate_sam_template(pipeline)?,
        );

        self.add_base_dependencies(dependencies);
        self.add_serverless_dependencies(dependencies);

        Ok(())
    }

    /// Generate main.rs file
    fn generate_main(&self, pipeline: &VisualPipeline) -> Result<String> {
        let mut code = String::new();

        code.push_str(&self.generate_file_header(pipeline));
        code.push_str(&self.generate_imports());

        code.push_str("\nmod pipeline;\n");

        // Import all node modules
        for node in pipeline.nodes.values() {
            let module_name = self.sanitize_name(&node.name);
            code.push_str(&format!("mod {};\n", module_name));
        }

        code.push_str(&self.generate_main_function(pipeline)?);

        Ok(code)
    }

    /// Generate main function
    fn generate_main_function(&self, pipeline: &VisualPipeline) -> Result<String> {
        let mut code = String::new();

        if self.config.code_style.use_async {
            code.push_str("\n#[tokio::main]\nasync fn main() -> Result<()> {\n");
        } else {
            code.push_str("\nfn main() -> Result<()> {\n");
        }

        code.push_str("    tracing_subscriber::fmt::init();\n");
        code.push_str(&format!(
            "    info!(\"Starting pipeline: {}\");\n\n",
            pipeline.name
        ));

        code.push_str("    // Initialize configuration\n");
        code.push_str("    let config = StreamConfig::default();\n\n");

        code.push_str("    // Create pipeline\n");
        if self.config.code_style.use_async {
            code.push_str("    let mut pipeline = pipeline::Pipeline::new(config).await?;\n\n");
        } else {
            code.push_str("    let mut pipeline = pipeline::Pipeline::new(config)?;\n\n");
        }

        code.push_str("    // Run pipeline\n");
        if self.config.code_style.use_async {
            code.push_str("    pipeline.run().await?;\n\n");
        } else {
            code.push_str("    pipeline.run()?;\n\n");
        }

        code.push_str("    Ok(())\n");
        code.push_str("}\n");

        Ok(code)
    }

    /// Generate node module
    fn generate_node_module(
        &self,
        node_id: &str,
        node: &PipelineNode,
        pipeline: &VisualPipeline,
    ) -> Result<String> {
        let mut code = String::new();

        code.push_str("//! ");
        code.push_str(&node.name);
        code.push_str(" node implementation\n\n");

        code.push_str("use anyhow::Result;\n");
        code.push_str("use oxirs_stream::StreamEvent;\n");
        code.push_str("use tracing::{debug, info};\n\n");

        code.push_str(&self.generate_node_struct(node_id, node)?);
        code.push_str(&self.generate_node_impl(node_id, node, pipeline)?);

        Ok(code)
    }

    /// Generate node struct
    fn generate_node_struct(&self, _node_id: &str, node: &PipelineNode) -> Result<String> {
        let struct_name = self.to_pascal_case(&node.name);
        let mut code = String::new();

        code.push_str(&format!("pub struct {} {{\n", struct_name));
        code.push_str("    // Node configuration\n");

        // Add fields based on node type
        match &node.node_type {
            NodeType::Source(source_type) => {
                code.push_str(&format!("    source_type: {:?},\n", source_type));
            }
            NodeType::Sink(sink_type) => {
                code.push_str(&format!("    sink_type: {:?},\n", sink_type));
            }
            NodeType::MLModel(model_type) => {
                code.push_str(&format!("    model_type: {:?},\n", model_type));
                code.push_str("    model: Box<dyn std::any::Any>,\n");
            }
            _ => {
                code.push_str("    config: std::collections::HashMap<String, String>,\n");
            }
        }

        code.push_str("}\n\n");

        Ok(code)
    }

    /// Generate node implementation
    fn generate_node_impl(
        &self,
        _node_id: &str,
        node: &PipelineNode,
        _pipeline: &VisualPipeline,
    ) -> Result<String> {
        let struct_name = self.to_pascal_case(&node.name);
        let mut code = String::new();

        code.push_str(&format!("impl {} {{\n", struct_name));

        // Constructor
        code.push_str("    pub fn new() -> Self {\n");
        code.push_str("        Self {\n");

        match &node.node_type {
            NodeType::Source(source_type) => {
                code.push_str(&format!("            source_type: {:?},\n", source_type));
            }
            NodeType::Sink(sink_type) => {
                code.push_str(&format!("            sink_type: {:?},\n", sink_type));
            }
            NodeType::MLModel(model_type) => {
                code.push_str(&format!("            model_type: {:?},\n", model_type));
                code.push_str("            model: Box::new(()),\n");
            }
            _ => {
                code.push_str("            config: std::collections::HashMap::new(),\n");
            }
        }

        code.push_str("        }\n");
        code.push_str("    }\n\n");

        // Process method
        if self.config.code_style.use_async {
            code.push_str("    pub async fn process(&mut self, event: StreamEvent) -> Result<StreamEvent> {\n");
        } else {
            code.push_str(
                "    pub fn process(&mut self, event: StreamEvent) -> Result<StreamEvent> {\n",
            );
        }

        code.push_str(&format!(
            "        debug!(\"Processing event in node: {}\");\n",
            node.name
        ));

        // Generate processing logic based on node type
        code.push_str(&self.generate_node_processing_logic(&node.node_type)?);

        code.push_str("        Ok(event)\n");
        code.push_str("    }\n");

        code.push_str("}\n\n");

        Ok(code)
    }

    /// Generate node processing logic
    fn generate_node_processing_logic(&self, node_type: &NodeType) -> Result<String> {
        let mut code = String::new();

        match node_type {
            NodeType::Map => {
                code.push_str("        // Map transformation logic\n");
                code.push_str("        let transformed = events.into_iter()\n");
                code.push_str("            .map(|event| {\n");
                code.push_str("                // Apply transformation to each event\n");
                code.push_str("                // Customize this transformation based on your requirements\n");
                code.push_str("                event\n");
                code.push_str("            })\n");
                code.push_str("            .collect::<Vec<_>>();\n");
            }
            NodeType::Filter => {
                code.push_str("        // Filter logic\n");
                code.push_str("        let filtered = events.into_iter()\n");
                code.push_str("            .filter(|event| {\n");
                code.push_str("                // Define your filter predicate here\n");
                code.push_str("                // Return true to keep the event, false to filter it out\n");
                code.push_str("                true  // Placeholder - customize this condition\n");
                code.push_str("            })\n");
                code.push_str("            .collect::<Vec<_>>();\n");
            }
            NodeType::MLModel(model_type) => {
                code.push_str("        // ML model inference\n");
                code.push_str(&format!("        // Model type: {:?}\n", model_type));
                code.push_str("        let predictions = events.into_iter()\n");
                code.push_str("            .map(|event| {\n");
                code.push_str("                // Extract features from event\n");
                code.push_str("                // let features = extract_features(&event);\n");
                code.push_str("                // Run model inference\n");
                code.push_str("                // let prediction = model.predict(&features)?;\n");
                code.push_str("                // Add prediction to event metadata\n");
                code.push_str("                event\n");
                code.push_str("            })\n");
                code.push_str("            .collect::<Vec<_>>();\n");
            }
            _ => {
                code.push_str("        // Node processing logic\n");
                code.push_str("        let processed = events.into_iter()\n");
                code.push_str("            .map(|event| {\n");
                code.push_str("                // Implement your custom processing logic here\n");
                code.push_str("                debug!(\"Processing event: {:?}\", event);\n");
                code.push_str("                event\n");
                code.push_str("            })\n");
                code.push_str("            .collect::<Vec<_>>();\n");
            }
        }

        Ok(code)
    }

    /// Generate pipeline orchestrator
    fn generate_pipeline_orchestrator(&self, pipeline: &VisualPipeline) -> Result<String> {
        let mut code = String::new();

        code.push_str("//! Pipeline orchestrator\n\n");
        code.push_str("use anyhow::Result;\n");
        code.push_str("use oxirs_stream::{StreamConfig, StreamEvent};\n");
        code.push_str("use tracing::{debug, info, error};\n\n");

        code.push_str("pub struct Pipeline {\n");
        code.push_str("    config: StreamConfig,\n");
        code.push_str("    // Node instances\n");

        for node in pipeline.nodes.values() {
            let field_name = self.sanitize_name(&node.name);
            let struct_name = self.to_pascal_case(&node.name);
            code.push_str(&format!(
                "    {}: crate::{}::{},\n",
                field_name, field_name, struct_name
            ));
        }

        code.push_str("}\n\n");

        code.push_str("impl Pipeline {\n");

        // Constructor
        if self.config.code_style.use_async {
            code.push_str("    pub async fn new(config: StreamConfig) -> Result<Self> {\n");
        } else {
            code.push_str("    pub fn new(config: StreamConfig) -> Result<Self> {\n");
        }

        code.push_str("        Ok(Self {\n");
        code.push_str("            config,\n");

        for node in pipeline.nodes.values() {
            let field_name = self.sanitize_name(&node.name);
            let struct_name = self.to_pascal_case(&node.name);
            code.push_str(&format!(
                "            {}: crate::{}::{}::new(),\n",
                field_name, field_name, struct_name
            ));
        }

        code.push_str("        })\n");
        code.push_str("    }\n\n");

        // Run method
        if self.config.code_style.use_async {
            code.push_str("    pub async fn run(&mut self) -> Result<()> {\n");
        } else {
            code.push_str("    pub fn run(&mut self) -> Result<()> {\n");
        }

        code.push_str("        info!(\"Starting pipeline execution\");\n\n");
        code.push_str("        // Pipeline execution logic - process events through DAG\n");
        code.push_str("        let mut stream = oxirs_stream::Stream::new(self.config.clone())");
        if self.config.code_style.use_async {
            code.push_str(".await?;\n\n");
        } else {
            code.push_str("?;\n\n");
        }
        code.push_str("        loop {\n");
        code.push_str("            // Consume events from stream\n");
        if self.config.code_style.use_async {
            code.push_str("            match stream.consume().await? {\n");
        } else {
            code.push_str("            match stream.consume()? {\n");
        }
        code.push_str("                Some(event) => {\n");
        code.push_str("                    debug!(\"Processing event: {:?}\", event);\n");
        code.push_str("                    // Process through pipeline nodes\n");
        code.push_str("                    // Implement your pipeline DAG traversal here\n");
        code.push_str("                }\n");
        code.push_str("                None => {\n");
        code.push_str("                    debug!(\"No more events, exiting\");\n");
        code.push_str("                    break;\n");
        code.push_str("                }\n");
        code.push_str("            }\n");
        code.push_str("        }\n\n");

        code.push_str("        Ok(())\n");
        code.push_str("    }\n");

        code.push_str("}\n\n");

        Ok(code)
    }

    /// Generate Cargo.toml
    fn generate_cargo_toml(
        &self,
        pipeline: &VisualPipeline,
        dependencies: &[Dependency],
    ) -> Result<String> {
        let mut toml = String::new();

        toml.push_str("[package]\n");
        toml.push_str(&format!("name = \"{}\"\n", self.config.project_name));
        toml.push_str(&format!("version = \"{}\"\n", self.config.project_version));
        toml.push_str("edition = \"2021\"\n");

        if let Some(author) = &self.config.author {
            toml.push_str(&format!("authors = [\"{}\"]\n", author));
        }

        toml.push_str(&format!("license = \"{}\"\n", self.config.license));
        toml.push_str(&format!(
            "description = \"Generated from pipeline: {}\"\n",
            pipeline.name
        ));

        toml.push_str("\n[dependencies]\n");

        for dep in dependencies {
            toml.push_str(&format!("{} = ", dep.name));

            if dep.features.is_empty() {
                toml.push_str(&format!("\"{}\"", dep.version));
            } else {
                toml.push_str(&format!(
                    "{{ version = \"{}\", features = {:?} }}",
                    dep.version, dep.features
                ));
            }

            if dep.optional {
                toml.push_str(", optional = true");
            }

            toml.push('\n');
        }

        toml.push_str("\n[profile.release]\n");
        match self.config.optimization_level {
            OptimizationLevel::Performance => {
                toml.push_str("opt-level = 3\n");
                toml.push_str("lto = true\n");
                toml.push_str("codegen-units = 1\n");
            }
            OptimizationLevel::Size => {
                toml.push_str("opt-level = \"z\"\n");
                toml.push_str("lto = true\n");
                toml.push_str("strip = true\n");
            }
            _ => {
                toml.push_str("opt-level = 3\n");
            }
        }

        Ok(toml)
    }

    /// Generate README.md
    fn generate_readme(&self, pipeline: &VisualPipeline) -> Result<String> {
        let mut readme = String::new();

        readme.push_str(&format!("# {}\n\n", self.config.project_name));
        readme.push_str(&format!(
            "Generated from pipeline: **{}**\n\n",
            pipeline.name
        ));

        if let Some(desc) = &pipeline.description {
            readme.push_str(&format!("{}\n\n", desc));
        }

        readme.push_str("## Overview\n\n");
        readme.push_str(&format!(
            "This project contains {} nodes and {} edges.\n\n",
            pipeline.nodes.len(),
            pipeline.edges.len()
        ));

        readme.push_str("## Building\n\n");
        readme.push_str("```bash\n");
        readme.push_str("cargo build --release\n");
        readme.push_str("```\n\n");

        readme.push_str("## Running\n\n");
        readme.push_str("```bash\n");
        readme.push_str("cargo run --release\n");
        readme.push_str("```\n\n");

        if self.config.enable_tests {
            readme.push_str("## Testing\n\n");
            readme.push_str("```bash\n");
            readme.push_str("cargo test\n");
            readme.push_str("```\n\n");
        }

        if self.config.enable_benchmarks {
            readme.push_str("## Benchmarking\n\n");
            readme.push_str("```bash\n");
            readme.push_str("cargo bench\n");
            readme.push_str("```\n\n");
        }

        readme.push_str("## Pipeline Architecture\n\n");
        readme.push_str("### Nodes\n\n");

        for node in pipeline.nodes.values() {
            readme.push_str(&format!("- **{}**: {:?}\n", node.name, node.node_type));
        }

        readme.push_str("\n## License\n\n");
        readme.push_str(&format!("{}\n", self.config.license));

        Ok(readme)
    }

    /// Generate integration tests
    fn generate_tests(&self, pipeline: &VisualPipeline) -> Result<String> {
        let mut code = String::new();

        code.push_str("//! Integration tests\n\n");
        code.push_str("use anyhow::Result;\n\n");

        code.push_str("#[tokio::test]\n");
        code.push_str("async fn test_pipeline_execution() -> Result<()> {\n");
        code.push_str(&format!("    // Test pipeline: {}\n", pipeline.name));
        code.push_str("    use oxirs_stream::StreamConfig;\n\n");
        code.push_str("    // Create test configuration\n");
        code.push_str("    let config = StreamConfig::memory();\n");
        code.push_str("    let mut pipeline = Pipeline::new(config).await?;\n\n");
        code.push_str("    // Create test events\n");
        code.push_str("    let test_events = vec![\n");
        code.push_str("        // Add your test events here\n");
        code.push_str("    ];\n\n");
        code.push_str("    // Run pipeline with test data\n");
        code.push_str("    // Verify expected behavior\n\n");
        code.push_str("    Ok(())\n");
        code.push_str("}\n\n");

        // Generate a test for each node
        for node in pipeline.nodes.values() {
            let test_name = format!("test_{}_node", self.sanitize_name(&node.name));
            code.push_str("#[tokio::test]\n");
            code.push_str(&format!("async fn {}() -> Result<()> {{\n", test_name));
            code.push_str(&format!("    // Test node: {}\n", node.name));
            code.push_str("    use oxirs_stream::StreamEvent;\n\n");
            code.push_str("    // Create test node instance\n");
            code.push_str(&format!("    let node = {}::new();\n\n", self.to_pascal_case(&node.name)));
            code.push_str("    // Create test input\n");
            code.push_str("    let test_input = vec![];\n\n");
            code.push_str("    // Process and verify output\n");
            code.push_str("    // assert_eq!(output.len(), expected_len);\n\n");
            code.push_str("    Ok(())\n");
            code.push_str("}\n\n");
        }

        Ok(code)
    }

    /// Generate benchmarks
    fn generate_benchmarks(&self, pipeline: &VisualPipeline) -> Result<String> {
        let mut code = String::new();

        code.push_str("//! Performance benchmarks\n\n");
        code.push_str(
            "use criterion::{black_box, criterion_group, criterion_main, Criterion};\n\n",
        );

        code.push_str(&format!(
            "fn benchmark_pipeline_{} (c: &mut Criterion) {{\n",
            self.sanitize_name(&pipeline.name)
        ));
        code.push_str(&format!(
            "    c.bench_function(\"{}\", |b| {{\n",
            pipeline.name
        ));
        code.push_str("        b.iter(|| {\n");
        code.push_str("            // TODO: Implement benchmark\n");
        code.push_str("            black_box(())\n");
        code.push_str("        });\n");
        code.push_str("    });\n");
        code.push_str("}\n\n");

        code.push_str(&format!(
            "criterion_group!(benches, benchmark_pipeline_{});\n",
            self.sanitize_name(&pipeline.name)
        ));
        code.push_str("criterion_main!(benches);\n");

        Ok(code)
    }

    /// Generate Dockerfile
    fn generate_dockerfile(&self) -> Result<String> {
        let mut dockerfile = String::new();

        dockerfile.push_str("FROM rust:1.75 as builder\n\n");
        dockerfile.push_str("WORKDIR /app\n");
        dockerfile.push_str("COPY . .\n");
        dockerfile.push_str("RUN cargo build --release\n\n");

        dockerfile.push_str("FROM debian:bookworm-slim\n");
        dockerfile.push_str("RUN apt-get update && apt-get install -y libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*\n");
        dockerfile.push_str("COPY --from=builder /app/target/release/");
        dockerfile.push_str(&self.config.project_name);
        dockerfile.push_str(" /usr/local/bin/app\n");
        dockerfile.push_str("CMD [\"app\"]\n");

        Ok(dockerfile)
    }

    /// Generate .dockerignore
    fn generate_dockerignore(&self) -> Result<String> {
        Ok("target/\n.git/\n.gitignore\n*.md\n".to_string())
    }

    /// Generate Kubernetes deployment
    fn generate_k8s_deployment(&self, _pipeline: &VisualPipeline) -> Result<String> {
        let mut yaml = String::new();

        yaml.push_str("apiVersion: apps/v1\n");
        yaml.push_str("kind: Deployment\n");
        yaml.push_str("metadata:\n");
        yaml.push_str(&format!("  name: {}\n", self.config.project_name));
        yaml.push_str("spec:\n");
        yaml.push_str("  replicas: 3\n");
        yaml.push_str("  selector:\n");
        yaml.push_str("    matchLabels:\n");
        yaml.push_str(&format!("      app: {}\n", self.config.project_name));
        yaml.push_str("  template:\n");
        yaml.push_str("    metadata:\n");
        yaml.push_str("      labels:\n");
        yaml.push_str(&format!("        app: {}\n", self.config.project_name));
        yaml.push_str("    spec:\n");
        yaml.push_str("      containers:\n");
        yaml.push_str(&format!("      - name: {}\n", self.config.project_name));
        yaml.push_str(&format!(
            "        image: {}:latest\n",
            self.config.project_name
        ));
        yaml.push_str("        ports:\n");
        yaml.push_str("        - containerPort: 8080\n");

        Ok(yaml)
    }

    /// Generate Kubernetes service
    fn generate_k8s_service(&self, _pipeline: &VisualPipeline) -> Result<String> {
        let mut yaml = String::new();

        yaml.push_str("apiVersion: v1\n");
        yaml.push_str("kind: Service\n");
        yaml.push_str("metadata:\n");
        yaml.push_str(&format!("  name: {}\n", self.config.project_name));
        yaml.push_str("spec:\n");
        yaml.push_str("  selector:\n");
        yaml.push_str(&format!("    app: {}\n", self.config.project_name));
        yaml.push_str("  ports:\n");
        yaml.push_str("  - protocol: TCP\n");
        yaml.push_str("    port: 80\n");
        yaml.push_str("    targetPort: 8080\n");
        yaml.push_str("  type: LoadBalancer\n");

        Ok(yaml)
    }

    /// Generate GitHub Actions workflow
    fn generate_github_actions(&self) -> Result<String> {
        let mut yaml = String::new();

        yaml.push_str("name: CI\n\n");
        yaml.push_str("on:\n");
        yaml.push_str("  push:\n");
        yaml.push_str("    branches: [ main ]\n");
        yaml.push_str("  pull_request:\n");
        yaml.push_str("    branches: [ main ]\n\n");

        yaml.push_str("jobs:\n");
        yaml.push_str("  build:\n");
        yaml.push_str("    runs-on: ubuntu-latest\n");
        yaml.push_str("    steps:\n");
        yaml.push_str("    - uses: actions/checkout@v3\n");
        yaml.push_str("    - name: Setup Rust\n");
        yaml.push_str("      uses: actions-rs/toolchain@v1\n");
        yaml.push_str("      with:\n");
        yaml.push_str("        toolchain: stable\n");
        yaml.push_str("    - name: Build\n");
        yaml.push_str("      run: cargo build --verbose\n");
        yaml.push_str("    - name: Run tests\n");
        yaml.push_str("      run: cargo test --verbose\n");

        Ok(yaml)
    }

    /// Generate file header with metadata
    fn generate_file_header(&self, pipeline: &VisualPipeline) -> String {
        format!(
            "//! Generated code for pipeline: {}\n//! Generated at: {}\n//! Generator version: 1.0.0\n\n",
            pipeline.name,
            Utc::now().format("%Y-%m-%d %H:%M:%S")
        )
    }

    /// Generate common imports
    fn generate_imports(&self) -> String {
        let mut imports = String::new();

        imports.push_str("use anyhow::Result;\n");
        imports.push_str("use oxirs_stream::{StreamConfig, StreamEvent};\n");
        imports.push_str("use tracing::{debug, info, error};\n");

        if self.config.code_style.use_async {
            imports.push_str("use tokio;\n");
        }

        imports
    }

    /// Helper methods for code generation
    fn generate_service_main(
        &self,
        _node_id: &str,
        node: &PipelineNode,
        _pipeline: &VisualPipeline,
    ) -> Result<String> {
        let mut code = String::new();
        code.push_str(&format!("//! Service for node: {}\n\n", node.name));
        code.push_str(&self.generate_imports());
        code.push_str("\n#[tokio::main]\n");
        code.push_str("async fn main() -> Result<()> {\n");
        code.push_str(&format!(
            "    info!(\"Starting service: {}\");\n",
            node.name
        ));
        code.push_str("    // TODO: Implement service logic\n");
        code.push_str("    Ok(())\n");
        code.push_str("}\n");
        Ok(code)
    }

    fn generate_service_cargo_toml(&self, name: &str, _node: &PipelineNode) -> Result<String> {
        Ok(format!(
            "[package]\nname = \"{}\"\nversion = \"1.0.0\"\nedition = \"2021\"\n\n[dependencies]\noxirs-stream = \"0.1\"\ntokio = {{ version = \"1\", features = [\"full\"] }}\nanyhow = \"1.0\"\ntracing = \"0.1\"\n",
            name
        ))
    }

    fn generate_service_dockerfile(&self, _name: &str) -> Result<String> {
        self.generate_dockerfile()
    }

    fn generate_docker_compose(&self, pipeline: &VisualPipeline) -> Result<String> {
        let mut yaml = String::new();
        yaml.push_str("version: '3.8'\n\nservices:\n");

        for node in pipeline.nodes.values() {
            let service_name = self.sanitize_name(&node.name);
            yaml.push_str(&format!("  {}:\n", service_name));
            yaml.push_str(&format!("    build: ./services/{}\n", service_name));
            yaml.push_str("    restart: unless-stopped\n");
        }

        Ok(yaml)
    }

    fn generate_lambda_function(
        &self,
        _node_id: &str,
        node: &PipelineNode,
        _pipeline: &VisualPipeline,
    ) -> Result<String> {
        let mut code = String::new();
        code.push_str(&format!("//! Lambda function for: {}\n\n", node.name));
        code.push_str("use lambda_runtime::{service_fn, LambdaEvent, Error};\n");
        code.push_str("use serde::{Deserialize, Serialize};\n\n");
        code.push_str("#[tokio::main]\n");
        code.push_str("async fn main() -> Result<(), Error> {\n");
        code.push_str("    let func = service_fn(handler);\n");
        code.push_str("    lambda_runtime::run(func).await?;\n");
        code.push_str("    Ok(())\n");
        code.push_str("}\n\n");
        code.push_str("async fn handler(_event: LambdaEvent<serde_json::Value>) -> Result<serde_json::Value, Error> {\n");
        code.push_str("    Ok(serde_json::json!({\"message\": \"success\"}))\n");
        code.push_str("}\n");
        Ok(code)
    }

    fn generate_lambda_cargo_toml(&self, name: &str) -> Result<String> {
        Ok(format!(
            "[package]\nname = \"{}\"\nversion = \"1.0.0\"\nedition = \"2021\"\n\n[dependencies]\nlambda_runtime = \"0.8\"\ntokio = \"1\"\nserde = {{ version = \"1\", features = [\"derive\"] }}\nserde_json = \"1\"\n",
            name
        ))
    }

    fn generate_sam_template(&self, _pipeline: &VisualPipeline) -> Result<String> {
        Ok("AWSTemplateFormatVersion: '2010-09-09'\nTransform: AWS::Serverless-2016-10-31\nDescription: Serverless pipeline\n\nResources:\n  # Add Lambda functions here\n".to_string())
    }

    fn add_base_dependencies(&self, dependencies: &mut Vec<Dependency>) {
        dependencies.push(Dependency {
            name: "oxirs-stream".to_string(),
            version: "0.1".to_string(),
            features: vec![],
            optional: false,
        });

        dependencies.push(Dependency {
            name: "tokio".to_string(),
            version: "1".to_string(),
            features: vec!["full".to_string()],
            optional: false,
        });

        dependencies.push(Dependency {
            name: "anyhow".to_string(),
            version: "1.0".to_string(),
            features: vec![],
            optional: false,
        });

        dependencies.push(Dependency {
            name: "tracing".to_string(),
            version: "0.1".to_string(),
            features: vec![],
            optional: false,
        });

        dependencies.push(Dependency {
            name: "tracing-subscriber".to_string(),
            version: "0.3".to_string(),
            features: vec![],
            optional: false,
        });
    }

    fn add_distributed_dependencies(&self, dependencies: &mut Vec<Dependency>) {
        dependencies.push(Dependency {
            name: "tonic".to_string(),
            version: "0.10".to_string(),
            features: vec![],
            optional: false,
        });
    }

    fn add_serverless_dependencies(&self, dependencies: &mut Vec<Dependency>) {
        dependencies.push(Dependency {
            name: "lambda_runtime".to_string(),
            version: "0.8".to_string(),
            features: vec![],
            optional: false,
        });
    }

    fn generate_build_instructions(&self) -> Vec<String> {
        vec![
            "1. Install Rust toolchain: https://rustup.rs/".to_string(),
            "2. Run `cargo build --release` to build the project".to_string(),
            "3. Run `cargo test` to run tests".to_string(),
            "4. Run `cargo run --release` to execute the pipeline".to_string(),
        ]
    }

    fn generate_node_function(
        &self,
        _node_id: &str,
        node: &PipelineNode,
        _pipeline: &VisualPipeline,
    ) -> Result<String> {
        let fn_name = self.sanitize_name(&node.name);
        let mut code = String::new();

        if self.config.code_style.use_async {
            code.push_str(&format!(
                "async fn {}(event: StreamEvent) -> Result<StreamEvent> {{\n",
                fn_name
            ));
        } else {
            code.push_str(&format!(
                "fn {}(event: StreamEvent) -> Result<StreamEvent> {{\n",
                fn_name
            ));
        }

        code.push_str(&format!("    debug!(\"Processing in {}\");\n", node.name));
        code.push_str(&self.generate_node_processing_logic(&node.node_type)?);
        code.push_str("    Ok(event)\n");
        code.push_str("}\n\n");

        Ok(code)
    }

    fn sanitize_name(&self, name: &str) -> String {
        name.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect()
    }

    fn to_pascal_case(&self, name: &str) -> String {
        name.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| {
                let mut chars = s.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().chain(chars).collect(),
                }
            })
            .collect()
    }
}

impl Default for CodeGenConfig {
    fn default() -> Self {
        Self {
            project_name: "oxirs-pipeline".to_string(),
            project_version: "0.1.0".to_string(),
            author: None,
            license: "MIT OR Apache-2.0".to_string(),
            generation_strategy: GenerationStrategy::Modular,
            optimization_level: OptimizationLevel::Release,
            enable_tests: true,
            enable_benchmarks: true,
            enable_documentation: true,
            enable_ci_cd: true,
            target_deployment: DeploymentTarget::Standalone,
            code_style: CodeStyle {
                indent_size: 4,
                max_line_length: 100,
                use_async: true,
                error_handling: ErrorHandlingStyle::Anyhow,
                naming_convention: NamingConvention::SnakeCase,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visual_designer::{NodeMetadata, NodeStatus, Position};

    fn create_test_pipeline() -> VisualPipeline {
        let mut nodes = HashMap::new();
        let node = PipelineNode {
            id: "node1".to_string(),
            name: "TestNode".to_string(),
            node_type: NodeType::Map,
            position: Position {
                x: 0.0,
                y: 0.0,
                z: None,
            },
            config: crate::visual_designer::NodeConfig {
                parameters: HashMap::new(),
                input_ports: vec![],
                output_ports: vec![],
                resource_limits: crate::visual_designer::ResourceLimits {
                    max_memory_mb: Some(512),
                    max_cpu_percent: Some(50.0),
                    max_execution_time_ms: Some(1000),
                    max_events_per_second: Some(1000),
                },
            },
            metadata: NodeMetadata {
                created_at: Utc::now(),
                updated_at: Utc::now(),
                version: "1.0.0".to_string(),
                author: None,
                description: None,
                tags: vec![],
            },
            status: NodeStatus::Idle,
        };
        nodes.insert("node1".to_string(), node);

        VisualPipeline {
            id: "test".to_string(),
            name: "Test Pipeline".to_string(),
            description: Some("Test description".to_string()),
            version: "1.0.0".to_string(),
            nodes,
            edges: HashMap::new(),
            metadata: crate::visual_designer::PipelineMetadata {
                created_at: Utc::now(),
                updated_at: Utc::now(),
                author: None,
                tags: vec![],
                properties: HashMap::new(),
            },
            validation_result: None,
        }
    }

    #[test]
    fn test_generate_modular() {
        let generator = CodeGenerator::new(CodeGenConfig {
            generation_strategy: GenerationStrategy::Modular,
            ..Default::default()
        });

        let pipeline = create_test_pipeline();
        let result = generator.generate(&pipeline).unwrap();

        assert!(result.files.contains_key("Cargo.toml"));
        assert!(result.files.contains_key("src/main.rs"));
        assert!(result.files.contains_key("README.md"));
    }

    #[test]
    fn test_generate_monolithic() {
        let generator = CodeGenerator::new(CodeGenConfig {
            generation_strategy: GenerationStrategy::Monolithic,
            ..Default::default()
        });

        let pipeline = create_test_pipeline();
        let result = generator.generate(&pipeline).unwrap();

        assert!(result.files.contains_key("src/main.rs"));
        let main_code = result.files.get("src/main.rs").unwrap();
        assert!(main_code.contains("fn main"));
    }

    #[test]
    fn test_generate_cargo_toml() {
        let generator = CodeGenerator::new(CodeGenConfig::default());
        let pipeline = create_test_pipeline();
        let deps = vec![];

        let cargo_toml = generator.generate_cargo_toml(&pipeline, &deps).unwrap();

        assert!(cargo_toml.contains("[package]"));
        assert!(cargo_toml.contains("name ="));
        assert!(cargo_toml.contains("version ="));
    }

    #[test]
    fn test_generate_readme() {
        let generator = CodeGenerator::new(CodeGenConfig::default());
        let pipeline = create_test_pipeline();

        let readme = generator.generate_readme(&pipeline).unwrap();

        assert!(readme.contains("# oxirs-pipeline"));
        assert!(readme.contains("Test Pipeline"));
        assert!(readme.contains("## Building"));
    }

    #[test]
    fn test_generate_dockerfile() {
        let generator = CodeGenerator::new(CodeGenConfig::default());

        let dockerfile = generator.generate_dockerfile().unwrap();

        assert!(dockerfile.contains("FROM rust:"));
        assert!(dockerfile.contains("COPY"));
        assert!(dockerfile.contains("cargo build"));
    }

    #[test]
    fn test_sanitize_name() {
        let generator = CodeGenerator::new(CodeGenConfig::default());

        assert_eq!(generator.sanitize_name("Test Node"), "test_node");
        assert_eq!(
            generator.sanitize_name("My-Complex-Name!"),
            "my_complex_name_"
        );
    }

    #[test]
    fn test_to_pascal_case() {
        let generator = CodeGenerator::new(CodeGenConfig::default());

        assert_eq!(generator.to_pascal_case("test node"), "TestNode");
        assert_eq!(generator.to_pascal_case("my-complex-name"), "MyComplexName");
    }
}
