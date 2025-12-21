//! Migration Tools from Other Streaming Platforms
//!
//! This module provides tools to migrate streaming applications from other
//! platforms to oxirs-stream with minimal code changes.
//!
//! # Supported Platforms
//!
//! - Apache Kafka Streams
//! - Apache Flink
//! - Apache Spark Streaming
//! - Apache Storm
//! - Apache Pulsar Functions
//! - AWS Kinesis
//! - Google Dataflow/Beam

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

use crate::error::StreamError;

/// Source platform type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SourcePlatform {
    /// Apache Kafka Streams
    KafkaStreams,
    /// Apache Flink
    Flink,
    /// Apache Spark Streaming
    SparkStreaming,
    /// Apache Storm
    Storm,
    /// Apache Pulsar Functions
    PulsarFunctions,
    /// AWS Kinesis Data Analytics
    KinesisAnalytics,
    /// Google Cloud Dataflow
    Dataflow,
    /// Apache Beam (generic)
    Beam,
    /// Custom platform
    Custom(String),
}

/// Migration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    /// Source platform
    pub source_platform: SourcePlatform,
    /// Source code directory
    pub source_dir: PathBuf,
    /// Output directory
    pub output_dir: PathBuf,
    /// Generate compatibility wrappers
    pub generate_wrappers: bool,
    /// Preserve original comments
    pub preserve_comments: bool,
    /// Generate tests
    pub generate_tests: bool,
    /// Target Rust edition
    pub rust_edition: String,
    /// Additional dependencies
    pub extra_dependencies: Vec<String>,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            source_platform: SourcePlatform::KafkaStreams,
            source_dir: PathBuf::from("./source"),
            output_dir: PathBuf::from("./migrated"),
            generate_wrappers: true,
            preserve_comments: true,
            generate_tests: true,
            rust_edition: "2021".to_string(),
            extra_dependencies: Vec::new(),
        }
    }
}

/// Migration report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationReport {
    /// Report ID
    pub report_id: String,
    /// Source platform
    pub source_platform: SourcePlatform,
    /// Migration timestamp
    pub timestamp: SystemTime,
    /// Files processed
    pub files_processed: usize,
    /// Lines of code converted
    pub lines_converted: usize,
    /// Successful conversions
    pub successful: usize,
    /// Failed conversions
    pub failed: usize,
    /// Warnings generated
    pub warnings: Vec<MigrationWarning>,
    /// Errors encountered
    pub errors: Vec<MigrationError>,
    /// Generated files
    pub generated_files: Vec<GeneratedFile>,
    /// Manual review required
    pub manual_review_items: Vec<ManualReviewItem>,
    /// Migration suggestions
    pub suggestions: Vec<MigrationSuggestion>,
    /// Compatibility score (0-100)
    pub compatibility_score: f64,
}

/// Migration warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
    /// File location
    pub file: Option<PathBuf>,
    /// Line number
    pub line: Option<usize>,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Migration error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// File location
    pub file: Option<PathBuf>,
    /// Line number
    pub line: Option<usize>,
    /// Is recoverable
    pub recoverable: bool,
}

/// Generated file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedFile {
    /// File path
    pub path: PathBuf,
    /// File type
    pub file_type: GeneratedFileType,
    /// Lines of code
    pub lines: usize,
    /// Original source file
    pub source_file: Option<PathBuf>,
}

/// Generated file type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GeneratedFileType {
    /// Main source code
    Source,
    /// Compatibility wrapper
    Wrapper,
    /// Test file
    Test,
    /// Configuration
    Config,
    /// Documentation
    Documentation,
}

/// Item requiring manual review
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualReviewItem {
    /// Item ID
    pub id: String,
    /// Description
    pub description: String,
    /// File location
    pub file: PathBuf,
    /// Line range
    pub line_range: (usize, usize),
    /// Priority
    pub priority: ReviewPriority,
    /// Reason for manual review
    pub reason: String,
    /// Suggested approach
    pub suggestion: String,
}

/// Review priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Ord, PartialOrd, Eq)]
pub enum ReviewPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical - must be reviewed
    Critical,
}

/// Migration suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationSuggestion {
    /// Suggestion category
    pub category: SuggestionCategory,
    /// Suggestion title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Code example
    pub example: Option<String>,
    /// References
    pub references: Vec<String>,
}

/// Suggestion category
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SuggestionCategory {
    /// Performance improvement
    Performance,
    /// Code style
    CodeStyle,
    /// Best practice
    BestPractice,
    /// Security
    Security,
    /// Idiomatic Rust
    RustIdiom,
}

/// Concept mapping between platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptMapping {
    /// Source concept name
    pub source_name: String,
    /// Target oxirs-stream equivalent
    pub target_name: String,
    /// Description of the mapping
    pub description: String,
    /// Code transformation pattern
    pub pattern: Option<String>,
    /// Example source code
    pub source_example: Option<String>,
    /// Example target code
    pub target_example: Option<String>,
}

/// API mapping for code transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIMapping {
    /// Source API call
    pub source_api: String,
    /// Target API call
    pub target_api: String,
    /// Parameter mappings
    pub param_mappings: HashMap<String, String>,
    /// Return type mapping
    pub return_type_mapping: Option<String>,
    /// Notes
    pub notes: String,
}

/// Migration tool for converting streaming applications
pub struct MigrationTool {
    /// Configuration
    config: MigrationConfig,
    /// Concept mappings
    concept_mappings: Arc<RwLock<Vec<ConceptMapping>>>,
    /// API mappings
    api_mappings: Arc<RwLock<Vec<APIMapping>>>,
    /// Migration statistics
    stats: Arc<RwLock<MigrationStats>>,
}

/// Migration statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MigrationStats {
    /// Total migrations performed
    pub total_migrations: u64,
    /// Successful migrations
    pub successful_migrations: u64,
    /// Average compatibility score
    pub avg_compatibility_score: f64,
    /// Total lines converted
    pub total_lines_converted: u64,
    /// Total files processed
    pub total_files_processed: u64,
}

impl MigrationTool {
    /// Create a new migration tool
    /// Note: Call load_default_mappings().await after creation to initialize default mappings
    pub fn new(config: MigrationConfig) -> Self {
        Self {
            config: config.clone(),
            concept_mappings: Arc::new(RwLock::new(Vec::new())),
            api_mappings: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(MigrationStats::default())),
        }
    }

    /// Perform migration analysis without generating code
    pub async fn analyze(&self) -> Result<MigrationReport, StreamError> {
        let mut report = MigrationReport {
            report_id: uuid::Uuid::new_v4().to_string(),
            source_platform: self.config.source_platform.clone(),
            timestamp: SystemTime::now(),
            files_processed: 0,
            lines_converted: 0,
            successful: 0,
            failed: 0,
            warnings: Vec::new(),
            errors: Vec::new(),
            generated_files: Vec::new(),
            manual_review_items: Vec::new(),
            suggestions: Vec::new(),
            compatibility_score: 0.0,
        };

        // Scan source directory
        let source_files = self.scan_source_directory().await?;
        report.files_processed = source_files.len();

        // Analyze each file
        for file_path in &source_files {
            match self.analyze_file(file_path).await {
                Ok(analysis) => {
                    report.successful += 1;
                    report.lines_converted += analysis.lines;
                    report.warnings.extend(analysis.warnings);
                    report.manual_review_items.extend(analysis.review_items);
                }
                Err(e) => {
                    report.failed += 1;
                    report.errors.push(MigrationError {
                        code: "ANALYSIS_ERROR".to_string(),
                        message: e.to_string(),
                        file: Some(file_path.to_path_buf()),
                        line: None,
                        recoverable: false,
                    });
                }
            }
        }

        // Generate suggestions
        report.suggestions = self.generate_suggestions(&report).await;

        // Calculate compatibility score
        report.compatibility_score = self.calculate_compatibility_score(&report);

        Ok(report)
    }

    /// Perform full migration
    pub async fn migrate(&self) -> Result<MigrationReport, StreamError> {
        let mut report = self.analyze().await?;

        // Create output directory
        if !self.config.output_dir.exists() {
            std::fs::create_dir_all(&self.config.output_dir).map_err(|e| {
                StreamError::Io(format!("Failed to create output directory: {}", e))
            })?;
        }

        // Generate Cargo.toml
        let cargo_toml = self.generate_cargo_toml().await;
        let cargo_path = self.config.output_dir.join("Cargo.toml");
        std::fs::write(&cargo_path, cargo_toml)
            .map_err(|e| StreamError::Io(format!("Failed to write Cargo.toml: {}", e)))?;

        report.generated_files.push(GeneratedFile {
            path: cargo_path,
            file_type: GeneratedFileType::Config,
            lines: 30,
            source_file: None,
        });

        // Generate main library file
        let lib_rs = self.generate_lib_rs().await;
        let lib_path = self.config.output_dir.join("src").join("lib.rs");
        std::fs::create_dir_all(self.config.output_dir.join("src")).ok();
        std::fs::write(&lib_path, lib_rs)
            .map_err(|e| StreamError::Io(format!("Failed to write lib.rs: {}", e)))?;

        report.generated_files.push(GeneratedFile {
            path: lib_path,
            file_type: GeneratedFileType::Source,
            lines: 50,
            source_file: None,
        });

        // Generate compatibility wrappers if requested
        if self.config.generate_wrappers {
            let wrapper = self.generate_compatibility_wrapper().await;
            let wrapper_path = self.config.output_dir.join("src").join("compat.rs");
            std::fs::write(&wrapper_path, wrapper)
                .map_err(|e| StreamError::Io(format!("Failed to write compat.rs: {}", e)))?;

            report.generated_files.push(GeneratedFile {
                path: wrapper_path,
                file_type: GeneratedFileType::Wrapper,
                lines: 200,
                source_file: None,
            });
        }

        // Generate tests if requested
        if self.config.generate_tests {
            let tests = self.generate_tests().await;
            let test_path = self.config.output_dir.join("tests").join("integration.rs");
            std::fs::create_dir_all(self.config.output_dir.join("tests")).ok();
            std::fs::write(&test_path, tests)
                .map_err(|e| StreamError::Io(format!("Failed to write tests: {}", e)))?;

            report.generated_files.push(GeneratedFile {
                path: test_path,
                file_type: GeneratedFileType::Test,
                lines: 100,
                source_file: None,
            });
        }

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_migrations += 1;
        stats.successful_migrations += 1;
        stats.total_files_processed += report.files_processed as u64;
        stats.total_lines_converted += report.lines_converted as u64;
        stats.avg_compatibility_score = (stats.avg_compatibility_score
            * (stats.total_migrations - 1) as f64
            + report.compatibility_score)
            / stats.total_migrations as f64;

        Ok(report)
    }

    /// Get concept mappings for a platform
    pub async fn get_concept_mappings(&self) -> Vec<ConceptMapping> {
        self.concept_mappings.read().await.clone()
    }

    /// Get API mappings
    pub async fn get_api_mappings(&self) -> Vec<APIMapping> {
        self.api_mappings.read().await.clone()
    }

    /// Add custom concept mapping
    pub async fn add_concept_mapping(&self, mapping: ConceptMapping) {
        let mut mappings = self.concept_mappings.write().await;
        mappings.push(mapping);
    }

    /// Add custom API mapping
    pub async fn add_api_mapping(&self, mapping: APIMapping) {
        let mut mappings = self.api_mappings.write().await;
        mappings.push(mapping);
    }

    /// Get migration statistics
    pub async fn get_stats(&self) -> MigrationStats {
        self.stats.read().await.clone()
    }

    /// Generate migration guide for the source platform
    pub async fn generate_guide(&self) -> String {
        let mut guide = String::new();

        guide.push_str(&format!(
            "# Migration Guide: {} to oxirs-stream\n\n",
            self.platform_name()
        ));

        guide.push_str("## Overview\n\n");
        guide.push_str(
            "This guide helps you migrate your streaming application to oxirs-stream.\n\n",
        );

        guide.push_str("## Key Concepts\n\n");

        let mappings = self.concept_mappings.read().await;
        for mapping in mappings.iter() {
            guide.push_str(&format!(
                "### {} → {}\n\n{}\n\n",
                mapping.source_name, mapping.target_name, mapping.description
            ));

            if let Some(ref source) = mapping.source_example {
                guide.push_str("**Before:**\n```\n");
                guide.push_str(source);
                guide.push_str("\n```\n\n");
            }

            if let Some(ref target) = mapping.target_example {
                guide.push_str("**After:**\n```rust\n");
                guide.push_str(target);
                guide.push_str("\n```\n\n");
            }
        }

        guide.push_str("## API Reference\n\n");
        guide.push_str("| Source API | oxirs-stream API | Notes |\n");
        guide.push_str("|------------|------------------|-------|\n");

        let api_mappings = self.api_mappings.read().await;
        for mapping in api_mappings.iter() {
            guide.push_str(&format!(
                "| `{}` | `{}` | {} |\n",
                mapping.source_api, mapping.target_api, mapping.notes
            ));
        }

        guide.push_str("\n## Next Steps\n\n");
        guide.push_str("1. Review the generated code\n");
        guide.push_str("2. Address manual review items\n");
        guide.push_str("3. Run the test suite\n");
        guide.push_str("4. Benchmark performance\n");
        guide.push_str("5. Deploy gradually with feature flags\n");

        guide
    }

    // Private helper methods

    async fn load_default_mappings(&mut self) {
        let mut concept_mappings = self.concept_mappings.write().await;
        let mut api_mappings = self.api_mappings.write().await;

        match self.config.source_platform {
            SourcePlatform::KafkaStreams => {
                // Kafka Streams concept mappings
                concept_mappings.push(ConceptMapping {
                    source_name: "KStream".to_string(),
                    target_name: "Stream".to_string(),
                    description: "Unbounded stream of records".to_string(),
                    pattern: Some("stream!".to_string()),
                    source_example: Some("KStream<String, String> stream = builder.stream(\"topic\");".to_string()),
                    target_example: Some("let stream = StreamBuilder::new()\n    .source(KafkaSource::new(\"topic\"))\n    .build();".to_string()),
                });

                concept_mappings.push(ConceptMapping {
                    source_name: "KTable".to_string(),
                    target_name: "StateStore".to_string(),
                    description: "Changelog stream / table".to_string(),
                    pattern: None,
                    source_example: Some(
                        "KTable<String, Long> table = builder.table(\"topic\");".to_string(),
                    ),
                    target_example: Some(
                        "let state = StateStore::new(\"table\")\n    .with_changelog(\"topic\");"
                            .to_string(),
                    ),
                });

                // API mappings
                api_mappings.push(APIMapping {
                    source_api: "stream.mapValues()".to_string(),
                    target_api: "stream.map()".to_string(),
                    param_mappings: HashMap::new(),
                    return_type_mapping: None,
                    notes: "Use map with tuple destructuring".to_string(),
                });

                api_mappings.push(APIMapping {
                    source_api: "stream.filter()".to_string(),
                    target_api: "stream.filter()".to_string(),
                    param_mappings: HashMap::new(),
                    return_type_mapping: None,
                    notes: "Direct equivalent".to_string(),
                });

                api_mappings.push(APIMapping {
                    source_api: "stream.groupByKey()".to_string(),
                    target_api: "stream.group_by_key()".to_string(),
                    param_mappings: HashMap::new(),
                    return_type_mapping: None,
                    notes: "Similar semantics".to_string(),
                });

                api_mappings.push(APIMapping {
                    source_api: "stream.windowedBy()".to_string(),
                    target_api: "stream.window()".to_string(),
                    param_mappings: {
                        let mut map = HashMap::new();
                        map.insert(
                            "TimeWindows.of()".to_string(),
                            "TumblingWindow::new()".to_string(),
                        );
                        map.insert(
                            "SlidingWindows.of()".to_string(),
                            "SlidingWindow::new()".to_string(),
                        );
                        map
                    },
                    return_type_mapping: None,
                    notes: "Window types map directly".to_string(),
                });
            }

            SourcePlatform::Flink => {
                // Flink concept mappings
                concept_mappings.push(ConceptMapping {
                    source_name: "DataStream".to_string(),
                    target_name: "Stream".to_string(),
                    description: "Core streaming abstraction".to_string(),
                    pattern: None,
                    source_example: Some(
                        "DataStream<String> stream = env.addSource(source);".to_string(),
                    ),
                    target_example: Some(
                        "let stream = StreamBuilder::new().source(source).build();".to_string(),
                    ),
                });

                concept_mappings.push(ConceptMapping {
                    source_name: "KeyedStream".to_string(),
                    target_name: "GroupedStream".to_string(),
                    description: "Partitioned stream by key".to_string(),
                    pattern: None,
                    source_example: None,
                    target_example: None,
                });

                api_mappings.push(APIMapping {
                    source_api: "stream.keyBy()".to_string(),
                    target_api: "stream.key_by()".to_string(),
                    param_mappings: HashMap::new(),
                    return_type_mapping: None,
                    notes: "Use closure for key extraction".to_string(),
                });

                api_mappings.push(APIMapping {
                    source_api: "stream.process()".to_string(),
                    target_api: "stream.process()".to_string(),
                    param_mappings: HashMap::new(),
                    return_type_mapping: None,
                    notes: "Implement ProcessFunction trait".to_string(),
                });
            }

            SourcePlatform::SparkStreaming => {
                // Spark Streaming concept mappings
                concept_mappings.push(ConceptMapping {
                    source_name: "DStream".to_string(),
                    target_name: "Stream".to_string(),
                    description: "Discretized stream".to_string(),
                    pattern: None,
                    source_example: Some("val stream = ssc.socketTextStream(host, port)".to_string()),
                    target_example: Some("let stream = StreamBuilder::new()\n    .source(TcpSource::new(host, port))\n    .build();".to_string()),
                });

                api_mappings.push(APIMapping {
                    source_api: "stream.transform()".to_string(),
                    target_api: "stream.map()".to_string(),
                    param_mappings: HashMap::new(),
                    return_type_mapping: None,
                    notes: "Use map for transformations".to_string(),
                });

                api_mappings.push(APIMapping {
                    source_api: "stream.foreachRDD()".to_string(),
                    target_api: "stream.for_each()".to_string(),
                    param_mappings: HashMap::new(),
                    return_type_mapping: None,
                    notes: "Processes each micro-batch".to_string(),
                });
            }

            _ => {
                // Generic mappings for other platforms
                concept_mappings.push(ConceptMapping {
                    source_name: "Stream".to_string(),
                    target_name: "Stream".to_string(),
                    description: "Core streaming abstraction".to_string(),
                    pattern: None,
                    source_example: None,
                    target_example: None,
                });
            }
        }
    }

    async fn scan_source_directory(&self) -> Result<Vec<PathBuf>, StreamError> {
        let mut files = Vec::new();

        if !self.config.source_dir.exists() {
            return Ok(files);
        }

        let extension = match self.config.source_platform {
            SourcePlatform::KafkaStreams | SourcePlatform::Flink | SourcePlatform::Storm => "java",
            SourcePlatform::SparkStreaming => "scala",
            SourcePlatform::PulsarFunctions => "java",
            SourcePlatform::KinesisAnalytics | SourcePlatform::Dataflow | SourcePlatform::Beam => {
                "java"
            }
            SourcePlatform::Custom(_) => "java",
        };

        Self::scan_directory_recursive(&self.config.source_dir, extension, &mut files)?;

        Ok(files)
    }

    fn scan_directory_recursive(
        dir: &Path,
        extension: &str,
        files: &mut Vec<PathBuf>,
    ) -> Result<(), StreamError> {
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)
                .map_err(|e| StreamError::Io(format!("Failed to read directory: {}", e)))?
            {
                let entry =
                    entry.map_err(|e| StreamError::Io(format!("Failed to read entry: {}", e)))?;
                let path = entry.path();

                if path.is_dir() {
                    Self::scan_directory_recursive(&path, extension, files)?;
                } else if path.extension().map(|e| e == extension).unwrap_or(false) {
                    files.push(path);
                }
            }
        }

        Ok(())
    }

    async fn analyze_file(&self, file_path: &Path) -> Result<FileAnalysis, StreamError> {
        // In a real implementation, we'd parse the source file
        // For now, return a mock analysis
        Ok(FileAnalysis {
            lines: 100,
            warnings: vec![MigrationWarning {
                code: "DEPRECATED_API".to_string(),
                message: "Some APIs may need manual review".to_string(),
                file: Some(file_path.to_path_buf()),
                line: None,
                suggestion: Some("Check API mappings".to_string()),
            }],
            review_items: vec![],
        })
    }

    async fn generate_suggestions(&self, report: &MigrationReport) -> Vec<MigrationSuggestion> {
        let mut suggestions = Vec::new();

        // Performance suggestions
        suggestions.push(MigrationSuggestion {
            category: SuggestionCategory::Performance,
            title: "Use async/await for I/O operations".to_string(),
            description: "oxirs-stream is built on Tokio async runtime. Ensure all I/O operations use async methods.".to_string(),
            example: Some("async fn process(event: Event) -> Result<Output, Error> {\n    // Use .await for async operations\n}".to_string()),
            references: vec!["https://tokio.rs/".to_string()],
        });

        // Best practice suggestions
        suggestions.push(MigrationSuggestion {
            category: SuggestionCategory::BestPractice,
            title: "Use structured error handling".to_string(),
            description: "Replace exceptions with Result types for better error propagation.".to_string(),
            example: Some("fn process() -> Result<(), StreamError> {\n    // Return errors instead of throwing\n}".to_string()),
            references: vec![],
        });

        // Rust idiom suggestions
        if report.files_processed > 0 {
            suggestions.push(MigrationSuggestion {
                category: SuggestionCategory::RustIdiom,
                title: "Use iterators instead of loops".to_string(),
                description:
                    "Rust iterators are often more performant and idiomatic than explicit loops."
                        .to_string(),
                example: Some("let sum: i32 = values.iter().map(|x| x * 2).sum();".to_string()),
                references: vec![],
            });
        }

        suggestions
    }

    fn calculate_compatibility_score(&self, report: &MigrationReport) -> f64 {
        if report.files_processed == 0 {
            return 100.0;
        }

        let success_rate = report.successful as f64 / report.files_processed as f64;
        let warning_penalty = (report.warnings.len() as f64 * 2.0).min(20.0);
        let error_penalty = (report.errors.len() as f64 * 5.0).min(50.0);
        let review_penalty = (report.manual_review_items.len() as f64).min(10.0);

        (success_rate * 100.0 - warning_penalty - error_penalty - review_penalty).max(0.0)
    }

    async fn generate_cargo_toml(&self) -> String {
        format!(
            r#"[package]
name = "migrated-stream"
version = "0.1.0"
edition = "{}"

[dependencies]
oxirs-stream = "0.1"
tokio = {{ version = "1", features = ["full"] }}
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
{}
"#,
            self.config.rust_edition,
            self.config.extra_dependencies.join("\n")
        )
    }

    async fn generate_lib_rs(&self) -> String {
        format!(
            r#"//! Migrated streaming application from {}
//! Generated by oxirs-stream migration tool

{}
pub mod compat;

pub use oxirs_stream::prelude::*;

// Your migrated stream processors go here
"#,
            self.platform_name(),
            if self.config.generate_wrappers {
                ""
            } else {
                "// Compatibility wrappers disabled\n"
            }
        )
    }

    async fn generate_compatibility_wrapper(&self) -> String {
        match self.config.source_platform {
            SourcePlatform::KafkaStreams => r#"//! Compatibility wrappers for Kafka Streams API

use oxirs_stream::prelude::*;

/// KStream-like wrapper for familiar API
pub struct KStreamCompat<K, V> {
    inner: Stream<(K, V)>,
}

impl<K, V> KStreamCompat<K, V>
where
    K: Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    pub fn new(stream: Stream<(K, V)>) -> Self {
        Self { inner: stream }
    }

    pub fn map_values<F, V2>(self, f: F) -> KStreamCompat<K, V2>
    where
        F: Fn(V) -> V2 + Send + Sync + Clone + 'static,
        V2: Clone + Send + Sync + 'static,
    {
        // Implementation would go here
        todo!()
    }

    pub fn filter<F>(self, predicate: F) -> Self
    where
        F: Fn(&K, &V) -> bool + Send + Sync + Clone + 'static,
    {
        // Implementation would go here
        todo!()
    }
}

/// KTable-like wrapper
pub struct KTableCompat<K, V> {
    store: StateStore<K, V>,
}
"#
            .to_string(),
            SourcePlatform::Flink => r#"//! Compatibility wrappers for Flink API

use oxirs_stream::prelude::*;

/// DataStream-like wrapper
pub struct DataStreamCompat<T> {
    inner: Stream<T>,
}

impl<T> DataStreamCompat<T>
where
    T: Clone + Send + Sync + 'static,
{
    pub fn new(stream: Stream<T>) -> Self {
        Self { inner: stream }
    }

    pub fn key_by<K, F>(self, key_selector: F) -> KeyedStreamCompat<K, T>
    where
        K: Clone + Send + Sync + 'static,
        F: Fn(&T) -> K + Send + Sync + Clone + 'static,
    {
        // Implementation would go here
        todo!()
    }
}

/// KeyedStream-like wrapper
pub struct KeyedStreamCompat<K, T> {
    inner: GroupedStream<K, T>,
}
"#
            .to_string(),
            _ => r#"//! Generic compatibility wrappers

use oxirs_stream::prelude::*;

// Add platform-specific wrappers as needed
"#
            .to_string(),
        }
    }

    async fn generate_tests(&self) -> String {
        r#"//! Integration tests for migrated application

use oxirs_stream::prelude::*;

#[tokio::test]
async fn test_basic_stream() {
    // Add your tests here
    assert!(true);
}

#[tokio::test]
async fn test_window_operations() {
    // Test window operations
    assert!(true);
}

#[tokio::test]
async fn test_aggregations() {
    // Test aggregations
    assert!(true);
}
"#
        .to_string()
    }

    fn platform_name(&self) -> String {
        match &self.config.source_platform {
            SourcePlatform::KafkaStreams => "Kafka Streams".to_string(),
            SourcePlatform::Flink => "Apache Flink".to_string(),
            SourcePlatform::SparkStreaming => "Spark Streaming".to_string(),
            SourcePlatform::Storm => "Apache Storm".to_string(),
            SourcePlatform::PulsarFunctions => "Pulsar Functions".to_string(),
            SourcePlatform::KinesisAnalytics => "Kinesis Analytics".to_string(),
            SourcePlatform::Dataflow => "Google Dataflow".to_string(),
            SourcePlatform::Beam => "Apache Beam".to_string(),
            SourcePlatform::Custom(name) => name.clone(),
        }
    }
}

/// Helper struct for file analysis
struct FileAnalysis {
    lines: usize,
    warnings: Vec<MigrationWarning>,
    review_items: Vec<ManualReviewItem>,
}

/// Quick start helper for common migrations
pub struct QuickStart;

impl QuickStart {
    /// Create a Kafka Streams migration tool
    pub fn from_kafka_streams(source_dir: &str, output_dir: &str) -> MigrationTool {
        MigrationTool::new(MigrationConfig {
            source_platform: SourcePlatform::KafkaStreams,
            source_dir: PathBuf::from(source_dir),
            output_dir: PathBuf::from(output_dir),
            ..Default::default()
        })
    }

    /// Create a Flink migration tool
    pub fn from_flink(source_dir: &str, output_dir: &str) -> MigrationTool {
        MigrationTool::new(MigrationConfig {
            source_platform: SourcePlatform::Flink,
            source_dir: PathBuf::from(source_dir),
            output_dir: PathBuf::from(output_dir),
            ..Default::default()
        })
    }

    /// Create a Spark Streaming migration tool
    pub fn from_spark(source_dir: &str, output_dir: &str) -> MigrationTool {
        MigrationTool::new(MigrationConfig {
            source_platform: SourcePlatform::SparkStreaming,
            source_dir: PathBuf::from(source_dir),
            output_dir: PathBuf::from(output_dir),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_migration_tool_creation() {
        let config = MigrationConfig::default();
        let mut tool = MigrationTool::new(config);
        tool.load_default_mappings().await;

        let mappings = tool.get_concept_mappings().await;
        assert!(!mappings.is_empty());
    }

    #[tokio::test]
    async fn test_kafka_streams_mappings() {
        let mut tool = QuickStart::from_kafka_streams("/tmp/source", "/tmp/output");
        tool.load_default_mappings().await;

        let concept_mappings = tool.get_concept_mappings().await;
        let has_kstream = concept_mappings.iter().any(|m| m.source_name == "KStream");
        assert!(has_kstream);
    }

    #[tokio::test]
    async fn test_flink_mappings() {
        let mut tool = QuickStart::from_flink("/tmp/source", "/tmp/output");
        tool.load_default_mappings().await; // Load mappings

        let concept_mappings = tool.get_concept_mappings().await;
        let has_datastream = concept_mappings
            .iter()
            .any(|m| m.source_name == "DataStream");
        assert!(has_datastream);
    }

    #[tokio::test]
    async fn test_custom_mapping() {
        let config = MigrationConfig::default();
        let tool = MigrationTool::new(config);

        tool.add_concept_mapping(ConceptMapping {
            source_name: "CustomConcept".to_string(),
            target_name: "OxirsConcept".to_string(),
            description: "Custom mapping".to_string(),
            pattern: None,
            source_example: None,
            target_example: None,
        })
        .await;

        let mappings = tool.get_concept_mappings().await;
        let has_custom = mappings.iter().any(|m| m.source_name == "CustomConcept");
        assert!(has_custom);
    }

    #[tokio::test]
    async fn test_generate_guide() {
        let tool = QuickStart::from_kafka_streams("/tmp/source", "/tmp/output");

        let guide = tool.generate_guide().await;
        assert!(guide.contains("Migration Guide"));
        assert!(guide.contains("Kafka Streams"));
    }

    #[tokio::test]
    async fn test_analyze_empty_directory() {
        let config = MigrationConfig {
            source_dir: PathBuf::from("/tmp/nonexistent"),
            output_dir: PathBuf::from("/tmp/output"),
            ..Default::default()
        };

        let tool = MigrationTool::new(config);
        let report = tool.analyze().await.unwrap();

        assert_eq!(report.files_processed, 0);
        assert_eq!(report.compatibility_score, 100.0);
    }

    #[tokio::test]
    async fn test_api_mappings() {
        let mut tool = QuickStart::from_kafka_streams("/tmp/source", "/tmp/output");
        tool.load_default_mappings().await; // Load mappings

        let api_mappings = tool.get_api_mappings().await;
        let has_filter = api_mappings.iter().any(|m| m.source_api.contains("filter"));
        assert!(has_filter);
    }

    #[tokio::test]
    async fn test_compatibility_score() {
        let config = MigrationConfig::default();
        let tool = MigrationTool::new(config);

        // Empty directory should have 100% compatibility
        let report = tool.analyze().await.unwrap();
        assert!(report.compatibility_score >= 0.0 && report.compatibility_score <= 100.0);
    }

    #[tokio::test]
    async fn test_spark_mappings() {
        let mut tool = QuickStart::from_spark("/tmp/source", "/tmp/output");
        tool.load_default_mappings().await;

        let mappings = tool.get_concept_mappings().await;
        let has_dstream = mappings.iter().any(|m| m.source_name == "DStream");
        assert!(has_dstream);
    }

    #[tokio::test]
    async fn test_migration_stats() {
        let config = MigrationConfig::default();
        let tool = MigrationTool::new(config);

        let stats = tool.get_stats().await;
        assert_eq!(stats.total_migrations, 0);
    }
}
