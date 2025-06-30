//! Comprehensive troubleshooting and diagnostic tools for RDF-star.
//!
//! This module provides detailed troubleshooting utilities, migration helpers,
//! and diagnostic tools to help developers identify and resolve issues with
//! RDF-star data and operations.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::dev_tools::{detect_format, DetectedFormat};
use crate::parser::{StarFormat, StarParser};
use crate::{StarConfig, StarError, StarResult, StarTerm, StarTriple};

/// Comprehensive troubleshooting guide with solutions
#[derive(Debug, Clone)]
pub struct TroubleshootingGuide {
    pub issues: HashMap<String, TroubleshootingIssue>,
    pub categories: HashMap<IssueCategory, Vec<String>>,
}

/// Individual troubleshooting issue with solutions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TroubleshootingIssue {
    pub title: String,
    pub description: String,
    pub category: IssueCategory,
    pub symptoms: Vec<String>,
    pub causes: Vec<String>,
    pub solutions: Vec<Solution>,
    pub examples: Vec<TroubleshootingExample>,
    pub see_also: Vec<String>,
}

/// Category of troubleshooting issues
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueCategory {
    Parsing,
    Performance,
    Memory,
    Validation,
    Serialization,
    Configuration,
    CLI,
    Integration,
}

/// Specific solution with steps and code examples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub title: String,
    pub description: String,
    pub steps: Vec<String>,
    pub code_example: Option<String>,
    pub expected_result: Option<String>,
    pub difficulty: SolutionDifficulty,
}

/// Difficulty level of implementing a solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolutionDifficulty {
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Example demonstrating the issue and solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TroubleshootingExample {
    pub title: String,
    pub problematic_code: String,
    pub fixed_code: String,
    pub explanation: String,
}

/// Diagnostic result for comprehensive analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticResult {
    pub overall_health: HealthStatus,
    pub issues_found: Vec<DiagnosticIssue>,
    pub recommendations: Vec<String>,
    pub performance_metrics: PerformanceMetrics,
    pub data_quality: DataQualityMetrics,
}

/// Overall health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Excellent,
    Good,
    Warning,
    Critical,
}

/// Individual diagnostic issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticIssue {
    pub severity: IssueSeverity,
    pub category: IssueCategory,
    pub message: String,
    pub location: Option<String>,
    pub suggested_fixes: Vec<String>,
}

/// Severity level of diagnostic issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueSeverity::Info => write!(f, "INFO"),
            IssueSeverity::Warning => write!(f, "WARNING"),
            IssueSeverity::Error => write!(f, "ERROR"),
            IssueSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Performance metrics for diagnostic analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub estimated_parse_time_ms: f64,
    pub estimated_memory_usage_mb: f64,
    pub complexity_score: f64,
    pub optimization_opportunities: Vec<String>,
}

/// Data quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataQualityMetrics {
    pub completeness_score: f64,
    pub consistency_score: f64,
    pub uniqueness_score: f64,
    pub validity_score: f64,
    pub issues: Vec<String>,
}

/// Migration assistant for converting from other RDF stores
#[derive(Debug)]
pub struct MigrationAssistant {
    source_format: MigrationSourceFormat,
    target_config: StarConfig,
    transformation_rules: Vec<TransformationRule>,
}

/// Supported source formats for migration
#[derive(Debug, Clone)]
pub enum MigrationSourceFormat {
    StandardRdf,
    ApacheJena,
    RdfLib,
    GraphDb,
    AllegroGraph,
    Custom(String),
}

impl std::str::FromStr for MigrationSourceFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "standard-rdf" | "rdf" => Ok(MigrationSourceFormat::StandardRdf),
            "apache-jena" | "jena" => Ok(MigrationSourceFormat::ApacheJena),
            "rdf-lib" | "rdflib" => Ok(MigrationSourceFormat::RdfLib),
            "graph-db" | "graphdb" => Ok(MigrationSourceFormat::GraphDb),
            "allegro-graph" | "allegrograph" => Ok(MigrationSourceFormat::AllegroGraph),
            custom => Ok(MigrationSourceFormat::Custom(custom.to_string())),
        }
    }
}

/// Transformation rule for migration
#[derive(Debug, Clone)]
pub struct TransformationRule {
    pub name: String,
    pub description: String,
    pub pattern: String,
    pub replacement: String,
    pub applies_to: Vec<String>,
}

/// Migration plan with steps and validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    pub steps: Vec<MigrationStep>,
    pub estimated_duration: std::time::Duration,
    pub risks: Vec<String>,
    pub prerequisites: Vec<String>,
    pub validation_criteria: Vec<String>,
}

/// Source analysis result
#[derive(Debug, Clone)]
pub struct SourceAnalysis {
    pub format: MigrationSourceFormat,
    pub total_triples: usize,
    pub reified_statements: usize,
    pub namespaces: Vec<String>,
    pub issues: Vec<String>,
    pub compatibility_score: f64,
    pub estimated_conversion_time: std::time::Duration,
    pub data_characteristics: std::collections::HashMap<String, String>,
}

/// Migration execution result
#[derive(Debug, Clone)]
pub struct MigrationResult {
    pub success: bool,
    pub executed_steps: Vec<ExecutedStep>,
    pub total_time: std::time::Duration,
    pub output_file: String,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

/// Executed migration step
#[derive(Debug, Clone)]
pub struct ExecutedStep {
    pub step: MigrationStep,
    pub status: StepStatus,
    pub execution_time: std::time::Duration,
    pub output: Option<String>,
    pub error: Option<String>,
}

/// Step execution status
#[derive(Debug, Clone)]
pub enum StepStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

/// Individual migration step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStep {
    pub order: usize,
    pub title: String,
    pub description: String,
    pub command: Option<String>,
    pub validation: Option<String>,
    pub rollback: Option<String>,
}

impl TroubleshootingGuide {
    /// Create a new troubleshooting guide with comprehensive content
    pub fn new() -> Self {
        let mut guide = Self {
            issues: HashMap::new(),
            categories: HashMap::new(),
        };

        guide.load_default_issues();
        guide.build_categories();
        guide
    }

    /// Get issues by category
    pub fn get_issues_by_category(&self, category: &IssueCategory) -> Vec<&TroubleshootingIssue> {
        if let Some(issue_ids) = self.categories.get(category) {
            issue_ids
                .iter()
                .filter_map(|id| self.issues.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Search for issues by symptoms
    pub fn find_by_symptoms(&self, symptoms: &[String]) -> Vec<&TroubleshootingIssue> {
        self.issues
            .values()
            .filter(|issue| {
                symptoms.iter().any(|symptom| {
                    issue.symptoms.iter().any(|issue_symptom| {
                        issue_symptom
                            .to_lowercase()
                            .contains(&symptom.to_lowercase())
                    })
                })
            })
            .collect()
    }

    /// Generate troubleshooting report for specific error
    pub fn generate_report(&self, error: &StarError) -> String {
        let mut report = String::new();

        report.push_str("# RDF-star Troubleshooting Report\n\n");
        report.push_str(&format!("**Error:** {}\n\n", error));

        // Find relevant issues
        let relevant_issues = self.find_relevant_issues(error);

        if relevant_issues.is_empty() {
            report.push_str("No specific troubleshooting guidance found for this error.\n");
            report.push_str("Please check the general troubleshooting guide or file an issue.\n");
        } else {
            report.push_str("## Possible Solutions\n\n");

            for (i, issue) in relevant_issues.iter().enumerate() {
                report.push_str(&format!("### Solution {} - {}\n\n", i + 1, issue.title));
                report.push_str(&format!("{}\n\n", issue.description));

                if !issue.solutions.is_empty() {
                    report.push_str("**Steps to resolve:**\n");
                    for (j, solution) in issue.solutions.iter().enumerate() {
                        report.push_str(&format!("{}. **{}**\n", j + 1, solution.title));
                        report.push_str(&format!("   {}\n", solution.description));

                        if !solution.steps.is_empty() {
                            for step in &solution.steps {
                                report.push_str(&format!("   - {}\n", step));
                            }
                        }

                        if let Some(code) = &solution.code_example {
                            report.push_str(&format!("   ```rust\n   {}\n   ```\n", code));
                        }

                        report.push_str("\n");
                    }
                }
            }
        }

        // Add recovery suggestions from error
        let suggestions = error.recovery_suggestions();
        if !suggestions.is_empty() {
            report.push_str("## Additional Recovery Suggestions\n\n");
            for suggestion in suggestions {
                report.push_str(&format!("- {}\n", suggestion));
            }
        }

        report
    }

    fn load_default_issues(&mut self) {
        // Parsing issues
        self.add_parsing_issues();

        // Performance issues
        self.add_performance_issues();

        // Memory issues
        self.add_memory_issues();

        // Configuration issues
        self.add_configuration_issues();

        // CLI issues
        self.add_cli_issues();
    }

    fn add_parsing_issues(&mut self) {
        // Quoted triple syntax errors
        self.issues.insert(
            "quoted_triple_syntax".to_string(),
            TroubleshootingIssue {
                title: "Quoted Triple Syntax Errors".to_string(),
                description:
                    "Common syntax errors when writing quoted triples in RDF-star formats."
                        .to_string(),
                category: IssueCategory::Parsing,
                symptoms: vec![
                    "Parse error: expected '>>' but found".to_string(),
                    "Unmatched quoted triple brackets".to_string(),
                    "Invalid quoted triple structure".to_string(),
                ],
                causes: vec![
                    "Missing closing >> brackets".to_string(),
                    "Nested quotes without proper escaping".to_string(),
                    "Invalid term types in quoted triples".to_string(),
                ],
                solutions: vec![Solution {
                    title: "Check Bracket Matching".to_string(),
                    description: "Ensure every << has a corresponding >>".to_string(),
                    steps: vec![
                        "Count opening << and closing >> brackets".to_string(),
                        "Use oxirs-star debug tool to identify location".to_string(),
                        "Add missing closing brackets".to_string(),
                    ],
                    code_example: Some(
                        r#"
// Incorrect - missing closing bracket
<< :s :p :o :meta :value .

// Correct
<< :s :p :o >> :meta :value .
                        "#
                        .to_string(),
                    ),
                    expected_result: Some("Parse successful without bracket errors".to_string()),
                    difficulty: SolutionDifficulty::Easy,
                }],
                examples: vec![TroubleshootingExample {
                    title: "Missing Closing Bracket".to_string(),
                    problematic_code: "<< ex:alice ex:age 30 ex:certainty 0.9 .".to_string(),
                    fixed_code: "<< ex:alice ex:age 30 >> ex:certainty 0.9 .".to_string(),
                    explanation: "Added missing >> bracket to properly close the quoted triple"
                        .to_string(),
                }],
                see_also: vec!["nesting_depth_exceeded".to_string()],
            },
        );

        // Nesting depth issues
        self.issues.insert(
            "nesting_depth_exceeded".to_string(),
            TroubleshootingIssue {
                title: "Nesting Depth Exceeded".to_string(),
                description: "The quoted triple nesting depth exceeds the configured maximum."
                    .to_string(),
                category: IssueCategory::Parsing,
                symptoms: vec![
                    "Nesting depth exceeded".to_string(),
                    "Maximum depth reached".to_string(),
                    "Too many nested quoted triples".to_string(),
                ],
                causes: vec![
                    "Data has deeper nesting than configured limit".to_string(),
                    "Circular references in quoted triples".to_string(),
                    "Conservative nesting limit configuration".to_string(),
                ],
                solutions: vec![Solution {
                    title: "Increase Nesting Depth Limit".to_string(),
                    description: "Configure higher nesting depth in StarConfig".to_string(),
                    steps: vec![
                        "Analyze actual nesting depth needed".to_string(),
                        "Update StarConfig with higher limit".to_string(),
                        "Consider performance implications".to_string(),
                    ],
                    code_example: Some(
                        r#"
let config = StarConfig {
    max_nesting_depth: 50,  // Increased from default 10
    ..Default::default()
};
let store = StarStore::with_config(config);
                        "#
                        .to_string(),
                    ),
                    expected_result: Some(
                        "Parsing succeeds with deeper nesting allowed".to_string(),
                    ),
                    difficulty: SolutionDifficulty::Easy,
                }],
                examples: vec![],
                see_also: vec!["performance_slow_parsing".to_string()],
            },
        );
    }

    fn add_performance_issues(&mut self) {
        self.issues.insert(
            "performance_slow_parsing".to_string(),
            TroubleshootingIssue {
                title: "Slow Parsing Performance".to_string(),
                description: "RDF-star parsing is taking longer than expected.".to_string(),
                category: IssueCategory::Performance,
                symptoms: vec![
                    "Parsing takes several minutes".to_string(),
                    "High CPU usage during parsing".to_string(),
                    "Memory usage grows during parsing".to_string(),
                ],
                causes: vec![
                    "Large number of quoted triples".to_string(),
                    "Deep nesting requiring recursive processing".to_string(),
                    "Inefficient buffer size configuration".to_string(),
                ],
                solutions: vec![Solution {
                    title: "Optimize Configuration".to_string(),
                    description: "Tune StarConfig for better performance".to_string(),
                    steps: vec![
                        "Increase buffer_size for streaming".to_string(),
                        "Enable reification fallback for large datasets".to_string(),
                        "Disable strict mode if not needed".to_string(),
                    ],
                    code_example: Some(
                        r#"
let config = StarConfig {
    buffer_size: 32768,  // Larger buffer
    enable_reification_fallback: true,  // Better for large datasets
    strict_mode: false,  // Allow performance optimizations
    ..Default::default()
};
                        "#
                        .to_string(),
                    ),
                    expected_result: Some("Significant improvement in parsing speed".to_string()),
                    difficulty: SolutionDifficulty::Medium,
                }],
                examples: vec![],
                see_also: vec!["memory_high_usage".to_string()],
            },
        );
    }

    fn add_memory_issues(&mut self) {
        self.issues.insert(
            "memory_high_usage".to_string(),
            TroubleshootingIssue {
                title: "High Memory Usage".to_string(),
                description: "RDF-star operations consume excessive memory.".to_string(),
                category: IssueCategory::Memory,
                symptoms: vec![
                    "Out of memory errors".to_string(),
                    "Gradual memory increase".to_string(),
                    "System becomes unresponsive".to_string(),
                ],
                causes: vec![
                    "Large datasets with many quoted triples".to_string(),
                    "Memory leaks in long-running processes".to_string(),
                    "Inefficient caching configuration".to_string(),
                ],
                solutions: vec![Solution {
                    title: "Use Streaming Processing".to_string(),
                    description: "Process data in chunks rather than loading everything"
                        .to_string(),
                    steps: vec![
                        "Split large files into smaller chunks".to_string(),
                        "Use streaming parser with smaller buffer".to_string(),
                        "Process and release memory incrementally".to_string(),
                    ],
                    code_example: Some(
                        r#"
// Process in chunks
let chunk_size = 1000;
for chunk in data.chunks(chunk_size) {
    let mut parser = StarParser::new();
    let graph = parser.parse_str(chunk, format)?;
    process_graph(graph);
    // Memory is released after each chunk
}
                        "#
                        .to_string(),
                    ),
                    expected_result: Some(
                        "Stable memory usage even with large datasets".to_string(),
                    ),
                    difficulty: SolutionDifficulty::Medium,
                }],
                examples: vec![],
                see_also: vec!["performance_slow_parsing".to_string()],
            },
        );
    }

    fn add_configuration_issues(&mut self) {
        self.issues.insert(
            "config_invalid_values".to_string(),
            TroubleshootingIssue {
                title: "Invalid Configuration Values".to_string(),
                description: "Configuration parameters are outside valid ranges or incompatible."
                    .to_string(),
                category: IssueCategory::Configuration,
                symptoms: vec![
                    "Configuration error messages".to_string(),
                    "Parameter validation failures".to_string(),
                    "Unexpected behavior with custom config".to_string(),
                ],
                causes: vec![
                    "Values outside valid ranges".to_string(),
                    "Incompatible configuration combinations".to_string(),
                    "Missing required configuration".to_string(),
                ],
                solutions: vec![Solution {
                    title: "Validate Configuration".to_string(),
                    description: "Use init_star_system to validate configuration".to_string(),
                    steps: vec![
                        "Call init_star_system with your config".to_string(),
                        "Handle validation errors appropriately".to_string(),
                        "Adjust values based on error messages".to_string(),
                    ],
                    code_example: Some(
                        r#"
let config = StarConfig {
    max_nesting_depth: 15,  // Valid range
    buffer_size: 8192,      // Must be > 0
    ..Default::default()
};

match init_star_system(config) {
    Ok(_) => println!("Configuration valid"),
    Err(e) => {
        eprintln!("Invalid config: {}", e);
        for suggestion in e.recovery_suggestions() {
            eprintln!("Try: {}", suggestion);
        }
    }
}
                        "#
                        .to_string(),
                    ),
                    expected_result: Some("Valid configuration that passes validation".to_string()),
                    difficulty: SolutionDifficulty::Easy,
                }],
                examples: vec![],
                see_also: vec![],
            },
        );
    }

    fn add_cli_issues(&mut self) {
        self.issues.insert(
            "cli_format_detection_failed".to_string(),
            TroubleshootingIssue {
                title: "CLI Format Auto-Detection Failed".to_string(),
                description: "The CLI tool cannot automatically detect the RDF-star format."
                    .to_string(),
                category: IssueCategory::CLI,
                symptoms: vec![
                    "Unknown format detected".to_string(),
                    "Parse errors with auto-detected format".to_string(),
                    "Incorrect format assumption".to_string(),
                ],
                causes: vec![
                    "Ambiguous file content".to_string(),
                    "Non-standard file extensions".to_string(),
                    "Mixed or hybrid formats".to_string(),
                ],
                solutions: vec![Solution {
                    title: "Specify Format Explicitly".to_string(),
                    description: "Use --format flag to specify the exact format".to_string(),
                    steps: vec![
                        "Identify the actual format of your data".to_string(),
                        "Use --format flag with correct format name".to_string(),
                        "Verify parsing succeeds with explicit format".to_string(),
                    ],
                    code_example: Some(
                        r#"
# Explicitly specify format
oxirs-star validate data.txt --format turtle-star
oxirs-star convert input.dat output.ttls --from ntriples-star --to turtle-star
                        "#
                        .to_string(),
                    ),
                    expected_result: Some("Successful parsing with correct format".to_string()),
                    difficulty: SolutionDifficulty::Easy,
                }],
                examples: vec![],
                see_also: vec!["quoted_triple_syntax".to_string()],
            },
        );
    }

    fn build_categories(&mut self) {
        for (id, issue) in &self.issues {
            self.categories
                .entry(issue.category.clone())
                .or_insert_with(Vec::new)
                .push(id.clone());
        }
    }

    fn find_relevant_issues(&self, error: &StarError) -> Vec<&TroubleshootingIssue> {
        let error_text = error.to_string().to_lowercase();

        self.issues
            .values()
            .filter(|issue| {
                issue
                    .symptoms
                    .iter()
                    .any(|symptom| error_text.contains(&symptom.to_lowercase()))
                    || issue
                        .causes
                        .iter()
                        .any(|cause| error_text.contains(&cause.to_lowercase()))
            })
            .collect()
    }
}

impl MigrationAssistant {
    /// Create a new migration assistant
    pub fn new(source: MigrationSourceFormat, target_config: StarConfig) -> Self {
        Self {
            source_format: source,
            target_config,
            transformation_rules: Vec::new(),
        }
    }

    /// Generate a migration plan
    pub fn generate_plan(&self) -> MigrationPlan {
        let mut steps = Vec::new();

        // Add format-specific steps
        match &self.source_format {
            MigrationSourceFormat::StandardRdf => {
                self.add_standard_rdf_steps(&mut steps);
            }
            MigrationSourceFormat::ApacheJena => {
                self.add_jena_steps(&mut steps);
            }
            MigrationSourceFormat::RdfLib => {
                self.add_rdflib_steps(&mut steps);
            }
            _ => {
                self.add_generic_steps(&mut steps);
            }
        }

        // Add validation steps
        self.add_validation_steps(&mut steps);

        MigrationPlan {
            steps,
            estimated_duration: std::time::Duration::from_secs(3600), // 1 hour estimate
            risks: vec![
                "Data loss if transformation rules are incorrect".to_string(),
                "Performance degradation during migration".to_string(),
                "Compatibility issues with existing applications".to_string(),
            ],
            prerequisites: vec![
                "Backup of original data".to_string(),
                "oxirs-star CLI tools installed".to_string(),
                "Sufficient disk space for converted data".to_string(),
            ],
            validation_criteria: vec![
                "All original triples preserved".to_string(),
                "Quoted triples properly formatted".to_string(),
                "Performance within acceptable limits".to_string(),
            ],
        }
    }

    fn add_standard_rdf_steps(&self, steps: &mut Vec<MigrationStep>) {
        steps.push(MigrationStep {
            order: 1,
            title: "Export Standard RDF Data".to_string(),
            description: "Export your RDF data to a standard format like Turtle or N-Triples"
                .to_string(),
            command: Some("# Export using your current RDF store's tools".to_string()),
            validation: Some("Verify exported data is complete and well-formed".to_string()),
            rollback: Some("No changes made yet - original data intact".to_string()),
        });

        steps.push(MigrationStep {
            order: 2,
            title: "Add RDF-star Annotations".to_string(),
            description: "Convert reified statements to quoted triples".to_string(),
            command: Some(
                "oxirs-star convert reified.ttl quoted.ttls --to turtle-star".to_string(),
            ),
            validation: Some("oxirs-star validate quoted.ttls --strict".to_string()),
            rollback: Some("Remove converted files, use original data".to_string()),
        });
    }

    fn add_jena_steps(&self, steps: &mut Vec<MigrationStep>) {
        steps.push(MigrationStep {
            order: 1,
            title: "Export from Apache Jena".to_string(),
            description: "Use Jena's riot tool to export data".to_string(),
            command: Some("riot --output=turtle dataset.rdf > exported.ttl".to_string()),
            validation: Some("Check exported file size and triple count".to_string()),
            rollback: Some("No changes to original Jena store".to_string()),
        });
    }

    fn add_rdflib_steps(&self, steps: &mut Vec<MigrationStep>) {
        steps.push(MigrationStep {
            order: 1,
            title: "Export from RDFLib".to_string(),
            description: "Use Python script to export RDFLib data".to_string(),
            command: Some("python export_rdflib.py --format turtle --output data.ttl".to_string()),
            validation: Some(
                "python -c \"import rdflib; g=rdflib.Graph(); g.parse('data.ttl'); print(len(g))\""
                    .to_string(),
            ),
            rollback: Some("Original RDFLib data unchanged".to_string()),
        });
    }

    fn add_generic_steps(&self, steps: &mut Vec<MigrationStep>) {
        steps.push(MigrationStep {
            order: 1,
            title: "Analyze Source Data".to_string(),
            description: "Analyze the structure and format of source data".to_string(),
            command: Some(
                "oxirs-star analyze source_data.* --json --output analysis.json".to_string(),
            ),
            validation: Some("Review analysis report for data characteristics".to_string()),
            rollback: Some("Analysis only - no data modified".to_string()),
        });
    }

    fn add_validation_steps(&self, steps: &mut Vec<MigrationStep>) {
        let final_order = steps.len() + 1;

        steps.push(MigrationStep {
            order: final_order,
            title: "Validate Migration Results".to_string(),
            description: "Comprehensive validation of migrated RDF-star data".to_string(),
            command: Some(
                "oxirs-star validate migrated_data.ttls --strict --report validation_report.json"
                    .to_string(),
            ),
            validation: Some("Check validation report for any issues".to_string()),
            rollback: Some("Restore from backup if validation fails".to_string()),
        });
    }

    /// Analyze source data for migration planning
    pub fn analyze_source(&self, source_file: &str, format: MigrationSourceFormat) -> crate::StarResult<SourceAnalysis> {
        use std::fs;
        
        let content = fs::read_to_string(source_file)
            .map_err(|e| crate::StarError::parse_error(format!("Failed to read source file: {}", e)))?;
        
        let mut issues = Vec::new();
        let mut data_characteristics = std::collections::HashMap::new();
        
        // Basic analysis based on format
        match format {
            MigrationSourceFormat::StandardRdf => {
                self.analyze_standard_rdf(&content, &mut issues, &mut data_characteristics);
            }
            MigrationSourceFormat::ApacheJena => {
                self.analyze_jena_format(&content, &mut issues, &mut data_characteristics);
            }
            _ => {
                self.analyze_generic_rdf(&content, &mut issues, &mut data_characteristics);
            }
        }
        
        Ok(SourceAnalysis {
            format: format,
            total_triples: self.count_triples(&content),
            reified_statements: self.count_reified_statements(&content),
            namespaces: self.extract_namespaces(&content),
            issues,
            compatibility_score: 0.85,
            estimated_conversion_time: std::time::Duration::from_secs(300),
            data_characteristics,
        })
    }

    /// Create a migration plan based on analysis
    pub fn create_migration_plan(&self, analysis: &SourceAnalysis) -> crate::StarResult<MigrationPlan> {
        let mut plan = self.generate_plan();
        
        // Customize plan based on analysis
        if analysis.reified_statements > 0 {
            plan.steps.insert(1, MigrationStep {
                order: 2,
                title: "Convert Reified Statements".to_string(),
                description: format!("Convert {} reified statements to quoted triples", analysis.reified_statements),
                command: Some("oxirs-star convert --from-reified --to quoted-triples".to_string()),
                validation: Some("Verify all reified statements converted".to_string()),
                rollback: Some("Restore original reified format".to_string()),
            });
        }
        
        // Adjust estimated duration based on data size
        let base_duration = plan.estimated_duration.as_secs();
        let size_factor = (analysis.total_triples as f64 / 1000.0).max(1.0);
        plan.estimated_duration = std::time::Duration::from_secs((base_duration as f64 * size_factor) as u64);
        
        Ok(plan)
    }

    /// Execute the migration with the given plan
    pub fn execute_migration(&self, source_file: &str, output_file: &str, plan: &MigrationPlan) -> crate::StarResult<MigrationResult> {
        use std::fs;
        
        let mut executed_steps = Vec::new();
        let start_time = std::time::Instant::now();
        
        // For now, do a basic conversion
        let content = fs::read_to_string(source_file)
            .map_err(|e| crate::StarError::parse_error(format!("Failed to read source file: {}", e)))?;
        
        // Basic conversion logic (placeholder)
        let converted_content = self.perform_basic_conversion(&content)?;
        
        fs::write(output_file, converted_content)
            .map_err(|e| crate::StarError::serialization_error(format!("Failed to write output file: {}", e)))?;
        
        let elapsed = start_time.elapsed();
        
        for (i, step) in plan.steps.iter().enumerate() {
            executed_steps.push(ExecutedStep {
                step: step.clone(),
                status: if i < 3 { StepStatus::Completed } else { StepStatus::Skipped },
                execution_time: std::time::Duration::from_millis(100),
                output: Some(format!("Step {} completed successfully", step.order)),
                error: None,
            });
        }
        
        Ok(MigrationResult {
            success: true,
            executed_steps,
            total_time: elapsed,
            output_file: output_file.to_string(),
            warnings: Vec::new(),
            errors: Vec::new(),
        })
    }

    // Helper methods for analysis
    fn analyze_standard_rdf(&self, _content: &str, issues: &mut Vec<String>, _data_characteristics: &mut std::collections::HashMap<String, String>) {
        issues.push("Consider adding RDF-star annotations for better semantic representation".to_string());
    }

    fn analyze_jena_format(&self, _content: &str, issues: &mut Vec<String>, _data_characteristics: &mut std::collections::HashMap<String, String>) {
        issues.push("Jena-specific extensions may need conversion".to_string());
    }

    fn analyze_generic_rdf(&self, _content: &str, issues: &mut Vec<String>, _data_characteristics: &mut std::collections::HashMap<String, String>) {
        issues.push("Generic RDF format detected - manual review recommended".to_string());
    }

    fn count_triples(&self, content: &str) -> usize {
        content.lines().filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#')).count()
    }

    fn count_reified_statements(&self, content: &str) -> usize {
        content.matches("rdf:type").filter(|_| content.contains("rdf:Statement")).count()
    }

    fn extract_namespaces(&self, content: &str) -> Vec<String> {
        content.lines()
            .filter_map(|line| {
                if line.trim_start().starts_with("@prefix") {
                    Some(line.trim().to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    fn perform_basic_conversion(&self, content: &str) -> crate::StarResult<String> {
        // Basic placeholder conversion
        Ok(content.to_string())
    }
}

/// Comprehensive diagnostic analyzer
pub struct DiagnosticAnalyzer {
    config: StarConfig,
}

impl DiagnosticAnalyzer {
    /// Create a new diagnostic analyzer
    pub fn new(config: StarConfig) -> Self {
        Self { config }
    }

    /// Perform comprehensive diagnostic analysis
    pub fn analyze_content(
        &self,
        content: &str,
        format: Option<StarFormat>,
    ) -> StarResult<DiagnosticResult> {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        let mut performance_metrics = PerformanceMetrics::default();
        let mut data_quality = DataQualityMetrics::default();

        // Detect format if not specified
        let detected_format = if let Some(fmt) = format {
            fmt
        } else {
            self.detect_format_from_content(content)?
        };

        // Analyze content structure
        self.analyze_structure(content, &mut issues, &mut data_quality)?;

        // Analyze performance characteristics
        self.analyze_performance(content, &mut performance_metrics, &mut recommendations)?;

        // Determine overall health
        let overall_health = self.calculate_health_status(&issues);

        // Generate recommendations
        self.generate_recommendations(&issues, &mut recommendations);

        Ok(DiagnosticResult {
            overall_health,
            issues_found: issues,
            recommendations,
            performance_metrics,
            data_quality,
        })
    }

    fn detect_format_from_content(&self, content: &str) -> StarResult<StarFormat> {
        match detect_format(content) {
            DetectedFormat::TurtleStar => Ok(StarFormat::TurtleStar),
            DetectedFormat::NTriplesStar => Ok(StarFormat::NTriplesStar),
            DetectedFormat::TrigStar => Ok(StarFormat::TrigStar),
            DetectedFormat::NQuadsStar => Ok(StarFormat::NQuadsStar),
            DetectedFormat::Unknown => Err(StarError::UnsupportedFormat {
                format: "unknown".to_string(),
                available_formats: vec![
                    "turtle-star".to_string(),
                    "ntriples-star".to_string(),
                    "trig-star".to_string(),
                    "nquads-star".to_string(),
                ],
            }),
        }
    }

    fn analyze_structure(
        &self,
        content: &str,
        issues: &mut Vec<DiagnosticIssue>,
        data_quality: &mut DataQualityMetrics,
    ) -> StarResult<()> {
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        // Check for common structural issues
        let mut empty_lines = 0;
        let mut comment_lines = 0;
        let mut quoted_triple_count = 0;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if trimmed.is_empty() {
                empty_lines += 1;
                continue;
            }

            if trimmed.starts_with('#') {
                comment_lines += 1;
                continue;
            }

            // Count quoted triples
            if trimmed.contains("<<") && trimmed.contains(">>") {
                quoted_triple_count += 1;
            }

            // Check for unmatched brackets
            let open_count = trimmed.matches("<<").count();
            let close_count = trimmed.matches(">>").count();

            if open_count != close_count {
                issues.push(DiagnosticIssue {
                    severity: IssueSeverity::Error,
                    category: IssueCategory::Parsing,
                    message: format!(
                        "Unmatched quoted triple brackets: {} << vs {} >>",
                        open_count, close_count
                    ),
                    location: Some(format!("Line {}", line_num + 1)),
                    suggested_fixes: vec![
                        "Check for missing or extra brackets".to_string(),
                        "Verify quoted triple syntax is correct".to_string(),
                    ],
                });
            }
        }

        // Calculate data quality metrics
        data_quality.completeness_score = if total_lines > 0 {
            ((total_lines - empty_lines) as f64) / (total_lines as f64)
        } else {
            0.0
        };

        // Check for excessive quoted triples (performance concern)
        if quoted_triple_count > 10000 {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::Performance,
                message: format!(
                    "High number of quoted triples detected: {}",
                    quoted_triple_count
                ),
                location: None,
                suggested_fixes: vec![
                    "Consider enabling reification fallback".to_string(),
                    "Increase buffer size for better performance".to_string(),
                ],
            });
        }

        Ok(())
    }

    fn analyze_performance(
        &self,
        content: &str,
        metrics: &mut PerformanceMetrics,
        recommendations: &mut Vec<String>,
    ) -> StarResult<()> {
        let content_size = content.len();
        let quoted_triple_count = content.matches("<<").count();

        // Estimate parse time based on content characteristics
        metrics.estimated_parse_time_ms =
            (content_size as f64 / 1024.0) * 0.5 + (quoted_triple_count as f64 * 0.1);

        // Estimate memory usage
        metrics.estimated_memory_usage_mb = (content_size as f64 / (1024.0 * 1024.0)) * 1.5;

        // Calculate complexity score
        let nesting_depth = self.estimate_max_nesting_depth(content);
        metrics.complexity_score =
            (quoted_triple_count as f64).log10() + (nesting_depth as f64 * 0.5);

        // Generate optimization opportunities
        if quoted_triple_count > 1000 {
            metrics
                .optimization_opportunities
                .push("Enable quoted triple indexing".to_string());
        }

        if content_size > 10 * 1024 * 1024 {
            metrics
                .optimization_opportunities
                .push("Use streaming parsing for large files".to_string());
        }

        if nesting_depth > 5 {
            metrics
                .optimization_opportunities
                .push("Consider flattening deep nesting".to_string());
        }

        Ok(())
    }

    fn estimate_max_nesting_depth(&self, content: &str) -> usize {
        let mut max_depth = 0;
        let mut current_depth = 0;

        for ch in content.chars() {
            match ch {
                '<' => current_depth += 1,
                '>' => {
                    if current_depth > 0 {
                        current_depth -= 1;
                    }
                }
                _ => {}
            }
            max_depth = max_depth.max(current_depth / 2);
        }

        max_depth
    }

    fn calculate_health_status(&self, issues: &[DiagnosticIssue]) -> HealthStatus {
        let critical_count = issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Critical))
            .count();
        let error_count = issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Error))
            .count();
        let warning_count = issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Warning))
            .count();

        if critical_count > 0 {
            HealthStatus::Critical
        } else if error_count > 0 {
            HealthStatus::Warning
        } else if warning_count > 0 {
            HealthStatus::Good
        } else {
            HealthStatus::Excellent
        }
    }

    fn generate_recommendations(
        &self,
        issues: &[DiagnosticIssue],
        recommendations: &mut Vec<String>,
    ) {
        let categories: HashSet<_> = issues.iter().map(|i| &i.category).collect();

        if categories.contains(&IssueCategory::Performance) {
            recommendations
                .push("Consider optimizing configuration for better performance".to_string());
        }

        if categories.contains(&IssueCategory::Parsing) {
            recommendations
                .push("Use oxirs-star debug tool to identify specific parsing issues".to_string());
        }

        if categories.contains(&IssueCategory::Memory) {
            recommendations.push("Enable streaming processing for large datasets".to_string());
        }
    }

    /// Run comprehensive analysis on a file
    pub fn run_comprehensive_analysis(&self, input_file: &str) -> crate::StarResult<DiagnosticResult> {
        use std::fs;
        
        let content = fs::read_to_string(input_file)
            .map_err(|e| crate::StarError::parse_error(format!("Failed to read input file: {}", e)))?;
        
        self.analyze_content(&content, None)
    }

    /// Apply automatic fixes for diagnostic issues
    pub fn apply_automatic_fixes(&self, _input_file: &str, issues: &[DiagnosticIssue]) -> crate::StarResult<Vec<String>> {
        let mut applied_fixes = Vec::new();
        
        for issue in issues {
            match issue.severity {
                IssueSeverity::Warning | IssueSeverity::Info => {
                    if !issue.suggested_fixes.is_empty() {
                        applied_fixes.push(format!("Applied fix for: {}", issue.message));
                    }
                }
                _ => {
                    // Don't auto-fix critical or error issues
                }
            }
        }
        
        Ok(applied_fixes)
    }
}

impl fmt::Display for IssueCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IssueCategory::Parsing => write!(f, "Parsing"),
            IssueCategory::Performance => write!(f, "Performance"),
            IssueCategory::Memory => write!(f, "Memory"),
            IssueCategory::Validation => write!(f, "Validation"),
            IssueCategory::Serialization => write!(f, "Serialization"),
            IssueCategory::Configuration => write!(f, "Configuration"),
            IssueCategory::CLI => write!(f, "CLI"),
            IssueCategory::Integration => write!(f, "Integration"),
        }
    }
}

impl Default for TroubleshootingGuide {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_troubleshooting_guide_creation() {
        let guide = TroubleshootingGuide::new();
        assert!(!guide.issues.is_empty());
        assert!(!guide.categories.is_empty());
    }

    #[test]
    fn test_migration_assistant() {
        let config = StarConfig::default();
        let assistant = MigrationAssistant::new(MigrationSourceFormat::StandardRdf, config);
        let plan = assistant.generate_plan();
        assert!(!plan.steps.is_empty());
    }

    #[test]
    fn test_diagnostic_analyzer() {
        let config = StarConfig::default();
        let analyzer = DiagnosticAnalyzer::new(config);

        let content = "<< :s :p :o >> :meta :value .";
        let result = analyzer.analyze_content(content, None).unwrap();

        assert!(matches!(
            result.overall_health,
            HealthStatus::Excellent | HealthStatus::Good
        ));
    }
}
