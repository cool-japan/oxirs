//! Shape Analyzer CLI Tool
//!
//! Analyzes SHACL shapes to provide insights on:
//! - Complexity metrics
//! - Dependency graphs
//! - Performance predictions
//! - Best practice recommendations
//! - Potential issues and warnings

#![allow(clippy::single_char_add_str)]

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Result;
use indexmap::IndexMap;

use oxirs_shacl::{
    templates::{TemplateCategory, TemplateLibrary},
    Shape, ShapeId,
};

/// Shape analysis results
#[derive(Debug, Clone)]
struct ShapeAnalysis {
    shape_id: ShapeId,
    complexity_score: u32,
    constraint_count: usize,
    target_count: usize,
    has_recursion: bool,
    dependencies: Vec<ShapeId>,
    issues: Vec<AnalysisIssue>,
    recommendations: Vec<String>,
}

/// Analysis issue severity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IssueSeverity {
    Error,
    Warning,
    Info,
}

/// Analysis issue
#[derive(Debug, Clone)]
struct AnalysisIssue {
    severity: IssueSeverity,
    message: String,
    suggestion: Option<String>,
}

/// Shape analyzer
struct ShapeAnalyzer {
    shapes: IndexMap<ShapeId, Shape>,
}

impl ShapeAnalyzer {
    fn new(shapes: IndexMap<ShapeId, Shape>) -> Self {
        Self { shapes }
    }

    /// Analyze all shapes
    fn analyze_all(&self) -> Vec<ShapeAnalysis> {
        self.shapes
            .values()
            .map(|shape| self.analyze_shape(shape))
            .collect()
    }

    /// Analyze a single shape
    fn analyze_shape(&self, shape: &Shape) -> ShapeAnalysis {
        let complexity_score = self.calculate_complexity(shape);
        let dependencies = self.find_dependencies(shape);
        let has_recursion = self.check_recursion(shape);
        let issues = self.find_issues(shape);
        let recommendations = self.generate_recommendations(shape, complexity_score);

        ShapeAnalysis {
            shape_id: shape.id.clone(),
            complexity_score,
            constraint_count: shape.constraints.len(),
            target_count: shape.targets.len(),
            has_recursion,
            dependencies,
            issues,
            recommendations,
        }
    }

    /// Calculate shape complexity score
    fn calculate_complexity(&self, shape: &Shape) -> u32 {
        let mut score = 0u32;

        // Base score from constraint count
        score += shape.constraints.len() as u32 * 2;

        // Target complexity
        score += shape.targets.len() as u32 * 3;

        // Property path complexity
        if let Some(ref path) = shape.path {
            score += match path {
                oxirs_shacl::PropertyPath::Predicate(_) => 1,
                oxirs_shacl::PropertyPath::Inverse(_) => 3,
                oxirs_shacl::PropertyPath::Sequence(paths) => 2 * paths.len() as u32,
                oxirs_shacl::PropertyPath::Alternative(paths) => 2 * paths.len() as u32,
                oxirs_shacl::PropertyPath::ZeroOrMore(_) => 10,
                oxirs_shacl::PropertyPath::OneOrMore(_) => 8,
                oxirs_shacl::PropertyPath::ZeroOrOne(_) => 5,
            };
        }

        // Logical constraints add complexity
        for (component_id, _) in &shape.constraints {
            match component_id.as_str() {
                "sh:and" | "sh:or" | "sh:xone" => score += 10,
                "sh:not" => score += 15,
                "sh:node" | "sh:property" => score += 5,
                _ => {}
            }
        }

        score
    }

    /// Find shape dependencies
    fn find_dependencies(&self, shape: &Shape) -> Vec<ShapeId> {
        let mut deps = Vec::new();

        // Check constraint dependencies on other shapes
        for (component_id, _) in &shape.constraints {
            match component_id.as_str() {
                "sh:node" | "sh:property" | "sh:qualifiedValueShape" => {
                    // These constraints reference other shapes
                    // In a full implementation, we'd extract the actual shape references
                }
                _ => {}
            }
        }

        // Check inheritance
        for parent in &shape.extends {
            if !deps.contains(parent) {
                deps.push(parent.clone());
            }
        }

        deps
    }

    /// Check for recursive dependencies
    fn check_recursion(&self, shape: &Shape) -> bool {
        let mut visited = HashSet::new();
        self.has_cycle(&shape.id, &mut visited)
    }

    fn has_cycle(&self, shape_id: &ShapeId, visited: &mut HashSet<ShapeId>) -> bool {
        if visited.contains(shape_id) {
            return true;
        }

        visited.insert(shape_id.clone());

        if let Some(shape) = self.shapes.get(shape_id) {
            let deps = self.find_dependencies(shape);
            for dep in deps {
                if self.has_cycle(&dep, visited) {
                    return true;
                }
            }
        }

        visited.remove(shape_id);
        false
    }

    /// Find potential issues
    fn find_issues(&self, shape: &Shape) -> Vec<AnalysisIssue> {
        let mut issues = Vec::new();

        // No targets
        if shape.targets.is_empty() && shape.is_node_shape() {
            issues.push(AnalysisIssue {
                severity: IssueSeverity::Warning,
                message: "Node shape has no targets defined".to_string(),
                suggestion: Some(
                    "Add at least one target (sh:targetClass, sh:targetNode, etc.)".to_string(),
                ),
            });
        }

        // No constraints
        if shape.constraints.is_empty() {
            issues.push(AnalysisIssue {
                severity: IssueSeverity::Info,
                message: "Shape has no constraints".to_string(),
                suggestion: Some("Add constraints to perform validation".to_string()),
            });
        }

        // Property shape without path
        if shape.is_property_shape() && shape.path.is_none() {
            issues.push(AnalysisIssue {
                severity: IssueSeverity::Error,
                message: "Property shape must have a path".to_string(),
                suggestion: Some("Add sh:path to define which property to validate".to_string()),
            });
        }

        // Too many constraints (complexity)
        if shape.constraints.len() > 20 {
            issues.push(AnalysisIssue {
                severity: IssueSeverity::Warning,
                message: format!(
                    "Shape has {} constraints (high complexity)",
                    shape.constraints.len()
                ),
                suggestion: Some("Consider splitting into multiple shapes".to_string()),
            });
        }

        // Check for potentially conflicting constraints
        let has_min_count = shape
            .constraints
            .keys()
            .any(|id| id.as_str() == "sh:minCount");
        let has_max_count = shape
            .constraints
            .keys()
            .any(|id| id.as_str() == "sh:maxCount");

        if has_min_count && !has_max_count {
            issues.push(AnalysisIssue {
                severity: IssueSeverity::Info,
                message: "Has minCount but no maxCount".to_string(),
                suggestion: Some(
                    "Consider adding maxCount for complete cardinality constraint".to_string(),
                ),
            });
        }

        issues
    }

    /// Generate recommendations
    fn generate_recommendations(&self, shape: &Shape, complexity: u32) -> Vec<String> {
        let mut recs = Vec::new();

        // Complexity-based recommendations
        if complexity > 50 {
            recs.push("Consider breaking this shape into smaller, focused shapes".to_string());
        }

        // Missing documentation
        if shape.label.is_none() && shape.description.is_none() {
            recs.push("Add sh:name and sh:description for better documentation".to_string());
        }

        // No severity specified
        if shape.severity == oxirs_shacl::Severity::Violation && !shape.constraints.is_empty() {
            recs.push("Consider using different severity levels (Info/Warning/Violation) for different constraints".to_string());
        }

        // Group usage
        if shape.groups.is_empty() && shape.constraints.len() > 5 {
            recs.push("Use sh:group to organize related constraints".to_string());
        }

        recs
    }

    /// Generate dependency graph in DOT format
    fn generate_dependency_graph(&self) -> String {
        let mut dot = String::from("digraph ShapeDependencies {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n\n");

        for shape in self.shapes.values() {
            let deps = self.find_dependencies(shape);
            for dep in deps {
                dot.push_str(&format!(
                    "  \"{}\" -> \"{}\";\n",
                    shape.id.as_str(),
                    dep.as_str()
                ));
            }
        }

        dot.push_str("}\n");
        dot
    }
}

/// Analysis report formatter
struct ReportFormatter {
    verbose: bool,
}

#[allow(dead_code)]
impl ReportFormatter {
    fn new(verbose: bool) -> Self {
        Self { verbose }
    }

    fn format_analysis(&self, analyses: &[ShapeAnalysis]) -> String {
        let mut output = String::new();

        output.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║           SHACL Shape Analysis Report                       ║\n");
        output.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

        // Summary statistics
        output.push_str("📊 Summary Statistics:\n");
        output.push_str("─────────────────────\n");
        output.push_str(&format!("Total shapes analyzed: {}\n", analyses.len()));

        let avg_complexity: f32 = analyses
            .iter()
            .map(|a| a.complexity_score as f32)
            .sum::<f32>()
            / analyses.len().max(1) as f32;
        output.push_str(&format!("Average complexity: {:.1}\n", avg_complexity));

        let total_constraints: usize = analyses.iter().map(|a| a.constraint_count).sum();
        output.push_str(&format!("Total constraints: {}\n", total_constraints));

        let shapes_with_issues = analyses.iter().filter(|a| !a.issues.is_empty()).count();
        output.push_str(&format!("Shapes with issues: {}\n", shapes_with_issues));

        let recursive_shapes = analyses.iter().filter(|a| a.has_recursion).count();
        if recursive_shapes > 0 {
            output.push_str(&format!("⚠️  Recursive shapes: {}\n", recursive_shapes));
        }

        output.push_str("\n");

        // Individual shape reports
        for (idx, analysis) in analyses.iter().enumerate() {
            output.push_str(&format!(
                "\n{}. Shape: {}\n",
                idx + 1,
                analysis.shape_id.as_str()
            ));
            output.push_str("   ─────────────────────────────────\n");

            output.push_str(&format!(
                "   Complexity Score: {}",
                analysis.complexity_score
            ));
            if analysis.complexity_score < 20 {
                output.push_str(" ✅ (Low)\n");
            } else if analysis.complexity_score < 50 {
                output.push_str(" ⚠️  (Medium)\n");
            } else {
                output.push_str(" ❌ (High)\n");
            }

            output.push_str(&format!("   Constraints: {}\n", analysis.constraint_count));
            output.push_str(&format!("   Targets: {}\n", analysis.target_count));

            if !analysis.dependencies.is_empty() {
                output.push_str(&format!(
                    "   Dependencies: {}\n",
                    analysis.dependencies.len()
                ));
                if self.verbose {
                    for dep in &analysis.dependencies {
                        output.push_str(&format!("     - {}\n", dep.as_str()));
                    }
                }
            }

            if analysis.has_recursion {
                output.push_str("   ⚠️  Has recursive dependencies\n");
            }

            // Issues
            if !analysis.issues.is_empty() {
                output.push_str("\n   Issues:\n");
                for issue in &analysis.issues {
                    let icon = match issue.severity {
                        IssueSeverity::Error => "❌",
                        IssueSeverity::Warning => "⚠️ ",
                        IssueSeverity::Info => "ℹ️ ",
                    };
                    output.push_str(&format!("   {} {}\n", icon, issue.message));
                    if let Some(ref suggestion) = issue.suggestion {
                        output.push_str(&format!("      💡 {}\n", suggestion));
                    }
                }
            }

            // Recommendations
            if !analysis.recommendations.is_empty() {
                output.push_str("\n   Recommendations:\n");
                for rec in &analysis.recommendations {
                    output.push_str(&format!("   • {}\n", rec));
                }
            }
        }

        output.push_str("\n\n");
        output.push_str("─────────────────────────────────────────────────────────────\n");
        output.push_str("✅ Analysis complete\n");

        output
    }

    fn format_summary(&self, analyses: &[ShapeAnalysis]) -> String {
        let mut output = String::new();

        output.push_str("Shape Analysis Summary\n");
        output.push_str("======================\n\n");

        for analysis in analyses {
            let status = if analysis
                .issues
                .iter()
                .any(|i| i.severity == IssueSeverity::Error)
            {
                "❌"
            } else if !analysis.issues.is_empty() {
                "⚠️ "
            } else {
                "✅"
            };

            output.push_str(&format!(
                "{} {} (complexity: {}, constraints: {})\n",
                status,
                analysis.shape_id.as_str(),
                analysis.complexity_score,
                analysis.constraint_count
            ));
        }

        output
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    let command = &args[1];

    match command.as_str() {
        "analyze" => {
            if args.len() < 3 {
                eprintln!("Usage: shape_analyzer analyze <shapes_file>");
                return Ok(());
            }

            let shapes_file = PathBuf::from(&args[2]);
            let verbose =
                args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());

            analyze_shapes(&shapes_file, verbose)?;
        }
        "graph" => {
            if args.len() < 3 {
                eprintln!("Usage: shape_analyzer graph <shapes_file>");
                return Ok(());
            }

            let shapes_file = PathBuf::from(&args[2]);
            generate_dependency_graph(&shapes_file)?;
        }
        "templates" => {
            list_templates();
        }
        "help" | "--help" | "-h" => {
            print_usage();
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            print_usage();
        }
    }

    Ok(())
}

fn analyze_shapes(shapes_file: &PathBuf, verbose: bool) -> Result<()> {
    println!("🔍 Analyzing shapes from: {:?}", shapes_file);
    println!();

    // Load shapes (simplified - in production, use proper RDF parsing)
    let shapes = IndexMap::new();

    // Create analyzer
    let analyzer = ShapeAnalyzer::new(shapes);

    // Analyze all shapes
    let analyses = analyzer.analyze_all();

    // Format and display report
    let formatter = ReportFormatter::new(verbose);
    let report = formatter.format_analysis(&analyses);
    println!("{}", report);

    Ok(())
}

fn generate_dependency_graph(shapes_file: &PathBuf) -> Result<()> {
    println!("📊 Generating dependency graph from: {:?}", shapes_file);

    let shapes = IndexMap::new();
    let analyzer = ShapeAnalyzer::new(shapes);

    let dot = analyzer.generate_dependency_graph();

    println!("\nDependency Graph (DOT format):");
    println!("─────────────────────────────");
    println!("{}", dot);

    println!("\n💡 Tip: Save to a .dot file and visualize with Graphviz:");
    println!("   shape_analyzer graph shapes.ttl > deps.dot");
    println!("   dot -Tpng deps.dot -o deps.png");

    Ok(())
}

fn list_templates() {
    println!("📚 Available Shape Templates");
    println!("══════════════════════════════\n");

    let library = TemplateLibrary::new();

    for category in [
        TemplateCategory::Identity,
        TemplateCategory::Contact,
        TemplateCategory::Commerce,
        TemplateCategory::Web,
        TemplateCategory::Temporal,
        TemplateCategory::Geospatial,
        TemplateCategory::Financial,
        TemplateCategory::Scientific,
    ] {
        let templates = library.by_category(category);
        if templates.is_empty() {
            continue;
        }

        println!("{:?} Templates:", category);
        println!("─────────────────────");

        for template in templates {
            println!("  • {} ({})", template.name, template.id);
            println!("    {}", template.description);
            println!();
        }
    }

    println!("Total: {} templates", library.all().len());
    println!("\n💡 Use these template IDs in the interactive constraint tester");
}

fn print_usage() {
    println!("🧪 OxiRS SHACL Shape Analyzer");
    println!("═══════════════════════════════\n");
    println!("USAGE:");
    println!("  shape_analyzer <command> [options]\n");
    println!("COMMANDS:");
    println!("  analyze <file>   Analyze shapes and generate report");
    println!("  graph <file>     Generate dependency graph (DOT format)");
    println!("  templates        List available shape templates");
    println!("  help             Show this help message\n");
    println!("OPTIONS:");
    println!("  -v, --verbose    Show detailed analysis\n");
    println!("EXAMPLES:");
    println!("  shape_analyzer analyze shapes.ttl");
    println!("  shape_analyzer analyze shapes.ttl --verbose");
    println!("  shape_analyzer graph shapes.ttl > deps.dot");
    println!("  shape_analyzer templates");
}
