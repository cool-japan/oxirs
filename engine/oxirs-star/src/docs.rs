//! Comprehensive documentation and examples for RDF-star.
//!
//! This module provides extensive documentation, tutorials, and examples
//! for working with RDF-star in various scenarios.

use std::collections::HashMap;

/// Documentation and tutorial system for RDF-star
pub struct StarDocumentation {
    examples: HashMap<String, StarExample>,
    tutorials: HashMap<String, Tutorial>,
    best_practices: Vec<BestPractice>,
}

/// Individual code example with explanation
#[derive(Debug, Clone)]
pub struct StarExample {
    pub title: String,
    pub description: String,
    pub code: String,
    pub expected_output: Option<String>,
    pub difficulty: Difficulty,
    pub tags: Vec<String>,
    pub related_concepts: Vec<String>,
}

/// Interactive tutorial
#[derive(Debug, Clone)]
pub struct Tutorial {
    pub title: String,
    pub description: String,
    pub lessons: Vec<Lesson>,
    pub prerequisites: Vec<String>,
    pub estimated_time: std::time::Duration,
}

/// Individual lesson within a tutorial
#[derive(Debug, Clone)]
pub struct Lesson {
    pub title: String,
    pub content: String,
    pub code_examples: Vec<StarExample>,
    pub exercises: Vec<Exercise>,
}

/// Practice exercise
#[derive(Debug, Clone)]
pub struct Exercise {
    pub question: String,
    pub code_template: Option<String>,
    pub solution: String,
    pub hints: Vec<String>,
}

/// Best practice recommendation
#[derive(Debug, Clone)]
pub struct BestPractice {
    pub title: String,
    pub description: String,
    pub do_example: Option<String>,
    pub dont_example: Option<String>,
    pub rationale: String,
    pub category: PracticeCategory,
}

/// Difficulty levels for examples and tutorials
#[derive(Debug, Clone, PartialEq)]
pub enum Difficulty {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Categories for best practices
#[derive(Debug, Clone)]
pub enum PracticeCategory {
    Performance,
    Security,
    Maintainability,
    Correctness,
    Interoperability,
}

impl StarDocumentation {
    /// Create a new documentation system with default content
    pub fn new() -> Self {
        let mut docs = Self {
            examples: HashMap::new(),
            tutorials: HashMap::new(),
            best_practices: Vec::new(),
        };

        docs.load_default_content();
        docs
    }

    /// Get all available examples
    pub fn get_examples(&self) -> &HashMap<String, StarExample> {
        &self.examples
    }

    /// Get examples by difficulty level
    pub fn get_examples_by_difficulty(&self, difficulty: Difficulty) -> Vec<&StarExample> {
        self.examples
            .values()
            .filter(|example| example.difficulty == difficulty)
            .collect()
    }

    /// Get examples by tag
    pub fn get_examples_by_tag(&self, tag: &str) -> Vec<&StarExample> {
        self.examples
            .values()
            .filter(|example| example.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Get a specific example
    pub fn get_example(&self, key: &str) -> Option<&StarExample> {
        self.examples.get(key)
    }

    /// Get all tutorials
    pub fn get_tutorials(&self) -> &HashMap<String, Tutorial> {
        &self.tutorials
    }

    /// Get a specific tutorial
    pub fn get_tutorial(&self, key: &str) -> Option<&Tutorial> {
        self.tutorials.get(key)
    }

    /// Get best practices by category
    pub fn get_best_practices(&self, category: Option<PracticeCategory>) -> Vec<&BestPractice> {
        if let Some(_cat) = category {
            self.best_practices
                .iter()
                .filter(|practice| matches!(&practice.category, _cat))
                .collect()
        } else {
            self.best_practices.iter().collect()
        }
    }

    /// Generate markdown documentation
    pub fn generate_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str("# OxiRS-Star Documentation\n\n");
        md.push_str("Comprehensive guide to RDF-star implementation in Rust.\n\n");

        // Table of contents
        md.push_str("## Table of Contents\n\n");
        md.push_str("1. [Quick Start](#quick-start)\n");
        md.push_str("2. [Examples](#examples)\n");
        md.push_str("3. [Tutorials](#tutorials)\n");
        md.push_str("4. [Best Practices](#best-practices)\n");
        md.push_str("5. [API Reference](#api-reference)\n\n");

        // Quick start
        md.push_str("## Quick Start\n\n");
        md.push_str(&self.generate_quick_start_section());

        // Examples
        md.push_str("## Examples\n\n");
        md.push_str(&self.generate_examples_section());

        // Tutorials
        md.push_str("## Tutorials\n\n");
        md.push_str(&self.generate_tutorials_section());

        // Best practices
        md.push_str("## Best Practices\n\n");
        md.push_str(&self.generate_best_practices_section());

        md
    }

    /// Generate HTML documentation
    pub fn generate_html(&self) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>OxiRS-Star Documentation</title>\n");
        html.push_str("<style>\n");
        html.push_str(&self.get_css_styles());
        html.push_str("</style>\n</head>\n<body>\n");

        html.push_str("<div class=\"container\">\n");
        html.push_str("<h1>OxiRS-Star Documentation</h1>\n");

        // Navigation
        html.push_str("<nav class=\"nav\">\n");
        html.push_str("<a href=\"#examples\">Examples</a>\n");
        html.push_str("<a href=\"#tutorials\">Tutorials</a>\n");
        html.push_str("<a href=\"#best-practices\">Best Practices</a>\n");
        html.push_str("</nav>\n");

        // Examples section
        html.push_str("<section id=\"examples\">\n");
        html.push_str("<h2>Examples</h2>\n");
        for example in self.examples.values() {
            html.push_str("<div class=\"example\">\n");
            html.push_str(&format!("<h3>{}</h3>\n", example.title));
            html.push_str(&format!("<p>{}</p>\n", example.description));
            html.push_str(&format!(
                "<div class=\"difficulty {:?}\">Difficulty: {:?}</div>\n",
                example.difficulty, example.difficulty
            ));
            html.push_str(&format!("<pre><code>{}</code></pre>\n", example.code));
            if let Some(output) = &example.expected_output {
                html.push_str(&format!("<div class=\"output\"><strong>Expected Output:</strong><br><pre>{output}</pre></div>\n"));
            }
            html.push_str("</div>\n");
        }
        html.push_str("</section>\n");

        html.push_str("</div>\n</body>\n</html>");
        html
    }

    // Private methods for loading default content

    fn load_default_content(&mut self) {
        self.load_basic_examples();
        self.load_advanced_examples();
        self.load_tutorials();
        self.load_best_practices();
    }

    fn load_basic_examples(&mut self) {
        // Basic RDF-star triple creation
        self.examples.insert(
            "basic_quoted_triple".to_string(),
            StarExample {
                title: "Creating a Basic Quoted Triple".to_string(),
                description:
                    "Learn how to create a quoted triple and use it as a subject or object."
                        .to_string(),
                code: r#"use oxirs_star::{StarStore, StarTriple, StarTerm};

// Create a quoted triple
let quoted = StarTriple::new(
    StarTerm::iri("http://example.org/person1")?,
    StarTerm::iri("http://example.org/age")?,
    StarTerm::literal("25")?,
);

// Use the quoted triple as a subject
let meta_triple = StarTriple::new(
    StarTerm::quoted_triple(quoted),
    StarTerm::iri("http://example.org/certainty")?,
    StarTerm::literal("0.9")?,
);

let mut store = StarStore::new();
store.insert(&meta_triple)?;

println!("Stored {} triples", store.len());"#
                    .to_string(),
                expected_output: Some("Stored 1 triples".to_string()),
                difficulty: Difficulty::Beginner,
                tags: vec!["basic".to_string(), "quoted-triples".to_string()],
                related_concepts: vec!["RDF-star".to_string(), "meta-data".to_string()],
            },
        );

        // Parsing RDF-star
        self.examples.insert(
            "parsing_turtle_star".to_string(),
            StarExample {
                title: "Parsing Turtle-star Format".to_string(),
                description: "Parse RDF-star data from Turtle-star format.".to_string(),
                code: r#"use oxirs_star::parser::{StarParser, StarFormat};

let turtle_star = "@prefix ex: <http://example.org/> .\n\n<< ex:person1 ex:age 25 >> ex:certainty 0.9 .\n<< ex:person1 ex:name \"Alice\" >> ex:source ex:census2020 .";

let mut parser = StarParser::new();
let graph = parser.parse_str(turtle_star, StarFormat::TurtleStar)?;

println!("Parsed {} triples", graph.len());
for triple in &graph {
    println!("Triple: {}", triple);
}"#.to_string(),
                expected_output: Some(r#"Parsed 2 triples
Triple: << ex:person1 ex:age 25 >> ex:certainty 0.9
Triple: << ex:person1 ex:name "Alice" >> ex:source ex:census2020"#.to_string()),
                difficulty: Difficulty::Beginner,
                tags: vec!["parsing".to_string(), "turtle_star".to_string()],
                related_concepts: vec!["Turtle_star".to_string(), "parsing".to_string()],
            }
        );

        // Serialization
        self.examples.insert(
            "serializing_ntriples_star".to_string(),
            StarExample {
                title: "Serializing to N_Triples_star".to_string(),
                description: "Convert RDF-star data to N-Triples-star format.".to_string(),
                code: r#"use oxirs_star::serializer::{StarSerializer, SerializationOptions};
use oxirs_star::parser::{StarParser, StarFormat};

// Parse some data
let turtle_star = "<< <http://example.org/s> <http://example.org/p> \"object\" >> <http://example.org/certainty> \"0.8\" .";

let mut parser = StarParser::new();
let graph = parser.parse_str(turtle_star, StarFormat::TurtleStar)?;

// Serialize to N-Triples-star
let mut serializer = StarSerializer::new();
let options = SerializationOptions::default();
let ntriples_output = serializer.serialize_graph(&graph, StarFormat::NTriplesStar, &options)?;

println!("N-Triples-star output:\n{}", ntriples_output);"#.to_string(),
                expected_output: Some(r#"N-Triples-star output:
<< <http://example.org/s> <http://example.org/p> "object" >> <http://example.org/certainty> "0.8" ."#.to_string()),
                difficulty: Difficulty::Beginner,
                tags: vec!["serialization".to_string(), "ntriples-star".to_string()],
                related_concepts: vec!["N-Triples-star".to_string(), "serialization".to_string()],
            }
        );
    }

    fn load_advanced_examples(&mut self) {
        // Nested quoted triples
        self.examples.insert(
            "nested_quoted_triples".to_string(),
            StarExample {
                title: "Working with Nested Quoted Triples".to_string(),
                description: "Create and manipulate deeply nested quoted triples.".to_string(),
                code: r#"use oxirs_star::{StarStore, StarTriple, StarTerm};

// Create a base triple
let base_triple = StarTriple::new(
    StarTerm::iri("http://example.org/alice")?,
    StarTerm::iri("http://example.org/age")?,
    StarTerm::literal("30")?,
);

// Create a quoted triple about the base triple
let meta_triple = StarTriple::new(
    StarTerm::quoted_triple(base_triple),
    StarTerm::iri("http://example.org/certainty")?,
    StarTerm::literal("0.9")?,
);

// Create another level of nesting
let meta_meta_triple = StarTriple::new(
    StarTerm::quoted_triple(meta_triple),
    StarTerm::iri("http://example.org/source")?,
    StarTerm::iri("http://example.org/study2023")?,
);

let mut store = StarStore::new();
store.insert(&meta_meta_triple)?;

println!("Created triple with nesting depth: {}", store.max_nesting_depth());"#
                    .to_string(),
                expected_output: Some("Created triple with nesting depth: 2".to_string()),
                difficulty: Difficulty::Advanced,
                tags: vec!["nested".to_string(), "advanced".to_string()],
                related_concepts: vec!["nesting".to_string(), "recursion".to_string()],
            },
        );

        // Performance optimization
        self.examples.insert(
            "performance_optimization".to_string(),
            StarExample {
                title: "Performance Optimization Techniques".to_string(),
                description:
                    "Optimize parsing and querying performance for large RDF-star datasets."
                        .to_string(),
                code: r#"use oxirs_star::{StarStore, StarConfig};
use oxirs_star::parser::StarParser;
use oxirs_star::profiling::StarProfiler;

// Configure for performance
let config = StarConfig {
    enable_indexing: true,
    index_quoted_triples: true,
    cache_size: 10000,
    parallel_parsing: true,
    ..Default::default()
};

let mut store = StarStore::with_config(config);
let mut profiler = StarProfiler::new();

// Profile parsing operation
let large_dataset = generate_large_dataset(10000); // Helper function
let parsed_graph = profiler.profile_parsing(
    StarFormat::TurtleStar,
    large_dataset.len(),
    || {
        let mut parser = StarParser::new();
        parser.parse_str(&large_dataset, StarFormat::TurtleStar).unwrap()
    }
);

// Profile insertion
profiler.start_operation("bulk_insert");
for triple in parsed_graph {
    store.insert(&triple)?;
}
profiler.end_operation();

let report = profiler.generate_report();
println!("Performance report: {:#?}", report);"#
                    .to_string(),
                expected_output: None,
                difficulty: Difficulty::Expert,
                tags: vec![
                    "performance".to_string(),
                    "profiling".to_string(),
                    "optimization".to_string(),
                ],
                related_concepts: vec![
                    "profiling".to_string(),
                    "indexing".to_string(),
                    "performance".to_string(),
                ],
            },
        );
    }

    fn load_tutorials(&mut self) {
        // Beginner tutorial
        let beginner_tutorial = Tutorial {
            title: "Introduction to RDF-star".to_string(),
            description: "Learn the basics of RDF-star and how to use it for metadata annotation.".to_string(),
            estimated_time: std::time::Duration::from_secs(3600), // 1 hour
            prerequisites: vec!["Basic knowledge of RDF".to_string(), "Rust programming".to_string()],
            lessons: vec![
                Lesson {
                    title: "What is RDF-star?".to_string(),
                    content: r#"
RDF-star is an extension to RDF that allows triples to be used as subjects or objects
in other triples. This enables direct annotation of statements with metadata such as:

- Certainty scores
- Provenance information
- Temporal validity
- Source attribution

The key concept is the "quoted triple" - a triple that is referenced but not asserted.
                    "#.to_string(),
                    code_examples: vec![
                        self.examples.get("basic_quoted_triple").unwrap().clone()
                    ],
                    exercises: vec![
                        Exercise {
                            question: "Create a quoted triple representing 'John knows Mary' and annotate it with a confidence score of 0.85".to_string(),
                            code_template: Some(r#"
let base_triple = StarTriple::new(
    StarTerm::iri("http://example.org/john")?,
    // Fill in the predicate
    // Fill in the object
);

let confidence_triple = StarTriple::new(
    // Use the base triple as subject
    // Add certainty predicate
    // Add confidence value
);
                            "#.to_string()),
                            solution: r#"
let base_triple = StarTriple::new(
    StarTerm::iri("http://example.org/john")?,
    StarTerm::iri("http://example.org/knows")?,
    StarTerm::iri("http://example.org/mary")?,
);

let confidence_triple = StarTriple::new(
    StarTerm::quoted_triple(base_triple),
    StarTerm::iri("http://example.org/certainty")?,
    StarTerm::literal("0.85")?,
);
                            "#.to_string(),
                            hints: vec![
                                "Use StarTerm::quoted_triple() to reference the base triple".to_string(),
                                "Confidence scores are typically represented as literals".to_string(),
                            ],
                        }
                    ],
                },
                Lesson {
                    title: "Parsing RDF-star Formats".to_string(),
                    content: r#"
RDF-star supports several serialization formats:

1. **Turtle-star (.ttls)** - Extended Turtle with quoted triple syntax
2. **N-Triples-star (.nts)** - Line-based format for streaming
3. **TriG-star (.trigs)** - Named graph support
4. **N-Quads-star (.nqs)** - Quad-based format

The syntax uses << >> to denote quoted triples.
                    "#.to_string(),
                    code_examples: vec![
                        self.examples.get("parsing_turtle_star").unwrap().clone()
                    ],
                    exercises: vec![],
                },
            ],
        };

        self.tutorials
            .insert("beginner".to_string(), beginner_tutorial);
    }

    fn load_best_practices(&mut self) {
        // Performance best practices
        self.best_practices.push(BestPractice {
            title: "Use Appropriate Index Settings".to_string(),
            description: "Configure indexing based on your query patterns to optimize performance.".to_string(),
            do_example: Some(r#"
// Good: Enable quoted triple indexing when needed
let config = StarConfig {
    enable_indexing: true,
    index_quoted_triples: true,
    ..Default::default()
};
let store = StarStore::with_config(config);
            "#.to_string()),
            dont_example: Some(r#"
// Bad: Default config may not be optimal for your use case
let store = StarStore::new(); // Uses default indexing
            "#.to_string()),
            rationale: "Proper indexing dramatically improves query performance, especially for complex quoted triple patterns.".to_string(),
            category: PracticeCategory::Performance,
        });

        // Security best practices
        self.best_practices.push(BestPractice {
            title: "Validate Input Data".to_string(),
            description: "Always validate RDF-star input to prevent malformed data from corrupting your store.".to_string(),
            do_example: Some(r#"
// Good: Use strict parsing with error handling
let mut parser = StarParser::new();
parser.set_strict_mode(true);

match parser.parse_str(input, format) {
    Ok(graph) => process_graph(graph),
    Err(e) => {
        log::error!("Invalid RDF-star input: {}", e);
        return Err(ValidationError::InvalidInput);
    }
}
            "#.to_string()),
            dont_example: Some(r#"
// Bad: Blindly accepting any input
let graph = parser.parse_str(input, format).unwrap();
process_graph(graph);
            "#.to_string()),
            rationale: "Malformed RDF-star can lead to inconsistent state, performance issues, or security vulnerabilities.".to_string(),
            category: PracticeCategory::Security,
        });

        // Maintainability best practices
        self.best_practices.push(BestPractice {
            title: "Use Meaningful IRIs and Prefixes".to_string(),
            description: "Choose clear, consistent naming conventions for your RDF-star vocabulary.".to_string(),
            do_example: Some(r#"
// Good: Clear, consistent vocabulary
@prefix meta: <http://example.org/metadata/> .
@prefix trust: <http://example.org/trust/> .

<< :person1 :age 25 >> meta:confidence trust:high .
<< :person1 :age 25 >> meta:source :census2020 .
            "#.to_string()),
            dont_example: Some(r#"
// Bad: Unclear, inconsistent naming
<< :p1 :a 25 >> :c :h .
<< :p1 :a 25 >> :s :c20 .
            "#.to_string()),
            rationale: "Clear naming makes RDF-star data self-documenting and easier to maintain and debug.".to_string(),
            category: PracticeCategory::Maintainability,
        });
    }

    // Helper methods for documentation generation

    fn generate_quick_start_section(&self) -> String {
        format!(
            r#"
Get started with RDF-star in just a few lines of code:

```rust
{}
```

This example shows the core concepts of RDF-star: creating quoted triples and using them to annotate statements with metadata.
"#,
            self.examples
                .get("basic_quoted_triple")
                .map(|e| &e.code)
                .unwrap_or(&"".to_string())
        )
    }

    fn generate_examples_section(&self) -> String {
        let mut section = String::new();

        for difficulty in [
            Difficulty::Beginner,
            Difficulty::Intermediate,
            Difficulty::Advanced,
            Difficulty::Expert,
        ] {
            let examples = self.get_examples_by_difficulty(difficulty.clone());
            if !examples.is_empty() {
                section.push_str(&format!("### {difficulty:?} Examples\n\n"));

                for example in examples {
                    section.push_str(&format!("#### {}\n\n", example.title));
                    section.push_str(&format!("{}\n\n", example.description));
                    section.push_str(&format!("```rust\n{}\n```\n\n", example.code));

                    if let Some(output) = &example.expected_output {
                        section
                            .push_str(&format!("**Expected Output:**\n```\n{output}\n```\n\n"));
                    }
                }
            }
        }

        section
    }

    fn generate_tutorials_section(&self) -> String {
        let mut section = String::new();

        for tutorial in self.tutorials.values() {
            section.push_str(&format!("### {}\n\n", tutorial.title));
            section.push_str(&format!("{}\n\n", tutorial.description));
            section.push_str(&format!(
                "**Estimated Time:** {:?}\n\n",
                tutorial.estimated_time
            ));

            if !tutorial.prerequisites.is_empty() {
                section.push_str("**Prerequisites:**\n");
                for prereq in &tutorial.prerequisites {
                    section.push_str(&format!("- {prereq}\n"));
                }
                section.push('\n');
            }

            section.push_str("**Lessons:**\n");
            for (i, lesson) in tutorial.lessons.iter().enumerate() {
                section.push_str(&format!("{}. {}\n", i + 1, lesson.title));
            }
            section.push('\n');
        }

        section
    }

    fn generate_best_practices_section(&self) -> String {
        let mut section = String::new();

        for category in [
            PracticeCategory::Performance,
            PracticeCategory::Security,
            PracticeCategory::Maintainability,
            PracticeCategory::Correctness,
            PracticeCategory::Interoperability,
        ] {
            let practices = self.get_best_practices(Some(category.clone()));
            if !practices.is_empty() {
                section.push_str(&format!("### {category:?} Best Practices\n\n"));

                for practice in practices {
                    section.push_str(&format!("#### {}\n\n", practice.title));
                    section.push_str(&format!("{}\n\n", practice.description));

                    if let Some(do_example) = &practice.do_example {
                        section.push_str("**Do:**\n");
                        section.push_str(&format!("```rust\n{do_example}\n```\n\n"));
                    }

                    if let Some(dont_example) = &practice.dont_example {
                        section.push_str("**Don't:**\n");
                        section.push_str(&format!("```rust\n{dont_example}\n```\n\n"));
                    }

                    section.push_str(&format!("**Rationale:** {}\n\n", practice.rationale));
                }
            }
        }

        section
    }

    fn get_css_styles(&self) -> String {
        r#"
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.container {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    padding: 40px;
}

.nav {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 6px;
    margin-bottom: 30px;
}

.nav a {
    margin-right: 20px;
    text-decoration: none;
    color: #007bff;
    font-weight: 500;
}

.nav a:hover {
    text-decoration: underline;
}

.example {
    border: 1px solid #e9ecef;
    border-radius: 6px;
    padding: 20px;
    margin-bottom: 20px;
    background: #f8f9fa;
}

.difficulty {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: bold;
    margin-bottom: 15px;
}

.difficulty.Beginner { background: #d4edda; color: #155724; }
.difficulty.Intermediate { background: #fff3cd; color: #856404; }
.difficulty.Advanced { background: #f8d7da; color: #721c24; }
.difficulty.Expert { background: #d1ecf1; color: #0c5460; }

pre {
    background: #f1f3f4;
    padding: 15px;
    border-radius: 4px;
    overflow-x: auto;
    font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
}

code {
    font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
    font-size: 0.9em;
}

.output {
    background: #e8f5e8;
    border: 1px solid #c3e6c3;
    border-radius: 4px;
    padding: 10px;
    margin-top: 15px;
}

h1, h2, h3 {
    color: #2c3e50;
}
        "#
        .to_string()
    }
}

impl Default for StarDocumentation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_documentation_creation() {
        let docs = StarDocumentation::new();
        assert!(!docs.examples.is_empty());
        assert!(!docs.tutorials.is_empty());
        assert!(!docs.best_practices.is_empty());
    }

    #[test]
    fn test_example_filtering() {
        let docs = StarDocumentation::new();

        let beginner_examples = docs.get_examples_by_difficulty(Difficulty::Beginner);
        assert!(!beginner_examples.is_empty());

        let parsing_examples = docs.get_examples_by_tag("parsing");
        assert!(!parsing_examples.is_empty());
    }

    #[test]
    fn test_markdown_generation() {
        let docs = StarDocumentation::new();
        let markdown = docs.generate_markdown();

        assert!(markdown.contains("# OxiRS-Star Documentation"));
        assert!(markdown.contains("## Examples"));
        assert!(markdown.contains("## Tutorials"));
        assert!(markdown.contains("## Best Practices"));
    }

    #[test]
    fn test_html_generation() {
        let docs = StarDocumentation::new();
        let html = docs.generate_html();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<title>OxiRS-Star Documentation</title>"));
        assert!(html.contains("class=\"example\""));
    }

    #[test]
    fn test_best_practices_categorization() {
        let docs = StarDocumentation::new();

        let performance_practices = docs.get_best_practices(Some(PracticeCategory::Performance));
        assert!(!performance_practices.is_empty());

        let security_practices = docs.get_best_practices(Some(PracticeCategory::Security));
        assert!(!security_practices.is_empty());
    }
}
