//! Interactive Constraint Tester CLI Tool
//!
//! A REPL-style interactive tool for testing SHACL constraints.
//! Allows developers to experiment with shapes, constraints, and validation
//! in an interactive environment.

use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use oxirs_core::{ConcreteStore, NamedNode};
use oxirs_shacl::{
    constraints::Constraint, validation::ValidationEngine, ConstraintComponentId, Severity, Shape,
    ShapeId, ShapeType, ValidationConfig,
};

/// Interactive testing session
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestingSession {
    /// Current working shapes
    shapes: HashMap<String, Shape>,

    /// Test data store (serialized)
    test_data: Vec<String>,

    /// Session history
    history: Vec<String>,

    /// Session name
    name: String,

    /// Last validation results
    last_results: Option<String>,
}

impl TestingSession {
    fn new(name: String) -> Self {
        Self {
            shapes: HashMap::new(),
            test_data: Vec::new(),
            history: Vec::new(),
            name,
            last_results: None,
        }
    }

    fn add_command(&mut self, cmd: String) {
        self.history.push(cmd);
    }

    fn save(&self, path: &PathBuf) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    fn load(path: &PathBuf) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let session: TestingSession = serde_json::from_str(&json)?;
        Ok(session)
    }
}

/// Interactive constraint tester
struct ConstraintTester {
    session: TestingSession,
    store: ConcreteStore,
    verbose: bool,
}

impl ConstraintTester {
    fn new(session_name: String, verbose: bool) -> Result<Self> {
        Ok(Self {
            session: TestingSession::new(session_name),
            store: ConcreteStore::new().map_err(|e| anyhow!("Failed to create store: {}", e))?,
            verbose,
        })
    }

    fn run(&mut self) -> Result<()> {
        println!("🧪 OxiRS SHACL Interactive Constraint Tester");
        println!("============================================");
        println!("Session: {}", self.session.name);
        println!("Type 'help' for available commands");
        println!();

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            print!("shacl> ");
            stdout.flush()?;

            let mut line = String::new();
            stdin.lock().read_line(&mut line)?;

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            self.session.add_command(line.to_string());

            match self.process_command(line) {
                Ok(should_exit) => {
                    if should_exit {
                        println!("Exiting interactive tester.");
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("❌ Error: {}", e);
                }
            }
        }

        Ok(())
    }

    fn process_command(&mut self, line: &str) -> Result<bool> {
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.is_empty() {
            return Ok(false);
        }

        match parts[0] {
            "help" | "h" | "?" => {
                self.show_help();
                Ok(false)
            }
            "exit" | "quit" | "q" => Ok(true),
            "create" | "new" => {
                self.create_shape(&parts[1..])?;
                Ok(false)
            }
            "add" => {
                self.add_constraint(&parts[1..])?;
                Ok(false)
            }
            "test" => {
                self.test_constraint(&parts[1..])?;
                Ok(false)
            }
            "validate" => {
                self.validate_all()?;
                Ok(false)
            }
            "list" | "ls" => {
                self.list_shapes();
                Ok(false)
            }
            "show" => {
                self.show_shape(&parts[1..])?;
                Ok(false)
            }
            "data" => {
                self.add_test_data(&parts[1..])?;
                Ok(false)
            }
            "clear" => {
                self.clear_data()?;
                Ok(false)
            }
            "save" => {
                self.save_session(&parts[1..])?;
                Ok(false)
            }
            "load" => {
                self.load_session(&parts[1..])?;
                Ok(false)
            }
            "examples" => {
                self.show_examples(&parts[1..])?;
                Ok(false)
            }
            "suggest" => {
                self.suggest_constraints(&parts[1..])?;
                Ok(false)
            }
            "history" => {
                self.show_history();
                Ok(false)
            }
            "delete" | "remove" | "rm" => {
                self.delete_shape(&parts[1..])?;
                Ok(false)
            }
            _ => {
                eprintln!(
                    "Unknown command: {}. Type 'help' for available commands.",
                    parts[0]
                );
                Ok(false)
            }
        }
    }

    fn show_help(&self) {
        println!("Available commands:");
        println!();
        println!("Shape Management:");
        println!("  create <name> [node|property]  - Create a new shape");
        println!("  add <shape> <constraint>       - Add constraint to shape");
        println!("  list                           - List all shapes");
        println!("  show <shape>                   - Show shape details");
        println!("  delete <shape>                 - Delete a shape");
        println!();
        println!("Testing:");
        println!("  test <shape> <data>            - Test a specific constraint");
        println!("  validate                       - Validate all test data");
        println!("  data <turtle>                  - Add test data (Turtle format)");
        println!("  clear                          - Clear test data");
        println!();
        println!("Session:");
        println!("  save <file>                    - Save session to file");
        println!("  load <file>                    - Load session from file");
        println!("  history                        - Show command history");
        println!();
        println!("Help:");
        println!("  examples [constraint]          - Show constraint examples");
        println!("  suggest <domain>               - Suggest constraints for domain");
        println!("  help                           - Show this help message");
        println!("  exit                           - Exit the tester");
        println!();
    }

    fn create_shape(&mut self, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(anyhow!("Usage: create <name> [node|property]"));
        }

        let name = args[0];
        let shape_type = if args.len() > 1 {
            match args[1] {
                "node" => ShapeType::NodeShape,
                "property" => ShapeType::PropertyShape,
                _ => return Err(anyhow!("Invalid shape type. Use 'node' or 'property'")),
            }
        } else {
            ShapeType::NodeShape
        };

        let shape_id = ShapeId::new(format!("http://example.org/shapes#{}", name));
        let shape = Shape::new(shape_id.clone(), shape_type);

        self.session.shapes.insert(name.to_string(), shape);
        println!("✅ Created shape: {}", name);

        Ok(())
    }

    fn add_constraint(&mut self, args: &[&str]) -> Result<()> {
        if args.len() < 2 {
            return Err(anyhow!(
                "Usage: add <shape> <constraint_type> [parameters...]"
            ));
        }

        let shape_name = args[0];
        let constraint_type = args[1];

        let shape = self
            .session
            .shapes
            .get_mut(shape_name)
            .ok_or_else(|| anyhow!("Shape '{}' not found", shape_name))?;

        // Parse constraint based on type
        let (component_id, constraint) = match constraint_type {
            "minCount" => {
                let count = args
                    .get(2)
                    .ok_or_else(|| anyhow!("minCount requires a count parameter"))?
                    .parse::<u32>()
                    .context("Invalid count value")?;

                use oxirs_shacl::constraints::cardinality_constraints::MinCountConstraint;
                (
                    ConstraintComponentId::new("sh:minCount"),
                    Constraint::MinCount(MinCountConstraint { min_count: count }),
                )
            }
            "maxCount" => {
                let count = args
                    .get(2)
                    .ok_or_else(|| anyhow!("maxCount requires a count parameter"))?
                    .parse::<u32>()
                    .context("Invalid count value")?;

                use oxirs_shacl::constraints::cardinality_constraints::MaxCountConstraint;
                (
                    ConstraintComponentId::new("sh:maxCount"),
                    Constraint::MaxCount(MaxCountConstraint { max_count: count }),
                )
            }
            "datatype" => {
                let datatype = args
                    .get(2)
                    .ok_or_else(|| anyhow!("datatype requires a datatype IRI"))?;

                let datatype_node =
                    NamedNode::new(*datatype).map_err(|e| anyhow!("Invalid IRI: {}", e))?;

                use oxirs_shacl::constraints::value_constraints::DatatypeConstraint;
                (
                    ConstraintComponentId::new("sh:datatype"),
                    Constraint::Datatype(DatatypeConstraint {
                        datatype_iri: datatype_node,
                    }),
                )
            }
            "pattern" => {
                let pattern = args
                    .get(2)
                    .ok_or_else(|| anyhow!("pattern requires a regex pattern"))?;

                use oxirs_shacl::constraints::string_constraints::PatternConstraint;
                (
                    ConstraintComponentId::new("sh:pattern"),
                    Constraint::Pattern(PatternConstraint {
                        pattern: pattern.to_string(),
                        flags: None,
                        message: None,
                    }),
                )
            }
            "minLength" => {
                let length = args
                    .get(2)
                    .ok_or_else(|| anyhow!("minLength requires a length parameter"))?
                    .parse::<u32>()
                    .context("Invalid length value")?;

                use oxirs_shacl::constraints::string_constraints::MinLengthConstraint;
                (
                    ConstraintComponentId::new("sh:minLength"),
                    Constraint::MinLength(MinLengthConstraint { min_length: length }),
                )
            }
            "maxLength" => {
                let length = args
                    .get(2)
                    .ok_or_else(|| anyhow!("maxLength requires a length parameter"))?
                    .parse::<u32>()
                    .context("Invalid length value")?;

                use oxirs_shacl::constraints::string_constraints::MaxLengthConstraint;
                (
                    ConstraintComponentId::new("sh:maxLength"),
                    Constraint::MaxLength(MaxLengthConstraint { max_length: length }),
                )
            }
            "class" => {
                let class_iri = args
                    .get(2)
                    .ok_or_else(|| anyhow!("class requires a class IRI"))?;

                let class_node =
                    NamedNode::new(*class_iri).map_err(|e| anyhow!("Invalid IRI: {}", e))?;

                use oxirs_shacl::constraints::value_constraints::ClassConstraint;
                (
                    ConstraintComponentId::new("sh:class"),
                    Constraint::Class(ClassConstraint {
                        class_iri: class_node,
                    }),
                )
            }
            _ => {
                return Err(anyhow!("Unknown constraint type: {}", constraint_type));
            }
        };

        shape.add_constraint(component_id, constraint);
        println!(
            "✅ Added {} constraint to shape '{}'",
            constraint_type, shape_name
        );

        Ok(())
    }

    fn test_constraint(&mut self, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(anyhow!("Usage: test <shape>"));
        }

        let shape_name = args[0];
        let shape = self
            .session
            .shapes
            .get(shape_name)
            .ok_or_else(|| anyhow!("Shape '{}' not found", shape_name))?
            .clone();

        // Create a validation engine with just this shape
        let mut shapes = IndexMap::new();
        shapes.insert(shape.id.clone(), shape.clone());

        let config = ValidationConfig::default();
        let mut engine = ValidationEngine::new(&shapes, config);

        // Run validation
        let report = engine
            .validate_store(&self.store)
            .context("Validation failed")?;

        // Display results
        println!();
        println!("Validation Results for '{}':", shape_name);
        println!("─────────────────────────────");

        if report.conforms {
            println!("✅ CONFORMS - No violations found");
        } else {
            println!("❌ DOES NOT CONFORM");
            println!();
            println!("Violations ({}):", report.violations.len());

            for (idx, violation) in report.violations.iter().enumerate() {
                println!();
                println!("  {}. Severity: {:?}", idx + 1, violation.result_severity);
                if let Some(ref msg) = &violation.result_message {
                    println!("     Message: {}", msg);
                }
                println!("     Focus: {}", violation.focus_node);
                if let Some(ref value) = violation.value {
                    println!("     Value: {}", value);
                }
            }
        }
        println!();

        // Save results
        let results_json = serde_json::to_string_pretty(&report)?;
        self.session.last_results = Some(results_json);

        Ok(())
    }

    fn validate_all(&mut self) -> Result<()> {
        if self.session.shapes.is_empty() {
            println!("⚠️  No shapes defined. Use 'create' to define shapes first.");
            return Ok(());
        }

        let shapes: IndexMap<ShapeId, Shape> = self
            .session
            .shapes
            .values()
            .map(|s| (s.id.clone(), s.clone()))
            .collect();

        let config = ValidationConfig::default();
        let mut engine = ValidationEngine::new(&shapes, config);

        let report = engine
            .validate_store(&self.store)
            .context("Validation failed")?;

        println!();
        println!("Full Validation Results:");
        println!("════════════════════════");
        println!("Conforms: {}", report.conforms);
        println!("Total violations: {}", report.violations.len());
        println!();

        if !report.violations.is_empty() {
            // Group by severity
            let mut by_severity: HashMap<Severity, Vec<_>> = HashMap::new();
            for v in &report.violations {
                by_severity.entry(v.result_severity).or_default().push(v);
            }

            for severity in [Severity::Violation, Severity::Warning, Severity::Info] {
                if let Some(viols) = by_severity.get(&severity) {
                    println!("{} ({}):", severity, viols.len());
                    for v in viols {
                        print!("  • ");
                        if let Some(ref msg) = &v.result_message {
                            print!("{}", msg);
                        }
                        print!(" [{}]", v.focus_node);
                        println!();
                    }
                    println!();
                }
            }
        }

        Ok(())
    }

    fn list_shapes(&self) {
        if self.session.shapes.is_empty() {
            println!("No shapes defined.");
            return;
        }

        println!();
        println!("Defined Shapes:");
        println!("───────────────");

        for (name, shape) in &self.session.shapes {
            let shape_type = match shape.shape_type {
                ShapeType::NodeShape => "Node",
                ShapeType::PropertyShape => "Property",
            };

            println!("  • {} ({} shape)", name, shape_type);
            println!("    Constraints: {}", shape.constraints.len());

            if self.verbose {
                for (comp_id, _) in &shape.constraints {
                    println!("      - {}", comp_id.as_str());
                }
            }
        }
        println!();
    }

    fn show_shape(&self, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(anyhow!("Usage: show <shape>"));
        }

        let shape_name = args[0];
        let shape = self
            .session
            .shapes
            .get(shape_name)
            .ok_or_else(|| anyhow!("Shape '{}' not found", shape_name))?;

        println!();
        println!("Shape: {}", shape_name);
        println!("══════════════════════════");
        println!("ID: {}", shape.id);
        println!("Type: {:?}", shape.shape_type);
        println!("Severity: {:?}", shape.severity);
        println!("Active: {}", !shape.deactivated);
        println!();

        if !shape.constraints.is_empty() {
            println!("Constraints:");
            for (comp_id, constraint) in &shape.constraints {
                println!("  • {}", comp_id.as_str());
                println!("    {:?}", constraint);
            }
        } else {
            println!("No constraints defined.");
        }
        println!();

        Ok(())
    }

    fn add_test_data(&mut self, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(anyhow!("Usage: data <turtle-triple>"));
        }

        let turtle_data = args.join(" ");

        // Try to parse as Turtle (simplified - in production use proper parser)
        self.session.test_data.push(turtle_data.clone());

        println!("✅ Added test data (stored for session)");
        println!("Note: Full Turtle parsing requires loading from file");

        Ok(())
    }

    fn clear_data(&mut self) -> Result<()> {
        self.store =
            ConcreteStore::new().map_err(|e| anyhow!("Failed to create new store: {}", e))?;
        self.session.test_data.clear();
        println!("✅ Cleared all test data");
        Ok(())
    }

    fn save_session(&self, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(anyhow!("Usage: save <filename>"));
        }

        let path = PathBuf::from(args[0]);
        self.session.save(&path)?;

        println!("✅ Saved session to: {:?}", path);
        Ok(())
    }

    fn load_session(&mut self, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(anyhow!("Usage: load <filename>"));
        }

        let path = PathBuf::from(args[0]);
        self.session = TestingSession::load(&path)?;

        println!("✅ Loaded session: {}", self.session.name);
        println!("   Shapes: {}", self.session.shapes.len());
        println!("   History entries: {}", self.session.history.len());

        Ok(())
    }

    fn show_examples(&self, args: &[&str]) -> Result<()> {
        let constraint_type = args.first().copied().unwrap_or("all");

        println!();
        println!("SHACL Constraint Examples:");
        println!("══════════════════════════");
        println!();

        if constraint_type == "all" || constraint_type == "minCount" {
            println!("minCount - Minimum cardinality");
            println!("  add myShape minCount 1");
            println!("  Ensures at least 1 value exists");
            println!();
        }
        if constraint_type == "all" || constraint_type == "maxCount" {
            println!("maxCount - Maximum cardinality");
            println!("  add myShape maxCount 5");
            println!("  Ensures at most 5 values exist");
            println!();
        }
        if constraint_type == "all" || constraint_type == "datatype" {
            println!("datatype - RDF datatype constraint");
            println!("  add myShape datatype http://www.w3.org/2001/XMLSchema#string");
            println!("  add myShape datatype http://www.w3.org/2001/XMLSchema#integer");
            println!("  Ensures values have the specified datatype");
            println!();
        }
        if constraint_type == "all" || constraint_type == "pattern" {
            println!("pattern - Regular expression pattern");
            println!("  add myShape pattern '^[A-Z].*'");
            println!("  add myShape pattern '[0-9]{{3}}-[0-9]{{4}}'");
            println!("  Validates string values against regex");
            println!();
        }
        if constraint_type == "all"
            || constraint_type == "minLength"
            || constraint_type == "maxLength"
        {
            println!("minLength / maxLength - String length constraints");
            println!("  add myShape minLength 3");
            println!("  add myShape maxLength 100");
            println!("  Validates string length");
            println!();
        }
        if constraint_type == "all" || constraint_type == "class" {
            println!("class - RDF class constraint");
            println!("  add myShape class http://xmlns.com/foaf/0.1/Person");
            println!("  Ensures values are instances of specified class");
            println!();
        }

        if constraint_type != "all"
            && ![
                "minCount",
                "maxCount",
                "datatype",
                "pattern",
                "minLength",
                "maxLength",
                "class",
            ]
            .contains(&constraint_type)
        {
            println!("Unknown constraint type: {}", constraint_type);
            println!("Try: minCount, maxCount, datatype, pattern, class");
        }

        Ok(())
    }

    fn suggest_constraints(&self, args: &[&str]) -> Result<()> {
        let domain = args.first().copied().unwrap_or("general");

        println!();
        println!("Constraint Suggestions for: {}", domain);
        println!("═══════════════════════════════");
        println!();

        match domain {
            "person" | "foaf" => {
                println!("For Person/FOAF data:");
                println!("  • minCount 1 maxCount 1  - for names, emails");
                println!("  • pattern for email: ^[^@]+@[^@]+\\.[^@]+$");
                println!("  • datatype xsd:string    - for text fields");
                println!("  • class foaf:Person      - for person nodes");
                println!();
            }
            "schema" | "schema.org" => {
                println!("For Schema.org data:");
                println!("  • minCount 1             - for required properties");
                println!("  • datatype xsd:dateTime  - for dates");
                println!("  • datatype xsd:decimal   - for prices");
                println!("  • class schema:Product   - for products");
                println!();
            }
            "general" => {
                println!("General best practices:");
                println!("  • Use minCount/maxCount for cardinality");
                println!("  • Use datatype for type safety");
                println!("  • Use pattern for format validation");
                println!("  • Use class for semantic types");
                println!("  • Start simple, add constraints iteratively");
                println!();
            }
            _ => {
                println!("Unknown domain: {}", domain);
                println!("Try: person, schema, general");
            }
        }

        Ok(())
    }

    fn show_history(&self) {
        println!();
        println!("Command History:");
        println!("════════════════");

        if self.session.history.is_empty() {
            println!("No commands in history.");
        } else {
            for (idx, cmd) in self.session.history.iter().enumerate() {
                println!("  {:3}. {}", idx + 1, cmd);
            }
        }
        println!();
    }

    fn delete_shape(&mut self, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(anyhow!("Usage: delete <shape>"));
        }

        let shape_name = args[0];

        if self.session.shapes.remove(shape_name).is_some() {
            println!("✅ Deleted shape: {}", shape_name);
        } else {
            println!("⚠️  Shape '{}' not found", shape_name);
        }

        Ok(())
    }
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let session_name = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "default".to_string());

    let verbose = args.contains(&"--verbose".to_string()) || args.contains(&"-v".to_string());

    // Create and run tester
    let mut tester = ConstraintTester::new(session_name, verbose)?;
    tester.run()?;

    Ok(())
}
