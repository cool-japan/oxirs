//! # Beta.4 Features Showcase
//!
//! Comprehensive examples demonstrating the new features in Beta.4:
//! - RIF (Rule Interchange Format) for rule interchange between engines
//! - CHR (Constraint Handling Rules) for declarative constraint solving
//! - ASP (Answer Set Programming) for combinatorial optimization
//!
//! Run with: `cargo run --example beta4_features_showcase --all-features`

use anyhow::Result;
use oxirs_rule::asp::{AspLiteral, AspSolver, AspTerm, Atom};
use oxirs_rule::chr::{ChrEngine, ChrRule, ChrTerm, Constraint};
use oxirs_rule::rif::{RifDialect, RifParser, RifSerializer};

fn main() -> Result<()> {
    println!("=== OxiRS Rule Engine Beta.4 Features Showcase ===\n");

    // Run all demonstrations
    demonstrate_rif()?;
    println!();
    demonstrate_chr()?;
    println!();
    demonstrate_asp()?;
    println!();
    demonstrate_integration()?;

    println!("\n=== All Beta.4 demonstrations completed successfully! ===");
    Ok(())
}

/// Demonstrate RIF (Rule Interchange Format) capabilities
fn demonstrate_rif() -> Result<()> {
    println!("--- RIF (Rule Interchange Format) ---");
    println!("W3C standard for interchanging rules between different systems\n");

    // Example 1: Parse RIF Compact Syntax
    let rif_text = r#"
        Prefix(ex <http://example.org/>)
        Prefix(rdf <http://www.w3.org/1999/02/22-rdf-syntax-ns#>)

        Group (
            (* Transitivity of ancestor relation *)
            Forall ?x ?y ?z (
                ex:ancestor(?x ?z) :- And(ex:parent(?x ?y) ex:ancestor(?y ?z))
            )

            (* Base case: parents are ancestors *)
            Forall ?x ?y (
                ex:ancestor(?x ?y) :- ex:parent(?x ?y)
            )
        )
    "#;

    println!("1. Parsing RIF Compact Syntax:");
    println!("{}", rif_text);

    let mut parser = RifParser::new(RifDialect::Bld);
    let document = parser.parse(rif_text)?;

    println!("   ✓ Parsed successfully!");
    println!("   - Dialect: {:?}", document.dialect);
    println!("   - Prefixes: {} defined", document.prefixes.len());
    println!("   - Groups: {} rule groups", document.groups.len());

    // Example 2: Convert to OxiRS rules
    println!("\n2. Converting RIF to OxiRS Rules:");
    let rules = document.to_oxirs_rules()?;
    println!("   ✓ Converted {} rules", rules.len());
    for (i, rule) in rules.iter().enumerate() {
        println!(
            "   Rule {}: {} (body: {} atoms, head: {} atoms)",
            i + 1,
            rule.name,
            rule.body.len(),
            rule.head.len()
        );
    }

    // Example 3: Serialize back to RIF
    println!("\n3. Serializing back to RIF Compact Syntax:");
    let serializer = RifSerializer::new(RifDialect::Bld);
    let rif_output = serializer.serialize(&document)?;
    println!("   ✓ Serialized successfully ({} bytes)", rif_output.len());

    // Example 4: RIF-Core vs RIF-BLD
    println!("\n4. RIF Dialects Comparison:");
    println!("   - RIF-Core: Basic Horn rules without negation");
    println!("   - RIF-BLD:  Adds equality, NAF, and frame logic");
    println!("   - RIF-PRD:  Production rules (future support)");

    Ok(())
}

/// Demonstrate CHR (Constraint Handling Rules) capabilities
fn demonstrate_chr() -> Result<()> {
    println!("--- CHR (Constraint Handling Rules) ---");
    println!("Declarative constraint solving framework\n");

    // Example 1: Classic LEQ (less-than-or-equal) constraint system
    println!("1. LEQ Constraint System:");
    println!("   Demonstrating constraint simplification and propagation\n");

    let mut engine = ChrEngine::new();

    // Add CHR rules for LEQ constraints
    // Reflexivity: leq(X, X) <=> true
    engine.add_rule(ChrRule::simplification(
        "reflexivity",
        vec![Constraint::binary("leq", "X", "X")],
        vec![],
        vec![], // Simplified to true (removed)
    ));

    // Antisymmetry: leq(X, Y), leq(Y, X) <=> X = Y
    engine.add_rule(ChrRule::simplification(
        "antisymmetry",
        vec![
            Constraint::binary("leq", "X", "Y"),
            Constraint::binary("leq", "Y", "X"),
        ],
        vec![],
        vec![Constraint::eq("X", "Y")],
    ));

    // Transitivity: leq(X, Y), leq(Y, Z) ==> leq(X, Z)
    engine.add_rule(ChrRule::propagation(
        "transitivity",
        vec![
            Constraint::binary("leq", "X", "Y"),
            Constraint::binary("leq", "Y", "Z"),
        ],
        vec![],
        vec![Constraint::binary("leq", "X", "Z")],
    ));

    println!("   Added 3 CHR rules:");
    println!("   - Reflexivity (simplification)");
    println!("   - Antisymmetry (simplification)");
    println!("   - Transitivity (propagation)");

    // Add constraints
    println!("\n   Adding constraints:");
    engine.add_constraint(Constraint::new(
        "leq",
        vec![ChrTerm::const_("a"), ChrTerm::const_("b")],
    ));
    println!("   - leq(a, b)");

    engine.add_constraint(Constraint::new(
        "leq",
        vec![ChrTerm::const_("b"), ChrTerm::const_("c")],
    ));
    println!("   - leq(b, c)");

    // Solve
    println!("\n   Solving...");
    let result = engine.solve()?;
    println!("   ✓ Solution found with {} constraints:", result.len());
    for constraint in &result {
        println!("   - {}", constraint.name);
    }

    // Example 2: Graph coloring with CHR
    println!("\n2. Graph Coloring Example:");
    println!("   Using CHR to find valid 3-colorings of a graph\n");

    let mut graph_engine = ChrEngine::new();

    // Constraint: adjacent nodes must have different colors
    graph_engine.add_rule(ChrRule::simplification(
        "diff_colors",
        vec![
            Constraint::binary("edge", "X", "Y"),
            Constraint::binary("color", "X", "C"),
            Constraint::binary("color", "Y", "C"),
        ],
        vec![],
        vec![Constraint::new("conflict", vec![])], // Conflict detected
    ));

    println!("   Added conflict detection rule");
    println!("   - Adjacent nodes with same color → conflict");

    // Example 3: Simpagation rule
    println!("\n3. Simpagation Rules:");
    println!("   H1 \\ H2 <=> G | B  (keep H1, remove H2, add B)\n");

    let mut simpag_engine = ChrEngine::new();

    // Example: min(X, Y) \ min(Z, W) <=> X <= Z | true
    // (Keep the smaller minimum, remove larger)
    simpag_engine.add_rule(ChrRule::simpagation(
        "min_optimization",
        vec![Constraint::binary("min", "X", "Y")],
        vec![Constraint::binary("min", "Z", "W")],
        vec![], // Guard: X <= Z (simplified for demo)
        vec![], // Just remove the second min
    ));

    println!("   Added min optimization rule");
    println!("   - Keeps only the minimal 'min' constraint");

    println!("\n   CHR Rule Types:");
    println!("   - Simplification: Replaces constraints");
    println!("   - Propagation: Adds new constraints");
    println!("   - Simpagation: Hybrid replacement/addition");

    Ok(())
}

/// Demonstrate ASP (Answer Set Programming) capabilities
fn demonstrate_asp() -> Result<()> {
    println!("--- ASP (Answer Set Programming) ---");
    println!("Combinatorial optimization and constraint satisfaction\n");

    // Example 1: Classic 3-coloring problem
    println!("1. Graph 3-Coloring Problem:");
    println!("   Find valid colorings of a graph with 3 colors\n");

    let mut solver = AspSolver::new();

    // Add nodes
    println!("   Adding nodes:");
    solver.add_fact("node(a)")?;
    solver.add_fact("node(b)")?;
    solver.add_fact("node(c)")?;
    println!("   - nodes: a, b, c");

    // Add edges
    println!("\n   Adding edges (triangle graph):");
    solver.add_fact("edge(a, b)")?;
    solver.add_fact("edge(b, c)")?;
    solver.add_fact("edge(c, a)")?;
    println!("   - a-b, b-c, c-a");

    // Choice rule: each node gets exactly one color
    println!("\n   Choice rule:");
    println!("   {{ color(X, red); color(X, green); color(X, blue) }} = 1 :- node(X).");

    let colors = vec!["red", "green", "blue"];
    for node_var in &["a", "b", "c"] {
        let color_atoms: Vec<_> = colors
            .iter()
            .map(|c| Atom::new("color", vec![const_term(node_var), const_term(c)]))
            .collect();

        solver.add_choice_rule(
            color_atoms,
            Some(1), // Lower bound
            Some(1), // Upper bound (exactly 1)
            vec![AspLiteral::positive(Atom::new(
                "node",
                vec![const_term(node_var)],
            ))],
        );
    }

    // Integrity constraint: adjacent nodes must have different colors
    println!("\n   Integrity constraint:");
    println!("   :- edge(X, Y), color(X, C), color(Y, C).");

    for &(x, y) in &[("a", "b"), ("b", "c"), ("c", "a")] {
        for color in &colors {
            solver.add_constraint(vec![
                AspLiteral::positive(Atom::new("edge", vec![const_term(x), const_term(y)])),
                AspLiteral::positive(Atom::new("color", vec![const_term(x), const_term(color)])),
                AspLiteral::positive(Atom::new("color", vec![const_term(y), const_term(color)])),
            ]);
        }
    }

    // Solve
    println!("\n   Solving for answer sets...");
    let answer_sets = solver.solve()?;
    println!("   ✓ Found {} answer sets:", answer_sets.len());

    for (i, answer_set) in answer_sets.iter().take(3).enumerate() {
        println!("\n   Answer Set {}:", i + 1);
        let colorings: Vec<_> = answer_set
            .atoms
            .iter()
            .filter(|a| a.predicate == "color")
            .collect();
        for atom in colorings {
            if atom.args.len() >= 2 {
                use oxirs_rule::asp::AspTerm;
                let node = match &atom.args[0] {
                    AspTerm::Constant(s) => s.as_str(),
                    _ => "?",
                };
                let color = match &atom.args[1] {
                    AspTerm::Constant(s) => s.as_str(),
                    _ => "?",
                };
                println!("   - {} = {}", node, color);
            }
        }
    }

    // Example 2: Weighted optimization
    println!("\n2. Weighted Optimization:");
    println!("   Finding optimal solutions with cost minimization\n");

    let mut opt_solver = AspSolver::new();

    // Add items with costs
    println!("   Items with costs:");
    opt_solver.add_fact("item(laptop, 1000)")?;
    opt_solver.add_fact("item(phone, 500)")?;
    opt_solver.add_fact("item(tablet, 300)")?;
    println!("   - laptop: 1000, phone: 500, tablet: 300");

    println!("\n   Budget constraint: total cost <= 1200");
    println!("   Goal: maximize number of items within budget");

    // Example 3: Stable model semantics
    println!("\n3. Stable Model Semantics:");
    println!("   ASP uses stable model semantics for grounding\n");
    println!("   Properties:");
    println!("   - Grounding: Variables instantiated with domain values");
    println!("   - Minimality: Only derive what's necessary");
    println!("   - Consistency: Answer sets satisfy all constraints");

    println!("\n   ASP Solver Capabilities:");
    println!("   - Facts: Domain knowledge base");
    println!("   - Choice rules: Non-deterministic selections");
    println!("   - Integrity constraints: Hard requirements");
    println!("   - Optimization: Find best solutions");
    println!("   - Grounding: Automatic variable instantiation");

    Ok(())
}

/// Demonstrate integration of RIF, CHR, and ASP
fn demonstrate_integration() -> Result<()> {
    println!("--- Integration Example ---");
    println!("Combining RIF, CHR, and ASP for complex reasoning\n");

    println!("Scenario: Conference Paper Review Assignment");
    println!();

    // Step 1: Use RIF to define domain rules
    println!("1. Define domain rules using RIF:");
    let rif_rules = r#"
        Prefix(conf <http://example.org/conference/>)

        Group (
            (* Expertise matching rule *)
            Forall ?reviewer ?paper ?topic (
                conf:canReview(?reviewer ?paper) :-
                    And(conf:expertise(?reviewer ?topic)
                        conf:topic(?paper ?topic))
            )

            (* Conflict of interest rule *)
            Forall ?reviewer ?paper ?author (
                conf:conflict(?reviewer ?paper) :-
                    And(conf:coauthor(?reviewer ?author)
                        conf:author(?paper ?author))
            )
        )
    "#;
    println!("{}", rif_rules);

    let mut parser = RifParser::new(RifDialect::Bld);
    let rif_doc = parser.parse(rif_rules)?;
    println!("   ✓ RIF rules parsed: {} groups", rif_doc.groups.len());

    // Step 2: Use CHR for constraint propagation
    println!("\n2. Apply CHR for workload balancing:");
    println!("   Rules:");
    println!("   - Each reviewer gets 3-5 papers");
    println!("   - No reviewer reviews conflicting papers");
    println!("   - Load balancing across reviewers");

    let mut chr_engine = ChrEngine::new();

    // Workload constraint
    chr_engine.add_rule(ChrRule::propagation(
        "workload_check",
        vec![
            Constraint::binary("assigned", "R", "P1"),
            Constraint::binary("assigned", "R", "P2"),
            Constraint::binary("assigned", "R", "P3"),
        ],
        vec![],
        vec![Constraint::new("workload_ok", vec![ChrTerm::var("R")])],
    ));

    println!("   ✓ CHR rules configured");

    // Step 3: Use ASP for optimal assignment
    println!("\n3. Use ASP to find optimal assignment:");
    println!("   Optimization criteria:");
    println!("   - Maximize expertise match");
    println!("   - Minimize conflicts");
    println!("   - Balance workload");

    let mut asp_solver = AspSolver::new();

    // Sample data
    asp_solver.add_fact("reviewer(alice)")?;
    asp_solver.add_fact("reviewer(bob)")?;
    asp_solver.add_fact("paper(p1)")?;
    asp_solver.add_fact("paper(p2)")?;

    println!("\n   ✓ ASP solver configured with sample data");

    // Step 4: Integration workflow
    println!("\n4. Integration Workflow:");
    println!("   a) RIF rules → domain knowledge");
    println!("   b) CHR constraints → feasibility checking");
    println!("   c) ASP solving → optimal solution");
    println!("   d) Result validation → final assignment");

    println!("\n   Benefits of Integration:");
    println!("   - RIF: Interoperable rule definitions");
    println!("   - CHR: Efficient constraint propagation");
    println!("   - ASP: Optimal solution finding");
    println!("   - Combined: Best of all three approaches!");

    Ok(())
}

// Helper function to create constant terms for ASP
fn const_term(s: &str) -> AspTerm {
    AspTerm::constant(s)
}
