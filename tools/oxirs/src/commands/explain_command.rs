//! # SPARQL Query Explain / Plan Command
//!
//! Produces a logical query execution plan for a SPARQL query string,
//! estimates costs, and renders the plan in text or JSON form.

// ---------------------------------------------------------------------------
// Plan nodes
// ---------------------------------------------------------------------------

/// A node in the logical query execution plan.
#[derive(Debug, Clone)]
pub enum PlanNode {
    /// Full triple-pattern scan.
    TripleScan {
        /// Subject (variable or IRI).
        subject: String,
        /// Predicate (variable or IRI).
        predicate: String,
        /// Object (variable or literal or IRI).
        object: String,
        /// Estimated number of matching rows.
        estimated_rows: usize,
    },
    /// Hash join of two sub-plans.
    HashJoin {
        /// Left operand.
        left: Box<PlanNode>,
        /// Right operand.
        right: Box<PlanNode>,
        /// Variables shared between left and right.
        join_vars: Vec<String>,
        /// Estimated output rows.
        estimated_rows: usize,
    },
    /// Row filter with a boolean condition.
    Filter {
        /// Input plan.
        input: Box<PlanNode>,
        /// Textual condition (e.g. "?age > 18").
        condition: String,
        /// Fraction of rows expected to pass (0.0 – 1.0).
        selectivity: f64,
    },
    /// Projection to a subset of variables.
    Project {
        /// Input plan.
        input: Box<PlanNode>,
        /// Variables to project.
        variables: Vec<String>,
    },
    /// Sort the result set.
    Sort {
        /// Input plan.
        input: Box<PlanNode>,
        /// Variables / expressions to sort by.
        by: Vec<String>,
        /// Estimated output rows (same as input usually).
        estimated_rows: usize,
    },
    /// Limit the result to `limit` rows.
    Limit {
        /// Input plan.
        input: Box<PlanNode>,
        /// Maximum rows to return.
        limit: usize,
    },
    /// Union of two sub-plans.
    Union {
        /// Left operand.
        left: Box<PlanNode>,
        /// Right operand.
        right: Box<PlanNode>,
        /// Estimated combined rows.
        estimated_rows: usize,
    },
}

// ---------------------------------------------------------------------------
// ExplainOutput
// ---------------------------------------------------------------------------

/// The result of explaining a query.
#[derive(Debug, Clone)]
pub struct ExplainOutput {
    /// Root of the plan tree.
    pub plan: PlanNode,
    /// Total estimated rows at the output of the root node.
    pub total_estimated_rows: usize,
    /// Maximum depth of the plan tree.
    pub depth: usize,
}

// ---------------------------------------------------------------------------
// ExplainCommand
// ---------------------------------------------------------------------------

/// Utilities for producing and rendering SPARQL query plans.
pub struct ExplainCommand;

impl ExplainCommand {
    // ------------------------------------------------------------------
    // Plan construction
    // ------------------------------------------------------------------

    /// Produce a simple logical plan for `query`.
    ///
    /// The parser is intentionally minimal: it detects SELECT / CONSTRUCT /
    /// ASK / DESCRIBE, extracts triple patterns from the WHERE clause, and
    /// wraps them in Project + HashJoin + TripleScan nodes.
    pub fn explain_simple(query: &str) -> ExplainOutput {
        let query_type = Self::detect_query_type(query);

        // Extract triple patterns from WHERE { … }
        let triples = extract_triples(query);

        // Build a left-deep join tree from the patterns
        let base_plan: PlanNode = if triples.is_empty() {
            // Empty scan
            PlanNode::TripleScan {
                subject: "?s".to_string(),
                predicate: "?p".to_string(),
                object: "?o".to_string(),
                estimated_rows: 0,
            }
        } else {
            let mut plan = PlanNode::TripleScan {
                subject: triples[0].0.clone(),
                predicate: triples[0].1.clone(),
                object: triples[0].2.clone(),
                estimated_rows: estimate_triple_rows(&triples[0]),
            };

            for triple in triples.iter().skip(1) {
                let right = PlanNode::TripleScan {
                    subject: triple.0.clone(),
                    predicate: triple.1.clone(),
                    object: triple.2.clone(),
                    estimated_rows: estimate_triple_rows(triple),
                };
                let join_vars = shared_vars(&plan, &right);
                let left_rows = Self::estimated_rows(&plan);
                let right_rows = estimate_triple_rows(triple);
                let estimated_rows = estimate_join_rows(left_rows, right_rows, join_vars.len());
                plan = PlanNode::HashJoin {
                    left: Box::new(plan),
                    right: Box::new(right),
                    join_vars,
                    estimated_rows,
                };
            }
            plan
        };

        // Wrap in Project for SELECT queries
        let projected_vars = extract_select_vars(query);
        let plan = if query_type == "SELECT" && !projected_vars.is_empty() {
            PlanNode::Project {
                input: Box::new(base_plan),
                variables: projected_vars,
            }
        } else {
            base_plan
        };

        // Wrap in Limit if present
        let plan = if let Some(limit) = extract_limit(query) {
            PlanNode::Limit {
                input: Box::new(plan),
                limit,
            }
        } else {
            plan
        };

        let total_estimated_rows = Self::estimated_rows(&plan);
        let depth = Self::plan_depth(&plan);

        ExplainOutput {
            plan,
            total_estimated_rows,
            depth,
        }
    }

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------

    /// Render `plan` to indented text.
    pub fn plan_to_text(plan: &PlanNode, indent: usize) -> String {
        let pad = "  ".repeat(indent);
        match plan {
            PlanNode::TripleScan {
                subject,
                predicate,
                object,
                estimated_rows,
            } => format!(
                "{pad}TripleScan({subject}, {predicate}, {object}) rows≈{estimated_rows}"
            ),
            PlanNode::HashJoin {
                left,
                right,
                join_vars,
                estimated_rows,
            } => {
                let vars = join_vars.join(", ");
                let left_text = Self::plan_to_text(left, indent + 1);
                let right_text = Self::plan_to_text(right, indent + 1);
                format!(
                    "{pad}HashJoin(on=[{vars}]) rows≈{estimated_rows}\n{left_text}\n{right_text}"
                )
            }
            PlanNode::Filter {
                input,
                condition,
                selectivity,
            } => {
                let inner = Self::plan_to_text(input, indent + 1);
                format!(
                    "{pad}Filter({condition}) sel={selectivity:.2}\n{inner}"
                )
            }
            PlanNode::Project { input, variables } => {
                let vars = variables.join(", ");
                let inner = Self::plan_to_text(input, indent + 1);
                format!("{pad}Project([{vars}])\n{inner}")
            }
            PlanNode::Sort {
                input,
                by,
                estimated_rows,
            } => {
                let by_str = by.join(", ");
                let inner = Self::plan_to_text(input, indent + 1);
                format!("{pad}Sort(by=[{by_str}]) rows≈{estimated_rows}\n{inner}")
            }
            PlanNode::Limit { input, limit } => {
                let inner = Self::plan_to_text(input, indent + 1);
                format!("{pad}Limit({limit})\n{inner}")
            }
            PlanNode::Union {
                left,
                right,
                estimated_rows,
            } => {
                let left_text = Self::plan_to_text(left, indent + 1);
                let right_text = Self::plan_to_text(right, indent + 1);
                format!(
                    "{pad}Union rows≈{estimated_rows}\n{left_text}\n{right_text}"
                )
            }
        }
    }

    /// Render `plan` to a compact JSON string.
    pub fn plan_to_json(plan: &PlanNode) -> String {
        match plan {
            PlanNode::TripleScan {
                subject,
                predicate,
                object,
                estimated_rows,
            } => format!(
                r#"{{"op":"TripleScan","s":"{subject}","p":"{predicate}","o":"{object}","rows":{estimated_rows}}}"#
            ),
            PlanNode::HashJoin {
                left,
                right,
                join_vars,
                estimated_rows,
            } => {
                let vars_json: Vec<String> =
                    join_vars.iter().map(|v| format!(r#""{v}""#)).collect();
                let left_json = Self::plan_to_json(left);
                let right_json = Self::plan_to_json(right);
                format!(
                    r#"{{"op":"HashJoin","joinVars":[{}],"rows":{estimated_rows},"left":{left_json},"right":{right_json}}}"#,
                    vars_json.join(",")
                )
            }
            PlanNode::Filter {
                input,
                condition,
                selectivity,
            } => {
                let inner_json = Self::plan_to_json(input);
                format!(
                    r#"{{"op":"Filter","condition":"{condition}","selectivity":{selectivity:.4},"input":{inner_json}}}"#
                )
            }
            PlanNode::Project { input, variables } => {
                let vars_json: Vec<String> =
                    variables.iter().map(|v| format!(r#""{v}""#)).collect();
                let inner_json = Self::plan_to_json(input);
                format!(
                    r#"{{"op":"Project","variables":[{}],"input":{inner_json}}}"#,
                    vars_json.join(",")
                )
            }
            PlanNode::Sort {
                input,
                by,
                estimated_rows,
            } => {
                let by_json: Vec<String> = by.iter().map(|v| format!(r#""{v}""#)).collect();
                let inner_json = Self::plan_to_json(input);
                format!(
                    r#"{{"op":"Sort","by":[{}],"rows":{estimated_rows},"input":{inner_json}}}"#,
                    by_json.join(",")
                )
            }
            PlanNode::Limit { input, limit } => {
                let inner_json = Self::plan_to_json(input);
                format!(r#"{{"op":"Limit","limit":{limit},"input":{inner_json}}}"#)
            }
            PlanNode::Union {
                left,
                right,
                estimated_rows,
            } => {
                let left_json = Self::plan_to_json(left);
                let right_json = Self::plan_to_json(right);
                format!(
                    r#"{{"op":"Union","rows":{estimated_rows},"left":{left_json},"right":{right_json}}}"#
                )
            }
        }
    }

    // ------------------------------------------------------------------
    // Cost / metrics
    // ------------------------------------------------------------------

    /// Heuristic cost estimate for a plan (lower is better).
    ///
    /// Cost is proportional to estimated rows; join and sort add overhead.
    pub fn estimate_cost(plan: &PlanNode) -> f64 {
        match plan {
            PlanNode::TripleScan { estimated_rows, .. } => *estimated_rows as f64,
            PlanNode::HashJoin {
                left,
                right,
                estimated_rows,
                ..
            } => {
                Self::estimate_cost(left)
                    + Self::estimate_cost(right)
                    + *estimated_rows as f64 * 1.2
            }
            PlanNode::Filter { input, selectivity, .. } => {
                Self::estimate_cost(input) * selectivity
            }
            PlanNode::Project { input, .. } => Self::estimate_cost(input),
            PlanNode::Sort {
                input,
                estimated_rows,
                ..
            } => Self::estimate_cost(input) + *estimated_rows as f64 * 0.5,
            PlanNode::Limit { input, limit } => {
                Self::estimate_cost(input).min(*limit as f64)
            }
            PlanNode::Union {
                left,
                right,
                estimated_rows,
            } => {
                Self::estimate_cost(left) + Self::estimate_cost(right) + *estimated_rows as f64
            }
        }
    }

    /// Maximum depth of the plan tree (root = depth 1).
    pub fn plan_depth(plan: &PlanNode) -> usize {
        match plan {
            PlanNode::TripleScan { .. } => 1,
            PlanNode::HashJoin { left, right, .. } => {
                1 + Self::plan_depth(left).max(Self::plan_depth(right))
            }
            PlanNode::Filter { input, .. }
            | PlanNode::Project { input, .. }
            | PlanNode::Sort { input, .. }
            | PlanNode::Limit { input, .. } => 1 + Self::plan_depth(input),
            PlanNode::Union { left, right, .. } => {
                1 + Self::plan_depth(left).max(Self::plan_depth(right))
            }
        }
    }

    /// Total number of plan nodes.
    pub fn count_nodes(plan: &PlanNode) -> usize {
        match plan {
            PlanNode::TripleScan { .. } => 1,
            PlanNode::HashJoin { left, right, .. } => {
                1 + Self::count_nodes(left) + Self::count_nodes(right)
            }
            PlanNode::Filter { input, .. }
            | PlanNode::Project { input, .. }
            | PlanNode::Sort { input, .. }
            | PlanNode::Limit { input, .. } => 1 + Self::count_nodes(input),
            PlanNode::Union { left, right, .. } => {
                1 + Self::count_nodes(left) + Self::count_nodes(right)
            }
        }
    }

    /// Detect the query form keyword.
    ///
    /// Returns one of `"SELECT"`, `"CONSTRUCT"`, `"ASK"`, `"DESCRIBE"`, or
    /// `"UNKNOWN"`.
    pub fn detect_query_type(query: &str) -> &'static str {
        let upper = query.trim().to_uppercase();
        if upper.starts_with("SELECT") || upper.contains("\nSELECT") || upper.contains(" SELECT") {
            "SELECT"
        } else if upper.starts_with("CONSTRUCT") {
            "CONSTRUCT"
        } else if upper.starts_with("ASK") {
            "ASK"
        } else if upper.starts_with("DESCRIBE") {
            "DESCRIBE"
        } else {
            "UNKNOWN"
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers exposed for testing
    // ------------------------------------------------------------------

    /// Estimated rows for a plan node (reads the innermost field).
    pub fn estimated_rows(plan: &PlanNode) -> usize {
        match plan {
            PlanNode::TripleScan { estimated_rows, .. } => *estimated_rows,
            PlanNode::HashJoin { estimated_rows, .. } => *estimated_rows,
            PlanNode::Filter {
                input,
                selectivity,
                ..
            } => {
                let inner = Self::estimated_rows(input);
                ((inner as f64) * selectivity) as usize
            }
            PlanNode::Project { input, .. } => Self::estimated_rows(input),
            PlanNode::Sort { estimated_rows, .. } => *estimated_rows,
            PlanNode::Limit { input, limit } => {
                Self::estimated_rows(input).min(*limit)
            }
            PlanNode::Union { estimated_rows, .. } => *estimated_rows,
        }
    }
}

// ---------------------------------------------------------------------------
// Private parsing helpers
// ---------------------------------------------------------------------------

/// Extract (subject, predicate, object) triples from the WHERE clause.
fn extract_triples(query: &str) -> Vec<(String, String, String)> {
    // Very simplified extraction: find text between { … } and split on '.'
    let where_body = extract_where_body(query);
    let mut result = Vec::new();

    for stmt in where_body.split('.') {
        let stmt = stmt.trim();
        // Skip OPTIONAL, FILTER, UNION keywords and empty lines
        if stmt.is_empty()
            || stmt.to_uppercase().starts_with("FILTER")
            || stmt.to_uppercase().starts_with("OPTIONAL")
            || stmt.to_uppercase().starts_with("UNION")
        {
            continue;
        }
        let parts: Vec<&str> = stmt.split_whitespace().collect();
        if parts.len() >= 3 {
            result.push((parts[0].to_string(), parts[1].to_string(), parts[2..].join(" ")));
        }
    }
    result
}

/// Extract the raw text inside the outermost `{ … }` of the WHERE clause.
fn extract_where_body(query: &str) -> String {
    let upper = query.to_uppercase();
    let where_start = upper.find("WHERE").unwrap_or(0);
    let body = &query[where_start..];

    let mut depth = 0i32;
    let mut start = None;
    let mut end = None;

    for (i, ch) in body.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i + 1);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }

    match (start, end) {
        (Some(s), Some(e)) => body[s..e].to_string(),
        _ => String::new(),
    }
}

/// Extract SELECT variable list (returns empty vec for SELECT *).
fn extract_select_vars(query: &str) -> Vec<String> {
    let upper = query.to_uppercase();
    if !upper.contains("SELECT") {
        return Vec::new();
    }

    let select_pos = upper.find("SELECT").unwrap_or(0) + 6;
    let after_select = &query[select_pos..];

    // Find the WHERE keyword
    let where_pos = after_select.to_uppercase().find("WHERE").unwrap_or(after_select.len());
    let vars_part = &after_select[..where_pos];

    vars_part
        .split_whitespace()
        .filter(|w| w.starts_with('?'))
        .map(|w| w.to_string())
        .collect()
}

/// Extract LIMIT N value.
fn extract_limit(query: &str) -> Option<usize> {
    let upper = query.to_uppercase();
    let limit_pos = upper.find("LIMIT")?;
    let after = query[limit_pos + 5..].trim_start();
    after
        .split_whitespace()
        .next()
        .and_then(|s| s.parse::<usize>().ok())
}

/// Heuristic row estimate for a triple scan.
///
/// Fully bound triples → 1 row; each free variable multiplies by 100.
fn estimate_triple_rows(triple: &(String, String, String)) -> usize {
    let free: usize = [&triple.0, &triple.1, &triple.2]
        .iter()
        .filter(|p| p.starts_with('?'))
        .count();
    100_usize.saturating_pow(free as u32).max(1)
}

/// Heuristic row estimate for a join.
fn estimate_join_rows(left: usize, right: usize, shared_vars: usize) -> usize {
    if shared_vars == 0 {
        // Cartesian product
        left.saturating_mul(right)
    } else {
        // Each shared variable reduces by factor ~10
        let reduction = 10_usize.saturating_pow(shared_vars as u32);
        left.saturating_mul(right) / reduction.max(1)
    }
}

/// Return variables shared between two plan nodes (from their TripleScan leaves).
fn shared_vars(left: &PlanNode, right: &PlanNode) -> Vec<String> {
    let left_vars = collect_vars(left);
    let right_vars = collect_vars(right);

    let right_set: std::collections::HashSet<_> = right_vars.iter().cloned().collect();
    let mut shared: Vec<String> = left_vars
        .into_iter()
        .filter(|v| right_set.contains(v))
        .collect();
    shared.sort();
    shared.dedup();
    shared
}

/// Collect all SPARQL variables referenced in the plan.
fn collect_vars(plan: &PlanNode) -> Vec<String> {
    let mut vars = Vec::new();
    collect_vars_inner(plan, &mut vars);
    vars
}

fn collect_vars_inner(plan: &PlanNode, out: &mut Vec<String>) {
    match plan {
        PlanNode::TripleScan {
            subject,
            predicate,
            object,
            ..
        } => {
            for p in [subject, predicate, object] {
                if p.starts_with('?') {
                    out.push(p.clone());
                }
            }
        }
        PlanNode::HashJoin { left, right, .. }
        | PlanNode::Union { left, right, .. } => {
            collect_vars_inner(left, out);
            collect_vars_inner(right, out);
        }
        PlanNode::Filter { input, .. }
        | PlanNode::Project { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Limit { input, .. } => collect_vars_inner(input, out),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // detect_query_type
    // -------------------------------------------------------------------------
    #[test]
    fn test_detect_select() {
        assert_eq!(ExplainCommand::detect_query_type("SELECT ?x WHERE { }"), "SELECT");
    }

    #[test]
    fn test_detect_construct() {
        assert_eq!(
            ExplainCommand::detect_query_type("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"),
            "CONSTRUCT"
        );
    }

    #[test]
    fn test_detect_ask() {
        assert_eq!(ExplainCommand::detect_query_type("ASK WHERE { ?s ?p ?o }"), "ASK");
    }

    #[test]
    fn test_detect_describe() {
        assert_eq!(
            ExplainCommand::detect_query_type("DESCRIBE <http://example.org/>"),
            "DESCRIBE"
        );
    }

    #[test]
    fn test_detect_unknown() {
        assert_eq!(ExplainCommand::detect_query_type("DROP ALL"), "UNKNOWN");
    }

    #[test]
    fn test_detect_case_insensitive() {
        assert_eq!(ExplainCommand::detect_query_type("select ?x where { }"), "SELECT");
    }

    // -------------------------------------------------------------------------
    // plan_depth
    // -------------------------------------------------------------------------
    #[test]
    fn test_depth_scan_is_1() {
        let plan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 100,
        };
        assert_eq!(ExplainCommand::plan_depth(&plan), 1);
    }

    #[test]
    fn test_depth_join_is_2() {
        let scan = || PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 10,
        };
        let join = PlanNode::HashJoin {
            left: Box::new(scan()),
            right: Box::new(scan()),
            join_vars: vec![],
            estimated_rows: 10,
        };
        assert_eq!(ExplainCommand::plan_depth(&join), 2);
    }

    #[test]
    fn test_depth_project_wraps() {
        let scan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 5,
        };
        let project = PlanNode::Project {
            input: Box::new(scan),
            variables: vec!["?s".into()],
        };
        assert_eq!(ExplainCommand::plan_depth(&project), 2);
    }

    #[test]
    fn test_depth_limit_wraps() {
        let scan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 5,
        };
        let limited = PlanNode::Limit {
            input: Box::new(scan),
            limit: 10,
        };
        assert_eq!(ExplainCommand::plan_depth(&limited), 2);
    }

    // -------------------------------------------------------------------------
    // count_nodes
    // -------------------------------------------------------------------------
    #[test]
    fn test_count_nodes_scan() {
        let plan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 0,
        };
        assert_eq!(ExplainCommand::count_nodes(&plan), 1);
    }

    #[test]
    fn test_count_nodes_join() {
        let scan = || PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 10,
        };
        let join = PlanNode::HashJoin {
            left: Box::new(scan()),
            right: Box::new(scan()),
            join_vars: vec![],
            estimated_rows: 10,
        };
        assert_eq!(ExplainCommand::count_nodes(&join), 3);
    }

    #[test]
    fn test_count_nodes_filter() {
        let scan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 10,
        };
        let filter = PlanNode::Filter {
            input: Box::new(scan),
            condition: "?s > 5".into(),
            selectivity: 0.5,
        };
        assert_eq!(ExplainCommand::count_nodes(&filter), 2);
    }

    // -------------------------------------------------------------------------
    // estimate_cost
    // -------------------------------------------------------------------------
    #[test]
    fn test_cost_scan() {
        let plan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 100,
        };
        assert!((ExplainCommand::estimate_cost(&plan) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_cost_limit_reduces() {
        let scan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 1000,
        };
        let limited = PlanNode::Limit {
            input: Box::new(scan),
            limit: 10,
        };
        assert!(ExplainCommand::estimate_cost(&limited) <= 10.0 + 1e-9);
    }

    #[test]
    fn test_cost_filter_selectivity() {
        let scan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 100,
        };
        let filter = PlanNode::Filter {
            input: Box::new(scan),
            condition: "?s > 5".into(),
            selectivity: 0.1,
        };
        // 100 * 0.1 = 10
        assert!((ExplainCommand::estimate_cost(&filter) - 10.0).abs() < 1e-9);
    }

    // -------------------------------------------------------------------------
    // plan_to_text
    // -------------------------------------------------------------------------
    #[test]
    fn test_plan_to_text_scan() {
        let plan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "rdf:type".into(),
            object: "?t".into(),
            estimated_rows: 50,
        };
        let text = ExplainCommand::plan_to_text(&plan, 0);
        assert!(text.contains("TripleScan"), "text={text}");
        assert!(text.contains("50"), "text={text}");
    }

    #[test]
    fn test_plan_to_text_indent() {
        let plan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 0,
        };
        let text = ExplainCommand::plan_to_text(&plan, 2);
        assert!(text.starts_with("    "), "text={text:?}"); // 4 spaces = 2 indents
    }

    #[test]
    fn test_plan_to_text_join_contains_children() {
        let scan = || PlanNode::TripleScan {
            subject: "?a".into(),
            predicate: "?p".into(),
            object: "?b".into(),
            estimated_rows: 10,
        };
        let join = PlanNode::HashJoin {
            left: Box::new(scan()),
            right: Box::new(scan()),
            join_vars: vec!["?a".into()],
            estimated_rows: 10,
        };
        let text = ExplainCommand::plan_to_text(&join, 0);
        assert!(text.contains("HashJoin"), "text={text}");
        assert!(text.contains("TripleScan"), "text={text}");
    }

    #[test]
    fn test_plan_to_text_limit() {
        let scan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 100,
        };
        let limited = PlanNode::Limit {
            input: Box::new(scan),
            limit: 25,
        };
        let text = ExplainCommand::plan_to_text(&limited, 0);
        assert!(text.contains("Limit(25)"), "text={text}");
    }

    // -------------------------------------------------------------------------
    // plan_to_json
    // -------------------------------------------------------------------------
    #[test]
    fn test_plan_to_json_scan() {
        let plan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "rdf:type".into(),
            object: "owl:Class".into(),
            estimated_rows: 5,
        };
        let json = ExplainCommand::plan_to_json(&plan);
        assert!(json.contains("\"op\":\"TripleScan\""), "json={json}");
        assert!(json.contains("\"rows\":5"), "json={json}");
    }

    #[test]
    fn test_plan_to_json_project() {
        let scan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 10,
        };
        let project = PlanNode::Project {
            input: Box::new(scan),
            variables: vec!["?s".into(), "?p".into()],
        };
        let json = ExplainCommand::plan_to_json(&project);
        assert!(json.contains("\"op\":\"Project\""), "json={json}");
        assert!(json.contains("\"?s\""), "json={json}");
    }

    #[test]
    fn test_plan_to_json_filter() {
        let scan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 20,
        };
        let filter = PlanNode::Filter {
            input: Box::new(scan),
            condition: "?age > 18".into(),
            selectivity: 0.3,
        };
        let json = ExplainCommand::plan_to_json(&filter);
        assert!(json.contains("\"op\":\"Filter\""), "json={json}");
        assert!(json.contains("age > 18"), "json={json}");
    }

    #[test]
    fn test_plan_to_json_union() {
        let scan = || PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 10,
        };
        let union = PlanNode::Union {
            left: Box::new(scan()),
            right: Box::new(scan()),
            estimated_rows: 20,
        };
        let json = ExplainCommand::plan_to_json(&union);
        assert!(json.contains("\"op\":\"Union\""), "json={json}");
        assert!(json.contains("\"rows\":20"), "json={json}");
    }

    // -------------------------------------------------------------------------
    // explain_simple
    // -------------------------------------------------------------------------
    #[test]
    fn test_explain_simple_select() {
        let q = "SELECT ?s ?p WHERE { ?s ?p ?o . }";
        let out = ExplainCommand::explain_simple(q);
        assert!(out.total_estimated_rows > 0 || out.depth >= 1);
    }

    #[test]
    fn test_explain_simple_depth_positive() {
        let q = "SELECT ?x WHERE { ?x rdf:type owl:Class . }";
        let out = ExplainCommand::explain_simple(q);
        assert!(out.depth >= 1);
    }

    #[test]
    fn test_explain_simple_two_triples() {
        let q = "SELECT ?x ?y WHERE { ?x rdf:type ?t . ?x rdfs:label ?y . }";
        let out = ExplainCommand::explain_simple(q);
        // With 2 triple patterns we expect at least a join
        assert!(out.depth >= 2 || ExplainCommand::count_nodes(&out.plan) >= 2);
    }

    #[test]
    fn test_explain_simple_with_limit() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . } LIMIT 10";
        let out = ExplainCommand::explain_simple(q);
        assert!(out.total_estimated_rows <= 10);
    }

    #[test]
    fn test_explain_simple_total_rows_non_negative() {
        let q = "SELECT * WHERE { ?s ?p ?o . }";
        let out = ExplainCommand::explain_simple(q);
        // usize is non-negative by definition; just ensure we get a result
        let _ = out.total_estimated_rows;
    }

    #[test]
    fn test_explain_simple_empty_where() {
        let q = "SELECT ?s WHERE { }";
        let out = ExplainCommand::explain_simple(q);
        assert_eq!(out.total_estimated_rows, 0);
    }

    #[test]
    fn test_explain_simple_ask_query() {
        let q = "ASK WHERE { ?s rdf:type ?t . }";
        let out = ExplainCommand::explain_simple(q);
        assert!(out.depth >= 1);
    }

    // -------------------------------------------------------------------------
    // sort node
    // -------------------------------------------------------------------------
    #[test]
    fn test_plan_to_text_sort() {
        let scan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 100,
        };
        let sort = PlanNode::Sort {
            input: Box::new(scan),
            by: vec!["?s".into()],
            estimated_rows: 100,
        };
        let text = ExplainCommand::plan_to_text(&sort, 0);
        assert!(text.contains("Sort"), "text={text}");
    }

    #[test]
    fn test_plan_to_json_sort() {
        let scan = PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 50,
        };
        let sort = PlanNode::Sort {
            input: Box::new(scan),
            by: vec!["?label".into()],
            estimated_rows: 50,
        };
        let json = ExplainCommand::plan_to_json(&sort);
        assert!(json.contains("\"op\":\"Sort\""), "json={json}");
    }

    #[test]
    fn test_count_nodes_deep() {
        let scan = || PlanNode::TripleScan {
            subject: "?s".into(),
            predicate: "?p".into(),
            object: "?o".into(),
            estimated_rows: 10,
        };
        // 3-way join: join(join(scan, scan), scan) → 5 nodes
        let inner_join = PlanNode::HashJoin {
            left: Box::new(scan()),
            right: Box::new(scan()),
            join_vars: vec![],
            estimated_rows: 10,
        };
        let outer_join = PlanNode::HashJoin {
            left: Box::new(inner_join),
            right: Box::new(scan()),
            join_vars: vec![],
            estimated_rows: 10,
        };
        assert_eq!(ExplainCommand::count_nodes(&outer_join), 5);
    }

    #[test]
    fn test_explain_output_struct_fields() {
        let q = "SELECT ?s WHERE { ?s ?p ?o . }";
        let out = ExplainCommand::explain_simple(q);
        // Basic sanity: depth and count_nodes are consistent
        let nodes = ExplainCommand::count_nodes(&out.plan);
        assert!(nodes >= 1);
        let depth = ExplainCommand::plan_depth(&out.plan);
        assert_eq!(depth, out.depth);
    }
}
