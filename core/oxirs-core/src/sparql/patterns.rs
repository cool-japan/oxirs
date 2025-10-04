//! SPARQL pattern matching: OPTIONAL and UNION clauses

use crate::model::{Quad, Term};
use crate::rdf_store::VariableBinding;
use crate::Result;

/// Simple triple pattern for matching
#[derive(Debug, Clone)]
pub struct SimpleTriplePattern {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
}

/// Pattern group (required or optional)
#[derive(Debug, Clone)]
pub struct PatternGroup {
    pub patterns: Vec<SimpleTriplePattern>,
    pub optional: bool,
}

/// Union group for SPARQL UNION clause
#[derive(Debug, Clone)]
pub struct UnionGroup {
    pub branches: Vec<Vec<PatternGroup>>,
}

/// Check if query contains UNION
pub fn has_union(sparql: &str) -> bool {
    let sparql_upper = sparql.to_uppercase();
    sparql_upper.contains(" UNION ")
        || sparql_upper.contains("\nUNION\n")
        || sparql_upper.contains("{UNION")
}

/// Find matching closing brace
pub fn find_matching_brace(text: &str, start_pos: usize) -> Option<usize> {
    let chars: Vec<char> = text.chars().collect();
    if start_pos >= chars.len() || chars[start_pos] != '{' {
        return None;
    }

    let mut brace_count = 1;
    for i in (start_pos + 1)..chars.len() {
        if chars[i] == '{' {
            brace_count += 1;
        } else if chars[i] == '}' {
            brace_count -= 1;
            if brace_count == 0 {
                return Some(i);
            }
        }
    }

    None
}

/// Parse a simple triple pattern from text
pub fn parse_simple_pattern(text: &str) -> Option<SimpleTriplePattern> {
    // Simple pattern: ?s ?p ?o . or <uri> <uri> "literal" .
    let text = text.trim();

    // Split by periods and process each potential pattern
    for line in text.split('.') {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Skip FILTER, BIND, VALUES, UNION keywords
        let line_upper = line.to_uppercase();
        if line_upper.contains("FILTER")
            || line_upper.contains("BIND")
            || line_upper.contains("VALUES")
            || line_upper.contains("UNION")
        {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            return Some(SimpleTriplePattern {
                subject: Some(parts[0].to_string()),
                predicate: Some(parts[1].to_string()),
                object: Some(parts[2..].join(" ")),
            });
        }
    }

    None
}

/// Extract pattern groups (required and optional) from WHERE clause
pub fn extract_pattern_groups(sparql: &str) -> Result<Vec<PatternGroup>> {
    let mut groups = Vec::new();

    if let Some(where_start) = sparql.to_uppercase().find("WHERE") {
        let where_clause = &sparql[where_start + 5..];

        // Find the main WHERE block
        if let Some(start_brace) = where_clause.find('{') {
            if let Some(end_brace) = find_matching_brace(where_clause, start_brace) {
                let pattern_text = &where_clause[start_brace + 1..end_brace];

                // Check for OPTIONAL blocks
                let sparql_upper = pattern_text.to_uppercase();
                if sparql_upper.contains("OPTIONAL") {
                    // Parse with OPTIONAL support
                    let mut pos = 0;
                    let mut required_patterns = Vec::new();

                    while pos < pattern_text.len() {
                        // Look for OPTIONAL keyword
                        if let Some(opt_pos) = pattern_text[pos..].to_uppercase().find("OPTIONAL") {
                            let abs_pos = pos + opt_pos;

                            // Add any required patterns before OPTIONAL
                            let before_optional = &pattern_text[pos..abs_pos];
                            if let Some(req_pattern) = parse_simple_pattern(before_optional) {
                                required_patterns.push(req_pattern);
                            }

                            // Find OPTIONAL block
                            let after_optional = &pattern_text[abs_pos + 8..];
                            if let Some(opt_brace) = after_optional.find('{') {
                                if let Some(opt_end) =
                                    find_matching_brace(after_optional, opt_brace)
                                {
                                    let optional_content = &after_optional[opt_brace + 1..opt_end];

                                    // Parse optional patterns
                                    if let Some(opt_pattern) =
                                        parse_simple_pattern(optional_content)
                                    {
                                        groups.push(PatternGroup {
                                            patterns: vec![opt_pattern],
                                            optional: true,
                                        });
                                    }

                                    pos = abs_pos + 8 + opt_end + 1;
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        } else {
                            // No more OPTIONAL, add remaining as required
                            if let Some(req_pattern) = parse_simple_pattern(&pattern_text[pos..]) {
                                required_patterns.push(req_pattern);
                            }
                            break;
                        }
                    }

                    // Add required patterns group
                    if !required_patterns.is_empty() {
                        groups.push(PatternGroup {
                            patterns: required_patterns,
                            optional: false,
                        });
                    }
                } else {
                    // No OPTIONAL - all patterns are required
                    if let Some(pattern) = parse_simple_pattern(pattern_text) {
                        groups.push(PatternGroup {
                            patterns: vec![pattern],
                            optional: false,
                        });
                    }
                }
            }
        }
    }

    Ok(groups)
}

/// Apply optional patterns to extend existing bindings
pub fn apply_optional_patterns<F>(
    bindings: Vec<VariableBinding>,
    patterns: &[SimpleTriplePattern],
    query_quads: F,
) -> Result<Vec<VariableBinding>>
where
    F: Fn(&SimpleTriplePattern) -> Result<Vec<Quad>>,
{
    let mut new_results = Vec::new();

    for binding in bindings {
        let mut extended = false;

        // Try to extend this binding with optional patterns
        for pattern in patterns {
            let matching_quads = query_quads(pattern)?;

            for quad in matching_quads {
                let mut new_binding = binding.clone();
                let mut compatible = true;

                // Check subject compatibility
                if let Some(var) = &pattern.subject {
                    if let Some(var_name) = var.strip_prefix('?') {
                        if let Some(existing) = binding.get(var_name) {
                            let new_term = Term::from(quad.subject().clone());
                            if format!("{:?}", existing) != format!("{:?}", new_term) {
                                compatible = false;
                            }
                        } else {
                            new_binding
                                .bind(var_name.to_string(), Term::from(quad.subject().clone()));
                        }
                    }
                }

                // Check predicate compatibility
                if compatible {
                    if let Some(var) = &pattern.predicate {
                        if let Some(var_name) = var.strip_prefix('?') {
                            if let Some(existing) = binding.get(var_name) {
                                let new_term = Term::from(quad.predicate().clone());
                                if format!("{:?}", existing) != format!("{:?}", new_term) {
                                    compatible = false;
                                }
                            } else {
                                new_binding.bind(
                                    var_name.to_string(),
                                    Term::from(quad.predicate().clone()),
                                );
                            }
                        }
                    }
                }

                // Check object compatibility
                if compatible {
                    if let Some(var) = &pattern.object {
                        if let Some(var_name) = var.strip_prefix('?') {
                            if let Some(existing) = binding.get(var_name) {
                                let new_term = Term::from(quad.object().clone());
                                if format!("{:?}", existing) != format!("{:?}", new_term) {
                                    compatible = false;
                                }
                            } else {
                                new_binding
                                    .bind(var_name.to_string(), Term::from(quad.object().clone()));
                            }
                        }
                    }
                }

                if compatible {
                    new_results.push(new_binding);
                    extended = true;
                }
            }
        }

        // If no optional pattern matched, keep original binding
        if !extended {
            new_results.push(binding);
        }
    }

    Ok(new_results)
}

/// Extract UNION groups from WHERE clause
pub fn extract_union_groups(sparql: &str) -> Result<Option<UnionGroup>> {
    if !has_union(sparql) {
        return Ok(None);
    }

    if let Some(where_start) = sparql.to_uppercase().find("WHERE") {
        let where_clause = &sparql[where_start + 5..];

        if let Some(start_brace) = where_clause.find('{') {
            if let Some(end_brace) = find_matching_brace(where_clause, start_brace) {
                let content = &where_clause[start_brace + 1..end_brace];

                // Split by UNION keyword
                let mut branches = Vec::new();
                let mut current_branch = String::new();

                let mut pos = 0;
                while pos < content.len() {
                    if let Some(union_pos) = content[pos..].to_uppercase().find(" UNION ") {
                        let abs_pos = pos + union_pos;
                        current_branch.push_str(&content[pos..abs_pos]);

                        // Parse the branch we accumulated
                        if let Some(branch) = parse_union_branch(&current_branch)? {
                            branches.push(branch);
                        }

                        current_branch.clear();
                        pos = abs_pos + 7; // Skip " UNION "
                    } else {
                        // Last branch
                        current_branch.push_str(&content[pos..]);
                        break;
                    }
                }

                // Parse final branch
                if !current_branch.trim().is_empty() {
                    if let Some(branch) = parse_union_branch(&current_branch)? {
                        branches.push(branch);
                    }
                }

                if !branches.is_empty() {
                    return Ok(Some(UnionGroup { branches }));
                }
            }
        }
    }

    Ok(None)
}

/// Parse a single UNION branch
pub fn parse_union_branch(branch_text: &str) -> Result<Option<Vec<PatternGroup>>> {
    let branch_text = branch_text.trim();

    // Branch can be either { pattern } or just pattern
    let pattern_text = if branch_text.starts_with('{') && branch_text.ends_with('}') {
        &branch_text[1..branch_text.len() - 1]
    } else {
        branch_text
    };

    let mut groups = Vec::new();

    // Check for OPTIONAL in the branch
    if pattern_text.to_uppercase().contains("OPTIONAL") {
        // Parse with OPTIONAL support
        let mut pos = 0;
        let mut required_patterns = Vec::new();

        while pos < pattern_text.len() {
            if let Some(opt_pos) = pattern_text[pos..].to_uppercase().find("OPTIONAL") {
                let abs_pos = pos + opt_pos;

                // Add required patterns before OPTIONAL
                let before_optional = &pattern_text[pos..abs_pos];
                if let Some(req_pattern) = parse_simple_pattern(before_optional) {
                    required_patterns.push(req_pattern);
                }

                // Find OPTIONAL block
                let after_optional = &pattern_text[abs_pos + 8..];
                if let Some(opt_brace) = after_optional.find('{') {
                    if let Some(opt_end) = find_matching_brace(after_optional, opt_brace) {
                        let optional_content = &after_optional[opt_brace + 1..opt_end];

                        if let Some(opt_pattern) = parse_simple_pattern(optional_content) {
                            groups.push(PatternGroup {
                                patterns: vec![opt_pattern],
                                optional: true,
                            });
                        }

                        pos = abs_pos + 8 + opt_end + 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                // No more OPTIONAL
                if let Some(req_pattern) = parse_simple_pattern(&pattern_text[pos..]) {
                    required_patterns.push(req_pattern);
                }
                break;
            }
        }

        if !required_patterns.is_empty() {
            groups.push(PatternGroup {
                patterns: required_patterns,
                optional: false,
            });
        }
    } else {
        // No OPTIONAL - simple pattern
        if let Some(pattern) = parse_simple_pattern(pattern_text) {
            groups.push(PatternGroup {
                patterns: vec![pattern],
                optional: false,
            });
        }
    }

    if groups.is_empty() {
        Ok(None)
    } else {
        Ok(Some(groups))
    }
}

/// Execute a SELECT query with UNION (needs to be implemented in RdfStore)
/// This is a placeholder - actual implementation stays in RdfStore
pub fn execute_union_query_placeholder() {
    // This function signature is here for reference
    // The actual execute_union_query must stay in RdfStore because it needs access to self
}
