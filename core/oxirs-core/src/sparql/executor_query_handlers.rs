//! SPARQL query handler implementations for [`QueryExecutor`].
//!
//! This module provides all `execute_*` methods and the supporting helper methods
//! for the [`QueryExecutor`] type, split out to keep individual files under 2000 lines.

use super::aggregates::{apply_aggregates, extract_aggregates, AggregateExpression};
use super::executor::{QueryExecutor, ValuesClause};
use super::expressions::{apply_bind_expressions, extract_bind_expressions, BindExpression};
use super::filter::{evaluate_filters, extract_filter_expressions, FilterExpression};
use super::modifiers::OrderBy;
use super::parser::extract_select_variables;
use super::patterns::{PatternGroup, SimpleTriplePattern, UnionGroup};
use super::query_locator::{locate_where_brace, locate_where_group};
use crate::model::*;
use crate::rdf_store::{OxirsQueryResults, StorageBackend, VariableBinding};
use crate::Result;

impl<'a> QueryExecutor<'a> {
    /// Execute a SELECT query
    pub(super) fn execute_select_query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        // Basic pattern matching for simple SELECT queries
        // Pattern: SELECT ?var WHERE { ?s ?p ?o } LIMIT n OFFSET m
        //
        // The `WHERE` keyword is optional per SPARQL 1.1
        // (`WhereClause ::= 'WHERE'? GroupGraphPattern`); locate the graph
        // pattern group whether or not it is spelled.

        if locate_where_brace(sparql).is_some() {
            // Check for DISTINCT modifier
            let has_distinct = sparql.to_uppercase().contains("SELECT DISTINCT");

            let mut variables = self.extract_select_variables(sparql)?;

            // Check for UNION - handle separately
            if let Some(union_groups) = self.extract_union_groups(sparql)? {
                return self.execute_union_query(sparql, variables, has_distinct, union_groups);
            }

            let pattern_groups = self.extract_pattern_groups(sparql)?;

            // Track pattern matching operations
            let total_patterns: usize = pattern_groups.iter().map(|g| g.patterns.len()).sum();
            for _ in 0..total_patterns {
                self.pattern_counter.inc();
            }

            // Extract all patterns for variable detection
            let all_patterns: Vec<&SimpleTriplePattern> =
                pattern_groups.iter().flat_map(|g| &g.patterns).collect();

            // Handle SELECT * - extract variables from pattern
            if variables.len() == 1 && variables[0] == "*" {
                variables.clear();
                for pattern in &all_patterns {
                    if let Some(var) = &pattern.subject {
                        if let Some(var_name) = var.strip_prefix('?') {
                            if !variables.contains(&var_name.to_string()) {
                                variables.push(var_name.to_string());
                            }
                        }
                    }
                    if let Some(var) = &pattern.predicate {
                        if let Some(var_name) = var.strip_prefix('?') {
                            if !variables.contains(&var_name.to_string()) {
                                variables.push(var_name.to_string());
                            }
                        }
                    }
                    if let Some(var) = &pattern.object {
                        if let Some(var_name) = var.strip_prefix('?') {
                            if !variables.contains(&var_name.to_string()) {
                                variables.push(var_name.to_string());
                            }
                        }
                    }
                }
            }

            // Separate required and optional patterns
            let required_groups: Vec<&PatternGroup> =
                pattern_groups.iter().filter(|g| !g.optional).collect();
            let optional_groups: Vec<&PatternGroup> =
                pattern_groups.iter().filter(|g| g.optional).collect();

            let mut results = Vec::new();

            // Check for VALUES clause
            let values_clause = self.extract_values_clause(sparql)?;
            if let Some(values) = &values_clause {
                // Apply VALUES to create initial bindings
                results = self.apply_values_clause(values)?;
            }

            // Execute required patterns and join with VALUES if present
            let pattern_results = if !required_groups.is_empty() {
                let mut pattern_bindings = Vec::new();

                for group in required_groups {
                    for pattern in &group.patterns {
                        let matching_quads = self.query_quads_by_pattern(pattern)?;

                        for quad in matching_quads {
                            let mut binding = VariableBinding::new();

                            // Bind variables based on the pattern
                            if let Some(var) = &pattern.subject {
                                if let Some(var_name) = var.strip_prefix('?') {
                                    binding.bind(
                                        var_name.to_string(),
                                        Term::from(quad.subject().clone()),
                                    );
                                }
                            }

                            if let Some(var) = &pattern.predicate {
                                if let Some(var_name) = var.strip_prefix('?') {
                                    binding.bind(
                                        var_name.to_string(),
                                        Term::from(quad.predicate().clone()),
                                    );
                                }
                            }

                            if let Some(var) = &pattern.object {
                                if let Some(var_name) = var.strip_prefix('?') {
                                    binding.bind(
                                        var_name.to_string(),
                                        Term::from(quad.object().clone()),
                                    );
                                }
                            }

                            pattern_bindings.push(binding);
                        }
                    }
                }
                pattern_bindings
            } else {
                Vec::new()
            };

            // Join VALUES results with pattern results if both present
            if !results.is_empty() && !pattern_results.is_empty() {
                // Join: keep only bindings that are compatible
                let mut joined = Vec::new();
                for values_binding in &results {
                    for pattern_binding in &pattern_results {
                        // Check compatibility: shared variables must have same values
                        let mut compatible = true;
                        for (var, val) in values_binding.bindings.iter() {
                            if let Some(pattern_val) = pattern_binding.get(var) {
                                if val != pattern_val {
                                    compatible = false;
                                    break;
                                }
                            }
                        }

                        if compatible {
                            // Merge bindings
                            let mut merged = values_binding.clone();
                            for (var, val) in pattern_binding.bindings.iter() {
                                if !merged.bindings.contains_key(var) {
                                    merged.bind(var.clone(), val.clone());
                                }
                            }
                            joined.push(merged);
                        }
                    }
                }
                results = joined;
            } else if !pattern_results.is_empty() {
                // Only patterns, no VALUES
                results = pattern_results;
            }
            // else: Only VALUES, results already set

            // If no required patterns and no VALUES, start with empty binding for OPTIONAL
            if results.is_empty() && !optional_groups.is_empty() {
                results.push(VariableBinding::new());
            }

            // Apply optional patterns to each existing result
            for optional_group in optional_groups {
                results = self.apply_optional_patterns(results, &optional_group.patterns)?;
            }

            // Apply BIND expressions
            let bind_expressions = self.extract_bind_expressions(sparql)?;
            if !bind_expressions.is_empty() {
                results = self.apply_bind_expressions(results, &bind_expressions)?;
                // Add BIND variables to output variables
                for bind_expr in &bind_expressions {
                    if !variables.contains(&bind_expr.variable) {
                        variables.push(bind_expr.variable.clone());
                    }
                }
            }

            // Apply FILTER if present
            let filter_expressions = self.extract_filter_expressions(sparql)?;
            if !filter_expressions.is_empty() {
                results.retain(|binding| self.evaluate_filters(binding, &filter_expressions));
            }

            // Check for aggregate expressions
            let aggregates = self.extract_aggregates(sparql)?;
            if !aggregates.is_empty() {
                // Apply aggregates and replace variables with aggregate results
                let (agg_results, agg_vars) = self.apply_aggregates(results, &aggregates)?;
                return Ok(OxirsQueryResults::from_bindings(agg_results, agg_vars));
            }

            // Apply DISTINCT if present
            if has_distinct {
                results = self.remove_duplicate_bindings(results, &variables);
            }

            // Apply ORDER BY if present
            if let Some(order_by) = self.extract_order_by(sparql)? {
                self.sort_results(&mut results, &order_by);
            }

            // Apply OFFSET and LIMIT
            let offset = self.extract_offset(sparql)?;
            let limit = self.extract_limit(sparql)?;

            // Apply offset
            if offset > 0 {
                results = results.into_iter().skip(offset).collect();
            }

            // Apply limit
            if let Some(limit_value) = limit {
                results.truncate(limit_value);
            }

            Ok(OxirsQueryResults::from_bindings(results, variables))
        } else {
            Ok(OxirsQueryResults::new())
        }
    }

    /// Execute a SELECT query with UNION
    pub(super) fn execute_union_query(
        &self,
        sparql: &str,
        mut variables: Vec<String>,
        has_distinct: bool,
        union_groups: UnionGroup,
    ) -> Result<OxirsQueryResults> {
        let mut all_results = Vec::new();

        // Execute each UNION branch independently
        for branch_groups in union_groups.branches {
            // Execute this branch
            let mut branch_results = Vec::new();

            // Extract all patterns from this branch for variable detection
            let all_patterns: Vec<&SimpleTriplePattern> =
                branch_groups.iter().flat_map(|g| &g.patterns).collect();

            // Handle SELECT * - extract variables from all patterns
            if variables.len() == 1 && variables[0] == "*" {
                variables.clear();
                for pattern in &all_patterns {
                    if let Some(var) = &pattern.subject {
                        if let Some(var_name) = var.strip_prefix('?') {
                            if !variables.contains(&var_name.to_string()) {
                                variables.push(var_name.to_string());
                            }
                        }
                    }
                    if let Some(var) = &pattern.predicate {
                        if let Some(var_name) = var.strip_prefix('?') {
                            if !variables.contains(&var_name.to_string()) {
                                variables.push(var_name.to_string());
                            }
                        }
                    }
                    if let Some(var) = &pattern.object {
                        if let Some(var_name) = var.strip_prefix('?') {
                            if !variables.contains(&var_name.to_string()) {
                                variables.push(var_name.to_string());
                            }
                        }
                    }
                }
            }

            // Separate required and optional patterns for this branch
            let required_groups: Vec<&PatternGroup> =
                branch_groups.iter().filter(|g| !g.optional).collect();
            let optional_groups: Vec<&PatternGroup> =
                branch_groups.iter().filter(|g| g.optional).collect();

            // Execute required patterns first
            for group in required_groups {
                for pattern in &group.patterns {
                    let matching_quads = self.query_quads_by_pattern(pattern)?;

                    for quad in matching_quads {
                        let mut binding = VariableBinding::new();

                        // Bind variables based on the pattern
                        if let Some(var) = &pattern.subject {
                            if let Some(var_name) = var.strip_prefix('?') {
                                binding
                                    .bind(var_name.to_string(), Term::from(quad.subject().clone()));
                            }
                        }

                        if let Some(var) = &pattern.predicate {
                            if let Some(var_name) = var.strip_prefix('?') {
                                binding.bind(
                                    var_name.to_string(),
                                    Term::from(quad.predicate().clone()),
                                );
                            }
                        }

                        if let Some(var) = &pattern.object {
                            if let Some(var_name) = var.strip_prefix('?') {
                                binding
                                    .bind(var_name.to_string(), Term::from(quad.object().clone()));
                            }
                        }

                        branch_results.push(binding);
                    }
                }
            }

            // If no required patterns, start with empty binding
            if branch_results.is_empty() && !optional_groups.is_empty() {
                branch_results.push(VariableBinding::new());
            }

            // Apply optional patterns to each existing result
            for optional_group in optional_groups {
                branch_results =
                    self.apply_optional_patterns(branch_results, &optional_group.patterns)?;
            }

            // Merge branch results into all results
            all_results.extend(branch_results);
        }

        // Apply BIND expressions
        let bind_expressions = self.extract_bind_expressions(sparql)?;
        if !bind_expressions.is_empty() {
            all_results = self.apply_bind_expressions(all_results, &bind_expressions)?;
        }

        // Apply FILTER if present
        let filter_expressions = self.extract_filter_expressions(sparql)?;
        if !filter_expressions.is_empty() {
            all_results.retain(|binding| self.evaluate_filters(binding, &filter_expressions));
        }

        // Apply DISTINCT if present
        if has_distinct {
            all_results = self.remove_duplicate_bindings(all_results, &variables);
        }

        // Apply ORDER BY if present
        if let Some(order_by) = self.extract_order_by(sparql)? {
            self.sort_results(&mut all_results, &order_by);
        }

        // Apply OFFSET and LIMIT
        let offset = self.extract_offset(sparql)?;
        let limit = self.extract_limit(sparql)?;

        if offset > 0 {
            all_results = all_results.into_iter().skip(offset).collect();
        }

        if let Some(limit_value) = limit {
            all_results.truncate(limit_value);
        }

        Ok(OxirsQueryResults::from_bindings(all_results, variables))
    }

    /// Execute an ASK query
    pub(super) fn execute_ask_query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        // Basic ASK query: check if any triples match the pattern
        let triple_patterns = self.extract_triple_patterns(sparql)?;

        for pattern in triple_patterns {
            let matching_quads = self.query_quads_by_pattern(&pattern)?;
            if !matching_quads.is_empty() {
                return Ok(OxirsQueryResults::from_boolean(true));
            }
        }

        Ok(OxirsQueryResults::from_boolean(false))
    }

    /// Execute a CONSTRUCT query
    pub(super) fn execute_construct_query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        // Basic CONSTRUCT query: return matching triples
        let triple_patterns = self.extract_triple_patterns(sparql)?;
        let mut result_quads = Vec::new();

        for pattern in triple_patterns {
            let matching_quads = self.query_quads_by_pattern(&pattern)?;
            result_quads.extend(matching_quads);
        }

        Ok(OxirsQueryResults::from_graph(result_quads))
    }

    /// Execute a DESCRIBE query
    pub(super) fn execute_describe_query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        // Basic DESCRIBE query: return all triples about the specified resource
        let resources = self.extract_describe_resources(sparql)?;
        let mut result_quads = Vec::new();

        for resource in resources {
            // Find all triples where the resource is subject or object
            let subject_pattern = SimpleTriplePattern {
                subject: Some(resource.clone()),
                predicate: None,
                object: None,
            };
            let object_pattern = SimpleTriplePattern {
                subject: None,
                predicate: None,
                object: Some(resource),
            };

            let subject_quads = self.query_quads(
                subject_pattern
                    .subject
                    .as_ref()
                    .and_then(|s| Self::string_to_subject(s))
                    .as_ref(),
                subject_pattern
                    .predicate
                    .as_ref()
                    .and_then(|p| Self::string_to_predicate(p))
                    .as_ref(),
                subject_pattern
                    .object
                    .as_ref()
                    .and_then(|o| Self::string_to_object(o))
                    .as_ref(),
                None,
            )?;
            let object_quads = self.query_quads(
                object_pattern
                    .subject
                    .as_ref()
                    .and_then(|s| Self::string_to_subject(s))
                    .as_ref(),
                object_pattern
                    .predicate
                    .as_ref()
                    .and_then(|p| Self::string_to_predicate(p))
                    .as_ref(),
                object_pattern
                    .object
                    .as_ref()
                    .and_then(|o| Self::string_to_object(o))
                    .as_ref(),
                None,
            )?;

            result_quads.extend(subject_quads);
            result_quads.extend(object_quads);
        }

        Ok(OxirsQueryResults::from_graph(result_quads))
    }

    // -------------------------------------------------------------------------
    // Delegation helpers: these call free functions from sibling modules
    // (all re-exported via `super::*` through `mod.rs`)
    // -------------------------------------------------------------------------

    pub(super) fn extract_select_variables(&self, sparql: &str) -> Result<Vec<String>> {
        extract_select_variables(sparql)
    }

    pub(super) fn extract_aggregates(&self, sparql: &str) -> Result<Vec<AggregateExpression>> {
        extract_aggregates(sparql)
    }

    pub(super) fn extract_limit(&self, sparql: &str) -> Result<Option<usize>> {
        let sparql_upper = sparql.to_uppercase();

        if let Some(limit_start) = sparql_upper.find("LIMIT") {
            let after_limit = &sparql[limit_start + 5..];

            for token in after_limit.split_whitespace() {
                if let Ok(limit_value) = token.parse::<usize>() {
                    return Ok(Some(limit_value));
                }
            }
        }

        Ok(None)
    }

    pub(super) fn extract_order_by(&self, sparql: &str) -> Result<Option<OrderBy>> {
        let sparql_upper = sparql.to_uppercase();

        if let Some(order_start) = sparql_upper.find("ORDER BY") {
            let after_order = &sparql[order_start + 8..];

            let mut descending = false;
            let tokens: Vec<&str> = after_order.split_whitespace().collect();

            if tokens.is_empty() {
                return Ok(None);
            }

            let mut var_token = tokens[0];

            if tokens.len() > 1 {
                let modifier = tokens[1].to_uppercase();
                if modifier == "DESC" {
                    descending = true;
                } else if modifier == "ASC" {
                    descending = false;
                }
            } else if var_token.to_uppercase().ends_with("DESC") {
                var_token = var_token.trim_end_matches("DESC").trim_end_matches("desc");
                descending = true;
            } else if var_token.to_uppercase().ends_with("ASC") {
                var_token = var_token.trim_end_matches("ASC").trim_end_matches("asc");
            }

            if var_token.to_uppercase().starts_with("DESC(") {
                descending = true;
                var_token = var_token[5..].trim_end_matches(')');
            } else if var_token.to_uppercase().starts_with("ASC(") {
                var_token = var_token[4..].trim_end_matches(')');
            }

            let variable = var_token.trim_start_matches('?').trim().to_string();

            if !variable.is_empty() {
                return Ok(Some(OrderBy {
                    variable,
                    descending,
                }));
            }
        }

        Ok(None)
    }

    pub(super) fn remove_duplicate_bindings(
        &self,
        results: Vec<VariableBinding>,
        variables: &[String],
    ) -> Vec<VariableBinding> {
        use std::collections::HashSet;

        let mut seen = HashSet::new();
        let mut unique_results = Vec::new();

        for binding in results {
            let mut signature = String::new();
            for var in variables {
                if let Some(term) = binding.get(var) {
                    signature.push_str(&format!("{:?}|", term));
                } else {
                    signature.push_str("UNBOUND|");
                }
            }

            if seen.insert(signature) {
                unique_results.push(binding);
            }
        }

        unique_results
    }

    pub(super) fn apply_aggregates(
        &self,
        results: Vec<VariableBinding>,
        aggregates: &[AggregateExpression],
    ) -> Result<(Vec<VariableBinding>, Vec<String>)> {
        apply_aggregates(results, aggregates)
    }

    pub(super) fn sort_results(&self, results: &mut [VariableBinding], order_by: &OrderBy) {
        results.sort_by(|a, b| {
            let a_val = a.get(&order_by.variable);
            let b_val = b.get(&order_by.variable);

            let cmp = match (a_val, b_val) {
                (Some(a_term), Some(b_term)) => self.compare_terms(a_term, b_term),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            };

            if order_by.descending {
                cmp.reverse()
            } else {
                cmp
            }
        });
    }

    pub(super) fn compare_terms(&self, a: &Term, b: &Term) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        match (a, b) {
            (Term::Literal(a_lit), Term::Literal(b_lit)) => {
                let a_val = a_lit.value();
                let b_val = b_lit.value();

                if let (Ok(a_num), Ok(b_num)) = (a_val.parse::<f64>(), b_val.parse::<f64>()) {
                    if a_num < b_num {
                        Ordering::Less
                    } else if a_num > b_num {
                        Ordering::Greater
                    } else {
                        Ordering::Equal
                    }
                } else {
                    a_val.cmp(b_val)
                }
            }
            (Term::NamedNode(a_node), Term::NamedNode(b_node)) => {
                a_node.as_str().cmp(b_node.as_str())
            }
            (Term::BlankNode(a_bnode), Term::BlankNode(b_bnode)) => {
                a_bnode.as_str().cmp(b_bnode.as_str())
            }
            // Mixed types: Literals < URIs < BlankNodes
            (Term::Literal(_), _) => Ordering::Less,
            (_, Term::Literal(_)) => Ordering::Greater,
            (Term::NamedNode(_), Term::BlankNode(_)) => Ordering::Less,
            (Term::BlankNode(_), Term::NamedNode(_)) => Ordering::Greater,
            _ => Ordering::Equal,
        }
    }

    pub(super) fn extract_offset(&self, sparql: &str) -> Result<usize> {
        let sparql_upper = sparql.to_uppercase();

        if let Some(offset_start) = sparql_upper.find("OFFSET") {
            let after_offset = &sparql[offset_start + 6..];

            for token in after_offset.split_whitespace() {
                if let Ok(offset_value) = token.parse::<usize>() {
                    return Ok(offset_value);
                }
            }
        }

        Ok(0)
    }

    pub(super) fn extract_bind_expressions(&self, sparql: &str) -> Result<Vec<BindExpression>> {
        extract_bind_expressions(sparql)
    }

    pub(super) fn apply_bind_expressions(
        &self,
        results: Vec<VariableBinding>,
        binds: &[BindExpression],
    ) -> Result<Vec<VariableBinding>> {
        apply_bind_expressions(results, binds)
    }

    pub(super) fn extract_values_clause(&self, sparql: &str) -> Result<Option<ValuesClause>> {
        let sparql_upper = sparql.to_uppercase();

        if let Some(values_start) = sparql_upper.find("VALUES") {
            let after_values = &sparql[values_start + 6..].trim_start();

            if let Some(paren_start) = after_values.find('(') {
                if let Some(paren_end) = after_values[paren_start..].find(')') {
                    let var_list = &after_values[paren_start + 1..paren_start + paren_end];
                    let variables: Vec<String> = var_list
                        .split_whitespace()
                        .filter_map(|v| v.strip_prefix('?'))
                        .map(|v| v.to_string())
                        .collect();

                    if variables.is_empty() {
                        return Ok(None);
                    }

                    let after_vars = &after_values[paren_start + paren_end + 1..].trim_start();
                    if let Some(brace_start) = after_vars.find('{') {
                        if let Some(brace_end) = after_vars[brace_start..].rfind('}') {
                            let data_block = &after_vars[brace_start + 1..brace_start + brace_end];

                            let mut rows = Vec::new();
                            let mut current_pos = 0;

                            while current_pos < data_block.len() {
                                if let Some(row_start) = data_block[current_pos..].find('(') {
                                    let abs_row_start = current_pos + row_start;
                                    if let Some(row_end) = data_block[abs_row_start..].find(')') {
                                        let row_content =
                                            &data_block[abs_row_start + 1..abs_row_start + row_end];

                                        let values: Vec<String> = row_content
                                            .split_whitespace()
                                            .map(|v| {
                                                if (v.starts_with('"') && v.ends_with('"'))
                                                    || (v.starts_with('\'') && v.ends_with('\''))
                                                {
                                                    v[1..v.len() - 1].to_string()
                                                } else {
                                                    v.to_string()
                                                }
                                            })
                                            .collect();

                                        if values.len() == variables.len() {
                                            rows.push(values);
                                        }

                                        current_pos = abs_row_start + row_end + 1;
                                    } else {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }

                            if !rows.is_empty() {
                                return Ok(Some(ValuesClause { variables, rows }));
                            }
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    pub(super) fn apply_values_clause(
        &self,
        values: &ValuesClause,
    ) -> Result<Vec<VariableBinding>> {
        let mut bindings = Vec::new();

        for row in &values.rows {
            let mut binding = VariableBinding::new();

            for (idx, var_name) in values.variables.iter().enumerate() {
                if idx < row.len() {
                    let value_str = &row[idx];

                    let term = if value_str.starts_with('<') && value_str.ends_with('>') {
                        let uri = &value_str[1..value_str.len() - 1];
                        Term::from(NamedNode::new_unchecked(uri))
                    } else {
                        Term::from(Literal::new(value_str.clone()))
                    };

                    binding.bind(var_name.clone(), term);
                }
            }

            bindings.push(binding);
        }

        Ok(bindings)
    }

    pub(super) fn extract_filter_expressions(&self, sparql: &str) -> Result<Vec<FilterExpression>> {
        extract_filter_expressions(sparql)
    }

    pub(super) fn evaluate_filters(
        &self,
        binding: &VariableBinding,
        filters: &[FilterExpression],
    ) -> bool {
        evaluate_filters(binding, filters)
    }

    pub(super) fn apply_optional_patterns(
        &self,
        current_results: Vec<VariableBinding>,
        optional_patterns: &[SimpleTriplePattern],
    ) -> Result<Vec<VariableBinding>> {
        let mut new_results = Vec::new();

        for binding in current_results {
            let mut extended = false;

            for pattern in optional_patterns {
                let matching_quads = self.query_quads_by_pattern(pattern)?;

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
                                    new_binding.bind(
                                        var_name.to_string(),
                                        Term::from(quad.object().clone()),
                                    );
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

            if !extended {
                new_results.push(binding);
            }
        }

        Ok(new_results)
    }

    pub(super) fn extract_pattern_groups(&self, sparql: &str) -> Result<Vec<PatternGroup>> {
        let mut groups = Vec::new();

        // Locate the graph pattern group, honoring the optional `WHERE` keyword
        // (`WhereClause ::= 'WHERE'? GroupGraphPattern`).
        if let Some((where_open, where_close)) = locate_where_group(sparql) {
            let pattern_text = &sparql[where_open + 1..where_close];

            let sparql_upper = pattern_text.to_uppercase();
            if sparql_upper.contains("OPTIONAL") {
                let mut pos = 0;
                let mut required_patterns = Vec::new();

                while pos < pattern_text.len() {
                    if let Some(opt_pos) = pattern_text[pos..].to_uppercase().find("OPTIONAL") {
                        let abs_pos = pos + opt_pos;

                        let before_optional = &pattern_text[pos..abs_pos];
                        if let Some(req_pattern) = self.parse_simple_pattern(before_optional) {
                            required_patterns.push(req_pattern);
                        }

                        let after_optional = &pattern_text[abs_pos + 8..];
                        if let Some(opt_brace) = after_optional.find('{') {
                            if let Some(opt_end) =
                                self.find_matching_brace(after_optional, opt_brace)
                            {
                                let optional_content = &after_optional[opt_brace + 1..opt_end];

                                if let Some(opt_pattern) =
                                    self.parse_simple_pattern(optional_content)
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
                        if let Some(req_pattern) = self.parse_simple_pattern(&pattern_text[pos..]) {
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
                let patterns = self.extract_triple_patterns(sparql)?;
                if !patterns.is_empty() {
                    groups.push(PatternGroup {
                        patterns,
                        optional: false,
                    });
                }
            }
        }

        if groups.is_empty() {
            let patterns = self.extract_triple_patterns(sparql)?;
            if !patterns.is_empty() {
                groups.push(PatternGroup {
                    patterns,
                    optional: false,
                });
            }
        }

        Ok(groups)
    }

    pub(super) fn find_matching_brace(&self, text: &str, start_pos: usize) -> Option<usize> {
        let mut brace_count = 1;
        let chars: Vec<char> = text.chars().collect();

        for (i, &ch) in chars.iter().enumerate().skip(start_pos + 1) {
            if ch == '{' {
                brace_count += 1;
            } else if ch == '}' {
                brace_count -= 1;
                if brace_count == 0 {
                    return Some(i);
                }
            }
        }

        None
    }

    pub(super) fn parse_simple_pattern(&self, text: &str) -> Option<SimpleTriplePattern> {
        let tokens: Vec<&str> = text
            .split_whitespace()
            .filter(|t| !t.is_empty() && *t != "." && *t != ";")
            .collect();

        if tokens.len() >= 3 {
            Some(SimpleTriplePattern {
                subject: Some(tokens[0].to_string()),
                predicate: Some(tokens[1].to_string()),
                object: Some(tokens[2].to_string()),
            })
        } else {
            None
        }
    }

    pub(super) fn has_union(&self, sparql: &str) -> bool {
        sparql.to_uppercase().contains("UNION")
    }

    pub(super) fn extract_union_groups(&self, sparql: &str) -> Result<Option<UnionGroup>> {
        if !self.has_union(sparql) {
            return Ok(None);
        }

        // Locate the graph pattern group, honoring the optional `WHERE` keyword.
        if let Some((where_open, where_close)) = locate_where_group(sparql) {
            let pattern_text = &sparql[where_open + 1..where_close];

            let mut branches = Vec::new();
            let mut current_pos = 0;
            let pattern_upper = pattern_text.to_uppercase();

            let mut union_positions = Vec::new();
            let mut search_pos = 0;
            while let Some(pos) = pattern_upper[search_pos..].find("UNION") {
                let abs_pos = search_pos + pos;
                union_positions.push(abs_pos);
                search_pos = abs_pos + 5;
            }

            for &union_pos in &union_positions {
                let branch_text = &pattern_text[current_pos..union_pos];
                if let Some(branch_groups) = self.parse_union_branch(branch_text)? {
                    branches.push(branch_groups);
                }
                current_pos = union_pos + 5;
            }

            if current_pos < pattern_text.len() {
                let branch_text = &pattern_text[current_pos..];
                if let Some(branch_groups) = self.parse_union_branch(branch_text)? {
                    branches.push(branch_groups);
                }
            }

            if !branches.is_empty() {
                return Ok(Some(UnionGroup { branches }));
            }
        }

        Ok(None)
    }

    pub(super) fn parse_union_branch(
        &self,
        branch_text: &str,
    ) -> Result<Option<Vec<PatternGroup>>> {
        let trimmed = branch_text.trim();

        let content = if trimmed.starts_with('{') && trimmed.ends_with('}') {
            &trimmed[1..trimmed.len() - 1]
        } else {
            trimmed
        };

        let temp_query = format!("SELECT * WHERE {{ {} }}", content);
        let groups = self.extract_pattern_groups(&temp_query)?;

        if groups.is_empty() {
            Ok(None)
        } else {
            Ok(Some(groups))
        }
    }

    pub(super) fn extract_triple_patterns(&self, sparql: &str) -> Result<Vec<SimpleTriplePattern>> {
        let mut patterns = Vec::new();

        // Locate the graph pattern group, honoring the optional `WHERE` keyword.
        // For CONSTRUCT queries, `locate_where_group` yields the pattern group
        // (the group that follows the template) rather than the template itself.
        if let Some((where_open, where_close)) = locate_where_group(sparql) {
            let pattern_text = &sparql[where_open + 1..where_close];
            let tokens: Vec<&str> = pattern_text.split_whitespace().collect();

            if tokens.len() >= 3 {
                let subject = if tokens[0] == "." {
                    None
                } else {
                    Some(tokens[0].to_string())
                };
                let predicate = if tokens[1] == "." {
                    None
                } else {
                    Some(tokens[1].to_string())
                };
                let object = if tokens[2] == "." {
                    None
                } else {
                    Some(tokens[2].to_string())
                };

                patterns.push(SimpleTriplePattern {
                    subject,
                    predicate,
                    object,
                });
            }
        }

        Ok(patterns)
    }

    pub(super) fn extract_describe_resources(&self, sparql: &str) -> Result<Vec<String>> {
        let mut resources = Vec::new();

        if let Some(describe_start) = sparql.to_uppercase().find("DESCRIBE") {
            let describe_clause = &sparql[describe_start + 8..];

            for token in describe_clause.split_whitespace() {
                if (token.starts_with('<') && token.ends_with('>')) || token.starts_with('?') {
                    resources.push(token.to_string());
                }

                if token.to_uppercase() == "WHERE" {
                    break;
                }
            }
        }

        Ok(resources)
    }

    pub(super) fn query_quads_by_pattern(
        &self,
        pattern: &SimpleTriplePattern,
    ) -> Result<Vec<Quad>> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                let results_vec = index.find_quads(
                    pattern
                        .subject
                        .as_ref()
                        .and_then(|s| Self::string_to_subject(s))
                        .as_ref(),
                    pattern
                        .predicate
                        .as_ref()
                        .and_then(|p| Self::string_to_predicate(p))
                        .as_ref(),
                    pattern
                        .object
                        .as_ref()
                        .and_then(|o| Self::string_to_object(o))
                        .as_ref(),
                    None,
                );
                Ok(results_vec)
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage.read().expect("storage lock should not be poisoned");

                // Narrow via the interned permutation indexes for every bound
                // component that parses *faithfully* into a term (one whose
                // `to_string()` round-trips to the pattern text). Passing such a
                // term to `query_quads` picks the best permutation and, when the
                // term is not interned at all, short-circuits to an empty result.
                // A bound component that does not round-trip (e.g. a language- or
                // datatype-tagged literal the simple parser cannot reconstruct)
                // is left unbound for indexing and enforced only by the string
                // filter below, so the overall result is byte-for-byte identical
                // to the previous full-set string scan.
                let subject_term =
                    Self::bound_component(&pattern.subject).and_then(Self::faithful_subject);
                let predicate_term =
                    Self::bound_component(&pattern.predicate).and_then(Self::faithful_predicate);
                let object_term =
                    Self::bound_component(&pattern.object).and_then(Self::faithful_object);

                let candidates = storage.query_quads(
                    subject_term.as_ref(),
                    predicate_term.as_ref(),
                    object_term.as_ref(),
                    None,
                );

                // Re-apply every bound component as an exact string match. This
                // is redundant for faithfully-indexed columns (cheap) and is the
                // sole enforcement for non-round-tripping ones. The graph is
                // never constrained here, matching the prior behavior.
                let string_matches = |bound: Option<&str>, actual: String| match bound {
                    Some(text) => actual == text,
                    None => true,
                };
                let results = candidates
                    .into_iter()
                    .filter(|quad| {
                        string_matches(
                            Self::bound_component(&pattern.subject),
                            quad.subject().to_string(),
                        ) && string_matches(
                            Self::bound_component(&pattern.predicate),
                            quad.predicate().to_string(),
                        ) && string_matches(
                            Self::bound_component(&pattern.object),
                            quad.object().to_string(),
                        )
                    })
                    .collect();

                Ok(results)
            }
        }
    }

    /// A bound (non-variable) pattern component, or `None` when the component is
    /// absent or a `?`-prefixed variable.
    fn bound_component(component: &Option<String>) -> Option<&str> {
        match component {
            Some(value) if !value.starts_with('?') => Some(value.as_str()),
            _ => None,
        }
    }

    /// Parse a bound subject into a term usable for index selection, but only if
    /// it round-trips (`term.to_string() == text`). Non-round-tripping forms
    /// return `None` so they are enforced purely by the string filter.
    fn faithful_subject(text: &str) -> Option<Subject> {
        Self::string_to_subject(text).filter(|term| term.to_string() == text)
    }

    /// Parse a bound predicate for index selection if it round-trips.
    fn faithful_predicate(text: &str) -> Option<Predicate> {
        Self::string_to_predicate(text).filter(|term| term.to_string() == text)
    }

    /// Parse a bound object for index selection if it round-trips.
    fn faithful_object(text: &str) -> Option<Object> {
        Self::string_to_object(text).filter(|term| term.to_string() == text)
    }

    pub(super) fn query_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Result<Vec<Quad>> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                Ok(index.find_quads(subject, predicate, object, graph_name))
            }
            StorageBackend::Memory(storage) => {
                let storage = storage.read().expect("storage lock should not be poisoned");
                Ok(storage.query_quads(subject, predicate, object, graph_name))
            }
            StorageBackend::Persistent(storage, _) => {
                let storage = storage.read().expect("storage lock should not be poisoned");
                Ok(storage.query_quads(subject, predicate, object, graph_name))
            }
        }
    }

    pub(super) fn string_to_subject(s: &str) -> Option<Subject> {
        if s.starts_with('<') && s.ends_with('>') {
            Some(Subject::NamedNode(NamedNode::new_unchecked(
                &s[1..s.len() - 1],
            )))
        } else if s.starts_with('_') {
            Some(Subject::BlankNode(BlankNode::new_unchecked(s)))
        } else {
            None
        }
    }

    pub(super) fn string_to_predicate(p: &str) -> Option<Predicate> {
        if p.starts_with('<') && p.ends_with('>') {
            Some(Predicate::NamedNode(NamedNode::new_unchecked(
                &p[1..p.len() - 1],
            )))
        } else {
            None
        }
    }

    pub(super) fn string_to_object(o: &str) -> Option<Object> {
        if o.starts_with('<') && o.ends_with('>') {
            Some(Object::NamedNode(NamedNode::new_unchecked(
                &o[1..o.len() - 1],
            )))
        } else if o.starts_with('_') {
            Some(Object::BlankNode(BlankNode::new_unchecked(o)))
        } else if o.starts_with('"') {
            Some(Object::Literal(Literal::new(o.trim_matches('"'))))
        } else {
            None
        }
    }
}
