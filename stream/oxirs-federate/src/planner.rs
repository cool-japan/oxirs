//! Query Planning and Decomposition
//!
//! This module handles the analysis of queries and creates execution plans
//! for federated query processing across multiple services.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Duration;
use tracing::{debug, info, warn};
use sparql_syntax::{Query, ast, parser::ParseError};

use crate::{FederatedService, ServiceRegistry, ServiceCapability};

/// Query planner for federated queries
#[derive(Debug)]
pub struct QueryPlanner {
    config: QueryPlannerConfig,
}

impl QueryPlanner {
    /// Create a new query planner with default configuration
    pub fn new() -> Self {
        Self {
            config: QueryPlannerConfig::default(),
        }
    }

    /// Create a new query planner with custom configuration
    pub fn with_config(config: QueryPlannerConfig) -> Self {
        Self { config }
    }

    /// Analyze a SPARQL query and extract planning information
    pub async fn analyze_sparql(&self, query: &str) -> Result<QueryInfo> {
        debug!("Analyzing SPARQL query: {}", query);

        // Parse the SPARQL query using proper parser
        let parsed_query = Query::parse(query, None)
            .map_err(|e| anyhow!("Failed to parse SPARQL query: {}", e))?;

        let query_type = self.determine_query_type(&parsed_query);
        let patterns = self.extract_patterns_from_ast(&parsed_query)?;
        let service_clauses = self.extract_service_clauses_from_ast(&parsed_query)?;
        let filters = self.extract_filters_from_ast(&parsed_query)?;
        let variables = self.extract_variables_from_ast(&parsed_query)?;
        let complexity = self.calculate_complexity(&patterns, &filters, &service_clauses);

        Ok(QueryInfo {
            query_type,
            original_query: query.to_string(),
            patterns,
            service_clauses,
            filters,
            variables,
            complexity,
            estimated_cost: self.estimate_query_cost(&patterns, &service_clauses),
        })
    }

    /// Analyze a GraphQL query and extract planning information
    pub async fn analyze_graphql(&self, query: &str, variables: Option<&serde_json::Value>) -> Result<QueryInfo> {
        debug!("Analyzing GraphQL query: {}", query);

        // Parse GraphQL query structure
        let query_type = QueryType::GraphQLQuery; // TODO: Detect mutations/subscriptions
        let selections = self.extract_graphql_selections(query)?;
        let graphql_variables = variables.cloned().unwrap_or(serde_json::Value::Null);
        
        // Convert GraphQL selections to patterns for planning
        let patterns = self.graphql_to_patterns(&selections)?;
        let complexity = self.calculate_graphql_complexity(&selections);

        Ok(QueryInfo {
            query_type,
            original_query: query.to_string(),
            patterns,
            service_clauses: Vec::new(), // GraphQL doesn't have SERVICE clauses
            filters: Vec::new(), // TODO: Extract GraphQL filters/arguments
            variables: HashSet::new(), // TODO: Extract GraphQL variables
            complexity,
            estimated_cost: self.estimate_graphql_cost(&selections),
        })
    }

    /// Create an execution plan for a SPARQL query
    pub async fn plan_sparql(&self, query_info: &QueryInfo, registry: &ServiceRegistry) -> Result<ExecutionPlan> {
        info!("Planning SPARQL execution for {} patterns", query_info.patterns.len());

        let mut plan = ExecutionPlan {
            query_id: uuid::Uuid::new_v4().to_string(),
            query_type: query_info.query_type,
            steps: Vec::new(),
            estimated_duration: Duration::from_secs(0),
            parallelizable_steps: Vec::new(),
            dependencies: HashMap::new(),
        };

        // Handle explicit SERVICE clauses first
        for service_clause in &query_info.service_clauses {
            let step = self.create_service_step(service_clause, registry)?;
            plan.steps.push(step);
        }

        // Group remaining patterns by compatible services
        let remaining_patterns: Vec<_> = query_info.patterns.iter()
            .filter(|pattern| !self.is_pattern_in_service_clause(pattern, &query_info.service_clauses))
            .collect();

        if !remaining_patterns.is_empty() {
            let service_assignments = self.assign_patterns_to_services(&remaining_patterns, registry)?;
            
            for (service_id, patterns) in service_assignments {
                let step = ExecutionStep {
                    step_id: uuid::Uuid::new_v4().to_string(),
                    step_type: StepType::ServiceQuery,
                    service_id: Some(service_id),
                    query_fragment: self.build_sparql_fragment(&patterns),
                    expected_variables: self.extract_pattern_variables(&patterns),
                    estimated_duration: Duration::from_millis(100),
                    dependencies: Vec::new(),
                    parallel_group: None,
                };
                plan.steps.push(step);
            }
        }

        // Add join steps if multiple services are involved
        if plan.steps.len() > 1 {
            let join_step = ExecutionStep {
                step_id: uuid::Uuid::new_v4().to_string(),
                step_type: StepType::Join,
                service_id: None,
                query_fragment: "-- Join results from multiple services".to_string(),
                expected_variables: query_info.variables.clone(),
                estimated_duration: Duration::from_millis(50),
                dependencies: plan.steps.iter().map(|s| s.step_id.clone()).collect(),
                parallel_group: None,
            };
            plan.steps.push(join_step);
        }

        self.optimize_plan(&mut plan);
        Ok(plan)
    }

    /// Create an execution plan for a GraphQL query
    pub async fn plan_graphql(&self, query_info: &QueryInfo, registry: &ServiceRegistry) -> Result<ExecutionPlan> {
        info!("Planning GraphQL execution");

        let mut plan = ExecutionPlan {
            query_id: uuid::Uuid::new_v4().to_string(),
            query_type: query_info.query_type,
            steps: Vec::new(),
            estimated_duration: Duration::from_secs(0),
            parallelizable_steps: Vec::new(),
            dependencies: HashMap::new(),
        };

        // Find GraphQL services that can handle this query
        let graphql_services: Vec<_> = registry
            .get_services_with_capability(&ServiceCapability::GraphQLQuery)
            .collect();

        if graphql_services.is_empty() {
            return Err(anyhow!("No GraphQL services available for federation"));
        }

        // For now, create a single step per GraphQL service
        // TODO: Implement more sophisticated GraphQL federation logic
        for service in graphql_services {
            let step = ExecutionStep {
                step_id: uuid::Uuid::new_v4().to_string(),
                step_type: StepType::GraphQLQuery,
                service_id: Some(service.id.clone()),
                query_fragment: query_info.original_query.clone(),
                expected_variables: HashSet::new(),
                estimated_duration: Duration::from_millis(200),
                dependencies: Vec::new(),
                parallel_group: Some(0), // GraphQL services can be queried in parallel
            };
            plan.steps.push(step);
        }

        // Add schema stitching step if multiple services
        if plan.steps.len() > 1 {
            let stitch_step = ExecutionStep {
                step_id: uuid::Uuid::new_v4().to_string(),
                step_type: StepType::SchemaStitch,
                service_id: None,
                query_fragment: "-- Stitch GraphQL schemas".to_string(),
                expected_variables: HashSet::new(),
                estimated_duration: Duration::from_millis(30),
                dependencies: plan.steps.iter().map(|s| s.step_id.clone()).collect(),
                parallel_group: None,
            };
            plan.steps.push(stitch_step);
        }

        self.optimize_plan(&mut plan);
        Ok(plan)
    }

    // Helper methods for query analysis

    fn determine_query_type(&self, query: &Query) -> QueryType {
        match query {
            Query::Select { .. } => QueryType::SparqlSelect,
            Query::Construct { .. } => QueryType::SparqlConstruct,
            Query::Ask { .. } => QueryType::SparqlAsk,
            Query::Describe { .. } => QueryType::SparqlDescribe,
            Query::Update { .. } => QueryType::SparqlUpdate,
        }
    }

    fn extract_patterns_from_ast(&self, query: &Query) -> Result<Vec<TriplePattern>> {
        let mut patterns = Vec::new();
        
        match query {
            Query::Select { pattern, .. } |
            Query::Construct { pattern, .. } |
            Query::Ask { pattern, .. } |
            Query::Describe { pattern, .. } => {
                self.collect_patterns_from_graph_pattern(pattern, &mut patterns)?;
            }
            Query::Update { .. } => {
                // Handle update patterns - simplified for now
            }
        }
        
        Ok(patterns)
    }

    fn collect_patterns_from_graph_pattern(
        &self,
        pattern: &ast::GraphPattern,
        patterns: &mut Vec<TriplePattern>
    ) -> Result<()> {
        match pattern {
            ast::GraphPattern::Bgp { patterns: triples } => {
                for triple in triples {
                    patterns.push(TriplePattern {
                        subject: self.term_to_string(&triple.subject),
                        predicate: self.term_to_string(&triple.predicate),
                        object: self.term_to_string(&triple.object),
                        pattern_string: format!("{} {} {}", 
                            self.term_to_string(&triple.subject),
                            self.term_to_string(&triple.predicate),
                            self.term_to_string(&triple.object)
                        ),
                    });
                }
            }
            ast::GraphPattern::Join { left, right } => {
                self.collect_patterns_from_graph_pattern(left, patterns)?;
                self.collect_patterns_from_graph_pattern(right, patterns)?;
            }
            ast::GraphPattern::Union { left, right } => {
                self.collect_patterns_from_graph_pattern(left, patterns)?;
                self.collect_patterns_from_graph_pattern(right, patterns)?;
            }
            ast::GraphPattern::LeftJoin { left, right, .. } => {
                self.collect_patterns_from_graph_pattern(left, patterns)?;
                self.collect_patterns_from_graph_pattern(right, patterns)?;
            }
            ast::GraphPattern::Filter { expr: _, inner } => {
                self.collect_patterns_from_graph_pattern(inner, patterns)?;
            }
            ast::GraphPattern::Service { name: _, inner, .. } => {
                self.collect_patterns_from_graph_pattern(inner, patterns)?;
            }
            ast::GraphPattern::Group { inner, .. } => {
                self.collect_patterns_from_graph_pattern(inner, patterns)?;
            }
            ast::GraphPattern::Graph { name: _, inner } => {
                self.collect_patterns_from_graph_pattern(inner, patterns)?;
            }
            ast::GraphPattern::Extend { inner, .. } => {
                self.collect_patterns_from_graph_pattern(inner, patterns)?;
            }
            ast::GraphPattern::Minus { left, right } => {
                self.collect_patterns_from_graph_pattern(left, patterns)?;
                self.collect_patterns_from_graph_pattern(right, patterns)?;
            }
            ast::GraphPattern::Values { .. } => {
                // Handle VALUES clauses if needed
            }
            ast::GraphPattern::OrderBy { inner, .. } => {
                self.collect_patterns_from_graph_pattern(inner, patterns)?;
            }
            ast::GraphPattern::Project { inner, .. } => {
                self.collect_patterns_from_graph_pattern(inner, patterns)?;
            }
            ast::GraphPattern::Distinct { inner } => {
                self.collect_patterns_from_graph_pattern(inner, patterns)?;
            }
            ast::GraphPattern::Reduced { inner } => {
                self.collect_patterns_from_graph_pattern(inner, patterns)?;
            }
            ast::GraphPattern::Slice { inner, .. } => {
                self.collect_patterns_from_graph_pattern(inner, patterns)?;
            }
        }
        Ok(())
    }

    fn term_to_string(&self, term: &ast::TermPattern) -> String {
        match term {
            ast::TermPattern::NamedNode(node) => format!("<{}>", node),
            ast::TermPattern::BlankNode(bnode) => format!("_:{}", bnode),
            ast::TermPattern::Literal(literal) => format!("\"{}\"", literal.value()),
            ast::TermPattern::Variable(var) => format!("?{}", var),
        }
    }

    fn extract_service_clauses_from_ast(&self, query: &Query) -> Result<Vec<ServiceClause>> {
        let mut services = Vec::new();
        
        match query {
            Query::Select { pattern, .. } |
            Query::Construct { pattern, .. } |
            Query::Ask { pattern, .. } |
            Query::Describe { pattern, .. } => {
                self.collect_service_clauses_from_pattern(pattern, &mut services)?;
            }
            Query::Update { .. } => {
                // Handle update service clauses if needed
            }
        }
        
        Ok(services)
    }

    fn collect_service_clauses_from_pattern(
        &self,
        pattern: &ast::GraphPattern,
        services: &mut Vec<ServiceClause>
    ) -> Result<()> {
        match pattern {
            ast::GraphPattern::Service { name, inner, silent } => {
                let service_url = match name {
                    ast::TermPattern::NamedNode(node) => node.to_string(),
                    ast::TermPattern::Variable(var) => format!("?{}", var),
                    _ => "unknown".to_string(),
                };
                
                // Convert inner pattern back to SPARQL string (simplified)
                let subquery = format!("SELECT * WHERE {{ {} }}", "?s ?p ?o"); // Placeholder
                
                services.push(ServiceClause {
                    service_url,
                    subquery,
                    silent: *silent,
                });
            }
            ast::GraphPattern::Join { left, right } => {
                self.collect_service_clauses_from_pattern(left, services)?;
                self.collect_service_clauses_from_pattern(right, services)?;
            }
            ast::GraphPattern::Union { left, right } => {
                self.collect_service_clauses_from_pattern(left, services)?;
                self.collect_service_clauses_from_pattern(right, services)?;
            }
            ast::GraphPattern::LeftJoin { left, right, .. } => {
                self.collect_service_clauses_from_pattern(left, services)?;
                self.collect_service_clauses_from_pattern(right, services)?;
            }
            ast::GraphPattern::Filter { inner, .. } => {
                self.collect_service_clauses_from_pattern(inner, services)?;
            }
            ast::GraphPattern::Group { inner, .. } => {
                self.collect_service_clauses_from_pattern(inner, services)?;
            }
            ast::GraphPattern::Graph { inner, .. } => {
                self.collect_service_clauses_from_pattern(inner, services)?;
            }
            ast::GraphPattern::Extend { inner, .. } => {
                self.collect_service_clauses_from_pattern(inner, services)?;
            }
            ast::GraphPattern::Minus { left, right } => {
                self.collect_service_clauses_from_pattern(left, services)?;
                self.collect_service_clauses_from_pattern(right, services)?;
            }
            ast::GraphPattern::OrderBy { inner, .. } => {
                self.collect_service_clauses_from_pattern(inner, services)?;
            }
            ast::GraphPattern::Project { inner, .. } => {
                self.collect_service_clauses_from_pattern(inner, services)?;
            }
            ast::GraphPattern::Distinct { inner } => {
                self.collect_service_clauses_from_pattern(inner, services)?;
            }
            ast::GraphPattern::Reduced { inner } => {
                self.collect_service_clauses_from_pattern(inner, services)?;
            }
            ast::GraphPattern::Slice { inner, .. } => {
                self.collect_service_clauses_from_pattern(inner, services)?;
            }
            _ => {
                // Other patterns don't contain service clauses
            }
        }
        Ok(())
    }

    fn extract_filters_from_ast(&self, query: &Query) -> Result<Vec<FilterExpression>> {
        let mut filters = Vec::new();
        
        match query {
            Query::Select { pattern, .. } |
            Query::Construct { pattern, .. } |
            Query::Ask { pattern, .. } |
            Query::Describe { pattern, .. } => {
                self.collect_filters_from_pattern(pattern, &mut filters)?;
            }
            Query::Update { .. } => {
                // Handle update filters if needed
            }
        }
        
        Ok(filters)
    }

    fn collect_filters_from_pattern(
        &self,
        pattern: &ast::GraphPattern,
        filters: &mut Vec<FilterExpression>
    ) -> Result<()> {
        match pattern {
            ast::GraphPattern::Filter { expr, inner } => {
                // Convert expression to string (simplified)
                let expr_string = format!("{:?}", expr); // TODO: Proper expression serialization
                let variables = HashSet::new(); // TODO: Extract variables from expression
                
                filters.push(FilterExpression {
                    expression: expr_string,
                    variables,
                });
                
                self.collect_filters_from_pattern(inner, filters)?;
            }
            ast::GraphPattern::Join { left, right } => {
                self.collect_filters_from_pattern(left, filters)?;
                self.collect_filters_from_pattern(right, filters)?;
            }
            ast::GraphPattern::Union { left, right } => {
                self.collect_filters_from_pattern(left, filters)?;
                self.collect_filters_from_pattern(right, filters)?;
            }
            ast::GraphPattern::LeftJoin { left, right, .. } => {
                self.collect_filters_from_pattern(left, filters)?;
                self.collect_filters_from_pattern(right, filters)?;
            }
            ast::GraphPattern::Service { inner, .. } => {
                self.collect_filters_from_pattern(inner, filters)?;
            }
            ast::GraphPattern::Group { inner, .. } => {
                self.collect_filters_from_pattern(inner, filters)?;
            }
            ast::GraphPattern::Graph { inner, .. } => {
                self.collect_filters_from_pattern(inner, filters)?;
            }
            ast::GraphPattern::Extend { inner, .. } => {
                self.collect_filters_from_pattern(inner, filters)?;
            }
            ast::GraphPattern::Minus { left, right } => {
                self.collect_filters_from_pattern(left, filters)?;
                self.collect_filters_from_pattern(right, filters)?;
            }
            ast::GraphPattern::OrderBy { inner, .. } => {
                self.collect_filters_from_pattern(inner, filters)?;
            }
            ast::GraphPattern::Project { inner, .. } => {
                self.collect_filters_from_pattern(inner, filters)?;
            }
            ast::GraphPattern::Distinct { inner } => {
                self.collect_filters_from_pattern(inner, filters)?;
            }
            ast::GraphPattern::Reduced { inner } => {
                self.collect_filters_from_pattern(inner, filters)?;
            }
            ast::GraphPattern::Slice { inner, .. } => {
                self.collect_filters_from_pattern(inner, filters)?;
            }
            _ => {
                // Other patterns don't contain filters
            }
        }
        Ok(())
    }

    fn extract_variables_from_ast(&self, query: &Query) -> Result<HashSet<String>> {
        let mut variables = HashSet::new();
        
        match query {
            Query::Select { pattern, selection, .. } => {
                // Extract projection variables
                match selection {
                    ast::QuerySelection::Distinct(vars) | ast::QuerySelection::Reduced(vars) => {
                        for var in vars {
                            match var {
                                ast::SelectionMember::Variable(v) => {
                                    variables.insert(format!("?{}", v));
                                }
                                ast::SelectionMember::Expression(_, alias) => {
                                    if let Some(alias_var) = alias {
                                        variables.insert(format!("?{}", alias_var));
                                    }
                                }
                            }
                        }
                    }
                    ast::QuerySelection::Asterisk => {
                        // Extract all variables from pattern
                        self.collect_variables_from_pattern(pattern, &mut variables)?;
                    }
                }
            }
            Query::Construct { pattern, .. } |
            Query::Ask { pattern, .. } |
            Query::Describe { pattern, .. } => {
                self.collect_variables_from_pattern(pattern, &mut variables)?;
            }
            Query::Update { .. } => {
                // Handle update variables if needed
            }
        }
        
        Ok(variables)
    }

    fn collect_variables_from_pattern(
        &self,
        pattern: &ast::GraphPattern,
        variables: &mut HashSet<String>
    ) -> Result<()> {
        match pattern {
            ast::GraphPattern::Bgp { patterns: triples } => {
                for triple in triples {
                    if let ast::TermPattern::Variable(var) = &triple.subject {
                        variables.insert(format!("?{}", var));
                    }
                    if let ast::TermPattern::Variable(var) = &triple.predicate {
                        variables.insert(format!("?{}", var));
                    }
                    if let ast::TermPattern::Variable(var) = &triple.object {
                        variables.insert(format!("?{}", var));
                    }
                }
            }
            ast::GraphPattern::Join { left, right } => {
                self.collect_variables_from_pattern(left, variables)?;
                self.collect_variables_from_pattern(right, variables)?;
            }
            ast::GraphPattern::Union { left, right } => {
                self.collect_variables_from_pattern(left, variables)?;
                self.collect_variables_from_pattern(right, variables)?;
            }
            ast::GraphPattern::LeftJoin { left, right, .. } => {
                self.collect_variables_from_pattern(left, variables)?;
                self.collect_variables_from_pattern(right, variables)?;
            }
            ast::GraphPattern::Filter { inner, .. } => {
                self.collect_variables_from_pattern(inner, variables)?;
            }
            ast::GraphPattern::Service { inner, .. } => {
                self.collect_variables_from_pattern(inner, variables)?;
            }
            ast::GraphPattern::Group { inner, .. } => {
                self.collect_variables_from_pattern(inner, variables)?;
            }
            ast::GraphPattern::Graph { inner, .. } => {
                self.collect_variables_from_pattern(inner, variables)?;
            }
            ast::GraphPattern::Extend { inner, variable, .. } => {
                variables.insert(format!("?{}", variable));
                self.collect_variables_from_pattern(inner, variables)?;
            }
            ast::GraphPattern::Minus { left, right } => {
                self.collect_variables_from_pattern(left, variables)?;
                self.collect_variables_from_pattern(right, variables)?;
            }
            ast::GraphPattern::OrderBy { inner, .. } => {
                self.collect_variables_from_pattern(inner, variables)?;
            }
            ast::GraphPattern::Project { inner, variables: proj_vars } => {
                for var in proj_vars {
                    variables.insert(format!("?{}", var));
                }
                self.collect_variables_from_pattern(inner, variables)?;
            }
            ast::GraphPattern::Distinct { inner } => {
                self.collect_variables_from_pattern(inner, variables)?;
            }
            ast::GraphPattern::Reduced { inner } => {
                self.collect_variables_from_pattern(inner, variables)?;
            }
            ast::GraphPattern::Slice { inner, .. } => {
                self.collect_variables_from_pattern(inner, variables)?;
            }
            _ => {
                // Other patterns
            }
        }
        Ok(())
    }

    fn detect_query_type(&self, query: &str) -> QueryType {
        let query_upper = query.to_uppercase();
        if query_upper.trim_start().starts_with("SELECT") {
            QueryType::SparqlSelect
        } else if query_upper.trim_start().starts_with("CONSTRUCT") {
            QueryType::SparqlConstruct
        } else if query_upper.trim_start().starts_with("ASK") {
            QueryType::SparqlAsk
        } else if query_upper.trim_start().starts_with("DESCRIBE") {
            QueryType::SparqlDescribe
        } else if query_upper.contains("INSERT") || query_upper.contains("DELETE") {
            QueryType::SparqlUpdate
        } else {
            QueryType::Unknown
        }
    }

    fn extract_triple_patterns(&self, query: &str) -> Result<Vec<TriplePattern>> {
        // Simplified pattern extraction - would need a proper SPARQL parser
        let mut patterns = Vec::new();
        
        // Look for basic triple patterns in WHERE clauses
        if let Some(where_start) = query.to_uppercase().find("WHERE") {
            let where_clause = &query[where_start..];
            
            // Very basic pattern detection - this needs improvement
            let lines: Vec<&str> = where_clause.lines().collect();
            for line in lines {
                let trimmed = line.trim();
                if trimmed.contains(" ") && !trimmed.starts_with('#') && !trimmed.is_empty() {
                    patterns.push(TriplePattern {
                        subject: "?s".to_string(), // Simplified
                        predicate: "?p".to_string(),
                        object: "?o".to_string(),
                        pattern_string: trimmed.to_string(),
                    });
                }
            }
        }

        Ok(patterns)
    }

    fn extract_service_clauses(&self, query: &str) -> Result<Vec<ServiceClause>> {
        let mut services = Vec::new();
        let query_upper = query.to_uppercase();
        
        // Find SERVICE clauses
        let mut pos = 0;
        while let Some(service_pos) = query_upper[pos..].find("SERVICE") {
            let absolute_pos = pos + service_pos;
            
            // Extract service URL and subquery
            if let Some(url_start) = query[absolute_pos..].find('<') {
                if let Some(url_end) = query[absolute_pos + url_start..].find('>') {
                    let service_url = &query[absolute_pos + url_start + 1..absolute_pos + url_start + url_end];
                    
                    // Extract the subquery (simplified)
                    let subquery = "SELECT * WHERE { ?s ?p ?o }".to_string(); // Placeholder
                    
                    services.push(ServiceClause {
                        service_url: service_url.to_string(),
                        subquery,
                        silent: query_upper[absolute_pos..].contains("SILENT"),
                    });
                }
            }
            
            pos = absolute_pos + 7; // Move past "SERVICE"
        }

        Ok(services)
    }

    fn extract_filters(&self, query: &str) -> Result<Vec<FilterExpression>> {
        let mut filters = Vec::new();
        let query_upper = query.to_uppercase();
        
        // Find FILTER clauses
        let mut pos = 0;
        while let Some(filter_pos) = query_upper[pos..].find("FILTER") {
            let absolute_pos = pos + filter_pos;
            
            // Extract filter expression (simplified)
            if let Some(paren_start) = query[absolute_pos..].find('(') {
                let mut paren_count = 0;
                let mut filter_end = absolute_pos + paren_start;
                
                for (i, c) in query[absolute_pos + paren_start..].char_indices() {
                    match c {
                        '(' => paren_count += 1,
                        ')' => {
                            paren_count -= 1;
                            if paren_count == 0 {
                                filter_end = absolute_pos + paren_start + i + 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                
                let filter_expr = &query[absolute_pos + paren_start + 1..filter_end - 1];
                filters.push(FilterExpression {
                    expression: filter_expr.to_string(),
                    variables: self.extract_variables_from_expression(filter_expr),
                });
            }
            
            pos = absolute_pos + 6; // Move past "FILTER"
        }

        Ok(filters)
    }

    fn extract_variables(&self, query: &str) -> Result<HashSet<String>> {
        let mut variables = HashSet::new();
        
        // Find all variables (starting with ?)
        for word in query.split_whitespace() {
            if word.starts_with('?') {
                let var_name = word.trim_end_matches(&['.', ',', ';', ')', '}'][..]);
                variables.insert(var_name.to_string());
            }
        }

        Ok(variables)
    }

    fn extract_variables_from_expression(&self, expr: &str) -> HashSet<String> {
        let mut variables = HashSet::new();
        
        for word in expr.split_whitespace() {
            if word.starts_with('?') {
                let var_name = word.trim_end_matches(&['.', ',', ';', ')', '}'][..]);
                variables.insert(var_name.to_string());
            }
        }

        variables
    }

    fn extract_graphql_selections(&self, _query: &str) -> Result<Vec<GraphQLSelection>> {
        // TODO: Implement proper GraphQL parsing
        Ok(vec![GraphQLSelection {
            name: "placeholder".to_string(),
            arguments: HashMap::new(),
            selections: Vec::new(),
        }])
    }

    fn graphql_to_patterns(&self, _selections: &[GraphQLSelection]) -> Result<Vec<TriplePattern>> {
        // TODO: Convert GraphQL selections to RDF patterns
        Ok(Vec::new())
    }

    fn calculate_complexity(&self, patterns: &[TriplePattern], filters: &[FilterExpression], services: &[ServiceClause]) -> QueryComplexity {
        let base_complexity = patterns.len() + filters.len() * 2 + services.len() * 3;
        
        if base_complexity < 5 {
            QueryComplexity::Low
        } else if base_complexity < 15 {
            QueryComplexity::Medium
        } else if base_complexity < 30 {
            QueryComplexity::High
        } else {
            QueryComplexity::VeryHigh
        }
    }

    fn calculate_graphql_complexity(&self, _selections: &[GraphQLSelection]) -> QueryComplexity {
        // TODO: Implement GraphQL complexity analysis
        QueryComplexity::Medium
    }

    fn estimate_query_cost(&self, patterns: &[TriplePattern], services: &[ServiceClause]) -> u64 {
        (patterns.len() * 10 + services.len() * 50) as u64
    }

    fn estimate_graphql_cost(&self, _selections: &[GraphQLSelection]) -> u64 {
        100 // Placeholder
    }

    fn create_service_step(&self, service_clause: &ServiceClause, registry: &ServiceRegistry) -> Result<ExecutionStep> {
        // Find the service in the registry
        let service = registry
            .get_all_services()
            .find(|s| s.endpoint == service_clause.service_url)
            .ok_or_else(|| anyhow!("Service not found: {}", service_clause.service_url))?;

        Ok(ExecutionStep {
            step_id: uuid::Uuid::new_v4().to_string(),
            step_type: StepType::ServiceQuery,
            service_id: Some(service.id.clone()),
            query_fragment: service_clause.subquery.clone(),
            expected_variables: HashSet::new(), // TODO: Extract from subquery
            estimated_duration: Duration::from_millis(150),
            dependencies: Vec::new(),
            parallel_group: None,
        })
    }

    fn assign_patterns_to_services(&self, patterns: &[&TriplePattern], registry: &ServiceRegistry) -> Result<HashMap<String, Vec<TriplePattern>>> {
        let mut assignments = HashMap::new();
        
        // Simple assignment: try to assign all patterns to SPARQL services
        let sparql_services: Vec<_> = registry
            .get_services_with_capability(&ServiceCapability::SparqlQuery)
            .collect();

        if sparql_services.is_empty() {
            return Err(anyhow!("No SPARQL services available"));
        }

        // For now, assign all patterns to the first available service
        // TODO: Implement more sophisticated pattern-to-service assignment
        if let Some(service) = sparql_services.first() {
            assignments.insert(
                service.id.clone(),
                patterns.iter().map(|p| (*p).clone()).collect(),
            );
        }

        Ok(assignments)
    }

    fn is_pattern_in_service_clause(&self, _pattern: &TriplePattern, _service_clauses: &[ServiceClause]) -> bool {
        // TODO: Check if pattern is already handled by a SERVICE clause
        false
    }

    fn build_sparql_fragment(&self, patterns: &[TriplePattern]) -> String {
        let pattern_strings: Vec<String> = patterns
            .iter()
            .map(|p| p.pattern_string.clone())
            .collect();
        
        format!("SELECT * WHERE {{\n  {}\n}}", pattern_strings.join(" .\n  "))
    }

    fn extract_pattern_variables(&self, patterns: &[TriplePattern]) -> HashSet<String> {
        let mut variables = HashSet::new();
        
        for pattern in patterns {
            // Extract variables from subject, predicate, object
            if pattern.subject.starts_with('?') {
                variables.insert(pattern.subject.clone());
            }
            if pattern.predicate.starts_with('?') {
                variables.insert(pattern.predicate.clone());
            }
            if pattern.object.starts_with('?') {
                variables.insert(pattern.object.clone());
            }
        }

        variables
    }

    fn optimize_plan(&self, plan: &mut ExecutionPlan) {
        // Identify parallelizable steps
        let mut parallel_groups = HashMap::new();
        let mut group_id = 0;

        for step in &mut plan.steps {
            if step.dependencies.is_empty() && step.service_id.is_some() {
                step.parallel_group = Some(group_id);
                parallel_groups.entry(group_id).or_insert_with(Vec::new).push(step.step_id.clone());
            }
        }

        plan.parallelizable_steps = parallel_groups.into_values().collect();

        // Estimate total duration
        let mut max_parallel_duration = Duration::from_secs(0);
        let mut sequential_duration = Duration::from_secs(0);

        for step in &plan.steps {
            if step.parallel_group.is_some() {
                max_parallel_duration = max_parallel_duration.max(step.estimated_duration);
            } else {
                sequential_duration += step.estimated_duration;
            }
        }

        plan.estimated_duration = max_parallel_duration + sequential_duration;
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the query planner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlannerConfig {
    pub max_services_per_query: usize,
    pub optimization_level: OptimizationLevel,
    pub timeout: Duration,
    pub enable_caching: bool,
    pub cost_threshold: u64,
}

impl Default for QueryPlannerConfig {
    fn default() -> Self {
        Self {
            max_services_per_query: 10,
            optimization_level: OptimizationLevel::Balanced,
            timeout: Duration::from_secs(30),
            enable_caching: true,
            cost_threshold: 1000,
        }
    }
}

/// Optimization levels for query planning
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Balanced,
    Aggressive,
}

/// Information extracted from query analysis
#[derive(Debug, Clone)]
pub struct QueryInfo {
    pub query_type: QueryType,
    pub original_query: String,
    pub patterns: Vec<TriplePattern>,
    pub service_clauses: Vec<ServiceClause>,
    pub filters: Vec<FilterExpression>,
    pub variables: HashSet<String>,
    pub complexity: QueryComplexity,
    pub estimated_cost: u64,
}

/// Types of queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    SparqlSelect,
    SparqlConstruct,
    SparqlAsk,
    SparqlDescribe,
    SparqlUpdate,
    GraphQLQuery,
    GraphQLMutation,
    GraphQLSubscription,
    Unknown,
}

/// RDF triple pattern
#[derive(Debug, Clone)]
pub struct TriplePattern {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub pattern_string: String,
}

/// SPARQL SERVICE clause
#[derive(Debug, Clone)]
pub struct ServiceClause {
    pub service_url: String,
    pub subquery: String,
    pub silent: bool,
}

/// SPARQL FILTER expression
#[derive(Debug, Clone)]
pub struct FilterExpression {
    pub expression: String,
    pub variables: HashSet<String>,
}

/// GraphQL selection
#[derive(Debug, Clone)]
pub struct GraphQLSelection {
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
    pub selections: Vec<GraphQLSelection>,
}

/// Query complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryComplexity {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Execution plan for federated queries
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub query_id: String,
    pub query_type: QueryType,
    pub steps: Vec<ExecutionStep>,
    pub estimated_duration: Duration,
    pub parallelizable_steps: Vec<Vec<String>>, // Groups of step IDs that can run in parallel
    pub dependencies: HashMap<String, Vec<String>>, // Step dependencies
}

/// Individual step in an execution plan
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub step_id: String,
    pub step_type: StepType,
    pub service_id: Option<String>,
    pub query_fragment: String,
    pub expected_variables: HashSet<String>,
    pub estimated_duration: Duration,
    pub dependencies: Vec<String>, // IDs of steps this step depends on
    pub parallel_group: Option<usize>, // Which parallel group this step belongs to
}

/// Types of execution steps
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepType {
    ServiceQuery,
    GraphQLQuery,
    Join,
    Union,
    Filter,
    SchemaStitch,
    Aggregate,
    Sort,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ServiceRegistry, FederatedService};

    #[tokio::test]
    async fn test_query_planner_creation() {
        let planner = QueryPlanner::new();
        assert_eq!(planner.config.max_services_per_query, 10);
    }

    #[tokio::test]
    async fn test_sparql_query_analysis() {
        let planner = QueryPlanner::new();
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        
        let result = planner.analyze_sparql(query).await;
        assert!(result.is_ok());
        
        let query_info = result.unwrap();
        assert_eq!(query_info.query_type, QueryType::SparqlSelect);
        assert!(!query_info.variables.is_empty());
    }

    #[tokio::test]
    async fn test_query_type_detection() {
        let planner = QueryPlanner::new();
        
        assert_eq!(planner.detect_query_type("SELECT * WHERE { ?s ?p ?o }"), QueryType::SparqlSelect);
        assert_eq!(planner.detect_query_type("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"), QueryType::SparqlConstruct);
        assert_eq!(planner.detect_query_type("ASK { ?s ?p ?o }"), QueryType::SparqlAsk);
        assert_eq!(planner.detect_query_type("DESCRIBE <http://example.org>"), QueryType::SparqlDescribe);
    }

    #[tokio::test]
    async fn test_service_clause_extraction() {
        let planner = QueryPlanner::new();
        let query = "SELECT * WHERE { SERVICE <http://example.org/sparql> { ?s ?p ?o } }";
        
        let services = planner.extract_service_clauses(query).unwrap();
        assert_eq!(services.len(), 1);
        assert_eq!(services[0].service_url, "http://example.org/sparql");
    }

    #[tokio::test]
    async fn test_execution_plan_creation() {
        let planner = QueryPlanner::new();
        let mut registry = ServiceRegistry::new();
        
        let service = FederatedService::new_sparql(
            "test-service".to_string(),
            "Test Service".to_string(),
            "http://example.com/sparql".to_string(),
        );
        registry.register(service).await.unwrap();

        let query_info = QueryInfo {
            query_type: QueryType::SparqlSelect,
            original_query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            patterns: vec![TriplePattern {
                subject: "?s".to_string(),
                predicate: "?p".to_string(),
                object: "?o".to_string(),
                pattern_string: "?s ?p ?o".to_string(),
            }],
            service_clauses: Vec::new(),
            filters: Vec::new(),
            variables: ["?s", "?p", "?o"].into_iter().map(|s| s.to_string()).collect(),
            complexity: QueryComplexity::Low,
            estimated_cost: 10,
        };

        let plan = planner.plan_sparql(&query_info, &registry).await.unwrap();
        assert!(!plan.steps.is_empty());
        assert_eq!(plan.query_type, QueryType::SparqlSelect);
    }
}