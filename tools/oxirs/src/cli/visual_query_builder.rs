//! Visual Query Builder
//!
//! Interactive SPARQL query construction with step-by-step guidance.
//! Supports SELECT, ASK, CONSTRUCT, and DESCRIBE queries with full
//! SPARQL 1.1 feature support.

use crate::cli::error::{CliError, CliResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Write};

/// Query type for SPARQL queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    Select,
    Ask,
    Construct,
    Describe,
}

impl QueryType {
    pub fn as_str(&self) -> &'static str {
        match self {
            QueryType::Select => "SELECT",
            QueryType::Ask => "ASK",
            QueryType::Construct => "CONSTRUCT",
            QueryType::Describe => "DESCRIBE",
        }
    }

    pub fn parse_type(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "SELECT" => Some(QueryType::Select),
            "ASK" => Some(QueryType::Ask),
            "CONSTRUCT" => Some(QueryType::Construct),
            "DESCRIBE" => Some(QueryType::Describe),
            _ => None,
        }
    }
}

/// Triple pattern in SPARQL
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriplePattern {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl TriplePattern {
    pub fn new(subject: String, predicate: String, object: String) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    pub fn to_sparql(&self) -> String {
        format!("{} {} {}", self.subject, self.predicate, self.object)
    }
}

/// FILTER expression
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilterExpression {
    pub expression: String,
}

impl FilterExpression {
    pub fn new(expression: String) -> Self {
        Self { expression }
    }

    pub fn to_sparql(&self) -> String {
        format!("FILTER ({})", self.expression)
    }
}

/// OPTIONAL clause
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionalClause {
    pub patterns: Vec<TriplePattern>,
}

impl OptionalClause {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    pub fn add_pattern(&mut self, pattern: TriplePattern) {
        self.patterns.push(pattern);
    }

    pub fn to_sparql(&self) -> String {
        let patterns: Vec<String> = self
            .patterns
            .iter()
            .map(|p| format!("    {}", p.to_sparql()))
            .collect();
        format!("  OPTIONAL {{\n{}\n  }}", patterns.join(" .\n"))
    }
}

impl Default for OptionalClause {
    fn default() -> Self {
        Self::new()
    }
}

/// ORDER BY clause
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderByClause {
    pub variable: String,
    pub ascending: bool,
}

impl OrderByClause {
    pub fn new(variable: String, ascending: bool) -> Self {
        Self {
            variable,
            ascending,
        }
    }

    pub fn to_sparql(&self) -> String {
        if self.ascending {
            format!("ORDER BY ?{}", self.variable)
        } else {
            format!("ORDER BY DESC(?{})", self.variable)
        }
    }
}

/// Query builder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryBuilderConfig {
    pub auto_add_prefixes: bool,
    pub validate_on_build: bool,
    pub show_preview: bool,
}

impl QueryBuilderConfig {
    pub fn new() -> Self {
        Self {
            auto_add_prefixes: true,
            validate_on_build: true,
            show_preview: true,
        }
    }
}

impl Default for QueryBuilderConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Visual SPARQL query builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualQueryBuilder {
    query_type: Option<QueryType>,
    variables: Vec<String>,
    prefixes: HashMap<String, String>,
    triple_patterns: Vec<TriplePattern>,
    filters: Vec<FilterExpression>,
    optionals: Vec<OptionalClause>,
    order_by: Vec<OrderByClause>,
    distinct: bool,
    limit: Option<u64>,
    offset: Option<u64>,
    graph: Option<String>,
    config: QueryBuilderConfig,
}

impl VisualQueryBuilder {
    /// Create a new visual query builder
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();
        // Add common prefixes
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        prefixes.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        prefixes.insert(
            "dc".to_string(),
            "http://purl.org/dc/elements/1.1/".to_string(),
        );

        Self {
            query_type: None,
            variables: Vec::new(),
            prefixes,
            triple_patterns: Vec::new(),
            filters: Vec::new(),
            optionals: Vec::new(),
            order_by: Vec::new(),
            distinct: false,
            limit: None,
            offset: None,
            graph: None,
            config: QueryBuilderConfig::new(),
        }
    }

    /// Create a new builder with custom configuration
    pub fn with_config(config: QueryBuilderConfig) -> Self {
        let mut builder = Self::new();
        builder.config = config;
        builder
    }

    /// Set the query type
    pub fn set_query_type(&mut self, query_type: QueryType) {
        self.query_type = Some(query_type);
    }

    /// Add a variable for SELECT query
    pub fn add_variable(&mut self, variable: String) -> CliResult<()> {
        if self.variables.contains(&variable) {
            return Err(CliError::invalid_arguments(format!(
                "Variable '{}' already added",
                variable
            )));
        }
        self.variables.push(variable);
        Ok(())
    }

    /// Add a prefix
    pub fn add_prefix(&mut self, prefix: String, uri: String) {
        self.prefixes.insert(prefix, uri);
    }

    /// Add a triple pattern
    pub fn add_triple_pattern(&mut self, subject: String, predicate: String, object: String) {
        self.triple_patterns
            .push(TriplePattern::new(subject, predicate, object));
    }

    /// Add a FILTER expression
    pub fn add_filter(&mut self, expression: String) {
        self.filters.push(FilterExpression::new(expression));
    }

    /// Add an OPTIONAL clause
    pub fn add_optional(&mut self, optional: OptionalClause) {
        self.optionals.push(optional);
    }

    /// Add ORDER BY clause
    pub fn add_order_by(&mut self, variable: String, ascending: bool) {
        self.order_by.push(OrderByClause::new(variable, ascending));
    }

    /// Set DISTINCT modifier
    pub fn set_distinct(&mut self, distinct: bool) {
        self.distinct = distinct;
    }

    /// Set LIMIT
    pub fn set_limit(&mut self, limit: Option<u64>) {
        self.limit = limit;
    }

    /// Set OFFSET
    pub fn set_offset(&mut self, offset: Option<u64>) {
        self.offset = offset;
    }

    /// Set GRAPH
    pub fn set_graph(&mut self, graph: Option<String>) {
        self.graph = graph;
    }

    /// Reset the builder
    pub fn reset(&mut self) {
        self.query_type = None;
        self.variables.clear();
        self.triple_patterns.clear();
        self.filters.clear();
        self.optionals.clear();
        self.order_by.clear();
        self.distinct = false;
        self.limit = None;
        self.offset = None;
        self.graph = None;
    }

    /// Build the SPARQL query
    pub fn build(&self) -> CliResult<String> {
        let query_type = self
            .query_type
            .ok_or_else(|| CliError::invalid_arguments("Query type not set".to_string()))?;

        let mut query = String::new();

        // Add prefixes
        for (prefix, uri) in &self.prefixes {
            query.push_str(&format!("PREFIX {}: <{}>\n", prefix, uri));
        }

        if !self.prefixes.is_empty() {
            query.push('\n');
        }

        // Add query type and variables
        match query_type {
            QueryType::Select => {
                query.push_str("SELECT ");
                if self.distinct {
                    query.push_str("DISTINCT ");
                }
                if self.variables.is_empty() {
                    query.push('*');
                } else {
                    for var in &self.variables {
                        query.push_str(&format!("?{} ", var));
                    }
                }
                query.push('\n');
            }
            QueryType::Ask => {
                query.push_str("ASK\n");
            }
            QueryType::Construct => {
                query.push_str("CONSTRUCT {\n");
                for pattern in &self.triple_patterns {
                    query.push_str(&format!("  {} .\n", pattern.to_sparql()));
                }
                query.push_str("}\n");
            }
            QueryType::Describe => {
                query.push_str("DESCRIBE ");
                if self.variables.is_empty() {
                    return Err(CliError::invalid_arguments(
                        "DESCRIBE query requires at least one variable or URI".to_string(),
                    ));
                }
                for var in &self.variables {
                    query.push_str(&format!("?{} ", var));
                }
                query.push('\n');
            }
        }

        // Add WHERE clause
        query.push_str("WHERE {\n");

        // Add GRAPH if specified
        if let Some(graph) = &self.graph {
            query.push_str(&format!("  GRAPH <{}> {{\n", graph));
        }

        // Add triple patterns
        for pattern in &self.triple_patterns {
            let indent = if self.graph.is_some() { "    " } else { "  " };
            query.push_str(&format!("{}{} .\n", indent, pattern.to_sparql()));
        }

        // Add FILTER clauses
        for filter in &self.filters {
            let indent = if self.graph.is_some() { "    " } else { "  " };
            query.push_str(&format!("{}{}\n", indent, filter.to_sparql()));
        }

        // Close GRAPH if specified
        if self.graph.is_some() {
            query.push_str("  }\n");
        }

        // Add OPTIONAL clauses
        for optional in &self.optionals {
            query.push_str(&optional.to_sparql());
            query.push('\n');
        }

        query.push_str("}\n");

        // Add ORDER BY
        if !self.order_by.is_empty() {
            for order in &self.order_by {
                query.push_str(&format!("{}\n", order.to_sparql()));
            }
        }

        // Add LIMIT
        if let Some(limit) = self.limit {
            query.push_str(&format!("LIMIT {}\n", limit));
        }

        // Add OFFSET
        if let Some(offset) = self.offset {
            query.push_str(&format!("OFFSET {}\n", offset));
        }

        Ok(query)
    }

    /// Interactive query building
    pub fn build_interactive(&mut self) -> CliResult<String> {
        println!("\n=== Visual SPARQL Query Builder ===\n");

        // Step 1: Choose query type
        self.prompt_query_type()?;

        // Step 2: Add prefixes
        self.prompt_prefixes()?;

        // Step 3: Add variables (for SELECT/DESCRIBE)
        if let Some(QueryType::Select | QueryType::Describe) = self.query_type {
            self.prompt_variables()?;
        }

        // Step 4: Add triple patterns
        self.prompt_triple_patterns()?;

        // Step 5: Add filters
        self.prompt_filters()?;

        // Step 6: Add optional clauses
        self.prompt_optionals()?;

        // Step 7: Add modifiers
        self.prompt_modifiers()?;

        // Build and preview query
        let query = self.build()?;

        if self.config.show_preview {
            println!("\n=== Generated SPARQL Query ===\n");
            println!("{}", query);
            println!();
        }

        Ok(query)
    }

    fn prompt_query_type(&mut self) -> CliResult<()> {
        println!("Step 1: Select Query Type");
        println!("  1. SELECT - Retrieve specific variables");
        println!("  2. ASK    - Boolean query");
        println!("  3. CONSTRUCT - Build RDF graph");
        println!("  4. DESCRIBE - Describe resources");
        print!("\nEnter choice (1-4): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let query_type = match input.trim() {
            "1" => QueryType::Select,
            "2" => QueryType::Ask,
            "3" => QueryType::Construct,
            "4" => QueryType::Describe,
            _ => return Err(CliError::invalid_arguments("Invalid choice".to_string())),
        };

        self.set_query_type(query_type);
        println!("✓ Query type set to: {}\n", query_type.as_str());
        Ok(())
    }

    fn prompt_prefixes(&mut self) -> CliResult<()> {
        println!("Step 2: Manage Prefixes");
        println!("Common prefixes already added:");
        for (prefix, uri) in &self.prefixes {
            println!("  PREFIX {}: <{}>", prefix, uri);
        }

        print!("\nAdd custom prefix? (y/n): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if input.trim().to_lowercase() == "y" {
            loop {
                print!("Prefix name (or 'done' to finish): ");
                io::stdout().flush()?;

                let mut prefix = String::new();
                io::stdin().read_line(&mut prefix)?;
                let prefix = prefix.trim();

                if prefix == "done" {
                    break;
                }

                print!("Prefix URI: ");
                io::stdout().flush()?;

                let mut uri = String::new();
                io::stdin().read_line(&mut uri)?;

                self.add_prefix(prefix.to_string(), uri.trim().to_string());
                println!("✓ Added prefix: {}\n", prefix);
            }
        }

        println!();
        Ok(())
    }

    fn prompt_variables(&mut self) -> CliResult<()> {
        println!("Step 3: Add Variables");
        print!("Enter variables to select (comma-separated, or '*' for all): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let input = input.trim();
        if input != "*" {
            for var in input.split(',') {
                let var = var.trim().trim_start_matches('?');
                if !var.is_empty() {
                    self.add_variable(var.to_string())?;
                }
            }
            println!("✓ Added {} variable(s)\n", self.variables.len());
        } else {
            println!("✓ Selecting all variables (*)\n");
        }

        Ok(())
    }

    fn prompt_triple_patterns(&mut self) -> CliResult<()> {
        println!("Step 4: Add Triple Patterns");
        println!("Enter triple patterns (use ?var for variables, <uri> for URIs)");
        println!("Examples:");
        println!("  ?person rdf:type foaf:Person");
        println!("  ?person foaf:name ?name");

        loop {
            print!("\nSubject (or 'done' to finish): ");
            io::stdout().flush()?;

            let mut subject = String::new();
            io::stdin().read_line(&mut subject)?;
            let subject = subject.trim();

            if subject == "done" {
                break;
            }

            print!("Predicate: ");
            io::stdout().flush()?;

            let mut predicate = String::new();
            io::stdin().read_line(&mut predicate)?;

            print!("Object: ");
            io::stdout().flush()?;

            let mut object = String::new();
            io::stdin().read_line(&mut object)?;

            self.add_triple_pattern(
                subject.to_string(),
                predicate.trim().to_string(),
                object.trim().to_string(),
            );
            println!("✓ Added triple pattern");
        }

        println!(
            "\n✓ Added {} triple pattern(s)\n",
            self.triple_patterns.len()
        );
        Ok(())
    }

    fn prompt_filters(&mut self) -> CliResult<()> {
        println!("Step 5: Add FILTER Clauses (Optional)");
        print!("Add filters? (y/n): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if input.trim().to_lowercase() == "y" {
            loop {
                print!("Filter expression (or 'done' to finish): ");
                io::stdout().flush()?;

                let mut filter = String::new();
                io::stdin().read_line(&mut filter)?;
                let filter = filter.trim();

                if filter == "done" {
                    break;
                }

                self.add_filter(filter.to_string());
                println!("✓ Added filter");
            }
        }

        println!();
        Ok(())
    }

    fn prompt_optionals(&mut self) -> CliResult<()> {
        println!("Step 6: Add OPTIONAL Clauses (Optional)");
        print!("Add optional patterns? (y/n): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if input.trim().to_lowercase() == "y" {
            loop {
                print!("Add OPTIONAL clause? (y/n): ");
                io::stdout().flush()?;

                let mut add_opt = String::new();
                io::stdin().read_line(&mut add_opt)?;

                if add_opt.trim().to_lowercase() != "y" {
                    break;
                }

                let mut optional = OptionalClause::new();
                println!("Enter triple patterns for OPTIONAL clause:");

                loop {
                    print!("Subject (or 'done' to finish): ");
                    io::stdout().flush()?;

                    let mut subject = String::new();
                    io::stdin().read_line(&mut subject)?;
                    let subject = subject.trim();

                    if subject == "done" {
                        break;
                    }

                    print!("Predicate: ");
                    io::stdout().flush()?;

                    let mut predicate = String::new();
                    io::stdin().read_line(&mut predicate)?;

                    print!("Object: ");
                    io::stdout().flush()?;

                    let mut object = String::new();
                    io::stdin().read_line(&mut object)?;

                    optional.add_pattern(TriplePattern::new(
                        subject.to_string(),
                        predicate.trim().to_string(),
                        object.trim().to_string(),
                    ));
                    println!("✓ Added pattern to OPTIONAL");
                }

                self.add_optional(optional);
                println!("✓ Added OPTIONAL clause");
            }
        }

        println!();
        Ok(())
    }

    fn prompt_modifiers(&mut self) -> CliResult<()> {
        println!("Step 7: Query Modifiers (Optional)");

        // DISTINCT
        print!("Use DISTINCT? (y/n): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        self.set_distinct(input.trim().to_lowercase() == "y");

        // ORDER BY
        print!("Add ORDER BY? (y/n): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if input.trim().to_lowercase() == "y" {
            print!("Variable to order by: ");
            io::stdout().flush()?;

            let mut var = String::new();
            io::stdin().read_line(&mut var)?;

            print!("Ascending? (y/n): ");
            io::stdout().flush()?;

            let mut asc = String::new();
            io::stdin().read_line(&mut asc)?;

            self.add_order_by(
                var.trim().trim_start_matches('?').to_string(),
                asc.trim().to_lowercase() == "y",
            );
        }

        // LIMIT
        print!("Add LIMIT? (number or press Enter to skip): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().is_empty() {
            if let Ok(limit) = input.trim().parse::<u64>() {
                self.set_limit(Some(limit));
            }
        }

        // OFFSET
        print!("Add OFFSET? (number or press Enter to skip): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().is_empty() {
            if let Ok(offset) = input.trim().parse::<u64>() {
                self.set_offset(Some(offset));
            }
        }

        println!();
        Ok(())
    }

    /// Get query statistics
    pub fn stats(&self) -> QueryBuilderStats {
        QueryBuilderStats {
            query_type: self.query_type,
            variable_count: self.variables.len(),
            prefix_count: self.prefixes.len(),
            triple_pattern_count: self.triple_patterns.len(),
            filter_count: self.filters.len(),
            optional_count: self.optionals.len(),
            has_limit: self.limit.is_some(),
            has_offset: self.offset.is_some(),
            has_order_by: !self.order_by.is_empty(),
            is_distinct: self.distinct,
        }
    }
}

impl Default for VisualQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Query builder statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryBuilderStats {
    pub query_type: Option<QueryType>,
    pub variable_count: usize,
    pub prefix_count: usize,
    pub triple_pattern_count: usize,
    pub filter_count: usize,
    pub optional_count: usize,
    pub has_limit: bool,
    pub has_offset: bool,
    pub has_order_by: bool,
    pub is_distinct: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_type_conversion() {
        assert_eq!(QueryType::parse_type("SELECT"), Some(QueryType::Select));
        assert_eq!(QueryType::parse_type("ask"), Some(QueryType::Ask));
        assert_eq!(
            QueryType::parse_type("CONSTRUCT"),
            Some(QueryType::Construct)
        );
        assert_eq!(QueryType::parse_type("describe"), Some(QueryType::Describe));
        assert_eq!(QueryType::parse_type("invalid"), None);
    }

    #[test]
    fn test_triple_pattern() {
        let pattern = TriplePattern::new(
            "?person".to_string(),
            "rdf:type".to_string(),
            "foaf:Person".to_string(),
        );
        assert_eq!(pattern.to_sparql(), "?person rdf:type foaf:Person");
    }

    #[test]
    fn test_filter_expression() {
        let filter = FilterExpression::new("?age > 18".to_string());
        assert_eq!(filter.to_sparql(), "FILTER (?age > 18)");
    }

    #[test]
    fn test_optional_clause() {
        let mut optional = OptionalClause::new();
        optional.add_pattern(TriplePattern::new(
            "?person".to_string(),
            "foaf:email".to_string(),
            "?email".to_string(),
        ));

        let sparql = optional.to_sparql();
        assert!(sparql.contains("OPTIONAL"));
        assert!(sparql.contains("?person foaf:email ?email"));
    }

    #[test]
    fn test_order_by_clause() {
        let order_asc = OrderByClause::new("name".to_string(), true);
        assert_eq!(order_asc.to_sparql(), "ORDER BY ?name");

        let order_desc = OrderByClause::new("age".to_string(), false);
        assert_eq!(order_desc.to_sparql(), "ORDER BY DESC(?age)");
    }

    #[test]
    fn test_simple_select_query() {
        let mut builder = VisualQueryBuilder::new();
        builder.set_query_type(QueryType::Select);
        builder.add_variable("name".to_string()).unwrap();
        builder.add_triple_pattern(
            "?person".to_string(),
            "foaf:name".to_string(),
            "?name".to_string(),
        );

        let query = builder.build().unwrap();
        assert!(query.contains("SELECT ?name"));
        assert!(query.contains("?person foaf:name ?name"));
    }

    #[test]
    fn test_select_all_variables() {
        let mut builder = VisualQueryBuilder::new();
        builder.set_query_type(QueryType::Select);
        builder.add_triple_pattern(
            "?person".to_string(),
            "foaf:name".to_string(),
            "?name".to_string(),
        );

        let query = builder.build().unwrap();
        assert!(query.contains("SELECT *"));
    }

    #[test]
    fn test_select_with_filter() {
        let mut builder = VisualQueryBuilder::new();
        builder.set_query_type(QueryType::Select);
        builder.add_variable("name".to_string()).unwrap();
        builder.add_triple_pattern(
            "?person".to_string(),
            "foaf:name".to_string(),
            "?name".to_string(),
        );
        builder.add_triple_pattern(
            "?person".to_string(),
            "foaf:age".to_string(),
            "?age".to_string(),
        );
        builder.add_filter("?age > 18".to_string());

        let query = builder.build().unwrap();
        assert!(query.contains("FILTER (?age > 18)"));
    }

    #[test]
    fn test_select_with_optional() {
        let mut builder = VisualQueryBuilder::new();
        builder.set_query_type(QueryType::Select);
        builder.add_variable("name".to_string()).unwrap();
        builder.add_variable("email".to_string()).unwrap();
        builder.add_triple_pattern(
            "?person".to_string(),
            "foaf:name".to_string(),
            "?name".to_string(),
        );

        let mut optional = OptionalClause::new();
        optional.add_pattern(TriplePattern::new(
            "?person".to_string(),
            "foaf:email".to_string(),
            "?email".to_string(),
        ));
        builder.add_optional(optional);

        let query = builder.build().unwrap();
        assert!(query.contains("OPTIONAL"));
        assert!(query.contains("?person foaf:email ?email"));
    }

    #[test]
    fn test_select_with_modifiers() {
        let mut builder = VisualQueryBuilder::new();
        builder.set_query_type(QueryType::Select);
        builder.add_variable("name".to_string()).unwrap();
        builder.add_triple_pattern(
            "?person".to_string(),
            "foaf:name".to_string(),
            "?name".to_string(),
        );
        builder.set_distinct(true);
        builder.add_order_by("name".to_string(), true);
        builder.set_limit(Some(10));
        builder.set_offset(Some(5));

        let query = builder.build().unwrap();
        assert!(query.contains("SELECT DISTINCT"));
        assert!(query.contains("ORDER BY ?name"));
        assert!(query.contains("LIMIT 10"));
        assert!(query.contains("OFFSET 5"));
    }

    #[test]
    fn test_ask_query() {
        let mut builder = VisualQueryBuilder::new();
        builder.set_query_type(QueryType::Ask);
        builder.add_triple_pattern(
            "?person".to_string(),
            "rdf:type".to_string(),
            "foaf:Person".to_string(),
        );

        let query = builder.build().unwrap();
        assert!(query.contains("ASK"));
        assert!(query.contains("?person rdf:type foaf:Person"));
    }

    #[test]
    fn test_construct_query() {
        let mut builder = VisualQueryBuilder::new();
        builder.set_query_type(QueryType::Construct);
        builder.add_triple_pattern(
            "?person".to_string(),
            "foaf:name".to_string(),
            "?name".to_string(),
        );

        let query = builder.build().unwrap();
        assert!(query.contains("CONSTRUCT {"));
        assert!(query.contains("?person foaf:name ?name"));
    }

    #[test]
    fn test_describe_query() {
        let mut builder = VisualQueryBuilder::new();
        builder.set_query_type(QueryType::Describe);
        builder.add_variable("person".to_string()).unwrap();
        builder.add_triple_pattern(
            "?person".to_string(),
            "rdf:type".to_string(),
            "foaf:Person".to_string(),
        );

        let query = builder.build().unwrap();
        assert!(query.contains("DESCRIBE ?person"));
    }

    #[test]
    fn test_query_with_graph() {
        let mut builder = VisualQueryBuilder::new();
        builder.set_query_type(QueryType::Select);
        builder.add_variable("name".to_string()).unwrap();
        builder.set_graph(Some("http://example.org/graph".to_string()));
        builder.add_triple_pattern(
            "?person".to_string(),
            "foaf:name".to_string(),
            "?name".to_string(),
        );

        let query = builder.build().unwrap();
        assert!(query.contains("GRAPH <http://example.org/graph>"));
    }

    #[test]
    fn test_builder_reset() {
        let mut builder = VisualQueryBuilder::new();
        builder.set_query_type(QueryType::Select);
        builder.add_variable("name".to_string()).unwrap();
        builder.add_triple_pattern(
            "?person".to_string(),
            "foaf:name".to_string(),
            "?name".to_string(),
        );

        builder.reset();

        assert!(builder.query_type.is_none());
        assert!(builder.variables.is_empty());
        assert!(builder.triple_patterns.is_empty());
    }

    #[test]
    fn test_duplicate_variable() {
        let mut builder = VisualQueryBuilder::new();
        builder.add_variable("name".to_string()).unwrap();
        let result = builder.add_variable("name".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_query_builder_stats() {
        let mut builder = VisualQueryBuilder::new();
        builder.set_query_type(QueryType::Select);
        builder.add_variable("name".to_string()).unwrap();
        builder.add_triple_pattern(
            "?person".to_string(),
            "foaf:name".to_string(),
            "?name".to_string(),
        );
        builder.add_filter("?age > 18".to_string());
        builder.set_limit(Some(10));

        let stats = builder.stats();
        assert_eq!(stats.query_type, Some(QueryType::Select));
        assert_eq!(stats.variable_count, 1);
        assert_eq!(stats.triple_pattern_count, 1);
        assert_eq!(stats.filter_count, 1);
        assert!(stats.has_limit);
    }

    #[test]
    fn test_no_query_type_error() {
        let builder = VisualQueryBuilder::new();
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_describe_without_variables_error() {
        let mut builder = VisualQueryBuilder::new();
        builder.set_query_type(QueryType::Describe);
        let result = builder.build();
        assert!(result.is_err());
    }
}
