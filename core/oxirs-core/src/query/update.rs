//! SPARQL UPDATE execution engine

use crate::{
    model::{GraphName, NamedNode, Quad},
    query::algebra::{Expression, GraphPattern, GraphTarget, QuadPattern, Update, UpdateOperation},
    query::{AlgebraTriplePattern, TermPattern},
    vocab::xsd,
    OxirsError, Result, Store,
};
use std::collections::HashMap;

/// SPARQL UPDATE executor
pub struct UpdateExecutor<'a> {
    store: &'a dyn Store,
}

impl<'a> UpdateExecutor<'a> {
    /// Create a new update executor
    pub fn new(store: &'a dyn Store) -> Self {
        Self { store }
    }

    /// Execute a SPARQL UPDATE request
    pub fn execute(&self, update: &Update) -> Result<()> {
        for operation in &update.operations {
            self.execute_operation(operation)?;
        }
        Ok(())
    }

    /// Execute a single update operation
    fn execute_operation(&self, operation: &UpdateOperation) -> Result<()> {
        match operation {
            UpdateOperation::InsertData { data } => self.execute_insert_data(data),
            UpdateOperation::DeleteData { data } => self.execute_delete_data(data),
            UpdateOperation::DeleteWhere { pattern } => self.execute_delete_where(pattern),
            UpdateOperation::Modify {
                delete,
                insert,
                where_clause,
                using: _,
            } => self.execute_modify(delete, insert, where_clause),
            UpdateOperation::Load {
                source,
                destination,
                silent,
            } => self.execute_load(source, destination, *silent),
            UpdateOperation::Clear { graph, silent } => self.execute_clear(graph, *silent),
            UpdateOperation::Create { graph, silent } => self.execute_create(graph, *silent),
            UpdateOperation::Drop { graph, silent } => self.execute_drop(graph, *silent),
            UpdateOperation::Copy {
                source,
                destination,
                silent,
            } => self.execute_copy(source, destination, *silent),
            UpdateOperation::Move {
                source,
                destination,
                silent,
            } => self.execute_move(source, destination, *silent),
            UpdateOperation::Add {
                source,
                destination,
                silent,
            } => self.execute_add(source, destination, *silent),
        }
    }

    /// Execute INSERT DATA operation
    fn execute_insert_data(&self, data: &[Quad]) -> Result<()> {
        for quad in data {
            self.store.insert_quad(quad.clone())?;
        }
        Ok(())
    }

    /// Execute DELETE DATA operation
    fn execute_delete_data(&self, data: &[Quad]) -> Result<()> {
        for quad in data {
            self.store.remove_quad(quad)?;
        }
        Ok(())
    }

    /// Execute DELETE WHERE operation
    fn execute_delete_where(&self, patterns: &[QuadPattern]) -> Result<()> {
        // Find all quads matching the pattern and delete them
        for pattern in patterns {
            let matching_quads = self.find_matching_quads(pattern)?;
            for quad in matching_quads {
                self.store.remove_quad(&quad)?;
            }
        }
        Ok(())
    }

    /// Execute INSERT/DELETE WHERE operation
    fn execute_modify(
        &self,
        delete_patterns: &Option<Vec<QuadPattern>>,
        insert_patterns: &Option<Vec<QuadPattern>>,
        where_clause: &GraphPattern,
    ) -> Result<()> {
        // First, execute the WHERE clause to get variable bindings
        let solutions = self.evaluate_graph_pattern(where_clause)?;

        // For each solution, apply the delete and insert patterns
        for solution in solutions {
            // Execute delete patterns first
            if let Some(delete_patterns) = delete_patterns {
                for pattern in delete_patterns {
                    if let Some(quad) = self.instantiate_quad_pattern(pattern, &solution)? {
                        self.store.remove_quad(&quad)?;
                    }
                }
            }

            // Then execute insert patterns
            if let Some(insert_patterns) = insert_patterns {
                for pattern in insert_patterns {
                    if let Some(quad) = self.instantiate_quad_pattern(pattern, &solution)? {
                        self.store.insert_quad(quad)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Execute LOAD operation
    fn execute_load(
        &self,
        source: &NamedNode,
        _destination: &Option<NamedNode>,
        _silent: bool,
    ) -> Result<()> {
        // For now, return an error as we don't have HTTP loading capability
        // In a full implementation, this would fetch RDF from the source IRI
        Err(OxirsError::Update(format!(
            "LOAD operation not implemented for source: {}",
            source
        )))
    }

    /// Execute CLEAR operation
    fn execute_clear(&self, graph: &GraphTarget, _silent: bool) -> Result<()> {
        match graph {
            GraphTarget::Default => {
                // Clear default graph
                let default_graph = GraphName::DefaultGraph;
                let quads = self
                    .store
                    .find_quads(None, None, None, Some(&default_graph))?;
                for quad in quads {
                    self.store.remove_quad(&quad)?;
                }
            }
            GraphTarget::Named(graph_name) => {
                // Clear named graph
                let graph = GraphName::NamedNode(graph_name.clone());
                let quads = self.store.find_quads(None, None, None, Some(&graph))?;
                for quad in quads {
                    self.store.remove_quad(&quad)?;
                }
            }
            GraphTarget::All => {
                // Clear all graphs
                let quads = self.store.find_quads(None, None, None, None)?;
                for quad in quads {
                    self.store.remove_quad(&quad)?;
                }
            }
        }
        Ok(())
    }

    /// Execute CREATE operation
    fn execute_create(&self, _graph: &NamedNode, _silent: bool) -> Result<()> {
        // Graph creation is implicit in most RDF stores
        // For now, this is a no-op
        Ok(())
    }

    /// Execute DROP operation
    fn execute_drop(&self, graph: &GraphTarget, _silent: bool) -> Result<()> {
        // DROP is similar to CLEAR but also removes the graph container
        // For now, we implement it the same as CLEAR
        self.execute_clear(graph, _silent)
    }

    /// Execute COPY operation
    fn execute_copy(
        &self,
        source: &GraphTarget,
        destination: &GraphTarget,
        _silent: bool,
    ) -> Result<()> {
        // First clear destination, then copy from source
        self.execute_clear(destination, true)?;

        let source_quads = self.get_quads_from_target(source)?;
        for quad in source_quads {
            let dest_quad = self.move_quad_to_target(&quad, destination)?;
            self.store.insert_quad(dest_quad)?;
        }
        Ok(())
    }

    /// Execute MOVE operation
    fn execute_move(
        &self,
        source: &GraphTarget,
        destination: &GraphTarget,
        _silent: bool,
    ) -> Result<()> {
        // MOVE = COPY + DROP source
        self.execute_copy(source, destination, true)?;
        self.execute_drop(source, true)?;
        Ok(())
    }

    /// Execute ADD operation
    fn execute_add(
        &self,
        source: &GraphTarget,
        destination: &GraphTarget,
        _silent: bool,
    ) -> Result<()> {
        // ADD is like COPY but doesn't clear destination first
        let source_quads = self.get_quads_from_target(source)?;
        for quad in source_quads {
            let dest_quad = self.move_quad_to_target(&quad, destination)?;
            self.store.insert_quad(dest_quad)?;
        }
        Ok(())
    }

    /// Find all quads matching a quad pattern
    fn find_matching_quads(&self, pattern: &QuadPattern) -> Result<Vec<Quad>> {
        // Convert pattern to query parameters and find matching quads
        let subject = self.term_pattern_to_subject(&pattern.subject)?;
        let predicate = self.term_pattern_to_predicate(&pattern.predicate)?;
        let object = self.term_pattern_to_object(&pattern.object)?;
        let graph = pattern
            .graph
            .as_ref()
            .map(|g| self.term_pattern_to_graph_name(g))
            .transpose()?;

        self.store.find_quads(
            subject.as_ref(),
            predicate.as_ref(),
            object.as_ref(),
            graph.as_ref(),
        )
    }

    /// Convert TermPattern to Subject (only if concrete)
    fn term_pattern_to_subject(
        &self,
        pattern: &TermPattern,
    ) -> Result<Option<crate::model::Subject>> {
        match pattern {
            TermPattern::NamedNode(n) => Ok(Some(crate::model::Subject::NamedNode(n.clone()))),
            TermPattern::BlankNode(b) => Ok(Some(crate::model::Subject::BlankNode(b.clone()))),
            TermPattern::Variable(_) => Ok(None), // Variables match anything
            TermPattern::Literal(_) => Err(OxirsError::Update(
                "Subject cannot be a literal".to_string(),
            )),
        }
    }

    /// Convert TermPattern to Predicate (only if concrete)
    fn term_pattern_to_predicate(
        &self,
        pattern: &TermPattern,
    ) -> Result<Option<crate::model::Predicate>> {
        match pattern {
            TermPattern::NamedNode(n) => Ok(Some(crate::model::Predicate::NamedNode(n.clone()))),
            TermPattern::Variable(_) => Ok(None), // Variables match anything
            TermPattern::BlankNode(_) | TermPattern::Literal(_) => Err(OxirsError::Update(
                "Predicate must be a named node".to_string(),
            )),
        }
    }

    /// Convert TermPattern to Object (only if concrete)
    fn term_pattern_to_object(
        &self,
        pattern: &TermPattern,
    ) -> Result<Option<crate::model::Object>> {
        match pattern {
            TermPattern::NamedNode(n) => Ok(Some(crate::model::Object::NamedNode(n.clone()))),
            TermPattern::BlankNode(b) => Ok(Some(crate::model::Object::BlankNode(b.clone()))),
            TermPattern::Literal(l) => Ok(Some(crate::model::Object::Literal(l.clone()))),
            TermPattern::Variable(_) => Ok(None), // Variables match anything
        }
    }

    /// Convert TermPattern to GraphName (only if concrete)
    fn term_pattern_to_graph_name(&self, pattern: &TermPattern) -> Result<GraphName> {
        match pattern {
            TermPattern::NamedNode(n) => Ok(GraphName::NamedNode(n.clone())),
            TermPattern::Variable(_) => Ok(GraphName::DefaultGraph), // Default for variables
            TermPattern::BlankNode(_) | TermPattern::Literal(_) => Err(OxirsError::Update(
                "Graph name must be a named node".to_string(),
            )),
        }
    }

    /// Evaluate a graph pattern to get variable bindings
    fn evaluate_graph_pattern(
        &self,
        pattern: &GraphPattern,
    ) -> Result<Vec<HashMap<String, crate::model::Term>>> {
        use crate::query::{QueryEngine, QueryResult};

        // Create a temporary SELECT query to evaluate the pattern
        let query_engine = QueryEngine::new();

        // Convert the graph pattern to a SPARQL query string
        let sparql_query = self.graph_pattern_to_sparql(pattern)?;

        // Execute the query
        match query_engine.query(&sparql_query, self.store)? {
            QueryResult::Select {
                variables: _,
                bindings,
            } => {
                // Convert the bindings to the expected format
                let mut solutions = Vec::new();
                for binding in bindings {
                    let mut solution = HashMap::new();
                    for (var_name, term) in binding {
                        solution.insert(var_name, term);
                    }
                    solutions.push(solution);
                }
                Ok(solutions)
            }
            _ => Err(OxirsError::Update(
                "Expected SELECT query result for WHERE clause evaluation".to_string(),
            )),
        }
    }

    /// Instantiate a quad pattern with variable bindings
    fn instantiate_quad_pattern(
        &self,
        pattern: &QuadPattern,
        solution: &HashMap<String, crate::model::Term>,
    ) -> Result<Option<Quad>> {
        use crate::model::*;

        // Instantiate subject
        let subject = match &pattern.subject {
            TermPattern::Variable(var) => {
                if let Some(term) = solution.get(var.name()) {
                    match term {
                        Term::NamedNode(n) => Subject::NamedNode(n.clone()),
                        Term::BlankNode(b) => Subject::BlankNode(b.clone()),
                        _ => return Ok(None), // Invalid subject type
                    }
                } else {
                    return Ok(None); // Unbound variable
                }
            }
            TermPattern::NamedNode(n) => Subject::NamedNode(n.clone()),
            TermPattern::BlankNode(b) => Subject::BlankNode(b.clone()),
            TermPattern::Literal(_) => return Ok(None), // Subject cannot be literal
        };

        // Instantiate predicate
        let predicate = match &pattern.predicate {
            TermPattern::Variable(var) => {
                if let Some(Term::NamedNode(n)) = solution.get(var.name()) {
                    Predicate::NamedNode(n.clone())
                } else {
                    return Ok(None); // Unbound variable or invalid predicate type
                }
            }
            TermPattern::NamedNode(n) => Predicate::NamedNode(n.clone()),
            _ => return Ok(None), // Predicate must be named node
        };

        // Instantiate object
        let object = match &pattern.object {
            TermPattern::Variable(var) => {
                if let Some(term) = solution.get(var.name()) {
                    match term {
                        Term::NamedNode(n) => Object::NamedNode(n.clone()),
                        Term::BlankNode(b) => Object::BlankNode(b.clone()),
                        Term::Literal(l) => Object::Literal(l.clone()),
                        _ => return Ok(None), // Invalid object type
                    }
                } else {
                    return Ok(None); // Unbound variable
                }
            }
            TermPattern::NamedNode(n) => Object::NamedNode(n.clone()),
            TermPattern::BlankNode(b) => Object::BlankNode(b.clone()),
            TermPattern::Literal(l) => Object::Literal(l.clone()),
        };

        // Instantiate graph name
        let graph_name = match &pattern.graph {
            Some(graph_pattern) => match graph_pattern {
                TermPattern::Variable(var) => {
                    if let Some(Term::NamedNode(n)) = solution.get(var.name()) {
                        GraphName::NamedNode(n.clone())
                    } else {
                        return Ok(None); // Unbound variable or invalid graph name
                    }
                }
                TermPattern::NamedNode(n) => GraphName::NamedNode(n.clone()),
                _ => return Ok(None), // Graph name must be named node
            },
            None => GraphName::DefaultGraph,
        };

        Ok(Some(Quad::new(subject, predicate, object, graph_name)))
    }

    /// Get all quads from a graph target
    fn get_quads_from_target(&self, target: &GraphTarget) -> Result<Vec<Quad>> {
        match target {
            GraphTarget::Default => {
                let graph = GraphName::DefaultGraph;
                self.store.find_quads(None, None, None, Some(&graph))
            }
            GraphTarget::Named(graph_name) => {
                let graph = GraphName::NamedNode(graph_name.clone());
                self.store.find_quads(None, None, None, Some(&graph))
            }
            GraphTarget::All => self.store.find_quads(None, None, None, None),
        }
    }

    /// Move a quad to a different graph target
    fn move_quad_to_target(&self, quad: &Quad, target: &GraphTarget) -> Result<Quad> {
        match target {
            GraphTarget::Default => Ok(Quad::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
                GraphName::DefaultGraph,
            )),
            GraphTarget::Named(graph_name) => Ok(Quad::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
                GraphName::NamedNode(graph_name.clone()),
            )),
            GraphTarget::All => {
                // For "ALL", we keep the original graph
                Ok(quad.clone())
            }
        }
    }

    /// Convert a graph pattern to a SPARQL query string
    fn graph_pattern_to_sparql(&self, pattern: &GraphPattern) -> Result<String> {
        use crate::query::algebra::*;

        match pattern {
            GraphPattern::Bgp(triple_patterns) => {
                let mut sparql = String::from("SELECT * WHERE { ");
                for (i, triple_pattern) in triple_patterns.iter().enumerate() {
                    if i > 0 {
                        sparql.push_str(" . ");
                    }
                    sparql.push_str(&self.triple_pattern_to_sparql(triple_pattern)?);
                }
                sparql.push_str(" }");
                Ok(sparql)
            }
            GraphPattern::Join(left, right) => {
                let left_sparql = self.graph_pattern_to_sparql(left)?;
                let right_sparql = self.graph_pattern_to_sparql(right)?;

                // Extract WHERE clause content from both patterns
                let left_where = self.extract_where_clause(&left_sparql)?;
                let right_where = self.extract_where_clause(&right_sparql)?;

                Ok(format!(
                    "SELECT * WHERE {{ {} . {} }}",
                    left_where, right_where
                ))
            }
            GraphPattern::Filter { expr, inner } => {
                let inner_sparql = self.graph_pattern_to_sparql(inner)?;
                let inner_where = self.extract_where_clause(&inner_sparql)?;
                let filter_expr = self.expression_to_sparql(expr)?;

                Ok(format!(
                    "SELECT * WHERE {{ {} FILTER ({}) }}",
                    inner_where, filter_expr
                ))
            }
            GraphPattern::Union(left, right) => {
                let left_sparql = self.graph_pattern_to_sparql(left)?;
                let right_sparql = self.graph_pattern_to_sparql(right)?;

                let left_where = self.extract_where_clause(&left_sparql)?;
                let right_where = self.extract_where_clause(&right_sparql)?;

                Ok(format!(
                    "SELECT * WHERE {{ {{ {} }} UNION {{ {} }} }}",
                    left_where, right_where
                ))
            }
            _ => Err(OxirsError::Update(format!(
                "Graph pattern type not yet supported in SPARQL conversion: {:?}",
                pattern
            ))),
        }
    }

    /// Convert a triple pattern to SPARQL syntax
    fn triple_pattern_to_sparql(&self, pattern: &AlgebraTriplePattern) -> Result<String> {
        let subject = self.term_pattern_to_sparql(&pattern.subject)?;
        let predicate = self.term_pattern_to_sparql(&pattern.predicate)?;
        let object = self.term_pattern_to_sparql(&pattern.object)?;

        Ok(format!("{} {} {}", subject, predicate, object))
    }

    /// Convert a term pattern to SPARQL syntax
    fn term_pattern_to_sparql(&self, pattern: &TermPattern) -> Result<String> {
        match pattern {
            TermPattern::Variable(var) => Ok(format!("?{}", var.name())),
            TermPattern::NamedNode(node) => Ok(format!("<{}>", node.as_str())),
            TermPattern::BlankNode(blank) => Ok(format!("_:{}", blank.as_str())),
            TermPattern::Literal(literal) => {
                if let Some(lang) = literal.language() {
                    Ok(format!("\"{}\"@{}", literal.value(), lang))
                } else if literal.datatype() != xsd::STRING.as_ref() {
                    Ok(format!("\"{}\"^^<{}>", literal.value(), literal.datatype()))
                } else {
                    Ok(format!("\"{}\"", literal.value()))
                }
            }
        }
    }

    /// Convert an expression to SPARQL syntax
    fn expression_to_sparql(&self, expr: &Expression) -> Result<String> {
        match expr {
            Expression::Variable(var) => Ok(format!("?{}", var.name())),
            Expression::Term(term) => match term {
                crate::model::Term::NamedNode(n) => Ok(format!("<{}>", n.as_str())),
                crate::model::Term::BlankNode(b) => Ok(format!("_:{}", b.as_str())),
                crate::model::Term::Literal(l) => {
                    if let Some(lang) = l.language() {
                        Ok(format!("\"{}\"@{}", l.value(), lang))
                    } else if l.datatype() != xsd::STRING.as_ref() {
                        Ok(format!("\"{}\"^^<{}>", l.value(), l.datatype()))
                    } else {
                        Ok(format!("\"{}\"", l.value()))
                    }
                }
                _ => Err(OxirsError::Update(
                    "Unsupported term type in expression".to_string(),
                )),
            },
            Expression::Equal(left, right) => {
                let left_sparql = self.expression_to_sparql(left)?;
                let right_sparql = self.expression_to_sparql(right)?;
                Ok(format!("({} = {})", left_sparql, right_sparql))
            }
            Expression::And(left, right) => {
                let left_sparql = self.expression_to_sparql(left)?;
                let right_sparql = self.expression_to_sparql(right)?;
                Ok(format!("({} && {})", left_sparql, right_sparql))
            }
            Expression::Or(left, right) => {
                let left_sparql = self.expression_to_sparql(left)?;
                let right_sparql = self.expression_to_sparql(right)?;
                Ok(format!("({} || {})", left_sparql, right_sparql))
            }
            Expression::Not(inner) => {
                let inner_sparql = self.expression_to_sparql(inner)?;
                Ok(format!("(!{})", inner_sparql))
            }
            _ => Err(OxirsError::Update(format!(
                "Expression type not yet supported in SPARQL conversion: {:?}",
                expr
            ))),
        }
    }

    /// Extract WHERE clause content from a SPARQL query
    fn extract_where_clause(&self, sparql: &str) -> Result<String> {
        if let Some(start) = sparql.find("WHERE {") {
            let where_start = start + 7; // Length of "WHERE {"
            if let Some(end) = sparql.rfind('}') {
                let where_content = &sparql[where_start..end].trim();
                Ok(where_content.to_string())
            } else {
                Err(OxirsError::Update(
                    "Malformed SPARQL query: missing closing brace".to_string(),
                ))
            }
        } else {
            Err(OxirsError::Update(
                "Malformed SPARQL query: missing WHERE clause".to_string(),
            ))
        }
    }
}

/// SPARQL UPDATE parser (simplified)
pub struct UpdateParser;

impl UpdateParser {
    /// Create a new update parser
    pub fn new() -> Self {
        Self
    }

    /// Parse a SPARQL UPDATE string into an Update struct
    pub fn parse(&self, update_str: &str) -> Result<Update> {
        // This is a very simplified parser that only handles basic cases
        // A full implementation would need a complete SPARQL UPDATE grammar parser

        let trimmed = update_str.trim();

        if trimmed.starts_with("INSERT DATA") {
            self.parse_insert_data(trimmed)
        } else if trimmed.starts_with("DELETE DATA") {
            self.parse_delete_data(trimmed)
        } else if trimmed.starts_with("CLEAR") {
            self.parse_clear(trimmed)
        } else {
            Err(OxirsError::Parse(format!(
                "Unsupported UPDATE operation: {}",
                trimmed
            )))
        }
    }

    /// Parse INSERT DATA operation
    fn parse_insert_data(&self, update_str: &str) -> Result<Update> {
        use crate::query::algebra::UpdateOperation;

        // Extract the data block from "INSERT DATA { ... }"
        let data_start = update_str.find('{');
        let data_end = update_str.rfind('}');

        if let (Some(start), Some(end)) = (data_start, data_end) {
            let data_block = &update_str[start + 1..end].trim();

            // Parse the quads from the data block
            let quads = self.parse_quad_data(data_block)?;

            Ok(Update {
                base: None,
                prefixes: HashMap::new(),
                operations: vec![UpdateOperation::InsertData { data: quads }],
            })
        } else {
            Err(OxirsError::Parse(
                "Malformed INSERT DATA: missing data block".to_string(),
            ))
        }
    }

    /// Parse DELETE DATA operation
    fn parse_delete_data(&self, update_str: &str) -> Result<Update> {
        use crate::query::algebra::UpdateOperation;

        // Extract the data block from "DELETE DATA { ... }"
        let data_start = update_str.find('{');
        let data_end = update_str.rfind('}');

        if let (Some(start), Some(end)) = (data_start, data_end) {
            let data_block = &update_str[start + 1..end].trim();

            // Parse the quads from the data block
            let quads = self.parse_quad_data(data_block)?;

            Ok(Update {
                base: None,
                prefixes: HashMap::new(),
                operations: vec![UpdateOperation::DeleteData { data: quads }],
            })
        } else {
            Err(OxirsError::Parse(
                "Malformed DELETE DATA: missing data block".to_string(),
            ))
        }
    }

    /// Parse quad data from a data block
    fn parse_quad_data(&self, _data_block: &str) -> Result<Vec<Quad>> {
        // For now, return empty vector - in a full implementation this would parse the quad data
        Ok(Vec::new())
    }

    /// Parse CLEAR operation
    fn parse_clear(&self, update_str: &str) -> Result<Update> {
        use crate::query::algebra::{GraphTarget, UpdateOperation};

        let trimmed = update_str.trim();
        let silent = trimmed.contains("SILENT");

        let graph_target = if trimmed.contains("DEFAULT") {
            GraphTarget::Default
        } else if trimmed.contains("ALL") {
            GraphTarget::All
        } else if let Some(graph_start) = trimmed.find("GRAPH") {
            // Extract the graph IRI
            let after_graph = &trimmed[graph_start + 5..].trim();
            if let Some(iri_start) = after_graph.find('<') {
                if let Some(iri_end) = after_graph.find('>') {
                    let iri_str = &after_graph[iri_start + 1..iri_end];
                    let graph_node = NamedNode::new(iri_str)
                        .map_err(|e| OxirsError::Parse(format!("Invalid graph IRI: {}", e)))?;
                    GraphTarget::Named(graph_node)
                } else {
                    return Err(OxirsError::Parse(
                        "Malformed graph IRI in CLEAR".to_string(),
                    ));
                }
            } else {
                return Err(OxirsError::Parse("Missing graph IRI in CLEAR".to_string()));
            }
        } else {
            GraphTarget::Default // Default if no target specified
        };

        Ok(Update {
            base: None,
            prefixes: HashMap::new(),
            operations: vec![UpdateOperation::Clear {
                graph: graph_target,
                silent,
            }],
        })
    }
}
