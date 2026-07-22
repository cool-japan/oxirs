//! SPARQL query parser
//!
//! This is a placeholder implementation that will be enhanced with full
//! SPARQL 1.1 parsing capabilities in future iterations.

use crate::model::{BlankNode, Literal, NamedNode, Variable};
use crate::query::algebra::{AlgebraTriplePattern, TermPattern as AlgebraTermPattern};
use crate::query::sparql_algebra::{
    Expression, GraphPattern, OrderExpression, TermPattern, TriplePattern,
};
use crate::query::sparql_query::Query;
use crate::OxirsError;
use std::collections::HashMap;

/// A SPARQL parser
#[derive(Debug, Clone, Default)]
pub struct SparqlParser {
    base_iri: Option<NamedNode>,
    prefixes: HashMap<String, NamedNode>,
}

impl SparqlParser {
    /// Creates a new SPARQL parser
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the base IRI for resolving relative IRIs
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, OxirsError> {
        self.base_iri = Some(NamedNode::new(base_iri.into())?);
        Ok(self)
    }

    /// Adds a prefix mapping
    pub fn with_prefix(
        mut self,
        prefix: impl Into<String>,
        iri: impl Into<String>,
    ) -> Result<Self, OxirsError> {
        self.prefixes
            .insert(prefix.into(), NamedNode::new(iri.into())?);
        Ok(self)
    }

    /// Parses a SPARQL query string - alias for parse_query
    pub fn parse(&self, query: &str) -> Result<Query, OxirsError> {
        self.parse_query(query)
    }

    /// Parses a SPARQL query string
    pub fn parse_query(&self, query: &str) -> Result<Query, OxirsError> {
        // This is a simplified parser for demonstration
        // Full implementation would use a proper parser generator

        let query = query.trim();

        // Very basic SELECT query detection
        if query.to_uppercase().starts_with("SELECT") {
            self.parse_select_query(query)
        } else if query.to_uppercase().starts_with("CONSTRUCT") {
            self.parse_construct_query(query)
        } else if query.to_uppercase().starts_with("ASK") {
            self.parse_ask_query(query)
        } else if query.to_uppercase().starts_with("DESCRIBE") {
            self.parse_describe_query(query)
        } else {
            Err(OxirsError::Parse(
                "Unsupported query form. Query must start with SELECT, CONSTRUCT, ASK, or DESCRIBE"
                    .to_string(),
            ))
        }
    }

    // Private helper methods for parsing different query forms

    fn parse_select_query(&self, query: &str) -> Result<Query, OxirsError> {
        let upper = query.to_uppercase();
        let where_start = upper
            .find("WHERE")
            .ok_or_else(|| OxirsError::Parse("SELECT query must have WHERE clause".to_string()))?;

        // ---- Projection / DISTINCT / REDUCED (between SELECT and WHERE) ----
        // "SELECT" is 6 chars; everything up to WHERE is the result specification.
        let select_spec = query[6..where_start].trim();
        let (distinct, reduced, projection) = self.parse_select_spec(select_spec)?;

        // ---- WHERE clause and the trailing solution modifiers ----
        let (where_clause, modifiers) = split_where_and_modifiers(&query[where_start + 5..])?;
        let inner = self.parse_where_clause(where_clause)?;
        let sol_mods = self.parse_solution_modifiers(modifiers)?;

        // Assemble the algebra following SPARQL evaluation order:
        // Slice( Distinct/Reduced( Project( OrderBy( WHERE ) ) ) )
        let mut pattern = inner;

        if !sol_mods.order_by.is_empty() {
            pattern = GraphPattern::OrderBy {
                inner: Box::new(pattern),
                expression: sol_mods.order_by,
            };
        }

        if let Some(vars) = projection {
            pattern = GraphPattern::Project {
                inner: Box::new(pattern),
                variables: vars,
            };
        }

        if distinct {
            pattern = GraphPattern::Distinct {
                inner: Box::new(pattern),
            };
        } else if reduced {
            pattern = GraphPattern::Reduced {
                inner: Box::new(pattern),
            };
        }

        if sol_mods.limit.is_some() || sol_mods.offset.is_some() {
            pattern = GraphPattern::Slice {
                inner: Box::new(pattern),
                start: sol_mods.offset.unwrap_or(0),
                length: sol_mods.limit,
            };
        }

        Ok(Query::Select {
            dataset: None,
            pattern,
            base_iri: self.base_iri.as_ref().map(|iri| iri.as_str().to_string()),
        })
    }

    /// Parse the result specification of a SELECT query (the text between the
    /// `SELECT` keyword and the `WHERE` clause).
    ///
    /// Returns `(distinct, reduced, projection)` where `projection` is `None`
    /// for `SELECT *` and `Some(vars)` for an explicit variable list.
    ///
    /// Fail-loud: expression projections (`(?x + 1 AS ?y)`, aggregates) are
    /// not yet supported and produce an explicit error rather than being
    /// silently ignored.
    #[allow(clippy::type_complexity)]
    fn parse_select_spec(
        &self,
        spec: &str,
    ) -> Result<(bool, bool, Option<Vec<Variable>>), OxirsError> {
        let mut rest = spec.trim();
        let mut distinct = false;
        let mut reduced = false;

        // Leading DISTINCT / REDUCED modifier (case-insensitive, whole word).
        let upper = rest.to_uppercase();
        if upper.starts_with("DISTINCT")
            && rest[8..].chars().next().map_or(true, |c| c.is_whitespace())
        {
            distinct = true;
            rest = rest[8..].trim();
        } else if upper.starts_with("REDUCED")
            && rest[7..].chars().next().map_or(true, |c| c.is_whitespace())
        {
            reduced = true;
            rest = rest[7..].trim();
        }

        if rest == "*" {
            return Ok((distinct, reduced, None));
        }

        if rest.is_empty() {
            return Err(OxirsError::Parse(
                "SELECT query must specify a projection ('*' or a variable list)".to_string(),
            ));
        }

        // Expression projections / aggregates are not supported yet: fail loud.
        if rest.contains('(') || rest.to_uppercase().contains(" AS ") {
            return Err(OxirsError::Parse(format!(
                "Unsupported SELECT projection (expressions/aggregates not yet supported): '{rest}'"
            )));
        }

        let mut variables = Vec::new();
        for token in rest.split_whitespace() {
            if let Some(stripped) = token.strip_prefix('?').or_else(|| token.strip_prefix('$')) {
                if stripped.is_empty() {
                    return Err(OxirsError::Parse(
                        "Empty variable name in SELECT projection".to_string(),
                    ));
                }
                variables.push(Variable::new(token)?);
            } else {
                return Err(OxirsError::Parse(format!(
                    "Invalid SELECT projection token (expected a variable): '{token}'"
                )));
            }
        }

        Ok((distinct, reduced, Some(variables)))
    }

    /// Parse the trailing solution modifiers of a query (everything after the
    /// closing `}` of the WHERE clause): `GROUP BY`, `HAVING`, `ORDER BY`,
    /// `LIMIT`, `OFFSET`.
    ///
    /// Fail-loud: `GROUP BY` / `HAVING` are not yet supported and produce an
    /// explicit error instead of being silently dropped.
    fn parse_solution_modifiers(&self, text: &str) -> Result<SolutionModifiers, OxirsError> {
        let mut mods = SolutionModifiers::default();
        let text = text.trim();
        if text.is_empty() {
            return Ok(mods);
        }

        let upper = text.to_uppercase();
        if upper.contains("GROUP BY") {
            return Err(OxirsError::Parse(
                "GROUP BY is not yet supported by this query engine".to_string(),
            ));
        }
        if let Some(pos) = find_keyword(&upper, "HAVING") {
            // Only reject a real HAVING clause (word boundary already ensured by
            // find_keyword). Fail loud rather than ignore the constraint.
            let _ = pos;
            return Err(OxirsError::Parse(
                "HAVING is not yet supported by this query engine".to_string(),
            ));
        }

        // Locate ORDER BY / LIMIT / OFFSET boundaries by keyword position.
        let order_pos = find_keyword(&upper, "ORDER BY");
        let limit_pos = find_keyword(&upper, "LIMIT");
        let offset_pos = find_keyword(&upper, "OFFSET");

        // ORDER BY spans from just after "ORDER BY" to the first of LIMIT/OFFSET
        // that follows it.
        if let Some(op) = order_pos {
            let start = op + "ORDER BY".len();
            let mut end = text.len();
            for cand in [limit_pos, offset_pos].into_iter().flatten() {
                if cand > op && cand < end {
                    end = cand;
                }
            }
            let order_text = text[start..end].trim();
            mods.order_by = self.parse_order_conditions(order_text)?;
        }

        if let Some(lp) = limit_pos {
            mods.limit = Some(parse_trailing_integer(text, &upper, lp, "LIMIT")?);
        }
        if let Some(op) = offset_pos {
            mods.offset = Some(parse_trailing_integer(text, &upper, op, "OFFSET")?);
        }

        Ok(mods)
    }

    fn parse_construct_query(&self, query: &str) -> Result<Query, OxirsError> {
        // Find CONSTRUCT template and WHERE clause
        let construct_start = query
            .to_uppercase()
            .find("CONSTRUCT")
            .expect("CONSTRUCT keyword should be present in construct query")
            + 9;
        let where_start = query.to_uppercase().find("WHERE").ok_or_else(|| {
            OxirsError::Parse("CONSTRUCT query must have WHERE clause".to_string())
        })?;

        // Parse template (simplified - just get the content between braces)
        let construct_clause = query[construct_start..where_start].trim();
        let algebra_template = self.parse_construct_template(construct_clause)?;
        let template: Vec<TriplePattern> = algebra_template
            .iter()
            .map(|p| self.convert_triple_pattern(p))
            .collect();

        // Parse WHERE clause
        let pattern = self.parse_where_clause(&query[where_start + 5..])?;

        Ok(Query::Construct {
            template,
            dataset: None,
            pattern,
            base_iri: self.base_iri.as_ref().map(|iri| iri.as_str().to_string()),
        })
    }

    fn parse_ask_query(&self, query: &str) -> Result<Query, OxirsError> {
        let where_start = query
            .to_uppercase()
            .find("WHERE")
            .ok_or_else(|| OxirsError::Parse("ASK query must have WHERE clause".to_string()))?;

        let pattern = self.parse_where_clause(&query[where_start + 5..])?;

        Ok(Query::Ask {
            dataset: None,
            pattern,
            base_iri: self.base_iri.as_ref().map(|iri| iri.as_str().to_string()),
        })
    }

    fn parse_describe_query(&self, query: &str) -> Result<Query, OxirsError> {
        let where_start = query.to_uppercase().find("WHERE").ok_or_else(|| {
            OxirsError::Parse("DESCRIBE query must have WHERE clause".to_string())
        })?;

        let pattern = self.parse_where_clause(&query[where_start + 5..])?;

        Ok(Query::Describe {
            dataset: None,
            pattern,
            base_iri: self.base_iri.as_ref().map(|iri| iri.as_str().to_string()),
        })
    }

    fn parse_construct_template(
        &self,
        template_text: &str,
    ) -> Result<Vec<AlgebraTriplePattern>, OxirsError> {
        let content = template_text.trim();
        if !content.starts_with('{') || !content.ends_with('}') {
            return Err(OxirsError::Parse(
                "CONSTRUCT template must be enclosed in {}".to_string(),
            ));
        }

        let content = content[1..content.len() - 1].trim();
        let mut triple_patterns: Vec<AlgebraTriplePattern> = Vec::new();

        // Split by periods, but respect IRI brackets
        let triple_strings = self.split_triples_by_period(content);

        for triple_str in triple_strings {
            let triple_str = triple_str.trim();
            if triple_str.is_empty() {
                continue;
            }
            // A CONSTRUCT template may only contain triple patterns. A FILTER
            // here is malformed SPARQL — fail loud instead of dropping it.
            if starts_with_keyword(triple_str, "FILTER") {
                return Err(OxirsError::Parse(
                    "FILTER is not allowed inside a CONSTRUCT template".to_string(),
                ));
            }

            // Parse triple pattern (subject predicate object)
            let parts: Vec<&str> = triple_str.split_whitespace().collect();
            if parts.len() < 3 {
                return Err(OxirsError::Parse(format!(
                    "Invalid triple pattern: '{triple_str}'"
                )));
            }

            let subject = self.parse_term_pattern(parts[0])?;
            let predicate = self.parse_term_pattern(parts[1])?;
            let object = self.parse_term_pattern(parts[2])?;

            // Validate subject pattern (literals can't be subjects)
            if matches!(subject, TermPattern::Literal(_)) {
                return Err(OxirsError::Parse("Literals cannot be subjects".to_string()));
            }

            // Validate predicate pattern (only named nodes and variables allowed)
            if !matches!(
                predicate,
                TermPattern::NamedNode(_) | TermPattern::Variable(_)
            ) {
                return Err(OxirsError::Parse(
                    "Predicates must be named nodes or variables".to_string(),
                ));
            }

            // Convert sparql_algebra::TermPattern to algebra::TermPattern
            let algebra_subject = self.convert_to_algebra_term(&subject)?;
            let algebra_predicate = self.convert_to_algebra_term(&predicate)?;
            let algebra_object = self.convert_to_algebra_term(&object)?;

            triple_patterns.push(AlgebraTriplePattern::new(
                algebra_subject,
                algebra_predicate,
                algebra_object,
            ));
        }

        Ok(triple_patterns)
    }

    // Helper method to convert sparql_algebra::TermPattern to algebra::TermPattern
    fn convert_to_algebra_term(
        &self,
        term: &TermPattern,
    ) -> Result<AlgebraTermPattern, OxirsError> {
        match term {
            TermPattern::NamedNode(n) => Ok(AlgebraTermPattern::NamedNode(n.clone())),
            TermPattern::BlankNode(b) => Ok(AlgebraTermPattern::BlankNode(b.clone())),
            TermPattern::Literal(l) => Ok(AlgebraTermPattern::Literal(l.clone())),
            TermPattern::Variable(v) => Ok(AlgebraTermPattern::Variable(v.clone())),
            #[cfg(feature = "sparql-12")]
            TermPattern::Triple(_) => Err(OxirsError::Parse(
                "Quoted triples not supported in construct templates".to_string(),
            )),
        }
    }

    fn parse_where_clause(&self, where_text: &str) -> Result<GraphPattern, OxirsError> {
        // Very simplified parsing - just extract basic triple patterns
        let content = where_text.trim();
        if !content.starts_with('{') || !content.ends_with('}') {
            return Err(OxirsError::Parse(
                "WHERE clause must be enclosed in {}".to_string(),
            ));
        }

        let content = content[1..content.len() - 1].trim();
        let mut triple_patterns: Vec<TriplePattern> = Vec::new();
        let mut filters: Vec<Expression> = Vec::new();

        // Split by periods, but respect IRI brackets and parentheses.
        let triple_strings = self.split_triples_by_period(content);

        for triple_str in triple_strings {
            let triple_str = triple_str.trim();
            if triple_str.is_empty() {
                continue;
            }

            // FILTER constraints must be parsed and applied, never dropped.
            // A dropped FILTER silently over-broadens results, so a FILTER we
            // cannot parse must surface as an explicit error (fail-loud).
            if starts_with_keyword(triple_str, "FILTER") {
                let expr_text = triple_str["FILTER".len()..].trim();
                filters.push(self.parse_filter_expression(expr_text)?);
                continue;
            }

            // Parse triple pattern (subject predicate object)
            let parts: Vec<&str> = triple_str.split_whitespace().collect();
            if parts.len() < 3 {
                return Err(OxirsError::Parse(format!(
                    "Invalid triple pattern: '{triple_str}'"
                )));
            }

            let subject = self.parse_term_pattern(parts[0])?;
            let predicate = self.parse_term_pattern(parts[1])?;
            let object = self.parse_term_pattern(parts[2])?;

            triple_patterns.push(TriplePattern::new(subject, predicate, object));
        }

        let mut pattern = GraphPattern::Bgp {
            patterns: triple_patterns,
        };

        // Wrap the BGP in a Filter for each parsed FILTER constraint.
        for expr in filters {
            pattern = GraphPattern::Filter {
                expr,
                inner: Box::new(pattern),
            };
        }

        Ok(pattern)
    }

    fn parse_term_pattern(&self, term: &str) -> Result<TermPattern, OxirsError> {
        if term.starts_with('?') || term.starts_with('$') {
            Variable::new(term).map(TermPattern::Variable)
        } else if term.starts_with('<') && term.ends_with('>') {
            let iri = &term[1..term.len() - 1];
            NamedNode::new(iri).map(TermPattern::NamedNode)
        } else if term.starts_with('"') && term.ends_with('"') {
            let value = &term[1..term.len() - 1];
            Ok(TermPattern::Literal(Literal::new(value)))
        } else if term.starts_with("_:") {
            BlankNode::new(term).map(TermPattern::BlankNode)
        } else if let Some(colon_pos) = term.find(':') {
            // Prefixed name
            let prefix = &term[..colon_pos];
            let local = &term[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                let iri = format!("{}{}", namespace.as_str(), local);
                NamedNode::new(iri).map(TermPattern::NamedNode)
            } else {
                Err(OxirsError::Parse(format!("Unknown prefix: {prefix}")))
            }
        } else {
            Err(OxirsError::Parse(format!("Invalid term pattern: {term}")))
        }
    }

    /// Convert algebra TermPattern to sparql_algebra TermPattern
    fn convert_term_pattern(&self, term: &AlgebraTermPattern) -> TermPattern {
        match term {
            AlgebraTermPattern::NamedNode(n) => TermPattern::NamedNode(n.clone()),
            AlgebraTermPattern::BlankNode(b) => TermPattern::BlankNode(b.clone()),
            AlgebraTermPattern::Literal(l) => TermPattern::Literal(l.clone()),
            AlgebraTermPattern::Variable(v) => TermPattern::Variable(v.clone()),
            AlgebraTermPattern::QuotedTriple(_) => {
                panic!("RDF-star quoted triples not yet supported in SPARQL algebra conversion")
            }
        }
    }

    /// Convert AlgebraTriplePattern to sparql_algebra TriplePattern
    fn convert_triple_pattern(&self, pattern: &AlgebraTriplePattern) -> TriplePattern {
        TriplePattern::new(
            self.convert_term_pattern(&pattern.subject),
            self.convert_term_pattern(&pattern.predicate),
            self.convert_term_pattern(&pattern.object),
        )
    }

    /// Convert sparql_algebra TermPattern back to algebra TermPattern
    #[allow(clippy::only_used_in_recursion)]
    fn convert_term_pattern_back(&self, term: &TermPattern) -> AlgebraTermPattern {
        match term {
            TermPattern::NamedNode(n) => AlgebraTermPattern::NamedNode(n.clone()),
            TermPattern::BlankNode(b) => AlgebraTermPattern::BlankNode(b.clone()),
            TermPattern::Literal(l) => AlgebraTermPattern::Literal(l.clone()),
            TermPattern::Variable(v) => AlgebraTermPattern::Variable(v.clone()),
            #[cfg(feature = "sparql-12")]
            TermPattern::Triple(triple_pattern) => {
                // RDF-star: Triple patterns in term position (quoted triples)
                // Convert the nested triple pattern recursively
                let subject = self.convert_term_pattern_back(&triple_pattern.subject);
                let predicate = self.convert_term_pattern_back(&triple_pattern.predicate);
                let object = self.convert_term_pattern_back(&triple_pattern.object);

                // Create a quoted triple pattern (RDF-star feature)
                // This represents a triple that appears as a term in another triple
                AlgebraTermPattern::QuotedTriple(Box::new(crate::query::AlgebraTriplePattern::new(
                    subject, predicate, object,
                )))
            }
        }
    }

    /// Convert sparql_algebra TriplePattern back to AlgebraTriplePattern
    pub fn convert_triple_pattern_back(&self, pattern: &TriplePattern) -> AlgebraTriplePattern {
        AlgebraTriplePattern::new(
            self.convert_term_pattern_back(&pattern.subject),
            self.convert_term_pattern_back(&pattern.predicate),
            self.convert_term_pattern_back(&pattern.object),
        )
    }

    /// Split triples by period while respecting IRI brackets
    fn split_triples_by_period(&self, content: &str) -> Vec<String> {
        let mut triples = Vec::new();
        let mut current = String::new();
        let mut in_iri = false;
        let mut in_literal = false;
        let mut escape_next = false;
        let mut paren_depth: usize = 0;

        for ch in content.chars() {
            if escape_next {
                current.push(ch);
                escape_next = false;
                continue;
            }

            match ch {
                '\\' => {
                    escape_next = true;
                    current.push(ch);
                }
                '<' if !in_literal => {
                    in_iri = true;
                    current.push(ch);
                }
                '>' if in_iri && !in_literal => {
                    in_iri = false;
                    current.push(ch);
                }
                '"' => {
                    in_literal = !in_literal;
                    current.push(ch);
                }
                '(' if !in_iri && !in_literal => {
                    paren_depth += 1;
                    current.push(ch);
                }
                ')' if !in_iri && !in_literal => {
                    paren_depth = paren_depth.saturating_sub(1);
                    current.push(ch);
                }
                // A '.' only terminates a triple at the top level: never inside
                // an IRI, a string literal, or a parenthesised FILTER expression
                // (which may legitimately contain '.' in decimals, IRIs, etc.).
                '.' if !in_iri && !in_literal && paren_depth == 0 => {
                    // End of triple
                    let trimmed = current.trim();
                    if !trimmed.is_empty() {
                        triples.push(trimmed.to_string());
                    }
                    current.clear();
                }
                _ => {
                    current.push(ch);
                }
            }
        }

        // Don't forget the last triple if there's no trailing period
        let trimmed = current.trim();
        if !trimmed.is_empty() {
            triples.push(trimmed.to_string());
        }

        triples
    }

    /// Parse the text of a FILTER constraint into a SPARQL algebra
    /// [`Expression`].
    ///
    /// Accepts an optionally parenthesised boolean expression, e.g.
    /// `(?o > 5 && ?name = "x")` or `?o > 5`. Fail-loud: any construct that
    /// cannot be parsed yields an explicit [`OxirsError::Parse`] rather than
    /// silently discarding the constraint.
    pub(crate) fn parse_filter_expression(&self, text: &str) -> Result<Expression, OxirsError> {
        let tokens = tokenize_expression(text)?;
        let mut parser = ExprParser {
            tokens,
            pos: 0,
            prefixes: &self.prefixes,
        };
        let expr = parser.parse_or()?;
        if parser.pos != parser.tokens.len() {
            return Err(OxirsError::Parse(format!(
                "Trailing tokens in FILTER expression: '{text}'"
            )));
        }
        Ok(expr)
    }

    /// Parse an `ORDER BY` condition list (the text following the `ORDER BY`
    /// keyword) into algebra [`OrderExpression`]s.
    fn parse_order_conditions(&self, text: &str) -> Result<Vec<OrderExpression>, OxirsError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(OxirsError::Parse(
                "ORDER BY requires at least one ordering condition".to_string(),
            ));
        }

        let tokens = tokenize_expression(text)?;
        let mut parser = ExprParser {
            tokens,
            pos: 0,
            prefixes: &self.prefixes,
        };

        let mut conditions = Vec::new();
        while parser.pos < parser.tokens.len() {
            let condition = match parser.peek() {
                Some(ExprToken::Ident(kw)) if kw.eq_ignore_ascii_case("ASC") => {
                    parser.pos += 1;
                    parser.expect(&ExprToken::LParen)?;
                    let inner = parser.parse_or()?;
                    parser.expect(&ExprToken::RParen)?;
                    OrderExpression::Asc(inner)
                }
                Some(ExprToken::Ident(kw)) if kw.eq_ignore_ascii_case("DESC") => {
                    parser.pos += 1;
                    parser.expect(&ExprToken::LParen)?;
                    let inner = parser.parse_or()?;
                    parser.expect(&ExprToken::RParen)?;
                    OrderExpression::Desc(inner)
                }
                _ => OrderExpression::Asc(parser.parse_primary()?),
            };
            conditions.push(condition);
        }

        Ok(conditions)
    }
}

/// Parsed trailing solution modifiers of a SELECT query.
#[derive(Debug, Default)]
struct SolutionModifiers {
    order_by: Vec<OrderExpression>,
    limit: Option<usize>,
    offset: Option<usize>,
}

/// A token in a FILTER / ORDER BY expression.
#[derive(Debug, Clone, PartialEq)]
enum ExprToken {
    /// A fully-formed term expression (variable, IRI, literal, boolean, number).
    Term(Expression),
    /// A bare identifier (function name or keyword such as `ASC`, `BOUND`).
    Ident(String),
    LParen,
    RParen,
    Comma,
    Or,
    And,
    Not,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
    Star,
    Slash,
}

/// Recursive-descent parser over a token stream, producing algebra
/// [`Expression`]s. Kept intentionally small: it covers the boolean/relational/
/// arithmetic core plus `BOUND`, and fails loud on everything else.
struct ExprParser<'a> {
    tokens: Vec<ExprToken>,
    pos: usize,
    prefixes: &'a HashMap<String, NamedNode>,
}

impl ExprParser<'_> {
    fn peek(&self) -> Option<&ExprToken> {
        self.tokens.get(self.pos)
    }

    fn expect(&mut self, tok: &ExprToken) -> Result<(), OxirsError> {
        if self.peek() == Some(tok) {
            self.pos += 1;
            Ok(())
        } else {
            Err(OxirsError::Parse(format!(
                "Expected {tok:?} in FILTER expression, found {:?}",
                self.peek()
            )))
        }
    }

    fn parse_or(&mut self) -> Result<Expression, OxirsError> {
        let mut left = self.parse_and()?;
        while self.peek() == Some(&ExprToken::Or) {
            self.pos += 1;
            let right = self.parse_and()?;
            left = Expression::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expression, OxirsError> {
        let mut left = self.parse_relational()?;
        while self.peek() == Some(&ExprToken::And) {
            self.pos += 1;
            let right = self.parse_relational()?;
            left = Expression::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_relational(&mut self) -> Result<Expression, OxirsError> {
        let left = self.parse_additive()?;
        let op = match self.peek() {
            Some(ExprToken::Eq) => Some(0),
            Some(ExprToken::Ne) => Some(1),
            Some(ExprToken::Lt) => Some(2),
            Some(ExprToken::Le) => Some(3),
            Some(ExprToken::Gt) => Some(4),
            Some(ExprToken::Ge) => Some(5),
            _ => None,
        };
        if let Some(op) = op {
            self.pos += 1;
            let right = self.parse_additive()?;
            let (l, r) = (Box::new(left), Box::new(right));
            Ok(match op {
                0 => Expression::Equal(l, r),
                1 => Expression::Not(Box::new(Expression::Equal(l, r))),
                2 => Expression::Less(l, r),
                3 => Expression::LessOrEqual(l, r),
                4 => Expression::Greater(l, r),
                _ => Expression::GreaterOrEqual(l, r),
            })
        } else {
            Ok(left)
        }
    }

    fn parse_additive(&mut self) -> Result<Expression, OxirsError> {
        let mut left = self.parse_multiplicative()?;
        loop {
            match self.peek() {
                Some(ExprToken::Plus) => {
                    self.pos += 1;
                    let right = self.parse_multiplicative()?;
                    left = Expression::Add(Box::new(left), Box::new(right));
                }
                Some(ExprToken::Minus) => {
                    self.pos += 1;
                    let right = self.parse_multiplicative()?;
                    left = Expression::Subtract(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expression, OxirsError> {
        let mut left = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(ExprToken::Star) => {
                    self.pos += 1;
                    let right = self.parse_unary()?;
                    left = Expression::Multiply(Box::new(left), Box::new(right));
                }
                Some(ExprToken::Slash) => {
                    self.pos += 1;
                    let right = self.parse_unary()?;
                    left = Expression::Divide(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expression, OxirsError> {
        match self.peek() {
            Some(ExprToken::Not) => {
                self.pos += 1;
                Ok(Expression::Not(Box::new(self.parse_unary()?)))
            }
            Some(ExprToken::Minus) => {
                self.pos += 1;
                let operand = self.parse_unary()?;
                // Fold a leading minus into a numeric literal so negative
                // constants are directly usable in comparisons.
                if let Expression::Literal(lit) = &operand {
                    if let Some(neg) = negate_numeric_literal(lit) {
                        return Ok(Expression::Literal(neg));
                    }
                }
                Ok(Expression::UnaryMinus(Box::new(operand)))
            }
            Some(ExprToken::Plus) => {
                self.pos += 1;
                Ok(self.parse_unary()?)
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expression, OxirsError> {
        match self.peek().cloned() {
            Some(ExprToken::LParen) => {
                self.pos += 1;
                let expr = self.parse_or()?;
                self.expect(&ExprToken::RParen)?;
                Ok(expr)
            }
            Some(ExprToken::Term(expr)) => {
                self.pos += 1;
                Ok(expr)
            }
            Some(ExprToken::Ident(name)) => {
                // Ident followed by '(' is a function call; otherwise it must be
                // a prefixed name resolvable via the parser's prefix map.
                if self.tokens.get(self.pos + 1) == Some(&ExprToken::LParen) {
                    self.parse_function(&name)
                } else {
                    self.pos += 1;
                    self.resolve_prefixed_name(&name)
                }
            }
            other => Err(OxirsError::Parse(format!(
                "Unexpected token in FILTER expression: {other:?}"
            ))),
        }
    }

    /// Resolve a prefixed name (`ex:foo`) into a [`NamedNode`] expression using
    /// the parser's declared prefixes. Fail-loud on unknown prefixes.
    fn resolve_prefixed_name(&self, name: &str) -> Result<Expression, OxirsError> {
        if let Some(colon) = name.find(':') {
            let prefix = &name[..colon];
            let local = &name[colon + 1..];
            if let Some(namespace) = self.prefixes.get(prefix) {
                let iri = format!("{}{}", namespace.as_str(), local);
                return Ok(Expression::NamedNode(NamedNode::new(iri)?));
            }
            return Err(OxirsError::Parse(format!(
                "Unknown prefix '{prefix}' in FILTER expression"
            )));
        }
        Err(OxirsError::Parse(format!(
            "Unsupported bare identifier '{name}' in FILTER expression"
        )))
    }

    fn parse_function(&mut self, name: &str) -> Result<Expression, OxirsError> {
        self.pos += 1; // consume the identifier
        self.expect(&ExprToken::LParen)?;

        // Collect argument expressions.
        let mut args = Vec::new();
        if self.peek() != Some(&ExprToken::RParen) {
            args.push(self.parse_or()?);
            while self.peek() == Some(&ExprToken::Comma) {
                self.pos += 1;
                args.push(self.parse_or()?);
            }
        }
        self.expect(&ExprToken::RParen)?;

        match name.to_ascii_uppercase().as_str() {
            "BOUND" => {
                if args.len() != 1 {
                    return Err(OxirsError::Parse(
                        "BOUND() expects exactly one variable argument".to_string(),
                    ));
                }
                match args.into_iter().next() {
                    Some(Expression::Variable(v)) => Ok(Expression::Bound(v)),
                    _ => Err(OxirsError::Parse(
                        "BOUND() argument must be a variable".to_string(),
                    )),
                }
            }
            other => Err(OxirsError::Parse(format!(
                "FILTER function '{other}' is not yet supported"
            ))),
        }
    }
}

/// Negate a numeric literal, returning `None` for non-numeric literals.
fn negate_numeric_literal(lit: &Literal) -> Option<Literal> {
    let dt = lit.datatype();
    let dt_str = dt.as_str();
    let is_numeric = matches!(
        dt_str,
        "http://www.w3.org/2001/XMLSchema#integer"
            | "http://www.w3.org/2001/XMLSchema#decimal"
            | "http://www.w3.org/2001/XMLSchema#double"
            | "http://www.w3.org/2001/XMLSchema#float"
    );
    if !is_numeric {
        return None;
    }
    let value = lit.value();
    let negated = if let Some(stripped) = value.strip_prefix('-') {
        stripped.to_string()
    } else {
        format!("-{value}")
    };
    Some(Literal::new_typed(
        negated,
        NamedNode::new_unchecked(dt_str),
    ))
}

/// Tokenize a FILTER / ORDER BY expression string.
fn tokenize_expression(input: &str) -> Result<Vec<ExprToken>, OxirsError> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        match c {
            '(' => {
                tokens.push(ExprToken::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(ExprToken::RParen);
                i += 1;
            }
            ',' => {
                tokens.push(ExprToken::Comma);
                i += 1;
            }
            '+' => {
                tokens.push(ExprToken::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(ExprToken::Minus);
                i += 1;
            }
            '*' => {
                tokens.push(ExprToken::Star);
                i += 1;
            }
            '/' => {
                tokens.push(ExprToken::Slash);
                i += 1;
            }
            '&' => {
                if chars.get(i + 1) == Some(&'&') {
                    tokens.push(ExprToken::And);
                    i += 2;
                } else {
                    return Err(OxirsError::Parse(
                        "Unexpected '&' in expression".to_string(),
                    ));
                }
            }
            '|' => {
                if chars.get(i + 1) == Some(&'|') {
                    tokens.push(ExprToken::Or);
                    i += 2;
                } else {
                    return Err(OxirsError::Parse(
                        "Unexpected '|' in expression".to_string(),
                    ));
                }
            }
            '=' => {
                tokens.push(ExprToken::Eq);
                i += 1;
            }
            '!' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(ExprToken::Ne);
                    i += 2;
                } else {
                    tokens.push(ExprToken::Not);
                    i += 1;
                }
            }
            '<' => {
                // '<' may open an IRI (`<http://...>`) or be a comparison operator.
                if let Some(end) = find_iri_end(&chars, i) {
                    let iri: String = chars[i + 1..end].iter().collect();
                    tokens.push(ExprToken::Term(Expression::NamedNode(NamedNode::new(iri)?)));
                    i = end + 1;
                } else if chars.get(i + 1) == Some(&'=') {
                    tokens.push(ExprToken::Le);
                    i += 2;
                } else {
                    tokens.push(ExprToken::Lt);
                    i += 1;
                }
            }
            '>' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(ExprToken::Ge);
                    i += 2;
                } else {
                    tokens.push(ExprToken::Gt);
                    i += 1;
                }
            }
            '?' | '$' => {
                let start = i;
                i += 1;
                while i < chars.len() && is_var_char(chars[i]) {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                tokens.push(ExprToken::Term(Expression::Variable(Variable::new(name)?)));
            }
            '"' | '\'' => {
                let (lit, next) = tokenize_string_literal(&chars, i)?;
                tokens.push(ExprToken::Term(Expression::Literal(lit)));
                i = next;
            }
            _ if c.is_ascii_digit()
                || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit())) =>
            {
                let (lit, next) = tokenize_number(&chars, i)?;
                tokens.push(ExprToken::Term(Expression::Literal(lit)));
                i = next;
            }
            _ if is_name_start(c) => {
                let (tok, next) = tokenize_name(&chars, i)?;
                tokens.push(tok);
                i = next;
            }
            other => {
                return Err(OxirsError::Parse(format!(
                    "Unexpected character '{other}' in FILTER expression"
                )));
            }
        }
    }

    Ok(tokens)
}

/// If `<` at `start` opens a valid single-line IRI ref, return the index of the
/// closing `>`. Returns `None` when it is a comparison operator instead.
fn find_iri_end(chars: &[char], start: usize) -> Option<usize> {
    let mut j = start + 1;
    while j < chars.len() {
        match chars[j] {
            '>' => return Some(j),
            // Whitespace or another '<' means this was a comparison operator.
            c if c.is_whitespace() => return None,
            '<' => return None,
            _ => j += 1,
        }
    }
    None
}

fn is_var_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_name_start(c: char) -> bool {
    c.is_alphabetic() || c == '_' || c == ':'
}

fn is_name_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == ':' || c == '-' || c == '.'
}

/// Tokenize a bare name: a prefixed name (`ex:foo`), a boolean literal, or a
/// function/keyword identifier.
fn tokenize_name(chars: &[char], start: usize) -> Result<(ExprToken, usize), OxirsError> {
    let mut i = start;
    while i < chars.len() && is_name_char(chars[i]) {
        i += 1;
    }
    // Trailing '.' is not part of the name (it is a triple terminator).
    while i > start && chars[i - 1] == '.' {
        i -= 1;
    }
    let name: String = chars[start..i].iter().collect();

    if name.eq_ignore_ascii_case("true") {
        return Ok((
            ExprToken::Term(Expression::Literal(Literal::new_typed(
                "true",
                crate::vocab::xsd::BOOLEAN.clone(),
            ))),
            i,
        ));
    }
    if name.eq_ignore_ascii_case("false") {
        return Ok((
            ExprToken::Term(Expression::Literal(Literal::new_typed(
                "false",
                crate::vocab::xsd::BOOLEAN.clone(),
            ))),
            i,
        ));
    }

    // Function names, keywords (ASC/DESC/BOUND) and prefixed names are all
    // emitted as `Ident`; the parser resolves them by context.
    Ok((ExprToken::Ident(name), i))
}

/// Tokenize a numeric literal, assigning the appropriate XSD datatype.
fn tokenize_number(chars: &[char], start: usize) -> Result<(Literal, usize), OxirsError> {
    let mut i = start;
    let mut has_dot = false;
    let mut has_exp = false;

    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() {
            i += 1;
        } else if c == '.' && !has_dot && !has_exp {
            // Only consume '.' if a digit follows (otherwise it is a terminator).
            if chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()) {
                has_dot = true;
                i += 1;
            } else {
                break;
            }
        } else if (c == 'e' || c == 'E') && !has_exp {
            has_exp = true;
            i += 1;
            if matches!(chars.get(i), Some('+') | Some('-')) {
                i += 1;
            }
        } else {
            break;
        }
    }

    let lexical: String = chars[start..i].iter().collect();
    let datatype = if has_exp {
        crate::vocab::xsd::DOUBLE.clone()
    } else if has_dot {
        crate::vocab::xsd::DECIMAL.clone()
    } else {
        crate::vocab::xsd::INTEGER.clone()
    };
    Ok((Literal::new_typed(lexical, datatype), i))
}

/// Tokenize a string literal with optional language tag or datatype.
fn tokenize_string_literal(chars: &[char], start: usize) -> Result<(Literal, usize), OxirsError> {
    let quote = chars[start];
    let mut i = start + 1;
    let mut value = String::new();

    while i < chars.len() {
        let c = chars[i];
        if c == '\\' {
            if let Some(&next) = chars.get(i + 1) {
                let unescaped = match next {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '"' => '"',
                    '\'' => '\'',
                    other => other,
                };
                value.push(unescaped);
                i += 2;
                continue;
            }
            return Err(OxirsError::Parse(
                "Unterminated escape in string literal".to_string(),
            ));
        }
        if c == quote {
            i += 1;
            break;
        }
        value.push(c);
        i += 1;
    }

    // Optional language tag or datatype.
    if chars.get(i) == Some(&'@') {
        let lang_start = i + 1;
        let mut j = lang_start;
        while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '-') {
            j += 1;
        }
        let lang: String = chars[lang_start..j].iter().collect();
        let lit = Literal::new_lang(value, lang)
            .map_err(|e| OxirsError::Parse(format!("Invalid language tag: {e}")))?;
        return Ok((lit, j));
    }
    if chars.get(i) == Some(&'^') && chars.get(i + 1) == Some(&'^') {
        let mut j = i + 2;
        if chars.get(j) == Some(&'<') {
            if let Some(end) = find_iri_end(chars, j) {
                let iri: String = chars[j + 1..end].iter().collect();
                let lit = Literal::new_typed(value, NamedNode::new(iri)?);
                return Ok((lit, end + 1));
            }
        }
        // Prefixed datatype: consume a name; resolution is not supported here,
        // so fail loud rather than silently produce a plain string.
        while j < chars.len() && is_name_char(chars[j]) {
            j += 1;
        }
        return Err(OxirsError::Parse(
            "Prefixed datatypes on FILTER string literals are not yet supported".to_string(),
        ));
    }

    Ok((Literal::new(value), i))
}

/// Split the text following the `WHERE` keyword into the braced group pattern
/// (including its enclosing braces) and the trailing solution-modifier text.
fn split_where_and_modifiers(after_where: &str) -> Result<(&str, &str), OxirsError> {
    let bytes = after_where.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    if idx >= bytes.len() || bytes[idx] != b'{' {
        return Err(OxirsError::Parse(
            "WHERE clause must be enclosed in {}".to_string(),
        ));
    }

    let open = idx;
    let mut depth = 0i32;
    let mut in_iri = false;
    let mut in_literal = false;
    let mut escape = false;
    let mut close = None;

    for (offset, ch) in after_where[open..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' => escape = true,
            '"' if !in_iri => in_literal = !in_literal,
            '<' if !in_literal => in_iri = true,
            '>' if in_iri && !in_literal => in_iri = false,
            '{' if !in_iri && !in_literal => depth += 1,
            '}' if !in_iri && !in_literal => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + offset);
                    break;
                }
            }
            _ => {}
        }
    }

    let close =
        close.ok_or_else(|| OxirsError::Parse("Unbalanced braces in WHERE clause".to_string()))?;

    let where_clause = &after_where[open..=close];
    let modifiers = after_where[close + 1..].trim();
    Ok((where_clause, modifiers))
}

/// Returns the byte offset of `keyword` in `upper` (an already-uppercased
/// haystack) when it appears as a whole token, or `None`.
fn find_keyword(upper: &str, keyword: &str) -> Option<usize> {
    let mut search_from = 0;
    while let Some(rel) = upper[search_from..].find(keyword) {
        let pos = search_from + rel;
        let before_ok = pos == 0
            || !upper.as_bytes()[pos - 1].is_ascii_alphanumeric()
                && upper.as_bytes()[pos - 1] != b'_';
        let after = pos + keyword.len();
        let after_ok = after >= upper.len()
            || !upper.as_bytes()[after].is_ascii_alphanumeric() && upper.as_bytes()[after] != b'_';
        if before_ok && after_ok {
            return Some(pos);
        }
        search_from = pos + keyword.len();
    }
    None
}

/// Parse the non-negative integer that follows a `LIMIT` / `OFFSET` keyword.
fn parse_trailing_integer(
    text: &str,
    upper: &str,
    keyword_pos: usize,
    keyword: &str,
) -> Result<usize, OxirsError> {
    let _ = upper;
    let after = keyword_pos + keyword.len();
    let rest = text[after..].trim_start();
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return Err(OxirsError::Parse(format!(
            "{keyword} must be followed by a non-negative integer"
        )));
    }
    digits
        .parse::<usize>()
        .map_err(|e| OxirsError::Parse(format!("Invalid {keyword} value '{digits}': {e}")))
}

/// Case-insensitive whole-word keyword check at the start of a fragment.
fn starts_with_keyword(fragment: &str, keyword: &str) -> bool {
    let trimmed = fragment.trim_start();
    if trimmed.len() < keyword.len() {
        return false;
    }
    if !trimmed[..keyword.len()].eq_ignore_ascii_case(keyword) {
        return false;
    }
    trimmed[keyword.len()..]
        .chars()
        .next()
        .map_or(true, |c| !c.is_alphanumeric() && c != '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select_query() {
        let parser = SparqlParser::new();
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }";
        let result = parser.parse_query(query);
        assert!(result.is_ok());

        if let Ok(Query::Select { pattern, .. }) = result {
            // An explicit projection now wraps the BGP in a Project node.
            let bgp = match pattern {
                GraphPattern::Project { inner, variables } => {
                    assert_eq!(variables.len(), 3);
                    *inner
                }
                other => other,
            };
            match bgp {
                GraphPattern::Bgp { patterns } => {
                    assert_eq!(patterns.len(), 1);
                    // Verify it's a triple pattern with variables
                    let triple = &patterns[0];
                    assert!(matches!(triple.subject, TermPattern::Variable(_)));
                    assert!(matches!(triple.predicate, TermPattern::Variable(_)));
                    assert!(matches!(triple.object, TermPattern::Variable(_)));
                }
                _ => panic!("Expected BGP pattern"),
            }
        } else {
            panic!("Expected SELECT query");
        }
    }

    #[test]
    fn test_ask_query() {
        let parser = SparqlParser::new();
        let query = "ASK WHERE { ?s ?p ?o . }";
        let result = parser.parse_query(query);
        assert!(result.is_ok());

        if let Ok(Query::Ask { pattern, .. }) = result {
            match pattern {
                GraphPattern::Bgp { patterns } => {
                    assert_eq!(patterns.len(), 1);
                }
                _ => panic!("Expected BGP pattern"),
            }
        } else {
            panic!("Expected ASK query");
        }
    }

    #[test]
    fn test_construct_query() {
        let parser = SparqlParser::new();
        let query = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o . }";
        let result = parser.parse_query(query);
        assert!(result.is_ok());

        if let Ok(Query::Construct {
            template, pattern, ..
        }) = result
        {
            assert_eq!(template.len(), 1);
            match pattern {
                GraphPattern::Bgp { patterns } => {
                    assert_eq!(patterns.len(), 1);
                }
                _ => panic!("Expected BGP pattern"),
            }
        } else {
            panic!("Expected CONSTRUCT query");
        }
    }

    #[test]
    fn test_parse_with_prefix() {
        let parser = SparqlParser::new()
            .with_prefix("ex", "http://example.org/")
            .expect("operation should succeed");

        let query = "SELECT ?s WHERE { ex:subject ?p ?o . }";
        let result = parser.parse_query(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_query() {
        let parser = SparqlParser::new();
        let query = "INVALID QUERY";
        let result = parser.parse_query(query);
        assert!(result.is_err());
    }

    // ---- Regression tests for the core-query hardening findings ----

    /// P0: a FILTER must be parsed into a `GraphPattern::Filter`, never dropped.
    #[test]
    fn regression_filter_not_dropped() {
        let parser = SparqlParser::new();
        let query = "SELECT ?s WHERE { ?s ?p ?o . FILTER(?o > 5) }";
        let parsed = parser.parse_query(query).expect("should parse");
        let Query::Select { pattern, .. } = parsed else {
            panic!("expected SELECT");
        };
        // Project( Filter( Bgp ) )
        let inner = match pattern {
            GraphPattern::Project { inner, .. } => *inner,
            other => other,
        };
        match inner {
            GraphPattern::Filter { expr, .. } => {
                assert!(matches!(expr, Expression::Greater(_, _)));
            }
            other => panic!("expected Filter, got {other:?}"),
        }
    }

    /// P0: a FILTER whose expression cannot be parsed must fail loud, not drop.
    #[test]
    fn regression_filter_unparseable_fails_loud() {
        let parser = SparqlParser::new();
        // '@@@' is not a valid expression token.
        let query = "SELECT ?s WHERE { ?s ?p ?o . FILTER(@@@) }";
        assert!(parser.parse_query(query).is_err());
    }

    /// P0: a FILTER inside a CONSTRUCT template is malformed and must error.
    #[test]
    fn regression_filter_in_construct_template_errors() {
        let parser = SparqlParser::new();
        let query = "CONSTRUCT { ?s ?p ?o . FILTER(?o > 1) } WHERE { ?s ?p ?o . }";
        assert!(parser.parse_query(query).is_err());
    }

    /// FILTER with a string comparison parses into the expected algebra.
    #[test]
    fn regression_filter_string_comparison_parses() {
        let parser = SparqlParser::new();
        let query = "SELECT ?s WHERE { ?s ?p ?name . FILTER(?name > \"m\") }";
        let parsed = parser.parse_query(query).expect("should parse");
        let Query::Select { pattern, .. } = parsed else {
            panic!("expected SELECT");
        };
        let inner = match pattern {
            GraphPattern::Project { inner, .. } => *inner,
            other => other,
        };
        assert!(matches!(inner, GraphPattern::Filter { .. }));
    }

    /// FILTER with a decimal literal must not be split on the decimal point.
    #[test]
    fn regression_filter_decimal_not_split() {
        let parser = SparqlParser::new();
        let query = "SELECT ?s WHERE { ?s ?p ?o . FILTER(?o >= 5.5) }";
        let parsed = parser
            .parse_query(query)
            .expect("should parse decimal filter");
        let Query::Select { pattern, .. } = parsed else {
            panic!("expected SELECT");
        };
        let inner = match pattern {
            GraphPattern::Project { inner, .. } => *inner,
            other => other,
        };
        match inner {
            GraphPattern::Filter { expr, .. } => {
                assert!(matches!(expr, Expression::GreaterOrEqual(_, _)));
            }
            other => panic!("expected Filter, got {other:?}"),
        }
    }

    /// P1: projection, DISTINCT, ORDER BY and LIMIT/OFFSET must be captured.
    #[test]
    fn regression_projection_and_modifiers_parsed() {
        let parser = SparqlParser::new();
        let query =
            "SELECT DISTINCT ?name WHERE { ?s ?p ?name . } ORDER BY ?name LIMIT 10 OFFSET 2";
        let parsed = parser.parse_query(query).expect("should parse");
        let Query::Select { pattern, .. } = parsed else {
            panic!("expected SELECT");
        };
        // Slice( Distinct( Project( OrderBy( Bgp ) ) ) )
        let (start, length, inner) = match pattern {
            GraphPattern::Slice {
                inner,
                start,
                length,
            } => (start, length, *inner),
            other => panic!("expected Slice, got {other:?}"),
        };
        assert_eq!(start, 2);
        assert_eq!(length, Some(10));

        let distinct_inner = match inner {
            GraphPattern::Distinct { inner } => *inner,
            other => panic!("expected Distinct, got {other:?}"),
        };
        let project_inner = match distinct_inner {
            GraphPattern::Project { inner, variables } => {
                assert_eq!(variables.len(), 1);
                assert_eq!(variables[0].name(), "name");
                *inner
            }
            other => panic!("expected Project, got {other:?}"),
        };
        assert!(matches!(project_inner, GraphPattern::OrderBy { .. }));
    }

    /// P1: an unsupported modifier (GROUP BY) must fail loud, not be ignored.
    #[test]
    fn regression_group_by_fails_loud() {
        let parser = SparqlParser::new();
        let query = "SELECT ?s WHERE { ?s ?p ?o . } GROUP BY ?s";
        assert!(parser.parse_query(query).is_err());
    }

    /// SELECT * keeps its all-variables semantics (no Project wrapper).
    #[test]
    fn regression_select_star_no_projection() {
        let parser = SparqlParser::new();
        let query = "SELECT * WHERE { ?s ?p ?o . }";
        let parsed = parser.parse_query(query).expect("should parse");
        let Query::Select { pattern, .. } = parsed else {
            panic!("expected SELECT");
        };
        assert!(matches!(pattern, GraphPattern::Bgp { .. }));
    }

    /// LIMIT without a following integer is an error.
    #[test]
    fn regression_limit_requires_integer() {
        let parser = SparqlParser::new();
        let query = "SELECT ?s WHERE { ?s ?p ?o . } LIMIT";
        assert!(parser.parse_query(query).is_err());
    }
}
