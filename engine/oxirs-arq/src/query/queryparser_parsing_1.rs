//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{Algebra, BinaryOperator, Expression, Literal, Term, Variable};
use anyhow::{bail, Result};
use oxirs_core::model::NamedNode;
use std::collections::HashMap;

use super::types::Token;

use super::queryparser_type::QueryParser;

/// Choose the XSD datatype of a numeric literal from its lexical form, matching
/// the SPARQL/Turtle grammar: a fractional/whole number containing an exponent
/// marker (`e`/`E`) is an `xsd:double`, one containing a decimal point (but no
/// exponent) is an `xsd:decimal`, and everything else is an `xsd:integer`. This
/// keeps term-based BGP matching correct — e.g. the object literal `42` matches
/// a stored `"42"^^xsd:integer` rather than being mis-typed as `xsd:decimal`.
fn numeric_literal_datatype(lexical: &str) -> &'static str {
    if lexical.contains(['e', 'E']) {
        "http://www.w3.org/2001/XMLSchema#double"
    } else if lexical.contains('.') {
        "http://www.w3.org/2001/XMLSchema#decimal"
    } else {
        "http://www.w3.org/2001/XMLSchema#integer"
    }
}

/// Split the FILTER(s) that a parsed OPTIONAL group carries at its top level off
/// the group pattern, folding them into a single conjunctive expression.
///
/// `parse_group_graph_pattern` collects a group's bare `FILTER`s by wrapping the
/// fully-joined group as `Filter { pattern, condition }` (nested once per
/// filter). SPARQL 1.1 §18.2.2.3 requires those filters to become the
/// enclosing `LeftJoin`'s filter — evaluated over the MERGED (left+right)
/// bindings — rather than staying nested inside the `right` operand where a
/// reference to a left-only variable would be unbound. Only TOP-LEVEL `Filter`
/// layers are peeled: a `FILTER` inside a nested subgroup is a join operand of
/// the OPTIONAL group and stays put.
fn split_optional_filter(pattern: Algebra) -> (Algebra, Option<Expression>) {
    let mut current = pattern;
    let mut condition: Option<Expression> = None;
    while let Algebra::Filter {
        pattern: inner,
        condition: c,
    } = current
    {
        condition = Some(match condition {
            Some(existing) => Expression::Binary {
                op: BinaryOperator::And,
                left: Box::new(c),
                right: Box::new(existing),
            },
            None => c,
        });
        current = *inner;
    }
    (current, condition)
}

impl QueryParser {
    pub(super) fn parse_term(&mut self) -> Result<Term> {
        self.skip_whitespace_and_newlines();
        match self.peek() {
            Some(Token::Variable(var)) => {
                let var = var.clone();
                self.advance();
                let variable = Variable::new(&var)?;
                self.variables.insert(variable.clone());
                Ok(Term::Variable(variable))
            }
            Some(Token::Iri(iri)) => {
                let iri = iri.clone();
                self.advance();
                Ok(Term::Iri(NamedNode::new_unchecked(iri)))
            }
            Some(Token::PrefixedName(prefix, local)) => {
                let prefix = prefix.clone();
                let local = local.clone();
                self.advance();
                let full_iri = self.resolve_prefixed_name(&prefix, &local)?;
                Ok(Term::Iri(NamedNode::new_unchecked(full_iri)))
            }
            Some(Token::A) => {
                // `a` in a term position is the rdf:type predicate shorthand.
                self.advance();
                Ok(Term::Iri(NamedNode::new_unchecked(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                )))
            }
            Some(Token::StringLiteral(value)) => {
                let value = value.clone();
                self.advance();
                Ok(Term::Literal(Literal {
                    value,
                    language: None,
                    datatype: None,
                }))
            }
            Some(Token::RdfLiteral {
                value,
                language,
                datatype,
            }) => {
                let value = value.clone();
                let language = language.clone();
                let datatype = datatype.clone();
                self.advance();
                let datatype = match datatype {
                    Some(raw) => Some(self.resolve_datatype(&raw)?),
                    None => None,
                };
                Ok(Term::Literal(Literal {
                    value,
                    language,
                    datatype,
                }))
            }
            Some(Token::NumericLiteral(value)) => {
                let value = value.clone();
                self.advance();
                let datatype = numeric_literal_datatype(&value);
                Ok(Term::Literal(Literal {
                    value,
                    language: None,
                    datatype: Some(NamedNode::new_unchecked(datatype)),
                }))
            }
            Some(Token::BooleanLiteral(value)) => {
                let value = *value;
                self.advance();
                Ok(Term::Literal(Literal {
                    value: value.to_string(),
                    language: None,
                    datatype: Some(NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            Some(Token::BlankNode(id)) => {
                let id = id.clone();
                self.advance();
                Ok(Term::BlankNode(id))
            }
            _ => bail!("Expected term"),
        }
    }
    pub(super) fn parse_optional_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Optional)?;
        self.expect_token(Token::LeftBrace)?;
        let pattern = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;
        // Lift the OPTIONAL group's own FILTER(s) into the LeftJoin filter so
        // they are evaluated over the merged left+right bindings (SPARQL 1.1
        // "Transform OPTIONAL" rule), instead of being evaluated standalone on
        // `right` where a left-only variable would be unbound.
        let (right, filter) = split_optional_filter(pattern);
        Ok(Algebra::LeftJoin {
            left: Box::new(Algebra::Table),
            right: Box::new(right),
            filter,
        })
    }
    pub(super) fn parse_union_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Union)?;
        let pattern = if self.match_token(&Token::LeftBrace) {
            let p = self.parse_group_graph_pattern()?;
            self.expect_token(Token::RightBrace)?;
            p
        } else {
            self.parse_graph_pattern()?
        };
        Ok(Algebra::Union {
            left: Box::new(Algebra::Table),
            right: Box::new(pattern),
        })
    }
    pub(super) fn parse_minus_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Minus)?;
        self.expect_token(Token::LeftBrace)?;
        let pattern = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;
        Ok(Algebra::Minus {
            left: Box::new(Algebra::Table),
            right: Box::new(pattern),
        })
    }
    pub(super) fn parse_filter_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Filter)?;
        let condition = self.parse_expression()?;
        Ok(Algebra::Filter {
            pattern: Box::new(Algebra::Table),
            condition,
        })
    }
    pub(super) fn parse_bind_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Bind)?;
        self.expect_token(Token::LeftParen)?;
        let expr = self.parse_expression()?;
        self.expect_token(Token::As)?;
        let var = self.expect_variable()?;
        self.expect_token(Token::RightParen)?;
        Ok(Algebra::Extend {
            pattern: Box::new(Algebra::Table),
            variable: var,
            expr,
        })
    }
    pub(super) fn parse_service_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Service)?;
        let silent = self.match_token(&Token::Silent);
        let endpoint = self.parse_term()?;
        self.expect_token(Token::LeftBrace)?;
        let pattern = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;
        Ok(Algebra::Service {
            endpoint,
            pattern: Box::new(pattern),
            silent,
        })
    }
    pub(super) fn parse_graph_pattern_named(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Graph)?;
        let graph = self.parse_term()?;
        self.expect_token(Token::LeftBrace)?;
        let pattern = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;
        Ok(Algebra::Graph {
            graph,
            pattern: Box::new(pattern),
        })
    }
    pub(super) fn parse_values_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Values)?;
        let mut variables = Vec::new();
        let mut bindings = Vec::new();
        if self.match_token(&Token::LeftParen) {
            while !self.match_token(&Token::RightParen) {
                variables.push(self.expect_variable()?);
            }
        } else {
            variables.push(self.expect_variable()?);
        }
        self.expect_token(Token::LeftBrace)?;
        while !self.match_token(&Token::RightBrace) {
            let mut binding = HashMap::new();
            if self.match_token(&Token::LeftParen) {
                for var in &variables {
                    // `UNDEF` leaves the variable unbound for this row: omit it
                    // from the binding map entirely rather than binding it to a
                    // literal/IRI (SPARQL 1.1 InlineData). A concrete term is
                    // parsed and inserted as usual.
                    if self.match_token(&Token::Undef) {
                        continue;
                    }
                    let term = self.parse_term()?;
                    binding.insert(var.clone(), term);
                }
                self.expect_token(Token::RightParen)?;
            } else if !variables.is_empty() {
                if self.match_token(&Token::Undef) {
                    // Single-variable row with an unbound value: leave the map
                    // empty so the variable is unbound for this row.
                } else {
                    let term = self.parse_term()?;
                    binding.insert(variables[0].clone(), term);
                }
            }
            bindings.push(binding);
        }
        Ok(Algebra::Values {
            variables,
            bindings,
        })
    }
    pub(super) fn parse_or_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_and_expression()?;
        while self.match_token(&Token::Or) {
            let right = self.parse_and_expression()?;
            expr = Expression::Binary {
                op: BinaryOperator::Or,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }
    pub(super) fn parse_and_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_equality_expression()?;
        while self.match_token(&Token::And) {
            let right = self.parse_equality_expression()?;
            expr = Expression::Binary {
                op: BinaryOperator::And,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }
    pub(super) fn parse_equality_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_relational_expression()?;
        while let Some(op) = self.match_equality_operator() {
            let right = self.parse_relational_expression()?;
            expr = Expression::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }
    pub(super) fn parse_relational_expression(&mut self) -> Result<Expression> {
        let expr = self.parse_additive_expression()?;
        // `Expression IN ( … )` / `Expression NOT IN ( … )`. The right operand is
        // an expression list, carried as a `list(…)` function call so the
        // evaluator's `collect_in_list` recognises it; membership reuses the
        // existing `terms_equal` semantics.
        if matches!(self.peek(), Some(Token::In)) {
            self.advance();
            let list = self.parse_expression_list()?;
            return Ok(Expression::Binary {
                op: BinaryOperator::In,
                left: Box::new(expr),
                right: Box::new(list),
            });
        }
        if matches!(self.peek(), Some(Token::Not))
            && matches!(self.tokens.get(self.position + 1), Some(Token::In))
        {
            self.advance(); // NOT
            self.advance(); // IN
            let list = self.parse_expression_list()?;
            return Ok(Expression::Binary {
                op: BinaryOperator::NotIn,
                left: Box::new(expr),
                right: Box::new(list),
            });
        }
        let mut expr = expr;
        while let Some(op) = self.match_relational_operator() {
            let right = self.parse_additive_expression()?;
            expr = Expression::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }
    /// Parse a parenthesised `ExpressionList` — `'(' ( Expression ( ',' … )* )? ')'`
    /// — as a `list(…)` function call. An empty list `()` yields a zero-argument
    /// `list()`, so `?x IN ()` is always false and `?x NOT IN ()` always true.
    pub(super) fn parse_expression_list(&mut self) -> Result<Expression> {
        self.expect_token(Token::LeftParen)?;
        let mut args = Vec::new();
        if !self.match_token(&Token::RightParen) {
            loop {
                args.push(self.parse_expression()?);
                if self.match_token(&Token::Comma) {
                    continue;
                }
                self.expect_token(Token::RightParen)?;
                break;
            }
        }
        Ok(Expression::Function {
            name: "list".to_string(),
            args,
        })
    }
}
