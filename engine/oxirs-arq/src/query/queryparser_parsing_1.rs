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
            Some(Token::StringLiteral(value)) => {
                let value = value.clone();
                self.advance();
                Ok(Term::Literal(Literal {
                    value,
                    language: None,
                    datatype: None,
                }))
            }
            Some(Token::NumericLiteral(value)) => {
                let value = value.clone();
                self.advance();
                Ok(Term::Literal(Literal {
                    value,
                    language: None,
                    datatype: Some(NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#decimal",
                    )),
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
        Ok(Algebra::LeftJoin {
            left: Box::new(Algebra::Table),
            right: Box::new(pattern),
            filter: None,
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
        let silent = self.match_token(&Token::Not);
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
                    let term = self.parse_term()?;
                    binding.insert(var.clone(), term);
                }
                self.expect_token(Token::RightParen)?;
            } else if !variables.is_empty() {
                let term = self.parse_term()?;
                binding.insert(variables[0].clone(), term);
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
        let mut expr = self.parse_additive_expression()?;
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
}
