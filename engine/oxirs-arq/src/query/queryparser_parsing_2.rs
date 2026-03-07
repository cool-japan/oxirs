//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{Algebra, Expression, Literal, TriplePattern, Variable};
use crate::update::{GraphReference, QuadPattern, UpdateOperation};
use anyhow::{bail, Result};
use oxirs_core::model::NamedNode;
use std::collections::HashMap;

use super::types::{Token, UpdateRequest};

use super::queryparser_type::QueryParser;

impl QueryParser {
    pub(super) fn parse_additive_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_multiplicative_expression()?;
        while let Some(op) = self.match_additive_operator() {
            let right = self.parse_multiplicative_expression()?;
            expr = Expression::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }
    pub(super) fn parse_multiplicative_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_unary_expression()?;
        while let Some(op) = self.match_multiplicative_operator() {
            let right = self.parse_unary_expression()?;
            expr = Expression::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }
    pub(super) fn parse_unary_expression(&mut self) -> Result<Expression> {
        if let Some(op) = self.match_unary_operator() {
            let expr = self.parse_unary_expression()?;
            Ok(Expression::Unary {
                op,
                operand: Box::new(expr),
            })
        } else {
            self.parse_primary_expression()
        }
    }
    pub(super) fn parse_primary_expression(&mut self) -> Result<Expression> {
        match self.peek() {
            Some(Token::Variable(var)) => {
                let var = var.clone();
                self.advance();
                Ok(Expression::Variable(Variable::new(var)?))
            }
            Some(Token::Iri(iri)) => {
                let iri = iri.clone();
                self.advance();
                Ok(Expression::Iri(NamedNode::new_unchecked(iri)))
            }
            Some(Token::StringLiteral(value)) | Some(Token::NumericLiteral(value)) => {
                let value = value.clone();
                self.advance();
                Ok(Expression::Literal(Literal {
                    value,
                    language: None,
                    datatype: None,
                }))
            }
            Some(Token::BooleanLiteral(value)) => {
                let value = *value;
                self.advance();
                Ok(Expression::Literal(Literal {
                    value: value.to_string(),
                    language: None,
                    datatype: None,
                }))
            }
            Some(Token::LeftParen) => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect_token(Token::RightParen)?;
                Ok(expr)
            }
            Some(Token::PrefixedName(prefix, local)) => {
                let prefix = prefix.clone();
                let local = local.clone();
                let name = format!("{prefix}:{local}");
                self.advance();
                if self.match_token(&Token::LeftParen) {
                    let mut args = Vec::new();
                    while !self.match_token(&Token::RightParen) {
                        args.push(self.parse_expression()?);
                        if !self.match_token(&Token::Comma) {
                            self.expect_token(Token::RightParen)?;
                            break;
                        }
                    }
                    Ok(Expression::Function { name, args })
                } else {
                    let full_iri = if let Some(base) = self.prefixes.get(&prefix) {
                        format!("{base}{local}")
                    } else {
                        name
                    };
                    Ok(Expression::Iri(NamedNode::new_unchecked(full_iri)))
                }
            }
            _ => bail!("Expected primary expression"),
        }
    }
    pub(super) fn parse_construct_template(&mut self) -> Result<Vec<TriplePattern>> {
        let mut triples = Vec::new();
        while !self.is_at_end() && !matches!(self.peek(), Some(Token::RightBrace)) {
            let triple = self.parse_triple_pattern()?;
            triples.push(triple);
            if !self.match_token(&Token::Dot) {
                break;
            }
        }
        Ok(triples)
    }
    pub(super) fn expect_variable(&mut self) -> Result<Variable> {
        if let Some(Token::Variable(var)) = self.peek() {
            let var = var.clone();
            self.advance();
            Ok(Variable::new(var)?)
        } else {
            bail!("Expected variable")
        }
    }
    /// Parse UPDATE request with multiple operations
    pub(super) fn parse_update_request(&mut self) -> Result<UpdateRequest> {
        let mut update_request = UpdateRequest {
            operations: Vec::new(),
            prefixes: HashMap::new(),
            base_iri: None,
        };
        self.skip_whitespace();
        while let Some(token) = self.peek() {
            match token {
                Token::Prefix => {
                    self.advance();
                    let prefix = self.expect_prefixed_name()?.0;
                    let iri = self.expect_iri()?;
                    update_request.prefixes.insert(prefix.clone(), iri.clone());
                    self.prefixes.insert(prefix, iri);
                }
                Token::Base => {
                    self.advance();
                    let iri = self.expect_iri()?;
                    update_request.base_iri = Some(iri.clone());
                    self.base_iri = Some(iri);
                }
                _ => break,
            }
        }
        while !self.is_at_end() {
            self.skip_whitespace();
            let operation = match self.peek() {
                Some(Token::Insert) => self.parse_insert_operation()?,
                Some(Token::Delete) => self.parse_delete_operation()?,
                Some(Token::Clear) => self.parse_clear_operation()?,
                Some(Token::Drop) => self.parse_drop_operation()?,
                Some(Token::Create) => self.parse_create_operation()?,
                Some(Token::Load) => self.parse_load_operation()?,
                Some(Token::Copy) => self.parse_copy_operation()?,
                Some(Token::Move) => self.parse_move_operation()?,
                Some(Token::Add) => self.parse_add_operation()?,
                Some(Token::With) => {
                    self.advance();
                    let graph_iri = self.expect_iri()?;
                    let graph_ref = GraphReference::Iri(graph_iri);
                    let mut operation = match self.peek() {
                        Some(Token::Insert) => self.parse_insert_operation()?,
                        Some(Token::Delete) => self.parse_delete_operation()?,
                        _ => bail!("Expected INSERT or DELETE after WITH clause"),
                    };
                    match &mut operation {
                        UpdateOperation::DeleteInsertWhere { using, .. } => {
                            if using.is_none() {
                                *using = Some(vec![graph_ref]);
                            }
                        }
                        UpdateOperation::InsertWhere { template, .. } => {
                            for quad in template {
                                if quad.graph.is_none() {
                                    quad.graph = Some(graph_ref.clone());
                                }
                            }
                        }
                        UpdateOperation::DeleteWhere { .. } => {}
                        _ => {}
                    }
                    operation
                }
                Some(Token::Eof) => break,
                _ => bail!("Expected UPDATE operation"),
            };
            update_request.operations.push(operation);
            self.match_token(&Token::Semicolon);
            self.skip_whitespace();
        }
        Ok(update_request)
    }
    /// Parse INSERT WHERE operation
    pub(super) fn parse_insert_where(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::LeftBrace)?;
        let template = self.parse_quad_pattern_data()?;
        self.expect_token(Token::RightBrace)?;
        self.expect_token(Token::Where)?;
        self.expect_token(Token::LeftBrace)?;
        let where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;
        Ok(UpdateOperation::InsertWhere {
            pattern: Box::new(where_clause),
            template,
        })
    }
    /// Parse DELETE WHERE operation
    pub(super) fn parse_delete_where(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Where)?;
        self.expect_token(Token::LeftBrace)?;
        let patterns = self.parse_quad_pattern_data()?;
        self.expect_token(Token::RightBrace)?;
        let triple_patterns: Vec<TriplePattern> = patterns
            .into_iter()
            .map(|qp| TriplePattern::new(qp.subject, qp.predicate, qp.object))
            .collect();
        Ok(UpdateOperation::DeleteWhere {
            pattern: Box::new(Algebra::Bgp(triple_patterns)),
        })
    }
    /// Parse DELETE ... INSERT ... WHERE operation
    pub(super) fn parse_delete_insert_where(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::LeftBrace)?;
        let delete_patterns = self.parse_quad_pattern_data()?;
        self.expect_token(Token::RightBrace)?;
        let insert_patterns = if self.match_token(&Token::Insert) {
            self.expect_token(Token::LeftBrace)?;
            let patterns = self.parse_quad_pattern_data()?;
            self.expect_token(Token::RightBrace)?;
            Some(patterns)
        } else {
            None
        };
        self.expect_token(Token::Where)?;
        self.expect_token(Token::LeftBrace)?;
        let where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;
        if let Some(insert_patterns) = insert_patterns {
            Ok(UpdateOperation::DeleteInsertWhere {
                delete_template: delete_patterns,
                insert_template: insert_patterns,
                pattern: Box::new(where_clause),
                using: None,
            })
        } else {
            let triple_patterns: Vec<TriplePattern> = delete_patterns
                .into_iter()
                .map(|qp| TriplePattern::new(qp.subject, qp.predicate, qp.object))
                .collect();
            Ok(UpdateOperation::DeleteWhere {
                pattern: Box::new(Algebra::Bgp(triple_patterns)),
            })
        }
    }
    /// Parse quad data for INSERT/DELETE DATA
    pub(super) fn parse_quad_data(&mut self) -> Result<Vec<QuadPattern>> {
        let mut quads = Vec::new();
        while !self.is_at_end() && !matches!(self.peek(), Some(Token::RightBrace)) {
            let quad = self.parse_quad()?;
            quads.push(quad);
            self.match_token(&Token::Dot);
        }
        Ok(quads)
    }
}
