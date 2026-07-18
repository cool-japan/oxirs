//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{Algebra, Expression, Literal, TriplePattern, UnaryOperator, Variable};
use crate::update::{GraphReference, QuadPattern, UpdateOperation};
use anyhow::{bail, Result};
use oxirs_core::model::NamedNode;
use std::collections::HashMap;

use super::types::{Token, UpdateRequest};

use super::queryparser_type::QueryParser;

/// Validate the argument count of a SPARQL 1.1 built-in call at parse time, so a
/// malformed call (e.g. `REGEX(?x)`, `IF(?a, ?b)`) surfaces as a client parse
/// error (4xx) rather than failing deep in execution. `name` is the canonical
/// lower-case built-in name produced by `builtin_call_name`.
fn validate_builtin_arity(name: &str, argc: usize) -> Result<()> {
    let ok = match name {
        // No-argument built-ins.
        "now" | "rand" | "uuid" | "struuid" => argc == 0,
        // Zero or one argument.
        "bnode" => argc <= 1,
        // Exactly one argument.
        "str" | "lang" | "datatype" | "bound" | "iri" | "uri" | "abs" | "ceil" | "floor"
        | "round" | "strlen" | "ucase" | "lcase" | "encode_for_uri" | "year" | "month" | "day"
        | "hours" | "minutes" | "seconds" | "timezone" | "tz" | "md5" | "sha1" | "sha256"
        | "sha384" | "sha512" | "isiri" | "isuri" | "isblank" | "isliteral" | "isnumeric" => {
            argc == 1
        }
        // Exactly two arguments.
        "langmatches" | "contains" | "strstarts" | "strends" | "strbefore" | "strafter"
        | "strlang" | "strdt" | "sameterm" => argc == 2,
        // Two or three arguments.
        "regex" | "substr" => (2..=3).contains(&argc),
        // Exactly three arguments.
        "if" => argc == 3,
        // Three or four arguments.
        "replace" => (3..=4).contains(&argc),
        // At least one argument.
        "coalesce" => argc >= 1,
        // Variadic (zero or more): CONCAT.
        "concat" => true,
        // Any name not in the table imposes no arity constraint here.
        _ => true,
    };
    if ok {
        Ok(())
    } else {
        bail!("built-in {name} called with wrong number of arguments ({argc})")
    }
}

/// Take the single argument of a validated unary built-in. Arity is checked by
/// [`validate_builtin_arity`] before this is reached, so `args` holds exactly
/// one element; the empty-string fallback keeps the parser total without an
/// `unwrap`.
fn pop_single_arg(args: Vec<Expression>) -> Expression {
    args.into_iter()
        .next()
        .unwrap_or(Expression::Literal(Literal {
            value: String::new(),
            language: None,
            datatype: None,
        }))
}

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
            Some(Token::RdfLiteral {
                value,
                language,
                datatype,
            }) => {
                // A language-tagged or explicitly-typed literal in an expression,
                // e.g. `FILTER(?l = "hokkaido"@ja)` or `?x = "1"^^xsd:integer`.
                let value = value.clone();
                let language = language.clone();
                let datatype = datatype.clone();
                self.advance();
                let datatype = match datatype {
                    Some(raw) => Some(self.resolve_datatype(&raw)?),
                    None => None,
                };
                Ok(Expression::Literal(Literal {
                    value,
                    language,
                    datatype,
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
            Some(Token::BuiltIn(name)) => {
                let name = name.clone();
                self.advance();
                self.parse_builtin_call(&name)
            }
            Some(Token::PrefixedName(prefix, local)) => {
                let prefix = prefix.clone();
                let local = local.clone();
                let name = format!("{prefix}:{local}");
                self.advance();
                if self.match_token(&Token::LeftParen) {
                    let mut args = Vec::new();
                    // `COUNT(*)` in an expression context (e.g. `HAVING (COUNT(*)
                    // > 1)`): the star is the count-all form, carried as an empty
                    // argument list. It is only accepted when it is the sole
                    // token before `)`, so `SUM(?a * ?b)` (a multiplication) is
                    // untouched.
                    if matches!(self.peek(), Some(Token::Star) | Some(Token::Multiply))
                        && matches!(self.tokens.get(self.position + 1), Some(Token::RightParen))
                    {
                        self.advance(); // `*`
                        self.advance(); // `)`
                        return Ok(Expression::Function { name, args });
                    }
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
            // `EXISTS { GroupGraphPattern }` as a filter expression. `NOT EXISTS`
            // is `NOT` (a unary operator) applied to this, i.e. `!EXISTS { … }`,
            // and needs no separate arm.
            Some(Token::Exists) => {
                self.advance();
                self.expect_token(Token::LeftBrace)?;
                let pattern = self.parse_group_graph_pattern()?;
                self.expect_token(Token::RightBrace)?;
                Ok(Expression::Exists(Box::new(pattern)))
            }
            _ => bail!("Expected primary expression"),
        }
    }
    /// Parse a SPARQL 1.1 `BuiltInCall` whose name token has already been
    /// consumed. `name` is the canonical lower-case built-in name.
    ///
    /// The argument list is parsed with the ordinary expression grammar, its
    /// arity is validated, and the call is lowered to the AST shape the
    /// evaluator expects: the type-check predicates and `BOUND` become dedicated
    /// [`Expression`] variants (`Unary` / `Bound`), `IF` becomes `Conditional`,
    /// and every other built-in becomes an `Expression::Function` keyed by its
    /// canonical name (matching the evaluator's function table).
    pub(super) fn parse_builtin_call(&mut self, name: &str) -> Result<Expression> {
        self.expect_token(Token::LeftParen)?;
        let mut args = Vec::new();
        // A built-in with no arguments closes immediately, e.g. `NOW()`.
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
        validate_builtin_arity(name, args.len())?;

        // Lower to the dedicated AST variant when one exists, so the evaluator
        // reaches its native handler rather than the generic function table.
        match name {
            "isiri" | "isuri" => Ok(Expression::Unary {
                op: UnaryOperator::IsIri,
                operand: Box::new(pop_single_arg(args)),
            }),
            "isblank" => Ok(Expression::Unary {
                op: UnaryOperator::IsBlank,
                operand: Box::new(pop_single_arg(args)),
            }),
            "isliteral" => Ok(Expression::Unary {
                op: UnaryOperator::IsLiteral,
                operand: Box::new(pop_single_arg(args)),
            }),
            "isnumeric" => Ok(Expression::Unary {
                op: UnaryOperator::IsNumeric,
                operand: Box::new(pop_single_arg(args)),
            }),
            "bound" => match pop_single_arg(args) {
                Expression::Variable(var) => Ok(Expression::Bound(var)),
                _ => bail!("BOUND requires a variable argument"),
            },
            "if" => {
                let mut it = args.into_iter();
                let condition = Box::new(it.next().unwrap_or(Expression::Literal(Literal {
                    value: "false".to_string(),
                    language: None,
                    datatype: None,
                })));
                let then_expr = Box::new(it.next().unwrap_or(Expression::Literal(Literal {
                    value: String::new(),
                    language: None,
                    datatype: None,
                })));
                let else_expr = Box::new(it.next().unwrap_or(Expression::Literal(Literal {
                    value: String::new(),
                    language: None,
                    datatype: None,
                })));
                Ok(Expression::Conditional {
                    condition,
                    then_expr,
                    else_expr,
                })
            }
            _ => Ok(Expression::Function {
                name: name.to_string(),
                args,
            }),
        }
    }
    pub(super) fn parse_construct_template(&mut self) -> Result<Vec<TriplePattern>> {
        let mut triples = Vec::new();
        while !self.is_at_end() && !matches!(self.peek(), Some(Token::RightBrace)) {
            self.skip_whitespace_and_newlines();
            if matches!(self.peek(), Some(Token::RightBrace)) {
                break;
            }
            triples.extend(self.parse_triples_same_subject()?);
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
                        UpdateOperation::DeleteInsertWhere { using, .. } if using.is_none() => {
                            *using = Some(vec![graph_ref]);
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
