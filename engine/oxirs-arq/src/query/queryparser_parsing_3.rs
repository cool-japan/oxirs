//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{check_aggregate_arity, Expression, GroupCondition, OrderCondition};
use anyhow::Result;

use super::types::{Query, Token};

use super::queryparser_type::QueryParser;

/// Reject aggregate function calls with the wrong argument count inside a
/// `HAVING` condition at parse time.
///
/// `HAVING` is parsed by the generic expression grammar, which accepts any
/// argument count for a function call, so a malformed aggregate such as `SUM()`
/// or `COUNT(?a, ?b)` would otherwise parse cleanly and only fail deep in
/// execution — surfacing to the HTTP layer as a 500 instead of a 400 parse
/// error. This walk mirrors the aggregate-hoisting recursion in the executor
/// (`rewrite_having_aggregates`): it descends `Function` / `Binary` / `Unary` /
/// `Conditional` shapes and validates each function call via the shared
/// [`check_aggregate_arity`] helper, so parser and executor reject identically.
/// The walk is scoped strictly to the `HAVING` condition.
fn validate_having_aggregate_arity(expr: &Expression) -> Result<()> {
    match expr {
        Expression::Function { name, args } => {
            check_aggregate_arity(name, args.len()).map_err(|msg| anyhow::anyhow!(msg))?;
            for arg in args {
                validate_having_aggregate_arity(arg)?;
            }
            Ok(())
        }
        Expression::Binary { left, right, .. } => {
            validate_having_aggregate_arity(left)?;
            validate_having_aggregate_arity(right)
        }
        Expression::Unary { operand, .. } => validate_having_aggregate_arity(operand),
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            validate_having_aggregate_arity(condition)?;
            validate_having_aggregate_arity(then_expr)?;
            validate_having_aggregate_arity(else_expr)
        }
        _ => Ok(()),
    }
}

impl QueryParser {
    /// Parse a SPARQL query string into a Query AST
    pub fn parse(&mut self, query_str: &str) -> Result<Query> {
        self.tokenize(query_str)?;
        self.parse_query()
    }
    pub(super) fn parse_solution_modifiers(&mut self, query: &mut Query) -> Result<()> {
        if self.match_token(&Token::GroupBy) {
            // The tokenizer emits `GROUP` as `Token::GroupBy` and the trailing
            // `BY` as `Token::OrderBy`; swallow that stray keyword so the
            // grouping expression list is read rather than mistaken for the end
            // of the modifier (`is_solution_modifier_end` treats `OrderBy` as a
            // terminator).
            self.match_token(&Token::OrderBy);
            while !self.is_at_end() && !self.is_solution_modifier_end() {
                // A grouping condition is a bare `Var`, a `BuiltInCall` /
                // `FunctionCall`, or the parenthesised `'(' Expression ('AS'
                // Var)? ')'` form — where the `AS` alias lives INSIDE the
                // parentheses (`GROUP BY (LANG(?l) AS ?g)`).
                let (expr, alias) = if matches!(self.peek(), Some(Token::LeftParen)) {
                    self.advance(); // consume '('
                    let expr = self.parse_expression()?;
                    let alias = if self.match_token(&Token::As) {
                        Some(self.expect_variable()?)
                    } else {
                        None
                    };
                    self.expect_token(Token::RightParen)?;
                    (expr, alias)
                } else {
                    (self.parse_expression()?, None)
                };
                query.group_by.push(GroupCondition { expr, alias });
            }
        }
        if self.match_token(&Token::Having) {
            let having = self.parse_expression()?;
            validate_having_aggregate_arity(&having)?;
            query.having = Some(having);
        }
        if self.match_token(&Token::OrderBy) {
            // `ORDER` and its trailing `BY` both tokenize to `Token::OrderBy`;
            // swallow the second keyword before reading the order conditions.
            self.match_token(&Token::OrderBy);
            while !self.is_at_end() && !self.is_solution_modifier_end() {
                let ascending = if self.match_token(&Token::Desc) {
                    false
                } else {
                    self.match_token(&Token::Asc);
                    true
                };
                let expr = self.parse_expression()?;
                query.order_by.push(OrderCondition { expr, ascending });
            }
        }
        // `LimitOffsetClauses ::= LimitClause OffsetClause? | OffsetClause
        // LimitClause?` (SPARQL 1.1 §18.5): BOTH orders are legal. A fixed
        // LIMIT-then-OFFSET sequence silently drops the LIMIT of an
        // `OFFSET n LIMIT m` tail — the trailing `LIMIT` is never consumed, so
        // the query returns every row past the offset instead of `m` rows (an
        // HTTP-200 wrong answer). Read the two clauses in a loop that accepts
        // whichever keyword comes next, in either order, until neither appears.
        loop {
            if self.match_token(&Token::Limit) {
                query.limit = Some(self.parse_limit_offset_value("LIMIT")?);
            } else if self.match_token(&Token::Offset) {
                query.offset = Some(self.parse_limit_offset_value("OFFSET")?);
            } else {
                break;
            }
        }
        Ok(())
    }
    /// Read the mandatory non-negative integer argument of a `LIMIT` / `OFFSET`
    /// clause. A missing, non-numeric, non-integer or out-of-range value is a
    /// parse error (surfaced as a 4xx) rather than being silently dropped —
    /// which would otherwise return every row instead of the intended cap.
    fn parse_limit_offset_value(&mut self, keyword: &str) -> Result<usize> {
        match self.peek() {
            Some(Token::NumericLiteral(num)) => {
                let num = num.clone();
                let value = num.parse::<usize>().map_err(|_| {
                    anyhow::anyhow!("{keyword} requires a non-negative integer, got `{num}`")
                })?;
                self.advance();
                Ok(value)
            }
            other => Err(anyhow::anyhow!(
                "{keyword} requires an integer argument, got {other:?}"
            )),
        }
    }
}
