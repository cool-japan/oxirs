//! # QueryParser - SELECT projection-item parsing
//!
//! Parses parenthesized SELECT projections `( Expression AS ?var )`, including
//! the SPARQL 1.1 aggregate functions (`COUNT`, `SUM`, `MIN`, `MAX`, `AVG`,
//! `SAMPLE`, `GROUP_CONCAT`). Aggregates are recognised syntactically by the
//! leading function name followed by `(`; any other parenthesized head is
//! parsed by the shared expression grammar.

use crate::algebra::Aggregate;
use anyhow::{bail, Result};

use super::types::{ProjectionItem, Token};

use super::queryparser_type::QueryParser;

/// The seven SPARQL 1.1 set aggregate functions, distinguished from ordinary
/// function calls so their special argument grammar (`*`, `DISTINCT`,
/// `SEPARATOR=`) can be parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AggregateKind {
    Count,
    Sum,
    Min,
    Max,
    Avg,
    Sample,
    GroupConcat,
}

/// Map a bare (unprefixed) identifier to its aggregate kind, case-insensitively.
fn aggregate_kind(local: &str) -> Option<AggregateKind> {
    match local.to_ascii_uppercase().as_str() {
        "COUNT" => Some(AggregateKind::Count),
        "SUM" => Some(AggregateKind::Sum),
        "MIN" => Some(AggregateKind::Min),
        "MAX" => Some(AggregateKind::Max),
        "AVG" => Some(AggregateKind::Avg),
        "SAMPLE" => Some(AggregateKind::Sample),
        "GROUP_CONCAT" => Some(AggregateKind::GroupConcat),
        _ => None,
    }
}

impl QueryParser {
    /// Parse a parenthesized SELECT projection: `( Expression AS ?var )`.
    ///
    /// The leading `(` must be the current token. The inner head is parsed as an
    /// aggregate when it is one of the SPARQL set functions, otherwise via the
    /// ordinary expression grammar. A missing `AS ?var` is a parse error.
    pub(super) fn parse_projection_paren_item(&mut self) -> Result<ProjectionItem> {
        self.expect_token(Token::LeftParen)?;
        let aggregate = self.try_parse_aggregate()?;
        let item_head = match aggregate {
            Some(aggregate) => ProjectionHead::Aggregate(aggregate),
            None => ProjectionHead::Expression(self.parse_expression()?),
        };
        if !self.match_token(&Token::As) {
            bail!(
                "parenthesized SELECT projection requires 'AS ?var', found {:?}",
                self.peek()
            );
        }
        let alias = self.expect_variable()?;
        self.expect_token(Token::RightParen)?;
        Ok(match item_head {
            ProjectionHead::Aggregate(aggregate) => ProjectionItem::Aggregate { aggregate, alias },
            ProjectionHead::Expression(expr) => ProjectionItem::Expression { expr, alias },
        })
    }

    /// If the upcoming tokens form a SPARQL aggregate call (`NAME ( … )`),
    /// consume and return it; otherwise leave the cursor untouched and return
    /// `None` so the caller can fall back to the expression grammar.
    pub(super) fn try_parse_aggregate(&mut self) -> Result<Option<Aggregate>> {
        // Recognise an unprefixed aggregate name WITHOUT consuming, so the
        // fall-through to `parse_expression` sees an unchanged token stream.
        let kind = match self.peek() {
            Some(Token::PrefixedName(prefix, local)) if prefix.is_empty() => aggregate_kind(local),
            _ => None,
        };
        let Some(kind) = kind else {
            return Ok(None);
        };
        // A `(` MUST immediately follow the name for it to be an aggregate call;
        // otherwise it is a bare IRI/name and belongs to the expression grammar.
        if !matches!(self.tokens.get(self.position + 1), Some(Token::LeftParen)) {
            return Ok(None);
        }
        self.advance(); // aggregate name
        self.expect_token(Token::LeftParen)?;
        let distinct = self.match_token(&Token::Distinct);
        let aggregate = match kind {
            AggregateKind::Count => {
                // `COUNT(*)` counts solutions; `COUNT(expr)` counts bound values.
                if self.match_token(&Token::Star) || self.match_token(&Token::Multiply) {
                    Aggregate::Count {
                        distinct,
                        expr: None,
                    }
                } else {
                    let expr = self.parse_expression()?;
                    Aggregate::Count {
                        distinct,
                        expr: Some(expr),
                    }
                }
            }
            AggregateKind::Sum => Aggregate::Sum {
                distinct,
                expr: self.parse_expression()?,
            },
            AggregateKind::Min => Aggregate::Min {
                distinct,
                expr: self.parse_expression()?,
            },
            AggregateKind::Max => Aggregate::Max {
                distinct,
                expr: self.parse_expression()?,
            },
            AggregateKind::Avg => Aggregate::Avg {
                distinct,
                expr: self.parse_expression()?,
            },
            AggregateKind::Sample => Aggregate::Sample {
                distinct,
                expr: self.parse_expression()?,
            },
            AggregateKind::GroupConcat => {
                let expr = self.parse_expression()?;
                let separator = self.parse_group_concat_separator()?;
                Aggregate::GroupConcat {
                    distinct,
                    expr,
                    separator,
                }
            }
        };
        self.expect_token(Token::RightParen)?;
        Ok(Some(aggregate))
    }

    /// Parse the optional `; SEPARATOR = "…"` tail of a `GROUP_CONCAT` call.
    fn parse_group_concat_separator(&mut self) -> Result<Option<String>> {
        if !self.match_token(&Token::Semicolon) {
            return Ok(None);
        }
        match self.peek() {
            Some(Token::PrefixedName(prefix, local))
                if prefix.is_empty() && local.eq_ignore_ascii_case("SEPARATOR") =>
            {
                self.advance();
            }
            other => bail!(
                "expected 'SEPARATOR' after ';' in GROUP_CONCAT, found {:?}",
                other
            ),
        }
        self.expect_token(Token::Equal)?;
        match self.peek() {
            Some(Token::StringLiteral(sep)) => {
                let sep = sep.clone();
                self.advance();
                Ok(Some(sep))
            }
            other => bail!(
                "expected a string separator after 'SEPARATOR=' in GROUP_CONCAT, found {:?}",
                other
            ),
        }
    }
}

/// Internal carrier distinguishing the two parenthesized-projection heads while
/// the trailing `AS ?var` is still being parsed.
enum ProjectionHead {
    Aggregate(Aggregate),
    Expression(crate::algebra::Expression),
}
