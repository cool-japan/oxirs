//! # QueryParser - SPARQL 1.1 SubSelect (`{ SELECT … }`)
//!
//! A `SubSelect` (SPARQL 1.1 §8.2.4) is an independent, **non-correlated**
//! query nested inside an enclosing group graph pattern. Its result set is
//! evaluated on its own and then joined into the surrounding pattern on the
//! shared (projected) variables — correlation, when needed, is expressed with
//! `EXISTS` / `MINUS`, never with a SubSelect.
//!
//! The parser lowers a SubSelect to an ordinary [`Algebra`] tree — a
//! `Project` over the WHERE algebra, wrapped with the same solution-modifier
//! stack the top-level query uses (`Group` → `Having` → `Extend` → `OrderBy`
//! → `Project` → `Distinct` → `Slice`). Because `execute_serial` already
//! evaluates every one of those nodes recursively, and a `Join` evaluates both
//! operands through `execute_serial`, no executor wiring is required: the
//! lowered tree joins into the outer BGP and Just Works. Projection restricts
//! the visible columns, so an inner variable that is not projected never leaks
//! to the outer pattern.

use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};

use crate::algebra::{Aggregate, Algebra, Expression, Variable};

use super::types::{DatasetClause, ProjectionItem, Query, QueryType, Token};

use super::queryparser_type::QueryParser;

/// Build an empty [`Query`] shell for a nested SubSelect. The prologue
/// (prefixes / base) is query-wide, so resolution keeps using the parser's own
/// `self.prefixes`; these fields exist only to keep the throwaway record
/// self-consistent.
fn empty_subquery(query_type: QueryType) -> Query {
    Query {
        query_type,
        select_variables: Vec::new(),
        where_clause: Algebra::Zero,
        order_by: Vec::new(),
        group_by: Vec::new(),
        having: None,
        limit: None,
        offset: None,
        distinct: false,
        reduced: false,
        construct_template: Vec::new(),
        prefixes: HashMap::new(),
        base_iri: None,
        dataset: DatasetClause::default(),
        projection_items: Vec::new(),
        describe_targets: Vec::new(),
        describe_all: false,
    }
}

impl QueryParser {
    /// Parse a `SubSelect` positioned at its leading `SELECT` (or `ASK`) token.
    ///
    /// The caller has already consumed the opening `{` and is responsible for
    /// consuming the closing `}`. The token stream and prologue are shared with
    /// the outer parser, so the cursor advances in place and declared prefixes
    /// are inherited.
    pub(super) fn parse_sub_select(&mut self) -> Result<Algebra> {
        self.skip_whitespace_and_newlines();
        match self.peek() {
            Some(Token::Select) => {
                let mut sub = empty_subquery(QueryType::Select);
                self.parse_select_query(&mut sub)?;
                self.build_subquery_algebra(&sub)
            }
            // `ASK` is not a SubSelect in the SPARQL grammar (SubSelect is
            // SELECT only). Accept it so `{ ASK { … } }` parses rather than
            // 400-ing, but treat it as its bare WHERE pattern — no projection
            // and no join-time boolean semantics.
            Some(Token::Ask) => {
                let mut sub = empty_subquery(QueryType::Ask);
                self.parse_ask_query(&mut sub)?;
                Ok(sub.where_clause)
            }
            other => bail!("expected SELECT or ASK to open a subquery, found {other:?}"),
        }
    }

    /// Lower a parsed SubSelect [`Query`] to an [`Algebra`] tree, applying the
    /// SPARQL 1.1 §18.2.4 solution-modifier order:
    /// WHERE → Group → Having → Extend → OrderBy → Project → Distinct → Slice.
    ///
    /// This mirrors the server's `build_select_algebra` exactly so a SubSelect
    /// evaluates identically to the same query run standalone.
    pub(super) fn build_subquery_algebra(&self, query: &Query) -> Result<Algebra> {
        let mut alg = query.where_clause.clone();

        // ASK carries no projection / modifiers into a join; its WHERE pattern
        // is the whole contribution.
        if query.query_type == QueryType::Ask {
            return Ok(alg);
        }

        // Aggregate projections `(AGG(...) AS ?alias)`, in projection order.
        let aggregates: Vec<(Variable, Aggregate)> = query
            .projection_items
            .iter()
            .filter_map(|item| match item {
                ProjectionItem::Aggregate { aggregate, alias } => {
                    Some((alias.clone(), aggregate.clone()))
                }
                _ => None,
            })
            .collect();

        let has_grouping = !aggregates.is_empty() || !query.group_by.is_empty();

        // In an aggregate query every plain projected variable must be a
        // grouping key; projecting a non-grouped, non-aggregated variable is a
        // SPARQL error (fail loud rather than emit a silently-unbound column).
        if has_grouping {
            let grouped: HashSet<&Variable> = query
                .group_by
                .iter()
                .filter_map(|gc| match &gc.expr {
                    Expression::Variable(v) => Some(v),
                    _ => None,
                })
                .chain(query.group_by.iter().filter_map(|gc| gc.alias.as_ref()))
                .collect();
            for item in &query.projection_items {
                if let ProjectionItem::Variable(var) = item {
                    if !grouped.contains(var) {
                        bail!(
                            "subquery SELECT variable ?{} must be a GROUP BY key or wrapped in \
                             an aggregate function",
                            var.name()
                        );
                    }
                }
            }
        }

        if has_grouping {
            alg = Algebra::Group {
                pattern: Box::new(alg),
                variables: query.group_by.clone(),
                aggregates,
            };
        }

        if let Some(condition) = query.having.clone() {
            alg = Algebra::Having {
                pattern: Box::new(alg),
                condition,
            };
        }

        // Projected expressions become Extend nodes, in projection order, so a
        // later `(expr AS ?v)` may reference an alias bound by an earlier one.
        for item in &query.projection_items {
            if let ProjectionItem::Expression { expr, alias } = item {
                alg = Algebra::Extend {
                    pattern: Box::new(alg),
                    variable: alias.clone(),
                    expr: expr.clone(),
                };
            }
        }

        if !query.order_by.is_empty() {
            alg = Algebra::OrderBy {
                pattern: Box::new(alg),
                conditions: query.order_by.clone(),
            };
        }

        // `select_variables` carries the ordered output columns. Empty ==
        // `SELECT *` (project every in-scope variable), so only the explicit
        // projection restricts the columns visible to the outer join.
        if !query.select_variables.is_empty() {
            alg = Algebra::Project {
                pattern: Box::new(alg),
                variables: query.select_variables.clone(),
            };
        }

        if query.distinct {
            alg = Algebra::Distinct {
                pattern: Box::new(alg),
            };
        }

        if query.limit.is_some() || query.offset.is_some() {
            alg = Algebra::Slice {
                pattern: Box::new(alg),
                offset: query.offset,
                limit: query.limit,
            };
        }

        Ok(alg)
    }
}
