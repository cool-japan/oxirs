//! # QueryParser - blank-node property lists `[ … ]` and RDF collections `( … )`
//!
//! SPARQL 1.1 Turtle-style syntactic sugar (§4.2.2 / §4.2.3, `TriplesNode`).
//! Both expand, at parse time, into ordinary triple patterns anchored on a
//! freshly-minted anonymous node:
//!
//! * `[ :p :o ; :q :r ]` → a fresh node `_:b` plus `_:b :p :o` and `_:b :q :r`.
//!   The bare `[]` is just a fresh node with no anchored triples.
//! * `( :a :b :c )` → an `rdf:first`/`rdf:rest` chain terminated by `rdf:nil`;
//!   the empty `()` is `rdf:nil` itself.
//!
//! Both forms nest (a collection item or a `[ ]` object may itself be a
//! `TriplesNode`) and appear in subject AND object position, in both WHERE
//! graph patterns and CONSTRUCT templates. The anonymous node is lowered
//! context-sensitively — see [`QueryParser::fresh_blank_node_term`].

use anyhow::{bail, Result};
use oxirs_core::model::NamedNode;

use crate::algebra::{Term, TriplePattern, Variable};

use super::types::Token;

use super::queryparser_type::QueryParser;

pub(super) const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
pub(super) const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
pub(super) const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";

impl QueryParser {
    /// Mint a fresh anonymous node for a `[ ]` / `( )` expansion, unique within
    /// the query via [`QueryParser::blank_node_counter`].
    ///
    /// The lowering is context-sensitive (see the field docs on
    /// `in_construct_template`):
    ///
    /// * CONSTRUCT template → `Term::BlankNode`, so `instantiate_construct`
    ///   mints a genuinely fresh blank node per solution row.
    /// * WHERE graph pattern → a non-distinguished `Term::Variable`. The store
    ///   matches a query blank node as an existential variable; a `Term::BlankNode`
    ///   there would instead be looked up as a specific stored blank node and
    ///   match nothing. The label is unlikely to collide with a user variable and
    ///   is created unchecked (its exact spelling only needs to be unique).
    pub(super) fn fresh_blank_node_term(&mut self) -> Term {
        let n = self.blank_node_counter;
        self.blank_node_counter += 1;
        if self.in_construct_template {
            Term::BlankNode(format!("b{n}"))
        } else {
            Term::Variable(Variable::new_unchecked(format!("_anon_{n}")))
        }
    }

    fn iri_term(iri: &str) -> Term {
        Term::Iri(NamedNode::new_unchecked(iri))
    }

    /// Parse a `GraphNode`: a `TriplesNode` (`[ … ]` / `( … )`) or a plain term.
    /// Any triples produced by an expansion are appended to `out`.
    pub(super) fn parse_graph_node(&mut self, out: &mut Vec<TriplePattern>) -> Result<Term> {
        self.skip_whitespace_and_newlines();
        if matches!(
            self.peek(),
            Some(Token::LeftBracket) | Some(Token::LeftParen)
        ) {
            self.parse_triples_node(out)
        } else {
            self.parse_term()
        }
    }

    /// Parse a `TriplesNode` — a blank-node property list `[ … ]` or an RDF
    /// collection `( … )` — returning the anonymous node that anchors it and
    /// appending its expansion triples to `out`.
    pub(super) fn parse_triples_node(&mut self, out: &mut Vec<TriplePattern>) -> Result<Term> {
        if self.match_token(&Token::LeftBracket) {
            // BlankNodePropertyList `[ PropertyListNotEmpty ]`, or the empty
            // ANON `[]`. Either way the node itself is fresh.
            let node = self.fresh_blank_node_term();
            self.skip_whitespace_and_newlines();
            if !matches!(self.peek(), Some(Token::RightBracket)) {
                self.parse_predicate_object_list(&node, out)?;
                self.skip_whitespace_and_newlines();
            }
            self.expect_token(Token::RightBracket)?;
            Ok(node)
        } else if self.match_token(&Token::LeftParen) {
            self.parse_collection_body(out)
        } else {
            bail!("expected '[' or '(' to open a blank-node property list or collection")
        }
    }

    /// Parse the body of an RDF collection after the opening `(` has been
    /// consumed, expanding `( g1 g2 … gn )` into the `rdf:first`/`rdf:rest`
    /// chain and returning its head node. The empty collection `()` is `rdf:nil`.
    fn parse_collection_body(&mut self, out: &mut Vec<TriplePattern>) -> Result<Term> {
        let rdf_first = Self::iri_term(RDF_FIRST);
        let rdf_rest = Self::iri_term(RDF_REST);
        let rdf_nil = Self::iri_term(RDF_NIL);

        self.skip_whitespace_and_newlines();
        if self.match_token(&Token::RightParen) {
            return Ok(rdf_nil);
        }

        let head = self.fresh_blank_node_term();
        let mut current = head.clone();
        loop {
            // `GraphNode+`: each item may itself be a nested `[ ]` / `( )`.
            let item = self.parse_graph_node(out)?;
            out.push(TriplePattern::new(current.clone(), rdf_first.clone(), item));
            self.skip_whitespace_and_newlines();
            if matches!(self.peek(), Some(Token::RightParen)) {
                out.push(TriplePattern::new(
                    current.clone(),
                    rdf_rest.clone(),
                    rdf_nil.clone(),
                ));
                break;
            }
            let next = self.fresh_blank_node_term();
            out.push(TriplePattern::new(
                current.clone(),
                rdf_rest.clone(),
                next.clone(),
            ));
            current = next;
        }
        self.expect_token(Token::RightParen)?;
        Ok(head)
    }
}
