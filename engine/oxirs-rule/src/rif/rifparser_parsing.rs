//! # RifParser - parsing Methods
//!
//! This module contains method implementations for `RifParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use std::collections::HashMap;

use super::rifparser_type::RifParser;
use super::types::{
    RifAtom, RifConst, RifDialect, RifDocument, RifExternal, RifForall, RifFormula, RifGroup,
    RifLiteral, RifPositionalAtom, RifRule, RifSentence, RifTerm, RifVar,
};

impl RifParser {
    /// Create a new parser
    pub fn new(dialect: RifDialect) -> Self {
        Self {
            dialect,
            prefixes: HashMap::new(),
            pos: 0,
            input: String::new(),
        }
    }
    /// Parse RIF Compact Syntax
    pub fn parse(&mut self, input: &str) -> Result<RifDocument> {
        self.input = input.to_string();
        self.pos = 0;
        self.prefixes.clear();
        let mut doc = RifDocument::new(self.dialect);
        self.skip_ws();
        while self.pos < self.input.len() {
            self.skip_ws();
            if self.pos >= self.input.len() {
                break;
            }
            if self.try_consume("Document") {
                self.parse_document_wrapper(&mut doc)?;
            } else if self.try_consume("Base") {
                doc.base = Some(self.parse_iri()?);
            } else if self.try_consume("Prefix") {
                let (prefix, iri) = self.parse_prefix_decl()?;
                doc.add_prefix(&prefix, &iri);
                self.prefixes.insert(prefix, iri);
            } else if self.try_consume("Import") {
                doc.imports.push(self.parse_import()?);
            } else if self.try_consume("Group") {
                doc.add_group(self.parse_group()?);
            } else if self.try_consume("Forall") {
                let forall = self.parse_forall()?;
                let group = RifGroup {
                    id: None,
                    sentences: vec![RifSentence::Forall(forall)],
                    metadata: HashMap::new(),
                };
                doc.add_group(group);
            } else {
                if let Ok(rule) = self.parse_rule() {
                    let group = RifGroup {
                        id: None,
                        sentences: vec![RifSentence::Rule(Box::new(rule))],
                        metadata: HashMap::new(),
                    };
                    doc.add_group(group);
                } else {
                    self.pos += 1;
                }
            }
        }
        doc.prefixes = self.prefixes.clone();
        Ok(doc)
    }
    /// Parse rule group
    pub(super) fn parse_group(&mut self) -> Result<RifGroup> {
        self.skip_ws();
        let id = if self.check("_") || self.check_alpha() {
            Some(self.parse_name()?)
        } else {
            None
        };
        self.skip_ws();
        self.expect("(")?;
        self.skip_ws();
        let mut sentences = Vec::new();
        while !self.check(")") {
            self.skip_ws();
            if self.check(")") {
                break;
            }
            if self.try_consume("Forall") {
                sentences.push(RifSentence::Forall(self.parse_forall()?));
            } else if self.try_consume("Group") {
                sentences.push(RifSentence::Group(self.parse_group()?));
            } else {
                if let Ok(rule) = self.parse_rule() {
                    sentences.push(RifSentence::Rule(Box::new(rule)));
                }
            }
            self.skip_ws();
        }
        self.expect(")")?;
        Ok(RifGroup {
            id,
            sentences,
            metadata: HashMap::new(),
        })
    }
    /// Parse Forall quantification
    pub(super) fn parse_forall(&mut self) -> Result<RifForall> {
        self.skip_ws();
        let mut variables = Vec::new();
        while !self.check("(") && !self.check_eof() {
            self.skip_ws();
            if self.try_consume("?") {
                let name = self.parse_name()?;
                variables.push(RifVar::new(&name));
            } else if self.check("(") {
                break;
            } else {
                self.pos += 1;
            }
            self.skip_ws();
        }
        self.expect("(")?;
        self.skip_ws();
        let rule = self.parse_rule()?;
        let formula = Box::new(RifSentence::Rule(Box::new(rule)));
        self.skip_ws();
        self.expect(")")?;
        Ok(RifForall { variables, formula })
    }
    /// Parse a rule: head :- body or head <- body
    pub(super) fn parse_rule(&mut self) -> Result<RifRule> {
        self.skip_ws();
        let head = self.parse_formula()?;
        self.skip_ws();
        let has_body = self.try_consume(":-") || self.try_consume("<-") || self.try_consume("If");
        let body = if has_body {
            self.skip_ws();
            self.parse_formula()?
        } else {
            RifFormula::And(vec![])
        };
        Ok(RifRule {
            id: None,
            head,
            body,
            variables: Vec::new(),
            metadata: HashMap::new(),
        })
    }
    /// Parse a formula
    pub(super) fn parse_formula(&mut self) -> Result<RifFormula> {
        self.skip_ws();
        if self.try_consume("And") {
            return self.parse_and_formula();
        }
        if self.try_consume("Or") {
            return self.parse_or_formula();
        }
        if self.try_consume("Naf") || self.try_consume("Not") || self.try_consume("\\+") {
            self.skip_ws();
            self.expect("(")?;
            let inner = self.parse_formula()?;
            self.expect(")")?;
            return Ok(RifFormula::Naf(Box::new(inner)));
        }
        if self.try_consume("Neg") {
            self.skip_ws();
            self.expect("(")?;
            let inner = self.parse_formula()?;
            self.expect(")")?;
            return Ok(RifFormula::Neg(Box::new(inner)));
        }
        if self.try_consume("Exists") {
            return self.parse_exists_formula();
        }
        if self.try_consume("External") {
            return self.parse_external_formula();
        }
        self.parse_atom_formula()
    }
    /// Parse And formula
    pub(super) fn parse_and_formula(&mut self) -> Result<RifFormula> {
        self.skip_ws();
        self.expect("(")?;
        self.skip_ws();
        let mut formulas = Vec::new();
        while !self.check(")") {
            formulas.push(self.parse_formula()?);
            self.skip_ws();
        }
        self.expect(")")?;
        Ok(RifFormula::And(formulas))
    }
    /// Parse Or formula
    pub(super) fn parse_or_formula(&mut self) -> Result<RifFormula> {
        self.skip_ws();
        self.expect("(")?;
        self.skip_ws();
        let mut formulas = Vec::new();
        while !self.check(")") {
            formulas.push(self.parse_formula()?);
            self.skip_ws();
        }
        self.expect(")")?;
        Ok(RifFormula::Or(formulas))
    }
    /// Parse Exists formula
    pub(super) fn parse_exists_formula(&mut self) -> Result<RifFormula> {
        self.skip_ws();
        let mut variables = Vec::new();
        while self.try_consume("?") {
            let name = self.parse_name()?;
            variables.push(RifVar::new(&name));
            self.skip_ws();
        }
        self.expect("(")?;
        let inner = self.parse_formula()?;
        self.expect(")")?;
        Ok(RifFormula::Exists(variables, Box::new(inner)))
    }
    /// Parse External formula
    pub(super) fn parse_external_formula(&mut self) -> Result<RifFormula> {
        self.skip_ws();
        self.expect("(")?;
        let atom = self.parse_atom()?;
        self.expect(")")?;
        Ok(RifFormula::External(RifExternal {
            content: Box::new(atom),
        }))
    }
    /// Parse an atom
    pub(super) fn parse_atom(&mut self) -> Result<RifAtom> {
        self.skip_ws();
        let predicate = self.parse_term()?;
        self.skip_ws();
        if !self.try_consume("(") {
            return Ok(RifAtom::Positional(RifPositionalAtom {
                predicate,
                args: vec![],
            }));
        }
        self.skip_ws();
        let mut args = Vec::new();
        while !self.check(")") {
            args.push(self.parse_term()?);
            self.skip_ws();
            self.try_consume(",");
            self.skip_ws();
        }
        self.expect(")")?;
        Ok(RifAtom::Positional(RifPositionalAtom { predicate, args }))
    }
    /// Parse a term
    pub(super) fn parse_term(&mut self) -> Result<RifTerm> {
        self.skip_ws();
        if self.try_consume("?") {
            let name = self.parse_name()?;
            return Ok(RifTerm::Var(RifVar::new(&name)));
        }
        if self.check("<") {
            let iri = self.parse_iri()?;
            return Ok(RifTerm::Const(RifConst::Iri(iri)));
        }
        if self.check("\"") {
            let lit = self.parse_literal()?;
            return Ok(RifTerm::Const(RifConst::Literal(lit)));
        }
        if self.check_digit() || self.check("-") || self.check("+") {
            let num = self.parse_number()?;
            return Ok(RifTerm::Const(RifConst::Literal(RifLiteral {
                value: num,
                datatype: Some("xsd:integer".to_string()),
                lang: None,
            })));
        }
        if self.try_consume("List") {
            return self.parse_list();
        }
        let name = self.parse_prefixed_name()?;
        Ok(RifTerm::Const(RifConst::Local(name)))
    }
    /// Parse a string literal
    pub(super) fn parse_literal(&mut self) -> Result<RifLiteral> {
        self.expect("\"")?;
        let mut value = String::new();
        while self.pos < self.input.len() && !self.check("\"") {
            if self.check("\\") {
                self.pos += 1;
                if self.pos < self.input.len() {
                    let c = self
                        .input
                        .chars()
                        .nth(self.pos)
                        .expect("position is within input bounds");
                    match c {
                        'n' => value.push('\n'),
                        't' => value.push('\t'),
                        'r' => value.push('\r'),
                        '\\' => value.push('\\'),
                        '"' => value.push('"'),
                        _ => value.push(c),
                    }
                    self.pos += 1;
                }
            } else {
                value.push(
                    self.input
                        .chars()
                        .nth(self.pos)
                        .expect("position is within input bounds"),
                );
                self.pos += 1;
            }
        }
        self.expect("\"")?;
        let (datatype, lang) = if self.try_consume("^^") {
            let dt = if self.check("<") {
                self.parse_iri()?
            } else {
                self.parse_prefixed_name()?
            };
            (Some(dt), None)
        } else if self.try_consume("@") {
            let lang = self.parse_name()?;
            (None, Some(lang))
        } else {
            (None, None)
        };
        Ok(RifLiteral {
            value,
            datatype,
            lang,
        })
    }
}
