//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::uuid_simple;
use crate::{Rule, RuleAtom, Term};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// RIF rule structure
#[derive(Debug, Clone)]
pub struct RifRule {
    /// Optional rule name/ID
    pub id: Option<String>,
    /// Rule body (conditions)
    pub body: RifFormula,
    /// Rule head (conclusion)
    pub head: RifFormula,
    /// Quantified variables
    pub variables: Vec<RifVar>,
    /// Rule metadata
    pub metadata: HashMap<String, String>,
}
/// RIF atomic formula
#[derive(Debug, Clone)]
pub enum RifAtom {
    /// Positional atom: pred(arg1, arg2, ...)
    Positional(RifPositionalAtom),
    /// Named-argument atom: pred(name1->arg1, name2->arg2, ...)
    Named(RifNamedAtom),
    /// Equality atom: term1 = term2
    Equal(RifTerm, RifTerm),
    /// External call: External(pred(args))
    External(RifExternal),
}
/// RIF document structure
#[derive(Debug, Clone)]
pub struct RifDocument {
    /// Document dialect
    pub dialect: RifDialect,
    /// Prefix declarations
    pub prefixes: HashMap<String, String>,
    /// Import directives
    pub imports: Vec<RifImport>,
    /// Rule groups
    pub groups: Vec<RifGroup>,
    /// Base IRI
    pub base: Option<String>,
    /// Document metadata
    pub metadata: HashMap<String, String>,
}
impl RifDocument {
    /// Create a new RIF document
    pub fn new(dialect: RifDialect) -> Self {
        Self {
            dialect,
            prefixes: HashMap::new(),
            imports: Vec::new(),
            groups: Vec::new(),
            base: None,
            metadata: HashMap::new(),
        }
    }
    /// Add a prefix declaration
    pub fn add_prefix(&mut self, prefix: &str, iri: &str) {
        self.prefixes.insert(prefix.to_string(), iri.to_string());
    }
    /// Add a rule group
    pub fn add_group(&mut self, group: RifGroup) {
        self.groups.push(group);
    }
    /// Convert to OxiRS rules
    pub fn to_oxirs_rules(&self) -> Result<Vec<Rule>> {
        let converter = RifConverter::new(self.prefixes.clone());
        let mut rules = Vec::new();
        for group in &self.groups {
            for sentence in &group.sentences {
                if let RifSentence::Rule(rif_rule) = sentence {
                    rules.push(converter.convert_rule(rif_rule)?);
                }
            }
        }
        Ok(rules)
    }
    /// Create from OxiRS rules
    pub fn from_oxirs_rules(rules: &[Rule], dialect: RifDialect) -> Self {
        let converter = RifConverter::new(HashMap::new());
        let mut doc = Self::new(dialect);
        let sentences: Vec<RifSentence> = rules
            .iter()
            .map(|r| RifSentence::Rule(Box::new(converter.rule_to_rif(r))))
            .collect();
        doc.add_group(RifGroup {
            id: None,
            sentences,
            metadata: HashMap::new(),
        });
        doc
    }
    /// Expand prefixed IRIs to full IRIs
    pub fn expand_iri(&self, prefixed: &str) -> String {
        if let Some((prefix, local)) = prefixed.split_once(':') {
            if let Some(base_iri) = self.prefixes.get(prefix) {
                return format!("{}{}", base_iri, local);
            }
        }
        prefixed.to_string()
    }
}
/// RIF frame formula (F-logic style)
#[derive(Debug, Clone)]
pub struct RifFrame {
    /// Object term
    pub object: RifTerm,
    /// Slot-value pairs
    pub slots: Vec<(RifTerm, RifTerm)>,
}
/// RIF variable
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RifVar {
    /// Variable name (without '?')
    pub name: String,
    /// Optional type annotation
    pub var_type: Option<String>,
}
impl RifVar {
    /// Create a new variable
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            var_type: None,
        }
    }
    /// Create a typed variable
    pub fn typed(name: &str, var_type: &str) -> Self {
        Self {
            name: name.to_string(),
            var_type: Some(var_type.to_string()),
        }
    }
}
/// RIF function application
#[derive(Debug, Clone)]
pub struct RifFunc {
    /// Function name
    pub name: Box<RifTerm>,
    /// Arguments
    pub args: Vec<RifTerm>,
}
/// RIF formula (logical expression)
#[derive(Debug, Clone)]
pub enum RifFormula {
    /// Atomic formula
    Atom(RifAtom),
    /// Conjunction (And)
    And(Vec<RifFormula>),
    /// Disjunction (Or)
    Or(Vec<RifFormula>),
    /// Negation as failure (NAF)
    Naf(Box<RifFormula>),
    /// Classical negation (BLD)
    Neg(Box<RifFormula>),
    /// Existential quantification
    Exists(Vec<RifVar>, Box<RifFormula>),
    /// External predicate call
    External(RifExternal),
    /// Equality
    Equal(RifTerm, RifTerm),
    /// Frame formula (F-logic)
    Frame(RifFrame),
    /// Membership (instance-of)
    Member(RifTerm, RifTerm),
    /// Subclass
    Subclass(RifTerm, RifTerm),
}
impl RifFormula {
    /// Create an atomic formula
    pub fn atom(predicate: &str, args: Vec<RifTerm>) -> Self {
        Self::Atom(RifAtom::Positional(RifPositionalAtom {
            predicate: RifTerm::Const(RifConst::Iri(predicate.to_string())),
            args,
        }))
    }
    /// Create a conjunction
    pub fn and(formulas: Vec<RifFormula>) -> Self {
        Self::And(formulas)
    }
    /// Create a NAF formula
    pub fn naf(formula: RifFormula) -> Self {
        Self::Naf(Box::new(formula))
    }
}
/// RIF term
#[derive(Debug, Clone)]
pub enum RifTerm {
    /// Variable
    Var(RifVar),
    /// Constant
    Const(RifConst),
    /// Function application
    Func(Box<RifFunc>),
    /// List
    List(Vec<RifTerm>),
    /// External function
    External(Box<RifExternal>),
}
impl RifTerm {
    /// Create a variable term
    pub fn var(name: &str) -> Self {
        Self::Var(RifVar::new(name))
    }
    /// Create an IRI constant
    pub fn iri(iri: &str) -> Self {
        Self::Const(RifConst::Iri(iri.to_string()))
    }
    /// Create a string literal
    pub fn string(s: &str) -> Self {
        Self::Const(RifConst::Literal(RifLiteral {
            value: s.to_string(),
            datatype: Some("xsd:string".to_string()),
            lang: None,
        }))
    }
    /// Create an integer literal
    pub fn integer(n: i64) -> Self {
        Self::Const(RifConst::Literal(RifLiteral {
            value: n.to_string(),
            datatype: Some("xsd:integer".to_string()),
            lang: None,
        }))
    }
}
/// RIF rule group
#[derive(Debug, Clone)]
pub struct RifGroup {
    /// Optional group ID
    pub id: Option<String>,
    /// Sentences in the group
    pub sentences: Vec<RifSentence>,
    /// Group metadata
    pub metadata: HashMap<String, String>,
}
/// RIF dialect variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RifDialect {
    /// RIF-Core: Basic Horn rules
    Core,
    /// RIF-BLD: Basic Logic Dialect with equality and NAF
    #[default]
    Bld,
    /// RIF-PRD: Production Rule Dialect (future)
    Prd,
}
/// RIF universal quantification
#[derive(Debug, Clone)]
pub struct RifForall {
    /// Quantified variables
    pub variables: Vec<RifVar>,
    /// The quantified formula (usually a rule)
    pub formula: Box<RifSentence>,
}
/// Named-argument atomic formula
#[derive(Debug, Clone)]
pub struct RifNamedAtom {
    /// Predicate
    pub predicate: RifTerm,
    /// Named arguments (name -> value)
    pub args: HashMap<String, RifTerm>,
}
/// RIF sentence (rule or fact)
#[derive(Debug, Clone)]
pub enum RifSentence {
    /// A rule with body and head
    Rule(Box<RifRule>),
    /// A ground fact
    Fact(RifFact),
    /// A group of sentences
    Group(RifGroup),
    /// Forall quantification
    Forall(RifForall),
}
/// RIF import directive
#[derive(Debug, Clone)]
pub struct RifImport {
    /// Location of imported document
    pub location: String,
    /// Optional profile IRI
    pub profile: Option<String>,
}
/// Serialize RIF documents to Compact Syntax
#[derive(Debug)]
pub struct RifSerializer {
    /// Target dialect (used in serialization header)
    #[allow(dead_code)]
    dialect: RifDialect,
    /// Indentation level (reserved for pretty printing)
    #[allow(dead_code)]
    indent: usize,
}
impl RifSerializer {
    /// Create a new serializer
    pub fn new(dialect: RifDialect) -> Self {
        Self { dialect, indent: 0 }
    }
    /// Serialize a RIF document to Compact Syntax
    pub fn serialize(&self, doc: &RifDocument) -> Result<String> {
        let mut output = String::new();
        output.push_str(&format!(
            "(* RIF-{} Document *)\n\n",
            match doc.dialect {
                RifDialect::Core => "Core",
                RifDialect::Bld => "BLD",
                RifDialect::Prd => "PRD",
            }
        ));
        if let Some(base) = &doc.base {
            output.push_str(&format!("Base(<{}>)\n", base));
        }
        for (prefix, iri) in &doc.prefixes {
            output.push_str(&format!("Prefix({} <{}>)\n", prefix, iri));
        }
        if !doc.prefixes.is_empty() {
            output.push('\n');
        }
        for import in &doc.imports {
            if let Some(profile) = &import.profile {
                output.push_str(&format!("Import(<{}> <{}>)\n", import.location, profile));
            } else {
                output.push_str(&format!("Import(<{}>)\n", import.location));
            }
        }
        if !doc.imports.is_empty() {
            output.push('\n');
        }
        for group in &doc.groups {
            output.push_str(&self.serialize_group(group)?);
            output.push('\n');
        }
        Ok(output)
    }
    /// Serialize a group
    fn serialize_group(&self, group: &RifGroup) -> Result<String> {
        let mut output = String::new();
        if let Some(id) = &group.id {
            output.push_str(&format!("Group {} (\n", id));
        } else {
            output.push_str("Group (\n");
        }
        for sentence in &group.sentences {
            output.push_str("  ");
            output.push_str(&self.serialize_sentence(sentence)?);
            output.push('\n');
        }
        output.push_str(")\n");
        Ok(output)
    }
    /// Serialize a sentence
    fn serialize_sentence(&self, sentence: &RifSentence) -> Result<String> {
        match sentence {
            RifSentence::Rule(rule) => self.serialize_rule(rule),
            RifSentence::Fact(fact) => self.serialize_atom(&fact.atom),
            RifSentence::Group(group) => self.serialize_group(group),
            RifSentence::Forall(forall) => self.serialize_forall(forall),
        }
    }
    /// Serialize a Forall
    fn serialize_forall(&self, forall: &RifForall) -> Result<String> {
        let vars: Vec<String> = forall
            .variables
            .iter()
            .map(|v| format!("?{}", v.name))
            .collect();
        let inner = self.serialize_sentence(&forall.formula)?;
        Ok(format!("Forall {} ({})", vars.join(" "), inner))
    }
    /// Serialize a rule
    fn serialize_rule(&self, rule: &RifRule) -> Result<String> {
        let head = self.serialize_formula(&rule.head)?;
        let body = self.serialize_formula(&rule.body)?;
        let rule_str = if body.is_empty() || body == "And()" {
            head
        } else {
            format!("{} :- {}", head, body)
        };
        if let Some(id) = &rule.id {
            Ok(format!("(* {} *) {}", id, rule_str))
        } else {
            Ok(rule_str)
        }
    }
    /// Serialize a formula
    fn serialize_formula(&self, formula: &RifFormula) -> Result<String> {
        match formula {
            RifFormula::Atom(atom) => self.serialize_atom(atom),
            RifFormula::And(formulas) => {
                if formulas.is_empty() {
                    return Ok("And()".to_string());
                }
                let parts: Result<Vec<String>> =
                    formulas.iter().map(|f| self.serialize_formula(f)).collect();
                Ok(format!("And({})", parts?.join(" ")))
            }
            RifFormula::Or(formulas) => {
                let parts: Result<Vec<String>> =
                    formulas.iter().map(|f| self.serialize_formula(f)).collect();
                Ok(format!("Or({})", parts?.join(" ")))
            }
            RifFormula::Naf(inner) => Ok(format!("Naf({})", self.serialize_formula(inner)?)),
            RifFormula::Neg(inner) => Ok(format!("Neg({})", self.serialize_formula(inner)?)),
            RifFormula::Exists(vars, inner) => {
                let var_strs: Vec<String> = vars.iter().map(|v| format!("?{}", v.name)).collect();
                Ok(format!(
                    "Exists {} ({})",
                    var_strs.join(" "),
                    self.serialize_formula(inner)?
                ))
            }
            RifFormula::External(ext) => {
                Ok(format!("External({})", self.serialize_atom(&ext.content)?))
            }
            RifFormula::Equal(left, right) => Ok(format!(
                "{} = {}",
                self.serialize_term(left)?,
                self.serialize_term(right)?
            )),
            RifFormula::Frame(frame) => self.serialize_frame(frame),
            RifFormula::Member(term, class) => Ok(format!(
                "{}#{}",
                self.serialize_term(term)?,
                self.serialize_term(class)?
            )),
            RifFormula::Subclass(sub, sup) => Ok(format!(
                "{}##{}",
                self.serialize_term(sub)?,
                self.serialize_term(sup)?
            )),
        }
    }
    /// Serialize an atom
    fn serialize_atom(&self, atom: &RifAtom) -> Result<String> {
        match atom {
            RifAtom::Positional(pos) => {
                let pred = self.serialize_term(&pos.predicate)?;
                if pos.args.is_empty() {
                    Ok(pred)
                } else {
                    let args: Result<Vec<String>> =
                        pos.args.iter().map(|t| self.serialize_term(t)).collect();
                    Ok(format!("{}({})", pred, args?.join(" ")))
                }
            }
            RifAtom::Named(named) => {
                let pred = self.serialize_term(&named.predicate)?;
                let args: Vec<String> = named
                    .args
                    .iter()
                    .map(|(k, v)| format!("{}->{}", k, self.serialize_term(v).unwrap_or_default()))
                    .collect();
                Ok(format!("{}({})", pred, args.join(" ")))
            }
            RifAtom::Equal(left, right) => Ok(format!(
                "{} = {}",
                self.serialize_term(left)?,
                self.serialize_term(right)?
            )),
            RifAtom::External(ext) => {
                Ok(format!("External({})", self.serialize_atom(&ext.content)?))
            }
        }
    }
    /// Serialize a term
    fn serialize_term(&self, term: &RifTerm) -> Result<String> {
        match term {
            RifTerm::Var(v) => Ok(format!("?{}", v.name)),
            RifTerm::Const(c) => match c {
                RifConst::Iri(iri) => Ok(format!("<{}>", iri)),
                RifConst::Local(name) => Ok(name.clone()),
                RifConst::Literal(lit) => {
                    let mut s = format!("\"{}\"", lit.value);
                    if let Some(dt) = &lit.datatype {
                        s.push_str(&format!("^^{}", dt));
                    } else if let Some(lang) = &lit.lang {
                        s.push_str(&format!("@{}", lang));
                    }
                    Ok(s)
                }
            },
            RifTerm::Func(f) => {
                let name = self.serialize_term(&f.name)?;
                let args: Result<Vec<String>> =
                    f.args.iter().map(|t| self.serialize_term(t)).collect();
                Ok(format!("{}({})", name, args?.join(" ")))
            }
            RifTerm::List(items) => {
                let parts: Result<Vec<String>> =
                    items.iter().map(|t| self.serialize_term(t)).collect();
                Ok(format!("List({})", parts?.join(" ")))
            }
            RifTerm::External(ext) => {
                Ok(format!("External({})", self.serialize_atom(&ext.content)?))
            }
        }
    }
    /// Serialize a frame
    fn serialize_frame(&self, frame: &RifFrame) -> Result<String> {
        let obj = self.serialize_term(&frame.object)?;
        let slots: Vec<String> = frame
            .slots
            .iter()
            .map(|(k, v)| {
                format!(
                    "{}->{}",
                    self.serialize_term(k).unwrap_or_default(),
                    self.serialize_term(v).unwrap_or_default()
                )
            })
            .collect();
        Ok(format!("{}[{}]", obj, slots.join(" ")))
    }
}
/// RIF constant
#[derive(Debug, Clone)]
pub enum RifConst {
    /// IRI
    Iri(String),
    /// Local name
    Local(String),
    /// Typed literal
    Literal(RifLiteral),
}
/// RIF fact (ground atomic formula)
#[derive(Debug, Clone)]
pub struct RifFact {
    /// The atomic formula
    pub atom: RifAtom,
    /// Optional fact ID
    pub id: Option<String>,
}
/// RIF typed literal
#[derive(Debug, Clone)]
pub struct RifLiteral {
    /// Lexical value
    pub value: String,
    /// Datatype IRI
    pub datatype: Option<String>,
    /// Language tag
    pub lang: Option<String>,
}
/// RIF external predicate/function call
#[derive(Debug, Clone)]
pub struct RifExternal {
    /// External function/predicate
    pub content: Box<RifAtom>,
}
/// Convert between RIF and OxiRS rule formats
#[derive(Debug)]
pub struct RifConverter {
    /// Prefix mappings (reserved for future prefix expansion)
    #[allow(dead_code)]
    prefixes: HashMap<String, String>,
}
impl RifConverter {
    /// Create a new converter with prefix mappings
    pub fn new(prefixes: HashMap<String, String>) -> Self {
        Self { prefixes }
    }
    /// Convert a RIF rule to OxiRS Rule
    pub fn convert_rule(&self, rif_rule: &RifRule) -> Result<Rule> {
        let name = rif_rule
            .id
            .clone()
            .unwrap_or_else(|| format!("rule_{}", uuid_simple()));
        let body = self.convert_formula_to_atoms(&rif_rule.body)?;
        let head = self.convert_formula_to_atoms(&rif_rule.head)?;
        Ok(Rule { name, body, head })
    }
    /// Convert RIF formula to list of RuleAtoms
    fn convert_formula_to_atoms(&self, formula: &RifFormula) -> Result<Vec<RuleAtom>> {
        match formula {
            RifFormula::Atom(atom) => Ok(vec![Self::convert_atom(atom)?]),
            RifFormula::And(formulas) => {
                let mut atoms = Vec::new();
                for f in formulas {
                    atoms.extend(self.convert_formula_to_atoms(f)?);
                }
                Ok(atoms)
            }
            RifFormula::Or(_) => Err(anyhow!("Disjunction in rule body not supported")),
            RifFormula::Naf(inner) => {
                let inner_atoms = self.convert_formula_to_atoms(inner)?;
                if inner_atoms.len() == 1 {
                    if let RuleAtom::Triple {
                        subject,
                        predicate,
                        object,
                    } = &inner_atoms[0]
                    {
                        Ok(vec![RuleAtom::Builtin {
                            name: "not".to_string(),
                            args: vec![subject.clone(), predicate.clone(), object.clone()],
                        }])
                    } else {
                        Err(anyhow!("NAF only supported for triple patterns"))
                    }
                } else {
                    Err(anyhow!("NAF of complex formula not supported"))
                }
            }
            RifFormula::Neg(_) => Err(anyhow!("Classical negation not supported")),
            RifFormula::Exists(_, inner) => self.convert_formula_to_atoms(inner),
            RifFormula::External(ext) => Ok(vec![Self::convert_atom(&ext.content)?]),
            RifFormula::Equal(left, right) => Ok(vec![RuleAtom::Builtin {
                name: "equal".to_string(),
                args: vec![Self::convert_term(left), Self::convert_term(right)],
            }]),
            RifFormula::Frame(frame) => self.convert_frame(frame),
            RifFormula::Member(term, class) => Ok(vec![RuleAtom::Triple {
                subject: Self::convert_term(term),
                predicate: Term::Constant("rdf:type".to_string()),
                object: Self::convert_term(class),
            }]),
            RifFormula::Subclass(sub, sup) => Ok(vec![RuleAtom::Triple {
                subject: Self::convert_term(sub),
                predicate: Term::Constant("rdfs:subClassOf".to_string()),
                object: Self::convert_term(sup),
            }]),
        }
    }
    /// Convert RIF atom to RuleAtom
    fn convert_atom(atom: &RifAtom) -> Result<RuleAtom> {
        match atom {
            RifAtom::Positional(pos) => {
                let predicate = Self::convert_term(&pos.predicate);
                if pos.args.len() == 2 {
                    Ok(RuleAtom::Triple {
                        subject: Self::convert_term(&pos.args[0]),
                        predicate,
                        object: Self::convert_term(&pos.args[1]),
                    })
                } else {
                    let pred_name = match &predicate {
                        Term::Constant(c) => c.clone(),
                        Term::Variable(v) => v.clone(),
                        _ => "unknown".to_string(),
                    };
                    Ok(RuleAtom::Builtin {
                        name: pred_name,
                        args: pos.args.iter().map(Self::convert_term).collect(),
                    })
                }
            }
            RifAtom::Named(named) => {
                let pred_name = match &Self::convert_term(&named.predicate) {
                    Term::Constant(c) => c.clone(),
                    _ => "unnamed".to_string(),
                };
                let args: Vec<Term> = named.args.values().map(Self::convert_term).collect();
                Ok(RuleAtom::Builtin {
                    name: pred_name,
                    args,
                })
            }
            RifAtom::Equal(left, right) => Ok(RuleAtom::Builtin {
                name: "equal".to_string(),
                args: vec![Self::convert_term(left), Self::convert_term(right)],
            }),
            RifAtom::External(ext) => Self::convert_atom(&ext.content),
        }
    }
    /// Convert RIF term to Term
    fn convert_term(term: &RifTerm) -> Term {
        match term {
            RifTerm::Var(v) => Term::Variable(v.name.clone()),
            RifTerm::Const(c) => match c {
                RifConst::Iri(iri) => Term::Constant(iri.clone()),
                RifConst::Local(name) => Term::Constant(name.clone()),
                RifConst::Literal(lit) => Term::Literal(lit.value.clone()),
            },
            RifTerm::Func(f) => {
                let func_name = match &*f.name {
                    RifTerm::Const(RifConst::Local(n)) => n.clone(),
                    RifTerm::Const(RifConst::Iri(iri)) => iri.clone(),
                    _ => "func".to_string(),
                };
                Term::Function {
                    name: func_name,
                    args: f.args.iter().map(Self::convert_term).collect(),
                }
            }
            RifTerm::List(items) => Term::Function {
                name: "list".to_string(),
                args: items.iter().map(Self::convert_term).collect(),
            },
            RifTerm::External(ext) => {
                if let RifAtom::Positional(pos) = &*ext.content {
                    let name = match &pos.predicate {
                        RifTerm::Const(RifConst::Local(n)) => n.clone(),
                        _ => "external".to_string(),
                    };
                    Term::Function {
                        name,
                        args: pos.args.iter().map(Self::convert_term).collect(),
                    }
                } else {
                    Term::Constant("external".to_string())
                }
            }
        }
    }
    /// Convert frame to triple atoms
    fn convert_frame(&self, frame: &RifFrame) -> Result<Vec<RuleAtom>> {
        let subject = Self::convert_term(&frame.object);
        let mut atoms = Vec::new();
        for (slot, value) in &frame.slots {
            atoms.push(RuleAtom::Triple {
                subject: subject.clone(),
                predicate: Self::convert_term(slot),
                object: Self::convert_term(value),
            });
        }
        Ok(atoms)
    }
    /// Convert OxiRS Rule to RIF rule
    pub fn rule_to_rif(&self, rule: &Rule) -> RifRule {
        let head = self.atoms_to_formula(&rule.head);
        let body = self.atoms_to_formula(&rule.body);
        RifRule {
            id: Some(rule.name.clone()),
            head,
            body,
            variables: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    /// Convert list of RuleAtoms to RIF formula
    fn atoms_to_formula(&self, atoms: &[RuleAtom]) -> RifFormula {
        if atoms.is_empty() {
            return RifFormula::And(vec![]);
        }
        let formulas: Vec<RifFormula> = atoms.iter().map(|a| self.atom_to_rif(a)).collect();
        if formulas.len() == 1 {
            formulas
                .into_iter()
                .next()
                .expect("formulas validated to have exactly one element")
        } else {
            RifFormula::And(formulas)
        }
    }
    /// Convert RuleAtom to RIF formula
    fn atom_to_rif(&self, atom: &RuleAtom) -> RifFormula {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => RifFormula::Atom(RifAtom::Positional(RifPositionalAtom {
                predicate: Self::term_to_rif(predicate),
                args: vec![Self::term_to_rif(subject), Self::term_to_rif(object)],
            })),
            RuleAtom::Builtin { name, args } => {
                RifFormula::Atom(RifAtom::Positional(RifPositionalAtom {
                    predicate: RifTerm::Const(RifConst::Local(name.clone())),
                    args: args.iter().map(Self::term_to_rif).collect(),
                }))
            }
            RuleAtom::NotEqual { left, right } => {
                RifFormula::Atom(RifAtom::Positional(RifPositionalAtom {
                    predicate: RifTerm::Const(RifConst::Local("notEqual".to_string())),
                    args: vec![Self::term_to_rif(left), Self::term_to_rif(right)],
                }))
            }
            RuleAtom::GreaterThan { left, right } => {
                RifFormula::Atom(RifAtom::Positional(RifPositionalAtom {
                    predicate: RifTerm::Const(RifConst::Local("greaterThan".to_string())),
                    args: vec![Self::term_to_rif(left), Self::term_to_rif(right)],
                }))
            }
            RuleAtom::LessThan { left, right } => {
                RifFormula::Atom(RifAtom::Positional(RifPositionalAtom {
                    predicate: RifTerm::Const(RifConst::Local("lessThan".to_string())),
                    args: vec![Self::term_to_rif(left), Self::term_to_rif(right)],
                }))
            }
        }
    }
    /// Convert Term to RIF term
    fn term_to_rif(term: &Term) -> RifTerm {
        match term {
            Term::Variable(v) => RifTerm::Var(RifVar::new(v)),
            Term::Constant(c) => {
                if c.starts_with("http://") || c.starts_with("https://") {
                    RifTerm::Const(RifConst::Iri(c.clone()))
                } else {
                    RifTerm::Const(RifConst::Local(c.clone()))
                }
            }
            Term::Literal(l) => RifTerm::Const(RifConst::Literal(RifLiteral {
                value: l.clone(),
                datatype: None,
                lang: None,
            })),
            Term::Function { name, args } => RifTerm::Func(Box::new(RifFunc {
                name: Box::new(RifTerm::Const(RifConst::Local(name.clone()))),
                args: args.iter().map(Self::term_to_rif).collect(),
            })),
        }
    }
}
/// Positional atomic formula
#[derive(Debug, Clone)]
pub struct RifPositionalAtom {
    /// Predicate (IRI or local name)
    pub predicate: RifTerm,
    /// Positional arguments
    pub args: Vec<RifTerm>,
}
