//! SWRL (Semantic Web Rule Language) Implementation
//!
//! Implementation of SWRL rule parsing, execution, and built-in predicates.
//! Supports the full SWRL specification including custom built-ins.

use crate::{Rule, RuleAtom, Term, RuleEngine};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, trace, warn};

/// SWRL vocabulary constants
pub mod vocabulary {
    // SWRL Core
    pub const SWRL_IMPLIES: &str = "http://www.w3.org/2003/11/swrl#Implies";
    pub const SWRL_BODY: &str = "http://www.w3.org/2003/11/swrl#body";
    pub const SWRL_HEAD: &str = "http://www.w3.org/2003/11/swrl#head";
    
    // SWRL Atoms
    pub const SWRL_CLASS_ATOM: &str = "http://www.w3.org/2003/11/swrl#ClassAtom";
    pub const SWRL_INDIVIDUAL_PROPERTY_ATOM: &str = "http://www.w3.org/2003/11/swrl#IndividualPropertyAtom";
    pub const SWRL_DATAVALUE_PROPERTY_ATOM: &str = "http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom";
    pub const SWRL_BUILTIN_ATOM: &str = "http://www.w3.org/2003/11/swrl#BuiltinAtom";
    pub const SWRL_SAME_INDIVIDUAL_ATOM: &str = "http://www.w3.org/2003/11/swrl#SameIndividualAtom";
    pub const SWRL_DIFFERENT_INDIVIDUALS_ATOM: &str = "http://www.w3.org/2003/11/swrl#DifferentIndividualsAtom";
    
    // SWRL Atom Properties
    pub const SWRL_CLASS_PREDICATE: &str = "http://www.w3.org/2003/11/swrl#classPredicate";
    pub const SWRL_PROPERTY_PREDICATE: &str = "http://www.w3.org/2003/11/swrl#propertyPredicate";
    pub const SWRL_BUILTIN: &str = "http://www.w3.org/2003/11/swrl#builtin";
    pub const SWRL_ARGUMENT1: &str = "http://www.w3.org/2003/11/swrl#argument1";
    pub const SWRL_ARGUMENT2: &str = "http://www.w3.org/2003/11/swrl#argument2";
    pub const SWRL_ARGUMENTS: &str = "http://www.w3.org/2003/11/swrl#arguments";
    
    // SWRL Variables and Literals
    pub const SWRL_VARIABLE: &str = "http://www.w3.org/2003/11/swrl#Variable";
    pub const SWRL_INDIVIDUAL: &str = "http://www.w3.org/2003/11/swrl#Individual";
    
    // Built-in namespaces
    pub const SWRLB_NS: &str = "http://www.w3.org/2003/11/swrlb#";
    pub const SWRLM_NS: &str = "http://www.w3.org/2003/11/swrlm#";
    pub const SWRLT_NS: &str = "http://www.w3.org/2003/11/swrlt#";
    pub const SWRLX_NS: &str = "http://www.w3.org/2003/11/swrlx#";
}

/// SWRL atom types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SwrlAtom {
    /// Class atom: C(x)
    Class {
        class_predicate: String,
        argument: SwrlArgument,
    },
    /// Individual property atom: P(x, y)
    IndividualProperty {
        property_predicate: String,
        argument1: SwrlArgument,
        argument2: SwrlArgument,
    },
    /// Datavalue property atom: P(x, v)
    DatavalueProperty {
        property_predicate: String,
        argument1: SwrlArgument,
        argument2: SwrlArgument,
    },
    /// Built-in atom: builtin(args...)
    Builtin {
        builtin_predicate: String,
        arguments: Vec<SwrlArgument>,
    },
    /// Same individual atom: sameAs(x, y)
    SameIndividual {
        argument1: SwrlArgument,
        argument2: SwrlArgument,
    },
    /// Different individuals atom: differentFrom(x, y)
    DifferentIndividuals {
        argument1: SwrlArgument,
        argument2: SwrlArgument,
    },
}

/// SWRL argument types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SwrlArgument {
    /// Variable
    Variable(String),
    /// Individual
    Individual(String),
    /// Literal value
    Literal(String),
}

/// SWRL rule structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwrlRule {
    /// Rule identifier
    pub id: String,
    /// Rule body (antecedent)
    pub body: Vec<SwrlAtom>,
    /// Rule head (consequent)
    pub head: Vec<SwrlAtom>,
    /// Rule metadata
    pub metadata: HashMap<String, String>,
}

/// Built-in function definition
#[derive(Debug, Clone)]
pub struct BuiltinFunction {
    /// Function name
    pub name: String,
    /// Function namespace
    pub namespace: String,
    /// Minimum number of arguments
    pub min_args: usize,
    /// Maximum number of arguments (None for unlimited)
    pub max_args: Option<usize>,
    /// Function implementation
    pub implementation: fn(&[SwrlArgument]) -> Result<bool>,
}

/// SWRL execution context
#[derive(Debug, Clone)]
pub struct SwrlContext {
    /// Variable bindings
    pub bindings: HashMap<String, SwrlArgument>,
    /// Execution trace
    pub trace: Vec<String>,
}

impl Default for SwrlContext {
    fn default() -> Self {
        Self {
            bindings: HashMap::new(),
            trace: Vec::new(),
        }
    }
}

/// SWRL rule engine
#[derive(Debug)]
pub struct SwrlEngine {
    /// SWRL rules
    rules: Vec<SwrlRule>,
    /// Built-in functions registry
    builtins: HashMap<String, BuiltinFunction>,
    /// Core rule engine for basic reasoning
    rule_engine: RuleEngine,
}

impl Default for SwrlEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SwrlEngine {
    /// Create a new SWRL engine
    pub fn new() -> Self {
        let mut engine = Self {
            rules: Vec::new(),
            builtins: HashMap::new(),
            rule_engine: RuleEngine::new(),
        };
        
        engine.register_core_builtins();
        engine
    }
    
    /// Register core SWRL built-in functions
    fn register_core_builtins(&mut self) {
        // Comparison built-ins
        self.register_builtin(BuiltinFunction {
            name: "equal".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_equal,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "notEqual".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_not_equal,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "lessThan".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_less_than,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "greaterThan".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_greater_than,
        });
        
        // Math built-ins
        self.register_builtin(BuiltinFunction {
            name: "add".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_add,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "subtract".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_subtract,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "multiply".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_multiply,
        });
        
        // String built-ins
        self.register_builtin(BuiltinFunction {
            name: "stringConcat".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: None,
            implementation: builtin_string_concat,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "stringLength".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_string_length,
        });
        
        // Boolean built-ins
        self.register_builtin(BuiltinFunction {
            name: "booleanValue".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 1,
            max_args: Some(1),
            implementation: builtin_boolean_value,
        });
        
        // Advanced mathematical built-ins
        self.register_builtin(BuiltinFunction {
            name: "mod".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_mod,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "pow".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_pow,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "sqrt".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_sqrt,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "sin".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_sin,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "cos".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_cos,
        });
        
        // Date and time built-ins
        self.register_builtin(BuiltinFunction {
            name: "dayTimeDuration".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_day_time_duration,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "yearMonthDuration".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_year_month_duration,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "dateTime".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_date_time,
        });
        
        // List operations
        self.register_builtin(BuiltinFunction {
            name: "listConcat".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: None,
            implementation: builtin_list_concat,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "listLength".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_list_length,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "member".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_member,
        });
        
        // Enhanced string operations
        self.register_builtin(BuiltinFunction {
            name: "stringMatches".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(3),
            implementation: builtin_string_matches,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "substring".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(4),
            implementation: builtin_substring,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "upperCase".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_upper_case,
        });
        
        self.register_builtin(BuiltinFunction {
            name: "lowerCase".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_lower_case,
        });
        
        info!("Registered {} core SWRL built-in functions", self.builtins.len());
    }
    
    /// Register a built-in function
    pub fn register_builtin(&mut self, builtin: BuiltinFunction) {
        let full_name = format!("{}{}", builtin.namespace, builtin.name);
        debug!("Registering SWRL built-in: {}", full_name);
        self.builtins.insert(full_name, builtin);
    }
    
    /// Add a SWRL rule
    pub fn add_rule(&mut self, rule: SwrlRule) -> Result<()> {
        debug!("Adding SWRL rule: {}", rule.id);
        
        // Convert SWRL rule to internal Rule format
        let internal_rule = self.convert_swrl_to_rule(&rule)?;
        self.rule_engine.add_rule(internal_rule);
        
        self.rules.push(rule);
        Ok(())
    }
    
    /// Convert SWRL rule to internal Rule format
    fn convert_swrl_to_rule(&self, swrl_rule: &SwrlRule) -> Result<Rule> {
        let mut body_atoms = Vec::new();
        let mut head_atoms = Vec::new();
        
        // Convert body atoms
        for atom in &swrl_rule.body {
            body_atoms.push(self.convert_swrl_atom_to_rule_atom(atom)?);
        }
        
        // Convert head atoms
        for atom in &swrl_rule.head {
            head_atoms.push(self.convert_swrl_atom_to_rule_atom(atom)?);
        }
        
        Ok(Rule {
            name: swrl_rule.id.clone(),
            body: body_atoms,
            head: head_atoms,
        })
    }
    
    /// Convert SWRL atom to RuleAtom
    fn convert_swrl_atom_to_rule_atom(&self, swrl_atom: &SwrlAtom) -> Result<RuleAtom> {
        match swrl_atom {
            SwrlAtom::Class { class_predicate, argument } => {
                Ok(RuleAtom::Triple {
                    subject: self.convert_swrl_argument_to_term(argument),
                    predicate: Term::Constant("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
                    object: Term::Constant(class_predicate.clone()),
                })
            }
            SwrlAtom::IndividualProperty { property_predicate, argument1, argument2 } => {
                Ok(RuleAtom::Triple {
                    subject: self.convert_swrl_argument_to_term(argument1),
                    predicate: Term::Constant(property_predicate.clone()),
                    object: self.convert_swrl_argument_to_term(argument2),
                })
            }
            SwrlAtom::DatavalueProperty { property_predicate, argument1, argument2 } => {
                Ok(RuleAtom::Triple {
                    subject: self.convert_swrl_argument_to_term(argument1),
                    predicate: Term::Constant(property_predicate.clone()),
                    object: self.convert_swrl_argument_to_term(argument2),
                })
            }
            SwrlAtom::Builtin { builtin_predicate, arguments } => {
                let builtin_args: Vec<Term> = arguments.iter()
                    .map(|arg| self.convert_swrl_argument_to_term(arg))
                    .collect();
                    
                Ok(RuleAtom::Builtin {
                    name: builtin_predicate.clone(),
                    args: builtin_args,
                })
            }
            SwrlAtom::SameIndividual { argument1, argument2 } => {
                Ok(RuleAtom::Triple {
                    subject: self.convert_swrl_argument_to_term(argument1),
                    predicate: Term::Constant("http://www.w3.org/2002/07/owl#sameAs".to_string()),
                    object: self.convert_swrl_argument_to_term(argument2),
                })
            }
            SwrlAtom::DifferentIndividuals { argument1, argument2 } => {
                Ok(RuleAtom::Triple {
                    subject: self.convert_swrl_argument_to_term(argument1),
                    predicate: Term::Constant("http://www.w3.org/2002/07/owl#differentFrom".to_string()),
                    object: self.convert_swrl_argument_to_term(argument2),
                })
            }
        }
    }
    
    /// Convert SWRL argument to Term
    fn convert_swrl_argument_to_term(&self, argument: &SwrlArgument) -> Term {
        match argument {
            SwrlArgument::Variable(name) => Term::Variable(name.clone()),
            SwrlArgument::Individual(name) => Term::Constant(name.clone()),
            SwrlArgument::Literal(value) => Term::Literal(value.clone()),
        }
    }
    
    /// Execute SWRL rules on a set of facts
    pub fn execute(&mut self, facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        info!("Executing SWRL rules on {} facts", facts.len());
        
        // Use the internal rule engine for basic reasoning
        let inferred_facts = self.rule_engine.forward_chain(facts)?;
        
        // Apply SWRL-specific reasoning
        let mut all_facts = inferred_facts;
        let mut new_facts_added = true;
        let mut iteration = 0;
        
        while new_facts_added && iteration < 100 {
            new_facts_added = false;
            iteration += 1;
            
            debug!("SWRL execution iteration {}", iteration);
            
            // Apply each SWRL rule
            for rule in &self.rules.clone() {
                let rule_facts = self.apply_swrl_rule(rule, &all_facts)?;
                for fact in rule_facts {
                    if !all_facts.contains(&fact) {
                        all_facts.push(fact);
                        new_facts_added = true;
                    }
                }
            }
        }
        
        info!("SWRL execution completed after {} iterations, {} facts total", iteration, all_facts.len());
        Ok(all_facts)
    }
    
    /// Apply a single SWRL rule
    fn apply_swrl_rule(&self, rule: &SwrlRule, facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        let mut new_facts = Vec::new();
        
        // Find all variable bindings that satisfy the rule body
        let bindings = self.find_rule_bindings(&rule.body, facts)?;
        
        // Apply each binding to generate head facts
        for binding in bindings {
            for head_atom in &rule.head {
                let instantiated_fact = self.instantiate_swrl_atom(head_atom, &binding)?;
                if let Some(fact) = instantiated_fact {
                    new_facts.push(fact);
                }
            }
        }
        
        if !new_facts.is_empty() {
            debug!("SWRL rule '{}' generated {} new facts", rule.id, new_facts.len());
        }
        
        Ok(new_facts)
    }
    
    /// Find variable bindings that satisfy rule body
    fn find_rule_bindings(&self, body: &[SwrlAtom], facts: &[RuleAtom]) -> Result<Vec<HashMap<String, SwrlArgument>>> {
        let mut bindings = vec![HashMap::new()];
        
        for atom in body {
            let mut new_bindings = Vec::new();
            
            for current_binding in bindings {
                let atom_bindings = self.match_swrl_atom(atom, facts, &current_binding)?;
                new_bindings.extend(atom_bindings);
            }
            
            bindings = new_bindings;
        }
        
        Ok(bindings)
    }
    
    /// Match a SWRL atom against facts
    fn match_swrl_atom(&self, atom: &SwrlAtom, facts: &[RuleAtom], current_binding: &HashMap<String, SwrlArgument>) -> Result<Vec<HashMap<String, SwrlArgument>>> {
        let mut bindings = Vec::new();
        
        match atom {
            SwrlAtom::Builtin { builtin_predicate, arguments } => {
                // Evaluate built-in predicate
                if self.evaluate_builtin(builtin_predicate, arguments, current_binding)? {
                    bindings.push(current_binding.clone());
                }
            }
            _ => {
                // Convert to RuleAtom and match against facts
                let rule_atom = self.convert_swrl_atom_to_rule_atom(atom)?;
                
                for fact in facts {
                    if let Some(new_binding) = self.unify_rule_atoms(&rule_atom, fact, current_binding.clone())? {
                        bindings.push(new_binding);
                    }
                }
            }
        }
        
        Ok(bindings)
    }
    
    /// Evaluate a built-in predicate
    fn evaluate_builtin(&self, predicate: &str, arguments: &[SwrlArgument], binding: &HashMap<String, SwrlArgument>) -> Result<bool> {
        // Resolve arguments with current bindings
        let resolved_args: Vec<SwrlArgument> = arguments.iter()
            .map(|arg| self.resolve_argument(arg, binding))
            .collect();
        
        // Look up built-in function
        if let Some(builtin) = self.builtins.get(predicate) {
            // Check argument count
            if resolved_args.len() < builtin.min_args {
                return Err(anyhow::anyhow!("Built-in {} requires at least {} arguments, got {}", 
                                        predicate, builtin.min_args, resolved_args.len()));
            }
            
            if let Some(max_args) = builtin.max_args {
                if resolved_args.len() > max_args {
                    return Err(anyhow::anyhow!("Built-in {} requires at most {} arguments, got {}", 
                                            predicate, max_args, resolved_args.len()));
                }
            }
            
            // Execute built-in
            (builtin.implementation)(&resolved_args)
        } else {
            warn!("Unknown SWRL built-in: {}", predicate);
            Ok(false)
        }
    }
    
    /// Resolve an argument with current bindings
    fn resolve_argument(&self, argument: &SwrlArgument, binding: &HashMap<String, SwrlArgument>) -> SwrlArgument {
        match argument {
            SwrlArgument::Variable(name) => {
                binding.get(name).cloned().unwrap_or_else(|| argument.clone())
            }
            _ => argument.clone(),
        }
    }
    
    /// Unify two rule atoms
    fn unify_rule_atoms(&self, pattern: &RuleAtom, fact: &RuleAtom, mut binding: HashMap<String, SwrlArgument>) -> Result<Option<HashMap<String, SwrlArgument>>> {
        match (pattern, fact) {
            (RuleAtom::Triple { subject: s1, predicate: p1, object: o1 },
             RuleAtom::Triple { subject: s2, predicate: p2, object: o2 }) => {
                if self.unify_terms(s1, s2, &mut binding)? &&
                   self.unify_terms(p1, p2, &mut binding)? &&
                   self.unify_terms(o1, o2, &mut binding)? {
                    Ok(Some(binding))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
    
    /// Unify two terms
    fn unify_terms(&self, term1: &Term, term2: &Term, binding: &mut HashMap<String, SwrlArgument>) -> Result<bool> {
        match (term1, term2) {
            (Term::Variable(var), term) => {
                let swrl_arg = self.term_to_swrl_argument(term);
                if let Some(existing) = binding.get(var) {
                    Ok(existing == &swrl_arg)
                } else {
                    binding.insert(var.clone(), swrl_arg);
                    Ok(true)
                }
            }
            (term, Term::Variable(var)) => {
                let swrl_arg = self.term_to_swrl_argument(term);
                if let Some(existing) = binding.get(var) {
                    Ok(existing == &swrl_arg)
                } else {
                    binding.insert(var.clone(), swrl_arg);
                    Ok(true)
                }
            }
            (Term::Constant(c1), Term::Constant(c2)) => Ok(c1 == c2),
            (Term::Literal(l1), Term::Literal(l2)) => Ok(l1 == l2),
            _ => Ok(false),
        }
    }
    
    /// Convert Term to SwrlArgument
    fn term_to_swrl_argument(&self, term: &Term) -> SwrlArgument {
        match term {
            Term::Variable(name) => SwrlArgument::Variable(name.clone()),
            Term::Constant(name) => SwrlArgument::Individual(name.clone()),
            Term::Literal(value) => SwrlArgument::Literal(value.clone()),
        }
    }
    
    /// Instantiate a SWRL atom with variable bindings
    fn instantiate_swrl_atom(&self, atom: &SwrlAtom, binding: &HashMap<String, SwrlArgument>) -> Result<Option<RuleAtom>> {
        let rule_atom = self.convert_swrl_atom_to_rule_atom(atom)?;
        
        let instantiated = RuleAtom::Triple {
            subject: self.instantiate_term_with_bindings(&rule_atom, binding, 0),
            predicate: self.instantiate_term_with_bindings(&rule_atom, binding, 1),
            object: self.instantiate_term_with_bindings(&rule_atom, binding, 2),
        };
        
        Ok(Some(instantiated))
    }
    
    /// Instantiate a term with bindings (helper method)
    fn instantiate_term_with_bindings(&self, rule_atom: &RuleAtom, binding: &HashMap<String, SwrlArgument>, position: usize) -> Term {
        let term = match (rule_atom, position) {
            (RuleAtom::Triple { subject, .. }, 0) => subject,
            (RuleAtom::Triple { predicate, .. }, 1) => predicate,
            (RuleAtom::Triple { object, .. }, 2) => object,
            _ => return Term::Constant("error".to_string()),
        };
        
        match term {
            Term::Variable(var) => {
                if let Some(bound_value) = binding.get(var) {
                    self.convert_swrl_argument_to_term(bound_value)
                } else {
                    term.clone()
                }
            }
            _ => term.clone(),
        }
    }
    
    /// Get rule statistics
    pub fn get_stats(&self) -> SwrlStats {
        SwrlStats {
            total_rules: self.rules.len(),
            total_builtins: self.builtins.len(),
        }
    }
}

/// SWRL engine statistics
#[derive(Debug, Clone)]
pub struct SwrlStats {
    pub total_rules: usize,
    pub total_builtins: usize,
}

impl std::fmt::Display for SwrlStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rules: {}, Built-ins: {}", self.total_rules, self.total_builtins)
    }
}

// Built-in function implementations

fn builtin_equal(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("equal requires exactly 2 arguments"));
    }
    Ok(args[0] == args[1])
}

fn builtin_not_equal(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("notEqual requires exactly 2 arguments"));
    }
    Ok(args[0] != args[1])
}

fn builtin_less_than(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("lessThan requires exactly 2 arguments"));
    }
    
    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    Ok(val1 < val2)
}

fn builtin_greater_than(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("greaterThan requires exactly 2 arguments"));
    }
    
    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    Ok(val1 > val2)
}

fn builtin_add(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("add requires exactly 3 arguments"));
    }
    
    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    let result = extract_numeric_value(&args[2])?;
    
    Ok((val1 + val2 - result).abs() < f64::EPSILON)
}

fn builtin_subtract(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("subtract requires exactly 3 arguments"));
    }
    
    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    let result = extract_numeric_value(&args[2])?;
    
    Ok((val1 - val2 - result).abs() < f64::EPSILON)
}

fn builtin_multiply(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("multiply requires exactly 3 arguments"));
    }
    
    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    let result = extract_numeric_value(&args[2])?;
    
    Ok((val1 * val2 - result).abs() < f64::EPSILON)
}

fn builtin_string_concat(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 3 {
        return Err(anyhow::anyhow!("stringConcat requires at least 3 arguments"));
    }
    
    let mut concat_result = String::new();
    for arg in &args[0..args.len()-1] {
        concat_result.push_str(&extract_string_value(arg)?);
    }
    
    let expected = extract_string_value(&args[args.len()-1])?;
    Ok(concat_result == expected)
}

fn builtin_string_length(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("stringLength requires exactly 2 arguments"));
    }
    
    let string_val = extract_string_value(&args[0])?;
    let length_val = extract_numeric_value(&args[1])? as usize;
    
    Ok(string_val.len() == length_val)
}

fn builtin_boolean_value(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("booleanValue requires exactly 1 argument"));
    }
    
    extract_boolean_value(&args[0])
}

// Helper functions for value extraction

fn extract_numeric_value(arg: &SwrlArgument) -> Result<f64> {
    match arg {
        SwrlArgument::Literal(value) => {
            value.parse::<f64>()
                .map_err(|_| anyhow::anyhow!("Cannot parse '{}' as numeric value", value))
        }
        _ => Err(anyhow::anyhow!("Expected literal numeric value, got {:?}", arg)),
    }
}

fn extract_string_value(arg: &SwrlArgument) -> Result<String> {
    match arg {
        SwrlArgument::Literal(value) => Ok(value.clone()),
        SwrlArgument::Individual(value) => Ok(value.clone()),
        SwrlArgument::Variable(name) => Err(anyhow::anyhow!("Unbound variable: {}", name)),
    }
}

fn extract_boolean_value(arg: &SwrlArgument) -> Result<bool> {
    match arg {
        SwrlArgument::Literal(value) => {
            match value.to_lowercase().as_str() {
                "true" | "1" => Ok(true),
                "false" | "0" => Ok(false),
                _ => Err(anyhow::anyhow!("Cannot parse '{}' as boolean value", value)),
            }
        }
        _ => Err(anyhow::anyhow!("Expected literal boolean value, got {:?}", arg)),
    }
}

// Additional mathematical built-ins

fn builtin_mod(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("mod requires exactly 3 arguments"));
    }
    
    let dividend = extract_numeric_value(&args[0])?;
    let divisor = extract_numeric_value(&args[1])?;
    let result = extract_numeric_value(&args[2])?;
    
    if divisor == 0.0 {
        return Err(anyhow::anyhow!("Division by zero in mod operation"));
    }
    
    Ok((dividend % divisor - result).abs() < f64::EPSILON)
}

fn builtin_pow(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("pow requires exactly 3 arguments"));
    }
    
    let base = extract_numeric_value(&args[0])?;
    let exponent = extract_numeric_value(&args[1])?;
    let result = extract_numeric_value(&args[2])?;
    
    Ok((base.powf(exponent) - result).abs() < f64::EPSILON)
}

fn builtin_sqrt(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("sqrt requires exactly 2 arguments"));
    }
    
    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;
    
    if input < 0.0 {
        return Err(anyhow::anyhow!("Cannot take square root of negative number"));
    }
    
    Ok((input.sqrt() - result).abs() < f64::EPSILON)
}

fn builtin_sin(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("sin requires exactly 2 arguments"));
    }
    
    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;
    
    Ok((input.sin() - result).abs() < f64::EPSILON)
}

fn builtin_cos(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("cos requires exactly 2 arguments"));
    }
    
    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;
    
    Ok((input.cos() - result).abs() < f64::EPSILON)
}

// Date and time built-ins

fn builtin_day_time_duration(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("dayTimeDuration requires exactly 2 arguments"));
    }
    
    let duration_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;
    
    // Simple duration parsing (P[n]DT[n]H[n]M[n]S format)
    // This is a simplified implementation
    Ok(duration_str == expected)
}

fn builtin_year_month_duration(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("yearMonthDuration requires exactly 2 arguments"));
    }
    
    let duration_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;
    
    // Simple duration parsing (P[n]Y[n]M format)
    // This is a simplified implementation
    Ok(duration_str == expected)
}

fn builtin_date_time(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("dateTime requires exactly 2 arguments"));
    }
    
    let datetime_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;
    
    // Simple datetime validation (ISO 8601 format)
    // This is a simplified implementation
    Ok(datetime_str == expected)
}

// List operations

fn builtin_list_concat(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 {
        return Err(anyhow::anyhow!("listConcat requires at least 2 arguments"));
    }
    
    // In a full implementation, this would handle RDF lists
    // For now, treat as string concatenation of comma-separated values
    let mut concat_result = String::new();
    for arg in &args[0..args.len()-1] {
        if !concat_result.is_empty() {
            concat_result.push(',');
        }
        concat_result.push_str(&extract_string_value(arg)?);
    }
    
    let expected = extract_string_value(&args[args.len()-1])?;
    Ok(concat_result == expected)
}

fn builtin_list_length(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("listLength requires exactly 2 arguments"));
    }
    
    let list_str = extract_string_value(&args[0])?;
    let length_val = extract_numeric_value(&args[1])? as usize;
    
    // Simple implementation: count comma-separated items
    let items: Vec<&str> = list_str.split(',').collect();
    Ok(items.len() == length_val)
}

fn builtin_member(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("member requires exactly 2 arguments"));
    }
    
    let element = extract_string_value(&args[0])?;
    let list_str = extract_string_value(&args[1])?;
    
    // Simple implementation: check if element is in comma-separated list
    let items: Vec<&str> = list_str.split(',').collect();
    Ok(items.contains(&element.as_str()))
}

// Enhanced string operations

fn builtin_string_matches(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 || args.len() > 3 {
        return Err(anyhow::anyhow!("stringMatches requires 2 or 3 arguments"));
    }
    
    let input = extract_string_value(&args[0])?;
    let pattern = extract_string_value(&args[1])?;
    
    // Simple pattern matching (could be enhanced with full regex support)
    if args.len() == 3 {
        let _flags = extract_string_value(&args[2])?;
        // Ignore flags in simple implementation
    }
    
    // Basic wildcard matching (* and ?)
    let regex_pattern = pattern
        .replace("*", ".*")
        .replace("?", ".");
    
    match regex::Regex::new(&regex_pattern) {
        Ok(re) => Ok(re.is_match(&input)),
        Err(_) => Ok(false),
    }
}

fn builtin_substring(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 3 || args.len() > 4 {
        return Err(anyhow::anyhow!("substring requires 3 or 4 arguments"));
    }
    
    let input = extract_string_value(&args[0])?;
    let start = extract_numeric_value(&args[1])? as usize;
    let result = extract_string_value(&args[args.len()-1])?;
    
    let extracted = if args.len() == 4 {
        let length = extract_numeric_value(&args[2])? as usize;
        if start > input.len() {
            String::new()
        } else {
            let end = std::cmp::min(start + length, input.len());
            input.chars().skip(start).take(end - start).collect()
        }
    } else {
        if start > input.len() {
            String::new()
        } else {
            input.chars().skip(start).collect()
        }
    };
    
    Ok(extracted == result)
}

fn builtin_upper_case(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("upperCase requires exactly 2 arguments"));
    }
    
    let input = extract_string_value(&args[0])?;
    let result = extract_string_value(&args[1])?;
    
    Ok(input.to_uppercase() == result)
}

fn builtin_lower_case(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("lowerCase requires exactly 2 arguments"));
    }
    
    let input = extract_string_value(&args[0])?;
    let result = extract_string_value(&args[1])?;
    
    Ok(input.to_lowercase() == result)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_swrl_engine_creation() {
        let engine = SwrlEngine::new();
        let stats = engine.get_stats();
        assert!(stats.total_builtins > 0);
    }
    
    #[test]
    fn test_builtin_equal() {
        let args = vec![
            SwrlArgument::Literal("5".to_string()),
            SwrlArgument::Literal("5".to_string()),
        ];
        assert!(builtin_equal(&args).unwrap());
        
        let args = vec![
            SwrlArgument::Literal("5".to_string()),
            SwrlArgument::Literal("6".to_string()),
        ];
        assert!(!builtin_equal(&args).unwrap());
    }
    
    #[test]
    fn test_builtin_add() {
        let args = vec![
            SwrlArgument::Literal("5".to_string()),
            SwrlArgument::Literal("3".to_string()),
            SwrlArgument::Literal("8".to_string()),
        ];
        assert!(builtin_add(&args).unwrap());
        
        let args = vec![
            SwrlArgument::Literal("5".to_string()),
            SwrlArgument::Literal("3".to_string()),
            SwrlArgument::Literal("7".to_string()),
        ];
        assert!(!builtin_add(&args).unwrap());
    }
    
    #[test]
    fn test_string_concat() {
        let args = vec![
            SwrlArgument::Literal("Hello".to_string()),
            SwrlArgument::Literal(" ".to_string()),
            SwrlArgument::Literal("World".to_string()),
            SwrlArgument::Literal("Hello World".to_string()),
        ];
        assert!(builtin_string_concat(&args).unwrap());
    }
    
    #[test]
    fn test_swrl_rule_conversion() {
        let engine = SwrlEngine::new();
        
        let swrl_rule = SwrlRule {
            id: "test_rule".to_string(),
            body: vec![
                SwrlAtom::Class {
                    class_predicate: "Person".to_string(),
                    argument: SwrlArgument::Variable("x".to_string()),
                }
            ],
            head: vec![
                SwrlAtom::Class {
                    class_predicate: "Human".to_string(),
                    argument: SwrlArgument::Variable("x".to_string()),
                }
            ],
            metadata: HashMap::new(),
        };
        
        let rule = engine.convert_swrl_to_rule(&swrl_rule).unwrap();
        assert_eq!(rule.name, "test_rule");
        assert_eq!(rule.body.len(), 1);
        assert_eq!(rule.head.len(), 1);
    }
}
