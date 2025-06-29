//! SWRL (Semantic Web Rule Language) Implementation
//!
//! Implementation of SWRL rule parsing, execution, and built-in predicates.
//! Supports the full SWRL specification including custom built-ins.

use crate::{Rule, RuleAtom, RuleEngine, Term};
use anyhow::Result;
use regex::{Regex, RegexBuilder};
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
    pub const SWRL_INDIVIDUAL_PROPERTY_ATOM: &str =
        "http://www.w3.org/2003/11/swrl#IndividualPropertyAtom";
    pub const SWRL_DATAVALUE_PROPERTY_ATOM: &str =
        "http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom";
    pub const SWRL_BUILTIN_ATOM: &str = "http://www.w3.org/2003/11/swrl#BuiltinAtom";
    pub const SWRL_SAME_INDIVIDUAL_ATOM: &str = "http://www.w3.org/2003/11/swrl#SameIndividualAtom";
    pub const SWRL_DIFFERENT_INDIVIDUALS_ATOM: &str =
        "http://www.w3.org/2003/11/swrl#DifferentIndividualsAtom";

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

    // SWRL-X Temporal Extensions
    pub const SWRLX_TEMPORAL_NS: &str = "http://www.w3.org/2003/11/swrlx/temporal#";

    // Temporal predicates
    pub const SWRLX_BEFORE: &str = "http://www.w3.org/2003/11/swrlx/temporal#before";
    pub const SWRLX_AFTER: &str = "http://www.w3.org/2003/11/swrlx/temporal#after";
    pub const SWRLX_DURING: &str = "http://www.w3.org/2003/11/swrlx/temporal#during";
    pub const SWRLX_OVERLAPS: &str = "http://www.w3.org/2003/11/swrlx/temporal#overlaps";
    pub const SWRLX_MEETS: &str = "http://www.w3.org/2003/11/swrlx/temporal#meets";
    pub const SWRLX_STARTS: &str = "http://www.w3.org/2003/11/swrlx/temporal#starts";
    pub const SWRLX_FINISHES: &str = "http://www.w3.org/2003/11/swrlx/temporal#finishes";
    pub const SWRLX_EQUALS: &str = "http://www.w3.org/2003/11/swrlx/temporal#equals";

    // Interval operations
    pub const SWRLX_INTERVAL_DURATION: &str =
        "http://www.w3.org/2003/11/swrlx/temporal#intervalDuration";
    pub const SWRLX_INTERVAL_START: &str = "http://www.w3.org/2003/11/swrlx/temporal#intervalStart";
    pub const SWRLX_INTERVAL_END: &str = "http://www.w3.org/2003/11/swrlx/temporal#intervalEnd";
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

        // Geographic operations
        self.register_builtin(BuiltinFunction {
            name: "distance".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 5,
            max_args: Some(5),
            implementation: builtin_distance,
        });

        self.register_builtin(BuiltinFunction {
            name: "within".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 5,
            max_args: Some(5),
            implementation: builtin_within,
        });

        // Advanced mathematical functions
        self.register_builtin(BuiltinFunction {
            name: "tan".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_tan,
        });

        self.register_builtin(BuiltinFunction {
            name: "asin".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_asin,
        });

        self.register_builtin(BuiltinFunction {
            name: "acos".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_acos,
        });

        self.register_builtin(BuiltinFunction {
            name: "atan".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_atan,
        });

        self.register_builtin(BuiltinFunction {
            name: "log".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_log,
        });

        self.register_builtin(BuiltinFunction {
            name: "exp".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_exp,
        });

        // Temporal operations
        self.register_builtin(BuiltinFunction {
            name: "dateAdd".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 4,
            max_args: Some(4),
            implementation: builtin_date_add,
        });

        self.register_builtin(BuiltinFunction {
            name: "dateDiff".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_date_diff,
        });

        self.register_builtin(BuiltinFunction {
            name: "now".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 1,
            max_args: Some(1),
            implementation: builtin_now,
        });

        // SWRL-X Temporal Extensions
        self.register_builtin(BuiltinFunction {
            name: "before".to_string(),
            namespace: vocabulary::SWRLX_TEMPORAL_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_temporal_before,
        });

        self.register_builtin(BuiltinFunction {
            name: "after".to_string(),
            namespace: vocabulary::SWRLX_TEMPORAL_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_temporal_after,
        });

        self.register_builtin(BuiltinFunction {
            name: "during".to_string(),
            namespace: vocabulary::SWRLX_TEMPORAL_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_temporal_during,
        });

        self.register_builtin(BuiltinFunction {
            name: "overlaps".to_string(),
            namespace: vocabulary::SWRLX_TEMPORAL_NS.to_string(),
            min_args: 4,
            max_args: Some(4),
            implementation: builtin_temporal_overlaps,
        });

        self.register_builtin(BuiltinFunction {
            name: "meets".to_string(),
            namespace: vocabulary::SWRLX_TEMPORAL_NS.to_string(),
            min_args: 4,
            max_args: Some(4),
            implementation: builtin_temporal_meets,
        });

        self.register_builtin(BuiltinFunction {
            name: "intervalDuration".to_string(),
            namespace: vocabulary::SWRLX_TEMPORAL_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_interval_duration,
        });

        // Additional mathematical functions
        self.register_builtin(BuiltinFunction {
            name: "abs".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_abs,
        });

        self.register_builtin(BuiltinFunction {
            name: "floor".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_floor,
        });

        self.register_builtin(BuiltinFunction {
            name: "ceil".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_ceil,
        });

        self.register_builtin(BuiltinFunction {
            name: "round".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_round,
        });

        // Additional list operations
        self.register_builtin(BuiltinFunction {
            name: "first".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_list_first,
        });

        self.register_builtin(BuiltinFunction {
            name: "rest".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(2),
            implementation: builtin_list_rest,
        });

        self.register_builtin(BuiltinFunction {
            name: "nth".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_list_nth,
        });

        self.register_builtin(BuiltinFunction {
            name: "append".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 3,
            max_args: Some(3),
            implementation: builtin_list_append,
        });

        // Enhanced string operations with regex
        self.register_builtin(BuiltinFunction {
            name: "stringMatchesRegex".to_string(),
            namespace: vocabulary::SWRLB_NS.to_string(),
            min_args: 2,
            max_args: Some(3),
            implementation: builtin_string_matches_regex,
        });

        // Additional geographic operations
        self.register_builtin(BuiltinFunction {
            name: "contains".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 9,
            max_args: Some(9),
            implementation: builtin_geo_contains,
        });

        self.register_builtin(BuiltinFunction {
            name: "intersects".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 8,
            max_args: Some(8),
            implementation: builtin_geo_intersects,
        });

        self.register_builtin(BuiltinFunction {
            name: "area".to_string(),
            namespace: vocabulary::SWRLX_NS.to_string(),
            min_args: 5,
            max_args: Some(5),
            implementation: builtin_geo_area,
        });

        info!(
            "Registered {} core SWRL built-in functions",
            self.builtins.len()
        );
    }

    /// Register a built-in function
    pub fn register_builtin(&mut self, builtin: BuiltinFunction) {
        let full_name = format!("{}{}", builtin.namespace, builtin.name);
        debug!("Registering SWRL built-in: {}", full_name);
        self.builtins.insert(full_name, builtin);
    }

    /// Register a custom built-in function with validation
    pub fn register_custom_builtin(
        &mut self,
        name: String,
        namespace: String,
        min_args: usize,
        max_args: Option<usize>,
        implementation: fn(&[SwrlArgument]) -> Result<bool>,
    ) -> Result<(), String> {
        // Validate namespace
        if !namespace.starts_with("http://") && !namespace.starts_with("https://") {
            return Err(format!(
                "Invalid namespace '{}': must be a valid IRI",
                namespace
            ));
        }

        // Validate argument constraints
        if let Some(max) = max_args {
            if min_args > max {
                return Err(format!(
                    "Invalid argument constraints: min_args ({}) > max_args ({})",
                    min_args, max
                ));
            }
        }

        let builtin = BuiltinFunction {
            name: name.clone(),
            namespace: namespace.clone(),
            min_args,
            max_args,
            implementation,
        };

        self.register_builtin(builtin);

        info!("Registered custom SWRL built-in: {}{}", namespace, name);
        Ok(())
    }

    /// List all registered built-in functions
    pub fn list_builtins(&self) -> Vec<String> {
        self.builtins.keys().cloned().collect()
    }

    /// Check if a built-in function is registered
    pub fn has_builtin(&self, name: &str) -> bool {
        self.builtins.contains_key(name)
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
            SwrlAtom::Class {
                class_predicate,
                argument,
            } => Ok(RuleAtom::Triple {
                subject: self.convert_swrl_argument_to_term(argument),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant(class_predicate.clone()),
            }),
            SwrlAtom::IndividualProperty {
                property_predicate,
                argument1,
                argument2,
            } => Ok(RuleAtom::Triple {
                subject: self.convert_swrl_argument_to_term(argument1),
                predicate: Term::Constant(property_predicate.clone()),
                object: self.convert_swrl_argument_to_term(argument2),
            }),
            SwrlAtom::DatavalueProperty {
                property_predicate,
                argument1,
                argument2,
            } => Ok(RuleAtom::Triple {
                subject: self.convert_swrl_argument_to_term(argument1),
                predicate: Term::Constant(property_predicate.clone()),
                object: self.convert_swrl_argument_to_term(argument2),
            }),
            SwrlAtom::Builtin {
                builtin_predicate,
                arguments,
            } => {
                let builtin_args: Vec<Term> = arguments
                    .iter()
                    .map(|arg| self.convert_swrl_argument_to_term(arg))
                    .collect();

                Ok(RuleAtom::Builtin {
                    name: builtin_predicate.clone(),
                    args: builtin_args,
                })
            }
            SwrlAtom::SameIndividual {
                argument1,
                argument2,
            } => Ok(RuleAtom::Triple {
                subject: self.convert_swrl_argument_to_term(argument1),
                predicate: Term::Constant("http://www.w3.org/2002/07/owl#sameAs".to_string()),
                object: self.convert_swrl_argument_to_term(argument2),
            }),
            SwrlAtom::DifferentIndividuals {
                argument1,
                argument2,
            } => Ok(RuleAtom::Triple {
                subject: self.convert_swrl_argument_to_term(argument1),
                predicate: Term::Constant(
                    "http://www.w3.org/2002/07/owl#differentFrom".to_string(),
                ),
                object: self.convert_swrl_argument_to_term(argument2),
            }),
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

        info!(
            "SWRL execution completed after {} iterations, {} facts total",
            iteration,
            all_facts.len()
        );
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
            debug!(
                "SWRL rule '{}' generated {} new facts",
                rule.id,
                new_facts.len()
            );
        }

        Ok(new_facts)
    }

    /// Find variable bindings that satisfy rule body
    fn find_rule_bindings(
        &self,
        body: &[SwrlAtom],
        facts: &[RuleAtom],
    ) -> Result<Vec<HashMap<String, SwrlArgument>>> {
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
    fn match_swrl_atom(
        &self,
        atom: &SwrlAtom,
        facts: &[RuleAtom],
        current_binding: &HashMap<String, SwrlArgument>,
    ) -> Result<Vec<HashMap<String, SwrlArgument>>> {
        let mut bindings = Vec::new();

        match atom {
            SwrlAtom::Builtin {
                builtin_predicate,
                arguments,
            } => {
                // Evaluate built-in predicate
                if self.evaluate_builtin(builtin_predicate, arguments, current_binding)? {
                    bindings.push(current_binding.clone());
                }
            }
            _ => {
                // Convert to RuleAtom and match against facts
                let rule_atom = self.convert_swrl_atom_to_rule_atom(atom)?;

                for fact in facts {
                    if let Some(new_binding) =
                        self.unify_rule_atoms(&rule_atom, fact, current_binding.clone())?
                    {
                        bindings.push(new_binding);
                    }
                }
            }
        }

        Ok(bindings)
    }

    /// Evaluate a built-in predicate
    fn evaluate_builtin(
        &self,
        predicate: &str,
        arguments: &[SwrlArgument],
        binding: &HashMap<String, SwrlArgument>,
    ) -> Result<bool> {
        // Resolve arguments with current bindings
        let resolved_args: Vec<SwrlArgument> = arguments
            .iter()
            .map(|arg| self.resolve_argument(arg, binding))
            .collect();

        // Look up built-in function
        if let Some(builtin) = self.builtins.get(predicate) {
            // Check argument count
            if resolved_args.len() < builtin.min_args {
                return Err(anyhow::anyhow!(
                    "Built-in {} requires at least {} arguments, got {}",
                    predicate,
                    builtin.min_args,
                    resolved_args.len()
                ));
            }

            if let Some(max_args) = builtin.max_args {
                if resolved_args.len() > max_args {
                    return Err(anyhow::anyhow!(
                        "Built-in {} requires at most {} arguments, got {}",
                        predicate,
                        max_args,
                        resolved_args.len()
                    ));
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
    fn resolve_argument(
        &self,
        argument: &SwrlArgument,
        binding: &HashMap<String, SwrlArgument>,
    ) -> SwrlArgument {
        match argument {
            SwrlArgument::Variable(name) => binding
                .get(name)
                .cloned()
                .unwrap_or_else(|| argument.clone()),
            _ => argument.clone(),
        }
    }

    /// Unify two rule atoms
    fn unify_rule_atoms(
        &self,
        pattern: &RuleAtom,
        fact: &RuleAtom,
        mut binding: HashMap<String, SwrlArgument>,
    ) -> Result<Option<HashMap<String, SwrlArgument>>> {
        match (pattern, fact) {
            (
                RuleAtom::Triple {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
                RuleAtom::Triple {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                if self.unify_terms(s1, s2, &mut binding)?
                    && self.unify_terms(p1, p2, &mut binding)?
                    && self.unify_terms(o1, o2, &mut binding)?
                {
                    Ok(Some(binding))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Unify two terms
    fn unify_terms(
        &self,
        term1: &Term,
        term2: &Term,
        binding: &mut HashMap<String, SwrlArgument>,
    ) -> Result<bool> {
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
    fn instantiate_swrl_atom(
        &self,
        atom: &SwrlAtom,
        binding: &HashMap<String, SwrlArgument>,
    ) -> Result<Option<RuleAtom>> {
        let rule_atom = self.convert_swrl_atom_to_rule_atom(atom)?;

        let instantiated = RuleAtom::Triple {
            subject: self.instantiate_term_with_bindings(&rule_atom, binding, 0),
            predicate: self.instantiate_term_with_bindings(&rule_atom, binding, 1),
            object: self.instantiate_term_with_bindings(&rule_atom, binding, 2),
        };

        Ok(Some(instantiated))
    }

    /// Instantiate a term with bindings (helper method)
    fn instantiate_term_with_bindings(
        &self,
        rule_atom: &RuleAtom,
        binding: &HashMap<String, SwrlArgument>,
        position: usize,
    ) -> Term {
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
        write!(
            f,
            "Rules: {}, Built-ins: {}",
            self.total_rules, self.total_builtins
        )
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

pub fn builtin_add(args: &[SwrlArgument]) -> Result<bool> {
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

pub fn builtin_multiply(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("multiply requires exactly 3 arguments"));
    }

    let val1 = extract_numeric_value(&args[0])?;
    let val2 = extract_numeric_value(&args[1])?;
    let result = extract_numeric_value(&args[2])?;

    Ok((val1 * val2 - result).abs() < f64::EPSILON)
}

pub fn builtin_string_concat(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 3 {
        return Err(anyhow::anyhow!(
            "stringConcat requires at least 3 arguments"
        ));
    }

    let mut concat_result = String::new();
    for arg in &args[0..args.len() - 1] {
        concat_result.push_str(&extract_string_value(arg)?);
    }

    let expected = extract_string_value(&args[args.len() - 1])?;
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
        SwrlArgument::Literal(value) => value
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("Cannot parse '{}' as numeric value", value)),
        _ => Err(anyhow::anyhow!(
            "Expected literal numeric value, got {:?}",
            arg
        )),
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
        SwrlArgument::Literal(value) => match value.to_lowercase().as_str() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => Err(anyhow::anyhow!("Cannot parse '{}' as boolean value", value)),
        },
        _ => Err(anyhow::anyhow!(
            "Expected literal boolean value, got {:?}",
            arg
        )),
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

pub fn builtin_pow(args: &[SwrlArgument]) -> Result<bool> {
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
        return Err(anyhow::anyhow!(
            "Cannot take square root of negative number"
        ));
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
        return Err(anyhow::anyhow!(
            "dayTimeDuration requires exactly 2 arguments"
        ));
    }

    let duration_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple duration parsing (P[n]DT[n]H[n]M[n]S format)
    // This is a simplified implementation
    Ok(duration_str == expected)
}

fn builtin_year_month_duration(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "yearMonthDuration requires exactly 2 arguments"
        ));
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
    for arg in &args[0..args.len() - 1] {
        if !concat_result.is_empty() {
            concat_result.push(',');
        }
        concat_result.push_str(&extract_string_value(arg)?);
    }

    let expected = extract_string_value(&args[args.len() - 1])?;
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
    let regex_pattern = pattern.replace("*", ".*").replace("?", ".");

    match Regex::new(&regex_pattern) {
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
    let result = extract_string_value(&args[args.len() - 1])?;

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

pub fn builtin_upper_case(args: &[SwrlArgument]) -> Result<bool> {
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

// Advanced mathematical built-ins
fn builtin_tan(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("tan requires exactly 2 arguments"));
    }

    let x = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((x.tan() - result).abs() < f64::EPSILON)
}

fn builtin_asin(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("asin requires exactly 2 arguments"));
    }

    let x = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    if x < -1.0 || x > 1.0 {
        return Ok(false);
    }

    Ok((x.asin() - result).abs() < f64::EPSILON)
}

fn builtin_acos(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("acos requires exactly 2 arguments"));
    }

    let x = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    if x < -1.0 || x > 1.0 {
        return Ok(false);
    }

    Ok((x.acos() - result).abs() < f64::EPSILON)
}

fn builtin_atan(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("atan requires exactly 2 arguments"));
    }

    let x = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((x.atan() - result).abs() < f64::EPSILON)
}

fn builtin_log(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("log requires exactly 2 arguments"));
    }

    let x = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    if x <= 0.0 {
        return Ok(false);
    }

    Ok((x.ln() - result).abs() < f64::EPSILON)
}

fn builtin_exp(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("exp requires exactly 2 arguments"));
    }

    let x = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((x.exp() - result).abs() < f64::EPSILON)
}

// Geographic operations
fn builtin_distance(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 5 {
        return Err(anyhow::anyhow!(
            "distance requires exactly 5 arguments: lat1, lon1, lat2, lon2, result"
        ));
    }

    let lat1 = extract_numeric_value(&args[0])?.to_radians();
    let lon1 = extract_numeric_value(&args[1])?.to_radians();
    let lat2 = extract_numeric_value(&args[2])?.to_radians();
    let lon2 = extract_numeric_value(&args[3])?.to_radians();
    let expected_distance = extract_numeric_value(&args[4])?;

    // Haversine formula for great circle distance
    let earth_radius = 6371.0; // Earth radius in kilometers
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    let distance = earth_radius * c;

    Ok((distance - expected_distance).abs() < 0.001) // 1 meter tolerance
}

fn builtin_within(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 5 {
        return Err(anyhow::anyhow!(
            "within requires exactly 5 arguments: lat1, lon1, lat2, lon2, max_distance"
        ));
    }

    let lat1 = extract_numeric_value(&args[0])?.to_radians();
    let lon1 = extract_numeric_value(&args[1])?.to_radians();
    let lat2 = extract_numeric_value(&args[2])?.to_radians();
    let lon2 = extract_numeric_value(&args[3])?.to_radians();
    let max_distance = extract_numeric_value(&args[4])?;

    // Haversine formula for great circle distance
    let earth_radius = 6371.0; // Earth radius in kilometers
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    let distance = earth_radius * c;

    Ok(distance <= max_distance)
}

// Temporal operations
fn builtin_date_add(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 4 {
        return Err(anyhow::anyhow!(
            "dateAdd requires exactly 4 arguments: date, duration, unit, result"
        ));
    }

    let date_str = extract_string_value(&args[0])?;
    let duration = extract_numeric_value(&args[1])? as i64;
    let unit = extract_string_value(&args[2])?;
    let expected_result = extract_string_value(&args[3])?;

    // Parse ISO 8601 date string (simplified - in production would use chrono)
    if let Ok(timestamp) = date_str.parse::<i64>() {
        let seconds_to_add = match unit.as_str() {
            "seconds" => duration,
            "minutes" => duration * 60,
            "hours" => duration * 3600,
            "days" => duration * 86400,
            "weeks" => duration * 604800,
            _ => return Err(anyhow::anyhow!("Unsupported time unit: {}", unit)),
        };

        let result_timestamp = timestamp + seconds_to_add;
        Ok(result_timestamp.to_string() == expected_result)
    } else {
        // For proper date strings, would need chrono crate
        Ok(false)
    }
}

fn builtin_date_diff(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "dateDiff requires exactly 3 arguments: date1, date2, result"
        ));
    }

    let date1_str = extract_string_value(&args[0])?;
    let date2_str = extract_string_value(&args[1])?;
    let expected_diff = extract_numeric_value(&args[2])?;

    // Parse timestamps (simplified - in production would use chrono)
    if let (Ok(timestamp1), Ok(timestamp2)) = (date1_str.parse::<i64>(), date2_str.parse::<i64>()) {
        let diff_seconds = (timestamp2 - timestamp1).abs() as f64;
        Ok((diff_seconds - expected_diff).abs() < 1.0) // 1 second tolerance
    } else {
        Ok(false)
    }
}

fn builtin_now(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 1 {
        return Err(anyhow::anyhow!("now requires exactly 1 argument: result"));
    }

    let expected_result = extract_string_value(&args[0])?;

    // Get current timestamp (simplified - in production would use proper time crate)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| anyhow::anyhow!("Time error: {}", e))?
        .as_secs();

    // Check if the expected result is close to current time (within 1 second)
    if let Ok(expected_timestamp) = expected_result.parse::<u64>() {
        Ok((now as i64 - expected_timestamp as i64).abs() <= 1)
    } else {
        Ok(false)
    }
}

// SWRL-X Temporal Extensions
fn builtin_temporal_before(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "temporal:before requires exactly 2 arguments: time1, time2"
        ));
    }

    let time1 = extract_numeric_value(&args[0])?;
    let time2 = extract_numeric_value(&args[1])?;

    Ok(time1 < time2)
}

fn builtin_temporal_after(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "temporal:after requires exactly 2 arguments: time1, time2"
        ));
    }

    let time1 = extract_numeric_value(&args[0])?;
    let time2 = extract_numeric_value(&args[1])?;

    Ok(time1 > time2)
}

fn builtin_temporal_during(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "temporal:during requires exactly 3 arguments: time, interval_start, interval_end"
        ));
    }

    let time = extract_numeric_value(&args[0])?;
    let interval_start = extract_numeric_value(&args[1])?;
    let interval_end = extract_numeric_value(&args[2])?;

    Ok(time >= interval_start && time <= interval_end)
}

fn builtin_temporal_overlaps(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 4 {
        return Err(anyhow::anyhow!(
            "temporal:overlaps requires exactly 4 arguments: start1, end1, start2, end2"
        ));
    }

    let start1 = extract_numeric_value(&args[0])?;
    let end1 = extract_numeric_value(&args[1])?;
    let start2 = extract_numeric_value(&args[2])?;
    let end2 = extract_numeric_value(&args[3])?;

    // Two intervals overlap if they have any time in common
    Ok(start1 < end2 && start2 < end1)
}

fn builtin_temporal_meets(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 4 {
        return Err(anyhow::anyhow!(
            "temporal:meets requires exactly 4 arguments: start1, end1, start2, end2"
        ));
    }

    let start1 = extract_numeric_value(&args[0])?;
    let end1 = extract_numeric_value(&args[1])?;
    let start2 = extract_numeric_value(&args[2])?;
    let _end2 = extract_numeric_value(&args[3])?;

    // First interval meets second if end of first equals start of second
    Ok((end1 - start2).abs() < f64::EPSILON)
}

fn builtin_interval_duration(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "temporal:intervalDuration requires exactly 3 arguments: start, end, duration"
        ));
    }

    let start = extract_numeric_value(&args[0])?;
    let end = extract_numeric_value(&args[1])?;
    let expected_duration = extract_numeric_value(&args[2])?;

    let actual_duration = (end - start).abs();
    Ok((actual_duration - expected_duration).abs() < f64::EPSILON)
}

// Additional mathematical built-in functions

fn builtin_abs(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("abs requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((input.abs() - result).abs() < f64::EPSILON)
}

fn builtin_floor(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("floor requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((input.floor() - result).abs() < f64::EPSILON)
}

fn builtin_ceil(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("ceil requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((input.ceil() - result).abs() < f64::EPSILON)
}

fn builtin_round(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("round requires exactly 2 arguments"));
    }

    let input = extract_numeric_value(&args[0])?;
    let result = extract_numeric_value(&args[1])?;

    Ok((input.round() - result).abs() < f64::EPSILON)
}

// Additional list operations

fn builtin_list_first(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("first requires exactly 2 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple implementation: get first item from comma-separated list
    let items: Vec<&str> = list_str.split(',').collect();
    if items.is_empty() {
        Ok(expected.is_empty())
    } else {
        Ok(items[0] == expected)
    }
}

fn builtin_list_rest(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("rest requires exactly 2 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple implementation: get all but first item from comma-separated list
    let items: Vec<&str> = list_str.split(',').collect();
    if items.len() <= 1 {
        Ok(expected.is_empty())
    } else {
        let rest = items[1..].join(",");
        Ok(rest == expected)
    }
}

fn builtin_list_nth(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("nth requires exactly 3 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let index = extract_numeric_value(&args[1])? as usize;
    let expected = extract_string_value(&args[2])?;

    // Simple implementation: get nth item from comma-separated list (0-indexed)
    let items: Vec<&str> = list_str.split(',').collect();
    if index >= items.len() {
        return Err(anyhow::anyhow!(
            "Index {} out of bounds for list of length {}",
            index,
            items.len()
        ));
    }
    Ok(items[index] == expected)
}

fn builtin_list_append(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("append requires exactly 3 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let item = extract_string_value(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    // Simple implementation: append item to comma-separated list
    let result = if list_str.is_empty() {
        item
    } else {
        format!("{},{}", list_str, item)
    };
    Ok(result == expected)
}

// Enhanced string operations with full regex support

fn builtin_string_matches_regex(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 || args.len() > 3 {
        return Err(anyhow::anyhow!(
            "stringMatchesRegex requires 2 or 3 arguments"
        ));
    }

    let input = extract_string_value(&args[0])?;
    let pattern = extract_string_value(&args[1])?;

    // Full regex support with optional flags
    let regex_builder = if args.len() == 3 {
        let flags = extract_string_value(&args[2])?;
        let mut builder = RegexBuilder::new(&pattern);

        for flag in flags.chars() {
            match flag {
                'i' => {
                    builder.case_insensitive(true);
                }
                'm' => {
                    builder.multi_line(true);
                }
                's' => {
                    builder.dot_matches_new_line(true);
                }
                'x' => {
                    builder.ignore_whitespace(true);
                }
                _ => return Err(anyhow::anyhow!("Unknown regex flag: {}", flag)),
            }
        }
        builder
    } else {
        RegexBuilder::new(&pattern)
    };

    match regex_builder.build() {
        Ok(re) => Ok(re.is_match(&input)),
        Err(e) => Err(anyhow::anyhow!("Invalid regex pattern: {}", e)),
    }
}

// Additional geographic operations

fn builtin_geo_contains(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 9 {
        return Err(anyhow::anyhow!("contains requires exactly 9 arguments: poly_min_lat, poly_min_lon, poly_max_lat, poly_max_lon, point_lat, point_lon, result"));
    }

    // Simple bounding box containment check
    let min_lat = extract_numeric_value(&args[0])?;
    let min_lon = extract_numeric_value(&args[1])?;
    let max_lat = extract_numeric_value(&args[2])?;
    let max_lon = extract_numeric_value(&args[3])?;
    let point_lat = extract_numeric_value(&args[4])?;
    let point_lon = extract_numeric_value(&args[5])?;
    let expected_result = extract_string_value(&args[6])? == "true";

    let contains = point_lat >= min_lat
        && point_lat <= max_lat
        && point_lon >= min_lon
        && point_lon <= max_lon;

    Ok(contains == expected_result)
}

fn builtin_geo_intersects(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 8 {
        return Err(anyhow::anyhow!("intersects requires exactly 8 arguments: box1_min_lat, box1_min_lon, box1_max_lat, box1_max_lon, box2_min_lat, box2_min_lon, box2_max_lat, box2_max_lon"));
    }

    // Check if two bounding boxes intersect
    let box1_min_lat = extract_numeric_value(&args[0])?;
    let box1_min_lon = extract_numeric_value(&args[1])?;
    let box1_max_lat = extract_numeric_value(&args[2])?;
    let box1_max_lon = extract_numeric_value(&args[3])?;
    let box2_min_lat = extract_numeric_value(&args[4])?;
    let box2_min_lon = extract_numeric_value(&args[5])?;
    let box2_max_lat = extract_numeric_value(&args[6])?;
    let box2_max_lon = extract_numeric_value(&args[7])?;

    // Two boxes intersect if they overlap in both dimensions
    let intersects = !(box1_max_lat < box2_min_lat
        || box2_max_lat < box1_min_lat
        || box1_max_lon < box2_min_lon
        || box2_max_lon < box1_min_lon);

    Ok(intersects)
}

fn builtin_geo_area(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 5 {
        return Err(anyhow::anyhow!(
            "area requires exactly 5 arguments: min_lat, min_lon, max_lat, max_lon, result"
        ));
    }

    // Calculate approximate area of a bounding box in square kilometers
    let min_lat = extract_numeric_value(&args[0])?;
    let min_lon = extract_numeric_value(&args[1])?;
    let max_lat = extract_numeric_value(&args[2])?;
    let max_lon = extract_numeric_value(&args[3])?;
    let expected_area = extract_numeric_value(&args[4])?;

    // Simple approximation: treat Earth as sphere
    const EARTH_RADIUS_KM: f64 = 6371.0;

    // Convert to radians
    let lat1_rad = min_lat.to_radians();
    let lat2_rad = max_lat.to_radians();
    let lon_diff_rad = (max_lon - min_lon).to_radians();

    // Approximate area calculation
    let area = EARTH_RADIUS_KM * EARTH_RADIUS_KM * lon_diff_rad * (lat2_rad.sin() - lat1_rad.sin());

    // Allow some tolerance for floating point comparison
    Ok((area.abs() - expected_area).abs() < 0.1)
}

/// Temporal interval representation
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalInterval {
    pub start: f64,
    pub end: f64,
}

impl TemporalInterval {
    pub fn new(start: f64, end: f64) -> Result<Self> {
        if start > end {
            return Err(anyhow::anyhow!(
                "Invalid interval: start ({}) > end ({})",
                start,
                end
            ));
        }
        Ok(Self { start, end })
    }

    /// Check if this interval is before another interval
    pub fn before(&self, other: &TemporalInterval) -> bool {
        self.end < other.start
    }

    /// Check if this interval is after another interval
    pub fn after(&self, other: &TemporalInterval) -> bool {
        self.start > other.end
    }

    /// Check if this interval overlaps with another interval
    pub fn overlaps(&self, other: &TemporalInterval) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Check if this interval meets another interval
    pub fn meets(&self, other: &TemporalInterval) -> bool {
        (self.end - other.start).abs() < f64::EPSILON
    }

    /// Check if this interval contains a time point
    pub fn contains(&self, time: f64) -> bool {
        time >= self.start && time <= self.end
    }

    /// Get the duration of this interval
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// Custom built-in registry for user-defined functions
pub struct CustomBuiltinRegistry {
    /// Registry of custom built-in functions
    functions: HashMap<String, Box<dyn Fn(&[SwrlArgument]) -> Result<bool> + Send + Sync>>,
    /// Metadata about registered functions
    metadata: HashMap<String, BuiltinMetadata>,
}

/// Metadata for custom built-in functions
#[derive(Debug, Clone)]
pub struct BuiltinMetadata {
    pub name: String,
    pub namespace: String,
    pub description: String,
    pub min_args: usize,
    pub max_args: Option<usize>,
    pub example_usage: String,
}

impl CustomBuiltinRegistry {
    /// Create a new custom built-in registry
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Register a custom built-in function
    pub fn register<F>(&mut self, metadata: BuiltinMetadata, function: F) -> Result<()>
    where
        F: Fn(&[SwrlArgument]) -> Result<bool> + Send + Sync + 'static,
    {
        let full_name = format!("{}{}", metadata.namespace, metadata.name);

        if self.functions.contains_key(&full_name) {
            return Err(anyhow::anyhow!(
                "Built-in function '{}' already registered",
                full_name
            ));
        }

        self.functions.insert(full_name.clone(), Box::new(function));
        self.metadata.insert(full_name.clone(), metadata);

        info!("Registered custom built-in function: {}", full_name);
        Ok(())
    }

    /// Execute a custom built-in function
    pub fn execute(&self, name: &str, args: &[SwrlArgument]) -> Result<bool> {
        if let Some(function) = self.functions.get(name) {
            // Validate argument count
            if let Some(meta) = self.metadata.get(name) {
                if args.len() < meta.min_args {
                    return Err(anyhow::anyhow!(
                        "Too few arguments for '{}': expected at least {}, got {}",
                        name,
                        meta.min_args,
                        args.len()
                    ));
                }
                if let Some(max_args) = meta.max_args {
                    if args.len() > max_args {
                        return Err(anyhow::anyhow!(
                            "Too many arguments for '{}': expected at most {}, got {}",
                            name,
                            max_args,
                            args.len()
                        ));
                    }
                }
            }

            function(args)
        } else {
            Err(anyhow::anyhow!("Unknown built-in function: {}", name))
        }
    }

    /// List all registered custom built-in functions
    pub fn list_functions(&self) -> Vec<&BuiltinMetadata> {
        self.metadata.values().collect()
    }

    /// Get metadata for a specific function
    pub fn get_metadata(&self, name: &str) -> Option<&BuiltinMetadata> {
        self.metadata.get(name)
    }
}

impl Default for CustomBuiltinRegistry {
    fn default() -> Self {
        Self::new()
    }
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
            body: vec![SwrlAtom::Class {
                class_predicate: "Person".to_string(),
                argument: SwrlArgument::Variable("x".to_string()),
            }],
            head: vec![SwrlAtom::Class {
                class_predicate: "Human".to_string(),
                argument: SwrlArgument::Variable("x".to_string()),
            }],
            metadata: HashMap::new(),
        };

        let rule = engine.convert_swrl_to_rule(&swrl_rule).unwrap();
        assert_eq!(rule.name, "test_rule");
        assert_eq!(rule.body.len(), 1);
        assert_eq!(rule.head.len(), 1);
    }
}
