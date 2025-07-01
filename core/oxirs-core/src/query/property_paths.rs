//! SPARQL 1.2 Property Paths implementation
//!
//! This module implements enhanced property paths for SPARQL 1.2,
//! allowing complex graph navigation patterns.

use crate::model::{NamedNode, Term, Variable};
use crate::query::algebra::{TermPattern, TriplePattern};
use crate::OxirsError;
use std::collections::HashSet;
use std::fmt;

/// Property path expression
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PropertyPath {
    /// Direct predicate (e.g., `:knows`)
    Predicate(NamedNode),

    /// Inverse path (e.g., `^:knows`)
    Inverse(Box<PropertyPath>),

    /// Sequence of paths (e.g., `:knows/:worksFor`)
    Sequence(Box<PropertyPath>, Box<PropertyPath>),

    /// Alternative paths (e.g., `:knows|:friendOf`)
    Alternative(Box<PropertyPath>, Box<PropertyPath>),

    /// Zero or more repetitions (e.g., `:knows*`)
    ZeroOrMore(Box<PropertyPath>),

    /// One or more repetitions (e.g., `:knows+`)
    OneOrMore(Box<PropertyPath>),

    /// Zero or one occurrence (e.g., `:knows?`)
    ZeroOrOne(Box<PropertyPath>),

    /// Negated property set (e.g., `!(:knows|:hates)`)
    NegatedPropertySet(Vec<NamedNode>),

    /// Fixed length path (SPARQL 1.2 extension)
    FixedLength(Box<PropertyPath>, usize),

    /// Range length path (SPARQL 1.2 extension)
    RangeLength(Box<PropertyPath>, usize, Option<usize>),

    /// Distinct path (SPARQL 1.2 extension)
    Distinct(Box<PropertyPath>),
}

impl PropertyPath {
    /// Create a simple predicate path
    pub fn predicate(iri: NamedNode) -> Self {
        PropertyPath::Predicate(iri)
    }

    /// Create an inverse path
    pub fn inverse(path: PropertyPath) -> Self {
        PropertyPath::Inverse(Box::new(path))
    }

    /// Create a sequence path
    pub fn sequence(left: PropertyPath, right: PropertyPath) -> Self {
        PropertyPath::Sequence(Box::new(left), Box::new(right))
    }

    /// Create an alternative path
    pub fn alternative(left: PropertyPath, right: PropertyPath) -> Self {
        PropertyPath::Alternative(Box::new(left), Box::new(right))
    }

    /// Create a zero-or-more path
    pub fn zero_or_more(path: PropertyPath) -> Self {
        PropertyPath::ZeroOrMore(Box::new(path))
    }

    /// Create a one-or-more path
    pub fn one_or_more(path: PropertyPath) -> Self {
        PropertyPath::OneOrMore(Box::new(path))
    }

    /// Create a zero-or-one path
    pub fn zero_or_one(path: PropertyPath) -> Self {
        PropertyPath::ZeroOrOne(Box::new(path))
    }

    /// Create a negated property set
    pub fn negated_set(predicates: Vec<NamedNode>) -> Self {
        PropertyPath::NegatedPropertySet(predicates)
    }

    /// Create a fixed length path (SPARQL 1.2)
    pub fn fixed_length(path: PropertyPath, n: usize) -> Self {
        PropertyPath::FixedLength(Box::new(path), n)
    }

    /// Create a range length path (SPARQL 1.2)
    pub fn range_length(path: PropertyPath, min: usize, max: Option<usize>) -> Self {
        PropertyPath::RangeLength(Box::new(path), min, max)
    }

    /// Create a distinct path (SPARQL 1.2)
    pub fn distinct(path: PropertyPath) -> Self {
        PropertyPath::Distinct(Box::new(path))
    }

    /// Check if this path is simple (just a predicate)
    pub fn is_simple(&self) -> bool {
        matches!(self, PropertyPath::Predicate(_))
    }

    /// Get the minimum length of this path
    pub fn min_length(&self) -> usize {
        match self {
            PropertyPath::Predicate(_) => 1,
            PropertyPath::Inverse(p) => p.min_length(),
            PropertyPath::Sequence(l, r) => l.min_length() + r.min_length(),
            PropertyPath::Alternative(l, r) => l.min_length().min(r.min_length()),
            PropertyPath::ZeroOrMore(_) => 0,
            PropertyPath::OneOrMore(p) => p.min_length(),
            PropertyPath::ZeroOrOne(_) => 0,
            PropertyPath::NegatedPropertySet(_) => 1,
            PropertyPath::FixedLength(_, n) => *n,
            PropertyPath::RangeLength(_, min, _) => *min,
            PropertyPath::Distinct(p) => p.min_length(),
        }
    }

    /// Get the maximum length of this path (None = unbounded)
    pub fn max_length(&self) -> Option<usize> {
        match self {
            PropertyPath::Predicate(_) => Some(1),
            PropertyPath::Inverse(p) => p.max_length(),
            PropertyPath::Sequence(l, r) => match (l.max_length(), r.max_length()) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
            PropertyPath::Alternative(l, r) => match (l.max_length(), r.max_length()) {
                (Some(a), Some(b)) => Some(a.max(b)),
                _ => None,
            },
            PropertyPath::ZeroOrMore(_) => None,
            PropertyPath::OneOrMore(_) => None,
            PropertyPath::ZeroOrOne(p) => p.max_length().map(|_| 1),
            PropertyPath::NegatedPropertySet(_) => Some(1),
            PropertyPath::FixedLength(_, n) => Some(*n),
            PropertyPath::RangeLength(_, _, max) => *max,
            PropertyPath::Distinct(p) => p.max_length(),
        }
    }

    /// Collect all predicates mentioned in this path
    pub fn predicates(&self) -> HashSet<&NamedNode> {
        let mut predicates = HashSet::new();
        self.collect_predicates(&mut predicates);
        predicates
    }

    fn collect_predicates<'a>(&'a self, predicates: &mut HashSet<&'a NamedNode>) {
        match self {
            PropertyPath::Predicate(p) => {
                predicates.insert(p);
            }
            PropertyPath::Inverse(p) => p.collect_predicates(predicates),
            PropertyPath::Sequence(l, r) => {
                l.collect_predicates(predicates);
                r.collect_predicates(predicates);
            }
            PropertyPath::Alternative(l, r) => {
                l.collect_predicates(predicates);
                r.collect_predicates(predicates);
            }
            PropertyPath::ZeroOrMore(p)
            | PropertyPath::OneOrMore(p)
            | PropertyPath::ZeroOrOne(p)
            | PropertyPath::Distinct(p) => p.collect_predicates(predicates),
            PropertyPath::FixedLength(p, _) | PropertyPath::RangeLength(p, _, _) => {
                p.collect_predicates(predicates)
            }
            PropertyPath::NegatedPropertySet(ps) => {
                for p in ps {
                    predicates.insert(p);
                }
            }
        }
    }
}

impl fmt::Display for PropertyPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyPath::Predicate(p) => write!(f, "{}", p),
            PropertyPath::Inverse(p) => write!(f, "^{}", p),
            PropertyPath::Sequence(l, r) => write!(f, "{}/{}", l, r),
            PropertyPath::Alternative(l, r) => write!(f, "{}|{}", l, r),
            PropertyPath::ZeroOrMore(p) => write!(f, "{}*", p),
            PropertyPath::OneOrMore(p) => write!(f, "{}+", p),
            PropertyPath::ZeroOrOne(p) => write!(f, "{}?", p),
            PropertyPath::NegatedPropertySet(ps) => {
                write!(f, "!(")?;
                for (i, p) in ps.iter().enumerate() {
                    if i > 0 {
                        write!(f, "|")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ")")
            }
            PropertyPath::FixedLength(p, n) => write!(f, "{}{{{}}}", p, n),
            PropertyPath::RangeLength(p, min, max) => match max {
                Some(m) => write!(f, "{}{{{},{}}}", p, min, m),
                None => write!(f, "{}{{{},}}", p, min),
            },
            PropertyPath::Distinct(p) => write!(f, "DISTINCT({})", p),
        }
    }
}

/// Property path pattern for use in queries
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PropertyPathPattern {
    /// Subject of the path
    pub subject: TermPattern,
    /// The property path
    pub path: PropertyPath,
    /// Object of the path
    pub object: TermPattern,
}

impl PropertyPathPattern {
    /// Create a new property path pattern
    pub fn new(subject: TermPattern, path: PropertyPath, object: TermPattern) -> Self {
        PropertyPathPattern {
            subject,
            path,
            object,
        }
    }

    /// Convert a simple property path to a regular triple pattern
    pub fn to_triple_pattern(&self) -> Option<TriplePattern> {
        use crate::model::pattern::{ObjectPattern, PredicatePattern, SubjectPattern};

        match &self.path {
            PropertyPath::Predicate(p) => {
                let subject = match &self.subject {
                    TermPattern::Variable(v) => Some(SubjectPattern::Variable(v.clone())),
                    TermPattern::NamedNode(n) => Some(SubjectPattern::NamedNode(n.clone())),
                    TermPattern::BlankNode(b) => Some(SubjectPattern::BlankNode(b.clone())),
                    _ => None,
                };

                let predicate = Some(PredicatePattern::NamedNode(p.clone()));

                let object = match &self.object {
                    TermPattern::Variable(v) => Some(ObjectPattern::Variable(v.clone())),
                    TermPattern::NamedNode(n) => Some(ObjectPattern::NamedNode(n.clone())),
                    TermPattern::BlankNode(b) => Some(ObjectPattern::BlankNode(b.clone())),
                    TermPattern::Literal(l) => Some(ObjectPattern::Literal(l.clone())),
                    _ => None,
                };

                Some(TriplePattern {
                    subject,
                    predicate,
                    object,
                })
            }
            _ => None,
        }
    }

    /// Check if this pattern contains variables
    pub fn has_variables(&self) -> bool {
        self.subject.is_variable() || self.object.is_variable()
    }

    /// Get all variables in this pattern
    pub fn variables(&self) -> Vec<Variable> {
        let mut vars = Vec::new();
        if let TermPattern::Variable(v) = &self.subject {
            vars.push(v.clone());
        }
        if let TermPattern::Variable(v) = &self.object {
            vars.push(v.clone());
        }
        vars
    }
}

/// Property path evaluator
pub struct PropertyPathEvaluator {
    /// Maximum depth for recursive paths
    max_depth: usize,
    /// Enable cycle detection
    cycle_detection: bool,
    /// Enable distinct paths (SPARQL 1.2)
    distinct_paths: bool,
}

impl PropertyPathEvaluator {
    /// Create a new evaluator with default settings
    pub fn new() -> Self {
        PropertyPathEvaluator {
            max_depth: 100,
            cycle_detection: true,
            distinct_paths: false,
        }
    }

    /// Set maximum recursion depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Enable or disable cycle detection
    pub fn with_cycle_detection(mut self, enable: bool) -> Self {
        self.cycle_detection = enable;
        self
    }

    /// Enable distinct paths (SPARQL 1.2)
    pub fn with_distinct_paths(mut self, enable: bool) -> Self {
        self.distinct_paths = enable;
        self
    }

    /// Evaluate a property path pattern
    /// This is a placeholder - actual implementation would query the graph
    pub fn evaluate(&self, pattern: &PropertyPathPattern) -> Result<Vec<(Term, Term)>, OxirsError> {
        // Placeholder implementation
        Ok(Vec::new())
    }
}

/// Property path optimizer for query planning
pub struct PropertyPathOptimizer {
    /// Enable path rewriting
    rewrite_enabled: bool,
    /// Enable path decomposition
    decompose_enabled: bool,
}

impl PropertyPathOptimizer {
    /// Create new optimizer
    pub fn new() -> Self {
        PropertyPathOptimizer {
            rewrite_enabled: true,
            decompose_enabled: true,
        }
    }

    /// Optimize a property path
    pub fn optimize(&self, path: PropertyPath) -> PropertyPath {
        if !self.rewrite_enabled {
            return path;
        }

        // Apply optimization rules
        self.optimize_recursive(path)
    }

    fn optimize_recursive(&self, path: PropertyPath) -> PropertyPath {
        match path {
            // Optimize p/p to p{2}
            PropertyPath::Sequence(ref l, ref r) if l == r => {
                PropertyPath::FixedLength(l.clone(), 2)
            }

            // Optimize p? | p+ to p*
            PropertyPath::Alternative(ref l, ref r) => match (l.as_ref(), r.as_ref()) {
                (PropertyPath::ZeroOrOne(p1), PropertyPath::OneOrMore(p2)) if p1 == p2 => {
                    PropertyPath::ZeroOrMore(p1.clone())
                }
                (PropertyPath::OneOrMore(p1), PropertyPath::ZeroOrOne(p2)) if p1 == p2 => {
                    PropertyPath::ZeroOrMore(p1.clone())
                }
                _ => PropertyPath::Alternative(
                    Box::new(self.optimize_recursive(*l.clone())),
                    Box::new(self.optimize_recursive(*r.clone())),
                ),
            },

            // Recursively optimize nested paths
            PropertyPath::Inverse(p) => {
                PropertyPath::Inverse(Box::new(self.optimize_recursive(*p)))
            }
            PropertyPath::Sequence(l, r) => PropertyPath::Sequence(
                Box::new(self.optimize_recursive(*l)),
                Box::new(self.optimize_recursive(*r)),
            ),
            PropertyPath::ZeroOrMore(p) => {
                PropertyPath::ZeroOrMore(Box::new(self.optimize_recursive(*p)))
            }
            PropertyPath::OneOrMore(p) => {
                PropertyPath::OneOrMore(Box::new(self.optimize_recursive(*p)))
            }
            PropertyPath::ZeroOrOne(p) => {
                PropertyPath::ZeroOrOne(Box::new(self.optimize_recursive(*p)))
            }
            PropertyPath::FixedLength(p, n) => {
                PropertyPath::FixedLength(Box::new(self.optimize_recursive(*p)), n)
            }
            PropertyPath::RangeLength(p, min, max) => {
                PropertyPath::RangeLength(Box::new(self.optimize_recursive(*p)), min, max)
            }
            PropertyPath::Distinct(p) => {
                PropertyPath::Distinct(Box::new(self.optimize_recursive(*p)))
            }

            // Base cases
            PropertyPath::Predicate(_) | PropertyPath::NegatedPropertySet(_) => path,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_path_creation() {
        let p1 = NamedNode::new("http://example.org/knows").unwrap();
        let p2 = NamedNode::new("http://example.org/likes").unwrap();

        // Simple predicate
        let path = PropertyPath::predicate(p1.clone());
        assert_eq!(path.min_length(), 1);
        assert_eq!(path.max_length(), Some(1));

        // Sequence
        let seq = PropertyPath::sequence(
            PropertyPath::predicate(p1.clone()),
            PropertyPath::predicate(p2.clone()),
        );
        assert_eq!(seq.min_length(), 2);
        assert_eq!(seq.max_length(), Some(2));

        // Zero or more
        let star = PropertyPath::zero_or_more(PropertyPath::predicate(p1.clone()));
        assert_eq!(star.min_length(), 0);
        assert_eq!(star.max_length(), None);

        // Fixed length
        let fixed = PropertyPath::fixed_length(PropertyPath::predicate(p1.clone()), 3);
        assert_eq!(fixed.min_length(), 3);
        assert_eq!(fixed.max_length(), Some(3));
    }

    #[test]
    fn test_property_path_display() {
        let p1 = NamedNode::new("http://example.org/p").unwrap();
        let p2 = NamedNode::new("http://example.org/q").unwrap();

        let path = PropertyPath::sequence(
            PropertyPath::predicate(p1.clone()),
            PropertyPath::zero_or_more(PropertyPath::predicate(p2.clone())),
        );

        let expected = format!("{}/{}*", p1, p2);
        assert_eq!(format!("{}", path), expected);
    }

    #[test]
    fn test_path_optimization() {
        let optimizer = PropertyPathOptimizer::new();
        let p = PropertyPath::predicate(NamedNode::new("http://example.org/p").unwrap());

        // Optimize p/p to p{2}
        let seq = PropertyPath::sequence(p.clone(), p.clone());
        let optimized = optimizer.optimize(seq);
        assert!(matches!(optimized, PropertyPath::FixedLength(_, 2)));

        // Optimize p? | p+ to p*
        let alt = PropertyPath::alternative(
            PropertyPath::zero_or_one(p.clone()),
            PropertyPath::one_or_more(p.clone()),
        );
        let optimized = optimizer.optimize(alt);
        assert!(matches!(optimized, PropertyPath::ZeroOrMore(_)));
    }
}
