//! Algebra transformations: optimization rewrites, simplification, canonicalization.
//!
//! Provides a `PatternTransform` trait and concrete rewrite passes that can be
//! composed to form an optimization pipeline over SPARQL algebra trees.

use super::sparql_algebra_types::{Expression, GraphPattern};

// ────────────────────────────────────────────────────────────────────────────
// Visitor / transform trait
// ────────────────────────────────────────────────────────────────────────────

/// A depth-first mutable transform over `GraphPattern` trees.
///
/// Implementors override only the cases they care about.  The default
/// `transform` method recursively rebuilds the pattern bottom-up.
pub trait PatternTransform {
    /// Entry point: transform `pattern` and return the (possibly new) root.
    fn transform(&mut self, pattern: GraphPattern) -> GraphPattern {
        self.transform_pattern(pattern)
    }

    /// Called for each node after its children have been transformed.
    fn transform_pattern(&mut self, pattern: GraphPattern) -> GraphPattern {
        apply_default_transform(self, pattern)
    }

    /// Optional hook for expressions embedded in patterns.
    fn transform_expression(&mut self, expr: Expression) -> Expression {
        expr
    }
}

/// Apply the default bottom-up rebuild using `transform`.
pub fn apply_default_transform<T: PatternTransform + ?Sized>(
    t: &mut T,
    pattern: GraphPattern,
) -> GraphPattern {
    match pattern {
        GraphPattern::Bgp { .. } | GraphPattern::Path { .. } | GraphPattern::Values { .. } => {
            pattern
        }

        GraphPattern::Join { left, right } => {
            let left = t.transform(*left);
            let right = t.transform(*right);
            GraphPattern::Join {
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        GraphPattern::LeftJoin {
            left,
            right,
            expression,
        } => {
            let left = t.transform(*left);
            let right = t.transform(*right);
            let expression = expression.map(|e| t.transform_expression(e));
            GraphPattern::LeftJoin {
                left: Box::new(left),
                right: Box::new(right),
                expression,
            }
        }
        GraphPattern::Filter { expr, inner } => {
            let inner = t.transform(*inner);
            let expr = t.transform_expression(expr);
            GraphPattern::Filter {
                expr,
                inner: Box::new(inner),
            }
        }
        GraphPattern::Union { left, right } => {
            let left = t.transform(*left);
            let right = t.transform(*right);
            GraphPattern::Union {
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        GraphPattern::Graph { name, inner } => {
            let inner = t.transform(*inner);
            GraphPattern::Graph {
                name,
                inner: Box::new(inner),
            }
        }
        GraphPattern::Extend {
            inner,
            variable,
            expression,
        } => {
            let inner = t.transform(*inner);
            let expression = t.transform_expression(expression);
            GraphPattern::Extend {
                inner: Box::new(inner),
                variable,
                expression,
            }
        }
        GraphPattern::Minus { left, right } => {
            let left = t.transform(*left);
            let right = t.transform(*right);
            GraphPattern::Minus {
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        GraphPattern::OrderBy { inner, expression } => {
            let inner = t.transform(*inner);
            GraphPattern::OrderBy {
                inner: Box::new(inner),
                expression,
            }
        }
        GraphPattern::Project { inner, variables } => {
            let inner = t.transform(*inner);
            GraphPattern::Project {
                inner: Box::new(inner),
                variables,
            }
        }
        GraphPattern::Distinct { inner } => {
            let inner = t.transform(*inner);
            GraphPattern::Distinct {
                inner: Box::new(inner),
            }
        }
        GraphPattern::Reduced { inner } => {
            let inner = t.transform(*inner);
            GraphPattern::Reduced {
                inner: Box::new(inner),
            }
        }
        GraphPattern::Slice {
            inner,
            start,
            length,
        } => {
            let inner = t.transform(*inner);
            GraphPattern::Slice {
                inner: Box::new(inner),
                start,
                length,
            }
        }
        GraphPattern::Group {
            inner,
            variables,
            aggregates,
        } => {
            let inner = t.transform(*inner);
            GraphPattern::Group {
                inner: Box::new(inner),
                variables,
                aggregates,
            }
        }
        GraphPattern::Service {
            name,
            inner,
            silent,
        } => {
            let inner = t.transform(*inner);
            GraphPattern::Service {
                name,
                inner: Box::new(inner),
                silent,
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Simplification pass
// ────────────────────────────────────────────────────────────────────────────

/// Simplification rewrite pass.
///
/// Currently applies:
/// - Flatten nested `Join` chains that contain an empty BGP.
/// - Remove `Distinct` inside `Distinct`.
pub struct SimplifyTransform;

impl PatternTransform for SimplifyTransform {
    fn transform_pattern(&mut self, pattern: GraphPattern) -> GraphPattern {
        // First recursively transform children via the default visitor.
        let pattern = apply_default_transform(self, pattern);

        match pattern {
            // Distinct(Distinct(x)) => Distinct(x)
            GraphPattern::Distinct { inner } => {
                if let GraphPattern::Distinct { inner: inner2 } = *inner {
                    GraphPattern::Distinct { inner: inner2 }
                } else {
                    GraphPattern::Distinct { inner }
                }
            }
            // Join with an empty BGP on either side can be eliminated.
            GraphPattern::Join { left, right } => {
                let left_empty = matches!(
                    *left,
                    GraphPattern::Bgp { ref patterns } if patterns.is_empty()
                );
                let right_empty = matches!(
                    *right,
                    GraphPattern::Bgp { ref patterns } if patterns.is_empty()
                );
                if left_empty {
                    *right
                } else if right_empty {
                    *left
                } else {
                    GraphPattern::Join { left, right }
                }
            }
            other => other,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Canonicalization pass
// ────────────────────────────────────────────────────────────────────────────

/// Canonicalize a `GraphPattern` by applying all simplification rewrites.
///
/// Returns a semantically equivalent but syntactically normalized pattern.
pub fn canonicalize(pattern: GraphPattern) -> GraphPattern {
    SimplifyTransform.transform(pattern)
}
