//! Tests for SHACL constraint optimization (core engine + strategies)

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::constraints::{ClassConstraint, Constraint, DatatypeConstraint, MinCountConstraint};
    use crate::optimization::core_engine::ConstraintCache;
    use crate::optimization::core_strategies::ConstraintDependencyAnalyzer;
    use crate::{constraints::ConstraintContext, ShapeId};
    use oxirs_core::model::{NamedNode, Term};

    #[test]
    fn test_constraint_cache() {
        let cache = ConstraintCache::new(100, Duration::from_secs(60));

        let constraint = Constraint::Class(ClassConstraint {
            class_iri: NamedNode::new("http://example.org/Person").expect("valid IRI"),
        });

        let context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/john").expect("valid IRI")),
            ShapeId::new("PersonShape"),
        );

        // Should be cache miss initially
        assert!(cache.get(&constraint, &context).is_none());
    }

    #[test]
    fn test_constraint_ordering() {
        let analyzer = ConstraintDependencyAnalyzer::default();

        let constraints = vec![
            Constraint::Class(ClassConstraint {
                class_iri: NamedNode::new("http://example.org/Person").expect("valid IRI"),
            }),
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
            Constraint::Datatype(DatatypeConstraint {
                datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#string")
                    .expect("valid IRI"),
            }),
        ];

        let optimized = analyzer.optimize_constraint_order(constraints);

        // MinCount should come first (low selectivity), then datatype, then class
        assert!(matches!(optimized[0], Constraint::MinCount(_)));
    }
}
