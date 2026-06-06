//! Unit tests for the SPARQL algebra module.

#[cfg(test)]
mod tests {
    use crate::model::{NamedNode, Variable};
    use crate::query::sparql_algebra::{
        BuiltInFunction, GraphPattern, PropertyPathExpression, TermPattern, TriplePattern,
    };
    use crate::query::sparql_algebra_types::Expression;

    #[test]
    fn test_property_path_display() {
        let p1 = NamedNode::new("http://example.org/p1").expect("valid IRI");
        let p2 = NamedNode::new("http://example.org/p2").expect("valid IRI");

        let path = PropertyPathExpression::Sequence(
            Box::new(PropertyPathExpression::NamedNode(p1)),
            Box::new(PropertyPathExpression::NamedNode(p2)),
        );

        assert!(path.to_string().contains("/"));
    }

    #[test]
    fn test_basic_graph_pattern() {
        let subject = TermPattern::Variable(Variable::new("s").expect("valid variable name"));
        let predicate = TermPattern::Variable(Variable::new("p").expect("valid variable name"));
        let object = TermPattern::Variable(Variable::new("o").expect("valid variable name"));

        let triple = TriplePattern::new(subject, predicate, object);
        let bgp = GraphPattern::Bgp {
            patterns: vec![triple],
        };

        let mut sse = String::new();
        bgp.fmt_sse(&mut sse).expect("operation should succeed");
        assert!(sse.contains("bgp"));
        assert!(sse.contains("?s"));
        assert!(sse.contains("?p"));
        assert!(sse.contains("?o"));
    }

    #[test]
    fn test_expression_formatting() {
        let var1 = Expression::Variable(Variable::new("x").expect("valid variable name"));
        let var2 = Expression::Variable(Variable::new("y").expect("valid variable name"));
        let expr = Expression::Add(Box::new(var1), Box::new(var2));

        let mut sse = String::new();
        expr.fmt_sse(&mut sse).expect("operation should succeed");
        assert!(sse.contains("+ ?x ?y"));
    }

    #[test]
    fn test_built_in_function() {
        let func = BuiltInFunction::Str;
        assert_eq!(func.to_string(), "STR");

        let mut sse = String::new();
        func.fmt_sse(&mut sse).expect("operation should succeed");
        assert_eq!(sse, "str");
    }
}
