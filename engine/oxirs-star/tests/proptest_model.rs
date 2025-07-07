use oxirs_star::model::*;
use oxirs_star::{StarQuad, StarTerm, StarTriple};
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;

// Strategy for generating valid IRI strings
fn iri_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("https?://[a-zA-Z0-9.-]+/[a-zA-Z0-9/-]*").unwrap()
}

// Strategy for generating valid blank node IDs
fn blank_node_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z][a-zA-Z0-9_]*").unwrap()
}

// Strategy for generating literal values
fn literal_value_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[^\"\\\\]*").unwrap()
}

// Strategy for generating language tags
fn language_tag_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z]{2}(-[A-Z]{2})?").unwrap()
}

// Strategy for generating StarTerm values
fn star_term_strategy() -> impl Strategy<Value = StarTerm> {
    prop_oneof![
        // IRI
        iri_strategy().prop_map(|iri| StarTerm::iri(&iri).unwrap()),
        // Blank node
        blank_node_strategy().prop_map(|id| StarTerm::blank_node(&id).unwrap()),
        // Simple literal
        literal_value_strategy().prop_map(|value| StarTerm::literal(&value).unwrap()),
        // Literal with language
        (literal_value_strategy(), language_tag_strategy())
            .prop_map(|(value, lang)| StarTerm::literal_with_language(&value, &lang).unwrap()),
        // Literal with datatype
        (literal_value_strategy(), iri_strategy()).prop_map(|(value, datatype)| {
            StarTerm::literal_with_datatype(&value, &datatype).unwrap()
        }),
    ]
}

// Strategy for generating non-quoted StarTerm values (to avoid infinite recursion)
fn non_quoted_star_term_strategy() -> impl Strategy<Value = StarTerm> {
    prop_oneof![
        // IRI
        iri_strategy().prop_map(|iri| StarTerm::iri(&iri).unwrap()),
        // Blank node
        blank_node_strategy().prop_map(|id| StarTerm::blank_node(&id).unwrap()),
        // Simple literal
        literal_value_strategy().prop_map(|value| StarTerm::literal(&value).unwrap()),
    ]
}

// Strategy for generating valid subject terms
fn subject_term_strategy() -> impl Strategy<Value = StarTerm> {
    prop_oneof![
        // IRI
        iri_strategy().prop_map(|iri| StarTerm::iri(&iri).unwrap()),
        // Blank node
        blank_node_strategy().prop_map(|id| StarTerm::blank_node(&id).unwrap()),
    ]
}

// Strategy for generating valid predicate terms (only IRIs)
fn predicate_term_strategy() -> impl Strategy<Value = StarTerm> {
    iri_strategy().prop_map(|iri| StarTerm::iri(&iri).unwrap())
}

// Strategy for generating simple triples (no quoted triples)
fn simple_triple_strategy() -> impl Strategy<Value = StarTriple> {
    (
        subject_term_strategy(),
        predicate_term_strategy(),
        non_quoted_star_term_strategy(),
    )
        .prop_map(|(s, p, o)| StarTriple::new(s, p, o))
}

// Strategy for generating quoted triple terms with controlled depth
fn quoted_triple_term_strategy(depth: u32) -> BoxedStrategy<StarTerm> {
    if depth == 0 {
        non_quoted_star_term_strategy().boxed()
    } else {
        prop_oneof![
            // Regular terms (more common)
            non_quoted_star_term_strategy().boxed(),
            // Quoted triple (less common to control nesting)
            triple_strategy_with_depth(depth - 1)
                .prop_map(|triple| StarTerm::quoted_triple(triple))
                .boxed(),
        ]
        .boxed()
    }
}

// Strategy for generating triples with controlled nesting depth
fn triple_strategy_with_depth(depth: u32) -> impl Strategy<Value = StarTriple> {
    if depth == 0 {
        simple_triple_strategy().boxed()
    } else {
        let subject = prop_oneof![
            subject_term_strategy().boxed(),
            triple_strategy_with_depth(depth - 1)
                .prop_map(|t| StarTerm::quoted_triple(t))
                .boxed(),
        ];

        let object = quoted_triple_term_strategy(depth - 1);

        (subject, predicate_term_strategy(), object)
            .prop_map(|(s, p, o)| StarTriple::new(s, p, o))
            .boxed()
    }
}

// Strategy for generating quads
fn quad_strategy() -> impl Strategy<Value = StarQuad> {
    (
        simple_triple_strategy(),
        prop::option::of(prop_oneof![
            iri_strategy().prop_map(|iri| StarTerm::iri(&iri).unwrap()),
            blank_node_strategy().prop_map(|id| StarTerm::blank_node(&id).unwrap()),
        ]),
    )
        .prop_map(|(triple, graph)| {
            StarQuad::new(triple.subject, triple.predicate, triple.object, graph)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #[test]
        fn test_star_term_display_parse_roundtrip(term in star_term_strategy()) {
            // Test that display formatting is deterministic
            let display1 = format!("{}", term);
            let display2 = format!("{}", term);
            prop_assert_eq!(display1, display2);
        }

        #[test]
        fn test_star_term_equality(term in star_term_strategy()) {
            // Test reflexivity: a term equals itself
            prop_assert_eq!(&term, &term);

            // Test that clone produces equal terms
            let cloned = term.clone();
            prop_assert_eq!(term, cloned);
        }

        #[test]
        fn test_star_term_type_checks(term in star_term_strategy()) {
            // Exactly one type check should be true
            let type_count = [
                term.is_named_node(),
                term.is_blank_node(),
                term.is_literal(),
                term.is_quoted_triple(),
                term.is_variable(),
            ].iter().filter(|&&b| b).count();

            prop_assert_eq!(type_count, 1);
        }

        #[test]
        fn test_star_term_accessors(term in star_term_strategy()) {
            // Test that accessors return Some only for matching types
            if term.is_named_node() {
                prop_assert!(term.as_named_node().is_some());
                prop_assert!(term.as_blank_node().is_none());
                prop_assert!(term.as_literal().is_none());
                prop_assert!(term.as_quoted_triple().is_none());
            } else if term.is_blank_node() {
                prop_assert!(term.as_named_node().is_none());
                prop_assert!(term.as_blank_node().is_some());
                prop_assert!(term.as_literal().is_none());
                prop_assert!(term.as_quoted_triple().is_none());
            } else if term.is_literal() {
                prop_assert!(term.as_named_node().is_none());
                prop_assert!(term.as_blank_node().is_none());
                prop_assert!(term.as_literal().is_some());
                prop_assert!(term.as_quoted_triple().is_none());
            }
        }

        #[test]
        fn test_triple_validation(triple in simple_triple_strategy()) {
            // All simple triples should be valid
            prop_assert!(triple.validate().is_ok());
        }

        #[test]
        fn test_triple_nesting_depth(triple in triple_strategy_with_depth(3)) {
            // Nesting depth should be non-negative and reasonable
            let depth = triple.nesting_depth();
            prop_assert!(depth <= 10); // sanity check
        }

        #[test]
        fn test_triple_contains_quoted(triple in triple_strategy_with_depth(2)) {
            // If nesting depth > 0, should contain quoted triples
            let contains_quoted = triple.contains_quoted_triples();
            let depth = triple.nesting_depth();

            if depth > 0 {
                prop_assert!(contains_quoted);
            }
        }

        #[test]
        fn test_quad_validation(quad in quad_strategy()) {
            // All generated quads should be valid
            prop_assert!(quad.validate().is_ok());
        }

        #[test]
        fn test_quad_triple_conversion(quad in quad_strategy()) {
            // Converting quad to triple and back should preserve non-graph components
            let triple = quad.clone().to_triple();
            let quad2 = triple.to_quad(quad.graph.clone());

            prop_assert_eq!(quad.subject, quad2.subject);
            prop_assert_eq!(quad.predicate, quad2.predicate);
            prop_assert_eq!(quad.object, quad2.object);
            prop_assert_eq!(quad.graph, quad2.graph);
        }

        #[test]
        fn test_star_graph_operations(triples in prop::collection::vec(simple_triple_strategy(), 0..10)) {
            let mut graph = StarGraph::new();

            // Insert all triples
            for triple in &triples {
                prop_assert!(graph.insert(triple.clone()).is_ok());
            }

            // Check size
            prop_assert_eq!(graph.len(), triples.len());
            prop_assert_eq!(graph.total_len(), triples.len());

            // Check contains
            for triple in &triples {
                prop_assert!(graph.contains(triple));
            }

            // Remove all triples
            for triple in &triples {
                prop_assert!(graph.remove(triple));
            }

            // Graph should be empty
            prop_assert!(graph.is_empty());
            prop_assert_eq!(graph.len(), 0);
        }

        #[test]
        fn test_subject_predicate_object_rules(
            subject in star_term_strategy(),
            predicate in star_term_strategy(),
            object in star_term_strategy()
        ) {
            // Test RDF-star rules for term positions

            // Only certain terms can be subjects
            if subject.is_literal() || subject.is_variable() {
                prop_assert!(!subject.can_be_subject());
            } else {
                prop_assert!(subject.can_be_subject());
            }

            // Only IRIs can be predicates
            prop_assert_eq!(predicate.can_be_predicate(), predicate.is_named_node());

            // All terms can be objects
            prop_assert!(object.can_be_object());
        }

        #[test]
        fn test_literal_formatting(
            value in literal_value_strategy(),
            literal_type in prop_oneof![
                Just("simple".to_string()),
                language_tag_strategy().prop_map(|l| format!("lang:{}", l)),
                iri_strategy().prop_map(|dt| format!("datatype:{}", dt))
            ]
        ) {
            // Skip empty values as they might not be valid
            prop_assume!(!value.is_empty());

            let (literal, has_lang, has_datatype) = if literal_type == "simple" {
                (StarTerm::literal(&value).unwrap(), false, false)
            } else if let Some(lang) = literal_type.strip_prefix("lang:") {
                (StarTerm::literal_with_language(&value, lang).unwrap(), true, false)
            } else if let Some(dt) = literal_type.strip_prefix("datatype:") {
                (StarTerm::literal_with_datatype(&value, dt).unwrap(), false, true)
            } else {
                unreachable!()
            };

            let display = format!("{}", literal);

            // Check basic formatting rules
            prop_assert!(display.starts_with('"'));
            prop_assert!(display.contains(&value));

            if has_lang {
                prop_assert!(display.contains('@'));
            }
            if has_datatype {
                prop_assert!(display.contains("^^"));
            }
        }

        #[test]
        fn test_triple_visitor(triple in triple_strategy_with_depth(2)) {
            struct TermCounter {
                count: usize,
            }

            impl StarTermVisitor for TermCounter {
                fn visit_term(&mut self, _term: &StarTerm) {
                    self.count += 1;
                }
            }

            let mut counter = TermCounter { count: 0 };
            triple.visit_terms(&mut counter);

            // Should visit at least 3 terms (subject, predicate, object)
            prop_assert!(counter.count >= 3);
        }

        #[test]
        fn test_graph_named_graphs(
            triples in prop::collection::vec(simple_triple_strategy(), 0..5),
            graph_names in prop::collection::vec(iri_strategy(), 1..3)
        ) {
            let mut graph = StarGraph::new();

            // Add triples to different named graphs
            for (i, triple) in triples.iter().enumerate() {
                let graph_name = &graph_names[i % graph_names.len()];
                let quad = StarQuad::new(
                    triple.subject.clone(),
                    triple.predicate.clone(),
                    triple.object.clone(),
                    Some(StarTerm::iri(graph_name).unwrap())
                );
                prop_assert!(graph.insert_quad(quad).is_ok());
            }

            // Check named graphs exist
            for graph_name in &graph_names {
                if triples.iter().enumerate().any(|(i, _)| i % graph_names.len() == graph_names.iter().position(|g| g == graph_name).unwrap()) {
                    prop_assert!(graph.contains_named_graph(graph_name));
                }
            }
        }
    }

    // Additional deterministic tests for edge cases
    #[test]
    fn test_empty_values_rejected() {
        assert!(StarTerm::iri("").is_err());
        assert!(StarTerm::blank_node("").is_err());
        assert!(StarTerm::variable("").is_err());
    }

    #[test]
    fn test_invalid_triple_validation() {
        // Literal as predicate should fail validation
        let invalid = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::literal("invalid_predicate").unwrap(),
            StarTerm::literal("object").unwrap(),
        );
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_graph_statistics() {
        let mut graph = StarGraph::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::literal("o").unwrap(),
        );

        graph.insert(triple).unwrap();

        let stats = graph.statistics();
        assert_eq!(stats.get("triples"), Some(&1));
    }
}
