use oxirs_star::model::{StarTerm, StarTriple};
use oxirs_star::store::StarStore;
use proptest::prelude::*;
use std::collections::HashSet;

// Generate simple triples for store testing
fn simple_triple_strategy() -> impl Strategy<Value = StarTriple> {
    (
        prop::string::string_regex("http://example\\.org/[a-z]+[0-9]*").unwrap(),
        prop::string::string_regex("http://example\\.org/[a-z]+").unwrap(),
        prop_oneof![
            prop::string::string_regex("http://example\\.org/[a-z]+[0-9]*")
                .unwrap()
                .prop_map(|s| (s, None::<String>, None::<String>)),
            prop::string::string_regex("[a-zA-Z0-9]+")
                .unwrap()
                .prop_map(|s| (s, None::<String>, None::<String>)),
            (
                prop::string::string_regex("[a-zA-Z0-9]+").unwrap(),
                prop::string::string_regex("[a-z]{2}").unwrap()
            )
                .prop_map(|(val, lang)| (val, Some(lang), None::<String>)),
        ],
    )
        .prop_map(|(subj, pred, (obj_val, lang, dt))| {
            let subject = StarTerm::iri(&subj).unwrap();
            let predicate = StarTerm::iri(&pred).unwrap();
            let object = if let Some(l) = lang {
                StarTerm::literal_with_language(&obj_val, &l).unwrap()
            } else if obj_val.starts_with("http") {
                StarTerm::iri(&obj_val).unwrap()
            } else {
                StarTerm::literal(&obj_val).unwrap()
            };

            StarTriple::new(subject, predicate, object)
        })
}

// Generate quoted triples with controlled depth
fn quoted_triple_strategy() -> impl Strategy<Value = StarTriple> {
    simple_triple_strategy().prop_flat_map(|inner| {
        (
            Just(inner),
            prop::string::string_regex("http://example\\.org/meta/[a-z]+").unwrap(),
            prop::string::string_regex("[0-9.]+").unwrap(),
        )
            .prop_map(|(inner_triple, pred, obj)| {
                StarTriple::new(
                    StarTerm::quoted_triple(inner_triple),
                    StarTerm::iri(&pred).unwrap(),
                    StarTerm::literal(&obj).unwrap(),
                )
            })
    })
}

// Generate a mix of simple and quoted triples
fn mixed_triple_strategy() -> impl Strategy<Value = StarTriple> {
    prop_oneof![
        simple_triple_strategy().boxed(),
        quoted_triple_strategy().boxed(),
    ]
}

// Generate query patterns (with None for wildcards)
fn query_pattern_strategy(
) -> impl Strategy<Value = (Option<StarTerm>, Option<StarTerm>, Option<StarTerm>)> {
    (
        prop::option::of(
            prop::string::string_regex("http://example\\.org/[a-z]+[0-9]*")
                .unwrap()
                .prop_map(|s| StarTerm::iri(&s).unwrap()),
        ),
        prop::option::of(
            prop::string::string_regex("http://example\\.org/[a-z]+")
                .unwrap()
                .prop_map(|s| StarTerm::iri(&s).unwrap()),
        ),
        prop::option::of(prop_oneof![
            prop::string::string_regex("http://example\\.org/[a-z]+[0-9]*")
                .unwrap()
                .prop_map(|s| StarTerm::iri(&s).unwrap()),
            prop::string::string_regex("[a-zA-Z0-9]+")
                .unwrap()
                .prop_map(|s| StarTerm::literal(&s).unwrap()),
        ]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #[test]
        fn test_store_insert_remove_contains(
            triples in prop::collection::vec(simple_triple_strategy(), 0..20)
        ) {
            let store = StarStore::new();
            let unique_triples: Vec<_> = triples.into_iter()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();

            // Insert all triples
            for triple in &unique_triples {
                prop_assert!(store.insert(triple).is_ok());
                prop_assert!(store.contains(triple));
            }

            // Check size
            prop_assert_eq!(store.len(), unique_triples.len());

            // Remove half the triples
            let to_remove = unique_triples.len() / 2;
            for triple in unique_triples.iter().take(to_remove) {
                prop_assert!(store.remove(triple).unwrap());
                prop_assert!(!store.contains(triple));
            }

            // Check remaining size
            prop_assert_eq!(store.len(), unique_triples.len() - to_remove);

            // Remaining triples should still be there
            for triple in unique_triples.iter().skip(to_remove) {
                prop_assert!(store.contains(triple));
            }
        }

        #[test]
        fn test_store_query_patterns(
            triples in prop::collection::vec(simple_triple_strategy(), 0..10),
            pattern in query_pattern_strategy()
        ) {
            let store = StarStore::new();

            // Insert all triples
            for triple in &triples {
                store.insert(triple).unwrap();
            }

            // Query with pattern
            let (s_pattern, p_pattern, o_pattern) = pattern;
            let results = store.query_triples(
                s_pattern.as_ref(),
                p_pattern.as_ref(),
                o_pattern.as_ref()
            ).unwrap();

            // All results should match the pattern
            for result in &results {
                if let Some(ref s) = s_pattern {
                    prop_assert_eq!(&result.subject, s);
                }
                if let Some(ref p) = p_pattern {
                    prop_assert_eq!(&result.predicate, p);
                }
                if let Some(ref o) = o_pattern {
                    prop_assert_eq!(&result.object, o);
                }
            }

            // Results should be a subset of inserted triples
            prop_assert!(results.len() <= triples.len());
        }

        #[test]
        fn test_store_quoted_triple_operations(
            quoted_triples in prop::collection::vec(quoted_triple_strategy(), 1..10)
        ) {
            let store = StarStore::new();
            let unique_quoted: Vec<_> = quoted_triples.into_iter()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();

            // Insert quoted triples
            for triple in &unique_quoted {
                prop_assert!(store.insert(triple).is_ok());
            }

            // Find triples containing specific quoted patterns
            for triple in &unique_quoted {
                if let StarTerm::QuotedTriple(inner) = &triple.subject {
                    let containing = store.find_triples_containing_quoted(inner);
                    prop_assert!(containing.contains(triple));
                }
            }

            // Check statistics
            let stats = store.statistics();
            prop_assert_eq!(stats.quoted_triples_count, unique_quoted.len());
            if !unique_quoted.is_empty() {
                prop_assert!(stats.max_nesting_encountered >= 1);
            }
        }

        #[test]
        fn test_store_clear_operations(
            triples1 in prop::collection::vec(simple_triple_strategy(), 0..10),
            triples2 in prop::collection::vec(simple_triple_strategy(), 0..10)
        ) {
            let store = StarStore::new();

            // Insert first batch
            for triple in &triples1 {
                store.insert(triple).unwrap();
            }
            prop_assert_eq!(store.len(), triples1.len());

            // Clear store
            prop_assert!(store.clear().is_ok());
            prop_assert!(store.is_empty());
            prop_assert_eq!(store.len(), 0);

            // Insert second batch
            for triple in &triples2 {
                store.insert(triple).unwrap();
            }
            prop_assert_eq!(store.len(), triples2.len());
        }

        #[test]
        fn test_store_graph_conversion(
            triples in prop::collection::vec(mixed_triple_strategy(), 0..15)
        ) {
            let store = StarStore::new();
            let unique_triples: Vec<_> = triples.into_iter()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();

            // Insert triples
            for triple in &unique_triples {
                store.insert(triple).unwrap();
            }

            // Convert to graph
            let graph = store.to_graph();
            prop_assert_eq!(graph.len(), unique_triples.len());

            // All triples should be in the graph
            for triple in &unique_triples {
                prop_assert!(graph.contains(triple));
            }

            // Create new store from graph
            let store2 = StarStore::new();
            prop_assert!(store2.from_graph(&graph).is_ok());
            prop_assert_eq!(store2.len(), store.len());

            // Both stores should contain same triples
            for triple in &unique_triples {
                prop_assert!(store2.contains(triple));
            }
        }

        #[test]
        fn test_store_nesting_depth_queries(
            simple in prop::collection::vec(simple_triple_strategy(), 0..5),
            quoted in prop::collection::vec(quoted_triple_strategy(), 0..5)
        ) {
            let store = StarStore::new();

            // Insert simple triples (depth 0)
            for triple in &simple {
                store.insert(triple).unwrap();
            }

            // Insert quoted triples (depth >= 1)
            for triple in &quoted {
                store.insert(triple).unwrap();
            }

            // Query by nesting depth
            let depth_0 = store.find_triples_by_nesting_depth(0, Some(0));
            let depth_1_plus = store.find_triples_by_nesting_depth(1, None);

            // Simple triples have depth 0 (unless they contain quoted triples)
            let simple_without_quoted: Vec<_> = simple.iter()
                .filter(|t| !t.contains_quoted_triples())
                .collect();
            prop_assert_eq!(depth_0.len(), simple_without_quoted.len());

            // Quoted triples have depth >= 1
            prop_assert!(depth_1_plus.len() >= quoted.len());
        }

        #[test]
        fn test_store_concurrent_safety(
            triples in prop::collection::vec(simple_triple_strategy(), 0..20)
        ) {
            use std::sync::Arc;
            use std::thread;

            let store = Arc::new(StarStore::new());
            let unique_triples: Vec<_> = triples.into_iter()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();

            // Split triples for concurrent insertion
            let mid = unique_triples.len() / 2;
            let (first_half, second_half) = unique_triples.split_at(mid);

            let store1 = Arc::clone(&store);
            let triples1 = first_half.to_vec();
            let handle1 = thread::spawn(move || {
                for triple in triples1 {
                    store1.insert(&triple).unwrap();
                }
            });

            let store2 = Arc::clone(&store);
            let triples2 = second_half.to_vec();
            let handle2 = thread::spawn(move || {
                for triple in triples2 {
                    store2.insert(&triple).unwrap();
                }
            });

            handle1.join().unwrap();
            handle2.join().unwrap();

            // All triples should be in the store
            prop_assert_eq!(store.len(), unique_triples.len());
            for triple in &unique_triples {
                prop_assert!(store.contains(triple));
            }
        }

        #[test]
        fn test_store_optimize(
            triples in prop::collection::vec(mixed_triple_strategy(), 0..20)
        ) {
            let store = StarStore::new();

            // Insert triples
            for triple in &triples {
                store.insert(triple).unwrap();
            }

            let size_before = store.len();

            // Optimize should not change content
            prop_assert!(store.optimize().is_ok());
            prop_assert_eq!(store.len(), size_before);

            // All triples should still be there
            for triple in &triples {
                prop_assert!(store.contains(triple));
            }
        }

        #[test]
        fn test_store_iterator_safety(
            triples in prop::collection::vec(simple_triple_strategy(), 1..10)
        ) {
            let store = StarStore::new();

            // Insert triples
            for triple in &triples {
                store.insert(triple).unwrap();
            }

            // Iterate and collect
            let collected: Vec<_> = store.iter().collect();
            prop_assert_eq!(collected.len(), triples.len());

            // All inserted triples should be in the iteration
            for triple in &triples {
                prop_assert!(collected.contains(triple));
            }
        }
    }

    // Additional deterministic tests
    #[test]
    fn test_deeply_nested_quoted_triples() {
        let store = StarStore::new();

        // Create deeply nested structure
        let mut current = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::literal("o").unwrap(),
        );

        for i in 0..5 {
            current = StarTriple::new(
                StarTerm::quoted_triple(current),
                StarTerm::iri(&format!("http://example.org/level{}", i)).unwrap(),
                StarTerm::literal(&format!("{}", i)).unwrap(),
            );
        }

        assert!(store.insert(&current).is_ok());
        assert_eq!(store.len(), 1);

        let stats = store.statistics();
        assert_eq!(stats.quoted_triples_count, 1);
        assert!(stats.max_nesting_encountered >= 5);
    }

    #[test]
    fn test_quoted_pattern_matching() {
        let store = StarStore::new();

        // Create a quoted triple
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        store.insert(&outer).unwrap();

        // Find by quoted pattern
        let results = store.find_triples_by_quoted_pattern(
            Some(&StarTerm::iri("http://example.org/alice").unwrap()),
            None,
            None,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], outer);
    }
}
