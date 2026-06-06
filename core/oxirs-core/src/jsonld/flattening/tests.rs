//! Tests for the JSON-LD 1.1 Flattening algorithm.
//!
//! Covers all major scenarios of the W3C JSON-LD 1.1 Flattening specification:
//! <https://www.w3.org/TR/json-ld11-api/#flattening-algorithm>

#[cfg(test)]
mod flattening_tests {
    use crate::jsonld::flattening::{
        flatten, generate_node_map, is_list_object, is_node_object, is_value_object,
        BlankNodeIdMapper, FlatteningOptions, JsonLdValue,
    };
    use indexmap::IndexMap;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn obj(pairs: &[(&str, JsonLdValue)]) -> JsonLdValue {
        let mut m: IndexMap<String, JsonLdValue> = IndexMap::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        JsonLdValue::Object(m)
    }

    fn arr(items: Vec<JsonLdValue>) -> JsonLdValue {
        JsonLdValue::Array(items)
    }

    fn s(v: &str) -> JsonLdValue {
        JsonLdValue::Str(v.to_string())
    }

    fn default_opts() -> FlatteningOptions {
        FlatteningOptions::default()
    }

    fn get_graph(result: &JsonLdValue) -> &[JsonLdValue] {
        let m = result.as_object().expect("result should be object");
        m.get("@graph")
            .and_then(|v| v.as_array())
            .expect("result should have @graph array")
    }

    // -----------------------------------------------------------------------
    // Test 1: flatten a simple triple
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_simple_triple() {
        let input = arr(vec![obj(&[
            ("@id", s("http://example.org/subject")),
            ("@type", arr(vec![s("http://example.org/Class")])),
        ])]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        assert_eq!(graph.len(), 1, "one subject in output");
        let node = graph[0].as_object().expect("node is object");
        assert_eq!(
            node.get("@id").and_then(|v| v.as_str()),
            Some("http://example.org/subject"),
            "@id preserved"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: flatten nested object → two top-level entries
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_nested_object() {
        // subject has a property whose value is a nested node object.
        let nested = obj(&[
            ("@id", s("http://example.org/B")),
            ("@type", arr(vec![s("http://example.org/Thing")])),
        ]);
        let prop_val = arr(vec![nested]);
        let input = arr(vec![obj(&[
            ("@id", s("http://example.org/A")),
            ("http://example.org/link", prop_val),
        ])]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        // Both A and B should appear at the top level.
        let ids: Vec<&str> = graph
            .iter()
            .filter_map(|n| n.as_object())
            .filter_map(|m| m.get("@id"))
            .filter_map(|v| v.as_str())
            .collect();

        assert!(ids.contains(&"http://example.org/A"), "A present");
        assert!(ids.contains(&"http://example.org/B"), "B present");
    }

    // -----------------------------------------------------------------------
    // Test 3: blank node renaming
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_blank_node_renaming() {
        let input = arr(vec![obj(&[
            ("@id", s("_:b123")),
            ("@type", arr(vec![s("http://example.org/Class")])),
        ])]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        assert_eq!(graph.len(), 1);
        let id = graph[0]
            .as_object()
            .and_then(|m| m.get("@id"))
            .and_then(|v| v.as_str())
            .expect("@id present");

        assert!(
            id.starts_with("_:b"),
            "blank node renamed to canonical form"
        );
        // The canonical form should start with _:b0
        assert_eq!(id, "_:b0", "first blank node becomes _:b0");
    }

    // -----------------------------------------------------------------------
    // Test 4: blank node consistency — same original ID → same new ID
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_blank_node_consistent() {
        // Two node objects reference the same blank node.
        let ref_node = obj(&[("@id", s("_:orig"))]);
        let input = arr(vec![
            obj(&[
                ("@id", s("http://example.org/A")),
                ("http://example.org/link", arr(vec![ref_node.clone()])),
            ]),
            obj(&[
                ("@id", s("http://example.org/B")),
                ("http://example.org/link", arr(vec![ref_node])),
            ]),
        ]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        // Find all @id values in link properties.
        let mut blank_ids: Vec<String> = Vec::new();
        for node in graph {
            if let Some(m) = node.as_object() {
                if let Some(arr_val) = m.get("http://example.org/link").and_then(|v| v.as_array()) {
                    for item in arr_val {
                        if let Some(inner) = item.as_object() {
                            if let Some(id) = inner.get("@id").and_then(|v| v.as_str()) {
                                if id.starts_with("_:") {
                                    blank_ids.push(id.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        // All references to _:orig must have the same renamed ID.
        assert!(
            blank_ids.len() >= 2,
            "expected at least two references to blank node"
        );
        let first = &blank_ids[0];
        for id in &blank_ids {
            assert_eq!(
                id, first,
                "all references to same blank node have same canonical ID"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: all properties preserved
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_preserves_all_properties() {
        let input = arr(vec![obj(&[
            ("@id", s("http://example.org/S")),
            (
                "http://example.org/name",
                arr(vec![obj(&[("@value", s("Alice"))])]),
            ),
            (
                "http://example.org/age",
                arr(vec![obj(&[
                    ("@value", s("30")),
                    ("@type", s("http://www.w3.org/2001/XMLSchema#integer")),
                ])]),
            ),
        ])]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        assert_eq!(graph.len(), 1);
        let m = graph[0].as_object().expect("object");
        assert!(
            m.contains_key("http://example.org/name"),
            "name property present"
        );
        assert!(
            m.contains_key("http://example.org/age"),
            "age property present"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: list value preservation
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_list_value() {
        let list_val = obj(&[(
            "@list",
            arr(vec![obj(&[("@value", s("a"))]), obj(&[("@value", s("b"))])]),
        )]);

        let input = arr(vec![obj(&[
            ("@id", s("http://example.org/S")),
            ("http://example.org/items", arr(vec![list_val])),
        ])]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        assert_eq!(graph.len(), 1);
        let m = graph[0].as_object().expect("object");
        // The items property should contain the list.
        let items = m
            .get("http://example.org/items")
            .and_then(|v| v.as_array())
            .expect("items array");
        assert!(!items.is_empty(), "items not empty");
        // Find the @list object in items.
        let has_list = items.iter().any(is_list_object);
        assert!(has_list, "list object preserved");
    }

    // -----------------------------------------------------------------------
    // Test 7: @type always as array
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_type_coercion() {
        let input = arr(vec![obj(&[
            ("@id", s("http://example.org/S")),
            ("@type", arr(vec![s("http://example.org/Person")])),
        ])]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        assert_eq!(graph.len(), 1);
        let m = graph[0].as_object().expect("object");
        // @type in flat output should be an array.
        let types = m
            .get("@type")
            .and_then(|v| v.as_array())
            .expect("@type should be array");
        assert_eq!(types.len(), 1);
        assert_eq!(
            types[0],
            JsonLdValue::Str("http://example.org/Person".to_string())
        );
    }

    // -----------------------------------------------------------------------
    // Test 8: ordered output
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_ordered_nodes() {
        let input = arr(vec![
            obj(&[("@id", s("http://example.org/Z"))]),
            obj(&[("@id", s("http://example.org/A"))]),
            obj(&[("@id", s("http://example.org/M"))]),
        ]);

        let mut opts = default_opts();
        opts.ordered = true;
        let result = flatten(&input, None, &opts).expect("flatten ok");
        let graph = get_graph(&result);

        let ids: Vec<&str> = graph
            .iter()
            .filter_map(|n| n.as_object())
            .filter_map(|m| m.get("@id"))
            .filter_map(|v| v.as_str())
            .collect();

        assert_eq!(
            ids,
            vec![
                "http://example.org/A",
                "http://example.org/M",
                "http://example.org/Z",
            ],
            "nodes sorted lexicographically when ordered=true"
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: unordered is fine
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_unordered_nodes() {
        let input = arr(vec![
            obj(&[("@id", s("http://example.org/B"))]),
            obj(&[("@id", s("http://example.org/A"))]),
        ]);
        let mut opts = default_opts();
        opts.ordered = false;
        let result = flatten(&input, None, &opts).expect("flatten ok");
        let graph = get_graph(&result);
        assert_eq!(graph.len(), 2, "both nodes present");
    }

    // -----------------------------------------------------------------------
    // Test 10: named graph content appears as nested @graph
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_named_graph() {
        let inner_node = obj(&[
            ("@id", s("http://example.org/Inner")),
            ("@type", arr(vec![s("http://example.org/Thing")])),
        ]);
        let input = arr(vec![obj(&[
            ("@id", s("http://example.org/Graph")),
            ("@graph", arr(vec![inner_node])),
        ])]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        // The graph node itself should appear at top level.
        let ids: Vec<&str> = graph
            .iter()
            .filter_map(|n| n.as_object())
            .filter_map(|m| m.get("@id"))
            .filter_map(|v| v.as_str())
            .collect();
        assert!(
            ids.contains(&"http://example.org/Graph"),
            "graph node present at top level"
        );
        // The graph node should have a @graph property with inner content.
        let graph_node = graph
            .iter()
            .find(|n| {
                n.as_object()
                    .and_then(|m| m.get("@id"))
                    .and_then(|v| v.as_str())
                    == Some("http://example.org/Graph")
            })
            .expect("graph node found");
        let inner = graph_node
            .as_object()
            .and_then(|m| m.get("@graph"))
            .and_then(|v| v.as_array())
            .expect("@graph array");
        let inner_ids: Vec<&str> = inner
            .iter()
            .filter_map(|n| n.as_object())
            .filter_map(|m| m.get("@id"))
            .filter_map(|v| v.as_str())
            .collect();
        assert!(
            inner_ids.contains(&"http://example.org/Inner"),
            "inner node in @graph"
        );
    }

    // -----------------------------------------------------------------------
    // Test 11: multiple subjects
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_multiple_subjects() {
        let input = arr(vec![
            obj(&[
                ("@id", s("http://example.org/Alice")),
                ("@type", arr(vec![s("http://schema.org/Person")])),
            ]),
            obj(&[
                ("@id", s("http://example.org/Bob")),
                ("@type", arr(vec![s("http://schema.org/Person")])),
            ]),
        ]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        assert_eq!(graph.len(), 2, "two subjects in output");
        let ids: Vec<&str> = graph
            .iter()
            .filter_map(|n| n.as_object())
            .filter_map(|m| m.get("@id"))
            .filter_map(|v| v.as_str())
            .collect();
        assert!(ids.contains(&"http://example.org/Alice"));
        assert!(ids.contains(&"http://example.org/Bob"));
    }

    // -----------------------------------------------------------------------
    // Test 12: @reverse properties
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_reverse_property() {
        // B.reverse:knows.A means A.knows.B in the forward direction.
        let input = arr(vec![obj(&[
            ("@id", s("http://example.org/B")),
            (
                "@reverse",
                obj(&[(
                    "http://example.org/knows",
                    arr(vec![obj(&[("@id", s("http://example.org/A"))])]),
                )]),
            ),
        ])]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        // Both A and B must appear.
        let ids: Vec<&str> = graph
            .iter()
            .filter_map(|n| n.as_object())
            .filter_map(|m| m.get("@id"))
            .filter_map(|v| v.as_str())
            .collect();
        assert!(
            ids.contains(&"http://example.org/A"),
            "A appears (reverse source)"
        );
        assert!(ids.contains(&"http://example.org/B"), "B appears");
    }

    // -----------------------------------------------------------------------
    // Test 13: deeply nested objects all appear at top level
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_deeply_nested() {
        // A → B → C
        let c = obj(&[("@id", s("http://example.org/C"))]);
        let b = obj(&[
            ("@id", s("http://example.org/B")),
            ("http://example.org/child", arr(vec![c])),
        ]);
        let a = obj(&[
            ("@id", s("http://example.org/A")),
            ("http://example.org/child", arr(vec![b])),
        ]);
        let input = arr(vec![a]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        let ids: Vec<&str> = graph
            .iter()
            .filter_map(|n| n.as_object())
            .filter_map(|m| m.get("@id"))
            .filter_map(|v| v.as_str())
            .collect();

        assert!(ids.contains(&"http://example.org/A"), "A at top level");
        assert!(ids.contains(&"http://example.org/B"), "B at top level");
        assert!(ids.contains(&"http://example.org/C"), "C at top level");
    }

    // -----------------------------------------------------------------------
    // Test 14: duplicate subjects are merged
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_duplicate_merge() {
        // Same subject referenced in two separate top-level entries.
        let input = arr(vec![
            obj(&[
                ("@id", s("http://example.org/S")),
                (
                    "http://example.org/name",
                    arr(vec![obj(&[("@value", s("Alice"))])]),
                ),
            ]),
            obj(&[
                ("@id", s("http://example.org/S")),
                (
                    "http://example.org/age",
                    arr(vec![obj(&[("@value", s("30"))])]),
                ),
            ]),
        ]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        // Should produce a single merged subject entry.
        assert_eq!(graph.len(), 1, "duplicate subjects merged to one");
        let m = graph[0].as_object().expect("object");
        assert!(m.contains_key("http://example.org/name"), "name merged");
        assert!(m.contains_key("http://example.org/age"), "age merged");
    }

    // -----------------------------------------------------------------------
    // Test 15: empty input
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_empty_input() {
        let input = arr(vec![]);
        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);
        assert!(graph.is_empty(), "@graph should be empty for empty input");
    }

    // -----------------------------------------------------------------------
    // Test 16: value object not treated as node
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_value_object_preserved() {
        let val_obj = obj(&[("@value", s("hello"))]);
        assert!(is_value_object(&val_obj), "recognized as value object");
        assert!(!is_node_object(&val_obj), "not a node object");

        let input = arr(vec![obj(&[
            ("@id", s("http://example.org/S")),
            ("http://example.org/name", arr(vec![val_obj])),
        ])]);

        let result = flatten(&input, None, &default_opts()).expect("flatten ok");
        let graph = get_graph(&result);

        // The value object should remain nested under the name property, not promoted.
        assert_eq!(graph.len(), 1, "only one subject, value is not promoted");
        let m = graph[0].as_object().expect("object");
        let name_vals = m
            .get("http://example.org/name")
            .and_then(|v| v.as_array())
            .expect("name array");
        assert!(!name_vals.is_empty());
        assert!(is_value_object(&name_vals[0]), "value object preserved");
    }

    // -----------------------------------------------------------------------
    // Test 17: node map basic
    // -----------------------------------------------------------------------
    #[test]
    fn test_node_map_basic() {
        let input = vec![
            obj(&[
                ("@id", s("http://example.org/A")),
                ("@type", arr(vec![s("http://example.org/Person")])),
            ]),
            obj(&[("@id", s("http://example.org/B"))]),
        ];

        let opts = default_opts();
        let node_map = generate_node_map(&input, &opts).expect("node map ok");

        let default_graph = node_map.default_graph();
        assert!(
            default_graph.nodes.contains_key("http://example.org/A"),
            "A in node map"
        );
        assert!(
            default_graph.nodes.contains_key("http://example.org/B"),
            "B in node map"
        );

        let a = &default_graph.nodes["http://example.org/A"];
        assert_eq!(a.types, vec!["http://example.org/Person"]);
    }

    // -----------------------------------------------------------------------
    // Test 18: BlankNodeIdMapper basic
    // -----------------------------------------------------------------------
    #[test]
    fn test_blank_node_mapper() {
        let mut mapper = BlankNodeIdMapper::new();

        let id0 = mapper.map("_:alpha");
        let id1 = mapper.map("_:beta");
        let id0_again = mapper.map("_:alpha");

        assert_eq!(id0, "_:b0", "first blank node is _:b0");
        assert_eq!(id1, "_:b1", "second blank node is _:b1");
        assert_eq!(id0, id0_again, "mapping is idempotent");
    }

    // -----------------------------------------------------------------------
    // Test 19: BlankNodeIdMapper reset
    // -----------------------------------------------------------------------
    #[test]
    fn test_blank_node_mapper_reset() {
        let mut mapper = BlankNodeIdMapper::new();
        mapper.map("_:x");
        mapper.map("_:y");
        mapper.reset();

        let after_reset = mapper.map("_:z");
        assert_eq!(after_reset, "_:b0", "after reset first blank node is _:b0");
    }

    // -----------------------------------------------------------------------
    // Test 20: flatten with context (compaction applied)
    // -----------------------------------------------------------------------
    #[test]
    fn test_flatten_with_context() {
        let context = obj(&[("ex", s("http://example.org/"))]);

        let input = arr(vec![obj(&[
            ("@id", s("http://example.org/Alice")),
            ("@type", arr(vec![s("http://example.org/Person")])),
        ])]);

        let result =
            flatten(&input, Some(&context), &default_opts()).expect("flatten with context ok");

        // Result must have @context.
        let m = result.as_object().expect("result is object");
        assert!(m.contains_key("@context"), "@context present");

        // @graph should still be there (possibly compacted).
        assert!(m.contains_key("@graph"), "@graph present");
    }

    // -----------------------------------------------------------------------
    // Test 21: is_list_object helper
    // -----------------------------------------------------------------------
    #[test]
    fn test_is_list_object() {
        let list = obj(&[("@list", arr(vec![s("a")]))]);
        let non_list = obj(&[("@id", s("http://example.org/X"))]);
        let val = obj(&[("@value", s("hello"))]);

        assert!(is_list_object(&list), "recognized as list object");
        assert!(!is_list_object(&non_list), "node object not list");
        assert!(!is_list_object(&val), "value object not list");
    }

    // -----------------------------------------------------------------------
    // Test 22: is_node_object helper
    // -----------------------------------------------------------------------
    #[test]
    fn test_is_node_object_helper() {
        let node = obj(&[("@id", s("http://example.org/X"))]);
        let val = obj(&[("@value", s("hello"))]);
        let list = obj(&[("@list", arr(vec![]))]);
        let scalar = s("plain string");

        assert!(is_node_object(&node), "node object recognized");
        assert!(!is_node_object(&val), "value object not a node object");
        assert!(!is_node_object(&list), "list object not a node object");
        assert!(!is_node_object(&scalar), "scalar not a node object");
    }
}
