//! Tests for the JSON-LD 1.1 Compaction Algorithm.
//!
//! Covers the main compaction scenarios specified in:
//! <https://www.w3.org/TR/json-ld11-api/#compaction-algorithm>

#[cfg(test)]
mod compaction_tests {
    use crate::jsonld::compaction::{
        compact, compact_array, compact_iri, compact_node, compact_value, CompactionOptions,
        ContainerType, JsonLdContext, JsonLdValue, TermDefinition,
    };
    use indexmap::IndexMap;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_obj(pairs: &[(&str, JsonLdValue)]) -> JsonLdValue {
        let mut map: IndexMap<String, JsonLdValue> = IndexMap::new();
        for (k, v) in pairs {
            map.insert((*k).to_string(), v.clone());
        }
        JsonLdValue::Object(map)
    }

    fn schema_ctx() -> JsonLdContext {
        let mut ctx = JsonLdContext::new();
        ctx.add_prefix("schema", "http://schema.org/");
        ctx
    }

    fn xsd_ctx() -> JsonLdContext {
        let mut ctx = JsonLdContext::new();
        ctx.add_prefix("xsd", "http://www.w3.org/2001/XMLSchema#");
        ctx
    }

    fn default_opts() -> CompactionOptions {
        CompactionOptions::default()
    }

    // -----------------------------------------------------------------------
    // Test 1: compact_iri — simple prefix
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_simple_iri() {
        let input = make_obj(&[(
            "@id",
            JsonLdValue::Str("http://schema.org/name".to_string()),
        )]);
        let ctx = schema_ctx();
        let opts = default_opts();
        let result = compact(&input, &ctx, &opts).expect("compact succeeded");
        // The @id should be compacted to schema:name
        if let JsonLdValue::Object(ref m) = result {
            let id = m.get("@id").and_then(|v| v.as_str()).unwrap_or("");
            assert_eq!(id, "schema:name", "IRI should compact to schema:name");
        } else {
            panic!("Expected Object result");
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: compact plain string @value wrapper drop
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_string_value() {
        let value = make_obj(&[("@value", JsonLdValue::Str("hello".to_string()))]);
        let ctx = JsonLdContext::new();
        let result = compact_value(&ctx, None, &value).expect("compact_value succeeded");
        assert_eq!(
            result,
            JsonLdValue::Str("hello".to_string()),
            "Plain string @value should drop wrapper"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: typed literal compaction
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_typed_literal() {
        let ctx = xsd_ctx();
        let value = make_obj(&[
            ("@value", JsonLdValue::Str("42".to_string())),
            (
                "@type",
                JsonLdValue::Str("http://www.w3.org/2001/XMLSchema#integer".to_string()),
            ),
        ]);
        let result = compact_value(&ctx, None, &value).expect("compact_value succeeded");
        // Should produce {"@value": "42", "@type": "xsd:integer"}
        if let JsonLdValue::Object(ref m) = result {
            let typ = m.get("@type").and_then(|v| v.as_str()).unwrap_or("");
            assert_eq!(typ, "xsd:integer", "Type should compact to xsd:integer");
            let val = m.get("@value").and_then(|v| v.as_str()).unwrap_or("");
            assert_eq!(val, "42");
        } else {
            panic!("Expected Object for typed literal: {:?}", result);
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: language literal
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_language_literal() {
        let ctx = JsonLdContext::new();
        let value = make_obj(&[
            ("@value", JsonLdValue::Str("bonjour".to_string())),
            ("@language", JsonLdValue::Str("fr".to_string())),
        ]);
        let result = compact_value(&ctx, None, &value).expect("compact_value succeeded");
        if let JsonLdValue::Object(ref m) = result {
            let lang = m.get("@language").and_then(|v| v.as_str()).unwrap_or("");
            assert_eq!(lang, "fr");
            let val = m.get("@value").and_then(|v| v.as_str()).unwrap_or("");
            assert_eq!(val, "bonjour");
        } else {
            panic!("Expected Object for language literal: {:?}", result);
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: compact_arrays=true collapses single element
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_arrays_true() {
        let ctx = JsonLdContext::new();
        let opts = CompactionOptions {
            compact_arrays: true,
            ..Default::default()
        };
        let items = vec![JsonLdValue::Str("hello".to_string())];
        let result = compact_array(&ctx, "prop", &items, &opts).expect("compact_array ok");
        // Single element should collapse.
        assert_eq!(result, JsonLdValue::Str("hello".to_string()));
    }

    // -----------------------------------------------------------------------
    // Test 6: compact_arrays=false keeps single element as array
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_arrays_false() {
        let ctx = JsonLdContext::new();
        let opts = CompactionOptions {
            compact_arrays: false,
            ..Default::default()
        };
        let items = vec![JsonLdValue::Str("hello".to_string())];
        let result = compact_array(&ctx, "prop", &items, &opts).expect("compact_array ok");
        // Should stay as array.
        match result {
            JsonLdValue::Array(a) => assert_eq!(a.len(), 1),
            other => panic!("Expected Array, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: compact_nested_node — recursive node compaction
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_nested_node() {
        let mut ctx = schema_ctx();
        ctx.add_prefix("foaf", "http://xmlns.com/foaf/0.1/");

        let inner = make_obj(&[(
            "@id",
            JsonLdValue::Str("http://schema.org/alice".to_string()),
        )]);
        let mut outer_map: IndexMap<String, JsonLdValue> = IndexMap::new();
        outer_map.insert(
            "@id".to_string(),
            JsonLdValue::Str("http://schema.org/bob".to_string()),
        );
        outer_map.insert(
            "http://xmlns.com/foaf/0.1/knows".to_string(),
            JsonLdValue::Array(vec![inner]),
        );
        let opts = default_opts();
        let result = compact_node(&ctx, &ctx, None, &outer_map, &opts).expect("compact_node ok");

        if let JsonLdValue::Object(ref m) = result {
            let id = m.get("@id").and_then(|v| v.as_str()).unwrap_or("");
            assert_eq!(id, "schema:bob");
            // foaf:knows should be compacted.
            assert!(
                m.contains_key("foaf:knows"),
                "Expected foaf:knows in result, got keys: {:?}",
                m.keys().collect::<Vec<_>>()
            );
        } else {
            panic!("Expected Object result");
        }
    }

    // -----------------------------------------------------------------------
    // Test 8: @vocab mapping for unmatched properties
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_vocab_mapping() {
        let mut ctx = JsonLdContext::new();
        ctx.vocab = Some("http://example.org/".to_string());

        let result = compact_iri(&ctx, "http://example.org/name", None, true, false);
        assert_eq!(
            result, "name",
            "Vocab-relative IRI should compact to suffix"
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: reverse property compaction
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_reverse_property() {
        let mut ctx = JsonLdContext::new();
        let mut rev_def = TermDefinition::simple("http://example.org/isChildOf");
        rev_def.reverse_property = true;
        ctx.add_term("hasParent", rev_def);

        // compact_iri with reverse=true should find the reverse term.
        let result = compact_iri(&ctx, "http://example.org/isChildOf", None, true, true);
        assert_eq!(result, "hasParent");
    }

    // -----------------------------------------------------------------------
    // Test 10: @language container — language map
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_language_map() {
        let mut ctx = JsonLdContext::new();
        let mut name_def = TermDefinition::simple("http://schema.org/name");
        name_def.container = vec![ContainerType::Language];
        ctx.add_term("name", name_def);

        let values = vec![
            make_obj(&[
                ("@value", JsonLdValue::Str("hello".to_string())),
                ("@language", JsonLdValue::Str("en".to_string())),
            ]),
            make_obj(&[
                ("@value", JsonLdValue::Str("bonjour".to_string())),
                ("@language", JsonLdValue::Str("fr".to_string())),
            ]),
        ];

        let mut node_map: IndexMap<String, JsonLdValue> = IndexMap::new();
        node_map.insert(
            "http://schema.org/name".to_string(),
            JsonLdValue::Array(values),
        );

        let opts = default_opts();
        let result = compact_node(&ctx, &ctx, None, &node_map, &opts).expect("compact_node ok");

        if let JsonLdValue::Object(ref m) = result {
            let name = m.get("name").expect("name key present");
            if let JsonLdValue::Object(ref lang_map) = name {
                let en = lang_map.get("en").and_then(|v| v.as_str()).unwrap_or("");
                let fr = lang_map.get("fr").and_then(|v| v.as_str()).unwrap_or("");
                assert_eq!(en, "hello");
                assert_eq!(fr, "bonjour");
            } else {
                panic!("Expected language map object, got {:?}", name);
            }
        } else {
            panic!("Expected Object result");
        }
    }

    // -----------------------------------------------------------------------
    // Test 11: @type container — type map
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_type_map() {
        let mut ctx = JsonLdContext::new();
        let mut prop_def = TermDefinition::simple("http://example.org/label");
        prop_def.container = vec![ContainerType::Type];
        ctx.add_term("label", prop_def);
        ctx.add_prefix("xsd", "http://www.w3.org/2001/XMLSchema#");

        let values = vec![make_obj(&[
            ("@value", JsonLdValue::Str("42".to_string())),
            (
                "@type",
                JsonLdValue::Str("http://www.w3.org/2001/XMLSchema#integer".to_string()),
            ),
        ])];

        let mut node_map: IndexMap<String, JsonLdValue> = IndexMap::new();
        node_map.insert(
            "http://example.org/label".to_string(),
            JsonLdValue::Array(values),
        );

        let opts = default_opts();
        let result = compact_node(&ctx, &ctx, None, &node_map, &opts).expect("compact_node ok");

        if let JsonLdValue::Object(ref m) = result {
            let label = m.get("label").expect("label key present");
            // Should be a type-keyed map.
            assert!(
                matches!(label, JsonLdValue::Object(_)),
                "Expected type map: {:?}",
                label
            );
        } else {
            panic!("Expected Object result");
        }
    }

    // -----------------------------------------------------------------------
    // Test 12: @list container
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_list_container() {
        let mut ctx = JsonLdContext::new();
        let mut prop_def = TermDefinition::simple("http://example.org/items");
        prop_def.container = vec![ContainerType::List];
        ctx.add_term("items", prop_def);

        // Expanded form: [{"@list": ["a", "b"]}]
        let mut list_map: IndexMap<String, JsonLdValue> = IndexMap::new();
        list_map.insert(
            "@list".to_string(),
            JsonLdValue::Array(vec![
                JsonLdValue::Str("a".to_string()),
                JsonLdValue::Str("b".to_string()),
            ]),
        );
        let values = vec![JsonLdValue::Object(list_map)];

        let mut node_map: IndexMap<String, JsonLdValue> = IndexMap::new();
        node_map.insert(
            "http://example.org/items".to_string(),
            JsonLdValue::Array(values),
        );

        let opts = default_opts();
        let result = compact_node(&ctx, &ctx, None, &node_map, &opts).expect("compact_node ok");

        if let JsonLdValue::Object(ref m) = result {
            // items should be present as an array (list contents).
            let items = m.get("items").expect("items key present");
            match items {
                JsonLdValue::Array(a) => {
                    assert_eq!(a.len(), 2, "Expected 2 list items");
                }
                other => panic!("Expected Array for list, got {:?}", other),
            }
        } else {
            panic!("Expected Object result");
        }
    }

    // -----------------------------------------------------------------------
    // Test 13: @index container
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_index_map() {
        let mut ctx = JsonLdContext::new();
        let mut prop_def = TermDefinition::simple("http://example.org/data");
        prop_def.container = vec![ContainerType::Index];
        ctx.add_term("data", prop_def);

        // Values with @index.
        let mut v1: IndexMap<String, JsonLdValue> = IndexMap::new();
        v1.insert("@index".to_string(), JsonLdValue::Str("a".to_string()));
        v1.insert(
            "@value".to_string(),
            JsonLdValue::Str("value-a".to_string()),
        );

        let mut v2: IndexMap<String, JsonLdValue> = IndexMap::new();
        v2.insert("@index".to_string(), JsonLdValue::Str("b".to_string()));
        v2.insert(
            "@value".to_string(),
            JsonLdValue::Str("value-b".to_string()),
        );

        let values = vec![JsonLdValue::Object(v1), JsonLdValue::Object(v2)];
        let mut node_map: IndexMap<String, JsonLdValue> = IndexMap::new();
        node_map.insert(
            "http://example.org/data".to_string(),
            JsonLdValue::Array(values),
        );

        let opts = default_opts();
        let result = compact_node(&ctx, &ctx, None, &node_map, &opts).expect("compact_node ok");

        if let JsonLdValue::Object(ref m) = result {
            let data = m.get("data").expect("data key present");
            match data {
                JsonLdValue::Object(idx_map) => {
                    assert!(idx_map.contains_key("a"), "Expected 'a' key in index map");
                    assert!(idx_map.contains_key("b"), "Expected 'b' key in index map");
                }
                other => panic!("Expected index Object, got {:?}", other),
            }
        } else {
            panic!("Expected Object result");
        }
    }

    // -----------------------------------------------------------------------
    // Test 14: compact full realistic document
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_full_document() {
        let mut ctx = JsonLdContext::new();
        ctx.add_prefix("schema", "http://schema.org/");
        ctx.add_prefix("foaf", "http://xmlns.com/foaf/0.1/");
        ctx.add_prefix("xsd", "http://www.w3.org/2001/XMLSchema#");

        // Expanded JSON-LD document with 5 triples.
        let mut name_val: IndexMap<String, JsonLdValue> = IndexMap::new();
        name_val.insert("@value".to_string(), JsonLdValue::Str("Alice".to_string()));

        let mut age_val: IndexMap<String, JsonLdValue> = IndexMap::new();
        age_val.insert("@value".to_string(), JsonLdValue::Str("30".to_string()));
        age_val.insert(
            "@type".to_string(),
            JsonLdValue::Str("http://www.w3.org/2001/XMLSchema#integer".to_string()),
        );

        let mut knows_node: IndexMap<String, JsonLdValue> = IndexMap::new();
        knows_node.insert(
            "@id".to_string(),
            JsonLdValue::Str("http://schema.org/bob".to_string()),
        );

        let mut node: IndexMap<String, JsonLdValue> = IndexMap::new();
        node.insert(
            "@id".to_string(),
            JsonLdValue::Str("http://schema.org/alice".to_string()),
        );
        node.insert(
            "@type".to_string(),
            JsonLdValue::Array(vec![JsonLdValue::Str(
                "http://schema.org/Person".to_string(),
            )]),
        );
        node.insert(
            "http://xmlns.com/foaf/0.1/name".to_string(),
            JsonLdValue::Array(vec![JsonLdValue::Object(name_val)]),
        );
        node.insert(
            "http://schema.org/age".to_string(),
            JsonLdValue::Array(vec![JsonLdValue::Object(age_val)]),
        );
        node.insert(
            "http://schema.org/knows".to_string(),
            JsonLdValue::Array(vec![JsonLdValue::Object(knows_node)]),
        );

        let input = JsonLdValue::Array(vec![JsonLdValue::Object(node)]);
        let opts = default_opts();
        let result = compact(&input, &ctx, &opts).expect("compact succeeded");

        if let JsonLdValue::Object(ref m) = result {
            // @id and @type should be compacted.
            let id = m.get("@id").and_then(|v| v.as_str()).unwrap_or("");
            assert!(
                id.contains("schema:alice") || id.contains("alice"),
                "ID should be compacted: {}",
                id
            );
            // foaf:name or name should be present.
            let has_name = m.contains_key("foaf:name")
                || m.contains_key("name")
                || m.keys().any(|k| k.contains("name"));
            assert!(
                has_name,
                "Should have name property, keys: {:?}",
                m.keys().collect::<Vec<_>>()
            );
        } else {
            panic!("Expected Object result, got {:?}", result);
        }
    }

    // -----------------------------------------------------------------------
    // Test 15: IRI with no matching term stays absolute when vocab=false
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_iri_no_vocab() {
        let ctx = JsonLdContext::new(); // empty context, no vocab
        let result = compact_iri(&ctx, "http://unknown.example.org/prop", None, false, false);
        assert_eq!(
            result, "http://unknown.example.org/prop",
            "Unmatched IRI should remain absolute"
        );
    }

    // -----------------------------------------------------------------------
    // Test 16: find_term with @type constraint
    // -----------------------------------------------------------------------
    #[test]
    fn test_find_term_type_constraint() {
        use crate::jsonld::compaction::find_term;
        let mut ctx = JsonLdContext::new();

        let mut int_def = TermDefinition::simple("http://example.org/value");
        int_def.type_mapping = Some("http://www.w3.org/2001/XMLSchema#integer".to_string());
        ctx.add_term("intValue", int_def);

        let mut str_def = TermDefinition::simple("http://example.org/value");
        str_def.type_mapping = Some("http://www.w3.org/2001/XMLSchema#string".to_string());
        ctx.add_term("strValue", str_def);

        // Find term matching integer type.
        let term = find_term(
            &ctx,
            "http://example.org/value",
            None,
            &[],
            "@type",
            "http://www.w3.org/2001/XMLSchema#integer",
        );
        assert_eq!(
            term.as_deref(),
            Some("intValue"),
            "Should find intValue term for integer type"
        );
    }

    // -----------------------------------------------------------------------
    // Test 17: nested reverse properties
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_nested_reverse() {
        let mut ctx = JsonLdContext::new();
        ctx.add_prefix("ex", "http://example.org/");

        // @reverse object.
        let mut rev_inner: IndexMap<String, JsonLdValue> = IndexMap::new();
        rev_inner.insert(
            "http://example.org/parent".to_string(),
            JsonLdValue::Array(vec![make_obj(&[(
                "@id",
                JsonLdValue::Str("http://example.org/dad".to_string()),
            )])]),
        );

        let mut node_map: IndexMap<String, JsonLdValue> = IndexMap::new();
        node_map.insert(
            "@id".to_string(),
            JsonLdValue::Str("http://example.org/child".to_string()),
        );
        node_map.insert("@reverse".to_string(), JsonLdValue::Object(rev_inner));

        let opts = default_opts();
        let result = compact_node(&ctx, &ctx, None, &node_map, &opts).expect("compact_node ok");

        if let JsonLdValue::Object(ref m) = result {
            let id = m.get("@id").and_then(|v| v.as_str()).unwrap_or("");
            assert_eq!(id, "ex:child");
            // @reverse or ex:parent should be present.
            let has_reverse = m.contains_key("@reverse") || m.contains_key("ex:parent");
            assert!(
                has_reverse,
                "Should have reverse property, keys: {:?}",
                m.keys().collect::<Vec<_>>()
            );
        } else {
            panic!("Expected Object result");
        }
    }

    // -----------------------------------------------------------------------
    // Test 18: protected context roundtrip structure
    // -----------------------------------------------------------------------
    #[test]
    fn test_compact_protected_context_roundtrip() {
        let mut ctx = JsonLdContext::new();
        let mut protected_def = TermDefinition::simple("http://example.org/safeProp");
        protected_def.protected = true;
        ctx.add_term("safe", protected_def);
        ctx.add_prefix("ex", "http://example.org/");

        let mut node_map: IndexMap<String, JsonLdValue> = IndexMap::new();
        node_map.insert(
            "http://example.org/safeProp".to_string(),
            JsonLdValue::Array(vec![JsonLdValue::Object({
                let mut m: IndexMap<String, JsonLdValue> = IndexMap::new();
                m.insert(
                    "@value".to_string(),
                    JsonLdValue::Str("protected".to_string()),
                );
                m
            })]),
        );

        let opts = default_opts();
        let result = compact_node(&ctx, &ctx, None, &node_map, &opts).expect("compact_node ok");

        if let JsonLdValue::Object(ref m) = result {
            // "safe" term should be used for http://example.org/safeProp.
            assert!(
                m.contains_key("safe"),
                "Expected 'safe' key from protected term, keys: {:?}",
                m.keys().collect::<Vec<_>>()
            );
            // Value should be compacted (plain string).
            let val = m.get("safe");
            assert!(
                matches!(val, Some(JsonLdValue::Str(_))),
                "Expected plain string value, got {:?}",
                val
            );
        } else {
            panic!("Expected Object result");
        }
    }
}
