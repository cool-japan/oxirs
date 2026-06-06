//! Tests for the JSON-LD 1.1 Framing implementation.
//!
//! Covers W3C JSON-LD Framing spec scenarios using `serde_json::json!` macros.

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::jsonld::framing::{EmbedPolicy, FramingOptions, FramingState, JsonLdFramer};
    use crate::jsonld::framing_embed::{apply_defaults, apply_explicit, EmbedDecision};
    use crate::jsonld::framing_match::{matches_frame, matches_id, matches_type};

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn default_framer() -> JsonLdFramer {
        JsonLdFramer::new(FramingOptions::default())
    }

    fn person_type() -> &'static str {
        "http://schema.org/Person"
    }

    fn org_type() -> &'static str {
        "http://schema.org/Organization"
    }

    // ── 1. test_frame_basic_type ─────────────────────────────────────────────

    #[test]
    fn test_frame_basic_type() {
        let input = json!([
            {
                "@id": "http://example.org/alice",
                "@type": [person_type()],
                "http://schema.org/name": [{"@value": "Alice"}]
            },
            {
                "@id": "http://example.org/acme",
                "@type": [org_type()],
                "http://schema.org/name": [{"@value": "Acme Corp"}]
            }
        ]);
        let frame = json!({"@type": person_type()});

        let framer = default_framer();
        let result = framer.frame(&input, &frame).expect("frame should succeed");

        let graph = result["@graph"].as_array().expect("@graph must be array");
        assert_eq!(graph.len(), 1, "Only the Person should match");
        assert_eq!(graph[0]["@id"], "http://example.org/alice");
    }

    // ── 2. test_frame_no_match ───────────────────────────────────────────────

    #[test]
    fn test_frame_no_match() {
        let input = json!([
            {
                "@id": "http://example.org/alice",
                "@type": [person_type()]
            }
        ]);
        let frame = json!({"@type": "http://schema.org/Event"});

        let framer = default_framer();
        let result = framer.frame(&input, &frame).expect("frame should succeed");
        let graph = result["@graph"].as_array().expect("@graph must be array");
        assert!(graph.is_empty(), "No subject matches Event type");
    }

    // ── 3. test_frame_id_match ───────────────────────────────────────────────

    #[test]
    fn test_frame_id_match() {
        let input = json!([
            {
                "@id": "http://example.org/alice",
                "@type": [person_type()]
            },
            {
                "@id": "http://example.org/bob",
                "@type": [person_type()]
            }
        ]);
        let frame = json!({"@id": "http://example.org/alice"});

        let framer = default_framer();
        let result = framer.frame(&input, &frame).expect("frame should succeed");
        let graph = result["@graph"].as_array().expect("@graph must be array");
        assert_eq!(graph.len(), 1);
        assert_eq!(graph[0]["@id"], "http://example.org/alice");
    }

    // ── 4. test_frame_embed_first ────────────────────────────────────────────

    #[test]
    fn test_frame_embed_first() {
        let options = FramingOptions {
            embed: EmbedPolicy::First,
            ..Default::default()
        };
        let mut state = FramingState::new();

        // First visit → Full
        let decision = crate::jsonld::framing_embed::apply_embed_policy(
            &mut state,
            "http://example.org/s1",
            EmbedPolicy::First,
        );
        assert_eq!(decision, EmbedDecision::Full);

        // Simulate marking as embedded
        state.embedded.insert("http://example.org/s1".to_string());

        // Second visit → Skip
        let decision2 = crate::jsonld::framing_embed::apply_embed_policy(
            &mut state,
            "http://example.org/s1",
            EmbedPolicy::First,
        );
        assert_eq!(decision2, EmbedDecision::Skip);
        let _ = options;
    }

    // ── 5. test_frame_embed_always ───────────────────────────────────────────

    #[test]
    fn test_frame_embed_always() {
        let mut state = FramingState::new();

        let d1 = crate::jsonld::framing_embed::apply_embed_policy(
            &mut state,
            "http://example.org/s2",
            EmbedPolicy::Always,
        );
        assert_eq!(d1, EmbedDecision::Full);

        state.embedded.insert("http://example.org/s2".to_string());

        // Always policy: second visit → Skip (cycle protection)
        let d2 = crate::jsonld::framing_embed::apply_embed_policy(
            &mut state,
            "http://example.org/s2",
            EmbedPolicy::Always,
        );
        assert_eq!(d2, EmbedDecision::Skip);
    }

    // ── 6. test_frame_embed_never ────────────────────────────────────────────

    #[test]
    fn test_frame_embed_never() {
        let mut state = FramingState::new();

        let d = crate::jsonld::framing_embed::apply_embed_policy(
            &mut state,
            "http://example.org/s3",
            EmbedPolicy::Never,
        );
        assert_eq!(d, EmbedDecision::Skip);
    }

    // ── 7. test_frame_explicit_true ──────────────────────────────────────────

    #[test]
    fn test_frame_explicit_true() {
        let options = FramingOptions {
            explicit: true,
            ..Default::default()
        };
        let subject = json!({
            "@id": "http://example.org/alice",
            "@type": [person_type()],
            "http://schema.org/name": [{"@value": "Alice"}],
            "http://schema.org/email": [{"@value": "alice@example.org"}]
        });
        let frame = json!({
            "@type": person_type(),
            "http://schema.org/name": [{}]
        });

        let result = apply_explicit(&subject, &frame, &options);
        let obj = result.as_object().expect("must be object");

        assert!(obj.contains_key("@id"));
        assert!(obj.contains_key("@type"));
        assert!(
            obj.contains_key("http://schema.org/name"),
            "name is in frame so must be kept"
        );
        assert!(
            !obj.contains_key("http://schema.org/email"),
            "email is NOT in frame so must be pruned"
        );
    }

    // ── 8. test_frame_explicit_false ─────────────────────────────────────────

    #[test]
    fn test_frame_explicit_false() {
        let options = FramingOptions {
            explicit: false,
            ..Default::default()
        };
        let subject = json!({
            "@id": "http://example.org/alice",
            "http://schema.org/name": [{"@value": "Alice"}],
            "http://schema.org/email": [{"@value": "alice@example.org"}]
        });
        let frame = json!({"http://schema.org/name": [{}]});

        let result = apply_explicit(&subject, &frame, &options);
        let obj = result.as_object().expect("must be object");

        // All properties must survive when explicit=false
        assert!(obj.contains_key("http://schema.org/name"));
        assert!(obj.contains_key("http://schema.org/email"));
    }

    // ── 9. test_frame_require_all_true ───────────────────────────────────────

    #[test]
    fn test_frame_require_all_true() {
        let options = FramingOptions {
            require_all: true,
            ..Default::default()
        };

        let subject_both = json!({
            "@id": "http://example.org/alice",
            "http://schema.org/name": [{"@value": "Alice"}],
            "http://schema.org/email": [{"@value": "alice@example.org"}]
        });
        let subject_one = json!({
            "@id": "http://example.org/bob",
            "http://schema.org/name": [{"@value": "Bob"}]
        });

        let frame = json!({
            "http://schema.org/name": [{}],
            "http://schema.org/email": [{}]
        });

        assert!(
            matches_frame(&subject_both, &frame, &options),
            "Subject with both properties should match"
        );
        assert!(
            !matches_frame(&subject_one, &frame, &options),
            "Subject with only name should not match (require_all)"
        );
    }

    // ── 10. test_frame_require_all_false ─────────────────────────────────────

    #[test]
    fn test_frame_require_all_false() {
        let options = FramingOptions {
            require_all: false,
            ..Default::default()
        };

        let subject_one = json!({
            "@id": "http://example.org/bob",
            "http://schema.org/name": [{"@value": "Bob"}]
        });
        let frame = json!({
            "http://schema.org/name": [{}],
            "http://schema.org/email": [{}]
        });

        assert!(
            matches_frame(&subject_one, &frame, &options),
            "Subject matching any property should match when require_all=false"
        );
    }

    // ── 11. test_frame_omit_default_false ────────────────────────────────────

    #[test]
    fn test_frame_omit_default_false() {
        let options = FramingOptions {
            omit_default: false,
            ..Default::default()
        };

        let frame = json!({
            "http://schema.org/name": [{"@default": "Unknown"}]
        });

        let mut output = json!({"@id": "http://example.org/anon"});
        apply_defaults(&mut output, &frame, &options);

        let obj = output.as_object().expect("must be object");
        assert!(
            obj.contains_key("http://schema.org/name"),
            "Missing property should get its @default value"
        );
        assert_eq!(output["http://schema.org/name"][0]["@value"], "Unknown");
    }

    // ── 12. test_frame_omit_default_true ─────────────────────────────────────

    #[test]
    fn test_frame_omit_default_true() {
        let options = FramingOptions {
            omit_default: true,
            ..Default::default()
        };

        let frame = json!({
            "http://schema.org/name": [{"@default": "Unknown"}]
        });

        let mut output = json!({"@id": "http://example.org/anon"});
        apply_defaults(&mut output, &frame, &options);

        let obj = output.as_object().expect("must be object");
        assert!(
            !obj.contains_key("http://schema.org/name"),
            "Property should NOT be injected when omit_default=true"
        );
    }

    // ── 13. test_frame_nested ────────────────────────────────────────────────

    #[test]
    fn test_frame_nested() {
        let input = json!([
            {
                "@id": "http://example.org/alice",
                "@type": [person_type()],
                "http://schema.org/knows": [{"@id": "http://example.org/bob"}]
            },
            {
                "@id": "http://example.org/bob",
                "@type": [person_type()],
                "http://schema.org/name": [{"@value": "Bob"}]
            }
        ]);

        // Frame: find Persons and embed their "knows" connections
        let frame = json!({
            "@type": person_type(),
            "http://schema.org/knows": [{
                "@type": person_type()
            }]
        });

        let framer = default_framer();
        let result = framer
            .frame(&input, &frame)
            .expect("framing should succeed");
        let graph = result["@graph"].as_array().expect("@graph must be array");

        // Alice and Bob both match the Person type; Alice should embed Bob
        assert!(!graph.is_empty());

        // Find Alice in the output
        let alice = graph
            .iter()
            .find(|n| n["@id"] == "http://example.org/alice");
        assert!(alice.is_some(), "Alice must be in the framed output");
    }

    // ── 14. test_frame_cyclic_link_policy ────────────────────────────────────

    #[test]
    fn test_frame_cyclic_link_policy() {
        let mut state = FramingState::new();
        let id = "http://example.org/cyclic";

        // First visit: Full
        let d1 =
            crate::jsonld::framing_embed::apply_embed_policy(&mut state, id, EmbedPolicy::Link);
        assert_eq!(d1, EmbedDecision::Full);

        state.embedded.insert(id.to_string());
        state.link.insert(id.to_string(), json!({"@id": id}));

        // Second visit: Link (not Full, avoids infinite recursion)
        let d2 =
            crate::jsonld::framing_embed::apply_embed_policy(&mut state, id, EmbedPolicy::Link);
        assert_eq!(d2, EmbedDecision::Link);
    }

    // ── 15. test_match_type_single ───────────────────────────────────────────

    #[test]
    fn test_match_type_single() {
        let subject = json!({
            "@id": "http://example.org/alice",
            "@type": [person_type()]
        });

        assert!(
            matches_type(&subject, &json!(person_type())),
            "Single type string should match"
        );
        assert!(
            !matches_type(&subject, &json!(org_type())),
            "Different type should not match"
        );
    }

    // ── 16. test_match_type_any ───────────────────────────────────────────────

    #[test]
    fn test_match_type_any() {
        let subject = json!({
            "@type": [person_type()]
        });

        // Frame type is an array — any overlap is sufficient
        let frame_types = json!([person_type(), org_type()]);
        assert!(
            matches_type(&subject, &frame_types),
            "Subject should match when it has at least one type from the frame array"
        );
    }

    // ── 17. test_apply_explicit_prune ────────────────────────────────────────

    #[test]
    fn test_apply_explicit_prune() {
        let options = FramingOptions {
            explicit: true,
            ..Default::default()
        };

        let subject = json!({
            "@id": "http://example.org/x",
            "http://example.org/propA": [{"@value": "a"}],
            "http://example.org/propB": [{"@value": "b"}],
            "http://example.org/propC": [{"@value": "c"}]
        });

        let frame = json!({
            "http://example.org/propA": [{}],
            "http://example.org/propB": [{}]
        });

        let result = apply_explicit(&subject, &frame, &options);
        let obj = result.as_object().expect("must be object");

        assert!(obj.contains_key("http://example.org/propA"));
        assert!(obj.contains_key("http://example.org/propB"));
        assert!(
            !obj.contains_key("http://example.org/propC"),
            "propC not in frame and should be pruned"
        );
    }

    // ── 18. test_apply_defaults_add ──────────────────────────────────────────

    #[test]
    fn test_apply_defaults_add() {
        let options = FramingOptions {
            omit_default: false,
            ..Default::default()
        };

        let frame = json!({
            "http://example.org/propA": [{"@default": 42}],
            "http://example.org/propB": [{"@default": "hello"}]
        });

        let mut output = json!({"@id": "http://example.org/y"});
        apply_defaults(&mut output, &frame, &options);

        let obj = output.as_object().expect("must be object");

        assert!(
            obj.contains_key("http://example.org/propA"),
            "propA default must be injected"
        );
        assert!(
            obj.contains_key("http://example.org/propB"),
            "propB default must be injected"
        );

        assert_eq!(output["http://example.org/propA"][0]["@value"], 42);
        assert_eq!(output["http://example.org/propB"][0]["@value"], "hello");
    }

    // ── Extra: test_matches_id_exact ─────────────────────────────────────────

    #[test]
    fn test_matches_id_exact() {
        let subject = json!({"@id": "http://example.org/alice"});

        assert!(matches_id(&subject, &json!("http://example.org/alice")));
        assert!(!matches_id(&subject, &json!("http://example.org/bob")));
    }

    // ── Extra: test_matches_id_wildcard ──────────────────────────────────────

    #[test]
    fn test_matches_id_wildcard() {
        let subject = json!({"@id": "http://example.org/alice"});
        // An empty object acts as a wildcard for @id
        assert!(matches_id(&subject, &json!({})));
    }

    // ── Extra: full framing integration test ─────────────────────────────────

    #[test]
    fn test_frame_full_integration() {
        let input = json!([
            {
                "@id": "http://example.org/alice",
                "@type": [person_type()],
                "http://schema.org/name": [{"@value": "Alice"}],
                "http://schema.org/age": [{"@value": 30}]
            },
            {
                "@id": "http://example.org/bob",
                "@type": [person_type()],
                "http://schema.org/name": [{"@value": "Bob"}],
                "http://schema.org/age": [{"@value": 25}]
            }
        ]);

        let frame = json!({"@type": person_type()});
        let options = FramingOptions::default();
        let framer = JsonLdFramer::new(options);
        let result = framer.frame(&input, &frame).expect("framing must succeed");

        assert!(result["@graph"].is_array());
        let graph = result["@graph"].as_array().unwrap();
        assert_eq!(graph.len(), 2, "Both persons should be in the output");
    }
}
