//! Tests for the JSON-LD expansion algorithm sub-modules.
//!
//! Covers types, step helpers, and context-propagation logic.

#[cfg(test)]
mod tests {
    use crate::jsonld::expansion_algorithm::JsonLdExpansionConverter;
    use crate::jsonld::expansion_context::JsonLdEvent;
    use crate::jsonld::profile::JsonLdProcessingMode;
    use json_event_parser::JsonEvent;
    use std::borrow::Cow;

    #[test]
    fn test_expansion_converter_new_default() {
        let converter =
            JsonLdExpansionConverter::new(None, false, false, JsonLdProcessingMode::JsonLd1_1);
        assert!(!converter.is_end());
        assert!(!converter.streaming);
        assert!(!converter.lenient);
        assert!(converter.base_url.is_none());
    }

    #[test]
    fn test_expansion_converter_streaming_mode() {
        let converter =
            JsonLdExpansionConverter::new(None, true, false, JsonLdProcessingMode::JsonLd1_1);
        assert!(converter.streaming);
    }

    #[test]
    fn test_expansion_converter_lenient_mode() {
        let converter =
            JsonLdExpansionConverter::new(None, false, true, JsonLdProcessingMode::JsonLd1_1);
        assert!(converter.lenient);
    }

    #[test]
    fn test_expansion_converter_context_stack_non_empty() {
        let converter =
            JsonLdExpansionConverter::new(None, false, false, JsonLdProcessingMode::JsonLd1_1);
        // context() must not panic — it uses .expect() on the non-empty stack
        let _ctx = converter.context();
    }

    #[test]
    fn test_expansion_converter_state_stack_non_empty() {
        let converter =
            JsonLdExpansionConverter::new(None, false, false, JsonLdProcessingMode::JsonLd1_1);
        assert!(!converter.state.is_empty());
    }

    /// Helper: feed a sequence of `JsonEvent`s into a fresh non-streaming converter
    /// and collect all emitted `JsonLdEvent`s, ignoring syntax errors.
    fn expand_events(events: &[JsonEvent<'_>]) -> Vec<JsonLdEvent> {
        let mut conv =
            JsonLdExpansionConverter::new(None, false, false, JsonLdProcessingMode::JsonLd1_1);
        let mut results = Vec::new();
        let mut errors = Vec::new();
        for event in events {
            conv.convert_event(event.clone(), &mut results, &mut errors);
        }
        results
    }

    /// Helper: feed a sequence of `JsonEvent`s into a fresh *streaming* converter.
    fn expand_events_streaming(events: &[JsonEvent<'_>]) -> Vec<JsonLdEvent> {
        let mut conv =
            JsonLdExpansionConverter::new(None, true, false, JsonLdProcessingMode::JsonLd1_1);
        let mut results = Vec::new();
        let mut errors = Vec::new();
        for event in events {
            conv.convert_event(event.clone(), &mut results, &mut errors);
        }
        results
    }

    /// Count `IndexValue` events in the result stream.
    fn index_values(events: &[JsonLdEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|e| {
                if let JsonLdEvent::IndexValue(v) = e {
                    Some(v.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Test: an `@index` container expansion emits `IndexValue` for each key.
    ///
    /// JSON-LD spec §13.8: when a term has `@container: ["@index"]`, the object
    /// keys of the index map become `@index` annotations on the expanded node objects.
    ///
    /// Input (abbreviated):
    /// ```json
    /// {
    ///   "@context": {
    ///     "data": { "@id": "http://example.org/data", "@container": "@index" }
    ///   },
    ///   "data": {
    ///     "key1": { "@id": "http://example.org/node1" },
    ///     "key2": { "@id": "http://example.org/node2" }
    ///   }
    /// }
    /// ```
    ///
    /// Expected: two `IndexValue` events — `"key1"` and `"key2"`.
    #[test]
    fn test_index_container_emits_index_value_non_streaming() {
        // Build the JSON event stream manually.
        // Outer object { "@context": { ... }, "data": { ... } }
        let s = |v: &'static str| JsonEvent::String(Cow::Borrowed(v));
        let key = |k: &'static str| JsonEvent::ObjectKey(Cow::Borrowed(k));

        let events = vec![
            JsonEvent::StartObject,
            // @context
            key("@context"),
            JsonEvent::StartObject,
            key("data"),
            JsonEvent::StartObject,
            key("@id"),
            s("http://example.org/data"),
            key("@container"),
            s("@index"),
            JsonEvent::EndObject,
            JsonEvent::EndObject,
            // "data": { "key1": {...}, "key2": {...} }
            key("data"),
            JsonEvent::StartObject,
            key("key1"),
            JsonEvent::StartObject,
            key("@id"),
            s("http://example.org/node1"),
            JsonEvent::EndObject,
            key("key2"),
            JsonEvent::StartObject,
            key("@id"),
            s("http://example.org/node2"),
            JsonEvent::EndObject,
            JsonEvent::EndObject,
            JsonEvent::EndObject,
            JsonEvent::Eof,
        ];

        let results = expand_events(&events);
        let idx = index_values(&results);
        assert!(
            idx.contains(&"key1"),
            "Expected IndexValue('key1') in results, got: {idx:?}"
        );
        assert!(
            idx.contains(&"key2"),
            "Expected IndexValue('key2') in results, got: {idx:?}"
        );
        assert_eq!(
            idx.len(),
            2,
            "Expected exactly 2 IndexValue events, got: {idx:?}"
        );
    }

    /// Same test in streaming mode.
    #[test]
    fn test_index_container_emits_index_value_streaming() {
        let s = |v: &'static str| JsonEvent::String(Cow::Borrowed(v));
        let key = |k: &'static str| JsonEvent::ObjectKey(Cow::Borrowed(k));

        let events = vec![
            JsonEvent::StartObject,
            key("@context"),
            JsonEvent::StartObject,
            key("data"),
            JsonEvent::StartObject,
            key("@id"),
            s("http://example.org/data"),
            key("@container"),
            s("@index"),
            JsonEvent::EndObject,
            JsonEvent::EndObject,
            key("data"),
            JsonEvent::StartObject,
            key("alpha"),
            JsonEvent::StartObject,
            key("@id"),
            s("http://example.org/alpha"),
            JsonEvent::EndObject,
            key("beta"),
            JsonEvent::StartObject,
            key("@id"),
            s("http://example.org/beta"),
            JsonEvent::EndObject,
            JsonEvent::EndObject,
            JsonEvent::EndObject,
            JsonEvent::Eof,
        ];

        let results = expand_events_streaming(&events);
        let idx = index_values(&results);
        assert!(
            idx.contains(&"alpha"),
            "Expected IndexValue('alpha') in streaming results, got: {idx:?}"
        );
        assert!(
            idx.contains(&"beta"),
            "Expected IndexValue('beta') in streaming results, got: {idx:?}"
        );
        assert_eq!(
            idx.len(),
            2,
            "Expected exactly 2 IndexValue events in streaming mode, got: {idx:?}"
        );
    }

    /// Test that `IndexValue` events appear INSIDE the corresponding StartObject/EndObject pair.
    #[test]
    fn test_index_value_position_inside_object() {
        let s = |v: &'static str| JsonEvent::String(Cow::Borrowed(v));
        let key = |k: &'static str| JsonEvent::ObjectKey(Cow::Borrowed(k));

        let events = vec![
            JsonEvent::StartObject,
            key("@context"),
            JsonEvent::StartObject,
            key("items"),
            JsonEvent::StartObject,
            key("@id"),
            s("http://example.org/items"),
            key("@container"),
            s("@index"),
            JsonEvent::EndObject,
            JsonEvent::EndObject,
            key("items"),
            JsonEvent::StartObject,
            key("mykey"),
            JsonEvent::StartObject,
            key("@id"),
            s("http://example.org/x"),
            JsonEvent::EndObject,
            JsonEvent::EndObject,
            JsonEvent::EndObject,
            JsonEvent::Eof,
        ];

        let results = expand_events(&events);

        // Find the position of StartObject, IndexValue, and EndObject
        let start_pos = results
            .iter()
            .position(|e| matches!(e, JsonLdEvent::StartObject { .. }));
        let idx_pos = results
            .iter()
            .position(|e| matches!(e, JsonLdEvent::IndexValue(_)));
        let end_pos = results
            .iter()
            .position(|e| matches!(e, JsonLdEvent::EndObject));

        assert!(start_pos.is_some(), "StartObject expected in results");
        assert!(idx_pos.is_some(), "IndexValue expected in results");
        assert!(end_pos.is_some(), "EndObject expected in results");

        if let (Some(s), Some(i), Some(e)) = (start_pos, idx_pos, end_pos) {
            assert!(
                s < i,
                "StartObject must come before IndexValue (s={s}, i={i})"
            );
            assert!(
                i < e,
                "IndexValue must come before EndObject (i={i}, e={e})"
            );
        }
    }

    /// Test that `@index` as a plain property on a node object also emits `IndexValue`.
    #[test]
    fn test_inline_at_index_emits_index_value() {
        let s = |v: &'static str| JsonEvent::String(Cow::Borrowed(v));
        let key = |k: &'static str| JsonEvent::ObjectKey(Cow::Borrowed(k));

        // {"@index": "myAnnotation", "@id": "http://example.org/x"}
        let events = vec![
            JsonEvent::StartObject,
            key("@type"),
            s("http://example.org/Thing"),
            key("@index"),
            s("myAnnotation"),
            JsonEvent::EndObject,
            JsonEvent::Eof,
        ];

        let results = expand_events(&events);
        let idx = index_values(&results);
        assert!(
            idx.contains(&"myAnnotation"),
            "Expected IndexValue('myAnnotation') for inline @index, got: {idx:?}"
        );
    }
}
