//! Tests for the JSON-LD expansion algorithm sub-modules.
//!
//! Covers types, step helpers, and context-propagation logic.

#[cfg(test)]
mod tests {
    use crate::jsonld::expansion_algorithm::JsonLdExpansionConverter;
    use crate::jsonld::profile::JsonLdProcessingMode;

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
}
