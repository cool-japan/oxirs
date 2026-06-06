//! Tests for the JSON-LD expansion module.

#[cfg(test)]
mod tests {
    use crate::jsonld::expansion::JsonLdExpansionConverter;
    use crate::jsonld::expansion::JsonLdValue;
    use crate::jsonld::profile::JsonLdProcessingMode;

    #[test]
    fn test_expansion_converter_creation() {
        let converter =
            JsonLdExpansionConverter::new(None, false, false, JsonLdProcessingMode::JsonLd1_1);
        assert!(!converter.is_end());
    }

    #[test]
    fn test_expansion_converter_streaming_mode() {
        let converter =
            JsonLdExpansionConverter::new(None, true, false, JsonLdProcessingMode::JsonLd1_1);
        assert!(!converter.is_end());
    }

    #[test]
    fn test_expansion_converter_lenient_mode() {
        let converter =
            JsonLdExpansionConverter::new(None, false, true, JsonLdProcessingMode::JsonLd1_1);
        assert!(!converter.is_end());
    }
}
