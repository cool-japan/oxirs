//! Property-based tests for SAMM metadata
//!
//! These tests use proptest to automatically generate test cases and verify
//! invariants that should hold for all SAMM model elements.

use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, ElementMetadata, ModelElement, Property,
};
use proptest::prelude::*;

/// Strategy for generating valid URNs
fn valid_urn() -> impl Strategy<Value = String> {
    prop::string::string_regex(r"urn:samm:org\.[a-z]+:[0-9]+\.[0-9]+\.[0-9]+#[A-Z][a-zA-Z0-9]*")
        .expect("regex is valid")
}

/// Strategy for generating language codes
fn language_code() -> impl Strategy<Value = String> {
    prop::sample::select(vec!["en", "de", "fr", "es", "ja", "zh"]).prop_map(|s| s.to_string())
}

/// Strategy for generating non-empty text
fn non_empty_text() -> impl Strategy<Value = String> {
    prop::string::string_regex(r"[A-Z][a-zA-Z0-9 ]{3,50}").expect("regex is valid")
}

proptest! {
    /// Test that ElementMetadata preserves preferred names correctly
    #[test]
    fn element_metadata_preserves_preferred_names(
        urn in valid_urn(),
        lang in language_code(),
        name in non_empty_text(),
    ) {
        let mut metadata = ElementMetadata::new(urn.clone());
        metadata.add_preferred_name(lang.clone(), name.clone());

        // Retrieved name should match what was added
        prop_assert_eq!(metadata.get_preferred_name(&lang), Some(name.as_str()));
    }

    /// Test that ElementMetadata preserves descriptions correctly
    #[test]
    fn element_metadata_preserves_descriptions(
        urn in valid_urn(),
        lang in language_code(),
        description in non_empty_text(),
    ) {
        let mut metadata = ElementMetadata::new(urn.clone());
        metadata.add_description(lang.clone(), description.clone());

        // Retrieved description should match what was added
        prop_assert_eq!(metadata.get_description(&lang), Some(description.as_str()));
    }

    /// Test that ElementMetadata falls back to English correctly
    #[test]
    fn element_metadata_fallback_to_english(
        urn in valid_urn(),
        en_name in non_empty_text(),
    ) {
        let mut metadata = ElementMetadata::new(urn);
        metadata.add_preferred_name("en".to_string(), en_name.clone());

        // When requesting a non-existent language, should fallback to English
        prop_assert_eq!(metadata.get_preferred_name("xx"), Some(en_name.as_str()));
    }

    /// Test that Aspect name extraction works correctly
    #[test]
    fn aspect_name_extraction(
        namespace in prop::string::string_regex(r"urn:samm:org\.[a-z]+:[0-9]+\.[0-9]+\.[0-9]+").expect("regex"),
        name in prop::string::string_regex(r"[A-Z][a-zA-Z0-9]{3,20}").expect("regex"),
    ) {
        let urn = format!("{}#{}", namespace, name);
        let aspect = Aspect::new(urn.clone());

        // The extracted name should match the fragment after '#'
        prop_assert_eq!(aspect.name(), name);
        prop_assert_eq!(aspect.urn(), &urn);
    }

    /// Test that Property optional flag works correctly
    #[test]
    fn property_optional_flag(
        urn in valid_urn(),
    ) {
        let property = Property::new(urn.clone());
        // By default, properties should not be optional
        prop_assert!(!property.optional);

        let optional_property = Property::new(urn.clone()).as_optional();
        // After marking as optional, should be true
        prop_assert!(optional_property.optional);
    }

    /// Test that Property collection flag works correctly
    #[test]
    fn property_collection_flag(
        urn in valid_urn(),
    ) {
        let property = Property::new(urn.clone());
        // By default, properties should not be collections
        prop_assert!(!property.is_collection);

        let collection_property = Property::new(urn.clone()).as_collection();
        // After marking as collection, should be true
        prop_assert!(collection_property.is_collection);
    }

    /// Test that Property payload name works correctly
    #[test]
    fn property_payload_name(
        urn in valid_urn(),
        payload_name in prop::string::string_regex(r"[a-z_][a-z0-9_]{3,30}").expect("regex"),
    ) {
        let property = Property::new(urn.clone()).with_payload_name(payload_name.clone());

        // The effective name should be the payload name
        prop_assert_eq!(&property.effective_name(), &payload_name);
        prop_assert_eq!(property.payload_name.as_ref(), Some(&payload_name));
    }

    /// Test that URN format is validated correctly
    #[test]
    fn urn_format_validation(
        namespace in prop::string::string_regex(r"urn:samm:org\.[a-z]{3,10}:[0-9]+\.[0-9]+\.[0-9]+").expect("regex"),
        fragment in prop::string::string_regex(r"[A-Z][a-zA-Z0-9]{2,30}").expect("regex"),
    ) {
        let urn = format!("{}#{}", namespace, fragment);
        let metadata = ElementMetadata::new(urn.clone());

        // URN should be stored correctly
        prop_assert_eq!(&metadata.urn, &urn);

        // URN should contain the required prefix
        prop_assert!(metadata.urn.starts_with("urn:samm:org."));

        // URN should contain a '#' separator
        prop_assert!(metadata.urn.contains('#'));
    }

    /// Test that multiple language support works correctly
    #[test]
    fn multiple_languages(
        urn in valid_urn(),
        en_name in non_empty_text(),
        de_name in non_empty_text(),
        fr_name in non_empty_text(),
    ) {
        let mut metadata = ElementMetadata::new(urn);
        metadata.add_preferred_name("en".to_string(), en_name.clone());
        metadata.add_preferred_name("de".to_string(), de_name.clone());
        metadata.add_preferred_name("fr".to_string(), fr_name.clone());

        // Each language should retrieve its own name
        prop_assert_eq!(metadata.get_preferred_name("en"), Some(en_name.as_str()));
        prop_assert_eq!(metadata.get_preferred_name("de"), Some(de_name.as_str()));
        prop_assert_eq!(metadata.get_preferred_name("fr"), Some(fr_name.as_str()));
    }

    /// Test that see references are preserved
    #[test]
    fn see_references_preserved(
        urn in valid_urn(),
        url1 in prop::string::string_regex(r"https://example\.com/[a-z]{3,10}").expect("regex"),
        url2 in prop::string::string_regex(r"https://test\.org/[a-z]{3,10}").expect("regex"),
    ) {
        let mut metadata = ElementMetadata::new(urn);
        metadata.add_see_ref(url1.clone());
        metadata.add_see_ref(url2.clone());

        // See references should be in the order they were added
        prop_assert_eq!(metadata.see_refs.len(), 2);
        prop_assert_eq!(&metadata.see_refs[0], &url1);
        prop_assert_eq!(&metadata.see_refs[1], &url2);
    }
}

#[cfg(test)]
mod characteristic_proptests {
    use super::*;

    proptest! {
        /// Test that Characteristic data type is preserved
        #[test]
        fn characteristic_preserves_data_type(
            urn in valid_urn(),
            data_type in prop::sample::select(vec![
                "xsd:string", "xsd:integer", "xsd:float", "xsd:boolean", "xsd:dateTime"
            ]).prop_map(|s| s.to_string()),
        ) {
            let char = Characteristic::new(urn, CharacteristicKind::Trait)
                .with_data_type(data_type.clone());

            // Data type should be preserved
            prop_assert_eq!(char.data_type, Some(data_type));
        }

        /// Test that Characteristic kind is accessible
        #[test]
        fn characteristic_kind_accessible(
            urn in valid_urn(),
        ) {
            let char1 = Characteristic::new(urn.clone(), CharacteristicKind::Trait);
            prop_assert!(matches!(char1.kind(), CharacteristicKind::Trait));

            let char2 = Characteristic::new(urn.clone(), CharacteristicKind::Code);
            prop_assert!(matches!(char2.kind(), CharacteristicKind::Code));
        }
    }
}
