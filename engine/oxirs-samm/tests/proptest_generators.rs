//! Property-based tests for SAMM code generators
//!
//! These tests verify that code generators produce valid output for randomly generated models.

use oxirs_samm::generators::{
    generate_graphql, generate_python, generate_sql, generate_typescript, PythonOptions,
    SqlDialect, TsOptions,
};
use oxirs_samm::metamodel::{Aspect, Characteristic, CharacteristicKind, ModelElement, Property};
use proptest::prelude::*;

/// Strategy for generating a valid Aspect with random properties
fn valid_aspect() -> impl Strategy<Value = Aspect> {
    (1usize..=10usize).prop_flat_map(|num_props| {
        (
            prop::string::string_regex(r"urn:samm:org\.[a-z]+:1\.0\.0#[A-Z][a-zA-Z0-9]{3,15}")
                .expect("valid regex"),
            prop::collection::vec(
                prop::string::string_regex(r"[A-Z][a-zA-Z0-9]{3,15}").expect("valid regex"),
                num_props..=num_props,
            ),
        )
            .prop_map(|(urn, prop_names)| {
                let mut aspect = Aspect::new(urn);
                aspect
                    .metadata
                    .add_preferred_name("en".to_string(), "Test Aspect".to_string());

                for (i, name) in prop_names.iter().enumerate() {
                    let prop_urn = format!("urn:samm:org.test:1.0.0#prop{}", i);
                    let char_urn = format!("urn:samm:org.test:1.0.0#Char{}", i);

                    let mut prop = Property::new(prop_urn).with_characteristic(
                        Characteristic::new(char_urn, CharacteristicKind::Trait)
                            .with_data_type("xsd:string".to_string()),
                    );

                    prop.metadata
                        .add_preferred_name("en".to_string(), name.clone());
                    aspect.add_property(prop);
                }

                aspect
            })
    })
}

proptest! {
    /// Test that TypeScript generator always produces valid output
    #[test]
    fn typescript_generator_produces_output(aspect in valid_aspect()) {
        let result = generate_typescript(&aspect, TsOptions::default());
        prop_assert!(result.is_ok());

        let output = result.unwrap();
        // Output should not be empty
        prop_assert!(!output.is_empty());
        // Output should contain interface keyword
        prop_assert!(output.contains("interface"));
    }

    /// Test that GraphQL generator always produces valid output
    #[test]
    fn graphql_generator_produces_output(aspect in valid_aspect()) {
        let result = generate_graphql(&aspect);
        prop_assert!(result.is_ok());

        let output = result.unwrap();
        // Output should not be empty
        prop_assert!(!output.is_empty());
        // Output should contain type keyword
        prop_assert!(output.contains("type"));
    }

    /// Test that Python generator always produces valid output
    #[test]
    fn python_generator_produces_output(aspect in valid_aspect()) {
        let result = generate_python(&aspect, PythonOptions::default());
        prop_assert!(result.is_ok());

        let output = result.unwrap();
        // Output should not be empty
        prop_assert!(!output.is_empty());
        // Output should contain class keyword
        prop_assert!(output.contains("class"));
    }

    /// Test that SQL generator always produces valid output
    #[test]
    fn sql_generator_produces_output(aspect in valid_aspect()) {
        let result = generate_sql(&aspect, SqlDialect::PostgreSql);
        prop_assert!(result.is_ok());

        let output = result.unwrap();
        // Output should not be empty
        prop_assert!(!output.is_empty());
        // Output should contain CREATE TABLE
        prop_assert!(output.contains("CREATE TABLE"));
    }

    /// Test that all generators handle aspects with varying property counts
    #[test]
    fn generators_scale_with_properties(
        prop_count in 1usize..=20usize,
    ) {
        let mut aspect = Aspect::new("urn:samm:org.test:1.0.0#TestAspect".to_string());
        aspect.metadata.add_preferred_name("en".to_string(), "Test".to_string());

        for i in 0..prop_count {
            let prop = Property::new(format!("urn:samm:org.test:1.0.0#prop{}", i))
                .with_characteristic(
                    Characteristic::new(
                        format!("urn:samm:org.test:1.0.0#Char{}", i),
                        CharacteristicKind::Trait
                    ).with_data_type("xsd:string".to_string())
                );
            aspect.add_property(prop);
        }

        // All generators should succeed
        prop_assert!(generate_typescript(&aspect, TsOptions::default()).is_ok());
        prop_assert!(generate_graphql(&aspect).is_ok());
        prop_assert!(generate_python(&aspect, PythonOptions::default()).is_ok());
        prop_assert!(generate_sql(&aspect, SqlDialect::PostgreSql).is_ok());
    }

    /// Test that TypeScript output contains all property names
    #[test]
    fn typescript_includes_all_properties(aspect in valid_aspect()) {
        let output = generate_typescript(&aspect, TsOptions::default()).unwrap();

        // Every property should appear in the output
        for prop in aspect.properties() {
            let prop_name = prop.name();
            prop_assert!(
                output.contains(&prop_name),
                "Output should contain property: {}",
                prop_name
            );
        }
    }

    /// Test that GraphQL output contains all property names
    #[test]
    fn graphql_includes_all_properties(aspect in valid_aspect()) {
        let output = generate_graphql(&aspect).unwrap();

        // Every property should appear in the output
        for prop in aspect.properties() {
            let prop_name = prop.name();
            prop_assert!(
                output.contains(&prop_name),
                "Output should contain property: {}",
                prop_name
            );
        }
    }

    /// Test that SQL output creates columns for all properties
    #[test]
    fn sql_includes_all_properties(aspect in valid_aspect()) {
        let output = generate_sql(&aspect, SqlDialect::PostgreSql).unwrap();

        // Every property should have a corresponding column
        for prop in aspect.properties() {
            // SQL uses snake_case, so convert property name
            let snake_name = prop.name()
                .chars()
                .flat_map(|c| {
                    if c.is_uppercase() {
                        vec!['_', c.to_ascii_lowercase()]
                    } else {
                        vec![c]
                    }
                })
                .collect::<String>()
                .trim_start_matches('_')
                .to_string();

            prop_assert!(
                output.to_lowercase().contains(&snake_name.to_lowercase()),
                "Output should contain column for property: {} (as {})",
                prop.name(),
                snake_name
            );
        }
    }
}
