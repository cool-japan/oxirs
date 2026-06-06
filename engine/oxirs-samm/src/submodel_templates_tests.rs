//! Submodel Templates Tests
//!
//! All `#[cfg(test)]` blocks for the submodel templates modules.

#[cfg(test)]
mod tests {
    use super::super::{
        ElementType, Multiplicity, SubmodelTemplate, TemplateCategory, TemplateConstraint,
        TemplateElement, TemplateRegistry, TemplateVersion, ValidationSeverity, ValueType,
    };
    use std::collections::HashMap;

    fn std_registry() -> TemplateRegistry {
        TemplateRegistry::with_standards()
    }

    // ── Registry basic tests ──────────────────────────────────────────────────

    #[test]
    fn test_registry_with_standards_count() {
        let reg = std_registry();
        assert_eq!(reg.count(), 8);
    }

    #[test]
    fn test_registry_new_empty() {
        let reg = TemplateRegistry::new();
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_registry_register_custom() {
        let mut reg = TemplateRegistry::new();
        reg.register(SubmodelTemplate {
            idta_id: "CUSTOM-001".to_string(),
            semantic_id: "https://example.org/CustomTemplate".to_string(),
            name: "Custom Template".to_string(),
            version: TemplateVersion::new(1, 0, 0),
            description: "A custom template".to_string(),
            required_elements: vec![],
            optional_elements: vec![],
            constraints: vec![],
            category: TemplateCategory::Other,
            tags: vec!["custom".to_string()],
        });
        assert_eq!(reg.count(), 1);
    }

    // ── Get by ID ─────────────────────────────────────────────────────────────

    #[test]
    fn test_get_by_semantic_id() {
        let reg = std_registry();
        let tmpl = reg
            .get("https://admin-shell.io/zvei/nameplate/2/0/Nameplate")
            .expect("should find nameplate");
        assert_eq!(tmpl.name, "Digital Nameplate");
    }

    #[test]
    fn test_get_by_idta_id() {
        let reg = std_registry();
        let tmpl = reg.get("IDTA 02006-2-0").expect("should find by IDTA ID");
        assert_eq!(tmpl.name, "Digital Nameplate");
    }

    #[test]
    fn test_get_by_name_alias() {
        let reg = std_registry();
        let tmpl = reg.get("digital nameplate").expect("should find by name");
        assert_eq!(tmpl.idta_id, "IDTA 02006-2-0");
    }

    #[test]
    fn test_get_not_found() {
        let reg = std_registry();
        assert!(reg.get("nonexistent").is_none());
    }

    // ── Version management ────────────────────────────────────────────────────

    #[test]
    fn test_get_specific_version() {
        let mut reg = TemplateRegistry::new();
        reg.register(SubmodelTemplate {
            idta_id: "T-1".to_string(),
            semantic_id: "https://ex.org/t".to_string(),
            name: "Test".to_string(),
            version: TemplateVersion::new(1, 0, 0),
            description: "v1".to_string(),
            required_elements: vec![],
            optional_elements: vec![],
            constraints: vec![],
            category: TemplateCategory::Other,
            tags: vec![],
        });
        reg.register(SubmodelTemplate {
            idta_id: "T-1".to_string(),
            semantic_id: "https://ex.org/t".to_string(),
            name: "Test".to_string(),
            version: TemplateVersion::new(2, 0, 0),
            description: "v2".to_string(),
            required_elements: vec![],
            optional_elements: vec![],
            constraints: vec![],
            category: TemplateCategory::Other,
            tags: vec![],
        });

        let v1 = reg
            .get_version("https://ex.org/t", &TemplateVersion::new(1, 0, 0))
            .expect("v1");
        assert_eq!(v1.description, "v1");

        let latest = reg.get("https://ex.org/t").expect("latest");
        assert_eq!(latest.description, "v2");
    }

    #[test]
    fn test_version_count() {
        let mut reg = TemplateRegistry::new();
        for i in 0..3 {
            reg.register(SubmodelTemplate {
                idta_id: "T".to_string(),
                semantic_id: "https://ex.org/t".to_string(),
                name: "Test".to_string(),
                version: TemplateVersion::new(i, 0, 0),
                description: format!("v{i}"),
                required_elements: vec![],
                optional_elements: vec![],
                constraints: vec![],
                category: TemplateCategory::Other,
                tags: vec![],
            });
        }
        assert_eq!(reg.version_count(), 3);
        assert_eq!(reg.count(), 1);
    }

    // ── Search ────────────────────────────────────────────────────────────────

    #[test]
    fn test_search_by_keyword() {
        let reg = std_registry();
        let results = reg.search("nameplate");
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_search_by_tag() {
        let reg = std_registry();
        let results = reg.search("sustainability");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Carbon Footprint");
    }

    #[test]
    fn test_search_by_idta_id() {
        let reg = std_registry();
        let results = reg.search("02008");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Time Series Data");
    }

    #[test]
    fn test_search_no_results() {
        let reg = std_registry();
        let results = reg.search("zzz_nonexistent_zzz");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_case_insensitive() {
        let reg = std_registry();
        let results = reg.search("CARBON");
        assert_eq!(results.len(), 1);
    }

    // ── By category ───────────────────────────────────────────────────────────

    #[test]
    fn test_by_category_identification() {
        let reg = std_registry();
        let results = reg.by_category(TemplateCategory::Identification);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Digital Nameplate");
    }

    #[test]
    fn test_by_category_sustainability() {
        let reg = std_registry();
        let results = reg.by_category(TemplateCategory::Sustainability);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_by_category_empty() {
        let reg = std_registry();
        let results = reg.by_category(TemplateCategory::SafetyCompliance);
        assert!(results.is_empty());
    }

    // ── Validation ────────────────────────────────────────────────────────────

    #[test]
    fn test_validate_valid_instance() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("ManufacturerName".to_string(), "Siemens AG".to_string());
        elements.insert(
            "ManufacturerProductDesignation".to_string(),
            "S7-1500".to_string(),
        );
        elements.insert("SerialNumber".to_string(), "SN-001".to_string());

        let result = reg
            .validate_instance("IDTA 02006-2-0", &elements)
            .expect("template found");
        assert!(result.is_valid);
        assert_eq!(result.matched_elements, 3);
        assert!(result.missing_required.is_empty());
    }

    #[test]
    fn test_validate_missing_required() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("ManufacturerName".to_string(), "Siemens AG".to_string());

        let result = reg
            .validate_instance("IDTA 02006-2-0", &elements)
            .expect("template found");
        assert!(!result.is_valid);
        assert_eq!(result.missing_required.len(), 2);
    }

    #[test]
    fn test_validate_with_optional_elements() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("ManufacturerName".to_string(), "Test".to_string());
        elements.insert(
            "ManufacturerProductDesignation".to_string(),
            "Prod".to_string(),
        );
        elements.insert("SerialNumber".to_string(), "SN-1".to_string());
        elements.insert("YearOfConstruction".to_string(), "2024".to_string());

        let result = reg
            .validate_instance("IDTA 02006-2-0", &elements)
            .expect("template found");
        assert!(result.is_valid);
        assert_eq!(result.matched_elements, 4);
    }

    #[test]
    fn test_validate_extra_elements_warning() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("ManufacturerName".to_string(), "Test".to_string());
        elements.insert(
            "ManufacturerProductDesignation".to_string(),
            "Prod".to_string(),
        );
        elements.insert("SerialNumber".to_string(), "SN-1".to_string());
        elements.insert("UnknownElement".to_string(), "extra".to_string());

        let result = reg
            .validate_instance("IDTA 02006-2-0", &elements)
            .expect("template found");
        assert!(result.is_valid);
        assert!(!result.warnings.is_empty());
        assert_eq!(result.extra_elements.len(), 1);
    }

    #[test]
    fn test_validate_enum_constraint_valid() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("EntryNode".to_string(), "root".to_string());
        elements.insert("ArcheType".to_string(), "FullBoM".to_string());

        let result = reg
            .validate_instance("IDTA 02011-1-0", &elements)
            .expect("found");
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_enum_constraint_invalid() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("EntryNode".to_string(), "root".to_string());
        elements.insert("ArcheType".to_string(), "InvalidType".to_string());

        let result = reg
            .validate_instance("IDTA 02011-1-0", &elements)
            .expect("found");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_numeric_range() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("ProductCarbonFootprint".to_string(), "data".to_string());
        elements.insert("CO2Equivalent".to_string(), "-5".to_string());

        let result = reg
            .validate_instance("IDTA 02023-0-9", &elements)
            .expect("found");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_string_length() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("SoftwareName".to_string(), "MyApp".to_string());
        elements.insert("SoftwareVersion".to_string(), String::new());

        let result = reg
            .validate_instance("IDTA 02005-1-0", &elements)
            .expect("found");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_nonexistent_template() {
        let reg = std_registry();
        let elements = HashMap::new();
        assert!(reg.validate_instance("nonexistent", &elements).is_none());
    }

    #[test]
    fn test_conformance_score() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("ManufacturerName".to_string(), "Test".to_string());

        let result = reg
            .validate_instance("IDTA 02006-2-0", &elements)
            .expect("found");
        assert!(result.conformance_score > 0.0);
        assert!(result.conformance_score < 1.0);
    }

    // ── TemplateVersion tests ─────────────────────────────────────────────────

    #[test]
    fn test_version_display() {
        let v = TemplateVersion::new(1, 2, 3);
        assert_eq!(format!("{v}"), "1.2.3");
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = TemplateVersion::new(1, 0, 0);
        let v12 = TemplateVersion::new(1, 2, 0);
        let v2 = TemplateVersion::new(2, 0, 0);
        assert!(v1.is_compatible_with(&v12));
        assert!(!v1.is_compatible_with(&v2));
    }

    #[test]
    fn test_version_ordering() {
        let v1 = TemplateVersion::new(1, 0, 0);
        let v12 = TemplateVersion::new(1, 2, 0);
        let v2 = TemplateVersion::new(2, 0, 0);
        assert!(v1 < v12);
        assert!(v12 < v2);
    }

    // ── Display tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_element_type_display() {
        assert_eq!(format!("{}", ElementType::Property), "Property");
        assert_eq!(
            format!("{}", ElementType::Collection),
            "SubmodelElementCollection"
        );
        assert_eq!(format!("{}", ElementType::Entity), "Entity");
    }

    #[test]
    fn test_value_type_display() {
        assert_eq!(format!("{}", ValueType::String), "xs:string");
        assert_eq!(format!("{}", ValueType::Integer), "xs:integer");
        assert_eq!(
            format!("{}", ValueType::Custom("custom".to_string())),
            "custom"
        );
    }

    #[test]
    fn test_multiplicity_display() {
        assert_eq!(format!("{}", Multiplicity::One), "[1]");
        assert_eq!(format!("{}", Multiplicity::ZeroOrOne), "[0..1]");
        assert_eq!(format!("{}", Multiplicity::ZeroOrMore), "[0..*]");
        assert_eq!(format!("{}", Multiplicity::OneOrMore), "[1..*]");
    }

    #[test]
    fn test_multiplicity_is_required() {
        assert!(Multiplicity::One.is_required());
        assert!(Multiplicity::OneOrMore.is_required());
        assert!(!Multiplicity::ZeroOrOne.is_required());
        assert!(!Multiplicity::ZeroOrMore.is_required());
    }

    #[test]
    fn test_category_display() {
        assert_eq!(
            format!("{}", TemplateCategory::Identification),
            "Identification"
        );
        assert_eq!(format!("{}", TemplateCategory::TimeSeries), "Time Series");
    }

    // ── List all ──────────────────────────────────────────────────────────────

    #[test]
    fn test_list_all() {
        let reg = std_registry();
        let all = reg.list_all();
        assert_eq!(all.len(), 8);
    }

    // ── Conditional required constraint ───────────────────────────────────────

    #[test]
    fn test_conditional_required_constraint() {
        let mut reg = TemplateRegistry::new();
        reg.register(SubmodelTemplate {
            idta_id: "T-COND".to_string(),
            semantic_id: "https://ex.org/cond".to_string(),
            name: "Conditional Test".to_string(),
            version: TemplateVersion::new(1, 0, 0),
            description: "Test conditional constraints".to_string(),
            required_elements: vec![],
            optional_elements: vec![
                TemplateElement {
                    id_short: "A".to_string(),
                    semantic_id: None,
                    element_type: ElementType::Property,
                    value_type: Some(ValueType::String),
                    description: "Element A".to_string(),
                    multiplicity: Multiplicity::ZeroOrOne,
                    children: vec![],
                    example_value: None,
                },
                TemplateElement {
                    id_short: "B".to_string(),
                    semantic_id: None,
                    element_type: ElementType::Property,
                    value_type: Some(ValueType::String),
                    description: "Element B".to_string(),
                    multiplicity: Multiplicity::ZeroOrOne,
                    children: vec![],
                    example_value: None,
                },
            ],
            constraints: vec![TemplateConstraint::ConditionalRequired {
                condition_element: "A".to_string(),
                required_element: "B".to_string(),
            }],
            category: TemplateCategory::Other,
            tags: vec![],
        });

        let mut elements = HashMap::new();
        elements.insert("A".to_string(), "value".to_string());
        let result = reg.validate_instance("T-COND", &elements).expect("found");
        assert!(!result.is_valid);

        elements.insert("B".to_string(), "value".to_string());
        let result = reg.validate_instance("T-COND", &elements).expect("found");
        assert!(result.is_valid);
    }

    // ── Pattern constraint ────────────────────────────────────────────────────

    #[test]
    fn test_pattern_constraint() {
        let mut reg = TemplateRegistry::new();
        reg.register(SubmodelTemplate {
            idta_id: "T-PAT".to_string(),
            semantic_id: "https://ex.org/pat".to_string(),
            name: "Pattern Test".to_string(),
            version: TemplateVersion::new(1, 0, 0),
            description: "Test pattern".to_string(),
            required_elements: vec![TemplateElement {
                id_short: "Email".to_string(),
                semantic_id: None,
                element_type: ElementType::Property,
                value_type: Some(ValueType::String),
                description: "Email".to_string(),
                multiplicity: Multiplicity::One,
                children: vec![],
                example_value: None,
            }],
            optional_elements: vec![],
            constraints: vec![TemplateConstraint::Pattern {
                element: "Email".to_string(),
                pattern: r"^[^@]+@[^@]+\.[^@]+$".to_string(),
            }],
            category: TemplateCategory::Other,
            tags: vec![],
        });

        let mut elements = HashMap::new();
        elements.insert("Email".to_string(), "user@example.com".to_string());
        let result = reg.validate_instance("T-PAT", &elements).expect("found");
        assert!(result.is_valid);

        elements.insert("Email".to_string(), "not-an-email".to_string());
        let result = reg.validate_instance("T-PAT", &elements).expect("found");
        assert!(!result.is_valid);
    }
}
