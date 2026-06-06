//! Unit tests for the Jena parity verifier.

use super::jena_parity::{FeatureStatus, JenaParityChecker, ParityCategory, ParityFeature};

fn checker() -> JenaParityChecker {
    JenaParityChecker::full_comparison()
}

#[test]
fn test_checker_is_non_empty() {
    assert!(!checker().all_features().is_empty());
}

#[test]
fn test_checker_has_minimum_features() {
    assert!(
        checker().all_features().len() >= 100,
        "Should track at least 100 features"
    );
}

#[test]
fn test_at_parity_non_empty() {
    assert!(!checker().at_parity().is_empty());
}

#[test]
fn test_beyond_jena_non_empty() {
    assert!(!checker().beyond_jena_features().is_empty());
}

#[test]
fn test_gaps_may_be_non_empty() {
    // Gaps are expected (partial/not-implemented Jena features)
    let binding = checker();
    let g = binding.gaps();
    assert!(
        !g.is_empty(),
        "There should be some gap features for realism"
    );
}

#[test]
fn test_query_language_category_exists() {
    assert!(!checker()
        .by_category(&ParityCategory::QueryLanguage)
        .is_empty());
}

#[test]
fn test_data_formats_category_exists() {
    assert!(!checker()
        .by_category(&ParityCategory::DataFormats)
        .is_empty());
}

#[test]
fn test_reasoning_category_exists() {
    assert!(!checker().by_category(&ParityCategory::Reasoning).is_empty());
}

#[test]
fn test_storage_category_exists() {
    assert!(!checker().by_category(&ParityCategory::Storage).is_empty());
}

#[test]
fn test_protocols_category_exists() {
    assert!(!checker().by_category(&ParityCategory::Protocols).is_empty());
}

#[test]
fn test_security_category_exists() {
    assert!(!checker().by_category(&ParityCategory::Security).is_empty());
}

#[test]
fn test_networking_category_exists() {
    assert!(!checker()
        .by_category(&ParityCategory::Networking)
        .is_empty());
}

#[test]
fn test_geospatial_category_exists() {
    assert!(!checker()
        .by_category(&ParityCategory::Geospatial)
        .is_empty());
}

#[test]
fn test_validation_category_exists() {
    assert!(!checker()
        .by_category(&ParityCategory::Validation)
        .is_empty());
}

#[test]
fn test_streaming_category_exists() {
    assert!(!checker().by_category(&ParityCategory::Streaming).is_empty());
}

#[test]
fn test_ai_category_exists() {
    assert!(!checker().by_category(&ParityCategory::Ai).is_empty());
}

#[test]
fn test_tooling_category_exists() {
    assert!(!checker().by_category(&ParityCategory::Tooling).is_empty());
}

#[test]
fn test_industrial_category_exists() {
    assert!(!checker()
        .by_category(&ParityCategory::Industrial)
        .is_empty());
}

#[test]
fn test_sparql_select_is_at_parity() {
    let c = checker();
    let f = c
        .all_features()
        .iter()
        .find(|f| f.name == "SPARQL 1.1 SELECT");
    assert!(f.is_some(), "SPARQL 1.1 SELECT should be tracked");
    assert!(f.unwrap().is_at_parity());
}

#[test]
fn test_sparql_ask_is_at_parity() {
    let c = checker();
    let f = c.all_features().iter().find(|f| f.name == "SPARQL 1.1 ASK");
    assert!(f.is_some());
    assert!(f.unwrap().is_at_parity());
}

#[test]
fn test_sparql_update_is_at_parity() {
    let c = checker();
    let f = c.all_features().iter().find(|f| f.name.contains("Update"));
    assert!(f.is_some());
    assert!(f.unwrap().is_at_parity());
}

#[test]
fn test_rdf_star_is_at_parity() {
    let c = checker();
    let f = c
        .all_features()
        .iter()
        .find(|f| f.name.contains("RDF-star"));
    assert!(f.is_some());
    assert!(f.unwrap().is_at_parity());
}

#[test]
fn test_owl2_rl_is_at_parity() {
    let c = checker();
    let f = c
        .all_features()
        .iter()
        .find(|f| f.name == "OWL 2 RL Reasoning");
    assert!(f.is_some());
    assert!(f.unwrap().is_at_parity());
}

#[test]
fn test_owl2_el_is_at_parity() {
    let c = checker();
    let f = c
        .all_features()
        .iter()
        .find(|f| f.name == "OWL 2 EL Reasoning");
    assert!(f.is_some());
    assert!(f.unwrap().is_at_parity());
}

#[test]
fn test_owl2_ql_is_parity() {
    let c = checker();
    let f = c
        .all_features()
        .iter()
        .find(|f| f.name == "OWL 2 QL Reasoning");
    assert!(f.is_some());
    assert!(
        f.unwrap().is_at_parity(),
        "OWL 2 QL Reasoning should be at parity after UCQ rewriting implementation"
    );
}

#[test]
fn test_graphql_is_beyond_jena() {
    let c = checker();
    let f = c.all_features().iter().find(|f| f.name == "GraphQL API");
    assert!(f.is_some());
    assert!(matches!(
        f.unwrap().oxirs_support,
        FeatureStatus::BeyondJena
    ));
}

#[test]
fn test_graphrag_is_beyond_jena() {
    let c = checker();
    let f = c
        .all_features()
        .iter()
        .find(|f| f.name.contains("GraphRAG"));
    assert!(f.is_some());
    assert!(matches!(
        f.unwrap().oxirs_support,
        FeatureStatus::BeyondJena
    ));
}

#[test]
fn test_vector_search_is_beyond_jena() {
    let c = checker();
    let f = c
        .all_features()
        .iter()
        .find(|f| f.name.contains("Vector") || f.name.contains("HNSW"));
    assert!(f.is_some());
    assert!(matches!(
        f.unwrap().oxirs_support,
        FeatureStatus::BeyondJena
    ));
}

#[test]
fn test_did_is_beyond_jena() {
    let c = checker();
    let f = c.all_features().iter().find(|f| f.name.contains("DID"));
    assert!(f.is_some());
    assert!(matches!(
        f.unwrap().oxirs_support,
        FeatureStatus::BeyondJena
    ));
}

#[test]
fn test_modbus_is_beyond_jena() {
    let c = checker();
    let f = c
        .all_features()
        .iter()
        .find(|f| f.name.contains("Modbus TCP"));
    assert!(f.is_some());
    assert!(matches!(
        f.unwrap().oxirs_support,
        FeatureStatus::BeyondJena
    ));
}

#[test]
fn test_canbus_is_beyond_jena() {
    let c = checker();
    let f = c.all_features().iter().find(|f| f.name.contains("CANbus"));
    assert!(f.is_some());
    assert!(matches!(
        f.unwrap().oxirs_support,
        FeatureStatus::BeyondJena
    ));
}

#[test]
fn test_tdb2_is_at_parity() {
    let c = checker();
    let f = c.all_features().iter().find(|f| f.name.contains("TDB2"));
    assert!(f.is_some());
    assert!(f.unwrap().is_at_parity());
}

#[test]
fn test_shacl_core_is_at_parity() {
    let c = checker();
    let f = c
        .all_features()
        .iter()
        .find(|f| f.name.contains("SHACL 1.0 Core"));
    assert!(f.is_some());
    assert!(f.unwrap().is_at_parity());
}

#[test]
fn test_geosparql_core_is_at_parity() {
    let c = checker();
    let f = c
        .all_features()
        .iter()
        .find(|f| f.name == "GeoSPARQL 1.1 Core");
    assert!(f.is_some());
    assert!(f.unwrap().is_at_parity());
}

#[test]
fn test_turtle_is_at_parity() {
    let c = checker();
    let f = c.all_features().iter().find(|f| f.name == "Turtle 1.1");
    assert!(f.is_some());
    assert!(f.unwrap().is_at_parity());
}

#[test]
fn test_json_ld_is_at_parity() {
    let c = checker();
    let f = c.all_features().iter().find(|f| f.name.contains("JSON-LD"));
    assert!(f.is_some());
    assert!(f.unwrap().is_at_parity());
}

#[test]
fn test_jena_parity_percentage_positive() {
    let s = checker().summary();
    assert!(s.jena_parity_percentage() > 0.0);
    assert!(s.jena_parity_percentage() <= 100.0);
}

#[test]
fn test_weighted_coverage_positive() {
    let c = checker();
    assert!(c.weighted_coverage_percentage() > 0.0);
    assert!(c.weighted_coverage_percentage() <= 100.0);
}

#[test]
fn test_beyond_jena_count_significant() {
    let s = checker().summary();
    assert!(
        s.beyond_jena >= 20,
        "OxiRS should have at least 20 beyond-Jena features"
    );
}

#[test]
fn test_summary_counts_sum_correctly() {
    let c = checker();
    let s = c.summary();
    assert_eq!(
        s.implemented + s.partial + s.not_implemented + s.beyond_jena,
        s.total_features
    );
}

#[test]
fn test_generate_report_non_empty() {
    assert!(!checker().generate_report().is_empty());
}

#[test]
fn test_report_contains_summary() {
    assert!(checker().generate_report().contains("## Summary"));
}

#[test]
fn test_report_contains_feature_details_header() {
    assert!(checker()
        .generate_report()
        .contains("## Feature Details by Category"));
}

#[test]
fn test_report_contains_ok_indicator() {
    assert!(checker().generate_report().contains("[OK]"));
}

#[test]
fn test_report_contains_beyond_indicator() {
    assert!(checker().generate_report().contains("[+]"));
}

#[test]
fn test_report_contains_partial_indicator() {
    assert!(checker().generate_report().contains("[~]"));
}

#[test]
fn test_report_contains_parity_percentage() {
    assert!(checker().generate_report().contains("parity"));
}

#[test]
fn test_feature_status_completion_percentages() {
    assert_eq!(FeatureStatus::Implemented.completion_percentage(), 100);
    assert_eq!(FeatureStatus::BeyondJena.completion_percentage(), 100);
    assert_eq!(FeatureStatus::NotImplemented.completion_percentage(), 0);
    assert_eq!(
        FeatureStatus::PartiallyImplemented { percentage: 75 }.completion_percentage(),
        75
    );
}

#[test]
fn test_feature_status_is_complete() {
    assert!(FeatureStatus::Implemented.is_complete());
    assert!(FeatureStatus::BeyondJena.is_complete());
    assert!(!FeatureStatus::NotImplemented.is_complete());
    assert!(!FeatureStatus::PartiallyImplemented { percentage: 75 }.is_complete());
}

#[test]
fn test_feature_status_labels() {
    assert_eq!(FeatureStatus::Implemented.label(), "Implemented");
    assert_eq!(FeatureStatus::BeyondJena.label(), "Beyond Jena");
    assert_eq!(FeatureStatus::NotImplemented.label(), "Not Implemented");
    assert_eq!(
        FeatureStatus::PartiallyImplemented { percentage: 50 }.label(),
        "Partial"
    );
}

#[test]
fn test_feature_status_indicators() {
    assert_eq!(FeatureStatus::Implemented.indicator(), "[OK]");
    assert_eq!(FeatureStatus::BeyondJena.indicator(), "[+]");
    assert_eq!(FeatureStatus::NotImplemented.indicator(), "[X]");
    assert_eq!(
        FeatureStatus::PartiallyImplemented { percentage: 50 }.indicator(),
        "[~]"
    );
}

#[test]
fn test_category_labels_non_empty() {
    let cats = [
        ParityCategory::QueryLanguage,
        ParityCategory::DataFormats,
        ParityCategory::Reasoning,
        ParityCategory::Security,
        ParityCategory::Streaming,
        ParityCategory::Ai,
        ParityCategory::Storage,
        ParityCategory::Networking,
        ParityCategory::Protocols,
        ParityCategory::Geospatial,
        ParityCategory::Validation,
        ParityCategory::Tooling,
        ParityCategory::Industrial,
    ];
    for cat in &cats {
        assert!(!cat.label().is_empty());
    }
}

#[test]
fn test_parity_builder() {
    let f = ParityFeature::parity("Test Feature", ParityCategory::Tooling, None);
    assert!(f.jena_support);
    assert!(matches!(f.oxirs_support, FeatureStatus::Implemented));
    assert!(f.is_at_parity());
}

#[test]
fn test_partial_builder() {
    let f = ParityFeature::partial(
        "Partial Feature",
        ParityCategory::Tooling,
        60,
        Some("60% done"),
    );
    assert!(f.jena_support);
    assert!(!f.is_at_parity());
    assert_eq!(f.oxirs_completion(), 60);
}

#[test]
fn test_missing_builder() {
    let f = ParityFeature::missing("Missing Feature", ParityCategory::Tooling, None);
    assert!(f.jena_support);
    assert!(!f.is_at_parity());
    assert_eq!(f.oxirs_completion(), 0);
}

#[test]
fn test_beyond_jena_builder() {
    let f = ParityFeature::beyond_jena("OxiRS Extra", ParityCategory::Tooling, Some("unique"));
    assert!(!f.jena_support);
    assert!(f.is_at_parity());
    assert!(matches!(f.oxirs_support, FeatureStatus::BeyondJena));
}

#[test]
fn test_default_checker_is_empty() {
    let c = JenaParityChecker::default();
    assert_eq!(c.all_features().len(), 0);
}

#[test]
fn test_register_custom_feature() {
    let mut c = JenaParityChecker::new();
    c.register(ParityFeature::parity(
        "Custom Feature",
        ParityCategory::Tooling,
        None,
    ));
    assert_eq!(c.all_features().len(), 1);
}

#[test]
fn test_kafka_is_beyond_jena() {
    let c = checker();
    let f = c.all_features().iter().find(|f| f.name.contains("Kafka"));
    assert!(f.is_some());
    assert!(matches!(
        f.unwrap().oxirs_support,
        FeatureStatus::BeyondJena
    ));
}

#[test]
fn test_wasm_is_beyond_jena() {
    let c = checker();
    let f = c.all_features().iter().find(|f| f.name.contains("WASM"));
    assert!(f.is_some());
    assert!(matches!(
        f.unwrap().oxirs_support,
        FeatureStatus::BeyondJena
    ));
}

#[test]
fn test_fuseki_rest_api_is_at_parity() {
    let c = checker();
    let f = c.all_features().iter().find(|f| f.name.contains("Fuseki"));
    assert!(f.is_some());
    assert!(f.unwrap().is_at_parity());
}
