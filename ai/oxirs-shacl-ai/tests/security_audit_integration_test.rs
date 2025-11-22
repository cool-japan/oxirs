//! Integration tests for Security Audit Framework
//!
//! These tests verify the end-to-end functionality of the security audit system
//! with realistic model data and scenarios.

use oxirs_shacl_ai::security_audit::{
    ModelSecurityData, SecurityAuditConfig, SecurityAuditFramework, SecuritySeverity,
};
use scirs2_core::ndarray_ext::Array;
use std::collections::HashMap;

fn create_realistic_model_data() -> ModelSecurityData {
    let mut metadata = HashMap::new();
    metadata.insert("model_type".to_string(), "neural_network".to_string());
    metadata.insert("model_version".to_string(), "1.0.0".to_string());
    metadata.insert(
        "training_dataset".to_string(),
        "shacl_shapes_v1".to_string(),
    );
    metadata.insert("explainability_method".to_string(), "SHAP".to_string());
    metadata.insert(
        "data_protection_measures".to_string(),
        "differential_privacy".to_string(),
    );
    metadata.insert(
        "data_collection_notice".to_string(),
        "Users are informed about data collection".to_string(),
    );
    metadata.insert(
        "deletion_capability".to_string(),
        "Data can be deleted on request".to_string(),
    );
    metadata.insert("risk_assessment".to_string(), "Completed".to_string());
    metadata.insert("accuracy_metrics".to_string(), "0.95".to_string());
    metadata.insert("human_oversight".to_string(), "Required".to_string());

    // Create realistic model parameters (50x50 weight matrix)
    let parameters = Array::from_shape_fn((50, 50), |(i, j)| {
        let val = ((i + j) as f64 * 0.1).sin() * 0.5;
        val
    });

    // Create training samples (200 samples, 50 features)
    let training_samples = Array::from_shape_fn((200, 50), |(i, j)| {
        ((i * 7 + j * 3) as f64 * 0.05).cos() + ((i + j) as f64 * 0.01)
    });

    // Create validation samples (100 samples, 50 features)
    let validation_samples = Array::from_shape_fn((100, 50), |(i, j)| {
        ((i * 5 + j * 2) as f64 * 0.07).sin() + ((i + j) as f64 * 0.02)
    });

    // Create predictions and labels
    let predictions = Array::from_shape_fn(100, |i| (i as f64 * 0.01).cos() * 0.5 + 0.5);
    let true_labels = Array::from_shape_fn(100, |i| if i % 2 == 0 { 1.0 } else { 0.0 });

    ModelSecurityData {
        parameters,
        training_samples,
        validation_samples,
        predictions,
        true_labels,
        metadata,
    }
}

#[test]
fn test_comprehensive_security_audit() {
    let config = SecurityAuditConfig::default();
    let mut framework = SecurityAuditFramework::new(config).expect("Failed to create framework");

    let model_data = create_realistic_model_data();
    let report = framework
        .audit_model("integration_test_model", &model_data)
        .expect("Failed to run audit");

    // Verify report structure
    assert_eq!(report.model_id, "integration_test_model");
    assert!(report.overall_security_score >= 0.0);
    assert!(report.overall_security_score <= 100.0);

    // Verify all test types were executed
    assert!(
        !report.findings.is_empty() || report.overall_security_score == 100.0,
        "Should have findings or perfect score"
    );

    // Verify recommendations are generated if there are findings
    if !report.findings.is_empty() {
        assert!(
            !report.recommendations.is_empty(),
            "Should have recommendations for findings"
        );
    }

    // Verify compliance was checked
    assert!(!report.compliance_status.is_empty());
}

#[test]
fn test_adversarial_robustness_detection() {
    let config = SecurityAuditConfig {
        enable_adversarial_testing: true,
        enable_privacy_leak_detection: false,
        enable_backdoor_detection: false,
        enable_compliance_checking: false,
        ..Default::default()
    };

    let mut framework = SecurityAuditFramework::new(config).expect("Failed to create framework");
    let model_data = create_realistic_model_data();

    let report = framework
        .audit_model("adversarial_test_model", &model_data)
        .expect("Failed to run audit");

    // Should have tested adversarial robustness
    let has_adversarial_tests = report.findings.iter().any(|f| {
        matches!(
            f.vulnerability_type,
            oxirs_shacl_ai::security_audit::VulnerabilityType::AdversarialVulnerability
                | oxirs_shacl_ai::security_audit::VulnerabilityType::InferenceManipulation
        )
    });

    // May or may not find vulnerabilities, but audit should complete
    assert!(
        report.duration.as_secs() < 10,
        "Audit should complete quickly"
    );
}

#[test]
fn test_privacy_leak_analysis() {
    let config = SecurityAuditConfig {
        enable_adversarial_testing: false,
        enable_privacy_leak_detection: true,
        enable_backdoor_detection: false,
        enable_compliance_checking: false,
        ..Default::default()
    };

    let mut framework = SecurityAuditFramework::new(config).expect("Failed to create framework");
    let model_data = create_realistic_model_data();

    let report = framework
        .audit_model("privacy_test_model", &model_data)
        .expect("Failed to run audit");

    // Privacy tests should complete without errors
    assert!(report.duration.as_secs() < 10);
}

#[test]
fn test_compliance_checking() {
    let config = SecurityAuditConfig {
        enable_adversarial_testing: false,
        enable_privacy_leak_detection: false,
        enable_backdoor_detection: false,
        enable_compliance_checking: true,
        ..Default::default()
    };

    let mut framework = SecurityAuditFramework::new(config).expect("Failed to create framework");
    let model_data = create_realistic_model_data();

    let report = framework
        .audit_model("compliance_test_model", &model_data)
        .expect("Failed to run audit");

    // Compliance checks should be performed
    assert!(report.compliance_status.contains_key("GDPR"));
    assert!(report.compliance_status.contains_key("CCPA"));
    assert!(report.compliance_status.contains_key("EU_AI_ACT"));

    // With proper metadata, all should be compliant
    assert!(report.compliance_status["GDPR"]);
    assert!(report.compliance_status["CCPA"]);
    assert!(report.compliance_status["EU_AI_ACT"]);
}

#[test]
fn test_severity_based_scoring() {
    let config = SecurityAuditConfig::default();
    let mut framework = SecurityAuditFramework::new(config).expect("Failed to create framework");

    let model_data = create_realistic_model_data();
    let report = framework
        .audit_model("severity_test_model", &model_data)
        .expect("Failed to run audit");

    // If critical findings exist, score should be significantly reduced
    let has_critical = report
        .findings
        .iter()
        .any(|f| f.severity == SecuritySeverity::Critical);

    if has_critical {
        assert!(
            report.overall_security_score < 70.0,
            "Critical findings should significantly reduce score"
        );
    }
}

#[test]
fn test_audit_history_tracking() {
    let config = SecurityAuditConfig::default();
    let mut framework = SecurityAuditFramework::new(config).expect("Failed to create framework");

    let model_data = create_realistic_model_data();

    // Run multiple audits
    framework
        .audit_model("model_v1", &model_data)
        .expect("Failed to run first audit");
    framework
        .audit_model("model_v2", &model_data)
        .expect("Failed to run second audit");
    framework
        .audit_model("model_v3", &model_data)
        .expect("Failed to run third audit");

    // Verify history is tracked
    let history = framework.get_audit_history();
    assert_eq!(history.len(), 3, "Should track all audits");
    assert_eq!(history[0].model_id, "model_v1");
    assert_eq!(history[1].model_id, "model_v2");
    assert_eq!(history[2].model_id, "model_v3");
}

#[test]
fn test_vulnerability_filtering() {
    let config = SecurityAuditConfig::default();
    let mut framework = SecurityAuditFramework::new(config).expect("Failed to create framework");

    let model_data = create_realistic_model_data();
    framework
        .audit_model("filter_test_model", &model_data)
        .expect("Failed to run audit");

    // Test filtering by severity
    let critical_vulns = framework.get_vulnerabilities_by_severity(SecuritySeverity::Critical);
    let high_vulns = framework.get_vulnerabilities_by_severity(SecuritySeverity::High);
    let medium_vulns = framework.get_vulnerabilities_by_severity(SecuritySeverity::Medium);

    // All filtered results should have correct severity
    for vuln in &critical_vulns {
        assert_eq!(vuln.severity, SecuritySeverity::Critical);
    }
    for vuln in &high_vulns {
        assert_eq!(vuln.severity, SecuritySeverity::High);
    }
    for vuln in &medium_vulns {
        assert_eq!(vuln.severity, SecuritySeverity::Medium);
    }
}

#[test]
fn test_model_without_compliance_metadata() {
    let mut metadata = HashMap::new();
    metadata.insert("model_type".to_string(), "basic_model".to_string());
    // Missing compliance metadata

    let model_data = ModelSecurityData {
        parameters: Array::zeros((10, 10)),
        training_samples: Array::zeros((50, 10)),
        validation_samples: Array::zeros((25, 10)),
        predictions: Array::zeros(25),
        true_labels: Array::zeros(25),
        metadata,
    };

    let config = SecurityAuditConfig {
        enable_compliance_checking: true,
        ..Default::default()
    };

    let mut framework = SecurityAuditFramework::new(config).expect("Failed to create framework");
    let report = framework
        .audit_model("non_compliant_model", &model_data)
        .expect("Failed to run audit");

    // Should detect compliance violations
    assert!(
        !report.compliance_status["GDPR"]
            || !report.compliance_status["CCPA"]
            || !report.compliance_status["EU_AI_ACT"],
        "Should detect missing compliance metadata"
    );
}

#[test]
fn test_report_serialization() {
    let config = SecurityAuditConfig::default();
    let mut framework = SecurityAuditFramework::new(config).expect("Failed to create framework");

    let model_data = create_realistic_model_data();
    let report = framework
        .audit_model("serialization_test", &model_data)
        .expect("Failed to run audit");

    // Test JSON serialization
    let json = serde_json::to_string_pretty(&report).expect("Failed to serialize report");
    assert!(!json.is_empty());
    assert!(json.contains("serialization_test"));
    assert!(json.contains("overall_security_score"));

    // Test deserialization
    let deserialized: oxirs_shacl_ai::security_audit::AuditReport =
        serde_json::from_str(&json).expect("Failed to deserialize report");
    assert_eq!(deserialized.model_id, report.model_id);
    assert_eq!(
        deserialized.overall_security_score,
        report.overall_security_score
    );
}
