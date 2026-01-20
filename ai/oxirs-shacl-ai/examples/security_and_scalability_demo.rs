//! Comprehensive demonstration of Security Audit and Scalability Testing
//!
//! This example shows how to use the security audit framework and scalability
//! testing suite together to ensure your SHACL-AI models are both secure and
//! performant in production environments.

use oxirs_shacl_ai::scalability_testing::{
    ScalabilityTestConfig, ScalabilityTestingFramework, SlaThresholds, TestWorkload,
};
use oxirs_shacl_ai::security_audit::{
    ModelSecurityData, SecurityAuditConfig, SecurityAuditFramework, SecuritySeverity,
};
use scirs2_core::ndarray_ext::Array;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS SHACL-AI: Security & Scalability Testing Demo ===\n");

    // 1. Setup Model Data for Security Audit
    println!("📊 Setting up model data for security testing...");
    let model_data = create_production_model_data();
    println!("   ✓ Model with 100x100 parameters and 500 training samples\n");

    // 2. Run Security Audit
    println!("🔒 Running Security Audit...");
    let security_report = run_security_audit(&model_data)?;
    print_security_report(&security_report);

    // 3. Run Scalability Tests
    println!("\n⚡ Running Scalability Tests...");
    let scalability_report = run_scalability_tests()?;
    print_scalability_report(&scalability_report);

    // 4. Combined Analysis
    println!("\n📈 Combined Security & Scalability Analysis:");
    print_combined_analysis(&security_report, &scalability_report);

    println!("\n✅ Demo completed successfully!");
    Ok(())
}

/// Create realistic production model data
fn create_production_model_data() -> ModelSecurityData {
    let mut metadata = HashMap::new();
    metadata.insert("model_type".to_string(), "shacl_validator".to_string());
    metadata.insert("model_version".to_string(), "2.1.0".to_string());
    metadata.insert(
        "training_dataset".to_string(),
        "rdf_graphs_2025".to_string(),
    );
    metadata.insert("explainability_method".to_string(), "SHAP".to_string());
    metadata.insert(
        "data_protection_measures".to_string(),
        "differential_privacy_epsilon_1.0".to_string(),
    );
    metadata.insert(
        "data_collection_notice".to_string(),
        "Transparent data collection policy".to_string(),
    );
    metadata.insert(
        "deletion_capability".to_string(),
        "GDPR-compliant data deletion".to_string(),
    );
    metadata.insert(
        "risk_assessment".to_string(),
        "Low-risk AI system assessment completed".to_string(),
    );
    metadata.insert(
        "accuracy_metrics".to_string(),
        "F1=0.94, Precision=0.96, Recall=0.92".to_string(),
    );
    metadata.insert(
        "human_oversight".to_string(),
        "Human review required for critical decisions".to_string(),
    );

    // Create model parameters (100x100 weight matrix)
    let parameters = Array::from_shape_fn((100, 100), |(i, j)| {
        ((i as f64 * 0.1).sin() + (j as f64 * 0.1).cos()) * 0.3
    });

    // Create training samples (500 samples, 100 features)
    let training_samples = Array::from_shape_fn((500, 100), |(i, j)| {
        ((i * 7 + j * 3) as f64 * 0.03).cos() + ((i + j) as f64 * 0.005)
    });

    // Create validation samples (200 samples, 100 features)
    let validation_samples = Array::from_shape_fn((200, 100), |(i, j)| {
        ((i * 5 + j * 2) as f64 * 0.04).sin() + ((i + j) as f64 * 0.007)
    });

    // Create predictions and labels
    let predictions = Array::from_shape_fn(200, |i| {
        ((i as f64 * 0.02).cos() * 0.4 + 0.6).clamp(0.0, 1.0)
    });
    let true_labels = Array::from_shape_fn(200, |i| {
        if (i % 3 == 0) || (i % 5 == 0) {
            1.0
        } else {
            0.0
        }
    });

    ModelSecurityData {
        parameters,
        training_samples,
        validation_samples,
        predictions,
        true_labels,
        metadata,
    }
}

/// Run comprehensive security audit
fn run_security_audit(
    model_data: &ModelSecurityData,
) -> Result<oxirs_shacl_ai::security_audit::AuditReport, Box<dyn std::error::Error>> {
    let config = SecurityAuditConfig {
        enable_adversarial_testing: true,
        enable_privacy_leak_detection: true,
        enable_backdoor_detection: true,
        enable_compliance_checking: true,
        severity_threshold: SecuritySeverity::Medium,
        ..Default::default()
    };

    let mut framework = SecurityAuditFramework::new(config)?;
    let report = framework.audit_model("production_shacl_validator_v2", model_data)?;

    Ok(report)
}

/// Print security audit report
fn print_security_report(report: &oxirs_shacl_ai::security_audit::AuditReport) {
    println!("   Model: {}", report.model_id);
    println!(
        "   Security Score: {:.1}/100",
        report.overall_security_score
    );
    println!("   Audit Duration: {:?}", report.duration);
    println!(
        "   Status: {}",
        if report.passed {
            "✅ PASSED"
        } else {
            "⚠️  NEEDS ATTENTION"
        }
    );

    println!("\n   📋 Findings: {}", report.findings.len());
    for (i, finding) in report.findings.iter().enumerate().take(3) {
        println!(
            "      {}. [{:?}] {}",
            i + 1,
            finding.severity,
            finding.title
        );
    }
    if report.findings.len() > 3 {
        println!("      ... and {} more", report.findings.len() - 3);
    }

    println!("\n   🛡️  Compliance Status:");
    for (regulation, compliant) in &report.compliance_status {
        println!(
            "      {} {}",
            regulation,
            if *compliant { "✅" } else { "❌" }
        );
    }

    if !report.recommendations.is_empty() {
        println!("\n   💡 Top Recommendations:");
        for (i, rec) in report.recommendations.iter().enumerate().take(2) {
            println!("      {}. {}", i + 1, rec.title);
        }
    }
}

/// Run scalability tests
fn run_scalability_tests(
) -> Result<oxirs_shacl_ai::scalability_testing::ScalabilityTestReport, Box<dyn std::error::Error>>
{
    let sla_thresholds = SlaThresholds {
        max_latency_ms: 500.0,
        max_memory_mb: 2048.0,
        min_throughput_ops: 100.0,
        max_error_rate: 0.01,
        max_cpu_percent: 75.0,
    };

    let config = ScalabilityTestConfig {
        enable_load_testing: true,
        enable_stress_testing: true,
        enable_spike_testing: true,
        enable_endurance_testing: false, // Disabled for demo (takes too long)
        min_dataset_size: 100,
        max_dataset_size: 50_000,
        max_concurrent_users: 500,
        sla_thresholds,
        ..Default::default()
    };

    let mut framework = ScalabilityTestingFramework::new(config)?;

    let mut operation_mix = HashMap::new();
    operation_mix.insert("shape_validation".to_string(), 0.6);
    operation_mix.insert("constraint_checking".to_string(), 0.3);
    operation_mix.insert("pattern_matching".to_string(), 0.1);

    let workload = TestWorkload {
        description: "Production SHACL validation workload".to_string(),
        dataset_sizes: vec![100, 1_000, 5_000, 10_000],
        concurrent_users: vec![10, 50, 100, 200],
        operation_mix,
    };

    let report = framework.run_all_tests(&workload)?;

    Ok(report)
}

/// Print scalability test report
fn print_scalability_report(report: &oxirs_shacl_ai::scalability_testing::ScalabilityTestReport) {
    println!("   Workload: {}", report.workload_description);
    println!("   Scalability Score: {:.1}/100", report.scalability_score);
    println!("   Test Duration: {:?}", report.duration);

    if let Some(ref load_results) = report.load_test_results {
        println!("\n   📊 Load Test Results:");
        println!(
            "      Avg Latency: {:.2}ms (P95: {:.2}ms, P99: {:.2}ms)",
            load_results.avg_latency_ms, load_results.p95_latency_ms, load_results.p99_latency_ms
        );
        println!(
            "      Throughput: {:.1} ops/sec",
            load_results.throughput_ops
        );
        println!(
            "      Success Rate: {:.1}%",
            (load_results.successful_requests as f64 / load_results.total_requests as f64) * 100.0
        );
    }

    if let Some(ref stress_results) = report.stress_test_results {
        println!("\n   💪 Stress Test Results:");
        println!(
            "      Breaking Point: {} concurrent users",
            stress_results.breaking_point_users
        );
        println!(
            "      Peak Throughput: {:.1} ops/sec",
            stress_results.peak_throughput
        );
        println!(
            "      Recovery Time: {:.0}ms",
            stress_results.recovery_time_ms
        );
    }

    if let Some(ref spike_results) = report.spike_test_results {
        println!("\n   ⚡ Spike Test Results:");
        println!(
            "      Max Latency During Spike: {:.2}ms",
            spike_results.max_latency_during_spike_ms
        );
        println!(
            "      Recovery: {}",
            if spike_results.recovery_successful {
                "✅ Successful"
            } else {
                "⚠️  Failed"
            }
        );
    }

    println!("\n   ✅ SLA Compliance:");
    println!(
        "      Overall: {}",
        if report.sla_compliance.overall_compliant {
            "✅ COMPLIANT"
        } else {
            "⚠️  VIOLATIONS DETECTED"
        }
    );
    if !report.sla_compliance.violations.is_empty() {
        println!("      Violations:");
        for violation in report.sla_compliance.violations.iter().take(3) {
            println!("         - {}", violation);
        }
    }

    if !report.bottlenecks_identified.is_empty() {
        println!(
            "\n   🔍 Bottlenecks Identified: {}",
            report.bottlenecks_identified.len()
        );
        for (i, bottleneck) in report.bottlenecks_identified.iter().enumerate().take(2) {
            println!(
                "      {}. {} ({:?})",
                i + 1,
                bottleneck.description,
                bottleneck.severity
            );
        }
    }
}

/// Print combined analysis
fn print_combined_analysis(
    security_report: &oxirs_shacl_ai::security_audit::AuditReport,
    scalability_report: &oxirs_shacl_ai::scalability_testing::ScalabilityTestReport,
) {
    let security_score = security_report.overall_security_score;
    let scalability_score = scalability_report.scalability_score;
    let combined_score = (security_score + scalability_score) / 2.0;

    println!(
        "   Combined Production Readiness Score: {:.1}/100",
        combined_score
    );

    let readiness = if combined_score >= 85.0 {
        "✅ PRODUCTION READY"
    } else if combined_score >= 70.0 {
        "⚠️  READY WITH MINOR IMPROVEMENTS"
    } else if combined_score >= 50.0 {
        "⚠️  NEEDS IMPROVEMENT"
    } else {
        "❌ NOT PRODUCTION READY"
    };

    println!("   Production Readiness: {}", readiness);

    println!("\n   📝 Key Recommendations:");

    // Security recommendations
    if security_score < 80.0 {
        println!("      • Address security vulnerabilities before deployment");
    }
    if !security_report.compliance_status.values().all(|&v| v) {
        println!("      • Ensure full regulatory compliance (GDPR, CCPA, EU AI Act)");
    }

    // Scalability recommendations
    if scalability_score < 80.0 {
        println!("      • Optimize performance to meet production SLAs");
    }
    if !scalability_report.sla_compliance.overall_compliant {
        println!("      • Address SLA violations before scaling to production");
    }
    if !scalability_report.bottlenecks_identified.is_empty() {
        println!(
            "      • Resolve {} performance bottleneck(s)",
            scalability_report.bottlenecks_identified.len()
        );
    }

    println!("\n   🎯 Next Steps:");
    if combined_score >= 85.0 {
        println!("      1. Proceed with production deployment");
        println!("      2. Set up continuous monitoring");
        println!("      3. Schedule regular security audits");
    } else {
        println!("      1. Review and address all findings");
        println!("      2. Re-run tests after improvements");
        println!("      3. Validate fixes in staging environment");
    }
}
