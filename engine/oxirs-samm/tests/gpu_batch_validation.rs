//! Integration tests for GPU batch validation.

use oxirs_samm::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};
use oxirs_samm::validation::batch::BatchValidator;

// Helper: a minimal valid aspect
fn valid_aspect(suffix: &str) -> Aspect {
    let aspect_urn = format!("urn:samm:org.example:1.0.0#{}", suffix);
    // Use the namespace (without fragment) for property/char URNs so that
    // property names (extracted as the fragment) start with lowercase.
    let ns = "urn:samm:org.example:1.0.0";
    let mut aspect = Aspect::new(aspect_urn);
    let char = Characteristic::new(
        format!("{}#speedChar", ns),
        CharacteristicKind::Measurement {
            unit: "unit:kilometre".to_string(),
        },
    )
    .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());
    // Property URN fragment "speed" satisfies camelCase.
    let prop = Property::new(format!("{}#speed", ns)).with_characteristic(char);
    aspect.add_property(prop);
    aspect
}

// Helper: an invalid aspect (no properties)
fn invalid_aspect(suffix: &str) -> Aspect {
    Aspect::new(format!("urn:samm:org.example:1.0.0#{}", suffix))
}

#[test]
fn test_batch_validator_cpu_basic() {
    let a1 = valid_aspect("A1");
    let a2 = valid_aspect("A2");
    let a3 = valid_aspect("A3");

    let reports = BatchValidator::new().validate_batch(&[&a1, &a2, &a3]);
    assert_eq!(reports.len(), 3, "should produce one report per aspect");
}

#[test]
fn test_batch_validator_empty_batch() {
    let reports = BatchValidator::new().validate_batch(&[]);
    assert!(reports.is_empty(), "empty input must yield empty output");
}

#[test]
fn test_batch_validator_cpu_mixed() {
    let valid = valid_aspect("GoodAspect");
    let invalid = invalid_aspect("BadAspect");

    let reports = BatchValidator::new().validate_batch(&[&valid, &invalid]);
    assert_eq!(reports.len(), 2);
    assert!(reports[0].is_valid, "first aspect (valid) should pass");
    assert!(
        !reports[1].is_valid,
        "second aspect (no properties) should fail"
    );
}

#[test]
fn test_batch_validator_with_gpu_false() {
    let a = valid_aspect("GpuFalse");

    let reports = BatchValidator::new().with_gpu(false).validate_batch(&[&a]);

    assert_eq!(reports.len(), 1);
    // CPU path should succeed for a valid aspect
    assert!(reports[0].is_valid);
}

#[test]
fn test_batch_validator_gpu_true_graceful_fallback() {
    // With gpu=true, the code should fall back to CPU gracefully
    // when the GPU backend is unavailable (compiled without `gpu` feature).
    let a1 = valid_aspect("GpuTrue1");
    let a2 = invalid_aspect("GpuTrue2");

    let reports = BatchValidator::new()
        .with_gpu(true)
        .validate_batch(&[&a1, &a2]);

    assert_eq!(
        reports.len(),
        2,
        "should still produce two reports after fallback"
    );
    assert!(reports[0].is_valid);
    assert!(!reports[1].is_valid);
}

#[test]
fn test_batch_validator_gpu_vs_cpu_same_result() {
    // GPU path (with fallback) and CPU path must produce the same validity
    // for the same aspects.
    let aspects: Vec<Aspect> = (0..5).map(|i| valid_aspect(&format!("Eq{}", i))).collect();
    let refs: Vec<&Aspect> = aspects.iter().collect();

    let cpu_reports = BatchValidator::new().with_gpu(false).validate_batch(&refs);

    let gpu_reports = BatchValidator::new().with_gpu(true).validate_batch(&refs);

    assert_eq!(cpu_reports.len(), gpu_reports.len());
    for (c, g) in cpu_reports.iter().zip(gpu_reports.iter()) {
        assert_eq!(
            c.is_valid, g.is_valid,
            "GPU fallback and CPU must agree on validity"
        );
    }
}

#[test]
fn test_batch_validator_single_invalid() {
    let inv = invalid_aspect("Solo");
    let reports = BatchValidator::new().validate_batch(&[&inv]);
    assert_eq!(reports.len(), 1);
    assert!(!reports[0].is_valid);
    assert!(!reports[0].errors.is_empty());
}

#[test]
fn test_batch_validator_default() {
    // Default::default() and BatchValidator::new() must behave the same.
    let a = valid_aspect("Default");
    let r1 = BatchValidator::new().validate_batch(&[&a]);
    let r2 = BatchValidator::default().validate_batch(&[&a]);
    assert_eq!(r1.len(), r2.len());
    assert_eq!(r1[0].is_valid, r2[0].is_valid);
}
