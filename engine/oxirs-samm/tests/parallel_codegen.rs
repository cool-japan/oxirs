//! Integration tests for parallel code generation.

use oxirs_samm::generators::parallel::ParallelGenerator;
use oxirs_samm::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};

/// A minimal aspect with one valid property.
fn minimal_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
    let char = Characteristic::new(
        "urn:samm:org.example:1.0.0#SpeedChar".to_string(),
        CharacteristicKind::Measurement {
            unit: "unit:kilometrePerHour".to_string(),
        },
    )
    .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());
    let prop =
        Property::new("urn:samm:org.example:1.0.0#speed".to_string()).with_characteristic(char);
    aspect.add_property(prop);
    aspect
}

/// An aspect with no properties.
fn empty_aspect() -> Aspect {
    Aspect::new("urn:samm:org.example:1.0.0#EmptyAspect".to_string())
}

#[test]
fn test_parallel_generate_all() {
    let aspect = minimal_aspect();
    let result = ParallelGenerator::new().generate_all(&aspect);

    // We expect exactly 10 generators to have run.
    assert_eq!(
        result.outputs.len(),
        10,
        "all 10 generators should be present"
    );

    // Check that all keys are present.
    let expected_keys = [
        "aas",
        "dtdl",
        "graphql",
        "java",
        "jsonld",
        "payload",
        "python",
        "scala",
        "sql",
        "typescript",
    ];
    for key in &expected_keys {
        assert!(
            result.outputs.contains_key(key),
            "missing generator '{}'",
            key
        );
    }

    // Every result must be Ok or Err(String) — never panic.
    for (name, output) in &result.outputs {
        match output {
            Ok(code) => {
                assert!(
                    !code.is_empty(),
                    "generator '{}' produced empty output",
                    name
                );
            }
            Err(e) => {
                eprintln!("generator '{}' returned error (acceptable): {}", name, e);
            }
        }
    }
}

#[test]
fn test_parallel_vs_sequential_consistency() {
    // Compare parallel output with sequential runs for deterministic generators.
    // (Excludes `payload` which uses an RNG.)
    use oxirs_samm::generators::{
        generate_aas, generate_dtdl, generate_graphql, generate_java, generate_jsonld,
        generate_scala, generate_sql, generate_typescript, AasFormat, JavaOptions, ScalaOptions,
        SqlDialect, TsOptions,
    };

    let aspect = minimal_aspect();
    let parallel_result = ParallelGenerator::new().generate_deterministic(&aspect);

    let sequential: std::collections::BTreeMap<&'static str, Result<String, String>> = [
        (
            "graphql",
            generate_graphql(&aspect).map_err(|e| e.to_string()),
        ),
        (
            "typescript",
            generate_typescript(&aspect, TsOptions::default()).map_err(|e| e.to_string()),
        ),
        (
            "java",
            generate_java(&aspect, JavaOptions::default()).map_err(|e| e.to_string()),
        ),
        (
            "scala",
            generate_scala(&aspect, ScalaOptions::default()).map_err(|e| e.to_string()),
        ),
        (
            "sql",
            generate_sql(&aspect, SqlDialect::PostgreSql).map_err(|e| e.to_string()),
        ),
        (
            "jsonld",
            generate_jsonld(&aspect).map_err(|e| e.to_string()),
        ),
        ("dtdl", generate_dtdl(&aspect).map_err(|e| e.to_string())),
        (
            "aas",
            generate_aas(&aspect, AasFormat::Json).map_err(|e| e.to_string()),
        ),
    ]
    .into_iter()
    .collect();

    for (name, seq_output) in &sequential {
        let par_output = parallel_result
            .outputs
            .get(*name)
            .expect("parallel result must contain this generator");

        match (seq_output, par_output) {
            (Ok(seq), Ok(par)) => {
                assert_eq!(
                    seq, par,
                    "generator '{}': parallel and sequential output differ",
                    name
                );
            }
            (Err(_), Err(_)) => {
                // Both errored — that's consistent.
            }
            _ => {
                panic!("generator '{}': one succeeded and the other failed", name);
            }
        }
    }
}

#[test]
fn test_parallel_empty_aspect() {
    // An aspect with no properties should not panic; generators may return
    // errors but must not panic.
    let aspect = empty_aspect();
    let result = ParallelGenerator::new().generate_all(&aspect);

    assert_eq!(result.outputs.len(), 10, "should still have 10 entries");

    // None of the generators should have panicked (ensured by reaching here).
    for (name, output) in &result.outputs {
        match output {
            Ok(code) => {
                // Empty aspects may produce minimal valid output
                let _ = code;
            }
            Err(e) => {
                eprintln!(
                    "generator '{}' errored for empty aspect (acceptable): {}",
                    name, e
                );
            }
        }
    }
}

#[test]
fn test_parallel_result_sorted() {
    // The outputs BTreeMap must be sorted alphabetically.
    let aspect = minimal_aspect();
    let result = ParallelGenerator::new().generate_all(&aspect);

    let keys: Vec<&&str> = result.outputs.keys().collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "outputs must be sorted alphabetically");
}

#[test]
fn test_parallel_generator_default() {
    let aspect = minimal_aspect();
    let r1 = ParallelGenerator::new().generate_all(&aspect);
    let r2 = ParallelGenerator.generate_all(&aspect);
    // Key sets must be identical.
    assert_eq!(r1.outputs.len(), r2.outputs.len());
    let k1: Vec<_> = r1.outputs.keys().collect();
    let k2: Vec<_> = r2.outputs.keys().collect();
    assert_eq!(k1, k2);
}
