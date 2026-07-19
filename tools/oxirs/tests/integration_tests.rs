use oxirs::{run, BenchmarkAction, Cli, Commands, ConfigAction};
use tempfile::tempdir;

/// Build a `Cli` wrapping `command` with default global options.
fn cli(command: Commands) -> Cli {
    Cli {
        command,
        verbose: false,
        config: None,
        quiet: false,
        no_color: false,
        interactive: false,
        profile: None,
        completion: None,
    }
}

#[tokio::test]
async fn test_config_init_command() {
    let temp_dir = tempdir().unwrap();
    let config_path = temp_dir.path().join("test_config.toml");

    let cli = Cli {
        command: Commands::Config {
            action: ConfigAction::Init {
                output: config_path.clone(),
            },
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: false,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    assert!(result.is_ok(), "Config init should succeed");
    assert!(config_path.exists(), "Config file should be created");

    // Check that the config file has expected content
    let content = std::fs::read_to_string(&config_path).unwrap();
    assert!(
        content.contains("[server]"),
        "Config should contain server section"
    );
    assert!(
        content.contains("[datasets]"),
        "Config should contain datasets section"
    );
}

#[tokio::test]
async fn test_config_validate_command() {
    let temp_dir = tempdir().unwrap();
    let config_path = temp_dir.path().join("test_config.toml");

    // First create a config file
    let config_content = r#"
[server]
host = "localhost"
port = 3030

[datasets]
"#;
    std::fs::write(&config_path, config_content).unwrap();

    let cli = Cli {
        command: Commands::Config {
            action: ConfigAction::Validate {
                config: config_path,
            },
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: false,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    assert!(
        result.is_ok(),
        "Config validation should succeed for valid config"
    );
}

#[tokio::test]
async fn test_init_command_memory() {
    let temp_dir = tempdir().unwrap();
    let dataset_path = temp_dir.path().join("test_dataset");

    let cli = Cli {
        command: Commands::Init {
            name: "test_dataset".to_string(),
            format: "memory".to_string(),
            location: Some(dataset_path.clone()),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: false,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    assert!(
        result.is_ok(),
        "Init command should succeed for memory format"
    );
}

#[tokio::test]
async fn test_init_command_tdb2() {
    let temp_dir = tempdir().unwrap();
    let dataset_path = temp_dir.path().join("test_dataset_tdb2");

    let cli = Cli {
        command: Commands::Init {
            name: "test_dataset_tdb2".to_string(),
            format: "tdb2".to_string(),
            location: Some(dataset_path.clone()),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: false,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    assert!(
        result.is_ok(),
        "Init command should succeed for tdb2 format"
    );
    assert!(dataset_path.exists(), "Dataset directory should be created");

    let config_path = dataset_path.join("oxirs.toml");
    assert!(
        config_path.exists(),
        "Dataset config file should be created"
    );
}

#[tokio::test]
async fn test_init_command_invalid_format() {
    let temp_dir = tempdir().unwrap();
    let dataset_path = temp_dir.path().join("test_dataset_invalid");

    let cli = Cli {
        command: Commands::Init {
            name: "test_dataset_invalid".to_string(),
            format: "invalid_format".to_string(),
            location: Some(dataset_path),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: false,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    assert!(
        result.is_err(),
        "Init command should fail for invalid format"
    );
}

#[tokio::test]
async fn test_export_command_basic() {
    let temp_dir = tempdir().unwrap();
    let output_path = temp_dir.path().join("output.ttl");

    let cli = Cli {
        command: Commands::Export {
            dataset: "nonexistent".to_string(),
            file: output_path,
            format: "turtle".to_string(),
            graph: None,
            resume: false,
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: false,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    // Should fail because dataset doesn't exist
    assert!(
        result.is_err(),
        "Export should fail for nonexistent dataset"
    );
}

#[tokio::test]
async fn test_query_command_basic() {
    let cli = Cli {
        command: Commands::Query {
            dataset: "nonexistent".to_string(),
            query: "SELECT * WHERE { ?s ?p ?o } LIMIT 10".to_string(),
            file: false,
            output: "table".to_string(),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: false,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    // Should fail because dataset doesn't exist
    assert!(result.is_err(), "Query should fail for nonexistent dataset");
}

#[tokio::test]
async fn test_benchmark_command_basic() {
    let cli = Cli {
        command: Commands::Benchmark {
            action: BenchmarkAction::Run {
                dataset: "nonexistent".to_string(),
                suite: "sp2bench".to_string(),
                iterations: 1,
                output: None,
                detailed: false,
                warmup: 0,
            },
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: false,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    // Should fail because dataset doesn't exist
    assert!(
        result.is_err(),
        "Benchmark should fail for nonexistent dataset"
    );
}

#[tokio::test]
async fn test_benchmark_invalid_suite() {
    let cli = Cli {
        command: Commands::Benchmark {
            action: BenchmarkAction::Run {
                dataset: "test".to_string(),
                suite: "invalid_suite".to_string(),
                iterations: 1,
                output: None,
                detailed: false,
                warmup: 0,
            },
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: false,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    // Should fail because suite is invalid
    assert!(result.is_err(), "Benchmark should fail for invalid suite");
}

// ─────────────────────────────────────────────────────────────────────────────
// G3: newly-wired command integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_lint_command_clean_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("clean.ttl");
    std::fs::write(
        &path,
        "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .\n",
    )
    .unwrap();

    let result = run(cli(Commands::Lint {
        file: path,
        format: "turtle".to_string(),
        max_literal_length: 200,
        strict: true,
    }))
    .await;
    assert!(
        result.is_ok(),
        "clean turtle should lint without error: {result:?}"
    );
}

#[tokio::test]
async fn test_lint_command_strict_flags_errors() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dirty.ttl");
    // `foaf:` is used but never declared -> error-severity issue.
    std::fs::write(&path, "foaf:Person a foaf:Class .\n").unwrap();

    let result = run(cli(Commands::Lint {
        file: path,
        format: "turtle".to_string(),
        max_literal_length: 200,
        strict: true,
    }))
    .await;
    assert!(
        result.is_err(),
        "strict lint must fail when error-severity issues exist"
    );
}

#[tokio::test]
async fn test_lint_command_missing_file_fails() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("does_not_exist.ttl");

    let result = run(cli(Commands::Lint {
        file: path,
        format: "turtle".to_string(),
        max_literal_length: 200,
        strict: false,
    }))
    .await;
    assert!(result.is_err(), "lint on a missing file must fail loudly");
}

#[tokio::test]
async fn test_merge_command_two_files_writes_output() {
    let dir = tempdir().unwrap();
    let a = dir.path().join("a.ttl");
    let b = dir.path().join("b.nt");
    let out = dir.path().join("merged.ttl");
    std::fs::write(
        &a,
        "@prefix ex: <http://example.org/> .\nex:s1 ex:p ex:o1 .\n",
    )
    .unwrap();
    std::fs::write(
        &b,
        "<http://example.org/s2> <http://example.org/p> <http://example.org/o2> .\n",
    )
    .unwrap();

    let result = run(cli(Commands::Merge {
        inputs: vec![a, b],
        output: Some(out.clone()),
        mode: "set-union".to_string(),
        format: "turtle".to_string(),
        dry_run: false,
        provenance: false,
    }))
    .await;
    assert!(result.is_ok(), "merge should succeed: {result:?}");
    assert!(out.exists(), "merge output file should be written");
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("@prefix"), "turtle output expected");
}

#[tokio::test]
async fn test_merge_command_dry_run_writes_nothing() {
    let dir = tempdir().unwrap();
    let a = dir.path().join("a.ttl");
    let out = dir.path().join("should_not_exist.ttl");
    std::fs::write(
        &a,
        "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .\n",
    )
    .unwrap();

    let result = run(cli(Commands::Merge {
        inputs: vec![a],
        output: Some(out.clone()),
        mode: "set-union".to_string(),
        format: "ntriples".to_string(),
        dry_run: true,
        provenance: false,
    }))
    .await;
    assert!(result.is_ok(), "dry-run merge should succeed: {result:?}");
    assert!(!out.exists(), "dry-run must not write an output file");
}

#[tokio::test]
async fn test_merge_command_unparseable_input_fails() {
    let dir = tempdir().unwrap();
    let a = dir.path().join("garbage.nt");
    std::fs::write(&a, "this is not valid n-triples @@@\n").unwrap();

    let result = run(cli(Commands::Merge {
        inputs: vec![a],
        output: None,
        mode: "set-union".to_string(),
        format: "turtle".to_string(),
        dry_run: false,
        provenance: false,
    }))
    .await;
    assert!(result.is_err(), "unparseable input must fail loudly");
}

#[tokio::test]
async fn test_jena_parity_text() {
    let result = run(cli(Commands::JenaParity {
        format: "text".to_string(),
    }))
    .await;
    assert!(
        result.is_ok(),
        "jena-parity text should succeed: {result:?}"
    );
}

#[tokio::test]
async fn test_jena_parity_markdown() {
    let result = run(cli(Commands::JenaParity {
        format: "markdown".to_string(),
    }))
    .await;
    assert!(result.is_ok(), "jena-parity markdown should succeed");
}

#[tokio::test]
async fn test_jena_parity_json() {
    let result = run(cli(Commands::JenaParity {
        format: "json".to_string(),
    }))
    .await;
    assert!(result.is_ok(), "jena-parity json should succeed");
}

#[tokio::test]
async fn test_jena_parity_unknown_format_fails() {
    let result = run(cli(Commands::JenaParity {
        format: "yaml".to_string(),
    }))
    .await;
    assert!(result.is_err(), "unknown parity format must fail loudly");
}

#[tokio::test]
async fn test_monitor_rejects_bad_scheme() {
    // Validation only — no network is contacted.
    let result = run(cli(Commands::Monitor {
        endpoint: "ftp://example.org/sparql".to_string(),
        interval: 1,
        count: 1,
        timeout: 5,
        threshold: 5_000,
    }))
    .await;
    assert!(result.is_err(), "non-http endpoint must be rejected");
}

#[tokio::test]
async fn test_monitor_rejects_zero_count() {
    // Validation only — no network is contacted.
    let result = run(cli(Commands::Monitor {
        endpoint: "http://localhost:3030/sparql".to_string(),
        interval: 1,
        count: 0,
        timeout: 5,
        threshold: 5_000,
    }))
    .await;
    assert!(result.is_err(), "zero probe count must be rejected");
}

#[tokio::test]
async fn test_detect_format_command_turtle() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("sample.ttl");
    std::fs::write(
        &path,
        "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .\n",
    )
    .unwrap();

    let result = run(cli(Commands::DetectFormat {
        file: path,
        output: None,
    }))
    .await;
    assert!(
        result.is_ok(),
        "format detection should succeed: {result:?}"
    );
}

#[tokio::test]
async fn test_detect_format_command_missing_file_fails() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nope.unknownext");

    let result = run(cli(Commands::DetectFormat {
        file: path,
        output: None,
    }))
    .await;
    assert!(
        result.is_err(),
        "detecting a missing/unknown file must fail loudly"
    );
}

#[tokio::test]
async fn test_serve_dry_run_valid_config() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("oxirs.toml");
    std::fs::write(
        &config_path,
        "[server]\nhost = \"localhost\"\nport = 3030\n",
    )
    .unwrap();

    let result = run(cli(Commands::Serve {
        config: config_path,
        port: 3030,
        host: "localhost".to_string(),
        graphql: false,
        dry_run: true,
    }))
    .await;
    assert!(
        result.is_ok(),
        "valid serve dry-run should succeed without binding: {result:?}"
    );
}

#[tokio::test]
async fn test_serve_dry_run_invalid_port_fails() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("oxirs.toml");
    std::fs::write(&config_path, "[server]\n").unwrap();

    let result = run(cli(Commands::Serve {
        config: config_path,
        port: 0,
        host: "localhost".to_string(),
        graphql: false,
        dry_run: true,
    }))
    .await;
    assert!(
        result.is_err(),
        "serve dry-run must reject an invalid (0) port"
    );
}
