use oxirs::{run, BenchmarkAction, Cli, Commands, ConfigAction};
use tempfile::tempdir;

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
