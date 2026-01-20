//! CI/CD Integration module
//!
//! Provides comprehensive CI/CD integration capabilities including:
//! - Test result reporting (JUnit XML, TAP)
//! - Performance regression detection
//! - Docker integration helpers
//! - CI workflow template generation

use super::CommandResult;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Test result for CI/CD reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub name: String,
    pub status: TestStatus,
    pub duration_ms: f64,
    pub message: Option<String>,
    pub stacktrace: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
    Error,
}

/// Test suite results
#[derive(Debug, Serialize, Deserialize)]
pub struct TestSuite {
    pub name: String,
    pub tests: Vec<TestResult>,
    pub timestamp: String,
    pub hostname: String,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub errors: usize,
    pub duration_ms: f64,
}

impl TestSuite {
    pub fn new(name: String) -> Self {
        TestSuite {
            name,
            tests: Vec::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            hostname: hostname::get()
                .ok()
                .and_then(|h| h.into_string().ok())
                .unwrap_or_else(|| "unknown".to_string()),
            total: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            errors: 0,
            duration_ms: 0.0,
        }
    }

    pub fn add_test(&mut self, test: TestResult) {
        self.total += 1;
        self.duration_ms += test.duration_ms;

        match test.status {
            TestStatus::Passed => self.passed += 1,
            TestStatus::Failed => self.failed += 1,
            TestStatus::Skipped => self.skipped += 1,
            TestStatus::Error => self.errors += 1,
        }

        self.tests.push(test);
    }

    /// Generate JUnit XML format
    pub fn to_junit_xml(&self) -> Result<String, Box<dyn std::error::Error>> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str(&format!(
            "<testsuite name=\"{}\" tests=\"{}\" failures=\"{}\" errors=\"{}\" skipped=\"{}\" time=\"{:.3}\" timestamp=\"{}\" hostname=\"{}\">\n",
            escape_xml(&self.name),
            self.total,
            self.failed,
            self.errors,
            self.skipped,
            self.duration_ms / 1000.0,
            self.timestamp,
            escape_xml(&self.hostname)
        ));

        for test in &self.tests {
            xml.push_str(&format!(
                "  <testcase name=\"{}\" classname=\"{}\" time=\"{:.3}\"",
                escape_xml(&test.name),
                escape_xml(&self.name),
                test.duration_ms / 1000.0
            ));

            match test.status {
                TestStatus::Passed => {
                    xml.push_str("/>\n");
                }
                TestStatus::Failed => {
                    xml.push_str(">\n");
                    xml.push_str(&format!(
                        "    <failure message=\"{}\">{}</failure>\n",
                        escape_xml(&test.message.clone().unwrap_or_default()),
                        escape_xml(&test.stacktrace.clone().unwrap_or_default())
                    ));
                    xml.push_str("  </testcase>\n");
                }
                TestStatus::Skipped => {
                    xml.push_str(">\n");
                    xml.push_str(&format!(
                        "    <skipped message=\"{}\"/>\n",
                        escape_xml(&test.message.clone().unwrap_or_default())
                    ));
                    xml.push_str("  </testcase>\n");
                }
                TestStatus::Error => {
                    xml.push_str(">\n");
                    xml.push_str(&format!(
                        "    <error message=\"{}\">{}</error>\n",
                        escape_xml(&test.message.clone().unwrap_or_default()),
                        escape_xml(&test.stacktrace.clone().unwrap_or_default())
                    ));
                    xml.push_str("  </testcase>\n");
                }
            }
        }

        xml.push_str("</testsuite>\n");
        Ok(xml)
    }

    /// Generate TAP (Test Anything Protocol) format
    pub fn to_tap(&self) -> Result<String, Box<dyn std::error::Error>> {
        let mut tap = String::new();
        tap.push_str("TAP version 13\n");
        tap.push_str(&format!("1..{}\n", self.total));

        for (i, test) in self.tests.iter().enumerate() {
            let test_num = i + 1;

            match test.status {
                TestStatus::Passed => {
                    tap.push_str(&format!("ok {} - {}\n", test_num, test.name));
                }
                TestStatus::Failed => {
                    tap.push_str(&format!("not ok {} - {}\n", test_num, test.name));
                    if let Some(msg) = &test.message {
                        tap.push_str(&format!("  ---\n  message: {}\n  ...\n", msg));
                    }
                }
                TestStatus::Skipped => {
                    tap.push_str(&format!("ok {} - {} # SKIP\n", test_num, test.name));
                }
                TestStatus::Error => {
                    tap.push_str(&format!("not ok {} - {} # ERROR\n", test_num, test.name));
                    if let Some(msg) = &test.message {
                        tap.push_str(&format!("  ---\n  error: {}\n  ...\n", msg));
                    }
                }
            }
        }

        tap.push_str(&format!("# tests {}\n", self.total));
        tap.push_str(&format!("# pass {}\n", self.passed));
        tap.push_str(&format!("# fail {}\n", self.failed));
        tap.push_str(&format!("# skip {}\n", self.skipped));
        tap.push_str(&format!("# duration {:.3}s\n", self.duration_ms / 1000.0));

        Ok(tap)
    }
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Generate test report from benchmark results
pub async fn generate_test_report(
    input: PathBuf,
    output: PathBuf,
    format: String,
) -> CommandResult {
    println!("Generating test report");
    println!("Input: {}", input.display());
    println!("Output: {}", output.display());
    println!("Format: {}\n", format);

    // Load benchmark results
    let content = fs::read_to_string(&input)?;
    let benchmark_results: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON: {}", e))?;

    // Convert to test suite
    let mut suite = TestSuite::new("OxiRS Benchmark Suite".to_string());

    if let Some(queries) = benchmark_results
        .get("query_results")
        .and_then(|v| v.as_array())
    {
        for query in queries {
            let name = query
                .get("query_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            let avg_time = query
                .get("avg_time")
                .and_then(|v| v.get("secs"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
                * 1000.0;

            let success_rate = query
                .get("success_rate")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);

            let status = if success_rate >= 0.95 {
                TestStatus::Passed
            } else if success_rate > 0.0 {
                TestStatus::Failed
            } else {
                TestStatus::Error
            };

            let message = if status != TestStatus::Passed {
                Some(format!("Success rate: {:.1}%", success_rate * 100.0))
            } else {
                None
            };

            suite.add_test(TestResult {
                name: name.to_string(),
                status,
                duration_ms: avg_time,
                message,
                stacktrace: None,
            });
        }
    }

    // Generate report in requested format
    let report = match format.as_str() {
        "junit" | "xml" => suite.to_junit_xml()?,
        "tap" => suite.to_tap()?,
        "json" => serde_json::to_string_pretty(&suite)
            .map_err(|e| format!("Failed to serialize JSON: {}", e))?,
        _ => {
            return Err(format!(
                "Unsupported format '{}'. Supported: junit, tap, json",
                format
            )
            .into())
        }
    };

    // Write report
    fs::write(&output, report)?;

    println!("✓ Test report generated successfully");
    println!("  Format: {}", format);
    println!("  Total tests: {}", suite.total);
    println!("  Passed: {}", suite.passed);
    println!("  Failed: {}", suite.failed);
    println!("  Errors: {}", suite.errors);
    println!("  Skipped: {}", suite.skipped);
    println!("  Output: {}", output.display());

    if suite.failed > 0 || suite.errors > 0 {
        return Err("Test failures detected".into());
    }

    Ok(())
}

/// Generate Docker integration files
pub async fn generate_docker_files(output_dir: PathBuf) -> CommandResult {
    println!("Generating Docker integration files");
    println!("Output directory: {}\n", output_dir.display());

    // Create output directory
    fs::create_dir_all(&output_dir)?;

    // Generate Dockerfile
    let dockerfile = r#"# OxiRS Docker Image
FROM rust:1.75-slim as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy project files
COPY . .

# Build release binary
RUN cargo build --release -p oxirs

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /app/target/release/oxirs /usr/local/bin/oxirs

# Create data directory
RUN mkdir -p /data
WORKDIR /data

# Expose SPARQL endpoint port
EXPOSE 3030

# Set entrypoint
ENTRYPOINT ["oxirs"]
CMD ["--help"]
"#;

    fs::write(output_dir.join("Dockerfile"), dockerfile)?;
    println!("✓ Generated Dockerfile");

    // Generate docker-compose.yml
    let docker_compose = r#"version: '3.8'

services:
  oxirs:
    build: .
    ports:
      - "3030:3030"
    volumes:
      - ./data:/data
      - ./config:/config
    environment:
      - OXIRS_LOG_LEVEL=info
    command: serve /config/oxirs.toml

  # Optional: Add Prometheus for monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  # Optional: Add Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
"#;

    fs::write(output_dir.join("docker-compose.yml"), docker_compose)?;
    println!("✓ Generated docker-compose.yml");

    // Generate .dockerignore
    let dockerignore = r#"target/
.git/
.gitignore
*.md
.env
*.log
"#;

    fs::write(output_dir.join(".dockerignore"), dockerignore)?;
    println!("✓ Generated .dockerignore");

    // Generate Makefile for Docker operations
    let makefile = r#"# OxiRS Docker Makefile

.PHONY: build run stop clean test

# Build Docker image
build:
	docker build -t oxirs:latest .

# Run OxiRS server
run:
	docker-compose up -d

# Stop services
stop:
	docker-compose down

# Clean containers and images
clean:
	docker-compose down -v
	docker rmi oxirs:latest

# Run tests in container
test:
	docker build -t oxirs:test --target builder .
	docker run --rm oxirs:test cargo test

# Build and run
up: build run

# View logs
logs:
	docker-compose logs -f oxirs
"#;

    fs::write(output_dir.join("Makefile"), makefile)?;
    println!("✓ Generated Makefile");

    println!("\nDocker files generated successfully!");
    println!("Usage:");
    println!("  make build  - Build Docker image");
    println!("  make run    - Start services with docker-compose");
    println!("  make test   - Run tests in container");
    println!("  make logs   - View container logs");

    Ok(())
}

/// Generate GitHub Actions workflow
pub async fn generate_github_workflow(output_file: PathBuf) -> CommandResult {
    println!("Generating GitHub Actions workflow");
    println!("Output: {}\n", output_file.display());

    let workflow = r#"name: OxiRS CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  test:
    name: Test Suite
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta]

    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: dtolnay/rust-toolchain@master
      with:
        toolchain: ${{ matrix.rust }}

    - name: Cache cargo registry
      uses: actions/cache@v4
      with:
        path: ~/.cargo/registry
        key: ${{ runner.os }}-cargo-registry-${{ hashFiles('**/Cargo.lock') }}

    - name: Cache cargo index
      uses: actions/cache@v4
      with:
        path: ~/.cargo/git
        key: ${{ runner.os }}-cargo-index-${{ hashFiles('**/Cargo.lock') }}

    - name: Cache target directory
      uses: actions/cache@v4
      with:
        path: target
        key: ${{ runner.os }}-target-${{ hashFiles('**/Cargo.lock') }}

    - name: Run tests
      run: cargo test --workspace --all-features

    - name: Run CLI tests
      run: cargo test -p oxirs --all-features

  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable

    - name: Run benchmarks
      run: |
        cargo build --release -p oxirs
        ./target/release/oxirs benchmark run testdata --suite sp2bench --iterations 10 -o baseline.json

    - name: Upload benchmark results
      uses: actions/upload-artifact@v4
      with:
        name: benchmark-results
        path: baseline.json

    - name: Compare with baseline
      if: github.event_name == 'pull_request'
      run: |
        # Download baseline from main branch
        # Compare and fail if regression > 10%
        ./target/release/oxirs benchmark compare main-baseline.json baseline.json --threshold 10.0

  lint:
    name: Rustfmt and Clippy
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
      with:
        components: rustfmt, clippy

    - name: Check formatting
      run: cargo fmt --all -- --check

    - name: Run clippy
      run: cargo clippy --workspace --all-targets --all-features -- -D warnings

  coverage:
    name: Code Coverage
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable

    - name: Install tarpaulin
      run: cargo install cargo-tarpaulin

    - name: Generate coverage
      run: cargo tarpaulin --workspace --all-features --out Xml

    - name: Upload to codecov
      uses: codecov/codecov-action@v4
      with:
        files: ./cobertura.xml

  docker:
    name: Docker Build
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Build Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: false
        tags: oxirs:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max

    - name: Test Docker image
      run: |
        docker run --rm oxirs:latest --version

  release:
    name: Release
    runs-on: ubuntu-latest
    needs: [test, lint, benchmark]
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')

    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable

    - name: Build release binary
      run: cargo build --release -p oxirs

    - name: Create release
      uses: softprops/action-gh-release@v1
      with:
        files: |
          target/release/oxirs
        generate_release_notes: true
"#;

    // Create .github/workflows directory if needed
    if let Some(parent) = output_file.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&output_file, workflow)?;

    println!("✓ GitHub Actions workflow generated successfully");
    println!("  File: {}", output_file.display());
    println!("\nWorkflow includes:");
    println!("  • Cross-platform testing (Linux, macOS, Windows)");
    println!("  • Performance benchmarking with regression detection");
    println!("  • Code linting (rustfmt + clippy)");
    println!("  • Code coverage reporting");
    println!("  • Docker build validation");
    println!("  • Automated releases on tags");

    Ok(())
}

/// Generate GitLab CI configuration
pub async fn generate_gitlab_ci(output_file: PathBuf) -> CommandResult {
    println!("Generating GitLab CI configuration");
    println!("Output: {}\n", output_file.display());

    let gitlab_ci = r#"# OxiRS GitLab CI/CD Configuration

image: rust:latest

variables:
  CARGO_HOME: $CI_PROJECT_DIR/.cargo
  RUST_BACKTRACE: "1"

cache:
  key: "${CI_COMMIT_REF_SLUG}"
  paths:
    - .cargo/
    - target/

stages:
  - test
  - benchmark
  - lint
  - build
  - deploy

# Test job
test:
  stage: test
  script:
    - cargo test --workspace --all-features --verbose
  artifacts:
    when: always
    reports:
      junit: test-results.xml
  after_script:
    - cargo test --workspace --all-features -- --format=json | cargo2junit > test-results.xml

# Benchmark job
benchmark:
  stage: benchmark
  script:
    - cargo build --release -p oxirs
    - ./target/release/oxirs benchmark run testdata --suite sp2bench --iterations 10 -o benchmark.json
    - |
      if [ -f baseline.json ]; then
        ./target/release/oxirs benchmark compare baseline.json benchmark.json --threshold 10.0
      fi
  artifacts:
    paths:
      - benchmark.json
    expire_in: 30 days

# Lint job
lint:
  stage: lint
  script:
    - rustup component add rustfmt clippy
    - cargo fmt --all -- --check
    - cargo clippy --workspace --all-targets --all-features -- -D warnings

# Build Docker image
docker-build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest
  only:
    - main
    - tags

# Deploy to production
deploy:
  stage: deploy
  script:
    - echo "Deploy to production server"
    # Add your deployment commands here
  only:
    - tags
  when: manual
"#;

    fs::write(&output_file, gitlab_ci)?;

    println!("✓ GitLab CI configuration generated successfully");
    println!("  File: {}", output_file.display());
    println!("\nConfiguration includes:");
    println!("  • Automated testing with JUnit reports");
    println!("  • Performance benchmarking");
    println!("  • Code linting and formatting checks");
    println!("  • Docker image building");
    println!("  • Manual deployment to production");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_suite_creation() {
        let mut suite = TestSuite::new("Test Suite".to_string());

        suite.add_test(TestResult {
            name: "test1".to_string(),
            status: TestStatus::Passed,
            duration_ms: 100.0,
            message: None,
            stacktrace: None,
        });

        suite.add_test(TestResult {
            name: "test2".to_string(),
            status: TestStatus::Failed,
            duration_ms: 50.0,
            message: Some("Failed".to_string()),
            stacktrace: None,
        });

        assert_eq!(suite.total, 2);
        assert_eq!(suite.passed, 1);
        assert_eq!(suite.failed, 1);
        assert_eq!(suite.duration_ms, 150.0);
    }

    #[test]
    fn test_junit_xml_generation() {
        let mut suite = TestSuite::new("JUnit Test".to_string());

        suite.add_test(TestResult {
            name: "test_pass".to_string(),
            status: TestStatus::Passed,
            duration_ms: 100.0,
            message: None,
            stacktrace: None,
        });

        let xml = suite.to_junit_xml().unwrap();

        assert!(xml.contains("<?xml version"));
        assert!(xml.contains("<testsuite"));
        assert!(xml.contains("test_pass"));
        assert!(xml.contains("tests=\"1\""));
    }

    #[test]
    fn test_tap_generation() {
        let mut suite = TestSuite::new("TAP Test".to_string());

        suite.add_test(TestResult {
            name: "test_tap".to_string(),
            status: TestStatus::Passed,
            duration_ms: 50.0,
            message: None,
            stacktrace: None,
        });

        let tap = suite.to_tap().unwrap();

        assert!(tap.contains("TAP version 13"));
        assert!(tap.contains("1..1"));
        assert!(tap.contains("ok 1 - test_tap"));
    }

    #[test]
    fn test_xml_escaping() {
        let escaped = escape_xml("<test> & \"quoted\" 'value'");
        assert_eq!(
            escaped,
            "&lt;test&gt; &amp; &quot;quoted&quot; &apos;value&apos;"
        );
    }
}
