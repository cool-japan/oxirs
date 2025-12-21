# Contributing to OxiRS

Thank you for your interest in contributing to OxiRS! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [RFC Process](#rfc-process)
- [Release Process](#release-process)

## Code of Conduct

OxiRS adheres to the Rust Code of Conduct. By participating in this project, you agree to abide by its terms.

## Getting Started

### Prerequisites

- **Rust**: 1.70+ (MSRV - Minimum Supported Rust Version)
- **Development Tools**:
  - `cargo-nextest` for testing
  - `cargo-clippy` for linting
  - `rustfmt` for code formatting
  - `cargo-deny` for dependency checks (optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs

# Run setup script
./scripts/setup-dev.sh

# Build the project
cargo build --workspace

# Run tests
cargo nextest run --no-fail-fast
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Make Changes

- Follow the coding standards below
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Your Changes

We use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

```
feat: add SPARQL 1.2 aggregation support
fix: resolve memory leak in triple store
docs: update README with new examples
test: add integration tests for federation
refactor: simplify query optimizer logic
perf: optimize triple pattern matching
```

Sign off your commits using the `-s` flag (DCO 1.1):

```bash
git commit -s -m "feat: add new feature"
```

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Coding Standards

### Rust Style

- **Formatting**: Use `rustfmt` with default settings
  ```bash
  cargo fmt --all
  ```

- **Linting**: Pass all `clippy` checks
  ```bash
  cargo clippy --workspace --all-targets -- -D warnings
  ```

- **Naming Conventions**:
  - Variables/functions: `snake_case`
  - Types/traits: `PascalCase`
  - Constants: `SCREAMING_SNAKE_CASE`
  - Modules: `snake_case`

### Code Organization

- **File Size**: Keep files under 2000 lines. Use [SplitRS](https://github.com/cool-japan/splitrs) for refactoring large files
- **Module Structure**: Follow the existing module hierarchy
- **Imports**: Group and order imports logically
  ```rust
  // Standard library
  use std::collections::HashMap;

  // External crates
  use anyhow::Result;
  use serde::{Deserialize, Serialize};

  // Internal crates
  use oxirs_core::Triple;

  // Local modules
  use crate::error::FusekiError;
  ```

### SciRS2 Integration

OxiRS uses the SciRS2 scientific computing library instead of direct `rand` or `ndarray` usage:

```rust
// ❌ WRONG
use ndarray::Array2;
use rand::thread_rng;

// ✅ CORRECT
use scirs2_core::ndarray_ext::Array2;
use scirs2_core::random::rng;
```

See `CLAUDE.md` for full SciRS2 integration guidelines.

### Error Handling

- Use `anyhow::Result` for application code
- Use `thiserror` for library error types
- Provide context with `.context()` or `.with_context()`
- Document error conditions in function docs

```rust
use anyhow::{Context, Result};

fn load_config(path: &Path) -> Result<Config> {
    std::fs::read_to_string(path)
        .context("Failed to read config file")?
        .parse()
        .context("Failed to parse config")
}
```

## Testing Guidelines

### Test Coverage

- Aim for 95%+ test coverage
- Write unit tests for all public APIs
- Add integration tests for cross-module functionality
- Include property-based tests using `proptest` where appropriate

### Running Tests

```bash
# Run all tests
cargo nextest run --no-fail-fast

# Run tests for a specific crate
cargo nextest run -p oxirs-core

# Run with all features
cargo nextest run --all-features

# Run specific test
cargo nextest run test_name
```

### Test Organization

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Arrange
        let input = create_test_input();

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected);
    }

    #[test]
    fn test_error_case() {
        let result = function_that_should_fail();
        assert!(result.is_err());
    }
}
```

### Benchmarks

Add benchmarks for performance-critical code using `criterion`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_query(c: &mut Criterion) {
    c.bench_function("sparql_select", |b| {
        b.iter(|| execute_query(black_box(&query)))
    });
}

criterion_group!(benches, benchmark_query);
criterion_main!(benches);
```

## Documentation

### Code Documentation

- Add doc comments to all public items
- Include examples in doc comments
- Document panics, errors, and safety requirements

```rust
/// Executes a SPARQL query against the RDF store.
///
/// # Arguments
///
/// * `query` - The SPARQL query string to execute
/// * `dataset` - The dataset to query against
///
/// # Returns
///
/// Returns the query results or an error if the query is invalid.
///
/// # Examples
///
/// ```
/// use oxirs_core::Store;
///
/// let store = Store::new();
/// let results = store.query("SELECT * WHERE { ?s ?p ?o }")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The query syntax is invalid
/// - The dataset is not accessible
/// - An I/O error occurs
pub fn execute_query(query: &str, dataset: &Dataset) -> Result<QueryResults> {
    // Implementation
}
```

### Documentation Files

- Update README.md for user-facing changes
- Add entries to CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/)
- Create RFCs for major features (see RFC Process below)

## Pull Request Process

### Before Submitting

1. ✅ All tests pass (`cargo nextest run --no-fail-fast`)
2. ✅ Code is formatted (`cargo fmt --all --check`)
3. ✅ No clippy warnings (`cargo clippy --workspace --all-targets -- -D warnings`)
4. ✅ Documentation is updated
5. ✅ CHANGELOG.md is updated (if applicable)

### PR Description

Include in your PR description:

- **Summary**: What does this PR do?
- **Motivation**: Why is this change needed?
- **Testing**: How was this tested?
- **Breaking Changes**: List any breaking changes
- **Related Issues**: Link to related issues

### Review Process

- At least one maintainer approval required
- All CI checks must pass
- Address review comments promptly
- Keep PRs focused and reasonably sized

### Merging

- PRs are squash-merged to maintain clean history
- Commit message follows Conventional Commits
- Delete branch after merge

## RFC Process

For significant changes, submit an RFC (Request for Comments):

### When to Use RFCs

- New major features
- Breaking API changes
- Architectural changes
- New module additions

### RFC Process

1. Create a file in `rfcs/` directory:
   ```
   rfcs/YYYY-MM-DD-feature-name.md
   ```

2. RFC Template:
   ```markdown
   # RFC: Feature Name

   **Author**: Your Name
   **Date**: YYYY-MM-DD
   **Status**: Draft | Under Review | Accepted | Rejected

   ## Summary
   One-paragraph explanation of the feature.

   ## Motivation
   Why are we doing this? What use cases does it support?

   ## Design
   Detailed technical design.

   ## Alternatives
   What other designs were considered?

   ## Unresolved Questions
   What questions remain?
   ```

3. Submit PR with RFC
4. 14-day comment window
5. Lazy consensus (accepted unless objections)
6. Implementation follows after acceptance

## Release Process

OxiRS follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Version Bumps

Versions are managed at the workspace level in the root `Cargo.toml`:

```toml
[workspace.package]
version = "0.1.0-beta.2"
```

### Release Checklist

1. Update CHANGELOG.md
2. Update version in Cargo.toml
3. Run full test suite
4. Update documentation
5. Create git tag
6. Publish to crates.io (maintainers only)
7. Create GitHub release

## Development Tips

### Using SplitRS for Refactoring

For files exceeding 2000 lines:

```bash
splitrs --input src/large_file.rs \
        --output src/large_file/ \
        --split-impl-blocks \
        --max-impl-lines 200
```

### Useful Commands

```bash
# Check for outdated dependencies
cargo outdated

# Security audit
cargo audit

# Generate documentation
cargo doc --workspace --all-features --no-deps --open

# Check code coverage (requires cargo-tarpaulin)
cargo tarpaulin --workspace --all-features

# Profile tests
cargo nextest run --profile ci
```

### IDE Setup

Recommended extensions for VS Code:

- rust-analyzer
- Even Better TOML
- Error Lens
- GitLens

## Getting Help

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Chat**: Join our community (link TBD)

## License

By contributing to OxiRS, you agree that your contributions will be licensed under both MIT and Apache-2.0 licenses (dual license).

## Recognition

Contributors are recognized in:

- CHANGELOG.md (for significant contributions)
- GitHub contributors page
- Release notes

Thank you for contributing to OxiRS! 🚀
