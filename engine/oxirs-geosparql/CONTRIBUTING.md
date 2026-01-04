# Contributing to OxiRS GeoSPARQL

Thank you for your interest in contributing to OxiRS GeoSPARQL! This guide will help you get started with contributing to this high-performance Rust implementation of the OGC GeoSPARQL standard.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Performance Considerations](#performance-considerations)
10. [Community](#community)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

**Expected Behavior:**
- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other contributors

**Unacceptable Behavior:**
- Harassment, trolling, or inflammatory comments
- Personal attacks or insults
- Publishing others' private information
- Any conduct that could reasonably be considered inappropriate

---

## Getting Started

### Prerequisites

- **Rust**: 1.75.0 or later (use `rustup` to install)
- **System Dependencies:**
  - **Linux:** `libproj-dev`, `proj-bin`, `libgeos-dev`
  - **macOS:** `brew install proj geos`
  - **Windows:** Limited PROJ/GEOS support (core features work)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs/engine/oxirs-geosparql

# Install Rust toolchain
rustup install stable
rustup default stable

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libproj-dev proj-bin libgeos-dev

# Install development tools
cargo install nextest cargo-llvm-cov cargo-audit

# Run tests to verify setup
cargo nextest run --all-features
```

---

## Development Setup

### Recommended Tools

- **Editor:** VS Code with rust-analyzer extension
- **Testing:** `cargo nextest` (faster than `cargo test`)
- **Coverage:** `cargo llvm-cov`
- **Linting:** `cargo clippy`
- **Formatting:** `cargo fmt`
- **Benchmarking:** `cargo bench`

### VS Code Configuration

Create `.vscode/settings.json`:

```json
{
  "rust-analyzer.cargo.features": "all",
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.checkOnSave.extraArgs": ["--all-features", "--", "-D", "warnings"],
  "editor.formatOnSave": true,
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  }
}
```

### Building the Project

```bash
# Standard build
cargo build --package oxirs-geosparql

# Build with all features
cargo build --package oxirs-geosparql --all-features

# Release build with optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --package oxirs-geosparql --release --features performance

# Check compilation without building
cargo check --package oxirs-geosparql --all-features
```

---

## How to Contribute

### Finding Issues

Good starting points:
- **Good First Issue:** Issues tagged for newcomers
- **Help Wanted:** Issues needing assistance
- **Documentation:** Improve guides, examples, or API docs
- **Performance:** Optimize hot paths or algorithms
- **Testing:** Add test coverage or new test cases

Browse issues: https://github.com/cool-japan/oxirs/issues

### Reporting Bugs

When reporting bugs, include:

1. **Description:** Clear summary of the issue
2. **Steps to Reproduce:** Minimal code example
3. **Expected Behavior:** What should happen
4. **Actual Behavior:** What actually happens
5. **Environment:**
   - OS and version
   - Rust version (`rustc --version`)
   - oxirs-geosparql version
   - Enabled features

**Example Bug Report:**

```markdown
**Bug:** WKT parsing fails for 3D polygons with holes

**Steps to Reproduce:**
```rust
use oxirs_geosparql::geometry::Geometry;

let wkt = "POLYGON Z((0 0 0, 10 0 0, 10 10 10, 0 10 10, 0 0 0), (2 2 2, 8 2 2, 8 8 8, 2 8 8, 2 2 2))";
let geom = Geometry::from_wkt(wkt); // Fails here
```

**Expected:** Should parse successfully
**Actual:** Returns `ParseError: Invalid WKT`

**Environment:**
- OS: Ubuntu 22.04
- Rust: 1.75.0
- oxirs-geosparql: 0.1.0-rc.2
- Features: `all-features`
```

### Suggesting Features

For feature requests, include:

1. **Use Case:** What problem does this solve?
2. **Proposed Solution:** How should it work?
3. **Alternatives:** Other approaches considered
4. **Impact:** Who benefits from this feature?

Check existing feature requests first to avoid duplicates.

---

## Coding Standards

### Rust Style Guide

We follow the official [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/). Key points:

#### Naming Conventions

```rust
// Types: PascalCase
struct GeometryProcessor { }
enum SpatialIndex { }
trait TopologicalRelation { }

// Functions, variables: snake_case
fn calculate_distance(point_a: &Point, point_b: &Point) -> f64 { }
let spatial_index = SpatialIndex::new();

// Constants: SCREAMING_SNAKE_CASE
const MAX_GEOMETRY_SIZE: usize = 1_000_000;
const DEFAULT_CRS_URI: &str = "http://www.opengis.net/def/crs/EPSG/0/4326";

// Lifetimes: 'lowercase
fn process_geometry<'a>(geom: &'a Geometry) -> &'a Point { }
```

#### Code Organization

```rust
// 1. Imports
use std::collections::HashMap;
use oxirs_core::RdfNode;

// 2. Type definitions
pub struct Geometry {
    // Fields...
}

// 3. Implementations
impl Geometry {
    // Associated functions (constructors)
    pub fn new(geom: GeoGeometry) -> Self { }

    // Methods
    pub fn is_3d(&self) -> bool { }
}

// 4. Trait implementations
impl Display for Geometry {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result { }
}

// 5. Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometry_creation() { }
}
```

### Error Handling

Use `Result<T, GeoSparqlError>` for fallible operations:

```rust
use crate::error::{GeoSparqlError, Result};

pub fn parse_wkt(wkt: &str) -> Result<Geometry> {
    // Validate input
    if wkt.is_empty() {
        return Err(GeoSparqlError::ParseError(
            "WKT string cannot be empty".to_string()
        ));
    }

    // Parse WKT
    let parsed = wkt::Wkt::from_str(wkt)
        .map_err(|e| GeoSparqlError::ParseError(format!("Invalid WKT: {}", e)))?;

    // Convert to geometry
    convert_wkt_to_geometry(&parsed)
}
```

**Don't use `unwrap()` or `expect()` in library code** (tests are OK).

### Documentation

Every public item must have documentation:

```rust
/// Calculate the distance between two geometries.
///
/// This function computes the minimum Euclidean distance between two
/// geometries using the specified coordinate reference system.
///
/// # Arguments
///
/// * `geom1` - The first geometry
/// * `geom2` - The second geometry
///
/// # Returns
///
/// The minimum distance in the units of the CRS, or an error if the
/// geometries have incompatible CRS.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::geometric_operations::distance;
///
/// let p1 = Geometry::from_wkt("POINT(0 0)").unwrap();
/// let p2 = Geometry::from_wkt("POINT(3 4)").unwrap();
///
/// let dist = distance(&p1, &p2).unwrap();
/// assert!((dist - 5.0).abs() < 1e-10);
/// ```
///
/// # Errors
///
/// Returns `GeoSparqlError::CrsIncompatible` if the geometries have
/// different coordinate reference systems.
pub fn distance(geom1: &Geometry, geom2: &Geometry) -> Result<f64> {
    // Implementation...
}
```

### Performance Guidelines

1. **Avoid Allocations:** Use references and slices when possible
2. **Use `&str` over `String`:** For function parameters
3. **Preallocate Vectors:** Use `Vec::with_capacity()` when size is known
4. **Batch Operations:** Provide batch APIs for bulk processing
5. **Zero-Copy:** Use `Cow<str>` or borrowing when applicable

```rust
// Good: Avoid allocation
pub fn process_geometries(geometries: &[Geometry]) -> Result<Vec<f64>> {
    let mut results = Vec::with_capacity(geometries.len()); // Preallocate

    for geom in geometries {
        results.push(calculate_area(geom)?);
    }

    Ok(results)
}

// Bad: Unnecessary cloning
pub fn process_geometries(geometries: Vec<Geometry>) -> Result<Vec<f64>> {
    let mut results = Vec::new();

    for geom in geometries {
        let cloned = geom.clone(); // Unnecessary
        results.push(calculate_area(&cloned)?);
    }

    Ok(results)
}
```

### SciRS2 Integration Policy

**ALWAYS use SciRS2 for scientific computing** (see CLAUDE.md):

```rust
// ✅ CORRECT: Use scirs2-core
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::random::Random;
use scirs2_core::simd::SimdArray;

// ❌ WRONG: Direct dependencies
use ndarray::{Array2};  // Don't use directly
use rand::Rng;          // Don't use directly
```

---

## Testing Guidelines

### Test Organization

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Unit tests: Test individual functions
    #[test]
    fn test_point_creation() {
        let point = Geometry::from_wkt("POINT(10 20)").unwrap();
        assert!(!point.is_3d());
    }

    // Integration tests: Test multiple components
    #[test]
    fn test_distance_calculation_workflow() {
        let p1 = Geometry::from_wkt("POINT(0 0)").unwrap();
        let p2 = Geometry::from_wkt("POINT(3 4)").unwrap();

        let dist = distance(&p1, &p2).unwrap();
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }

    // Edge case tests: Test boundary conditions
    #[test]
    fn test_distance_same_point() {
        let p = Geometry::from_wkt("POINT(5 5)").unwrap();
        let dist = distance(&p, &p).unwrap();
        assert_relative_eq!(dist, 0.0, epsilon = 1e-10);
    }

    // Error case tests: Test error handling
    #[test]
    fn test_invalid_wkt_returns_error() {
        let result = Geometry::from_wkt("INVALID WKT");
        assert!(result.is_err());
    }
}
```

### Test Coverage Requirements

- **All public APIs** must have tests
- **Error paths** must be tested
- **Edge cases** must be covered
- **Target:** >90% code coverage

### Running Tests

```bash
# Run all tests
cargo nextest run --package oxirs-geosparql --all-features

# Run specific test
cargo nextest run --package oxirs-geosparql distance_calculation

# Run with coverage
cargo llvm-cov --package oxirs-geosparql --all-features --html
# Open: target/llvm-cov/html/index.html

# Run property-based tests
cargo test --package oxirs-geosparql property_tests

# Run stress tests
cargo nextest run --package oxirs-geosparql stress_tests --test-threads 1

# Run benchmarks
cargo bench --package oxirs-geosparql
```

### Benchmarking

Add benchmarks for performance-critical code:

```rust
// benches/my_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::geometric_operations::distance;

fn benchmark_distance(c: &mut Criterion) {
    let p1 = Geometry::from_wkt("POINT(0 0)").unwrap();
    let p2 = Geometry::from_wkt("POINT(3 4)").unwrap();

    c.bench_function("distance_calculation", |b| {
        b.iter(|| {
            distance(black_box(&p1), black_box(&p2))
        })
    });
}

criterion_group!(benches, benchmark_distance);
criterion_main!(benches);
```

---

## Documentation

### API Documentation

- **Every public item** must have `///` documentation
- Include **examples** in doc comments
- Document **errors** that can occur
- Add **links** to related items

### User Guides

When adding new features, update:
- `README.md` - High-level overview
- `docs/COOKBOOK.md` - Add recipe examples
- `docs/PERFORMANCE_TUNING.md` - Add optimization tips (if applicable)
- Examples in `examples/` directory

### Building Documentation

```bash
# Build and open documentation
cargo doc --package oxirs-geosparql --all-features --open

# Check for broken links
cargo doc --package oxirs-geosparql --all-features
```

---

## Pull Request Process

### Before Submitting

**Checklist:**

- [ ] Code compiles without warnings (`cargo build --all-features`)
- [ ] All tests pass (`cargo nextest run --all-features`)
- [ ] Clippy is happy (`cargo clippy --all-features -- -D warnings`)
- [ ] Code is formatted (`cargo fmt --check`)
- [ ] Documentation is updated
- [ ] Examples added (if new feature)
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] Benchmarks added (if performance-critical)

### PR Template

```markdown
## Description

Brief description of changes.

## Motivation

Why is this change needed? What problem does it solve?

## Changes

- Added feature X
- Fixed bug Y
- Improved performance of Z

## Testing

How was this tested?

- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing
- [ ] Benchmarks

## Checklist

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Clippy passes
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

## Related Issues

Fixes #123
```

### CI Checks

All PRs must pass:
- ✅ Format check (`cargo fmt --check`)
- ✅ Clippy (`cargo clippy -- -D warnings`)
- ✅ Tests (Linux, macOS, Windows)
- ✅ Property tests
- ✅ Stress tests
- ✅ Documentation build
- ✅ Security audit
- ✅ MSRV check (Rust 1.75.0)

### Review Process

1. **Automated CI** runs on all PRs
2. **Maintainer review** - expect feedback within 1-2 days
3. **Address feedback** - update PR based on comments
4. **Final approval** - maintainer approves and merges

### Merging

- **Squash commits** for feature PRs (keep history clean)
- **Merge commits** for multi-commit PRs (preserve history)
- **Delete branch** after merge

---

## Performance Considerations

### Profiling

Use profiling tools to find bottlenecks:

```bash
# CPU profiling with flamegraph
cargo install flamegraph
cargo flamegraph --bench my_benchmark

# Memory profiling with valgrind
cargo install cargo-valgrind
cargo valgrind test --package oxirs-geosparql test_name
```

### Benchmarking Best Practices

1. **Use `black_box()`** to prevent compiler optimizations
2. **Warm up** before measuring
3. **Run multiple iterations** for statistical significance
4. **Compare with baseline** using `cargo bench`

### Performance Regression Detection

CI automatically detects performance regressions:
- Benchmarks run on every PR
- Alert if performance degrades >50%
- Compare with baseline on main branch

---

## Community

### Getting Help

- **GitHub Discussions:** https://github.com/cool-japan/oxirs/discussions
- **Issues:** https://github.com/cool-japan/oxirs/issues
- **Documentation:** https://docs.rs/oxirs-geosparql

### Maintainers

- Primary maintainers listed in `Cargo.toml`
- Contact via GitHub issues or discussions

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md`
- Release notes
- GitHub contributors page

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (Apache 2.0 or MIT).

---

## Additional Resources

- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [The Rust Programming Language Book](https://doc.rust-lang.org/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [GeoSPARQL 1.1 Specification](http://www.opengis.net/doc/IS/geosparql/1.1)
- [OxiRS Documentation](../README.md)

---

## Questions?

Don't hesitate to ask questions! We're here to help:
- Open an issue: https://github.com/cool-japan/oxirs/issues/new
- Start a discussion: https://github.com/cool-japan/oxirs/discussions/new

**Thank you for contributing to OxiRS GeoSPARQL!** 🎉
