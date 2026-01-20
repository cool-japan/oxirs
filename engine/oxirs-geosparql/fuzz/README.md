# Fuzzing Tests for oxirs-geosparql

This directory contains fuzzing tests for the oxirs-geosparql parser functions using [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz).

## Overview

Fuzzing is an automated testing technique that feeds random or malformed data to functions to discover bugs, panics, and edge cases. The fuzzing targets in this directory test the robustness of various parsers against malformed input.

## Available Fuzz Targets

- **fuzz_wkt_parser** - Tests WKT (Well-Known Text) parsing
- **fuzz_geojson_parser** - Tests GeoJSON parsing
- **fuzz_ewkb_parser** - Tests EWKB (Extended Well-Known Binary) and EWKT parsing
- **fuzz_gml_parser** - Tests GML (Geography Markup Language) parsing
- **fuzz_flatgeobuf_parser** - Tests FlatGeobuf binary format parsing
- **fuzz_zero_copy_wkt** - Tests zero-copy WKT parser performance and correctness

## Prerequisites

Fuzzing requires **nightly Rust** and **cargo-fuzz**:

```bash
# Install nightly Rust
rustup install nightly

# Install cargo-fuzz
cargo install cargo-fuzz
```

## Running Fuzzing Tests

### Basic Usage

```bash
# Switch to nightly (required for fuzzing)
cd engine/oxirs-geosparql

# Run a specific fuzz target (runs indefinitely until stopped with Ctrl+C)
cargo +nightly fuzz run fuzz_wkt_parser

# Run with a time limit (e.g., 60 seconds)
cargo +nightly fuzz run fuzz_wkt_parser -- -max_total_time=60

# Run with a maximum number of runs
cargo +nightly fuzz run fuzz_wkt_parser -- -runs=1000000
```

### Running All Targets

```bash
# Run each target for 5 minutes
for target in $(cargo +nightly fuzz list); do
    echo "Fuzzing $target for 5 minutes..."
    cargo +nightly fuzz run $target -- -max_total_time=300
done
```

### Advanced Options

```bash
# Run with custom number of jobs (parallel workers)
cargo +nightly fuzz run fuzz_wkt_parser -- -jobs=4

# Run with a dictionary file to guide fuzzing
cargo +nightly fuzz run fuzz_wkt_parser -- -dict=wkt_dictionary.txt

# Run with AddressSanitizer for memory error detection (default)
cargo +nightly fuzz run fuzz_wkt_parser

# Generate a coverage report
cargo +nightly fuzz coverage fuzz_wkt_parser
```

## Corpus Management

Fuzzing generates a corpus of test inputs that trigger unique code paths:

```bash
# View the corpus directory
ls fuzz/corpus/fuzz_wkt_parser/

# Add custom seed inputs to guide fuzzing
echo "POINT(1 2)" > fuzz/corpus/fuzz_wkt_parser/valid_point.txt
echo "LINESTRING(0 0, 1 1)" > fuzz/corpus/fuzz_wkt_parser/valid_linestring.txt

# Minimize the corpus (remove redundant inputs)
cargo +nightly fuzz cmin fuzz_wkt_parser

# Merge multiple corpora
cargo +nightly fuzz cmin fuzz_wkt_parser fuzz/corpus/fuzz_wkt_parser fuzz/corpus/fuzz_zero_copy_wkt
```

## Analyzing Crashes

If a fuzz target discovers a crash:

```bash
# Crashes are saved to fuzz/artifacts/<target_name>/
ls fuzz/artifacts/fuzz_wkt_parser/

# Reproduce a specific crash
cargo +nightly fuzz run fuzz_wkt_parser fuzz/artifacts/fuzz_wkt_parser/crash-abc123

# Debug with more verbosity
RUST_BACKTRACE=1 cargo +nightly fuzz run fuzz_wkt_parser fuzz/artifacts/fuzz_wkt_parser/crash-abc123
```

## Continuous Fuzzing

For long-term fuzzing campaigns (e.g., overnight or on CI):

```bash
# Run indefinitely in the background
nohup cargo +nightly fuzz run fuzz_wkt_parser > fuzz.log 2>&1 &

# Monitor progress
tail -f fuzz.log

# Stop fuzzing
pkill -f "cargo.*fuzz.*fuzz_wkt_parser"
```

## Integration with CI/CD

Example GitHub Actions workflow snippet:

```yaml
- name: Run fuzzing tests
  run: |
    rustup install nightly
    cargo install cargo-fuzz
    cd engine/oxirs-geosparql
    cargo +nightly fuzz run fuzz_wkt_parser -- -max_total_time=300
```

## Expected Behavior

All fuzz targets should:
- **Never panic** - All errors should be handled gracefully via `Result` types
- **Handle invalid UTF-8** - Binary parsers should not assume valid UTF-8
- **Handle truncated input** - Parsers should validate input length
- **Handle malformed structures** - Return appropriate errors for invalid formats

If a fuzz target panics or crashes, it indicates a bug that should be fixed.

## Performance Considerations

- Fuzzing is CPU-intensive and will use 100% of allocated cores
- Expected execution rate: 1000-10000 executions/second depending on target complexity
- Corpus grows over time - consider periodic minimization
- Use `--release` builds for faster fuzzing (cargo-fuzz does this by default)

## Adding New Fuzz Targets

To add a new fuzz target:

1. Create a new file in `fuzz/fuzz_targets/fuzz_new_parser.rs`:

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = your_parser_function(data);
});
```

2. Add the target to `fuzz/Cargo.toml`:

```toml
[[bin]]
name = "fuzz_new_parser"
path = "fuzz_targets/fuzz_new_parser.rs"
test = false
doc = false
bench = false
```

3. Run the new target:

```bash
cargo +nightly fuzz run fuzz_new_parser
```

## Resources

- [cargo-fuzz book](https://rust-fuzz.github.io/book/cargo-fuzz.html)
- [libFuzzer documentation](https://llvm.org/docs/LibFuzzer.html)
- [Rust Fuzz Book](https://rust-fuzz.github.io/book/)
- [AFL++ (alternative fuzzer)](https://github.com/AFLplusplus/AFLplusplus)

## Troubleshooting

### "error: the option `Z` is only accepted on the nightly compiler"

**Solution**: Use `cargo +nightly fuzz` instead of `cargo fuzz`.

### "error: could not compile `libfuzzer-sys`"

**Solution**: Make sure you're using a recent nightly toolchain:
```bash
rustup update nightly
```

### Fuzzing is too slow

**Solution**:
- Reduce input size limits in the fuzz target
- Use multiple jobs: `-- -jobs=8`
- Use a faster machine or cloud fuzzing service

### Too many false positives

**Solution**:
- Add input validation to reject trivially invalid inputs early
- Use a dictionary file to guide fuzzing toward valid syntax
- Adjust timeout settings: `-- -timeout=10`

## Contributing

When contributing new parsers to oxirs-geosparql, please:
1. Add a corresponding fuzz target
2. Run fuzzing for at least 1 hour before submitting PR
3. Include any discovered corpus inputs in the PR

---

**Note**: Fuzzing requires nightly Rust. Stable Rust builds cannot run cargo-fuzz, but the fuzz targets are maintained for security and robustness testing.
