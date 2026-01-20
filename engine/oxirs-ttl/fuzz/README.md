# Fuzzing Infrastructure for oxirs-ttl

This directory contains fuzzing targets for the oxirs-ttl parser and serializer components. Fuzzing helps discover crashes, panics, and unexpected behavior when processing malformed or edge-case inputs.

## Available Fuzz Targets

1. **turtle_parser** - Fuzzes the Turtle parser with arbitrary input
2. **ntriples_parser** - Fuzzes the N-Triples parser
3. **nquads_parser** - Fuzzes the N-Quads parser
4. **trig_parser** - Fuzzes the TriG parser
5. **turtle_serializer** - Fuzzes round-trip parsing and serialization

## Prerequisites

Install cargo-fuzz (if not already installed):

```bash
cargo install cargo-fuzz
```

## Running Fuzz Tests

### Run a specific fuzz target

```bash
# From the oxirs-ttl directory
cargo fuzz run turtle_parser

# Run with timeout (recommended for CI)
cargo fuzz run turtle_parser -- -max_total_time=60

# Run with specific number of iterations
cargo fuzz run turtle_parser -- -runs=10000
```

### Run all fuzz targets

```bash
# Run each target for 60 seconds
./fuzz/run_all_fuzzers.sh
```

### Check for existing crashes

```bash
cargo fuzz cmin turtle_parser
```

### Reproduce a crash

```bash
# After finding a crash artifact
cargo fuzz run turtle_parser fuzz/artifacts/turtle_parser/crash-abc123
```

## Continuous Integration

For CI environments, run each fuzzer with a time limit:

```bash
cargo fuzz run turtle_parser -- -max_total_time=300  # 5 minutes
cargo fuzz run ntriples_parser -- -max_total_time=300
cargo fuzz run nquads_parser -- -max_total_time=300
cargo fuzz run trig_parser -- -max_total_time=300
cargo fuzz run turtle_serializer -- -max_total_time=300
```

## Analyzing Results

Fuzzing results are stored in:
- `fuzz/corpus/` - Interesting test cases discovered
- `fuzz/artifacts/` - Crashes and failing inputs

### Coverage Analysis

```bash
# Generate coverage report
cargo fuzz coverage turtle_parser
```

## Best Practices

1. **Regular Fuzzing**: Run fuzzers regularly (e.g., nightly CI jobs)
2. **Corpus Management**: Keep discovered corpus cases for regression testing
3. **Crash Triage**: Fix crashes in order of severity (crashes > hangs > slow inputs)
4. **Seed Corpus**: Add known valid RDF files to `fuzz/corpus/` for better coverage

## Performance Targets

- No crashes on arbitrary input
- No panics on malformed data
- Graceful error handling for all invalid inputs
- Parse time < 1s for inputs up to 1MB

## Known Issues

### Build Environment Dependencies

The fuzz targets may require specific system dependencies:
- **Python Framework**: Some systems may have Python framework linking issues
- **OpenBLAS**: Required by scirs2-core dependencies

If you encounter linker errors:
1. Ensure Python development headers are installed
2. Check OpenBLAS installation: `brew install openblas` (macOS)
3. Try building without sanitizers: `cargo build --manifest-path fuzz/Cargo.toml --bin turtle_parser`

## Troubleshooting

### Out of Memory

```bash
# Limit memory usage
cargo fuzz run turtle_parser -- -rss_limit_mb=2048
```

### Slow Fuzzing

```bash
# Use multiple jobs for parallel fuzzing
cargo fuzz run turtle_parser -- -jobs=4
```

### Build Issues

```bash
# Clean and rebuild
cargo fuzz clean
cargo fuzz build
```

## Integration with Test Suite

Crashes found during fuzzing should be:
1. Minimized with `cargo fuzz cmin`
2. Converted to regression tests in `tests/`
3. Fixed and verified with the full test suite

## References

- [cargo-fuzz documentation](https://rust-fuzz.github.io/book/)
- [libFuzzer options](https://llvm.org/docs/LibFuzzer.html#options)
- [AFL fuzzing guide](https://github.com/rust-fuzz/afl.rs)
