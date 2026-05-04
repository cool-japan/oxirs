# Performance SLAs for oxirs-core

## Overview

`oxirs-core` ships a regression-detection harness for named performance
Service-Level Objectives (SLOs). The harness is built on two public types:

- [`BenchmarkResult`][br] — produced by `BenchmarkResult::measure(name, n, closure)`.
- [`assert_meets_slo`][ams] — checks a `BenchmarkResult` against an `SloTarget`.

[br]: ../src/perf_sla.rs
[ams]: ../src/perf_sla.rs

## Running SLA tests

SLA gate tests are marked `#[ignore]` so they don't run in CI by default
(timing thresholds are machine-dependent). To run them on a release build:

```
cargo test --release -p oxirs-core -- --ignored
```

To run only the term-equality SLO:

```
cargo test --release -p oxirs-core -- --ignored sla_suite_term_equality
```

## Threshold behaviour by build mode

| Build mode | Timing assertions |
|------------|------------------|
| `debug`    | Skipped (always Ok) |
| `release`  | Enforced |

This is implemented via `#[cfg(debug_assertions)]` in `assert_meets_slo`.
The rationale: debug builds are typically 5–50× slower than release, so
asserting latency thresholds in debug would produce constant false failures.

## Baseline update workflow

When you intentionally improve or change a performance characteristic:

1. Run the SLA tests in release mode and record results:
   ```
   cargo test --release -p oxirs-core -- --ignored 2>&1 | grep sla_
   ```
2. Update `perf_baseline.json` with the new measured values.
3. Open a PR with the updated baseline file alongside the code change.
4. The PR description should note the before/after measurements.

## SLO targets

| Name | p50 target | p99 target | Throughput target | Regression allowance |
|------|-----------|-----------|-------------------|---------------------|
| `sla_term_equality` | 100µs | 1 000µs | — | 10% |
| `sla_ntriples_line_count_1k` | 500 000µs | 2 000 000µs | — | 10% |

Note: the ntriples line-count target is intentionally very relaxed (it measures
1 000 iterations of `.lines().count()` over a 1 k-line string). It is a canary,
not a hard deadline.

## Adding a new SLO

1. Add a `BenchmarkResult::measure(...)` call in `tests/perf_sla.rs` under a
   new `#[test] #[ignore]` function whose name starts with `sla_suite_`.
2. Define an `SloTarget` with the desired thresholds.
3. Call `assert_meets_slo(&result, &target).expect("...")`.
4. Add a matching entry to `benches/sla_suite.rs` with `criterion`.
5. Update `perf_baseline.json` with the initial measured values.
6. Document the new SLO in the table above.

## Notes

- All timing assertions are skipped in debug builds.
- Thresholds include a configurable regression allowance (default 10%).
- `BenchmarkResult::measure` sorts per-call durations to compute p50/p99.
  It is intentionally simple — for production profiling use `criterion`.
