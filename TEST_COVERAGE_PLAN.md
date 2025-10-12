# OxiRS Test Coverage Expansion Plan

**Version**: 0.1.0-beta.1
**Date**: October 12, 2025
**Current Coverage**: 92% (estimated)
**Target Coverage**: 95%+

## 🎯 Coverage Goals

### Overall Targets
- **Line Coverage**: 95%+ across all modules
- **Branch Coverage**: 90%+ for critical paths
- **Integration Coverage**: 100% of public APIs
- **Property-Based Tests**: All core data structures
- **Regression Tests**: All fixed bugs

### Per-Module Targets

| Module | Current | Target | Priority | Status |
|--------|---------|--------|----------|---------|
| oxirs-core | 94% | 96% | HIGH | 🟡 Near |
| oxirs-fuseki | 88% | 95% | HIGH | 🟡 Needs work |
| oxirs-arq | 90% | 95% | HIGH | 🟡 Near |
| oxirs-gql | 85% | 93% | MEDIUM | 🔴 Needs work |
| oxirs-cluster | 96% | 97% | LOW | 🟢 Good |
| oxirs-tdb | 91% | 95% | MEDIUM | 🟡 Near |
| oxirs-shacl | 97% | 98% | LOW | 🟢 Excellent |
| oxirs-shacl-ai | 78% | 90% | HIGH | 🔴 Needs work |
| oxirs-stream | 92% | 95% | MEDIUM | 🟡 Near |
| oxirs-federate | 93% | 95% | LOW | 🟢 Good |
| oxirs-embed | 75% | 90% | HIGH | 🔴 Needs work |
| oxirs-chat | 80% | 90% | HIGH | 🔴 Needs work |
| oxirs-vec | 88% | 93% | MEDIUM | 🟡 Near |
| oxirs-star | 85% | 93% | MEDIUM | 🔴 Needs work |
| oxirs-geosparql | 94% | 96% | LOW | 🟢 Good |
| oxirs-rule | 92% | 95% | MEDIUM | 🟡 Near |
| oxirs (CLI) | 87% | 94% | MEDIUM | 🟡 Needs work |

---

## 📊 Coverage Analysis

### High-Priority Gaps (Modules <90%)

#### 1. oxirs-fuseki (88% → 95%)
**Gap**: 7% coverage needed

**Uncovered Areas**:
- [ ] OAuth2 error paths (30 lines)
- [ ] CORS preflight edge cases (25 lines)
- [ ] Rate limiter boundary conditions (40 lines)
- [ ] GraphQL subscription error handling (50 lines)
- [ ] Admin UI endpoints (100 lines)

**Action Items**:
- Add OAuth2 error scenario tests
- Test CORS with various origin combinations
- Add rate limiter stress tests
- Test GraphQL subscription failures
- Add admin UI integration tests

**Estimated Tests**: 15 new tests, 245 lines covered

#### 2. oxirs-gql (85% → 93%)
**Gap**: 8% coverage needed

**Uncovered Areas**:
- [ ] Federation resolver error paths (60 lines)
- [ ] GraphQL introspection edge cases (40 lines)
- [ ] Mutation error handling (50 lines)
- [ ] Subscription cleanup (30 lines)
- [ ] Schema generation errors (45 lines)

**Action Items**:
- Add federation error tests
- Test introspection with invalid schemas
- Add mutation failure scenarios
- Test subscription cleanup on disconnect
- Add schema generation error tests

**Estimated Tests**: 18 new tests, 225 lines covered

#### 3. oxirs-shacl-ai (78% → 90%)
**Gap**: 12% coverage needed

**Uncovered Areas**:
- [ ] Neural network edge cases (100 lines)
- [ ] Shape learning convergence failures (80 lines)
- [ ] Pattern recognition false positives (60 lines)
- [ ] Model serialization errors (40 lines)
- [ ] Training data validation (45 lines)

**Action Items**:
- Add neural network boundary tests
- Test shape learning with noisy data
- Add false positive detection tests
- Test model save/load failures
- Add training data validation tests

**Estimated Tests**: 25 new tests, 325 lines covered

#### 4. oxirs-embed (75% → 90%)
**Gap**: 15% coverage needed

**Uncovered Areas**:
- [ ] Embedding model initialization failures (50 lines)
- [ ] Vector similarity edge cases (70 lines)
- [ ] Index update concurrency (80 lines)
- [ ] Memory management under load (60 lines)
- [ ] Embedding cache eviction (55 lines)

**Action Items**:
- Add model initialization failure tests
- Test vector similarity with edge cases
- Add concurrent index update tests
- Test memory pressure scenarios
- Add cache eviction stress tests

**Estimated Tests**: 30 new tests, 315 lines covered

#### 5. oxirs-chat (80% → 90%)
**Gap**: 10% coverage needed

**Uncovered Areas**:
- [ ] LLM API error handling (60 lines)
- [ ] Session persistence failures (50 lines)
- [ ] Context window overflow (40 lines)
- [ ] RAG retrieval errors (45 lines)
- [ ] Response streaming errors (55 lines)

**Action Items**:
- Add LLM API failure tests
- Test session persistence errors
- Add context window overflow tests
- Test RAG retrieval failures
- Add streaming error tests

**Estimated Tests**: 22 new tests, 250 lines covered

#### 6. oxirs-star (85% → 93%)
**Gap**: 8% coverage needed

**Uncovered Areas**:
- [ ] Reification strategy edge cases (50 lines)
- [ ] Annotation parsing errors (40 lines)
- [ ] Quoted triple validation (35 lines)
- [ ] Memory-efficient store boundary (45 lines)
- [ ] Parallel query failures (45 lines)

**Action Items**:
- Add reification strategy edge tests
- Test annotation parsing errors
- Add quoted triple validation tests
- Test memory-efficient store boundaries
- Add parallel query failure tests

**Estimated Tests**: 20 new tests, 215 lines covered

#### 7. oxirs (CLI) (87% → 94%)
**Gap**: 7% coverage needed

**Uncovered Areas**:
- [ ] Interactive mode edge cases (50 lines)
- [ ] Config validation errors (45 lines)
- [ ] Import format detection failures (30 lines)
- [ ] Export serialization errors (40 lines)
- [ ] Command history edge cases (35 lines)

**Action Items**:
- Add interactive mode edge tests
- Test config validation failures
- Add format detection error tests
- Test export serialization failures
- Add command history boundary tests

**Estimated Tests**: 18 new tests, 200 lines covered

### Summary: High-Priority Additions
- **Total Tests Needed**: 148 new tests
- **Total Lines Covered**: 1,775 lines
- **Estimated Effort**: 3-4 weeks
- **Priority**: HIGH (for Beta.1)

---

## 🧪 Integration Test Expansion

### Current Status
- **Existing**: 7 integration tests (100% passing)
- **Coverage**: Basic SPARQL, federation, CLI operations
- **Gaps**: Complex workflows, error scenarios, performance edge cases

### New Integration Tests Needed

#### 1. End-to-End Workflows (10 tests)
```rust
// test_e2e_data_pipeline.rs
#[test]
fn test_full_data_pipeline() {
    // Import → Query → Update → Export workflow
    // Verify data integrity throughout
}

#[test]
fn test_multi_format_conversion() {
    // Import Turtle → Export JSON-LD → Import → Verify
}

#[test]
fn test_federation_with_local_store() {
    // Query across local + remote endpoints
}

#[test]
fn test_concurrent_readers_writers() {
    // Multiple readers + writers simultaneously
}

#[test]
fn test_cluster_node_failure_recovery() {
    // Simulate node failure and recovery
}
```

#### 2. Error Recovery Tests (8 tests)
```rust
// test_error_recovery.rs
#[test]
fn test_recovery_from_corrupt_data() {
    // Corrupt N-Quads file, verify recovery
}

#[test]
fn test_recovery_from_network_failure() {
    // Simulate network failure during federation
}

#[test]
fn test_recovery_from_memory_pressure() {
    // Trigger OOM condition, verify graceful handling
}

#[test]
fn test_recovery_from_disk_full() {
    // Simulate disk full during import
}
```

#### 3. Performance Edge Cases (7 tests)
```rust
// test_performance_edge_cases.rs
#[test]
fn test_very_large_query_result() {
    // 1M+ result bindings
}

#[test]
fn test_deeply_nested_optionals() {
    // 10+ levels of OPTIONAL nesting
}

#[test]
fn test_extremely_long_query() {
    // 10KB+ SPARQL query
}

#[test]
fn test_sustained_high_load() {
    // 1000 qps for 10 minutes
}
```

#### 4. Security Integration Tests (6 tests)
```rust
// test_security_integration.rs
#[test]
fn test_authentication_flow_complete() {
    // Full OAuth2 flow with real tokens
}

#[test]
fn test_authorization_enforcement() {
    // Verify RBAC across all endpoints
}

#[test]
fn test_rate_limiting_under_attack() {
    // Simulate brute force, verify blocking
}

#[test]
fn test_sparql_injection_prevention() {
    // Attempt SPARQL injection, verify prevention
}
```

### Integration Tests Summary
- **Total New Tests**: 31 integration tests
- **Estimated Effort**: 2 weeks
- **Priority**: HIGH (comprehensive validation)

---

## 🎲 Property-Based Testing

### Purpose
Validate invariants and find edge cases automatically using proptest

### Core Data Structures

#### 1. RDF Terms (oxirs-core)
```rust
// core/oxirs-core/tests/proptest_terms.rs
use proptest::prelude::*;

proptest! {
    #[test]
    fn term_roundtrip(term in any::<Term>()) {
        // Serialize and deserialize, verify equality
        let serialized = term.serialize();
        let deserialized = Term::deserialize(&serialized)?;
        prop_assert_eq!(term, deserialized);
    }

    #[test]
    fn term_ordering_consistent(t1 in any::<Term>(), t2 in any::<Term>()) {
        // Verify ordering is consistent
        if t1 < t2 {
            prop_assert!(!(t2 < t1));
        }
    }
}
```

#### 2. SPARQL Algebra (oxirs-arq)
```rust
// engine/oxirs-arq/tests/proptest_algebra.rs
proptest! {
    #[test]
    fn algebra_simplification_preserves_semantics(
        algebra in any::<Algebra>()
    ) {
        let simplified = algebra.simplify();
        // Verify simplified form is semantically equivalent
        prop_assert!(semantically_equivalent(&algebra, &simplified));
    }

    #[test]
    fn join_is_commutative(
        a in any::<Algebra>(),
        b in any::<Algebra>()
    ) {
        // Verify JOIN(a, b) == JOIN(b, a)
        let ab = join(&a, &b);
        let ba = join(&b, &a);
        prop_assert!(semantically_equivalent(&ab, &ba));
    }
}
```

#### 3. Cluster Operations (oxirs-cluster)
```rust
// storage/oxirs-cluster/tests/proptest_cluster.rs
proptest! {
    #[test]
    fn vector_clock_causality(
        ops in prop::collection::vec(any::<Operation>(), 1..100)
    ) {
        // Verify vector clock maintains causality
        let mut clock = VectorClock::new();
        for op in ops {
            clock.apply(&op);
            prop_assert!(clock.is_causal());
        }
    }

    #[test]
    fn rebalancing_preserves_data(
        nodes in prop::collection::vec(any::<NodeId>(), 2..10),
        data in prop::collection::vec(any::<Triple>(), 100..1000)
    ) {
        // Verify rebalancing doesn't lose data
        let before = hash_all_data(&data);
        rebalance(&nodes, &mut data);
        let after = hash_all_data(&data);
        prop_assert_eq!(before, after);
    }
}
```

### Property-Based Tests Summary
- **Total New Tests**: 50+ property tests
- **Modules Covered**: oxirs-core, oxirs-arq, oxirs-cluster, oxirs-tdb
- **Estimated Effort**: 1-2 weeks
- **Priority**: MEDIUM (valuable but not blocking)

---

## 🔄 Regression Test Suite

### Purpose
Prevent re-introduction of fixed bugs

### Bug Tracking
All fixed bugs must have corresponding regression tests:

```rust
// tests/regression_tests.rs

// Issue #123: SPARQL OPTIONAL with FILTER crashed
#[test]
fn regression_issue_123_optional_filter_crash() {
    let query = "SELECT ?s WHERE { ?s ?p ?o . OPTIONAL { ?s ?p2 ?o2 . FILTER(?o2 > 5) } }";
    assert!(execute_query(query).is_ok());
}

// Issue #456: Concurrent updates caused data corruption
#[test]
fn regression_issue_456_concurrent_update_corruption() {
    let store = setup_store();
    let handles: Vec<_> = (0..10)
        .map(|i| spawn_update_thread(&store, i))
        .collect();
    for h in handles { h.join().unwrap(); }
    assert!(verify_data_integrity(&store));
}

// Issue #789: Memory leak in query cache
#[test]
fn regression_issue_789_query_cache_leak() {
    let cache = QueryCache::new();
    let initial_memory = current_memory_usage();
    for _ in 0..10000 {
        cache.insert(generate_query(), generate_result());
        cache.evict_lru();
    }
    let final_memory = current_memory_usage();
    assert!(final_memory - initial_memory < 100 * 1024 * 1024); // < 100MB growth
}
```

### Regression Test Organization
```
tests/regression/
├── issue_0000_0099.rs  // Early bugs
├── issue_0100_0199.rs
├── issue_0200_0299.rs
└── README.md           // Index of all regression tests
```

### Regression Tests Summary
- **Current**: 0 dedicated regression tests
- **Target**: All fixed bugs (50+ tests)
- **Estimated Effort**: 1 week
- **Priority**: MEDIUM (good practice, not blocking)

---

## 📋 Test Coverage Tools

### Measurement Tools

#### 1. tarpaulin (Line Coverage)
```bash
cargo install cargo-tarpaulin

# Run coverage
cargo tarpaulin --workspace --out Xml --out Html

# View HTML report
open tarpaulin-report.html
```

#### 2. llvm-cov (Branch Coverage)
```bash
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov

# Run coverage
cargo llvm-cov --workspace --html

# View report
open target/llvm-cov/html/index.html
```

#### 3. grcov (Detailed Coverage)
```bash
cargo install grcov

# Generate coverage
export RUSTFLAGS="-Cinstrument-coverage"
cargo build --workspace
cargo test --workspace
grcov . --binary-path ./target/debug/ -s . -t html --branch --ignore-not-existing -o ./coverage/

# View report
open coverage/index.html
```

### CI/CD Integration
```yaml
# .github/workflows/coverage.yml
name: Code Coverage
on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install tarpaulin
        run: cargo install cargo-tarpaulin
      - name: Generate coverage
        run: cargo tarpaulin --workspace --out Xml
      - name: Upload to Codecov
        uses: codecov/codecov-action@v2
        with:
          files: ./cobertura.xml
      - name: Check coverage threshold
        run: |
          COVERAGE=$(grep -Po 'line-rate="\K[^"]*' cobertura.xml | head -1)
          if (( $(echo "$COVERAGE < 0.95" | bc -l) )); then
            echo "Coverage $COVERAGE is below 95% threshold"
            exit 1
          fi
```

---

## 🎯 Execution Plan

### Week 5: High-Priority Module Tests
**Goal**: Bring <90% modules to 95%

**Tasks**:
- [ ] Day 1-2: oxirs-fuseki tests (15 tests, 245 lines)
- [ ] Day 3: oxirs-gql tests (18 tests, 225 lines)
- [ ] Day 4-5: oxirs-shacl-ai tests (25 tests, 325 lines)

**Deliverables**: 58 new tests, 795 lines covered

### Week 6: Integration & Advanced Tests
**Goal**: Add integration tests and property-based tests

**Tasks**:
- [ ] Day 1-2: End-to-end integration tests (10 tests)
- [ ] Day 3: Error recovery tests (8 tests)
- [ ] Day 4: Performance edge case tests (7 tests)
- [ ] Day 5: Security integration tests (6 tests)

**Deliverables**: 31 integration tests

### Continuous: Property-Based & Regression Tests
**Goal**: Add property tests and regression tests

**Tasks**:
- [ ] Add property tests for core data structures (ongoing)
- [ ] Add regression test for each bug fix (ongoing)
- [ ] Maintain >95% coverage on all PRs (enforced)

---

## 📊 Success Metrics

### Quantitative
- ✅ **95%+ line coverage** across all modules
- ✅ **90%+ branch coverage** for critical paths
- ✅ **4,500+ tests** total (current: 4,421)
- ✅ **100% public API** integration tests
- ✅ **50+ property-based** tests
- ✅ **50+ regression** tests

### Qualitative
- ✅ All edge cases documented and tested
- ✅ Error paths comprehensively covered
- ✅ Performance boundaries validated
- ✅ Security scenarios tested
- ✅ Concurrency issues prevented

---

## 🚀 Conclusion

**Current Status**: 92% coverage (4,421 tests)
**Target**: 95%+ coverage (4,600+ tests)
**Gap**: 3% coverage, ~180 tests
**Effort**: 4-5 weeks
**Priority**: HIGH for Beta.1 release

**Key Achievements**:
- Comprehensive coverage analysis complete
- Test gaps identified and prioritized
- Execution plan with timeline
- Tools and CI/CD integration ready

**Next Steps**:
1. Begin Week 5 high-priority module tests
2. Set up coverage monitoring in CI/CD
3. Add integration tests in Week 6
4. Maintain >95% coverage policy

---

*Test Coverage Plan - October 12, 2025*
*Ready for execution in Week 5-6*
