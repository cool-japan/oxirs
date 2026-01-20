# OxiRS Comprehensive Benchmark Suite

**Version**: 0.1.0
**Date**: 2026-01-06
**Purpose**: Performance validation for production readiness

## 🎯 Benchmark Objectives

### Primary Goals
1. **Validate Performance Claims**: Verify 10-50x improvement over baseline
2. **Identify Bottlenecks**: Find performance optimization opportunities
3. **Regression Prevention**: Establish baseline for CI/CD monitoring
4. **Capacity Planning**: Determine production resource requirements
5. **Scalability Validation**: Test horizontal and vertical scaling

### Success Criteria
- **Query Performance**: p95 < 100ms for simple queries, < 1s for complex queries
- **Throughput**: 1000+ qps for SELECT queries
- **Memory Efficiency**: < 10GB for 10M triples
- **Concurrency**: Linear scalability up to 16 cores
- **Large Datasets**: Support 100M+ triples with < 2x performance degradation

---

## 📊 Benchmark Categories

### 1. SPARQL Query Performance

#### 1.1 Simple SELECT Queries
**Purpose**: Test basic triple pattern matching performance

**Scenarios**:
```sparql
# Benchmark: simple_select_single_pattern
SELECT ?s ?o WHERE {
  ?s <http://xmlns.com/foaf/0.1/name> ?o
}

# Benchmark: simple_select_two_patterns
SELECT ?s ?name ?email WHERE {
  ?s <http://xmlns.com/foaf/0.1/name> ?name .
  ?s <http://xmlns.com/foaf/0.1/mbox> ?email
}

# Benchmark: simple_select_with_filter
SELECT ?s ?name WHERE {
  ?s <http://xmlns.com/foaf/0.1/name> ?name .
  FILTER(STRLEN(?name) > 5)
}
```

**Dataset Sizes**: 1K, 10K, 100K, 1M, 10M, 100M triples
**Expected Performance**:
- 1K-100K: < 10ms p95
- 1M: < 50ms p95
- 10M: < 100ms p95
- 100M: < 500ms p95

#### 1.2 Complex JOIN Queries
**Purpose**: Test multi-way join performance

**Scenarios**:
```sparql
# Benchmark: two_way_join
SELECT ?person ?name ?email WHERE {
  ?person a <http://xmlns.com/foaf/0.1/Person> .
  ?person <http://xmlns.com/foaf/0.1/name> ?name .
  ?person <http://xmlns.com/foaf/0.1/mbox> ?email
}

# Benchmark: four_way_join
SELECT ?person ?name ?email ?age ?phone WHERE {
  ?person a <http://xmlns.com/foaf/0.1/Person> .
  ?person <http://xmlns.com/foaf/0.1/name> ?name .
  ?person <http://xmlns.com/foaf/0.1/mbox> ?email .
  ?person <http://xmlns.com/foaf/0.1/age> ?age .
  ?person <http://xmlns.com/foaf/0.1/phone> ?phone
}

# Benchmark: complex_join_with_optional
SELECT ?person ?name ?email ?website WHERE {
  ?person <http://xmlns.com/foaf/0.1/name> ?name .
  ?person <http://xmlns.com/foaf/0.1/mbox> ?email .
  OPTIONAL { ?person <http://xmlns.com/foaf/0.1/homepage> ?website }
}
```

**Expected Performance**:
- 2-way join: < 100ms p95 (10M triples)
- 4-way join: < 500ms p95 (10M triples)
- With OPTIONAL: < 200ms p95 (10M triples)

#### 1.3 Aggregation Queries
**Purpose**: Test GROUP BY, COUNT, AVG performance

**Scenarios**:
```sparql
# Benchmark: simple_count
SELECT (COUNT(*) AS ?count) WHERE {
  ?s ?p ?o
}

# Benchmark: count_distinct
SELECT (COUNT(DISTINCT ?p) AS ?count) WHERE {
  ?s ?p ?o
}

# Benchmark: group_by_predicate
SELECT ?p (COUNT(?s) AS ?count) WHERE {
  ?s ?p ?o
} GROUP BY ?p

# Benchmark: complex_aggregation
SELECT ?type (COUNT(?s) AS ?count) (AVG(?age) AS ?avgAge) WHERE {
  ?s a ?type .
  ?s <http://xmlns.com/foaf/0.1/age> ?age
} GROUP BY ?type
HAVING (COUNT(?s) > 100)
```

**Expected Performance**:
- Simple COUNT: < 500ms p95 (10M triples)
- COUNT DISTINCT: < 1s p95 (10M triples)
- GROUP BY: < 2s p95 (10M triples)
- Complex aggregation: < 3s p95 (10M triples)

#### 1.4 Graph Pattern Queries
**Purpose**: Test UNION, OPTIONAL, FILTER performance

**Scenarios**:
```sparql
# Benchmark: union_query
SELECT ?entity WHERE {
  { ?entity a <http://xmlns.com/foaf/0.1/Person> }
  UNION
  { ?entity a <http://schema.org/Person> }
}

# Benchmark: nested_optional
SELECT ?s ?name ?email ?phone WHERE {
  ?s <http://xmlns.com/foaf/0.1/name> ?name .
  OPTIONAL {
    ?s <http://xmlns.com/foaf/0.1/mbox> ?email .
    OPTIONAL { ?s <http://xmlns.com/foaf/0.1/phone> ?phone }
  }
}

# Benchmark: complex_filter
SELECT ?s ?name ?age WHERE {
  ?s <http://xmlns.com/foaf/0.1/name> ?name .
  ?s <http://xmlns.com/foaf/0.1/age> ?age .
  FILTER(?age >= 18 && ?age <= 65 && REGEX(?name, "^[A-Z]"))
}
```

**Expected Performance**:
- UNION: < 200ms p95 (10M triples)
- Nested OPTIONAL: < 300ms p95 (10M triples)
- Complex FILTER: < 400ms p95 (10M triples)

---

### 2. Data Loading Performance

#### 2.1 Bulk Import
**Purpose**: Test large dataset import performance

**Scenarios**:
- Import 1M triples (N-Triples format)
- Import 10M triples (N-Triples format)
- Import 100M triples (N-Triples format)
- Import 1M triples (Turtle format)
- Import 10M triples (Turtle format)

**Expected Performance**:
- N-Triples: 100K+ triples/second
- Turtle: 50K+ triples/second
- Memory usage: < 1GB for 10M triples during import

#### 2.2 Incremental Updates
**Purpose**: Test SPARQL UPDATE performance

**Scenarios**:
```sparql
# Benchmark: single_insert
INSERT DATA {
  <http://example.org/person123> <http://xmlns.com/foaf/0.1/name> "Alice" .
}

# Benchmark: batch_insert_100
INSERT DATA {
  # 100 triples
  ...
}

# Benchmark: batch_insert_1000
INSERT DATA {
  # 1000 triples
  ...
}

# Benchmark: delete_where
DELETE WHERE {
  ?s <http://xmlns.com/foaf/0.1/age> ?age .
  FILTER(?age < 18)
}
```

**Expected Performance**:
- Single INSERT: < 1ms p95
- Batch INSERT (100): < 10ms p95
- Batch INSERT (1000): < 100ms p95
- DELETE WHERE: < 500ms p95 (10M triples)

---

### 3. Concurrency & Scalability

#### 3.1 Concurrent Queries
**Purpose**: Test multi-threaded query execution

**Scenarios**:
- 1 thread: baseline
- 2 threads: expect ~1.8x throughput
- 4 threads: expect ~3.5x throughput
- 8 threads: expect ~6.5x throughput
- 16 threads: expect ~11x throughput

**Workload**: Mix of SELECT, ASK, CONSTRUCT queries

**Expected Performance**:
- Linear scalability up to CPU core count
- No deadlocks or race conditions
- Consistent p95 latency (< 2x degradation at max concurrency)

#### 3.2 Mixed Workload
**Purpose**: Test read/write concurrency

**Scenarios**:
- 80% reads, 20% writes
- 50% reads, 50% writes
- 20% reads, 80% writes

**Expected Performance**:
- No significant read latency increase with < 50% writes
- Write throughput: 10K+ writes/second
- Read throughput: 100K+ reads/second (simple queries)

---

### 4. Memory & Resource Usage

#### 4.1 Memory Efficiency
**Purpose**: Test memory usage under various loads

**Scenarios**:
- Dataset size: 1M, 10M, 100M triples
- Query complexity: Simple, medium, complex
- Concurrent queries: 1, 10, 100

**Expected Performance**:
- 1M triples: < 500MB RSS
- 10M triples: < 5GB RSS
- 100M triples: < 50GB RSS
- No memory leaks (constant RSS under sustained load)

#### 4.2 Cache Performance
**Purpose**: Test query result caching effectiveness

**Scenarios**:
- Cold cache (first query)
- Warm cache (repeated query)
- Cache eviction (LRU)

**Expected Performance**:
- Warm cache: < 1ms p95
- Cache hit rate: > 80% for repeated queries
- Cache eviction: < 10ms p95

---

### 5. Large Dataset Performance

#### 5.1 100M Triple Dataset
**Purpose**: Validate production-scale performance

**Dataset**: DBpedia subset (100M triples)

**Queries**:
1. Simple SELECT: < 1s p95
2. 2-way JOIN: < 3s p95
3. Aggregation: < 10s p95
4. Complex query: < 30s p95

**Resource Usage**:
- Memory: < 100GB
- Disk: < 200GB
- CPU: < 50% utilization at 100 qps

#### 5.2 Scalability Test
**Purpose**: Test performance degradation with dataset growth

**Datasets**: 1M, 10M, 50M, 100M, 500M triples

**Queries**: Standard benchmark suite

**Expected Performance**:
- Query latency: O(log n) or better
- Import throughput: > 50K triples/second at all sizes
- Memory usage: O(n) with efficient packing

---

## 🔧 Benchmark Implementation

### Directory Structure
```
benchmarks/
├── BENCHMARK_SUITE.md (this file)
├── scenarios/
│   ├── sparql_queries.yaml
│   ├── data_loading.yaml
│   ├── concurrency.yaml
│   └── large_datasets.yaml
├── datasets/
│   ├── generate_1m.sh
│   ├── generate_10m.sh
│   ├── generate_100m.sh
│   └── dbpedia_100m.sh
├── results/
│   └── .gitkeep
└── scripts/
    ├── run_all_benchmarks.sh
    ├── run_sparql_benchmarks.sh
    ├── run_concurrency_benchmarks.sh
    ├── analyze_results.py
    └── compare_versions.py
```

### Benchmark Execution

#### Quick Benchmark (5 minutes)
```bash
./scripts/run_quick_benchmark.sh
```
- 1K-100K triple datasets
- Basic queries only
- Single-threaded

#### Standard Benchmark (30 minutes)
```bash
./scripts/run_standard_benchmark.sh
```
- 1K-10M triple datasets
- All query types
- Up to 8 threads

#### Full Benchmark (4 hours)
```bash
./scripts/run_full_benchmark.sh
```
- 1K-100M triple datasets
- All scenarios
- Full concurrency testing

#### CI/CD Benchmark (10 minutes)
```bash
./scripts/run_ci_benchmark.sh
```
- 1K-1M triple datasets
- Representative queries
- Regression detection

---

## 📈 Performance Targets

### Baseline (Apache Jena 4.x)

| Metric | Jena 4.x | OxiRS Target | Target Improvement |
|--------|----------|--------------|-------------------|
| Simple SELECT (1M) | 50ms p95 | < 10ms p95 | **5x faster** |
| 2-way JOIN (1M) | 200ms p95 | < 40ms p95 | **5x faster** |
| COUNT (10M) | 5s p95 | < 500ms p95 | **10x faster** |
| Import (1M) | 30s | < 10s | **3x faster** |
| Memory (10M) | 10GB | < 5GB | **2x more efficient** |
| Concurrency (8 cores) | 4x throughput | 6x throughput | **50% better scaling** |

### Stretch Goals
- Simple SELECT: < 5ms p95 (1M triples) = **10x faster**
- Complex queries: < 100ms p95 (10M triples) = **50x faster**
- Import: 200K+ triples/second = **6x faster**
- Memory: < 3GB for 10M triples = **3x more efficient**

---

## 🎯 Benchmark Execution Plan

### Phase 1: Baseline Establishment (Week 3)
- [ ] Set up benchmark infrastructure
- [ ] Generate test datasets (1K-100M triples)
- [ ] Run baseline benchmarks on Jena 4.x
- [ ] Document baseline performance

### Phase 2: OxiRS Benchmarking (Week 3)
- [ ] Run OxiRS benchmarks (all scenarios)
- [ ] Collect performance metrics
- [ ] Identify bottlenecks
- [ ] Document initial results

### Phase 3: Optimization (Week 4)
- [ ] Profile slow queries
- [ ] Implement optimizations
- [ ] Re-run benchmarks
- [ ] Validate improvements

### Phase 4: Validation (Week 4)
- [ ] Compare OxiRS vs Jena performance
- [ ] Validate improvement claims
- [ ] Document final results
- [ ] Publish benchmark report

---

## 📊 Results Analysis

### Metrics to Collect
1. **Latency**: p50, p95, p99, max
2. **Throughput**: queries/second, triples/second
3. **Resource Usage**: CPU%, memory (RSS/virtual), disk I/O
4. **Scalability**: speedup vs thread count
5. **Stability**: std deviation, outliers

### Analysis Tools
- **Visualization**: Grafana dashboards
- **Statistical Analysis**: Python (pandas, matplotlib)
- **Comparison**: Side-by-side Jena vs OxiRS charts
- **Regression Detection**: Automated alerts for >10% degradation

---

## 🚀 Continuous Benchmarking

### CI/CD Integration
```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run CI benchmarks
        run: ./scripts/run_ci_benchmark.sh
      - name: Compare with baseline
        run: ./scripts/compare_versions.py
      - name: Detect regressions
        run: |
          if [ $REGRESSION_DETECTED -eq 1 ]; then
            echo "::error::Performance regression detected"
            exit 1
          fi
```

### Nightly Full Benchmarks
- Run full benchmark suite every night
- Track performance trends over time
- Alert on significant regressions (>10%)
- Publish results to monitoring dashboard

---

## 📝 Benchmark Report Template

### Executive Summary
- **Date**: YYYY-MM-DD
- **Version**: 0.1.0
- **Status**: PASS/FAIL
- **Overall Improvement**: Xx faster than Jena

### Detailed Results
- **SPARQL Queries**: Performance breakdown by query type
- **Data Loading**: Import throughput and latency
- **Concurrency**: Scalability metrics
- **Memory**: Resource usage analysis
- **Large Datasets**: 100M triple performance

### Comparison with Baseline
- Side-by-side charts
- Improvement percentages
- Target achievement status

### Bottlenecks Identified
- Performance issues found
- Optimization opportunities
- Action items

### Recommendations
- Production deployment guidelines
- Resource requirements
- Optimization suggestions

---

*Benchmark Suite Documentation - January 6, 2026*
*Production-ready benchmark suite for v0.1.0*
