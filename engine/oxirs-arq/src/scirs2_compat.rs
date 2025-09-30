//! Temporary compatibility shim for missing scirs2-core APIs
//!
//! This module provides temporary implementations of metrics, profiling, and ML pipeline APIs
//! that are not yet available in scirs2-core beta.3. Once scirs2-core beta.4 is released with
//! these APIs, this module should be removed and all usage should migrate to scirs2-core.
//!
//! MIGRATION PLAN: When scirs2-core beta.4 is available:
//! 1. Remove this module
//! 2. Change imports from `crate::scirs2_compat::*` to `scirs2_core::metrics::*`, etc.
//! 3. Remove fallback dependencies from Cargo.toml (prometheus, once_cell, sysinfo)

use once_cell::sync::Lazy;
use prometheus::{
    Histogram as PrometheusHistogram, HistogramOpts, IntCounter as PrometheusIntCounter,
    IntGauge as PrometheusIntGauge, Opts, Registry,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Global registry for all metrics
static REGISTRY: Lazy<Registry> = Lazy::new(Registry::new);
static COUNTERS: Lazy<Mutex<HashMap<String, PrometheusIntCounter>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static GAUGES: Lazy<Mutex<HashMap<String, PrometheusIntGauge>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static HISTOGRAMS: Lazy<Mutex<HashMap<String, PrometheusHistogram>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Counter metric (compatible with scirs2_core::metrics::Counter)
#[derive(Clone, Debug)]
pub struct Counter {
    name: String,
    inner: PrometheusIntCounter,
}

impl Counter {
    pub fn new(name: &str) -> Self {
        let mut counters = COUNTERS.lock().unwrap();
        let inner = counters.entry(name.to_string()).or_insert_with(|| {
            let counter = PrometheusIntCounter::with_opts(Opts::new(name, name)).unwrap();
            REGISTRY.register(Box::new(counter.clone())).ok();
            counter
        });
        Self {
            name: name.to_string(),
            inner: inner.clone(),
        }
    }

    pub fn increment(&self) {
        self.inner.inc();
    }

    pub fn increment_by(&self, n: u64) {
        self.inner.inc_by(n);
    }

    pub fn value(&self) -> u64 {
        self.inner.get()
    }
}

/// Gauge metric (compatible with scirs2_core::metrics::Gauge)
#[derive(Clone, Debug)]
pub struct Gauge {
    name: String,
    inner: PrometheusIntGauge,
}

impl Gauge {
    pub fn new(name: &str) -> Self {
        let mut gauges = GAUGES.lock().unwrap();
        let inner = gauges.entry(name.to_string()).or_insert_with(|| {
            let gauge = PrometheusIntGauge::with_opts(Opts::new(name, name)).unwrap();
            REGISTRY.register(Box::new(gauge.clone())).ok();
            gauge
        });
        Self {
            name: name.to_string(),
            inner: inner.clone(),
        }
    }

    pub fn set(&self, value: i64) {
        self.inner.set(value);
    }

    pub fn increment(&self) {
        self.inner.inc();
    }

    pub fn decrement(&self) {
        self.inner.dec();
    }

    pub fn value(&self) -> i64 {
        self.inner.get()
    }
}

/// Histogram metric (compatible with scirs2_core::metrics::Histogram)
#[derive(Clone, Debug)]
pub struct Histogram {
    name: String,
    inner: PrometheusHistogram,
}

impl Histogram {
    pub fn new(name: &str) -> Self {
        let mut histograms = HISTOGRAMS.lock().unwrap();
        let inner = histograms.entry(name.to_string()).or_insert_with(|| {
            let histogram = PrometheusHistogram::with_opts(HistogramOpts::new(name, name)).unwrap();
            REGISTRY.register(Box::new(histogram.clone())).ok();
            histogram
        });
        Self {
            name: name.to_string(),
            inner: inner.clone(),
        }
    }

    pub fn observe(&self, value: f64) {
        self.inner.observe(value);
    }

    pub fn record(&self, value: f64) {
        self.observe(value);
    }
}

/// Timer metric for measuring durations (compatible with scirs2_core::metrics::Timer)
#[derive(Clone, Debug)]
pub struct Timer {
    histogram: Histogram,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        let histogram_name = format!("{}_duration_seconds", name);
        Self {
            histogram: Histogram::new(&histogram_name),
        }
    }

    pub fn record(&self, duration: Duration) {
        self.histogram.observe(duration.as_secs_f64());
    }

    pub fn time<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        self.record(start.elapsed());
        result
    }
}

/// Profiler for performance profiling (compatible with scirs2_core::profiling::Profiler)
pub struct Profiler {
    timings: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
    active_spans: Arc<Mutex<HashMap<String, Instant>>>,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            timings: Arc::new(Mutex::new(HashMap::new())),
            active_spans: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn start(&self, name: &str) {
        let mut spans = self.active_spans.lock().unwrap();
        spans.insert(name.to_string(), Instant::now());
    }

    pub fn stop(&self, name: &str) {
        let mut spans = self.active_spans.lock().unwrap();
        if let Some(start_time) = spans.remove(name) {
            let duration = start_time.elapsed();
            let mut timings = self.timings.lock().unwrap();
            timings
                .entry(name.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }

    pub fn get_timing(&self, name: &str) -> Option<Duration> {
        let timings = self.timings.lock().unwrap();
        timings.get(name).and_then(|v| v.last().copied())
    }

    pub fn get_average_timing(&self, name: &str) -> Option<Duration> {
        let timings = self.timings.lock().unwrap();
        timings.get(name).map(|v| {
            let total: Duration = v.iter().sum();
            total / v.len() as u32
        })
    }

    pub fn clear(&self) {
        self.timings.lock().unwrap().clear();
        self.active_spans.lock().unwrap().clear();
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

// Placeholder types for ML pipeline (not fully implemented yet)
// These are minimal stubs to allow compilation

/// ML Pipeline placeholder (scirs2_core::ml_pipeline::MLPipeline)
pub struct MLPipeline {
    name: String,
}

impl MLPipeline {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    pub fn train(&mut self, _data: &[f64]) -> anyhow::Result<()> {
        // Placeholder: No actual training implementation
        Ok(())
    }

    pub fn predict(&self, _input: &[f64]) -> anyhow::Result<Vec<f64>> {
        // Placeholder: No actual prediction implementation
        Ok(vec![])
    }
}

/// Model Predictor placeholder (scirs2_core::ml_pipeline::ModelPredictor)
pub trait ModelPredictor {
    fn predict(&self, input: &[f64]) -> anyhow::Result<Vec<f64>>;
}

impl ModelPredictor for MLPipeline {
    fn predict(&self, input: &[f64]) -> anyhow::Result<Vec<f64>> {
        self.predict(input)
    }
}

/// SimdArray placeholder (scirs2_core::simd::SimdArray)
///
/// This is a simplified version without actual SIMD optimizations.
/// In scirs2-core beta.4, this should use the native SimdArray implementation.
#[derive(Clone, Debug)]
pub struct SimdArray<T> {
    data: Vec<T>,
}

impl<T: Clone + Default> SimdArray<T> {
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![T::default(); len],
        }
    }
}

impl SimdArray<bool> {
    pub fn ones(len: usize) -> Self {
        Self {
            data: vec![true; len],
        }
    }
}

impl SimdArray<u64> {
    pub fn ones(len: usize) -> Self {
        Self {
            data: vec![1; len],
        }
    }
}

impl SimdArray<f64> {
    pub fn ones(len: usize) -> Self {
        Self {
            data: vec![1.0; len],
        }
    }
}

impl<T: Clone> SimdArray<T> {
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl<T> std::ops::Index<usize> for SimdArray<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> std::ops::IndexMut<usize> for SimdArray<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}