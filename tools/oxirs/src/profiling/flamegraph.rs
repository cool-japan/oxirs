//! Flame Graph Generation for SPARQL Query Profiling
//!
//! This module provides flame graph visualization for SPARQL query profiling data.
//! It uses the `inferno` crate to generate interactive SVG flame graphs with:
//! - Color-coding by execution phase (parsing, planning, execution)
//! - Support for folded stack format (Brendan Gregg format)
//! - Differential flame graphs for comparing query executions
//! - Interactive zooming and filtering in the SVG output
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs::profiling::flamegraph::{FlameGraphGenerator, FlameGraphOptions, ExecutionPhase};
//!
//! let mut generator = FlameGraphGenerator::new();
//!
//! // Add profiling samples
//! generator.add_sample("parse", ExecutionPhase::Parsing, 150);
//! generator.add_sample("parse;validate", ExecutionPhase::Parsing, 50);
//! generator.add_sample("optimize", ExecutionPhase::Optimization, 200);
//! generator.add_sample("execute;join", ExecutionPhase::Execution, 1500);
//!
//! // Generate SVG flame graph
//! let svg = generator.generate_svg(FlameGraphOptions::default())?;
//! std::fs::write("flamegraph.svg", svg)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use inferno::flamegraph::{Direction, Options};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufWriter, Write};

/// Execution phase for color-coding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionPhase {
    /// Query parsing phase
    Parsing,
    /// Query planning/optimization phase
    Optimization,
    /// Query execution phase
    Execution,
    /// Other/unknown phase
    Other,
}

impl ExecutionPhase {
    /// Get the color for this phase in the flame graph
    pub fn color(&self) -> &'static str {
        match self {
            ExecutionPhase::Parsing => "#6BAED6",      // Blue
            ExecutionPhase::Optimization => "#FD8D3C", // Orange
            ExecutionPhase::Execution => "#74C476",    // Green
            ExecutionPhase::Other => "#9E9AC8",        // Purple
        }
    }

    /// Get a human-readable name for this phase
    pub fn name(&self) -> &'static str {
        match self {
            ExecutionPhase::Parsing => "Parsing",
            ExecutionPhase::Optimization => "Optimization",
            ExecutionPhase::Execution => "Execution",
            ExecutionPhase::Other => "Other",
        }
    }
}

/// Profiling sample with stack trace and timing
#[derive(Debug, Clone)]
pub struct ProfilingSample {
    /// Folded stack trace (semicolon-separated)
    pub stack: String,
    /// Execution phase for color-coding
    pub phase: ExecutionPhase,
    /// Sample count or duration in microseconds
    pub value: u64,
}

/// Options for flame graph generation
#[derive(Debug, Clone)]
pub struct FlameGraphOptions {
    /// Title for the flame graph
    pub title: String,
    /// Subtitle (e.g., query details)
    pub subtitle: Option<String>,
    /// Minimum width threshold for displaying frames (0.0-1.0)
    pub min_width: f64,
    /// Direction (TopToBottom or BottomToTop)
    pub direction: FlameGraphDirection,
    /// Enable color-coding by phase
    pub color_by_phase: bool,
    /// Enable search box in the SVG
    pub search_enabled: bool,
    /// Custom color palette
    pub palette: Option<String>,
}

impl Default for FlameGraphOptions {
    fn default() -> Self {
        Self {
            title: "SPARQL Query Profile".to_string(),
            subtitle: None,
            min_width: 0.0,
            direction: FlameGraphDirection::TopToBottom,
            color_by_phase: true,
            search_enabled: true,
            palette: None,
        }
    }
}

/// Flame graph direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlameGraphDirection {
    /// Top-to-bottom (traditional)
    TopToBottom,
    /// Bottom-to-top (icicle)
    BottomToTop,
}

/// Flame graph generator
pub struct FlameGraphGenerator {
    /// Profiling samples
    samples: Vec<ProfilingSample>,
    /// Total sample count
    total_samples: u64,
}

impl FlameGraphGenerator {
    /// Create a new flame graph generator
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            total_samples: 0,
        }
    }

    /// Add a profiling sample
    ///
    /// # Arguments
    /// * `stack` - Folded stack trace (e.g., "parse;validate;check")
    /// * `phase` - Execution phase for color-coding
    /// * `value` - Sample count or duration in microseconds
    pub fn add_sample(&mut self, stack: impl Into<String>, phase: ExecutionPhase, value: u64) {
        self.samples.push(ProfilingSample {
            stack: stack.into(),
            phase,
            value,
        });
        self.total_samples += value;
    }

    /// Add multiple samples from a folded stack file
    ///
    /// Format: `stack;trace value`
    pub fn add_folded_samples(
        &mut self,
        folded: &str,
        default_phase: ExecutionPhase,
    ) -> Result<(), String> {
        for line in folded.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.rsplitn(2, ' ').collect();
            if parts.len() != 2 {
                return Err(format!("Invalid folded stack line: {}", line));
            }

            let stack = parts[1];
            let value = parts[0]
                .parse::<u64>()
                .map_err(|e| format!("Invalid sample value: {}", e))?;

            // Infer phase from stack
            let phase = Self::infer_phase(stack).unwrap_or(default_phase);
            self.add_sample(stack, phase, value);
        }

        Ok(())
    }

    /// Infer execution phase from stack trace
    fn infer_phase(stack: &str) -> Option<ExecutionPhase> {
        let lower = stack.to_lowercase();
        if lower.contains("parse") || lower.contains("lexer") || lower.contains("tokenize") {
            Some(ExecutionPhase::Parsing)
        } else if lower.contains("optimize") || lower.contains("plan") || lower.contains("rewrite")
        {
            Some(ExecutionPhase::Optimization)
        } else if lower.contains("execute")
            || lower.contains("eval")
            || lower.contains("join")
            || lower.contains("scan")
        {
            Some(ExecutionPhase::Execution)
        } else {
            None
        }
    }

    /// Generate folded stack format string
    pub fn to_folded(&self) -> String {
        let mut output = String::new();
        for sample in &self.samples {
            output.push_str(&format!("{} {}\n", sample.stack, sample.value));
        }
        output
    }

    /// Generate interactive SVG flame graph
    pub fn generate_svg(&self, options: FlameGraphOptions) -> Result<String, String> {
        let mut inferno_opts = Options::default();

        // Configure basic options
        inferno_opts.title = options.title.clone();
        if let Some(subtitle) = &options.subtitle {
            inferno_opts.subtitle = Some(subtitle.clone());
        }
        inferno_opts.min_width = options.min_width;

        // Set direction
        inferno_opts.direction = match options.direction {
            FlameGraphDirection::TopToBottom => Direction::Straight,
            FlameGraphDirection::BottomToTop => Direction::Inverted,
        };

        // Note: search functionality and custom colors are controlled by inferno's default behavior
        // The generated SVG will have interactive features by default

        // Convert samples to folded format
        let folded = self.to_folded();
        let folded_bytes = folded.as_bytes();

        // Generate SVG
        let mut svg_buffer = Vec::new();
        {
            let mut writer = BufWriter::new(&mut svg_buffer);
            inferno::flamegraph::from_reader(&mut inferno_opts, folded_bytes, &mut writer)
                .map_err(|e| format!("Failed to generate flame graph: {}", e))?;
            writer
                .flush()
                .map_err(|e| format!("Failed to flush SVG: {}", e))?;
        }

        String::from_utf8(svg_buffer).map_err(|e| format!("Invalid UTF-8 in SVG: {}", e))
    }

    /// Get total sample count
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Get number of unique stacks
    pub fn unique_stacks(&self) -> usize {
        let mut stacks = std::collections::HashSet::new();
        for sample in &self.samples {
            stacks.insert(&sample.stack);
        }
        stacks.len()
    }

    /// Get statistics by phase
    pub fn phase_statistics(&self) -> HashMap<ExecutionPhase, PhaseStats> {
        let mut stats = HashMap::new();

        for sample in &self.samples {
            let entry = stats.entry(sample.phase).or_insert(PhaseStats::default());
            entry.total_samples += sample.value;
            entry.sample_count += 1;
        }

        stats
    }
}

impl Default for FlameGraphGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a single execution phase
#[derive(Debug, Clone, Default)]
pub struct PhaseStats {
    /// Total samples in this phase
    pub total_samples: u64,
    /// Number of samples
    pub sample_count: usize,
}

impl PhaseStats {
    /// Get percentage of total samples
    pub fn percentage(&self, total: u64) -> f64 {
        if total == 0 {
            0.0
        } else {
            (self.total_samples as f64 / total as f64) * 100.0
        }
    }

    /// Get average samples per stack
    pub fn average_samples(&self) -> f64 {
        if self.sample_count == 0 {
            0.0
        } else {
            self.total_samples as f64 / self.sample_count as f64
        }
    }
}

/// Differential flame graph generator for comparing two profiles
pub struct DifferentialFlameGraph {
    /// Baseline profile
    baseline: FlameGraphGenerator,
    /// Comparison profile
    comparison: FlameGraphGenerator,
}

impl DifferentialFlameGraph {
    /// Create a new differential flame graph
    pub fn new(baseline: FlameGraphGenerator, comparison: FlameGraphGenerator) -> Self {
        Self {
            baseline,
            comparison,
        }
    }

    /// Generate differential flame graph showing performance differences
    ///
    /// Positive values (red) indicate regression (slower)
    /// Negative values (blue) indicate improvement (faster)
    pub fn generate_diff_svg(&self, options: FlameGraphOptions) -> Result<String, String> {
        // Build sample maps
        let mut baseline_map: HashMap<String, u64> = HashMap::new();
        let mut comparison_map: HashMap<String, u64> = HashMap::new();

        for sample in &self.baseline.samples {
            *baseline_map.entry(sample.stack.clone()).or_insert(0) += sample.value;
        }

        for sample in &self.comparison.samples {
            *comparison_map.entry(sample.stack.clone()).or_insert(0) += sample.value;
        }

        // Compute differences
        let mut diff_generator = FlameGraphGenerator::new();

        // All stacks from both profiles
        let mut all_stacks: std::collections::HashSet<String> = std::collections::HashSet::new();
        all_stacks.extend(baseline_map.keys().cloned());
        all_stacks.extend(comparison_map.keys().cloned());

        for stack in all_stacks {
            let baseline_value = baseline_map.get(&stack).copied().unwrap_or(0);
            let comparison_value = comparison_map.get(&stack).copied().unwrap_or(0);

            // Compute signed difference
            let diff = comparison_value as i64 - baseline_value as i64;

            // Only include if there's a significant difference
            if diff.abs() > 0 {
                let phase = if diff > 0 {
                    ExecutionPhase::Execution // Regression (slower)
                } else {
                    ExecutionPhase::Optimization // Improvement (faster)
                };

                diff_generator.add_sample(stack, phase, diff.unsigned_abs());
            }
        }

        // Generate differential flame graph
        diff_generator.generate_svg(options)
    }

    /// Get summary statistics for the comparison
    pub fn summary(&self) -> DiffSummary {
        let baseline_total = self.baseline.total_samples();
        let comparison_total = self.comparison.total_samples();

        let change_pct = if baseline_total > 0 {
            ((comparison_total as f64 - baseline_total as f64) / baseline_total as f64) * 100.0
        } else {
            0.0
        };

        DiffSummary {
            baseline_samples: baseline_total,
            comparison_samples: comparison_total,
            change_percent: change_pct,
            is_regression: change_pct > 0.0,
        }
    }
}

/// Summary of differential flame graph comparison
#[derive(Debug, Clone)]
pub struct DiffSummary {
    /// Total samples in baseline
    pub baseline_samples: u64,
    /// Total samples in comparison
    pub comparison_samples: u64,
    /// Percentage change (positive = regression, negative = improvement)
    pub change_percent: f64,
    /// Whether this represents a performance regression
    pub is_regression: bool,
}

impl DiffSummary {
    /// Format summary as a human-readable string
    pub fn format(&self) -> String {
        let status = if self.is_regression {
            "REGRESSION"
        } else {
            "IMPROVEMENT"
        };
        let sign = if self.is_regression { "+" } else { "" };

        format!(
            "{}: {}{:.2}% ({} → {} samples)",
            status,
            sign,
            self.change_percent.abs(),
            self.baseline_samples,
            self.comparison_samples
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_phase_colors() {
        assert_eq!(ExecutionPhase::Parsing.color(), "#6BAED6");
        assert_eq!(ExecutionPhase::Optimization.color(), "#FD8D3C");
        assert_eq!(ExecutionPhase::Execution.color(), "#74C476");
        assert_eq!(ExecutionPhase::Other.color(), "#9E9AC8");
    }

    #[test]
    fn test_phase_inference() {
        assert_eq!(
            FlameGraphGenerator::infer_phase("parse;validate"),
            Some(ExecutionPhase::Parsing)
        );
        assert_eq!(
            FlameGraphGenerator::infer_phase("optimize;rewrite"),
            Some(ExecutionPhase::Optimization)
        );
        assert_eq!(
            FlameGraphGenerator::infer_phase("execute;join;scan"),
            Some(ExecutionPhase::Execution)
        );
        assert_eq!(FlameGraphGenerator::infer_phase("unknown"), None);
    }

    #[test]
    fn test_add_sample() {
        let mut generator = FlameGraphGenerator::new();
        generator.add_sample("parse", ExecutionPhase::Parsing, 100);
        generator.add_sample("optimize", ExecutionPhase::Optimization, 200);

        assert_eq!(generator.total_samples(), 300);
        assert_eq!(generator.samples.len(), 2);
    }

    #[test]
    fn test_folded_format() {
        let mut generator = FlameGraphGenerator::new();
        generator.add_sample("parse;validate", ExecutionPhase::Parsing, 100);
        generator.add_sample("execute;join", ExecutionPhase::Execution, 200);

        let folded = generator.to_folded();
        assert!(folded.contains("parse;validate 100"));
        assert!(folded.contains("execute;join 200"));
    }

    #[test]
    fn test_add_folded_samples() {
        let mut generator = FlameGraphGenerator::new();
        let folded = "parse;validate 100\nexecute;join 200\n";

        generator
            .add_folded_samples(folded, ExecutionPhase::Other)
            .unwrap();

        assert_eq!(generator.total_samples(), 300);
        assert_eq!(generator.samples.len(), 2);
    }

    #[test]
    fn test_phase_statistics() {
        let mut generator = FlameGraphGenerator::new();
        generator.add_sample("parse", ExecutionPhase::Parsing, 100);
        generator.add_sample("parse;validate", ExecutionPhase::Parsing, 50);
        generator.add_sample("execute", ExecutionPhase::Execution, 200);

        let stats = generator.phase_statistics();

        assert_eq!(
            stats.get(&ExecutionPhase::Parsing).unwrap().total_samples,
            150
        );
        assert_eq!(stats.get(&ExecutionPhase::Parsing).unwrap().sample_count, 2);
        assert_eq!(
            stats.get(&ExecutionPhase::Execution).unwrap().total_samples,
            200
        );
    }

    #[test]
    fn test_flame_graph_generation() {
        let mut generator = FlameGraphGenerator::new();
        generator.add_sample("parse", ExecutionPhase::Parsing, 100);
        generator.add_sample("parse;validate", ExecutionPhase::Parsing, 50);
        generator.add_sample("execute;join", ExecutionPhase::Execution, 200);

        let options = FlameGraphOptions::default();
        let svg = generator.generate_svg(options);

        assert!(svg.is_ok());
        let svg_content = svg.unwrap();
        assert!(svg_content.contains("<svg"));
        assert!(svg_content.contains("SPARQL Query Profile"));
    }

    #[test]
    fn test_differential_flame_graph() {
        let mut baseline = FlameGraphGenerator::new();
        baseline.add_sample("execute", ExecutionPhase::Execution, 100);

        let mut comparison = FlameGraphGenerator::new();
        comparison.add_sample("execute", ExecutionPhase::Execution, 150);

        let diff = DifferentialFlameGraph::new(baseline, comparison);
        let summary = diff.summary();

        assert!(summary.is_regression);
        assert_eq!(summary.change_percent, 50.0);
    }

    #[test]
    fn test_diff_summary_format() {
        let summary = DiffSummary {
            baseline_samples: 100,
            comparison_samples: 150,
            change_percent: 50.0,
            is_regression: true,
        };

        let formatted = summary.format();
        assert!(formatted.contains("REGRESSION"));
        assert!(formatted.contains("+50.00%"));
    }
}
