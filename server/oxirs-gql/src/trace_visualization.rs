//! Trace Visualization and Analysis Tools
//!
//! This module provides tools for visualizing and analyzing distributed traces,
//! helping to understand system behavior and identify performance issues.
//!
//! # Features
//!
//! - **Timeline Visualization**: Gantt-chart style trace timeline
//! - **Flame Graph**: CPU-style flame graph for span hierarchies
//! - **Dependency Graph**: Service dependency visualization
//! - **Critical Path**: Identify the critical path in trace execution
//! - **Statistics**: Aggregate trace statistics and metrics
//! - **Export Formats**: JSON, HTML, SVG, ASCII
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_gql::trace_visualization::{TraceVisualizer, VisualizationFormat};
//!
//! let visualizer = TraceVisualizer::new();
//! let timeline = visualizer.generate_timeline(&spans, VisualizationFormat::HTML)?;
//! let critical_path = visualizer.find_critical_path(&spans)?;
//! ```

use crate::trace_correlation::Span;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Visualization format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualizationFormat {
    /// JSON format
    JSON,
    /// HTML format
    HTML,
    /// SVG format
    SVG,
    /// ASCII art
    ASCII,
    /// Mermaid diagram
    Mermaid,
}

/// Timeline entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    /// Span ID
    pub span_id: String,
    /// Span name
    pub name: String,
    /// Start offset from trace start (ms)
    pub start_offset_ms: u64,
    /// Duration (ms)
    pub duration_ms: u64,
    /// Depth level
    pub depth: usize,
    /// Parent span ID
    pub parent_id: Option<String>,
}

/// Flame graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameGraphNode {
    /// Node name
    pub name: String,
    /// Value (duration in ms)
    pub value: u64,
    /// Children nodes
    pub children: Vec<FlameGraphNode>,
}

/// Dependency edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// Source service
    pub from: String,
    /// Target service
    pub to: String,
    /// Number of calls
    pub call_count: usize,
    /// Total duration (ms)
    pub total_duration_ms: u64,
}

/// Critical path segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPathSegment {
    /// Span ID
    pub span_id: String,
    /// Span name
    pub name: String,
    /// Duration (ms)
    pub duration_ms: u64,
    /// Cumulative duration (ms)
    pub cumulative_duration_ms: u64,
}

/// Trace statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStatistics {
    /// Total number of spans
    pub total_spans: usize,
    /// Total trace duration (ms)
    pub total_duration_ms: u64,
    /// Maximum depth
    pub max_depth: usize,
    /// Number of services
    pub service_count: usize,
    /// Number of errors
    pub error_count: usize,
    /// Average span duration (ms)
    pub avg_span_duration_ms: f64,
    /// Span duration percentiles (p50, p95, p99)
    pub duration_percentiles: (u64, u64, u64),
}

/// Trace visualizer
pub struct TraceVisualizer;

impl TraceVisualizer {
    /// Create new trace visualizer
    pub fn new() -> Self {
        Self
    }

    /// Generate timeline visualization
    pub fn generate_timeline(
        &self,
        spans: &[Span],
        format: VisualizationFormat,
    ) -> Result<String, String> {
        if spans.is_empty() {
            return Err("No spans to visualize".to_string());
        }

        let entries = self.build_timeline_entries(spans)?;

        match format {
            VisualizationFormat::JSON => Ok(serde_json::to_string_pretty(&entries)
                .map_err(|e| format!("JSON serialization failed: {}", e))?),
            VisualizationFormat::HTML => Ok(self.timeline_to_html(&entries)),
            VisualizationFormat::ASCII => Ok(self.timeline_to_ascii(&entries)),
            VisualizationFormat::Mermaid => Ok(self.timeline_to_mermaid(&entries)),
            _ => Err("Unsupported format for timeline".to_string()),
        }
    }

    /// Build timeline entries from spans
    fn build_timeline_entries(&self, spans: &[Span]) -> Result<Vec<TimelineEntry>, String> {
        if spans.is_empty() {
            return Ok(Vec::new());
        }

        // Find trace start time
        let trace_start = spans
            .iter()
            .map(|s| s.start_time)
            .min()
            .ok_or("No start time found")?;

        // Build span hierarchy
        let hierarchy = self.build_hierarchy(spans);

        // Calculate depths
        let depths = self.calculate_depths(spans, &hierarchy);

        // Create timeline entries
        let mut entries: Vec<TimelineEntry> = spans
            .iter()
            .map(|span| {
                let start_offset = span
                    .start_time
                    .duration_since(trace_start)
                    .unwrap_or_default()
                    .as_millis() as u64;

                TimelineEntry {
                    span_id: span.span_id.clone(),
                    name: span.name.clone(),
                    start_offset_ms: start_offset,
                    duration_ms: span.duration_ms.unwrap_or(0),
                    depth: *depths.get(&span.span_id).unwrap_or(&0),
                    parent_id: span.parent_span_id.clone(),
                }
            })
            .collect();

        // Sort by start time
        entries.sort_by_key(|e| e.start_offset_ms);

        Ok(entries)
    }

    /// Build span hierarchy
    fn build_hierarchy(&self, spans: &[Span]) -> HashMap<String, Vec<String>> {
        let mut hierarchy: HashMap<String, Vec<String>> = HashMap::new();

        for span in spans {
            if let Some(parent_id) = &span.parent_span_id {
                hierarchy
                    .entry(parent_id.clone())
                    .or_default()
                    .push(span.span_id.clone());
            }
        }

        hierarchy
    }

    /// Calculate depths for each span
    fn calculate_depths(
        &self,
        spans: &[Span],
        hierarchy: &HashMap<String, Vec<String>>,
    ) -> HashMap<String, usize> {
        let mut depths = HashMap::new();

        // Find root spans (no parent)
        let roots: Vec<_> = spans
            .iter()
            .filter(|s| s.parent_span_id.is_none())
            .collect();

        // Calculate depth recursively
        for root in roots {
            self.calculate_depth_recursive(&root.span_id, 0, hierarchy, &mut depths);
        }

        depths
    }

    /// Calculate depth recursively
    #[allow(clippy::only_used_in_recursion)]
    fn calculate_depth_recursive(
        &self,
        span_id: &str,
        depth: usize,
        hierarchy: &HashMap<String, Vec<String>>,
        depths: &mut HashMap<String, usize>,
    ) {
        depths.insert(span_id.to_string(), depth);

        if let Some(children) = hierarchy.get(span_id) {
            for child_id in children {
                self.calculate_depth_recursive(child_id, depth + 1, hierarchy, depths);
            }
        }
    }

    /// Convert timeline to HTML
    fn timeline_to_html(&self, entries: &[TimelineEntry]) -> String {
        let mut html = String::from("<html><head><style>\n");
        html.push_str(".timeline { font-family: monospace; }\n");
        html.push_str(".span { background: #4CAF50; color: white; padding: 2px 5px; margin: 2px; display: inline-block; }\n");
        html.push_str("</style></head><body><div class='timeline'>\n");

        for entry in entries {
            let indent = "  ".repeat(entry.depth);
            html.push_str(&format!(
                "{}<div class='span'>{} ({}ms)</div><br>\n",
                indent, entry.name, entry.duration_ms
            ));
        }

        html.push_str("</div></body></html>");
        html
    }

    /// Convert timeline to ASCII
    fn timeline_to_ascii(&self, entries: &[TimelineEntry]) -> String {
        let mut ascii = String::new();

        for entry in entries {
            let indent = "  ".repeat(entry.depth);
            ascii.push_str(&format!(
                "{}├─ {} ({}ms)\n",
                indent, entry.name, entry.duration_ms
            ));
        }

        ascii
    }

    /// Convert timeline to Mermaid diagram
    fn timeline_to_mermaid(&self, entries: &[TimelineEntry]) -> String {
        let mut mermaid = String::from(
            "gantt\n    title Trace Timeline\n    dateFormat x\n    axisFormat %L\n\n",
        );

        for entry in entries {
            mermaid.push_str(&format!(
                "    {} : {}, {}ms\n",
                entry.name, entry.start_offset_ms, entry.duration_ms
            ));
        }

        mermaid
    }

    /// Generate flame graph
    pub fn generate_flame_graph(&self, spans: &[Span]) -> Result<FlameGraphNode, String> {
        if spans.is_empty() {
            return Err("No spans to visualize".to_string());
        }

        // Find root spans
        let roots: Vec<_> = spans
            .iter()
            .filter(|s| s.parent_span_id.is_none())
            .collect();

        if roots.is_empty() {
            return Err("No root spans found".to_string());
        }

        // Build flame graph from first root
        let root = roots[0];
        Ok(self.build_flame_graph_node(root, spans))
    }

    /// Build flame graph node recursively
    #[allow(clippy::only_used_in_recursion)]
    fn build_flame_graph_node(&self, span: &Span, all_spans: &[Span]) -> FlameGraphNode {
        let children: Vec<FlameGraphNode> = all_spans
            .iter()
            .filter(|s| s.parent_span_id.as_ref() == Some(&span.span_id))
            .map(|child| self.build_flame_graph_node(child, all_spans))
            .collect();

        FlameGraphNode {
            name: span.name.clone(),
            value: span.duration_ms.unwrap_or(0),
            children,
        }
    }

    /// Generate dependency graph
    pub fn generate_dependency_graph(&self, spans: &[Span]) -> Vec<DependencyEdge> {
        let mut edges: HashMap<(String, String), DependencyEdge> = HashMap::new();

        for span in spans {
            if let Some(parent_id) = &span.parent_span_id {
                // Find parent span
                if let Some(parent) = spans.iter().find(|s| &s.span_id == parent_id) {
                    let from = parent.name.clone();
                    let to = span.name.clone();
                    let key = (from.clone(), to.clone());

                    edges
                        .entry(key)
                        .and_modify(|e| {
                            e.call_count += 1;
                            e.total_duration_ms += span.duration_ms.unwrap_or(0);
                        })
                        .or_insert(DependencyEdge {
                            from,
                            to,
                            call_count: 1,
                            total_duration_ms: span.duration_ms.unwrap_or(0),
                        });
                }
            }
        }

        edges.into_values().collect()
    }

    /// Find critical path
    pub fn find_critical_path(&self, spans: &[Span]) -> Result<Vec<CriticalPathSegment>, String> {
        if spans.is_empty() {
            return Err("No spans to analyze".to_string());
        }

        // Find root spans
        let roots: Vec<_> = spans
            .iter()
            .filter(|s| s.parent_span_id.is_none())
            .collect();

        if roots.is_empty() {
            return Err("No root spans found".to_string());
        }

        // Find critical path from first root
        let root = roots[0];
        let path = self.find_critical_path_recursive(root, spans, 0);

        Ok(path)
    }

    /// Find critical path recursively
    #[allow(clippy::only_used_in_recursion)]
    fn find_critical_path_recursive(
        &self,
        span: &Span,
        all_spans: &[Span],
        cumulative: u64,
    ) -> Vec<CriticalPathSegment> {
        let duration = span.duration_ms.unwrap_or(0);
        let new_cumulative = cumulative + duration;

        let mut path = vec![CriticalPathSegment {
            span_id: span.span_id.clone(),
            name: span.name.clone(),
            duration_ms: duration,
            cumulative_duration_ms: new_cumulative,
        }];

        // Find child with longest duration
        let longest_child = all_spans
            .iter()
            .filter(|s| s.parent_span_id.as_ref() == Some(&span.span_id))
            .max_by_key(|s| s.duration_ms.unwrap_or(0));

        if let Some(child) = longest_child {
            path.extend(self.find_critical_path_recursive(child, all_spans, new_cumulative));
        }

        path
    }

    /// Calculate trace statistics
    pub fn calculate_statistics(&self, spans: &[Span]) -> TraceStatistics {
        if spans.is_empty() {
            return TraceStatistics {
                total_spans: 0,
                total_duration_ms: 0,
                max_depth: 0,
                service_count: 0,
                error_count: 0,
                avg_span_duration_ms: 0.0,
                duration_percentiles: (0, 0, 0),
            };
        }

        let total_spans = spans.len();

        // Total duration (root span duration)
        let total_duration_ms = spans
            .iter()
            .filter(|s| s.parent_span_id.is_none())
            .map(|s| s.duration_ms.unwrap_or(0))
            .max()
            .unwrap_or(0);

        // Max depth
        let hierarchy = self.build_hierarchy(spans);
        let depths = self.calculate_depths(spans, &hierarchy);
        let max_depth = depths.values().max().copied().unwrap_or(0);

        // Service count (unique span names)
        let services: HashSet<_> = spans.iter().map(|s| s.name.as_str()).collect();
        let service_count = services.len();

        // Error count
        let error_count = spans
            .iter()
            .filter(|s| matches!(s.status, crate::trace_correlation::SpanStatus::Error))
            .count();

        // Average span duration
        let total_span_duration: u64 = spans.iter().map(|s| s.duration_ms.unwrap_or(0)).sum();
        let avg_span_duration_ms = total_span_duration as f64 / total_spans as f64;

        // Duration percentiles
        let mut durations: Vec<u64> = spans.iter().map(|s| s.duration_ms.unwrap_or(0)).collect();
        durations.sort_unstable();

        let p50 = durations[durations.len() / 2];
        let p95 = durations[durations.len() * 95 / 100];
        let p99 = durations[durations.len() * 99 / 100];

        TraceStatistics {
            total_spans,
            total_duration_ms,
            max_depth,
            service_count,
            error_count,
            avg_span_duration_ms,
            duration_percentiles: (p50, p95, p99),
        }
    }
}

impl Default for TraceVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace_correlation::{SpanKind, TraceContext};

    fn create_test_spans() -> Vec<Span> {
        let ctx = TraceContext::new("test");

        let mut root = Span::new("root".to_string(), ctx.clone(), SpanKind::Server);
        root.finish();

        let mut child1 = Span::new("child1".to_string(), ctx.clone(), SpanKind::Internal)
            .with_parent(root.span_id.clone());
        child1.finish();

        let mut child2 = Span::new("child2".to_string(), ctx, SpanKind::Internal)
            .with_parent(root.span_id.clone());
        child2.finish();

        vec![root, child1, child2]
    }

    #[test]
    fn test_visualization_format() {
        let formats = [
            VisualizationFormat::JSON,
            VisualizationFormat::HTML,
            VisualizationFormat::SVG,
            VisualizationFormat::ASCII,
            VisualizationFormat::Mermaid,
        ];

        assert_eq!(formats.len(), 5);
    }

    #[test]
    fn test_timeline_entry_creation() {
        let entry = TimelineEntry {
            span_id: "span1".to_string(),
            name: "operation".to_string(),
            start_offset_ms: 0,
            duration_ms: 100,
            depth: 0,
            parent_id: None,
        };

        assert_eq!(entry.span_id, "span1");
        assert_eq!(entry.duration_ms, 100);
    }

    #[test]
    fn test_trace_visualizer_creation() {
        let _visualizer = TraceVisualizer::new();
        // Just test that it can be created
    }

    #[test]
    fn test_build_hierarchy() {
        let visualizer = TraceVisualizer::new();
        let spans = create_test_spans();

        let hierarchy = visualizer.build_hierarchy(&spans);

        assert!(hierarchy.contains_key(&spans[0].span_id));
        assert_eq!(hierarchy.get(&spans[0].span_id).unwrap().len(), 2);
    }

    #[test]
    fn test_calculate_depths() {
        let visualizer = TraceVisualizer::new();
        let spans = create_test_spans();

        let hierarchy = visualizer.build_hierarchy(&spans);
        let depths = visualizer.calculate_depths(&spans, &hierarchy);

        assert_eq!(depths.get(&spans[0].span_id), Some(&0));
        assert_eq!(depths.get(&spans[1].span_id), Some(&1));
        assert_eq!(depths.get(&spans[2].span_id), Some(&1));
    }

    #[test]
    fn test_build_timeline_entries() {
        let visualizer = TraceVisualizer::new();
        let spans = create_test_spans();

        let entries = visualizer.build_timeline_entries(&spans).unwrap();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].depth, 0);
    }

    #[test]
    fn test_generate_timeline_json() {
        let visualizer = TraceVisualizer::new();
        let spans = create_test_spans();

        let result = visualizer.generate_timeline(&spans, VisualizationFormat::JSON);

        assert!(result.is_ok());
        let json = result.unwrap();
        assert!(json.contains("span_id"));
    }

    #[test]
    fn test_generate_timeline_html() {
        let visualizer = TraceVisualizer::new();
        let spans = create_test_spans();

        let result = visualizer.generate_timeline(&spans, VisualizationFormat::HTML);

        assert!(result.is_ok());
        let html = result.unwrap();
        assert!(html.contains("<html>"));
        assert!(html.contains("timeline"));
    }

    #[test]
    fn test_generate_timeline_ascii() {
        let visualizer = TraceVisualizer::new();
        let spans = create_test_spans();

        let result = visualizer.generate_timeline(&spans, VisualizationFormat::ASCII);

        assert!(result.is_ok());
        let ascii = result.unwrap();
        assert!(ascii.contains("├─"));
    }

    #[test]
    fn test_generate_timeline_mermaid() {
        let visualizer = TraceVisualizer::new();
        let spans = create_test_spans();

        let result = visualizer.generate_timeline(&spans, VisualizationFormat::Mermaid);

        assert!(result.is_ok());
        let mermaid = result.unwrap();
        assert!(mermaid.contains("gantt"));
    }

    #[test]
    fn test_generate_timeline_empty() {
        let visualizer = TraceVisualizer::new();
        let spans: Vec<Span> = vec![];

        let result = visualizer.generate_timeline(&spans, VisualizationFormat::JSON);

        assert!(result.is_err());
    }

    #[test]
    fn test_generate_flame_graph() {
        let visualizer = TraceVisualizer::new();
        let spans = create_test_spans();

        let result = visualizer.generate_flame_graph(&spans);

        assert!(result.is_ok());
        let root = result.unwrap();
        assert_eq!(root.name, "root");
        assert_eq!(root.children.len(), 2);
    }

    #[test]
    fn test_generate_flame_graph_empty() {
        let visualizer = TraceVisualizer::new();
        let spans: Vec<Span> = vec![];

        let result = visualizer.generate_flame_graph(&spans);

        assert!(result.is_err());
    }

    #[test]
    fn test_generate_dependency_graph() {
        let visualizer = TraceVisualizer::new();
        let spans = create_test_spans();

        let edges = visualizer.generate_dependency_graph(&spans);

        assert_eq!(edges.len(), 2);
        assert!(edges.iter().any(|e| e.from == "root" && e.to == "child1"));
        assert!(edges.iter().any(|e| e.from == "root" && e.to == "child2"));
    }

    #[test]
    fn test_find_critical_path() {
        let visualizer = TraceVisualizer::new();
        let spans = create_test_spans();

        let result = visualizer.find_critical_path(&spans);

        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(!path.is_empty());
        assert_eq!(path[0].name, "root");
    }

    #[test]
    fn test_find_critical_path_empty() {
        let visualizer = TraceVisualizer::new();
        let spans: Vec<Span> = vec![];

        let result = visualizer.find_critical_path(&spans);

        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_statistics() {
        let visualizer = TraceVisualizer::new();
        let spans = create_test_spans();

        let stats = visualizer.calculate_statistics(&spans);

        assert_eq!(stats.total_spans, 3);
        assert_eq!(stats.max_depth, 1);
        assert_eq!(stats.service_count, 3); // root, child1, child2
        assert_eq!(stats.error_count, 0);
    }

    #[test]
    fn test_calculate_statistics_empty() {
        let visualizer = TraceVisualizer::new();
        let spans: Vec<Span> = vec![];

        let stats = visualizer.calculate_statistics(&spans);

        assert_eq!(stats.total_spans, 0);
        assert_eq!(stats.total_duration_ms, 0);
    }

    #[test]
    fn test_critical_path_segment() {
        let segment = CriticalPathSegment {
            span_id: "span1".to_string(),
            name: "operation".to_string(),
            duration_ms: 100,
            cumulative_duration_ms: 100,
        };

        assert_eq!(segment.duration_ms, 100);
        assert_eq!(segment.cumulative_duration_ms, 100);
    }

    #[test]
    fn test_dependency_edge() {
        let edge = DependencyEdge {
            from: "service1".to_string(),
            to: "service2".to_string(),
            call_count: 5,
            total_duration_ms: 500,
        };

        assert_eq!(edge.call_count, 5);
        assert_eq!(edge.total_duration_ms, 500);
    }

    #[test]
    fn test_flame_graph_node() {
        let node = FlameGraphNode {
            name: "root".to_string(),
            value: 100,
            children: vec![],
        };

        assert_eq!(node.name, "root");
        assert_eq!(node.value, 100);
        assert!(node.children.is_empty());
    }

    #[test]
    fn test_timeline_to_html_format() {
        let visualizer = TraceVisualizer::new();
        let entries = vec![TimelineEntry {
            span_id: "span1".to_string(),
            name: "test".to_string(),
            start_offset_ms: 0,
            duration_ms: 100,
            depth: 0,
            parent_id: None,
        }];

        let html = visualizer.timeline_to_html(&entries);

        assert!(html.contains("<html>"));
        assert!(html.contains("test"));
        assert!(html.contains("100ms"));
    }

    #[test]
    fn test_timeline_to_ascii_format() {
        let visualizer = TraceVisualizer::new();
        let entries = vec![TimelineEntry {
            span_id: "span1".to_string(),
            name: "test".to_string(),
            start_offset_ms: 0,
            duration_ms: 100,
            depth: 0,
            parent_id: None,
        }];

        let ascii = visualizer.timeline_to_ascii(&entries);

        assert!(ascii.contains("├─"));
        assert!(ascii.contains("test"));
        assert!(ascii.contains("100ms"));
    }

    #[test]
    fn test_statistics_percentiles() {
        let visualizer = TraceVisualizer::new();
        let spans = create_test_spans();

        let stats = visualizer.calculate_statistics(&spans);

        let (p50, p95, p99) = stats.duration_percentiles;
        assert!(p95 >= p50);
        assert!(p99 >= p95);
    }
}
