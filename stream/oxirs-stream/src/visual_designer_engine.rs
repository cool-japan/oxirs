//! # Visual Designer — Engine
//!
//! Core engine components: `VisualStreamDesigner`, `PipelineValidator`,
//! and `PipelineOptimizer` with SVG/DOT/Mermaid export, validation logic,
//! and automatic pipeline optimization.

use anyhow::{anyhow, Result};
use chrono::Utc;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

use crate::visual_designer_types::{
    BackpressureStrategy, Breakpoint, DataType, DebugMetrics, DebuggerConfig, DebuggerState,
    EdgeConfig, EdgeMetadata, EdgeStyle, EdgeType, ErrorHandling, ExportFormat, ImpactLevel,
    LineType, NodeConfig, NodeMetadata, NodeStatus, NodeType, OptimizationResult,
    OptimizationSuggestion, OptimizationType, PipelineDebugger, PipelineEdge, PipelineInfo,
    PipelineMetadata, PipelineMetrics, PipelineNode, PipelineOptimizer, PipelineValidator, Port,
    PortType, Position, ResourceLimits, ValidationError, ValidationErrorType, ValidationResult,
    ValidationWarning, ValidationWarningType, VisualDesignerConfig, VisualPipeline,
};

/// Visual designer main struct
pub struct VisualStreamDesigner {
    config: VisualDesignerConfig,
    pipelines: Arc<RwLock<HashMap<String, VisualPipeline>>>,
    debuggers: Arc<RwLock<HashMap<String, PipelineDebugger>>>,
    validator: Arc<PipelineValidator>,
    optimizer: Arc<PipelineOptimizer>,
}

impl VisualStreamDesigner {
    /// Create a new visual stream designer
    pub fn new(config: VisualDesignerConfig) -> Self {
        Self {
            config: config.clone(),
            pipelines: Arc::new(RwLock::new(HashMap::new())),
            debuggers: Arc::new(RwLock::new(HashMap::new())),
            validator: Arc::new(PipelineValidator::new(config.clone())),
            optimizer: Arc::new(PipelineOptimizer::new(config)),
        }
    }

    /// Create a new empty pipeline
    pub async fn create_pipeline(
        &self,
        name: String,
        description: Option<String>,
    ) -> Result<String> {
        let pipeline = VisualPipeline {
            id: Uuid::new_v4().to_string(),
            name,
            description,
            version: "1.0.0".to_string(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            metadata: PipelineMetadata {
                created_at: Utc::now(),
                updated_at: Utc::now(),
                author: None,
                tags: Vec::new(),
                properties: HashMap::new(),
            },
            validation_result: None,
        };

        let id = pipeline.id.clone();
        self.pipelines.write().await.insert(id.clone(), pipeline);

        info!("Created new pipeline: {}", id);
        Ok(id)
    }

    /// Add a node to the pipeline
    pub async fn add_node(
        &self,
        pipeline_id: &str,
        name: String,
        node_type: NodeType,
        position: Position,
    ) -> Result<String> {
        let mut pipelines = self.pipelines.write().await;
        let pipeline = pipelines
            .get_mut(pipeline_id)
            .ok_or_else(|| anyhow!("Pipeline not found"))?;

        let node = PipelineNode {
            id: Uuid::new_v4().to_string(),
            name,
            node_type: node_type.clone(),
            position,
            config: Self::default_node_config(&node_type),
            metadata: NodeMetadata {
                created_at: Utc::now(),
                updated_at: Utc::now(),
                version: "1.0.0".to_string(),
                author: None,
                description: None,
                tags: Vec::new(),
            },
            status: NodeStatus::Idle,
        };

        let node_id = node.id.clone();
        pipeline.nodes.insert(node_id.clone(), node);
        pipeline.metadata.updated_at = Utc::now();

        debug!("Added node {} to pipeline {}", node_id, pipeline_id);
        Ok(node_id)
    }

    /// Add an edge connecting two nodes
    pub async fn add_edge(
        &self,
        pipeline_id: &str,
        source_node_id: String,
        source_port_id: String,
        target_node_id: String,
        target_port_id: String,
    ) -> Result<String> {
        let mut pipelines = self.pipelines.write().await;
        let pipeline = pipelines
            .get_mut(pipeline_id)
            .ok_or_else(|| anyhow!("Pipeline not found"))?;

        // Validate nodes exist
        if !pipeline.nodes.contains_key(&source_node_id) {
            return Err(anyhow!("Source node not found"));
        }
        if !pipeline.nodes.contains_key(&target_node_id) {
            return Err(anyhow!("Target node not found"));
        }

        let edge = PipelineEdge {
            id: Uuid::new_v4().to_string(),
            source_node_id,
            source_port_id,
            target_node_id,
            target_port_id,
            edge_type: EdgeType::DataFlow,
            config: EdgeConfig {
                buffer_size: 1000,
                backpressure_strategy: BackpressureStrategy::Block,
                error_handling: ErrorHandling::Propagate,
            },
            metadata: EdgeMetadata {
                created_at: Utc::now(),
                label: None,
                style: EdgeStyle {
                    color: "#000000".to_string(),
                    thickness: 2.0,
                    line_type: LineType::Solid,
                },
            },
        };

        let edge_id = edge.id.clone();
        pipeline.edges.insert(edge_id.clone(), edge);
        pipeline.metadata.updated_at = Utc::now();

        debug!("Added edge {} to pipeline {}", edge_id, pipeline_id);
        Ok(edge_id)
    }

    /// Validate a pipeline
    pub async fn validate_pipeline(&self, pipeline_id: &str) -> Result<ValidationResult> {
        let pipelines = self.pipelines.read().await;
        let pipeline = pipelines
            .get(pipeline_id)
            .ok_or_else(|| anyhow!("Pipeline not found"))?;

        self.validator.validate(pipeline).await
    }

    /// Optimize a pipeline
    pub async fn optimize_pipeline(&self, pipeline_id: &str) -> Result<OptimizationResult> {
        let pipelines = self.pipelines.read().await;
        let pipeline = pipelines
            .get(pipeline_id)
            .ok_or_else(|| anyhow!("Pipeline not found"))?;

        self.optimizer.optimize(pipeline).await
    }

    /// Export pipeline to specified format
    pub async fn export_pipeline(&self, pipeline_id: &str, format: ExportFormat) -> Result<String> {
        let pipelines = self.pipelines.read().await;
        let pipeline = pipelines
            .get(pipeline_id)
            .ok_or_else(|| anyhow!("Pipeline not found"))?;

        match format {
            ExportFormat::Json => serde_json::to_string_pretty(pipeline)
                .map_err(|e| anyhow!("JSON export failed: {}", e)),
            ExportFormat::Yaml => {
                serde_yaml::to_string(pipeline).map_err(|e| anyhow!("YAML export failed: {}", e))
            }
            ExportFormat::Dot => export_dot(pipeline),
            ExportFormat::Mermaid => export_mermaid(pipeline),
            ExportFormat::Svg => export_svg(pipeline),
            ExportFormat::Png => Err(anyhow!(
                "PNG export requires SVG rasterization; use ExportFormat::Svg and convert with \
                 resvg/tiny-skia externally (these crates depend on miniz_oxide which is \
                 prohibited by COOLJAPAN Pure Rust Policy)"
            )),
        }
    }

    /// Import pipeline from string
    pub async fn import_pipeline(&self, data: &str, format: ExportFormat) -> Result<String> {
        let pipeline: VisualPipeline = match format {
            ExportFormat::Json => {
                serde_json::from_str(data).map_err(|e| anyhow!("JSON import failed: {}", e))?
            }
            ExportFormat::Yaml => {
                serde_yaml::from_str(data).map_err(|e| anyhow!("YAML import failed: {}", e))?
            }
            _ => return Err(anyhow!("Import format not supported")),
        };

        let id = pipeline.id.clone();
        self.pipelines.write().await.insert(id.clone(), pipeline);

        info!("Imported pipeline: {}", id);
        Ok(id)
    }

    /// Create a debugger for a pipeline
    pub async fn create_debugger(
        &self,
        pipeline_id: &str,
        config: DebuggerConfig,
    ) -> Result<String> {
        let pipelines = self.pipelines.read().await;
        let pipeline = pipelines
            .get(pipeline_id)
            .ok_or_else(|| anyhow!("Pipeline not found"))?;

        let max_event_history = config.max_event_history;
        let debugger = PipelineDebugger {
            pipeline: pipeline.clone(),
            config,
            state: Arc::new(RwLock::new(DebuggerState {
                is_running: false,
                is_paused: false,
                current_node_id: None,
                execution_stack: Vec::new(),
                variables: HashMap::new(),
                metrics: DebugMetrics::default(),
            })),
            breakpoints: Arc::new(RwLock::new(HashMap::new())),
            event_history: Arc::new(RwLock::new(std::collections::VecDeque::with_capacity(
                max_event_history,
            ))),
        };

        let debugger_id = Uuid::new_v4().to_string();
        self.debuggers
            .write()
            .await
            .insert(debugger_id.clone(), debugger);

        info!(
            "Created debugger {} for pipeline {}",
            debugger_id, pipeline_id
        );
        Ok(debugger_id)
    }

    /// Add a breakpoint to a debugger
    pub async fn add_breakpoint(
        &self,
        debugger_id: &str,
        node_id: String,
        condition: Option<String>,
    ) -> Result<String> {
        let debuggers = self.debuggers.read().await;
        let debugger = debuggers
            .get(debugger_id)
            .ok_or_else(|| anyhow!("Debugger not found"))?;

        let breakpoint = Breakpoint {
            id: Uuid::new_v4().to_string(),
            node_id,
            condition,
            enabled: true,
            hit_count: 0,
            max_hits: None,
        };

        let breakpoint_id = breakpoint.id.clone();
        debugger
            .breakpoints
            .write()
            .await
            .insert(breakpoint_id.clone(), breakpoint);

        debug!(
            "Added breakpoint {} to debugger {}",
            breakpoint_id, debugger_id
        );
        Ok(breakpoint_id)
    }

    /// Get debugger state
    pub async fn get_debugger_state(&self, debugger_id: &str) -> Result<DebuggerState> {
        let debuggers = self.debuggers.read().await;
        let debugger = debuggers
            .get(debugger_id)
            .ok_or_else(|| anyhow!("Debugger not found"))?;

        let state = debugger.state.read().await.clone();
        drop(debuggers); // Explicitly drop to avoid borrow issues
        Ok(state)
    }

    /// Get event history from debugger
    pub async fn get_event_history(
        &self,
        debugger_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<crate::visual_designer_types::DebugEvent>> {
        let debuggers = self.debuggers.read().await;
        let debugger = debuggers
            .get(debugger_id)
            .ok_or_else(|| anyhow!("Debugger not found"))?;

        let history = debugger.event_history.read().await;
        let limit = limit.unwrap_or(history.len());

        Ok(history.iter().rev().take(limit).cloned().collect())
    }

    /// Get default node configuration based on type
    fn default_node_config(_node_type: &NodeType) -> NodeConfig {
        NodeConfig {
            parameters: HashMap::new(),
            input_ports: vec![Port {
                id: "input".to_string(),
                name: "Input".to_string(),
                port_type: PortType::Input,
                data_type: DataType::StreamEvent,
                required: true,
            }],
            output_ports: vec![Port {
                id: "output".to_string(),
                name: "Output".to_string(),
                port_type: PortType::Output,
                data_type: DataType::StreamEvent,
                required: false,
            }],
            resource_limits: ResourceLimits {
                max_memory_mb: Some(1024),
                max_cpu_percent: Some(50.0),
                max_execution_time_ms: Some(5000),
                max_events_per_second: Some(10000),
            },
        }
    }

    /// List all pipelines
    pub async fn list_pipelines(&self) -> Vec<PipelineInfo> {
        let pipelines = self.pipelines.read().await;
        pipelines
            .values()
            .map(|p| PipelineInfo {
                id: p.id.clone(),
                name: p.name.clone(),
                version: p.version.clone(),
                node_count: p.nodes.len(),
                edge_count: p.edges.len(),
                created_at: p.metadata.created_at,
                updated_at: p.metadata.updated_at,
            })
            .collect()
    }

    /// Get pipeline details
    pub async fn get_pipeline(&self, pipeline_id: &str) -> Result<VisualPipeline> {
        let pipelines = self.pipelines.read().await;
        pipelines
            .get(pipeline_id)
            .cloned()
            .ok_or_else(|| anyhow!("Pipeline not found"))
    }

    /// Delete a pipeline
    pub async fn delete_pipeline(&self, pipeline_id: &str) -> Result<()> {
        let mut pipelines = self.pipelines.write().await;
        pipelines
            .remove(pipeline_id)
            .ok_or_else(|| anyhow!("Pipeline not found"))?;

        info!("Deleted pipeline: {}", pipeline_id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pipeline validator
// ---------------------------------------------------------------------------

impl PipelineValidator {
    pub fn new(config: VisualDesignerConfig) -> Self {
        Self { config }
    }

    pub async fn validate(&self, pipeline: &VisualPipeline) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check for cyclic dependencies
        if self.has_cycle(pipeline) {
            errors.push(ValidationError {
                error_type: ValidationErrorType::CyclicDependency,
                message: "Pipeline contains cyclic dependencies".to_string(),
                node_id: None,
                edge_id: None,
            });
        }

        // Check for disconnected nodes
        let disconnected = self.find_disconnected_nodes(pipeline);
        for node_id in disconnected {
            warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::UnusedPort,
                message: format!("Node {} is disconnected", node_id),
                node_id: Some(node_id),
                suggestion: Some("Connect this node to the pipeline".to_string()),
            });
        }

        // Check data type compatibility
        for edge in pipeline.edges.values() {
            if let Err(e) = self.check_data_type_compatibility(pipeline, edge) {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::IncompatibleDataTypes,
                    message: e.to_string(),
                    node_id: None,
                    edge_id: Some(edge.id.clone()),
                });
            }
        }

        // Check resource limits
        if pipeline.nodes.len() > self.config.max_nodes {
            errors.push(ValidationError {
                error_type: ValidationErrorType::ResourceLimitExceeded,
                message: format!(
                    "Pipeline exceeds maximum node count: {} > {}",
                    pipeline.nodes.len(),
                    self.config.max_nodes
                ),
                node_id: None,
                edge_id: None,
            });
        }

        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            validated_at: Utc::now(),
        })
    }

    fn has_cycle(&self, pipeline: &VisualPipeline) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for node_id in pipeline.nodes.keys() {
            if Self::detect_cycle_util(pipeline, node_id, &mut visited, &mut rec_stack) {
                return true;
            }
        }

        false
    }

    fn detect_cycle_util(
        pipeline: &VisualPipeline,
        node_id: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        if !visited.contains(node_id) {
            visited.insert(node_id.to_string());
            rec_stack.insert(node_id.to_string());

            // Get all neighbors
            let neighbors: Vec<String> = pipeline
                .edges
                .values()
                .filter(|e| e.source_node_id == node_id)
                .map(|e| e.target_node_id.clone())
                .collect();

            for neighbor in neighbors {
                if !visited.contains(&neighbor)
                    && Self::detect_cycle_util(pipeline, &neighbor, visited, rec_stack)
                {
                    return true;
                }
                if rec_stack.contains(&neighbor) {
                    return true;
                }
            }
        }

        rec_stack.remove(node_id);
        false
    }

    fn find_disconnected_nodes(&self, pipeline: &VisualPipeline) -> Vec<String> {
        let mut connected = HashSet::new();

        for edge in pipeline.edges.values() {
            connected.insert(edge.source_node_id.clone());
            connected.insert(edge.target_node_id.clone());
        }

        pipeline
            .nodes
            .keys()
            .filter(|id| !connected.contains(*id))
            .cloned()
            .collect()
    }

    fn check_data_type_compatibility(
        &self,
        _pipeline: &VisualPipeline,
        _edge: &PipelineEdge,
    ) -> Result<()> {
        // Simplified check - in real implementation, would verify port data types match
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pipeline optimizer
// ---------------------------------------------------------------------------

impl PipelineOptimizer {
    pub fn new(config: VisualDesignerConfig) -> Self {
        Self { config }
    }

    pub async fn optimize(&self, pipeline: &VisualPipeline) -> Result<OptimizationResult> {
        let mut suggestions = Vec::new();

        // Analyze pipeline structure
        let metrics = self.analyze_structure(pipeline);

        // Generate optimization suggestions — trigger when either the longest path exceeds
        // 10 hops (max_chain_length) or the average depth across all nodes exceeds 10.
        if metrics.max_chain_length > 10 || metrics.avg_chain_length > 10.0 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::ReduceChainLength,
                impact: ImpactLevel::High,
                description: "Consider breaking long chains into parallel branches".to_string(),
                estimated_improvement: 30.0,
            });
        }

        if metrics.parallel_opportunities > 0 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::IncreaseParallelism,
                impact: ImpactLevel::Medium,
                description: format!(
                    "Found {} opportunities for parallel processing",
                    metrics.parallel_opportunities
                ),
                estimated_improvement: 15.0 * metrics.parallel_opportunities as f64,
            });
        }

        Ok(OptimizationResult {
            original_metrics: metrics,
            suggestions,
            optimized_at: Utc::now(),
        })
    }

    fn analyze_structure(&self, pipeline: &VisualPipeline) -> PipelineMetrics {
        let mut total_chain_length = 0usize;
        let mut max_depth = 0usize;
        let chain_count = pipeline.nodes.len();

        // Simple chain length calculation
        for node_id in pipeline.nodes.keys() {
            let depth = self.calculate_depth(pipeline, node_id);
            total_chain_length += depth;
            if depth > max_depth {
                max_depth = depth;
            }
        }

        PipelineMetrics {
            node_count: pipeline.nodes.len(),
            edge_count: pipeline.edges.len(),
            avg_chain_length: if chain_count > 0 {
                total_chain_length as f64 / chain_count as f64
            } else {
                0.0
            },
            max_chain_length: max_depth,
            parallel_opportunities: self.count_parallel_opportunities(pipeline),
            bottleneck_nodes: Vec::new(),
        }
    }

    fn calculate_depth(&self, pipeline: &VisualPipeline, node_id: &str) -> usize {
        let mut depth = 0;
        let mut current = node_id.to_string();

        while let Some(edge) = pipeline
            .edges
            .values()
            .find(|e| e.target_node_id == current)
        {
            depth += 1;
            current = edge.source_node_id.clone();
        }

        depth
    }

    fn count_parallel_opportunities(&self, pipeline: &VisualPipeline) -> usize {
        let mut count = 0;

        for node_id in pipeline.nodes.keys() {
            let outgoing: Vec<_> = pipeline
                .edges
                .values()
                .filter(|e| e.source_node_id == *node_id)
                .collect();

            if outgoing.len() > 1 {
                count += 1;
            }
        }

        count
    }
}

// ---------------------------------------------------------------------------
// Export helpers (free functions)
// ---------------------------------------------------------------------------

/// Export pipeline to SVG format
pub fn export_svg(pipeline: &VisualPipeline) -> Result<String> {
    // Layout constants
    const NODE_W: f64 = 140.0;
    const NODE_H: f64 = 44.0;
    const H_GAP: f64 = 60.0;
    const V_GAP: f64 = 70.0;
    const MARGIN: f64 = 30.0;

    // Collect nodes sorted deterministically so layout is stable.
    let mut nodes: Vec<(&String, &PipelineNode)> = pipeline.nodes.iter().collect();
    nodes.sort_by_key(|(id, _)| id.as_str());

    // Assign positions: use stored position when non-zero, otherwise fall back to
    // a simple grid layout so the SVG is always valid even for freshly created pipelines.
    let cols = ((nodes.len() as f64).sqrt().ceil() as usize).max(1);
    let mut node_positions: HashMap<&str, (f64, f64)> = HashMap::new();

    for (idx, (id, node)) in nodes.iter().enumerate() {
        let (cx, cy) = if node.position.x.abs() > 1e-9 || node.position.y.abs() > 1e-9 {
            (node.position.x, node.position.y)
        } else {
            let col = idx % cols;
            let row = idx / cols;
            (
                MARGIN + col as f64 * (NODE_W + H_GAP),
                MARGIN + row as f64 * (NODE_H + V_GAP),
            )
        };
        node_positions.insert(id.as_str(), (cx, cy));
    }

    // Canvas size
    let total_w = if node_positions.is_empty() {
        200.0
    } else {
        node_positions
            .values()
            .map(|(x, _)| *x)
            .fold(f64::NEG_INFINITY, f64::max)
            + NODE_W
            + MARGIN
    };
    let total_h = if node_positions.is_empty() {
        100.0
    } else {
        node_positions
            .values()
            .map(|(_, y)| *y)
            .fold(f64::NEG_INFINITY, f64::max)
            + NODE_H
            + MARGIN
    };

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">"#,
        w = total_w,
        h = total_h
    ));
    svg.push('\n');

    // Styles embedded inline for portability
    svg.push_str(
        r##"<defs><style>
  .node-rect{fill:#4a90d9;stroke:#2c5f8a;stroke-width:1.5;rx:6;ry:6;}
  .node-label{fill:#fff;font-family:sans-serif;font-size:12px;text-anchor:middle;dominant-baseline:middle;}
  .edge-line{stroke:#888;stroke-width:1.5;marker-end:url(#arrow);}
</style>
<marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
  <path d="M0,0 L0,6 L8,3 z" fill="#888"/>
</marker></defs>
"##,
    );

    // Draw edges first (behind nodes)
    for edge in pipeline.edges.values() {
        if let (Some(&(sx, sy)), Some(&(tx, ty))) = (
            node_positions.get(edge.source_node_id.as_str()),
            node_positions.get(edge.target_node_id.as_str()),
        ) {
            let x1 = sx + NODE_W;
            let y1 = sy + NODE_H / 2.0;
            let x2 = tx;
            let y2 = ty + NODE_H / 2.0;
            svg.push_str(&format!(
                r#"<line class="edge-line" x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}"/>"#
            ));
            svg.push('\n');
        }
    }

    // Draw nodes
    for (id, node) in &nodes {
        if let Some(&(x, y)) = node_positions.get(id.as_str()) {
            let label = xml_escape(&node.name);
            svg.push_str(&format!(
                r#"<rect class="node-rect" x="{x:.1}" y="{y:.1}" width="{w}" height="{h}" rx="6" ry="6"/>"#,
                w = NODE_W,
                h = NODE_H
            ));
            svg.push('\n');
            svg.push_str(&format!(
                r#"<text class="node-label" x="{cx:.1}" y="{cy:.1}">{label}</text>"#,
                cx = x + NODE_W / 2.0,
                cy = y + NODE_H / 2.0,
            ));
            svg.push('\n');
        }
    }

    // Pipeline name as title
    let title = xml_escape(&pipeline.name);
    svg.push_str(&format!(
        r##"<text x="{x}" y="16" font-family="sans-serif" font-size="14" fill="#333">{title}</text>"##,
        x = MARGIN
    ));
    svg.push('\n');

    svg.push_str("</svg>\n");
    Ok(svg)
}

/// Export pipeline to DOT format (GraphViz)
pub fn export_dot(pipeline: &VisualPipeline) -> Result<String> {
    let mut dot = String::new();
    dot.push_str(&format!("digraph \"{}\" {{\n", pipeline.name));
    dot.push_str("  rankdir=LR;\n");
    dot.push_str("  node [shape=box];\n\n");

    // Add nodes
    for (id, node) in &pipeline.nodes {
        let label = format!("{}\\n{:?}", node.name, node.node_type);
        dot.push_str(&format!("  \"{}\" [label=\"{}\"];\n", id, label));
    }

    dot.push('\n');

    // Add edges
    for edge in pipeline.edges.values() {
        dot.push_str(&format!(
            "  \"{}\" -> \"{}\";\n",
            edge.source_node_id, edge.target_node_id
        ));
    }

    dot.push_str("}\n");
    Ok(dot)
}

/// Export pipeline to Mermaid format
pub fn export_mermaid(pipeline: &VisualPipeline) -> Result<String> {
    let mut mermaid = String::new();
    mermaid.push_str("graph LR\n");

    // Add nodes
    for (id, node) in &pipeline.nodes {
        let label = format!("{}[{}]", id, node.name);
        mermaid.push_str(&format!("  {}\n", label));
    }

    // Add edges
    for edge in pipeline.edges.values() {
        mermaid.push_str(&format!(
            "  {} --> {}\n",
            edge.source_node_id, edge.target_node_id
        ));
    }

    Ok(mermaid)
}

/// Escape special XML/HTML characters for safe embedding in SVG text elements.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}
