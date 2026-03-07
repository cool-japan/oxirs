//! # RDF Graph Visualization
//!
//! Produces layout data for rendering an RDF graph in a 2-D canvas.  The
//! primary layout algorithm is Fruchterman-Reingold force-directed placement.
//! All state is kept in plain data structures so the module can run in a
//! WASM environment without access to a real DOM.
//!
//! ## Features
//!
//! - Fruchterman-Reingold force-directed layout simulation
//! - Node data generation (position, label, type/color mapping)
//! - Edge data generation (source, target, label, curvature)
//! - Subgraph extraction (BFS/DFS from a focal node)
//! - Viewport state management (zoom, pan)
//! - Node clustering by namespace or RDF type
//! - Layout configuration (spring constant, repulsion, damping, iterations)
//! - Graph statistics (degree distribution, connected components)
//!
//! ## Example
//!
//! ```rust
//! use oxirs_wasm::graph_visualizer::{
//!     GraphData, LayoutConfig, GraphVisualizer, NodeData, EdgeData,
//! };
//!
//! let mut graph = GraphData::new();
//! graph.add_triple("http://example.org/alice", "http://xmlns.com/foaf/0.1/knows", "http://example.org/bob");
//! graph.add_triple("http://example.org/bob", "http://xmlns.com/foaf/0.1/name", "\"Bob\"");
//!
//! let config = LayoutConfig::default();
//! let mut viz = GraphVisualizer::new(config);
//! viz.compute_layout(&graph);
//! let nodes = viz.node_data();
//! let edges = viz.edge_data();
//! assert!(nodes.len() >= 2);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ─── Configuration ───────────────────────────────────────────────────────────

/// Parameters for the Fruchterman-Reingold layout algorithm.
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// Width of the layout area in logical pixels.
    pub width: f64,
    /// Height of the layout area in logical pixels.
    pub height: f64,
    /// Attractive spring constant (controls edge spring strength).
    pub spring_constant: f64,
    /// Repulsive constant (controls node repulsion strength).
    pub repulsion: f64,
    /// Velocity damping factor applied each iteration (0..1).
    pub damping: f64,
    /// Number of simulation iterations.
    pub iterations: usize,
    /// Minimum distance between nodes to avoid division by zero.
    pub min_distance: f64,
    /// Initial temperature for simulated annealing.
    pub initial_temperature: f64,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            spring_constant: 2.0,
            repulsion: 5000.0,
            damping: 0.85,
            iterations: 100,
            min_distance: 1.0,
            initial_temperature: 100.0,
        }
    }
}

// ─── Graph data model ────────────────────────────────────────────────────────

/// A single RDF triple stored in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RdfTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// An RDF graph stored as a set of triples with fast node lookup.
#[derive(Debug, Clone, Default)]
pub struct GraphData {
    triples: Vec<RdfTriple>,
    nodes: HashSet<String>,
}

impl GraphData {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple to the graph.
    pub fn add_triple(
        &mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) {
        let s = subject.into();
        let p = predicate.into();
        let o = object.into();
        self.nodes.insert(s.clone());
        self.nodes.insert(o.clone());
        self.triples.push(RdfTriple {
            subject: s,
            predicate: p,
            object: o,
        });
    }

    /// Number of triples.
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Number of distinct nodes (subjects + objects).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Iterator over triples.
    pub fn triples(&self) -> &[RdfTriple] {
        &self.triples
    }

    /// Iterator over node identifiers.
    pub fn nodes(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(String::as_str)
    }

    /// Whether a node exists.
    pub fn contains_node(&self, id: &str) -> bool {
        self.nodes.contains(id)
    }
}

// ─── Subgraph extraction ─────────────────────────────────────────────────────

/// Extraction mode for subgraph operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalMode {
    /// Breadth-first search from the focal node.
    Bfs,
    /// Depth-first search from the focal node.
    Dfs,
}

/// Extract a subgraph of at most `max_depth` hops from `focal_node`.
pub fn extract_subgraph(
    graph: &GraphData,
    focal_node: &str,
    max_depth: usize,
    mode: TraversalMode,
) -> GraphData {
    if !graph.contains_node(focal_node) {
        return GraphData::new();
    }

    // Build adjacency list (undirected — both directions)
    let mut adj: HashMap<&str, Vec<(&str, &str, &str)>> = HashMap::new();
    for t in &graph.triples {
        adj.entry(t.subject.as_str()).or_default().push((
            t.subject.as_str(),
            t.predicate.as_str(),
            t.object.as_str(),
        ));
        adj.entry(t.object.as_str()).or_default().push((
            t.subject.as_str(),
            t.predicate.as_str(),
            t.object.as_str(),
        ));
    }

    let mut visited: HashSet<&str> = HashSet::new();
    let mut collected_triples: HashSet<(&str, &str, &str)> = HashSet::new();

    match mode {
        TraversalMode::Bfs => {
            let mut queue: VecDeque<(&str, usize)> = VecDeque::new();
            queue.push_back((focal_node, 0));
            visited.insert(focal_node);
            while let Some((node, depth)) = queue.pop_front() {
                if depth >= max_depth {
                    continue;
                }
                if let Some(edges) = adj.get(node) {
                    for &(s, p, o) in edges {
                        collected_triples.insert((s, p, o));
                        let neighbour = if s == node { o } else { s };
                        if visited.insert(neighbour) {
                            queue.push_back((neighbour, depth + 1));
                        }
                    }
                }
            }
        }
        TraversalMode::Dfs => {
            let mut stack: Vec<(&str, usize)> = vec![(focal_node, 0)];
            visited.insert(focal_node);
            while let Some((node, depth)) = stack.pop() {
                if depth >= max_depth {
                    continue;
                }
                if let Some(edges) = adj.get(node) {
                    for &(s, p, o) in edges {
                        collected_triples.insert((s, p, o));
                        let neighbour = if s == node { o } else { s };
                        if visited.insert(neighbour) {
                            stack.push((neighbour, depth + 1));
                        }
                    }
                }
            }
        }
    }

    let mut sub = GraphData::new();
    for (s, p, o) in collected_triples {
        sub.add_triple(s, p, o);
    }
    sub
}

// ─── Node / edge data for rendering ──────────────────────────────────────────

/// Rendered node data ready for the front-end canvas.
#[derive(Debug, Clone)]
pub struct NodeData {
    /// Node identifier (IRI or literal).
    pub id: String,
    /// Display label (local name or short literal).
    pub label: String,
    /// X position after layout.
    pub x: f64,
    /// Y position after layout.
    pub y: f64,
    /// Node type hint ("iri", "literal", "blank").
    pub node_type: String,
    /// Suggested CSS/SVG colour.
    pub color: String,
    /// Namespace extracted from IRI (empty for non-IRI nodes).
    pub namespace: String,
}

/// Rendered edge data ready for the front-end canvas.
#[derive(Debug, Clone)]
pub struct EdgeData {
    /// Source node ID.
    pub source: String,
    /// Target node ID.
    pub target: String,
    /// Display label (predicate local name).
    pub label: String,
    /// Curvature hint for multi-edges (0.0 = straight).
    pub curvature: f64,
}

// ─── Viewport state ──────────────────────────────────────────────────────────

/// Zoom/pan viewport state for the visualization canvas.
#[derive(Debug, Clone)]
pub struct Viewport {
    /// Current zoom level (1.0 = 100%).
    pub zoom: f64,
    /// Horizontal pan offset in logical pixels.
    pub pan_x: f64,
    /// Vertical pan offset in logical pixels.
    pub pan_y: f64,
    /// Minimum allowed zoom level.
    pub min_zoom: f64,
    /// Maximum allowed zoom level.
    pub max_zoom: f64,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            min_zoom: 0.1,
            max_zoom: 10.0,
        }
    }
}

impl Viewport {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a zoom delta, clamped to [min_zoom, max_zoom].
    pub fn apply_zoom(&mut self, delta: f64) {
        let new_zoom = (self.zoom + delta).clamp(self.min_zoom, self.max_zoom);
        self.zoom = new_zoom;
    }

    /// Apply a pan delta.
    pub fn apply_pan(&mut self, dx: f64, dy: f64) {
        self.pan_x += dx;
        self.pan_y += dy;
    }

    /// Reset to defaults.
    pub fn reset(&mut self) {
        self.zoom = 1.0;
        self.pan_x = 0.0;
        self.pan_y = 0.0;
    }

    /// Transform a layout coordinate to screen coordinate.
    pub fn to_screen(&self, x: f64, y: f64) -> (f64, f64) {
        (x * self.zoom + self.pan_x, y * self.zoom + self.pan_y)
    }
}

// ─── Graph statistics ────────────────────────────────────────────────────────

/// Statistics about the graph topology.
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Number of nodes.
    pub node_count: usize,
    /// Number of edges (triples).
    pub edge_count: usize,
    /// Degree distribution: degree -> count.
    pub degree_distribution: HashMap<usize, usize>,
    /// Number of connected components.
    pub connected_components: usize,
    /// Maximum degree.
    pub max_degree: usize,
    /// Average degree.
    pub avg_degree: f64,
}

/// Compute graph statistics.
pub fn compute_stats(graph: &GraphData) -> GraphStats {
    let mut degree_map: HashMap<&str, usize> = HashMap::new();
    for t in &graph.triples {
        *degree_map.entry(t.subject.as_str()).or_default() += 1;
        *degree_map.entry(t.object.as_str()).or_default() += 1;
    }

    // Include isolated nodes with degree 0
    for node in graph.nodes() {
        degree_map.entry(node).or_default();
    }

    let mut degree_distribution: HashMap<usize, usize> = HashMap::new();
    let mut max_degree: usize = 0;
    let mut total_degree: usize = 0;

    for &deg in degree_map.values() {
        *degree_distribution.entry(deg).or_default() += 1;
        if deg > max_degree {
            max_degree = deg;
        }
        total_degree += deg;
    }

    let node_count = graph.node_count();
    let avg_degree = if node_count > 0 {
        total_degree as f64 / node_count as f64
    } else {
        0.0
    };

    // Connected components via BFS
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for t in &graph.triples {
        adj.entry(t.subject.as_str())
            .or_default()
            .push(t.object.as_str());
        adj.entry(t.object.as_str())
            .or_default()
            .push(t.subject.as_str());
    }

    let mut visited: HashSet<&str> = HashSet::new();
    let mut components = 0_usize;
    for node in graph.nodes() {
        if visited.insert(node) {
            components += 1;
            let mut queue = VecDeque::new();
            queue.push_back(node);
            while let Some(n) = queue.pop_front() {
                if let Some(neighbours) = adj.get(n) {
                    for &nb in neighbours {
                        if visited.insert(nb) {
                            queue.push_back(nb);
                        }
                    }
                }
            }
        }
    }

    GraphStats {
        node_count,
        edge_count: graph.triple_count(),
        degree_distribution,
        connected_components: components,
        max_degree,
        avg_degree,
    }
}

// ─── Node clustering ─────────────────────────────────────────────────────────

/// Cluster nodes by IRI namespace prefix.
pub fn cluster_by_namespace(graph: &GraphData) -> HashMap<String, Vec<String>> {
    let mut clusters: HashMap<String, Vec<String>> = HashMap::new();
    for node in graph.nodes() {
        let ns = extract_namespace(node);
        clusters.entry(ns).or_default().push(node.to_string());
    }
    // Sort within each cluster for deterministic output
    for members in clusters.values_mut() {
        members.sort();
    }
    clusters
}

/// Cluster nodes by rdf:type (heuristic: look for triples with predicate
/// containing "type" or "rdf:type").
pub fn cluster_by_type(graph: &GraphData) -> HashMap<String, Vec<String>> {
    let mut type_map: HashMap<String, Vec<String>> = HashMap::new();
    let type_predicates: HashSet<&str> = [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "rdf:type",
        "a",
    ]
    .into_iter()
    .collect();

    for t in &graph.triples {
        if type_predicates.contains(t.predicate.as_str()) {
            type_map
                .entry(t.object.clone())
                .or_default()
                .push(t.subject.clone());
        }
    }

    // Nodes without an explicit type go into "untyped"
    let typed_nodes: HashSet<String> = type_map.values().flatten().cloned().collect();
    let mut untyped = Vec::new();
    for node in graph.nodes() {
        if !typed_nodes.contains(node) {
            untyped.push(node.to_string());
        }
    }
    if !untyped.is_empty() {
        untyped.sort();
        type_map.insert("(untyped)".to_string(), untyped);
    }

    for members in type_map.values_mut() {
        members.sort();
    }
    type_map
}

// ─── Helper functions ────────────────────────────────────────────────────────

/// Extract the namespace from an IRI (everything up to and including the last
/// `#` or `/`).  Non-IRI strings return an empty string.
fn extract_namespace(iri: &str) -> String {
    if iri.starts_with('"') || iri.starts_with("_:") {
        return String::new();
    }
    if let Some(pos) = iri.rfind('#') {
        return iri[..=pos].to_string();
    }
    if let Some(pos) = iri.rfind('/') {
        return iri[..=pos].to_string();
    }
    String::new()
}

/// Extract the local name from an IRI (everything after the last `#` or `/`).
fn extract_local_name(iri: &str) -> String {
    if let Some(pos) = iri.rfind('#') {
        return iri[pos + 1..].to_string();
    }
    if let Some(pos) = iri.rfind('/') {
        return iri[pos + 1..].to_string();
    }
    iri.to_string()
}

/// Classify a term as "iri", "literal", or "blank".
fn classify_node(id: &str) -> &'static str {
    if id.starts_with('"') {
        "literal"
    } else if id.starts_with("_:") {
        "blank"
    } else {
        "iri"
    }
}

/// Map a node type to a default colour.
fn color_for_type(node_type: &str) -> &'static str {
    match node_type {
        "iri" => "#4A90D9",
        "literal" => "#50C878",
        "blank" => "#B0B0B0",
        _ => "#888888",
    }
}

// ─── Graph visualizer (Fruchterman-Reingold) ─────────────────────────────────

/// The main visualizer. After calling `compute_layout` the caller can retrieve
/// `NodeData` and `EdgeData` for rendering.
pub struct GraphVisualizer {
    config: LayoutConfig,
    positions: HashMap<String, (f64, f64)>,
    viewport: Viewport,
}

impl GraphVisualizer {
    /// Create a new visualizer with the given layout config.
    pub fn new(config: LayoutConfig) -> Self {
        Self {
            config,
            positions: HashMap::new(),
            viewport: Viewport::new(),
        }
    }

    /// Access the viewport.
    pub fn viewport(&self) -> &Viewport {
        &self.viewport
    }

    /// Mutable access to the viewport.
    pub fn viewport_mut(&mut self) -> &mut Viewport {
        &mut self.viewport
    }

    /// Run the Fruchterman-Reingold layout algorithm.
    pub fn compute_layout(&mut self, graph: &GraphData) {
        let nodes: Vec<String> = {
            let mut v: Vec<String> = graph.nodes().map(String::from).collect();
            v.sort();
            v
        };
        let n = nodes.len();
        if n == 0 {
            self.positions.clear();
            return;
        }

        let node_idx: HashMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();

        // Initial positions: deterministic grid
        let cols = (n as f64).sqrt().ceil() as usize;
        let spacing_x = self.config.width / (cols as f64 + 1.0);
        let rows = (n + cols - 1) / cols;
        let spacing_y = self.config.height / (rows as f64 + 1.0);

        let mut xs = vec![0.0_f64; n];
        let mut ys = vec![0.0_f64; n];
        for (i, _node) in nodes.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;
            xs[i] = spacing_x * (col as f64 + 1.0);
            ys[i] = spacing_y * (row as f64 + 1.0);
        }

        // Ideal distance
        let area = self.config.width * self.config.height;
        let k = (area / n.max(1) as f64).sqrt();

        // Build edge index pairs
        let edges: Vec<(usize, usize)> = graph
            .triples()
            .iter()
            .filter_map(|t| {
                let si = node_idx.get(t.subject.as_str())?;
                let oi = node_idx.get(t.object.as_str())?;
                Some((*si, *oi))
            })
            .collect();

        let mut temp = self.config.initial_temperature;
        let cool = temp / self.config.iterations.max(1) as f64;

        for _iter in 0..self.config.iterations {
            let mut dx = vec![0.0_f64; n];
            let mut dy = vec![0.0_f64; n];

            // Repulsive forces
            for i in 0..n {
                for j in (i + 1)..n {
                    let diff_x = xs[i] - xs[j];
                    let diff_y = ys[i] - ys[j];
                    let dist = (diff_x * diff_x + diff_y * diff_y)
                        .sqrt()
                        .max(self.config.min_distance);
                    let force = self.config.repulsion / (dist * dist);
                    let fx = (diff_x / dist) * force;
                    let fy = (diff_y / dist) * force;
                    dx[i] += fx;
                    dy[i] += fy;
                    dx[j] -= fx;
                    dy[j] -= fy;
                }
            }

            // Attractive forces along edges
            for &(si, oi) in &edges {
                let diff_x = xs[si] - xs[oi];
                let diff_y = ys[si] - ys[oi];
                let dist = (diff_x * diff_x + diff_y * diff_y)
                    .sqrt()
                    .max(self.config.min_distance);
                let force = self.config.spring_constant * (dist - k) / dist;
                let fx = diff_x * force;
                let fy = diff_y * force;
                dx[si] -= fx;
                dy[si] -= fy;
                dx[oi] += fx;
                dy[oi] += fy;
            }

            // Apply displacements with temperature limit and damping
            for i in 0..n {
                let disp = (dx[i] * dx[i] + dy[i] * dy[i])
                    .sqrt()
                    .max(self.config.min_distance);
                let capped = disp.min(temp);
                xs[i] += (dx[i] / disp) * capped * self.config.damping;
                ys[i] += (dy[i] / disp) * capped * self.config.damping;

                // Clamp to canvas bounds
                xs[i] = xs[i].clamp(10.0, self.config.width - 10.0);
                ys[i] = ys[i].clamp(10.0, self.config.height - 10.0);
            }

            temp -= cool;
            if temp < 0.0 {
                temp = 0.0;
            }
        }

        // Store results
        self.positions.clear();
        for (i, node) in nodes.iter().enumerate() {
            self.positions.insert(node.clone(), (xs[i], ys[i]));
        }
    }

    /// Generate `NodeData` for all laid-out nodes.
    pub fn node_data(&self) -> Vec<NodeData> {
        let mut result: Vec<NodeData> = self
            .positions
            .iter()
            .map(|(id, &(x, y))| {
                let nt = classify_node(id);
                NodeData {
                    id: id.clone(),
                    label: extract_local_name(id),
                    x,
                    y,
                    node_type: nt.to_string(),
                    color: color_for_type(nt).to_string(),
                    namespace: extract_namespace(id),
                }
            })
            .collect();
        result.sort_by(|a, b| a.id.cmp(&b.id));
        result
    }

    /// Generate `EdgeData` for all triples whose endpoints have been laid out.
    pub fn edge_data_from(&self, graph: &GraphData) -> Vec<EdgeData> {
        // Count parallel edges for curvature
        let mut edge_counts: HashMap<(String, String), usize> = HashMap::new();
        for t in graph.triples() {
            *edge_counts
                .entry((t.subject.clone(), t.object.clone()))
                .or_default() += 1;
        }

        let mut edge_index: HashMap<(String, String), usize> = HashMap::new();
        let mut result = Vec::new();

        for t in graph.triples() {
            if !self.positions.contains_key(&t.subject) || !self.positions.contains_key(&t.object) {
                continue;
            }

            let key = (t.subject.clone(), t.object.clone());
            let idx = edge_index.entry(key.clone()).or_default();
            let total = edge_counts.get(&key).copied().unwrap_or(1);
            let curvature = if total <= 1 {
                0.0
            } else {
                (*idx as f64 - (total as f64 - 1.0) / 2.0) * 0.2
            };
            *idx += 1;

            result.push(EdgeData {
                source: t.subject.clone(),
                target: t.object.clone(),
                label: extract_local_name(&t.predicate),
                curvature,
            });
        }

        result
    }

    /// Convenience: generate edge data using an internal reference.
    ///
    /// If you have already laid out `graph`, pass it again to obtain edges.
    pub fn edge_data(&self) -> Vec<EdgeData> {
        // Without a stored reference we return empty; caller should use edge_data_from.
        Vec::new()
    }

    /// Get the position of a specific node (if laid out).
    pub fn position_of(&self, id: &str) -> Option<(f64, f64)> {
        self.positions.get(id).copied()
    }

    /// Total number of positioned nodes.
    pub fn positioned_node_count(&self) -> usize {
        self.positions.len()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> GraphData {
        let mut g = GraphData::new();
        g.add_triple(
            "http://example.org/alice",
            "http://xmlns.com/foaf/0.1/knows",
            "http://example.org/bob",
        );
        g.add_triple(
            "http://example.org/bob",
            "http://xmlns.com/foaf/0.1/knows",
            "http://example.org/carol",
        );
        g.add_triple(
            "http://example.org/alice",
            "http://xmlns.com/foaf/0.1/name",
            "\"Alice\"",
        );
        g.add_triple(
            "http://example.org/bob",
            "http://xmlns.com/foaf/0.1/name",
            "\"Bob\"",
        );
        g
    }

    // ── GraphData tests ─────────────────────────────────────────────────

    #[test]
    fn test_graph_data_empty() {
        let g = GraphData::new();
        assert_eq!(g.triple_count(), 0);
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_graph_data_add_triple() {
        let mut g = GraphData::new();
        g.add_triple("s", "p", "o");
        assert_eq!(g.triple_count(), 1);
        assert_eq!(g.node_count(), 2); // s and o
    }

    #[test]
    fn test_graph_data_node_dedup() {
        let mut g = GraphData::new();
        g.add_triple("a", "p", "b");
        g.add_triple("b", "q", "c");
        assert_eq!(g.node_count(), 3); // a, b, c
    }

    #[test]
    fn test_graph_data_contains_node() {
        let g = sample_graph();
        assert!(g.contains_node("http://example.org/alice"));
        assert!(g.contains_node("\"Bob\""));
        assert!(!g.contains_node("nonexistent"));
    }

    #[test]
    fn test_graph_data_triples_accessor() {
        let g = sample_graph();
        assert_eq!(g.triples().len(), 4);
    }

    // ── LayoutConfig tests ──────────────────────────────────────────────

    #[test]
    fn test_layout_config_defaults() {
        let cfg = LayoutConfig::default();
        assert!(cfg.width > 0.0);
        assert!(cfg.height > 0.0);
        assert!(cfg.spring_constant > 0.0);
        assert!(cfg.repulsion > 0.0);
        assert!(cfg.damping > 0.0 && cfg.damping <= 1.0);
        assert!(cfg.iterations > 0);
    }

    // ── Fruchterman-Reingold layout tests ───────────────────────────────

    #[test]
    fn test_layout_empty_graph() {
        let mut viz = GraphVisualizer::new(LayoutConfig::default());
        viz.compute_layout(&GraphData::new());
        assert_eq!(viz.positioned_node_count(), 0);
        assert!(viz.node_data().is_empty());
    }

    #[test]
    fn test_layout_single_edge() {
        let mut g = GraphData::new();
        g.add_triple("a", "p", "b");
        let mut viz = GraphVisualizer::new(LayoutConfig::default());
        viz.compute_layout(&g);
        assert_eq!(viz.positioned_node_count(), 2);
        let pos_a = viz.position_of("a");
        let pos_b = viz.position_of("b");
        assert!(pos_a.is_some());
        assert!(pos_b.is_some());
    }

    #[test]
    fn test_layout_nodes_within_bounds() {
        let g = sample_graph();
        let cfg = LayoutConfig {
            width: 500.0,
            height: 400.0,
            ..Default::default()
        };
        let mut viz = GraphVisualizer::new(cfg.clone());
        viz.compute_layout(&g);
        for nd in viz.node_data() {
            assert!(
                nd.x >= 0.0 && nd.x <= cfg.width,
                "x out of bounds: {}",
                nd.x
            );
            assert!(
                nd.y >= 0.0 && nd.y <= cfg.height,
                "y out of bounds: {}",
                nd.y
            );
        }
    }

    #[test]
    fn test_layout_all_nodes_positioned() {
        let g = sample_graph();
        let mut viz = GraphVisualizer::new(LayoutConfig::default());
        viz.compute_layout(&g);
        assert_eq!(viz.positioned_node_count(), g.node_count());
    }

    #[test]
    fn test_layout_different_iterations() {
        let g = sample_graph();
        let cfg1 = LayoutConfig {
            iterations: 10,
            ..Default::default()
        };
        let cfg2 = LayoutConfig {
            iterations: 200,
            ..Default::default()
        };
        let mut viz1 = GraphVisualizer::new(cfg1);
        let mut viz2 = GraphVisualizer::new(cfg2);
        viz1.compute_layout(&g);
        viz2.compute_layout(&g);
        // Both should position all nodes
        assert_eq!(viz1.positioned_node_count(), g.node_count());
        assert_eq!(viz2.positioned_node_count(), g.node_count());
    }

    // ── NodeData tests ──────────────────────────────────────────────────

    #[test]
    fn test_node_data_iri() {
        let mut g = GraphData::new();
        g.add_triple("http://example.org/x", "p", "http://example.org/y");
        let mut viz = GraphVisualizer::new(LayoutConfig::default());
        viz.compute_layout(&g);
        let nodes = viz.node_data();
        for nd in &nodes {
            if nd.id == "http://example.org/x" {
                assert_eq!(nd.node_type, "iri");
                assert_eq!(nd.label, "x");
                assert_eq!(nd.color, "#4A90D9");
                assert_eq!(nd.namespace, "http://example.org/");
            }
        }
    }

    #[test]
    fn test_node_data_literal() {
        let mut g = GraphData::new();
        g.add_triple("http://example.org/x", "p", "\"hello\"");
        let mut viz = GraphVisualizer::new(LayoutConfig::default());
        viz.compute_layout(&g);
        let nodes = viz.node_data();
        let lit = nodes.iter().find(|n| n.id == "\"hello\"");
        assert!(lit.is_some());
        let lit = lit.expect("literal node should exist");
        assert_eq!(lit.node_type, "literal");
        assert_eq!(lit.color, "#50C878");
    }

    #[test]
    fn test_node_data_blank() {
        let mut g = GraphData::new();
        g.add_triple("_:b0", "p", "_:b1");
        let mut viz = GraphVisualizer::new(LayoutConfig::default());
        viz.compute_layout(&g);
        let nodes = viz.node_data();
        for nd in &nodes {
            assert_eq!(nd.node_type, "blank");
            assert_eq!(nd.color, "#B0B0B0");
        }
    }

    #[test]
    fn test_node_data_sorted() {
        let g = sample_graph();
        let mut viz = GraphVisualizer::new(LayoutConfig::default());
        viz.compute_layout(&g);
        let nodes = viz.node_data();
        for i in 1..nodes.len() {
            assert!(nodes[i - 1].id <= nodes[i].id);
        }
    }

    // ── EdgeData tests ──────────────────────────────────────────────────

    #[test]
    fn test_edge_data_from() {
        let g = sample_graph();
        let mut viz = GraphVisualizer::new(LayoutConfig::default());
        viz.compute_layout(&g);
        let edges = viz.edge_data_from(&g);
        assert_eq!(edges.len(), g.triple_count());
    }

    #[test]
    fn test_edge_data_labels() {
        let mut g = GraphData::new();
        g.add_triple("s", "http://example.org/predicate#rel", "o");
        let mut viz = GraphVisualizer::new(LayoutConfig::default());
        viz.compute_layout(&g);
        let edges = viz.edge_data_from(&g);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].label, "rel");
    }

    #[test]
    fn test_edge_curvature_single_edge() {
        let mut g = GraphData::new();
        g.add_triple("a", "p", "b");
        let mut viz = GraphVisualizer::new(LayoutConfig::default());
        viz.compute_layout(&g);
        let edges = viz.edge_data_from(&g);
        assert_eq!(edges.len(), 1);
        assert!((edges[0].curvature - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edge_curvature_parallel_edges() {
        let mut g = GraphData::new();
        g.add_triple("a", "p1", "b");
        g.add_triple("a", "p2", "b");
        let mut viz = GraphVisualizer::new(LayoutConfig::default());
        viz.compute_layout(&g);
        let edges = viz.edge_data_from(&g);
        assert_eq!(edges.len(), 2);
        // Parallel edges should get different curvatures
        assert!((edges[0].curvature - edges[1].curvature).abs() > 0.01);
    }

    // ── Subgraph extraction tests ───────────────────────────────────────

    #[test]
    fn test_subgraph_bfs_depth1() {
        let g = sample_graph();
        let sub = extract_subgraph(&g, "http://example.org/alice", 1, TraversalMode::Bfs);
        // Should include alice's direct neighbours
        assert!(sub.contains_node("http://example.org/alice"));
        assert!(sub.contains_node("http://example.org/bob"));
        assert!(sub.triple_count() > 0);
    }

    #[test]
    fn test_subgraph_dfs_depth1() {
        let g = sample_graph();
        let sub = extract_subgraph(&g, "http://example.org/alice", 1, TraversalMode::Dfs);
        assert!(sub.contains_node("http://example.org/alice"));
        assert!(sub.triple_count() > 0);
    }

    #[test]
    fn test_subgraph_depth0() {
        let g = sample_graph();
        let sub = extract_subgraph(&g, "http://example.org/alice", 0, TraversalMode::Bfs);
        // depth=0 means no expansion: only the focal node, no triples
        assert_eq!(sub.triple_count(), 0);
    }

    #[test]
    fn test_subgraph_nonexistent_focal() {
        let g = sample_graph();
        let sub = extract_subgraph(&g, "nonexistent", 5, TraversalMode::Bfs);
        assert_eq!(sub.triple_count(), 0);
        assert_eq!(sub.node_count(), 0);
    }

    #[test]
    fn test_subgraph_full_depth() {
        let g = sample_graph();
        let sub = extract_subgraph(&g, "http://example.org/alice", 100, TraversalMode::Bfs);
        // With enough depth, should include the whole graph
        assert_eq!(sub.triple_count(), g.triple_count());
    }

    // ── Viewport tests ──────────────────────────────────────────────────

    #[test]
    fn test_viewport_default() {
        let vp = Viewport::new();
        assert!((vp.zoom - 1.0).abs() < f64::EPSILON);
        assert!((vp.pan_x - 0.0).abs() < f64::EPSILON);
        assert!((vp.pan_y - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_viewport_zoom_in() {
        let mut vp = Viewport::new();
        vp.apply_zoom(0.5);
        assert!((vp.zoom - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_viewport_zoom_clamp_max() {
        let mut vp = Viewport {
            max_zoom: 2.0,
            ..Default::default()
        };
        vp.apply_zoom(5.0);
        assert!((vp.zoom - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_viewport_zoom_clamp_min() {
        let mut vp = Viewport {
            min_zoom: 0.5,
            ..Default::default()
        };
        vp.apply_zoom(-2.0);
        assert!((vp.zoom - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_viewport_pan() {
        let mut vp = Viewport::new();
        vp.apply_pan(10.0, -20.0);
        assert!((vp.pan_x - 10.0).abs() < f64::EPSILON);
        assert!((vp.pan_y - (-20.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_viewport_reset() {
        let mut vp = Viewport::new();
        vp.apply_zoom(2.0);
        vp.apply_pan(100.0, 200.0);
        vp.reset();
        assert!((vp.zoom - 1.0).abs() < f64::EPSILON);
        assert!((vp.pan_x - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_viewport_to_screen() {
        let vp = Viewport {
            zoom: 2.0,
            pan_x: 10.0,
            pan_y: 20.0,
            ..Default::default()
        };
        let (sx, sy) = vp.to_screen(5.0, 3.0);
        assert!((sx - 20.0).abs() < f64::EPSILON); // 5*2+10
        assert!((sy - 26.0).abs() < f64::EPSILON); // 3*2+20
    }

    // ── Graph statistics tests ──────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let g = GraphData::new();
        let s = compute_stats(&g);
        assert_eq!(s.node_count, 0);
        assert_eq!(s.edge_count, 0);
        assert_eq!(s.connected_components, 0);
        assert_eq!(s.max_degree, 0);
        assert!((s.avg_degree - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_single_triple() {
        let mut g = GraphData::new();
        g.add_triple("a", "p", "b");
        let s = compute_stats(&g);
        assert_eq!(s.node_count, 2);
        assert_eq!(s.edge_count, 1);
        assert_eq!(s.connected_components, 1);
        assert_eq!(s.max_degree, 1);
    }

    #[test]
    fn test_stats_degree_distribution() {
        let mut g = GraphData::new();
        g.add_triple("a", "p1", "b");
        g.add_triple("a", "p2", "c");
        let s = compute_stats(&g);
        // a has degree 2, b has degree 1, c has degree 1
        assert_eq!(s.degree_distribution.get(&2), Some(&1));
        assert_eq!(s.degree_distribution.get(&1), Some(&2));
    }

    #[test]
    fn test_stats_connected_components() {
        let mut g = GraphData::new();
        g.add_triple("a", "p", "b");
        g.add_triple("c", "q", "d");
        let s = compute_stats(&g);
        assert_eq!(s.connected_components, 2);
    }

    #[test]
    fn test_stats_avg_degree() {
        let g = sample_graph();
        let s = compute_stats(&g);
        assert!(s.avg_degree > 0.0);
    }

    // ── Namespace clustering tests ──────────────────────────────────────

    #[test]
    fn test_cluster_by_namespace() {
        let g = sample_graph();
        let clusters = cluster_by_namespace(&g);
        assert!(clusters.contains_key("http://example.org/"));
        // Literals have empty namespace
        assert!(clusters.contains_key(""));
    }

    #[test]
    fn test_cluster_by_namespace_hash_fragment() {
        let mut g = GraphData::new();
        g.add_triple(
            "http://example.org/ns#Alice",
            "p",
            "http://example.org/ns#Bob",
        );
        let clusters = cluster_by_namespace(&g);
        assert!(clusters.contains_key("http://example.org/ns#"));
    }

    // ── Type clustering tests ───────────────────────────────────────────

    #[test]
    fn test_cluster_by_type_no_types() {
        let g = sample_graph();
        let clusters = cluster_by_type(&g);
        // No rdf:type triples -> everything is "(untyped)"
        assert!(clusters.contains_key("(untyped)"));
    }

    #[test]
    fn test_cluster_by_type_with_types() {
        let mut g = GraphData::new();
        g.add_triple(
            "http://example.org/alice",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://xmlns.com/foaf/0.1/Person",
        );
        g.add_triple(
            "http://example.org/bob",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://xmlns.com/foaf/0.1/Person",
        );
        let clusters = cluster_by_type(&g);
        let persons = clusters
            .get("http://xmlns.com/foaf/0.1/Person")
            .expect("should have Person cluster");
        assert_eq!(persons.len(), 2);
    }

    // ── Helper function tests ───────────────────────────────────────────

    #[test]
    fn test_extract_namespace_slash() {
        assert_eq!(
            extract_namespace("http://example.org/alice"),
            "http://example.org/"
        );
    }

    #[test]
    fn test_extract_namespace_hash() {
        assert_eq!(
            extract_namespace("http://example.org/ns#Alice"),
            "http://example.org/ns#"
        );
    }

    #[test]
    fn test_extract_namespace_literal() {
        assert_eq!(extract_namespace("\"hello\""), "");
    }

    #[test]
    fn test_extract_namespace_blank() {
        assert_eq!(extract_namespace("_:b0"), "");
    }

    #[test]
    fn test_extract_local_name() {
        assert_eq!(extract_local_name("http://example.org/alice"), "alice");
        assert_eq!(extract_local_name("http://example.org/ns#Bob"), "Bob");
        assert_eq!(extract_local_name("simple"), "simple");
    }

    #[test]
    fn test_classify_node_iri() {
        assert_eq!(classify_node("http://example.org/x"), "iri");
    }

    #[test]
    fn test_classify_node_literal() {
        assert_eq!(classify_node("\"hello\""), "literal");
    }

    #[test]
    fn test_classify_node_blank() {
        assert_eq!(classify_node("_:b0"), "blank");
    }

    #[test]
    fn test_color_for_type_values() {
        assert_eq!(color_for_type("iri"), "#4A90D9");
        assert_eq!(color_for_type("literal"), "#50C878");
        assert_eq!(color_for_type("blank"), "#B0B0B0");
        assert_eq!(color_for_type("unknown"), "#888888");
    }
}
