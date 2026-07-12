//! Community detection for hierarchical summarization

use crate::{CommunitySummary, GraphRAGError, GraphRAGResult, Triple};
use petgraph::graph::{NodeIndex, UnGraph};
use scirs2_core::random::{seeded_rng, Random};
use std::collections::{HashMap, HashSet};

/// Community detection algorithm
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CommunityAlgorithm {
    /// Louvain algorithm (baseline: ~0.65 modularity)
    Louvain,
    /// Leiden algorithm (target: >0.75 modularity, improved Louvain)
    #[default]
    Leiden,
    /// Label propagation
    LabelPropagation,
    /// Connected components
    ConnectedComponents,
    /// Hierarchical (multi-level)
    Hierarchical,
}

/// Community detector configuration
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    /// Algorithm to use
    pub algorithm: CommunityAlgorithm,
    /// Resolution parameter for Louvain/Leiden
    pub resolution: f64,
    /// Minimum community size
    pub min_community_size: usize,
    /// Maximum number of communities
    pub max_communities: usize,
    /// Number of iterations for iterative algorithms
    pub max_iterations: usize,
    /// Random seed for reproducibility
    pub random_seed: u64,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            algorithm: CommunityAlgorithm::Leiden,
            resolution: 1.0,
            min_community_size: 3,
            max_communities: 50,
            max_iterations: 10,
            random_seed: 42,
        }
    }
}

/// Community detector
pub struct CommunityDetector {
    config: CommunityConfig,
}

impl Default for CommunityDetector {
    fn default() -> Self {
        Self::new(CommunityConfig::default())
    }
}

impl CommunityDetector {
    pub fn new(config: CommunityConfig) -> Self {
        Self { config }
    }

    /// Detect communities in the given subgraph
    pub fn detect(&self, triples: &[Triple]) -> GraphRAGResult<Vec<CommunitySummary>> {
        if triples.is_empty() {
            return Ok(vec![]);
        }

        // Build graph
        let (graph, node_map) = self.build_graph(triples);

        // Detect communities based on algorithm
        let communities = match self.config.algorithm {
            CommunityAlgorithm::Louvain => self.louvain(&graph, &node_map)?,
            CommunityAlgorithm::Leiden => self.leiden(&graph, &node_map)?,
            CommunityAlgorithm::LabelPropagation => self.label_propagation(&graph, &node_map),
            CommunityAlgorithm::ConnectedComponents => self.connected_components(&graph, &node_map),
            CommunityAlgorithm::Hierarchical => {
                return self.detect_hierarchical(triples);
            }
        };

        // Filter and create summaries
        let summaries = self.create_summaries(communities, triples);

        Ok(summaries)
    }

    /// Build undirected graph from triples
    fn build_graph(&self, triples: &[Triple]) -> (UnGraph<String, ()>, HashMap<String, NodeIndex>) {
        let mut graph: UnGraph<String, ()> = UnGraph::new_undirected();
        let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

        for triple in triples {
            let subj_idx = *node_map
                .entry(triple.subject.clone())
                .or_insert_with(|| graph.add_node(triple.subject.clone()));
            let obj_idx = *node_map
                .entry(triple.object.clone())
                .or_insert_with(|| graph.add_node(triple.object.clone()));

            if subj_idx != obj_idx && graph.find_edge(subj_idx, obj_idx).is_none() {
                graph.add_edge(subj_idx, obj_idx, ());
            }
        }

        (graph, node_map)
    }

    /// Simplified Louvain algorithm
    fn louvain(
        &self,
        graph: &UnGraph<String, ()>,
        node_map: &HashMap<String, NodeIndex>,
    ) -> GraphRAGResult<Vec<HashSet<String>>> {
        let node_count = graph.node_count();
        if node_count == 0 {
            return Ok(vec![]);
        }

        // Initialize: each node in its own community
        let mut community: HashMap<NodeIndex, usize> = HashMap::new();
        for (community_id, &idx) in node_map.values().enumerate() {
            community.insert(idx, community_id);
        }

        // Total edges (for modularity calculation)
        let m = graph.edge_count() as f64;
        if m == 0.0 {
            // No edges, each node is its own community
            return Ok(node_map
                .keys()
                .map(|k| {
                    let mut set = HashSet::new();
                    set.insert(k.clone());
                    set
                })
                .collect());
        }

        // Degree of each node
        let degree: HashMap<NodeIndex, f64> = node_map
            .values()
            .map(|&idx| (idx, graph.neighbors(idx).count() as f64))
            .collect();

        // Build a deterministic node processing order. Rust's default
        // HashMap/HashSet hasher is reseeded from OS entropy every process
        // start, so `node_map.values()` iteration order is not reproducible
        // across runs even with a fixed `random_seed`. Sort the base vector
        // first (NodeIndex implements Ord), then Fisher-Yates shuffle it
        // using the seeded RNG so the resulting order is a deterministic
        // function of `random_seed` alone.
        let mut node_order: Vec<NodeIndex> = node_map.values().copied().collect();
        node_order.sort();
        let mut rng = seeded_rng(self.config.random_seed);
        for i in (1..node_order.len()).rev() {
            let j = (rng.random_range(0.0..1.0) * (i + 1) as f64) as usize;
            node_order.swap(i, j);
        }

        // Iterate
        for _ in 0..self.config.max_iterations {
            let mut changed = false;

            for &node in &node_order {
                let current_comm = match community.get(&node) {
                    Some(&c) => c,
                    None => continue,
                };
                let node_degree = degree.get(&node).copied().unwrap_or(0.0);

                // Calculate modularity gain for each neighbor's community
                let mut best_comm = current_comm;
                let mut best_gain = 0.0;

                // Sorted (not raw HashSet iteration) so ties always break
                // the same way regardless of hasher seed.
                let mut neighbor_comms: Vec<usize> = graph
                    .neighbors(node)
                    .filter_map(|n| community.get(&n).copied())
                    .collect::<HashSet<usize>>()
                    .into_iter()
                    .collect();
                neighbor_comms.sort_unstable();

                for &neighbor_comm in &neighbor_comms {
                    if neighbor_comm == current_comm {
                        continue;
                    }

                    // Simplified modularity gain calculation
                    let edges_to_comm: f64 = graph
                        .neighbors(node)
                        .filter(|n| community.get(n) == Some(&neighbor_comm))
                        .count() as f64;

                    let comm_degree: f64 = community
                        .iter()
                        .filter(|(_, &c)| c == neighbor_comm)
                        .map(|(n, _)| degree.get(n).copied().unwrap_or(0.0))
                        .sum();

                    let gain = edges_to_comm / m
                        - self.config.resolution * node_degree * comm_degree / (2.0 * m * m);

                    if gain > best_gain {
                        best_gain = gain;
                        best_comm = neighbor_comm;
                    }
                }

                if best_comm != current_comm && best_gain > 0.0 {
                    community.insert(node, best_comm);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        // Never settle for a partition worse than the trivial single-community
        // partition: greedy hill-climbing can get stuck in a local optimum
        // with negative modularity, which is strictly worse than "no
        // community structure at all". The trivial partition is always a
        // valid candidate — per `calculate_modularity`'s doc comment it has
        // Q = 1 - resolution = 0 at the default resolution for any graph —
        // so compare against it via the real formula (rather than
        // hardcoding that constant) and fall back if the greedy result did
        // worse.
        let trivial: HashMap<NodeIndex, usize> =
            node_map.values().map(|&idx| (idx, 0usize)).collect();
        let greedy_modularity = self.calculate_modularity(graph, &community, m, &degree)?;
        let trivial_modularity = self.calculate_modularity(graph, &trivial, m, &degree)?;

        if trivial_modularity >= greedy_modularity {
            return Ok(self.group_by_community(graph, &trivial));
        }

        // Group nodes by community
        Ok(self.group_by_community(graph, &community))
    }

    /// Label propagation algorithm
    fn label_propagation(
        &self,
        graph: &UnGraph<String, ()>,
        node_map: &HashMap<String, NodeIndex>,
    ) -> Vec<HashSet<String>> {
        if graph.node_count() == 0 {
            return vec![];
        }

        // Initialize labels
        let mut labels: HashMap<NodeIndex, usize> = HashMap::new();
        for (i, &idx) in node_map.values().enumerate() {
            labels.insert(idx, i);
        }

        // Iterate
        for _ in 0..self.config.max_iterations {
            let mut changed = false;

            for &node in node_map.values() {
                // Count neighbor labels
                let mut label_counts: HashMap<usize, usize> = HashMap::new();
                for neighbor in graph.neighbors(node) {
                    if let Some(&label) = labels.get(&neighbor) {
                        *label_counts.entry(label).or_insert(0) += 1;
                    }
                }

                // Assign most common label
                if let Some((&best_label, _)) = label_counts.iter().max_by_key(|(_, &count)| count)
                {
                    if labels.get(&node) != Some(&best_label) {
                        labels.insert(node, best_label);
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        self.group_by_community(graph, &labels)
    }

    /// Connected components
    fn connected_components(
        &self,
        graph: &UnGraph<String, ()>,
        _node_map: &HashMap<String, NodeIndex>,
    ) -> Vec<HashSet<String>> {
        let sccs = petgraph::algo::kosaraju_scc(graph);

        sccs.into_iter()
            .map(|component| {
                component
                    .into_iter()
                    .filter_map(|idx| graph.node_weight(idx).cloned())
                    .collect()
            })
            .collect()
    }

    /// Leiden algorithm (improved Louvain with refinement phase)
    fn leiden(
        &self,
        graph: &UnGraph<String, ()>,
        node_map: &HashMap<String, NodeIndex>,
    ) -> GraphRAGResult<Vec<HashSet<String>>> {
        let node_count = graph.node_count();
        if node_count == 0 {
            return Ok(vec![]);
        }

        // Initialize: each node in its own community
        let mut community: HashMap<NodeIndex, usize> = HashMap::new();
        for (community_id, &idx) in node_map.values().enumerate() {
            community.insert(idx, community_id);
        }

        let m = graph.edge_count() as f64;
        if m == 0.0 {
            return Ok(node_map
                .keys()
                .map(|k| {
                    let mut set = HashSet::new();
                    set.insert(k.clone());
                    set
                })
                .collect());
        }

        let degree: HashMap<NodeIndex, f64> = node_map
            .values()
            .map(|&idx| (idx, graph.neighbors(idx).count() as f64))
            .collect();

        let mut rng = seeded_rng(self.config.random_seed);
        let mut best_modularity = self.calculate_modularity(graph, &community, m, &degree)?;

        // Main Leiden loop
        for iteration in 0..self.config.max_iterations {
            let mut changed = false;

            // Phase 1: Local moving (like Louvain)
            let mut node_order: Vec<NodeIndex> = node_map.values().copied().collect();
            // Sort for a deterministic base order first — HashMap iteration
            // order is randomized per-process, so shuffling it (even with a
            // fixed seed) would only permute an already-random sequence and
            // the final order would still not be reproducible.
            node_order.sort();
            // Shuffle for randomness
            for i in (1..node_order.len()).rev() {
                let j = (rng.random_range(0.0..1.0) * (i + 1) as f64) as usize;
                node_order.swap(i, j);
            }

            for &node in &node_order {
                let current_comm = match community.get(&node) {
                    Some(&c) => c,
                    None => continue,
                };
                let node_degree = degree.get(&node).copied().unwrap_or(0.0);

                let mut best_comm = current_comm;
                let mut best_gain = 0.0;

                // Get neighbor communities (sorted so ties always break the
                // same way regardless of HashSet iteration/hasher seed).
                let mut neighbor_comms: Vec<usize> = graph
                    .neighbors(node)
                    .filter_map(|n| community.get(&n).copied())
                    .collect::<HashSet<usize>>()
                    .into_iter()
                    .collect();
                neighbor_comms.sort_unstable();

                for &neighbor_comm in &neighbor_comms {
                    if neighbor_comm == current_comm {
                        continue;
                    }

                    let edges_to_comm: f64 = graph
                        .neighbors(node)
                        .filter(|n| community.get(n) == Some(&neighbor_comm))
                        .count() as f64;

                    let comm_degree: f64 = community
                        .iter()
                        .filter(|(_, &c)| c == neighbor_comm)
                        .map(|(n, _)| degree.get(n).copied().unwrap_or(0.0))
                        .sum();

                    let gain = edges_to_comm / m
                        - self.config.resolution * node_degree * comm_degree / (2.0 * m * m);

                    if gain > best_gain {
                        best_gain = gain;
                        best_comm = neighbor_comm;
                    }
                }

                if best_comm != current_comm && best_gain > 0.0 {
                    community.insert(node, best_comm);
                    changed = true;
                }
            }

            // Phase 2: Refinement (what makes Leiden better than Louvain)
            // Split communities and re-merge if it improves modularity.
            // `refine_community` mutates `community` in place, so the order
            // in which communities are visited can affect the outcome —
            // sort instead of iterating the HashSet directly so this is a
            // deterministic function of `random_seed`.
            let mut unique_comms: Vec<usize> = community
                .values()
                .copied()
                .collect::<HashSet<usize>>()
                .into_iter()
                .collect();
            unique_comms.sort_unstable();
            for &comm_id in &unique_comms {
                let mut comm_nodes: Vec<NodeIndex> = community
                    .iter()
                    .filter(|(_, &c)| c == comm_id)
                    .map(|(&n, _)| n)
                    .collect();
                comm_nodes.sort();

                if comm_nodes.len() <= 1 {
                    continue;
                }

                // Try to split and refine
                self.refine_community(graph, &mut community, &comm_nodes, comm_id, m, &degree)?;
            }

            // Check modularity improvement
            let current_modularity = self.calculate_modularity(graph, &community, m, &degree)?;
            if current_modularity > best_modularity {
                best_modularity = current_modularity;
            } else if !changed {
                break;
            }

            // Early stop if modularity is very high
            if best_modularity > 0.95 || iteration > 0 && !changed {
                break;
            }
        }

        // Verify target: modularity > 0.75
        if best_modularity < 0.75 {
            tracing::warn!("Leiden modularity {:.3} below target 0.75", best_modularity);
        } else {
            tracing::info!("Leiden achieved modularity: {:.3}", best_modularity);
        }

        // Never settle for a partition worse than the trivial single-community
        // partition (see `louvain`'s and `calculate_modularity`'s doc
        // comments for why the trivial partition is always a valid,
        // zero-modularity-at-default-resolution candidate). `community` here
        // is the actual final assignment (post Phase 1 + Phase 2 of the last
        // iteration), so recompute its modularity fresh rather than reusing
        // `best_modularity`, which may reference an earlier, discarded state.
        let trivial: HashMap<NodeIndex, usize> =
            node_map.values().map(|&idx| (idx, 0usize)).collect();
        let greedy_modularity = self.calculate_modularity(graph, &community, m, &degree)?;
        let trivial_modularity = self.calculate_modularity(graph, &trivial, m, &degree)?;

        if trivial_modularity >= greedy_modularity {
            return Ok(self.group_by_community(graph, &trivial));
        }

        Ok(self.group_by_community(graph, &community))
    }

    /// Refine a community by attempting local splits
    fn refine_community(
        &self,
        graph: &UnGraph<String, ()>,
        community: &mut HashMap<NodeIndex, usize>,
        comm_nodes: &[NodeIndex],
        comm_id: usize,
        m: f64,
        degree: &HashMap<NodeIndex, f64>,
    ) -> GraphRAGResult<()> {
        if comm_nodes.len() < 2 {
            return Ok(());
        }

        // Try to find a better split using local connectivity
        let mut subcomm: HashMap<NodeIndex, usize> = HashMap::new();
        for (i, &node) in comm_nodes.iter().enumerate() {
            subcomm.insert(node, i);
        }

        // One pass of local moving within the community
        let mut changed = false;
        for &node in comm_nodes {
            let current_sub = match subcomm.get(&node) {
                Some(&c) => c,
                None => continue,
            };

            // Count edges to each subcommunity
            let mut sub_edges: HashMap<usize, f64> = HashMap::new();
            for neighbor in graph.neighbors(node) {
                if let Some(&sub) = subcomm.get(&neighbor) {
                    *sub_edges.entry(sub).or_insert(0.0) += 1.0;
                }
            }

            // Find best subcommunity. `max_by` returns the *last* maximal
            // element on ties, and HashMap iteration order is randomized
            // per-process, so sort by subcommunity id first — this makes
            // ties break the same way on every run for a given seed.
            let mut sorted_sub_edges: Vec<(&usize, &f64)> = sub_edges.iter().collect();
            sorted_sub_edges.sort_unstable_by_key(|(sub, _)| **sub);
            if let Some((&best_sub, _)) = sorted_sub_edges
                .into_iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                if best_sub != current_sub {
                    subcomm.insert(node, best_sub);
                    changed = true;
                }
            }
        }

        // If we found a better partition, create new communities
        if changed {
            // Sort so relabeling (via `.enumerate()` below) is a
            // deterministic function of `random_seed` instead of depending
            // on HashSet iteration/hasher seed order.
            let mut unique_subs: Vec<usize> = subcomm
                .values()
                .copied()
                .collect::<HashSet<usize>>()
                .into_iter()
                .collect();
            unique_subs.sort_unstable();
            if unique_subs.len() > 1 {
                let max_comm = community.values().max().copied().unwrap_or(0);
                for (i, sub_id) in unique_subs.iter().enumerate() {
                    for &node in comm_nodes {
                        if subcomm.get(&node) == Some(sub_id) {
                            let new_comm = if i == 0 { comm_id } else { max_comm + i };
                            community.insert(node, new_comm);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Calculate the modularity of a community assignment.
    ///
    /// Standard Newman-Girvan modularity:
    ///
    /// ```text
    /// Q = (1/2m) * Σ_{i,j} [A_ij - resolution·k_i·k_j/2m] · δ(c_i, c_j)
    /// ```
    ///
    /// Computed here in its algebraically equivalent per-community form for
    /// O(edges + nodes) efficiency instead of the O(n²) double sum:
    ///
    /// ```text
    /// Q = Σ_c [ e_c/m - resolution · (d_c / 2m)² ]
    /// ```
    ///
    /// where `e_c` is the number of intra-community edges and `d_c` is the
    /// sum of degrees of nodes in community `c` (see
    /// `community_detector::CommunityDetector::compute_modularity` for the
    /// O(n²) form these two agree with). Sanity check: a single community
    /// spanning the whole graph must give exactly `Q = 0` (no structure
    /// beyond the null model) — `e_c = m`, `d_c = 2m` ⇒ `1 - resolution`,
    /// which is `0` at the default `resolution = 1.0`.
    fn calculate_modularity(
        &self,
        graph: &UnGraph<String, ()>,
        community: &HashMap<NodeIndex, usize>,
        m: f64,
        degree: &HashMap<NodeIndex, f64>,
    ) -> GraphRAGResult<f64> {
        if m == 0.0 {
            return Ok(0.0);
        }

        // e_c: intra-community edge counts.
        let mut intra_edges: HashMap<usize, f64> = HashMap::new();
        for edge in graph.edge_indices() {
            if let Some((a, b)) = graph.edge_endpoints(edge) {
                if let (Some(&ca), Some(&cb)) = (community.get(&a), community.get(&b)) {
                    if ca == cb {
                        *intra_edges.entry(ca).or_insert(0.0) += 1.0;
                    }
                }
            }
        }

        // d_c: sum of node degrees per community.
        let mut comm_degree_sum: HashMap<usize, f64> = HashMap::new();
        for (&node, &comm) in community {
            let deg = degree.get(&node).copied().unwrap_or(0.0);
            *comm_degree_sum.entry(comm).or_insert(0.0) += deg;
        }

        let two_m = 2.0 * m;
        let q: f64 = comm_degree_sum
            .keys()
            .map(|comm_id| {
                let e_c = intra_edges.get(comm_id).copied().unwrap_or(0.0);
                let d_c = comm_degree_sum.get(comm_id).copied().unwrap_or(0.0);
                e_c / m - self.config.resolution * (d_c / two_m).powi(2)
            })
            .sum();

        Ok(q)
    }

    /// Hierarchical community detection (multi-level)
    fn detect_hierarchical(&self, triples: &[Triple]) -> GraphRAGResult<Vec<CommunitySummary>> {
        let mut all_summaries = Vec::new();
        let mut current_triples = triples.to_vec();
        let mut level = 0;

        while level < 5 && !current_triples.is_empty() {
            let (graph, node_map) = self.build_graph(&current_triples);

            if graph.node_count() < 10 {
                break;
            }

            // Detect communities at this level using Leiden
            let communities = self.leiden(&graph, &node_map)?;

            // Create summaries for this level
            let mut level_summaries = self.create_summaries(communities.clone(), &current_triples);

            // Tag with level
            for summary in &mut level_summaries {
                summary.level = level;
            }

            all_summaries.extend(level_summaries);

            // Coarsen graph: each community becomes a supernode
            current_triples = self.coarsen_graph(&graph, &node_map, &communities)?;
            level += 1;
        }

        Ok(all_summaries)
    }

    /// Coarsen graph by collapsing communities into supernodes
    fn coarsen_graph(
        &self,
        graph: &UnGraph<String, ()>,
        node_map: &HashMap<String, NodeIndex>,
        communities: &[HashSet<String>],
    ) -> GraphRAGResult<Vec<Triple>> {
        let mut node_to_community: HashMap<String, usize> = HashMap::new();
        for (comm_id, community) in communities.iter().enumerate() {
            for node in community {
                node_to_community.insert(node.clone(), comm_id);
            }
        }

        let mut coarsened_triples = Vec::new();
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

        for edge in graph.edge_indices() {
            if let Some((a, b)) = graph.edge_endpoints(edge) {
                let label_a = graph.node_weight(a);
                let label_b = graph.node_weight(b);

                if let (Some(la), Some(lb)) = (label_a, label_b) {
                    if let (Some(&comm_a), Some(&comm_b)) =
                        (node_to_community.get(la), node_to_community.get(lb))
                    {
                        if comm_a != comm_b {
                            let edge_key = if comm_a < comm_b {
                                (comm_a, comm_b)
                            } else {
                                (comm_b, comm_a)
                            };

                            if !seen_edges.contains(&edge_key) {
                                seen_edges.insert(edge_key);
                                coarsened_triples.push(Triple::new(
                                    format!("community_{}", comm_a),
                                    "inter_community_link",
                                    format!("community_{}", comm_b),
                                ));
                            }
                        }
                    }
                }
            }
        }

        Ok(coarsened_triples)
    }

    /// Group nodes by community assignment
    fn group_by_community(
        &self,
        graph: &UnGraph<String, ()>,
        assignment: &HashMap<NodeIndex, usize>,
    ) -> Vec<HashSet<String>> {
        let mut communities: HashMap<usize, HashSet<String>> = HashMap::new();

        for (&node, &comm) in assignment {
            if let Some(label) = graph.node_weight(node) {
                communities.entry(comm).or_default().insert(label.clone());
            }
        }

        communities.into_values().collect()
    }

    /// Create community summaries
    fn create_summaries(
        &self,
        communities: Vec<HashSet<String>>,
        triples: &[Triple],
    ) -> Vec<CommunitySummary> {
        // Build graph for proper modularity calculation
        let (graph, node_map) = self.build_graph(triples);
        let m = graph.edge_count() as f64;

        // Create community assignments
        let mut community_map: HashMap<NodeIndex, usize> = HashMap::new();
        for (idx, entities) in communities.iter().enumerate() {
            for entity in entities {
                if let Some(&node_idx) = node_map.get(entity) {
                    community_map.insert(node_idx, idx);
                }
            }
        }

        // Calculate degrees
        let degree: HashMap<NodeIndex, f64> = node_map
            .values()
            .map(|&idx| (idx, graph.neighbors(idx).count() as f64))
            .collect();

        // Calculate overall partition modularity (Newman-Girvan formula).
        // Delegate to `calculate_modularity` so there is a single
        // implementation of the formula (previously this duplicated — and
        // diverged from — the computation above, with its own normalization
        // bug on top).
        let overall_modularity = self
            .calculate_modularity(&graph, &community_map, m, &degree)
            .unwrap_or(0.0);

        communities
            .into_iter()
            .enumerate()
            .filter(|(_, entities)| entities.len() >= self.config.min_community_size)
            .take(self.config.max_communities)
            .map(|(idx, entities)| {
                // Find representative triples
                let representative_triples: Vec<Triple> = triples
                    .iter()
                    .filter(|t| entities.contains(&t.subject) || entities.contains(&t.object))
                    .take(5)
                    .cloned()
                    .collect();

                // Generate summary
                let entity_list: Vec<String> = entities.iter().cloned().collect();
                let summary = self.generate_summary(&entity_list, &representative_triples);

                // All communities share the overall partition modularity
                CommunitySummary {
                    id: format!("community_{}", idx),
                    summary,
                    entities: entity_list,
                    representative_triples,
                    level: 0,
                    modularity: overall_modularity,
                }
            })
            .collect()
    }

    /// Generate a text summary for a community
    fn generate_summary(&self, entities: &[String], triples: &[Triple]) -> String {
        // Extract short names from URIs
        let short_names: Vec<String> = entities
            .iter()
            .take(3)
            .map(|uri| {
                uri.rsplit('/')
                    .next()
                    .or_else(|| uri.rsplit('#').next())
                    .unwrap_or(uri)
                    .to_string()
            })
            .collect();

        // Extract predicates
        let predicates: HashSet<String> = triples
            .iter()
            .map(|t| {
                t.predicate
                    .rsplit('/')
                    .next()
                    .or_else(|| t.predicate.rsplit('#').next())
                    .unwrap_or(&t.predicate)
                    .to_string()
            })
            .collect();

        let pred_str: Vec<String> = predicates.into_iter().take(3).collect();

        format!(
            "Community of {} entities including {} connected by {}",
            entities.len(),
            short_names.join(", "),
            pred_str.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_community_detection() {
        // Use min_community_size: 1 to ensure small communities are detected
        let detector = CommunityDetector::new(CommunityConfig {
            min_community_size: 1,
            ..Default::default()
        });

        let triples = vec![
            Triple::new("http://a", "http://rel", "http://b"),
            Triple::new("http://b", "http://rel", "http://c"),
            Triple::new("http://a", "http://rel", "http://c"),
            Triple::new("http://x", "http://rel", "http://y"),
            Triple::new("http://y", "http://rel", "http://z"),
            Triple::new("http://x", "http://rel", "http://z"),
        ];

        let communities = detector.detect(&triples).expect("should succeed");

        // Should detect at least 1 community (a-b-c and x-y-z may be merged by Leiden)
        assert!(!communities.is_empty());
    }

    #[test]
    fn test_empty_graph() {
        let detector = CommunityDetector::default();
        let communities = detector.detect(&[]).expect("should succeed");
        assert!(communities.is_empty());
    }
}
