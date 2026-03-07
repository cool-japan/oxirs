//! Connected-components analysis: weakly and strongly connected components

use super::adapter::{NodeId, RdfGraphAdapter};
use std::collections::VecDeque;

/// Connected-component algorithms for RDF graphs.
pub struct ConnectedComponents;

impl ConnectedComponents {
    /// Weakly connected components (edge direction ignored).
    ///
    /// Uses BFS on the undirected version of the graph.
    pub fn weakly_connected(graph: &RdfGraphAdapter) -> Vec<Vec<NodeId>> {
        let n = graph.node_count();
        let mut visited = vec![false; n];
        let mut components: Vec<Vec<NodeId>> = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut component: Vec<NodeId> = Vec::new();
            let mut queue: VecDeque<NodeId> = VecDeque::new();
            queue.push_back(start);
            visited[start] = true;

            while let Some(u) = queue.pop_front() {
                component.push(u);
                // Follow both outgoing and incoming edges
                for &(v, _) in &graph.adjacency[u] {
                    if !visited[v] {
                        visited[v] = true;
                        queue.push_back(v);
                    }
                }
                for &(v, _) in &graph.reverse_adjacency[u] {
                    if !visited[v] {
                        visited[v] = true;
                        queue.push_back(v);
                    }
                }
            }
            components.push(component);
        }
        components
    }

    /// Strongly connected components via Tarjan's DFS algorithm.
    ///
    /// Returns a list of SCCs in reverse topological order.
    pub fn strongly_connected(graph: &RdfGraphAdapter) -> Vec<Vec<NodeId>> {
        let n = graph.node_count();
        let mut index_counter = 0usize;
        let mut stack: Vec<NodeId> = Vec::new();
        let mut lowlink = vec![usize::MAX; n];
        let mut index = vec![usize::MAX; n];
        let mut on_stack = vec![false; n];
        let mut sccs: Vec<Vec<NodeId>> = Vec::new();

        for v in 0..n {
            if index[v] == usize::MAX {
                Self::tarjan_dfs(
                    v,
                    graph,
                    &mut index_counter,
                    &mut stack,
                    &mut lowlink,
                    &mut index,
                    &mut on_stack,
                    &mut sccs,
                );
            }
        }
        sccs
    }

    /// Iterative Tarjan DFS (avoids stack overflow on large graphs).
    #[allow(clippy::too_many_arguments)]
    fn tarjan_dfs(
        root: NodeId,
        graph: &RdfGraphAdapter,
        index_counter: &mut usize,
        stack: &mut Vec<NodeId>,
        lowlink: &mut [usize],
        index: &mut [usize],
        on_stack: &mut [bool],
        sccs: &mut Vec<Vec<NodeId>>,
    ) {
        // Iterative version using an explicit call stack.
        // Each frame stores: (node, iterator position into adjacency list)
        let mut call_stack: Vec<(NodeId, usize)> = Vec::new();

        // Initialise root
        index[root] = *index_counter;
        lowlink[root] = *index_counter;
        *index_counter += 1;
        stack.push(root);
        on_stack[root] = true;
        call_stack.push((root, 0));

        while let Some((v, edge_idx)) = call_stack.last_mut() {
            let v = *v;
            let adj = &graph.adjacency[v];

            if *edge_idx < adj.len() {
                let (w, _) = adj[*edge_idx];
                *edge_idx += 1;

                if index[w] == usize::MAX {
                    // Tree edge – recurse
                    index[w] = *index_counter;
                    lowlink[w] = *index_counter;
                    *index_counter += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    call_stack.push((w, 0));
                } else if on_stack[w] {
                    // Back edge
                    let lv = lowlink[v];
                    lowlink[v] = lv.min(index[w]);
                }
            } else {
                // All neighbours processed – pop frame
                call_stack.pop();

                if let Some(&(parent, _)) = call_stack.last() {
                    let lv = lowlink[parent];
                    lowlink[parent] = lv.min(lowlink[v]);
                }

                // Check if v is a root of an SCC
                if lowlink[v] == index[v] {
                    let mut scc: Vec<NodeId> = Vec::new();
                    loop {
                        let w = stack.pop().unwrap_or(v);
                        on_stack[w] = false;
                        scc.push(w);
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(scc);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_graph(edges: &[(&str, &str)]) -> RdfGraphAdapter {
        let triples: Vec<(String, String, String)> = edges
            .iter()
            .map(|(s, o)| (s.to_string(), "ex:rel".to_string(), o.to_string()))
            .collect();
        RdfGraphAdapter::from_triples(&triples)
    }

    // ── Weakly connected ──────────────────────────────────────────────────

    #[test]
    fn test_wcc_single_component() {
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C")]);
        let comps = ConnectedComponents::weakly_connected(&g);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].len(), 3);
    }

    #[test]
    fn test_wcc_two_disconnected() {
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:C", "ex:D")]);
        let comps = ConnectedComponents::weakly_connected(&g);
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_wcc_empty_graph() {
        let g = RdfGraphAdapter::from_triples(&[]);
        let comps = ConnectedComponents::weakly_connected(&g);
        assert!(comps.is_empty());
    }

    #[test]
    fn test_wcc_all_nodes_covered() {
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:C", "ex:D"), ("ex:E", "ex:F")]);
        let comps = ConnectedComponents::weakly_connected(&g);
        let total: usize = comps.iter().map(|c| c.len()).sum();
        assert_eq!(total, g.node_count());
    }

    #[test]
    fn test_wcc_directed_treated_undirected() {
        // A → B  and  C → B  are weakly connected via B
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:C", "ex:B")]);
        let comps = ConnectedComponents::weakly_connected(&g);
        assert_eq!(comps.len(), 1);
    }

    // ── Strongly connected ────────────────────────────────────────────────

    #[test]
    fn test_scc_simple_cycle() {
        // A → B → C → A  forms one SCC
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C"), ("ex:C", "ex:A")]);
        let sccs = ConnectedComponents::strongly_connected(&g);
        // One SCC of size 3
        let big: Vec<&Vec<NodeId>> = sccs.iter().filter(|c| c.len() == 3).collect();
        assert_eq!(big.len(), 1);
    }

    #[test]
    fn test_scc_dag_all_trivial() {
        // In a DAG every SCC is a single node
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C")]);
        let sccs = ConnectedComponents::strongly_connected(&g);
        assert_eq!(sccs.len(), 3);
        for scc in &sccs {
            assert_eq!(scc.len(), 1);
        }
    }

    #[test]
    fn test_scc_two_cycles() {
        // Cycle1: A→B→A   Cycle2: C→D→C
        let g = build_graph(&[
            ("ex:A", "ex:B"),
            ("ex:B", "ex:A"),
            ("ex:C", "ex:D"),
            ("ex:D", "ex:C"),
        ]);
        let sccs = ConnectedComponents::strongly_connected(&g);
        let size2: Vec<&Vec<NodeId>> = sccs.iter().filter(|c| c.len() == 2).collect();
        assert_eq!(size2.len(), 2);
    }

    #[test]
    fn test_scc_empty_graph() {
        let g = RdfGraphAdapter::from_triples(&[]);
        let sccs = ConnectedComponents::strongly_connected(&g);
        assert!(sccs.is_empty());
    }

    #[test]
    fn test_scc_all_nodes_covered() {
        let g = build_graph(&[
            ("ex:A", "ex:B"),
            ("ex:B", "ex:C"),
            ("ex:C", "ex:A"),
            ("ex:D", "ex:E"),
        ]);
        let sccs = ConnectedComponents::strongly_connected(&g);
        let total: usize = sccs.iter().map(|c| c.len()).sum();
        assert_eq!(total, g.node_count());
    }

    #[test]
    fn test_scc_cycle_with_tail() {
        // A → B → C → B  (B–C cycle, A is a tail)
        let g = build_graph(&[("ex:A", "ex:B"), ("ex:B", "ex:C"), ("ex:C", "ex:B")]);
        let sccs = ConnectedComponents::strongly_connected(&g);
        // B and C form one SCC of size 2; A is trivial
        let large: Vec<&Vec<NodeId>> = sccs.iter().filter(|c| c.len() == 2).collect();
        assert_eq!(large.len(), 1);
    }
}
