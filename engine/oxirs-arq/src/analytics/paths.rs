//! Shortest-path algorithms: Dijkstra, BFS, path reconstruction

use super::adapter::{NodeId, RdfGraphAdapter};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};

/// Shortest-path algorithms for RDF graphs.
pub struct ShortestPaths;

impl ShortestPaths {
    /// Single-source shortest paths using Dijkstra's algorithm (weighted edges).
    ///
    /// Returns a map from every reachable `NodeId` to its minimum-weight
    /// distance from `source`.  Unreachable nodes are omitted.
    pub fn dijkstra(graph: &RdfGraphAdapter, source: NodeId) -> HashMap<NodeId, f64> {
        let n = graph.node_count();
        if source >= n {
            return HashMap::new();
        }

        const INF: f64 = f64::INFINITY;
        let mut dist = vec![INF; n];
        dist[source] = 0.0;

        // Min-heap: (distance × 1_000_000 as u64, node) – we wrap f64 in Reverse
        let mut heap: BinaryHeap<(Reverse<u64>, NodeId)> = BinaryHeap::new();
        heap.push((Reverse(0), source));

        while let Some((Reverse(raw_d), u)) = heap.pop() {
            let d = raw_d as f64 / 1_000_000.0;
            if d > dist[u] {
                continue; // stale entry
            }
            for &(v, w) in &graph.adjacency[u] {
                if w < 0.0 {
                    continue; // ignore negative weights
                }
                let nd = d + w;
                if nd < dist[v] {
                    dist[v] = nd;
                    let raw_nd = (nd * 1_000_000.0) as u64;
                    heap.push((Reverse(raw_nd), v));
                }
            }
        }

        dist.into_iter()
            .enumerate()
            .filter(|(_, d)| d.is_finite())
            .collect()
    }

    /// Single-source BFS distances (hop count, unweighted).
    ///
    /// Returns a map from every reachable `NodeId` to its BFS depth from
    /// `source`.  Unreachable nodes are omitted.
    pub fn bfs_distance(graph: &RdfGraphAdapter, source: NodeId) -> HashMap<NodeId, usize> {
        let n = graph.node_count();
        if source >= n {
            return HashMap::new();
        }

        let mut dist: HashMap<NodeId, usize> = HashMap::new();
        dist.insert(source, 0);
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            let d = dist[&u];
            for &(v, _) in &graph.adjacency[u] {
                if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(v) {
                    e.insert(d + 1);
                    queue.push_back(v);
                }
            }
        }
        dist
    }

    /// Find **one** shortest path (by hop count) between `source` and `target`.
    ///
    /// Returns `None` if no path exists.
    pub fn path(graph: &RdfGraphAdapter, source: NodeId, target: NodeId) -> Option<Vec<NodeId>> {
        let n = graph.node_count();
        if source >= n || target >= n {
            return None;
        }
        if source == target {
            return Some(vec![source]);
        }

        let mut prev: Vec<Option<NodeId>> = vec![None; n];
        let mut visited = vec![false; n];
        visited[source] = true;
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            if u == target {
                // Reconstruct path
                let mut path: Vec<NodeId> = Vec::new();
                let mut cur = target;
                loop {
                    path.push(cur);
                    match prev[cur] {
                        Some(p) => cur = p,
                        None => break,
                    }
                }
                path.reverse();
                return Some(path);
            }
            for &(v, _) in &graph.adjacency[u] {
                if !visited[v] {
                    visited[v] = true;
                    prev[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build unweighted graph
    fn build_uw(edges: &[(&str, &str)]) -> RdfGraphAdapter {
        let triples: Vec<(String, String, String)> = edges
            .iter()
            .map(|(s, o)| (s.to_string(), "ex:rel".to_string(), o.to_string()))
            .collect();
        RdfGraphAdapter::from_triples(&triples)
    }

    // ── Dijkstra tests ────────────────────────────────────────────────────

    #[test]
    fn test_dijkstra_trivial() {
        let g = build_uw(&[]);
        let d = ShortestPaths::dijkstra(&g, 0);
        assert!(d.is_empty());
    }

    #[test]
    fn test_dijkstra_source_distance_zero() {
        let g = build_uw(&[("ex:A", "ex:B"), ("ex:B", "ex:C")]);
        let src = g.get_node_id("ex:A").unwrap();
        let d = ShortestPaths::dijkstra(&g, src);
        assert!((d[&src] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_dijkstra_direct_edge() {
        let g = build_uw(&[("ex:A", "ex:B")]);
        let a = g.get_node_id("ex:A").unwrap();
        let b = g.get_node_id("ex:B").unwrap();
        let d = ShortestPaths::dijkstra(&g, a);
        assert!((d[&b] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dijkstra_multi_hop() {
        let g = build_uw(&[("ex:A", "ex:B"), ("ex:B", "ex:C"), ("ex:C", "ex:D")]);
        let a = g.get_node_id("ex:A").unwrap();
        let d_id = g.get_node_id("ex:D").unwrap();
        let d = ShortestPaths::dijkstra(&g, a);
        assert!((d[&d_id] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_dijkstra_unreachable_omitted() {
        let g = build_uw(&[("ex:A", "ex:B"), ("ex:C", "ex:D")]);
        let a = g.get_node_id("ex:A").unwrap();
        let c_id = g.get_node_id("ex:C").unwrap();
        let d = ShortestPaths::dijkstra(&g, a);
        assert!(!d.contains_key(&c_id));
    }

    #[test]
    fn test_dijkstra_out_of_range_source() {
        let g = build_uw(&[("ex:A", "ex:B")]);
        let d = ShortestPaths::dijkstra(&g, 999);
        assert!(d.is_empty());
    }

    #[test]
    fn test_dijkstra_shortest_path_chosen() {
        // Direct path A→D = 10; shorter path A→B→C→D = 1+1+1 = 3
        let triples = vec![
            (
                "ex:A".to_string(),
                "ex:heavy".to_string(),
                "ex:D".to_string(),
            ),
            ("ex:A".to_string(), "ex:rel".to_string(), "ex:B".to_string()),
            ("ex:B".to_string(), "ex:rel".to_string(), "ex:C".to_string()),
            ("ex:C".to_string(), "ex:rel".to_string(), "ex:D".to_string()),
        ];
        let g = RdfGraphAdapter::from_triples_weighted(&triples, |pred, _| {
            if pred == "ex:heavy" {
                10.0
            } else {
                1.0
            }
        });
        let a = g.get_node_id("ex:A").unwrap();
        let d_id = g.get_node_id("ex:D").unwrap();
        let dist = ShortestPaths::dijkstra(&g, a);
        assert!(
            dist[&d_id] < 5.0,
            "should take shorter path, got {}",
            dist[&d_id]
        );
    }

    // ── BFS distance tests ────────────────────────────────────────────────

    #[test]
    fn test_bfs_source_zero() {
        let g = build_uw(&[("ex:A", "ex:B")]);
        let a = g.get_node_id("ex:A").unwrap();
        let d = ShortestPaths::bfs_distance(&g, a);
        assert_eq!(d[&a], 0);
    }

    #[test]
    fn test_bfs_direct_neighbor() {
        let g = build_uw(&[("ex:A", "ex:B")]);
        let a = g.get_node_id("ex:A").unwrap();
        let b = g.get_node_id("ex:B").unwrap();
        let d = ShortestPaths::bfs_distance(&g, a);
        assert_eq!(d[&b], 1);
    }

    #[test]
    fn test_bfs_multi_hop() {
        let g = build_uw(&[("ex:A", "ex:B"), ("ex:B", "ex:C"), ("ex:C", "ex:D")]);
        let a = g.get_node_id("ex:A").unwrap();
        let d_id = g.get_node_id("ex:D").unwrap();
        let d = ShortestPaths::bfs_distance(&g, a);
        assert_eq!(d[&d_id], 3);
    }

    #[test]
    fn test_bfs_unreachable_omitted() {
        let g = build_uw(&[("ex:A", "ex:B"), ("ex:C", "ex:D")]);
        let a = g.get_node_id("ex:A").unwrap();
        let c_id = g.get_node_id("ex:C").unwrap();
        let d = ShortestPaths::bfs_distance(&g, a);
        assert!(!d.contains_key(&c_id));
    }

    #[test]
    fn test_bfs_out_of_range() {
        let g = build_uw(&[("ex:A", "ex:B")]);
        let d = ShortestPaths::bfs_distance(&g, 999);
        assert!(d.is_empty());
    }

    // ── Path reconstruction tests ─────────────────────────────────────────

    #[test]
    fn test_path_trivial_self() {
        let g = build_uw(&[("ex:A", "ex:B")]);
        let a = g.get_node_id("ex:A").unwrap();
        let p = ShortestPaths::path(&g, a, a).unwrap();
        assert_eq!(p, vec![a]);
    }

    #[test]
    fn test_path_direct() {
        let g = build_uw(&[("ex:A", "ex:B")]);
        let a = g.get_node_id("ex:A").unwrap();
        let b = g.get_node_id("ex:B").unwrap();
        let p = ShortestPaths::path(&g, a, b).unwrap();
        assert_eq!(p, vec![a, b]);
    }

    #[test]
    fn test_path_multi_hop() {
        let g = build_uw(&[("ex:A", "ex:B"), ("ex:B", "ex:C")]);
        let a = g.get_node_id("ex:A").unwrap();
        let c = g.get_node_id("ex:C").unwrap();
        let p = ShortestPaths::path(&g, a, c).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(*p.first().unwrap(), a);
        assert_eq!(*p.last().unwrap(), c);
    }

    #[test]
    fn test_path_no_path() {
        let g = build_uw(&[("ex:A", "ex:B"), ("ex:C", "ex:D")]);
        let a = g.get_node_id("ex:A").unwrap();
        let c = g.get_node_id("ex:C").unwrap();
        assert!(ShortestPaths::path(&g, a, c).is_none());
    }

    #[test]
    fn test_path_out_of_range() {
        let g = build_uw(&[("ex:A", "ex:B")]);
        let a = g.get_node_id("ex:A").unwrap();
        assert!(ShortestPaths::path(&g, a, 999).is_none());
        assert!(ShortestPaths::path(&g, 999, a).is_none());
    }

    #[test]
    fn test_path_starts_at_source_ends_at_target() {
        let g = build_uw(&[("ex:A", "ex:B"), ("ex:B", "ex:C"), ("ex:C", "ex:D")]);
        let a = g.get_node_id("ex:A").unwrap();
        let d = g.get_node_id("ex:D").unwrap();
        let p = ShortestPaths::path(&g, a, d).unwrap();
        assert_eq!(*p.first().unwrap(), a);
        assert_eq!(*p.last().unwrap(), d);
    }
}
