//! # RegionManager - dijkstra_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result as ClusterResult;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Dijkstra's algorithm implementation
    /// Returns (distances, predecessors)
    pub(super) fn dijkstra(
        &self,
        graph: &[Vec<f64>],
        source: usize,
    ) -> ClusterResult<(Vec<f64>, Vec<Option<usize>>)> {
        let n = graph.len();
        let mut distances = vec![f64::INFINITY; n];
        let mut predecessors = vec![None; n];
        let mut visited = vec![false; n];
        distances[source] = 0.0;
        for _ in 0..n {
            let mut min_dist = f64::INFINITY;
            let mut min_node = None;
            for i in 0..n {
                if !visited[i] && distances[i] < min_dist {
                    min_dist = distances[i];
                    min_node = Some(i);
                }
            }
            let Some(u) = min_node else {
                break;
            };
            visited[u] = true;
            for v in 0..n {
                if !visited[v] && graph[u][v] != f64::INFINITY {
                    let alt = distances[u] + graph[u][v];
                    if alt < distances[v] {
                        distances[v] = alt;
                        predecessors[v] = Some(u);
                    }
                }
            }
        }
        Ok((distances, predecessors))
    }
}
