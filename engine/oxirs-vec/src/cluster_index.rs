//! K-means clustering index for approximate nearest-neighbour search.
//!
//! Features:
//! * Lloyd's algorithm with configurable `k` and maximum iterations
//! * Cluster assignment (nearest centroid)
//! * Centroid tracking (incremental updates as vectors are inserted)
//! * Cluster statistics (size, intra-cluster variance, centroid drift)
//! * Cluster merge (merge two closest clusters)
//! * Cluster split (split the largest cluster)
//! * ANN index search (probe nearest clusters for a query vector)

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the cluster index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterError {
    /// Requested `k` is 0 or larger than the number of vectors.
    InvalidK { k: usize, n: usize },
    /// Operation requested on an empty index.
    EmptyIndex,
    /// The cluster id does not exist.
    UnknownCluster(usize),
    /// Cannot merge a cluster with itself.
    SameCluster,
    /// A vector with this id is already in the index.
    DuplicateId(String),
    /// Vectors have incompatible dimensionalities.
    DimMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for ClusterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClusterError::InvalidK { k, n } => {
                write!(f, "k={k} is invalid for {n} vectors")
            }
            ClusterError::EmptyIndex => write!(f, "the index is empty"),
            ClusterError::UnknownCluster(id) => write!(f, "unknown cluster id {id}"),
            ClusterError::SameCluster => write!(f, "cannot merge a cluster with itself"),
            ClusterError::DuplicateId(id) => write!(f, "duplicate vector id '{id}'"),
            ClusterError::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two equal-length slices.
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Euclidean distance.
fn dist(a: &[f32], b: &[f32]) -> f32 {
    sq_dist(a, b).sqrt()
}

/// Component-wise mean of a collection of vectors.
/// Returns `None` when `vectors` is empty.
#[allow(dead_code)]
fn mean_vector(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    if vectors.is_empty() {
        return None;
    }
    let dim = vectors[0].len();
    let mut sum = vec![0.0f32; dim];
    for v in vectors {
        for (s, x) in sum.iter_mut().zip(v.iter()) {
            *s += x;
        }
    }
    let n = vectors.len() as f32;
    Some(sum.into_iter().map(|s| s / n).collect())
}

// ---------------------------------------------------------------------------
// Cluster statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for a single cluster.
#[derive(Debug, Clone)]
pub struct ClusterStats {
    /// Cluster identifier.
    pub cluster_id: usize,
    /// Number of vectors assigned to this cluster.
    pub size: usize,
    /// Mean squared distance from members to centroid (intra-cluster variance).
    pub variance: f32,
    /// Euclidean distance the centroid moved on the last update.
    pub centroid_drift: f32,
    /// The centroid coordinates.
    pub centroid: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Cluster index
// ---------------------------------------------------------------------------

/// A stored vector entry.
#[derive(Debug, Clone)]
struct Entry {
    vector: Vec<f32>,
    cluster_id: usize,
}

/// K-means clustering index supporting ANN search by cluster probing.
#[derive(Debug, Clone)]
pub struct ClusterIndex {
    /// Number of clusters.
    k: usize,
    /// Maximum Lloyd iterations during `build`.
    max_iter: usize,
    /// Dimensionality (set on first insertion).
    dim: Option<usize>,
    /// All stored vectors keyed by their string id.
    entries: HashMap<String, Entry>,
    /// Ordered list of vector ids (for centroid computation by cluster).
    id_order: Vec<String>,
    /// Centroids: one row per cluster.
    centroids: Vec<Vec<f32>>,
    /// Previous centroids for drift computation.
    prev_centroids: Vec<Vec<f32>>,
    /// Next cluster id counter (reserved for future merge/split expansion).
    #[allow(dead_code)]
    next_cluster_id: usize,
}

impl ClusterIndex {
    /// Create a new index.
    ///
    /// * `k`        — number of clusters (must be ≥ 1)
    /// * `max_iter` — maximum Lloyd iterations when `build` is called
    pub fn new(k: usize, max_iter: usize) -> Self {
        Self {
            k,
            max_iter,
            dim: None,
            entries: HashMap::new(),
            id_order: Vec::new(),
            centroids: Vec::new(),
            prev_centroids: Vec::new(),
            next_cluster_id: k,
        }
    }

    /// Current number of clusters.
    pub fn num_clusters(&self) -> usize {
        self.centroids.len()
    }

    /// Current number of stored vectors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when no vectors have been inserted.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // -----------------------------------------------------------------------
    // Insertion
    // -----------------------------------------------------------------------

    /// Insert a vector into the index, assigning it to the nearest centroid.
    ///
    /// If no clustering has been built yet (`centroids` is empty), the vector
    /// is stored without a valid cluster assignment and an initial cluster will
    /// be assigned during `build`.
    ///
    /// Returns an error when:
    /// * The id is already present.
    /// * The vector dimensionality is inconsistent with existing vectors.
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<(), ClusterError> {
        // Dimension check
        match self.dim {
            None => self.dim = Some(vector.len()),
            Some(d) if d != vector.len() => {
                return Err(ClusterError::DimMismatch {
                    expected: d,
                    got: vector.len(),
                })
            }
            _ => {}
        }
        if self.entries.contains_key(&id) {
            return Err(ClusterError::DuplicateId(id));
        }
        let cluster_id = if self.centroids.is_empty() {
            0 // placeholder; cluster assigned after build()
        } else {
            self.nearest_centroid_idx(&vector)
        };
        self.entries
            .insert(id.clone(), Entry { vector, cluster_id });
        self.id_order.push(id);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Build (Lloyd's algorithm)
    // -----------------------------------------------------------------------

    /// Run Lloyd's K-means algorithm to assign all vectors to clusters.
    ///
    /// Initialisation: k-means++ style — first centroid chosen as the first
    /// vector, subsequent centroids chosen as the furthest from the current set.
    pub fn build(&mut self) -> Result<(), ClusterError> {
        let n = self.entries.len();
        if n == 0 {
            return Err(ClusterError::EmptyIndex);
        }
        let effective_k = self.k.min(n);
        if effective_k == 0 {
            return Err(ClusterError::InvalidK { k: self.k, n });
        }

        let all_vecs: Vec<Vec<f32>> = self
            .id_order
            .iter()
            .map(|id| self.entries[id].vector.clone())
            .collect();
        let dim = all_vecs[0].len();

        // K-means++ initialisation
        let mut centroids = vec![all_vecs[0].clone()];
        while centroids.len() < effective_k {
            let mut max_dist = f32::NEG_INFINITY;
            let mut best_idx = 0usize;
            for (i, v) in all_vecs.iter().enumerate() {
                let min_d = centroids
                    .iter()
                    .map(|c| sq_dist(v, c))
                    .fold(f32::INFINITY, f32::min);
                if min_d > max_dist {
                    max_dist = min_d;
                    best_idx = i;
                }
            }
            centroids.push(all_vecs[best_idx].clone());
        }

        // Lloyd iterations
        for _ in 0..self.max_iter {
            // Assignment step
            let assignments: Vec<usize> = all_vecs
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            sq_dist(v, a)
                                .partial_cmp(&sq_dist(v, b))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .collect();

            // Update step
            let mut new_centroids = vec![vec![0.0f32; dim]; effective_k];
            let mut counts = vec![0usize; effective_k];
            for (v, &c) in all_vecs.iter().zip(assignments.iter()) {
                for (nc, x) in new_centroids[c].iter_mut().zip(v.iter()) {
                    *nc += x;
                }
                counts[c] += 1;
            }
            let mut converged = true;
            for (i, nc) in new_centroids.iter_mut().enumerate() {
                let cnt = counts[i].max(1);
                for x in nc.iter_mut() {
                    *x /= cnt as f32;
                }
                if dist(nc, &centroids[i]) > 1e-6 {
                    converged = false;
                }
            }
            centroids = new_centroids;
            if converged {
                break;
            }
        }

        // Final assignment of all entries
        let assignments: Vec<usize> = all_vecs
            .iter()
            .map(|v| {
                centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        sq_dist(v, a)
                            .partial_cmp(&sq_dist(v, b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();

        self.prev_centroids = self.centroids.clone();
        self.centroids = centroids;
        for (i, id) in self.id_order.iter().enumerate() {
            if let Some(entry) = self.entries.get_mut(id) {
                entry.cluster_id = assignments[i];
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Cluster assignment
    // -----------------------------------------------------------------------

    /// Return the index of the centroid nearest to `query`.
    pub fn assign(&self, query: &[f32]) -> Option<usize> {
        if self.centroids.is_empty() {
            return None;
        }
        Some(self.nearest_centroid_idx(query))
    }

    fn nearest_centroid_idx(&self, query: &[f32]) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                sq_dist(query, a)
                    .partial_cmp(&sq_dist(query, b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // Cluster statistics
    // -----------------------------------------------------------------------

    /// Return statistics for the cluster with the given id.
    pub fn cluster_stats(&self, cluster_id: usize) -> Result<ClusterStats, ClusterError> {
        if cluster_id >= self.centroids.len() {
            return Err(ClusterError::UnknownCluster(cluster_id));
        }
        let centroid = &self.centroids[cluster_id];
        let members: Vec<&Vec<f32>> = self
            .entries
            .values()
            .filter(|e| e.cluster_id == cluster_id)
            .map(|e| &e.vector)
            .collect();
        let size = members.len();
        let variance = if size == 0 {
            0.0
        } else {
            members.iter().map(|v| sq_dist(v, centroid)).sum::<f32>() / size as f32
        };
        let drift = if self.prev_centroids.len() > cluster_id {
            dist(centroid, &self.prev_centroids[cluster_id])
        } else {
            0.0
        };
        Ok(ClusterStats {
            cluster_id,
            size,
            variance,
            centroid_drift: drift,
            centroid: centroid.clone(),
        })
    }

    /// Return statistics for all clusters.
    pub fn all_cluster_stats(&self) -> Vec<ClusterStats> {
        (0..self.centroids.len())
            .filter_map(|id| self.cluster_stats(id).ok())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Cluster merge
    // -----------------------------------------------------------------------

    /// Merge clusters `a` and `b` into a single cluster.
    ///
    /// The merged centroid is the weighted mean of the two cluster centroids.
    /// All members of both clusters are reassigned to the lower id; the higher
    /// id slot is removed by swapping with the last centroid and truncating.
    pub fn merge_clusters(&mut self, a: usize, b: usize) -> Result<(), ClusterError> {
        if a == b {
            return Err(ClusterError::SameCluster);
        }
        let n = self.centroids.len();
        if a >= n {
            return Err(ClusterError::UnknownCluster(a));
        }
        if b >= n {
            return Err(ClusterError::UnknownCluster(b));
        }
        let (keep, remove) = if a < b { (a, b) } else { (b, a) };

        // Count members in each cluster for weighted centroid
        let count_keep = self
            .entries
            .values()
            .filter(|e| e.cluster_id == keep)
            .count();
        let count_remove = self
            .entries
            .values()
            .filter(|e| e.cluster_id == remove)
            .count();
        let total = count_keep + count_remove;

        let dim = self.centroids[0].len();
        let mut merged = vec![0.0f32; dim];
        let w_keep = count_keep as f32 / total.max(1) as f32;
        let w_remove = count_remove as f32 / total.max(1) as f32;
        for (m, (ck, cr)) in merged.iter_mut().zip(
            self.centroids[keep]
                .iter()
                .zip(self.centroids[remove].iter()),
        ) {
            *m = ck * w_keep + cr * w_remove;
        }
        self.centroids[keep] = merged;

        // Reassign members of `remove` to `keep`
        let last = n - 1;
        for entry in self.entries.values_mut() {
            if entry.cluster_id == remove {
                entry.cluster_id = keep;
            }
            // Fix up entries pointing to the swapped-in last centroid
            if remove != last && entry.cluster_id == last {
                entry.cluster_id = remove;
            }
        }

        // Swap `remove` with last and truncate
        self.centroids.swap(remove, last);
        self.centroids.pop();
        if !self.prev_centroids.is_empty() {
            if self.prev_centroids.len() > last {
                self.prev_centroids.swap(remove, last);
                self.prev_centroids.pop();
            } else {
                self.prev_centroids.clear();
            }
        }
        Ok(())
    }

    /// Merge the two closest clusters (by centroid distance).
    pub fn merge_closest_clusters(&mut self) -> Result<(), ClusterError> {
        let n = self.centroids.len();
        if n < 2 {
            return Err(ClusterError::EmptyIndex);
        }
        let (mut best_i, mut best_j) = (0, 1);
        let mut best_dist = f32::INFINITY;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dist(&self.centroids[i], &self.centroids[j]);
                if d < best_dist {
                    best_dist = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        self.merge_clusters(best_i, best_j)
    }

    // -----------------------------------------------------------------------
    // Cluster split
    // -----------------------------------------------------------------------

    /// Split the largest cluster into two by creating a new centroid displaced
    /// by the principal direction (first PCA component approximated as the
    /// direction of greatest variance along each axis).
    ///
    /// The new centroid is appended to the end; half the members are
    /// re-assigned to it.
    pub fn split_largest_cluster(&mut self) -> Result<(), ClusterError> {
        if self.centroids.is_empty() {
            return Err(ClusterError::EmptyIndex);
        }
        // Find the largest cluster
        let mut counts = vec![0usize; self.centroids.len()];
        for entry in self.entries.values() {
            if entry.cluster_id < counts.len() {
                counts[entry.cluster_id] += 1;
            }
        }
        let largest = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i);
        let cluster_id = match largest {
            Some(id) if counts[id] >= 2 => id,
            _ => return Err(ClusterError::EmptyIndex),
        };

        let members: Vec<String> = self
            .id_order
            .iter()
            .filter(|id| {
                self.entries
                    .get(*id)
                    .map(|e| e.cluster_id == cluster_id)
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        let member_vecs: Vec<&Vec<f32>> = members
            .iter()
            .filter_map(|id| self.entries.get(id).map(|e| &e.vector))
            .collect();

        let dim = self.centroids[0].len();
        // Compute per-axis variance; split along highest-variance axis
        let centroid = self.centroids[cluster_id].clone();
        let mut axis_var = vec![0.0f32; dim];
        for v in &member_vecs {
            for (d, x) in v.iter().enumerate() {
                let diff = x - centroid[d];
                axis_var[d] += diff * diff;
            }
        }
        let n_members = member_vecs.len() as f32;
        for ax in &mut axis_var {
            *ax /= n_members;
        }
        let split_axis = axis_var
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let spread = axis_var[split_axis].sqrt() * 0.5;

        let mut c1 = centroid.clone();
        let mut c2 = centroid.clone();
        c1[split_axis] -= spread;
        c2[split_axis] += spread;

        let new_id = self.centroids.len();
        self.centroids[cluster_id] = c1.clone();
        self.centroids.push(c2.clone());

        // Reassign each member to the nearer of the two new centroids
        let half = members.len() / 2;
        for (i, member_id) in members.iter().enumerate() {
            if let Some(entry) = self.entries.get_mut(member_id) {
                entry.cluster_id = if i < half { cluster_id } else { new_id };
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // ANN search by cluster probing
    // -----------------------------------------------------------------------

    /// Search for the `top_k` nearest vectors to `query` by probing the
    /// `n_probes` closest cluster centroids.
    ///
    /// Returns a list of `(id, distance)` pairs sorted by ascending distance.
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        n_probes: usize,
    ) -> Result<Vec<(String, f32)>, ClusterError> {
        if self.entries.is_empty() {
            return Err(ClusterError::EmptyIndex);
        }
        let n_probes = n_probes.min(self.centroids.len());

        // Rank clusters by distance from query
        let mut cluster_dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, dist(query, c)))
            .collect();
        cluster_dists
            .sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Collect candidates from the nearest n_probes clusters
        let probe_set: std::collections::HashSet<usize> = cluster_dists
            .iter()
            .take(n_probes)
            .map(|(i, _)| *i)
            .collect();

        let mut candidates: Vec<(String, f32)> = self
            .entries
            .iter()
            .filter(|(_, e)| probe_set.contains(&e.cluster_id))
            .map(|(id, e)| (id.clone(), dist(query, &e.vector)))
            .collect();

        candidates.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);
        Ok(candidates)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index(k: usize) -> ClusterIndex {
        ClusterIndex::new(k, 20)
    }

    fn insert_and_build(k: usize, vectors: Vec<(&str, Vec<f32>)>) -> ClusterIndex {
        let mut idx = make_index(k);
        for (id, v) in vectors {
            idx.insert(id.to_string(), v).expect("insert");
        }
        idx.build().expect("build");
        idx
    }

    // -----------------------------------------------------------------------
    // Basic insertion
    // -----------------------------------------------------------------------

    #[test]
    fn test_insert_single_vector() {
        let mut idx = make_index(1);
        idx.insert("v0".into(), vec![1.0, 0.0]).expect("insert");
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_insert_duplicate_id_error() {
        let mut idx = make_index(1);
        idx.insert("v0".into(), vec![1.0, 0.0]).expect("insert");
        let result = idx.insert("v0".into(), vec![0.0, 1.0]);
        assert!(matches!(result, Err(ClusterError::DuplicateId(_))));
    }

    #[test]
    fn test_insert_dim_mismatch_error() {
        let mut idx = make_index(1);
        idx.insert("v0".into(), vec![1.0, 0.0]).expect("insert");
        let result = idx.insert("v1".into(), vec![1.0, 0.0, 0.0]);
        assert!(matches!(result, Err(ClusterError::DimMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // Build
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_single_cluster() {
        let idx = insert_and_build(
            1,
            vec![
                ("a", vec![1.0, 0.0]),
                ("b", vec![2.0, 0.0]),
                ("c", vec![3.0, 0.0]),
            ],
        );
        assert_eq!(idx.num_clusters(), 1);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn test_build_two_clusters() {
        // Two well-separated groups
        let idx = insert_and_build(
            2,
            vec![
                ("a", vec![0.0, 0.0]),
                ("b", vec![0.1, 0.0]),
                ("c", vec![10.0, 0.0]),
                ("d", vec![10.1, 0.0]),
            ],
        );
        assert_eq!(idx.num_clusters(), 2);
    }

    #[test]
    fn test_build_empty_index_error() {
        let mut idx = make_index(3);
        let result = idx.build();
        assert!(matches!(result, Err(ClusterError::EmptyIndex)));
    }

    // -----------------------------------------------------------------------
    // Cluster assignment
    // -----------------------------------------------------------------------

    #[test]
    fn test_assign_returns_cluster() {
        let idx = insert_and_build(
            2,
            vec![
                ("a", vec![0.0f32, 0.0]),
                ("b", vec![0.0, 0.1]),
                ("c", vec![10.0, 0.0]),
                ("d", vec![10.0, 0.1]),
            ],
        );
        let cluster_near_origin = idx.assign(&[0.05, 0.0]).expect("assign");
        let cluster_far = idx.assign(&[10.05, 0.0]).expect("assign");
        assert_ne!(cluster_near_origin, cluster_far);
    }

    #[test]
    fn test_assign_empty_returns_none() {
        let idx = make_index(2);
        assert!(idx.assign(&[1.0, 0.0]).is_none());
    }

    // -----------------------------------------------------------------------
    // Cluster statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_cluster_stats_size() {
        let idx = insert_and_build(1, vec![("a", vec![1.0f32, 0.0]), ("b", vec![2.0, 0.0])]);
        let stats = idx.cluster_stats(0).expect("stats");
        assert_eq!(stats.cluster_id, 0);
        assert_eq!(stats.size, 2);
    }

    #[test]
    fn test_cluster_stats_unknown_error() {
        let idx = insert_and_build(1, vec![("a", vec![1.0f32, 0.0])]);
        let result = idx.cluster_stats(99);
        assert!(matches!(result, Err(ClusterError::UnknownCluster(99))));
    }

    #[test]
    fn test_cluster_stats_variance_non_negative() {
        let idx = insert_and_build(
            1,
            vec![
                ("a", vec![1.0f32, 0.0]),
                ("b", vec![2.0, 0.0]),
                ("c", vec![3.0, 0.0]),
            ],
        );
        let stats = idx.cluster_stats(0).expect("stats");
        assert!(stats.variance >= 0.0);
    }

    #[test]
    fn test_all_cluster_stats_count() {
        let idx = insert_and_build(
            2,
            vec![
                ("a", vec![0.0f32, 0.0]),
                ("b", vec![0.0, 0.1]),
                ("c", vec![10.0, 0.0]),
                ("d", vec![10.0, 0.1]),
            ],
        );
        assert_eq!(idx.all_cluster_stats().len(), 2);
    }

    // -----------------------------------------------------------------------
    // Cluster merge
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_clusters_reduces_count() {
        let mut idx = insert_and_build(
            2,
            vec![
                ("a", vec![0.0f32, 0.0]),
                ("b", vec![0.0, 0.1]),
                ("c", vec![10.0, 0.0]),
                ("d", vec![10.0, 0.1]),
            ],
        );
        idx.merge_clusters(0, 1).expect("merge");
        assert_eq!(idx.num_clusters(), 1);
    }

    #[test]
    fn test_merge_same_cluster_error() {
        let mut idx = insert_and_build(1, vec![("a", vec![1.0f32, 0.0])]);
        let result = idx.merge_clusters(0, 0);
        assert!(matches!(result, Err(ClusterError::SameCluster)));
    }

    #[test]
    fn test_merge_closest_clusters() {
        let mut idx = insert_and_build(
            3,
            vec![
                ("a", vec![0.0f32, 0.0]),
                ("b", vec![0.0, 0.1]),
                ("c", vec![5.0, 0.0]),
                ("d", vec![100.0, 0.0]),
                ("e", vec![100.0, 0.1]),
            ],
        );
        let before = idx.num_clusters();
        idx.merge_closest_clusters().expect("merge");
        assert_eq!(idx.num_clusters(), before - 1);
    }

    #[test]
    fn test_merge_unknown_cluster_error() {
        let mut idx = insert_and_build(1, vec![("a", vec![1.0f32, 0.0])]);
        let result = idx.merge_clusters(0, 99);
        assert!(matches!(result, Err(ClusterError::UnknownCluster(99))));
    }

    // -----------------------------------------------------------------------
    // Cluster split
    // -----------------------------------------------------------------------

    #[test]
    fn test_split_largest_cluster_increases_count() {
        let mut idx = insert_and_build(
            1,
            vec![
                ("a", vec![1.0f32, 0.0]),
                ("b", vec![2.0, 0.0]),
                ("c", vec![3.0, 0.0]),
                ("d", vec![4.0, 0.0]),
            ],
        );
        let before = idx.num_clusters();
        idx.split_largest_cluster().expect("split");
        assert_eq!(idx.num_clusters(), before + 1);
    }

    #[test]
    fn test_split_empty_error() {
        let mut idx = make_index(1);
        assert!(matches!(
            idx.split_largest_cluster(),
            Err(ClusterError::EmptyIndex)
        ));
    }

    // -----------------------------------------------------------------------
    // ANN search
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_returns_nearest() {
        let idx = insert_and_build(
            2,
            vec![
                ("origin", vec![0.0f32, 0.0]),
                ("near", vec![0.1, 0.0]),
                ("far", vec![10.0, 0.0]),
            ],
        );
        let results = idx.search(&[0.0, 0.0], 1, 2).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "origin");
    }

    #[test]
    fn test_search_top_k_limit() {
        let idx = insert_and_build(
            1,
            vec![
                ("a", vec![0.0f32, 0.0]),
                ("b", vec![1.0, 0.0]),
                ("c", vec![2.0, 0.0]),
            ],
        );
        let results = idx.search(&[0.0, 0.0], 2, 1).expect("search");
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_search_empty_error() {
        let idx = make_index(2);
        let result = idx.search(&[0.0, 0.0], 3, 1);
        assert!(matches!(result, Err(ClusterError::EmptyIndex)));
    }

    #[test]
    fn test_search_results_sorted_by_distance() {
        let idx = insert_and_build(
            1,
            vec![
                ("near", vec![0.1f32, 0.0]),
                ("mid", vec![1.0, 0.0]),
                ("far", vec![5.0, 0.0]),
            ],
        );
        let results = idx.search(&[0.0, 0.0], 3, 1).expect("search");
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    // -----------------------------------------------------------------------
    // is_empty / len
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_empty_initial() {
        let idx = make_index(2);
        assert!(idx.is_empty());
    }

    #[test]
    fn test_len_after_inserts() {
        let mut idx = make_index(2);
        idx.insert("v0".into(), vec![0.0f32, 0.0]).expect("insert");
        idx.insert("v1".into(), vec![1.0, 0.0]).expect("insert");
        assert_eq!(idx.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Additional edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_k_equals_n() {
        // k == number of vectors → each vector is its own centroid
        let idx = insert_and_build(
            3,
            vec![
                ("a", vec![0.0f32, 0.0]),
                ("b", vec![5.0, 0.0]),
                ("c", vec![10.0, 0.0]),
            ],
        );
        assert_eq!(idx.num_clusters(), 3);
    }

    #[test]
    fn test_build_more_k_than_vectors_clamped() {
        // k > n → clamped to n
        let idx = insert_and_build(10, vec![("a", vec![0.0f32, 0.0]), ("b", vec![1.0, 0.0])]);
        assert_eq!(idx.num_clusters(), 2);
    }

    #[test]
    fn test_assign_after_build_consistent() {
        let idx = insert_and_build(1, vec![("a", vec![3.0f32, 3.0]), ("b", vec![3.1, 3.1])]);
        // Both should map to cluster 0 (only cluster)
        assert_eq!(idx.assign(&[3.05, 3.05]), Some(0));
    }

    #[test]
    fn test_cluster_stats_single_member_zero_variance() {
        let mut idx = make_index(1);
        idx.insert("only".into(), vec![7.0f32, 7.0])
            .expect("insert");
        idx.build().expect("build");
        let stats = idx.cluster_stats(0).expect("stats");
        assert_eq!(stats.size, 1);
        assert!(stats.variance < 1e-6);
    }

    #[test]
    fn test_all_stats_total_size_equals_len() {
        let idx = insert_and_build(
            2,
            vec![
                ("a", vec![0.0f32, 0.0]),
                ("b", vec![0.0, 0.1]),
                ("c", vec![10.0, 0.0]),
                ("d", vec![10.0, 0.1]),
            ],
        );
        let total: usize = idx.all_cluster_stats().iter().map(|s| s.size).sum();
        assert_eq!(total, idx.len());
    }

    #[test]
    fn test_merge_all_members_accounted_for() {
        let mut idx = insert_and_build(
            2,
            vec![
                ("a", vec![0.0f32, 0.0]),
                ("b", vec![0.0, 0.1]),
                ("c", vec![10.0, 0.0]),
                ("d", vec![10.0, 0.1]),
            ],
        );
        idx.merge_clusters(0, 1).expect("merge");
        let stats = idx.cluster_stats(0).expect("stats");
        assert_eq!(stats.size, 4); // all members merged
    }

    #[test]
    fn test_split_two_disjoint_halves() {
        let idx = insert_and_build(
            1,
            vec![
                ("lo1", vec![-5.0f32, 0.0]),
                ("lo2", vec![-4.0, 0.0]),
                ("hi1", vec![4.0, 0.0]),
                ("hi2", vec![5.0, 0.0]),
            ],
        );
        let stats = idx.all_cluster_stats();
        assert!(!stats.is_empty());
    }

    #[test]
    fn test_search_n_probes_one_finds_all_in_cluster() {
        let idx = insert_and_build(
            1,
            vec![
                ("a", vec![1.0f32, 0.0]),
                ("b", vec![2.0, 0.0]),
                ("c", vec![3.0, 0.0]),
            ],
        );
        let results = idx.search(&[2.0, 0.0], 3, 1).expect("search");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_cluster_index_3d_vectors() {
        let idx = insert_and_build(
            2,
            vec![
                ("a", vec![0.0f32, 0.0, 0.0]),
                ("b", vec![0.1, 0.0, 0.0]),
                ("c", vec![10.0, 10.0, 10.0]),
                ("d", vec![10.1, 10.0, 10.0]),
            ],
        );
        let r = idx.search(&[0.05, 0.0, 0.0], 1, 1).expect("search");
        assert!(!r.is_empty());
    }

    #[test]
    fn test_assign_unknown_without_build_returns_none_or_some() {
        let mut idx = make_index(2);
        idx.insert("v0".into(), vec![1.0f32, 0.0]).expect("insert");
        // Before build: centroids empty → None
        assert!(idx.assign(&[1.0, 0.0]).is_none());
    }

    #[test]
    fn test_cluster_error_display() {
        let e = ClusterError::InvalidK { k: 0, n: 5 };
        assert!(e.to_string().contains("invalid"));
        let e2 = ClusterError::EmptyIndex;
        assert!(e2.to_string().contains("empty"));
        let e3 = ClusterError::DuplicateId("x".into());
        assert!(e3.to_string().contains("x"));
        let e4 = ClusterError::DimMismatch {
            expected: 3,
            got: 2,
        };
        assert!(e4.to_string().contains("mismatch"));
    }

    #[test]
    fn test_search_multi_probe() {
        let idx = insert_and_build(
            3,
            vec![
                ("c1a", vec![0.0f32, 0.0]),
                ("c1b", vec![0.1, 0.0]),
                ("c2a", vec![5.0, 0.0]),
                ("c2b", vec![5.1, 0.0]),
                ("c3a", vec![10.0, 0.0]),
                ("c3b", vec![10.1, 0.0]),
            ],
        );
        // Probe 2 clusters should find at least 2 results
        let results = idx.search(&[0.0, 0.0], 5, 2).expect("search");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_split_then_search() {
        let mut idx = insert_and_build(
            1,
            vec![
                ("a", vec![-3.0f32, 0.0]),
                ("b", vec![-2.0, 0.0]),
                ("c", vec![2.0, 0.0]),
                ("d", vec![3.0, 0.0]),
            ],
        );
        idx.split_largest_cluster().expect("split");
        let results = idx.search(&[-3.0, 0.0], 2, 2).expect("search");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_merge_then_build_consistent() {
        let mut idx = insert_and_build(
            2,
            vec![
                ("a", vec![0.0f32, 0.0]),
                ("b", vec![0.1, 0.0]),
                ("c", vec![10.0, 0.0]),
                ("d", vec![10.1, 0.0]),
            ],
        );
        idx.merge_clusters(0, 1).expect("merge");
        assert_eq!(idx.num_clusters(), 1);
        // After merge, should still be searchable
        let r = idx.search(&[5.0, 0.0], 2, 1).expect("search");
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_num_clusters_zero_before_build() {
        let idx = make_index(3);
        assert_eq!(idx.num_clusters(), 0);
    }

    #[test]
    fn test_build_single_vector() {
        let idx = insert_and_build(1, vec![("solo", vec![1.0f32, 2.0])]);
        assert_eq!(idx.num_clusters(), 1);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_search_returns_correct_id() {
        let idx = insert_and_build(
            1,
            vec![("x_near", vec![0.01f32, 0.0]), ("x_far", vec![100.0, 0.0])],
        );
        let r = idx.search(&[0.0, 0.0], 1, 1).expect("search");
        assert_eq!(r[0].0, "x_near");
    }

    #[test]
    fn test_search_distance_is_non_negative() {
        let idx = insert_and_build(1, vec![("a", vec![1.0f32, 0.0]), ("b", vec![2.0, 0.0])]);
        let r = idx.search(&[0.0, 0.0], 2, 1).expect("search");
        for (_, d) in &r {
            assert!(*d >= 0.0);
        }
    }

    #[test]
    fn test_cluster_stats_centroid_len() {
        let idx = insert_and_build(1, vec![("a", vec![1.0f32, 2.0, 3.0])]);
        let s = idx.cluster_stats(0).expect("stats");
        assert_eq!(s.centroid.len(), 3);
    }

    #[test]
    fn test_merge_closest_with_two_clusters() {
        let mut idx = insert_and_build(
            2,
            vec![
                ("near_a", vec![0.0f32, 0.0]),
                ("near_b", vec![0.1, 0.0]),
                ("far_a", vec![100.0, 0.0]),
                ("far_b", vec![100.1, 0.0]),
            ],
        );
        // Merging the two clusters (they are the only two) should work
        idx.merge_closest_clusters().expect("merge");
        assert_eq!(idx.num_clusters(), 1);
    }

    #[test]
    fn test_cluster_error_unknown_cluster_display() {
        let e = ClusterError::UnknownCluster(42);
        assert!(e.to_string().contains("42"));
    }
}
