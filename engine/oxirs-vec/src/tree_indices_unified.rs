//! Unified [`TreeIndex`] dispatcher over all tree-based index implementations.
//!
//! [`TreeIndex`] selects one concrete tree (ball / KD / VP / cover / random
//! projection) based on [`TreeIndexConfig::tree_type`] and exposes a single
//! [`VectorIndex`] interface over it.

use crate::tree_indices_balltree::BallTree;
use crate::tree_indices_covertree::CoverTree;
use crate::tree_indices_kdtree::KdTree;
use crate::tree_indices_rptree::RandomProjectionTree;
use crate::tree_indices_types::{TreeIndexConfig, TreeType};
use crate::tree_indices_vptree::VpTree;
use crate::{Vector, VectorIndex};
use anyhow::Result;

/// Unified tree index interface
pub struct TreeIndex {
    tree_type: TreeType,
    ball_tree: Option<BallTree>,
    kd_tree: Option<KdTree>,
    vp_tree: Option<VpTree>,
    cover_tree: Option<CoverTree>,
    rp_tree: Option<RandomProjectionTree>,
}

impl TreeIndex {
    pub fn new(config: TreeIndexConfig) -> Self {
        let tree_type = config.tree_type;

        let (ball_tree, kd_tree, vp_tree, cover_tree, rp_tree) = match tree_type {
            TreeType::BallTree => (Some(BallTree::new(config)), None, None, None, None),
            TreeType::KdTree => (None, Some(KdTree::new(config)), None, None, None),
            TreeType::VpTree => (None, None, Some(VpTree::new(config)), None, None),
            TreeType::CoverTree => (None, None, None, Some(CoverTree::new(config)), None),
            TreeType::RandomProjectionTree => (
                None,
                None,
                None,
                None,
                Some(RandomProjectionTree::new(config)),
            ),
        };

        Self {
            tree_type,
            ball_tree,
            kd_tree,
            vp_tree,
            cover_tree,
            rp_tree,
        }
    }

    pub fn build(&mut self) -> Result<()> {
        match self.tree_type {
            TreeType::BallTree => self
                .ball_tree
                .as_mut()
                .expect("ball_tree should be initialized for BallTree type")
                .build(),
            TreeType::KdTree => self
                .kd_tree
                .as_mut()
                .expect("kd_tree should be initialized for KdTree type")
                .build(),
            TreeType::VpTree => self
                .vp_tree
                .as_mut()
                .expect("vp_tree should be initialized for VpTree type")
                .build(),
            TreeType::CoverTree => self
                .cover_tree
                .as_mut()
                .expect("cover_tree should be initialized for CoverTree type")
                .build(),
            TreeType::RandomProjectionTree => self
                .rp_tree
                .as_mut()
                .expect("rp_tree should be initialized for RandomProjectionTree type")
                .build(),
        }
    }

    fn search_internal(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        match self.tree_type {
            TreeType::BallTree => self
                .ball_tree
                .as_ref()
                .expect("ball_tree should be initialized for BallTree type")
                .search(query, k),
            TreeType::KdTree => self
                .kd_tree
                .as_ref()
                .expect("kd_tree should be initialized for KdTree type")
                .search(query, k),
            TreeType::VpTree => self
                .vp_tree
                .as_ref()
                .expect("vp_tree should be initialized for VpTree type")
                .search(query, k),
            TreeType::CoverTree => self
                .cover_tree
                .as_ref()
                .expect("cover_tree should be initialized for CoverTree type")
                .search(query, k),
            TreeType::RandomProjectionTree => self
                .rp_tree
                .as_ref()
                .expect("rp_tree should be initialized for RandomProjectionTree type")
                .search(query, k),
        }
    }
}

impl VectorIndex for TreeIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        let data = match self.tree_type {
            TreeType::BallTree => {
                &mut self
                    .ball_tree
                    .as_mut()
                    .expect("ball_tree should be initialized for BallTree type")
                    .data
            }
            TreeType::KdTree => {
                &mut self
                    .kd_tree
                    .as_mut()
                    .expect("kd_tree should be initialized for KdTree type")
                    .data
            }
            TreeType::VpTree => {
                &mut self
                    .vp_tree
                    .as_mut()
                    .expect("vp_tree should be initialized for VpTree type")
                    .data
            }
            TreeType::CoverTree => {
                &mut self
                    .cover_tree
                    .as_mut()
                    .expect("cover_tree should be initialized for CoverTree type")
                    .data
            }
            TreeType::RandomProjectionTree => {
                &mut self
                    .rp_tree
                    .as_mut()
                    .expect("rp_tree should be initialized for RandomProjectionTree type")
                    .data
            }
        };

        data.push((uri, vector));
        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let query_f32 = query.as_f32();
        let results = self.search_internal(&query_f32, k);

        let data = match self.tree_type {
            TreeType::BallTree => {
                &self
                    .ball_tree
                    .as_ref()
                    .expect("ball_tree should be initialized for BallTree type")
                    .data
            }
            TreeType::KdTree => {
                &self
                    .kd_tree
                    .as_ref()
                    .expect("kd_tree should be initialized for KdTree type")
                    .data
            }
            TreeType::VpTree => {
                &self
                    .vp_tree
                    .as_ref()
                    .expect("vp_tree should be initialized for VpTree type")
                    .data
            }
            TreeType::CoverTree => {
                &self
                    .cover_tree
                    .as_ref()
                    .expect("cover_tree should be initialized for CoverTree type")
                    .data
            }
            TreeType::RandomProjectionTree => {
                &self
                    .rp_tree
                    .as_ref()
                    .expect("rp_tree should be initialized for RandomProjectionTree type")
                    .data
            }
        };

        Ok(results
            .into_iter()
            .map(|(idx, dist)| (data[idx].0.clone(), dist))
            .collect())
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        let query_f32 = query.as_f32();
        let all_results = self.search_internal(&query_f32, 1000); // Search more broadly

        let data = match self.tree_type {
            TreeType::BallTree => {
                &self
                    .ball_tree
                    .as_ref()
                    .expect("ball_tree should be initialized for BallTree type")
                    .data
            }
            TreeType::KdTree => {
                &self
                    .kd_tree
                    .as_ref()
                    .expect("kd_tree should be initialized for KdTree type")
                    .data
            }
            TreeType::VpTree => {
                &self
                    .vp_tree
                    .as_ref()
                    .expect("vp_tree should be initialized for VpTree type")
                    .data
            }
            TreeType::CoverTree => {
                &self
                    .cover_tree
                    .as_ref()
                    .expect("cover_tree should be initialized for CoverTree type")
                    .data
            }
            TreeType::RandomProjectionTree => {
                &self
                    .rp_tree
                    .as_ref()
                    .expect("rp_tree should be initialized for RandomProjectionTree type")
                    .data
            }
        };

        Ok(all_results
            .into_iter()
            .filter(|(_, dist)| *dist <= threshold)
            .map(|(idx, dist)| (data[idx].0.clone(), dist))
            .collect())
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        let data = match self.tree_type {
            TreeType::BallTree => {
                &self
                    .ball_tree
                    .as_ref()
                    .expect("ball_tree should be initialized for BallTree type")
                    .data
            }
            TreeType::KdTree => {
                &self
                    .kd_tree
                    .as_ref()
                    .expect("kd_tree should be initialized for KdTree type")
                    .data
            }
            TreeType::VpTree => {
                &self
                    .vp_tree
                    .as_ref()
                    .expect("vp_tree should be initialized for VpTree type")
                    .data
            }
            TreeType::CoverTree => {
                &self
                    .cover_tree
                    .as_ref()
                    .expect("cover_tree should be initialized for CoverTree type")
                    .data
            }
            TreeType::RandomProjectionTree => {
                &self
                    .rp_tree
                    .as_ref()
                    .expect("rp_tree should be initialized for RandomProjectionTree type")
                    .data
            }
        };

        data.iter().find(|(u, _)| u == uri).map(|(_, v)| v)
    }
}

// Add rand to dependencies for VP-Tree and Random Projection Tree
// Note: Replaced with scirs2_core::random

// Placeholder for async task spawning - integrate with oxirs-core::parallel
async fn spawn_task<F, T>(f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    // In practice, this would use oxirs-core::parallel's task spawning
    f()
}
