//! Cover Tree implementation for nearest-neighbor search.
//!
//! A cover tree (navigating net) organizes points into levels with provable
//! bounds on search performance.

use crate::tree_indices_types::TreeIndexConfig;
use crate::Vector;
use anyhow::Result;
use std::cmp::Ordering;

/// Cover Tree implementation
pub struct CoverTree {
    pub(crate) root: Option<Box<CoverNode>>,
    pub(crate) data: Vec<(String, Vector)>,
    pub(crate) config: TreeIndexConfig,
    base: f32,
}

pub(crate) struct CoverNode {
    /// Point index
    point: usize,
    /// Level in the tree
    level: i32,
    /// Children at the same or lower level
    #[allow(clippy::vec_box)] // Box is necessary for recursive structure
    children: Vec<Box<CoverNode>>,
}

impl CoverTree {
    pub fn new(config: TreeIndexConfig) -> Self {
        Self {
            root: None,
            data: Vec::new(),
            config,
            base: 2.0, // Base for the covering constant
        }
    }

    pub fn build(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }

        // Initialize with first point
        self.root = Some(Box::new(CoverNode {
            point: 0,
            level: self.get_level(0),
            children: Vec::new(),
        }));

        // Insert remaining points
        for idx in 1..self.data.len() {
            self.insert(idx)?;
        }

        Ok(())
    }

    fn get_level(&self, _point_idx: usize) -> i32 {
        // Simple heuristic for initial level
        ((self.data.len() as f32).log2() as i32).max(0)
    }

    fn insert(&mut self, point_idx: usize) -> Result<()> {
        // Simplified insert - in practice, this would be more complex
        // to maintain the cover tree invariants
        let level = self.get_level(point_idx);
        if let Some(root) = &mut self.root {
            root.children.push(Box::new(CoverNode {
                point: point_idx,
                level,
                children: Vec::new(),
            }));
        }
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.root.is_none() {
            return Vec::new();
        }

        let mut results = Vec::new();
        self.search_node(
            self.root
                .as_ref()
                .expect("tree should have root after build"),
            query,
            k,
            &mut results,
        );

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results
    }

    #[allow(clippy::only_used_in_recursion)]
    fn search_node(
        &self,
        node: &CoverNode,
        query: &[f32],
        k: usize,
        results: &mut Vec<(usize, f32)>,
    ) {
        // Prevent excessive recursion depth
        if results.len() >= k * 10 {
            return;
        }

        let point_data = &self.data[node.point].1.as_f32();
        let dist = self.config.distance_metric.distance(query, point_data);

        results.push((node.point, dist));

        // Search children
        for child in &node.children {
            self.search_node(child, query, k, results);
        }
    }
}
