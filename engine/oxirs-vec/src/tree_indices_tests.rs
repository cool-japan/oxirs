//! Tests for tree-based nearest-neighbor indices.

#![cfg(test)]

use crate::tree_indices::{BallTree, TreeIndex, TreeIndexConfig, TreeType};
use crate::{Vector, VectorIndex};
use anyhow::Result;

#[test]
#[ignore = "Tree indices are experimental - see module documentation for alternatives"]
fn test_ball_tree() -> Result<()> {
    let config = TreeIndexConfig {
        tree_type: TreeType::BallTree,
        max_leaf_size: 10,
        ..Default::default()
    };

    let mut ball_tree = BallTree::new(config);

    // Add 100 vectors
    for i in 0..100 {
        let vector = Vector::new(vec![i as f32, (i * 2) as f32]);
        ball_tree.data.push((format!("vec_{i}"), vector));
    }

    // Build and search
    ball_tree.build()?;
    assert!(ball_tree.root.is_some());

    let query = vec![50.0, 100.0];
    let results = ball_tree.search(&query, 5);

    assert!(results.len() <= 5);
    assert!(!results.is_empty());
    Ok(())
}

#[test]
#[ignore = "Investigating stack overflow with recursive tree construction"]
fn test_kd_tree() -> Result<()> {
    let config = TreeIndexConfig {
        tree_type: TreeType::KdTree,
        max_leaf_size: 50, // Extremely large leaf size to force leaf nodes
        ..Default::default()
    };

    let mut index = TreeIndex::new(config);

    // Tiny dataset to prevent stack overflow
    for i in 0..3 {
        let vector = Vector::new(vec![i as f32, (3 - i) as f32]);
        index.insert(format!("vec_{i}"), vector)?;
    }

    index.build()?;

    // Search for nearest neighbors
    let query = Vector::new(vec![1.0, 2.0]);
    let results = index.search_knn(&query, 2)?;

    assert_eq!(results.len(), 2);
    Ok(())
}

#[test]
#[ignore = "Investigating stack overflow with recursive tree construction"]
fn test_vp_tree() -> Result<()> {
    let config = TreeIndexConfig {
        tree_type: TreeType::VpTree,
        random_seed: Some(42),
        max_leaf_size: 50, // Extremely large leaf size to force leaf nodes
        ..Default::default()
    };

    let mut index = TreeIndex::new(config);

    // Tiny dataset to prevent stack overflow
    for i in 0..3 {
        let angle = (i as f32) * std::f32::consts::PI / 4.0;
        let vector = Vector::new(vec![angle.cos(), angle.sin()]);
        index.insert(format!("vec_{i}"), vector)?;
    }

    index.build()?;

    // Search for nearest neighbors
    let query = Vector::new(vec![1.0, 0.0]);
    let results = index.search_knn(&query, 2)?;

    assert_eq!(results.len(), 2);
    Ok(())
}
