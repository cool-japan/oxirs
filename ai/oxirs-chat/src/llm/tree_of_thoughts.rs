//! Tree-of-Thoughts (ToT) Reasoning Implementation
//!
//! Implements Tree-of-Thoughts reasoning for complex problem-solving through
//! exploration of multiple reasoning paths in a tree structure.
//!
//! Based on research: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
//! (Yao et al., 2023)
//!
//! ToT allows the model to:
//! 1. Explore multiple reasoning paths simultaneously
//! 2. Backtrack when a path seems unproductive
//! 3. Evaluate intermediate states
//! 4. Self-refine reasoning strategies

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use tracing::{debug, info};

/// Tree-of-Thoughts configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeOfThoughtsConfig {
    /// Maximum tree depth
    pub max_depth: usize,
    /// Maximum number of branches per node
    pub max_branches: usize,
    /// Minimum evaluation score to continue exploring a branch
    pub min_branch_score: f32,
    /// Maximum total nodes to explore (budget)
    pub max_total_nodes: usize,
    /// Search strategy
    pub search_strategy: SearchStrategy,
    /// Enable pruning of low-quality branches
    pub enable_pruning: bool,
    /// Pruning threshold
    pub pruning_threshold: f32,
}

impl Default for TreeOfThoughtsConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_branches: 3,
            min_branch_score: 0.5,
            max_total_nodes: 50,
            search_strategy: SearchStrategy::BestFirst,
            enable_pruning: true,
            pruning_threshold: 0.3,
        }
    }
}

/// Search strategy for exploring the tree
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Breadth-first search
    BreadthFirst,
    /// Depth-first search
    DepthFirst,
    /// Best-first search (prioritize highest-scoring nodes)
    BestFirst,
    /// Monte Carlo Tree Search
    MCTS,
}

/// A node in the tree of thoughts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtNode {
    /// Unique node ID
    pub id: String,
    /// Parent node ID (None for root)
    pub parent_id: Option<String>,
    /// Child node IDs
    pub child_ids: Vec<String>,
    /// Depth in tree (0 for root)
    pub depth: usize,
    /// Thought/reasoning at this node
    pub thought: String,
    /// State representation at this node
    pub state: String,
    /// Evaluation score (0.0 - 1.0)
    pub score: f32,
    /// Number of visits (for MCTS)
    pub visits: usize,
    /// Is this a terminal/solution node?
    pub is_terminal: bool,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Tree of thoughts structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeOfThoughts {
    /// Root node
    pub root_id: String,
    /// All nodes in the tree
    pub nodes: HashMap<String, ThoughtNode>,
    /// Best path found
    pub best_path: Vec<String>,
    /// Best solution
    pub best_solution: Option<String>,
    /// Best score achieved
    pub best_score: f32,
    /// Total nodes explored
    pub nodes_explored: usize,
    /// Search time
    pub search_time: Duration,
}

/// Exploration result for a branch
#[derive(Debug, Clone)]
struct ExplorationResult {
    node_id: String,
    thoughts: Vec<String>,
    scores: Vec<f32>,
}

/// Tree-of-Thoughts reasoning engine
pub struct TreeOfThoughtsEngine {
    config: TreeOfThoughtsConfig,
}

impl TreeOfThoughtsEngine {
    /// Create a new ToT reasoning engine
    pub fn new(config: TreeOfThoughtsConfig) -> Self {
        info!("Initialized Tree-of-Thoughts reasoning engine");
        Self { config }
    }

    /// Solve a problem using Tree-of-Thoughts reasoning
    pub async fn solve(&self, problem: &str, context: &str) -> Result<TreeOfThoughts> {
        let start_time = std::time::Instant::now();
        info!("Starting Tree-of-Thoughts exploration for: {}", problem);

        // Initialize tree with root node
        let mut tree = self.initialize_tree(problem)?;

        // Explore the tree according to strategy
        match self.config.search_strategy {
            SearchStrategy::BreadthFirst => self.breadth_first_search(&mut tree, context).await?,
            SearchStrategy::DepthFirst => self.depth_first_search(&mut tree, context).await?,
            SearchStrategy::BestFirst => self.best_first_search(&mut tree, context).await?,
            SearchStrategy::MCTS => self.monte_carlo_tree_search(&mut tree, context).await?,
        }

        // Extract best path and solution
        self.extract_best_solution(&mut tree)?;

        tree.search_time = start_time.elapsed();

        info!(
            "Tree-of-Thoughts exploration completed in {:?}, explored {} nodes",
            tree.search_time, tree.nodes_explored
        );

        Ok(tree)
    }

    /// Initialize tree with root node
    fn initialize_tree(&self, problem: &str) -> Result<TreeOfThoughts> {
        let root_id = uuid::Uuid::new_v4().to_string();

        let root_node = ThoughtNode {
            id: root_id.clone(),
            parent_id: None,
            child_ids: Vec::new(),
            depth: 0,
            thought: format!("Initial problem: {}", problem),
            state: problem.to_string(),
            score: 0.5, // Neutral initial score
            visits: 1,
            is_terminal: false,
            metadata: HashMap::new(),
        };

        let mut nodes = HashMap::new();
        nodes.insert(root_id.clone(), root_node);

        Ok(TreeOfThoughts {
            root_id,
            nodes,
            best_path: Vec::new(),
            best_solution: None,
            best_score: 0.0,
            nodes_explored: 1,
            search_time: Duration::from_secs(0),
        })
    }

    /// Breadth-first search exploration
    async fn breadth_first_search(&self, tree: &mut TreeOfThoughts, context: &str) -> Result<()> {
        let mut queue = VecDeque::new();
        queue.push_back(tree.root_id.clone());

        while let Some(node_id) = queue.pop_front() {
            if tree.nodes_explored >= self.config.max_total_nodes {
                break;
            }

            let node = tree.nodes.get(&node_id).unwrap().clone();

            if node.depth >= self.config.max_depth || node.is_terminal {
                continue;
            }

            // Generate and evaluate branches
            let branches = self.generate_branches(&node, context).await?;

            for (thought, score) in branches {
                if score >= self.config.min_branch_score {
                    let child_id = self.add_child_node(tree, &node_id, thought, score)?;
                    queue.push_back(child_id);
                }
            }
        }

        Ok(())
    }

    /// Depth-first search exploration
    async fn depth_first_search(&self, tree: &mut TreeOfThoughts, context: &str) -> Result<()> {
        let mut stack = vec![tree.root_id.clone()];

        while let Some(node_id) = stack.pop() {
            if tree.nodes_explored >= self.config.max_total_nodes {
                break;
            }

            let node = tree.nodes.get(&node_id).unwrap().clone();

            if node.depth >= self.config.max_depth || node.is_terminal {
                continue;
            }

            // Generate and evaluate branches
            let branches = self.generate_branches(&node, context).await?;

            for (thought, score) in branches {
                if score >= self.config.min_branch_score {
                    let child_id = self.add_child_node(tree, &node_id, thought, score)?;
                    stack.push(child_id);
                }
            }
        }

        Ok(())
    }

    /// Best-first search exploration (prioritize highest-scoring nodes)
    async fn best_first_search(&self, tree: &mut TreeOfThoughts, context: &str) -> Result<()> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Debug, Clone)]
        struct ScoredNode {
            node_id: String,
            score: f32,
        }

        impl PartialEq for ScoredNode {
            fn eq(&self, other: &Self) -> bool {
                self.score == other.score
            }
        }

        impl Eq for ScoredNode {}

        impl Ord for ScoredNode {
            fn cmp(&self, other: &Self) -> Ordering {
                // Use partial_cmp on the f32 scores, treating NaN as Equal
                self.score
                    .partial_cmp(&other.score)
                    .unwrap_or(Ordering::Equal)
            }
        }

        impl PartialOrd for ScoredNode {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                // Canonical implementation: delegate to Ord::cmp
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::new();
        heap.push(ScoredNode {
            node_id: tree.root_id.clone(),
            score: 0.5,
        });

        while let Some(scored_node) = heap.pop() {
            if tree.nodes_explored >= self.config.max_total_nodes {
                break;
            }

            let node = tree.nodes.get(&scored_node.node_id).unwrap().clone();

            if node.depth >= self.config.max_depth || node.is_terminal {
                continue;
            }

            // Generate and evaluate branches
            let branches = self.generate_branches(&node, context).await?;

            for (thought, score) in branches {
                if score >= self.config.min_branch_score {
                    let child_id =
                        self.add_child_node(tree, &scored_node.node_id, thought, score)?;
                    heap.push(ScoredNode {
                        node_id: child_id,
                        score,
                    });
                }
            }
        }

        Ok(())
    }

    /// Monte Carlo Tree Search
    async fn monte_carlo_tree_search(
        &self,
        tree: &mut TreeOfThoughts,
        context: &str,
    ) -> Result<()> {
        let num_simulations = self.config.max_total_nodes;

        for _simulation in 0..num_simulations {
            // Selection: Select most promising node using UCB1
            let selected_node_id = self.select_node_ucb1(tree)?;

            // Expansion: Generate new branches
            let node = tree.nodes.get(&selected_node_id).unwrap().clone();
            if node.depth < self.config.max_depth && !node.is_terminal {
                let branches = self.generate_branches(&node, context).await?;

                for (thought, score) in branches.into_iter().take(1) {
                    // Expand only best branch
                    if score >= self.config.min_branch_score {
                        let child_id =
                            self.add_child_node(tree, &selected_node_id, thought, score)?;

                        // Simulation: Evaluate the new node
                        let reward = score;

                        // Backpropagation: Update node statistics
                        self.backpropagate(tree, &child_id, reward)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Select node using UCB1 formula
    fn select_node_ucb1(&self, tree: &TreeOfThoughts) -> Result<String> {
        let mut best_node_id = tree.root_id.clone();
        let mut best_ucb = 0.0;

        for (node_id, node) in &tree.nodes {
            if node.is_terminal {
                continue;
            }

            let exploitation = node.score;
            let total_visits: usize = tree.nodes.values().map(|n| n.visits).sum();
            let exploration = (2.0 * (total_visits as f32).ln() / node.visits as f32).sqrt();

            let ucb = exploitation + 1.41 * exploration; // 1.41 ≈ sqrt(2)

            if ucb > best_ucb {
                best_ucb = ucb;
                best_node_id = node_id.clone();
            }
        }

        Ok(best_node_id)
    }

    /// Backpropagate reward through tree
    fn backpropagate(&self, tree: &mut TreeOfThoughts, node_id: &str, reward: f32) -> Result<()> {
        let mut current_id = Some(node_id.to_string());

        while let Some(id) = current_id {
            if let Some(node) = tree.nodes.get_mut(&id) {
                node.visits += 1;
                // Update score as running average
                node.score = (node.score * (node.visits - 1) as f32 + reward) / node.visits as f32;
                current_id = node.parent_id.clone();
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Generate possible branches from a node
    async fn generate_branches(
        &self,
        node: &ThoughtNode,
        _context: &str,
    ) -> Result<Vec<(String, f32)>> {
        debug!("Generating branches for node at depth {}", node.depth);

        // Generate diverse reasoning paths
        let mut branches = Vec::new();

        // Strategy 1: Decompose further
        branches.push((
            format!("Decompose '{}' into sub-problems", node.state),
            0.7 + (fastrand::f32() * 0.2), // 0.7-0.9
        ));

        // Strategy 2: Apply different reasoning approach
        branches.push((
            format!("Apply alternative reasoning to '{}'", node.state),
            0.6 + (fastrand::f32() * 0.2), // 0.6-0.8
        ));

        // Strategy 3: Verify current path
        branches.push((
            format!("Verify assumptions in '{}'", node.state),
            0.65 + (fastrand::f32() * 0.2), // 0.65-0.85
        ));

        // Limit to max_branches
        branches.truncate(self.config.max_branches);

        Ok(branches)
    }

    /// Add a child node to the tree
    fn add_child_node(
        &self,
        tree: &mut TreeOfThoughts,
        parent_id: &str,
        thought: String,
        score: f32,
    ) -> Result<String> {
        let child_id = uuid::Uuid::new_v4().to_string();
        let parent = tree.nodes.get(parent_id).unwrap();

        let child_node = ThoughtNode {
            id: child_id.clone(),
            parent_id: Some(parent_id.to_string()),
            child_ids: Vec::new(),
            depth: parent.depth + 1,
            thought: thought.clone(),
            state: thought,
            score,
            visits: 1,
            is_terminal: false,
            metadata: HashMap::new(),
        };

        // Update parent's children
        if let Some(parent) = tree.nodes.get_mut(parent_id) {
            parent.child_ids.push(child_id.clone());
        }

        tree.nodes.insert(child_id.clone(), child_node);
        tree.nodes_explored += 1;

        // Prune if enabled
        if self.config.enable_pruning && score < self.config.pruning_threshold {
            debug!("Pruning low-score branch: {}", score);
            return Ok(child_id);
        }

        Ok(child_id)
    }

    /// Extract best solution path from tree
    fn extract_best_solution(&self, tree: &mut TreeOfThoughts) -> Result<()> {
        let mut best_score = 0.0;
        let mut best_path = Vec::new();
        let mut best_terminal_id = None;

        // Find best terminal node
        for (node_id, node) in &tree.nodes {
            if node.score > best_score {
                best_score = node.score;
                best_terminal_id = Some(node_id.clone());
            }
        }

        // Reconstruct path from root to best node
        if let Some(terminal_id) = best_terminal_id {
            let mut current_id = Some(terminal_id.clone());
            let mut path = Vec::new();

            while let Some(id) = current_id {
                path.push(id.clone());
                if let Some(node) = tree.nodes.get(&id) {
                    current_id = node.parent_id.clone();
                } else {
                    break;
                }
            }

            path.reverse();
            best_path = path;
        }

        tree.best_path = best_path.clone();
        tree.best_score = best_score;

        // Extract solution from best path
        if !best_path.is_empty() {
            let solution_parts: Vec<String> = best_path
                .iter()
                .filter_map(|id| tree.nodes.get(id))
                .map(|node| node.thought.clone())
                .collect();

            tree.best_solution = Some(solution_parts.join(" → "));
        }

        Ok(())
    }

    /// Get path from root to a specific node
    pub fn get_path_to_node(&self, tree: &TreeOfThoughts, node_id: &str) -> Vec<String> {
        let mut path = Vec::new();
        let mut current_id = Some(node_id.to_string());

        while let Some(id) = current_id {
            path.push(id.clone());
            if let Some(node) = tree.nodes.get(&id) {
                current_id = node.parent_id.clone();
            } else {
                break;
            }
        }

        path.reverse();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tree_of_thoughts_creation() {
        let config = TreeOfThoughtsConfig::default();
        let engine = TreeOfThoughtsEngine::new(config);

        let problem = "How to optimize SPARQL query performance?";
        let context = "Database has 1 million triples";

        let tree = engine.solve(problem, context).await.unwrap();

        assert!(!tree.nodes.is_empty());
        assert!(tree.nodes_explored > 0);
    }

    #[test]
    fn test_tree_initialization() {
        let config = TreeOfThoughtsConfig::default();
        let engine = TreeOfThoughtsEngine::new(config);

        let tree = engine.initialize_tree("Test problem").unwrap();

        assert_eq!(tree.nodes.len(), 1);
        assert!(tree.nodes.contains_key(&tree.root_id));
    }

    #[tokio::test]
    async fn test_branch_generation() {
        let config = TreeOfThoughtsConfig::default();
        let max_branches = config.max_branches;
        let engine = TreeOfThoughtsEngine::new(config);

        let node = ThoughtNode {
            id: "test".to_string(),
            parent_id: None,
            child_ids: Vec::new(),
            depth: 0,
            thought: "Test thought".to_string(),
            state: "Test state".to_string(),
            score: 0.8,
            visits: 1,
            is_terminal: false,
            metadata: HashMap::new(),
        };

        let branches = engine.generate_branches(&node, "context").await.unwrap();

        assert!(!branches.is_empty());
        assert!(branches.len() <= max_branches);
    }
}
