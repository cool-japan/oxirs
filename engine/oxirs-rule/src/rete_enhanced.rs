//! Enhanced RETE Pattern Matching Network
//!
//! This module provides a full-featured RETE implementation with:
//! - Complete beta join support with complex variable bindings
//! - Advanced memory management strategies
//! - Conflict resolution mechanisms
//! - Truth maintenance and fact retraction
//! - Builtin predicate evaluation

use crate::forward::Substitution;
use crate::{Rule, RuleAtom, Term};
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet, VecDeque, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, trace, warn};

/// Enhanced token with timestamp and priority
#[derive(Debug, Clone)]
pub struct EnhancedToken {
    /// Current variable bindings
    pub bindings: Substitution,
    /// Facts that contributed to this token
    pub facts: Vec<RuleAtom>,
    /// Creation timestamp
    pub timestamp: Instant,
    /// Priority for conflict resolution
    pub priority: i32,
    /// Rule specificity (number of conditions)
    pub specificity: usize,
    /// Justification (which rules/facts led to this token)
    pub justification: Vec<String>,
}

impl EnhancedToken {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            facts: Vec::new(),
            timestamp: Instant::now(),
            priority: 0,
            specificity: 0,
            justification: Vec::new(),
        }
    }

    pub fn with_fact(fact: RuleAtom) -> Self {
        let mut token = Self::new();
        token.facts.push(fact);
        token
    }

    /// Merge two tokens with proper conflict resolution
    pub fn merge(left: &Self, right: &Self) -> Result<Self> {
        let mut merged = Self::new();
        
        // Merge bindings with conflict detection
        merged.bindings.extend(left.bindings.clone());
        for (var, value) in &right.bindings {
            if let Some(existing) = merged.bindings.get(var) {
                if !terms_compatible(existing, value) {
                    return Err(anyhow!("Binding conflict for variable {}: {:?} vs {:?}", 
                                       var, existing, value));
                }
            } else {
                merged.bindings.insert(var.clone(), value.clone());
            }
        }
        
        // Combine facts
        merged.facts.extend(left.facts.clone());
        merged.facts.extend(right.facts.clone());
        
        // Use earliest timestamp
        merged.timestamp = left.timestamp.min(right.timestamp);
        
        // Combine priorities and specificity
        merged.priority = left.priority.max(right.priority);
        merged.specificity = left.specificity + right.specificity;
        
        // Merge justifications
        merged.justification.extend(left.justification.clone());
        merged.justification.extend(right.justification.clone());
        
        Ok(merged)
    }
}

/// Memory management strategies
#[derive(Debug, Clone, Copy)]
pub enum MemoryStrategy {
    /// Keep all tokens
    Unlimited,
    /// Limit by count
    LimitCount(usize),
    /// Limit by age
    LimitAge(Duration),
    /// Least Recently Used
    LRU(usize),
    /// Adaptive based on memory pressure
    Adaptive,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Copy)]
pub enum ConflictResolution {
    /// First match wins
    First,
    /// Most recent wins
    Recency,
    /// Most specific rule wins
    Specificity,
    /// Highest priority wins
    Priority,
    /// Combined scoring
    Combined,
}

/// Enhanced beta memory with advanced features
#[derive(Debug)]
pub struct BetaMemory {
    /// Left side tokens
    left_tokens: VecDeque<EnhancedToken>,
    /// Right side tokens
    right_tokens: VecDeque<EnhancedToken>,
    /// Index for fast lookup by variable values
    left_index: HashMap<String, HashMap<Term, Vec<usize>>>,
    right_index: HashMap<String, HashMap<Term, Vec<usize>>>,
    /// Memory strategy
    memory_strategy: MemoryStrategy,
    /// Access timestamps for LRU
    access_times: HashMap<usize, Instant>,
    /// Memory statistics
    stats: MemoryStats,
}

#[derive(Debug, Default)]
pub struct MemoryStats {
    pub total_joins: usize,
    pub successful_joins: usize,
    pub evictions: usize,
    pub peak_size: usize,
}

impl BetaMemory {
    pub fn new(strategy: MemoryStrategy) -> Self {
        Self {
            left_tokens: VecDeque::new(),
            right_tokens: VecDeque::new(),
            left_index: HashMap::new(),
            right_index: HashMap::new(),
            memory_strategy: strategy,
            access_times: HashMap::new(),
            stats: MemoryStats::default(),
        }
    }

    /// Add a token to the left side
    pub fn add_left(&mut self, token: EnhancedToken, join_vars: &[String]) -> usize {
        let idx = self.left_tokens.len();
        self.left_tokens.push_back(token.clone());
        
        // Update indices for join variables
        for var in join_vars {
            if let Some(value) = token.bindings.get(var) {
                self.left_index
                    .entry(var.clone())
                    .or_default()
                    .entry(value.clone())
                    .or_default()
                    .push(idx);
            }
        }
        
        self.access_times.insert(idx, Instant::now());
        self.apply_memory_management();
        self.stats.peak_size = self.stats.peak_size.max(self.left_tokens.len());
        
        idx
    }

    /// Add a token to the right side
    pub fn add_right(&mut self, token: EnhancedToken, join_vars: &[String]) -> usize {
        let idx = self.right_tokens.len();
        self.right_tokens.push_back(token.clone());
        
        // Update indices for join variables
        for var in join_vars {
            if let Some(value) = token.bindings.get(var) {
                self.right_index
                    .entry(var.clone())
                    .or_default()
                    .entry(value.clone())
                    .or_default()
                    .push(idx);
            }
        }
        
        self.access_times.insert(idx + 1000000, Instant::now()); // Offset for right side
        self.apply_memory_management();
        self.stats.peak_size = self.stats.peak_size.max(self.right_tokens.len());
        
        idx
    }

    /// Find matching tokens using indices
    pub fn find_matches_indexed(
        &mut self,
        token: &EnhancedToken,
        is_left: bool,
        join_vars: &[String],
    ) -> Vec<EnhancedToken> {
        self.stats.total_joins += 1;
        let mut matches = Vec::new();
        
        if join_vars.is_empty() {
            // Cartesian product if no join variables
            let tokens = if is_left { &self.right_tokens } else { &self.left_tokens };
            matches.extend(tokens.iter().cloned());
        } else {
            // Use indices for efficient lookup
            let indices = if is_left { &self.right_index } else { &self.left_index };
            let mut candidate_indices = HashSet::new();
            
            for var in join_vars {
                if let Some(value) = token.bindings.get(var) {
                    if let Some(var_index) = indices.get(var) {
                        if let Some(token_indices) = var_index.get(value) {
                            if candidate_indices.is_empty() {
                                candidate_indices.extend(token_indices);
                            } else {
                                // Intersection with existing candidates
                                candidate_indices.retain(|idx| token_indices.contains(idx));
                            }
                        }
                    }
                }
            }
            
            // Retrieve matching tokens
            let tokens = if is_left { &self.right_tokens } else { &self.left_tokens };
            for &idx in &candidate_indices {
                if let Some(match_token) = tokens.get(idx) {
                    matches.push(match_token.clone());
                    
                    // Update access time for LRU
                    let access_key = if is_left { idx + 1000000 } else { idx };
                    self.access_times.insert(access_key, Instant::now());
                }
            }
        }
        
        if !matches.is_empty() {
            self.stats.successful_joins += 1;
        }
        
        matches
    }

    /// Apply memory management strategy
    fn apply_memory_management(&mut self) {
        match self.memory_strategy {
            MemoryStrategy::LimitCount(max_count) => {
                while self.left_tokens.len() + self.right_tokens.len() > max_count {
                    self.evict_oldest();
                }
            }
            MemoryStrategy::LimitAge(max_age) => {
                let now = Instant::now();
                self.left_tokens.retain(|token| now.duration_since(token.timestamp) < max_age);
                self.right_tokens.retain(|token| now.duration_since(token.timestamp) < max_age);
                self.rebuild_indices();
            }
            MemoryStrategy::LRU(max_count) => {
                while self.left_tokens.len() + self.right_tokens.len() > max_count {
                    self.evict_lru();
                }
            }
            MemoryStrategy::Adaptive => {
                // Simple adaptive strategy based on join success rate
                let success_rate = if self.stats.total_joins > 0 {
                    self.stats.successful_joins as f64 / self.stats.total_joins as f64
                } else {
                    1.0
                };
                
                // If success rate is low, be more aggressive with eviction
                if success_rate < 0.1 && self.left_tokens.len() + self.right_tokens.len() > 1000 {
                    self.evict_oldest();
                }
            }
            MemoryStrategy::Unlimited => {}
        }
    }

    /// Evict oldest token
    fn evict_oldest(&mut self) {
        if self.left_tokens.len() > self.right_tokens.len() {
            self.left_tokens.pop_front();
        } else if !self.right_tokens.is_empty() {
            self.right_tokens.pop_front();
        }
        self.stats.evictions += 1;
        self.rebuild_indices();
    }

    /// Evict least recently used token
    fn evict_lru(&mut self) {
        if let Some((&oldest_key, _)) = self.access_times.iter()
            .min_by_key(|(_, &time)| time) {
            
            if oldest_key < 1000000 {
                // Left side
                if oldest_key < self.left_tokens.len() {
                    self.left_tokens.remove(oldest_key);
                }
            } else {
                // Right side
                let idx = oldest_key - 1000000;
                if idx < self.right_tokens.len() {
                    self.right_tokens.remove(idx);
                }
            }
            
            self.stats.evictions += 1;
            self.rebuild_indices();
        }
    }

    /// Rebuild indices after eviction
    fn rebuild_indices(&mut self) {
        self.left_index.clear();
        self.right_index.clear();
        self.access_times.clear();
        
        // Rebuild left index
        for (idx, token) in self.left_tokens.iter().enumerate() {
            for (var, value) in &token.bindings {
                self.left_index
                    .entry(var.clone())
                    .or_default()
                    .entry(value.clone())
                    .or_default()
                    .push(idx);
            }
            self.access_times.insert(idx, Instant::now());
        }
        
        // Rebuild right index
        for (idx, token) in self.right_tokens.iter().enumerate() {
            for (var, value) in &token.bindings {
                self.right_index
                    .entry(var.clone())
                    .or_default()
                    .entry(value.clone())
                    .or_default()
                    .push(idx);
            }
            self.access_times.insert(idx + 1000000, Instant::now());
        }
    }
}

/// Enhanced beta join node
#[derive(Debug)]
pub struct BetaJoinNode {
    /// Node ID
    pub id: usize,
    /// Left parent node ID
    pub left_parent: usize,
    /// Right parent node ID
    pub right_parent: usize,
    /// Variables to join on
    pub join_variables: Vec<String>,
    /// Additional join conditions
    pub conditions: Vec<JoinCondition>,
    /// Beta memory
    pub memory: BetaMemory,
    /// Children nodes
    pub children: Vec<usize>,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolution,
}

/// Join condition types
#[derive(Debug, Clone)]
pub enum JoinCondition {
    /// Variable equality (already handled by join_variables)
    VarEquality { left_var: String, right_var: String },
    /// Comparison between variables
    VarComparison {
        left_var: String,
        right_var: String,
        op: ComparisonOp,
    },
    /// Builtin predicate
    Builtin {
        predicate: String,
        args: Vec<JoinArg>,
    },
    /// Negation
    Not(Box<JoinCondition>),
    /// Conjunction
    And(Vec<JoinCondition>),
    /// Disjunction
    Or(Vec<JoinCondition>),
}

#[derive(Debug, Clone)]
pub enum JoinArg {
    LeftVar(String),
    RightVar(String),
    Constant(Term),
}

#[derive(Debug, Clone, Copy)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

impl BetaJoinNode {
    pub fn new(
        id: usize,
        left_parent: usize,
        right_parent: usize,
        memory_strategy: MemoryStrategy,
        conflict_resolution: ConflictResolution,
    ) -> Self {
        Self {
            id,
            left_parent,
            right_parent,
            join_variables: Vec::new(),
            conditions: Vec::new(),
            memory: BetaMemory::new(memory_strategy),
            children: Vec::new(),
            conflict_resolution,
        }
    }

    /// Perform the join operation
    pub fn join(
        &mut self,
        token: EnhancedToken,
        from_left: bool,
    ) -> Result<Vec<EnhancedToken>> {
        let mut results = Vec::new();
        
        if from_left {
            // Add to left memory
            self.memory.add_left(token.clone(), &self.join_variables);
            
            // Find matches in right memory
            let matches = self.memory.find_matches_indexed(&token, true, &self.join_variables);
            
            for right_token in matches {
                if let Ok(joined) = self.try_join(&token, &right_token) {
                    results.push(joined);
                }
            }
        } else {
            // Add to right memory
            self.memory.add_right(token.clone(), &self.join_variables);
            
            // Find matches in left memory
            let matches = self.memory.find_matches_indexed(&token, false, &self.join_variables);
            
            for left_token in matches {
                if let Ok(joined) = self.try_join(&left_token, &token) {
                    results.push(joined);
                }
            }
        }
        
        // Apply conflict resolution if multiple results
        if results.len() > 1 {
            results = self.apply_conflict_resolution(results);
        }
        
        Ok(results)
    }

    /// Try to join two tokens
    fn try_join(
        &self,
        left: &EnhancedToken,
        right: &EnhancedToken,
    ) -> Result<EnhancedToken> {
        // First check join variables match
        for var in &self.join_variables {
            if let (Some(left_val), Some(right_val)) = (left.bindings.get(var), right.bindings.get(var)) {
                if !terms_compatible(left_val, right_val) {
                    return Err(anyhow!("Join variable {} doesn't match", var));
                }
            }
        }
        
        // Then check additional conditions
        for condition in &self.conditions {
            if !self.evaluate_condition(condition, left, right)? {
                return Err(anyhow!("Join condition failed"));
            }
        }
        
        // Merge tokens
        EnhancedToken::merge(left, right)
    }

    /// Evaluate a join condition
    fn evaluate_condition(
        &self,
        condition: &JoinCondition,
        left: &EnhancedToken,
        right: &EnhancedToken,
    ) -> Result<bool> {
        match condition {
            JoinCondition::VarEquality { left_var, right_var } => {
                let left_val = left.bindings.get(left_var);
                let right_val = right.bindings.get(right_var);
                Ok(match (left_val, right_val) {
                    (Some(lv), Some(rv)) => terms_compatible(lv, rv),
                    _ => false,
                })
            }
            JoinCondition::VarComparison { left_var, right_var, op } => {
                let left_val = left.bindings.get(left_var);
                let right_val = right.bindings.get(right_var);
                Ok(match (left_val, right_val) {
                    (Some(lv), Some(rv)) => evaluate_comparison(lv, rv, *op)?,
                    _ => false,
                })
            }
            JoinCondition::Builtin { predicate, args } => {
                evaluate_builtin(predicate, args, left, right)
            }
            JoinCondition::Not(cond) => {
                Ok(!self.evaluate_condition(cond, left, right)?)
            }
            JoinCondition::And(conds) => {
                for cond in conds {
                    if !self.evaluate_condition(cond, left, right)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            JoinCondition::Or(conds) => {
                for cond in conds {
                    if self.evaluate_condition(cond, left, right)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }

    /// Apply conflict resolution strategy
    fn apply_conflict_resolution(&self, mut tokens: Vec<EnhancedToken>) -> Vec<EnhancedToken> {
        match self.conflict_resolution {
            ConflictResolution::First => {
                tokens.truncate(1);
                tokens
            }
            ConflictResolution::Recency => {
                tokens.sort_by_key(|t| std::cmp::Reverse(t.timestamp));
                tokens.truncate(1);
                tokens
            }
            ConflictResolution::Specificity => {
                tokens.sort_by_key(|t| std::cmp::Reverse(t.specificity));
                tokens.truncate(1);
                tokens
            }
            ConflictResolution::Priority => {
                tokens.sort_by_key(|t| std::cmp::Reverse(t.priority));
                tokens.truncate(1);
                tokens
            }
            ConflictResolution::Combined => {
                // Combined scoring: priority * 1000 + specificity * 10 + recency
                tokens.sort_by_key(|t| {
                    let recency_score = t.timestamp.elapsed().as_secs() as i32;
                    std::cmp::Reverse(t.priority * 1000 + t.specificity as i32 * 10 - recency_score)
                });
                tokens.truncate(1);
                tokens
            }
        }
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> &MemoryStats {
        &self.memory.stats
    }
}

/// Check if two terms are compatible (can unify)
fn terms_compatible(t1: &Term, t2: &Term) -> bool {
    match (t1, t2) {
        (Term::Variable(_), _) | (_, Term::Variable(_)) => true,
        (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
        (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
        _ => false,
    }
}

/// Evaluate a comparison between terms
fn evaluate_comparison(left: &Term, right: &Term, op: ComparisonOp) -> Result<bool> {
    // Try to parse as numbers for numeric comparison
    let left_num = parse_numeric(left);
    let right_num = parse_numeric(right);
    
    match (left_num, right_num) {
        (Some(l), Some(r)) => Ok(match op {
            ComparisonOp::Equal => (l - r).abs() < f64::EPSILON,
            ComparisonOp::NotEqual => (l - r).abs() >= f64::EPSILON,
            ComparisonOp::Less => l < r,
            ComparisonOp::LessEqual => l <= r,
            ComparisonOp::Greater => l > r,
            ComparisonOp::GreaterEqual => l >= r,
        }),
        _ => {
            // String comparison
            let left_str = term_to_string(left);
            let right_str = term_to_string(right);
            Ok(match op {
                ComparisonOp::Equal => left_str == right_str,
                ComparisonOp::NotEqual => left_str != right_str,
                ComparisonOp::Less => left_str < right_str,
                ComparisonOp::LessEqual => left_str <= right_str,
                ComparisonOp::Greater => left_str > right_str,
                ComparisonOp::GreaterEqual => left_str >= right_str,
            })
        }
    }
}

/// Parse a term as a numeric value
fn parse_numeric(term: &Term) -> Option<f64> {
    match term {
        Term::Literal(s) | Term::Constant(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// Convert term to string for comparison
fn term_to_string(term: &Term) -> String {
    match term {
        Term::Variable(v) => format!("?{}", v),
        Term::Constant(c) => c.clone(),
        Term::Literal(l) => l.clone(),
    }
}

/// Evaluate builtin predicates
fn evaluate_builtin(
    predicate: &str,
    args: &[JoinArg],
    left: &EnhancedToken,
    right: &EnhancedToken,
) -> Result<bool> {
    // Get argument values
    let arg_values: Vec<Option<Term>> = args.iter().map(|arg| {
        match arg {
            JoinArg::LeftVar(var) => left.bindings.get(var).cloned(),
            JoinArg::RightVar(var) => right.bindings.get(var).cloned(),
            JoinArg::Constant(term) => Some(term.clone()),
        }
    }).collect();
    
    // Check all arguments are bound
    if arg_values.iter().any(|v| v.is_none()) {
        return Ok(false);
    }
    
    let values: Vec<Term> = arg_values.into_iter().map(|v| v.unwrap()).collect();
    
    // Evaluate builtin
    match predicate {
        "regex" => {
            if values.len() >= 2 {
                if let (Term::Literal(text), Term::Literal(pattern)) = (&values[0], &values[1]) {
                    let re = regex::Regex::new(pattern)
                        .map_err(|e| anyhow!("Invalid regex: {}", e))?;
                    Ok(re.is_match(text))
                } else {
                    Ok(false)
                }
            } else {
                Ok(false)
            }
        }
        "contains" => {
            if values.len() >= 2 {
                let s1 = term_to_string(&values[0]);
                let s2 = term_to_string(&values[1]);
                Ok(s1.contains(&s2))
            } else {
                Ok(false)
            }
        }
        "starts_with" => {
            if values.len() >= 2 {
                let s1 = term_to_string(&values[0]);
                let s2 = term_to_string(&values[1]);
                Ok(s1.starts_with(&s2))
            } else {
                Ok(false)
            }
        }
        "numeric_add" => {
            if values.len() >= 3 {
                if let (Some(n1), Some(n2), Some(result)) = 
                    (parse_numeric(&values[0]), parse_numeric(&values[1]), parse_numeric(&values[2])) {
                    Ok((n1 + n2 - result).abs() < f64::EPSILON)
                } else {
                    Ok(false)
                }
            } else {
                Ok(false)
            }
        }
        _ => {
            warn!("Unknown builtin predicate: {}", predicate);
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_token_merge() {
        let mut token1 = EnhancedToken::new();
        token1.bindings.insert("X".to_string(), Term::Constant("a".to_string()));
        token1.bindings.insert("Y".to_string(), Term::Constant("b".to_string()));
        
        let mut token2 = EnhancedToken::new();
        token2.bindings.insert("Y".to_string(), Term::Constant("b".to_string()));
        token2.bindings.insert("Z".to_string(), Term::Constant("c".to_string()));
        
        let merged = EnhancedToken::merge(&token1, &token2).unwrap();
        assert_eq!(merged.bindings.len(), 3);
        assert_eq!(merged.bindings.get("X"), Some(&Term::Constant("a".to_string())));
        assert_eq!(merged.bindings.get("Y"), Some(&Term::Constant("b".to_string())));
        assert_eq!(merged.bindings.get("Z"), Some(&Term::Constant("c".to_string())));
    }

    #[test]
    fn test_beta_memory_indexed_lookup() {
        let mut memory = BetaMemory::new(MemoryStrategy::Unlimited);
        
        // Add some tokens
        let mut token1 = EnhancedToken::new();
        token1.bindings.insert("X".to_string(), Term::Constant("a".to_string()));
        memory.add_left(token1, &["X".to_string()]);
        
        let mut token2 = EnhancedToken::new();
        token2.bindings.insert("X".to_string(), Term::Constant("b".to_string()));
        memory.add_left(token2, &["X".to_string()]);
        
        // Search for matching tokens
        let mut search_token = EnhancedToken::new();
        search_token.bindings.insert("X".to_string(), Term::Constant("a".to_string()));
        
        let matches = memory.find_matches_indexed(&search_token, false, &["X".to_string()]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].bindings.get("X"), Some(&Term::Constant("a".to_string())));
    }

    #[test]
    fn test_memory_eviction_strategies() {
        // Test count limit
        let mut memory = BetaMemory::new(MemoryStrategy::LimitCount(2));
        
        for i in 0..5 {
            let mut token = EnhancedToken::new();
            token.bindings.insert("X".to_string(), Term::Constant(i.to_string()));
            memory.add_left(token, &["X".to_string()]);
        }
        
        assert!(memory.left_tokens.len() <= 2);
        assert!(memory.stats.evictions > 0);
    }

    #[test]
    fn test_join_conditions() {
        let node = BetaJoinNode::new(
            1, 0, 0, 
            MemoryStrategy::Unlimited,
            ConflictResolution::First
        );
        
        let mut left = EnhancedToken::new();
        left.bindings.insert("X".to_string(), Term::Literal("5".to_string()));
        
        let mut right = EnhancedToken::new();
        right.bindings.insert("Y".to_string(), Term::Literal("10".to_string()));
        
        // Test numeric comparison
        let cond = JoinCondition::VarComparison {
            left_var: "X".to_string(),
            right_var: "Y".to_string(),
            op: ComparisonOp::Less,
        };
        
        assert!(node.evaluate_condition(&cond, &left, &right).unwrap());
    }

    #[test]
    fn test_builtin_evaluation() {
        let mut left = EnhancedToken::new();
        left.bindings.insert("text".to_string(), Term::Literal("hello world".to_string()));
        
        let right = EnhancedToken::new();
        
        // Test regex builtin
        let args = vec![
            JoinArg::LeftVar("text".to_string()),
            JoinArg::Constant(Term::Literal("hello.*".to_string())),
        ];
        
        assert!(evaluate_builtin("regex", &args, &left, &right).unwrap());
        
        // Test contains builtin
        let args = vec![
            JoinArg::LeftVar("text".to_string()),
            JoinArg::Constant(Term::Literal("world".to_string())),
        ];
        
        assert!(evaluate_builtin("contains", &args, &left, &right).unwrap());
    }
}