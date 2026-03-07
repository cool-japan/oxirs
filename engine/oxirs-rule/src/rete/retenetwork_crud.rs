//! # ReteNetwork - crud Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::rete_enhanced::{BetaJoinNode, ConflictResolution, EnhancedToken, MemoryStrategy};
use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

use super::functions::{NodeId, ACTIVE_TOKENS, TOKEN_CLONES};
use super::retenetwork_type::ReteNetwork;
use super::types::{JoinCondition, ReteNode, Token};

impl ReteNetwork {
    /// Create a new RETE network
    pub fn new() -> Self {
        Self::with_strategies(MemoryStrategy::Adaptive, ConflictResolution::Combined)
    }
    /// Create a new RETE network with specific strategies
    pub fn with_strategies(memory: MemoryStrategy, conflict: ConflictResolution) -> Self {
        let mut network = Self {
            nodes: HashMap::new(),
            next_node_id: 0,
            token_memory: HashMap::new(),
            alpha_memory: HashMap::new(),
            beta_memory: HashMap::new(),
            enhanced_beta_nodes: HashMap::new(),
            root_id: 0,
            pattern_index: HashMap::new(),
            debug_mode: false,
            memory_strategy: memory,
            conflict_resolution: conflict,
        };
        network.root_id = network.create_node(ReteNode::Root);
        network
            .token_memory
            .insert(network.root_id, vec![Token::new()]);
        network
    }
    /// Create a new node and return its ID
    pub(super) fn create_node(&mut self, node: ReteNode) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.insert(id, node);
        self.token_memory.insert(id, Vec::new());
        if self.debug_mode {
            debug!("Created node {}: {:?}", id, self.nodes.get(&id));
        }
        id
    }
    /// Add a rule to the RETE network
    pub fn add_rule(&mut self, rule: &Rule) -> Result<()> {
        info!("Adding rule '{}' to RETE network", rule.name);
        if rule.body.is_empty() {
            warn!("Rule '{}' has empty body, skipping", rule.name);
            return Ok(());
        }
        let mut current_nodes = vec![self.root_id];
        let mut pending_filters: Vec<RuleAtom> = Vec::new();
        for (i, condition) in rule.body.iter().enumerate() {
            if self.is_filter_condition(condition) {
                pending_filters.push(condition.clone());
                continue;
            }
            let mut next_nodes = Vec::new();
            for &current_node in &current_nodes {
                let node_id = if i == 0 || current_node == self.root_id {
                    self.create_alpha_node(condition.clone(), current_node)?
                } else {
                    self.create_beta_join(current_node, condition.clone())?
                };
                next_nodes.push(node_id);
            }
            for node_id in &next_nodes {
                if self.enhanced_beta_nodes.contains_key(node_id) {
                    for filter in &pending_filters {
                        if let Some(join_condition) = self.convert_filter_to_join_condition(filter)
                        {
                            self.add_filter_to_beta_node(*node_id, join_condition)?;
                        }
                    }
                }
            }
            pending_filters.clear();
            current_nodes = next_nodes;
        }
        if !pending_filters.is_empty() {
            let mut replacements: Vec<(usize, NodeId)> = Vec::new();
            for (idx, &node_id) in current_nodes.iter().enumerate() {
                if self.enhanced_beta_nodes.contains_key(&node_id) {
                    for filter in &pending_filters {
                        if let Some(join_condition) = self.convert_filter_to_join_condition(filter)
                        {
                            self.add_filter_to_beta_node(node_id, join_condition)?;
                        }
                    }
                } else {
                    if let Some(ReteNode::Alpha { .. }) = self.nodes.get(&node_id).cloned() {
                        let filter_beta_id =
                            self.create_filter_beta_node(node_id, &pending_filters)?;
                        replacements.push((idx, filter_beta_id));
                    }
                }
            }
            for (idx, filter_beta_id) in replacements {
                current_nodes[idx] = filter_beta_id;
            }
        }
        for &parent_id in &current_nodes {
            self.create_production_node(&rule.name, &rule.head, parent_id)?;
        }
        if self.debug_mode {
            debug!("Rule '{}' compiled into RETE network", rule.name);
        }
        Ok(())
    }
    /// Create a filter-only beta node for applying filter conditions after an alpha node
    /// This is used when a pattern is followed directly by filter conditions
    pub(super) fn create_filter_beta_node(
        &mut self,
        alpha_parent: NodeId,
        filters: &[RuleAtom],
    ) -> Result<NodeId> {
        let node_id = self.create_node(ReteNode::Beta {
            left_parent: alpha_parent,
            right_parent: alpha_parent,
            join_condition: JoinCondition::default(),
            children: Vec::new(),
        });
        self.add_child(alpha_parent, node_id)?;
        self.beta_memory.insert(node_id, (Vec::new(), Vec::new()));
        let mut enhanced_node = BetaJoinNode::new(
            node_id,
            alpha_parent,
            alpha_parent,
            self.memory_strategy,
            self.conflict_resolution,
        );
        for filter in filters {
            if let Some(join_condition) = self.convert_filter_to_join_condition(filter) {
                if self.debug_mode {
                    debug!(
                        "Adding filter condition {:?} to filter beta node {}",
                        join_condition, node_id
                    );
                }
                enhanced_node.conditions.push(join_condition);
            }
        }
        self.enhanced_beta_nodes.insert(node_id, enhanced_node);
        if self.debug_mode {
            debug!(
                "Created filter beta node {} with {} filters",
                node_id,
                filters.len()
            );
        }
        Ok(node_id)
    }
    /// Create an alpha node for pattern matching
    pub(super) fn create_alpha_node(
        &mut self,
        pattern: RuleAtom,
        _parent: NodeId,
    ) -> Result<NodeId> {
        let pattern_key = self.pattern_key(&pattern);
        if let Some(existing_nodes) = self.pattern_index.get(&pattern_key) {
            if let Some(&existing_id) = existing_nodes.first() {
                return Ok(existing_id);
            }
        }
        let node_id = self.create_node(ReteNode::Alpha {
            pattern: pattern.clone(),
            children: Vec::new(),
        });
        self.pattern_index
            .entry(pattern_key)
            .or_default()
            .push(node_id);
        self.alpha_memory.insert(node_id, HashSet::new());
        if self.debug_mode {
            debug!("Created alpha node {} for pattern: {:?}", node_id, pattern);
        }
        Ok(node_id)
    }
    /// Create a beta join node
    pub(super) fn create_beta_join(
        &mut self,
        left_parent: NodeId,
        right_pattern: RuleAtom,
    ) -> Result<NodeId> {
        let right_parent = self.create_alpha_node(right_pattern, self.root_id)?;
        let join_condition = self.analyze_join_conditions(left_parent, right_parent)?;
        let node_id = self.create_node(ReteNode::Beta {
            left_parent,
            right_parent,
            join_condition: join_condition.clone(),
            children: Vec::new(),
        });
        self.add_child(left_parent, node_id)?;
        self.add_child(right_parent, node_id)?;
        self.beta_memory.insert(node_id, (Vec::new(), Vec::new()));
        let mut enhanced_node = BetaJoinNode::new(
            node_id,
            left_parent,
            right_parent,
            self.memory_strategy,
            self.conflict_resolution,
        );
        enhanced_node.join_variables = join_condition
            .constraints
            .iter()
            .map(|(var, _)| var.clone())
            .collect();
        if enhanced_node.join_variables.is_empty() {
            if let (Some(left_pattern), Some(right_pattern)) = (
                self.get_node_pattern(left_parent)?,
                self.get_node_pattern(right_parent)?,
            ) {
                let left_vars = self.extract_variables(&left_pattern);
                let right_vars = self.extract_variables(&right_pattern);
                for left_var in &left_vars {
                    if right_vars.contains(left_var) {
                        enhanced_node.join_variables.push(left_var.clone());
                    }
                }
                if self.debug_mode && !enhanced_node.join_variables.is_empty() {
                    debug!(
                        "Fallback join variable detection found variables: {:?} from patterns left: {:?}, right: {:?}",
                        enhanced_node.join_variables, left_pattern, right_pattern
                    );
                }
            }
        }
        if enhanced_node.join_variables.is_empty() {
            for constraint in &join_condition.constraints {
                if let Some(var) = constraint.0.strip_prefix("join_") {
                    enhanced_node.join_variables.push(var.to_string());
                }
                if let Some(var) = constraint.1.strip_prefix("join_") {
                    enhanced_node.join_variables.push(var.to_string());
                }
            }
        }
        for filter in &join_condition.filters {
            match filter.as_str() {
                "type_constraint" => {
                    enhanced_node
                        .conditions
                        .push(crate::rete_enhanced::JoinCondition::Builtin {
                            predicate: "type_check".to_string(),
                            args: vec![],
                        });
                }
                "domain_range_constraint" => {
                    enhanced_node
                        .conditions
                        .push(crate::rete_enhanced::JoinCondition::Builtin {
                            predicate: "domain_range_check".to_string(),
                            args: vec![],
                        });
                }
                _ => {}
            }
        }
        self.enhanced_beta_nodes.insert(node_id, enhanced_node);
        if self.debug_mode {
            debug!(
                "Created enhanced beta join node {} (left: {}, right: {})",
                node_id, left_parent, right_parent
            );
        }
        Ok(node_id)
    }
    /// Analyze join conditions between two nodes
    pub(super) fn analyze_join_conditions(
        &self,
        left_id: NodeId,
        right_id: NodeId,
    ) -> Result<JoinCondition> {
        let mut constraints = Vec::new();
        let mut filters = Vec::new();
        let left_pattern = self.get_node_pattern(left_id)?;
        let right_pattern = self.get_node_pattern(right_id)?;
        if let (Some(left_pattern), Some(right_pattern)) = (left_pattern, right_pattern) {
            let left_vars = self.extract_variables(&left_pattern);
            let right_vars = self.extract_variables(&right_pattern);
            for left_var in &left_vars {
                if right_vars.contains(left_var) {
                    constraints.push((left_var.clone(), left_var.clone()));
                }
            }
            if self.should_add_type_constraint(&left_pattern, &right_pattern) {
                filters.push("type_constraint".to_string());
            }
            if self.should_add_domain_range_constraint(&left_pattern, &right_pattern) {
                filters.push("domain_range_constraint".to_string());
            }
        }
        if self.debug_mode && !constraints.is_empty() {
            debug!(
                "Generated {} join constraints between nodes {} and {}",
                constraints.len(),
                left_id,
                right_id
            );
        }
        Ok(JoinCondition {
            constraints,
            filters,
        })
    }
    /// Extract variables from a rule atom
    pub(super) fn extract_variables(&self, atom: &RuleAtom) -> Vec<String> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                let mut vars = Vec::new();
                if let Term::Variable(var) = subject {
                    vars.push(var.clone());
                }
                if let Term::Variable(var) = predicate {
                    vars.push(var.clone());
                }
                if let Term::Variable(var) = object {
                    vars.push(var.clone());
                }
                vars
            }
            RuleAtom::Builtin { args, .. } => {
                let mut vars = Vec::new();
                for arg in args {
                    if let Term::Variable(var) = arg {
                        vars.push(var.clone());
                    }
                }
                vars
            }
            RuleAtom::NotEqual { left, right } => {
                let mut vars = Vec::new();
                if let Term::Variable(var) = left {
                    vars.push(var.clone());
                }
                if let Term::Variable(var) = right {
                    vars.push(var.clone());
                }
                vars
            }
            RuleAtom::GreaterThan { left, right } => {
                let mut vars = Vec::new();
                if let Term::Variable(var) = left {
                    vars.push(var.clone());
                }
                if let Term::Variable(var) = right {
                    vars.push(var.clone());
                }
                vars
            }
            RuleAtom::LessThan { left, right } => {
                let mut vars = Vec::new();
                if let Term::Variable(var) = left {
                    vars.push(var.clone());
                }
                if let Term::Variable(var) = right {
                    vars.push(var.clone());
                }
                vars
            }
        }
    }
    /// Add a fact to the network
    /// OPTIMIZED: Collect matching alpha nodes to avoid cloning entire map
    pub fn add_fact(&mut self, fact: RuleAtom) -> Result<Vec<RuleAtom>> {
        if self.debug_mode {
            debug!("Adding fact to RETE network: {:?}", fact);
        }
        let mut derived_facts = Vec::new();
        let mut matching_alphas = Vec::new();
        for (&node_id, node) in &self.nodes {
            if let ReteNode::Alpha { pattern, .. } = node {
                if let Some(substitution) = self.unify_atoms(pattern, &fact, &HashMap::new())? {
                    matching_alphas.push((node_id, substitution));
                }
            }
        }
        for (node_id, substitution) in matching_alphas {
            if let Some(memory) = self.alpha_memory.get_mut(&node_id) {
                memory.insert(fact.clone());
            }
            TOKEN_CLONES.inc();
            let mut token = Token::with_fact(fact.clone());
            token.bindings = substitution;
            ACTIVE_TOKENS.set(self.token_memory.values().map(|v| v.len()).sum::<usize>() as f64);
            let new_facts = self.propagate_token(node_id, token)?;
            derived_facts.extend(new_facts);
        }
        Ok(derived_facts)
    }
    /// Check if a fact matches an alpha node pattern
    #[allow(dead_code)]
    pub(super) fn matches_alpha_pattern(&self, alpha_id: NodeId, fact: &RuleAtom) -> Result<bool> {
        if let Some(ReteNode::Alpha { pattern, .. }) = self.nodes.get(&alpha_id) {
            Ok(self.unify_atoms(pattern, fact, &HashMap::new())?.is_some())
        } else {
            Ok(false)
        }
    }
    /// Propagate a token through the network
    /// OPTIMIZED: Extract necessary data to avoid node clones
    pub(super) fn propagate_token(
        &mut self,
        node_id: NodeId,
        token: Token,
    ) -> Result<Vec<RuleAtom>> {
        let mut derived_facts = Vec::new();
        let node_type = self.nodes.get(&node_id).map(|node| match node {
            ReteNode::Alpha { children, .. } => (0, children.clone(), None, None, None),
            ReteNode::Beta {
                join_condition,
                children,
                ..
            } => (
                1,
                children.clone(),
                Some(join_condition.clone()),
                None,
                None,
            ),
            ReteNode::Production {
                rule_name,
                rule_head,
                ..
            } => (
                2,
                Vec::new(),
                None,
                Some(rule_name.clone()),
                Some(rule_head.clone()),
            ),
            _ => (3, Vec::new(), None, None, None),
        });
        match node_type {
            Some((0, children, _, _, _)) => {
                for &child_id in &children {
                    TOKEN_CLONES.inc();
                    let new_facts = self.propagate_token(child_id, token.clone())?;
                    derived_facts.extend(new_facts);
                }
            }
            Some((1, children, Some(join_condition), _, _)) => {
                let joined_tokens = self.perform_beta_join(node_id, token, &join_condition)?;
                for joined_token in joined_tokens {
                    for &child_id in &children {
                        TOKEN_CLONES.inc();
                        let new_facts = self.propagate_token(child_id, joined_token.clone())?;
                        derived_facts.extend(new_facts);
                    }
                }
            }
            Some((2, _, _, Some(rule_name), Some(rule_head))) => {
                let new_facts = self.execute_production(&rule_name, &rule_head, &token)?;
                derived_facts.extend(new_facts);
            }
            _ => {}
        }
        Ok(derived_facts)
    }
    /// Perform beta join operation with proper memory management
    pub(super) fn perform_beta_join(
        &mut self,
        beta_id: NodeId,
        token: Token,
        join_condition: &JoinCondition,
    ) -> Result<Vec<Token>> {
        println!("perform_beta_join called with beta_id={beta_id}, token={token:?}");
        if self.enhanced_beta_nodes.contains_key(&beta_id) {
            let from_left = self.is_left_token(&token, beta_id)?;
            let mut enhanced_token = EnhancedToken::new();
            enhanced_token.bindings = token.bindings.clone();
            enhanced_token.facts = token.facts.clone();
            enhanced_token.priority = 0;
            enhanced_token.specificity = token.facts.len();
            let enhanced_beta_node = self
                .enhanced_beta_nodes
                .get_mut(&beta_id)
                .expect("enhanced beta node should exist for beta_id");
            let enhanced_results = enhanced_beta_node.join(enhanced_token, from_left)?;
            let mut joined_tokens = Vec::new();
            for enhanced in enhanced_results {
                let mut regular_token = Token::new();
                regular_token.bindings = enhanced.bindings;
                regular_token.facts = enhanced.facts;
                regular_token.tags = enhanced.justification;
                joined_tokens.push(regular_token);
            }
            if self.debug_mode && !joined_tokens.is_empty() {
                debug!(
                    "Enhanced beta join {} produced {} joined tokens",
                    beta_id,
                    joined_tokens.len()
                );
                if let Some(stats) = self.enhanced_beta_nodes.get(&beta_id) {
                    debug!("Beta join stats: {:?}", stats.get_stats());
                }
            }
            println!(
                "Enhanced beta join produced {} joined tokens",
                joined_tokens.len()
            );
            for (i, token) in joined_tokens.iter().enumerate() {
                println!("  Joined token {i}: {token:?}");
            }
            Ok(joined_tokens)
        } else {
            let is_left_token = self.is_left_token(&token, beta_id)?;
            let mut joined_tokens = Vec::new();
            println!("Using fallback beta join implementation, is_left_token: {is_left_token}");
            if is_left_token {
                let right_tokens: Vec<_> = {
                    let (_, right_memory) = self.beta_memory.get(&beta_id).ok_or_else(|| {
                        anyhow::anyhow!("Beta memory not found for node {}", beta_id)
                    })?;
                    right_memory.to_vec()
                };
                {
                    let (left_memory, _) = self.beta_memory.get_mut(&beta_id).ok_or_else(|| {
                        anyhow::anyhow!("Beta memory not found for node {}", beta_id)
                    })?;
                    left_memory.push(token.clone());
                }
                println!("Left token: {token:?}");
                println!("Available right tokens: {} tokens", right_tokens.len());
                for (i, right_token) in right_tokens.iter().enumerate() {
                    println!("  Right token {i}: {right_token:?}");
                    if self.satisfies_join_condition(&token, right_token, join_condition)? {
                        println!("    Join condition satisfied! Creating joined token...");
                        let joined = self.join_tokens(&token, right_token)?;
                        println!("    Joined token: {joined:?}");
                        joined_tokens.push(joined);
                    } else {
                        println!("    Join condition NOT satisfied");
                    }
                }
            } else {
                let left_tokens: Vec<_> = {
                    let (left_memory, _) = self.beta_memory.get(&beta_id).ok_or_else(|| {
                        anyhow::anyhow!("Beta memory not found for node {}", beta_id)
                    })?;
                    left_memory.to_vec()
                };
                {
                    let (_, right_memory) =
                        self.beta_memory.get_mut(&beta_id).ok_or_else(|| {
                            anyhow::anyhow!("Beta memory not found for node {}", beta_id)
                        })?;
                    right_memory.push(token.clone());
                }
                println!("Right token: {token:?}");
                println!("Available left tokens: {} tokens", left_tokens.len());
                for (i, left_token) in left_tokens.iter().enumerate() {
                    println!("  Left token {i}: {left_token:?}");
                    if self.satisfies_join_condition(left_token, &token, join_condition)? {
                        println!("    Join condition satisfied! Creating joined token...");
                        let joined = self.join_tokens(left_token, &token)?;
                        println!("    Joined token: {joined:?}");
                        joined_tokens.push(joined);
                    } else {
                        println!("    Join condition NOT satisfied");
                    }
                }
            }
            const MAX_MEMORY_SIZE: usize = 10000;
            {
                let (left_memory, right_memory) = self
                    .beta_memory
                    .get_mut(&beta_id)
                    .ok_or_else(|| anyhow::anyhow!("Beta memory not found for node {}", beta_id))?;
                if left_memory.len() > MAX_MEMORY_SIZE {
                    left_memory.drain(0..left_memory.len() / 2);
                }
                if right_memory.len() > MAX_MEMORY_SIZE {
                    right_memory.drain(0..right_memory.len() / 2);
                }
            }
            println!(
                "Fallback beta join produced {} joined tokens",
                joined_tokens.len()
            );
            for (i, token) in joined_tokens.iter().enumerate() {
                println!("  Fallback joined token {i}: {token:?}");
            }
            Ok(joined_tokens)
        }
    }
}
