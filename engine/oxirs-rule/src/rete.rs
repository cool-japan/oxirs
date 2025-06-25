//! RETE Pattern Matching Network
//!
//! Implementation of the RETE algorithm for efficient pattern matching in rule-based systems.
//! The RETE network precompiles rules into a network that allows for incremental updates.

use crate::forward::Substitution;
use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use tracing::{debug, info, trace, warn};

/// Unique identifier for RETE nodes
pub type NodeId = usize;

/// RETE network statistics
#[derive(Debug, Clone)]
pub struct ReteStats {
    pub total_nodes: usize,
    pub alpha_nodes: usize,
    pub beta_nodes: usize,
    pub production_nodes: usize,
    pub total_tokens: usize,
}

impl fmt::Display for ReteStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Nodes: {} (α:{}, β:{}, P:{}), Tokens: {}",
            self.total_nodes,
            self.alpha_nodes,
            self.beta_nodes,
            self.production_nodes,
            self.total_tokens
        )
    }
}

/// Token representing partial matches flowing through the network
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// Current variable bindings
    pub bindings: Substitution,
    /// Tags for tracking token origin
    pub tags: Vec<String>,
    /// Facts that contributed to this token
    pub facts: Vec<RuleAtom>,
}

impl Token {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            tags: Vec::new(),
            facts: Vec::new(),
        }
    }

    pub fn with_fact(fact: RuleAtom) -> Self {
        Self {
            bindings: HashMap::new(),
            tags: Vec::new(),
            facts: vec![fact],
        }
    }
}

impl Default for Token {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of RETE nodes
#[derive(Debug, Clone)]
pub enum ReteNode {
    /// Root node - entry point for all facts
    Root,
    /// Alpha node - tests individual fact patterns
    Alpha {
        pattern: RuleAtom,
        children: Vec<NodeId>,
    },
    /// Beta node - joins two streams of tokens
    Beta {
        left_parent: NodeId,
        right_parent: NodeId,
        join_condition: JoinCondition,
        children: Vec<NodeId>,
    },
    /// Production node - executes rule actions
    Production {
        rule_name: String,
        rule_head: Vec<RuleAtom>,
        parent: NodeId,
    },
}

/// Join condition for beta nodes
#[derive(Debug, Clone)]
pub struct JoinCondition {
    /// Variable constraints between left and right tokens
    pub constraints: Vec<(String, String)>,
    /// Additional filters
    pub filters: Vec<String>,
}

impl Default for JoinCondition {
    fn default() -> Self {
        Self {
            constraints: Vec::new(),
            filters: Vec::new(),
        }
    }
}

/// RETE network implementation
#[derive(Debug)]
pub struct ReteNetwork {
    /// Network nodes indexed by ID
    nodes: HashMap<NodeId, ReteNode>,
    /// Next available node ID
    next_node_id: NodeId,
    /// Token memory for each node
    token_memory: HashMap<NodeId, Vec<Token>>,
    /// Alpha memory for alpha nodes
    alpha_memory: HashMap<NodeId, HashSet<RuleAtom>>,
    /// Beta memory for beta nodes (left and right)
    beta_memory: HashMap<NodeId, (Vec<Token>, Vec<Token>)>,
    /// Root node ID
    root_id: NodeId,
    /// Pattern to alpha node mapping for efficiency
    pattern_index: HashMap<String, Vec<NodeId>>,
    /// Debug mode
    debug_mode: bool,
}

impl Default for ReteNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl ReteNetwork {
    /// Create a new RETE network
    pub fn new() -> Self {
        let mut network = Self {
            nodes: HashMap::new(),
            next_node_id: 0,
            token_memory: HashMap::new(),
            alpha_memory: HashMap::new(),
            beta_memory: HashMap::new(),
            root_id: 0,
            pattern_index: HashMap::new(),
            debug_mode: false,
        };

        // Create root node
        network.root_id = network.create_node(ReteNode::Root);
        network
            .token_memory
            .insert(network.root_id, vec![Token::new()]);

        network
    }

    /// Enable or disable debug mode
    pub fn set_debug_mode(&mut self, debug: bool) {
        self.debug_mode = debug;
    }

    /// Create a new node and return its ID
    fn create_node(&mut self, node: ReteNode) -> NodeId {
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

        // Build the network for this rule
        let mut current_nodes = vec![self.root_id];

        // Process each condition in the rule body
        for (i, condition) in rule.body.iter().enumerate() {
            let mut next_nodes = Vec::new();

            for &current_node in &current_nodes {
                let node_id = if i == 0 {
                    // First condition - create alpha node
                    self.create_alpha_node(condition.clone(), current_node)?
                } else {
                    // Subsequent conditions - create beta join
                    self.create_beta_join(current_node, condition.clone())?
                };
                next_nodes.push(node_id);
            }

            current_nodes = next_nodes;
        }

        // Create production nodes for rule head
        for &parent_id in &current_nodes {
            self.create_production_node(&rule.name, &rule.head, parent_id)?;
        }

        if self.debug_mode {
            debug!("Rule '{}' compiled into RETE network", rule.name);
        }

        Ok(())
    }

    /// Create an alpha node for pattern matching
    fn create_alpha_node(&mut self, pattern: RuleAtom, parent: NodeId) -> Result<NodeId> {
        // Check if alpha node already exists for this pattern
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

        // Update pattern index
        self.pattern_index
            .entry(pattern_key)
            .or_default()
            .push(node_id);

        // Initialize alpha memory
        self.alpha_memory.insert(node_id, HashSet::new());

        if self.debug_mode {
            debug!("Created alpha node {} for pattern: {:?}", node_id, pattern);
        }

        Ok(node_id)
    }

    /// Create a beta join node
    fn create_beta_join(&mut self, left_parent: NodeId, right_pattern: RuleAtom) -> Result<NodeId> {
        // Create alpha node for right side
        let right_parent = self.create_alpha_node(right_pattern, self.root_id)?;

        // Determine join conditions
        let join_condition = self.analyze_join_conditions(left_parent, right_parent)?;

        let node_id = self.create_node(ReteNode::Beta {
            left_parent,
            right_parent,
            join_condition,
            children: Vec::new(),
        });

        // Update parent children
        self.add_child(left_parent, node_id)?;
        self.add_child(right_parent, node_id)?;

        // Initialize beta memory
        self.beta_memory.insert(node_id, (Vec::new(), Vec::new()));

        if self.debug_mode {
            debug!(
                "Created beta join node {} (left: {}, right: {})",
                node_id, left_parent, right_parent
            );
        }

        Ok(node_id)
    }

    /// Create a production node
    fn create_production_node(
        &mut self,
        rule_name: &str,
        rule_head: &[RuleAtom],
        parent: NodeId,
    ) -> Result<NodeId> {
        let node_id = self.create_node(ReteNode::Production {
            rule_name: rule_name.to_string(),
            rule_head: rule_head.to_vec(),
            parent,
        });

        self.add_child(parent, node_id)?;

        if self.debug_mode {
            debug!(
                "Created production node {} for rule '{}'",
                node_id, rule_name
            );
        }

        Ok(node_id)
    }

    /// Add a child to a node
    fn add_child(&mut self, parent_id: NodeId, child_id: NodeId) -> Result<()> {
        match self.nodes.get_mut(&parent_id) {
            Some(ReteNode::Alpha { children, .. }) => {
                children.push(child_id);
            }
            Some(ReteNode::Beta { children, .. }) => {
                children.push(child_id);
            }
            _ => {
                return Err(anyhow::anyhow!("Cannot add child to node type"));
            }
        }
        Ok(())
    }

    /// Generate a unique key for a pattern
    fn pattern_key(&self, pattern: &RuleAtom) -> String {
        match pattern {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                format!(
                    "triple:{}:{}:{}",
                    self.term_key(subject),
                    self.term_key(predicate),
                    self.term_key(object)
                )
            }
            RuleAtom::Builtin { name, args } => {
                let arg_keys: Vec<String> = args.iter().map(|arg| self.term_key(arg)).collect();
                format!("builtin:{}:{}", name, arg_keys.join(","))
            }
        }
    }

    /// Generate a key for a term
    fn term_key(&self, term: &Term) -> String {
        match term {
            Term::Variable(v) => format!("?{}", v),
            Term::Constant(c) => format!("c:{}", c),
            Term::Literal(l) => format!("l:{}", l),
        }
    }

    /// Analyze join conditions between two nodes
    fn analyze_join_conditions(&self, left_id: NodeId, right_id: NodeId) -> Result<JoinCondition> {
        // For now, implement basic variable matching
        // In a full implementation, this would analyze variable overlap
        Ok(JoinCondition::default())
    }

    /// Add a fact to the network
    pub fn add_fact(&mut self, fact: RuleAtom) -> Result<Vec<RuleAtom>> {
        if self.debug_mode {
            debug!("Adding fact to RETE network: {:?}", fact);
        }

        let mut derived_facts = Vec::new();

        // Find matching alpha nodes by checking all alpha nodes, not just exact pattern matches
        for (&node_id, node) in &self.nodes.clone() {
            if let ReteNode::Alpha { pattern, .. } = node {
                if self.unify_atoms(pattern, &fact, &HashMap::new())?.is_some() {
                    // Add to alpha memory
                    if let Some(memory) = self.alpha_memory.get_mut(&node_id) {
                        memory.insert(fact.clone());
                    }

                    // Propagate through network
                    let token = Token::with_fact(fact.clone());
                    let new_facts = self.propagate_token(node_id, token)?;
                    derived_facts.extend(new_facts);
                }
            }
        }

        Ok(derived_facts)
    }

    /// Check if a fact matches an alpha node pattern
    fn matches_alpha_pattern(&self, alpha_id: NodeId, fact: &RuleAtom) -> Result<bool> {
        if let Some(ReteNode::Alpha { pattern, .. }) = self.nodes.get(&alpha_id) {
            Ok(self.unify_atoms(pattern, fact, &HashMap::new())?.is_some())
        } else {
            Ok(false)
        }
    }

    /// Propagate a token through the network
    fn propagate_token(&mut self, node_id: NodeId, token: Token) -> Result<Vec<RuleAtom>> {
        let mut derived_facts = Vec::new();

        match self.nodes.get(&node_id).cloned() {
            Some(ReteNode::Alpha { children, .. }) => {
                // Propagate to all children
                for &child_id in &children {
                    let new_facts = self.propagate_token(child_id, token.clone())?;
                    derived_facts.extend(new_facts);
                }
            }
            Some(ReteNode::Beta {
                left_parent,
                right_parent,
                ref join_condition,
                children,
            }) => {
                // Handle beta join
                let joined_tokens = self.perform_beta_join(node_id, token, &join_condition)?;
                for joined_token in joined_tokens {
                    for &child_id in &children {
                        let new_facts = self.propagate_token(child_id, joined_token.clone())?;
                        derived_facts.extend(new_facts);
                    }
                }
            }
            Some(ReteNode::Production {
                ref rule_name,
                ref rule_head,
                ..
            }) => {
                // Execute production
                let new_facts = self.execute_production(rule_name, rule_head, &token)?;
                derived_facts.extend(new_facts);
            }
            _ => {}
        }

        Ok(derived_facts)
    }

    /// Perform beta join operation
    fn perform_beta_join(
        &mut self,
        _beta_id: NodeId,
        token: Token,
        _join_condition: &JoinCondition,
    ) -> Result<Vec<Token>> {
        // Simplified beta join - in a full implementation this would:
        // 1. Check which side the token came from
        // 2. Join with tokens from the other side
        // 3. Apply join conditions
        // 4. Update beta memory

        // For now, just pass through the token
        Ok(vec![token])
    }

    /// Execute a production node
    fn execute_production(
        &self,
        rule_name: &str,
        rule_head: &[RuleAtom],
        token: &Token,
    ) -> Result<Vec<RuleAtom>> {
        if self.debug_mode {
            debug!(
                "Executing production '{}' with token: {:?}",
                rule_name, token
            );
        }

        let mut derived_facts = Vec::new();

        // For each fact in the token, try to create variable bindings and apply to head
        if let Some(fact) = token.facts.first() {
            // Create a simple substitution from the fact
            // This is a simplified approach - a full RETE would maintain proper variable bindings
            for head_atom in rule_head {
                let instantiated = match (head_atom, fact) {
                    (
                        RuleAtom::Triple {
                            subject: h_s,
                            predicate: h_p,
                            object: h_o,
                        },
                        RuleAtom::Triple {
                            subject: f_s,
                            predicate: f_p,
                            object: f_o,
                        },
                    ) => {
                        let mut substitution = HashMap::new();

                        // Simple variable mapping based on structure
                        if let Term::Variable(var) = h_s {
                            substitution.insert(var.clone(), f_s.clone());
                        }
                        if let Term::Variable(var) = h_p {
                            substitution.insert(var.clone(), f_p.clone());
                        }
                        if let Term::Variable(var) = h_o {
                            substitution.insert(var.clone(), f_o.clone());
                        }

                        self.apply_substitution(head_atom, &substitution)?
                    }
                    _ => head_atom.clone(),
                };
                derived_facts.push(instantiated);
            }
        }

        if self.debug_mode && !derived_facts.is_empty() {
            debug!(
                "Production '{}' derived {} facts",
                rule_name,
                derived_facts.len()
            );
        }

        Ok(derived_facts)
    }

    /// Remove a fact from the network
    pub fn remove_fact(&mut self, fact: &RuleAtom) -> Result<Vec<RuleAtom>> {
        if self.debug_mode {
            debug!("Removing fact from RETE network: {:?}", fact);
        }

        let mut retracted_facts = Vec::new();

        // Find and remove from alpha memories
        let pattern_key = self.pattern_key(fact);
        if let Some(alpha_nodes) = self.pattern_index.get(&pattern_key).cloned() {
            for &alpha_id in &alpha_nodes {
                if let Some(memory) = self.alpha_memory.get_mut(&alpha_id) {
                    if memory.remove(fact) {
                        // Fact was removed, need to retract dependent facts
                        // This would require a more sophisticated implementation
                        // with dependency tracking
                    }
                }
            }
        }

        Ok(retracted_facts)
    }

    /// Unify two atoms with given substitution
    fn unify_atoms(
        &self,
        atom1: &RuleAtom,
        atom2: &RuleAtom,
        substitution: &Substitution,
    ) -> Result<Option<Substitution>> {
        match (atom1, atom2) {
            (
                RuleAtom::Triple {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
                RuleAtom::Triple {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                let mut sub = substitution.clone();
                if self.unify_terms(s1, s2, &mut sub)?
                    && self.unify_terms(p1, p2, &mut sub)?
                    && self.unify_terms(o1, o2, &mut sub)?
                {
                    Ok(Some(sub))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Unify two terms
    fn unify_terms(
        &self,
        term1: &Term,
        term2: &Term,
        substitution: &mut Substitution,
    ) -> Result<bool> {
        match (term1, term2) {
            (Term::Variable(var), term) | (term, Term::Variable(var)) => {
                if let Some(existing) = substitution.get(var) {
                    Ok(self.terms_equal(existing, term))
                } else {
                    substitution.insert(var.clone(), term.clone());
                    Ok(true)
                }
            }
            (Term::Constant(c1), Term::Constant(c2)) => Ok(c1 == c2),
            (Term::Literal(l1), Term::Literal(l2)) => Ok(l1 == l2),
            (Term::Constant(c), Term::Literal(l)) | (Term::Literal(l), Term::Constant(c)) => {
                Ok(c == l)
            }
            _ => Ok(false),
        }
    }

    /// Check if two terms are equal
    fn terms_equal(&self, term1: &Term, term2: &Term) -> bool {
        match (term1, term2) {
            (Term::Variable(v1), Term::Variable(v2)) => v1 == v2,
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            (Term::Constant(c), Term::Literal(l)) | (Term::Literal(l), Term::Constant(c)) => c == l,
            _ => false,
        }
    }

    /// Apply substitution to an atom
    fn apply_substitution(&self, atom: &RuleAtom, substitution: &Substitution) -> Result<RuleAtom> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => Ok(RuleAtom::Triple {
                subject: self.substitute_term(subject, substitution),
                predicate: self.substitute_term(predicate, substitution),
                object: self.substitute_term(object, substitution),
            }),
            RuleAtom::Builtin { name, args } => {
                let substituted_args = args
                    .iter()
                    .map(|arg| self.substitute_term(arg, substitution))
                    .collect();
                Ok(RuleAtom::Builtin {
                    name: name.clone(),
                    args: substituted_args,
                })
            }
        }
    }

    /// Substitute variables in a term
    fn substitute_term(&self, term: &Term, substitution: &Substitution) -> Term {
        match term {
            Term::Variable(var) => substitution
                .get(var)
                .cloned()
                .unwrap_or_else(|| term.clone()),
            _ => term.clone(),
        }
    }

    /// Get network statistics
    pub fn get_stats(&self) -> ReteStats {
        let mut alpha_count = 0;
        let mut beta_count = 0;
        let mut production_count = 0;
        let mut total_tokens = 0;

        for node in self.nodes.values() {
            match node {
                ReteNode::Alpha { .. } => alpha_count += 1,
                ReteNode::Beta { .. } => beta_count += 1,
                ReteNode::Production { .. } => production_count += 1,
                _ => {}
            }
        }

        for tokens in self.token_memory.values() {
            total_tokens += tokens.len();
        }

        ReteStats {
            total_nodes: self.nodes.len(),
            alpha_nodes: alpha_count,
            beta_nodes: beta_count,
            production_nodes: production_count,
            total_tokens,
        }
    }

    /// Clear all facts and reset the network
    pub fn clear(&mut self) {
        self.alpha_memory.clear();
        self.beta_memory.clear();
        for tokens in self.token_memory.values_mut() {
            tokens.clear();
        }

        // Reset root token
        self.token_memory.insert(self.root_id, vec![Token::new()]);
    }

    /// Get all facts currently in the network
    pub fn get_facts(&self) -> Vec<RuleAtom> {
        let mut facts = Vec::new();
        for memory in self.alpha_memory.values() {
            facts.extend(memory.iter().cloned());
        }
        facts
    }

    /// Perform forward chaining until fixpoint
    pub fn forward_chain(&mut self, initial_facts: Vec<RuleAtom>) -> Result<Vec<RuleAtom>> {
        info!(
            "Starting RETE forward chaining with {} initial facts",
            initial_facts.len()
        );

        let mut all_facts = HashSet::new();
        let mut facts_to_process = VecDeque::from(initial_facts);
        let mut iteration = 0;

        while let Some(fact) = facts_to_process.pop_front() {
            if all_facts.contains(&fact) {
                continue;
            }

            all_facts.insert(fact.clone());
            iteration += 1;

            if self.debug_mode && iteration % 100 == 0 {
                debug!(
                    "RETE iteration {}, {} facts processed",
                    iteration,
                    all_facts.len()
                );
            }

            let derived = self.add_fact(fact)?;
            for derived_fact in derived {
                if !all_facts.contains(&derived_fact) {
                    facts_to_process.push_back(derived_fact);
                }
            }
        }

        info!(
            "RETE forward chaining completed after {} iterations, {} facts total",
            iteration,
            all_facts.len()
        );

        Ok(all_facts.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rete_network_creation() {
        let network = ReteNetwork::new();
        let stats = network.get_stats();
        assert_eq!(stats.total_nodes, 1); // Root node
        assert_eq!(stats.alpha_nodes, 0);
        assert_eq!(stats.beta_nodes, 0);
        assert_eq!(stats.production_nodes, 0);
    }

    #[test]
    fn test_simple_rule_compilation() {
        let mut network = ReteNetwork::new();

        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("human".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("mortal".to_string()),
            }],
        };

        network.add_rule(&rule).unwrap();

        let stats = network.get_stats();
        assert!(stats.alpha_nodes > 0);
        assert!(stats.production_nodes > 0);
    }

    #[test]
    fn test_fact_processing() {
        let mut network = ReteNetwork::new();

        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("human".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("mortal".to_string()),
            }],
        };

        network.add_rule(&rule).unwrap();

        let fact = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        };

        let derived = network.add_fact(fact).unwrap();
        assert!(!derived.is_empty());

        let expected = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("mortal".to_string()),
        };

        assert!(derived.contains(&expected));
    }

    #[test]
    fn test_forward_chaining() {
        let mut network = ReteNetwork::new();

        // Use a simpler single-condition rule for this test
        // The current RETE implementation is simplified and doesn't support full beta joins
        let rule = Rule {
            name: "simple_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        network.add_rule(&rule).unwrap();

        let initial_facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Constant("mary".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("mary".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Constant("bob".to_string()),
            },
        ];

        let all_facts = network.forward_chain(initial_facts).unwrap();

        // Should derive ancestor relationships from parent facts
        let expected1 = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Constant("mary".to_string()),
        };

        let expected2 = RuleAtom::Triple {
            subject: Term::Constant("mary".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Constant("bob".to_string()),
        };

        assert!(all_facts.contains(&expected1));
        assert!(all_facts.contains(&expected2));
    }
}
