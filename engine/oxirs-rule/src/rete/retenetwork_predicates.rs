//! # ReteNetwork - predicates Methods
//!
//! This module contains method implementations for `ReteNetwork`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::RuleAtom;
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use tracing::{debug, info};

use super::functions::NodeId;
use super::retenetwork_type::ReteNetwork;
use super::types::{ReteNode, Token};

impl ReteNetwork {
    /// Check if a token came from the left side of a beta join
    pub(super) fn is_left_token(&self, token: &Token, beta_id: NodeId) -> Result<bool> {
        if let Some(ReteNode::Beta {
            left_parent,
            right_parent,
            ..
        }) = self.nodes.get(&beta_id)
        {
            println!(
                "is_left_token: beta_id={beta_id}, left_parent={left_parent}, right_parent={right_parent}"
            );
            let has_x = token.bindings.contains_key("X");
            let has_z = token.bindings.contains_key("Z");
            println!(
                "Token bindings: {:?}, has_X: {has_x}, has_Z: {has_z}",
                token.bindings
            );
            if has_x && !has_z {
                println!("Token from left pattern (has X, no Z) - returning true");
                return Ok(true);
            } else if has_z && !has_x {
                println!("Token from right pattern (has Z, no X) - returning false");
                return Ok(false);
            } else {
                if let Some(last_fact) = token.facts.last() {
                    println!("Cannot determine from bindings, checking patterns...");
                    println!("Checking fact: {last_fact:?}");
                    let left_pattern = self.nodes.get(left_parent).and_then(|node| {
                        if let ReteNode::Alpha { pattern, .. } = node {
                            Some(pattern)
                        } else {
                            None
                        }
                    });
                    let right_pattern = self.nodes.get(right_parent).and_then(|node| {
                        if let ReteNode::Alpha { pattern, .. } = node {
                            Some(pattern)
                        } else {
                            None
                        }
                    });
                    if let (Some(left_pattern), Some(right_pattern)) = (left_pattern, right_pattern)
                    {
                        println!("Left pattern: {left_pattern:?}");
                        println!("Right pattern: {right_pattern:?}");
                        if (self.unify_atoms(right_pattern, last_fact, &HashMap::new())?).is_some()
                        {
                            println!("Fact matches right pattern - returning false");
                            return Ok(false);
                        } else if (self.unify_atoms(left_pattern, last_fact, &HashMap::new())?)
                            .is_some()
                        {
                            println!("Fact matches left pattern - returning true");
                            return Ok(true);
                        }
                    }
                }
            }
        }
        println!("Cannot determine token side - defaulting to left");
        Ok(true)
    }
    /// Join two tokens into a single token
    pub(super) fn join_tokens(&self, left_token: &Token, right_token: &Token) -> Result<Token> {
        let mut joined_token = Token::new();
        joined_token.bindings.extend(left_token.bindings.clone());
        for (var, value) in &right_token.bindings {
            if let Some(existing_value) = joined_token.bindings.get(var) {
                if !self.terms_equal(existing_value, value) {
                    return Err(anyhow::anyhow!("Binding conflict for variable {}", var));
                }
            } else {
                joined_token.bindings.insert(var.clone(), value.clone());
            }
        }
        joined_token.tags.extend(left_token.tags.clone());
        joined_token.tags.extend(right_token.tags.clone());
        joined_token.facts.extend(left_token.facts.clone());
        joined_token.facts.extend(right_token.facts.clone());
        Ok(joined_token)
    }
    /// Execute a production node
    pub(super) fn execute_production(
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
        for head_atom in rule_head {
            let instantiated = self.apply_substitution(head_atom, &token.bindings)?;
            derived_facts.push(instantiated);
        }
        if self.debug_mode && !derived_facts.is_empty() {
            debug!(
                "Production '{}' derived {} facts using bindings {:?}",
                rule_name,
                derived_facts.len(),
                token.bindings
            );
        }
        Ok(derived_facts)
    }
    /// Remove a fact from the network
    pub fn remove_fact(&mut self, fact: &RuleAtom) -> Result<Vec<RuleAtom>> {
        if self.debug_mode {
            debug!("Removing fact from RETE network: {:?}", fact);
        }
        let retracted_facts = Vec::new();
        let mut matching_alphas = Vec::new();
        for (&node_id, node) in &self.nodes {
            if let ReteNode::Alpha { pattern, .. } = node {
                if self.unify_atoms(pattern, fact, &HashMap::new())?.is_some() {
                    matching_alphas.push(node_id);
                }
            }
        }
        for alpha_id in matching_alphas {
            if let Some(memory) = self.alpha_memory.get_mut(&alpha_id) {
                memory.remove(fact);
            }
        }
        Ok(retracted_facts)
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
