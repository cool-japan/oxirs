//! Rule Composition - Modules, Inheritance, and Templates
//!
//! Provides advanced rule composition capabilities for organizing, reusing, and
//! managing complex rule sets.
//!
//! # Features
//!
//! - **Rule Modules**: Organize rules into packages with namespaces
//! - **Rule Inheritance**: Create derived rules that extend base rules
//! - **Rule Templates**: Parameterized rules for code reuse
//! - **Import/Export**: Module-level import and dependency management
//! - **Validation**: Ensure composition constraints are satisfied
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::composition::{RuleModule, RuleTemplate, CompositionManager};
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! let mut manager = CompositionManager::new();
//!
//! // Create a module
//! let mut module = RuleModule::new("ontology".to_string());
//! module.add_rule(Rule {
//!     name: "subclass_transitivity".to_string(),
//!     body: vec![],
//!     head: vec![],
//! });
//!
//! // Register the module
//! manager.register_module(module).unwrap();
//!
//! // Create a template
//! let template = RuleTemplate::new(
//!     "property_domain".to_string(),
//!     vec!["property".to_string(), "class".to_string()],
//! );
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

/// Rule module for organizing related rules
#[derive(Debug, Clone)]
pub struct RuleModule {
    /// Module name/identifier
    pub name: String,
    /// Module version
    pub version: String,
    /// Module description
    pub description: String,
    /// Rules in this module
    pub rules: Vec<Rule>,
    /// Modules this module depends on
    pub dependencies: Vec<String>,
    /// Module metadata
    pub metadata: HashMap<String, String>,
    /// Module namespace
    pub namespace: String,
}

impl RuleModule {
    /// Create a new rule module
    pub fn new(name: String) -> Self {
        Self {
            name: name.clone(),
            version: "1.0.0".to_string(),
            description: String::new(),
            rules: Vec::new(),
            dependencies: Vec::new(),
            metadata: HashMap::new(),
            namespace: name,
        }
    }

    /// Create a module with specific version
    pub fn with_version(name: String, version: String) -> Self {
        Self {
            name,
            version,
            description: String::new(),
            rules: Vec::new(),
            dependencies: Vec::new(),
            metadata: HashMap::new(),
            namespace: String::new(),
        }
    }

    /// Add a rule to the module
    pub fn add_rule(&mut self, rule: Rule) {
        debug!("Adding rule '{}' to module '{}'", rule.name, self.name);
        self.rules.push(rule);
    }

    /// Add multiple rules
    pub fn add_rules(&mut self, rules: Vec<Rule>) {
        for rule in rules {
            self.add_rule(rule);
        }
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, module_name: String) {
        if !self.dependencies.contains(&module_name) {
            debug!("Module '{}' now depends on '{}'", self.name, module_name);
            self.dependencies.push(module_name);
        }
    }

    /// Set module description
    pub fn set_description(&mut self, description: String) {
        self.description = description;
    }

    /// Set namespace
    pub fn set_namespace(&mut self, namespace: String) {
        self.namespace = namespace;
    }

    /// Get all rules in this module
    pub fn get_rules(&self) -> &[Rule] {
        &self.rules
    }

    /// Find a rule by name
    pub fn find_rule(&self, name: &str) -> Option<&Rule> {
        self.rules.iter().find(|r| r.name == name)
    }

    /// Remove a rule by name
    pub fn remove_rule(&mut self, name: &str) -> Option<Rule> {
        if let Some(pos) = self.rules.iter().position(|r| r.name == name) {
            Some(self.rules.remove(pos))
        } else {
            None
        }
    }

    /// Get module statistics
    pub fn get_stats(&self) -> ModuleStats {
        ModuleStats {
            name: self.name.clone(),
            version: self.version.clone(),
            rule_count: self.rules.len(),
            dependency_count: self.dependencies.len(),
        }
    }
}

/// Module statistics
#[derive(Debug, Clone)]
pub struct ModuleStats {
    pub name: String,
    pub version: String,
    pub rule_count: usize,
    pub dependency_count: usize,
}

/// Rule template for parameterized rules
#[derive(Debug, Clone)]
pub struct RuleTemplate {
    /// Template name
    pub name: String,
    /// Template parameters
    pub parameters: Vec<String>,
    /// Template body (with parameter placeholders)
    pub body_template: Vec<RuleAtom>,
    /// Template head (with parameter placeholders)
    pub head_template: Vec<RuleAtom>,
    /// Template description
    pub description: String,
}

impl RuleTemplate {
    /// Create a new rule template
    pub fn new(name: String, parameters: Vec<String>) -> Self {
        Self {
            name,
            parameters,
            body_template: Vec::new(),
            head_template: Vec::new(),
            description: String::new(),
        }
    }

    /// Set template body
    pub fn set_body(&mut self, body: Vec<RuleAtom>) {
        self.body_template = body;
    }

    /// Set template head
    pub fn set_head(&mut self, head: Vec<RuleAtom>) {
        self.head_template = head;
    }

    /// Set description
    pub fn set_description(&mut self, description: String) {
        self.description = description;
    }

    /// Instantiate template with concrete values
    pub fn instantiate(&self, name: String, args: &HashMap<String, Term>) -> Result<Rule> {
        // Validate arguments
        for param in &self.parameters {
            if !args.contains_key(param) {
                return Err(anyhow::anyhow!("Missing template parameter: {}", param));
            }
        }

        // Substitute parameters in body
        let body = self
            .body_template
            .iter()
            .map(|atom| self.substitute_atom(atom, args))
            .collect::<Result<Vec<_>>>()?;

        // Substitute parameters in head
        let head = self
            .head_template
            .iter()
            .map(|atom| self.substitute_atom(atom, args))
            .collect::<Result<Vec<_>>>()?;

        Ok(Rule { name, body, head })
    }

    fn substitute_atom(&self, atom: &RuleAtom, args: &HashMap<String, Term>) -> Result<RuleAtom> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => Ok(RuleAtom::Triple {
                subject: self.substitute_term(subject, args)?,
                predicate: self.substitute_term(predicate, args)?,
                object: self.substitute_term(object, args)?,
            }),
            RuleAtom::Builtin {
                name,
                args: builtin_args,
            } => Ok(RuleAtom::Builtin {
                name: name.clone(),
                args: builtin_args
                    .iter()
                    .map(|a| self.substitute_term(a, args))
                    .collect::<Result<Vec<_>>>()?,
            }),
            RuleAtom::GreaterThan { left, right } => Ok(RuleAtom::GreaterThan {
                left: self.substitute_term(left, args)?,
                right: self.substitute_term(right, args)?,
            }),
            RuleAtom::LessThan { left, right } => Ok(RuleAtom::LessThan {
                left: self.substitute_term(left, args)?,
                right: self.substitute_term(right, args)?,
            }),
            RuleAtom::NotEqual { left, right } => Ok(RuleAtom::NotEqual {
                left: self.substitute_term(left, args)?,
                right: self.substitute_term(right, args)?,
            }),
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    fn substitute_term(&self, term: &Term, args: &HashMap<String, Term>) -> Result<Term> {
        match term {
            Term::Constant(c) => {
                // Check if this is a parameter placeholder (e.g., ${param})
                if c.starts_with("${") && c.ends_with('}') {
                    let param_name = &c[2..c.len() - 1];
                    if let Some(value) = args.get(param_name) {
                        Ok(value.clone())
                    } else {
                        Err(anyhow::anyhow!(
                            "Unknown template parameter: {}",
                            param_name
                        ))
                    }
                } else {
                    Ok(term.clone())
                }
            }
            Term::Function {
                name,
                args: func_args,
            } => {
                let substituted_args = func_args
                    .iter()
                    .map(|a| self.substitute_term(a, args))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Term::Function {
                    name: name.clone(),
                    args: substituted_args,
                })
            }
            _ => Ok(term.clone()),
        }
    }
}

/// Rule inheritance for creating derived rules
#[derive(Debug, Clone)]
pub struct RuleInheritance {
    /// Base rule name
    pub base_rule: String,
    /// Derived rule name
    pub derived_rule: String,
    /// Additional body atoms
    pub additional_body: Vec<RuleAtom>,
    /// Additional head atoms
    pub additional_head: Vec<RuleAtom>,
    /// Override body (if Some, replaces base body)
    pub override_body: Option<Vec<RuleAtom>>,
    /// Override head (if Some, replaces base head)
    pub override_head: Option<Vec<RuleAtom>>,
}

impl RuleInheritance {
    /// Create new inheritance relationship
    pub fn new(base_rule: String, derived_rule: String) -> Self {
        Self {
            base_rule,
            derived_rule,
            additional_body: Vec::new(),
            additional_head: Vec::new(),
            override_body: None,
            override_head: None,
        }
    }

    /// Add additional body atoms
    pub fn add_body_atoms(&mut self, atoms: Vec<RuleAtom>) {
        self.additional_body.extend(atoms);
    }

    /// Add additional head atoms
    pub fn add_head_atoms(&mut self, atoms: Vec<RuleAtom>) {
        self.additional_head.extend(atoms);
    }

    /// Set override body (completely replaces base body)
    pub fn set_body_override(&mut self, body: Vec<RuleAtom>) {
        self.override_body = Some(body);
    }

    /// Set override head (completely replaces base head)
    pub fn set_head_override(&mut self, head: Vec<RuleAtom>) {
        self.override_head = Some(head);
    }

    /// Create derived rule from base rule
    pub fn derive(&self, base: &Rule) -> Result<Rule> {
        let body = if let Some(override_body) = &self.override_body {
            override_body.clone()
        } else {
            let mut body = base.body.clone();
            body.extend(self.additional_body.clone());
            body
        };

        let head = if let Some(override_head) = &self.override_head {
            override_head.clone()
        } else {
            let mut head = base.head.clone();
            head.extend(self.additional_head.clone());
            head
        };

        Ok(Rule {
            name: self.derived_rule.clone(),
            body,
            head,
        })
    }
}

/// Composition manager for modules, templates, and inheritance
pub struct CompositionManager {
    /// Registered modules
    modules: HashMap<String, RuleModule>,
    /// Registered templates
    templates: HashMap<String, RuleTemplate>,
    /// Inheritance relationships
    inheritances: Vec<RuleInheritance>,
    /// Module dependency graph
    dependency_graph: HashMap<String, HashSet<String>>,
}

impl Default for CompositionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositionManager {
    /// Create a new composition manager
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            templates: HashMap::new(),
            inheritances: Vec::new(),
            dependency_graph: HashMap::new(),
        }
    }

    /// Register a module
    pub fn register_module(&mut self, module: RuleModule) -> Result<()> {
        let name = module.name.clone();

        // Check for conflicts
        if self.modules.contains_key(&name) {
            return Err(anyhow::anyhow!("Module '{}' already registered", name));
        }

        // Update dependency graph
        for dep in &module.dependencies {
            self.dependency_graph
                .entry(name.clone())
                .or_default()
                .insert(dep.clone());
        }

        info!("Registered module '{}' (v{})", name, module.version);
        self.modules.insert(name, module);

        Ok(())
    }

    /// Get a module by name
    pub fn get_module(&self, name: &str) -> Option<&RuleModule> {
        self.modules.get(name)
    }

    /// Get all modules
    pub fn get_all_modules(&self) -> Vec<&RuleModule> {
        self.modules.values().collect()
    }

    /// Remove a module
    pub fn remove_module(&mut self, name: &str) -> Option<RuleModule> {
        self.dependency_graph.remove(name);
        self.modules.remove(name)
    }

    /// Register a template
    pub fn register_template(&mut self, template: RuleTemplate) -> Result<()> {
        let name = template.name.clone();

        if self.templates.contains_key(&name) {
            return Err(anyhow::anyhow!("Template '{}' already registered", name));
        }

        debug!("Registered template '{}'", name);
        self.templates.insert(name, template);

        Ok(())
    }

    /// Get a template by name
    pub fn get_template(&self, name: &str) -> Option<&RuleTemplate> {
        self.templates.get(name)
    }

    /// Instantiate a template
    pub fn instantiate_template(
        &self,
        template_name: &str,
        rule_name: String,
        args: &HashMap<String, Term>,
    ) -> Result<Rule> {
        let template = self
            .templates
            .get(template_name)
            .ok_or_else(|| anyhow::anyhow!("Template '{}' not found", template_name))?;

        template.instantiate(rule_name, args)
    }

    /// Register an inheritance relationship
    pub fn register_inheritance(&mut self, inheritance: RuleInheritance) {
        debug!(
            "Registered inheritance: {} -> {}",
            inheritance.base_rule, inheritance.derived_rule
        );
        self.inheritances.push(inheritance);
    }

    /// Get all rules from all modules
    pub fn get_all_rules(&self) -> Vec<Rule> {
        let mut rules = Vec::new();
        for module in self.modules.values() {
            rules.extend(module.rules.clone());
        }
        rules
    }

    /// Get rules from a specific module
    pub fn get_module_rules(&self, module_name: &str) -> Option<Vec<Rule>> {
        self.modules.get(module_name).map(|m| m.rules.clone())
    }

    /// Apply all inheritance relationships
    pub fn apply_inheritances(&self) -> Result<Vec<Rule>> {
        let mut derived_rules = Vec::new();

        for inheritance in &self.inheritances {
            // Find base rule across all modules
            let base_rule = self.find_rule(&inheritance.base_rule).ok_or_else(|| {
                anyhow::anyhow!("Base rule '{}' not found", inheritance.base_rule)
            })?;

            let derived = inheritance.derive(&base_rule)?;
            derived_rules.push(derived);
        }

        Ok(derived_rules)
    }

    /// Find a rule by name across all modules
    fn find_rule(&self, name: &str) -> Option<Rule> {
        for module in self.modules.values() {
            if let Some(rule) = module.find_rule(name) {
                return Some(rule.clone());
            }
        }
        None
    }

    /// Check for dependency cycles
    pub fn check_dependency_cycles(&self) -> Result<()> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for module_name in self.modules.keys() {
            if self.has_cycle(module_name, &mut visited, &mut rec_stack) {
                return Err(anyhow::anyhow!(
                    "Circular dependency detected involving module '{}'",
                    module_name
                ));
            }
        }

        Ok(())
    }

    fn has_cycle(
        &self,
        module: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        if rec_stack.contains(module) {
            return true;
        }

        if visited.contains(module) {
            return false;
        }

        visited.insert(module.to_string());
        rec_stack.insert(module.to_string());

        if let Some(deps) = self.dependency_graph.get(module) {
            for dep in deps {
                if self.has_cycle(dep, visited, rec_stack) {
                    return true;
                }
            }
        }

        rec_stack.remove(module);
        false
    }

    /// Get module load order (topological sort of dependencies)
    pub fn get_load_order(&self) -> Result<Vec<String>> {
        self.check_dependency_cycles()?;

        // Build reverse dependency graph (who depends on me)
        let mut reverse_deps: HashMap<String, Vec<String>> = HashMap::new();
        let mut out_degree: HashMap<String, usize> = HashMap::new();

        // Initialize all modules with 0 out-degree
        for module in self.modules.keys() {
            out_degree.insert(module.clone(), 0);
        }

        // Calculate out-degrees and build reverse graph
        for (module, deps) in &self.dependency_graph {
            out_degree.insert(module.clone(), deps.len());
            for dep in deps {
                reverse_deps
                    .entry(dep.clone())
                    .or_default()
                    .push(module.clone());
            }
        }

        // Find modules with no dependencies (out-degree = 0)
        let mut queue: Vec<String> = out_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(name, _)| name.clone())
            .collect();

        let mut result = Vec::new();

        while let Some(module) = queue.pop() {
            result.push(module.clone());

            // For each module that depends on this one, decrease its out-degree
            if let Some(dependents) = reverse_deps.get(&module) {
                for dependent in dependents {
                    if let Some(degree) = out_degree.get_mut(dependent) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(dependent.clone());
                        }
                    }
                }
            }
        }

        if result.len() != self.modules.len() {
            return Err(anyhow::anyhow!("Failed to determine module load order"));
        }

        Ok(result)
    }

    /// Get composition statistics
    pub fn get_stats(&self) -> CompositionStats {
        let total_rules: usize = self.modules.values().map(|m| m.rules.len()).sum();

        CompositionStats {
            total_modules: self.modules.len(),
            total_templates: self.templates.len(),
            total_inheritances: self.inheritances.len(),
            total_rules,
        }
    }
}

/// Composition statistics
#[derive(Debug, Clone)]
pub struct CompositionStats {
    pub total_modules: usize,
    pub total_templates: usize,
    pub total_inheritances: usize,
    pub total_rules: usize,
}

impl std::fmt::Display for CompositionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Modules: {}, Templates: {}, Inheritances: {}, Total Rules: {}",
            self.total_modules, self.total_templates, self.total_inheritances, self.total_rules
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_creation() {
        let mut module = RuleModule::new("test".to_string());
        module.set_description("Test module".to_string());

        assert_eq!(module.name, "test");
        assert_eq!(module.description, "Test module");
        assert!(module.rules.is_empty());
    }

    #[test]
    fn test_module_add_rules() {
        let mut module = RuleModule::new("test".to_string());

        let rule = Rule {
            name: "rule1".to_string(),
            body: vec![],
            head: vec![],
        };

        module.add_rule(rule);
        assert_eq!(module.rules.len(), 1);
    }

    #[test]
    fn test_template_instantiation() {
        let mut template = RuleTemplate::new(
            "property_domain".to_string(),
            vec!["property".to_string(), "class".to_string()],
        );

        template.set_body(vec![RuleAtom::Triple {
            subject: Term::Variable("x".to_string()),
            predicate: Term::Constant("${property}".to_string()),
            object: Term::Variable("y".to_string()),
        }]);

        template.set_head(vec![RuleAtom::Triple {
            subject: Term::Variable("x".to_string()),
            predicate: Term::Constant("rdf:type".to_string()),
            object: Term::Constant("${class}".to_string()),
        }]);

        let mut args = HashMap::new();
        args.insert(
            "property".to_string(),
            Term::Constant("foaf:name".to_string()),
        );
        args.insert(
            "class".to_string(),
            Term::Constant("foaf:Person".to_string()),
        );

        let rule = template
            .instantiate("test_rule".to_string(), &args)
            .unwrap();

        assert_eq!(rule.name, "test_rule");
        assert_eq!(rule.body.len(), 1);
        assert_eq!(rule.head.len(), 1);
    }

    #[test]
    fn test_inheritance() {
        let base_rule = Rule {
            name: "base".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("x".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Person".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("x".to_string()),
                predicate: Term::Constant("category".to_string()),
                object: Term::Constant("human".to_string()),
            }],
        };

        let mut inheritance = RuleInheritance::new("base".to_string(), "derived".to_string());
        inheritance.add_body_atoms(vec![RuleAtom::Triple {
            subject: Term::Variable("x".to_string()),
            predicate: Term::Constant("age".to_string()),
            object: Term::Variable("age".to_string()),
        }]);

        let derived = inheritance.derive(&base_rule).unwrap();

        assert_eq!(derived.name, "derived");
        assert_eq!(derived.body.len(), 2); // Base + additional
        assert_eq!(derived.head.len(), 1); // Same as base
    }

    #[test]
    fn test_composition_manager() {
        let mut manager = CompositionManager::new();

        let module = RuleModule::new("test".to_string());
        manager.register_module(module).unwrap();

        assert_eq!(manager.modules.len(), 1);
        assert!(manager.get_module("test").is_some());
    }

    #[test]
    fn test_dependency_cycles() {
        let mut manager = CompositionManager::new();

        let mut module_a = RuleModule::new("A".to_string());
        module_a.add_dependency("B".to_string());

        let mut module_b = RuleModule::new("B".to_string());
        module_b.add_dependency("A".to_string());

        manager.register_module(module_a).unwrap();
        manager.register_module(module_b).unwrap();

        // Should detect cycle
        assert!(manager.check_dependency_cycles().is_err());
    }

    #[test]
    fn test_load_order() {
        let mut manager = CompositionManager::new();

        let mut module_a = RuleModule::new("A".to_string());
        module_a.add_dependency("B".to_string());

        let module_b = RuleModule::new("B".to_string());

        manager.register_module(module_b).unwrap();
        manager.register_module(module_a).unwrap();

        let order = manager.get_load_order().unwrap();

        // B should come before A
        let b_index = order.iter().position(|m| m == "B").unwrap();
        let a_index = order.iter().position(|m| m == "A").unwrap();
        assert!(b_index < a_index);
    }
}
