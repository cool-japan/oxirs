//! Simplified Rete algorithm network for forward-chain rule matching.
//!
//! The Rete algorithm efficiently evaluates many conditions against a working memory
//! by sharing common sub-condition evaluations in an alpha-beta network.

use std::collections::HashMap;

// ─────────────────────────────────────────────────
// WME – Working Memory Element (a single fact/triple)
// ─────────────────────────────────────────────────

/// A Working Memory Element: a triple of (subject, predicate, object).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WME {
    pub fields: [String; 3],
}

impl WME {
    /// Construct a new WME.
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        WME {
            fields: [s.into(), p.into(), o.into()],
        }
    }

    /// Subject field.
    pub fn subject(&self) -> &str {
        &self.fields[0]
    }

    /// Predicate field.
    pub fn predicate(&self) -> &str {
        &self.fields[1]
    }

    /// Object field.
    pub fn object(&self) -> &str {
        &self.fields[2]
    }

    /// Check whether this WME satisfies the given condition.
    pub fn matches_condition(&self, cond: &Condition) -> bool {
        let field_val = match cond.field {
            CondField::Subject => &self.fields[0],
            CondField::Predicate => &self.fields[1],
            CondField::Object => &self.fields[2],
        };
        match &cond.test {
            CondTest::Constant(c) => field_val == c,
            CondTest::Variable(_) => true, // variables always match
            CondTest::Any => true,
        }
    }
}

// ─────────────────────────────────────────────────
// Condition and helpers
// ─────────────────────────────────────────────────

/// Which field of a WME a condition applies to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CondField {
    Subject,
    Predicate,
    Object,
}

/// The kind of test a condition performs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CondTest {
    /// Must equal an exact constant.
    Constant(String),
    /// Binds/tests a named variable (always matches, but records binding).
    Variable(String),
    /// Wildcard – matches anything.
    Any,
}

/// A single condition in a production's LHS.
#[derive(Debug, Clone)]
pub struct Condition {
    pub field: CondField,
    pub test: CondTest,
}

impl Condition {
    /// Shorthand: constant-test condition on a field.
    pub fn constant(field: CondField, value: impl Into<String>) -> Self {
        Condition {
            field,
            test: CondTest::Constant(value.into()),
        }
    }

    /// Shorthand: variable-binding condition on a field.
    pub fn variable(field: CondField, name: impl Into<String>) -> Self {
        Condition {
            field,
            test: CondTest::Variable(name.into()),
        }
    }

    /// Shorthand: wildcard condition on a field.
    pub fn any(field: CondField) -> Self {
        Condition {
            field,
            test: CondTest::Any,
        }
    }
}

// ─────────────────────────────────────────────────
// Production – a rule
// ─────────────────────────────────────────────────

/// A production rule: when all its conditions are satisfied by WMEs, it fires.
#[derive(Debug, Clone)]
pub struct Production {
    pub id: String,
    pub conditions: Vec<Condition>,
    pub action: String,
}

impl Production {
    /// Construct a new production.
    pub fn new(
        id: impl Into<String>,
        conditions: Vec<Condition>,
        action: impl Into<String>,
    ) -> Self {
        Production {
            id: id.into(),
            conditions,
            action: action.into(),
        }
    }
}

// ─────────────────────────────────────────────────
// Alpha memory – WMEs that passed a single-condition test
// ─────────────────────────────────────────────────

/// Stores WMEs that have matched a single alpha-network node.
#[derive(Debug, Clone, Default)]
pub struct AlphaMemory {
    pub wmes: Vec<WME>,
}

impl AlphaMemory {
    pub fn new() -> Self {
        AlphaMemory { wmes: Vec::new() }
    }

    pub fn add(&mut self, wme: WME) {
        self.wmes.push(wme);
    }

    /// Remove a WME by value; returns `true` if found and removed.
    pub fn remove(&mut self, wme: &WME) -> bool {
        if let Some(pos) = self.wmes.iter().position(|w| w == wme) {
            self.wmes.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        self.wmes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.wmes.is_empty()
    }
}

// ─────────────────────────────────────────────────
// Token – a partial match through beta network
// ─────────────────────────────────────────────────

/// A partial (or complete) match through the beta network.
#[derive(Debug, Clone)]
pub struct Token {
    pub wmes: Vec<WME>,
    /// Variable bindings accumulated by this token.
    pub bindings: HashMap<String, String>,
}

impl Token {
    pub fn new() -> Self {
        Token {
            wmes: Vec::new(),
            bindings: HashMap::new(),
        }
    }

    /// Extend this token with a WME and any new variable bindings.
    pub fn extend(&self, wme: WME, new_bindings: HashMap<String, String>) -> Token {
        let mut t = self.clone();
        t.wmes.push(wme);
        t.bindings.extend(new_bindings);
        t
    }
}

impl Default for Token {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────
// Beta memory – partial matches
// ─────────────────────────────────────────────────

/// Stores partial matches (tokens) for the beta network.
#[derive(Debug, Clone, Default)]
pub struct BetaMemory {
    pub tokens: Vec<Token>,
}

impl BetaMemory {
    pub fn new() -> Self {
        BetaMemory { tokens: Vec::new() }
    }
}

// ─────────────────────────────────────────────────
// ReteNetwork – the top-level Rete network
// ─────────────────────────────────────────────────

/// A simplified Rete forward-chaining network.
///
/// The network maintains alpha memories (per-condition WME sets) and a
/// set of productions.  When a WME is added, the network propagates it
/// through the alpha memories and checks each production for complete matches.
pub struct ReteNetwork {
    alpha_memories: HashMap<String, AlphaMemory>,
    beta_memories: Vec<BetaMemory>,
    productions: Vec<Production>,
    all_wmes: Vec<WME>,
}

impl Default for ReteNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl ReteNetwork {
    /// Create an empty Rete network.
    pub fn new() -> Self {
        ReteNetwork {
            alpha_memories: HashMap::new(),
            beta_memories: Vec::new(),
            productions: Vec::new(),
            all_wmes: Vec::new(),
        }
    }

    /// Add a production to the network.
    pub fn add_production(&mut self, production: Production) {
        self.productions.push(production);
    }

    /// Add a WME to the working memory.
    ///
    /// Returns the ids of all productions that were fully triggered by the
    /// current state of the working memory after this WME is added.
    pub fn add_wme(&mut self, wme: WME) -> Vec<String> {
        self.all_wmes.push(wme.clone());

        // Update alpha memories for every condition in every production
        for prod in &self.productions {
            for cond in &prod.conditions {
                if wme.matches_condition(cond) {
                    let key = alpha_key(cond);
                    self.alpha_memories.entry(key).or_default().add(wme.clone());
                }
            }
        }

        // Check which productions now have complete matches
        self.evaluate_productions()
    }

    /// Remove a WME from working memory.
    ///
    /// Returns `true` if the WME was found and removed.
    pub fn remove_wme(&mut self, wme: &WME) -> bool {
        let pos = self.all_wmes.iter().position(|w| w == wme);
        if let Some(p) = pos {
            self.all_wmes.remove(p);
            // Remove from all alpha memories that contain it
            for am in self.alpha_memories.values_mut() {
                am.remove(wme);
            }
            true
        } else {
            false
        }
    }

    /// Number of productions registered.
    pub fn production_count(&self) -> usize {
        self.productions.len()
    }

    /// Number of WMEs in working memory.
    pub fn wme_count(&self) -> usize {
        self.all_wmes.len()
    }

    /// Retrieve the alpha memory for a given key.
    pub fn get_alpha_memory(&self, key: &str) -> Option<&AlphaMemory> {
        self.alpha_memories.get(key)
    }

    /// Clear all WMEs and alpha memory contents (productions are retained).
    pub fn clear(&mut self) {
        self.all_wmes.clear();
        self.alpha_memories.clear();
        self.beta_memories.clear();
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Evaluate every production against the current alpha memories.
    ///
    /// Returns the ids of fully triggered productions.
    fn evaluate_productions(&self) -> Vec<String> {
        let mut triggered = Vec::new();

        'prod: for prod in &self.productions {
            if prod.conditions.is_empty() {
                triggered.push(prod.id.clone());
                continue;
            }

            // Try to find a consistent set of WMEs satisfying all conditions
            let first_cond = &prod.conditions[0];
            let candidates = self.wmes_matching_condition(first_cond);

            for first_wme in candidates {
                if let Some(token) =
                    self.try_match_conditions(&prod.conditions, 0, Token::new(), first_wme)
                {
                    let _ = token; // token contains full bindings, but we only report id
                    triggered.push(prod.id.clone());
                    continue 'prod; // report each production at most once per add_wme call
                }
            }
        }

        triggered
    }

    /// Recursively attempt to match conditions `conds[depth..]` starting
    /// from `wme` being bound at `depth`.
    fn try_match_conditions(
        &self,
        conds: &[Condition],
        depth: usize,
        token: Token,
        wme: &WME,
    ) -> Option<Token> {
        if !wme.matches_condition(&conds[depth]) {
            return None;
        }

        // Collect variable bindings for this condition / WME pair
        let new_bindings = collect_bindings(&conds[depth], wme);

        // Check consistency with existing bindings in the token
        for (var, val) in &new_bindings {
            if let Some(existing) = token.bindings.get(var.as_str()) {
                if existing != val {
                    return None; // variable conflict
                }
            }
        }

        let new_token = token.extend(wme.clone(), new_bindings);

        if depth + 1 == conds.len() {
            return Some(new_token);
        }

        // Recurse for the next condition
        let next_cond = &conds[depth + 1];
        for candidate in self.wmes_matching_condition(next_cond) {
            if let Some(t) =
                self.try_match_conditions(conds, depth + 1, new_token.clone(), candidate)
            {
                return Some(t);
            }
        }
        None
    }

    /// Return all WMEs in working memory that match the condition.
    fn wmes_matching_condition<'a>(&'a self, cond: &Condition) -> Vec<&'a WME> {
        self.all_wmes
            .iter()
            .filter(|w| w.matches_condition(cond))
            .collect()
    }
}

/// Compute the canonical alpha-memory key for a condition.
fn alpha_key(cond: &Condition) -> String {
    let field = match cond.field {
        CondField::Subject => "s",
        CondField::Predicate => "p",
        CondField::Object => "o",
    };
    let test = match &cond.test {
        CondTest::Constant(c) => format!("={c}"),
        CondTest::Variable(v) => format!("?{v}"),
        CondTest::Any => "*".to_string(),
    };
    format!("{field}:{test}")
}

/// Extract variable bindings from a single condition matched against a WME.
fn collect_bindings(cond: &Condition, wme: &WME) -> HashMap<String, String> {
    let mut bindings = HashMap::new();
    if let CondTest::Variable(name) = &cond.test {
        let value = match cond.field {
            CondField::Subject => wme.subject(),
            CondField::Predicate => wme.predicate(),
            CondField::Object => wme.object(),
        };
        bindings.insert(name.clone(), value.to_string());
    }
    bindings
}

// ─────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn wme(s: &str, p: &str, o: &str) -> WME {
        WME::new(s, p, o)
    }

    fn const_cond(field: CondField, val: &str) -> Condition {
        Condition::constant(field, val)
    }

    fn var_cond(field: CondField, name: &str) -> Condition {
        Condition::variable(field, name)
    }

    fn any_cond(field: CondField) -> Condition {
        Condition::any(field)
    }

    // ── WME tests ──────────────────────────────────────────────

    #[test]
    fn test_wme_fields() {
        let w = wme("alice", "knows", "bob");
        assert_eq!(w.subject(), "alice");
        assert_eq!(w.predicate(), "knows");
        assert_eq!(w.object(), "bob");
        assert_eq!(w.fields[0], "alice");
        assert_eq!(w.fields[1], "knows");
        assert_eq!(w.fields[2], "bob");
    }

    #[test]
    fn test_wme_matches_constant_subject() {
        let w = wme("alice", "knows", "bob");
        let c = const_cond(CondField::Subject, "alice");
        assert!(w.matches_condition(&c));
    }

    #[test]
    fn test_wme_no_match_constant_subject() {
        let w = wme("alice", "knows", "bob");
        let c = const_cond(CondField::Subject, "carol");
        assert!(!w.matches_condition(&c));
    }

    #[test]
    fn test_wme_matches_variable_always() {
        let w = wme("alice", "knows", "bob");
        let c = var_cond(CondField::Subject, "x");
        assert!(w.matches_condition(&c));
    }

    #[test]
    fn test_wme_matches_any_always() {
        let w = wme("alice", "knows", "bob");
        assert!(w.matches_condition(&any_cond(CondField::Subject)));
        assert!(w.matches_condition(&any_cond(CondField::Predicate)));
        assert!(w.matches_condition(&any_cond(CondField::Object)));
    }

    #[test]
    fn test_wme_matches_constant_predicate() {
        let w = wme("a", "type", "Person");
        assert!(w.matches_condition(&const_cond(CondField::Predicate, "type")));
        assert!(!w.matches_condition(&const_cond(CondField::Predicate, "label")));
    }

    #[test]
    fn test_wme_matches_constant_object() {
        let w = wme("a", "type", "Person");
        assert!(w.matches_condition(&const_cond(CondField::Object, "Person")));
        assert!(!w.matches_condition(&const_cond(CondField::Object, "Animal")));
    }

    // ── ReteNetwork basic operations ──────────────────────────

    #[test]
    fn test_new_network_empty() {
        let net = ReteNetwork::new();
        assert_eq!(net.production_count(), 0);
        assert_eq!(net.wme_count(), 0);
    }

    #[test]
    fn test_add_production_increases_count() {
        let mut net = ReteNetwork::new();
        net.add_production(Production::new("p1", vec![], "action1"));
        assert_eq!(net.production_count(), 1);
    }

    #[test]
    fn test_add_wme_increases_count() {
        let mut net = ReteNetwork::new();
        net.add_wme(wme("a", "b", "c"));
        assert_eq!(net.wme_count(), 1);
    }

    #[test]
    fn test_remove_wme_decreases_count() {
        let mut net = ReteNetwork::new();
        let w = wme("a", "b", "c");
        net.add_wme(w.clone());
        assert_eq!(net.wme_count(), 1);
        assert!(net.remove_wme(&w));
        assert_eq!(net.wme_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_wme_returns_false() {
        let mut net = ReteNetwork::new();
        let w = wme("x", "y", "z");
        assert!(!net.remove_wme(&w));
    }

    #[test]
    fn test_clear_removes_all_wmes() {
        let mut net = ReteNetwork::new();
        net.add_wme(wme("a", "b", "c"));
        net.add_wme(wme("d", "e", "f"));
        net.clear();
        assert_eq!(net.wme_count(), 0);
    }

    #[test]
    fn test_clear_keeps_productions() {
        let mut net = ReteNetwork::new();
        net.add_production(Production::new("p1", vec![], "a"));
        net.add_wme(wme("a", "b", "c"));
        net.clear();
        assert_eq!(net.production_count(), 1);
        assert_eq!(net.wme_count(), 0);
    }

    // ── Single condition production ────────────────────────────

    #[test]
    fn test_single_cond_production_fires() {
        let mut net = ReteNetwork::new();
        let cond = const_cond(CondField::Predicate, "type");
        net.add_production(Production::new("p_type", vec![cond], "assert type"));
        let triggered = net.add_wme(wme("alice", "type", "Person"));
        assert!(triggered.contains(&"p_type".to_string()));
    }

    #[test]
    fn test_single_cond_production_no_fire_on_mismatch() {
        let mut net = ReteNetwork::new();
        let cond = const_cond(CondField::Predicate, "type");
        net.add_production(Production::new("p_type", vec![cond], "assert type"));
        let triggered = net.add_wme(wme("alice", "label", "Alice"));
        assert!(!triggered.contains(&"p_type".to_string()));
    }

    #[test]
    fn test_variable_cond_always_fires() {
        let mut net = ReteNetwork::new();
        let cond = var_cond(CondField::Subject, "x");
        net.add_production(Production::new("p_any_s", vec![cond], "any subject"));
        let triggered = net.add_wme(wme("whatever", "p", "o"));
        assert!(triggered.contains(&"p_any_s".to_string()));
    }

    #[test]
    fn test_any_cond_always_fires() {
        let mut net = ReteNetwork::new();
        let cond = any_cond(CondField::Predicate);
        net.add_production(Production::new("p_wildcard", vec![cond], "any"));
        let triggered = net.add_wme(wme("s", "anything", "o"));
        assert!(triggered.contains(&"p_wildcard".to_string()));
    }

    // ── Multi-condition production ─────────────────────────────

    #[test]
    fn test_two_cond_production_fires_when_both_satisfied() {
        let mut net = ReteNetwork::new();
        let conds = vec![
            const_cond(CondField::Predicate, "type"),
            const_cond(CondField::Object, "Person"),
        ];
        net.add_production(Production::new("p_person", conds, "is person"));

        net.add_wme(wme("alice", "type", "Person"));
        let triggered = net.add_wme(wme("alice", "type", "Person")); // same WME again isn't reused
                                                                     // Actually the network checks all wmes, so just adding the second wme triggers it
        let _ = triggered;
        // Check: after alice type Person added, let's re-query by clearing and re-adding
        net.clear();
        let t1 = net.add_wme(wme("alice", "type", "Person"));
        assert!(t1.contains(&"p_person".to_string()));
    }

    #[test]
    fn test_two_cond_production_needs_two_wmes() {
        let mut net = ReteNetwork::new();
        // Production: ?x type Person, ?x knows ?y
        let conds = vec![
            const_cond(CondField::Predicate, "type"),
            const_cond(CondField::Predicate, "knows"),
        ];
        net.add_production(Production::new("p_knows_person", conds, "action"));

        let t1 = net.add_wme(wme("alice", "type", "Person"));
        assert!(!t1.contains(&"p_knows_person".to_string()));

        let t2 = net.add_wme(wme("alice", "knows", "bob"));
        assert!(t2.contains(&"p_knows_person".to_string()));
    }

    #[test]
    fn test_three_cond_production() {
        let mut net = ReteNetwork::new();
        let conds = vec![
            const_cond(CondField::Predicate, "a"),
            const_cond(CondField::Predicate, "b"),
            const_cond(CondField::Predicate, "c"),
        ];
        net.add_production(Production::new("p_three", conds, "action"));

        net.add_wme(wme("s1", "a", "o1"));
        net.add_wme(wme("s2", "b", "o2"));
        let t3 = net.add_wme(wme("s3", "c", "o3"));
        assert!(t3.contains(&"p_three".to_string()));
    }

    // ── Multiple productions triggered ─────────────────────────

    #[test]
    fn test_multiple_productions_can_fire() {
        let mut net = ReteNetwork::new();
        net.add_production(Production::new(
            "p1",
            vec![const_cond(CondField::Predicate, "type")],
            "a1",
        ));
        net.add_production(Production::new(
            "p2",
            vec![any_cond(CondField::Subject)],
            "a2",
        ));
        let triggered = net.add_wme(wme("alice", "type", "Person"));
        assert!(triggered.contains(&"p1".to_string()));
        assert!(triggered.contains(&"p2".to_string()));
    }

    #[test]
    fn test_only_matching_productions_trigger() {
        let mut net = ReteNetwork::new();
        net.add_production(Production::new(
            "p_type",
            vec![const_cond(CondField::Predicate, "type")],
            "a",
        ));
        net.add_production(Production::new(
            "p_label",
            vec![const_cond(CondField::Predicate, "label")],
            "b",
        ));
        let triggered = net.add_wme(wme("x", "type", "Y"));
        assert!(triggered.contains(&"p_type".to_string()));
        assert!(!triggered.contains(&"p_label".to_string()));
    }

    // ── Alpha memory ──────────────────────────────────────────

    #[test]
    fn test_alpha_memory_populated_after_add_wme() {
        let mut net = ReteNetwork::new();
        let cond = const_cond(CondField::Predicate, "type");
        net.add_production(Production::new("p", vec![cond.clone()], ""));
        net.add_wme(wme("alice", "type", "Person"));

        let key = super::alpha_key(&cond);
        let am = net.get_alpha_memory(&key);
        assert!(am.is_some());
        assert_eq!(am.unwrap().len(), 1);
    }

    #[test]
    fn test_get_alpha_memory_returns_none_for_unknown_key() {
        let net = ReteNetwork::new();
        assert!(net.get_alpha_memory("nonexistent").is_none());
    }

    // ── Remove WME ────────────────────────────────────────────

    #[test]
    fn test_remove_wme_from_alpha_memory() {
        let mut net = ReteNetwork::new();
        let cond = const_cond(CondField::Predicate, "type");
        net.add_production(Production::new("p", vec![cond.clone()], ""));
        let w = wme("alice", "type", "Person");
        net.add_wme(w.clone());
        net.remove_wme(&w);
        let key = super::alpha_key(&cond);
        if let Some(am) = net.get_alpha_memory(&key) {
            assert_eq!(am.len(), 0);
        }
    }

    // ── Variable binding propagation ──────────────────────────

    #[test]
    fn test_variable_binding_in_token() {
        let w = wme("alice", "type", "Person");
        let cond = var_cond(CondField::Subject, "x");
        let bindings = super::collect_bindings(&cond, &w);
        assert_eq!(bindings.get("x"), Some(&"alice".to_string()));
    }

    #[test]
    fn test_variable_binding_predicate() {
        let w = wme("a", "knows", "b");
        let cond = var_cond(CondField::Predicate, "pred");
        let bindings = super::collect_bindings(&cond, &w);
        assert_eq!(bindings.get("pred"), Some(&"knows".to_string()));
    }

    #[test]
    fn test_constant_cond_no_bindings() {
        let w = wme("a", "b", "c");
        let cond = const_cond(CondField::Subject, "a");
        let bindings = super::collect_bindings(&cond, &w);
        assert!(bindings.is_empty());
    }

    // ── Partial match / no complete match ─────────────────────

    #[test]
    fn test_no_trigger_when_only_partial_match() {
        let mut net = ReteNetwork::new();
        let conds = vec![
            const_cond(CondField::Predicate, "type"),
            const_cond(CondField::Predicate, "knows"),
        ];
        net.add_production(Production::new("p", conds, "action"));
        let triggered = net.add_wme(wme("alice", "type", "Person"));
        assert!(!triggered.contains(&"p".to_string()));
    }

    // ── Empty production ──────────────────────────────────────

    #[test]
    fn test_zero_condition_production_fires_on_any_wme() {
        let mut net = ReteNetwork::new();
        net.add_production(Production::new("p_empty", vec![], "always fires"));
        let triggered = net.add_wme(wme("anything", "here", "works"));
        assert!(triggered.contains(&"p_empty".to_string()));
    }

    // ── Alpha memory key uniqueness ───────────────────────────

    #[test]
    fn test_alpha_key_constant() {
        let c = const_cond(CondField::Subject, "alice");
        assert_eq!(super::alpha_key(&c), "s:=alice");
    }

    #[test]
    fn test_alpha_key_variable() {
        let c = var_cond(CondField::Predicate, "p");
        assert_eq!(super::alpha_key(&c), "p:?p");
    }

    #[test]
    fn test_alpha_key_any() {
        let c = any_cond(CondField::Object);
        assert_eq!(super::alpha_key(&c), "o:*");
    }

    // ── AlphaMemory unit tests ────────────────────────────────

    #[test]
    fn test_alpha_memory_add_remove() {
        let mut am = AlphaMemory::new();
        let w = wme("a", "b", "c");
        am.add(w.clone());
        assert_eq!(am.len(), 1);
        assert!(am.remove(&w));
        assert!(am.is_empty());
    }

    #[test]
    fn test_alpha_memory_remove_missing() {
        let mut am = AlphaMemory::new();
        let w = wme("a", "b", "c");
        assert!(!am.remove(&w));
    }

    #[test]
    fn test_alpha_memory_default() {
        let am = AlphaMemory::default();
        assert!(am.is_empty());
    }

    // ── Token unit tests ──────────────────────────────────────

    #[test]
    fn test_token_extend() {
        let t = Token::new();
        let w = wme("s", "p", "o");
        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), "s".to_string());
        let t2 = t.extend(w.clone(), bindings);
        assert_eq!(t2.wmes.len(), 1);
        assert_eq!(t2.bindings.get("x"), Some(&"s".to_string()));
    }

    #[test]
    fn test_token_default() {
        let t = Token::default();
        assert!(t.wmes.is_empty());
        assert!(t.bindings.is_empty());
    }

    // ── Default impl ──────────────────────────────────────────

    #[test]
    fn test_rete_network_default() {
        let net = ReteNetwork::default();
        assert_eq!(net.production_count(), 0);
    }

    #[test]
    fn test_production_id_preserved() {
        let mut net = ReteNetwork::new();
        let prod = Production::new("unique_id_42", vec![], "act");
        net.add_production(prod);
        assert_eq!(net.productions[0].id, "unique_id_42");
    }

    #[test]
    fn test_add_many_wmes() {
        let mut net = ReteNetwork::new();
        for i in 0..20_u32 {
            net.add_wme(wme(
                format!("s{i}").as_str(),
                "pred",
                format!("o{i}").as_str(),
            ));
        }
        assert_eq!(net.wme_count(), 20);
    }

    #[test]
    fn test_production_action_field() {
        let p = Production::new("p1", vec![], "ASSERT(fact)");
        assert_eq!(p.action, "ASSERT(fact)");
    }
}
