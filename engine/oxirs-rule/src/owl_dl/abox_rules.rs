//! Core ABox inference rules 1–13 for the OWL 2 DL reasoner.
//!
//! These rules cover the 60%-coverage baseline features:
//! individual classification, property chains, nominals, hasValue,
//! allValuesFrom, someValuesFrom, transitivity, symmetry, inverseOf,
//! domain/range, sameAs, and the inconsistency checks.

use std::collections::HashMap;
use std::collections::HashSet;

use super::{mk_triple, vocab, Owl2DLReasoner, RuleFirings, Triple};

// ── Rule applications ─────────────────────────────────────────────────────────

impl Owl2DLReasoner {
    /// Rule 1: x rdf:type C, C rdfs:subClassOf D → x rdf:type D
    pub(crate) fn apply_subclass_propagation(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        for (ind, pred, class) in triples {
            if pred != vocab::RDF_TYPE {
                continue;
            }
            for (sub, sup) in &self.subclass_of {
                if sub == class {
                    let t = mk_triple(ind, vocab::RDF_TYPE, sup);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.subclass_propagation += 1;
                    }
                }
            }
        }
    }

    /// Rule 2: x rdf:type C, C owl:equivalentClass D → x rdf:type D
    pub(crate) fn apply_equivalent_class_propagation(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        for (ind, pred, class) in triples {
            if pred != vocab::RDF_TYPE {
                continue;
            }
            for (c1, c2) in &self.equivalent_classes {
                if c1 == class {
                    let t = mk_triple(ind, vocab::RDF_TYPE, c2);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.equivalent_class += 1;
                    }
                }
            }
        }
    }

    /// Rule 3 (Nominal): if ind ∈ NominalClass.members → ind rdf:type NominalClass
    pub(crate) fn apply_nominal_classification(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        // Collect all known individuals (anything that appears as subject)
        let known_subjects: HashSet<&str> = triples.iter().map(|(s, _, _)| s.as_str()).collect();

        for nominal in &self.nominal_classes {
            for member in &nominal.members {
                if known_subjects.contains(member.as_str()) {
                    let t = mk_triple(member, vocab::RDF_TYPE, &nominal.class_iri);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.nominal_classification += 1;
                    }
                }
            }
        }
    }

    /// Rule 4 (HasValue forward): x rdf:type R, R.hasValue v, R.onProperty P → x P v
    pub(crate) fn apply_has_value_forward(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        for (ind, pred, class) in triples {
            if pred != vocab::RDF_TYPE {
                continue;
            }
            for r in &self.has_value_restrictions {
                if &r.restriction_class == class {
                    let t = mk_triple(ind, &r.property, &r.value);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.has_value_forward += 1;
                    }
                }
            }
        }
    }

    /// Rule 5 (HasValue backward): x P v → x rdf:type R  (where R onProperty P hasValue v)
    pub(crate) fn apply_has_value_backward(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        for r in &self.has_value_restrictions {
            for (ind, pred, obj) in triples {
                if pred == &r.property && obj == &r.value {
                    let t = mk_triple(ind, vocab::RDF_TYPE, &r.restriction_class);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.has_value_backward += 1;
                    }
                }
            }
        }
    }

    /// Rule 6 (AllValuesFrom): x rdf:type R, x P y, R onProperty P allValuesFrom C → y rdf:type C
    pub(crate) fn apply_all_values_from(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        for avf in &self.all_values_restrictions {
            // Find all individuals of type restriction_class
            let members: Vec<&str> = triples
                .iter()
                .filter(|(_, p, o)| p == vocab::RDF_TYPE && o == &avf.restriction_class)
                .map(|(s, _, _)| s.as_str())
                .collect();

            for member in members {
                // Find all fillers for property from this member
                for (s, p, filler) in triples {
                    if s.as_str() == member && p == &avf.property {
                        let t = mk_triple(filler, vocab::RDF_TYPE, &avf.filler_class);
                        if !triples.contains(&t) {
                            new_triples.insert(t);
                            firings.all_values_from += 1;
                        }
                    }
                }
            }
        }
    }

    /// Rule 7 (SomeValuesFrom): x P y, y rdf:type C → x rdf:type R
    ///  (where R onProperty P someValuesFrom C)
    pub(crate) fn apply_some_values_from(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        for svf in &self.some_values_restrictions {
            for (x, p, y) in triples {
                if p != &svf.property {
                    continue;
                }
                // Check y rdf:type filler_class
                if triples.contains(&mk_triple(y, vocab::RDF_TYPE, &svf.filler_class)) {
                    let t = mk_triple(x, vocab::RDF_TYPE, &svf.restriction_class);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.some_values_from += 1;
                    }
                }
            }
        }
    }

    /// Rule 8 (Property Chain): x P1 v1, v1 P2 v2, …, v_{n-1} Pn y → x P y
    ///
    /// Uses iterative depth-first traversal over the chain segments.
    pub(crate) fn apply_property_chains(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        // Build a predicate-indexed view for fast joins
        let mut pred_index: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
        for (s, p, o) in triples.iter() {
            pred_index.entry(p.as_str()).or_default().push((s, o));
        }

        for chain in &self.property_chains {
            if chain.chain.is_empty() {
                continue;
            }
            let derived = super::evaluate_chain(&chain.chain, &pred_index);
            for (start, end) in derived {
                let t = mk_triple(start, &chain.entailed_property, end);
                if !triples.contains(&t) {
                    new_triples.insert(t);
                    firings.property_chain += 1;
                }
            }
        }
    }

    /// Rule 9 (Transitivity): P transitive, x P y, y P z → x P z
    pub(crate) fn apply_transitivity(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        let transitive_props: Vec<String> = self
            .property_chars
            .iter()
            .filter(|(_, c)| c.is_transitive)
            .map(|(p, _)| p.clone())
            .collect();

        for prop in &transitive_props {
            // Collect all (s,o) pairs for this property
            let pairs: Vec<(&str, &str)> = triples
                .iter()
                .filter(|(_, p, _)| p == prop)
                .map(|(s, _, o)| (s.as_str(), o.as_str()))
                .collect();

            // Build adjacency: from → [to]
            let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
            for (s, o) in &pairs {
                adj.entry(s).or_default().push(o);
            }

            // BFS closure for each starting node
            for (start, _) in &pairs {
                let reachable = super::bfs_reachable(start, &adj);
                for reached in reachable {
                    if reached != *start {
                        let t = mk_triple(start, prop, reached);
                        if !triples.contains(&t) {
                            new_triples.insert(t);
                            firings.transitivity += 1;
                        }
                    }
                }
            }
        }
    }

    /// Rule 10 (Symmetry): P symmetric, x P y → y P x
    pub(crate) fn apply_symmetry(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        let symmetric_props: Vec<String> = self
            .property_chars
            .iter()
            .filter(|(_, c)| c.is_symmetric)
            .map(|(p, _)| p.clone())
            .collect();

        for prop in &symmetric_props {
            for (s, p, o) in triples {
                if p == prop {
                    let t = mk_triple(o, prop, s);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.symmetry += 1;
                    }
                }
            }
        }
    }

    /// Rule 11 (InverseOf): P1 owl:inverseOf P2, x P1 y → y P2 x
    pub(crate) fn apply_inverse_of(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        for (prop, chars) in &self.property_chars {
            if let Some(inverse) = &chars.inverse_of {
                for (s, p, o) in triples {
                    if p == prop {
                        let t = mk_triple(o, inverse, s);
                        if !triples.contains(&t) {
                            new_triples.insert(t);
                            firings.inverse_property += 1;
                        }
                    }
                }
            }
        }
    }

    /// Rule 12 (Domain/Range): P rdfs:domain C, x P y → x rdf:type C
    ///                          P rdfs:range C, x P y → y rdf:type C
    pub(crate) fn apply_domain_range(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        // Collect domain/range axioms
        let domains: Vec<(&str, &str)> = triples
            .iter()
            .filter(|(_, p, _)| p == vocab::RDFS_DOMAIN)
            .map(|(prop, _, cls)| (prop.as_str(), cls.as_str()))
            .collect();

        let ranges: Vec<(&str, &str)> = triples
            .iter()
            .filter(|(_, p, _)| p == vocab::RDFS_RANGE)
            .map(|(prop, _, cls)| (prop.as_str(), cls.as_str()))
            .collect();

        for (prop, cls) in &domains {
            for (s, p, _o) in triples {
                if p == *prop && p != vocab::RDF_TYPE {
                    let t = mk_triple(s, vocab::RDF_TYPE, cls);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.domain_range += 1;
                    }
                }
            }
        }

        for (prop, cls) in &ranges {
            for (_s, p, o) in triples {
                if p == *prop && p != vocab::RDF_TYPE {
                    let t = mk_triple(o, vocab::RDF_TYPE, cls);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.domain_range += 1;
                    }
                }
            }
        }
    }

    /// Rule 13 (sameAs propagation): x owl:sameAs y, x rdf:type C → y rdf:type C
    pub(crate) fn apply_same_as_propagation(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        let same_as_pairs: Vec<(&str, &str)> = triples
            .iter()
            .filter(|(_, p, _)| p == vocab::OWL_SAME_AS)
            .map(|(s, _, o)| (s.as_str(), o.as_str()))
            .collect();

        for (x, y) in &same_as_pairs {
            if x == y {
                continue;
            }
            // Propagate all types from x to y
            for (s, p, o) in triples {
                if s.as_str() == *x && p == vocab::RDF_TYPE {
                    let t = mk_triple(y, vocab::RDF_TYPE, o);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.same_as_propagation += 1;
                    }
                }
            }
        }
    }

    // ── Inconsistency checks ──────────────────────────────────────────────────

    /// Check for asymmetric property violations: x P y and y P x
    pub(crate) fn check_asymmetry_violations(&mut self, _firings: &RuleFirings) {
        let asymmetric_props: Vec<String> = self
            .property_chars
            .iter()
            .filter(|(_, c)| c.is_asymmetric)
            .map(|(p, _)| p.clone())
            .collect();

        for prop in &asymmetric_props {
            let pairs: Vec<(String, String)> = self
                .abox
                .iter()
                .filter(|(_, p, _)| p == prop)
                .map(|(s, _, o)| (s.clone(), o.clone()))
                .collect();

            for (s, o) in &pairs {
                if s != o && self.abox.contains(&mk_triple(o, prop, s)) {
                    let msg =
                        format!("AsymmetricProperty violation: {s} {prop} {o} and {o} {prop} {s}");
                    if !self.inconsistencies.contains(&msg) {
                        self.inconsistencies.push(msg);
                    }
                }
            }
        }
    }

    /// Check for irreflexive property violations: x P x
    pub(crate) fn check_irreflexivity_violations(&mut self) {
        let irreflexive_props: Vec<String> = self
            .property_chars
            .iter()
            .filter(|(_, c)| c.is_irreflexive)
            .map(|(p, _)| p.clone())
            .collect();

        for prop in &irreflexive_props {
            for (s, p, o) in &self.abox {
                if p == prop && s == o {
                    let msg = format!("IrreflexiveProperty violation: {s} {prop} {s}");
                    if !self.inconsistencies.contains(&msg) {
                        self.inconsistencies.push(msg);
                    }
                }
            }
        }
    }

    /// Check for disjoint class violations: ind rdf:type C1, ind rdf:type C2
    pub(crate) fn check_disjoint_violations(&mut self) {
        for (c1, c2) in &self.disjoint_classes.clone() {
            let c1_members: Vec<String> = self
                .abox
                .iter()
                .filter(|(_, p, o)| p == vocab::RDF_TYPE && o == c1)
                .map(|(s, _, _)| s.clone())
                .collect();

            for ind in c1_members {
                if self.abox.contains(&mk_triple(&ind, vocab::RDF_TYPE, c2)) {
                    let msg = format!("DisjointWith violation: {ind} is both {c1} and {c2}");
                    if !self.inconsistencies.contains(&msg) {
                        self.inconsistencies.push(msg);
                    }
                }
            }
        }
    }

    /// Check for owl:Nothing membership
    pub(crate) fn check_nothing_violations(&mut self) {
        for (ind, p, o) in &self.abox {
            if p == vocab::RDF_TYPE && o == vocab::OWL_NOTHING {
                let msg =
                    format!("owl:Nothing violation: {ind} rdf:type owl:Nothing (bottom concept)");
                if !self.inconsistencies.contains(&msg) {
                    self.inconsistencies.push(msg);
                }
            }
        }
    }
}
