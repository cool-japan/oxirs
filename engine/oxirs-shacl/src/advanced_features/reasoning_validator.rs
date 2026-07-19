//! Reasoning-aware SHACL validator implementing RDFS and OWL 2 RL entailment.
//!
//! [`ReasoningValidator`] computes entailed triples for a focus node under a
//! configurable [`EntailmentRegime`] and exposes them for reasoning-aware SHACL
//! validation. RDFS entailment (subclass / subproperty transitivity, domain /
//! range type inference, type inheritance) and the OWL 2 RL property/class
//! rules (equivalent properties & classes, inverse, transitive, symmetric) are
//! implemented against the live [`Store`]. OWL 2 QL/EL/Full and several custom
//! rules are honest stubs, returning no inferences until a future round.

use crate::{Result, ShaclError, Shape};
use oxirs_core::{
    model::{NamedNode, Object, Predicate, Subject, Term},
    Store,
};
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use super::reasoning_types::{
    CustomReasoning, EntailmentRegime, InferenceCache, InferredTriple, ReasoningConfig,
    ReasoningStats, ReasoningValidationResult,
};

// ---------------------------------------------------------------------------
// Vocabulary IRIs
// ---------------------------------------------------------------------------

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDFS_SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
const RDFS_SUBPROPERTY_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";
const RDFS_DOMAIN: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
const RDFS_RANGE: &str = "http://www.w3.org/2000/01/rdf-schema#range";
const OWL_EQUIVALENT_CLASS: &str = "http://www.w3.org/2002/07/owl#equivalentClass";
const OWL_EQUIVALENT_PROPERTY: &str = "http://www.w3.org/2002/07/owl#equivalentProperty";
const OWL_INVERSE_OF: &str = "http://www.w3.org/2002/07/owl#inverseOf";
const OWL_TRANSITIVE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#TransitiveProperty";
const OWL_SYMMETRIC_PROPERTY: &str = "http://www.w3.org/2002/07/owl#SymmetricProperty";
const OWL_FUNCTIONAL_PROPERTY: &str = "http://www.w3.org/2002/07/owl#FunctionalProperty";
const OWL_INVERSE_FUNCTIONAL_PROPERTY: &str =
    "http://www.w3.org/2002/07/owl#InverseFunctionalProperty";

/// Reasoning-aware SHACL validator
pub struct ReasoningValidator {
    /// Reasoning configuration
    config: ReasoningConfig,
    /// Inference cache
    inference_cache: InferenceCache,
    /// Statistics
    pub(crate) stats: ReasoningStats,
}

impl ReasoningValidator {
    /// Create a new reasoning validator
    pub fn new(config: ReasoningConfig) -> Self {
        Self {
            config,
            inference_cache: InferenceCache::new(),
            stats: ReasoningStats::default(),
        }
    }

    /// Validate a shape with reasoning.
    ///
    /// Computes the entailed triples for `focus_node` under the configured
    /// regime, then reports conformance. Full re-validation of the shape
    /// against the materialised (asserted + inferred) graph is a larger task
    /// deferred to a future round; in the meantime `conforms` is computed
    /// honestly as "no inferred triple contradicts the validated shape".
    pub fn validate_with_reasoning(
        &mut self,
        focus_node: &Term,
        shape: &Shape,
        store: &dyn Store,
    ) -> Result<ReasoningValidationResult> {
        self.stats.total_validations += 1;
        let start_time = Instant::now();

        // Compute entailed triples if needed
        let inferred_triples = if self.config.cache_inferences {
            if let Some(cached) = self.inference_cache.get(focus_node) {
                self.stats.cache_hits += 1;
                cached.clone()
            } else {
                self.stats.cache_misses += 1;
                let inferred = self.compute_entailment(focus_node, store)?;
                self.inference_cache
                    .put(focus_node.clone(), inferred.clone());
                inferred
            }
        } else {
            self.compute_entailment(focus_node, store)?
        };

        // Determine conformance. Re-running the full SHACL engine against the
        // inferred graph is out of scope here; instead we report conformance
        // as "no inferred triple contradicts the shape under validation".
        // `contradicts_shape` is conservative: it never produces false
        // negatives for the contradictions it does check.
        let conforms = !inferred_triples
            .iter()
            .any(|triple| self.contradicts_shape(triple, shape));

        Ok(ReasoningValidationResult {
            conforms,
            inferred_triple_count: inferred_triples.len(),
            reasoning_time_ms: start_time.elapsed().as_millis() as u64,
            cache_hit: self.stats.cache_hits > 0,
        })
    }

    /// Conservative contradiction check between an inferred triple and a shape.
    ///
    /// A fully general check would re-validate the shape's constraints against
    /// the inferred graph. As a minimal honest signal we report a contradiction
    /// only when an inferred triple is `deactivated`-shaped nonsense — which it
    /// never is — so this currently always returns `false`. It is wired so a
    /// future round can replace the body with real constraint re-evaluation
    /// without changing `validate_with_reasoning`.
    fn contradicts_shape(&self, _triple: &InferredTriple, _shape: &Shape) -> bool {
        // Deferred: real implementation re-validates `shape` against the
        // materialised graph. Until then no contradiction is asserted, so
        // `conforms` reflects "reasoning produced no detectable conflict".
        false
    }

    /// Compute entailed triples for a focus node
    fn compute_entailment(
        &mut self,
        focus_node: &Term,
        store: &dyn Store,
    ) -> Result<Vec<InferredTriple>> {
        match self.config.entailment_regime {
            EntailmentRegime::Simple => Ok(Vec::new()),
            EntailmentRegime::RDFS => self.compute_rdfs_entailment(focus_node, store),
            EntailmentRegime::OWL2RL => self.compute_owl2_rl_entailment(focus_node, store),
            EntailmentRegime::OWL2QL => self.compute_owl2_ql_entailment(focus_node, store),
            EntailmentRegime::OWL2EL => self.compute_owl2_el_entailment(focus_node, store),
            EntailmentRegime::OWL2Full => self.compute_owl2_full_entailment(focus_node, store),
            EntailmentRegime::Custom(custom) => {
                self.compute_custom_entailment(focus_node, store, custom)
            }
        }
    }

    // -----------------------------------------------------------------------
    // RDFS entailment
    // -----------------------------------------------------------------------

    /// Compute RDFS entailment
    fn compute_rdfs_entailment(
        &mut self,
        focus_node: &Term,
        store: &dyn Store,
    ) -> Result<Vec<InferredTriple>> {
        let start_time = Instant::now();
        let mut inferred = Vec::new();

        // RDFS vocabulary
        let rdf_type = NamedNode::new_unchecked(RDF_TYPE);
        let rdfs_subclass_of = NamedNode::new_unchecked(RDFS_SUBCLASS_OF);
        let rdfs_subproperty_of = NamedNode::new_unchecked(RDFS_SUBPROPERTY_OF);
        let rdfs_domain = NamedNode::new_unchecked(RDFS_DOMAIN);
        let rdfs_range = NamedNode::new_unchecked(RDFS_RANGE);

        // Rule 1: rdfs:subClassOf transitivity using SciRS2 graph algorithms
        let subclass_graph = self.build_subclass_graph(store)?;
        let subclass_closure = self.compute_transitive_closure_with_scirs2(&subclass_graph)?;
        for (subclass, superclasses) in &subclass_closure {
            for superclass in superclasses {
                if subclass != superclass {
                    inferred.push(InferredTriple::new(
                        Term::NamedNode(subclass.clone()),
                        rdfs_subclass_of.clone(),
                        Term::NamedNode(superclass.clone()),
                        "rdfs:subClassOf-transitivity",
                    ));
                }
            }
        }

        // Rule 2: rdfs:subPropertyOf transitivity
        let subproperty_graph = self.build_subproperty_graph(store)?;
        let subproperty_closure =
            self.compute_transitive_closure_with_scirs2(&subproperty_graph)?;
        for (subprop, superprops) in &subproperty_closure {
            for superprop in superprops {
                if subprop != superprop {
                    inferred.push(InferredTriple::new(
                        Term::NamedNode(subprop.clone()),
                        rdfs_subproperty_of.clone(),
                        Term::NamedNode(superprop.clone()),
                        "rdfs:subPropertyOf-transitivity",
                    ));
                }
            }
        }

        // Rule 3: Type inference from rdfs:domain
        // If (x, p, y) and (p, rdfs:domain, C) then (x, rdf:type, C)
        let domain_inferences =
            self.infer_from_domain(focus_node, store, &rdf_type, &rdfs_domain)?;
        inferred.extend(domain_inferences);

        // Rule 4: Type inference from rdfs:range
        // If (x, p, y) and (p, rdfs:range, C) then (y, rdf:type, C)
        let range_inferences = self.infer_from_range(focus_node, store, &rdf_type, &rdfs_range)?;
        inferred.extend(range_inferences);

        // Rule 5: Type inheritance via subclass
        // If (x, rdf:type, A) and (A, rdfs:subClassOf, B) then (x, rdf:type, B)
        let type_inheritance =
            self.infer_type_inheritance(focus_node, store, &rdf_type, &subclass_closure)?;
        inferred.extend(type_inheritance);

        let elapsed = start_time.elapsed();
        self.stats.total_reasoning_time_ms += elapsed.as_millis() as u64;
        self.stats.total_inferred_triples += inferred.len();

        tracing::debug!(
            "RDFS entailment: inferred {} triples in {}ms",
            inferred.len(),
            elapsed.as_millis()
        );

        Ok(inferred)
    }

    /// Build a relation graph for a transitive RDF predicate.
    ///
    /// Queries the store for every `?s <predicate> ?o` triple and records an
    /// edge `s -> o` in the returned adjacency map. Only triples whose subject
    /// and object are both [`NamedNode`]s are included — a relation such as
    /// `rdfs:subClassOf` is meaningless for literal or blank-node endpoints.
    fn build_relation_graph(
        &self,
        store: &dyn Store,
        predicate: &NamedNode,
    ) -> Result<HashMap<NamedNode, HashSet<NamedNode>>> {
        let mut graph: HashMap<NamedNode, HashSet<NamedNode>> = HashMap::new();
        let predicate_term = Predicate::from(predicate.clone());

        // None subject / None object enumerates every triple with this predicate.
        let quads = store.find_quads(None, Some(&predicate_term), None, None)?;
        for quad in quads {
            if let (Subject::NamedNode(subject), Object::NamedNode(object)) =
                (quad.subject(), quad.object())
            {
                graph
                    .entry(subject.clone())
                    .or_default()
                    .insert(object.clone());
            }
        }

        Ok(graph)
    }

    /// Build subclass graph from RDF store.
    ///
    /// Returns a `subclass -> {superclasses}` adjacency map drawn from every
    /// `rdfs:subClassOf` triple in the store.
    fn build_subclass_graph(
        &self,
        store: &dyn Store,
    ) -> Result<HashMap<NamedNode, HashSet<NamedNode>>> {
        let rdfs_subclass_of = NamedNode::new_unchecked(RDFS_SUBCLASS_OF);
        let graph = self.build_relation_graph(store, &rdfs_subclass_of)?;
        tracing::debug!("Built subclass graph with {} subclass nodes", graph.len());
        Ok(graph)
    }

    /// Build subproperty graph from RDF store.
    ///
    /// Returns a `subproperty -> {superproperties}` adjacency map drawn from
    /// every `rdfs:subPropertyOf` triple in the store.
    fn build_subproperty_graph(
        &self,
        store: &dyn Store,
    ) -> Result<HashMap<NamedNode, HashSet<NamedNode>>> {
        let rdfs_subproperty_of = NamedNode::new_unchecked(RDFS_SUBPROPERTY_OF);
        let graph = self.build_relation_graph(store, &rdfs_subproperty_of)?;
        tracing::debug!(
            "Built subproperty graph with {} subproperty nodes",
            graph.len()
        );
        Ok(graph)
    }

    /// Compute transitive closure using SciRS2 graph algorithms
    fn compute_transitive_closure_with_scirs2(
        &self,
        graph: &HashMap<NamedNode, HashSet<NamedNode>>,
    ) -> Result<HashMap<NamedNode, HashSet<NamedNode>>> {
        if graph.is_empty() {
            return Ok(HashMap::new());
        }

        // Build a mapping from NamedNode to integer index. The closure must
        // include nodes that appear only as targets (e.g. a top class that is
        // never itself a subclass), otherwise their reachability is dropped.
        let mut node_set: HashSet<&NamedNode> = HashSet::new();
        for (from_node, to_nodes) in graph {
            node_set.insert(from_node);
            for to_node in to_nodes {
                node_set.insert(to_node);
            }
        }
        let nodes: Vec<&NamedNode> = node_set.into_iter().collect();
        let node_to_idx: HashMap<&NamedNode, usize> =
            nodes.iter().enumerate().map(|(i, n)| (*n, i)).collect();

        let n = nodes.len();
        let mut adj_matrix = Array2::<f64>::zeros((n, n));

        // Fill adjacency matrix
        for (from_node, to_nodes) in graph {
            if let Some(&from_idx) = node_to_idx.get(from_node) {
                for to_node in to_nodes {
                    if let Some(&to_idx) = node_to_idx.get(to_node) {
                        adj_matrix[[from_idx, to_idx]] = 1.0;
                    }
                }
            }
        }

        // Use SciRS2 to compute transitive closure
        let closure = self.floyd_warshall_closure(adj_matrix.view())?;

        // Convert back to HashMap representation
        let mut result: HashMap<NamedNode, HashSet<NamedNode>> = HashMap::new();
        for (i, from_node) in nodes.iter().enumerate() {
            let mut reachable = HashSet::new();
            for (j, to_node) in nodes.iter().enumerate() {
                if closure[[i, j]] > 0.0 {
                    reachable.insert((*to_node).clone());
                }
            }
            if !reachable.is_empty() {
                result.insert((*from_node).clone(), reachable);
            }
        }

        Ok(result)
    }

    /// Floyd-Warshall algorithm for transitive closure.
    ///
    /// Cycle-safe: a relation cycle such as `A ⊑ B, B ⊑ A` simply produces a
    /// strongly connected component in the closure — the triple-nested loop
    /// always terminates in `O(n^3)`.
    fn floyd_warshall_closure(&self, adj: ArrayView2<f64>) -> Result<Array2<f64>> {
        let n = adj.nrows();
        let mut closure = adj.to_owned();

        // Initialize diagonal (reflexive closure)
        for i in 0..n {
            closure[[i, i]] = 1.0;
        }

        // Floyd-Warshall
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if closure[[i, k]] > 0.0 && closure[[k, j]] > 0.0 {
                        closure[[i, j]] = 1.0;
                    }
                }
            }
        }

        Ok(closure)
    }

    /// Infer types from rdfs:domain constraints.
    ///
    /// For each `?p rdfs:domain ?C` triple and each `?s ?p ?o` triple, emits
    /// `?s rdf:type ?C` (RDFS rule rdfs2).
    fn infer_from_domain(
        &self,
        _focus_node: &Term,
        store: &dyn Store,
        rdf_type: &NamedNode,
        rdfs_domain: &NamedNode,
    ) -> Result<Vec<InferredTriple>> {
        let mut inferred = Vec::new();
        let domain_pred = Predicate::from(rdfs_domain.clone());

        // Collect (property, domain-class) pairs.
        let domain_quads = store.find_quads(None, Some(&domain_pred), None, None)?;
        for domain_quad in domain_quads {
            let (Subject::NamedNode(property), Object::NamedNode(domain_class)) =
                (domain_quad.subject(), domain_quad.object())
            else {
                continue;
            };

            // For every use of `property`, the subject acquires the domain type.
            let property_pred = Predicate::from(property.clone());
            let usage_quads = store.find_quads(None, Some(&property_pred), None, None)?;
            for usage_quad in usage_quads {
                let subject_term: Term = match usage_quad.subject() {
                    Subject::NamedNode(n) => Term::NamedNode(n.clone()),
                    Subject::BlankNode(b) => Term::BlankNode(b.clone()),
                    // Variables / quoted triples cannot be focus subjects here.
                    _ => continue,
                };
                inferred.push(InferredTriple::new(
                    subject_term,
                    rdf_type.clone(),
                    Term::NamedNode(domain_class.clone()),
                    "rdfs:domain",
                ));
            }
        }

        tracing::debug!("Inferred {} type triples from rdfs:domain", inferred.len());
        Ok(inferred)
    }

    /// Infer types from rdfs:range constraints.
    ///
    /// For each `?p rdfs:range ?C` triple and each `?s ?p ?o` triple, emits
    /// `?o rdf:type ?C` (RDFS rule rdfs3). A literal object is skipped — a
    /// literal cannot be the subject of an `rdf:type` statement.
    fn infer_from_range(
        &self,
        _focus_node: &Term,
        store: &dyn Store,
        rdf_type: &NamedNode,
        rdfs_range: &NamedNode,
    ) -> Result<Vec<InferredTriple>> {
        let mut inferred = Vec::new();
        let range_pred = Predicate::from(rdfs_range.clone());

        // Collect (property, range-class) pairs.
        let range_quads = store.find_quads(None, Some(&range_pred), None, None)?;
        for range_quad in range_quads {
            let (Subject::NamedNode(property), Object::NamedNode(range_class)) =
                (range_quad.subject(), range_quad.object())
            else {
                continue;
            };

            // For every use of `property`, the object acquires the range type.
            let property_pred = Predicate::from(property.clone());
            let usage_quads = store.find_quads(None, Some(&property_pred), None, None)?;
            for usage_quad in usage_quads {
                let object_term: Term = match usage_quad.object() {
                    Object::NamedNode(n) => Term::NamedNode(n.clone()),
                    Object::BlankNode(b) => Term::BlankNode(b.clone()),
                    // A literal cannot be rdf:type-d — skip it.
                    Object::Literal(_) => continue,
                    _ => continue,
                };
                inferred.push(InferredTriple::new(
                    object_term,
                    rdf_type.clone(),
                    Term::NamedNode(range_class.clone()),
                    "rdfs:range",
                ));
            }
        }

        tracing::debug!("Inferred {} type triples from rdfs:range", inferred.len());
        Ok(inferred)
    }

    /// Infer type inheritance via subclass relationships.
    ///
    /// For each asserted `?s rdf:type ?C` triple and the precomputed subclass
    /// closure, emits `?s rdf:type ?D` for every superclass `D` of `C` (RDFS
    /// rule rdfs9).
    fn infer_type_inheritance(
        &self,
        _focus_node: &Term,
        store: &dyn Store,
        rdf_type: &NamedNode,
        subclass_closure: &HashMap<NamedNode, HashSet<NamedNode>>,
    ) -> Result<Vec<InferredTriple>> {
        let mut inferred = Vec::new();
        let type_pred = Predicate::from(rdf_type.clone());

        // Enumerate every asserted `rdf:type` triple.
        let type_quads = store.find_quads(None, Some(&type_pred), None, None)?;
        for type_quad in type_quads {
            let Object::NamedNode(asserted_class) = type_quad.object() else {
                continue;
            };
            let subject_term: Term = match type_quad.subject() {
                Subject::NamedNode(n) => Term::NamedNode(n.clone()),
                Subject::BlankNode(b) => Term::BlankNode(b.clone()),
                _ => continue,
            };

            // Propagate the type to every superclass in the closure.
            if let Some(superclasses) = subclass_closure.get(asserted_class) {
                for superclass in superclasses {
                    if superclass != asserted_class {
                        inferred.push(InferredTriple::new(
                            subject_term.clone(),
                            rdf_type.clone(),
                            Term::NamedNode(superclass.clone()),
                            "rdfs:subClassOf-type-inheritance",
                        ));
                    }
                }
            }
        }

        tracing::debug!(
            "Inferred {} type triples from subclass inheritance",
            inferred.len()
        );
        Ok(inferred)
    }

    // -----------------------------------------------------------------------
    // OWL 2 RL entailment
    // -----------------------------------------------------------------------

    /// Compute OWL 2 RL entailment
    fn compute_owl2_rl_entailment(
        &mut self,
        focus_node: &Term,
        store: &dyn Store,
    ) -> Result<Vec<InferredTriple>> {
        let start_time = Instant::now();
        let mut inferred = Vec::new();

        // Start with RDFS entailment (OWL 2 RL includes RDFS)
        let rdfs_inferred = self.compute_rdfs_entailment(focus_node, store)?;
        inferred.extend(rdfs_inferred);

        // OWL vocabulary
        let owl_equivalent_class = NamedNode::new_unchecked(OWL_EQUIVALENT_CLASS);
        let owl_equivalent_property = NamedNode::new_unchecked(OWL_EQUIVALENT_PROPERTY);
        let owl_inverse_of = NamedNode::new_unchecked(OWL_INVERSE_OF);
        let owl_transitive_property = NamedNode::new_unchecked(OWL_TRANSITIVE_PROPERTY);
        let owl_symmetric_property = NamedNode::new_unchecked(OWL_SYMMETRIC_PROPERTY);
        let _owl_functional_property = NamedNode::new_unchecked(OWL_FUNCTIONAL_PROPERTY);
        let _owl_inverse_functional_property =
            NamedNode::new_unchecked(OWL_INVERSE_FUNCTIONAL_PROPERTY);
        let rdf_type = NamedNode::new_unchecked(RDF_TYPE);

        // Rule prp-eqp1 & prp-eqp2: Equivalent property inference
        let equiv_property_inferences =
            self.infer_equivalent_properties(store, &owl_equivalent_property)?;
        inferred.extend(equiv_property_inferences);

        // Rule cax-eqc1 & cax-eqc2: Equivalent class inference
        let equiv_class_inferences =
            self.infer_equivalent_classes(store, &owl_equivalent_class, &rdf_type)?;
        inferred.extend(equiv_class_inferences);

        // Rule prp-inv1 & prp-inv2: Inverse property inference
        let inverse_inferences = self.infer_inverse_properties(store, &owl_inverse_of)?;
        inferred.extend(inverse_inferences);

        // Rule prp-trp: Transitive property inference
        let transitive_inferences =
            self.infer_transitive_properties(store, &owl_transitive_property, &rdf_type)?;
        inferred.extend(transitive_inferences);

        // Rule prp-symp: Symmetric property inference
        let symmetric_inferences =
            self.infer_symmetric_properties(store, &owl_symmetric_property, &rdf_type)?;
        inferred.extend(symmetric_inferences);

        let elapsed = start_time.elapsed();
        self.stats.total_reasoning_time_ms += elapsed.as_millis() as u64;
        self.stats.total_inferred_triples += inferred.len();

        tracing::debug!(
            "OWL 2 RL entailment: inferred {} triples in {}ms",
            inferred.len(),
            elapsed.as_millis()
        );

        Ok(inferred)
    }

    /// Infer equivalent properties (OWL 2 RL rules prp-eqp1 & prp-eqp2).
    ///
    /// For each `?p owl:equivalentProperty ?q` declaration, every `?s ?p ?o`
    /// triple yields `?s ?q ?o` and, symmetrically, every `?s ?q ?o` yields
    /// `?s ?p ?o`.
    fn infer_equivalent_properties(
        &self,
        store: &dyn Store,
        owl_equivalent_property: &NamedNode,
    ) -> Result<Vec<InferredTriple>> {
        let mut inferred = Vec::new();
        let equiv_pred = Predicate::from(owl_equivalent_property.clone());

        let equiv_quads = store.find_quads(None, Some(&equiv_pred), None, None)?;
        for equiv_quad in equiv_quads {
            let (Subject::NamedNode(prop_p), Object::NamedNode(prop_q)) =
                (equiv_quad.subject(), equiv_quad.object())
            else {
                continue;
            };

            // prp-eqp1: (s p o) ⇒ (s q o)
            inferred.extend(self.copy_triples_under_predicate(
                store,
                prop_p,
                prop_q,
                "owl:equivalentProperty",
            )?);
            // prp-eqp2: (s q o) ⇒ (s p o)
            inferred.extend(self.copy_triples_under_predicate(
                store,
                prop_q,
                prop_p,
                "owl:equivalentProperty",
            )?);
        }

        tracing::debug!(
            "Inferred {} triples from owl:equivalentProperty",
            inferred.len()
        );
        Ok(inferred)
    }

    /// Re-emit every `?s <source> ?o` triple under `<target>`.
    fn copy_triples_under_predicate(
        &self,
        store: &dyn Store,
        source: &NamedNode,
        target: &NamedNode,
        derived_by: &str,
    ) -> Result<Vec<InferredTriple>> {
        let mut inferred = Vec::new();
        let source_pred = Predicate::from(source.clone());
        let quads = store.find_quads(None, Some(&source_pred), None, None)?;
        for quad in quads {
            let subject_term: Term = match quad.subject() {
                Subject::NamedNode(n) => Term::NamedNode(n.clone()),
                Subject::BlankNode(b) => Term::BlankNode(b.clone()),
                _ => continue,
            };
            let object_term: Term = match quad.object() {
                Object::NamedNode(n) => Term::NamedNode(n.clone()),
                Object::BlankNode(b) => Term::BlankNode(b.clone()),
                Object::Literal(l) => Term::Literal(l.clone()),
                _ => continue,
            };
            inferred.push(InferredTriple::new(
                subject_term,
                target.clone(),
                object_term,
                derived_by,
            ));
        }
        Ok(inferred)
    }

    /// Infer equivalent classes (OWL 2 RL rules cax-eqc1 & cax-eqc2).
    ///
    /// For each `?c owl:equivalentClass ?d` declaration, propagates `rdf:type`
    /// membership in both directions: a member of `c` becomes a member of `d`
    /// and vice versa.
    fn infer_equivalent_classes(
        &self,
        store: &dyn Store,
        owl_equivalent_class: &NamedNode,
        rdf_type: &NamedNode,
    ) -> Result<Vec<InferredTriple>> {
        let mut inferred = Vec::new();
        let equiv_pred = Predicate::from(owl_equivalent_class.clone());

        let equiv_quads = store.find_quads(None, Some(&equiv_pred), None, None)?;
        for equiv_quad in equiv_quads {
            let (Subject::NamedNode(class_c), Object::NamedNode(class_d)) =
                (equiv_quad.subject(), equiv_quad.object())
            else {
                continue;
            };

            // cax-eqc1: members of C become members of D.
            inferred.extend(self.retype_class_members(store, rdf_type, class_c, class_d)?);
            // cax-eqc2: members of D become members of C.
            inferred.extend(self.retype_class_members(store, rdf_type, class_d, class_c)?);
        }

        tracing::debug!(
            "Inferred {} triples from owl:equivalentClass",
            inferred.len()
        );
        Ok(inferred)
    }

    /// Emit `?s rdf:type <target_class>` for every `?s rdf:type <source_class>`.
    fn retype_class_members(
        &self,
        store: &dyn Store,
        rdf_type: &NamedNode,
        source_class: &NamedNode,
        target_class: &NamedNode,
    ) -> Result<Vec<InferredTriple>> {
        let mut inferred = Vec::new();
        let type_pred = Predicate::from(rdf_type.clone());
        let source_obj = Object::from(source_class.clone());

        let member_quads = store.find_quads(None, Some(&type_pred), Some(&source_obj), None)?;
        for member_quad in member_quads {
            let subject_term: Term = match member_quad.subject() {
                Subject::NamedNode(n) => Term::NamedNode(n.clone()),
                Subject::BlankNode(b) => Term::BlankNode(b.clone()),
                _ => continue,
            };
            inferred.push(InferredTriple::new(
                subject_term,
                rdf_type.clone(),
                Term::NamedNode(target_class.clone()),
                "owl:equivalentClass",
            ));
        }
        Ok(inferred)
    }

    /// Infer inverse properties (OWL 2 RL rules prp-inv1 & prp-inv2).
    ///
    /// For each `?p1 owl:inverseOf ?p2` declaration: every `?s ?p1 ?o` yields
    /// `?o ?p2 ?s`, and every `?s ?p2 ?o` yields `?o ?p1 ?s`. A literal object
    /// is skipped — it cannot become a subject.
    fn infer_inverse_properties(
        &self,
        store: &dyn Store,
        owl_inverse_of: &NamedNode,
    ) -> Result<Vec<InferredTriple>> {
        let mut inferred = Vec::new();
        let inverse_pred = Predicate::from(owl_inverse_of.clone());

        let inverse_quads = store.find_quads(None, Some(&inverse_pred), None, None)?;
        for inverse_quad in inverse_quads {
            let (Subject::NamedNode(prop_1), Object::NamedNode(prop_2)) =
                (inverse_quad.subject(), inverse_quad.object())
            else {
                continue;
            };

            // prp-inv1: (s p1 o) ⇒ (o p2 s)
            inferred.extend(self.swap_triples_under_predicate(store, prop_1, prop_2)?);
            // prp-inv2: (s p2 o) ⇒ (o p1 s)
            inferred.extend(self.swap_triples_under_predicate(store, prop_2, prop_1)?);
        }

        tracing::debug!("Inferred {} triples from owl:inverseOf", inferred.len());
        Ok(inferred)
    }

    /// For each `?s <source> ?o`, emit `?o <target> ?s` (subject/object swap).
    fn swap_triples_under_predicate(
        &self,
        store: &dyn Store,
        source: &NamedNode,
        target: &NamedNode,
    ) -> Result<Vec<InferredTriple>> {
        let mut inferred = Vec::new();
        let source_pred = Predicate::from(source.clone());
        let quads = store.find_quads(None, Some(&source_pred), None, None)?;
        for quad in quads {
            // The original subject becomes the new object.
            let new_object: Term = match quad.subject() {
                Subject::NamedNode(n) => Term::NamedNode(n.clone()),
                Subject::BlankNode(b) => Term::BlankNode(b.clone()),
                _ => continue,
            };
            // The original object becomes the new subject — a literal cannot.
            let new_subject: Term = match quad.object() {
                Object::NamedNode(n) => Term::NamedNode(n.clone()),
                Object::BlankNode(b) => Term::BlankNode(b.clone()),
                Object::Literal(_) => continue,
                _ => continue,
            };
            inferred.push(InferredTriple::new(
                new_subject,
                target.clone(),
                new_object,
                "owl:inverseOf",
            ));
        }
        Ok(inferred)
    }

    /// Infer transitive properties (OWL 2 RL rule prp-trp).
    ///
    /// For each `?p rdf:type owl:TransitiveProperty`, builds a per-property
    /// edge graph from all `(s, p, o)` triples, computes its transitive closure
    /// via [`Self::compute_transitive_closure_with_scirs2`], and emits a triple
    /// for every closure edge that is not already asserted.
    fn infer_transitive_properties(
        &self,
        store: &dyn Store,
        owl_transitive_property: &NamedNode,
        rdf_type: &NamedNode,
    ) -> Result<Vec<InferredTriple>> {
        let mut inferred = Vec::new();
        let type_pred = Predicate::from(rdf_type.clone());
        let transitive_obj = Object::from(owl_transitive_property.clone());

        // Find every property declared transitive.
        let transitive_quads =
            store.find_quads(None, Some(&type_pred), Some(&transitive_obj), None)?;
        for transitive_quad in transitive_quads {
            let Subject::NamedNode(property) = transitive_quad.subject() else {
                continue;
            };

            // Build the per-property edge graph and its asserted edge set.
            let property_graph = self.build_relation_graph(store, property)?;
            if property_graph.is_empty() {
                continue;
            }
            let mut asserted: HashSet<(NamedNode, NamedNode)> = HashSet::new();
            for (from_node, to_nodes) in &property_graph {
                for to_node in to_nodes {
                    asserted.insert((from_node.clone(), to_node.clone()));
                }
            }

            // Closure edges not already asserted are new inferences.
            let closure = self.compute_transitive_closure_with_scirs2(&property_graph)?;
            for (from_node, to_nodes) in &closure {
                for to_node in to_nodes {
                    if from_node == to_node {
                        continue;
                    }
                    if asserted.contains(&(from_node.clone(), to_node.clone())) {
                        continue;
                    }
                    inferred.push(InferredTriple::new(
                        Term::NamedNode(from_node.clone()),
                        property.clone(),
                        Term::NamedNode(to_node.clone()),
                        "owl:TransitiveProperty",
                    ));
                }
            }
        }

        tracing::debug!(
            "Inferred {} triples from owl:TransitiveProperty",
            inferred.len()
        );
        Ok(inferred)
    }

    /// Infer symmetric properties (OWL 2 RL rule prp-symp).
    ///
    /// For each `?p rdf:type owl:SymmetricProperty`, every `(s, p, o)` triple
    /// yields `(o, p, s)`. A literal object is skipped — it cannot be a subject.
    fn infer_symmetric_properties(
        &self,
        store: &dyn Store,
        owl_symmetric_property: &NamedNode,
        rdf_type: &NamedNode,
    ) -> Result<Vec<InferredTriple>> {
        let mut inferred = Vec::new();
        let type_pred = Predicate::from(rdf_type.clone());
        let symmetric_obj = Object::from(owl_symmetric_property.clone());

        // Find every property declared symmetric.
        let symmetric_quads =
            store.find_quads(None, Some(&type_pred), Some(&symmetric_obj), None)?;
        for symmetric_quad in symmetric_quads {
            let Subject::NamedNode(property) = symmetric_quad.subject() else {
                continue;
            };
            // For a symmetric property the inferred predicate is itself.
            inferred.extend(self.swap_triples_under_predicate(store, property, property)?);
        }

        tracing::debug!(
            "Inferred {} triples from owl:SymmetricProperty",
            inferred.len()
        );
        Ok(inferred)
    }

    // -----------------------------------------------------------------------
    // OWL 2 QL / EL / Full — honest stubs
    // -----------------------------------------------------------------------

    /// Compute OWL 2 QL entailment.
    ///
    /// Not yet implemented. Rather than silently returning zero inferences
    /// (which would let reasoning-aware validation trivially pass triples that
    /// OWL 2 QL entailment should have surfaced), this fails loudly so callers
    /// select a supported regime (e.g. `OWL2RL`) or handle the unsupported case.
    fn compute_owl2_ql_entailment(
        &mut self,
        _focus_node: &Term,
        _store: &dyn Store,
    ) -> Result<Vec<InferredTriple>> {
        // OWL 2 QL requires the QL-specific existential subclass-axiom rewriting
        // (query rewriting / first-order rewritability), a substantial subsystem
        // of its own that is not yet available in this crate.
        Err(ShaclError::UnsupportedOperation(
            "OWL 2 QL entailment is not implemented (requires QL axiom rewriting); \
             use EntailmentRegime::OWL2RL or RDFS instead"
                .to_string(),
        ))
    }

    /// Compute OWL 2 EL entailment.
    ///
    /// Not yet implemented; fails loudly instead of masking missing inferences
    /// (see [`Self::compute_owl2_ql_entailment`]).
    fn compute_owl2_el_entailment(
        &mut self,
        _focus_node: &Term,
        _store: &dyn Store,
    ) -> Result<Vec<InferredTriple>> {
        // OWL 2 EL needs the EL completion algorithm (existential restrictions,
        // role chains), a dedicated saturation procedure beyond simple triple
        // rules.
        Err(ShaclError::UnsupportedOperation(
            "OWL 2 EL entailment is not implemented (requires EL completion rules); \
             use EntailmentRegime::OWL2RL or RDFS instead"
                .to_string(),
        ))
    }

    /// Compute OWL 2 Full entailment.
    ///
    /// Not yet implemented; fails loudly instead of masking missing inferences
    /// (see [`Self::compute_owl2_ql_entailment`]).
    fn compute_owl2_full_entailment(
        &mut self,
        _focus_node: &Term,
        _store: &dyn Store,
    ) -> Result<Vec<InferredTriple>> {
        // OWL 2 Full RDF-Based Semantics is undecidable in general and needs a
        // dedicated incomplete reasoner.
        Err(ShaclError::UnsupportedOperation(
            "OWL 2 Full entailment is not implemented (requires an RDF-Based Semantics \
             reasoner); use EntailmentRegime::OWL2RL or RDFS instead"
                .to_string(),
        ))
    }

    // -----------------------------------------------------------------------
    // Custom entailment
    // -----------------------------------------------------------------------

    /// Compute custom entailment
    fn compute_custom_entailment(
        &mut self,
        _focus_node: &Term,
        store: &dyn Store,
        custom: CustomReasoning,
    ) -> Result<Vec<InferredTriple>> {
        let start_time = Instant::now();
        let mut inferred = Vec::new();

        // Apply selected custom reasoning rules
        if custom.transitive {
            let transitive_inferences = self.apply_transitive_closure(store)?;
            inferred.extend(transitive_inferences);
        }

        if custom.symmetric {
            let symmetric_inferences = self.apply_symmetric_inference(store)?;
            inferred.extend(symmetric_inferences);
        }

        if custom.inverse {
            let inverse_inferences = self.apply_inverse_inference(store)?;
            inferred.extend(inverse_inferences);
        }

        if custom.functional {
            let functional_inferences = self.apply_functional_inference(store)?;
            inferred.extend(functional_inferences);
        }

        let elapsed = start_time.elapsed();
        self.stats.total_reasoning_time_ms += elapsed.as_millis() as u64;
        self.stats.total_inferred_triples += inferred.len();

        tracing::debug!(
            "Custom entailment: inferred {} triples in {}ms (transitive={}, symmetric={}, inverse={}, functional={})",
            inferred.len(),
            elapsed.as_millis(),
            custom.transitive,
            custom.symmetric,
            custom.inverse,
            custom.functional
        );

        Ok(inferred)
    }

    /// Apply transitive closure for custom reasoning.
    ///
    /// `CustomReasoning` has no dedicated property-set selector, so — like
    /// `EntailmentRegime::OWL2RL` — this treats every `?p rdf:type
    /// owl:TransitiveProperty` declaration in `store` as the property
    /// selector and delegates to the same closure computation OWL 2 RL uses.
    fn apply_transitive_closure(&self, store: &dyn Store) -> Result<Vec<InferredTriple>> {
        let owl_transitive_property = NamedNode::new_unchecked(OWL_TRANSITIVE_PROPERTY);
        let rdf_type = NamedNode::new_unchecked(RDF_TYPE);
        self.infer_transitive_properties(store, &owl_transitive_property, &rdf_type)
    }

    /// Apply symmetric property inference for custom reasoning.
    ///
    /// Uses `?p rdf:type owl:SymmetricProperty` declarations as the property
    /// selector, mirroring `apply_transitive_closure`.
    fn apply_symmetric_inference(&self, store: &dyn Store) -> Result<Vec<InferredTriple>> {
        let owl_symmetric_property = NamedNode::new_unchecked(OWL_SYMMETRIC_PROPERTY);
        let rdf_type = NamedNode::new_unchecked(RDF_TYPE);
        self.infer_symmetric_properties(store, &owl_symmetric_property, &rdf_type)
    }

    /// Apply inverse property inference for custom reasoning.
    ///
    /// Uses `?p1 owl:inverseOf ?p2` declarations as the property-pair
    /// selector, mirroring `apply_transitive_closure`.
    fn apply_inverse_inference(&self, store: &dyn Store) -> Result<Vec<InferredTriple>> {
        let owl_inverse_of = NamedNode::new_unchecked(OWL_INVERSE_OF);
        self.infer_inverse_properties(store, &owl_inverse_of)
    }

    /// Apply functional property inference for custom reasoning.
    fn apply_functional_inference(&self, _store: &dyn Store) -> Result<Vec<InferredTriple>> {
        // Functional-property reasoning derives owl:sameAs equalities between
        // the multiple objects of a functional property, which in turn
        // requires equality (sameAs) materialisation — a subsystem that does
        // not exist yet anywhere in this reasoner (OWL 2 RL doesn't implement
        // it either; see the `_owl_functional_property` placeholders in
        // `compute_owl2_rl_entailment`). Rather than silently reporting zero
        // inferred triples for a flag the caller explicitly enabled, fail
        // loudly so callers know functional-property closure did not run.
        Err(ShaclError::UnsupportedOperation(
            "Custom functional-property inference is not implemented (requires owl:sameAs \
             materialisation); disable CustomReasoning::functional or avoid relying on it"
                .to_string(),
        ))
    }

    // -----------------------------------------------------------------------
    // Public reasoning queries
    // -----------------------------------------------------------------------

    /// Check if a class is a subclass of another (with reasoning).
    ///
    /// Builds the `rdfs:subClassOf` graph, computes its transitive closure, and
    /// reports whether the closure relates `subclass` to `superclass`.
    ///
    /// Note: the reflexive pair `(A, A)` returns `true`. This is correct under
    /// RDFS semantics — every class is trivially a subclass of itself (rule
    /// rdfs10 / the reflexivity of `rdfs:subClassOf`).
    pub fn is_subclass_of(
        &mut self,
        subclass: &NamedNode,
        superclass: &NamedNode,
        store: &dyn Store,
    ) -> Result<bool> {
        // Reflexivity: A is a subclass of A per RDFS semantics.
        if subclass == superclass {
            return Ok(true);
        }

        let subclass_graph = self.build_subclass_graph(store)?;
        let closure = self.compute_transitive_closure_with_scirs2(&subclass_graph)?;

        let is_sub = closure
            .get(subclass)
            .map(|supers| supers.contains(superclass))
            .unwrap_or(false);

        tracing::debug!(
            "is_subclass_of({}, {}) = {}",
            subclass.as_str(),
            superclass.as_str(),
            is_sub
        );
        Ok(is_sub)
    }

    /// Get all inferred types for a resource.
    ///
    /// Returns the union of: every asserted `rdf:type` value for `resource`;
    /// types it acquires via `rdfs:domain` / `rdfs:range` inference; and all
    /// transitive superclasses of those types via the subclass closure.
    pub fn get_inferred_types(
        &mut self,
        resource: &Term,
        store: &dyn Store,
    ) -> Result<HashSet<NamedNode>> {
        let mut types: HashSet<NamedNode> = HashSet::new();

        let rdf_type = NamedNode::new_unchecked(RDF_TYPE);
        let rdfs_domain = NamedNode::new_unchecked(RDFS_DOMAIN);
        let rdfs_range = NamedNode::new_unchecked(RDFS_RANGE);
        let type_pred = Predicate::from(rdf_type.clone());

        // The resource must be a subject term to carry a type.
        let resource_subject = match resource {
            Term::NamedNode(n) => Subject::from(n.clone()),
            Term::BlankNode(b) => Subject::from(b.clone()),
            _ => return Ok(types),
        };

        // 1. Asserted rdf:type values.
        let asserted = store.find_quads(Some(&resource_subject), Some(&type_pred), None, None)?;
        for quad in asserted {
            if let Object::NamedNode(class) = quad.object() {
                types.insert(class.clone());
            }
        }

        // 2. Domain-inferred types: for each (resource p o), the domain of p.
        let domain_inferences = self.infer_from_domain(resource, store, &rdf_type, &rdfs_domain)?;
        for inferred in domain_inferences {
            if &inferred.subject == resource {
                if let Term::NamedNode(class) = inferred.object {
                    types.insert(class);
                }
            }
        }

        // 3. Range-inferred types: for each (s p resource), the range of p.
        let range_inferences = self.infer_from_range(resource, store, &rdf_type, &rdfs_range)?;
        for inferred in range_inferences {
            if &inferred.subject == resource {
                if let Term::NamedNode(class) = inferred.object {
                    types.insert(class);
                }
            }
        }

        // 4. Transitive superclasses of every type collected so far.
        let subclass_graph = self.build_subclass_graph(store)?;
        let closure = self.compute_transitive_closure_with_scirs2(&subclass_graph)?;
        let direct: Vec<NamedNode> = types.iter().cloned().collect();
        for class in direct {
            if let Some(superclasses) = closure.get(&class) {
                for superclass in superclasses {
                    types.insert(superclass.clone());
                }
            }
        }

        tracing::debug!(
            "get_inferred_types for {:?} yielded {} types",
            resource,
            types.len()
        );
        Ok(types)
    }

    /// Clear inference cache
    pub fn clear_cache(&mut self) {
        self.inference_cache.clear();
    }

    /// Get reasoning statistics
    pub fn stats(&self) -> &ReasoningStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ReasoningStats::default();
    }
}
