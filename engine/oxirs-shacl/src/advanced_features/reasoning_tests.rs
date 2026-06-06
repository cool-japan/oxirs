//! Tests for reasoning-aware SHACL validation.
//!
//! Covers the configuration and helper types plus the RDFS / OWL 2 RL
//! entailment logic of [`ReasoningValidator`], using an in-memory
//! [`ConcreteStore`] populated through the `insert_triple` helper.

#![cfg(test)]

use super::reasoning_probabilistic::{EvidenceData, ProbabilisticConfig, ProbabilisticValidator};
use super::reasoning_types::{
    ClosedWorldValidator, CustomReasoning, EntailmentRegime, InferenceCache, NafGoal,
    ReasoningConfig, ReasoningStats,
};
use super::reasoning_validator::ReasoningValidator;
use oxirs_core::{
    model::{GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject, Term},
    ConcreteStore,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Insert a triple into the store: subject --predicate--> object.
fn insert_triple(store: &ConcreteStore, subject: &Term, predicate_iri: &str, object: Term) {
    let subj = match subject {
        Term::NamedNode(n) => Subject::from(n.clone()),
        _ => panic!("subject must be named node"),
    };
    let pred = Predicate::from(NamedNode::new(predicate_iri).expect("valid IRI"));
    let obj = match object {
        Term::NamedNode(n) => Object::from(n.clone()),
        Term::Literal(l) => Object::from(l),
        Term::BlankNode(b) => Object::from(b),
        _ => panic!("only NamedNode, Literal, and BlankNode are supported as objects"),
    };
    let quad = Quad::new(subj, pred, obj, GraphName::DefaultGraph);
    store.insert_quad(quad).expect("insert quad");
}

fn iri(s: &str) -> Term {
    Term::NamedNode(NamedNode::new(s).expect("valid IRI"))
}

fn nn(s: &str) -> NamedNode {
    NamedNode::new(s).expect("valid IRI")
}

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

fn owl2rl_validator() -> ReasoningValidator {
    ReasoningValidator::new(ReasoningConfig {
        entailment_regime: EntailmentRegime::OWL2RL,
        ..Default::default()
    })
}

fn rdfs_validator() -> ReasoningValidator {
    ReasoningValidator::new(ReasoningConfig::default())
}

// ---------------------------------------------------------------------------
// Original config / type tests
// ---------------------------------------------------------------------------

mod tests {
    use super::*;

    #[test]
    fn test_reasoning_config_creation() {
        let config = ReasoningConfig::default();
        assert_eq!(config.entailment_regime, EntailmentRegime::RDFS);
        assert!(!config.closed_world_assumption);
    }

    #[test]
    fn test_entailment_regimes() {
        let rdfs = EntailmentRegime::RDFS;
        let owl2rl = EntailmentRegime::OWL2RL;
        assert_ne!(rdfs, owl2rl);
    }

    #[test]
    fn test_reasoning_validator_creation() {
        let config = ReasoningConfig::default();
        let validator = ReasoningValidator::new(config);
        assert_eq!(validator.stats().total_validations, 0);
    }

    #[test]
    fn test_inference_cache() {
        let mut cache = InferenceCache::new();
        let term = Term::NamedNode(NamedNode::new_unchecked("http://example.org/test"));
        let inferred = vec![];
        cache.put(term.clone(), inferred);
        assert!(cache.get(&term).is_some());
    }

    #[test]
    fn test_closed_world_validator() {
        let mut validator = ClosedWorldValidator::new();
        let predicate = NamedNode::new_unchecked("http://example.org/knows");
        validator.register_predicate(predicate.clone());
        assert!(validator.is_known_predicate(&predicate));
    }

    #[test]
    fn test_naf_goal_creation() {
        let goal = NafGoal::new(None, None, None);
        assert!(goal.subject.is_none());
        assert!(goal.predicate.is_none());
        assert!(goal.object.is_none());
    }

    #[test]
    fn test_reasoning_stats() {
        let stats = ReasoningStats {
            total_validations: 100,
            cache_hits: 75,
            cache_misses: 25,
            total_inferred_triples: 500,
            total_reasoning_time_ms: 1000,
        };
        assert_eq!(stats.cache_hit_rate(), 0.75);
        assert_eq!(stats.average_reasoning_time_ms(), 10.0);
    }

    #[test]
    fn test_custom_reasoning() {
        let custom = CustomReasoning::default();
        assert!(custom.transitive);
        assert!(custom.symmetric);
        assert!(custom.inverse);
        assert!(custom.functional);
    }

    #[test]
    fn test_probabilistic_validator_creation() {
        let validator = ProbabilisticValidator::default_config();
        assert_eq!(validator.stats().total_validations, 0);
    }

    #[test]
    fn test_probabilistic_config_defaults() {
        let config = ProbabilisticConfig::default();
        assert_eq!(config.default_prior, 0.5);
        assert_eq!(config.confidence_level, 0.95);
        assert_eq!(config.min_evidence_count, 10);
    }

    #[test]
    fn test_bayesian_update() {
        let mut validator = ProbabilisticValidator::default_config();

        // First observation: constraint satisfied
        let result1 = validator.validate_probabilistic("test_constraint", true, 0.9);
        assert!(result1.satisfaction_probability > 0.5);

        // Second observation: constraint satisfied again
        let result2 = validator.validate_probabilistic("test_constraint", true, 0.9);
        // Posterior should increase with more evidence
        assert!(result2.satisfaction_probability >= result1.satisfaction_probability);
    }

    #[test]
    fn test_uncertainty_computation() {
        let validator = ProbabilisticValidator::default_config();

        // Certain event (p=1.0) has zero uncertainty
        assert_eq!(validator.compute_uncertainty_pub(1.0), 0.0);

        // Certain event (p=0.0) has zero uncertainty
        assert_eq!(validator.compute_uncertainty_pub(0.0), 0.0);

        // Maximum uncertainty at p=0.5
        let uncertainty = validator.compute_uncertainty_pub(0.5);
        assert!(uncertainty > 0.9); // Should be close to 1.0
    }

    #[test]
    fn test_confidence_interval() {
        let mut validator = ProbabilisticValidator::default_config();

        // Add multiple evidence points
        for i in 0..20 {
            validator.add_evidence_pub(EvidenceData {
                constraint_id: "test".to_string(),
                satisfied: i < 15, // 75% satisfaction rate
                confidence: 0.9,
                timestamp: chrono::Utc::now(),
            });
        }

        let evidence = validator.get_evidence_for_constraint_pub("test");
        let (lower, upper) = validator.compute_confidence_interval_pub(&evidence);

        // CI should bracket the true value (0.75)
        assert!(lower <= 0.75);
        assert!(upper >= 0.75);
        // CI should be reasonably narrow
        assert!(upper - lower < 0.5);
    }

    #[test]
    fn test_aggregate_probability_independence() {
        let mut validator = ProbabilisticValidator::default_config();

        // Set priors for multiple constraints
        validator.set_prior("c1".to_string(), 0.9);
        validator.set_prior("c2".to_string(), 0.8);

        let constraints = vec!["c1".to_string(), "c2".to_string()];
        let aggregate = validator.compute_aggregate_probability(&constraints);

        // Should be product (0.9 * 0.8 = 0.72)
        assert!((aggregate - 0.72).abs() < 0.01);
    }

    #[test]
    fn test_evidence_collection() {
        let mut validator = ProbabilisticValidator::default_config();

        validator.add_evidence_pub(EvidenceData {
            constraint_id: "test1".to_string(),
            satisfied: true,
            confidence: 0.9,
            timestamp: chrono::Utc::now(),
        });

        validator.add_evidence_pub(EvidenceData {
            constraint_id: "test2".to_string(),
            satisfied: false,
            confidence: 0.8,
            timestamp: chrono::Utc::now(),
        });

        assert_eq!(validator.stats().total_evidence, 2);

        let evidence1 = validator.get_evidence_for_constraint_pub("test1");
        assert_eq!(evidence1.len(), 1);
        assert!(evidence1[0].satisfied);
    }

    #[test]
    fn test_prior_updates() {
        let mut validator = ProbabilisticValidator::default_config();

        // Set initial prior
        validator.set_prior("test".to_string(), 0.3);
        assert_eq!(
            *validator
                .get_priors()
                .get("test")
                .expect("key should exist"),
            0.3
        );

        // After validation, prior should be updated
        validator.validate_probabilistic("test", true, 0.9);
        let updated_prior = *validator
            .get_priors()
            .get("test")
            .expect("key should exist");

        // Updated prior should be higher than initial
        assert!(updated_prior > 0.3);
    }

    #[test]
    fn test_monte_carlo_sampling() {
        let config = ProbabilisticConfig {
            use_monte_carlo: true,
            mc_sample_count: 10000,
            ..Default::default()
        };

        let mut validator = ProbabilisticValidator::new(config);
        validator.set_prior("c1".to_string(), 0.5);
        validator.set_prior("c2".to_string(), 0.5);

        let constraints = vec!["c1".to_string(), "c2".to_string()];
        let mc_estimate = validator.compute_aggregate_probability(&constraints);

        // Monte Carlo estimate should be close to 0.25 (0.5 * 0.5)
        assert!((mc_estimate - 0.25).abs() < 0.05);
    }

    #[test]
    fn test_probability_distribution_estimation() {
        let mut validator = ProbabilisticValidator::default_config();

        // Add evidence for distribution estimation
        for i in 0..50 {
            validator.add_evidence_pub(EvidenceData {
                constraint_id: "dist_test".to_string(),
                satisfied: i < 40, // 80% satisfaction
                confidence: 0.9,
                timestamp: chrono::Utc::now(),
            });
        }

        let distribution = validator.estimate_probability_distribution("dist_test", 10);

        // Should have 10 bins
        assert_eq!(distribution.len(), 10);

        // Distribution should be centered around high probability
        let high_prob_bins: Vec<_> = distribution.iter().filter(|(x, _)| *x > 0.7).collect();
        assert!(!high_prob_bins.is_empty());
    }
}

// ---------------------------------------------------------------------------
// New RDFS / OWL 2 RL entailment tests
// ---------------------------------------------------------------------------

mod entailment {
    use super::*;

    // ---- Subclass closure: cycle safety ----------------------------------

    #[test]
    fn test_subclass_cycle_terminates() {
        // A ⊑ B and B ⊑ A — the closure must not loop forever.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/A"),
            RDFS_SUBCLASS_OF,
            iri("http://ex/B"),
        );
        insert_triple(
            &store,
            &iri("http://ex/B"),
            RDFS_SUBCLASS_OF,
            iri("http://ex/A"),
        );

        let mut validator = rdfs_validator();
        // If the closure looped, this call would never return.
        let a_sub_b = validator
            .is_subclass_of(&nn("http://ex/A"), &nn("http://ex/B"), &store)
            .expect("subclass check");
        let b_sub_a = validator
            .is_subclass_of(&nn("http://ex/B"), &nn("http://ex/A"), &store)
            .expect("subclass check");
        assert!(a_sub_b, "A ⊑ B in a cycle");
        assert!(b_sub_a, "B ⊑ A in a cycle");
    }

    // ---- Subclass closure: 3-level chain ---------------------------------

    #[test]
    fn test_subclass_three_level_chain_transitive() {
        // A ⊑ B ⊑ C — the transitive A ⊑ C must be inferred.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/A"),
            RDFS_SUBCLASS_OF,
            iri("http://ex/B"),
        );
        insert_triple(
            &store,
            &iri("http://ex/B"),
            RDFS_SUBCLASS_OF,
            iri("http://ex/C"),
        );

        let mut validator = rdfs_validator();
        let a_sub_c = validator
            .is_subclass_of(&nn("http://ex/A"), &nn("http://ex/C"), &store)
            .expect("subclass check");
        assert!(a_sub_c, "A ⊑ C must be transitively inferred");
    }

    // ---- is_subclass_of: true / false / reflexive ------------------------

    #[test]
    fn test_is_subclass_of_direct_true() {
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/Cat"),
            RDFS_SUBCLASS_OF,
            iri("http://ex/Animal"),
        );

        let mut validator = rdfs_validator();
        assert!(validator
            .is_subclass_of(&nn("http://ex/Cat"), &nn("http://ex/Animal"), &store)
            .expect("subclass check"));
    }

    #[test]
    fn test_is_subclass_of_false() {
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/Cat"),
            RDFS_SUBCLASS_OF,
            iri("http://ex/Animal"),
        );

        let mut validator = rdfs_validator();
        // Animal is not a subclass of Cat.
        assert!(!validator
            .is_subclass_of(&nn("http://ex/Animal"), &nn("http://ex/Cat"), &store)
            .expect("subclass check"));
    }

    #[test]
    fn test_is_subclass_of_reflexive() {
        // Reflexivity is correct per RDFS: every class is a subclass of itself.
        let store = ConcreteStore::new().expect("store");
        let mut validator = rdfs_validator();
        assert!(validator
            .is_subclass_of(&nn("http://ex/Anything"), &nn("http://ex/Anything"), &store)
            .expect("subclass check"));
    }

    // ---- infer_from_domain -----------------------------------------------

    #[test]
    fn test_infer_from_domain_produces_type() {
        // hasPet rdfs:domain Person; Alice hasPet Rex ⇒ Alice rdf:type Person.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/hasPet"),
            RDFS_DOMAIN,
            iri("http://ex/Person"),
        );
        insert_triple(
            &store,
            &iri("http://ex/Alice"),
            "http://ex/hasPet",
            iri("http://ex/Rex"),
        );

        let mut validator = rdfs_validator();
        let types = validator
            .get_inferred_types(&iri("http://ex/Alice"), &store)
            .expect("inferred types");
        assert!(
            types.contains(&nn("http://ex/Person")),
            "Alice must be inferred as a Person via rdfs:domain"
        );
    }

    // ---- infer_from_range -------------------------------------------------

    #[test]
    fn test_infer_from_range_produces_type() {
        // hasPet rdfs:range Animal; Alice hasPet Rex ⇒ Rex rdf:type Animal.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/hasPet"),
            RDFS_RANGE,
            iri("http://ex/Animal"),
        );
        insert_triple(
            &store,
            &iri("http://ex/Alice"),
            "http://ex/hasPet",
            iri("http://ex/Rex"),
        );

        let mut validator = rdfs_validator();
        let types = validator
            .get_inferred_types(&iri("http://ex/Rex"), &store)
            .expect("inferred types");
        assert!(
            types.contains(&nn("http://ex/Animal")),
            "Rex must be inferred as an Animal via rdfs:range"
        );
    }

    #[test]
    fn test_infer_from_range_skips_literal_object() {
        // age rdfs:range integer; Alice age "30" — a literal cannot be typed,
        // so no inference is produced for the literal object.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/age"),
            RDFS_RANGE,
            iri("http://www.w3.org/2001/XMLSchema#integer"),
        );
        insert_triple(
            &store,
            &iri("http://ex/Alice"),
            "http://ex/age",
            Term::Literal(Literal::new("30")),
        );

        // Full RDFS entailment must not panic and must not emit a type for the
        // literal "30".
        let mut validator = rdfs_validator();
        let result = validator
            .validate_with_reasoning(&iri("http://ex/Alice"), &dummy_shape(), &store)
            .expect("reasoning");
        // No (literal rdf:type integer) triple should have been derived.
        // The only way to assert this indirectly: the focus node Alice itself
        // is the subject, and her inferred types must be empty (no domain set).
        let alice_types = validator
            .get_inferred_types(&iri("http://ex/Alice"), &store)
            .expect("types");
        assert!(alice_types.is_empty());
        // result is produced without error — literal range was skipped safely.
        let _ = result;
    }

    // ---- type inheritance -------------------------------------------------

    #[test]
    fn test_get_inferred_types_includes_subclass_parent() {
        // Rex rdf:type Cat; Cat ⊑ Animal ⇒ Rex's inferred types include Animal.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/Rex"),
            RDF_TYPE,
            iri("http://ex/Cat"),
        );
        insert_triple(
            &store,
            &iri("http://ex/Cat"),
            RDFS_SUBCLASS_OF,
            iri("http://ex/Animal"),
        );

        let mut validator = rdfs_validator();
        let types = validator
            .get_inferred_types(&iri("http://ex/Rex"), &store)
            .expect("inferred types");
        assert!(types.contains(&nn("http://ex/Cat")), "asserted type Cat");
        assert!(
            types.contains(&nn("http://ex/Animal")),
            "superclass Animal via subclass closure"
        );
    }

    // ---- equivalent properties -------------------------------------------

    #[test]
    fn test_equivalent_property_propagation() {
        // hasName owl:equivalentProperty label; Alice hasName "Alice".
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/hasName"),
            OWL_EQUIVALENT_PROPERTY,
            iri("http://ex/label"),
        );
        insert_triple(
            &store,
            &iri("http://ex/Alice"),
            "http://ex/hasName",
            iri("http://ex/AliceName"),
        );

        let mut validator = owl2rl_validator();
        let result = validator
            .validate_with_reasoning(&iri("http://ex/Alice"), &dummy_shape(), &store)
            .expect("reasoning");
        // At least: the equivalent-property copy is inferred.
        assert!(
            result.inferred_triple_count > 0,
            "owl:equivalentProperty must produce inferred triples"
        );
    }

    // ---- equivalent classes ----------------------------------------------

    #[test]
    fn test_equivalent_class_propagation() {
        // Person owl:equivalentClass Human; Alice rdf:type Person.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/Person"),
            OWL_EQUIVALENT_CLASS,
            iri("http://ex/Human"),
        );
        insert_triple(
            &store,
            &iri("http://ex/Alice"),
            RDF_TYPE,
            iri("http://ex/Person"),
        );

        let mut validator = owl2rl_validator();
        let result = validator
            .validate_with_reasoning(&iri("http://ex/Alice"), &dummy_shape(), &store)
            .expect("reasoning");
        assert!(
            result.inferred_triple_count > 0,
            "owl:equivalentClass must propagate rdf:type"
        );
    }

    // ---- inverse properties: both directions -----------------------------

    #[test]
    fn test_inverse_property_reversal_both_directions() {
        // hasParent owl:inverseOf hasChild.
        // Bob hasParent Alice  ⇒ Alice hasChild Bob.
        // Carol hasChild Dave  ⇒ Dave hasParent Carol.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/hasParent"),
            OWL_INVERSE_OF,
            iri("http://ex/hasChild"),
        );
        insert_triple(
            &store,
            &iri("http://ex/Bob"),
            "http://ex/hasParent",
            iri("http://ex/Alice"),
        );
        insert_triple(
            &store,
            &iri("http://ex/Carol"),
            "http://ex/hasChild",
            iri("http://ex/Dave"),
        );

        let mut validator = owl2rl_validator();
        let result = validator
            .validate_with_reasoning(&iri("http://ex/Bob"), &dummy_shape(), &store)
            .expect("reasoning");
        // prp-inv1 gives (Alice hasChild Bob); prp-inv2 gives
        // (Dave hasParent Carol). Both directions ⇒ at least 2 inferences.
        assert!(
            result.inferred_triple_count >= 2,
            "inverseOf must fire in both directions, got {}",
            result.inferred_triple_count
        );
    }

    #[test]
    fn test_inverse_property_skips_literal() {
        // hasValue owl:inverseOf valueOf; Thing hasValue "literal".
        // The literal cannot become a subject — no inverse triple is produced.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/hasValue"),
            OWL_INVERSE_OF,
            iri("http://ex/valueOf"),
        );
        insert_triple(
            &store,
            &iri("http://ex/Thing"),
            "http://ex/hasValue",
            Term::Literal(Literal::new("literal")),
        );

        let mut validator = owl2rl_validator();
        // Must not panic; literal endpoint is safely skipped.
        let result = validator
            .validate_with_reasoning(&iri("http://ex/Thing"), &dummy_shape(), &store)
            .expect("reasoning");
        // No inverse triple from the literal — only the (empty) RDFS pass runs.
        assert_eq!(
            result.inferred_triple_count, 0,
            "literal object must be skipped by inverseOf"
        );
    }

    // ---- symmetric properties --------------------------------------------

    #[test]
    fn test_symmetric_property_reversal() {
        // friendOf rdf:type owl:SymmetricProperty; Alice friendOf Bob
        //   ⇒ Bob friendOf Alice.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/friendOf"),
            RDF_TYPE,
            iri(OWL_SYMMETRIC_PROPERTY),
        );
        insert_triple(
            &store,
            &iri("http://ex/Alice"),
            "http://ex/friendOf",
            iri("http://ex/Bob"),
        );

        let mut validator = owl2rl_validator();
        let result = validator
            .validate_with_reasoning(&iri("http://ex/Alice"), &dummy_shape(), &store)
            .expect("reasoning");
        assert!(
            result.inferred_triple_count > 0,
            "symmetric property must produce the reversed triple"
        );
    }

    #[test]
    fn test_symmetric_property_skips_literal() {
        // overlaps rdf:type owl:SymmetricProperty; A overlaps "literal".
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/overlaps"),
            RDF_TYPE,
            iri(OWL_SYMMETRIC_PROPERTY),
        );
        insert_triple(
            &store,
            &iri("http://ex/A"),
            "http://ex/overlaps",
            Term::Literal(Literal::new("literal")),
        );

        let mut validator = owl2rl_validator();
        let result = validator
            .validate_with_reasoning(&iri("http://ex/A"), &dummy_shape(), &store)
            .expect("reasoning");
        assert_eq!(
            result.inferred_triple_count, 0,
            "literal object must be skipped by symmetric inference"
        );
    }

    // ---- transitive properties -------------------------------------------

    #[test]
    fn test_transitive_property_closure() {
        // ancestorOf rdf:type owl:TransitiveProperty.
        // A ancestorOf B, B ancestorOf C ⇒ A ancestorOf C inferred.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/ancestorOf"),
            RDF_TYPE,
            iri(OWL_TRANSITIVE_PROPERTY),
        );
        insert_triple(
            &store,
            &iri("http://ex/A"),
            "http://ex/ancestorOf",
            iri("http://ex/B"),
        );
        insert_triple(
            &store,
            &iri("http://ex/B"),
            "http://ex/ancestorOf",
            iri("http://ex/C"),
        );

        let mut validator = owl2rl_validator();
        let result = validator
            .validate_with_reasoning(&iri("http://ex/A"), &dummy_shape(), &store)
            .expect("reasoning");
        // The closure edge (A ancestorOf C) is not asserted ⇒ at least 1
        // transitive inference.
        assert!(
            result.inferred_triple_count >= 1,
            "transitive property must infer the closure edge"
        );
    }

    #[test]
    fn test_transitive_property_cycle_terminates() {
        // A transitive property with a cycle must not loop the closure.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/linked"),
            RDF_TYPE,
            iri(OWL_TRANSITIVE_PROPERTY),
        );
        insert_triple(
            &store,
            &iri("http://ex/X"),
            "http://ex/linked",
            iri("http://ex/Y"),
        );
        insert_triple(
            &store,
            &iri("http://ex/Y"),
            "http://ex/linked",
            iri("http://ex/X"),
        );

        let mut validator = owl2rl_validator();
        // Must terminate despite the cycle.
        let result = validator
            .validate_with_reasoning(&iri("http://ex/X"), &dummy_shape(), &store)
            .expect("reasoning");
        let _ = result;
    }

    // ---- subproperty closure ---------------------------------------------

    #[test]
    fn test_subproperty_transitivity_inferred() {
        // p1 ⊑ p2 ⊑ p3 — RDFS entailment must infer p1 ⊑ p3.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/p1"),
            RDFS_SUBPROPERTY_OF,
            iri("http://ex/p2"),
        );
        insert_triple(
            &store,
            &iri("http://ex/p2"),
            RDFS_SUBPROPERTY_OF,
            iri("http://ex/p3"),
        );

        let mut validator = rdfs_validator();
        let result = validator
            .validate_with_reasoning(&iri("http://ex/anything"), &dummy_shape(), &store)
            .expect("reasoning");
        // Direct p1⊑p2, p2⊑p3 plus transitive p1⊑p3 ⇒ at least 3 triples.
        assert!(
            result.inferred_triple_count >= 3,
            "subproperty transitivity should infer p1 ⊑ p3, got {}",
            result.inferred_triple_count
        );
    }

    // ---- validate_with_reasoning honest conformance ----------------------

    #[test]
    fn test_validate_with_reasoning_reports_conforms() {
        // With no contradicting inference, conforms must be honestly true.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/Cat"),
            RDFS_SUBCLASS_OF,
            iri("http://ex/Animal"),
        );

        let mut validator = rdfs_validator();
        let result = validator
            .validate_with_reasoning(&iri("http://ex/Rex"), &dummy_shape(), &store)
            .expect("reasoning");
        assert!(result.conforms);
        assert!(result.inferred_triple_count >= 1);
    }

    // ---- closed-world assumption -----------------------------------------

    #[test]
    fn test_cwa_absent_triple_is_false() {
        let store = ConcreteStore::new().expect("store");
        let cwa = ClosedWorldValidator::new();
        // Nothing in the store ⇒ the triple is false under CWA.
        let is_false = cwa
            .is_false_under_cwa(
                &iri("http://ex/A"),
                &nn("http://ex/knows"),
                &iri("http://ex/B"),
                &store,
            )
            .expect("cwa check");
        assert!(is_false, "absent triple must be false under CWA");
    }

    #[test]
    fn test_cwa_present_triple_is_not_false() {
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/A"),
            "http://ex/knows",
            iri("http://ex/B"),
        );
        let cwa = ClosedWorldValidator::new();
        let is_false = cwa
            .is_false_under_cwa(
                &iri("http://ex/A"),
                &nn("http://ex/knows"),
                &iri("http://ex/B"),
                &store,
            )
            .expect("cwa check");
        assert!(!is_false, "asserted triple must not be false under CWA");
    }

    // ---- negation as failure ---------------------------------------------

    #[test]
    fn test_naf_unprovable_goal_fails() {
        use super::super::reasoning_types::NegationAsFailure;
        let store = ConcreteStore::new().expect("store");
        let naf = NegationAsFailure::new();
        let goal = NafGoal::new(
            Some(iri("http://ex/A")),
            Some(nn("http://ex/knows")),
            Some(iri("http://ex/B")),
        );
        assert!(
            naf.fails(&goal, &store).expect("naf check"),
            "an unprovable goal must fail under NAF"
        );
    }

    #[test]
    fn test_naf_provable_goal_succeeds() {
        use super::super::reasoning_types::NegationAsFailure;
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/A"),
            "http://ex/knows",
            iri("http://ex/B"),
        );
        let naf = NegationAsFailure::new();
        let goal = NafGoal::new(
            Some(iri("http://ex/A")),
            Some(nn("http://ex/knows")),
            Some(iri("http://ex/B")),
        );
        assert!(
            !naf.fails(&goal, &store).expect("naf check"),
            "a provable goal must not fail under NAF"
        );
    }

    // ---- OWL 2 QL/EL/Full honest stubs -----------------------------------

    #[test]
    fn test_owl2_ql_el_full_stubs_return_empty() {
        // The deferred profiles must honestly produce no inferences.
        let store = ConcreteStore::new().expect("store");
        insert_triple(
            &store,
            &iri("http://ex/A"),
            RDFS_SUBCLASS_OF,
            iri("http://ex/B"),
        );

        for regime in [
            EntailmentRegime::OWL2QL,
            EntailmentRegime::OWL2EL,
            EntailmentRegime::OWL2Full,
        ] {
            let mut validator = ReasoningValidator::new(ReasoningConfig {
                entailment_regime: regime,
                ..Default::default()
            });
            let result = validator
                .validate_with_reasoning(&iri("http://ex/A"), &dummy_shape(), &store)
                .expect("reasoning");
            assert_eq!(
                result.inferred_triple_count, 0,
                "{regime:?} is an honest stub and must infer nothing"
            );
        }
    }

    /// Build a minimal `Shape` for tests that only need `validate_with_reasoning`
    /// to run; the shape body is not inspected by the conformance check yet.
    fn dummy_shape() -> crate::Shape {
        crate::Shape::node_shape(crate::ShapeId::new("http://ex/TestShape"))
    }
}
