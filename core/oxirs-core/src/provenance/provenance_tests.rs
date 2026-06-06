//! Unit tests for the W3C PROV-O provenance module.

use super::*;
use crate::model::{Literal, NamedNode, Object, Predicate, Subject, Triple};

fn nn(iri: &str) -> NamedNode {
    NamedNode::new_unchecked(iri)
}

// ── AgentType ──────────────────────────────────────────────────────────

#[test]
fn test_agent_type_software_agent_iri() {
    assert_eq!(
        AgentType::SoftwareAgent.as_iri().as_str(),
        "http://www.w3.org/ns/prov#SoftwareAgent"
    );
}

#[test]
fn test_agent_type_person_iri() {
    assert_eq!(
        AgentType::Person.as_iri().as_str(),
        "http://www.w3.org/ns/prov#Person"
    );
}

#[test]
fn test_agent_type_organization_iri() {
    assert_eq!(
        AgentType::Organization.as_iri().as_str(),
        "http://www.w3.org/ns/prov#Organization"
    );
}

#[test]
fn test_agent_type_equality() {
    assert_eq!(AgentType::Person, AgentType::Person);
    assert_ne!(AgentType::Person, AgentType::Organization);
}

// ── ProvEntity ─────────────────────────────────────────────────────────

#[test]
fn test_entity_new_has_type_triple() {
    let entity = ProvEntity::new(nn("http://example.org/data1"));
    let triples = entity.to_triples();
    assert!(
        triples.iter().any(|t| {
            matches!(t.predicate(), Predicate::NamedNode(p) if p.as_str().contains("type"))
                && matches!(t.object(), Object::NamedNode(o) if o.as_str().contains("Entity"))
        }),
        "entity must have rdf:type prov:Entity triple"
    );
}

#[test]
fn test_entity_new_no_extra_attributes() {
    let entity = ProvEntity::new(nn("http://example.org/data1"));
    // Only the type triple
    assert_eq!(entity.to_triples().len(), 1);
}

#[test]
fn test_entity_with_attributes() {
    let label_pred = nn("http://www.w3.org/2000/01/rdf-schema#label");
    let entity = ProvEntity::with_attributes(
        nn("http://example.org/data1"),
        vec![(label_pred, Object::Literal(Literal::new("Dataset 1")))],
    );
    let triples = entity.to_triples();
    assert_eq!(triples.len(), 2); // type + label
}

#[test]
fn test_entity_attributes_are_emitted() {
    let pred = nn("http://example.org/customPred");
    let entity = ProvEntity::with_attributes(
        nn("http://example.org/data1"),
        vec![(pred.clone(), Object::Literal(Literal::new("custom value")))],
    );
    let triples = entity.to_triples();
    assert!(triples.iter().any(|t| {
        matches!(t.predicate(), Predicate::NamedNode(p) if p.as_str() == pred.as_str())
    }));
}

#[test]
fn test_entity_iri_preserved() {
    let iri = "http://example.org/myentity";
    let entity = ProvEntity::new(nn(iri));
    assert_eq!(entity.iri.as_str(), iri);
}

#[test]
fn test_entity_iri_is_subject_in_triples() {
    let iri = "http://example.org/myentity";
    let entity = ProvEntity::new(nn(iri));
    let triples = entity.to_triples();
    for triple in &triples {
        assert!(
            matches!(triple.subject(), Subject::NamedNode(s) if s.as_str() == iri),
            "entity IRI must be subject of all its triples"
        );
    }
}

// ── ProvActivity ───────────────────────────────────────────────────────

#[test]
fn test_activity_new_has_type_triple() {
    let activity = ProvActivity::new(nn("http://example.org/query1"));
    let triples = activity.to_triples();
    assert!(
        triples.iter().any(|t| {
            matches!(t.object(), Object::NamedNode(o) if o.as_str().contains("Activity"))
        }),
        "activity must have rdf:type prov:Activity triple"
    );
}

#[test]
fn test_activity_with_start_time() {
    let activity = ProvActivity::with_times(
        nn("http://example.org/query1"),
        Some("2026-02-24T10:00:00Z".to_string()),
        None,
        vec![],
    );
    let triples = activity.to_triples();
    assert!(triples.iter().any(|t| {
        matches!(t.predicate(), Predicate::NamedNode(p) if p.as_str().contains("startedAtTime"))
    }));
}

#[test]
fn test_activity_with_end_time() {
    let activity = ProvActivity::with_times(
        nn("http://example.org/query1"),
        None,
        Some("2026-02-24T10:05:00Z".to_string()),
        vec![],
    );
    let triples = activity.to_triples();
    assert!(triples.iter().any(|t| {
        matches!(t.predicate(), Predicate::NamedNode(p) if p.as_str().contains("endedAtTime"))
    }));
}

#[test]
fn test_activity_with_both_times() {
    let activity = ProvActivity::with_times(
        nn("http://example.org/query1"),
        Some("2026-02-24T10:00:00Z".to_string()),
        Some("2026-02-24T10:05:00Z".to_string()),
        vec![],
    );
    let triples = activity.to_triples();
    // type + startedAt + endedAt = 3
    assert_eq!(triples.len(), 3);
}

#[test]
fn test_activity_no_times() {
    let activity = ProvActivity::new(nn("http://example.org/query1"));
    // Only type triple
    assert_eq!(activity.to_triples().len(), 1);
}

#[test]
fn test_activity_with_attributes() {
    let activity = ProvActivity::with_times(
        nn("http://example.org/query1"),
        None,
        None,
        vec![(
            nn("http://example.org/desc"),
            Object::Literal(Literal::new("SPARQL query")),
        )],
    );
    let triples = activity.to_triples();
    // type + desc attribute
    assert_eq!(triples.len(), 2);
}

#[test]
fn test_activity_iri_is_subject() {
    let iri = "http://example.org/query1";
    let activity = ProvActivity::new(nn(iri));
    let triples = activity.to_triples();
    assert!(triples
        .iter()
        .all(|t| { matches!(t.subject(), Subject::NamedNode(s) if s.as_str() == iri) }));
}

// ── ProvAgent ──────────────────────────────────────────────────────────

#[test]
fn test_agent_new_has_type_agent_triple() {
    let agent = ProvAgent::new(nn("http://example.org/oxirs"), AgentType::SoftwareAgent);
    let triples = agent.to_triples();
    assert!(triples.iter().any(|t| {
        matches!(t.object(), Object::NamedNode(o) if o.as_str() == format!("{PROV_NS}Agent"))
    }));
}

#[test]
fn test_agent_software_type_triple() {
    let agent = ProvAgent::new(nn("http://example.org/oxirs"), AgentType::SoftwareAgent);
    let triples = agent.to_triples();
    assert!(triples.iter().any(|t| {
        matches!(t.object(), Object::NamedNode(o) if o.as_str() == format!("{PROV_NS}SoftwareAgent"))
    }));
}

#[test]
fn test_agent_person_type_triple() {
    let agent = ProvAgent::new(nn("http://example.org/alice"), AgentType::Person);
    let triples = agent.to_triples();
    assert!(triples
        .iter()
        .any(|t| { matches!(t.object(), Object::NamedNode(o) if o.as_str().contains("Person")) }));
}

#[test]
fn test_agent_organization_type_triple() {
    let agent = ProvAgent::new(nn("http://example.org/acme"), AgentType::Organization);
    let triples = agent.to_triples();
    assert!(triples.iter().any(|t| {
        matches!(t.object(), Object::NamedNode(o) if o.as_str().contains("Organization"))
    }));
}

#[test]
fn test_agent_with_attributes() {
    let agent = ProvAgent::with_attributes(
        nn("http://example.org/oxirs"),
        AgentType::SoftwareAgent,
        vec![(
            nn("http://example.org/version"),
            Object::Literal(Literal::new("0.2.0")),
        )],
    );
    let triples = agent.to_triples();
    // type prov:Agent + type SoftwareAgent + version attr = 3
    assert_eq!(triples.len(), 3);
}

#[test]
fn test_agent_iri_is_subject_in_triples() {
    let iri = "http://example.org/myagent";
    let agent = ProvAgent::new(nn(iri), AgentType::Organization);
    let triples = agent.to_triples();
    for triple in &triples {
        assert!(
            matches!(triple.subject(), Subject::NamedNode(s) if s.as_str() == iri),
            "agent IRI must be subject of all its triples"
        );
    }
}

// ── ProvRelationKind ───────────────────────────────────────────────────

#[test]
fn test_relation_kind_was_generated_by_predicate() {
    assert!(ProvRelationKind::WasGeneratedBy
        .as_predicate()
        .as_str()
        .contains("wasGeneratedBy"));
}

#[test]
fn test_relation_kind_was_derived_from_predicate() {
    assert!(ProvRelationKind::WasDerivedFrom
        .as_predicate()
        .as_str()
        .contains("wasDerivedFrom"));
}

#[test]
fn test_relation_kind_was_attributed_to_predicate() {
    assert!(ProvRelationKind::WasAttributedTo
        .as_predicate()
        .as_str()
        .contains("wasAttributedTo"));
}

#[test]
fn test_relation_kind_used_predicate() {
    assert!(ProvRelationKind::Used
        .as_predicate()
        .as_str()
        .contains("used"));
}

#[test]
fn test_relation_kind_was_associated_with_predicate() {
    assert!(ProvRelationKind::WasAssociatedWith
        .as_predicate()
        .as_str()
        .contains("wasAssociatedWith"));
}

#[test]
fn test_relation_kind_was_informed_by_predicate() {
    assert!(ProvRelationKind::WasInformedBy
        .as_predicate()
        .as_str()
        .contains("wasInformedBy"));
}

#[test]
fn test_relation_kind_acted_on_behalf_of_predicate() {
    assert!(ProvRelationKind::ActedOnBehalfOf
        .as_predicate()
        .as_str()
        .contains("actedOnBehalfOf"));
}

#[test]
fn test_all_seven_relation_kinds_produce_distinct_predicates() {
    let kinds = [
        ProvRelationKind::WasGeneratedBy,
        ProvRelationKind::WasDerivedFrom,
        ProvRelationKind::WasAttributedTo,
        ProvRelationKind::Used,
        ProvRelationKind::WasAssociatedWith,
        ProvRelationKind::WasInformedBy,
        ProvRelationKind::ActedOnBehalfOf,
    ];
    let predicates: Vec<String> = kinds
        .iter()
        .map(|k| k.as_predicate().as_str().to_string())
        .collect();
    let unique: std::collections::HashSet<_> = predicates.iter().collect();
    assert_eq!(
        unique.len(),
        7,
        "all 7 relation kinds must have unique predicates"
    );
}

// ── ProvRelation ───────────────────────────────────────────────────────

#[test]
fn test_relation_to_triple_correct_predicate() {
    let relation = ProvRelation::new(
        ProvRelationKind::WasGeneratedBy,
        nn("http://example.org/result"),
        nn("http://example.org/query"),
    );
    let triple = relation.to_triple();
    assert!(
        matches!(triple.predicate(), Predicate::NamedNode(p) if p.as_str().contains("wasGeneratedBy"))
    );
}

#[test]
fn test_relation_to_triple_correct_subject() {
    let relation = ProvRelation::new(
        ProvRelationKind::Used,
        nn("http://example.org/query"),
        nn("http://example.org/input"),
    );
    let triple = relation.to_triple();
    assert!(
        matches!(triple.subject(), Subject::NamedNode(s) if s.as_str() == "http://example.org/query")
    );
}

#[test]
fn test_relation_to_triple_correct_object() {
    let relation = ProvRelation::new(
        ProvRelationKind::Used,
        nn("http://example.org/query"),
        nn("http://example.org/input"),
    );
    let triple = relation.to_triple();
    assert!(
        matches!(triple.object(), Object::NamedNode(o) if o.as_str() == "http://example.org/input")
    );
}

#[test]
fn test_relation_with_qualifier() {
    let relation = ProvRelation::with_qualifier(
        ProvRelationKind::WasGeneratedBy,
        nn("http://example.org/result"),
        nn("http://example.org/query"),
        nn("http://example.org/qual1"),
    );
    assert!(relation.qualifier.is_some());
    assert_eq!(
        relation
            .qualifier
            .as_ref()
            .expect("operation should succeed")
            .as_str(),
        "http://example.org/qual1"
    );
}

#[test]
fn test_relation_no_qualifier_by_default() {
    let relation = ProvRelation::new(
        ProvRelationKind::WasInformedBy,
        nn("http://example.org/q2"),
        nn("http://example.org/q1"),
    );
    assert!(relation.qualifier.is_none());
}

// ── ProvBundle ─────────────────────────────────────────────────────────

#[test]
fn test_bundle_new_is_empty() {
    let bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    assert!(bundle.entities.is_empty());
    assert!(bundle.activities.is_empty());
    assert!(bundle.agents.is_empty());
    assert!(bundle.relations.is_empty());
}

#[test]
fn test_bundle_to_rdf_includes_bundle_type() {
    let bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    let triples = bundle.to_rdf();
    assert!(triples
        .iter()
        .any(|t| { matches!(t.object(), Object::NamedNode(o) if o.as_str().contains("Bundle")) }));
}

#[test]
fn test_bundle_add_entity() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_entity(ProvEntity::new(nn("http://example.org/e1")));
    assert_eq!(bundle.entities.len(), 1);
}

#[test]
fn test_bundle_add_activity() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_activity(ProvActivity::new(nn("http://example.org/a1")));
    assert_eq!(bundle.activities.len(), 1);
}

#[test]
fn test_bundle_add_agent() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_agent(ProvAgent::new(
        nn("http://example.org/ag1"),
        AgentType::SoftwareAgent,
    ));
    assert_eq!(bundle.agents.len(), 1);
}

#[test]
fn test_bundle_add_relation() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_relation(ProvRelation::new(
        ProvRelationKind::WasGeneratedBy,
        nn("http://example.org/r"),
        nn("http://example.org/a"),
    ));
    assert_eq!(bundle.relations.len(), 1);
}

#[test]
fn test_bundle_to_rdf_contains_entity_type() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_entity(ProvEntity::new(nn("http://example.org/e1")));
    let triples = bundle.to_rdf();
    assert!(triples
        .iter()
        .any(|t| { matches!(t.object(), Object::NamedNode(o) if o.as_str().contains("Entity")) }));
}

#[test]
fn test_bundle_to_rdf_contains_activity_type() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_activity(ProvActivity::new(nn("http://example.org/a1")));
    let triples = bundle.to_rdf();
    assert!(triples.iter().any(|t| {
        matches!(t.object(), Object::NamedNode(o) if o.as_str().contains("Activity"))
    }));
}

#[test]
fn test_bundle_to_rdf_contains_agent_type() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_agent(ProvAgent::new(
        nn("http://example.org/ag1"),
        AgentType::SoftwareAgent,
    ));
    let triples = bundle.to_rdf();
    assert!(triples
        .iter()
        .any(|t| { matches!(t.object(), Object::NamedNode(o) if o.as_str().contains("Agent")) }));
}

#[test]
fn test_bundle_to_rdf_contains_relation_triple() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_relation(ProvRelation::new(
        ProvRelationKind::WasGeneratedBy,
        nn("http://example.org/r"),
        nn("http://example.org/a"),
    ));
    let triples = bundle.to_rdf();
    assert!(triples.iter().any(|t| {
        matches!(t.predicate(), Predicate::NamedNode(p) if p.as_str().contains("wasGeneratedBy"))
    }));
}

#[test]
fn test_bundle_full_to_rdf_triple_count() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_entity(ProvEntity::new(nn("http://example.org/e1")));
    bundle.add_activity(ProvActivity::new(nn("http://example.org/a1")));
    bundle.add_agent(ProvAgent::new(
        nn("http://example.org/ag1"),
        AgentType::Person,
    ));
    bundle.add_relation(ProvRelation::new(
        ProvRelationKind::WasGeneratedBy,
        nn("http://example.org/e1"),
        nn("http://example.org/a1"),
    ));
    let triples = bundle.to_rdf();
    // bundle type(1) + entity type(1) + activity type(1) + agent(2) + relation(1) = 6
    assert!(
        triples.len() >= 6,
        "expected at least 6 triples, got {}",
        triples.len()
    );
}

// ── ProvBundle::from_rdf ───────────────────────────────────────────────

#[test]
fn test_bundle_round_trip_entity() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_entity(ProvEntity::new(nn("http://example.org/e1")));
    let triples = bundle.to_rdf();
    let restored = ProvBundle::from_rdf(&triples).expect("from_rdf should succeed");
    assert_eq!(restored.entities.len(), 1);
}

#[test]
fn test_bundle_round_trip_activity() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_activity(ProvActivity::with_times(
        nn("http://example.org/a1"),
        Some("2026-02-24T10:00:00Z".to_string()),
        Some("2026-02-24T10:05:00Z".to_string()),
        vec![],
    ));
    let triples = bundle.to_rdf();
    let restored = ProvBundle::from_rdf(&triples).expect("from_rdf should succeed");
    assert_eq!(restored.activities.len(), 1);
    assert_eq!(
        restored.activities[0].started_at.as_deref(),
        Some("2026-02-24T10:00:00Z")
    );
    assert_eq!(
        restored.activities[0].ended_at.as_deref(),
        Some("2026-02-24T10:05:00Z")
    );
}

#[test]
fn test_bundle_round_trip_agent_software() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_agent(ProvAgent::new(
        nn("http://example.org/ag1"),
        AgentType::SoftwareAgent,
    ));
    let triples = bundle.to_rdf();
    let restored = ProvBundle::from_rdf(&triples).expect("from_rdf should succeed");
    assert_eq!(restored.agents.len(), 1);
    assert_eq!(restored.agents[0].agent_type, AgentType::SoftwareAgent);
}

#[test]
fn test_bundle_round_trip_agent_person() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_agent(ProvAgent::new(
        nn("http://example.org/alice"),
        AgentType::Person,
    ));
    let triples = bundle.to_rdf();
    let restored = ProvBundle::from_rdf(&triples).expect("from_rdf should succeed");
    assert_eq!(restored.agents.len(), 1);
    assert_eq!(restored.agents[0].agent_type, AgentType::Person);
}

#[test]
fn test_bundle_round_trip_agent_organization() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_agent(ProvAgent::new(
        nn("http://example.org/acme"),
        AgentType::Organization,
    ));
    let triples = bundle.to_rdf();
    let restored = ProvBundle::from_rdf(&triples).expect("from_rdf should succeed");
    assert_eq!(restored.agents.len(), 1);
    assert_eq!(restored.agents[0].agent_type, AgentType::Organization);
}

#[test]
fn test_bundle_round_trip_relation() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    bundle.add_entity(ProvEntity::new(nn("http://example.org/e1")));
    bundle.add_activity(ProvActivity::new(nn("http://example.org/a1")));
    bundle.add_relation(ProvRelation::new(
        ProvRelationKind::WasGeneratedBy,
        nn("http://example.org/e1"),
        nn("http://example.org/a1"),
    ));
    let triples = bundle.to_rdf();
    let restored = ProvBundle::from_rdf(&triples).expect("from_rdf should succeed");
    assert_eq!(restored.relations.len(), 1);
    assert_eq!(restored.relations[0].kind, ProvRelationKind::WasGeneratedBy);
}

#[test]
fn test_bundle_from_rdf_missing_bundle_declaration() {
    // Triples without prov:Bundle declaration should fail
    let triples = vec![Triple::new(
        nn("http://example.org/e1"),
        nn("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
        nn("http://www.w3.org/ns/prov#Entity"),
    )];
    let result = ProvBundle::from_rdf(&triples);
    assert!(result.is_err());
}

#[test]
fn test_bundle_from_rdf_bundle_iri_preserved() {
    let bundle = ProvBundle::new(nn("http://example.org/mybundle"));
    let triples = bundle.to_rdf();
    let restored = ProvBundle::from_rdf(&triples).expect("from_rdf should succeed");
    assert_eq!(restored.iri.as_str(), "http://example.org/mybundle");
}

#[test]
fn test_bundle_round_trip_all_relation_kinds() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    let kinds = vec![
        ProvRelationKind::WasGeneratedBy,
        ProvRelationKind::WasDerivedFrom,
        ProvRelationKind::WasAttributedTo,
        ProvRelationKind::Used,
        ProvRelationKind::WasAssociatedWith,
        ProvRelationKind::WasInformedBy,
        ProvRelationKind::ActedOnBehalfOf,
    ];
    for kind in &kinds {
        bundle.add_relation(ProvRelation::new(
            kind.clone(),
            nn("http://example.org/s"),
            nn("http://example.org/o"),
        ));
    }
    let triples = bundle.to_rdf();
    let restored = ProvBundle::from_rdf(&triples).expect("from_rdf should succeed");
    assert_eq!(restored.relations.len(), kinds.len());
}

#[test]
fn test_bundle_multiple_entities() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    for i in 0..5 {
        bundle.add_entity(ProvEntity::new(nn(&format!("http://example.org/e{i}"))));
    }
    let triples = bundle.to_rdf();
    let restored = ProvBundle::from_rdf(&triples).expect("from_rdf should succeed");
    assert_eq!(restored.entities.len(), 5);
}

#[test]
fn test_bundle_multiple_activities() {
    let mut bundle = ProvBundle::new(nn("http://example.org/bundle1"));
    for i in 0..3 {
        bundle.add_activity(ProvActivity::new(nn(&format!("http://example.org/a{i}"))));
    }
    let triples = bundle.to_rdf();
    let restored = ProvBundle::from_rdf(&triples).expect("from_rdf should succeed");
    assert_eq!(restored.activities.len(), 3);
}

// ── QueryProvenanceTracker ─────────────────────────────────────────────

#[test]
fn test_query_tracker_to_bundle_has_entities() {
    let tracker = QueryProvenanceTracker::new(
        nn("http://example.org/query1"),
        "2026-02-24T10:00:00Z".to_string(),
        nn("http://example.org/oxirs"),
        nn("http://example.org/dataset"),
        nn("http://example.org/result"),
    );
    let bundle = tracker.to_bundle();
    assert_eq!(bundle.entities.len(), 2);
}

#[test]
fn test_query_tracker_to_bundle_has_activity() {
    let tracker = QueryProvenanceTracker::new(
        nn("http://example.org/query1"),
        "2026-02-24T10:00:00Z".to_string(),
        nn("http://example.org/oxirs"),
        nn("http://example.org/dataset"),
        nn("http://example.org/result"),
    );
    let bundle = tracker.to_bundle();
    assert_eq!(bundle.activities.len(), 1);
}

#[test]
fn test_query_tracker_to_bundle_has_software_agent() {
    let tracker = QueryProvenanceTracker::new(
        nn("http://example.org/query1"),
        "2026-02-24T10:00:00Z".to_string(),
        nn("http://example.org/oxirs"),
        nn("http://example.org/dataset"),
        nn("http://example.org/result"),
    );
    let bundle = tracker.to_bundle();
    assert_eq!(bundle.agents.len(), 1);
    assert_eq!(bundle.agents[0].agent_type, AgentType::SoftwareAgent);
}

#[test]
fn test_query_tracker_to_bundle_has_four_relations() {
    let tracker = QueryProvenanceTracker::new(
        nn("http://example.org/query1"),
        "2026-02-24T10:00:00Z".to_string(),
        nn("http://example.org/oxirs"),
        nn("http://example.org/dataset"),
        nn("http://example.org/result"),
    );
    let bundle = tracker.to_bundle();
    assert_eq!(bundle.relations.len(), 4);
}

#[test]
fn test_query_tracker_to_bundle_was_generated_by() {
    let tracker = QueryProvenanceTracker::new(
        nn("http://example.org/query1"),
        "2026-02-24T10:00:00Z".to_string(),
        nn("http://example.org/oxirs"),
        nn("http://example.org/dataset"),
        nn("http://example.org/result"),
    );
    let bundle = tracker.to_bundle();
    assert!(bundle
        .relations
        .iter()
        .any(|r| r.kind == ProvRelationKind::WasGeneratedBy));
}

#[test]
fn test_query_tracker_to_bundle_used() {
    let tracker = QueryProvenanceTracker::new(
        nn("http://example.org/query1"),
        "2026-02-24T10:00:00Z".to_string(),
        nn("http://example.org/oxirs"),
        nn("http://example.org/dataset"),
        nn("http://example.org/result"),
    );
    let bundle = tracker.to_bundle();
    assert!(bundle
        .relations
        .iter()
        .any(|r| r.kind == ProvRelationKind::Used));
}

#[test]
fn test_query_tracker_to_bundle_was_associated_with() {
    let tracker = QueryProvenanceTracker::new(
        nn("http://example.org/query1"),
        "2026-02-24T10:00:00Z".to_string(),
        nn("http://example.org/oxirs"),
        nn("http://example.org/dataset"),
        nn("http://example.org/result"),
    );
    let bundle = tracker.to_bundle();
    assert!(bundle
        .relations
        .iter()
        .any(|r| r.kind == ProvRelationKind::WasAssociatedWith));
}

#[test]
fn test_query_tracker_to_bundle_was_attributed_to() {
    let tracker = QueryProvenanceTracker::new(
        nn("http://example.org/query1"),
        "2026-02-24T10:00:00Z".to_string(),
        nn("http://example.org/oxirs"),
        nn("http://example.org/dataset"),
        nn("http://example.org/result"),
    );
    let bundle = tracker.to_bundle();
    assert!(bundle
        .relations
        .iter()
        .any(|r| r.kind == ProvRelationKind::WasAttributedTo));
}

#[test]
fn test_query_tracker_with_query_text() {
    let tracker = QueryProvenanceTracker::new(
        nn("http://example.org/query1"),
        "2026-02-24T10:00:00Z".to_string(),
        nn("http://example.org/oxirs"),
        nn("http://example.org/dataset"),
        nn("http://example.org/result"),
    )
    .with_query_text("SELECT * WHERE { ?s ?p ?o }");
    assert!(tracker.query_text.is_some());
    let bundle = tracker.to_bundle();
    assert!(bundle.activities[0]
        .attributes
        .iter()
        .any(|(_, v)| { matches!(v, Object::Literal(l) if l.value().contains("SELECT")) }));
}

#[test]
fn test_query_tracker_to_bundle_to_rdf() {
    let tracker = QueryProvenanceTracker::new(
        nn("http://example.org/query1"),
        "2026-02-24T10:00:00Z".to_string(),
        nn("http://example.org/oxirs"),
        nn("http://example.org/dataset"),
        nn("http://example.org/result"),
    );
    let bundle = tracker.to_bundle();
    let triples = bundle.to_rdf();
    assert!(triples.len() > 5);
}

#[test]
fn test_query_tracker_executed_at_in_activity() {
    let tracker = QueryProvenanceTracker::new(
        nn("http://example.org/query1"),
        "2026-02-24T10:00:00Z".to_string(),
        nn("http://example.org/oxirs"),
        nn("http://example.org/dataset"),
        nn("http://example.org/result"),
    );
    let bundle = tracker.to_bundle();
    let activity = &bundle.activities[0];
    assert_eq!(activity.started_at.as_deref(), Some("2026-02-24T10:00:00Z"));
}

// ── PROV-O namespace constants ─────────────────────────────────────────

#[test]
fn test_prov_ns_constant() {
    assert_eq!(PROV_NS, "http://www.w3.org/ns/prov#");
}

#[test]
fn test_xsd_ns_constant() {
    assert_eq!(XSD_NS, "http://www.w3.org/2001/XMLSchema#");
}

#[test]
fn test_rdf_ns_constant() {
    assert_eq!(RDF_NS, "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
}
