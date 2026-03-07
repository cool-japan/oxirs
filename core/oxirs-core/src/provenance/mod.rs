//! W3C PROV-O ontology support for RDF data provenance tracking
//!
//! This module implements the W3C PROV-O (Provenance Ontology) for tracking
//! data provenance in RDF graphs. It supports the core PROV-O classes
//! (Entity, Activity, Agent), relations, and bundles.
//!
//! # References
//! - <https://www.w3.org/TR/prov-o/>
//! - <https://www.w3.org/TR/prov-dm/>

use crate::model::{Literal, NamedNode, Object, Predicate, Subject, Triple};
use crate::OxirsError;
use std::collections::HashMap;

/// PROV-O namespace prefix
pub const PROV_NS: &str = "http://www.w3.org/ns/prov#";

/// XSD namespace prefix
pub const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";

/// RDF namespace prefix
pub const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

/// Build a PROV-O IRI from a local name
fn prov_iri(local: &str) -> NamedNode {
    NamedNode::new_unchecked(format!("{PROV_NS}{local}"))
}

/// Build an XSD datatype IRI
fn xsd_iri(local: &str) -> NamedNode {
    NamedNode::new_unchecked(format!("{XSD_NS}{local}"))
}

/// Build an RDF IRI
fn rdf_iri(local: &str) -> NamedNode {
    NamedNode::new_unchecked(format!("{RDF_NS}{local}"))
}

// ── Agent type ───────────────────────────────────────────────────────────────

/// The type of a PROV-O agent (prov:SoftwareAgent, prov:Person, prov:Organization)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AgentType {
    /// prov:SoftwareAgent — a running software system
    SoftwareAgent,
    /// prov:Person — a human being
    Person,
    /// prov:Organization — a social or legal institution
    Organization,
}

impl AgentType {
    /// Return the PROV-O IRI for this agent type
    pub fn as_iri(&self) -> NamedNode {
        match self {
            AgentType::SoftwareAgent => prov_iri("SoftwareAgent"),
            AgentType::Person => prov_iri("Person"),
            AgentType::Organization => prov_iri("Organization"),
        }
    }
}

// ── Core PROV-O classes ───────────────────────────────────────────────────────

/// A PROV-O Entity — something whose provenance we track (prov:Entity)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvEntity {
    /// The IRI identifying this entity
    pub iri: NamedNode,
    /// Arbitrary attribute-value pairs attached to this entity
    pub attributes: Vec<(NamedNode, Object)>,
}

impl ProvEntity {
    /// Create a new PROV-O entity with the given IRI and no extra attributes
    pub fn new(iri: NamedNode) -> Self {
        Self {
            iri,
            attributes: Vec::new(),
        }
    }

    /// Create a new entity with additional attributes
    pub fn with_attributes(iri: NamedNode, attributes: Vec<(NamedNode, Object)>) -> Self {
        Self { iri, attributes }
    }

    /// Emit RDF triples that represent this entity
    pub fn to_triples(&self) -> Vec<Triple> {
        let mut triples = Vec::new();
        // rdf:type prov:Entity
        triples.push(Triple::new(
            self.iri.clone(),
            rdf_iri("type"),
            prov_iri("Entity"),
        ));
        for (pred, obj) in &self.attributes {
            triples.push(Triple::new(self.iri.clone(), pred.clone(), obj.clone()));
        }
        triples
    }
}

/// A PROV-O Activity — something that occurred over a period of time (prov:Activity)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvActivity {
    /// The IRI identifying this activity
    pub iri: NamedNode,
    /// Optional start time as an XSD dateTime string
    pub started_at: Option<String>,
    /// Optional end time as an XSD dateTime string
    pub ended_at: Option<String>,
    /// Arbitrary attribute-value pairs attached to this activity
    pub attributes: Vec<(NamedNode, Object)>,
}

impl ProvActivity {
    /// Create a new PROV-O activity with only an IRI
    pub fn new(iri: NamedNode) -> Self {
        Self {
            iri,
            started_at: None,
            ended_at: None,
            attributes: Vec::new(),
        }
    }

    /// Create a new activity with optional start/end times and attributes
    pub fn with_times(
        iri: NamedNode,
        started_at: Option<String>,
        ended_at: Option<String>,
        attributes: Vec<(NamedNode, Object)>,
    ) -> Self {
        Self {
            iri,
            started_at,
            ended_at,
            attributes,
        }
    }

    /// Emit RDF triples that represent this activity
    pub fn to_triples(&self) -> Vec<Triple> {
        let mut triples = Vec::new();
        let xsd_datetime = xsd_iri("dateTime");

        // rdf:type prov:Activity
        triples.push(Triple::new(
            self.iri.clone(),
            rdf_iri("type"),
            prov_iri("Activity"),
        ));

        if let Some(ref start) = self.started_at {
            triples.push(Triple::new(
                self.iri.clone(),
                prov_iri("startedAtTime"),
                Literal::new_typed(start.as_str(), xsd_datetime.clone()),
            ));
        }

        if let Some(ref end) = self.ended_at {
            triples.push(Triple::new(
                self.iri.clone(),
                prov_iri("endedAtTime"),
                Literal::new_typed(end.as_str(), xsd_datetime.clone()),
            ));
        }

        for (pred, obj) in &self.attributes {
            triples.push(Triple::new(self.iri.clone(), pred.clone(), obj.clone()));
        }

        triples
    }
}

/// A PROV-O Agent — something responsible for an activity (prov:Agent)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvAgent {
    /// The IRI identifying this agent
    pub iri: NamedNode,
    /// The specific sub-type of agent
    pub agent_type: AgentType,
    /// Arbitrary attribute-value pairs attached to this agent
    pub attributes: Vec<(NamedNode, Object)>,
}

impl ProvAgent {
    /// Create a new PROV-O agent
    pub fn new(iri: NamedNode, agent_type: AgentType) -> Self {
        Self {
            iri,
            agent_type,
            attributes: Vec::new(),
        }
    }

    /// Create a new agent with extra attributes
    pub fn with_attributes(
        iri: NamedNode,
        agent_type: AgentType,
        attributes: Vec<(NamedNode, Object)>,
    ) -> Self {
        Self {
            iri,
            agent_type,
            attributes,
        }
    }

    /// Emit RDF triples that represent this agent
    pub fn to_triples(&self) -> Vec<Triple> {
        let mut triples = Vec::new();

        // rdf:type prov:Agent
        triples.push(Triple::new(
            self.iri.clone(),
            rdf_iri("type"),
            prov_iri("Agent"),
        ));

        // rdf:type <specific-agent-type>
        triples.push(Triple::new(
            self.iri.clone(),
            rdf_iri("type"),
            self.agent_type.as_iri(),
        ));

        for (pred, obj) in &self.attributes {
            triples.push(Triple::new(self.iri.clone(), pred.clone(), obj.clone()));
        }

        triples
    }
}

// ── PROV-O relations ──────────────────────────────────────────────────────────

/// The kind of PROV-O relation connecting two resources
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProvRelationKind {
    /// entity prov:wasGeneratedBy activity
    WasGeneratedBy,
    /// entity prov:wasDerivedFrom entity
    WasDerivedFrom,
    /// entity prov:wasAttributedTo agent
    WasAttributedTo,
    /// activity prov:used entity
    Used,
    /// activity prov:wasAssociatedWith agent
    WasAssociatedWith,
    /// activity prov:wasInformedBy activity
    WasInformedBy,
    /// agent prov:actedOnBehalfOf agent
    ActedOnBehalfOf,
}

impl ProvRelationKind {
    /// Return the PROV-O predicate IRI for this relation kind
    pub fn as_predicate(&self) -> NamedNode {
        match self {
            ProvRelationKind::WasGeneratedBy => prov_iri("wasGeneratedBy"),
            ProvRelationKind::WasDerivedFrom => prov_iri("wasDerivedFrom"),
            ProvRelationKind::WasAttributedTo => prov_iri("wasAttributedTo"),
            ProvRelationKind::Used => prov_iri("used"),
            ProvRelationKind::WasAssociatedWith => prov_iri("wasAssociatedWith"),
            ProvRelationKind::WasInformedBy => prov_iri("wasInformedBy"),
            ProvRelationKind::ActedOnBehalfOf => prov_iri("actedOnBehalfOf"),
        }
    }
}

/// A single PROV-O relation connecting two IRIs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvRelation {
    /// The kind of relation (determines the predicate IRI)
    pub kind: ProvRelationKind,
    /// The subject of the relation
    pub subject: NamedNode,
    /// The object of the relation
    pub object: NamedNode,
    /// Optional qualifier (for qualified relations such as prov:qualifiedGeneration)
    pub qualifier: Option<NamedNode>,
}

impl ProvRelation {
    /// Create a new PROV-O relation
    pub fn new(kind: ProvRelationKind, subject: NamedNode, object: NamedNode) -> Self {
        Self {
            kind,
            subject,
            object,
            qualifier: None,
        }
    }

    /// Create a new PROV-O relation with a qualifier
    pub fn with_qualifier(
        kind: ProvRelationKind,
        subject: NamedNode,
        object: NamedNode,
        qualifier: NamedNode,
    ) -> Self {
        Self {
            kind,
            subject,
            object,
            qualifier: Some(qualifier),
        }
    }

    /// Emit the RDF triple for this relation
    pub fn to_triple(&self) -> Triple {
        Triple::new(
            self.subject.clone(),
            self.kind.as_predicate(),
            self.object.clone(),
        )
    }
}

// ── Provenance Bundle ─────────────────────────────────────────────────────────

/// A PROV-O Bundle — a named collection of provenance statements
///
/// Bundles allow grouping provenance statements that describe the same
/// provenance record. They map naturally to named graphs in RDF datasets.
#[derive(Debug, Clone)]
pub struct ProvBundle {
    /// IRI identifying this bundle
    pub iri: NamedNode,
    /// Entities in this bundle
    pub entities: Vec<ProvEntity>,
    /// Activities in this bundle
    pub activities: Vec<ProvActivity>,
    /// Agents in this bundle
    pub agents: Vec<ProvAgent>,
    /// Relations between entities, activities, and agents
    pub relations: Vec<ProvRelation>,
}

impl ProvBundle {
    /// Create an empty provenance bundle
    pub fn new(iri: NamedNode) -> Self {
        Self {
            iri,
            entities: Vec::new(),
            activities: Vec::new(),
            agents: Vec::new(),
            relations: Vec::new(),
        }
    }

    /// Add an entity to this bundle
    pub fn add_entity(&mut self, entity: ProvEntity) {
        self.entities.push(entity);
    }

    /// Add an activity to this bundle
    pub fn add_activity(&mut self, activity: ProvActivity) {
        self.activities.push(activity);
    }

    /// Add an agent to this bundle
    pub fn add_agent(&mut self, agent: ProvAgent) {
        self.agents.push(agent);
    }

    /// Add a relation to this bundle
    pub fn add_relation(&mut self, relation: ProvRelation) {
        self.relations.push(relation);
    }

    /// Serialize the bundle to a flat Vec of RDF triples.
    ///
    /// The bundle IRI is typed as prov:Bundle. All entities, activities,
    /// agents, and relations are serialized into the same flat triple list.
    pub fn to_rdf(&self) -> Vec<Triple> {
        let mut triples = Vec::new();

        // Declare the bundle itself
        triples.push(Triple::new(
            self.iri.clone(),
            rdf_iri("type"),
            prov_iri("Bundle"),
        ));

        for entity in &self.entities {
            triples.extend(entity.to_triples());
        }
        for activity in &self.activities {
            triples.extend(activity.to_triples());
        }
        for agent in &self.agents {
            triples.extend(agent.to_triples());
        }
        for relation in &self.relations {
            triples.push(relation.to_triple());
        }

        triples
    }

    /// Parse a bundle from a slice of RDF triples.
    ///
    /// This is a best-effort round-trip: it reconstructs entities, activities,
    /// agents, and relations from the triple patterns defined by PROV-O.
    pub fn from_rdf(triples: &[Triple]) -> Result<Self, OxirsError> {
        // Group triples by subject
        let mut by_subject: HashMap<String, Vec<&Triple>> = HashMap::new();
        for triple in triples {
            let key = match triple.subject() {
                Subject::NamedNode(n) => n.as_str().to_string(),
                Subject::BlankNode(b) => b.as_str().to_string(),
                _ => continue,
            };
            by_subject.entry(key).or_default().push(triple);
        }

        let type_pred_full = format!("{RDF_NS}type");
        let bundle_type_full = format!("{PROV_NS}Bundle");

        // Determine the bundle IRI — it is typed as prov:Bundle
        let bundle_iri_str = triples
            .iter()
            .find(|t| {
                matches!(t.predicate(), Predicate::NamedNode(p) if p.as_str() == type_pred_full)
                    && matches!(t.object(), Object::NamedNode(o) if o.as_str() == bundle_type_full)
            })
            .and_then(|t| match t.subject() {
                Subject::NamedNode(n) => Some(n.as_str().to_string()),
                _ => None,
            })
            .ok_or_else(|| OxirsError::Parse("No prov:Bundle declaration found".to_string()))?;

        let bundle_iri = NamedNode::new_unchecked(bundle_iri_str.clone());

        // Classify subjects by rdf:type
        let entity_type = format!("{PROV_NS}Entity");
        let activity_type = format!("{PROV_NS}Activity");
        let agent_type_iri_str = format!("{PROV_NS}Agent");
        let software_type = format!("{PROV_NS}SoftwareAgent");
        let person_type = format!("{PROV_NS}Person");
        let org_type = format!("{PROV_NS}Organization");

        let mut entities: Vec<ProvEntity> = Vec::new();
        let mut activities: Vec<ProvActivity> = Vec::new();
        let mut agents: Vec<ProvAgent> = Vec::new();
        let mut relations: Vec<ProvRelation> = Vec::new();

        // Build relation kind map (predicate IRI -> kind)
        let relation_kind_map: HashMap<String, ProvRelationKind> = [
            (
                format!("{PROV_NS}wasGeneratedBy"),
                ProvRelationKind::WasGeneratedBy,
            ),
            (
                format!("{PROV_NS}wasDerivedFrom"),
                ProvRelationKind::WasDerivedFrom,
            ),
            (
                format!("{PROV_NS}wasAttributedTo"),
                ProvRelationKind::WasAttributedTo,
            ),
            (format!("{PROV_NS}used"), ProvRelationKind::Used),
            (
                format!("{PROV_NS}wasAssociatedWith"),
                ProvRelationKind::WasAssociatedWith,
            ),
            (
                format!("{PROV_NS}wasInformedBy"),
                ProvRelationKind::WasInformedBy,
            ),
            (
                format!("{PROV_NS}actedOnBehalfOf"),
                ProvRelationKind::ActedOnBehalfOf,
            ),
        ]
        .into_iter()
        .collect();

        // Scan each triple for relations
        for triple in triples {
            let subj_iri = match triple.subject() {
                Subject::NamedNode(n) => n.clone(),
                _ => continue,
            };
            let pred_str = match triple.predicate() {
                Predicate::NamedNode(p) => p.as_str().to_string(),
                _ => continue,
            };
            let obj_iri = match triple.object() {
                Object::NamedNode(o) => o.clone(),
                _ => continue,
            };

            if let Some(kind) = relation_kind_map.get(&pred_str) {
                relations.push(ProvRelation::new(kind.clone(), subj_iri, obj_iri));
            }
        }

        // Classify each subject
        for (subj_str, subj_triples) in &by_subject {
            if subj_str == &bundle_iri_str {
                continue;
            }

            // Collect types for this subject
            let types: Vec<String> = subj_triples
                .iter()
                .filter(|t| {
                    matches!(t.predicate(), Predicate::NamedNode(p) if p.as_str() == type_pred_full)
                })
                .filter_map(|t| match t.object() {
                    Object::NamedNode(o) => Some(o.as_str().to_string()),
                    _ => None,
                })
                .collect();

            let iri = NamedNode::new_unchecked(subj_str.clone());

            // Collect non-type, non-relation attributes
            let attributes: Vec<(NamedNode, Object)> = subj_triples
                .iter()
                .filter_map(|t| {
                    if let Predicate::NamedNode(p) = t.predicate() {
                        let p_str = p.as_str();
                        if p_str == type_pred_full {
                            return None;
                        }
                        if relation_kind_map.contains_key(p_str) {
                            return None;
                        }
                        Some((p.clone(), t.object().clone()))
                    } else {
                        None
                    }
                })
                .collect();

            if types.contains(&entity_type) {
                entities.push(ProvEntity::with_attributes(iri, attributes));
            } else if types.contains(&activity_type) {
                let start_pred = format!("{PROV_NS}startedAtTime");
                let end_pred = format!("{PROV_NS}endedAtTime");

                let start = attributes
                    .iter()
                    .find(|(p, _)| p.as_str() == start_pred)
                    .and_then(|(_, o)| match o {
                        Object::Literal(l) => Some(l.value().to_string()),
                        _ => None,
                    });
                let end = attributes
                    .iter()
                    .find(|(p, _)| p.as_str() == end_pred)
                    .and_then(|(_, o)| match o {
                        Object::Literal(l) => Some(l.value().to_string()),
                        _ => None,
                    });
                let extra_attrs: Vec<(NamedNode, Object)> = attributes
                    .into_iter()
                    .filter(|(p, _)| p.as_str() != start_pred && p.as_str() != end_pred)
                    .collect();
                activities.push(ProvActivity::with_times(iri, start, end, extra_attrs));
            } else if types.contains(&agent_type_iri_str) {
                let agent_kind = if types.contains(&software_type) {
                    AgentType::SoftwareAgent
                } else if types.contains(&person_type) {
                    AgentType::Person
                } else if types.contains(&org_type) {
                    AgentType::Organization
                } else {
                    AgentType::Person
                };
                agents.push(ProvAgent::with_attributes(iri, agent_kind, attributes));
            }
        }

        Ok(Self {
            iri: bundle_iri,
            entities,
            activities,
            agents,
            relations,
        })
    }
}

// ── Query provenance tracker ──────────────────────────────────────────────────

/// Track the provenance of a SPARQL query execution
///
/// This captures who executed a query, when, against what dataset, and
/// what result dataset was produced. It can be exported as a PROV-O bundle.
#[derive(Debug, Clone)]
pub struct QueryProvenanceTracker {
    /// IRI identifying this specific query execution
    pub query_iri: NamedNode,
    /// When the query was executed (XSD dateTime string)
    pub executed_at: String,
    /// IRI of the software agent that ran the query
    pub executed_by: NamedNode,
    /// IRI of the input dataset (the graph queried over)
    pub input_dataset: NamedNode,
    /// IRI of the result dataset (the query output)
    pub result_dataset: NamedNode,
    /// Optional SPARQL query string
    pub query_text: Option<String>,
}

impl QueryProvenanceTracker {
    /// Create a new query provenance tracker
    pub fn new(
        query_iri: NamedNode,
        executed_at: String,
        executed_by: NamedNode,
        input_dataset: NamedNode,
        result_dataset: NamedNode,
    ) -> Self {
        Self {
            query_iri,
            executed_at,
            executed_by,
            input_dataset,
            result_dataset,
            query_text: None,
        }
    }

    /// Attach the original SPARQL query text as a prov:value attribute
    pub fn with_query_text(mut self, text: impl Into<String>) -> Self {
        self.query_text = Some(text.into());
        self
    }

    /// Convert this tracker to a PROV-O bundle
    ///
    /// The bundle represents:
    /// - `result_dataset` was generated by `query_iri`
    /// - `query_iri` used `input_dataset`
    /// - `query_iri` was associated with `executed_by`
    /// - `result_dataset` was attributed to `executed_by`
    pub fn to_bundle(&self) -> ProvBundle {
        let bundle_iri =
            NamedNode::new_unchecked(format!("{}/provenance", self.query_iri.as_str()));
        let mut bundle = ProvBundle::new(bundle_iri);

        // Entities: input and result datasets
        bundle.add_entity(ProvEntity::new(self.input_dataset.clone()));
        bundle.add_entity(ProvEntity::new(self.result_dataset.clone()));

        // Activity: the query execution itself
        let mut activity_attrs: Vec<(NamedNode, Object)> = Vec::new();
        if let Some(ref text) = self.query_text {
            activity_attrs.push((
                prov_iri("value"),
                Object::Literal(Literal::new(text.as_str())),
            ));
        }

        bundle.add_activity(ProvActivity::with_times(
            self.query_iri.clone(),
            Some(self.executed_at.clone()),
            Some(self.executed_at.clone()),
            activity_attrs,
        ));

        // Agent: the software agent
        bundle.add_agent(ProvAgent::new(
            self.executed_by.clone(),
            AgentType::SoftwareAgent,
        ));

        // Relations
        bundle.add_relation(ProvRelation::new(
            ProvRelationKind::WasGeneratedBy,
            self.result_dataset.clone(),
            self.query_iri.clone(),
        ));
        bundle.add_relation(ProvRelation::new(
            ProvRelationKind::Used,
            self.query_iri.clone(),
            self.input_dataset.clone(),
        ));
        bundle.add_relation(ProvRelation::new(
            ProvRelationKind::WasAssociatedWith,
            self.query_iri.clone(),
            self.executed_by.clone(),
        ));
        bundle.add_relation(ProvRelation::new(
            ProvRelationKind::WasAttributedTo,
            self.result_dataset.clone(),
            self.executed_by.clone(),
        ));

        bundle
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(triples.iter().any(|t| {
            matches!(t.object(), Object::NamedNode(o) if o.as_str().contains("Person"))
        }));
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
            relation.qualifier.as_ref().unwrap().as_str(),
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
        assert!(triples.iter().any(|t| {
            matches!(t.object(), Object::NamedNode(o) if o.as_str().contains("Bundle"))
        }));
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
        assert!(triples.iter().any(|t| {
            matches!(t.object(), Object::NamedNode(o) if o.as_str().contains("Entity"))
        }));
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
        assert!(triples.iter().any(|t| {
            matches!(t.object(), Object::NamedNode(o) if o.as_str().contains("Agent"))
        }));
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
}
