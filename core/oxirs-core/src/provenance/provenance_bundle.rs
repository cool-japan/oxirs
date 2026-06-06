//! PROV-O bundles and query-provenance tracking.
//!
//! This module defines [`ProvBundle`], a named collection of provenance
//! statements that maps naturally to RDF named graphs, and
//! [`QueryProvenanceTracker`], which captures the provenance of a SPARQL
//! query execution and exports it as a bundle.

use crate::model::{Literal, NamedNode, Object, Predicate, Subject, Triple};
use crate::OxirsError;
use std::collections::HashMap;

use super::provenance_types::{
    prov_iri, rdf_iri, AgentType, ProvActivity, ProvAgent, ProvEntity, ProvRelation,
    ProvRelationKind, PROV_NS, RDF_NS,
};

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
