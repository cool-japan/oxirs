//! Core PROV-O ontology types: agents, entities, activities, and relations.
//!
//! This module defines the core W3C PROV-O classes (Entity, Activity, Agent)
//! and the relations that connect them, along with namespace constants and
//! IRI-construction helpers shared across the provenance module.

use crate::model::{Literal, NamedNode, Object, Triple};

/// PROV-O namespace prefix
pub const PROV_NS: &str = "http://www.w3.org/ns/prov#";

/// XSD namespace prefix
pub const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";

/// RDF namespace prefix
pub const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

/// Build a PROV-O IRI from a local name
pub(crate) fn prov_iri(local: &str) -> NamedNode {
    NamedNode::new_unchecked(format!("{PROV_NS}{local}"))
}

/// Build an XSD datatype IRI
pub(crate) fn xsd_iri(local: &str) -> NamedNode {
    NamedNode::new_unchecked(format!("{XSD_NS}{local}"))
}

/// Build an RDF IRI
pub(crate) fn rdf_iri(local: &str) -> NamedNode {
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
