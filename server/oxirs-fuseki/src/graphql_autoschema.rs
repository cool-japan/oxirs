//! RDF-vocabulary-driven GraphQL schema auto-generation, bridged from
//! [`oxirs-gql`](oxirs_gql).
//!
//! # Why this exists
//!
//! The `/graphql` endpoint in [`crate::graphql_integration`] serves a *fixed*,
//! hand-written `async-graphql` catalog schema (datasets / sparqlQuery / triples
//! / search / statistics). That is useful for administration but says nothing
//! about the *shape of the data*: a client cannot query `person { name }` unless
//! someone hand-writes a `Person` type.
//!
//! The sibling `oxirs-gql` crate solves exactly that: its
//! [`SchemaGenerator`](oxirs_gql::schema_generator::SchemaGenerator) renders a
//! GraphQL SDL from an [`RdfVocabulary`](oxirs_gql::schema_types::RdfVocabulary)
//! (classes and properties with domains/ranges) so the object types and fields
//! mirror the ontology.
//!
//! # Why we build the vocabulary here (and not via `generate_from_store`)
//!
//! `oxirs-gql`'s own `generate_from_store` extracts the vocabulary by running
//! `PREFIX`/`UNION`/`OPTIONAL`/`FILTER` SPARQL against
//! [`oxirs_gql::RdfStore`]'s built-in (deliberately simplified) query engine,
//! which cannot execute those forms — so it silently yields an empty schema.
//! This module instead **infers the vocabulary directly from the triple
//! stream** (honouring explicit `rdfs:Class` / `owl:*Property` /
//! `rdfs:domain` / `rdfs:range` declarations when present, and falling back to
//! instance-`rdf:type` and predicate-usage inference otherwise) and hands the
//! finished vocabulary to oxirs-gql's public
//! [`SchemaGenerator::generate_sdl_from_vocabulary`]. This works on any dataset,
//! ontology-annotated or not.
//!
//! It is **purely additive** — the existing `async-graphql` `/graphql` handler
//! is untouched. `oxirs-gql` uses `juniper` internally, which coexists with
//! `async-graphql` because only the generated *SDL string* crosses the boundary.
//!
//! # Status
//!
//! This wires up schema *generation*. Executing GraphQL queries against the
//! generated schema (GraphQL → SPARQL translation over the live fuseki store)
//! is the larger follow-up tracked in the integration design memo.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, Result};
use oxirs_core::model::{Object, Predicate, Subject, Triple};
use oxirs_gql::schema_generator::SchemaGenerator;
use oxirs_gql::schema_types::{PropertyType, RdfClass, RdfProperty, RdfVocabulary};

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDF_PROPERTY: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property";
const RDFS_CLASS: &str = "http://www.w3.org/2000/01/rdf-schema#Class";
const RDFS_DOMAIN: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
const RDFS_RANGE: &str = "http://www.w3.org/2000/01/rdf-schema#range";
const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
const RDFS_COMMENT: &str = "http://www.w3.org/2000/01/rdf-schema#comment";
const OWL_CLASS: &str = "http://www.w3.org/2002/07/owl#Class";
const OWL_DATATYPE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#DatatypeProperty";
const OWL_OBJECT_PROPERTY: &str = "http://www.w3.org/2002/07/owl#ObjectProperty";
const OWL_ANNOTATION_PROPERTY: &str = "http://www.w3.org/2002/07/owl#AnnotationProperty";
const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";

/// Generate a GraphQL SDL from a set of RDF triples.
///
/// The triples are scanned to infer an [`RdfVocabulary`] (see the module docs
/// for the inference rules) which is rendered to SDL by oxirs-gql. Supplying
/// ontology triples (`Class a rdfs:Class`, `prop a owl:DatatypeProperty`,
/// `rdfs:domain` / `rdfs:range`) yields the richest schema; plain instance data
/// still yields usable object types via `rdf:type` / predicate-usage inference.
pub fn generate_sdl_from_triples<'a, I>(triples: I) -> Result<String>
where
    I: IntoIterator<Item = &'a Triple>,
{
    let vocabulary = build_vocabulary_from_triples(triples);
    if vocabulary.classes.is_empty() {
        return Err(anyhow!(
            "no RDF classes could be inferred from the dataset (no rdfs:Class / owl:Class \
             declarations and no rdf:type instance data); cannot auto-generate a GraphQL schema"
        ));
    }
    SchemaGenerator::new()
        .generate_sdl_from_vocabulary(vocabulary)
        .map_err(|e| anyhow!("oxirs-gql SDL rendering failed: {e}"))
}

/// Generate a GraphQL SDL for a fuseki dataset by extracting its triples with a
/// `CONSTRUCT` query and delegating to [`generate_sdl_from_triples`].
///
/// `dataset` selects the named dataset (`None` = the default dataset). The whole
/// dataset is materialized; callers on large datasets should prefer a streaming
/// store adapter once it lands (see the module docs).
pub fn generate_sdl_for_dataset(
    store: &crate::store::Store,
    dataset: Option<&str>,
) -> Result<String> {
    let results = store
        .query_dataset("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }", dataset)
        .map_err(|e| anyhow!("dataset CONSTRUCT query failed: {e}"))?;

    match &results.inner {
        oxirs_core::query::QueryResult::Construct(triples) => {
            generate_sdl_from_triples(triples.iter())
        }
        other => Err(anyhow!(
            "expected a CONSTRUCT result to introspect the vocabulary, got {:?}",
            std::mem::discriminant(other)
        )),
    }
}

// ── vocabulary inference ────────────────────────────────────────────────────

fn subject_iri(s: &Subject) -> Option<&str> {
    match s {
        Subject::NamedNode(n) => Some(n.as_str()),
        _ => None,
    }
}

fn predicate_iri(p: &Predicate) -> &str {
    match p {
        Predicate::NamedNode(n) => n.as_str(),
        Predicate::Variable(v) => v.as_str(),
    }
}

/// What an object term contributes to inference: a resource IRI (→ object
/// property), a typed/plain literal with its datatype IRI (→ data property), or
/// something we ignore for schema purposes (blank nodes, quoted triples).
enum ObjInfo<'a> {
    Iri(&'a str),
    Literal(String),
    Ignore,
}

fn object_info(o: &Object) -> ObjInfo<'_> {
    match o {
        Object::NamedNode(n) => ObjInfo::Iri(n.as_str()),
        Object::Literal(l) => ObjInfo::Literal(l.datatype().as_str().to_string()),
        _ => ObjInfo::Ignore,
    }
}

/// Infer an [`RdfVocabulary`] from a triple stream.
///
/// Rules (explicit declarations always win over inference):
/// * A class is any subject of `?c a rdfs:Class` / `owl:Class`, plus any object
///   of an `?s rdf:type ?c` instance triple.
/// * A property is any subject of `?p a owl:{Datatype,Object,Annotation}Property`
///   / `rdf:Property`, plus any predicate actually used in the data (other than
///   the RDF/RDFS schema vocabulary).
/// * `rdfs:domain` / `rdfs:range` / `rdfs:label` / `rdfs:comment` refine a
///   property; missing domain/range are inferred from the subject's `rdf:type`s
///   and the object kind (IRI → object property, literal → data property with
///   the literal's datatype, defaulting to `xsd:string`).
/// * Each property is attached to every class in its (declared or inferred)
///   domain.
fn build_vocabulary_from_triples<'a, I>(triples: I) -> RdfVocabulary
where
    I: IntoIterator<Item = &'a Triple>,
{
    let mut classes: BTreeSet<String> = BTreeSet::new();
    let mut prop_types: BTreeMap<String, PropertyType> = BTreeMap::new();
    let mut prop_domains: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut prop_ranges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut prop_labels: BTreeMap<String, String> = BTreeMap::new();
    let mut prop_comments: BTreeMap<String, String> = BTreeMap::new();
    let mut class_labels: BTreeMap<String, String> = BTreeMap::new();
    let mut class_comments: BTreeMap<String, String> = BTreeMap::new();
    // subject IRI → its rdf:type class IRIs (for domain inference).
    let mut subject_types: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    // used predicate → observed object kind (IRI vs literal-with-datatype).
    let mut used_predicates: BTreeMap<String, ObservedKind> = BTreeMap::new();

    // First pass: classify triples.
    let triples: Vec<&Triple> = triples.into_iter().collect();
    for triple in &triples {
        let Some(subj) = subject_iri(triple.subject()) else {
            continue;
        };
        let pred = predicate_iri(triple.predicate());
        let obj = object_info(triple.object());

        match pred {
            RDF_TYPE => {
                if let ObjInfo::Iri(type_iri) = obj {
                    match type_iri {
                        RDFS_CLASS | OWL_CLASS => {
                            classes.insert(subj.to_string());
                        }
                        OWL_DATATYPE_PROPERTY => {
                            prop_types.insert(subj.to_string(), PropertyType::DataProperty);
                        }
                        OWL_OBJECT_PROPERTY => {
                            prop_types.insert(subj.to_string(), PropertyType::ObjectProperty);
                        }
                        OWL_ANNOTATION_PROPERTY | RDF_PROPERTY => {
                            prop_types
                                .entry(subj.to_string())
                                .or_insert(PropertyType::AnnotationProperty);
                        }
                        // Ordinary instance typing: the object is a class.
                        _ => {
                            classes.insert(type_iri.to_string());
                            subject_types
                                .entry(subj.to_string())
                                .or_default()
                                .insert(type_iri.to_string());
                        }
                    }
                }
            }
            RDFS_DOMAIN => {
                if let ObjInfo::Iri(dom) = obj {
                    prop_domains
                        .entry(subj.to_string())
                        .or_default()
                        .insert(dom.to_string());
                    classes.insert(dom.to_string());
                }
            }
            RDFS_RANGE => {
                if let ObjInfo::Iri(rng) = obj {
                    prop_ranges
                        .entry(subj.to_string())
                        .or_default()
                        .insert(rng.to_string());
                }
            }
            RDFS_LABEL => {
                if let ObjInfo::Literal(_) = obj {
                    if let Object::Literal(l) = triple.object() {
                        class_labels
                            .entry(subj.to_string())
                            .or_insert_with(|| l.value().to_string());
                        prop_labels
                            .entry(subj.to_string())
                            .or_insert_with(|| l.value().to_string());
                    }
                }
            }
            RDFS_COMMENT => {
                if let Object::Literal(l) = triple.object() {
                    class_comments
                        .entry(subj.to_string())
                        .or_insert_with(|| l.value().to_string());
                    prop_comments
                        .entry(subj.to_string())
                        .or_insert_with(|| l.value().to_string());
                }
            }
            // Any other predicate is a data/object property used by the data.
            _ => {
                let kind = match &obj {
                    ObjInfo::Iri(_) => ObservedKind::Resource,
                    ObjInfo::Literal(dt) => ObservedKind::Literal(dt.clone()),
                    ObjInfo::Ignore => ObservedKind::Unknown,
                };
                used_predicates
                    .entry(pred.to_string())
                    .and_modify(|existing| existing.merge(&kind))
                    .or_insert(kind);
            }
        }
    }

    // Second pass: fold predicate usage into property metadata (only when the
    // predicate was not explicitly declared with domain/range).
    for (pred, kind) in &used_predicates {
        prop_types.entry(pred.clone()).or_insert(match kind {
            ObservedKind::Resource => PropertyType::ObjectProperty,
            _ => PropertyType::DataProperty,
        });
        // Infer domain from the rdf:type of subjects that use this predicate.
        if !prop_domains.contains_key(pred) {
            let mut inferred_domains: BTreeSet<String> = BTreeSet::new();
            for triple in &triples {
                if predicate_iri(triple.predicate()) != pred {
                    continue;
                }
                if let Some(s) = subject_iri(triple.subject()) {
                    if let Some(types) = subject_types.get(s) {
                        inferred_domains.extend(types.iter().cloned());
                    }
                }
            }
            if !inferred_domains.is_empty() {
                prop_domains.insert(pred.clone(), inferred_domains);
            }
        }
        // Infer range: the literal datatype, or xsd:string as a safe default.
        if !prop_ranges.contains_key(pred) {
            if let ObservedKind::Literal(dt) = kind {
                let range = if dt.is_empty() {
                    XSD_STRING.to_string()
                } else {
                    dt.clone()
                };
                prop_ranges.entry(pred.clone()).or_default().insert(range);
            }
        }
    }

    // Build the vocabulary maps. Attach each property to the classes in its
    // domain so the generator renders it as a field on those object types.
    let mut class_props: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (prop, domains) in &prop_domains {
        for dom in domains {
            class_props
                .entry(dom.clone())
                .or_default()
                .insert(prop.clone());
        }
    }

    let properties: BTreeMap<String, RdfProperty> = prop_types
        .keys()
        .map(|uri| {
            let property = RdfProperty {
                uri: uri.clone(),
                label: prop_labels.get(uri).cloned(),
                comment: prop_comments.get(uri).cloned(),
                domain: prop_domains
                    .get(uri)
                    .map(|d| d.iter().cloned().collect())
                    .unwrap_or_default(),
                range: prop_ranges
                    .get(uri)
                    .map(|r| r.iter().cloned().collect())
                    .unwrap_or_default(),
                property_type: prop_types
                    .get(uri)
                    .cloned()
                    .unwrap_or(PropertyType::AnnotationProperty),
                functional: false,
                inverse_functional: false,
            };
            (uri.clone(), property)
        })
        .collect();

    let classes: BTreeMap<String, RdfClass> = classes
        .into_iter()
        .map(|uri| {
            let class = RdfClass {
                label: class_labels.get(&uri).cloned(),
                comment: class_comments.get(&uri).cloned(),
                super_classes: Vec::new(),
                properties: class_props
                    .get(&uri)
                    .map(|p| p.iter().cloned().collect())
                    .unwrap_or_default(),
                uri: uri.clone(),
            };
            (uri, class)
        })
        .collect();

    RdfVocabulary {
        classes: classes.into_iter().collect(),
        properties: properties.into_iter().collect(),
        namespaces: default_namespaces(),
    }
}

fn default_namespaces() -> std::collections::HashMap<String, String> {
    let mut ns = std::collections::HashMap::new();
    ns.insert(
        "rdf".to_string(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
    );
    ns.insert(
        "rdfs".to_string(),
        "http://www.w3.org/2000/01/rdf-schema#".to_string(),
    );
    ns.insert(
        "owl".to_string(),
        "http://www.w3.org/2002/07/owl#".to_string(),
    );
    ns
}

/// The kind of object a predicate has been observed with, used to decide
/// data- vs object-property and to infer a range.
enum ObservedKind {
    Resource,
    Literal(String),
    Unknown,
}

impl ObservedKind {
    fn merge(&mut self, other: &ObservedKind) {
        // A predicate seen with any resource object is treated as an object
        // property; otherwise the first concrete literal datatype wins.
        if matches!(other, ObservedKind::Resource) {
            *self = ObservedKind::Resource;
        } else if matches!(self, ObservedKind::Unknown) {
            *self = other.clone_kind();
        }
    }

    fn clone_kind(&self) -> ObservedKind {
        match self {
            ObservedKind::Resource => ObservedKind::Resource,
            ObservedKind::Literal(dt) => ObservedKind::Literal(dt.clone()),
            ObservedKind::Unknown => ObservedKind::Unknown,
        }
    }
}
