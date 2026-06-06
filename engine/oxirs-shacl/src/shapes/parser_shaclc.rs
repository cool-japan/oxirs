#![allow(dead_code)]

use std::collections::HashSet;

use oxirs_core::{
    graph::Graph,
    model::{BlankNode, NamedNode, Object, Predicate, Subject, Term},
};

use crate::{
    constraints::{
        cardinality_constraints::{MaxCountConstraint, MinCountConstraint},
        comparison_constraints::{
            DisjointConstraint, EqualsConstraint, HasValueConstraint, InConstraint,
            LessThanConstraint, LessThanOrEqualsConstraint,
        },
        range_constraints::{
            MaxExclusiveConstraint, MaxInclusiveConstraint, MinExclusiveConstraint,
            MinInclusiveConstraint,
        },
        string_constraints::{
            LanguageInConstraint, MaxLengthConstraint, MinLengthConstraint, PatternConstraint,
            UniqueLangConstraint,
        },
        value_constraints::{ClassConstraint, DatatypeConstraint, NodeKindConstraint},
        Constraint,
    },
    paths::PropertyPath,
    Result, ShaclError, Shape, ShapeType, SHACL_VOCAB,
};

use super::parser_types::{
    extract_bool_literal, extract_integer_literal, extract_named_node, rdf_first_node,
    rdf_nil_node, rdf_rest_node,
};

pub(crate) fn determine_shape_type(graph: &Graph, shape_node: &NamedNode) -> Result<ShapeType> {
    let rdf_type = super::parser_types::rdf_type_node()?;

    let node_shape_type = NamedNode::new("http://www.w3.org/ns/shacl#NodeShape")
        .map_err(|e| ShaclError::ShapeParsing(format!("Invalid NodeShape IRI: {e}")))?;

    let property_shape_type = NamedNode::new("http://www.w3.org/ns/shacl#PropertyShape")
        .map_err(|e| ShaclError::ShapeParsing(format!("Invalid PropertyShape IRI: {e}")))?;

    let type_triples = graph.query_triples(
        Some(&Subject::NamedNode(shape_node.clone())),
        Some(&Predicate::NamedNode(rdf_type)),
        None,
    );

    for triple in type_triples {
        if let Object::NamedNode(type_node) = triple.object() {
            if *type_node == property_shape_type {
                return Ok(ShapeType::PropertyShape);
            } else if *type_node == node_shape_type {
                return Ok(ShapeType::NodeShape);
            }
        }
    }

    let path_prop = NamedNode::new("http://www.w3.org/ns/shacl#path")
        .map_err(|e| ShaclError::ShapeParsing(format!("Invalid path property IRI: {e}")))?;

    let path_triples = graph.query_triples(
        Some(&Subject::NamedNode(shape_node.clone())),
        Some(&Predicate::NamedNode(path_prop)),
        None,
    );

    if !path_triples.is_empty() {
        return Ok(ShapeType::PropertyShape);
    }

    Ok(ShapeType::NodeShape)
}

pub(crate) fn parse_node_kind(
    node_kind_iri: &NamedNode,
) -> Result<crate::constraints::value_constraints::NodeKind> {
    use crate::constraints::value_constraints::NodeKind;

    match node_kind_iri.as_str() {
        "http://www.w3.org/ns/shacl#IRI" => Ok(NodeKind::Iri),
        "http://www.w3.org/ns/shacl#BlankNode" => Ok(NodeKind::BlankNode),
        "http://www.w3.org/ns/shacl#Literal" => Ok(NodeKind::Literal),
        "http://www.w3.org/ns/shacl#BlankNodeOrIRI" => Ok(NodeKind::BlankNodeOrIri),
        "http://www.w3.org/ns/shacl#BlankNodeOrLiteral" => Ok(NodeKind::BlankNodeOrLiteral),
        "http://www.w3.org/ns/shacl#IRIOrLiteral" => Ok(NodeKind::IriOrLiteral),
        _ => Err(ShaclError::ShapeParsing(format!(
            "Unknown node kind: {node_kind_iri}"
        ))),
    }
}

pub(crate) fn parse_severity(severity_iri: &NamedNode) -> Result<crate::Severity> {
    use crate::Severity;

    match severity_iri.as_str() {
        "http://www.w3.org/ns/shacl#Violation" => Ok(Severity::Violation),
        "http://www.w3.org/ns/shacl#Warning" => Ok(Severity::Warning),
        "http://www.w3.org/ns/shacl#Info" => Ok(Severity::Info),
        _ => Err(ShaclError::ShapeParsing(format!(
            "Unknown severity: {severity_iri}"
        ))),
    }
}

pub(crate) fn get_named_node_object(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Option<NamedNode>> {
    let triples = graph.query_triples(
        Some(subject),
        Some(&Predicate::NamedNode(predicate.clone())),
        None,
    );
    for triple in triples {
        if let Some(n) = extract_named_node(triple.object()) {
            return Ok(Some(n));
        }
    }
    Ok(None)
}

pub(crate) fn get_literal_object(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Option<oxirs_core::model::Literal>> {
    let triples = graph.query_triples(
        Some(subject),
        Some(&Predicate::NamedNode(predicate.clone())),
        None,
    );
    for triple in triples {
        if let Object::Literal(lit) = triple.object() {
            return Ok(Some(lit.clone()));
        }
    }
    Ok(None)
}

pub(crate) fn get_string_object(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Option<String>> {
    if let Some(literal) = get_literal_object(graph, subject, predicate)? {
        return Ok(Some(literal.value().to_string()));
    }
    Ok(None)
}

pub(crate) fn get_string_with_language(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Option<(String, String)>> {
    if let Some(literal) = get_literal_object(graph, subject, predicate)? {
        let value = literal.value().to_string();
        let lang_tag = literal.language().unwrap_or("").to_string();
        return Ok(Some((value, lang_tag)));
    }
    Ok(None)
}

pub(crate) fn get_integer_object(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Option<i64>> {
    if let Some(literal) = get_literal_object(graph, subject, predicate)? {
        return Ok(extract_integer_literal(&Object::Literal(literal)));
    }
    Ok(None)
}

pub(crate) fn get_boolean_object(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Option<bool>> {
    if let Some(literal) = get_literal_object(graph, subject, predicate)? {
        return Ok(extract_bool_literal(&Object::Literal(literal)));
    }
    Ok(None)
}

pub(crate) fn get_named_node_for_subject(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Option<NamedNode>> {
    get_named_node_object(graph, subject, predicate)
}

pub(crate) fn get_string_for_subject(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Option<String>> {
    get_string_object(graph, subject, predicate)
}

pub(crate) fn get_string_with_language_for_subject(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Option<(String, String)>> {
    get_string_with_language(graph, subject, predicate)
}

pub(crate) fn get_integer_for_subject(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Option<i64>> {
    get_integer_object(graph, subject, predicate)
}

pub(crate) fn get_numeric_for_subject(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Option<oxirs_core::model::Literal>> {
    let triples = graph.query_triples(
        Some(subject),
        Some(&Predicate::NamedNode(predicate.clone())),
        None,
    );
    for triple in triples {
        if let Object::Literal(literal) = triple.object() {
            return Ok(Some(literal.clone()));
        }
    }
    Ok(None)
}

pub(crate) fn parse_property_path_object(
    graph: &Graph,
    path_object: &Object,
) -> Result<PropertyPath> {
    match path_object {
        Object::NamedNode(node) => Ok(PropertyPath::predicate(node.clone())),
        Object::BlankNode(blank_node) => parse_complex_property_path(graph, blank_node),
        _ => Err(ShaclError::ShapeParsing(
            "Invalid property path object type".to_string(),
        )),
    }
}

pub(crate) fn parse_complex_property_path(
    graph: &Graph,
    blank_node: &BlankNode,
) -> Result<PropertyPath> {
    if let Some(p) = parse_inverse_path(graph, blank_node)? {
        return Ok(p);
    }
    if let Some(p) = parse_alternative_path(graph, blank_node)? {
        return Ok(p);
    }
    if let Some(p) = parse_sequence_path(graph, blank_node)? {
        return Ok(p);
    }
    if let Some(p) = parse_zero_or_more_path(graph, blank_node)? {
        return Ok(p);
    }
    if let Some(p) = parse_one_or_more_path(graph, blank_node)? {
        return Ok(p);
    }
    if let Some(p) = parse_zero_or_one_path(graph, blank_node)? {
        return Ok(p);
    }
    Err(ShaclError::ShapeParsing(
        "Unrecognized complex property path structure".to_string(),
    ))
}

fn parse_inverse_path(graph: &Graph, blank_node: &BlankNode) -> Result<Option<PropertyPath>> {
    let inverse_prop = NamedNode::new("http://www.w3.org/ns/shacl#inversePath")
        .map_err(|e| ShaclError::ShapeParsing(format!("Invalid inversePath property IRI: {e}")))?;

    let triples = graph.query_triples(
        Some(&Subject::BlankNode(blank_node.clone())),
        Some(&Predicate::NamedNode(inverse_prop)),
        None,
    );

    for triple in triples {
        if let Object::NamedNode(property_node) = triple.object() {
            return Ok(Some(PropertyPath::inverse(PropertyPath::predicate(
                property_node.clone(),
            ))));
        }
    }
    Ok(None)
}

fn parse_alternative_path(graph: &Graph, blank_node: &BlankNode) -> Result<Option<PropertyPath>> {
    let alternative_prop =
        NamedNode::new("http://www.w3.org/ns/shacl#alternativePath").map_err(|e| {
            ShaclError::ShapeParsing(format!("Invalid alternativePath property IRI: {e}"))
        })?;

    let triples = graph.query_triples(
        Some(&Subject::BlankNode(blank_node.clone())),
        Some(&Predicate::NamedNode(alternative_prop)),
        None,
    );

    for triple in triples {
        let paths = parse_rdf_list_as_paths(graph, triple.object())?;
        if !paths.is_empty() {
            return Ok(Some(PropertyPath::Alternative(paths)));
        }
    }
    Ok(None)
}

fn parse_sequence_path(graph: &Graph, blank_node: &BlankNode) -> Result<Option<PropertyPath>> {
    let first_prop = rdf_first_node()?;

    let first_triples = graph.query_triples(
        Some(&Subject::BlankNode(blank_node.clone())),
        Some(&Predicate::NamedNode(first_prop)),
        None,
    );

    if !first_triples.is_empty() {
        let paths = parse_rdf_list_as_paths(graph, &Object::BlankNode(blank_node.clone()))?;
        if paths.len() > 1 {
            return Ok(Some(PropertyPath::Sequence(paths)));
        }
    }
    Ok(None)
}

fn parse_zero_or_more_path(graph: &Graph, blank_node: &BlankNode) -> Result<Option<PropertyPath>> {
    let zero_or_more_prop =
        NamedNode::new("http://www.w3.org/ns/shacl#zeroOrMorePath").map_err(|e| {
            ShaclError::ShapeParsing(format!("Invalid zeroOrMorePath property IRI: {e}"))
        })?;

    let triples = graph.query_triples(
        Some(&Subject::BlankNode(blank_node.clone())),
        Some(&Predicate::NamedNode(zero_or_more_prop)),
        None,
    );

    if let Some(triple) = triples.into_iter().next() {
        let inner_path = parse_property_path_object(graph, triple.object())?;
        return Ok(Some(PropertyPath::ZeroOrMore(Box::new(inner_path))));
    }
    Ok(None)
}

fn parse_one_or_more_path(graph: &Graph, blank_node: &BlankNode) -> Result<Option<PropertyPath>> {
    let one_or_more_prop =
        NamedNode::new("http://www.w3.org/ns/shacl#oneOrMorePath").map_err(|e| {
            ShaclError::ShapeParsing(format!("Invalid oneOrMorePath property IRI: {e}"))
        })?;

    let triples = graph.query_triples(
        Some(&Subject::BlankNode(blank_node.clone())),
        Some(&Predicate::NamedNode(one_or_more_prop)),
        None,
    );

    if let Some(triple) = triples.into_iter().next() {
        let inner_path = parse_property_path_object(graph, triple.object())?;
        return Ok(Some(PropertyPath::OneOrMore(Box::new(inner_path))));
    }
    Ok(None)
}

fn parse_zero_or_one_path(graph: &Graph, blank_node: &BlankNode) -> Result<Option<PropertyPath>> {
    let zero_or_one_prop =
        NamedNode::new("http://www.w3.org/ns/shacl#zeroOrOnePath").map_err(|e| {
            ShaclError::ShapeParsing(format!("Invalid zeroOrOnePath property IRI: {e}"))
        })?;

    let triples = graph.query_triples(
        Some(&Subject::BlankNode(blank_node.clone())),
        Some(&Predicate::NamedNode(zero_or_one_prop)),
        None,
    );

    if let Some(triple) = triples.into_iter().next() {
        let inner_path = parse_property_path_object(graph, triple.object())?;
        return Ok(Some(PropertyPath::ZeroOrOne(Box::new(inner_path))));
    }
    Ok(None)
}

pub(crate) fn parse_rdf_list_as_paths(
    graph: &Graph,
    list_object: &Object,
) -> Result<Vec<PropertyPath>> {
    let mut paths = Vec::new();
    let mut current = list_object.clone();

    let first_prop = rdf_first_node()?;
    let rest_prop = rdf_rest_node()?;
    let nil_node = rdf_nil_node()?;

    loop {
        match &current {
            Object::BlankNode(blank_node) => {
                let first_triples = graph.query_triples(
                    Some(&Subject::BlankNode(blank_node.clone())),
                    Some(&Predicate::NamedNode(first_prop.clone())),
                    None,
                );

                if let Some(triple) = first_triples.into_iter().next() {
                    let path = parse_property_path_object(graph, triple.object())?;
                    paths.push(path);
                } else {
                    break;
                }

                let rest_triples = graph.query_triples(
                    Some(&Subject::BlankNode(blank_node.clone())),
                    Some(&Predicate::NamedNode(rest_prop.clone())),
                    None,
                );

                if let Some(triple) = rest_triples.into_iter().next() {
                    current = triple.object().clone();
                } else {
                    break;
                }
            }
            Object::NamedNode(node) => {
                if *node == nil_node {
                    break;
                } else {
                    let path = parse_property_path_object(graph, &current)?;
                    paths.push(path);
                    break;
                }
            }
            _ => break,
        }
    }

    Ok(paths)
}

pub(crate) fn parse_rdf_list_to_terms(graph: &Graph, list_object: &Object) -> Result<Vec<Term>> {
    let mut terms = Vec::new();
    let mut current = list_object.clone();

    let first_prop = rdf_first_node()?;
    let rest_prop = rdf_rest_node()?;
    let nil_node = rdf_nil_node()?;

    loop {
        match &current {
            Object::BlankNode(blank_node) => {
                let first_triples = graph.query_triples(
                    Some(&Subject::BlankNode(blank_node.clone())),
                    Some(&Predicate::NamedNode(first_prop.clone())),
                    None,
                );

                if let Some(triple) = first_triples.into_iter().next() {
                    let term = match triple.object() {
                        Object::NamedNode(node) => Term::NamedNode(node.clone()),
                        Object::BlankNode(node) => Term::BlankNode(node.clone()),
                        Object::Literal(lit) => Term::Literal(lit.clone()),
                        Object::Variable(_) | Object::QuotedTriple(_) => continue,
                    };
                    terms.push(term);
                } else {
                    break;
                }

                let rest_triples = graph.query_triples(
                    Some(&Subject::BlankNode(blank_node.clone())),
                    Some(&Predicate::NamedNode(rest_prop.clone())),
                    None,
                );

                if let Some(triple) = rest_triples.into_iter().next() {
                    current = triple.object().clone();
                } else {
                    break;
                }
            }
            Object::NamedNode(node) => {
                if *node == nil_node {
                    break;
                } else {
                    terms.push(Term::NamedNode(node.clone()));
                    break;
                }
            }
            Object::Literal(lit) => {
                terms.push(Term::Literal(lit.clone()));
                break;
            }
            _ => break,
        }
    }

    Ok(terms)
}

pub(crate) fn parse_rdf_list_as_terms(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Vec<Term>> {
    let mut terms = Vec::new();

    let triples = graph.query_triples(
        Some(subject),
        Some(&Predicate::NamedNode(predicate.clone())),
        None,
    );

    for triple in triples {
        let list_terms = parse_rdf_list_to_terms(graph, triple.object())?;
        terms.extend(list_terms);
    }

    Ok(terms)
}

pub(crate) fn parse_rdf_list_as_strings(
    graph: &Graph,
    subject: &Subject,
    predicate: &NamedNode,
) -> Result<Vec<String>> {
    let terms = parse_rdf_list_as_terms(graph, subject, predicate)?;
    let mut strings = Vec::new();

    for term in terms {
        if let Term::Literal(lit) = term {
            strings.push(lit.value().to_string());
        }
    }

    Ok(strings)
}

pub(crate) fn parse_constraints_from_graph(
    graph: &Graph,
    shape_node: &NamedNode,
    shape: &mut Shape,
) -> Result<()> {
    let shape_subject = Subject::NamedNode(shape_node.clone());

    if let Some(class_iri) = get_named_node_object(graph, &shape_subject, &SHACL_VOCAB.class)? {
        let constraint = Constraint::Class(ClassConstraint { class_iri });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(datatype_iri) = get_named_node_object(graph, &shape_subject, &SHACL_VOCAB.datatype)?
    {
        let constraint = Constraint::Datatype(DatatypeConstraint { datatype_iri });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(node_kind_iri) =
        get_named_node_object(graph, &shape_subject, &SHACL_VOCAB.node_kind)?
    {
        let node_kind = parse_node_kind(&node_kind_iri)?;
        let constraint = Constraint::NodeKind(NodeKindConstraint { node_kind });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(min_count) = get_integer_object(graph, &shape_subject, &SHACL_VOCAB.min_count)? {
        let constraint = Constraint::MinCount(MinCountConstraint {
            min_count: min_count.try_into().unwrap_or(0),
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(max_count) = get_integer_object(graph, &shape_subject, &SHACL_VOCAB.max_count)? {
        let constraint = Constraint::MaxCount(MaxCountConstraint {
            max_count: max_count.try_into().unwrap_or(0),
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(min_exclusive) =
        get_literal_object(graph, &shape_subject, &SHACL_VOCAB.min_exclusive)?
    {
        let constraint = Constraint::MinExclusive(MinExclusiveConstraint {
            min_value: min_exclusive,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(max_exclusive) =
        get_literal_object(graph, &shape_subject, &SHACL_VOCAB.max_exclusive)?
    {
        let constraint = Constraint::MaxExclusive(MaxExclusiveConstraint {
            max_value: max_exclusive,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(min_inclusive) =
        get_literal_object(graph, &shape_subject, &SHACL_VOCAB.min_inclusive)?
    {
        let constraint = Constraint::MinInclusive(MinInclusiveConstraint {
            min_value: min_inclusive,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(max_inclusive) =
        get_literal_object(graph, &shape_subject, &SHACL_VOCAB.max_inclusive)?
    {
        let constraint = Constraint::MaxInclusive(MaxInclusiveConstraint {
            max_value: max_inclusive,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(min_length) = get_integer_object(graph, &shape_subject, &SHACL_VOCAB.min_length)? {
        let constraint = Constraint::MinLength(MinLengthConstraint {
            min_length: min_length.try_into().unwrap_or(0),
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(max_length) = get_integer_object(graph, &shape_subject, &SHACL_VOCAB.max_length)? {
        let constraint = Constraint::MaxLength(MaxLengthConstraint {
            max_length: max_length.try_into().unwrap_or(0),
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(pattern) = get_string_object(graph, &shape_subject, &SHACL_VOCAB.pattern)? {
        let flags = get_string_object(graph, &shape_subject, &SHACL_VOCAB.flags)?;
        let constraint = Constraint::Pattern(PatternConstraint {
            pattern,
            flags,
            message: None,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(unique_lang) = get_boolean_object(graph, &shape_subject, &SHACL_VOCAB.unique_lang)?
    {
        let constraint = Constraint::UniqueLang(UniqueLangConstraint { unique_lang });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(equals_property) =
        get_named_node_object(graph, &shape_subject, &SHACL_VOCAB.equals)?
    {
        let constraint = Constraint::Equals(EqualsConstraint {
            property: Term::NamedNode(equals_property),
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(disjoint_property) =
        get_named_node_object(graph, &shape_subject, &SHACL_VOCAB.disjoint)?
    {
        let constraint = Constraint::Disjoint(DisjointConstraint {
            property: Term::NamedNode(disjoint_property),
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(less_than_property) =
        get_named_node_object(graph, &shape_subject, &SHACL_VOCAB.less_than)?
    {
        let constraint = Constraint::LessThan(LessThanConstraint {
            property: Term::NamedNode(less_than_property),
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(less_than_or_equals_property) =
        get_named_node_object(graph, &shape_subject, &SHACL_VOCAB.less_than_or_equals)?
    {
        let constraint = Constraint::LessThanOrEquals(LessThanOrEqualsConstraint {
            property: Term::NamedNode(less_than_or_equals_property),
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    let in_values = parse_rdf_list_as_terms(graph, &shape_subject, &SHACL_VOCAB.in_list)?;
    if !in_values.is_empty() {
        let constraint = Constraint::In(InConstraint { values: in_values });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    let has_value_triples = graph.query_triples(
        Some(&shape_subject),
        Some(&Predicate::NamedNode(SHACL_VOCAB.has_value.clone())),
        None,
    );
    for triple in has_value_triples {
        let value_term = match triple.object() {
            Object::NamedNode(node) => Term::NamedNode(node.clone()),
            Object::BlankNode(node) => Term::BlankNode(node.clone()),
            Object::Literal(lit) => Term::Literal(lit.clone()),
            Object::Variable(_) | Object::QuotedTriple(_) => continue,
        };
        let constraint = Constraint::HasValue(HasValueConstraint { value: value_term });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    let language_in_values =
        parse_rdf_list_as_strings(graph, &shape_subject, &SHACL_VOCAB.language_in)?;
    if !language_in_values.is_empty() {
        let constraint = Constraint::LanguageIn(LanguageInConstraint {
            languages: language_in_values,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    Ok(())
}

pub(crate) fn parse_constraints_from_blank_node(
    graph: &Graph,
    blank_subject: &Subject,
    shape: &mut Shape,
) -> Result<()> {
    if let Some(min_count) = get_integer_for_subject(graph, blank_subject, &SHACL_VOCAB.min_count)?
    {
        let constraint = Constraint::MinCount(MinCountConstraint {
            min_count: min_count as u32,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(max_count) = get_integer_for_subject(graph, blank_subject, &SHACL_VOCAB.max_count)?
    {
        let constraint = Constraint::MaxCount(MaxCountConstraint {
            max_count: max_count as u32,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(datatype_node) =
        get_named_node_for_subject(graph, blank_subject, &SHACL_VOCAB.datatype)?
    {
        let constraint = Constraint::Datatype(DatatypeConstraint {
            datatype_iri: datatype_node,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(node_kind_iri) =
        get_named_node_for_subject(graph, blank_subject, &SHACL_VOCAB.node_kind)?
    {
        if let Ok(node_kind) = parse_node_kind(&node_kind_iri) {
            let constraint = Constraint::NodeKind(NodeKindConstraint { node_kind });
            shape.add_constraint(constraint.component_id(), constraint);
        }
    }

    if let Some(pattern) = get_string_for_subject(graph, blank_subject, &SHACL_VOCAB.pattern)? {
        let flags = get_string_for_subject(graph, blank_subject, &SHACL_VOCAB.flags)?;
        let constraint = Constraint::Pattern(PatternConstraint {
            pattern,
            flags,
            message: None,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(class_node) = get_named_node_for_subject(graph, blank_subject, &SHACL_VOCAB.class)?
    {
        let constraint = Constraint::Class(ClassConstraint {
            class_iri: class_node,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    let in_values = parse_rdf_list_as_terms(graph, blank_subject, &SHACL_VOCAB.in_list)?;
    if !in_values.is_empty() {
        let constraint = Constraint::In(InConstraint { values: in_values });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    let has_value_triples = graph.query_triples(
        Some(blank_subject),
        Some(&Predicate::NamedNode(SHACL_VOCAB.has_value.clone())),
        None,
    );
    for triple in has_value_triples {
        let value_term = match triple.object() {
            Object::NamedNode(node) => Term::NamedNode(node.clone()),
            Object::BlankNode(node) => Term::BlankNode(node.clone()),
            Object::Literal(lit) => Term::Literal(lit.clone()),
            Object::Variable(_) | Object::QuotedTriple(_) => continue,
        };
        let constraint = Constraint::HasValue(HasValueConstraint { value: value_term });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(min_length) =
        get_integer_for_subject(graph, blank_subject, &SHACL_VOCAB.min_length)?
    {
        let constraint = Constraint::MinLength(MinLengthConstraint {
            min_length: min_length as u32,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(max_length) =
        get_integer_for_subject(graph, blank_subject, &SHACL_VOCAB.max_length)?
    {
        let constraint = Constraint::MaxLength(MaxLengthConstraint {
            max_length: max_length as u32,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(min_exclusive) =
        get_numeric_for_subject(graph, blank_subject, &SHACL_VOCAB.min_exclusive)?
    {
        let constraint = Constraint::MinExclusive(MinExclusiveConstraint {
            min_value: min_exclusive,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(max_exclusive) =
        get_numeric_for_subject(graph, blank_subject, &SHACL_VOCAB.max_exclusive)?
    {
        let constraint = Constraint::MaxExclusive(MaxExclusiveConstraint {
            max_value: max_exclusive,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(min_inclusive) =
        get_numeric_for_subject(graph, blank_subject, &SHACL_VOCAB.min_inclusive)?
    {
        let constraint = Constraint::MinInclusive(MinInclusiveConstraint {
            min_value: min_inclusive,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    if let Some(max_inclusive) =
        get_numeric_for_subject(graph, blank_subject, &SHACL_VOCAB.max_inclusive)?
    {
        let constraint = Constraint::MaxInclusive(MaxInclusiveConstraint {
            max_value: max_inclusive,
        });
        shape.add_constraint(constraint.component_id(), constraint);
    }

    Ok(())
}

pub(crate) fn parse_metadata_from_blank_node(
    graph: &Graph,
    blank_subject: &Subject,
    shape: &mut Shape,
) -> Result<()> {
    if let Some(label) = get_string_for_subject(graph, blank_subject, &SHACL_VOCAB.label)? {
        shape.label = Some(label);
    }

    if let Some(description) =
        get_string_for_subject(graph, blank_subject, &SHACL_VOCAB.description)?
    {
        shape.description = Some(description);
    }

    if let Some((message, lang_tag)) =
        get_string_with_language_for_subject(graph, blank_subject, &SHACL_VOCAB.message)?
    {
        shape.messages.insert(lang_tag, message);
    }

    Ok(())
}

pub(crate) fn find_shape_iris(graph: &Graph) -> Result<Vec<String>> {
    let mut shape_iris: HashSet<String> = HashSet::new();

    let shacl_ns = "http://www.w3.org/ns/shacl#";
    let rdf_type = super::parser_types::rdf_type_node()?;

    let shape_types = vec![
        NamedNode::new(format!("{shacl_ns}NodeShape"))
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid NodeShape IRI: {e}")))?,
        NamedNode::new(format!("{shacl_ns}PropertyShape"))
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid PropertyShape IRI: {e}")))?,
    ];

    for shape_type in shape_types {
        let triples = graph.query_triples(
            None,
            Some(&Predicate::NamedNode(rdf_type.clone())),
            Some(&Object::NamedNode(shape_type)),
        );

        for triple in triples {
            if let Subject::NamedNode(shape_node) = triple.subject() {
                shape_iris.insert(shape_node.as_str().to_string());
            }
        }
    }

    let shape_properties = vec![
        "targetClass",
        "targetNode",
        "targetObjectsOf",
        "targetSubjectsOf",
        "property",
        "path",
        "node",
        "class",
        "datatype",
        "minCount",
        "maxCount",
    ];

    for prop_name in shape_properties {
        let prop_iri = NamedNode::new(format!("{shacl_ns}{prop_name}"))
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid property IRI: {e}")))?;

        let triples = graph.query_triples(None, Some(&Predicate::NamedNode(prop_iri)), None);

        for triple in triples {
            if let Subject::NamedNode(shape_node) = triple.subject() {
                shape_iris.insert(shape_node.as_str().to_string());
            }
        }
    }

    tracing::info!("Discovered {} shape IRIs in graph", shape_iris.len());
    Ok(shape_iris.into_iter().collect())
}
