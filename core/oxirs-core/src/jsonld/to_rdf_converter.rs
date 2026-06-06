//! JSON-LD expanded-event to RDF quad conversion.
//!
//! Provides the [`JsonLdToRdfConverter`] state machine that turns the stream of
//! [`JsonLdEvent`]s produced by expansion into RDF [`Quad`]s, plus the JSON
//! number canonicalization helper used to derive `xsd:integer` / `xsd:double`
//! literal lexical forms.

use super::expansion::{JsonLdEvent, JsonLdValue};
use crate::model::*;
use crate::vocab::{rdf, xsd};
use std::fmt::Write;

pub(super) enum JsonLdToRdfState {
    StartObject {
        types: Vec<NamedOrBlankNode>,
        /// Events before the @id event
        buffer: Vec<JsonLdEvent>,
        /// Nesting level of objects, useful during buffering
        nesting: usize,
    },
    Object(Option<NamedOrBlankNode>),
    Property {
        id: Option<NamedNode>,
        reverse: bool,
    },
    List(Option<NamedOrBlankNode>),
    Graph(Option<GraphName>),
}

pub(super) struct JsonLdToRdfConverter {
    pub(super) state: Vec<JsonLdToRdfState>,
    pub(super) lenient: bool,
}

impl JsonLdToRdfConverter {
    pub(super) fn convert_event(&mut self, event: JsonLdEvent, results: &mut Vec<Quad>) {
        #[expect(clippy::expect_used)]
        let state = self.state.pop().expect("Empty stack");
        match state {
            JsonLdToRdfState::StartObject {
                types,
                mut buffer,
                nesting,
            } => {
                match event {
                    JsonLdEvent::Id(id) => {
                        if nesting > 0 {
                            buffer.push(JsonLdEvent::Id(id));
                            self.state.push(JsonLdToRdfState::StartObject {
                                types,
                                buffer,
                                nesting,
                            });
                        } else {
                            let id = self.convert_named_or_blank_node(id);
                            self.emit_quads_for_new_object(id.as_ref(), types, results);
                            self.state.push(JsonLdToRdfState::Object(id));
                            for event in buffer {
                                self.convert_event(event, results);
                            }
                        }
                    }
                    JsonLdEvent::EndObject => {
                        if nesting > 0 {
                            buffer.push(JsonLdEvent::EndObject);
                            self.state.push(JsonLdToRdfState::StartObject {
                                types,
                                buffer,
                                nesting: nesting - 1,
                            });
                        } else {
                            let id = Some(BlankNode::default().into());
                            self.emit_quads_for_new_object(id.as_ref(), types, results);
                            if !buffer.is_empty() {
                                self.state.push(JsonLdToRdfState::Object(id));
                                for event in buffer {
                                    self.convert_event(event, results);
                                }
                                // We properly end after playing the buffer
                                self.convert_event(JsonLdEvent::EndObject, results);
                            }
                        }
                    }
                    JsonLdEvent::StartObject { .. } => {
                        buffer.push(event);
                        self.state.push(JsonLdToRdfState::StartObject {
                            types,
                            buffer,
                            nesting: nesting + 1,
                        });
                    }
                    _ => {
                        buffer.push(event);
                        self.state.push(JsonLdToRdfState::StartObject {
                            types,
                            buffer,
                            nesting,
                        });
                    }
                }
            }
            JsonLdToRdfState::Object(id) => match event {
                JsonLdEvent::Id(_) => {
                    unreachable!("Should have buffered before @id")
                }
                JsonLdEvent::EndObject => (),
                JsonLdEvent::StartProperty { name, reverse } => {
                    self.state.push(JsonLdToRdfState::Object(id));
                    self.state.push(JsonLdToRdfState::Property {
                        id: if self.has_defined_last_predicate() {
                            self.convert_named_node(name)
                        } else {
                            None // We do not want to emit if one of the parent property is not emitted
                        },
                        reverse,
                    });
                }
                JsonLdEvent::StartGraph => {
                    let graph_name = id.clone().map(Into::into);
                    self.state.push(JsonLdToRdfState::Object(id));
                    self.state.push(JsonLdToRdfState::Graph(graph_name));
                }
                JsonLdEvent::StartObject { .. }
                | JsonLdEvent::Value { .. }
                | JsonLdEvent::EndProperty
                | JsonLdEvent::EndGraph
                | JsonLdEvent::StartList
                | JsonLdEvent::EndList
                | JsonLdEvent::StartSet
                | JsonLdEvent::EndSet => unreachable!(),
            },
            JsonLdToRdfState::Property { .. } => match event {
                JsonLdEvent::StartObject { types } => {
                    self.state.push(state);
                    self.state.push(JsonLdToRdfState::StartObject {
                        types: types
                            .into_iter()
                            .filter_map(|t| self.convert_named_or_blank_node(t))
                            .collect(),
                        buffer: Vec::new(),
                        nesting: 0,
                    });
                }
                JsonLdEvent::Value {
                    value,
                    r#type,
                    language,
                } => {
                    self.state.push(state);
                    self.emit_quad_for_new_literal(
                        self.convert_literal(value, language, r#type),
                        results,
                    )
                }
                JsonLdEvent::EndProperty => (),
                JsonLdEvent::StartList => {
                    self.state.push(state);
                    self.state.push(JsonLdToRdfState::List(None));
                }
                JsonLdEvent::StartSet | JsonLdEvent::EndSet => {
                    self.state.push(state);
                }
                JsonLdEvent::StartProperty { .. }
                | JsonLdEvent::Id(_)
                | JsonLdEvent::EndObject
                | JsonLdEvent::StartGraph
                | JsonLdEvent::EndGraph
                | JsonLdEvent::EndList => unreachable!(),
            },
            JsonLdToRdfState::List(current_node) => match event {
                JsonLdEvent::StartObject { types } => {
                    self.add_new_list_node_state(current_node, results);
                    self.state.push(JsonLdToRdfState::StartObject {
                        types: types
                            .into_iter()
                            .filter_map(|t| self.convert_named_or_blank_node(t))
                            .collect(),
                        buffer: Vec::new(),
                        nesting: 0,
                    })
                }
                JsonLdEvent::Value {
                    value,
                    r#type,
                    language,
                } => {
                    self.add_new_list_node_state(current_node, results);
                    self.emit_quad_for_new_literal(
                        self.convert_literal(value, language, r#type),
                        results,
                    )
                }
                JsonLdEvent::StartList => {
                    self.add_new_list_node_state(current_node, results);
                    self.state.push(JsonLdToRdfState::List(None));
                }
                JsonLdEvent::EndList => {
                    if let Some(previous_node) = current_node {
                        if let Some(graph_name) = self.last_graph_name() {
                            results.push(Quad::new(
                                previous_node,
                                rdf::REST.clone(),
                                rdf::NIL.clone(),
                                graph_name.clone(),
                            ));
                        }
                    } else {
                        self.emit_quads_for_new_object(
                            Some(&rdf::NIL.clone().into()),
                            Vec::new(),
                            results,
                        )
                    }
                }
                JsonLdEvent::StartSet | JsonLdEvent::EndSet => {
                    // TODO: this is bad
                    self.state.push(JsonLdToRdfState::List(current_node));
                }
                JsonLdEvent::EndObject
                | JsonLdEvent::StartProperty { .. }
                | JsonLdEvent::EndProperty
                | JsonLdEvent::Id(_)
                | JsonLdEvent::StartGraph
                | JsonLdEvent::EndGraph => unreachable!(),
            },
            JsonLdToRdfState::Graph(_) => match event {
                JsonLdEvent::StartObject { types } => {
                    self.state.push(state);
                    self.state.push(JsonLdToRdfState::StartObject {
                        types: types
                            .into_iter()
                            .filter_map(|t| self.convert_named_or_blank_node(t))
                            .collect(),
                        buffer: Vec::new(),
                        nesting: 0,
                    });
                }
                JsonLdEvent::Value { .. } => {
                    self.state.push(state);
                }
                JsonLdEvent::EndGraph => (),
                JsonLdEvent::StartGraph
                | JsonLdEvent::StartProperty { .. }
                | JsonLdEvent::EndProperty
                | JsonLdEvent::Id(_)
                | JsonLdEvent::EndObject
                | JsonLdEvent::StartList
                | JsonLdEvent::EndList
                | JsonLdEvent::StartSet
                | JsonLdEvent::EndSet => unreachable!(),
            },
        }
    }

    fn emit_quads_for_new_object(
        &self,
        id: Option<&NamedOrBlankNode>,
        types: Vec<NamedOrBlankNode>,
        results: &mut Vec<Quad>,
    ) {
        let Some(id) = id else {
            return;
        };
        let Some(graph_name) = self.last_graph_name() else {
            return;
        };
        if let (Some(subject), Some((predicate, reverse))) =
            (self.last_subject(), self.last_predicate())
        {
            results.push(if reverse {
                Quad::new(
                    id.clone(),
                    predicate.to_owned(),
                    subject.clone(),
                    graph_name.clone(),
                )
            } else {
                Quad::new(
                    subject.clone(),
                    predicate.to_owned(),
                    id.clone(),
                    graph_name.clone(),
                )
            })
        }
        for t in types {
            results.push(Quad::new(
                id.clone(),
                rdf::TYPE.clone(),
                t,
                graph_name.clone(),
            ))
        }
    }

    fn emit_quad_for_new_literal(&self, literal: Option<Literal>, results: &mut Vec<Quad>) {
        let Some(literal) = literal else {
            return;
        };
        let Some(graph_name) = self.last_graph_name() else {
            return;
        };
        let Some(subject) = self.last_subject() else {
            return;
        };
        let Some((predicate, reverse)) = self.last_predicate() else {
            return;
        };
        if reverse {
            return;
        }
        results.push(Quad::new(
            subject.clone(),
            predicate.to_owned(),
            literal,
            graph_name.clone(),
        ))
    }

    fn add_new_list_node_state(
        &mut self,
        current_node: Option<NamedOrBlankNode>,
        results: &mut Vec<Quad>,
    ) {
        let new_node = BlankNode::default();
        if let Some(previous_node) = current_node {
            if let Some(graph_name) = self.last_graph_name() {
                results.push(Quad::new(
                    previous_node,
                    rdf::REST.clone(),
                    new_node.clone(),
                    graph_name.clone(),
                ));
            }
        } else {
            self.emit_quads_for_new_object(Some(&new_node.clone().into()), Vec::new(), results)
        }
        self.state
            .push(JsonLdToRdfState::List(Some(new_node.into())));
    }

    fn convert_named_or_blank_node(&self, value: String) -> Option<NamedOrBlankNode> {
        Some(if let Some(bnode_id) = value.strip_prefix("_:") {
            if self.lenient {
                Some(BlankNode::new_unchecked(bnode_id))
            } else {
                BlankNode::new(bnode_id).ok()
            }?
            .into()
        } else {
            self.convert_named_node(value)?.into()
        })
    }

    fn convert_named_node(&self, value: String) -> Option<NamedNode> {
        if self.lenient {
            Some(NamedNode::new_unchecked(value))
        } else {
            NamedNode::new(&value).ok()
        }
    }

    fn convert_literal(
        &self,
        value: JsonLdValue,
        language: Option<String>,
        r#type: Option<String>,
    ) -> Option<Literal> {
        let r#type = if let Some(t) = r#type {
            Some(self.convert_named_node(t)?)
        } else {
            None
        };
        Some(match value {
            JsonLdValue::String(value) => {
                if let Some(language) = language {
                    if r#type.is_some_and(|t| t.as_str() != rdf::LANG_STRING.as_str()) {
                        return None; // Expansion already returns an error
                    }
                    if self.lenient {
                        Literal::new_language_tagged_literal_unchecked(value, language)
                    } else {
                        Literal::new_language_tagged_literal(value, &language).ok()?
                    }
                } else if let Some(datatype) = r#type {
                    Literal::new_typed_literal(value, datatype)
                } else {
                    Literal::new_simple_literal(value)
                }
            }
            JsonLdValue::Number(value) => {
                if language.is_some() {
                    return None; // Expansion already returns an error
                }
                let value = canonicalize_json_number(
                    &value,
                    r#type
                        .as_ref()
                        .is_some_and(|t| t.as_str() == xsd::DOUBLE.as_str()),
                )
                .unwrap_or(RdfJsonNumber::Double(value));
                match value {
                    RdfJsonNumber::Integer(value) => Literal::new_typed_literal(
                        value,
                        r#type.unwrap_or_else(|| xsd::INTEGER.clone()),
                    ),
                    RdfJsonNumber::Double(value) => Literal::new_typed_literal(
                        value,
                        r#type.unwrap_or_else(|| xsd::DOUBLE.clone()),
                    ),
                }
            }
            JsonLdValue::Boolean(value) => {
                if language.is_some() {
                    return None; // Expansion already returns an error
                }
                Literal::new_typed_literal(
                    if value { "true" } else { "false" },
                    r#type.unwrap_or_else(|| xsd::BOOLEAN.clone()),
                )
            }
        })
    }

    fn last_subject(&self) -> Option<&NamedOrBlankNode> {
        for state in self.state.iter().rev() {
            match state {
                JsonLdToRdfState::Object(id) => {
                    return id.as_ref();
                }
                JsonLdToRdfState::StartObject { .. } => {
                    unreachable!()
                }
                JsonLdToRdfState::Property { .. } => (),
                JsonLdToRdfState::List(id) => return id.as_ref(),
                JsonLdToRdfState::Graph(_) => {
                    return None;
                }
            }
        }
        None
    }

    fn last_predicate(&self) -> Option<(NamedNodeRef<'_>, bool)> {
        for state in self.state.iter().rev() {
            match state {
                JsonLdToRdfState::Property { id, reverse } => {
                    return Some((id.as_ref()?.as_ref(), *reverse));
                }
                JsonLdToRdfState::StartObject { .. } | JsonLdToRdfState::Object(_) => (),
                JsonLdToRdfState::List(_) => return Some((rdf::FIRST.as_ref(), false)),
                JsonLdToRdfState::Graph(_) => {
                    return None;
                }
            }
        }
        None
    }

    fn has_defined_last_predicate(&self) -> bool {
        for state in self.state.iter().rev() {
            if let JsonLdToRdfState::Property { id, .. } = state {
                return id.is_some();
            }
        }
        true
    }

    fn last_graph_name(&self) -> Option<&GraphName> {
        for state in self.state.iter().rev() {
            match state {
                JsonLdToRdfState::Graph(graph) => {
                    return graph.as_ref();
                }
                JsonLdToRdfState::StartObject { .. }
                | JsonLdToRdfState::Object(_)
                | JsonLdToRdfState::Property { .. }
                | JsonLdToRdfState::List(_) => (),
            }
        }
        None
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub(super) enum RdfJsonNumber {
    Integer(String),
    Double(String),
}

/// Canonicalizes the JSON number to xsd:double canonical form.
pub(super) fn canonicalize_json_number(value: &str, always_double: bool) -> Option<RdfJsonNumber> {
    // We parse
    let (value, is_negative) = if let Some(value) = value.strip_prefix('-') {
        (value, true)
    } else if let Some(value) = value.strip_prefix('+') {
        (value, false)
    } else {
        (value, false)
    };
    let (value, exp) = value.split_once(['e', 'E']).unwrap_or((value, "0"));
    let (mut integer_part, mut decimal_part) = value.split_once('.').unwrap_or((value, ""));
    let mut exp = exp.parse::<i64>().ok()?;

    // We normalize
    // We trim the zeros
    while let Some(c) = integer_part.strip_prefix('0') {
        integer_part = c;
    }
    while let Some(c) = decimal_part.strip_suffix('0') {
        decimal_part = c;
    }
    if decimal_part.is_empty() {
        while let Some(c) = integer_part.strip_suffix('0') {
            integer_part = c;
            exp = exp.checked_add(1)?;
        }
    }
    if integer_part.is_empty() {
        while let Some(c) = decimal_part.strip_prefix('0') {
            decimal_part = c;
            exp = exp.checked_sub(1)?;
        }
    }

    // We set the exponent in the 0.XXXEYYY form
    let exp_change = i64::try_from(integer_part.len()).ok()?;
    exp = exp.checked_add(exp_change)?;

    // We handle the zero case
    if integer_part.is_empty() && decimal_part.is_empty() {
        integer_part = "0";
        exp = 1;
    }

    // We serialize
    let mut buffer = String::with_capacity(value.len());
    if is_negative {
        buffer.push('-');
    }
    let digits_count = i64::try_from(integer_part.len() + decimal_part.len()).ok()?;
    Some(if !always_double && exp >= digits_count && exp < 21 {
        buffer.push_str(integer_part);
        buffer.push_str(decimal_part);
        buffer.extend((0..(exp - digits_count)).map(|_| '0'));
        RdfJsonNumber::Integer(buffer)
    } else {
        let mut all_digits = integer_part.chars().chain(decimal_part.chars());
        buffer.push(all_digits.next()?);
        buffer.push('.');
        if digits_count == 1 {
            buffer.push('0');
        } else {
            buffer.extend(all_digits);
        }
        write!(&mut buffer, "E{}", exp.checked_sub(1)?).ok()?;
        RdfJsonNumber::Double(buffer)
    })
}
