//! JSON-LD expansion algorithm — the `JsonLdExpansionConverter` struct and its implementation.

use super::context::{
    has_keyword_form, JsonLdContext, JsonLdContextProcessor, JsonLdLoadDocumentOptions,
    JsonLdRemoteDocument,
};
use super::error::JsonLdErrorCode;
use super::expansion_algorithm_value as value_handlers;
use super::expansion_context::{to_owned_event, JsonLdEvent, JsonLdExpansionState, JsonLdValue};
use super::profile::JsonLdProcessingMode;
use super::{JsonLdSyntaxError, MAX_CONTEXT_RECURSION};
use json_event_parser::JsonEvent;
use oxiri::Iri;
use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::{Arc, Mutex};

/// Applies the [Expansion Algorithm](https://www.w3.org/TR/json-ld-api/#expansion-algorithms)
pub struct JsonLdExpansionConverter {
    pub(super) state: Vec<JsonLdExpansionState>,
    pub(super) context: Vec<(JsonLdContext, usize)>,
    pub(super) is_end: bool,
    pub(super) streaming: bool,
    pub(super) lenient: bool,
    pub(super) base_url: Option<Iri<String>>,
    pub(super) context_processor: JsonLdContextProcessor,
}

#[expect(clippy::expect_used, clippy::unwrap_in_result)]
impl JsonLdExpansionConverter {
    pub fn new(
        base_url: Option<Iri<String>>,
        streaming: bool,
        lenient: bool,
        processing_mode: JsonLdProcessingMode,
    ) -> Self {
        Self {
            state: vec![JsonLdExpansionState::Element {
                active_property: None,
                is_array: false,
                container: &[],
                reverse: false,
            }],
            context: vec![(JsonLdContext::new_empty(base_url.clone()), 0)],
            is_end: false,
            streaming,
            lenient,
            base_url,
            context_processor: JsonLdContextProcessor {
                processing_mode,
                lenient,
                max_context_recursion: MAX_CONTEXT_RECURSION,
                remote_context_cache: Arc::new(Mutex::new(HashMap::new())), /* TODO: share in the parser */
                load_document_callback: None,
            },
        }
    }

    pub fn is_end(&self) -> bool {
        self.is_end
    }

    pub fn with_load_document_callback(
        mut self,
        callback: impl Fn(
                &str,
                &JsonLdLoadDocumentOptions,
            ) -> Result<JsonLdRemoteDocument, Box<dyn Error + Send + Sync>>
            + Send
            + Sync
            + UnwindSafe
            + RefUnwindSafe
            + 'static,
    ) -> Self {
        self.context_processor.load_document_callback = Some(Arc::new(callback));
        self
    }

    pub fn convert_event(
        &mut self,
        event: JsonEvent<'_>,
        results: &mut Vec<JsonLdEvent>,
        errors: &mut Vec<JsonLdSyntaxError>,
    ) {
        if self.state.len() > 4096 {
            errors.push(JsonLdSyntaxError::msg("Too large state stack"));
            return;
        }
        if event == JsonEvent::Eof {
            self.is_end = true;
            return;
        }

        // Large hack to fetch the last state but keep it if we are in an array
        let state = self.state.pop().expect("Empty stack");
        match state {
            JsonLdExpansionState::Element {
                active_property,
                is_array,
                container,
                reverse,
            } => {
                match event {
                    JsonEvent::Null => {
                        // 1)
                        if is_array {
                            self.state.push(JsonLdExpansionState::Element {
                                active_property,
                                is_array,
                                container,
                                reverse,
                            });
                        }
                    }
                    JsonEvent::String(value) => self.on_literal_value(
                        JsonLdValue::String(value.into()),
                        active_property,
                        is_array,
                        container,
                        reverse,
                        results,
                        errors,
                    ),
                    JsonEvent::Number(value) => self.on_literal_value(
                        JsonLdValue::Number(value.into()),
                        active_property,
                        is_array,
                        container,
                        reverse,
                        results,
                        errors,
                    ),
                    JsonEvent::Boolean(value) => self.on_literal_value(
                        JsonLdValue::Boolean(value),
                        active_property,
                        is_array,
                        container,
                        reverse,
                        results,
                        errors,
                    ),
                    JsonEvent::StartArray => {
                        // 5)
                        if is_array {
                            self.state.push(JsonLdExpansionState::Element {
                                active_property: active_property.clone(),
                                is_array,
                                container,
                                reverse,
                            });
                        }
                        if container.contains(&"@list") {
                            if reverse {
                                errors.push(JsonLdSyntaxError::msg_and_code(
                                    "Lists are not allowed inside of reverse properties",
                                    JsonLdErrorCode::InvalidReversePropertyValue,
                                ))
                            }
                            results.push(JsonLdEvent::StartList);
                            self.state.push(JsonLdExpansionState::ListOrSetContainer {
                                needs_end_object: false,
                                end_event: Some(JsonLdEvent::EndList),
                            })
                        }
                        if container.contains(&"@set") && !is_array {
                            results.push(JsonLdEvent::StartSet);
                            self.state.push(JsonLdExpansionState::ListOrSetContainer {
                                needs_end_object: false,
                                end_event: Some(JsonLdEvent::EndSet),
                            })
                        }
                        self.state.push(JsonLdExpansionState::Element {
                            active_property,
                            is_array: true,
                            container,
                            reverse,
                        });
                    }
                    JsonEvent::EndArray => (),
                    JsonEvent::StartObject => {
                        if is_array {
                            self.state.push(JsonLdExpansionState::Element {
                                active_property: active_property.clone(),
                                is_array,
                                container,
                                reverse,
                            });
                        } else if container.contains(&"@index") {
                            self.state
                                .push(JsonLdExpansionState::IndexContainer { active_property });
                            return;
                        } else if container.contains(&"@language") {
                            self.state.push(JsonLdExpansionState::LanguageContainer);
                            return;
                        }
                        self.push_same_context();
                        self.state.push(if self.streaming {
                            JsonLdExpansionState::ObjectOrContainerStartStreaming {
                                active_property,
                                container: if is_array { &[] } else { container },
                                reverse,
                            }
                        } else {
                            JsonLdExpansionState::ObjectOrContainerStart {
                                buffer: Vec::new(),
                                depth: 1,
                                current_key: None,
                                active_property,
                                container: if is_array { &[] } else { container },
                                reverse,
                            }
                        });
                    }
                    JsonEvent::EndObject | JsonEvent::ObjectKey(_) | JsonEvent::Eof => {
                        unreachable!()
                    }
                }
            }
            JsonLdExpansionState::ObjectOrContainerStart {
                mut buffer,
                mut depth,
                mut current_key,
                active_property,
                container,
                reverse,
            } => {
                // We have to buffer everything to make sure we get the @context key even if it's at the end
                match event {
                    JsonEvent::String(_)
                    | JsonEvent::Number(_)
                    | JsonEvent::Boolean(_)
                    | JsonEvent::Null => {
                        buffer
                            .last_mut()
                            .expect("buffer validated to be non-empty")
                            .1
                            .push(to_owned_event(event));
                    }
                    JsonEvent::ObjectKey(key) => {
                        if depth == 1 {
                            buffer.push((key.clone().into(), Vec::new()));
                            current_key = Some(key.into());
                        } else {
                            buffer
                                .last_mut()
                                .expect("buffer validated to be non-empty")
                                .1
                                .push(to_owned_event(JsonEvent::ObjectKey(key)));
                        }
                    }
                    JsonEvent::EndArray | JsonEvent::EndObject => {
                        if depth > 1 {
                            buffer
                                .last_mut()
                                .expect("buffer validated to be non-empty")
                                .1
                                .push(to_owned_event(event));
                        }
                        depth -= 1;
                    }
                    JsonEvent::StartArray | JsonEvent::StartObject => {
                        buffer
                            .last_mut()
                            .expect("buffer validated to be non-empty")
                            .1
                            .push(to_owned_event(event));
                        depth += 1;
                    }
                    JsonEvent::Eof => unreachable!(),
                }
                if depth == 0 {
                    // We look for @context @type, @id and @graph
                    let mut context_value = None;
                    let mut type_data = None;
                    let mut id_data = None;
                    let mut graph_data = Vec::new();
                    let mut other_data = Vec::with_capacity(buffer.len());
                    for (key, value) in buffer {
                        let expanded = self.expand_iri(key.as_str().into(), false, true, errors);
                        match expanded.as_deref() {
                            Some("@context") => {
                                if context_value.is_some() {
                                    errors.push(JsonLdSyntaxError::msg("@context is defined twice"))
                                }
                                context_value = Some(value);
                            }
                            Some("@type") => {
                                if type_data.is_some() {
                                    errors.push(JsonLdSyntaxError::msg("@type is defined twice"))
                                }
                                type_data = Some((key, value));
                            }
                            Some("@id") => {
                                if id_data.is_some() {
                                    errors.push(JsonLdSyntaxError::msg("@id is defined twice"))
                                }
                                id_data = Some((key, value));
                            }
                            Some("@graph") => {
                                graph_data.push((key, value));
                            }
                            _ => other_data.push((key, value)),
                        }
                    }
                    self.state
                        .push(JsonLdExpansionState::ObjectOrContainerStartStreaming {
                            active_property,
                            container,
                            reverse,
                        });

                    // We first process @context, @type and @id then other then graph
                    if let Some(context) = context_value {
                        self.push_new_context(context, errors);
                    }
                    for (key, value) in type_data
                        .into_iter()
                        .chain(id_data)
                        .chain(other_data)
                        .chain(graph_data)
                    {
                        self.convert_event(JsonEvent::ObjectKey(key.into()), results, errors);
                        for event in value {
                            self.convert_event(event, results, errors);
                        }
                    }
                    self.convert_event(JsonEvent::EndObject, results, errors);
                } else {
                    self.state
                        .push(JsonLdExpansionState::ObjectOrContainerStart {
                            buffer,
                            depth,
                            current_key,
                            active_property,
                            container,
                            reverse,
                        });
                }
            }
            JsonLdExpansionState::ObjectOrContainerStartStreaming {
                active_property,
                container,
                reverse,
            } => match event {
                JsonEvent::ObjectKey(key) => {
                    if let Some(iri) = self.expand_iri(key.as_ref().into(), false, true, errors) {
                        match iri.as_ref() {
                            "@context" => self.state.push(JsonLdExpansionState::Context {
                                buffer: Vec::new(),
                                depth: 0,
                                active_property,
                                container,
                                reverse,
                            }),
                            "@index" => {
                                self.state.push(
                                    JsonLdExpansionState::ObjectOrContainerStartStreaming {
                                        active_property,
                                        container,
                                        reverse,
                                    },
                                );
                                self.state.push(JsonLdExpansionState::Index);
                            }
                            "@list" => {
                                if active_property.is_some() {
                                    if reverse {
                                        errors.push(JsonLdSyntaxError::msg_and_code(
                                            "Lists are not allowed inside of reverse properties",
                                            JsonLdErrorCode::InvalidReversePropertyValue,
                                        ))
                                    }
                                    self.state.push(JsonLdExpansionState::ListOrSetContainer {
                                        needs_end_object: true,
                                        end_event: Some(JsonLdEvent::EndList),
                                    });
                                    self.state.push(JsonLdExpansionState::Element {
                                        is_array: false,
                                        active_property,
                                        container: &[],
                                        reverse: false,
                                    });
                                    results.push(JsonLdEvent::StartList);
                                } else {
                                    // We don't have an active property, we skip the list
                                    self.state
                                        .push(JsonLdExpansionState::Skip { is_array: false });
                                    self.state
                                        .push(JsonLdExpansionState::Skip { is_array: false });
                                }
                            }
                            "@set" => {
                                let has_property = active_property.is_some();
                                self.state.push(JsonLdExpansionState::ListOrSetContainer {
                                    needs_end_object: true,
                                    end_event: has_property.then_some(JsonLdEvent::EndSet),
                                });
                                self.state.push(JsonLdExpansionState::Element {
                                    is_array: false,
                                    active_property,
                                    container: &[],
                                    reverse: false,
                                });
                                if has_property {
                                    results.push(JsonLdEvent::StartSet);
                                }
                            }
                            _ => {
                                if container.contains(&"@list") {
                                    results.push(JsonLdEvent::StartList);
                                    self.state.push(JsonLdExpansionState::ListOrSetContainer {
                                        needs_end_object: false,
                                        end_event: Some(JsonLdEvent::EndList),
                                    });
                                } else if container.contains(&"@set") {
                                    results.push(JsonLdEvent::StartSet);
                                    self.state.push(JsonLdExpansionState::ListOrSetContainer {
                                        needs_end_object: false,
                                        end_event: Some(JsonLdEvent::EndSet),
                                    });
                                }
                                self.state.push(JsonLdExpansionState::ObjectStart {
                                    types: Vec::new(),
                                    id: None,
                                    seen_id: false,
                                    active_property,
                                    reverse,
                                });
                                self.convert_event(JsonEvent::ObjectKey(key), results, errors)
                            }
                        }
                    } else {
                        self.state.push(JsonLdExpansionState::ObjectStart {
                            types: Vec::new(),
                            id: None,
                            seen_id: false,
                            active_property,
                            reverse,
                        });
                        self.convert_event(JsonEvent::ObjectKey(key), results, errors)
                    }
                }
                JsonEvent::EndObject => {
                    self.state.push(JsonLdExpansionState::ObjectStart {
                        types: Vec::new(),
                        id: None,
                        seen_id: false,
                        active_property,
                        reverse,
                    });
                    self.convert_event(JsonEvent::EndObject, results, errors)
                }
                _ => unreachable!("Inside of an object"),
            },
            JsonLdExpansionState::Context {
                mut buffer,
                mut depth,
                active_property,
                container,
                reverse,
            } => {
                match event {
                    JsonEvent::String(_)
                    | JsonEvent::Number(_)
                    | JsonEvent::Boolean(_)
                    | JsonEvent::Null
                    | JsonEvent::ObjectKey(_) => buffer.push(to_owned_event(event)),
                    JsonEvent::EndArray | JsonEvent::EndObject => {
                        buffer.push(to_owned_event(event));
                        depth -= 1;
                    }
                    JsonEvent::StartArray | JsonEvent::StartObject => {
                        buffer.push(to_owned_event(event));
                        depth += 1;
                    }
                    JsonEvent::Eof => unreachable!(),
                }
                if depth == 0 {
                    self.push_new_context(buffer, errors);
                    self.state
                        .push(JsonLdExpansionState::ObjectOrContainerStartStreaming {
                            active_property,
                            container,
                            reverse,
                        });
                } else {
                    self.state.push(JsonLdExpansionState::Context {
                        buffer,
                        depth,
                        active_property,
                        container,
                        reverse,
                    });
                }
            }
            JsonLdExpansionState::ObjectStart {
                types,
                id,
                seen_id,
                active_property,
                reverse,
            } => match event {
                JsonEvent::ObjectKey(key) => {
                    if let Some(iri) = self.expand_iri(key.as_ref().into(), false, true, errors) {
                        match iri.as_ref() {
                            "@type" => {
                                if seen_id && !self.lenient {
                                    errors.push(JsonLdSyntaxError::msg_and_code(
                                        "@type must be the first key of an object or right after @context",
                                        JsonLdErrorCode::InvalidStreamingKeyOrder,
                                    ))
                                }
                                self.state.push(JsonLdExpansionState::ObjectType {
                                    id,
                                    types,
                                    is_array: false,
                                    active_property,
                                    reverse,
                                });
                            }
                            "@value" | "@language" => {
                                if types.len() > 1 {
                                    errors.push(JsonLdSyntaxError::msg_and_code(
                                        "Only a single @type is allowed when @value is present",
                                        JsonLdErrorCode::InvalidTypedValue,
                                    ));
                                }
                                if id.is_some() {
                                    errors.push(JsonLdSyntaxError::msg_and_code(
                                        "@value and @id are incompatible",
                                        JsonLdErrorCode::InvalidValueObject,
                                    ));
                                }
                                if reverse {
                                    errors.push(JsonLdSyntaxError::msg_and_code(
                                        "Literals are not allowed inside of reverse properties",
                                        JsonLdErrorCode::InvalidReversePropertyValue,
                                    ))
                                }
                                self.state.push(JsonLdExpansionState::Value {
                                    r#type: types.into_iter().next(),
                                    value: None,
                                    language: None,
                                });
                                self.convert_event(JsonEvent::ObjectKey(key), results, errors);
                            }
                            "@id" => {
                                if id.is_some() {
                                    errors.push(JsonLdSyntaxError::msg_and_code(
                                        "Only a single @id is allowed",
                                        JsonLdErrorCode::CollidingKeywords,
                                    ));
                                }
                                self.state.push(JsonLdExpansionState::ObjectId {
                                    types,
                                    id,
                                    from_start: true,
                                    reverse,
                                });
                            }
                            "@graph"
                                if id.is_none() && types.is_empty() && self.state.is_empty() =>
                            {
                                // Graph only for @context
                                self.state.push(JsonLdExpansionState::RootGraph);
                                self.state.push(JsonLdExpansionState::Element {
                                    active_property: None,
                                    is_array: false,
                                    container: &[],
                                    reverse: false,
                                })
                            }
                            "@index" => {
                                self.state.push(JsonLdExpansionState::ObjectStart {
                                    types,
                                    id,
                                    seen_id,
                                    active_property,
                                    reverse,
                                });
                                self.state.push(JsonLdExpansionState::Index);
                            }
                            _ => {
                                results.push(JsonLdEvent::StartObject { types });
                                let has_emitted_id = id.is_some();
                                if let Some(id) = id {
                                    results.push(JsonLdEvent::Id(id));
                                }
                                self.state.push(JsonLdExpansionState::Object {
                                    in_property: false,
                                    has_emitted_id,
                                });
                                self.convert_event(JsonEvent::ObjectKey(key), results, errors);
                            }
                        }
                    } else {
                        self.state.push(JsonLdExpansionState::ObjectStart {
                            types,
                            id,
                            seen_id,
                            active_property,
                            reverse,
                        });
                        self.state
                            .push(JsonLdExpansionState::Skip { is_array: false });
                    }
                }
                JsonEvent::EndObject => {
                    results.push(JsonLdEvent::StartObject { types });
                    if let Some(id) = id {
                        results.push(JsonLdEvent::Id(id));
                    }
                    results.push(JsonLdEvent::EndObject);
                    self.pop_context();
                }
                _ => unreachable!("Inside of an object"),
            },
            JsonLdExpansionState::ObjectType {
                mut types,
                id,
                is_array,
                active_property,
                reverse,
            } => {
                match event {
                    JsonEvent::Null | JsonEvent::Number(_) | JsonEvent::Boolean(_) => {
                        // 13.4.4.1)
                        errors.push(JsonLdSyntaxError::msg_and_code(
                            "@type value must be a string",
                            JsonLdErrorCode::InvalidTypeValue,
                        ));
                        if is_array {
                            self.state.push(JsonLdExpansionState::ObjectType {
                                types,
                                id,
                                is_array,
                                active_property,
                                reverse,
                            });
                        } else {
                            self.state.push(JsonLdExpansionState::ObjectStart {
                                types,
                                id,
                                seen_id: false,
                                active_property,
                                reverse,
                            });
                        }
                    }
                    JsonEvent::String(value) => {
                        // 13.4.4.4)
                        if let Some(iri) = self.expand_iri(value, true, true, errors) {
                            if has_keyword_form(&iri) {
                                errors.push(JsonLdSyntaxError::msg(format!(
                                    "{iri} is not a valid value for @type"
                                )));
                            } else {
                                types.push(iri.into());
                            }
                        }
                        if is_array {
                            self.state.push(JsonLdExpansionState::ObjectType {
                                types,
                                id,
                                is_array,
                                active_property,
                                reverse,
                            });
                        } else {
                            self.state.push(JsonLdExpansionState::ObjectStart {
                                types,
                                id,
                                seen_id: false,
                                active_property,
                                reverse,
                            });
                        }
                    }
                    JsonEvent::StartArray => {
                        self.state.push(JsonLdExpansionState::ObjectType {
                            types,
                            id,
                            is_array: true,
                            active_property,
                            reverse,
                        });
                        if is_array {
                            errors.push(JsonLdSyntaxError::msg_and_code(
                                "@type cannot contain a nested array",
                                JsonLdErrorCode::InvalidTypeValue,
                            ));
                            self.state
                                .push(JsonLdExpansionState::Skip { is_array: true });
                        }
                    }
                    JsonEvent::EndArray => {
                        self.state.push(JsonLdExpansionState::ObjectStart {
                            types,
                            id,
                            seen_id: false,
                            active_property,
                            reverse,
                        });
                    }
                    JsonEvent::StartObject => {
                        // 13.4.4.1)
                        errors.push(JsonLdSyntaxError::msg_and_code(
                            "@type value must be a string",
                            JsonLdErrorCode::InvalidTypeValue,
                        ));
                        if is_array {
                            self.state.push(JsonLdExpansionState::ObjectType {
                                types,
                                id,
                                is_array: true,
                                active_property,
                                reverse,
                            });
                        } else {
                            self.state.push(JsonLdExpansionState::ObjectStart {
                                types,
                                id,
                                seen_id: false,
                                active_property,
                                reverse,
                            });
                        }
                        self.state
                            .push(JsonLdExpansionState::Skip { is_array: false });
                    }
                    JsonEvent::ObjectKey(_) | JsonEvent::EndObject | JsonEvent::Eof => {
                        unreachable!()
                    }
                }
            }
            JsonLdExpansionState::ObjectId {
                types,
                mut id,
                from_start,
                reverse,
            } => {
                if let JsonEvent::String(new_id) = event {
                    if let Some(new_id) = self.expand_iri(new_id, true, false, errors) {
                        if has_keyword_form(&new_id) {
                            errors.push(JsonLdSyntaxError::msg(
                                "@id value must be an IRI or a blank node",
                            ));
                        } else {
                            id = Some(new_id.into());
                        }
                        self.state.push(if from_start {
                            JsonLdExpansionState::ObjectStart {
                                types,
                                id,
                                seen_id: true,
                                active_property: None,
                                reverse,
                            }
                        } else {
                            if let Some(id) = id {
                                results.push(JsonLdEvent::Id(id));
                            }
                            JsonLdExpansionState::Object {
                                in_property: false,
                                has_emitted_id: true,
                            }
                        })
                    } else {
                        self.state
                            .push(JsonLdExpansionState::Skip { is_array: false });
                    }
                } else {
                    errors.push(JsonLdSyntaxError::msg_and_code(
                        "@id value must be a string",
                        JsonLdErrorCode::InvalidIdValue,
                    ));
                    self.state.push(if from_start {
                        JsonLdExpansionState::ObjectStart {
                            types,
                            id,
                            seen_id: true,
                            active_property: None,
                            reverse,
                        }
                    } else {
                        JsonLdExpansionState::Object {
                            in_property: false,
                            has_emitted_id: true,
                        }
                    });
                    self.state
                        .push(JsonLdExpansionState::Skip { is_array: false });
                    self.convert_event(event, results, errors);
                }
            }
            JsonLdExpansionState::Object {
                in_property,
                has_emitted_id,
            } => value_handlers::handle_object_state(
                self,
                in_property,
                has_emitted_id,
                event,
                results,
                errors,
            ),
            JsonLdExpansionState::ReverseStart => {
                value_handlers::handle_reverse_start_state(self, event, results, errors)
            }
            JsonLdExpansionState::Reverse { in_property } => {
                value_handlers::handle_reverse_state(self, in_property, event, results, errors)
            }
            JsonLdExpansionState::Value {
                r#type,
                value,
                language,
            } => value_handlers::handle_value_state(
                self, r#type, value, language, event, results, errors,
            ),
            JsonLdExpansionState::ValueValue { r#type, language } => {
                value_handlers::handle_value_value_state(
                    self, r#type, language, event, results, errors,
                )
            }
            JsonLdExpansionState::ValueLanguage { value, r#type } => {
                value_handlers::handle_value_language_state(
                    self, r#type, value, event, results, errors,
                )
            }
            JsonLdExpansionState::ValueType { value, language } => {
                value_handlers::handle_value_type_state(
                    self, value, language, event, results, errors,
                )
            }
            JsonLdExpansionState::Index => {
                if let JsonEvent::String(_) = event {
                    // TODO: properly emit if we implement expansion output
                } else {
                    errors.push(JsonLdSyntaxError::msg_and_code(
                        "@index value must be a string",
                        JsonLdErrorCode::InvalidIndexValue,
                    ));
                    self.state
                        .push(JsonLdExpansionState::Skip { is_array: false });
                    self.convert_event(event, results, errors);
                }
            }
            JsonLdExpansionState::Graph => {
                results.push(JsonLdEvent::EndGraph);
                self.convert_event(event, results, errors)
            }
            JsonLdExpansionState::RootGraph => match event {
                JsonEvent::ObjectKey(key) => {
                    errors.push(JsonLdSyntaxError::msg_and_code(
                        format!(
                            "@graph must be the last property of the object, found {key} after it"
                        ),
                        JsonLdErrorCode::InvalidStreamingKeyOrder,
                    ));
                    self.state.push(JsonLdExpansionState::RootGraph);
                    self.state
                        .push(JsonLdExpansionState::Skip { is_array: false });
                }
                JsonEvent::EndObject => (),
                _ => unreachable!(),
            },
            JsonLdExpansionState::ListOrSetContainer {
                needs_end_object,
                end_event,
            } => {
                if needs_end_object {
                    match event {
                        JsonEvent::EndObject => {
                            results.extend(end_event);
                            self.pop_context();
                        }
                        JsonEvent::ObjectKey(key) => {
                            self.state.push(JsonLdExpansionState::ListOrSetContainer {
                                needs_end_object,
                                end_event,
                            });
                            if let Some(iri) =
                                self.expand_iri(key.as_ref().into(), false, true, errors)
                            {
                                if iri == "@index" {
                                    self.state.push(JsonLdExpansionState::Index);
                                } else {
                                    errors.push(JsonLdSyntaxError::msg_and_code(
                                        format!(
                                            "@list must be the only key of an object, {key} found"
                                        ),
                                        JsonLdErrorCode::InvalidSetOrListObject,
                                    ));
                                    self.state
                                        .push(JsonLdExpansionState::Skip { is_array: false });
                                }
                            } else {
                                self.state
                                    .push(JsonLdExpansionState::Skip { is_array: false });
                            }
                        }
                        _ => unreachable!(),
                    }
                } else {
                    results.extend(end_event);
                    self.convert_event(event, results, errors)
                }
            }
            JsonLdExpansionState::IndexContainer { active_property } => match event {
                JsonEvent::EndObject => (),
                JsonEvent::ObjectKey(_) => {
                    // TODO: emit @index
                    self.state.push(JsonLdExpansionState::IndexContainer {
                        active_property: active_property.clone(),
                    });
                    self.state.push(JsonLdExpansionState::Element {
                        active_property,
                        is_array: false,
                        container: &[],
                        reverse: false,
                    })
                }
                _ => unreachable!(),
            },
            JsonLdExpansionState::LanguageContainer => match event {
                JsonEvent::EndObject => (),
                JsonEvent::ObjectKey(language) => {
                    self.state.push(JsonLdExpansionState::LanguageContainer);
                    self.state
                        .push(JsonLdExpansionState::LanguageContainerValue {
                            language: language.into(),
                            is_array: false,
                        })
                }
                _ => unreachable!(),
            },
            JsonLdExpansionState::LanguageContainerValue { language, is_array } => {
                value_handlers::handle_language_container_value_state(
                    self, language, is_array, event, results, errors,
                )
            }
            JsonLdExpansionState::Skip { is_array } => {
                value_handlers::handle_skip_state(self, is_array, event)
            }
        }
    }

    /// [IRI Expansion](https://www.w3.org/TR/json-ld-api/#iri-expansion)
    fn expand_iri<'a>(
        &mut self,
        value: Cow<'a, str>,
        document_relative: bool,
        vocab: bool,
        errors: &mut Vec<JsonLdSyntaxError>,
    ) -> Option<Cow<'a, str>> {
        self.context_processor.expand_iri(
            &mut self
                .context
                .last_mut()
                .expect("The context stack must not be empty")
                .0,
            value,
            document_relative,
            vocab,
            None,
            &mut HashMap::new(),
            errors,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Emit a literal value appearing in element context.
    ///
    /// The heavy lifting (including [value expansion](https://www.w3.org/TR/json-ld-api/#value-expansion))
    /// lives in [`value_handlers::on_literal_value_free`] to keep this module
    /// within the workspace size policy.
    #[allow(clippy::too_many_arguments)]
    fn on_literal_value(
        &mut self,
        value: JsonLdValue,
        active_property: Option<String>,
        is_array: bool,
        container: &'static [&'static str],
        reverse: bool,
        results: &mut Vec<JsonLdEvent>,
        errors: &mut Vec<JsonLdSyntaxError>,
    ) {
        value_handlers::on_literal_value_free(
            self,
            value,
            active_property,
            is_array,
            container,
            reverse,
            results,
            errors,
        )
    }

    pub fn context(&self) -> &JsonLdContext {
        &self
            .context
            .last()
            .expect("The context stack must not be empty")
            .0
    }

    fn push_same_context(&mut self) {
        value_handlers::push_same_context_free(self)
    }

    fn push_new_context(
        &mut self,
        context: Vec<JsonEvent<'static>>,
        errors: &mut Vec<JsonLdSyntaxError>,
    ) {
        value_handlers::push_new_context_free(self, context, errors)
    }

    pub(super) fn pop_context(&mut self) {
        let Some((last_context, mut last_count)) = self.context.pop() else {
            return;
        };
        last_count -= 1;
        if last_count > 0 || self.context.is_empty() {
            // We always keep a context to allow reading the root context at the end of the document
            self.context.push((last_context, last_count));
        }
    }
}
