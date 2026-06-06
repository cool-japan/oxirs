//! Value-state handlers for the JSON-LD expansion algorithm.
//!
//! These are free functions that accept a mutable reference to
//! `JsonLdExpansionConverter` so that the heavy Value/ValueValue/ValueLanguage/
//! ValueType and LanguageContainerValue/Skip state handlers can live outside
//! `expansion_algorithm.rs` while still having full access to the converter's
//! pub(super) fields.

use super::context::{has_keyword_form, json_node_from_events, JsonLdContext};
use super::error::JsonLdErrorCode;
use super::expansion_algorithm::JsonLdExpansionConverter;
use super::expansion_context::{JsonLdEvent, JsonLdExpansionState, JsonLdValue};
use super::JsonLdSyntaxError;
use json_event_parser::JsonEvent;
use oxiri::Iri;
use std::borrow::Cow;
use std::collections::HashMap;

// -----------------------------------------------------------------------
// IRI expansion free function (mirrors the private method)
// -----------------------------------------------------------------------

/// Expand an IRI using the converter's current context stack.
pub(super) fn expand_iri_free<'a>(
    conv: &mut JsonLdExpansionConverter,
    value: Cow<'a, str>,
    document_relative: bool,
    vocab: bool,
    errors: &mut Vec<JsonLdSyntaxError>,
) -> Option<Cow<'a, str>> {
    conv.context_processor.expand_iri(
        &mut conv
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

/// Return a reference to the current active context.
pub(super) fn active_context(conv: &JsonLdExpansionConverter) -> &JsonLdContext {
    &conv
        .context
        .last()
        .expect("The context stack must not be empty")
        .0
}

// -----------------------------------------------------------------------
// Value state handler
// -----------------------------------------------------------------------

/// Handles `JsonLdExpansionState::Value`.
pub(super) fn handle_value_state(
    conv: &mut JsonLdExpansionConverter,
    r#type: Option<String>,
    value: Option<JsonLdValue>,
    language: Option<String>,
    event: JsonEvent<'_>,
    results: &mut Vec<JsonLdEvent>,
    errors: &mut Vec<JsonLdSyntaxError>,
) {
    match event {
        JsonEvent::ObjectKey(key) => {
            if let Some(iri) = expand_iri_free(conv, key, false, true, errors) {
                match iri.as_ref() {
                    "@value" => {
                        if value.is_some() {
                            errors.push(JsonLdSyntaxError::msg_and_code(
                                "@value cannot be set multiple times",
                                JsonLdErrorCode::InvalidValueObject,
                            ));
                            conv.state.push(JsonLdExpansionState::Value {
                                r#type,
                                value,
                                language,
                            });
                            conv.state
                                .push(JsonLdExpansionState::Skip { is_array: false });
                        } else {
                            conv.state
                                .push(JsonLdExpansionState::ValueValue { r#type, language });
                        }
                    }
                    "@language" => {
                        if language.is_some() {
                            errors.push(JsonLdSyntaxError::msg_and_code(
                                "@language cannot be set multiple times",
                                JsonLdErrorCode::CollidingKeywords,
                            ));
                            conv.state.push(JsonLdExpansionState::Value {
                                r#type,
                                value,
                                language,
                            });
                            conv.state
                                .push(JsonLdExpansionState::Skip { is_array: false });
                        } else {
                            conv.state
                                .push(JsonLdExpansionState::ValueLanguage { r#type, value });
                        }
                    }
                    "@type" => {
                        if !conv.lenient {
                            errors.push(JsonLdSyntaxError::msg_and_code(
                                "@type must be the first key of an object or right after @context",
                                JsonLdErrorCode::InvalidStreamingKeyOrder,
                            ))
                        }
                        if r#type.is_some() {
                            errors.push(JsonLdSyntaxError::msg_and_code(
                                "@type cannot be set multiple times",
                                JsonLdErrorCode::CollidingKeywords,
                            ));
                            conv.state.push(JsonLdExpansionState::Value {
                                r#type,
                                value,
                                language,
                            });
                            conv.state
                                .push(JsonLdExpansionState::Skip { is_array: false });
                        } else {
                            conv.state
                                .push(JsonLdExpansionState::ValueType { value, language });
                        }
                    }
                    "@context" => {
                        errors.push(JsonLdSyntaxError::msg_and_code(
                            "@context must be the first key of an object",
                            JsonLdErrorCode::InvalidStreamingKeyOrder,
                        ));
                        conv.state.push(JsonLdExpansionState::Value {
                            r#type,
                            value,
                            language,
                        });
                        conv.state
                            .push(JsonLdExpansionState::Skip { is_array: false });
                    }
                    "@index" => {
                        conv.state.push(JsonLdExpansionState::Value {
                            r#type,
                            value,
                            language,
                        });
                        conv.state.push(JsonLdExpansionState::Index);
                    }
                    _ if has_keyword_form(&iri) => {
                        errors.push(JsonLdSyntaxError::msg_and_code(
                            format!("Unsupported JSON-Ld keyword inside of a @value: {iri}"),
                            JsonLdErrorCode::InvalidValueObject,
                        ));
                        conv.state.push(JsonLdExpansionState::Value {
                            r#type,
                            value,
                            language,
                        });
                        conv.state
                            .push(JsonLdExpansionState::Skip { is_array: false });
                    }
                    _ => {
                        errors.push(JsonLdSyntaxError::msg_and_code(
                            format!("Objects with @value cannot contain properties, {iri} found"),
                            JsonLdErrorCode::InvalidValueObject,
                        ));
                        conv.state.push(JsonLdExpansionState::Value {
                            r#type,
                            value,
                            language,
                        });
                        conv.state
                            .push(JsonLdExpansionState::Skip { is_array: false });
                    }
                }
            } else {
                conv.state.push(JsonLdExpansionState::Value {
                    r#type,
                    value,
                    language,
                });
                conv.state
                    .push(JsonLdExpansionState::Skip { is_array: false });
            }
        }
        JsonEvent::EndObject => {
            if let Some(value) = value {
                let mut is_valid = true;
                if language.is_some() && r#type.is_some() {
                    errors.push(JsonLdSyntaxError::msg_and_code(
                        "@type and @language cannot be used together",
                        JsonLdErrorCode::InvalidValueObject,
                    ));
                    is_valid = false;
                }
                if language.is_some() && !matches!(value, JsonLdValue::String(_)) {
                    errors.push(JsonLdSyntaxError::msg_and_code(
                        "@language can be used only on a string @value",
                        JsonLdErrorCode::InvalidLanguageTaggedValue,
                    ));
                    is_valid = false;
                }
                if let Some(r#type) = &r#type {
                    if r#type.starts_with("_:") {
                        errors.push(JsonLdSyntaxError::msg_and_code(
                            "@type cannot be a blank node",
                            JsonLdErrorCode::InvalidTypedValue,
                        ));
                        is_valid = false;
                    } else if !conv.lenient {
                        if let Err(e) = Iri::parse(r#type.as_str()) {
                            errors.push(JsonLdSyntaxError::msg_and_code(
                                format!("@type value '{type}' must be an IRI: {e}"),
                                JsonLdErrorCode::InvalidTypedValue,
                            ));
                            is_valid = false;
                        }
                    }
                }
                if is_valid {
                    results.push(JsonLdEvent::Value {
                        value,
                        r#type,
                        language,
                    })
                }
            }
            conv.pop_context();
        }
        JsonEvent::Null
        | JsonEvent::String(_)
        | JsonEvent::Number(_)
        | JsonEvent::Boolean(_)
        | JsonEvent::StartArray
        | JsonEvent::EndArray
        | JsonEvent::StartObject
        | JsonEvent::Eof => unreachable!(),
    }
}

// -----------------------------------------------------------------------
// ValueValue state handler
// -----------------------------------------------------------------------

/// Handles `JsonLdExpansionState::ValueValue`.
pub(super) fn handle_value_value_state(
    conv: &mut JsonLdExpansionConverter,
    r#type: Option<String>,
    language: Option<String>,
    event: JsonEvent<'_>,
    results: &mut Vec<JsonLdEvent>,
    errors: &mut Vec<JsonLdSyntaxError>,
) {
    match event {
        JsonEvent::Null => conv.state.push(JsonLdExpansionState::Value {
            r#type,
            value: None,
            language,
        }),
        JsonEvent::Number(value) => conv.state.push(JsonLdExpansionState::Value {
            r#type,
            value: Some(JsonLdValue::Number(value.into())),
            language,
        }),
        JsonEvent::Boolean(value) => conv.state.push(JsonLdExpansionState::Value {
            r#type,
            value: Some(JsonLdValue::Boolean(value)),
            language,
        }),
        JsonEvent::String(value) => conv.state.push(JsonLdExpansionState::Value {
            r#type,
            value: Some(JsonLdValue::String(value.into())),
            language,
        }),
        _ => {
            errors.push(JsonLdSyntaxError::msg_and_code(
                "@value value must be a string, number, boolean or null",
                JsonLdErrorCode::InvalidValueObjectValue,
            ));
            conv.state.push(JsonLdExpansionState::Value {
                r#type,
                value: None,
                language,
            });
            conv.state
                .push(JsonLdExpansionState::Skip { is_array: false });
            conv.convert_event(event, results, errors);
        }
    }
}

// -----------------------------------------------------------------------
// ValueLanguage state handler
// -----------------------------------------------------------------------

/// Handles `JsonLdExpansionState::ValueLanguage`.
pub(super) fn handle_value_language_state(
    conv: &mut JsonLdExpansionConverter,
    r#type: Option<String>,
    value: Option<JsonLdValue>,
    event: JsonEvent<'_>,
    results: &mut Vec<JsonLdEvent>,
    errors: &mut Vec<JsonLdSyntaxError>,
) {
    if let JsonEvent::String(language) = event {
        conv.state.push(JsonLdExpansionState::Value {
            r#type,
            value,
            language: Some(language.into()),
        })
    } else {
        errors.push(JsonLdSyntaxError::msg_and_code(
            "@value value must be a string",
            JsonLdErrorCode::InvalidLanguageTaggedString,
        ));
        conv.state.push(JsonLdExpansionState::Value {
            r#type,
            value,
            language: None,
        });
        conv.state
            .push(JsonLdExpansionState::Skip { is_array: false });
        conv.convert_event(event, results, errors);
    }
}

// -----------------------------------------------------------------------
// ValueType state handler
// -----------------------------------------------------------------------

/// Handles `JsonLdExpansionState::ValueType`.
pub(super) fn handle_value_type_state(
    conv: &mut JsonLdExpansionConverter,
    value: Option<JsonLdValue>,
    language: Option<String>,
    event: JsonEvent<'_>,
    results: &mut Vec<JsonLdEvent>,
    errors: &mut Vec<JsonLdSyntaxError>,
) {
    if let JsonEvent::String(t) = event {
        let mut r#type = expand_iri_free(conv, t, true, true, errors);
        if let Some(iri) = &r#type {
            if has_keyword_form(iri) {
                errors.push(JsonLdSyntaxError::msg_and_code(
                    format!("{iri} is not a valid value for @type"),
                    JsonLdErrorCode::InvalidTypedValue,
                ));
                r#type = None
            }
        }
        conv.state.push(JsonLdExpansionState::Value {
            r#type: r#type.map(Into::into),
            value,
            language,
        })
    } else {
        errors.push(JsonLdSyntaxError::msg_and_code(
            "@type value must be a string when @value is present",
            JsonLdErrorCode::InvalidTypedValue,
        ));
        conv.state.push(JsonLdExpansionState::Value {
            r#type: None,
            value,
            language,
        });
        conv.state
            .push(JsonLdExpansionState::Skip { is_array: false });
        conv.convert_event(event, results, errors);
    }
}

// -----------------------------------------------------------------------
// LanguageContainerValue state handler
// -----------------------------------------------------------------------

/// Handles `JsonLdExpansionState::LanguageContainerValue`.
pub(super) fn handle_language_container_value_state(
    conv: &mut JsonLdExpansionConverter,
    language: String,
    is_array: bool,
    event: JsonEvent<'_>,
    results: &mut Vec<JsonLdEvent>,
    errors: &mut Vec<JsonLdSyntaxError>,
) {
    match event {
        JsonEvent::Null => {
            if is_array {
                conv.state
                    .push(JsonLdExpansionState::LanguageContainerValue { language, is_array });
            }
        }
        JsonEvent::String(value) => {
            if is_array {
                conv.state
                    .push(JsonLdExpansionState::LanguageContainerValue {
                        language: language.clone(),
                        is_array,
                    });
            }
            results.push(JsonLdEvent::Value {
                value: JsonLdValue::String(value.into()),
                r#type: None,
                language: (language != "@none"
                    && expand_iri_free(conv, language.as_str().into(), false, false, errors)
                        != Some("@none".into()))
                .then_some(language),
            })
        }
        JsonEvent::StartArray => {
            conv.state
                .push(JsonLdExpansionState::LanguageContainerValue {
                    language,
                    is_array: true,
                });
            if is_array {
                errors.push(JsonLdSyntaxError::msg_and_code(
                    "The values in a @language map must be null or strings",
                    JsonLdErrorCode::InvalidLanguageMapValue,
                ));
                conv.state
                    .push(JsonLdExpansionState::Skip { is_array: true })
            }
        }
        JsonEvent::EndArray => (),
        _ => {
            if is_array {
                conv.state
                    .push(JsonLdExpansionState::LanguageContainerValue { language, is_array });
            }
            errors.push(JsonLdSyntaxError::msg_and_code(
                "The values in a @language map must be null or strings",
                JsonLdErrorCode::InvalidLanguageMapValue,
            ));
            conv.state
                .push(JsonLdExpansionState::Skip { is_array: false });
            conv.convert_event(event, results, errors);
        }
    }
}

// -----------------------------------------------------------------------
// Skip state handler
// -----------------------------------------------------------------------

/// Handles `JsonLdExpansionState::Skip`.
pub(super) fn handle_skip_state(
    conv: &mut JsonLdExpansionConverter,
    is_array: bool,
    event: JsonEvent<'_>,
) {
    match event {
        JsonEvent::String(_) | JsonEvent::Number(_) | JsonEvent::Boolean(_) | JsonEvent::Null => {
            if is_array {
                conv.state.push(JsonLdExpansionState::Skip { is_array });
            }
        }
        JsonEvent::EndArray | JsonEvent::EndObject => (),
        JsonEvent::StartArray => {
            if is_array {
                conv.state.push(JsonLdExpansionState::Skip { is_array });
            }
            conv.state
                .push(JsonLdExpansionState::Skip { is_array: true });
        }
        JsonEvent::StartObject => {
            if is_array {
                conv.state.push(JsonLdExpansionState::Skip { is_array });
            }
            conv.state
                .push(JsonLdExpansionState::Skip { is_array: false });
        }
        JsonEvent::ObjectKey(_) => {
            conv.state
                .push(JsonLdExpansionState::Skip { is_array: false });
            conv.state
                .push(JsonLdExpansionState::Skip { is_array: false });
        }
        JsonEvent::Eof => unreachable!(),
    }
}

// -----------------------------------------------------------------------
// Value expansion helper
// -----------------------------------------------------------------------

/// [Value Expansion](https://www.w3.org/TR/json-ld-api/#value-expansion)
pub(super) fn expand_value_free(
    conv: &mut JsonLdExpansionConverter,
    active_property: &str,
    value: JsonLdValue,
    reverse: bool,
    results: &mut Vec<JsonLdEvent>,
    errors: &mut Vec<JsonLdSyntaxError>,
) {
    let active_ctx = active_context(conv);
    let mut r#type = None;
    let mut language = None;
    if let Some(term_definition) = active_ctx.term_definitions.get(active_property) {
        if let Some(type_mapping) = &term_definition.type_mapping {
            match type_mapping.as_ref() {
                // 1)
                "@id" => {
                    if let JsonLdValue::String(value) = value {
                        if let Some(id) = expand_iri_free(conv, value.into(), true, false, errors) {
                            results.push(JsonLdEvent::StartObject { types: Vec::new() });
                            results.push(JsonLdEvent::Id(id.into()));
                            results.push(JsonLdEvent::EndObject);
                        }
                        return;
                    }
                }
                // 2)
                "@vocab" => {
                    if let JsonLdValue::String(value) = value {
                        if let Some(id) = expand_iri_free(conv, value.into(), true, true, errors) {
                            results.push(JsonLdEvent::StartObject { types: Vec::new() });
                            results.push(JsonLdEvent::Id(id.into()));
                            results.push(JsonLdEvent::EndObject);
                        }
                        return;
                    }
                }
                // 4)
                "@none" => (),
                _ => {
                    r#type = Some(type_mapping.clone());
                }
            }
        }
        // 5)
        if matches!(value, JsonLdValue::String(_)) {
            let active_ctx2 = active_context(conv);
            language = term_definition
                .language_mapping
                .clone()
                .unwrap_or_else(|| active_ctx2.default_language.clone());
        }
    } else {
        // 5)
        if matches!(value, JsonLdValue::String(_)) && language.is_none() {
            let active_ctx2 = active_context(conv);
            language.clone_from(&active_ctx2.default_language);
        }
    }
    if reverse {
        errors.push(JsonLdSyntaxError::msg_and_code(
            "Literals are not allowed inside of reverse properties",
            JsonLdErrorCode::InvalidReversePropertyValue,
        ))
    }
    results.push(JsonLdEvent::Value {
        value,
        r#type,
        language,
    });
}

// -----------------------------------------------------------------------
// Literal-value expansion free function
// -----------------------------------------------------------------------

/// Expand a literal value appearing in element context.
///
/// Mirrors `JsonLdExpansionConverter::on_literal_value`.
#[allow(clippy::too_many_arguments)]
pub(super) fn on_literal_value_free(
    conv: &mut JsonLdExpansionConverter,
    value: JsonLdValue,
    active_property: Option<String>,
    is_array: bool,
    container: &'static [&'static str],
    reverse: bool,
    results: &mut Vec<JsonLdEvent>,
    errors: &mut Vec<JsonLdSyntaxError>,
) {
    if !is_array {
        if container.contains(&"@list") {
            if reverse {
                errors.push(JsonLdSyntaxError::msg_and_code(
                    "Lists are not allowed inside of reverse properties",
                    JsonLdErrorCode::InvalidReversePropertyValue,
                ))
            }
            results.push(JsonLdEvent::StartList);
        } else if container.contains(&"@set") {
            results.push(JsonLdEvent::StartSet);
        }
    }
    if let Some(active_property) = &active_property {
        expand_value_free(conv, active_property, value, reverse, results, errors);
    }
    if is_array {
        conv.state.push(JsonLdExpansionState::Element {
            active_property,
            is_array,
            container,
            reverse,
        });
    } else if container.contains(&"@list") {
        results.push(JsonLdEvent::EndList);
    } else if container.contains(&"@set") {
        results.push(JsonLdEvent::EndSet);
    }
}

// -----------------------------------------------------------------------
// Context management free functions
// -----------------------------------------------------------------------

/// Increment the ref-count on the current context (used when entering a nested object).
pub(super) fn push_same_context_free(conv: &mut JsonLdExpansionConverter) {
    conv.context
        .last_mut()
        .expect("The context stack must not be empty")
        .1 += 1;
}

/// Process and push a new context from buffered events.
pub(super) fn push_new_context_free(
    conv: &mut JsonLdExpansionConverter,
    context: Vec<JsonEvent<'static>>,
    errors: &mut Vec<JsonLdSyntaxError>,
) {
    let new_ctx = conv.context_processor.process_context(
        &conv
            .context
            .last()
            .expect("The context stack must not be empty")
            .0,
        json_node_from_events(context.into_iter().map(Ok))
            .expect("context events should produce valid JSON node"),
        conv.base_url.as_ref(),
        &mut Vec::new(),
        false,
        true,
        true,
        errors,
    );
    if let Some((last_context, last_count)) = conv.context.pop() {
        if last_count > 1 {
            conv.context.push((last_context, last_count - 1));
        }
    }
    conv.context.push((new_ctx, 1));
}

// -----------------------------------------------------------------------
// Object state handler
// -----------------------------------------------------------------------

/// Handles `JsonLdExpansionState::Object`.
pub(super) fn handle_object_state(
    conv: &mut JsonLdExpansionConverter,
    in_property: bool,
    has_emitted_id: bool,
    event: JsonEvent<'_>,
    results: &mut Vec<JsonLdEvent>,
    errors: &mut Vec<JsonLdSyntaxError>,
) {
    if in_property {
        results.push(JsonLdEvent::EndProperty);
    }
    match event {
        JsonEvent::EndObject => {
            results.push(JsonLdEvent::EndObject);
            conv.pop_context();
        }
        JsonEvent::ObjectKey(key) => {
            if let Some(iri) = expand_iri_free(conv, key.as_ref().into(), false, true, errors) {
                match iri.as_ref() {
                    "@id" => {
                        if has_emitted_id {
                            errors.push(JsonLdSyntaxError::msg("Duplicated @id key"));
                            conv.state.push(JsonLdExpansionState::Object {
                                in_property: false,
                                has_emitted_id: true,
                            });
                            conv.state
                                .push(JsonLdExpansionState::Skip { is_array: false });
                        } else {
                            conv.state.push(JsonLdExpansionState::ObjectId {
                                types: Vec::new(),
                                id: None,
                                from_start: false,
                                reverse: false,
                            });
                        }
                    }
                    "@graph" => {
                        conv.state.push(JsonLdExpansionState::Object {
                            in_property: false,
                            has_emitted_id,
                        });
                        conv.state.push(JsonLdExpansionState::Graph);
                        conv.state.push(JsonLdExpansionState::Element {
                            is_array: false,
                            active_property: None,
                            container: &[],
                            reverse: false,
                        });
                        results.push(JsonLdEvent::StartGraph);
                    }
                    "@context" => {
                        errors.push(JsonLdSyntaxError::msg_and_code(
                            "@context must be the first key of an object",
                            JsonLdErrorCode::InvalidStreamingKeyOrder,
                        ));
                        conv.state.push(JsonLdExpansionState::Object {
                            in_property: false,
                            has_emitted_id,
                        });
                        conv.state
                            .push(JsonLdExpansionState::Skip { is_array: false });
                    }
                    "@type" => {
                        // TODO: be nice and allow this if lenient
                        errors.push(JsonLdSyntaxError::msg_and_code(
                            "@type must be the first key of an object or right after @context",
                            JsonLdErrorCode::InvalidStreamingKeyOrder,
                        ));
                        conv.state.push(JsonLdExpansionState::Object {
                            in_property: false,
                            has_emitted_id,
                        });
                        conv.state
                            .push(JsonLdExpansionState::Skip { is_array: false });
                    }
                    "@index" => {
                        conv.state.push(JsonLdExpansionState::Object {
                            in_property: false,
                            has_emitted_id,
                        });
                        conv.state.push(JsonLdExpansionState::Index);
                    }
                    "@reverse" => {
                        conv.state.push(JsonLdExpansionState::Object {
                            in_property: false,
                            has_emitted_id,
                        });
                        conv.state.push(JsonLdExpansionState::ReverseStart);
                    }
                    _ if has_keyword_form(&iri) => {
                        errors.push(if iri == "@list" || iri == "@set" {
                            JsonLdSyntaxError::msg_and_code(
                                "@list and @set must be the only keys of an object",
                                JsonLdErrorCode::InvalidSetOrListObject,
                            )
                        } else if iri == "@context" {
                            JsonLdSyntaxError::msg_and_code(
                                "@context must be the first key of an object",
                                JsonLdErrorCode::InvalidStreamingKeyOrder,
                            )
                        } else {
                            JsonLdSyntaxError::msg(format!("Unsupported JSON-LD keyword: {iri}"))
                        });
                        conv.state.push(JsonLdExpansionState::Object {
                            in_property: false,
                            has_emitted_id,
                        });
                        conv.state
                            .push(JsonLdExpansionState::Skip { is_array: false });
                    }
                    _ => {
                        let (container, reverse) = active_context(conv)
                            .term_definitions
                            .get(key.as_ref())
                            .map_or(([].as_slice(), false), |term_definition| {
                                (
                                    term_definition.container_mapping,
                                    term_definition.reverse_property,
                                )
                            });
                        conv.state.push(JsonLdExpansionState::Object {
                            in_property: true,
                            has_emitted_id,
                        });
                        conv.state.push(JsonLdExpansionState::Element {
                            active_property: Some(key.clone().into()),
                            is_array: false,
                            container,
                            reverse,
                        });
                        results.push(JsonLdEvent::StartProperty {
                            name: iri.into(),
                            reverse,
                        });
                    }
                }
            } else {
                conv.state.push(JsonLdExpansionState::Object {
                    in_property: false,
                    has_emitted_id,
                });
                conv.state
                    .push(JsonLdExpansionState::Skip { is_array: false });
            }
        }
        JsonEvent::Null
        | JsonEvent::String(_)
        | JsonEvent::Number(_)
        | JsonEvent::Boolean(_)
        | JsonEvent::StartArray
        | JsonEvent::EndArray
        | JsonEvent::StartObject
        | JsonEvent::Eof => unreachable!(),
    }
}

// -----------------------------------------------------------------------
// Reverse state handlers
// -----------------------------------------------------------------------

/// Handles `JsonLdExpansionState::ReverseStart`.
pub(super) fn handle_reverse_start_state(
    conv: &mut JsonLdExpansionConverter,
    event: JsonEvent<'_>,
    results: &mut Vec<JsonLdEvent>,
    errors: &mut Vec<JsonLdSyntaxError>,
) {
    if matches!(event, JsonEvent::StartObject) {
        conv.state
            .push(JsonLdExpansionState::Reverse { in_property: false });
    } else {
        errors.push(JsonLdSyntaxError::msg_and_code(
            "@reverse value must be a JSON object",
            JsonLdErrorCode::InvalidReverseValue,
        ));
        conv.state
            .push(JsonLdExpansionState::Skip { is_array: false });
        conv.convert_event(event, results, errors);
    }
}

/// Handles `JsonLdExpansionState::Reverse`.
pub(super) fn handle_reverse_state(
    conv: &mut JsonLdExpansionConverter,
    in_property: bool,
    event: JsonEvent<'_>,
    results: &mut Vec<JsonLdEvent>,
    errors: &mut Vec<JsonLdSyntaxError>,
) {
    if in_property {
        results.push(JsonLdEvent::EndProperty);
    }
    match event {
        JsonEvent::EndObject => (),
        JsonEvent::ObjectKey(key) => {
            if let Some(iri) = expand_iri_free(conv, key.as_ref().into(), false, true, errors) {
                if has_keyword_form(&iri) {
                    errors.push(JsonLdSyntaxError::msg_and_code(
                        format!("@reverse object value cannot contain any keyword, found {iri}",),
                        JsonLdErrorCode::InvalidReversePropertyMap,
                    ));
                    conv.state
                        .push(JsonLdExpansionState::Reverse { in_property: false });
                    conv.state
                        .push(JsonLdExpansionState::Skip { is_array: false });
                } else {
                    let (container, reverse) = active_context(conv)
                        .term_definitions
                        .get(key.as_ref())
                        .map_or(([].as_slice(), false), |term_definition| {
                            (
                                term_definition.container_mapping,
                                term_definition.reverse_property,
                            )
                        });
                    let reverse = !reverse; // We are in @reverse
                    conv.state
                        .push(JsonLdExpansionState::Reverse { in_property: true });
                    conv.state.push(JsonLdExpansionState::Element {
                        active_property: Some(key.clone().into()),
                        is_array: false,
                        container,
                        reverse,
                    });
                    results.push(JsonLdEvent::StartProperty {
                        name: iri.into(),
                        reverse,
                    });
                }
            } else {
                conv.state
                    .push(JsonLdExpansionState::Reverse { in_property: false });
                conv.state
                    .push(JsonLdExpansionState::Skip { is_array: false });
            }
        }
        JsonEvent::Null
        | JsonEvent::String(_)
        | JsonEvent::Number(_)
        | JsonEvent::Boolean(_)
        | JsonEvent::StartArray
        | JsonEvent::EndArray
        | JsonEvent::StartObject
        | JsonEvent::Eof => unreachable!(),
    }
}
