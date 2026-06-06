//! JSON-LD expansion state machine types, event types, and helper utilities.

use json_event_parser::JsonEvent;
use std::borrow::Cow;

/// JSON-LD output events emitted during expansion
pub enum JsonLdEvent {
    StartObject {
        types: Vec<String>,
    },
    EndObject,
    StartProperty {
        name: String,
        reverse: bool,
    },
    EndProperty,
    Id(String),
    Value {
        value: JsonLdValue,
        r#type: Option<String>,
        language: Option<String>,
    },
    StartGraph,
    EndGraph,
    StartList,
    EndList,
    StartSet,
    EndSet,
}

/// JSON-LD value types
pub enum JsonLdValue {
    String(String),
    Number(String),
    Boolean(bool),
}

/// Internal state machine states for the expansion algorithm
pub(crate) enum JsonLdExpansionState {
    Element {
        active_property: Option<String>,
        is_array: bool,
        container: &'static [&'static str],
        reverse: bool,
    },
    ObjectOrContainerStart {
        buffer: Vec<(String, Vec<JsonEvent<'static>>)>,
        depth: usize,
        current_key: Option<String>,
        active_property: Option<String>,
        container: &'static [&'static str],
        reverse: bool,
    },
    ObjectOrContainerStartStreaming {
        active_property: Option<String>,
        container: &'static [&'static str],
        reverse: bool,
    },
    Context {
        buffer: Vec<JsonEvent<'static>>,
        depth: usize,
        active_property: Option<String>,
        container: &'static [&'static str],
        reverse: bool,
    },
    ObjectStart {
        types: Vec<String>,
        id: Option<String>,
        seen_id: bool,
        active_property: Option<String>,
        reverse: bool,
    },
    ObjectType {
        types: Vec<String>,
        id: Option<String>,
        is_array: bool,
        active_property: Option<String>,
        reverse: bool,
    },
    ObjectId {
        types: Vec<String>,
        id: Option<String>,
        from_start: bool,
        reverse: bool,
    },
    Object {
        in_property: bool,
        has_emitted_id: bool,
    },
    ReverseStart,
    Reverse {
        in_property: bool,
    },
    Value {
        r#type: Option<String>,
        value: Option<JsonLdValue>,
        language: Option<String>,
    },
    ValueValue {
        r#type: Option<String>,
        language: Option<String>,
    },
    ValueLanguage {
        r#type: Option<String>,
        value: Option<JsonLdValue>,
    },
    ValueType {
        value: Option<JsonLdValue>,
        language: Option<String>,
    },
    Index,
    Graph,
    RootGraph,
    ListOrSetContainer {
        needs_end_object: bool,
        end_event: Option<JsonLdEvent>,
    },
    IndexContainer {
        active_property: Option<String>,
    },
    LanguageContainer,
    LanguageContainerValue {
        language: String,
        is_array: bool,
    },
    Skip {
        is_array: bool,
    },
}

/// Convert a borrowed `JsonEvent` to an owned (static lifetime) version.
pub(crate) fn to_owned_event(event: JsonEvent<'_>) -> JsonEvent<'static> {
    match event {
        JsonEvent::String(s) => JsonEvent::String(Cow::Owned(s.into())),
        JsonEvent::Number(n) => JsonEvent::Number(Cow::Owned(n.into())),
        JsonEvent::Boolean(b) => JsonEvent::Boolean(b),
        JsonEvent::Null => JsonEvent::Null,
        JsonEvent::StartArray => JsonEvent::StartArray,
        JsonEvent::EndArray => JsonEvent::EndArray,
        JsonEvent::StartObject => JsonEvent::StartObject,
        JsonEvent::EndObject => JsonEvent::EndObject,
        JsonEvent::ObjectKey(k) => JsonEvent::ObjectKey(Cow::Owned(k.into())),
        JsonEvent::Eof => JsonEvent::Eof,
    }
}
