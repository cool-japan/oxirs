use serde_json::{json, Value};

use crate::metamodel::{Aspect, CharacteristicKind};

pub fn aspect_has_collection(aspect: &Aspect) -> bool {
    aspect.properties().iter().any(|prop| {
        prop.characteristic
            .as_ref()
            .map(|ch| is_collection_kind(ch.kind()))
            .unwrap_or(false)
    })
}

pub fn is_collection_kind(kind: &CharacteristicKind) -> bool {
    matches!(
        kind,
        CharacteristicKind::Collection { .. }
            | CharacteristicKind::List { .. }
            | CharacteristicKind::Set { .. }
            | CharacteristicKind::SortedSet { .. }
            | CharacteristicKind::TimeSeries { .. }
    )
}

pub fn make_nullable_v31(schema: Value) -> Value {
    let has_simple_type = schema
        .as_object()
        .and_then(|m| m.get("type"))
        .map(|t| t.is_string())
        .unwrap_or(false);

    if has_simple_type {
        let mut out = schema.as_object().cloned().unwrap_or_default();
        if let Some(t) = out.remove("type") {
            out.insert("type".to_string(), json!([t, "null"]));
        }
        Value::Object(out)
    } else {
        json!({ "oneOf": [schema, { "type": "null" }] })
    }
}

pub fn xsd_to_openapi_type(dt: &str) -> &'static str {
    if dt.ends_with("boolean") {
        return "boolean";
    }
    if dt.ends_with("int")
        || dt.ends_with("integer")
        || dt.ends_with("long")
        || dt.ends_with("short")
        || dt.ends_with("byte")
        || dt.ends_with("unsignedInt")
        || dt.ends_with("unsignedLong")
        || dt.ends_with("unsignedShort")
        || dt.ends_with("positiveInteger")
        || dt.ends_with("nonNegativeInteger")
    {
        return "integer";
    }
    if dt.ends_with("decimal") || dt.ends_with("float") || dt.ends_with("double") {
        return "number";
    }
    "string"
}

pub fn xsd_to_openapi_format(dt: &str) -> Option<&'static str> {
    if dt.ends_with("float") {
        return Some("float");
    }
    if dt.ends_with("double") {
        return Some("double");
    }
    if dt.ends_with("int") || dt.ends_with("integer") {
        return Some("int32");
    }
    if dt.ends_with("long") {
        return Some("int64");
    }
    if dt.ends_with("dateTime") || dt.ends_with("dateTimeStamp") {
        return Some("date-time");
    }
    if dt.ends_with("date") {
        return Some("date");
    }
    if dt.ends_with("base64Binary") {
        return Some("byte");
    }
    if dt.ends_with("hexBinary") {
        return Some("binary");
    }
    None
}

pub fn to_kebab_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('-');
        }
        result.push(ch.to_ascii_lowercase());
    }
    result
}

pub fn to_camel_case(s: &str) -> String {
    if s.is_empty() {
        return s.to_string();
    }
    let mut chars = s.chars();
    let first = chars.next().expect("non-empty string").to_ascii_lowercase();
    format!("{}{}", first, chars.as_str())
}
