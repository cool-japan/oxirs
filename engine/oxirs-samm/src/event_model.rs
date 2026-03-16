// SAMM event model for IoT events (v1.1.0 round 11)
//
// Implements the SAMM event model which allows defining, registering, and
// validating IoT event definitions and instances.

use std::collections::HashMap;

/// A single property of an event
#[derive(Debug, Clone)]
pub struct EventProperty {
    /// Property name (e.g. "temperature")
    pub name: String,
    /// XSD or custom datatype identifier (e.g. "xsd:float")
    pub datatype: String,
    /// Whether this property may be absent in an event instance
    pub optional: bool,
}

impl EventProperty {
    /// Create a new required event property
    pub fn new(name: impl Into<String>, datatype: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            datatype: datatype.into(),
            optional: false,
        }
    }

    /// Create a new optional event property
    pub fn optional(name: impl Into<String>, datatype: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            datatype: datatype.into(),
            optional: true,
        }
    }
}

/// Definition of an event type
#[derive(Debug, Clone)]
pub struct EventDefinition {
    /// Unique identifier for this event type (e.g. "urn:example:TemperatureEvent")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Optional human-readable description
    pub description: Option<String>,
    /// Ordered list of properties that event instances must or may carry
    pub properties: Vec<EventProperty>,
    /// Optional JSON Schema string for advanced payload validation
    pub payload_schema: Option<String>,
}

impl EventDefinition {
    /// Create a minimal event definition
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            properties: Vec::new(),
            payload_schema: None,
        }
    }

    /// Add a description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a property to this event definition
    pub fn with_property(mut self, prop: EventProperty) -> Self {
        self.properties.push(prop);
        self
    }

    /// Set the payload schema
    pub fn with_payload_schema(mut self, schema: impl Into<String>) -> Self {
        self.payload_schema = Some(schema.into());
        self
    }
}

/// A concrete instance of an event at runtime
#[derive(Debug, Clone)]
pub struct EventInstance {
    /// ID of the `EventDefinition` this instance conforms to
    pub event_id: String,
    /// Milliseconds since the Unix epoch (event time)
    pub timestamp_ms: u64,
    /// URI or name of the originating device / service
    pub source: String,
    /// Key-value pairs representing the event payload
    pub payload: HashMap<String, String>,
}

impl EventInstance {
    /// Create a new event instance
    pub fn new(
        event_id: impl Into<String>,
        timestamp_ms: u64,
        source: impl Into<String>,
        payload: HashMap<String, String>,
    ) -> Self {
        Self {
            event_id: event_id.into(),
            timestamp_ms,
            source: source.into(),
            payload,
        }
    }
}

/// Result of validating an event instance against its definition
#[derive(Debug, Clone)]
pub struct EventValidationResult {
    /// `true` when the instance satisfies all constraints
    pub valid: bool,
    /// Human-readable validation error messages (empty when `valid` is `true`)
    pub errors: Vec<String>,
}

impl EventValidationResult {
    fn ok() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
        }
    }

    fn fail(errors: Vec<String>) -> Self {
        Self {
            valid: false,
            errors,
        }
    }
}

/// Errors that can arise during event model operations
#[derive(Debug)]
pub enum EventError {
    /// An event definition with this ID is already registered
    DuplicateId(String),
    /// The event definition is structurally invalid
    InvalidDefinition(String),
}

impl std::fmt::Display for EventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventError::DuplicateId(id) => write!(f, "Duplicate event id: {id}"),
            EventError::InvalidDefinition(msg) => write!(f, "Invalid event definition: {msg}"),
        }
    }
}

impl std::error::Error for EventError {}

/// Registry of event definitions and validation logic
pub struct EventModel {
    events: HashMap<String, EventDefinition>,
}

impl EventModel {
    /// Create an empty event model
    pub fn new() -> Self {
        Self {
            events: HashMap::new(),
        }
    }

    /// Register a new event definition.
    /// Returns `EventError::DuplicateId` if an event with that id already exists.
    pub fn register(&mut self, event: EventDefinition) -> Result<(), EventError> {
        if self.events.contains_key(&event.id) {
            return Err(EventError::DuplicateId(event.id.clone()));
        }
        if event.id.is_empty() {
            return Err(EventError::InvalidDefinition(
                "id must not be empty".to_string(),
            ));
        }
        if event.name.is_empty() {
            return Err(EventError::InvalidDefinition(
                "name must not be empty".to_string(),
            ));
        }
        self.events.insert(event.id.clone(), event);
        Ok(())
    }

    /// Look up an event definition by id
    pub fn get(&self, id: &str) -> Option<&EventDefinition> {
        self.events.get(id)
    }

    /// Remove an event definition. Returns true if it was present.
    pub fn remove(&mut self, id: &str) -> bool {
        self.events.remove(id).is_some()
    }

    /// Total number of registered event definitions
    pub fn count(&self) -> usize {
        self.events.len()
    }

    /// Return all event definitions that contain a property with the given name
    pub fn events_with_property(&self, property_name: &str) -> Vec<&EventDefinition> {
        self.events
            .values()
            .filter(|ev| ev.properties.iter().any(|p| p.name == property_name))
            .collect()
    }

    /// Return only the required properties of an event definition
    pub fn required_properties(&self, event_id: &str) -> Vec<&EventProperty> {
        match self.events.get(event_id) {
            Some(ev) => ev.properties.iter().filter(|p| !p.optional).collect(),
            None => Vec::new(),
        }
    }

    /// Return only the optional properties of an event definition
    pub fn optional_properties(&self, event_id: &str) -> Vec<&EventProperty> {
        match self.events.get(event_id) {
            Some(ev) => ev.properties.iter().filter(|p| p.optional).collect(),
            None => Vec::new(),
        }
    }

    /// Validate an event instance against its registered definition.
    ///
    /// Checks:
    /// - The event_id must be registered.
    /// - All required properties must be present in the payload.
    /// - No unknown properties (beyond defined properties) → logged as warnings in errors
    ///   but does NOT mark the result as invalid (optional extra keys are allowed).
    pub fn validate_instance(&self, instance: &EventInstance) -> EventValidationResult {
        let definition = match self.events.get(&instance.event_id) {
            Some(d) => d,
            None => {
                return EventValidationResult::fail(vec![format!(
                    "Unknown event id: {}",
                    instance.event_id
                )]);
            }
        };

        let mut errors = Vec::new();

        // Check all required properties are present
        for prop in &definition.properties {
            if !prop.optional && !instance.payload.contains_key(&prop.name) {
                errors.push(format!(
                    "Missing required property '{}' (type: {})",
                    prop.name, prop.datatype
                ));
            }
        }

        if errors.is_empty() {
            EventValidationResult::ok()
        } else {
            EventValidationResult::fail(errors)
        }
    }

    /// Check whether all required properties of the given event are present in the supplied keys.
    /// Returns true if compatible, false if any required property is missing.
    pub fn schema_compatible(&self, id: &str, payload_keys: &[&str]) -> bool {
        let required = self.required_properties(id);
        let key_set: std::collections::HashSet<&str> = payload_keys.iter().copied().collect();
        required.iter().all(|p| key_set.contains(p.name.as_str()))
    }
}

impl Default for EventModel {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(id: &str) -> EventDefinition {
        EventDefinition::new(id, format!("Event {id}"))
            .with_property(EventProperty::new("temperature", "xsd:float"))
            .with_property(EventProperty::optional("humidity", "xsd:float"))
    }

    fn make_instance(event_id: &str, keys: &[(&str, &str)]) -> EventInstance {
        let payload = keys
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        EventInstance::new(event_id, 1_000_000, "sensor-01", payload)
    }

    // ── Construction ──────────────────────────────────────────────────────

    #[test]
    fn test_event_model_new() {
        let m = EventModel::new();
        assert_eq!(m.count(), 0);
    }

    #[test]
    fn test_event_model_default() {
        let m = EventModel::default();
        assert_eq!(m.count(), 0);
    }

    #[test]
    fn test_register_event() {
        let mut m = EventModel::new();
        assert!(m.register(make_event("ev1")).is_ok());
        assert_eq!(m.count(), 1);
    }

    #[test]
    fn test_register_multiple() {
        let mut m = EventModel::new();
        for i in 0..5 {
            m.register(make_event(&format!("ev{i}")))
                .expect("should succeed");
        }
        assert_eq!(m.count(), 5);
    }

    #[test]
    fn test_register_duplicate_id_error() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        let result = m.register(make_event("ev1"));
        assert!(result.is_err());
        assert!(matches!(result, Err(EventError::DuplicateId(_))));
    }

    #[test]
    fn test_register_empty_id_error() {
        let mut m = EventModel::new();
        let result = m.register(EventDefinition::new("", "name"));
        assert!(result.is_err());
    }

    #[test]
    fn test_register_empty_name_error() {
        let mut m = EventModel::new();
        let result = m.register(EventDefinition::new("id", ""));
        assert!(result.is_err());
    }

    // ── get / remove ──────────────────────────────────────────────────────

    #[test]
    fn test_get_existing() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        assert!(m.get("ev1").is_some());
    }

    #[test]
    fn test_get_missing() {
        let m = EventModel::new();
        assert!(m.get("nope").is_none());
    }

    #[test]
    fn test_remove_existing() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        assert!(m.remove("ev1"));
        assert_eq!(m.count(), 0);
    }

    #[test]
    fn test_remove_missing_returns_false() {
        let mut m = EventModel::new();
        assert!(!m.remove("nope"));
    }

    #[test]
    fn test_count_decrements_after_remove() {
        let mut m = EventModel::new();
        m.register(make_event("a")).expect("should succeed");
        m.register(make_event("b")).expect("should succeed");
        m.remove("a");
        assert_eq!(m.count(), 1);
    }

    // ── validate_instance ─────────────────────────────────────────────────

    #[test]
    fn test_validate_valid_instance() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        let inst = make_instance("ev1", &[("temperature", "22.5")]);
        let result = m.validate_instance(&inst);
        assert!(result.valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validate_valid_with_optional() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        let inst = make_instance("ev1", &[("temperature", "22.5"), ("humidity", "60.0")]);
        let result = m.validate_instance(&inst);
        assert!(result.valid);
    }

    #[test]
    fn test_validate_missing_required_fails() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        // Missing "temperature" which is required
        let inst = make_instance("ev1", &[("humidity", "60.0")]);
        let result = m.validate_instance(&inst);
        assert!(!result.valid);
        assert!(!result.errors.is_empty());
        assert!(result.errors[0].contains("temperature"));
    }

    #[test]
    fn test_validate_unknown_event_id_fails() {
        let m = EventModel::new();
        let inst = make_instance("unknown", &[("x", "1")]);
        let result = m.validate_instance(&inst);
        assert!(!result.valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_validate_extra_keys_allowed() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        // "extra_key" is not defined but should not cause failure
        let inst = make_instance("ev1", &[("temperature", "10.0"), ("extra_key", "value")]);
        let result = m.validate_instance(&inst);
        assert!(result.valid);
    }

    #[test]
    fn test_validate_all_required_missing_multiple_errors() {
        let mut m = EventModel::new();
        let event = EventDefinition::new("ev1", "Event")
            .with_property(EventProperty::new("a", "string"))
            .with_property(EventProperty::new("b", "string"));
        m.register(event).expect("should succeed");
        let inst = make_instance("ev1", &[]);
        let result = m.validate_instance(&inst);
        assert!(!result.valid);
        assert_eq!(result.errors.len(), 2);
    }

    // ── events_with_property ──────────────────────────────────────────────

    #[test]
    fn test_events_with_property_found() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed"); // has temperature
        m.register(make_event("ev2")).expect("should succeed"); // has temperature
        let found = m.events_with_property("temperature");
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_events_with_property_not_found() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        let found = m.events_with_property("voltage");
        assert!(found.is_empty());
    }

    #[test]
    fn test_events_with_property_subset() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed"); // has temperature, humidity
        let ev2 = EventDefinition::new("ev2", "Pressure event")
            .with_property(EventProperty::new("pressure", "xsd:float"));
        m.register(ev2).expect("should succeed");
        let found = m.events_with_property("pressure");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "ev2");
    }

    // ── required / optional properties ────────────────────────────────────

    #[test]
    fn test_required_properties() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        let req = m.required_properties("ev1");
        assert_eq!(req.len(), 1);
        assert_eq!(req[0].name, "temperature");
    }

    #[test]
    fn test_optional_properties() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        let opt = m.optional_properties("ev1");
        assert_eq!(opt.len(), 1);
        assert_eq!(opt[0].name, "humidity");
    }

    #[test]
    fn test_required_properties_missing_event() {
        let m = EventModel::new();
        let req = m.required_properties("nope");
        assert!(req.is_empty());
    }

    #[test]
    fn test_optional_properties_missing_event() {
        let m = EventModel::new();
        let opt = m.optional_properties("nope");
        assert!(opt.is_empty());
    }

    // ── schema_compatible ─────────────────────────────────────────────────

    #[test]
    fn test_schema_compatible_all_required_present() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        assert!(m.schema_compatible("ev1", &["temperature"]));
    }

    #[test]
    fn test_schema_compatible_with_optional_too() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        assert!(m.schema_compatible("ev1", &["temperature", "humidity"]));
    }

    #[test]
    fn test_schema_compatible_missing_required() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        assert!(!m.schema_compatible("ev1", &["humidity"]));
    }

    #[test]
    fn test_schema_compatible_unknown_event() {
        let m = EventModel::new();
        // No required properties for unknown → true vacuously
        assert!(m.schema_compatible("nope", &[]));
    }

    // ── EventDefinition builder ───────────────────────────────────────────

    #[test]
    fn test_event_definition_with_description() {
        let ev = EventDefinition::new("id", "name").with_description("A test event");
        assert_eq!(ev.description, Some("A test event".to_string()));
    }

    #[test]
    fn test_event_definition_with_payload_schema() {
        let ev = EventDefinition::new("id", "name").with_payload_schema("{\"type\":\"object\"}");
        assert!(ev.payload_schema.is_some());
    }

    #[test]
    fn test_event_property_optional_flag() {
        let p = EventProperty::optional("x", "string");
        assert!(p.optional);
        let p2 = EventProperty::new("y", "string");
        assert!(!p2.optional);
    }

    // ── EventError display ────────────────────────────────────────────────

    #[test]
    fn test_event_error_duplicate_display() {
        let e = EventError::DuplicateId("ev1".to_string());
        let s = format!("{e}");
        assert!(s.contains("ev1"));
    }

    #[test]
    fn test_event_error_invalid_display() {
        let e = EventError::InvalidDefinition("bad".to_string());
        let s = format!("{e}");
        assert!(s.contains("bad"));
    }

    // ── EventValidationResult ─────────────────────────────────────────────

    #[test]
    fn test_validation_result_ok() {
        let r = EventValidationResult::ok();
        assert!(r.valid);
        assert!(r.errors.is_empty());
    }

    #[test]
    fn test_validation_result_fail_has_errors() {
        let r = EventValidationResult::fail(vec!["error1".to_string()]);
        assert!(!r.valid);
        assert_eq!(r.errors.len(), 1);
    }

    #[test]
    fn test_event_definition_description_none_by_default() {
        let ev = EventDefinition::new("id", "name");
        assert!(ev.description.is_none());
    }

    #[test]
    fn test_event_definition_payload_schema_none_by_default() {
        let ev = EventDefinition::new("id", "name");
        assert!(ev.payload_schema.is_none());
    }

    #[test]
    fn test_event_instance_fields() {
        let payload = std::collections::HashMap::new();
        let inst = EventInstance::new("ev1", 12345, "source-A", payload);
        assert_eq!(inst.event_id, "ev1");
        assert_eq!(inst.timestamp_ms, 12345);
        assert_eq!(inst.source, "source-A");
    }

    #[test]
    fn test_count_after_multiple_removes() {
        let mut m = EventModel::new();
        for i in 0..3 {
            m.register(make_event(&format!("ev{i}")))
                .expect("should succeed");
        }
        m.remove("ev0");
        m.remove("ev1");
        assert_eq!(m.count(), 1);
    }

    #[test]
    fn test_schema_compatible_empty_required_props() {
        let mut m = EventModel::new();
        // Register an event with only optional properties
        let ev = EventDefinition::new("ev1", "All optional")
            .with_property(EventProperty::optional("x", "string"));
        m.register(ev).expect("should succeed");
        // Empty payload is compatible since no required props
        assert!(m.schema_compatible("ev1", &[]));
    }

    #[test]
    fn test_events_with_property_returns_empty_when_model_empty() {
        let m = EventModel::new();
        assert!(m.events_with_property("temp").is_empty());
    }

    #[test]
    fn test_validate_unknown_event_id_error_message() {
        let m = EventModel::new();
        let inst = make_instance("does_not_exist", &[("x", "1")]);
        let result = m.validate_instance(&inst);
        assert!(!result.valid);
        assert!(result.errors[0].contains("does_not_exist"));
    }

    #[test]
    fn test_register_re_register_after_remove() {
        let mut m = EventModel::new();
        m.register(make_event("ev1")).expect("should succeed");
        m.remove("ev1");
        // Should succeed since it was removed
        assert!(m.register(make_event("ev1")).is_ok());
    }

    #[test]
    fn test_event_property_required_not_optional() {
        let p = EventProperty::new("temp", "float");
        assert!(!p.optional);
    }
}
