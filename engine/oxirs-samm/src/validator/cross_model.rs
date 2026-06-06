//! Cross-model reference validation.
//!
//! Validates external URN references that span independently loaded SAMM
//! aspect models. When a property in model `A` references a type (entity,
//! characteristic, …) defined in model `B`, the resolver looks the target
//! URN up in a [`ModelRegistry`] and reports unresolved or type-mismatched
//! references.
//!
//! This closes the ESMF parity gap "Cross-model reference validation",
//! mirroring `io.openmanufacturing.sds.aspectmodel.resolver.services.SammAspectMetaModelResourceResolver`
//! from the upstream `esmf-sdk`.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_samm::metamodel::Aspect;
//! use oxirs_samm::validator::{
//!     cross_model_validator, ModelRegistry, ResolvedElementKind,
//! };
//!
//! let mut registry = ModelRegistry::new();
//! // populate registry with externally resolved elements...
//! # let aspect = Aspect::new("urn:samm:org.example:1.0.0#MyAspect".to_string());
//! let validator = cross_model_validator(registry);
//! let errors = validator.validate_references(&aspect);
//! assert!(errors.is_empty());
//! ```

use crate::metamodel::{
    Aspect, Characteristic, CharacteristicKind, Constraint, Entity, ModelElement, Operation,
    Property,
};
use std::collections::{HashMap, HashSet};

/// Kind of SAMM element exposed by a resolved external model.
///
/// Used so that the validator can detect a property pointing at a URN whose
/// target turns out to be a `Characteristic`, or a data-type slot pointing at
/// a URN whose target is an `Operation`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolvedElementKind {
    /// `samm:Aspect`
    Aspect,
    /// `samm:Property`
    Property,
    /// `samm:Characteristic` (or any `samm-c:*` characteristic subclass)
    Characteristic,
    /// `samm:Entity`
    Entity,
    /// `samm:Operation`
    Operation,
    /// `samm:Event`
    Event,
    /// A primitive xsd / RDF data type
    DataType,
}

impl ResolvedElementKind {
    /// Human-readable name used in error messages.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Aspect => "Aspect",
            Self::Property => "Property",
            Self::Characteristic => "Characteristic",
            Self::Entity => "Entity",
            Self::Operation => "Operation",
            Self::Event => "Event",
            Self::DataType => "DataType",
        }
    }
}

/// A single SAMM element resolved from an externally loaded model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedElement {
    /// URN of the element.
    pub urn: String,
    /// Kind of element (so the validator can check expected vs. actual).
    pub kind: ResolvedElementKind,
    /// URN of the model file that defined the element.
    pub source_model_urn: String,
}

impl ResolvedElement {
    /// Build a new resolved element record.
    pub fn new(
        urn: impl Into<String>,
        kind: ResolvedElementKind,
        source_model_urn: impl Into<String>,
    ) -> Self {
        Self {
            urn: urn.into(),
            kind,
            source_model_urn: source_model_urn.into(),
        }
    }
}

/// Registry of externally resolved SAMM models, keyed by element URN.
///
/// Use this to feed the [`CrossModelReferenceValidator`] with every element
/// it should treat as known. Anything not in the registry that is referenced
/// by URN — and is not internal to the model under validation — will be
/// flagged as `Unresolved`.
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    elements: HashMap<String, ResolvedElement>,
}

impl ModelRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a resolved element.
    pub fn insert(&mut self, element: ResolvedElement) {
        self.elements.insert(element.urn.clone(), element);
    }

    /// Number of registered elements.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// `true` if no elements are registered.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Look up an element by URN.
    pub fn get(&self, urn: &str) -> Option<&ResolvedElement> {
        self.elements.get(urn)
    }

    /// Iterator over all registered elements.
    pub fn iter(&self) -> impl Iterator<Item = &ResolvedElement> {
        self.elements.values()
    }

    /// Register a whole aspect: the aspect itself, then every property,
    /// characteristic, operation, event, and nested entity declared on it.
    ///
    /// Convenience for callers who already have an in-memory model.
    pub fn register_aspect(&mut self, aspect: &Aspect) {
        let source = model_urn(aspect.urn());
        self.insert(ResolvedElement::new(
            aspect.urn(),
            ResolvedElementKind::Aspect,
            &source,
        ));

        for property in aspect.properties() {
            self.register_property(property, &source);
        }
        for operation in aspect.operations() {
            self.register_operation(operation, &source);
        }
        for event in aspect.events() {
            self.insert(ResolvedElement::new(
                event.urn(),
                ResolvedElementKind::Event,
                &source,
            ));
            for parameter in event.parameters() {
                self.register_property(parameter, &source);
            }
        }
    }

    fn register_property(&mut self, property: &Property, source: &str) {
        self.insert(ResolvedElement::new(
            property.urn(),
            ResolvedElementKind::Property,
            source,
        ));
        if let Some(ref characteristic) = property.characteristic {
            self.register_characteristic(characteristic, source);
        }
    }

    fn register_characteristic(&mut self, characteristic: &Characteristic, source: &str) {
        self.insert(ResolvedElement::new(
            characteristic.urn(),
            ResolvedElementKind::Characteristic,
            source,
        ));
        match &characteristic.kind {
            CharacteristicKind::Either { left, right } => {
                self.register_characteristic(left, source);
                self.register_characteristic(right, source);
            }
            CharacteristicKind::Collection {
                element_characteristic,
            }
            | CharacteristicKind::List {
                element_characteristic,
            }
            | CharacteristicKind::Set {
                element_characteristic,
            }
            | CharacteristicKind::SortedSet {
                element_characteristic,
            }
            | CharacteristicKind::TimeSeries {
                element_characteristic,
            } => {
                if let Some(inner) = element_characteristic.as_deref() {
                    self.register_characteristic(inner, source);
                }
            }
            _ => {}
        }
    }

    fn register_operation(&mut self, operation: &Operation, source: &str) {
        self.insert(ResolvedElement::new(
            operation.urn(),
            ResolvedElementKind::Operation,
            source,
        ));
        for input in operation.input() {
            self.register_property(input, source);
        }
        if let Some(output) = operation.output() {
            self.register_property(output, source);
        }
    }

    /// Register a free-standing `Entity` (entities are not yet attached to
    /// `Aspect` in the in-memory model but may live in separately loaded
    /// resources).
    pub fn register_entity(&mut self, entity: &Entity) {
        let source = model_urn(entity.urn());
        self.insert(ResolvedElement::new(
            entity.urn(),
            ResolvedElementKind::Entity,
            &source,
        ));
        for property in entity.properties() {
            self.register_property(property, &source);
        }
    }
}

/// An error raised by the cross-model reference validator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossModelError {
    /// A URN reference could not be resolved against the registry.
    Unresolved {
        /// URN of the element that emitted the reference (e.g. property URN).
        from: String,
        /// URN that was looked up.
        to: String,
        /// Reference-site kind hint (what the referrer expected `to` to be).
        expected: ResolvedElementKind,
    },
    /// A URN reference was resolved but to the wrong kind of element.
    TypeMismatch {
        /// URN of the element that emitted the reference.
        from: String,
        /// URN that was looked up.
        to: String,
        /// Kind the referrer expected.
        expected: ResolvedElementKind,
        /// Kind the registry actually returned.
        got: ResolvedElementKind,
    },
}

impl std::fmt::Display for CrossModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unresolved { from, to, expected } => write!(
                f,
                "Cross-model reference unresolved: <{from}> references <{to}> as {expected}, but the target was not found in any loaded model",
                expected = expected.as_str(),
            ),
            Self::TypeMismatch {
                from,
                to,
                expected,
                got,
            } => write!(
                f,
                "Cross-model reference type mismatch: <{from}> expects <{to}> to be a {expected}, but the registry resolves it to a {got}",
                expected = expected.as_str(),
                got = got.as_str(),
            ),
        }
    }
}

impl std::error::Error for CrossModelError {}

/// Validator that checks every URN reference inside an `Aspect` against a
/// [`ModelRegistry`].
///
/// Construct via [`cross_model_validator`].
#[derive(Debug, Clone)]
pub struct CrossModelReferenceValidator {
    registry: ModelRegistry,
}

impl CrossModelReferenceValidator {
    /// Construct directly from a registry.
    pub fn new(registry: ModelRegistry) -> Self {
        Self { registry }
    }

    /// Immutable access to the underlying registry.
    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }

    /// Walk every property, characteristic, operation, and event in `aspect`
    /// and resolve every URN reference. Returns one entry per failed
    /// resolution (no error means everything resolved cleanly).
    pub fn validate_references(&self, aspect: &Aspect) -> Vec<CrossModelError> {
        let mut errors = Vec::new();
        let mut visited_chars: HashSet<String> = HashSet::new();

        let internal_model_urn = model_urn(aspect.urn());

        for property in aspect.properties() {
            self.check_property(property, &internal_model_urn, &mut errors, &mut visited_chars);
        }
        for operation in aspect.operations() {
            self.check_operation(operation, &internal_model_urn, &mut errors, &mut visited_chars);
        }
        for event in aspect.events() {
            for parameter in event.parameters() {
                self.check_property(parameter, &internal_model_urn, &mut errors, &mut visited_chars);
            }
        }

        errors
    }

    fn check_property(
        &self,
        property: &Property,
        internal_model_urn: &str,
        errors: &mut Vec<CrossModelError>,
        visited_chars: &mut HashSet<String>,
    ) {
        if let Some(ref parent_urn) = property.extends {
            self.check_reference(
                property.urn(),
                parent_urn,
                ResolvedElementKind::Property,
                internal_model_urn,
                errors,
            );
        }
        if let Some(ref characteristic) = property.characteristic {
            self.check_reference(
                property.urn(),
                characteristic.urn(),
                ResolvedElementKind::Characteristic,
                internal_model_urn,
                errors,
            );
            self.check_characteristic(characteristic, internal_model_urn, errors, visited_chars);
        }
    }

    fn check_characteristic(
        &self,
        characteristic: &Characteristic,
        internal_model_urn: &str,
        errors: &mut Vec<CrossModelError>,
        visited_chars: &mut HashSet<String>,
    ) {
        if !visited_chars.insert(characteristic.urn().to_string()) {
            return;
        }

        if let Some(ref data_type) = characteristic.data_type {
            if !is_primitive_xsd_type(data_type) {
                self.check_reference(
                    characteristic.urn(),
                    data_type,
                    ResolvedElementKind::DataType,
                    internal_model_urn,
                    errors,
                );
            }
        }

        for constraint in &characteristic.constraints {
            self.check_constraint(characteristic.urn(), constraint, internal_model_urn, errors);
        }

        match &characteristic.kind {
            CharacteristicKind::Either { left, right } => {
                self.check_reference(
                    characteristic.urn(),
                    left.urn(),
                    ResolvedElementKind::Characteristic,
                    internal_model_urn,
                    errors,
                );
                self.check_characteristic(left, internal_model_urn, errors, visited_chars);
                self.check_reference(
                    characteristic.urn(),
                    right.urn(),
                    ResolvedElementKind::Characteristic,
                    internal_model_urn,
                    errors,
                );
                self.check_characteristic(right, internal_model_urn, errors, visited_chars);
            }
            CharacteristicKind::Collection {
                element_characteristic,
            }
            | CharacteristicKind::List {
                element_characteristic,
            }
            | CharacteristicKind::Set {
                element_characteristic,
            }
            | CharacteristicKind::SortedSet {
                element_characteristic,
            }
            | CharacteristicKind::TimeSeries {
                element_characteristic,
            } => {
                if let Some(inner) = element_characteristic.as_deref() {
                    self.check_reference(
                        characteristic.urn(),
                        inner.urn(),
                        ResolvedElementKind::Characteristic,
                        internal_model_urn,
                        errors,
                    );
                    self.check_characteristic(inner, internal_model_urn, errors, visited_chars);
                }
            }
            CharacteristicKind::SingleEntity { entity_type } => {
                self.check_reference(
                    characteristic.urn(),
                    entity_type,
                    ResolvedElementKind::Entity,
                    internal_model_urn,
                    errors,
                );
            }
            _ => {}
        }
    }

    fn check_constraint(
        &self,
        owner_urn: &str,
        constraint: &Constraint,
        _internal_model_urn: &str,
        _errors: &mut Vec<CrossModelError>,
    ) {
        // Reserved hook: existing SAMM 2.x constraint kinds embed only
        // primitive values, but future constraint kinds (e.g. CodedConstraint
        // referencing an external code list URN) will surface external URN
        // references here.
        let _ = (owner_urn, constraint);
    }

    fn check_operation(
        &self,
        operation: &Operation,
        internal_model_urn: &str,
        errors: &mut Vec<CrossModelError>,
        visited_chars: &mut HashSet<String>,
    ) {
        for input in operation.input() {
            self.check_property(input, internal_model_urn, errors, visited_chars);
        }
        if let Some(output) = operation.output() {
            self.check_property(output, internal_model_urn, errors, visited_chars);
        }
    }

    fn check_reference(
        &self,
        from: &str,
        to: &str,
        expected: ResolvedElementKind,
        internal_model_urn: &str,
        errors: &mut Vec<CrossModelError>,
    ) {
        if !is_urn_reference(to) {
            return;
        }
        if model_urn(to) == internal_model_urn {
            return;
        }
        match self.registry.get(to) {
            None => errors.push(CrossModelError::Unresolved {
                from: from.to_string(),
                to: to.to_string(),
                expected,
            }),
            Some(found) => {
                if found.kind != expected {
                    errors.push(CrossModelError::TypeMismatch {
                        from: from.to_string(),
                        to: to.to_string(),
                        expected,
                        got: found.kind,
                    });
                }
            }
        }
    }
}

/// Construct a [`CrossModelReferenceValidator`] from a [`ModelRegistry`].
///
/// Sugar for `CrossModelReferenceValidator::new(registry)`.
pub fn cross_model_validator(registry: ModelRegistry) -> CrossModelReferenceValidator {
    CrossModelReferenceValidator::new(registry)
}

/// Extract the model-identifying prefix from a SAMM URN.
///
/// SAMM URNs follow the shape `urn:samm:<namespace>:<version>#<localName>`;
/// the *model* URN is the portion up to (but not including) the `#`. Two URNs
/// share a model iff they share this prefix.
fn model_urn(urn: &str) -> String {
    match urn.rsplit_once('#') {
        Some((prefix, _)) => prefix.to_string(),
        None => urn.to_string(),
    }
}

/// `true` iff the string looks like a SAMM/RDF URN reference (not a literal).
fn is_urn_reference(value: &str) -> bool {
    value.starts_with("urn:") || value.starts_with("http://") || value.starts_with("https://")
}

/// `true` iff the data type string is a known primitive xsd/rdf type that
/// does not require external resolution.
fn is_primitive_xsd_type(data_type: &str) -> bool {
    data_type.starts_with("http://www.w3.org/2001/XMLSchema#")
        || data_type.starts_with("xsd:")
        || data_type.starts_with("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        || data_type.starts_with("rdf:")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Characteristic, CharacteristicKind, Property};

    fn make_aspect_a() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:org.example.a:1.0.0#AspectA".to_string());

        let char_external = Characteristic::new(
            "urn:samm:org.example.b:1.0.0#TextChar".to_string(),
            CharacteristicKind::Trait,
        );
        let property = Property::new("urn:samm:org.example.a:1.0.0#name".to_string())
            .with_characteristic(char_external);
        aspect.add_property(property);

        aspect
    }

    #[test]
    fn unresolved_external_reference_is_flagged() {
        let aspect = make_aspect_a();
        let registry = ModelRegistry::new();
        let validator = cross_model_validator(registry);

        let errors = validator.validate_references(&aspect);
        assert_eq!(errors.len(), 1, "expected one unresolved error: {errors:?}");
        match &errors[0] {
            CrossModelError::Unresolved { from, to, expected } => {
                assert_eq!(from, "urn:samm:org.example.a:1.0.0#name");
                assert_eq!(to, "urn:samm:org.example.b:1.0.0#TextChar");
                assert_eq!(*expected, ResolvedElementKind::Characteristic);
            }
            other => panic!("expected Unresolved, got {other:?}"),
        }
    }

    #[test]
    fn internal_reference_passes() {
        let mut aspect = Aspect::new("urn:samm:org.example.a:1.0.0#AspectA".to_string());
        let internal_char = Characteristic::new(
            "urn:samm:org.example.a:1.0.0#LocalChar".to_string(),
            CharacteristicKind::Trait,
        );
        let property = Property::new("urn:samm:org.example.a:1.0.0#localProp".to_string())
            .with_characteristic(internal_char);
        aspect.add_property(property);

        let validator = cross_model_validator(ModelRegistry::new());
        let errors = validator.validate_references(&aspect);
        assert!(
            errors.is_empty(),
            "internal references must not fail: {errors:?}"
        );
    }

    #[test]
    fn resolved_external_reference_passes() {
        let aspect = make_aspect_a();
        let mut registry = ModelRegistry::new();
        registry.insert(ResolvedElement::new(
            "urn:samm:org.example.b:1.0.0#TextChar",
            ResolvedElementKind::Characteristic,
            "urn:samm:org.example.b:1.0.0",
        ));

        let validator = cross_model_validator(registry);
        let errors = validator.validate_references(&aspect);
        assert!(
            errors.is_empty(),
            "registered external references should resolve cleanly: {errors:?}"
        );
    }

    #[test]
    fn type_mismatch_property_referencing_characteristic_urn() {
        let mut aspect = Aspect::new("urn:samm:org.example.a:1.0.0#AspectA".to_string());
        let prop = Property::new("urn:samm:org.example.a:1.0.0#alias".to_string())
            .extends("urn:samm:org.example.b:1.0.0#NotAProperty".to_string());
        aspect.add_property(prop);

        let mut registry = ModelRegistry::new();
        registry.insert(ResolvedElement::new(
            "urn:samm:org.example.b:1.0.0#NotAProperty",
            ResolvedElementKind::Characteristic,
            "urn:samm:org.example.b:1.0.0",
        ));

        let validator = cross_model_validator(registry);
        let errors = validator.validate_references(&aspect);
        assert_eq!(errors.len(), 1, "expected one type-mismatch: {errors:?}");
        match &errors[0] {
            CrossModelError::TypeMismatch {
                expected, got, ..
            } => {
                assert_eq!(*expected, ResolvedElementKind::Property);
                assert_eq!(*got, ResolvedElementKind::Characteristic);
            }
            other => panic!("expected TypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn empty_model_has_no_errors() {
        let aspect = Aspect::new("urn:samm:org.example.a:1.0.0#Empty".to_string());
        let validator = cross_model_validator(ModelRegistry::new());
        assert!(validator.validate_references(&aspect).is_empty());
    }

    #[test]
    fn multi_model_graph_a_to_b_to_c_resolves() {
        // Aspect A references a Characteristic in model B,
        // which itself references an Entity in model C via SingleEntity.
        let entity_c_urn = "urn:samm:org.example.c:1.0.0#PointEntity".to_string();
        let inner_char = Characteristic::new(
            "urn:samm:org.example.b:1.0.0#PointChar".to_string(),
            CharacteristicKind::SingleEntity {
                entity_type: entity_c_urn.clone(),
            },
        );
        let mut aspect = Aspect::new("urn:samm:org.example.a:1.0.0#AspectA".to_string());
        aspect.add_property(
            Property::new("urn:samm:org.example.a:1.0.0#point".to_string())
                .with_characteristic(inner_char),
        );

        let mut registry = ModelRegistry::new();
        registry.insert(ResolvedElement::new(
            "urn:samm:org.example.b:1.0.0#PointChar",
            ResolvedElementKind::Characteristic,
            "urn:samm:org.example.b:1.0.0",
        ));
        registry.insert(ResolvedElement::new(
            entity_c_urn,
            ResolvedElementKind::Entity,
            "urn:samm:org.example.c:1.0.0",
        ));

        let validator = cross_model_validator(registry);
        let errors = validator.validate_references(&aspect);
        assert!(
            errors.is_empty(),
            "A→B→C chain must resolve when every hop is registered: {errors:?}"
        );
    }

    #[test]
    fn primitive_xsd_data_type_does_not_trigger_resolution() {
        let mut aspect = Aspect::new("urn:samm:org.example.a:1.0.0#AspectA".to_string());
        let char_local = Characteristic::new(
            "urn:samm:org.example.a:1.0.0#TextChar".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
        aspect.add_property(
            Property::new("urn:samm:org.example.a:1.0.0#name".to_string())
                .with_characteristic(char_local),
        );

        let validator = cross_model_validator(ModelRegistry::new());
        assert!(validator.validate_references(&aspect).is_empty());
    }

    #[test]
    fn model_urn_strips_local_name() {
        assert_eq!(
            model_urn("urn:samm:org.example.a:1.0.0#AspectA"),
            "urn:samm:org.example.a:1.0.0"
        );
        assert_eq!(model_urn("urn:samm:no:fragment"), "urn:samm:no:fragment");
    }

    #[test]
    fn register_aspect_populates_every_element() {
        let mut aspect = Aspect::new("urn:samm:org.example.b:1.0.0#AspectB".to_string());
        let characteristic = Characteristic::new(
            "urn:samm:org.example.b:1.0.0#Q".to_string(),
            CharacteristicKind::Trait,
        );
        aspect.add_property(
            Property::new("urn:samm:org.example.b:1.0.0#q".to_string())
                .with_characteristic(characteristic),
        );

        let mut registry = ModelRegistry::new();
        registry.register_aspect(&aspect);

        assert!(registry
            .get("urn:samm:org.example.b:1.0.0#AspectB")
            .map(|e| e.kind == ResolvedElementKind::Aspect)
            .unwrap_or(false));
        assert!(registry
            .get("urn:samm:org.example.b:1.0.0#q")
            .map(|e| e.kind == ResolvedElementKind::Property)
            .unwrap_or(false));
        assert!(registry
            .get("urn:samm:org.example.b:1.0.0#Q")
            .map(|e| e.kind == ResolvedElementKind::Characteristic)
            .unwrap_or(false));
    }
}
