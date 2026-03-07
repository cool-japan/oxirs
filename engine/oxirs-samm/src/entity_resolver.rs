/// SAMM entity resolution.
///
/// Resolves SAMM entity definitions from URN identifiers, traverses
/// `extends` hierarchies, flattens inherited properties, detects circular
/// references, and provides a registry for lookup and comparison.
use std::collections::{HashMap, HashSet};

// ── Entity property ──────────────────────────────────────────────────────────

/// A property belonging to a SAMM entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityProperty {
    /// Property name (e.g. `"temperature"`).
    pub name: String,
    /// XSD or custom data type (e.g. `"xsd:float"`).
    pub data_type: String,
    /// Whether the property is optional.
    pub optional: bool,
    /// Human-readable description.
    pub description: String,
}

impl EntityProperty {
    /// Create a new required property.
    pub fn new(
        name: impl Into<String>,
        data_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            data_type: data_type.into(),
            optional: false,
            description: description.into(),
        }
    }

    /// Create a new optional property.
    pub fn optional(
        name: impl Into<String>,
        data_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            data_type: data_type.into(),
            optional: true,
            description: description.into(),
        }
    }
}

// ── Entity definition ────────────────────────────────────────────────────────

/// A SAMM entity definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityDefinition {
    /// URN identifier (e.g. `"urn:samm:org.example:1.0.0#Vehicle"`).
    pub urn: String,
    /// Short entity name.
    pub name: String,
    /// Human-readable description.
    pub description: Option<String>,
    /// URN of the parent entity this entity extends (if any).
    pub extends: Option<String>,
    /// Whether this entity is abstract (cannot be instantiated).
    pub is_abstract: bool,
    /// Properties declared directly on this entity (not inherited).
    pub properties: Vec<EntityProperty>,
}

impl EntityDefinition {
    /// Create a new concrete entity definition.
    pub fn new(urn: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            urn: urn.into(),
            name: name.into(),
            description: None,
            extends: None,
            is_abstract: false,
            properties: Vec::new(),
        }
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set parent entity URN.
    pub fn with_extends(mut self, parent_urn: impl Into<String>) -> Self {
        self.extends = Some(parent_urn.into());
        self
    }

    /// Mark this entity as abstract.
    pub fn as_abstract(mut self) -> Self {
        self.is_abstract = true;
        self
    }

    /// Add a property.
    pub fn with_property(mut self, prop: EntityProperty) -> Self {
        self.properties.push(prop);
        self
    }
}

// ── Error type ───────────────────────────────────────────────────────────────

/// Errors that can occur during entity resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityResolverError {
    /// The requested entity was not found in the registry.
    NotFound(String),
    /// A circular reference was detected in the `extends` chain.
    CircularReference {
        /// The URNs forming the cycle.
        cycle: Vec<String>,
    },
    /// Attempted to instantiate an abstract entity.
    AbstractEntity(String),
    /// Duplicate entity URN.
    DuplicateUrn(String),
    /// Generic resolution error.
    ResolutionError(String),
}

impl std::fmt::Display for EntityResolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(urn) => write!(f, "entity not found: {urn}"),
            Self::CircularReference { cycle } => {
                write!(f, "circular reference: {}", cycle.join(" -> "))
            }
            Self::AbstractEntity(urn) => {
                write!(f, "cannot instantiate abstract entity: {urn}")
            }
            Self::DuplicateUrn(urn) => write!(f, "duplicate entity URN: {urn}"),
            Self::ResolutionError(msg) => write!(f, "resolution error: {msg}"),
        }
    }
}

impl std::error::Error for EntityResolverError {}

// ── Property diff ────────────────────────────────────────────────────────────

/// Describes a single difference between two entity definitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyDiff {
    /// Property exists only in the left entity.
    OnlyInLeft(String),
    /// Property exists only in the right entity.
    OnlyInRight(String),
    /// Property exists in both but differs.
    Modified {
        /// Property name.
        name: String,
        /// Description of what differs.
        difference: String,
    },
}

/// Result of comparing two entities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityComparison {
    /// URN of the left entity.
    pub left_urn: String,
    /// URN of the right entity.
    pub right_urn: String,
    /// `true` when both entities are structurally equal.
    pub is_equal: bool,
    /// List of differences (empty when equal).
    pub diffs: Vec<PropertyDiff>,
}

// ── Flattened entity ─────────────────────────────────────────────────────────

/// An entity with all inherited properties merged in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlattenedEntity {
    /// The URN of the entity.
    pub urn: String,
    /// The entity name.
    pub name: String,
    /// All properties (own + inherited), earliest ancestor first.
    pub properties: Vec<EntityProperty>,
    /// The extends chain (from root ancestor to this entity).
    pub hierarchy: Vec<String>,
    /// Whether this entity is abstract.
    pub is_abstract: bool,
}

// ── Entity registry / resolver ───────────────────────────────────────────────

/// Registry and resolver for SAMM entity definitions.
pub struct EntityResolver {
    entities: HashMap<String, EntityDefinition>,
}

impl EntityResolver {
    /// Create an empty resolver.
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
        }
    }

    /// Register an entity definition.
    pub fn register(&mut self, entity: EntityDefinition) -> Result<(), EntityResolverError> {
        if self.entities.contains_key(&entity.urn) {
            return Err(EntityResolverError::DuplicateUrn(entity.urn.clone()));
        }
        self.entities.insert(entity.urn.clone(), entity);
        Ok(())
    }

    /// Look up an entity by URN.
    pub fn get(&self, urn: &str) -> Option<&EntityDefinition> {
        self.entities.get(urn)
    }

    /// Remove an entity from the registry.
    pub fn remove(&mut self, urn: &str) -> Option<EntityDefinition> {
        self.entities.remove(urn)
    }

    /// List all registered entity URNs.
    pub fn list_urns(&self) -> Vec<&str> {
        self.entities.keys().map(|k| k.as_str()).collect()
    }

    /// Return the number of registered entities.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Returns `true` if no entities are registered.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    // ── Hierarchy traversal ──────────────────────────────────────────────────

    /// Walk the `extends` chain from `urn` up to the root ancestor, returning
    /// the chain as a list of URNs (starting from the root).
    pub fn hierarchy(&self, urn: &str) -> Result<Vec<String>, EntityResolverError> {
        let mut chain = Vec::new();
        let mut visited = HashSet::new();
        let mut current = urn.to_string();

        loop {
            if !visited.insert(current.clone()) {
                chain.push(current);
                return Err(EntityResolverError::CircularReference { cycle: chain });
            }
            let entity = self
                .entities
                .get(&current)
                .ok_or_else(|| EntityResolverError::NotFound(current.clone()))?;
            chain.push(current.clone());
            match &entity.extends {
                Some(parent_urn) => {
                    current = parent_urn.clone();
                }
                None => break,
            }
        }

        chain.reverse();
        Ok(chain)
    }

    /// Check whether an entity is abstract.
    pub fn is_abstract(&self, urn: &str) -> Result<bool, EntityResolverError> {
        let entity = self
            .entities
            .get(urn)
            .ok_or_else(|| EntityResolverError::NotFound(urn.to_string()))?;
        Ok(entity.is_abstract)
    }

    /// Detect circular references starting from `urn`.
    pub fn has_circular_reference(&self, urn: &str) -> bool {
        self.hierarchy(urn).is_err()
    }

    // ── Property flattening ──────────────────────────────────────────────────

    /// Resolve all properties (own + inherited) for an entity, returning a
    /// `FlattenedEntity`.  Properties from parent entities come first.
    pub fn flatten(&self, urn: &str) -> Result<FlattenedEntity, EntityResolverError> {
        let hierarchy = self.hierarchy(urn)?;
        let mut all_properties = Vec::new();
        let mut seen_names = HashSet::new();

        for ancestor_urn in &hierarchy {
            let entity = self
                .entities
                .get(ancestor_urn.as_str())
                .ok_or_else(|| EntityResolverError::NotFound(ancestor_urn.clone()))?;
            for prop in &entity.properties {
                if seen_names.insert(prop.name.clone()) {
                    all_properties.push(prop.clone());
                }
            }
        }

        let entity = self
            .entities
            .get(urn)
            .ok_or_else(|| EntityResolverError::NotFound(urn.to_string()))?;

        Ok(FlattenedEntity {
            urn: urn.to_string(),
            name: entity.name.clone(),
            properties: all_properties,
            hierarchy,
            is_abstract: entity.is_abstract,
        })
    }

    /// Resolve only the directly declared properties of an entity (no inheritance).
    pub fn own_properties(&self, urn: &str) -> Result<Vec<EntityProperty>, EntityResolverError> {
        let entity = self
            .entities
            .get(urn)
            .ok_or_else(|| EntityResolverError::NotFound(urn.to_string()))?;
        Ok(entity.properties.clone())
    }

    // ── Extends resolution ───────────────────────────────────────────────────

    /// Return the immediate parent entity URN, if any.
    pub fn parent_urn(&self, urn: &str) -> Result<Option<String>, EntityResolverError> {
        let entity = self
            .entities
            .get(urn)
            .ok_or_else(|| EntityResolverError::NotFound(urn.to_string()))?;
        Ok(entity.extends.clone())
    }

    /// Return all entities that directly extend the given entity.
    pub fn children(&self, parent_urn: &str) -> Vec<&EntityDefinition> {
        self.entities
            .values()
            .filter(|e| e.extends.as_deref() == Some(parent_urn))
            .collect()
    }

    // ── Entity comparison ────────────────────────────────────────────────────

    /// Compare two entities structurally (based on flattened properties).
    pub fn compare(
        &self,
        left_urn: &str,
        right_urn: &str,
    ) -> Result<EntityComparison, EntityResolverError> {
        let left = self.flatten(left_urn)?;
        let right = self.flatten(right_urn)?;

        let left_map: HashMap<&str, &EntityProperty> = left
            .properties
            .iter()
            .map(|p| (p.name.as_str(), p))
            .collect();
        let right_map: HashMap<&str, &EntityProperty> = right
            .properties
            .iter()
            .map(|p| (p.name.as_str(), p))
            .collect();

        let all_names: HashSet<&str> = left_map.keys().chain(right_map.keys()).copied().collect();

        let mut diffs = Vec::new();
        for name in &all_names {
            match (left_map.get(name), right_map.get(name)) {
                (Some(_), None) => {
                    diffs.push(PropertyDiff::OnlyInLeft(name.to_string()));
                }
                (None, Some(_)) => {
                    diffs.push(PropertyDiff::OnlyInRight(name.to_string()));
                }
                (Some(lp), Some(rp)) => {
                    if lp != rp {
                        let mut desc = Vec::new();
                        if lp.data_type != rp.data_type {
                            desc.push(format!("data_type: {} vs {}", lp.data_type, rp.data_type));
                        }
                        if lp.optional != rp.optional {
                            desc.push(format!("optional: {} vs {}", lp.optional, rp.optional));
                        }
                        if lp.description != rp.description {
                            desc.push("description differs".to_string());
                        }
                        diffs.push(PropertyDiff::Modified {
                            name: name.to_string(),
                            difference: desc.join("; "),
                        });
                    }
                }
                (None, None) => {} // unreachable
            }
        }

        Ok(EntityComparison {
            left_urn: left_urn.to_string(),
            right_urn: right_urn.to_string(),
            is_equal: diffs.is_empty(),
            diffs,
        })
    }
}

impl Default for EntityResolver {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn vehicle_urn() -> String {
        "urn:samm:org.example:1.0.0#Vehicle".to_string()
    }
    fn car_urn() -> String {
        "urn:samm:org.example:1.0.0#Car".to_string()
    }
    fn ev_urn() -> String {
        "urn:samm:org.example:1.0.0#ElectricVehicle".to_string()
    }

    fn vehicle_entity() -> EntityDefinition {
        EntityDefinition::new(vehicle_urn(), "Vehicle")
            .with_description("Base vehicle entity")
            .as_abstract()
            .with_property(EntityProperty::new(
                "vin",
                "xsd:string",
                "Vehicle Identification Number",
            ))
            .with_property(EntityProperty::new(
                "manufacturer",
                "xsd:string",
                "Manufacturer name",
            ))
    }

    fn car_entity() -> EntityDefinition {
        EntityDefinition::new(car_urn(), "Car")
            .with_extends(vehicle_urn())
            .with_property(EntityProperty::new(
                "doors",
                "xsd:integer",
                "Number of doors",
            ))
    }

    fn ev_entity() -> EntityDefinition {
        EntityDefinition::new(ev_urn(), "ElectricVehicle")
            .with_extends(car_urn())
            .with_property(EntityProperty::new(
                "batteryCapacity",
                "xsd:float",
                "Battery capacity in kWh",
            ))
            .with_property(EntityProperty::optional(
                "range",
                "xsd:float",
                "Range in km",
            ))
    }

    fn build_resolver() -> EntityResolver {
        let mut resolver = EntityResolver::new();
        resolver
            .register(vehicle_entity())
            .expect("register vehicle");
        resolver.register(car_entity()).expect("register car");
        resolver.register(ev_entity()).expect("register ev");
        resolver
    }

    // ── Registration and lookup ─────────────────────────────────────────────

    #[test]
    fn test_register_and_get() {
        let mut resolver = EntityResolver::new();
        resolver.register(vehicle_entity()).expect("register");
        let entity = resolver.get(&vehicle_urn());
        assert!(entity.is_some());
        assert_eq!(entity.expect("exists").name, "Vehicle");
    }

    #[test]
    fn test_register_duplicate() {
        let mut resolver = EntityResolver::new();
        resolver.register(vehicle_entity()).expect("first");
        let result = resolver.register(vehicle_entity());
        assert!(result.is_err());
        match result {
            Err(EntityResolverError::DuplicateUrn(u)) => assert_eq!(u, vehicle_urn()),
            other => panic!("expected DuplicateUrn, got {other:?}"),
        }
    }

    #[test]
    fn test_get_nonexistent() {
        let resolver = EntityResolver::new();
        assert!(resolver.get("urn:nonexistent").is_none());
    }

    #[test]
    fn test_remove() {
        let mut resolver = EntityResolver::new();
        resolver.register(vehicle_entity()).expect("register");
        let removed = resolver.remove(&vehicle_urn());
        assert!(removed.is_some());
        assert!(resolver.get(&vehicle_urn()).is_none());
    }

    #[test]
    fn test_list_urns() {
        let resolver = build_resolver();
        let urns = resolver.list_urns();
        assert_eq!(urns.len(), 3);
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut resolver = EntityResolver::new();
        assert!(resolver.is_empty());
        assert_eq!(resolver.len(), 0);
        resolver.register(vehicle_entity()).expect("register");
        assert!(!resolver.is_empty());
        assert_eq!(resolver.len(), 1);
    }

    #[test]
    fn test_default() {
        let resolver = EntityResolver::default();
        assert!(resolver.is_empty());
    }

    // ── Hierarchy traversal ─────────────────────────────────────────────────

    #[test]
    fn test_hierarchy_root() {
        let resolver = build_resolver();
        let hierarchy = resolver.hierarchy(&vehicle_urn()).expect("hierarchy");
        assert_eq!(hierarchy, vec![vehicle_urn()]);
    }

    #[test]
    fn test_hierarchy_one_level() {
        let resolver = build_resolver();
        let hierarchy = resolver.hierarchy(&car_urn()).expect("hierarchy");
        assert_eq!(hierarchy, vec![vehicle_urn(), car_urn()]);
    }

    #[test]
    fn test_hierarchy_two_levels() {
        let resolver = build_resolver();
        let hierarchy = resolver.hierarchy(&ev_urn()).expect("hierarchy");
        assert_eq!(hierarchy, vec![vehicle_urn(), car_urn(), ev_urn()]);
    }

    #[test]
    fn test_hierarchy_not_found() {
        let resolver = EntityResolver::new();
        let result = resolver.hierarchy("urn:unknown");
        assert!(result.is_err());
    }

    // ── Circular reference detection ────────────────────────────────────────

    #[test]
    fn test_circular_reference() {
        let mut resolver = EntityResolver::new();
        resolver
            .register(EntityDefinition::new("urn:A", "A").with_extends("urn:B"))
            .expect("register A");
        resolver
            .register(EntityDefinition::new("urn:B", "B").with_extends("urn:A"))
            .expect("register B");
        assert!(resolver.has_circular_reference("urn:A"));
        let result = resolver.hierarchy("urn:A");
        assert!(result.is_err());
        match result {
            Err(EntityResolverError::CircularReference { cycle }) => {
                assert!(cycle.contains(&"urn:A".to_string()));
                assert!(cycle.contains(&"urn:B".to_string()));
            }
            other => panic!("expected CircularReference, got {other:?}"),
        }
    }

    #[test]
    fn test_no_circular_reference() {
        let resolver = build_resolver();
        assert!(!resolver.has_circular_reference(&ev_urn()));
    }

    // ── Abstract entity detection ───────────────────────────────────────────

    #[test]
    fn test_is_abstract_true() {
        let resolver = build_resolver();
        assert!(resolver.is_abstract(&vehicle_urn()).expect("ok"));
    }

    #[test]
    fn test_is_abstract_false() {
        let resolver = build_resolver();
        assert!(!resolver.is_abstract(&car_urn()).expect("ok"));
    }

    #[test]
    fn test_is_abstract_not_found() {
        let resolver = EntityResolver::new();
        let result = resolver.is_abstract("urn:unknown");
        assert!(result.is_err());
    }

    // ── Property flattening ─────────────────────────────────────────────────

    #[test]
    fn test_flatten_root() {
        let resolver = build_resolver();
        let flat = resolver.flatten(&vehicle_urn()).expect("flatten");
        assert_eq!(flat.properties.len(), 2);
        assert_eq!(flat.hierarchy.len(), 1);
        assert!(flat.is_abstract);
    }

    #[test]
    fn test_flatten_one_level() {
        let resolver = build_resolver();
        let flat = resolver.flatten(&car_urn()).expect("flatten");
        // Vehicle has 2 + Car has 1 = 3
        assert_eq!(flat.properties.len(), 3);
        assert_eq!(flat.properties[0].name, "vin");
        assert_eq!(flat.properties[1].name, "manufacturer");
        assert_eq!(flat.properties[2].name, "doors");
    }

    #[test]
    fn test_flatten_two_levels() {
        let resolver = build_resolver();
        let flat = resolver.flatten(&ev_urn()).expect("flatten");
        // Vehicle(2) + Car(1) + EV(2) = 5
        assert_eq!(flat.properties.len(), 5);
        let names: Vec<&str> = flat.properties.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"vin"));
        assert!(names.contains(&"manufacturer"));
        assert!(names.contains(&"doors"));
        assert!(names.contains(&"batteryCapacity"));
        assert!(names.contains(&"range"));
    }

    #[test]
    fn test_flatten_deduplicates_properties() {
        // If child re-declares a parent property, it should not appear twice.
        let mut resolver = EntityResolver::new();
        resolver
            .register(
                EntityDefinition::new("urn:parent", "Parent").with_property(EntityProperty::new(
                    "name",
                    "xsd:string",
                    "Name",
                )),
            )
            .expect("register parent");
        resolver
            .register(
                EntityDefinition::new("urn:child", "Child")
                    .with_extends("urn:parent")
                    .with_property(EntityProperty::new("name", "xsd:string", "Overridden name")),
            )
            .expect("register child");
        let flat = resolver.flatten("urn:child").expect("flatten");
        // Only one "name" property (from parent, since it was seen first).
        assert_eq!(flat.properties.len(), 1);
    }

    // ── Own properties ──────────────────────────────────────────────────────

    #[test]
    fn test_own_properties() {
        let resolver = build_resolver();
        let props = resolver.own_properties(&car_urn()).expect("own_properties");
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].name, "doors");
    }

    #[test]
    fn test_own_properties_not_found() {
        let resolver = EntityResolver::new();
        let result = resolver.own_properties("urn:unknown");
        assert!(result.is_err());
    }

    // ── Parent URN ──────────────────────────────────────────────────────────

    #[test]
    fn test_parent_urn_some() {
        let resolver = build_resolver();
        let parent = resolver.parent_urn(&car_urn()).expect("ok");
        assert_eq!(parent, Some(vehicle_urn()));
    }

    #[test]
    fn test_parent_urn_none() {
        let resolver = build_resolver();
        let parent = resolver.parent_urn(&vehicle_urn()).expect("ok");
        assert!(parent.is_none());
    }

    #[test]
    fn test_parent_urn_not_found() {
        let resolver = EntityResolver::new();
        assert!(resolver.parent_urn("urn:unknown").is_err());
    }

    // ── Children ────────────────────────────────────────────────────────────

    #[test]
    fn test_children() {
        let resolver = build_resolver();
        let children = resolver.children(&vehicle_urn());
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].name, "Car");
    }

    #[test]
    fn test_children_none() {
        let resolver = build_resolver();
        let children = resolver.children(&ev_urn());
        assert!(children.is_empty());
    }

    // ── Entity comparison ───────────────────────────────────────────────────

    #[test]
    fn test_compare_same_entity() {
        let resolver = build_resolver();
        let cmp = resolver.compare(&car_urn(), &car_urn()).expect("compare");
        assert!(cmp.is_equal);
        assert!(cmp.diffs.is_empty());
    }

    #[test]
    fn test_compare_different_entities() {
        let resolver = build_resolver();
        let cmp = resolver.compare(&car_urn(), &ev_urn()).expect("compare");
        assert!(!cmp.is_equal);
        // EV has batteryCapacity and range that Car does not have.
        assert!(!cmp.diffs.is_empty());
    }

    #[test]
    fn test_compare_property_diff_types() {
        let mut resolver = EntityResolver::new();
        resolver
            .register(
                EntityDefinition::new("urn:a", "A").with_property(EntityProperty::new(
                    "x",
                    "xsd:string",
                    "X",
                )),
            )
            .expect("register a");
        resolver
            .register(
                EntityDefinition::new("urn:b", "B").with_property(EntityProperty::new(
                    "x",
                    "xsd:integer",
                    "X",
                )),
            )
            .expect("register b");
        let cmp = resolver.compare("urn:a", "urn:b").expect("compare");
        assert!(!cmp.is_equal);
        let modified = cmp
            .diffs
            .iter()
            .find(|d| matches!(d, PropertyDiff::Modified { .. }));
        assert!(modified.is_some());
    }

    #[test]
    fn test_compare_only_in_left() {
        let mut resolver = EntityResolver::new();
        resolver
            .register(
                EntityDefinition::new("urn:a", "A")
                    .with_property(EntityProperty::new("x", "xsd:string", "X"))
                    .with_property(EntityProperty::new("y", "xsd:string", "Y")),
            )
            .expect("register a");
        resolver
            .register(EntityDefinition::new("urn:b", "B"))
            .expect("register b");
        let cmp = resolver.compare("urn:a", "urn:b").expect("compare");
        let left_only: Vec<_> = cmp
            .diffs
            .iter()
            .filter(|d| matches!(d, PropertyDiff::OnlyInLeft(_)))
            .collect();
        assert_eq!(left_only.len(), 2);
    }

    #[test]
    fn test_compare_only_in_right() {
        let mut resolver = EntityResolver::new();
        resolver
            .register(EntityDefinition::new("urn:a", "A"))
            .expect("register a");
        resolver
            .register(
                EntityDefinition::new("urn:b", "B").with_property(EntityProperty::new(
                    "z",
                    "xsd:float",
                    "Z",
                )),
            )
            .expect("register b");
        let cmp = resolver.compare("urn:a", "urn:b").expect("compare");
        let right_only: Vec<_> = cmp
            .diffs
            .iter()
            .filter(|d| matches!(d, PropertyDiff::OnlyInRight(_)))
            .collect();
        assert_eq!(right_only.len(), 1);
    }

    // ── EntityProperty ──────────────────────────────────────────────────────

    #[test]
    fn test_entity_property_new() {
        let p = EntityProperty::new("temp", "xsd:float", "Temperature");
        assert_eq!(p.name, "temp");
        assert_eq!(p.data_type, "xsd:float");
        assert!(!p.optional);
    }

    #[test]
    fn test_entity_property_optional() {
        let p = EntityProperty::optional("tag", "xsd:string", "A tag");
        assert!(p.optional);
    }

    // ── EntityDefinition builder ────────────────────────────────────────────

    #[test]
    fn test_entity_definition_builder() {
        let e = EntityDefinition::new("urn:test", "Test")
            .with_description("A test entity")
            .with_extends("urn:parent")
            .as_abstract()
            .with_property(EntityProperty::new("x", "xsd:int", "X"));
        assert_eq!(e.name, "Test");
        assert_eq!(e.description, Some("A test entity".into()));
        assert_eq!(e.extends, Some("urn:parent".into()));
        assert!(e.is_abstract);
        assert_eq!(e.properties.len(), 1);
    }

    // ── Error display ───────────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let err = EntityResolverError::NotFound("urn:x".into());
        assert!(err.to_string().contains("urn:x"));

        let err2 = EntityResolverError::CircularReference {
            cycle: vec!["urn:a".into(), "urn:b".into()],
        };
        assert!(err2.to_string().contains("urn:a -> urn:b"));

        let err3 = EntityResolverError::AbstractEntity("urn:abs".into());
        assert!(err3.to_string().contains("abstract"));

        let err4 = EntityResolverError::DuplicateUrn("urn:dup".into());
        assert!(err4.to_string().contains("duplicate"));

        let err5 = EntityResolverError::ResolutionError("oops".into());
        assert!(err5.to_string().contains("oops"));
    }

    // ── FlattenedEntity ─────────────────────────────────────────────────────

    #[test]
    fn test_flattened_entity_hierarchy() {
        let resolver = build_resolver();
        let flat = resolver.flatten(&ev_urn()).expect("flatten");
        assert_eq!(flat.hierarchy.len(), 3);
        assert_eq!(flat.hierarchy[0], vehicle_urn());
        assert_eq!(flat.hierarchy[2], ev_urn());
    }

    #[test]
    fn test_flattened_entity_name() {
        let resolver = build_resolver();
        let flat = resolver.flatten(&ev_urn()).expect("flatten");
        assert_eq!(flat.name, "ElectricVehicle");
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_entity_with_no_properties() {
        let mut resolver = EntityResolver::new();
        resolver
            .register(EntityDefinition::new("urn:empty", "Empty"))
            .expect("register");
        let flat = resolver.flatten("urn:empty").expect("flatten");
        assert!(flat.properties.is_empty());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut resolver = EntityResolver::new();
        assert!(resolver.remove("urn:nope").is_none());
    }

    #[test]
    fn test_children_nonexistent_parent() {
        let resolver = EntityResolver::new();
        let children = resolver.children("urn:nope");
        assert!(children.is_empty());
    }
}
