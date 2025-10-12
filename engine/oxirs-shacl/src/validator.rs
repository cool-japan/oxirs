//! SHACL Validator implementation

use crate::constraints::Constraint;
use crate::paths::PropertyPath;
use crate::report::ValidationReport;
use crate::shape_import;
use crate::shapes;
use crate::targets::Target;
use crate::validation;
use crate::{ConstraintComponentId, Result, Severity, ShaclError, ShapeId, ValidationConfig};
use crate::{Shape, ShapeMetadata, ShapeType};

use indexmap::IndexMap;
use oxirs_core::model::{Literal, Term};
use oxirs_core::Store;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
#[derive(Debug)]

pub struct Validator {
    /// All loaded shapes indexed by ID
    shapes: IndexMap<ShapeId, Shape>,

    /// Shape dependency graph for optimization
    shape_dependencies: petgraph::Graph<ShapeId, ()>,

    /// Target cache for performance
    target_cache: HashMap<String, Vec<Term>>,

    /// Configuration for validation
    config: ValidationConfig,

    /// Shape import manager for handling external references
    import_manager: shape_import::ShapeImportManager,
}

impl Validator {
    /// Create a new validator with default configuration
    pub fn new() -> Self {
        Self {
            shapes: IndexMap::new(),
            shape_dependencies: petgraph::Graph::new(),
            target_cache: HashMap::new(),
            config: ValidationConfig::default(),
            import_manager: shape_import::ShapeImportManager::with_default_config(),
        }
    }

    /// Create a new validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self {
            shapes: IndexMap::new(),
            shape_dependencies: petgraph::Graph::new(),
            target_cache: HashMap::new(),
            config,
            import_manager: shape_import::ShapeImportManager::with_default_config(),
        }
    }

    /// Create a new validator with custom import configuration
    pub fn with_import_config(
        config: ValidationConfig,
        import_config: shape_import::ShapeImportConfig,
    ) -> Self {
        Self {
            shapes: IndexMap::new(),
            shape_dependencies: petgraph::Graph::new(),
            target_cache: HashMap::new(),
            config,
            import_manager: shape_import::ShapeImportManager::new(import_config),
        }
    }

    /// Add a shape to the validator
    pub fn add_shape(&mut self, shape: Shape) -> Result<()> {
        let shape_id = shape.id.clone();

        // Validate the shape itself
        self.validate_shape(&shape)?;

        // Add to shape collection
        self.shapes.insert(shape_id.clone(), shape);

        // Update dependency graph
        self.update_shape_dependencies(shape_id)?;

        // Clear target cache as it may be invalidated
        self.target_cache.clear();

        Ok(())
    }

    /// Remove a shape from the validator
    pub fn remove_shape(&mut self, shape_id: &ShapeId) -> Option<Shape> {
        let removed = self.shapes.shift_remove(shape_id);
        if removed.is_some() {
            self.target_cache.clear();

            // Remove from dependency graph
            use petgraph::visit::IntoNodeReferences;
            let mut node_to_remove = None;

            // Find the node index for this shape
            for (node_idx, node_shape_id) in self.shape_dependencies.node_references() {
                if node_shape_id == shape_id {
                    node_to_remove = Some(node_idx);
                    break;
                }
            }

            // Remove the node if found
            if let Some(node_idx) = node_to_remove {
                self.shape_dependencies.remove_node(node_idx);
            }
        }
        removed
    }

    /// Get all shapes
    pub fn shapes(&self) -> &IndexMap<ShapeId, Shape> {
        &self.shapes
    }

    /// Get a specific shape by ID
    pub fn get_shape(&self, shape_id: &ShapeId) -> Option<&Shape> {
        self.shapes.get(shape_id)
    }

    /// Load shapes from an RDF graph in a store
    pub fn load_shapes_from_store(
        &mut self,
        store: &dyn Store,
        graph_name: Option<&str>,
    ) -> Result<usize> {
        let mut parser = shapes::ShapeParser::new();
        let shapes = parser.parse_shapes_from_store(store, graph_name)?;

        let count = shapes.len();
        for shape in shapes {
            self.add_shape(shape)?;
        }

        Ok(count)
    }

    /// Load shapes from RDF data
    pub fn load_shapes_from_rdf(
        &mut self,
        rdf_data: &str,
        format: &str,
        base_iri: Option<&str>,
    ) -> Result<usize> {
        let mut parser = shapes::ShapeParser::new();
        let shapes = parser.parse_shapes_from_rdf(rdf_data, format, base_iri)?;

        let count = shapes.len();
        for shape in shapes {
            self.add_shape(shape)?;
        }

        Ok(count)
    }

    /// Load shapes from RDF data with automatic import processing
    ///
    /// This method processes import directives (owl:imports, sh:include, sh:imports)
    /// found in the RDF data and recursively loads referenced shapes.
    pub fn load_shapes_with_imports(
        &mut self,
        rdf_data: &str,
        format: &str,
        base_iri: Option<&str>,
    ) -> Result<shape_import::ImportResult> {
        // First, load the primary shapes normally
        let mut parser = shapes::ShapeParser::new();
        let primary_shapes = parser.parse_shapes_from_rdf(rdf_data, format, base_iri)?;

        let mut total_count = primary_shapes.len();
        let mut warnings = Vec::new();
        let mut dependency_chain = Vec::new();

        // Add primary shapes
        for shape in primary_shapes {
            self.add_shape(shape)?;
        }

        // Extract and process import directives
        let import_directives = self.import_manager.extract_import_directives(rdf_data)?;

        for directive in import_directives {
            match self.import_manager.import_shapes(&directive, 0) {
                Ok(import_result) => {
                    // Add imported shapes
                    for shape in import_result.shapes {
                        if let Err(e) = self.add_shape(shape) {
                            warnings.push(format!("Failed to add imported shape: {e}"));
                        } else {
                            total_count += 1;
                        }
                    }

                    // Merge warnings and dependency chain
                    warnings.extend(import_result.warnings);
                    dependency_chain.extend(import_result.dependency_chain);
                }
                Err(e) => {
                    warnings.push(format!(
                        "Failed to import from {}: {}",
                        directive.source_iri, e
                    ));
                }
            }
        }

        // Create comprehensive import result
        Ok(shape_import::ImportResult {
            shapes: self.shapes.values().cloned().collect(),
            metadata: shape_import::ImportMetadata {
                source_iri: base_iri.unwrap_or("inline").to_string(),
                shape_count: total_count,
                imported_at: chrono::Utc::now().to_rfc3339(),
                import_depth: 0,
                content_type: Some(format.to_string()),
                content_size: rdf_data.len(),
                content_hash: {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    rdf_data.hash(&mut hasher);
                    format!("{:x}", hasher.finish())
                },
            },
            warnings,
            dependency_chain,
        })
    }

    /// Load shapes from external source using import directive
    pub fn load_shapes_from_external(
        &mut self,
        directive: &shape_import::ImportDirective,
    ) -> Result<shape_import::ImportResult> {
        let import_result = self.import_manager.import_shapes(directive, 0)?;

        // Add imported shapes to validator
        let mut _added_count = 0;
        let mut warnings = import_result.warnings.clone();

        for shape in &import_result.shapes {
            match self.add_shape(shape.clone()) {
                Ok(()) => _added_count += 1,
                Err(e) => warnings.push(format!("Failed to add shape {}: {}", shape.id, e)),
            }
        }

        Ok(shape_import::ImportResult {
            shapes: import_result.shapes,
            metadata: import_result.metadata,
            warnings,
            dependency_chain: import_result.dependency_chain,
        })
    }

    /// Load shapes from URL with automatic format detection
    pub fn load_shapes_from_url(
        &mut self,
        url: &str,
        import_type: Option<shape_import::ImportType>,
    ) -> Result<shape_import::ImportResult> {
        let directive = shape_import::ImportDirective {
            source_iri: url.to_string(),
            target_namespace: None,
            specific_shapes: None,
            import_type: import_type.unwrap_or(shape_import::ImportType::Include),
            format_hint: None,
        };

        self.load_shapes_from_external(&directive)
    }

    /// Load specific shapes from external source
    pub fn load_specific_shapes_from_external(
        &mut self,
        source_iri: &str,
        shape_iris: Vec<String>,
        target_namespace: Option<String>,
    ) -> Result<shape_import::ImportResult> {
        let directive = shape_import::ImportDirective {
            source_iri: source_iri.to_string(),
            target_namespace,
            specific_shapes: Some(shape_iris),
            import_type: shape_import::ImportType::Selective,
            format_hint: None,
        };

        self.load_shapes_from_external(&directive)
    }

    /// Resolve and load external references found in loaded shapes
    pub fn resolve_external_references(&mut self) -> Result<Vec<shape_import::ImportResult>> {
        let mut current_shapes: Vec<Shape> = self.shapes.values().cloned().collect();
        self.import_manager
            .resolve_external_references(&mut current_shapes)
    }

    /// Get import statistics
    pub fn get_import_statistics(&self) -> &shape_import::ImportStatistics {
        self.import_manager.get_statistics()
    }

    /// Clear import cache
    pub fn clear_import_cache(&mut self) {
        self.import_manager.clear_cache();
    }

    /// Check for circular dependencies in imported shapes
    pub fn check_import_dependencies(&self) -> Result<()> {
        self.import_manager.check_circular_dependencies()
    }

    /// Configure import security settings
    pub fn configure_import_security(
        &mut self,
        allow_http: bool,
        allow_file: bool,
        max_resource_size: usize,
    ) {
        self.import_manager =
            shape_import::ShapeImportManager::new(shape_import::ShapeImportConfig {
                allow_http,
                allow_file,
                max_resource_size,
                ..shape_import::ShapeImportConfig::default()
            });
    }

    /// Validate data in a store against all loaded shapes
    pub fn validate_store(
        &self,
        store: &dyn Store,
        config: Option<ValidationConfig>,
    ) -> Result<ValidationReport> {
        let config = config.unwrap_or_else(|| self.config.clone());
        let mut engine = validation::ValidationEngine::new(&self.shapes, config);
        engine.validate_store(store)
    }

    /// Validate specific nodes against a specific shape
    pub fn validate_nodes(
        &self,
        store: &dyn Store,
        shape_id: &ShapeId,
        nodes: &[Term],
        config: Option<ValidationConfig>,
    ) -> Result<ValidationReport> {
        let shape = self
            .get_shape(shape_id)
            .ok_or_else(|| ShaclError::ValidationEngine(format!("Shape not found: {shape_id}")))?;

        let config = config.unwrap_or_else(|| self.config.clone());
        let mut engine = validation::ValidationEngine::new(&self.shapes, config);
        engine.validate_nodes(store, shape, nodes)
    }

    /// Validate a single node against a specific shape
    pub fn validate_node(
        &self,
        store: &dyn Store,
        shape_id: &ShapeId,
        node: &Term,
        config: Option<ValidationConfig>,
    ) -> Result<ValidationReport> {
        self.validate_nodes(store, shape_id, std::slice::from_ref(node), config)
    }

    /// Comprehensive shape validation to ensure shapes graphs are themselves valid
    fn validate_shape(&self, shape: &Shape) -> Result<()> {
        // 1. Basic shape type validation
        self.validate_shape_type(shape)?;

        // 2. Validate shape targets
        self.validate_shape_targets(shape)?;

        // 3. Validate property paths
        self.validate_shape_paths(shape)?;

        // 4. Validate individual constraints
        self.validate_shape_constraints(shape)?;

        // 5. Validate constraint combinations
        self.validate_constraint_combinations(shape)?;

        // 6. Validate metadata
        self.validate_shape_metadata(shape)?;

        // 7. Validate inheritance references
        self.validate_inheritance_references(shape)?;

        // 8. Enhanced SHACL specification compliance
        self.validate_shacl_compliance(shape)?;

        // 9. Validate constraint parameter restrictions
        self.validate_constraint_parameters(shape)?;

        // 10. Validate logical constraint consistency
        self.validate_logical_consistency(shape)?;

        Ok(())
    }

    /// Validate shape type requirements
    fn validate_shape_type(&self, shape: &Shape) -> Result<()> {
        match shape.shape_type {
            ShapeType::PropertyShape => {
                if shape.path.is_none() {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Property shape '{}' must have a property path",
                        shape.id
                    )));
                }

                // Property shapes can have targets, though it's more common for node shapes
                // We'll allow all target types but log a warning for unusual combinations
                if !shape.targets.is_empty() {
                    for target in &shape.targets {
                        if let Target::Class(_) = target {
                            tracing::debug!(
                                "Property shape '{}' has a class target, which is valid but unusual",
                                shape.id
                            );
                        }
                        // All target types are valid for property shapes
                    }
                }
            }
            ShapeType::NodeShape => {
                if shape.path.is_some() {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Node shape '{}' should not have a property path",
                        shape.id
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate shape targets
    fn validate_shape_targets(&self, shape: &Shape) -> Result<()> {
        for target in &shape.targets {
            match target {
                Target::Class(class_iri) => {
                    // Validate that the class IRI is a valid IRI
                    self.validate_iri_format(class_iri.as_str())?;
                }
                Target::Node(node_term) => {
                    // Validate that the node term is valid
                    if let Term::NamedNode(node_iri) = node_term {
                        self.validate_iri_format(node_iri.as_str())?;
                    }
                    // Other term types (blank nodes, literals) are also valid targets
                }
                Target::SubjectsOf(property_iri) => {
                    // Validate that the property IRI is a valid IRI
                    self.validate_iri_format(property_iri.as_str())?;
                }
                Target::ObjectsOf(property_iri) => {
                    // Validate that the property IRI is a valid IRI
                    self.validate_iri_format(property_iri.as_str())?;
                }
                Target::Sparql(sparql_target) => {
                    // Validate SPARQL query syntax
                    self.validate_sparql_query(&sparql_target.query)?;
                }
                Target::Implicit(class_iri) => {
                    // Validate that the implicit class IRI is a valid IRI
                    self.validate_iri_format(class_iri.as_str())?;
                }
                Target::Union(union_target) => {
                    // Recursively validate all targets in the union
                    for nested_target in &union_target.targets {
                        self.validate_shape_targets(&Shape {
                            id: shape.id.clone(),
                            shape_type: shape.shape_type.clone(),
                            targets: vec![nested_target.clone()],
                            constraints: indexmap::IndexMap::new(),
                            path: None,
                            deactivated: false,
                            label: None,
                            description: None,
                            groups: vec![],
                            order: None,
                            severity: shape.severity,
                            messages: indexmap::IndexMap::new(),
                            extends: vec![],
                            priority: None,
                            metadata: ShapeMetadata::default(),
                        })?;
                    }
                }
                Target::Intersection(intersection_target) => {
                    // Recursively validate all targets in the intersection
                    for nested_target in &intersection_target.targets {
                        self.validate_shape_targets(&Shape {
                            id: shape.id.clone(),
                            shape_type: shape.shape_type.clone(),
                            targets: vec![nested_target.clone()],
                            constraints: indexmap::IndexMap::new(),
                            path: None,
                            deactivated: false,
                            label: None,
                            description: None,
                            groups: vec![],
                            order: None,
                            severity: shape.severity,
                            messages: indexmap::IndexMap::new(),
                            extends: vec![],
                            priority: None,
                            metadata: ShapeMetadata::default(),
                        })?;
                    }
                }
                Target::Difference(difference_target) => {
                    // Validate both primary and exclusion targets
                    self.validate_shape_targets(&Shape {
                        id: shape.id.clone(),
                        shape_type: shape.shape_type.clone(),
                        targets: vec![(*difference_target.primary_target).clone()],
                        constraints: indexmap::IndexMap::new(),
                        path: None,
                        deactivated: false,
                        label: None,
                        description: None,
                        groups: vec![],
                        order: None,
                        severity: shape.severity,
                        messages: indexmap::IndexMap::new(),
                        extends: vec![],
                        priority: None,
                        metadata: ShapeMetadata::default(),
                    })?;
                    self.validate_shape_targets(&Shape {
                        id: shape.id.clone(),
                        shape_type: shape.shape_type.clone(),
                        targets: vec![(*difference_target.exclusion_target).clone()],
                        constraints: indexmap::IndexMap::new(),
                        path: None,
                        deactivated: false,
                        label: None,
                        description: None,
                        groups: vec![],
                        order: None,
                        severity: shape.severity,
                        messages: indexmap::IndexMap::new(),
                        extends: vec![],
                        priority: None,
                        metadata: ShapeMetadata::default(),
                    })?;
                }
                Target::Conditional(conditional_target) => {
                    // Validate the base target
                    self.validate_shape_targets(&Shape {
                        id: shape.id.clone(),
                        shape_type: shape.shape_type.clone(),
                        targets: vec![(*conditional_target.base_target).clone()],
                        constraints: indexmap::IndexMap::new(),
                        path: None,
                        deactivated: false,
                        label: None,
                        description: None,
                        groups: vec![],
                        order: None,
                        severity: shape.severity,
                        messages: indexmap::IndexMap::new(),
                        extends: vec![],
                        priority: None,
                        metadata: ShapeMetadata::default(),
                    })?;
                }
                Target::Hierarchical(hierarchical_target) => {
                    // Validate the root target
                    self.validate_shape_targets(&Shape {
                        id: shape.id.clone(),
                        shape_type: shape.shape_type.clone(),
                        targets: vec![(*hierarchical_target.root_target).clone()],
                        constraints: indexmap::IndexMap::new(),
                        path: None,
                        deactivated: false,
                        label: None,
                        description: None,
                        groups: vec![],
                        order: None,
                        severity: shape.severity,
                        messages: indexmap::IndexMap::new(),
                        extends: vec![],
                        priority: None,
                        metadata: ShapeMetadata::default(),
                    })?;
                }
                Target::PathBased(path_based_target) => {
                    // Validate the start target
                    self.validate_shape_targets(&Shape {
                        id: shape.id.clone(),
                        shape_type: shape.shape_type.clone(),
                        targets: vec![(*path_based_target.start_target).clone()],
                        constraints: indexmap::IndexMap::new(),
                        path: None,
                        deactivated: false,
                        label: None,
                        description: None,
                        groups: vec![],
                        order: None,
                        severity: shape.severity,
                        messages: indexmap::IndexMap::new(),
                        extends: vec![],
                        priority: None,
                        metadata: ShapeMetadata::default(),
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Validate property paths
    fn validate_shape_paths(&self, shape: &Shape) -> Result<()> {
        if let Some(path) = &shape.path {
            self.validate_property_path(path)?;
        }
        Ok(())
    }

    /// Validate property path structure
    fn validate_property_path(&self, path: &PropertyPath) -> Result<()> {
        match path {
            PropertyPath::Predicate(iri) => {
                self.validate_iri_format(iri.as_str())?;
            }
            PropertyPath::Inverse(inner_path) => {
                self.validate_property_path(inner_path)?;
            }
            PropertyPath::Sequence(paths) => {
                if paths.is_empty() {
                    return Err(ShaclError::ShapeParsing(
                        "Sequence path cannot be empty".to_string(),
                    ));
                }
                for p in paths {
                    self.validate_property_path(p)?;
                }
            }
            PropertyPath::Alternative(paths) => {
                if paths.len() < 2 {
                    return Err(ShaclError::ShapeParsing(
                        "Alternative path must have at least 2 alternatives".to_string(),
                    ));
                }
                for p in paths {
                    self.validate_property_path(p)?;
                }
            }
            PropertyPath::ZeroOrMore(inner_path)
            | PropertyPath::OneOrMore(inner_path)
            | PropertyPath::ZeroOrOne(inner_path) => {
                self.validate_property_path(inner_path)?;
            }
        }
        Ok(())
    }

    /// Validate shape constraints
    fn validate_shape_constraints(&self, shape: &Shape) -> Result<()> {
        for (component_id, constraint) in &shape.constraints {
            // Validate individual constraint
            constraint.validate().map_err(|e| {
                ShaclError::ShapeParsing(format!(
                    "Invalid constraint '{}' in shape '{}': {}",
                    component_id, shape.id, e
                ))
            })?;
        }
        Ok(())
    }

    /// Validate constraint combinations for logical consistency
    fn validate_constraint_combinations(&self, shape: &Shape) -> Result<()> {
        let constraints = &shape.constraints;

        // Check for conflicting cardinality constraints
        if let (Some(min_count), Some(max_count)) = (
            constraints.get(&ConstraintComponentId::new("minCount")),
            constraints.get(&ConstraintComponentId::new("maxCount")),
        ) {
            if let (Constraint::MinCount(min), Constraint::MaxCount(max)) = (min_count, max_count) {
                if min.min_count > max.max_count {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Shape '{}' has conflicting cardinality: minCount ({}) > maxCount ({})",
                        shape.id, min.min_count, max.max_count
                    )));
                }
            }
        }

        // Check for conflicting string length constraints
        if let (Some(min_length), Some(max_length)) = (
            constraints.get(&ConstraintComponentId::new("minLength")),
            constraints.get(&ConstraintComponentId::new("maxLength")),
        ) {
            if let (Constraint::MinLength(min), Constraint::MaxLength(max)) =
                (min_length, max_length)
            {
                if min.min_length > max.max_length {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Shape '{}' has conflicting string length: minLength ({}) > maxLength ({})",
                        shape.id, min.min_length, max.max_length
                    )));
                }
            }
        }

        // Check for conflicting numeric range constraints
        self.validate_numeric_range_consistency(shape, constraints)?;

        Ok(())
    }

    /// Validate numeric range constraint consistency
    fn validate_numeric_range_consistency(
        &self,
        shape: &Shape,
        constraints: &IndexMap<ConstraintComponentId, Constraint>,
    ) -> Result<()> {
        // Extract numeric bounds (as literals for basic consistency check)
        let mut min_inclusive: Option<&Literal> = None;
        let mut max_inclusive: Option<&Literal> = None;
        let mut min_exclusive: Option<&Literal> = None;
        let mut max_exclusive: Option<&Literal> = None;

        for (_, constraint) in constraints {
            match constraint {
                Constraint::MinInclusive(val) => min_inclusive = Some(&val.min_value),
                Constraint::MaxInclusive(val) => max_inclusive = Some(&val.max_value),
                Constraint::MinExclusive(val) => min_exclusive = Some(&val.min_value),
                Constraint::MaxExclusive(val) => max_exclusive = Some(&val.max_value),
                _ => {}
            }
        }

        // Basic consistency check - try to parse as numbers for simple validation
        if let (Some(min_inc), Some(max_inc)) = (min_inclusive, max_inclusive) {
            if let (Ok(min_val), Ok(max_val)) = (
                min_inc.value().parse::<f64>(),
                max_inc.value().parse::<f64>(),
            ) {
                if min_val > max_val {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Shape '{}' has inconsistent range: minInclusive ({}) > maxInclusive ({})",
                        shape.id, min_inc, max_inc
                    )));
                }
            }
        }

        if let (Some(min_exc), Some(max_exc)) = (min_exclusive, max_exclusive) {
            if let (Ok(min_val), Ok(max_val)) = (
                min_exc.value().parse::<f64>(),
                max_exc.value().parse::<f64>(),
            ) {
                if min_val >= max_val {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Shape '{}' has inconsistent range: minExclusive ({}) >= maxExclusive ({})",
                        shape.id, min_exc, max_exc
                    )));
                }
            }
        }

        Ok(())
    }

    /// Validate shape metadata
    fn validate_shape_metadata(&self, shape: &Shape) -> Result<()> {
        // Validate severity
        match shape.severity {
            Severity::Info | Severity::Warning | Severity::Violation => {
                // Valid severities
            }
        }

        // Validate messages (check for valid language tags)
        for (lang_tag, _message) in &shape.messages {
            if !self.is_valid_language_tag(lang_tag) {
                return Err(ShaclError::ShapeParsing(format!(
                    "Shape '{}' has invalid language tag: '{}'",
                    shape.id, lang_tag
                )));
            }
        }

        Ok(())
    }

    /// Validate inheritance references
    fn validate_inheritance_references(&self, shape: &Shape) -> Result<()> {
        for parent_shape_id in &shape.extends {
            // Check if the parent shape is known
            if !self.shapes.contains_key(parent_shape_id) {
                tracing::warn!(
                    "Shape '{}' extends unknown shape '{}'",
                    shape.id,
                    parent_shape_id
                );
                // This is a warning, not an error, as the parent shape might be defined later
            }
        }
        Ok(())
    }

    /// Enhanced SHACL specification compliance validation
    fn validate_shacl_compliance(&self, shape: &Shape) -> Result<()> {
        // Check for required properties on property shapes
        if shape.is_property_shape() && shape.path.is_none() {
            return Err(ShaclError::ShapeParsing(format!(
                "Property shape '{}' must have exactly one sh:path property",
                shape.id
            )));
        }

        // Validate that node shapes don't have sh:path
        if shape.is_node_shape() && shape.path.is_some() {
            return Err(ShaclError::ShapeParsing(format!(
                "Node shape '{}' must not have sh:path property",
                shape.id
            )));
        }

        // Check for valid target combinations
        self.validate_target_combinations(shape)?;

        // Validate constraint applicability to shape type
        self.validate_constraint_applicability(shape)?;

        // Check for deprecated or experimental features
        self.validate_feature_usage(shape)?;

        Ok(())
    }

    /// Validate target combinations
    fn validate_target_combinations(&self, shape: &Shape) -> Result<()> {
        // Node shapes should typically have targets, property shapes may or may not
        if shape.is_node_shape() && shape.targets.is_empty() {
            tracing::warn!(
                "Node shape '{}' has no targets - it will not validate any data",
                shape.id
            );
        }

        // Check for redundant targets
        let mut unique_targets = HashSet::new();
        for target in &shape.targets {
            let target_key = format!("{target:?}");
            if unique_targets.contains(&target_key) {
                tracing::warn!("Shape '{}' has duplicate target: {:?}", shape.id, target);
            }
            unique_targets.insert(target_key);
        }

        Ok(())
    }

    /// Validate constraint applicability to shape type
    fn validate_constraint_applicability(&self, shape: &Shape) -> Result<()> {
        for (constraint_id, constraint) in &shape.constraints {
            // Some constraints only make sense for property shapes
            match constraint {
                Constraint::MinCount(_) | Constraint::MaxCount(_) => {
                    if shape.is_node_shape() {
                        tracing::warn!(
                            "Cardinality constraint '{}' in node shape '{}' has no effect",
                            constraint_id,
                            shape.id
                        );
                    }
                }
                Constraint::UniqueLang(_) => {
                    if shape.is_node_shape() {
                        tracing::warn!(
                            "UniqueLang constraint in node shape '{}' may not be meaningful",
                            shape.id
                        );
                    }
                }
                _ => {} // Other constraints are valid for both types
            }
        }
        Ok(())
    }

    /// Validate feature usage for warnings about deprecated/experimental features
    fn validate_feature_usage(&self, _shape: &Shape) -> Result<()> {
        // This could be expanded to check for deprecated SHACL features
        // or warn about experimental extensions
        Ok(())
    }

    /// Validate constraint parameters for correctness
    fn validate_constraint_parameters(&self, shape: &Shape) -> Result<()> {
        for (constraint_id, constraint) in &shape.constraints {
            match constraint {
                Constraint::Pattern(pattern_constraint) => {
                    // Validate regex pattern
                    if let Err(e) = regex::Regex::new(&pattern_constraint.pattern) {
                        return Err(ShaclError::ShapeParsing(format!(
                            "Invalid regex pattern in constraint '{}' for shape '{}': {}",
                            constraint_id, shape.id, e
                        )));
                    }
                }
                Constraint::LanguageIn(lang_constraint) => {
                    // Validate language tags
                    for lang_tag in &lang_constraint.languages {
                        if !self.is_valid_language_tag(lang_tag) {
                            return Err(ShaclError::ShapeParsing(format!(
                                "Invalid language tag '{}' in constraint '{}' for shape '{}'",
                                lang_tag, constraint_id, shape.id
                            )));
                        }
                    }
                }
                Constraint::In(in_constraint) => {
                    // Validate that the list is not empty
                    if in_constraint.values.is_empty() {
                        return Err(ShaclError::ShapeParsing(format!(
                            "sh:in constraint in shape '{}' cannot have empty value list",
                            shape.id
                        )));
                    }
                }
                Constraint::QualifiedValueShape(qvs_constraint) => {
                    // Validate qualified cardinality constraints
                    if qvs_constraint.qualified_min_count.is_none()
                        && qvs_constraint.qualified_max_count.is_none()
                    {
                        return Err(ShaclError::ShapeParsing(format!(
                            "QualifiedValueShape constraint in shape '{}' must have at least one of sh:qualifiedMinCount or sh:qualifiedMaxCount",
                            shape.id
                        )));
                    }

                    if let (Some(min), Some(max)) = (
                        qvs_constraint.qualified_min_count,
                        qvs_constraint.qualified_max_count,
                    ) {
                        if min > max {
                            return Err(ShaclError::ShapeParsing(format!(
                                "QualifiedValueShape constraint in shape '{}' has qualifiedMinCount ({}) > qualifiedMaxCount ({})",
                                shape.id, min, max
                            )));
                        }
                    }
                }
                _ => {} // Other constraints validated elsewhere
            }
        }
        Ok(())
    }

    /// Validate logical consistency of constraint combinations
    fn validate_logical_consistency(&self, shape: &Shape) -> Result<()> {
        let constraints = &shape.constraints;

        // Check for impossible combinations
        self.validate_impossible_combinations(shape, constraints)?;

        // Check for redundant constraints
        self.validate_redundant_constraints(shape, constraints)?;

        // Check for conflicting constraints
        self.validate_conflicting_constraints(shape, constraints)?;

        Ok(())
    }

    /// Check for impossible constraint combinations
    fn validate_impossible_combinations(
        &self,
        shape: &Shape,
        constraints: &IndexMap<ConstraintComponentId, Constraint>,
    ) -> Result<()> {
        // Check for NodeKind + Datatype conflicts
        if let (Some(node_kind), Some(datatype)) = (
            constraints.get(&ConstraintComponentId::new("nodeKind")),
            constraints.get(&ConstraintComponentId::new("datatype")),
        ) {
            if let (Constraint::NodeKind(nk), Constraint::Datatype(_)) = (node_kind, datatype) {
                use crate::constraints::value_constraints::NodeKind;
                match nk.node_kind {
                    NodeKind::Iri | NodeKind::BlankNode | NodeKind::BlankNodeOrIri => {
                        return Err(ShaclError::ShapeParsing(format!(
                            "Shape '{}' has conflicting nodeKind and datatype constraints - IRIs and blank nodes cannot have datatypes",
                            shape.id
                        )));
                    }
                    _ => {} // Literal node kinds can have datatypes
                }
            }
        }

        Ok(())
    }

    /// Check for redundant constraints
    fn validate_redundant_constraints(
        &self,
        shape: &Shape,
        _constraints: &IndexMap<ConstraintComponentId, Constraint>,
    ) -> Result<()> {
        // This could be expanded to detect more redundant patterns
        tracing::debug!(
            "Checking for redundant constraints in shape '{}'...",
            shape.id
        );
        Ok(())
    }

    /// Check for conflicting constraints beyond basic cases
    fn validate_conflicting_constraints(
        &self,
        shape: &Shape,
        constraints: &IndexMap<ConstraintComponentId, Constraint>,
    ) -> Result<()> {
        // Check for class/datatype conflicts
        if let (Some(Constraint::Class(_)), Some(Constraint::Datatype(_))) = (
            constraints.get(&ConstraintComponentId::new("class")),
            constraints.get(&ConstraintComponentId::new("datatype")),
        ) {
            tracing::warn!(
                "Shape '{}' has both class and datatype constraints - ensure they are compatible",
                shape.id
            );
        }

        Ok(())
    }

    /// Validate IRI format
    fn validate_iri_format(&self, iri: &str) -> Result<()> {
        use url::Url;

        // Check if it's a valid absolute IRI
        if iri.starts_with("http://") || iri.starts_with("https://") || iri.starts_with("urn:") {
            Url::parse(iri)
                .map_err(|e| ShaclError::ShapeParsing(format!("Invalid IRI '{iri}': {e}")))?;
        } else if iri.contains(':') {
            // Could be a prefixed name - validate prefix part
            if let Some(colon_pos) = iri.find(':') {
                let prefix = &iri[..colon_pos];
                if prefix.is_empty() {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Invalid prefixed name '{iri}': empty prefix"
                    )));
                }
            }
        } else {
            return Err(ShaclError::ShapeParsing(format!(
                "Invalid IRI '{iri}': must be absolute or prefixed"
            )));
        }

        Ok(())
    }

    /// Validate SPARQL query syntax (basic validation)
    fn validate_sparql_query(&self, query: &str) -> Result<()> {
        // Basic validation - check for required keywords
        let query_upper = query.to_uppercase();

        if !query_upper.contains("SELECT")
            && !query_upper.contains("ASK")
            && !query_upper.contains("CONSTRUCT")
            && !query_upper.contains("DESCRIBE")
        {
            return Err(ShaclError::ShapeParsing(
                "SPARQL query must contain a valid query form (SELECT, ASK, CONSTRUCT, or DESCRIBE)".to_string()
            ));
        }

        // Check for balanced braces
        let mut brace_count = 0;
        for char in query.chars() {
            match char {
                '{' => brace_count += 1,
                '}' => brace_count -= 1,
                _ => {}
            }
            if brace_count < 0 {
                return Err(ShaclError::ShapeParsing(
                    "SPARQL query has unbalanced braces".to_string(),
                ));
            }
        }

        if brace_count != 0 {
            return Err(ShaclError::ShapeParsing(
                "SPARQL query has unbalanced braces".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if a language tag is valid (basic check)
    fn is_valid_language_tag(&self, tag: &str) -> bool {
        // Very basic validation - language tags should be 2-3 characters, possibly with region
        // Format: language[-region] e.g., "en", "en-US", "fr-CA"
        if tag.is_empty() || tag.len() > 8 {
            return false;
        }

        let parts: Vec<&str> = tag.split('-').collect();
        if parts.is_empty() || parts.len() > 2 {
            return false;
        }

        // First part should be 2-3 character language code
        let lang = parts[0];
        if lang.len() < 2 || lang.len() > 3 || !lang.chars().all(|c| c.is_alphabetic()) {
            return false;
        }

        // Optional second part should be 2-3 character region code
        if parts.len() == 2 {
            let region = parts[1];
            if region.len() < 2 || region.len() > 3 || !region.chars().all(|c| c.is_alphabetic()) {
                return false;
            }
        }

        true
    }

    /// Update shape dependency graph
    fn update_shape_dependencies(&mut self, shape_id: ShapeId) -> Result<()> {
        // Find or create node for this shape
        let shape_node = self.get_or_create_shape_node(&shape_id);

        // Collect all dependencies first to avoid borrow checker issues
        let mut all_dependencies = Vec::new();

        if let Some(shape) = self.shapes.get(&shape_id) {
            // Collect dependencies from inheritance
            all_dependencies.extend(shape.extends.iter().cloned());

            // Collect dependencies from constraints
            for (_, constraint) in &shape.constraints {
                let dependencies = self.extract_constraint_dependencies(constraint);
                all_dependencies.extend(dependencies);
            }
        }

        // Now add all the edges
        for dep_shape_id in all_dependencies {
            let dep_node = self.get_or_create_shape_node(&dep_shape_id);
            self.shape_dependencies.add_edge(shape_node, dep_node, ());
        }

        // Check for circular dependencies
        if petgraph::algo::is_cyclic_directed(&self.shape_dependencies) {
            return Err(ShaclError::ShapeParsing(
                "circular dependency detected in shape definitions".to_string(),
            ));
        }

        Ok(())
    }

    /// Get or create a node in the dependency graph for a shape
    fn get_or_create_shape_node(&mut self, shape_id: &ShapeId) -> petgraph::graph::NodeIndex {
        use petgraph::visit::IntoNodeReferences;

        // Check if node already exists
        for (node_idx, node_shape_id) in self.shape_dependencies.node_references() {
            if node_shape_id == shape_id {
                return node_idx;
            }
        }

        // Create new node
        self.shape_dependencies.add_node(shape_id.clone())
    }

    /// Extract shape dependencies from a constraint
    fn extract_constraint_dependencies(&self, constraint: &Constraint) -> Vec<ShapeId> {
        let mut dependencies = Vec::new();

        match constraint {
            Constraint::Node(node_constraint) => {
                dependencies.push(node_constraint.shape.clone());
            }
            Constraint::QualifiedValueShape(qualified_constraint) => {
                dependencies.push(qualified_constraint.shape.clone());
            }
            Constraint::Not(not_constraint) => {
                dependencies.push(not_constraint.shape.clone());
            }
            Constraint::And(and_constraint) => {
                dependencies.extend(and_constraint.shapes.clone());
            }
            Constraint::Or(or_constraint) => {
                dependencies.extend(or_constraint.shapes.clone());
            }
            Constraint::Xone(xone_constraint) => {
                dependencies.extend(xone_constraint.shapes.clone());
            }
            _ => {
                // Other constraints don't have shape dependencies
            }
        }

        dependencies
    }

    /// Get optimal shape evaluation order based on dependencies and sh:order property
    pub fn get_evaluation_order(&self) -> Vec<ShapeId> {
        use petgraph::algo::toposort;

        // Perform topological sort on the dependency graph
        let mut shape_ids: Vec<ShapeId> = match toposort(&self.shape_dependencies, None) {
            Ok(sorted_nodes) => {
                // Map node indices back to shape IDs
                sorted_nodes
                    .into_iter()
                    .filter_map(|node_idx| self.shape_dependencies.node_weight(node_idx).cloned())
                    .collect()
            }
            Err(_) => {
                // If there's a cycle (shouldn't happen as we check during add),
                // fall back to arbitrary order
                tracing::warn!("Dependency cycle detected during evaluation order calculation");
                self.shapes.keys().cloned().collect()
            }
        };

        // Secondary sort by sh:order property for stable ordering
        shape_ids.sort_by(|a: &ShapeId, b: &ShapeId| {
            let shape_a = self.shapes.get(a);
            let shape_b = self.shapes.get(b);

            match (shape_a, shape_b) {
                (Some(s_a), Some(s_b)) => {
                    // Primary sort by order (lower values first)
                    s_a.order
                        .unwrap_or(0)
                        .cmp(&s_b.order.unwrap_or(0))
                        // Secondary sort by priority (higher values first)
                        .then_with(|| s_b.effective_priority().cmp(&s_a.effective_priority()))
                        // Tertiary sort by shape ID for stability
                        .then_with(|| a.as_str().cmp(b.as_str()))
                }
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.as_str().cmp(b.as_str()),
            }
        });

        shape_ids
    }

    /// Clear all internal caches
    pub fn clear_caches(&mut self) {
        self.target_cache.clear();
    }

    /// Resolve inherited constraints for a shape
    pub fn resolve_inherited_constraints(
        &self,
        shape_id: &ShapeId,
    ) -> Result<IndexMap<ConstraintComponentId, Constraint>> {
        let mut resolved_constraints = IndexMap::new();
        let mut visited = HashSet::new();

        self.collect_inherited_constraints(shape_id, &mut resolved_constraints, &mut visited)?;

        Ok(resolved_constraints)
    }

    /// Recursively collect constraints from a shape and its parents
    fn collect_inherited_constraints(
        &self,
        shape_id: &ShapeId,
        resolved_constraints: &mut IndexMap<ConstraintComponentId, Constraint>,
        visited: &mut HashSet<ShapeId>,
    ) -> Result<()> {
        // Avoid infinite recursion
        if visited.contains(shape_id) {
            return Ok(());
        }
        visited.insert(shape_id.clone());

        if let Some(shape) = self.shapes.get(shape_id) {
            // First process parent shapes (depth-first)
            for parent_id in &shape.extends {
                self.collect_inherited_constraints(parent_id, resolved_constraints, visited)?;
            }

            // Then add this shape's constraints, potentially overriding parent constraints
            for (component_id, constraint) in &shape.constraints {
                resolved_constraints.insert(component_id.clone(), constraint.clone());
            }
        }

        Ok(())
    }

    /// Resolve shapes with priority-based conflict resolution
    /// Returns shapes sorted by priority (highest first)
    pub fn resolve_shapes_by_priority(&self, shape_ids: &[ShapeId]) -> Vec<&Shape> {
        let mut shapes: Vec<&Shape> = shape_ids
            .iter()
            .filter_map(|id| self.shapes.get(id))
            .collect();

        // Sort by priority (highest first), then by ID for stability
        shapes.sort_by(|a, b| {
            b.effective_priority()
                .cmp(&a.effective_priority())
                .then_with(|| a.id.as_str().cmp(b.id.as_str()))
        });

        shapes
    }

    /// Get all shapes that a given shape inherits from (transitively)
    pub fn get_all_parent_shapes(&self, shape_id: &ShapeId) -> Result<Vec<ShapeId>> {
        let mut parents = Vec::new();
        let mut visited = HashSet::new();

        self.collect_parent_shapes(shape_id, &mut parents, &mut visited)?;

        Ok(parents)
    }

    fn collect_parent_shapes(
        &self,
        shape_id: &ShapeId,
        parents: &mut Vec<ShapeId>,
        visited: &mut HashSet<ShapeId>,
    ) -> Result<()> {
        if visited.contains(shape_id) {
            return Ok(());
        }
        visited.insert(shape_id.clone());

        if let Some(shape) = self.shapes.get(shape_id) {
            for parent_id in &shape.extends {
                parents.push(parent_id.clone());
                self.collect_parent_shapes(parent_id, parents, visited)?;
            }
        }

        Ok(())
    }

    /// Get validation statistics
    pub fn get_statistics(&self) -> ValidationStats {
        ValidationStats {
            total_shapes: self.shapes.len(),
            node_shapes: self.shapes.values().filter(|s| s.is_node_shape()).count(),
            property_shapes: self
                .shapes
                .values()
                .filter(|s| s.is_property_shape())
                .count(),
            active_shapes: self.shapes.values().filter(|s| s.is_active()).count(),
            deactivated_shapes: self.shapes.values().filter(|s| !s.is_active()).count(),
        }
    }
}

impl Default for Validator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStats {
    pub total_shapes: usize,
    pub node_shapes: usize,
    pub property_shapes: usize,
    pub active_shapes: usize,
    pub deactivated_shapes: usize,
}

/// Builder for creating validators with custom configuration
#[derive(Debug)]
pub struct ValidatorBuilder {
    config: ValidationConfig,
}

impl ValidatorBuilder {
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    pub fn max_violations(mut self, max: usize) -> Self {
        self.config.max_violations = max;
        self
    }

    pub fn include_info(mut self, include: bool) -> Self {
        self.config.include_info = include;
        self
    }

    pub fn include_warnings(mut self, include: bool) -> Self {
        self.config.include_warnings = include;
        self
    }

    pub fn fail_fast(mut self, fail_fast: bool) -> Self {
        self.config.fail_fast = fail_fast;
        self
    }

    pub fn max_recursion_depth(mut self, depth: usize) -> Self {
        self.config.max_recursion_depth = depth;
        self
    }

    pub fn timeout_ms(mut self, timeout: Option<u64>) -> Self {
        self.config.timeout_ms = timeout;
        self
    }

    pub fn parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }

    pub fn context(mut self, context: HashMap<String, String>) -> Self {
        self.config.context = context;
        self
    }

    pub fn build(self) -> Validator {
        Validator::with_config(self.config)
    }
}

impl Default for ValidatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
