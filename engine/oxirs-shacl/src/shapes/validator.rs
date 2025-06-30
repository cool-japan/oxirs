//! Shape validator for validating that shapes graphs are well-formed

use std::collections::{HashMap, HashSet};

use crate::{paths::PropertyPath, targets::Target, Result, ShaclError, Shape, ShapeId};

/// Shape validator for validating that shapes graphs are well-formed
#[derive(Debug)]
pub struct ShapeValidator {
    /// Enable strict validation mode
    strict_mode: bool,
    /// Maximum recursion depth for shape reference validation
    max_depth: usize,
}

impl ShapeValidator {
    /// Create a new shape validator
    pub fn new() -> Self {
        Self {
            strict_mode: false,
            max_depth: 20,
        }
    }

    /// Create a new validator in strict mode
    pub fn new_strict() -> Self {
        Self {
            strict_mode: true,
            max_depth: 20,
        }
    }

    /// Set maximum recursion depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Validate a collection of shapes
    pub fn validate_shapes(&self, shapes: &[Shape]) -> Result<ShapeValidationReport> {
        let mut report = ShapeValidationReport::new();
        let mut shape_map = HashMap::new();

        // Build shape map for reference validation
        for shape in shapes {
            shape_map.insert(shape.id.clone(), shape);
        }

        // Validate each shape
        for shape in shapes {
            match self.validate_single_shape(shape, &shape_map, 0) {
                Ok(shape_report) => {
                    report.add_shape_result(shape_report);
                }
                Err(e) => {
                    let mut shape_report = SingleShapeValidationReport::new(shape.id.clone());
                    shape_report.add_error(format!("Critical validation error: {}", e));
                    report.add_shape_result(shape_report);
                }
            }
        }

        // Validate shape dependencies and circular references
        self.validate_shape_dependencies(&shape_map, &mut report)?;

        Ok(report)
    }

    /// Validate a single shape
    fn validate_single_shape(
        &self,
        shape: &Shape,
        shape_map: &HashMap<ShapeId, &Shape>,
        depth: usize,
    ) -> Result<SingleShapeValidationReport> {
        if depth > self.max_depth {
            return Err(ShaclError::ShapeValidation(format!(
                "Maximum validation depth {} exceeded for shape {}",
                self.max_depth, shape.id
            )));
        }

        let mut report = SingleShapeValidationReport::new(shape.id.clone());

        // Validate shape structure
        self.validate_shape_structure(shape, &mut report);

        // Validate targets
        self.validate_shape_targets(shape, &mut report);

        // Validate property path (for property shapes)
        if shape.is_property_shape() {
            self.validate_property_path(shape, &mut report);
        }

        // Validate constraints
        self.validate_shape_constraints(shape, shape_map, &mut report, depth);

        // Validate metadata consistency
        self.validate_shape_metadata(shape, &mut report);

        Ok(report)
    }

    /// Validate basic shape structure
    fn validate_shape_structure(&self, shape: &Shape, report: &mut SingleShapeValidationReport) {
        // Check shape type consistency
        match shape.shape_type {
            crate::ShapeType::NodeShape => {
                if shape.path.is_some() {
                    report.add_error("Node shapes must not have sh:path property".to_string());
                }
            }
            crate::ShapeType::PropertyShape => {
                if shape.path.is_none() {
                    report.add_error("Property shapes must have exactly one sh:path property".to_string());
                }
            }
        }

        // Validate shape ID
        if shape.id.as_str().is_empty() {
            report.add_error("Shape ID cannot be empty".to_string());
        }
    }

    /// Validate shape targets
    fn validate_shape_targets(&self, shape: &Shape, report: &mut SingleShapeValidationReport) {
        if shape.is_node_shape() && shape.targets.is_empty() {
            report.add_warning("Node shape has no targets - it will not validate any data".to_string());
        }

        for target in &shape.targets {
            if let Err(e) = self.validate_target(target) {
                report.add_error(format!("Invalid target: {}", e));
            }
        }
    }

    /// Validate a specific target
    fn validate_target(&self, target: &Target) -> Result<()> {
        match target {
            Target::Class(_) | Target::Node(_) | Target::ObjectsOf(_) | Target::SubjectsOf(_) | Target::Implicit(_) => {
                // Basic validation passed
                Ok(())
            }
            Target::Sparql(sparql_target) => {
                // Basic SPARQL query validation
                if sparql_target.query.trim().is_empty() {
                    return Err(ShaclError::ShapeValidation("SPARQL target query cannot be empty".to_string()));
                }
                Ok(())
            }
        }
    }

    /// Validate property path
    fn validate_property_path(&self, shape: &Shape, report: &mut SingleShapeValidationReport) {
        if let Some(path) = &shape.path {
            if let Err(e) = self.validate_path_structure(path) {
                report.add_error(format!("Invalid property path: {}", e));
            }
        }
    }

    /// Validate property path structure
    fn validate_path_structure(&self, path: &PropertyPath) -> Result<()> {
        match path {
            PropertyPath::Predicate(_) => Ok(()),
            PropertyPath::Inverse(inner) => self.validate_path_structure(inner),
            PropertyPath::Sequence(paths) => {
                if paths.is_empty() {
                    return Err(ShaclError::ShapeValidation("Sequence path cannot be empty".to_string()));
                }
                for p in paths {
                    self.validate_path_structure(p)?;
                }
                Ok(())
            }
            PropertyPath::Alternative(paths) => {
                if paths.len() < 2 {
                    return Err(ShaclError::ShapeValidation("Alternative path must have at least 2 alternatives".to_string()));
                }
                for p in paths {
                    self.validate_path_structure(p)?;
                }
                Ok(())
            }
            PropertyPath::ZeroOrMore(inner) | PropertyPath::OneOrMore(inner) | PropertyPath::ZeroOrOne(inner) => {
                self.validate_path_structure(inner)
            }
        }
    }

    /// Validate shape constraints
    fn validate_shape_constraints(
        &self,
        shape: &Shape,
        shape_map: &HashMap<ShapeId, &Shape>,
        report: &mut SingleShapeValidationReport,
        depth: usize,
    ) {
        for (component_id, constraint) in &shape.constraints {
            // Validate individual constraint
            if let Err(e) = constraint.validate() {
                report.add_error(format!("Invalid constraint '{}': {}", component_id, e));
            }

            // Validate constraint references to other shapes (if any)
            if let Err(e) = self.validate_constraint_references(constraint, shape_map, depth) {
                report.add_error(format!("Invalid constraint reference in '{}': {}", component_id, e));
            }
        }
    }

    /// Validate constraint references to other shapes
    fn validate_constraint_references(
        &self,
        constraint: &crate::Constraint,
        shape_map: &HashMap<ShapeId, &Shape>,
        depth: usize,
    ) -> Result<()> {
        use crate::Constraint;

        match constraint {
            Constraint::Node(node_constraint) => {
                if !shape_map.contains_key(&node_constraint.shape) {
                    return Err(ShaclError::ShapeValidation(format!(
                        "Referenced shape '{}' not found",
                        node_constraint.shape
                    )));
                }
            }
            Constraint::QualifiedValueShape(qvs_constraint) => {
                if !shape_map.contains_key(&qvs_constraint.shape) {
                    return Err(ShaclError::ShapeValidation(format!(
                        "Referenced shape '{}' not found in qualified value shape",
                        qvs_constraint.shape
                    )));
                }
            }
            Constraint::And(and_constraint) => {
                for shape_id in &and_constraint.shapes {
                    if !shape_map.contains_key(shape_id) {
                        return Err(ShaclError::ShapeValidation(format!(
                            "Referenced shape '{}' not found in AND constraint",
                            shape_id
                        )));
                    }
                }
            }
            Constraint::Or(or_constraint) => {
                for shape_id in &or_constraint.shapes {
                    if !shape_map.contains_key(shape_id) {
                        return Err(ShaclError::ShapeValidation(format!(
                            "Referenced shape '{}' not found in OR constraint",
                            shape_id
                        )));
                    }
                }
            }
            Constraint::Xone(xone_constraint) => {
                for shape_id in &xone_constraint.shapes {
                    if !shape_map.contains_key(shape_id) {
                        return Err(ShaclError::ShapeValidation(format!(
                            "Referenced shape '{}' not found in XONE constraint",
                            shape_id
                        )));
                    }
                }
            }
            Constraint::Not(not_constraint) => {
                if !shape_map.contains_key(&not_constraint.shape) {
                    return Err(ShaclError::ShapeValidation(format!(
                        "Referenced shape '{}' not found in NOT constraint",
                        not_constraint.shape
                    )));
                }
            }
            _ => {
                // Other constraints don't reference shapes
            }
        }

        Ok(())
    }

    /// Validate shape metadata
    fn validate_shape_metadata(&self, shape: &Shape, report: &mut SingleShapeValidationReport) {
        // Validate severity
        match shape.severity {
            crate::Severity::Info | crate::Severity::Warning | crate::Severity::Violation => {
                // Valid severities
            }
        }

        // Check for reasonable order values
        if let Some(order) = shape.order {
            if order < 0 {
                report.add_warning(format!("Shape order {} is negative", order));
            }
        }

        // Check for reasonable priority values
        if let Some(priority) = shape.priority {
            if priority < 0 {
                report.add_warning(format!("Shape priority {} is negative", priority));
            }
        }
    }

    /// Validate shape dependencies and detect circular references
    fn validate_shape_dependencies(
        &self,
        shape_map: &HashMap<ShapeId, &Shape>,
        report: &mut ShapeValidationReport,
    ) -> Result<()> {
        for shape_id in shape_map.keys() {
            let mut visited = HashSet::new();
            if self.has_circular_dependency(shape_id, shape_map, &mut visited) {
                report.add_global_error(format!("Circular dependency detected starting from shape '{}'", shape_id));
            }
        }
        Ok(())
    }

    /// Check for circular dependencies starting from a specific shape
    fn has_circular_dependency(
        &self,
        shape_id: &ShapeId,
        shape_map: &HashMap<ShapeId, &Shape>,
        visited: &mut HashSet<ShapeId>,
    ) -> bool {
        if visited.contains(shape_id) {
            return true;
        }

        visited.insert(shape_id.clone());

        if let Some(shape) = shape_map.get(shape_id) {
            // Check inheritance dependencies
            for parent_id in &shape.extends {
                if self.has_circular_dependency(parent_id, shape_map, visited) {
                    return true;
                }
            }

            // Check constraint dependencies
            for constraint in shape.constraints.values() {
                if self.constraint_has_circular_dependency(constraint, shape_map, visited) {
                    return true;
                }
            }
        }

        visited.remove(shape_id);
        false
    }

    /// Check if a constraint has circular dependencies
    fn constraint_has_circular_dependency(
        &self,
        constraint: &crate::Constraint,
        shape_map: &HashMap<ShapeId, &Shape>,
        visited: &mut HashSet<ShapeId>,
    ) -> bool {
        use crate::Constraint;

        match constraint {
            Constraint::Node(node_constraint) => {
                self.has_circular_dependency(&node_constraint.shape, shape_map, visited)
            }
            Constraint::QualifiedValueShape(qvs_constraint) => {
                self.has_circular_dependency(&qvs_constraint.shape, shape_map, visited)
            }
            Constraint::And(and_constraint) => {
                for shape_id in &and_constraint.shapes {
                    if self.has_circular_dependency(shape_id, shape_map, visited) {
                        return true;
                    }
                }
                false
            }
            Constraint::Or(or_constraint) => {
                for shape_id in &or_constraint.shapes {
                    if self.has_circular_dependency(shape_id, shape_map, visited) {
                        return true;
                    }
                }
                false
            }
            Constraint::Xone(xone_constraint) => {
                for shape_id in &xone_constraint.shapes {
                    if self.has_circular_dependency(shape_id, shape_map, visited) {
                        return true;
                    }
                }
                false
            }
            Constraint::Not(not_constraint) => {
                self.has_circular_dependency(&not_constraint.shape, shape_map, visited)
            }
            _ => false,
        }
    }
}

impl Default for ShapeValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Shape validation report
#[derive(Debug, Clone)]
pub struct ShapeValidationReport {
    /// Validation results for individual shapes
    pub shape_results: Vec<SingleShapeValidationReport>,
    /// Global validation errors
    pub global_errors: Vec<String>,
    /// Overall validation status
    pub is_valid: bool,
}

impl ShapeValidationReport {
    pub fn new() -> Self {
        Self {
            shape_results: Vec::new(),
            global_errors: Vec::new(),
            is_valid: true,
        }
    }

    pub fn add_shape_result(&mut self, result: SingleShapeValidationReport) {
        if !result.is_valid() {
            self.is_valid = false;
        }
        self.shape_results.push(result);
    }

    pub fn add_global_error(&mut self, error: String) {
        self.global_errors.push(error);
        self.is_valid = false;
    }

    pub fn total_errors(&self) -> usize {
        self.global_errors.len()
            + self
                .shape_results
                .iter()
                .map(|r| r.errors.len())
                .sum::<usize>()
    }

    pub fn total_warnings(&self) -> usize {
        self.shape_results
            .iter()
            .map(|r| r.warnings.len())
            .sum::<usize>()
    }
}

impl Default for ShapeValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation report for a single shape
#[derive(Debug, Clone)]
pub struct SingleShapeValidationReport {
    pub shape_id: ShapeId,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl SingleShapeValidationReport {
    pub fn new(shape_id: ShapeId) -> Self {
        Self {
            shape_id,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}