/// SAMM operation registry for aspect operations.
///
/// Implements a registry for SAMM `Operation` elements.  Each operation has a
/// unique IRI id, optional description, a list of typed inputs, an optional
/// output, and error entity IRIs.
use std::collections::HashMap;

use thiserror::Error;

// ── Types ─────────────────────────────────────────────────────────────────────

/// The type of an input or output binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputOutputType {
    /// A single value of the named type.
    Single(String),
    /// A collection of the named type.
    Collection(String),
    /// An optional value of the named type.
    Optional(String),
}

impl InputOutputType {
    /// Returns the inner type name regardless of variant.
    pub fn type_name(&self) -> &str {
        match self {
            Self::Single(t) | Self::Collection(t) | Self::Optional(t) => t,
        }
    }
}

/// A single input parameter for an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationInput {
    /// Name of the input parameter.
    pub name: String,
    /// Type of this input parameter.
    pub input_type: InputOutputType,
    /// If `true` the input does not need to be provided by callers.
    pub optional: bool,
}

/// The output descriptor of an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationOutput {
    /// Type of the operation output.
    pub output_type: InputOutputType,
}

/// A SAMM operation definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operation {
    /// IRI identifier of this operation.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Ordered list of input parameters.
    pub inputs: Vec<OperationInput>,
    /// Optional output descriptor.
    pub output: Option<OperationOutput>,
    /// Error entity IRIs this operation can raise.
    pub errors: Vec<String>,
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors returned by `OperationRegistry`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RegistryError {
    /// An operation with the same id was already registered.
    #[error("duplicate operation id: {0}")]
    DuplicateId(String),
    /// The operation failed validation checks.
    #[error("invalid operation: {0}")]
    InvalidOperation(String),
}

// ── Summary ───────────────────────────────────────────────────────────────────

/// Aggregate summary of the registry.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RegistrySummary {
    /// Total number of registered operations.
    pub total: usize,
    /// Number of operations that have an output descriptor.
    pub with_output: usize,
    /// Number of operations without an output descriptor.
    pub without_output: usize,
    /// Total number of input parameters across all operations.
    pub total_inputs: usize,
}

// ── OperationRegistry ─────────────────────────────────────────────────────────

/// Registry holding all operations for an aspect model.
#[derive(Debug, Default)]
pub struct OperationRegistry {
    operations: HashMap<String, Operation>,
}

impl OperationRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new operation.  Returns an error if an operation with the
    /// same `id` already exists, or if the operation is invalid.
    pub fn register(&mut self, op: Operation) -> Result<(), RegistryError> {
        // Validate before inserting
        let warnings = self.validate_operation(&op);
        if warnings.iter().any(|w| w.starts_with("ERROR")) {
            return Err(RegistryError::InvalidOperation(warnings.join("; ")));
        }
        if self.operations.contains_key(&op.id) {
            return Err(RegistryError::DuplicateId(op.id.clone()));
        }
        self.operations.insert(op.id.clone(), op);
        Ok(())
    }

    /// Retrieve an operation by id.
    pub fn get(&self, id: &str) -> Option<&Operation> {
        self.operations.get(id)
    }

    /// Remove an operation by id.  Returns `true` if it existed.
    pub fn remove(&mut self, id: &str) -> bool {
        self.operations.remove(id).is_some()
    }

    /// Find all operations with the given human-readable `name`.
    pub fn find_by_name(&self, name: &str) -> Vec<&Operation> {
        self.operations
            .values()
            .filter(|op| op.name == name)
            .collect()
    }

    /// All operations that have an output descriptor.
    pub fn operations_with_output(&self) -> Vec<&Operation> {
        self.operations
            .values()
            .filter(|op| op.output.is_some())
            .collect()
    }

    /// All operations without an output descriptor.
    pub fn operations_without_output(&self) -> Vec<&Operation> {
        self.operations
            .values()
            .filter(|op| op.output.is_none())
            .collect()
    }

    /// Validate an operation and return a list of human-readable messages.
    /// Messages beginning with `"ERROR"` indicate a blocking problem.
    pub fn validate_operation(&self, op: &Operation) -> Vec<String> {
        let mut msgs: Vec<String> = Vec::new();

        if op.id.is_empty() {
            msgs.push("ERROR: operation id must not be empty".to_string());
        }
        if op.name.is_empty() {
            msgs.push("WARNING: operation name is empty".to_string());
        }
        // Input name uniqueness
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for input in &op.inputs {
            *seen.entry(input.name.as_str()).or_insert(0) += 1;
        }
        for (name, count) in &seen {
            if *count > 1 {
                msgs.push(format!("WARNING: duplicate input name '{name}'"));
            }
        }

        msgs
    }

    /// Number of registered operations.
    pub fn count(&self) -> usize {
        self.operations.len()
    }

    /// All registered operations as an unsorted vec.
    pub fn all(&self) -> Vec<&Operation> {
        self.operations.values().collect()
    }

    /// Aggregate summary of the registry.
    pub fn summary(&self) -> RegistrySummary {
        let total = self.operations.len();
        let with_output = self
            .operations
            .values()
            .filter(|o| o.output.is_some())
            .count();
        let without_output = total - with_output;
        let total_inputs = self.operations.values().map(|o| o.inputs.len()).sum();
        RegistrySummary {
            total,
            with_output,
            without_output,
            total_inputs,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper constructors

    fn single(t: &str) -> InputOutputType {
        InputOutputType::Single(t.into())
    }

    fn collection(t: &str) -> InputOutputType {
        InputOutputType::Collection(t.into())
    }

    fn optional(t: &str) -> InputOutputType {
        InputOutputType::Optional(t.into())
    }

    fn input(name: &str, ty: InputOutputType, optional: bool) -> OperationInput {
        OperationInput {
            name: name.into(),
            input_type: ty,
            optional,
        }
    }

    fn output(ty: InputOutputType) -> OperationOutput {
        OperationOutput { output_type: ty }
    }

    fn op_no_output(id: &str, name: &str) -> Operation {
        Operation {
            id: id.into(),
            name: name.into(),
            description: None,
            inputs: vec![],
            output: None,
            errors: vec![],
        }
    }

    fn op_with_output(id: &str, name: &str, ty: InputOutputType) -> Operation {
        Operation {
            id: id.into(),
            name: name.into(),
            description: None,
            inputs: vec![],
            output: Some(output(ty)),
            errors: vec![],
        }
    }

    // ── register / get ────────────────────────────────────────────────────────

    #[test]
    fn test_register_and_get() {
        let mut reg = OperationRegistry::new();
        let op = op_no_output("op:1", "myOp");
        reg.register(op.clone()).expect("should succeed");
        let retrieved = reg.get("op:1").expect("should succeed");
        assert_eq!(retrieved.name, "myOp");
    }

    #[test]
    fn test_get_nonexistent_returns_none() {
        let reg = OperationRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_register_duplicate_id_error() {
        let mut reg = OperationRegistry::new();
        reg.register(op_no_output("op:1", "a"))
            .expect("should succeed");
        let err = reg.register(op_no_output("op:1", "b")).unwrap_err();
        assert!(matches!(err, RegistryError::DuplicateId(id) if id == "op:1"));
    }

    #[test]
    fn test_register_empty_id_error() {
        let mut reg = OperationRegistry::new();
        let err = reg.register(op_no_output("", "name")).unwrap_err();
        assert!(matches!(err, RegistryError::InvalidOperation(_)));
    }

    // ── remove ────────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing() {
        let mut reg = OperationRegistry::new();
        reg.register(op_no_output("op:1", "a"))
            .expect("should succeed");
        assert!(reg.remove("op:1"));
        assert!(reg.get("op:1").is_none());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut reg = OperationRegistry::new();
        assert!(!reg.remove("nope"));
    }

    #[test]
    fn test_remove_decrements_count() {
        let mut reg = OperationRegistry::new();
        reg.register(op_no_output("op:1", "a"))
            .expect("should succeed");
        reg.register(op_no_output("op:2", "b"))
            .expect("should succeed");
        reg.remove("op:1");
        assert_eq!(reg.count(), 1);
    }

    // ── find_by_name ──────────────────────────────────────────────────────────

    #[test]
    fn test_find_by_name_single() {
        let mut reg = OperationRegistry::new();
        reg.register(op_no_output("op:1", "getStatus"))
            .expect("should succeed");
        let found = reg.find_by_name("getStatus");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "op:1");
    }

    #[test]
    fn test_find_by_name_multiple() {
        let mut reg = OperationRegistry::new();
        reg.register(op_no_output("op:1", "reset"))
            .expect("should succeed");
        reg.register(op_no_output("op:2", "reset"))
            .expect("should succeed");
        let found = reg.find_by_name("reset");
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_find_by_name_not_found() {
        let reg = OperationRegistry::new();
        assert!(reg.find_by_name("missing").is_empty());
    }

    // ── operations_with/without_output ────────────────────────────────────────

    #[test]
    fn test_operations_with_output() {
        let mut reg = OperationRegistry::new();
        reg.register(op_with_output("op:1", "a", single("String")))
            .expect("should succeed");
        reg.register(op_no_output("op:2", "b"))
            .expect("should succeed");
        let with_out = reg.operations_with_output();
        assert_eq!(with_out.len(), 1);
        assert_eq!(with_out[0].id, "op:1");
    }

    #[test]
    fn test_operations_without_output() {
        let mut reg = OperationRegistry::new();
        reg.register(op_no_output("op:1", "a"))
            .expect("should succeed");
        reg.register(op_with_output("op:2", "b", single("Bool")))
            .expect("should succeed");
        let without = reg.operations_without_output();
        assert_eq!(without.len(), 1);
        assert_eq!(without[0].id, "op:1");
    }

    #[test]
    fn test_all_without_output_when_empty() {
        let reg = OperationRegistry::new();
        assert!(reg.operations_without_output().is_empty());
    }

    // ── validate_operation ────────────────────────────────────────────────────

    #[test]
    fn test_validate_operation_valid() {
        let reg = OperationRegistry::new();
        let op = op_no_output("op:1", "myOp");
        let msgs = reg.validate_operation(&op);
        assert!(msgs.iter().all(|m| !m.starts_with("ERROR")));
    }

    #[test]
    fn test_validate_operation_empty_id() {
        let reg = OperationRegistry::new();
        let op = op_no_output("", "x");
        let msgs = reg.validate_operation(&op);
        assert!(msgs.iter().any(|m| m.starts_with("ERROR")));
    }

    #[test]
    fn test_validate_operation_empty_name_warning() {
        let reg = OperationRegistry::new();
        let op = op_no_output("op:1", "");
        let msgs = reg.validate_operation(&op);
        assert!(msgs.iter().any(|m| m.starts_with("WARNING")));
    }

    #[test]
    fn test_validate_operation_duplicate_input_names() {
        let reg = OperationRegistry::new();
        let op = Operation {
            id: "op:1".into(),
            name: "op".into(),
            description: None,
            inputs: vec![
                input("dup", single("String"), false),
                input("dup", single("Int"), false),
            ],
            output: None,
            errors: vec![],
        };
        let msgs = reg.validate_operation(&op);
        assert!(msgs.iter().any(|m| m.contains("duplicate")));
    }

    // ── count / all ───────────────────────────────────────────────────────────

    #[test]
    fn test_count_zero_initially() {
        let reg = OperationRegistry::new();
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_count_after_register() {
        let mut reg = OperationRegistry::new();
        reg.register(op_no_output("op:1", "a"))
            .expect("should succeed");
        reg.register(op_no_output("op:2", "b"))
            .expect("should succeed");
        assert_eq!(reg.count(), 2);
    }

    #[test]
    fn test_all_returns_all() {
        let mut reg = OperationRegistry::new();
        reg.register(op_no_output("op:1", "a"))
            .expect("should succeed");
        reg.register(op_no_output("op:2", "b"))
            .expect("should succeed");
        assert_eq!(reg.all().len(), 2);
    }

    #[test]
    fn test_all_empty_registry() {
        let reg = OperationRegistry::new();
        assert!(reg.all().is_empty());
    }

    // ── summary ───────────────────────────────────────────────────────────────

    #[test]
    fn test_summary_totals() {
        let mut reg = OperationRegistry::new();
        reg.register(op_with_output("op:1", "a", single("String")))
            .expect("should succeed");
        reg.register(op_no_output("op:2", "b"))
            .expect("should succeed");
        let s = reg.summary();
        assert_eq!(s.total, 2);
        assert_eq!(s.with_output, 1);
        assert_eq!(s.without_output, 1);
    }

    #[test]
    fn test_summary_total_inputs() {
        let mut reg = OperationRegistry::new();
        let op1 = Operation {
            id: "op:1".into(),
            name: "a".into(),
            description: None,
            inputs: vec![
                input("i1", single("String"), false),
                input("i2", single("Int"), false),
            ],
            output: None,
            errors: vec![],
        };
        let op2 = Operation {
            id: "op:2".into(),
            name: "b".into(),
            description: None,
            inputs: vec![input("i1", single("Bool"), true)],
            output: None,
            errors: vec![],
        };
        reg.register(op1).expect("should succeed");
        reg.register(op2).expect("should succeed");
        let s = reg.summary();
        assert_eq!(s.total_inputs, 3);
    }

    #[test]
    fn test_summary_empty() {
        let reg = OperationRegistry::new();
        let s = reg.summary();
        assert_eq!(s, RegistrySummary::default());
    }

    // ── InputOutputType ───────────────────────────────────────────────────────

    #[test]
    fn test_single_type_name() {
        assert_eq!(single("Foo").type_name(), "Foo");
    }

    #[test]
    fn test_collection_type_name() {
        assert_eq!(collection("Bar").type_name(), "Bar");
    }

    #[test]
    fn test_optional_type_name() {
        assert_eq!(optional("Baz").type_name(), "Baz");
    }

    #[test]
    fn test_input_optional_flag() {
        let i = input("x", optional("String"), true);
        assert!(i.optional);
    }

    #[test]
    fn test_operation_with_description() {
        let op = Operation {
            id: "op:desc".into(),
            name: "described".into(),
            description: Some("does something".into()),
            inputs: vec![],
            output: None,
            errors: vec![],
        };
        let mut reg = OperationRegistry::new();
        reg.register(op).expect("should succeed");
        let retrieved = reg.get("op:desc").expect("should succeed");
        assert_eq!(retrieved.description.as_deref(), Some("does something"));
    }

    #[test]
    fn test_operation_with_error_entities() {
        let op = Operation {
            id: "op:err".into(),
            name: "risky".into(),
            description: None,
            inputs: vec![],
            output: None,
            errors: vec!["urn:example:NotFound".into(), "urn:example:Timeout".into()],
        };
        let mut reg = OperationRegistry::new();
        reg.register(op).expect("should succeed");
        assert_eq!(reg.get("op:err").expect("should succeed").errors.len(), 2);
    }

    #[test]
    fn test_operation_with_collection_input() {
        let op = Operation {
            id: "op:col".into(),
            name: "batch".into(),
            description: None,
            inputs: vec![input("items", collection("Item"), false)],
            output: Some(output(single("Result"))),
            errors: vec![],
        };
        let mut reg = OperationRegistry::new();
        reg.register(op).expect("should succeed");
        let retrieved = reg.get("op:col").expect("should succeed");
        assert!(matches!(
            retrieved.inputs[0].input_type,
            InputOutputType::Collection(_)
        ));
    }

    #[test]
    fn test_registry_error_display() {
        let e = RegistryError::DuplicateId("op:x".into());
        assert!(e.to_string().contains("op:x"));
        let e2 = RegistryError::InvalidOperation("bad".into());
        assert!(e2.to_string().contains("bad"));
    }
}
