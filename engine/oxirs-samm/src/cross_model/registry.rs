//! Cross-model registry: load and index multiple SAMM model namespaces.
//!
//! A [`CrossModelRegistry`] holds a collection of [`ModelEntry`] values —
//! one per loaded SAMM model file (or programmatically constructed model).
//! It maintains a flat URN index so that any individual URN (type, property,
//! characteristic, …) can be resolved back to the owning model in O(1).

use std::collections::HashMap;

use super::CrossModelError;

/// A single SAMM model loaded into the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntry {
    /// The model namespace, e.g. `urn:samm:org.example:1.0.0`.
    ///
    /// This acts as the primary key in the registry.
    pub namespace: String,

    /// Path to the Turtle file this model was loaded from, or `None` for
    /// programmatically-constructed models.
    pub file_path: Option<String>,

    /// All URNs (types, properties, characteristics, …) exported by this model.
    ///
    /// Every URN here is indexed in the flat URN → namespace map so that
    /// [`CrossModelRegistry::resolve_urn`] runs in O(1).
    pub exported_urns: Vec<String>,
}

/// Registry holding multiple loaded SAMM model namespaces and their exported URNs.
///
/// Use [`register_model`](CrossModelRegistry::register_model) to add models and
/// [`resolve_urn`](CrossModelRegistry::resolve_urn) to look up individual URNs.
#[derive(Debug, Clone, Default)]
pub struct CrossModelRegistry {
    /// namespace → ModelEntry
    models: HashMap<String, ModelEntry>,

    /// urn → namespace (owning model)
    urn_index: HashMap<String, String>,
}

impl CrossModelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new model.
    ///
    /// Returns [`CrossModelError::EmptyNamespace`] if the entry's namespace is
    /// empty, [`CrossModelError::DuplicateNamespace`] if the namespace was
    /// already registered, or [`CrossModelError::DuplicateUrn`] if any of the
    /// exported URNs collide with a URN already registered under a different
    /// namespace.
    pub fn register_model(&mut self, entry: ModelEntry) -> Result<(), CrossModelError> {
        if entry.namespace.is_empty() {
            return Err(CrossModelError::EmptyNamespace);
        }

        if self.models.contains_key(&entry.namespace) {
            return Err(CrossModelError::DuplicateNamespace(entry.namespace));
        }

        // Check for URN collisions before mutating state.
        for urn in &entry.exported_urns {
            if let Some(existing_ns) = self.urn_index.get(urn.as_str()) {
                if existing_ns != &entry.namespace {
                    return Err(CrossModelError::DuplicateUrn {
                        urn: urn.clone(),
                        existing_namespace: existing_ns.clone(),
                    });
                }
            }
        }

        // Commit: index every exported URN.
        for urn in &entry.exported_urns {
            self.urn_index.insert(urn.clone(), entry.namespace.clone());
        }

        self.models.insert(entry.namespace.clone(), entry);
        Ok(())
    }

    /// Look up the [`ModelEntry`] that owns `urn`.
    ///
    /// Returns `None` if no registered model exports this URN.
    pub fn resolve_urn(&self, urn: &str) -> Option<&ModelEntry> {
        let namespace = self.urn_index.get(urn)?;
        self.models.get(namespace.as_str())
    }

    /// Return all registered namespaces in arbitrary order.
    pub fn all_namespaces(&self) -> Vec<&str> {
        self.models.keys().map(String::as_str).collect()
    }

    /// Return all URNs exported by the model with the given namespace.
    ///
    /// Returns an empty `Vec` if the namespace is not registered.
    pub fn urns_in_namespace(&self, namespace: &str) -> Vec<&str> {
        match self.models.get(namespace) {
            Some(entry) => entry.exported_urns.iter().map(String::as_str).collect(),
            None => Vec::new(),
        }
    }

    /// Remove a namespace and its URNs from the registry.
    ///
    /// Returns the [`ModelEntry`] that was removed, or `None` if the namespace
    /// was not registered.
    pub fn remove_namespace(&mut self, namespace: &str) -> Option<ModelEntry> {
        let entry = self.models.remove(namespace)?;
        for urn in &entry.exported_urns {
            self.urn_index.remove(urn.as_str());
        }
        Some(entry)
    }

    /// Number of models currently in the registry.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// `true` iff the registry holds no models.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(namespace: &str, urns: &[&str]) -> ModelEntry {
        ModelEntry {
            namespace: namespace.to_string(),
            file_path: None,
            exported_urns: urns.iter().map(|u| u.to_string()).collect(),
        }
    }

    // 1 — register a model and resolve one of its URNs.
    #[test]
    fn test_register_model() {
        let mut registry = CrossModelRegistry::new();
        let entry = make_entry(
            "urn:samm:org.example:1.0.0",
            &[
                "urn:samm:org.example:1.0.0#Temperature",
                "urn:samm:org.example:1.0.0#SpeedChar",
            ],
        );
        registry.register_model(entry).unwrap();

        let resolved = registry.resolve_urn("urn:samm:org.example:1.0.0#Temperature");
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap().namespace, "urn:samm:org.example:1.0.0");
    }

    // 2 — registering the same namespace twice returns DuplicateNamespace.
    #[test]
    fn test_duplicate_namespace() {
        let mut registry = CrossModelRegistry::new();
        let entry_a = make_entry(
            "urn:samm:org.example:1.0.0",
            &["urn:samm:org.example:1.0.0#Foo"],
        );
        let entry_b = make_entry(
            "urn:samm:org.example:1.0.0",
            &["urn:samm:org.example:1.0.0#Bar"],
        );

        registry.register_model(entry_a).unwrap();
        let err = registry.register_model(entry_b).unwrap_err();

        assert_eq!(
            err,
            CrossModelError::DuplicateNamespace("urn:samm:org.example:1.0.0".to_string())
        );
    }

    // 3 — same URN in two different models returns DuplicateUrn.
    #[test]
    fn test_duplicate_urn() {
        let mut registry = CrossModelRegistry::new();
        let shared_urn = "urn:samm:org.shared:1.0.0#CommonType";
        registry
            .register_model(make_entry("urn:samm:org.alpha:1.0.0", &[shared_urn]))
            .unwrap();

        let err = registry
            .register_model(make_entry("urn:samm:org.beta:1.0.0", &[shared_urn]))
            .unwrap_err();

        match err {
            CrossModelError::DuplicateUrn {
                urn,
                existing_namespace,
            } => {
                assert_eq!(urn, shared_urn);
                assert_eq!(existing_namespace, "urn:samm:org.alpha:1.0.0");
            }
            other => panic!("expected DuplicateUrn, got {other:?}"),
        }
    }

    // 4 — returns None for an unknown URN.
    #[test]
    fn test_resolve_unknown_urn() {
        let registry = CrossModelRegistry::new();
        assert!(registry
            .resolve_urn("urn:samm:org.unknown:1.0.0#Ghost")
            .is_none());
    }

    // 5 — empty registry has len 0 and is_empty.
    #[test]
    fn test_len_empty() {
        let registry = CrossModelRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    // 6 — returns correct namespaces after multiple registers.
    #[test]
    fn test_all_namespaces() {
        let mut registry = CrossModelRegistry::new();
        registry
            .register_model(make_entry("urn:samm:org.a:1.0.0", &[]))
            .unwrap();
        registry
            .register_model(make_entry("urn:samm:org.b:1.0.0", &[]))
            .unwrap();
        registry
            .register_model(make_entry("urn:samm:org.c:2.0.0", &[]))
            .unwrap();

        let mut ns = registry.all_namespaces();
        ns.sort_unstable();
        assert_eq!(
            ns,
            vec![
                "urn:samm:org.a:1.0.0",
                "urn:samm:org.b:1.0.0",
                "urn:samm:org.c:2.0.0",
            ]
        );
    }

    // 7 — after remove, URN no longer resolves.
    #[test]
    fn test_remove_namespace() {
        let mut registry = CrossModelRegistry::new();
        let ns = "urn:samm:org.removable:1.0.0";
        let urn = "urn:samm:org.removable:1.0.0#Ephemeral";

        registry.register_model(make_entry(ns, &[urn])).unwrap();
        assert!(registry.resolve_urn(urn).is_some());

        let removed = registry.remove_namespace(ns);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().namespace, ns);

        // URN must no longer resolve.
        assert!(registry.resolve_urn(urn).is_none());
        // Registry must be empty.
        assert!(registry.is_empty());
    }
}
