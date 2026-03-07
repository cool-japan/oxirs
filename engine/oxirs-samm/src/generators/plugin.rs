//! Plugin architecture for custom code generators
//!
//! This module provides a plugin system that allows users to register custom code generators
//! without modifying the oxirs-samm core library. Custom generators can be written in external
//! crates and registered at runtime.
//!
//! # Example
//!
//! ```rust
//! use oxirs_samm::generators::plugin::{CodeGenerator, GeneratorRegistry};
//! use oxirs_samm::metamodel::{Aspect, ModelElement};
//! use oxirs_samm::SammError;
//!
//! // Define a custom generator
//! struct MyCustomGenerator;
//!
//! impl CodeGenerator for MyCustomGenerator {
//!     fn name(&self) -> &str {
//!         "my-custom"
//!     }
//!
//!     fn description(&self) -> &str {
//!         "My custom code generator"
//!     }
//!
//!     fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
//!         Ok(format!("// Custom code for: {}", aspect.name()))
//!     }
//!
//!     fn file_extension(&self) -> &str {
//!         "custom"
//!     }
//! }
//!
//! // Register and use the generator
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut registry = GeneratorRegistry::new();
//! registry.register(Box::new(MyCustomGenerator));
//!
//! // Later, use the generator
//! # use oxirs_samm::metamodel::ElementMetadata;
//! # let aspect = Aspect::new("urn:samm:org.example:1.0.0#MyAspect".to_string());
//! if let Some(generator) = registry.get("my-custom") {
//!     let code = generator.generate(&aspect)?;
//!     println!("{}", code);
//! }
//! # Ok(())
//! # }
//! ```

use crate::metamodel::Aspect;
use crate::SammError;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Trait that all code generators must implement
///
/// Custom generators can implement this trait to provide code generation
/// for any target language or format.
pub trait CodeGenerator: Send + Sync {
    /// Returns the unique name of this generator
    ///
    /// This name is used to identify and retrieve the generator from the registry.
    fn name(&self) -> &str;

    /// Returns a human-readable description of this generator
    fn description(&self) -> &str;

    /// Generates code from an Aspect model
    ///
    /// # Arguments
    ///
    /// * `aspect` - The SAMM Aspect model to generate code from
    ///
    /// # Returns
    ///
    /// Generated code as a string, or an error if generation fails
    fn generate(&self, aspect: &Aspect) -> Result<String, SammError>;

    /// Returns the file extension for generated files (without the dot)
    ///
    /// # Example
    ///
    /// ```
    /// # struct MyGenerator;
    /// # impl oxirs_samm::generators::plugin::CodeGenerator for MyGenerator {
    /// #     fn name(&self) -> &str { "my-gen" }
    /// #     fn description(&self) -> &str { "My generator" }
    /// #     fn generate(&self, _: &oxirs_samm::metamodel::Aspect) -> Result<String, oxirs_samm::SammError> { Ok(String::new()) }
    ///     fn file_extension(&self) -> &str {
    ///         "ts" // TypeScript files
    ///     }
    /// # }
    /// ```
    fn file_extension(&self) -> &str;

    /// Optional: Returns the MIME type of the generated code
    ///
    /// Default implementation returns "text/plain"
    fn mime_type(&self) -> &str {
        "text/plain"
    }

    /// Optional: Returns generator metadata (version, author, etc.)
    ///
    /// Default implementation returns empty metadata
    fn metadata(&self) -> GeneratorMetadata {
        GeneratorMetadata::default()
    }

    /// Optional: Validates that this generator can process the given aspect
    ///
    /// Default implementation always returns Ok(())
    fn validate(&self, _aspect: &Aspect) -> Result<(), SammError> {
        Ok(())
    }
}

/// Metadata about a code generator
#[derive(Debug, Clone, Default)]
pub struct GeneratorMetadata {
    /// Generator version
    pub version: Option<String>,
    /// Generator author
    pub author: Option<String>,
    /// Generator license
    pub license: Option<String>,
    /// Generator homepage/repository
    pub homepage: Option<String>,
    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

// Built-in generator wrappers

/// TypeScript generator wrapper
struct TypeScriptGenerator;

impl CodeGenerator for TypeScriptGenerator {
    fn name(&self) -> &str {
        "typescript"
    }

    fn description(&self) -> &str {
        "TypeScript interface generator with type definitions"
    }

    fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        use crate::generators::typescript::{generate_typescript, TsOptions};
        generate_typescript(aspect, TsOptions::default())
    }

    fn file_extension(&self) -> &str {
        "ts"
    }

    fn mime_type(&self) -> &str {
        "text/typescript"
    }
}

/// Python generator wrapper
struct PythonGenerator;

impl CodeGenerator for PythonGenerator {
    fn name(&self) -> &str {
        "python"
    }

    fn description(&self) -> &str {
        "Python dataclass generator with type hints"
    }

    fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        use crate::generators::python::{generate_python, PythonOptions};
        generate_python(aspect, PythonOptions::default())
    }

    fn file_extension(&self) -> &str {
        "py"
    }

    fn mime_type(&self) -> &str {
        "text/x-python"
    }
}

/// Java generator wrapper
struct JavaGenerator;

impl CodeGenerator for JavaGenerator {
    fn name(&self) -> &str {
        "java"
    }

    fn description(&self) -> &str {
        "Java POJO generator with getters/setters"
    }

    fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        use crate::generators::java::{generate_java, JavaOptions};
        generate_java(aspect, JavaOptions::default())
    }

    fn file_extension(&self) -> &str {
        "java"
    }

    fn mime_type(&self) -> &str {
        "text/x-java-source"
    }
}

/// Scala generator wrapper
struct ScalaGenerator;

impl CodeGenerator for ScalaGenerator {
    fn name(&self) -> &str {
        "scala"
    }

    fn description(&self) -> &str {
        "Scala case class generator"
    }

    fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        use crate::generators::scala::{generate_scala, ScalaOptions};
        generate_scala(aspect, ScalaOptions::default())
    }

    fn file_extension(&self) -> &str {
        "scala"
    }

    fn mime_type(&self) -> &str {
        "text/x-scala"
    }
}

/// GraphQL generator wrapper
struct GraphQLGenerator;

impl CodeGenerator for GraphQLGenerator {
    fn name(&self) -> &str {
        "graphql"
    }

    fn description(&self) -> &str {
        "GraphQL schema generator with types and queries"
    }

    fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        use crate::generators::graphql::generate_graphql;
        generate_graphql(aspect)
    }

    fn file_extension(&self) -> &str {
        "graphql"
    }

    fn mime_type(&self) -> &str {
        "application/graphql"
    }
}

/// SQL generator wrapper
struct SqlGenerator;

impl CodeGenerator for SqlGenerator {
    fn name(&self) -> &str {
        "sql"
    }

    fn description(&self) -> &str {
        "SQL DDL generator for PostgreSQL"
    }

    fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        use crate::generators::sql::{generate_sql, SqlDialect};
        generate_sql(aspect, SqlDialect::PostgreSql)
    }

    fn file_extension(&self) -> &str {
        "sql"
    }

    fn mime_type(&self) -> &str {
        "application/sql"
    }
}

/// JSON-LD generator wrapper
struct JsonLdGenerator;

impl CodeGenerator for JsonLdGenerator {
    fn name(&self) -> &str {
        "jsonld"
    }

    fn description(&self) -> &str {
        "JSON-LD generator with semantic context"
    }

    fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        use crate::generators::jsonld::generate_jsonld;
        generate_jsonld(aspect)
    }

    fn file_extension(&self) -> &str {
        "jsonld"
    }

    fn mime_type(&self) -> &str {
        "application/ld+json"
    }
}

/// JSON Payload generator wrapper
struct PayloadGenerator;

impl CodeGenerator for PayloadGenerator {
    fn name(&self) -> &str {
        "payload"
    }

    fn description(&self) -> &str {
        "JSON payload generator with sample test data"
    }

    fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        use crate::generators::payload::generate_payload;
        generate_payload(aspect, true) // Generate with example values
    }

    fn file_extension(&self) -> &str {
        "json"
    }

    fn mime_type(&self) -> &str {
        "application/json"
    }
}

/// DTDL generator wrapper
struct DtdlGenerator;

impl CodeGenerator for DtdlGenerator {
    fn name(&self) -> &str {
        "dtdl"
    }

    fn description(&self) -> &str {
        "DTDL v3 generator for Azure Digital Twins"
    }

    fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        use crate::generators::dtdl::generate_dtdl;
        generate_dtdl(aspect)
    }

    fn file_extension(&self) -> &str {
        "json"
    }

    fn mime_type(&self) -> &str {
        "application/json"
    }

    fn metadata(&self) -> GeneratorMetadata {
        GeneratorMetadata {
            version: Some("3.0".to_string()),
            author: Some("OxiRS Team".to_string()),
            license: Some("MIT OR Apache-2.0".to_string()),
            homepage: Some("https://github.com/cool-japan/oxirs".to_string()),
            custom: {
                let mut custom = HashMap::new();
                custom.insert("dtdl_version".to_string(), "3".to_string());
                custom.insert("target".to_string(), "Azure Digital Twins".to_string());
                custom
            },
        }
    }
}

/// Registry for managing code generators
///
/// The registry allows registering custom generators and retrieving them by name.
/// It is thread-safe and can be shared across threads.
///
/// # Example
///
/// ```rust
/// use oxirs_samm::generators::plugin::{CodeGenerator, GeneratorRegistry};
/// use oxirs_samm::metamodel::{Aspect, ModelElement};
/// use oxirs_samm::SammError;
///
/// struct SimpleGenerator;
///
/// impl CodeGenerator for SimpleGenerator {
///     fn name(&self) -> &str { "simple" }
///     fn description(&self) -> &str { "A simple generator" }
///     fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
///         Ok(format!("// {}", aspect.name()))
///     }
///     fn file_extension(&self) -> &str { "txt" }
/// }
///
/// let mut registry = GeneratorRegistry::new();
/// registry.register(Box::new(SimpleGenerator));
///
/// assert!(registry.get("simple").is_some());
/// assert!(registry.get("nonexistent").is_none());
/// ```
pub struct GeneratorRegistry {
    generators: Arc<RwLock<HashMap<String, Box<dyn CodeGenerator>>>>,
}

impl GeneratorRegistry {
    /// Creates a new empty generator registry
    pub fn new() -> Self {
        Self {
            generators: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Creates a registry with all built-in generators
    ///
    /// This includes generators for TypeScript, Python, Java, Scala,
    /// GraphQL, SQL, Rust, and other built-in formats.
    pub fn with_builtin() -> Self {
        let registry = Self::new();
        registry.register_builtin();
        registry
    }

    /// Registers a custom code generator
    ///
    /// # Arguments
    ///
    /// * `generator` - The generator to register
    ///
    /// # Panics
    ///
    /// Panics if the generator name is empty or if the registry lock is poisoned
    pub fn register(&self, generator: Box<dyn CodeGenerator>) {
        let name = generator.name().to_string();
        assert!(!name.is_empty(), "Generator name cannot be empty");

        let mut generators = self
            .generators
            .write()
            .expect("write lock should not be poisoned");
        generators.insert(name, generator);
    }

    /// Retrieves a generator by name
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the generator to retrieve
    ///
    /// # Returns
    ///
    /// A reference to the generator if found, None otherwise
    pub fn get(&self, name: &str) -> Option<GeneratorRef> {
        let generators = self
            .generators
            .read()
            .expect("read lock should not be poisoned");
        if generators.contains_key(name) {
            Some(GeneratorRef {
                registry: Arc::clone(&self.generators),
                name: name.to_string(),
            })
        } else {
            None
        }
    }

    /// Removes a generator from the registry
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the generator to remove
    ///
    /// # Returns
    ///
    /// true if the generator was removed, false if it wasn't found
    pub fn remove(&self, name: &str) -> bool {
        let mut generators = self
            .generators
            .write()
            .expect("write lock should not be poisoned");
        generators.remove(name).is_some()
    }

    /// Lists all registered generator names
    pub fn list(&self) -> Vec<String> {
        let generators = self
            .generators
            .read()
            .expect("read lock should not be poisoned");
        generators.keys().cloned().collect()
    }

    /// Returns the number of registered generators
    pub fn count(&self) -> usize {
        let generators = self
            .generators
            .read()
            .expect("read lock should not be poisoned");
        generators.len()
    }

    /// Clears all generators from the registry
    pub fn clear(&self) {
        let mut generators = self
            .generators
            .write()
            .expect("write lock should not be poisoned");
        generators.clear();
    }

    /// Registers all built-in generators
    fn register_builtin(&self) {
        // Register all built-in code generators
        self.register(Box::new(TypeScriptGenerator));
        self.register(Box::new(PythonGenerator));
        self.register(Box::new(JavaGenerator));
        self.register(Box::new(ScalaGenerator));
        self.register(Box::new(GraphQLGenerator));
        self.register(Box::new(SqlGenerator));
        self.register(Box::new(JsonLdGenerator));
        self.register(Box::new(PayloadGenerator));
        self.register(Box::new(DtdlGenerator));
    }
}

impl Default for GeneratorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for GeneratorRegistry {
    fn clone(&self) -> Self {
        Self {
            generators: Arc::clone(&self.generators),
        }
    }
}

/// A reference to a generator in the registry
///
/// This allows safe access to generators across threads
pub struct GeneratorRef {
    registry: Arc<RwLock<HashMap<String, Box<dyn CodeGenerator>>>>,
    name: String,
}

impl GeneratorRef {
    /// Generates code using this generator
    ///
    /// # Arguments
    ///
    /// * `aspect` - The aspect to generate code from
    ///
    /// # Returns
    ///
    /// Generated code or an error
    pub fn generate(&self, aspect: &Aspect) -> Result<String, SammError> {
        let generators = self
            .registry
            .read()
            .expect("read lock should not be poisoned");
        if let Some(generator) = generators.get(&self.name) {
            generator.generate(aspect)
        } else {
            Err(SammError::ValidationError(format!(
                "Generator '{}' not found",
                self.name
            )))
        }
    }

    /// Returns the file extension for this generator
    pub fn file_extension(&self) -> String {
        let generators = self
            .registry
            .read()
            .expect("read lock should not be poisoned");
        if let Some(generator) = generators.get(&self.name) {
            generator.file_extension().to_string()
        } else {
            "txt".to_string()
        }
    }

    /// Returns the description of this generator
    pub fn description(&self) -> String {
        let generators = self
            .registry
            .read()
            .expect("read lock should not be poisoned");
        if let Some(generator) = generators.get(&self.name) {
            generator.description().to_string()
        } else {
            String::new()
        }
    }

    /// Returns the metadata of this generator
    pub fn metadata(&self) -> GeneratorMetadata {
        let generators = self
            .registry
            .read()
            .expect("read lock should not be poisoned");
        if let Some(generator) = generators.get(&self.name) {
            generator.metadata()
        } else {
            GeneratorMetadata::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::ElementMetadata;

    struct TestGenerator {
        name: String,
        output: String,
    }

    impl CodeGenerator for TestGenerator {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Test generator"
        }

        fn generate(&self, _aspect: &Aspect) -> Result<String, SammError> {
            Ok(self.output.clone())
        }

        fn file_extension(&self) -> &str {
            "test"
        }
    }

    #[test]
    fn test_registry_new() {
        let registry = GeneratorRegistry::new();
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_register_and_get() {
        let registry = GeneratorRegistry::new();
        let generator = Box::new(TestGenerator {
            name: "test-gen".to_string(),
            output: "test output".to_string(),
        });

        registry.register(generator);
        assert_eq!(registry.count(), 1);

        let gen_ref = registry.get("test-gen");
        assert!(gen_ref.is_some());
        assert_eq!(
            gen_ref.expect("operation should succeed").file_extension(),
            "test"
        );
    }

    #[test]
    fn test_generate_code() {
        let registry = GeneratorRegistry::new();
        registry.register(Box::new(TestGenerator {
            name: "test-gen".to_string(),
            output: "Hello, World!".to_string(),
        }));

        let aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());

        if let Some(gen_ref) = registry.get("test-gen") {
            let code = gen_ref
                .generate(&aspect)
                .expect("generation should succeed");
            assert_eq!(code, "Hello, World!");
        } else {
            panic!("Generator not found");
        }
    }

    #[test]
    fn test_remove_generator() {
        let registry = GeneratorRegistry::new();
        registry.register(Box::new(TestGenerator {
            name: "test-gen".to_string(),
            output: "test".to_string(),
        }));

        assert_eq!(registry.count(), 1);
        assert!(registry.remove("test-gen"));
        assert_eq!(registry.count(), 0);
        assert!(!registry.remove("test-gen"));
    }

    #[test]
    fn test_list_generators() {
        let registry = GeneratorRegistry::new();
        registry.register(Box::new(TestGenerator {
            name: "gen1".to_string(),
            output: "output1".to_string(),
        }));
        registry.register(Box::new(TestGenerator {
            name: "gen2".to_string(),
            output: "output2".to_string(),
        }));

        let list = registry.list();
        assert_eq!(list.len(), 2);
        assert!(list.contains(&"gen1".to_string()));
        assert!(list.contains(&"gen2".to_string()));
    }

    #[test]
    fn test_clear_registry() {
        let registry = GeneratorRegistry::new();
        registry.register(Box::new(TestGenerator {
            name: "gen1".to_string(),
            output: "output1".to_string(),
        }));
        registry.register(Box::new(TestGenerator {
            name: "gen2".to_string(),
            output: "output2".to_string(),
        }));

        assert_eq!(registry.count(), 2);
        registry.clear();
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_registry_clone() {
        let registry1 = GeneratorRegistry::new();
        registry1.register(Box::new(TestGenerator {
            name: "test-gen".to_string(),
            output: "test".to_string(),
        }));

        let registry2 = registry1.clone();
        assert_eq!(registry2.count(), 1);
        assert!(registry2.get("test-gen").is_some());
    }

    #[test]
    fn test_generator_metadata() {
        struct MetadataGenerator;

        impl CodeGenerator for MetadataGenerator {
            fn name(&self) -> &str {
                "meta-gen"
            }

            fn description(&self) -> &str {
                "Generator with metadata"
            }

            fn generate(&self, _aspect: &Aspect) -> Result<String, SammError> {
                Ok(String::new())
            }

            fn file_extension(&self) -> &str {
                "meta"
            }

            fn metadata(&self) -> GeneratorMetadata {
                GeneratorMetadata {
                    version: Some("1.0.0".to_string()),
                    author: Some("Test Author".to_string()),
                    license: Some("MIT".to_string()),
                    homepage: Some("https://example.com".to_string()),
                    custom: HashMap::new(),
                }
            }
        }

        let registry = GeneratorRegistry::new();
        registry.register(Box::new(MetadataGenerator));

        if let Some(gen_ref) = registry.get("meta-gen") {
            let metadata = gen_ref.metadata();
            assert_eq!(metadata.version, Some("1.0.0".to_string()));
            assert_eq!(metadata.author, Some("Test Author".to_string()));
            assert_eq!(metadata.license, Some("MIT".to_string()));
            assert_eq!(metadata.homepage, Some("https://example.com".to_string()));
        } else {
            panic!("Generator not found");
        }
    }

    #[test]
    fn test_registry_with_builtin() {
        let registry = GeneratorRegistry::with_builtin();

        // Should have all 9 built-in generators
        assert_eq!(registry.count(), 9);

        // Check that specific generators are present
        assert!(registry.get("typescript").is_some());
        assert!(registry.get("python").is_some());
        assert!(registry.get("java").is_some());
        assert!(registry.get("scala").is_some());
        assert!(registry.get("graphql").is_some());
        assert!(registry.get("sql").is_some());
        assert!(registry.get("jsonld").is_some());
        assert!(registry.get("payload").is_some());
        assert!(registry.get("dtdl").is_some());
    }

    #[test]
    fn test_builtin_generator_properties() {
        let registry = GeneratorRegistry::with_builtin();

        // Test TypeScript generator
        if let Some(ts_gen) = registry.get("typescript") {
            assert_eq!(ts_gen.file_extension(), "ts");
            assert_eq!(
                ts_gen.description(),
                "TypeScript interface generator with type definitions"
            );
        } else {
            panic!("TypeScript generator not found");
        }

        // Test Python generator
        if let Some(py_gen) = registry.get("python") {
            assert_eq!(py_gen.file_extension(), "py");
            assert_eq!(
                py_gen.description(),
                "Python dataclass generator with type hints"
            );
        } else {
            panic!("Python generator not found");
        }

        // Test GraphQL generator
        if let Some(gql_gen) = registry.get("graphql") {
            assert_eq!(gql_gen.file_extension(), "graphql");
            assert_eq!(
                gql_gen.description(),
                "GraphQL schema generator with types and queries"
            );
        } else {
            panic!("GraphQL generator not found");
        }
    }

    #[test]
    fn test_builtin_generator_code_generation() {
        let registry = GeneratorRegistry::with_builtin();
        let aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());

        // Test that generators can actually generate code
        if let Some(ts_gen) = registry.get("typescript") {
            let result = ts_gen.generate(&aspect);
            assert!(result.is_ok());
            let code = result.expect("result should be Ok");
            assert!(!code.is_empty());
            assert!(code.contains("TestAspect") || code.contains("interface"));
        }

        if let Some(py_gen) = registry.get("python") {
            let result = py_gen.generate(&aspect);
            assert!(result.is_ok());
            let code = result.expect("result should be Ok");
            assert!(!code.is_empty());
        }
    }
}
