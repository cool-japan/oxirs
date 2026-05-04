//! Parallel code generation for all SAMM generators using rayon.
//!
//! [`ParallelGenerator`] runs every built-in code generator concurrently over
//! a single [`Aspect`] and collects the results into a [`ParallelGeneratorResult`].
//! The result map is keyed by a stable `&'static str` name and sorted
//! deterministically via [`std::collections::BTreeMap`].
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_samm::metamodel::Aspect;
//! use oxirs_samm::generators::parallel::ParallelGenerator;
//!
//! let aspect = Aspect::new("urn:samm:org.example:1.0.0#Movement".to_string());
//! let result = ParallelGenerator::new().generate_all(&aspect);
//! for (name, output) in &result.outputs {
//!     match output {
//!         Ok(code) => println!("{}: {} bytes", name, code.len()),
//!         Err(e)   => eprintln!("{}: error – {}", name, e),
//!     }
//! }
//! ```

use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::metamodel::Aspect;

/// Type alias for the generator closure list to avoid clippy `type_complexity`.
type GeneratorEntry<'a> = (
    &'static str,
    Box<dyn Fn() -> Result<String, String> + Send + Sync + 'a>,
);

use super::{
    generate_aas, generate_dtdl, generate_graphql, generate_java, generate_jsonld,
    generate_payload, generate_python, generate_scala, generate_sql, generate_typescript,
    AasFormat, JavaOptions, PythonOptions, ScalaOptions, SqlDialect, TsOptions,
};

/// Output of [`ParallelGenerator::generate_all`].
///
/// `outputs` maps a stable generator name to either the generated source
/// string or a human-readable error message.  The map is sorted by key so
/// iteration order is deterministic across runs.
#[derive(Debug)]
pub struct ParallelGeneratorResult {
    /// Sorted map of generator name → generation outcome.
    pub outputs: BTreeMap<&'static str, Result<String, String>>,
}

impl ParallelGeneratorResult {
    /// Returns `true` when every generator produced output without errors.
    pub fn all_succeeded(&self) -> bool {
        self.outputs.values().all(|r| r.is_ok())
    }

    /// Returns the number of generators that produced an error.
    pub fn error_count(&self) -> usize {
        self.outputs.values().filter(|r| r.is_err()).count()
    }

    /// Returns the output for the named generator, if present.
    pub fn get(&self, name: &str) -> Option<&Result<String, String>> {
        self.outputs.get(name)
    }
}

/// Runs all SAMM generators in parallel and collects their results.
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_samm::metamodel::Aspect;
/// use oxirs_samm::generators::parallel::ParallelGenerator;
///
/// let aspect = Aspect::new("urn:samm:org.example:1.0.0#Foo".to_string());
/// let result = ParallelGenerator::new().generate_all(&aspect);
/// assert!(!result.outputs.is_empty());
/// ```
#[derive(Debug, Default)]
pub struct ParallelGenerator;

impl ParallelGenerator {
    /// Create a new `ParallelGenerator`.
    pub fn new() -> Self {
        Self
    }

    /// Run all built-in generators in parallel and return the collected results.
    ///
    /// Each generator is executed on a rayon worker thread.  Errors from
    /// individual generators are captured as `Err(String)` and do **not**
    /// cause this method to fail.
    pub fn generate_all(&self, aspect: &Aspect) -> ParallelGeneratorResult {
        // Build a list of (name, thunk) pairs.
        // The closures capture the aspect by reference; rayon requires `Send`,
        // which `Aspect` satisfies via its `Serialize`/`Clone` derives.
        let generators: Vec<GeneratorEntry<'_>> = vec![
            (
                "graphql",
                Box::new(|| generate_graphql(aspect).map_err(|e| e.to_string())),
            ),
            (
                "typescript",
                Box::new(|| {
                    generate_typescript(aspect, TsOptions::default()).map_err(|e| e.to_string())
                }),
            ),
            (
                "python",
                Box::new(|| {
                    generate_python(aspect, PythonOptions::default()).map_err(|e| e.to_string())
                }),
            ),
            (
                "java",
                Box::new(|| {
                    generate_java(aspect, JavaOptions::default()).map_err(|e| e.to_string())
                }),
            ),
            (
                "scala",
                Box::new(|| {
                    generate_scala(aspect, ScalaOptions::default()).map_err(|e| e.to_string())
                }),
            ),
            (
                "sql",
                Box::new(|| {
                    generate_sql(aspect, SqlDialect::PostgreSql).map_err(|e| e.to_string())
                }),
            ),
            (
                "jsonld",
                Box::new(|| generate_jsonld(aspect).map_err(|e| e.to_string())),
            ),
            (
                "payload",
                Box::new(|| generate_payload(aspect, false).map_err(|e| e.to_string())),
            ),
            (
                "dtdl",
                Box::new(|| generate_dtdl(aspect).map_err(|e| e.to_string())),
            ),
            (
                "aas",
                Box::new(|| generate_aas(aspect, AasFormat::Json).map_err(|e| e.to_string())),
            ),
        ];

        // Run all generators in parallel and collect into a BTreeMap.
        let outputs: BTreeMap<&'static str, Result<String, String>> = generators
            .into_par_iter()
            .map(|(name, gen)| (name, gen()))
            .collect();

        ParallelGeneratorResult { outputs }
    }

    /// Run a specific subset of generators sequentially and return the results.
    ///
    /// Useful when the caller wants only deterministic generators (i.e. those
    /// whose output does not depend on runtime randomness).
    pub fn generate_deterministic(&self, aspect: &Aspect) -> ParallelGeneratorResult {
        // Deterministic generators: all except `payload` (which uses an RNG).
        let generators: Vec<GeneratorEntry<'_>> = vec![
            (
                "graphql",
                Box::new(|| generate_graphql(aspect).map_err(|e| e.to_string())),
            ),
            (
                "typescript",
                Box::new(|| {
                    generate_typescript(aspect, TsOptions::default()).map_err(|e| e.to_string())
                }),
            ),
            (
                "python",
                Box::new(|| {
                    generate_python(aspect, PythonOptions::default()).map_err(|e| e.to_string())
                }),
            ),
            (
                "java",
                Box::new(|| {
                    generate_java(aspect, JavaOptions::default()).map_err(|e| e.to_string())
                }),
            ),
            (
                "scala",
                Box::new(|| {
                    generate_scala(aspect, ScalaOptions::default()).map_err(|e| e.to_string())
                }),
            ),
            (
                "sql",
                Box::new(|| {
                    generate_sql(aspect, SqlDialect::PostgreSql).map_err(|e| e.to_string())
                }),
            ),
            (
                "jsonld",
                Box::new(|| generate_jsonld(aspect).map_err(|e| e.to_string())),
            ),
            (
                "dtdl",
                Box::new(|| generate_dtdl(aspect).map_err(|e| e.to_string())),
            ),
            (
                "aas",
                Box::new(|| generate_aas(aspect, AasFormat::Json).map_err(|e| e.to_string())),
            ),
        ];

        let outputs: BTreeMap<&'static str, Result<String, String>> = generators
            .into_par_iter()
            .map(|(name, gen)| (name, gen()))
            .collect();

        ParallelGeneratorResult { outputs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Characteristic, CharacteristicKind, Property};

    fn minimal_aspect() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#Test".to_string());
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#TestChar".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
        let prop = Property::new("urn:samm:org.example:1.0.0#testProp".to_string())
            .with_characteristic(char);
        aspect.add_property(prop);
        aspect
    }

    #[test]
    fn test_parallel_generate_all_produces_ten_entries() {
        let aspect = minimal_aspect();
        let result = ParallelGenerator::new().generate_all(&aspect);
        // We register 10 generators
        assert_eq!(result.outputs.len(), 10);
    }

    #[test]
    fn test_parallel_generate_deterministic_nine_entries() {
        let aspect = minimal_aspect();
        let result = ParallelGenerator::new().generate_deterministic(&aspect);
        // payload excluded
        assert_eq!(result.outputs.len(), 9);
        assert!(!result.outputs.contains_key("payload"));
    }

    #[test]
    fn test_parallel_result_helpers() {
        let mut outputs = BTreeMap::new();
        outputs.insert("ok", Ok("code".to_string()));
        outputs.insert("fail", Err("oops".to_string()));
        let result = ParallelGeneratorResult { outputs };
        assert!(!result.all_succeeded());
        assert_eq!(result.error_count(), 1);
        assert!(result.get("ok").is_some());
        assert!(result.get("missing").is_none());
    }
}
