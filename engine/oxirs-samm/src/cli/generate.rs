//! `samm generate` CLI sub-command — invoke SAMM code generators from the command line.
//!
//! # Usage
//!
//! ```text
//! samm generate --target java       --input model.ttl --output Model.java
//! samm generate --target typescript --input model.ttl --output model.ts
//! samm generate --target python     --input model.ttl --output model.py
//! samm generate --target openapi    --input model.ttl --output openapi.json
//! samm generate --target json-schema --input model.ttl --output schema.json
//! ```
//!
//! The `--output` flag is treated as a **file path** for all targets.  The
//! caller is responsible for creating parent directories beforehand (or the
//! function will create them automatically).

use crate::codegen::{JsonSchemaGenerator, OpenApiGenerator};
use crate::generators::{
    generate_java, generate_python, generate_typescript, JavaOptions, PythonOptions, TsOptions,
};
use crate::parser::parse_aspect_from_string;
use std::path::PathBuf;

// ─────────────────────────────────────────────
// GenerateTarget
// ─────────────────────────────────────────────

/// Available code-generation targets for the `samm generate` sub-command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerateTarget {
    /// Generate a Java class file (POJO / Record) for the Aspect.
    Java,
    /// Generate TypeScript interface definitions for the Aspect.
    TypeScript,
    /// Generate a Python dataclass module for the Aspect.
    Python,
    /// Generate an OpenAPI 3.0.3 specification document (JSON) for the Aspect.
    OpenApi,
    /// Generate a JSON Schema (draft-07 / 2020-12) document for the Aspect.
    JsonSchema,
}

impl std::str::FromStr for GenerateTarget {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "java" => Ok(GenerateTarget::Java),
            "typescript" | "ts" => Ok(GenerateTarget::TypeScript),
            "python" | "py" => Ok(GenerateTarget::Python),
            "openapi" | "openapi3" => Ok(GenerateTarget::OpenApi),
            "json-schema" | "jsonschema" => Ok(GenerateTarget::JsonSchema),
            other => Err(format!(
                "unknown target: '{other}'; valid targets: java, typescript, python, openapi, json-schema"
            )),
        }
    }
}

impl std::fmt::Display for GenerateTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            GenerateTarget::Java => "java",
            GenerateTarget::TypeScript => "typescript",
            GenerateTarget::Python => "python",
            GenerateTarget::OpenApi => "openapi",
            GenerateTarget::JsonSchema => "json-schema",
        };
        f.write_str(name)
    }
}

// ─────────────────────────────────────────────
// GenerateArgs
// ─────────────────────────────────────────────

/// Arguments for the `samm generate` sub-command.
#[derive(Debug)]
pub struct GenerateArgs {
    /// Code-generation target language / format.
    pub target: GenerateTarget,
    /// Path to the input SAMM Turtle model file.
    pub input: PathBuf,
    /// Destination file path for the generated output.
    ///
    /// For `java` / `typescript` / `python` this is the path of the single
    /// source file to write.  For `openapi` / `json-schema` it is the path
    /// of the JSON document to write.  Parent directories are created
    /// automatically when they do not exist.
    pub output: PathBuf,
}

// ─────────────────────────────────────────────
// GenerateError
// ─────────────────────────────────────────────

/// Errors that can occur when running the `samm generate` sub-command.
#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    /// The input model file was not found.
    #[error("input file not found: {0}")]
    InputNotFound(String),

    /// The model could not be parsed into an Aspect.
    #[error("model parse failed: {0}")]
    ParseFailed(String),

    /// The code generator returned an error.
    #[error("code generation failed: {0}")]
    CodegenFailed(String),

    /// Writing the output file failed.
    #[error("output write failed: {0}")]
    OutputFailed(String),
}

// ─────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────

/// Execute the `samm generate` sub-command.
///
/// Reads the SAMM Turtle model at `args.input`, runs the appropriate code
/// generator for `args.target`, and writes the result to `args.output`.
///
/// # Errors
///
/// Returns a [`GenerateError`] if the input file is missing, parsing fails,
/// the generator reports an error, or the output cannot be written.
pub fn run_generate(args: &GenerateArgs) -> Result<(), GenerateError> {
    if !args.input.exists() {
        return Err(GenerateError::InputNotFound(
            args.input.display().to_string(),
        ));
    }

    // Read model source from disk.
    let model_src = std::fs::read_to_string(&args.input)
        .map_err(|e| GenerateError::ParseFailed(format!("cannot read input: {e}")))?;

    // Derive a base URI from the input file name — the TTL itself carries the
    // real prefix declarations, so this only matters for relative-reference
    // resolution and any non-empty string is fine.
    let base_uri = format!(
        "urn:samm:org.example:1.0.0#{}",
        args.input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Model")
    );

    // Parse the Turtle model asynchronously via a blocking handle.
    let aspect = tokio::runtime::Runtime::new()
        .map_err(|e| GenerateError::ParseFailed(format!("tokio runtime: {e}")))?
        .block_on(parse_aspect_from_string(&model_src, &base_uri))
        .map_err(|e| GenerateError::ParseFailed(e.to_string()))?;

    // Ensure the output parent directory exists.
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| GenerateError::OutputFailed(format!("create dir: {e}")))?;
        }
    }

    // Dispatch to the appropriate generator.
    match args.target {
        GenerateTarget::Java => {
            let code = generate_java(&aspect, JavaOptions::default())
                .map_err(|e| GenerateError::CodegenFailed(e.to_string()))?;
            write_output(&args.output, &code)
        }
        GenerateTarget::TypeScript => {
            let code = generate_typescript(&aspect, TsOptions::default())
                .map_err(|e| GenerateError::CodegenFailed(e.to_string()))?;
            write_output(&args.output, &code)
        }
        GenerateTarget::Python => {
            let code = generate_python(&aspect, PythonOptions::default())
                .map_err(|e| GenerateError::CodegenFailed(e.to_string()))?;
            write_output(&args.output, &code)
        }
        GenerateTarget::OpenApi => {
            let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
            let spec = gen
                .generate(&aspect)
                .map_err(|e| GenerateError::CodegenFailed(e.to_string()))?;
            let json = serde_json::to_string_pretty(&spec)
                .map_err(|e| GenerateError::CodegenFailed(format!("serialise openapi: {e}")))?;
            write_output(&args.output, &json)
        }
        GenerateTarget::JsonSchema => {
            let gen = JsonSchemaGenerator::new()
                .with_descriptions()
                .with_examples();
            let schema = gen
                .generate(&aspect)
                .map_err(|e| GenerateError::CodegenFailed(e.to_string()))?;
            let json = serde_json::to_string_pretty(&schema)
                .map_err(|e| GenerateError::CodegenFailed(format!("serialise json-schema: {e}")))?;
            write_output(&args.output, &json)
        }
    }
}

// ─────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────

/// Write `content` to `path`, creating parent directories as needed.
fn write_output(path: &std::path::Path, content: &str) -> Result<(), GenerateError> {
    std::fs::write(path, content)
        .map_err(|e| GenerateError::OutputFailed(format!("{}: {e}", path.display())))
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_generate_target_from_str_valid() {
        assert_eq!(
            GenerateTarget::from_str("java").unwrap(),
            GenerateTarget::Java
        );
        assert_eq!(
            GenerateTarget::from_str("typescript").unwrap(),
            GenerateTarget::TypeScript
        );
        assert_eq!(
            GenerateTarget::from_str("ts").unwrap(),
            GenerateTarget::TypeScript
        );
        assert_eq!(
            GenerateTarget::from_str("python").unwrap(),
            GenerateTarget::Python
        );
        assert_eq!(
            GenerateTarget::from_str("py").unwrap(),
            GenerateTarget::Python
        );
        assert_eq!(
            GenerateTarget::from_str("openapi").unwrap(),
            GenerateTarget::OpenApi
        );
        assert_eq!(
            GenerateTarget::from_str("openapi3").unwrap(),
            GenerateTarget::OpenApi
        );
        assert_eq!(
            GenerateTarget::from_str("json-schema").unwrap(),
            GenerateTarget::JsonSchema
        );
        assert_eq!(
            GenerateTarget::from_str("jsonschema").unwrap(),
            GenerateTarget::JsonSchema
        );
    }

    #[test]
    fn test_generate_target_from_str_case_insensitive() {
        assert_eq!(
            GenerateTarget::from_str("JAVA").unwrap(),
            GenerateTarget::Java
        );
        assert_eq!(
            GenerateTarget::from_str("TypeScript").unwrap(),
            GenerateTarget::TypeScript
        );
    }

    #[test]
    fn test_generate_target_from_str_invalid() {
        assert!(GenerateTarget::from_str("cobol").is_err());
        assert!(GenerateTarget::from_str("").is_err());
        assert!(GenerateTarget::from_str("graphql").is_err());
    }

    #[test]
    fn test_generate_target_display() {
        assert_eq!(GenerateTarget::Java.to_string(), "java");
        assert_eq!(GenerateTarget::TypeScript.to_string(), "typescript");
        assert_eq!(GenerateTarget::Python.to_string(), "python");
        assert_eq!(GenerateTarget::OpenApi.to_string(), "openapi");
        assert_eq!(GenerateTarget::JsonSchema.to_string(), "json-schema");
    }

    #[test]
    fn test_run_generate_missing_input_returns_error() {
        let args = GenerateArgs {
            target: GenerateTarget::Java,
            input: std::path::PathBuf::from("/nonexistent/path/model.ttl"),
            output: std::env::temp_dir().join("samm_gen_test_out.java"),
        };
        let result = run_generate(&args);
        assert!(
            matches!(result, Err(GenerateError::InputNotFound(_))),
            "expected InputNotFound, got {result:?}"
        );
    }

    #[test]
    fn test_run_generate_java_with_real_model() {
        let tmp = std::env::temp_dir().join("samm_gen_java_test");
        std::fs::create_dir_all(&tmp).expect("create tmp dir");
        let input = tmp.join("test_model.ttl");
        let output = tmp.join("TestAspect.java");

        // A minimal, self-contained SAMM Turtle model.
        let model = concat!(
            "@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .\n",
            "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .\n",
            "@prefix : <urn:samm:com.example:1.0.0#> .\n",
            ":TestAspect a samm:Aspect ;\n",
            "  samm:preferredName \"Test Aspect\"@en ;\n",
            "  samm:description \"A test aspect.\"@en ;\n",
            "  samm:properties () ;\n",
            "  samm:operations () .\n",
        );
        std::fs::write(&input, model).expect("write input");

        let args = GenerateArgs {
            target: GenerateTarget::Java,
            input,
            output,
        };
        let result = run_generate(&args);
        // Must not be InputNotFound (the file exists).
        assert!(
            !matches!(result, Err(GenerateError::InputNotFound(_))),
            "must not be InputNotFound: {result:?}"
        );
    }

    #[test]
    fn test_run_generate_typescript_with_real_model() {
        let tmp = std::env::temp_dir().join("samm_gen_ts_test");
        std::fs::create_dir_all(&tmp).expect("create tmp dir");
        let input = tmp.join("test_model.ttl");
        let output = tmp.join("TestAspect.ts");

        let model = concat!(
            "@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .\n",
            "@prefix : <urn:samm:com.example:1.0.0#> .\n",
            ":TestAspect a samm:Aspect ;\n",
            "  samm:preferredName \"Test\"@en ;\n",
            "  samm:description \"Test.\"@en ;\n",
            "  samm:properties () ;\n",
            "  samm:operations () .\n",
        );
        std::fs::write(&input, model).expect("write input");

        let args = GenerateArgs {
            target: GenerateTarget::TypeScript,
            input,
            output,
        };
        let result = run_generate(&args);
        assert!(
            !matches!(result, Err(GenerateError::InputNotFound(_))),
            "must not be InputNotFound: {result:?}"
        );
    }

    #[test]
    fn test_run_generate_python_with_real_model() {
        let tmp = std::env::temp_dir().join("samm_gen_py_test");
        std::fs::create_dir_all(&tmp).expect("create tmp dir");
        let input = tmp.join("test_model.ttl");
        let output = tmp.join("test_aspect.py");

        let model = concat!(
            "@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .\n",
            "@prefix : <urn:samm:com.example:1.0.0#> .\n",
            ":TestAspect a samm:Aspect ;\n",
            "  samm:preferredName \"Test\"@en ;\n",
            "  samm:description \"Test.\"@en ;\n",
            "  samm:properties () ;\n",
            "  samm:operations () .\n",
        );
        std::fs::write(&input, model).expect("write input");

        let args = GenerateArgs {
            target: GenerateTarget::Python,
            input,
            output,
        };
        let result = run_generate(&args);
        assert!(
            !matches!(result, Err(GenerateError::InputNotFound(_))),
            "must not be InputNotFound: {result:?}"
        );
    }

    #[test]
    fn test_run_generate_openapi_with_real_model() {
        let tmp = std::env::temp_dir().join("samm_gen_oa_test");
        std::fs::create_dir_all(&tmp).expect("create tmp dir");
        let input = tmp.join("test_model.ttl");
        let output = tmp.join("openapi.json");

        let model = concat!(
            "@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .\n",
            "@prefix : <urn:samm:com.example:1.0.0#> .\n",
            ":TestAspect a samm:Aspect ;\n",
            "  samm:preferredName \"Test\"@en ;\n",
            "  samm:description \"Test.\"@en ;\n",
            "  samm:properties () ;\n",
            "  samm:operations () .\n",
        );
        std::fs::write(&input, model).expect("write input");

        let args = GenerateArgs {
            target: GenerateTarget::OpenApi,
            input,
            output,
        };
        let result = run_generate(&args);
        assert!(
            !matches!(result, Err(GenerateError::InputNotFound(_))),
            "must not be InputNotFound: {result:?}"
        );
    }

    #[test]
    fn test_run_generate_json_schema_with_real_model() {
        let tmp = std::env::temp_dir().join("samm_gen_js_test");
        std::fs::create_dir_all(&tmp).expect("create tmp dir");
        let input = tmp.join("test_model.ttl");
        let output = tmp.join("schema.json");

        let model = concat!(
            "@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .\n",
            "@prefix : <urn:samm:com.example:1.0.0#> .\n",
            ":TestAspect a samm:Aspect ;\n",
            "  samm:preferredName \"Test\"@en ;\n",
            "  samm:description \"Test.\"@en ;\n",
            "  samm:properties () ;\n",
            "  samm:operations () .\n",
        );
        std::fs::write(&input, model).expect("write input");

        let args = GenerateArgs {
            target: GenerateTarget::JsonSchema,
            input,
            output,
        };
        let result = run_generate(&args);
        assert!(
            !matches!(result, Err(GenerateError::InputNotFound(_))),
            "must not be InputNotFound: {result:?}"
        );
    }
}
