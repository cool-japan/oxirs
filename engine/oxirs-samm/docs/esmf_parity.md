# ESMF SDK 2.x Parity Report — oxirs-samm

> **Generated automatically** by `cargo run --bin parity_report`.
> Do not edit by hand — regenerate after updating `esmf_catalog.toml`.

---

## Summary

| Category | ✅ Implemented | ⚠️ Partial | ❌ Missing | Total |
|---|---|---|---|---|
| Aspect Modeling | 6 | 2 | 1 | 9 |
| Validation | 3 | 0 | 2 | 5 |
| Code Generation | 7 | 0 | 0 | 7 |
| OpenAPI Emission | 1 | 0 | 2 | 3 |
| JSON-LD Profiles | 1 | 1 | 1 | 3 |
| Model Resolution | 2 | 1 | 0 | 3 |
| Command-Line Tooling | 0 | 1 | 2 | 3 |
| **Total** | **20** | **5** | **8** | **33** |

## Detailed Parity Matrix

### Aspect Modeling

| Feature | Status | oxirs-samm Module | ESMF Reference | Notes |
|---|---|---|---|---|
| Aspect definition | ✅ Implemented | `metamodel::aspect` | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#aspect) | Core Aspect element with properties, operations and events fully modelled. |
| Property (mandatory and optional) | ✅ Implemented | `metamodel::property` | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#property) | Property with optional flag, characteristic binding and example values. |
| Operation | ✅ Implemented | `metamodel::operation` | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#operation) | Operation with typed input and output parameters. |
| Event | ✅ Implemented | `event_model` | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#event) | Event element with parameter list; lifecycle hooks are partial. |
| Characteristic definition (Trait) | ✅ Implemented | `metamodel::characteristic` | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#characteristics) | Characteristic trait hierarchy including Measurement, Enumeration, Collection, etc. |
| Entity (complex structured type) | ✅ Implemented | `metamodel::entity` | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#entity) | Entity with extends-hierarchy and nested property lists. |
| Abstract entity / entity inheritance | ⚠️ Partial | — | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#abstract-entity) | Single-level inheritance supported; multi-level abstract chains not fully validated. |
| Constraint types (Range, Length, RegularExpression, Encoding, FixedPoint) | ⚠️ Partial | `constraint_validator` | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#constraints) | Range, Length, and RegularExpression constraints implemented; FixedPoint and Encoding constraints are partial. |
| Either characteristic | ❌ Missing | — | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#either-characteristic) | Union-type Either characteristic not yet modelled in the Rust meta-model. |

### Validation

| Feature | Status | oxirs-samm Module | ESMF Reference | Notes |
|---|---|---|---|---|
| Turtle model syntactic validation | ✅ Implemented | `validator` | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#validation) | Validates Turtle file syntax via the integrated oxttl parser. |
| SAMM SHACL shape validation | ✅ Implemented | `validator::shacl_validator` | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#model-validation) | Validates aspect models against SAMM SHACL shapes via oxirs-shacl integration. |
| Semantic violation detection (cyclic dependencies) | ✅ Implemented | `aspect_validator` | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#model-validation) | Cycle detection, required-property checks, and cardinality validation. |
| Cross-model reference validation | ❌ Missing | — | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#cross-model-reference) | External URN references across independently loaded model files not validated. |
| Batch validation with progress reporting | ✅ Implemented | `validation::batch` | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#validation) | Parallel batch validation with optional GPU acceleration. |

### Code Generation

| Feature | Status | oxirs-samm Module | ESMF Reference | Notes |
|---|---|---|---|---|
| Java POJO generation | ✅ Implemented | `generators::java` | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#code-generation) | Generates Java POJO classes with Lombok annotations from aspect models. |
| TypeScript interface generation | ✅ Implemented | `generators::typescript` | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#code-generation) | Generates TypeScript interfaces including optional field handling. |
| Python dataclass generation | ✅ Implemented | `generators::python` | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#code-generation) | Generates Python 3.10+ dataclasses with type hints. |
| JSON Schema generation | ✅ Implemented | `codegen::json_schema` | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#code-generation) | Emits JSON Schema draft-07; additional drafts are not yet supported. |
| Static payload example generation | ✅ Implemented | `generators::payload` | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#code-generation) | Generates sample JSON payloads from characteristic types and example values. |
| Scala case-class generation | ✅ Implemented | `generators::scala` | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#code-generation) | Generates Scala 3 case classes; Scala 2 compatibility not guaranteed. |
| SQL DDL generation | ✅ Implemented | `generators::sql` | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#code-generation) | PostgreSQL and SQLite DDL; Oracle/MSSQL dialects are missing. |

### OpenAPI Emission

| Feature | Status | oxirs-samm Module | ESMF Reference | Notes |
|---|---|---|---|---|
| OpenAPI 3.0 schema generation | ✅ Implemented | `codegen::openapi` | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#open-api-generation) | Generates OpenAPI 3.0.x YAML/JSON specs from aspect model definitions. |
| OpenAPI 3.1 schema generation | ❌ Missing | — | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#open-api-generation) | OpenAPI 3.1 format with JSON Schema 2020-12 alignment not yet implemented. |
| Pagination extension in OpenAPI output | ❌ Missing | — | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#open-api-generation) | SAMM pagination extension (x-samm-pagination) blocks not emitted. |

### JSON-LD Profiles

| Feature | Status | oxirs-samm Module | ESMF Reference | Notes |
|---|---|---|---|---|
| JSON-LD context generation | ✅ Implemented | `generators::jsonld` | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#mapping-to-json-ld) | Emits a JSON-LD 1.1 @context document mapping aspect properties to IRIs. |
| JSON-LD compaction / framing | ❌ Missing | — | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#mapping-to-json-ld) | JSON-LD compaction algorithm and framing API not implemented. |
| RDF/JSON-LD serialisation round-trip | ⚠️ Partial | — | [spec](https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#mapping-to-json-ld) | Serialization implemented; lossless round-trip from parsed RDF back to JSON-LD is untested. |

### Model Resolution

| Feature | Status | oxirs-samm Module | ESMF Reference | Notes |
|---|---|---|---|---|
| file: URI and local filesystem model loading | ✅ Implemented | `parser::resolver` | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#loading-models) | Resolves urn:samm: identifiers to local .ttl files via configurable base path. |
| External HTTP model fetching | ⚠️ Partial | — | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#loading-models) | HTTP fetch via reqwest implemented; caching and ETags not yet wired. |
| Model version migration (SAMM 1.x → 2.x) | ✅ Implemented | `migration` | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#migration) | Automated migration from BAMM 1.x / SAMM 2.x older minor versions. |

### Command-Line Tooling

| Feature | Status | oxirs-samm Module | ESMF Reference | Notes |
|---|---|---|---|---|
| validate command | ⚠️ Partial | — | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/samm-cli.html#validate) | Basic validation output available; structured JSON violation reports not yet emitted. |
| generate command (Java / TypeScript / Python) | ❌ Missing | — | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/samm-cli.html#generate) | No oxirs CLI binary wires the code generators to a generate sub-command yet. |
| aspect command (list / describe) | ❌ Missing | — | [spec](https://eclipse-esmf.github.io/esmf-sdk/2.9.7/samm-cli.html#aspect) | Introspection sub-command for listing properties / operations not implemented. |

---

*Report generated by `oxirs-samm` — ESMF SDK 2.x parity matrix.*
