# OxiRS SAMM - API Stability Guarantees

**Version**: 0.1.0
**Status**: Production Release
**Date**: 2026-01-06

## Overview

This document outlines the API stability guarantees for the OxiRS SAMM crate. As a production release, we commit to maintaining backward compatibility for all public APIs unless explicitly marked as unstable.

## Stability Levels

### Stable APIs (100% Coverage)

These APIs are **guaranteed to remain backward compatible** through the 0.1.x release series:

#### Core Metamodel Types

All public types in `oxirs_samm::metamodel` are **stable**:

- `Aspect` - Root element for SAMM models
- `Property` - Named feature with characteristics
- `Characteristic` - Semantic description of property values
- `Entity` - Complex data structure
- `Operation` - Aspect operations
- `Event` - Aspect events
- `CharacteristicKind` - Enumeration of characteristic types
- `ModelElement` - Trait for all model elements
- `ElementMetadata` - Metadata for model elements

**Guarantees**:
- Struct field names and types will not change
- Public methods will maintain their signatures
- New fields may be added but will have sensible defaults
- Serialization format (JSON, RDF) will remain compatible

#### Parser API

The `oxirs_samm::parser` module provides **stable** parsing capabilities:

- `parse_aspect_model(&str)` → `Result<Aspect>` - Parse from file path
- `SammTurtleParser` - Low-level RDF/Turtle parser
- `ModelResolver` - URN resolution system
  - `resolve_urn(&str)` → `Result<PathBuf>`
  - `load_element(&str)` → `Result<String>`
  - `add_models_root(PathBuf)`
  - `add_remote_base(String)` - HTTP/HTTPS resolution
  - `set_http_timeout(u64)`

**Guarantees**:
- Parse functions will continue to accept the same input formats
- URN resolution algorithm will remain compatible with SAMM 2.3.0 specification
- New parsing features will be additive only
- Error types may be enhanced but existing variants will remain

#### Validator API

The `oxirs_samm::validator` module provides **stable** validation:

- `validate_aspect(&Aspect)` → `Result<ValidationResult>`
- `ShaclValidator` - SHACL-based validation
- `ValidationResult` - Validation outcome with errors
- `ValidationError` - Individual validation error

**Guarantees**:
- Validation rules will only become more strict (not more lenient)
- New validation checks will be opt-in when possible
- Error messages may improve but structure will remain

#### Code Generators

All generators in `oxirs_samm::generators` are **stable**:

- `generate_rust(&Aspect, RustOptions)` → `Result<String>`
- `generate_typescript(&Aspect, TsOptions)` → `Result<String>`
- `generate_python(&Aspect, PyOptions)` → `Result<String>`
- `generate_java(&Aspect, JavaOptions)` → `Result<String>`
- `generate_scala(&Aspect, ScalaOptions)` → `Result<String>`
- `generate_graphql(&Aspect)` → `Result<String>`
- `generate_sql(&Aspect, SqlDialect)` → `Result<String>`
- `generate_payload(&Aspect)` → `Result<Value>`

**Guarantees**:
- Generated code format and structure will remain compatible
- Option structs will accept new fields with defaults
- Generated code will remain valid in target languages
- Constraint-aware generation will be maintained

#### Performance & Production

Performance APIs in `oxirs_samm::performance` and `oxirs_samm::production` are **stable**:

- `PerformanceConfig` - Configuration structure
- `BatchProcessor` - Parallel processing
- `ModelCache` - Model caching
- `ProductionConfig` - Production settings
- `MetricsCollector` - Metrics collection
- `health_check()` - Health status

**Guarantees**:
- Configuration fields will maintain backward compatibility
- Metrics collection will remain consistent
- Performance characteristics will not regress significantly

#### Error Types

All error types in `oxirs_samm::error` are **stable**:

- `SammError` - Main error enumeration
- `SourceLocation` - Error location information
- `Result<T>` - Standard result type

**Guarantees**:
- Existing error variants will remain
- New error variants may be added
- Error messages may improve
- Display implementation will remain compatible

### Unstable/Experimental APIs

These APIs may change in future releases:

- **AAS Integration** (`oxirs_samm::aas_parser`) - May evolve to support AAS V4.0
- **Template System** (`oxirs_samm::templates`) - API may be enhanced
- **Advanced Serializers** - Future formats may be added

**Note**: These APIs are fully functional but may receive breaking changes in 0.2.x releases.

## Versioning Policy

We follow [Semantic Versioning](https://semver.org/) with the following interpretation:

### 0.1.x Patch Releases

**Allowed Changes**:
- Bug fixes that don't change public API
- Performance improvements
- Documentation improvements
- Internal refactoring
- New features that don't affect existing APIs (additive only)

**Not Allowed**:
- Breaking changes to stable APIs
- Removal of public APIs
- Changes to struct field types
- Changes to function signatures

### 0.2.0 Minor Release

**Allowed Changes**:
- Everything from patch releases
- Breaking changes to **unstable** APIs only
- New public APIs
- Deprecation warnings for future removal
- Enhanced validation rules (opt-in)

**Not Allowed**:
- Breaking changes to **stable** APIs

### 1.0.0 Major Release

**Allowed Changes**:
- Breaking changes to all APIs
- Removal of deprecated APIs
- Major architectural changes

## Deprecation Policy

Before removing or significantly changing any stable API:

1. **Deprecation Warning** - Mark API as `#[deprecated]` with migration guidance
2. **Grace Period** - Maintain deprecated API for at least one minor release (0.x.0)
3. **Migration Guide** - Provide clear documentation on migration path
4. **Removal** - Only in next major version (1.0.0+)

## Testing Guarantees

All stable APIs have:
- ✅ Unit test coverage (>85%)
- ✅ Integration test coverage
- ✅ Property-based testing where applicable
- ✅ Benchmark coverage
- ✅ Documentation examples that compile

## SciRS2 Integration Stability

**Current Status**: Using SciRS2-core v0.1.x

**Guarantees**:
- Will track SciRS2 minor version updates
- Breaking changes in SciRS2 will be absorbed internally
- Public API will remain stable regardless of SciRS2 changes

**Current Integration Points**:
- `scirs2-core::random` - Random value generation
- `scirs2-core::profiling` - Performance profiling
- `scirs2-core::metrics` - Metrics collection (planned)
- `scirs2-graph` - Graph algorithms (planned)

## Extension Points

### Safe Extension Mechanisms

You can safely extend OxiRS SAMM functionality through:

1. **Custom Generators** - Implement your own code generators
2. **Custom Validators** - Add domain-specific validation rules
3. **Custom Templates** - Use template system for custom formats
4. **Model Resolvers** - Add custom URN resolution strategies

These extension points will remain stable and supported.

## Binary Compatibility

**Guarantees**:
- Within 0.1.x series: Full binary compatibility
- Across 0.x.y releases: Source compatibility only
- 1.0.0+: Both source and binary compatibility

## Supported Rust Versions

**Minimum Supported Rust Version (MSRV)**: 1.70

**Policy**:
- MSRV will not increase in patch releases (0.1.x)
- MSRV may increase in minor releases (0.x.0) with advance notice
- Major releases (1.0.0+) may require latest stable Rust

## Feature Flags

All feature flags are **stable**:

- `default` - Core SAMM functionality
- `codegen` - Code generation support
- `aas` - Asset Administration Shell support
- `aasx-thumbnails` - AASX thumbnail support (requires `image`)
- `graphviz` - Diagram rendering (requires `graphviz-rust`)

**Guarantees**:
- Feature flags will not be removed without deprecation
- Feature flag behavior will remain consistent
- New feature flags may be added

## SAMM Specification Compatibility

**Current Version**: SAMM 2.3.0

**Guarantees**:
- Full compliance with SAMM 2.3.0 specification
- Backward compatibility with SAMM 2.0.0, 2.1.0, 2.2.0
- Support for future SAMM versions will be additive only
- Breaking changes in SAMM spec will be opt-in

## Support & Contact

**Issues**: https://github.com/cool-japan/oxirs/issues
**Documentation**: https://docs.rs/oxirs-samm
**Repository**: https://github.com/cool-japan/oxirs

## Change Log

All changes are documented in:
- `CHANGELOG.md` - Detailed change log
- `TODO.md` - Future enhancements
- Release notes - GitHub releases

## Commitment

We are committed to:
- ✅ Maintaining API stability for stable APIs
- ✅ Clear communication about changes
- ✅ Providing migration paths for breaking changes
- ✅ Listening to community feedback
- ✅ Comprehensive testing before releases

---

**Last Updated**: 2026-01-06
**Next Review**: Upon 0.2.0 release planning
