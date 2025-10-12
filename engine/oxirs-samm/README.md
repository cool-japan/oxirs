# OxiRS SAMM - Semantic Aspect Meta Model for Rust

[![Version](https://img.shields.io/badge/version-0.1.0--alpha.3-orange)](https://github.com/cool-japan/oxirs/releases)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](../../../LICENSE)

**Status**: Alpha Release (v0.1.0-alpha.3)
⚠️ APIs may change. Not recommended for production use.

## Overview

OxiRS SAMM is a Rust implementation of the [Semantic Aspect Meta Model (SAMM)](https://eclipse-esmf.github.io/samm-specification/), which enables the creation of semantic models to describe digital twins and their aspects.

SAMM was developed by the [Eclipse Semantic Modeling Framework (ESMF)](https://github.com/eclipse-esmf) project and provides a standardized way to model domain-specific aspects of digital twins.

## Features

- ✅ **SAMM 2.0.0-2.3.0 Support**: Version-agnostic implementation of SAMM metamodel
- ✅ **RDF/Turtle Parsing**: Load SAMM models from Turtle (.ttl) files with oxttl
- ✅ **Type-Safe Metamodel**: Rust structs for all SAMM elements
- ✅ **CLI Integration**: Full `oxirs aspect` command suite (validate, prettyprint, to) - Java ESMF SDK compatible
- ✅ **Code Generation**: 16 output formats including Rust, Python, Java, Scala, TypeScript, GraphQL, and more
- 🚧 **SHACL Validation**: Validate models against SAMM shapes (in progress)
- ✅ **Extended Formats**: OpenAPI, AsyncAPI, HTML, JSON Schema, AAS, SQL DDL, and more

## Quick Start

### Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxirs-samm = "0.1.0-alpha.3"
```

### Usage

```rust
use oxirs_samm::parser::parse_aspect_model;
use oxirs_samm::metamodel::Aspect;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse a SAMM model from a Turtle file
    let aspect = parse_aspect_model("path/to/AspectModel.ttl").await?;

    println!("Aspect: {}", aspect.name());
    println!("Properties:");
    for property in aspect.properties() {
        println!("  - {}", property.name());
    }

    Ok(())
}
```

## SAMM Metamodel

OxiRS SAMM implements all core SAMM metamodel elements:

### Core Elements

- **Aspect**: Root element describing a digital twin's specific aspect
- **Property**: Named feature with a defined characteristic
- **Characteristic**: Describes the semantics of a property's value
- **Entity**: Complex data structure with multiple properties
- **Operation**: Function that can be performed on an aspect
- **Event**: Occurrence that can be emitted by an aspect

### Characteristics

Supports all SAMM characteristic types:

| Type | Description |
|------|-------------|
| `Trait` | Basic characteristic |
| `Quantifiable` | Characteristic with unit |
| `Measurement` | Quantifiable with specific value |
| `Enumeration` | Set of allowed values |
| `State` | Enumeration representing states |
| `Duration` | Time duration |
| `Collection` | Collection of values |
| `List` | Ordered collection |
| `Set` | Unordered unique collection |
| `SortedSet` | Sorted collection |
| `TimeSeries` | Time-indexed data |
| `Code` | String with encoding |
| `Either` | One of two alternatives |
| `SingleEntity` | Single entity instance |
| `StructuredValue` | Value with structure |

### Constraints

Supports all SAMM constraint types:

- `LanguageConstraint`: Restrict to specific language
- `LocaleConstraint`: Restrict to specific locale
- `RangeConstraint`: Value range limits
- `LengthConstraint`: String/collection length limits
- `RegularExpressionConstraint`: Pattern matching
- `EncodingConstraint`: Character encoding
- `FixedPointConstraint`: Decimal precision

## Architecture

```
oxirs-samm/
├── src/
│   ├── lib.rs                # Crate entry point
│   ├── error.rs              # Error types
│   ├── metamodel/            # SAMM metamodel types
│   │   ├── aspect.rs
│   │   ├── property.rs
│   │   ├── characteristic.rs
│   │   ├── entity.rs
│   │   └── operation.rs
│   ├── parser/               # RDF/Turtle parser
│   │   ├── ttl_parser.rs
│   │   └── resolver.rs
│   └── validator/            # SHACL validation
│       └── shacl_validator.rs
└── tests/
    └── fixtures/             # Example SAMM models
```

## Examples

### Defining an Aspect

```rust
use oxirs_samm::metamodel::{Aspect, Property, Characteristic, CharacteristicKind};

let mut aspect = Aspect::new("urn:samm:com.example:1.0.0#Movement".to_string());
aspect.metadata.add_preferred_name("en".to_string(), "Movement".to_string());

let property = Property::new("urn:samm:com.example:1.0.0#speed".to_string())
    .with_characteristic(
        Characteristic::new(
            "urn:samm:com.example:1.0.0#SpeedCharacteristic".to_string(),
            CharacteristicKind::Measurement {
                unit: "unit:kilometrePerHour".to_string(),
            },
        )
    );

aspect.add_property(property);
```

### Parsing from Turtle

```rust
use oxirs_samm::parser::parse_aspect_from_string;

let ttl = r#"
@prefix : <urn:samm:com.example:1.0.0#> .
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix samm-c: <urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#> .

:Movement a samm:Aspect ;
   samm:preferredName "Movement"@en ;
   samm:properties ( :speed ) .

:speed a samm:Property ;
   samm:characteristic :SpeedCharacteristic .

:SpeedCharacteristic a samm-c:Measurement ;
   samm:dataType xsd:float ;
   samm-c:unit unit:kilometrePerHour .
"#;

let aspect = parse_aspect_from_string(ttl, "http://example.org").await?;
```

## CLI Integration

The `oxirs aspect` command suite provides tools for working with SAMM Aspect Models (Java ESMF SDK compatible):

```bash
# Validate a SAMM Aspect model
oxirs aspect validate AspectModel.ttl
oxirs aspect validate AspectModel.ttl --detailed
oxirs aspect validate AspectModel.ttl --format json

# Pretty-print a model
oxirs aspect prettyprint AspectModel.ttl
oxirs aspect prettyprint AspectModel.ttl -o formatted.ttl

# Generate Rust code
oxirs aspect AspectModel.ttl to rust

# Generate Markdown documentation
oxirs aspect AspectModel.ttl to markdown

# Generate JSON Schema
oxirs aspect AspectModel.ttl to jsonschema

# Generate OpenAPI 3.1 specification
oxirs aspect AspectModel.ttl to openapi

# Generate AsyncAPI 2.6 specification
oxirs aspect AspectModel.ttl to asyncapi

# Generate HTML documentation
oxirs aspect AspectModel.ttl to html

# Generate GraphQL schema
oxirs aspect AspectModel.ttl to graphql

# Generate TypeScript interfaces
oxirs aspect AspectModel.ttl to typescript

# Generate Python dataclasses
oxirs aspect AspectModel.ttl to python

# Generate Java code (POJOs/Records)
oxirs aspect AspectModel.ttl to java

# Generate Scala case classes
oxirs aspect AspectModel.ttl to scala

# Generate SQL DDL (PostgreSQL)
oxirs aspect AspectModel.ttl to sql --format postgresql

# Generate AAS (Asset Administration Shell) JSON
oxirs aspect AspectModel.ttl to aas --format json

# Generate diagram (DOT format)
oxirs aspect AspectModel.ttl to diagram --format dot

# Generate diagram (SVG format, requires Graphviz installed)
oxirs aspect AspectModel.ttl to diagram --format svg

# Generate sample JSON payload
oxirs aspect AspectModel.ttl to payload

# Generate JSON-LD with semantic context
oxirs aspect AspectModel.ttl to jsonld
```

### Java ESMF SDK Compatibility

OxiRS is a **drop-in replacement** for the Java ESMF SDK. Simply replace `samm` with `oxirs` in your commands:

```bash
# Java ESMF SDK
samm aspect Movement.ttl to aas --format xml

# OxiRS (identical syntax, just replace 'samm' with 'oxirs')
oxirs aspect Movement.ttl to aas --format xml
```

### AAS Integration - Industry 4.0 Support

OxiRS Alpha.3 includes bidirectional AAS (Asset Administration Shell) integration:

```bash
# Convert AAS Submodel Templates to SAMM Aspect Models
oxirs aas AssetAdminShell.aasx to aspect
oxirs aas AssetAdminShell.aasx to aspect -d output/
oxirs aas AssetAdminShell.aasx to aspect -s 1 -s 2 -d output/

# List all submodel templates in an AAS file
oxirs aas AssetAdminShell.aasx list

# Supports all AAS formats (XML, JSON, AASX)
oxirs aas file.xml to aspect        # XML format
oxirs aas file.json to aspect       # JSON format
oxirs aas file.aasx to aspect       # AASX (default)
```

**Implementation Status** (Alpha.3):
- ✅ CLI command structure and routing
- ✅ File format detection (XML/JSON/AASX)
- ✅ Java ESMF SDK compatible syntax
- 🚧 AAS parser implementation (in progress)
- 🚧 Submodel template extraction (in progress)
- 🚧 AAS to SAMM conversion logic (in progress)

**Complete command comparison**: See [SAMM_CLI_COMPARISON.md](./SAMM_CLI_COMPARISON.md)

## Development Status

### Phase 1: Foundation ✅ (Completed)

- [x] SAMM metamodel type definitions
- [x] Basic parser structure
- [x] Validator foundation
- [x] Unit tests (21 tests passing)

### Phase 2: Parser Implementation ✅ (Completed)

- [x] Copy SAMM 2.2.0 example models from Eclipse ESMF
- [x] Implement Turtle parser with oxttl/oxrdf
- [x] Add version detection (2.0.0, 2.1.0, 2.2.0, 2.3.0)
- [x] Add blank node support for RDF lists
- [x] Add built-in characteristic handling
- [x] Integration tests (6 tests passing)

### Phase 3: CLI Integration ✅ (Completed)

- [x] `oxirs aspect validate` command with JSON/text output (Java ESMF SDK compatible)
- [x] `oxirs aspect prettyprint` command
- [x] `oxirs aspect <model> to rust` - Generate Rust structs
- [x] `oxirs aspect <model> to markdown` - Generate documentation
- [x] Command-line argument parsing with clap
- [x] Drop-in replacement for Java ESMF SDK (`samm` → `oxirs`)

### Phase 4: Extended Code Generation ✅ (Completed in v0.1.0-alpha.3)

- [x] Rust code generation with serde support
- [x] Markdown documentation generation
- [x] JSON Schema (Draft 2020-12) generation
- [x] OpenAPI 3.1.0 specification generation
- [x] AsyncAPI 2.6 specification generation
- [x] HTML documentation with modern styling
- [x] AAS (Asset Administration Shell) - XML/JSON/AASX generation
- [x] GraphQL schema generation with Query types
- [x] TypeScript interface generation with JSDoc
- [x] Python dataclass generation (Pydantic support)
- [x] Java code generation (Records/POJOs with Jackson/Lombok)
- [x] Scala case class generation (Scala 2.13/3 with Circe/Play JSON)
- [x] SQL DDL generation (PostgreSQL, MySQL, SQLite)
- [x] JSON-LD generation with semantic context
- [x] Sample payload generation for testing
- [x] Diagram generation (Graphviz DOT/SVG/PNG)
- [x] Custom AASX thumbnails (with `aasx-thumbnails` feature)

### Future Enhancements 📅 (Planned)

- [ ] Template system for custom output formats
- [ ] Advanced SHACL validation engine
- [ ] Multi-file code generation (packages/modules)
- [ ] Custom code generation hooks and plugins

## References

- [SAMM Specification](https://eclipse-esmf.github.io/samm-specification/snapshot/index.html)
- [Eclipse ESMF Project](https://github.com/eclipse-esmf)
- [Eclipse ESMF SDK (Java)](https://github.com/eclipse-esmf/esmf-sdk)
- [OxiRS Project](https://github.com/cool-japan/oxirs)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../../../LICENSE-MIT))

at your option.

## Contributing

Contributions are welcome! This is part of the OxiRS project. Please see the main [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

---

**Note**: This crate is part of the OxiRS ecosystem, a Rust-native platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning.
