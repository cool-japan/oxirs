# DTDL Generator Guide

## Overview

The DTDL (Digital Twins Definition Language) generator converts SAMM Aspect models to DTDL v3 Interface definitions for use with Azure Digital Twins.

## What is DTDL?

DTDL is Microsoft's modeling language for describing:
- IoT devices and sensors
- Digital twins and their capabilities
- Telemetry, properties, commands, and relationships
- Complex object hierarchies

DTDL models can be deployed to Azure Digital Twins for cloud-based digital twin management.

## SAMM to DTDL Mapping

| SAMM Element | DTDL Element | Notes |
|--------------|--------------|-------|
| **Aspect** | Interface | Root digital twin model |
| **Property** | Property/Telemetry | Writable state vs read-only data |
| **Operation** | Command | Callable functions on the twin |
| **Event** | Telemetry | Event data emitted by the twin |
| **Entity** | Object/Map | Complex nested structures |
| **Characteristic** | Schema | Data type and constraints |

### URN to DTMI Conversion

SAMM URNs are automatically converted to DTMI (Digital Twin Model Identifier) format:

```
urn:samm:com.example:1.0.0#Movement
→ dtmi:com:example:Movement;1

urn:samm:org.eclipse.esmf:2.3.0#Aspect
→ dtmi:org:eclipse:esmf:Aspect;2
```

**Conversion Rules:**
- Namespace dots (`.`) → colons (`:`)
- Version: Extract major version only (1.0.0 → 1)
- Fragment (`#`) → DTMI separator (`:`)

### Data Type Mapping

| XSD Type | DTDL Schema | Example |
|----------|-------------|---------|
| `xsd:int`, `xsd:integer` | `integer` | Whole numbers |
| `xsd:long` | `long` | 64-bit integers |
| `xsd:float` | `float` | 32-bit floating point |
| `xsd:double` | `double` | 64-bit floating point |
| `xsd:boolean` | `boolean` | true/false |
| `xsd:string` | `string` | Text data |
| `xsd:dateTime` | `dateTime` | ISO 8601 timestamp |
| `xsd:date` | `date` | ISO 8601 date |
| `xsd:time` | `time` | ISO 8601 time |
| `xsd:duration` | `duration` | ISO 8601 duration |

## Usage

### CLI Command

```bash
# Basic generation
oxirs aspect to Movement.ttl dtdl

# Save to file
oxirs aspect to Movement.ttl dtdl -o Movement.json

# Generate from multiple models
for model in models/*.ttl; do
  oxirs aspect to "$model" dtdl -o "azure/$(basename "$model" .ttl).json"
done
```

### Programmatic API

```rust
use oxirs_samm::generators::dtdl::{generate_dtdl, generate_dtdl_with_options, DtdlOptions};
use oxirs_samm::parser::parse_aspect_model;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse SAMM model
    let aspect = parse_aspect_model("Movement.ttl").await?;

    // Generate DTDL with default options
    let dtdl = generate_dtdl(&aspect)?;
    println!("{}", dtdl);

    // Generate with custom options
    let options = DtdlOptions {
        version: 3,
        include_descriptions: true,
        include_display_names: true,
        compact: false,
        all_writable: false,
    };
    let custom_dtdl = generate_dtdl_with_options(&aspect, options)?;

    // Save to file
    std::fs::write("Movement.json", custom_dtdl)?;

    Ok(())
}
```

## Generation Options

### DtdlOptions

```rust
pub struct DtdlOptions {
    /// DTDL version (default: 3)
    pub version: u8,

    /// Include descriptions in output (default: true)
    pub include_descriptions: bool,

    /// Include display names in output (default: true)
    pub include_display_names: true,

    /// Generate compact JSON without indentation (default: false)
    pub compact: bool,

    /// Mark all properties as writable (default: false)
    pub all_writable: bool,
}
```

### Examples

**Compact Output:**
```rust
let options = DtdlOptions {
    compact: true,
    ..Default::default()
};
let dtdl = generate_dtdl_with_options(&aspect, options)?;
// Produces: {"@context":"dtmi:dtdl:context;3","@id":"dtmi:..."}
```

**Minimal Output (no descriptions):**
```rust
let options = DtdlOptions {
    include_descriptions: false,
    include_display_names: false,
    ..Default::default()
};
```

**All Writable Properties:**
```rust
let options = DtdlOptions {
    all_writable: true,  // All properties become "Property" instead of "Telemetry"
    ..Default::default()
};
```

## Complete Example

### Input: SAMM Aspect Model (Turtle)

```turtle
@prefix : <urn:samm:com.example.vehicle:1.0.0#> .
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix samm-c: <urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#> .
@prefix unit: <urn:samm:org.eclipse.esmf.samm:unit:2.3.0#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:Movement a samm:Aspect ;
   samm:preferredName "Movement"@en ;
   samm:description "Vehicle movement tracking"@en ;
   samm:properties ( :speed :position ) ;
   samm:operations ( :emergencyStop ) .

:speed a samm:Property ;
   samm:preferredName "speed"@en ;
   samm:description "Current speed in km/h"@en ;
   samm:characteristic :SpeedCharacteristic .

:SpeedCharacteristic a samm-c:Measurement ;
   samm:dataType xsd:float ;
   samm-c:unit unit:kilometrePerHour .

:position a samm:Property ;
   samm:preferredName "position"@en ;
   samm:description "GPS coordinates"@en ;
   samm:characteristic :PositionCharacteristic .

:PositionCharacteristic a samm-c:Trait ;
   samm:dataType xsd:string .

:emergencyStop a samm:Operation ;
   samm:preferredName "emergencyStop"@en ;
   samm:description "Emergency stop command"@en .
```

### Output: DTDL Interface (JSON)

```json
{
  "@context": "dtmi:dtdl:context;3",
  "@id": "dtmi:com:example:vehicle:Movement;1",
  "@type": "Interface",
  "displayName": "Movement",
  "description": "Vehicle movement tracking",
  "contents": [
    {
      "@type": "Property",
      "name": "speed",
      "displayName": "speed",
      "description": "Current speed in km/h",
      "schema": "float"
    },
    {
      "@type": "Property",
      "name": "position",
      "displayName": "position",
      "description": "GPS coordinates",
      "schema": "string"
    },
    {
      "@type": "Command",
      "name": "emergencyStop",
      "displayName": "emergencyStop",
      "description": "Emergency stop command"
    }
  ]
}
```

## Azure Digital Twins Integration

### 1. Generate DTDL Models

```bash
# Generate DTDL from all your SAMM models
mkdir -p azure-twins/models
for model in samm-models/*.ttl; do
  oxirs aspect to "$model" dtdl -o "azure-twins/models/$(basename "$model" .ttl).json"
done
```

### 2. Validate DTDL Models (Optional)

```bash
# Install Azure CLI and DTDL validator
npm install -g @azure/dtdl-validator

# Validate generated models
dtdl-validator azure-twins/models/*.json
```

### 3. Upload to Azure Digital Twins

```bash
# Login to Azure
az login

# Create Digital Twins instance (if needed)
az dt create --resource-group myResourceGroup --dt-name myDigitalTwin

# Upload models
for model in azure-twins/models/*.json; do
  az dt model create --dt-name myDigitalTwin --models "$model"
done

# List uploaded models
az dt model list --dt-name myDigitalTwin
```

### 4. Create Twin Instances

```bash
# Create a twin instance from the model
az dt twin create \
  --dt-name myDigitalTwin \
  --dtmi "dtmi:com:example:vehicle:Movement;1" \
  --twin-id vehicle-001 \
  --properties '{
    "speed": 65.5,
    "position": "35.6895,139.6917",
    "isMoving": true
  }'
```

## Advanced Features

### Property vs Telemetry

DTDL distinguishes between:
- **Property**: Writable state (device configuration, settings)
- **Telemetry**: Read-only streaming data (sensor readings, events)

The generator automatically chooses based on the SAMM property's `optional` flag:
- Required properties → DTDL Property (writable)
- Optional properties → DTDL Telemetry (read-only)

Override this behavior:
```rust
let options = DtdlOptions {
    all_writable: true,  // Force all to Property
    ..Default::default()
};
```

### Complex Schemas

For complex data types, the generator uses DTDL Object/Array schemas:

| SAMM Characteristic | DTDL Schema |
|---------------------|-------------|
| Collection, List, Set | `dtmi:dtdl:instance:Schema:Array;3` |
| Entity, StructuredValue | `dtmi:dtdl:instance:Schema:Object;3` |
| Either (union types) | `dtmi:dtdl:instance:Schema:Object;3` |

### DTDL Relationships (Future)

DTDL supports Relationships between digital twins. This will be added in a future release when SAMM adds relationship modeling support.

## Testing

The DTDL generator includes comprehensive tests:

```bash
# Run all DTDL tests
cargo test -p oxirs-samm generators::dtdl

# Run specific test
cargo test -p oxirs-samm generators::dtdl::tests::test_to_dtmi_conversion
```

**Test Coverage:**
- ✅ URN to DTMI conversion (3 test cases)
- ✅ XSD to DTDL schema mapping (8 type mappings)
- ✅ CamelCase conversion (5 test cases)
- ✅ JSON escaping
- ✅ Basic aspect generation
- ✅ Aspect with properties
- ✅ Aspect with operations
- ✅ Compact output mode
- ✅ Options (descriptions, display names)
- ✅ Invalid URN error handling

## Plugin System Integration

The DTDL generator is automatically registered in the plugin system:

```rust
use oxirs_samm::generators::plugin::GeneratorRegistry;

let registry = GeneratorRegistry::with_builtin();

// DTDL generator is automatically available
if let Some(generator) = registry.get("dtdl") {
    let dtdl = generator.generate(&aspect)?;
    println!("{}", dtdl);
}

// List all generators
for name in registry.list() {
    println!("Generator: {}", name);
}
// Output includes: typescript, python, java, scala, graphql, sql, jsonld, payload, dtdl
```

## Limitations

### Current Implementation

- ✅ **Supported**: Properties, Telemetry, Commands, basic schemas
- ⚠️ **Partial**: Complex object schemas (mapped to Object type)
- ❌ **Not Yet**: Relationships, Components, Semantic Types

### Future Enhancements

1. **Relationships**: When SAMM adds relationship support
2. **Components**: Nested interface composition
3. **Semantic Types**: DTDL semantic type annotations
4. **Advanced Schemas**: Enum, Map with full type definitions
5. **DTDL v4**: When specification is released

## References

- [DTDL v3 Specification](https://github.com/Azure/opendigitaltwins-dtdl/blob/master/DTDL/v3/DTDL.v3.md)
- [Azure Digital Twins Documentation](https://learn.microsoft.com/en-us/azure/digital-twins/)
- [DTMI Specification](https://github.com/Azure/opendigitaltwins-dtdl/blob/master/DTDL/v3/DTDL.v3.md#digital-twin-model-identifier)
- [SAMM Specification](https://eclipse-esmf.github.io/samm-specification/)
- [OxiRS SAMM Module](./README.md)

## Examples

See the complete runnable example:
```bash
cargo run --example dtdl_generation
```

## FAQ

**Q: Can I use DTDL models with other cloud platforms?**
A: DTDL is Azure-specific. For multi-cloud support, generate both DTDL (Azure) and AAS (Industry 4.0 standard) from the same SAMM model.

**Q: How do I validate generated DTDL?**
A: Use Microsoft's DTDL validator: `npm install -g @azure/dtdl-validator`

**Q: Can I customize the DTMI namespace?**
A: The namespace comes from your SAMM model's URN. Use the SAMM namespace that matches your organization's domain.

**Q: What about DTDL v2?**
A: The generator targets DTDL v3 (latest). For v2, use `DtdlOptions { version: 2, .. }` (experimental).

## Troubleshooting

### Invalid DTMI Error

```
Error: Invalid SAMM URN: urn:invalid. Expected format: urn:samm:namespace:version#name
```

**Solution**: Ensure your SAMM URN follows the format: `urn:samm:<namespace>:<version>#<name>`

### Schema Mapping Issues

If your SAMM characteristic uses a custom data type not in the XSD standard, it will default to `string`. To customize:

1. Use standard XSD types in your SAMM models
2. Or extend the `map_xsd_to_dtdl_schema()` function for custom mappings

### JSON Validation Errors

All generated DTDL is validated as JSON. If you encounter parsing errors:

1. Check for special characters in descriptions (they're auto-escaped)
2. Verify your SAMM model parses correctly: `oxirs aspect validate <file>.ttl`
3. Report issues at https://github.com/cool-japan/oxirs/issues

## Contributing

The DTDL generator is part of oxirs-samm. Contributions welcome:

1. **Additional DTDL features**: Relationships, Components, Semantic Types
2. **Schema improvements**: Better mapping for SAMM Constraints
3. **Test cases**: More comprehensive DTDL generation scenarios
4. **Documentation**: Additional examples and use cases

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

---

**Generated by OxiRS v0.1.0-rc.1**
*DTDL Generator - Azure Digital Twins Integration*
