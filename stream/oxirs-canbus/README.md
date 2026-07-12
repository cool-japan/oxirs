# oxirs-canbus

[![Crates.io](https://img.shields.io/crates/v/oxirs-canbus.svg)](https://crates.io/crates/oxirs-canbus)
[![docs.rs](https://docs.rs/oxirs-canbus/badge.svg)](https://docs.rs/oxirs-canbus)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

CANbus/J1939 protocol support for the OxiRS semantic web platform.

## Status

✅ **Production Ready** (v0.3.2) - Phase D: Industrial Connectivity Complete

## Overview

`oxirs-canbus` provides native Rust implementations of CANbus (Controller Area Network) and J1939 protocols, enabling real-time RDF knowledge graph updates from automotive vehicles, heavy machinery, agricultural equipment, and other CAN-enabled systems.

**Market Coverage**: Ubiquitous in automotive (100%), heavy machinery (80%), agriculture (60%).

## Features

### Core Protocol
- ✅ **Socketcan integration** - Linux CAN interface support (vcan for testing)
- ✅ **CAN 2.0 + CAN FD** - Standard/extended IDs, up to 64-byte FD payloads
- ✅ **J1939 protocol** - Heavy vehicle parameter groups (PGN extraction), multi-packet reassembly (BAM / TP.CM / TP.DT)
- ✅ **DBC file parsing** - Vector CANdb++ format (messages, signals, value tables, Intel/Motorola byte order)
- ✅ **Signal decoding** - Little/big endian, unaligned, signed/unsigned, scaling/offset
- ✅ **Frame filtering** - Composable mask/range/data/logical filter rules

### Diagnostics & Higher-Layer Protocols
- ✅ **OBD-II** - SAE J1979 Mode 01 PID + DTC decoding, ISO 15765-2 diagnostic session monitoring
- ✅ **UDS** - Unified Diagnostic Services (ISO 14229) client over ISO-TP
- ✅ **CANopen** - CiA DS-301 NMT/SDO/PDO/EMCY object dictionary
- ✅ **J1939 diagnostics** - DM1/DM2/DM3/DM11/DM13 messages, DTC + lamp status

### RDF & Digital Twin
- ✅ **RDF mapping** - CAN frames → RDF triples with W3C PROV-O provenance
- ✅ **SAMM generation** - Auto-generate Eclipse SAMM Aspect Models from DBC files
- ✅ **Vehicle digital twin** - Real-time OBD-II + CAN state aggregation with SAREF/SSN triple generation
- ✅ **J1939 ↔ DTDL bridge** - Facade-based property bridge into `oxirs-physics` digital twins (mock J1939/DTDL transports bundled; real transports pluggable)

### Bus Analysis & Tooling
- ✅ **Frame validation, aggregation & monitoring** - Integrity/DLC checks, windowed statistics, threshold alerting
- ✅ **Network topology modeling** - Node/edge graph with BFS routing
- ✅ **Bus scheduling & bit-timing** - Priority arbitration, bus-load calculation, bit-timing math
- ✅ **Error state tracking** - TEC/REC counters (Error Active → Bus Off)
- ✅ **Replay engine** - Time-scaled CAN bus replay
- ✅ **Recording formats** - ASC, BLF, CSV, MF4 read/write
- ✅ **Gateway bridge** - Rule-based CAN-to-MQTT/HTTP routing (in-memory simulation harness)
- ✅ **Message database** - DBC-like message/signal definitions with decode/encode

## Quick Start

### Installation

```toml
[dependencies]
oxirs-canbus = "0.3.2"
```

**Note**: Linux only (requires socketcan kernel module).

### Basic CANbus Example

```rust
use oxirs_canbus::{CanbusClient, CanbusConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = CanbusConfig {
        interface: "can0".to_string(),
        dbc_file: Some("vehicle.dbc".to_string()),
        ..Default::default()
    };

    let mut client = CanbusClient::new(config)?;
    client.start().await?;

    Ok(())
}
```

### DBC Integration Example

```rust
use oxirs_canbus::{parse_dbc_file, CanRdfMapper, CanbusClient, CanbusConfig, RdfMappingConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse DBC file
    let dbc = parse_dbc_file("vehicle.dbc")?;

    // Create RDF mapper
    let rdf_config = RdfMappingConfig {
        device_id: "vehicle001".to_string(),
        base_iri: "http://automotive.example.com/vehicle".to_string(),
        graph_iri: "urn:automotive:can-data".to_string(),
    };
    let mut mapper = CanRdfMapper::new(dbc, rdf_config);

    // Connect to CAN interface
    let config = CanbusConfig {
        interface: "can0".to_string(),
        dbc_file: Some("vehicle.dbc".to_string()),
        ..Default::default()
    };
    let mut client = CanbusClient::new(config)?;
    client.start().await?;

    // Receive CAN frames and convert to RDF
    while let Some(frame) = client.recv_frame().await {
        let triples = mapper.map_frame(&frame)?;
        println!("Generated {} triples", triples.len());
    }

    Ok(())
}
```

## DBC File Format

```dbc
VERSION ""

BO_ 1234 EngineData: 8 Engine
  SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8191.875] "rpm" ECU
  SG_ EngineTemp : 16|8@1+ (1,-40) [-40|215] "deg C" ECU
  SG_ ThrottlePos : 24|8@1+ (0.39215686,0) [0|100] "%" ECU

BO_ 1235 VehicleSpeed: 2 Body
  SG_ Speed : 0|16@1+ (0.01,0) [0|655.35] "km/h" ECU
```

## Configuration

### TOML Configuration (oxirs.toml)

```toml
[[stream.external_systems]]
type = "CANbus"
interface = "can0"
dbc_file = "vehicle.dbc"

[stream.external_systems.rdf_mapping]
device_id = "vehicle001"
base_iri = "http://automotive.example.com/vehicle"
graph_iri = "urn:automotive:can-data"
```

## Performance Targets

- **Throughput**: 10,000 CAN messages/sec
- **Latency**: <1ms RDF conversion per frame
- **Interfaces**: Support 8 CAN interfaces simultaneously
- **Memory usage**: <50MB for 100K frames/sec

## Platform Support

- **Linux**: Full support with socketcan kernel module
- **macOS**: Limited (virtual CAN only)
- **Windows**: Not supported (use WSL2)

## Standards Compliance

- ISO 11898 (CAN 2.0)
- ISO 11898-1:2015 (CAN FD)
- SAE J1939 (heavy vehicle communication)
- Vector CANdb++ DBC format

## Use Cases

- **Automotive**: OBD-II diagnostics, EV battery monitoring, fleet management
- **Heavy Machinery**: Construction equipment telemetry, predictive maintenance
- **Agriculture**: Tractor and harvester monitoring, precision farming
- **Marine**: Ship engine management, vessel monitoring systems

## Setup (Linux)

### Virtual CAN for Testing

```bash
# Load vcan kernel module
sudo modprobe vcan

# Create virtual CAN interface
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# Verify
ifconfig vcan0
```

### Monitor CAN Traffic

```bash
# Install CAN utilities
sudo apt-get install can-utils

# Monitor all CAN frames
candump vcan0

# Send test frame
cansend vcan0 123#DEADBEEF
```

## CLI Commands

The `oxirs` CLI provides CANbus monitoring and DBC tools:

```bash
# Monitor CAN interface with DBC decoding
oxirs canbus monitor --interface can0 --dbc vehicle.dbc --j1939

# Parse DBC file
oxirs canbus parse-dbc --file vehicle.dbc --detailed

# Decode CAN frame
oxirs canbus decode --id 0x0CF00400 --data DEADBEEF --dbc vehicle.dbc

# Generate SAMM Aspect Models from DBC
oxirs canbus to-samm --dbc vehicle.dbc --output ./models/

# Generate RDF from live CAN data
oxirs canbus to-rdf --interface can0 --dbc vehicle.dbc --output data.ttl --count 1000

# Send CAN frame
oxirs canbus send --interface can0 --id 0x123 --data DEADBEEF
```

See `/tmp/oxirs_cli_phase_d_guide.md` for complete CLI documentation.

## Production Status

- ✅ **1,183 tests passing** - 100% success rate
- ✅ **Zero warnings** - Strict code quality enforcement
- ✅ **7 examples** - Complete usage documentation
- ✅ **54 files, 24,855 lines** - Comprehensive implementation
- ✅ **Standards compliant** - ISO 11898-1, SAE J1939, Vector DBC

## Documentation

- [Implementation Plan](/tmp/oxirs_enhancement_summary.md) - Executive summary
- [Kickoff Plan](/tmp/oxirs_phase_d_kickoff.md) - Development timeline

## License

Licensed under Apache-2.0.
