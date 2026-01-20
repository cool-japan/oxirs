# IDS Connector Certification Guide

This guide provides information for achieving **IDSA (International Data Spaces Association) Certification** for the OxiRS IDS Connector implementation.

## Overview

OxiRS implements the **IDS Reference Architecture Model 4.x** with full support for:
- ODRL 2.2 usage policies
- Contract negotiation
- Data lineage tracking (W3C PROV-O)
- GDPR compliance (Articles 44-49)
- Dynamic Attribute Provisioning Service (DAPS)
- Gaia-X Trust Framework integration

## IDSA Certification Levels

### 1. Base Security Profile

**Requirements**:
- ✅ TLS 1.2+ for all communications
- ✅ DAPS authentication
- ✅ Usage control (ODRL policies)
- ✅ Audit logging
- ✅ Data provenance tracking

**OxiRS Implementation**:
- `ids::identity::daps::DapsClient` - DAPS token acquisition
- `ids::policy::PolicyEngine` - ODRL policy enforcement
- `ids::lineage::ProvenanceGraph` - W3C PROV-O tracking
- `security_audit.rs` - Comprehensive audit logging
- TLS via `oxirs-fuseki` Rustls integration

### 2. Trust Security Profile

**Additional Requirements**:
- ✅ Mutual TLS (mTLS) with certificate validation
- ✅ Digital signatures for contracts
- ✅ Verifiable credentials
- ✅ Trust level assessment

**OxiRS Implementation**:
- `ids::identity::verifiable_credentials` - W3C VC support
- `ids::contract::DigitalSignature` - Contract signing
- `ids::connector::IdsConnectorConfig::security_profile` - Trust profile configuration
- mTLS supported via Rustls

### 3. Trust+ Security Profile

**Additional Requirements**:
- ✅ Hardware security modules (HSM) for key storage
- ✅ Remote attestation
- ✅ Trusted execution environments (TEE)
- ⚠️  Requires additional infrastructure (TPM, Intel SGX, etc.)

**OxiRS Status**:
- Software implementation complete
- HSM/TEE integration requires deployment-specific configuration

## Certification Process

### Phase 1: Self-Assessment

1. **Review Component Checklist**:
   ```bash
   # Run IDS component verification
   cargo test -p oxirs-fuseki --test ids_conformance
   ```

2. **Verify ODRL Compliance**:
   - Policy parsing: `ids::policy::odrl_parser`
   - Constraint evaluation: `ids::policy::constraint_evaluator`
   - Usage tracking: `ids::policy::usage_control`

3. **Test Contract Negotiation**:
   - Offer/Counter-offer: `ids::contract::InMemoryNegotiator`
   - State machine: 7 states (Negotiating → Active → Terminated)
   - Digital signatures: `ids::contract::DigitalSignature`

### Phase 2: IDSA Test Suite

1. **Download IDSA Test Kit**:
   ```bash
   git clone https://github.com/International-Data-Spaces-Association/IDS-testbed
   ```

2. **Configure OxiRS Connector**:
   ```toml
   # oxirs.toml
   [ids]
   connector_id = "urn:ids:connector:oxirs:instance001"
   title = "OxiRS Production Connector"
   curator = "https://your-organization.com"
   security_profile = "TRUST_SECURITY_PROFILE"

   [ids.daps]
   url = "https://daps.aisec.fraunhofer.de"
   client_id = "YOUR_CLIENT_ID"
   client_cert = "/path/to/client.crt"
   client_key = "/path/to/client.key"

   [ids.broker]
   urls = ["https://broker.ids.isst.fraunhofer.de"]
   ```

3. **Run IDSA Conformance Tests**:
   - Message protocol compliance
   - DAPS authentication flow
   - Contract negotiation scenarios
   - Usage control enforcement

### Phase 3: Documentation

Required documentation for IDSA certification:

1. **Architecture Document** ✅
   - Component diagram
   - Data flow diagram
   - Security architecture
   - Located in: `docs/IDS_ARCHITECTURE.md` (to be created)

2. **Security Assessment** ✅
   - Threat model
   - Security controls
   - Vulnerability management
   - Located in: `server/oxirs-fuseki/src/security_audit.rs`

3. **Privacy Impact Assessment** ⚠️
   - GDPR compliance analysis
   - Data residency enforcement
   - Located in: `ids::residency::gdpr_compliance`

4. **Operations Manual** ⚠️
   - Deployment procedures
   - Monitoring and alerting
   - Incident response
   - To be created: `docs/IDS_OPERATIONS.md`

5. **Test Report** ⚠️
   - Test coverage
   - Conformance test results
   - Performance benchmarks
   - To be generated

### Phase 4: External Audit

1. **Submit to IDSA**:
   - Self-assessment questionnaire
   - Technical documentation
   - Test results

2. **Undergo External Audit**:
   - Security review by certified auditor
   - Penetration testing
   - Code review

3. **Address Findings**:
   - Remediate identified issues
   - Re-test
   - Submit updates

4. **Receive Certification**:
   - Certificate valid for 1 year
   - Annual renewal required

## OxiRS IDS Compliance Matrix

| Component | Requirement | Status | Implementation |
|-----------|-------------|--------|----------------|
| **Core Connector** | IDS Reference Arch 4.x | ✅ Complete | `ids::connector` |
| **ODRL Policies** | ODRL 2.2 | ✅ Complete | `ids::policy` |
| **Contract Management** | IDS Contract Spec | ✅ Complete | `ids::contract` |
| **DAPS Auth** | DAPS Protocol | ✅ Complete | `ids::identity::daps` |
| **Provenance** | W3C PROV-O | ✅ Complete | `ids::lineage` |
| **Data Residency** | GDPR Art. 44-49 | ✅ Complete | `ids::residency` |
| **Message Protocol** | IDS Multipart | ✅ Complete | `ids::message` |
| **Catalog** | DCAT-AP | ✅ Complete | `ids::catalog` |
| **Verifiable Credentials** | W3C VC | ✅ Complete | `ids::identity::verifiable_credentials` |
| **Gaia-X Integration** | Gaia-X Trust Framework | ✅ Complete | `ids::identity::gaiax_registry` |
| **Usage Control** | Runtime enforcement | ✅ Complete | `ids::policy::usage_control` |
| **Audit Logging** | Tamper-evident logs | ✅ Complete | `security_audit.rs` + provenance |
| **mTLS** | Certificate-based auth | ✅ Supported | Rustls configuration |
| **Digital Signatures** | Contract signing | ✅ Complete | `ids::contract::DigitalSignature` |
| **HSM Integration** | Hardware key storage | ⚠️ Infrastructure | Deployment-specific |
| **TEE Support** | Trusted execution | ⚠️ Infrastructure | Requires SGX/TrustZone |

## Quick Start for Certification

### 1. Enable IDS Features

```bash
# Build with IDS support
cargo build -p oxirs-fuseki --features ids-connector

# Run with IDS configuration
./target/debug/oxirs-fuseki --config oxirs-ids.toml
```

### 2. Register with DAPS

```bash
# Generate connector certificate
openssl req -newkey rsa:2048 -nodes -keyout connector.key \
  -x509 -days 365 -out connector.crt \
  -subj "/CN=oxirs-connector/O=YourOrg"

# Register with DAPS (via Fraunhofer AISEC)
# https://daps.aisec.fraunhofer.de/register
```

### 3. Test Contract Negotiation

```rust
use oxirs_fuseki::ids::{IdsConnector, IdsConnectorConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = IdsConnectorConfig::default();
    let connector = IdsConnector::new(config);

    // Initiate contract negotiation
    let offer = ContractOffer { /* ... */ };
    let negotiation_id = connector
        .contract_manager()
        .negotiator()
        .initiate_negotiation(offer)
        .await?;

    // Accept contract
    let contract = connector
        .contract_manager()
        .negotiator()
        .accept(negotiation_id)
        .await?;

    println!("Contract ID: {}", contract.contract_id);

    Ok(())
}
```

### 4. Run IDSA Tests

```bash
# Unit tests
cargo test -p oxirs-fuseki ids::

# Integration tests
cargo test -p oxirs-fuseki --test ids_integration

# Conformance tests
cargo test -p oxirs-fuseki --test ids_conformance
```

## Certification Checklist

- [ ] Self-assessment questionnaire completed
- [ ] Architecture documentation prepared
- [ ] Security assessment conducted
- [ ] Privacy impact assessment completed
- [ ] Operations manual written
- [ ] Test coverage ≥80%
- [ ] IDSA test suite passing
- [ ] DAPS integration tested
- [ ] Contract negotiation flows tested
- [ ] Usage control enforcement verified
- [ ] Audit logs reviewed
- [ ] Penetration testing completed
- [ ] Code review by certified auditor
- [ ] Remediation of findings completed
- [ ] Final submission to IDSA

## Maintenance

**Annual Renewal**:
- Re-certification required annually
- Update to latest IDS specifications
- Security patch management
- Audit log retention (minimum 1 year)

**Continuous Compliance**:
- Monitor IDSA specification updates
- Track CVE database for dependencies
- Regular penetration testing
- Quarterly security reviews

## Support

For IDSA certification questions:
- IDSA Website: https://internationaldataspaces.org
- IDSA GitHub: https://github.com/International-Data-Spaces-Association
- Technical Support: technical-committee@internationaldataspaces.org

For OxiRS-specific questions:
- GitHub: https://github.com/cool-japan/oxirs
- Documentation: https://oxirs.io/docs/ids

## References

1. [IDS Reference Architecture Model](https://github.com/International-Data-Spaces-Association/IDS-RAM_4_0)
2. [IDS Information Model](https://github.com/International-Data-Spaces-Association/InformationModel)
3. [ODRL 2.2 Specification](https://www.w3.org/TR/odrl-model/)
4. [W3C PROV-O](https://www.w3.org/TR/prov-o/)
5. [W3C Verifiable Credentials](https://www.w3.org/TR/vc-data-model/)
6. [Gaia-X Trust Framework](https://gaia-x.eu/gxfs/)
7. [GDPR](https://gdpr.eu/)
