# OxiRS Enterprise Features Policy

*RFC-002 | Effective: 2026-05-01 | Owner: COOLJAPAN OU (Team Kitasan)*

---

## 1. Purpose

This document decomposes the "Enterprise features" umbrella item that appears across all 26 OxiRS crate TODO.md files into concrete, actionable engineering items. Each item is mapped to the owning crate, its implementation status, and its remaining work. This RFC unblocks planning and eliminates the VAGUE / OVERSIZED label from downstream TODO items.

---

## 2. Feature Decomposition

### 2.1 Authentication — SSO / SAML / OIDC / LDAP / Client-Cert

| Sub-feature | Crate | Module | Status | Remaining Work |
|---|---|---|---|---|
| SAML 2.0 SP | `oxirs-fuseki` | `auth/saml.rs` (feature: `saml`) | Done | None |
| OIDC client | `oxirs-fuseki` | `auth/oauth.rs` | Done | None |
| LDAP / Active Directory | `oxirs-fuseki` | `auth/ldap.rs` | Done | None |
| Certificate-based client auth | `oxirs-fuseki` | `auth/certificate.rs` | Done | None |

**Summary:** All four authentication mechanisms are fully implemented behind their respective Cargo features. No further work required for authentication itself.

**Cargo feature gate pattern:**
```toml
[features]
saml = ["samael"]
ldap = ["ldap3"]
```

---

### 2.2 Role-Based Access Control (RBAC) and ReBAC

| Sub-feature | Crate | Module | Status | Remaining Work |
|---|---|---|---|---|
| Core RBAC engine | `oxirs-fuseki` | `auth/rbac.rs` | Done | None |
| Graph-level RBAC | `oxirs-fuseki` | `auth/graph_acl.rs` | Done | None |
| Relationship-Based Access Control (ReBAC) | `oxirs-fuseki` | `auth/rdf_rebac.rs` | Done | None |
| Enterprise policy templates | `oxirs-fuseki` | `auth/policy_templates.rs` | **Done** | DBA, ReadOnly, Auditor built-in role templates with serialisable `RoleTemplate` structs and `PolicyTemplateRegistry` — completed 2026-05-17 |

**DBA Role template** covers: full SPARQL read/write, graph create/drop, admin endpoints, update processor access.

**ReadOnly Role template** covers: SPARQL SELECT/CONSTRUCT/DESCRIBE/ASK, graph store GET, no UPDATE, no admin.

**Auditor Role template** covers: read-only access to audit trail endpoints, read-only SPARQL, no data modification, no admin panel (data plane only).

**Implementation:** `server/oxirs-fuseki/src/auth/policy_templates.rs` — re-exported from `auth/mod.rs`. Completed 2026-05-17.

---

### 2.3 Audit Logging (SOC2 / GDPR)

| Sub-feature | Crate | Module | Status | Remaining Work |
|---|---|---|---|---|
| Structured audit trail | `oxirs-core` | `audit/` | **Done** | Implemented: `event.rs`, `logger.rs`, `filter.rs`, `gdpr.rs` — completed 2026-05-02 |
| Authentication events | `oxirs-core` | `audit/event.rs` | **Done** | `AuditEventKind::Authentication` — completed 2026-05-02 |
| Query events | `oxirs-core` | `audit/event.rs` | **Done** | `AuditEventKind::DataAccess` — completed 2026-05-02 |
| Data-modification events | `oxirs-core` | `audit/event.rs` | **Done** | `AuditEventKind::DataModification` — completed 2026-05-02 |
| Admin events | `oxirs-core` | `audit/event.rs` | **Done** | `AuditEventKind::Admin` — completed 2026-05-02 |
| GDPR data subject export | `oxirs-core` | `audit/gdpr.rs` | **Done** | `GdprService::data_subject_report` — completed 2026-05-02 |
| GDPR data subject purge | `oxirs-core` | `audit/gdpr.rs` | **Done** | `GdprService::pseudonymise` (Article 17 erasure) — completed 2026-05-02 |
| Query logger (basic) | `oxirs-fuseki` | (existing) | Done | Already implemented in `query_logger` module |

**Audit module design** (`core/oxirs-core/src/audit/`):

```
audit/
  mod.rs          — AuditLogger trait, AuditEvent enum, AuditSink trait
  auth_events.rs  — LoginSuccess, LoginFailure, TokenIssued, TokenRevoked
  query_events.rs — QueryExecuted { tenant, query_text, duration_ms, row_count }
  mutation_events.rs — GraphUpdated, GraphDropped, DataImported
  admin_events.rs — ConfigChanged, UserCreated, RoleAssigned, PolicyChanged
  gdpr.rs         — DataSubjectExportRequest, DataSubjectPurgeRequest, AuditEntryRedactor
```

Event fields: `event_id: Uuid`, `timestamp: DateTime<Utc>`, `actor: ActorId`, `resource: ResourceId`, `outcome: Outcome { Success | Failure(reason) }`, `metadata: HashMap<String, String>`.

Sinks: `JsonLinesSink` (newline-delimited JSON to file), `SyslogSink`, and an `InMemorySink` for tests. Implementors implement `AuditSink::write(&AuditEvent) -> Result<()>`. The `AuditLogger` fans out to multiple sinks.

GDPR compliance:
- `DataSubjectExportRequest { subject_id }` → returns all audit entries referencing that subject.
- `DataSubjectPurgeRequest { subject_id, before: DateTime<Utc> }` → redacts PII fields from matching entries, replacing with `[REDACTED]` and recording a `PurgeRecord` for compliance evidence.
- Retention policy: configurable `max_retention_days`; entries older than the threshold are automatically evicted from the in-memory sink and flagged for deletion in persistent sinks.

**Implementation target:** `core/oxirs-core/src/audit/` — new module directory. Estimated: 500–700 lines + 30 tests.

---

### 2.4 Multi-Tenant Isolation

| Sub-feature | Crate | Module | Status | Remaining Work |
|---|---|---|---|---|
| Namespace-based sharding | `oxirs-cluster` | (existing) | Done | None |
| Vector store multi-tenancy | `oxirs-vec` | `multi_tenancy/` | Done | Fully implemented (sla, admission_controller, priority_queue) |
| Cross-tenant query isolation in ARQ | `oxirs-arq` | `query_governor.rs` | Done | Resource governor (wall-time, row-count, triple-scan budgets) implemented 2026-05-01 |
| Per-tenant SLA classes | `oxirs-core` | `sla/` | Done | Moved from oxirs-vec, fully shared 2026-05-01 |
| Tenant config registry | `oxirs-arq` | `tenant_config.rs` | Done | TenantConfig + TenantConfigRegistry implemented 2026-05-01 |

**Summary:** Multi-tenant isolation is fully implemented across the stack. No further work required.

---

### 2.5 Compliance Posture

| Sub-feature | Crate | Feature Flag | Status | Remaining Work |
|---|---|---|---|---|
| FIPS 140-2 compliant crypto | `oxirs-did`, `oxirs-fuseki` | `fips` | **Done** | `fips = []` feature gate added to both crates; FIPS boundary documented in `docs/policies/fips-boundary.md` — completed 2026-05-17 |
| SOC2 Type II evidence | `oxirs-core` | (via audit trail) | **Done** | Evidence collected via `AuditLogger` sinks (`JsonLineAuditLogger`, `CompositeAuditLogger`) — completed 2026-05-02 |
| GDPR data minimization | `oxirs-core` | (via audit trail) | **Done** | `AuditEvent::data_subject_id`, Article 30 mapping via `AuditEventKind` — completed 2026-05-02 |
| GDPR retention policies | `oxirs-core` | (via audit trail) | **Done** | `AuditFilter` with time-range predicates in `filter.rs` — completed 2026-05-02 |
| GDPR right-to-erasure | `oxirs-core` | (via audit trail) | **Done** | `GdprService::pseudonymise` in `gdpr.rs` — completed 2026-05-02 |

**FIPS feature gate design:**

```toml
# workspace Cargo.toml
[features]
fips = ["ring/fips"]   # ring supports FIPS 140-2 when built with boringssl
```

The `fips` feature must be additive and must not change the public API surface. Default features remain 100% pure Rust (no C/Fortran). The FIPS boundary document (`docs/policies/fips-boundary.md`) will enumerate which cryptographic operations are covered.

---

### 2.6 Enterprise Support SLA

This section documents response time targets applicable to enterprise customer agreements. These are operator-facing commitments, not code-level items.

| Priority | Description | First Response | Update Cadence | Resolution Target |
|---|---|---|---|---|
| P1 — Critical | Service unavailable, data loss risk, security breach | 1 hour | Every 2 hours | 24 hours |
| P2 — High | Major feature unavailable, significant performance degradation | 4 hours | Every 8 hours | 72 hours |
| P3 — Medium | Feature impaired, workaround available | 1 business day | Every 2 business days | 14 calendar days |
| P4 — Low | Minor issue, cosmetic, documentation gap | 3 business days | Weekly | Best effort |

Response times above apply during the Active Support phase of the LTS window (see `docs/policies/lts.md`, Section 3). During the Security-Only phase, only P1 tickets that constitute security incidents are in scope.

Enterprise customers should file issues via the GitHub Issues tracker with an `enterprise-support` label. Confidential security issues must use GitHub Security Advisories.

---

## 3. Implementation Priority Order

All originally-planned enterprise items are now complete as of 2026-05-17.

Historical sequence (completed):

1. **Audit module** (`oxirs-core/src/audit/`) — completed 2026-05-02.
2. **GDPR sub-features** (export via `data_subject_report`, purge via `pseudonymise`, retention via `AuditFilter`) — completed 2026-05-02.
3. **Enterprise RBAC policy templates** (`oxirs-fuseki/src/auth/policy_templates.rs`) — completed 2026-05-17.
4. **FIPS feature gate** (`fips = []` in `oxirs-did` and `oxirs-fuseki`, boundary doc RFC-003) — completed 2026-05-17.

---

## 4. Feature Map Summary

| Feature Area | Primary Crate | Status | Priority |
|---|---|---|---|
| SAML 2.0 SP | `oxirs-fuseki` | Done | — |
| OIDC client | `oxirs-fuseki` | Done | — |
| LDAP/AD integration | `oxirs-fuseki` | Done | — |
| Certificate client auth | `oxirs-fuseki` | Done | — |
| Core RBAC | `oxirs-fuseki` | Done | — |
| Graph-level RBAC | `oxirs-fuseki` | Done | — |
| ReBAC | `oxirs-fuseki` | Done | — |
| RBAC policy templates | `oxirs-fuseki` | Done (2026-05-17) | P2 |
| Audit trail module | `oxirs-core` | Done (2026-05-02) | P1 |
| GDPR export/purge | `oxirs-core` | Done (2026-05-02) | P1 |
| Namespace sharding | `oxirs-cluster` | Done | — |
| Vector multi-tenancy | `oxirs-vec` | Done | — |
| ARQ query governor | `oxirs-arq` | Done | — |
| Per-tenant SLA classes | `oxirs-core` | Done | — |
| FIPS 140-2 feature gate | `oxirs-did`, `oxirs-fuseki` | Done (2026-05-17) | P3 |
| SOC2 evidence collection | `oxirs-core` | Done (2026-05-02) | P1 |
| Enterprise support SLA | Ops/Policy | Defined (this doc) | — |

---

## 5. Applicability Across the Workspace

This decomposition applies to all 26 OxiRS crates. Each crate's TODO.md previously carried a single `- [ ] Enterprise features` line; that line now references this RFC and the specific child items relevant to each crate. Crates that do not own an enterprise feature directly (e.g., `oxirs-ttl`, `oxirs-canbus`) benefit from the workspace-level audit trail and SLA infrastructure delivered via `oxirs-core` and `oxirs-arq`.

---

*OxiRS Enterprise Features Policy RFC-002 — COOLJAPAN OU (Team Kitasan)*
