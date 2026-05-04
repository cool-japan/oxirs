# OxiRS Long-Term Support Policy

*RFC-001 | Effective: 2026-05-01 | Owner: COOLJAPAN OU (Team Kitasan)*

---

## 1. Purpose

This document defines the Long-Term Support (LTS) policy for the OxiRS workspace. It provides a concrete, operational framework covering LTS designation criteria, support windows, backport rules, security response SLAs, branch lifecycle, and EOL communication. All 26 crates in the workspace inherit this policy.

---

## 2. LTS Designation Criteria

A release is designated **LTS** when:

- The Git branch name is `X.Y.0` where `X ≥ 1` and `Y` is even (e.g., `1.0.0`, `1.2.0`, `2.0.0`).
- The branch was cut from `master` by an authorized maintainer and tagged `vX.Y.0`.

Point releases on an LTS branch (e.g., `1.0.1`, `1.0.2`) are **automatically LTS** and carry the same support window as the parent branch.

Non-LTS releases are any branch where `Y` is odd (e.g., `1.1.0`, `1.3.0`), plus all pre-`1.0.0` branches. These receive limited security-only support as described in Section 4.

### Designation Examples

| Branch / Tag | LTS? | Reason |
|---|---|---|
| `1.0.0` / `1.0.x` | Yes | X=1 ≥ 1, Y=0 (even) |
| `1.1.0` / `1.1.x` | No | Y=1 (odd) |
| `1.2.0` / `1.2.x` | Yes | X=1 ≥ 1, Y=2 (even) |
| `2.0.0` / `2.0.x` | Yes | X=2 ≥ 1, Y=0 (even) |
| `0.3.0` / `0.x.y` | No | X=0 (pre-stable) |

---

## 3. Support Windows

### LTS Releases

| Phase | Duration | Coverage |
|---|---|---|
| Active Support | 24 months from branch cut date | Bug fixes, security patches, correctness fixes, performance regressions |
| Security-Only | 12 months after active support ends (months 25–36) | Security patches only (CVSS ≥ 4.0) |
| End of Life (EOL) | After month 36 | No further patches; branch archived read-only |

Total LTS support window: **36 months** from branch cut date.

### Non-LTS (Regular) Releases

| Phase | Duration | Coverage |
|---|---|---|
| Active Support | Until next minor release ships | Bug fixes and security patches |
| Security-Only | 6 months from branch cut date | Security patches only (CVSS ≥ 7.0) |
| End of Life | After 6 months | No further patches |

---

## 4. Backport Criteria

The following criteria govern what changes are admitted to LTS branches after the branch cut.

### Always Backported

- **Security CVEs** with CVSS ≥ 7.0 (High or Critical severity).
- **Correctness bugs** with demonstrated data-loss risk (silent data corruption, incorrect query results that overwrite stored data, serialization bugs that destroy information).

### Never Backported

- New features of any kind.
- Performance improvements that do not fix a correctness or security issue.
- API additions or extensions.
- Dependency upgrades that are not security-driven.
- Refactoring or code style changes.

### Case-by-Case (maintainer discretion)

- Correctness bugs without data-loss risk but with significant user impact.
- Dependency upgrades required to resolve a transitive security advisory.
- Build system fixes required to compile on current stable Rust toolchain.

All backport decisions are recorded in the GitHub pull request against the LTS branch with a `backport: lts-X.Y` label and a justification comment referencing one of the above criteria.

---

## 5. Security Response SLA

| CVSS Score Range | Severity | Patch Deadline |
|---|---|---|
| 9.0 – 10.0 | Critical | 7 calendar days from confirmed report |
| 7.0 – 8.9 | High | 30 calendar days from confirmed report |
| 4.0 – 6.9 | Medium | 90 calendar days from confirmed report |
| 0.1 – 3.9 | Low | Best effort; no hard deadline |

The clock starts when a report is confirmed by a maintainer (i.e., triage is complete and the vulnerability is accepted). Reports submitted via GitHub Security Advisories are preferred; the triage SLA for initial acknowledgement is 3 business days.

Security patches ship as point releases on all active LTS branches simultaneously. Non-LTS branches in their security-only window receive patches for High and Critical CVEs only.

---

## 6. Release Branch Lifecycle

```
master ──┬──────────────────────────────────────────►
         │
    branch cut
         │
    1.0.0 ─── 1.0.1 ─── 1.0.2 ─── ... ─── 1.0.N
         │
         │◄──── active support (24 months) ────►│◄── security-only (12 months) ────►│ EOL
```

### Rules After Branch Cut

1. **Bug-fix only**: only patches meeting the backport criteria (Section 4) are admitted.
2. **Merge path**: patches are first merged to `master`, then cherry-picked to the LTS branch via a dedicated pull request. Direct commits to an LTS branch without a corresponding `master` commit are prohibited unless the bug is LTS-specific.
3. **Tagging**: each point release on the LTS branch is tagged `vX.Y.Z` and published to crates.io when explicitly authorised.
4. **Rust edition**: the LTS branch pins the minimum supported Rust version (MSRV) at the time of branch cut. MSRV is never bumped on an LTS branch except to resolve a security issue with no other mitigation.

### EOL Announcement

EOL for an LTS branch is announced no fewer than **90 days** before the EOL date via:

- A GitHub Release entry tagged `eol-notice-X.Y`.
- A pinned notice in the GitHub repository.
- A GitHub Security Advisory (informational) linking to the successor LTS branch.

---

## 7. Communication Channels

| Event | Channel |
|---|---|
| New LTS designation | GitHub Releases page, tagged `lts` |
| Security patch release | GitHub Security Advisories + GitHub Releases |
| EOL notice | GitHub Releases (90-day advance) + pinned repo notice |
| MSRV changes | GitHub Releases release notes |

---

## 8. Current LTS Status

| Branch | Tag | LTS? | Branch Cut Date | Active Support Until | Security-Only Until | EOL |
|---|---|---|---|---|---|---|
| `1.0.0` | `v1.0.0` | **Yes (current LTS)** | 2026-02-25 | 2028-02-25 | 2029-02-25 | 2029-02-25 |

All crates in the workspace at `master` HEAD as of 2026-02-25 are covered by the `v1.0.0` LTS designation.

The next planned LTS branch is `1.2.0` (target: Q4 2026, pending roadmap confirmation).

---

## 9. Applicability Across the Workspace

This policy applies uniformly to all 26 crates in the OxiRS workspace:

`oxirs-core`, `oxirs-fuseki`, `oxirs-gql`, `oxirs-arq`, `oxirs-rule`, `oxirs-shacl`,
`oxirs-samm`, `oxirs-geosparql`, `oxirs-vec`, `oxirs-ttl`, `oxirs-star`, `oxirs-tdb`,
`oxirs-cluster`, `oxirs-tsdb`, `oxirs-stream`, `oxirs-federate`, `oxirs-modbus`,
`oxirs-canbus`, `oxirs-embed`, `oxirs-shacl-ai`, `oxirs-chat`, `oxirs-physics`,
`oxirs-graphrag`, `oxirs-did`, `oxirs-wasm`, `oxirs` (tools).

Each crate's TODO.md references this document for LTS items.

---

*OxiRS LTS Policy RFC-001 — COOLJAPAN OU (Team Kitasan)*
