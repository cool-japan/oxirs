//! Write-routing rules for multi-region active-active deployments.
//!
//! Each incoming write is matched against an ordered list of
//! [`WriteRoutingRule`]s. The first rule whose `matches()` predicate returns
//! `true` decides the home region; if no rule matches, the table falls back
//! to its `default_region`.
//!
//! Routing decisions also consult the live region health snapshot from
//! [`super::health_probe::RegionHealthSnapshot`]: a region marked `Failed` is
//! skipped and its writes are deflected to the next eligible target taken
//! from the `failover_chain`.
//!
//! The routing table is intentionally schema-light — it operates on a
//! `(subject, tenant)` pair plus the write timestamp — so it can be embedded
//! anywhere the TSDB write path produces those fields.

use std::collections::BTreeSet;

use thiserror::Error;

use super::health_probe::{RegionHealthSnapshot, RegionStatus};

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`RoutingTable::route`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RoutingError {
    /// Every region the rule and its failover chain pointed at was unhealthy.
    #[error("no healthy region in routing chain (tried: {tried})")]
    NoHealthyRegion {
        /// Comma-separated list of regions that were considered.
        tried: String,
    },
    /// A rule referenced a region that the routing table doesn't know about.
    #[error("rule '{rule}' targets unknown region '{region}'")]
    UnknownRegion {
        /// Rule name from [`WriteRoutingRule::name`].
        rule: String,
        /// Region id that wasn't in the table.
        region: RegionId,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Region identifier (e.g. `us-east-1`).
pub type RegionId = String;

/// Context for a single routing decision.
#[derive(Debug, Clone)]
pub struct RouteContext {
    /// Subject IRI (or any string that uniquely identifies the write target).
    pub subject: String,
    /// Optional tenant identifier (used in multi-tenant deployments).
    pub tenant: Option<String>,
    /// Wall-clock timestamp of the write in Unix epoch milliseconds.
    pub timestamp_ms: i64,
}

impl RouteContext {
    /// Build a context with no tenant.
    pub fn new(subject: impl Into<String>, timestamp_ms: i64) -> Self {
        Self {
            subject: subject.into(),
            tenant: None,
            timestamp_ms,
        }
    }

    /// Builder helper to attach a tenant id.
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }
}

/// Final routing decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDecision {
    /// Region whose Raft group should accept the write.
    pub region: RegionId,
    /// Name of the rule that produced this decision (or `"default"` for the
    /// fallback path).
    pub rule_name: String,
    /// Whether the decision was made by failover from an unhealthy primary.
    pub via_failover: bool,
}

/// Single rule used by [`RoutingTable`].
#[derive(Debug, Clone)]
pub struct WriteRoutingRule {
    /// Human-readable rule name (used in [`RouteDecision::rule_name`]).
    pub name: String,
    /// Optional subject prefix the write must start with (`None` ⇒ match
    /// everything).
    pub subject_prefix: Option<String>,
    /// Optional tenant id the write must carry (`None` ⇒ match every tenant).
    pub tenant_id: Option<String>,
    /// Region that handles writes matched by this rule.
    pub primary_region: RegionId,
    /// Ordered list of regions to fall back to if `primary_region` is
    /// unhealthy.
    pub failover_chain: Vec<RegionId>,
}

impl WriteRoutingRule {
    /// Build a rule that matches every write and routes to `region`.
    pub fn default_to(region: impl Into<RegionId>) -> Self {
        Self {
            name: "default".into(),
            subject_prefix: None,
            tenant_id: None,
            primary_region: region.into(),
            failover_chain: Vec::new(),
        }
    }

    /// Builder: name the rule.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Builder: require a subject prefix.
    pub fn with_subject_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.subject_prefix = Some(prefix.into());
        self
    }

    /// Builder: require a tenant id.
    pub fn with_tenant_id(mut self, tenant: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant.into());
        self
    }

    /// Builder: append a failover region.
    pub fn with_failover(mut self, region: impl Into<RegionId>) -> Self {
        self.failover_chain.push(region.into());
        self
    }

    /// Returns `true` when the rule's predicates accept `ctx`.
    pub fn matches(&self, ctx: &RouteContext) -> bool {
        if let Some(prefix) = &self.subject_prefix {
            if !ctx.subject.starts_with(prefix) {
                return false;
            }
        }
        if let Some(want) = &self.tenant_id {
            match &ctx.tenant {
                Some(t) if t == want => {}
                _ => return false,
            }
        }
        true
    }
}

/// Ordered list of [`WriteRoutingRule`]s plus a default region.
#[derive(Debug, Clone)]
pub struct RoutingTable {
    rules: Vec<WriteRoutingRule>,
    default_region: RegionId,
    default_failover: Vec<RegionId>,
}

impl RoutingTable {
    /// Build a table with no rules; every write falls through to `default`.
    pub fn default_to(default: impl Into<RegionId>) -> Self {
        Self {
            rules: Vec::new(),
            default_region: default.into(),
            default_failover: Vec::new(),
        }
    }

    /// Append a rule (rules are evaluated in insertion order).
    pub fn push_rule(&mut self, rule: WriteRoutingRule) {
        self.rules.push(rule);
    }

    /// Configure failover targets that get tried when `default_region` is
    /// unhealthy.
    pub fn with_default_failover(mut self, chain: Vec<RegionId>) -> Self {
        self.default_failover = chain;
        self
    }

    /// Returns all rules in evaluation order (does not include the default
    /// fallback).
    pub fn rules(&self) -> &[WriteRoutingRule] {
        &self.rules
    }

    /// Returns the default region.
    pub fn default_region(&self) -> &RegionId {
        &self.default_region
    }

    /// Returns the default failover chain.
    pub fn default_failover(&self) -> &[RegionId] {
        &self.default_failover
    }

    /// Pick the home region for `ctx` honouring `health`.
    ///
    /// Algorithm:
    /// 1. Iterate rules in order; first that matches becomes the candidate.
    /// 2. Walk that rule's `[primary, failover…]` chain and return the first
    ///    region whose status is *not* `Failed`. The decision flag
    ///    `via_failover` is set when the primary was skipped.
    /// 3. If no rule matched, walk `[default_region, default_failover…]`.
    /// 4. Return [`RoutingError::NoHealthyRegion`] when every candidate was
    ///    `Failed`.
    pub fn route(
        &self,
        ctx: &RouteContext,
        health: &RegionHealthSnapshot,
    ) -> Result<RouteDecision, RoutingError> {
        for rule in &self.rules {
            if !rule.matches(ctx) {
                continue;
            }
            let chain = std::iter::once(&rule.primary_region).chain(rule.failover_chain.iter());
            return select_first_healthy(rule.name.clone(), chain, health);
        }
        let chain = std::iter::once(&self.default_region).chain(self.default_failover.iter());
        select_first_healthy("default".to_string(), chain, health)
    }

    /// Verify that every rule references a region present in `regions`.
    ///
    /// Returns the first violation encountered; this is intended as a
    /// post-build sanity check during deployment configuration.
    pub fn validate(&self, regions: &BTreeSet<RegionId>) -> Result<(), RoutingError> {
        for rule in &self.rules {
            if !regions.contains(&rule.primary_region) {
                return Err(RoutingError::UnknownRegion {
                    rule: rule.name.clone(),
                    region: rule.primary_region.clone(),
                });
            }
            for r in &rule.failover_chain {
                if !regions.contains(r) {
                    return Err(RoutingError::UnknownRegion {
                        rule: rule.name.clone(),
                        region: r.clone(),
                    });
                }
            }
        }
        if !regions.contains(&self.default_region) {
            return Err(RoutingError::UnknownRegion {
                rule: "default".into(),
                region: self.default_region.clone(),
            });
        }
        for r in &self.default_failover {
            if !regions.contains(r) {
                return Err(RoutingError::UnknownRegion {
                    rule: "default".into(),
                    region: r.clone(),
                });
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper
// ─────────────────────────────────────────────────────────────────────────────

fn select_first_healthy<'a, I>(
    rule_name: String,
    chain: I,
    health: &RegionHealthSnapshot,
) -> Result<RouteDecision, RoutingError>
where
    I: IntoIterator<Item = &'a RegionId>,
{
    let mut tried: Vec<RegionId> = Vec::new();
    let mut iter = chain.into_iter();
    let primary = match iter.next() {
        Some(r) => r,
        None => {
            return Err(RoutingError::NoHealthyRegion {
                tried: String::from("(empty chain)"),
            });
        }
    };
    tried.push(primary.clone());
    if health.status_of(primary) != RegionStatus::Failed {
        return Ok(RouteDecision {
            region: primary.clone(),
            rule_name,
            via_failover: false,
        });
    }
    for fallback in iter {
        tried.push(fallback.clone());
        if health.status_of(fallback) != RegionStatus::Failed {
            return Ok(RouteDecision {
                region: fallback.clone(),
                rule_name,
                via_failover: true,
            });
        }
    }
    Err(RoutingError::NoHealthyRegion {
        tried: tried.join(", "),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_region::HealthConfig;
    use crate::multi_region::HealthProbe;

    fn snapshot(regions: &[(&str, RegionStatus)]) -> RegionHealthSnapshot {
        let names: Vec<RegionId> = regions.iter().map(|(r, _)| r.to_string()).collect();
        let mut probe = HealthProbe::new(names, HealthConfig::default());
        for (r, s) in regions {
            probe.force_status(*r, *s);
        }
        probe.snapshot()
    }

    #[test]
    fn rule_matches_subject_prefix() {
        let rule = WriteRoutingRule::default_to("us").with_subject_prefix("metrics.cpu");
        assert!(rule.matches(&RouteContext::new("metrics.cpu.host01", 1)));
        assert!(!rule.matches(&RouteContext::new("metrics.mem.host01", 1)));
    }

    #[test]
    fn rule_matches_tenant() {
        let rule = WriteRoutingRule::default_to("us").with_tenant_id("tenant-a");
        assert!(rule.matches(&RouteContext::new("foo", 1).with_tenant("tenant-a")));
        assert!(!rule.matches(&RouteContext::new("foo", 1)));
        assert!(!rule.matches(&RouteContext::new("foo", 1).with_tenant("tenant-b")));
    }

    #[test]
    fn route_uses_first_matching_rule() {
        let mut table = RoutingTable::default_to("us");
        table.push_rule(
            WriteRoutingRule::default_to("eu")
                .named("eu-customers")
                .with_tenant_id("eu-tenant"),
        );
        table.push_rule(
            WriteRoutingRule::default_to("ap")
                .named("ap-customers")
                .with_tenant_id("ap-tenant"),
        );
        let snap = snapshot(&[
            ("us", RegionStatus::Healthy),
            ("eu", RegionStatus::Healthy),
            ("ap", RegionStatus::Healthy),
        ]);
        let dec = table
            .route(&RouteContext::new("foo", 1).with_tenant("eu-tenant"), &snap)
            .expect("route");
        assert_eq!(dec.region, "eu");
        assert_eq!(dec.rule_name, "eu-customers");
        assert!(!dec.via_failover);
    }

    #[test]
    fn route_falls_back_to_default() {
        let table = RoutingTable::default_to("us");
        let snap = snapshot(&[("us", RegionStatus::Healthy)]);
        let dec = table
            .route(&RouteContext::new("anything", 0), &snap)
            .unwrap();
        assert_eq!(dec.region, "us");
        assert_eq!(dec.rule_name, "default");
    }

    #[test]
    fn route_fails_over_when_primary_failed() {
        let mut table = RoutingTable::default_to("us");
        table.push_rule(
            WriteRoutingRule::default_to("us")
                .with_failover("eu")
                .with_failover("ap")
                .named("primary-us"),
        );
        let snap = snapshot(&[
            ("us", RegionStatus::Failed),
            ("eu", RegionStatus::Healthy),
            ("ap", RegionStatus::Healthy),
        ]);
        let dec = table
            .route(&RouteContext::new("subj", 1), &snap)
            .expect("route");
        assert_eq!(dec.region, "eu");
        assert!(dec.via_failover);
    }

    #[test]
    fn route_skips_failed_default() {
        let table =
            RoutingTable::default_to("us").with_default_failover(vec!["eu".into(), "ap".into()]);
        let snap = snapshot(&[
            ("us", RegionStatus::Failed),
            ("eu", RegionStatus::Failed),
            ("ap", RegionStatus::Healthy),
        ]);
        let dec = table
            .route(&RouteContext::new("anything", 0), &snap)
            .unwrap();
        assert_eq!(dec.region, "ap");
        assert!(dec.via_failover);
    }

    #[test]
    fn route_errors_when_all_unhealthy() {
        let table = RoutingTable::default_to("us").with_default_failover(vec!["eu".into()]);
        let snap = snapshot(&[("us", RegionStatus::Failed), ("eu", RegionStatus::Failed)]);
        let err = table
            .route(&RouteContext::new("anything", 0), &snap)
            .unwrap_err();
        match err {
            RoutingError::NoHealthyRegion { tried } => {
                assert!(tried.contains("us"));
                assert!(tried.contains("eu"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_unknown_regions() {
        let mut table = RoutingTable::default_to("us");
        table.push_rule(WriteRoutingRule::default_to("ghost").named("ghost"));
        let mut regions = BTreeSet::new();
        regions.insert("us".to_string());
        assert!(table.validate(&regions).is_err());
        regions.insert("ghost".to_string());
        assert!(table.validate(&regions).is_ok());
    }

    #[test]
    fn suspect_regions_still_accepted() {
        // Suspect (degraded but not failed) should still be picked.
        let table = RoutingTable::default_to("us");
        let snap = snapshot(&[("us", RegionStatus::Suspect)]);
        let dec = table
            .route(&RouteContext::new("anything", 0), &snap)
            .unwrap();
        assert_eq!(dec.region, "us");
        assert!(!dec.via_failover);
    }

    #[test]
    fn rule_priority_order_respected() {
        let mut table = RoutingTable::default_to("default-region");
        table.push_rule(WriteRoutingRule::default_to("first").named("first-rule"));
        // Second rule would also match but the first one wins.
        table.push_rule(WriteRoutingRule::default_to("second").named("second-rule"));
        let snap = snapshot(&[
            ("first", RegionStatus::Healthy),
            ("second", RegionStatus::Healthy),
            ("default-region", RegionStatus::Healthy),
        ]);
        let dec = table.route(&RouteContext::new("x", 0), &snap).unwrap();
        assert_eq!(dec.region, "first");
        assert_eq!(dec.rule_name, "first-rule");
    }

    #[test]
    fn empty_chain_returns_error() {
        // Build by hand to test the defensive path.
        let snap = RegionHealthSnapshot::default();
        let dec = select_first_healthy::<std::iter::Empty<&RegionId>>(
            "weird".into(),
            std::iter::empty(),
            &snap,
        );
        assert!(dec.is_err());
    }
}
