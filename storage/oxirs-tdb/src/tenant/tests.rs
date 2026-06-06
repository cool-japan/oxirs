use std::sync::Arc;

use super::isolation::{TenantHandle, TenantIsolationLayer, TenantStore};
use super::registry::TenantRegistry;
use super::types::{TenantConfig, TenantError, TenantId};

fn registry() -> Arc<TenantRegistry> {
    Arc::new(TenantRegistry::new())
}

fn create_tenant(registry: &TenantRegistry, name: &str) -> TenantId {
    let id = TenantId::new(name).unwrap();
    registry
        .create_tenant(id.clone(), TenantConfig::unlimited())
        .unwrap();
    id
}

#[test]
fn test_tenant_id_valid() {
    let id = TenantId::new("acme_corp-42").unwrap();
    assert_eq!(id.as_str(), "acme_corp-42");
}

#[test]
fn test_tenant_id_empty_rejected() {
    assert!(TenantId::new("").is_err());
}

#[test]
fn test_tenant_id_special_chars_rejected() {
    assert!(TenantId::new("bad/id").is_err());
    assert!(TenantId::new("bad id").is_err());
    assert!(TenantId::new("bad@id").is_err());
}

#[test]
fn test_tenant_id_too_long_rejected() {
    let long = "a".repeat(129);
    assert!(TenantId::new(&long).is_err());
}

#[test]
fn test_tenant_id_namespace_prefix() {
    let id = TenantId::new("testco").unwrap();
    assert_eq!(id.namespace_prefix(), "tenant:testco:");
}

#[test]
fn test_create_and_list_tenants() {
    let reg = registry();
    let id1 = TenantId::new("alpha").unwrap();
    let id2 = TenantId::new("beta").unwrap();
    reg.create_tenant(id1.clone(), TenantConfig::unlimited())
        .unwrap();
    reg.create_tenant(id2.clone(), TenantConfig::unlimited())
        .unwrap();
    let list = reg.list_tenants();
    assert_eq!(list.len(), 2);
    assert!(list.contains(&id1));
    assert!(list.contains(&id2));
}

#[test]
fn test_create_duplicate_tenant_fails() {
    let reg = registry();
    let id = TenantId::new("dup").unwrap();
    reg.create_tenant(id.clone(), TenantConfig::unlimited())
        .unwrap();
    let result = reg.create_tenant(id, TenantConfig::unlimited());
    assert!(matches!(result, Err(TenantError::AlreadyExists(_))));
}

#[test]
fn test_delete_tenant() {
    let reg = registry();
    let id = create_tenant(&reg, "todelete");
    reg.delete_tenant(&id).unwrap();
    assert!(!reg.exists(&id));
}

#[test]
fn test_delete_nonexistent_tenant_fails() {
    let reg = registry();
    let id = TenantId::new("ghost").unwrap();
    let result = reg.delete_tenant(&id);
    assert!(matches!(result, Err(TenantError::NotFound(_))));
}

#[test]
fn test_exists() {
    let reg = registry();
    let id = TenantId::new("exists_check").unwrap();
    assert!(!reg.exists(&id));
    reg.create_tenant(id.clone(), TenantConfig::unlimited())
        .unwrap();
    assert!(reg.exists(&id));
}

#[test]
fn test_insert_and_query() {
    let reg = registry();
    let id = create_tenant(&reg, "t1");
    let store = TenantStore::new(Arc::clone(&reg));

    store.insert(&id, "http://s1", "http://p", "obj").unwrap();
    let results = store.query(&id, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "http://s1");
}

#[test]
fn test_query_with_subject_filter() {
    let reg = registry();
    let id = create_tenant(&reg, "t2");
    let store = TenantStore::new(Arc::clone(&reg));

    store.insert(&id, "http://s1", "http://p", "a").unwrap();
    store.insert(&id, "http://s2", "http://p", "b").unwrap();
    let results = store.query(&id, Some("http://s1")).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].2, "a");
}

#[test]
fn test_tenant_isolation() {
    let reg = registry();
    let t1 = create_tenant(&reg, "tenant_a");
    let t2 = create_tenant(&reg, "tenant_b");
    let store = TenantStore::new(Arc::clone(&reg));

    store
        .insert(&t1, "http://secret", "http://p", "sensitive_value")
        .unwrap();

    let results_b = store.query(&t2, None).unwrap();
    assert!(
        results_b.is_empty(),
        "Tenant B must not see Tenant A's data"
    );
}

#[test]
fn test_cross_tenant_access_blocked_and_audited() {
    let reg = registry();
    let t1 = create_tenant(&reg, "accessor");
    let t2 = create_tenant(&reg, "target");
    let store = TenantStore::new(Arc::clone(&reg));

    let result = store.cross_tenant_access_check(&t1, &t2);
    assert!(matches!(result, Err(TenantError::CrossTenantAccess { .. })));

    let events = reg.audit_log().events();
    assert_eq!(events.len(), 1);
    assert!(events[0].blocked);
}

#[test]
fn test_triple_quota_enforced() {
    let reg = registry();
    let id = TenantId::new("limited").unwrap();
    reg.create_tenant(id.clone(), TenantConfig::with_limits(3, 0, 0))
        .unwrap();
    let store = TenantStore::new(Arc::clone(&reg));

    store.insert(&id, "s1", "p", "o").unwrap();
    store.insert(&id, "s2", "p", "o").unwrap();
    store.insert(&id, "s3", "p", "o").unwrap();
    let result = store.insert(&id, "s4", "p", "o");
    assert!(matches!(result, Err(TenantError::QuotaTriples { .. })));
}

#[test]
fn test_graph_quota_enforced() {
    let reg = registry();
    let id = TenantId::new("gquota").unwrap();
    reg.create_tenant(id.clone(), TenantConfig::with_limits(0, 2, 0))
        .unwrap();
    let store = TenantStore::new(Arc::clone(&reg));

    reg.pre_write_check(&id, "http://p", Some("graph1"))
        .unwrap();
    reg.pre_write_check(&id, "http://p", Some("graph2"))
        .unwrap();
    let result = reg.pre_write_check(&id, "http://p", Some("graph3"));
    assert!(matches!(result, Err(TenantError::QuotaGraphs { .. })));

    let _ = store;
}

#[test]
fn test_predicate_allowlist_enforced() {
    let reg = registry();
    let id = TenantId::new("strict").unwrap();
    let config = TenantConfig {
        max_triples: 0,
        max_graphs: 0,
        quota_bytes: 0,
        allowed_predicates: vec!["http://allowed".to_string()],
        allowed_prefixes: vec![],
        active: true,
    };
    reg.create_tenant(id.clone(), config).unwrap();
    let store = TenantStore::new(Arc::clone(&reg));

    store.insert(&id, "s", "http://allowed", "o").unwrap();
    let result = store.insert(&id, "s", "http://forbidden", "o");
    assert!(matches!(
        result,
        Err(TenantError::PredicateNotAllowed { .. })
    ));
}

#[test]
fn test_inactive_tenant_rejects_writes() {
    let reg = registry();
    let id = create_tenant(&reg, "inactive_t");
    reg.set_active(&id, false).unwrap();
    let store = TenantStore::new(Arc::clone(&reg));

    let result = store.insert(&id, "s", "p", "o");
    assert!(matches!(result, Err(TenantError::Inactive(_))));
}

#[test]
fn test_stats_triple_count_increments() {
    let reg = registry();
    let id = create_tenant(&reg, "stats_t");
    let store = TenantStore::new(Arc::clone(&reg));

    store.insert(&id, "s1", "p", "o").unwrap();
    store.insert(&id, "s2", "p", "o").unwrap();

    let stats = reg.stats(&id).unwrap();
    assert_eq!(stats.triple_count, 2);
    assert_eq!(stats.writes, 2);
}

#[test]
fn test_stats_reads_increments() {
    let reg = registry();
    let id = create_tenant(&reg, "read_stats");
    let store = TenantStore::new(Arc::clone(&reg));

    store.query(&id, None).unwrap();
    store.query(&id, None).unwrap();

    let stats = reg.stats(&id).unwrap();
    assert_eq!(stats.reads, 2);
}

#[test]
fn test_delete_subject() {
    let reg = registry();
    let id = create_tenant(&reg, "del_t");
    let store = TenantStore::new(Arc::clone(&reg));

    store.insert(&id, "s1", "p", "o").unwrap();
    store.insert(&id, "s1", "p2", "o2").unwrap();
    let removed = store.delete_subject(&id, "s1").unwrap();
    assert_eq!(removed, 2);
    let results = store.query(&id, None).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_purge_removes_all_tenant_data() {
    let reg = registry();
    let t1 = create_tenant(&reg, "purge_a");
    let t2 = create_tenant(&reg, "purge_b");
    let store = TenantStore::new(Arc::clone(&reg));

    store.insert(&t1, "s", "p", "o").unwrap();
    store.insert(&t2, "s", "p", "o").unwrap();

    let purged = store.purge(&t1).unwrap();
    assert_eq!(purged, 1);
    assert!(store.query(&t1, None).unwrap().is_empty());
    assert_eq!(store.query(&t2, None).unwrap().len(), 1);
}

#[test]
fn test_audit_log_records_cross_tenant() {
    let reg = registry();
    let t1 = create_tenant(&reg, "audit_a");
    let t2 = create_tenant(&reg, "audit_b");
    let store = TenantStore::new(Arc::clone(&reg));

    let _ = store.cross_tenant_access_check(&t1, &t2);
    let events = reg.audit_log().events_for_tenant(&t1);
    assert!(!events.is_empty());
    assert!(events.iter().all(|e| e.blocked));
}

#[test]
fn test_audit_log_same_tenant_no_event() {
    let reg = registry();
    let t1 = create_tenant(&reg, "self_access");
    let store = TenantStore::new(Arc::clone(&reg));

    let result = store.cross_tenant_access_check(&t1, &t1);
    assert!(result.is_ok());
    assert!(reg.audit_log().is_empty());
}

#[test]
fn test_update_config() {
    let reg = registry();
    let id = create_tenant(&reg, "update_cfg");
    let new_cfg = TenantConfig::with_limits(500, 10, 1024 * 1024);
    reg.update_config(&id, new_cfg.clone()).unwrap();
    let cfg = reg.config(&id).unwrap();
    assert_eq!(cfg.max_triples, 500);
    assert_eq!(cfg.max_graphs, 10);
}

#[test]
fn test_triple_count() {
    let reg = registry();
    let id = create_tenant(&reg, "tcount");
    let store = TenantStore::new(Arc::clone(&reg));

    assert_eq!(store.triple_count(&id).unwrap(), 0);
    store.insert(&id, "s1", "p", "o").unwrap();
    store.insert(&id, "s1", "p2", "o2").unwrap();
    assert_eq!(store.triple_count(&id).unwrap(), 2);
}

#[test]
fn test_allowed_prefixes_permits_matching_graph() {
    let reg = registry();
    let id = TenantId::new("prefix_ok").unwrap();
    let config = TenantConfig {
        max_triples: 0,
        max_graphs: 0,
        quota_bytes: 0,
        allowed_predicates: vec![],
        allowed_prefixes: vec!["urn:tenant:prefix_ok:".to_string()],
        active: true,
    };
    reg.create_tenant(id.clone(), config).unwrap();

    let result = reg.pre_write_check(&id, "http://p", Some("urn:tenant:prefix_ok:data"));
    assert!(result.is_ok());
}

#[test]
fn test_allowed_prefixes_rejects_non_matching_graph() {
    let reg = registry();
    let id = TenantId::new("prefix_rej").unwrap();
    let config = TenantConfig {
        max_triples: 0,
        max_graphs: 0,
        quota_bytes: 0,
        allowed_predicates: vec![],
        allowed_prefixes: vec!["urn:tenant:prefix_rej:".to_string()],
        active: true,
    };
    reg.create_tenant(id.clone(), config).unwrap();

    let result = reg.pre_write_check(&id, "http://p", Some("urn:other:graph"));
    assert!(matches!(
        result,
        Err(TenantError::GraphPrefixNotAllowed { .. })
    ));
}

#[test]
fn test_empty_allowed_prefixes_permits_any_graph() {
    let reg = registry();
    let id = TenantId::new("prefix_any").unwrap();
    reg.create_tenant(id.clone(), TenantConfig::unlimited())
        .unwrap();

    let result = reg.pre_write_check(&id, "http://p", Some("urn:any:graph:here"));
    assert!(result.is_ok());
}

#[test]
fn test_isolation_layer_create_and_get() {
    let layer = TenantIsolationLayer::new();
    let id = TenantId::new("iso_a").unwrap();
    let _handle = layer
        .create_tenant(id.clone(), TenantConfig::unlimited())
        .unwrap();

    assert!(layer.exists(&id));
    let retrieved = layer.get_tenant(&id);
    assert!(retrieved.is_some());
}

#[test]
fn test_isolation_layer_get_nonexistent() {
    let layer = TenantIsolationLayer::new();
    let id = TenantId::new("ghost_iso").unwrap();
    assert!(layer.get_tenant(&id).is_none());
}

#[test]
fn test_isolation_layer_list_tenants() {
    let layer = TenantIsolationLayer::new();
    layer
        .create_tenant(TenantId::new("t_alpha").unwrap(), TenantConfig::unlimited())
        .unwrap();
    layer
        .create_tenant(TenantId::new("t_beta").unwrap(), TenantConfig::unlimited())
        .unwrap();
    let list = layer.list_tenants();
    assert_eq!(list.len(), 2);
}

#[test]
fn test_isolation_layer_delete_tenant() {
    let layer = TenantIsolationLayer::new();
    let id = TenantId::new("del_iso").unwrap();
    let handle = layer
        .create_tenant(id.clone(), TenantConfig::unlimited())
        .unwrap();

    handle.insert_triple("s", "p", "o").unwrap();
    let deleted = layer.delete_tenant(&id).unwrap();
    assert_eq!(deleted, 1);
    assert!(!layer.exists(&id));
}

#[test]
fn test_isolation_layer_delete_returns_triple_count() {
    let layer = TenantIsolationLayer::new();
    let id = TenantId::new("del_count").unwrap();
    let handle = layer
        .create_tenant(id.clone(), TenantConfig::unlimited())
        .unwrap();

    for i in 0..5 {
        handle.insert_triple(&format!("s{}", i), "p", "o").unwrap();
    }
    let deleted = layer.delete_tenant(&id).unwrap();
    assert!(deleted > 0);
}

#[test]
fn test_isolation_layer_graph_iri() {
    let id = TenantId::new("giri").unwrap();
    let iri = TenantIsolationLayer::graph_iri(&id, "dataset1");
    assert_eq!(iri, "urn:tenant:giri:dataset1");
}

#[test]
fn test_handle_insert_and_query() {
    let layer = TenantIsolationLayer::new();
    let id = TenantId::new("handle_ins").unwrap();
    let handle = layer
        .create_tenant(id.clone(), TenantConfig::unlimited())
        .unwrap();

    handle
        .insert_triple("http://subject", "http://pred", "value")
        .unwrap();
    let results = handle.query(None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "http://subject");
}

#[test]
fn test_handle_triple_count() {
    let layer = TenantIsolationLayer::new();
    let handle = layer
        .create_tenant(TenantId::new("hcount").unwrap(), TenantConfig::unlimited())
        .unwrap();

    assert_eq!(handle.triple_count(), 0);
    handle.insert_triple("s1", "p", "o").unwrap();
    handle.insert_triple("s2", "p", "o").unwrap();
    assert_eq!(handle.triple_count(), 2);
}

#[test]
fn test_handle_graph_count() {
    let layer = TenantIsolationLayer::new();
    let config = TenantConfig::with_limits(0, 100, 0);
    let handle = layer
        .create_tenant(TenantId::new("hgraph").unwrap(), config)
        .unwrap();

    let reg = layer.registry();
    let id = handle.tenant_id().clone();
    reg.pre_write_check(&id, "http://p", Some("graph1"))
        .unwrap();
    reg.pre_write_check(&id, "http://p", Some("graph2"))
        .unwrap();
    reg.pre_write_check(&id, "http://p", Some("graph2"))
        .unwrap();

    assert_eq!(handle.graph_count(), 2);
}

#[test]
fn test_handle_delete_subject() {
    let layer = TenantIsolationLayer::new();
    let handle = layer
        .create_tenant(TenantId::new("hdel").unwrap(), TenantConfig::unlimited())
        .unwrap();

    handle.insert_triple("s1", "p1", "o1").unwrap();
    handle.insert_triple("s1", "p2", "o2").unwrap();
    let removed = handle.delete_subject("s1").unwrap();
    assert_eq!(removed, 2);
    assert_eq!(handle.triple_count(), 0);
}

#[test]
fn test_handle_purge() {
    let layer = TenantIsolationLayer::new();
    let handle = layer
        .create_tenant(TenantId::new("hpurge").unwrap(), TenantConfig::unlimited())
        .unwrap();

    handle.insert_triple("s", "p", "o").unwrap();
    let purged = handle.purge().unwrap();
    assert!(purged > 0);
}

#[test]
fn test_handle_stats() {
    let layer = TenantIsolationLayer::new();
    let handle = layer
        .create_tenant(TenantId::new("hstats").unwrap(), TenantConfig::unlimited())
        .unwrap();

    handle.insert_triple("s", "p", "o").unwrap();
    let stats = handle.stats().unwrap();
    assert_eq!(stats.triple_count, 1);
    assert_eq!(stats.writes, 1);
}

#[test]
fn test_handle_insert_in_graph_allowed() {
    let layer = TenantIsolationLayer::new();
    let id = TenantId::new("hgraph_ok").unwrap();
    let config = TenantConfig {
        max_triples: 0,
        max_graphs: 0,
        quota_bytes: 0,
        allowed_predicates: vec![],
        allowed_prefixes: vec!["urn:tenant:hgraph_ok:".to_string()],
        active: true,
    };
    let handle = layer.create_tenant(id, config).unwrap();

    let result = handle.insert_triple_in_graph("s", "p", "o", "urn:tenant:hgraph_ok:mydata");
    assert!(result.is_ok());
}

#[test]
fn test_handle_insert_in_graph_rejected() {
    let layer = TenantIsolationLayer::new();
    let id = TenantId::new("hgraph_rej").unwrap();
    let config = TenantConfig {
        max_triples: 0,
        max_graphs: 0,
        quota_bytes: 0,
        allowed_predicates: vec![],
        allowed_prefixes: vec!["urn:tenant:hgraph_rej:".to_string()],
        active: true,
    };
    let handle = layer.create_tenant(id, config).unwrap();

    let result = handle.insert_triple_in_graph("s", "p", "o", "urn:other:graph");
    assert!(matches!(
        result,
        Err(TenantError::GraphPrefixNotAllowed { .. })
    ));
}

#[test]
fn test_isolation_layer_create_duplicate_fails() {
    let layer = TenantIsolationLayer::new();
    let id = TenantId::new("dup_layer").unwrap();
    layer
        .create_tenant(id.clone(), TenantConfig::unlimited())
        .unwrap();
    let result = layer.create_tenant(id, TenantConfig::unlimited());
    assert!(matches!(result, Err(TenantError::AlreadyExists(_))));
}

#[test]
fn test_multiple_handles_share_store() {
    let layer = TenantIsolationLayer::new();
    let id = TenantId::new("shared_store").unwrap();
    let h1 = layer
        .create_tenant(id.clone(), TenantConfig::unlimited())
        .unwrap();
    let h2 = layer.get_tenant(&id).unwrap();

    h1.insert_triple("s", "p", "o").unwrap();
    assert_eq!(h2.triple_count(), 1);
}
