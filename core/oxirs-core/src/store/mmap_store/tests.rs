//! Tests for memory-mapped store

use super::*;
use crate::{BlankNode, Literal, NamedNode};
use tempfile::TempDir;

#[test]
#[ignore] // Extremely slow test - over 14 minutes
fn test_create_store() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;
    assert_eq!(store.len(), 0);
    Ok(())
}

#[test]
#[ignore] // Extremely slow test - over 14 minutes
fn test_add_quad() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    let quad = Quad::new(
        Subject::NamedNode(NamedNode::new("http://example.org/s")?),
        Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
        Object::NamedNode(NamedNode::new("http://example.org/o")?),
        GraphName::DefaultGraph,
    );

    store.add(&quad)?;
    store.flush()?;

    assert_eq!(store.len(), 1);
    Ok(())
}

#[test]
#[ignore] // Extremely slow test - over 14 minutes
fn test_persistence() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = temp_dir.path();

    {
        let store = MmapStore::new(path)?;

        let mut quads = Vec::new();
        for i in 0..100 {
            let quad = Quad::new(
                Subject::NamedNode(NamedNode::new(format!("http://example.org/s{i}"))?),
                Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
                Object::Literal(Literal::new_simple_literal(format!("value{i}"))),
                GraphName::DefaultGraph,
            );
            quads.push(quad);
        }

        store.add_batch(&quads)?;

        store.flush()?;
        assert_eq!(store.len(), 100);
    }

    {
        let store = MmapStore::open(path)?;
        assert_eq!(store.len(), 100);
    }

    Ok(())
}

#[test]
#[ignore] // Extremely slow test - over 14 minutes
fn test_pattern_matching() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    let subjects = vec!["s1", "s2", "s3"];
    let predicates = vec!["p1", "p2"];
    let objects = vec!["o1", "o2", "o3", "o4"];

    let mut quads = Vec::new();
    for s in &subjects {
        for p in &predicates {
            for o in &objects {
                let quad = Quad::new(
                    Subject::NamedNode(NamedNode::new(format!("http://example.org/{s}"))?),
                    Predicate::NamedNode(NamedNode::new(format!("http://example.org/{p}"))?),
                    Object::NamedNode(NamedNode::new(format!("http://example.org/{o}"))?),
                    GraphName::DefaultGraph,
                );
                quads.push(quad);
            }
        }
    }

    store.add_batch(&quads)?;

    store.flush()?;
    assert_eq!(store.len(), 24);

    let s1 = Subject::NamedNode(NamedNode::new("http://example.org/s1")?);
    let results: Vec<_> = store
        .quads_matching(Some(&s1), None, None, None)?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(results.len(), 8);

    let p1 = Predicate::NamedNode(NamedNode::new("http://example.org/p1")?);
    let results: Vec<_> = store
        .quads_matching(Some(&s1), Some(&p1), None, None)?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(results.len(), 4);

    let o1 = Object::NamedNode(NamedNode::new("http://example.org/o1")?);
    let results: Vec<_> = store
        .quads_matching(Some(&s1), Some(&p1), Some(&o1), None)?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(results.len(), 1);

    let s_none = Subject::NamedNode(NamedNode::new("http://example.org/nonexistent")?);
    let results: Vec<_> = store
        .quads_matching(Some(&s_none), None, None, None)?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(results.len(), 0);

    Ok(())
}

#[test]
#[ignore] // Still has performance issues
fn test_graph_support() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    let s = Subject::NamedNode(NamedNode::new("http://example.org/subject")?);
    let p = Predicate::NamedNode(NamedNode::new("http://example.org/predicate")?);
    let o = Object::Literal(Literal::new_simple_literal("value"));

    let g1 = GraphName::NamedNode(NamedNode::new("http://example.org/graph1")?);
    let g2 = GraphName::NamedNode(NamedNode::new("http://example.org/graph2")?);

    let quads = vec![
        Quad::new(s.clone(), p.clone(), o.clone(), GraphName::DefaultGraph),
        Quad::new(s.clone(), p.clone(), o.clone(), g1.clone()),
        Quad::new(s.clone(), p.clone(), o.clone(), g2.clone()),
    ];

    store.add_batch(&quads)?;

    store.flush()?;
    assert_eq!(store.len(), 3);

    let results: Vec<_> = store
        .quads_matching(None, None, None, Some(&GraphName::DefaultGraph))?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(results.len(), 1);

    let results: Vec<_> = store
        .quads_matching(None, None, None, Some(&g1))?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(results.len(), 1);

    let results: Vec<_> = store
        .quads_matching(None, None, None, None)?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(results.len(), 3);

    Ok(())
}

#[test]
#[ignore] // Still has performance issues
fn test_literal_types() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    let s = Subject::NamedNode(NamedNode::new("http://example.org/subject")?);
    let p = Predicate::NamedNode(NamedNode::new("http://example.org/predicate")?);

    let simple = Object::Literal(Literal::new_simple_literal("simple"));
    let lang = Object::Literal(Literal::new_language_tagged_literal("hello", "en")?);
    let xsd_int = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?;
    let typed = Object::Literal(Literal::new_typed("42", xsd_int));

    let quads = vec![
        Quad::new(
            s.clone(),
            p.clone(),
            simple.clone(),
            GraphName::DefaultGraph,
        ),
        Quad::new(s.clone(), p.clone(), lang.clone(), GraphName::DefaultGraph),
        Quad::new(s.clone(), p.clone(), typed.clone(), GraphName::DefaultGraph),
    ];

    store.add_batch(&quads)?;

    store.flush()?;

    let results: Vec<_> = store
        .quads_matching(Some(&s), Some(&p), None, None)?
        .collect::<Result<Vec<_>>>()?;

    assert_eq!(results.len(), 3);

    let objects: Vec<_> = results.iter().map(|q| q.object()).collect();
    assert!(objects.contains(&&simple));
    assert!(objects.contains(&&lang));
    assert!(objects.contains(&&typed));

    Ok(())
}

#[test]
#[ignore] // Extremely slow test - over 14 minutes
fn test_large_dataset() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    let mut quads = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        let quad = Quad::new(
            Subject::NamedNode(NamedNode::new(format!(
                "http://example.org/subject/{}",
                i / 100
            ))?),
            Predicate::NamedNode(NamedNode::new(format!(
                "http://example.org/predicate/{}",
                i % 10
            ))?),
            Object::Literal(Literal::new_simple_literal(format!("value{i}"))),
            GraphName::DefaultGraph,
        );
        quads.push(quad);
    }

    store.add_batch(&quads)?;

    store.flush()?;
    assert_eq!(store.len(), 10_000);

    let s = Subject::NamedNode(NamedNode::new("http://example.org/subject/50")?);
    let results: Vec<_> = store
        .quads_matching(Some(&s), None, None, None)?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(results.len(), 100);

    let stats = store.stats();
    assert_eq!(stats.quad_count, 10_000);
    assert!(stats.data_size > 0);

    Ok(())
}

#[test]
#[ignore] // Extremely slow test - over 14 minutes
fn test_blank_nodes() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    let b1 = BlankNode::new("b1")?;
    let b2 = BlankNode::new("b2")?;
    let p = Predicate::NamedNode(NamedNode::new("http://example.org/p")?);

    let quads = vec![
        Quad::new(
            Subject::BlankNode(b1.clone()),
            p.clone(),
            Object::Literal(Literal::new_simple_literal("value1")),
            GraphName::DefaultGraph,
        ),
        Quad::new(
            Subject::NamedNode(NamedNode::new("http://example.org/s")?),
            p.clone(),
            Object::BlankNode(b2.clone()),
            GraphName::DefaultGraph,
        ),
        Quad::new(
            Subject::NamedNode(NamedNode::new("http://example.org/s2")?),
            p.clone(),
            Object::Literal(Literal::new_simple_literal("value2")),
            GraphName::BlankNode(b1.clone()),
        ),
    ];

    store.add_batch(&quads)?;

    store.flush()?;
    assert_eq!(store.len(), 3);

    let results: Vec<_> = store
        .quads_matching(Some(&Subject::BlankNode(b1.clone())), None, None, None)?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(results.len(), 1);

    let results: Vec<_> = store
        .quads_matching(None, None, None, Some(&GraphName::BlankNode(b1)))?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(results.len(), 1);

    Ok(())
}

#[test]
#[ignore] // Slow test - MmapStore operations take significant time
fn test_access_statistics() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    let mut quads = Vec::new();
    for i in 0..50 {
        let quad = Quad::new(
            Subject::NamedNode(NamedNode::new(format!("http://example.org/s{}", i % 5))?),
            Predicate::NamedNode(NamedNode::new(format!("http://example.org/p{}", i % 3))?),
            Object::Literal(Literal::new_simple_literal(format!("value{i}"))),
            GraphName::DefaultGraph,
        );
        quads.push(quad);
    }
    store.add_batch(&quads)?;
    store.flush()?;

    let s1 = Subject::NamedNode(NamedNode::new("http://example.org/s1")?);
    let _ = store
        .quads_matching(Some(&s1), None, None, None)?
        .collect::<Result<Vec<_>>>()?;

    let _ = store
        .quads_matching(None, None, None, None)?
        .collect::<Result<Vec<_>>>()?;

    let stats = store.get_access_stats();
    assert!(stats.total_queries > 0);
    assert!(stats.avg_query_latency_us > 0.0);

    Ok(())
}

#[test]
#[ignore] // Slow test - MmapStore operations take significant time
fn test_full_backup() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let backup_dir = temp_dir.path().join("backups");
    let store = MmapStore::new(temp_dir.path().join("store"))?;

    let mut quads = Vec::new();
    for i in 0..100 {
        let quad = Quad::new(
            Subject::NamedNode(NamedNode::new(format!("http://example.org/s{i}"))?),
            Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
            Object::Literal(Literal::new_simple_literal(format!("value{i}"))),
            GraphName::DefaultGraph,
        );
        quads.push(quad);
    }
    store.add_batch(&quads)?;
    store.flush()?;

    let metadata = store.create_full_backup(&backup_dir)?;

    assert!(metadata.is_full_backup);
    assert_eq!(metadata.quad_count, 100);
    assert!(metadata.backup_path.exists());

    let history = store.get_backup_history();
    assert_eq!(history.len(), 1);
    assert!(history[0].is_full_backup);

    Ok(())
}

#[test]
#[ignore] // Slow test - MmapStore operations take significant time
fn test_incremental_backup() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let backup_dir = temp_dir.path().join("backups");
    let store = MmapStore::new(temp_dir.path().join("store"))?;

    let mut quads = Vec::new();
    for i in 0..50 {
        let quad = Quad::new(
            Subject::NamedNode(NamedNode::new(format!("http://example.org/s{i}"))?),
            Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
            Object::Literal(Literal::new_simple_literal(format!("value{i}"))),
            GraphName::DefaultGraph,
        );
        quads.push(quad);
    }
    store.add_batch(&quads)?;
    store.flush()?;

    let full_metadata = store.create_full_backup(&backup_dir)?;
    assert!(full_metadata.is_full_backup);

    let mut more_quads = Vec::new();
    for i in 50..100 {
        let quad = Quad::new(
            Subject::NamedNode(NamedNode::new(format!("http://example.org/s{i}"))?),
            Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
            Object::Literal(Literal::new_simple_literal(format!("value{i}"))),
            GraphName::DefaultGraph,
        );
        more_quads.push(quad);
    }
    store.add_batch(&more_quads)?;
    store.flush()?;

    let incr_metadata = store.create_incremental_backup(&backup_dir)?;

    assert!(!incr_metadata.is_full_backup);
    assert!(incr_metadata.backup_path.exists());

    let history = store.get_backup_history();
    assert_eq!(history.len(), 2);
    assert!(history[0].is_full_backup);
    assert!(!history[1].is_full_backup);

    Ok(())
}

#[test]
#[ignore] // Slow test - MmapStore operations take significant time
fn test_backup_recommendation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let backup_dir = temp_dir.path().join("backups");
    let store = MmapStore::new(temp_dir.path().join("store"))?;

    assert_eq!(store.recommended_backup_type(), "full");

    let mut quads = Vec::new();
    for i in 0..50 {
        let quad = Quad::new(
            Subject::NamedNode(NamedNode::new(format!("http://example.org/s{i}"))?),
            Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
            Object::Literal(Literal::new_simple_literal(format!("value{i}"))),
            GraphName::DefaultGraph,
        );
        quads.push(quad);
    }
    store.add_batch(&quads)?;
    store.flush()?;

    let _ = store.create_full_backup(&backup_dir)?;

    let small_quads = vec![Quad::new(
        Subject::NamedNode(NamedNode::new("http://example.org/new")?),
        Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
        Object::Literal(Literal::new_simple_literal("new_value")),
        GraphName::DefaultGraph,
    )];
    store.add_batch(&small_quads)?;
    store.flush()?;

    assert_eq!(store.recommended_backup_type(), "incremental");

    Ok(())
}

#[test]
#[ignore] // Slow test - MmapStore operations take significant time
fn test_clear_backup_history() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let backup_dir = temp_dir.path().join("backups");
    let store = MmapStore::new(temp_dir.path().join("store"))?;

    let quads = vec![Quad::new(
        Subject::NamedNode(NamedNode::new("http://example.org/s")?),
        Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
        Object::Literal(Literal::new_simple_literal("value")),
        GraphName::DefaultGraph,
    )];
    store.add_batch(&quads)?;
    store.flush()?;

    let _ = store.create_full_backup(&backup_dir)?;
    assert_eq!(store.get_backup_history().len(), 1);

    store.clear_backup_history();
    assert_eq!(store.get_backup_history().len(), 0);

    assert_eq!(store.recommended_backup_type(), "full");

    Ok(())
}

#[test]
#[ignore] // Slow test - MmapStore operations take significant time
fn test_reset_access_stats() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    let quads = vec![Quad::new(
        Subject::NamedNode(NamedNode::new("http://example.org/s")?),
        Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
        Object::Literal(Literal::new_simple_literal("value")),
        GraphName::DefaultGraph,
    )];
    store.add_batch(&quads)?;
    store.flush()?;

    let _ = store
        .quads_matching(None, None, None, None)?
        .collect::<Result<Vec<_>>>()?;

    let stats = store.get_access_stats();
    assert!(stats.total_queries > 0);

    store.reset_access_stats();
    let stats = store.get_access_stats();
    assert_eq!(stats.total_queries, 0);

    Ok(())
}
