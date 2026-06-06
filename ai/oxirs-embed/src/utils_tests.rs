//! Tests for utils modules

#[cfg(test)]
mod tests {
    use crate::quick_start::{
        add_triples_from_strings, cosine_similarity, create_simple_transe_model,
        generate_sample_kg_data, parse_triple_from_string, quick_performance_test,
    };
    use crate::utils_io::{data_loader, dataset_splitter};
    use crate::utils_math::compute_dataset_statistics;
    use crate::EmbeddingModel;
    use anyhow::Result;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_triples_from_tsv() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "subject\tpredicate\tobject")?;
        writeln!(temp_file, "alice\tknows\tbob")?;
        writeln!(temp_file, "bob\tlikes\tcharlie")?;

        let triples = data_loader::load_triples_from_tsv(temp_file.path())?;
        assert_eq!(triples.len(), 2);
        assert_eq!(
            triples[0],
            ("alice".to_string(), "knows".to_string(), "bob".to_string())
        );

        Ok(())
    }

    #[test]
    fn test_dataset_split() -> Result<()> {
        let triples = vec![
            ("a".to_string(), "r1".to_string(), "b".to_string()),
            ("b".to_string(), "r2".to_string(), "c".to_string()),
            ("c".to_string(), "r3".to_string(), "d".to_string()),
            ("d".to_string(), "r4".to_string(), "e".to_string()),
        ];

        let split = dataset_splitter::split_dataset(triples, 0.6, 0.2, Some(42))?;

        assert_eq!(split.train.len(), 2);
        assert_eq!(split.validation.len(), 0);
        assert_eq!(split.test.len(), 2);

        Ok(())
    }

    #[test]
    fn test_load_triples_from_jsonl() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(
            temp_file,
            r#"{{"subject": "alice", "predicate": "knows", "object": "bob"}}"#
        )?;
        writeln!(
            temp_file,
            r#"{{"subject": "bob", "predicate": "likes", "object": "charlie"}}"#
        )?;

        let triples = data_loader::load_triples_from_jsonl(temp_file.path())?;
        assert_eq!(triples.len(), 2);
        assert_eq!(
            triples[0],
            ("alice".to_string(), "knows".to_string(), "bob".to_string())
        );

        Ok(())
    }

    #[test]
    fn test_save_triples_to_jsonl() -> Result<()> {
        let triples = vec![
            ("alice".to_string(), "knows".to_string(), "bob".to_string()),
            (
                "bob".to_string(),
                "likes".to_string(),
                "charlie".to_string(),
            ),
        ];

        let temp_file = NamedTempFile::new()?;
        data_loader::save_triples_to_jsonl(&triples, temp_file.path())?;

        let loaded_triples = data_loader::load_triples_from_jsonl(temp_file.path())?;
        assert_eq!(loaded_triples, triples);

        Ok(())
    }

    #[test]
    fn test_load_triples_auto_detect() -> Result<()> {
        let mut tsv_file = NamedTempFile::with_suffix(".tsv")?;
        writeln!(tsv_file, "subject\tpredicate\tobject")?;
        writeln!(tsv_file, "alice\tknows\tbob")?;

        let triples = data_loader::load_triples_auto_detect(tsv_file.path())?;
        assert_eq!(triples.len(), 1);

        let mut jsonl_file = NamedTempFile::with_suffix(".jsonl")?;
        writeln!(
            jsonl_file,
            r#"{{"subject": "alice", "predicate": "knows", "object": "bob"}}"#
        )?;

        let triples = data_loader::load_triples_auto_detect(jsonl_file.path())?;
        assert_eq!(triples.len(), 1);
        assert_eq!(
            triples[0],
            ("alice".to_string(), "knows".to_string(), "bob".to_string())
        );

        Ok(())
    }

    #[test]
    fn test_dataset_statistics() {
        let triples = vec![
            ("alice".to_string(), "knows".to_string(), "bob".to_string()),
            (
                "bob".to_string(),
                "knows".to_string(),
                "charlie".to_string(),
            ),
            (
                "alice".to_string(),
                "likes".to_string(),
                "charlie".to_string(),
            ),
        ];

        let stats = compute_dataset_statistics(&triples);

        assert_eq!(stats.num_triples, 3);
        assert_eq!(stats.num_entities, 3);
        assert_eq!(stats.num_relations, 2);
        assert!(stats.avg_degree > 0.0);
    }

    #[test]
    fn test_create_simple_transe_model() {
        let model = create_simple_transe_model();
        assert_eq!(model.config().dimensions, 128);
        assert_eq!(model.config().learning_rate, 0.01);
        assert_eq!(model.config().max_epochs, 100);
    }

    #[test]
    fn test_parse_triple_from_string() -> Result<()> {
        let triple = parse_triple_from_string("alice knows bob")?;
        assert_eq!(triple.subject.iri.as_str(), "http://example.org/alice");
        assert_eq!(triple.predicate.iri.as_str(), "http://example.org/knows");
        assert_eq!(triple.object.iri.as_str(), "http://example.org/bob");

        let triple2 = parse_triple_from_string(
            "http://example.org/alice http://example.org/knows http://example.org/bob",
        )?;
        assert_eq!(triple2.subject.iri.as_str(), "http://example.org/alice");

        assert!(parse_triple_from_string("alice knows").is_err());

        Ok(())
    }

    #[test]
    fn test_add_triples_from_strings() -> Result<()> {
        let mut model = create_simple_transe_model();
        let triple_strings = &[
            "alice knows bob",
            "bob likes charlie",
            "charlie follows alice",
        ];

        let added_count = add_triples_from_strings(&mut model, triple_strings)?;
        assert_eq!(added_count, 3);

        Ok(())
    }

    #[test]
    fn test_cosine_similarity() -> Result<()> {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&a, &b)?;
        assert!((similarity - 1.0).abs() < 1e-10);

        let c = vec![0.0, 1.0, 0.0];
        let similarity2 = cosine_similarity(&a, &c)?;
        assert!((similarity2 - 0.0).abs() < 1e-10);

        let d = vec![1.0, 0.0];
        assert!(cosine_similarity(&a, &d).is_err());

        Ok(())
    }

    #[test]
    fn test_generate_sample_kg_data() {
        let triples = generate_sample_kg_data(5, 3);
        assert!(!triples.is_empty());

        for (subject, relation, object) in &triples {
            assert!(subject.starts_with("http://example.org/entity_"));
            assert!(relation.starts_with("http://example.org/relation_"));
            assert!(object.starts_with("http://example.org/entity_"));
            assert_ne!(subject, object);
        }
    }

    #[test]
    fn test_quick_performance_test() {
        let duration = quick_performance_test("test_operation", 100, || {
            let _sum: i32 = (1..10).sum();
        });

        let _nanos = duration.as_nanos();
    }
}
