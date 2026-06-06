//! # HDT Tests
//!
//! Tests for the HDT reader, dictionary, triples section, format utilities,
//! and CRC checksums.  All tests use hand-crafted in-memory byte payloads via
//! the `from_bytes` constructor and internal helpers so no real HDT files are
//! required on disk.

#[cfg(test)]
mod hdt_tests {
    use std::io::Cursor;

    use crate::hdt::{
        dictionary::{parse_plain_dictionary, DictionarySection, HdtDictionary},
        format::{compute_crc16, compute_crc32, read_vbyte, write_vbyte},
        triples::{bitmap_access, bitmap_rank, HdtTriplesSection},
        HdtError, HdtHeader, HdtReader, HdtTriple,
    };

    // -----------------------------------------------------------------------
    // Helper: build a minimal valid HDT byte payload
    // -----------------------------------------------------------------------

    /// Magic bytes + header block + dictionary + triples section.
    ///
    /// Layout:
    /// ```text
    /// [5]  magic "$HDT\x01"
    /// [8]  header size (LE u64)
    /// [?]  header property block (key=value\n lines)
    /// [4]  shared section length (LE u32)
    /// [?]  shared null-separated strings
    /// [4]  subjects section length (LE u32)
    /// [?]  subjects null-separated strings
    /// [4]  predicates section length (LE u32)
    /// [?]  predicates null-separated strings
    /// [4]  objects section length (LE u32)
    /// [?]  objects null-separated strings
    /// [?]  triples section (see HdtTriplesSection::parse format)
    /// ```
    fn build_minimal_hdt(
        hdr_props: &str,
        shared: &[&str],
        subjects: &[&str],
        predicates: &[&str],
        objects: &[&str],
        triples_bytes: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(b"$HDT\x01");

        // Header block
        let hdr_bytes = hdr_props.as_bytes();
        buf.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(hdr_bytes);

        // Dictionary sections helper
        let mut push_dict_section = |strings: &[&str]| {
            let mut sec = Vec::new();
            for s in strings {
                sec.extend_from_slice(s.as_bytes());
                sec.push(0u8); // null terminator
            }
            buf.extend_from_slice(&(sec.len() as u32).to_le_bytes());
            buf.extend_from_slice(&sec);
        };

        push_dict_section(shared);
        push_dict_section(subjects);
        push_dict_section(predicates);
        push_dict_section(objects);

        buf.extend_from_slice(triples_bytes);
        buf
    }

    /// Build a minimal triples section for N triples with given IDs.
    ///
    /// This produces a trivially flat adjacency list where every (s,p) pair
    /// has exactly one object, so bitmap_y and bitmap_z are all 1 (boundary).
    fn build_flat_triples(triples: &[(u32, u32, u32)]) -> Vec<u8> {
        let count_sy = triples.len() as u32;
        let count_z = triples.len() as u32;

        let mut buf = Vec::new();
        buf.extend_from_slice(&count_sy.to_le_bytes());
        buf.extend_from_slice(&count_z.to_le_bytes());

        // array_y: predicate IDs
        for &(_, p, _) in triples {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        // array_z: object IDs
        for &(_, _, o) in triples {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        // bitmap_y_raw: all 1 (each sy slot is the last for its subject)
        for _ in triples {
            buf.extend_from_slice(&1u32.to_le_bytes());
        }
        // bitmap_z_raw: all 1 (each z slot is the last for its (s,p) pair)
        for _ in triples {
            buf.extend_from_slice(&1u32.to_le_bytes());
        }
        buf
    }

    // -----------------------------------------------------------------------
    // 1. test_vbyte_encode_single_byte
    // -----------------------------------------------------------------------

    #[test]
    fn test_vbyte_encode_single_byte() {
        // Values 0..=127 fit in one byte; high bit is 0 (no continuation).
        assert_eq!(write_vbyte(0), vec![0x00]);
        assert_eq!(write_vbyte(1), vec![0x01]);
        assert_eq!(write_vbyte(42), vec![0x2A]);
        assert_eq!(write_vbyte(127), vec![0x7F]);
    }

    // -----------------------------------------------------------------------
    // 2. test_vbyte_encode_multi_byte
    // -----------------------------------------------------------------------

    #[test]
    fn test_vbyte_encode_multi_byte() {
        // 128 requires 2 bytes: low 7 bits = 0, continue; upper bits = 1.
        assert_eq!(write_vbyte(128), vec![0x80, 0x01]);
        // 300 = 0b1_0010_1100
        assert_eq!(write_vbyte(300), vec![0b1010_1100, 0b0000_0010]);
        // 16384 = 2^14 requires 3 bytes
        let enc = write_vbyte(16384);
        assert_eq!(enc.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 3. test_vbyte_roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_vbyte_roundtrip() {
        let values = [0u64, 1, 63, 64, 127, 128, 255, 1000, 16383, 16384, 2_097_151, 2_097_152];
        for v in values {
            let encoded = write_vbyte(v);
            let mut cur = Cursor::new(&encoded);
            let decoded = read_vbyte(&mut cur).expect("vbyte roundtrip");
            assert_eq!(decoded, v, "roundtrip failed for {}", v);
        }
    }

    // -----------------------------------------------------------------------
    // 4. test_crc16_known_value
    // -----------------------------------------------------------------------

    #[test]
    fn test_crc16_known_value() {
        // CRC-16/CCITT-FALSE: "123456789" → 0x29B1
        assert_eq!(compute_crc16(b"123456789"), 0x29B1);
    }

    // -----------------------------------------------------------------------
    // 5. test_crc32_known_value
    // -----------------------------------------------------------------------

    #[test]
    fn test_crc32_known_value() {
        // CRC-32/ISO-HDLC: "123456789" → 0xCBF43926
        assert_eq!(compute_crc32(b"123456789"), 0xCBF4_3926);
    }

    // -----------------------------------------------------------------------
    // 6. test_dictionary_section_plain
    // -----------------------------------------------------------------------

    #[test]
    fn test_dictionary_section_plain() {
        let data = b"apple\0banana\0cherry\0";
        let section = DictionarySection::from_plain(data).expect("parse plain");
        assert_eq!(section.terms, vec!["apple", "banana", "cherry"]);
    }

    // -----------------------------------------------------------------------
    // 7. test_dictionary_section_front_coded
    // -----------------------------------------------------------------------

    #[test]
    fn test_dictionary_section_front_coded() {
        // k=2: entry 0 is full, entry 1 is delta, entry 2 is full, …
        // Terms: "abc", "abd", "xyz"
        // anchor: "abc\0"
        // delta:  [vbyte(2)] "d\0"   (prefix len=2, suffix="d")
        // anchor: "xyz\0"
        let mut data = Vec::new();
        data.extend_from_slice(b"abc\0");
        data.push(0x02); // vbyte(2)  = shared prefix len
        data.extend_from_slice(b"d\0");
        data.extend_from_slice(b"xyz\0");

        let section = DictionarySection::from_front_coded(&data, 2).expect("parse front-coded");
        assert_eq!(section.terms, vec!["abc", "abd", "xyz"]);
    }

    // -----------------------------------------------------------------------
    // 8. test_dictionary_id_to_term
    // -----------------------------------------------------------------------

    #[test]
    fn test_dictionary_id_to_term() {
        let data = b"alpha\0beta\0gamma\0";
        let section = DictionarySection::from_plain(data).expect("parse");
        // 1-based lookup
        assert_eq!(section.id_to_term(1), Some("alpha"));
        assert_eq!(section.id_to_term(2), Some("beta"));
        assert_eq!(section.id_to_term(3), Some("gamma"));
        assert_eq!(section.id_to_term(0), None);
        assert_eq!(section.id_to_term(4), None);
    }

    // -----------------------------------------------------------------------
    // 9. test_dictionary_term_to_id
    // -----------------------------------------------------------------------

    #[test]
    fn test_dictionary_term_to_id() {
        // Binary search works on sorted data
        let data = b"alpha\0beta\0gamma\0";
        let section = DictionarySection::from_plain(data).expect("parse");
        assert_eq!(section.term_to_id("alpha"), Some(1));
        assert_eq!(section.term_to_id("beta"), Some(2));
        assert_eq!(section.term_to_id("gamma"), Some(3));
        assert_eq!(section.term_to_id("delta"), None);
    }

    // -----------------------------------------------------------------------
    // 10. test_bitmap_access
    // -----------------------------------------------------------------------

    #[test]
    fn test_bitmap_access() {
        // Build a bitmap with bits 0, 63, 64 set.
        let mut bm = vec![0u64; 2];
        bm[0] = 1u64 | (1u64 << 63); // bits 0 and 63 in word 0
        bm[1] = 1u64;                 // bit 64 (=bit 0 of word 1)

        assert!(bitmap_access(&bm, 0));
        assert!(!bitmap_access(&bm, 1));
        assert!(bitmap_access(&bm, 63));
        assert!(bitmap_access(&bm, 64));
        assert!(!bitmap_access(&bm, 65));
        // Out-of-range → false
        assert!(!bitmap_access(&bm, 128));
    }

    // -----------------------------------------------------------------------
    // 11. test_bitmap_rank
    // -----------------------------------------------------------------------

    #[test]
    fn test_bitmap_rank() {
        // bits 0, 2, 4 are set → rank at pos 5 = 3
        let bm = vec![0b0001_0101u64];
        assert_eq!(bitmap_rank(&bm, 0), 0);
        assert_eq!(bitmap_rank(&bm, 1), 1); // bit 0 counts
        assert_eq!(bitmap_rank(&bm, 3), 2); // bits 0,2
        assert_eq!(bitmap_rank(&bm, 5), 3); // bits 0,2,4
        assert_eq!(bitmap_rank(&bm, 64), 3);
    }

    // -----------------------------------------------------------------------
    // 12. test_read_write_vbyte_large
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_write_vbyte_large() {
        // Values > 2^21 require at least 4 bytes
        let large_values = [2_097_152u64, 10_000_000, u32::MAX as u64, u64::MAX / 2];
        for v in large_values {
            let encoded = write_vbyte(v);
            assert!(encoded.len() >= 4, "expected >= 4 bytes for {}", v);
            let mut cur = Cursor::new(&encoded);
            let decoded = read_vbyte(&mut cur).expect("decode large vbyte");
            assert_eq!(decoded, v);
        }
    }

    // -----------------------------------------------------------------------
    // 13. test_hdt_reader_invalid_magic
    // -----------------------------------------------------------------------

    #[test]
    fn test_hdt_reader_invalid_magic() {
        let bad = b"not-hdt-data at all".to_vec();
        let err = HdtReader::from_bytes(bad).expect_err("should fail with invalid magic");
        assert!(
            matches!(err, HdtError::InvalidMagic { .. }),
            "expected InvalidMagic, got {:?}",
            err
        );
    }

    // -----------------------------------------------------------------------
    // 14. test_dictionary_shared_so_lookup
    // -----------------------------------------------------------------------

    #[test]
    fn test_dictionary_shared_so_lookup() {
        let mut d = HdtDictionary::new();
        d.shared.push("<http://example.org/Alice>".to_owned());
        d.shared.push("<http://example.org/Bob>".to_owned());
        d.subjects.push("<http://example.org/Charlie>".to_owned());

        // Shared terms are accessible as both subject and object
        assert_eq!(d.lookup_subject(1), Some("<http://example.org/Alice>"));
        assert_eq!(d.lookup_object(1), Some("<http://example.org/Alice>"));
        assert_eq!(d.lookup_subject(2), Some("<http://example.org/Bob>"));
        assert_eq!(d.lookup_object(2), Some("<http://example.org/Bob>"));

        // Subject-only term is only in subject position
        assert_eq!(d.lookup_subject(3), Some("<http://example.org/Charlie>"));
        assert_eq!(d.lookup_object(3), None);
    }

    // -----------------------------------------------------------------------
    // 15. test_subject_count_from_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_subject_count_from_stats() {
        let hdr_props =
            "triples=5\nsubjects=3\npredicates=2\nobjects=4\nshared=1\nformat=hdt/plain\n";
        let triples_bytes = build_flat_triples(&[]);
        let data = build_minimal_hdt(hdr_props, &[], &[], &[], &[], &triples_bytes);
        let reader = HdtReader::from_bytes(data).expect("parse");
        let stats = reader.stats();
        assert_eq!(stats.triple_count, 5);
        assert_eq!(stats.distinct_subjects, 3);
        assert_eq!(stats.distinct_predicates, 2);
        assert_eq!(stats.distinct_objects, 4);
        assert_eq!(stats.shared_so_count, 1);
    }

    // -----------------------------------------------------------------------
    // 16. test_front_coding_k4
    // -----------------------------------------------------------------------

    #[test]
    fn test_front_coding_k4() {
        // k=4: entries 0 and 4 are anchors; entries 1,2,3 are deltas.
        // Terms: "abcde", "abcdf", "abcdg", "abcdh", "xyz"
        let mut data = Vec::new();
        data.extend_from_slice(b"abcde\0"); // anchor 0
        data.push(4);                        // vbyte(4) = 4 shared bytes "abcd"
        data.extend_from_slice(b"f\0");      // suffix
        data.push(4);
        data.extend_from_slice(b"g\0");
        data.push(4);
        data.extend_from_slice(b"h\0");
        data.extend_from_slice(b"xyz\0");    // anchor 4

        let section = DictionarySection::from_front_coded(&data, 4).expect("k=4 decode");
        assert_eq!(section.terms.len(), 5);
        assert_eq!(section.terms[0], "abcde");
        assert_eq!(section.terms[1], "abcdf");
        assert_eq!(section.terms[2], "abcdg");
        assert_eq!(section.terms[3], "abcdh");
        assert_eq!(section.terms[4], "xyz");
    }

    // -----------------------------------------------------------------------
    // Additional tests (retained from original test suite)
    // -----------------------------------------------------------------------

    #[test]
    fn test_hdt_magic_bytes() {
        let bad = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let err = HdtReader::from_bytes(bad).expect_err("bad magic");
        assert!(matches!(err, HdtError::InvalidMagic { .. }));
    }

    #[test]
    fn test_dictionary_lookup_shared() {
        let mut d = HdtDictionary::new();
        d.shared.push("<http://example.org/Alice>".to_owned());
        d.shared.push("<http://example.org/Bob>".to_owned());

        assert_eq!(d.lookup_subject(1), Some("<http://example.org/Alice>"));
        assert_eq!(d.lookup_subject(2), Some("<http://example.org/Bob>"));
        assert_eq!(d.lookup_object(1), Some("<http://example.org/Alice>"));
        assert_eq!(d.lookup_object(2), Some("<http://example.org/Bob>"));
    }

    #[test]
    fn test_dictionary_lookup_subject_only() {
        let mut d = HdtDictionary::new();
        d.shared.push("<http://shared>".to_owned());
        d.subjects.push("<http://subject-only>".to_owned());

        assert_eq!(d.lookup_subject(2), Some("<http://subject-only>"));
    }

    #[test]
    fn test_dictionary_lookup_predicate() {
        let mut d = HdtDictionary::new();
        d.predicates
            .push("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>".to_owned());
        d.predicates.push("<http://schema.org/name>".to_owned());

        assert_eq!(
            d.lookup_predicate(1),
            Some("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
        );
        assert_eq!(d.lookup_predicate(2), Some("<http://schema.org/name>"));
    }

    #[test]
    fn test_dictionary_lookup_object_only() {
        let mut d = HdtDictionary::new();
        d.shared.push("<http://shared>".to_owned());
        d.objects.push("\"Alice\"".to_owned());

        assert_eq!(d.lookup_object(2), Some("\"Alice\""));
    }

    #[test]
    fn test_dictionary_invalid_id_zero() {
        let mut d = HdtDictionary::new();
        d.shared.push("<http://x>".to_owned());
        assert_eq!(d.lookup_subject(0), None);
        assert_eq!(d.lookup_predicate(0), None);
        assert_eq!(d.lookup_object(0), None);
    }

    #[test]
    fn test_dictionary_out_of_range() {
        let mut d = HdtDictionary::new();
        d.shared.push("<http://x>".to_owned());
        assert_eq!(d.lookup_subject(999), None);
        assert_eq!(d.lookup_predicate(999), None);
        assert_eq!(d.lookup_object(999), None);
    }

    #[test]
    fn test_dictionary_shared_count() {
        let mut d = HdtDictionary::new();
        d.shared.push("s1".to_owned());
        d.shared.push("s2".to_owned());
        d.subjects.push("so1".to_owned());

        assert_eq!(d.subject_count(), 3);
        assert_eq!(d.object_count(), 2);
    }

    #[test]
    fn test_parse_plain_dictionary_single() {
        let data = b"hello\0";
        let result = parse_plain_dictionary(data).expect("parse");
        assert_eq!(result, vec!["hello".to_owned()]);
    }

    #[test]
    fn test_parse_plain_dictionary_multiple() {
        let data = b"alpha\0beta\0gamma\0";
        let result = parse_plain_dictionary(data).expect("parse");
        assert_eq!(result, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_parse_plain_dictionary_empty() {
        let result = parse_plain_dictionary(b"").expect("parse empty");
        assert!(result.is_empty());
    }

    #[test]
    fn test_triples_iter_basic() {
        let raw = build_flat_triples(&[(1, 1, 1), (2, 1, 2)]);
        let section = HdtTriplesSection::parse(&raw).expect("parse");
        let ids: Vec<(u32, u32, u32)> = section.iter_ids().collect();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0].1, 1);
        assert_eq!(ids[0].2, 1);
        assert_eq!(ids[1].2, 2);
    }

    #[test]
    fn test_triples_section_round_trip() {
        let input = vec![(1u32, 1u32, 1u32), (1, 2, 3), (2, 1, 2)];
        let raw = build_flat_triples(&input);
        let section = HdtTriplesSection::parse(&raw).expect("parse");
        let ids: Vec<(u32, u32, u32)> = section.iter_ids().collect();
        assert_eq!(ids.len(), input.len());
    }

    #[test]
    fn test_header_triple_count() {
        let hdr_props =
            "triples=42\nsubjects=10\npredicates=5\nobjects=30\nshared=3\nformat=hdt/plain\n";
        let triples_bytes = build_flat_triples(&[]);
        let data = build_minimal_hdt(hdr_props, &[], &[], &[], &[], &triples_bytes);
        let reader = HdtReader::from_bytes(data).expect("parse");
        assert_eq!(reader.header().triples_count, 42);
        assert_eq!(reader.header().subjects_count, 10);
        assert_eq!(reader.header().predicates_count, 5);
        assert_eq!(reader.header().objects_count, 30);
        assert_eq!(reader.header().shared_count, 3);
        assert_eq!(reader.header().format, "hdt/plain");
    }

    #[test]
    fn test_hdt_reader_from_bytes_empty() {
        let err = HdtReader::from_bytes(vec![]).expect_err("empty should fail");
        assert!(matches!(err, HdtError::InvalidMagic { .. }));
    }

    fn build_two_triple_hdt() -> Vec<u8> {
        let hdr =
            "triples=2\nsubjects=2\npredicates=1\nobjects=2\nshared=0\nformat=hdt/plain\n";
        let shared: &[&str] = &[];
        let subjects: &[&str] = &["<http://s1>", "<http://s2>"];
        let predicates: &[&str] = &["<http://p>"];
        let objects: &[&str] = &["<http://o1>", "<http://o2>"];
        let triples_bytes = build_flat_triples(&[(1, 1, 1), (2, 1, 2)]);
        build_minimal_hdt(hdr, shared, subjects, predicates, objects, &triples_bytes)
    }

    #[test]
    fn test_triple_lookup_subject() {
        let data = build_two_triple_hdt();
        let reader = HdtReader::from_bytes(data).expect("parse");
        assert_eq!(reader.lookup_subject(1).expect("lookup"), "<http://s1>");
        assert_eq!(reader.lookup_subject(2).expect("lookup"), "<http://s2>");
    }

    #[test]
    fn test_triple_lookup_predicate() {
        let data = build_two_triple_hdt();
        let reader = HdtReader::from_bytes(data).expect("parse");
        assert_eq!(reader.lookup_predicate(1).expect("lookup"), "<http://p>");
    }

    #[test]
    fn test_triple_lookup_object() {
        let data = build_two_triple_hdt();
        let reader = HdtReader::from_bytes(data).expect("parse");
        assert_eq!(reader.lookup_object(1).expect("lookup"), "<http://o1>");
        assert_eq!(reader.lookup_object(2).expect("lookup"), "<http://o2>");
    }

    #[test]
    fn test_triples_iterator_resolves_strings() {
        let data = build_two_triple_hdt();
        let reader = HdtReader::from_bytes(data).expect("parse");
        let triples: Result<Vec<HdtTriple>, _> = reader.triples().collect();
        let triples = triples.expect("resolve");
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0].predicate, "<http://p>");
        assert_eq!(triples[1].predicate, "<http://p>");
    }
}
