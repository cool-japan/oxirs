//! Comprehensive tests for the CSVW parser and converter.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::csvw::{
        converter::{CsvwConverter, CsvwConverterConfig},
        reader::{parse_csv, CsvReader},
        schema::{ColumnDef, CsvwMetadata, TableSchema},
        CsvwError,
    };
    use std::io::Cursor;

    // ────────────────────────────────────────────────────────────────────────
    // 1. Schema parsing
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_csvw_metadata_from_json() {
        let json = r#"
        {
            "@id": "http://example.org/data.csv",
            "tableSchema": {
                "columns": [
                    {"name": "id",    "datatype": "http://www.w3.org/2001/XMLSchema#integer"},
                    {"name": "name",  "datatype": "http://www.w3.org/2001/XMLSchema#string"},
                    {"name": "score", "datatype": "http://www.w3.org/2001/XMLSchema#decimal"}
                ],
                "primaryKey": ["id"]
            },
            "aboutUrl": "http://example.org/row/{id}"
        }
        "#;

        let meta = CsvwMetadata::from_json(json).expect("parse should succeed");
        assert_eq!(meta.id.as_deref(), Some("http://example.org/data.csv"));
        assert_eq!(meta.table_schema.columns.len(), 3);
        assert_eq!(meta.table_schema.columns[0].name, "id");
        assert_eq!(meta.table_schema.columns[1].name, "name");
        assert_eq!(meta.table_schema.columns[2].name, "score");
        assert_eq!(
            meta.about_url.as_deref(),
            Some("http://example.org/row/{id}")
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 2. Column lookup
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_metadata_column_lookup() {
        let meta = make_simple_metadata();
        let col = meta.column("name").expect("column 'name' should exist");
        assert_eq!(col.name, "name");
        assert_eq!(
            col.datatype.as_deref(),
            Some("http://www.w3.org/2001/XMLSchema#string")
        );
    }

    #[test]
    fn test_metadata_missing_column() {
        let meta = make_simple_metadata();
        assert!(meta.column("nonexistent").is_none());
    }

    // ────────────────────────────────────────────────────────────────────────
    // 3. CSV reader — basic
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_csv_reader_simple() {
        let input = "a,b,c\n1,2,3\n4,5,6\n";
        let cursor = Cursor::new(input.as_bytes());
        let mut reader = CsvReader::new(cursor);

        let rec1 = reader.read_record().unwrap().expect("first record");
        assert_eq!(rec1.fields, vec!["a", "b", "c"]);

        let rec2 = reader.read_record().unwrap().expect("second record");
        assert_eq!(rec2.fields, vec!["1", "2", "3"]);

        let rec3 = reader.read_record().unwrap().expect("third record");
        assert_eq!(rec3.fields, vec!["4", "5", "6"]);

        assert!(reader.read_record().unwrap().is_none(), "should be EOF");
    }

    // ────────────────────────────────────────────────────────────────────────
    // 4. Quoted fields with embedded comma
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_csv_reader_quoted_fields() {
        let input = "name,address\n\"Alice\",\"123 Main St, Springfield\"\n";
        let cursor = Cursor::new(input.as_bytes());
        let mut reader = CsvReader::new(cursor);

        let _header = reader.read_record().unwrap().unwrap();
        let data = reader.read_record().unwrap().expect("data row");
        assert_eq!(data.fields[0], "Alice");
        assert_eq!(data.fields[1], "123 Main St, Springfield");
    }

    // ────────────────────────────────────────────────────────────────────────
    // 5. Empty fields
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_csv_reader_empty_fields() {
        let input = "a,,c\n";
        let cursor = Cursor::new(input.as_bytes());
        let mut reader = CsvReader::new(cursor);

        let rec = reader.read_record().unwrap().expect("record");
        assert_eq!(rec.fields.len(), 3, "must have 3 fields");
        assert_eq!(rec.fields[0], "a");
        assert_eq!(rec.fields[1], "");
        assert_eq!(rec.fields[2], "c");
    }

    // ────────────────────────────────────────────────────────────────────────
    // 6. Quoted field containing a newline
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_csv_reader_newline_in_field() {
        let input = "id,notes\n1,\"line one\nline two\"\n";
        let cursor = Cursor::new(input.as_bytes());
        let mut reader = CsvReader::new(cursor);

        let _header = reader.read_record().unwrap().unwrap();
        let rec = reader.read_record().unwrap().expect("data row");
        assert_eq!(rec.fields[0], "1");
        assert_eq!(rec.fields[1], "line one\nline two");
    }

    // ────────────────────────────────────────────────────────────────────────
    // 7. parse_csv convenience function
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_csv_convenience() {
        let input = "x,y\n10,20\n30,40\n";
        let (headers, records) = parse_csv(input).expect("parse");
        assert_eq!(headers, vec!["x", "y"]);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].fields, vec!["10", "20"]);
        assert_eq!(records[1].fields, vec!["30", "40"]);
    }

    // ────────────────────────────────────────────────────────────────────────
    // 8. Converter — basic two-column, two-row scenario
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_converter_basic() {
        let csv = "id,name\n1,Alice\n2,Bob\n";
        let (headers, records) = parse_csv(csv).unwrap();

        let metadata = make_simple_metadata();
        let config = CsvwConverterConfig::default();
        let converter = CsvwConverter::new(config);
        let stmts = converter.convert(&headers, &records, &metadata).unwrap();

        // 2 rows × 2 columns = 4 statements.
        assert_eq!(stmts.len(), 4);

        // First row subjects should both reference row 1.
        assert!(stmts[0].subject.contains('1') || stmts[0].subject.contains("row"));
        assert!(!stmts[0].object.is_empty());
    }

    // ────────────────────────────────────────────────────────────────────────
    // 9. Suppress output
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_converter_suppress_output() {
        let csv = "id,secret\n1,hidden\n";
        let (headers, records) = parse_csv(csv).unwrap();

        let metadata = CsvwMetadata {
            table_schema: TableSchema {
                columns: vec![
                    ColumnDef {
                        name: "id".into(),
                        suppress_output: false,
                        ..Default::default()
                    },
                    ColumnDef {
                        name: "secret".into(),
                        suppress_output: true, // suppressed
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
            ..Default::default()
        };

        let stmts = CsvwConverter::new(Default::default())
            .convert(&headers, &records, &metadata)
            .unwrap();

        // Only 1 statement — "secret" is suppressed.
        assert_eq!(stmts.len(), 1);
        // The remaining statement is for "id".
        assert!(stmts[0].predicate.contains("id"));
    }

    // ────────────────────────────────────────────────────────────────────────
    // 10. Datatype annotation
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_converter_datatype() {
        let csv = "count\n42\n";
        let (headers, records) = parse_csv(csv).unwrap();

        let metadata = CsvwMetadata {
            table_schema: TableSchema {
                columns: vec![ColumnDef {
                    name: "count".into(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#integer".into()),
                    ..Default::default()
                }],
                ..Default::default()
            },
            ..Default::default()
        };

        let stmts = CsvwConverter::new(Default::default())
            .convert(&headers, &records, &metadata)
            .unwrap();

        assert_eq!(stmts.len(), 1);
        assert!(
            stmts[0].object.contains("XMLSchema#integer"),
            "object should carry xsd:integer datatype: {}",
            stmts[0].object
        );
        assert!(
            stmts[0].object.contains("42"),
            "object should contain the value: {}",
            stmts[0].object
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 11. Custom propertyUrl
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_converter_property_url() {
        let csv = "label\nhello\n";
        let (headers, records) = parse_csv(csv).unwrap();

        let custom_pred = "http://schema.org/name";
        let metadata = CsvwMetadata {
            table_schema: TableSchema {
                columns: vec![ColumnDef {
                    name: "label".into(),
                    property_url: Some(custom_pred.into()),
                    ..Default::default()
                }],
                ..Default::default()
            },
            ..Default::default()
        };

        let stmts = CsvwConverter::new(Default::default())
            .convert(&headers, &records, &metadata)
            .unwrap();

        assert_eq!(stmts.len(), 1);
        assert!(
            stmts[0].predicate.contains("schema.org/name"),
            "predicate should use propertyUrl: {}",
            stmts[0].predicate
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 12. aboutUrl template
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_converter_about_url() {
        let csv = "id,val\n99,hello\n";
        let (headers, records) = parse_csv(csv).unwrap();

        let metadata = CsvwMetadata {
            about_url: Some("http://example.org/item/{id}".into()),
            table_schema: TableSchema {
                columns: vec![
                    ColumnDef {
                        name: "id".into(),
                        ..Default::default()
                    },
                    ColumnDef {
                        name: "val".into(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
            ..Default::default()
        };

        let stmts = CsvwConverter::new(Default::default())
            .convert(&headers, &records, &metadata)
            .unwrap();

        // Both statements share the same expanded subject.
        assert!(
            stmts[0].subject.contains("item/99"),
            "subject should use aboutUrl template: {}",
            stmts[0].subject
        );
        assert_eq!(stmts[0].subject, stmts[1].subject);
    }

    // ────────────────────────────────────────────────────────────────────────
    // 13. Empty CSV → zero statements
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_converter_empty_csv() {
        let csv = "id,name\n"; // header only, no data rows
        let (headers, records) = parse_csv(csv).unwrap();
        assert!(records.is_empty());

        let metadata = make_simple_metadata();
        let stmts = CsvwConverter::new(Default::default())
            .convert(&headers, &records, &metadata)
            .unwrap();

        assert!(stmts.is_empty(), "no data rows → no statements");
    }

    // ────────────────────────────────────────────────────────────────────────
    // 14. Primary key parsed from JSON
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_schema_primary_key() {
        let json = r#"
        {
            "tableSchema": {
                "columns": [
                    {"name": "id"},
                    {"name": "name"}
                ],
                "primaryKey": ["id"]
            }
        }
        "#;
        let meta = CsvwMetadata::from_json(json).unwrap();
        assert_eq!(meta.table_schema.primary_key, vec!["id"]);
    }

    // ────────────────────────────────────────────────────────────────────────
    // 15. Error: malformed JSON
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_metadata_malformed_json() {
        let result = CsvwMetadata::from_json("{not valid json}");
        assert!(
            matches!(result, Err(CsvwError::JsonError(_))),
            "should return JsonError for bad JSON"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 16. Full roundtrip: write CSV → parse metadata JSON → convert → check count
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_csv_roundtrip_with_schema() {
        // Build a three-column CSV in memory.
        let csv =
            "product,price,inStock\nWidgetA,9.99,true\nWidgetB,19.99,false\nWidgetC,4.50,true\n";

        let metadata_json = r#"
        {
            "tableSchema": {
                "columns": [
                    {
                        "name": "product",
                        "propertyUrl": "http://schema.org/name",
                        "datatype": "http://www.w3.org/2001/XMLSchema#string"
                    },
                    {
                        "name": "price",
                        "propertyUrl": "http://schema.org/price",
                        "datatype": "http://www.w3.org/2001/XMLSchema#decimal"
                    },
                    {
                        "name": "inStock",
                        "propertyUrl": "http://schema.org/availability",
                        "datatype": "http://www.w3.org/2001/XMLSchema#boolean"
                    }
                ],
                "primaryKey": ["product"]
            },
            "aboutUrl": "http://example.org/product/{product}"
        }
        "#;

        let meta = CsvwMetadata::from_json(metadata_json).expect("metadata parse");
        let (headers, records) = parse_csv(csv).expect("csv parse");

        assert_eq!(records.len(), 3, "three data rows");

        let config = CsvwConverterConfig {
            base_iri: "http://example.org/product/".into(),
            start_row: 1,
        };
        let converter = CsvwConverter::new(config);
        let stmts = converter.convert(&headers, &records, &meta).unwrap();

        // 3 rows × 3 columns = 9 statements.
        assert_eq!(stmts.len(), 9, "9 RDF statements expected");

        // Spot-check: first statement should be about WidgetA.
        assert!(
            stmts[0].subject.contains("WidgetA"),
            "first subject should be WidgetA: {}",
            stmts[0].subject
        );
        assert!(
            stmts[0].predicate.contains("schema.org/name"),
            "first predicate should be schema.org/name: {}",
            stmts[0].predicate
        );
        assert!(
            stmts[0].object.contains("WidgetA"),
            "first object value should be WidgetA: {}",
            stmts[0].object
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────────

    fn make_simple_metadata() -> CsvwMetadata {
        CsvwMetadata {
            id: Some("http://example.org/data.csv".into()),
            table_schema: TableSchema {
                columns: vec![
                    ColumnDef {
                        name: "id".into(),
                        datatype: Some("http://www.w3.org/2001/XMLSchema#integer".into()),
                        ..Default::default()
                    },
                    ColumnDef {
                        name: "name".into(),
                        datatype: Some("http://www.w3.org/2001/XMLSchema#string".into()),
                        ..Default::default()
                    },
                ],
                primary_key: vec!["id".into()],
            },
            about_url: None,
        }
    }
}
