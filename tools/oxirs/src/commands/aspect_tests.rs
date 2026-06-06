//! Tests for SAMM Aspect Model processing

#[cfg(test)]
mod tests {
    use crate::commands::aspect_types::{map_xsd_to_json_schema, map_xsd_to_rust, to_snake_case};

    #[test]
    fn test_map_xsd_to_rust_string() {
        assert_eq!(map_xsd_to_rust("xsd:string"), "String");
        assert_eq!(
            map_xsd_to_rust("http://www.w3.org/2001/XMLSchema#string"),
            "String"
        );
    }

    #[test]
    fn test_map_xsd_to_rust_numeric() {
        assert_eq!(map_xsd_to_rust("xsd:int"), "i32");
        assert_eq!(map_xsd_to_rust("xsd:integer"), "i32");
        assert_eq!(map_xsd_to_rust("xsd:long"), "i64");
        assert_eq!(map_xsd_to_rust("xsd:float"), "f32");
        assert_eq!(map_xsd_to_rust("xsd:double"), "f64");
    }

    #[test]
    fn test_map_xsd_to_rust_boolean() {
        assert_eq!(map_xsd_to_rust("xsd:boolean"), "bool");
    }

    #[test]
    fn test_map_xsd_to_rust_date() {
        assert_eq!(map_xsd_to_rust("xsd:date"), "chrono::NaiveDate");
        assert_eq!(
            map_xsd_to_rust("xsd:dateTime"),
            "chrono::DateTime<chrono::Utc>"
        );
    }

    #[test]
    fn test_map_xsd_to_rust_unknown() {
        assert_eq!(map_xsd_to_rust("unknown:type"), "String");
    }

    #[test]
    fn test_map_xsd_to_json_schema_string() {
        let (t, f) = map_xsd_to_json_schema("xsd:string");
        assert_eq!(t, "string");
        assert!(f.is_none());
    }

    #[test]
    fn test_map_xsd_to_json_schema_integer() {
        let (t, f) = map_xsd_to_json_schema("xsd:int");
        assert_eq!(t, "integer");
        assert_eq!(f, Some("int32".to_string()));
    }

    #[test]
    fn test_map_xsd_to_json_schema_date_time() {
        let (t, f) = map_xsd_to_json_schema("xsd:dateTime");
        assert_eq!(t, "string");
        assert_eq!(f, Some("date-time".to_string()));
    }

    #[test]
    fn test_to_snake_case_pascal() {
        assert_eq!(to_snake_case("MyProperty"), "my_property");
        assert_eq!(to_snake_case("SomeAspectName"), "some_aspect_name");
    }

    #[test]
    fn test_to_snake_case_already_lower() {
        assert_eq!(to_snake_case("lowercase"), "lowercase");
    }

    #[test]
    fn test_to_snake_case_single() {
        assert_eq!(to_snake_case("A"), "a");
    }
}
