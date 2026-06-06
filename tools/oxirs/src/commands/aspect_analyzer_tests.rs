//! Tests for aspect_analyzer sub-modules
#[cfg(test)]
mod tests {
    use crate::commands::aspect_analyzer_types::{
        map_xsd_to_json_schema, map_xsd_to_rust, to_snake_case,
    };

    #[test]
    fn test_map_xsd_to_rust_string() {
        assert_eq!(map_xsd_to_rust("xsd:string"), "String");
        assert_eq!(
            map_xsd_to_rust("http://www.w3.org/2001/XMLSchema#string"),
            "String"
        );
    }

    #[test]
    fn test_map_xsd_to_rust_integer() {
        assert_eq!(map_xsd_to_rust("xsd:integer"), "i64");
        assert_eq!(map_xsd_to_rust("xsd:int"), "i32");
        assert_eq!(map_xsd_to_rust("xsd:long"), "i64");
        assert_eq!(map_xsd_to_rust("xsd:short"), "i16");
    }

    #[test]
    fn test_map_xsd_to_rust_float() {
        assert_eq!(map_xsd_to_rust("xsd:float"), "f32");
        assert_eq!(map_xsd_to_rust("xsd:double"), "f64");
        assert_eq!(map_xsd_to_rust("xsd:decimal"), "f64");
    }

    #[test]
    fn test_map_xsd_to_rust_bool() {
        assert_eq!(map_xsd_to_rust("xsd:boolean"), "bool");
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
        assert!(f.is_none());
    }

    #[test]
    fn test_map_xsd_to_json_schema_datetime() {
        let (t, f) = map_xsd_to_json_schema("xsd:dateTime");
        assert_eq!(t, "string");
        assert_eq!(f, Some("date-time".to_string()));
    }

    #[test]
    fn test_to_snake_case_simple() {
        assert_eq!(to_snake_case("MyProperty"), "my_property");
    }

    #[test]
    fn test_to_snake_case_already_snake() {
        assert_eq!(to_snake_case("my_property"), "my_property");
    }

    #[test]
    fn test_to_snake_case_camel() {
        assert_eq!(to_snake_case("someProperty"), "some_property");
    }
}
