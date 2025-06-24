//! Tests for OxiRS GraphQL implementation

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::*,
        types::*,
        rdf_scalars::*,
        execution::*,
        resolvers::*,
    };
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_ast_value_creation() {
        // Test basic value creation
        let string_val = Value::StringValue("test".to_string());
        let int_val = Value::IntValue(42);
        let float_val = Value::FloatValue(3.14);
        let bool_val = Value::BooleanValue(true);
        let null_val = Value::NullValue;

        assert!(matches!(string_val, Value::StringValue(_)));
        assert!(matches!(int_val, Value::IntValue(42)));
        assert!(matches!(float_val, Value::FloatValue(_)));
        assert!(matches!(bool_val, Value::BooleanValue(true)));
        assert!(matches!(null_val, Value::NullValue));
    }

    #[test]
    fn test_list_and_object_values() {
        // Test list value
        let list_val = Value::ListValue(vec![
            Value::IntValue(1),
            Value::IntValue(2),
            Value::IntValue(3),
        ]);

        if let Value::ListValue(items) = list_val {
            assert_eq!(items.len(), 3);
        } else {
            panic!("Expected ListValue");
        }

        // Test object value
        let mut obj = HashMap::new();
        obj.insert("name".to_string(), Value::StringValue("test".to_string()));
        obj.insert("age".to_string(), Value::IntValue(25));

        let obj_val = Value::ObjectValue(obj);
        if let Value::ObjectValue(map) = obj_val {
            assert_eq!(map.len(), 2);
            assert!(map.contains_key("name"));
            assert!(map.contains_key("age"));
        } else {
            panic!("Expected ObjectValue");
        }
    }

    #[test]
    fn test_builtin_scalars() {
        // Test built-in scalar types
        let string_scalar = BuiltinScalars::string();
        let int_scalar = BuiltinScalars::int();
        let float_scalar = BuiltinScalars::float();
        let boolean_scalar = BuiltinScalars::boolean();
        let id_scalar = BuiltinScalars::id();

        assert_eq!(string_scalar.name, "String");
        assert_eq!(int_scalar.name, "Int");
        assert_eq!(float_scalar.name, "Float");
        assert_eq!(boolean_scalar.name, "Boolean");
        assert_eq!(id_scalar.name, "ID");
    }

    #[test]
    fn test_schema_creation() {
        let mut schema = Schema::new();
        
        // Test that built-in scalars are present
        assert!(schema.get_type("String").is_some());
        assert!(schema.get_type("Int").is_some());
        assert!(schema.get_type("Float").is_some());
        assert!(schema.get_type("Boolean").is_some());
        assert!(schema.get_type("ID").is_some());

        // Test adding custom type
        let custom_type = ObjectType::new("User".to_string())
            .with_description("A user object".to_string())
            .with_field(
                "id".to_string(),
                FieldType::new(
                    "id".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::id()),
                ),
            )
            .with_field(
                "name".to_string(),
                FieldType::new(
                    "name".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::string()),
                ),
            );

        schema.add_type(GraphQLType::Object(custom_type));
        assert!(schema.get_type("User").is_some());
    }

    #[test]
    fn test_rdf_scalars() {
        // Test IRI scalar
        let iri_scalar = RdfScalars::iri();
        assert_eq!(iri_scalar.name, "IRI");

        // Test Literal scalar
        let literal_scalar = RdfScalars::literal();
        assert_eq!(literal_scalar.name, "Literal");

        // Test DateTime scalar
        let datetime_scalar = RdfScalars::datetime();
        assert_eq!(datetime_scalar.name, "DateTime");

        // Test Duration scalar
        let duration_scalar = RdfScalars::duration();
        assert_eq!(duration_scalar.name, "Duration");

        // Test GeoLocation scalar
        let geolocation_scalar = RdfScalars::geolocation();
        assert_eq!(geolocation_scalar.name, "GeoLocation");

        // Test LangString scalar
        let langstring_scalar = RdfScalars::lang_string();
        assert_eq!(langstring_scalar.name, "LangString");
    }

    #[test]
    fn test_iri_validation() {
        // Test valid IRI
        let valid_iri = IRI::new("http://example.org/test".to_string());
        assert!(valid_iri.is_ok());

        // Test invalid IRI (empty)
        let invalid_iri = IRI::new("".to_string());
        assert!(invalid_iri.is_err());

        // Test invalid IRI (no scheme)
        let invalid_iri2 = IRI::new("example.org".to_string());
        assert!(invalid_iri2.is_err());
    }

    #[test]
    fn test_literal_creation() {
        // Test simple literal
        let literal = Literal::new("Hello World".to_string());
        assert_eq!(literal.value, "Hello World");
        assert!(!literal.is_language_tagged());
        assert!(!literal.is_typed());

        // Test language-tagged literal
        let lang_literal = Literal::new("Bonjour".to_string())
            .with_language("fr".to_string());
        assert!(lang_literal.is_language_tagged());
        assert!(!lang_literal.is_typed());
        assert_eq!(lang_literal.language.unwrap(), "fr");

        // Test typed literal
        let iri = IRI::new("http://www.w3.org/2001/XMLSchema#integer".to_string()).unwrap();
        let typed_literal = Literal::new("42".to_string())
            .with_datatype(iri);
        assert!(!typed_literal.is_language_tagged());
        assert!(typed_literal.is_typed());
    }

    #[test]
    fn test_geolocation() {
        // Test valid coordinates
        let geo = GeoLocation::new(40.7128, -74.0060);
        assert!(geo.is_ok());
        let geo = geo.unwrap();
        assert_eq!(geo.latitude, 40.7128);
        assert_eq!(geo.longitude, -74.0060);
        assert!(geo.altitude.is_none());

        // Test with altitude
        let geo_with_alt = GeoLocation::new(40.7128, -74.0060)
            .unwrap()
            .with_altitude(10.0);
        assert_eq!(geo_with_alt.altitude, Some(10.0));

        // Test invalid latitude
        let invalid_lat = GeoLocation::new(91.0, 0.0);
        assert!(invalid_lat.is_err());

        // Test invalid longitude
        let invalid_lng = GeoLocation::new(0.0, 181.0);
        assert!(invalid_lng.is_err());
    }

    #[test]
    fn test_duration() {
        let duration = Duration::new(60, 500_000_000);
        assert_eq!(duration.seconds, 60);
        assert_eq!(duration.nanoseconds, 500_000_000);
        assert_eq!(duration.total_seconds(), 60.5);

        let duration_from_seconds = Duration::from_seconds(120);
        assert_eq!(duration_from_seconds.seconds, 120);
        assert_eq!(duration_from_seconds.nanoseconds, 0);

        let duration_from_millis = Duration::from_millis(1500);
        assert_eq!(duration_from_millis.seconds, 1);
        assert_eq!(duration_from_millis.nanoseconds, 500_000_000);
    }

    #[test]
    fn test_execution_context() {
        let context = ExecutionContext::new()
            .with_operation_name("TestQuery".to_string());

        assert_eq!(context.operation_name, Some("TestQuery".to_string()));
        assert!(context.variables.is_empty());
        assert!(!context.request_id.is_empty());

        let mut variables = HashMap::new();
        variables.insert("userId".to_string(), Value::StringValue("123".to_string()));

        let context_with_vars = ExecutionContext::new()
            .with_variables(variables);

        assert_eq!(context_with_vars.variables.len(), 1);
        assert!(context_with_vars.variables.contains_key("userId"));
    }

    #[test]
    fn test_graphql_error() {
        let error = GraphQLError::new("Test error".to_string())
            .with_path(vec!["user".to_string(), "name".to_string()])
            .with_location(SourceLocation::new(1, 10));

        assert_eq!(error.message, "Test error");
        assert_eq!(error.path, vec!["user", "name"]);
        assert_eq!(error.locations.len(), 1);
        assert_eq!(error.locations[0].line, 1);
        assert_eq!(error.locations[0].column, 10);
    }

    #[test]
    fn test_execution_result() {
        let result = ExecutionResult::new()
            .with_data(serde_json::json!({"hello": "world"}))
            .with_error(GraphQLError::new("Warning".to_string()));

        assert!(result.data.is_some());
        assert!(result.has_errors());
        assert_eq!(result.errors.len(), 1);
    }

    #[tokio::test]
    async fn test_introspection_resolver() {
        let resolver = IntrospectionResolver::new();
        let context = ExecutionContext::new();
        let args = HashMap::new();

        // Test __schema field
        let result = resolver.resolve_field("__schema", &args, &context).await;
        assert!(result.is_ok());

        // Test __type field
        let result = resolver.resolve_field("__type", &args, &context).await;
        assert!(result.is_ok());

        // Test unknown field
        let result = resolver.resolve_field("unknown", &args, &context).await;
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), Value::NullValue));
    }

    #[test]
    fn test_resolver_registry() {
        let mut registry = ResolverRegistry::new();
        
        // Create an RDF store for testing
        let store = Arc::new(crate::RdfStore::new().unwrap());
        
        // Set up default resolvers
        registry.setup_default_resolvers(store);

        // Test that resolvers were registered
        assert!(registry.get("Query").is_some());
        assert!(registry.get("__Schema").is_some());
        assert!(registry.get("__Type").is_some());
        assert!(registry.get("NonExistent").is_none());
    }

    #[test]
    fn test_graphql_type_display() {
        let string_type = GraphQLType::Scalar(BuiltinScalars::string());
        assert_eq!(format!("{}", string_type), "String");

        let list_type = GraphQLType::List(Box::new(GraphQLType::Scalar(BuiltinScalars::int())));
        assert_eq!(format!("{}", list_type), "[Int]");

        let non_null_type = GraphQLType::NonNull(Box::new(GraphQLType::Scalar(BuiltinScalars::string())));
        assert_eq!(format!("{}", non_null_type), "String!");

        let non_null_list = GraphQLType::NonNull(Box::new(GraphQLType::List(Box::new(GraphQLType::Scalar(BuiltinScalars::string())))));
        assert_eq!(format!("{}", non_null_list), "[String]!");
    }

    #[test]
    fn test_graphql_type_properties() {
        let string_type = GraphQLType::Scalar(BuiltinScalars::string());
        assert!(string_type.is_nullable());
        assert!(string_type.is_scalar());
        assert!(!string_type.is_list());
        assert!(!string_type.is_object());

        let non_null_type = GraphQLType::NonNull(Box::new(string_type));
        assert!(!non_null_type.is_nullable());

        let list_type = GraphQLType::List(Box::new(GraphQLType::Scalar(BuiltinScalars::int())));
        assert!(list_type.is_list());

        let object_type = GraphQLType::Object(ObjectType::new("User".to_string()));
        assert!(object_type.is_object());
    }

    #[test]
    fn test_document_structure() {
        let field = Field {
            alias: None,
            name: "hello".to_string(),
            arguments: vec![],
            directives: vec![],
            selection_set: None,
        };

        let selection_set = SelectionSet {
            selections: vec![Selection::Field(field)],
        };

        let operation = OperationDefinition {
            operation_type: OperationType::Query,
            name: Some("TestQuery".to_string()),
            variable_definitions: vec![],
            directives: vec![],
            selection_set,
        };

        let document = Document {
            definitions: vec![Definition::Operation(operation)],
        };

        assert_eq!(document.definitions.len(), 1);
        if let Definition::Operation(op) = &document.definitions[0] {
            assert_eq!(op.name, Some("TestQuery".to_string()));
            assert!(matches!(op.operation_type, OperationType::Query));
            assert_eq!(op.selection_set.selections.len(), 1);
        } else {
            panic!("Expected operation definition");
        }
    }
}