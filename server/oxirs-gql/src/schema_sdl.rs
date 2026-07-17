//! SDL serialization and naming helpers for [`SchemaGenerator`].
//!
//! Provides the GraphQL Schema Definition Language (SDL) writer used to render
//! a generated [`Schema`], plus the URI-to-name
//! conversion utilities (PascalCase / camelCase / pluralization).

use crate::schema_generator::SchemaGenerator;
use crate::types::*;
use std::fmt::Write;

impl SchemaGenerator {
    /// Extract a URI's local name (the fragment after `#`, or otherwise the
    /// last `/`-separated path segment) and PascalCase it into a GraphQL
    /// type name.
    ///
    /// Note: `str::split` always yields at least one item, even when the
    /// separator is absent (the whole string, unsplit) -- so naively taking
    /// `uri.split('#').next_back()` is *always* `Some(..)` and would always
    /// take this branch, silently making the `split('/')` fallback dead
    /// code and PascalCase-ing the *entire* URI (scheme, host, punctuation
    /// and all) for any class/property URI that doesn't contain `#`. Most
    /// real-world vocabularies (schema.org, DBpedia, ...) use `/`-style
    /// URIs, so this matters in practice; use `rfind` to only take the
    /// fragment/segment when the separator is actually present.
    pub(crate) fn uri_to_graphql_name(&self, uri: &str) -> String {
        let local_name = if let Some(idx) = uri.rfind('#') {
            &uri[idx + 1..]
        } else if let Some(idx) = uri.rfind('/') {
            &uri[idx + 1..]
        } else {
            uri
        };

        if local_name.is_empty() {
            "Resource".to_string()
        } else {
            self.to_pascal_case(local_name)
        }
    }

    pub(crate) fn to_pascal_case(&self, input: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = true;

        for ch in input.chars() {
            if ch.is_alphanumeric() {
                if capitalize_next {
                    result.push(ch.to_uppercase().next().unwrap_or(ch));
                    capitalize_next = false;
                } else {
                    result.push(ch);
                }
            } else {
                capitalize_next = true;
            }
        }

        result
    }

    pub(crate) fn to_camel_case(&self, input: &str) -> String {
        let pascal = self.to_pascal_case(input);
        if let Some(first_char) = pascal.chars().next() {
            first_char.to_lowercase().collect::<String>() + &pascal[first_char.len_utf8()..]
        } else {
            pascal
        }
    }

    pub(crate) fn pluralize(&self, word: &str) -> String {
        if word.ends_with('s') || word.ends_with("sh") || word.ends_with("ch") {
            format!("{word}es")
        } else if let Some(stripped) = word.strip_suffix('y') {
            format!("{stripped}ies")
        } else {
            format!("{word}s")
        }
    }

    pub(crate) fn schema_to_sdl(&self, schema: &Schema) -> String {
        let mut sdl = String::new();

        // Write schema definition
        writeln!(sdl, "schema {{").expect("writing to String should not fail");
        if let Some(ref query) = schema.query_type {
            writeln!(sdl, "  query: {query}").expect("writing to String should not fail");
        }
        if let Some(ref mutation) = schema.mutation_type {
            writeln!(sdl, "  mutation: {mutation}").expect("writing to String should not fail");
        }
        if let Some(ref subscription) = schema.subscription_type {
            writeln!(sdl, "  subscription: {subscription}")
                .expect("writing to String should not fail");
        }
        writeln!(sdl, "}}").expect("writing to String should not fail");
        writeln!(sdl).expect("writing to String should not fail");

        // Write type definitions
        for graphql_type in schema.types.values() {
            match graphql_type {
                GraphQLType::Object(obj) => {
                    self.write_object_type_sdl(&mut sdl, obj);
                }
                GraphQLType::Scalar(scalar)
                    if !["String", "Int", "Float", "Boolean", "ID"]
                        .contains(&scalar.name.as_str()) =>
                {
                    self.write_scalar_type_sdl(&mut sdl, scalar);
                }
                GraphQLType::Enum(enum_type) => {
                    self.write_enum_type_sdl(&mut sdl, enum_type);
                }
                GraphQLType::Interface(interface) => {
                    self.write_interface_type_sdl(&mut sdl, interface);
                }
                GraphQLType::Union(union_type) => {
                    self.write_union_type_sdl(&mut sdl, union_type);
                }
                _ => {} // Skip other types
            }
        }

        sdl
    }

    fn write_object_type_sdl(&self, sdl: &mut String, obj: &ObjectType) {
        if let Some(ref description) = obj.description {
            writeln!(sdl, "\"\"\"\n{description}\n\"\"\"")
                .expect("writing to String should not fail");
        }

        write!(sdl, "type {}", obj.name).expect("writing to String should not fail");

        if !obj.interfaces.is_empty() {
            write!(sdl, " implements {}", obj.interfaces.join(" & "))
                .expect("writing to String should not fail");
        }

        writeln!(sdl, " {{").expect("writing to String should not fail");

        for field in obj.fields.values() {
            self.write_field_sdl(sdl, field);
        }

        writeln!(sdl, "}}").expect("writing to String should not fail");
        writeln!(sdl).expect("writing to String should not fail");
    }

    fn write_field_sdl(&self, sdl: &mut String, field: &FieldType) {
        if let Some(ref description) = field.description {
            writeln!(sdl, "  \"{description}\"").expect("writing to String should not fail");
        }

        write!(sdl, "  {}", field.name).expect("writing to String should not fail");

        if !field.arguments.is_empty() {
            write!(sdl, "(").expect("writing to String should not fail");
            let args: Vec<String> = field
                .arguments
                .values()
                .map(|arg| format!("{}: {}", arg.name, arg.argument_type))
                .collect();
            write!(sdl, "{}", args.join(", ")).expect("writing to String should not fail");
            write!(sdl, ")").expect("writing to String should not fail");
        }

        writeln!(sdl, ": {}", field.field_type).expect("writing to String should not fail");
    }

    fn write_scalar_type_sdl(&self, sdl: &mut String, scalar: &ScalarType) {
        if let Some(ref description) = scalar.description {
            writeln!(sdl, "\"\"\"\n{description}\n\"\"\"")
                .expect("writing to String should not fail");
        }
        writeln!(sdl, "scalar {}", scalar.name).expect("writing to String should not fail");
        writeln!(sdl).expect("writing to String should not fail");
    }

    fn write_enum_type_sdl(&self, sdl: &mut String, enum_type: &EnumType) {
        if let Some(ref description) = enum_type.description {
            writeln!(sdl, "\"\"\"\n{description}\n\"\"\"")
                .expect("writing to String should not fail");
        }
        writeln!(sdl, "enum {} {{", enum_type.name).expect("writing to String should not fail");

        for value in enum_type.values.values() {
            if let Some(ref description) = value.description {
                writeln!(sdl, "  \"{description}\"").expect("writing to String should not fail");
            }
            writeln!(sdl, "  {}", value.name).expect("writing to String should not fail");
        }

        writeln!(sdl, "}}").expect("writing to String should not fail");
        writeln!(sdl).expect("writing to String should not fail");
    }

    fn write_interface_type_sdl(&self, sdl: &mut String, interface: &InterfaceType) {
        if let Some(ref description) = interface.description {
            writeln!(sdl, "\"\"\"\n{description}\n\"\"\"")
                .expect("writing to String should not fail");
        }
        writeln!(sdl, "interface {} {{", interface.name)
            .expect("writing to String should not fail");

        for field in interface.fields.values() {
            self.write_field_sdl(sdl, field);
        }

        writeln!(sdl, "}}").expect("writing to String should not fail");
        writeln!(sdl).expect("writing to String should not fail");
    }

    fn write_union_type_sdl(&self, sdl: &mut String, union_type: &UnionType) {
        if let Some(ref description) = union_type.description {
            writeln!(sdl, "\"\"\"\n{description}\n\"\"\"")
                .expect("writing to String should not fail");
        }
        writeln!(
            sdl,
            "union {} = {}",
            union_type.name,
            union_type.types.join(" | ")
        )
        .expect("writing to String should not fail");
        writeln!(sdl).expect("writing to String should not fail");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `uri_to_graphql_name` used to PascalCase the
    /// *entire* URI (scheme, host and all) for any class/property URI
    /// without a `#` fragment, because `uri.split('#').next_back()` is
    /// always `Some(..)` even when `#` is absent (it just yields the whole
    /// string), which made the `/`-segment fallback unreachable. Most
    /// real-world vocabularies use `/`-style URIs, so this must extract
    /// just the last path segment.
    #[test]
    fn test_uri_to_graphql_name_uses_last_path_segment_for_slash_uris() {
        let generator = SchemaGenerator::new();
        assert_eq!(
            generator.uri_to_graphql_name("http://example.org/Person"),
            "Person"
        );
        assert_eq!(
            generator.uri_to_graphql_name("http://xmlns.com/foaf/0.1/name"),
            "Name"
        );
        assert_eq!(
            generator.uri_to_graphql_name("http://schema.org/BlogPosting"),
            "BlogPosting"
        );
    }

    #[test]
    fn test_uri_to_graphql_name_uses_fragment_for_hash_uris() {
        let generator = SchemaGenerator::new();
        assert_eq!(
            generator.uri_to_graphql_name("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            "Type"
        );
    }

    #[test]
    fn test_uri_to_graphql_name_falls_back_to_resource_for_empty_local_name() {
        let generator = SchemaGenerator::new();
        assert_eq!(generator.uri_to_graphql_name(""), "Resource");
        assert_eq!(
            generator.uri_to_graphql_name("http://example.org/"),
            "Resource"
        );
    }
}
