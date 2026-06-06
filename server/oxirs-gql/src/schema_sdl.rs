//! SDL serialization and naming helpers for [`SchemaGenerator`].
//!
//! Provides the GraphQL Schema Definition Language (SDL) writer used to render
//! a generated [`Schema`](crate::types::Schema), plus the URI-to-name
//! conversion utilities (PascalCase / camelCase / pluralization).

use crate::schema_generator::SchemaGenerator;
use crate::types::*;
use std::fmt::Write;

impl SchemaGenerator {
    pub(crate) fn uri_to_graphql_name(&self, uri: &str) -> String {
        if let Some(fragment) = uri.split('#').next_back() {
            self.to_pascal_case(fragment)
        } else if let Some(segment) = uri.split('/').next_back() {
            self.to_pascal_case(segment)
        } else {
            "Resource".to_string()
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
