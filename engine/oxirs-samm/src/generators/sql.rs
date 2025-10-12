//! SAMM to SQL Schema Generator
//!
//! Generates SQL DDL (Data Definition Language) from SAMM Aspect models.
//! Supports PostgreSQL, MySQL, and SQLite dialects.

use crate::error::SammError;
use crate::metamodel::{Aspect, ModelElement};

/// SQL database dialect
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlDialect {
    /// PostgreSQL
    PostgreSql,
    /// MySQL
    MySql,
    /// SQLite
    Sqlite,
}

/// Generate SQL DDL from SAMM Aspect
pub fn generate_sql(aspect: &Aspect, dialect: SqlDialect) -> Result<String, SammError> {
    match dialect {
        SqlDialect::PostgreSql => generate_postgresql(aspect),
        SqlDialect::MySql => generate_mysql(aspect),
        SqlDialect::Sqlite => generate_sqlite(aspect),
    }
}

/// Generate PostgreSQL DDL
fn generate_postgresql(aspect: &Aspect) -> Result<String, SammError> {
    let table_name = to_snake_case(&aspect.name());
    let mut sql = String::new();
    let mut foreign_keys = Vec::new();

    sql.push_str(&format!("-- PostgreSQL DDL for {}\n", aspect.name()));
    sql.push_str(&format!("CREATE TABLE {} (\n", table_name));
    sql.push_str("  id BIGSERIAL PRIMARY KEY,\n");

    for prop in aspect.properties() {
        let col_name = to_snake_case(&prop.name());

        // Check if this property references an Entity
        let (sql_type, is_foreign_key) = if let Some(char) = &prop.characteristic {
            match char.kind() {
                crate::metamodel::CharacteristicKind::SingleEntity { entity_type } => {
                    // This is a foreign key to another entity
                    let ref_table =
                        to_snake_case(entity_type.split('#').next_back().unwrap_or(entity_type));
                    foreign_keys.push((col_name.clone(), ref_table));
                    ("BIGINT".to_string(), true)
                }
                _ => {
                    if let Some(dt) = &char.data_type {
                        (map_xsd_to_postgresql(dt), false)
                    } else {
                        ("TEXT".to_string(), false)
                    }
                }
            }
        } else {
            ("TEXT".to_string(), false)
        };

        let nullable = if prop.optional { "" } else { " NOT NULL" };
        sql.push_str(&format!("  {} {}{},\n", col_name, sql_type, nullable));
    }

    sql.push_str("  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,\n");
    sql.push_str("  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP");

    // Add foreign key constraints
    for (col_name, ref_table) in &foreign_keys {
        sql.push_str(",\n");
        sql.push_str(&format!(
            "  CONSTRAINT fk_{}_{} FOREIGN KEY ({}) REFERENCES {} (id)",
            table_name, col_name, col_name, ref_table
        ));
    }

    sql.push_str("\n);\n\n");

    // Add indexes for required fields and foreign keys
    for prop in aspect.properties() {
        if !prop.optional {
            let col_name = to_snake_case(&prop.name());
            sql.push_str(&format!(
                "CREATE INDEX idx_{}_{} ON {} ({});\n",
                table_name, col_name, table_name, col_name
            ));
        }
    }

    // Add indexes on foreign keys
    for (col_name, _) in &foreign_keys {
        sql.push_str(&format!(
            "CREATE INDEX idx_{}_{}_fk ON {} ({});\n",
            table_name, col_name, table_name, col_name
        ));
    }

    Ok(sql)
}

/// Generate MySQL DDL
fn generate_mysql(aspect: &Aspect) -> Result<String, SammError> {
    let table_name = to_snake_case(&aspect.name());
    let mut sql = String::new();
    let mut foreign_keys = Vec::new();

    sql.push_str(&format!("-- MySQL DDL for {}\n", aspect.name()));
    sql.push_str(&format!("CREATE TABLE {} (\n", table_name));
    sql.push_str("  id BIGINT AUTO_INCREMENT PRIMARY KEY,\n");

    for prop in aspect.properties() {
        let col_name = to_snake_case(&prop.name());

        let (sql_type, _is_foreign_key) = if let Some(char) = &prop.characteristic {
            match char.kind() {
                crate::metamodel::CharacteristicKind::SingleEntity { entity_type } => {
                    let ref_table =
                        to_snake_case(entity_type.split('#').next_back().unwrap_or(entity_type));
                    foreign_keys.push((col_name.clone(), ref_table));
                    ("BIGINT".to_string(), true)
                }
                _ => {
                    if let Some(dt) = &char.data_type {
                        (map_xsd_to_mysql(dt), false)
                    } else {
                        ("TEXT".to_string(), false)
                    }
                }
            }
        } else {
            ("TEXT".to_string(), false)
        };

        let nullable = if prop.optional { "" } else { " NOT NULL" };
        sql.push_str(&format!("  {} {}{},\n", col_name, sql_type, nullable));
    }

    sql.push_str("  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n");
    sql.push_str("  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP");

    // Add foreign key constraints
    for (col_name, ref_table) in &foreign_keys {
        sql.push_str(",\n");
        sql.push_str(&format!(
            "  CONSTRAINT fk_{}_{} FOREIGN KEY ({}) REFERENCES {} (id)",
            table_name, col_name, col_name, ref_table
        ));
    }

    sql.push_str("\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;\n\n");

    // Add indexes on foreign keys
    for (col_name, _) in &foreign_keys {
        sql.push_str(&format!(
            "CREATE INDEX idx_{}_{}_fk ON {} ({});\n",
            table_name, col_name, table_name, col_name
        ));
    }

    Ok(sql)
}

/// Generate SQLite DDL
fn generate_sqlite(aspect: &Aspect) -> Result<String, SammError> {
    let table_name = to_snake_case(&aspect.name());
    let mut sql = String::new();
    let mut foreign_keys = Vec::new();

    sql.push_str(&format!("-- SQLite DDL for {}\n", aspect.name()));
    sql.push_str(&format!("CREATE TABLE {} (\n", table_name));
    sql.push_str("  id INTEGER PRIMARY KEY AUTOINCREMENT,\n");

    for prop in aspect.properties() {
        let col_name = to_snake_case(&prop.name());

        let (sql_type, _is_foreign_key) = if let Some(char) = &prop.characteristic {
            match char.kind() {
                crate::metamodel::CharacteristicKind::SingleEntity { entity_type } => {
                    let ref_table =
                        to_snake_case(entity_type.split('#').next_back().unwrap_or(entity_type));
                    foreign_keys.push((col_name.clone(), ref_table));
                    ("INTEGER".to_string(), true)
                }
                _ => {
                    if let Some(dt) = &char.data_type {
                        (map_xsd_to_sqlite(dt), false)
                    } else {
                        ("TEXT".to_string(), false)
                    }
                }
            }
        } else {
            ("TEXT".to_string(), false)
        };

        let nullable = if prop.optional { "" } else { " NOT NULL" };
        sql.push_str(&format!("  {} {}{},\n", col_name, sql_type, nullable));
    }

    sql.push_str("  created_at TEXT DEFAULT CURRENT_TIMESTAMP,\n");
    sql.push_str("  updated_at TEXT DEFAULT CURRENT_TIMESTAMP");

    // Add foreign key constraints
    for (col_name, ref_table) in &foreign_keys {
        sql.push_str(",\n");
        sql.push_str(&format!(
            "  FOREIGN KEY ({}) REFERENCES {} (id)",
            col_name, ref_table
        ));
    }

    sql.push_str("\n);\n\n");

    // Add indexes on foreign keys
    for (col_name, _) in &foreign_keys {
        sql.push_str(&format!(
            "CREATE INDEX idx_{}_{}_fk ON {} ({});\n",
            table_name, col_name, table_name, col_name
        ));
    }

    Ok(sql)
}

/// Map XSD types to PostgreSQL types
fn map_xsd_to_postgresql(xsd_type: &str) -> String {
    match xsd_type {
        t if t.ends_with("string") => "TEXT".to_string(),
        t if t.ends_with("int") | t.ends_with("integer") => "INTEGER".to_string(),
        t if t.ends_with("long") => "BIGINT".to_string(),
        t if t.ends_with("short") | t.ends_with("byte") => "SMALLINT".to_string(),
        t if t.ends_with("decimal") => "NUMERIC".to_string(),
        t if t.ends_with("float") => "REAL".to_string(),
        t if t.ends_with("double") => "DOUBLE PRECISION".to_string(),
        t if t.ends_with("boolean") => "BOOLEAN".to_string(),
        t if t.ends_with("date") => "DATE".to_string(),
        t if t.ends_with("dateTime") | t.ends_with("dateTimeStamp") => {
            "TIMESTAMP WITH TIME ZONE".to_string()
        }
        t if t.ends_with("time") => "TIME".to_string(),
        t if t.ends_with("anyURI") => "TEXT".to_string(),
        _ => "TEXT".to_string(),
    }
}

/// Map XSD types to MySQL types
fn map_xsd_to_mysql(xsd_type: &str) -> String {
    match xsd_type {
        t if t.ends_with("string") => "VARCHAR(255)".to_string(),
        t if t.ends_with("int") | t.ends_with("integer") => "INT".to_string(),
        t if t.ends_with("long") => "BIGINT".to_string(),
        t if t.ends_with("short") | t.ends_with("byte") => "SMALLINT".to_string(),
        t if t.ends_with("decimal") => "DECIMAL(10,2)".to_string(),
        t if t.ends_with("float") => "FLOAT".to_string(),
        t if t.ends_with("double") => "DOUBLE".to_string(),
        t if t.ends_with("boolean") => "BOOLEAN".to_string(),
        t if t.ends_with("date") => "DATE".to_string(),
        t if t.ends_with("dateTime") | t.ends_with("dateTimeStamp") => "DATETIME".to_string(),
        t if t.ends_with("time") => "TIME".to_string(),
        t if t.ends_with("anyURI") => "TEXT".to_string(),
        _ => "TEXT".to_string(),
    }
}

/// Map XSD types to SQLite types
fn map_xsd_to_sqlite(xsd_type: &str) -> String {
    match xsd_type {
        t if t.ends_with("string") => "TEXT".to_string(),
        t if t.ends_with("int")
            | t.ends_with("integer")
            | t.ends_with("long")
            | t.ends_with("short")
            | t.ends_with("byte") =>
        {
            "INTEGER".to_string()
        }
        t if t.ends_with("decimal") | t.ends_with("float") | t.ends_with("double") => {
            "REAL".to_string()
        }
        t if t.ends_with("boolean") => "INTEGER".to_string(), // 0 or 1
        t if t.ends_with("date")
            | t.ends_with("dateTime")
            | t.ends_with("dateTimeStamp")
            | t.ends_with("time") =>
        {
            "TEXT".to_string()
        }
        _ => "TEXT".to_string(),
    }
}

/// Convert PascalCase/camelCase to snake_case
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(ch.to_lowercase().next().unwrap());
        } else {
            result.push(ch);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sql_dialect_variants() {
        assert_eq!(SqlDialect::PostgreSql, SqlDialect::PostgreSql);
        assert_ne!(SqlDialect::MySql, SqlDialect::Sqlite);
    }

    #[test]
    fn test_snake_case_conversion() {
        assert_eq!(to_snake_case("MovementAspect"), "movement_aspect");
        assert_eq!(to_snake_case("position"), "position");
    }
}
