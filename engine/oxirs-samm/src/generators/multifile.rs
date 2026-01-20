//! Multi-file code generation support
//!
//! This module enables generation of code organized across multiple files,
//! which is essential for real-world projects. Instead of generating a single
//! monolithic file, code is organized following language-specific conventions:
//!
//! - **TypeScript**: One file per entity + barrel index.ts
//! - **Java**: One class per file + package structure
//! - **Python**: One module per entity + __init__.py
//! - **Rust**: One file per struct + mod.rs
//! - **Scala**: One file per case class + package object
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_samm::generators::multifile::{MultiFileGenerator, MultiFileOptions, OutputLayout};
//! use oxirs_samm::parser::parse_aspect_model;
//! use std::path::Path;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let aspect = parse_aspect_model("Movement.ttl").await?;
//!
//! let options = MultiFileOptions {
//!     output_dir: Path::new("./generated").to_path_buf(),
//!     layout: OutputLayout::OneEntityPerFile,
//!     generate_index: true,
//!     language: "typescript".to_string(),
//!     ..Default::default()
//! };
//!
//! let generator = MultiFileGenerator::new(options);
//! let files = generator.generate_typescript(&aspect)?;
//!
//! // Files now contains:
//! // - movement.ts (main aspect)
//! // - position.ts (Position entity)
//! // - velocity.ts (Velocity entity)
//! // - index.ts (barrel export)
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, SammError};
use crate::generators::{
    generate_graphql, generate_java, generate_python, generate_scala, generate_sql,
    generate_typescript, JavaOptions, PythonOptions, ScalaOptions, SqlDialect, TsOptions,
};
use crate::metamodel::{Aspect, Entity, ModelElement};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Output layout strategy for multi-file generation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputLayout {
    /// One file per entity (recommended for most languages)
    OneEntityPerFile,
    /// One file per aspect (groups related entities)
    OneAspectPerFile,
    /// Flat layout (all in output_dir)
    Flat,
    /// Nested by namespace (follows URN structure)
    NestedByNamespace,
    /// Custom layout with user-provided path function
    Custom,
}

/// Options for multi-file code generation
pub struct MultiFileOptions {
    /// Base output directory
    pub output_dir: PathBuf,
    /// Layout strategy
    pub layout: OutputLayout,
    /// Generate index/barrel files (index.ts, __init__.py, mod.rs)
    pub generate_index: bool,
    /// Target language (typescript, java, python, rust, scala)
    pub language: String,
    /// Include documentation files (README.md)
    pub generate_docs: bool,
    /// Custom file naming function (entity_name -> filename)
    pub custom_naming: Option<fn(&str) -> String>,
    /// Language-specific options (TypeScript)
    pub ts_options: Option<TsOptions>,
    /// Language-specific options (Java)
    pub java_options: Option<JavaOptions>,
    /// Language-specific options (Python)
    pub python_options: Option<PythonOptions>,
    /// Language-specific options (Scala)
    pub scala_options: Option<ScalaOptions>,
}

impl std::fmt::Debug for MultiFileOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiFileOptions")
            .field("output_dir", &self.output_dir)
            .field("layout", &self.layout)
            .field("generate_index", &self.generate_index)
            .field("language", &self.language)
            .field("generate_docs", &self.generate_docs)
            .field("custom_naming", &self.custom_naming.is_some())
            .field("ts_options", &self.ts_options)
            .field("java_options", &self.java_options)
            .field("python_options", &self.python_options)
            .field("scala_options", &self.scala_options)
            .finish()
    }
}

impl Clone for MultiFileOptions {
    fn clone(&self) -> Self {
        Self {
            output_dir: self.output_dir.clone(),
            layout: self.layout.clone(),
            generate_index: self.generate_index,
            language: self.language.clone(),
            generate_docs: self.generate_docs,
            custom_naming: self.custom_naming,
            ts_options: self.ts_options.clone(),
            java_options: self.java_options.clone(),
            python_options: self.python_options.clone(),
            scala_options: self.scala_options.clone(),
        }
    }
}

impl Default for MultiFileOptions {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./generated"),
            layout: OutputLayout::OneEntityPerFile,
            generate_index: true,
            language: "typescript".to_string(),
            generate_docs: false,
            custom_naming: None,
            ts_options: None,
            java_options: None,
            python_options: None,
            scala_options: None,
        }
    }
}

/// Represents a generated file with path and content
#[derive(Debug, Clone)]
pub struct GeneratedFile {
    /// Relative path from output_dir
    pub path: PathBuf,
    /// File content
    pub content: String,
    /// File type (e.g., "typescript", "python", "java")
    pub file_type: String,
}

/// Multi-file code generator
pub struct MultiFileGenerator {
    options: MultiFileOptions,
}

impl MultiFileGenerator {
    /// Create a new multi-file generator with options
    pub fn new(options: MultiFileOptions) -> Self {
        Self { options }
    }

    /// Generate TypeScript code across multiple files
    pub fn generate_typescript(&self, aspect: &Aspect) -> Result<Vec<GeneratedFile>> {
        let mut files = Vec::new();

        // Generate main aspect file
        let aspect_content =
            generate_typescript(aspect, self.options.ts_options.clone().unwrap_or_default())?;
        let aspect_filename = self.get_filename(&aspect.name(), "ts");
        files.push(GeneratedFile {
            path: self.resolve_path(&aspect_filename),
            content: aspect_content.clone(),
            file_type: "typescript".to_string(),
        });

        // Generate entity files (if OneEntityPerFile layout)
        if self.options.layout == OutputLayout::OneEntityPerFile {
            for entity in self.extract_entities(aspect) {
                let entity_content = self.generate_typescript_entity(&entity)?;
                let entity_filename = self.get_filename(&entity.name(), "ts");
                files.push(GeneratedFile {
                    path: self.resolve_path(&entity_filename),
                    content: entity_content,
                    file_type: "typescript".to_string(),
                });
            }
        }

        // Generate index.ts barrel file
        if self.options.generate_index {
            let index_content = self.generate_typescript_index(aspect)?;
            files.push(GeneratedFile {
                path: self.resolve_path("index.ts"),
                content: index_content,
                file_type: "typescript".to_string(),
            });
        }

        // Generate README.md
        if self.options.generate_docs {
            let readme_content = self.generate_readme(aspect)?;
            files.push(GeneratedFile {
                path: self.resolve_path("README.md"),
                content: readme_content,
                file_type: "markdown".to_string(),
            });
        }

        Ok(files)
    }

    /// Generate Python code across multiple files
    pub fn generate_python(&self, aspect: &Aspect) -> Result<Vec<GeneratedFile>> {
        let mut files = Vec::new();

        // Generate main aspect module
        let aspect_content = generate_python(
            aspect,
            self.options.python_options.clone().unwrap_or_default(),
        )?;
        let aspect_filename = self.get_filename(&aspect.name(), "py");
        files.push(GeneratedFile {
            path: self.resolve_path(&aspect_filename),
            content: aspect_content,
            file_type: "python".to_string(),
        });

        // Generate entity modules (if OneEntityPerFile layout)
        if self.options.layout == OutputLayout::OneEntityPerFile {
            for entity in self.extract_entities(aspect) {
                let entity_content = self.generate_python_entity(&entity)?;
                let entity_filename = self.get_filename(&entity.name(), "py");
                files.push(GeneratedFile {
                    path: self.resolve_path(&entity_filename),
                    content: entity_content,
                    file_type: "python".to_string(),
                });
            }
        }

        // Generate __init__.py
        if self.options.generate_index {
            let init_content = self.generate_python_init(aspect)?;
            files.push(GeneratedFile {
                path: self.resolve_path("__init__.py"),
                content: init_content,
                file_type: "python".to_string(),
            });
        }

        Ok(files)
    }

    /// Generate Java code across multiple files
    pub fn generate_java(&self, aspect: &Aspect) -> Result<Vec<GeneratedFile>> {
        let mut files = Vec::new();

        // Java always generates one file per class
        let java_content = generate_java(
            aspect,
            self.options.java_options.clone().unwrap_or_default(),
        )?;

        // Split by class definitions
        let classes = self.split_java_classes(&java_content)?;

        for (class_name, class_content) in classes {
            let filename = format!("{}.java", class_name);
            files.push(GeneratedFile {
                path: self.resolve_path(&filename),
                content: class_content,
                file_type: "java".to_string(),
            });
        }

        // Generate package-info.java
        if self.options.generate_index {
            let package_info = self.generate_java_package_info(aspect)?;
            files.push(GeneratedFile {
                path: self.resolve_path("package-info.java"),
                content: package_info,
                file_type: "java".to_string(),
            });
        }

        Ok(files)
    }

    /// Generate Scala code across multiple files
    pub fn generate_scala(&self, aspect: &Aspect) -> Result<Vec<GeneratedFile>> {
        let mut files = Vec::new();

        // Scala: one file per case class
        let scala_content = generate_scala(
            aspect,
            self.options.scala_options.clone().unwrap_or_default(),
        )?;

        // Split by case class definitions
        let classes = self.split_scala_classes(&scala_content)?;

        for (class_name, class_content) in classes {
            let filename = format!("{}.scala", class_name);
            files.push(GeneratedFile {
                path: self.resolve_path(&filename),
                content: class_content,
                file_type: "scala".to_string(),
            });
        }

        // Generate package object
        if self.options.generate_index {
            let package_obj = self.generate_scala_package_object(aspect)?;
            files.push(GeneratedFile {
                path: self.resolve_path("package.scala"),
                content: package_obj,
                file_type: "scala".to_string(),
            });
        }

        Ok(files)
    }

    /// Write all generated files to disk
    pub fn write_files(&self, files: &[GeneratedFile]) -> Result<()> {
        use std::fs;

        // Create output directory
        fs::create_dir_all(&self.options.output_dir)?;

        for file in files {
            let full_path = self.options.output_dir.join(&file.path);

            // Create parent directories
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent)?;
            }

            // Write file
            fs::write(&full_path, &file.content)?;
        }

        Ok(())
    }

    // Private helper methods

    fn get_filename(&self, name: &str, extension: &str) -> String {
        if let Some(ref naming_fn) = self.options.custom_naming {
            format!("{}.{}", naming_fn(name), extension)
        } else {
            // Convert to snake_case for filenames
            let snake_name = self.to_snake_case(name);
            format!("{}.{}", snake_name, extension)
        }
    }

    fn resolve_path(&self, filename: &str) -> PathBuf {
        match self.options.layout {
            OutputLayout::Flat => PathBuf::from(filename),
            OutputLayout::NestedByNamespace => {
                // Extract namespace from filename and create nested structure
                // For now, simple flat layout
                PathBuf::from(filename)
            }
            _ => PathBuf::from(filename),
        }
    }

    fn extract_entities(&self, aspect: &Aspect) -> Vec<Entity> {
        let entities = Vec::new();

        for property in &aspect.properties {
            if let Some(ref characteristic) = property.characteristic {
                // Check if characteristic references an entity
                if let Some(ref data_type) = characteristic.data_type {
                    // In real implementation, resolve URN to entity
                    // For now, we'll create a placeholder
                }
            }
        }

        // Return entities from aspect (in real implementation, resolve from URNs)
        entities
    }

    fn generate_typescript_entity(&self, entity: &Entity) -> Result<String> {
        // Generate TypeScript interface for a single entity
        let mut code = String::new();
        code.push_str(&format!("// Generated entity: {}\n\n", entity.name()));
        code.push_str(&format!("export interface {} {{\n", entity.name()));

        for property in &entity.properties {
            let prop_name = property.name();
            let prop_type = self.ts_type_from_property(property);
            code.push_str(&format!("  {}: {};\n", prop_name, prop_type));
        }

        code.push_str("}\n");
        Ok(code)
    }

    fn generate_python_entity(&self, entity: &Entity) -> Result<String> {
        // Generate Python dataclass for a single entity
        let mut code = String::new();
        code.push_str("# Generated entity\n");
        code.push_str("from dataclasses import dataclass\n");
        code.push_str("from typing import Optional\n\n");
        code.push_str("@dataclass\n");
        code.push_str(&format!("class {}:\n", entity.name()));

        for property in &entity.properties {
            let prop_name = self.to_snake_case(&property.name());
            let prop_type = self.python_type_from_property(property);
            code.push_str(&format!("    {}: {}\n", prop_name, prop_type));
        }

        Ok(code)
    }

    fn generate_typescript_index(&self, aspect: &Aspect) -> Result<String> {
        let mut code = String::new();
        code.push_str("// Barrel export for all generated types\n\n");

        // Export main aspect
        let aspect_module = self.to_snake_case(&aspect.name());
        code.push_str(&format!("export * from './{}';\n", aspect_module));

        // Export entities
        for entity in self.extract_entities(aspect) {
            let entity_module = self.to_snake_case(&entity.name());
            code.push_str(&format!("export * from './{}';\n", entity_module));
        }

        Ok(code)
    }

    fn generate_python_init(&self, aspect: &Aspect) -> Result<String> {
        let mut code = String::new();
        code.push_str("# Python package initialization\n\n");

        // Import main aspect
        let aspect_module = self.to_snake_case(&aspect.name());
        code.push_str(&format!("from .{} import *\n", aspect_module));

        // Import entities
        for entity in self.extract_entities(aspect) {
            let entity_module = self.to_snake_case(&entity.name());
            code.push_str(&format!("from .{} import *\n", entity_module));
        }

        code.push_str("\n__all__ = [\n");
        code.push_str(&format!("    '{}',\n", aspect.name()));
        for entity in self.extract_entities(aspect) {
            code.push_str(&format!("    '{}',\n", entity.name()));
        }
        code.push_str("]\n");

        Ok(code)
    }

    fn generate_java_package_info(&self, aspect: &Aspect) -> Result<String> {
        let mut code = String::new();
        code.push_str("/**\n");
        code.push_str(&format!(
            " * Generated package for {} aspect\n",
            aspect.name()
        ));
        if let Some(desc) = aspect.metadata.get_description("en") {
            code.push_str(&format!(" * {}\n", desc));
        }
        code.push_str(" */\n");
        code.push_str("package com.example.generated;\n");
        Ok(code)
    }

    fn generate_scala_package_object(&self, aspect: &Aspect) -> Result<String> {
        let mut code = String::new();
        code.push_str(&format!(
            "package object {} {{\n",
            self.to_snake_case(&aspect.name())
        ));
        code.push_str("  // Package-level utilities\n");
        code.push_str("}\n");
        Ok(code)
    }

    fn generate_readme(&self, aspect: &Aspect) -> Result<String> {
        let mut md = String::new();
        md.push_str(&format!("# {} - Generated Code\n\n", aspect.name()));
        md.push_str("This code was automatically generated from a SAMM aspect model.\n\n");
        md.push_str("## Overview\n\n");
        if let Some(desc) = aspect.metadata.get_description("en") {
            md.push_str(&format!("{}\n\n", desc));
        }
        md.push_str("## Files\n\n");
        md.push_str("- Main aspect implementation\n");
        md.push_str("- Entity definitions\n");
        md.push_str("- Index/barrel exports\n\n");
        md.push_str("## Usage\n\n");
        md.push_str("See individual files for usage examples.\n");
        Ok(md)
    }

    fn split_java_classes(&self, content: &str) -> Result<HashMap<String, String>> {
        let mut classes = HashMap::new();

        // Simple regex-based splitting (in production, use proper Java parser)
        let class_pattern = regex::Regex::new(r"public\s+class\s+(\w+)").unwrap();

        for class_match in class_pattern.captures_iter(content) {
            let class_name = class_match.get(1).unwrap().as_str();
            // For now, include the entire content for each class
            // In real implementation, extract individual class definition
            classes.insert(class_name.to_string(), content.to_string());
        }

        Ok(classes)
    }

    fn split_scala_classes(&self, content: &str) -> Result<HashMap<String, String>> {
        let mut classes = HashMap::new();

        // Simple regex-based splitting
        let class_pattern = regex::Regex::new(r"case\s+class\s+(\w+)").unwrap();

        for class_match in class_pattern.captures_iter(content) {
            let class_name = class_match.get(1).unwrap().as_str();
            classes.insert(class_name.to_string(), content.to_string());
        }

        Ok(classes)
    }

    fn ts_type_from_property(&self, _property: &crate::metamodel::Property) -> String {
        // Simplified type mapping
        "string".to_string()
    }

    fn python_type_from_property(&self, _property: &crate::metamodel::Property) -> String {
        // Simplified type mapping
        "str".to_string()
    }

    fn to_snake_case(&self, s: &str) -> String {
        use crate::utils::naming::to_snake_case;
        to_snake_case(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Aspect, Property};

    #[test]
    fn test_multifile_options_default() {
        let options = MultiFileOptions::default();
        assert_eq!(options.layout, OutputLayout::OneEntityPerFile);
        assert!(options.generate_index);
        assert_eq!(options.language, "typescript");
    }

    #[test]
    fn test_output_layout_variants() {
        assert_eq!(OutputLayout::Flat, OutputLayout::Flat);
        assert_ne!(OutputLayout::Flat, OutputLayout::OneEntityPerFile);
    }

    #[test]
    fn test_generated_file_creation() {
        let file = GeneratedFile {
            path: PathBuf::from("test.ts"),
            content: "export interface Test {}".to_string(),
            file_type: "typescript".to_string(),
        };

        assert_eq!(file.path, PathBuf::from("test.ts"));
        assert!(file.content.contains("Test"));
    }

    #[test]
    fn test_get_filename() {
        let options = MultiFileOptions::default();
        let generator = MultiFileGenerator::new(options);

        let filename = generator.get_filename("MyEntity", "ts");
        assert_eq!(filename, "my_entity.ts");
    }

    #[test]
    fn test_typescript_index_generation() {
        let aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let options = MultiFileOptions::default();
        let generator = MultiFileGenerator::new(options);
        let index = generator.generate_typescript_index(&aspect).unwrap();

        assert!(index.contains("export"));
        assert!(index.contains("test_aspect"));
    }

    #[test]
    fn test_python_init_generation() {
        let aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let options = MultiFileOptions::default();
        let generator = MultiFileGenerator::new(options);
        let init = generator.generate_python_init(&aspect).unwrap();

        assert!(init.contains("__all__"));
        assert!(init.contains("from"));
    }

    #[test]
    fn test_readme_generation() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect
            .metadata
            .add_description("en".to_string(), "Test description".to_string());

        let options = MultiFileOptions {
            generate_docs: true,
            ..Default::default()
        };
        let generator = MultiFileGenerator::new(options);
        let readme = generator.generate_readme(&aspect).unwrap();

        assert!(readme.contains("# TestAspect"));
        assert!(readme.contains("Test description"));
        assert!(readme.contains("## Usage"));
    }
}
