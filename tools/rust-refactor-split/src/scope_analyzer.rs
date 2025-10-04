//! Module scope analysis for correct impl block placement
//!
//! This module solves the critical problem of where impl blocks can live in Rust's module system.
//!
//! Key Insight: In Rust, impl blocks must either:
//! 1. Be in the same module as the type they implement
//! 2. Use #[path] attributes to include them as submodules
//! 3. Live in a parent module that includes the type

use std::collections::HashMap;
use syn::{Item, ItemImpl};

/// Analyzes module scope and determines correct placement for impl blocks
pub struct ScopeAnalyzer {
    /// Maps type names to their module location
    type_locations: HashMap<String, TypeLocation>,

    /// Impl blocks that need to be placed
    impl_blocks: Vec<ImplBlockInfo>,
}

/// Location of a type definition in the module hierarchy
#[derive(Debug, Clone)]
pub struct TypeLocation {
    /// Name of the type
    pub type_name: String,

    /// Module where the type is defined
    pub module_path: String,

    /// Whether the type should have a dedicated module for its impls
    pub needs_impl_module: bool,
}

/// Information about an impl block that needs placement
#[derive(Clone)]
pub struct ImplBlockInfo {
    /// Name of the type this impl is for
    pub type_name: String,

    /// The impl block itself
    pub impl_item: ItemImpl,

    /// Suggested module name for this impl group
    pub suggested_module: String,

    /// Number of methods in this impl block
    pub method_count: usize,
}

/// Strategy for organizing impl blocks
#[derive(Debug, Clone, PartialEq)]
pub enum ImplOrganizationStrategy {
    /// Keep all impl blocks in the type's module
    Inline,

    /// Create a submodule for impl blocks with #[path] includes
    Submodule {
        /// Name of the parent module containing the type
        parent_module: String,

        /// Names of impl block modules
        impl_modules: Vec<String>,
    },

    /// Use a wrapper module pattern (type + impls together)
    Wrapper {
        /// Name of the wrapper module
        module_name: String,
    },
}

impl ScopeAnalyzer {
    pub fn new() -> Self {
        Self {
            type_locations: HashMap::new(),
            impl_blocks: Vec::new(),
        }
    }

    /// Analyze a parsed file to determine type locations
    pub fn analyze_types(&mut self, items: &[Item]) {
        for item in items {
            match item {
                Item::Struct(s) => {
                    self.register_type(&s.ident.to_string(), "types");
                }
                Item::Enum(e) => {
                    self.register_type(&e.ident.to_string(), "types");
                }
                _ => {}
            }
        }
    }

    fn register_type(&mut self, type_name: &str, module_path: &str) {
        self.type_locations.insert(
            type_name.to_string(),
            TypeLocation {
                type_name: type_name.to_string(),
                module_path: module_path.to_string(),
                needs_impl_module: false,
            },
        );
    }

    /// Mark a type as needing a dedicated impl module
    pub fn mark_needs_impl_module(&mut self, type_name: &str) {
        if let Some(location) = self.type_locations.get_mut(type_name) {
            location.needs_impl_module = true;
        }
    }

    /// Register an impl block that needs placement
    pub fn register_impl_block(
        &mut self,
        type_name: String,
        impl_item: ItemImpl,
        suggested_module: String,
        method_count: usize,
    ) {
        self.impl_blocks.push(ImplBlockInfo {
            type_name,
            impl_item,
            suggested_module,
            method_count,
        });
    }

    /// Determine the best organization strategy for a type's impl blocks
    pub fn determine_strategy(&self, type_name: &str) -> ImplOrganizationStrategy {
        let impl_blocks: Vec<_> = self
            .impl_blocks
            .iter()
            .filter(|b| b.type_name == type_name)
            .collect();

        if impl_blocks.is_empty() {
            return ImplOrganizationStrategy::Inline;
        }

        let total_methods: usize = impl_blocks.iter().map(|b| b.method_count).sum();

        // Decision tree for organization strategy
        if total_methods < 10 {
            // Small number of methods - keep inline
            ImplOrganizationStrategy::Inline
        } else if impl_blocks.len() == 1 {
            // One large impl block - use wrapper pattern
            ImplOrganizationStrategy::Wrapper {
                module_name: format!("{}_module", type_name.to_lowercase()),
            }
        } else {
            // Multiple impl blocks - use submodule pattern
            let impl_modules: Vec<String> = impl_blocks
                .iter()
                .map(|b| b.suggested_module.clone())
                .collect();

            ImplOrganizationStrategy::Submodule {
                parent_module: format!("{}_type", type_name.to_lowercase()),
                impl_modules,
            }
        }
    }

    /// Generate the correct module structure code for a type with impl blocks
    pub fn generate_module_structure(&self, type_name: &str) -> ModuleStructure {
        let strategy = self.determine_strategy(type_name);

        match strategy {
            ImplOrganizationStrategy::Inline => {
                ModuleStructure {
                    type_module: format!("{}_type", type_name.to_lowercase()),
                    needs_path_attributes: false,
                    path_includes: Vec::new(),
                    re_exports: vec![format!("pub use {}::*;", format!("{}_type", type_name.to_lowercase()))],
                }
            }

            ImplOrganizationStrategy::Wrapper { module_name } => {
                ModuleStructure {
                    type_module: module_name.clone(),
                    needs_path_attributes: false,
                    path_includes: Vec::new(),
                    re_exports: vec![format!("pub use {}::*;", module_name)],
                }
            }

            ImplOrganizationStrategy::Submodule { parent_module, impl_modules } => {
                let path_includes: Vec<String> = impl_modules
                    .iter()
                    .map(|m| format!("#[path = \"{}.rs\"]\nmod {};", m, m.replace('-', "_")))
                    .collect();

                ModuleStructure {
                    type_module: parent_module.clone(),
                    needs_path_attributes: true,
                    path_includes,
                    re_exports: vec![format!("pub use {}::*;", parent_module)],
                }
            }
        }
    }

    /// Generate the type definition module content with proper impl includes
    pub fn generate_type_module_content(
        &self,
        type_name: &str,
        type_item: &Item,
        _struct_visibility: &str,
    ) -> String {
        let strategy = self.determine_strategy(type_name);
        let mut content = String::new();

        content.push_str("//! Auto-generated type module\n\n");

        // Add necessary imports
        content.push_str("use std::collections::{HashMap, HashSet};\n");
        content.push_str("use super::super::types::*;\n\n");

        // Generate the struct definition with correct visibility
        match type_item {
            Item::Struct(s) => {
                let struct_code = quote::quote! { #s }.to_string();
                content.push_str(&struct_code);
                content.push_str("\n\n");
            }
            Item::Enum(e) => {
                let enum_code = quote::quote! { #e }.to_string();
                content.push_str(&enum_code);
                content.push_str("\n\n");
            }
            _ => {}
        }

        // Add impl module includes if using submodule strategy
        if let ImplOrganizationStrategy::Submodule { impl_modules, .. } = strategy {
            content.push_str("// Include impl block modules\n");
            for module in impl_modules {
                content.push_str(&format!("#[path = \"{}.rs\"]\n", module));
                content.push_str(&format!("mod {};\n", module.replace('-', "_")));
            }
        }

        content
    }

    /// Infer the correct visibility for struct fields when splitting impl blocks
    pub fn infer_field_visibility(&self, type_name: &str) -> FieldVisibility {
        let strategy = self.determine_strategy(type_name);

        match strategy {
            ImplOrganizationStrategy::Inline => FieldVisibility::Private,
            ImplOrganizationStrategy::Wrapper { .. } => FieldVisibility::Private,
            ImplOrganizationStrategy::Submodule { .. } => {
                // Need pub(super) for fields accessed from impl modules
                FieldVisibility::PubSuper
            }
        }
    }
}

/// Recommended visibility for struct fields
#[derive(Debug, Clone, PartialEq)]
pub enum FieldVisibility {
    Private,
    PubSuper,
    PubCrate,
    Pub,
}

/// Generated module structure information
#[derive(Debug, Clone)]
pub struct ModuleStructure {
    /// Name of the module containing the type
    pub type_module: String,

    /// Whether #[path] attributes are needed
    pub needs_path_attributes: bool,

    /// Path include statements to add
    pub path_includes: Vec<String>,

    /// Re-export statements for mod.rs
    pub re_exports: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_analyzer_creation() {
        let analyzer = ScopeAnalyzer::new();
        assert!(analyzer.type_locations.is_empty());
        assert!(analyzer.impl_blocks.is_empty());
    }

    #[test]
    fn test_register_type() {
        let mut analyzer = ScopeAnalyzer::new();
        analyzer.register_type("TestStruct", "types");

        assert_eq!(analyzer.type_locations.len(), 1);
        assert!(analyzer.type_locations.contains_key("TestStruct"));
    }

    #[test]
    fn test_strategy_inline_few_methods() {
        let mut analyzer = ScopeAnalyzer::new();
        analyzer.register_type("SmallStruct", "types");

        // Register one impl with few methods
        let impl_item = syn::parse_quote! {
            impl SmallStruct {
                pub fn method1(&self) {}
                pub fn method2(&self) {}
            }
        };

        analyzer.register_impl_block(
            "SmallStruct".to_string(),
            impl_item,
            "smallstruct_methods".to_string(),
            2,
        );

        let strategy = analyzer.determine_strategy("SmallStruct");
        assert_eq!(strategy, ImplOrganizationStrategy::Inline);
    }

    #[test]
    fn test_field_visibility_inference() {
        let mut analyzer = ScopeAnalyzer::new();
        analyzer.register_type("TestStruct", "types");

        // Inline strategy -> private fields
        let visibility = analyzer.infer_field_visibility("TestStruct");
        assert_eq!(visibility, FieldVisibility::Private);
    }
}
