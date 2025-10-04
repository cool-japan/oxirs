//! Import statement analysis and generation for refactored modules

use std::collections::{HashMap, HashSet};
use syn::{
    visit::Visit, Expr, GenericArgument, ImplItemFn, Item, PathArguments, Stmt, Type, TypePath,
};

/// Tracks type usage and generates appropriate use statements
pub struct ImportAnalyzer {
    /// Types referenced in methods (type name -> potential paths)
    used_types: HashMap<String, HashSet<String>>,

    /// Known type mappings from original file
    type_mappings: HashMap<String, String>,

    /// Standard library types that don't need explicit imports
    std_types: HashSet<String>,
}

impl ImportAnalyzer {
    pub fn new() -> Self {
        let mut std_types = HashSet::new();

        // Common std types
        std_types.insert("String".to_string());
        std_types.insert("Vec".to_string());
        std_types.insert("Option".to_string());
        std_types.insert("Result".to_string());
        std_types.insert("Box".to_string());
        std_types.insert("Arc".to_string());
        std_types.insert("Rc".to_string());
        std_types.insert("HashMap".to_string());
        std_types.insert("HashSet".to_string());
        std_types.insert("BTreeMap".to_string());
        std_types.insert("BTreeSet".to_string());
        std_types.insert("VecDeque".to_string());

        Self {
            used_types: HashMap::new(),
            type_mappings: HashMap::new(),
            std_types,
        }
    }

    /// Analyze a file to build type mappings
    pub fn analyze_file(&mut self, file: &syn::File) {
        for item in &file.items {
            match item {
                Item::Use(use_item) => {
                    self.extract_use_mapping(use_item);
                }
                Item::Struct(s) => {
                    self.type_mappings.insert(s.ident.to_string(), format!("super::types::{}", s.ident));
                }
                Item::Enum(e) => {
                    self.type_mappings.insert(e.ident.to_string(), format!("super::types::{}", e.ident));
                }
                Item::Type(t) => {
                    // Type alias
                    self.type_mappings.insert(t.ident.to_string(), format!("super::types::{}", t.ident));
                }
                _ => {}
            }
        }
    }

    fn extract_use_mapping(&mut self, use_item: &syn::ItemUse) {
        // Extract use statement to build mappings
        // This is simplified - full implementation would parse the use tree
        let use_str = quote::quote!(#use_item).to_string();

        // Extract simple patterns like "use foo::Bar"
        if let Some(last_segment) = use_str.split("::").last() {
            let type_name = last_segment.trim_end_matches(';').trim();
            if !type_name.is_empty() && type_name.chars().next().unwrap().is_uppercase() {
                self.type_mappings.insert(type_name.to_string(), use_str.replace("use ", "").trim_end_matches(';').trim().to_string());
            }
        }
    }

    /// Analyze methods to find used types
    pub fn analyze_methods(&mut self, methods: &[&ImplItemFn]) {
        for method in methods {
            let mut visitor = TypeVisitor::new();
            visitor.visit_impl_item_fn(method);

            for type_name in visitor.types_used {
                self.used_types
                    .entry(type_name.clone())
                    .or_insert_with(HashSet::new)
                    .insert("unknown".to_string());
            }
        }
    }

    /// Generate use statements for a module
    pub fn generate_use_statements(&self, types_needed: &[String]) -> Vec<String> {
        let mut use_statements = HashSet::new();
        let mut std_collections = HashSet::new();
        let mut crate_imports = HashSet::new();
        let mut super_imports = HashSet::new();

        for type_name in types_needed {
            // Skip primitive types
            if self.is_primitive(type_name) {
                continue;
            }

            // Check if it's a std type
            if self.std_types.contains(type_name) {
                if type_name == "HashMap" || type_name == "HashSet" || type_name == "VecDeque" || type_name == "BTreeMap" || type_name == "BTreeSet" {
                    std_collections.insert(type_name.clone());
                }
                continue;
            }

            // Check if we have a mapping
            if let Some(path) = self.type_mappings.get(type_name) {
                if path.starts_with("super::") {
                    super_imports.insert(path.clone());
                } else if path.starts_with("crate::") {
                    crate_imports.insert(path.clone());
                } else {
                    use_statements.insert(path.clone());
                }
            }
        }

        let mut result = Vec::new();

        // Add std::collections if needed
        if !std_collections.is_empty() {
            let collections: Vec<_> = std_collections.into_iter().collect();
            result.push(format!("use std::collections::{{{}}};", collections.join(", ")));
        }

        // Add super imports
        if !super_imports.is_empty() {
            for import in super_imports {
                result.push(format!("use {};", import));
            }
        }

        // Add crate imports
        if !crate_imports.is_empty() {
            for import in crate_imports {
                result.push(format!("use {};", import));
            }
        }

        // Add other use statements
        for stmt in use_statements {
            result.push(format!("use {};", stmt));
        }

        result.sort();
        result
    }

    fn is_primitive(&self, type_name: &str) -> bool {
        matches!(
            type_name,
            "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64"
                | "u128" | "usize" | "f32" | "f64" | "bool" | "char" | "str" | "()"
        )
    }

    /// Infer common imports for impl blocks
    pub fn infer_common_imports(&self) -> Vec<String> {
        self.infer_imports_with_depth(1)
    }

    /// Infer imports with specified module depth (number of super:: needed)
    pub fn infer_imports_with_depth(&self, depth: usize) -> Vec<String> {
        let super_prefix = "super::".repeat(depth);
        vec![
            "use std::collections::{HashMap, HashSet};".to_string(),
            format!("use {}types::*;", super_prefix),
            format!("use {}PropertyPathEvaluator;", super_prefix),
        ]
    }
}

/// Visitor to collect type references in methods
struct TypeVisitor {
    types_used: HashSet<String>,
}

impl TypeVisitor {
    fn new() -> Self {
        Self {
            types_used: HashSet::new(),
        }
    }

    fn extract_type_name(&mut self, ty: &Type) {
        match ty {
            Type::Path(TypePath { path, .. }) => {
                if let Some(segment) = path.segments.last() {
                    self.types_used.insert(segment.ident.to_string());

                    // Also check generic arguments
                    if let PathArguments::AngleBracketed(args) = &segment.arguments {
                        for arg in &args.args {
                            if let GenericArgument::Type(inner_ty) = arg {
                                self.extract_type_name(inner_ty);
                            }
                        }
                    }
                }
            }
            Type::Reference(r) => {
                self.extract_type_name(&r.elem);
            }
            Type::Tuple(t) => {
                for elem in &t.elems {
                    self.extract_type_name(elem);
                }
            }
            _ => {}
        }
    }
}

impl<'ast> Visit<'ast> for TypeVisitor {
    fn visit_type(&mut self, ty: &'ast Type) {
        self.extract_type_name(ty);
        syn::visit::visit_type(self, ty);
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        // Extract types from expressions (like method calls)
        match expr {
            Expr::MethodCall(method_call) => {
                // Track method receiver type if possible
                syn::visit::visit_expr(self, &method_call.receiver);
            }
            Expr::Path(path) => {
                if let Some(segment) = path.path.segments.last() {
                    // Might be a type name (like enum variant)
                    let name = segment.ident.to_string();
                    if name.chars().next().unwrap().is_uppercase() {
                        self.types_used.insert(name);
                    }
                }
            }
            _ => {}
        }
        syn::visit::visit_expr(self, expr);
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        // Extract types from let statements
        if let Stmt::Local(local) = stmt {
            if let Some(init) = &local.init {
                syn::visit::visit_expr(self, &init.expr);
            }
        }
        syn::visit::visit_stmt(self, stmt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_analyzer_std_types() {
        let analyzer = ImportAnalyzer::new();
        assert!(analyzer.std_types.contains("String"));
        assert!(analyzer.std_types.contains("HashMap"));
    }

    #[test]
    fn test_primitive_detection() {
        let analyzer = ImportAnalyzer::new();
        assert!(analyzer.is_primitive("i32"));
        assert!(analyzer.is_primitive("bool"));
        assert!(!analyzer.is_primitive("String"));
    }

    #[test]
    fn test_generate_use_statements() {
        let analyzer = ImportAnalyzer::new();
        let types = vec!["i32".to_string(), "String".to_string()];
        let statements = analyzer.generate_use_statements(&types);

        // Should not generate use statements for primitives and std types
        assert!(statements.is_empty() || statements.iter().all(|s| !s.contains("i32")));
    }
}
