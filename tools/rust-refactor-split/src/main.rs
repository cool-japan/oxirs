//! AST-based Rust file refactoring tool
//!
//! Splits large Rust files into modules while properly handling:
//! - Multiple impl blocks for the same type
//! - Doc comments and attributes
//! - Complex type hierarchies
//! - Proper module re-exports
//! - Large impl block splitting with method boundary detection
//! - Automatic use statement generation

mod method_analyzer;
mod import_analyzer;
mod scope_analyzer;

use anyhow::{Context, Result};
use clap::Parser;
use method_analyzer::{ImplBlockAnalyzer, MethodGroup};
use import_analyzer::ImportAnalyzer;
use scope_analyzer::ScopeAnalyzer;
use quote::ToTokens;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use syn::{File, Item, ItemImpl};

#[derive(Parser)]
#[command(name = "rust-refactor-split")]
#[command(about = "Split large Rust files into properly organized modules")]
struct Args {
    /// Input Rust file to split
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory for modules
    #[arg(short, long)]
    output: PathBuf,

    /// Maximum lines per module (default: 1000)
    #[arg(short, long, default_value = "1000")]
    max_lines: usize,

    /// Split large impl blocks (experimental)
    #[arg(long)]
    split_impl_blocks: bool,

    /// Maximum lines per impl block before splitting (default: 500)
    #[arg(long, default_value = "500")]
    max_impl_lines: usize,

    /// Dry run - show what would be done without making changes
    #[arg(short = 'n', long)]
    dry_run: bool,
}

#[derive(Clone)]
struct TypeInfo {
    name: String,
    item: Item,
    impls: Vec<Item>,
    doc_comments: Vec<String>,
    /// Large impl blocks that should be split
    large_impls: Vec<(ItemImpl, Vec<MethodGroup>)>,
}

struct FileAnalyzer {
    types: HashMap<String, TypeInfo>,
    standalone_items: Vec<Item>,
    split_impl_blocks: bool,
    max_impl_lines: usize,
    scope_analyzer: ScopeAnalyzer,
}

impl FileAnalyzer {
    fn new(split_impl_blocks: bool, max_impl_lines: usize) -> Self {
        Self {
            types: HashMap::new(),
            standalone_items: Vec::new(),
            split_impl_blocks,
            max_impl_lines,
            scope_analyzer: ScopeAnalyzer::new(),
        }
    }

    fn analyze(&mut self, file: &File) {
        // First pass: analyze all types with scope analyzer
        self.scope_analyzer.analyze_types(&file.items);

        for item in &file.items {
            match item {
                Item::Struct(s) => {
                    let name = s.ident.to_string();
                    self.types.insert(
                        name.clone(),
                        TypeInfo {
                            name,
                            item: item.clone(),
                            impls: Vec::new(),
                            doc_comments: Vec::new(),
                            large_impls: Vec::new(),
                        },
                    );
                }
                Item::Enum(e) => {
                    let name = e.ident.to_string();
                    self.types.insert(
                        name.clone(),
                        TypeInfo {
                            name,
                            item: item.clone(),
                            impls: Vec::new(),
                            doc_comments: Vec::new(),
                            large_impls: Vec::new(),
                        },
                    );
                }
                Item::Impl(i) => {
                    if let Some(type_name) = Self::get_impl_type_name(i) {
                        if let Some(type_info) = self.types.get_mut(&type_name) {
                            // Check if impl block is large and should be split
                            if self.split_impl_blocks {
                                // Analyze impl block to get accurate line count from methods
                                let mut analyzer = ImplBlockAnalyzer::new();
                                analyzer.analyze(i);
                                let impl_lines = analyzer.get_total_lines();

                                if impl_lines > self.max_impl_lines && analyzer.get_total_methods() > 1 {
                                    // Split this impl block
                                    let groups = analyzer.group_methods(self.max_impl_lines);

                                    if !groups.is_empty() {
                                        // Register each group as an impl block with scope analyzer
                                        for group in &groups {
                                            let module_name = format!("{}_{}", type_name.to_lowercase(), group.suggest_name());
                                            self.scope_analyzer.register_impl_block(
                                                type_name.clone(),
                                                i.clone(),
                                                module_name,
                                                group.methods.len(),
                                            );
                                        }
                                        // Mark this type as needing an impl module
                                        self.scope_analyzer.mark_needs_impl_module(&type_name);
                                        type_info.large_impls.push((i.clone(), groups));
                                    } else {
                                        type_info.impls.push(item.clone());
                                    }
                                } else {
                                    type_info.impls.push(item.clone());
                                }
                            } else {
                                type_info.impls.push(item.clone());
                            }
                        } else {
                            // Impl for unknown type - keep as standalone
                            self.standalone_items.push(item.clone());
                        }
                    } else {
                        self.standalone_items.push(item.clone());
                    }
                }
                _ => {
                    self.standalone_items.push(item.clone());
                }
            }
        }
    }

    fn get_impl_type_name(impl_item: &syn::ItemImpl) -> Option<String> {
        if let syn::Type::Path(type_path) = &*impl_item.self_ty {
            if let Some(segment) = type_path.path.segments.last() {
                return Some(segment.ident.to_string());
            }
        }
        None
    }

    /// Get recommended visibility for a type's fields based on impl organization
    fn get_field_visibility(&self, type_name: &str) -> scope_analyzer::FieldVisibility {
        self.scope_analyzer.infer_field_visibility(type_name)
    }

    /// Get organization strategy for a type's impl blocks
    fn get_organization_strategy(&self, type_name: &str) -> scope_analyzer::ImplOrganizationStrategy {
        self.scope_analyzer.determine_strategy(type_name)
    }

    fn group_by_module(&self, max_lines: usize) -> Vec<Module> {
        let mut modules = Vec::new();
        let mut module_name_counts: HashMap<String, usize> = HashMap::new();

        // Process types with large impl blocks separately
        for type_info in self.types.values() {
            if !type_info.large_impls.is_empty() {
                // Determine organization strategy for this type
                let _strategy = self.get_organization_strategy(&type_info.name);
                let _visibility = self.get_field_visibility(&type_info.name);
                // TODO: Use strategy and visibility in module generation

                // Create a module for this type with split impl blocks
                for (impl_block, method_groups) in &type_info.large_impls {
                    for (_idx, group) in method_groups.iter().enumerate() {
                        let base_name = if method_groups.len() == 1 {
                            format!("{}_impl", type_info.name.to_lowercase())
                        } else {
                            format!("{}_{}", type_info.name.to_lowercase(), group.suggest_name())
                        };

                        // Ensure unique module names
                        let module_name = if let Some(count) = module_name_counts.get(&base_name) {
                            let unique_name = format!("{}_{}", base_name, count + 1);
                            module_name_counts.insert(base_name.clone(), count + 1);
                            unique_name
                        } else {
                            module_name_counts.insert(base_name.clone(), 0);
                            base_name
                        };

                        let mut module = Module::new(module_name);
                        module.impl_type_name = Some(type_info.name.clone());
                        module.impl_self_ty = Some(impl_block.self_ty.clone());
                        module.method_group = Some(group.clone());
                        modules.push(module);
                    }
                }

                // Create main module for the type definition
                let mut type_module = Module::new(format!("{}_type", type_info.name.to_lowercase()));
                type_module.field_visibility = Some(_visibility.clone());
                type_module.types.push(TypeInfo {
                    name: type_info.name.clone(),
                    item: type_info.item.clone(),
                    impls: type_info.impls.clone(),
                    doc_comments: type_info.doc_comments.clone(),
                    large_impls: vec![],
                });
                modules.push(type_module);
            }
        }

        // Process regular types
        let mut current_module = Module::new("types".to_string());
        let mut current_lines = 0;

        let regular_types: Vec<_> = self.types.values()
            .filter(|t| t.large_impls.is_empty())
            .collect();

        for type_info in regular_types {
            let type_lines = type_info.estimate_lines();

            if current_lines + type_lines > max_lines && !current_module.types.is_empty() {
                modules.push(current_module);
                current_module = Module::new(format!("types_{}", modules.len() + 1));
                current_lines = 0;
            }

            current_module.types.push(type_info.clone());
            current_lines += type_lines;
        }

        if !current_module.types.is_empty() {
            modules.push(current_module);
        }

        // Add standalone items to a separate module
        if !self.standalone_items.is_empty() {
            let mut standalone_module = Module::new("functions".to_string());
            standalone_module.standalone_items = self.standalone_items.clone();
            modules.push(standalone_module);
        }

        modules
    }
}

#[derive(Clone)]
struct Module {
    name: String,
    types: Vec<TypeInfo>,
    standalone_items: Vec<Item>,
    /// Type name for impl block splitting
    impl_type_name: Option<String>,
    /// Self type for impl block
    impl_self_ty: Option<Box<syn::Type>>,
    /// Method group for split impl blocks
    method_group: Option<MethodGroup>,
    /// Recommended field visibility for types in this module
    field_visibility: Option<scope_analyzer::FieldVisibility>,
}

impl Module {
    fn new(name: String) -> Self {
        Self {
            name,
            types: Vec::new(),
            standalone_items: Vec::new(),
            impl_type_name: None,
            impl_self_ty: None,
            method_group: None,
            field_visibility: None,
        }
    }

    fn generate_content(&self, original_file: &File) -> String {
        let mut content = String::new();

        // Extract and preserve module-level attributes and comments
        content.push_str("//! Auto-generated module\n\n");

        // Generate use statements using ImportAnalyzer
        let mut import_analyzer = ImportAnalyzer::new();
        import_analyzer.analyze_file(original_file);

        // For impl block modules, generate context-aware imports
        if let Some(type_name) = &self.impl_type_name {
            // Import std collections (always useful for impl blocks)
            content.push_str("use std::collections::{HashMap, HashSet};\n");

            // Import the type from its type module
            // Type modules are named as {type_name}_type
            let type_module_name = format!("{}_type", type_name.to_lowercase());
            content.push_str(&format!("use super::{}::{};\n", type_module_name, type_name));
            content.push('\n');
        }

        // Generate impl block from method group if this is a split impl module
        if let Some(method_group) = &self.method_group {
            if let Some(type_name) = &self.impl_type_name {
                // Build a complete impl block using syn
                let mut impl_items = Vec::new();
                for method in &method_group.methods {
                    impl_items.push(syn::ImplItem::Fn(method.item.clone()));
                }

                let impl_block = syn::ItemImpl {
                    attrs: Vec::new(),
                    defaultness: None,
                    unsafety: None,
                    impl_token: Default::default(),
                    generics: Default::default(),
                    trait_: None,
                    self_ty: self.impl_self_ty.clone().unwrap_or_else(|| {
                        Box::new(syn::parse_str::<syn::Type>(type_name).unwrap())
                    }),
                    brace_token: Default::default(),
                    items: impl_items,
                };

                // Use prettyplease to format
                let formatted = prettyplease::unparse(&syn::File {
                    shebang: None,
                    attrs: Vec::new(),
                    items: vec![syn::Item::Impl(impl_block)],
                });

                content.push_str(&formatted);
                return content;
            }
        }

        // Generate content for regular type modules

        // First, collect all types used in this module
        let mut types_used = std::collections::HashSet::new();
        for type_info in &self.types {
            // Extract types from struct/enum fields
            if let Item::Struct(s) = &type_info.item {
                for field in &s.fields {
                    extract_type_names(&field.ty, &mut types_used);
                }
            } else if let Item::Enum(e) = &type_info.item {
                for variant in &e.variants {
                    for field in &variant.fields {
                        extract_type_names(&field.ty, &mut types_used);
                    }
                }
            }
        }

        // Generate imports for types used
        if !types_used.is_empty() {
            let needs_collections = types_used.iter().any(|t|
                t == "HashMap" || t == "HashSet" || t == "BTreeMap" ||
                t == "BTreeSet" || t == "VecDeque"
            );

            if needs_collections {
                let collection_types: Vec<_> = types_used.iter()
                    .filter(|t| ["HashMap", "HashSet", "BTreeMap", "BTreeSet", "VecDeque"].contains(&t.as_str()))
                    .cloned()
                    .collect();
                if !collection_types.is_empty() {
                    content.push_str(&format!("use std::collections::{{{}}};\n", collection_types.join(", ")));
                }
            }
            content.push('\n');
        }

        let mut items = Vec::new();

        for type_info in &self.types {
            // Apply field visibility based on self.field_visibility
            let item = if let Some(ref vis) = self.field_visibility {
                apply_field_visibility(type_info.item.clone(), vis)
            } else {
                type_info.item.clone()
            };
            items.push(item);
            items.extend(type_info.impls.clone());
        }

        items.extend(self.standalone_items.clone());

        if !items.is_empty() {
            let formatted = prettyplease::unparse(&syn::File {
                shebang: None,
                attrs: Vec::new(),
                items,
            });
            content.push_str(&formatted);
        }

        content
    }
}

impl TypeInfo {
    fn estimate_lines(&self) -> usize {
        let item_lines = self.item.to_token_stream().to_string().lines().count();
        let impl_lines: usize = self
            .impls
            .iter()
            .map(|i| i.to_token_stream().to_string().lines().count())
            .sum();
        item_lines + impl_lines
    }
}

/// Extract type names from a syn::Type
fn extract_type_names(ty: &syn::Type, types: &mut HashSet<String>) {
    match ty {
        syn::Type::Path(type_path) => {
            if let Some(segment) = type_path.path.segments.last() {
                let type_name = segment.ident.to_string();
                // Add the main type
                types.insert(type_name);

                // Check for generic arguments
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    for arg in &args.args {
                        if let syn::GenericArgument::Type(inner_ty) = arg {
                            extract_type_names(inner_ty, types);
                        }
                    }
                }
            }
        }
        syn::Type::Reference(type_ref) => {
            extract_type_names(&type_ref.elem, types);
        }
        syn::Type::Slice(type_slice) => {
            extract_type_names(&type_slice.elem, types);
        }
        syn::Type::Array(type_array) => {
            extract_type_names(&type_array.elem, types);
        }
        syn::Type::Ptr(type_ptr) => {
            extract_type_names(&type_ptr.elem, types);
        }
        syn::Type::Tuple(type_tuple) => {
            for elem in &type_tuple.elems {
                extract_type_names(elem, types);
            }
        }
        _ => {}
    }
}

/// Apply field visibility modifications to a struct or enum
fn apply_field_visibility(item: Item, visibility: &scope_analyzer::FieldVisibility) -> Item {
    match item {
        Item::Struct(mut s) => {
            // Only modify if we need pub(super) or other non-default visibility
            match visibility {
                scope_analyzer::FieldVisibility::PubSuper => {
                    // Modify each field to have pub(super) visibility
                    for field in &mut s.fields {
                        if matches!(field.vis, syn::Visibility::Inherited) {
                            // Field is currently private, make it pub(super)
                            field.vis = syn::parse_quote!(pub(super));
                        }
                    }
                }
                scope_analyzer::FieldVisibility::PubCrate => {
                    for field in &mut s.fields {
                        if matches!(field.vis, syn::Visibility::Inherited) {
                            field.vis = syn::parse_quote!(pub(crate));
                        }
                    }
                }
                scope_analyzer::FieldVisibility::Pub => {
                    for field in &mut s.fields {
                        if matches!(field.vis, syn::Visibility::Inherited) {
                            field.vis = syn::parse_quote!(pub);
                        }
                    }
                }
                scope_analyzer::FieldVisibility::Private => {
                    // Keep fields private (no change)
                }
            }
            Item::Struct(s)
        }
        Item::Enum(mut e) => {
            // Apply visibility to enum variant fields
            match visibility {
                scope_analyzer::FieldVisibility::PubSuper => {
                    for variant in &mut e.variants {
                        for field in &mut variant.fields {
                            if matches!(field.vis, syn::Visibility::Inherited) {
                                field.vis = syn::parse_quote!(pub(super));
                            }
                        }
                    }
                }
                scope_analyzer::FieldVisibility::PubCrate => {
                    for variant in &mut e.variants {
                        for field in &mut variant.fields {
                            if matches!(field.vis, syn::Visibility::Inherited) {
                                field.vis = syn::parse_quote!(pub(crate));
                            }
                        }
                    }
                }
                scope_analyzer::FieldVisibility::Pub => {
                    for variant in &mut e.variants {
                        for field in &mut variant.fields {
                            if matches!(field.vis, syn::Visibility::Inherited) {
                                field.vis = syn::parse_quote!(pub);
                            }
                        }
                    }
                }
                scope_analyzer::FieldVisibility::Private => {
                    // Keep fields private
                }
            }
            Item::Enum(e)
        }
        other => other, // Return unchanged for non-struct/enum items
    }
}

fn generate_mod_rs(modules: &[Module], _output_dir: &Path) -> Result<String> {
    let mut content = String::from("//! Auto-generated module structure\n\n");

    for module in modules {
        content.push_str(&format!("pub mod {};\n", module.name));
    }

    content.push_str("\n// Re-export all types\n");
    for module in modules {
        content.push_str(&format!("pub use {}::*;\n", module.name));
    }

    Ok(content)
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Read and parse the input file
    let source_code = fs::read_to_string(&args.input)
        .context(format!("Failed to read input file: {:?}", args.input))?;

    let syntax_tree: File =
        syn::parse_file(&source_code).context("Failed to parse Rust source code")?;

    println!("Analyzing file: {:?}", args.input);
    println!("Total items: {}", syntax_tree.items.len());
    if args.split_impl_blocks {
        println!("Impl block splitting enabled (max {} lines per impl)", args.max_impl_lines);
    }

    // Analyze the file
    let mut analyzer = FileAnalyzer::new(args.split_impl_blocks, args.max_impl_lines);
    analyzer.analyze(&syntax_tree);

    println!("Found {} types", analyzer.types.len());
    println!("Found {} standalone items", analyzer.standalone_items.len());

    // Group into modules
    let modules = analyzer.group_by_module(args.max_lines);
    println!("Generated {} modules", modules.len());

    if args.dry_run {
        println!("\nDry run - would create:");
        for module in &modules {
            println!(
                "  - {}.rs ({} types, {} items)",
                module.name,
                module.types.len(),
                module.standalone_items.len()
            );
        }
        return Ok(());
    }

    // Create output directory
    fs::create_dir_all(&args.output)?;

    // Write module files
    for module in &modules {
        let module_path = args.output.join(format!("{}.rs", module.name));
        let content = module.generate_content(&syntax_tree);
        fs::write(&module_path, content)
            .context(format!("Failed to write module: {:?}", module_path))?;
        println!("Created: {:?}", module_path);
    }

    // Write mod.rs
    let mod_content = generate_mod_rs(&modules, &args.output)?;
    let mod_path = args.output.join("mod.rs");
    fs::write(&mod_path, mod_content).context("Failed to write mod.rs")?;
    println!("Created: {:?}", mod_path);

    println!("\nRefactoring complete!");
    println!(
        "Original file: {} lines",
        source_code.lines().count()
    );
    println!("Generated {} module files", modules.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impl_type_extraction() {
        let code = r#"
            struct Foo;
            impl Foo {
                fn bar() {}
            }
        "#;

        let file = syn::parse_file(code).unwrap();
        let mut analyzer = FileAnalyzer::new(false, 500);
        analyzer.analyze(&file);

        assert_eq!(analyzer.types.len(), 1);
        assert_eq!(analyzer.types.get("Foo").unwrap().impls.len(), 1);
    }
}
