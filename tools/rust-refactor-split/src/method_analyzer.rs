//! Method boundary detection and analysis for splitting large impl blocks

use std::collections::{HashMap, HashSet};
use syn::{visit::Visit, Expr, ExprCall, ExprMethodCall, ImplItem, ImplItemFn, ItemImpl};

/// Information about a method within an impl block
#[derive(Clone)]
pub struct MethodInfo {
    pub name: String,
    pub item: ImplItemFn,
    pub calls_methods: HashSet<String>,
    pub line_count: usize,
}

/// Analyzer for impl blocks to detect method boundaries and dependencies
pub struct ImplBlockAnalyzer {
    methods: Vec<MethodInfo>,
}

impl ImplBlockAnalyzer {
    pub fn new() -> Self {
        Self {
            methods: Vec::new(),
        }
    }

    /// Analyze an impl block and extract method information
    pub fn analyze(&mut self, impl_item: &ItemImpl) {
        for item in &impl_item.items {
            if let ImplItem::Fn(method) = item {
                let method_info = self.analyze_method(method);
                self.methods.push(method_info);
            }
        }
    }

    fn analyze_method(&self, method: &ImplItemFn) -> MethodInfo {
        let name = method.sig.ident.to_string();
        let mut visitor = MethodCallVisitor::new();
        visitor.visit_impl_item_fn(method);

        // Use heuristic for line count since token stream loses formatting
        // Average Rust method is 25-35 lines; use token stream as base and multiply
        let token_lines = quote::ToTokens::to_token_stream(method)
            .to_string()
            .lines()
            .count();

        // Heuristic: multiply by 15 to approximate real formatting
        // A 2-line token stream method is typically ~30 lines in real code
        let line_count = token_lines.max(1) * 15;

        MethodInfo {
            name,
            item: method.clone(),
            calls_methods: visitor.called_methods,
            line_count,
        }
    }

    /// Group methods into clusters based on dependencies
    pub fn group_methods(&self, max_lines_per_group: usize) -> Vec<MethodGroup> {
        // Build dependency graph
        let dep_graph = self.build_dependency_graph();

        // Find strongly connected components (method clusters)
        let clusters = self.find_clusters(&dep_graph);

        // Group clusters into modules respecting size limits
        self.create_groups(clusters, max_lines_per_group)
    }

    fn build_dependency_graph(&self) -> HashMap<String, HashSet<String>> {
        let mut graph = HashMap::new();

        for method in &self.methods {
            graph.insert(method.name.clone(), method.calls_methods.clone());
        }

        graph
    }

    fn find_clusters(&self, _graph: &HashMap<String, HashSet<String>>) -> Vec<Vec<String>> {
        // Simple clustering: group methods that call each other
        let mut clusters: Vec<Vec<String>> = Vec::new();
        let mut assigned: HashSet<String> = HashSet::new();

        for method in &self.methods {
            if assigned.contains(&method.name) {
                continue;
            }

            let mut cluster = vec![method.name.clone()];
            assigned.insert(method.name.clone());

            // Find methods that this method calls or that call this method
            for other_method in &self.methods {
                if assigned.contains(&other_method.name) {
                    continue;
                }

                let calls_other = method.calls_methods.contains(&other_method.name);
                let called_by_other = other_method.calls_methods.contains(&method.name);

                if calls_other || called_by_other {
                    cluster.push(other_method.name.clone());
                    assigned.insert(other_method.name.clone());
                }
            }

            clusters.push(cluster);
        }

        clusters
    }

    fn create_groups(&self, clusters: Vec<Vec<String>>, max_lines: usize) -> Vec<MethodGroup> {
        let mut groups = Vec::new();
        let method_map: HashMap<String, &MethodInfo> = self
            .methods
            .iter()
            .map(|m| (m.name.clone(), m))
            .collect();

        for cluster in clusters {
            let mut current_group = MethodGroup::new();
            let mut current_lines = 0;

            for method_name in &cluster {
                if let Some(method) = method_map.get(method_name) {
                    if current_lines + method.line_count > max_lines && !current_group.methods.is_empty() {
                        groups.push(current_group);
                        current_group = MethodGroup::new();
                        current_lines = 0;
                    }

                    current_group.methods.push((*method).clone());
                    current_lines += method.line_count;
                }
            }

            if !current_group.methods.is_empty() {
                groups.push(current_group);
            }
        }

        groups
    }

    pub fn get_total_methods(&self) -> usize {
        self.methods.len()
    }

    pub fn get_total_lines(&self) -> usize {
        self.methods.iter().map(|m| m.line_count).sum()
    }
}

/// A group of related methods
#[derive(Clone)]
pub struct MethodGroup {
    pub methods: Vec<MethodInfo>,
}

impl MethodGroup {
    fn new() -> Self {
        Self {
            methods: Vec::new(),
        }
    }

    pub fn total_lines(&self) -> usize {
        self.methods.iter().map(|m| m.line_count).sum()
    }

    pub fn suggest_name(&self) -> String {
        if self.methods.is_empty() {
            return "methods".to_string();
        }

        // Try to find a common prefix or theme
        let first_method = &self.methods[0].name;

        // Common patterns
        if first_method.starts_with("test_") || first_method.starts_with("check_") {
            return first_method.split('_').next().unwrap_or("methods").to_string() + "_methods";
        }

        if first_method.starts_with("get_") || first_method.starts_with("set_") {
            return "accessors".to_string();
        }

        if first_method.starts_with("handle_") || first_method.starts_with("process_") {
            return "handlers".to_string();
        }

        // Fallback: use first method name
        format!("{}_group", first_method)
    }
}

/// Visitor to find method calls within a method body
struct MethodCallVisitor {
    called_methods: HashSet<String>,
}

impl MethodCallVisitor {
    fn new() -> Self {
        Self {
            called_methods: HashSet::new(),
        }
    }
}

impl<'ast> Visit<'ast> for MethodCallVisitor {
    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        self.called_methods.insert(node.method.to_string());
        syn::visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        // Try to extract function name from call expression
        if let Expr::Path(path) = &*node.func {
            if let Some(segment) = path.path.segments.last() {
                self.called_methods.insert(segment.ident.to_string());
            }
        }
        syn::visit::visit_expr_call(self, node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_method_analysis() {
        let impl_block: ItemImpl = parse_quote! {
            impl MyStruct {
                fn foo(&self) {
                    self.bar();
                }

                fn bar(&self) {
                    println!("bar");
                }

                fn baz(&self) {
                    self.foo();
                }
            }
        };

        let mut analyzer = ImplBlockAnalyzer::new();
        analyzer.analyze(&impl_block);

        assert_eq!(analyzer.get_total_methods(), 3);
        assert!(analyzer.methods.iter().any(|m| m.name == "foo"));
        assert!(analyzer.methods.iter().any(|m| m.name == "bar"));
        assert!(analyzer.methods.iter().any(|m| m.name == "baz"));
    }

    #[test]
    fn test_method_grouping() {
        let impl_block: ItemImpl = parse_quote! {
            impl MyStruct {
                fn foo(&self) {
                    self.bar();
                }

                fn bar(&self) {
                    println!("bar");
                }

                fn unrelated(&self) {
                    println!("unrelated");
                }
            }
        };

        let mut analyzer = ImplBlockAnalyzer::new();
        analyzer.analyze(&impl_block);

        let groups = analyzer.group_methods(1000);
        assert!(!groups.is_empty());
    }
}
