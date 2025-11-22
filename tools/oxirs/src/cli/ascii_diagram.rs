//! ASCII Art Diagram Generator for RDF Graphs
//!
//! Provides visual representation of RDF triples using ASCII art for terminal display.
//! Supports different layout styles and handles large graphs with intelligent summarization.

use std::collections::{HashMap, HashSet};
use std::io::Write;

use crate::cli::error::CliResult as Result;

/// Represents an RDF triple for diagram generation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DiagramTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Layout style for ASCII diagrams
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutStyle {
    /// Tree layout with hierarchical structure
    Tree,
    /// Graph layout with connections
    Graph,
    /// Compact layout for dense graphs
    Compact,
    /// List layout showing triples as a list
    List,
}

/// Configuration for ASCII diagram generation
#[derive(Debug, Clone)]
pub struct DiagramConfig {
    /// Layout style
    pub style: LayoutStyle,
    /// Maximum width of the diagram
    pub max_width: usize,
    /// Maximum nodes to display (0 = unlimited)
    pub max_nodes: usize,
    /// Maximum edges to display (0 = unlimited)
    pub max_edges: usize,
    /// Whether to show full URIs or abbreviated
    pub abbreviate_uris: bool,
    /// Whether to use Unicode box drawing characters
    pub use_unicode: bool,
}

impl Default for DiagramConfig {
    fn default() -> Self {
        Self {
            style: LayoutStyle::Tree,
            max_width: 120,
            max_nodes: 50,
            max_edges: 100,
            abbreviate_uris: true,
            use_unicode: true,
        }
    }
}

/// ASCII diagram generator
pub struct AsciiDiagramGenerator {
    config: DiagramConfig,
}

impl AsciiDiagramGenerator {
    /// Create a new ASCII diagram generator
    pub fn new(config: DiagramConfig) -> Self {
        Self { config }
    }

    /// Generate ASCII diagram from triples
    pub fn generate(&self, triples: &[DiagramTriple], writer: &mut dyn Write) -> Result<()> {
        if triples.is_empty() {
            writeln!(writer, "(no triples to display)")?;
            return Ok(());
        }

        match self.config.style {
            LayoutStyle::Tree => self.generate_tree(triples, writer),
            LayoutStyle::Graph => self.generate_graph(triples, writer),
            LayoutStyle::Compact => self.generate_compact(triples, writer),
            LayoutStyle::List => self.generate_list(triples, writer),
        }
    }

    /// Generate tree layout
    fn generate_tree(&self, triples: &[DiagramTriple], writer: &mut dyn Write) -> Result<()> {
        // Build node hierarchy
        let mut subjects: HashSet<String> = HashSet::new();
        let mut objects: HashSet<String> = HashSet::new();
        let mut edges: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for triple in triples.iter().take(self.config.max_edges) {
            subjects.insert(triple.subject.clone());
            objects.insert(triple.object.clone());
            edges
                .entry(triple.subject.clone())
                .or_default()
                .push((triple.predicate.clone(), triple.object.clone()));
        }

        // Find root nodes (subjects that are not objects)
        let roots: Vec<String> = subjects
            .iter()
            .filter(|s| !objects.contains(*s))
            .cloned()
            .collect();

        if roots.is_empty() {
            // No clear roots, use first subject
            if let Some(triple) = triples.first() {
                self.render_tree_node(
                    &triple.subject,
                    &edges,
                    writer,
                    "",
                    true,
                    &mut HashSet::new(),
                )?;
            }
        } else {
            // Render each root
            for (idx, root) in roots.iter().enumerate() {
                let is_last = idx == roots.len() - 1;
                self.render_tree_node(root, &edges, writer, "", is_last, &mut HashSet::new())?;
            }
        }

        Ok(())
    }

    /// Render a tree node recursively
    fn render_tree_node(
        &self,
        node: &str,
        edges: &HashMap<String, Vec<(String, String)>>,
        writer: &mut dyn Write,
        prefix: &str,
        is_last: bool,
        visited: &mut HashSet<String>,
    ) -> Result<()> {
        // Check for cycles
        if visited.contains(node) {
            writeln!(
                writer,
                "{}{}",
                prefix,
                self.format_node(&format!("{} (cyclic reference)", node))
            )?;
            return Ok(());
        }
        visited.insert(node.to_string());

        // Draw current node
        let (branch, continuation) = if self.config.use_unicode {
            if is_last {
                ("└── ", "    ")
            } else {
                ("├── ", "│   ")
            }
        } else if is_last {
            ("`-- ", "    ")
        } else {
            ("|-- ", "|   ")
        };

        writeln!(writer, "{}{}{}", prefix, branch, self.format_node(node))?;

        // Draw edges to children
        if let Some(children) = edges.get(node) {
            for (idx, (predicate, object)) in children.iter().enumerate() {
                let is_last_child = idx == children.len() - 1;
                let child_prefix = format!("{}{}", prefix, continuation);

                // Draw predicate
                let pred_branch = if self.config.use_unicode {
                    if is_last_child {
                        "└─["
                    } else {
                        "├─["
                    }
                } else if is_last_child {
                    "`-["
                } else {
                    "|-["
                };

                writeln!(
                    writer,
                    "{}{}{}]",
                    child_prefix,
                    pred_branch,
                    self.format_predicate(predicate)
                )?;

                // Draw object
                let obj_prefix = format!(
                    "{}{}",
                    child_prefix,
                    if is_last_child {
                        "  "
                    } else if self.config.use_unicode {
                        "│ "
                    } else {
                        "| "
                    }
                );
                self.render_tree_node(object, edges, writer, &obj_prefix, true, visited)?;
            }
        }

        visited.remove(node);
        Ok(())
    }

    /// Generate graph layout
    fn generate_graph(&self, triples: &[DiagramTriple], writer: &mut dyn Write) -> Result<()> {
        writeln!(writer, "RDF Graph (ASCII representation):")?;
        writeln!(writer)?;

        let connector = if self.config.use_unicode { "─" } else { "-" };
        let arrow = if self.config.use_unicode { "→" } else { "->" };

        for (idx, triple) in triples.iter().enumerate() {
            if self.config.max_edges > 0 && idx >= self.config.max_edges {
                writeln!(writer, "... ({} more triples)", triples.len() - idx)?;
                break;
            }

            let subject = self.format_node(&triple.subject);
            let predicate = self.format_predicate(&triple.predicate);
            let object = self.format_node(&triple.object);

            // Calculate available width
            let total_len = subject.len() + predicate.len() + object.len() + 10;
            let connector_count = if total_len < self.config.max_width {
                (self.config.max_width - total_len) / 2
            } else {
                2
            };

            writeln!(
                writer,
                "{}  {}{}{}  {}  {}",
                subject,
                connector.repeat(connector_count),
                predicate,
                connector.repeat(connector_count),
                arrow,
                object
            )?;
        }

        Ok(())
    }

    /// Generate compact layout
    fn generate_compact(&self, triples: &[DiagramTriple], writer: &mut dyn Write) -> Result<()> {
        writeln!(writer, "RDF Triples (Compact):")?;
        writeln!(writer)?;

        // Group by subject
        let mut grouped: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for triple in triples {
            grouped
                .entry(triple.subject.clone())
                .or_default()
                .push((triple.predicate.clone(), triple.object.clone()));
        }

        for (idx, (subject, predicates)) in grouped.iter().enumerate() {
            if self.config.max_nodes > 0 && idx >= self.config.max_nodes {
                writeln!(writer, "... ({} more subjects)", grouped.len() - idx)?;
                break;
            }

            writeln!(writer, "{}", self.format_node(subject))?;
            for (pred_idx, (predicate, object)) in predicates.iter().enumerate() {
                let prefix = if pred_idx == predicates.len() - 1 {
                    if self.config.use_unicode {
                        "  └─"
                    } else {
                        "  `-"
                    }
                } else if self.config.use_unicode {
                    "  ├─"
                } else {
                    "  |-"
                };

                writeln!(
                    writer,
                    "{} {} {}",
                    prefix,
                    self.format_predicate(predicate),
                    self.format_node(object)
                )?;
            }
            writeln!(writer)?;
        }

        Ok(())
    }

    /// Generate list layout
    fn generate_list(&self, triples: &[DiagramTriple], writer: &mut dyn Write) -> Result<()> {
        writeln!(writer, "RDF Triples:")?;
        writeln!(writer)?;

        for (idx, triple) in triples.iter().enumerate() {
            if self.config.max_edges > 0 && idx >= self.config.max_edges {
                writeln!(writer, "... ({} more triples)", triples.len() - idx)?;
                break;
            }

            writeln!(
                writer,
                "{:4}. {} {} {}",
                idx + 1,
                self.format_node(&triple.subject),
                self.format_predicate(&triple.predicate),
                self.format_node(&triple.object)
            )?;
        }

        Ok(())
    }

    /// Format a node (subject or object) for display
    fn format_node(&self, node: &str) -> String {
        if self.config.abbreviate_uris {
            self.abbreviate_uri(node)
        } else {
            node.to_string()
        }
    }

    /// Format a predicate for display
    fn format_predicate(&self, predicate: &str) -> String {
        if self.config.abbreviate_uris {
            self.abbreviate_uri(predicate)
        } else {
            predicate.to_string()
        }
    }

    /// Abbreviate a URI by extracting the local name
    fn abbreviate_uri(&self, uri: &str) -> String {
        // Remove angle brackets if present
        let uri = uri.trim_start_matches('<').trim_end_matches('>');

        // Check for common prefixes
        let common_prefixes = [
            ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:"),
            ("http://www.w3.org/2000/01/rdf-schema#", "rdfs:"),
            ("http://www.w3.org/2002/07/owl#", "owl:"),
            ("http://www.w3.org/ns/shacl#", "sh:"),
            ("http://xmlns.com/foaf/0.1/", "foaf:"),
            ("http://purl.org/dc/elements/1.1/", "dc:"),
            ("http://purl.org/dc/terms/", "dct:"),
            ("http://schema.org/", "schema:"),
        ];

        for (prefix, abbrev) in &common_prefixes {
            if let Some(stripped) = uri.strip_prefix(prefix) {
                return format!("{}{}", abbrev, stripped);
            }
        }

        // Extract local name from URI
        if let Some(pos) = uri.rfind(['/', '#']) {
            uri[pos + 1..].to_string()
        } else {
            uri.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abbreviate_uri() {
        let config = DiagramConfig::default();
        let generator = AsciiDiagramGenerator::new(config);

        assert_eq!(
            generator.abbreviate_uri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            "rdf:type"
        );
        assert_eq!(
            generator.abbreviate_uri("http://xmlns.com/foaf/0.1/name"),
            "foaf:name"
        );
        assert_eq!(
            generator.abbreviate_uri("http://example.org/person/John"),
            "John"
        );
    }

    #[test]
    fn test_tree_layout_simple() {
        let triples = vec![
            DiagramTriple {
                subject: "ex:John".to_string(),
                predicate: "rdf:type".to_string(),
                object: "foaf:Person".to_string(),
            },
            DiagramTriple {
                subject: "ex:John".to_string(),
                predicate: "foaf:name".to_string(),
                object: "\"John Doe\"".to_string(),
            },
        ];

        let config = DiagramConfig::default();
        let generator = AsciiDiagramGenerator::new(config);

        let mut output = Vec::new();
        generator.generate(&triples, &mut output).unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("ex:John"));
        assert!(result.contains("type"));
        assert!(result.contains("name"));
    }

    #[test]
    fn test_graph_layout() {
        let triples = vec![
            DiagramTriple {
                subject: "ex:John".to_string(),
                predicate: "foaf:knows".to_string(),
                object: "ex:Jane".to_string(),
            },
            DiagramTriple {
                subject: "ex:Jane".to_string(),
                predicate: "foaf:knows".to_string(),
                object: "ex:Bob".to_string(),
            },
        ];

        let config = DiagramConfig {
            style: LayoutStyle::Graph,
            ..Default::default()
        };
        let generator = AsciiDiagramGenerator::new(config);

        let mut output = Vec::new();
        generator.generate(&triples, &mut output).unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("RDF Graph"));
        assert!(result.contains("ex:John"));
        assert!(result.contains("ex:Jane"));
    }

    #[test]
    fn test_compact_layout() {
        let triples = vec![
            DiagramTriple {
                subject: "ex:John".to_string(),
                predicate: "rdf:type".to_string(),
                object: "foaf:Person".to_string(),
            },
            DiagramTriple {
                subject: "ex:John".to_string(),
                predicate: "foaf:name".to_string(),
                object: "\"John Doe\"".to_string(),
            },
            DiagramTriple {
                subject: "ex:Jane".to_string(),
                predicate: "rdf:type".to_string(),
                object: "foaf:Person".to_string(),
            },
        ];

        let config = DiagramConfig {
            style: LayoutStyle::Compact,
            ..Default::default()
        };
        let generator = AsciiDiagramGenerator::new(config);

        let mut output = Vec::new();
        generator.generate(&triples, &mut output).unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("Compact"));
        assert!(result.contains("ex:John"));
        assert!(result.contains("ex:Jane"));
    }

    #[test]
    fn test_list_layout() {
        let triples = vec![
            DiagramTriple {
                subject: "ex:John".to_string(),
                predicate: "rdf:type".to_string(),
                object: "foaf:Person".to_string(),
            },
            DiagramTriple {
                subject: "ex:John".to_string(),
                predicate: "foaf:name".to_string(),
                object: "\"John Doe\"".to_string(),
            },
        ];

        let config = DiagramConfig {
            style: LayoutStyle::List,
            ..Default::default()
        };
        let generator = AsciiDiagramGenerator::new(config);

        let mut output = Vec::new();
        generator.generate(&triples, &mut output).unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("RDF Triples:"));
        assert!(result.contains("1."));
        assert!(result.contains("2."));
    }

    #[test]
    fn test_empty_triples() {
        let triples = vec![];

        let config = DiagramConfig::default();
        let generator = AsciiDiagramGenerator::new(config);

        let mut output = Vec::new();
        generator.generate(&triples, &mut output).unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("no triples"));
    }

    #[test]
    fn test_max_edges_limit() {
        let triples = vec![
            DiagramTriple {
                subject: "ex:S1".to_string(),
                predicate: "ex:p".to_string(),
                object: "ex:O1".to_string(),
            },
            DiagramTriple {
                subject: "ex:S2".to_string(),
                predicate: "ex:p".to_string(),
                object: "ex:O2".to_string(),
            },
            DiagramTriple {
                subject: "ex:S3".to_string(),
                predicate: "ex:p".to_string(),
                object: "ex:O3".to_string(),
            },
        ];

        let config = DiagramConfig {
            style: LayoutStyle::List,
            max_edges: 2,
            ..Default::default()
        };
        let generator = AsciiDiagramGenerator::new(config);

        let mut output = Vec::new();
        generator.generate(&triples, &mut output).unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("more triples"));
    }
}
