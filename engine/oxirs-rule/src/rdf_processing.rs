//! RDF Data Processing Module
//!
//! This module provides functionality for processing RDF data from various formats
//! including RDF/XML, Turtle, N-Triples, N-Quads, and JSON-LD.

use crate::rdf_integration::{convert_rule_atom, NamespaceManager, RdfRuleAtom};
use crate::rdf_integration::RdfTerm;  // Import the enum, not the trait
use crate::{RuleAtom, Term as RuleTerm};
use anyhow::{anyhow, Result};
use oxirs_core::model::{Dataset, Graph, Quad, Triple};
use oxirs_core::{OxirsError, Store};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt};

/// RDF format types supported for rule processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdfFormat {
    /// RDF/XML format
    RdfXml,
    /// Turtle format
    Turtle,
    /// N-Triples format
    NTriples,
    /// N-Quads format
    NQuads,
    /// JSON-LD format
    JsonLd,
    /// TriG format
    TriG,
}

impl RdfFormat {

    /// Detect format from file extension
    pub fn from_extension(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "rdf" | "xml" => Some(RdfFormat::RdfXml),
            "ttl" | "turtle" => Some(RdfFormat::Turtle),
            "nt" => Some(RdfFormat::NTriples),
            "nq" => Some(RdfFormat::NQuads),
            "jsonld" | "json" => Some(RdfFormat::JsonLd),
            "trig" => Some(RdfFormat::TriG),
            _ => None,
        }
    }

    /// Get MIME type for the format
    pub fn mime_type(&self) -> &'static str {
        match self {
            RdfFormat::RdfXml => "application/rdf+xml",
            RdfFormat::Turtle => "text/turtle",
            RdfFormat::NTriples => "application/n-triples",
            RdfFormat::NQuads => "application/n-quads",
            RdfFormat::JsonLd => "application/ld+json",
            RdfFormat::TriG => "application/trig",
        }
    }
}

/// RDF data processor for loading and parsing RDF data
pub struct RdfProcessor {
    store: Arc<Store>,
    namespaces: NamespaceManager,
    /// Configuration for processing
    config: ProcessingConfig,
}

/// Configuration for RDF processing
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    /// Whether to validate IRIs
    pub validate_iris: bool,
    /// Whether to resolve relative IRIs
    pub resolve_relative: bool,
    /// Base IRI for resolution
    pub base_iri: Option<String>,
    /// Maximum file size in bytes (0 = unlimited)
    pub max_file_size: usize,
    /// Whether to use streaming for large files
    pub use_streaming: bool,
    /// Streaming threshold in bytes
    pub streaming_threshold: usize,
    /// Whether to collect statistics
    pub collect_stats: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            validate_iris: true,
            resolve_relative: true,
            base_iri: None,
            max_file_size: 0, // Unlimited
            use_streaming: true,
            streaming_threshold: 10 * 1024 * 1024, // 10MB
            collect_stats: false,
        }
    }
}

/// Statistics collected during processing
#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub triples_processed: usize,
    pub quads_processed: usize,
    pub subjects: HashSet<String>,
    pub predicates: HashSet<String>,
    pub objects: HashSet<String>,
    pub graphs: HashSet<String>,
    pub parse_errors: Vec<String>,
    pub processing_time: std::time::Duration,
}

impl RdfProcessor {
    /// Create a new RDF processor
    pub fn new(store: Arc<Store>) -> Self {
        Self {
            store,
            namespaces: NamespaceManager::new(),
            config: ProcessingConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(store: Arc<Store>, config: ProcessingConfig) -> Self {
        let mut namespaces = NamespaceManager::new();
        if let Some(base) = &config.base_iri {
            namespaces.set_base(base.clone());
        }
        
        Self {
            store,
            namespaces,
            config,
        }
    }

    /// Process RDF data from a file
    pub async fn process_file(&mut self, path: &Path) -> Result<ProcessingStats> {
        let format = RdfFormat::from_extension(path)
            .ok_or_else(|| anyhow!("Unknown RDF format for file: {:?}", path))?;
        
        // Check file size
        let metadata = tokio::fs::metadata(path).await?;
        if self.config.max_file_size > 0 && metadata.len() > self.config.max_file_size as u64 {
            return Err(anyhow!("File size exceeds maximum: {} bytes", metadata.len()));
        }
        
        // Decide whether to use streaming
        let use_streaming = self.config.use_streaming && 
            metadata.len() > self.config.streaming_threshold as u64;
        
        let file = tokio::fs::File::open(path).await?;
        
        if use_streaming {
            self.process_stream(file, format).await
        } else {
            let mut contents = Vec::new();
            let mut reader = tokio::io::BufReader::new(file);
            reader.read_to_end(&mut contents).await?;
            self.process_data(&contents, format).await
        }
    }

    /// Process RDF data from bytes
    pub async fn process_data(&mut self, data: &[u8], format: RdfFormat) -> Result<ProcessingStats> {
        let start_time = std::time::Instant::now();
        let mut stats = ProcessingStats::default();
        
        match format {
            RdfFormat::RdfXml | RdfFormat::Turtle | RdfFormat::NTriples => {
                self.process_graph_format(data, format, &mut stats)?;
            }
            RdfFormat::NQuads | RdfFormat::TriG => {
                self.process_dataset_format(data, format, &mut stats)?;
            }
            RdfFormat::JsonLd => {
                self.process_jsonld(data, &mut stats).await?;
            }
        }
        
        stats.processing_time = start_time.elapsed();
        Ok(stats)
    }

    /// Process RDF data from a stream
    pub async fn process_stream(
        &mut self,
        mut reader: impl AsyncRead + Unpin,
        format: RdfFormat,
    ) -> Result<ProcessingStats> {
        let start_time = std::time::Instant::now();
        let mut stats = ProcessingStats::default();
        
        // For streaming, we need to handle line-based formats specially
        match format {
            RdfFormat::NTriples | RdfFormat::NQuads => {
                let reader = tokio::io::BufReader::new(reader);
                self.process_line_based_stream(reader, format, &mut stats).await?;
            }
            _ => {
                // For other formats, we need to buffer the entire content
                let mut buffer = Vec::new();
                reader.read_to_end(&mut buffer).await?;
                self.process_data(&buffer, format).await?;
            }
        }
        
        stats.processing_time = start_time.elapsed();
        Ok(stats)
    }

    /// Process graph formats (RDF/XML, Turtle, N-Triples)
    fn process_graph_format(
        &mut self,
        data: &[u8],
        format: RdfFormat,
        stats: &mut ProcessingStats,
    ) -> Result<()> {
        let graph_format = format.to_graph_format()
            .ok_or_else(|| anyhow!("Not a graph format: {:?}", format))?;
        
        let graph = Graph::parse(
            std::io::Cursor::new(data),
            graph_format,
            self.config.base_iri.as_deref(),
        )?;
        
        // Add triples to store and collect stats
        for triple in graph.iter() {
            self.store.insert(&Quad::from(triple.clone()))?;
            
            if self.config.collect_stats {
                stats.triples_processed += 1;
                stats.subjects.insert(triple.subject.to_string());
                stats.predicates.insert(triple.predicate.to_string());
                stats.objects.insert(triple.object.to_string());
            }
        }
        
        // Extract namespaces from Turtle
        if format == RdfFormat::Turtle {
            self.extract_turtle_prefixes(std::str::from_utf8(data)?)?;
        }
        
        Ok(())
    }

    /// Process dataset formats (N-Quads, TriG)
    fn process_dataset_format(
        &mut self,
        data: &[u8],
        format: RdfFormat,
        stats: &mut ProcessingStats,
    ) -> Result<()> {
        let dataset_format = format.to_dataset_format()
            .ok_or_else(|| anyhow!("Not a dataset format: {:?}", format))?;
        
        let dataset = Dataset::parse(
            std::io::Cursor::new(data),
            dataset_format,
            self.config.base_iri.as_deref(),
        )?;
        
        // Add quads to store and collect stats
        for quad in dataset.iter() {
            self.store.insert(quad)?;
            
            if self.config.collect_stats {
                stats.quads_processed += 1;
                stats.subjects.insert(quad.subject.to_string());
                stats.predicates.insert(quad.predicate.to_string());
                stats.objects.insert(quad.object.to_string());
                if let Some(graph) = &quad.graph_name {
                    stats.graphs.insert(graph.to_string());
                }
            }
        }
        
        Ok(())
    }

    /// Process JSON-LD format
    async fn process_jsonld(&mut self, data: &[u8], stats: &mut ProcessingStats) -> Result<()> {
        // For now, we'll use oxirs-core's JSON-LD support if available
        // This is a placeholder for full JSON-LD processing
        
        use oxirs_core::jsonld;
        
        let json: serde_json::Value = serde_json::from_slice(data)?;
        let quads = jsonld::to_rdf(&json, None)?;
        
        for quad in quads {
            self.store.insert(&quad)?;
            
            if self.config.collect_stats {
                stats.quads_processed += 1;
                stats.subjects.insert(quad.subject.to_string());
                stats.predicates.insert(quad.predicate.to_string());
                stats.objects.insert(quad.object.to_string());
                if let Some(graph) = &quad.graph_name {
                    stats.graphs.insert(graph.to_string());
                }
            }
        }
        
        Ok(())
    }

    /// Process line-based formats in streaming mode
    async fn process_line_based_stream(
        &mut self,
        reader: tokio::io::BufReader<impl AsyncRead + Unpin>,
        format: RdfFormat,
        stats: &mut ProcessingStats,
    ) -> Result<()> {
        let mut lines = reader.lines();
        let mut line_number = 0;
        
        while let Some(line) = lines.next_line().await? {
            line_number += 1;
            
            // Skip empty lines and comments
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            
            // Parse the line
            match format {
                RdfFormat::NTriples => {
                    match self.parse_ntriples_line(&line) {
                        Ok(triple) => {
                            self.store.insert(&Quad::from(triple))?;
                            stats.triples_processed += 1;
                        }
                        Err(e) => {
                            if self.config.collect_stats {
                                stats.parse_errors.push(format!("Line {}: {}", line_number, e));
                            }
                        }
                    }
                }
                RdfFormat::NQuads => {
                    match self.parse_nquads_line(&line) {
                        Ok(quad) => {
                            self.store.insert(&quad)?;
                            stats.quads_processed += 1;
                        }
                        Err(e) => {
                            if self.config.collect_stats {
                                stats.parse_errors.push(format!("Line {}: {}", line_number, e));
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        
        Ok(())
    }

    /// Parse a single N-Triples line
    fn parse_ntriples_line(&self, line: &str) -> Result<Triple> {
        // Use oxirs-core's parser
        let mut cursor = std::io::Cursor::new(line.as_bytes());
        let graph = Graph::parse(&mut cursor, GraphFormat::NTriples, None)?;
        let triple = graph.iter().next()
            .ok_or_else(|| anyhow!("No triple parsed from line"))?;
        Ok(triple.clone())
    }

    /// Parse a single N-Quads line
    fn parse_nquads_line(&self, line: &str) -> Result<Quad> {
        // Use oxirs-core's parser
        let mut cursor = std::io::Cursor::new(line.as_bytes());
        let dataset = Dataset::parse(&mut cursor, DatasetFormat::NQuads, None)?;
        let quad = dataset.iter().next()
            .ok_or_else(|| anyhow!("No quad parsed from line"))?;
        Ok(quad.clone())
    }

    /// Extract namespace prefixes from Turtle content
    fn extract_turtle_prefixes(&mut self, content: &str) -> Result<()> {
        // Simple regex-based extraction for @prefix declarations
        let prefix_regex = regex::Regex::new(r"@prefix\s+(\w+):\s*<([^>]+)>\s*\.")?;
        
        for cap in prefix_regex.captures_iter(content) {
            if let (Some(prefix), Some(namespace)) = (cap.get(1), cap.get(2)) {
                self.namespaces.add_prefix(
                    prefix.as_str().to_string(),
                    namespace.as_str().to_string(),
                );
            }
        }
        
        // Extract @base declaration
        let base_regex = regex::Regex::new(r"@base\s*<([^>]+)>\s*\.")?;
        if let Some(cap) = base_regex.captures(content) {
            if let Some(base) = cap.get(1) {
                self.namespaces.set_base(base.as_str().to_string());
            }
        }
        
        Ok(())
    }

    /// Convert loaded RDF data to rule atoms
    pub fn to_rule_atoms(&self) -> Result<Vec<RuleAtom>> {
        let mut atoms = Vec::new();
        
        for quad in self.store.iter() {
            let atom = if quad.graph_name.is_some() {
                // For now, skip quads with named graphs
                // In the future, we could represent these as 4-ary predicates
                continue;
            } else {
                RuleAtom::Triple {
                    subject: self.term_to_rule_term(&quad.subject.into())?,
                    predicate: self.term_to_rule_term(&quad.predicate.into())?,
                    object: self.term_to_rule_term(&quad.object)?,
                }
            };
            
            atoms.push(atom);
        }
        
        Ok(atoms)
    }

    /// Convert an RDF term to a rule term
    fn term_to_rule_term(&self, term: &oxirs_core::model::Term) -> Result<RuleTerm> {
        use oxirs_core::model::Term;
        
        match term {
            Term::NamedNode(n) => Ok(RuleTerm::Constant(self.namespaces.compact(n.as_str()))),
            Term::BlankNode(b) => Ok(RuleTerm::Constant(format!("_:{}", b.as_str()))),
            Term::Literal(l) => {
                if let Some(lang) = l.language() {
                    Ok(RuleTerm::Literal(format!("{}@{}", l.value(), lang)))
                } else if let Some(dt) = l.datatype() {
                    Ok(RuleTerm::Literal(format!("{}^^{}", l.value(), self.namespaces.compact(dt.as_str()))))
                } else {
                    Ok(RuleTerm::Literal(l.value().to_string()))
                }
            }
            Term::Variable(v) => Ok(RuleTerm::Variable(v.name().to_string())),
            Term::QuotedTriple(_) => Err(anyhow!("Quoted triples not yet supported in rules")),
        }
    }

    /// Get collected namespaces
    pub fn namespaces(&self) -> &NamespaceManager {
        &self.namespaces
    }

    /// Get the underlying store
    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }
}

/// Memory-efficient fact manager for large datasets
pub struct FactManager {
    /// In-memory facts (limited size)
    memory_facts: Vec<RdfRuleAtom>,
    /// Persistent storage path
    storage_path: Option<std::path::PathBuf>,
    /// Maximum facts in memory
    max_memory_facts: usize,
    /// Statistics
    total_facts: usize,
}

impl FactManager {
    /// Create a new fact manager
    pub fn new(max_memory_facts: usize) -> Self {
        Self {
            memory_facts: Vec::new(),
            storage_path: None,
            max_memory_facts,
            total_facts: 0,
        }
    }

    /// Enable persistent storage
    pub fn with_storage(mut self, path: std::path::PathBuf) -> Self {
        self.storage_path = Some(path);
        self
    }

    /// Add a fact
    pub fn add_fact(&mut self, fact: RdfRuleAtom) -> Result<()> {
        self.total_facts += 1;
        
        if self.memory_facts.len() < self.max_memory_facts {
            self.memory_facts.push(fact);
        } else if let Some(path) = &self.storage_path {
            // Spill to disk
            self.spill_to_disk(&fact)?;
        } else {
            return Err(anyhow!("Memory limit reached and no storage path configured"));
        }
        
        Ok(())
    }

    /// Spill fact to disk
    fn spill_to_disk(&self, fact: &RdfRuleAtom) -> Result<()> {
        // Implementation would serialize fact to disk
        // This is a placeholder
        Ok(())
    }

    /// Get an iterator over all facts
    pub fn iter(&self) -> impl Iterator<Item = &RdfRuleAtom> {
        self.memory_facts.iter()
    }

    /// Get total fact count
    pub fn total_count(&self) -> usize {
        self.total_facts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_format_detection() {
        assert_eq!(
            RdfFormat::from_extension(Path::new("test.ttl")),
            Some(RdfFormat::Turtle)
        );
        assert_eq!(
            RdfFormat::from_extension(Path::new("test.rdf")),
            Some(RdfFormat::RdfXml)
        );
        assert_eq!(
            RdfFormat::from_extension(Path::new("test.jsonld")),
            Some(RdfFormat::JsonLd)
        );
    }

    #[tokio::test]
    async fn test_process_turtle() {
        let store = Arc::new(Store::new().unwrap());
        let mut processor = RdfProcessor::new(store.clone());
        
        let turtle_data = r#"
            @prefix ex: <http://example.org/> .
            ex:subject ex:predicate ex:object .
        "#;
        
        let stats = processor.process_data(turtle_data.as_bytes(), RdfFormat::Turtle).await.unwrap();
        
        assert_eq!(stats.triples_processed, 1);
        assert_eq!(store.len().unwrap(), 1);
    }

    #[tokio::test]
    async fn test_process_ntriples() {
        let store = Arc::new(Store::new().unwrap());
        let mut processor = RdfProcessor::new(store.clone());
        
        let ntriples_data = r#"<http://example.org/subject> <http://example.org/predicate> <http://example.org/object> ."#;
        
        let stats = processor.process_data(ntriples_data.as_bytes(), RdfFormat::NTriples).await.unwrap();
        
        assert_eq!(stats.triples_processed, 1);
        assert_eq!(store.len().unwrap(), 1);
    }

    #[tokio::test]
    async fn test_streaming_processing() {
        let store = Arc::new(Store::new().unwrap());
        let config = ProcessingConfig {
            use_streaming: true,
            streaming_threshold: 0, // Force streaming
            ..Default::default()
        };
        let mut processor = RdfProcessor::with_config(store.clone(), config);
        
        // Create a temporary file with N-Triples data
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "<http://example.org/s1> <http://example.org/p> <http://example.org/o1> .").unwrap();
        writeln!(temp_file, "<http://example.org/s2> <http://example.org/p> <http://example.org/o2> .").unwrap();
        
        let stats = processor.process_file(temp_file.path()).await.unwrap();
        
        assert_eq!(stats.triples_processed, 2);
        assert_eq!(store.len().unwrap(), 2);
    }

    #[test]
    fn test_fact_manager() {
        let mut manager = FactManager::new(2);
        
        let fact1 = RdfRuleAtom::Triple {
            subject: RdfTerm::NamedNode(NamedNode::new("http://example.org/s1").unwrap()),
            predicate: RdfTerm::NamedNode(NamedNode::new("http://example.org/p").unwrap()),
            object: RdfTerm::NamedNode(NamedNode::new("http://example.org/o1").unwrap()),
        };
        
        manager.add_fact(fact1.clone()).unwrap();
        assert_eq!(manager.total_count(), 1);
        
        manager.add_fact(fact1.clone()).unwrap();
        assert_eq!(manager.total_count(), 2);
        
        // Third fact would exceed memory limit without storage
        assert!(manager.add_fact(fact1).is_err());
    }
}