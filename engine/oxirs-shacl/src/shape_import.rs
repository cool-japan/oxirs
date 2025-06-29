//! Shape Import and External Reference System
//!
//! Provides comprehensive support for importing external SHACL shapes, handling
//! dependencies, resolving references, and managing circular dependencies.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use url::Url;

use oxirs_core::{model::NamedNode, Store};

use crate::{
    iri_resolver::{IriResolutionError, IriResolver},
    shapes::{object_to_term, ShapeParser},
    Result, ShaclError, Shape, ShapeId,
};

/// Shape import configuration
#[derive(Debug, Clone)]
pub struct ShapeImportConfig {
    /// Maximum import depth to prevent infinite recursion
    pub max_import_depth: usize,
    /// Timeout for fetching external resources
    pub fetch_timeout: Duration,
    /// Cache TTL for external resources
    pub cache_ttl: Duration,
    /// Allow HTTP imports (security consideration)
    pub allow_http: bool,
    /// Allow file:// imports
    pub allow_file: bool,
    /// Custom headers for HTTP requests
    pub http_headers: HashMap<String, String>,
    /// Maximum size for imported resources
    pub max_resource_size: usize,
    /// Enable aggressive caching
    pub enable_caching: bool,
}

impl Default for ShapeImportConfig {
    fn default() -> Self {
        Self {
            max_import_depth: 10,
            fetch_timeout: Duration::from_secs(30),
            cache_ttl: Duration::from_secs(3600), // 1 hour
            allow_http: true,
            allow_file: false, // Security: disabled by default
            http_headers: HashMap::new(),
            max_resource_size: 10 * 1024 * 1024, // 10MB
            enable_caching: true,
        }
    }
}

/// Import directive for external shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportDirective {
    /// Source IRI to import from
    pub source_iri: String,
    /// Optional target namespace for imported shapes
    pub target_namespace: Option<String>,
    /// Specific shapes to import (None = import all)
    pub specific_shapes: Option<Vec<String>>,
    /// Import type
    pub import_type: ImportType,
    /// Optional format hint
    pub format_hint: Option<String>,
}

/// Types of imports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportType {
    /// Include all shapes from source
    Include,
    /// Import specific shapes only
    Selective,
    /// Import as dependency (load but don't expose)
    Dependency,
    /// Import with namespace mapping
    NamespaceMapping(String),
}

/// Cached import result
#[derive(Debug, Clone)]
struct CachedImport {
    /// Imported shapes
    shapes: Vec<Shape>,
    /// Import timestamp
    cached_at: Instant,
    /// Source IRI
    source_iri: String,
    /// Import metadata
    metadata: ImportMetadata,
}

/// Import metadata for tracking and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportMetadata {
    /// Source IRI
    pub source_iri: String,
    /// Number of shapes imported
    pub shape_count: usize,
    /// Import timestamp
    pub imported_at: String,
    /// Import depth in dependency chain
    pub import_depth: usize,
    /// Content type of source
    pub content_type: Option<String>,
    /// Size of imported content
    pub content_size: usize,
    /// Checksum of imported content
    pub content_hash: String,
}

/// Import result with metadata
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// Successfully imported shapes
    pub shapes: Vec<Shape>,
    /// Import metadata
    pub metadata: ImportMetadata,
    /// Any warnings during import
    pub warnings: Vec<String>,
    /// Dependency chain
    pub dependency_chain: Vec<String>,
}

/// Shape import manager
#[derive(Debug)]
pub struct ShapeImportManager {
    /// Import configuration
    config: ShapeImportConfig,
    /// IRI resolver for namespace and IRI resolution
    iri_resolver: IriResolver,
    /// Cache for imported shapes
    import_cache: HashMap<String, CachedImport>,
    /// Currently processing imports (for circular dependency detection)
    processing_imports: HashSet<String>,
    /// Import statistics
    stats: ImportStatistics,
    /// Dependency graph
    dependency_graph: HashMap<String, Vec<String>>,
}

/// Import statistics
#[derive(Debug, Default)]
pub struct ImportStatistics {
    /// Total imports attempted
    pub total_imports: usize,
    /// Successful imports
    pub successful_imports: usize,
    /// Failed imports
    pub failed_imports: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Total shapes imported
    pub total_shapes_imported: usize,
    /// Average import time
    pub average_import_time_ms: f64,
}

impl ShapeImportManager {
    /// Create a new shape import manager
    pub fn new(config: ShapeImportConfig) -> Self {
        Self {
            config,
            iri_resolver: IriResolver::new(),
            import_cache: HashMap::new(),
            processing_imports: HashSet::new(),
            stats: ImportStatistics::default(),
            dependency_graph: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(ShapeImportConfig::default())
    }

    /// Set IRI resolver for namespace handling
    pub fn with_iri_resolver(mut self, resolver: IriResolver) -> Self {
        self.iri_resolver = resolver;
        self
    }

    /// Import shapes from external source
    pub fn import_shapes(
        &mut self,
        directive: &ImportDirective,
        current_depth: usize,
    ) -> Result<ImportResult> {
        let start_time = Instant::now();
        self.stats.total_imports += 1;

        // Check import depth
        if current_depth > self.config.max_import_depth {
            return Err(ShaclError::Configuration(format!(
                "Maximum import depth {} exceeded",
                self.config.max_import_depth
            )));
        }

        // Resolve source IRI
        let resolved_iri = self.iri_resolver.resolve_iri(&directive.source_iri)?;

        // Check for circular dependency
        if self.processing_imports.contains(&resolved_iri) {
            return Err(ShaclError::Configuration(format!(
                "Circular import dependency detected: {}",
                resolved_iri
            )));
        }

        // Check cache first
        if self.config.enable_caching {
            if let Some(cached) = self.get_cached_import(&resolved_iri) {
                let cached_shapes = cached.shapes.clone();
                let cached_metadata = cached.metadata.clone();
                let dependency_chain = self.build_dependency_chain(&resolved_iri);
                self.stats.cache_hits += 1;
                return Ok(ImportResult {
                    shapes: cached_shapes,
                    metadata: cached_metadata,
                    warnings: Vec::new(),
                    dependency_chain,
                });
            }
        }

        // Mark as processing
        self.processing_imports.insert(resolved_iri.clone());

        // Perform the import
        let result = self.import_from_source(&resolved_iri, directive, current_depth);

        // Remove from processing set
        self.processing_imports.remove(&resolved_iri);

        // Update statistics
        let import_time = start_time.elapsed();
        self.update_import_stats(import_time, result.is_ok());

        result
    }

    /// Import shapes from a specific source
    fn import_from_source(
        &mut self,
        source_iri: &str,
        directive: &ImportDirective,
        current_depth: usize,
    ) -> Result<ImportResult> {
        // Validate source IRI scheme
        self.validate_source_iri(source_iri)?;

        // Fetch content from source
        let (content, content_type, content_size) = self.fetch_content(source_iri)?;

        // Determine format
        let format = self.determine_format(&content_type, &directive.format_hint, source_iri);

        // Parse shapes from content
        let mut parser = ShapeParser::new();
        let shapes = parser.parse_shapes_from_rdf(&content, &format, Some(source_iri))?;

        // Filter shapes if selective import
        let filtered_shapes = self.filter_shapes(shapes, directive)?;

        // Process nested imports
        let mut all_shapes = filtered_shapes;
        let mut warnings = Vec::new();
        let mut dependency_chain = vec![source_iri.to_string()];

        // Look for import directives in the loaded shapes
        let nested_imports = self.extract_import_directives(&content)?;
        for nested_directive in nested_imports {
            match self.import_shapes(&nested_directive, current_depth + 1) {
                Ok(nested_result) => {
                    all_shapes.extend(nested_result.shapes);
                    dependency_chain.extend(nested_result.dependency_chain);
                }
                Err(e) => {
                    warnings.push(format!(
                        "Failed to import nested dependency {}: {}",
                        nested_directive.source_iri, e
                    ));
                }
            }
        }

        // Create metadata
        let metadata = ImportMetadata {
            source_iri: source_iri.to_string(),
            shape_count: all_shapes.len(),
            imported_at: chrono::Utc::now().to_rfc3339(),
            import_depth: current_depth,
            content_type: Some(content_type),
            content_size,
            content_hash: self.compute_content_hash(&content),
        };

        // Cache the result
        if self.config.enable_caching {
            self.cache_import(source_iri, &all_shapes, &metadata);
        }

        // Update dependency graph
        self.update_dependency_graph(source_iri, &dependency_chain);

        // Update statistics
        self.stats.successful_imports += 1;
        self.stats.total_shapes_imported += all_shapes.len();

        Ok(ImportResult {
            shapes: all_shapes,
            metadata,
            warnings,
            dependency_chain,
        })
    }

    /// Validate source IRI based on security policy
    fn validate_source_iri(&self, source_iri: &str) -> Result<()> {
        let url = Url::parse(source_iri).map_err(|e| {
            ShaclError::Configuration(format!("Invalid source IRI {}: {}", source_iri, e))
        })?;

        match url.scheme() {
            "http" => {
                if !self.config.allow_http {
                    return Err(ShaclError::Configuration(
                        "HTTP imports are disabled for security".to_string(),
                    ));
                }
            }
            "https" => {} // Always allowed
            "file" => {
                if !self.config.allow_file {
                    return Err(ShaclError::Configuration(
                        "File imports are disabled for security".to_string(),
                    ));
                }
            }
            "urn" => {} // URN schemes allowed for abstract references
            scheme => {
                return Err(ShaclError::Configuration(format!(
                    "Unsupported URI scheme: {}",
                    scheme
                )));
            }
        }

        Ok(())
    }

    /// Fetch content from external source
    fn fetch_content(&self, source_iri: &str) -> Result<(String, String, usize)> {
        let url = Url::parse(source_iri).map_err(|e| {
            ShaclError::Configuration(format!("Invalid source IRI {}: {}", source_iri, e))
        })?;

        match url.scheme() {
            "http" | "https" => self.fetch_http_content(source_iri),
            "file" => self.fetch_file_content(&url),
            _ => Err(ShaclError::Configuration(format!(
                "Cannot fetch content from scheme: {}",
                url.scheme()
            ))),
        }
    }

    /// Fetch content via HTTP(S)
    fn fetch_http_content(&self, url: &str) -> Result<(String, String, usize)> {
        let parsed_url = Url::parse(url)
            .map_err(|e| ShaclError::Configuration(format!("Invalid URL {}: {}", url, e)))?;

        // Validate scheme
        match parsed_url.scheme() {
            "http" => {
                if !self.config.allow_http {
                    return Err(ShaclError::Configuration(
                        "HTTP imports are disabled for security".to_string(),
                    ));
                }
            }
            "https" => {}
            _ => {
                return Err(ShaclError::Configuration(format!(
                    "Unsupported URL scheme: {}",
                    parsed_url.scheme()
                )));
            }
        }

        // Build HTTP client with proper configuration
        let client = reqwest::blocking::ClientBuilder::new()
            .timeout(self.config.fetch_timeout)
            .user_agent("OxiRS-SHACL/1.0")
            .danger_accept_invalid_certs(false) // Always validate SSL certificates
            .redirect(reqwest::redirect::Policy::limited(5)) // Limit redirects for security
            .build()
            .map_err(|e| {
                ShaclError::Configuration(format!("Failed to create HTTP client: {}", e))
            })?;

        // Build request with custom headers
        let mut request = client.get(url);
        for (key, value) in &self.config.http_headers {
            request = request.header(key, value);
        }

        // Add standard headers for RDF content
        request = request.header("Accept", "text/turtle, application/rdf+xml, application/ld+json, application/n-triples, text/plain, */*;q=0.1");

        // Send request
        let response = request
            .send()
            .map_err(|e| ShaclError::Configuration(format!("HTTP request failed: {}", e)))?;

        // Check response status
        if !response.status().is_success() {
            return Err(ShaclError::Configuration(format!(
                "HTTP request failed with status {}: {}",
                response.status(),
                response
                    .status()
                    .canonical_reason()
                    .unwrap_or("Unknown error")
            )));
        }

        // Extract content type from response headers
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|ct| {
                // Extract just the media type, ignore charset and other parameters
                ct.split(';').next().unwrap_or(ct).trim().to_string()
            })
            .unwrap_or_else(|| "application/octet-stream".to_string());

        // Check content length before downloading if provided
        if let Some(content_length) = response.headers().get("content-length") {
            if let Ok(length_str) = content_length.to_str() {
                if let Ok(length) = length_str.parse::<usize>() {
                    if length > self.config.max_resource_size {
                        return Err(ShaclError::Configuration(format!(
                            "Resource too large: {} bytes (max: {})",
                            length, self.config.max_resource_size
                        )));
                    }
                }
            }
        }

        // Download content
        let content = response.text().map_err(|e| {
            ShaclError::Configuration(format!("Failed to read response body: {}", e))
        })?;

        // Check actual content size
        let content_size = content.len();
        if content_size > self.config.max_resource_size {
            return Err(ShaclError::Configuration(format!(
                "Resource too large: {} bytes (max: {})",
                content_size, self.config.max_resource_size
            )));
        }

        // Validate content is not empty
        if content.is_empty() {
            return Err(ShaclError::Configuration(
                "Retrieved empty content from HTTP resource".to_string(),
            ));
        }

        Ok((content, content_type, content_size))
    }

    /// Fetch content from file system
    fn fetch_file_content(&self, url: &Url) -> Result<(String, String, usize)> {
        let path = url
            .to_file_path()
            .map_err(|_| ShaclError::Configuration(format!("Invalid file path: {}", url)))?;

        let content = std::fs::read_to_string(&path).map_err(|e| {
            ShaclError::Configuration(format!("Failed to read file {}: {}", path.display(), e))
        })?;

        // Check size limit
        if content.len() > self.config.max_resource_size {
            return Err(ShaclError::Configuration(format!(
                "File size {} exceeds limit {}",
                content.len(),
                self.config.max_resource_size
            )));
        }

        // Determine content type from file extension
        let content_type = match path.extension().and_then(|ext| ext.to_str()) {
            Some("ttl") | Some("turtle") => "text/turtle",
            Some("nt") => "application/n-triples",
            Some("rdf") | Some("xml") => "application/rdf+xml",
            Some("jsonld") => "application/ld+json",
            _ => "text/turtle", // Default
        };

        let content_len = content.len();
        Ok((content, content_type.to_string(), content_len))
    }

    /// Determine RDF format from content type and hints
    fn determine_format(
        &self,
        content_type: &str,
        format_hint: &Option<String>,
        source_iri: &str,
    ) -> String {
        // Use format hint if provided
        if let Some(hint) = format_hint {
            return hint.clone();
        }

        // Use content type
        match content_type {
            "text/turtle" => "turtle".to_string(),
            "application/n-triples" => "ntriples".to_string(),
            "application/rdf+xml" => "rdfxml".to_string(),
            "application/ld+json" => "jsonld".to_string(),
            _ => {
                // Try to guess from file extension
                if source_iri.ends_with(".ttl") || source_iri.ends_with(".turtle") {
                    "turtle".to_string()
                } else if source_iri.ends_with(".nt") {
                    "ntriples".to_string()
                } else if source_iri.ends_with(".rdf") || source_iri.ends_with(".xml") {
                    "rdfxml".to_string()
                } else if source_iri.ends_with(".jsonld") {
                    "jsonld".to_string()
                } else {
                    "turtle".to_string() // Default
                }
            }
        }
    }

    /// Filter shapes based on import directive
    fn filter_shapes(&self, shapes: Vec<Shape>, directive: &ImportDirective) -> Result<Vec<Shape>> {
        match &directive.import_type {
            ImportType::Include => Ok(shapes),
            ImportType::Selective => {
                if let Some(specific_shapes) = &directive.specific_shapes {
                    let specific_set: HashSet<String> = specific_shapes.iter().cloned().collect();
                    Ok(shapes
                        .into_iter()
                        .filter(|shape| specific_set.contains(shape.id.as_str()))
                        .collect())
                } else {
                    Ok(shapes)
                }
            }
            ImportType::Dependency => Ok(shapes), // Dependencies are loaded but marked differently
            ImportType::NamespaceMapping(target_namespace) => {
                // Implement namespace remapping
                let mut remapped_shapes = Vec::new();
                for mut shape in shapes {
                    // Remap shape ID if needed
                    if let Some(new_id) = self.remap_shape_id(&shape.id, target_namespace) {
                        shape.id = new_id;
                    }

                    // Remap target IRIs in targets
                    for target in &mut shape.targets {
                        self.remap_target_iris(target, target_namespace);
                    }

                    // Remap constraint references
                    for (_, constraint) in &mut shape.constraints {
                        self.remap_constraint_iris(constraint, target_namespace);
                    }

                    remapped_shapes.push(shape);
                }
                Ok(remapped_shapes)
            }
        }
    }

    /// Extract import directives from RDF content
    fn extract_import_directives(&self, content: &str) -> Result<Vec<ImportDirective>> {
        use oxirs_core::model::{Object, Predicate, Subject};
        use oxirs_core::parser::{Parser, ParserConfig, RdfFormat};

        let mut directives = Vec::new();

        // Parse the content as RDF to extract import statements
        let parser_config = ParserConfig::default();
        let parser = Parser::with_config(RdfFormat::Turtle, parser_config);

        match parser.parse_str_to_quads(content) {
            Ok(quads) => {
                // Create a temporary graph for querying
                let mut graph = oxirs_core::graph::Graph::new();
                for quad in quads {
                    if quad.is_default_graph() {
                        graph.add_triple(quad.to_triple());
                    }
                }

                // Look for owl:imports statements
                if let Ok(owl_imports) = NamedNode::new("http://www.w3.org/2002/07/owl#imports") {
                    let triples =
                        graph.query_triples(None, Some(&Predicate::NamedNode(owl_imports)), None);

                    for triple in triples {
                        if let Ok(oxirs_core::model::Term::NamedNode(import_iri)) =
                            object_to_term(triple.object())
                        {
                            directives.push(ImportDirective {
                                source_iri: import_iri.as_str().to_string(),
                                target_namespace: None,
                                specific_shapes: None,
                                import_type: ImportType::Include,
                                format_hint: None,
                            });
                        }
                    }
                }

                // Look for custom SHACL import properties
                // sh:include - include all shapes from target
                if let Ok(sh_include) = NamedNode::new("http://www.w3.org/ns/shacl#include") {
                    let triples =
                        graph.query_triples(None, Some(&Predicate::NamedNode(sh_include)), None);

                    for triple in triples {
                        if let Ok(oxirs_core::model::Term::NamedNode(import_iri)) =
                            object_to_term(triple.object())
                        {
                            directives.push(ImportDirective {
                                source_iri: import_iri.as_str().to_string(),
                                target_namespace: None,
                                specific_shapes: None,
                                import_type: ImportType::Include,
                                format_hint: None,
                            });
                        }
                    }
                }

                // Look for sh:imports with selective import
                if let Ok(sh_imports) = NamedNode::new("http://www.w3.org/ns/shacl#imports") {
                    let triples =
                        graph.query_triples(None, Some(&Predicate::NamedNode(sh_imports)), None);

                    for triple in triples {
                        if let Ok(oxirs_core::model::Term::NamedNode(import_iri)) =
                            object_to_term(triple.object())
                        {
                            // Look for optional selective shape specification
                            // This could be extended to parse sh:includeShapes property
                            directives.push(ImportDirective {
                                source_iri: import_iri.as_str().to_string(),
                                target_namespace: None,
                                specific_shapes: None,
                                import_type: ImportType::Selective,
                                format_hint: None,
                            });
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to parse RDF content for import directives: {}", e);
                // Don't fail the entire import, just return empty directives
            }
        }

        Ok(directives)
    }

    /// Get cached import if valid
    fn get_cached_import(&self, source_iri: &str) -> Option<&CachedImport> {
        if let Some(cached) = self.import_cache.get(source_iri) {
            if cached.cached_at.elapsed() <= self.config.cache_ttl {
                return Some(cached);
            }
        }
        None
    }

    /// Cache import result
    fn cache_import(&mut self, source_iri: &str, shapes: &[Shape], metadata: &ImportMetadata) {
        let cached = CachedImport {
            shapes: shapes.to_vec(),
            cached_at: Instant::now(),
            source_iri: source_iri.to_string(),
            metadata: metadata.clone(),
        };
        self.import_cache.insert(source_iri.to_string(), cached);
    }

    /// Compute content hash for caching
    fn compute_content_hash(&self, content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Build dependency chain for import
    fn build_dependency_chain(&self, source_iri: &str) -> Vec<String> {
        self.dependency_graph
            .get(source_iri)
            .cloned()
            .unwrap_or_else(|| vec![source_iri.to_string()])
    }

    /// Update dependency graph
    fn update_dependency_graph(&mut self, source_iri: &str, chain: &[String]) {
        self.dependency_graph
            .insert(source_iri.to_string(), chain.to_vec());
    }

    /// Update import statistics
    fn update_import_stats(&mut self, duration: Duration, success: bool) {
        if success {
            self.stats.successful_imports += 1;
        } else {
            self.stats.failed_imports += 1;
        }

        // Update average import time
        let new_time_ms = duration.as_millis() as f64;
        if self.stats.total_imports == 1 {
            self.stats.average_import_time_ms = new_time_ms;
        } else {
            self.stats.average_import_time_ms = (self.stats.average_import_time_ms
                * (self.stats.total_imports - 1) as f64
                + new_time_ms)
                / self.stats.total_imports as f64;
        }
    }

    /// Clear import cache
    pub fn clear_cache(&mut self) {
        self.import_cache.clear();
    }

    /// Get import statistics
    pub fn get_statistics(&self) -> &ImportStatistics {
        &self.stats
    }

    /// Check for circular dependencies in the import graph
    pub fn check_circular_dependencies(&self) -> Result<()> {
        for (source, deps) in &self.dependency_graph {
            if self.has_circular_dependency(source, deps, &mut HashSet::new()) {
                return Err(ShaclError::Configuration(format!(
                    "Circular dependency detected starting from: {}",
                    source
                )));
            }
        }
        Ok(())
    }

    /// Helper method to detect circular dependencies
    fn has_circular_dependency(
        &self,
        current: &str,
        dependencies: &[String],
        visited: &mut HashSet<String>,
    ) -> bool {
        if visited.contains(current) {
            return true;
        }

        visited.insert(current.to_string());

        for dep in dependencies {
            if let Some(dep_deps) = self.dependency_graph.get(dep) {
                if self.has_circular_dependency(dep, dep_deps, visited) {
                    return true;
                }
            }
        }

        visited.remove(current);
        false
    }

    /// Resolve external shape references
    pub fn resolve_external_references(
        &mut self,
        shapes: &mut [Shape],
    ) -> Result<Vec<ImportResult>> {
        let mut import_results = Vec::new();

        // Scan shapes for external references
        for shape in shapes.iter() {
            // Check constraint references that might be external
            for (_component_id, constraint) in &shape.constraints {
                if let Some(external_refs) = self.extract_external_references(constraint) {
                    for external_ref in external_refs {
                        let directive = ImportDirective {
                            source_iri: external_ref,
                            target_namespace: None,
                            specific_shapes: None,
                            import_type: ImportType::Dependency,
                            format_hint: None,
                        };

                        match self.import_shapes(&directive, 0) {
                            Ok(result) => import_results.push(result),
                            Err(e) => {
                                tracing::warn!("Failed to resolve external reference: {}", e);
                            }
                        }
                    }
                }
            }
        }

        Ok(import_results)
    }

    /// Extract external references from a constraint
    fn extract_external_references(
        &self,
        constraint: &crate::constraints::Constraint,
    ) -> Option<Vec<String>> {
        let mut external_refs = Vec::new();

        match constraint {
            crate::constraints::Constraint::Node(node_constraint) => {
                // Node constraints reference other shapes
                let shape_iri = node_constraint.shape.as_str();
                if self.is_external_reference(shape_iri) {
                    external_refs.push(shape_iri.to_string());
                }
            }
            crate::constraints::Constraint::QualifiedValueShape(qvs_constraint) => {
                // Qualified value shape constraints reference other shapes
                let shape_iri = qvs_constraint.shape.as_str();
                if self.is_external_reference(shape_iri) {
                    external_refs.push(shape_iri.to_string());
                }
            }
            crate::constraints::Constraint::And(and_constraint) => {
                // AND constraints may reference multiple shapes
                for shape_id in &and_constraint.shapes {
                    let shape_iri = shape_id.as_str();
                    if self.is_external_reference(shape_iri) {
                        external_refs.push(shape_iri.to_string());
                    }
                }
            }
            crate::constraints::Constraint::Or(or_constraint) => {
                // OR constraints may reference multiple shapes
                for shape_id in &or_constraint.shapes {
                    let shape_iri = shape_id.as_str();
                    if self.is_external_reference(shape_iri) {
                        external_refs.push(shape_iri.to_string());
                    }
                }
            }
            crate::constraints::Constraint::Xone(xone_constraint) => {
                // XONE constraints may reference multiple shapes
                for shape_id in &xone_constraint.shapes {
                    let shape_iri = shape_id.as_str();
                    if self.is_external_reference(shape_iri) {
                        external_refs.push(shape_iri.to_string());
                    }
                }
            }
            crate::constraints::Constraint::Not(not_constraint) => {
                // NOT constraints reference a shape
                let shape_iri = not_constraint.shape.as_str();
                if self.is_external_reference(shape_iri) {
                    external_refs.push(shape_iri.to_string());
                }
            }
            crate::constraints::Constraint::Sparql(sparql_constraint) => {
                // SPARQL constraints might reference external resources in the query
                let query = &sparql_constraint.query;

                // Look for GRAPH clauses with IRIs
                if let Some(graph_iris) = self.extract_graph_iris_from_sparql(query) {
                    for graph_iri in graph_iris {
                        if self.is_external_reference(&graph_iri) {
                            external_refs.push(graph_iri);
                        }
                    }
                }

                // Look for SERVICE clauses (federated queries)
                if let Some(service_iris) = self.extract_service_iris_from_sparql(query) {
                    for service_iri in service_iris {
                        if self.is_external_reference(&service_iri) {
                            external_refs.push(service_iri);
                        }
                    }
                }
            }
            // Other constraint types don't typically have external references
            _ => {}
        }

        if external_refs.is_empty() {
            None
        } else {
            Some(external_refs)
        }
    }

    /// Check if an IRI refers to an external resource that needs importing
    fn is_external_reference(&self, iri: &str) -> bool {
        // Consider an IRI external if it's:
        // 1. An absolute HTTP/HTTPS IRI (not a local reference)
        // 2. Not in our current cache
        // 3. Not using a local base IRI

        if let Ok(parsed_url) = Url::parse(iri) {
            match parsed_url.scheme() {
                "http" | "https" => {
                    // Check if it's already in our cache
                    !self.import_cache.contains_key(iri)
                }
                "file" => {
                    // File URLs are considered external if not in cache
                    !self.import_cache.contains_key(iri)
                }
                _ => false,
            }
        } else {
            // Not a valid absolute IRI, probably a local reference
            false
        }
    }

    /// Extract GRAPH IRIs from SPARQL query
    fn extract_graph_iris_from_sparql(&self, query: &str) -> Option<Vec<String>> {
        use regex::Regex;

        // Basic regex to find GRAPH clauses - could be improved
        let graph_regex = Regex::new(r"(?i)GRAPH\s+<([^>]+)>").ok()?;
        let mut iris = Vec::new();

        for captures in graph_regex.captures_iter(query) {
            if let Some(iri_match) = captures.get(1) {
                iris.push(iri_match.as_str().to_string());
            }
        }

        if iris.is_empty() {
            None
        } else {
            Some(iris)
        }
    }

    /// Extract SERVICE IRIs from SPARQL query
    fn extract_service_iris_from_sparql(&self, query: &str) -> Option<Vec<String>> {
        use regex::Regex;

        // Basic regex to find SERVICE clauses - could be improved
        let service_regex = Regex::new(r"(?i)SERVICE\s+<([^>]+)>").ok()?;
        let mut iris = Vec::new();

        for captures in service_regex.captures_iter(query) {
            if let Some(iri_match) = captures.get(1) {
                iris.push(iri_match.as_str().to_string());
            }
        }

        if iris.is_empty() {
            None
        } else {
            Some(iris)
        }
    }

    /// Remap a shape ID to a new namespace
    fn remap_shape_id(&self, shape_id: &ShapeId, target_namespace: &str) -> Option<ShapeId> {
        let original_iri = shape_id.as_str();

        // Try to extract the local part from the original IRI
        if let Some(hash_pos) = original_iri.rfind('#') {
            let local_part = &original_iri[hash_pos + 1..];
            let new_iri = format!("{}{}", target_namespace, local_part);
            Some(ShapeId::new(new_iri))
        } else if let Some(slash_pos) = original_iri.rfind('/') {
            let local_part = &original_iri[slash_pos + 1..];
            let new_iri = if target_namespace.ends_with('#') || target_namespace.ends_with('/') {
                format!("{}{}", target_namespace, local_part)
            } else {
                format!("{}#{}", target_namespace, local_part)
            };
            Some(ShapeId::new(new_iri))
        } else {
            // Can't extract local part, use original
            None
        }
    }

    /// Remap target IRIs in a target definition
    fn remap_target_iris(&self, target: &mut Target, target_namespace: &str) {
        match target {
            Target::Class(class_iri) => {
                if let Some(remapped) = self.remap_iri_string(class_iri.as_str(), target_namespace)
                {
                    if let Ok(new_node) = NamedNode::new(remapped) {
                        *class_iri = new_node;
                    }
                }
            }
            Target::Node(node_iri) => {
                if let Some(remapped) = self.remap_iri_string(node_iri.as_str(), target_namespace) {
                    if let Ok(new_node) = NamedNode::new(remapped) {
                        *node_iri = new_node;
                    }
                }
            }
            Target::ObjectsOf(prop_iri) => {
                if let Some(remapped) = self.remap_iri_string(prop_iri.as_str(), target_namespace) {
                    if let Ok(new_node) = NamedNode::new(remapped) {
                        *prop_iri = new_node;
                    }
                }
            }
            Target::SubjectsOf(prop_iri) => {
                if let Some(remapped) = self.remap_iri_string(prop_iri.as_str(), target_namespace) {
                    if let Ok(new_node) = NamedNode::new(remapped) {
                        *prop_iri = new_node;
                    }
                }
            }
            // For SPARQL targets, we'd need to parse and rewrite the query
            Target::Sparql(_) => {
                // Complex remapping for SPARQL targets would require SPARQL parsing
                tracing::warn!("SPARQL target remapping not yet implemented");
            }
        }
    }

    /// Remap constraint IRIs in a constraint
    fn remap_constraint_iris(
        &self,
        constraint: &mut crate::constraints::Constraint,
        target_namespace: &str,
    ) {
        use crate::constraints::Constraint;

        match constraint {
            Constraint::Node(node_constraint) => {
                if let Some(remapped) =
                    self.remap_shape_id(&node_constraint.shape, target_namespace)
                {
                    node_constraint.shape = remapped;
                }
            }
            Constraint::QualifiedValueShape(qvs_constraint) => {
                if let Some(remapped) = self.remap_shape_id(&qvs_constraint.shape, target_namespace)
                {
                    qvs_constraint.shape = remapped;
                }
            }
            Constraint::And(and_constraint) => {
                for shape_id in &mut and_constraint.shapes {
                    if let Some(remapped) = self.remap_shape_id(shape_id, target_namespace) {
                        *shape_id = remapped;
                    }
                }
            }
            Constraint::Or(or_constraint) => {
                for shape_id in &mut or_constraint.shapes {
                    if let Some(remapped) = self.remap_shape_id(shape_id, target_namespace) {
                        *shape_id = remapped;
                    }
                }
            }
            Constraint::Xone(xone_constraint) => {
                for shape_id in &mut xone_constraint.shapes {
                    if let Some(remapped) = self.remap_shape_id(shape_id, target_namespace) {
                        *shape_id = remapped;
                    }
                }
            }
            Constraint::Not(not_constraint) => {
                if let Some(remapped) = self.remap_shape_id(&not_constraint.shape, target_namespace)
                {
                    not_constraint.shape = remapped;
                }
            }
            Constraint::Class(class_constraint) => {
                if let Some(remapped) =
                    self.remap_iri_string(class_constraint.class.as_str(), target_namespace)
                {
                    if let Ok(new_node) = NamedNode::new(remapped) {
                        class_constraint.class = new_node;
                    }
                }
            }
            Constraint::In(in_constraint) => {
                for value in &mut in_constraint.values {
                    if let crate::constraints::ValueType::IRI(iri) = value {
                        if let Some(remapped) =
                            self.remap_iri_string(iri.as_str(), target_namespace)
                        {
                            if let Ok(new_node) = NamedNode::new(remapped) {
                                *iri = new_node;
                            }
                        }
                    }
                }
            }
            // Other constraints may not have IRIs that need remapping
            _ => {}
        }
    }

    /// Helper to remap a single IRI string to a new namespace
    fn remap_iri_string(&self, original_iri: &str, target_namespace: &str) -> Option<String> {
        // Extract local part and combine with target namespace
        if let Some(hash_pos) = original_iri.rfind('#') {
            let local_part = &original_iri[hash_pos + 1..];
            Some(format!("{}{}", target_namespace, local_part))
        } else if let Some(slash_pos) = original_iri.rfind('/') {
            let local_part = &original_iri[slash_pos + 1..];
            if target_namespace.ends_with('#') || target_namespace.ends_with('/') {
                Some(format!("{}{}", target_namespace, local_part))
            } else {
                Some(format!("{}#{}", target_namespace, local_part))
            }
        } else {
            // Can't extract local part, return None to keep original
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_config_default() {
        let config = ShapeImportConfig::default();
        assert_eq!(config.max_import_depth, 10);
        assert!(config.allow_http);
        assert!(!config.allow_file);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_import_manager_creation() {
        let manager = ShapeImportManager::default();
        assert_eq!(manager.stats.total_imports, 0);
        assert_eq!(manager.import_cache.len(), 0);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let manager = ShapeImportManager::default();
        // Start with empty dependency graph
        assert!(manager.check_circular_dependencies().is_ok());
    }

    #[test]
    fn test_source_iri_validation() {
        let config = ShapeImportConfig {
            allow_http: false,
            allow_file: false,
            ..Default::default()
        };
        let manager = ShapeImportManager::new(config);

        // HTTPS should always be allowed
        assert!(manager
            .validate_source_iri("https://example.org/shapes.ttl")
            .is_ok());

        // HTTP should be blocked when disabled
        assert!(manager
            .validate_source_iri("http://example.org/shapes.ttl")
            .is_err());

        // File should be blocked when disabled
        assert!(manager
            .validate_source_iri("file:///path/to/shapes.ttl")
            .is_err());
    }

    #[test]
    fn test_format_determination() {
        let manager = ShapeImportManager::default();

        assert_eq!(
            manager.determine_format("text/turtle", &None, "test.ttl"),
            "turtle"
        );

        assert_eq!(
            manager.determine_format("application/rdf+xml", &None, "test.rdf"),
            "rdfxml"
        );

        // Format hint should override content type
        assert_eq!(
            manager.determine_format("text/turtle", &Some("jsonld".to_string()), "test.ttl"),
            "jsonld"
        );
    }
}
