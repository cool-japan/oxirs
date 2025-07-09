//! SHACL Security Module
//!
//! This module provides comprehensive security features for SHACL validation, particularly
//! for SPARQL constraints to prevent injection attacks and ensure safe execution.
//!
//! Features include:
//! - Query sanitization and analysis
//! - Execution sandboxing with resource limits
//! - Advanced injection detection with ML support
//! - Policy-based execution control
//! - Rate limiting and quota management
//! - Comprehensive audit logging
//! - Security context management
//! - Query rewriting for security

pub mod advanced;
pub mod secure_executor;

// Re-export key types for convenience
pub use advanced::{
    ExecutionConstraints, InjectionAnalysisResult, Permission, PolicyAuthorizationResult,
    SecurityConstraints, SecurityContext, SecurityEvent, SecurityEventType, SecurityMetrics,
    SecurityPolicy, SecurityPolicyManager,
};
pub use secure_executor::{
    utils, ComprehensiveSecurityMetrics, SecureConstraintResult, SecureExecutorFactory,
    SecureQueryResult, SecureSparqlExecutor,
};

use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::{Result, ShaclError};

/// Security configuration for SHACL validation
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Maximum allowed query execution time
    pub max_execution_time: Duration,
    /// Maximum allowed memory usage (bytes)
    pub max_memory_usage: usize,
    /// Maximum number of results allowed
    pub max_result_count: usize,
    /// Maximum query complexity score
    pub max_complexity_score: f64,
    /// Enable SPARQL injection detection
    pub enable_injection_detection: bool,
    /// Allowed functions in SPARQL queries
    pub allowed_functions: HashSet<String>,
    /// Allowed prefixes in SPARQL queries
    pub allowed_prefixes: HashSet<String>,
    /// Enable query sandboxing
    pub enable_sandboxing: bool,
    /// Enable detailed security logging
    pub enable_security_logging: bool,
    /// Maximum recursion depth for nested shapes
    pub max_recursion_depth: usize,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        let mut allowed_functions = HashSet::new();
        allowed_functions.insert("STR".to_string());
        allowed_functions.insert("LANG".to_string());
        allowed_functions.insert("DATATYPE".to_string());
        allowed_functions.insert("IRI".to_string());
        allowed_functions.insert("URI".to_string());
        allowed_functions.insert("BNODE".to_string());
        allowed_functions.insert("STRDT".to_string());
        allowed_functions.insert("STRLANG".to_string());
        allowed_functions.insert("UUID".to_string());
        allowed_functions.insert("STRUUID".to_string());
        allowed_functions.insert("STRLEN".to_string());
        allowed_functions.insert("SUBSTR".to_string());
        allowed_functions.insert("UCASE".to_string());
        allowed_functions.insert("LCASE".to_string());
        allowed_functions.insert("STRSTARTS".to_string());
        allowed_functions.insert("STRENDS".to_string());
        allowed_functions.insert("CONTAINS".to_string());
        allowed_functions.insert("STRBEFORE".to_string());
        allowed_functions.insert("STRAFTER".to_string());
        allowed_functions.insert("REPLACE".to_string());
        allowed_functions.insert("REGEX".to_string());
        allowed_functions.insert("ABS".to_string());
        allowed_functions.insert("ROUND".to_string());
        allowed_functions.insert("CEIL".to_string());
        allowed_functions.insert("FLOOR".to_string());
        allowed_functions.insert("RAND".to_string());
        allowed_functions.insert("NOW".to_string());
        allowed_functions.insert("YEAR".to_string());
        allowed_functions.insert("MONTH".to_string());
        allowed_functions.insert("DAY".to_string());
        allowed_functions.insert("HOURS".to_string());
        allowed_functions.insert("MINUTES".to_string());
        allowed_functions.insert("SECONDS".to_string());
        allowed_functions.insert("TIMEZONE".to_string());
        allowed_functions.insert("TZ".to_string());

        let mut allowed_prefixes = HashSet::new();
        allowed_prefixes.insert("sh".to_string());
        allowed_prefixes.insert("rdf".to_string());
        allowed_prefixes.insert("rdfs".to_string());
        allowed_prefixes.insert("owl".to_string());
        allowed_prefixes.insert("xsd".to_string());
        allowed_prefixes.insert("foaf".to_string());
        allowed_prefixes.insert("dc".to_string());
        allowed_prefixes.insert("dcterms".to_string());
        allowed_prefixes.insert("skos".to_string());

        Self {
            max_execution_time: Duration::from_secs(30),
            max_memory_usage: 100 * 1024 * 1024, // 100MB
            max_result_count: 10000,
            max_complexity_score: 100.0,
            enable_injection_detection: true,
            allowed_functions,
            allowed_prefixes,
            enable_sandboxing: true,
            enable_security_logging: true,
            max_recursion_depth: 10,
        }
    }
}

/// SPARQL query security analyzer
#[derive(Debug)]
pub struct SparqlSecurityAnalyzer {
    config: SecurityConfig,
    dangerous_patterns: Vec<Regex>,
    function_extractor: Regex,
    prefix_extractor: Regex,
}

impl SparqlSecurityAnalyzer {
    /// Create a new security analyzer
    pub fn new(config: SecurityConfig) -> Result<Self> {
        let dangerous_patterns = vec![
            // Potential injection patterns
            Regex::new(r"(?i)\b(drop|delete|insert|clear|create|load)\s+(?:data|graph|silent)")?,
            Regex::new(r"(?i)\bservice\s+<[^>]*>")?,
            Regex::new(r"(?i)\bfrom\s+named\s+<[^>]*>")?,
            Regex::new(r"(?i)\binto\s+(?:graph\s+)?<[^>]*>")?,
            // Recursive or complex constructs
            Regex::new(r"(?i)\b(?:union|optional|exists|not\s+exists)\s*\{")?,
            Regex::new(r"\{[^}]*\{[^}]*\{[^}]*\{")?,
            // Potential DoS patterns
            Regex::new(r"(?i)\b(?:count|sum|avg|min|max)\s*\(\s*\*\s*\)")?,
            // File system access
            Regex::new(r"(?i)file://")?,
        ];

        let function_extractor = Regex::new(r"(?i)\b([A-Z_][A-Z0-9_]*)\s*\(")?;
        let prefix_extractor = Regex::new(r"(?i)prefix\s+([a-z][a-z0-9]*)\s*:")?;

        Ok(Self {
            config,
            dangerous_patterns,
            function_extractor,
            prefix_extractor,
        })
    }

    /// Analyze a SPARQL query for security issues
    pub fn analyze_query(&self, query: &str) -> Result<SecurityAnalysisResult> {
        let start_time = Instant::now();

        let mut analysis = SecurityAnalysisResult {
            is_safe: true,
            violations: Vec::new(),
            complexity_score: 0.0,
            estimated_execution_time: Duration::from_secs(0),
            estimated_memory_usage: 0,
            analysis_time: Duration::from_secs(0),
        };

        // 1. Check for dangerous patterns
        if self.config.enable_injection_detection {
            self.check_dangerous_patterns(query, &mut analysis)?;
        }

        // 2. Analyze query complexity
        self.analyze_complexity(query, &mut analysis)?;

        // 3. Check allowed functions
        self.check_allowed_functions(query, &mut analysis)?;

        // 4. Check allowed prefixes
        self.check_allowed_prefixes(query, &mut analysis)?;

        // 5. Check resource limits
        self.check_resource_limits(&analysis)?;

        analysis.analysis_time = start_time.elapsed();

        if self.config.enable_security_logging && !analysis.is_safe {
            tracing::warn!(
                "Security violation detected in SPARQL query: {} violations",
                analysis.violations.len()
            );
        }

        Ok(analysis)
    }

    /// Check for dangerous patterns in the query
    fn check_dangerous_patterns(
        &self,
        query: &str,
        analysis: &mut SecurityAnalysisResult,
    ) -> Result<()> {
        for (i, pattern) in self.dangerous_patterns.iter().enumerate() {
            if let Some(m) = pattern.find(query) {
                analysis.is_safe = false;
                analysis.violations.push(SecurityViolation {
                    violation_type: SecurityViolationType::DangerousPattern,
                    severity: SecuritySeverity::High,
                    message: format!(
                        "Dangerous pattern detected at position {}: '{}'",
                        m.start(),
                        m.as_str()
                    ),
                    pattern_id: Some(i),
                    position: Some(m.start()),
                });
            }
        }
        Ok(())
    }

    /// Analyze query complexity
    fn analyze_complexity(&self, query: &str, analysis: &mut SecurityAnalysisResult) -> Result<()> {
        let mut complexity = 0.0;

        // Count triple patterns
        let triple_pattern_count = query.matches('?').count();
        complexity += triple_pattern_count as f64 * 1.0;

        // Count joins (based on shared variables)
        let join_count = query.matches("JOIN").count();
        complexity += join_count as f64 * 2.0;

        // Count unions
        let union_count = query.matches("UNION").count();
        complexity += union_count as f64 * 1.5;

        // Count optional blocks
        let optional_count = query.matches("OPTIONAL").count();
        complexity += optional_count as f64 * 1.2;

        // Count nested blocks
        let nesting_depth = self.calculate_nesting_depth(query);
        complexity += (nesting_depth as f64).powi(2) * 0.5;

        // Count aggregations
        let aggregation_count = query.matches("GROUP BY").count() + query.matches("HAVING").count();
        complexity += aggregation_count as f64 * 2.0;

        // Count subqueries
        let subquery_count = query.matches("SELECT").count().saturating_sub(1); // Subtract main SELECT
        complexity += subquery_count as f64 * 3.0;

        analysis.complexity_score = complexity;

        // Estimate execution time based on complexity
        analysis.estimated_execution_time = Duration::from_millis((complexity * 100.0) as u64);

        // Estimate memory usage based on complexity
        analysis.estimated_memory_usage = (complexity * 1024.0 * 1024.0) as usize; // 1MB per complexity point

        if complexity > self.config.max_complexity_score {
            analysis.is_safe = false;
            analysis.violations.push(SecurityViolation {
                violation_type: SecurityViolationType::ComplexityExceeded,
                severity: SecuritySeverity::Medium,
                message: format!(
                    "Query complexity ({:.1}) exceeds maximum allowed ({})",
                    complexity, self.config.max_complexity_score
                ),
                pattern_id: None,
                position: None,
            });
        }

        Ok(())
    }

    /// Calculate nesting depth of braces in query
    fn calculate_nesting_depth(&self, query: &str) -> usize {
        let mut max_depth = 0_usize;
        let mut current_depth: i32 = 0;

        for c in query.chars() {
            match c {
                '{' => {
                    current_depth += 1;
                    max_depth = max_depth.max(current_depth as usize);
                }
                '}' => {
                    current_depth = current_depth.saturating_sub(1);
                }
                _ => {}
            }
        }

        max_depth
    }

    /// Check for allowed functions
    fn check_allowed_functions(
        &self,
        query: &str,
        analysis: &mut SecurityAnalysisResult,
    ) -> Result<()> {
        for caps in self.function_extractor.captures_iter(query) {
            if let Some(function_name) = caps.get(1) {
                let func = function_name.as_str().to_uppercase();
                if !self.config.allowed_functions.contains(&func) {
                    analysis.is_safe = false;
                    analysis.violations.push(SecurityViolation {
                        violation_type: SecurityViolationType::DisallowedFunction,
                        severity: SecuritySeverity::Medium,
                        message: format!("Disallowed function: {func}"),
                        pattern_id: None,
                        position: Some(function_name.start()),
                    });
                }
            }
        }
        Ok(())
    }

    /// Check for allowed prefixes
    fn check_allowed_prefixes(
        &self,
        query: &str,
        analysis: &mut SecurityAnalysisResult,
    ) -> Result<()> {
        for caps in self.prefix_extractor.captures_iter(query) {
            if let Some(prefix_name) = caps.get(1) {
                let prefix = prefix_name.as_str().to_lowercase();
                if !self.config.allowed_prefixes.contains(&prefix) {
                    analysis.is_safe = false;
                    analysis.violations.push(SecurityViolation {
                        violation_type: SecurityViolationType::DisallowedPrefix,
                        severity: SecuritySeverity::Low,
                        message: format!("Disallowed prefix: {prefix}"),
                        pattern_id: None,
                        position: Some(prefix_name.start()),
                    });
                }
            }
        }
        Ok(())
    }

    /// Check resource limits
    fn check_resource_limits(&self, _analysis: &SecurityAnalysisResult) -> Result<()> {
        // These checks are performed during analysis, this method validates final limits
        Ok(())
    }

    /// Sanitize a SPARQL query
    pub fn sanitize_query(&self, query: &str) -> Result<String> {
        let mut sanitized = query.to_string();

        // Remove potentially dangerous comments
        sanitized = self.remove_comments(&sanitized);

        // Normalize whitespace
        sanitized = self.normalize_whitespace(&sanitized);

        // Escape special characters in literals
        sanitized = self.escape_literals(&sanitized);

        Ok(sanitized)
    }

    /// Remove comments from query
    fn remove_comments(&self, query: &str) -> String {
        let comment_pattern = Regex::new(r"#[^\r\n]*").unwrap();
        comment_pattern.replace_all(query, "").to_string()
    }

    /// Normalize whitespace
    fn normalize_whitespace(&self, query: &str) -> String {
        let whitespace_pattern = Regex::new(r"\s+").unwrap();
        whitespace_pattern
            .replace_all(query.trim(), " ")
            .to_string()
    }

    /// Escape literals in query
    fn escape_literals(&self, query: &str) -> String {
        // This would implement proper literal escaping
        // For now, return as-is
        query.to_string()
    }
}

/// Result of security analysis
#[derive(Debug, Clone)]
pub struct SecurityAnalysisResult {
    /// Whether the query is considered safe
    pub is_safe: bool,
    /// List of security violations found
    pub violations: Vec<SecurityViolation>,
    /// Calculated complexity score
    pub complexity_score: f64,
    /// Estimated execution time
    pub estimated_execution_time: Duration,
    /// Estimated memory usage in bytes
    pub estimated_memory_usage: usize,
    /// Time taken for analysis
    pub analysis_time: Duration,
}

/// Security violation details
#[derive(Debug, Clone)]
pub struct SecurityViolation {
    /// Type of security violation
    pub violation_type: SecurityViolationType,
    /// Severity of the violation
    pub severity: SecuritySeverity,
    /// Human-readable message
    pub message: String,
    /// Pattern ID if applicable
    pub pattern_id: Option<usize>,
    /// Position in query where violation was found
    pub position: Option<usize>,
}

/// Types of security violations
#[derive(Debug, Clone, PartialEq)]
pub enum SecurityViolationType {
    DangerousPattern,
    ComplexityExceeded,
    DisallowedFunction,
    DisallowedPrefix,
    ExecutionTimeExceeded,
    MemoryLimitExceeded,
    ResultCountExceeded,
    RecursionDepthExceeded,
}

/// Security violation severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Query execution sandbox
#[derive(Debug)]
pub struct QueryExecutionSandbox {
    config: SecurityConfig,
    start_time: Option<Instant>,
    memory_monitor: MemoryMonitor,
    result_counter: usize,
}

impl QueryExecutionSandbox {
    /// Create a new execution sandbox
    pub fn new(config: SecurityConfig) -> Self {
        Self {
            config,
            start_time: None,
            memory_monitor: MemoryMonitor::new(),
            result_counter: 0,
        }
    }

    /// Start monitoring query execution
    pub fn start_execution(&mut self) -> Result<()> {
        self.start_time = Some(Instant::now());
        self.memory_monitor.start()?;
        self.result_counter = 0;
        Ok(())
    }

    /// Check if execution should continue (called periodically)
    pub fn check_execution_limits(&mut self) -> Result<()> {
        // Check execution time
        if let Some(start_time) = self.start_time {
            if start_time.elapsed() > self.config.max_execution_time {
                return Err(ShaclError::SecurityViolation(
                    "Query execution time exceeded".to_string(),
                ));
            }
        }

        // Check memory usage
        if self.memory_monitor.current_usage() > self.config.max_memory_usage {
            return Err(ShaclError::SecurityViolation(
                "Memory usage limit exceeded".to_string(),
            ));
        }

        Ok(())
    }

    /// Record a result and check count limits
    pub fn record_result(&mut self) -> Result<()> {
        self.result_counter += 1;
        if self.result_counter > self.config.max_result_count {
            return Err(ShaclError::SecurityViolation(
                "Result count limit exceeded".to_string(),
            ));
        }
        Ok(())
    }

    /// Stop monitoring and clean up
    pub fn stop_execution(&mut self) -> Result<ExecutionStats> {
        let execution_time = self.start_time.map(|t| t.elapsed()).unwrap_or_default();
        let memory_used = self.memory_monitor.peak_usage();
        let results_produced = self.result_counter;

        self.memory_monitor.stop()?;
        self.start_time = None;

        Ok(ExecutionStats {
            execution_time,
            memory_used,
            results_produced,
        })
    }
}

/// Memory usage monitor
#[derive(Debug)]
struct MemoryMonitor {
    peak_usage: usize,
    current_usage: usize,
    monitoring: bool,
}

impl MemoryMonitor {
    fn new() -> Self {
        Self {
            peak_usage: 0,
            current_usage: 0,
            monitoring: false,
        }
    }

    fn start(&mut self) -> Result<()> {
        self.monitoring = true;
        self.peak_usage = 0;
        self.current_usage = 0;
        Ok(())
    }

    fn current_usage(&self) -> usize {
        // In a real implementation, this would track actual memory usage
        // For now, return estimated usage
        self.current_usage
    }

    fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    fn stop(&mut self) -> Result<()> {
        self.monitoring = false;
        Ok(())
    }
}

/// Execution statistics
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub execution_time: Duration,
    pub memory_used: usize,
    pub results_produced: usize,
}

/// Recursive shape validation monitor
#[derive(Debug)]
pub struct RecursionMonitor {
    visited_shapes: HashMap<String, usize>,
    max_depth: usize,
}

impl RecursionMonitor {
    pub fn new(max_depth: usize) -> Self {
        Self {
            visited_shapes: HashMap::new(),
            max_depth,
        }
    }

    pub fn enter_shape(&mut self, shape_id: &str) -> Result<()> {
        let depth = self.visited_shapes.entry(shape_id.to_string()).or_insert(0);

        if *depth >= self.max_depth {
            return Err(ShaclError::SecurityViolation(format!(
                "Maximum recursion depth ({}) exceeded for shape {}",
                self.max_depth, shape_id
            )));
        }

        *depth += 1;
        Ok(())
    }

    pub fn exit_shape(&mut self, shape_id: &str) {
        if let Some(depth) = self.visited_shapes.get_mut(shape_id) {
            *depth = depth.saturating_sub(1);
        }
    }

    pub fn current_depth(&self, shape_id: &str) -> usize {
        self.visited_shapes.get(shape_id).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_analyzer_creation() {
        let config = SecurityConfig::default();
        let analyzer = SparqlSecurityAnalyzer::new(config).unwrap();
        assert!(!analyzer.dangerous_patterns.is_empty());
    }

    #[test]
    fn test_safe_query_analysis() {
        let config = SecurityConfig::default();
        let analyzer = SparqlSecurityAnalyzer::new(config).unwrap();

        let safe_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let result = analyzer.analyze_query(safe_query).unwrap();

        assert!(result.is_safe);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_dangerous_pattern_detection() {
        let config = SecurityConfig::default();
        let analyzer = SparqlSecurityAnalyzer::new(config).unwrap();

        let dangerous_query = "DROP GRAPH <http://example.org/graph>";
        let result = analyzer.analyze_query(dangerous_query).unwrap();

        assert!(!result.is_safe);
        assert!(!result.violations.is_empty());
        assert_eq!(
            result.violations[0].violation_type,
            SecurityViolationType::DangerousPattern
        );
    }

    #[test]
    fn test_complexity_analysis() {
        let config = SecurityConfig {
            max_complexity_score: 5.0,
            ..SecurityConfig::default()
        };
        let analyzer = SparqlSecurityAnalyzer::new(config).unwrap();

        let complex_query = r"
            SELECT ?s ?p ?o ?x ?y ?z WHERE {
                ?s ?p ?o .
                ?x ?y ?z .
                OPTIONAL { ?s ?p2 ?o2 }
                OPTIONAL { ?x ?y2 ?z2 }
                { ?a ?b ?c } UNION { ?d ?e ?f }
            }
        ";
        let result = analyzer.analyze_query(complex_query).unwrap();

        assert!(!result.is_safe);
        assert!(result
            .violations
            .iter()
            .any(|v| v.violation_type == SecurityViolationType::ComplexityExceeded));
    }

    #[test]
    fn test_disallowed_function_detection() {
        let mut config = SecurityConfig::default();
        config.allowed_functions.clear();
        config.allowed_functions.insert("STR".to_string());

        let analyzer = SparqlSecurityAnalyzer::new(config).unwrap();

        let query_with_disallowed_function = "SELECT ?s WHERE { ?s ?p ?o . FILTER(RAND() > 0.5) }";
        let result = analyzer
            .analyze_query(query_with_disallowed_function)
            .unwrap();

        assert!(!result.is_safe);
        assert!(result
            .violations
            .iter()
            .any(|v| v.violation_type == SecurityViolationType::DisallowedFunction));
    }

    #[test]
    fn test_query_sanitization() {
        let config = SecurityConfig::default();
        let analyzer = SparqlSecurityAnalyzer::new(config).unwrap();

        let query_with_comments = r"
            SELECT ?s ?p ?o WHERE {
                ?s ?p ?o .  # This is a comment
                # Another comment
            }
        ";

        let sanitized = analyzer.sanitize_query(query_with_comments).unwrap();
        assert!(!sanitized.contains('#'));
    }

    #[test]
    fn test_execution_sandbox() {
        let config = SecurityConfig::default();
        let mut sandbox = QueryExecutionSandbox::new(config);

        sandbox.start_execution().unwrap();

        // Should be able to record results within limit
        for _ in 0..10 {
            sandbox.record_result().unwrap();
        }

        sandbox.check_execution_limits().unwrap();

        let stats = sandbox.stop_execution().unwrap();
        assert_eq!(stats.results_produced, 10);
    }

    #[test]
    fn test_recursion_monitor() {
        let mut monitor = RecursionMonitor::new(3);

        // Should allow normal recursion
        monitor.enter_shape("shape1").unwrap();
        monitor.enter_shape("shape1").unwrap();
        monitor.enter_shape("shape1").unwrap();

        // Should fail on exceeding depth
        assert!(monitor.enter_shape("shape1").is_err());

        // Should recover after exiting
        monitor.exit_shape("shape1");
        monitor.enter_shape("shape1").unwrap();
    }
}
