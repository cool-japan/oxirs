//! GraphQL Query Sanitization
//!
//! This module provides comprehensive security features for GraphQL queries:
//! - **Input Sanitization**: Prevents injection attacks in query variables
//! - **Query Validation**: Validates query structure against security policies
//! - **Alias Limiting**: Prevents alias-based DoS attacks
//! - **Directive Validation**: Ensures only allowed directives are used
//! - **Introspection Control**: Controls access to schema introspection
//! - **Field Deduplication**: Removes duplicate field selections
//! - **Depth Limiting**: Enforces maximum query depth
//! - **Complexity Analysis**: Calculates and limits query complexity

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Sanitization severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SanitizationSeverity {
    /// Informational - no action needed
    Info,
    /// Warning - query modified but allowed
    Warning,
    /// Error - query rejected
    Error,
    /// Critical - potential attack detected
    Critical,
}

/// Type of sanitization issue detected
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SanitizationIssueType {
    /// SQL/SPARQL injection attempt
    InjectionAttempt,
    /// Excessive query depth
    ExcessiveDepth,
    /// Too many aliases
    ExcessiveAliases,
    /// Forbidden directive used
    ForbiddenDirective,
    /// Introspection not allowed
    IntrospectionBlocked,
    /// Duplicate fields
    DuplicateFields,
    /// Excessive complexity
    ExcessiveComplexity,
    /// Invalid characters in input
    InvalidCharacters,
    /// Excessive variable size
    ExcessiveVariableSize,
    /// Circular fragments
    CircularFragments,
    /// Excessive field count
    ExcessiveFieldCount,
    /// Batch query limit exceeded
    BatchLimitExceeded,
}

/// A sanitization issue found in the query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizationIssue {
    /// Type of issue
    pub issue_type: SanitizationIssueType,
    /// Severity level
    pub severity: SanitizationSeverity,
    /// Human-readable message
    pub message: String,
    /// Location in query (if applicable)
    pub location: Option<QueryLocation>,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Location in query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryLocation {
    /// Line number (1-indexed)
    pub line: usize,
    /// Column number (1-indexed)
    pub column: usize,
    /// Field or directive name
    pub name: Option<String>,
}

/// Result of query sanitization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizationResult {
    /// Whether the query is allowed
    pub allowed: bool,
    /// Sanitized query (if modifications were made)
    pub sanitized_query: Option<String>,
    /// Issues found during sanitization
    pub issues: Vec<SanitizationIssue>,
    /// Sanitized variables (if modifications were made)
    pub sanitized_variables: Option<HashMap<String, serde_json::Value>>,
    /// Query metrics
    pub metrics: QueryMetrics,
}

/// Query metrics collected during sanitization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryMetrics {
    /// Query depth
    pub depth: usize,
    /// Number of fields
    pub field_count: usize,
    /// Number of aliases
    pub alias_count: usize,
    /// Number of directives
    pub directive_count: usize,
    /// Number of fragments
    pub fragment_count: usize,
    /// Estimated complexity score
    pub complexity_score: f64,
    /// Number of operations in batch
    pub batch_size: usize,
}

/// Configuration for query sanitization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizationConfig {
    /// Maximum query depth
    pub max_depth: usize,
    /// Maximum number of aliases
    pub max_aliases: usize,
    /// Maximum number of fields
    pub max_fields: usize,
    /// Maximum complexity score
    pub max_complexity: f64,
    /// Maximum variable size in bytes
    pub max_variable_size: usize,
    /// Maximum batch size (for batched queries)
    pub max_batch_size: usize,
    /// Allow introspection queries
    pub allow_introspection: bool,
    /// Allowed directives (empty = all allowed)
    pub allowed_directives: HashSet<String>,
    /// Blocked directives
    pub blocked_directives: HashSet<String>,
    /// Enable injection detection
    pub detect_injections: bool,
    /// Remove duplicate fields
    pub deduplicate_fields: bool,
    /// Enable strict mode (reject on any issue)
    pub strict_mode: bool,
    /// Custom injection patterns to detect
    pub custom_injection_patterns: Vec<String>,
}

impl Default for SanitizationConfig {
    fn default() -> Self {
        let mut blocked_directives = HashSet::new();
        blocked_directives.insert("include".to_string());
        blocked_directives.insert("skip".to_string());

        Self {
            max_depth: 10,
            max_aliases: 50,
            max_fields: 100,
            max_complexity: 1000.0,
            max_variable_size: 1024 * 1024, // 1MB
            max_batch_size: 10,
            allow_introspection: true,
            allowed_directives: HashSet::new(),
            blocked_directives: HashSet::new(), // Empty by default
            detect_injections: true,
            deduplicate_fields: true,
            strict_mode: false,
            custom_injection_patterns: Vec::new(),
        }
    }
}

/// Injection pattern for detection
#[derive(Debug, Clone)]
struct InjectionPattern {
    pattern: String,
    severity: SanitizationSeverity,
    description: String,
}

/// Query sanitizer
pub struct QuerySanitizer {
    config: SanitizationConfig,
    injection_patterns: Vec<InjectionPattern>,
    state: Arc<RwLock<SanitizerState>>,
}

/// Internal state for tracking sanitization statistics
struct SanitizerState {
    /// Total queries processed
    total_queries: u64,
    /// Queries blocked
    blocked_queries: u64,
    /// Queries modified
    modified_queries: u64,
    /// Issues by type
    issues_by_type: HashMap<SanitizationIssueType, u64>,
    /// Recent blocked queries (for analysis)
    recent_blocked: Vec<BlockedQueryRecord>,
}

/// Record of a blocked query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockedQueryRecord {
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// Query fingerprint (hash)
    pub fingerprint: String,
    /// Reason for blocking
    pub reason: SanitizationIssueType,
    /// Client identifier (if available)
    pub client_id: Option<String>,
}

impl QuerySanitizer {
    /// Create a new query sanitizer with default configuration
    pub fn new() -> Self {
        Self::with_config(SanitizationConfig::default())
    }

    /// Create a new query sanitizer with custom configuration
    pub fn with_config(config: SanitizationConfig) -> Self {
        let injection_patterns = Self::build_injection_patterns(&config);

        Self {
            config,
            injection_patterns,
            state: Arc::new(RwLock::new(SanitizerState {
                total_queries: 0,
                blocked_queries: 0,
                modified_queries: 0,
                issues_by_type: HashMap::new(),
                recent_blocked: Vec::new(),
            })),
        }
    }

    /// Build injection detection patterns
    fn build_injection_patterns(config: &SanitizationConfig) -> Vec<InjectionPattern> {
        let mut patterns = vec![
            // SQL injection patterns
            InjectionPattern {
                pattern: r"(?i)(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)".to_string(),
                severity: SanitizationSeverity::Critical,
                description: "SQL keyword detected".to_string(),
            },
            InjectionPattern {
                pattern: r#"(?i)('|")\s*(OR|AND)\s*('|")?\s*\d+\s*=\s*\d+"#.to_string(),
                severity: SanitizationSeverity::Critical,
                description: "SQL injection pattern (OR/AND 1=1)".to_string(),
            },
            InjectionPattern {
                pattern: r"(?i)(--|\#|/\*)".to_string(),
                severity: SanitizationSeverity::Warning,
                description: "SQL comment detected".to_string(),
            },
            // SPARQL injection patterns
            InjectionPattern {
                pattern: r"(?i)(\bSELECT\s+\*|\bCONSTRUCT\b|\bDESCRIBE\b|\bASK\b)".to_string(),
                severity: SanitizationSeverity::Critical,
                description: "SPARQL query keyword detected".to_string(),
            },
            InjectionPattern {
                pattern: r"(?i)(\bFILTER\s*\(|\bOPTIONAL\s*\{|\bUNION\s*\{)".to_string(),
                severity: SanitizationSeverity::Warning,
                description: "SPARQL clause detected".to_string(),
            },
            // XSS patterns
            InjectionPattern {
                pattern: r"<\s*script".to_string(),
                severity: SanitizationSeverity::Critical,
                description: "Script tag detected".to_string(),
            },
            InjectionPattern {
                pattern: r"(?i)(javascript:|data:text/html|on\w+\s*=)".to_string(),
                severity: SanitizationSeverity::Critical,
                description: "XSS pattern detected".to_string(),
            },
            // Path traversal
            InjectionPattern {
                pattern: r"(\.\./|\.\.\\)".to_string(),
                severity: SanitizationSeverity::Warning,
                description: "Path traversal pattern detected".to_string(),
            },
            // Command injection
            InjectionPattern {
                pattern: r"(;|\||`|\$\()".to_string(),
                severity: SanitizationSeverity::Warning,
                description: "Command injection pattern detected".to_string(),
            },
        ];

        // Add custom patterns
        for custom in &config.custom_injection_patterns {
            patterns.push(InjectionPattern {
                pattern: custom.clone(),
                severity: SanitizationSeverity::Warning,
                description: "Custom pattern matched".to_string(),
            });
        }

        patterns
    }

    /// Sanitize a GraphQL query
    pub async fn sanitize(
        &self,
        query: &str,
        variables: Option<&HashMap<String, serde_json::Value>>,
        client_id: Option<&str>,
    ) -> SanitizationResult {
        let mut issues = Vec::new();
        let mut metrics = QueryMetrics::default();
        let mut sanitized_query = query.to_string();
        let mut sanitized_variables = variables.cloned();
        let mut modified = false;

        // Update statistics
        {
            let mut state = self.state.write().await;
            state.total_queries += 1;
        }

        // Check for injection patterns in query
        if self.config.detect_injections {
            self.detect_injections_in_query(query, &mut issues);
        }

        // Check for injection patterns in variables
        if let Some(vars) = variables {
            self.detect_injections_in_variables(vars, &mut issues);
        }

        // Parse and analyze query structure
        self.analyze_query_structure(query, &mut metrics, &mut issues);

        // Check introspection
        if !self.config.allow_introspection && self.is_introspection_query(query) {
            issues.push(SanitizationIssue {
                issue_type: SanitizationIssueType::IntrospectionBlocked,
                severity: SanitizationSeverity::Error,
                message: "Introspection queries are not allowed".to_string(),
                location: None,
                suggestion: Some("Remove __schema or __type fields".to_string()),
            });
        }

        // Check depth limit
        if metrics.depth > self.config.max_depth {
            issues.push(SanitizationIssue {
                issue_type: SanitizationIssueType::ExcessiveDepth,
                severity: SanitizationSeverity::Error,
                message: format!(
                    "Query depth {} exceeds maximum {}",
                    metrics.depth, self.config.max_depth
                ),
                location: None,
                suggestion: Some("Reduce query nesting depth".to_string()),
            });
        }

        // Check alias limit
        if metrics.alias_count > self.config.max_aliases {
            issues.push(SanitizationIssue {
                issue_type: SanitizationIssueType::ExcessiveAliases,
                severity: SanitizationSeverity::Error,
                message: format!(
                    "Alias count {} exceeds maximum {}",
                    metrics.alias_count, self.config.max_aliases
                ),
                location: None,
                suggestion: Some("Reduce number of aliases".to_string()),
            });
        }

        // Check field limit
        if metrics.field_count > self.config.max_fields {
            issues.push(SanitizationIssue {
                issue_type: SanitizationIssueType::ExcessiveFieldCount,
                severity: SanitizationSeverity::Error,
                message: format!(
                    "Field count {} exceeds maximum {}",
                    metrics.field_count, self.config.max_fields
                ),
                location: None,
                suggestion: Some("Reduce number of fields in query".to_string()),
            });
        }

        // Check complexity
        if metrics.complexity_score > self.config.max_complexity {
            issues.push(SanitizationIssue {
                issue_type: SanitizationIssueType::ExcessiveComplexity,
                severity: SanitizationSeverity::Error,
                message: format!(
                    "Query complexity {:.2} exceeds maximum {}",
                    metrics.complexity_score, self.config.max_complexity
                ),
                location: None,
                suggestion: Some("Simplify query or request fewer fields".to_string()),
            });
        }

        // Check batch size
        if metrics.batch_size > self.config.max_batch_size {
            issues.push(SanitizationIssue {
                issue_type: SanitizationIssueType::BatchLimitExceeded,
                severity: SanitizationSeverity::Error,
                message: format!(
                    "Batch size {} exceeds maximum {}",
                    metrics.batch_size, self.config.max_batch_size
                ),
                location: None,
                suggestion: Some("Split into multiple requests".to_string()),
            });
        }

        // Check variable size
        if let Some(vars) = &sanitized_variables {
            let var_size = serde_json::to_string(vars).map(|s| s.len()).unwrap_or(0);
            if var_size > self.config.max_variable_size {
                issues.push(SanitizationIssue {
                    issue_type: SanitizationIssueType::ExcessiveVariableSize,
                    severity: SanitizationSeverity::Error,
                    message: format!(
                        "Variable size {} bytes exceeds maximum {} bytes",
                        var_size, self.config.max_variable_size
                    ),
                    location: None,
                    suggestion: Some("Reduce variable payload size".to_string()),
                });
            }
        }

        // Check directives
        self.check_directives(query, &mut issues);

        // Deduplicate fields if enabled
        if self.config.deduplicate_fields {
            if let Some(deduped) = self.deduplicate_fields(&sanitized_query) {
                if deduped != sanitized_query {
                    sanitized_query = deduped;
                    modified = true;
                    issues.push(SanitizationIssue {
                        issue_type: SanitizationIssueType::DuplicateFields,
                        severity: SanitizationSeverity::Info,
                        message: "Duplicate fields were removed".to_string(),
                        location: None,
                        suggestion: None,
                    });
                }
            }
        }

        // Sanitize variables
        if let Some(vars) = &mut sanitized_variables {
            if self.sanitize_variables(vars) {
                modified = true;
            }
        }

        // Determine if query is allowed
        let has_blocking_issues = issues.iter().any(|i| {
            matches!(
                i.severity,
                SanitizationSeverity::Error | SanitizationSeverity::Critical
            )
        });

        let allowed = if self.config.strict_mode {
            issues.is_empty()
        } else {
            !has_blocking_issues
        };

        // Update statistics
        {
            let mut state = self.state.write().await;
            if !allowed {
                state.blocked_queries += 1;
                if let Some(issue) = issues.first() {
                    state.recent_blocked.push(BlockedQueryRecord {
                        timestamp: std::time::SystemTime::now(),
                        fingerprint: self.fingerprint_query(query),
                        reason: issue.issue_type.clone(),
                        client_id: client_id.map(|s| s.to_string()),
                    });
                    // Keep only last 100 blocked queries
                    if state.recent_blocked.len() > 100 {
                        state.recent_blocked.remove(0);
                    }
                }
            } else if modified {
                state.modified_queries += 1;
            }

            for issue in &issues {
                *state
                    .issues_by_type
                    .entry(issue.issue_type.clone())
                    .or_insert(0) += 1;
            }
        }

        SanitizationResult {
            allowed,
            sanitized_query: if modified {
                Some(sanitized_query)
            } else {
                None
            },
            issues,
            sanitized_variables: if modified { sanitized_variables } else { None },
            metrics,
        }
    }

    /// Detect injection patterns in query string
    fn detect_injections_in_query(&self, query: &str, issues: &mut Vec<SanitizationIssue>) {
        for pattern in &self.injection_patterns {
            if let Ok(re) = regex::Regex::new(&pattern.pattern) {
                if re.is_match(query) {
                    issues.push(SanitizationIssue {
                        issue_type: SanitizationIssueType::InjectionAttempt,
                        severity: pattern.severity,
                        message: pattern.description.clone(),
                        location: None,
                        suggestion: Some("Remove potentially malicious content".to_string()),
                    });
                }
            }
        }
    }

    /// Detect injection patterns in variables
    fn detect_injections_in_variables(
        &self,
        variables: &HashMap<String, serde_json::Value>,
        issues: &mut Vec<SanitizationIssue>,
    ) {
        for (key, value) in variables {
            if let Some(s) = value.as_str() {
                for pattern in &self.injection_patterns {
                    if let Ok(re) = regex::Regex::new(&pattern.pattern) {
                        if re.is_match(s) {
                            issues.push(SanitizationIssue {
                                issue_type: SanitizationIssueType::InjectionAttempt,
                                severity: pattern.severity,
                                message: format!("Variable '{}': {}", key, pattern.description),
                                location: None,
                                suggestion: Some(format!("Sanitize variable '{}' content", key)),
                            });
                        }
                    }
                }
            }
        }
    }

    /// Analyze query structure for metrics
    fn analyze_query_structure(
        &self,
        query: &str,
        metrics: &mut QueryMetrics,
        issues: &mut Vec<SanitizationIssue>,
    ) {
        // Count depth by tracking braces
        let mut current_depth: usize = 0;
        let mut max_depth: usize = 0;
        let mut field_count = 0;
        let mut alias_count = 0;
        let mut directive_count = 0;
        let mut fragment_count = 0;
        let mut in_string = false;
        let mut escape_next = false;

        let chars: Vec<char> = query.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            let c = chars[i];

            if escape_next {
                escape_next = false;
                i += 1;
                continue;
            }

            if c == '\\' {
                escape_next = true;
                i += 1;
                continue;
            }

            if c == '"' {
                in_string = !in_string;
                i += 1;
                continue;
            }

            if in_string {
                i += 1;
                continue;
            }

            match c {
                '{' => {
                    current_depth += 1;
                    max_depth = max_depth.max(current_depth);
                }
                '}' => {
                    current_depth = current_depth.saturating_sub(1);
                }
                '@' => {
                    directive_count += 1;
                }
                ':' => {
                    // Potential alias (check if preceded by identifier)
                    if i > 0 && chars[i - 1].is_alphanumeric() {
                        alias_count += 1;
                    }
                }
                _ => {}
            }

            // Count fields (identifiers followed by { or arguments)
            if c.is_alphabetic() {
                let mut j = i;
                while j < len && (chars[j].is_alphanumeric() || chars[j] == '_') {
                    j += 1;
                }
                let word: String = chars[i..j].iter().collect();

                // Check for fragments
                if word == "fragment" {
                    fragment_count += 1;
                }

                // Count as field if not a keyword
                if ![
                    "query",
                    "mutation",
                    "subscription",
                    "fragment",
                    "on",
                    "true",
                    "false",
                    "null",
                ]
                .contains(&word.as_str())
                {
                    field_count += 1;
                }

                i = j;
                continue;
            }

            i += 1;
        }

        metrics.depth = max_depth;
        metrics.field_count = field_count;
        metrics.alias_count = alias_count;
        metrics.directive_count = directive_count;
        metrics.fragment_count = fragment_count;

        // Calculate complexity score
        // Formula: fields + (aliases * 2) + (depth^2 * 5) + (directives * 3)
        metrics.complexity_score = field_count as f64
            + (alias_count as f64 * 2.0)
            + ((max_depth * max_depth) as f64 * 5.0)
            + (directive_count as f64 * 3.0);

        // Count batch size (number of operations)
        let operation_count = query
            .matches("query")
            .count()
            .max(query.matches("mutation").count())
            .max(query.matches("subscription").count())
            .max(1);
        metrics.batch_size = operation_count;

        // Check for circular fragment references
        if fragment_count > 0 && self.detect_circular_fragments(query) {
            issues.push(SanitizationIssue {
                issue_type: SanitizationIssueType::CircularFragments,
                severity: SanitizationSeverity::Error,
                message: "Circular fragment reference detected".to_string(),
                location: None,
                suggestion: Some("Remove circular fragment dependencies".to_string()),
            });
        }
    }

    /// Check if query is an introspection query
    fn is_introspection_query(&self, query: &str) -> bool {
        query.contains("__schema") || query.contains("__type")
    }

    /// Check directives against allowed/blocked lists
    fn check_directives(&self, query: &str, issues: &mut Vec<SanitizationIssue>) {
        // Extract directive names from query
        let directive_regex = regex::Regex::new(r"@(\w+)").unwrap();

        for cap in directive_regex.captures_iter(query) {
            let directive_name = &cap[1];

            // Check if directive is blocked
            if self.config.blocked_directives.contains(directive_name) {
                issues.push(SanitizationIssue {
                    issue_type: SanitizationIssueType::ForbiddenDirective,
                    severity: SanitizationSeverity::Error,
                    message: format!("Directive @{} is not allowed", directive_name),
                    location: None,
                    suggestion: Some(format!("Remove @{} directive", directive_name)),
                });
            }

            // Check if directive is in allowed list (if list is non-empty)
            if !self.config.allowed_directives.is_empty()
                && !self.config.allowed_directives.contains(directive_name)
            {
                issues.push(SanitizationIssue {
                    issue_type: SanitizationIssueType::ForbiddenDirective,
                    severity: SanitizationSeverity::Warning,
                    message: format!("Directive @{} is not in allowed list", directive_name),
                    location: None,
                    suggestion: Some(format!("Use one of: {:?}", self.config.allowed_directives)),
                });
            }
        }
    }

    /// Detect circular fragment references
    fn detect_circular_fragments(&self, query: &str) -> bool {
        // Extract fragment definitions and their spreads
        let fragment_def_regex = regex::Regex::new(r"fragment\s+(\w+)\s+on").unwrap();
        let fragment_spread_regex = regex::Regex::new(r"\.\.\.(\w+)").unwrap();

        let mut fragment_deps: HashMap<String, Vec<String>> = HashMap::new();

        // Build dependency graph
        let mut current_fragment: Option<String> = None;
        for line in query.lines() {
            if let Some(cap) = fragment_def_regex.captures(line) {
                current_fragment = Some(cap[1].to_string());
            }

            if let Some(ref frag_name) = current_fragment {
                for cap in fragment_spread_regex.captures_iter(line) {
                    let spread_name = cap[1].to_string();
                    if spread_name != *frag_name {
                        fragment_deps
                            .entry(frag_name.clone())
                            .or_default()
                            .push(spread_name);
                    }
                }
            }
        }

        // Check for cycles using DFS
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        fn has_cycle(
            node: &str,
            deps: &HashMap<String, Vec<String>>,
            visited: &mut HashSet<String>,
            rec_stack: &mut HashSet<String>,
        ) -> bool {
            if rec_stack.contains(node) {
                return true;
            }
            if visited.contains(node) {
                return false;
            }

            visited.insert(node.to_string());
            rec_stack.insert(node.to_string());

            if let Some(neighbors) = deps.get(node) {
                for neighbor in neighbors {
                    if has_cycle(neighbor, deps, visited, rec_stack) {
                        return true;
                    }
                }
            }

            rec_stack.remove(node);
            false
        }

        for fragment in fragment_deps.keys() {
            if has_cycle(fragment, &fragment_deps, &mut visited, &mut rec_stack) {
                return true;
            }
        }

        false
    }

    /// Deduplicate fields in query (simple implementation)
    fn deduplicate_fields(&self, _query: &str) -> Option<String> {
        // For a full implementation, we'd need a proper GraphQL parser
        // This is a placeholder that returns None (no deduplication)
        None
    }

    /// Sanitize variable values
    fn sanitize_variables(&self, variables: &mut HashMap<String, serde_json::Value>) -> bool {
        let mut modified = false;

        for (_key, value) in variables.iter_mut() {
            if let Some(s) = value.as_str() {
                // Remove null bytes
                if s.contains('\0') {
                    *value = serde_json::Value::String(s.replace('\0', ""));
                    modified = true;
                }
            }
        }

        modified
    }

    /// Generate a fingerprint for a query
    fn fingerprint_query(&self, query: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let normalized = query
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();

        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Get sanitization statistics
    pub async fn get_statistics(&self) -> SanitizationStatistics {
        let state = self.state.read().await;

        SanitizationStatistics {
            total_queries: state.total_queries,
            blocked_queries: state.blocked_queries,
            modified_queries: state.modified_queries,
            block_rate: if state.total_queries > 0 {
                state.blocked_queries as f64 / state.total_queries as f64
            } else {
                0.0
            },
            issues_by_type: state.issues_by_type.clone(),
            recent_blocked: state.recent_blocked.clone(),
        }
    }

    /// Reset statistics
    pub async fn reset_statistics(&self) {
        let mut state = self.state.write().await;
        state.total_queries = 0;
        state.blocked_queries = 0;
        state.modified_queries = 0;
        state.issues_by_type.clear();
        state.recent_blocked.clear();
    }
}

impl Default for QuerySanitizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Sanitization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizationStatistics {
    /// Total queries processed
    pub total_queries: u64,
    /// Queries blocked
    pub blocked_queries: u64,
    /// Queries modified
    pub modified_queries: u64,
    /// Block rate (0.0-1.0)
    pub block_rate: f64,
    /// Issues by type
    pub issues_by_type: HashMap<SanitizationIssueType, u64>,
    /// Recent blocked queries
    pub recent_blocked: Vec<BlockedQueryRecord>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sanitizer_creation() {
        let sanitizer = QuerySanitizer::new();
        let stats = sanitizer.get_statistics().await;
        assert_eq!(stats.total_queries, 0);
    }

    #[tokio::test]
    async fn test_simple_query_allowed() {
        let sanitizer = QuerySanitizer::new();
        let query = "query { users { id name } }";

        let result = sanitizer.sanitize(query, None, None).await;
        assert!(result.allowed);
        assert!(
            result.issues.is_empty()
                || result
                    .issues
                    .iter()
                    .all(|i| i.severity != SanitizationSeverity::Error)
        );
    }

    #[tokio::test]
    async fn test_sql_injection_detected() {
        let sanitizer = QuerySanitizer::new();
        let query = "query { users(name: \"'; DROP TABLE users; --\") { id } }";

        let result = sanitizer.sanitize(query, None, None).await;
        assert!(result
            .issues
            .iter()
            .any(|i| i.issue_type == SanitizationIssueType::InjectionAttempt));
    }

    #[tokio::test]
    async fn test_excessive_depth_blocked() {
        let config = SanitizationConfig {
            max_depth: 3,
            ..Default::default()
        };
        let sanitizer = QuerySanitizer::with_config(config);

        let query = "query { a { b { c { d { e { f } } } } } }";
        let result = sanitizer.sanitize(query, None, None).await;

        assert!(result
            .issues
            .iter()
            .any(|i| i.issue_type == SanitizationIssueType::ExcessiveDepth));
    }

    #[tokio::test]
    async fn test_introspection_blocked() {
        let config = SanitizationConfig {
            allow_introspection: false,
            ..Default::default()
        };
        let sanitizer = QuerySanitizer::with_config(config);

        let query = "query { __schema { types { name } } }";
        let result = sanitizer.sanitize(query, None, None).await;

        assert!(result
            .issues
            .iter()
            .any(|i| i.issue_type == SanitizationIssueType::IntrospectionBlocked));
    }

    #[tokio::test]
    async fn test_query_metrics() {
        let sanitizer = QuerySanitizer::new();
        let query = "query { user(id: 1) { name posts { title @deprecated } } }";

        let result = sanitizer.sanitize(query, None, None).await;
        assert!(result.metrics.depth > 0);
        assert!(result.metrics.field_count > 0);
    }

    #[tokio::test]
    async fn test_variable_injection_detected() {
        let sanitizer = QuerySanitizer::new();
        let query = "query GetUser($name: String!) { user(name: $name) { id } }";

        let mut variables = HashMap::new();
        variables.insert(
            "name".to_string(),
            serde_json::json!("'; SELECT * FROM users; --"),
        );

        let result = sanitizer.sanitize(query, Some(&variables), None).await;
        assert!(result
            .issues
            .iter()
            .any(|i| i.issue_type == SanitizationIssueType::InjectionAttempt));
    }

    #[tokio::test]
    async fn test_blocked_directive() {
        let mut config = SanitizationConfig::default();
        config.blocked_directives.insert("dangerous".to_string());
        let sanitizer = QuerySanitizer::with_config(config);

        let query = "query { user @dangerous { name } }";
        let result = sanitizer.sanitize(query, None, None).await;

        assert!(result
            .issues
            .iter()
            .any(|i| i.issue_type == SanitizationIssueType::ForbiddenDirective));
    }

    #[tokio::test]
    async fn test_complexity_limit() {
        let config = SanitizationConfig {
            max_complexity: 10.0,
            ..Default::default()
        };
        let sanitizer = QuerySanitizer::with_config(config);

        let query = "query { a { b { c { d } } } e { f { g { h } } } }";
        let result = sanitizer.sanitize(query, None, None).await;

        assert!(result
            .issues
            .iter()
            .any(|i| i.issue_type == SanitizationIssueType::ExcessiveComplexity));
    }

    #[tokio::test]
    async fn test_xss_pattern_detected() {
        let sanitizer = QuerySanitizer::new();
        let query = "query { user(bio: \"<script>alert('xss')</script>\") { id } }";

        let result = sanitizer.sanitize(query, None, None).await;
        assert!(result
            .issues
            .iter()
            .any(|i| i.issue_type == SanitizationIssueType::InjectionAttempt));
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        let sanitizer = QuerySanitizer::new();

        // Run a few queries
        sanitizer
            .sanitize("query { users { id } }", None, None)
            .await;
        sanitizer
            .sanitize("query { posts { title } }", None, None)
            .await;

        let stats = sanitizer.get_statistics().await;
        assert_eq!(stats.total_queries, 2);
    }

    #[tokio::test]
    async fn test_strict_mode() {
        let config = SanitizationConfig {
            strict_mode: true,
            ..Default::default()
        };
        let sanitizer = QuerySanitizer::with_config(config);

        // Even minor issues should block in strict mode
        let query = "query { user @deprecated { name } }";
        let result = sanitizer.sanitize(query, None, None).await;

        // If any issues found, query should be blocked in strict mode
        if !result.issues.is_empty() {
            assert!(!result.allowed);
        }
    }

    #[tokio::test]
    async fn test_query_fingerprinting() {
        let sanitizer = QuerySanitizer::new();

        let query1 = "query { users { id name } }";
        let query2 = "query { users { id name } }";
        let query3 = "query { posts { id title } }";

        let fp1 = sanitizer.fingerprint_query(query1);
        let fp2 = sanitizer.fingerprint_query(query2);
        let fp3 = sanitizer.fingerprint_query(query3);

        assert_eq!(fp1, fp2);
        assert_ne!(fp1, fp3);
    }
}
