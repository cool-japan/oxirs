//! Content Security Policy (CSP) for GraphQL Server
//!
//! This module implements Content Security Policy headers to protect the GraphQL server
//! and clients from various attacks:
//! - **XSS Protection**: Prevents cross-site scripting attacks
//! - **Clickjacking Prevention**: Prevents UI redressing attacks
//! - **Code Injection Prevention**: Prevents unauthorized script execution
//! - **Data Exfiltration Protection**: Controls where data can be sent
//! - **Mixed Content Prevention**: Enforces HTTPS for all resources
//! - **Frame Embedding Control**: Controls where the page can be embedded
//!
//! ## Usage
//!
//! ```rust,no_run
//! use oxirs_gql::content_security_policy::{CspBuilder, CspDirective, CspSource};
//!
//! // Create a strict CSP policy
//! let csp = CspBuilder::new()
//!     .add_directive(CspDirective::DefaultSrc, vec![CspSource::SelfOrigin])
//!     .add_directive(CspDirective::ScriptSrc, vec![
//!         CspSource::SelfOrigin,
//!         CspSource::Nonce("abc123".to_string()),
//!     ])
//!     .add_directive(CspDirective::StyleSrc, vec![
//!         CspSource::SelfOrigin,
//!         CspSource::UnsafeInline, // For GraphQL Playground
//!     ])
//!     .upgrade_insecure_requests(true)
//!     .build();
//!
//! // Apply to HTTP response headers
//! let header_value = csp.to_header_value();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// CSP directive types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CspDirective {
    /// Fallback for other fetch directives
    DefaultSrc,
    /// Valid sources for JavaScript
    ScriptSrc,
    /// Valid sources for stylesheets
    StyleSrc,
    /// Valid sources for images
    ImgSrc,
    /// Valid sources for fonts
    FontSrc,
    /// Valid sources for XMLHttpRequest, fetch(), WebSocket
    ConnectSrc,
    /// Valid sources for `<frame>`, `<iframe>`
    FrameSrc,
    /// Valid sources for `<object>`, `<embed>`, `<applet>`
    ObjectSrc,
    /// Valid sources for web workers
    WorkerSrc,
    /// Valid sources for manifest files
    ManifestSrc,
    /// Valid sources for `<form>` submissions
    FormAction,
    /// Valid parents that can embed using `<frame>`, `<iframe>`
    FrameAncestors,
    /// Valid sources for `<base>` URIs
    BaseUri,
    /// Valid sources for `<audio>`, `<video>`
    MediaSrc,
    /// Sandbox restrictions
    Sandbox,
    /// Report URI for violations (deprecated, use report-to)
    ReportUri,
    /// Report endpoint name for violations
    ReportTo,
}

impl CspDirective {
    /// Get directive name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DefaultSrc => "default-src",
            Self::ScriptSrc => "script-src",
            Self::StyleSrc => "style-src",
            Self::ImgSrc => "img-src",
            Self::FontSrc => "font-src",
            Self::ConnectSrc => "connect-src",
            Self::FrameSrc => "frame-src",
            Self::ObjectSrc => "object-src",
            Self::WorkerSrc => "worker-src",
            Self::ManifestSrc => "manifest-src",
            Self::FormAction => "form-action",
            Self::FrameAncestors => "frame-ancestors",
            Self::BaseUri => "base-uri",
            Self::MediaSrc => "media-src",
            Self::Sandbox => "sandbox",
            Self::ReportUri => "report-uri",
            Self::ReportTo => "report-to",
        }
    }
}

/// CSP source values
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CspSource {
    /// 'self' - same origin
    SelfOrigin,
    /// 'none' - no sources allowed
    None,
    /// 'unsafe-inline' - allow inline scripts/styles
    UnsafeInline,
    /// 'unsafe-eval' - allow eval()
    UnsafeEval,
    /// 'strict-dynamic' - trust scripts with nonce/hash
    StrictDynamic,
    /// 'nonce-{value}' - cryptographic nonce
    Nonce(String),
    /// 'sha256-{hash}' - cryptographic hash
    Sha256Hash(String),
    /// 'sha384-{hash}' - cryptographic hash
    Sha384Hash(String),
    /// 'sha512-{hash}' - cryptographic hash
    Sha512Hash(String),
    /// Specific URI (e.g., <https://example.com>)
    Uri(String),
    /// Wildcard scheme (e.g., https:)
    Scheme(String),
    /// Wildcard host (e.g., *.example.com)
    Host(String),
}

impl CspSource {
    /// Convert to string representation
    pub fn as_str(&self) -> String {
        match self {
            Self::SelfOrigin => "'self'".to_string(),
            Self::None => "'none'".to_string(),
            Self::UnsafeInline => "'unsafe-inline'".to_string(),
            Self::UnsafeEval => "'unsafe-eval'".to_string(),
            Self::StrictDynamic => "'strict-dynamic'".to_string(),
            Self::Nonce(nonce) => format!("'nonce-{}'", nonce),
            Self::Sha256Hash(hash) => format!("'sha256-{}'", hash),
            Self::Sha384Hash(hash) => format!("'sha384-{}'", hash),
            Self::Sha512Hash(hash) => format!("'sha512-{}'", hash),
            Self::Uri(uri) => uri.clone(),
            Self::Scheme(scheme) => format!("{}:", scheme),
            Self::Host(host) => host.clone(),
        }
    }
}

/// Sandbox directive values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SandboxValue {
    /// Allow form submission
    AllowForms,
    /// Allow same-origin access
    AllowSameOrigin,
    /// Allow scripts
    AllowScripts,
    /// Allow popups
    AllowPopups,
    /// Allow modals
    AllowModals,
    /// Allow orientation lock
    AllowOrientationLock,
    /// Allow pointer lock
    AllowPointerLock,
    /// Allow presentation
    AllowPresentation,
    /// Allow popups to escape sandbox
    AllowPopupsToEscapeSandbox,
    /// Allow top navigation
    AllowTopNavigation,
}

impl SandboxValue {
    /// Get sandbox value as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AllowForms => "allow-forms",
            Self::AllowSameOrigin => "allow-same-origin",
            Self::AllowScripts => "allow-scripts",
            Self::AllowPopups => "allow-popups",
            Self::AllowModals => "allow-modals",
            Self::AllowOrientationLock => "allow-orientation-lock",
            Self::AllowPointerLock => "allow-pointer-lock",
            Self::AllowPresentation => "allow-presentation",
            Self::AllowPopupsToEscapeSandbox => "allow-popups-to-escape-sandbox",
            Self::AllowTopNavigation => "allow-top-navigation",
        }
    }
}

/// Content Security Policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSecurityPolicy {
    /// CSP directives
    directives: HashMap<CspDirective, Vec<CspSource>>,
    /// Sandbox values
    sandbox_values: Vec<SandboxValue>,
    /// Upgrade insecure requests
    upgrade_insecure_requests: bool,
    /// Block all mixed content
    block_all_mixed_content: bool,
    /// Report only mode (doesn't enforce)
    report_only: bool,
}

impl ContentSecurityPolicy {
    /// Create a new CSP builder
    pub fn builder() -> CspBuilder {
        CspBuilder::new()
    }

    /// Create a strict default policy for GraphQL servers
    pub fn strict_default() -> Self {
        CspBuilder::new()
            .add_directive(CspDirective::DefaultSrc, vec![CspSource::SelfOrigin])
            .add_directive(
                CspDirective::ScriptSrc,
                vec![CspSource::SelfOrigin, CspSource::StrictDynamic],
            )
            .add_directive(CspDirective::StyleSrc, vec![CspSource::SelfOrigin])
            .add_directive(CspDirective::ImgSrc, vec![CspSource::SelfOrigin])
            .add_directive(CspDirective::FontSrc, vec![CspSource::SelfOrigin])
            .add_directive(
                CspDirective::ConnectSrc,
                vec![CspSource::SelfOrigin, CspSource::Scheme("wss".to_string())],
            )
            .add_directive(CspDirective::FrameSrc, vec![CspSource::None])
            .add_directive(CspDirective::ObjectSrc, vec![CspSource::None])
            .add_directive(CspDirective::FrameAncestors, vec![CspSource::None])
            .add_directive(CspDirective::BaseUri, vec![CspSource::SelfOrigin])
            .upgrade_insecure_requests(true)
            .block_all_mixed_content(true)
            .build()
    }

    /// Create a development-friendly policy (for GraphQL Playground, GraphiQL)
    pub fn development_friendly() -> Self {
        CspBuilder::new()
            .add_directive(CspDirective::DefaultSrc, vec![CspSource::SelfOrigin])
            .add_directive(
                CspDirective::ScriptSrc,
                vec![
                    CspSource::SelfOrigin,
                    CspSource::UnsafeInline,
                    CspSource::UnsafeEval, // For GraphQL Playground
                ],
            )
            .add_directive(
                CspDirective::StyleSrc,
                vec![
                    CspSource::SelfOrigin,
                    CspSource::UnsafeInline, // For GraphQL Playground
                ],
            )
            .add_directive(
                CspDirective::ImgSrc,
                vec![
                    CspSource::SelfOrigin,
                    CspSource::Scheme("data".to_string()),
                    CspSource::Scheme("https".to_string()),
                ],
            )
            .add_directive(CspDirective::FontSrc, vec![CspSource::SelfOrigin])
            .add_directive(
                CspDirective::ConnectSrc,
                vec![
                    CspSource::SelfOrigin,
                    CspSource::Scheme("ws".to_string()),
                    CspSource::Scheme("wss".to_string()),
                ],
            )
            .upgrade_insecure_requests(false) // Allow HTTP in dev
            .build()
    }

    /// Generate CSP header value
    pub fn to_header_value(&self) -> String {
        let mut parts = Vec::new();

        // Add directives
        for (directive, sources) in &self.directives {
            let sources_str = sources
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            parts.push(format!("{} {}", directive.as_str(), sources_str));
        }

        // Add sandbox if present
        if !self.sandbox_values.is_empty() {
            let sandbox_str = self
                .sandbox_values
                .iter()
                .map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            parts.push(format!("sandbox {}", sandbox_str));
        }

        // Add special directives
        if self.upgrade_insecure_requests {
            parts.push("upgrade-insecure-requests".to_string());
        }

        if self.block_all_mixed_content {
            parts.push("block-all-mixed-content".to_string());
        }

        parts.join("; ")
    }

    /// Get header name based on report-only mode
    pub fn header_name(&self) -> &'static str {
        if self.report_only {
            "Content-Security-Policy-Report-Only"
        } else {
            "Content-Security-Policy"
        }
    }

    /// Check if policy is in report-only mode
    pub fn is_report_only(&self) -> bool {
        self.report_only
    }

    /// Get directives
    pub fn directives(&self) -> &HashMap<CspDirective, Vec<CspSource>> {
        &self.directives
    }

    /// Generate a nonce for script-src
    pub fn generate_nonce() -> String {
        // Use fastrand for cryptographically secure random bytes
        let bytes: Vec<u8> = (0..16).map(|_| fastrand::u8(..)).collect();
        use base64::engine::general_purpose::STANDARD;
        use base64::Engine;
        STANDARD.encode(&bytes)
    }

    /// Calculate SHA-256 hash of inline script/style
    pub fn calculate_sha256(content: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let result = hasher.finalize();
        use base64::engine::general_purpose::STANDARD;
        use base64::Engine;
        STANDARD.encode(result)
    }

    /// Calculate SHA-384 hash of inline script/style
    pub fn calculate_sha384(content: &str) -> String {
        use sha2::{Digest, Sha384};
        let mut hasher = Sha384::new();
        hasher.update(content.as_bytes());
        let result = hasher.finalize();
        use base64::engine::general_purpose::STANDARD;
        use base64::Engine;
        STANDARD.encode(result)
    }

    /// Calculate SHA-512 hash of inline script/style
    pub fn calculate_sha512(content: &str) -> String {
        use sha2::{Digest, Sha512};
        let mut hasher = Sha512::new();
        hasher.update(content.as_bytes());
        let result = hasher.finalize();
        use base64::engine::general_purpose::STANDARD;
        use base64::Engine;
        STANDARD.encode(result)
    }
}

impl fmt::Display for ContentSecurityPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_header_value())
    }
}

/// Builder for Content Security Policy
#[derive(Debug, Clone)]
pub struct CspBuilder {
    directives: HashMap<CspDirective, Vec<CspSource>>,
    sandbox_values: Vec<SandboxValue>,
    upgrade_insecure_requests: bool,
    block_all_mixed_content: bool,
    report_only: bool,
}

impl CspBuilder {
    /// Create a new CSP builder
    pub fn new() -> Self {
        Self {
            directives: HashMap::new(),
            sandbox_values: Vec::new(),
            upgrade_insecure_requests: false,
            block_all_mixed_content: false,
            report_only: false,
        }
    }

    /// Add a directive with sources
    pub fn add_directive(mut self, directive: CspDirective, sources: Vec<CspSource>) -> Self {
        self.directives.insert(directive, sources);
        self
    }

    /// Add a source to an existing directive
    pub fn add_source(mut self, directive: CspDirective, source: CspSource) -> Self {
        self.directives.entry(directive).or_default().push(source);
        self
    }

    /// Add sandbox values
    pub fn sandbox(mut self, values: Vec<SandboxValue>) -> Self {
        self.sandbox_values = values;
        self
    }

    /// Enable upgrade insecure requests
    pub fn upgrade_insecure_requests(mut self, enable: bool) -> Self {
        self.upgrade_insecure_requests = enable;
        self
    }

    /// Enable block all mixed content
    pub fn block_all_mixed_content(mut self, enable: bool) -> Self {
        self.block_all_mixed_content = enable;
        self
    }

    /// Set report-only mode
    pub fn report_only(mut self, enable: bool) -> Self {
        self.report_only = enable;
        self
    }

    /// Build the CSP
    pub fn build(self) -> ContentSecurityPolicy {
        ContentSecurityPolicy {
            directives: self.directives,
            sandbox_values: self.sandbox_values,
            upgrade_insecure_requests: self.upgrade_insecure_requests,
            block_all_mixed_content: self.block_all_mixed_content,
            report_only: self.report_only,
        }
    }
}

impl Default for CspBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Violation report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CspViolationReport {
    /// Document URI where violation occurred
    #[serde(rename = "document-uri")]
    pub document_uri: String,
    /// Violated directive
    #[serde(rename = "violated-directive")]
    pub violated_directive: String,
    /// Effective directive
    #[serde(rename = "effective-directive")]
    pub effective_directive: String,
    /// Original policy
    #[serde(rename = "original-policy")]
    pub original_policy: String,
    /// Blocked URI
    #[serde(rename = "blocked-uri")]
    pub blocked_uri: String,
    /// Status code
    #[serde(rename = "status-code")]
    pub status_code: u16,
    /// Referrer
    pub referrer: Option<String>,
    /// Source file
    #[serde(rename = "source-file")]
    pub source_file: Option<String>,
    /// Line number
    #[serde(rename = "line-number")]
    pub line_number: Option<u32>,
    /// Column number
    #[serde(rename = "column-number")]
    pub column_number: Option<u32>,
}

/// CSP violation handler
pub trait CspViolationHandler: Send + Sync {
    /// Handle a CSP violation report
    fn handle_violation(&self, report: CspViolationReport);
}

/// Default violation handler that logs violations
#[derive(Debug, Clone)]
pub struct LoggingViolationHandler;

impl CspViolationHandler for LoggingViolationHandler {
    fn handle_violation(&self, report: CspViolationReport) {
        tracing::warn!(
            directive = %report.violated_directive,
            blocked_uri = %report.blocked_uri,
            document_uri = %report.document_uri,
            "CSP violation detected"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csp_directive_as_str() {
        assert_eq!(CspDirective::DefaultSrc.as_str(), "default-src");
        assert_eq!(CspDirective::ScriptSrc.as_str(), "script-src");
        assert_eq!(CspDirective::StyleSrc.as_str(), "style-src");
    }

    #[test]
    fn test_csp_source_as_str() {
        assert_eq!(CspSource::SelfOrigin.as_str(), "'self'");
        assert_eq!(CspSource::None.as_str(), "'none'");
        assert_eq!(CspSource::UnsafeInline.as_str(), "'unsafe-inline'");
        assert_eq!(
            CspSource::Nonce("abc123".to_string()).as_str(),
            "'nonce-abc123'"
        );
        assert_eq!(
            CspSource::Uri("https://example.com".to_string()).as_str(),
            "https://example.com"
        );
    }

    #[test]
    fn test_csp_builder_basic() {
        let csp = CspBuilder::new()
            .add_directive(CspDirective::DefaultSrc, vec![CspSource::SelfOrigin])
            .build();

        let header = csp.to_header_value();
        assert!(header.contains("default-src 'self'"));
    }

    #[test]
    fn test_csp_strict_default() {
        let csp = ContentSecurityPolicy::strict_default();
        let header = csp.to_header_value();

        assert!(header.contains("default-src 'self'"));
        assert!(header.contains("frame-src 'none'"));
        assert!(header.contains("object-src 'none'"));
        assert!(header.contains("upgrade-insecure-requests"));
        assert!(header.contains("block-all-mixed-content"));
    }

    #[test]
    fn test_csp_development_friendly() {
        let csp = ContentSecurityPolicy::development_friendly();
        let header = csp.to_header_value();

        assert!(header.contains("script-src 'self' 'unsafe-inline' 'unsafe-eval'"));
        assert!(header.contains("style-src 'self' 'unsafe-inline'"));
        assert!(!header.contains("upgrade-insecure-requests"));
    }

    #[test]
    fn test_csp_with_nonce() {
        let nonce = "abc123";
        let csp = CspBuilder::new()
            .add_directive(
                CspDirective::ScriptSrc,
                vec![CspSource::SelfOrigin, CspSource::Nonce(nonce.to_string())],
            )
            .build();

        let header = csp.to_header_value();
        assert!(header.contains("'nonce-abc123'"));
    }

    #[test]
    fn test_csp_with_hash() {
        let hash = "xyz789";
        let csp = CspBuilder::new()
            .add_directive(
                CspDirective::ScriptSrc,
                vec![
                    CspSource::SelfOrigin,
                    CspSource::Sha256Hash(hash.to_string()),
                ],
            )
            .build();

        let header = csp.to_header_value();
        assert!(header.contains("'sha256-xyz789'"));
    }

    #[test]
    fn test_csp_sandbox() {
        let csp = CspBuilder::new()
            .sandbox(vec![SandboxValue::AllowScripts, SandboxValue::AllowForms])
            .build();

        let header = csp.to_header_value();
        assert!(header.contains("sandbox allow-scripts allow-forms"));
    }

    #[test]
    fn test_csp_report_only() {
        let csp = CspBuilder::new()
            .add_directive(CspDirective::DefaultSrc, vec![CspSource::SelfOrigin])
            .report_only(true)
            .build();

        assert_eq!(csp.header_name(), "Content-Security-Policy-Report-Only");
        assert!(csp.is_report_only());
    }

    #[test]
    fn test_generate_nonce() {
        let nonce1 = ContentSecurityPolicy::generate_nonce();
        let nonce2 = ContentSecurityPolicy::generate_nonce();

        assert!(!nonce1.is_empty());
        assert!(!nonce2.is_empty());
        assert_ne!(nonce1, nonce2); // Should be different
    }

    #[test]
    fn test_calculate_sha256() {
        let content = "console.log('hello');";
        let hash = ContentSecurityPolicy::calculate_sha256(content);

        assert!(!hash.is_empty());
        // Same content should produce same hash
        assert_eq!(hash, ContentSecurityPolicy::calculate_sha256(content));
    }

    #[test]
    fn test_calculate_sha384() {
        let content = "alert('test');";
        let hash = ContentSecurityPolicy::calculate_sha384(content);

        assert!(!hash.is_empty());
        assert_eq!(hash, ContentSecurityPolicy::calculate_sha384(content));
    }

    #[test]
    fn test_calculate_sha512() {
        let content = "document.write('foo');";
        let hash = ContentSecurityPolicy::calculate_sha512(content);

        assert!(!hash.is_empty());
        assert_eq!(hash, ContentSecurityPolicy::calculate_sha512(content));
    }

    #[test]
    fn test_csp_multiple_directives() {
        let csp = CspBuilder::new()
            .add_directive(CspDirective::DefaultSrc, vec![CspSource::SelfOrigin])
            .add_directive(
                CspDirective::ScriptSrc,
                vec![CspSource::SelfOrigin, CspSource::StrictDynamic],
            )
            .add_directive(CspDirective::StyleSrc, vec![CspSource::SelfOrigin])
            .add_directive(CspDirective::ImgSrc, vec![CspSource::SelfOrigin])
            .build();

        let header = csp.to_header_value();
        assert!(header.contains("default-src"));
        assert!(header.contains("script-src"));
        assert!(header.contains("style-src"));
        assert!(header.contains("img-src"));
    }

    #[test]
    fn test_csp_add_source() {
        let csp = CspBuilder::new()
            .add_source(CspDirective::ScriptSrc, CspSource::SelfOrigin)
            .add_source(CspDirective::ScriptSrc, CspSource::StrictDynamic)
            .build();

        let header = csp.to_header_value();
        assert!(header.contains("script-src 'self' 'strict-dynamic'"));
    }

    #[test]
    fn test_csp_with_scheme() {
        let csp = CspBuilder::new()
            .add_directive(
                CspDirective::ConnectSrc,
                vec![CspSource::SelfOrigin, CspSource::Scheme("wss".to_string())],
            )
            .build();

        let header = csp.to_header_value();
        assert!(header.contains("connect-src 'self' wss:"));
    }

    #[test]
    fn test_logging_violation_handler() {
        let handler = LoggingViolationHandler;
        let report = CspViolationReport {
            document_uri: "https://example.com".to_string(),
            violated_directive: "script-src".to_string(),
            effective_directive: "script-src".to_string(),
            original_policy: "default-src 'self'".to_string(),
            blocked_uri: "https://evil.com/script.js".to_string(),
            status_code: 200,
            referrer: None,
            source_file: None,
            line_number: None,
            column_number: None,
        };

        // Should not panic
        handler.handle_violation(report);
    }
}
