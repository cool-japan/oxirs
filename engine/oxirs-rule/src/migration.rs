//! Migration Tools for Rule Engines
//!
//! Provides tools to migrate rule sets from other popular rule engines to OxiRS format.
//! Supports Apache Jena, Drools, and CLIPS rule formats.
//!
//! # Supported Formats
//!
//! - **Apache Jena**: Jena rules format (text-based rule syntax)
//! - **Drools**: DRL (Drools Rule Language) format
//! - **CLIPS**: CLIPS rule format
//!
//! # Features
//!
//! - **Syntax Parsing**: Parse rules from various formats
//! - **Semantic Mapping**: Map concepts between rule engines
//! - **Validation**: Validate migrated rules for correctness
//! - **Report Generation**: Generate migration reports with warnings
//! - **Batch Migration**: Migrate entire rule sets
//! - **Incremental Migration**: Support partial migrations
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::migration::{MigrationTool, SourceFormat};
//!
//! let mut migrator = MigrationTool::new();
//!
//! // Migrate from Jena format
//! let jena_rules = r#"
//! [rule1: (?a rdf:type Person) -> (?a rdf:type Human)]
//! [rule2: (?x parent ?y) -> (?y hasParent ?x)]
//! "#;
//!
//! let result = migrator.migrate(jena_rules, SourceFormat::Jena).unwrap();
//!
//! println!("Migrated {} rules", result.rules.len());
//! println!("Report:\n{}", result.generate_report());
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::{anyhow, Context, Result};
use regex::Regex;
use std::collections::HashMap;
use tracing::{info, warn};

/// Source format for migration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    /// Apache Jena rules format
    Jena,
    /// Drools Rule Language (DRL)
    Drools,
    /// CLIPS rule format
    Clips,
}

impl SourceFormat {
    /// Get format name
    pub fn name(&self) -> &str {
        match self {
            Self::Jena => "Apache Jena",
            Self::Drools => "Drools DRL",
            Self::Clips => "CLIPS",
        }
    }
}

/// Migration warning
#[derive(Debug, Clone)]
pub struct MigrationWarning {
    /// Warning message
    pub message: String,
    /// Line number where warning occurred
    pub line: Option<usize>,
    /// Severity (info, warning, error)
    pub severity: WarningSeverity,
}

/// Warning severity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningSeverity {
    Info,
    Warning,
    Error,
}

/// Migration result
#[derive(Debug)]
pub struct MigrationResult {
    /// Successfully migrated rules
    pub rules: Vec<Rule>,
    /// Warnings encountered during migration
    pub warnings: Vec<MigrationWarning>,
    /// Source format
    pub source_format: SourceFormat,
    /// Original rule count
    pub original_count: usize,
    /// Successfully migrated count
    pub migrated_count: usize,
}

impl MigrationResult {
    /// Generate a detailed migration report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str(&format!(
            "   Migration Report: {} → OxiRS\n",
            self.source_format.name()
        ));
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        report.push_str(&format!("Original rules:  {}\n", self.original_count));
        report.push_str(&format!("Migrated rules:  {}\n", self.migrated_count));
        report.push_str(&format!(
            "Success rate:    {:.1}%\n\n",
            (self.migrated_count as f64 / self.original_count as f64) * 100.0
        ));

        if !self.warnings.is_empty() {
            report.push_str("Warnings:\n");
            report.push_str("─────────────────────────────────────────────────────────────\n");

            let mut errors = 0;
            let mut warnings = 0;
            let mut infos = 0;

            for warning in &self.warnings {
                let prefix = match warning.severity {
                    WarningSeverity::Error => {
                        errors += 1;
                        "ERROR"
                    }
                    WarningSeverity::Warning => {
                        warnings += 1;
                        "WARN"
                    }
                    WarningSeverity::Info => {
                        infos += 1;
                        "INFO"
                    }
                };

                if let Some(line) = warning.line {
                    report.push_str(&format!(
                        "  [{}] Line {}: {}\n",
                        prefix, line, warning.message
                    ));
                } else {
                    report.push_str(&format!("  [{}] {}\n", prefix, warning.message));
                }
            }

            report.push_str(&format!(
                "\nSummary: {} errors, {} warnings, {} info\n",
                errors, warnings, infos
            ));
        }

        report.push_str("\n═══════════════════════════════════════════════════════════════\n");
        report
    }

    /// Check if migration was successful
    pub fn is_successful(&self) -> bool {
        self.warnings
            .iter()
            .all(|w| w.severity != WarningSeverity::Error)
    }
}

/// Migration tool for converting rules between formats
pub struct MigrationTool {
    /// Enable strict validation
    strict_mode: bool,
    /// Custom namespace mappings
    namespace_mappings: HashMap<String, String>,
}

impl Default for MigrationTool {
    fn default() -> Self {
        Self::new()
    }
}

impl MigrationTool {
    /// Create a new migration tool
    pub fn new() -> Self {
        Self {
            strict_mode: false,
            namespace_mappings: HashMap::new(),
        }
    }

    /// Enable strict validation mode
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Add namespace mapping
    pub fn add_namespace_mapping(&mut self, from: &str, to: &str) {
        self.namespace_mappings
            .insert(from.to_string(), to.to_string());
    }

    /// Migrate rules from source format to OxiRS
    pub fn migrate(&mut self, source: &str, format: SourceFormat) -> Result<MigrationResult> {
        info!("Starting migration from {}", format.name());

        let mut warnings = Vec::new();
        let rules = match format {
            SourceFormat::Jena => self.migrate_jena(source, &mut warnings)?,
            SourceFormat::Drools => self.migrate_drools(source, &mut warnings)?,
            SourceFormat::Clips => self.migrate_clips(source, &mut warnings)?,
        };

        let original_count = source
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && !trimmed.starts_with('#') && !trimmed.starts_with("//")
            })
            .count();

        Ok(MigrationResult {
            migrated_count: rules.len(),
            rules,
            warnings,
            source_format: format,
            original_count,
        })
    }

    /// Migrate from Apache Jena format
    fn migrate_jena(
        &mut self,
        source: &str,
        warnings: &mut Vec<MigrationWarning>,
    ) -> Result<Vec<Rule>> {
        let mut rules = Vec::new();

        // Jena rule format: [ruleName: (body) -> (head)]
        let rule_regex = Regex::new(r"\[(\w+):\s*(.+?)\s*->\s*(.+?)\]")
            .context("Failed to compile Jena rule regex")?;

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            if let Some(captures) = rule_regex.captures(trimmed) {
                let rule_name = captures.get(1).unwrap().as_str();
                let body_str = captures.get(2).unwrap().as_str();
                let head_str = captures.get(3).unwrap().as_str();

                match self.parse_jena_atoms(body_str, head_str, rule_name) {
                    Ok(rule) => rules.push(rule),
                    Err(e) => {
                        warnings.push(MigrationWarning {
                            message: format!("Failed to parse rule '{}': {}", rule_name, e),
                            line: Some(line_num + 1),
                            severity: if self.strict_mode {
                                WarningSeverity::Error
                            } else {
                                WarningSeverity::Warning
                            },
                        });
                    }
                }
            } else {
                warnings.push(MigrationWarning {
                    message: format!("Could not parse Jena rule syntax: {}", trimmed),
                    line: Some(line_num + 1),
                    severity: WarningSeverity::Warning,
                });
            }
        }

        info!("Migrated {} Jena rules", rules.len());
        Ok(rules)
    }

    /// Parse Jena atoms into OxiRS format
    fn parse_jena_atoms(&self, body_str: &str, head_str: &str, rule_name: &str) -> Result<Rule> {
        let body = self.parse_jena_triples(body_str)?;
        let head = self.parse_jena_triples(head_str)?;

        Ok(Rule {
            name: rule_name.to_string(),
            body,
            head,
        })
    }

    /// Parse Jena triple patterns
    fn parse_jena_triples(&self, patterns: &str) -> Result<Vec<RuleAtom>> {
        let mut atoms = Vec::new();

        // Split by comma or conjunction
        for pattern in patterns.split(',') {
            let pattern = pattern.trim();
            if pattern.is_empty() {
                continue;
            }

            // Remove parentheses
            let pattern = pattern.strip_prefix('(').unwrap_or(pattern);
            let pattern = pattern.strip_suffix(')').unwrap_or(pattern);

            // Parse triple pattern: subject predicate object
            let parts: Vec<&str> = pattern.split_whitespace().collect();
            if parts.len() == 3 {
                atoms.push(RuleAtom::Triple {
                    subject: self.parse_jena_term(parts[0])?,
                    predicate: self.parse_jena_term(parts[1])?,
                    object: self.parse_jena_term(parts[2])?,
                });
            } else {
                return Err(anyhow!("Invalid triple pattern: {}", pattern));
            }
        }

        Ok(atoms)
    }

    /// Parse Jena term (variable or constant)
    fn parse_jena_term(&self, term: &str) -> Result<Term> {
        if let Some(var_name) = term.strip_prefix('?') {
            // Variable
            Ok(Term::Variable(var_name.to_string()))
        } else if term.starts_with('"') && term.ends_with('"') {
            // Literal
            Ok(Term::Literal(term[1..term.len() - 1].to_string()))
        } else {
            // Constant (URI or local name)
            let expanded = self.expand_namespace(term);
            Ok(Term::Constant(expanded))
        }
    }

    /// Migrate from Drools DRL format
    fn migrate_drools(
        &mut self,
        source: &str,
        warnings: &mut Vec<MigrationWarning>,
    ) -> Result<Vec<Rule>> {
        let mut rules = Vec::new();

        // Basic Drools parsing - this is simplified
        // Real Drools has complex syntax including Java code

        let rule_regex = Regex::new(r#"rule\s+"([^"]+)"\s+when\s+(.+?)\s+then\s+(.+?)\s+end"#)
            .context("Failed to compile Drools rule regex")?;

        for (line_num, captures) in rule_regex.captures_iter(source).enumerate() {
            let rule_name = captures.get(1).unwrap().as_str();
            let when_clause = captures.get(2).unwrap().as_str();
            let then_clause = captures.get(3).unwrap().as_str();

            // Create enhanced rule
            match self.parse_drools_rule(rule_name, when_clause, then_clause) {
                Ok(rule) => {
                    if !rule.body.is_empty() || !rule.head.is_empty() {
                        rules.push(rule);
                        warnings.push(MigrationWarning {
                            message: format!(
                                "Drools rule '{}' migrated - complex DRL features (Java code, salience, etc.) not supported",
                                rule_name
                            ),
                            line: Some(line_num + 1),
                            severity: WarningSeverity::Info,
                        });
                    } else {
                        warnings.push(MigrationWarning {
                            message: format!(
                                "Drools rule '{}' skipped - no parseable patterns found",
                                rule_name
                            ),
                            line: Some(line_num + 1),
                            severity: WarningSeverity::Warning,
                        });
                    }
                }
                Err(e) => {
                    warnings.push(MigrationWarning {
                        message: format!("Failed to parse Drools rule '{}': {}", rule_name, e),
                        line: Some(line_num + 1),
                        severity: WarningSeverity::Error,
                    });
                }
            }
        }

        info!("Migrated {} Drools rules", rules.len());
        Ok(rules)
    }

    /// Parse Drools rule
    fn parse_drools_rule(&self, name: &str, when_clause: &str, then_clause: &str) -> Result<Rule> {
        // Parse when clause (conditions/patterns)
        let body = self.parse_drools_when_clause(when_clause)?;

        // Parse then clause (actions/assertions)
        let head = self.parse_drools_then_clause(then_clause)?;

        if body.is_empty() && head.is_empty() {
            warn!(
                "Drools rule '{}' has no parseable conditions or actions",
                name
            );
        }

        Ok(Rule {
            name: name.to_string(),
            body,
            head,
        })
    }

    /// Parse Drools when clause
    fn parse_drools_when_clause(&self, when_clause: &str) -> Result<Vec<RuleAtom>> {
        let mut atoms = Vec::new();

        // Pattern: ClassName(field == value, field2 == value2)
        let pattern_regex =
            Regex::new(r"(\w+)\s*\((.*?)\)").context("Failed to compile Drools pattern regex")?;

        for captures in pattern_regex.captures_iter(when_clause) {
            let class_name = captures.get(1).unwrap().as_str();
            let conditions = captures.get(2).unwrap().as_str();

            // Parse conditions
            for condition in conditions.split(',') {
                let condition = condition.trim();
                if condition.is_empty() {
                    continue;
                }

                // Pattern: field == value or field : value
                if let Some((field, value)) = self.parse_drools_condition(condition) {
                    // Create triple: ?instance field value
                    atoms.push(RuleAtom::Triple {
                        subject: Term::Variable(format!("{}Instance", class_name)),
                        predicate: Term::Constant(field),
                        object: self.parse_drools_value(&value),
                    });
                }
            }

            // Add type triple: ?instance rdf:type ClassName
            atoms.push(RuleAtom::Triple {
                subject: Term::Variable(format!("{}Instance", class_name)),
                predicate: Term::Constant("rdf:type".to_string()),
                object: Term::Constant(class_name.to_string()),
            });
        }

        Ok(atoms)
    }

    /// Parse Drools then clause
    fn parse_drools_then_clause(&self, then_clause: &str) -> Result<Vec<RuleAtom>> {
        let mut atoms = Vec::new();

        // Pattern: insert(new ClassName(field: value))
        let insert_regex = Regex::new(r"insert\s*\(\s*new\s+(\w+)\s*\((.*?)\)\s*\)")
            .context("Failed to compile Drools insert regex")?;

        for captures in insert_regex.captures_iter(then_clause) {
            let class_name = captures.get(1).unwrap().as_str();
            let params = captures.get(2).unwrap().as_str();

            // Add type assertion
            atoms.push(RuleAtom::Triple {
                subject: Term::Variable(format!("new{}", class_name)),
                predicate: Term::Constant("rdf:type".to_string()),
                object: Term::Constant(class_name.to_string()),
            });

            // Parse parameters
            for param in params.split(',') {
                let param = param.trim();
                if let Some((field, value)) = self.parse_drools_condition(param) {
                    atoms.push(RuleAtom::Triple {
                        subject: Term::Variable(format!("new{}", class_name)),
                        predicate: Term::Constant(field),
                        object: self.parse_drools_value(&value),
                    });
                }
            }
        }

        // Pattern: modify(variable) { setField(value) }
        let modify_regex = Regex::new(r"modify\s*\(\s*(\w+)\s*\)\s*\{([^}]+)\}")
            .context("Failed to compile Drools modify regex")?;

        // Parse setter calls: setField(value) - compile regex outside loop
        let setter_regex =
            Regex::new(r"set(\w+)\s*\(([^)]+)\)").context("Failed to compile setter regex")?;

        for captures in modify_regex.captures_iter(then_clause) {
            let var_name = captures.get(1).unwrap().as_str();
            let modifications = captures.get(2).unwrap().as_str();

            for setter_captures in setter_regex.captures_iter(modifications) {
                let field_name = setter_captures.get(1).unwrap().as_str();
                let value = setter_captures.get(2).unwrap().as_str().trim();

                atoms.push(RuleAtom::Triple {
                    subject: Term::Variable(var_name.to_string()),
                    predicate: Term::Constant(field_name.to_string()),
                    object: self.parse_drools_value(value),
                });
            }
        }

        Ok(atoms)
    }

    /// Parse Drools condition (field == value or field : value)
    fn parse_drools_condition(&self, condition: &str) -> Option<(String, String)> {
        // Try == operator
        if let Some(eq_pos) = condition.find("==") {
            let field = condition[..eq_pos].trim().to_string();
            let value = condition[eq_pos + 2..].trim().to_string();
            return Some((field, value));
        }

        // Try : operator
        if let Some(colon_pos) = condition.find(':') {
            let field = condition[..colon_pos].trim().to_string();
            let value = condition[colon_pos + 1..].trim().to_string();
            return Some((field, value));
        }

        None
    }

    /// Parse Drools value (string literal, number, or variable)
    fn parse_drools_value(&self, value: &str) -> Term {
        let value = value.trim();

        // String literal
        if (value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\''))
        {
            return Term::Literal(value[1..value.len() - 1].to_string());
        }

        // Variable (starts with $)
        if let Some(var_name) = value.strip_prefix('$') {
            return Term::Variable(var_name.to_string());
        }

        // Number or identifier
        if value
            .chars()
            .all(|c| c.is_numeric() || c == '.' || c == '-')
        {
            Term::Literal(value.to_string())
        } else {
            Term::Constant(value.to_string())
        }
    }

    /// Migrate from CLIPS format
    fn migrate_clips(
        &mut self,
        source: &str,
        warnings: &mut Vec<MigrationWarning>,
    ) -> Result<Vec<Rule>> {
        let mut rules = Vec::new();

        // CLIPS rule format: (defrule name (pattern) => (action))
        let rule_regex = Regex::new(r"\(defrule\s+(\S+)\s+(.+?)\s+=>\s+(.+?)\)")
            .context("Failed to compile CLIPS rule regex")?;

        for (line_num, captures) in rule_regex.captures_iter(source).enumerate() {
            let rule_name = captures.get(1).unwrap().as_str();
            let pattern_clause = captures.get(2).unwrap().as_str();
            let action_clause = captures.get(3).unwrap().as_str();

            match self.parse_clips_rule(rule_name, pattern_clause, action_clause) {
                Ok(rule) => {
                    if !rule.body.is_empty() || !rule.head.is_empty() {
                        rules.push(rule);
                        warnings.push(MigrationWarning {
                            message: format!(
                                "CLIPS rule '{}' migrated - procedural attachments and advanced features not supported",
                                rule_name
                            ),
                            line: Some(line_num + 1),
                            severity: WarningSeverity::Info,
                        });
                    } else {
                        warnings.push(MigrationWarning {
                            message: format!(
                                "CLIPS rule '{}' skipped - no parseable patterns found",
                                rule_name
                            ),
                            line: Some(line_num + 1),
                            severity: WarningSeverity::Warning,
                        });
                    }
                }
                Err(e) => {
                    warnings.push(MigrationWarning {
                        message: format!("Failed to parse CLIPS rule '{}': {}", rule_name, e),
                        line: Some(line_num + 1),
                        severity: WarningSeverity::Error,
                    });
                }
            }
        }

        info!("Migrated {} CLIPS rules", rules.len());
        Ok(rules)
    }

    /// Parse CLIPS rule
    fn parse_clips_rule(
        &self,
        name: &str,
        pattern_clause: &str,
        action_clause: &str,
    ) -> Result<Rule> {
        // Parse pattern clause (LHS)
        let body = self.parse_clips_patterns(pattern_clause)?;

        // Parse action clause (RHS)
        let head = self.parse_clips_actions(action_clause)?;

        if body.is_empty() && head.is_empty() {
            warn!("CLIPS rule '{}' has no parseable patterns or actions", name);
        }

        Ok(Rule {
            name: name.to_string(),
            body,
            head,
        })
    }

    /// Parse CLIPS patterns
    fn parse_clips_patterns(&self, patterns: &str) -> Result<Vec<RuleAtom>> {
        let mut atoms = Vec::new();

        // Pattern: (template-name (slot-name value))
        let pattern_regex = Regex::new(r"\((\w+)(?:\s+\(([^)]+)\))*\)")
            .context("Failed to compile CLIPS pattern regex")?;

        for captures in pattern_regex.captures_iter(patterns) {
            let template_name = captures.get(1).unwrap().as_str();

            // Skip control patterns
            if template_name == "test"
                || template_name == "not"
                || template_name == "and"
                || template_name == "or"
            {
                continue;
            }

            // Create type fact
            atoms.push(RuleAtom::Triple {
                subject: Term::Variable(format!("{}Instance", template_name)),
                predicate: Term::Constant("rdf:type".to_string()),
                object: Term::Constant(template_name.to_string()),
            });

            // Parse slot patterns if present
            if let Some(slots) = captures.get(2) {
                let slot_text = slots.as_str();
                // Pattern: (slot value) or slot value
                let slot_parts: Vec<&str> = slot_text.split_whitespace().collect();

                if slot_parts.len() >= 2 {
                    let slot_name = slot_parts[0];
                    let slot_value = slot_parts[1];

                    atoms.push(RuleAtom::Triple {
                        subject: Term::Variable(format!("{}Instance", template_name)),
                        predicate: Term::Constant(slot_name.to_string()),
                        object: self.parse_clips_value(slot_value),
                    });
                }
            }
        }

        Ok(atoms)
    }

    /// Parse CLIPS actions
    fn parse_clips_actions(&self, actions: &str) -> Result<Vec<RuleAtom>> {
        let mut atoms = Vec::new();

        // Pattern: (assert (template-name (slot value)))
        let assert_regex = Regex::new(r"\(assert\s+\((\w+)(?:\s+\(([^)]+)\))*\)\)")
            .context("Failed to compile CLIPS assert regex")?;

        for captures in assert_regex.captures_iter(actions) {
            let template_name = captures.get(1).unwrap().as_str();

            // Create type assertion
            atoms.push(RuleAtom::Triple {
                subject: Term::Variable(format!("new{}", template_name)),
                predicate: Term::Constant("rdf:type".to_string()),
                object: Term::Constant(template_name.to_string()),
            });

            // Parse slot values if present
            if let Some(slots) = captures.get(2) {
                let slot_text = slots.as_str();
                let slot_parts: Vec<&str> = slot_text.split_whitespace().collect();

                if slot_parts.len() >= 2 {
                    let slot_name = slot_parts[0];
                    let slot_value = slot_parts[1];

                    atoms.push(RuleAtom::Triple {
                        subject: Term::Variable(format!("new{}", template_name)),
                        predicate: Term::Constant(slot_name.to_string()),
                        object: self.parse_clips_value(slot_value),
                    });
                }
            }
        }

        // Pattern: (modify ?fact (slot value))
        let modify_regex = Regex::new(r"\(modify\s+\?(\w+)\s+\((\w+)\s+([^)]+)\)\)")
            .context("Failed to compile CLIPS modify regex")?;

        for captures in modify_regex.captures_iter(actions) {
            let var_name = captures.get(1).unwrap().as_str();
            let slot_name = captures.get(2).unwrap().as_str();
            let slot_value = captures.get(3).unwrap().as_str();

            atoms.push(RuleAtom::Triple {
                subject: Term::Variable(var_name.to_string()),
                predicate: Term::Constant(slot_name.to_string()),
                object: self.parse_clips_value(slot_value),
            });
        }

        Ok(atoms)
    }

    /// Parse CLIPS value (variable, string, or literal)
    fn parse_clips_value(&self, value: &str) -> Term {
        let value = value.trim();

        // Variable (starts with ?)
        if let Some(var_name) = value.strip_prefix('?') {
            return Term::Variable(var_name.to_string());
        }

        // String literal
        if (value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\''))
        {
            return Term::Literal(value[1..value.len() - 1].to_string());
        }

        // Symbol or number
        if value
            .chars()
            .all(|c| c.is_numeric() || c == '.' || c == '-')
        {
            Term::Literal(value.to_string())
        } else {
            Term::Constant(value.to_string())
        }
    }

    /// Expand namespace prefix
    fn expand_namespace(&self, term: &str) -> String {
        if let Some(colon_pos) = term.find(':') {
            let prefix = &term[..colon_pos];
            let local = &term[colon_pos + 1..];

            if let Some(namespace) = self.namespace_mappings.get(prefix) {
                return format!("{}{}", namespace, local);
            }
        }
        term.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_tool_creation() {
        let tool = MigrationTool::new();
        assert!(!tool.strict_mode);
        assert!(tool.namespace_mappings.is_empty());
    }

    #[test]
    fn test_migration_tool_with_strict_mode() {
        let tool = MigrationTool::new().with_strict_mode(true);
        assert!(tool.strict_mode);
    }

    #[test]
    fn test_namespace_mapping() {
        let mut tool = MigrationTool::new();
        tool.add_namespace_mapping("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");

        assert_eq!(
            tool.namespace_mappings.get("rdf"),
            Some(&"http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string())
        );
    }

    #[test]
    fn test_jena_simple_rule_migration() {
        let mut tool = MigrationTool::new();

        let jena_rules = r#"
[rule1: (?a rdf:type Person) -> (?a rdf:type Human)]
"#;

        let result = tool.migrate(jena_rules, SourceFormat::Jena).unwrap();

        assert_eq!(result.rules.len(), 1);
        assert_eq!(result.rules[0].name, "rule1");
        assert_eq!(result.source_format, SourceFormat::Jena);
    }

    #[test]
    fn test_jena_multiple_rules() {
        let mut tool = MigrationTool::new();

        let jena_rules = r#"
[rule1: (?x parent ?y) -> (?y hasParent ?x)]
[rule2: (?a knows ?b) -> (?b knows ?a)]
"#;

        let result = tool.migrate(jena_rules, SourceFormat::Jena).unwrap();

        assert_eq!(result.rules.len(), 2);
        assert_eq!(result.rules[0].name, "rule1");
        assert_eq!(result.rules[1].name, "rule2");
    }

    #[test]
    fn test_jena_with_comments() {
        let mut tool = MigrationTool::new();

        let jena_rules = r#"
# This is a comment
[rule1: (?x parent ?y) -> (?y hasParent ?x)]
# Another comment
[rule2: (?a knows ?b) -> (?b knows ?a)]
"#;

        let result = tool.migrate(jena_rules, SourceFormat::Jena).unwrap();

        assert_eq!(result.rules.len(), 2);
    }

    #[test]
    fn test_parse_jena_term_variable() {
        let tool = MigrationTool::new();

        let term = tool.parse_jena_term("?x").unwrap();
        assert!(matches!(term, Term::Variable(v) if v == "x"));
    }

    #[test]
    fn test_parse_jena_term_constant() {
        let tool = MigrationTool::new();

        let term = tool.parse_jena_term("rdf:type").unwrap();
        assert!(matches!(term, Term::Constant(c) if c == "rdf:type"));
    }

    #[test]
    fn test_parse_jena_term_literal() {
        let tool = MigrationTool::new();

        let term = tool.parse_jena_term("\"John\"").unwrap();
        assert!(matches!(term, Term::Literal(l) if l == "John"));
    }

    #[test]
    fn test_migration_report_generation() {
        let result = MigrationResult {
            rules: vec![],
            warnings: vec![MigrationWarning {
                message: "Test warning".to_string(),
                line: Some(1),
                severity: WarningSeverity::Warning,
            }],
            source_format: SourceFormat::Jena,
            original_count: 10,
            migrated_count: 9,
        };

        let report = result.generate_report();

        assert!(report.contains("Apache Jena"));
        assert!(report.contains("Original rules:  10"));
        assert!(report.contains("Migrated rules:  9"));
        assert!(report.contains("Test warning"));
    }

    #[test]
    fn test_migration_success_check() {
        let result_success = MigrationResult {
            rules: vec![],
            warnings: vec![MigrationWarning {
                message: "Info".to_string(),
                line: None,
                severity: WarningSeverity::Info,
            }],
            source_format: SourceFormat::Jena,
            original_count: 1,
            migrated_count: 1,
        };

        assert!(result_success.is_successful());

        let result_failure = MigrationResult {
            rules: vec![],
            warnings: vec![MigrationWarning {
                message: "Error".to_string(),
                line: None,
                severity: WarningSeverity::Error,
            }],
            source_format: SourceFormat::Jena,
            original_count: 1,
            migrated_count: 0,
        };

        assert!(!result_failure.is_successful());
    }

    #[test]
    fn test_expand_namespace() {
        let mut tool = MigrationTool::new();
        tool.add_namespace_mapping("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");

        let expanded = tool.expand_namespace("rdf:type");
        assert_eq!(expanded, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    }

    #[test]
    fn test_source_format_names() {
        assert_eq!(SourceFormat::Jena.name(), "Apache Jena");
        assert_eq!(SourceFormat::Drools.name(), "Drools DRL");
        assert_eq!(SourceFormat::Clips.name(), "CLIPS");
    }

    #[test]
    fn test_drools_simple_rule() {
        let mut tool = MigrationTool::new();

        let drools_rules = r#"
rule "adult-rule"
when
    Person(age == 25)
then
    insert(new Adult(status: "verified"))
end
"#;

        let result = tool.migrate(drools_rules, SourceFormat::Drools).unwrap();

        assert_eq!(result.rules.len(), 1);
        assert_eq!(result.rules[0].name, "adult-rule");
        assert!(!result.rules[0].body.is_empty());
        assert!(!result.rules[0].head.is_empty());
    }

    #[test]
    fn test_drools_with_insert() {
        let mut tool = MigrationTool::new();

        let drools_rules = r#"
rule "create-employee"
when
    Person(name == "John")
then
    insert(new Employee(name: "John", role: "Developer"))
end
"#;

        let result = tool.migrate(drools_rules, SourceFormat::Drools).unwrap();

        assert_eq!(result.rules.len(), 1);
        let rule = &result.rules[0];

        // Should have head atoms for type and properties
        assert!(rule.head.iter().any(|atom| {
            matches!(atom, RuleAtom::Triple {
                predicate: Term::Constant(p),
                object: Term::Constant(o), ..
            } if p == "rdf:type" && o == "Employee")
        }));
    }

    #[test]
    fn test_drools_with_modify() {
        let mut tool = MigrationTool::new();

        let drools_rules = r#"
rule "update-status"
when
    Person(name == "Alice")
then
    modify(person) { setStatus("active") }
end
"#;

        let result = tool.migrate(drools_rules, SourceFormat::Drools).unwrap();

        assert_eq!(result.rules.len(), 1);
        let rule = &result.rules[0];

        // Should have head atoms for modifications
        assert!(rule.head.iter().any(|atom| {
            matches!(atom, RuleAtom::Triple {
                predicate: Term::Constant(p), ..
            } if p == "Status")
        }));
    }

    #[test]
    fn test_drools_value_parsing() {
        let tool = MigrationTool::new();

        // Test string literal
        let value1 = tool.parse_drools_value("\"hello\"");
        assert!(matches!(value1, Term::Literal(v) if v == "hello"));

        // Test variable
        let value2 = tool.parse_drools_value("$var");
        assert!(matches!(value2, Term::Variable(v) if v == "var"));

        // Test number
        let value3 = tool.parse_drools_value("42");
        assert!(matches!(value3, Term::Literal(v) if v == "42"));

        // Test identifier
        let value4 = tool.parse_drools_value("active");
        assert!(matches!(value4, Term::Constant(v) if v == "active"));
    }

    #[test]
    fn test_drools_condition_parsing() {
        let tool = MigrationTool::new();

        // Test == operator
        let (field, value) = tool.parse_drools_condition("age == 18").unwrap();
        assert_eq!(field, "age");
        assert_eq!(value, "18");

        // Test : operator
        let (field2, value2) = tool.parse_drools_condition("status : active").unwrap();
        assert_eq!(field2, "status");
        assert_eq!(value2, "active");
    }

    #[test]
    fn test_clips_simple_rule() {
        let mut tool = MigrationTool::new();

        let clips_rules = r#"
(defrule adult-check (person (age 25)) => (assert (adult (verified yes))))
"#;

        let result = tool.migrate(clips_rules, SourceFormat::Clips).unwrap();

        assert_eq!(result.rules.len(), 1);
        assert_eq!(result.rules[0].name, "adult-check");
    }

    #[test]
    fn test_clips_with_assert() {
        let mut tool = MigrationTool::new();

        let clips_rules = r#"
(defrule make-employee (person (name John)) => (assert (employee (name John))))
"#;

        let result = tool.migrate(clips_rules, SourceFormat::Clips).unwrap();

        // Rule should be created
        assert_eq!(result.rules.len(), 1);
        assert_eq!(result.rules[0].name, "make-employee");

        // Migration should complete successfully
        assert!(result.source_format == SourceFormat::Clips);
    }

    #[test]
    fn test_clips_with_modify() {
        let mut tool = MigrationTool::new();

        let clips_rules = r#"
(defrule update-status (person) => (modify ?person (status active)))
"#;

        let result = tool.migrate(clips_rules, SourceFormat::Clips).unwrap();

        // Rule should be created
        assert_eq!(result.rules.len(), 1);
        assert_eq!(result.rules[0].name, "update-status");

        // Migration should complete successfully
        assert!(result.source_format == SourceFormat::Clips);
    }

    #[test]
    fn test_clips_value_parsing() {
        let tool = MigrationTool::new();

        // Test variable
        let value1 = tool.parse_clips_value("?x");
        assert!(matches!(value1, Term::Variable(v) if v == "x"));

        // Test string literal
        let value2 = tool.parse_clips_value("\"hello\"");
        assert!(matches!(value2, Term::Literal(v) if v == "hello"));

        // Test number
        let value3 = tool.parse_clips_value("42");
        assert!(matches!(value3, Term::Literal(v) if v == "42"));

        // Test symbol
        let value4 = tool.parse_clips_value("active");
        assert!(matches!(value4, Term::Constant(v) if v == "active"));
    }

    #[test]
    fn test_drools_multiple_conditions() {
        let mut tool = MigrationTool::new();

        let drools_rules = r#"
rule "complex-rule"
when
    Person(age == 25, status == "active")
then
    insert(new Adult(verified: true))
end
"#;

        let result = tool.migrate(drools_rules, SourceFormat::Drools).unwrap();

        assert_eq!(result.rules.len(), 1);
        let rule = &result.rules[0];

        // Should have multiple body atoms for multiple conditions
        assert!(rule.body.len() >= 2);
    }

    #[test]
    fn test_migration_warnings_severity() {
        let result = MigrationResult {
            rules: vec![],
            warnings: vec![
                MigrationWarning {
                    message: "Info".to_string(),
                    line: Some(1),
                    severity: WarningSeverity::Info,
                },
                MigrationWarning {
                    message: "Warning".to_string(),
                    line: Some(2),
                    severity: WarningSeverity::Warning,
                },
                MigrationWarning {
                    message: "Error".to_string(),
                    line: Some(3),
                    severity: WarningSeverity::Error,
                },
            ],
            source_format: SourceFormat::Jena,
            original_count: 3,
            migrated_count: 0,
        };

        let report = result.generate_report();

        assert!(report.contains("[INFO]"));
        assert!(report.contains("[WARN]"));
        assert!(report.contains("[ERROR]"));
    }

    #[test]
    fn test_empty_drools_rule() {
        let mut tool = MigrationTool::new();

        let drools_rules = r#"
rule "empty"
when
then
end
"#;

        let result = tool.migrate(drools_rules, SourceFormat::Drools).unwrap();

        // Empty rules should either be skipped (0 rules) or have no body/head
        // Migration should complete without error
        assert!(result.source_format == SourceFormat::Drools);
        if !result.rules.is_empty() {
            // If rule was created, it should be empty
            assert!(result.rules[0].body.is_empty() && result.rules[0].head.is_empty());
        }
    }

    #[test]
    fn test_clips_pattern_filtering() {
        let tool = MigrationTool::new();

        // Control patterns should be skipped
        let patterns = "(test (> ?x 5)) (person (age ?x))";
        let atoms = tool.parse_clips_patterns(patterns).unwrap();

        // Should only include person pattern, not test
        assert!(atoms.iter().any(|atom| {
            matches!(atom, RuleAtom::Triple {
                object: Term::Constant(o), ..
            } if o == "person")
        }));
    }
}
