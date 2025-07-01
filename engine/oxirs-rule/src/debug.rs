//! # Rule Engine Debugging Tools
//!
//! Advanced debugging capabilities for rule engines including execution visualization,
//! derivation path tracing, performance profiling, and conflict detection.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{self, Display};
use std::time::{Duration, Instant};

use crate::{Rule, RuleAtom, RuleEngine, Term};

/// Debug trace entry recording rule execution steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub timestamp: u64,
    pub rule_name: String,
    pub action: TraceAction,
    pub input_facts: Vec<RuleAtom>,
    pub output_facts: Vec<RuleAtom>,
    pub duration: Duration,
    pub memory_usage: usize,
}

/// Types of trace actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceAction {
    RuleExecution,
    FactAddition,
    Unification,
    BackwardChaining,
    ForwardChaining,
    ReteActivation,
}

/// Derivation path showing how a fact was derived
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivationPath {
    pub target_fact: RuleAtom,
    pub steps: Vec<DerivationStep>,
    pub total_depth: usize,
    pub involved_rules: HashSet<String>,
}

/// Single step in a derivation path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivationStep {
    pub rule_name: String,
    pub premises: Vec<RuleAtom>,
    pub conclusion: RuleAtom,
    pub unification: HashMap<String, String>,
    pub step_number: usize,
}

/// Performance metrics for rule execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_execution_time: Duration,
    pub rule_execution_times: HashMap<String, Duration>,
    pub rule_execution_counts: HashMap<String, usize>,
    pub facts_processed: usize,
    pub facts_derived: usize,
    pub memory_peak: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Rule conflict detection and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleConflict {
    pub conflict_type: ConflictType,
    pub involved_rules: Vec<String>,
    pub conflicting_facts: Vec<RuleAtom>,
    pub severity: ConflictSeverity,
    pub resolution_suggestion: String,
}

/// Types of rule conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    ContradictoryConclusions,
    CircularDependency,
    RedundantRules,
    UnreachableRules,
    PerformanceBottleneck,
}

/// Severity levels for conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictSeverity {
    Critical,
    Warning,
    Info,
}

/// Interactive debugging session
#[derive(Debug)]
pub struct DebugSession {
    pub trace: Vec<TraceEntry>,
    pub derivations: HashMap<String, DerivationPath>,
    pub metrics: PerformanceMetrics,
    pub conflicts: Vec<RuleConflict>,
    pub breakpoints: HashSet<String>,
    pub step_mode: bool,
    pub current_step: usize,
}

/// Enhanced rule engine with debugging capabilities
#[derive(Debug)]
pub struct DebuggableRuleEngine {
    pub engine: RuleEngine,
    pub debug_session: DebugSession,
    pub debug_enabled: bool,
}

impl DebuggableRuleEngine {
    /// Create a new debuggable rule engine
    pub fn new() -> Self {
        Self {
            engine: RuleEngine::new(),
            debug_session: DebugSession::new(),
            debug_enabled: false,
        }
    }

    /// Enable debugging with optional configuration
    pub fn enable_debugging(&mut self, step_mode: bool) {
        self.debug_enabled = true;
        self.debug_session.step_mode = step_mode;
        self.debug_session.clear();
    }

    /// Disable debugging
    pub fn disable_debugging(&mut self) {
        self.debug_enabled = false;
        self.debug_session.clear();
    }

    /// Add a breakpoint on a specific rule
    pub fn add_breakpoint(&mut self, rule_name: &str) {
        self.debug_session.breakpoints.insert(rule_name.to_string());
    }

    /// Remove a breakpoint
    pub fn remove_breakpoint(&mut self, rule_name: &str) {
        self.debug_session.breakpoints.remove(rule_name);
    }

    /// Execute forward chaining with debugging
    pub fn debug_forward_chain(&mut self, facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        if !self.debug_enabled {
            return self.engine.forward_chain(facts);
        }

        let start_time = Instant::now();
        let initial_memory = self.estimate_memory_usage();

        // Record fact addition
        self.record_trace_entry(TraceEntry {
            timestamp: self.current_timestamp(),
            rule_name: "__fact_addition__".to_string(),
            action: TraceAction::FactAddition,
            input_facts: vec![],
            output_facts: facts.to_vec(),
            duration: Duration::from_nanos(0),
            memory_usage: initial_memory,
        });

        // Execute with tracing
        let result = self.traced_forward_chain(facts)?;

        // Update metrics
        self.debug_session.metrics.total_execution_time = start_time.elapsed();
        self.debug_session.metrics.facts_processed = facts.len();
        self.debug_session.metrics.facts_derived = result.len() - facts.len();
        self.debug_session.metrics.memory_peak = self.estimate_memory_usage();

        // Analyze for conflicts
        self.analyze_conflicts();

        Ok(result)
    }

    /// Execute backward chaining with debugging
    pub fn debug_backward_chain(&mut self, goal: &RuleAtom) -> Result<bool> {
        if !self.debug_enabled {
            return self.engine.backward_chain(goal);
        }

        let start_time = Instant::now();

        // Build derivation path
        let derivation = self.build_derivation_path(goal)?;
        if let Some(path) = derivation {
            let goal_key = format!("{:?}", goal);
            self.debug_session.derivations.insert(goal_key, path);
        }

        let result = self.traced_backward_chain(goal)?;

        // Record performance
        let duration = start_time.elapsed();
        self.record_trace_entry(TraceEntry {
            timestamp: self.current_timestamp(),
            rule_name: "__backward_chain__".to_string(),
            action: TraceAction::BackwardChaining,
            input_facts: vec![goal.clone()],
            output_facts: if result { vec![goal.clone()] } else { vec![] },
            duration,
            memory_usage: self.estimate_memory_usage(),
        });

        Ok(result)
    }

    /// Get execution trace
    pub fn get_trace(&self) -> &[TraceEntry] {
        &self.debug_session.trace
    }

    /// Get derivation path for a fact
    pub fn get_derivation(&self, fact: &RuleAtom) -> Option<&DerivationPath> {
        let key = format!("{:?}", fact);
        self.debug_session.derivations.get(&key)
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.debug_session.metrics
    }

    /// Get detected conflicts
    pub fn get_conflicts(&self) -> &[RuleConflict] {
        &self.debug_session.conflicts
    }

    /// Generate debug report
    pub fn generate_debug_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== RULE ENGINE DEBUG REPORT ===\n\n");

        // Performance metrics
        report.push_str("PERFORMANCE METRICS:\n");
        report.push_str(&format!(
            "Total execution time: {:?}\n",
            self.debug_session.metrics.total_execution_time
        ));
        report.push_str(&format!(
            "Facts processed: {}\n",
            self.debug_session.metrics.facts_processed
        ));
        report.push_str(&format!(
            "Facts derived: {}\n",
            self.debug_session.metrics.facts_derived
        ));
        report.push_str(&format!(
            "Memory peak: {} bytes\n",
            self.debug_session.metrics.memory_peak
        ));
        report.push_str(&format!(
            "Cache hits: {}\n",
            self.debug_session.metrics.cache_hits
        ));
        report.push_str(&format!(
            "Cache misses: {}\n",
            self.debug_session.metrics.cache_misses
        ));
        report.push_str("\n");

        // Rule execution times
        report.push_str("RULE EXECUTION TIMES:\n");
        let mut rule_times: Vec<_> = self
            .debug_session
            .metrics
            .rule_execution_times
            .iter()
            .collect();
        rule_times.sort_by(|a, b| b.1.cmp(a.1));
        for (rule, time) in rule_times.iter().take(10) {
            let count = self
                .debug_session
                .metrics
                .rule_execution_counts
                .get(*rule)
                .unwrap_or(&0);
            report.push_str(&format!(
                "  {}: {:?} (executed {} times)\n",
                rule, time, count
            ));
        }
        report.push_str("\n");

        // Conflicts
        if !self.debug_session.conflicts.is_empty() {
            report.push_str("DETECTED CONFLICTS:\n");
            for conflict in &self.debug_session.conflicts {
                report.push_str(&format!(
                    "  {:?} - {:?}: {}\n",
                    conflict.severity, conflict.conflict_type, conflict.resolution_suggestion
                ));
            }
            report.push_str("\n");
        }

        // Derivation paths
        if !self.debug_session.derivations.is_empty() {
            report.push_str("DERIVATION PATHS:\n");
            for (fact, path) in &self.debug_session.derivations {
                report.push_str(&format!("  Fact: {}\n", fact));
                report.push_str(&format!("  Depth: {}\n", path.total_depth));
                report.push_str(&format!("  Rules involved: {:?}\n", path.involved_rules));
                for step in &path.steps {
                    report.push_str(&format!(
                        "    Step {}: {} -> {:?}\n",
                        step.step_number, step.rule_name, step.conclusion
                    ));
                }
                report.push_str("\n");
            }
        }

        report
    }

    /// Export debug data to JSON
    pub fn export_debug_data(&self) -> Result<String> {
        #[derive(Serialize)]
        struct DebugData {
            trace: Vec<TraceEntry>,
            derivations: HashMap<String, DerivationPath>,
            metrics: PerformanceMetrics,
            conflicts: Vec<RuleConflict>,
        }

        let data = DebugData {
            trace: self.debug_session.trace.clone(),
            derivations: self.debug_session.derivations.clone(),
            metrics: self.debug_session.metrics.clone(),
            conflicts: self.debug_session.conflicts.clone(),
        };

        serde_json::to_string_pretty(&data)
            .map_err(|e| anyhow!("Failed to serialize debug data: {}", e))
    }

    // Private implementation methods

    fn traced_forward_chain(&mut self, facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        // Implementation would integrate with the actual forward chaining
        // For now, delegate to the underlying engine
        self.engine.forward_chain(facts)
    }

    fn traced_backward_chain(&mut self, goal: &RuleAtom) -> Result<bool> {
        // Implementation would integrate with the actual backward chaining
        // For now, delegate to the underlying engine
        self.engine.backward_chain(goal)
    }

    fn build_derivation_path(&self, goal: &RuleAtom) -> Result<Option<DerivationPath>> {
        // Placeholder implementation - would build actual derivation tree
        Ok(Some(DerivationPath {
            target_fact: goal.clone(),
            steps: vec![],
            total_depth: 0,
            involved_rules: HashSet::new(),
        }))
    }

    fn analyze_conflicts(&mut self) {
        // Analyze rule conflicts and populate conflicts vector
        self.detect_circular_dependencies();
        self.detect_contradictory_rules();
        self.detect_redundant_rules();
        self.detect_performance_bottlenecks();
    }

    fn detect_circular_dependencies(&mut self) {
        // Implementation for detecting circular rule dependencies
        // This would analyze the rule dependency graph
    }

    fn detect_contradictory_rules(&mut self) {
        // Implementation for detecting contradictory rule conclusions
    }

    fn detect_redundant_rules(&mut self) {
        // Implementation for detecting redundant rules
    }

    fn detect_performance_bottlenecks(&mut self) {
        // Analyze performance metrics to identify bottlenecks
        for (rule_name, duration) in &self.debug_session.metrics.rule_execution_times {
            if duration.as_millis() > 100 {
                // Threshold for slow rules
                self.debug_session.conflicts.push(RuleConflict {
                    conflict_type: ConflictType::PerformanceBottleneck,
                    involved_rules: vec![rule_name.clone()],
                    conflicting_facts: vec![],
                    severity: ConflictSeverity::Warning,
                    resolution_suggestion: format!(
                        "Rule '{}' is slow ({}ms). Consider optimization.",
                        rule_name,
                        duration.as_millis()
                    ),
                });
            }
        }
    }

    fn record_trace_entry(&mut self, entry: TraceEntry) {
        // Extract values before moving the entry
        let rule_name = entry.rule_name.clone();
        let duration = entry.duration;

        self.debug_session.trace.push(entry);

        // Update metrics
        if let Some(current_count) = self
            .debug_session
            .metrics
            .rule_execution_counts
            .get_mut(&rule_name)
        {
            *current_count += 1;
        } else {
            self.debug_session
                .metrics
                .rule_execution_counts
                .insert(rule_name.clone(), 1);
        }

        // Update timing
        if let Some(current_time) = self
            .debug_session
            .metrics
            .rule_execution_times
            .get_mut(&rule_name)
        {
            *current_time += duration;
        } else {
            self.debug_session
                .metrics
                .rule_execution_times
                .insert(rule_name, duration);
        }
    }

    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn estimate_memory_usage(&self) -> usize {
        // Rough estimation of memory usage
        std::mem::size_of::<RuleEngine>()
            + self.debug_session.trace.len() * std::mem::size_of::<TraceEntry>()
            + self.debug_session.derivations.len() * 1024 // Rough estimate
    }
}

impl DebugSession {
    fn new() -> Self {
        Self {
            trace: Vec::new(),
            derivations: HashMap::new(),
            metrics: PerformanceMetrics::default(),
            conflicts: Vec::new(),
            breakpoints: HashSet::new(),
            step_mode: false,
            current_step: 0,
        }
    }

    fn clear(&mut self) {
        self.trace.clear();
        self.derivations.clear();
        self.conflicts.clear();
        self.metrics = PerformanceMetrics::default();
        self.current_step = 0;
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_execution_time: Duration::from_nanos(0),
            rule_execution_times: HashMap::new(),
            rule_execution_counts: HashMap::new(),
            facts_processed: 0,
            facts_derived: 0,
            memory_peak: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}

impl Display for ConflictType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConflictType::ContradictoryConclusions => write!(f, "Contradictory Conclusions"),
            ConflictType::CircularDependency => write!(f, "Circular Dependency"),
            ConflictType::RedundantRules => write!(f, "Redundant Rules"),
            ConflictType::UnreachableRules => write!(f, "Unreachable Rules"),
            ConflictType::PerformanceBottleneck => write!(f, "Performance Bottleneck"),
        }
    }
}

impl Display for ConflictSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConflictSeverity::Critical => write!(f, "CRITICAL"),
            ConflictSeverity::Warning => write!(f, "WARNING"),
            ConflictSeverity::Info => write!(f, "INFO"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debuggable_engine_creation() {
        let engine = DebuggableRuleEngine::new();
        assert!(!engine.debug_enabled);
        assert!(engine.debug_session.trace.is_empty());
    }

    #[test]
    fn test_debugging_enable_disable() {
        let mut engine = DebuggableRuleEngine::new();

        engine.enable_debugging(true);
        assert!(engine.debug_enabled);
        assert!(engine.debug_session.step_mode);

        engine.disable_debugging();
        assert!(!engine.debug_enabled);
    }

    #[test]
    fn test_breakpoint_management() {
        let mut engine = DebuggableRuleEngine::new();

        engine.add_breakpoint("test_rule");
        assert!(engine.debug_session.breakpoints.contains("test_rule"));

        engine.remove_breakpoint("test_rule");
        assert!(!engine.debug_session.breakpoints.contains("test_rule"));
    }

    #[test]
    fn test_trace_entry_recording() {
        let mut engine = DebuggableRuleEngine::new();
        engine.enable_debugging(false);

        let entry = TraceEntry {
            timestamp: engine.current_timestamp(),
            rule_name: "test_rule".to_string(),
            action: TraceAction::RuleExecution,
            input_facts: vec![],
            output_facts: vec![],
            duration: Duration::from_millis(10),
            memory_usage: 1024,
        };

        engine.record_trace_entry(entry);
        assert_eq!(engine.debug_session.trace.len(), 1);
        assert_eq!(
            engine
                .debug_session
                .metrics
                .rule_execution_counts
                .get("test_rule"),
            Some(&1)
        );
    }

    #[test]
    fn test_performance_bottleneck_detection() {
        let mut engine = DebuggableRuleEngine::new();
        engine.enable_debugging(false);

        // Add a slow rule execution
        engine
            .debug_session
            .metrics
            .rule_execution_times
            .insert("slow_rule".to_string(), Duration::from_millis(200));

        engine.analyze_conflicts();

        // Should detect performance bottleneck
        assert!(!engine.debug_session.conflicts.is_empty());
        assert!(engine
            .debug_session
            .conflicts
            .iter()
            .any(|c| { matches!(c.conflict_type, ConflictType::PerformanceBottleneck) }));
    }

    #[test]
    fn test_debug_report_generation() {
        let mut engine = DebuggableRuleEngine::new();
        engine.enable_debugging(false);

        // Add some test data
        engine.debug_session.metrics.facts_processed = 100;
        engine.debug_session.metrics.facts_derived = 50;
        engine.debug_session.metrics.total_execution_time = Duration::from_millis(500);

        let report = engine.generate_debug_report();
        assert!(report.contains("RULE ENGINE DEBUG REPORT"));
        assert!(report.contains("Facts processed: 100"));
        assert!(report.contains("Facts derived: 50"));
    }

    #[test]
    fn test_debug_data_export() {
        let engine = DebuggableRuleEngine::new();
        let json_result = engine.export_debug_data();
        assert!(json_result.is_ok());

        let json_data = json_result.unwrap();
        assert!(json_data.contains("trace"));
        assert!(json_data.contains("metrics"));
        assert!(json_data.contains("conflicts"));
    }
}
