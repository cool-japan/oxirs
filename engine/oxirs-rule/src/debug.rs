//! # Rule Engine Debugging Tools
//!
//! Advanced debugging capabilities for rule engines including execution visualization,
//! derivation path tracing, performance profiling, and conflict detection.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};
use std::time::{Duration, Instant};

use crate::{RuleAtom, RuleEngine};

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

/// Debugger state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebuggerState {
    /// Running normally
    Running,
    /// Paused at breakpoint
    Paused,
    /// Stepping through execution
    Stepping,
    /// Execution complete
    Finished,
}

/// Debug command for interactive debugging
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DebugCommand {
    /// Continue execution until next breakpoint
    Continue,
    /// Step to next rule execution
    Step,
    /// Step over current rule (execute without entering)
    Next,
    /// Step out of current rule
    StepOut,
    /// Print current state
    Print,
    /// List breakpoints
    ListBreakpoints,
    /// Show call stack
    Backtrace,
    /// Quit debugging
    Quit,
}

/// Stack frame for call stack tracking
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// Rule name
    pub rule_name: String,
    /// Current substitutions
    pub substitutions: HashMap<String, String>,
    /// Input facts for this frame
    pub input_facts: Vec<RuleAtom>,
    /// Frame depth
    pub depth: usize,
}

/// Watch expression for monitoring values
#[derive(Debug, Clone)]
pub struct WatchExpression {
    /// Expression name/description
    pub name: String,
    /// Variable to watch
    pub variable: String,
    /// Last known value
    pub last_value: Option<String>,
    /// Break when value changes
    pub break_on_change: bool,
}

/// Conditional breakpoint
#[derive(Debug, Clone)]
pub struct ConditionalBreakpoint {
    /// Rule name to break on
    pub rule_name: String,
    /// Condition that must be true (variable = value)
    pub condition: Option<BreakpointCondition>,
    /// Hit count (break after N hits)
    pub hit_count: Option<usize>,
    /// Current hit count
    pub current_hits: usize,
    /// Is breakpoint enabled
    pub enabled: bool,
}

/// Breakpoint condition
#[derive(Debug, Clone)]
pub struct BreakpointCondition {
    /// Variable name
    pub variable: String,
    /// Expected value
    pub value: String,
    /// Comparison operator
    pub operator: ConditionOperator,
}

/// Condition operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionOperator {
    /// Equal
    Equals,
    /// Not equal
    NotEquals,
    /// Contains substring
    Contains,
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
    /// Debugger state
    pub state: DebuggerState,
    /// Call stack
    pub call_stack: Vec<StackFrame>,
    /// Conditional breakpoints
    pub conditional_breakpoints: Vec<ConditionalBreakpoint>,
    /// Watch expressions
    pub watch_expressions: Vec<WatchExpression>,
    /// Current substitutions (variable bindings)
    pub current_substitutions: HashMap<String, String>,
    /// Paused events queue (for async notification)
    pub paused_events: Vec<String>,
}

/// Enhanced rule engine with debugging capabilities
#[derive(Debug)]
pub struct DebuggableRuleEngine {
    pub engine: RuleEngine,
    pub debug_session: DebugSession,
    pub debug_enabled: bool,
}

impl Default for DebuggableRuleEngine {
    fn default() -> Self {
        Self::new()
    }
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
            let goal_key = format!("{goal:?}");
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
        let key = format!("{fact:?}");
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
        report.push('\n');

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
            report.push_str(&format!("  {rule}: {time:?} (executed {count} times)\n"));
        }
        report.push('\n');

        // Conflicts
        if !self.debug_session.conflicts.is_empty() {
            report.push_str("DETECTED CONFLICTS:\n");
            for conflict in &self.debug_session.conflicts {
                report.push_str(&format!(
                    "  {:?} - {:?}: {}\n",
                    conflict.severity, conflict.conflict_type, conflict.resolution_suggestion
                ));
            }
            report.push('\n');
        }

        // Derivation paths
        if !self.debug_session.derivations.is_empty() {
            report.push_str("DERIVATION PATHS:\n");
            for (fact, path) in &self.debug_session.derivations {
                report.push_str(&format!("  Fact: {fact}\n"));
                report.push_str(&format!("  Depth: {}\n", path.total_depth));
                report.push_str(&format!(
                    "  Rules involved: {}\n",
                    path.involved_rules
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                for step in &path.steps {
                    report.push_str(&format!(
                        "    Step {}: {} -> {:?}\n",
                        step.step_number, step.rule_name, step.conclusion
                    ));
                }
                report.push('\n');
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
            state: DebuggerState::Running,
            call_stack: Vec::new(),
            conditional_breakpoints: Vec::new(),
            watch_expressions: Vec::new(),
            current_substitutions: HashMap::new(),
            paused_events: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.trace.clear();
        self.derivations.clear();
        self.conflicts.clear();
        self.metrics = PerformanceMetrics::default();
        self.current_step = 0;
        self.state = DebuggerState::Running;
        self.call_stack.clear();
        self.current_substitutions.clear();
        self.paused_events.clear();
        // Keep breakpoints and watch expressions
    }
}

impl DebuggableRuleEngine {
    /// Add a conditional breakpoint
    pub fn add_conditional_breakpoint(
        &mut self,
        rule_name: &str,
        condition: Option<BreakpointCondition>,
        hit_count: Option<usize>,
    ) {
        self.debug_session
            .conditional_breakpoints
            .push(ConditionalBreakpoint {
                rule_name: rule_name.to_string(),
                condition,
                hit_count,
                current_hits: 0,
                enabled: true,
            });
    }

    /// Add a watch expression
    pub fn add_watch(&mut self, name: &str, variable: &str, break_on_change: bool) {
        self.debug_session.watch_expressions.push(WatchExpression {
            name: name.to_string(),
            variable: variable.to_string(),
            last_value: None,
            break_on_change,
        });
    }

    /// Remove a watch expression
    pub fn remove_watch(&mut self, name: &str) {
        self.debug_session
            .watch_expressions
            .retain(|w| w.name != name);
    }

    /// Get current call stack
    pub fn get_call_stack(&self) -> &[StackFrame] {
        &self.debug_session.call_stack
    }

    /// Get current variable bindings
    pub fn get_substitutions(&self) -> &HashMap<String, String> {
        &self.debug_session.current_substitutions
    }

    /// Get value of a specific variable
    pub fn get_variable(&self, name: &str) -> Option<&String> {
        self.debug_session.current_substitutions.get(name)
    }

    /// Set a variable value (for debugging)
    pub fn set_variable(&mut self, name: &str, value: &str) {
        self.debug_session
            .current_substitutions
            .insert(name.to_string(), value.to_string());
    }

    /// Execute a debug command
    pub fn execute_command(&mut self, command: DebugCommand) -> String {
        match command {
            DebugCommand::Continue => {
                self.debug_session.state = DebuggerState::Running;
                self.debug_session.step_mode = false;
                "Continuing execution...".to_string()
            }
            DebugCommand::Step => {
                self.debug_session.state = DebuggerState::Stepping;
                self.debug_session.step_mode = true;
                "Stepping to next rule...".to_string()
            }
            DebugCommand::Next => {
                // Step over - execute current rule without entering
                self.debug_session.state = DebuggerState::Stepping;
                "Stepping over...".to_string()
            }
            DebugCommand::StepOut => {
                // Step out of current frame
                if !self.debug_session.call_stack.is_empty() {
                    self.debug_session.call_stack.pop();
                    "Stepping out of current frame...".to_string()
                } else {
                    "No frame to step out of".to_string()
                }
            }
            DebugCommand::Print => self.format_current_state(),
            DebugCommand::ListBreakpoints => self.format_breakpoints(),
            DebugCommand::Backtrace => self.format_backtrace(),
            DebugCommand::Quit => {
                self.debug_session.state = DebuggerState::Finished;
                "Quitting debugger...".to_string()
            }
        }
    }

    /// Check if execution should pause
    pub fn should_pause(&self, rule_name: &str) -> bool {
        // Check step mode
        if self.debug_session.step_mode {
            return true;
        }

        // Check simple breakpoints
        if self.debug_session.breakpoints.contains(rule_name) {
            return true;
        }

        // Check conditional breakpoints
        for bp in &self.debug_session.conditional_breakpoints {
            if bp.enabled && bp.rule_name == rule_name {
                // Check hit count
                if let Some(hit_count) = bp.hit_count {
                    if bp.current_hits >= hit_count {
                        continue;
                    }
                }

                // Check condition
                if let Some(ref condition) = bp.condition {
                    if !self.evaluate_condition(condition) {
                        continue;
                    }
                }

                return true;
            }
        }

        false
    }

    /// Evaluate a breakpoint condition
    fn evaluate_condition(&self, condition: &BreakpointCondition) -> bool {
        if let Some(value) = self
            .debug_session
            .current_substitutions
            .get(&condition.variable)
        {
            match condition.operator {
                ConditionOperator::Equals => value == &condition.value,
                ConditionOperator::NotEquals => value != &condition.value,
                ConditionOperator::Contains => value.contains(&condition.value),
            }
        } else {
            false
        }
    }

    /// Push a stack frame
    pub fn push_frame(&mut self, rule_name: &str, input_facts: Vec<RuleAtom>) {
        let depth = self.debug_session.call_stack.len();
        self.debug_session.call_stack.push(StackFrame {
            rule_name: rule_name.to_string(),
            substitutions: self.debug_session.current_substitutions.clone(),
            input_facts,
            depth,
        });
    }

    /// Pop a stack frame
    pub fn pop_frame(&mut self) -> Option<StackFrame> {
        self.debug_session.call_stack.pop()
    }

    /// Update substitutions
    pub fn update_substitutions(&mut self, substitutions: HashMap<String, String>) {
        // Check watch expressions for changes
        for watch in &mut self.debug_session.watch_expressions {
            if let Some(new_value) = substitutions.get(&watch.variable) {
                let changed = watch.last_value.as_ref() != Some(new_value);
                if changed && watch.break_on_change {
                    self.debug_session.paused_events.push(format!(
                        "Watch '{}' changed: {:?} -> {}",
                        watch.name, watch.last_value, new_value
                    ));
                    self.debug_session.state = DebuggerState::Paused;
                }
                watch.last_value = Some(new_value.clone());
            }
        }

        self.debug_session.current_substitutions = substitutions;
    }

    /// Format current debugger state
    fn format_current_state(&self) -> String {
        let mut output = String::new();

        output.push_str("=== Current State ===\n");
        output.push_str(&format!("Debugger: {:?}\n", self.debug_session.state));
        output.push_str(&format!("Step: {}\n", self.debug_session.current_step));

        output.push_str("\nVariables:\n");
        for (name, value) in &self.debug_session.current_substitutions {
            output.push_str(&format!("  {} = {}\n", name, value));
        }

        if !self.debug_session.watch_expressions.is_empty() {
            output.push_str("\nWatches:\n");
            for watch in &self.debug_session.watch_expressions {
                output.push_str(&format!(
                    "  {}: {} = {:?}\n",
                    watch.name, watch.variable, watch.last_value
                ));
            }
        }

        output
    }

    /// Format breakpoints list
    fn format_breakpoints(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Breakpoints ===\n");

        // Simple breakpoints
        for bp in &self.debug_session.breakpoints {
            output.push_str(&format!("  {} (simple)\n", bp));
        }

        // Conditional breakpoints
        for (i, bp) in self
            .debug_session
            .conditional_breakpoints
            .iter()
            .enumerate()
        {
            let status = if bp.enabled { "enabled" } else { "disabled" };
            let condition = bp
                .condition
                .as_ref()
                .map(|c| format!(" when {} {:?} {}", c.variable, c.operator, c.value))
                .unwrap_or_default();
            let hit_info = bp
                .hit_count
                .map(|h| format!(" (hit {}/{})", bp.current_hits, h))
                .unwrap_or_default();

            output.push_str(&format!(
                "  [{}] {}{}{} ({})\n",
                i, bp.rule_name, condition, hit_info, status
            ));
        }

        output
    }

    /// Format call stack backtrace
    fn format_backtrace(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Call Stack ===\n");

        for (i, frame) in self.debug_session.call_stack.iter().rev().enumerate() {
            output.push_str(&format!(
                "  #{} {} (depth: {})\n",
                i, frame.rule_name, frame.depth
            ));

            if !frame.substitutions.is_empty() {
                output.push_str("    Bindings:\n");
                for (var, val) in &frame.substitutions {
                    output.push_str(&format!("      {} = {}\n", var, val));
                }
            }
        }

        if self.debug_session.call_stack.is_empty() {
            output.push_str("  (empty)\n");
        }

        output
    }

    /// Get debugger state
    pub fn get_state(&self) -> DebuggerState {
        self.debug_session.state
    }

    /// Check if debugger is paused
    pub fn is_paused(&self) -> bool {
        self.debug_session.state == DebuggerState::Paused
    }

    /// Get paused events
    pub fn get_paused_events(&mut self) -> Vec<String> {
        std::mem::take(&mut self.debug_session.paused_events)
    }

    /// Increment hit counter for conditional breakpoint
    pub fn increment_hit_count(&mut self, rule_name: &str) {
        for bp in &mut self.debug_session.conditional_breakpoints {
            if bp.rule_name == rule_name {
                bp.current_hits += 1;
            }
        }
    }

    /// Reset hit counts for all breakpoints
    pub fn reset_hit_counts(&mut self) {
        for bp in &mut self.debug_session.conditional_breakpoints {
            bp.current_hits = 0;
        }
    }

    /// Enable/disable a conditional breakpoint
    pub fn set_breakpoint_enabled(&mut self, index: usize, enabled: bool) -> bool {
        if let Some(bp) = self.debug_session.conditional_breakpoints.get_mut(index) {
            bp.enabled = enabled;
            true
        } else {
            false
        }
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

    #[test]
    fn test_conditional_breakpoint() {
        let mut engine = DebuggableRuleEngine::new();
        engine.enable_debugging(false);

        // Add conditional breakpoint
        let condition = BreakpointCondition {
            variable: "X".to_string(),
            value: "test".to_string(),
            operator: ConditionOperator::Equals,
        };
        engine.add_conditional_breakpoint("test_rule", Some(condition), None);

        // Set variable to meet condition
        engine.set_variable("X", "test");

        // Should pause
        assert!(engine.should_pause("test_rule"));

        // Different value should not pause
        engine.set_variable("X", "other");
        assert!(!engine.should_pause("test_rule"));
    }

    #[test]
    fn test_hit_count_breakpoint() {
        let mut engine = DebuggableRuleEngine::new();
        engine.enable_debugging(false);

        // Add breakpoint with hit count
        engine.add_conditional_breakpoint("count_rule", None, Some(3));

        // Should pause first 3 times
        assert!(engine.should_pause("count_rule"));
        engine.increment_hit_count("count_rule");
        assert!(engine.should_pause("count_rule"));
        engine.increment_hit_count("count_rule");
        assert!(engine.should_pause("count_rule"));
        engine.increment_hit_count("count_rule");

        // After 3 hits, should not pause
        assert!(!engine.should_pause("count_rule"));
    }

    #[test]
    fn test_watch_expressions() {
        let mut engine = DebuggableRuleEngine::new();
        engine.enable_debugging(false);

        engine.add_watch("x_watch", "X", true);

        // Initial update
        let mut subs = HashMap::new();
        subs.insert("X".to_string(), "value1".to_string());
        engine.update_substitutions(subs);

        // Check watch has value
        assert!(engine.debug_session.watch_expressions[0]
            .last_value
            .is_some());

        // Update with new value - should trigger pause
        let mut subs2 = HashMap::new();
        subs2.insert("X".to_string(), "value2".to_string());
        engine.update_substitutions(subs2);

        assert!(engine.is_paused());
        let events = engine.get_paused_events();
        assert!(!events.is_empty());
    }

    #[test]
    fn test_call_stack() {
        let mut engine = DebuggableRuleEngine::new();
        engine.enable_debugging(false);

        // Push frames
        engine.push_frame("rule1", vec![]);
        engine.push_frame("rule2", vec![]);
        engine.push_frame("rule3", vec![]);

        // Check stack depth
        assert_eq!(engine.get_call_stack().len(), 3);

        // Pop frame
        let frame = engine.pop_frame();
        assert!(frame.is_some());
        assert_eq!(frame.unwrap().rule_name, "rule3");

        // Check stack after pop
        assert_eq!(engine.get_call_stack().len(), 2);
    }

    #[test]
    fn test_debug_commands() {
        let mut engine = DebuggableRuleEngine::new();
        engine.enable_debugging(true);

        // Test step command
        let result = engine.execute_command(DebugCommand::Step);
        assert!(result.contains("Stepping"));
        assert!(engine.debug_session.step_mode);

        // Test continue command
        let result = engine.execute_command(DebugCommand::Continue);
        assert!(result.contains("Continuing"));
        assert!(!engine.debug_session.step_mode);

        // Test quit command
        let result = engine.execute_command(DebugCommand::Quit);
        assert!(result.contains("Quitting"));
        assert_eq!(engine.get_state(), DebuggerState::Finished);
    }

    #[test]
    fn test_format_backtrace() {
        let mut engine = DebuggableRuleEngine::new();
        engine.enable_debugging(false);

        // Add some frames with substitutions
        engine.set_variable("X", "value1");
        engine.push_frame("rule1", vec![]);
        engine.set_variable("Y", "value2");
        engine.push_frame("rule2", vec![]);

        let backtrace = engine.execute_command(DebugCommand::Backtrace);
        assert!(backtrace.contains("Call Stack"));
        assert!(backtrace.contains("rule1"));
        assert!(backtrace.contains("rule2"));
    }

    #[test]
    fn test_breakpoint_enable_disable() {
        let mut engine = DebuggableRuleEngine::new();
        engine.enable_debugging(false);

        engine.add_conditional_breakpoint("test_rule", None, None);

        // Initially enabled
        assert!(engine.should_pause("test_rule"));

        // Disable
        engine.set_breakpoint_enabled(0, false);
        assert!(!engine.should_pause("test_rule"));

        // Re-enable
        engine.set_breakpoint_enabled(0, true);
        assert!(engine.should_pause("test_rule"));
    }

    #[test]
    fn test_condition_operators() {
        let mut engine = DebuggableRuleEngine::new();
        engine.enable_debugging(false);

        // Contains operator
        let condition = BreakpointCondition {
            variable: "X".to_string(),
            value: "sub".to_string(),
            operator: ConditionOperator::Contains,
        };
        engine.add_conditional_breakpoint("rule1", Some(condition), None);

        engine.set_variable("X", "substring");
        assert!(engine.should_pause("rule1"));

        // Not equals operator
        engine.debug_session.conditional_breakpoints.clear();
        let condition2 = BreakpointCondition {
            variable: "Y".to_string(),
            value: "excluded".to_string(),
            operator: ConditionOperator::NotEquals,
        };
        engine.add_conditional_breakpoint("rule2", Some(condition2), None);

        engine.set_variable("Y", "included");
        assert!(engine.should_pause("rule2"));

        engine.set_variable("Y", "excluded");
        assert!(!engine.should_pause("rule2"));
    }

    #[test]
    fn test_step_out_command() {
        let mut engine = DebuggableRuleEngine::new();
        engine.enable_debugging(false);

        // Push some frames
        engine.push_frame("rule1", vec![]);
        engine.push_frame("rule2", vec![]);

        // Step out should pop frame
        let result = engine.execute_command(DebugCommand::StepOut);
        assert!(result.contains("Stepping out"));
        assert_eq!(engine.get_call_stack().len(), 1);

        // Step out again
        engine.execute_command(DebugCommand::StepOut);
        assert_eq!(engine.get_call_stack().len(), 0);

        // Step out with empty stack
        let result = engine.execute_command(DebugCommand::StepOut);
        assert!(result.contains("No frame"));
    }
}
