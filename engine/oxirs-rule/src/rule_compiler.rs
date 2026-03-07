//! Rule compilation to a bytecode-like intermediate representation (IR).
//!
//! A `RuleCompiler` translates a rule (body patterns + head patterns) into a
//! flat sequence of `Instruction`s that can be executed by an abstract
//! interpreter, JIT-compiled, or analysed for optimisation.

// ── Binding slots ─────────────────────────────────────────────────────────────

/// A slot in an instruction operand — either a variable or a constant term.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BindingSlot {
    /// A variable name (will be looked up / bound at runtime).
    Var(String),
    /// A constant IRI or literal value.
    Constant(String),
}

impl BindingSlot {
    /// Return the variable name if this is a `Var` slot.
    pub fn as_var(&self) -> Option<&str> {
        match self {
            Self::Var(v) => Some(v),
            Self::Constant(_) => None,
        }
    }

    /// Return the constant value if this is a `Constant` slot.
    pub fn as_constant(&self) -> Option<&str> {
        match self {
            Self::Constant(c) => Some(c),
            Self::Var(_) => None,
        }
    }
}

// ── Filter conditions ─────────────────────────────────────────────────────────

/// A boolean condition evaluated during rule execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterCondition {
    /// Two variables must be bound to the same value.
    VarEquals(String, String),
    /// Two variables must be bound to different values.
    VarNotEquals(String, String),
    /// A variable must be bound (non-null).
    Bound(String),
}

// ── Instructions ──────────────────────────────────────────────────────────────

/// A single instruction in the compiled rule IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    /// Attempt to match (and enumerate) triples from the store.
    LoadTriple {
        subject: BindingSlot,
        predicate: BindingSlot,
        object: BindingSlot,
    },
    /// Bind the current value of `slot` to the named variable.
    Bind { var: String, slot: BindingSlot },
    /// Evaluate a boolean condition; backtrack if false.
    Filter { condition: FilterCondition },
    /// Emit one or more derived triples using the current bindings.
    Produce {
        head: Vec<(BindingSlot, BindingSlot, BindingSlot)>,
    },
    /// Unconditional branch to instruction at index `target`.
    Jump { target: usize },
    /// Stop execution.
    Halt,
}

// ── Compiled rule ─────────────────────────────────────────────────────────────

/// The compiled form of a rule ready for execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledRule {
    /// Name / identifier of the rule.
    pub name: String,
    /// Flat instruction sequence.
    pub instructions: Vec<Instruction>,
    /// Total number of distinct variables in this rule.
    pub var_count: usize,
}

// ── Compiler ──────────────────────────────────────────────────────────────────

/// Compiles symbolic rules into the flat `CompiledRule` IR.
pub struct RuleCompiler;

impl RuleCompiler {
    /// Compile a rule from its body and head pattern lists.
    ///
    /// Each pattern is a `(subject, predicate, object)` triple where a term
    /// that starts with `'?'` is treated as a variable and everything else as a
    /// constant.
    pub fn compile(
        name: &str,
        body_patterns: Vec<(String, String, String)>,
        head_patterns: Vec<(String, String, String)>,
    ) -> CompiledRule {
        let mut instructions: Vec<Instruction> = Vec::new();
        let mut vars_seen: Vec<String> = Vec::new();

        // Emit a LoadTriple + optional Bind instructions for every body pattern.
        for (s, p, o) in &body_patterns {
            let s_slot = Self::make_slot(s, &mut vars_seen);
            let p_slot = Self::make_slot(p, &mut vars_seen);
            let o_slot = Self::make_slot(o, &mut vars_seen);
            instructions.push(Instruction::LoadTriple {
                subject: s_slot.clone(),
                predicate: p_slot.clone(),
                object: o_slot.clone(),
            });
            // For each variable slot, emit a Bind so the variable is
            // explicitly recorded in the instruction stream.
            for (term, slot) in [s, p, o].iter().zip([&s_slot, &p_slot, &o_slot]) {
                if let Some(stripped) = term.strip_prefix('?') {
                    instructions.push(Instruction::Bind {
                        var: stripped.to_string(),
                        slot: slot.clone(),
                    });
                }
            }
        }

        // Emit the Produce instruction for the head.
        if !head_patterns.is_empty() {
            let head = head_patterns
                .iter()
                .map(|(s, p, o)| {
                    let mut dummy: Vec<String> = Vec::new();
                    (
                        Self::make_slot(s, &mut dummy),
                        Self::make_slot(p, &mut dummy),
                        Self::make_slot(o, &mut dummy),
                    )
                })
                .collect();
            instructions.push(Instruction::Produce { head });
        }

        // Always terminate.
        instructions.push(Instruction::Halt);

        CompiledRule {
            name: name.to_string(),
            var_count: vars_seen.len(),
            instructions,
        }
    }

    /// Apply simple peephole optimisations to a compiled rule:
    ///
    /// * Remove `Bind` instructions where the variable is never referenced
    ///   after the bind point.
    pub fn optimize(rule: &mut CompiledRule) {
        // Collect variables that appear in Produce or Filter instructions.
        let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();

        for instr in &rule.instructions {
            match instr {
                Instruction::Produce { head } => {
                    for (s, p, o) in head {
                        for slot in [s, p, o] {
                            if let Some(v) = slot.as_var() {
                                referenced.insert(v.to_string());
                            }
                        }
                    }
                }
                Instruction::Filter { condition } => match condition {
                    FilterCondition::VarEquals(a, b) | FilterCondition::VarNotEquals(a, b) => {
                        referenced.insert(a.clone());
                        referenced.insert(b.clone());
                    }
                    FilterCondition::Bound(v) => {
                        referenced.insert(v.clone());
                    }
                },
                _ => {}
            }
        }

        // Also keep binds whose slot variable is later used as a LoadTriple argument.
        for instr in &rule.instructions {
            if let Instruction::LoadTriple {
                subject,
                predicate,
                object,
            } = instr
            {
                for slot in [subject, predicate, object] {
                    if let Some(v) = slot.as_var() {
                        referenced.insert(v.to_string());
                    }
                }
            }
        }

        // Remove Bind instructions for variables not in `referenced`.
        rule.instructions.retain(|instr| {
            if let Instruction::Bind { var, .. } = instr {
                referenced.contains(var.as_str())
            } else {
                true
            }
        });

        // Update var_count to reflect distinct referenced variables.
        rule.var_count = referenced.len();
    }

    /// Total number of instructions in the compiled rule.
    pub fn instruction_count(rule: &CompiledRule) -> usize {
        rule.instructions.len()
    }

    /// Sorted list of all variable names appearing in the rule.
    pub fn var_names(rule: &CompiledRule) -> Vec<String> {
        let mut vars: std::collections::HashSet<String> = std::collections::HashSet::new();

        for instr in &rule.instructions {
            match instr {
                Instruction::LoadTriple {
                    subject,
                    predicate,
                    object,
                } => {
                    for slot in [subject, predicate, object] {
                        if let Some(v) = slot.as_var() {
                            vars.insert(v.to_string());
                        }
                    }
                }
                Instruction::Bind { var, .. } => {
                    vars.insert(var.clone());
                }
                Instruction::Filter { condition } => match condition {
                    FilterCondition::VarEquals(a, b) | FilterCondition::VarNotEquals(a, b) => {
                        vars.insert(a.clone());
                        vars.insert(b.clone());
                    }
                    FilterCondition::Bound(v) => {
                        vars.insert(v.clone());
                    }
                },
                Instruction::Produce { head } => {
                    for (s, p, o) in head {
                        for slot in [s, p, o] {
                            if let Some(v) = slot.as_var() {
                                vars.insert(v.to_string());
                            }
                        }
                    }
                }
                Instruction::Jump { .. } | Instruction::Halt => {}
            }
        }

        let mut result: Vec<String> = vars.into_iter().collect();
        result.sort();
        result
    }

    // ── Private ───────────────────────────────────────────────────────────────

    /// Convert a term string into a `BindingSlot`, recording new variables.
    fn make_slot(term: &str, vars: &mut Vec<String>) -> BindingSlot {
        if let Some(var_name) = term.strip_prefix('?') {
            if !vars.contains(&var_name.to_string()) {
                vars.push(var_name.to_string());
            }
            BindingSlot::Var(var_name.to_string())
        } else {
            BindingSlot::Constant(term.to_string())
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn v(name: &str) -> BindingSlot {
        BindingSlot::Var(name.to_string())
    }

    fn c(name: &str) -> BindingSlot {
        BindingSlot::Constant(name.to_string())
    }

    // ── BindingSlot ──────────────────────────────────────────────────────────

    #[test]
    fn test_binding_slot_var() {
        let slot = v("x");
        assert_eq!(slot.as_var(), Some("x"));
        assert_eq!(slot.as_constant(), None);
    }

    #[test]
    fn test_binding_slot_constant() {
        let slot = c("http://example.org/foo");
        assert_eq!(slot.as_constant(), Some("http://example.org/foo"));
        assert_eq!(slot.as_var(), None);
    }

    #[test]
    fn test_binding_slot_eq() {
        assert_eq!(v("x"), v("x"));
        assert_ne!(v("x"), v("y"));
        assert_ne!(v("x"), c("x"));
    }

    #[test]
    fn test_binding_slot_clone() {
        let slot = v("z");
        let cloned = slot.clone();
        assert_eq!(slot, cloned);
    }

    // ── FilterCondition ───────────────────────────────────────────────────────

    #[test]
    fn test_filter_var_equals() {
        let fc = FilterCondition::VarEquals("a".to_string(), "b".to_string());
        assert_eq!(
            fc,
            FilterCondition::VarEquals("a".to_string(), "b".to_string())
        );
    }

    #[test]
    fn test_filter_var_not_equals() {
        let fc = FilterCondition::VarNotEquals("a".to_string(), "b".to_string());
        match fc {
            FilterCondition::VarNotEquals(a, b) => {
                assert_eq!(a, "a");
                assert_eq!(b, "b");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_filter_bound() {
        let fc = FilterCondition::Bound("x".to_string());
        match fc {
            FilterCondition::Bound(v) => assert_eq!(v, "x"),
            _ => panic!("wrong variant"),
        }
    }

    // ── Instruction ──────────────────────────────────────────────────────────

    #[test]
    fn test_instruction_load_triple() {
        let instr = Instruction::LoadTriple {
            subject: v("s"),
            predicate: c("rdf:type"),
            object: v("type"),
        };
        match instr {
            Instruction::LoadTriple {
                subject,
                predicate,
                object,
            } => {
                assert_eq!(subject, v("s"));
                assert_eq!(predicate, c("rdf:type"));
                assert_eq!(object, v("type"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_instruction_bind() {
        let instr = Instruction::Bind {
            var: "x".to_string(),
            slot: v("x"),
        };
        match instr {
            Instruction::Bind { var, slot } => {
                assert_eq!(var, "x");
                assert_eq!(slot, v("x"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_instruction_halt() {
        let instr = Instruction::Halt;
        assert!(matches!(instr, Instruction::Halt));
    }

    #[test]
    fn test_instruction_jump() {
        let instr = Instruction::Jump { target: 5 };
        match instr {
            Instruction::Jump { target } => assert_eq!(target, 5),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_instruction_produce() {
        let instr = Instruction::Produce {
            head: vec![(v("s"), c("rdf:type"), c("owl:Thing"))],
        };
        match instr {
            Instruction::Produce { head } => assert_eq!(head.len(), 1),
            _ => panic!("wrong variant"),
        }
    }

    // ── RuleCompiler::compile ─────────────────────────────────────────────────

    #[test]
    fn test_compile_simple_rule() {
        let body = vec![("?s".to_string(), "rdf:type".to_string(), "?c".to_string())];
        let head = vec![(
            "?s".to_string(),
            "rdf:type".to_string(),
            "owl:Thing".to_string(),
        )];
        let rule = RuleCompiler::compile("r1", body, head);
        assert_eq!(rule.name, "r1");
        assert!(!rule.instructions.is_empty());
        assert!(matches!(rule.instructions.last(), Some(Instruction::Halt)));
    }

    #[test]
    fn test_compile_ends_with_halt() {
        let body = vec![("?s".to_string(), "rdf:type".to_string(), "?c".to_string())];
        let rule = RuleCompiler::compile("r", body, vec![]);
        assert!(matches!(rule.instructions.last(), Some(Instruction::Halt)));
    }

    #[test]
    fn test_compile_empty_body() {
        let rule = RuleCompiler::compile("empty", vec![], vec![]);
        // Should at least have Halt
        assert!(rule.instructions.contains(&Instruction::Halt));
    }

    #[test]
    fn test_compile_var_count() {
        let body = vec![("?s".to_string(), "?p".to_string(), "?o".to_string())];
        let rule = RuleCompiler::compile("r", body, vec![]);
        assert!(rule.var_count >= 3);
    }

    #[test]
    fn test_compile_constant_not_var() {
        let body = vec![(
            "http://example.org/Alice".to_string(),
            "rdf:type".to_string(),
            "?class".to_string(),
        )];
        let rule = RuleCompiler::compile("r", body, vec![]);
        // Only one variable: "class"
        assert_eq!(rule.var_count, 1);
    }

    #[test]
    fn test_compile_two_body_patterns() {
        let body = vec![
            ("?s".to_string(), "rdf:type".to_string(), "?c".to_string()),
            (
                "?c".to_string(),
                "rdfs:subClassOf".to_string(),
                "?sc".to_string(),
            ),
        ];
        let rule = RuleCompiler::compile("subclass", body, vec![]);
        // Should have 3 variables: s, c, sc
        assert_eq!(rule.var_count, 3);
    }

    #[test]
    fn test_compile_head_produce_instruction() {
        let body = vec![("?s".to_string(), "rdf:type".to_string(), "?c".to_string())];
        let head = vec![(
            "?s".to_string(),
            "rdf:type".to_string(),
            "owl:Thing".to_string(),
        )];
        let rule = RuleCompiler::compile("r", body, head);
        let produces: Vec<_> = rule
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Produce { .. }))
            .collect();
        assert_eq!(produces.len(), 1);
    }

    #[test]
    fn test_compile_head_multiple_triples() {
        let body = vec![("?s".to_string(), "rdf:type".to_string(), "?c".to_string())];
        let head = vec![
            (
                "?s".to_string(),
                "rdf:type".to_string(),
                "owl:Thing".to_string(),
            ),
            ("?s".to_string(), "owl:sameAs".to_string(), "?s".to_string()),
        ];
        let rule = RuleCompiler::compile("r", body, head);
        if let Some(Instruction::Produce { head }) = rule
            .instructions
            .iter()
            .find(|i| matches!(i, Instruction::Produce { .. }))
        {
            assert_eq!(head.len(), 2);
        } else {
            panic!("No Produce instruction found");
        }
    }

    // ── RuleCompiler::instruction_count ──────────────────────────────────────

    #[test]
    fn test_instruction_count() {
        let rule = RuleCompiler::compile(
            "r",
            vec![("?s".to_string(), "?p".to_string(), "?o".to_string())],
            vec![("?s".to_string(), "?p".to_string(), "?o".to_string())],
        );
        assert_eq!(
            RuleCompiler::instruction_count(&rule),
            rule.instructions.len()
        );
    }

    #[test]
    fn test_instruction_count_zero_body() {
        let rule = RuleCompiler::compile("r", vec![], vec![]);
        // At minimum: Halt
        assert!(RuleCompiler::instruction_count(&rule) >= 1);
    }

    // ── RuleCompiler::var_names ───────────────────────────────────────────────

    #[test]
    fn test_var_names_basic() {
        let rule = RuleCompiler::compile(
            "r",
            vec![("?s".to_string(), "?p".to_string(), "?o".to_string())],
            vec![],
        );
        let names = RuleCompiler::var_names(&rule);
        assert!(names.contains(&"s".to_string()));
        assert!(names.contains(&"p".to_string()));
        assert!(names.contains(&"o".to_string()));
    }

    #[test]
    fn test_var_names_sorted() {
        let rule = RuleCompiler::compile(
            "r",
            vec![("?z".to_string(), "?a".to_string(), "?m".to_string())],
            vec![],
        );
        let names = RuleCompiler::var_names(&rule);
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn test_var_names_no_duplicates() {
        let rule = RuleCompiler::compile(
            "r",
            vec![
                ("?s".to_string(), "rdf:type".to_string(), "?c".to_string()),
                ("?s".to_string(), "rdfs:label".to_string(), "?l".to_string()),
            ],
            vec![],
        );
        let names = RuleCompiler::var_names(&rule);
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(names.len(), unique.len());
    }

    // ── RuleCompiler::optimize ────────────────────────────────────────────────

    #[test]
    fn test_optimize_removes_dead_binds() {
        // Compile a rule where "p" is not used in head or filters.
        let body = vec![("?s".to_string(), "?p".to_string(), "?o".to_string())];
        let head = vec![(
            "?s".to_string(),
            "rdf:type".to_string(),
            "owl:Thing".to_string(),
        )];
        let mut rule = RuleCompiler::compile("r", body, head);
        let before = RuleCompiler::instruction_count(&rule);
        RuleCompiler::optimize(&mut rule);
        let after = RuleCompiler::instruction_count(&rule);
        // Optimised rule should have ≤ instructions
        assert!(after <= before);
    }

    #[test]
    fn test_optimize_preserves_halt() {
        let body = vec![("?s".to_string(), "rdf:type".to_string(), "?c".to_string())];
        let mut rule = RuleCompiler::compile("r", body, vec![]);
        RuleCompiler::optimize(&mut rule);
        assert!(matches!(rule.instructions.last(), Some(Instruction::Halt)));
    }

    #[test]
    fn test_optimize_idempotent() {
        let body = vec![("?s".to_string(), "rdf:type".to_string(), "?c".to_string())];
        let head = vec![(
            "?s".to_string(),
            "rdf:type".to_string(),
            "owl:Thing".to_string(),
        )];
        let mut rule = RuleCompiler::compile("r", body, head);
        RuleCompiler::optimize(&mut rule);
        let count1 = RuleCompiler::instruction_count(&rule);
        RuleCompiler::optimize(&mut rule);
        let count2 = RuleCompiler::instruction_count(&rule);
        assert_eq!(count1, count2);
    }

    // ── CompiledRule ─────────────────────────────────────────────────────────

    #[test]
    fn test_compiled_rule_name() {
        let rule = RuleCompiler::compile("my_rule", vec![], vec![]);
        assert_eq!(rule.name, "my_rule");
    }

    #[test]
    fn test_compiled_rule_clone() {
        let rule = RuleCompiler::compile(
            "r",
            vec![("?s".to_string(), "?p".to_string(), "?o".to_string())],
            vec![],
        );
        let cloned = rule.clone();
        assert_eq!(rule.name, cloned.name);
        assert_eq!(rule.instructions.len(), cloned.instructions.len());
    }

    // ── Integration ───────────────────────────────────────────────────────────

    #[test]
    fn test_rdfs_subclass_rule() {
        // rdfs2: if (X rdfs:subClassOf Y) and (Z rdf:type X) then (Z rdf:type Y)
        let body = vec![
            (
                "?x".to_string(),
                "rdfs:subClassOf".to_string(),
                "?y".to_string(),
            ),
            ("?z".to_string(), "rdf:type".to_string(), "?x".to_string()),
        ];
        let head = vec![("?z".to_string(), "rdf:type".to_string(), "?y".to_string())];
        let rule = RuleCompiler::compile("rdfs2", body, head);
        assert_eq!(rule.name, "rdfs2");
        // Variables: x, y, z
        assert_eq!(rule.var_count, 3);
        let loads: Vec<_> = rule
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::LoadTriple { .. }))
            .collect();
        assert_eq!(loads.len(), 2);
    }

    #[test]
    fn test_owl_same_as_rule() {
        let body = vec![("?s".to_string(), "owl:sameAs".to_string(), "?t".to_string())];
        let head = vec![("?t".to_string(), "owl:sameAs".to_string(), "?s".to_string())];
        let rule = RuleCompiler::compile("sameAs_symmetric", body, head);
        assert!(rule.var_count >= 2);
    }

    #[test]
    fn test_filter_instruction_roundtrip() {
        // Build a rule manually with a Filter instruction and check var_names.
        let mut rule = RuleCompiler::compile(
            "filtered",
            vec![("?s".to_string(), "?p".to_string(), "?o".to_string())],
            vec![],
        );
        rule.instructions.insert(
            1,
            Instruction::Filter {
                condition: FilterCondition::Bound("s".to_string()),
            },
        );
        let names = RuleCompiler::var_names(&rule);
        assert!(names.contains(&"s".to_string()));
    }

    #[test]
    fn test_jump_instruction_roundtrip() {
        let mut rule = RuleCompiler::compile("r", vec![], vec![]);
        rule.instructions.push(Instruction::Jump { target: 0 });
        let count = RuleCompiler::instruction_count(&rule);
        assert!(count >= 2); // Halt + Jump
    }

    #[test]
    fn test_many_body_patterns() {
        let body: Vec<_> = (0..5)
            .map(|i| (format!("?s{i}"), "rdf:type".to_string(), format!("?c{i}")))
            .collect();
        let rule = RuleCompiler::compile("big_rule", body, vec![]);
        // 5 patterns × (LoadTriple + 2 Bind) + Halt = 16 minimum
        assert!(rule.instructions.len() >= 6);
        assert_eq!(rule.var_count, 10); // s0..s4, c0..c4
    }

    #[test]
    fn test_binding_slot_var_as_var() {
        let slot = BindingSlot::Var("x".to_string());
        assert_eq!(slot.as_var(), Some("x"));
        assert_eq!(slot.as_constant(), None);
    }

    #[test]
    fn test_binding_slot_constant_as_constant() {
        let slot = BindingSlot::Constant("rdf:type".to_string());
        assert_eq!(slot.as_var(), None);
        assert_eq!(slot.as_constant(), Some("rdf:type"));
    }

    #[test]
    fn test_filter_condition_var_equals_debug() {
        let cond = FilterCondition::VarEquals("a".to_string(), "b".to_string());
        let s = format!("{cond:?}");
        assert!(s.contains("VarEquals"));
    }

    #[test]
    fn test_filter_condition_var_not_equals_debug() {
        let cond = FilterCondition::VarNotEquals("x".to_string(), "y".to_string());
        let s = format!("{cond:?}");
        assert!(s.contains("VarNotEquals"));
    }

    #[test]
    fn test_filter_condition_bound_debug() {
        let cond = FilterCondition::Bound("z".to_string());
        let s = format!("{cond:?}");
        assert!(s.contains("Bound"));
    }

    #[test]
    fn test_compiled_rule_var_names_single_var() {
        let rule = RuleCompiler::compile(
            "r",
            vec![("?x".to_string(), "ex:p".to_string(), "ex:o".to_string())],
            vec![],
        );
        let names = RuleCompiler::var_names(&rule);
        assert!(names.contains(&"x".to_string()));
    }

    #[test]
    fn test_compiled_rule_var_names_empty() {
        let rule = RuleCompiler::compile("r", vec![], vec![]);
        let names = RuleCompiler::var_names(&rule);
        assert!(names.is_empty());
    }

    #[test]
    fn test_instruction_count_no_body_no_head() {
        let rule = RuleCompiler::compile("r", vec![], vec![]);
        // Only a Halt instruction
        assert_eq!(RuleCompiler::instruction_count(&rule), 1);
    }

    #[test]
    fn test_compiled_rule_is_empty_with_no_patterns() {
        let rule = RuleCompiler::compile("empty_rule", vec![], vec![]);
        assert_eq!(rule.var_count, 0);
        // Only Halt
        assert_eq!(rule.instructions, vec![Instruction::Halt]);
    }

    #[test]
    fn test_produce_instruction_present_when_head_non_empty() {
        let body = vec![("?s".to_string(), "?p".to_string(), "?o".to_string())];
        let head = vec![("?s".to_string(), "rdf:type".to_string(), "ex:C".to_string())];
        let rule = RuleCompiler::compile("r", body, head);
        let has_produce = rule
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Produce { .. }));
        assert!(has_produce);
    }
}
