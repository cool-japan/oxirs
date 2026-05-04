//! Integration tests for JIT compilation phases b and c — Cranelift filter, join-key,
//! and ORDER BY codegen.
//!
//! All tests require the `jit` feature:
//!   cargo nextest run -p oxirs-arq --features jit --test jit_test

#![cfg(feature = "jit")]

use std::collections::HashMap;

use oxirs_arq::jit::{
    // Phase b
    BinOp,
    BuiltinFunc,
    // Phase c — join
    CompiledJoinKey,
    // Phase c — order
    CompiledOrder,
    FilterCompiler,
    FilterExpr,
    JitFilterCache,
    JoinCompiler,
    JoinKeySpec,
    OrderCompiler,
    OrderKeySpec,
    VarIndexMap,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compiler() -> FilterCompiler {
    FilterCompiler::new()
}

fn var_map_single(name: &str) -> VarIndexMap {
    let mut vm = VarIndexMap::new();
    vm.insert(name.to_string(), 0);
    vm
}

fn var_map_two(a: &str, b: &str) -> VarIndexMap {
    let mut vm = VarIndexMap::new();
    vm.insert(a.to_string(), 0);
    vm.insert(b.to_string(), 1);
    vm
}

fn binding1(name: &str, v: f64) -> HashMap<String, f64> {
    let mut b = HashMap::new();
    b.insert(name.to_string(), v);
    b
}

fn binding2(a: &str, va: f64, b: &str, vb: f64) -> HashMap<String, f64> {
    let mut m = HashMap::new();
    m.insert(a.to_string(), va);
    m.insert(b.to_string(), vb);
    m
}

fn compile_and_eval(
    expr: FilterExpr,
    vm: VarIndexMap,
    binding: &HashMap<String, f64>,
) -> Option<bool> {
    let cf = compiler()
        .compile(&expr, vm)
        .expect("compile should not fail")
        .expect("expression should be in supported subset");
    cf.evaluate(binding)
}

// ---------------------------------------------------------------------------
// Literal-only tests
// ---------------------------------------------------------------------------

#[test]
fn test_literal_true() {
    // 1.0 > 0.5 → true
    let expr = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(FilterExpr::Literal(1.0)),
        right: Box::new(FilterExpr::Literal(0.5)),
    };
    let binding: HashMap<String, f64> = HashMap::new();
    assert_eq!(
        compile_and_eval(expr, VarIndexMap::new(), &binding),
        Some(true)
    );
}

#[test]
fn test_literal_false() {
    // 1.0 > 2.0 → false
    let expr = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(FilterExpr::Literal(1.0)),
        right: Box::new(FilterExpr::Literal(2.0)),
    };
    let binding: HashMap<String, f64> = HashMap::new();
    assert_eq!(
        compile_and_eval(expr, VarIndexMap::new(), &binding),
        Some(false)
    );
}

// ---------------------------------------------------------------------------
// Variable + literal comparisons
// ---------------------------------------------------------------------------

#[test]
fn test_variable_gt_literal() {
    // ?x > 3.0, x = 5.0 → true
    let expr = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(3.0)),
    };
    let vm = var_map_single("x");
    assert_eq!(compile_and_eval(expr, vm, &binding1("x", 5.0)), Some(true));
}

#[test]
fn test_variable_lt_literal_false() {
    // ?x > 3.0, x = 1.0 → false
    let expr = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(3.0)),
    };
    let vm = var_map_single("x");
    assert_eq!(compile_and_eval(expr, vm, &binding1("x", 1.0)), Some(false));
}

// ---------------------------------------------------------------------------
// Logical combinators
// ---------------------------------------------------------------------------

#[test]
fn test_logical_and_both_true() {
    // (?x > 0) && (?x < 10), x = 5 → true
    let gt_zero = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(0.0)),
    };
    let lt_ten = FilterExpr::BinOp {
        op: BinOp::Lt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(10.0)),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::And,
        left: Box::new(gt_zero),
        right: Box::new(lt_ten),
    };
    let vm = var_map_single("x");
    assert_eq!(compile_and_eval(expr, vm, &binding1("x", 5.0)), Some(true));
}

#[test]
fn test_logical_and_one_false() {
    // (?x > 0) && (?x < 10), x = 15 → false
    let gt_zero = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(0.0)),
    };
    let lt_ten = FilterExpr::BinOp {
        op: BinOp::Lt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(10.0)),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::And,
        left: Box::new(gt_zero),
        right: Box::new(lt_ten),
    };
    let vm = var_map_single("x");
    assert_eq!(
        compile_and_eval(expr, vm, &binding1("x", 15.0)),
        Some(false)
    );
}

#[test]
fn test_logical_or_first_true() {
    // (?x < 0) || (?x > 3), x = 5 → true
    let lt_zero = FilterExpr::BinOp {
        op: BinOp::Lt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(0.0)),
    };
    let gt_three = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(3.0)),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::Or,
        left: Box::new(lt_zero),
        right: Box::new(gt_three),
    };
    let vm = var_map_single("x");
    assert_eq!(compile_and_eval(expr, vm, &binding1("x", 5.0)), Some(true));
}

#[test]
fn test_logical_not() {
    // !(?x > 5), x = 3 → true
    let gt_five = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(5.0)),
    };
    let expr = FilterExpr::UnaryNot(Box::new(gt_five));
    let vm = var_map_single("x");
    assert_eq!(compile_and_eval(expr, vm, &binding1("x", 3.0)), Some(true));
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

#[test]
fn test_arithmetic_add() {
    // (?x + ?y) > 10.0, x = 7, y = 4 → true
    let add = FilterExpr::BinOp {
        op: BinOp::Add,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Variable("y".to_string())),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(add),
        right: Box::new(FilterExpr::Literal(10.0)),
    };
    let vm = var_map_two("x", "y");
    assert_eq!(
        compile_and_eval(expr, vm, &binding2("x", 7.0, "y", 4.0)),
        Some(true)
    );
}

#[test]
fn test_arithmetic_sub_false() {
    // (?x - ?y) > 10.0, x = 7, y = 4 → false (7-4=3 < 10)
    let sub = FilterExpr::BinOp {
        op: BinOp::Sub,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Variable("y".to_string())),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(sub),
        right: Box::new(FilterExpr::Literal(10.0)),
    };
    let vm = var_map_two("x", "y");
    assert_eq!(
        compile_and_eval(expr, vm, &binding2("x", 7.0, "y", 4.0)),
        Some(false)
    );
}

// ---------------------------------------------------------------------------
// Built-in functions
// ---------------------------------------------------------------------------

#[test]
fn test_abs() {
    // ABS(?x) > 3.0, x = -5.0 → true
    let abs_x = FilterExpr::Builtin {
        func: BuiltinFunc::Abs,
        arg: Box::new(FilterExpr::Variable("x".to_string())),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(abs_x),
        right: Box::new(FilterExpr::Literal(3.0)),
    };
    let vm = var_map_single("x");
    assert_eq!(compile_and_eval(expr, vm, &binding1("x", -5.0)), Some(true));
}

#[test]
fn test_ceil() {
    // CEIL(?x) = 5.0, x = 4.1 → true
    let ceil_x = FilterExpr::Builtin {
        func: BuiltinFunc::Ceil,
        arg: Box::new(FilterExpr::Variable("x".to_string())),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::Eq,
        left: Box::new(ceil_x),
        right: Box::new(FilterExpr::Literal(5.0)),
    };
    let vm = var_map_single("x");
    assert_eq!(compile_and_eval(expr, vm, &binding1("x", 4.1)), Some(true));
}

#[test]
fn test_floor() {
    // FLOOR(?x) = 4.0, x = 4.9 → true
    let floor_x = FilterExpr::Builtin {
        func: BuiltinFunc::Floor,
        arg: Box::new(FilterExpr::Variable("x".to_string())),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::Eq,
        left: Box::new(floor_x),
        right: Box::new(FilterExpr::Literal(4.0)),
    };
    let vm = var_map_single("x");
    assert_eq!(compile_and_eval(expr, vm, &binding1("x", 4.9)), Some(true));
}

#[test]
fn test_round() {
    // ROUND(?x) = 5.0, x = 4.6 → true
    let round_x = FilterExpr::Builtin {
        func: BuiltinFunc::Round,
        arg: Box::new(FilterExpr::Variable("x".to_string())),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::Eq,
        left: Box::new(round_x),
        right: Box::new(FilterExpr::Literal(5.0)),
    };
    let vm = var_map_single("x");
    assert_eq!(compile_and_eval(expr, vm, &binding1("x", 4.6)), Some(true));
}

#[test]
fn test_round_half_up_positive() {
    // SPARQL ROUND(0.5) = 1.0 (round-half-away-from-zero, not IEEE 754 nearest-even).
    // IEEE 754 `nearest` gives 0.0 for this input; SPARQL requires 1.0.
    let round_x = FilterExpr::Builtin {
        func: BuiltinFunc::Round,
        arg: Box::new(FilterExpr::Variable("x".to_string())),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::Eq,
        left: Box::new(round_x),
        right: Box::new(FilterExpr::Literal(1.0)),
    };
    let vm = var_map_single("x");
    assert_eq!(compile_and_eval(expr, vm, &binding1("x", 0.5)), Some(true));
}

#[test]
fn test_round_half_toward_positive_infinity_negative() {
    // XPath fn:round: "the one that is closest to positive infinity is returned".
    // ROUND(-2.5) = -2.0 (not -3.0), because -2 is closer to +∞ than -3.
    // ROUND(-0.5) = 0.0  (not -1.0), because  0 is closer to +∞ than -1.
    // Both are captured by the formula floor(x + 0.5).
    let round_x = FilterExpr::Builtin {
        func: BuiltinFunc::Round,
        arg: Box::new(FilterExpr::Variable("x".to_string())),
    };

    // -2.5 → -2.0
    let expr_neg25 = FilterExpr::BinOp {
        op: BinOp::Eq,
        left: Box::new(round_x.clone()),
        right: Box::new(FilterExpr::Literal(-2.0)),
    };
    let vm = var_map_single("x");
    assert_eq!(
        compile_and_eval(expr_neg25, vm.clone(), &binding1("x", -2.5)),
        Some(true)
    );

    // -0.5 → 0.0
    let expr_neg05 = FilterExpr::BinOp {
        op: BinOp::Eq,
        left: Box::new(round_x),
        right: Box::new(FilterExpr::Literal(0.0)),
    };
    assert_eq!(
        compile_and_eval(expr_neg05, vm, &binding1("x", -0.5)),
        Some(true)
    );
}

// ---------------------------------------------------------------------------
// Fall-back / error conditions
// ---------------------------------------------------------------------------

#[test]
fn test_missing_variable_returns_none() {
    // Compile OK, but binding is missing "x" → evaluate returns None
    let expr = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(3.0)),
    };
    let vm = var_map_single("x");
    let cf = compiler()
        .compile(&expr, vm)
        .expect("compile ok")
        .expect("in supported subset");
    let empty: HashMap<String, f64> = HashMap::new();
    assert_eq!(cf.evaluate(&empty), None);
}

#[test]
fn test_compiled_filter_evaluate_correct_result() {
    // Comprehensive check: ?x >= 0.0 && ?x <= 100.0
    let ge_zero = FilterExpr::BinOp {
        op: BinOp::Ge,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(0.0)),
    };
    let le_hundred = FilterExpr::BinOp {
        op: BinOp::Le,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(100.0)),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::And,
        left: Box::new(ge_zero),
        right: Box::new(le_hundred),
    };
    let vm = var_map_single("x");
    let cf = compiler()
        .compile(&expr, vm)
        .expect("compile ok")
        .expect("in supported subset");

    assert_eq!(cf.evaluate(&binding1("x", 50.0)), Some(true));
    assert_eq!(cf.evaluate(&binding1("x", -1.0)), Some(false));
    assert_eq!(cf.evaluate(&binding1("x", 101.0)), Some(false));
    assert_eq!(cf.evaluate(&binding1("x", 0.0)), Some(true));
    assert_eq!(cf.evaluate(&binding1("x", 100.0)), Some(true));
}

// ---------------------------------------------------------------------------
// JitFilterCache integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_jit_cache_get_before_insert_miss() {
    let cache = JitFilterCache::new(64).expect("cache init");
    assert!(cache.get(9999).is_none());
}

#[test]
fn test_jit_cache_insert_then_get_hit() {
    let cache = JitFilterCache::new(64).expect("cache init");
    let mut vm = VarIndexMap::new();
    vm.insert("x".to_string(), 0);
    let expr = FilterExpr::BinOp {
        op: BinOp::Lt,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(100.0)),
    };
    let compiled = cache
        .compile_and_insert(42, &expr, vm)
        .expect("compile ok")
        .expect("in supported subset");
    assert_eq!(cache.len(), 1);

    let hit = cache.get(42).expect("should be in cache");
    assert!(std::sync::Arc::ptr_eq(&compiled, &hit));

    // Correctness of the cached entry
    assert_eq!(hit.evaluate(&binding1("x", 50.0)), Some(true));
    assert_eq!(hit.evaluate(&binding1("x", 150.0)), Some(false));
}

#[test]
fn test_jit_cache_stats() {
    let cache = JitFilterCache::new(64).expect("cache init");
    let mut vm = VarIndexMap::new();
    vm.insert("v".to_string(), 0);
    let expr = FilterExpr::BinOp {
        op: BinOp::Ge,
        left: Box::new(FilterExpr::Variable("v".to_string())),
        right: Box::new(FilterExpr::Literal(0.0)),
    };
    cache.compile_and_insert(1, &expr, vm).expect("ok");
    let _ = cache.get(1);
    let stats = cache.stats();
    assert_eq!(stats.compile_count, 1);
    assert!(stats.hit_count >= 1);
    assert_eq!(stats.len, 1);
}

// ---------------------------------------------------------------------------
// Equality / inequality
// ---------------------------------------------------------------------------

#[test]
fn test_eq_comparison() {
    // ?x = 5.0, x = 5.0 → true
    let expr = FilterExpr::BinOp {
        op: BinOp::Eq,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(5.0)),
    };
    let vm = var_map_single("x");
    assert_eq!(compile_and_eval(expr, vm, &binding1("x", 5.0)), Some(true));
}

#[test]
fn test_ne_comparison() {
    // ?x != 5.0, x = 3.0 → true
    let expr = FilterExpr::BinOp {
        op: BinOp::Ne,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Literal(5.0)),
    };
    let vm = var_map_single("x");
    assert_eq!(compile_and_eval(expr, vm, &binding1("x", 3.0)), Some(true));
}

#[test]
fn test_multiply_and_compare() {
    // ?x * ?y >= 20.0, x = 4, y = 5 → true (4*5=20)
    let mul = FilterExpr::BinOp {
        op: BinOp::Mul,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Variable("y".to_string())),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::Ge,
        left: Box::new(mul),
        right: Box::new(FilterExpr::Literal(20.0)),
    };
    let vm = var_map_two("x", "y");
    assert_eq!(
        compile_and_eval(expr, vm, &binding2("x", 4.0, "y", 5.0)),
        Some(true)
    );
}

#[test]
fn test_divide_and_compare() {
    // ?x / ?y > 2.0, x = 10, y = 4 → true (10/4=2.5)
    let div = FilterExpr::BinOp {
        op: BinOp::Div,
        left: Box::new(FilterExpr::Variable("x".to_string())),
        right: Box::new(FilterExpr::Variable("y".to_string())),
    };
    let expr = FilterExpr::BinOp {
        op: BinOp::Gt,
        left: Box::new(div),
        right: Box::new(FilterExpr::Literal(2.0)),
    };
    let vm = var_map_two("x", "y");
    assert_eq!(
        compile_and_eval(expr, vm, &binding2("x", 10.0, "y", 4.0)),
        Some(true)
    );
}

// ===========================================================================
// Phase c — JoinCompiler tests
// ===========================================================================

fn join_compiler() -> JoinCompiler {
    JoinCompiler::new()
}

fn order_compiler() -> OrderCompiler {
    OrderCompiler::new()
}

fn compile_join(specs: &[JoinKeySpec]) -> CompiledJoinKey {
    join_compiler()
        .compile(specs)
        .expect("JoinCompiler::compile should not fail")
}

fn compile_order(specs: &[OrderKeySpec]) -> CompiledOrder {
    order_compiler()
        .compile(specs)
        .expect("OrderCompiler::compile should not fail")
}

// ---------------------------------------------------------------------------
// Single-key join tests
// ---------------------------------------------------------------------------

#[test]
fn test_join_single_key_match() {
    // epsilon mode: 1.0 vs 1.0 → match
    let spec = JoinKeySpec {
        left_idx: 0,
        right_idx: 0,
        numeric_epsilon: true,
    };
    let cj = compile_join(&[spec]);
    assert!(cj.compare(&[1.0], &[1.0]));
}

#[test]
fn test_join_single_key_no_match() {
    // epsilon mode: 1.0 vs 2.0 → no match
    let spec = JoinKeySpec {
        left_idx: 0,
        right_idx: 0,
        numeric_epsilon: true,
    };
    let cj = compile_join(&[spec]);
    assert!(!cj.compare(&[1.0], &[2.0]));
}

// ---------------------------------------------------------------------------
// Multi-key join tests
// ---------------------------------------------------------------------------

#[test]
fn test_join_multi_key_all_match() {
    // Two epsilon keys; left=[3.0, 7.0], right=[3.0, 7.0] → all match
    let specs = vec![
        JoinKeySpec {
            left_idx: 0,
            right_idx: 0,
            numeric_epsilon: true,
        },
        JoinKeySpec {
            left_idx: 1,
            right_idx: 1,
            numeric_epsilon: true,
        },
    ];
    let cj = compile_join(&specs);
    assert!(cj.compare(&[3.0, 7.0], &[3.0, 7.0]));
}

#[test]
fn test_join_multi_key_first_matches_second_doesnt() {
    // First key matches, second doesn't → overall no match
    let specs = vec![
        JoinKeySpec {
            left_idx: 0,
            right_idx: 0,
            numeric_epsilon: true,
        },
        JoinKeySpec {
            left_idx: 1,
            right_idx: 1,
            numeric_epsilon: true,
        },
    ];
    let cj = compile_join(&specs);
    assert!(!cj.compare(&[3.0, 7.0], &[3.0, 8.0]));
}

// ---------------------------------------------------------------------------
// Epsilon comparison tests
// ---------------------------------------------------------------------------

#[test]
fn test_join_epsilon_comparison_close_values() {
    // Values within 1e-9 → match
    let spec = JoinKeySpec {
        left_idx: 0,
        right_idx: 0,
        numeric_epsilon: true,
    };
    let cj = compile_join(&[spec]);
    let left = 1.0_f64;
    let right = left + 5e-10; // well within 1e-9
    assert!(cj.compare(&[left], &[right]));
}

#[test]
fn test_join_epsilon_comparison_far_values() {
    // Values differing by 1e-6 → no match
    let spec = JoinKeySpec {
        left_idx: 0,
        right_idx: 0,
        numeric_epsilon: true,
    };
    let cj = compile_join(&[spec]);
    let left = 1.0_f64;
    let right = left + 1e-6;
    assert!(!cj.compare(&[left], &[right]));
}

// ---------------------------------------------------------------------------
// Exact (bitcast) comparison — NaN semantics
// ---------------------------------------------------------------------------

#[test]
fn test_join_exact_nan_same_bits_equal() {
    // Exact mode: bitcast both NaN values to i64 and use icmp Equal.
    // A NaN value with itself has identical bit patterns → icmp Equal returns 1 (equal).
    // This is the documented bitcast semantics; see join_compiler.rs § "Exact bit equality".
    let spec = JoinKeySpec {
        left_idx: 0,
        right_idx: 0,
        numeric_epsilon: false, // exact / bitcast mode
    };
    let cj = compile_join(&[spec]);
    let nan = f64::NAN;
    // Both sides hold the *same* NaN constant, so bits are identical → compare as equal.
    assert!(
        cj.compare(&[nan], &[nan]),
        "bitcast NaN with identical bits should compare as equal in exact mode"
    );
}

// ---------------------------------------------------------------------------
// key_count
// ---------------------------------------------------------------------------

#[test]
fn test_join_key_count() {
    let specs = vec![
        JoinKeySpec {
            left_idx: 0,
            right_idx: 0,
            numeric_epsilon: true,
        },
        JoinKeySpec {
            left_idx: 1,
            right_idx: 1,
            numeric_epsilon: false,
        },
        JoinKeySpec {
            left_idx: 2,
            right_idx: 2,
            numeric_epsilon: true,
        },
    ];
    let cj = compile_join(&specs);
    assert_eq!(cj.key_count(), 3);
}

// ===========================================================================
// Phase c — OrderCompiler tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Single-column ascending
// ---------------------------------------------------------------------------

#[test]
fn test_order_asc_less_than() {
    // 1.0 < 2.0 ascending → Less
    let spec = OrderKeySpec {
        col_idx: 0,
        ascending: true,
    };
    let co = compile_order(&[spec]);
    assert_eq!(co.compare(&[1.0], &[2.0]), std::cmp::Ordering::Less);
}

#[test]
fn test_order_asc_greater_than() {
    // 3.0 > 2.0 ascending → Greater
    let spec = OrderKeySpec {
        col_idx: 0,
        ascending: true,
    };
    let co = compile_order(&[spec]);
    assert_eq!(co.compare(&[3.0], &[2.0]), std::cmp::Ordering::Greater);
}

#[test]
fn test_order_asc_equal() {
    // 5.0 == 5.0 ascending → Equal
    let spec = OrderKeySpec {
        col_idx: 0,
        ascending: true,
    };
    let co = compile_order(&[spec]);
    assert_eq!(co.compare(&[5.0], &[5.0]), std::cmp::Ordering::Equal);
}

// ---------------------------------------------------------------------------
// Single-column descending
// ---------------------------------------------------------------------------

#[test]
fn test_order_desc_less_than_returns_greater() {
    // 1.0 < 2.0 in descending order → in the sorted order 1.0 comes *after* 2.0 → Greater
    let spec = OrderKeySpec {
        col_idx: 0,
        ascending: false,
    };
    let co = compile_order(&[spec]);
    assert_eq!(co.compare(&[1.0], &[2.0]), std::cmp::Ordering::Greater);
}

// ---------------------------------------------------------------------------
// Multi-column tests
// ---------------------------------------------------------------------------

#[test]
fn test_order_multi_key_first_equal_second_decides() {
    // col0: equal; col1 ascending: 1.0 < 2.0 → Less
    let specs = vec![
        OrderKeySpec {
            col_idx: 0,
            ascending: true,
        },
        OrderKeySpec {
            col_idx: 1,
            ascending: true,
        },
    ];
    let co = compile_order(&specs);
    assert_eq!(
        co.compare(&[5.0, 1.0], &[5.0, 2.0]),
        std::cmp::Ordering::Less
    );
}

#[test]
fn test_order_multi_key_first_decides() {
    // col0 ascending: 1.0 < 3.0 → Less regardless of col1
    let specs = vec![
        OrderKeySpec {
            col_idx: 0,
            ascending: true,
        },
        OrderKeySpec {
            col_idx: 1,
            ascending: true,
        },
    ];
    let co = compile_order(&specs);
    assert_eq!(
        co.compare(&[1.0, 9.0], &[3.0, 1.0]),
        std::cmp::Ordering::Less
    );
}

// ---------------------------------------------------------------------------
// compare() returns correct Ordering enum variant
// ---------------------------------------------------------------------------

#[test]
fn test_order_compare_returns_correct_ordering_enum() {
    let spec = OrderKeySpec {
        col_idx: 0,
        ascending: true,
    };
    let co = compile_order(&[spec]);

    assert_eq!(co.compare(&[0.0], &[1.0]), std::cmp::Ordering::Less);
    assert_eq!(co.compare(&[1.0], &[1.0]), std::cmp::Ordering::Equal);
    assert_eq!(co.compare(&[2.0], &[1.0]), std::cmp::Ordering::Greater);
}

// ---------------------------------------------------------------------------
// col_count
// ---------------------------------------------------------------------------

#[test]
fn test_order_col_count() {
    let specs = vec![
        OrderKeySpec {
            col_idx: 0,
            ascending: true,
        },
        OrderKeySpec {
            col_idx: 1,
            ascending: false,
        },
    ];
    let co = compile_order(&specs);
    assert_eq!(co.col_count(), 2);
}

// ===========================================================================
// Phase d — ProjectCompiler / DistinctCompiler / HavingCompiler tests
// ===========================================================================

mod phase_d {
    use oxirs_arq::jit::{
        AggVarMap, BinOp, DistinctCompiler, DistinctKeySpec, FilterExpr, HavingCompiler,
        ProjectCompiler, ProjectSpec,
    };
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // ProjectCompiler tests
    // -----------------------------------------------------------------------

    fn project_compiler() -> ProjectCompiler {
        ProjectCompiler::new()
    }

    #[test]
    fn test_project_identity() {
        // project[0, 1, 2] from [1.0, 2.0, 3.0] → [1.0, 2.0, 3.0]
        let specs = vec![
            ProjectSpec { src_idx: 0 },
            ProjectSpec { src_idx: 1 },
            ProjectSpec { src_idx: 2 },
        ];
        let cp = project_compiler().compile(&specs).expect("compile ok");
        assert_eq!(cp.output_width(), 3);
        let src = [1.0f64, 2.0, 3.0];
        let mut dst = Vec::new();
        assert!(cp.extract(&src, &mut dst));
        assert_eq!(dst, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_project_reorder() {
        // project[2, 0, 1] from [1.0, 2.0, 3.0] → [3.0, 1.0, 2.0]
        let specs = vec![
            ProjectSpec { src_idx: 2 },
            ProjectSpec { src_idx: 0 },
            ProjectSpec { src_idx: 1 },
        ];
        let cp = project_compiler().compile(&specs).expect("compile ok");
        let src = [1.0f64, 2.0, 3.0];
        let mut dst = Vec::new();
        assert!(cp.extract(&src, &mut dst));
        assert_eq!(dst, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_project_subset() {
        // project[1] from [1.0, 2.0, 3.0] → [2.0]
        let specs = vec![ProjectSpec { src_idx: 1 }];
        let cp = project_compiler().compile(&specs).expect("compile ok");
        assert_eq!(cp.output_width(), 1);
        let src = [1.0f64, 2.0, 3.0];
        let mut dst = Vec::new();
        assert!(cp.extract(&src, &mut dst));
        assert_eq!(dst, vec![2.0]);
    }

    #[test]
    fn test_project_empty_specs() {
        // project[] → [] (no-op, returns true)
        let specs: Vec<ProjectSpec> = vec![];
        let cp = project_compiler().compile(&specs).expect("compile ok");
        assert_eq!(cp.output_width(), 0);
        let src = [1.0f64, 2.0, 3.0];
        let mut dst = Vec::new();
        assert!(cp.extract(&src, &mut dst));
        assert!(dst.is_empty());
    }

    #[test]
    fn test_project_src_bounds_check() {
        // project[5] from [1.0, 2.0] → out-of-bounds → returns false
        let specs = vec![ProjectSpec { src_idx: 5 }];
        let cp = project_compiler().compile(&specs).expect("compile ok");
        let src = [1.0f64, 2.0]; // only 2 elements; idx 5 is out of bounds
        let mut dst = Vec::new();
        assert!(!cp.extract(&src, &mut dst));
    }

    #[test]
    fn test_project_dst_resize() {
        // dst is automatically resized to output_width before calling the JIT function
        let specs = vec![ProjectSpec { src_idx: 0 }, ProjectSpec { src_idx: 2 }];
        let cp = project_compiler().compile(&specs).expect("compile ok");
        let src = [10.0f64, 20.0, 30.0];
        let mut dst: Vec<f64> = vec![999.0, 999.0, 999.0, 999.0]; // oversized
        assert!(cp.extract(&src, &mut dst));
        // After extract(), dst should be exactly output_width() = 2 elements
        assert_eq!(dst.len(), 2);
        assert_eq!(dst[0], 10.0);
        assert_eq!(dst[1], 30.0);
    }

    // -----------------------------------------------------------------------
    // DistinctCompiler tests
    // -----------------------------------------------------------------------

    fn distinct_compiler() -> DistinctCompiler {
        DistinctCompiler::new()
    }

    #[test]
    fn test_distinct_same_row_same_hash() {
        // hash([1.0, 2.0]) called twice → same value
        let specs = vec![
            DistinctKeySpec { col_idx: 0 },
            DistinctKeySpec { col_idx: 1 },
        ];
        let cd = distinct_compiler().compile(&specs).expect("compile ok");
        let row = [1.0f64, 2.0];
        let h1 = cd.hash_key(&row).expect("ok");
        let h2 = cd.hash_key(&row).expect("ok");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_distinct_different_rows_different_hash() {
        // hash([1.0]) ≠ hash([2.0]) with overwhelming probability
        let specs = vec![DistinctKeySpec { col_idx: 0 }];
        let cd = distinct_compiler().compile(&specs).expect("compile ok");
        let h1 = cd.hash_key(&[1.0f64]).expect("ok");
        let h2 = cd.hash_key(&[2.0f64]).expect("ok");
        assert_ne!(h1, h2, "1.0 and 2.0 should produce different FNV-1a hashes");
    }

    #[test]
    fn test_distinct_col_subset() {
        // hash over col[0] only ≠ hash over col[0, 1] on [1.0, 2.0] (different column sets)
        let row = [1.0f64, 2.0];
        let specs1 = vec![DistinctKeySpec { col_idx: 0 }];
        let specs2 = vec![
            DistinctKeySpec { col_idx: 0 },
            DistinctKeySpec { col_idx: 1 },
        ];
        let cd1 = distinct_compiler().compile(&specs1).expect("compile ok");
        let cd2 = distinct_compiler().compile(&specs2).expect("compile ok");
        let h1 = cd1.hash_key(&row).expect("ok");
        let h2 = cd2.hash_key(&row).expect("ok");
        assert_ne!(
            h1, h2,
            "hash of col[0] alone should differ from hash of col[0, 1]"
        );
    }

    #[test]
    fn test_distinct_empty_specs() {
        // No columns → Some(FNV_OFFSET_BASIS) constant regardless of row content
        let specs: Vec<DistinctKeySpec> = vec![];
        let cd = distinct_compiler().compile(&specs).expect("compile ok");
        let fnv_basis: i64 = -3_750_763_034_362_895_579_i64; // 0xcbf29ce484222325 as i64
        assert_eq!(cd.hash_key(&[]).expect("ok"), fnv_basis);
        assert_eq!(cd.hash_key(&[1.0, 2.0, 3.0]).expect("ok"), fnv_basis);
    }

    #[test]
    fn test_distinct_bounds_check() {
        // col_idx=5 on row of len 2 → None (out-of-bounds)
        let specs = vec![DistinctKeySpec { col_idx: 5 }];
        let cd = distinct_compiler().compile(&specs).expect("compile ok");
        let row = [1.0f64, 2.0];
        assert!(cd.hash_key(&row).is_none());
    }

    #[test]
    fn test_distinct_dedup_set() {
        // Three rows; rows 0 and 2 are equal on col[0]; row 1 differs → 2 distinct hashes
        let specs = vec![DistinctKeySpec { col_idx: 0 }];
        let cd = distinct_compiler().compile(&specs).expect("compile ok");
        let row_a = [42.0f64, 100.0]; // col[0] = 42.0
        let row_b = [7.0f64, 200.0]; // col[0] = 7.0  (different)
        let row_c = [42.0f64, 300.0]; // col[0] = 42.0 (same as row_a)
        let h_a = cd.hash_key(&row_a).expect("ok");
        let h_b = cd.hash_key(&row_b).expect("ok");
        let h_c = cd.hash_key(&row_c).expect("ok");
        assert_eq!(h_a, h_c, "rows with same col[0] must hash identically");
        assert_ne!(h_a, h_b, "rows with different col[0] must hash differently");
        // Collect into a dedup set — should have exactly 2 unique hashes
        let hashes: std::collections::HashSet<i64> = [h_a, h_b, h_c].into_iter().collect();
        assert_eq!(hashes.len(), 2);
    }

    // -----------------------------------------------------------------------
    // HavingCompiler tests
    // -----------------------------------------------------------------------

    fn having_compiler() -> HavingCompiler {
        HavingCompiler::new()
    }

    fn agg_map(pairs: &[(&str, usize)]) -> AggVarMap {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn binding(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn test_having_sum_gt_threshold_pass() {
        // HAVING (?sum > 100) with sum = 150 → true
        let mut c = having_compiler();
        let map = agg_map(&[("sum", 0)]);
        let expr = FilterExpr::BinOp {
            op: BinOp::Gt,
            left: Box::new(FilterExpr::Variable("sum".to_string())),
            right: Box::new(FilterExpr::Literal(100.0)),
        };
        let cf = c.compile_having(expr, map).expect("compile ok");
        assert_eq!(cf.evaluate(&binding(&[("sum", 150.0)])), Some(true));
    }

    #[test]
    fn test_having_sum_gt_threshold_fail() {
        // HAVING (?sum > 100) with sum = 50 → false
        let mut c = having_compiler();
        let map = agg_map(&[("sum", 0)]);
        let expr = FilterExpr::BinOp {
            op: BinOp::Gt,
            left: Box::new(FilterExpr::Variable("sum".to_string())),
            right: Box::new(FilterExpr::Literal(100.0)),
        };
        let cf = c.compile_having(expr, map).expect("compile ok");
        assert_eq!(cf.evaluate(&binding(&[("sum", 50.0)])), Some(false));
    }

    #[test]
    fn test_having_count_ge_pass() {
        // HAVING (?count >= 5) with count = 7 → true
        let mut c = having_compiler();
        let map = agg_map(&[("count", 0)]);
        let expr = FilterExpr::BinOp {
            op: BinOp::Ge,
            left: Box::new(FilterExpr::Variable("count".to_string())),
            right: Box::new(FilterExpr::Literal(5.0)),
        };
        let cf = c.compile_having(expr, map).expect("compile ok");
        assert_eq!(cf.evaluate(&binding(&[("count", 7.0)])), Some(true));
    }

    #[test]
    fn test_having_combined() {
        // HAVING (?sum > 100 && ?count >= 5) — test both true and one-false branches
        let mut c = having_compiler();
        let map = agg_map(&[("sum", 0), ("count", 1)]);
        let sum_gt = FilterExpr::BinOp {
            op: BinOp::Gt,
            left: Box::new(FilterExpr::Variable("sum".to_string())),
            right: Box::new(FilterExpr::Literal(100.0)),
        };
        let count_ge = FilterExpr::BinOp {
            op: BinOp::Ge,
            left: Box::new(FilterExpr::Variable("count".to_string())),
            right: Box::new(FilterExpr::Literal(5.0)),
        };
        let expr = FilterExpr::BinOp {
            op: BinOp::And,
            left: Box::new(sum_gt),
            right: Box::new(count_ge),
        };
        let cf = c.compile_having(expr, map).expect("compile ok");

        // Both conditions pass
        assert_eq!(
            cf.evaluate(&binding(&[("sum", 150.0), ("count", 7.0)])),
            Some(true)
        );
        // sum fails
        assert_eq!(
            cf.evaluate(&binding(&[("sum", 50.0), ("count", 7.0)])),
            Some(false)
        );
        // count fails
        assert_eq!(
            cf.evaluate(&binding(&[("sum", 150.0), ("count", 3.0)])),
            Some(false)
        );
    }

    #[test]
    fn test_having_unknown_var_error() {
        // Variable not in agg_var_map → Err(UnsupportedExpression)
        use oxirs_arq::jit::FilterCompilerError;
        let mut c = having_compiler();
        let map = agg_map(&[("sum", 0)]);
        let expr = FilterExpr::BinOp {
            op: BinOp::Gt,
            left: Box::new(FilterExpr::Variable("not_in_map".to_string())),
            right: Box::new(FilterExpr::Literal(0.0)),
        };
        let result = c.compile_having(expr, map);
        assert!(
            matches!(result, Err(FilterCompilerError::UnsupportedExpression(_))),
            "expected UnsupportedExpression, got: {:?}",
            result
        );
    }

    #[test]
    fn test_having_literal_only() {
        // HAVING (1 < 2) — always true regardless of aggregate values
        let mut c = having_compiler();
        let expr = FilterExpr::BinOp {
            op: BinOp::Lt,
            left: Box::new(FilterExpr::Literal(1.0)),
            right: Box::new(FilterExpr::Literal(2.0)),
        };
        let cf = c
            .compile_having(expr, AggVarMap::new())
            .expect("compile ok");
        let empty: HashMap<String, f64> = HashMap::new();
        assert_eq!(cf.evaluate(&empty), Some(true));
    }

    // -----------------------------------------------------------------------
    // Cross-operator integration test
    // -----------------------------------------------------------------------

    #[test]
    fn test_project_then_distinct() {
        // Project subset of columns, then hash the projected row.
        // Source row: [100.0, 42.0, 7.0]
        // Project: col[2, 0] → [7.0, 100.0]
        // Distinct: hash col[0] of projected row (i.e. 7.0)
        let proj_specs = vec![ProjectSpec { src_idx: 2 }, ProjectSpec { src_idx: 0 }];
        let cp = ProjectCompiler::new()
            .compile(&proj_specs)
            .expect("project compile ok");

        let hash_specs = vec![DistinctKeySpec { col_idx: 0 }];
        let cd = DistinctCompiler::new()
            .compile(&hash_specs)
            .expect("distinct compile ok");

        let src = [100.0f64, 42.0, 7.0];
        let mut projected = Vec::new();
        assert!(cp.extract(&src, &mut projected));
        assert_eq!(projected, vec![7.0, 100.0]);

        // Hash of projected[0] = 7.0
        let hash = cd.hash_key(&projected).expect("hash ok");

        // The same hash must be produced when hashing [7.0, *] directly
        let direct_hash_specs = vec![DistinctKeySpec { col_idx: 0 }];
        let cd2 = DistinctCompiler::new()
            .compile(&direct_hash_specs)
            .expect("distinct compile ok");
        let direct_hash = cd2.hash_key(&[7.0f64, 0.0]).expect("hash ok");

        assert_eq!(
            hash, direct_hash,
            "hash of projected row col[0] must equal hash of the same f64 value directly"
        );
    }
}
