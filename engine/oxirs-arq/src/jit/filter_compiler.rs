//! Cranelift-based SPARQL filter expression compiler (JIT phase b).
//!
//! Translates a supported subset of SPARQL [`Expression`] nodes into native machine code
//! using Cranelift, returning a [`CompiledFilter`] that evaluates much faster than the
//! interpreted expression evaluator for hot numeric-filter queries.
//!
//! # Unsupported expressions
//!
//! [`try_lower`] returns `None` (fall back to interpreter) for:
//! - String literals or language-tagged literals
//! - String functions, REGEX, CONTAINS, LANGMATCHES
//! - Type-test functions: ISIRI, ISBLANK, ISLITERAL, ISNUMERIC
//! - Mixed-type comparisons, BOUND, EXISTS / NOT EXISTS
//! - Any expression variant not in the supported subset

use std::collections::HashMap;
use std::sync::Arc;

use cranelift_codegen::ir::{
    condcodes::FloatCC, types, AbiParam, InstBuilder, MemFlagsData, Signature, Value,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::algebra::{BinaryOperator, Expression, Literal, UnaryOperator};

/// Maps variable name (without `?` prefix) to its index in the `f64` input slice.
pub type VarIndexMap = HashMap<String, usize>;

// ---------------------------------------------------------------------------
// Simplified filter IR
// ---------------------------------------------------------------------------

/// Simplified filter expression for JIT compilation.
///
/// Only the supported subset of SPARQL filter expressions is representable here.
/// Use [`try_lower`] to convert an [`Expression`] into this type.
#[derive(Debug, Clone)]
pub enum FilterExpr {
    /// A numeric constant (f64).
    Literal(f64),
    /// A variable reference — `?x` maps to a slot index in the caller's f64 slice.
    Variable(String),
    /// A binary operation (arithmetic, comparison, or logical).
    BinOp {
        op: BinOp,
        left: Box<FilterExpr>,
        right: Box<FilterExpr>,
    },
    /// Logical NOT of an inner boolean expression.
    UnaryNot(Box<FilterExpr>),
    /// A built-in numeric function applied to one argument.
    Builtin {
        func: BuiltinFunc,
        arg: Box<FilterExpr>,
    },
}

/// Binary operators supported by the JIT compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// Arithmetic `+`
    Add,
    /// Arithmetic `-`
    Sub,
    /// Arithmetic `*`
    Mul,
    /// Arithmetic `/`
    Div,
    /// Comparison `<`
    Lt,
    /// Comparison `>`
    Gt,
    /// Comparison `<=`
    Le,
    /// Comparison `>=`
    Ge,
    /// Comparison `=`
    Eq,
    /// Comparison `!=`
    Ne,
    /// Logical `&&`
    And,
    /// Logical `||`
    Or,
}

/// Built-in numeric functions supported by the JIT compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinFunc {
    /// `ABS(?x)` — absolute value
    Abs,
    /// `CEIL(?x)` — ceiling
    Ceil,
    /// `FLOOR(?x)` — floor
    Floor,
    /// `ROUND(?x)` — round half away from zero (SPARQL semantics, not IEEE 754 round-half-to-even)
    Round,
}

// ---------------------------------------------------------------------------
// Lowering: Expression → FilterExpr
// ---------------------------------------------------------------------------

/// XSD numeric datatype IRIs recognised by the JIT lowering pass.
const XSD_NUMERIC_TYPES: &[&str] = &[
    "http://www.w3.org/2001/XMLSchema#integer",
    "http://www.w3.org/2001/XMLSchema#decimal",
    "http://www.w3.org/2001/XMLSchema#double",
    "http://www.w3.org/2001/XMLSchema#float",
    "http://www.w3.org/2001/XMLSchema#int",
    "http://www.w3.org/2001/XMLSchema#long",
    "http://www.w3.org/2001/XMLSchema#short",
    "http://www.w3.org/2001/XMLSchema#byte",
    "http://www.w3.org/2001/XMLSchema#unsignedInt",
    "http://www.w3.org/2001/XMLSchema#unsignedLong",
    "http://www.w3.org/2001/XMLSchema#nonNegativeInteger",
    "http://www.w3.org/2001/XMLSchema#positiveInteger",
];

/// Try to parse an algebra [`Literal`] as an `f64`.
///
/// Returns `None` if the literal has a language tag, an unsupported datatype, or
/// if the value string cannot be parsed.
fn literal_to_f64(lit: &Literal) -> Option<f64> {
    // Language-tagged literals are never numeric
    if lit.language.is_some() {
        return None;
    }
    match &lit.datatype {
        Some(dt) => {
            let iri = dt.as_str();
            if !XSD_NUMERIC_TYPES.contains(&iri) {
                return None;
            }
            lit.value.parse::<f64>().ok()
        }
        None => {
            // Plain literal — attempt numeric parse only if it looks numeric
            if lit.value.parse::<f64>().is_ok() {
                lit.value.parse::<f64>().ok()
            } else {
                None
            }
        }
    }
}

/// Attempt to lower a SPARQL [`Expression`] into a [`FilterExpr`] and build a
/// [`VarIndexMap`] mapping variable names to their slice index.
///
/// Returns `None` if the expression uses unsupported operations.
pub fn try_lower(expr: &Expression) -> Option<(FilterExpr, VarIndexMap)> {
    let mut var_map: VarIndexMap = HashMap::new();
    let filter_expr = lower_expr(expr, &mut var_map)?;
    Some((filter_expr, var_map))
}

fn lower_expr(expr: &Expression, var_map: &mut VarIndexMap) -> Option<FilterExpr> {
    match expr {
        Expression::Literal(lit) => {
            let v = literal_to_f64(lit)?;
            Some(FilterExpr::Literal(v))
        }
        Expression::Variable(var) => {
            let name = var.name().to_string();
            let next_idx = var_map.len();
            let idx = *var_map.entry(name.clone()).or_insert(next_idx);
            // Sanity: ensure no duplicate variable gets a different index
            debug_assert_eq!(var_map[&name], idx);
            Some(FilterExpr::Variable(name))
        }
        Expression::Binary { op, left, right } => {
            let jit_op = match op {
                BinaryOperator::Add => BinOp::Add,
                BinaryOperator::Subtract => BinOp::Sub,
                BinaryOperator::Multiply => BinOp::Mul,
                BinaryOperator::Divide => BinOp::Div,
                BinaryOperator::Less => BinOp::Lt,
                BinaryOperator::Greater => BinOp::Gt,
                BinaryOperator::LessEqual => BinOp::Le,
                BinaryOperator::GreaterEqual => BinOp::Ge,
                BinaryOperator::Equal => BinOp::Eq,
                BinaryOperator::NotEqual => BinOp::Ne,
                BinaryOperator::And => BinOp::And,
                BinaryOperator::Or => BinOp::Or,
                // SameTerm / In / NotIn are not supported in this JIT subset
                _ => return None,
            };
            let l = lower_expr(left, var_map)?;
            let r = lower_expr(right, var_map)?;
            Some(FilterExpr::BinOp {
                op: jit_op,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        Expression::Unary { op, operand } => match op {
            UnaryOperator::Not => {
                let inner = lower_expr(operand, var_map)?;
                Some(FilterExpr::UnaryNot(Box::new(inner)))
            }
            // Unary + / - on numerics: convert to BinOp(0 ± expr)
            UnaryOperator::Plus => lower_expr(operand, var_map),
            UnaryOperator::Minus => {
                let inner = lower_expr(operand, var_map)?;
                Some(FilterExpr::BinOp {
                    op: BinOp::Sub,
                    left: Box::new(FilterExpr::Literal(0.0)),
                    right: Box::new(inner),
                })
            }
            // Type tests are not supported by the JIT subset
            _ => None,
        },
        Expression::Function { name, args } => {
            // We support only the four numeric built-ins
            let func = match name.to_uppercase().as_str() {
                "ABS" => BuiltinFunc::Abs,
                "CEIL" => BuiltinFunc::Ceil,
                "FLOOR" => BuiltinFunc::Floor,
                "ROUND" => BuiltinFunc::Round,
                _ => return None,
            };
            if args.len() != 1 {
                return None;
            }
            let arg = lower_expr(&args[0], var_map)?;
            Some(FilterExpr::Builtin {
                func,
                arg: Box::new(arg),
            })
        }
        // IRIs, Bound, Exists, NotExists, Conditional — not supported
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Compiled filter
// ---------------------------------------------------------------------------

/// The raw C-ABI function pointer produced by the JIT compiler.
///
/// # Safety
///
/// The function must be called with:
/// - `ptr`: a valid pointer to at least `n` `f64` values
/// - `n`: the number of f64 values (must equal `var_map.len()`)
///
/// Returns: `1` = true (pass filter), `0` = false (reject), `-1` = error
type FilterFn = unsafe extern "C" fn(*const f64, usize) -> i8;

/// A compiled SPARQL filter expression that can evaluate bindings via native code.
///
/// # Lifetime / ownership
///
/// The `CompiledFilter` holds an `Arc` to the [`JITModule`] that owns the compiled
/// function's code pages. Dropping the last `Arc<CompiledFilter>` frees those pages.
/// Callers that clone the `Arc<CompiledFilter>` may hold it independently of the
/// `JitFilterCache`.
pub struct CompiledFilter {
    /// Raw pointer into JIT-compiled memory.
    ///
    /// # Safety invariant
    ///
    /// `fn_ptr` is only valid while `_module_owner` is alive.
    fn_ptr: FilterFn,
    /// Variable name → slice-index mapping used when building the argument slice.
    pub var_map: VarIndexMap,
    /// Keeps the JITModule alive as long as this `CompiledFilter` exists.
    _module_owner: Arc<JITModuleOwner>,
}

// SAFETY: JITModule code pages are thread-safe for read-only execution once finalized.
// The module owner is protected behind an `Arc`; `fn_ptr` is an immutable function.
unsafe impl Send for CompiledFilter {}
unsafe impl Sync for CompiledFilter {}

impl std::fmt::Debug for CompiledFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledFilter")
            .field("var_map", &self.var_map)
            .finish_non_exhaustive()
    }
}

impl CompiledFilter {
    /// Evaluate the compiled filter with numeric variable bindings.
    ///
    /// Returns `Some(true)` if the filter passes, `Some(false)` if it rejects, or
    /// `None` if a required variable is missing from `binding` (caller should fall
    /// back to the interpreted evaluator).
    pub fn evaluate(&self, binding: &HashMap<String, f64>) -> Option<bool> {
        let n = self.var_map.len();
        let mut values = vec![0.0f64; n];
        for (name, &idx) in &self.var_map {
            match binding.get(name) {
                Some(&v) => values[idx] = v,
                // Missing variable → cannot evaluate, fall back to interpreter
                None => return None,
            }
        }
        // SAFETY:
        // - `values` has exactly `n` elements, matching what the compiled function expects.
        // - `fn_ptr` is valid because `_module_owner` is still alive (we hold a reference).
        // - The compiled function does not mutate the slice; no aliasing issues.
        let result = unsafe { (self.fn_ptr)(values.as_ptr(), values.len()) };
        match result {
            1 => Some(true),
            0 => Some(false),
            // -1 signals an internal error in the JIT function (should not happen for
            // well-formed expressions, but we handle it defensively)
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// JIT module owner (keeps code pages alive)
// ---------------------------------------------------------------------------

/// Wrapper that owns a `JITModule` so that its lifetime can be shared via `Arc`.
pub(crate) struct JITModuleOwner {
    // The `JITModule` must be stored here (never dropped while `CompiledFilter`s exist).
    // We hold it in a `Mutex` so `FilterCompiler` can call `finalize_definitions` safely,
    // but once the module is handed off (wrapped in `Arc<JITModuleOwner>`) we never
    // call mutable methods on it again.
    //
    // The field appears "never read" to the compiler because its purpose is ownership
    // / drop-guard, not data access.
    #[allow(dead_code)]
    module: std::sync::Mutex<JITModule>,
}

impl JITModuleOwner {
    /// Wrap a finalized `JITModule` so its code pages remain alive until the last
    /// `Arc<JITModuleOwner>` is dropped.
    pub(crate) fn new(module: JITModule) -> Self {
        JITModuleOwner {
            module: std::sync::Mutex::new(module),
        }
    }
}

// SAFETY: JITModule itself is not Send/Sync in the cranelift crate, but once its
// definitions are finalized the code pages are read-only and safe for concurrent
// execution. We wrap it in a Mutex for the rare case where internal cranelift state
// is accessed during function pointer resolution; the actual `FilterFn` calls never
// touch the module.
unsafe impl Send for JITModuleOwner {}
unsafe impl Sync for JITModuleOwner {}

// ---------------------------------------------------------------------------
// Filter compiler errors
// ---------------------------------------------------------------------------

/// Errors that can occur during JIT compilation of a filter expression.
#[derive(Debug, thiserror::Error)]
pub enum FilterCompilerError {
    /// The expression contains operations not supported by the JIT compiler.
    #[error("expression not in JIT-supported subset: {0}")]
    UnsupportedExpression(String),
    /// Cranelift reported a codegen error.
    #[error("JIT codegen error: {0}")]
    CodegenError(String),
    /// Function definition or linkage failed.
    #[error("JIT linkage error: {0}")]
    LinkageError(String),
    /// Native ISA builder failed to initialise.
    #[error("JIT ISA init error: {0}")]
    IsaInitError(String),
}

// ---------------------------------------------------------------------------
// Filter compiler
// ---------------------------------------------------------------------------

/// Compiles supported SPARQL filter expressions to native machine code via Cranelift.
///
/// Each call to [`compile`](FilterCompiler::compile) creates a new `JITModule`
/// (one module per compiled function) so that the resulting `CompiledFilter` can be
/// independently owned and dropped without interfering with other compiled filters.
pub struct FilterCompiler;

impl Default for FilterCompiler {
    fn default() -> Self {
        FilterCompiler
    }
}

impl FilterCompiler {
    /// Create a new `FilterCompiler`.
    ///
    /// This is infallible because no Cranelift state is initialised until
    /// [`compile`](Self::compile) is called.
    pub fn new() -> Self {
        FilterCompiler
    }

    /// Try to compile `expr` + `var_map` into native machine code.
    ///
    /// Returns `Ok(None)` if the expression is not in the supported subset (the
    /// caller should use the interpreted fallback).  Returns `Ok(Some(compiled))`
    /// on success.
    pub fn compile(
        &self,
        expr: &FilterExpr,
        var_map: VarIndexMap,
    ) -> Result<Option<CompiledFilter>, FilterCompilerError> {
        // Build a fresh JITModule for this function
        let module = build_jit_module()?;
        let (fn_ptr, module) = compile_filter_fn(module, expr, &var_map)?;

        // Wrap the module in an Arc so it stays alive as long as the CompiledFilter does
        let owner = Arc::new(JITModuleOwner::new(module));

        Ok(Some(CompiledFilter {
            fn_ptr,
            var_map,
            _module_owner: owner,
        }))
    }
}

// ---------------------------------------------------------------------------
// Cranelift module setup
// ---------------------------------------------------------------------------

fn build_jit_module() -> Result<JITModule, FilterCompilerError> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|e| FilterCompilerError::CodegenError(e.to_string()))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| FilterCompilerError::CodegenError(e.to_string()))?;
    // Speed over compilation time — we compile rarely but execute often
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| FilterCompilerError::CodegenError(e.to_string()))?;

    let flags = settings::Flags::new(flag_builder);
    let isa = cranelift_native::builder()
        .map_err(|e| FilterCompilerError::IsaInitError(e.to_string()))?
        .finish(flags)
        .map_err(|e| FilterCompilerError::IsaInitError(e.to_string()))?;

    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    Ok(JITModule::new(builder))
}

// ---------------------------------------------------------------------------
// Cranelift IR code generation
// ---------------------------------------------------------------------------

/// Compile a `FilterExpr` into a function inside `module` and return the
/// function pointer together with the (now finalized) module.
fn compile_filter_fn(
    mut module: JITModule,
    expr: &FilterExpr,
    var_map: &VarIndexMap,
) -> Result<(FilterFn, JITModule), FilterCompilerError> {
    // Pointer width for the host platform
    let ptr_type = module.isa().pointer_type();

    // Build function signature:  fn(*const f64, usize) -> i8
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(ptr_type)); // ptr: *const f64
    sig.params.push(AbiParam::new(ptr_type)); // n:   usize (pointer-width int)
    sig.returns.push(AbiParam::new(types::I8)); // return: i8 (1/0/-1)

    let func_id = module
        .declare_function("filter_fn", Linkage::Local, &sig)
        .map_err(|e| FilterCompilerError::LinkageError(e.to_string()))?;

    // Generate the function body
    {
        let mut ctx = module.make_context();
        ctx.func.signature = sig.clone();

        let mut fn_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Extract function parameters
        let ptr_val = builder.block_params(entry_block)[0];
        // n_val is the second param (not used for bounds-checking — caller guarantees it)
        // We keep it so the ABI matches; suppress "unused" warning via a black-box store.
        let _n_val = builder.block_params(entry_block)[1];

        // Compile the expression tree and get a result value
        let result_val = emit_expr(&mut builder, expr, var_map, ptr_val, ptr_type)?;

        // The result of a comparison / logical op is already I8 (0 or 1).
        // For a numeric-only expression at the top level (shouldn't happen in valid
        // SPARQL but handled defensively), we interpret non-zero as true.
        let result_i8 = coerce_to_i8(&mut builder, result_val)?;

        builder.ins().return_(&[result_i8]);
        builder.finalize();

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| FilterCompilerError::CodegenError(format!("{e:?}")))?;
    }

    module
        .finalize_definitions()
        .map_err(|e| FilterCompilerError::CodegenError(format!("finalize_definitions: {e:?}")))?;

    // SAFETY: The function was just defined and finalized above, so the pointer is valid.
    let raw_ptr = module.get_finalized_function(func_id);
    // SAFETY: We constructed the function with the matching signature.
    let fn_ptr: FilterFn = unsafe { std::mem::transmute(raw_ptr) };

    Ok((fn_ptr, module))
}

// ---------------------------------------------------------------------------
// Cranelift IR emitter
// ---------------------------------------------------------------------------

/// Recursively emit Cranelift IR for `expr` and return the produced `Value`.
///
/// The returned `Value` has type `F64` for numeric expressions and `I8` for boolean ones.
fn emit_expr(
    builder: &mut FunctionBuilder<'_>,
    expr: &FilterExpr,
    var_map: &VarIndexMap,
    ptr_val: Value,
    ptr_type: types::Type,
) -> Result<Value, FilterCompilerError> {
    match expr {
        FilterExpr::Literal(v) => {
            let val = builder.ins().f64const(*v);
            Ok(val)
        }

        FilterExpr::Variable(name) => {
            let idx = var_map.get(name).copied().ok_or_else(|| {
                FilterCompilerError::UnsupportedExpression(format!(
                    "variable '{}' not found in var_map",
                    name
                ))
            })?;
            let byte_offset = (idx * std::mem::size_of::<f64>()) as i32;
            // SAFETY (documented at the call site in `CompiledFilter::evaluate`):
            //   - ptr_val points to a slice of at least `var_map.len()` f64 values.
            //   - `byte_offset` is within bounds because `idx < var_map.len()`.
            // cranelift 0.133: the old `MemFlags::trusted()` value type is now
            // `MemFlagsData` (InstBuilder interns it into the DFG's MemFlagsSet).
            let val = builder
                .ins()
                .load(types::F64, MemFlagsData::trusted(), ptr_val, byte_offset);
            Ok(val)
        }

        FilterExpr::BinOp { op, left, right } => {
            emit_binop(builder, *op, left, right, var_map, ptr_val, ptr_type)
        }

        FilterExpr::UnaryNot(inner) => {
            let inner_val = emit_expr(builder, inner, var_map, ptr_val, ptr_type)?;
            let bool_val = coerce_to_i8(builder, inner_val)?;
            // NOT: flip 0↔1 via XOR with 1
            let notted = builder.ins().bxor_imm(bool_val, 1);
            Ok(notted)
        }

        FilterExpr::Builtin { func, arg } => {
            let arg_val = emit_expr(builder, arg, var_map, ptr_val, ptr_type)?;
            // Argument must be F64
            let f_val = coerce_to_f64(builder, arg_val)?;
            let result = match func {
                BuiltinFunc::Abs => builder.ins().fabs(f_val),
                BuiltinFunc::Ceil => builder.ins().ceil(f_val),
                BuiltinFunc::Floor => builder.ins().floor(f_val),
                BuiltinFunc::Round => emit_sparql_round(builder, f_val),
            };
            Ok(result)
        }
    }
}

// ---------------------------------------------------------------------------
// SPARQL ROUND semantics: round-half-away-from-zero
// ---------------------------------------------------------------------------

/// Emit SPARQL-conformant `ROUND(?x)`.
///
/// The SPARQL 1.1 spec (§17.4.4.4) delegates to XPath `fn:round`, which states:
/// "If there are two integers equally close to \$arg, then the one that is closest
/// to positive infinity is returned."  This means half-integers round *toward
/// positive infinity* — distinct from IEEE 754 round-half-to-even (banker's rounding)
/// and also distinct from "round-half-away-from-zero":
///
/// | Input | SPARQL (XPath fn:round) | IEEE 754 nearest | away-from-zero |
/// |-------|------------------------|------------------|----------------|
/// | 0.5   | 1.0                    | 0.0 (even)       | 1.0            |
/// | -0.5  | 0.0                    | 0.0 (even)       | -1.0           |
/// | 2.5   | 3.0                    | 2.0 (even)       | 3.0            |
/// | -2.5  | -2.0                   | -2.0 (even)      | -3.0           |
///
/// The formula `floor(x + 0.5)` satisfies the XPath semantics exactly for all
/// finite inputs and correctly propagates NaN and infinities.
fn emit_sparql_round(builder: &mut FunctionBuilder<'_>, x: Value) -> Value {
    // half = 0.5
    let half = builder.ins().f64const(0.5);
    // shifted = x + 0.5
    let shifted = builder.ins().fadd(x, half);
    // result = floor(x + 0.5) — rounds ties toward +infinity per XPath fn:round
    builder.ins().floor(shifted)
}

fn emit_binop(
    builder: &mut FunctionBuilder<'_>,
    op: BinOp,
    left: &FilterExpr,
    right: &FilterExpr,
    var_map: &VarIndexMap,
    ptr_val: Value,
    ptr_type: types::Type,
) -> Result<Value, FilterCompilerError> {
    match op {
        // Arithmetic: both operands must be F64
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
            let lv = emit_expr(builder, left, var_map, ptr_val, ptr_type)?;
            let rv = emit_expr(builder, right, var_map, ptr_val, ptr_type)?;
            let lf = coerce_to_f64(builder, lv)?;
            let rf = coerce_to_f64(builder, rv)?;
            let result = match op {
                BinOp::Add => builder.ins().fadd(lf, rf),
                BinOp::Sub => builder.ins().fsub(lf, rf),
                BinOp::Mul => builder.ins().fmul(lf, rf),
                BinOp::Div => builder.ins().fdiv(lf, rf),
                _ => unreachable!(),
            };
            Ok(result)
        }

        // Numeric comparisons: operands F64, result I8
        BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge | BinOp::Eq | BinOp::Ne => {
            let lv = emit_expr(builder, left, var_map, ptr_val, ptr_type)?;
            let rv = emit_expr(builder, right, var_map, ptr_val, ptr_type)?;
            let lf = coerce_to_f64(builder, lv)?;
            let rf = coerce_to_f64(builder, rv)?;
            let float_cc = match op {
                BinOp::Lt => FloatCC::LessThan,
                BinOp::Gt => FloatCC::GreaterThan,
                BinOp::Le => FloatCC::LessThanOrEqual,
                BinOp::Ge => FloatCC::GreaterThanOrEqual,
                BinOp::Eq => FloatCC::Equal,
                BinOp::Ne => FloatCC::NotEqual,
                _ => unreachable!(),
            };
            // In Cranelift 0.91+ boolean types were merged with integer types.
            // `fcmp` now returns I8 (0 or 1) directly — no `bint` needed.
            let cmp = builder.ins().fcmp(float_cc, lf, rf);
            // fcmp returns I8 in modern Cranelift
            Ok(cmp)
        }

        // Logical AND / OR: both operands boolean (I8)
        BinOp::And | BinOp::Or => {
            let lv = emit_expr(builder, left, var_map, ptr_val, ptr_type)?;
            let rv = emit_expr(builder, right, var_map, ptr_val, ptr_type)?;
            let lb = coerce_to_i8(builder, lv)?;
            let rb = coerce_to_i8(builder, rv)?;
            let result = match op {
                BinOp::And => builder.ins().band(lb, rb),
                BinOp::Or => builder.ins().bor(lb, rb),
                _ => unreachable!(),
            };
            Ok(result)
        }
    }
}

// ---------------------------------------------------------------------------
// Type coercion helpers
// ---------------------------------------------------------------------------

/// Coerce a Cranelift `Value` to `I8` (boolean).
///
/// If the value is already `I8`, returns it unchanged.
/// If the value is `F64` (should not happen at the top level but can happen
/// in `!<float>` or `<float> && <float>` constructs), we convert non-zero → 1.
fn coerce_to_i8(
    builder: &mut FunctionBuilder<'_>,
    val: Value,
) -> Result<Value, FilterCompilerError> {
    let ty = builder.func.dfg.value_type(val);
    if ty == types::I8 {
        return Ok(val);
    }
    if ty == types::F64 {
        // non-zero ordered float → 1, zero or NaN → 0.
        // `OrderedNotEqual` is true iff both operands are non-NaN *and* they differ,
        // so NaN inputs correctly produce 0 (false), matching SPARQL error semantics.
        let zero = builder.ins().f64const(0.0);
        let cmp = builder.ins().fcmp(FloatCC::OrderedNotEqual, val, zero);
        return Ok(cmp);
    }
    Err(FilterCompilerError::CodegenError(format!(
        "coerce_to_i8: unexpected type {:?}",
        ty
    )))
}

/// Coerce a Cranelift `Value` to `F64`.
///
/// Returns an error if the value is not already `F64` (booleans cannot be
/// meaningfully coerced to floats in the JIT-supported subset).
fn coerce_to_f64(
    builder: &mut FunctionBuilder<'_>,
    val: Value,
) -> Result<Value, FilterCompilerError> {
    let ty = builder.func.dfg.value_type(val);
    if ty == types::F64 {
        return Ok(val);
    }
    // Bool (I8) to float: 0 → 0.0, 1 → 1.0 via unsigned int-to-float
    if ty == types::I8 {
        let as_i32 = builder.ins().uextend(types::I32, val);
        let as_f64 = builder.ins().fcvt_from_uint(types::F64, as_i32);
        return Ok(as_f64);
    }
    Err(FilterCompilerError::CodegenError(format!(
        "coerce_to_f64: unexpected type {:?}",
        ty
    )))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{BinaryOperator, Expression, Literal, UnaryOperator};
    use oxirs_core::model::{NamedNode, Variable as CoreVariable};

    fn xsd_integer() -> NamedNode {
        NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").expect("valid XSD URI")
    }

    fn int_lit(v: i64) -> Expression {
        Expression::Literal(Literal::typed(v.to_string(), xsd_integer()))
    }

    fn var(name: &str) -> Expression {
        Expression::Variable(CoreVariable::new(name).expect("valid variable"))
    }

    // -- try_lower tests ----------------------------------------------------

    #[test]
    fn lower_integer_literal() {
        let expr = int_lit(42);
        let (fe, vm) = try_lower(&expr).expect("should lower");
        assert!(vm.is_empty());
        assert!(matches!(fe, FilterExpr::Literal(v) if (v - 42.0).abs() < 1e-9));
    }

    #[test]
    fn lower_variable() {
        let expr = var("x");
        let (fe, vm) = try_lower(&expr).expect("should lower");
        assert_eq!(vm.len(), 1);
        assert!(vm.contains_key("x"));
        assert!(matches!(fe, FilterExpr::Variable(n) if n == "x"));
    }

    #[test]
    fn lower_comparison() {
        let expr = Expression::Binary {
            op: BinaryOperator::Greater,
            left: Box::new(var("x")),
            right: Box::new(int_lit(5)),
        };
        let (fe, _vm) = try_lower(&expr).expect("should lower");
        assert!(matches!(fe, FilterExpr::BinOp { op: BinOp::Gt, .. }));
    }

    #[test]
    fn lower_lang_tagged_literal_fails() {
        let expr = Expression::Literal(Literal::with_language(
            "hello".to_string(),
            "en".to_string(),
        ));
        assert!(
            try_lower(&expr).is_none(),
            "lang-tagged literal should not lower"
        );
    }

    #[test]
    fn lower_iri_fails() {
        let expr = Expression::Iri(xsd_integer());
        assert!(try_lower(&expr).is_none(), "IRI should not lower");
    }

    #[test]
    fn lower_unary_not() {
        let inner = Expression::Binary {
            op: BinaryOperator::Greater,
            left: Box::new(var("x")),
            right: Box::new(int_lit(0)),
        };
        let expr = Expression::Unary {
            op: UnaryOperator::Not,
            operand: Box::new(inner),
        };
        let (fe, _vm) = try_lower(&expr).expect("should lower");
        assert!(matches!(fe, FilterExpr::UnaryNot(_)));
    }

    #[test]
    fn lower_abs_builtin() {
        let expr = Expression::Function {
            name: "ABS".to_string(),
            args: vec![var("x")],
        };
        let (fe, _) = try_lower(&expr).expect("should lower");
        assert!(matches!(
            fe,
            FilterExpr::Builtin {
                func: BuiltinFunc::Abs,
                ..
            }
        ));
    }
}
