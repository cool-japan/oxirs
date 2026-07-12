//! Cranelift-based JIT compiler for ORDER BY expression evaluation (JIT phase c).
//!
//! Produces a native multi-column comparator with signature:
//! ```text
//! fn(left: *const f64, right: *const f64, n_cols: usize) -> i8
//! ```
//! Returns `-1` if `left < right`, `0` if equal, `1` if `left > right`,
//! following the lexicographic sort order defined by the supplied [`OrderKeySpec`] slice.
//! Each column independently carries an ascending/descending flag.
//!
//! # IR strategy
//!
//! We use a *select-chain* approach: for each column, compute a signed comparison
//! result (`-1`, `0`, or `1`) as an `i8`, then combine with:
//! ```text
//! accumulated = select(accumulated != 0, accumulated, this_col_result)
//! ```
//! This evaluates all columns eagerly (no control-flow branches) but avoids needing
//! multi-block IR, making it easier to verify and maintain. For the number of ORDER BY
//! columns typical in SPARQL queries (1–5), the performance difference versus a
//! branch-based early-return is negligible.
//!
//! # Safety
//!
//! The compiled function is called via an `unsafe extern "C" fn` pointer.
//! Callers must guarantee:
//! - `left` and `right` are valid, aligned, non-null pointers to at least
//!   `max(spec.col_idx) + 1` `f64` values each.
//! - Both pointers remain valid for the duration of the call.

use std::cmp::Ordering;
use std::sync::Arc;

use cranelift_codegen::ir::{
    condcodes::FloatCC, types, AbiParam, InstBuilder, MemFlagsData, Signature,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use super::filter_compiler::JITModuleOwner;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One sort key: a column index and sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrderKeySpec {
    /// Index into the f64 row slice.
    pub col_idx: usize,
    /// `true` for ascending, `false` for descending.
    pub ascending: bool,
}

/// The C-ABI function pointer produced by [`OrderCompiler::compile`].
///
/// # Safety
///
/// `left` and `right` must each point to at least `max(col_idx) + 1` `f64` values.
/// `n_cols` must equal the number of [`OrderKeySpec`] entries used during compilation.
type OrderFn = unsafe extern "C" fn(*const f64, *const f64, usize) -> i8;

/// A compiled multi-column ORDER BY comparator.
///
/// The wrapped function returns `-1`, `0`, or `1` (less / equal / greater).
///
/// # Ownership / safety invariant
///
/// `fn_ptr` is only valid while `_module_owner` is alive.
pub struct CompiledOrder {
    /// JIT-compiled function pointer.
    fn_ptr: OrderFn,
    /// Kept for `col_count()`.
    specs: Vec<OrderKeySpec>,
    /// Keeps the `JITModule` code pages alive.
    _module_owner: Arc<JITModuleOwner>,
}

// SAFETY: JITModule code pages are read-only after finalisation;
// the module is protected by `Arc<JITModuleOwner>`.
unsafe impl Send for CompiledOrder {}
unsafe impl Sync for CompiledOrder {}

impl std::fmt::Debug for CompiledOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledOrder")
            .field("col_count", &self.specs.len())
            .finish_non_exhaustive()
    }
}

impl CompiledOrder {
    /// Compare two rows and return a [`std::cmp::Ordering`].
    ///
    /// # Safety
    ///
    /// `left` and `right` must have at least `max(spec.col_idx) + 1` elements each.
    pub fn compare(&self, left: &[f64], right: &[f64]) -> Ordering {
        let n = self.specs.len();
        // SAFETY:
        // - `fn_ptr` is valid because `_module_owner` is still alive.
        // - `left.as_ptr()` and `right.as_ptr()` are valid for the duration when
        //   the caller obeys the documented precondition (indices in-bounds).
        // - The compiled function does not mutate either slice.
        let r = unsafe { (self.fn_ptr)(left.as_ptr(), right.as_ptr(), n) };
        match r {
            i if i < 0 => Ordering::Less,
            0 => Ordering::Equal,
            _ => Ordering::Greater,
        }
    }

    /// The number of sort-key columns this comparator was compiled for.
    pub fn col_count(&self) -> usize {
        self.specs.len()
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during JIT compilation of an ORDER BY comparator.
#[derive(Debug, thiserror::Error)]
pub enum OrderCompilerError {
    /// No key specs were provided.
    #[error("order compiler requires at least one key spec")]
    NoKeys,
    /// Cranelift reported a codegen or linkage error.
    #[error("JIT codegen error: {0}")]
    CodegenError(String),
    /// ISA builder failed to initialise.
    #[error("JIT ISA init error: {0}")]
    IsaInitError(String),
}

// ---------------------------------------------------------------------------
// Compiler (ZST — new JITModule per compile call)
// ---------------------------------------------------------------------------

/// Compiles multi-column ORDER BY comparators to native machine code via Cranelift.
///
/// Each call to [`compile`](OrderCompiler::compile) creates a fresh `JITModule`
/// so that the resulting [`CompiledOrder`] is independently owned.
pub struct OrderCompiler;

impl Default for OrderCompiler {
    fn default() -> Self {
        OrderCompiler
    }
}

impl OrderCompiler {
    /// Create a new `OrderCompiler`.
    pub fn new() -> Self {
        OrderCompiler
    }

    /// Compile a multi-column ORDER BY comparator.
    ///
    /// The resulting function returns `-1` (less), `0` (equal), or `1` (greater),
    /// respecting each column's ascending/descending flag and short-circuiting on
    /// the first non-equal column.
    ///
    /// # Errors
    ///
    /// Returns [`OrderCompilerError::NoKeys`] if `specs` is empty.
    pub fn compile(&self, specs: &[OrderKeySpec]) -> Result<CompiledOrder, OrderCompilerError> {
        if specs.is_empty() {
            return Err(OrderCompilerError::NoKeys);
        }
        let module = build_jit_module()?;
        let (fn_ptr, module) = compile_order_fn(module, specs)?;
        let owner = Arc::new(JITModuleOwner::new(module));
        Ok(CompiledOrder {
            fn_ptr,
            specs: specs.to_vec(),
            _module_owner: owner,
        })
    }
}

// ---------------------------------------------------------------------------
// Cranelift module setup
// ---------------------------------------------------------------------------

fn build_jit_module() -> Result<JITModule, OrderCompilerError> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|e| OrderCompilerError::CodegenError(e.to_string()))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| OrderCompilerError::CodegenError(e.to_string()))?;
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| OrderCompilerError::CodegenError(e.to_string()))?;

    let flags = settings::Flags::new(flag_builder);
    let isa = cranelift_native::builder()
        .map_err(|e| OrderCompilerError::IsaInitError(e.to_string()))?
        .finish(flags)
        .map_err(|e| OrderCompilerError::IsaInitError(e.to_string()))?;

    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    Ok(JITModule::new(builder))
}

// ---------------------------------------------------------------------------
// Cranelift IR code generation
// ---------------------------------------------------------------------------

/// Compile an ORDER BY comparator function into `module` and return the function
/// pointer together with the (now finalized) module.
fn compile_order_fn(
    mut module: JITModule,
    specs: &[OrderKeySpec],
) -> Result<(OrderFn, JITModule), OrderCompilerError> {
    let ptr_type = module.isa().pointer_type();

    // Signature: fn(*const f64, *const f64, usize) -> i8
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(ptr_type)); // left ptr
    sig.params.push(AbiParam::new(ptr_type)); // right ptr
    sig.params.push(AbiParam::new(ptr_type)); // n_cols (ABI only)
    sig.returns.push(AbiParam::new(types::I8));

    let func_id = module
        .declare_function("order_fn", Linkage::Local, &sig)
        .map_err(|e| OrderCompilerError::CodegenError(e.to_string()))?;

    {
        let mut ctx = module.make_context();
        ctx.func.signature = sig.clone();

        let mut fn_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let left_ptr = builder.block_params(entry_block)[0];
        let right_ptr = builder.block_params(entry_block)[1];
        // Third param (n_cols) is ABI-only; suppress "unused" at the IR level.
        let _n_cols = builder.block_params(entry_block)[2];

        // Use a select-chain: `accumulated = select(accumulated != 0, accumulated, this_col)`.
        // Start with 0 (equal); first non-equal column overwrites it and all subsequent
        // `select` instructions leave it unchanged.
        let zero_i8 = builder.ins().iconst(types::I8, 0);
        let mut accumulated = zero_i8;

        for spec in specs {
            let col_result = emit_col_comparison(&mut builder, spec, left_ptr, right_ptr)?;
            // `accumulated != 0` — any non-zero i8 means a decision has been made.
            let is_decided = builder.ins().icmp_imm(
                cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                accumulated,
                0,
            );
            // select(is_decided, accumulated, col_result):
            //   if already decided → keep accumulated
            //   else                → use this column's result
            accumulated = builder.ins().select(is_decided, accumulated, col_result);
        }

        builder.ins().return_(&[accumulated]);
        builder.finalize();

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| OrderCompilerError::CodegenError(format!("{e:?}")))?;
    }

    module
        .finalize_definitions()
        .map_err(|e| OrderCompilerError::CodegenError(format!("finalize_definitions: {e:?}")))?;

    // SAFETY: The function was just defined and finalized above; the pointer is valid.
    let raw_ptr = module.get_finalized_function(func_id);
    // SAFETY: We built the function with exactly this signature.
    let fn_ptr: OrderFn = unsafe { std::mem::transmute(raw_ptr) };

    Ok((fn_ptr, module))
}

/// Emit a single-column comparison and return an `i8` value: `-1`, `0`, or `1`.
///
/// For ascending: returns `lt ? -1 : (eq ? 0 : 1)`.
/// For descending: returns `lt ? 1 : (eq ? 0 : -1)`.
///
/// Uses `select` to implement the three-way conditional without control-flow blocks.
fn emit_col_comparison(
    builder: &mut FunctionBuilder<'_>,
    spec: &OrderKeySpec,
    left_ptr: cranelift_codegen::ir::Value,
    right_ptr: cranelift_codegen::ir::Value,
) -> Result<cranelift_codegen::ir::Value, OrderCompilerError> {
    let offset = col_byte_offset(spec.col_idx)?;

    // Load both f64 values.
    // cranelift 0.133: the old `MemFlags::trusted()` value type is now `MemFlagsData`
    // (InstBuilder interns it into the DFG's MemFlagsSet internally).
    let lv = builder
        .ins()
        .load(types::F64, MemFlagsData::trusted(), left_ptr, offset);
    let rv = builder
        .ins()
        .load(types::F64, MemFlagsData::trusted(), right_ptr, offset);

    // fcmp returns I8 (0 or 1) in modern Cranelift — no bint needed.
    // NaN comparisons: LessThan → false (0), Equal → false (0), so NaN compares
    // neither less nor equal, resulting in the "greater" branch — consistent with
    // total ordering where NaN sorts last.
    let is_lt = builder.ins().fcmp(FloatCC::LessThan, lv, rv);
    let is_eq = builder.ins().fcmp(FloatCC::Equal, lv, rv);

    // Three-way result constants.
    let (neg_one, pos_one) = if spec.ascending {
        (
            builder.ins().iconst(types::I8, -1i64),
            builder.ins().iconst(types::I8, 1i64),
        )
    } else {
        // Descending: flip the sign of the non-equal cases.
        (
            builder.ins().iconst(types::I8, 1i64),
            builder.ins().iconst(types::I8, -1i64),
        )
    };
    let zero_i8 = builder.ins().iconst(types::I8, 0);

    // result = is_eq ? 0 : (is_lt ? neg_one : pos_one)
    // Built from the inside out:
    //   inner = select(is_lt, neg_one, pos_one)
    //   outer = select(is_eq,  zero,   inner   )
    let inner = builder.ins().select(is_lt, neg_one, pos_one);
    let result = builder.ins().select(is_eq, zero_i8, inner);

    Ok(result)
}

/// Convert a column index to a `i32` byte offset for Cranelift `load`.
fn col_byte_offset(idx: usize) -> Result<i32, OrderCompilerError> {
    let byte = idx.checked_mul(std::mem::size_of::<f64>()).ok_or_else(|| {
        OrderCompilerError::CodegenError(format!("column index {idx} overflows byte offset"))
    })?;
    i32::try_from(byte).map_err(|_| {
        OrderCompilerError::CodegenError(format!(
            "column index {idx} byte offset {} exceeds i32::MAX",
            byte
        ))
    })
}

// ---------------------------------------------------------------------------
// Unit tests (inline)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn compiler() -> OrderCompiler {
        OrderCompiler::new()
    }

    #[test]
    fn test_no_keys_error() {
        let result = compiler().compile(&[]);
        assert!(matches!(result, Err(OrderCompilerError::NoKeys)));
    }

    #[test]
    fn test_col_byte_offset_zero() {
        assert_eq!(col_byte_offset(0).expect("ok"), 0);
    }

    #[test]
    fn test_col_byte_offset_two() {
        assert_eq!(col_byte_offset(2).expect("ok"), 16);
    }
}
