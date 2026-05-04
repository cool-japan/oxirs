//! Cranelift-based JIT compiler for hash-join key comparison (JIT phase c).
//!
//! Produces a native function with signature:
//! ```text
//! fn(left: *const f64, right: *const f64, n_keys: usize) -> i8
//! ```
//! Returns `1` if all key columns match, `0` otherwise.
//!
//! # Comparison modes
//!
//! Each key column chooses one of two modes:
//! - **Epsilon**: `|left[i] - right[i]| < 1e-9` (ordered float comparison, NaN → not equal).
//! - **Exact**: bitcast both operands to `i64` and check `icmp Equal`. Under this mode,
//!   two values with identical bit patterns — including two copies of the *same* NaN — compare
//!   as equal, because IEEE 754 NaN bit patterns are just ordinary 64-bit integers to `icmp`.
//!
//! # Safety
//!
//! The compiled function is called via an `unsafe extern "C" fn` pointer.
//! Callers must guarantee:
//! - `left` and `right` are valid, aligned, non-null pointers to at least
//!   `max(spec.left_idx) + 1` and `max(spec.right_idx) + 1` `f64` values, respectively.
//! - Both pointers remain valid for the duration of the call.

use std::sync::Arc;

use cranelift_codegen::ir::{
    condcodes::{FloatCC, IntCC},
    types, AbiParam, InstBuilder, MemFlags, Signature,
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

/// Configuration for a single join key column.
#[derive(Debug, Clone)]
pub struct JoinKeySpec {
    /// Index into the left-row `f64` slice.
    pub left_idx: usize,
    /// Index into the right-row `f64` slice.
    pub right_idx: usize,
    /// If `true`, use epsilon comparison (`|a - b| < 1e-9`).
    /// If `false`, use exact bit equality (bitcast to `i64` then `icmp Equal`).
    pub numeric_epsilon: bool,
}

/// The C-ABI function pointer produced by [`JoinCompiler::compile`].
///
/// # Safety
///
/// `left` and `right` must each point to enough `f64` values to cover all
/// `left_idx` / `right_idx` in the compiled spec. `n_keys` must equal the
/// number of [`JoinKeySpec`] entries the function was compiled with.
type JoinKeyFn = unsafe extern "C" fn(*const f64, *const f64, usize) -> i8;

/// A compiled join-key comparator produced by [`JoinCompiler::compile`].
///
/// The wrapped function returns `1` if all key columns match and `0` otherwise.
///
/// # Ownership / safety invariant
///
/// `fn_ptr` is only valid while `_module_owner` is alive.
/// Cloning an `Arc<CompiledJoinKey>` extends the lifetime safely.
pub struct CompiledJoinKey {
    /// JIT-compiled function pointer.
    fn_ptr: JoinKeyFn,
    /// Key specs kept for `key_count()`.
    specs: Vec<JoinKeySpec>,
    /// Keeps the `JITModule` code pages alive.
    _module_owner: Arc<JITModuleOwner>,
}

// SAFETY: JITModule code pages are read-only after finalisation;
// the module is protected by `Arc<JITModuleOwner>`.
unsafe impl Send for CompiledJoinKey {}
unsafe impl Sync for CompiledJoinKey {}

impl std::fmt::Debug for CompiledJoinKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledJoinKey")
            .field("key_count", &self.specs.len())
            .finish_non_exhaustive()
    }
}

impl CompiledJoinKey {
    /// Compare two rows using the compiled function.
    ///
    /// Returns `true` if all key columns match according to each column's comparison mode.
    ///
    /// # Safety
    ///
    /// `left` must have at least `max(spec.left_idx) + 1` elements and
    /// `right` must have at least `max(spec.right_idx) + 1` elements.
    pub fn compare(&self, left: &[f64], right: &[f64]) -> bool {
        let n = self.specs.len();
        // SAFETY:
        // - `fn_ptr` is valid because `_module_owner` is still alive (we hold it).
        // - `left.as_ptr()` and `right.as_ptr()` are valid for `n` reads
        //   when the caller obeys the documented precondition.
        // - The compiled function does not mutate either slice.
        unsafe { (self.fn_ptr)(left.as_ptr(), right.as_ptr(), n) == 1 }
    }

    /// The number of key columns this comparator was compiled for.
    pub fn key_count(&self) -> usize {
        self.specs.len()
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during JIT compilation of a join-key comparator.
#[derive(Debug, thiserror::Error)]
pub enum JoinCompilerError {
    /// No key specs were provided.
    #[error("join compiler requires at least one key spec")]
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

/// Compiles join-key comparators to native machine code via Cranelift.
///
/// Each call to [`compile`](JoinCompiler::compile) creates a fresh `JITModule`
/// so that the resulting [`CompiledJoinKey`] is independently owned and can be
/// dropped without affecting other compiled functions.
pub struct JoinCompiler;

impl Default for JoinCompiler {
    fn default() -> Self {
        JoinCompiler
    }
}

impl JoinCompiler {
    /// Create a new `JoinCompiler`.
    pub fn new() -> Self {
        JoinCompiler
    }

    /// Compile a multi-key join comparator.
    ///
    /// The resulting function returns `1` if **all** key columns match, `0` if any diverges.
    ///
    /// # Errors
    ///
    /// Returns [`JoinCompilerError::NoKeys`] if `specs` is empty.
    /// Returns [`JoinCompilerError::CodegenError`] on Cranelift failures.
    pub fn compile(&self, specs: &[JoinKeySpec]) -> Result<CompiledJoinKey, JoinCompilerError> {
        if specs.is_empty() {
            return Err(JoinCompilerError::NoKeys);
        }
        let module = build_jit_module()?;
        let (fn_ptr, module) = compile_join_fn(module, specs)?;
        let owner = Arc::new(JITModuleOwner::new(module));
        Ok(CompiledJoinKey {
            fn_ptr,
            specs: specs.to_vec(),
            _module_owner: owner,
        })
    }
}

// ---------------------------------------------------------------------------
// Cranelift module setup
// ---------------------------------------------------------------------------

fn build_jit_module() -> Result<JITModule, JoinCompilerError> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|e| JoinCompilerError::CodegenError(e.to_string()))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| JoinCompilerError::CodegenError(e.to_string()))?;
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| JoinCompilerError::CodegenError(e.to_string()))?;

    let flags = settings::Flags::new(flag_builder);
    let isa = cranelift_native::builder()
        .map_err(|e| JoinCompilerError::IsaInitError(e.to_string()))?
        .finish(flags)
        .map_err(|e| JoinCompilerError::IsaInitError(e.to_string()))?;

    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    Ok(JITModule::new(builder))
}

// ---------------------------------------------------------------------------
// Cranelift IR code generation
// ---------------------------------------------------------------------------

/// Compile a join-key comparator function into `module` and return the function
/// pointer together with the (now finalized) module.
fn compile_join_fn(
    mut module: JITModule,
    specs: &[JoinKeySpec],
) -> Result<(JoinKeyFn, JITModule), JoinCompilerError> {
    let ptr_type = module.isa().pointer_type();

    // Signature: fn(*const f64, *const f64, usize) -> i8
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(ptr_type)); // left ptr
    sig.params.push(AbiParam::new(ptr_type)); // right ptr
    sig.params.push(AbiParam::new(ptr_type)); // n_keys (unused at runtime, kept for ABI)
    sig.returns.push(AbiParam::new(types::I8));

    let func_id = module
        .declare_function("join_key_fn", Linkage::Local, &sig)
        .map_err(|e| JoinCompilerError::CodegenError(e.to_string()))?;

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
        // Third param (n_keys) is only for ABI match; suppress "unused" at the IR level.
        let _n_keys = builder.block_params(entry_block)[2];

        // Emit one comparison per key and AND them together.
        // `accumulator` starts as the i8 value `1` (all-match).
        let one_i8 = builder.ins().iconst(types::I8, 1);
        let mut accumulator = one_i8;

        for spec in specs {
            let key_ok = emit_key_comparison(&mut builder, spec, left_ptr, right_ptr)?;
            // Logical AND of i8 values (both are 0 or 1)
            accumulator = builder.ins().band(accumulator, key_ok);
        }

        builder.ins().return_(&[accumulator]);
        builder.finalize();

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| JoinCompilerError::CodegenError(format!("{e:?}")))?;
    }

    module
        .finalize_definitions()
        .map_err(|e| JoinCompilerError::CodegenError(format!("finalize_definitions: {e:?}")))?;

    // SAFETY: The function was just defined and finalized above; the pointer is valid.
    let raw_ptr = module.get_finalized_function(func_id);
    // SAFETY: We built the function with exactly this signature.
    let fn_ptr: JoinKeyFn = unsafe { std::mem::transmute(raw_ptr) };

    Ok((fn_ptr, module))
}

/// Emit a single key-column comparison and return an `i8` value (1 = match, 0 = no match).
fn emit_key_comparison(
    builder: &mut FunctionBuilder<'_>,
    spec: &JoinKeySpec,
    left_ptr: cranelift_codegen::ir::Value,
    right_ptr: cranelift_codegen::ir::Value,
) -> Result<cranelift_codegen::ir::Value, JoinCompilerError> {
    let left_offset = byte_offset(spec.left_idx)?;
    let right_offset = byte_offset(spec.right_idx)?;

    // Load left[left_idx] and right[right_idx] as f64.
    // SAFETY comment for generated IR: caller guarantees indices are in-bounds.
    let lv = builder
        .ins()
        .load(types::F64, MemFlags::trusted(), left_ptr, left_offset);
    let rv = builder
        .ins()
        .load(types::F64, MemFlags::trusted(), right_ptr, right_offset);

    if spec.numeric_epsilon {
        // |lv - rv| < 1e-9
        // Note: if either operand is NaN, fsub returns NaN; fabs(NaN) is NaN;
        // fcmp LessThan(NaN, eps) = false → correctly returns 0 (not equal).
        let diff = builder.ins().fsub(lv, rv);
        let abs_diff = builder.ins().fabs(diff);
        let eps = builder.ins().f64const(1e-9);
        // fcmp returns I8 (0 or 1) in modern Cranelift — no bint needed.
        let cmp = builder.ins().fcmp(FloatCC::LessThan, abs_diff, eps);
        Ok(cmp)
    } else {
        // Exact bit equality: bitcast both to i64 then icmp Equal.
        // Under this mode two NaN values with identical bit patterns compare as equal,
        // because icmp treats them as plain integers — documented behaviour.
        let li = builder.ins().bitcast(types::I64, MemFlags::new(), lv);
        let ri = builder.ins().bitcast(types::I64, MemFlags::new(), rv);
        // icmp returns I8 (0 or 1) in modern Cranelift.
        let cmp = builder.ins().icmp(IntCC::Equal, li, ri);
        Ok(cmp)
    }
}

/// Convert a column index to a `i32` byte offset for Cranelift `load`.
///
/// Returns an error if the offset overflows `i32` (requires >268 million columns).
fn byte_offset(idx: usize) -> Result<i32, JoinCompilerError> {
    let byte = idx.checked_mul(std::mem::size_of::<f64>()).ok_or_else(|| {
        JoinCompilerError::CodegenError(format!("column index {idx} overflows byte offset"))
    })?;
    i32::try_from(byte).map_err(|_| {
        JoinCompilerError::CodegenError(format!(
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

    fn compiler() -> JoinCompiler {
        JoinCompiler::new()
    }

    #[test]
    fn test_no_keys_error() {
        let result = compiler().compile(&[]);
        assert!(matches!(result, Err(JoinCompilerError::NoKeys)));
    }

    #[test]
    fn test_byte_offset_zero() {
        assert_eq!(byte_offset(0).expect("ok"), 0);
    }

    #[test]
    fn test_byte_offset_one() {
        assert_eq!(byte_offset(1).expect("ok"), 8);
    }
}
