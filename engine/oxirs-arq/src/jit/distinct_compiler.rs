//! Cranelift-based JIT compiler for DISTINCT deduplication via FNV-1a hash (JIT phase d).
//!
//! Produces a native function with signature:
//! ```text
//! fn(ptr: *const f64, len: usize) -> i64
//! ```
//! Returns the FNV-1a hash of the selected key columns, treating each `f64` as its raw
//! 64-bit integer bit pattern (via bitcast to `i64`).  Returns `0` if any required column
//! index is out-of-bounds.
//!
//! # Hash semantics
//!
//! The hash implements FNV-1a over the selected column values in spec order:
//! ```text
//! hash = FNV_OFFSET_BASIS
//! for each spec:
//!     bits = bitcast(row[spec.col_idx] as f64 → i64)   // NaN-safe: same bits = same hash
//!     hash = (hash XOR bits) * FNV_PRIME
//! ```
//!
//! Two distinct `f64` values that differ only in NaN bit patterns will produce different
//! hashes.  The bitcast approach is intentional: SPARQL DISTINCT semantics compare by
//! term identity, not numeric equality, so distinct NaN payloads should hash differently.
//!
//! # Safety
//!
//! The compiled function is called via an `unsafe extern "C" fn` pointer.
//! Callers must guarantee:
//! - `ptr` is a valid, aligned, non-null pointer to at least `len` `f64` values.
//! - The pointer remains valid for the duration of the call.

use std::sync::Arc;

use cranelift_codegen::ir::{
    condcodes::IntCC, types, AbiParam, InstBuilder, MemFlagsData, Signature,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use super::filter_compiler::JITModuleOwner;

// ---------------------------------------------------------------------------
// FNV-1a constants
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit offset basis: `0xcbf29ce484222325` interpreted as a signed `i64`.
const FNV_OFFSET_BASIS: i64 = -3_750_763_034_362_895_579_i64;

/// FNV-1a 64-bit prime: `0x100000001b3`.
const FNV_PRIME: i64 = 1_099_511_628_211_i64;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for one key column in a DISTINCT deduplication hash.
#[derive(Debug, Clone, Copy)]
pub struct DistinctKeySpec {
    /// Column index in the `f64` row slice to include in the hash.
    pub col_idx: usize,
}

/// The C-ABI function pointer produced by [`DistinctCompiler::compile`].
///
/// # Safety
///
/// `ptr` must point to at least `len` `f64` values.  `len` must be at least
/// `max(spec.col_idx) + 1` for the result to be meaningful.
type DistinctHashFn = unsafe extern "C" fn(*const f64, usize) -> i64;

/// A compiled DISTINCT-deduplication hasher produced by [`DistinctCompiler::compile`].
///
/// # Ownership / safety invariant
///
/// `fn_ptr` is only valid while `_module_owner` is alive.
pub struct CompiledDistinct {
    /// JIT-compiled function pointer.
    fn_ptr: DistinctHashFn,
    /// Kept for bounds checking in [`hash_key`](CompiledDistinct::hash_key) and
    /// for [`key_count`](CompiledDistinct::key_count).
    specs: Vec<DistinctKeySpec>,
    /// Keeps the `JITModule` code pages alive.
    _module_owner: Arc<JITModuleOwner>,
}

// SAFETY: JITModule code pages are read-only after finalisation;
// the module is protected by `Arc<JITModuleOwner>`.
unsafe impl Send for CompiledDistinct {}
unsafe impl Sync for CompiledDistinct {}

impl std::fmt::Debug for CompiledDistinct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledDistinct")
            .field("key_count", &self.specs.len())
            .finish_non_exhaustive()
    }
}

impl CompiledDistinct {
    /// Hash the selected columns of a row.
    ///
    /// Returns `None` if any required column index is out-of-bounds (i.e.
    /// `row.len() ≤ max(col_idx)`).
    ///
    /// # Safety invariant
    ///
    /// The compiled function is invoked via a raw function pointer that is valid
    /// because `_module_owner` keeps the `JITModule` pages alive for the lifetime
    /// of `self`.
    pub fn hash_key(&self, row: &[f64]) -> Option<i64> {
        // SAFETY:
        // - `fn_ptr` is valid because `_module_owner` is still alive (we hold it in self).
        // - `row.as_ptr()` is a valid, aligned pointer to `row.len()` f64 values.
        // - The pointer remains valid for the duration of this call.
        let result = unsafe { (self.fn_ptr)(row.as_ptr(), row.len()) };

        // For the empty-specs case the IR directly returns FNV_OFFSET_BASIS without a
        // bounds check, and that is always valid.
        if self.specs.is_empty() {
            return Some(result);
        }

        // The IR returns 0 on bounds error.  We re-check in Rust to convert "error" to None
        // rather than returning a potentially misleading hash of 0.
        let max_col = self.specs.iter().map(|s| s.col_idx).max().unwrap_or(0);
        if row.len() <= max_col {
            return None;
        }

        Some(result)
    }

    /// Number of key columns included in the hash.
    pub fn key_count(&self) -> usize {
        self.specs.len()
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during JIT compilation of a DISTINCT hash function.
#[derive(Debug, thiserror::Error)]
pub enum DistinctCompilerError {
    /// Cranelift reported a codegen or linkage error.
    #[error("JIT codegen error: {0}")]
    CodegenError(String),
    /// ISA builder failed to initialise.
    #[error("JIT ISA init error: {0}")]
    IsaInitError(String),
    /// Function declaration or linkage failed.
    #[error("JIT linkage error: {0}")]
    LinkageError(String),
}

// ---------------------------------------------------------------------------
// Compiler (ZST — new JITModule per compile call)
// ---------------------------------------------------------------------------

/// Compiles DISTINCT hash functions to native machine code via Cranelift.
///
/// Each call to [`compile`](DistinctCompiler::compile) creates a fresh `JITModule`
/// so that the resulting [`CompiledDistinct`] is independently owned and can be dropped
/// without affecting other compiled functions.
pub struct DistinctCompiler;

impl Default for DistinctCompiler {
    fn default() -> Self {
        DistinctCompiler
    }
}

impl DistinctCompiler {
    /// Create a new `DistinctCompiler`.
    pub fn new() -> Self {
        DistinctCompiler
    }

    /// Compile a DISTINCT hash function for the given key column specs.
    ///
    /// If `specs` is empty, the compiled function returns `FNV_OFFSET_BASIS` unconditionally.
    pub fn compile(
        &mut self,
        specs: &[DistinctKeySpec],
    ) -> Result<CompiledDistinct, DistinctCompilerError> {
        let module = build_jit_module()?;
        let (fn_ptr, module) = compile_distinct_fn(module, specs)?;
        let owner = Arc::new(JITModuleOwner::new(module));
        Ok(CompiledDistinct {
            fn_ptr,
            specs: specs.to_vec(),
            _module_owner: owner,
        })
    }
}

// ---------------------------------------------------------------------------
// Cranelift module setup
// ---------------------------------------------------------------------------

fn build_jit_module() -> Result<JITModule, DistinctCompilerError> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|e| DistinctCompilerError::CodegenError(e.to_string()))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| DistinctCompilerError::CodegenError(e.to_string()))?;
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| DistinctCompilerError::CodegenError(e.to_string()))?;

    let flags = settings::Flags::new(flag_builder);
    let isa = cranelift_native::builder()
        .map_err(|e| DistinctCompilerError::IsaInitError(e.to_string()))?
        .finish(flags)
        .map_err(|e| DistinctCompilerError::IsaInitError(e.to_string()))?;

    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    Ok(JITModule::new(builder))
}

// ---------------------------------------------------------------------------
// Cranelift IR code generation
// ---------------------------------------------------------------------------

/// Compile a DISTINCT hash function into `module` and return the function pointer
/// together with the (now finalized) module.
fn compile_distinct_fn(
    mut module: JITModule,
    specs: &[DistinctKeySpec],
) -> Result<(DistinctHashFn, JITModule), DistinctCompilerError> {
    let ptr_type = module.isa().pointer_type();

    // Signature: fn(*const f64, usize) -> i64
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(ptr_type)); // ptr: *const f64
    sig.params.push(AbiParam::new(ptr_type)); // len: usize
    sig.returns.push(AbiParam::new(types::I64)); // return: i64 (hash or 0 on error)

    let func_id = module
        .declare_function("distinct_hash_fn", Linkage::Local, &sig)
        .map_err(|e| DistinctCompilerError::LinkageError(e.to_string()))?;

    {
        let mut ctx = module.make_context();
        ctx.func.signature = sig.clone();

        let mut fn_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

        if specs.is_empty() {
            // Fast path: no columns — return FNV_OFFSET_BASIS unconditionally.
            emit_constant_hash(&mut builder, ptr_type, FNV_OFFSET_BASIS);
        } else {
            emit_distinct_body(&mut builder, specs, ptr_type)?;
        }

        builder.finalize();

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| DistinctCompilerError::CodegenError(format!("{e:?}")))?;
    }

    module
        .finalize_definitions()
        .map_err(|e| DistinctCompilerError::CodegenError(format!("finalize_definitions: {e:?}")))?;

    // SAFETY: The function was just defined and finalized above; the pointer is valid.
    let raw_ptr = module.get_finalized_function(func_id);
    // SAFETY: We built the function with exactly this signature.
    let fn_ptr: DistinctHashFn = unsafe { std::mem::transmute(raw_ptr) };

    Ok((fn_ptr, module))
}

/// Emit a single-block function that immediately returns the given constant `i64`.
fn emit_constant_hash(builder: &mut FunctionBuilder<'_>, ptr_type: types::Type, constant: i64) {
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Consume params to satisfy the ABI.
    let _ptr: cranelift_codegen::ir::Value = builder.block_params(entry_block)[0];
    let _len: cranelift_codegen::ir::Value = builder.block_params(entry_block)[1];

    // Suppress unused type warning
    let _ = ptr_type;

    let val = builder.ins().iconst(types::I64, constant);
    builder.ins().return_(&[val]);
}

/// Emit the multi-block distinct hash body with a bounds check.
///
/// Control flow:
/// ```text
///   entry → [bounds_fail if len <= max_col_idx]
///   entry → body → return hash
///   bounds_fail → return 0
/// ```
fn emit_distinct_body(
    builder: &mut FunctionBuilder<'_>,
    specs: &[DistinctKeySpec],
    ptr_type: types::Type,
) -> Result<(), DistinctCompilerError> {
    // SAFETY: specs is non-empty at this call site.
    let max_col = specs.iter().map(|s| s.col_idx).max().unwrap_or(0);

    // Create blocks: entry, bounds_fail, body.
    let entry_block = builder.create_block();
    let bounds_fail_block = builder.create_block();
    let body_block = builder.create_block();

    // ---- Entry block ----
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let ptr = builder.block_params(entry_block)[0];
    let len = builder.block_params(entry_block)[1];

    // Bounds check: len <= max_col (UnsignedLessThanOrEqual)
    // We need len > max_col, i.e., len >= max_col + 1.
    let max_col_val = builder.ins().iconst(ptr_type, max_col as i64);
    let out_of_bounds = builder
        .ins()
        .icmp(IntCC::UnsignedLessThanOrEqual, len, max_col_val);

    builder
        .ins()
        .brif(out_of_bounds, bounds_fail_block, &[], body_block, &[]);

    // ---- Bounds-fail block ----
    builder.switch_to_block(bounds_fail_block);
    builder.seal_block(bounds_fail_block);
    let zero = builder.ins().iconst(types::I64, 0);
    builder.ins().return_(&[zero]);

    // ---- Body block: FNV-1a hash computation ----
    builder.switch_to_block(body_block);
    builder.seal_block(body_block);

    // Initialize hash with the FNV offset basis.
    let mut hash = builder.ins().iconst(types::I64, FNV_OFFSET_BASIS);

    for spec in specs {
        // Load f64 at ptr + col_idx * 8.
        let offset = builder
            .ins()
            .iconst(ptr_type, (spec.col_idx * std::mem::size_of::<f64>()) as i64);
        let addr = builder.ins().iadd(ptr, offset);
        // SAFETY comment for generated IR: bounds were verified above; load is safe.
        // cranelift 0.133: the old `MemFlags::new()` value type is now `MemFlagsData`
        // (InstBuilder interns it into the DFG's MemFlagsSet internally).
        let f64_val = builder.ins().load(types::F64, MemFlagsData::new(), addr, 0);

        // Bitcast f64 → i64 to treat the bit pattern as an integer.
        // This is the standard FNV approach for floating-point hashing.
        let bits = builder
            .ins()
            .bitcast(types::I64, MemFlagsData::new(), f64_val);

        // FNV-1a step: hash = (hash XOR bits) * FNV_PRIME
        hash = builder.ins().bxor(hash, bits);
        let prime = builder.ins().iconst(types::I64, FNV_PRIME);
        hash = builder.ins().imul(hash, prime);
    }

    builder.ins().return_(&[hash]);

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests (inline)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn compiler() -> DistinctCompiler {
        DistinctCompiler::new()
    }

    #[test]
    fn test_empty_specs_returns_fnv_basis() {
        let cd = compiler().compile(&[]).expect("compile ok");
        let row = [1.0f64, 2.0];
        let h = cd.hash_key(&row).expect("should return Some");
        assert_eq!(h, FNV_OFFSET_BASIS);
    }

    #[test]
    fn test_same_row_same_hash() {
        let specs = [
            DistinctKeySpec { col_idx: 0 },
            DistinctKeySpec { col_idx: 1 },
        ];
        let cd = compiler().compile(&specs).expect("compile ok");
        let row = [1.0f64, 2.0];
        let h1 = cd.hash_key(&row).expect("ok");
        let h2 = cd.hash_key(&row).expect("ok");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_different_rows_different_hash() {
        let specs = [DistinctKeySpec { col_idx: 0 }];
        let cd = compiler().compile(&specs).expect("compile ok");
        let h1 = cd.hash_key(&[1.0f64]).expect("ok");
        let h2 = cd.hash_key(&[2.0f64]).expect("ok");
        assert_ne!(
            h1, h2,
            "distinct f64 values should (almost certainly) hash differently"
        );
    }

    #[test]
    fn test_bounds_check_fails() {
        let specs = [DistinctKeySpec { col_idx: 5 }];
        let cd = compiler().compile(&specs).expect("compile ok");
        let row = [1.0f64, 2.0]; // only 2 elements; col_idx 5 is out of bounds
        assert!(cd.hash_key(&row).is_none());
    }
}
