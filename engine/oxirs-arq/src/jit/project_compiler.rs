//! Cranelift-based JIT compiler for SPARQL PROJECT column extraction (JIT phase d).
//!
//! Produces a native function with signature:
//! ```text
//! fn(src_ptr: *const f64, src_len: usize, dst_ptr: *mut f64, dst_len: usize) -> i8
//! ```
//! Returns `1` on success, `0` if either the source or destination slice is too short
//! to satisfy the requested column mapping.
//!
//! # Operator semantics
//!
//! The PROJECT operator selects and reorders columns from a source row into a
//! destination row.  Each [`ProjectSpec`] entry contributes one `f64` value
//! to the output by naming a source-row column index.  Output order follows
//! the order of the `specs` slice.
//!
//! # Safety
//!
//! The compiled function is called via an `unsafe extern "C" fn` pointer.
//! Callers must guarantee:
//! - `src_ptr` is a valid, aligned, non-null pointer to at least `src_len` `f64` values.
//! - `dst_ptr` is a valid, aligned, non-null pointer to at least `dst_len` `f64` values.
//! - Both pointers remain valid for the duration of the call.

use std::sync::Arc;

use cranelift_codegen::ir::{
    condcodes::IntCC, types, AbiParam, InstBuilder, MemFlagsData, Signature, Value,
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

/// Configuration for one output column in a PROJECT operation.
#[derive(Debug, Clone, Copy)]
pub struct ProjectSpec {
    /// Index into the source row `f64` slice that this output slot reads from.
    pub src_idx: usize,
}

/// The C-ABI function pointer produced by [`ProjectCompiler::compile`].
///
/// # Safety
///
/// - `src_ptr` must point to at least `src_len` `f64` values.
/// - `dst_ptr` must point to at least `dst_len` writable `f64` values.
/// - `dst_len` must be `≥ specs.len()`.
type ProjectFn = unsafe extern "C" fn(*const f64, usize, *mut f64, usize) -> i8;

/// A compiled PROJECT-operator column extractor produced by [`ProjectCompiler::compile`].
///
/// # Ownership / safety invariant
///
/// `fn_ptr` is only valid while `_module_owner` is alive.
pub struct CompiledProject {
    /// JIT-compiled function pointer.
    fn_ptr: ProjectFn,
    /// Kept for [`output_width`](CompiledProject::output_width).
    specs: Vec<ProjectSpec>,
    /// Keeps the `JITModule` code pages alive.
    _module_owner: Arc<JITModuleOwner>,
}

// SAFETY: JITModule code pages are read-only after finalisation;
// the module is protected by `Arc<JITModuleOwner>`.
unsafe impl Send for CompiledProject {}
unsafe impl Sync for CompiledProject {}

impl std::fmt::Debug for CompiledProject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledProject")
            .field("output_width", &self.specs.len())
            .finish_non_exhaustive()
    }
}

impl CompiledProject {
    /// Number of output columns this projector produces.
    pub fn output_width(&self) -> usize {
        self.specs.len()
    }

    /// Extract projected columns from `src` into `dst`.
    ///
    /// `dst` is resized to [`output_width`](Self::output_width) before the call.
    /// Returns `false` if `src` is too short to satisfy any required column index.
    ///
    /// # Safety invariant
    ///
    /// The compiled function is invoked via a raw function pointer that is valid
    /// because `_module_owner` keeps the `JITModule` pages alive for the lifetime
    /// of `self`.
    pub fn extract(&self, src: &[f64], dst: &mut Vec<f64>) -> bool {
        let out_len = self.specs.len();
        dst.resize(out_len, 0.0);
        // SAFETY:
        // - `fn_ptr` is valid because `_module_owner` is still alive (we hold it in self).
        // - `src.as_ptr()` is a valid, aligned pointer to `src.len()` f64 values.
        // - `dst.as_mut_ptr()` is a valid, aligned, writable pointer to `dst.len()` f64 values.
        // - Both slices remain valid for the duration of this call.
        let result = unsafe { (self.fn_ptr)(src.as_ptr(), src.len(), dst.as_mut_ptr(), dst.len()) };
        result == 1
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during JIT compilation of a PROJECT extractor.
#[derive(Debug, thiserror::Error)]
pub enum ProjectCompilerError {
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

/// Compiles PROJECT column-extraction operations to native machine code via Cranelift.
///
/// Each call to [`compile`](ProjectCompiler::compile) creates a fresh `JITModule`
/// so that the resulting [`CompiledProject`] is independently owned and can be dropped
/// without affecting other compiled functions.
pub struct ProjectCompiler;

impl Default for ProjectCompiler {
    fn default() -> Self {
        ProjectCompiler
    }
}

impl ProjectCompiler {
    /// Create a new `ProjectCompiler`.
    pub fn new() -> Self {
        ProjectCompiler
    }

    /// Compile a column-extraction function for the given `specs`.
    ///
    /// If `specs` is empty the compiled function is a no-op that always returns success.
    pub fn compile(
        &mut self,
        specs: &[ProjectSpec],
    ) -> Result<CompiledProject, ProjectCompilerError> {
        let module = build_jit_module()?;
        let (fn_ptr, module) = compile_project_fn(module, specs)?;
        let owner = Arc::new(JITModuleOwner::new(module));
        Ok(CompiledProject {
            fn_ptr,
            specs: specs.to_vec(),
            _module_owner: owner,
        })
    }
}

// ---------------------------------------------------------------------------
// Cranelift module setup
// ---------------------------------------------------------------------------

fn build_jit_module() -> Result<JITModule, ProjectCompilerError> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|e| ProjectCompilerError::CodegenError(e.to_string()))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| ProjectCompilerError::CodegenError(e.to_string()))?;
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| ProjectCompilerError::CodegenError(e.to_string()))?;

    let flags = settings::Flags::new(flag_builder);
    let isa = cranelift_native::builder()
        .map_err(|e| ProjectCompilerError::IsaInitError(e.to_string()))?
        .finish(flags)
        .map_err(|e| ProjectCompilerError::IsaInitError(e.to_string()))?;

    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    Ok(JITModule::new(builder))
}

// ---------------------------------------------------------------------------
// Cranelift IR code generation
// ---------------------------------------------------------------------------

/// Compile a project function into `module` and return the function pointer
/// together with the (now finalized) module.
fn compile_project_fn(
    mut module: JITModule,
    specs: &[ProjectSpec],
) -> Result<(ProjectFn, JITModule), ProjectCompilerError> {
    // Use the host pointer width for all params (usize / pointer types)
    let ptr_type = module.isa().pointer_type();

    // Signature: fn(*const f64, usize, *mut f64, usize) -> i8
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(ptr_type)); // src_ptr: *const f64
    sig.params.push(AbiParam::new(ptr_type)); // src_len: usize
    sig.params.push(AbiParam::new(ptr_type)); // dst_ptr: *mut f64
    sig.params.push(AbiParam::new(ptr_type)); // dst_len: usize
    sig.returns.push(AbiParam::new(types::I8)); // return: i8 (1 = ok, 0 = bounds error)

    let func_id = module
        .declare_function("project_fn", Linkage::Local, &sig)
        .map_err(|e| ProjectCompilerError::LinkageError(e.to_string()))?;

    {
        let mut ctx = module.make_context();
        ctx.func.signature = sig.clone();

        let mut fn_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

        if specs.is_empty() {
            // Fast path: no columns to project — emit a single-block return-1 function.
            emit_trivial_success(&mut builder, ptr_type);
        } else {
            emit_project_body(&mut builder, specs, ptr_type)?;
        }

        builder.finalize();

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| ProjectCompilerError::CodegenError(format!("{e:?}")))?;
    }

    module
        .finalize_definitions()
        .map_err(|e| ProjectCompilerError::CodegenError(format!("finalize_definitions: {e:?}")))?;

    // SAFETY: The function was just defined and finalized above; the pointer is valid.
    let raw_ptr = module.get_finalized_function(func_id);
    // SAFETY: We built the function with exactly this signature.
    let fn_ptr: ProjectFn = unsafe { std::mem::transmute(raw_ptr) };

    Ok((fn_ptr, module))
}

/// Emit a trivial single-block function that immediately returns 1 (success).
///
/// Used when there are no specs to process.
fn emit_trivial_success(builder: &mut FunctionBuilder<'_>, ptr_type: types::Type) {
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Consume all parameters to satisfy the ABI (4 params for project_fn).
    // This suppresses any "unused parameter" IR warnings from Cranelift.
    let _src_ptr: Value = builder.block_params(entry_block)[0];
    let _src_len: Value = builder.block_params(entry_block)[1];
    let _dst_ptr: Value = builder.block_params(entry_block)[2];
    let _dst_len: Value = builder.block_params(entry_block)[3];

    // Suppress unused type warning
    let _ = ptr_type;

    let one = builder.ins().iconst(types::I8, 1);
    builder.ins().return_(&[one]);
}

/// Emit the multi-block project body with bounds checks.
///
/// Control flow:
/// ```text
///   entry → [bounds_fail if src_len <= max_src_idx || dst_len < specs.len()]
///   entry → body → return 1
///   bounds_fail → return 0
/// ```
fn emit_project_body(
    builder: &mut FunctionBuilder<'_>,
    specs: &[ProjectSpec],
    ptr_type: types::Type,
) -> Result<(), ProjectCompilerError> {
    // Compute the maximum source index needed across all specs.
    // SAFETY: specs is non-empty at this call site.
    let max_src_idx = specs.iter().map(|s| s.src_idx).max().unwrap_or(0);

    // Create blocks: entry, bounds_fail, body.
    let entry_block = builder.create_block();
    let bounds_fail_block = builder.create_block();
    let body_block = builder.create_block();

    // ---- Entry block ----
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let src_ptr = builder.block_params(entry_block)[0];
    let src_len = builder.block_params(entry_block)[1];
    let dst_ptr = builder.block_params(entry_block)[2];
    let dst_len = builder.block_params(entry_block)[3];

    // Bounds check 1: src_len <= max_src_idx  (unsigned ≤, i.e., src cannot hold the largest src_idx)
    // We need src_len > max_src_idx, i.e., src_len >= max_src_idx + 1.
    // Failure condition: src_len <= max_src_idx (UnsignedLessThanOrEqual).
    let max_src_val = builder.ins().iconst(ptr_type, max_src_idx as i64);
    let src_too_short = builder
        .ins()
        .icmp(IntCC::UnsignedLessThanOrEqual, src_len, max_src_val);

    // Bounds check 2: dst_len < specs.len()
    // Failure condition: dst_len < specs.len() (UnsignedLessThan).
    let dst_need = builder.ins().iconst(ptr_type, specs.len() as i64);
    let dst_too_short = builder
        .ins()
        .icmp(IntCC::UnsignedLessThan, dst_len, dst_need);

    // Combine: either condition → fail
    let any_fail = builder.ins().bor(src_too_short, dst_too_short);

    // brif: if any_fail → bounds_fail_block, else → body_block
    builder
        .ins()
        .brif(any_fail, bounds_fail_block, &[], body_block, &[]);

    // ---- Bounds-fail block ----
    builder.switch_to_block(bounds_fail_block);
    builder.seal_block(bounds_fail_block);
    let zero_i8 = builder.ins().iconst(types::I8, 0);
    builder.ins().return_(&[zero_i8]);

    // ---- Body block ----
    builder.switch_to_block(body_block);
    builder.seal_block(body_block);

    // For each spec: load src[src_idx], store to dst[dst_i].
    for (dst_i, spec) in specs.iter().enumerate() {
        // Source load: src_ptr + src_idx * 8
        let src_offset = builder
            .ins()
            .iconst(ptr_type, (spec.src_idx * std::mem::size_of::<f64>()) as i64);
        let src_addr = builder.ins().iadd(src_ptr, src_offset);
        // SAFETY comment for generated IR: bounds were verified above; load is safe.
        // cranelift 0.133: the old `MemFlags::new()` value type is now `MemFlagsData`
        // (InstBuilder interns it into the DFG's MemFlagsSet internally).
        let val = builder
            .ins()
            .load(types::F64, MemFlagsData::new(), src_addr, 0);

        // Destination store: dst_ptr + dst_i * 8
        let dst_offset = builder
            .ins()
            .iconst(ptr_type, (dst_i * std::mem::size_of::<f64>()) as i64);
        let dst_addr = builder.ins().iadd(dst_ptr, dst_offset);
        // SAFETY comment for generated IR: bounds were verified above; store is safe.
        builder.ins().store(MemFlagsData::new(), val, dst_addr, 0);
    }

    let one_i8 = builder.ins().iconst(types::I8, 1);
    builder.ins().return_(&[one_i8]);

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests (inline)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn compiler() -> ProjectCompiler {
        ProjectCompiler::new()
    }

    #[test]
    fn test_empty_specs_compiles_and_succeeds() {
        let cp = compiler().compile(&[]).expect("compile ok");
        assert_eq!(cp.output_width(), 0);
        let src = [1.0f64, 2.0, 3.0];
        let mut dst = Vec::new();
        assert!(cp.extract(&src, &mut dst));
        assert!(dst.is_empty());
    }

    #[test]
    fn test_single_col_extract() {
        let specs = [ProjectSpec { src_idx: 1 }];
        let cp = compiler().compile(&specs).expect("compile ok");
        assert_eq!(cp.output_width(), 1);
        let src = [10.0f64, 20.0, 30.0];
        let mut dst = Vec::new();
        assert!(cp.extract(&src, &mut dst));
        assert_eq!(dst, vec![20.0]);
    }

    #[test]
    fn test_reorder_extract() {
        let specs = [
            ProjectSpec { src_idx: 2 },
            ProjectSpec { src_idx: 0 },
            ProjectSpec { src_idx: 1 },
        ];
        let cp = compiler().compile(&specs).expect("compile ok");
        let src = [1.0f64, 2.0, 3.0];
        let mut dst = Vec::new();
        assert!(cp.extract(&src, &mut dst));
        assert_eq!(dst, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_src_bounds_fail() {
        let specs = [ProjectSpec { src_idx: 5 }];
        let cp = compiler().compile(&specs).expect("compile ok");
        let src = [1.0f64, 2.0]; // only 2 elements; idx 5 is out of bounds
        let mut dst = Vec::new();
        assert!(!cp.extract(&src, &mut dst));
    }
}
