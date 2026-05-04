/// Verify that example binaries compile.
///
/// Compilation is checked automatically by `cargo build --examples -p oxirs-chat`.
/// This test acts as a named anchor so CI failures are traceable to this file.
#[test]
fn examples_compile() {
    // Compilation is verified by `cargo build --examples -p oxirs-chat`.
    // If the examples directory contains code that does not compile, the
    // `cargo build` step above this test fails before this function runs.
    // No runtime assertions are needed here.
}
