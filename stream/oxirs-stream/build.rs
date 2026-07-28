// Build script for generating Sparkplug B protobuf code.
//
// Uses `oxiproto-build` (COOLJAPAN OxiProto) rather than `prost-build`: the
// latter shells out to a `protoc` binary that must be installed separately,
// which made `--all-features` builds fail on any machine without it. OxiProto
// parses `.proto` in-process with a pure-Rust parser and still emits
// prost-compatible types, so `sparkplug_b.rs` and its consumers are unchanged.

fn main() {
    #[cfg(feature = "sparkplug")]
    {
        use std::path::PathBuf;

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        let manifest_path = PathBuf::from(&manifest_dir);

        let proto_path = manifest_path.join("proto/sparkplug_b.proto");
        let include_path = manifest_path.join("proto");

        // Use OUT_DIR for generated code (standard cargo build output directory)
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");

        // No `--experimental_allow_proto3_optional` equivalent is needed: that
        // flag only existed to unlock proto3 `optional` on older protoc
        // releases, and OxiProto's parser emits the synthetic oneofs for
        // proto3 field presence natively.
        oxiproto_build::Builder::new()
            .out_dir(&out_dir)
            .compile(&[proto_path], &[include_path])
            .expect("Failed to compile Sparkplug B protobuf");

        println!("cargo:rerun-if-changed=proto/sparkplug_b.proto");
    }
}
