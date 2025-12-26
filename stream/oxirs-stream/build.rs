// Build script for generating Sparkplug B protobuf code

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

        let mut config = prost_build::Config::new();
        config.out_dir(&out_dir);
        config.protoc_arg("--experimental_allow_proto3_optional");

        config
            .compile_protos(&[proto_path], &[include_path])
            .expect("Failed to compile Sparkplug B protobuf");

        println!("cargo:rerun-if-changed=proto/sparkplug_b.proto");
    }
}
