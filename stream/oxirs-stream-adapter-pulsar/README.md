# oxirs-stream-adapter-pulsar

Quarantined Apache Pulsar backend adapter for [`oxirs-stream`](../oxirs-stream).

`publish = false` — this crate never ships to crates.io. It exists so the live
Pulsar backend keeps working while its C FFI dependencies stay off
`oxirs-stream`'s published Pure-Rust surface, as required by the COOLJAPAN Pure
Rust Policy v2 (purity is measured on the full `--all-features` dependency
closure). See the module docs in `src/lib.rs` for the full rationale.

## Build prerequisite: a `protoc`, or the OxiProto shim

This crate depends on [`pulsar`](https://docs.rs/pulsar) 6.x, whose `build.rs`
calls `prost_build::compile_protos` on `PulsarApi.proto`. `prost-build` shells
out to a `protoc` executable, so **building this crate fails without one**:

```
error: failed to run custom build command for `pulsar v6.8.0`
  Could not find `protoc`. If `protoc` is installed, try setting the `PROTOC`
  environment variable to the path of the `protoc` binary.
```

`pulsar` is a third-party crate, so this cannot be fixed the way
`oxirs-stream`'s own Sparkplug B codegen was (that build script calls
`oxiproto-build` directly and needs no external binary). What *can* be done is
satisfy `PROTOC` with a pure-Rust stand-in.

### Recommended: `oxiproto-protoc`

[`oxiproto-cli`](https://crates.io/crates/oxiproto-cli) ships a second binary,
`oxiproto-protoc`, that implements the descriptor-set-generating subset of
`protoc`'s command line on top of OxiProto's pure-Rust `.proto` parser. Point
`PROTOC` at it and `pulsar` builds with no C++ toolchain involved:

```powershell
# Windows
cargo install oxiproto-cli
$env:PROTOC = "$env:USERPROFILE\.cargo\bin\oxiproto-protoc.exe"
cargo build -p oxirs-stream-adapter-pulsar
```

```sh
# Linux / macOS
cargo install oxiproto-cli
export PROTOC=$(which oxiproto-protoc)
cargo build -p oxirs-stream-adapter-pulsar
```

This is verified to build `pulsar` 6.8.0 end to end on Windows with no `protoc`
installed.

### Alternative: a real `protoc`

Installing upstream `protoc` (`winget install protobuf`, `apt install
protobuf-compiler`, `brew install protobuf`, …) also works and needs no `PROTOC`
setting once it is on `PATH`. The `pulsar` crate additionally offers a
`protobuf-src` feature that compiles protobuf from C++ source, but that
reintroduces a C++ toolchain requirement and is not recommended here.

## Usage

`PulsarProducer` / `PulsarConsumer` are constructed from
`oxirs_stream::StreamConfig` with a `StreamBackendType::Pulsar` selector and
exchange `oxirs_stream::StreamEvent` values, so they are drop-in compatible with
the published `oxirs-stream` types.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `pulsar` | on | Enables the real `pulsar`-backed producer/consumer. Turning it off drops the `pulsar` dependency (and with it the `protoc` prerequisite). |
