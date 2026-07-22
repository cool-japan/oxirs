//! OxiRS Fuseki Server Binary

use clap::Parser;
use oxirs_fuseki::Server;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "oxirs-fuseki")]
#[command(about = "OxiRS SPARQL server with Fuseki compatibility")]
struct Args {
    /// Server port
    #[arg(short, long, default_value = "3030")]
    port: u16,

    /// Server host
    #[arg(long, default_value = "localhost")]
    host: String,

    /// Dataset configuration file
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Dataset storage path
    #[arg(short, long)]
    dataset: Option<PathBuf>,

    /// Build a frozen mmap snapshot (`snapshot.oxsnap`) for the dataset and exit.
    ///
    /// Offline "bake" step for read-only deployments: reads
    /// `<dataset>/default.db/data.nq` and writes
    /// `<dataset>/default.db/snapshot.oxsnap` — exactly the file a later serve of
    /// the same `-d <dataset>` mmap-loads for a near-instant cold start (skipping
    /// the N-Quads re-parse). Requires `-d/--dataset`; performs no serving.
    #[arg(long)]
    build_snapshot: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use oxirs_fuseki::config::ServerConfig;

    // Install the Pure Rust crypto provider as the process default (idempotent).
    let _ = rustls::crypto::CryptoProvider::install_default((*oxitls::pure_provider()).clone());

    let args = Args::parse();

    // Load the configuration (from file when supplied, otherwise defaults) BEFORE
    // initializing logging, so `logging.{format,output,level,file_config}` are
    // actually applied instead of the previous hardcoded stdout-text default.
    let config = match &args.config {
        Some(path) => ServerConfig::from_file(path.clone())?,
        None => ServerConfig::default(),
    };

    // Initialize tracing from the logging config. A misconfigured file sink is a
    // fail-loud error rather than a silent downgrade to stdout.
    oxirs_fuseki::logging::init(&config.logging)
        .map_err(|e| -> Box<dyn std::error::Error> { e })?;

    // Offline snapshot bake: build `<dataset>/default.db/snapshot.oxsnap` and exit
    // without starting the server. The `default.db` suffix mirrors how the server
    // opens its default dataset (`Store::open` → `RdfStore::open(path/"default.db")`),
    // so the snapshot lands precisely where a later serve mmap-loads it.
    if args.build_snapshot {
        use oxirs_core::RdfStore;
        let dataset = args.dataset.ok_or_else(|| {
            Box::<dyn std::error::Error>::from(
                "--build-snapshot requires -d/--dataset <dir> (the same dir you serve with -d)",
            )
        })?;
        let db_dir = dataset.join("default.db");
        let snapshot_path = RdfStore::build_snapshot(&db_dir)?;
        println!("snapshot written: {}", snapshot_path.display());
        return Ok(());
    }

    // Thread the FULL config (datasets, per-dataset `read_only`, security, …) into
    // the builder when a config file was given, so it reaches `AppState`; the CLI
    // host/port win over the resolved socket address. Without a config file, fall
    // back to CLI host/port with a default config.
    let mut builder = Server::builder();
    if args.config.is_some() {
        builder = builder
            .host(config.server.host.clone())
            .port(config.server.port)
            .config(config);
    } else {
        builder = builder.host(args.host).port(args.port);
    }

    if let Some(dataset_path) = args.dataset {
        builder = builder.dataset_path(dataset_path.to_string_lossy());
    }

    let server = builder.build().await?;
    server.run().await?;

    Ok(())
}
