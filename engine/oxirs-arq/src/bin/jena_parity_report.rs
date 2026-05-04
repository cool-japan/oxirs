//! Standalone binary that prints the Apache Jena feature parity report to stdout.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oxirs-arq --bin jena_parity_report
//! ```
//!
//! The report is a markdown document summarising the implementation status of
//! every catalogued Apache Jena feature across all OxiRS crates.

fn main() {
    let matrix = match oxirs_arq::jena_parity::load_catalog() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: failed to parse embedded jena_catalog.toml: {e}");
            std::process::exit(1);
        }
    };

    let report = oxirs_arq::jena_parity::generate_jena_report(&matrix);
    print!("{report}");
}
