//! Binary that emits the ESMF SDK 2.x parity report to stdout.
//!
//! Usage:
//! ```text
//! cargo run --bin parity_report -p oxirs-samm
//! ```
//!
//! Redirect the output to regenerate `docs/esmf_parity.md`:
//! ```text
//! cargo run --bin parity_report -p oxirs-samm > docs/esmf_parity.md
//! ```

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matrix = oxirs_samm::parity::load_catalog()?;
    let report = oxirs_samm::parity::generate_report(&matrix);
    print!("{report}");
    Ok(())
}
