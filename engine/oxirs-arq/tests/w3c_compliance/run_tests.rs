//! W3C SPARQL Test Suite Runner

use anyhow::Result;
use std::env;
use std::path::PathBuf;

mod w3c_compliance;
use w3c_compliance::TestSuiteRunner;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let test_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("tests/w3c_compliance/data")
    };

    println!("Running W3C SPARQL compliance tests from: {:?}", test_dir);

    let mut runner = TestSuiteRunner::new(&test_dir)?;

    // Run specific test categories or all
    let categories = if args.len() > 2 {
        vec![args[2].clone()]
    } else {
        vec![
            "syntax-query",
            "functions",
            "aggregates",
            "bind",
            "property-path",
            "optional",
        ]
    };

    for category in categories {
        println!("\nRunning tests for category: {}", category);
        let manifest_path = test_dir.join(&category).join("manifest.ttl");

        if manifest_path.exists() {
            match runner.run_manifest(&manifest_path) {
                Ok(_) => println!("Category {} completed", category),
                Err(e) => println!("Error running category {}: {}", category, e),
            }
        } else {
            println!("Manifest not found for category: {}", category);
        }
    }

    runner.print_summary();

    Ok(())
}
