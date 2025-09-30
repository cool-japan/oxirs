#!/bin/bash
# Download W3C SPARQL 1.1 Test Suite

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="$SCRIPT_DIR/data"

echo "Setting up W3C SPARQL 1.1 Test Suite..."

# Create data directory
mkdir -p "$DATA_DIR"

# W3C SPARQL 1.1 test suite locations
SPARQL11_BASE="https://www.w3.org/2009/sparql/docs/tests/data-sparql11"

# Download test categories
declare -a TEST_CATEGORIES=(
    "syntax-query"
    "syntax-update" 
    "basic-update"
    "clear"
    "drop"
    "copy"
    "move"
    "add"
    "delete-data"
    "delete-where"
    "delete-insert"
    "construct"
    "ask"
    "functions"
    "aggregates"
    "bind"
    "bindings"
    "cast"
    "csv-tsv-res"
    "entailment"
    "exists"
    "grouping"
    "json-res"
    "negation"
    "optional"
    "optional-filter"
    "project-expression"
    "property-path"
    "regex"
    "service"
    "subquery"
    "syntax-fed"
    "update-silent"
)

echo "Downloading test manifests and data files..."

for category in "${TEST_CATEGORIES[@]}"; do
    echo "Processing category: $category"
    
    # Create category directory
    mkdir -p "$DATA_DIR/$category"
    
    # Download manifest
    MANIFEST_URL="$SPARQL11_BASE/$category/manifest.ttl"
    echo "  Downloading manifest from $MANIFEST_URL"
    curl -s -o "$DATA_DIR/$category/manifest.ttl" "$MANIFEST_URL" || echo "  Failed to download manifest for $category"
done

# Download additional test resources
echo ""
echo "Downloading common test data files..."

# Common data files used across tests
declare -a COMMON_FILES=(
    "data-r2/basic/data.ttl"
    "data-r2/basic/data-all.ttl"
    "data-r2/optional/data.ttl"
    "data-r2/union/data.ttl"
)

mkdir -p "$DATA_DIR/data-r2/basic"
mkdir -p "$DATA_DIR/data-r2/optional"
mkdir -p "$DATA_DIR/data-r2/union"

for file in "${COMMON_FILES[@]}"; do
    FILE_URL="https://www.w3.org/2001/sw/DataAccess/tests/$file"
    echo "  Downloading $file"
    curl -s -o "$DATA_DIR/$file" "$FILE_URL" || echo "  Failed to download $file"
done

echo ""
echo "Creating test runner script..."

# Create a simple test runner
cat > "$SCRIPT_DIR/run_tests.rs" << 'EOF'
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
EOF

echo ""
echo "Setup complete!"
echo ""
echo "To run the tests:"
echo "  1. Make sure test data is downloaded: bash $SCRIPT_DIR/download_tests.sh"
echo "  2. Run tests: cargo test --test w3c_compliance"
echo ""
echo "Note: The test data download may take some time depending on your connection."
echo "Some test files may fail to download if they're not available on the W3C server."