#!/bin/bash
# Run all fuzz targets for a specified duration (default: 60 seconds each)

set -e

DURATION=${1:-60}  # Default 60 seconds per fuzzer
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Running all fuzz targets for ${DURATION}s each"
echo "=========================================="

TARGETS=(
    "turtle_parser"
    "ntriples_parser"
    "nquads_parser"
    "trig_parser"
    "turtle_serializer"
)

for target in "${TARGETS[@]}"; do
    echo ""
    echo ">>> Fuzzing: $target (${DURATION}s)"
    echo "----------------------------------------"

    cargo fuzz run "$target" -- -max_total_time="$DURATION" || {
        echo "ERROR: Fuzzer $target crashed or failed!"
        echo "Check artifacts in fuzz/artifacts/$target/"
        exit 1
    }

    echo "✓ $target completed"
done

echo ""
echo "=========================================="
echo "All fuzz targets completed successfully!"
echo "=========================================="
echo ""
echo "Summary:"
for target in "${TARGETS[@]}"; do
    corpus_count=$(find "fuzz/corpus/$target" -type f 2>/dev/null | wc -l || echo 0)
    artifacts_count=$(find "fuzz/artifacts/$target" -type f 2>/dev/null | wc -l || echo 0)
    echo "  $target: $corpus_count corpus files, $artifacts_count artifacts"
done
