# OxiRS CLI Migration Guide

**Version**: 0.2.0  
**Last Updated**: March 5, 2026  
**Status**: Updated for v0.2.0

## Overview

This guide will help you migrate from other triple stores and SPARQL endpoints to OxiRS.

---

## Supported Migration Sources

OxiRS supports migration from the following systems:

### ✅ Currently Implemented

1. **Apache Jena TDB1** - `oxirs migrate from-tdb1`
2. **Apache Jena TDB2** - `oxirs migrate from-tdb2`
3. **OpenLink Virtuoso** - `oxirs migrate from-virtuoso`
4. **Eclipse RDF4J** - `oxirs migrate from-rdf4j`
5. **Blazegraph** - `oxirs migrate from-blazegraph`
6. **Ontotext GraphDB** - `oxirs migrate from-graphdb`

### 🚧 Planned (v0.3.0)

- Apache Jena Fuseki (SPARQL endpoint)
- Stardog
- AllegroGraph
- Amazon Neptune
- Azure Cosmos DB (Gremlin API)

---

## Quick Start Examples

### From Jena TDB2
```bash
oxirs migrate from-tdb2 /path/to/tdb2 my_dataset
```

### From Virtuoso
```bash
oxirs migrate from-virtuoso \
  --connection "host=localhost;port=1111;uid=dba;pwd=dba" \
  --dataset my_dataset
```

### From RDF4J
```bash
oxirs migrate from-rdf4j /path/to/rdf4j/repository my_dataset
```

### From Blazegraph
```bash
oxirs migrate from-blazegraph \
  --endpoint http://localhost:9999/blazegraph/sparql \
  --dataset my_dataset \
  --namespace kb
```

### From GraphDB
```bash
oxirs migrate from-graphdb \
  --endpoint http://localhost:7200 \
  --dataset my_dataset \
  --repository my_repo
```

---

## Format Migration

Convert between RDF formats:

```bash
# Turtle to N-Quads
oxirs migrate format input.ttl output.nq --from turtle --to nquads

# RDF/XML to JSON-LD
oxirs migrate format input.rdf output.jsonld --from rdfxml --to jsonld
```

---

## Best Practices

### Before Migration

1. ✅ **Backup source data** - Always create a backup before migration
2. ✅ **Test with subset** - Test migration with a small dataset first
3. ✅ **Check disk space** - Ensure sufficient disk space for target dataset
4. ✅ **Verify connectivity** - Test connection to source system

### During Migration

1. 📊 **Monitor progress** - Use `--verbose` flag for detailed progress
2. 🔄 **Use resume capability** - Large migrations support `--resume` flag
3. ⚡ **Adjust batch size** - Tune performance with `--batch-size` option
4. 🔍 **Validate continuously** - Check sample data during migration

### After Migration

1. ✅ **Verify data integrity** - Compare triple counts
2. ✅ **Run validation** - Use SHACL/ShEx validation if available
3. ✅ **Test queries** - Run representative queries
4. ✅ **Benchmark performance** - Compare query performance

---

## Troubleshooting

### Common Issues

#### Connection Timeout
```bash
# Increase timeout for remote endpoints
oxirs migrate from-virtuoso --connection "..." --timeout 300
```

#### Memory Issues
```bash
# Reduce batch size for large datasets
oxirs migrate from-tdb2 /path/to/tdb2 dataset --batch-size 1000
```

#### Encoding Issues
```bash
# Force UTF-8 encoding
oxirs migrate format input.ttl output.nq --encoding utf-8
```

---

## Advanced Topics

### Custom Migration Scripts

For complex migrations, use the OxiRS API directly:

```rust
// Example migration script (to be documented in v0.3.0)
// See oxirs/src/commands/migrate.rs for reference
```

### Performance Tuning

Migration performance tips:
- Use SSD storage for target dataset
- Increase system memory allocation
- Disable validation during import (re-validate after)
- Use parallel import for independent graphs

---

## See Also

- [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md) - Full command documentation
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options
- [BEST_PRACTICES.md](BEST_PRACTICES.md) - Best practices guide

---

**Note**: This migration guide is a work in progress. Detailed step-by-step migration procedures for each source system will be added in v0.3.0. For now, use `oxirs migrate --help` for current options.

**OxiRS CLI v0.2.0** - Migration guide
