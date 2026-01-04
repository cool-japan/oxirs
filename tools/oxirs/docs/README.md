# OxiRS CLI Documentation

**Version**: 0.1.0-rc.2
**Last Updated**: December 26, 2025
**Status**: Production-Ready

## Overview

Welcome to the OxiRS CLI comprehensive documentation. This documentation provides everything you need to use OxiRS effectively, from getting started to advanced production deployments.

## Documentation Structure

### Quick Start

- **[QUICKSTART.md](../QUICKSTART.md)** - Get up and running in 5 minutes
  - Installation instructions
  - Basic commands with examples
  - Common workflows
  - Performance benchmarks
  - **Start here if you're new to OxiRS!**

### Core Guides

1. **[COMMAND_REFERENCE.md](COMMAND_REFERENCE.md)** - Complete command reference
   - All 50+ CLI commands documented
   - Detailed usage examples for each command
   - Command options and flags
   - Exit codes and error handling
   - **Your go-to reference for any command**

2. **[INTERACTIVE.md](INTERACTIVE.md)** - Interactive mode guide
   - REPL features and usage
   - Tab completion and history
   - Multi-line queries
   - Query templates
   - Keyboard shortcuts
   - **Essential for interactive data exploration**

3. **[CONFIGURATION.md](CONFIGURATION.md)** - Configuration reference
   - Complete oxirs.toml documentation
   - Server configuration
   - Dataset configuration
   - Authentication and security
   - Performance tuning
   - Environment-specific profiles
   - **Required for production deployments**

4. **[BEST_PRACTICES.md](BEST_PRACTICES.md)** - Best practices guide
   - Production deployment patterns
   - Security best practices
   - Performance optimization
   - Monitoring and maintenance
   - Troubleshooting
   - Common workflows
   - **Must-read for production users**

### Additional Resources

- **[README.md](../README.md)** - Main CLI documentation
  - Feature overview
  - Installation
  - Command categories
  - Troubleshooting
  - Best practices section

- **[oxirs.toml.example](../oxirs.toml.example)** - Example configuration
  - Annotated configuration file
  - All options with comments
  - Example profiles (dev/staging/prod)

## Quick Links

### By Use Case

**I want to...**

- **Get started quickly** → [QUICKSTART.md](../QUICKSTART.md)
- **Look up a command** → [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md)
- **Use interactive mode** → [INTERACTIVE.md](INTERACTIVE.md)
- **Configure the server** → [CONFIGURATION.md](CONFIGURATION.md)
- **Deploy to production** → [BEST_PRACTICES.md](BEST_PRACTICES.md)
- **Optimize performance** → [BEST_PRACTICES.md#performance-tuning](BEST_PRACTICES.md#performance-tuning)
- **Secure my deployment** → [BEST_PRACTICES.md#security-best-practices](BEST_PRACTICES.md#security-best-practices)
- **Troubleshoot issues** → [BEST_PRACTICES.md#troubleshooting](BEST_PRACTICES.md#troubleshooting)

### By Topic

**Commands**
- [Core Commands](COMMAND_REFERENCE.md#core-commands) - init, query, import, export, serve, update
- [Data Processing](COMMAND_REFERENCE.md#data-processing) - riot, batch, migrate
- [Query Tools](COMMAND_REFERENCE.md#query-tools) - arq, explain, template, history
- [Storage Management](COMMAND_REFERENCE.md#storage-management) - tdbloader, tdbdump, tdbstats, etc.
- [Validation & Reasoning](COMMAND_REFERENCE.md#validation--reasoning) - shacl, shex, infer
- [SAMM & Industry 4.0](COMMAND_REFERENCE.md#samm--industry-40) - aspect, aas, package
- [Utility Commands](COMMAND_REFERENCE.md#utility-commands) - config, interactive, benchmark

**Configuration**
- [Server Configuration](CONFIGURATION.md#server-configuration)
- [Dataset Configuration](CONFIGURATION.md#dataset-configuration)
- [Authentication & Security](CONFIGURATION.md#authentication--security)
- [Performance & Optimization](CONFIGURATION.md#performance--optimization)
- [Logging & Monitoring](CONFIGURATION.md#logging--monitoring)
- [Feature Flags](CONFIGURATION.md#feature-flags)

**Production**
- [Production Deployment](BEST_PRACTICES.md#production-deployment)
- [Security Best Practices](BEST_PRACTICES.md#security-best-practices)
- [Monitoring & Maintenance](BEST_PRACTICES.md#monitoring--maintenance)
- [Backup Strategy](BEST_PRACTICES.md#backup-strategy)

## Documentation Coverage

All documentation is complete and production-ready.

| Area | Status |
|------|--------|
| Quick Start | ✅ Complete |
| Command Reference | ✅ Complete |
| Interactive Mode | ✅ Complete |
| Configuration | ✅ Complete |
| Best Practices | ✅ Complete |
| Main README | ✅ Complete |

**OxiRS Project**: ~1M lines of Rust code across all modules

## Learning Path

### Beginner Path

1. Read [QUICKSTART.md](../QUICKSTART.md) (15 minutes)
2. Try basic commands
3. Explore [INTERACTIVE.md](INTERACTIVE.md) for REPL usage
4. Reference [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md) as needed

### Intermediate Path

1. Complete Beginner Path
2. Read [BEST_PRACTICES.md](BEST_PRACTICES.md)
3. Set up custom [CONFIGURATION.md](CONFIGURATION.md)
4. Implement query optimization techniques
5. Set up monitoring and backups

### Advanced Path

1. Complete Intermediate Path
2. Read production sections in [BEST_PRACTICES.md](BEST_PRACTICES.md)
3. Implement advanced security (JWT, OAuth2)
4. Set up high-availability deployment
5. Optimize for your specific workload
6. Contribute to OxiRS development

## Documentation Quality

All documentation follows these standards:

- ✅ **Comprehensive** - Covers all features and options
- ✅ **Practical** - Includes real-world examples
- ✅ **Up-to-date** - Reflects rc.1 features and APIs
- ✅ **Well-organized** - Logical structure with TOC
- ✅ **Searchable** - Clear headings and cross-references
- ✅ **Tested** - All examples are verified to work
- ✅ **Production-ready** - Includes deployment guides

## Getting Help

### Documentation

Start with the documentation above. It's comprehensive and covers 99% of use cases.

### Command-Line Help

```bash
# General help
oxirs --help

# Command-specific help
oxirs query --help
oxirs serve --help

# Generate shell completion
oxirs --completion bash > /etc/bash_completion.d/oxirs
```

### Community

- **GitHub Issues**: https://github.com/cool-japan/oxirs/issues
- **Discussions**: https://github.com/cool-japan/oxirs/discussions
- **Changelog**: [CHANGELOG.md](../../../CHANGELOG.md)

### Contributing

Found an issue or want to improve the documentation?

1. Check [existing issues](https://github.com/cool-japan/oxirs/issues)
2. Open a new issue or pull request
3. Follow the contribution guidelines

## Version History

- **v0.1.0-rc.2** (December 2025) - Initial complete documentation
  - Command reference manual
  - Interactive mode guide
  - Configuration reference
  - Best practices guide

## Future Documentation

Planned for future releases:

- API Reference (for library users)
- GraphQL Guide
- Federation Guide
- Vector Search Guide
- SAMM/AAS Advanced Guide
- Video Tutorials
- Cookbook with recipes

---

## Quick Reference Card

### Most Used Commands

```bash
# Initialize dataset
oxirs init mydata

# Import data
oxirs import mydata file.ttl

# Query data
oxirs query mydata "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Start server
oxirs serve --config oxirs.toml

# Interactive mode
oxirs interactive --dataset mydata

# Export data
oxirs export mydata output.nq --format nquads

# Get help
oxirs --help
oxirs query --help
```

### Key Concepts

- **Dataset** - Named RDF data container (TDB2 or memory)
- **SPARQL** - Query language for RDF data
- **Triple** - Basic RDF unit: subject-predicate-object
- **Named Graph** - Grouped triples with URI identifier
- **Quad** - Triple + named graph

### File Formats

- `.ttl` - Turtle (recommended for human editing)
- `.nt` - N-Triples (fastest for bulk operations)
- `.nq` - N-Quads (includes named graphs)
- `.trig` - TriG (Turtle with named graphs)
- `.rdf`, `.xml` - RDF/XML
- `.jsonld` - JSON-LD

---

**OxiRS CLI v0.1.0-rc.2** - Production-ready semantic web toolkit with comprehensive documentation

**Last Updated**: December 26, 2025
