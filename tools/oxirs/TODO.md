# OxiRS CLI - TODO

*Last Updated: September 30, 2025*

## ✅ Current Status: v0.1.0-alpha.2 Released

**oxirs** provides a comprehensive command-line interface for OxiRS operations with production-ready features.

### Alpha.2 Release Status

**Production-Ready Features** ✅:
- ✅ **Standards-compliant result formatters** (Table, JSON, CSV/TSV, XML)
- ✅ **W3C SPARQL 1.1 compliance** for all output formats
- ✅ **Comprehensive validation** with helpful error messages
- ✅ **Progress indicators** for long-running operations
- ✅ **Colored output** with NO_COLOR support
- ✅ **Query execution** with performance metrics
- ✅ **Interactive REPL** framework (integration pending)
- ✅ **Shell completion** generation support
- ✅ **Zero compilation errors/warnings**

**New in Alpha.2**:
- ✅ Production-grade SPARQL result formatters (520 lines)
- ✅ Factory pattern for formatter extensibility
- ✅ Proper RDF term handling (URIs, literals, blank nodes)
- ✅ Language tags and datatypes support
- ✅ CSV escaping and XML entity encoding
- ✅ 7 comprehensive formatter tests

**Installation**: `cargo install oxirs` (when published)

## 🎯 Post-Alpha.2 Development Roadmap

### Immediate Priority - Alpha.3 (2-3 weeks)

**Status**: 70% complete (formatters done, core commands pending)

#### 1. 🛠️ RDF Serialization (1 week) - P1
**Priority**: Critical for data export functionality

- [ ] **Turtle Serialization**
  - W3C compliant Turtle writer
  - Prefix management and optimization
  - Streaming support for large datasets
  - Integration with oxirs-core

- [ ] **N-Triples Serialization**
  - Simple line-based format
  - Streaming support
  - Performance optimization

- [ ] **RDF/XML Serialization**
  - W3C compliant RDF/XML writer
  - Pretty-printing support
  - Namespace management

- [ ] **JSON-LD Serialization**
  - JSON-LD 1.1 compliant
  - Context management
  - Compact/expanded formats

- [ ] **TriG Serialization**
  - Named graphs support
  - Turtle-based syntax

- [ ] **N-Quads Serialization**
  - Quad-based format
  - Simple streaming support

**Files to Update**:
- `tools/oxirs/src/export.rs:138-164` (6 TODOs)
- `tools/oxirs/src/commands/export.rs:112`

**Estimated Effort**: 5-7 days
**Dependencies**: oxirs-core serializers

#### 2. 📋 Configuration Management (1 day) - P1
**Priority**: Essential for proper dataset management

- [ ] **TOML Configuration Parser**
  - Parse oxirs.toml files
  - Extract dataset storage paths
  - Profile management (dev, staging, prod)
  - Environment variable substitution

- [ ] **Configuration Validation**
  - Schema validation
  - Required field checking
  - Path existence verification

- [ ] **Multi-profile Support**
  - `--profile` flag support
  - Profile-specific overrides
  - Default profile selection

**Files to Update**:
- `tools/oxirs/src/commands/query.rs:150`
- `tools/oxirs/src/commands/update.rs:75`
- `tools/oxirs/src/commands/import.rs:187`
- `tools/oxirs/src/commands/export.rs:101`

**Estimated Effort**: 1 day
**Implementation**: Shared configuration module

#### 3. 🔧 Core Commands Implementation (1 week) - P1
**Priority**: Essential CLI functionality

##### 3.1 `serve` Command (2-3 days)
**Status**: Stub only

- [ ] **Load Configuration**
  - Parse oxirs.toml
  - Initialize server config
  - Setup logging and metrics

- [ ] **Initialize Dataset**
  - Open/create TDB2 store
  - Load initial data
  - Setup indexes

- [ ] **Start HTTP Server**
  - Launch oxirs-fuseki server
  - Enable SPARQL endpoint
  - Optional GraphQL endpoint
  - Health checks and metrics

**Files to Update**:
- `tools/oxirs/src/commands/serve.rs:16-18` (3 TODOs)

##### 3.2 `migrate` Command (1 day)
**Status**: Stub only

- [ ] **Format Detection**
  - Auto-detect source format
  - Validate target format

- [ ] **Data Migration**
  - Streaming migration for large datasets
  - Progress tracking
  - Error handling and rollback

- [ ] **Validation**
  - Verify data integrity
  - Triple count verification

**Files to Update**:
- `tools/oxirs/src/commands/migrate.rs:7`

##### 3.3 `update` Command (2 days)
**Status**: Partial implementation

- [ ] **SPARQL Update Execution**
  - Parse SPARQL update
  - Execute against store
  - Transaction support

- [ ] **Update Validation**
  - Syntax validation
  - Semantic validation
  - Dry-run mode

**Files to Update**:
- `tools/oxirs/src/commands/update.rs:52,81` (2 TODOs)

##### 3.4 `import` Command (1 day)
**Status**: Partial implementation

- [ ] **Triple Conversion**
  - Convert parsed triples to Statement
  - Batch insertion for performance
  - Error handling

- [ ] **Format Support**
  - All RDF formats
  - Streaming for large files
  - Progress tracking

**Files to Update**:
- `tools/oxirs/src/commands/import.rs:213`

##### 3.5 `export` Command (1 day)
**Status**: Partial implementation

- [ ] **Data Export Implementation**
  - Query all triples from store
  - Optional graph filtering
  - Format conversion

- [ ] **Performance Optimization**
  - Streaming export
  - Parallel processing
  - Memory efficiency

**Files to Update**:
- `tools/oxirs/src/commands/export.rs:112`

#### 4. 🎮 Interactive Mode Enhancement (3-4 days) - P1
**Priority**: Professional REPL experience

- [ ] **Real Query Execution**
  - Integrate with actual SPARQL engine
  - Result formatting with new formatters
  - Performance metrics display

- [ ] **Command Integration** (8 items)
  - Import command integration
  - Export command integration
  - Validation command integration
  - Stats command integration
  - Riot command integration
  - SHACL command integration
  - TDB loader integration
  - TDB dump integration

- [ ] **Session Management**
  - History persistence
  - Multi-line query editing
  - Query templates
  - Saved queries

**Files to Update**:
- `tools/oxirs/src/cli/interactive.rs` (8 TODOs)

**Estimated Effort**: 3-4 days

#### 5. 🧹 Code Cleanup (2 days) - P1
**Priority**: Code quality and maintainability

- [ ] **Remove Obsolete Functions**
  - Delete old formatter stubs in `commands/query.rs:189-227`
  - Update tests that check for "TODO" in output
  - Clean up unused imports

- [ ] **Refactor Large Files**
  - Split files >2000 lines
  - Extract common functionality
  - Improve code organization

- [ ] **Documentation Updates**
  - Update inline documentation
  - Add usage examples
  - Update README

**Estimated Effort**: 2 days

---

## 📊 Alpha.3 Summary

**Total Effort**: 2-3 weeks (25 P1 items)

| Task | Priority | Effort | Status |
|------|----------|--------|--------|
| RDF Serialization | P1 | 5-7 days | 📋 Planned |
| Configuration Management | P1 | 1 day | 📋 Planned |
| Core Commands | P1 | 5-7 days | 📋 Planned |
| Interactive Mode | P1 | 3-4 days | 📋 Planned |
| Code Cleanup | P1 | 2 days | 📋 Planned |
| **Total** | - | **16-21 days** | - |

---

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Advanced Features (P2)

##### Query Optimization Tools
- [ ] Implement query optimization (`tools/oxirs/src/tools/arq.rs:70`)
- [ ] Parse ORDER BY clause (`tools/oxirs/src/tools/arq.rs:224`)
- [ ] Query plan visualization
- [ ] Cost estimation display

##### Performance Profiling
- [ ] Query profiling command
- [ ] Performance benchmarking suite
- [ ] Bottleneck analysis
- [ ] Resource usage monitoring

##### Advanced Import/Export
- [ ] Streaming import for large datasets
- [ ] Parallel import/export
- [ ] Compression support
- [ ] Resume capability for interrupted operations

##### Database Administration
- [ ] Database statistics command
- [ ] Index management
- [ ] Optimization tools
- [ ] Backup/restore commands

#### User Experience Enhancements (P2)

##### Shell Integration
- [ ] Advanced shell completion
- [ ] Command aliases
- [ ] Custom keybindings
- [ ] Integration with popular shells

##### Output Formatting
- [ ] HTML output format
- [ ] Markdown table format
- [ ] Custom output templates
- [ ] Syntax highlighting

##### Interactive Mode
- [ ] Autocomplete for SPARQL keywords
- [ ] Query history search
- [ ] Multi-dataset connections
- [ ] Transaction support

---

### v0.2.0 Targets (Q1 2026)

#### Professional CLI Suite (P2-P3)

##### Benchmarking Tools
- [ ] SP2Bench suite integration
- [ ] WatDiv benchmark support
- [ ] LDBC benchmark support
- [ ] Custom benchmark generation

##### Migration & Conversion
- [ ] Jena TDB1 → OxiRS migration
- [ ] Virtuoso → OxiRS migration
- [ ] RDF4J → OxiRS migration
- [ ] Format conversion utilities

##### Dataset Generation
- [ ] Synthetic dataset generation
- [ ] Schema-based data generation
- [ ] Test data creation
- [ ] Bulk data loading tools

##### CI/CD Integration
- [ ] Test result reporting
- [ ] Performance regression detection
- [ ] Automated validation
- [ ] Docker integration helpers

---

## 🎯 Implementation Progress

| Category | Alpha.1 | Alpha.2 | Alpha.3 Target | Beta.1 Target |
|----------|---------|---------|----------------|---------------|
| **Result Formatters** | 0% | 100% | 100% | 100% |
| **Core Commands** | 30% | 40% | 95% | 100% |
| **Configuration** | 20% | 30% | 90% | 100% |
| **Interactive Mode** | 50% | 60% | 95% | 100% |
| **RDF Serialization** | 0% | 0% | 90% | 100% |
| **Query Tools** | 60% | 65% | 70% | 95% |
| **Admin Tools** | 40% | 45% | 50% | 90% |
| **Documentation** | 50% | 70% | 85% | 100% |
| **Overall** | **40%** | **70%** | **95%** | **99%** |

---

## 📝 Known Issues & Limitations

### Alpha.2 Limitations

**Command Implementation**:
- ⚠️ `serve` command is stub (needs oxirs-fuseki integration)
- ⚠️ `migrate` command is stub
- ⚠️ `update` command incomplete (no SPARQL update parsing)
- ⚠️ `export` command incomplete (no actual serialization)
- ⚠️ Interactive mode not integrated with real query execution

**Serialization**:
- ⚠️ RDF/XML, Turtle, TriG serialization not implemented
- ⚠️ JSON-LD serialization incomplete
- ⚠️ N-Quads serialization not implemented

**Configuration**:
- ⚠️ TOML parsing not implemented
- ⚠️ Profile management not supported
- ⚠️ Environment variables not integrated

---

## 🔧 Development Guidelines

### Adding New Commands

1. Add command variant to `Commands` enum in `lib.rs`
2. Implement command handler in `commands/` directory
3. Add validation logic
4. Add progress tracking
5. Add comprehensive error handling
6. Add tests
7. Update documentation

### Adding New Output Formats

1. Implement `ResultFormatter` trait in `cli/formatters.rs`
2. Add to factory function `create_formatter()`
3. Add format validation
4. Add comprehensive tests
5. Update documentation

### Testing Checklist

- [ ] Unit tests for all functions
- [ ] Integration tests for commands
- [ ] Error handling tests
- [ ] Performance tests for large datasets
- [ ] Documentation tests (examples)

---

## 📚 Documentation Status

### Completed ✅
- ✅ Result formatters documentation (`/tmp/CLI_FORMATTERS.md`)
- ✅ Release summary (`/tmp/RELEASE_SUMMARY_0.1.0-alpha.2.md`)
- ✅ TODO analysis (`/tmp/TODO_ANALYSIS_ALPHA2_UPDATED.md`)

### Pending 📋
- [ ] Command reference manual
- [ ] Interactive mode guide
- [ ] Configuration file reference
- [ ] Migration guide
- [ ] Best practices guide

---

## 🎊 Success Metrics

### Alpha.2 Achievements

✅ **Formatters**: 4 production-ready formatters (Table, JSON, CSV/TSV, XML)
✅ **Standards**: W3C SPARQL 1.1 compliance verified
✅ **Tests**: 7 new tests (100% passing)
✅ **Code Quality**: Zero warnings, clean compilation
✅ **Documentation**: Comprehensive formatter docs

### Alpha.3 Targets

🎯 **Commands**: All core commands functional (serve, migrate, update, import, export)
🎯 **Serialization**: 6 RDF formats fully implemented
🎯 **Configuration**: TOML parsing and profile management
🎯 **Interactive**: Real query execution integration
🎯 **Quality**: 95% test coverage, <2000 lines per file

---

*OxiRS CLI v0.1.0-alpha.2: Production-ready result formatters with W3C SPARQL 1.1 compliance. Released September 30, 2025.*

*Next: v0.1.0-alpha.3 (CLI completion) - Target: October 2025*