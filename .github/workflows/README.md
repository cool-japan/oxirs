# OxiRS CI/CD Workflows

This directory contains GitHub Actions workflows for automated testing, building, and deployment of OxiRS components.

## Workflows

### `fuseki-ci.yml` - OxiRS Fuseki CI/CD Pipeline

Comprehensive CI/CD pipeline for the oxirs-fuseki SPARQL server.

#### Triggered On
- Push to `main`, `develop`, `release/**`, `beta/**` branches
- Pull requests to `main`, `develop`
- Manual workflow dispatch
- Paths: Changes to fuseki, core, arq modules, or workflow files

#### Jobs

##### 1. Lint (`lint`)
- **Purpose**: Code quality checks
- **Steps**:
  - Formatting check with `cargo fmt`
  - Linting with `cargo clippy` (zero warnings enforced)
- **Runs on**: Ubuntu latest
- **Caching**: Cargo registry, git index, build artifacts

##### 2. Test Suite (`test`)
- **Purpose**: Comprehensive testing across platforms
- **Matrix**:
  - OS: Ubuntu, macOS
  - Rust: stable, nightly
- **Steps**:
  - Build with all features
  - Run unit tests
  - Run documentation tests
- **Test threads**: Limited to 1 for consistency

##### 3. Security Audit (`security`)
- **Purpose**: Dependency vulnerability scanning
- **Tools**: `cargo-audit`
- **Runs on**: Ubuntu latest

##### 4. Code Coverage (`coverage`)
- **Purpose**: Test coverage analysis
- **Tools**: `cargo-tarpaulin`
- **Output**: Codecov integration
- **Timeout**: 600 seconds

##### 5. Build Release (`build-release`)
- **Purpose**: Multi-platform binary builds
- **Matrix**:
  - Linux AMD64 (GNU libc)
  - Linux AMD64 (musl libc) - static binary
  - macOS AMD64 (Intel)
  - macOS ARM64 (Apple Silicon)
- **Features**:
  - Binary stripping for size optimization
  - Size reporting
  - Artifact upload (7-day retention)
- **Dependencies**: Passes `lint` and `test` jobs

##### 6. Docker Build (`docker`)
- **Purpose**: Container image build and registry push
- **Features**:
  - Multi-platform support (linux/amd64, linux/arm64)
  - Buildx caching
  - Metadata extraction for tags
  - Push to Docker Hub (on non-PR events)
- **Tags**: branch, PR, semver, SHA
- **Dependencies**: Passes `lint` and `test` jobs

##### 7. Performance Benchmarks (`benchmark`)
- **Purpose**: Performance regression testing
- **Trigger**: Push to `main` branch only
- **Tools**: `cargo bench`
- **Output**: Criterion reports (30-day retention)

##### 8. Deploy Staging (`deploy-staging`)
- **Purpose**: Automatic deployment to staging
- **Trigger**: Push to `develop` branch
- **Environment**: `staging`
- **Dependencies**: All build/test jobs pass
- **Note**: Deployment commands need to be configured

##### 9. Deploy Production (`deploy-production`)
- **Purpose**: Automatic deployment to production
- **Trigger**:
  - Push to `main` branch
  - Version tags (`v*`)
- **Environment**: `production`
- **Dependencies**: All build/test jobs pass
- **Note**: Deployment commands need to be configured

##### 10. GitHub Release (`release`)
- **Purpose**: Automated release creation
- **Trigger**: Version tags (`v*`)
- **Artifacts**: All platform binaries
- **Dependencies**: All build/test jobs pass

## Configuration

### Required Secrets

For full functionality, configure these GitHub repository secrets:

- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password/token
- `CODECOV_TOKEN`: Codecov upload token (optional, for private repos)

### Deployment Configuration

The staging and production deployment jobs are placeholders. Configure them based on your infrastructure:

**Kubernetes with kubectl:**
```yaml
- name: Deploy to staging
  run: |
    kubectl config use-context staging
    kubectl apply -f deployment/kubernetes/
    kubectl rollout status deployment/oxirs-fuseki
```

**Kubernetes with Helm:**
```yaml
- name: Deploy to production
  run: |
    helm upgrade --install oxirs-fuseki ./charts/oxirs-fuseki \
      --namespace production \
      --values values-production.yaml \
      --wait
```

**Terraform:**
```yaml
- name: Deploy with Terraform
  run: |
    cd deployment/terraform/aws
    terraform init
    terraform plan
    terraform apply -auto-approve
```

## Binary Size Targets

- **Target**: < 50MB
- **Current**: ~12MB (stripped)
- **Optimization**: Binaries are automatically stripped during CI

## Success Metrics

- ✅ **Tests**: 358+ unit tests passing
- ✅ **Warnings**: Zero warnings enforced via `RUSTFLAGS='-D warnings'`
- ✅ **Binary Size**: 12MB (well under 50MB target)
- ✅ **Platforms**: Linux (GNU/musl), macOS (Intel/ARM)
- ✅ **Security**: Automated vulnerability scanning
- ✅ **Coverage**: Automated coverage reporting

## Local Testing

Test the CI pipeline locally before pushing:

```bash
# Lint
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Test
cargo test --workspace --all-features

# Security audit
cargo install cargo-audit
cargo audit

# Build release
cargo build --release -p oxirs-fuseki --all-features
strip target/release/oxirs-fuseki
ls -lh target/release/oxirs-fuseki

# Benchmarks
cargo bench -p oxirs-fuseki --all-features

# Coverage (optional)
cargo install cargo-tarpaulin
cargo tarpaulin --workspace --all-features --timeout 600
```

## Caching Strategy

The workflow uses GitHub Actions cache to speed up builds:

1. **Cargo registry**: `~/.cargo/registry`
2. **Cargo git index**: `~/.cargo/git`
3. **Build artifacts**: `target/`
4. **Docker layers**: BuildKit cache

Cache keys are based on:
- OS and Rust version
- `Cargo.lock` hash

## Troubleshooting

### Build Failures

1. Check if all dependencies are available on crates.io
2. Verify Rust version compatibility
3. Review clippy warnings/errors
4. Check for platform-specific issues

### Test Failures

1. Tests run with `--test-threads=1` to avoid race conditions
2. Check for environment-specific assumptions
3. Review test timeout settings

### Binary Size Issues

If binary exceeds 50MB:

1. Enable link-time optimization (LTO) in `Cargo.toml`:
   ```toml
   [profile.release]
   lto = true
   codegen-units = 1
   ```

2. Strip symbols:
   ```bash
   strip target/release/oxirs-fuseki
   ```

3. Use `cargo-bloat` to identify large dependencies:
   ```bash
   cargo install cargo-bloat
   cargo bloat --release -n 20
   ```

## Future Enhancements

- [ ] Integration tests with real SPARQL endpoints
- [ ] Load testing with K6 or Gatling
- [ ] Automated security SAST/DAST scanning
- [ ] Multi-cloud deployment (AWS, GCP, Azure)
- [ ] Automated changelog generation
- [ ] Release notes automation
- [ ] Performance regression tracking
