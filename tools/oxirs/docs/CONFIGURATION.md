# OxiRS Configuration Reference

**Version**: 0.1.0-rc.1
**Last Updated**: December 21, 2025
**Status**: Production-Ready

## Overview

OxiRS uses TOML configuration files for server and advanced CLI settings. This guide provides a comprehensive reference for all configuration options.

## Table of Contents

- [Quick Start](#quick-start)
- [Server Configuration](#server-configuration)
- [Dataset Configuration](#dataset-configuration)
- [Authentication & Security](#authentication--security)
- [Performance & Optimization](#performance--optimization)
- [Logging & Monitoring](#logging--monitoring)
- [Feature Flags](#feature-flags)
- [Environment-Specific Profiles](#environment-specific-profiles)
- [Examples](#examples)

---

## Quick Start

### Generate Default Configuration

```bash
# Generate default config file
oxirs config init --output oxirs.toml

# Validate configuration
oxirs config validate oxirs.toml

# Show current configuration
oxirs config show oxirs.toml
```

### Minimal Configuration

```toml
# oxirs.toml - Minimal configuration
[server]
host = "localhost"
port = 3030

[[datasets]]
name = "mydata"
location = "./data/mydata"
type = "tdb2"
```

---

## Server Configuration

### Basic Server Settings

```toml
[server]
# Host address to bind to
# - "localhost" for local access only
# - "0.0.0.0" for all interfaces
# - Specific IP for single interface
host = "localhost"

# Port number
port = 3030

# Worker threads (default: CPU cores)
workers = 4

# Request timeout (seconds)
timeout = 300

# Keep-alive timeout (seconds)
keep_alive = 75

# Maximum request size (MB)
max_request_size = 10
```

### TLS/SSL Configuration

```toml
[server.tls]
# Enable TLS
enabled = true

# Certificate file path
cert = "/path/to/cert.pem"

# Private key file path
key = "/path/to/key.pem"

# Certificate chain (optional)
chain = "/path/to/chain.pem"

# TLS version
# Options: "tls12", "tls13"
version = "tls13"

# Cipher suites (optional, defaults to secure set)
ciphers = [
    "TLS_AES_256_GCM_SHA384",
    "TLS_AES_128_GCM_SHA256",
]
```

### CORS Configuration

```toml
[server.cors]
# Enable CORS
enabled = true

# Allowed origins
# Use ["*"] to allow all origins (not recommended for production)
allowed_origins = [
    "http://localhost:3000",
    "https://example.com",
]

# Allowed methods
allowed_methods = ["GET", "POST", "OPTIONS"]

# Allowed headers
allowed_headers = ["Content-Type", "Authorization"]

# Expose headers
expose_headers = ["Content-Length", "ETag"]

# Allow credentials
allow_credentials = true

# Max age (seconds)
max_age = 3600
```

---

## Dataset Configuration

### Basic Dataset

```toml
[[datasets]]
# Dataset name (must be unique)
name = "mydata"

# Storage location
location = "./data/mydata"

# Dataset type
# Options: "tdb2", "memory"
type = "tdb2"

# Read-only mode
readonly = false

# Enable at startup
enabled = true
```

### Advanced Dataset Settings

```toml
[[datasets]]
name = "advanced"
location = "./data/advanced"
type = "tdb2"

# Union default graph (includes all named graphs)
union_default_graph = false

# Enable text index
[datasets.features]
text_index = true
spatial_index = true
vector_search = false

# Cache settings
[datasets.cache]
# Query result cache size (MB)
result_cache = 256

# Triple pattern cache size (MB)
pattern_cache = 128

# Cache TTL (seconds)
ttl = 3600

# Performance tuning
[datasets.performance]
# Batch size for bulk operations
batch_size = 10000

# Enable parallel processing
parallel = true

# Number of threads for parallel operations
threads = 4
```

### Multiple Datasets

```toml
# Development dataset
[[datasets]]
name = "dev"
location = "./data/dev"
type = "memory"
enabled = true

# Production dataset
[[datasets]]
name = "prod"
location = "/var/lib/oxirs/prod"
type = "tdb2"
readonly = false
enabled = true

# Read-only archive
[[datasets]]
name = "archive"
location = "/mnt/storage/archive"
type = "tdb2"
readonly = true
enabled = true
```

---

## Authentication & Security

### JWT Authentication

```toml
[auth]
# Authentication type
# Options: "none", "basic", "jwt", "oauth2"
type = "jwt"

[auth.jwt]
# JWT secret key (REQUIRED - use environment variable)
secret = "${JWT_SECRET}"

# Token expiration (seconds)
expiration = 3600

# Issuer
issuer = "oxirs-server"

# Audience
audience = "oxirs-api"

# Algorithm
# Options: "HS256", "HS384", "HS512", "RS256", "RS384", "RS512"
algorithm = "HS256"

# Public key for RS* algorithms (optional)
# public_key = "/path/to/public.pem"
```

### Basic Authentication

```toml
[auth]
type = "basic"

[auth.basic]
# Admin credentials (use environment variables)
admin_user = "${ADMIN_USER}"
admin_password = "${ADMIN_PASSWORD}"

# Additional users
[[auth.basic.users]]
username = "reader"
password = "${READER_PASSWORD}"
roles = ["read"]

[[auth.basic.users]]
username = "writer"
password = "${WRITER_PASSWORD}"
roles = ["read", "write"]
```

### OAuth2

```toml
[auth]
type = "oauth2"

[auth.oauth2]
# Provider
# Options: "google", "github", "azure", "custom"
provider = "google"

# Client credentials
client_id = "${OAUTH_CLIENT_ID}"
client_secret = "${OAUTH_CLIENT_SECRET}"

# Redirect URI
redirect_uri = "http://localhost:3030/oauth/callback"

# Scopes
scopes = ["openid", "profile", "email"]

# Custom provider endpoints (for provider = "custom")
# authorize_url = "https://..."
# token_url = "https://..."
# userinfo_url = "https://..."
```

### Role-Based Access Control (RBAC)

```toml
[auth.rbac]
# Enable RBAC
enabled = true

# Default role for authenticated users
default_role = "reader"

# Role permissions
[auth.rbac.roles.admin]
permissions = ["read", "write", "delete", "admin"]

[auth.rbac.roles.writer]
permissions = ["read", "write"]

[auth.rbac.roles.reader]
permissions = ["read"]

# Dataset-specific permissions
[[auth.rbac.dataset_permissions]]
dataset = "sensitive"
role = "admin"
permissions = ["read", "write", "delete"]

[[auth.rbac.dataset_permissions]]
dataset = "public"
role = "reader"
permissions = ["read"]
```

---

## Performance & Optimization

### Query Optimization

```toml
[query]
# Enable query optimization
optimize = true

# Query timeout (seconds)
timeout = 300

# Maximum result set size
max_results = 100000

# Enable query caching
cache = true

# Cache size (MB)
cache_size = 512

# Cache TTL (seconds)
cache_ttl = 3600

# Enable parallel query execution
parallel = true

# Number of threads for parallel execution
threads = 8

# Enable SIMD acceleration
simd = true
```

### Connection Pooling

```toml
[connection_pool]
# Minimum pool size
min_size = 2

# Maximum pool size
max_size = 32

# Connection timeout (seconds)
timeout = 30

# Idle timeout (seconds)
idle_timeout = 600

# Max lifetime (seconds)
max_lifetime = 3600
```

### Rate Limiting

```toml
[rate_limit]
# Enable rate limiting
enabled = true

# Requests per minute
requests_per_minute = 1000

# Burst size
burst = 100

# Rate limit by IP
by_ip = true

# Rate limit by user
by_user = true

# Whitelist IPs (no rate limit)
whitelist = [
    "127.0.0.1",
    "10.0.0.0/8",
]
```

### DDoS Protection

```toml
[ddos_protection]
# Enable DDoS protection
enabled = true

# Maximum concurrent connections per IP
max_connections_per_ip = 100

# Request rate threshold (requests/second)
request_rate_threshold = 1000

# Ban duration (seconds)
ban_duration = 3600

# Automatic IP banning
auto_ban = true
```

---

## Logging & Monitoring

### Logging Configuration

```toml
[logging]
# Log level
# Options: "trace", "debug", "info", "warn", "error"
level = "info"

# Log format
# Options: "text", "json", "pretty"
format = "text"

# Log output
# Options: "stdout", "stderr", "file"
output = "stdout"

# Log file path (if output = "file")
file = "/var/log/oxirs/server.log"

# Log rotation
[logging.rotation]
# Enable rotation
enabled = true

# Max file size (MB)
max_size = 100

# Max age (days)
max_age = 30

# Max backups
max_backups = 10

# Compress old logs
compress = true

# Structured logging fields
[logging.fields]
# Include timestamp
timestamp = true

# Include source location
source = false

# Include thread ID
thread_id = false

# Custom fields
environment = "production"
service = "oxirs-server"
```

### Metrics & Monitoring

```toml
[metrics]
# Enable metrics collection
enabled = true

# Metrics endpoint
endpoint = "/metrics"

# Metrics format
# Options: "prometheus", "json"
format = "prometheus"

# Collection interval (seconds)
interval = 60

# Metrics to collect
collect = [
    "requests",
    "queries",
    "updates",
    "errors",
    "latency",
    "memory",
    "cpu",
]

# Prometheus settings
[metrics.prometheus]
# Histogram buckets for latency (ms)
latency_buckets = [10, 50, 100, 250, 500, 1000, 2500, 5000]

# Enable process metrics
process_metrics = true

# Enable Go/Rust runtime metrics
runtime_metrics = true
```

### Health Checks

```toml
[health]
# Enable health checks
enabled = true

# Liveness probe endpoint
liveness_endpoint = "/health/live"

# Readiness probe endpoint
readiness_endpoint = "/health/ready"

# Startup probe endpoint
startup_endpoint = "/health/startup"

# Check interval (seconds)
interval = 10

# Timeout (seconds)
timeout = 5

# Checks to perform
checks = [
    "database",
    "memory",
    "disk",
]

# Database check settings
[health.database]
# Query timeout (seconds)
timeout = 5

# Test query
query = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
```

---

## Feature Flags

### Experimental Features

```toml
[features]
# Enable RDF-star (RDF 1.2)
rdf_star = true

# Enable SPARQL 1.2 features
sparql_1_2 = true

# Enable GraphQL endpoint
graphql = true

# Enable vector search
vector_search = false

# Enable spatial queries (GeoSPARQL)
spatial_queries = true

# Enable reasoning/inference
reasoning = false

# Enable streaming updates
streaming = false

# Enable federation
federation = true

# Enable text search
text_search = true
```

### Beta Features

```toml
[features.beta]
# Enable AI-powered query suggestions
query_suggestions = false

# Enable automatic query optimization
auto_optimize = false

# Enable query result caching
smart_cache = false

# Enable compression
compression = true
```

---

## Environment-Specific Profiles

### Development Profile

```toml
# dev.toml
[server]
host = "localhost"
port = 3030
workers = 2

[auth]
type = "none"

[logging]
level = "debug"
format = "pretty"

[[datasets]]
name = "dev"
location = "./data/dev"
type = "memory"
```

### Staging Profile

```toml
# staging.toml
[server]
host = "0.0.0.0"
port = 8080
workers = 4

[server.tls]
enabled = true
cert = "/etc/oxirs/certs/staging.pem"
key = "/etc/oxirs/certs/staging-key.pem"

[auth]
type = "jwt"

[auth.jwt]
secret = "${JWT_SECRET}"

[logging]
level = "info"
format = "json"
output = "file"
file = "/var/log/oxirs/staging.log"

[[datasets]]
name = "staging"
location = "/var/lib/oxirs/staging"
type = "tdb2"
```

### Production Profile

```toml
# production.toml
[server]
host = "0.0.0.0"
port = 443
workers = 16
timeout = 300

[server.tls]
enabled = true
cert = "/etc/oxirs/certs/prod.pem"
key = "/etc/oxirs/certs/prod-key.pem"
version = "tls13"

[server.cors]
enabled = true
allowed_origins = ["https://example.com"]

[auth]
type = "jwt"

[auth.jwt]
secret = "${JWT_SECRET}"
expiration = 3600
algorithm = "RS256"
public_key = "/etc/oxirs/keys/public.pem"

[auth.rbac]
enabled = true
default_role = "reader"

[rate_limit]
enabled = true
requests_per_minute = 10000
burst = 500

[ddos_protection]
enabled = true
max_connections_per_ip = 100
auto_ban = true

[logging]
level = "warn"
format = "json"
output = "file"
file = "/var/log/oxirs/production.log"

[logging.rotation]
enabled = true
max_size = 100
max_age = 90
compress = true

[metrics]
enabled = true
format = "prometheus"

[health]
enabled = true

[[datasets]]
name = "production"
location = "/var/lib/oxirs/production"
type = "tdb2"
readonly = false

[datasets.cache]
result_cache = 2048
pattern_cache = 1024
ttl = 7200

[datasets.performance]
batch_size = 50000
parallel = true
threads = 16
```

---

## Examples

### Example 1: Simple Local Server

```toml
# oxirs-local.toml
[server]
host = "localhost"
port = 3030

[[datasets]]
name = "mydata"
location = "./data/mydata"
type = "tdb2"

[logging]
level = "info"
```

**Start server:**
```bash
oxirs serve --config oxirs-local.toml
```

### Example 2: Secure Production Server

```toml
# oxirs-secure.toml
[server]
host = "0.0.0.0"
port = 443
workers = 8

[server.tls]
enabled = true
cert = "/etc/letsencrypt/live/example.com/fullchain.pem"
key = "/etc/letsencrypt/live/example.com/privkey.pem"

[auth]
type = "jwt"

[auth.jwt]
secret = "${JWT_SECRET}"

[rate_limit]
enabled = true
requests_per_minute = 5000

[[datasets]]
name = "prod"
location = "/data/oxirs/prod"
type = "tdb2"
```

### Example 3: High-Performance Configuration

```toml
# oxirs-perf.toml
[server]
host = "0.0.0.0"
port = 8080
workers = 32
keep_alive = 120

[query]
optimize = true
parallel = true
threads = 16
simd = true
cache = true
cache_size = 4096

[connection_pool]
max_size = 128

[[datasets]]
name = "bigdata"
location = "/fast-ssd/oxirs/bigdata"
type = "tdb2"

[datasets.cache]
result_cache = 4096
pattern_cache = 2048

[datasets.performance]
batch_size = 100000
parallel = true
threads = 16
```

---

## Environment Variables

Configuration values can use environment variable substitution:

```toml
[server]
host = "${SERVER_HOST:-localhost}"
port = "${SERVER_PORT:-3030}"

[auth.jwt]
secret = "${JWT_SECRET}"

[logging]
level = "${LOG_LEVEL:-info}"
file = "${LOG_FILE:-/var/log/oxirs/server.log}"
```

**Set environment variables:**
```bash
export SERVER_HOST="0.0.0.0"
export SERVER_PORT="8080"
export JWT_SECRET="your-secret-key"
export LOG_LEVEL="debug"
```

---

## Validation

### Validate Configuration

```bash
# Validate config file
oxirs config validate oxirs.toml

# Validate with environment variables
JWT_SECRET=test oxirs config validate oxirs.toml

# Show expanded configuration
oxirs config show oxirs.toml
```

### Common Validation Errors

**Error**: Missing required field
```
Error: Missing required field 'server.host'
```

**Fix**: Add required field to configuration.

**Error**: Invalid value
```
Error: Invalid log level 'verbose'. Valid options: trace, debug, info, warn, error
```

**Fix**: Use valid value from allowed options.

**Error**: Undefined environment variable
```
Error: Environment variable 'JWT_SECRET' is not set
```

**Fix**: Set environment variable before starting server.

---

## See Also

- [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md) - Command reference
- [README.md](../README.md) - Getting started
- [oxirs.toml.example](../oxirs.toml.example) - Example configuration

---

**OxiRS v0.1.0-rc.1** - Production-ready semantic web server configuration
