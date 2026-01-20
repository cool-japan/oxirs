# IDS Connector Operations Manual

**OxiRS IDS Connector - Production Operations Guide**

Version: 0.1.0
Last Updated: 2026-01-06
Status: Production Ready

## Table of Contents

1. [Deployment](#deployment)
2. [Configuration](#configuration)
3. [Operations](#operations)
4. [Monitoring & Alerting](#monitoring--alerting)
5. [Troubleshooting](#troubleshooting)
6. [Security Operations](#security-operations)
7. [Backup & Recovery](#backup--recovery)
8. [Incident Response](#incident-response)

---

## Deployment

### Prerequisites

**System Requirements:**
- OS: Linux (Ubuntu 22.04+, RHEL 8+) or macOS 12+
- CPU: 4+ cores (recommended: 8+ cores)
- RAM: 8 GB minimum (recommended: 16 GB+)
- Storage: 50 GB minimum (recommended: 100 GB+ for persistent lineage)
- Network: Stable internet connection with HTTPS access

**Required External Services:**
- **DAPS (Dynamic Attribute Provisioning Service)** - For connector authentication
- **IDS Metadata Broker** (optional) - For catalog federation
- **Gaia-X Registry** (optional) - For participant verification

**Rust Toolchain:**
```bash
# Install Rust 1.75 or later
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update
```

### Installation Methods

#### Method 1: From Source (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs

# Build with IDS support
cargo build --release -p oxirs-fuseki

# Run with IDS configuration
./target/release/oxirs-fuseki --config oxirs-ids.toml
```

#### Method 2: Docker Container

```bash
# Build Docker image
docker build -t oxirs-fuseki:ids .

# Run container with IDS configuration
docker run -d \
  --name oxirs-fuseki-ids \
  -p 3030:3030 \
  -v $(pwd)/oxirs-ids.toml:/etc/oxirs/config.toml \
  -v $(pwd)/certs:/etc/oxirs/certs \
  -v $(pwd)/data:/data/oxirs \
  oxirs-fuseki:ids --config /etc/oxirs/config.toml
```

#### Method 3: Kubernetes Deployment

```yaml
# oxirs-ids-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-fuseki-ids
  namespace: data-space
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oxirs-fuseki-ids
  template:
    metadata:
      labels:
        app: oxirs-fuseki-ids
    spec:
      containers:
      - name: oxirs-fuseki
        image: oxirs-fuseki:ids-0.1.0
        ports:
        - containerPort: 3030
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: RUST_LOG
          value: "info"
        volumeMounts:
        - name: config
          mountPath: /etc/oxirs
        - name: certs
          mountPath: /etc/oxirs/certs
        - name: data
          mountPath: /data/oxirs
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /$/ping
            port: 3030
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /$/ping
            port: 3030
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: oxirs-ids-config
      - name: certs
        secret:
          secretName: oxirs-ids-certs
      - name: data
        persistentVolumeClaim:
          claimName: oxirs-ids-data
---
apiVersion: v1
kind: Service
metadata:
  name: oxirs-fuseki-ids
  namespace: data-space
spec:
  selector:
    app: oxirs-fuseki-ids
  ports:
  - name: http
    port: 3030
    targetPort: 3030
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f oxirs-ids-deployment.yaml
kubectl get pods -n data-space -w
```

### TLS Certificate Setup

#### Generate Self-Signed Certificates (Development)

```bash
# Generate connector certificate
openssl req -newkey rsa:4096 -nodes -keyout connector.key \
  -x509 -days 365 -out connector.crt \
  -subj "/CN=oxirs-connector/O=YourOrganization/C=EU"

# Verify certificate
openssl x509 -in connector.crt -text -noout
```

#### Production Certificates

**Option 1: Let's Encrypt (for public connectors)**
```bash
certbot certonly --standalone -d connector.your-domain.com
# Certificates stored in: /etc/letsencrypt/live/connector.your-domain.com/
```

**Option 2: Corporate PKI (for internal connectors)**
- Request certificate from your organization's Certificate Authority
- Ensure certificate includes:
  - Subject: Connector ID or domain name
  - Key Usage: Digital Signature, Key Encipherment
  - Extended Key Usage: TLS Web Server Authentication, TLS Web Client Authentication

### DAPS Registration

**Step 1: Obtain DAPS Client Credentials**

Visit: https://daps.aisec.fraunhofer.de/register

Provide:
- Organization details
- Connector certificate (PEM format)
- Use case description
- Expected number of connectors

You will receive:
- Client ID: `YOUR_CLIENT_ID`
- Client Secret: Embedded in certificate

**Step 2: Test DAPS Connection**

```bash
# Using curl
curl -X POST https://daps.aisec.fraunhofer.de/v2/token \
  --cert connector.crt \
  --key connector.key \
  -d "grant_type=client_credentials" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "scope=idsc:IDS_CONNECTOR_ATTRIBUTES_ALL"

# Expected response:
# {
#   "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
#   "token_type": "Bearer",
#   "expires_in": 3600
# }
```

---

## Configuration

### Minimal Production Configuration

```toml
# oxirs-ids-production.toml
[server]
port = 3030
host = "0.0.0.0"
admin_ui = false  # Disable in production
cors = false
max_connections = 5000
request_timeout_secs = 60
graceful_shutdown_timeout_secs = 30

[server.tls]
cert_path = "/etc/oxirs/certs/connector.crt"
key_path = "/etc/oxirs/certs/connector.key"
require_client_cert = false

[security]
auth_required = true

[security.jwt]
secret = "${JWT_SECRET}"  # From environment variable
issuer = "oxirs-fuseki-ids"
audience = "oxirs-api"
expiration_secs = 3600

[security.authentication]
enabled = true

[[security.users]]
username = "admin"
password_hash = "${ADMIN_PASSWORD_HASH}"
roles = ["admin", "ids_provider", "ids_consumer"]

[[security.users]]
username = "ids_api"
password_hash = "${API_PASSWORD_HASH}"
roles = ["ids_consumer"]

[monitoring]
enabled = true
prometheus_port = 9090

[monitoring.opentelemetry]
enabled = true
endpoint = "http://otel-collector:4317"
service_name = "oxirs-fuseki-ids-prod"
service_version = "0.1.0"

[logging]
level = "info"
format = "json"

[performance]
max_query_complexity = 1000
query_timeout_secs = 30
max_result_size_mb = 100
enable_query_caching = true

[http_protocol]
http2_enabled = true
http3_enabled = false
sparql_optimized = true

[datasets.default]
type = "persistent"
storage_path = "/data/oxirs/tdb"

[datasets.ids_metadata]
type = "persistent"
storage_path = "/data/oxirs/ids_metadata"
description = "IDS metadata catalog, contracts, and lineage"
```

### Environment Variables

Create `.env` file for sensitive configuration:

```bash
# .env
JWT_SECRET=your-generated-jwt-secret-minimum-32-chars
ADMIN_PASSWORD_HASH=$(echo -n "your-admin-password" | argon2 somesalt -id -t 2 -m 19456 -p 1 -l 32)
API_PASSWORD_HASH=$(echo -n "your-api-password" | argon2 somesalt -id -t 2 -m 19456 -p 1 -l 32)

# DAPS Configuration (future)
DAPS_URL=https://daps.aisec.fraunhofer.de
DAPS_CLIENT_ID=YOUR_CLIENT_ID
DAPS_CLIENT_CERT=/etc/oxirs/certs/daps-client.crt
DAPS_CLIENT_KEY=/etc/oxirs/certs/daps-client.key

# Broker Configuration (future)
IDS_BROKER_URL=https://broker.ids.isst.fraunhofer.de
IDS_BROKER_AUTO_REGISTER=true

# Gaia-X Configuration (future)
GAIAX_REGISTRY_URL=https://registry.gaia-x.eu
GAIAX_PARTICIPANT_ID=YOUR_PARTICIPANT_ID
```

Load environment variables:
```bash
set -a
source .env
set +a
```

---

## Operations

### Starting the Connector

#### Standalone

```bash
# Start with configuration file
./oxirs-fuseki --config oxirs-ids.toml

# Start with environment variables
RUST_LOG=info ./oxirs-fuseki --config oxirs-ids.toml

# Start in background
nohup ./oxirs-fuseki --config oxirs-ids.toml > oxirs.log 2>&1 &
```

#### Systemd Service

Create `/etc/systemd/system/oxirs-fuseki-ids.service`:

```ini
[Unit]
Description=OxiRS Fuseki IDS Connector
After=network.target

[Service]
Type=simple
User=oxirs
Group=oxirs
WorkingDirectory=/opt/oxirs
Environment=RUST_LOG=info
EnvironmentFile=/opt/oxirs/.env
ExecStart=/opt/oxirs/bin/oxirs-fuseki --config /opt/oxirs/config/oxirs-ids.toml
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=oxirs-fuseki-ids

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/data/oxirs

[Install]
WantedBy=multi-user.target
```

Manage service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable oxirs-fuseki-ids
sudo systemctl start oxirs-fuseki-ids
sudo systemctl status oxirs-fuseki-ids

# View logs
sudo journalctl -u oxirs-fuseki-ids -f
```

### Common Operations

#### 1. Check Connector Status

```bash
# Health check
curl http://localhost:3030/$/ping

# IDS connector info
curl http://localhost:3030/api/ids/connector

# Response:
# {
#   "connectorId": "urn:ids:connector:oxirs",
#   "title": "OxiRS IDS Connector",
#   "description": "Semantic Web Data Space Connector",
#   "securityProfile": "TrustSecurityProfile",
#   "supportedProtocols": ["Https", "MultipartFormData"],
#   "version": "4.2.7"
# }
```

#### 2. Add Resource to Catalog

```bash
curl -X POST http://localhost:3030/api/ids/catalog/resources \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "id": "urn:ids:resource:enterprise-knowledge-base",
    "title": "Enterprise Knowledge Base",
    "description": "Semantic knowledge graph for enterprise data",
    "keywords": ["rdf", "knowledge-graph", "ontology"],
    "publisher": "urn:ids:connector:oxirs"
  }'
```

#### 3. Initiate Contract Negotiation

```bash
curl -X POST http://localhost:3030/api/ids/contracts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "providerId": "urn:ids:connector:provider",
    "assetId": "urn:ids:resource:enterprise-knowledge-base",
    "consumer": {
      "id": "urn:ids:connector:consumer",
      "name": "Consumer Organization",
      "legalName": "Consumer Inc.",
      "description": "Data consumer"
    }
  }'

# Response:
# {
#   "contractId": "urn:ids:contract:abc123...",
#   "state": "Negotiating",
#   ...
# }
```

#### 4. Accept Contract

```bash
curl -X POST http://localhost:3030/api/ids/contracts/urn:ids:contract:abc123/accept \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Response:
# {
#   "contractId": "urn:ids:contract:abc123",
#   "state": "Accepted",
#   ...
# }
```

#### 5. Initiate Data Transfer

```bash
curl -X POST http://localhost:3030/api/ids/transfers \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "contractId": "urn:ids:contract:abc123",
    "source": {
      "type": "Https",
      "url": "https://provider.com/data/export"
    },
    "destination": {
      "type": "Https",
      "url": "https://consumer.com/data/import"
    },
    "protocol": "Https"
  }'
```

#### 6. Register with IDS Broker

```bash
curl -X POST http://localhost:3030/api/ids/broker/register \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "connectorId": "urn:ids:connector:oxirs",
    "title": "OxiRS Production Connector",
    "description": "Production IDS connector for enterprise data space",
    "curator": "https://your-organization.com",
    "maintainer": "https://your-organization.com",
    "endpoints": [
      {
        "endpointType": "IdsConnector",
        "url": "https://connector.your-domain.com",
        "protocol": "HTTPS"
      }
    ]
  }'
```

#### 7. Query Broker Catalog

```bash
curl -X POST http://localhost:3030/api/ids/broker/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "keywords": ["automotive", "supply-chain"],
    "limit": 10
  }'
```

### Graceful Shutdown

```bash
# Send SIGTERM for graceful shutdown
kill -SIGTERM $(pgrep oxirs-fuseki)

# Or with systemd
sudo systemctl stop oxirs-fuseki-ids

# Wait for active transfers to complete (max 30 seconds by default)
```

The connector will:
1. Stop accepting new requests
2. Complete active contract negotiations
3. Finish ongoing data transfers
4. Flush lineage data
5. Close database connections
6. Shut down after `graceful_shutdown_timeout_secs`

---

## Monitoring & Alerting

### Health Checks

#### Kubernetes Liveness Probe
```yaml
livenessProbe:
  httpGet:
    path: /$/ping
    port: 3030
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

#### Kubernetes Readiness Probe
```yaml
readinessProbe:
  httpGet:
    path: /$/ping
    port: 3030
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

### Prometheus Metrics

**Metrics Endpoint:** `http://localhost:9090/metrics`

**Key Metrics to Monitor:**

```promql
# Request rate
rate(oxirs_http_requests_total[5m])

# Request duration (p50, p95, p99)
histogram_quantile(0.50, rate(oxirs_http_request_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(oxirs_http_request_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(oxirs_http_request_duration_seconds_bucket[5m]))

# Error rate
rate(oxirs_http_requests_total{status=~"5.."}[5m])

# IDS-specific metrics
oxirs_ids_contracts_total{state="Active"}
oxirs_ids_transfers_total
rate(oxirs_ids_policy_evaluations_total[5m])
oxirs_ids_catalog_resources_total

# Resource usage
process_cpu_seconds_total
process_resident_memory_bytes
```

### Grafana Dashboard

Example dashboard JSON:

```json
{
  "dashboard": {
    "title": "OxiRS IDS Connector",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(oxirs_http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Active Contracts",
        "targets": [
          {
            "expr": "oxirs_ids_contracts_total{state='Active'}"
          }
        ]
      },
      {
        "title": "Policy Evaluation Rate",
        "targets": [
          {
            "expr": "rate(oxirs_ids_policy_evaluations_total[5m])"
          }
        ]
      },
      {
        "title": "Transfer Throughput",
        "targets": [
          {
            "expr": "rate(oxirs_ids_transfer_bytes_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

**Prometheus AlertManager Rules:**

```yaml
groups:
- name: oxirs_ids_alerts
  interval: 30s
  rules:
  - alert: HighErrorRate
    expr: rate(oxirs_http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate on OxiRS IDS Connector"
      description: "Error rate is {{ $value }} errors/sec"

  - alert: ContractNegotiationTimeout
    expr: oxirs_ids_contract_negotiation_duration_seconds > 300
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "Contract negotiation taking too long"
      description: "Negotiation duration: {{ $value }}s"

  - alert: DAPSTokenExpiringSoon
    expr: oxirs_ids_daps_token_expiry_seconds < 300
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "DAPS token expiring soon"
      description: "Token expires in {{ $value }}s"

  - alert: PolicyEvaluationFailureRate
    expr: rate(oxirs_ids_policy_evaluations_total{result="Deny"}[5m]) > 0.5
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High policy denial rate"
      description: "{{ $value }} denials/sec"

  - alert: HighMemoryUsage
    expr: process_resident_memory_bytes > 8e9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage on OxiRS"
      description: "Memory usage: {{ $value | humanize }}B"
```

### Logging

**Log Levels:**
- `ERROR` - Critical failures requiring immediate attention
- `WARN` - Non-critical issues (e.g., policy denials, expired tokens)
- `INFO` - Operational events (e.g., contract negotiation, transfers)
- `DEBUG` - Detailed debugging information
- `TRACE` - Very detailed execution traces

**Structured Logging (JSON format):**

```json
{
  "timestamp": "2026-01-06T10:30:45.123Z",
  "level": "INFO",
  "target": "oxirs_fuseki::ids::contract",
  "message": "Contract negotiation initiated",
  "fields": {
    "contract_id": "urn:ids:contract:abc123",
    "provider_id": "urn:ids:connector:provider",
    "consumer_id": "urn:ids:connector:consumer",
    "asset_id": "urn:ids:resource:enterprise-knowledge-base"
  }
}
```

**Log Aggregation with ELK Stack:**

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/oxirs/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "oxirs-ids-%{+yyyy.MM.dd}"

setup.kibana:
  host: "localhost:5601"
```

---

## Troubleshooting

### Common Issues

#### Issue 1: DAPS Authentication Failure

**Symptoms:**
- HTTP 401 errors from DAPS
- Log: `ERROR: DAPS token acquisition failed`

**Diagnosis:**
```bash
# Test DAPS connectivity
curl -v https://daps.aisec.fraunhofer.de/v2/token \
  --cert /etc/oxirs/certs/connector.crt \
  --key /etc/oxirs/certs/connector.key

# Check certificate validity
openssl x509 -in /etc/oxirs/certs/connector.crt -noout -dates
```

**Solutions:**
1. Verify certificate not expired
2. Check client ID matches DAPS registration
3. Ensure certificate chain is complete
4. Verify network connectivity to DAPS

#### Issue 2: Contract Negotiation Timeout

**Symptoms:**
- Contracts stuck in "Negotiating" state
- Log: `WARN: Contract negotiation timeout`

**Diagnosis:**
```bash
# Check active contracts
curl http://localhost:3030/api/ids/contracts \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" | jq '.[] | select(.state == "Negotiating")'

# Check policy evaluation logs
journalctl -u oxirs-fuseki-ids | grep "policy_evaluation"
```

**Solutions:**
1. Review ODRL policy constraints (may be too restrictive)
2. Check consumer/provider connectivity
3. Verify both parties have valid DAPS tokens
4. Increase `MAX_NEGOTIATION_ROUNDS` if needed

#### Issue 3: Policy Evaluation Always Denies

**Symptoms:**
- All data access requests denied
- Log: `INFO: Policy evaluation result: Deny`

**Diagnosis:**
```bash
# Test policy evaluation
curl -X POST http://localhost:3030/api/ids/policy/evaluate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "policyUri": "urn:ids:policy:test",
    "action": "USE",
    "resource": "urn:ids:resource:test",
    "context": {
      "connectorId": "urn:ids:connector:consumer",
      "currentTime": "2026-01-06T10:00:00Z"
    }
  }'
```

**Solutions:**
1. Check policy constraints (temporal, connector, etc.)
2. Verify connector ID in allowlist
3. Check usage count hasn't exceeded limit
4. Ensure contract is Active and not expired

#### Issue 4: High Memory Usage

**Symptoms:**
- Memory usage > 8 GB
- Out of memory errors

**Diagnosis:**
```bash
# Check memory usage
ps aux | grep oxirs-fuseki
top -p $(pgrep oxirs-fuseki)

# Check active transfers
curl http://localhost:3030/api/ids/transfers \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" | jq 'length'
```

**Solutions:**
1. Reduce `max_connections` in config
2. Enable `enable_query_caching = false` temporarily
3. Limit concurrent transfers
4. Increase available memory
5. Check for memory leaks (enable `RUST_BACKTRACE=1`)

#### Issue 5: Broker Registration Fails

**Symptoms:**
- HTTP 400/500 errors from broker
- Log: `ERROR: Broker registration failed`

**Diagnosis:**
```bash
# Test broker connectivity
curl -v https://broker.ids.isst.fraunhofer.de/infrastructure

# Check self-description format
curl http://localhost:3030/api/ids/connector/self-description \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Solutions:**
1. Verify broker URL is correct
2. Check self-description conforms to IDS Information Model
3. Ensure connector is registered with DAPS first
4. Verify network connectivity to broker
5. Check broker status (may be under maintenance)

### Debugging Tools

#### Enable Debug Logging

```bash
RUST_LOG=oxirs_fuseki::ids=debug ./oxirs-fuseki --config oxirs-ids.toml
```

#### Trace Specific Modules

```bash
RUST_LOG=oxirs_fuseki::ids::contract=trace,oxirs_fuseki::ids::policy=trace \
  ./oxirs-fuseki --config oxirs-ids.toml
```

#### Capture Network Traffic

```bash
# Capture HTTPS traffic to DAPS
sudo tcpdump -i any -s 0 -w daps-traffic.pcap \
  'host daps.aisec.fraunhofer.de and port 443'

# Analyze with Wireshark
wireshark daps-traffic.pcap
```

---

## Security Operations

### Security Checklist

- [ ] TLS 1.2+ enabled for all connections
- [ ] Strong cipher suites configured
- [ ] Certificate rotation automated
- [ ] DAPS tokens cached with TTL
- [ ] API authentication enabled (JWT)
- [ ] Role-based access control configured
- [ ] Audit logging enabled
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Input validation on all endpoints

### Certificate Rotation

**Automated Rotation with certbot:**

```bash
# Renew certificates
certbot renew --dry-run

# Hook for oxirs restart
cat > /etc/letsencrypt/renewal-hooks/deploy/oxirs-reload.sh << 'EOF'
#!/bin/bash
systemctl reload oxirs-fuseki-ids
EOF
chmod +x /etc/letsencrypt/renewal-hooks/deploy/oxirs-reload.sh
```

**Manual Rotation:**

```bash
# 1. Generate new certificate
openssl req -newkey rsa:4096 -nodes -keyout connector-new.key \
  -x509 -days 365 -out connector-new.crt \
  -subj "/CN=oxirs-connector/O=YourOrganization/C=EU"

# 2. Update configuration
sed -i 's|connector.crt|connector-new.crt|' /opt/oxirs/config/oxirs-ids.toml
sed -i 's|connector.key|connector-new.key|' /opt/oxirs/config/oxirs-ids.toml

# 3. Reload service (graceful reload without downtime)
systemctl reload oxirs-fuseki-ids

# 4. Verify new certificate in use
openssl s_client -connect localhost:3030 -showcerts
```

### Audit Log Analysis

**Key Events to Audit:**

1. Authentication failures
2. Contract negotiations (all states)
3. Policy denials
4. Data transfers
5. Configuration changes
6. Certificate rotations

**Query Audit Logs:**

```bash
# Recent authentication failures
journalctl -u oxirs-fuseki-ids --since "1 hour ago" | grep "authentication failed"

# Contract state changes
journalctl -u oxirs-fuseki-ids --since "today" | grep "contract_state_change"

# Policy denials
journalctl -u oxirs-fuseki-ids --since "today" | grep "policy_decision: Deny"
```

---

## Backup & Recovery

### Backup Strategy

**What to Backup:**

1. **Configuration Files**
   - `/opt/oxirs/config/oxirs-ids.toml`
   - `/opt/oxirs/.env`

2. **Certificates**
   - `/etc/oxirs/certs/connector.crt`
   - `/etc/oxirs/certs/connector.key`

3. **Persistent Data**
   - `/data/oxirs/tdb` - RDF triple store
   - `/data/oxirs/ids_metadata` - IDS metadata catalog

4. **Audit Logs** (optional)
   - `/var/log/oxirs/`

**Backup Script:**

```bash
#!/bin/bash
# backup-oxirs-ids.sh

BACKUP_DIR="/backup/oxirs-ids"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_PATH="$BACKUP_DIR/$TIMESTAMP"

mkdir -p "$BACKUP_PATH"

# Stop service for consistent backup
systemctl stop oxirs-fuseki-ids

# Backup configuration
cp -r /opt/oxirs/config "$BACKUP_PATH/"
cp /opt/oxirs/.env "$BACKUP_PATH/"

# Backup certificates
cp -r /etc/oxirs/certs "$BACKUP_PATH/"

# Backup data (with compression)
tar czf "$BACKUP_PATH/data.tar.gz" /data/oxirs/

# Restart service
systemctl start oxirs-fuseki-ids

# Retention: Keep last 7 days
find "$BACKUP_DIR" -type d -mtime +7 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_PATH"
```

**Automated Backup (cron):**

```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * /opt/oxirs/scripts/backup-oxirs-ids.sh >> /var/log/oxirs-backup.log 2>&1
```

### Recovery Procedure

**Scenario 1: Configuration Corruption**

```bash
# 1. Stop service
systemctl stop oxirs-fuseki-ids

# 2. Restore configuration from latest backup
LATEST_BACKUP=$(ls -t /backup/oxirs-ids/ | head -1)
cp -r /backup/oxirs-ids/$LATEST_BACKUP/config/* /opt/oxirs/config/
cp /backup/oxirs-ids/$LATEST_BACKUP/.env /opt/oxirs/

# 3. Verify configuration
/opt/oxirs/bin/oxirs-fuseki --config /opt/oxirs/config/oxirs-ids.toml --check

# 4. Restart service
systemctl start oxirs-fuseki-ids
```

**Scenario 2: Data Corruption**

```bash
# 1. Stop service
systemctl stop oxirs-fuseki-ids

# 2. Backup corrupted data
mv /data/oxirs /data/oxirs-corrupted-$(date +%Y%m%d)

# 3. Restore from backup
LATEST_BACKUP=$(ls -t /backup/oxirs-ids/ | head -1)
tar xzf /backup/oxirs-ids/$LATEST_BACKUP/data.tar.gz -C /

# 4. Restart service
systemctl start oxirs-fuseki-ids

# 5. Verify data integrity
curl http://localhost:3030/api/ids/catalog \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Scenario 3: Certificate Expiration**

```bash
# 1. Generate new certificate (do not stop service)
openssl req -newkey rsa:4096 -nodes -keyout /etc/oxirs/certs/connector-new.key \
  -x509 -days 365 -out /etc/oxirs/certs/connector-new.crt \
  -subj "/CN=oxirs-connector/O=YourOrganization/C=EU"

# 2. Update configuration
sed -i 's|connector.crt|connector-new.crt|' /opt/oxirs/config/oxirs-ids.toml
sed -i 's|connector.key|connector-new.key|' /opt/oxirs/config/oxirs-ids.toml

# 3. Graceful reload
systemctl reload oxirs-fuseki-ids

# 4. Re-register with DAPS using new certificate
# (follow DAPS registration procedure)
```

---

## Incident Response

### Incident Response Plan

#### Phase 1: Detection

**Monitoring Alerts:**
- High error rate
- Contract negotiation failures
- Policy evaluation anomalies
- DAPS authentication failures
- Unauthorized access attempts

**Initial Assessment:**
1. Confirm incident via multiple signals
2. Determine severity (P1: Critical, P2: High, P3: Medium, P4: Low)
3. Alert on-call team

#### Phase 2: Containment

**P1 (Critical) - Production Down:**
```bash
# 1. Isolate affected connector
systemctl stop oxirs-fuseki-ids

# 2. Preserve evidence
cp -r /var/log/oxirs /tmp/incident-$(date +%Y%m%d-%H%M%S)
cp -r /data/oxirs /tmp/incident-data-$(date +%Y%m%d-%H%M%S)

# 3. Analyze logs
journalctl -u oxirs-fuseki-ids --since "1 hour ago" > /tmp/incident-logs.txt

# 4. Notify stakeholders
```

**P2 (High) - Degraded Service:**
```bash
# 1. Reduce load
# Update config: max_connections = 100
# Restart: systemctl restart oxirs-fuseki-ids

# 2. Monitor impact
watch -n 5 'curl -s http://localhost:3030/$/ping'
```

#### Phase 3: Investigation

**Root Cause Analysis Checklist:**

- [ ] Review error logs for patterns
- [ ] Check recent configuration changes
- [ ] Verify certificate validity
- [ ] Test DAPS connectivity
- [ ] Review policy changes
- [ ] Check resource usage (CPU, memory, disk)
- [ ] Analyze network traffic
- [ ] Review recent deployments

#### Phase 4: Recovery

**Recovery Steps:**

1. Apply fix (configuration, code patch, certificate renewal)
2. Test fix in staging environment
3. Deploy fix to production
4. Monitor for 24 hours
5. Document incident and lessons learned

#### Phase 5: Post-Incident Review

**Post-Incident Report Template:**

```markdown
# Incident Report: [Incident ID]

## Summary
- Date: 2026-01-06
- Duration: 2 hours
- Severity: P1
- Impact: Full service outage

## Timeline
- 10:00 UTC: Alert triggered (high error rate)
- 10:05 UTC: On-call engineer paged
- 10:15 UTC: Incident confirmed (DAPS certificate expired)
- 10:30 UTC: Certificate renewed
- 11:00 UTC: Service restored
- 12:00 UTC: Incident resolved

## Root Cause
DAPS client certificate expired at 09:55 UTC. Automated renewal failed due to misconfigured certbot.

## Resolution
1. Manually renewed certificate
2. Updated certbot configuration
3. Added monitoring alert for certificate expiration

## Lessons Learned
- Need earlier certificate expiration alerts (30 days)
- Automate certificate renewal testing
- Document emergency certificate renewal procedure

## Action Items
- [ ] Set up alert 30 days before expiration
- [ ] Create runbook for emergency renewal
- [ ] Test certbot automation weekly
```

---

## Appendix

### API Reference

Complete API documentation: See `IDS_CERTIFICATION_GUIDE.md`

### Glossary

- **DAPS:** Dynamic Attribute Provisioning Service
- **ODRL:** Open Digital Rights Language
- **PROV-O:** Provenance Ontology
- **DCAT-AP:** Data Catalog Vocabulary - Application Profile
- **IDS:** International Data Spaces
- **IDSA:** International Data Spaces Association

### Support Contacts

- **Technical Support:** technical-support@your-organization.com
- **Security Incidents:** security@your-organization.com
- **IDSA Community:** https://internationaldataspaces.org/community/

---

**Document Version:** 1.0
**Approved By:** COOLJAPAN OU (Team Kitasan)
**Next Review:** Q2 2026
