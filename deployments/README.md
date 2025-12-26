# OxiRS Digital Twin Platform - Deployment Guide

This directory contains production deployment configurations for OxiRS Digital Twin Platform.

## Quick Start

### 1. Docker Compose Deployment (Recommended)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f oxirs-fuseki

# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes all data)
docker-compose down -v
```

**Services Started:**
- **OxiRS Fuseki**: SPARQL/NGSI-LD API at http://localhost:3030
- **Mosquitto MQTT**: MQTT broker at mqtt://localhost:1883
- **Prometheus**: Metrics at http://localhost:9090
- **Grafana**: Dashboards at http://localhost:3000 (admin/admin)

### 2. Docker Build Only

```bash
# Build image
docker build -t oxirs/fuseki:0.1.0-rc.1 .

# Run with default config
docker run -p 3030:3030 \
  -v oxirs-data:/data \
  oxirs/fuseki:0.1.0-rc.1

# Run with custom config
docker run -p 3030:3030 \
  -v $(pwd)/server/oxirs-fuseki/oxirs-digital-twin.toml:/data/config/oxirs.toml:ro \
  -v oxirs-data:/data \
  oxirs/fuseki:0.1.0-rc.1
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ OxiRS Fuseki │  │  Mosquitto   │  │ MQTT Bridge  │     │
│  │ :3030        │  │  MQTT Broker │  │ (Example)    │     │
│  │              │◄─┤  :1883       │◄─┤              │     │
│  └──────┬───────┘  └──────────────┘  └──────────────┘     │
│         │                                                   │
│         │ metrics                                           │
│         ▼                                                   │
│  ┌──────────────┐  ┌──────────────┐                       │
│  │ Prometheus   │  │   Grafana    │                       │
│  │ :9090        ├─►│   :3000      │                       │
│  └──────────────┘  └──────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration Files

### Mosquitto MQTT Broker

**Location**: `mosquitto/mosquitto.conf`

```conf
listener 1883
protocol mqtt

# Enable WebSocket
listener 9001
protocol websockets

# Development: allow anonymous
allow_anonymous true

# Production: require authentication
# allow_anonymous false
# password_file /mosquitto/config/passwd
```

**Generate password file (production)**:
```bash
docker run -it eclipse-mosquitto:2.0 mosquitto_passwd -c /tmp/passwd username
docker cp mosquitto:/tmp/passwd ./deployments/mosquitto/passwd
```

### Prometheus Monitoring

**Location**: `prometheus/prometheus.yml`

Scrapes metrics from:
- OxiRS Fuseki (http://oxirs-fuseki:3030/metrics)
- Prometheus itself (http://localhost:9090)

**Add custom alerts** (optional):
```yaml
# alerting_rules.yml
groups:
  - name: oxirs_alerts
    rules:
      - alert: HighQueryLatency
        expr: histogram_quantile(0.99, rate(oxirs_query_duration_seconds_bucket[5m])) > 1
        for: 5m
        annotations:
          summary: "OxiRS query latency is high"
```

### Grafana Dashboards

**Access**: http://localhost:3000 (admin/admin)

**Import OxiRS Dashboard**:
1. Go to Dashboards → Import
2. Upload `grafana/provisioning/dashboards/oxirs.json` (create from template)
3. Select Prometheus datasource

---

## Production Deployment

### 1. Environment Variables

Create `.env` file:

```bash
# OxiRS Configuration
RUST_LOG=info
OXIRS_DATA_DIR=/data/datasets
OXIRS_LOG_DIR=/data/logs

# MQTT Configuration
MQTT_BROKER=mosquitto:1883
MQTT_USERNAME=oxirs
MQTT_PASSWORD=secure_password_here

# Security
OXIRS_ADMIN_PASSWORD=change_me_in_production
IDS_DAPS_CLIENT_SECRET=your_daps_secret

# Monitoring
PROMETHEUS_RETENTION=30d
GRAFANA_ADMIN_PASSWORD=change_me_too
```

Use in docker-compose.yml:
```yaml
services:
  oxirs-fuseki:
    env_file: .env
```

### 2. TLS/HTTPS Configuration

**Mosquitto TLS**:
```conf
# mosquitto.conf
listener 8883
protocol mqtt
cafile /mosquitto/config/ca.crt
certfile /mosquitto/config/server.crt
keyfile /mosquitto/config/server.key
```

**OxiRS HTTPS** (use reverse proxy):
```yaml
# Add nginx reverse proxy
nginx:
  image: nginx:alpine
  ports:
    - "443:443"
  volumes:
    - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    - ./nginx/ssl:/etc/nginx/ssl:ro
```

### 3. Data Persistence

**Volumes created**:
- `oxirs-data`: RDF datasets
- `oxirs-logs`: Application logs
- `mosquitto-data`: MQTT persistence
- `prometheus-data`: Metrics storage
- `grafana-data`: Dashboards

**Backup strategy**:
```bash
# Backup all volumes
docker run --rm \
  -v oxirs-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/oxirs-data-$(date +%Y%m%d).tar.gz /data

# Restore from backup
docker run --rm \
  -v oxirs-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/oxirs-data-20250125.tar.gz -C /
```

### 4. Scaling with Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml oxirs

# Scale OxiRS instances
docker service scale oxirs_oxirs-fuseki=3

# View services
docker service ls
```

### 5. Kubernetes Deployment

See [`kubernetes/`](kubernetes/) directory for:
- Deployment manifests
- StatefulSet for persistence
- Service definitions
- Ingress configuration
- Helm charts

---

## Monitoring & Troubleshooting

### Health Checks

```bash
# OxiRS health
curl http://localhost:3030/$/ping

# MQTT broker
mosquitto_pub -h localhost -p 1883 -t test/topic -m "hello"

# Prometheus targets
curl http://localhost:9090/api/v1/targets
```

### Logs

```bash
# View all logs
docker-compose logs -f

# OxiRS only
docker-compose logs -f oxirs-fuseki

# Last 100 lines
docker-compose logs --tail=100 oxirs-fuseki

# Export logs
docker-compose logs oxirs-fuseki > oxirs-fuseki.log
```

### Metrics

**Prometheus Queries** (http://localhost:9090/graph):

```promql
# Query rate (queries per second)
rate(oxirs_queries_total[5m])

# P99 latency
histogram_quantile(0.99, rate(oxirs_query_duration_seconds_bucket[5m]))

# NGSI-LD entity count
oxirs_ngsi_ld_entities_total

# MQTT messages ingested
rate(oxirs_mqtt_messages_total[1m])
```

### Common Issues

**1. Port already in use**:
```bash
# Find process using port
lsof -i :3030

# Change port in docker-compose.yml
ports:
  - "3031:3030"
```

**2. Permission denied on volumes**:
```bash
# Fix ownership
sudo chown -R 1000:1000 /var/lib/docker/volumes/oxirs-data
```

**3. Out of memory**:
```yaml
# Add memory limits in docker-compose.yml
services:
  oxirs-fuseki:
    deploy:
      resources:
        limits:
          memory: 4G
```

---

## Use Case Examples

### Smart City Deployment

```yaml
# docker-compose.smart-city.yml
services:
  oxirs-fuseki:
    environment:
      - NGSI_LD_ENABLED=true
      - GEOSPARQL_ENABLED=true
    volumes:
      - ./configs/smart-city.toml:/data/config/oxirs.toml:ro
```

### Manufacturing Digital Twin

```yaml
# docker-compose.factory.yml
services:
  oxirs-fuseki:
    environment:
      - MQTT_BRIDGE_ENABLED=true
      - OPCUA_BRIDGE_ENABLED=true
    volumes:
      - ./configs/factory.toml:/data/config/oxirs.toml:ro

  mqtt-simulator:
    image: oxirs/mqtt-simulator
    environment:
      - MQTT_BROKER=mosquitto:1883
      - SIMULATION_CELLS=100
```

### Data Space Connector

```yaml
# docker-compose.data-space.yml
services:
  oxirs-fuseki:
    environment:
      - IDS_CONNECTOR_ENABLED=true
      - DAPS_URL=https://daps.aisec.fraunhofer.de
    volumes:
      - ./configs/data-space.toml:/data/config/oxirs.toml:ro
      - ./certs:/certs:ro
```

---

## Resources

- **Quick Start**: [`DIGITAL_TWIN_QUICKSTART.md`](../server/oxirs-fuseki/DIGITAL_TWIN_QUICKSTART.md)
- **IDS Certification**: [`IDS_CERTIFICATION_GUIDE.md`](../server/oxirs-fuseki/IDS_CERTIFICATION_GUIDE.md)
- **Main README**: [`README.md`](../README.md)
- **Configuration Reference**: [`oxirs-digital-twin.toml`](../server/oxirs-fuseki/oxirs-digital-twin.toml)

---

**Need Help?**
- GitHub Issues: https://github.com/cool-japan/oxirs/issues
- Documentation: https://docs.rs/oxirs
