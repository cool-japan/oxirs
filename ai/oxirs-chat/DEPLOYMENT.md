# OxiRS Chat - Production Deployment Guide

This guide covers deploying OxiRS Chat in production environments using Docker, Kubernetes, and cloud platforms.

## Table of Contents

- [Quick Start with Docker](#quick-start-with-docker)
- [Docker Compose Deployment](#docker-compose-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Platform Deployment](#cloud-platform-deployment)
- [Configuration](#configuration)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Best Practices](#security-best-practices)
- [Scaling and Performance](#scaling-and-performance)

## Quick Start with Docker

### Prerequisites

- Docker 20.10 or later
- Docker Compose 2.0 or later
- 4GB RAM minimum (8GB recommended)
- 10GB disk space

### 1. Build the Docker Image

```bash
cd /path/to/oxirs/ai/oxirs-chat
docker build -t oxirs-chat:latest -f Dockerfile ../..
```

### 2. Run with Docker

```bash
docker run -d \
  --name oxirs-chat \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY=your_key_here \
  -e RUST_LOG=info \
  oxirs-chat:latest
```

### 3. Test the Deployment

```bash
curl http://localhost:8080/health
```

## Docker Compose Deployment

### 1. Create Environment File

Create `.env` file:

```env
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...
GROQ_API_KEY=gsk_...
MISTRAL_API_KEY=...

# External Services
PUBMED_API_KEY=...

# Logging
RUST_LOG=info

# Grafana
GRAFANA_PASSWORD=secure_password_here
```

### 2. Start Services

```bash
docker-compose up -d
```

### 3. Access Services

- **OxiRS Chat API**: http://localhost:8080
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### 4. Load Dataset

```bash
# Copy RDF dataset to datasets directory
cp your-dataset.ttl datasets/

# Restart with dataset
docker-compose restart oxirs-chat
```

## Kubernetes Deployment

### 1. Create Kubernetes Manifests

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oxirs-chat
  labels:
    app: oxirs-chat
spec:
  replicas: 3
  selector:
    matchLabels:
      app: oxirs-chat
  template:
    metadata:
      labels:
        app: oxirs-chat
    spec:
      containers:
      - name: oxirs-chat
        image: oxirs-chat:latest
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: RUST_LOG
          value: "info"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: oxirs-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: config
          mountPath: /app/config
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: oxirs-data-pvc
      - name: config
        configMap:
          name: oxirs-config
---
apiVersion: v1
kind: Service
metadata:
  name: oxirs-chat
spec:
  selector:
    app: oxirs-chat
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

**secrets.yaml:**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: oxirs-secrets
type: Opaque
stringData:
  openai-api-key: "sk-..."
  anthropic-api-key: "sk-ant-..."
```

### 2. Deploy to Kubernetes

```bash
kubectl apply -f secrets.yaml
kubectl apply -f deployment.yaml
```

### 3. Scale the Deployment

```bash
kubectl scale deployment oxirs-chat --replicas=5
```

## Cloud Platform Deployment

### AWS ECS

```bash
# Build and push to ECR
aws ecr create-repository --repository-name oxirs-chat
docker tag oxirs-chat:latest <account>.dkr.ecr.us-east-1.amazonaws.com/oxirs-chat:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/oxirs-chat:latest

# Create ECS task definition and service
aws ecs create-service --cli-input-json file://ecs-service.json
```

### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/<project>/oxirs-chat
gcloud run deploy oxirs-chat \
  --image gcr.io/<project>/oxirs-chat \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars RUST_LOG=info
```

### Azure Container Instances

```bash
az container create \
  --resource-group oxirs-rg \
  --name oxirs-chat \
  --image oxirs-chat:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8080 \
  --environment-variables RUST_LOG=info
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Logging level | `info` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `COHERE_API_KEY` | Cohere API key | - |
| `GROQ_API_KEY` | Groq API key | - |
| `MISTRAL_API_KEY` | Mistral API key | - |
| `PUBMED_API_KEY` | PubMed API key | - |
| `OXIRS_CONFIG` | Config file path | `/app/config/oxirs.toml` |

### Configuration File (oxirs.toml)

```toml
[server]
host = "0.0.0.0"
port = 8080
max_connections = 1000
session_timeout = 3600

[llm]
default_provider = "openai"
temperature = 0.7
max_tokens = 2000

[rag]
enable_quantum = true
enable_consciousness = true
vector_search_enabled = true

[cache]
enabled = true
redis_url = "redis://redis:6379"

[monitoring]
enable_metrics = true
enable_tracing = true
```

## Monitoring and Observability

### Metrics

Access Prometheus metrics at `/metrics`:

- `oxirs_chat_requests_total` - Total requests
- `oxirs_chat_request_duration_seconds` - Request latency
- `oxirs_chat_sessions_active` - Active sessions
- `oxirs_chat_llm_calls_total` - LLM API calls
- `oxirs_chat_rag_retrieval_duration` - RAG retrieval time

### Logging

Structured JSON logs with tracing spans:

```json
{
  "timestamp": "2025-12-25T10:30:00Z",
  "level": "INFO",
  "target": "oxirs_chat",
  "message": "Processing message",
  "session_id": "abc123",
  "span": {
    "name": "process_message",
    "duration_ms": 234
  }
}
```

### Grafana Dashboards

Import pre-built dashboards:

```bash
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana-dashboard.json
```

## Security Best Practices

### 1. Use Secrets Management

```bash
# Kubernetes Secrets
kubectl create secret generic oxirs-secrets \
  --from-literal=openai-api-key=$OPENAI_API_KEY

# AWS Secrets Manager
aws secretsmanager create-secret \
  --name oxirs/api-keys \
  --secret-string file://secrets.json
```

### 2. Enable TLS/SSL

```yaml
# Ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: oxirs-chat-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - chat.example.com
    secretName: oxirs-tls
  rules:
  - host: chat.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: oxirs-chat
            port:
              number: 80
```

### 3. Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: oxirs-chat-netpol
spec:
  podSelector:
    matchLabels:
      app: oxirs-chat
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
```

### 4. Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: oxirs-quota
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
```

## Scaling and Performance

### Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: oxirs-chat-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: oxirs-chat
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Performance Tuning

```toml
[performance]
# Connection pooling
max_connections = 1000
connection_timeout = 30

# Caching
enable_semantic_cache = true
cache_ttl = 3600

# Concurrency
max_concurrent_requests = 100
request_queue_size = 500
```

### Load Testing

```bash
# Using k6
k6 run --vus 100 --duration 5m load-test.js

# Using wrk
wrk -t12 -c400 -d30s http://localhost:8080/api/chat
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Increase memory limits
   - Enable session cleanup
   - Adjust cache size

2. **High Latency**
   - Check LLM provider status
   - Enable Redis caching
   - Scale horizontally

3. **Connection Timeouts**
   - Increase timeout settings
   - Check network policies
   - Verify firewall rules

### Debug Mode

```bash
# Enable debug logging
docker run -e RUST_LOG=debug oxirs-chat:latest

# Attach to running container
docker exec -it oxirs-chat /bin/sh
```

## Support

- **Documentation**: https://docs.oxirs.org
- **Issues**: https://github.com/cool-japan/oxirs/issues
- **Discord**: https://discord.gg/oxirs

## License

Copyright © 2025 OxiRS Team. Licensed under MIT OR Apache-2.0.
