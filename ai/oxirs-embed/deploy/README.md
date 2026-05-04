# OxiRS Embed — Deployment Templates

Production deployment artifacts for the OxiRS Embed knowledge-graph
embedding service.  This directory is **operational** documentation; the
crate's developer-facing docs live in `../README.md` and `../TODO.md`.

## Layout

```
deploy/
├── Dockerfile                    # multi-stage build, scratch runtime, ~30 MB image
├── docker-compose.yml            # local stack: oxirs-embed + Prometheus + Grafana
├── README.md                     # (this file)
├── helm/oxirs-embed/             # Helm chart (Chart.yaml + values.yaml + templates/)
├── k8s/                          # raw K8s manifests for clusters without Helm
└── monitoring/                   # prometheus.yml + grafana-dashboard.json
```

## Local stack (Docker Compose)

```sh
# from repo root
docker compose -f ai/oxirs-embed/deploy/docker-compose.yml up --build
```

Endpoints:

| Component   | URL                                |
|-------------|------------------------------------|
| Embedding API | <http://localhost:8080>          |
| Prometheus  | <http://localhost:9091>            |
| Grafana     | <http://localhost:3000> (admin/admin) |

The Grafana dashboard `OxiRS Embed — Service Overview` is auto-loaded from
`monitoring/grafana-dashboard.json`.

## Kubernetes — Helm

```sh
# Install
helm install oxirs-embed ai/oxirs-embed/deploy/helm/oxirs-embed \
  --namespace oxirs --create-namespace

# Upgrade with custom values
helm upgrade oxirs-embed ai/oxirs-embed/deploy/helm/oxirs-embed \
  --namespace oxirs \
  --values my-values.yaml

# Render templates without applying (smoke check)
helm template oxirs-embed ai/oxirs-embed/deploy/helm/oxirs-embed
```

Key `values.yaml` knobs:

| Key                              | Default       | Notes                                    |
|----------------------------------|---------------|------------------------------------------|
| `image.tag`                      | `0.3.0`       | match the chart `appVersion`             |
| `replicaCount`                   | `2`           | ignored when `autoscaling.enabled`       |
| `distributed.enabled`            | `false`       | enable parameter-server training         |
| `distributed.numShards`          | `4`           | bounded 4-8 for the prototype            |
| `distributed.numWorkers`         | `4`           | bounded 4-8 for the prototype            |
| `distributed.updateMode`         | `async`       | `async` or `sync`                        |
| `distributed.shardingStrategy`   | `entity-hash` | `entity-hash` or `round-robin`           |
| `autoscaling.enabled`            | `true`        | HPA on CPU+memory                        |
| `podDisruptionBudget.enabled`    | `true`        | minAvailable: 1                          |
| `metrics.serviceMonitor.enabled` | `false`       | flip to `true` if running prometheus-operator |

### Parameter-server training mode

When `distributed.enabled: true` is set, the pod boots in a mode where it
operates as both an HTTP API server **and** an in-process parameter server.
The training driver hashes entity IDs across `distributed.numShards` shards
and spawns `distributed.numWorkers` worker tasks.  This is bounded to 4-8
workers per the v1.1.0 prototype design — for larger setups use a
specialised system (Horovod, Ray, DeepSpeed).

## Kubernetes — raw manifests

If you do not have Helm available, raw manifests live under `k8s/`:

```sh
kubectl create namespace oxirs
kubectl apply -f ai/oxirs-embed/deploy/k8s/configmap.yaml
kubectl apply -f ai/oxirs-embed/deploy/k8s/deployment.yaml
kubectl apply -f ai/oxirs-embed/deploy/k8s/service.yaml
kubectl apply -f ai/oxirs-embed/deploy/k8s/hpa.yaml
kubectl apply -f ai/oxirs-embed/deploy/k8s/pdb.yaml
```

The raw manifests are equivalent to the chart's defaults — they are useful
for inspection, GitOps diffs, and clusters that prohibit Helm.

## Observability

The service exports the following metrics on `/metrics` (port 9090):

| Metric                                    | Type      | Description                                    |
|-------------------------------------------|-----------|------------------------------------------------|
| `oxirs_embed_requests_total`              | counter   | API requests by route                          |
| `oxirs_embed_request_duration_seconds`    | histogram | API request latency                            |
| `oxirs_embed_embedding_latency_seconds`   | histogram | Per-embedding inference latency                |
| `oxirs_embed_active_models`               | gauge     | Number of loaded models                        |
| `oxirs_embed_distributed_workers`         | gauge     | Live worker count (when distributed enabled)   |
| `oxirs_embed_distributed_pushes_total`    | counter   | ParameterServer push operations                |
| `oxirs_embed_distributed_pulls_total`     | counter   | ParameterServer pull operations                |
| `oxirs_embed_distributed_barriers_total`  | counter   | Sync-mode barrier completions                  |
| `oxirs_embed_distributed_staleness`       | gauge     | Async-mode staleness per shard                 |

Probe paths:

- `GET /health` — process liveness; returns 200 if the embedding subsystem is alive
- `GET /ready` — readiness gate; returns 200 once at least one model is loaded

## Smoke validation

The repository ships two smoke-style integration tests that operators can
run *without* a live cluster:

```sh
# Helm template render — needs `helm` binary on PATH
cargo test -p oxirs-embed --test distributed_training -- --ignored helm_template_smoke

# Docker build — needs `docker` daemon and a network
cargo test -p oxirs-embed --test distributed_training -- --ignored docker_build_smoke
```

The non-ignored test cases verify that all expected files exist on disk.

## Security posture

- Runs as **non-root** UID 65532, group 65532.
- Container filesystem is **read-only**; only `/tmp` is a writable
  `emptyDir`.
- All Linux capabilities are dropped (`drop: [ALL]`).
- `seccompProfile: RuntimeDefault`.
- `automountServiceAccountToken: false`.

## Versioning

The chart `version` and `appVersion` track the Cargo crate version.  When
bumping the crate, also update:

- `deploy/helm/oxirs-embed/Chart.yaml` (`version`, `appVersion`)
- `deploy/helm/oxirs-embed/values.yaml` (`image.tag`)
- `deploy/k8s/deployment.yaml` (`image:` field)
- `deploy/README.md` (this file)
