# OxiRS Fuseki API Reference

Complete API reference for OxiRS Fuseki SPARQL server.

## Table of Contents

- [SPARQL Protocol Endpoints](#sparql-protocol-endpoints)
- [Graph Store Protocol](#graph-store-protocol)
- [REST API v2](#rest-api-v2)
- [GraphQL API](#graphql-api)
- [WebSocket API](#websocket-api)
- [Admin Endpoints](#admin-endpoints)
- [Authentication](#authentication)
- [Response Formats](#response-formats)
- [Error Handling](#error-handling)

## SPARQL Protocol Endpoints

### Query Endpoint

Execute SPARQL SELECT, ASK, CONSTRUCT, or DESCRIBE queries.

**Endpoint**: `POST /dataset/{dataset}/query`

**Methods**: `GET`, `POST`

**Headers**:
- `Content-Type`: `application/sparql-query` (POST with direct query)
- `Content-Type`: `application/x-www-form-urlencoded` (POST with form data)
- `Accept`: Requested response format (see [Response Formats](#response-formats))

**Parameters**:
- `query` (required): SPARQL query string
- `default-graph-uri` (optional): Default graph URI
- `named-graph-uri` (optional): Named graph URI(s)
- `timeout` (optional): Query timeout in seconds

**Example - POST with direct query**:
```bash
curl -X POST http://localhost:3030/default/query \
  -H "Content-Type: application/sparql-query" \
  -H "Accept: application/sparql-results+json" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
```

**Example - GET with URL parameters**:
```bash
curl "http://localhost:3030/default/query?query=SELECT%20*%20WHERE%20%7B%20%3Fs%20%3Fp%20%3Fo%20%7D%20LIMIT%2010"
```

**Response** (JSON):
```json
{
  "head": {
    "vars": ["s", "p", "o"]
  },
  "results": {
    "bindings": [
      {
        "s": {"type": "uri", "value": "http://example.org/subject"},
        "p": {"type": "uri", "value": "http://example.org/predicate"},
        "o": {"type": "literal", "value": "Object value"}
      }
    ]
  }
}
```

### Update Endpoint

Execute SPARQL UPDATE operations (INSERT, DELETE, LOAD, CLEAR, etc.).

**Endpoint**: `POST /dataset/{dataset}/update`

**Methods**: `POST`

**Headers**:
- `Content-Type`: `application/sparql-update` or `application/x-www-form-urlencoded`

**Parameters**:
- `update` (required): SPARQL update operation

**Example**:
```bash
curl -X POST http://localhost:3030/default/update \
  -H "Content-Type: application/sparql-update" \
  -d "INSERT DATA {
    <http://example.org/book1> <http://purl.org/dc/elements/1.1/title> \"Semantic Web\" .
  }"
```

**Response**:
```json
{
  "success": true,
  "message": "Update executed successfully",
  "modified_count": 1
}
```

## Graph Store Protocol

HTTP-based CRUD operations for RDF graphs.

### Get Graph

**Endpoint**: `GET /dataset/{dataset}/data`

**Parameters**:
- `graph` (optional): Named graph URI. Omit for default graph.
- `default` (optional): Force default graph

**Example**:
```bash
# Get default graph
curl http://localhost:3030/default/data

# Get named graph
curl "http://localhost:3030/default/data?graph=http://example.org/mygraph"
```

### Add/Replace Graph

**Endpoint**: `PUT /dataset/{dataset}/data`

**Methods**: `PUT` (replace), `POST` (merge)

**Headers**:
- `Content-Type`: RDF format (see [Response Formats](#response-formats))

**Example - Replace default graph**:
```bash
curl -X PUT http://localhost:3030/default/data \
  -H "Content-Type: text/turtle" \
  -d "@prefix ex: <http://example.org/> .
      ex:subject ex:predicate ex:object ."
```

**Example - Merge into named graph**:
```bash
curl -X POST "http://localhost:3030/default/data?graph=http://example.org/mygraph" \
  -H "Content-Type: text/turtle" \
  -d "@prefix ex: <http://example.org/> .
      ex:subject ex:predicate ex:object ."
```

### Delete Graph

**Endpoint**: `DELETE /dataset/{dataset}/data`

**Parameters**:
- `graph` (required for named graph): Named graph URI

**Example**:
```bash
curl -X DELETE "http://localhost:3030/default/data?graph=http://example.org/mygraph"
```

### Head (Check Existence)

**Endpoint**: `HEAD /dataset/{dataset}/data`

**Returns**: `200 OK` if graph exists, `404 Not Found` otherwise

**Example**:
```bash
curl -I http://localhost:3030/default/data
```

## REST API v2

Modern RESTful API with OpenAPI 3.0 specification.

**Base Path**: `/api/v2`

**API Documentation**: `http://localhost:3030/api/v2/docs` (Swagger UI)

### Datasets

#### List Datasets

```
GET /api/v2/datasets
```

**Response**:
```json
{
  "datasets": [
    {
      "name": "default",
      "type": "Memory",
      "triple_count": 1234,
      "created_at": "2025-11-10T00:00:00Z"
    }
  ],
  "total": 1
}
```

#### Get Dataset

```
GET /api/v2/datasets/{dataset}
```

**Response**:
```json
{
  "name": "default",
  "type": "Memory",
  "description": "Default in-memory dataset",
  "triple_count": 1234,
  "quad_count": 1234,
  "graph_count": 5,
  "created_at": "2025-11-10T00:00:00Z",
  "updated_at": "2025-11-10T12:00:00Z"
}
```

#### Create Dataset

```
POST /api/v2/datasets
Content-Type: application/json

{
  "name": "mydata",
  "type": "Persistent",
  "description": "My dataset",
  "options": {
    "data_dir": "./data/mydata"
  }
}
```

#### Delete Dataset

```
DELETE /api/v2/datasets/{dataset}
```

### Queries

#### Execute Query

```
POST /api/v2/datasets/{dataset}/query
Content-Type: application/json

{
  "query": "SELECT * WHERE { ?s ?p ?o } LIMIT 10",
  "timeout": 60
}
```

**Response**:
```json
{
  "results": {
    "head": {"vars": ["s", "p", "o"]},
    "results": {"bindings": [...]}
  },
  "execution_time_ms": 45,
  "result_count": 10
}
```

### Triples

#### List Triples

```
GET /api/v2/datasets/{dataset}/triples?limit=100&offset=0
```

**Response**:
```json
{
  "triples": [
    {
      "subject": "http://example.org/subject",
      "predicate": "http://example.org/predicate",
      "object": {"value": "Object", "type": "literal"}
    }
  ],
  "total": 1234,
  "page": {
    "limit": 100,
    "offset": 0
  }
}
```

#### Insert Triple

```
POST /api/v2/datasets/{dataset}/triples
Content-Type: application/json

{
  "subject": "http://example.org/subject",
  "predicate": "http://example.org/predicate",
  "object": {"value": "Object", "type": "literal"}
}
```

#### Delete Triple

```
DELETE /api/v2/datasets/{dataset}/triples
Content-Type: application/json

{
  "subject": "http://example.org/subject",
  "predicate": "http://example.org/predicate",
  "object": {"value": "Object", "type": "literal"}
}
```

### Statistics

```
GET /api/v2/stats
```

**Response**:
```json
{
  "datasets": 3,
  "total_triples": 10000,
  "total_queries": 5432,
  "uptime_seconds": 86400,
  "memory_usage_mb": 256,
  "cache_hit_rate": 0.85
}
```

## GraphQL API

Modern GraphQL interface with async-graphql.

**Endpoint**: `POST /graphql`

**Playground**: `http://localhost:3030/graphql` (when accessed via browser)

### Schema

```graphql
type Query {
  datasets: [Dataset!]!
  dataset(name: String!): Dataset
  executeQuery(dataset: String!, query: String!): QueryResult!
  search(dataset: String!, pattern: String!): [Triple!]!
  statistics: Statistics!
}

type Dataset {
  name: String!
  description: String
  tripleCount: Int!
  graphCount: Int!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type QueryResult {
  variables: [String!]!
  bindings: [Binding!]!
  executionTimeMs: Int!
}

type Triple {
  subject: String!
  predicate: String!
  object: RdfNode!
}

type Statistics {
  datasets: Int!
  totalTriples: Int!
  totalQueries: Int!
  uptimeSeconds: Int!
  memoryUsageMb: Int!
}
```

### Example Queries

**List datasets**:
```graphql
query {
  datasets {
    name
    tripleCount
    createdAt
  }
}
```

**Execute SPARQL query**:
```graphql
query {
  executeQuery(
    dataset: "default",
    query: "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
  ) {
    variables
    bindings {
      subject
      predicate
      object
    }
    executionTimeMs
  }
}
```

**Search triples**:
```graphql
query {
  search(dataset: "default", pattern: "Alice") {
    subject
    predicate
    object {
      value
      type
    }
  }
}
```

## WebSocket API

Real-time subscriptions and notifications.

### Query Subscriptions

**Endpoint**: `ws://localhost:3030/ws`

**Subscribe to query results**:
```json
{
  "type": "subscribe",
  "query": "SELECT * WHERE { ?s ?p ?o }",
  "dataset": "default",
  "poll_interval_ms": 5000
}
```

**Receive updates**:
```json
{
  "type": "query_result",
  "subscription_id": "uuid",
  "timestamp": "2025-11-10T12:00:00Z",
  "results": {
    "bindings": [...]
  }
}
```

**Unsubscribe**:
```json
{
  "type": "unsubscribe",
  "subscription_id": "uuid"
}
```

### Real-time Notifications

**Endpoint**: `ws://localhost:3030/notifications`

**Subscribe to events**:
```json
{
  "type": "subscribe",
  "filters": {
    "event_types": ["dataset_updated", "query_completed"],
    "datasets": ["default"],
    "severity": "info"
  }
}
```

**Receive notifications**:
```json
{
  "event_type": "dataset_updated",
  "dataset": "default",
  "timestamp": "2025-11-10T12:00:00Z",
  "severity": "info",
  "details": {
    "triples_added": 10,
    "triples_removed": 2
  }
}
```

**Event types**:
- `dataset_updated`: Dataset modified
- `dataset_created`: New dataset created
- `dataset_deleted`: Dataset removed
- `query_completed`: Query execution finished
- `query_failed`: Query execution failed
- `backup_started`: Backup operation started
- `backup_completed`: Backup completed
- `system_status`: System status change

## Admin Endpoints

Administrative operations (require admin permissions).

### Server Information

```
GET /$/server
```

**Response**:
```json
{
  "version": "0.1.0-rc.1",
  "build_date": "2025-12-21",
  "uptime_seconds": 86400,
  "rust_version": "1.75.0"
}
```

### Dataset Management

```
GET /$/datasets
POST /$/datasets/{dataset}
DELETE /$/datasets/{dataset}
```

### Performance Metrics

```
GET /$/admin/performance/stats
GET /$/admin/performance/memory
GET /$/admin/performance/concurrency
POST /$/admin/performance/gc
```

**Example - Get performance stats**:
```bash
curl http://localhost:3030/$/admin/performance/stats
```

**Response**:
```json
{
  "total_queries": 10000,
  "queries_per_second": 125.5,
  "avg_query_time_ms": 45.2,
  "cache_hit_rate": 0.85,
  "active_connections": 12
}
```

### API Key Management

```
GET /$/admin/api-keys
POST /$/admin/api-keys
GET /$/admin/api-keys/{key_id}
PUT /$/admin/api-keys/{key_id}
DELETE /$/admin/api-keys/{key_id}
GET /$/admin/api-keys/{key_id}/usage
```

**Create API key**:
```bash
curl -X POST http://localhost:3030/$/admin/api-keys \
  -H "Authorization: Bearer admin-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-app",
    "permissions": ["read", "write"],
    "expires_in_days": 90
  }'
```

**Response**:
```json
{
  "key_id": "uuid",
  "api_key": "ak_xxxxxxxxxxxxx",
  "name": "my-app",
  "permissions": ["read", "write"],
  "created_at": "2025-11-10T12:00:00Z",
  "expires_at": "2026-02-08T12:00:00Z"
}
```

## Authentication

### JWT Authentication

#### Login

```
POST /$/auth/login
Content-Type: application/json

{
  "username": "alice",
  "password": "secret"
}
```

**Response**:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 86400,
  "user": {
    "username": "alice",
    "roles": ["user", "admin"]
  }
}
```

#### Use JWT Token

Include in `Authorization` header:
```bash
curl -H "Authorization: Bearer eyJhbGc..." http://localhost:3030/default/query
```

### OAuth2/OIDC

#### Initiate OAuth2 Flow

```
GET /$/auth/oauth/{provider}/login
```

Redirects to OAuth provider for authentication.

#### OAuth2 Callback

```
GET /$/auth/oauth/{provider}/callback?code=...
```

Exchanges code for JWT token.

### API Key Authentication

Include in `X-API-Key` header:
```bash
curl -H "X-API-Key: ak_xxxxxxxxxxxxx" http://localhost:3030/default/query
```

### SAML Authentication

#### Initiate SAML SSO

```
GET /$/auth/saml/login
```

#### SAML Assertion Consumer Service

```
POST /$/auth/saml/acs
```

### Certificate Authentication

Use client certificates with mutual TLS:

```toml
[server.tls]
client_auth = "Required"
trusted_ca_path = "/path/to/ca.pem"
```

## Response Formats

### SPARQL Query Results

**JSON** (`application/sparql-results+json`):
```json
{
  "head": {"vars": ["s", "p", "o"]},
  "results": {
    "bindings": [
      {
        "s": {"type": "uri", "value": "..."},
        "p": {"type": "uri", "value": "..."},
        "o": {"type": "literal", "value": "...", "datatype": "..."}
      }
    ]
  }
}
```

**XML** (`application/sparql-results+xml`):
```xml
<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head>
    <variable name="s"/>
    <variable name="p"/>
    <variable name="o"/>
  </head>
  <results>
    <result>
      <binding name="s"><uri>...</uri></binding>
      ...
    </result>
  </results>
</sparql>
```

**CSV** (`text/csv`):
```csv
s,p,o
http://example.org/subject,http://example.org/predicate,"Object"
```

**TSV** (`text/tab-separated-values`):
```tsv
s	p	o
http://example.org/subject	http://example.org/predicate	"Object"
```

### RDF Graph Formats

- **Turtle** (`text/turtle`): `.ttl`
- **N-Triples** (`application/n-triples`): `.nt`
- **RDF/XML** (`application/rdf+xml`): `.rdf`
- **N-Quads** (`application/n-quads`): `.nq`
- **TriG** (`application/trig`): `.trig`
- **JSON-LD** (`application/ld+json`): `.jsonld`

## Error Handling

### Error Response Format

```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {
    "line": 5,
    "column": 12,
    "query": "SELECT..."
  }
}
```

### HTTP Status Codes

- `200 OK`: Success
- `400 Bad Request`: Invalid query or parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Dataset or resource not found
- `408 Request Timeout`: Query timeout
- `413 Payload Too Large`: Request too large
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Server overloaded

### Common Error Codes

- `invalid_query`: SPARQL syntax error
- `query_timeout`: Query exceeded time limit
- `dataset_not_found`: Dataset does not exist
- `authentication_failed`: Invalid credentials
- `permission_denied`: Insufficient permissions
- `rate_limit_exceeded`: Too many requests
- `service_unavailable`: Federation endpoint unavailable
- `internal_error`: Unexpected server error

### Example Error Responses

**Invalid SPARQL Query**:
```json
{
  "error": "invalid_query",
  "message": "Parse error at line 1, column 15",
  "details": {
    "line": 1,
    "column": 15,
    "query": "SELECT * WHRE { ?s ?p ?o }",
    "suggestion": "Did you mean WHERE?"
  }
}
```

**Query Timeout**:
```json
{
  "error": "query_timeout",
  "message": "Query exceeded timeout of 300 seconds",
  "details": {
    "timeout_seconds": 300,
    "elapsed_seconds": 305
  }
}
```

**Rate Limit**:
```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit of 1000 requests per minute exceeded",
  "details": {
    "limit": 1000,
    "window_seconds": 60,
    "retry_after_seconds": 45
  }
}
```

## Rate Limiting

**Headers**:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Time when limit resets (Unix timestamp)
- `Retry-After`: Seconds until retry is allowed (when rate limited)

**Example**:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 245
X-RateLimit-Reset: 1699632000
```

## Request IDs

All responses include a `X-Request-ID` header for tracing:
```
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
```

Use this ID when reporting issues or viewing logs.

## Pagination

For endpoints supporting pagination:

**Parameters**:
- `limit`: Number of results per page (default: 100, max: 1000)
- `offset`: Number of results to skip

**Response Headers**:
- `X-Total-Count`: Total number of results
- `Link`: Links to next/prev pages (RFC 5988)

**Example**:
```
X-Total-Count: 10000
Link: </api/v2/datasets/default/triples?offset=100&limit=100>; rel="next",
      </api/v2/datasets/default/triples?offset=0&limit=100>; rel="first"
```

## Compression

Response compression supported:
- `gzip`
- `br` (Brotli)

**Request header**:
```
Accept-Encoding: gzip, br
```

## CORS

CORS headers can be configured:

```toml
[server]
enable_cors = true
cors_allowed_origins = ["http://localhost:3000", "https://example.com"]
cors_allowed_methods = ["GET", "POST", "PUT", "DELETE"]
cors_allowed_headers = ["Content-Type", "Authorization"]
cors_max_age_secs = 3600
```

## Versioning

API versioning via:
- **Path**: `/api/v2/...`
- **Header**: `X-API-Version: 2`

## See Also

- [Getting Started Guide](GETTING_STARTED.md)
- [Performance Tuning](PERFORMANCE_TUNING.md)
- [Security Guide](SECURITY.md)
- [Examples](../examples/)
