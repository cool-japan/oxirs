# WebSocket Support for Live SPARQL Query Subscriptions

## Overview

OxiRS Fuseki now supports real-time SPARQL query subscriptions through WebSockets, enabling clients to receive live updates when query results change. This feature is essential for building reactive applications that need to stay synchronized with changing RDF data.

## Key Features

### 1. Live Query Subscriptions
- **Real-time Updates**: Automatically receive notifications when query results change
- **Multiple Subscriptions**: Support for multiple concurrent subscriptions per connection
- **Query Types**: Support for SELECT and CONSTRUCT queries
- **Change Detection**: Intelligent change detection to minimize unnecessary updates

### 2. Advanced Filtering
- **Change Thresholds**: Set minimum change percentages to trigger notifications
- **Variable Monitoring**: Monitor specific variables for changes
- **Debouncing**: Prevent notification floods with configurable debounce times
- **Rate Limiting**: Control notification frequency per subscription

### 3. Connection Management
- **Authentication**: Support for token-based authentication
- **Heartbeat**: Automatic ping/pong for connection health monitoring
- **Graceful Shutdown**: Proper cleanup of subscriptions on disconnect
- **Connection Limits**: Configurable limits per connection and globally

### 4. Performance Optimization
- **Query Caching**: Cache query results for efficient change detection
- **Parallel Evaluation**: Evaluate multiple subscriptions concurrently
- **Compression**: Optional message compression for bandwidth efficiency
- **Resource Management**: Automatic cleanup of inactive connections

## Architecture

```
┌─────────────────────────────────────────────┐
│            Client Application               │
└─────────────┬───────────────────────────────┘
              │ WebSocket Connection
┌─────────────▼───────────────────────────────┐
│         WebSocket Handler                   │
│  • Connection lifecycle                     │
│  • Message routing                         │
│  • Authentication                          │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│      Subscription Manager                   │
│  • Subscription registry                    │
│  • Query evaluation loop                   │
│  • Change detection                        │
│  • Notification dispatch                   │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│         Query Executor                      │
│  • SPARQL query execution                  │
│  • Result formatting                       │
│  • Performance monitoring                  │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│          RDF Store                          │
└─────────────────────────────────────────────┘
```

## WebSocket Protocol

### Connection Establishment

```javascript
const ws = new WebSocket('ws://localhost:3030/ws');

// Optional authentication
ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'your-auth-token'
    }));
};
```

### Message Types

#### 1. Subscribe to Query
```json
{
    "type": "subscribe",
    "query": "SELECT ?person ?name WHERE { ?person foaf:name ?name } LIMIT 10",
    "parameters": {
        "default_graph_uri": [],
        "named_graph_uri": ["http://example.org/people"],
        "timeout_ms": 5000,
        "format": "json"
    },
    "filter": {
        "min_change_threshold": 5.0,
        "monitored_variables": ["person", "name"],
        "debounce_ms": 1000,
        "rate_limit": 60
    }
}
```

#### 2. Subscription Confirmation
```json
{
    "type": "subscribed",
    "subscription_id": "550e8400-e29b-41d4-a716-446655440000",
    "query": "SELECT ?person ?name WHERE..."
}
```

#### 3. Query Update Notification
```json
{
    "type": "query_update",
    "subscription_id": "550e8400-e29b-41d4-a716-446655440000",
    "result": {
        "bindings": [
            {
                "person": "http://example.org/person1",
                "name": "Alice"
            },
            {
                "person": "http://example.org/person2",
                "name": "Bob"
            }
        ],
        "metadata": {
            "execution_time_ms": 15,
            "result_count": 2,
            "result_hash": 3847562
        }
    },
    "changes": {
        "added": [
            {"person": "http://example.org/person2", "name": "Bob"}
        ],
        "removed": [],
        "modified": []
    }
}
```

#### 4. Unsubscribe
```json
{
    "type": "unsubscribe",
    "subscription_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### 5. Heartbeat
```json
{
    "type": "ping",
    "timestamp": 1638360000000
}
```

Response:
```json
{
    "type": "pong",
    "timestamp": 1638360000000
}
```

#### 6. Error Messages
```json
{
    "type": "error",
    "code": "subscription_limit_exceeded",
    "message": "Maximum subscriptions per connection exceeded",
    "details": {
        "current": 100,
        "maximum": 100
    }
}
```

## Usage Examples

### JavaScript/TypeScript Client

```typescript
class SPARQLSubscriptionClient {
    private ws: WebSocket;
    private subscriptions: Map<string, (result: any) => void> = new Map();

    constructor(url: string, token?: string) {
        this.ws = new WebSocket(url);
        
        this.ws.onopen = () => {
            console.log('Connected to SPARQL WebSocket');
            if (token) {
                this.authenticate(token);
            }
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        // Heartbeat
        setInterval(() => {
            if (this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'ping',
                    timestamp: Date.now()
                }));
            }
        }, 30000);
    }

    authenticate(token: string) {
        this.send({
            type: 'auth',
            token: token
        });
    }

    subscribe(
        query: string,
        callback: (result: any) => void,
        options?: {
            graphs?: string[];
            filter?: any;
        }
    ) {
        const message = {
            type: 'subscribe',
            query: query,
            parameters: {
                default_graph_uri: [],
                named_graph_uri: options?.graphs || [],
                timeout_ms: 5000,
                format: 'json'
            },
            filter: options?.filter
        };

        this.send(message);
        
        // Store callback for later
        // (In real implementation, wait for subscription confirmation)
        const tempId = Date.now().toString();
        this.subscriptions.set(tempId, callback);
    }

    unsubscribe(subscriptionId: string) {
        this.send({
            type: 'unsubscribe',
            subscription_id: subscriptionId
        });
        this.subscriptions.delete(subscriptionId);
    }

    private handleMessage(message: any) {
        switch (message.type) {
            case 'subscribed':
                console.log('Subscription confirmed:', message.subscription_id);
                break;
                
            case 'query_update':
                const callback = this.subscriptions.get(message.subscription_id);
                if (callback) {
                    callback(message.result);
                }
                break;
                
            case 'error':
                console.error('Server error:', message);
                break;
                
            case 'pong':
                // Heartbeat response
                break;
        }
    }

    private send(message: any) {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }

    close() {
        this.ws.close();
    }
}

// Usage example
const client = new SPARQLSubscriptionClient('ws://localhost:3030/ws');

// Subscribe to person name changes
client.subscribe(
    `
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?person ?name ?email
    WHERE {
        ?person a foaf:Person ;
                foaf:name ?name ;
                foaf:mbox ?email .
    }
    LIMIT 100
    `,
    (result) => {
        console.log('Query results updated:', result);
        // Update UI with new results
    },
    {
        filter: {
            min_change_threshold: 1.0,
            monitored_variables: ['person', 'name'],
            debounce_ms: 500,
            rate_limit: 120
        }
    }
);
```

### Python Client

```python
import json
import asyncio
import websockets
from typing import Dict, Callable, Optional

class SPARQLSubscriptionClient:
    def __init__(self, url: str, token: Optional[str] = None):
        self.url = url
        self.token = token
        self.subscriptions: Dict[str, Callable] = {}
        
    async def connect(self):
        self.ws = await websockets.connect(self.url)
        
        if self.token:
            await self.authenticate(self.token)
            
        # Start message handler
        asyncio.create_task(self.message_handler())
        
        # Start heartbeat
        asyncio.create_task(self.heartbeat())
        
    async def authenticate(self, token: str):
        await self.send({
            'type': 'auth',
            'token': token
        })
        
    async def subscribe(
        self, 
        query: str, 
        callback: Callable,
        graphs: Optional[list] = None,
        filter: Optional[dict] = None
    ) -> str:
        message = {
            'type': 'subscribe',
            'query': query,
            'parameters': {
                'default_graph_uri': [],
                'named_graph_uri': graphs or [],
                'timeout_ms': 5000,
                'format': 'json'
            },
            'filter': filter
        }
        
        await self.send(message)
        # In real implementation, wait for subscription confirmation
        return 'pending'
        
    async def unsubscribe(self, subscription_id: str):
        await self.send({
            'type': 'unsubscribe',
            'subscription_id': subscription_id
        })
        self.subscriptions.pop(subscription_id, None)
        
    async def message_handler(self):
        async for message in self.ws:
            data = json.loads(message)
            
            if data['type'] == 'subscribed':
                print(f"Subscription confirmed: {data['subscription_id']}")
                
            elif data['type'] == 'query_update':
                callback = self.subscriptions.get(data['subscription_id'])
                if callback:
                    callback(data['result'])
                    
            elif data['type'] == 'error':
                print(f"Error: {data['message']}")
                
    async def heartbeat(self):
        while True:
            await asyncio.sleep(30)
            await self.send({
                'type': 'ping',
                'timestamp': int(time.time() * 1000)
            })
            
    async def send(self, message: dict):
        await self.ws.send(json.dumps(message))
        
    async def close(self):
        await self.ws.close()

# Usage
async def main():
    client = SPARQLSubscriptionClient('ws://localhost:3030/ws')
    await client.connect()
    
    def on_update(result):
        print(f"Results updated: {len(result['bindings'])} bindings")
        for binding in result['bindings']:
            print(f"  {binding}")
    
    await client.subscribe(
        """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person ?name
        WHERE {
            ?person a foaf:Person ;
                    foaf:name ?name .
        }
        LIMIT 50
        """,
        on_update,
        filter={
            'min_change_threshold': 5.0,
            'rate_limit': 60
        }
    )
    
    # Keep running
    await asyncio.Event().wait()

asyncio.run(main())
```

## Configuration

### Server Configuration

```toml
[websocket]
# Maximum subscriptions per WebSocket connection
max_subscriptions_per_connection = 100

# Maximum total subscriptions across all connections
max_total_subscriptions = 10000

# Query re-evaluation interval (seconds)
evaluation_interval = 1

# Connection timeout (seconds)
connection_timeout = 300

# Maximum message size (bytes)
max_message_size = 10485760  # 10MB

# Enable compression
enable_compression = true

# Heartbeat interval (seconds)
heartbeat_interval = 30
```

### Subscription Filters

#### Change Threshold
Only send updates if results change by at least X%:
```json
{
    "filter": {
        "min_change_threshold": 10.0
    }
}
```

#### Variable Monitoring
Only trigger updates when specific variables change:
```json
{
    "filter": {
        "monitored_variables": ["person", "status"]
    }
}
```

#### Debouncing
Prevent rapid successive updates:
```json
{
    "filter": {
        "debounce_ms": 1000
    }
}
```

#### Rate Limiting
Maximum notifications per minute:
```json
{
    "filter": {
        "rate_limit": 60
    }
}
```

## Performance Considerations

### Query Optimization
1. **Always use LIMIT**: Subscriptions must include LIMIT to prevent excessive resource usage
2. **Index frequently queried properties**: Ensure proper indexing for subscribed queries
3. **Avoid expensive operations**: Minimize use of FILTER, OPTIONAL, and complex patterns

### Resource Management
1. **Connection Limits**: Set appropriate limits based on server capacity
2. **Query Complexity**: Monitor and limit query complexity
3. **Evaluation Frequency**: Balance between real-time updates and server load
4. **Result Caching**: Leverage caching to minimize redundant query execution

### Scaling Considerations
1. **Horizontal Scaling**: Use load balancers with WebSocket session affinity
2. **Redis PubSub**: For multi-server deployments, use Redis for change notifications
3. **Query Partitioning**: Distribute subscriptions across multiple servers
4. **Connection Pooling**: Reuse connections for multiple subscriptions

## Security

### Authentication
```json
{
    "type": "auth",
    "token": "Bearer eyJhbGciOiJIUzI1NiIs..."
}
```

### Authorization
- Subscriptions inherit permissions from authenticated user
- Graph-level access control is enforced
- Rate limiting prevents abuse

### Best Practices
1. Always authenticate WebSocket connections
2. Use TLS (wss://) in production
3. Implement connection rate limiting
4. Monitor for suspicious subscription patterns
5. Set appropriate timeout values

## Troubleshooting

### Common Issues

#### Connection Drops
- Check heartbeat configuration
- Verify network timeouts
- Monitor server logs for errors

#### Missing Updates
- Verify query syntax
- Check filter configuration
- Ensure proper change detection
- Monitor evaluation loop performance

#### High Memory Usage
- Reduce max subscriptions
- Optimize query complexity
- Enable result compression
- Implement pagination

### Debug Mode

Enable debug logging:
```toml
[logging]
level = "debug"
websocket = "trace"
```

## Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `invalid_query` | SPARQL syntax error | Fix query syntax |
| `subscription_limit_exceeded` | Too many subscriptions | Unsubscribe from others |
| `authentication_required` | Not authenticated | Send auth message |
| `permission_denied` | Insufficient permissions | Check user permissions |
| `query_timeout` | Query took too long | Optimize query |
| `rate_limit_exceeded` | Too many requests | Reduce request rate |

## Future Enhancements

1. **GraphQL Subscriptions**: Support for GraphQL subscription syntax
2. **Aggregated Updates**: Batch multiple changes into single notification
3. **Partial Results**: Stream large result sets incrementally
4. **Query Templates**: Parameterized queries for efficiency
5. **Distributed Subscriptions**: Cross-cluster subscription support
6. **Machine Learning**: Predictive pre-fetching of likely changes
7. **Mobile SDK**: Native mobile libraries with offline support
8. **Analytics**: Subscription usage analytics and insights

## Conclusion

WebSocket support in OxiRS Fuseki enables building reactive, real-time applications on top of RDF data. With intelligent change detection, flexible filtering, and robust connection management, it provides a production-ready solution for live SPARQL query subscriptions.