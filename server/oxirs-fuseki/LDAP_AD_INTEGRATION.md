# LDAP/Active Directory Integration

## Overview

OxiRS Fuseki provides comprehensive LDAP and Active Directory authentication support, enabling enterprise directory integration for user authentication and group-based authorization.

## Features

### LDAP Protocol Support
- **LDAP v3** protocol compliance
- **LDAPS** (LDAP over TLS/SSL) support
- **StartTLS** for secure connections
- **Anonymous and authenticated bind** operations
- **Connection pooling** for improved performance

### Active Directory Features
- **Native AD support** with optimized settings
- **User Principal Name (UPN)** authentication
- **Nested group membership** resolution
- **Global Catalog** search support
- **Referral following** for multi-domain forests
- **Paged results** for large directories

### Authentication Features
- **Simple bind authentication** with username/password
- **User search** with configurable filters
- **Group membership** extraction
- **Role mapping** from LDAP groups to internal roles
- **User attribute** mapping
- **Caching** for improved performance

## Configuration

### Basic LDAP Configuration

Add LDAP configuration to your `oxirs.toml`:

```toml
[security.ldap]
server = "ldap://ldap.example.com:389"
bind_dn = "cn=service,ou=accounts,dc=example,dc=com"
bind_password = "service_password"
user_base_dn = "ou=people,dc=example,dc=com"
user_filter = "(uid={username})"
group_base_dn = "ou=groups,dc=example,dc=com"
group_filter = "(member={userdn})"
use_tls = false
```

### Active Directory Configuration

For Active Directory environments:

```toml
[security.ldap]
server = "ldap://dc1.corp.example.com:389"
bind_dn = "CN=Service Account,OU=Service Accounts,DC=corp,DC=example,DC=com"
bind_password = "service_password"
user_base_dn = "DC=corp,DC=example,DC=com"
user_filter = "(&(objectClass=user)(sAMAccountName={username}))"
group_base_dn = "DC=corp,DC=example,DC=com"
group_filter = "(&(objectClass=group)(member={userdn}))"
use_tls = true

[security.ldap.attributes]
username = "sAMAccountName"
email = "mail"
full_name = "cn"
display_name = "displayName"
member_of = "memberOf"
additional = ["department", "title", "telephoneNumber"]

[security.ldap.active_directory]
domain = "CORP.EXAMPLE.COM"
default_suffix = "@corp.example.com"
follow_referrals = true
use_paged_results = true
page_size = 1000

[security.ldap.group_config]
base_dn = "DC=corp,DC=example,DC=com"
filter = "(member:1.2.840.113556.1.4.1941:={userdn})"
name_attribute = "cn"
description_attribute = "description"
fetch_nested = true
max_depth = 5

[security.ldap.group_config.role_mappings]
"Domain Admins" = "admin"
"Fuseki Administrators" = "admin"
"Developers" = "writer"
"Data Scientists" = "writer"
"Users" = "reader"
"Domain Users" = "user"
```

### Advanced Configuration

```toml
[security.ldap]
server = "ldaps://ldap.example.com:636"
bind_dn = "cn=service,ou=accounts,dc=example,dc=com"
bind_password = "service_password"
user_base_dn = "ou=people,dc=example,dc=com"
user_filter = "(&(objectClass=inetOrgPerson)(|(uid={username})(mail={username})))"
group_base_dn = "ou=groups,dc=example,dc=com"
group_filter = "(|(member={userdn})(uniqueMember={userdn}))"
use_tls = true
tls_verify = true
timeout_secs = 30

[security.ldap.pool_config]
min_connections = 2
max_connections = 10
idle_timeout_secs = 300
max_lifetime_secs = 3600
health_check_interval_secs = 60

[security.ldap.attributes]
username = "uid"
email = "mail"
full_name = "cn"
display_name = "displayName"
member_of = "memberOf"
additional = ["employeeNumber", "department", "title"]

[security.ldap.group_config]
base_dn = "ou=groups,dc=example,dc=com"
filter = "(member={userdn})"
name_attribute = "cn"
description_attribute = "description"
fetch_nested = true
max_depth = 3

[security.ldap.group_config.role_mappings]
"admins" = "admin"
"developers" = "writer"
"analysts" = "reader"
"*-admin" = "admin"      # Pattern matching with wildcards
"team-*" = "user"
```

## Authentication Flow

### 1. User Login

```bash
curl -X POST http://localhost:3030/auth/ldap/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john.doe",
    "password": "password123",
    "domain": "corp.example.com"  # Optional for AD
  }'

# Response:
{
  "success": true,
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "user": {
    "username": "john.doe",
    "roles": ["writer", "user"],
    "email": "john.doe@example.com",
    "full_name": "John Doe"
  },
  "message": "LDAP authentication successful"
}
```

### 2. Using the Access Token

```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
     http://localhost:3030/sparql?query=SELECT%20*%20WHERE%20%7B%3Fs%20%3Fp%20%3Fo%7D
```

## API Endpoints

### Authentication Endpoints

#### POST /auth/ldap/login
Authenticate user against LDAP directory.

**Request Body:**
```json
{
  "username": "john.doe",
  "password": "password123",
  "domain": "optional.domain.com"
}
```

**Response:**
```json
{
  "success": true,
  "access_token": "session_or_jwt_token",
  "token_type": "Bearer",
  "expires_in": 3600,
  "user": {
    "username": "john.doe",
    "roles": ["writer"],
    "email": "john.doe@example.com",
    "full_name": "John Doe"
  },
  "message": "LDAP authentication successful"
}
```

### Administration Endpoints

#### GET /auth/ldap/test
Test LDAP connection with current configuration.

**Query Parameters:**
- `server` (optional): Override server URL
- `bind_dn` (optional): Override bind DN
- `bind_password` (optional): Override bind password

**Response:**
```json
{
  "success": true,
  "message": "LDAP connection successful",
  "details": "Connected to LDAP server at ldap://ldap.example.com:389"
}
```

#### GET /auth/ldap/groups
Get user's LDAP groups.

**Query Parameters:**
- `username`: Username to query

**Response:**
```json
{
  "success": true,
  "groups": [
    {
      "dn": "cn=developers,ou=groups,dc=example,dc=com",
      "name": "developers",
      "description": "Development team"
    },
    {
      "dn": "cn=users,ou=groups,dc=example,dc=com",
      "name": "users",
      "description": "All users"
    }
  ],
  "message": "Found 2 groups for user"
}
```

#### GET /auth/ldap/config
Get current LDAP configuration (without sensitive data).

**Response:**
```json
{
  "success": true,
  "configured": true,
  "server": "ldap://ldap.example.com:389",
  "use_tls": false,
  "user_base_dn": "ou=people,dc=example,dc=com",
  "group_base_dn": "ou=groups,dc=example,dc=com",
  "user_filter": "(uid={username})",
  "group_filter": "(member={userdn})"
}
```

## Group to Role Mapping

### Default Mappings

LDAP groups are automatically mapped to OxiRS Fuseki roles:

| LDAP Group Pattern | OxiRS Role | Permissions |
|-------------------|------------|-------------|
| *admin*, *administrators* | admin | Full access |
| *writer*, *developer*, *editor* | writer | Read, Write, Query, Update |
| *reader*, *viewer*, *analyst* | reader | Read, Query only |
| (any other) | user | Read, Query only |

### Custom Mappings

Configure custom group-to-role mappings:

```toml
[security.ldap.group_config.role_mappings]
# Exact matches
"Domain Admins" = "admin"
"Engineering Team" = "writer"
"Data Analysts" = "reader"

# Pattern matches (using wildcards)
"project-*-admin" = "admin"
"team-*-lead" = "writer"
"dept-*" = "user"
```

## Security Considerations

### Connection Security

#### LDAPS (Recommended)
```toml
[security.ldap]
server = "ldaps://ldap.example.com:636"
use_tls = true
tls_verify = true  # Verify server certificate
```

#### StartTLS
```toml
[security.ldap]
server = "ldap://ldap.example.com:389"
use_tls = true
tls_verify = true
```

### Service Account Best Practices

1. **Use a dedicated service account** with minimal privileges
2. **Restrict bind DN permissions** to user search only
3. **Use strong passwords** and rotate regularly
4. **Monitor service account usage** for anomalies

### Password Security

- Passwords are **never stored** locally
- All authentication requires **real-time LDAP bind**
- Failed attempts are **logged and tracked**
- Account lockout policies are **enforced**

### Caching Security

- User information is cached for **performance**
- Cache entries **expire after 5 minutes**
- Passwords are **never cached**
- Cache can be **manually cleared** if needed

## Performance Optimization

### Connection Pooling

```toml
[security.ldap.pool_config]
min_connections = 2      # Minimum idle connections
max_connections = 10     # Maximum total connections
idle_timeout_secs = 300  # Remove idle connections after 5 minutes
max_lifetime_secs = 3600 # Recreate connections after 1 hour
```

### Caching Strategy

- **User attributes** cached for 5 minutes
- **Group memberships** cached with user
- **Cache invalidation** on authentication failure
- **Automatic cleanup** of expired entries

### Query Optimization

```toml
# Use indexed attributes for searches
user_filter = "(&(objectClass=user)(sAMAccountName={username}))"

# Limit search scope
user_base_dn = "OU=Users,DC=corp,DC=example,DC=com"

# Use paged results for large directories
[security.ldap.active_directory]
use_paged_results = true
page_size = 1000
```

## Troubleshooting

### Common Issues

#### "LDAP authentication not configured"
Ensure LDAP configuration is present in your config file with all required fields.

#### "User not found"
- Check `user_base_dn` is correct
- Verify `user_filter` matches your schema
- Test with `ldapsearch` command line tool

#### "Invalid credentials"
- Verify service account credentials
- Check user password is correct
- Ensure account is not locked

#### "Connection timeout"
- Check network connectivity to LDAP server
- Verify firewall rules allow LDAP ports (389/636)
- Test with `telnet ldap.example.com 389`

### Debug Logging

Enable debug logging for LDAP operations:

```toml
[logging]
level = "debug"
auth = "trace"
ldap = "trace"
```

### Testing Tools

#### Test LDAP Connection
```bash
# Using ldapsearch
ldapsearch -x -H ldap://ldap.example.com:389 \
  -D "cn=service,dc=example,dc=com" \
  -w "password" \
  -b "ou=people,dc=example,dc=com" \
  "(uid=testuser)"

# Using OxiRS Fuseki endpoint
curl http://localhost:3030/auth/ldap/test
```

#### Test User Authentication
```bash
# Test with curl
curl -X POST http://localhost:3030/auth/ldap/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"testpass"}'
```

## Integration Examples

### JavaScript/TypeScript
```typescript
// LDAP authentication
async function authenticateWithLDAP(username: string, password: string) {
  const response = await fetch('/auth/ldap/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ username, password }),
  });
  
  const data = await response.json();
  
  if (data.success) {
    // Store token
    localStorage.setItem('access_token', data.access_token);
    return data.user;
  } else {
    throw new Error(data.message);
  }
}

// Use authenticated session
async function querySPARQL(query: string) {
  const token = localStorage.getItem('access_token');
  
  const response = await fetch('/sparql', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/sparql-query',
      'Authorization': `Bearer ${token}`,
    },
    body: query,
  });
  
  return response.json();
}
```

### Python
```python
import requests

class LDAPClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.token = None
    
    def authenticate(self, username, password, domain=None):
        """Authenticate with LDAP"""
        data = {
            'username': username,
            'password': password
        }
        if domain:
            data['domain'] = domain
        
        response = self.session.post(
            f"{self.base_url}/auth/ldap/login",
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                self.token = result['access_token']
                return result['user']
        
        raise Exception(f"Authentication failed: {response.text}")
    
    def query_sparql(self, query):
        """Execute SPARQL query with authentication"""
        if not self.token:
            raise Exception("Not authenticated")
        
        response = self.session.post(
            f"{self.base_url}/sparql",
            data=query,
            headers={
                'Content-Type': 'application/sparql-query',
                'Authorization': f'Bearer {self.token}'
            }
        )
        
        return response.json()

# Usage
client = LDAPClient('http://localhost:3030')
user = client.authenticate('john.doe', 'password123')
print(f"Authenticated as: {user['username']}")

results = client.query_sparql("SELECT * WHERE { ?s ?p ?o } LIMIT 10")
```

### Java
```java
import java.net.http.*;
import com.fasterxml.jackson.databind.ObjectMapper;

public class LDAPAuthExample {
    private final String baseUrl;
    private final HttpClient client;
    private final ObjectMapper mapper;
    private String accessToken;
    
    public LDAPAuthExample(String baseUrl) {
        this.baseUrl = baseUrl;
        this.client = HttpClient.newHttpClient();
        this.mapper = new ObjectMapper();
    }
    
    public User authenticate(String username, String password) throws Exception {
        var request = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/auth/ldap/login"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(
                mapper.writeValueAsString(Map.of(
                    "username", username,
                    "password", password
                ))
            ))
            .build();
        
        var response = client.send(request, HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() == 200) {
            var result = mapper.readValue(response.body(), LoginResponse.class);
            if (result.success) {
                this.accessToken = result.accessToken;
                return result.user;
            }
        }
        
        throw new Exception("Authentication failed");
    }
}
```

## Active Directory Specific Features

### User Principal Name (UPN) Authentication

Users can authenticate with their UPN:
```bash
curl -X POST http://localhost:3030/auth/ldap/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john.doe@corp.example.com",
    "password": "password123"
  }'
```

### Nested Group Resolution

The system automatically resolves nested group memberships:
```toml
[security.ldap.group_config]
fetch_nested = true
max_depth = 5  # Maximum nesting depth

# Uses AD-specific filter for recursive group membership
filter = "(member:1.2.840.113556.1.4.1941:={userdn})"
```

### Multi-Domain Support

For multi-domain forests:
```toml
[security.ldap.active_directory]
follow_referrals = true
domain = "PARENT.CORP.COM"

# Use Global Catalog port for cross-domain searches
server = "ldap://gc.parent.corp.com:3268"
```

## Migration from Other Systems

### From Apache Jena Fuseki

1. Export user/group mappings from existing system
2. Configure LDAP with equivalent group mappings
3. Test authentication with subset of users
4. Migrate in phases

### From Local Authentication

1. Ensure LDAP accounts exist for all users
2. Map local roles to LDAP groups
3. Configure LDAP in parallel with local auth
4. Gradually transition users to LDAP

## Future Enhancements

Planned improvements for LDAP/AD support:

1. **Kerberos Authentication**: Native Kerberos/SPNEGO support
2. **SASL Mechanisms**: Support for additional SASL authentication
3. **Multi-Forest Support**: Enhanced Active Directory forest navigation
4. **Smart Card Authentication**: PIV/CAC card support via LDAP
5. **Dynamic Group Discovery**: Automatic role mapping from LDAP schema
6. **LDAP Schema Validation**: Validate configuration against LDAP schema
7. **Performance Metrics**: Detailed LDAP operation metrics
8. **Failover Support**: Multiple LDAP server configuration

## Conclusion

The LDAP/Active Directory integration in OxiRS Fuseki provides enterprise-grade authentication with flexible configuration options, robust security features, and excellent performance characteristics. Whether connecting to OpenLDAP, Active Directory, or other LDAP-compliant directories, the system offers a seamless authentication experience with powerful group-based authorization capabilities.