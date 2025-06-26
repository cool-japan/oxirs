# OAuth2/OIDC Authentication Support

## Overview

OxiRS Fuseki now includes comprehensive OAuth2 and OpenID Connect (OIDC) authentication support, enabling enterprise-grade authentication through popular identity providers like Google, Microsoft Azure AD, Auth0, Keycloak, and more.

## Features

### OAuth2 Protocol Support
- **Authorization Code Flow** with PKCE (Proof Key for Code Exchange)
- **Refresh Token** support for seamless token renewal
- **State Parameter** validation for CSRF protection
- **Scope Management** for fine-grained permissions

### OpenID Connect (OIDC) Support
- **ID Token** validation and parsing
- **UserInfo Endpoint** integration
- **Discovery Document** support (`.well-known/openid-configuration`)
- **Claims Mapping** from OIDC to internal user model

### Security Features
- **PKCE by Default** for enhanced security
- **State Validation** to prevent CSRF attacks
- **Token Caching** with automatic cleanup
- **Secure Token Storage** in memory
- **Automatic Token Expiration** handling

## Configuration

### Basic Configuration

Add OAuth2 configuration to your `oxirs.toml`:

```toml
[security.oauth]
provider = "google"  # or "azure", "auth0", "keycloak", etc.
client_id = "your-client-id.apps.googleusercontent.com"
client_secret = "your-client-secret"
auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
token_url = "https://oauth2.googleapis.com/token"
user_info_url = "https://openidconnect.googleapis.com/v1/userinfo"
scopes = ["openid", "profile", "email"]
```

### Provider-Specific Examples

#### Google OAuth2
```toml
[security.oauth]
provider = "google"
client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_CLIENT_SECRET"
auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
token_url = "https://oauth2.googleapis.com/token"
user_info_url = "https://openidconnect.googleapis.com/v1/userinfo"
scopes = ["openid", "profile", "email"]
```

#### Microsoft Azure AD
```toml
[security.oauth]
provider = "azure"
client_id = "YOUR_APPLICATION_ID"
client_secret = "YOUR_CLIENT_SECRET"
auth_url = "https://login.microsoftonline.com/YOUR_TENANT_ID/oauth2/v2.0/authorize"
token_url = "https://login.microsoftonline.com/YOUR_TENANT_ID/oauth2/v2.0/token"
user_info_url = "https://graph.microsoft.com/v1.0/me"
scopes = ["openid", "profile", "email", "User.Read"]
```

#### Auth0
```toml
[security.oauth]
provider = "auth0"
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
auth_url = "https://YOUR_DOMAIN.auth0.com/authorize"
token_url = "https://YOUR_DOMAIN.auth0.com/oauth/token"
user_info_url = "https://YOUR_DOMAIN.auth0.com/userinfo"
scopes = ["openid", "profile", "email"]
```

#### Keycloak
```toml
[security.oauth]
provider = "keycloak"
client_id = "oxirs-fuseki"
client_secret = "YOUR_CLIENT_SECRET"
auth_url = "https://YOUR_KEYCLOAK/auth/realms/YOUR_REALM/protocol/openid-connect/auth"
token_url = "https://YOUR_KEYCLOAK/auth/realms/YOUR_REALM/protocol/openid-connect/token"
user_info_url = "https://YOUR_KEYCLOAK/auth/realms/YOUR_REALM/protocol/openid-connect/userinfo"
scopes = ["openid", "profile", "email"]
```

## Authentication Flow

### 1. Initiate OAuth2 Login

```bash
# Get the authorization URL
curl -X GET "http://localhost:3030/auth/oauth2/authorize?redirect_uri=http://localhost:3030/auth/oauth2/callback"

# Response:
{
  "success": true,
  "authorization_url": "https://accounts.google.com/o/oauth2/v2/auth?response_type=code&client_id=...",
  "state": "550e8400-e29b-41d4-a716-446655440000",
  "message": "OAuth2 authorization URL generated successfully"
}
```

### 2. User Authorization

Redirect the user to the `authorization_url`. After authorization, they will be redirected back to your callback URL with an authorization code.

### 3. Handle Callback

The callback endpoint automatically:
1. Validates the state parameter
2. Exchanges the authorization code for tokens
3. Fetches user information
4. Creates a session
5. Returns authentication tokens

```bash
# Callback URL example:
# http://localhost:3030/auth/oauth2/callback?code=4/0AX4XfWi...&state=550e8400...

# Response:
{
  "success": true,
  "access_token": "session_id_token",
  "token_type": "Bearer",
  "expires_in": 3600,
  "user": {
    "username": "john.doe@example.com",
    "roles": ["user"],
    "email": "john.doe@example.com",
    "full_name": "John Doe"
  },
  "message": "OAuth2 authentication successful"
}
```

### 4. Using the Access Token

Include the access token in subsequent requests:

```bash
curl -H "Authorization: Bearer session_id_token" \
     http://localhost:3030/sparql?query=SELECT%20*%20WHERE%20%7B%3Fs%20%3Fp%20%3Fo%7D
```

## API Endpoints

### Authorization Endpoints

#### GET /auth/oauth2/authorize
Initiates the OAuth2 authorization flow.

**Query Parameters:**
- `redirect_uri` (optional): Override default redirect URI
- `scope` (optional): Space-separated list of scopes
- `state` (optional): Client-provided state
- `use_pkce` (optional): Enable/disable PKCE (default: true)

#### GET /auth/oauth2/callback
Handles the OAuth2 authorization callback.

**Query Parameters:**
- `code`: Authorization code from provider
- `state`: State parameter for validation
- `error` (optional): Error from provider
- `error_description` (optional): Error details

### Token Management

#### POST /auth/oauth2/refresh
Refreshes an access token using a refresh token.

```json
{
  "refresh_token": "your_refresh_token"
}
```

#### GET /auth/oauth2/validate
Validates the current access token.

**Headers:**
- `Authorization: Bearer <access_token>`

### User Information

#### GET /auth/oauth2/userinfo
Retrieves user information from the OIDC provider.

**Headers:**
- `Authorization: Bearer <access_token>`

#### GET /auth/oauth2/config
Returns the current OAuth2 configuration (without secrets).

## Role and Permission Mapping

### Default Role Mapping

OIDC groups/roles are automatically mapped to OxiRS Fuseki roles:

| OIDC Group/Role | OxiRS Role | Permissions |
|-----------------|------------|-------------|
| admin, administrators | admin | Full access |
| writers, editors | writer | Read, Write, Query, Update |
| readers, viewers | reader | Read, Query only |
| (any other) | user | Read, Query only |

### Custom Role Mapping

You can configure custom role mappings in the configuration:

```toml
[security.oauth.role_mappings]
"engineering-team" = "writer"
"data-scientists" = "reader"
"platform-admins" = "admin"
```

## Security Considerations

### PKCE (Proof Key for Code Exchange)
- Enabled by default for all OAuth2 flows
- Provides protection against authorization code interception
- Recommended for all public clients

### State Parameter Validation
- Unique state parameter generated for each authorization request
- Validated on callback to prevent CSRF attacks
- Expires after 10 minutes

### Token Security
- Access tokens stored securely in memory
- Automatic cleanup of expired tokens
- No tokens persisted to disk

### HTTPS Requirements
- Always use HTTPS in production
- Configure TLS in the server configuration
- Redirect URIs must use HTTPS (except localhost for development)

## Troubleshooting

### Common Issues

#### "OAuth2 authentication not configured"
Ensure OAuth2 configuration is present in your config file and all required fields are provided.

#### "Invalid or expired OAuth2 state"
The state parameter has expired (10-minute timeout) or doesn't match. Restart the authentication flow.

#### "Failed to fetch OIDC discovery"
Check that the auth_url is correct and the provider supports OIDC discovery.

#### "Token exchange failed"
Verify client_id and client_secret are correct, and the redirect_uri matches exactly.

### Debug Logging

Enable debug logging for OAuth2:

```toml
[logging]
level = "debug"
auth = "trace"
```

### Testing OAuth2 Flow

1. **Test Provider Connection**:
```bash
curl http://localhost:3030/auth/oauth2/config
```

2. **Test Authorization URL Generation**:
```bash
curl "http://localhost:3030/auth/oauth2/authorize?redirect_uri=http://localhost:3030/callback"
```

3. **Test OIDC Discovery**:
```bash
curl http://localhost:3030/auth/oauth2/.well-known/openid-configuration
```

## Integration Examples

### JavaScript/TypeScript
```typescript
// Initiate OAuth2 login
async function initiateOAuth2Login() {
  const response = await fetch('/auth/oauth2/authorize');
  const data = await response.json();
  
  if (data.success) {
    // Save state for verification
    sessionStorage.setItem('oauth2_state', data.state);
    // Redirect to authorization URL
    window.location.href = data.authorization_url;
  }
}

// Handle callback (in callback page)
async function handleOAuth2Callback() {
  const params = new URLSearchParams(window.location.search);
  const code = params.get('code');
  const state = params.get('state');
  
  // Verify state matches
  const savedState = sessionStorage.getItem('oauth2_state');
  if (state !== savedState) {
    console.error('State mismatch - possible CSRF attack');
    return;
  }
  
  // The server will handle the token exchange automatically
  // and set the session cookie
}

// Use the authenticated session
async function querySPARQL(query) {
  const response = await fetch('/sparql', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/sparql-query',
    },
    body: query,
    credentials: 'include', // Include session cookie
  });
  
  return response.json();
}
```

### Python
```python
import requests
from urllib.parse import urlencode

class OAuth2Client:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
    
    def initiate_login(self, redirect_uri=None):
        """Initiate OAuth2 login flow"""
        params = {}
        if redirect_uri:
            params['redirect_uri'] = redirect_uri
        
        response = self.session.get(
            f"{self.base_url}/auth/oauth2/authorize",
            params=params
        )
        
        data = response.json()
        if data['success']:
            return data['authorization_url'], data['state']
        else:
            raise Exception(data['message'])
    
    def handle_callback(self, callback_url):
        """Handle OAuth2 callback"""
        # The callback URL will be handled by the server
        # Just make a request to any protected endpoint
        # The session cookie will be set automatically
        pass
    
    def query_sparql(self, query):
        """Execute SPARQL query with authentication"""
        response = self.session.post(
            f"{self.base_url}/sparql",
            data=query,
            headers={'Content-Type': 'application/sparql-query'}
        )
        
        return response.json()

# Usage
client = OAuth2Client('http://localhost:3030')
auth_url, state = client.initiate_login()
print(f"Visit: {auth_url}")
# After authentication, use the session
results = client.query_sparql("SELECT * WHERE { ?s ?p ?o } LIMIT 10")
```

## Advanced Configuration

### Multiple OAuth2 Providers

While the current implementation supports one OAuth2 provider at a time, you can implement provider selection logic in your application layer.

### Token Refresh Strategy

The system automatically handles token refresh when:
1. A refresh token is available
2. The access token is expired or near expiration
3. The refresh token itself is still valid

### Session Management

OAuth2 sessions are managed similarly to regular sessions:
- Default timeout: 3600 seconds (1 hour)
- Configurable via `security.session.timeout_secs`
- Automatic cleanup of expired sessions

## Future Enhancements

Planned improvements for OAuth2/OIDC support:

1. **Multiple Provider Support**: Allow configuration of multiple OAuth2 providers simultaneously
2. **Dynamic Client Registration**: Support for OAuth2 Dynamic Client Registration
3. **Token Introspection**: RFC 7662 token introspection endpoint support
4. **Proof of Possession**: Support for DPoP (Demonstrating Proof of Possession)
5. **Fine-grained Scopes**: Map OAuth2 scopes to specific OxiRS permissions
6. **SSO Logout**: Implement Single Sign-Out with OIDC providers
7. **JWT Access Tokens**: Direct JWT validation without provider round-trips
8. **Federated Queries**: Use OAuth2 tokens for federated SPARQL queries

## Conclusion

The OAuth2/OIDC implementation in OxiRS Fuseki provides enterprise-grade authentication with support for major identity providers. With automatic role mapping, PKCE security, and comprehensive token management, it offers a secure and user-friendly authentication solution for SPARQL endpoints.