# OxiRS Fuseki Ansible Playbooks

Ansible playbooks for deploying and managing OxiRS Fuseki SPARQL servers.

## Prerequisites

- Ansible 2.14+ installed on your control machine
- SSH access to target servers
- Python 3.8+ on target servers
- Sudo privileges on target servers

## Quick Start

### 1. Install Ansible

```bash
# On macOS
brew install ansible

# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install ansible

# Using pip
pip3 install ansible
```

### 2. Configure Inventory

Edit the appropriate inventory file for your environment:
- `inventory/production` - Production servers
- `inventory/staging` - Staging servers
- `inventory/local` - Local development

### 3. Configure Variables

Edit variables in:
- `group_vars/all.yml` - Global defaults
- `group_vars/production.yml` - Production overrides
- `host_vars/<hostname>.yml` - Host-specific overrides

### 4. Run the Playbook

```bash
# Deploy to production
ansible-playbook -i inventory/production site.yml

# Deploy to staging
ansible-playbook -i inventory/staging site.yml

# Deploy to local machine
ansible-playbook -i inventory/local site.yml
```

## Directory Structure

```
ansible/
├── site.yml                  # Main playbook
├── ansible.cfg               # Ansible configuration
├── inventory/                # Inventory files
│   ├── production           # Production inventory
│   ├── staging              # Staging inventory
│   └── local                # Local development
├── group_vars/              # Group variables
│   ├── all.yml              # Global variables
│   └── production.yml       # Production variables
├── host_vars/               # Host-specific variables
├── roles/                   # Ansible roles
│   ├── common/              # Common system setup
│   ├── oxirs-fuseki/        # OxiRS Fuseki installation
│   ├── security/            # Security hardening
│   └── monitoring/          # Monitoring setup
└── README.md                # This file
```

## Roles

### common
System-level configuration including:
- Package installation
- NTP/time synchronization
- System limits and kernel parameters
- Log rotation

### oxirs-fuseki
OxiRS Fuseki installation and configuration:
- User and directory creation
- Binary installation (pre-built or from source)
- Configuration file generation
- Systemd service setup
- TLS certificate management

### security
Security hardening:
- Firewall configuration (UFW/firewalld)
- SSH hardening
- Fail2ban setup
- Audit logging (auditd)
- Automatic security updates
- Intrusion detection (optional)

### monitoring
Observability setup:
- Prometheus Node Exporter
- Prometheus server (monitoring hosts only)
- Log monitoring
- Health check scripts
- Alerting (optional)

## Usage Examples

### Deploy to All Servers

```bash
ansible-playbook -i inventory/production site.yml
```

### Deploy Only Configuration Changes

```bash
ansible-playbook -i inventory/production site.yml --tags=config
```

### Run Only Security Hardening

```bash
ansible-playbook -i inventory/production site.yml --tags=security
```

### Deploy to Specific Host

```bash
ansible-playbook -i inventory/production site.yml --limit fuseki-prod-01
```

### Check Mode (Dry Run)

```bash
ansible-playbook -i inventory/production site.yml --check
```

### Verbose Output

```bash
ansible-playbook -i inventory/production site.yml -v   # Basic
ansible-playbook -i inventory/production site.yml -vv  # More verbose
ansible-playbook -i inventory/production site.yml -vvv # Debug mode
```

## Tags

Available tags for selective execution:

- `common` - Common system setup
- `security` - Security hardening
- `oxirs` - OxiRS Fuseki installation
- `install` - Installation tasks
- `config` - Configuration tasks
- `monitoring` - Monitoring setup
- `verify` - Verification and health checks

## Variables

### Key Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `oxirs_version` | `0.1.0-rc.1` | OxiRS Fuseki version |
| `oxirs_http_port` | `3030` | HTTP port |
| `oxirs_https_port` | `3031` | HTTPS port |
| `enable_tls` | `false` | Enable TLS/HTTPS |
| `enable_auth` | `false` | Enable authentication |
| `enable_metrics` | `true` | Enable Prometheus metrics |
| `max_memory` | `4096m` | Maximum memory allocation |
| `max_threads` | `4` | Maximum worker threads |
| `log_level` | `info` | Logging level |

### Environment-Specific Variables

Variables can be overridden at multiple levels:
1. `group_vars/all.yml` - Global defaults
2. `group_vars/<environment>.yml` - Environment-specific
3. `host_vars/<hostname>.yml` - Host-specific
4. Command line: `-e "variable=value"`

## Security

### SSH Keys

Ensure SSH keys are properly configured:

```bash
# Generate SSH key pair
ssh-keygen -t ed25519 -C "ansible@oxirs"

# Copy to target servers
ssh-copy-id -i ~/.ssh/oxirs-production.pem user@server
```

### Secrets Management

For production, use Ansible Vault for sensitive data:

```bash
# Create encrypted file
ansible-vault create group_vars/production_vault.yml

# Edit encrypted file
ansible-vault edit group_vars/production_vault.yml

# Run playbook with vault
ansible-playbook -i inventory/production site.yml --ask-vault-pass
```

### TLS Certificates

For production, replace self-signed certificates with proper certificates:

1. Place certificates in `roles/oxirs-fuseki/files/`:
   - `production_server.crt`
   - `production_server.key`

2. Update `roles/oxirs-fuseki/tasks/main.yml` to copy production certificates

## Monitoring

### Prometheus

After deployment, Prometheus is available at:
- `http://monitoring-server:9090`

### Grafana

To add Grafana (optional):

```bash
# Add grafana role to monitoring servers
- role: grafana
  tags: [grafana, monitoring]
```

### Health Checks

Verify deployment:

```bash
# Check OxiRS Fuseki status
ansible -i inventory/production fuseki_servers -m shell -a "systemctl status oxirs-fuseki"

# Check HTTP endpoint
ansible -i inventory/production fuseki_servers -m uri -a "url=http://localhost:3030/$/ping"

# View logs
ansible -i inventory/production fuseki_servers -m shell -a "journalctl -u oxirs-fuseki -n 50"
```

## Troubleshooting

### Connection Issues

```bash
# Test connectivity
ansible -i inventory/production all -m ping

# Check SSH connectivity
ansible -i inventory/production all -m shell -a "hostname"
```

### Service Issues

```bash
# Check service status
ansible -i inventory/production fuseki_servers -m shell -a "systemctl status oxirs-fuseki"

# View logs
ansible -i inventory/production fuseki_servers -m shell -a "journalctl -u oxirs-fuseki --since '10 minutes ago'"

# Restart service
ansible -i inventory/production fuseki_servers -m systemd -a "name=oxirs-fuseki state=restarted" --become
```

### Configuration Validation

```bash
# Validate configuration syntax
oxirs-fuseki --config /etc/oxirs-fuseki/oxirs.toml --validate

# Check Ansible syntax
ansible-playbook site.yml --syntax-check

# Dry run
ansible-playbook -i inventory/production site.yml --check
```

## Maintenance

### Updating OxiRS Fuseki

1. Update version in `group_vars/all.yml`
2. Run playbook with install tag:

```bash
ansible-playbook -i inventory/production site.yml --tags=install
```

### Rolling Updates

For zero-downtime updates:

```bash
# Update one server at a time
ansible-playbook -i inventory/production site.yml --limit fuseki-prod-01
# Test the update
ansible-playbook -i inventory/production site.yml --limit fuseki-prod-02
# Continue with remaining servers
```

### Backup and Restore

```bash
# Backup data
ansible -i inventory/production fuseki_servers -m shell \
  -a "tar czf /tmp/oxirs-backup-$(date +%Y%m%d).tar.gz /var/lib/oxirs-fuseki"

# Download backups
ansible -i inventory/production fuseki_servers -m fetch \
  -a "src=/tmp/oxirs-backup-*.tar.gz dest=./backups/"
```

## Support

For issues and questions:
- GitHub: https://github.com/cool-japan/oxirs
- Documentation: https://github.com/cool-japan/oxirs/tree/main/docs
- Issues: https://github.com/cool-japan/oxirs/issues

## License

See the main OxiRS project for licensing information.
