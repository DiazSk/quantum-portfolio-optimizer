# Real-time Risk Monitoring System - External Dependencies Configuration

## Overview
The real-time risk monitoring system requires several external services to function properly. This document provides configuration guidance for all external dependencies.

## Required External Services

### 1. Redis Cache Server
**Purpose**: High-performance caching for real-time risk metrics

**Installation:**
```bash
# Windows (using chocolatey)
choco install redis-64

# Linux/macOS
brew install redis

# Docker
docker run -d -p 6379:6379 redis:latest
```

**Configuration:**
```env
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=<optional_password>
RISK_CACHE_DURATION=60
```

**Testing Connection:**
```python
import redis
client = redis.from_url("redis://localhost:6379/0")
client.ping()  # Should return True
```

### 2. SMTP Email Service
**Purpose**: Email notifications for risk alerts

**Supported Providers:**
- Gmail SMTP
- Outlook/Hotmail SMTP
- Corporate SMTP servers
- SendGrid, AWS SES, etc.

**Configuration:**
```env
# Gmail Example
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_USE_TLS=true

# Corporate SMTP Example
SMTP_HOST=smtp.company.com
SMTP_PORT=587
SMTP_USERNAME=risk-alerts@company.com
SMTP_PASSWORD=<secure_password>
SMTP_USE_TLS=true
```

**Testing Email:**
```python
import smtplib
from email.mime.text import MIMEText

msg = MIMEText("Test email")
msg['Subject'] = "Risk Monitoring Test"
msg['From'] = "risk-alerts@company.com"
msg['To'] = "test@company.com"

with smtplib.SMTP('smtp.company.com', 587) as server:
    server.starttls()
    server.login('username', 'password')
    server.send_message(msg)
```

### 3. SMS Service (Optional)
**Purpose**: SMS notifications for critical alerts

#### Option A: Twilio
**Setup:**
1. Create Twilio account at https://www.twilio.com/
2. Get Account SID and Auth Token
3. Purchase a phone number

**Configuration:**
```env
TWILIO_ACCOUNT_SID=<your_account_sid>
TWILIO_AUTH_TOKEN=<your_auth_token>
TWILIO_PHONE_NUMBER=+1234567890
```

**Testing:**
```python
from twilio.rest import Client

client = Client(account_sid, auth_token)
message = client.messages.create(
    body="Test SMS alert",
    from_='+1234567890',
    to='+0987654321'
)
```

#### Option B: AWS SNS
**Setup:**
1. Configure AWS credentials
2. Enable SNS service
3. Configure SMS settings

**Configuration:**
```env
AWS_ACCESS_KEY_ID=<your_access_key>
AWS_SECRET_ACCESS_KEY=<your_secret_key>
AWS_SNS_REGION=us-east-1
```

### 4. PostgreSQL Database
**Purpose**: Alert configuration and audit trail storage

**Installation:**
```bash
# Windows
# Download from https://www.postgresql.org/download/windows/

# Linux
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:13
```

**Configuration:**
```env
DATABASE_URL=postgresql://username:password@localhost:5432/portfolio_db
```

**Database Setup:**
```sql
-- Run the migration script
\i src/database/migrations/002_add_risk_alert_tables.sql
```

### 5. WebSocket Server
**Purpose**: Real-time risk data broadcasting

**No external setup required** - built into FastAPI application

**Configuration:**
```env
WEBSOCKET_PORT=8000
WEBSOCKET_JWT_SECRET=<secure_jwt_secret>
```

## Environment Configuration Template

Create a `.env` file in the project root:

```env
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=
RISK_CACHE_DURATION=60

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_USE_TLS=true

# SMS Configuration (Optional - choose one)
# Twilio
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=

# AWS SNS
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_SNS_REGION=us-east-1

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/portfolio_db

# WebSocket Configuration
WEBSOCKET_PORT=8000
WEBSOCKET_JWT_SECRET=your-secret-key-here

# Risk Monitoring Settings
RISK_REFRESH_INTERVAL=30
MAX_ALERT_FREQUENCY=300
DEFAULT_RISK_THRESHOLDS_ENABLED=true
```

## Development vs Production Configurations

### Development Environment
```env
# Use local services
REDIS_URL=redis://localhost:6379/0
SMTP_HOST=smtp.gmail.com  # Personal Gmail for testing
DATABASE_URL=postgresql://dev:dev@localhost:5432/portfolio_dev
```

### Production Environment
```env
# Use managed services
REDIS_URL=redis://redis-cluster.prod.company.com:6379/0
SMTP_HOST=smtp.company.com  # Corporate SMTP
DATABASE_URL=postgresql://prod_user:secure_pass@db.prod.company.com:5432/portfolio_prod
```

## Service Health Checks

### Quick Health Check Script
```python
#!/usr/bin/env python3
"""
Health check script for real-time risk monitoring dependencies
"""

import redis
import smtplib
import psycopg2
import sys
from datetime import datetime

def check_redis():
    try:
        client = redis.from_url("redis://localhost:6379/0")
        client.ping()
        print("✅ Redis: Connected")
        return True
    except Exception as e:
        print(f"❌ Redis: {e}")
        return False

def check_smtp():
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.quit()
        print("✅ SMTP: Connection successful")
        return True
    except Exception as e:
        print(f"❌ SMTP: {e}")
        return False

def check_database():
    try:
        conn = psycopg2.connect("postgresql://username:password@localhost:5432/portfolio_db")
        conn.close()
        print("✅ Database: Connected")
        return True
    except Exception as e:
        print(f"❌ Database: {e}")
        return False

if __name__ == "__main__":
    print(f"Health Check - {datetime.now()}")
    print("-" * 40)
    
    checks = [
        check_redis(),
        check_smtp(),
        check_database()
    ]
    
    if all(checks):
        print("\n🎉 All services healthy!")
        sys.exit(0)
    else:
        print("\n⚠️ Some services need attention")
        sys.exit(1)
```

## Troubleshooting

### Common Issues

**Redis Connection Failed:**
- Check if Redis server is running
- Verify port 6379 is not blocked
- Test with `redis-cli ping`

**SMTP Authentication Failed:**
- Use app-specific passwords for Gmail
- Check if 2FA is enabled on email account
- Verify SMTP server settings

**SMS Service Failed:**
- Check Twilio account balance
- Verify phone number format (+1234567890)
- Test API credentials independently

**Database Connection Failed:**
- Check PostgreSQL service status
- Verify database exists and user permissions
- Test connection with `psql` client

### Performance Considerations

**Redis Memory Usage:**
- Monitor memory usage with `redis-cli info memory`
- Set appropriate `maxmemory` configuration
- Use Redis TTL for automatic cleanup

**Email Rate Limits:**
- Gmail: 500 emails/day for free accounts
- Corporate SMTP: Check with IT department
- Consider queueing for high-volume alerts

**SMS Costs:**
- Twilio: ~$0.0075 per SMS in US
- AWS SNS: ~$0.00645 per SMS
- Budget for expected alert volume

## Security Best Practices

1. **Use Environment Variables**: Never hardcode credentials
2. **Secure Secret Storage**: Use AWS Secrets Manager, Azure Key Vault, etc.
3. **Network Security**: Use VPNs, firewalls for production
4. **Access Control**: Limit database and Redis access
5. **Audit Logging**: Monitor all notification activities
6. **Regular Updates**: Keep all services updated

## Backup and Recovery

**Redis:**
- Enable RDB snapshots
- Use AOF for durability
- Backup strategy for critical cache data

**Database:**
- Regular PostgreSQL backups
- Point-in-time recovery setup
- Test restore procedures

**Configuration:**
- Version control for all configuration files
- Document all configuration changes
- Maintain environment-specific configs

## Monitoring and Alerting

**Service Monitoring:**
- Use health check endpoints
- Monitor service availability (99.9% uptime)
- Set up alerting for service failures

**Performance Monitoring:**
- Redis response times (<1ms)
- Email delivery rates (>99%)
- SMS delivery rates (>98%)
- Database query performance

**Capacity Planning:**
- Redis memory growth trends
- Email/SMS volume projections
- Database storage requirements
