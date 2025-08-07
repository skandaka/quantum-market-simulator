# Docker Deployment Guide

This guide provides comprehensive instructions for deploying the Quantum Market Simulator using Docker in various environments.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Production Docker Configuration](#production-docker-configuration)
4. [Environment Variables](#environment-variables)
5. [Database Setup](#database-setup)
6. [SSL/TLS Configuration](#ssltls-configuration)
7. [Monitoring Setup](#monitoring-setup)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **Docker**: Version 20.10+ 
- **Docker Compose**: Version 2.0+
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **CPU**: 4 cores minimum for quantum simulations

### Required Software
```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

## Local Development Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-org/quantum-market-simulator.git
cd quantum-market-simulator
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 3. Build and Start Services
```bash
# Build all services
docker-compose build

# Start in development mode
docker-compose up -d

# View logs
docker-compose logs -f
```

### 4. Verify Installation
```bash
# Check service health
docker-compose ps

# Test API endpoint
curl http://localhost:8000/api/v1/health

# Access frontend
open http://localhost:3000
```

## Production Docker Configuration

### 1. Production Docker Compose
Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./ssl:/etc/ssl/certs
    environment:
      - NODE_ENV=production
    depends_on:
      - backend
    restart: unless-stopped

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - QUANTUM_API_KEY=${QUANTUM_API_KEY}
    volumes:
      - ./logs:/app/logs
    depends_on:
      - database
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"

  database:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - frontend
      - backend
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 2. Production Dockerfile (Backend)
```dockerfile
# backend/Dockerfile.prod
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "app.main:app"]
```

### 3. Production Dockerfile (Frontend)
```dockerfile
# frontend/Dockerfile.prod
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80 443
CMD ["nginx", "-g", "daemon off;"]
```

## Environment Variables

### Core Configuration
```bash
# .env.production
ENVIRONMENT=production
DEBUG=false

# Database
DATABASE_URL=postgresql://user:password@database:5432/quantum_market_db
DB_HOST=database
DB_PORT=5432
DB_NAME=quantum_market_db
DB_USER=quantum_user
DB_PASSWORD=your_secure_password

# Redis
REDIS_URL=redis://redis:6379/0

# Security
SECRET_KEY=your_super_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key

# External APIs
QUANTUM_API_KEY=your_quantum_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_news_api_key

# Monitoring
SENTRY_DSN=your_sentry_dsn
LOG_LEVEL=INFO

# Performance
WORKERS=4
MAX_CONNECTIONS=1000
QUANTUM_TIMEOUT=300
```

### Security Configuration
```bash
# SSL/TLS
SSL_CERT_PATH=/etc/ssl/certs/cert.pem
SSL_KEY_PATH=/etc/ssl/certs/key.pem
SSL_REDIRECT=true

# CORS
CORS_ORIGINS=["https://yourdomain.com"]
CORS_CREDENTIALS=true

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_BURST=200
```

## Database Setup

### 1. PostgreSQL Configuration
```bash
# Initialize database
docker-compose exec database psql -U postgres -c "CREATE DATABASE quantum_market_db;"
docker-compose exec database psql -U postgres -c "CREATE USER quantum_user WITH PASSWORD 'your_password';"
docker-compose exec database psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE quantum_market_db TO quantum_user;"

# Run migrations
docker-compose exec backend python -m alembic upgrade head
```

### 2. Database Optimization
```sql
-- postgresql.conf optimizations
shared_preload_libraries = 'pg_stat_statements'
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
```

### 3. Backup Configuration
```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec -T database pg_dump -U quantum_user quantum_market_db > $BACKUP_DIR/backup_$DATE.sql
find $BACKUP_DIR -name "backup_*.sql" -mtime +7 -delete
EOF

chmod +x backup.sh

# Add to crontab
echo "0 2 * * * /path/to/backup.sh" | crontab -
```

## SSL/TLS Configuration

### 1. Generate SSL Certificates
```bash
# Using Let's Encrypt (recommended)
certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Copy certificates
mkdir -p ssl
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/cert.pem
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/key.pem
```

### 2. Nginx SSL Configuration
```nginx
# nginx/nginx.conf
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/certs/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;

    location / {
        proxy_pass http://frontend:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    location /ws/ {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring Setup

### 1. Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'quantum-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['database:5432']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
```

### 2. Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Quantum Market Simulator",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(http_request_duration_seconds)",
            "legendFormat": "Response Time"
          }
        ]
      },
      {
        "title": "Quantum Simulation Success Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(quantum_simulations_success_total[5m]) / rate(quantum_simulations_total[5m]) * 100"
          }
        ]
      }
    ]
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check logs
docker-compose logs service_name

# Check container status
docker-compose ps

# Restart specific service
docker-compose restart service_name
```

#### 2. Database Connection Issues
```bash
# Test database connectivity
docker-compose exec backend python -c "import psycopg2; print('DB connection successful')"

# Check database logs
docker-compose logs database

# Reset database
docker-compose down -v
docker-compose up -d database
```

#### 3. Memory Issues
```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
```

#### 4. SSL Certificate Issues
```bash
# Verify certificate
openssl x509 -in ssl/cert.pem -text -noout

# Test SSL connection
openssl s_client -connect yourdomain.com:443

# Renew Let's Encrypt certificate
certbot renew --dry-run
```

### Performance Optimization

#### 1. Resource Limits
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

#### 2. Caching Configuration
```bash
# Redis tuning
echo 'vm.overcommit_memory = 1' >> /etc/sysctl.conf
echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf
sysctl -p
```

#### 3. Database Tuning
```sql
-- Optimize for quantum workloads
ALTER SYSTEM SET shared_buffers = '25% of RAM';
ALTER SYSTEM SET effective_cache_size = '75% of RAM';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
SELECT pg_reload_conf();
```

### Health Checks

#### 1. Automated Health Monitoring
```bash
#!/bin/bash
# health-check.sh

services=("frontend" "backend" "database" "redis")

for service in "${services[@]}"; do
    if ! docker-compose ps $service | grep -q "Up"; then
        echo "ALERT: $service is down"
        # Send alert (email, Slack, etc.)
    fi
done
```

#### 2. Application Health Endpoints
```bash
# Test all health endpoints
curl -f http://localhost:8000/api/v1/health || echo "Backend unhealthy"
curl -f http://localhost:3000/health || echo "Frontend unhealthy"
curl -f http://localhost:9090/-/healthy || echo "Prometheus unhealthy"
```

### Backup and Recovery

#### 1. Full System Backup
```bash
#!/bin/bash
# backup-system.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/system_$DATE"

mkdir -p $BACKUP_DIR

# Database backup
docker-compose exec -T database pg_dump -U quantum_user quantum_market_db > $BACKUP_DIR/database.sql

# Redis backup
docker-compose exec redis redis-cli BGSAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb $BACKUP_DIR/

# Application data
docker-compose exec backend tar -czf - /app/data > $BACKUP_DIR/app_data.tar.gz

# Configuration
cp -r . $BACKUP_DIR/config/

echo "Backup completed: $BACKUP_DIR"
```

#### 2. Disaster Recovery
```bash
#!/bin/bash
# disaster-recovery.sh

BACKUP_DIR=$1

# Stop services
docker-compose down

# Restore database
docker-compose up -d database
sleep 10
cat $BACKUP_DIR/database.sql | docker-compose exec -T database psql -U quantum_user quantum_market_db

# Restore Redis
docker cp $BACKUP_DIR/dump.rdb $(docker-compose ps -q redis):/data/
docker-compose restart redis

# Restore application data
docker-compose exec -T backend tar -xzf - -C / < $BACKUP_DIR/app_data.tar.gz

# Start all services
docker-compose up -d

echo "Recovery completed from: $BACKUP_DIR"
```

### Maintenance

#### 1. Regular Maintenance Tasks
```bash
# Weekly maintenance script
#!/bin/bash
# maintenance.sh

# Update containers
docker-compose pull
docker-compose up -d

# Clean up old images
docker system prune -f

# Backup database
./backup.sh

# Update SSL certificates
certbot renew --quiet

# Restart services (if needed)
docker-compose restart nginx
```

#### 2. Log Rotation
```bash
# Configure logrotate
cat > /etc/logrotate.d/quantum-simulator << 'EOF'
/path/to/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    notifempty
    create 644 root root
    postrotate
        docker-compose restart backend frontend
    endscript
}
EOF
```

This completes the comprehensive Docker deployment guide. For additional help, refer to the [troubleshooting section](#troubleshooting) or contact support.
