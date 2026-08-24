# Deployment Guide - Fall Detection Cloud Backend

## Prerequisites

- Docker & Docker Compose installed
- Python 3.10+ (for local development)
- VPS or server with at least:
  - 4 CPU cores
  - 8 GB RAM
  - 50 GB disk space
  - Ubuntu 20.04+ or Windows Server

## 1. Environment Setup

### 1.1 Clone Repository

```bash
git clone <repository-url>
cd fall_detection_cloud
```

### 1.2 Create Environment Variables

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<secure_password>
POSTGRES_DB=falldetection

# RabbitMQ/MQTT
RABBITMQ_USER=<secure_user>
RABBITMQ_PASS=<secure_password>

# MinIO
MINIO_ROOT_USER=<secure_user>
MINIO_ROOT_PASSWORD=<secure_password>
MINIO_ENDPOINT=localhost:9000
MINIO_SECURE=false

# VLM (Gemini Flash)
GEMINI_API_KEY=<your_gemini_api_key>
VLM_TIMEOUT_SEC=8.0

# FCM (Firebase)
FCM_CREDENTIALS_FILE=/path/to/firebase-credentials.json

# Telegram
TELEGRAM_BOT_TOKEN=<your_bot_token>
TELEGRAM_CHAT_ID=<your_chat_id>

# Deduplication
DEDUP_WINDOW_SEC=2.0

# Telemetry
TELEMETRY_ONLINE_WINDOW_SEC=120

# VPS IP (for external access)
VPS_IP=<your_vps_ip>
```

## 2. Docker Deployment

### 2.1 Start All Services

```bash
docker-compose -f docker-compose.cloud.yml up -d
```

This will start:
- PostgreSQL (port 5432)
- RabbitMQ with MQTT plugin (ports 5672, 1883, 15672)
- MinIO (ports 9000, 9001)
- MediaMTX (ports 8554, 8888, 8889, 1935)
- FastAPI Backend (port 8000)

### 2.2 Check Service Status

```bash
docker-compose -f docker-compose.cloud.yml ps
```

### 2.3 View Logs

```bash
# All services
docker-compose -f docker-compose.cloud.yml logs -f

# Specific service
docker-compose -f docker-compose.cloud.yml logs -f backend
```

### 2.4 Stop Services

```bash
docker-compose -f docker-compose.cloud.yml down
```

## 3. Database Initialization

### 3.1 Run Database Migrations

```bash
docker-compose -f docker-compose.cloud.yml exec backend alembic upgrade head
```

### 3.2 Setup MinIO Buckets

```bash
docker-compose -f docker-compose.cloud.yml exec backend python scripts/setup_minio.py
```

This creates:
- `fall-events` bucket
- `false-positives` bucket
- `clips` bucket

## 4. Backup Configuration

### 4.1 Windows (Local/VPS)

Create a scheduled task to run the backup script:

```bash
# Edit the backup script to match your environment
# scripts/backup_database.bat

# Schedule via Windows Task Scheduler
# Trigger: Daily at 2:00 AM
# Action: Run scripts\backup_database.bat
```

### 4.2 Linux (VPS)

Create a cron job:

```bash
# Edit crontab
crontab -e

# Add this line for daily backup at 2:00 AM
0 2 * * * /path/to/scripts/backup_database.sh
```

## 5. Firewall Configuration

### 5.1 Required Ports

- **8000**: FastAPI Backend API
- **5432**: PostgreSQL (internal only)
- **5672**: RabbitMQ (internal only)
- **1883**: MQTT (from Edge)
- **15672**: RabbitMQ Management UI (optional)
- **9000**: MinIO API
- **9001**: MinIO Console (optional)
- **8554**: RTSP (from Edge)
- **8888**: HLS (to App)
- **8889**: WebRTC (optional)
- **1935**: RTMP (optional)

### 5.2 UFW (Ubuntu)

```bash
# Allow required ports
sudo ufw allow 8000/tcp
sudo ufw allow 1883/tcp
sudo ufw allow 9000/tcp
sudo ufw allow 9001/tcp
sudo ufw allow 8554/tcp
sudo ufw allow 8888/tcp
sudo ufw allow 8889/tcp
sudo ufw allow 1935/tcp

# Enable firewall
sudo ufw enable
```

### 5.3 Windows Firewall

Add inbound rules for the required ports through Windows Firewall with Advanced Security.

## 6. SSL/TLS Configuration (Production)

### 6.1 Using Nginx Reverse Proxy

Create `nginx.conf`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://localhost:8888;
    }
}
```

### 6.2 Using Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 7. Monitoring

### 7.1 Health Check

```bash
# Backend health
curl http://localhost:8000/health

# RabbitMQ Management
# Open http://localhost:15672 in browser
# Username: guest (or configured user)
# Password: guest (or configured password)

# MinIO Console
# Open http://localhost:9001 in browser
```

### 7.2 Log Monitoring

```bash
# Real-time logs
docker-compose -f docker-compose.cloud.yml logs -f backend

# Check for errors
docker-compose -f docker-compose.cloud.yml logs backend | grep ERROR
```

## 8. Troubleshooting

### 8.1 Container Won't Start

```bash
# Check logs
docker-compose -f docker-compose.cloud.yml logs <service_name>

# Restart specific service
docker-compose -f docker-compose.cloud.yml restart <service_name>

# Rebuild service
docker-compose -f docker-compose.cloud.yml up -d --build <service_name>
```

### 8.2 Database Connection Failed

```bash
# Check PostgreSQL is running
docker-compose -f docker-compose.cloud.yml ps db

# Check database logs
docker-compose -f docker-compose.cloud.yml logs db

# Verify credentials in .env
```

### 8.3 MQTT Connection Failed

```bash
# Check RabbitMQ is running
docker-compose -f docker-compose.cloud.yml ps rabbitmq

# Check MQTT plugin is enabled
docker-compose -f docker-compose.cloud.yml exec rabbitmq rabbitmq-plugins list

# Enable MQTT plugin if needed
docker-compose -f docker-compose.cloud.yml exec rabbitmq rabbitmq-plugins enable rabbitmq_mqtt
```

### 8.4 MinIO Connection Failed

```bash
# Check MinIO is running
docker-compose -f docker-compose.cloud.yml ps minio

# Re-run setup script
docker-compose -f docker-compose.cloud.yml exec backend python scripts/setup_minio.py
```

### 8.5 MediaMTX Stream Not Working

```bash
# Check MediaMTX is running
docker-compose -f docker-compose.cloud.yml ps mediamtx

# Check MediaMTX logs
docker-compose -f docker-compose.cloud.yml logs mediamtx

# Verify mediamtx.yml configuration
```

## 9. Testing Deployment

### 9.1 Run Full Trigger Test

```bash
docker-compose -f docker-compose.cloud.yml exec backend python scripts/test_full_trigger.py
```

### 9.2 Run Load Test

```bash
docker-compose -f docker-compose.cloud.yml exec backend python scripts/test_load.py --events 50
```

### 9.3 Verify API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# API documentation
# Open http://localhost:8000/docs in browser
```

## 10. Production Checklist

- [ ] Environment variables configured in `.env`
- [ ] All secrets changed from defaults
- [ ] Firewall configured
- [ ] SSL/TLS enabled
- [ ] Database backup scheduled
- [ ] MinIO buckets created
- [ ] MediaMTX configured
- [ ] Health check endpoint accessible
- [ ] Load test passed
- [ ] Full trigger test passed
- [ ] Log rotation configured
- [ ] Monitoring set up

## 11. Scaling Considerations

### 11.1 Horizontal Scaling

For higher load, you can scale the backend service:

```bash
docker-compose -f docker-compose.cloud.yml up -d --scale backend=3
```

Note: You'll need a load balancer (Nginx, HAProxy) in front of multiple backend instances.

### 11.2 Database Optimization

For production, consider:
- PostgreSQL connection pooling
- Database indexing optimization
- Read replicas for read-heavy workloads

### 11.3 MinIO Scaling

For large-scale storage:
- Use distributed MinIO setup
- Configure lifecycle policies for old data
- Enable versioning for critical data

## 12. Security Best Practices

- Never commit `.env` file to version control
- Use strong passwords for all services
- Enable SSL/TLS for all external connections
- Regularly update Docker images
- Implement rate limiting on API endpoints
- Use secrets management (HashiCorp Vault, AWS Secrets Manager) for production
- Enable audit logging for sensitive operations
- Regular security audits and penetration testing

## 13. Support

For issues or questions:
- Check logs: `docker-compose -f docker-compose.cloud.yml logs`
- Review documentation in `docs/` directory
- Contact: Slack channel #fall-detection-cloud
