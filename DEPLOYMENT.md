# Face Embedding API - Docker Deployment Guide

## 📦 Container Deployment

This API is containerized and ready for deployment on Ubuntu cloud clusters.

## 🚀 Quick Start

### 1. Prerequisites
- Docker (version 20.10+)
- Docker Compose (version 1.29+)
- Ubuntu 20.04 or later

### 2. Deploy Locally

```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### 3. Manual Deployment

```bash
# Build the image
docker-compose build

# Start the container
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f face-embedding-api
```

## 🌐 Production Deployment on Cloud

### Option 1: Docker on Ubuntu Server

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 2. Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# 3. Clone your repository
git clone <your-repo-url>
cd python-service

# 4. Create .env file with credentials
nano .env

# 5. Deploy
chmod +x deploy.sh
sudo ./deploy.sh
```

### Option 2: AWS ECS/Fargate

```bash
# 1. Build and tag image
docker build -t face-embedding-api:latest .

# 2. Tag for ECR
docker tag face-embedding-api:latest <aws-account-id>.dkr.ecr.<region>.amazonaws.com/face-embedding-api:latest

# 3. Push to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws-account-id>.dkr.ecr.<region>.amazonaws.com
docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/face-embedding-api:latest

# 4. Deploy to ECS using AWS Console or CLI
```

### Option 3: Kubernetes

```bash
# 1. Build and push to registry
docker build -t your-registry/face-embedding-api:v1 .
docker push your-registry/face-embedding-api:v1

# 2. Create Kubernetes deployment (see k8s-deployment.yaml below)
kubectl apply -f k8s-deployment.yaml

# 3. Create service
kubectl apply -f k8s-service.yaml
```

## 📝 Kubernetes Example

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-embedding-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: face-embedding-api
  template:
    metadata:
      labels:
        app: face-embedding-api
    spec:
      containers:
      - name: face-embedding-api
        image: your-registry/face-embedding-api:v1
        ports:
        - containerPort: 5010
        env:
        - name: BUCKET_NAME
          valueFrom:
            secretKeyRef:
              name: s3-credentials
              key: bucket-name
        - name: BUCKET_REGION
          valueFrom:
            secretKeyRef:
              name: s3-credentials
              key: bucket-region
        - name: ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: s3-credentials
              key: access-key
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: s3-credentials
              key: secret-key
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
            port: 5010
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 5010
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: face-embedding-api-service
spec:
  selector:
    app: face-embedding-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5010
  type: LoadBalancer
```

## 🔧 Configuration

### Environment Variables

Required in `.env` file:

```env
BUCKET_NAME=emojotdevstorage
BUCKET_REGION=us-west-2
ACCESS_KEY=your-access-key
SECRET_KEY=your-secret-key
```

### Resource Requirements

**Minimum:**
- CPU: 1 core
- RAM: 2GB
- Disk: 5GB

**Recommended (Production):**
- CPU: 2 cores
- RAM: 4GB
- Disk: 10GB

## 🐳 Docker Commands

```bash
# Build image
docker build -t face-embedding-api:latest .

# Run container
docker run -d \
  --name face-embed-api \
  -p 5010:5010 \
  --env-file .env \
  face-embedding-api:latest

# Stop container
docker stop face-embed-api

# Remove container
docker rm face-embed-api

# View logs
docker logs -f face-embed-api

# Execute shell in container
docker exec -it face-embed-api /bin/bash
```

## 🧪 Testing Deployment

```bash
# Test health endpoint
curl http://localhost:5010/health

# Test face embedding extraction
curl -X POST http://localhost:5010/api/face-embedding \
  -H "Content-Type: application/json" \
  -d '{"s3_url": "s3://your-bucket/path/to/image.jpg"}'
```

## 📊 Monitoring

### View Container Stats
```bash
docker stats face-embed-api
```

### View Logs
```bash
# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs face-embedding-api
```

## 🔄 Updates

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 🛡️ Security Best Practices

1. **Never commit .env file** - Add to .gitignore
2. **Use secrets management** - AWS Secrets Manager, HashiCorp Vault
3. **Limit container resources** - Set CPU/memory limits
4. **Run as non-root user** (optional enhancement)
5. **Scan images for vulnerabilities** - Use `docker scan` or Trivy

## 🚨 Troubleshooting

### Container won't start
```bash
docker-compose logs face-embedding-api
```

### Out of memory
```bash
# Increase memory limit in docker-compose.yml
memory: 8G
```

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "5011:5010"
```

### Models not downloading
```bash
# Exec into container and check internet connectivity
docker exec -it face-embed-api /bin/bash
curl -I https://github.com
```

## 📞 API Endpoints

- **Health Check:** `GET /health`
- **Extract Embedding:** `POST /api/face-embedding`

## 📈 Scaling

### Horizontal Scaling (Multiple Instances)

```bash
# Scale to 3 instances
docker-compose up -d --scale face-embedding-api=3
```

### Load Balancer (Nginx Example)

```nginx
upstream face_embed_backend {
    server localhost:5010;
    server localhost:5011;
    server localhost:5012;
}

server {
    listen 80;
    location / {
        proxy_pass http://face_embed_backend;
    }
}
```

## 📦 Image Size Optimization

Current image size: ~2.5GB (includes all ML models)

To reduce size:
- Use multi-stage builds
- Remove unused models
- Use Alpine Linux (requires additional dependencies)

## 🎯 Production Checklist

- [ ] `.env` file configured with production credentials
- [ ] Resource limits set appropriately
- [ ] Health checks configured
- [ ] Logging configured (centralized logging)
- [ ] Monitoring set up (Prometheus/Grafana)
- [ ] Auto-restart enabled
- [ ] Backup strategy in place
- [ ] SSL/TLS configured (if exposed directly)
- [ ] Rate limiting configured
- [ ] Security scanning completed

---

**Version:** 3.0  
**Last Updated:** November 2025
