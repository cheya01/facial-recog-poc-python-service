#!/bin/bash

# Face Embedding API Deployment Script
# This script builds and deploys the containerized API

set -e

echo "🚀 Starting Face Embedding API Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ Error: .env file not found!${NC}"
    echo "Please create a .env file with the following variables:"
    echo "  BUCKET_NAME=your-bucket-name"
    echo "  BUCKET_REGION=your-region"
    echo "  ACCESS_KEY=your-access-key"
    echo "  SECRET_KEY=your-secret-key"
    exit 1
fi

echo -e "${GREEN}✓ Found .env file${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Error: Docker is not installed!${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo -e "${GREEN}✓ Docker is installed${NC}"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠ Warning: docker-compose not found, trying docker compose...${NC}"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo -e "${GREEN}✓ Docker Compose is available${NC}"

# Stop existing containers
echo -e "${YELLOW}⏹ Stopping existing containers...${NC}"
$DOCKER_COMPOSE down || true

# Build the Docker image
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
$DOCKER_COMPOSE build --no-cache

# Start the container
echo -e "${YELLOW}▶️  Starting container...${NC}"
$DOCKER_COMPOSE up -d

# Wait for health check
echo -e "${YELLOW}⏳ Waiting for API to be healthy...${NC}"
sleep 10

# Check health
MAX_RETRIES=12
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:5010/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is healthy and running!${NC}"
        echo ""
        echo "📊 Container Status:"
        $DOCKER_COMPOSE ps
        echo ""
        echo "🌐 API Endpoints:"
        echo "  Health Check: http://localhost:5010/health"
        echo "  Face Embedding: http://localhost:5010/api/face-embedding (POST)"
        echo ""
        echo "📝 View Logs:"
        echo "  docker-compose logs -f face-embedding-api"
        echo ""
        echo -e "${GREEN}✅ Deployment Complete!${NC}"
        exit 0
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Attempt $RETRY_COUNT/$MAX_RETRIES..."
    sleep 5
done

echo -e "${RED}❌ API failed to become healthy${NC}"
echo "Showing logs:"
$DOCKER_COMPOSE logs
exit 1
