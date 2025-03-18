#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting training pipeline..."

# Function to check if previous command was successful
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 completed successfully${NC}"
    else
        echo -e "${RED}✗ $1 failed${NC}"
        exit 1
    fi
}

# Execute scaler
echo "1. Running data scaling..."
python scaler.py
check_status "Data scaling"

# Execute MLP
echo "2. Running MLP training..."
python mlp.py
check_status "MLP training"

# Execute KMeans
echo "3. Running KMeans clustering..."
python kmeans.py
check_status "KMeans clustering"

echo -e "${GREEN}✓ Training pipeline completed successfully!${NC}" 