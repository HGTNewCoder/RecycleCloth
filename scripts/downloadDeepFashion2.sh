#!/bin/bash
set -e  # stop if any command fails

ZIP_DIR="/home/ubuntu/iai2o/downloadedZip"
ZIP_FILE="$ZIP_DIR/deepfashion2-original-with-dataframes.zip"
DATASET_DIR="/home/ubuntu/iai2o/Datasets"

# Create directories if they don't exist
mkdir -p "$ZIP_DIR"
mkdir -p "$DATASET_DIR"

# Download dataset
curl -L -o "$ZIP_FILE" \
  https://www.kaggle.com/api/v1/datasets/download/thusharanair/deepfashion2-original-with-dataframes

# Unzip after download
unzip -o "$ZIP_FILE" -d "$DATASET_DIR"
``
