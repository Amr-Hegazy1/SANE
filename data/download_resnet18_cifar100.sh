#!/bin/bash
#SBATCH -w gpu-intel-pvc         # Specific node
#SBATCH -t 20:00:00                 # Time limit (max 24:00:00)
#SBATCH -o output_%j.txt            # Standard output (%j = job ID)
#SBATCH -e error_%j.txt             # Standard error

cd /storage/home/amr.hegazy/nobackup/SANE/data

# Define the URL and target directory
URL="https://zenodo.org/records/6977382/files/cifar100_resnet18_epoch60.zip"

TARGET_DIR="."

# Create the target directory if it doesn't exist
mkdir -p ${TARGET_DIR}

# Define the output file path
OUTPUT_FILE="${TARGET_DIR}/cifar100_resnet18_epoch60.zip"

# Download the zip file
curl -L ${URL} -o ${OUTPUT_FILE}

# Unzip the downloaded file
unzip ${OUTPUT_FILE} -d ${TARGET_DIR}

# Optionally, remove the zip file after extraction
rm ${OUTPUT_FILE}

echo "Download and extraction complete."
