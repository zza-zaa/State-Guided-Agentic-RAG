# Offline Deployment Guide

## Goal
Prepare everything on WSL/Ubuntu with internet, then upload to the offline server.

## Step 1
Create the project folder on Windows:

`C:\Users\12037\Desktop\project reproduce\csa_rag_aaai2027`

## Step 2
Open WSL and enter the mirrored directory:

```bash
cd /mnt/c/Users/12037/Desktop/project\ reproduce/csa_rag_aaai2027
```

## Step 3
Create the conda environment and download Python wheels:

```bash
conda env create -f environment.yml
conda activate csa-rag
bash scripts/build_wheelhouse.sh
```

## Step 4
Login to Hugging Face once if needed and download the selected models:

```bash
huggingface-cli login
python scripts/download_hf_models.py --profile default
```

## Step 5
Create the upload bundle:

```bash
bash scripts/package_for_server.sh
```

## Step 6
Upload `bundles/csa_rag_bundle.tar.gz` to the offline server.

## Step 7
On the server:

```bash
tar -xzf csa_rag_bundle.tar.gz
cd csa_rag_bundle
bash scripts/server_install_from_bundle.sh
```
