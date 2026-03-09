# DynEO Segmentation Inference Docker Guide

This guide explains how to build, run, and use the segmentation inference app in Docker.

## Prerequisites
- Docker and Docker Compose installed
- NVIDIA GPU and drivers (for GPU support)
- Project structure as shown in this repository

## Folder Structure
- `docker/` — Docker files and instructions
- `storage_read/` — Input data (read-only)
- `storage_write/` — Output/results (read-write)
- `hf_cache/` — HuggingFace cache (optional)
- `app/` — Main inference code

## Build the Docker Image

From the `docker/` directory:

```bash
docker compose build
```

## Run the Inference App

Start the container interactively:

```bash
docker compose run --rm --build inference
```

This will launch the app in `/workspace/segmentation`.

## Using the CLI

Once inside the container, you can use the CLI to run inference:

1. Place your input files in `storage_read/samples/`.
2. At the prompt, enter the relative path to a TIFF file or folder (e.g. `S2A_MSIL2A_20161104T054842_10m_clipped.tif` or the prefix `S2A_MSIL2A_20161104T054842` or a folder name).
3. Results will be written to `storage_write/`.

## GPU Support

The container is configured for NVIDIA GPU. Make sure you have the NVIDIA Container Toolkit installed.

## Troubleshooting
- Ensure input files are named correctly and both `_10m_clipped.tif` and `_20m_clipped.tif` exist for each sample.
- Check logs for errors.
- For HuggingFace models, cache is mounted at `/data/hf`.

## Example Workflow
1. Place your Sentinel-2 files in `storage_read/samples/`.
2. Run `docker compose run --rm inference`.
3. At the prompt, enter a file or folder name.
4. Check `storage_write/` for results.

## Stopping the App

Press `Ctrl+C` or run:

```bash
docker compose down
```

---
For advanced usage, see the comments in `docker-compose.yml` and the app code.

