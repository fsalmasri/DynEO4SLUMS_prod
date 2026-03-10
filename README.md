# DynEO Segmentation Inference Docker Guide

DynEO Segmentation Inference is a Python-based application for running semantic segmentation on Sentinel-2 satellite imagery. It leverages deep learning models to detect and analyze changes in land cover, supporting multi-temporal aggregation and trend analysis. The app is designed to run efficiently in a Docker container with GPU acceleration.


## Features

- **Satellite Segmentation**: Uses a UNet architecture with MobileNetV2 encoder for pixel-wise classification.
- **Multi-band Input**: Processes both 10m and 20m resolution bands from Sentinel-2.
- **Temporal Aggregation**: Aggregates predictions across multiple dates, computes persistence, trend, and disagreement maps.
- **Confidence Analysis**: Outputs confidence-weighted maps and appearance statistics.
- **Geospatial Output**: Results are saved as GeoTIFFs, PNGs, and GeoPackages (GPKG) for easy GIS integration.
- **Interactive CLI**: Allows users to select files or folders for inference and aggregation.

## Project Structure

- `app/` — Main Python code for CLI, inference, aggregation, and utilities.
- `storage_read/` — Input directory (read-only), expected to contain Sentinel-2 TIFFs and GPKG files.
- `storage_write/` — Output directory (read-write), where results are saved.
- `hf_cache/` — HuggingFace cache for model downloads (optional).
- `Dockerfile` — Multi-stage build for efficient containerization.
- `docker-compose.yml` — Service definition for running the app with GPU support.

## Input Requirements

- Sentinel-2 imagery split into `_10m_clipped.tif` and `_20m_clipped.tif` files.
- Presence and absence GeoPackage files for each date (optional for aggregation).
- Files should be placed in `storage_read/samples/`.


## Installation

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU and drivers (for CUDA support)

### Build the Docker Image

From the project root:

```bash
docker compose build
```

### Run the Inference App

Start the container interactively:

```bash
docker compose run --rm --build inference
```

## Usage

### CLI Workflow

- Place your input files in `storage_read/samples/`.
- At the prompt, enter the relative path to a TIFF file, prefix, or folder name.
- The app will check for required files, run inference, and save results to `storage_write/`.
- For folders, the app aggregates results across dates and generates summary maps.


### Output

- **GeoTIFFs**: Georeferenced raster outputs including segmentation masks, confidence maps, persistence maps, confidence-weighted persistence, first appearance, last appearance, confidence trend, prediction trend, and disagreement maps.

- **PNGs**: Visualizations of the generated rasters with UTM coordinate axes, titles, and legends for quick inspection.

- **GeoPackages (GPKG)**: Vectorized presence and absence polygons derived from the segmentation results, suitable for use in GIS tools.
- 
### Technical Details

- **Model**: UNet with MobileNetV2 encoder, loaded from checkpoint.
- **Frameworks**: PyTorch, rasterio, geopandas, matplotlib.
- **Aggregation**: Computes persistence, confidence-weighted persistence, first/last appearance, trend, and disagreement maps.
- **Docker**: Multi-stage build for reproducibility and performance. GPU support via NVIDIA Container Toolkit.



### Troubleshooting

- Ensure both `_10m_clipped.tif` and `_20m_clipped.tif` exist for each sample.
- Check logs for missing files or errors.
- GPU must be available and accessible to the container.

### License

This project is intended for research and internal use. See `LICENSE` for details.


## Example Workflow
1. Place your Sentinel-2 files in `storage_read/samples/`.
2. Run `docker compose run --rm inference`.
3. At the prompt, enter a file or folder name.
4. Check `storage_write/` for results.

## Stopping the App

Press `Ctrl+C`, type `exist` or run:

```bash
docker compose down
```



