import matplotlib
matplotlib.use("Agg")

import rasterio
import torch

from utils.infer_utils import (
    load_s2_rgb_and_labels,
    tile_image_and_label,
    reconstruct_from_tiles,
    save_pred_presence_absence_gpkgs,
    save_confidence_geotiff
)


def load_single_file(inp_path, tiff, band_ids):
    """
    Loads a single Sentinel-2 file, extracts image bands, labels, and metadata, and tiles the image for inference.

    Args:
        inp_path (Path): Path to the directory containing the TIFF files.
        tiff (str): Base TIFF filename (without band suffix).
        band_ids (list of int): List of band indices to load from the image.

    Returns:
        image (torch.Tensor): The loaded image tensor of shape (C, H, W).
        tiles (list of dict): List of tiles, each containing 'image', 'label', 'y', 'x'.
        meta (dict): Metadata dictionary with keys 'crs', 'transform', 'bounds', 'H', 'W'.

    Prints:
        - Image shape
        - Number of tiles
        - Status message
    """
    image, labels, meta = load_s2_rgb_and_labels(
    inp_path,
    tiff = tiff,
    contrast_scale=2.5,
    band_ids = band_ids
    )
    print(f"Image shape: {image.shape}")
    tiles = tile_image_and_label(image, labels, tile_size=64)
    print(f"Number of tiles: {len(tiles)}")

    return image, tiles, meta


def process_input(model, device, inp_path, tiff, band_ids, out_path):
    image, tiles, meta = load_single_file(inp_path, tiff, band_ids)
    print("File loaded successfully. Starting inference...")

    with torch.inference_mode():
        for tile in tiles:
            x = tile["image"].unsqueeze(0).to(device)
            conf = torch.sigmoid(model(x))
            pred = (conf > 0.5).float()
            tile["pred"] = pred[0, 0].cpu()
            tile["conf"] = conf[0, 0].cpu()

    C, H, W = image.shape
    print("Reconstruct the tiles...")
    pred_map = reconstruct_from_tiles(tiles, H, W, 'pred')
    conf_map = reconstruct_from_tiles(tiles, H, W, 'conf')
    pred_vis = pred_map.numpy().copy().astype("uint8")

    presence_fname = out_path / f"{tiff}_presence.gpkg"
    absence_fname = out_path / f"{tiff}_absence.gpkg"
    save_pred_presence_absence_gpkgs(pred_vis, meta, presence_fname, absence_fname)
    save_confidence_geotiff(conf_map, meta, out_path / "confidence.tif")



