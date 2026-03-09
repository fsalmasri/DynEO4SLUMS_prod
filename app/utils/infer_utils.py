import re
import csv
import numpy as np
from pathlib import Path

import torch

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes as rio_shapes
from rasterio.features import rasterize
from shapely.geometry import shape, box
import geopandas as gpd


def confusion_map(pred, label, ignore_value=255):
    """
    pred: (H,W) binary {0,1}
    label: (H,W) {0,1,255}
    """

    cm = np.full(label.shape, 255, dtype=np.uint8)

    valid = label != ignore_value

    # p = pred.astype(bool)
    # g = label.astype(bool)
    #
    # cm[(p == 0) & (g == 0) & valid] = 0  # TN
    # cm[(p == 1) & (g == 1) & valid] = 1  # TP
    # cm[(p == 1) & (g == 0) & valid] = 2  # FP
    # cm[(p == 0) & (g == 1) & valid] = 3  # FN

    cm[(pred == 0) & (label == 0) & valid] = 0  # TN
    cm[(pred == 1) & (label == 1) & valid] = 1  # TP
    cm[(pred == 1) & (label == 0) & valid] = 2  # FP
    cm[(pred == 0) & (label == 1) & valid] = 3  # FN

    return cm

def load_bands(TEN_M, TWENTY_M):



    # --- load the 10 m file (reference grid)
    with rasterio.open(TEN_M) as s10:
        ten = s10.read(out_dtype="float32")           # (4, H, W) -> B4,B3,B2,B8 ~  R, G, B, NIR
        H, W = s10.height, s10.width
        transform, crs = s10.transform, s10.crs
        print("10 m:", s10.count, "bands | res:", s10.res, "| size:", (H, W))

        rbounds = box(*s10.bounds)

        # names = list(s10.descriptions) if s10.descriptions else [None]*s10.count
        # print("S10 band names:")
        # for n in names:
        #     print(n)


    # --- upsample the 20 m file to match the 10 m grid
    with rasterio.open(TWENTY_M) as s20:
        twenty = s20.read(out_dtype="float32")
        # Try to get band names from descriptions; fallback to per-band tags
        # names = list(s20.descriptions)
        # print("S20 band names:")
        # for n in names:
        #     print(n)
        print("S2 Band count:", s20.count)

        # B5, B6, B7, B8A, B11, B12, AOT, CLD, SCL, SNW, WVP
        spectral_idxs = list(range(1, 7))  # 1..6 → B5,B6,B7,B8A,B11,B12 (VRED1, VRED2, VRED3, VRED4, SWIR1, SWIR2)
        up = np.zeros((len(spectral_idxs), H, W), dtype=np.float32)

        for k, band_idx in enumerate(spectral_idxs):
            reproject(
                source=rasterio.band(s20, band_idx),
                destination=up[k],
                src_transform=s20.transform, src_crs=s20.crs,
                dst_transform=transform,     dst_crs=crs,
                resampling=Resampling.bilinear,
            )
        print("20 m upsampled shape:", up.shape)  # expect (6,H,W)

    # --- (optional) build the full stack now
    stack = np.concatenate([ten, up], axis=0).astype("float32")  # (10,H,W)

    return stack, transform, crs, rbounds



def load_s2_rgb_and_labels(
        path_tif: Path,
        tiff: str,
        presence_gpkg: Path= None,
        absence_gpkg: Path = None,
        reflectance_scale=10000.0,
        contrast_scale=None,
        band_ids=None,  # <-- list of band indices, e.g. [0,1,2,3,4,6,8,9]
):
    """
    Loads Sentinel-2 RGB (B4,B3,B2) as float32 tensor (C,H,W),
    plus label mask aligned to the raster grid.

    Returns:
      rgb_t (torch.FloatTensor): (3,H,W) in [0,1]
      labels (np.uint8): (H,W) values {0,1}
      meta (dict): crs, transform, bounds, H, W
    """

    tiff_10 = path_tif / f"{tiff}_10m_clipped.tif"
    tiff_20 = path_tif / f"{tiff}_20m_clipped.tif"

    x, transform, crs, rbounds = load_bands(tiff_10, tiff_20)

    if band_ids is not None:
        # robust checks
        band_ids = list(map(int, band_ids))
        C = x.shape[0]
        bad = [b for b in band_ids if b < 0 or b >= C]
        if bad:
            raise ValueError(f"band_ids out of range (0..{C - 1}): {bad}")
        x = x[band_ids, :, :]


    H, W = x.shape[1], x.shape[2]

    # Scale reflectance to ~[0,1]
    x = x.astype(np.float32) / float(reflectance_scale)
    x = np.clip(x, 0.0, 1.0)

    # contrast stretch
    if contrast_scale is not None:
        x = np.clip(x * float(contrast_scale), 0.0, 1.0)

    x_t = torch.from_numpy(x)

    labels = None
    if presence_gpkg is not None:
        labels = compose_abs_pres_labels(
            presence_gpkg=presence_gpkg,
            absence_gpkg=absence_gpkg,
            raster_crs=crs,
            raster_bounds_geom=rbounds,
            W=W,
            H=H,
            transform=transform,
        )

    meta = {"crs": crs, "transform": transform, "bounds": rbounds, "H": H, "W": W}


    return x_t, labels, meta


def compose_abs_pres_labels(
    presence_gpkg: Path,
    absence_gpkg: Path,
    raster_crs,
    raster_bounds_geom,   # shapely box(*ref.bounds)
    W: int,
    H: int,
    transform,
    presence_value: int = 1,
    absence_value: int = 0,
    fill_value: int = 255,
) -> np.ndarray:
    """
    Create a label mask aligned to a reference raster grid.

    Returns:
        mask (H,W) uint8 with values {0,1}
    """

    # --- read
    pres = gpd.read_file(presence_gpkg)
    absn = gpd.read_file(absence_gpkg)

    if pres.crs is None or absn.crs is None:
        raise ValueError("Presence/absence GPKG must have a CRS defined.")

    # --- reproject to raster CRS
    pres = pres.to_crs(raster_crs)
    absn = absn.to_crs(raster_crs)

    # --- clip to raster bounds (keeps only relevant geometries)
    # geopandas.clip expects GeoDataFrame + geometry
    # Only clip if bounds geom is provided
    if raster_bounds_geom is not None:
        pres = gpd.clip(pres, raster_bounds_geom)
        absn = gpd.clip(absn, raster_bounds_geom)

    # --- prepare shapes
    pres_geoms = [g for g in pres.geometry if g is not None and not g.is_empty]
    absn_geoms = [g for g in absn.geometry if g is not None and not g.is_empty]

    # --- rasterize
    # Important: rasterize writes in order; put absence first, then presence
    # so presence wins on overlaps.
    shapes = [(g, absence_value) for g in absn_geoms] + [(g, presence_value) for g in pres_geoms]

    mask = rasterize(
        shapes=shapes,
        out_shape=(H, W),
        transform=transform,
        fill=fill_value,
        dtype="uint8",
        all_touched=False,   # keep strict boundaries; set True if you want more coverage
    )

    return mask


def tile_image_and_label(image, label=None, tile_size=64):
    """
    Tiles an image (and optionally a label mask) into non-overlapping patches.

    Args:
        image (np.ndarray): Shape (C, H, W), required.
        label (np.ndarray or None): Shape (H, W), optional. If None, only image tiles are created.
        tile_size (int): Size of each tile (default: 64).

    Returns:
        tiles (list of dict): Each dict contains:
            - 'image': (C, tile_size, tile_size) ndarray
            - 'label': (tile_size, tile_size) ndarray or None
            - 'y': int, top coordinate
            - 'x': int, left coordinate
        Only full tiles are included (partial tiles at edges are skipped).
    """
    C, H, W = image.shape
    tiles = []

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            img_tile = image[:, y:y+tile_size, x:x+tile_size]
            if label is not None:
                lbl_tile = label[y:y+tile_size, x:x+tile_size]
            else:
                lbl_tile = None

            # skip partial tiles
            if img_tile.shape[1] != tile_size or img_tile.shape[2] != tile_size:
                continue

            tiles.append({
                "image": img_tile,
                "label": lbl_tile,
                "y": y,
                "x": x,
            })

    return tiles


def reconstruct_from_tiles(tiles, H, W, split='pred'):
    full = torch.zeros((H, W), dtype=torch.float32)

    for t in tiles:
        y = t["y"]
        x = t["x"]

        if split == 'pred':
            data = t.get("pred", None)
        elif split == 'conf':
            data = t.get("conf", None)
        else:
            data = None
        if data is None:
            continue
        h, w = data.shape
        full[y:y + h, x:x + w] = data

    return full

def parse_run_name(name: str):
    # extract date and time
    date = name[:8]
    time = name[9:15]

    # extract channel list
    m = re.search(r"ch_(.+)", name)
    if not m:
        raise ValueError(f"No channel info found in: {name}")

    ch_str = m.group(1)
    channels = [int(c) for c in ch_str.split("-")]

    return {
        "date": date,
        "time": time,
        "channels": channels,
        "num_channels": len(channels),
    }


def write_values_csv(fname, pckg, tiff, date, TP, TN, FP, FN):
    pixel_acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    fg_acc = TP / (TP + FN + 1e-8)
    bg_acc = TN / (TN + FP + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)

    # ---- Clean Console Print ----
    print("=" * 50)
    print(f"Package : {pckg}")
    print(f"Date    : {date}")
    print(f"Tiff    : {tiff}")
    print("-" * 50)
    print(f"Pixel Acc      : {pixel_acc:.4f}")
    print(f"FG Recall      : {fg_acc:.4f}")
    print(f"BG Accuracy    : {bg_acc:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Dice / F1      : {dice:.4f}")
    print(f"IoU            : {iou:.4f}")
    print("=" * 50)
    print()

    file_exists = Path(fname).exists()

    with open(fname, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "package", "date", "tiff",
                "TP", "TN", "FP", "FN",
                "pixel_acc", "fg_recall", "bg_acc",
                "precision", "dice", "iou"
            ])

        writer.writerow([
            pckg, date, tiff,
            TP, TN, FP, FN,
            pixel_acc, fg_acc, bg_acc,
            precision, dice, iou
        ])

def save_pred_presence_absence_gpkgs(
    pred_np: np.ndarray,
    meta: dict,
    presence_out,
    absence_out,
):
    """
    pred_np: (H,W) uint8 with values {0,1,255}
    meta: {"crs": crs, "transform": transform, ...}
    Writes two GPKGs similar to your original presence/absence files.
    """


    pred_np = pred_np.astype("uint8")

    pres_records = []
    abs_records = []

    for geom, value in rio_shapes(pred_np, transform=meta["transform"]):
        value = int(value)
        if value == 255:
            continue
        g = shape(geom)
        if g.is_empty:
            continue
        if value == 1:
            pres_records.append({"geometry": g})
        elif value == 0:
            abs_records.append({"geometry": g})

    pres_gdf = gpd.GeoDataFrame(pres_records, crs=meta["crs"])
    abs_gdf  = gpd.GeoDataFrame(abs_records,  crs=meta["crs"])

    pres_gdf.to_file(presence_out, driver="GPKG")
    abs_gdf.to_file(absence_out, driver="GPKG")


def save_confidence_geotiff(conf_map, meta, output_path):
    if conf_map.ndim == 3:
        conf_map = conf_map[0]

    conf_np = conf_map.detach().cpu().numpy().astype("float32")

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=meta["H"],
        width=meta["W"],
        count=1,
        dtype="float32",
        crs=meta["crs"],
        transform=meta["transform"],
    ) as dst:
        dst.write(conf_np, 1)
