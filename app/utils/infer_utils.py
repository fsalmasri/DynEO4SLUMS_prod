import re
import csv
import numpy as np
from pathlib import Path

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm

import torch

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes as rio_shapes
from rasterio.features import rasterize
from shapely.geometry import shape, box
import geopandas as gpd


def _as_uint8(x):
    return x.astype(np.uint8, copy=False)

def _stack_preds(preds):
    # shape (T,H,W)
    return np.stack([_as_uint8(p) for p in preds], axis=0)

def _stack_confs(confs):
    # shape (T,H,W)
    return np.stack([_as_float32(c) for c in confs], axis=0)

def _as_float32(x):
    return x.astype(np.float32, copy=False)


def _parse_date_to_int(date_str: str) -> int:
    # accepts "YYYYMMDD" or "YYYY-MM-DD"
    s = date_str.replace("-", "")
    return int(s[:8])


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



def save_map(
    array,
    meta,
    output_path_base,
    dtype=None,
    cmap=None,
    mode="auto",
    discrete_max_unique=50,
    nodata_value=None,
    title=None,
    legend_title=None,
    num_ticks=5,
):
    """
    Save raster as GeoTIFF and PNG with proper UTM axes, title and legend.

    Parameters
    ----------
    array : np.ndarray
        2D array to save.
    meta : dict
        Must contain: H, W, crs, transform
    output_path_base : str
        Path without extension.
    dtype : str or None
        Output dtype for GeoTIFF.
    cmap : str or matplotlib colormap
        Colormap.
    mode : {"auto","discrete","continuous","trend"}
        Visualization mode.
    nodata_value : scalar or None
        Value treated as background in PNG.
    title : str
        Figure title.
    legend_title : str
        Title for colorbar / legend.
    """

    array = np.asarray(array)
    H, W = array.shape
    transform = meta["transform"]

    if dtype is None:
        dtype = str(array.dtype)

    out_name = str(output_path_base).lower()

    # ----------------------------
    # Decide mode
    # ----------------------------
    is_integer_array = np.issubdtype(array.dtype, np.integer)

    finite_vals = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array.ravel()
    unique_vals = np.unique(finite_vals)

    if mode == "auto":
        if "trend" in out_name:
            mode = "trend"
        elif "persistence" in out_name or "first_appearance" in out_name:
            mode = "discrete"
        elif is_integer_array and len(unique_vals) <= discrete_max_unique:
            mode = "discrete"
        else:
            mode = "continuous"

    # ----------------------------
    # Choose colormap
    # ----------------------------
    if cmap is None:
        if mode == "trend":
            cmap = "RdBu_r"
        elif mode == "discrete":
            cmap = "tab20"
        else:
            cmap = "viridis"

    # ----------------------------
    # Save GeoTIFF
    # ----------------------------
    with rasterio.open(
        str(output_path_base) + ".tif",
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype=dtype,
        crs=meta["crs"],
        transform=transform,
        nodata=nodata_value,
    ) as dst:
        dst.write(array.astype(dtype), 1)

    # ----------------------------
    # Compute map extent
    # ----------------------------
    left = transform.c
    top = transform.f
    right = left + transform.a * W
    bottom = top + transform.e * H

    extent = [left, right, bottom, top]

    # ----------------------------
    # Prepare plotting array
    # ----------------------------
    plot_array = array.astype(float)

    if nodata_value is not None:
        plot_array = np.ma.masked_where(array == nodata_value, plot_array)

    # ----------------------------
    # Plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    if mode == "discrete":

        im = ax.imshow(
            plot_array,
            cmap=cmap,
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )

        ticks = unique_vals
        cbar = fig.colorbar(im, ax=ax, ticks=ticks)

        labels = []
        for v in ticks:
            if nodata_value is not None and v == nodata_value:
                labels.append("Never")
            else:
                labels.append(str(int(v)))

        cbar.ax.set_yticklabels(labels)

    elif mode == "trend":

        vmax = np.nanpercentile(np.abs(array), 99)
        if vmax == 0:
            vmax = 1e-6

        im = ax.imshow(
            plot_array,
            cmap=cmap,
            extent=extent,
            origin="upper",
            vmin=-vmax,
            vmax=vmax,
        )

        cbar = fig.colorbar(im, ax=ax)

    else:

        im = ax.imshow(
            plot_array,
            cmap=cmap,
            extent=extent,
            origin="upper",
        )

        cbar = fig.colorbar(im, ax=ax)

    # ----------------------------
    # Legend title
    # ----------------------------
    if legend_title is not None:
        cbar.set_label(legend_title)

    # ----------------------------
    # Axis formatting
    # ----------------------------
    ax.set_xlabel("UTM Easting")
    ax.set_ylabel("UTM Northing")

    ax.xaxis.set_major_locator(mticker.MaxNLocator(num_ticks))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(num_ticks))

    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # ----------------------------
    # Title
    # ----------------------------
    if title is not None:
        ax.set_title(title, fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(str(output_path_base) + ".png", dpi=200)
    plt.close()

def save_categorical_year_map(
    array,
    output_path_base,
    title=None,
    legend_title="Year"
):
    """
    Save a categorical year map (e.g. first or last appearance).

    Parameters
    ----------
    array : np.ndarray
        2D array containing:
            0 -> never
            YYYYMMDD -> appearance date
    output_path_base : str or Path
        Output path without extension.
    title : str, optional
        Figure title.
    legend_title : str
        Title for the colorbar legend.
    """

    unique_vals = np.unique(array)
    unique_vals = unique_vals[unique_vals != 0]  # exclude "never"

    years = sorted(unique_vals)

    # categories: index -> value
    categories = [0] + years

    # build colormap
    colors = ["lightgray"] + plt.get_cmap("tab10")(np.linspace(0, 1, len(years))).tolist()
    cmap = ListedColormap(colors)

    norm = BoundaryNorm(np.arange(len(categories) + 1) - 0.5, len(categories))

    # map values -> category index
    mapped = np.zeros_like(array, dtype=int)
    for i, val in enumerate(categories):
        mapped[array == val] = i

    # plotting
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(mapped, cmap=cmap, norm=norm)
    ax.axis("off")

    # title
    if title is not None:
        ax.set_title(title, fontsize=14, pad=10)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(categories)))

    labels = ["Never"] + [str(y)[:4] for y in years]
    cbar.ax.set_yticklabels(labels)

    if legend_title is not None:
        cbar.set_label(legend_title)

    fig.tight_layout()
    fig.savefig(str(output_path_base) + ".png", dpi=200)
    plt.close(fig)


def disagreement_map(preds, ignore_value=255):
    """
    Pixel value = disagreement rate across time.
    Defined as: 1 - |2p-1| where p = fraction of ones among valid timesteps.
    - p=0 or 1 -> 0 disagreement (stable)
    - p=0.5 -> 1 disagreement (max)
    Output range [0,1].
    """
    p = persistence_map(preds, ignore_value=ignore_value, normalize=True)  # fraction ones
    return (1.0 - np.abs(2.0 * p - 1.0)).astype(np.float32)

def persistence_map(preds, ignore_value=255, normalize=True):
    """
    Pixel value = how many times it was predicted as 1 across all timesteps.
    Output is integer count of positive predictions per pixel.
    """
    P = _stack_preds(preds)  # (T,H,W)
    valid = (P != ignore_value)
    ones = (P == 1) & valid

    count_ones = np.sum(ones, axis=0).astype(np.int32)
    return count_ones

def trend_map(
    values_over_time,
    dates=None,
    ignore_mask=None,
    smoothing_sigma=1.0,
    min_valid=2,
    normalize_time=True,
    slope_threshold=0.08,
    delta_threshold=0.15,
    confidence_threshold=0.40,
    clamp_range=(-1.0, 1.0),
    return_debug=False,
):
    """
    Compute a pixel-wise temporal trend map from a sequence of confidence maps.

    This function fits a linear trend independently for each pixel across time
    and returns the regression slope as a 2D map.

    Main steps
    ----------
    1. Stack the input maps into a (T, H, W) array.
    2. Optionally mask invalid pixels at each timestamp using `ignore_mask`.
    3. Optionally smooth each pixel time series along the temporal axis while
       respecting NaN / ignored values.
    4. Compute a per-pixel least-squares linear slope using only valid
       timestamps for that pixel.
    5. Remove unreliable or weak trends using:
       - minimum number of valid observations
       - minimum absolute slope
       - minimum net confidence change from first valid to last valid
       - minimum mean confidence
    6. Optionally clip extreme slope values.

    Parameters
    ----------
    values_over_time : list of ndarray
        List of 2D arrays with shape (H, W). Each array is typically a
        confidence map for one timestamp.
    dates : list, optional
        List of dates corresponding to each time step. If None, uses
        evenly spaced time values [0, 1, ..., T-1].
        Dates are parsed with `_parse_date_to_int`.
    ignore_mask : list of ndarray, optional
        List of boolean arrays with shape (H, W). True means that pixel should
        be ignored at that timestamp.
    smoothing_sigma : float or None, default=1.0
        Standard deviation of the Gaussian smoothing applied along the temporal
        axis only. Set to 0 or None to disable smoothing.
    min_valid : int, default=2
        Minimum number of valid timestamps required to compute a meaningful
        trend for a pixel.
    normalize_time : bool, default=True
        If True, normalize the time vector to [0, 1]. In that case, the slope
        is approximately interpretable as total confidence change across the
        full observed period.
        If False, slope is expressed per raw time unit returned by
        `_parse_date_to_int` (typically days).
    slope_threshold : float, default=0.05
        Absolute slope threshold below which the trend is set to 0. This helps
        remove weak noisy trends.
    delta_threshold : float, default=0.10
        Minimum absolute net confidence change required between the first valid
        and the last valid timestamp for a pixel to keep its trend.
    confidence_threshold : float or None, default=0.30
        Minimum mean confidence across valid timestamps required to keep the
        trend. Set to None to disable this filter.
    clamp_range : tuple(float, float) or None, default=(-1.0, 1.0)
        Optional range used to clip slope values after filtering. Set to None
        to disable clipping.
    return_debug : bool, default=False
        If True, return a tuple:
            (slope_map, debug_dict)

    Returns
    -------
    slope : ndarray of shape (H, W), dtype float32
        Pixel-wise trend map.
        - Positive values: confidence increases over time
        - Negative values: confidence decreases over time
        - Zero: no reliable / meaningful trend after filtering

    Debug dictionary keys (if return_debug=True)
    --------------------------------------------
    smoothed_values : ndarray, shape (T, H, W)
        Final temporal stack used for regression, after masking and smoothing.
    valid_count : ndarray, shape (H, W)
        Number of valid timestamps per pixel.
    time_vector : ndarray, shape (T,)
        Time vector used for the regression.
    variance : ndarray, shape (H, W)
        Per-pixel temporal variance of the time variable over valid timestamps.
    delta : ndarray, shape (H, W)
        Net confidence change from first valid to last valid timestamp.
    mean_confidence : ndarray, shape (H, W)
        Mean confidence over valid timestamps.

    Notes
    -----
    - Persistence and trend are different quantities:
      persistence measures how often a pixel is positive,
      trend measures whether confidence is increasing or decreasing over time.
    - This function is intended for continuous confidence maps, not only binary
      masks.
    - Filtering weak slopes and weak net changes is recommended to suppress
      confidence jitter and produce a cleaner trend map.
    """
    if len(values_over_time) < 2:
        raise ValueError("Need at least 2 time steps.")

    V_list = [_as_float32(v) for v in values_over_time]
    shapes = [v.shape for v in V_list]
    if len(set(shapes)) != 1:
        raise ValueError(f"All maps must have the same shape, got: {shapes}")

    V = np.stack(V_list, axis=0)  # (T, H, W)
    T, H, W = V.shape

    # ------------------------------------------------------------------
    # Build time vector
    # ------------------------------------------------------------------
    if dates is None:
        t = np.arange(T, dtype=np.float32)
    else:
        if len(dates) != T:
            raise ValueError(f"len(dates)={len(dates)} must match T={T}")
        t = np.array([_parse_date_to_int(d) for d in dates], dtype=np.float32)

    if normalize_time:
        dt = float(t.max() - t.min())
        if dt > 0:
            t = (t - t.min()) / dt
        else:
            t = np.zeros_like(t, dtype=np.float32)

    # ------------------------------------------------------------------
    # Apply ignore mask
    # ------------------------------------------------------------------
    if ignore_mask is not None:
        if len(ignore_mask) != T:
            raise ValueError(f"len(ignore_mask)={len(ignore_mask)} must match T={T}")
        M = np.stack([np.asarray(m, dtype=bool) for m in ignore_mask], axis=0)
        if M.shape != V.shape:
            raise ValueError(f"ignore_mask stacked shape {M.shape} must match V shape {V.shape}")
        V = np.where(M, np.nan, V)

    # ------------------------------------------------------------------
    # Optional temporal smoothing while respecting NaNs
    # ------------------------------------------------------------------
    if smoothing_sigma is not None and smoothing_sigma > 0:
        if gaussian_filter1d is None:
            raise ImportError(
                "scipy is required for temporal smoothing. "
                "Install scipy or set smoothing_sigma=0."
            )

        valid = np.isfinite(V).astype(np.float32)
        V_filled = np.nan_to_num(V, nan=0.0)

        # Smooth numerator and denominator separately
        V_num = gaussian_filter1d(V_filled, sigma=smoothing_sigma, axis=0, mode="nearest")
        V_den = gaussian_filter1d(valid, sigma=smoothing_sigma, axis=0, mode="nearest")

        with np.errstate(invalid="ignore", divide="ignore"):
            V = V_num / np.maximum(V_den, 1e-8)

        # Restore unsupported locations as NaN
        V[V_den <= 1e-6] = np.nan

    # ------------------------------------------------------------------
    # Per-pixel regression using only valid timestamps for each pixel
    # ------------------------------------------------------------------
    t2 = t[:, None, None]          # (T, 1, 1)
    valid = np.isfinite(V)         # (T, H, W)
    n = valid.sum(axis=0).astype(np.float32)

    safe_n = np.maximum(n, 1.0)

    t_sum = np.sum(np.where(valid, t2, 0.0), axis=0)
    v_sum = np.sum(np.where(valid, V, 0.0), axis=0)

    t_mean = t_sum / safe_n
    v_mean = v_sum / safe_n

    cov = np.sum(np.where(valid, (t2 - t_mean) * (V - v_mean), 0.0), axis=0) / safe_n
    var = np.sum(np.where(valid, (t2 - t_mean) ** 2, 0.0), axis=0) / safe_n

    with np.errstate(invalid="ignore", divide="ignore"):
        slope = cov / np.maximum(var, 1e-8)

    # ------------------------------------------------------------------
    # Compute first-valid / last-valid net change per pixel
    # ------------------------------------------------------------------
    has_valid = valid.any(axis=0)

    first_idx = np.argmax(valid, axis=0)                    # first True along time
    last_idx = T - 1 - np.argmax(valid[::-1], axis=0)       # last True along time

    first_vals = np.take_along_axis(V, first_idx[None, :, :], axis=0)[0]
    last_vals = np.take_along_axis(V, last_idx[None, :, :], axis=0)[0]

    delta = last_vals - first_vals
    delta = np.where(has_valid, delta, np.nan)

    # Mean confidence over valid timestamps
    mean_conf = np.nanmean(V, axis=0)

    # ------------------------------------------------------------------
    # Filtering and cleanup
    # ------------------------------------------------------------------
    slope = slope.astype(np.float32)

    # Remove unreliable pixels
    slope[n < min_valid] = 0.0

    # Remove numerical issues
    slope = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove weak noisy trends
    if slope_threshold is not None and slope_threshold > 0:
        slope[np.abs(slope) < slope_threshold] = 0.0

    # Remove weak net confidence change
    if delta_threshold is not None and delta_threshold > 0:
        slope[np.abs(np.nan_to_num(delta, nan=0.0)) < delta_threshold] = 0.0

    # Remove trends in very low-confidence regions
    if confidence_threshold is not None:
        slope[np.nan_to_num(mean_conf, nan=0.0) < confidence_threshold] = 0.0

    # Optional clamp
    if clamp_range is not None:
        slope = np.clip(slope, clamp_range[0], clamp_range[1])

    slope = slope.astype(np.float32)

    if return_debug:
        return slope, {
            "smoothed_values": V,
            "valid_count": n,
            "time_vector": t,
            "variance": var.astype(np.float32),
            "delta": delta.astype(np.float32),
            "mean_confidence": mean_conf.astype(np.float32),
        }

    return slope