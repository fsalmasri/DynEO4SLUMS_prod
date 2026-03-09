from pathlib import Path
import numpy as np
import re
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import rasterize


def _as_float32(x):
    return x.astype(np.float32, copy=False)

def _as_uint8(x):
    return x.astype(np.uint8, copy=False)

def _parse_date_to_int(date_str: str) -> int:
    # accepts "YYYYMMDD" or "YYYY-MM-DD"
    s = date_str.replace("-", "")
    return int(s[:8])

def _stack_preds(preds):
    # shape (T,H,W)
    return np.stack([_as_uint8(p) for p in preds], axis=0)

def _stack_confs(confs):
    # shape (T,H,W)
    return np.stack([_as_float32(c) for c in confs], axis=0)

def persistence_map(preds, ignore_value=255, normalize=True):
    """
    Pixel value = how many times it was predicted as 1 across all timesteps.
    If normalize=True -> output in [0,1] as fraction of valid timesteps.
    """
    P = _stack_preds(preds)  # (T,H,W)
    valid = (P != ignore_value)
    ones = (P == 1) & valid

    count_valid = valid.sum(axis=0).astype(np.float32)
    count_ones = ones.sum(axis=0).astype(np.float32)

    if normalize:
        return np.divide(count_ones, np.maximum(count_valid, 1.0), dtype=np.float32)
    return count_ones.astype(np.float32)


def confidence_weighted_persistence_map(preds, confs, ignore_value=255):
    """
    Pixel value = sum(conf_t * [pred_t==1]) / (#valid timesteps)
    Range ~ [0,1]. Highlights stable+confident positives.
    """
    P = _stack_preds(preds)
    C = _stack_confs(confs)
    valid = (P != ignore_value)

    weighted = ((P == 1) & valid) * C  # (T,H,W)
    count_valid = valid.sum(axis=0).astype(np.float32)
    sum_weighted = weighted.sum(axis=0).astype(np.float32)

    return np.divide(sum_weighted, np.maximum(count_valid, 1.0), dtype=np.float32)


def first_appearance_map(preds, dates, confs=None, conf_thr=None, ignore_value=255, fill_value=0):
    """
    Pixel value = first date (YYYYMMDD as int) when pred becomes 1.
    If confs + conf_thr provided, require conf >= conf_thr at that timestep.
    If never appears -> fill_value (default 0).
    """
    P = _stack_preds(preds)
    T, H, W = P.shape
    date_ints = np.array([_parse_date_to_int(d) for d in dates], dtype=np.int32)

    if confs is not None and conf_thr is not None:
        C = _stack_confs(confs)
        appears = (P == 1) & (P != ignore_value) & (C >= float(conf_thr))
    else:
        appears = (P == 1) & (P != ignore_value)

    out = np.full((H, W), fill_value, dtype=np.int32)
    # find first True along time: for each pixel, locate smallest t where appears[t]=True
    any_appears = appears.any(axis=0)
    first_idx = np.argmax(appears, axis=0)  # argmax gives first index if any True, else 0
    out[any_appears] = date_ints[first_idx[any_appears]]
    return out


def last_appearance_map(preds, dates, confs=None, conf_thr=None, ignore_value=255, fill_value=0):
    """
    Pixel value = last date (YYYYMMDD as int) when pred is 1.
    Optional confidence gating like first_appearance_map.
    If never appears -> fill_value.
    """
    P = _stack_preds(preds)
    T, H, W = P.shape
    date_ints = np.array([_parse_date_to_int(d) for d in dates], dtype=np.int32)

    if confs is not None and conf_thr is not None:
        C = _stack_confs(confs)
        appears = (P == 1) & (P != ignore_value) & (C >= float(conf_thr))
    else:
        appears = (P == 1) & (P != ignore_value)

    out = np.full((H, W), fill_value, dtype=np.int32)
    any_appears = appears.any(axis=0)

    # last true index: reverse time then take first true
    appears_rev = appears[::-1]
    last_from_end = np.argmax(appears_rev, axis=0)
    last_idx = (T - 1) - last_from_end

    out[any_appears] = date_ints[last_idx[any_appears]]
    return out


def trend_map(values_over_time, dates=None, ignore_mask=None):
    """
    Pixel-wise linear trend slope over time.
    - values_over_time: list of (H,W) arrays (binary preds or continuous conf)
    - dates: optional list of date strings. If None -> use 0..T-1 as time.
    - ignore_mask: optional list of boolean masks (H,W) per time, True=ignore
    Returns slope map (float32). Positive = increasing over time.
    """
    V = np.stack([_as_float32(v) for v in values_over_time], axis=0)  # (T,H,W)
    T, H, W = V.shape

    if dates is None:
        t = np.arange(T, dtype=np.float32)
    else:
        t = np.array([_parse_date_to_int(d) for d in dates], dtype=np.float32)
        t = (t - t.min()) / max((t.max() - t.min()), 1.0)  # normalize time to ~[0,1]

    if ignore_mask is not None:
        M = np.stack(ignore_mask, axis=0).astype(bool)  # (T,H,W)
        V = np.where(M, np.nan, V)

    # slope = cov(t, V) / var(t) computed per pixel ignoring NaNs
    t2 = t.reshape(T, 1, 1)
    t_mean = np.nanmean(t2 * np.ones_like(V), axis=0)
    v_mean = np.nanmean(V, axis=0)

    cov = np.nanmean((t2 - t_mean) * (V - v_mean), axis=0)
    var = np.nanmean((t2 - t_mean) ** 2, axis=0)
    slope = cov / np.maximum(var, 1e-8)
    slope = np.nan_to_num(slope, nan=0.0).astype(np.float32)
    return slope


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


def load_date_folders(res_path: Path):
    """
    Scans all city subfolders in res_path, then all date subfolders in each city.
    Returns a sorted list of dicts for each city/date:
      {
        "city": city_name,
        "date": "YYYYMMDD",
        "dir": Path(...),
        "confidence_tif": Path(... or None),
        "presence_gpkg": Path(... or None),
        "absence_gpkg": Path(... or None),
        "metrics_csv": Path(... or None),
        "png": Path(... or None),
      }
    Ignores non-directory files and non-date folders.
    """

    records = []
    for date_path in res_path.iterdir():
        if not date_path.is_dir() or not re.fullmatch(r"\d{8}", date_path.name):
            continue
        confidence_tif = next(date_path.glob("confidence.tif"), None)
        # metrics_csv = next(date_path.glob("metrics.csv"), None)
        presence_gpkg = next(date_path.glob("*_presence.gpkg"), None)
        absence_gpkg  = next(date_path.glob("*_absence.gpkg"), None)
        # png = next(date_path.glob("cm_*.png"), None)
        records.append({
            # "city": city,
            "date": date_path.name,
            "dir": date_path,
            "confidence_tif": confidence_tif,
            "presence_gpkg": presence_gpkg,
            "absence_gpkg": absence_gpkg,
            # "metrics_csv": metrics_csv,
            # "png": png,
        })
    # Sort by city then date
    records.sort(key=lambda r: (r["date"]))

    return records


def read_confidence_and_meta(conf_tif: Path):
    with rasterio.open(conf_tif) as src:
        conf = src.read(1).astype(np.float32)
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "H": src.height,
            "W": src.width,
        }
    return conf, meta


def save_map(array, meta, output_path_base, dtype="float32", cmap="viridis"):
    """
    output_path_base: Path without extension
    Will save:
        output_path_base.tif
        output_path_base.png
    """

    # ---- Save GeoTIFF ----
    arr = array.astype(dtype)

    with rasterio.open(
        str(output_path_base) + ".tif",
        "w",
        driver="GTiff",
        height=meta["H"],
        width=meta["W"],
        count=1,
        dtype=dtype,
        crs=meta["crs"],
        transform=meta["transform"],
    ) as dst:
        dst.write(arr, 1)

    # ---- Save PNG preview ----
    plt.figure(figsize=(6, 6))
    plt.imshow(array, cmap=cmap)
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(output_path_base) + ".png", dpi=200)
    plt.close()


def save_categorical_year_map(array, output_path_base):
    """
    array contains:
      0 (never)
      YYYYMMDD values for appearance dates
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    unique_vals = np.unique(array)
    unique_vals = unique_vals[unique_vals != 0]  # exclude 0

    years = sorted(unique_vals)

    # Add 0 as first category
    categories = [0] + years

    # Create discrete colormap
    # colors = ["lightgray"] + plt.cm.tab10(np.linspace(0, 1, len(years))).tolist()
    colors = ["lightgray"] + plt.get_cmap('tab10')(np.linspace(0, 1, len(years))).tolist()
    cmap = ListedColormap(colors)

    norm = BoundaryNorm(np.arange(len(categories)+1)-0.5, len(categories))

    # Map original values to category index
    mapped = np.zeros_like(array, dtype=int)
    for i, val in enumerate(categories):
        mapped[array == val] = i

    plt.figure(figsize=(6, 6))
    im = plt.imshow(mapped, cmap=cmap, norm=norm)
    plt.axis("off")

    cbar = plt.colorbar(im, ticks=np.arange(len(categories)))
    labels = ["Never"] + [str(y)[:4] for y in years]
    cbar.ax.set_yticklabels(labels)

    plt.tight_layout()
    plt.savefig(str(output_path_base) + ".png", dpi=200)
    plt.close()

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



def aggregate_years(res_path):

    print(res_path)
    records = load_date_folders(res_path)
    print('Records loaded')


    aggregation_path = res_path / "aggregation"
    aggregation_path.mkdir(parents=True, exist_ok=True)

    preds_yearly = []
    confs_yearly = []
    dates = []
    meta0 = None
    for r in records:
        if not all([r["confidence_tif"], r["presence_gpkg"], r["absence_gpkg"]]):
            continue
        conf_map, meta = read_confidence_and_meta(r["confidence_tif"])
        if meta0 is None:
            meta0 = meta
        pred_mask = compose_abs_pres_labels(
            presence_gpkg=r["presence_gpkg"],
            absence_gpkg=r["absence_gpkg"],
            raster_crs=meta0["crs"],
            raster_bounds_geom=None,
            W=meta0["W"],
            H=meta0["H"],
            transform=meta0["transform"],
            presence_value=1,
            absence_value=0,
            fill_value=255,
        )
        preds_yearly.append(pred_mask)
        confs_yearly.append(conf_map)
        dates.append(r["date"])

        if not preds_yearly:
            print(f"No valid data founded, skipping aggregation.")
            continue


    # --- Compute all-years maps for this city ---
    pers = persistence_map(preds_yearly, normalize=True)
    pers_w = confidence_weighted_persistence_map(preds_yearly, confs_yearly)
    first = first_appearance_map(preds_yearly, dates, confs=confs_yearly, conf_thr=0.7)
    last  = last_appearance_map(preds_yearly, dates, confs=confs_yearly, conf_thr=0.7)
    trend_conf = trend_map(confs_yearly, dates=dates)

    # --- Robustness: clip and mask trend_conf ---
    std_conf_map = np.std(np.stack(confs_yearly, axis=0), axis=0)
    trend_conf = np.clip(trend_conf, -1, 1)
    trend_conf[std_conf_map < 0.01] = 0  # Set trend to zero if confidence is nearly constant

    # --- Restore trend_pred and dis calculations ---
    trend_pred = trend_map([(p == 1).astype(np.float32) for p in preds_yearly], dates=dates)
    dis = disagreement_map(preds_yearly)

    save_map(pers, meta0, aggregation_path / "persistence", dtype="float32", cmap="viridis")
    save_map(pers_w, meta0, aggregation_path / "confidence_weighted_persistence", dtype="float32", cmap="plasma")
    save_categorical_year_map(first, aggregation_path / "first_appearance")
    save_categorical_year_map(last,  aggregation_path / "last_appearance")
    save_map(trend_conf, meta0, aggregation_path / "trend_confidence", dtype="float32", cmap="coolwarm")
    save_map(trend_pred, meta0, aggregation_path / "trend_prediction", dtype="float32", cmap="coolwarm")
    save_map(dis, meta0, aggregation_path / "disagreement", dtype="float32", cmap="magma")
