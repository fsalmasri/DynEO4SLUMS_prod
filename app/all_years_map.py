from pathlib import Path
import numpy as np
import re
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import rasterize

from utils.infer_utils import save_map, save_categorical_year_map, persistence_map, disagreement_map
from utils.infer_utils import trend_map


from utils.infer_utils import _stack_preds, _stack_confs, _parse_date_to_int



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


def claculate_and_save(out_path, dates, preds_yearly, confs_yearly, meta0):
    # Filter dates to only include valid date strings (YYYYMMDD)
    filtered_dates = [d for d in dates if re.fullmatch(r"\d{8}", d)]
    # Filter preds and confs to match filtered_dates
    filtered_indices = [i for i, d in enumerate(dates) if d in filtered_dates]
    filtered_preds = [preds_yearly[i] for i in filtered_indices]
    filtered_confs = [confs_yearly[i] for i in filtered_indices]

    pers = persistence_map(filtered_preds, normalize=True)
    pers_w = confidence_weighted_persistence_map(filtered_preds, filtered_confs)
    first = first_appearance_map(filtered_preds, filtered_dates, confs=filtered_confs, conf_thr=0.7)
    last = last_appearance_map(filtered_preds, filtered_dates, confs=filtered_confs, conf_thr=0.7)
    trend_conf = trend_map(filtered_confs, dates=filtered_dates)

    # --- Restore trend_pred and dis calculations ---
    trend_pred = trend_map(
        [(p == 1).astype(np.float32) for p in filtered_preds],
        dates=filtered_dates,
        smoothing_sigma=0.0,
        slope_threshold=0.15,
        delta_threshold=0.5,
        confidence_threshold=None,
    )

    dis = disagreement_map(filtered_preds)

    save_map(
        pers,
        meta0,
        out_path / "persistence",
        dtype="int32",
        mode="discrete",
        title="Prediction Persistence",
        legend_title="Number of detections"
    )

    save_map(
        pers_w,
        meta0,
        out_path / "confidence_weighted_persistence",
        dtype="float32",
        mode="continuous",
        cmap="plasma",
        title="Confidence-Weighted Persistence",
        legend_title="Weighted persistence"
    )

    save_categorical_year_map(
        first,
        out_path / "first_appearance",
        title="First Appearance of Prediction",
        legend_title="Year"
    )

    save_categorical_year_map(
        last,
        out_path / "last_appearance",
        title="Last Appearance of Prediction",
        legend_title="Year"
    )

    save_map(
        trend_conf,
        meta0,
        out_path / "trend_confidence",
        dtype="float32",
        mode="trend",
        cmap="coolwarm",
        title="Confidence Trend",
        legend_title="Trend slope"
    )

    save_map(
        trend_pred,
        meta0,
        out_path / "trend_prediction",
        dtype="float32",
        mode="trend",
        cmap="coolwarm",
        title="Prediction Trend",
        legend_title="Trend slope"
    )

    save_map(
        dis,
        meta0,
        out_path / "disagreement",
        dtype="float32",
        mode="discrete",
        cmap="magma",
        title="Prediction Disagreement",
        legend_title="Disagreement score"
    )

def aggregate_years(res_path):

    print(f"Aggregating years started...")
    records = load_date_folders(res_path)
    print(f'Records loaded with {len(records)}')

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

    claculate_and_save(aggregation_path, dates, preds_yearly, confs_yearly, meta0)

    # Generate pairs of consecutive filtered dates (YYYYMMDD)
    date_pairs = [(dates[i], dates[i+1]) for i in range(len(dates)-1)]
    print(f"{len(date_pairs)} Consecutive date pairs for city: {date_pairs}")
    for date_pair in date_pairs:
        couple_dir = aggregation_path / f"{date_pair[0]}_{date_pair[1]}"
        couple_dir.mkdir(parents=True, exist_ok=True)

        claculate_and_save(couple_dir, date_pair, preds_yearly, confs_yearly, meta0)