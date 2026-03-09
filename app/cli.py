import argparse

import torch

import numpy as np
from pathlib import Path

from models import SatelliteSegModel
from utils.logging import setup_logging
from utils.infer_utils import parse_run_name
from inference import process_input
from all_years_map import aggregate_years

import logging
import sys
import re

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

setup_logging()
logger = logging.getLogger(__name__)

read_storage = Path("/workspace/segmentation/storage_read")
write_storage = Path("/workspace/segmentation/storage_write")


def setup_model():
    """
    Set up and load the segmentation model, returning the model, config, device, and model version.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_arch = "UNet"
    encoder_name = "mobilenet_v2" # "resnet18" #
    out_classes = 1

    model_ver = "20260225_235004_mmno_ch_0-1-2-3-4-5-9"
    checkpoint_dir = read_storage / f'models/{model_ver}/checkpoints'

    print(f'Model {model_ver} founded')
    cfg = parse_run_name(model_ver)
    print(model_ver, cfg)

    in_channels = cfg['num_channels']
    model = SatelliteSegModel(
        backend="smp",
        arch=model_arch,
        encoder_name=encoder_name,
        in_channels=in_channels,
        out_classes=out_classes,
        normalize=False
    )
    model = model.to(device)
    loaded = model.load_best_checkpoint(
        checkpoint_dir=checkpoint_dir,
        device=device,
        strict=True,  # False if head changed
    )

    if not loaded:
        logger.warning("Running inference with randomly initialized weights!")
    model.eval()

    return model, cfg, device, model_ver

def verify_file_channels(input_path):
    base = str(input_path)
    # Single if with 'or' for _10m_ or _20m_ pattern
    if re.search(r'_10m_', base) or re.search(r'_20m_', base):
        if re.search(r'_10m_', base):
            base_prefix = base.split('_10m_')[0]
        else:
            base_prefix = base.split('_20m_')[0]
        file_10 = Path(base_prefix + '_10m_clipped.tif')
        file_20 = Path(base_prefix + '_20m_clipped.tif')
        if file_10.exists() and file_20.exists():
            print("Type: File (both _10m_clipped and _20m_clipped exist)")
            return "File", Path(base_prefix).name
        else:
            print("Both _10m_clipped.tif and _20m_clipped.tif must exist for this file. Not found:")
            if not file_10.exists():
                print(f"- {file_10}")
            if not file_20.exists():
                print(f"- {file_20}")
            # Do not return here; continue to check for partial file and similar files/folders
            return None, None


def check_path(ds_root, user_input):
    input_path = ds_root / user_input

    # 1. Direct file or directory match
    if input_path.exists():
        if input_path.is_file():
            return verify_file_channels(input_path)

        elif input_path.is_dir():
            print("Type: Directory")
            return "Directory", input_path
        elif input_path.is_symlink():
            print("Type: Symlink")
            return "Symlink", input_path
        else:
            print("Type: Unknown")
            return None, None
    else:
        # Do not return here; continue to check for partial file and similar files/folders
        # If it's a file but not _10m_ or _20m_, check if a _10m_clipped version exists
        possible_file = input_path.parent / (input_path.stem + '_10m_clipped.tif')
        return verify_file_channels(possible_file)

def get_unique_out_path(base_path, name):
    """
    Returns a unique output path under base_path with the given name.
    If the folder already exists, appends a numeric suffix (_1, _2, ...).
    """
    out_path = base_path / name
    suffix = 1
    while out_path.exists():
        out_path = base_path / f"{name}_{suffix}"
        suffix += 1
    return out_path

def detect_dates_in_folder(folder_path):
    """
    Scans the given folder for subfolders or files containing an 8-digit date pattern (YYYYMMDD).
    Returns a dict mapping each unique date to a representative file name (using verify_file_channels logic).
    """
    date_pattern = re.compile(r"(19\d{6}|20\d{6})")
    date_to_file = {}
    for item in folder_path.iterdir():
        matches = date_pattern.findall(item.name)
        for d in matches:
            # Use verify_file_channels logic for file name extraction
            file_type, file_name = verify_file_channels(item)
            if file_type == "File" and file_name:
                date_to_file[d] = file_name
            else:
                # fallback: use Path(item).name
                date_to_file[d] = Path(item).name
    return date_to_file




if __name__ == "__main__":
    logger.info("Inference app started. Waiting for user input...")
    model, cfg, device, model_ver = setup_model()

    inp_path = read_storage / "samples"

    if sys.stdin.isatty():
        # Interactive mode
        while True:
            user_input = input("Enter the relative path to a TIFF file or folder inside storage_read (or 'exit' to quit): ").strip()
            if user_input.lower() == 'exit':
                print("Exiting inference app.")
                break

            t, input_name = check_path(inp_path, user_input)
            print(t, input_name)
            if t == "File" and input_name is not None:
                out_path = get_unique_out_path(write_storage, input_name)
                out_path.mkdir(parents=True, exist_ok=True)
                process_input(model, device, inp_path, input_name, cfg["channels"], out_path)
                continue

            elif t == "Directory" and input_name is not None:
                out_path = get_unique_out_path(write_storage, input_name.name)
                out_path.mkdir(parents=True, exist_ok=True)
                print(f"Output folder for directory: {out_path}")

                folder_path = inp_path / input_name
                date_to_file = detect_dates_in_folder(folder_path)

                dates_count = len(date_to_file)
                if dates_count == 0:
                    print("No date folders/files found in the directory.")
                elif dates_count == 1:
                    print(f"Single date detected: {list(date_to_file.keys())[0]}")
                    for k, v in date_to_file.items():
                        folder_out_path = out_path / k
                        folder_out_path.mkdir(parents=True, exist_ok=True)
                        process_input(model, device, folder_path, v, cfg["channels"], out_path = out_path / k)
                else:
                    print(f"{dates_count} Multiple dates detected: {', '.join(date_to_file.keys())}")
                    for k, v in date_to_file.items():
                        folder_out_path = out_path / k
                        folder_out_path.mkdir(parents=True, exist_ok=True)
                        process_input(model, device, folder_path, v, cfg["channels"], out_path=out_path / k)

                    aggregate_years(out_path)


            elif t is None or input_name is None:
                print("No valid file or directory found. Please try again.")
                # 3. Fallback: recursive search for similar files/folders
                matches = []
                user_input_lower = user_input.lower()
                for item in inp_path.rglob('*'):
                    if user_input_lower in item.name.lower():
                        matches.append(item)
                if matches:
                    print("Found similar files/folders:")
                    for match in matches:
                        print(f"- {match.relative_to(inp_path)}")
                    # Optionally, you can ask the user to select one
                else:
                    print("No similar files/folders found.")
