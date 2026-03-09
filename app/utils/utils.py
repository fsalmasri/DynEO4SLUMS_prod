
import os
import typing
import datetime
import numpy as np
import matplotlib.pyplot as plt
import logging


from segmentation_alternative_pipeline.dataset_raw import Dataset, get_train_transform
from torch.utils.data import DataLoader

# ----------------------------
# UTILS
# ----------------------------

# Create a directory to store the output masks
def make_output_dirs(root: str, data_type: str = "default", experiment: typing.Optional[str] = None, timestamp: typing.Optional[str] = None):
    ts = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}"
    if experiment:
        run_name = f"{ts}_{experiment}"

    base = os.path.join(root, "runs", data_type, run_name)
    subdirs = {
        "base": base,
        "checkpoints": os.path.join(base, "checkpoints"),
        "plots": os.path.join(base, "plots"),
        "train_outputs": os.path.join(base, "train_outputs"),
        "train_samples": os.path.join(base, "train_samples"),
        "val_outputs": os.path.join(base, "val_outputs"),
        "val_samples": os.path.join(base, "val_samples"),
        "test_outputs": os.path.join(base, "test_outputs"),
        "test_samples": os.path.join(base, "test_samples"),
    }
    for p in subdirs.values():
        os.makedirs(p, exist_ok=True)
    return subdirs


def visualize(output_dir, image_filename, data_type="0_1_255", **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())

        img = image
        # tensor -> numpy
        try:
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
        except Exception:
            pass

        img = np.array(img)

        # Check dim == 3 and shape like (H, W, 3)
        if img.ndim == 3 and img.shape[2] == 3:
            # This is the image format
            plt.imshow(img)
        elif img.ndim == 2:
            # This is the mask format
            if data_type == "0_1":
                plt.imshow(img, cmap='gray')
            else:
                h, w = img.shape
                col = np.zeros((h, w, 3), dtype=np.float32)
                col[img == 0] = np.array([0.0, 0.0, 1.0])
                col[img == 1] = np.array([1.0, 0.0, 0.0])
                col[img == 255] = np.array([0.5, 0.5, 0.5])
                plt.imshow(col)
        else:
            raise ValueError(f"Unsupported image shape for visualization: {img.shape}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, image_filename))
    plt.close()


def binary_stats_ignore_index(pred, target, ignore_index=255):
    """
    pred: LongTensor [B,H,W] ∈ {0,1}
    target: LongTensor [B,H,W] ∈ {0,1,255}
    """
    valid = target != ignore_index

    pred = pred[valid]
    target = target[valid]

    if pred.numel() == 0:
        return 0, 0, 0, 0

    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()

    return tp, fp, fn, tn


# Save training and validation metrics plots
def save_metrics_combined(plots_dir, history, data_type):
    os.makedirs(plots_dir, exist_ok=True)
    # determine epoch range using available histories
    n_epochs = max(
        len(history.get("train_losses", [])),
        len(history.get("train_iou", [])),
        len(history.get("train_dice", [])),
        len(history.get("train_precision", [])),
        len(history.get("train_recall", [])),
    )
    epochs = list(range(1, n_epochs + 1))

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    ax = axes.ravel()

    # Loss
    ax[0].plot(history.get("train_losses", []), label="Train Loss")
    ax[0].plot(history.get("val_losses", []), label="Val Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title(f"Loss ({data_type})")
    ax[0].legend()
    ax[0].grid(True)

    # IoU
    ax[1].plot(epochs, history.get("train_iou", []), marker="o", label="Train IoU")
    ax[1].plot(epochs, history.get("val_iou", []), marker="o", label="Val IoU")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("IoU")
    ax[1].set_title("IoU (@0.5)")
    ax[1].legend()
    ax[1].grid(True)

    # Dice
    ax[2].plot(epochs, history.get("train_dice", []), marker="o", label="Train Dice")
    ax[2].plot(epochs, history.get("val_dice", []), marker="o", label="Val Dice")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Dice")
    ax[2].set_title("Dice (@0.5)")
    ax[2].legend()
    ax[2].grid(True)

    # Recall
    ax[3].plot(epochs, history.get("train_recall", []), marker="o", label="Train Recall")
    ax[3].plot(epochs, history.get("val_recall", []), marker="o", label="Val Recall")
    ax[3].set_xlabel("Epoch")
    ax[3].set_ylabel("Recall")
    ax[3].set_title("Recall (@0.5)")
    ax[3].legend()
    ax[3].grid(True)

    # Precision
    ax[4].plot(epochs, history.get("train_precision", []), marker="o", label="Train Precision")
    ax[4].plot(epochs, history.get("val_precision", []), marker="o", label="Val Precision")
    ax[4].set_xlabel("Epoch")
    ax[4].set_ylabel("Precision")
    ax[4].set_title("Precision (@0.5)")
    ax[4].legend()
    ax[4].grid(True)

    # Weighted IoU or final summary in the last cell
    train_wiou = history.get("train_wiou", [])
    val_wiou = history.get("val_wiou", [])
    if train_wiou or val_wiou:
        ax[5].plot(epochs, train_wiou, marker="o", label="Train wIoU")
        ax[5].plot(epochs, val_wiou, marker="o", label="Val wIoU")
        ax[5].set_xlabel("Epoch")
        ax[5].set_ylabel("wIoU")
        ax[5].set_title("Weighted IoU")
        ax[5].legend()
        ax[5].grid(True)
    else:
        ax[5].axis("off")
        # add a small final-summary text if possible
        try:
            final_val_iou = history.get("val_iou", [])[-1]
            final_val_dice = history.get("val_dice", [])[-1]
            summary = f"Final Val IoU: {final_val_iou:.3f}\nFinal Val Dice: {final_val_dice:.3f}"
        except Exception:
            summary = "No final summary available"
        ax[5].text(0.5, 0.5, summary, ha="center", va="center", fontsize=12)

    fig.suptitle(f"Training metrics — {data_type}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(plots_dir, "metrics_combined.png")
    fig.savefig(out_path)
    plt.close(fig)


def get_types_ds(data_dir,input_image_reshape, foreground_class):
    train_tf = get_train_transform()

    # Define the data directories and create the datasets
    x_train_dir = os.path.join(data_dir, "train")
    y_train_dir = os.path.join(data_dir, "trainanot")

    x_val_dir = os.path.join(data_dir, "valid")
    y_val_dir = os.path.join(data_dir, "validanot")

    x_test_dir = os.path.join(data_dir, "test")
    y_test_dir = os.path.join(data_dir, "testanot")

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        input_image_reshape=input_image_reshape,
        foreground_class=foreground_class,
        augmentation=train_tf,  # Apply augmentations only to training set
    )
    valid_dataset = Dataset(
        x_val_dir,
        y_val_dir,
        input_image_reshape=input_image_reshape,
        foreground_class=foreground_class,
    )
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        input_image_reshape=input_image_reshape,
        foreground_class=foreground_class,
    )

    return train_dataset, valid_dataset, test_dataset

def get_loaders(data_dir, input_image_reshape, foreground_class, batch_size, channel_indices, raw_ds = True):

    # train_dataset, valid_dataset, test_dataset = get_types_ds(data_dir,input_image_reshape, foreground_class)
    # # Create the dataloaders using the datasets
    # logging.info(f"[{data_type}] Train size: {len(train_dataset)}")
    # logging.info(f"[{data_type}] Valid size: {len(valid_dataset)}")
    # logging.info(f"[{data_type}] Test size: {len(test_dataset)}")

    if raw_ds:
        train_tf = get_train_transform()
        train_dataset = Dataset(data_dir, split='train', augmentation=train_tf, channel_indices=channel_indices)
        valid_dataset = Dataset(data_dir, split='valid', channel_indices=channel_indices)
        test_dataset = Dataset(data_dir, split='test', channel_indices=channel_indices)

        logging.info(f"Train size: {len(train_dataset)}")
        logging.info(f"Valid size: {len(valid_dataset)}")
        logging.info(f"Test size: {len(test_dataset)}")


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader

