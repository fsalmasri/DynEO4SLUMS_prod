"""
A module defining the segmentation model using Segmentation Models PyTorch and MonAI.
"""
import os
import torch
import segmentation_models_pytorch as smp
from monai.networks.nets import (
    AttentionUnet,
    BasicUNet,
    BasicUNetPlusPlus,
    DynUNet,
    SegResNet,
    SegResNetDS,
)

import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SatelliteSegModel(torch.nn.Module):
    """
    Binary segmentation model wrapper that can switch between
    Segmentation Models PyTorch (SMP) and MONAI backbones.

    Args:
        backend (str): Either "smp" or "monai".
        model_name (str): Name of the model when backend="monai".
        arch (str): SMP architecture (e.g., "Unet", "FPN").
        encoder_name (str): SMP encoder (e.g., "resnet34", "vgg16").
        in_channels (int): Number of input channels.
        out_classes (int): Number of output classes.
        normalize (bool): Apply ImageNet normalization when channels==3.
        **kwargs: Extra args forwarded to the selected model ctor.
    """

    def __init__(
        self,
        backend="smp",
        model_name=None,
        arch=None,
        encoder_name=None,
        in_channels=3,
        out_classes=1,
        normalize=True,
        **kwargs,
    ):
        super().__init__()
        self.backend = backend.lower()
        self.normalize = normalize and in_channels == 3

        if self.normalize:
            self.register_buffer(
                "mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                persistent=False,
            )
        else:
            self.mean = None
            self.std = None

        if self.backend == "smp":
            if arch is None or encoder_name is None:
                raise ValueError("arch and encoder_name are required when backend='smp'")
            self.model = smp.create_model(
                arch,
                encoder_name=encoder_name,
                in_channels=in_channels,
                classes=out_classes,
                **kwargs,
            )
        elif self.backend == "monai":
            self.model = self._build_monai_model(
                model_name=model_name,
                in_channels=in_channels,
                out_classes=out_classes,
                **kwargs,
            )
        else:
            raise ValueError("backend must be either 'smp' or 'monai'")

    def forward(self, image):
        # Normalize image
        if self.normalize and self.mean is not None and self.std is not None:
            image = (image - self.mean) / self.std
        return self.model(image)

    def _build_monai_model(self, model_name, in_channels, out_classes, **kwargs):
        if not model_name:
            raise ValueError("model_name is required when backend='monai'")

        builders = {
            "dynunet_deep": lambda: DynUNet(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=out_classes,
                kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3]],
                strides=[[1, 1], [2, 2], [2, 2], [2, 2]],
                upsample_kernel_size=[[2, 2], [2, 2], [2, 2]],
                filters=[16, 32, 64, 128, 256],
                dropout=0.1,
                **kwargs,
            ),
            "dynunet_shallow": lambda: DynUNet(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=out_classes,
                kernel_size=[[3, 3], [3, 3], [3, 3]],
                strides=[[1, 1], [2, 2], [2, 2]],
                upsample_kernel_size=[[2, 2], [2, 2]],
                filters=[16, 32, 64, 128],
                dropout=0.05,
                **kwargs,
            ),
            "segresnet": lambda: SegResNet(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=out_classes,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                init_filters=32,
                dropout_prob=0.15,
                **kwargs,
            ),
            "segresnetds": lambda: SegResNetDS(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=out_classes,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                init_filters=32,
                dsdepth=2,
                **kwargs,
            ),
            "attention_unet": lambda: AttentionUnet(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=out_classes,
                channels=(16, 32, 64, 128),
                strides=(2, 2, 2, 2),
                dropout=0.1,
                **kwargs,
            ),
            "basic_unet": lambda: BasicUNet(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=out_classes,
                features=(16, 32, 64, 128, 256, 32),
                norm="instance",
                dropout=0.1,
                **kwargs,
            ),
            "basic_unetpp": lambda: BasicUNetPlusPlus(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=out_classes,
                features=(32, 32, 64, 128, 256, 32),
                norm="instance",
                dropout=0.15,
                **kwargs,
            ),
        }

        if model_name not in builders:
            raise ValueError(
                f"Unsupported MONAI model_name '{model_name}'. Supported: {list(builders.keys())}"
            )

        return builders[model_name]()

    def load_checkpoint(self, ckpt_path: str, device=None, strict: bool = True) -> bool:
        """
        Loads a checkpoint into the wrapped model.
        Supports:
          - raw state_dict (your current code)
          - dict with 'state_dict' or 'model_state_dict'
          - DataParallel/Distributed 'module.' prefixes

        Returns True if loaded, False otherwise.
        """
        device = device or next(self.parameters()).device

        if not ckpt_path or not os.path.isfile(ckpt_path):
            logging.warning(f"Checkpoint not found at {ckpt_path}.")
            return False

        try:
            logging.info(f"Loading checkpoint from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=device)

            # Common checkpoint formats
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt  # assume raw state_dict

            # Handle DataParallel/Distributed prefixes
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

            missing, unexpected = self.load_state_dict(state_dict, strict=strict)

            if missing:
                logging.warning(f"Missing keys when loading: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                logging.warning(
                    f"Unexpected keys when loading: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

            self.to(device)
            logging.info("Checkpoint loaded successfully.")
            return True

        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}.")
            return False

    def load_best_checkpoint(self, checkpoint_dir: str, filename: str = "best_model.pth",
                             device=None, strict: bool = True) -> bool:
        best_ckpt = os.path.join(checkpoint_dir, filename)
        return self.load_checkpoint(best_ckpt, device=device, strict=strict)