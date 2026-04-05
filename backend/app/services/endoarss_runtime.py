import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from backend.app.core.config import BASE_DIR, get_settings


class EndoArssRuntime:
    def __init__(
        self,
        repo_dir: str,
        checkpoint_path: str,
        pretrain_dir: str,
        device_name: str,
        input_height: int,
        input_width: int,
    ) -> None:
        self.repo_dir = self._resolve_path(repo_dir)
        self.checkpoint_path = self._resolve_path(checkpoint_path)
        self.pretrain_dir = self._resolve_path(pretrain_dir)
        self.device = self._resolve_device(device_name)
        self.input_height = input_height
        self.input_width = input_width
        self.activity_labels = ("marking", "injection", "dissection")
        self._model: nn.Module | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            self._add_repo_to_path()
            self._validate_paths()

            from aspp import DeepLabClassHead, DeepLabHead
            from LibMTL.architecture.MOLA import MOLA
            from LibMTL.model.dinov2 import Dinov2WithLoRA

            decoders = nn.ModuleDict(
                {
                    "segmentation": DeepLabHead(256, 7),
                    "classification": DeepLabClassHead(256, 3),
                }
            )
            encoder_class = lambda: Dinov2WithLoRA(
                backbone_size="base",
                r=4,
                image_shape=(224, 224),
                lora_type="dvlora",
                pretrained_path=str(self.pretrain_dir),
                residual_block_indexes=[],
                include_cls_token=True,
                use_cls_token=False,
                use_bn=False,
            )
            model = MOLA(
                task_name=["segmentation", "classification"],
                encoder_class=encoder_class,
                decoders=decoders,
                rep_grad=False,
                multi_input=False,
                device=self.device,
            )
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self._model = model

    def predict(self, image_rgb: Image.Image) -> dict:
        self.load()
        if self._model is None:
            raise RuntimeError("EndoARSS failed to initialize.")

        inputs = self._preprocess_image(image_rgb).to(self.device)
        with torch.inference_mode():
            outputs = self._model(inputs)

        class_probs = torch.softmax(outputs["classification"], dim=1)[0].cpu()
        seg_probs = torch.softmax(
            F.interpolate(
                outputs["segmentation"],
                size=(self.input_height, self.input_width),
                mode="bilinear",
                align_corners=True,
            ),
            dim=1,
        )[0].cpu()
        seg_mask = seg_probs.argmax(0)
        seg_confidence = float(seg_probs.max(0).values.mean().item())

        region_coverage: dict[int, float] = {}
        unique_ids, counts = torch.unique(seg_mask, return_counts=True)
        total_pixels = float(seg_mask.numel())
        for class_id, count in zip(unique_ids.tolist(), counts.tolist()):
            region_coverage[int(class_id)] = float(count) / total_pixels

        return {
            "classification": {
                label: float(score)
                for label, score in zip(self.activity_labels, class_probs.tolist())
            },
            "segmentation_confidence": seg_confidence,
            "region_coverage": region_coverage,
            "dominant_region_ids": sorted(
                [class_id for class_id in region_coverage.keys() if class_id != 0],
                key=lambda item: region_coverage[item],
                reverse=True,
            ),
        }

    def _preprocess_image(self, image_rgb: Image.Image) -> torch.Tensor:
        resized = image_rgb.resize(
            (self.input_width, self.input_height),
            resample=Image.Resampling.BILINEAR,
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    def _add_repo_to_path(self) -> None:
        repo_path = str(self.repo_dir)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    def _resolve_device(self, preferred: str) -> torch.device:
        if preferred.startswith("cuda") and torch.cuda.is_available():
            return torch.device(preferred)
        return torch.device("cpu")

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return BASE_DIR / path

    def _validate_paths(self) -> None:
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"EndoARSS repo was not found at {self.repo_dir}.")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"EndoARSS checkpoint was not found at {self.checkpoint_path}."
            )
        if not self.pretrain_dir.exists():
            raise FileNotFoundError(
                f"EndoARSS pretrain directory was not found at {self.pretrain_dir}."
            )


@lru_cache
def get_endoarss_runtime() -> EndoArssRuntime:
    settings = get_settings()
    return EndoArssRuntime(
        repo_dir=settings.endoarss_repo_dir,
        checkpoint_path=settings.endoarss_checkpoint_path,
        pretrain_dir=settings.endoarss_pretrain_dir,
        device_name=settings.endoarss_device,
        input_height=settings.endoarss_input_height,
        input_width=settings.endoarss_input_width,
    )
