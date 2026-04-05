import math
import threading
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import efficientnet_b1
from torchvision.transforms.functional import pil_to_tensor

from backend.app.core.config import BASE_DIR, get_settings
from backend.app.schemas.tus_rec import (
    TusRecPredictionResponse,
    TusRecStatusResponse,
)


class TusRecBaselineRuntime:
    def __init__(
        self,
        repo_dir: str,
        checkpoint_path: str,
        device_name: str,
        input_height: int,
        input_width: int,
    ) -> None:
        self.repo_dir = self._resolve_path(repo_dir)
        self.checkpoint_path = self._resolve_path(checkpoint_path)
        self.input_height = input_height
        self.input_width = input_width
        self.device = self._resolve_device(device_name)
        self._model: torch.nn.Module | None = None
        self._lock = threading.Lock()

    def status(self) -> TusRecStatusResponse:
        return TusRecStatusResponse(
            model_name="TUS-REC baseline efficientnet_b1",
            source_repo_dir=str(self.repo_dir),
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_exists=self.checkpoint_path.exists(),
            loaded=self._model is not None,
            device=str(self.device),
            input_height=self.input_height,
            input_width=self.input_width,
            output_format="rotation_zyx_rad + translation",
        )

    def predict_from_uploads(
        self,
        frame_0_name: str,
        frame_0_bytes: bytes,
        frame_1_name: str,
        frame_1_bytes: bytes,
    ) -> TusRecPredictionResponse:
        self.load()

        frame_0 = self._decode_frame(frame_0_bytes)
        frame_1 = self._decode_frame(frame_1_bytes)
        inputs = torch.stack([frame_0, frame_1], dim=0).unsqueeze(0).to(self.device)

        if self._model is None:
            raise RuntimeError("The TUS-REC model failed to load.")

        with torch.inference_mode():
            outputs = self._model(inputs).squeeze(0).cpu()

        parameters = [float(value) for value in outputs.tolist()]
        transform_matrix = self._parameters_to_matrix(parameters)

        return TusRecPredictionResponse(
            model_name="TUS-REC baseline efficientnet_b1",
            checkpoint_path=str(self.checkpoint_path),
            device=str(self.device),
            input_height=self.input_height,
            input_width=self.input_width,
            frame_0_name=frame_0_name,
            frame_1_name=frame_1_name,
            parameter_vector_zyx_translation=parameters,
            rotation_zyx_rad=parameters[:3],
            translation=parameters[3:],
            transform_matrix=transform_matrix,
        )

    def load(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"TUS-REC checkpoint was not found at {self.checkpoint_path}."
                )

            model = self._build_model()
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self._model = model

    def _build_model(self) -> torch.nn.Module:
        model = efficientnet_b1(weights=None)
        original_stem = model.features[0][0]
        model.features[0][0] = torch.nn.Conv2d(
            in_channels=2,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=original_stem.bias is not None,
        )
        model.classifier[1] = torch.nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=6,
        )
        return model

    def _decode_frame(self, payload: bytes) -> torch.Tensor:
        if not payload:
            raise ValueError("Uploaded frame is empty.")

        image = Image.open(BytesIO(payload)).convert("L")
        image = image.resize(
            (self.input_width, self.input_height),
            resample=Image.Resampling.BILINEAR,
        )
        return pil_to_tensor(image).squeeze(0).float() / 255.0

    def _resolve_device(self, preferred: str) -> torch.device:
        if preferred.startswith("cuda") and torch.cuda.is_available():
            return torch.device(preferred)
        return torch.device("cpu")

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return BASE_DIR / path

    def _parameters_to_matrix(self, parameters: list[float]) -> list[list[float]]:
        angle_z, angle_y, angle_x, tx, ty, tz = parameters

        cos_z = math.cos(angle_z)
        sin_z = math.sin(angle_z)
        cos_y = math.cos(angle_y)
        sin_y = math.sin(angle_y)
        cos_x = math.cos(angle_x)
        sin_x = math.sin(angle_x)

        rotation = [
            [
                cos_z * cos_y,
                cos_z * sin_y * sin_x - sin_z * cos_x,
                cos_z * sin_y * cos_x + sin_z * sin_x,
            ],
            [
                sin_z * cos_y,
                sin_z * sin_y * sin_x + cos_z * cos_x,
                sin_z * sin_y * cos_x - cos_z * sin_x,
            ],
            [
                -sin_y,
                cos_y * sin_x,
                cos_y * cos_x,
            ],
        ]

        return [
            [rotation[0][0], rotation[0][1], rotation[0][2], tx],
            [rotation[1][0], rotation[1][1], rotation[1][2], ty],
            [rotation[2][0], rotation[2][1], rotation[2][2], tz],
            [0.0, 0.0, 0.0, 1.0],
        ]


@lru_cache
def get_tus_rec_runtime() -> TusRecBaselineRuntime:
    settings = get_settings()
    return TusRecBaselineRuntime(
        repo_dir=settings.tus_rec_repo_dir,
        checkpoint_path=settings.tus_rec_checkpoint_path,
        device_name=settings.tus_rec_device,
        input_height=settings.tus_rec_input_height,
        input_width=settings.tus_rec_input_width,
    )
