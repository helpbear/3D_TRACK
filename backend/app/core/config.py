from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "3D Track Surgical AI Assistant"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    backend_cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ]
    )
    default_video_source: str = "demo://offline"
    endoarss_repo_dir: str = "EndoARSS"
    endoarss_checkpoint_path: str = (
        "EndoARSS/checkpoint_MTLESD/mtl_MOLA_DB_MTL_best_cls_w=1_seg_w=1_dinonet.pt"
    )
    endoarss_pretrain_dir: str = "EndoARSS/pretrain_weights"
    endoarss_device: str = "cuda"
    endoarss_input_height: int = 244
    endoarss_input_width: int = 244
    tus_rec_repo_dir: str = "tus-rec-challenge_baseline"
    tus_rec_checkpoint_path: str = (
        "tus-rec-challenge_baseline/results/"
        "seq_len2__efficientnet_b1__lr0.0001__pred_type_parameter__label_type_point/"
        "saved_model/best_validation_dist_model"
    )
    tus_rec_device: str = "cuda"
    tus_rec_input_height: int = 480
    tus_rec_input_width: int = 640

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
