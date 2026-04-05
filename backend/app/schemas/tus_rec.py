from pydantic import BaseModel, Field


class TusRecStatusResponse(BaseModel):
    model_name: str
    source_repo_dir: str
    checkpoint_path: str
    checkpoint_exists: bool
    loaded: bool
    device: str
    input_height: int
    input_width: int
    output_format: str


class TusRecPredictionResponse(BaseModel):
    model_name: str
    checkpoint_path: str
    device: str
    input_height: int
    input_width: int
    frame_0_name: str
    frame_1_name: str
    parameter_vector_zyx_translation: list[float] = Field(min_length=6, max_length=6)
    rotation_zyx_rad: list[float] = Field(min_length=3, max_length=3)
    translation: list[float] = Field(min_length=3, max_length=3)
    transform_matrix: list[list[float]]

