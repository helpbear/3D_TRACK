from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


class TaskCard(BaseModel):
    id: str
    name: str
    summary: str
    category: Literal["classification", "segmentation", "workflow", "safety"]
    outputs: list[str] = Field(default_factory=list)
    status: Literal["planned", "prototype", "ready"]


class TaskCatalogResponse(BaseModel):
    items: list[TaskCard]


class AnalysisRequest(BaseModel):
    task_id: str
    procedure: str = Field(default="endoscopic-submucosal-dissection")
    note: str = Field(default="", max_length=2000)
    session_id: str | None = None
    frame_hint: str | None = None


class ActivityScore(BaseModel):
    label: Literal["marking", "injection", "dissection", "idle"]
    score: float = Field(ge=0.0, le=1.0)


class OverlayTarget(BaseModel):
    id: str
    label: str
    kind: Literal["region", "instrument", "anatomy"]
    priority: Literal["primary", "supporting"]


class RiskFlag(BaseModel):
    code: str
    severity: Literal["info", "warning", "critical"]
    message: str


class AnalysisResponse(BaseModel):
    task_id: str
    procedure: str
    frame_name: str | None = None
    scene_summary: str
    primary_activity: Literal["marking", "injection", "dissection", "idle"]
    activity_scores: list[ActivityScore]
    visibility_score: float = Field(ge=0.0, le=1.0)
    risk_flags: list[RiskFlag]
    recommended_overlays: list[OverlayTarget]
    instrument_hints: list[str]
    safe_to_continue: bool
    confidence: float = Field(ge=0.0, le=1.0)
    findings: list[str]
    next_action: str
