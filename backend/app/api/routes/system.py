from fastapi import APIRouter

from backend.app.schemas.assistant import HealthResponse, TaskCatalogResponse
from backend.app.services.surgical_assistant import list_tasks

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", service="backend")


@router.get("/tasks", response_model=TaskCatalogResponse)
def get_tasks() -> TaskCatalogResponse:
    return TaskCatalogResponse(items=list_tasks())

