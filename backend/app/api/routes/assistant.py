from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import UnidentifiedImageError

from backend.app.schemas.assistant import AnalysisRequest, AnalysisResponse
from backend.app.services.surgical_assistant import analyze_frame, analyze_request

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(payload: AnalysisRequest) -> AnalysisResponse:
    return analyze_request(payload)


@router.post("/analyze-frame", response_model=AnalysisResponse)
async def analyze_uploaded_frame(
    task_id: str = Form(...),
    procedure: str = Form(default="endoscopic-submucosal-dissection"),
    note: str = Form(default=""),
    frame: UploadFile = File(...),
) -> AnalysisResponse:
    try:
        return analyze_frame(
            task_id=task_id,
            procedure=procedure,
            note=note,
            frame_name=frame.filename or "frame",
            frame_bytes=await frame.read(),
        )
    except (UnidentifiedImageError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
