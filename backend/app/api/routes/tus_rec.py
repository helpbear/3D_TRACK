from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import UnidentifiedImageError

from backend.app.schemas.tus_rec import (
    TusRecPredictionResponse,
    TusRecStatusResponse,
)
from backend.app.services.tus_rec_baseline import (
    TusRecBaselineRuntime,
    get_tus_rec_runtime,
)

router = APIRouter(prefix="/tus-rec", tags=["tus-rec"])


@router.get("/status", response_model=TusRecStatusResponse)
def get_status(
    runtime: TusRecBaselineRuntime = Depends(get_tus_rec_runtime),
) -> TusRecStatusResponse:
    return runtime.status()


@router.post("/predict", response_model=TusRecPredictionResponse)
async def predict(
    frame_0: UploadFile = File(...),
    frame_1: UploadFile = File(...),
    runtime: TusRecBaselineRuntime = Depends(get_tus_rec_runtime),
) -> TusRecPredictionResponse:
    try:
        return runtime.predict_from_uploads(
            frame_0_name=frame_0.filename or "frame_0",
            frame_0_bytes=await frame_0.read(),
            frame_1_name=frame_1.filename or "frame_1",
            frame_1_bytes=await frame_1.read(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except (UnidentifiedImageError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

