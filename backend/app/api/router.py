from fastapi import APIRouter

from backend.app.api.routes.assistant import router as assistant_router
from backend.app.api.routes.system import router as system_router
from backend.app.api.routes.tus_rec import router as tus_rec_router

api_router = APIRouter()
api_router.include_router(system_router)
api_router.include_router(assistant_router)
api_router.include_router(tus_rec_router)
