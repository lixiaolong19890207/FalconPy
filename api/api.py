from pathlib import Path
from typing import Dict

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from api.routers import vr

api_router = APIRouter()
api_router.include_router(vr.router, prefix="/vr", tags=["vr"])


CUR_PATH = Path(__file__).resolve().parent
DATA_PATH = CUR_PATH / 'templates/index.html'
with open(DATA_PATH, 'r') as f:
    HTML = f.read()


@api_router.get("/")
async def index():
    return HTMLResponse(HTML)

