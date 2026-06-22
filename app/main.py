import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config.logger import setup_logger
from app.routers.api_routes.tts_routes import router as tts_router
from app.routers.web_routes.web import router as web_router
from app.services.hkl_vits_inference import HKLVITSInference
warnings.filterwarnings("ignore")

logger = setup_logger(name=__name__)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Start Loading HKL ViTs model...")
    hkl_inference = HKLVITSInference.load()
    app.state.hkl_inference = hkl_inference
    logger.info(f"Successfully loaded HKL ViTs model...")
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.include_router(web_router)
app.include_router(tts_router, prefix="/api")
