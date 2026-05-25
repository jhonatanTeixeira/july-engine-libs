from fastapi import APIRouter
import logging
from pydantic import BaseModel
from typing import Optional
from .bridge_interface import BridgeInterface

logger = logging.getLogger("JulyEngine.Routers.Calculator")

bridge: BridgeInterface = None


def set_bridge(b: BridgeInterface):
    global bridge
    bridge = b


router = APIRouter(prefix="/system", tags=["System"])


class ResourceCheckRequest(BaseModel):
    model_path: Optional[str] = "model"
    model_id: Optional[str] = None
    filename: Optional[str] = None
    context_window: str | int = "4k"
    gpu_layers: Optional[int] = -1
    kv_cache_quantization: Optional[str] = "FP16"
    mmproj_path: Optional[str] = None
    mmproj_id: Optional[str] = None
    mmproj_filename: Optional[str] = None
    flash_attn: Optional[bool] = True
    n_seq_max: Optional[int] = 1
    offload_kqv: Optional[bool] = True
    logits_all: Optional[bool] = False
    vision_on_cpu: Optional[bool] = False


@router.post("/check-resources")
async def check_resources(req: ResourceCheckRequest):
    """
    Unified entry point for VRAM/RAM estimation.
    Supports local paths, HF cache, or remote scan.
    """
    return await bridge.process_resource_check(req.model_dump())
