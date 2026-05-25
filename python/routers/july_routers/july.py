import os
import time
import uuid
import base64
import logging
import asyncio
import json
from datetime import datetime
from dataclasses import asdict
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, Body
from fastapi.sse import EventSourceResponse
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("JulyEngine.Routers.July")

from .bridge_interface import BridgeInterface

bridge: BridgeInterface = None


def set_bridge(b: BridgeInterface):
    global bridge
    bridge = b


router = APIRouter(prefix="/july/v1", tags=["July Custom"])


class RagBatchDeleteRequest(BaseModel):
    ids: List[str]
    collection: str = "july_memory"


class SmartSearchRequest(BaseModel):
    prompt: str
    rag_model: Optional[str] = None
    llm_model: Optional[str] = None
    top_k: int = 5
    max_split_questions: int = 3
    collection: str = "july_memory"
    filter: Optional[Dict[str, Any]] = None
    structured_response: Optional[bool] = False
    stream_response: Optional[bool] = False


class SmartSearchResponse(BaseModel):
    results: List[Dict[str, Any]]


class RagAddRequest(BaseModel):
    model: Optional[str] = None
    text: str
    collection: str = "july_memory"
    metadata: Optional[Dict[str, Any]] = None


class RagDocument(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class RagAddBatchRequest(BaseModel):
    documents: List[RagDocument]
    collection: str = "july_memory"


class RagSearchRequest(BaseModel):
    model: Optional[str] = None
    query: str
    collection: str = "july_memory"
    top_k: int = 3
    filter: Optional[Dict[str, Any]] = None


class RagAddVectorRequest(BaseModel):
    vector: List[float]
    collection: str = "july_memory"
    metadata: Optional[Dict[str, Any]] = None


class RagUpdateRequest(BaseModel):
    id: str
    vector: List[float]
    collection: str = "july_memory"
    metadata: Optional[Dict[str, Any]] = None


async def save_upload_stream(upload_file: UploadFile, dest_folder: str = "storage/temp") -> str:
    """Lê o arquivo binário em pedaços e salva no disco sem inflar a RAM."""
    os.makedirs(dest_folder, exist_ok=True)
    
    # Gera um nome de arquivo único para evitar colisão de requests paralelos
    ext = upload_file.filename.split('.')[-1] if '.' in upload_file.filename else 'bin'
    file_name = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(dest_folder, file_name)
    
    # Lendo em chunks de 1MB
    chunk_size = 1024 * 1024 
    
    with open(file_path, "wb") as buffer:
        while True:
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break
            buffer.write(chunk)
            
    return file_path


@router.post("/vision/video/describe")
async def describe_video(
    http_request: Request,
    file: UploadFile = File(...),
    interval_sec: Optional[float] = Form(2.0), # Deixa o cliente escolher a densidade!
    frames_per_grid: Optional[int] = Form(4),  # Quantos frames por lote
    model: Optional[str] = Form(None),
    strategy: Optional[str] = Form("default"), # Pode ser "default", "interaction" ou "emotion"
    description_model: Optional[str] = Form(None),
    detect_changes: Optional[bool] = Form(False),
):
    """
    Analisa os frames visuais de um vídeo e retorna uma descrição detalhada 
    das ações, ambiente e pessoas. (Não transcreve áudio).
    """
    headers = dict(http_request.headers)
    
    # Salva o vídeo via stream (protegendo a RAM)
    saved_video_path = await save_upload_stream(file)
    
    payload = {
        "video_path": saved_video_path,
        "interval_sec": interval_sec,
        "frames_per_grid": frames_per_grid,
        "model": model,
        "strategy": strategy,
        "description_model": description_model,
        "detect_changes": detect_changes
    }
    
    try:
        # Agora o nome deixa claro que vamos invocar o VLM (Olhos), e não o STT (Ouvidos)
        result = await bridge.process_video_description(payload, headers)
        
        # Serializa o dataclass VideoAggregate para dicionário
        serializable_result = asdict(result) if hasattr(result, "__dataclass_fields__") else result
        
        # Converte datetime para string para serialização JSON
        if isinstance(serializable_result, dict) and "processed_at" in serializable_result:
            if isinstance(serializable_result["processed_at"], datetime):
                serializable_result["processed_at"] = serializable_result["processed_at"].isoformat()
        
        return JSONResponse(content={"visual_narrative": serializable_result})
    finally:
        if os.path.exists(saved_video_path):
            os.remove(saved_video_path)


@router.post("/vision/face/sync")
async def sync_faces_batch(http_request: Request, payload: Dict[str, Any]):
    """Sincroniza rostos de múltiplas imagens em lote (Detection + Embedding + RAG Matching)."""
    headers = dict(http_request.headers)
    results = await bridge.process_face_sync_batch(payload, headers)
    return JSONResponse(content={"results": results})


@router.post("/vision/faces/extract")
async def extract_faces(
    http_request: Request,
    files: List[UploadFile] = File(...), # Recebe N imagens num único POST
    model: Optional[str] = Form(None)
):
    headers = dict(http_request.headers)
    
    images_b64 = []
    for file in files:
        bytes_data = await file.read()
        b64_str = base64.b64encode(bytes_data).decode('utf-8')
        images_b64.append(b64_str)
        
    payload = {
        "images": images_b64,
        "model": model
    }
    
    description = await bridge.process_face_extraction(payload, headers)
    
    return JSONResponse(content={"faces_description": description})


@router.post("/vision/face/embedding")
async def get_face_embedding(
    http_request: Request,
    payload: dict
):
    """
    Recebe um crop de rosto em base64 e retorna o embedding ArcFace via DeepFace.
    Usado pelo july_photos para matching facial delegado à Engine.
    """
    image_b64 = payload.get("image")
    if not image_b64:
        return JSONResponse(status_code=400, content={"error": "Campo 'image' é obrigatório."})
    
    try:
        import io
        import numpy as np
        from PIL import Image as PILImage
        from deepface import DeepFace
        
        img_bytes = base64.b64decode(image_b64)
        img_pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        
        rep = DeepFace.represent(
            img_path=img_np,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False,
            align=True
        )
        embedding = rep[0]["embedding"]
        return JSONResponse(content={"embedding": embedding})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/vision/images/describe")
async def describe_images(
    http_request: Request,
    files: List[UploadFile] = File(...), # Aceita array de imagens
    prompt: str = Form("Describe these images in detail."), # Prompt customizável
    model: Optional[str] = Form(None)
):
    headers = dict(http_request.headers)
    
    images_b64 = []
    for file in files:
        bytes_data = await file.read()
        b64_str = base64.b64encode(bytes_data).decode('utf-8')
        images_b64.append(b64_str)
        
    payload = {
        "images": images_b64,
        "prompt": prompt,
        "model": model
    }
    
    # Chama o Bridge (que fará o repasse para o orquestrador VLM)
    descriptions = await bridge.process_image_description(payload, headers)
    
    # Devolvemos uma lista com a descrição de cada imagem, ou um consolidado
    return JSONResponse(content={"descriptions": descriptions})


@router.post("/vision/images/remove-background")
async def remove_background(
    http_request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None)
):
    headers = dict(http_request.headers)
    
    bytes_data = await file.read()
    image_b64 = base64.b64encode(bytes_data).decode('utf-8')
        
    payload = {
        "image": image_b64,
        "model": model
    }
    
    result = await bridge.process_image_remove_background(payload, headers)
    
    return JSONResponse(content={"image": result})


@router.post("/rag")
async def add_rag(
    http_request: Request,
    payload: RagAddRequest
):
    """Adiciona um texto/descrição ao banco vetorial da Engine com metadados."""
    headers = dict(http_request.headers)
    try:
        result = await bridge.process_rag_add(payload.model_dump(), headers)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/rag/batch")
async def add_rag_batch(
    http_request: Request,
    payload: RagAddBatchRequest
):
    """Insere múltiplos documentos no RAG em uma única chamada com metadados."""
    headers = dict(http_request.headers)
    try:
        result = await bridge.process_rag_batch_add(payload.model_dump(), headers)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/rag/search")
async def search_rag(
    http_request: Request,
    payload: RagSearchRequest
):
    """Busca avançada de contexto via Texto com suporte a filtros de metadados."""
    headers = dict(http_request.headers)
    try:
        result = await bridge.process_rag_search(payload.model_dump(), headers)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/rag/vector")
async def add_rag_vector(
    http_request: Request,
    payload: RagAddVectorRequest
):
    """Adiciona um vetor matemático bruto com metadados."""
    headers = dict(http_request.headers)
    try:
        result = await bridge.process_rag_vector_add(payload.model_dump(), headers)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.put("/rag/update")
async def update_rag_embedding(
    http_request: Request,
    payload: RagUpdateRequest
):
    """Substitui um Vetor Específico e seus metadados."""
    headers = dict(http_request.headers)
    try:
        result = await bridge.process_rag_update(payload.model_dump(), headers)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.delete("/rag/{item_id}")
async def delete_rag_single(
    item_id: str,
    request: Request,
    collection: str = "july_memory"
):
    """Deleta um único registro do RAG."""
    headers = dict(request.headers)
    try:
        payload = {"ids": [item_id], "collection": collection}
        result = await bridge.process_rag_delete(payload, headers)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in delete_rag_single: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/rag/batch-delete")
async def delete_rag_batch(
    request_data: RagBatchDeleteRequest,
    request: Request
):
    """Deleta múltiplos registros do RAG via POST (Batch standard)."""
    headers = dict(request.headers)
    try:
        payload = request_data.model_dump()
        result = await bridge.process_rag_delete(payload, headers)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in delete_rag_batch: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/rag/list")
async def list_rag_metadata(
    http_request: Request,
    collection: str = "july_memory"
):
    """Lista metadados (IDs, path, etc) de uma coleção sem carregar os vetores."""
    headers = dict(http_request.headers)
    
    try:
        payload = {"collection": collection}
        result = await bridge.process_rag_list(payload, headers)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in list_rag_metadata: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/rag/smart-search", response_model=None)
async def smart_search_rag(
    request: Request,
    payload: SmartSearchRequest
) -> Union[SmartSearchResponse, EventSourceResponse, JSONResponse]:
    """
    Realiza uma busca 'inteligente' no RAG (delegado ao Bridge -> Orchestrator -> Memory).
    """
    headers = dict(request.headers)
    
    try:
        # Resolvemos via Bridge seguindo a arquitetura do sistema
        result = await bridge.process_rag_smart_search(payload.model_dump(), headers)
        
        if payload.structured_response:

            if payload.stream_response:
                async def sse_formatter(generator):
                    try:
                        async for chunk_dict in generator:
                            # Pega o dicionário e transforma em string JSON com o prefixo 'data: '
                            chunk_str = json.dumps(chunk_dict)
                            yield f"data: {chunk_str}\n\n"
                    finally:
                        # O padrão OpenAI exige que o stream termine com a string [DONE]
                        yield "data: [DONE]\n\n"
                        
                # Retorna o StreamingResponse empacotando o nosso formatador
                return StreamingResponse(
                    sse_formatter(result.get("results")), 
                    media_type="text/event-stream"
                )

            return result.get("results")

        return SmartSearchResponse(results=result.get("results", []))
    except Exception as e:
        logger.error(f"Error in smart_search_rag: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/utils/extract-pdf")
async def extract_pdf_route(file: UploadFile = File(...)):
    """Extrai texto e ilustração de cada página de um PDF e transmite via SSE/NDJSON."""
    async def sse_generator():
        try:
            pdf_bytes = await file.read()
            events = await bridge.process_pdf_extract(pdf_bytes)
            for event in events:
                yield f"{json.dumps(event)}\n"
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}", exc_info=True)
            yield f"{json.dumps({'type': 'error', 'message': str(e)})}\n"

    return StreamingResponse(sse_generator(), media_type="application/x-ndjson")
